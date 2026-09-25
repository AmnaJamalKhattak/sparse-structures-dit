"""The main run end to end on real diffusers blocks at toy width: every condition, every
record, resume, pooling. The windows are placed by the same rule as the real run, on a
12-block stack whose planted registers stand in for a measured natural interval."""
import json

import pandas as pd
import pytest
import torch

from ditsinks import SweepConfig, q16_main as Q
from ditsinks.causal_engine import CausalTracer, GenerationDriver, run_traced_generation
from ditsinks.discovery import ChannelChoice, DiscoveryArtifact, LayerRanges
from ditsinks.questions import QuestionContext
from ditsinks.synthetic import build_tiny, planted_direction


def _context(model, tmp_path, *, n_dual=4, n_single=8, steps=4):
    cfg = SweepConfig(model=model, prompts=["a"], seeds=[0], height=64, width=64,
                      num_inference_steps=steps, capture_steps=[0], dtype="float32",
                      output_dir=str(tmp_path), save_images=False, highnorm_ratio=1.2)
    driver = GenerationDriver(cfg)
    family = cfg.spec.family
    driver.bundle = build_tiny(family, steps=steps, grid=4, n_dual=n_dual,
                               n_single=n_single)
    driver.transformer = driver.bundle.transformer
    driver.refs = driver.adapter.layers(driver.transformer)
    driver.grid, driver.n_img = driver.bundle.grid, driver.bundle.n_img
    driver.planted = planted_direction(driver.bundle.d_model)
    # v* is FITTED, as discovery fits it on a real checkpoint: the mean direction of the
    # loudest tokens at the selection block. The planted input direction is not it -- the
    # patch embedding maps it elsewhere.
    probe = CausalTracer(driver.adapter, driver.transformer,
                         direction=torch.ones(driver.bundle.d_model), layers=[5], steps=[0],
                         full_state_layers=[5], grid=driver.grid, cfg=cfg)
    fitted, _ = run_traced_generation(driver, probe, prompt_id=0, prompt="a", seed=0)
    states = fitted.at(0, 5).states
    loud = torch.argsort(states.norm(dim=-1), descending=True)[:3 if family == "flux1" else 1]
    vstar = torch.nn.functional.normalize(states[loud].mean(0), dim=0)
    artifact = DiscoveryArtifact(
        schema_version=1, checkpoint=model, repo_id="synthetic", vstar_file="v.pt",
        fitting_population=[], explained_variance=0.99, axis_convention="",
        sign_convention="", dominant_register_channel=ChannelChoice(3, 3, True, 1.0),
        late_growing_competitor=ChannelChoice(5, None, None, 1.0),
        massive_unspecific_control=ChannelChoice(7, None, None, 1.0),
        layer_ranges=LayerRanges((3, 4), (5, 7), (8, 11)), denoising_steps=[0],
        discovery_prompts=["z"], discovery_seeds=[9], confirmation_prompts=["a"],
        confirmation_seeds=[0], code_config_fingerprint="t", config_sha256="t")
    return QuestionContext.from_artifact(cfg, artifact, vstar, driver=driver, progress=False)


def _protocol(ctx, settings):
    n = int(ctx.driver.n_layers)
    windows, notes = Q.plan_depth_windows(n_layers=n, formation=4, natural_end=7, decline=6,
                                          per_side=1, first_block=1, gap=1)
    return Q.DepthProtocol(
        checkpoint=str(ctx.cfg.model), n_layers=n, seam=Q.seam_of(ctx), formation=4,
        natural_end=7, decline=6, selection_block=5, windows=windows, primary_early="E1",
        primary_late="L1", plateau=(5, 5), window_length=2, settings=settings.row(),
        notes=notes)


@pytest.fixture(scope="module", params=["tiny-flux1", "tiny-pixart"])
def unit(request, tmp_path_factory):
    tmp = tmp_path_factory.mktemp(request.param)
    ctx = _context(request.param, tmp)
    settings = Q.MainSettings(checkpoint=request.param, size=64, steps=4, highnorm_ratio=1.2,
                              min_carriers=1, windows_per_side=1, sink_threshold=1.0,
                              rule_max_share_of_image=1.0)
    protocol = _protocol(ctx, settings)
    catalog = Q.condition_catalog(protocol, include_check=True)
    result = Q.run_unit(ctx, protocol, prompt_id=0, prompt="a", seed=0, root=tmp,
                        conditions=catalog, log=lambda *a, **k: None)
    return dict(ctx=ctx, protocol=protocol, catalog=catalog, result=result, root=tmp)


def test_the_windows_fit_the_toy_stack_without_touching_the_removal_hooks(unit):
    protocol = unit["protocol"]
    assert protocol.windows["E1"] == (1, 2) and protocol.windows["L1"] == (9, 10)
    assert protocol.windows["removal"] == (3, 7)
    hooks = set(range(3, 9))                       # the interval and its terminal cleanup
    for name in ("E1", "L1"):
        a, b = protocol.windows[name]
        assert not set(range(a, b + 1)) & hooks


def test_every_condition_ran_and_left_its_records(unit):
    directory = Q.unit_directory(unit["root"], 0, 0)
    payload = json.loads((directory / "unit.json").read_text())
    assert payload["complete"] and unit["result"]["status"] == "done"
    edits = pd.read_csv(directory / "edits.csv")
    expected = {c.key for c in unit["catalog"]} - {"reference"}
    assert set(edits["condition"]) == expected
    # Every condition that edits did edit; the hooks-only check fired and changed nothing.
    assert (edits.set_index("condition")["edit_calls"] > 0).all()
    # A hook that fires but writes nothing would read as "no effect": every inducing
    # condition must have WRITTEN the state, and the matched targets must exist.
    writes = edits.set_index("condition")["n_writes"]
    inducing = [c.key for c in unit["catalog"] if c.window is not None
                and c.operator != "hooks_only"]
    assert (writes[inducing] > 0).all(), writes[inducing]
    assert len(pd.read_csv(directory / "register_match.csv")) > 0
    lifecycle = pd.read_csv(directory / "lifecycle.csv")
    assert {"reference"} | expected <= set(lifecycle["condition"])
    attention = pd.read_csv(directory / "attention.csv")
    assert "affected_clean_register_elsewhere" in attention.columns
    for name in ("boundaries", "sensitivity", "rule_audit", "register_match", "images"):
        assert (directory / f"{name}.csv").exists(), name


def test_the_induced_state_was_written_and_the_removed_one_is_gone(unit):
    directory = Q.unit_directory(unit["root"], 0, 0)
    induction = pd.read_csv(directory / "induction.csv")
    written = induction[induction["condition"] == "induce_E1"]
    assert len(written) and (written["writes"] > 0).all()
    assert (written["established"]).all(), "the matched state is written exactly"
    lifecycle = pd.read_csv(directory / "lifecycle.csv")
    step = unit["protocol"].readout_step

    def projection(condition, layer):
        row = lifecycle[(lifecycle["condition"] == condition) & (lifecycle["step"] == step)
                        & (lifecycle["layer"] == layer)]
        return float(row["carrier_projection"].iloc[0])

    # Inside the removal interval the carriers lose their v* projection.
    assert abs(projection("remove", 5)) < 0.5 * abs(projection("reference", 5))
    # The hooks-only check is bit-identical to the unmodified run.
    assert projection("hooks_only", 5) == pytest.approx(projection("reference", 5), abs=1e-6)
    removal = pd.read_csv(directory / "removal.csv")
    assert set(removal["condition"]) == {"remove", "remove_induce_E1", "remove_induce_L1"}


def test_a_complete_unit_is_skipped_and_the_units_pool(unit):
    again = Q.run_unit(unit["ctx"], unit["protocol"], prompt_id=0, prompt="a", seed=0,
                       root=unit["root"], log=lambda *a, **k: None)
    assert again["status"] == "skipped"
    pooled = Q.collect_units(unit["root"])
    assert len(pooled["units"]) == 1
    assert {"lifecycle", "attention", "removal", "induction", "images", "edits",
            "boundaries", "sensitivity"} <= set(pooled)
    gains = Q.writer_gain_table(pooled["lifecycle"], unit["protocol"])
    assert set(gains["condition"]) >= {"reference", "induce_E1"}
    ref = gains[gains["condition"] == "reference"]["writer_gain_vs_reference"]
    assert ref.dropna().eq(1.0).all()


def test_the_effect_table_uses_prompt_clustered_intervals():
    rows = []
    for prompt in range(4):
        for seed in (0, 42):
            for condition, base in (("induce_E1", 0.30), ("control_random_direction", 0.10)):
                rows.append(dict(checkpoint="flux1-dev", prompt_id=prompt, seed=seed,
                                 condition=condition, label=condition, group="depth",
                                 window="E1", lpips=base + 0.01 * prompt))
    table = Q.effect_table(pd.DataFrame(rows), metrics=("lpips",), iterations=200)
    assert set(table["condition"]) == {"induce_E1", "control_random_direction"}
    first = table[table["condition"] == "induce_E1"].iloc[0]
    assert first["n_units"] == 8 and first["n_prompts"] == 4
    assert first["ci_low"] <= first["mean"] <= first["ci_high"]
    contrast = Q.paired_contrast(pd.DataFrame(rows), treatment="induce_E1",
                                 reference="control_random_direction", iterations=200)
    assert contrast.value == pytest.approx(0.20) and contrast.excludes_zero


# ------------------------------------------------------------ guidance diagnostic
def _settings(model):
    return Q.MainSettings(checkpoint=model, size=64, steps=4, highnorm_ratio=1.2,
                          min_carriers=1, windows_per_side=1, sink_threshold=1.0,
                          rule_max_share_of_image=1.0)


def test_both_branches_edit_the_unconditional_row_and_record_it_once(tmp_path):
    from ditsinks.adapters import InterventionPoint

    ctx = _context("tiny-pixart", tmp_path)            # a CFG-style batch of two rows
    settings = _settings("tiny-pixart")
    protocol = _protocol(ctx, settings)
    tracer = Q._tracer(ctx, layers=range(protocol.n_layers), steps=protocol.record_steps,
                       attention_steps=protocol.attention_steps)
    clean, _ = run_traced_generation(
        ctx.driver, tracer, prompt_id=0, prompt="a", seed=0, condition="reference",
        probes=[(InterventionPoint.BLOCK_INPUT, [protocol.selection_block],
                 [protocol.readout_step])])
    state, _ = Q._prepare_unit(ctx, protocol, clean, settings, 0, 0)
    condition = next(c for c in Q.condition_catalog(protocol) if c.key == "induce_E1")
    block, step = protocol.windows["E1"][1], protocol.readout_step

    def unconditional_row(all_rows=None):
        log, sites = [], []
        plans = ([] if all_rows is None else
                 Q._plans(condition, protocol, state, ctx, log, sites, all_rows=all_rows))
        probe = Q._tracer(ctx, layers=[block], steps=[step], attention_steps=(), batch_row=0)
        trace, _ = run_traced_generation(ctx.driver, probe, prompt_id=0, prompt="a", seed=0,
                                         condition="probe", plans=plans,
                                         targets=state.frozen_targets if plans else None)
        return trace.at(step, block).norm.clone(), len(log)

    untouched, _ = unconditional_row()
    conditional_only, n_conditional = unconditional_row(all_rows=False)
    both, n_both = unconditional_row(all_rows=True)
    assert torch.equal(conditional_only, untouched), "the default leaves the other row alone"
    assert not torch.equal(both, untouched), "'both' edits the unconditional row too"
    assert n_conditional == n_both > 0, "records describe the conditional row only"


def test_guidance_override_sets_and_restores_the_scale():
    from types import SimpleNamespace

    cfg = SimpleNamespace(guidance_scale=4.5)
    ctx = SimpleNamespace(cfg=cfg, driver=SimpleNamespace(cfg=cfg))
    with Q.guidance_override(ctx, 1.0):
        assert cfg.guidance_scale == 1.0
    assert cfg.guidance_scale == 4.5
    with pytest.raises(RuntimeError):
        with Q.guidance_override(ctx, 2.0):
            raise RuntimeError("interrupted")
    assert cfg.guidance_scale == 4.5
    with Q.guidance_override(ctx, None):
        assert cfg.guidance_scale == 4.5


def test_a_diagnostic_unit_records_its_branches_and_guidance(tmp_path):
    ctx = _context("tiny-pixart", tmp_path)
    protocol = _protocol(ctx, _settings("tiny-pixart"))
    catalog = [c for c in Q.condition_catalog(protocol) if c.key in ("reference", "induce_E1")]
    before = ctx.cfg.guidance_scale
    Q.run_unit(ctx, protocol, prompt_id=0, prompt="a", seed=0, root=tmp_path / "both",
               conditions=catalog, edit_branches="both", guidance=1.0,
               log=lambda *a, **k: None)
    assert ctx.cfg.guidance_scale == before
    directory = Q.unit_directory(tmp_path / "both", 0, 0)
    payload = json.loads((directory / "unit.json").read_text())
    assert payload["edit_branches"] == "both" and payload["guidance_scale"] == 1.0
    # A both-passes unit checks where the unconditional pass holds its register.
    overlap = pd.read_csv(directory / "branch_overlap.csv")
    assert set(overlap["step"]) <= set(protocol.edit_steps) and len(overlap)
    assert "conditional_register_in_unconditional_pass" in payload
    # Complete for its own policy only.
    assert Q.unit_is_complete(tmp_path / "both", 0, 0, protocol, edit_branches="both")
    assert not Q.unit_is_complete(tmp_path / "both", 0, 0, protocol,
                                  edit_branches="conditional")
    images = pd.read_csv(directory / "images.csv")
    assert set(images["edit_branches"]) == {"both"} and set(images["guidance_scale"]) == {1.0}
    with pytest.raises(ValueError):
        Q.run_unit(ctx, protocol, prompt_id=1, prompt="a", seed=0, root=tmp_path / "bad",
                   conditions=catalog, edit_branches="unconditional",
                   log=lambda *a, **k: None)


@pytest.mark.parametrize("model, single_pass", [("tiny-pixart", False), ("tiny-flux1", True)])
def test_the_register_overlap_between_the_two_passes(tmp_path, model, single_pass):
    ctx = _context(model, tmp_path)
    protocol = _protocol(ctx, _settings(model))
    rows = Q.branch_register_overlap(ctx, protocol, prompt_id=0, prompt="a", seed=0)
    assert rows and {r["step"] for r in rows} <= set(protocol.edit_steps)
    assert all(r["single_pass"] is single_pass for r in rows)
    if single_pass:
        assert all(r["jaccard"] == 1.0 or r["n_conditional"] == 0 for r in rows)


def test_the_register_profile_reads_the_unit_records(unit):
    pooled = Q.collect_units(unit["root"])
    protocol = unit["protocol"]
    profile = Q.register_profile(pooled, {protocol.checkpoint: protocol})
    row = profile.iloc[0]
    assert row["register_tokens"] >= 1 and row["register_norm_over_median"] > 0
    assert row["passes_per_step"] == (2 if "pixart" in protocol.checkpoint
                                      and (row["guidance_scale"] or 0) > 1 else 1)
    late = Q.late_state_table(pooled, {protocol.checkpoint: protocol})
    assert {"reference", "induce_E1", "control_ordinary_positions"} <= set(late["condition"])
    assert set(late[late["condition"] == "control_ordinary_positions"]["positions"]) == {"other"}


def test_policy_table_pairs_the_units_every_policy_ran():
    rows = lambda policy_shift, units: pd.DataFrame(
        [dict(prompt_id=p, seed=s, condition=c, lpips=base + policy_shift)
         for p, s in units for c, base in (("induce_E1", 0.5), ("remove", 0.9))])
    table = Q.policy_table({"conditional": rows(0.0, [(0, 0), (1, 0), (2, 0)]),
                            "both": rows(-0.3, [(0, 0), (1, 0)])})
    assert list(table.columns) == ["condition", "n_pairs", "conditional", "both"]
    assert (table["n_pairs"] == 2).all()
    e1 = table[table["condition"] == "induce_E1"].iloc[0]
    assert e1["conditional"] == pytest.approx(0.5) and e1["both"] == pytest.approx(0.2)


def test_run_root_keeps_each_pass_policy_in_its_own_folder(tmp_path):
    assert Q.run_root(tmp_path, "conditional") == tmp_path
    assert Q.run_root(tmp_path, "both") == tmp_path / "both_passes"
    with pytest.raises(ValueError):
        Q.run_root(tmp_path, "unconditional")


def test_a_unit_written_before_the_policy_was_recorded_ran_the_conditional_pass(unit):
    directory = Q.unit_directory(unit["root"], 0, 0)
    payload = json.loads((directory / "unit.json").read_text())
    legacy = directory.parent.parent / "legacy"
    target = Q.unit_directory(legacy, 0, 0)
    target.mkdir(parents=True)
    payload.pop("edit_branches", None)
    (target / "unit.json").write_text(json.dumps(payload))
    protocol = unit["protocol"]
    assert Q.unit_is_complete(legacy, 0, 0, protocol)
    assert Q.unit_is_complete(legacy, 0, 0, protocol, edit_branches="conditional")
    assert not Q.unit_is_complete(legacy, 0, 0, protocol, edit_branches="both")
