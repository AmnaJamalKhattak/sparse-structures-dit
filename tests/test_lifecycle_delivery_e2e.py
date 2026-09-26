"""Register retiming: is the delivered state the one the computation really consumed?

The analysis in ``lifecycle.delivery_*`` rests on a claim about the model rather than
about arithmetic: that the tensor a suppression hook must control is the OUTPUT of the
block's adaptive norm, because that is what the QKV projection and the feed-forward read,
and that its per-token magnitude has already been divided out by then.

These tests check that claim against the running model instead of against the diffusers
source, and check that a ``BLOCK_INPUT`` edit is in fact upstream of it. Synthetic
weights, no download, no GPU.
"""
import pytest
import torch

from ditsinks import SweepConfig, lifecycle as LC
from ditsinks.adapters import InterventionPoint
from ditsinks.causal_engine import (CausalTracer, EditPlan, GenerationDriver,
                                    run_traced_generation, select_frozen_targets)
from ditsinks.synthetic import planted_direction

STEP = 1


def _setup(model="tiny-flux1"):
    cfg = SweepConfig(model=model, prompts=["a bowl of apples"], seeds=[0], height=64,
                      width=64, num_inference_steps=2, capture_steps=[STEP],
                      dtype="float32", output_dir="/tmp/q16-delivery", save_images=False)
    driver = GenerationDriver(cfg)
    vstar = planted_direction(driver.bundle.d_model)
    return cfg, driver, vstar


def _tracer(driver, vstar, cfg):
    return CausalTracer(driver.adapter, driver.transformer, direction=vstar,
                        layers=list(range(driver.n_layers)), steps=[STEP],
                        grid=driver.grid, cfg=cfg)


@pytest.mark.parametrize("model", ["tiny-flux1", "tiny-pixart"])
def test_the_probe_records_both_sides_of_the_consuming_normaliser(model):
    cfg, driver, vstar = _setup(model)
    layers = list(range(driver.n_layers))
    tracer = _tracer(driver, vstar, cfg)
    probe = LC.DeliveryProbe(driver.adapter, driver.transformer, tracer, layers,
                             direction=vstar)
    with probe:
        run_traced_generation(driver, tracer, prompt_id=0, prompt=cfg.prompts[0], seed=0,
                              condition="clean")
    recorded = [l for l in layers if probe.at(STEP, l)]
    assert recorded, "no block exposed a normaliser to probe"
    for layer in recorded:
        store = probe.at(STEP, layer)
        for key in LC._DELIVERY_KEYS:
            assert key in store, f"block {layer} is missing {key}"
            assert store[key].shape[0] == tracer.n_img


@pytest.mark.parametrize("model", ["tiny-flux1", "tiny-pixart"])
def test_what_the_attention_receives_is_measured_at_the_attention(model):
    """The measurement the whole delivery gate depends on, taken from the live model.

    A planted high-norm state is compared at two tensors: the residual the hook actually
    handed the block (the engine keeps it as a manipulation check) and the tensor the
    attention module was then CALLED with. If the second still carried the first's
    magnitude, delivery would have to be scored on norm rather than alignment.
    """
    cfg, driver, vstar = _setup(model)
    layers = list(range(driver.n_layers))
    v = LC._unit(vstar)

    def plant(image, ctx):
        return LC.induce_alpha(image, [4], alpha_target=60.0, direction=v)

    base = _tracer(driver, vstar, cfg)
    clean, _ = run_traced_generation(
        driver, base, prompt_id=0, prompt=cfg.prompts[0], seed=0, condition="clean",
        probes=[(InterventionPoint.BLOCK_INPUT, [0])])
    targets = select_frozen_targets(clean, layer=0, step=STEP,
                                    point=InterventionPoint.BLOCK_INPUT,
                                    direction=vstar, percentile=90.0, topk=2)
    tracer = _tracer(driver, vstar, cfg)
    probe = LC.DeliveryProbe(driver.adapter, driver.transformer, tracer, layers,
                             direction=vstar)
    with probe:
        run_traced_generation(
            driver, tracer, prompt_id=0, prompt=cfg.prompts[0], seed=0,
            condition="planted", targets=targets,
            plans=[EditPlan(edit=plant, point=InterventionPoint.BLOCK_INPUT,
                            layers=layers, steps=[STEP], label="plant")])
    rows = {int(r["layer"]): r for r in LC.delivered_rows(
        probe, step=STEP, layers=layers, carriers=[4])
        if r.get("delivered") != "not measured"}
    assert rows
    for layer, row in rows.items():
        # The residual the hook handed the block, straight from the engine's own record
        # of what the module received.
        handed = next(t for (st, ly, pt, stage, _), t
                      in tracer.trace.diagnostics.items()
                      if (st, ly, pt, stage) == (STEP, layer,
                                                 InterventionPoint.BLOCK_INPUT.value,
                                                 "x_next_module_input"))
        handed_spread = float(handed.norm(dim=-1).max() / handed.norm(dim=-1).median())
        assert handed_spread > 3.0, (
            f"block {layer}: the planted state should dominate the residual handed to "
            f"the block (spread {handed_spread:.2f})")
        assert row["consumed_norm_spread"] < 1.5, (
            f"block {layer}: the attention still sees token magnitude (spread "
            f"{row['consumed_norm_spread']:.2f} against {handed_spread:.2f} in the "
            "residual); the delivery gate assumes it does not")


@pytest.mark.parametrize("model,should_match", [("tiny-flux1", True),
                                                ("tiny-pixart", False)])
def test_the_norm_module_output_is_the_attention_input_only_on_flux(model, should_match):
    """Why delivery is probed at ATTENTION_INPUT and not at the norm module's output.

    FLUX's ``AdaLayerNormZero``/``AdaLayerNormZeroSingle`` apply their scale and shift
    inside the module, so the two tensors are bit-identical. PixArt's
    ``BasicTransformerBlock`` computes ``norm1(x) * (1 + scale_msa) + shift_msa`` in the
    block body, so they are not, and a per-channel scale is not a rotation, so the
    alignment each token delivers differs between them. Reading the module output would
    report a pre-modulation tensor as what the computation received.
    """
    cfg, driver, vstar = _setup(model)
    layers = list(range(driver.n_layers))

    def probe_at(point):
        tracer = _tracer(driver, vstar, cfg)
        probe = LC.DeliveryProbe(driver.adapter, driver.transformer, tracer, layers,
                                 direction=vstar, point=point)
        with probe:
            run_traced_generation(driver, tracer, prompt_id=0, prompt=cfg.prompts[0],
                                  seed=0, condition="clean")
        return probe

    at_attention = probe_at(InterventionPoint.ATTENTION_INPUT)
    at_norm = probe_at(InterventionPoint.PRE_KEY_NORM_RESIDUAL)
    compared = 0
    for layer in layers:
        a, b = at_attention.at(STEP, layer), at_norm.at(STEP, layer)
        if not a or "consumed_cosine" not in a or not b or "consumed_cosine" not in b:
            continue
        compared += 1
        same = torch.allclose(a["consumed_cosine"], b["consumed_cosine"], atol=1e-5)
        gap = float((a["consumed_cosine"] - b["consumed_cosine"]).abs().max())
        if should_match:
            assert same, (
                f"block {layer}: FLUX applies its modulation inside the norm module, so "
                f"these must be the same tensor (max cosine gap {gap:.4f})")
        else:
            assert not same, (
                f"block {layer}: PixArt applies its modulation outside norm1, so the "
                "module output cannot be what the attention receives -- if this ever "
                "passes, re-check the diffusers version before simplifying the probe")
            # And the gap is large enough to matter on the scale delivery is measured on.
            assert gap > 1e-3
    assert compared, "no block was compared"


def test_a_block_input_edit_reaches_the_consumer_and_suppression_removes_it():
    """The causal chain the schedule depends on, end to end.

    A ``BLOCK_INPUT`` hook runs before the block's own adaptive norm, so what it writes is
    what the QKV projection receives, and what it removes never arrives.
    """
    cfg, driver, vstar = _setup("tiny-flux1")
    layers = list(range(driver.n_layers))
    v = LC._unit(vstar)

    # A realistic register: the token's own content plus a large v* component, NOT a
    # pure multiple of v*. A pure multiple has no orthogonal part to keep, so the
    # norm-preserving removal has nothing to preserve and the test would be about a
    # degenerate case instead of about the schedule.
    def plant(image, ctx):
        return LC.induce_alpha(image, [4], alpha_target=60.0, direction=v)

    def plant_then_suppress(image, ctx):
        return LC.lifecycle_edit("suppress", [4], v)(plant(image, ctx), ctx)

    base = _tracer(driver, vstar, cfg)
    clean, _ = run_traced_generation(
        driver, base, prompt_id=0, prompt=cfg.prompts[0], seed=0, condition="clean",
        probes=[(InterventionPoint.BLOCK_INPUT, [0])])
    targets = select_frozen_targets(clean, layer=0, step=STEP,
                                    point=InterventionPoint.BLOCK_INPUT, direction=vstar,
                                    percentile=90.0, topk=2)

    def run(edit, label):
        tracer = _tracer(driver, vstar, cfg)
        probe = LC.DeliveryProbe(driver.adapter, driver.transformer, tracer, layers,
                                 direction=vstar)
        with probe:
            run_traced_generation(
                driver, tracer, prompt_id=0, prompt=cfg.prompts[0], seed=0,
                condition=label, targets=targets,
                plans=[EditPlan(edit=edit, point=InterventionPoint.BLOCK_INPUT,
                                layers=layers, steps=[STEP], label=label)])
        return LC.delivered_rows(probe, step=STEP, layers=layers, carriers=[4])

    planted = {int(r["layer"]): r for r in run(plant, "planted")
               if r.get("delivered") != "not measured"}
    cleaned = {int(r["layer"]): r for r in run(plant_then_suppress, "suppressed")
               if r.get("delivered") != "not measured"}
    assert planted and cleaned.keys() == planted.keys()
    for layer, row in planted.items():
        assert row["carrier_cos_max"] > 0.5, (
            f"block {layer}: a planted state aligned with v* must ARRIVE aligned, or a "
            "BLOCK_INPUT hook is not upstream of the consuming operation")
        assert cleaned[layer]["carrier_cos_max"] < row["carrier_cos_max"], (
            f"block {layer}: suppression at the block input must reduce what the "
            "consuming operation receives")
        # The meaningful bar is the block's own ordinary tokens, not zero. The adaptive
        # norm adds a token-independent shift, so every token's delivered cosine carries
        # the same offset; it cancels in a comparison against the ordinary population and
        # does not cancel against zero.
        assert (cleaned[layer]["carrier_cos_max"]
                <= cleaned[layer]["ordinary_cos_max"] + 1e-4), (
            f"block {layer}: after suppression the carrier must arrive no more aligned "
            "than an ordinary token of the same block")


@pytest.mark.parametrize("model", ["tiny-flux1", "tiny-pixart"])
def test_the_matched_state_is_exactly_what_each_maintained_block_receives(model):
    """Natural-register-matched induction, through the real hooks.

    The targets are calibrated from the clean trace at the INPUT of the natural block and
    written at the INPUT of each recipient block. What the block's own computation then
    receives, the engine's record of the hooked module's input, must carry the
    target projection AND the target norm, block by block across the window. Sizes are
    relative, so each recipient block gets a different absolute target.
    """
    cfg, driver, vstar = _setup(model)
    v = LC._unit(vstar)
    base = _tracer(driver, vstar, cfg)
    clean, _ = run_traced_generation(driver, base, prompt_id=0, prompt=cfg.prompts[0],
                                     seed=0, condition="clean",
                                     probes=[(InterventionPoint.BLOCK_INPUT, [0])])
    frozen = select_frozen_targets(clean, layer=0, step=STEP,
                                   point=InterventionPoint.BLOCK_INPUT, direction=vstar,
                                   percentile=90.0, topk=2)
    natural = clean.at(STEP, 3)
    carriers = [int(t) for t in torch.argsort(natural.norm, descending=True)[:2]]
    recipients = [1, 2]
    by_block, report = LC.calibrate_register_match(
        clean, step=STEP, natural_layers=[4], recipient_layers=recipients,
        carriers=carriers)
    assert set(by_block) == set(recipients) and len(report) == 2
    targets = LC.RegisterTargets()
    for layer, target in by_block.items():
        targets.add(STEP, layer, target)
    assert by_block[1].norm != pytest.approx(by_block[2].norm), (
        "relative targets should follow each recipient block's own scale")

    log = []
    tracer = _tracer(driver, vstar, cfg)
    run_traced_generation(
        driver, tracer, prompt_id=0, prompt=cfg.prompts[0], seed=0, condition="matched",
        targets=frozen,
        plans=[EditPlan(edit=LC.lifecycle_edit("induce_matched", carriers, vstar,
                                               register_targets=targets, records=log),
                        point=InterventionPoint.BLOCK_INPUT, layers=recipients,
                        steps=[STEP], label="matched")])
    for layer in recipients:
        received = next(t for (st, ly, pt, stage, _), t in tracer.trace.diagnostics.items()
                        if (st, ly, pt, stage) == (STEP, layer,
                                                   InterventionPoint.BLOCK_INPUT.value,
                                                   "x_next_module_input"))
        received = LC._conditional_row(received)
        for token in carriers:
            alpha, norm = by_block[layer].at(token)
            assert float(received[token] @ v) == pytest.approx(alpha, rel=1e-4, abs=1e-4)
            assert float(received[token].norm()) == pytest.approx(norm, rel=1e-4)
    written = {(r.layer, r.token): r for r in log if r.step == STEP}
    assert set(written) == {(l, t) for l in recipients for t in carriers}
    for (layer, token), record in written.items():
        assert record.norm_after == pytest.approx(by_block[layer].at(token)[1], rel=1e-4)


@pytest.mark.parametrize("model", ["tiny-flux1", "tiny-pixart"])
def test_a_pre_feed_forward_transfer_changes_only_what_the_feed_forward_reads(model):
    """The pre-feed-forward transfer site, checked against the running model.

    The hook sits on norm2's input. If it edited the residual stream, the edited token's
    block output would carry the source state; if it edits only the feed-forward's
    input, the output differs from clean by what the feed-forward WROTE, and no other
    token of that block changes, the feed-forward is per token, and attention has
    already run.
    """
    cfg, driver, vstar = _setup(model)
    layer, token = 1, 4
    probe_point = InterventionPoint.PRE_MLP_RESIDUAL
    base = _tracer(driver, vstar, cfg)
    clean, _ = run_traced_generation(
        driver, base, prompt_id=0, prompt=cfg.prompts[0], seed=0, condition="clean",
        probes=[(InterventionPoint.BLOCK_INPUT, [0]), (probe_point, [layer])])
    frozen = select_frozen_targets(clean, layer=0, step=STEP,
                                   point=InterventionPoint.BLOCK_INPUT, direction=vstar,
                                   percentile=90.0, topk=2)
    states = clean.probe(STEP, layer, probe_point)
    assert states is not None
    source = states[9] * 3.0 + 1.0            # some other token's feed-forward input

    tracer = _tracer(driver, vstar, cfg)
    log = []
    treated, _ = run_traced_generation(
        driver, tracer, prompt_id=0, prompt=cfg.prompts[0], seed=0, condition="transfer",
        targets=frozen,
        plans=[EditPlan(edit=LC.transplant_edit([token], {token: source}, vstar, records=log),
                        point=probe_point, layers=[layer], steps=[STEP], label="transfer")])
    received = next(t for (st, ly, pt, stage, _), t in tracer.trace.diagnostics.items()
                    if (st, ly, pt, stage) == (STEP, layer, probe_point.value,
                                               "x_next_module_input"))
    received = LC._conditional_row(received)
    assert float(torch.nn.functional.cosine_similarity(received[token], source, dim=0)) \
        == pytest.approx(1.0, abs=1e-4)
    out_clean, out_treated = clean.at(STEP, layer), treated.at(STEP, layer)
    moved = (out_treated.norm - out_clean.norm).abs() + \
        (out_treated.projection - out_clean.projection).abs()
    assert float(moved[token]) > 1e-4, "the feed-forward's write at the token changed"
    others = [t for t in range(int(moved.shape[0])) if t != token]
    assert float(moved[others].max()) < 1e-4, "no other token of the block may change"
    assert log and log[0].kind == "transplant"


def test_the_alignment_rule_catches_a_state_planted_at_a_new_position():
    """Relocation, in a live run: the state is put where no clean carrier
    was, and the clean-referenced rule must remove it at the next block's input."""
    cfg, driver, vstar = _setup("tiny-flux1")
    v = LC._unit(vstar)
    layers = list(range(driver.n_layers))
    base = _tracer(driver, vstar, cfg)
    clean, _ = run_traced_generation(driver, base, prompt_id=0, prompt=cfg.prompts[0],
                                     seed=0, condition="clean",
                                     probes=[(InterventionPoint.BLOCK_INPUT, [0])])
    frozen = select_frozen_targets(clean, layer=0, step=STEP,
                                   point=InterventionPoint.BLOCK_INPUT, direction=vstar,
                                   percentile=90.0, topk=2)
    carriers = [int(t) for t in torch.argsort(clean.at(STEP, 2).norm, descending=True)[:2]]
    planted = next(t for t in range(40) if t not in carriers)
    ref = LC.CleanStateReference(highnorm_ratio=3.0, alignment_quantile=0.99,
                                 frozen=carriers)
    ref.add(clean, steps=[STEP], layers=layers)
    sites = []
    suppressor = LC.Suppressor(vstar, mode="state", rule=LC.AlignmentRule(ref), sites=sites,
                               original=carriers)
    tracer = _tracer(driver, vstar, cfg)
    treated, _ = run_traced_generation(
        driver, tracer, prompt_id=0, prompt=cfg.prompts[0], seed=0, condition="relocated",
        targets=frozen,
        plans=[EditPlan(edit=LC.lifecycle_edit("induce", [planted], vstar, alpha_target=60.0),
                        point=InterventionPoint.BLOCK_INPUT, layers=[1], steps=[STEP],
                        label="plant"),
               EditPlan(edit=suppressor.edit, point=InterventionPoint.BLOCK_INPUT,
                        layers=[2, 3], steps=[STEP], label="suppress")])
    at_two = next(s for s in sites if s.layer == 2 and s.step == STEP)
    assert planted in at_two.new_ids, "the planted state at a new position was not caught"
    assert at_two.received_n_above_bar == 0
    rows = LC.received_state_rows(treated, sites, ref, step=STEP, layers=[2, 3],
                                  carriers=carriers)
    assert all(r["received_n_above_bar"] == 0 for r in rows)
    relocation = LC.regrowth_relocation_rows(treated, ref, step=STEP, layers=[1],
                                             carriers=carriers)
    assert planted in relocation[0]["new_aligned_ids"]
