"""Each question runs, produces the columns its figure needs, and states a verdict."""
import pytest
import torch

from ditsinks import SweepConfig
from ditsinks.causal_engine import GenerationDriver
from ditsinks.discovery import ChannelChoice, DiscoveryArtifact, LayerRanges
from ditsinks.questions import (QUESTION_RUNNERS, QUESTION_TITLES, QuestionContext, head_level,
                                layer_level, run_q1, run_q2, run_q3, run_q4, run_q5, run_q6)
from ditsinks.synthetic import planted_direction


@pytest.fixture(scope="module")
def ctx():
    cfg = SweepConfig(model="tiny-flux1", prompts=["a", "b"], seeds=[0], height=64, width=64,
                      num_inference_steps=2, capture_steps=[1], dtype="float32",
                      output_dir="/tmp/ditsinks-questions", save_images=False)
    driver = GenerationDriver(cfg)
    direction = planted_direction(driver.bundle.d_model)
    artifact = DiscoveryArtifact(
        schema_version=1, checkpoint="tiny-flux1", repo_id="synthetic", vstar_file="vstar.pt",
        fitting_population=[], explained_variance=0.9, axis_convention="", sign_convention="",
        dominant_register_channel=ChannelChoice(3, 3, True, 1.0),
        late_growing_competitor=ChannelChoice(5, None, None, 1.0),
        massive_unspecific_control=ChannelChoice(7, None, None, 1.0),
        layer_ranges=LayerRanges((0, 1), (1, 3), (3, 4)), denoising_steps=[1],
        discovery_prompts=["z"], discovery_seeds=[9], confirmation_prompts=["a", "b"],
        confirmation_seeds=[0], code_config_fingerprint="test", config_sha256="test")
    return QuestionContext.from_artifact(cfg, artifact, direction, driver=driver, progress=False,
                                         topk=3, percentile=95.0)


@pytest.fixture(scope="module")
def results(ctx):
    return {"q1": run_q1(ctx), "q2": run_q2(ctx, gammas=[0.0, 1.0]), "q3": run_q3(ctx),
            "q4": run_q4(ctx), "q5": run_q5(ctx), "q6": run_q6(ctx)}


def test_every_question_has_a_runner_and_a_title():
    assert set(QUESTION_RUNNERS) == set(QUESTION_TITLES) == {"q1", "q2", "q3", "q4", "q5", "q6"}


def test_every_question_produces_rows_and_a_verdict(results):
    for key, result in results.items():
        assert not result.tidy.empty, f"{key} produced no rows"
        assert result.verdict.startswith(key.upper() + ":"), result.verdict
        assert len(result.verdict) > 60, f"{key} verdict is too terse to be an answer"


def test_verdicts_never_print_a_raw_nan(results):
    """An unmeasurable quantity is reported in words, not as a bare 'nan'.

    Matched on word boundaries, because "dominant" legitimately contains the
    letters and a substring test would fail on every correct verdict.
    """
    import re

    for key, result in results.items():
        assert not re.search(r"(?<![A-Za-z])[-+]?nan(?![A-Za-z])", result.verdict,
                             flags=re.IGNORECASE), result.verdict


def test_q1_tidy_carries_everything_its_figure_reads(results):
    tidy = results["q1"].tidy
    for column in ("condition", "condition_label", "role", "layer", "head", "fate",
                   "head_retention", "attention_concentration", "vstar_projection_change",
                   "selection_rule", "prompt_id", "seed"):
        assert column in tidy.columns, column
    assert set(tidy["selection_rule"]) == {"percentile", "topk"}, "the threshold-free run is missing"
    assert not layer_level(tidy).empty and not head_level(tidy).empty


def test_q1_runs_every_control_the_plan_requires(results):
    conditions = set(results["q1"].tidy["condition"])
    assert {"sham", "random_tokens_zeroed", "norm_matched_tokens_zeroed",
            "offregister_direction_removal"} <= conditions
    assert {"direction_removal", "matched_ordinary_state", "ordinary_norm_clamp",
            "state_zeroed"} <= conditions


def test_q1_fate_labels_come_from_a_closed_set(results):
    from ditsinks.endpoints import FATES

    assert set(results["q1"].tables["fates"]["fate"]) <= set(FATES)
    summary = results["q1"].tables["retention_summary"]
    assert {"affected_head_macro", "all_head_macro", "affected_head_pooled",
            "n_affected_heads", "temporal_endpoint"} <= set(summary.columns)
    assert not results["q1"].tables["diagnostics"].empty


def test_q2_sweeps_a_dose_and_names_its_controls(results):
    dose = results["q2"].tables["dose_response"]
    primary = dose[dose["control"] == "dominant_channel"]
    assert set(primary["gamma"]) == {0.0, 1.0}
    assert set(primary["scope"]) == {"writer_only", "maintenance"}
    assert {"competing_channel", "random_channel", "matched_energy_direction",
            "ordinary_positions"} <= set(dose["control"])
    for column in ("vstar_projection_change", "original_sink_retained", "control_label"):
        assert column in dose.columns
    assert not results["q2"].tables["diagnostics"].empty
    assert "current_block_output" in set(results["q2"].tidy["temporal_endpoint"])


def test_q2_gamma_one_is_the_sham_scale(results):
    """Scaling a channel by one must leave the projection where the clean run put it."""
    dose = results["q2"].tables["dose_response"]
    sham = dose[(dose["control"] == "dominant_channel") & (dose["gamma"] == 1.0)]
    assert abs(float(sham["vstar_projection_change"].mean())) < 1e-6


def test_q3_classifies_recovery_and_tests_restoration(results):
    recovery = results["q3"].tables["recovery"]
    from ditsinks.endpoints import RECOVERY_KINDS

    assert set(recovery["recovery_kind"]) <= set(RECOVERY_KINDS)
    assert {"single_shot", "fixed_carrier_repeated", "dynamic_register_state"} <= set(recovery["scope"])
    assert "token_history" in results["q3"].tables
    assert "strict_summary" in results["q3"].tables
    assert "rescue" in set(recovery["role"]), "no clean-state restoration condition ran"


def test_q4_reports_both_patch_directions_against_a_sham(results):
    patches = results["q4"].tables["patch_effects"]
    assert set(patches["direction"]) == {"register_to_ordinary", "ordinary_to_register"}
    assert patches["sham_value"].notna().any(), "patch effects were not paired against a sham"
    assert "capture_rate" in set(patches["endpoint"])


def test_q4_records_whether_a_feature_varies_across_tokens(results):
    """A modulation term shared by every token cannot select which token wins."""
    separation = results["q4"].tables["separation"]
    assert "token_dependence" in separation.columns
    assert separation["separation"].between(0.0, 1.0).all()


def test_q5_marks_unsupported_rungs_instead_of_faking_them(results):
    ladder = results["q5"].tidy
    unsupported = ladder[~ladder["supported"].astype(bool)]
    assert not unsupported.empty, "FLUX should refuse the pre-position key rung"
    assert unsupported["capture_rate"].isna().all()
    assert unsupported["reason"].str.len().gt(0).all(), "a refusal must say why"
    assert "key_before_position" in set(unsupported["stage"])


def test_q5_runs_every_recipient_position(results):
    assert set(results["q5"].tidy["position"]) == {"natural_register", "adjacent_patch",
                                                   "random_ordinary"}
    ladder = results["q5"].tidy
    assert {"matched_clean_rate", "any_register_clean_rate", "temporal_endpoint",
            "transfer_mode", "source_capture_rate"} <= set(ladder.columns)
    assert {"copy", "move"} <= set(ladder[ladder["position"] != "natural_register"]["transfer_mode"])
    assert "same_operation" in set(ladder["temporal_endpoint"])


def test_q6_reports_a_lifetime_and_its_fallback(results):
    lifetime = results["q6"].tables["lifetime"]
    for column in ("lifetime_shift", "half_life_shift", "clean_lifetime", "condition_label",
                   "cosine_lifetime", "projection_lifetime", "high_norm_lifetime", "sink_lifetime"):
        assert column in lifetime.columns
    assert lifetime["half_life_shift"].notna().any(), "the continuous fallback was never computed"
    trajectory = results["q6"].tables["trajectory"]
    assert {"alpha", "mean_abs_alpha", "rms_alpha", "mean_perp_norm", "rms_perp_norm",
            "perpendicular_norm", "cosine", "angular_velocity"} <= set(trajectory.columns)
    assert "clean" in set(trajectory["condition"])


def test_q6_decomposition_names_literal_per_token_summaries(results):
    trajectory = results["q6"].tables["trajectory"]
    row = trajectory.iloc[0]
    assert row["perpendicular_norm"] == pytest.approx(row["mean_perp_norm"])
    assert row["rms_alpha"] ** 2 + row["rms_perp_norm"] ** 2 == pytest.approx(
        row["norm"] ** 2, rel=1e-4)


def test_results_save_a_tidy_table_and_a_verdict(results, tmp_path):
    written = results["q1"].save(tmp_path / "q1")
    assert written["tidy"].exists() and written["meta"].exists()
    assert "verdict" in written["meta"].read_text()


def test_context_never_reselects_layers_or_channels(ctx):
    """Everything the runners use comes from the frozen artifact."""
    assert ctx.dominant_channel == 3 and ctx.competitor_channel == 5 and ctx.control_channel == 7
    assert ctx.writer_layer == 1 and ctx.register_layers[0] == 1
    assert ctx.intervention_layer == ctx.register_layers[0]
    assert set(ctx.downstream_layers) <= set(ctx.observe_layers)
    assert min(ctx.downstream_layers) >= ctx.intervention_layer


# ------------------------------------------------------- Q1 geometry and resume
def test_direction_removal_preserves_the_magnitude(ctx):
    """Removing the v* component must not also shrink the token.

    x - <x,v*>v* has norm ||x||*sqrt(1-cos^2). For a register nearly collinear
    with v* (the tested claim), a plain removal collapses the norm, so
    the condition would stop isolating direction from magnitude.
    """
    from ditsinks.questions import Q1_CONDITIONS, _q1_edit

    condition = next(c for c in Q1_CONDITIONS if c.key == "direction_removal")
    width = int(ctx.vstar.numel())
    direction = ctx.vstar / ctx.vstar.norm()
    edit = _q1_edit(condition, ctx.vstar, [2], None)

    for label, token in (
            # A token exactly along v*: the degenerate case, where whatever survives
            # the subtraction is numerical noise.
            ("collinear", direction * 10.0),
            # And the ordinary case, where a real orthogonal component remains.
            ("partly aligned", (0.9 * direction + 0.436 * _orthogonal_to(direction)) * 10.0)):
        image = torch.randn(6, width) * 0.01
        image[2] = token
        before = float(image[2].norm())
        after = edit(image.clone(), None)
        assert float(after[2].norm()) == pytest.approx(before, rel=1e-3), \
            f"{label}: the magnitude was destroyed"
        assert abs(float(after[2] @ direction)) < 0.05 * before, \
            f"{label}: the direction was not removed"


def _orthogonal_to(direction: torch.Tensor) -> torch.Tensor:
    """A fixed unit vector at right angles to ``direction``, for building fixtures."""
    generator = torch.Generator().manual_seed(3)
    other = torch.randn(direction.shape, generator=generator)
    other = other - (other @ direction) * direction
    return other / other.norm()


def test_q1_has_a_control_matched_in_strength_to_the_intervention():
    """Zeroing is far more violent than removing a direction; one control must match."""
    from ditsinks.questions import Q1_CONDITIONS

    by_key = {c.key: c for c in Q1_CONDITIONS}
    matched = by_key["normmatched_direction_removal"]
    headline = by_key["direction_removal"]
    assert matched.edit == headline.edit, "the control must apply the identical operation"
    assert matched.group != headline.group and matched.role == "control"


def test_q1_figure_columns_separate_direction_from_magnitude(results):
    """Panel (d) needs both factors; their product alone cannot distinguish them."""
    tidy = results["q1"].tidy
    assert {"cosine", "target_norm_ratio"} <= set(tidy.columns)
    layers = layer_level(tidy)
    assert layers["cosine"].notna().any() and layers["target_norm_ratio"].notna().any()


def test_a_completed_question_is_reused_rather_than_recomputed(ctx, tmp_path):
    from dataclasses import replace

    from ditsinks.questions import QuestionResult, run_q1

    scoped = replace(ctx, output_dir=tmp_path, progress=False)
    first = run_q1(scoped)
    assert (tmp_path / "q1" / "q1_tidy.csv").exists(), "the runner did not save itself"
    again = run_q1(scoped)
    assert again.tidy.shape == first.tidy.shape and again.verdict == first.verdict
    # A reloaded result must be indistinguishable from a computed one.
    loaded = QuestionResult.load("q1", tmp_path / "q1")
    assert loaded.verdict == first.verdict and sorted(loaded.tables) == sorted(first.tables)


def test_resume_false_recomputes(ctx, tmp_path):
    from dataclasses import replace

    from ditsinks.questions import run_q1

    scoped = replace(ctx, output_dir=tmp_path, progress=False)
    first = run_q1(scoped)
    forced = run_q1(scoped, resume=False)
    assert forced.tidy.shape == first.tidy.shape


def test_load_says_plainly_when_there_is_nothing_saved(tmp_path):
    from ditsinks.questions import QuestionResult

    with pytest.raises(FileNotFoundError, match="no saved Q1"):
        QuestionResult.load("q1", tmp_path)


# ------------------------------------------------------- Q4 and the boundary
def _artifact(writer_range):
    """A discovery artifact whose writer range can be pointed anywhere."""
    return DiscoveryArtifact(
        schema_version=1, checkpoint="tiny-flux1", repo_id="synthetic", vstar_file="vstar.pt",
        fitting_population=[], explained_variance=0.9, axis_convention="", sign_convention="",
        dominant_register_channel=ChannelChoice(3, 3, True, 1.0),
        late_growing_competitor=ChannelChoice(5, None, None, 1.0),
        massive_unspecific_control=ChannelChoice(7, None, None, 1.0),
        layer_ranges=LayerRanges(writer_range, (2, 3), (3, 4)), denoising_steps=[1],
        discovery_prompts=["z"], discovery_seeds=[9], confirmation_prompts=["a", "b"],
        confirmation_seeds=[0], code_config_fingerprint="test", config_sha256="test")


def _context(ctx, writer_range):
    from ditsinks.questions import QuestionContext

    return QuestionContext.from_artifact(
        ctx.cfg, _artifact(writer_range), ctx.vstar, driver=ctx.driver, progress=False,
        topk=3, percentile=95.0)


def test_q4_steps_back_when_the_writer_range_crosses_the_block_boundary(ctx):
    """The tiny model is two dual blocks then three single ones, exactly as
    FLUX.1-schnell is nineteen then thirty-eight. A writer range that ends on the
    first single block is the case that produced no patching runs at all: the range
    is a claim about depth, and depth can cross an architectural boundary."""
    from ditsinks.questions import Q4_FEATURES

    refs = ctx.driver.adapter.layers(ctx.driver.transformer)
    kinds = [r.kind for r in refs]
    assert kinds[:2] == ["dual", "dual"] and kinds[2] == "single", kinds

    straddling = _context(ctx, (1, 2))
    assert straddling.writer_layer == 2, "the range's end is the single block"
    chosen = straddling.layer_exposing([f.point for f in Q4_FEATURES], straddling.writer_range)
    assert chosen == 1, "Q4 must step back to the dual block that exposes its features"
    assert refs[chosen].kind == "dual"


def test_q4_run_on_a_straddling_range_produces_patching_rows(ctx):
    """The regression itself: the run this reproduces returned nothing at all."""
    from ditsinks.questions import run_q4

    result = run_q4(_context(ctx, (1, 2)), resume=False)
    assert result.meta["writer_layer"] == 1 and result.meta["layer_kind"] == "dual"
    assert not result.meta["unsupported"], result.meta["unsupported"]
    patches = result.tables["patch_effects"]
    assert not patches.empty and patches["supported"].astype(bool).any()


def test_q4_records_a_refusal_rather_than_dropping_the_feature(ctx):
    """A feature that vanishes from the table reads as tested-and-null. It is not."""
    from ditsinks.questions import run_q4

    result = run_q4(_context(ctx, (3, 4)), resume=False)
    assert result.meta["layer_kind"] == "single"
    assert "pre_mlp_residual" in result.meta["unsupported"]
    patches = result.tables["patch_effects"]
    refused = patches[~patches["supported"].astype(bool)]
    assert not refused.empty, "the unsupported feature left no row"
    assert refused["effect"].isna().all(), "a refusal must not carry a number"
    assert refused["reason"].str.len().gt(0).all(), "a refusal must say why"
    # ...and the ones a single block *does* expose still ran.
    assert patches["supported"].astype(bool).any(), "no feature ran on a single block"


def test_q4_keeps_its_columns_even_when_a_feature_is_refused(ctx):
    from ditsinks.questions import run_q4

    patches = run_q4(_context(ctx, (3, 4)), resume=False).tables["patch_effects"]
    for column in ("patch", "direction", "endpoint", "effect", "supported", "reason"):
        assert column in patches.columns


# ------------------------------------------------- structure interaction maps
@pytest.fixture(scope="module")
def maps(ctx):
    from ditsinks.questions import run_structure_maps

    return run_structure_maps(ctx, units=ctx.units()[:1], save_images=False)


def test_structure_maps_cover_every_removal_and_structure(maps):
    from ditsinks.questions import REMOVALS, STRUCTURES

    assert set(maps.tidy["removal"]) == {k for k, _ in REMOVALS}
    assert set(maps.tidy["structure"]) == {k for k, _ in STRUCTURES}
    for column in ("row", "col", "token", "value", "is_frozen_target", "method"):
        assert column in maps.tidy.columns


def test_every_pair_carries_the_method_it_declares(maps):
    """The figure fills each panel by a declared rule, so the run must supply it."""
    from ditsinks.questions import REMOVALS, STRUCTURES, ablation_method

    have = set(zip(maps.tidy["removal"], maps.tidy["structure"], maps.tidy["method"]))
    for removal, _ in REMOVALS:
        for structure, _ in STRUCTURES:
            method = ablation_method(removal, structure)
            assert (removal, structure, method) in have, \
                f"no {method} panel for removing {removal} and measuring {structure}"


def test_bookkeeping_is_offered_only_where_it_is_defined(maps):
    """Attention is formed inside the block, so no mask on the residual stream
    afterwards can say where a head would have looked."""
    from ditsinks.questions import SUBTRACTABLE

    offered = {(r, s) for r, s, m in zip(maps.tidy["removal"], maps.tidy["structure"],
                                         maps.tidy["method"]) if m == "subtract"}
    assert offered == set(SUBTRACTABLE)
    assert ("dominant_channel", "attention_sinks") not in offered
    assert not any(r == "attention_sinks" for r, _ in offered)


def test_masking_the_channel_leaves_nothing_of_the_channel(maps):
    """The one panel that is flat by construction, and is shown for that reason."""
    cell = maps.tidy[(maps.tidy["removal"] == "dominant_channel")
                     & (maps.tidy["structure"] == "dominant_channel")
                     & (maps.tidy["method"] == "subtract")]
    assert not cell.empty
    assert cell["value"].abs().max() == 0.0


def test_masking_the_registers_takes_their_attention_share_to_zero(maps):
    """Dropping a patch from the competition and renormalising is what "this patch
    was not there to attend to" means for a distribution."""
    cell = maps.tidy[(maps.tidy["removal"] == "high_norm_tokens")
                     & (maps.tidy["structure"] == "attention_sinks")
                     & (maps.tidy["method"] == "subtract")]
    assert not cell.empty
    assert cell[cell["is_frozen_target"]]["value"].abs().max() == 0.0
    assert cell[~cell["is_frozen_target"]]["value"].sum() > 0


def test_masking_the_registers_takes_their_norm_down(maps):
    """Replacing the register states with ordinary ones, in the books, has to cost
    them their norm, otherwise the bookkeeping is not doing anything."""
    before = maps.tidy[(maps.tidy["removal"] == "clean")
                       & (maps.tidy["structure"] == "high_norm_tokens")]
    after = maps.tidy[(maps.tidy["removal"] == "high_norm_tokens")
                      & (maps.tidy["structure"] == "high_norm_tokens")
                      & (maps.tidy["method"] == "subtract")]
    assert after[after["is_frozen_target"]]["value"].mean() < \
        before[before["is_frozen_target"]]["value"].mean()


def test_the_channel_is_followed_through_depth(maps):
    """A single layer cannot separate "the edit never landed" from "the edit landed
    and the network put it back", and those are opposite conclusions."""
    trace = maps.tables["channel_trace"]
    assert not trace.empty
    for column in ("removal", "layer", "at_registers", "elsewhere"):
        assert column in trace.columns
    from ditsinks.questions import REMOVALS

    assert set(trace["removal"]) == {k for k, _ in REMOVALS}
    assert set(trace["layer"]) == set(maps.meta["observe_layers"]), \
        "every observed layer is already in the trace; none should be dropped"


def test_the_register_channel_stands_above_the_ordinary_one(maps):
    """The diagnostic is only readable against the level it is supposed to exceed."""
    clean = maps.tables["channel_trace"]
    clean = clean[clean["removal"] == "clean"]
    assert (clean["at_registers"] > clean["elsewhere"]).all()


def test_the_survival_table_reads_one_method_only(maps):
    """Averaging a bookkeeping panel with a re-run one reports neither quantity."""
    from ditsinks.questions import _structure_survival

    causal = _structure_survival(maps.tidy)
    subtract = _structure_survival(maps.tidy, method="subtract")
    assert not causal.empty and not subtract.empty
    key = ("dominant_channel", "high_norm_tokens")
    pick = lambda f: float(f[(f["removal"] == key[0]) & (f["structure"] == key[1])]["survival"]
                           .mean())
    assert pick(causal) != pytest.approx(pick(subtract), abs=1e-9), \
        "the two methods happened to agree exactly, which no longer tests anything"


def test_treated_maps_are_measured_against_the_untouched_yardstick(maps):
    """The high-norm map divides by the untouched median, not by its own: a run
    whose patch norms all fell together would otherwise look untouched."""
    from ditsinks.questions import structure_map

    class _Fake:
        norm = torch.tensor([1.0, 2.0, 3.0])
        channel_values = None
        incoming = None

    assert structure_map(_Fake(), "high_norm_tokens").tolist() == [0.5, 1.0, 1.5]
    assert structure_map(_Fake(), "high_norm_tokens", scale=4.0).tolist() == [0.25, 0.5, 0.75]


def test_each_removal_reduces_the_structure_it_targets(maps):
    """The diagonal is the manipulation check: an edit must do what it claims."""
    survival = (maps.tables["survival"].groupby(["removal", "structure"], observed=True)["survival"]
                .mean())
    for structure in ("high_norm_tokens", "dominant_channel", "attention_sinks"):
        if (structure, structure) not in survival.index:
            continue
        assert survival.loc[(structure, structure)] < 1.0, \
            f"removing {structure} did not reduce {structure}"


def test_removing_the_sink_leaves_the_residual_state_alone(maps):
    """Key-space suppression is the one intervention that separates routing
    from representation, so it must not move the norm or the channel."""
    survival = (maps.tables["survival"].groupby(["removal", "structure"], observed=True)["survival"]
                .mean())
    for structure in ("high_norm_tokens", "dominant_channel"):
        value = survival.loc[("attention_sinks", structure)]
        assert value == pytest.approx(1.0, abs=0.02), \
            f"suppressing the sink moved {structure} to {value:.2f} of clean"


def test_structure_maps_lay_onto_a_grid(maps):
    grid_rows = maps.tidy["row"].max() + 1
    grid_cols = maps.tidy["col"].max() + 1
    panels = maps.tidy.groupby(["removal", "structure", "method"], observed=True).size()
    assert set(panels) == {grid_rows * grid_cols}, \
        "every panel must hold exactly one value per patch"


def test_generated_images_survive_a_save_and_reload(maps, tmp_path):
    """A Colab runtime ending is the normal case; the images have to outlive it."""
    from PIL import Image

    from ditsinks.questions import QuestionResult

    maps.images["prompt0_seed0"] = Image.new("RGB", (8, 8), (10, 20, 30))
    written = maps.save(tmp_path)
    assert any(key.startswith("image_") for key in written)
    reloaded = QuestionResult.load("maps", tmp_path)
    assert set(reloaded.images) == set(maps.images)
    assert reloaded.images["prompt0_seed0"].size == (8, 8)


def test_structure_map_refuses_an_unknown_structure():
    from ditsinks.questions import structure_map

    with pytest.raises(KeyError, match="unknown structure"):
        structure_map(object(), "something_else")


def test_maps_verdict_reports_all_three_removals(maps):
    for phrase in ("High-norm tokens removed", "Dominant channel suppressed",
                   "Attention sinks suppressed"):
        assert phrase in maps.verdict
