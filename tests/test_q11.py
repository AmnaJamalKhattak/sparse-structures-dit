"""Q11 runs end to end, exports what its figures need, and states a verdict.

The synthetic checkpoint has no decoder, so image fidelity is genuinely unavailable
here.  That is the interesting case to test: the run has to survive it and say so,
rather than reporting a missing metric as a zero.
"""
import re

import pytest
import torch

from ditsinks import SweepConfig
from ditsinks import causal_figures as CF
from ditsinks import style as ST
from ditsinks.causal_engine import EditPlan, GenerationDriver
from ditsinks.discovery import ChannelChoice, DiscoveryArtifact, LayerRanges
from ditsinks.q11 import build_schemes, lifecycle_windows, run_q11
from ditsinks.questions import QuestionContext
from ditsinks.synthetic import planted_direction


@pytest.fixture(scope="module", autouse=True)
def paper_style():
    """Every figure in this project is drawn under the paper theme, so tests are too."""
    import matplotlib
    matplotlib.use("Agg")
    ST.use("paper")


@pytest.fixture(scope="module")
def ctx():
    cfg = SweepConfig(model="tiny-flux1", prompts=["a", "b"], seeds=[0], height=64, width=64,
                      num_inference_steps=2, capture_steps=[1], dtype="float32",
                      output_dir="/tmp/ditsinks-q11-tests", save_images=False)
    driver = GenerationDriver(cfg)
    artifact = DiscoveryArtifact(
        schema_version=1, checkpoint="tiny-flux1", repo_id="synthetic", vstar_file="vstar.pt",
        fitting_population=[], explained_variance=0.9, axis_convention="", sign_convention="",
        dominant_register_channel=ChannelChoice(3, 3, True, 1.0),
        late_growing_competitor=ChannelChoice(5, None, None, 1.0),
        massive_unspecific_control=ChannelChoice(7, None, None, 1.0),
        layer_ranges=LayerRanges((0, 1), (1, 3), (3, 4)), denoising_steps=[1],
        discovery_prompts=["z"], discovery_seeds=[9], confirmation_prompts=["a", "b"],
        confirmation_seeds=[0], code_config_fingerprint="test", config_sha256="test")
    return QuestionContext.from_artifact(cfg, artifact, planted_direction(driver.bundle.d_model),
                                         driver=driver, progress=False, topk=3, percentile=95.0)


@pytest.fixture(scope="module")
def result(ctx):
    return run_q11(ctx, bits=4, rank=1, resume=False)


def test_the_wrong_windows_match_the_register_zone_in_width(ctx):
    """Position, not budget, is what the window ablation varies.

    A wrong window of a different width would confound the two, and the ablation
    would stop being an ablation.
    """
    windows = lifecycle_windows(ctx)
    assert len(windows["early"]) == len(windows["preregistered"])
    assert windows["early"].isdisjoint({max(windows["preregistered"])})
    assert min(windows["late"]) > max(windows["preregistered"])


def test_every_headline_condition_is_built(ctx):
    keys = {s.key for s in build_schemes(ctx, bits=4)}
    assert {"clean", "uniform_int4", "dominant_channel", "vstar_all_layers",
            "vstar_lifecycle", "magnitude_topk", "random_direction",
            "vstar_early_window", "vstar_late_window"} <= keys


def test_the_matched_controls_cost_at_least_as_much_as_the_method(ctx):
    """The comparison must never be won by spending more.

    ``v*`` protection is charged one high-precision scalar per token; so are the
    random-direction and magnitude controls, and the magnitude one additionally pays
    to say which coordinate it chose.  If this inverts, the headline claim is not a
    matched-budget claim.
    """
    schemes = {s.key: s for s in build_schemes(ctx, bits=4)}
    width, layers = int(ctx.vstar.numel()), int(ctx.driver.n_layers)
    method = schemes["vstar_all_layers"].budget(width, layers).bits_per_token
    for control in ("random_direction", "magnitude_topk"):
        assert schemes[control].budget(width, layers).bits_per_token >= method


def test_the_run_produces_the_tables_its_figures_need(result):
    for name in ("summary", "mechanism", "budget", "pareto", "windows"):
        assert name in result.tables, name
        assert not result.tables[name].empty, f"{name} is empty"
    assert {"bits_per_coordinate", "condition", "role"} <= set(result.tables["pareto"].columns)


def test_the_mechanism_table_is_a_depth_profile(result):
    """One row per condition per observed layer, not one row per condition.

    A single layer cannot distinguish a lifecycle window from protection everywhere,
    because the two policies are identical until they first diverge.
    """
    mechanism = result.tables["mechanism"]
    assert "layer" in mechanism.columns
    assert mechanism.groupby("condition", observed=True)["layer"].nunique().min() > 1


def test_image_metrics_are_reported_as_unavailable_not_as_zero(result):
    """The synthetic checkpoint has no decoder; a missing metric must not read as perfect."""
    assert result.meta["images_available"] is False
    summary = result.tables["summary"]
    assert summary["rmse"].isna().all(), "a run with no decoder reported a pixel distance"


def test_the_verdict_answers_the_question_without_printing_a_nan(result):
    assert result.verdict.startswith("Q11:")
    assert len(result.verdict) > 120
    assert not re.search(r"(?<![A-Za-z])[-+]?nan(?![A-Za-z])", result.verdict,
                         flags=re.IGNORECASE), result.verdict


def test_the_verdict_disclaims_latency_but_not_memory(result):
    """The disclaimer has to be the right size.

    A QDQ run cannot speak to end-to-end latency, so the verdict must say so. It *can*
    speak to the activation footprint, which is arithmetic over the bit allocation rather
    than a measurement, and an over-broad disclaimer that threw that away would understate
    a real result.
    """
    lowered = result.verdict.lower()
    assert "latency is not claimed" in lowered, result.verdict
    assert "arithmetic over the bit allocation" in lowered, result.verdict
    for forbidden in ("faster", "speedup", "throughput"):
        assert forbidden not in lowered, f"a QDQ simulation must not claim {forbidden}"


def test_every_q11_figure_draws(result, tmp_path):
    checkpoint = result.meta["checkpoint"]
    figures = [
        CF.fig_q11_quantization_pareto(result.tables["pareto"], checkpoint=checkpoint),
        CF.fig_q11_equal_budget(result.tables["pareto"], checkpoint=checkpoint),
        CF.fig_q11_depth_profile(result.tables["mechanism"], checkpoint=checkpoint,
                                 register_layers=result.meta["register_layers"]),
        CF.fig_q11_window_ablation(result.tables["pareto"], result.tables["windows"],
                                   checkpoint=checkpoint, n_layers=result.meta["n_layers"]),
        CF.render(result, path=tmp_path / "q11.pdf"),
    ]
    for figure in figures:
        assert getattr(figure, "dv_caption", ""), "every figure carries its own caption"
    assert (tmp_path / "q11.pdf").exists()


def test_the_caption_states_the_simulation_is_not_a_speedup_claim(result):
    figure = CF.fig_q11_quantization_pareto(result.tables["pareto"])
    assert "latency" in figure.dv_caption.lower()


def test_an_edit_plan_can_cover_every_batch_row():
    """A precision policy is not branch-specific, unlike a causal intervention.

    On a CFG-batched model the unconditional branch runs through the same kernels, so
    a policy applied only to the conditional row would be half-applied.
    """
    assert EditPlan(edit=lambda x, ctx: x).all_batch_rows is False
    assert EditPlan(edit=lambda x, ctx: x, all_batch_rows=True).all_batch_rows is True


def test_the_prior_art_control_is_built_before_its_basis_exists(ctx):
    """The low-rank baseline must appear in the budget table, not only at run time.

    Its basis is calibrated from the clean pass, which has not happened when a notebook
    builds the scheme list to show its costs. If the scheme were omitted until a basis
    existed, the required prior-art control would silently never run.
    """
    keys = {s.key for s in build_schemes(ctx, bits=4, bases=None)}
    assert "lowrank_absorption" in keys


def test_the_prior_art_control_actually_protects_a_subspace(result):
    """A calibrated basis must reach the scheme, or it is charged for nothing.

    An uncalibrated subspace scheme degrades to plain quantization while still paying for
    protection in the budget, which would make it a control that cannot win by
    construction.
    """
    pareto = result.tables["pareto"].set_index("condition")
    assert "lowrank_absorption" in pareto.index
    plain = float(pareto.loc["uniform_int4", "state_relative_error"])
    lowrank = float(pareto.loc["lowrank_absorption", "state_relative_error"])
    assert lowrank != pytest.approx(plain, rel=1e-6), (
        "the low-rank control is identical to unprotected quantization, so its basis "
        "never reached the scheme")
