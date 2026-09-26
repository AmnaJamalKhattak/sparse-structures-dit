"""The control surface separates direction from magnitude, and the separation has to be exact.

The whole experiment rests on two claims about the operators: that ``beta`` moves
alignment without moving norm, and that ``gamma`` moves norm without moving direction.
If either leaks, every cell of the grid confounds the two variables it was built to
separate and no amount of downstream analysis recovers the difference.  Those two
invariants, the ``(1, 1)`` round trip, and the provenance of the rescue target are the
four things these tests exist to hold.
"""
import json
import math
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import torch

from ditsinks import SweepConfig
from ditsinks import causal_figures as CF
from ditsinks import control_surface as CS
from ditsinks import image_detail as ID
from ditsinks import style as ST
from ditsinks.causal_engine import EditContext, GenerationDriver, FrozenTargets, Trace
from ditsinks.adapters import InterventionPoint
from ditsinks.discovery import ChannelChoice, DiscoveryArtifact, LayerRanges
from ditsinks.questions import QuestionContext
from ditsinks.synthetic import planted_direction


BETAS = (0.0, 0.5, 1.0, 1.5, 2.0)
GAMMAS = (0.5, 0.75, 1.0, 1.25, 1.5)


@pytest.fixture(scope="module", autouse=True)
def paper_style():
    import matplotlib
    matplotlib.use("Agg")
    ST.use("paper")


@pytest.fixture
def states():
    """A loud, partly aligned population, the shape the real registers have."""
    torch.manual_seed(0)
    x = torch.randn(24, 96) * 1.5
    v = torch.randn(96)
    unit = CS._unit(v)
    x[:4] += 18.0 * unit                      # four strongly aligned high-norm tokens
    return x, v


# ============================================================= the two operators
def test_beta_moves_alignment_without_moving_norm(states):
    """The first invariant. A norm that drifts with beta confounds the two axes."""
    x, v = states
    for beta in BETAS:
        y, _ = CS.align_scale(x, v, beta)
        assert torch.allclose(y.norm(dim=-1), x.norm(dim=-1), atol=1e-4, rtol=1e-5), beta


def test_beta_raises_alignment_monotonically(states):
    """And it has to actually do something, in the direction the maths says.

    ``cos(y~, v) = beta.alpha / sqrt(|r|^2 + beta^2 alpha^2)`` is increasing in ``beta``,
    so an implementation that scaled the residual instead of the parallel part would
    preserve the norm and fail here.
    """
    x, v = states
    unit = CS._unit(v)
    cosines = []
    for beta in BETAS:
        y, _ = CS.align_scale(x, v, beta)
        cosines.append(float((y @ unit / y.norm(dim=-1)).abs().mean()))
    assert cosines == sorted(cosines), cosines
    assert cosines[0] < 1e-6, "beta=0 must remove the alignment entirely"
    assert cosines[-1] > cosines[2] > cosines[0]


def test_gamma_moves_norm_without_moving_direction(states):
    """The second invariant, and the one a reader will assume without checking."""
    x, v = states
    reference, _ = CS.align_scale(x, v, 1.5)
    for gamma in GAMMAS:
        out, _ = CS.control_surface_states(x, v, 1.5, gamma)
        assert torch.allclose(out.norm(dim=-1), gamma * x.norm(dim=-1), atol=1e-4, rtol=1e-5)
        cosine = torch.nn.functional.cosine_similarity(out, reference, dim=-1)
        assert torch.allclose(cosine, torch.ones_like(cosine), atol=1e-6), gamma


def test_the_centre_cell_reconstructs_the_clean_state(states):
    """``(beta, gamma) = (1, 1)`` is the experiment's zero and must be earned, not assumed.

    It is deliberately *not* short-circuited: the grid runs the real edit at the centre
    so that cell measures the model's numerical sensitivity. That only means anything if
    the operator round trip is itself exact, which is what this asserts.
    """
    x, v = states
    rebuilt, degenerate = CS.control_surface_states(x, v, 1.0, 1.0)
    assert degenerate == 0
    scale = float(x.abs().max())
    assert CS.identity_error(x, v) < 1e-5 * scale, CS.identity_error(x, v)
    assert torch.allclose(rebuilt, x, atol=1e-4, rtol=1e-5)


def test_a_token_collinear_with_vstar_takes_the_seeded_fallback():
    """Removing the alignment of a token that is *only* alignment needs a definition.

    ``y(0)`` vanishes, and renormalising the numerical residue would amplify noise or
    even restore the original direction with its sign flipped. The fallback is an
    explicit seeded orthogonal direction, and the count is returned so a run reports it
    instead of hiding it.
    """
    v = torch.randn(64)
    unit = CS._unit(v)
    x = torch.stack([7.0 * unit, 7.0 * unit + torch.randn(64)])
    out, degenerate = CS.align_scale(x, v, 0.0, seed=3)
    assert degenerate == 1, "the collinear token must be the one that falls back"
    assert torch.allclose(out.norm(dim=-1), x.norm(dim=-1), atol=1e-4, rtol=1e-5)
    assert abs(float(out[0] @ unit)) < 1e-4 * float(x[0].norm())
    again, _ = CS.align_scale(x, v, 0.0, seed=3)
    assert torch.allclose(out, again), "the fallback must be reproducible from the seed"


def test_energy_matching_is_exact_per_token(states):
    """A control that perturbs by a different amount is not a control.

    Matching per token rather than in total is the stronger contract: a run-level energy
    total can be met by a few tokens moving a great deal.
    """
    x, v = states
    edited, _ = CS.control_surface_states(x, v, 2.0, 1.0)
    target = torch.linspace(0.5, 4.0, x.shape[0])
    matched = CS.match_perturbation(x, edited, target)
    assert torch.allclose((matched - x).norm(dim=-1), target, atol=1e-4, rtol=1e-4)


def test_the_orthogonal_control_direction_is_orthogonal_and_reproducible(states):
    _, v = states
    unit = CS._unit(v)
    first = CS.orthogonal_direction(unit, int(unit.numel()), seed=11)
    assert abs(float(first @ unit)) < 1e-6
    assert abs(float(first.norm()) - 1.0) < 1e-6
    assert torch.allclose(first, CS.orthogonal_direction(unit, int(unit.numel()), seed=11))
    assert not torch.allclose(first, CS.orthogonal_direction(unit, int(unit.numel()), seed=12))


# ================================================================== the rescue
def _targets(clean: torch.Tensor, ids, *, norm_threshold=0.0, alignment_threshold=0.0):
    return FrozenTargets(prompt_id=0, seed=0, step=0, layer=0, n_img=int(clean.shape[0]),
                         register_ids=tuple(ids), topk_ids=tuple(ids),
                         random_ids=tuple(ids), clean_states=clean.clone(),
                         norm_threshold=norm_threshold,
                         alignment_threshold=alignment_threshold)


def _context(layer=0, step=0):
    return EditContext(layer=layer, step=step, point=InterventionPoint.BLOCK_INPUT,
                       targets=_targets(torch.zeros(2, 2), ()))


def test_the_rescue_target_comes_from_the_clean_trajectory_not_the_ablated_one():
    """The single most corruptible step in the whole design.

    ``alpha`` measured after the ablation is exactly the quantity the ablation destroyed.
    A rescue that reused it would restore nothing while looking like a rescue, and the
    mediation result would be an artefact. So the target is asserted against the clean
    value, and the ablated state is deliberately given a very different projection.
    """
    torch.manual_seed(0)
    channel = 5
    clean = torch.randn(8, 32)
    clean[:, channel] += 40.0                      # the dominant channel carries the state
    v = torch.zeros(32); v[channel] = 1.0; v[7] = 0.3
    unit = CS._unit(v)
    ids = [0, 1, 2]
    condition = CS.Condition("r", "r", "rescue", ablate_channel=True, rescue="vstar")
    edit = CS.make_edit(condition, direction=unit, token_ids=ids, dominant_channel=channel,
                        clean_states={(0, 0): clean}, seed=0)
    out = edit(clean.clone(), _context())
    index = torch.as_tensor(ids)
    restored = out[index].float() @ unit
    wanted = clean[index].float() @ unit
    assert torch.allclose(restored, wanted, atol=1e-3, rtol=1e-4), (restored, wanted)
    untouched = [t for t in range(8) if t not in ids]
    assert torch.allclose(out[untouched], clean[untouched]), "a rescue must not touch others"


def test_the_ablation_runs_before_the_rescue_inside_one_edit():
    """Hook ordering is not left to PyTorch's registration order.

    ``channel_restore_only`` ablates the coordinate and then puts it back. If the two
    steps ran the other way round the ablation would wipe the restoration, and the
    condition would be indistinguishable from a bare ablation. Composing them in one
    edit makes the order a property of the code rather than of the install sequence.
    """
    channel = 3
    clean = torch.randn(6, 16)
    clean[:, channel] += 25.0
    v = torch.zeros(16); v[channel] = 1.0
    ids = [0, 1]
    restore = CS.make_edit(
        CS.Condition("c", "c", "rescue", ablate_channel=True, rescue="channel"),
        direction=v, token_ids=ids, dominant_channel=channel,
        clean_states={(0, 0): clean}, seed=0)(clean.clone(), _context())
    ablate = CS.make_edit(
        CS.Condition("a", "a", "rescue", ablate_channel=True),
        direction=v, token_ids=ids, dominant_channel=channel,
        clean_states={(0, 0): clean}, seed=0)(clean.clone(), _context())
    assert torch.allclose(restore[ids, channel], clean[ids, channel], atol=1e-4)
    assert float(ablate[ids, channel].abs().max()) == 0.0


def test_the_orthogonal_rescue_matches_the_vstar_rescue_in_energy_and_is_orthogonal():
    """Equal size, different direction, so a difference cannot be about size."""
    torch.manual_seed(0)
    channel = 2
    clean = torch.randn(8, 24)
    clean[:, channel] += 30.0
    v = torch.zeros(24); v[channel] = 1.0; v[5] = 0.4
    unit = CS._unit(v)
    ids = [0, 1, 2, 3]
    index = torch.as_tensor(ids)
    made = {}
    for key, rescue in (("vstar", "vstar"), ("orthogonal", "orthogonal")):
        made[key] = CS.make_edit(
            CS.Condition(key, key, "rescue", ablate_channel=True, rescue=rescue),
            direction=unit, token_ids=ids, dominant_channel=channel,
            clean_states={(0, 0): clean}, seed=4)(clean.clone(), _context())
    ablated = CS.make_edit(
        CS.Condition("a", "a", "rescue", ablate_channel=True), direction=unit,
        token_ids=ids, dominant_channel=channel, clean_states={(0, 0): clean},
        seed=4)(clean.clone(), _context())
    energy = {k: (made[k][index] - ablated[index]).norm(dim=-1) for k in made}
    assert torch.allclose(energy["vstar"], energy["orthogonal"], atol=1e-3, rtol=1e-3)
    injected = made["orthogonal"][index] - ablated[index]
    assert float((injected.float() @ unit).abs().max()) < 1e-3 * float(injected.norm(dim=-1).max())


# =============================================================== the population
def test_every_selection_mode_reads_only_the_clean_pass(states):
    x, v = states
    targets = _targets(x, [0, 1, 2], norm_threshold=float(x.norm(dim=-1).median()),
                       alignment_threshold=0.1)
    for mode in CS.SELECTION_MODES:
        treatment = CS.select_treatment(targets, mode=mode, direction=v, topk=3)
        assert treatment.mode == mode
        assert treatment.rule, f"{mode} must say how it chose"
        assert all(0 <= t < x.shape[0] for t in treatment.token_ids)


def test_the_primary_population_is_the_intersection_of_both_bars(states):
    """Not "loud", not "aligned", but both, which is the target population."""
    x, v = states
    unit = CS._unit(v)
    projection = x @ unit
    cosine = projection / x.norm(dim=-1)
    norm_bar = float(x.norm(dim=-1).quantile(0.5))
    cosine_bar = float(cosine.abs().quantile(0.5))
    treatment = CS.select_treatment(_targets(x, [0], norm_threshold=norm_bar,
                                             alignment_threshold=cosine_bar),
                                    mode="highnorm_and_aligned", direction=v)
    expected = {int(t) for t in range(x.shape[0])
                if float(x[t].norm()) >= norm_bar and abs(float(cosine[t])) >= cosine_bar}
    assert set(treatment.token_ids) == expected
    assert "FALLBACK" not in treatment.rule


def test_an_empty_intersection_falls_back_loudly_rather_than_silently(states):
    """A run of no-ops is worse than a named compromise, so the fallback is in the rule."""
    x, v = states
    treatment = CS.select_treatment(_targets(x, [3], norm_threshold=1e9,
                                             alignment_threshold=1.01),
                                    mode="highnorm_and_aligned", direction=v)
    assert treatment.token_ids == (3,)
    assert "FALLBACK" in treatment.rule


def test_an_unknown_selection_mode_is_refused(states):
    x, v = states
    with pytest.raises(KeyError):
        CS.select_treatment(_targets(x, [0]), mode="whatever_looks_best", direction=v)


def test_sinkhood_uses_the_projects_own_threshold():
    """No new sink definition: the criterion is the uniform-share multiple in SweepConfig."""
    class Observation:
        incoming = torch.tensor([[0.90, 0.05, 0.05], [0.10, 0.80, 0.10]])
        qk_cosine = torch.tensor([[0.9, 0.1, 0.0], [0.2, 0.7, 0.1]])
    readout = CS.sink_readout(Observation(), n_tokens=3, threshold=2.0)
    # Uniform share is 1/3, so token 0 at 0.90 is 2.7x and token 2 at 0.10 is 0.3x.
    assert bool(readout["is_sink"][0]) and bool(readout["is_sink"][1])
    assert not bool(readout["is_sink"][2])
    assert readout["n_sink_heads"].tolist() == [1.0, 1.0, 0.0]
    assert int(readout["qk_rank"][0]) == 1


# ============================================================== image detail
def test_the_frequency_bands_are_calibrated_against_unstructured_noise():
    """The fractions are only readable beside their own uniform expectation."""
    random = np.random.RandomState(0)
    base = random.rand(64, 64).astype(np.float32) * 255
    bands = ID.spectrum_bands(base, base + random.randn(64, 64).astype(np.float32))
    for name, _, _ in ID.BANDS:
        assert 0.9 < bands[f"spectrum_{name}_over_uniform"] < 1.1, name


def test_a_low_frequency_difference_lands_in_the_low_band():
    random = np.random.RandomState(0)
    base = random.rand(64, 64).astype(np.float32) * 255
    _, columns = np.mgrid[0:64, 0:64]
    smooth = base + 8.0 * np.sin(2 * np.pi * columns / 64).astype(np.float32)
    bands = ID.spectrum_bands(base, smooth)
    assert bands["spectrum_low_fraction"] > 0.95
    assert bands["spectrum_high_over_low"] < 0.01


def test_a_difference_confined_to_edges_concentrates_on_clean_structure():
    """The readout that answers "contours, or composition?", checked against both.

    The base is textured rather than a bare step: a generated image has structure
    everywhere, and a synthetic image that is flat over most of the frame would exercise
    the tie-breaking in the mask instead of the quantity being tested.
    """
    random = np.random.RandomState(0)
    rows, columns = np.mgrid[0:64, 0:64]
    base = (90.0 + 40.0 * np.sin(2 * np.pi * columns / 21.0)
            + 6.0 * random.randn(64, 64)).astype(np.float32)
    base[:, 32:] += 90.0                          # plus one strong vertical contour
    gradient = ID.sobel_magnitude(base)
    mask = gradient >= np.quantile(gradient, 0.8)
    edges = base + 20.0 * mask.astype(np.float32)
    spread = base + 20.0 * np.ones_like(base)
    on_edges = ID.high_gradient_concentration(base, edges)
    everywhere = ID.high_gradient_concentration(base, spread)
    assert on_edges["detail_mask_fraction"] == pytest.approx(0.2, abs=0.01)
    assert on_edges["concentration_ratio"] > 2.0, on_edges
    assert everywhere["concentration_ratio"] < 1.5, everywhere


def test_an_identical_pair_reports_no_effect_rather_than_a_perfect_score():
    random = np.random.RandomState(1)
    base = random.rand(32, 32).astype(np.float32) * 255
    row = ID.detail_metrics(base, base)
    assert math.isnan(row["spectrum_high_fraction"])
    assert row["gradient_rmse"] == 0.0


def test_the_amplified_difference_is_centred_on_mid_grey():
    base = np.full((16, 16), 100.0, dtype=np.float32)
    array = ID.amplified_difference(base, base, factor=10.0)
    assert array is not None and int(array.min()) == 128 and int(array.max()) == 128


# ================================================================= end to end
@pytest.fixture(scope="module")
def ctx(tmp_path_factory):
    cfg = SweepConfig(model="tiny-flux1", prompts=["a"], seeds=[0], height=64, width=64,
                      num_inference_steps=4, capture_steps=[2], dtype="float32",
                      output_dir=str(tmp_path_factory.mktemp("q13")), save_images=False)
    driver = GenerationDriver(cfg)
    artifact = DiscoveryArtifact(
        schema_version=1, checkpoint="tiny-flux1", repo_id="synthetic", vstar_file="vstar.pt",
        fitting_population=[], explained_variance=0.9, axis_convention="", sign_convention="",
        dominant_register_channel=ChannelChoice(3, 3, True, 1.0),
        late_growing_competitor=ChannelChoice(5, None, None, 1.0),
        massive_unspecific_control=ChannelChoice(7, None, None, 1.0),
        layer_ranges=LayerRanges((0, 1), (1, 3), (3, 4)), denoising_steps=[2],
        discovery_prompts=["z"], discovery_seeds=[9], confirmation_prompts=["a"],
        confirmation_seeds=[0], code_config_fingerprint="t", config_sha256="t")
    return QuestionContext.from_artifact(cfg, artifact, planted_direction(driver.bundle.d_model),
                                         driver=driver, progress=False, topk=3, percentile=90.0)


@pytest.fixture(scope="module")
def run(ctx, tmp_path_factory):
    config = CS.ControlSurfaceConfig(
        experiment_root=tmp_path_factory.mktemp("q13-run"), betas=(0.0, 1.0, 2.0),
        gammas=(0.5, 1.0, 1.5), layer_scopes=("write",), save_images=False,
        save_difference_maps=False, run_controls=True)
    return CS.run_control_surface(ctx, config), config


def test_the_run_produces_every_table_the_plots_need(run):
    result, _ = run
    for name in ("token_metrics", "population_metrics", "attention_metrics",
                 "image_metrics", "treatments", "edit_diary"):
        frame = getattr(result, name)
        assert not frame.empty, f"{name} is empty"
    for column in ("model", "prompt_id", "prompt", "seed", "layer_scope", "beta", "gamma",
                   "intervention"):
        assert column in result.image_metrics.columns, column
    assert (result.root / "config.json").exists()
    assert (result.root / "metrics" / "token_metrics.csv").exists()


def test_the_grid_and_the_rescue_and_the_controls_all_ran(run):
    result, config = run
    arms = set(result.image_metrics["arm"])
    assert {"grid", "rescue", "controls"} <= arms, arms
    cells = result.image_metrics[result.image_metrics["arm"] == "grid"]
    assert len(cells) == len(config.betas) * len(config.gammas)
    assert {"channel_ablate", "channel_ablate_vstar_rescue",
            "channel_ablate_orthogonal_rescue", "channel_restore_only"} <= set(
                result.image_metrics["intervention"])


def test_beta_cannot_change_a_norm_threshold_count_but_gamma_can(run):
    """The operators' contract, observed through the whole model rather than in isolation.

    ``beta`` renormalises every treated token to its original length, so no count
    defined by a norm threshold can move along that axis. ``gamma`` scales the norm, so
    it must. This is the end-to-end version of the two unit invariants above, and it
    would catch a leak that only appears once the edit goes through a real forward pass.
    """
    _, _ = run[0], run[1]
    population = run[0].population_metrics
    grid = population[(population["arm"] == "grid")
                      & population["is_edited_layer"].fillna(False).astype(bool)]
    assert not grid.empty
    along_beta = grid[np.isclose(grid["gamma"], 1.0)].groupby("beta")["n_highnorm"].mean()
    along_gamma = grid[np.isclose(grid["beta"], 1.0)].groupby("gamma")["n_highnorm"].mean()
    assert float(along_beta.max() - along_beta.min()) == 0.0, along_beta.to_dict()
    assert float(along_gamma.max() - along_gamma.min()) > 0.0, along_gamma.to_dict()


def test_nothing_upstream_of_the_edit_moves(run):
    """The placebo gate. Any spread here is a leak, not a finding."""
    population = run[0].population_metrics
    upstream = population[~population["is_downstream"].fillna(True).astype(bool)]
    if upstream.empty:
        pytest.skip("no observed layer sits upstream of the edit in this configuration")
    spread = upstream.groupby(["step", "layer"], observed=True)["selected_alpha_mean"].agg(
        lambda values: float(values.max() - values.min()))
    assert float(spread.max()) < 1e-5, spread.to_dict()


def test_the_vstar_rescue_restores_the_projection_and_the_orthogonal_control_does_not(run):
    """The mediation contrast, measured through a full generation.

    Both conditions ablate the channel and then inject the same per-token L2; only the
    direction differs. If the orthogonal control recovered too, the result would be about
    perturbation size rather than about the direction.
    """
    population = run[0].population_metrics
    rescue = population[(population["arm"] == "rescue")
                        & population["is_downstream"].fillna(True).astype(bool)]
    by = rescue.groupby("condition")["selected_alpha_recovery"].mean()
    assert by["clean"] == pytest.approx(1.0, abs=1e-3)
    assert by["channel_ablate_vstar_rescue"] == pytest.approx(1.0, abs=0.1), by.to_dict()
    assert by["channel_ablate"] < 0.5, by.to_dict()
    assert by["channel_ablate_orthogonal_rescue"] < 0.5, by.to_dict()


def test_the_edit_diary_records_the_perturbation_energy_of_every_condition(run):
    """Requirement: report the actual L2 for every condition, not a nominal setting."""
    diary = run[0].edit_diary
    assert {"perturbation_l2", "perturbation_energy", "layer", "step"} <= set(diary.columns)
    assert diary["perturbation_l2"].notna().all()
    matched = diary[diary.get("energy_matched_to").notna()] if "energy_matched_to" \
        in diary.columns else pd.DataFrame()
    if not matched.empty:
        assert np.allclose(matched["perturbation_l2"], matched["match_target_l2"],
                           rtol=1e-3, atol=1e-4)


def test_a_resumed_run_regenerates_nothing_and_returns_the_same_numbers(run, ctx):
    """An interrupted Colab session is the normal case, not the exception."""
    result, config = run
    again = CS.run_control_surface(ctx, config)
    for name in ("token_metrics", "population_metrics", "attention_metrics", "image_metrics"):
        first, second = getattr(result, name), getattr(again, name)
        assert len(first) == len(second), name
    order = ["layer_scope", "intervention", "prompt_id", "seed"]
    a = result.image_metrics.sort_values(order).reset_index(drop=True)
    b = again.image_metrics.sort_values(order).reset_index(drop=True)
    numeric = [c for c in a.columns if c in b.columns and a[c].dtype.kind in "fi"]
    assert a[numeric].equals(b[numeric])


def test_the_verdict_reports_the_contrasts_without_choosing_a_hypothesis(run):
    verdict = run[0].verdict
    assert verdict.startswith("Q13 on ")
    assert "alignment axis spans" in verdict
    assert "Placebo gate" in verdict
    assert not re.search(r"(?<![A-Za-z])[-+]?nan(?![A-Za-z])", verdict, flags=re.IGNORECASE), verdict
    # The runner must not pick a winner: naming the hypotheses is fine, asserting one is not.
    assert "H1" in verdict and "judgement for the writeup" in verdict
    for forbidden in ("proves", "confirms that", "therefore v* is"):
        assert forbidden not in verdict.lower()


def test_every_q13_figure_draws(run, tmp_path):
    result, _ = run
    grid = result.population_metrics[result.population_metrics["arm"] == "grid"]
    figures = [
        CF.fig_q13_control_surface(grid, result.image_metrics, checkpoint="tiny-flux1",
                                   path=tmp_path / "surface.pdf"),
        CF.fig_q13_separation(result.token_metrics, checkpoint="tiny-flux1"),
        CF.fig_q13_rescue(result.population_metrics, result.image_metrics,
                          checkpoint="tiny-flux1"),
    ]
    for figure in figures:
        assert getattr(figure, "dv_caption", ""), "every figure carries its own caption"
    assert (tmp_path / "surface.pdf").exists()


def test_the_control_surface_figure_marks_the_clean_cell_and_names_both_axes(run):
    result, _ = run
    grid = result.population_metrics[result.population_metrics["arm"] == "grid"]
    caption = CF.fig_q13_control_surface(grid, result.image_metrics).dv_caption
    assert "fixed norm" in caption and "fixed direction" in caption
    assert "numerical floor" in caption.lower()


def test_the_contact_sheet_declines_rather_than_inventing_panels(run):
    """A checkpoint with no decoder has no images; a contact sheet of blanks is worse
    than an explicit refusal the notebook can report."""
    result, _ = run
    with pytest.raises(ValueError, match="contact sheet"):
        CF.fig_q13_contact_sheet(result.image_metrics, result.root)


# ==================================== the four caveat fixes, as regression guards
def test_the_aligned_population_count_is_the_one_beta_can_move():
    """``n_highnorm`` is pinned along the beta axis; this is the count that is not.

    beta renormalises every treated token, so a count defined by a norm threshold alone
    cannot move along that axis, which makes it a positive control, not an endpoint.
    The population the project actually cares about is loud AND aligned, and rotating a
    token away from v* does remove it from that set at fixed norm.
    """
    torch.manual_seed(0)
    v = torch.randn(48)
    unit = CS._unit(v)
    x = torch.randn(12, 48) * 0.4
    x[:5] = 9.0 * unit + 0.2 * torch.randn(5, 48)     # loud and aligned
    norm_bar = float(x.norm(dim=-1).median())
    cosine_bar = 0.5

    def counts(states):
        norms = states.norm(dim=-1)
        cosine = (states @ unit) / norms.clamp_min(1e-9)
        loud = norms >= norm_bar
        return int(loud.sum()), int((loud & (cosine.abs() >= cosine_bar)).sum())

    rotated, _ = CS.align_scale(x, v, 0.0)
    assert counts(rotated)[0] == counts(x)[0], "beta must not move the norm-only count"
    assert counts(rotated)[1] < counts(x)[1], "but it must move the loud-AND-aligned count"


def test_the_channel_axis_control_exists_and_rescales_the_channel_not_vstar():
    """The comparison that decides whether the beta axis tests geometry or one channel.

    Where v* is nearly axis-aligned, beta along v* and beta along e_{c*} are almost the
    same operation. The difference between them is exactly the part of v* that is not the
    dominant channel, so the control has to actually use the coordinate axis.
    """
    keys = {c.key for c in CS.control_conditions((0.0, 1.0, 2.0))}
    assert {"channel_axis_beta0", "channel_axis_beta2"} <= keys
    channel = 4
    torch.manual_seed(0)
    x = torch.randn(6, 32)
    x[:, channel] += 20.0
    v = CS._unit(torch.randn(32))
    condition = next(c for c in CS.control_conditions((0.0, 2.0))
                     if c.key == "channel_axis_beta0")
    assert condition.edit_direction == "channel"
    assert condition.match_energy_to == "beta0_gamma1"
    out = CS.make_edit(condition, direction=v, token_ids=[0, 1], dominant_channel=channel,
                       clean_states={}, seed=0)(x.clone(), _context())
    # beta=0 along e_{c*} removes that coordinate from the treated tokens, and the
    # energy match then rescales the whole perturbation, so the coordinate shrinks
    # sharply rather than vanishing exactly, and the untreated tokens never move.
    assert abs(float(out[0, channel])) < abs(float(x[0, channel]))
    assert torch.allclose(out[2:], x[2:]), "a control must not touch untreated tokens"


def test_the_cfg_diagnostic_is_small_and_pairs_with_its_conditional_only_twin():
    """Four named twins, not a second 5x5 grid, and each one knows its partner."""
    twins = CS.cfg_conditions((0.0, 0.5, 1.0, 1.5, 2.0), (0.5, 0.75, 1.0, 1.25, 1.5))
    assert len(twins) == 4, [c.key for c in twins]
    assert all(c.all_batch_rows for c in twins), "the whole point is the other branch"
    assert all(c.arm == "cfg" for c in twins)
    partners = {c.key: CS.cfg_partner(c.key) for c in twins}
    assert partners["allrows_beta0_gamma1"] == "beta0_gamma1"
    assert partners["allrows_channel_ablate"] == "channel_ablate"
    # The grid and rescue arms must actually contain those partners, or the join is empty.
    available = ({c.key for c in CS.grid_conditions((0.0, 0.5, 1.0, 1.5, 2.0),
                                                    (0.5, 0.75, 1.0, 1.25, 1.5))}
                 | {c.key for c in CS.RESCUE_CONDITIONS})
    assert set(partners.values()) <= available, set(partners.values()) - available


def test_the_cfg_diagnostic_is_skipped_where_it_would_be_a_no_op(run):
    """On a batch-1 pipeline, editing 'every row' is the same tensor operation.

    FLUX passes guidance as an embedding and runs the transformer at batch 1, so the arm
    would spend generations to reproduce the conditional-only numbers exactly. It must be
    skipped rather than run, and tiny-flux1 stands in for that case.
    """
    result, _ = run
    assert 'cfg' not in set(result.image_metrics['arm'])
    assert not result.image_metrics['cfg_batched'].any()
    assert set(result.image_metrics['edited_batch_rows']) == {'conditional'}


def test_image_distances_are_expressed_against_the_centre_cell_floor():
    """A raw distance is not interpretable without the (1, 1) cell beside it.

    The centre cell runs the full edit, so its distance is the model's sensitivity to a
    round trip rather than an effect. The floor is per prompt-seed-scope, because it is a
    property of the trajectory and not of the operator.
    """
    frame = pd.DataFrame([
        dict(arm='grid', beta=1.0, gamma=1.0, prompt_id=0, seed=0, layer_scope='write',
             rmse=2.0, lpips=0.01),
        dict(arm='grid', beta=0.0, gamma=1.0, prompt_id=0, seed=0, layer_scope='write',
             rmse=10.0, lpips=0.05),
        dict(arm='grid', beta=1.0, gamma=1.0, prompt_id=1, seed=0, layer_scope='write',
             rmse=4.0, lpips=0.02),
        dict(arm='grid', beta=0.0, gamma=1.0, prompt_id=1, seed=0, layer_scope='write',
             rmse=8.0, lpips=0.02),
    ])
    out = CS._add_floor_columns(frame)
    assert out['rmse_over_floor'].tolist() == [1.0, 5.0, 1.0, 2.0]
    assert out['lpips_over_floor'].tolist() == [1.0, 5.0, 1.0, 1.0]
    # An empty or floorless frame must degrade rather than raise.
    assert 'rmse_over_floor' not in CS._add_floor_columns(pd.DataFrame()).columns


def test_the_diary_indexes_its_batch_row_so_cfg_twins_are_not_double_counted():
    """``all_batch_rows`` runs the edit once per row, and the diary would count both.

    The conditional row is always first, so an index is enough to let a summary filter to
    it. Without one, the perturbation L2 reported for every CFG twin would be the mean of
    two identical notes, right by luck for a mean, wrong for a count or a sum, and
    silently wrong either way.
    """
    torch.manual_seed(0)
    x = torch.randn(6, 24)
    v = CS._unit(torch.randn(24))
    diary = []
    edit = CS.make_edit(CS.Condition("t", "t", "cfg", beta=0.0, all_batch_rows=True),
                        direction=v, token_ids=[0, 1], dominant_channel=2,
                        clean_states={}, seed=0, diary=diary)
    context = _context()
    edit(x.clone(), context)            # the conditional row
    edit(x.clone(), context)            # the unconditional row, same (step, layer)
    assert [note["batch_row_call"] for note in diary] == [0, 1]
    assert [note["is_conditional_row"] for note in diary] == [True, False]
    conditional = [n for n in diary if n["is_conditional_row"]]
    assert len(conditional) == 1


def test_store_keys_is_capped_so_a_stray_flag_cannot_exhaust_the_device():
    """The full attention tensors are ~312 MB per layer-step at 512px.

    Applied to a 20-layer register window that is 6.6 GB, on top of FLUX's own 24 GB. The
    flag stays available because a key norm needs it, but the number of layers it reaches
    is capped by a separate, explicit knob.
    """
    config = CS.ControlSurfaceConfig(experiment_root=Path("/tmp/unused"), store_keys=True)
    assert config.store_keys_layers == 1
    wide = CS.ControlSurfaceConfig(experiment_root=Path("/tmp/unused"), store_keys=True,
                                  store_keys_layers=4)
    assert wide.store_keys_layers == 4


def test_a_resumed_run_refuses_cells_from_a_different_configuration(ctx, tmp_path):
    """The hazard that could quietly corrupt a final run.

    The cache key identifies a *cell*, scope, condition, prompt, seed, not an
    *experiment*. Widening the grid, changing the treatment rule or pointing at another
    channel produces the same cell ids with different meanings, and a resumed run would
    mix them without a word. Cells carry a fingerprint of everything that could change a
    number, and a mismatch means recompute.
    """
    first = CS.ControlSurfaceConfig(
        experiment_root=tmp_path / "exp", betas=(0.0, 1.0), gammas=(1.0,),
        layer_scopes=("write",), save_images=False, save_difference_maps=False,
        run_controls=False, run_rescue=False)
    original = CS.run_control_surface(ctx, first)
    cached = sorted((tmp_path / "exp" / "cells").glob("*.json"))
    assert cached, "the first run must have written a cache"
    assert all(json.loads(p.read_text())["fingerprint"] == original.meta["fingerprint"]
               for p in cached)
    assert original.meta["stale_cells_discarded"] == 0

    # Same cell ids, different treatment rule -> every cached cell must be discarded.
    changed = CS.ControlSurfaceConfig(
        experiment_root=tmp_path / "exp", betas=(0.0, 1.0), gammas=(1.0,),
        layer_scopes=("write",), save_images=False, save_difference_maps=False,
        run_controls=False, run_rescue=False, selection_mode="topk_norm")
    second = CS.run_control_surface(ctx, changed)
    assert second.meta["fingerprint"] != original.meta["fingerprint"]
    assert second.meta["stale_cells_discarded"] == len(cached), second.meta

    # And a genuine resume under the same configuration still reuses everything.
    third = CS.run_control_surface(ctx, changed)
    assert third.meta["stale_cells_discarded"] == 0
    assert len(third.image_metrics) == len(second.image_metrics)


def test_the_fingerprint_ignores_what_cannot_change_a_number(ctx, tmp_path):
    """Output paths and image-saving flags must not invalidate a cache.

    A fingerprint that changed with the amplification factor would throw away hours of
    GPU time for a cosmetic setting, which would push callers toward resume=False and
    lose the protection entirely.
    """
    base = dict(betas=(0.0, 1.0), gammas=(1.0,), layer_scopes=("write",))
    a = CS.ControlSurfaceConfig(experiment_root=tmp_path / "a", amplification=10.0,
                                save_images=True, **base)
    b = CS.ControlSurfaceConfig(experiment_root=tmp_path / "b", amplification=25.0,
                                save_images=False, save_difference_maps=False, **base)
    assert CS.experiment_fingerprint(ctx, a) == CS.experiment_fingerprint(ctx, b)
    # But the grid itself must.
    c = CS.ControlSurfaceConfig(experiment_root=tmp_path / "c", betas=(0.0, 1.0, 2.0),
                                gammas=(1.0,), layer_scopes=("write",))
    assert CS.experiment_fingerprint(ctx, c) != CS.experiment_fingerprint(ctx, a)


def test_the_fingerprint_changes_with_vstar_itself(ctx, tmp_path):
    """A different direction is a different experiment, even at identical settings."""
    import dataclasses

    config = CS.ControlSurfaceConfig(experiment_root=tmp_path / "v", betas=(0.0, 1.0),
                                     gammas=(1.0,), layer_scopes=("write",))
    other = dataclasses.replace(ctx, vstar=torch.randn(int(ctx.vstar.numel())))
    assert CS.experiment_fingerprint(ctx, config) != CS.experiment_fingerprint(other, config)


def test_the_cfg_twins_are_identical_internally_and_differ_only_through_the_sampler():
    """Why the CFG arm is an image-level comparison, asserted rather than assumed.

    The tracer reads the conditional row, and batch rows do not mix inside the
    transformer, each is an independent element of one forward pass. So a twin and its
    conditional-only partner produce *identical* internal readouts whatever the guidance
    strength; they diverge only in the sampler, where ``u + s(t - u)`` combines the
    branches. This test pins that, because a reader who compared population metrics
    between the twins would find no difference and conclude the branch choice is
    irrelevant, which is the opposite of the truth.
    """
    torch.manual_seed(0)
    x = torch.randn(8, 32)
    v = CS._unit(torch.randn(32))
    condition = CS.Condition("t", "t", "grid", beta=0.0)
    twin = CS.Condition("t", "t", "cfg", beta=0.0, all_batch_rows=True)
    shared = dict(direction=v, token_ids=[0, 1, 2], dominant_channel=4,
                  clean_states={}, seed=0)
    one = CS.make_edit(condition, **shared)(x.clone(), _context())
    both = CS.make_edit(twin, **shared)(x.clone(), _context())
    # The edit applied to a single row is the same edit either way.
    assert torch.allclose(one, both, atol=0.0)
    assert condition.all_batch_rows is False and twin.all_batch_rows is True
    # And the docstring must say where the difference actually lives, so the arm is not
    # read on the wrong table.
    doc = CS.cfg_conditions.__doc__ or ""
    assert "images only" in doc and "sampler" in doc


def test_the_cfg_arm_does_not_spend_generations_on_a_no_op_twin():
    """A gamma twin at gamma=1 twins a cell that does nothing. Two wasted generations."""
    with_quiet = CS.cfg_conditions((0.0, 1.0, 2.0), (0.5, 1.0, 1.5))
    without = CS.cfg_conditions((0.0, 1.0, 2.0), (1.0,))
    assert "allrows_beta1_gamma0.5" in {c.key for c in with_quiet}
    assert not any(c.gamma == 1.0 and c.beta == 1.0 and not c.ablate_channel
                   for c in without), [c.key for c in without]


def test_the_rescue_arm_uses_the_projects_clustered_bootstrap_not_a_bare_sem(run):
    """Prompt-to-prompt variation is large and uninteresting; pair it out, then cluster.

    ``causal_stats.paired_effect`` pairs within unit so that variation cancels, and
    clusters by prompt so two seeds of one prompt are not two independent observations.
    A standard error across units does neither.
    """
    result, _ = run
    effects = result.rescue_effects
    assert not effects.empty
    assert {"condition", "reference", "metric", "effect", "ci_low", "ci_high",
            "n_units", "n_clusters", "has_interval", "excludes_zero"} <= set(effects.columns)
    # The reference is the ablation, because the question is how much damage was undone.
    assert set(effects["reference"]) == {"channel_ablate"}
    assert "channel_ablate" not in set(effects["condition"])
    assert {"channel_ablate_vstar_rescue", "channel_ablate_orthogonal_rescue",
            "channel_restore_only"} <= set(effects["condition"])


def test_an_underpowered_run_cannot_buy_a_confidence_interval(run):
    """The refusal must propagate, and the verdict must say so in words.

    ``bootstrap_mean`` declines below three clusters because a percentile bootstrap over
    two can only draw three distinct resamples. Papering over that with a SEM would let
    a two-prompt run print error bars it has not earned.
    """
    result, _ = run
    clusters = int(result.meta["n_prompt_clusters"])
    if clusters >= 3:
        assert result.meta["clustered_intervals_available"]
        assert "prompt-clustered bootstrap intervals" in result.verdict
    else:
        assert not result.meta["clustered_intervals_available"]
        assert not result.rescue_effects["has_interval"].any()
        assert "requires three" in result.verdict
        assert "not enough to quote" in result.verdict


def test_three_prompts_clear_the_bootstrap_floor(ctx, tmp_path):
    """The floor is a floor: at three clusters the bootstrap runs rather than declining.

    It still reports no usable interval here, and correctly so, the synthetic
    checkpoint's forward pass ignores the prompt text, so three prompts at one seed give
    three byte-identical trajectories and a degenerate resampling distribution. That is
    a fact about the stand-in model, not about the statistics, and the interval machinery
    is exercised on varying data in the test below.
    """
    import dataclasses

    wider = dataclasses.replace(ctx, prompts=["a", "b", "c"], seeds=[0])
    config = CS.ControlSurfaceConfig(
        experiment_root=tmp_path / "wide", betas=(1.0,), gammas=(1.0,),
        layer_scopes=("write",), run_grid=False, run_controls=False,
        save_images=False, save_difference_maps=False)
    result = CS.run_control_surface(wider, config)
    assert int(result.meta["n_prompt_clusters"]) == 3
    assert not result.rescue_effects.empty
    assert (result.rescue_effects["n_clusters"] == 3).all()


def test_the_clustered_interval_appears_once_the_prompts_actually_differ():
    """The statistics path itself, on data with between-prompt variation.

    Paired within unit against the ablation and clustered by prompt, so a condition whose
    every unit moves the same way gets an interval that excludes zero, and one that moves
    inconsistently does not.
    """
    rows = []
    for prompt in range(5):
        for condition, shift in (("channel_ablate", 0.0),
                                 ("channel_ablate_vstar_rescue", 0.9),
                                 ("channel_ablate_orthogonal_rescue", 0.02)):
            rows.append(dict(arm="rescue", condition=condition, prompt_id=prompt, seed=0,
                             is_downstream=True,
                             selected_alpha_recovery=0.2 + 0.05 * prompt + shift))
    effects = CS.rescue_effects(pd.DataFrame(rows), iterations=400)
    assert not effects.empty
    by = effects.set_index(["condition", "metric"])
    rescue = by.loc[("channel_ablate_vstar_rescue", "selected_alpha_recovery")]
    control = by.loc[("channel_ablate_orthogonal_rescue", "selected_alpha_recovery")]
    assert rescue["n_clusters"] == 5 and bool(rescue["has_interval"])
    assert rescue["effect"] == pytest.approx(0.9, abs=1e-6)
    assert bool(rescue["excludes_zero"]), rescue.to_dict()
    # The matched control moves far less, and the two intervals must not overlap.
    assert control["effect"] < rescue["effect"] / 10
    assert control["ci_high"] < rescue["ci_low"]


# ============ cache invalidation across analysis-code changes
def test_the_cache_fingerprint_covers_the_analysis_code():
    """The cache key must change when the analysis code that produces a column changes.

    Adding a diagnostic to the edit hook leaves every setting identical and every number
    different. Keyed on config alone, the cache would serve the old rows without the new
    columns, and a validation cell that requires them would fail while the code that emits
    them sits in the repo. Hashed by source text rather than git SHA, so a commit touching
    only a notebook does not discard GPU hours.
    """
    import importlib

    path = Path(CS.__file__)
    original = path.read_bytes()
    first = CS._analysis_code_digest()
    try:
        path.write_bytes(original + b"\n# a change to how a number is computed\n")
        importlib.reload(CS)
        assert CS._analysis_code_digest() != first
    finally:
        path.write_bytes(original)
        importlib.reload(CS)
    assert CS._analysis_code_digest() == first


def test_the_orthogonal_control_is_recorded_as_a_displacement_not_a_matched_null():
    r"""Matching injection L2 matches effort, not the resulting distance from clean.

    The rescue spends its budget moving back toward the clean state; an orthogonal step of
    the same length spends it moving sideways and lands at :math:`\sqrt 2` times the gap --
    *further* from clean than the ablation it controls for. Without this recorded, a
    downstream "recovery" from that condition reads as evidence against direction
    specificity when it is the response to a larger perturbation. Measured on a population
    where v* is nearly the dominant channel, which is the real FLUX geometry.
    """
    torch.manual_seed(0)
    channel, width, k = 5, 64, 6
    clean = torch.randn(12, width)
    clean[:, channel] += 300.0
    v = torch.zeros(width)
    v[channel] = 1.0
    v[9] = 0.05
    unit = CS._unit(v)
    ids = list(range(k))

    def run(key, **kwargs):
        diary = []
        CS.make_edit(CS.Condition(key, key, "rescue", **kwargs), direction=unit,
                     token_ids=ids, dominant_channel=channel,
                     clean_states={(0, 0): clean}, seed=0,
                     diary=diary)(clean.clone(), _context())
        return diary[0]

    ablate = run("channel_ablate", ablate_channel=True)
    rescue = run("vstar", ablate_channel=True, rescue="vstar")
    orthogonal = run("orthogonal", ablate_channel=True, rescue="orthogonal")

    assert ablate["rescue_closed_distance_ratio"] == pytest.approx(1.0, abs=1e-9)
    assert rescue["rescue_closed_distance_ratio"] < 0.2, rescue
    assert orthogonal["rescue_closed_distance_ratio"] == pytest.approx(2 ** 0.5, rel=0.05)
    # Same effort, opposite consequence for the distance from clean.
    assert rescue["rescue_injection_l2"] == pytest.approx(
        orthogonal["rescue_injection_l2"], rel=1e-6)
    assert (orthogonal["distance_from_clean_after_rescue_l2"]
            > ablate["distance_from_clean_after_ablation_l2"])


def test_the_distance_ratio_is_only_recorded_where_there_is_damage_to_undo():
    """A grid cell never ablates, so its "ratio" would divide by the incoming state.

    At the first edited layer that incoming state *is* the clean one, so the denominator is
    zero and the quotient would be a division by nothing dressed up as a finding.
    """
    torch.manual_seed(0)
    clean = torch.randn(8, 32)
    v = CS._unit(torch.randn(32))
    diary = []
    CS.make_edit(CS.Condition("g", "g", "grid", beta=0.0), direction=v, token_ids=[0, 1],
                 dominant_channel=3, clean_states={(0, 0): clean}, seed=0,
                 diary=diary)(clean.clone(), _context())
    note = diary[0]
    assert "rescue_closed_distance_ratio" not in note
    # The absolute gap is still recorded, because how far the edit moved the state is a
    # real quantity even when a ratio is not.
    assert note["distance_from_clean_after_rescue_l2"] > 0.0
    assert note["distance_from_clean_incoming_l2"] == pytest.approx(0.0, abs=1e-6)
    assert CS.displacement_report(pd.DataFrame(diary)).empty


def test_a_vanishing_clean_projection_suppresses_the_ratio_but_not_the_spread_units():
    """The metric that could turn a large perturbation into a 400% recovery.

    Beyond the register zone the projection collapses toward zero by design, so a ratio to
    the clean mean divides by a quantity on its way to nothing and a mean over depth is
    dominated by whichever layer divided by the smallest number. The ratio is suppressed
    where the denominator is unresolvable; the spread-normalised change is defined there
    and carries the same information without the division.
    """
    from ditsinks.causal_engine import LayerObservation, Trace

    torch.manual_seed(0)
    n = 40
    ids = (0, 1, 2)

    def trace_with(alpha_register):
        trace = Trace(prompt_id=0, seed=0, condition="clean")
        projection = torch.randn(n) * 1.0
        projection[list(ids)] = alpha_register
        trace.rows[(0, 0)] = LayerObservation(
            layer=0, step=0, projection=projection,
            norm=torch.full((n,), 10.0), cosine=projection / 10.0)
        return trace

    treatment = CS.Treatment(mode="register", token_ids=ids, rule="t", n_candidates=n)
    # A collapsed clean projection: 0.001 against an ordinary spread near 1.
    rows = CS.population_rows(trace_with(50.0), trace_with(0.001), treatment,
                              condition=CS.Condition("c", "c", "rescue"), layers=[0],
                              steps=[0], norm_threshold=1.0, sink_threshold=10.0,
                              recovery_floor=0.1)
    assert len(rows) == 1
    row = rows[0]
    assert row["recovery_denominator_resolvable"] is False
    assert math.isnan(row["selected_alpha_recovery"])
    assert np.isfinite(row["selected_alpha_change_in_spreads"])
    assert row["ordinary_alpha_spread"] > 0

    # A resolvable one keeps the ratio.
    ok = CS.population_rows(trace_with(60.0), trace_with(50.0), treatment,
                            condition=CS.Condition("c", "c", "rescue"), layers=[0],
                            steps=[0], norm_threshold=1.0, sink_threshold=10.0,
                            recovery_floor=0.1)[0]
    assert ok["recovery_denominator_resolvable"] is True
    assert ok["selected_alpha_recovery"] == pytest.approx(60.0 / 50.0, rel=1e-6)


def test_the_paired_effects_are_taken_where_the_register_state_exists():
    """A recovery averaged over the dissolution zone answers a different question.

    The register zone is where the state the rescue targets is present; beyond it the
    projection is on its way to zero for reasons that have nothing to do with the
    intervention.
    """
    rows = []
    for prompt in range(4):
        for condition, inside, outside in (("channel_ablate", 0.2, 0.2),
                                           ("channel_ablate_vstar_rescue", 1.0, 40.0)):
            rows.append(dict(arm="rescue", condition=condition, prompt_id=prompt, seed=0,
                             is_downstream=True, is_register_zone=True,
                             selected_alpha_recovery=inside))
            rows.append(dict(arm="rescue", condition=condition, prompt_id=prompt, seed=0,
                             is_downstream=True, is_register_zone=False,
                             selected_alpha_recovery=outside))
    effects = CS.rescue_effects(pd.DataFrame(rows), iterations=200)
    row = effects.set_index(["condition", "metric"]).loc[
        ("channel_ablate_vstar_rescue", "selected_alpha_recovery")]
    # In-zone effect is 1.0 - 0.2 = 0.8. Including the out-of-zone rows would give ~19.8.
    assert float(row["effect"]) == pytest.approx(0.8, abs=1e-6), float(row["effect"])


# ================================================ the intervention-depth comparison
class _DepthContext:
    """The three fields `layers_for` reads, without building a model."""

    def __init__(self, writer_layer, writer_range, register, n_layers):
        self.writer_layer = writer_layer
        self.writer_range = writer_range
        self.register_layers = register
        self.intervention_layer = register[0]
        self.driver = type("d", (), {"n_layers": n_layers})()


FLUX_GEOMETRY = _DepthContext(19, (17, 19), tuple(range(20, 40)), 57)
PIXART_GEOMETRY = _DepthContext(10, (8, 10), tuple(range(10, 21)), 28)


def test_the_depth_layers_come_from_the_frozen_ranges_not_from_flux_numbers():
    """18/20/24 on FLUX and 9/10/14 on PixArt, from one rule.

    The depth comparison is only a depth comparison if each layer sits in the phase it is
    named after. Hardcoding FLUX's numbers would put all three inside PixArt's
    *dissolution* zone, and the experiment would silently be measuring something else.
    """
    config = CS.ControlSurfaceConfig(experiment_root=Path("/tmp/unused"))
    for geometry, expected in ((FLUX_GEOMETRY, [18, 20, 24]),
                               (PIXART_GEOMETRY, [9, 10, 14])):
        got = [config.layers_for(geometry, scope)[0] for scope in
               ("birth", "boundary", "established")]
        assert got == expected, (got, expected)
    # `boundary` must be exactly what the existing `write` scope already uses, so the
    # depth arm's middle point is the layer the main grid was run at.
    assert (config.layers_for(FLUX_GEOMETRY, "boundary")
            == config.layers_for(FLUX_GEOMETRY, "write"))


def test_each_depth_lands_in_the_phase_it_is_named_after():
    for geometry in (FLUX_GEOMETRY, PIXART_GEOMETRY):
        table = CS.depth_layer_table(geometry)
        assert table["lands_in_named_phase"].all(), table.to_dict("records")
        assert int(table["distinct_layers"].iloc[0]) == 3


def test_a_register_zone_too_short_for_the_offset_is_reported_not_hidden():
    """Two scopes on one layer would compare a layer against itself.

    A model whose last layer is the register start leaves the established offset nowhere
    to go: it clamps back onto the boundary, and the run would report a "depth
    comparison" between two identical interventions. The table must say so.
    """
    cramped = _DepthContext(4, (3, 4), (5,), 6)         # boundary 5 is also the last layer
    table = CS.depth_layer_table(cramped)
    assert table.set_index("scope")["layer"].to_dict() == {
        "birth": 3, "boundary": 5, "established": 5}
    assert int(table["distinct_layers"].iloc[0]) == 2


def test_an_unknown_scope_still_names_the_depth_scopes():
    config = CS.ControlSurfaceConfig(experiment_root=Path("/tmp/unused"))
    with pytest.raises(KeyError, match="birth"):
        config.layers_for(FLUX_GEOMETRY, "whatever_looks_best")


def test_negative_beta_inverts_the_projection_at_unchanged_norm():
    """What the depth sweep's negative arm is for.

    beta < 0 must flip the sign of the v* component while leaving the token's length
    alone, or "opposite-v* state" is not what is being produced.
    """
    torch.manual_seed(0)
    v = torch.randn(96)
    unit = CS._unit(v)
    x = (0.6 * unit + 0.8 * CS._unit(torch.randn(96))).unsqueeze(0) * 40.0
    before = float(x @ unit)
    assert before > 0
    for beta in (-0.25, -1.0):
        y, degenerate = CS.align_scale(x, v, beta)
        assert degenerate == 0
        assert float(y @ unit) < 0, beta
        assert torch.allclose(y.norm(dim=-1), x.norm(dim=-1), atol=1e-4, rtol=1e-5)
    # And monotone across the whole sweep the depth arm uses.
    cosines = [float((CS.align_scale(x, v, b)[0] @ unit
                      / CS.align_scale(x, v, b)[0].norm(dim=-1))[0])
               for b in (-0.25, 0.0, 0.5, 1.0, 2.0)]
    assert cosines == sorted(cosines), cosines
    assert cosines[0] < 0 < cosines[-1]


def test_the_clean_and_realised_alignment_are_commensurable():
    """The check that makes panel (a) of the depth figure readable.

    Both are signed means, so their difference is exactly zero at beta = 1, the no-op
    cell. Comparing a mean of absolute values against a mean of signed ones breaks that
    identity wherever the register population has mixed signs, and the panel would show a
    gap at the one cell where there cannot be one.
    """
    rows = []
    for scope, clean in (("birth", 0.30), ("boundary", 0.95)):
        for beta in (-0.25, 0.0, 1.0, 2.0):
            realised = clean if beta == 1.0 else clean * beta * 0.9
            rows.append(dict(arm="grid", layer_scope=scope, layer=1, beta=beta, gamma=1.0,
                             is_edited_layer=True, selected_cosine_mean=realised,
                             clean_cosine_signed=clean, clean_cosine_abs=abs(clean),
                             clean_cosine_headroom=1.0 - abs(clean)))
    table = CS.depth_response(pd.DataFrame(rows))
    assert len(table) == 8
    identity = table[np.isclose(table["beta"], 1.0)]
    assert np.allclose(identity["cosine_gain"], 0.0, atol=1e-12), identity.to_dict("records")
    # And the headroom is the room |cos| had, which is what makes a flat response readable
    # as a ceiling rather than as a failure.
    birth = table[table["layer_scope"] == "birth"]["clean_cosine_headroom"].iloc[0]
    boundary = table[table["layer_scope"] == "boundary"]["clean_cosine_headroom"].iloc[0]
    assert birth == pytest.approx(0.70) and boundary == pytest.approx(0.05)
    assert birth > boundary, "the writer phase must have more room than the boundary"


def test_the_manifold_measure_separates_an_off_manifold_state_of_equal_norm():
    """The only quantity here that can answer "meaningful state, or corruption?".

    The operators fix norm and alignment by construction, so neither can distinguish a
    state the model could have produced from one it could not. A direction outside the
    space the clean tokens span must score lower even at identical length.
    """
    torch.manual_seed(0)
    width, rank = 256, 8
    basis = torch.linalg.qr(torch.randn(width, rank))[0].t()      # [rank, width]
    clean = torch.randn(400, rank) @ basis * 3.0                  # lives in the subspace
    subspace = CS.clean_subspace(clean, rank=rank)
    assert subspace and subspace["rank"] == rank
    assert subspace["clean_share_median"] > 0.99
    assert subspace["variance_explained"] > 0.99

    inside = clean[:4]
    outside = torch.linalg.qr(torch.randn(width, rank + 4))[0].t()[rank:]
    off = (torch.randn(4, 4) @ outside)
    off = off / off.norm(dim=-1, keepdim=True) * inside.norm(dim=-1, keepdim=True)
    assert torch.allclose(off.norm(dim=-1), inside.norm(dim=-1), rtol=1e-5)
    assert CS.onmanifold_energy(inside, subspace) > 0.98
    assert CS.onmanifold_energy(off, subspace) < 0.10


def test_a_subspace_that_captures_everything_is_flagged_as_undiscriminating():
    """A rank approaching the width makes every state look on-manifold.

    That happens on a narrow model, or with the rank set too high, and a reader would
    take the resulting 1.0 for a finding. The variance the subspace explains is recorded
    so the degenerate case is visible.
    """
    torch.manual_seed(0)
    narrow = torch.randn(64, 12)
    subspace = CS.clean_subspace(narrow, rank=32)
    assert subspace["rank"] < 12, "the rank must clamp to the data"
    # Rank 11 of 12 explains only ~0.97, so a variance threshold would call this
    # discriminative. It is not: there is nowhere left to be off-manifold. The dimension
    # is what decides.
    assert subspace["rank_fraction"] > 0.5
    assert subspace["is_discriminative"] is False
    # ~0.97 rather than ~1.0, because 11 of 12 directions is not quite all of them, but
    # near enough that no state has room to score low, which is the point.
    assert CS.onmanifold_energy(narrow, subspace) > 0.9
    # And a real restriction is flagged the other way, at a lower variance explained.
    torch.manual_seed(1)
    wide = CS.clean_subspace(torch.randn(400, 512), rank=32)
    assert wide["rank_fraction"] < 0.1 and wide["is_discriminative"] is True
    assert wide["variance_explained"] < subspace["variance_explained"]


def test_the_depth_arm_leaves_the_existing_scopes_and_grid_untouched():
    """Additive by construction: the main run must be bit-identical after this change."""
    config = CS.ControlSurfaceConfig(experiment_root=Path("/tmp/unused"))
    assert config.layer_scopes == ("write", "window")
    assert config.betas == (0.0, 0.5, 1.0, 1.5, 2.0)
    assert config.gammas == (0.5, 0.75, 1.0, 1.25, 1.5)
    assert config.layers_for(FLUX_GEOMETRY, "write") == [20]
    assert config.layers_for(FLUX_GEOMETRY, "window") == list(range(20, 40))
    assert {c.key for c in CS.RESCUE_CONDITIONS} == {
        "clean", "channel_ablate", "channel_ablate_vstar_rescue",
        "channel_ablate_orthogonal_rescue", "channel_restore_only"}


def test_the_depth_grid_builds_every_cell_including_the_negative_beta():
    betas, gammas = (-0.25, 0.0, 0.5, 1.0, 2.0), (0.5, 1.0, 1.5, 2.0)
    cells = CS.grid_conditions(betas, gammas)
    assert len(cells) == 20
    assert any(np.isclose(c.beta, -0.25) for c in cells)
    # The centre cell is still the reference, and its directory name survives a minus sign.
    centre = [c for c in cells if c.beta == 1.0 and c.gamma == 1.0]
    assert len(centre) == 1 and centre[0].role == "reference"
    negative = next(c for c in cells if np.isclose(c.beta, -0.25) and c.gamma == 0.5)
    assert negative.directory == "grid/beta_-0.25_gamma_0.5"
    # The controls anchor on the sweep's extremes, so a negative beta becomes the low end.
    controls = {c.key for c in CS.control_conditions(betas)}
    assert "channel_axis_beta-0.25" in controls and "orthogonal_beta2" in controls


def _depth_frame(clean_by_scope, *, with_image=False):
    """A minimal depth_response frame: three depths, three betas, one gamma."""
    layers = {"birth": 18, "boundary": 20, "established": 24}
    rows = []
    for scope, clean in clean_by_scope.items():
        for beta in (-0.25, 1.0, 2.0):
            realised = float(np.clip(clean + 0.3 * (beta - 1.0), -1.0, 1.0))
            row = dict(layer_scope=scope, layer=layers[scope], beta=beta,
                       realised_cosine=realised, clean_cosine=clean,
                       clean_cosine_abs=abs(clean), clean_cosine_headroom=1.0 - abs(clean),
                       onmanifold_ratio_to_clean=0.98, selected_alpha_mean=5.0 * beta)
            if with_image:
                row["lpips"] = 0.05 + 0.04 * abs(beta - 1.0)
            rows.append(row)
    return pd.DataFrame(rows)


def _assert_inside(figure, axis, texts):
    """Every label is drawn within the axes, in rendered pixels rather than data units.

    The two branches place their text in different coordinate systems, one in data
    coordinates beside the mark, one as an offset from it, so the only assertion that
    covers both is the one that matters anyway: does it land on the panel.
    """
    figure.canvas.draw()
    panel = axis.get_window_extent()
    for text in texts:
        box = text.get_window_extent()
        assert (panel.y0 - 1 <= box.y0 and box.y1 <= panel.y1 + 1
                and panel.x0 - 1 <= box.x0 and box.x1 <= panel.x1 + 1), (
            f"{text.get_text()!r} at {box.extents} escapes the panel {panel.extents}")


def test_the_depth_panel_is_labelled_as_a_signed_alignment():
    """beta < 0 inverts the sign, so the panel plots cos and not |cos|.

    The label carried absolute-value bars while the line reached -0.78, which reads as
    a broken measurement rather than as the inversion the negative arm exists to make.
    """
    figure = CF.fig_q13_depth_response(_depth_frame({"birth": 0.42, "boundary": 0.71,
                                                     "established": 0.95}))
    labels = [ax.get_ylabel() for ax in figure.axes]
    alignment = next(l for l in labels if "cos" in l)
    assert "signed" in alignment and "|" not in alignment, alignment
    plt.close(figure)


def test_three_indistinguishable_clean_levels_become_one_label_not_three():
    """Three leader lines to the same pixel say nothing and overlap the panel letter.

    On the synthetic checkpoint all three depths start from cos = 0.981, and the earlier
    spreading pushed two labels outside the axes where matplotlib drew them over the
    figure's own furniture.
    """
    figure = CF.fig_q13_depth_response(
        _depth_frame({"birth": 0.9815, "boundary": 0.9816, "established": 0.9808}))
    panel = figure.axes[0]
    clean = [t for t in panel.texts if t.get_text().startswith("clean")]
    assert len(clean) == 1, [t.get_text() for t in clean]
    assert "all 3 depths" in clean[0].get_text()
    _assert_inside(figure, panel, clean)
    plt.close(figure)

    # And again with the shared level low in the range, which is the other hanging side.
    figure = CF.fig_q13_depth_response(
        _depth_frame({"birth": -0.601, "boundary": -0.600, "established": -0.602}))
    panel = figure.axes[0]
    clean = [t for t in panel.texts if t.get_text().startswith("clean")]
    assert len(clean) == 1, [t.get_text() for t in clean]
    _assert_inside(figure, panel, clean)
    plt.close(figure)


def test_separated_clean_levels_keep_one_label_each_inside_the_axes():
    figure = CF.fig_q13_depth_response(_depth_frame({"birth": 0.10, "boundary": 0.55,
                                                     "established": 0.95}))
    panel = figure.axes[0]
    clean = [t for t in panel.texts if t.get_text().startswith("clean")]
    assert len(clean) == 3, [t.get_text() for t in clean]
    _assert_inside(figure, panel, clean)
    plt.close(figure)


def test_the_depth_caption_says_when_the_third_panel_is_not_an_image():
    """A checkpoint with no decoder falls back to the internal state; the caption said
    'what reached the image' either way, which would have been read as an image claim."""
    without = CF.fig_q13_depth_response(_depth_frame({"birth": 0.4, "boundary": 0.7,
                                                      "established": 0.9}))
    assert "No image distances were available" in without.dv_caption
    assert "own projection, not an image response" in without.dv_caption
    plt.close(without)

    with_images = CF.fig_q13_depth_response(
        _depth_frame({"birth": 0.4, "boundary": 0.7, "established": 0.9}, with_image=True))
    assert "What reached the image." in with_images.dv_caption
    assert "No image distances were available" not in with_images.dv_caption
    plt.close(with_images)


# ============================ the writer-phase stage probe (post-MLP intervention point)
def test_the_postmlp_suffix_moves_the_edit_to_the_block_output_and_nothing_else():
    """The suffix is the whole mechanism: same layer, different site inside the block."""
    config = CS.ControlSurfaceConfig(experiment_root=Path("/tmp/unused"))
    assert CS.split_scope("birth") == ("birth", None)
    assert CS.split_scope("birth_postmlp") == ("birth", InterventionPoint.BLOCK_OUTPUT)
    assert config.layers_for(FLUX_GEOMETRY, "birth_postmlp") == \
        config.layers_for(FLUX_GEOMETRY, "birth") == [18]
    assert config.point_for("birth") is InterventionPoint.BLOCK_INPUT
    assert config.point_for("birth_postmlp") is InterventionPoint.BLOCK_OUTPUT
    # Every scope that existed before this keeps the configuration's own point.
    for scope in ("write", "window", "boundary", "established"):
        assert config.point_for(scope) is config.point


def test_the_writer_offsets_are_the_writer_range_block_by_block():
    """FLUX 17/18/19 and PixArt 8/9/10, consecutive writer blocks, from the artifact."""
    config = CS.ControlSurfaceConfig(experiment_root=Path("/tmp/unused"))
    assert [config.layers_for(FLUX_GEOMETRY, f"writer+{k}")[0] for k in (0, 1, 2)] == \
        [17, 18, 19]
    assert [config.layers_for(PIXART_GEOMETRY, f"writer+{k}")[0] for k in (0, 1, 2)] == \
        [8, 9, 10]
    # 'birth' already names the middle one, so the two agree where they overlap.
    assert config.layers_for(FLUX_GEOMETRY, "writer+1") == \
        config.layers_for(FLUX_GEOMETRY, "birth")


def test_the_depth_table_names_the_site_as_well_as_the_layer():
    table = CS.depth_layer_table(FLUX_GEOMETRY, ("birth", "birth_postmlp", "boundary"))
    by_scope = table.set_index("scope")
    assert int(by_scope.loc["birth", "layer"]) == int(by_scope.loc["birth_postmlp", "layer"])
    assert by_scope.loc["birth", "point"] == "block_input"
    assert by_scope.loc["birth_postmlp", "point"] == "block_output"
    # Two scopes on one layer is no longer a collision when they edit different sites,
    # so the table reports both counts and the notebook can tell them apart.
    assert int(table["distinct_layers"].iloc[0]) == 2
    assert int(table["distinct_sites"].iloc[0]) == 3


def test_the_stage_order_follows_the_edit_point_not_the_column_order():
    """At BLOCK_INPUT the hook fires first and the rest is the block's response; at
    BLOCK_OUTPUT the hook fires last. Reading one order for both inverts the meaning."""
    early = CS._FORWARD_ORDER["block_input"]
    late = CS._FORWARD_ORDER["block_output"]
    assert early["after_edit"] < early["before_mlp"] < early["after_mlp_write"]
    assert late["before_mlp"] < late["after_mlp_write"] < late["after_edit"]
    for order in (early, late):
        assert order["after_next_block"] == 3, "the following block is always last"


def test_the_stage_probe_reads_all_four_stages_and_says_which_are_post_edit():
    """Built on a hand-made trace, so the wiring is checked without a generation."""
    width, n = 8, 6
    torch.manual_seed(0)
    v = torch.zeros(width); v[0] = 1.0
    ids = [1, 2]
    trace = Trace(prompt_id=0, seed=0, condition="t")
    pre_mlp, block_out, edited = torch.zeros(n, width), torch.zeros(n, width), torch.zeros(n, width)
    pre_mlp[:, 0], block_out[:, 0], edited[:, 0] = 2.0, 5.0, 0.0
    pre_mlp[:, 1] = block_out[:, 1] = edited[:, 1] = 1.0
    trace.probe_states[(3, 4, InterventionPoint.PRE_MLP_RESIDUAL.value)] = pre_mlp
    trace.probe_states[(3, 4, InterventionPoint.BLOCK_OUTPUT.value)] = block_out
    trace.diagnostics[(3, 4, InterventionPoint.BLOCK_OUTPUT.value, "x_post_hook", 0)] = edited
    rows = CS.stage_rows(trace, direction=v, token_ids=ids, layer=4, step=3,
                         point=InterventionPoint.BLOCK_OUTPUT)
    by_stage = {r["stage"]: r for r in rows}
    assert len(rows) == 4 and set(by_stage) == set(CS.STAGE_ORDER)
    assert by_stage["before_mlp"]["alpha_mean"] == pytest.approx(2.0)
    assert by_stage["after_mlp_write"]["alpha_mean"] == pytest.approx(5.0)
    assert by_stage["after_edit"]["alpha_mean"] == pytest.approx(0.0)
    # The following block was never recorded, so it is NaN rather than a quiet zero.
    assert by_stage["after_next_block"]["measured"] is False
    assert by_stage["before_mlp"]["is_after_the_edit"] is False
    assert by_stage["after_edit"]["is_after_the_edit"] is True


def test_a_block_without_a_post_attention_stage_is_dropped_rather_than_raising():
    """FLUX single blocks feed attention and the feed-forward from one normalised stream,
    so `pre_mlp_residual` does not exist at layer 19, and asking for it would raise."""
    class _Cap:
        def __init__(self, ok): self.supported = ok

    class _Adapter:
        def layers(self, _):
            return [type("R", (), {"index": i})() for i in (17, 18, 19)]

        def intervention_capability(self, ref, point):
            return _Cap(int(ref.index) < 19)

    ctx = type("C", (), {"driver": type("D", (), {"adapter": _Adapter(),
                                                  "transformer": None})()})()
    kept = CS._supported_layers(ctx, InterventionPoint.PRE_MLP_RESIDUAL, [17, 18, 19])
    assert kept == [17, 18], kept


def _stage_frame(**overrides):
    """A stage_probe frame with two sites, two depths and the three site betas."""
    rows = []
    for scope, layer in (("birth", 18), ("birth_postmlp", 18),
                         ("boundary", 20), ("boundary_postmlp", 20)):
        point = "block_output" if scope.endswith("_postmlp") else "block_input"
        order = CS._FORWARD_ORDER[point]
        for beta in (-0.25, 0.0, 1.0):
            for stage in CS.STAGE_ORDER:
                clean = {"before_mlp": -3.0, "after_mlp_write": -5.0,
                         "after_edit": -5.0 if point == "block_output" else -3.0,
                         "after_next_block": -6.0}[stage]
                if beta == 1.0 or order[stage] < order["after_edit"]:
                    value = clean
                elif stage == "after_edit":
                    value = 0.0 if beta == 0.0 else 1.2
                else:
                    value = clean * (0.5 if point == "block_input" else 0.15)
                rows.append(dict(layer_scope=scope, layer=layer, edit_point=point,
                                 beta=beta, stage=stage, forward_order=order[stage],
                                 alpha_mean=value, cosine_mean=value / 6.0))
    return pd.DataFrame(rows).assign(**overrides)


def test_the_stage_figure_never_shares_an_axis_between_the_two_edit_sites():
    """A block-input hook fires before the block and a block-output hook after, so one
    shared stage axis would read the block's response to the edit as its cause."""
    figure = CF.fig_q13_stage_probe(_stage_frame(), checkpoint="flux1-schnell", beta=0.0)
    trajectories = [ax for ax in figure.axes if ax.get_title().startswith("edit ")]
    assert len(trajectories) == 2, [ax.get_title() for ax in figure.axes]
    labels = [[t.get_text() for t in ax.get_xticklabels()] for ax in trajectories]
    assert labels[0] != labels[1], "the two sites cannot carry the same stage order"
    assert labels[0][0].startswith("the hook"), labels[0]
    assert labels[1][-2].startswith("the hook"), labels[1]
    plt.close(figure)


def test_a_sign_flip_is_only_reported_where_there_is_a_magnitude_to_have_a_sign():
    """Near total suppression the residue is noise about zero and its sign is a coin
    toss; reporting that as an inversion would publish float precision as a finding."""
    frame = _stage_frame()
    noise = ((frame["stage"] == "after_next_block") & np.isclose(frame["beta"], 0.0)
             & (frame["edit_point"] == "block_output"))
    frame.loc[noise, "alpha_mean"] = 0.01           # 0.17% of the clean -6.0, wrong sign
    figure = CF.fig_q13_stage_probe(frame, beta=0.0)
    flipped = [t.get_text() for ax in figure.axes for t in ax.texts
               if "sign flipped" in t.get_text()]
    assert not flipped, flipped
    plt.close(figure)

    frame.loc[noise, "alpha_mean"] = 3.0            # 50% of the clean, wrong sign: real
    figure = CF.fig_q13_stage_probe(frame, beta=0.0)
    flipped = [t.get_text() for ax in figure.axes for t in ax.texts
               if "sign flipped" in t.get_text()]
    assert flipped, "a genuine inversion must still be marked"
    plt.close(figure)


def test_the_stage_figure_refuses_a_beta_it_has_no_rows_for():
    with pytest.raises(ValueError, match="no rows at beta"):
        CF.fig_q13_stage_probe(_stage_frame(), beta=7.5)


# ================================================= the true rotation arm (not a rescale)
def _plane_setup(width=512, n=48, seed=0):
    torch.manual_seed(seed)
    v = torch.randn(width)
    x = torch.randn(n, width) * 2.0 + 1.5 * v
    return v, x


def test_the_rotation_preserves_the_norm_by_construction_not_by_correction():
    """The claim the whole arm rests on. No renormalisation is applied anywhere in
    `rotate_plane`; the norm is preserved because (alpha, b) -> (alpha', b') is an
    orthogonal map of R^2 and everything outside the plane is untouched."""
    v, x = _plane_setup()
    for target in ("random_orthogonal", "residual_pc1"):
        basis = CS.plane_basis(v, target, width=int(v.numel()), seed=3, residuals=x)
        for degrees in CS.ROTATION_ANGLES_DEG:
            out = CS.rotate_plane(x, basis["v"], basis["u"], math.radians(degrees))
            error = CS.rotation_norm_error(x, out)
            assert error < 1e-5, f"{target} at {degrees}deg drifted by {error:.3e}"
            assert error <= CS._ROTATION_NORM_TOLERANCE


def test_theta_zero_reproduces_the_activation():
    v, x = _plane_setup()
    basis = CS.plane_basis(v, "residual_pc1", width=int(v.numel()), residuals=x)
    out = CS.rotate_plane(x, basis["v"], basis["u"], 0.0)
    assert torch.allclose(out, x.float(), atol=1e-5), float((out - x.float()).abs().max())


def test_the_rotation_follows_the_stated_formula_and_leaves_the_rest_alone():
    """alpha' = alpha cos - b sin, b' = alpha sin + b cos, q untouched."""
    v, x = _plane_setup()
    basis = CS.plane_basis(v, "random_orthogonal", width=int(v.numel()), seed=11)
    unit_v, unit_u = basis["v"], basis["u"]
    alpha, b = x.float() @ unit_v, x.float() @ unit_u
    q = x.float() - alpha.unsqueeze(-1) * unit_v - b.unsqueeze(-1) * unit_u
    for degrees in (-90.0, -30.0, 15.0, 60.0):
        theta = math.radians(degrees)
        out = CS.rotate_plane(x, unit_v, unit_u, theta)
        assert torch.allclose(out @ unit_v, alpha * math.cos(theta) - b * math.sin(theta),
                              atol=1e-3), degrees
        assert torch.allclose(out @ unit_u, alpha * math.sin(theta) + b * math.cos(theta),
                              atol=1e-3), degrees
        out_q = (out - (out @ unit_v).unsqueeze(-1) * unit_v
                 - (out @ unit_u).unsqueeze(-1) * unit_u)
        assert torch.allclose(q, out_q, atol=1e-3), f"q moved at {degrees}deg"


def test_the_rotation_is_not_the_beta_operator():
    """The distinction between a rotation and the beta operator, as a test rather than a comment.

    beta rescales the v* coefficient and then restores the length, so it can never put
    weight on a chosen direction u. A rotation transfers the alignment from v* to u.
    """
    v, x = _plane_setup()
    basis = CS.plane_basis(v, "residual_pc1", width=int(v.numel()), residuals=x)
    unit_v, unit_u = basis["v"], basis["u"]
    rescaled, _ = CS.control_surface_states(x, unit_v, 2.0, 1.0)
    turned = CS.rotate_plane(x, unit_v, unit_u, math.radians(90.0))
    before_u = float(((x.float() @ unit_u) / x.float().norm(dim=-1)).mean())
    # beta leaves the u coordinate where it was; the rotation moves it by design.
    assert abs(float(((rescaled @ unit_u) / rescaled.norm(dim=-1)).mean()) - before_u) < 0.02
    assert abs(float(((turned @ unit_u) / turned.norm(dim=-1)).mean()) - before_u) > 0.2
    # And beta cannot invert the sign of cos at fixed norm the way a 90deg turn does.
    assert float(((turned @ unit_v) / turned.norm(dim=-1)).mean()) < \
        float(((x.float() @ unit_v) / x.float().norm(dim=-1)).mean())


def test_every_u_is_unit_and_orthogonal_to_vstar_however_it_was_obtained():
    v, x = _plane_setup()
    semantic = torch.randn(int(v.numel())) + 0.6 * v      # deliberately not orthogonal
    for target, extra in (("random_orthogonal", {}), ("residual_pc1", {"residuals": x}),
                          ("semantic_direction", {"semantic": semantic})):
        basis = CS.plane_basis(v, target, width=int(v.numel()), seed=5, **extra)
        assert abs(basis["u_dot_v"]) < 1e-5, (target, basis["u_dot_v"])
        assert abs(float(basis["u"].norm()) - 1.0) < 1e-5, target
        assert abs(float(basis["v"].norm()) - 1.0) < 1e-5, target


def test_semantic_direction_refuses_rather_than_substituting_a_random_one():
    """A random direction reported as semantic would be a fabricated result."""
    v, _ = _plane_setup()
    with pytest.raises(ValueError, match="no direction to use"):
        CS.plane_basis(v, "semantic_direction", width=int(v.numel()))
    with pytest.raises(ValueError, match="width"):
        CS.plane_basis(v, "semantic_direction", width=int(v.numel()),
                       semantic=torch.randn(7))


def test_residual_pc1_needs_the_clean_population_and_says_so():
    v, x = _plane_setup()
    with pytest.raises(ValueError, match="needs the clean token states"):
        CS.plane_basis(v, "residual_pc1", width=int(v.numel()))
    with pytest.raises(ValueError, match="at least two clean tokens"):
        CS.plane_basis(v, "residual_pc1", width=int(v.numel()), residuals=x[:1])


def test_a_target_parallel_to_vstar_is_refused_rather_than_rotated_in_noise():
    v, _ = _plane_setup()
    with pytest.raises(ValueError, match="parallel to v"):
        CS.plane_basis(v, "semantic_direction", width=int(v.numel()), semantic=v.clone())


def test_a_nearly_parallel_target_is_announced():
    """Built deterministically: 5% of a perpendicular direction added to v* itself, so
    almost nothing survives the orthogonalisation and the plane is mostly numerical."""
    v, _ = _plane_setup()
    width = int(v.numel())
    perpendicular = CS.orthogonal_direction(CS._unit(v), width, seed=1)
    nearly = CS._unit(v) + 0.05 * perpendicular
    basis = CS.plane_basis(v, "semantic_direction", width=width, semantic=nearly)
    assert basis["notes"] and "survives orthogonalisation" in basis["notes"][0]
    assert basis["independent_target_content"] < 0.1


def test_residual_pc1_has_a_pinned_sign_so_plus_theta_means_one_thing():
    v, x = _plane_setup()
    first = CS.plane_basis(v, "residual_pc1", width=int(v.numel()), residuals=x)["u"]
    second = CS.plane_basis(v, "residual_pc1", width=int(v.numel()), residuals=x)["u"]
    assert torch.allclose(first, second)
    assert float(first[int(first.abs().argmax())]) > 0.0


def test_the_rotation_conditions_cover_every_angle_and_name_the_reference():
    conditions = CS.rotation_conditions()
    assert len(conditions) == len(CS.ROTATION_ANGLES_DEG) * 2
    zero = [c for c in conditions if c.theta_deg == 0.0]
    assert len(zero) == 2 and all(c.role == "reference" for c in zero), \
        "theta = 0 is the numerical floor in every target, so it is generated not skipped"
    for condition in conditions:
        assert condition.arm == "rotation" and condition.edit_direction == "rotation"
        # A rotation is not a rescale: beta and gamma stay at identity throughout.
        assert condition.beta == 1.0 and condition.gamma == 1.0
    assert conditions[0].directory.startswith("rotation/random_orthogonal/theta_")


def test_the_rotation_arm_is_off_by_default_and_changes_nothing_else():
    config = CS.ControlSurfaceConfig(experiment_root=Path("/tmp/unused"))
    assert config.run_rotation is False
    assert config.semantic_direction is None
    assert config.betas == (0.0, 0.5, 1.0, 1.5, 2.0)
    assert config.layer_scopes == ("write", "window")


def test_a_broken_plane_raises_instead_of_being_renormalised_away():
    """If u is not orthonormal the norm moves, and the edit must fail loudly: correcting
    it would leave the norms right and every reported angle wrong."""
    width = 64
    v = torch.zeros(width); v[0] = 1.0
    clean = torch.randn(4, width)
    ids = [0, 1]
    condition = CS.rotation_conditions((45.0,), ("random_orthogonal",))[0]
    broken = dict(target="random_orthogonal", v=CS._unit(v),
                  u=CS._unit(v) * 0.5 + torch.randn(width) * 0.1,   # neither unit nor perp
                  target_cosine_with_vstar=0.0, independent_channel_content=1.0,
                  independent_target_content=1.0, u_dot_v=0.5, seed=0, notes=[])
    edit = CS.make_edit(condition, direction=v, token_ids=ids, dominant_channel=0,
                        clean_states={}, seed=0, planes={(0, 0): broken})
    with pytest.raises(AssertionError, match="preserves norm exactly"):
        edit(clean.clone(), _context())


def test_the_plane_share_is_the_sum_of_its_two_parts_and_never_below_cos_squared():
    """The inequality the arm's whole framing rests on.

    plane share = alpha^2/||x||^2 + b^2/||x||^2, and the first term IS cos^2(x, v*). So a
    token aligned with v* has a share of at least cos^2 for EVERY u, which is why the
    sum cannot distinguish one target from another and b^2 is the column that does.
    """
    v, x = _plane_setup()
    # A deliberately high-alignment population, like a real register token.
    aligned = 12.0 * CS._unit(v) + 0.06 * x
    for target, extra in (("random_orthogonal", {}),
                          ("residual_pc1", {"residuals": aligned})):
        basis = CS.plane_basis(v, target, width=int(v.numel()), seed=2, **extra)
        shares = CS._plane_shares(aligned, basis["v"], basis["u"])
        cosine = float(((aligned @ basis["v"]) / aligned.norm(dim=-1)).mean())
        assert cosine > 0.9, f"the fixture must be high-alignment, got {cosine:.3f}"
        assert shares["alpha2_share"] + shares["b2_share"] == \
            pytest.approx(shares["in_plane_energy_share"], abs=1e-6)
        assert shares["in_plane_energy_share"] >= shares["cos_squared_mean"] - 1e-6
        assert shares["in_plane_share_at_least_cos_squared"] is True
        # And the share is therefore large whatever u is, the point of the correction.
        assert shares["in_plane_energy_share"] > 0.9, target
        assert shares["out_of_plane_share"] == pytest.approx(
            1.0 - shares["in_plane_energy_share"], abs=1e-6)


def test_b_squared_is_what_distinguishes_one_target_from_another():
    """Both targets give a ~0.96 plane share on an aligned token; their b^2 differ by
    orders of magnitude. This is the reading the first version of the figure got wrong."""
    v, x = _plane_setup(width=3072, n=64)
    aligned = 12.0 * CS._unit(v) + 0.06 * x
    shares = {}
    for target, extra in (("random_orthogonal", {}),
                          ("residual_pc1", {"residuals": aligned})):
        basis = CS.plane_basis(v, target, width=3072, seed=2, **extra)
        shares[target] = CS._plane_shares(aligned, basis["v"], basis["u"])
    totals = [shares[t]["in_plane_energy_share"] for t in shares]
    assert abs(totals[0] - totals[1]) < 0.02, \
        "the plane shares are near-identical, so the sum cannot discriminate"
    assert shares["residual_pc1"]["b2_share"] > \
        10 * shares["random_orthogonal"]["b2_share"], \
        "b^2 is the quantity that separates a data-chosen u from a random one"


def test_a_share_below_cos_squared_raises_rather_than_being_reported():
    """Only reachable by mismatching the states, which is exactly the bug to catch."""
    width = 64
    v = torch.zeros(width); v[0] = 1.0
    u = torch.zeros(width); u[1] = 1.0
    x = torch.randn(4, width)
    shares = CS._plane_shares(x, v, u)
    assert shares["in_plane_share_at_least_cos_squared"] is True
    # A u that is NOT orthogonal to v double-counts alpha, so the two coordinates claim
    # more of the token than it has. That is the basis error the guard exists for, and
    # the inequality against an independently computed cos^2 is what makes it catchable.
    not_orthogonal = CS._unit(v) + 0.02 * u
    with pytest.raises(AssertionError, match="above 1|BELOW cos"):
        CS._plane_shares(torch.cat([x, (8.0 * v).unsqueeze(0)]), v, not_orthogonal)


# ========================================= rotating toward where the ordinary tokens point
def test_the_ordinary_targets_use_the_ordinary_population_not_the_treated_one():
    """`residual_pc1` is fitted on the treated slice; the ordinary targets are fitted on
    everything that is neither a register token nor high-norm. Passing the wrong one
    would make 'where ordinary tokens point' a statement about the register."""
    torch.manual_seed(0)
    width = 256
    v = torch.randn(width)
    ordinary = torch.randn(30, width) * 2.0 + 0.8 * v
    treated = torch.randn(4, width) * 2.0 + 9.0 * v
    for target in CS.TARGETS_NEEDING_ORDINARY:
        with pytest.raises(ValueError, match="ORDINARY tokens"):
            CS.plane_basis(v, target, width=width, residuals=treated)
        basis = CS.plane_basis(v, target, width=width, residuals=treated,
                               ordinary=ordinary)
        assert abs(basis["u_dot_v"]) < 1e-5
        assert abs(float(basis["u"].norm()) - 1.0) < 1e-5
    with pytest.raises(ValueError, match="at least two ordinary tokens"):
        CS.plane_basis(v, "ordinary_mean", width=width, ordinary=ordinary[:1])


def test_ordinary_mean_averages_unit_directions_not_raw_vectors():
    """Otherwise the longest ordinary token decides where 'the ordinary tokens point',
    which is a question about direction and not about length."""
    width = 32
    v = torch.zeros(width); v[0] = 1.0
    # Two directions, one of them 50x longer. A mean of raw vectors would follow it; a
    # mean of unit directions splits the difference.
    short = torch.zeros(width); short[1] = 1.0
    long = torch.zeros(width); long[2] = 50.0
    basis = CS.plane_basis(v, "ordinary_mean", width=width,
                           ordinary=torch.stack([short, short, long]))
    u = basis["u"]
    # Two of the three point along axis 1, so it must dominate, which a raw mean would
    # have lost entirely to the single long vector on axis 2.
    assert abs(float(u[1])) > abs(float(u[2])), (float(u[1]), float(u[2]))


def test_the_ordinary_direction_concentration_is_announced():
    """If the ordinary tokens are spread over the sphere their mean is mostly
    cancellation, and 'where ordinary tokens point' is not a well-posed target."""
    torch.manual_seed(3)
    width = 512
    v = torch.randn(width)
    isotropic = torch.randn(64, width)              # no shared direction at all
    basis = CS.plane_basis(v, "ordinary_mean", width=width, ordinary=isotropic)
    note = next((n for n in basis["notes"] if "concentration" in n), "")
    assert note, basis["notes"]
    value = float(note.split("=")[1].split(".")[0] + "." + note.split("=")[1].split(".")[1][:4])
    assert value < 0.3, f"isotropic tokens should concentrate near zero, got {value}"

    aligned = torch.randn(64, width) * 0.1 + 6.0 * CS._unit(torch.randn(width))
    basis = CS.plane_basis(v, "ordinary_mean", width=width, ordinary=aligned)
    note = next((n for n in basis["notes"] if "concentration" in n), "")
    assert "0.9" in note or "1.0" in note, note


def test_an_unknown_rotation_target_lists_the_real_ones():
    v = torch.randn(64)
    with pytest.raises(KeyError, match="ordinary_mean"):
        CS.plane_basis(v, "wishful_thinking", width=64)


# ============================================================= the visual rotation strip
def _strip_images(target="ordinary_mean", angles=(-90.0, 0.0, 90.0), with_paths=True):
    rows = []
    for angle in angles:
        rows.append(dict(rotation_target=target, theta_deg=angle,
                         image_path=f"stub/{angle:g}.png" if with_paths else None,
                         lpips=0.04 + abs(angle) / 900.0))
    return pd.DataFrame(rows)


def test_the_strip_refuses_rather_than_drawing_frames_of_nothing():
    """It is the one figure here that is entirely about the picture."""
    with pytest.raises(ValueError, match="no images were saved"):
        CF.fig_q13_rotation_strip(_strip_images(with_paths=False), "/tmp")
    with pytest.raises(ValueError, match="no rows for target"):
        CF.fig_q13_rotation_strip(_strip_images(), "/tmp", target="not_a_target")


def test_the_strip_gives_each_readout_its_own_axis(tmp_path):
    """A token count, a signed channel value and a sink strength share no scale, and
    dividing each by its unmodified value inverts the ones whose reference is negative --
    which an earlier version did, turning the channel panel upside down."""
    from PIL import Image

    (tmp_path / "stub").mkdir()
    angles = (-90.0, 0.0, 90.0)
    for angle in angles:
        Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(
            tmp_path / f"stub/{angle:g}.png")
    readouts = pd.DataFrame([
        dict(rotation_target="ordinary_mean", theta_deg=angle,
             n_highnorm=4.0, n_highnorm_and_aligned=2.0,
             dominant_channel_value=-0.47 + angle / 100.0,       # crosses zero
             selected_sink_strength_mean=1.06 - angle / 1000.0, n_sinks=1.0)
        for angle in angles])
    figure = CF.fig_q13_rotation_strip(_strip_images(angles=angles), tmp_path, readouts,
                                       target="ordinary_mean")
    panels = [ax for ax in figure.axes if ax.get_title() and not ax.images]
    assert len(panels) == 5, [ax.get_title() for ax in figure.axes]
    # Each panel's y range covers its own quantity, not a shared "share of unmodified".
    ranges = [ax.get_ylim() for ax in panels]
    assert len({tuple(round(v, 3) for v in r) for r in ranges}) > 1, \
        "the panels share one y range, so they were normalised onto one scale"
    channel = next(ax for ax in panels if "channel" in ax.get_title())
    low, high = channel.get_ylim()
    assert low < 0.0 < high, "the channel value crosses zero and the axis must show it"
    plt.close(figure)


def test_the_strip_marks_the_unmodified_frame_and_thins_a_long_sweep(tmp_path):
    from PIL import Image

    (tmp_path / "stub").mkdir()
    angles = tuple(float(a) for a in range(-90, 91, 10))
    for angle in angles:
        Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(
            tmp_path / f"stub/{angle:g}.png")
    figure = CF.fig_q13_rotation_strip(_strip_images(angles=angles), tmp_path,
                                       target="ordinary_mean", max_columns=7)
    pictures = [ax for ax in figure.axes if ax.images]
    assert len(pictures) == 7, f"{len(pictures)} columns for max_columns=7"
    outlined = [ax for ax in pictures if "unmodified" in ax.get_title()]
    assert len(outlined) == 1, [ax.get_title() for ax in pictures]
    # theta = 0 is kept when the sweep is thinned: it is the numerical floor.
    assert "0" in outlined[0].get_title()
    plt.close(figure)


# ============================================ the rotation arm across two devices
# The plane is built once per (step, layer) from the CLEAN probe states, which the tracer
# keeps on the CPU; the edit applies it to the LIVE tensor on the model's device. Every
# matmul between them crosses that boundary, and this suite has no GPU to exercise it
# end to end, so these two tests pin the structure that keeps it correct instead.
def test_every_rotation_helper_takes_its_basis_through_one_device_resolver():
    """A lint-style test: it checks structure, not a computed value.

    A cross-device matmul is not a wrong number, it is a crash on hardware this suite
    does not have, so no CPU test can exercise it end to end. What CAN be pinned is the
    structure: every helper that multiplies a state by a plane vector must obtain that
    vector from `_basis_like`, which is the one place the device move happens. A new
    helper that reaches into `plane["v"]` itself would reintroduce exactly the bug.
    """
    import inspect

    for function in (CS._rotation_note, CS._plane_coordinates, CS._plane_shares):
        source = inspect.getsource(function)
        assert "_basis_like(" in source, \
            f"{function.__name__} does not route its basis through _basis_like"
        assert 'plane["v"].float()' not in source and 'plane["u"].float()' not in source, \
            f"{function.__name__} uses a plane vector without moving it to the states"
    # The two that resolve their own basis inline, and must still do the move.
    for function in (CS.rotate_plane, CS._angle_from_vstar_deg):
        source = inspect.getsource(function)
        assert "device=" in source, \
            f"{function.__name__} never names a device, so it cannot be moving the basis"


def test_the_rotation_survives_states_on_a_different_device_from_the_plane():
    """Exercised with meta tensors, which reproduce the real error class on a CPU box:
    a matmul between a meta tensor and a cpu one raises the same device mismatch that
    cuda-against-cpu does."""
    width = 64
    v = torch.randn(width)
    u = CS.orthogonal_direction(CS._unit(v), width, seed=0)
    elsewhere = torch.randn(6, width, device="meta")     # the 'live' tensor
    assert v.device.type == "cpu" and elsewhere.device.type != "cpu"

    # The resolver puts the basis where the states are, which is the whole fix.
    moved_v, moved_u = CS._basis_like(v, u, elsewhere)
    assert moved_v.device == elsewhere.device and moved_u.device == elsewhere.device
    assert moved_v.dtype == torch.float32

    # And the two helpers that do not need .item() run straight through.
    assert CS.rotate_plane(elsewhere, v, u, 0.5).device == elsewhere.device
    assert CS._angle_from_vstar_deg(elsewhere, v).device == elsewhere.device
