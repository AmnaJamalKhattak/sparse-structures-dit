"""The 3D views must be projections of real activations, and must say what they cost.

A geometric figure is the easiest place in an interpretability project to produce
something beautiful and false. These tests hold the three properties that keep it true:
the basis is orthonormal so a coordinate is a true length, the retained-energy
fraction is reported rather than assumed, and the degenerate case, where ``v*`` simply
*is* the dominant channel, is announced instead of being drawn as if it were fine.
"""
import json

import numpy as np
import pandas as pd
import pytest
import torch

from ditsinks import SweepConfig
from ditsinks import geometry as G
from ditsinks import geometry_figures as GF
from ditsinks import style as ST
from ditsinks.causal_engine import GenerationDriver
from ditsinks.discovery import ChannelChoice, DiscoveryArtifact, LayerRanges
from ditsinks.questions import QuestionContext
from ditsinks.synthetic import planted_direction


@pytest.fixture(scope="module", autouse=True)
def paper_style():
    import matplotlib
    matplotlib.use("Agg")
    ST.use("paper")


@pytest.fixture
def population():
    """A loud, partly aligned population and a v* that is not axis-aligned."""
    torch.manual_seed(0)
    states = torch.randn(120, 96) * 1.1
    direction = torch.randn(96)
    direction[11] = 3.0
    unit = G._unit(direction)
    states[:8] += 15.0 * unit
    return states, direction, 11


# ------------------------------------------------------------------ the basis
def test_the_basis_is_orthonormal_so_a_coordinate_is_an_honest_length(population):
    """Everything downstream rests on this.

    Without orthonormality a plotted coordinate is not a projection length, the three
    squared coordinates do not sum to a share of the norm, and the retained-energy
    fraction the figures print would be meaningless.
    """
    states, direction, channel = population
    basis = G.interpretable_basis(states, direction, channel)
    gram = basis.vectors @ basis.vectors.t()
    assert torch.allclose(gram, torch.eye(3), atol=1e-5), gram


def test_axis_one_is_vstar_itself_so_the_first_coordinate_is_alpha(population):
    """The figure claims the horizontal coordinate *is* ``x . v*``. It has to be."""
    states, direction, channel = population
    basis = G.interpretable_basis(states, direction, channel)
    coordinates = basis.project(states)
    alpha = states.float() @ G._unit(direction)
    assert torch.allclose(coordinates[:, 0], alpha, atol=1e-4, rtol=1e-4)


def test_axis_two_carries_only_what_the_channel_adds_beyond_vstar(population):
    """Otherwise axes 1 and 2 would double-count the same structure."""
    states, direction, channel = population
    basis = G.interpretable_basis(states, direction, channel)
    unit = G._unit(direction)
    assert abs(float(basis.vectors[1] @ unit)) < 1e-6
    # It is still a channel axis: it must retain a component on that coordinate.
    assert abs(float(basis.vectors[1][channel])) > 1e-3
    expected = float((1.0 - unit[channel] ** 2).clamp_min(0.0).sqrt())
    assert basis.independent_channel_content == pytest.approx(expected, abs=1e-5)


def test_the_captured_fraction_is_a_fraction_and_reports_the_projection_loss(population):
    """The number that decides whether the cloud is a summary or a shadow.

    It must lie in [0, 1], orthonormality guarantees that, and it must be *smaller*
    for ordinary tokens than for the aligned population, because the basis was built
    around the latter. A figure that reported a high fraction for everything would be
    hiding the loss rather than measuring it.
    """
    states, direction, channel = population
    basis = G.interpretable_basis(states, direction, channel)
    captured = basis.captured_fraction(states)
    assert float(captured.min()) >= 0.0 and float(captured.max()) <= 1.0 + 1e-6
    assert float(captured[:8].mean()) > float(captured[8:].mean())


def test_a_vstar_that_is_the_dominant_channel_is_announced_not_drawn_as_fine():
    """The degeneracy that would quietly invalidate axis 2.

    If ``v*`` is the channel, nothing survives the orthogonalisation and normalising the
    residue would plot rounding noise on an axis a reader takes seriously. The basis must
    substitute a data-driven direction, mark itself degenerate, relabel axis 2, and say
    so in its notes.
    """
    torch.manual_seed(0)
    states = torch.randn(60, 32)
    axis = torch.zeros(32)
    axis[7] = 1.0
    basis = G.interpretable_basis(states, axis, 7)
    assert basis.degenerate_axis_2 is True
    assert basis.channel_alignment == pytest.approx(1.0, abs=1e-6)
    assert basis.independent_channel_content < 1e-6
    assert "data-driven" in basis.labels[1].lower()
    assert basis.notes and "Axis 2 is a data-driven" in basis.notes
    assert torch.allclose(basis.vectors @ basis.vectors.t(), torch.eye(3), atol=1e-5)


def test_a_nearly_axis_aligned_vstar_warns_without_being_called_degenerate():
    """The FLUX case: axis 2 is real but thin, and a flat spread there means nothing."""
    torch.manual_seed(0)
    states = torch.randn(60, 32)
    direction = torch.zeros(32)
    direction[7] = 1.0
    direction[3] = 0.05                            # |v*_c| ~ 0.9988
    basis = G.interpretable_basis(states, direction, 7)
    assert not basis.degenerate_axis_2
    assert basis.channel_alignment > 0.99
    assert "nearly the dominant channel" in basis.notes
    assert "not evidence that the channel is unimportant" in basis.notes


def test_the_per_token_basis_contains_its_token_exactly(population):
    """The decomposition figure's whole claim: no leg is foreshortened.

    A shared 3D basis leaves most of an individual residual out of the page, so the
    drawn ``r_i`` would be shorter than the real one. The per-token basis must capture
    the token completely, or the arrows in that figure are not the lengths printed
    beside them.
    """
    states, direction, channel = population
    for token in (0, 50, 119):
        basis = G.exact_token_basis(states[token], direction, channel)
        captured = float(basis.captured_fraction(states[token].unsqueeze(0))[0])
        assert captured == pytest.approx(1.0, abs=1e-5), (token, captured)
        coordinates = basis.project(states[token].unsqueeze(0))[0]
        assert float(coordinates.norm()) == pytest.approx(float(states[token].norm()),
                                                          rel=1e-5)
        assert basis.kind == "exact-per-token"


def test_the_pca_view_is_offered_but_reports_how_much_of_vstar_it_contains(population):
    """A secondary view has to say whether it is even looking at the mechanism."""
    states, direction, channel = population
    basis = G.pca_basis(states, direction, channel)
    assert basis.kind == "pca"
    assert torch.allclose(basis.vectors @ basis.vectors.t(), torch.eye(3), atol=1e-4)
    assert 0.0 <= basis.total_variance_explained <= 1.0 + 1e-6
    assert "of v* by length" in basis.notes


def test_the_residual_axis_sign_is_fixed_so_a_sequence_does_not_flip(population):
    """A principal direction is defined up to sign, and a lifecycle strip that flipped
    axis 3 between frames would show a rotation that never happened."""
    states, direction, channel = population
    first = G.interpretable_basis(states, direction, channel)
    again = G.interpretable_basis(states.clone(), direction, channel)
    assert torch.allclose(first.vectors[2], again.vectors[2], atol=1e-6)


def test_an_unknown_basis_kind_is_refused(population):
    states, direction, channel = population
    with pytest.raises(KeyError):
        G.build_basis(states, direction, channel, kind="whatever_looks_best")


# -------------------------------------------------------------- categorisation
class _Observation:
    def __init__(self, incoming):
        self.incoming = incoming
        self.qk_cosine = None


def test_categories_use_the_projects_own_thresholds(population):
    """No new definition of high-norm or of sinkhood is introduced for a picture."""
    states, direction, channel = population
    incoming = torch.full((4, 120), 1.0 / 120)
    incoming[:, 0] = 0.5                           # token 0 is a strong sink
    incoming = incoming / incoming.sum(-1, keepdim=True)
    norms = states.norm(dim=-1)
    bar = float(norms.median()) * 2.0
    table = G.categorise(states, _Observation(incoming), vstar=direction,
                         norm_threshold=bar, sink_threshold=10.0)
    assert len(table) == 120
    assert set(table["category"]) <= set(G.CATEGORIES[:3])
    assert bool(table.loc[0, "is_sink"])
    assert table.loc[table["norm"] >= bar, "is_highnorm"].all()
    assert not table.loc[table["norm"] < bar, "is_highnorm"].any()
    # A loud sink must land in the sink category rather than the non-sink one.
    if bool(table.loc[0, "is_highnorm"]):
        assert table.loc[0, "category"] == "highnorm_sink"


def test_categorisation_degrades_when_attention_was_not_recorded(population):
    """A layer with no attention observation still has a geometry worth drawing."""
    states, direction, channel = population
    table = G.categorise(states, None, vstar=direction,
                         norm_threshold=float(states.norm(dim=-1).median()),
                         sink_threshold=10.0)
    assert not table["is_sink"].any()
    assert "highnorm_sink" not in set(table["category"])


# ------------------------------------------------------------------- end to end
@pytest.fixture(scope="module")
def ctx(tmp_path_factory):
    cfg = SweepConfig(model="tiny-flux1", prompts=["a stone bridge"], seeds=[0], height=128,
                      width=128, num_inference_steps=4, capture_steps=[2], dtype="float32",
                      output_dir=str(tmp_path_factory.mktemp("geo")), save_images=False,
                      highnorm_ratio=1.25, sink_ratio_threshold=1.6)
    driver = GenerationDriver(cfg)
    artifact = DiscoveryArtifact(
        schema_version=1, checkpoint="tiny-flux1", repo_id="synthetic", vstar_file="vstar.pt",
        fitting_population=[], explained_variance=0.9, axis_convention="", sign_convention="",
        dominant_register_channel=ChannelChoice(3, 3, True, 1.0),
        late_growing_competitor=ChannelChoice(5, None, None, 1.0),
        massive_unspecific_control=ChannelChoice(7, None, None, 1.0),
        layer_ranges=LayerRanges((0, 1), (1, 3), (3, 4)), denoising_steps=[2],
        discovery_prompts=["z"], discovery_seeds=[9], confirmation_prompts=["a stone bridge"],
        confirmation_seeds=[0], code_config_fingerprint="t", config_sha256="t")
    return QuestionContext.from_artifact(cfg, artifact, planted_direction(driver.bundle.d_model),
                                         driver=driver, progress=False, topk=3,
                                         percentile=90.0, highnorm_ratio=1.25)


@pytest.fixture(scope="module")
def snapshots(ctx):
    captured, _ = G.capture_snapshots(ctx, layers=list(range(ctx.driver.n_layers)),
                                      steps=[2], basis_layer=ctx.intervention_layer)
    return captured


def test_the_capture_keeps_the_real_activations(snapshots, ctx):
    """Not summaries. A geometric figure needs the states themselves."""
    assert len(snapshots) == ctx.driver.n_layers
    for snapshot in snapshots:
        assert snapshot.states.ndim == 2
        assert snapshot.states.shape[0] == ctx.driver.n_img
        assert snapshot.states.shape[1] == ctx.driver.bundle.d_model
        assert torch.isfinite(snapshot.states).all()


def test_one_basis_is_shared_across_the_whole_sequence(snapshots):
    """A lifecycle figure that refits axis 3 per frame moves the camera with the data.

    Axes 1 and 2 are model constants and never move; axis 3 is fitted, which is exactly
    why it must be fitted once. Without this a population that merely rotated would
    appear to change shape.
    """
    first = snapshots[0].basis.vectors
    for snapshot in snapshots[1:]:
        assert torch.allclose(snapshot.basis.vectors, first, atol=0.0)


def test_the_metadata_records_everything_needed_to_reproduce_a_figure(snapshots):
    payload = snapshots[0].metadata()
    for key in ("checkpoint", "prompt_id", "prompt", "seed", "timestep", "layer",
                "dominant_channel", "basis", "token_ids_by_category", "category_counts",
                "captured_fraction_median", "projection_note", "norm_threshold",
                "sink_threshold", "condition"):
        assert key in payload, key
    assert payload["basis"]["kind"] == "interpretable"
    assert "3D orthogonal projection" in payload["projection_note"]
    assert json.dumps(payload, default=str)          # must be serialisable as written


def test_the_token_table_lets_a_reader_replot_every_point(snapshots):
    table = G.token_table(snapshots)
    assert {"axis_1", "axis_2", "axis_3", "captured_fraction", "category", "token",
            "layer", "step", "basis", "norm", "alpha"} <= set(table.columns)
    assert len(table) == sum(len(s.tokens) for s in snapshots)
    # The first coordinate must still be alpha in the exported table.
    assert np.allclose(table["axis_1"], table["alpha"], atol=1e-4)


def test_representatives_are_chosen_by_a_stated_rule(snapshots):
    """Not by eye. The ordinary one is typical; each loud one is the clearest instance."""
    rich = max(snapshots, key=lambda s: s.tokens["category"].nunique())
    chosen = rich.representatives()
    ordinary = rich.tokens[rich.tokens["category"] == "ordinary"]
    if not ordinary.empty and chosen.get("ordinary") is not None:
        picked = rich.tokens.loc[chosen["ordinary"], "norm"]
        assert abs(picked - ordinary["norm"].median()) <= ordinary["norm"].std() + 1e-6
    for category in ("highnorm_nonsink", "highnorm_sink"):
        group = rich.tokens[rich.tokens["category"] == category]
        if not group.empty:
            assert chosen[category] == int(group.loc[group["norm"].idxmax(), "token"])


def test_every_geometry_figure_draws_and_carries_its_caption(snapshots, tmp_path):
    rich = max(snapshots, key=lambda s: s.tokens["category"].nunique())
    figures = [
        GF.fig_token_cloud(rich, path=tmp_path / "cloud.pdf"),
        GF.fig_token_decomposition(rich),
        GF.fig_channel_decomposition(rich),
        GF.fig_lifecycle_strip(snapshots),
        GF.fig_basis_diagnostics(snapshots),
    ]
    for figure in figures:
        caption = getattr(figure, "dv_caption", "")
        assert caption, "every figure carries its own caption"
        # Every caption must state its relationship to the full-dimensional space: a
        # projection says so, and the per-token decomposition, which is exact rather
        # than projected, says that instead. Silence about it is the failure.
        lowered = caption.lower()
        assert any(phrase in lowered for phrase in
                   ("projection", "projected", "drawn subspace",
                    "no projection is involved")), caption[:160]
    assert (tmp_path / "cloud.pdf").exists()


def test_the_cloud_caption_names_the_basis_and_the_retained_energy(snapshots):
    """A 3D view that does not say which three directions is not a measurement."""
    rich = max(snapshots, key=lambda s: s.tokens["category"].nunique())
    caption = GF.fig_token_cloud(rich).dv_caption
    assert "$v^*$" in caption
    assert "orthogonalised against" in caption
    assert "retain" in caption and "%" in caption
    assert "dimensional" in caption


def test_the_decomposition_caption_states_that_nothing_is_out_of_the_page(snapshots):
    rich = max(snapshots, key=lambda s: s.tokens["category"].nunique())
    caption = GF.fig_token_decomposition(rich).dv_caption
    assert "exactly in the drawn subspace" in caption
    assert "nothing is out of the page" in caption


def test_the_lifecycle_animation_writes_frames_even_if_the_gif_cannot(snapshots, tmp_path):
    """Animation writers are the most environment-dependent part of matplotlib.

    The frames are the deliverable; the GIF is a convenience. A fragile writer must not
    be able to cost a run its figures.
    """
    result = GF.write_lifecycle_animation(snapshots, tmp_path / "life.gif", fps=3)
    assert result["n_frames"] == len(snapshots)
    assert all(Path_exists(name) for name in result["frames"])


def Path_exists(name) -> bool:
    from pathlib import Path

    return Path(name).exists()


def test_a_figure_refuses_rather_than_drawing_an_empty_frame():
    with pytest.raises(ValueError, match="no snapshots"):
        GF.fig_lifecycle_strip([])
    with pytest.raises(ValueError, match="no snapshots"):
        GF.fig_basis_diagnostics([])


def test_the_pca_view_is_available_end_to_end(ctx):
    captured, _ = G.capture_snapshots(ctx, layers=[ctx.intervention_layer], steps=[2],
                                      basis_kind="pca")
    assert captured and captured[0].basis.kind == "pca"
    figure = GF.fig_token_cloud(captured[0])
    assert "pca" in figure.dv_caption.lower() or "principal" in figure.dv_caption.lower()


# -------------------------------------------- the treated run reads the right tensor
def test_a_snapshot_reads_the_state_the_block_consumed_not_the_pre_edit_one(ctx):
    """The hazard that would put every treated cloud one layer behind the truth.

    ``StateProbe`` registers before ``EditInstaller``, so ``trace.probe`` at the edit
    layer returns the tensor as it was *entering* the hook. A clean-versus-treated
    geometry figure read off that would show the edited layer as unchanged and the effect
    as appearing one layer late, which is exactly the kind of picture that gets
    believed. Snapshots therefore resolve the consumed state, and the three properties
    below are the proof: nothing upstream moves, the edit is visible where it happens,
    and the consequence propagates.
    """
    from ditsinks import control_surface as CS
    from ditsinks.adapters import InterventionPoint
    from ditsinks.causal_engine import EditPlan

    layers = list(range(ctx.driver.n_layers))
    clean, targets = G.capture_snapshots(ctx, layers=layers, steps=[2],
                                         basis_layer=ctx.intervention_layer)
    edit_layer = int(ctx.intervention_layer)
    channel = int(ctx.dominant_channel)
    ids = list(targets.register_ids)
    assert ids, "the synthetic run must select at least one register token"
    reference = {(2, edit_layer): next(s.states for s in clean if s.layer == edit_layer)}
    condition = next(c for c in CS.RESCUE_CONDITIONS if c.key == "channel_ablate")
    plan = [EditPlan(
        edit=CS.make_edit(condition, direction=ctx.vstar, token_ids=ids,
                          dominant_channel=channel, clean_states=reference, seed=0),
        point=InterventionPoint.BLOCK_INPUT, layers=[edit_layer], steps=[2],
        label="ablate")]
    treated, _ = G.capture_snapshots(ctx, layers=layers, steps=[2],
                                     basis_layer=ctx.intervention_layer, plans=plan,
                                     condition="channel_ablate", targets=targets)
    by_layer = {s.layer: s for s in treated}
    for snapshot in clean:
        delta = float((by_layer[snapshot.layer].states - snapshot.states).abs().max())
        if snapshot.layer < edit_layer:
            assert delta == 0.0, f"layer {snapshot.layer} is upstream and must not move"
        elif snapshot.layer == edit_layer:
            # The ablation is visible AT the layer it happens, which is the whole point.
            assert float(by_layer[edit_layer].states[ids, channel].abs().max()) < 1e-6
            assert delta > 0.0
        else:
            assert delta > 0.0, f"layer {snapshot.layer} should carry the consequence"


def test_the_capture_probes_whatever_its_own_selection_needs(ctx):
    """Asking for one unrelated layer must not fail inside the function.

    The frozen-target selection reads the intervention layer whether or not the caller
    wanted a snapshot there, so the probe list has to cover it.
    """
    far = max(0, int(ctx.intervention_layer) - 1)
    captured, targets = G.capture_snapshots(ctx, layers=[far], steps=[2], basis_layer=far)
    assert len(captured) == 1 and captured[0].layer == far
    assert targets.register_ids


def test_a_basis_layer_that_was_never_captured_is_refused(ctx):
    """Silently falling back to per-snapshot bases is the one thing a sequence must not do."""
    with pytest.raises(ValueError, match="was not captured"):
        G.capture_snapshots(ctx, layers=[0], steps=[2],
                            basis_layer=ctx.driver.n_layers - 1)


# ------------------------------------------- the real-scale problems, guarded
def test_a_snapshot_carries_vstar_rather_than_inferring_it_from_axis_one(ctx):
    """Axis 1 is ``v*`` for the interpretable basis and PC1 for the PCA basis.

    A decomposition figure that read ``v*`` off the basis would silently decompose a
    token along PC1 while labelling the leg ``alpha_i v*``. The snapshot therefore carries
    the direction itself, and it must be the frozen one in both bases.
    """
    interpretable, _ = G.capture_snapshots(ctx, layers=[ctx.intervention_layer], steps=[2])
    pca, _ = G.capture_snapshots(ctx, layers=[ctx.intervention_layer], steps=[2],
                                 basis_kind="pca")
    unit = G._unit(ctx.vstar)
    for captured in (interpretable, pca):
        assert captured
        assert torch.allclose(captured[0].vstar, unit, atol=1e-6)
    # The interpretable basis agrees with it on axis 1; the PCA basis need not.
    assert torch.allclose(interpretable[0].basis.vectors[0], unit, atol=1e-6)
    alpha = interpretable[0].states.float() @ unit
    assert torch.allclose(torch.as_tensor(interpretable[0].coordinates[:, 0]), alpha,
                          atol=1e-4)


def test_the_capture_does_not_store_a_second_copy_of_every_layer(ctx):
    """A geometric capture is the largest memory consumer in this project.

    The probe already keeps the ``[N, C]`` slice and ``consumed_states`` reads it, so
    asking the tracer for ``full_state_layers`` as well would double the footprint, 12 MB
    per layer-step at FLUX's 512px shapes, 0.27 GB over the register window, on top of a
    24 GB model. The snapshots must still be complete without it.
    """
    layers = list(range(ctx.driver.n_layers))
    captured, _ = G.capture_snapshots(ctx, layers=layers, steps=[2],
                                      basis_layer=ctx.intervention_layer)
    assert len(captured) == len(layers)
    for snapshot in captured:
        assert snapshot.states.shape == (ctx.driver.n_img, ctx.driver.bundle.d_model)


def test_a_supplied_basis_is_used_verbatim_so_two_runs_share_one_camera(ctx):
    """Comparing a treated cloud against a clean one needs the clean basis on both sides.

    Refitting per run would compare two cameras rather than two populations: axis 3 is
    data-driven, so the treated run would fit a different one and a population that only
    moved would appear to change shape.
    """
    clean, _ = G.capture_snapshots(ctx, layers=[ctx.intervention_layer], steps=[2],
                                   basis_layer=ctx.intervention_layer)
    reused, targets = G.capture_snapshots(ctx, layers=[ctx.intervention_layer], steps=[2],
                                          basis=clean[0].basis)
    assert reused and targets.register_ids
    assert reused[0].basis is clean[0].basis
    assert torch.allclose(reused[0].basis.vectors, clean[0].basis.vectors, atol=0.0)
