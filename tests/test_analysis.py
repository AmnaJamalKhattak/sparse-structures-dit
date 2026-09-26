"""Tests for the metrics, the v* fit, and the figures.

The mock result plants a known ground truth, registers born at layer 17, sinks
from layer 19, one dominant channel, one shared direction, so these assert that
the analysis *recovers* it, not merely that it runs.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pytest
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ditsinks import SweepConfig, run_sweep, style as st       # noqa: E402
from ditsinks import figures as F                              # noqa: E402
from ditsinks import metrics as M                              # noqa: E402
from ditsinks.mock import make_mock_result                     # noqa: E402
from ditsinks.runner import run_projection_sweep               # noqa: E402
from ditsinks.vstar import (attach_exact_selection,            # noqa: E402
                            exact_selection_table, fit_vstar)

BIRTH, SINK_LAYER, DECAY, CHANNEL = 17, 19, 40, 154


@pytest.fixture(scope="module")
def mock():
    st.use("light")
    return make_mock_result(birth_layer=BIRTH, sink_layer=SINK_LAYER, decay_layer=DECAY,
                            register_channel=CHANNEL, focus_layers=(30,))


@pytest.fixture(scope="module")
def mock_table(mock):
    return M.layer_table(mock)


# ----------------------------------------------------------------- definitions
def test_highnorm_and_massive_masks():
    norms = torch.tensor([1.0, 1.0, 1.0, 1.0, 30.0])
    assert M.highnorm_mask(norms, 3.0).tolist() == [False] * 4 + [True]
    ch = torch.tensor([1.0, 1.0, 50.0])
    assert M.massive_channel_mask(ch, 10.0).tolist() == [False, False, True]


def test_expected_jaccard_is_small_for_large_n():
    assert M.expected_jaccard(10, 1024) < 0.06
    assert np.isnan(M.expected_jaccard(0, 1024))


# ------------------------------------------------------------- recovering natural-register removal
def test_layer_table_recovers_the_planted_layers(mock_table):
    v = M.sink_layer_verdict(mock_table, threshold=10.0)
    hn = v[v["has_highnorm"]]["layer"]
    sk = v[v["has_sinks"]]["layer"]
    mc = v[v["has_massive_channels"]]["layer"]
    assert int(hn.min()) == BIRTH, "high-norm tokens should first appear at the planted birth layer"
    assert int(sk.min()) == SINK_LAYER, "sinks should first appear at the planted sink layer"
    assert int(mc.min()) == BIRTH
    # before the birth layer nothing stands out
    early = mock_table[mock_table["layer"] < BIRTH]
    assert float(early["norm_max_over_median"].max()) < 3.0
    assert float(early["sink_ratio_headmax"].max()) < 10.0


def test_sink_layers_couple_to_highnorm_tokens(mock_table):
    sink_rows = mock_table[mock_table["sink_ratio_headmax"] >= 10.0]
    assert not sink_rows.empty
    assert (sink_rows["jaccard_topk"] > sink_rows["jaccard_topk_chance"]).all()
    # In the plateau (before the planted decay) every head should sink on a
    # high-norm token; after the decay the registers fade while the sinks persist,
    # which is exactly the dissociation the figures are meant to show.
    plateau = sink_rows[sink_rows["layer"].between(SINK_LAYER, DECAY)]
    assert float(plateau["heads_sinking_on_highnorm"].mean()) > 0.95


def test_answer_text_names_the_right_layers(mock):
    text = M.answer_where_are_the_sinks(mock)
    assert f"L{SINK_LAYER}" in text
    assert f"L{BIRTH}" in text
    assert "attention sinks" in text and "massive activation chans" in text


def test_layer_value_matrix_shapes(mock):
    mat, layers = M.layer_value_matrix(mock, "norm", step=3)
    assert mat.shape[0] == len(layers) == mock.meta["n_layers"]
    mat_c, _ = M.layer_value_matrix(mock, "channel", step=3)
    assert mat_c.shape[1] >= 3072
    with pytest.raises(KeyError):
        M.layer_value_matrix(mock, "nope")


def test_head_table_flags_highnorm_sinks(mock):
    ht = M.head_table(mock)
    plateau = ht[ht["layer"].between(SINK_LAYER + 2, DECAY)]
    assert float(plateau["sink_is_highnorm"].mean()) > 0.95
    assert float(plateau["sink_token_norm_rank"].median()) <= 3


# ------------------------------------------------------------- recovering Q2
def test_vstar_recovers_the_planted_direction(mock):
    rep = fit_vstar(mock)
    planted = mock.meta["planted_direction"]
    cos = float(abs(rep.v @ planted) / (rep.v.norm() * planted.norm()))
    assert cos > 0.98, f"v* should recover the planted direction, got cos={cos:.3f}"
    assert rep.explained_variance > 0.8
    assert rep.top_channel == CHANNEL
    assert rep.massive_channel == CHANNEL
    assert rep.massive_channel_rank_in_vstar == 1


def test_vstar_separates_registers_from_ordinary_tokens(mock):
    rep = fit_vstar(mock)
    assert np.abs(rep.cos_registers).mean() > 0.8
    assert np.abs(rep.cos_controls).mean() < 0.2
    assert np.abs(rep.cos_random).mean() < 0.2


def test_vstar_is_prompt_and_seed_invariant(mock):
    rep = fit_vstar(mock)
    assert len(rep.condition_labels) == 4
    iu = np.triu_indices(rep.condition_cos.shape[0], k=1)
    assert float(rep.condition_cos[iu].min()) > 0.95


def test_vstar_summary_mentions_every_claim(mock):
    text = fit_vstar(mock).summary()
    for fragment in ("one direction?", "sign consistency", "top channel",
                     "massive channel", "direction vs magnitude"):
        assert fragment in text


def test_vstar_falls_back_loudly_when_nothing_is_high_norm():
    cfg = SweepConfig(model="tiny-flux1", prompts=["a"], seeds=[0], num_inference_steps=2,
                      capture_steps=[1], height=128, width=128)
    res = run_sweep(cfg, progress=False)
    rep = fit_vstar(res)
    assert rep.meta["fitted_on_highnorm"] is False
    assert "CAVEAT" in rep.summary()


def test_exact_projection_pass_matches_the_sweep(mock):
    cfg = SweepConfig(model="tiny-pixart", prompts=["a"], seeds=[0], num_inference_steps=2,
                      capture_steps=[1], height=128, width=128)
    res = run_sweep(cfg, progress=False)
    rep = fit_vstar(res)
    proj = run_projection_sweep(cfg, rep.v, progress=False)
    # the projection pass must see the same tokens the sweep did
    for key, rows in proj["rows"].items():
        rec = res.records[key]
        assert rows["proj"].numel() == rec.n_img
        assert torch.allclose(rows["norm"], rec.norms["post_block"], atol=1e-3)
    df = exact_selection_table(res, proj["rows"])
    assert not df.empty and {"hit_norm", "hit_direction", "hit_chance"} <= set(df.columns)
    rep2 = attach_exact_selection(rep, res, proj["rows"])
    assert rep2.coverage == 1.0 and rep2.meta["selection_is_exact"]


# ---------------------------------------------------------------- the figures
@pytest.mark.parametrize("quantity", ["norm", "attention", "attention_ratio", "channel"])
def test_distribution_figures(mock, quantity, tmp_path):
    fig = F.fig_layer_distribution(mock, quantity, path=tmp_path / f"{quantity}.png")
    assert (tmp_path / f"{quantity}.png").exists()
    plt.close(fig)


def test_median_normalised_distribution_centres_the_bulk(mock, tmp_path):
    fig = F.fig_layer_distribution(mock, "norm", normalize="median", path=tmp_path / "n.png")
    ax = fig.axes[0]
    assert ax.get_yscale() == "log"
    assert "median" in ax.get_ylabel()
    plt.close(fig)
    with pytest.raises(ValueError):
        F.fig_layer_distribution(mock, "norm", normalize="bogus")


def test_atlas_stacks_one_density_panel_per_phenomenon(mock, tmp_path):
    """The atlas replicates the layer-wise distribution form, once per phenomenon."""
    fig = F.fig_layer_atlas(mock, path=tmp_path / "atlas.png")
    data_axes = [a for a in fig.axes if a.collections and a.get_label() != "<colorbar>"]
    assert len(data_axes) == 3
    titles = [a.get_title(loc="left") for a in data_axes]
    assert titles == ["High-norm tokens", "Attention sinks", "Massive activation channels"]
    for a in data_axes:
        assert a.get_yscale() == "log"
        assert a.get_ylabel(), "every panel needs its quantity on the y axis"
    assert data_axes[-1].get_xlabel() == "Layer index"
    plt.close(fig)


def test_atlas_can_show_absolute_units(mock, tmp_path):
    """The absolute-unit variant plots the measured values, with the bulk drawn on top."""
    fig = F.fig_layer_atlas(mock, normalize="none", path=tmp_path / "abs.png")
    data_axes = [a for a in fig.axes if a.collections and a.get_label() != "<colorbar>"]
    assert len(data_axes) == 3
    assert [a.get_ylabel() for a in data_axes] == [
        "Norm value", "Incoming attention", "Peak activation value"]
    for a in data_axes:
        assert a.lines, "the per-layer median should be drawn in absolute units"
    assert "absolute units" in fig._suptitle.get_text()
    # the norm panel must span the real norm range, not a ratio around 1
    ymin, ymax = data_axes[0].get_ylim()
    assert ymax > 100, f"absolute panel looks normalised (ymax={ymax})"
    plt.close(fig)
    with pytest.raises(ValueError):
        F.fig_layer_atlas(mock, normalize="bogus")


def test_summary_has_three_series_and_a_threshold_raster(mock, tmp_path):
    fig = F.fig_layer_summary(mock, path=tmp_path / "summary.png")
    main = fig.axes[0]
    labels = [t.get_text() for t in main.get_legend().get_texts()]
    assert labels == ["High-norm tokens", "Attention sinks", "Massive activation channels"]
    assert main.get_yscale() == "log"
    assert len(fig.axes) == 2, "the threshold raster should be its own axes"
    plt.close(fig)


def test_other_layer_figures(mock, tmp_path):
    for fn, name in [(F.fig_layer_panels, "panels"), (F.fig_layer_head_sinks, "heads"),
                     (F.fig_text_vs_image_sink, "text")]:
        fig = fn(mock, path=tmp_path / f"{name}.png")
        assert (tmp_path / f"{name}.png").exists()
        plt.close(fig)


def test_head_grid_and_sink_profile(mock, tmp_path):
    fig = F.fig_attention_head_grid(mock, 30, path=tmp_path / "grid.png")
    plt.close(fig)
    fig = F.fig_sink_profile(mock, 30, path=tmp_path / "prof.png")
    plt.close(fig)
    with pytest.raises(ValueError, match="focus_layers"):
        F.fig_attention_head_grid(mock, 5)


def test_vstar_figures(mock, tmp_path):
    rep = fit_vstar(mock)
    fig = F.fig_vstar_identity(rep, mock, path=tmp_path / "vstar.png")
    texts = {t.get_text() for a in fig.axes for t in a.texts}
    assert {"(a)", "(b)", "(c)", "(d)", "(e)", "(f)"} <= texts, "panel labels are missing"
    titles = [a.get_title() for a in fig.axes]
    assert "Coordinates of $v^*$" in titles and "Emergence with depth" in titles
    plt.close(fig)
    fig = F.fig_vstar_channel_bridge(rep, mock, layer=30, path=tmp_path / "bridge.png")
    plt.close(fig)


def test_model_comparison_figure(mock, tmp_path):
    cfg = SweepConfig(model="tiny-pixart", prompts=["a"], seeds=[0], num_inference_steps=2,
                      capture_steps=[1], height=128, width=128)
    other = run_sweep(cfg, progress=False)
    fig = F.fig_model_comparison({"mock-flux1": mock, "tiny-pixart": other},
                                 path=tmp_path / "cmp.png")
    assert len(fig.axes) == 6
    plt.close(fig)


def test_theme_covers_every_phenomenon():
    for mode in ("light", "dark"):
        theme = st.use(mode)
        for phen in st.PHENOMENA:
            assert theme.color(phen).startswith("#")
            assert theme.cmap(phen) is not None
    st.use("light")


# ------------------------------------------------------------------- round trip
def test_result_save_and_load(mock, tmp_path):
    from ditsinks.runner import SweepResult

    p = mock.save(tmp_path / "r.pt")
    back = SweepResult.load(p)
    assert len(back.records) == len(mock.records)
    assert back.cfg.model == mock.cfg.model
    assert back.zone_boundary() == mock.zone_boundary()


# ------------------------------------------------- publication-style invariants
def _all_figures(mock, rep, tmp_path):
    """Every figure the notebook produces from an observational sweep."""
    builders = [
        ("distribution", lambda: F.fig_layer_distribution(mock, "norm")),
        ("atlas", lambda: F.fig_layer_atlas(mock)),
        ("summary", lambda: F.fig_layer_summary(mock)),
        ("panels", lambda: F.fig_layer_panels(mock)),
        ("sink_profile", lambda: F.fig_sink_profile(mock, 30)),
        ("head_sinks", lambda: F.fig_layer_head_sinks(mock)),
        ("text_vs_image", lambda: F.fig_text_vs_image_sink(mock)),
        ("vstar", lambda: F.fig_vstar_identity(rep, mock)),
        ("bridge", lambda: F.fig_vstar_channel_bridge(rep, mock, layer=30)),
        # carried over from v4
        ("spatial_maps", lambda: F.fig_spatial_token_maps(mock, 30)),
        ("spatial_overlap_maps", lambda: F.fig_spatial_overlap_maps(mock, 30)),
        ("rank_rank", lambda: F.fig_rank_rank(mock, 30)),
        ("norm_attention_curves", lambda: F.fig_norm_attention_curves(mock, 30)),
        ("norm_vs_attention", lambda: F.fig_norm_vs_attention_scatter(mock, 30)),
        ("overlap_by_stage", lambda: F.fig_overlap_by_stage(mock)),
        ("spearman", lambda: F.fig_spearman_heatmap(mock)),
        ("sink_rank_hist", lambda: F.fig_sink_rank_histogram(mock)),
        ("token_trajectories", lambda: F.fig_token_trajectories(mock)),
        ("percentile_sensitivity", lambda: F.fig_percentile_sensitivity(mock)),
        ("norm_ratio_substrate", lambda: F.fig_norm_ratio_substrate(mock)),
        ("head_sink_membership", lambda: F.fig_head_sink_membership(mock)),
        ("text_per_head", lambda: F.fig_text_attention_per_head(mock)),
        ("channel_landscape", lambda: F.fig_channel_landscape(mock)),
        ("vstar_projection", lambda: F.fig_vstar_projection_trajectory(rep, mock)),
        ("register_selection", lambda: F.fig_register_selection(mock)),
        ("qk_geometry", lambda: F.fig_qk_geometry(mock)),
        ("sparse_lifecycle", lambda: F.fig_sparse_lifecycle(mock, rep)),
    ]
    for name, build in builders:
        yield name, build()


def test_every_plot_labels_both_axes(mock, tmp_path):
    """Axis meaning belongs on the axis, never in a note under the figure."""
    rep = fit_vstar(mock)
    missing = []
    for name, fig in _all_figures(mock, rep, tmp_path):
        for ax in fig.axes:
            if not (ax.lines or ax.collections or ax.patches or ax.images):
                continue                                   # colorbar or spacer
            if ax.get_label() == "<colorbar>":
                continue
            shares_x = any(a is not ax and a.get_shared_x_axes().joined(ax, a)
                           for a in fig.axes)
            has_x = bool(ax.get_xlabel()) or bool(fig._supxlabel) or shares_x
            has_y = bool(ax.get_ylabel()) or bool(fig._supylabel)
            if not (has_x and has_y):
                missing.append((name, ax.get_title(), ax.get_xlabel(), ax.get_ylabel()))
        plt.close(fig)
    assert not missing, f"axes without labels: {missing}"


def test_axes_legends_never_cover_data(mock, tmp_path):
    """Information boxes must live in a reserved gutter, not over plotted marks."""
    rep = fit_vstar(mock)
    overlaps = []
    for name, fig in _all_figures(mock, rep, tmp_path):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for panel, ax in enumerate(fig.axes):
            legend = ax.get_legend()
            if legend is None:
                continue
            if ax.get_window_extent(renderer).overlaps(legend.get_window_extent(renderer)):
                overlaps.append((name, panel))
        plt.close(fig)
    assert not overlaps, f"legends covering plot areas: {overlaps}"


def test_qk_geometry_is_captured_as_a_compact_head_token_summary(mock):
    df = M.qk_geometry_table(mock)
    assert not df.empty
    assert {"register_key_rank", "register_key_topk", "register_key_cosine"} <= set(df.columns)
    assert df["register_key_rank"].between(1, df["n_image_keys"]).all()
    for rec in mock.records.values():
        assert rec.qk_mean_cosine.shape == (rec.n_heads, rec.n_img)


def test_dense_paper_figures_reserve_readable_text_space(mock):
    from matplotlib.text import Text

    rep = fit_vstar(mock)
    qk = F.fig_qk_geometry(mock)
    identity = F.fig_vstar_identity(rep, mock)
    assert tuple(qk.get_size_inches()) == pytest.approx((11.8, 3.8))
    assert tuple(identity.get_size_inches()) == pytest.approx((13.5, 8.0))
    assert max(len(ax.get_xticks()) for ax in qk.axes[:3]) <= 8
    assert qk._suptitle.get_fontweight() == "bold"
    assert identity._suptitle.get_fontweight() == "bold"
    assert all(ax.title.get_fontweight() == "bold" for ax in qk.axes[:3])
    assert all(ax.title.get_fontweight() == "bold" for ax in identity.axes if ax.get_title())
    for fig in (qk, identity):
        nonempty_text = [item for item in fig.findobj(Text) if item.get_text().strip()]
        assert nonempty_text
        assert all(item.get_fontweight() == "bold" for item in nonempty_text)
    plt.close(qk)
    plt.close(identity)


def test_no_figure_carries_an_explanatory_footnote(mock, tmp_path):
    """Method notes go in fig.dv_caption for the LaTeX caption, not onto the canvas."""
    rep = fit_vstar(mock)
    for name, fig in _all_figures(mock, rep, tmp_path):
        body = [t for t in fig.texts if t.get_position()[1] < 0.06]
        assert not body, f"{name} draws a footnote at {[t.get_text() for t in body]}"
        assert getattr(fig, "dv_caption", ""), f"{name} has no suggested caption"
        assert len(fig.dv_caption) > 80
        plt.close(fig)


def test_publication_typography_is_installed():
    import matplotlib as mpl

    st.use("paper")
    assert mpl.rcParams["font.family"] == ["serif"]
    assert "STIXGeneral" in mpl.rcParams["font.serif"]
    assert mpl.rcParams["font.weight"] == "bold"
    assert mpl.rcParams["mathtext.fontset"] == "stix"
    assert mpl.rcParams["mathtext.default"] == "bf"
    assert mpl.rcParams["axes.labelweight"] == "bold"
    assert mpl.rcParams["axes.titleweight"] == "bold"
    assert mpl.rcParams["figure.labelweight"] == "bold"
    assert mpl.rcParams["figure.titleweight"] == "bold"
    assert mpl.rcParams["xtick.direction"] == "in"
    assert mpl.rcParams["savefig.dpi"] >= 300
    with pytest.raises(ValueError):
        st.use("neon")
    st.use("paper")


def test_series_palette_is_colour_vision_safe_and_fixed():
    """Okabe-Ito slots, one per phenomenon, never reassigned."""
    assert st.SERIES["highnorm"] == "#0072B2"
    assert st.SERIES["sink"] == "#D55E00"
    assert st.SERIES["channel"] == "#009E73"
    assert len({st.SERIES[k] for k in st.PHENOMENA}) == 3


def test_every_v4_figure_has_a_v5_equivalent(mock, tmp_path):
    """The v4 notebook's observational figures all exist here."""
    rep = fit_vstar(mock)
    produced = {name for name, fig in _all_figures(mock, rep, tmp_path)}
    plt.close("all")
    v4_observational = {
        "attention_heatmaps": "head_sinks",            # per-head maps: see also fig_attention_head_grid
        "spatial_maps": "spatial_maps",
        "rank_rank": "rank_rank",
        "text_attention": "text_per_head",
        "norm_attention_curves": "norm_attention_curves",
        "scatter_norm_vs_attention": "norm_vs_attention",
        "overlap_summaries": "overlap_by_stage",
        "correlation_heatmaps": "spearman",
        "rank_histograms": "sink_rank_hist",
        "token_trajectories": "token_trajectories",
        "fig1_jaccard_vs_percentile": "percentile_sensitivity",
        "fig2_substrate_norm_ratio": "norm_ratio_substrate",
        "fig3_head_top1_in_highnorm": "head_sink_membership",
        "fig_vstar_identity": "vstar",
        "fig_birth_trajectory": "vstar_projection",
        "fig_selection": "register_selection",
    }
    missing = {v4: v5 for v4, v5 in v4_observational.items() if v5 not in produced}
    assert not missing, f"v4 figures without a v5 counterpart: {missing}"


def test_axis_labels_use_words_not_formulas(mock, tmp_path):
    """Axis labels name the quantity in plain terms; formulas belong in the caption."""
    rep = fit_vstar(mock)
    banned = ["\\max", "\\mathrm", "\\bar", "\\|", "\\cdot", "\\frac", "_{t", "_c$", "med_"]
    offenders = []
    for name, fig in _all_figures(mock, rep, tmp_path):
        labels = []
        for ax in fig.axes:
            labels += [ax.get_xlabel(), ax.get_ylabel()]
        labels += [t.get_text() for t in (fig._supxlabel, fig._supylabel) if t is not None]
        for text in labels:
            if any(b in text for b in banned):
                offenders.append((name, text))
        plt.close(fig)
    assert not offenders, f"formula axis labels: {offenders}"


# ------------------------------------------------------------- interventions
@pytest.fixture(scope="module")
def suite(tmp_path_factory):
    """Run the full intervention suite once on a synthetic model."""
    from ditsinks import interventions as IV

    st.use("paper")
    out_dir = tmp_path_factory.mktemp("interventions")
    cfg = SweepConfig(model="tiny-flux1", prompts=["a", "b"], seeds=[0, 1],
                      num_inference_steps=2, capture_steps=[1], height=128, width=128,
                      output_dir=str(out_dir), save_images=False)
    icfg = IV.InterventionConfig(base=cfg, target_ratio=1.2, max_targets=4,
                                 conditions=IV.ABLATION_SUITE + IV.MAGNITUDE_SWEEP_SUITE)
    out = IV.run_intervention_suite(icfg, progress=False)
    results = IV.load_intervention_results(out["root"])
    return dict(cfg=cfg, icfg=icfg, out=out, results=results)


def test_intervention_suite_runs_every_condition(suite):
    from ditsinks import interventions as IV

    expected = set(IV.ABLATION_SUITE + IV.MAGNITUDE_SWEEP_SUITE)
    assert set(suite["out"]["conditions"]) == expected
    head = suite["results"]["head"]
    assert set(head["condition"]) == expected
    assert not head.empty


def test_instrumentation_control_reproduces_the_baseline(suite):
    """If observation alone moves the sinks, no other condition can be trusted."""
    head = suite["results"]["head"]
    key = ["prompt_id", "seed", "step", "layer", "head_id"]
    base = head[head["condition"] == "baseline"].set_index(key)["sink_token"]
    sham = head[head["condition"] == "instrumentation_control"].set_index(key)["sink_token"]
    agreement = (base.reindex(sham.index) == sham).mean()
    assert agreement > 0.99, f"instrumentation control moved the sinks ({agreement:.3f})"


def test_edits_touch_the_intended_number_of_tokens(suite):
    log = suite["results"]["log"]
    per = log.groupby("condition")["n_edited"].mean()
    assert per["baseline"] == 0
    assert per["instrumentation_control"] > 0        # selects, but changes nothing
    # controls edit the same count as the ablation they control for
    assert per["magnitude_ablation_matched"] == per["magnitude_ablation"]
    assert per["magnitude_ablation_random"] == per["magnitude_ablation"]
    # a transfer moves one direction onto exactly one recipient
    for c in [c for c in per.index if c.startswith("direction_transfer")]:
        assert per[c] == 1, f"{c} edited {per[c]} tokens"
    tl = log[log["condition"] == "direction_transfer"]
    assert (tl["recipient_token"] >= 0).all() and (tl["source_token"] >= 0).all()
    assert (tl["recipient_token"] != tl["source_token"]).all()


def test_edit_hook_is_a_no_op_for_the_baseline():
    """The baseline must be bit-identical to running with no hook at all."""
    from ditsinks.adapters import get_adapter
    from ditsinks.interventions import CONDITIONS, InterventionConfig, TokenEditHook
    from ditsinks.synthetic import build_tiny, planted_direction

    bundle = build_tiny("flux1", steps=1, grid=6)
    v = planted_direction(bundle.d_model)
    adapter = get_adapter("flux1")
    cfg = SweepConfig(model="tiny-flux1", prompts=["a"], seeds=[0], num_inference_steps=1,
                      capture_steps=[0], height=128, width=128)

    def run(hook=None):
        grabbed = {}
        handles = [bundle.transformer.transformer_blocks[-1].register_forward_hook(
            lambda m, a, k, out: grabbed.__setitem__("o", out[1] if isinstance(out, tuple) else out),
            with_kwargs=True)]
        if hook is not None:
            handles.append(bundle.transformer.register_forward_pre_hook(
                hook.transformer_pre, with_kwargs=True))
            handles.append(bundle.transformer.single_transformer_blocks[0]
                           .register_forward_pre_hook(hook, with_kwargs=True))
        try:
            with torch.no_grad():
                bundle.call(bundle.transformer, 0, 0, v)
        finally:
            for h in handles:
                h.remove()
        return grabbed["o"].clone()

    clean = run()
    hook = TokenEditHook(InterventionConfig(base=cfg)).bind(adapter)
    hook.condition = CONDITIONS["instrumentation_control"]
    hook.begin_generation(0, 0)
    observed = run(hook)
    assert torch.equal(clean, observed), "the instrumentation control changed the forward pass"
    assert hook.log, "the hook never fired"


def test_intervention_analysis_tables(suite):
    from ditsinks import interventions as IV

    r = suite["results"]
    persistence = IV.sink_persistence(r)
    assert not persistence.empty and {"condition", "layer", "sink_unchanged"} <= set(persistence)
    assert persistence["sink_unchanged"].between(0, 1).all()
    assert not IV.head_agreement(r).empty
    assert not IV.norm_recovery(r).empty
    assert not IV.attention_reallocation(r).empty          # flux1 has text keys
    cap = IV.transfer_capture(r)
    assert not cap.empty and set(cap["which"]) == {"transfer", "baseline"}


def test_intervention_figures(suite, tmp_path):
    from ditsinks import interventions as IV

    r, root, cfg = suite["results"], suite["out"]["root"], suite["cfg"]
    rng = np.random.default_rng(0)
    from PIL import Image

    for d in sorted((root / "images").iterdir()):     # synthetic runs make no images
        for p in range(len(cfg.prompts)):
            for s in cfg.seeds:
                Image.fromarray((rng.random((32, 32, 3)) * 255).astype("uint8")).save(
                    d / f"prompt{p}_seed{s}.png")
    dist = IV.image_distances(root, cfg)
    tests = IV.paired_comparisons(dist)

    figs = [
        ("sink_persistence", F.fig_sink_persistence(r)),
        ("head_agreement", F.fig_head_agreement(r)),
        ("norm_recovery", F.fig_norm_recovery(r)),
        ("attention_reallocation", F.fig_attention_reallocation(r)),
        ("image_change", F.fig_image_change(dist, tests)),
        ("image_grid", F.fig_image_grid(root, cfg)),
        ("image_difference", F.fig_image_difference(root, cfg)),
        ("transfer_capture", F.fig_transfer_capture(r)),
        ("transfer_magnitude", F.fig_transfer_magnitude(r)),
    ]
    for name, fig in figs:
        assert getattr(fig, "dv_caption", ""), f"{name} has no caption"
        body = [t for t in fig.texts if t.get_position()[1] < 0.06 and t.get_text()]
        body = [t for t in body if t is not fig._supxlabel]
        assert not body, f"{name} draws a footnote"
        plt.close(fig)


def test_condition_labels_are_readable():
    """Figure and table labels must not expose the internal jargon."""
    from ditsinks.interventions import CONDITIONS, condition_label, condition_tick

    for key, cond in CONDITIONS.items():
        label = condition_label(key)
        assert label and label[0].isupper()
        for jargon in ("clamp", "sham", "scramble", "rotate", "seed_", "dose", "transplant"):
            assert jargon not in label.lower(), f"{key} label leaks jargon: {label}"
            assert jargon not in condition_tick(key).lower()
