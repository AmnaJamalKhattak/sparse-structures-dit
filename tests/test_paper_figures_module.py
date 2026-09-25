"""ditsinks.paper_figures: Figure 1 is drawn from one traced generation per model, and the
retiming grid from a Q16 unit folder and its frozen protocol."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from ditsinks import SweepConfig, paper_figures as P, q16_main as Q
from ditsinks.causal_engine import CausalTracer, GenerationDriver, run_traced_generation
from ditsinks.discovery import ChannelChoice, DiscoveryArtifact, LayerRanges
from ditsinks.questions import QuestionContext
from ditsinks.synthetic import build_tiny, planted_direction


def _context(model, tmp_path, *, steps=4):
    cfg = SweepConfig(model=model, prompts=["a"], seeds=[0], height=64, width=64,
                      num_inference_steps=steps, capture_steps=[0], dtype="float32",
                      output_dir=str(tmp_path), save_images=False, highnorm_ratio=1.2)
    driver = GenerationDriver(cfg)
    driver.bundle = build_tiny(cfg.spec.family, steps=steps, grid=4, n_dual=4, n_single=8)
    driver.transformer = driver.bundle.transformer
    driver.refs = driver.adapter.layers(driver.transformer)
    driver.grid, driver.n_img = driver.bundle.grid, driver.bundle.n_img
    driver.planted = planted_direction(driver.bundle.d_model)
    probe = CausalTracer(driver.adapter, driver.transformer,
                         direction=torch.ones(driver.bundle.d_model), layers=[5], steps=[0],
                         full_state_layers=[5], grid=driver.grid, cfg=cfg)
    fitted, _ = run_traced_generation(driver, probe, prompt_id=0, prompt="a", seed=0)
    states = fitted.at(0, 5).states
    loud = torch.argsort(states.norm(dim=-1), descending=True)[:3]
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


@pytest.fixture(scope="module", params=["tiny-flux1", "tiny-pixart"])
def ctx(request, tmp_path_factory):
    return _context(request.param, tmp_path_factory.mktemp(request.param))


def test_capture_reads_every_token_without_editing(ctx):
    cap = P.capture_teaser(ctx, prompt_id=0, prompt="a", seed=0, layer=5, step=0)
    n = cap.grid[0] * cap.grid[1]
    for name in ("norm", "channel_values", "attention", "cosine"):
        assert getattr(cap, name).shape == (n,), name
    assert np.all(np.abs(cap.cosine) <= 1 + 1e-5)
    assert 0.0 <= cap.channel_share <= 1.0
    assert cap.channel == ctx.dominant_channel
    # the same numbers the tracer gives an unedited run
    tracer = CausalTracer(ctx.driver.adapter, ctx.driver.transformer, direction=ctx.vstar,
                          layers=[5], steps=[0], channels=[cap.channel], grid=ctx.driver.grid,
                          cfg=ctx.cfg, n_img_hint=ctx.driver.n_img)
    trace, _ = run_traced_generation(ctx.driver, tracer, prompt_id=0, prompt="a", seed=0)
    np.testing.assert_allclose(cap.cosine, trace.at(0, 5).cosine.numpy(), atol=1e-6)
    np.testing.assert_allclose(cap.norm, trace.at(0, 5).norm.numpy(), rtol=1e-6)


def test_capture_round_trips(ctx, tmp_path):
    cap = P.capture_teaser(ctx, prompt_id=0, prompt="a", seed=0, layer=5, step=0)
    back = P.TeaserCapture.load(cap.save(tmp_path / "cap.npz"))
    for name in ("norm", "channel_values", "attention", "cosine"):
        np.testing.assert_allclose(getattr(back, name), getattr(cap, name))
    assert (back.checkpoint, back.layer, back.step, back.channel, back.grid) == \
        (cap.checkpoint, cap.layer, cap.step, cap.channel, cap.grid)
    assert back.channel_share == pytest.approx(cap.channel_share)


def _capture(norm, channel, attention, cosine, grid=(2, 3), checkpoint="flux1-dev"):
    return P.TeaserCapture(checkpoint=checkpoint, layer=20, step=14, prompt_id=0, seed=0,
                           prompt="p", channel=154, channel_share=0.99, grid=grid,
                           norm=np.asarray(norm, float), channel_values=np.asarray(channel, float),
                           attention=np.asarray(attention, float),
                           cosine=np.asarray(cosine, float))


def test_stats_are_computed_not_assumed():
    cap = _capture(norm=[1, 1, 1, 5, 1, 6], channel=[0, 0, 9, 8, 0, 7],
                   attention=[0.1, 0.1, 0.1, 0.3, 0.1, 0.3], cosine=[0.1, -0.2, 0.3, 0.99, 0.2, -0.98])
    stats = P.teaser_stats(cap, ratio=3.0)
    assert stats["high_norm_tokens"] == [3, 5]
    assert stats["n_shared"] == 1                   # token 2 outranks token 5 on the channel
    assert stats["high_norm_abs_cos_median"] == pytest.approx(0.985)
    assert stats["ordinary_abs_cos_median"] == pytest.approx(0.2)
    assert P._same_sentence(2, 1) == "1 of the 2 high-norm tokens leads all three"
    assert P._same_sentence(3, 2) == "2 of the 3 high-norm tokens lead all three"
    assert P._same_sentence(1, 1) == "the same token in all three"
    assert P._same_sentence(1, 0) == "the high-norm token does not lead all three"
    assert P._same_sentence(0, 0, ratio=2.5) == "no token reaches 2.5× the median norm"


def test_teaser_renders_both_models(ctx):
    caps = [P.capture_teaser(ctx, prompt_id=0, prompt="a", seed=0, layer=5, step=0)] * 2
    fig = P.fig_teaser(caps, ratio=1.2)
    assert fig.get_size_inches()[0] == pytest.approx(P.STACKED_WIDTH)
    assert len(fig.teaser_stats) == 2
    plt.close(fig)


def test_arrows_are_the_tokens_at_their_true_length_and_angle():
    """Row 2 draws every token once, from one origin to (component along v*, length of the
    rest), in median token norms, on equal axes shared by both panels."""
    from matplotlib.collections import LineCollection

    rng = np.random.default_rng(3)
    caps = []
    for checkpoint, n_high in (("flux1-dev", 5), ("pixart-sigma-1024", 1)):
        norm = rng.uniform(0.8, 1.2, 64)
        norm[:n_high] = rng.uniform(6, 12, n_high)
        cosine = rng.uniform(-0.1, 0.1, 64)
        cosine[:n_high] = rng.uniform(0.95, 0.99, n_high)
        caps.append(_capture(norm, rng.random(64), rng.random(64), cosine, grid=(8, 8),
                             checkpoint=checkpoint))
    fig = P.fig_teaser(caps, ratio=3.0)
    panels = [ax for ax in fig.axes if any(isinstance(c, LineCollection) for c in ax.collections)]
    assert len(panels) == 2
    scales = []
    for ax, cap, s in zip(panels, caps, fig.teaser_stats):
        along, across = P.along_across(cap)
        # the exact split: the legs of the right triangle over each token's norm
        ratio = cap.norm / np.median(cap.norm)
        np.testing.assert_allclose(np.hypot(along, across), ratio)
        np.testing.assert_allclose(along / ratio, cap.cosine)
        grey = next(c for c in ax.collections if isinstance(c, LineCollection))
        ends = np.array([seg[-1] for seg in grey.get_segments()])
        ordinary = np.setdiff1d(np.arange(64), s["high_norm_tokens"])
        np.testing.assert_allclose(ends, np.c_[along[ordinary], across[ordinary]])
        shafts = [line for line in ax.lines if line.get_alpha() != 0.0]
        tips = sorted(tuple(np.round(line.get_xydata()[-1], 9)) for line in shafts)
        want = sorted(tuple(np.round((along[t], across[t]), 9)) for t in s["high_norm_tokens"])
        assert tips == want
        # equal axes: one median token norm is as long across as it is up
        box = ax.get_position()
        w, h = box.width * fig.get_size_inches()[0], box.height * fig.get_size_inches()[1]
        x_units = np.diff(ax.get_xlim())[0] / w
        assert np.diff(ax.get_ylim())[0] / h == pytest.approx(x_units)
        scales.append(x_units)
        # every arrow below the band the key is printed in
        top = ax.get_ylim()[1]
        assert across.max() <= top - 0.25 * np.diff(ax.get_ylim())[0] + 1e-9
    assert scales[0] == pytest.approx(scales[1])             # one scale for both models
    assert fig.teaser_stats[0]["high_norm_angle_median"] == pytest.approx(
        np.degrees(np.arccos(fig.teaser_stats[0]["high_norm_abs_cos_median"])))
    plt.close(fig)


@pytest.fixture(scope="module")
def unit(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("q16unit")
    context = _context("tiny-flux1", tmp)
    settings = Q.MainSettings(checkpoint="tiny-flux1", size=64, steps=4, highnorm_ratio=1.2,
                              min_carriers=1, windows_per_side=1, sink_threshold=1.0,
                              rule_max_share_of_image=1.0)
    n = int(context.driver.n_layers)
    windows, notes = Q.plan_depth_windows(n_layers=n, formation=4, natural_end=7, decline=6,
                                          per_side=1, first_block=1, gap=1)
    protocol = Q.DepthProtocol(
        checkpoint="tiny-flux1", n_layers=n, seam=Q.seam_of(context), formation=4,
        natural_end=7, decline=6, selection_block=5, windows=windows, primary_early="E1",
        primary_late="L1", plateau=(5, 5), window_length=2, settings=settings.row(),
        notes=notes)
    catalog = Q.condition_catalog(protocol, include_check=False)
    Q.run_unit(context, protocol, prompt_id=0, prompt="a", seed=0, root=tmp,
               conditions=catalog, log=lambda *a, **k: None)
    # The toy model has no decoder, so the unit holds no pictures: give each condition one,
    # and an LPIPS for two of them, as a real unit's images/ and images.csv would.
    import pandas as pd
    from PIL import Image

    folder = Q.unit_directory(tmp, 0, 0)
    for i, c in enumerate(catalog):
        Image.new("RGB", (64, 64), (10 * i, 80, 120)).save(folder / "images" / f"{c.key}.png")
    table = pd.read_csv(folder / "images.csv")
    table["lpips"] = table["condition"].map({"induce_E1": 0.08, "remove": 0.52})
    table.to_csv(folder / "images.csv", index=False)
    return dict(root=tmp, protocol=protocol)


def test_condition_layers_come_from_the_protocol(unit):
    protocol = unit["protocol"]
    a, b = protocol.windows["E1"]
    assert P.condition_layers(protocol, "induce_E1") == P._span(a, b)
    assert P.condition_layers(protocol, "reference") == ""
    r0, r1 = protocol.windows["removal"]
    assert P.condition_layers(protocol, "remove").startswith(P._span(r0, r1))
    menu = P.available_conditions(protocol, unit["root"], (0, 0))
    assert [m["condition"] for m in menu][:2] == ["reference", "induce_E1"]


def test_retiming_grid_reads_a_unit(unit):
    conditions = ["reference", "induce_E1", "remove"]
    row = P.retiming_row_from_unit(unit["root"], unit["protocol"], prompt_id=0, seed=0,
                                   conditions=conditions, labels={"remove": "gone"},
                                   subtitle="3 register tokens", size=32)
    assert [p.key for p in row.panels] == conditions
    assert row.panels[2].label == "gone"
    assert [p.lpips for p in row.panels] == [None, 0.08, 0.52]
    assert row.panels[1].layers == P.condition_layers(unit["protocol"], "induce_E1")
    assert row.panels[0].image.shape == (32, 32, 3)          # downsampled to ``size``
    for layout in ("groups", "rows", "auto"):
        fig = P.fig_retiming_grid([row, row], layout=layout)
        assert fig.get_size_inches()[0] == pytest.approx(P.WIDTH)
        plt.close(fig)
    with pytest.raises(FileNotFoundError):
        P.retiming_row_from_unit(unit["root"], unit["protocol"], prompt_id=9, seed=9,
                                 conditions=conditions)
    n = P.unit_register_tokens(unit["root"], 0, 0)
    assert n is None or n >= 1
    assert P.unit_register_tokens(unit["root"], 9, 9) is None


def test_reference_check_reads_the_units_unmodified_image(unit):
    from PIL import Image

    folder = Q.unit_directory(unit["root"], 0, 0)
    with Image.open(folder / "images" / "reference.png") as image:
        same = np.asarray(image.convert("RGB"))
    cap = _capture([1] * 6, [0] * 6, [0.1] * 6, [0.1] * 6)
    cap.prompt_id, cap.seed = 0, 0
    cap.image = same.copy()
    check = P.reference_image_difference(cap, [unit["root"] / "missing", unit["root"]])
    assert check["mean_abs"] == 0.0 and check["path"].endswith("reference.png")
    cap.image = np.clip(same.astype(int) + 10, 0, 255).astype(np.uint8)
    assert P.reference_image_difference(cap, [unit["root"]])["mean_abs"] > 0
    cap.seed = 9                                           # no such unit
    assert P.reference_image_difference(cap, [unit["root"]]) is None


def test_caption_macros_are_the_figures_numbers(tmp_path):
    import shutil
    import subprocess

    flux = _capture(norm=[1, 1, 1, 5, 1, 6], channel=[0, 0, 9, 8, 0, 7],
                    attention=[0.1, 0.1, 0.1, 0.3, 0.1, 0.3],
                    cosine=[0.1, -0.2, 0.3, 0.99, 0.2, -0.98])
    pixart = _capture(norm=[1, 1, 1, 1, 1, 9], channel=[0, 0, 0, 0, 0, 7],
                      attention=[0.1, 0.1, 0.1, 0.1, 0.1, 0.5],
                      cosine=[0.1, -0.2, 0.3, 0.1, 0.2, 0.97], checkpoint="pixart-sigma-1024")
    pixart.channel, pixart.channel_share = 293, 0.912
    stats = [P.teaser_stats(c, ratio=3.0) for c in (flux, pixart)]
    tex = P.latex_teaser_macros(stats)
    assert "\\@namedef{qfig@flux@highnorm}{2}" in tex
    assert "\\@namedef{qfig@flux@shared}{1}" in tex
    assert "\\@namedef{qfig@flux@coshigh}{0.985}" in tex
    assert "\\@namedef{qfig@pixart@channel}{293}" in tex
    assert "\\@namedef{qfig@pixart@share}{91.2\\%}" in tex
    assert "\\@namedef{qfig@pixart@highnorm}{1}" in tex
    assert "\\@namedef{qfig@flux@when}{step 15}" in tex        # no sampler length recorded
    assert P.step_name(27, 28) == "final step" and P.step_name(4, 28) == "step 5 of 28"
    assert tex.count("\\makeatletter") == 1 and tex.rstrip().endswith("\\makeatother")
    path = P.write_teaser_tex(stats, tmp_path)
    assert path.name == "teaser_numbers.tex" and path.read_text() == tex
    if shutil.which("pdflatex") is None:
        pytest.skip("pdflatex is not installed")
    (tmp_path / "t.tex").write_text(
        "\\documentclass{article}\\begin{document}\\input{teaser_numbers.tex}"
        "[\\figone{flux}{highnorm}|\\figone{pixart}{share}|\\figone{flux}{nothing}]"
        "\\end{document}\n")
    subprocess.run(["pdflatex", "-interaction=nonstopmode", "t.tex"], cwd=tmp_path,
                   check=True, capture_output=True)
    log = (tmp_path / "t.log").read_text(errors="replace")
    assert "Undefined control sequence" not in log


def _population():
    """A Q13 rotation population: layer 18 edited, layers 20-23 the register zone, and
    layer 19 outside it; retention falls with the angle."""
    import pandas as pd
    rows = []
    for target in ("ordinary_mean", "random_orthogonal"):
        for theta in (5.0, 15.0):
            for prompt in (0, 1):
                for layer in range(16, 24):
                    rows.append(dict(
                        arm="rotation", rotation_target=target, theta_deg=theta,
                        prompt_id=prompt, seed=0, layer=layer,
                        is_edited_layer=layer == 18, is_register_zone=layer >= 20,
                        sink_retention=1.0 - theta / 20 - prompt / 10 + (layer - 20) / 100,
                        selected_sink_strength_mean=100.0 - theta))
    rows.append(dict(rows[0], arm="depth"))
    return pd.DataFrame(rows)


def test_rotation_table_reads_the_register_zone_after_the_edit():
    table = P.rotation_retention_table(_population())
    assert len(table) == 2 * 2 * 2
    assert set(table["first_layer"]) == {20} and set(table["last_layer"]) == {23}
    assert set(table["edit_layer"]) == {18}
    row = table[(table["rotation_target"] == "ordinary_mean") & (table["theta_deg"] == 15.0)
                & (table["prompt_id"] == 1)].iloc[0]
    # mean over layers 20-23 of 1 - 15/20 - 1/10 + (layer - 20)/100
    assert row["sink_retention"] == pytest.approx(0.15 + 0.015)
    assert row["sink_strength"] == pytest.approx(85.0)


def test_rotation_table_needs_an_edited_layer():
    population = _population()
    population["is_edited_layer"] = False
    with pytest.raises(ValueError, match="edited layer"):
        P.rotation_retention_table(population)


def test_direction_at_fixed_norm_draws_both_panels():
    import pandas as pd
    depth = pd.DataFrame([dict(layer=layer, beta=beta, sink_strength=s)
                          for layer in (18, 20, 24)
                          for beta, s in ((-0.25, 1.0), (0.0, 9.0), (1.0, 190.0))])
    rotation = P.rotation_retention_table(_population())
    fig = P.fig_direction_at_fixed_norm(depth, rotation)
    ax_a, ax_b = fig.axes
    assert len(ax_a.get_lines()) == 3 + 2      # one per layer, threshold, unmodified
    assert ax_a.get_yscale() == "log"
    assert [t.get_text() for t in ax_a.get_legend().get_texts()] == [
        "layer 18", "layer 20", "layer 24"]
    assert [t.get_text() for t in ax_b.get_legend().get_texts()] == [
        P.ROTATION_LABELS["random_orthogonal"], P.ROTATION_LABELS["ordinary_mean"]]
    assert ax_b.get_ylabel().endswith("layers 20–23")
    means = ax_b.get_lines()[0].get_ydata()
    assert np.all(np.diff(means) < 0)
    plt.close(fig)


def test_grid_headers_are_short_enough_for_a_panel(unit):
    protocol = unit["protocol"]
    key = f"remove_induce_{protocol.primary_early}"
    a, b = protocol.windows[protocol.primary_early]
    assert [P.condition_label(k) for k in ("reference", "remove", key)] == [
        "original", "removed", "induced"]
    assert P.condition_header(protocol, key) == P._span(a, b)
    assert P.condition_header(protocol, "remove") == "natural registers"
    for plain in ("induce_E1", "reference"):
        assert P.condition_header(protocol, plain) == P.condition_layers(protocol, plain)
    # the menu keeps the full description
    assert P.condition_layers(protocol, key).startswith(
        P._span(*protocol.windows["removal"]))


def test_a_long_label_is_shrunk_to_its_column(unit):
    import matplotlib as mpl

    from ditsinks import style

    row = P.retiming_row_from_unit(unit["root"], unit["protocol"], prompt_id=0, seed=0,
                                   conditions=["reference", "remove"], size=32,
                                   labels={"remove": "removed + early"})
    with mpl.rc_context():
        style.use("paper")      # a session that drew the other figures first (bold, 150 dpi)
        fig = P.fig_retiming_grid([row, row, row, row], layout="groups")
        renderer = fig.canvas.get_renderer()
        tile = (P.WIDTH - 3 * 0.16 - 4 * 0.04) / 8
        label = [t for t in fig.axes[0].texts if t.get_text() == "removed + early"][0]
        assert label.get_weight() in ("normal", 400)
        assert label.get_fontsize() < P.LABEL
        assert label.get_window_extent(renderer).width / fig.dpi <= tile + 0.02 + 1e-3
        plt.close(fig)


def test_figure_one_labels_its_marks_in_one_accent_colour():
    """Row 2 names v*, the high-norm tokens and the rest where they are drawn (no key), in
    blue and grey only, and nothing is cut off at the edge of the figure."""
    from matplotlib.collections import LineCollection
    from matplotlib.colors import same_color

    rng = np.random.default_rng(5)
    caps = []
    for checkpoint, n_high in (("flux1-dev", 5), ("pixart-sigma-1024", 1)):
        norm = rng.uniform(0.8, 1.2, 64)
        norm[:n_high] = rng.uniform(6, 12, n_high)
        cosine = rng.uniform(0.6, 0.8, 64)
        cosine[:n_high] = rng.uniform(0.97, 0.999, n_high)
        caps.append(_capture(norm, rng.random(64), rng.random(64), cosine, grid=(8, 8),
                             checkpoint=checkpoint))
    fig = P.fig_teaser(caps, ratio=3.0)
    texts = [t for ax in fig.axes for t in ax.texts if t.get_text()]
    words = [t.get_text() for t in texts]
    assert words.count("other tokens") == 2
    assert "high-norm tokens" in words and "high-norm token" in words
    assert words.count("$v^\\star$") == 2
    assert not any(ax.get_legend() for ax in fig.axes)
    assert not any("holds" in w for w in words)
    panels = [ax for ax in fig.axes if any(isinstance(c, LineCollection) for c in ax.collections)]
    for ax in panels:
        assert all(same_color(line.get_color(), P.VSTAR) for line in ax.lines)
        assert same_color(ax.spines["bottom"].get_edgecolor(), P.SOFT)   # axis is not blue
        grey = next(c for c in ax.collections if isinstance(c, LineCollection))
        assert same_color(grey.get_color()[0][:3], P.OTHER_TOKENS)
    renderer = fig.canvas.get_renderer()
    width, height = fig.get_size_inches() * fig.dpi
    for t in texts:
        box = t.get_window_extent(renderer)
        assert box.x0 >= -1 and box.x1 <= width + 1 and box.y0 >= -1 and box.y1 <= height + 1, \
            t.get_text()
    plt.close(fig)



def test_figure_one_stacks_the_models():
    """FLUX.1-dev's block over PixArt-Sigma's, each a row of maps over its row of arrows;
    the arrow panels aligned on one x axis; the text exactly that of the side-by-side
    layout."""
    from matplotlib.collections import LineCollection

    rng = np.random.default_rng(7)
    caps = []
    for checkpoint, n_high in (("flux1-dev", 4), ("pixart-sigma-1024", 1)):
        norm = rng.uniform(0.8, 1.2, 64)
        norm[:n_high] = rng.uniform(4, 12, n_high)
        cosine = rng.uniform(0.6, 0.8, 64)
        cosine[:n_high] = rng.uniform(0.97, 0.999, n_high)
        caps.append(_capture(norm, rng.random(64), rng.random(64), cosine, grid=(8, 8),
                             checkpoint=checkpoint))
    fig = P.fig_teaser(caps)
    assert fig.get_size_inches()[0] == pytest.approx(P.STACKED_WIDTH)
    panels = [ax for ax in fig.axes if any(isinstance(c, LineCollection) for c in ax.collections)]
    upper, lower = sorted(panels, key=lambda ax: -ax.get_position().y0)
    assert upper.get_xlim() == pytest.approx(lower.get_xlim())
    assert upper.get_position().x0 == pytest.approx(lower.get_position().x0)
    assert upper.get_position().width == pytest.approx(lower.get_position().width)
    assert all("median token norms" in ax.get_xlabel() for ax in (upper, lower))
    # every tile but the overlay the labels are drawn on (the toy captures have no photo)
    maps = sorted((ax for ax in fig.axes[1:] if ax not in panels),
                  key=lambda ax: -ax.get_position().y0)
    assert len(maps) == 8
    flux_maps, pixart_maps = maps[:4], maps[4:]
    # top to bottom: FLUX.1-dev maps, its arrows, PixArt-Sigma maps, its arrows
    assert min(ax.get_position().y0 for ax in flux_maps) > upper.get_position().y1
    assert upper.get_position().y0 > max(ax.get_position().y1 for ax in pixart_maps)
    assert min(ax.get_position().y0 for ax in pixart_maps) > lower.get_position().y1
    side = P.fig_teaser(caps, layout="columns")
    assert side.get_size_inches()[0] == pytest.approx(P.WIDTH)

    # the same words in both layouts: stacking moves the blocks, it does not reword them
    def words(figure):
        return sorted(t.get_text() for ax in figure.axes for t in ax.texts if t.get_text()) + \
            sorted(ax.get_xlabel() for ax in figure.axes if ax.get_xlabel())

    assert words(fig) == words(side)
    assert "image" in words(fig) and any(w.startswith("FLUX.1-dev  ·  layer") for w in words(fig))
    plt.close(fig)
    plt.close(side)
    with pytest.raises(ValueError):
        P.fig_teaser(caps, layout="rows")


def test_one_model_figures_share_the_scale_of_both():
    """Figure 1 (FLUX.1-dev) and the appendix figure (PixArt-Sigma) are drawn one model each
    but on the arrow scale of both, so their arrow lengths still compare."""
    from matplotlib.collections import LineCollection

    rng = np.random.default_rng(9)
    caps = []
    for checkpoint, n_high, top in (("flux1-dev", 4, 14), ("pixart-sigma-1024", 1, 5)):
        norm = rng.uniform(0.8, 1.2, 64)
        norm[:n_high] = rng.uniform(top - 1, top, n_high)
        cosine = rng.uniform(0.6, 0.8, 64)
        cosine[:n_high] = 0.99
        caps.append(_capture(norm, rng.random(64), rng.random(64), cosine, grid=(8, 8),
                             checkpoint=checkpoint))
    both = P.fig_teaser(caps)
    one = [P.fig_teaser([c], scale_with=caps) for c in caps]
    panel = lambda f: [ax for ax in f.axes
                       if any(isinstance(c, LineCollection) for c in ax.collections)]
    for f in one:
        (ax,) = panel(f)
        assert ax.get_xlim() == pytest.approx(panel(both)[0].get_xlim())
        assert f.get_size_inches()[0] == pytest.approx(P.STACKED_WIDTH)
        assert f.get_size_inches()[1] < both.get_size_inches()[1] / 1.8
    # one model across the full text width, at print size: bigger maps, the same words
    wide = P.fig_teaser([caps[0]], scale_with=caps, width=P.WIDTH)
    assert wide.get_size_inches()[0] == pytest.approx(P.WIDTH)
    assert sorted(t.get_text() for ax in wide.axes for t in ax.texts) == \
        sorted(t.get_text() for ax in one[0].axes for t in ax.texts)
    for f in [both, wide] + one:
        plt.close(f)
