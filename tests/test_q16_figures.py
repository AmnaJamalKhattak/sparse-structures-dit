"""Every figure renders from a pooled dataset shaped like the real one, and prints
only formal names."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pytest
from PIL import Image

from ditsinks import q16_figures as F
from ditsinks import q16_main as Q
from q16_synthetic_data import pooled


@pytest.fixture(scope="module")
def data():
    return pooled(n_prompts=4, seeds=(0, 42, 1234))


def _texts(fig):
    out = []
    for ax in fig.axes:
        out += [t.get_text() for t in ax.get_yticklabels() + ax.get_xticklabels()]
        out += [ax.get_xlabel(), ax.get_ylabel(), ax.get_title(), ax.get_title("left")]
        out += [t.get_text() for t in ax.texts]
        legend = ax.get_legend()
        if legend is not None:
            out += [t.get_text() for t in legend.get_texts()]
    for legend in fig.legends:
        out += [t.get_text() for t in legend.get_texts()]
    return [t for t in out if t]


@pytest.mark.parametrize("name", ["fig_design", "fig_image_effect_by_depth",
                                  "fig_attention_by_depth", "fig_writer_response",
                                  "fig_removal", "fig_specificity", "fig_threshold_range",
                                  "fig_boundaries"])
def test_every_figure_renders_with_formal_text_only(data, name):
    frames, protocols = data
    builders = {
        "fig_design": lambda: F.fig_design(protocols, frames["lifecycle"]),
        "fig_image_effect_by_depth": lambda: F.fig_image_effect_by_depth(
            frames["images"], protocols, iterations=100),
        "fig_attention_by_depth": lambda: F.fig_attention_by_depth(frames["lifecycle"],
                                                                   protocols),
        "fig_writer_response": lambda: F.fig_writer_response(frames["lifecycle"], protocols,
                                                             iterations=100),
        "fig_removal": lambda: F.fig_removal(frames["images"], frames["attention"], protocols,
                                             iterations=100),
        "fig_specificity": lambda: F.fig_specificity(frames["images"], protocols,
                                                     iterations=100),
        "fig_threshold_range": lambda: F.fig_threshold_range(frames["sensitivity"], protocols),
        "fig_boundaries": lambda: F.fig_boundaries(frames["boundaries"], protocols),
    }
    fig = builders[name]()
    assert fig is not None
    texts = _texts(fig)
    assert texts, "a figure with no text cannot be read"
    try:
        for text in texts:
            low = text.lower()
            assert "sham" not in low and "dironly" not in low, text
            assert "_" not in text, f"a code name reached the figure: {text!r}"
    finally:
        plt.close("all")


def test_the_image_figures_degrade_to_nothing_without_images(data):
    frames, protocols = data
    empty = frames["images"].iloc[0:0]
    assert F.fig_image_effect_by_depth(empty, protocols) is None
    assert F.fig_specificity(empty, protocols) is None


def test_make_all_figures_writes_pdf_png_and_captions(data, tmp_path):
    frames, protocols = data
    # Thumbnails for the example grid, as run_unit writes them.
    root = tmp_path / "flux1-dev"
    p = protocols["flux1-dev"]
    for prompt in range(4):
        folder = Q.unit_directory(root, prompt, 0) / "thumbnails"
        folder.mkdir(parents=True)
        for key in ["reference"] + [c.key for c in Q.condition_catalog(p)]:
            Image.new("RGB", (48, 48), (20 * prompt, 90, 160)).save(folder / f"{key}.jpg")
    written = F.make_all_figures(frames, protocols, tmp_path / "figures",
                                 root_by_model={"flux1-dev": root}, iterations=100)
    assert "fig_examples_flux1-dev" in written
    for paths in written.values():
        assert all(path.exists() and path.stat().st_size > 0 for path in paths)
    captions = (tmp_path / "figures" / "captions.md").read_text()
    assert "fig_image_effect_by_depth" in captions and "{" not in captions
    assert "sham" not in captions.lower()


# ------------------------------------------------------------------ the example grid
def _units(root, protocol, *, n_prompts=4, seeds=(0, 42), size=64, full=False):
    """Unit folders as run_unit writes them. Each condition changes one small patch, by an
    amount that grows with the prompt id and differs by seed, so 'most changed' is known."""
    import json
    import numpy as np

    keys = ["reference"] + [c.key for c in Q.condition_catalog(protocol)]
    for prompt in range(n_prompts):
        for seed in seeds:
            folder = Q.unit_directory(root, prompt, seed)
            (folder / "thumbnails").mkdir(parents=True)
            (folder / "images").mkdir()
            (folder / "unit.json").write_text(json.dumps(dict(prompt=f"prompt number {prompt}")))
            base = np.full((size, size, 3), 90, dtype=np.uint8)
            for i, key in enumerate(keys):
                array = base.copy()
                if key != "reference":
                    k = 2 + prompt * 3 + (seed == 42) + i % 3
                    array[40:40 + k, 10:10 + k] = 255
                Image.fromarray(array).save(folder / "thumbnails" / f"{key}.jpg", quality=95)
                if full:
                    Image.fromarray(array).resize((4 * size, 4 * size)).save(
                        folder / "images" / f"{key}.png")


def test_example_columns_presets_and_explicit_lists(data):
    _, protocols = data
    p = protocols["flux1-dev"]
    depth = F.example_columns(p, "depth")
    assert depth[0] == "reference" and depth[-1] == "remove"
    assert len(depth) == 2 + len(p.early_windows()) + len(p.late_windows()) + 1
    every = F.example_columns(p, "all")
    assert "hooks_only" not in every and len(every) == len(Q.condition_catalog(
        p, include_check=False))
    assert F.example_columns(p, ["remove", "induce_E1"]) == ["reference", "remove", "induce_E1"]
    assert len(F.example_columns(p, "key")) < len(every)
    with pytest.raises(ValueError):
        F.example_columns(p, ["induce_Q9"])


def test_the_examples_show_the_units_the_notebook_names(data, tmp_path):
    frames, protocols = data
    p = {"flux1-dev": protocols["flux1-dev"]}
    _units(tmp_path, p["flux1-dev"])
    # A pair is shown at its seed; a bare prompt id at its most-changed seed.
    figures = F.fig_examples({"flux1-dev": tmp_path}, p, None, units=[(2, 0), 1],
                             metric="pixel", detail=None)
    selection = figures["flux1-dev"].q16_selection
    assert list(zip(selection["prompt_id"], selection["seed"])) == [(2, 0), (1, 42)]
    assert "chosen by the authors" in figures["flux1-dev"].q16_caption
    plt.close("all")


def test_most_changed_picks_distinct_prompts_with_the_largest_change(data, tmp_path):
    frames, protocols = data
    p = {"flux1-dev": protocols["flux1-dev"]}
    _units(tmp_path, p["flux1-dev"], full=True)
    ranking = F.rank_example_units(None, "flux1-dev", F.example_columns(p["flux1-dev"]),
                                   root=tmp_path, metric="pixel")
    assert ranking["score"].is_monotonic_decreasing
    assert ranking.iloc[0]["prompt_id"] == 3 and ranking.iloc[0]["seed"] == 42
    figures = F.fig_examples({"flux1-dev": tmp_path}, p, None, n_rows=2, metric="pixel",
                             detail="both")
    fig = figures["flux1-dev"]
    assert list(fig.q16_selection["prompt_id"]) == [3, 2]
    assert "not as typical cases" in fig.q16_caption and "outlined" in fig.q16_caption
    for text in _texts(fig):
        assert "_" not in text, text
    plt.close("all")


def test_largest_columns_follow_the_measured_change(data, tmp_path):
    frames, protocols = data
    p = {"flux1-dev": protocols["flux1-dev"]}
    _units(tmp_path, p["flux1-dev"], n_prompts=4, seeds=(0, 42))
    figures = F.fig_examples({"flux1-dev": tmp_path}, p, frames["images"], conditions="largest",
                             n_columns=3, n_rows=2, seeds=[0], detail="crop")
    fig = figures["flux1-dev"]
    shown = [c for c in fig.q16_selection.columns
             if c.startswith(("induce", "remove", "extend", "control"))]
    # The synthetic effects are largest for relocation, removal and E3.
    assert set(shown) == {"remove_induce_E3", "remove", "remove_induce_L1"}
    assert set(fig.q16_selection["seed"]) == {0}
    plt.close("all")


def test_make_all_figures_passes_the_example_settings(data, tmp_path):
    frames, protocols = data
    root = tmp_path / "flux1-dev"
    _units(root, protocols["flux1-dev"])
    written = F.make_all_figures(frames, protocols, tmp_path / "figures",
                                 root_by_model={"flux1-dev": root}, iterations=50,
                                 examples=dict(units=[(0, 0), (3, 42)], conditions="key"))
    assert "fig_examples_flux1-dev" in written
    rows = (tmp_path / "figures" / "fig_examples_flux1-dev_rows.csv").read_text()
    assert "prompt number 3" in rows
    captions = (tmp_path / "figures" / "captions.md").read_text()
    assert "chosen by the authors" in captions


# ------------------------------------------------------------------ the guidance ablation
def _policies(frames):
    import numpy as np

    images = frames["images"]
    both = images[images["checkpoint"] == "pixart-sigma-1024"].copy()
    only = both.copy()
    early = only["condition"].str.startswith(("induce_E", "induce_N"))
    only.loc[early, "lpips"] = np.linspace(0.8, 0.95, int(early.sum()))
    return {"Conditional pass only": only, "Both passes": both}


def test_the_guidance_ablation_pairs_the_units_and_prints_formal_text(data):
    frames, protocols = data
    policies = _policies(frames)
    # A unit only one run holds is left out of both, so the comparison stays paired.
    policies["Both passes"] = policies["Both passes"][policies["Both passes"]["prompt_id"] > 0]
    fig = F.fig_guidance_ablation(policies, protocols["pixart-sigma-1024"], iterations=50)
    try:
        assert fig is not None
        texts = _texts(fig)
        assert "Conditional pass only" in texts and "Both passes" in texts
        assert all("_" not in t for t in texts), texts
        n_pairs = len(set(zip(policies["Both passes"]["prompt_id"],
                              policies["Both passes"]["seed"])))
        assert f"{n_pairs} prompt-seed pairs" in fig.q16_caption
    finally:
        plt.close("all")
    one = {"Both passes": policies["Both passes"]}
    assert F.fig_guidance_ablation(one, protocols["pixart-sigma-1024"]) is None


def test_make_all_figures_writes_the_guidance_ablation(data, tmp_path):
    frames, protocols = data
    written = F.make_all_figures(frames, protocols, tmp_path, iterations=50,
                                 ablation={"pixart-sigma-1024": _policies(frames)})
    assert "fig_guidance_ablation_pixart-sigma-1024" in written
    captions = (tmp_path / "captions.md").read_text()
    assert "multiplied by the guidance scale" in captions
    plt.close("all")


# ------------------------------------------------------------------ LaTeX output
def test_latex_macros_and_table_come_from_the_same_numbers(data):
    import re
    from ditsinks import q16_paper as P

    frames, protocols = data
    effects = Q.effect_table(frames["images"], metrics=("lpips",
                                                        "clip_prompt_similarity_change"),
                             iterations=50)
    late = Q.late_state_table(frames, protocols)
    tex = P.latex_macros(effects, late=late, protocols=protocols)
    row = effects[(effects["checkpoint"] == "flux1-dev") & (effects["condition"] == "induce_E3")
                  & (effects["metric"] == "lpips")].iloc[0]
    assert f"\\@namedef{{qret@flux@induce-E3@lpips}}{{{row['mean']:.3f}}}" in tex
    assert "qreg@pixart@blocks-E1" in tex and tex.count("\\makeatletter") == 1
    table = P.latex_retiming_table(effects, protocols, late)
    assert f"{row['mean']:.3f} [{row['ci_low']:.3f}, {row['ci_high']:.3f}]" in table
    # Balanced tabular: every body row has the declared number of columns.
    n_cols = 1 + 3 * len(protocols)
    body = [l for l in table.splitlines() if l.startswith("\\quad")]
    assert body and all(l.count(" & ") == n_cols - 1 for l in body)
    typeset = "\n".join(l for l in table.splitlines() if not l.lstrip().startswith("%"))
    assert "_" not in typeset.replace("\\_", ""), "raw underscore in typeset LaTeX"
