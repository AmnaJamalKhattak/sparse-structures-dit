"""Every causal figure is readable on paper: labelled axes, plain words, a caption."""
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pytest

from ditsinks import causal_figures as CF
from ditsinks import causal_stats as CS
from ditsinks import style as ST
from ditsinks.mock import make_mock_question_results

QUESTIONS = ("q1", "q2", "q3", "q4", "q5", "q6")

# A figure axis names the quantity in ordinary machine-learning words; the formula
# belongs in the caption, where a reader can take their time over it.
BANNED_ON_AXES = ("\\max", "\\mathrm", "\\bar", "\\|", "\\cdot", "\\frac", "\\sum", "_{t", "med_")
# Internal shorthand a reader outside the project would not recognise.
JARGON = ("vstar", "v_star", "cstar", "dose-response", "clamp", "transplant_", "topk",
          "head_retention", "capture_rate", "energy", "alpha", "perp")


@pytest.fixture(scope="module")
def results():
    ST.use("paper")
    return make_mock_question_results(prompts=5, seeds=2)


@pytest.fixture(scope="module")
def figures(results):
    made = {}
    for key in QUESTIONS:
        made[key] = CF.render(results[key])
    yield made
    for figure in made.values():
        plt.close(figure)


def _labels(figure):
    out = []
    for ax in figure.axes:
        out += [ax.get_xlabel(), ax.get_ylabel()]
    return [text for text in out if text]


def test_every_question_renders(figures):
    assert set(figures) == set(QUESTIONS)
    for key, figure in figures.items():
        assert len(figure.axes) >= 1, key


def test_every_figure_carries_a_self_contained_caption(figures):
    for key, figure in figures.items():
        caption = getattr(figure, "dv_caption", "")
        assert caption.startswith(key.upper() + "."), key
        assert len(caption) > 250, f"{key} caption is too thin to stand alone"
        assert caption.rstrip().endswith("."), key


def test_both_axes_are_labelled_on_every_data_panel(figures):
    for key, figure in figures.items():
        for index, ax in enumerate(figure.axes):
            if not (ax.lines or ax.patches or ax.images or ax.collections):
                continue          # colorbars and legend gutters carry no data
            if ax.get_label() == "<colorbar>":
                continue
            assert ax.get_xlabel(), f"{key} panel {index} has no x label"
            assert ax.get_ylabel(), f"{key} panel {index} has no y label"


def test_axis_labels_use_words_not_formulas(figures):
    offenders = [(key, text) for key, figure in figures.items() for text in _labels(figure)
                 if any(token in text for token in BANNED_ON_AXES)]
    assert not offenders, f"formula axis labels: {offenders}"


def test_axis_labels_avoid_internal_jargon(figures):
    offenders = [(key, text) for key, figure in figures.items() for text in _labels(figure)
                 if any(token in text.lower() for token in JARGON)]
    assert not offenders, f"jargon on the axes: {offenders}"


def test_condition_labels_are_written_for_a_reader(figures):
    """Tick labels come from the paper labels, not from the code's condition keys."""
    offenders = []
    for key, figure in figures.items():
        for ax in figure.axes:
            for tick in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
                text = tick.get_text()
                if re.fullmatch(r"[a-z0-9]+(_[a-z0-9]+)+", text):
                    offenders.append((key, text))
    assert not offenders, f"raw condition keys used as tick labels: {offenders}"


def test_a_sham_condition_is_always_drawn_in_grey():
    palette = CF._condition_colors(["direction_removal", "sham", "state_zeroed"])
    assert palette["sham"] == CF.SHAM
    assert palette["direction_removal"] != CF.SHAM


def test_fate_colours_are_an_ordered_ramp_not_the_condition_palette():
    """Outcomes are ordered, so they must not reuse the categorical line colours."""
    assert list(CF.FATE_COLORS) == list(CF.FATE_ORDER)
    assert CF.FATE_COLORS["none"] == CF.SHAM
    assert len(set(CF.FATE_COLORS.values())) == len(CF.FATE_ORDER)


def test_panel_letters_do_not_sit_on_top_of_a_title(figures):
    """A centred title wider than its own panel would run under the panel letter.

    Only multi-panel figures are constrained: a single-panel figure has the whole
    width for its title and carries no panel letter to collide with.
    """
    for key, figure in figures.items():
        panels = [ax for ax in figure.axes if ax.get_label() != "<colorbar>"]
        if len(panels) < 2:
            continue
        for ax in panels:
            title = ax.get_title()
            assert len(title) <= 34, f"{key}: title {title!r} is too wide for its panel"


def test_unsupported_rungs_are_marked_rather_than_drawn_as_zero(results):
    ladder = results["q5"].tidy
    untested = ladder[~ladder["supported"].astype(bool)]
    assert not untested.empty
    figure = CF.fig_q5_sufficiency_ladder(ladder)
    texts = [t.get_text() for t in figure.texts] + [
        t.get_text() for ax in figure.axes for t in ax.texts]
    assert any("not exposed" in text for text in texts)
    plt.close(figure)


def test_figures_refuse_an_incomplete_table():
    with pytest.raises(ValueError, match="missing columns"):
        CF.fig_q1_register_removal(pd.DataFrame({"condition": ["zero"], "layer": [1],
                                                 "head": [-1], "role": ["intervention"]}))
    with pytest.raises(ValueError, match="empty"):
        CF.fig_q5_sufficiency_ladder(pd.DataFrame())
    with pytest.raises(ValueError, match="missing columns"):
        CF.fig_q2_channel_dose_response(pd.DataFrame({"gamma": [0.0]}))


def test_render_refuses_an_unknown_question(results):
    class _Odd:
        question = "q9"
        tidy = pd.DataFrame()
        tables = {}

    with pytest.raises(KeyError, match="no figure for"):
        CF.render(_Odd())


def test_evidence_forest_marks_supported_and_unsupported_claims(results):
    evidence = CS.evidence_table(results, checkpoint="mock", iterations=200)
    figure = CF.fig_evidence_forest(evidence)
    assert figure.axes[0].get_xlabel()
    assert "bootstrap" in figure.dv_caption
    plt.close(figure)


def test_evidence_table_keeps_a_fixed_reader_facing_column_order(results):
    evidence = CS.evidence_table(results, checkpoint="mock", iterations=200)
    table = CF.evidence_table(evidence)
    assert list(table.columns) == [c for c in CF.EVIDENCE_COLUMNS if c in evidence.columns]
    assert "verdict" in table.columns and len(table) == len(evidence)


def test_circuit_summary_colours_every_arrow_it_is_given():
    status = {"writer_to_channel": "supported", "channel_to_direction": "refuted",
              "direction_to_key": "partial", "key_to_sink": "untested",
              "sink_to_end": "supported"}
    figure = CF.fig_circuit_summary(status, {k: "note" for k in status})
    labels = [t.get_text() for t in figure.axes[0].texts]
    assert {"Q1", "Q2", "Q4", "Q5", "Q6"} <= set(labels)
    assert "causal model" in figure.dv_caption
    plt.close(figure)


def test_verdict_table_pairs_each_question_with_its_answer(results):
    table = CF.verdict_table(results)
    assert list(table["question"]) == [q.upper() for q in QUESTIONS]
    assert table["answer"].str.len().gt(60).all()
    assert (table["rows_measured"] > 0).all()


# --------------------------------------------------- removal interaction views
def _survival_frame(checkpoint=None, prompts=6, seeds=2, seed=0):
    """Survival shares shaped like the claim: the two representational removals
    take all three structures down, the key-space removal takes only the sink."""
    import numpy as np

    truth = {("high_norm_tokens", "high_norm_tokens"): 0.06,
             ("high_norm_tokens", "dominant_channel"): 0.09,
             ("high_norm_tokens", "attention_sinks"): 0.14,
             ("dominant_channel", "high_norm_tokens"): 0.12,
             ("dominant_channel", "dominant_channel"): 0.02,
             ("dominant_channel", "attention_sinks"): 0.21,
             ("attention_sinks", "high_norm_tokens"): 0.97,
             ("attention_sinks", "dominant_channel"): 0.95,
             ("attention_sinks", "attention_sinks"): 0.05}
    rng = np.random.default_rng(seed)
    rows = []
    for (removal, structure), value in truth.items():
        for prompt in range(prompts):
            for s in range(seeds):
                row = dict(prompt_id=prompt, seed=s, removal=removal, structure=structure,
                           survival=float(np.clip(value + rng.normal(0, 0.03), 0, 1.4)))
                if checkpoint:
                    row["checkpoint"] = checkpoint
                rows.append(row)
    return pd.DataFrame(rows)


def test_a_blocked_question_draws_its_reasons_instead_of_raising():
    """A traceback mid-run reads as a broken pipeline, and buries the one thing
    worth reporting: why the architecture could not be asked."""
    from ditsinks.questions import QuestionResult

    reasons = {"pre_mlp_residual": "single blocks concatenate before one proj_out",
               "preexisting_direction": "single blocks concatenate before one proj_out"}
    result = QuestionResult("q4", pd.DataFrame(),
                            {"patch_effects": pd.DataFrame(), "separation": pd.DataFrame()},
                            "", meta={"unsupported": reasons, "checkpoint": "flux1-schnell"})
    figure = CF.render(result)
    printed = " ".join(t.get_text() for ax in figure.axes for t in ax.texts)
    assert "not testable" in printed
    assert "flux1-schnell" in printed
    for name in reasons:
        assert name.replace("_", " ") in printed
    assert "not testable" in figure.dv_caption or "no measurable rows" in figure.dv_caption
    plt.close(figure)


def test_a_builder_still_refuses_an_empty_table_on_its_own():
    """render() may substitute a placeholder; a builder may never invent data."""
    with pytest.raises(ValueError, match="empty"):
        CF.fig_q4_writer_selection(pd.DataFrame())


def test_an_unsupported_q4_feature_is_marked_not_omitted():
    """Dropping it from the figure would read as tested-and-found-nothing."""
    rows = []
    for patch, supported in (("modulated_stream", True), ("pre_mlp_residual", False)):
        for direction in ("register_to_ordinary", "ordinary_to_register"):
            for prompt in range(4):
                rows.append(dict(question="q4", patch=patch, patch_label=patch.replace("_", " "),
                                 direction=direction, direction_label=direction,
                                 endpoint="capture_rate", endpoint_label="Sink capture rate",
                                 prompt_id=prompt, seed=0, recipient=1,
                                 effect=0.3 if supported else float("nan"),
                                 supported=supported, reason="" if supported else "no such stage"))
    figure = CF.fig_q4_writer_selection(pd.DataFrame(rows))
    printed = " ".join(t.get_text() for ax in figure.axes for t in ax.texts)
    assert "not exposed" in printed, "the refused feature was drawn as an ordinary null"
    plt.close(figure)


def test_the_channel_trace_marks_where_the_suppression_acted():
    """Without the span the reader cannot tell a rebuild from a miss."""
    import numpy as np

    rng = np.random.default_rng(0)
    rows = []
    for removal in ("clean", "dominant_channel", "high_norm_tokens"):
        for prompt in range(4):
            for layer in range(6):
                level = 1.0 if removal == "clean" else (0.1 if layer in (2, 3) else 0.9)
                rows.append(dict(prompt_id=prompt, seed=0, removal=removal, layer=layer,
                                 at_registers=level + rng.normal(0, 0.01), elsewhere=0.05))
    figure = CF.fig_channel_trace(pd.DataFrame(rows), maintenance_layers=[2, 3], writer_layer=1)
    ax = figure.axes[0]
    assert ax.get_xlabel() and ax.get_ylabel()
    printed = " ".join(t.get_text() for t in ax.texts)
    assert "held down" in printed and "writer" in printed
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert any("Ordinary patches" in t for t in labels), "the floor must be drawn for scale"
    plt.close(figure)


def _strip_frame(prompts: int = 1, grid: int = 6, seed: int = 0) -> pd.DataFrame:
    """A maps tidy table with both methods, shaped like a real run's."""
    import numpy as np

    from ditsinks.questions import REMOVALS, STRUCTURES, ablation_method

    rng = np.random.default_rng(seed)
    rows = []
    for prompt in range(prompts):
        for removal, _ in REMOVALS:
            for structure, _ in STRUCTURES:
                methods = {"clean"} if removal == "clean" else \
                    {"causal", ablation_method(removal, structure)}
                for method in sorted(methods):
                    scale = 1.0 if method != "subtract" else 0.4
                    for r in range(grid):
                        for c in range(grid):
                            rows.append(dict(prompt_id=prompt, seed=0, removal=removal,
                                             structure=structure, method=method,
                                             token=r * grid + c, row=r, col=c,
                                             value=float(rng.random()) * scale,
                                             is_frozen_target=(r == c == 2)))
    return pd.DataFrame(rows)


def _panel_titles(figure):
    return [ax.get_title() for ax in figure.axes if ax.get_title()]


def test_ablation_strip_gives_every_measurement_its_own_row():
    figure = CF.fig_ablation_strip(_strip_frame(), image=None)
    panels = [ax for ax in figure.axes if ax.images]
    assert len(panels) == 27, "three removals by three measurements by three map columns"
    plt.close(figure)


def test_ablation_strip_names_the_method_on_every_after_panel():
    """A reader must never have to guess whether a panel was re-run or recomputed."""
    figure = CF.fig_ablation_strip(_strip_frame(), image=None)
    after = [t for t in _panel_titles(figure) if t.startswith("after")]
    assert len(after) == 9
    assert all(("recomputed" in t) or ("model re-run" in t) for t in after), after
    assert sum("recomputed" in t for t in after) == 5, "five pairs can be done by bookkeeping"
    plt.close(figure)


def test_ablation_strip_never_prints_the_internal_method_names():
    figure = CF.fig_ablation_strip(_strip_frame(), image=None)
    printed = " ".join(_panel_titles(figure)).lower()
    assert "subtract" not in printed and "causal" not in printed
    plt.close(figure)


def _strip_rows(figure, n_rows: int = 9):
    """The figure's panels grouped by row: image, what was removed, before, after."""
    return [figure.axes[4 * r:4 * r + 4] for r in range(n_rows)]


def test_ablation_strip_fills_each_panel_by_the_declared_method():
    """The whole point of the figure is that the two methods are not interchangeable."""
    import numpy as np

    frame = _strip_frame()
    figure = CF.fig_ablation_strip(frame, image=None)
    # Row 0 is the first block's first measurement: high-norm tokens removed,
    # high-norm tokens measured, which bookkeeping can do.
    after = _strip_rows(figure)[0][3]
    assert after.get_title() == "after \u2014 recomputed"
    cell = frame[(frame["removal"] == "high_norm_tokens")
                 & (frame["structure"] == "high_norm_tokens")
                 & (frame["method"] == "subtract")]
    assert np.allclose(after.images[0].get_array(), CF._to_grid(cell))
    plt.close(figure)


def test_ablation_strip_measures_before_and_after_on_one_scale():
    """A panel going dark has to mean the structure went away, not that the
    yardstick followed it down."""
    figure = CF.fig_ablation_strip(_strip_frame(), image=None)
    for index, (_, _, before, after) in enumerate(_strip_rows(figure)):
        assert before.images[0].get_clim() == after.images[0].get_clim(), \
            f"row {index} rescales between before and after"
        assert before.images[0].get_clim()[0] == 0.0, "zero is the floor for all three"
    plt.close(figure)


def test_ablation_strip_marks_the_frozen_register_patches():
    figure = CF.fig_ablation_strip(_strip_frame(), image=None)
    circled = [ax for ax in figure.axes if ax.images and ax.patches]
    assert len(circled) == 27, "the same patches are followed across every panel"
    plt.close(figure)


def test_ablation_strip_takes_one_unit_rather_than_pooling_prompts():
    """Maps from two prompts are two different images and cannot be averaged."""
    figure = CF.fig_ablation_strip(_strip_frame(prompts=3), image=None)
    assert "Prompt 0" in figure.dv_caption
    plt.close(figure)


def test_ablation_strip_needs_a_removal_to_show():
    only_reference = _strip_frame()
    only_reference = only_reference[only_reference["removal"] == "clean"]
    with pytest.raises(ValueError, match="at least one removal"):
        CF.fig_ablation_strip(only_reference)


def test_structure_interaction_never_mixes_the_two_methods():
    """The interaction matrix is the causal one; the bookkeeping panels have
    their own figure, and averaging them onto one grid would report neither."""
    import numpy as np

    frame = _strip_frame()
    figure = CF.fig_structure_interaction(frame)
    causal = frame[(frame["removal"] == "high_norm_tokens")
                   & (frame["structure"] == "high_norm_tokens")
                   & (frame["method"] == "causal")]
    drawn = [ax.images[0].get_array() for ax in figure.axes if ax.images]
    assert any(np.allclose(grid, CF._to_grid(causal)) for grid in drawn)
    plt.close(figure)


def test_structure_interaction_lays_maps_on_the_grid():
    import numpy as np

    rng = np.random.default_rng(0)
    rows = []
    for removal in ("clean", "high_norm_tokens", "attention_sinks"):
        for structure in ("high_norm_tokens", "dominant_channel", "attention_sinks"):
            for r in range(6):
                for c in range(6):
                    rows.append(dict(prompt_id=0, seed=0, removal=removal, structure=structure,
                                     token=r * 6 + c, row=r, col=c,
                                     value=float(rng.random()), is_frozen_target=(r == c == 2)))
    figure = CF.fig_structure_interaction(pd.DataFrame(rows))
    panels = [ax for ax in figure.axes if ax.images]
    assert len(panels) == 9, "three removals by three structures"
    assert "Rows are what was taken away" in figure.dv_caption
    plt.close(figure)
