"""Figures for the six causal experiments on register tokens.

The house style is the same one the observational atlas uses: a Times-like
serif, a closed axes box with inward ticks, the Okabe-Ito colour-vision-safe
palette, and axis labels written in ordinary machine-learning words placed on the
axes themselves.  Formulas belong in the caption, never on the canvas.

Every function takes a tidy table (one observation per row), returns a
Matplotlib figure carrying a self-contained ``dv_caption``, and refuses to plot
a table that is missing the columns its claim depends on.  Nothing here infers a
target, fills in a missing condition, or pools checkpoints.

Colour has one meaning per figure family:

* condition identity uses the categorical palette, with the sham always grey so
  the eye separates "we did nothing" from every real perturbation;
* a signed causal effect uses the diverging map centred on zero;
* a probability or rate uses the sequential probability map.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib import patheffects
import numpy as np
import pandas as pd

from . import style as st

BLUE = st.OKABE_ITO["blue"]
VERMILLION = st.OKABE_ITO["vermillion"]
GREEN = st.OKABE_ITO["green"]
PURPLE = st.OKABE_ITO["purple"]
SKY = st.OKABE_ITO["sky"]
ORANGE = st.OKABE_ITO["orange"]
YELLOW = st.OKABE_ITO["yellow"]
SHAM = "#8c8c8c"

CONDITION_CYCLE = (BLUE, VERMILLION, GREEN, PURPLE, SKY, ORANGE, YELLOW)

# Mutually exclusive downstream outcomes, in the order a reader should read them:
# the register comes back, it moves, a reserve takes over, attention spreads out,
# or nothing identifiable happens.
FATE_ORDER = ("same_position", "relocated", "reserve_takeover", "diffuse", "none")
def _fate_ramp() -> Dict[str, str]:
    """Outcome colour runs from "the register survives" to "the register is lost".

    The fates are ordered, not merely categorical, so they take the diverging map
    rather than the categorical palette, which also keeps them visually distinct
    from the condition colours used on the same page.
    """
    from matplotlib.colors import to_hex

    ramp = st.diverging_cmap()
    keys = ("same_position", "relocated", "reserve_takeover", "diffuse")
    return {**{k: to_hex(ramp(v)) for k, v in zip(keys, (0.04, 0.28, 0.72, 0.96))}, "none": SHAM}


FATE_COLORS = _fate_ramp()
FATE_LABELS = {"same_position": "Original carriers regain criterion",
               "relocated": "Another carrier wins (novelty unproven)",
               "reserve_takeover": "A non-target sink takes over",
               "diffuse": "Attention spreads out",
               "none": "No identifiable outcome"}
RECOVERY_COLORS = {"same_position": BLUE, "relocated": SKY, "none": SHAM}
RECOVERY_LABELS = {"same_position": "Original carrier wins again",
                   "relocated": "Another carrier wins (novelty unproven)",
                   "none": "No register state"}


# ------------------------------------------------------------------ helpers
def _need(df: pd.DataFrame, columns: Iterable[str], what: str = "figure") -> None:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError(f"cannot draw the {what}: the result table is empty")
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        raise ValueError(f"cannot draw the {what}: missing columns " + ", ".join(missing))


def _layer_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Layer-level rollups only.  ``head`` shadows ``DataFrame.head``, so index it."""
    return df[df["head"] == -1] if "head" in df.columns else df


def _head_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["head"] >= 0] if "head" in df.columns else df


def _pretty(value: Any) -> str:
    return str(value).replace("__", " · ").replace("_", " ")


def _condition_colors(conditions: Sequence[str]) -> Dict[str, str]:
    """Stable colours, with any sham or control-by-name condition held at grey."""
    palette: Dict[str, str] = {}
    index = 0
    for condition in conditions:
        if "sham" in str(condition).lower():
            palette[condition] = SHAM
            continue
        palette[condition] = CONDITION_CYCLE[index % len(CONDITION_CYCLE)]
        index += 1
    return palette


def _clustered_band(frame: pd.DataFrame, x: str, y: str, *, cluster: str = "prompt_id",
                    iterations: int = 400) -> pd.DataFrame:
    """Mean and prompt-clustered interval at each x, for a shaded band."""
    from .causal_stats import bootstrap_mean

    rows = []
    for level, group in frame.groupby(x, observed=True):
        estimate = bootstrap_mean(group, y, cluster=cluster, iterations=iterations)
        rows.append(dict(**{x: level}, mean=estimate.value, low=estimate.ci_low,
                         high=estimate.ci_high, n=estimate.n_units))
    out = pd.DataFrame(rows).sort_values(x).reset_index(drop=True)
    if not out.empty:
        out["low"] = out["low"].fillna(out["mean"])
        out["high"] = out["high"].fillna(out["mean"])
    return out


def _line_with_band(ax, frame: pd.DataFrame, x: str, y: str, *, color: str, label: str,
                    cluster: str = "prompt_id", marker: str = "o") -> None:
    stats = _clustered_band(frame, x, y, cluster=cluster)
    if stats.empty:
        return
    ax.plot(stats[x], stats["mean"], marker=marker, color=color, label=label, zorder=3)
    ax.fill_between(stats[x], stats["low"], stats["high"], color=color, alpha=0.16, linewidth=0,
                    zorder=2)


def _finish(fig, caption: str, path=None, tight: bool = True):
    fig.dv_caption = caption
    if tight:
        fig.tight_layout()
    if path:
        st.savefig(fig, Path(path))
    return fig


def _layout(fig, rect=None) -> None:
    """Run tight_layout where a colorbar spans several axes.

    Matplotlib warns that such a figure is not compatible with tight_layout and
    the result "might be incorrect".  For these figures it is not: the gridspec
    fixes the panel positions and the colorbar is placed against a named list of
    axes, and the output is checked.  The warning is silenced here, and only here,
    rather than left to clutter every test run.
    """
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*not compatible with tight_layout.*")
        fig.tight_layout(**({"rect": rect} if rect else {}))


def _panel_letters(fig, axes: Sequence[Any], start: int = 0) -> None:
    """Place (a), (b), ... above the left edge of each panel, after layout.

    Anchoring in figure coordinates rather than axes coordinates is what keeps a
    panel letter clear of a centred title: the letter sits above the panel's own
    left edge, where no tick label or title reaches.
    """
    fig.canvas.draw()
    for index, ax in enumerate(axes):
        box = ax.get_position()
        fig.text(max(box.x0 - 0.030, 0.002), min(box.y1 + 0.022, 0.995),
                 f"({chr(97 + start + index)})", ha="left", va="bottom",
                 fontsize=plt.rcParams["axes.titlesize"], fontweight="bold")


def _integer_layers(ax) -> None:
    """Layer indices are integers; matplotlib's default ticks are not."""
    from matplotlib.ticker import MaxNLocator

    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))


def _no_change_marker(ax, x: float, text: str = "no change") -> None:
    """Mark the scale at which an intervention does nothing, above the data."""
    ax.axvline(x, color=st.RULE, linestyle=(0, (4, 2)), linewidth=1.0, zorder=1)
    ax.annotate(text, xy=(x, 1.0), xycoords=("data", "axes fraction"), xytext=(3, -4),
                textcoords="offset points", ha="left", va="top", color=st.INK_SOFT,
                fontsize=plt.rcParams["legend.fontsize"] - 1)


def _shared_condition_legend(fig, axes: Sequence[Any], *, ncol: int = 4, y: float = 0.005) -> None:
    """One condition legend for the whole figure, so no panel hides its own data."""
    handles, labels, seen = [], [], set()
    for ax in axes:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in seen:
                seen.add(label)
                handles.append(handle)
                labels.append(label)
        if ax.get_legend() is not None:
            ax.get_legend().remove()
    if handles:
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, y), ncol=ncol,
                   frameon=False, fontsize=plt.rcParams["legend.fontsize"] - 1)


def _no_data(ax, message: str = "not measurable in this run") -> None:
    # Matplotlib's default ``axes.labelsize`` is a keyword ("medium"), and only becomes
    # a number once the paper style is applied. Saying "there is nothing to draw" is the
    # last place that should raise, so the size is resolved rather than assumed.
    size = plt.rcParams["axes.labelsize"]
    size = size - 1 if isinstance(size, (int, float)) else 9.0
    ax.text(0.5, 0.5, message, transform=ax.transAxes, ha="center", va="center",
            color=st.INK_SOFT, fontsize=size)
    ax.set_xticks([])
    ax.set_yticks([])


# ============================================================ natural-register removal
def fig_q1_register_removal(tidy: pd.DataFrame, fates: Optional[pd.DataFrame] = None, path=None):
    """What the routing does once the natural register population is gone."""
    _need(tidy, ["condition", "layer", "head", "role"], "Q1 figure")
    layers = _layer_rows(tidy)
    fates = fates if fates is not None and not fates.empty else tidy
    _need(fates, ["condition", "fate"], "Q1 fate panel")

    fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.0))
    (ax_fate, ax_retention), (ax_concentration, ax_projection) = axes

    # (a) mutually exclusive downstream outcome, per condition
    unit_keys = [k for k in ("condition", "condition_label", "role", "prompt_id", "seed",
                             "selection_rule") if k in fates.columns]
    runs = fates.drop_duplicates(subset=[k for k in unit_keys if k != "condition_label"] or ["condition"])
    counts = (runs.groupby(["condition", "fate"], observed=True).size().unstack(fill_value=0)
              .reindex(columns=list(FATE_ORDER), fill_value=0))
    shares = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    order = _order_by_role(runs, shares.index)
    shares = shares.reindex(order)
    left = np.zeros(len(shares))
    for fate in FATE_ORDER:
        values = shares[fate].to_numpy()
        ax_fate.barh(np.arange(len(shares)), values, left=left, height=0.72,
                     color=FATE_COLORS[fate], edgecolor="white", linewidth=0.6,
                     label=FATE_LABELS[fate])
        left += values
    ax_fate.set_yticks(np.arange(len(shares)), [_label_for(runs, c) for c in shares.index])
    ax_fate.set(xlim=(0, 1), xlabel="Share of prompt-seed runs", ylabel="Intervention",
                title="Outcome after removal")
    ax_fate.invert_yaxis()
    ax_fate.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, frameon=False,
                   fontsize=plt.rcParams["legend.fontsize"] - 1.5, columnspacing=0.9,
                   handlelength=1.1, handletextpad=0.4)

    # (b) does attention keep the clean sink, layer by layer
    _panel_by_layer(ax_retention, layers, "head_retention",
                    ylabel="Fraction of heads keeping the clean sink",
                    title="Sink identity by layer", ylim=(0, 1))

    # (c) does attention become diffuse
    _panel_by_layer(ax_concentration, layers, "attention_concentration",
                    ylabel="Strongest incoming attention share",
                    title="How concentrated is attention?")

    # (d) direction against magnitude.  The projection on v* is the product of the
    # two, so it can never separate "the token stopped pointing at v*" from "the
    # token got smaller", the distinction the whole account rests on.  Plotting
    # them as two axes puts each condition in the quadrant that says what it did.
    _direction_versus_magnitude(ax_projection, layers)

    for ax in (ax_retention, ax_concentration):
        _integer_layers(ax)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    _shared_condition_legend(fig, (ax_retention, ax_concentration, ax_projection), ncol=4,
                             y=0.005)
    _panel_letters(fig, (ax_fate, ax_retention, ax_concentration, ax_projection))

    caption = (
        "Q1. Removing the frozen high-norm register population at the first register layer. "
        "(a) Mutually exclusive downstream outcome per paired prompt-seed run. "
        "(b) Fraction of affected heads whose strongest image key is still the clean sink; an "
        "affected head is one whose clean sink was a removed token. "
        "(c) Mean strongest incoming attention share, the reference for calling attention diffuse. "
        "(d) Each intervention placed by how far it moved the target tokens' alignment with the "
        "frozen register direction $v^*$ (horizontal) and how far it moved their magnitude "
        "(vertical), both as paired changes against the sham run at the origin. The projection on "
        "$v^*$ is the product of these two quantities and cannot separate them, so they are "
        "reported on separate axes: a condition on the horizontal axis alone changed direction at "
        "constant size, one on the vertical axis alone changed size at constant direction. Whiskers "
        "are prompt-clustered 95\\% bootstrap intervals. All targets are frozen from the clean "
        "pass and never reselected after treatment.")
    return _finish(fig, caption, path, tight=False)


def _direction_versus_magnitude(ax, layers: pd.DataFrame, reference: str = "sham") -> None:
    """Each condition placed by how far it moved alignment and how far it moved size."""
    from .causal_stats import paired_effect

    needed = {"cosine", "target_norm_ratio", "condition"}
    if not needed <= set(layers.columns) or layers.empty:
        _no_data(ax)
        ax.set(xlabel="Change in alignment with the register direction",
               ylabel="Change in token norm over the layer median",
               title="Direction versus magnitude")
        return
    conditions = [c for c in _order_by_role(layers, layers["condition"].unique()) if c != reference]
    palette = _condition_colors(conditions)
    drawn = False
    for condition in conditions:
        alignment = paired_effect(layers, "cosine", treatment=condition, reference=reference,
                                  iterations=300)
        magnitude = paired_effect(layers, "target_norm_ratio", treatment=condition,
                                  reference=reference, iterations=300)
        if not (np.isfinite(alignment.value) and np.isfinite(magnitude.value)):
            continue
        ax.errorbar(alignment.value, magnitude.value,
                    xerr=_half_width(alignment), yerr=_half_width(magnitude),
                    fmt="o", markersize=7, color=palette[condition], ecolor=st.INK,
                    elinewidth=0.8, capsize=2.0, zorder=3,
                    label=_label_for(layers, condition))
        drawn = True
    if not drawn:
        _no_data(ax)
    else:
        ax.axhline(0.0, color=st.RULE, linewidth=1.0, zorder=1)
        ax.axvline(0.0, color=st.RULE, linewidth=1.0, zorder=1)
        ax.annotate("no change (sham)", xy=(0, 0), xytext=(-5, -5), textcoords="offset points",
                    ha="right", va="top", color=st.INK_SOFT,
                    fontsize=plt.rcParams["legend.fontsize"] - 1.5)
        # No legend here: the shared condition legend at the foot of the figure
        # already names every colour, and repeating it would crowd the panel.
        st.light_grid(ax, "y")
        st.light_grid(ax, "x")
    ax.set(xlabel="Change in alignment with the register direction",
           ylabel="Change in token norm over the layer median",
           title="Direction versus magnitude")


def _half_width(estimate) -> float:
    """Half the clustered interval, or zero when there is no usable interval."""
    if not getattr(estimate, "has_interval", False):
        return 0.0
    return float(abs(estimate.ci_high - estimate.ci_low) / 2.0)


def _order_by_role(frame: pd.DataFrame, conditions: Iterable[str]) -> List[str]:
    """Interventions first, controls after, so a reader compares like with like."""
    roles = (frame.groupby("condition")["role"].first().to_dict()
             if "role" in frame.columns else {})
    return sorted(conditions, key=lambda c: (roles.get(c, "intervention") != "intervention", str(c)))


def _label_for(frame: pd.DataFrame, condition: str) -> str:
    if "condition_label" in frame.columns:
        match = frame.loc[frame["condition"] == condition, "condition_label"].dropna()
        if len(match):
            return str(match.iloc[0])
    return _pretty(condition)


def _panel_by_layer(ax, layers: pd.DataFrame, column: str, *, ylabel: str, title: str,
                    ylim: Optional[Tuple[float, float]] = None, xlabel: str = "Transformer layer") -> None:
    if column not in layers.columns or layers[column].dropna().empty:
        _no_data(ax)
        ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
        return
    conditions = _order_by_role(layers, layers["condition"].unique())
    palette = _condition_colors(conditions)
    for condition in conditions:
        group = layers[layers["condition"] == condition]
        if group[column].dropna().empty:
            continue
        _line_with_band(ax, group, "layer", column, color=palette[condition],
                        label=_label_for(layers, condition))
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    if ylim:
        ax.set_ylim(*ylim)
    st.light_grid(ax, "y")
    ax.legend(frameon=False, fontsize=plt.rcParams["legend.fontsize"] - 1)


def _effect_bars(ax, layers: pd.DataFrame, column: str, *, xlabel: str, title: str,
                 reference: str = "sham") -> None:
    """Horizontal bars with prompt-clustered intervals, sham drawn as the zero line."""
    from .causal_stats import bootstrap_mean

    if column not in layers.columns or layers[column].dropna().empty:
        _no_data(ax)
        ax.set(xlabel=xlabel, title=title)
        return
    conditions = _order_by_role(layers, layers["condition"].unique())
    palette = _condition_colors(conditions)
    rows = []
    for condition in conditions:
        group = layers[layers["condition"] == condition]
        estimate = bootstrap_mean(group, column, iterations=400)
        if not math.isnan(estimate.value):
            rows.append((condition, estimate))
    if not rows:
        _no_data(ax)
        ax.set(xlabel=xlabel, title=title)
        return
    positions = np.arange(len(rows))
    values = [e.value for _, e in rows]
    low = [e.value - (e.ci_low if np.isfinite(e.ci_low) else e.value) for _, e in rows]
    high = [(e.ci_high if np.isfinite(e.ci_high) else e.value) - e.value for _, e in rows]
    ax.barh(positions, values, height=0.68, color=[palette[c] for c, _ in rows],
            edgecolor=st.INK, linewidth=0.5, zorder=2)
    ax.errorbar(values, positions, xerr=[low, high], fmt="none", ecolor=st.INK, elinewidth=0.9,
                capsize=2.5, zorder=3)
    ax.axvline(0.0, color=st.RULE, linewidth=1.0, zorder=1)
    ax.set_yticks(positions, [_label_for(layers, c) for c, _ in rows])
    ax.invert_yaxis()
    ax.set(xlabel=xlabel, title=title)
    st.light_grid(ax, "x")


# ============================================================ dominant-channel suppression
def fig_q2_channel_dose_response(dose: pd.DataFrame, path=None):
    """The dominant channel's live dose-response, with matched controls."""
    _need(dose, ["gamma", "scope", "control", "vstar_projection_change"], "Q2 figure")
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.0))
    ax_projection, ax_retention, ax_controls = axes
    primary = dose[dose["control"] == "dominant_channel"]

    scope_style = {"writer_only": (BLUE, "o", "-"), "maintenance": (VERMILLION, "s", "--")}
    for scope, (color, marker, linestyle) in scope_style.items():
        group = primary[primary["scope"] == scope]
        if group.empty:
            continue
        stats = _clustered_band(group, "gamma", "vstar_projection_change")
        ax_projection.plot(stats["gamma"], stats["mean"], marker=marker, linestyle=linestyle,
                           color=color, label=_pretty(scope))
        ax_projection.fill_between(stats["gamma"], stats["low"], stats["high"], color=color,
                                   alpha=0.15, linewidth=0)
        if "original_sink_retained" in group.columns and group["original_sink_retained"].notna().any():
            retention = _clustered_band(group, "gamma", "original_sink_retained")
            ax_retention.plot(retention["gamma"], retention["mean"], marker=marker,
                              linestyle=linestyle, color=color, label=_pretty(scope))
            ax_retention.fill_between(retention["gamma"], retention["low"], retention["high"],
                                      color=color, alpha=0.15, linewidth=0)
    for ax, ylabel, title in ((ax_projection, "Change in register-direction projection",
                               "Direction follows the channel"),
                              (ax_retention, "Fraction of heads keeping the clean sink",
                               "Routing follows the channel")):
        _no_change_marker(ax, 1.0)
        ax.set(xlabel="Dominant-channel scale factor", ylabel=ylabel, title=title)
        st.light_grid(ax, "y")
        ax.legend(frameon=False, title="Suppression scope",
                  fontsize=plt.rcParams["legend.fontsize"] - 1, title_fontsize=plt.rcParams["legend.fontsize"] - 1)
    if ax_retention.get_legend_handles_labels()[0] == []:
        _no_data(ax_retention)
        ax_retention.set(xlabel="Dominant-channel scale factor",
                         ylabel="Fraction of heads keeping the clean sink",
                         title="Routing follows the channel")
    ax_retention.set_ylim(0, 1)

    # (c) at full suppression, does the dominant channel beat every matched control?
    at_zero = dose[np.isclose(pd.to_numeric(dose["gamma"], errors="coerce").fillna(-1), 0.0)]
    _effect_bars_by(ax_controls, at_zero, "control", "vstar_projection_change",
                    label_column="control_label",
                    xlabel="Change in register-direction projection",
                    ylabel="What was suppressed",
                    title="Against matched controls",
                    highlight="dominant_channel")

    caption = (
        "Q2. Live scaling of the dominant register channel at the writer layer. "
        "(a) Paired change in the frozen register direction's projection against the channel "
        "scale $\\gamma$; $\\gamma=1$ leaves the model unchanged and is the sham reference. "
        "(b) The same sweep read on routing, as the fraction of affected heads keeping the clean "
        "sink. Writer-only scales the channel once, at the writing block; maintenance keeps "
        "scaling it through the following register-zone blocks. "
        "(c) At full suppression, the dominant channel compared with a late-growing competitor, a "
        "register-unspecific massive channel, a random channel, a removal of the same amount of "
        "magnitude along an unrelated direction, and the same scalar applied at ordinary positions. "
        "Lines and bars are means with "
        "prompt-clustered 95\\% bootstrap intervals over paired prompt-seed runs.")
    fig.tight_layout()
    _panel_letters(fig, axes)
    return _finish(fig, caption, path, tight=False)


def _effect_bars_by(ax, frame: pd.DataFrame, group_column: str, value: str, *, xlabel: str,
                    title: str, label_column: Optional[str] = None, ylabel: str = "",
                    highlight: Optional[str] = None, role_column: str = "role") -> None:
    """Horizontal effect bars grouped by one column, with clustered intervals.

    Whatever the figure is contrasting is coloured; everything acting as a control
    stays grey, so a reader sees at a glance which bar is the claim.
    """
    from .causal_stats import bootstrap_mean

    if frame is None or frame.empty or value not in frame.columns or frame[value].dropna().empty:
        _no_data(ax)
        ax.set(xlabel=xlabel, title=title)
        return
    rows = []
    for key, group in frame.groupby(group_column, observed=True):
        estimate = bootstrap_mean(group, value, iterations=400)
        if not math.isnan(estimate.value):
            label = (str(group[label_column].dropna().iloc[0])
                     if label_column and label_column in group and group[label_column].notna().any()
                     else _pretty(key))
            rows.append((key, label, estimate))
    if not rows:
        _no_data(ax)
        ax.set(xlabel=xlabel, title=title)
        return
    rows.sort(key=lambda r: abs(r[2].value), reverse=True)
    positions = np.arange(len(rows))
    values = [e.value for _, _, e in rows]
    roles = (frame.groupby(group_column)[role_column].first().to_dict()
             if role_column in frame.columns else {})
    palette = _condition_colors([k for k, _, _ in rows])
    colors = []
    for key, _, _ in rows:
        if highlight is not None:
            colors.append(VERMILLION if str(highlight) in str(key) else SHAM)
        elif roles:
            colors.append(palette[key] if roles.get(key, "intervention") == "intervention" else SHAM)
        else:
            colors.append(palette[key])
    low = [e.value - (e.ci_low if np.isfinite(e.ci_low) else e.value) for _, _, e in rows]
    high = [(e.ci_high if np.isfinite(e.ci_high) else e.value) - e.value for _, _, e in rows]
    ax.barh(positions, values, height=0.68, color=colors, edgecolor=st.INK, linewidth=0.5, zorder=2)
    ax.errorbar(values, positions, xerr=[low, high], fmt="none", ecolor=st.INK, elinewidth=0.9,
                capsize=2.5, zorder=3)
    ax.axvline(0.0, color=st.RULE, linewidth=1.0, zorder=1)
    ax.set_yticks(positions, [label for _, label, _ in rows])
    ax.invert_yaxis()
    ax.set(xlabel=xlabel, title=title)
    if ylabel:
        ax.set_ylabel(ylabel)
    st.light_grid(ax, "x")


# ============================================================ register regeneration
def fig_q3_regeneration(tidy: pd.DataFrame, recovery: Optional[pd.DataFrame] = None, path=None):
    """Whether, where and when the register direction comes back."""
    recovery = recovery if recovery is not None and not recovery.empty else tidy
    _need(recovery, ["scope", "recovery_kind"], "Q3 figure")
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.0))
    ax_kind, ax_when, ax_trajectory = axes

    keys = [k for k in ("condition", "scope", "prompt_id", "seed") if k in recovery.columns]
    runs = recovery.drop_duplicates(subset=keys) if keys else recovery
    table = (runs.groupby(["scope", "recovery_kind"], observed=True).size().unstack(fill_value=0)
             .reindex(columns=[k for k in RECOVERY_COLORS if k in
                               runs["recovery_kind"].unique()], fill_value=0))
    if table.empty or table.to_numpy().sum() == 0:
        _no_data(ax_kind)
    else:
        shares = table.div(table.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
        bottom = np.zeros(len(shares))
        for kind in shares.columns:
            values = shares[kind].to_numpy()
            ax_kind.bar(np.arange(len(shares)), values, bottom=bottom, width=0.62,
                        color=RECOVERY_COLORS.get(kind, SHAM), edgecolor="white", linewidth=0.6,
                        label=RECOVERY_LABELS.get(kind, _pretty(kind)))
            bottom += values
        ax_kind.set_xticks(np.arange(len(shares)), [_pretty(s) for s in shares.index])
        ax_kind.set_ylim(0, 1)
        ax_kind.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=1, frameon=False,
                       fontsize=plt.rcParams["legend.fontsize"] - 1)
    ax_kind.set(xlabel="Removal schedule", ylabel="Share of prompt-seed runs",
                title="Which carrier wins downstream?")

    layers = runs["first_recovery_layer"].dropna() if "first_recovery_layer" in runs else pd.Series(dtype=float)
    if layers.empty:
        _no_data(ax_when, "the direction never returned")
    else:
        # Side-by-side counts, not overlaid translucent histograms: overlapping
        # transparency invents a third colour that means nothing.
        present = runs.dropna(subset=["first_recovery_layer"])
        scopes = list(dict.fromkeys(present["scope"].astype(str)))
        edges = np.arange(int(layers.min()), int(layers.max()) + 2)
        width = 0.82 / max(len(scopes), 1)
        for index, scope in enumerate(scopes):
            values = present.loc[present["scope"].astype(str) == scope,
                                 "first_recovery_layer"].to_numpy(dtype=float)
            counts, _ = np.histogram(values, bins=np.append(edges, edges[-1] + 1) - 0.5)
            offsets = edges + (index - (len(scopes) - 1) / 2) * width
            ax_when.bar(offsets, counts[:len(edges)], width=width * 0.92,
                        color=CONDITION_CYCLE[index % len(CONDITION_CYCLE)], edgecolor=st.INK,
                        linewidth=0.4, label=_pretty(scope), zorder=2)
        ax_when.legend(frameon=False, fontsize=plt.rcParams["legend.fontsize"] - 1,
                       title="Removal schedule",
                       title_fontsize=plt.rcParams["legend.fontsize"] - 1)
        _integer_layers(ax_when)
    ax_when.set(xlabel="First layer at which the direction returns",
                ylabel="Number of prompt-seed runs", title="When the direction returns")
    st.light_grid(ax_when, "y")

    layer_rows = _layer_rows(tidy)
    _panel_by_layer(ax_trajectory, layer_rows, "max_cosine",
                    ylabel="Strongest register-direction alignment\namong all image tokens",
                    title="Recovery trajectory")
    _integer_layers(ax_trajectory)

    caption = (
        "Q3. Norm-preserving destruction of the register direction at a natural register. "
        "(a) Mutually exclusive carrier outcome: the strongest direction is at an original carrier, "
        "another carrier, or absent, under single-shot, fixed-carrier repeated, and dynamic-role "
        "ablation. Another-carrier capture is not itself paired-clean regeneration; that stricter "
        "endpoint is saved in the token-history table. (b) Distribution of the first layer whose strongest alignment crosses the "
        "frozen clean threshold. (c) The strongest alignment with $v^*$ over all image tokens at "
        "each layer; the restoration conditions patch the clean residual back at one later layer to "
        "locate where maintenance pressure is strongest. Bands are prompt-clustered 95\\% "
        "bootstrap intervals.")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    _panel_letters(fig, axes)
    return _finish(fig, caption, path, tight=False)


# ============================================================ writer-feature selection
def fig_q4_writer_selection(patches: pd.DataFrame, separation: Optional[pd.DataFrame] = None,
                            path=None, primary_endpoint: str = "capture_rate"):
    """Which upstream feature both transfers and prevents register formation.

    The claim is reciprocal, so the primary panel is drawn as a reciprocal figure:
    transfer runs to the right, prevention to the left, and a genuine writer
    feature makes a symmetric pair.  A feature that only moves one way is visible
    immediately as a one-sided bar.
    """
    _need(patches, ["patch", "direction", "endpoint", "effect"], "Q4 figure")
    from .causal_stats import bootstrap_mean

    has_separation = separation is not None and not separation.empty and \
        "separation" in separation.columns
    fig, axes = plt.subplots(1, 3 if has_separation else 2, figsize=(13.6 if has_separation else 9.4,
                                                                    4.4), squeeze=False)
    axes = list(axes[0])
    ax_primary, ax_grid = axes[0], axes[1]

    # (a) reciprocal signature on the primary endpoint
    primary = patches[patches["endpoint"].astype(str) == primary_endpoint]
    if primary.empty and len(patches):
        primary = patches[patches["endpoint"].astype(str) ==
                          str(patches["endpoint"].iloc[0])]
    features = list(dict.fromkeys(primary["patch"].astype(str)))
    positions = np.arange(len(features))[::-1]
    # A feature the architecture never exposed is marked, not drawn as a bar of
    # height zero, which would read as "we tested it and it did nothing".
    blocked = _blocked_features(primary, features)
    drawn = False
    for direction, color, sign in (("register_to_ordinary", GREEN, +1.0),
                                   ("ordinary_to_register", VERMILLION, -1.0)):
        subset = primary[primary["direction"].astype(str) == direction]
        if subset.empty:
            continue
        values, low, high = [], [], []
        for feature in features:
            if feature in blocked:
                values.append(float("nan"))
                low.append(0.0)
                high.append(0.0)
                continue
            estimate = bootstrap_mean(subset[subset["patch"].astype(str) == feature], "effect",
                                      iterations=400)
            magnitude = abs(estimate.value) if np.isfinite(estimate.value) else 0.0
            values.append(sign * magnitude)
            spread = (abs(estimate.ci_high - estimate.ci_low) / 2.0
                      if np.isfinite(estimate.ci_low) and np.isfinite(estimate.ci_high) else 0.0)
            low.append(spread)
            high.append(spread)
        ax_primary.barh(positions, values, height=0.62, color=color, edgecolor=st.INK,
                        linewidth=0.5, zorder=2, label=_direction_label(subset, direction)
                        .replace("\n", " "))
        ax_primary.errorbar(values, positions, xerr=[low, high], fmt="none", ecolor=st.INK,
                            elinewidth=0.9, capsize=2.5, zorder=3)
        drawn = True
    for index, feature in enumerate(features):
        if feature not in blocked:
            continue
        centre = float(positions[index])
        ax_primary.axhspan(centre - 0.45, centre + 0.45, color="0.92", zorder=0)
        ax_primary.text(0.5, centre, "not exposed as a separable\nstage by this architecture",
                        transform=ax_primary.get_yaxis_transform(), ha="center", va="center",
                        fontsize=plt.rcParams["legend.fontsize"] - 1.5, color=st.INK_SOFT,
                        zorder=1)
    if not drawn or len(blocked) == len(features):
        _no_data(ax_primary)
    else:
        ax_primary.axvline(0.0, color=st.INK, linewidth=1.0, zorder=4)
        ax_primary.set_yticks(positions, [_patch_label(primary, f) for f in features])
        limit = max(abs(np.array(ax_primary.get_xlim()))) or 1.0
        ax_primary.set_xlim(-limit, limit)
        ax_primary.set_xticks(ax_primary.get_xticks(),
                              [f"{abs(t):.2f}" for t in ax_primary.get_xticks()])
        ax_primary.set_xlim(-limit, limit)
        ax_primary.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False,
                          fontsize=plt.rcParams["legend.fontsize"] - 1)
        st.light_grid(ax_primary, "x")
    ax_primary.set(xlabel="Size of the paired effect against the sham hook",
                   ylabel="Upstream feature", title="Does it become a sink?")

    # (b) the reciprocal score on every measure: transfer minus prevention, which
    # is the quantity the claim is actually about, one column per measure.
    reciprocal = _reciprocal_scores(patches)
    if reciprocal.empty or not np.isfinite(reciprocal.to_numpy()).any():
        _no_data(ax_grid)
        ax_grid.set_title("All measures at once")
    else:
        scaled = reciprocal.div(reciprocal.abs().max().replace(0, np.nan), axis=1).fillna(0.0)
        image = ax_grid.imshow(scaled.to_numpy(), cmap=st.diverging_cmap(), vmin=-1, vmax=1,
                               aspect="auto")
        ax_grid.set_xticks(range(scaled.shape[1]),
                           [_endpoint_short(c) for c in scaled.columns], rotation=20, ha="right")
        ax_grid.set_yticks(range(scaled.shape[0]),
                           [_patch_label(patches, r) for r in scaled.index])
        for i in range(scaled.shape[0]):
            for j in range(scaled.shape[1]):
                raw = reciprocal.iat[i, j]
                if np.isfinite(raw):
                    ax_grid.text(j, i, f"{raw:+.2f}", ha="center", va="center",
                                 fontsize=plt.rcParams["legend.fontsize"] - 1,
                                 color=st.INK if abs(scaled.iat[i, j]) < 0.6 else "white")
        st.colorbar(fig, image, ax_grid, "Reciprocal score, relative to\nthe largest on that measure")
        ax_grid.set(xlabel="Measure", ylabel="Upstream feature", title="All measures at once")

    # (c) what already differs before the write
    if has_separation:
        ax = axes[2]
        means = separation.groupby("feature")["separation"].mean().dropna().sort_values()
        if means.empty:
            _no_data(ax, "no pre-write feature could be read")
        else:
            ax.barh(np.arange(len(means)), means.to_numpy(), height=0.62, color=BLUE,
                    edgecolor=st.INK, linewidth=0.5, zorder=2)
            ax.axvline(0.5, color=st.RULE, linestyle=(0, (4, 2)), linewidth=1.0, zorder=1)
            ax.annotate("chance", xy=(0.5, 1.0), xycoords=("data", "axes fraction"),
                        xytext=(3, -4), textcoords="offset points", ha="left", va="top",
                        color=st.INK_SOFT, fontsize=plt.rcParams["legend.fontsize"] - 1)
            ax.set_yticks(np.arange(len(means)), [_feature_label(separation, f) for f in means.index])
            ax.set_xlim(0.4, 1.0)
            st.light_grid(ax, "x")
        ax.set(xlabel="Separability of eventual registers\nfrom matched ordinary tokens",
               ylabel="Upstream feature", title="Before the write")

    caption = (
        "Q4. What selects the sparse positions that receive the register write. "
        "(a) The reciprocal signature on the primary endpoint: the magnitude of the paired effect "
        "of copying one upstream feature from an eventual register into a matched ordinary token "
        "(right) and of replacing it at an eventual register with the matched ordinary value "
        "(left), each against a sham hook on the same prompt, seed and noise. A credible writer "
        "feature produces both. "
        "(b) The reciprocal score -- the transfer effect minus the prevention effect -- on every "
        "measure the recipient could develop: whether it becomes a sink, whether it acquires the "
        "dominant channel, whether it aligns with the register direction, and whether its norm "
        "rises. Colour is that score divided by the largest score on the same measure, since the "
        "measures do not share units; the printed number is the raw score. "
        "(c) Rank-based separability of each pre-write feature between eventual registers and "
        "matched ordinary tokens; 0.5 is chance. This panel localises candidates and is not itself "
        "causal evidence. Bars carry prompt-clustered 95\\% bootstrap intervals.")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _panel_letters(fig, axes)
    return _finish(fig, caption, path, tight=False)


def _blocked_features(frame: pd.DataFrame, features: Sequence[str]) -> set:
    """Features the run could not ask, as opposed to asked and found nothing.

    A row is the evidence either way: the runner records an unsupported feature
    with ``supported`` false rather than omitting it, so a feature whose every row
    says so was never tested.  Falling back to "no finite effect anywhere" keeps
    this working on a table written before that column existed.
    """
    blocked = set()
    for feature in features:
        rows = frame[frame["patch"].astype(str) == str(feature)]
        if rows.empty:
            continue
        if "supported" in rows.columns:
            if not rows["supported"].astype(bool).any():
                blocked.add(feature)
        elif not np.isfinite(rows["effect"].to_numpy(dtype=float)).any():
            blocked.add(feature)
    return blocked


def _reciprocal_scores(patches: pd.DataFrame) -> pd.DataFrame:
    """Transfer effect minus prevention effect, per feature and measure.

    Copying a feature in should create register behaviour and taking it out
    should prevent it, so the difference of the two is the single number the
    writer-selection claim rests on. A one-sided feature scores near the size
    of its one side and is visibly weaker than a genuinely reciprocal one.
    """
    means = patches.pivot_table(index="patch", columns=["endpoint", "direction"], values="effect",
                                aggfunc="mean")
    rows = {}
    for endpoint in dict.fromkeys(means.columns.get_level_values(0)):
        block = means[endpoint]
        transfer = block.get("register_to_ordinary")
        prevent = block.get("ordinary_to_register")
        if transfer is None or prevent is None:
            continue
        rows[endpoint] = transfer - prevent
    return pd.DataFrame(rows)


def _endpoint_short(endpoint: str) -> str:
    return {"capture_rate": "Sink capture", "dominant_channel_value": "Dominant channel",
            "cosine": "Direction", "target_norm_ratio": "Norm"}.get(str(endpoint), _pretty(endpoint))


def _direction_short(direction: str) -> str:
    return {"register_to_ordinary": "transfer", "ordinary_to_register": "prevent"}.get(
        str(direction), _pretty(direction))


def _endpoint_title(frame: pd.DataFrame, endpoint: str) -> str:
    if "endpoint_label" in frame.columns and frame["endpoint_label"].notna().any():
        return str(frame["endpoint_label"].dropna().iloc[0])
    return _pretty(endpoint)


def _direction_label(frame: pd.DataFrame, direction: str) -> str:
    if "direction_label" in frame.columns:
        match = frame.loc[frame["direction"] == direction, "direction_label"].dropna()
        if len(match):
            return str(match.iloc[0]).replace(" ", "\n", 1)
    return _pretty(direction)


def _patch_label(frame: pd.DataFrame, patch: str) -> str:
    if "patch_label" in frame.columns:
        match = frame.loc[frame["patch"] == patch, "patch_label"].dropna()
        if len(match):
            return str(match.iloc[0])
    return _pretty(patch)


def _feature_label(frame: pd.DataFrame, feature: str) -> str:
    if "feature_label" in frame.columns:
        match = frame.loc[frame["feature"] == feature, "feature_label"].dropna()
        if len(match):
            return str(match.iloc[0])
    return _pretty(feature)


# ============================================================ sufficiency ladder
def fig_q5_sufficiency_ladder(ladder: pd.DataFrame, path=None):
    """The earliest representation whose transplant reproduces a natural sink."""
    _need(ladder, ["stage", "position", "capture_rate"], "Q5 figure")
    if "temporal_endpoint" in ladder:
        ladder = ladder[ladder["temporal_endpoint"] == "same_operation"]
    stages = list(dict.fromkeys(ladder["stage"].astype(str)))
    group_column = "position"
    if "transfer_mode" in ladder.columns:
        ladder = ladder.copy()
        ladder["display_group"] = ladder["position"].astype(str) + " · " + ladder["transfer_mode"].astype(str)
        group_column = "display_group"
    positions = list(dict.fromkeys(ladder[group_column].astype(str)))
    fig, ax = plt.subplots(figsize=(10.4, 4.6))
    width = 0.8 / max(len(positions), 1)
    palette = {position: CONDITION_CYCLE[i % len(CONDITION_CYCLE)]
               for i, position in enumerate(positions)}

    for index, position in enumerate(positions):
        subset = ladder[ladder[group_column].astype(str) == position]
        means = subset.groupby("stage", observed=True)["capture_rate"].mean().reindex(stages)
        offsets = np.arange(len(stages)) + (index - (len(positions) - 1) / 2) * width
        heights = means.to_numpy(dtype=float)
        drawn = np.nan_to_num(heights, nan=0.0)
        ax.bar(offsets, drawn, width=width * 0.9, color=palette[position], edgecolor=st.INK,
               linewidth=0.5, label=(_position_label(subset, str(subset["position"].iloc[0])) +
                                     (f" ({subset['transfer_mode'].iloc[0]})"
                                      if "transfer_mode" in subset else "")), zorder=2)
    # A stage the architecture does not expose is marked as untested, once, rather
    # than drawn as a bar of height zero, which would read as a measured null.
    untested = [i for i, stage in enumerate(stages)
                if not np.isfinite(ladder.loc[ladder["stage"].astype(str) == stage,
                                              "capture_rate"].to_numpy(dtype=float)).any()]
    for index in untested:
        ax.axvspan(index - 0.44, index + 0.44, color="0.92", zorder=0)
        ax.text(index, 0.5, "not exposed as a separable\nstage by this architecture",
                transform=ax.get_xaxis_transform(), ha="center", va="center", rotation=90,
                fontsize=plt.rcParams["legend.fontsize"] - 1.5, color=st.INK_SOFT, zorder=1)

    if "any_register_clean_rate" in ladder.columns and ladder["any_register_clean_rate"].notna().any():
        natural = float(ladder["any_register_clean_rate"].mean())
        ax.axhline(natural, color=VERMILLION, linestyle=(0, (5, 2)), linewidth=1.4, zorder=3,
                   label="Any natural register, clean run (context only)")
    if "matched_clean_rate" in ladder.columns:
        for index, position in enumerate(positions):
            baseline = ladder.loc[ladder[group_column].astype(str) == position,
                                  "matched_clean_rate"].mean()
            if np.isfinite(baseline):
                ax.scatter(np.arange(len(stages)), np.full(len(stages), baseline), marker="_",
                           s=90, color=palette[position], linewidths=1.5, zorder=4)
    ax.set_xticks(np.arange(len(stages)), [_stage_label(ladder, s) for s in stages],
                  rotation=18, ha="right")
    ax.set(xlabel="Transplanted representation, from residual state to final key",
           ylabel="Fraction of heads whose sink becomes the recipient",
           title="Where the register state becomes sufficient")
    ax.set_ylim(0, 1.0)
    st.light_grid(ax, "y")
    ax.legend(loc="upper left", frameon=False, fontsize=plt.rcParams["legend.fontsize"] - 1,
              ncol=1, title="Recipient position",
              title_fontsize=plt.rcParams["legend.fontsize"] - 1)

    caption = (
        "Q5. The residual-to-key sufficiency ladder. Each rung transplants one representation of a "
        "natural register into a recipient token and measures how often heads then sink on the "
        "recipient in the same attention operation. Copying the direction alone at the recipient's own magnitude is the known "
        "negative result; climbing towards the tensor attention actually consumes locates the "
        "missing ingredient. Coloured ticks give the matched clean capture of that exact recipient; "
        "the dashed line is the any-register clean rate and is context, not the bar's baseline. "
        "COPY and MOVE are retained as separate rows in the numeric table. Rungs an architecture "
        "does not expose as a separable module are marked and left "
        "untested rather than approximated by a neighbouring tensor.")
    return _finish(fig, caption, path)


def _position_label(frame: pd.DataFrame, position: str) -> str:
    if "position_label" in frame.columns:
        match = frame.loc[frame["position"] == position, "position_label"].dropna()
        if len(match):
            return str(match.iloc[0])
    return _pretty(position)


def _stage_label(frame: pd.DataFrame, stage: str) -> str:
    if "stage_label" in frame.columns:
        match = frame.loc[frame["stage"] == stage, "stage_label"].dropna()
        if len(match):
            return str(match.iloc[0]).replace(", ", ",\n")
    return _pretty(stage)


# ============================================================ register dissolution
def fig_q6_dissolution(trajectory: pd.DataFrame, lifetime: Optional[pd.DataFrame] = None,
                       path=None):
    """Whether late channel competition or geometric rotation ends the state."""
    _need(trajectory, ["condition", "layer", "alpha", "perpendicular_norm"], "Q6 figure")
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.1))
    ax_components, ax_channels, ax_lifetime = axes

    conditions = list(dict.fromkeys(trajectory["condition"].astype(str)))
    palette = _condition_colors(conditions)
    palette["clean"] = st.INK
    for condition in conditions:
        group = trajectory[trajectory["condition"].astype(str) == condition]
        curve = group.groupby("layer", observed=True).mean(numeric_only=True).reset_index()
        color = palette[condition]
        ax_components.plot(curve["layer"], curve["alpha"], color=color, linewidth=1.6,
                           label=_condition_display(group, condition), zorder=3)
        ax_components.plot(curve["layer"], curve["perpendicular_norm"], color=color,
                           linewidth=1.2, linestyle=(0, (4, 2)), alpha=0.85, zorder=2)
        if {"dominant_channel", "competing_channel"} <= set(curve.columns) and condition == "clean":
            ax_channels.plot(curve["layer"], curve["dominant_channel"], color=BLUE, marker="o",
                             label="Dominant register channel")
            ax_channels.plot(curve["layer"], curve["competing_channel"], color=VERMILLION,
                             marker="s", label="Late-growing competing channel")
    ax_components.set(xlabel="Transformer layer",
                      ylabel="Component magnitude of the register state",
                      title="Direction versus the rest")
    st.light_grid(ax_components, "y")
    _integer_layers(ax_components)
    style_handles = [plt.Line2D([], [], color=st.INK, linewidth=1.6,
                                label="Along the register direction"),
                     plt.Line2D([], [], color=st.INK, linewidth=1.2, linestyle=(0, (4, 2)),
                                label="Orthogonal component")]
    ax_components.add_artist(ax_components.legend(
        handles=style_handles, loc="upper right", frameon=False,
        fontsize=plt.rcParams["legend.fontsize"] - 1.5))

    if not ax_channels.get_legend_handles_labels()[0]:
        _no_data(ax_channels, "channel values were not recorded")
    else:
        ax_channels.legend(frameon=False, fontsize=plt.rcParams["legend.fontsize"] - 1)
        st.light_grid(ax_channels, "y")
        _integer_layers(ax_channels)
    ax_channels.set(xlabel="Transformer layer", ylabel="Mean absolute activation at the registers",
                    title="Channel handover, clean run")

    if lifetime is None or lifetime.empty:
        _no_data(ax_lifetime)
        ax_lifetime.set(title="Does the register live longer?")
    else:
        endpoints = [column for column in ("cosine_lifetime_shift", "projection_lifetime_shift",
                                            "high_norm_lifetime_shift", "sink_lifetime_shift")
                     if column in lifetime and lifetime[column].notna().any()]
        if endpoints:
            long = lifetime.melt(
                id_vars=[c for c in ("condition", "condition_label", "prompt_id", "seed")
                         if c in lifetime], value_vars=endpoints,
                var_name="lifetime_endpoint", value_name="endpoint_shift")
            long["condition_endpoint"] = (long["condition"].astype(str) + " / " +
                                           long["lifetime_endpoint"].str.replace("_lifetime_shift", ""))
            long["condition_endpoint_label"] = (long["condition_label"].astype(str) + " / " +
                                                  long["lifetime_endpoint"].str.replace("_lifetime_shift", ""))
            plot_frame, group, column, label = (long, "condition_endpoint", "endpoint_shift",
                                                 "condition_endpoint_label")
        else:
            column = ("lifetime_shift" if "lifetime_shift" in lifetime.columns
                      and lifetime["lifetime_shift"].notna().any() else "half_life_shift")
            plot_frame, group, label = lifetime, "condition", "condition_label"
        _effect_bars_by(ax_lifetime, plot_frame, group, column,
                        label_column=label,
                        xlabel="Change in register lifetime (layers)",
                        ylabel="Late intervention",
                        title="Does the register live longer?")

    caption = (
        "Q6. Causal test of what ends the register state. "
        "(a) The register state decomposed as a component along the frozen direction $v^*$ (solid) "
        "and everything orthogonal to it (dashed), through the late register zone. "
        "(b) Mean absolute activation of the dominant register channel and of the late-growing "
        "competing channel at the register tokens in the clean run, which is the handover the "
        "intervention targets. "
        "(c) Paired change in register lifetime under late suppression of the competitor, early "
        "amplification of it, a direction refresh that re-projects towards $v^*$ at unchanged "
        "magnitude, and matched random-channel and matched-magnitude controls. Prolongation under "
        "channel suppression supports channel competition; prolongation under direction refresh "
        "alone indicates a geometric rotation that no single channel explains.")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    _shared_condition_legend(fig, (ax_components,), ncol=4, y=0.005)
    _panel_letters(fig, axes)
    return _finish(fig, caption, path, tight=False)


def _condition_display(frame: pd.DataFrame, condition: str) -> str:
    if condition == "clean":
        return "Clean run"
    if "condition_label" in frame.columns and frame["condition_label"].notna().any():
        return str(frame["condition_label"].dropna().iloc[0])
    return _pretty(condition)


# ========================================================== cross-question
def fig_evidence_forest(evidence: pd.DataFrame, path=None):
    """Every predeclared claim as one effect with its prompt-clustered interval."""
    _need(evidence, ["question", "claim", "effect", "ci_low", "ci_high"], "evidence figure")
    frame = evidence.dropna(subset=["effect"]).reset_index(drop=True)
    if frame.empty:
        fig, ax = plt.subplots(figsize=(8.6, 3.0))
        _no_data(ax, "no claim could be evaluated on this run")
        return _finish(fig, "Evidence summary: no claim produced a measurable effect.", path)
    height = max(2.6, 0.52 * len(frame) + 1.4)
    fig, ax = plt.subplots(figsize=(9.6, height))
    positions = np.arange(len(frame))[::-1]
    supported = frame["verdict"].astype(str).str.startswith(("supported", "equivalent"))
    colors = [GREEN if flag else VERMILLION for flag in supported]
    low = (frame["effect"] - frame["ci_low"]).to_numpy(dtype=float)
    high = (frame["ci_high"] - frame["effect"]).to_numpy(dtype=float)
    ax.errorbar(frame["effect"], positions, xerr=[np.nan_to_num(low), np.nan_to_num(high)],
                fmt="none", ecolor=st.INK, elinewidth=1.0, capsize=3.0, zorder=2)
    ax.scatter(frame["effect"], positions, s=46, c=colors, edgecolor=st.INK, linewidth=0.6,
               zorder=3)
    ax.axvline(0.0, color=st.RULE, linewidth=1.0, zorder=1)
    labels = [f"{row.question}  {row.claim}" for row in frame.itertuples()]
    ax.set_yticks(positions, labels)
    ax.set(xlabel="Paired effect against the matched reference run",
           title="Predeclared causal claims and their evidence")
    st.light_grid(ax, "x")
    ax.set_ylim(-0.8, len(frame) - 0.2)
    caption = (
        "Summary of the predeclared causal claims. Each point is the paired effect of one "
        "intervention against its matched reference run on the same prompt, seed and noise; bars "
        "are prompt-clustered 95\\% bootstrap intervals, so several seeds of one prompt are not "
        "treated as independent. Green marks a claim supported in the predicted direction (or "
        "shown equivalent within its predeclared margin), vermillion a claim that is not.")
    return _finish(fig, caption, path)


def fig_circuit_summary(status: Mapping[str, str], effects: Optional[Mapping[str, str]] = None,
                        path=None):
    """The causal chain as a diagram, with each arrow's status from this run.

    One falsifiable causal explanation, drawn rather than left as a catalogue
    of ablations, with each arrow annotated by the experiment that tested it.
    """
    stages = [("Writer", "sparse upstream write"), ("Dominant channel", "$c^*$"),
              ("Register direction", "$v^*$"), ("Key geometry", "query-key advantage"),
              ("Sink routing", "attention anchor"), ("Dissolution", "state ends")]
    arrows = ["writer_to_channel", "channel_to_direction", "direction_to_key",
              "key_to_sink", "sink_to_end"]
    arrow_questions = {"writer_to_channel": "Q4", "channel_to_direction": "Q2",
                       "direction_to_key": "Q5", "key_to_sink": "Q1", "sink_to_end": "Q6"}
    colors = {"supported": GREEN, "refuted": VERMILLION, "untested": SHAM, "partial": ORANGE}

    fig, ax = plt.subplots(figsize=(13.0, 3.3))
    width, gap = 1.75, 0.72
    for index, (title, subtitle) in enumerate(stages):
        x = index * (width + gap)
        ax.add_patch(plt.Rectangle((x, 0.42), width, 0.62, facecolor="white",
                                   edgecolor=st.INK, linewidth=1.0, zorder=3))
        ax.text(x + width / 2, 0.84, title, ha="center", va="center", zorder=4,
                fontsize=plt.rcParams["axes.labelsize"] - 0.5, fontweight="bold")
        ax.text(x + width / 2, 0.58, subtitle, ha="center", va="center", zorder=4,
                fontsize=plt.rcParams["legend.fontsize"] - 1, color=st.INK_SOFT)
        if index < len(arrows):
            key = arrows[index]
            state = str(status.get(key, "untested")).lower()
            color = colors.get(state, SHAM)
            ax.annotate("", xy=(x + width + gap - 0.06, 0.73), xytext=(x + width + 0.06, 0.73),
                        arrowprops=dict(arrowstyle="-|>", color=color, linewidth=2.0,
                                        shrinkA=0, shrinkB=0), zorder=2)
            ax.text(x + width + gap / 2, 1.10, arrow_questions[key], ha="center", va="bottom",
                    fontsize=plt.rcParams["legend.fontsize"] - 1, color=color, fontweight="bold")
            note = (effects or {}).get(key, state)
            ax.text(x + width + gap / 2, 0.30, str(note), ha="center", va="top", rotation=0,
                    fontsize=plt.rcParams["legend.fontsize"] - 2, color=st.INK_SOFT)
    ax.set_xlim(-0.35, len(stages) * (width + gap) - gap + 0.35)
    ax.set_ylim(0.0, 1.35)
    ax.axis("off")
    handles = [plt.Line2D([], [], color=color, linewidth=2.4, label=state.capitalize())
               for state, color in colors.items()]
    ax.legend(handles=handles, frameon=False, ncol=4, loc="lower center",
              bbox_to_anchor=(0.5, -0.08), fontsize=plt.rcParams["legend.fontsize"] - 1)
    caption = (
        "The working causal model and what this run establishes about each arrow. Boxes are the "
        "stages of the register circuit, from the sparse upstream write through the dominant "
        "channel, the shared register direction, the resulting query-key advantage and the sink it "
        "produces, to the late computation that ends the state. Each arrow is labelled with the "
        "question that tests it and coloured by the outcome of that test on this checkpoint.")
    return _finish(fig, caption, path, tight=False)


# ------------------------------------------------------------------- tables
EVIDENCE_COLUMNS = ("question", "checkpoint", "claim", "primary_endpoint", "effect",
                    "ci_low", "ci_high", "n_units", "n_prompts", "predicted_direction_share",
                    "verdict")


def evidence_table(rows: pd.DataFrame) -> pd.DataFrame:
    """Compact claim-by-claim table for the notebook and the appendix."""
    _need(rows, ["question", "claim", "primary_endpoint", "effect", "ci_low", "ci_high", "verdict"],
          "evidence table")
    columns = [c for c in EVIDENCE_COLUMNS if c in rows.columns]
    return rows[columns].sort_values(["question", "claim"]).reset_index(drop=True)


def verdict_table(results: Mapping[str, Any]) -> pd.DataFrame:
    """One plain-language answer per question, for the top of a results section."""
    from .questions import QUESTION_TITLES

    rows = []
    for key in sorted(results):
        result = results[key]
        rows.append({"question": key.upper(), "asks": QUESTION_TITLES.get(key, ""),
                     "answer": getattr(result, "verdict", ""),
                     "rows_measured": int(len(getattr(result, "tidy", pd.DataFrame())))})
    return pd.DataFrame(rows)


# ===================================================  register-preserving quantization
# Identity in these figures is carried by direct labels, and role by colour *and*
# marker shape, so no condition is distinguished by colour alone.  The reference
# role is achromatic on purpose: it is a baseline, not a series, and drawing it
# in a categorical hue would invite it to be read as one more competitor.
Q11_ROLE_COLOR = {"condition": BLUE, "control": VERMILLION, "reference": SHAM}
Q11_ROLE_MARKER = {"condition": "o", "control": "s", "reference": "D"}
Q11_LINE_COLORS = (BLUE, VERMILLION, GREEN, ORANGE)

# Lower is better for an error; higher is better for an agreement.
Q11_METRIC_SENSE = {
    "lpips": ("LPIPS from the unquantized image", True),
    "rmse": ("pixel RMSE from the unquantized image", True),
    "psnr": ("PSNR against the unquantized image (dB)", False),
    "state_relative_error": ("register state relative error", True),
    "vstar_projection_error": ("relative error in the $v^*$ projection", True),
    "vstar_cosine": ("cosine with the unquantized register state", False),
    "sink_agreement": ("fraction of heads whose sink is unchanged", False),
    "highnorm_jaccard": ("overlap of the high-norm token set", False),
    "register_key_rank": ("register key rank (lower binds attention harder)", True),
}

# Short names for scatter labels. A sentence-length label is right beside a bar and
# wrong beside a point: ten of them in one panel collide into illegibility.
Q11_SHORT = {
    "clean": "unquantized",
    "uniform_int4": "uniform INT4",
    "uniform_int8": "uniform INT8",
    "dominant_channel": "dominant channel",
    "vstar_all_layers": "$v^*$, all layers",
    "vstar_lifecycle": "$v^*$, register zone",
    "vstar_measured_window": "$v^*$, measured window",
    "vstar_early_window": "$v^*$, early window",
    "vstar_late_window": "$v^*$, late window",
    "magnitude_topk": "largest coordinate",
    "random_direction": "random direction",
    "lowrank_absorption": "low-rank absorption",
}


def _q11_metric(frame: pd.DataFrame, preferred: Sequence[str]) -> Optional[str]:
    """The first requested metric this run actually measured.

    An image metric needs a decoder and a perceptual metric needs its weights; a
    synthetic or offline run has neither.  Choosing here, rather than assuming, is
    what lets the same figure serve both cases and say which one it drew.
    """
    for name in preferred:
        if name in frame.columns and frame[name].notna().any():
            return name
    return None


def _q11_frontier(x: np.ndarray, y: np.ndarray, lower_is_better: bool) -> np.ndarray:
    """Indices of the non-dominated points, left to right along cost.

    A point is on the frontier when nothing cheaper is also at least as good. This
    is the claim the Pareto panel makes, so it is computed rather than eyeballed.
    """
    order = np.argsort(x)
    keep, best = [], None
    for index in order:
        value = y[index]
        if value != value:
            continue
        if best is None or (value < best if lower_is_better else value > best):
            keep.append(int(index))
            best = value
    return np.array(keep, dtype=int)


def _spread_labels(values: Sequence[float], minimum_gap: float) -> List[float]:
    """Nudge label anchors apart just enough to stop them overlapping.

    A scatter of allocation policies clusters hard, and that clustering *is* the result
    since the policies cost nearly the same, so the labels have to be separated
    without moving the marks. One upward pass from the lowest anchor keeps the order
    the data has and moves each label the least distance that clears its neighbour.
    """
    order = sorted(range(len(values)), key=lambda i: values[i])
    placed = list(values)
    for position, index in enumerate(order):
        if position == 0:
            continue
        previous = placed[order[position - 1]]
        if placed[index] - previous < minimum_gap:
            placed[index] = previous + minimum_gap
    return placed


def fig_q11_quantization_pareto(pareto: pd.DataFrame, path=None, *, checkpoint: str = ""):
    """Cost against both outcomes: the picture, and the circuit that produced it.

    Two panels rather than two y-axes on one. The quantity on the left is a property
    of the output and the quantity on the right is a property of the mechanism; they
    share only the cost axis, and overlaying them would invite a reader to compare
    two incommensurable scales against a shared gridline.

    The unquantized reference is drawn as a rule rather than as a point. It sits at
    16 bits per coordinate, far from every quantized policy, and plotting it on the
    cost axis would compress the entire comparison into the left fifth of the panel
    to show one value that is a baseline rather than a competitor.
    """
    _need(pareto, ["condition", "condition_label", "role", "bits_per_coordinate"],
          "Q11 Pareto figure")
    image_metric = _q11_metric(pareto, ("lpips", "rmse", "psnr"))
    mechanism_metric = _q11_metric(pareto, ("vstar_cosine", "state_relative_error",
                                            "vstar_projection_error"))
    quantized = pareto[pareto["condition"] != "clean"]
    unquantized = pareto[pareto["condition"] == "clean"]

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.6))
    for ax, metric in zip(axes, (image_metric, mechanism_metric)):
        if metric is None or quantized.empty or not quantized[metric].notna().any():
            _no_data(ax, "not measurable in this run")
            continue
        label, lower_is_better = Q11_METRIC_SENSE.get(metric, (metric, True))
        frame = quantized[quantized[metric].notna()].sort_values("bits_per_coordinate")
        x = frame["bits_per_coordinate"].to_numpy(dtype=float)
        y = frame[metric].to_numpy(dtype=float)
        frontier = _q11_frontier(x, y, lower_is_better)
        if len(frontier) > 1:
            ax.plot(x[frontier], y[frontier], color=st.RULE, lw=1.0, ls=(0, (4, 2)), zorder=1)
        for index, row in enumerate(frame.itertuples()):
            role = str(getattr(row, "role", "condition"))
            ax.scatter(x[index], y[index], s=54, zorder=3,
                       color=Q11_ROLE_COLOR.get(role, BLUE),
                       marker=Q11_ROLE_MARKER.get(role, "o"),
                       edgecolor="white", linewidth=0.8)
        span = float(np.nanmax(y) - np.nanmin(y)) or 1.0
        anchors = _spread_labels(list(y), span * 0.085)
        for index, row in enumerate(frame.itertuples()):
            name = Q11_SHORT.get(str(row.condition), str(row.condition_label))
            ax.annotate(name, (x[index], y[index]), xytext=(x[index] + 0.045, anchors[index]),
                        fontsize=7.0, color=st.INK_SOFT, va="center",
                        arrowprops=dict(arrowstyle="-", color=st.GRID, lw=0.6,
                                        shrinkA=0, shrinkB=2))
        if not unquantized.empty and unquantized[metric].notna().any():
            value = float(unquantized[metric].iloc[0])
            ax.axhline(value, color=SHAM, lw=1.0, ls=(0, (2, 2)), zorder=0)
            ax.annotate("unquantized reference (16 bits/coordinate)", (0.99, value),
                        xycoords=("axes fraction", "data"), textcoords="offset points",
                        xytext=(0, 4), ha="right", fontsize=7.0, color=st.INK_SOFT)
        ax.set_xlim(float(x.min()) - 0.15, float(x.max()) + 0.75)
        ax.set_xlabel("effective activation cost (bits per coordinate)")
        ax.set_ylabel(label + ("  (lower is better)" if lower_is_better
                               else "  (higher is better)"))
        st.open_box(ax)
        st.light_grid(ax, "y")
    handles = [plt.Line2D([], [], marker=Q11_ROLE_MARKER[r], color="none",
                          markerfacecolor=Q11_ROLE_COLOR[r], markeredgecolor="white",
                          markersize=7, label=r)
               for r in ("condition", "control", "reference")]
    axes[0].legend(handles=handles, frameon=False, fontsize=7.5, loc="best")
    _panel_letters(fig, axes)
    head = f"{checkpoint}: " if checkpoint else ""
    return _finish(fig, (
        f"{head}effective activation bit cost against output fidelity (a) and mechanism "
        f"fidelity (b), one point per allocation policy. Cost counts the low-bit grid, the "
        f"per-token quantizer scales, and every high-precision scalar, including the index "
        f"bits a per-token choice of coordinates must transmit. The dashed curve joins the "
        f"non-dominated policies and the horizontal rule is the unquantized reference. "
        f"Quantize-dequantize simulation of an activation policy: weights are untouched and "
        f"no low-bit kernel is involved, so these results speak to numerical robustness and "
        f"not to latency or memory."), path)


def fig_q11_equal_budget(pareto: pd.DataFrame, path=None, *, checkpoint: str = "",
                         group: Optional[Sequence[str]] = None):
    """The headline ranking, with each policy's cost printed beside its name.

    Horizontal bars because the policy names are sentences, and sorted by the metric
    because rank is the question. The cost appears on every tick label rather than
    only in the caption: a ranking of policies that cost different amounts is not a
    result, and a reader must be able to see which rows are comparable.
    """
    _need(pareto, ["condition", "condition_label", "role", "bits_per_coordinate"],
          "Q11 equal-budget figure")
    metric = _q11_metric(pareto, ("lpips", "rmse", "state_relative_error",
                                  "vstar_projection_error"))
    if metric is None:
        fig, ax = plt.subplots(figsize=(7.2, 3.6))
        _no_data(ax, "no fidelity metric was measurable in this run")
        return _finish(fig, "Q11 equal-budget comparison: no fidelity metric was measurable.",
                       path)
    label, lower_is_better = Q11_METRIC_SENSE.get(metric, (metric, True))
    frame = pareto[pareto[metric].notna()].copy()
    if group:
        frame = frame[frame["condition"].isin(list(group))]
    references = frame[frame["role"] == "reference"]
    bars = frame[frame["role"] != "reference"].sort_values(metric, ascending=not lower_is_better)
    if bars.empty:
        fig, ax = plt.subplots(figsize=(7.2, 3.6))
        _no_data(ax, "no policy to compare")
        return _finish(fig, "Q11 equal-budget comparison: no policy to compare.", path)

    fig, ax = plt.subplots(figsize=(8.4, max(2.8, 0.44 * len(bars) + 1.6)))
    positions = np.arange(len(bars))
    values = bars[metric].to_numpy(dtype=float)
    ax.barh(positions, values, height=0.62,
            color=[Q11_ROLE_COLOR.get(str(r), BLUE) for r in bars["role"]],
            edgecolor="white", linewidth=1.0)
    ax.set_yticks(positions)
    ax.set_yticklabels([f"{row.condition_label}\n({row.bits_per_coordinate:.2f} bits/coord)"
                        for row in bars.itertuples()], fontsize=7.5)
    ax.invert_yaxis()
    for position, value in zip(positions, values):
        ax.annotate(f"{value:.4g}", (value, position), textcoords="offset points",
                    xytext=(4, 0), va="center", fontsize=7.5, color=st.INK_SOFT)
    top = float(np.nanmax(values)) * 1.16
    for _, row in references.iterrows():
        value = float(row[metric])
        if not (0 <= value <= top):
            continue
        ax.axvline(value, color=SHAM, lw=1.0, ls=(0, (4, 2)), zorder=0)
        # A reference sitting on the axis itself (an unquantized run has zero error)
        # would have its centred label hang off the left edge of the figure.
        edge = "left" if value <= top * 0.02 else "right" if value >= top * 0.98 else "center"
        ax.annotate(str(row["condition_label"]), (value, -0.75), fontsize=7,
                    color=st.INK_SOFT, ha=edge, va="bottom")
    ax.set_xlim(0, top)
    ax.set_ylim(len(bars) - 0.4, -1.25)
    ax.set_xlabel(label + ("  (lower is better)" if lower_is_better else "  (higher is better)"))
    st.open_box(ax)
    st.light_grid(ax, "x")
    head = f"{checkpoint}: " if checkpoint else ""
    return _finish(fig, (
        f"{head}{label} for each activation allocation policy, with its effective cost on "
        f"the tick label. Dashed rules mark the unquantized reference and the unprotected "
        f"low-bit floor. Blue marks policies derived from the causal mechanism and orange "
        f"marks the equal-budget controls. Only rows at equal cost are directly comparable; "
        f"the accompanying budget table gives the full accounting."), path)


def fig_q11_depth_profile(mechanism: pd.DataFrame, path=None, *, checkpoint: str = "",
                          register_layers: Sequence[int] = (),
                          conditions: Optional[Sequence[str]] = None):
    """Where in depth each policy loses the register, with the zone it should protect.

    A single layer cannot separate a lifecycle window from protection everywhere:
    the two are identical until the first layer where they diverge. The comparison
    only exists across depth, so the profile is the figure, not a summary of it.
    """
    _need(mechanism, ["layer", "condition", "condition_label"], "Q11 depth profile")
    metric = _q11_metric(mechanism, ("vstar_projection_error", "state_relative_error",
                                     "vstar_cosine"))
    if metric is None:
        fig, ax = plt.subplots(figsize=(7.2, 3.6))
        _no_data(ax, "no mechanism metric was measurable in this run")
        return _finish(fig, "Q11 depth profile: no mechanism metric was measurable.", path)
    label, lower_is_better = Q11_METRIC_SENSE.get(metric, (metric, True))
    keep = list(conditions) if conditions else [
        c for c in ("uniform_int4", "magnitude_topk", "vstar_all_layers", "vstar_lifecycle")
        if c in set(mechanism["condition"])]
    frame = mechanism[mechanism["condition"].isin(keep)]
    if frame.empty:
        fig, ax = plt.subplots(figsize=(7.2, 3.6))
        _no_data(ax, "none of the selected policies ran")
        return _finish(fig, "Q11 depth profile: none of the selected policies ran.", path)

    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    if len(register_layers):
        ax.axvspan(min(register_layers) - 0.5, max(register_layers) + 0.5,
                   color=BLUE, alpha=0.07, zorder=0)
        ax.annotate("register zone", (float(np.mean(list(register_layers))), 0.02),
                    xycoords=("data", "axes fraction"), ha="center", fontsize=7.5,
                    color=st.INK_SOFT)
    ends = []
    for index, condition in enumerate(keep):
        subset = frame[frame["condition"] == condition]
        rows = subset.groupby("layer", observed=True)[metric].mean().reset_index() \
            .sort_values("layer")
        if rows.empty:
            continue
        color = Q11_LINE_COLORS[index % len(Q11_LINE_COLORS)]
        name = str(subset["condition_label"].iloc[0])
        ax.plot(rows["layer"], rows[metric], color=color, lw=2.0, marker="o", markersize=4,
                markeredgecolor="white", markeredgewidth=0.6, label=name)
        ends.append((float(rows["layer"].iloc[-1]), float(rows[metric].iloc[-1]),
                     Q11_SHORT.get(condition, name), color))
    if ends:
        anchors = _spread_labels([e[1] for e in ends],
                                 (max(e[1] for e in ends) - min(e[1] for e in ends) or 1.0) * 0.09)
        for (layer, value, name, color), anchor in zip(ends, anchors):
            ax.annotate(name, (layer, value), xytext=(layer + 0.22, anchor), fontsize=7.5,
                        color=color, va="center",
                        arrowprops=dict(arrowstyle="-", color=color, lw=0.6, alpha=0.5,
                                        shrinkA=0, shrinkB=2))
    layers = sorted({int(v) for v in frame["layer"].dropna()})
    if layers:
        step = max(1, len(layers) // 12)
        ax.set_xticks(layers[::step])
        ax.set_xlim(min(layers) - 0.5, max(layers) + (max(layers) - min(layers)) * 0.26 + 0.6)
    ax.set_xlabel("transformer block")
    ax.set_ylabel(label + ("  (lower is better)" if lower_is_better else "  (higher is better)"))
    st.open_box(ax)
    st.light_grid(ax, "y")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left", ncol=2)
    head = f"{checkpoint}: " if checkpoint else ""
    return _finish(fig, (
        f"{head}{label} by depth for each allocation policy, averaged over prompt-seed "
        f"units. The shaded band is the register zone taken from the frozen discovery "
        f"artifact. Policies that agree up to their first point of divergence coincide to "
        f"the left of that point; the lifecycle claim is about what happens inside the band "
        f"and after it."), path)


def fig_q11_window_ablation(pareto: pd.DataFrame, windows: pd.DataFrame, path=None, *,
                            checkpoint: str = "", n_layers: int = 0):
    """Each protected window drawn where it sits in depth, labelled with what it bought.

    Position is the independent variable in this ablation, so it is drawn on a depth
    axis rather than reduced to a category on a bar chart. A reader can then see that
    the wrong windows are the same width as the right one, which is the control that
    makes the comparison mean anything.
    """
    _need(pareto, ["condition", "condition_label"], "Q11 window ablation")
    _need(windows, ["window", "layers"], "Q11 window ablation")
    metric = _q11_metric(pareto, ("lpips", "rmse", "state_relative_error",
                                  "vstar_projection_error"))
    if metric is None:
        fig, ax = plt.subplots(figsize=(7.2, 3.2))
        _no_data(ax, "no fidelity metric was measurable in this run")
        return _finish(fig, "Q11 window ablation: no fidelity metric was measurable.", path)
    label, lower_is_better = Q11_METRIC_SENSE.get(metric, (metric, True))

    span = {str(row["window"]): json.loads(row["layers"]) for _, row in windows.iterrows()}
    wanted = [("vstar_all_layers", "all", "every layer"),
              ("vstar_lifecycle", "preregistered", "register zone (preregistered)"),
              ("vstar_measured_window", "measured", "register lifetime (measured)"),
              ("vstar_early_window", "early", "wrong window, early"),
              ("vstar_late_window", "late", "wrong window, late")]
    present = [(c, w, t) for c, w, t in wanted if c in set(pareto["condition"]) and span.get(w)]
    if not present:
        fig, ax = plt.subplots(figsize=(7.2, 3.2))
        _no_data(ax, "no windowed policy ran")
        return _finish(fig, "Q11 window ablation: no windowed policy ran.", path)

    values = {str(r["condition"]): float(r[metric]) for _, r in pareto.iterrows()
              if r[metric] == r[metric]}
    scored = [values[c] for c, _, _ in present if c in values]
    best = (min(scored) if lower_is_better else max(scored)) if scored else None
    fig, ax = plt.subplots(figsize=(8.8, max(2.6, 0.58 * len(present) + 1.3)))
    for index, (condition, window, title) in enumerate(present):
        layers = span[window]
        value = values.get(condition, float("nan"))
        winner = best is not None and value == best
        color = BLUE if winner else VERMILLION
        # A one-block window has zero length as a line; give every window the width of
        # the blocks it actually covers so a narrow one stays visible and to scale.
        ax.barh(index, max(layers) - min(layers) + 1, left=min(layers) - 0.5, height=0.46,
                color=color, alpha=0.9 if winner else 0.6, edgecolor="white", linewidth=0.8)
        blocks = len(layers)
        text = (f"{title} — {value:.4g}" if value == value else f"{title} — not measurable")
        ax.annotate(f"{text}   [{blocks} block{'s' if blocks != 1 else ''}]",
                    (max(layers) + 0.6, index), va="center", fontsize=7.5, color=st.INK_SOFT)
    ax.set_yticks(range(len(present)))
    ax.set_yticklabels([""] * len(present))
    ax.invert_yaxis()
    depth = n_layers or (max(max(span[w]) for _, w, _ in present) + 1)
    ax.set_xlim(-0.6, depth * 1.6)
    ax.set_xticks([t for t in range(0, depth, max(1, depth // 10))])
    ax.set_xlabel("transformer block")
    st.open_box(ax)
    st.light_grid(ax, "x")
    head = f"{checkpoint}: " if checkpoint else ""
    return _finish(fig, (
        f"{head}each protected window drawn at the depth it occupies, annotated with the "
        f"{label} it achieved and its width in blocks. The wrong windows have the same "
        f"width as the register zone, so position rather than budget is what differs "
        f"between them. Blue marks the best window on this metric."), path)


def fig_q11_operating_regime(regime: pd.DataFrame, path=None, *, checkpoint: str = "",
                             operating_point: Optional[float] = None):
    """Where an activation allocation policy can pay, and where it cannot.

    The limit on this whole application. Compressing activations cannot accelerate
    a layer whose time goes on reading weights, so the ceiling is Amdahl over memory
    traffic and it depends on the activation share. Drawing that as a curve over sequence
    length, rather than quoting one number, is what lets a reader see that the policy is
    irrelevant at batch one with full-precision weights and material alongside a weight
    quantizer, which is the regime it is actually for.
    """
    _need(regime, ["n_tokens", "weight_label", "speedup_ceiling", "activation_share"],
          "Q11 operating-regime figure")
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.2))
    colors = {label: color for label, color in
              zip(sorted(regime["weight_label"].unique()), (VERMILLION, BLUE))}

    for ax, column, label in (
            (axes[0], "activation_share", "activations as a share of memory traffic"),
            (axes[1], "speedup_ceiling", "upper bound on end-to-end speedup")):
        for weight_label, group in regime.groupby("weight_label", observed=True):
            rows = group.sort_values("n_tokens")
            ax.plot(rows["n_tokens"], rows[column], color=colors[weight_label], lw=2.0,
                    marker="o", markersize=4, markeredgecolor="white", markeredgewidth=0.6,
                    label=f"weights {weight_label}")
            ax.annotate(f"weights {weight_label}",
                        (rows["n_tokens"].iloc[-1], rows[column].iloc[-1]),
                        textcoords="offset points", xytext=(6, 0), fontsize=7.5,
                        color=colors[weight_label], va="center")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("tokens per forward pass (batch x resolution)")
        ax.set_ylabel(label)
        if operating_point:
            ax.axvline(operating_point, color=SHAM, lw=1.0, ls=(0, (4, 2)), zorder=0)
            ax.annotate("this run", (operating_point, 0.02), xycoords=("data", "axes fraction"),
                        rotation=90, fontsize=7, color=st.INK_SOFT, ha="right", va="bottom")
        st.open_box(ax)
        st.light_grid(ax, "y")
    axes[0].axhline(0.5, color=st.GRID, lw=1.0, zorder=0)
    axes[0].annotate("half of traffic", (0.02, 0.5), xycoords=("axes fraction", "data"),
                     textcoords="offset points", xytext=(0, 3), fontsize=7,
                     color=st.INK_SOFT)
    axes[1].axhline(1.0, color=st.GRID, lw=1.0, zorder=0)
    axes[1].annotate("no gain", (0.02, 1.0), xycoords=("axes fraction", "data"),
                     textcoords="offset points", xytext=(0, 3), fontsize=7, color=st.INK_SOFT)
    # Both series are labelled at their right end; the legend repeats that for a reader
    # scanning the panel rather than following a line, and sits low-right where neither
    # curve nor panel letter reaches.
    axes[0].legend(frameon=False, fontsize=7.5, loc="lower right")
    for ax in axes:
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom, top + (top - bottom) * 0.10)
    _panel_letters(fig, axes)
    head = f"{checkpoint}: " if checkpoint else ""
    return _finish(fig, (
        f"{head}(a) the share of per-block memory traffic that activations occupy, and (b) the "
        f"resulting ceiling on end-to-end speedup from compressing them, against the number of "
        f"tokens in a forward pass. The ceiling is Amdahl over memory traffic: only the "
        f"activation share can shrink, and only by the ratio of the bit widths. It is an upper "
        f"bound that a real system falls short of, and it is reported because it bounds the "
        f"argument in the right direction -- where the ceiling is near one, no amount of kernel "
        f"work would make an activation policy pay, and the honest claim is the memory one."),
        path)

# ==========================================  compute-schedule figures
def fig_q12_compute_grid(field: pd.DataFrame, path=None, *, checkpoint: str = "",
                         register_layers: Sequence[int] = ()):
    """The two fields side by side: the free signal, and the thing it must predict.

    A diffusion transformer's compute budget is a layer-by-step grid, every cell computed
    on every generation. Drawing both quantities on that grid is the whole argument in one
    picture: if the bright and dark regions of (a) line up with those of (b), a free
    probe can tell you which cells to skip.

    Sequential maps, one hue each, because both quantities are magnitudes. They are
    not a shared scale on purpose: they are different quantities in different units,
    and a shared colourbar would invite reading a correspondence in absolute value rather
    than in pattern.
    """
    _need(field, ["layer", "step", "d_alpha", "block_delta"], "Q12 compute grid")
    # The house sequential maps, one per quantity, rather than two new ones.
    panels = (("d_alpha", "change in the register projection  (the free probe)",
               st.PROBABILITY_CMAP),
              ("block_delta", "relative movement of the block output  (what a cache loses)",
               st.DENSITY_CMAP))
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.4))
    for ax, (column, label, cmap) in zip(axes, panels):
        grid = (field.groupby(["layer", "step"], observed=True)[column].mean()
                .unstack("step").sort_index())
        if grid.empty:
            _no_data(ax, "not measurable in this run")
            continue
        mesh = ax.imshow(grid.to_numpy(dtype=float), aspect="auto", origin="lower",
                         cmap=cmap, interpolation="nearest",
                         extent=(float(grid.columns.min()) - 0.5,
                                 float(grid.columns.max()) + 0.5,
                                 float(grid.index.min()) - 0.5,
                                 float(grid.index.max()) + 0.5))
        if len(register_layers):
            for edge in (min(register_layers) - 0.5, max(register_layers) + 0.5):
                ax.axhline(edge, color="white", lw=1.0, ls=(0, (4, 2)), alpha=0.9)
            # White on a sequential map is legible over the dark end and invisible over
            # the light end, and which end the register zone lands on is a property of the
            # data. A thin dark halo makes the label readable either way.
            ax.annotate("register zone", (float(grid.columns.max()), float(
                np.mean(list(register_layers)))), textcoords="offset points", xytext=(-4, 0),
                ha="right", va="center", fontsize=7, color="white",
                path_effects=[patheffects.withStroke(linewidth=1.6, foreground="0.15")])
        ax.set_xlabel("denoising step")
        ax.set_ylabel("transformer block")
        st.colorbar(fig, mesh, ax, label)
        st.open_box(ax)
    _panel_letters(fig, axes)
    head = f"{checkpoint}: " if checkpoint else ""
    return _finish(fig, (
        f"{head}(a) the step-to-step change in the register projection, read from the frozen "
        f"register tokens, and (b) the relative movement of each block's own output between "
        f"consecutive steps -- the error a cross-step cache would incur at that cell. Each "
        f"cell is one block evaluation; every cell is computed on every generation today. "
        f"Dashed lines mark the register zone from the frozen discovery artifact. The two "
        f"panels carry different quantities and are scaled independently, so the claim to "
        f"read here is a correspondence of pattern, not of value."), path, tight=False)


# Four series at most, so every one gets its own validated hue. The remaining predictors
# stay in the correlations table: a fifth line would have to reuse a colour, and identity
# by colour has to be unique or the legend stops meaning anything.
Q12_DRAWN = ("d_alpha", "d_cosine", "step", "layer")


def fig_q12_policy_curves(curves: pd.DataFrame, path=None, *, checkpoint: str = "",
                          predictors: Sequence[str] = Q12_DRAWN):
    """What each signal costs you when it decides which block evaluations to skip.

    Every policy is the same: skip the cells the signal calls most redundant, up to the
    budget, so only the ranking differs. The two grey references are what make the plot
    readable as evidence rather than as a ranking: ``oracle`` skips the genuinely smallest
    movements and is the best any signal could do, ``random`` is what no information looks
    like. A useful signal lives between them, and well below ``step``, which is the
    baseline a fixed cache schedule already achieves for free.
    """
    _need(curves, ["predictor", "budget", "mean_error"], "Q12 policy curves")
    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    references = {"oracle": ("best possible", (0, (1, 1))),
                  "random": ("no information", (0, (4, 2)))}
    drawn = [p for p in predictors if p in set(curves["predictor"])]
    for predictor in references:
        group = curves[curves["predictor"] == predictor].sort_values("budget")
        if group.empty:
            continue
        label, dashes = references[predictor]
        ax.plot(group["budget"], group["mean_error"], color=SHAM, lw=1.6, ls=dashes, zorder=1)
        ax.annotate(label, (group["budget"].iloc[-1], group["mean_error"].iloc[-1]),
                    textcoords="offset points", xytext=(6, 0), fontsize=7.5,
                    color=st.INK_SOFT, va="center")
    ends = []
    for index, predictor in enumerate(drawn):
        rows = curves[curves["predictor"] == predictor].sort_values("budget")
        if rows.empty:
            continue
        color = Q11_LINE_COLORS[index % len(Q11_LINE_COLORS)]
        ax.plot(rows["budget"], rows["mean_error"], color=color, lw=2.0, marker="o",
                markersize=4, markeredgecolor="white", markeredgewidth=0.6, label=predictor,
                zorder=3)
        ends.append((float(rows["budget"].iloc[-1]), float(rows["mean_error"].iloc[-1]),
                     predictor, color))
    if ends:
        span = max(e[1] for e in ends) - min(e[1] for e in ends)
        anchors = _spread_labels([e[1] for e in ends], (span or 1.0) * 0.10)
        for (budget, value, predictor, color), anchor in zip(ends, anchors):
            ax.annotate(predictor, (budget, value), xytext=(budget + 0.018, anchor),
                        fontsize=7.5, color=color, va="center",
                        arrowprops=dict(arrowstyle="-", color=color, lw=0.6, alpha=0.5,
                                        shrinkA=0, shrinkB=2))
    ax.set_xlabel("fraction of block evaluations skipped")
    ax.set_ylabel("mean movement at the skipped cells  (lower is better)")
    right = float(curves["budget"].max())
    ax.set_xlim(float(curves["budget"].min()) - 0.02, right + (right * 0.34))
    st.open_box(ax)
    st.light_grid(ax, "y")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    head = f"{checkpoint}: " if checkpoint else ""
    return _finish(fig, (
        f"{head}error incurred by each signal when it selects which block evaluations to "
        f"skip, against the fraction skipped. All policies are identical apart from the "
        f"ranking they use, and each signal is oriented before ranking so that its low end "
        f"means redundant. Grey curves bound the comparison: the best any signal could do, "
        f"and no information at all. A mechanism-derived signal has to sit inside that band "
        f"and below the step-index curve to be worth implementing, because a fixed cache "
        f"schedule already achieves the latter without any mechanism."), path)

FIGURE_BUILDERS = {
    "q1": ("fates", fig_q1_register_removal),
    "q2": ("dose_response", fig_q2_channel_dose_response),
    "q3": ("recovery", fig_q3_regeneration),
    "q4": ("separation", fig_q4_writer_selection),
    "q5": (None, fig_q5_sufficiency_ladder),
    "q6": ("lifetime", fig_q6_dissolution),
    "q11": ("pareto", fig_q11_quantization_pareto),
    "q12": ("policy_curves", fig_q12_policy_curves),
    # "maps" is registered at the end of this module, once its builder is defined.
}


QUESTION_TITLES_SHORT = {
    "q1": "Q1. Removing the natural register population",
    "q2": "Q2. Live dominant-channel suppression",
    "q3": "Q3. Regeneration after direction destruction",
    "q4": "Q4. What selects the register positions",
    "q5": "Q5. The residual-to-key sufficiency ladder",
    "q6": "Q6. What ends the register state",
    "maps": "Structure interaction",
}


def fig_unavailable(question: str, reasons: Mapping[str, str] | None = None, path=None,
                    checkpoint: str = ""):
    """A figure for a question this architecture could not be asked.

    Drawn rather than raised.  A question that cannot run is a fact about the
    model, and the notebook reporting it should keep going and say so on the page:
    a traceback mid-run reads as a broken pipeline, and the reason, which is the
    actual finding, ends up nowhere in the output.

    Laid out from the wrapped line count rather than a fixed step, because the
    reasons are full sentences and a fixed step overlaps them.
    """
    reasons = dict(reasons or {})
    wrapped = {name: _wrap(str(why), 88) for name, why in sorted(reasons.items())}

    # Inches, top down, so the entries cannot collide however long the reasons are.
    title_h, check_h, label_h, line_h, gap_h, margin = 0.52, 0.30, 0.26, 0.19, 0.12, 0.28
    body = sum(label_h + line_h * len(lines) + gap_h for lines in wrapped.values())
    height = margin + title_h + (check_h if checkpoint else 0.0) + (body or 0.6) + margin
    fig, ax = plt.subplots(figsize=(8.2, height))
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set(edgecolor=st.RULE, linewidth=0.8, linestyle=(0, (4, 2)))

    def place(y_in):
        return 1.0 - y_in / height

    title = QUESTION_TITLES_SHORT.get(str(question).lower(), str(question).upper())
    cursor = margin
    ax.text(0.5, place(cursor), f"{title}: not testable on this architecture",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=plt.rcParams["axes.titlesize"], fontweight="bold")
    cursor += title_h
    if checkpoint:
        ax.text(0.5, place(cursor), checkpoint, transform=ax.transAxes, ha="center", va="top",
                fontsize=plt.rcParams["axes.labelsize"], color=st.INK_SOFT)
        cursor += check_h
    if not wrapped:
        _no_data(ax, "the run produced no measurable rows")
    for name, lines in wrapped.items():
        ax.text(0.045, place(cursor), f"\u2022  {str(name).replace('_', ' ')}",
                transform=ax.transAxes, ha="left", va="top",
                fontsize=plt.rcParams["axes.labelsize"] - 0.5, fontweight="bold")
        cursor += label_h
        ax.text(0.075, place(cursor), "\n".join(lines), transform=ax.transAxes, ha="left",
                va="top", fontsize=plt.rcParams["legend.fontsize"] - 1, color=st.INK_SOFT,
                linespacing=1.35)
        cursor += line_h * len(lines) + gap_h

    caption = (
        f"{title}. This question produced no measurable rows on this checkpoint"
        + (f" ({checkpoint})" if checkpoint else "") + ", because the intervention points it "
        "depends on are not separable module boundaries in this architecture. The reasons are "
        "listed rather than the question silently omitted: an intervention that cannot be "
        "performed is a property of the model, not a null result, and the two must not be read "
        "as the same thing.")
    return _finish(fig, caption, path)


def _wrap(text: str, width: int) -> List[str]:
    """Wrap a reason to a handful of lines, so the caller can size the figure."""
    import textwrap

    return textwrap.wrap(text, width=width)[:4] or [""]


def render(result, path=None):
    """Draw the main figure for one question result, whatever its shape.

    Including the shape where there is nothing to draw: a question the
    architecture blocked returns :func:`fig_unavailable` rather than raising, so a
    notebook run survives it.  The builders themselves still refuse an empty
    table, since they must never invent data, which is why the check lives here.
    """
    question = getattr(result, "question", "").lower()
    if question not in FIGURE_BUILDERS:
        raise KeyError(f"no figure for {question!r}; known: {sorted(FIGURE_BUILDERS)}")
    companion, builder = FIGURE_BUILDERS[question]
    tables = getattr(result, "tables", {}) or {}
    meta = getattr(result, "meta", {}) or {}
    primary = {"q2": "dose_response", "q4": "patch_effects", "q6": "trajectory",
               "q11": "pareto", "q12": "policy_curves"}.get(question)
    candidate = tables.get(primary, result.tidy) if primary else result.tidy
    if candidate is None or not isinstance(candidate, pd.DataFrame) or candidate.empty:
        return fig_unavailable(question, meta.get("unsupported"), path=path,
                               checkpoint=str(meta.get("checkpoint", "")))
    if question == "q2":
        return builder(tables.get("dose_response", result.tidy), path=path)
    if question == "q4":
        return builder(tables.get("patch_effects", result.tidy), tables.get("separation"), path=path)
    if question == "q5":
        return builder(result.tidy, path=path)
    if question == "q6":
        return builder(tables.get("trajectory", result.tidy), tables.get("lifetime"), path=path)
    if question == "q11":
        # The Pareto panel is the headline, and it needs the checkpoint for its caption:
        # a bit budget is only interpretable once the model it was measured on is named.
        return builder(tables.get("pareto", result.tidy), path=path,
                       checkpoint=str(meta.get("checkpoint", "")))
    if question == "q12":
        return builder(tables.get("policy_curves", result.tidy), path=path,
                       checkpoint=str(meta.get("checkpoint", "")))
    if question == "maps":
        images = getattr(result, "images", {}) or {}
        first = next(iter(images.values()), None)
        return builder(result.tidy, tables.get("survival"), image=first, path=path)
    return builder(result.tidy, tables.get(companion), path=path)


# ======================================================= structure interaction
STRUCTURE_CMAPS = {
    # Magnitudes share one sequential map; attention is a probability and takes the
    # other, so the two quantities are never confused across panels.
    "high_norm_tokens": st.DENSITY_CMAP,
    "dominant_channel": st.DENSITY_CMAP,
    "attention_sinks": st.PROBABILITY_CMAP,
}
STRUCTURE_UNITS = {
    "high_norm_tokens": "Residual norm, over the layer median",
    "dominant_channel": "Activation of the dominant channel",
    "attention_sinks": "Share of incoming attention",
}


def fig_structure_interaction(tidy: pd.DataFrame, survival: Optional[pd.DataFrame] = None,
                              image=None, path=None, prompt_id: Optional[int] = None,
                              seed: Optional[int] = None):
    """Remove each sparse structure in turn; look at all three on the image grid.

    Rows are what was taken away, columns are what is being looked at.  Reading
    across a row answers "when this is removed, what else goes with it"; reading
    down a column answers "what does this structure survive".  The diagonal is the
    manipulation check: it shows the intervention did what it claims.
    """
    from .questions import REMOVALS, STRUCTURES

    _need(tidy, ["removal", "structure", "row", "col", "value"], "structure-interaction figure")
    frame = tidy
    if prompt_id is not None:
        frame = frame[frame["prompt_id"] == prompt_id]
    if seed is not None:
        frame = frame[frame["seed"] == seed]
    # This figure is the causal one: every panel here is a run of the model with
    # the structure taken away.  The bookkeeping panels answer a different
    # question and have their own figure; averaging the two onto one grid would
    # report neither.
    if "method" in frame.columns:
        frame = frame[frame["method"].isin(["clean", "causal"])]
    if frame.empty:
        raise ValueError("no rows for the requested prompt and seed")
    if survival is None:
        from .questions import _structure_survival

        survival = _structure_survival(tidy)
    shares = (survival.groupby(["removal", "structure"], observed=True)["survival"].mean()
              if survival is not None and not survival.empty else None)

    removals = [(k, label) for k, label in REMOVALS if k in set(frame["removal"])]
    structures = [(k, label) for k, label in STRUCTURES if k in set(frame["structure"])]
    n_rows, n_cols = len(removals), len(structures)
    has_image = image is not None
    width = 2.35 * n_cols + (2.6 if has_image else 0) + 1.5
    fig = plt.figure(figsize=(width, 2.35 * n_rows + 1.1))
    spec = fig.add_gridspec(n_rows, n_cols + (1 if has_image else 0),
                            width_ratios=([1.1] if has_image else []) + [1] * n_cols,
                            hspace=0.14, wspace=0.08)
    offset = 1 if has_image else 0

    if has_image:
        ax = fig.add_subplot(spec[:, 0])
        ax.imshow(image)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("Generated image", pad=8)

    # One colour scale per column, taken from the untouched run, so a panel going
    # dark means the structure went away rather than that the scale moved with it.
    limits = {}
    for key, _ in structures:
        clean = frame[(frame["structure"] == key) & (frame["removal"] == "clean")]["value"]
        pool = clean if len(clean) else frame[frame["structure"] == key]["value"]
        limits[key] = (float(pool.min()), float(np.quantile(pool, 0.999)) or float(pool.max()))

    images, column_axes = {}, {key: [] for key, _ in structures}
    for r, (removal, removal_label) in enumerate(removals):
        for c, (structure, structure_label) in enumerate(structures):
            ax = fig.add_subplot(spec[r, c + offset])
            column_axes[structure].append(ax)
            cell = frame[(frame["removal"] == removal) & (frame["structure"] == structure)]
            grid = _to_grid(cell)
            if grid is None:
                _no_data(ax)
            else:
                low, high = limits[structure]
                images[structure] = ax.imshow(grid, cmap=STRUCTURE_CMAPS.get(structure, st.DENSITY_CMAP),
                                              vmin=low, vmax=high, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(structure_label, pad=8)
            if c == 0:
                ax.set_ylabel(removal_label, fontsize=plt.rcParams["axes.labelsize"] - 1)
            # No badge on the untouched row: "100% left" of the reference is not
            # a measurement, and printing it invites reading it as one.
            if removal != "clean" and shares is not None and (removal, structure) in shares.index:
                value = float(shares.loc[(removal, structure)])
                _annotate_share(ax, value, is_check=(removal == structure))
            if removal == structure and removal != "clean":
                for side in ax.spines.values():
                    side.set(edgecolor=VERMILLION, linewidth=1.8)

    for structure, _ in structures:
        if structure not in images:
            continue
        bar = fig.colorbar(images[structure], ax=column_axes[structure], fraction=0.045,
                           pad=0.015, location="bottom", shrink=0.85, aspect=26)
        bar.set_label(STRUCTURE_UNITS.get(structure, ""),
                      fontsize=plt.rcParams["legend.fontsize"] - 1)
        bar.outline.set_linewidth(0.8)
        bar.ax.tick_params(width=0.8, direction="in",
                           labelsize=plt.rcParams["xtick.labelsize"] - 2)

    caption = (
        "Removing each sparse structure in turn, and looking at all three on the same image "
        "patches. Rows are what was taken away; columns are what is being looked at. Each panel "
        "lays one quantity over the latent patch grid at a register-zone layer: the residual norm "
        "over the layer median, the absolute activation of the dominant register channel, and the "
        "share of incoming image-to-image attention that the strongest head sends to each patch. "
        "The three removals act at different places by design, so that none removes another by "
        "construction: high-norm tokens are replaced by matched ordinary states in the residual "
        "stream; the dominant channel is scaled to zero in the writing block's contribution and "
        "held there through the register zone; and the sinks are removed in key space alone, by "
        "presenting attention with the mean ordinary key at those positions while their residual "
        "state is left untouched. Each panel is annotated with how much of that structure survives "
        "at the frozen register patches, as a share of the untouched run. Outlined panels are the "
        "diagonal, where a removal is measured against itself: they check that each intervention "
        "did what it claims, and are not findings. Colour scales are fixed per column from the "
        "untouched run, so a panel going dark means the structure went away and not that the scale "
        "moved with it.")
    return _finish(fig, caption, path, tight=False)


def _to_grid(cell: pd.DataFrame) -> Optional[np.ndarray]:
    """Lay one structure's per-token values back onto the patch grid.

    Several rows can land on one patch (more than one prompt, seed or method in
    the frame), and they are averaged rather than allowed to overwrite each
    other, so a caller that forgot to narrow the frame gets a mean map instead of
    whichever row pandas happened to write last.
    """
    if cell.empty:
        return None
    rows = int(cell["row"].max()) + 1
    cols = int(cell["col"].max()) + 1
    total = np.zeros((rows, cols))
    count = np.zeros((rows, cols))
    r = cell["row"].to_numpy(dtype=int)
    c = cell["col"].to_numpy(dtype=int)
    np.add.at(total, (r, c), cell["value"].to_numpy(dtype=float))
    np.add.at(count, (r, c), 1.0)
    return np.where(count > 0, total / np.maximum(count, 1.0), np.nan)


def _annotate_share(ax, value: float, *, is_check: bool) -> None:
    """Print how much of the structure is left, so a panel is readable as a number."""
    text = f"{value:.0%} left"
    ax.text(0.03, 0.97, text, transform=ax.transAxes, ha="left", va="top",
            fontsize=plt.rcParams["legend.fontsize"] - 1.5, color="white",
            bbox=dict(boxstyle="square,pad=0.22", facecolor=(VERMILLION if is_check else st.INK),
                      edgecolor="none", alpha=0.85))


FIGURE_BUILDERS["maps"] = ("survival", fig_structure_interaction)


def fig_ablation_strip(tidy: pd.DataFrame, image=None, path=None,
                       prompt_id: Optional[int] = None, seed: Optional[int] = None,
                       title: str = ""):
    """Every ablation on the image grid, one measurement per row.

    The companion to the interaction matrix, and the one to read first.  The
    matrix compresses each panel to a single share; this keeps the panels, so the
    reader can see *where* on the image a structure sat and where it went, which
    is the part a number cannot carry.

    Each row is one question of the form "take X away, then look at Y", and holds
    that question's whole evidence: the image that was generated, the structure
    that was taken away, the measured structure before, and the measured structure
    after.  Rows are grouped into blocks by what was taken away, so reading down a
    block answers "when this goes, what goes with it".

    Rows are filled by one of two methods, named on every panel because they
    answer different questions.  *Recomputed* rows are bookkeeping: the same
    quantity, recomputed from the untouched states with the structure masked out
    of the arithmetic, with the model never re-run.  *Re-run* rows generate again
    with the structure removed, so every later block is free to respond.  Which
    one a row can use is not a choice: bookkeeping is available only where the
    measured quantity is a function of the residual-stream states and the removal
    is a mask on those same states.
    """
    from .questions import REMOVALS, STRUCTURES, ablation_method

    _need(tidy, ["removal", "structure", "row", "col", "value"], "ablation strip")
    frame = tidy
    for column, value in (("prompt_id", prompt_id), ("seed", seed)):
        if value is not None and column in frame.columns:
            frame = frame[frame[column] == value]
    # Maps from different prompts are different images and cannot be averaged, so
    # one unit is chosen rather than pooled, and the caption says which.
    unit = ""
    for column in ("prompt_id", "seed"):
        if column in frame.columns and frame[column].nunique() > 1:
            first = sorted(frame[column].unique())[0]
            frame = frame[frame[column] == first]
    if "prompt_id" in frame.columns and len(frame):
        unit = f"Prompt {int(frame['prompt_id'].iloc[0])}"
        if "seed" in frame.columns:
            unit += f", seed {int(frame['seed'].iloc[0])}"
    if frame.empty:
        raise ValueError("no rows for the requested prompt and seed")

    present = set(zip(frame["removal"], frame["structure"]))
    removals = [(k, label) for k, label in REMOVALS
                if k != "clean" and any(r == k for r, _ in present)]
    structures = [(k, label) for k, label in STRUCTURES if k in set(frame["structure"])]
    if not removals or not structures:
        raise ValueError("the ablation strip needs at least one removal and one structure")

    def panel(removal: str, structure: str, method: Optional[str] = None):
        cell = frame[(frame["removal"] == removal) & (frame["structure"] == structure)]
        if method is not None and "method" in cell.columns:
            cell = cell[cell["method"] == method]
        return _to_grid(cell)

    def method_for(removal: str, structure: str) -> Optional[str]:
        """The declared method where the run has it, and whatever it does have if not."""
        cell = frame[(frame["removal"] == removal) & (frame["structure"] == structure)]
        if cell.empty:
            return None
        if "method" not in cell.columns:
            return "causal"
        available = set(cell["method"])
        declared = ablation_method(removal, structure)
        return declared if declared in available else next(iter(sorted(available)), None)

    clean = {key: panel("clean", key) for key, _ in structures}
    # One scale per measured quantity, taken from the untouched run at the 99.9th
    # percentile, so a panel going dark means the structure went away and not that
    # the scale followed it down.  Zero is the floor for all three: they are a
    # norm, an absolute activation and a probability.
    limits = {}
    for key, _ in structures:
        reference = clean.get(key)
        if reference is None:                       # no untouched run: fall back to what there is
            pool = [g for g in (panel(r, key, method_for(r, key)) for r, _ in removals)
                    if g is not None]
            reference = np.concatenate([g.ravel() for g in pool]) if pool else None
        high = float(np.nanpercentile(reference, 99.9)) if reference is not None else 1.0
        limits[key] = (0.0, high if high > 0 else 1.0)

    targets = _target_positions(frame)
    plan = [(removal, removal_label, structure, structure_label)
            for removal, removal_label in removals
            for structure, structure_label in structures]

    # The panels are square (they are square images), so the figure is sized
    # from the panel outwards rather than left to tight_layout, which measures the
    # cells before the aspect ratio shrinks them and banks the difference as one
    # margin at the top.  Every length below is inches.
    n_rows = len(plan)
    side, title_h, gap_w = 2.15, 0.30, 0.13
    left_in, right_in, top_in = 0.62, 0.08, 0.30 + (0.26 if title else 0.0)
    bar_h, bar_gap, bottom_in = 0.13, 0.34, 0.42
    fig_w = left_in + 4 * side + 3 * gap_w + right_in
    fig_h = (top_in + n_rows * side + (n_rows - 1) * title_h
             + bar_gap + bar_h + bottom_in)
    fig = plt.figure(figsize=(fig_w, fig_h))
    spec = fig.add_gridspec(n_rows + 1, 4, height_ratios=[1] * n_rows + [bar_h / side],
                            hspace=title_h / side, wspace=gap_w / side,
                            left=left_in / fig_w, right=1 - right_in / fig_w,
                            top=1 - top_in / fig_h, bottom=bottom_in / fig_h)
    mappables: Dict[str, Any] = {}
    small = plt.rcParams["axes.titlesize"] - 2.5
    blocks: Dict[str, List[Any]] = {}

    for r, (removal, removal_label, structure, structure_label) in enumerate(plan):
        method = method_for(removal, structure)
        is_check = removal == structure
        short_removal = _removal_noun(removal_label)

        ax = fig.add_subplot(spec[r, 0])
        blocks.setdefault(removal, []).append(ax)
        if image is not None:
            ax.imshow(image)
        else:
            _no_data(ax, "image not saved")
        ax.set_xticks([])
        ax.set_yticks([])
        if r == 0:
            ax.set_title("Generated image", fontsize=small, pad=4)

        # Titles are one line each and carry only what changes.  The first two
        # columns are constant down a block and are named once at its top, where
        # the block label already says what is being removed; the last two change
        # every row and are named every row.
        first_in_block = r % len(structures) == 0
        panels = (
            (spec[r, 1], clean.get(removal), removal,
             f"{short_removal.capitalize()}, untouched" if first_in_block else "", False),
            (spec[r, 2], clean.get(structure), structure,
             f"{structure_label}, before", False),
            (spec[r, 3], panel(removal, structure, method), structure,
             f"after \u2014 {_method_phrase(method)}", True),
        )
        for cell_spec, grid, key, panel_title, is_after in panels:
            ax = fig.add_subplot(cell_spec)
            if grid is None:
                _no_data(ax)
            else:
                low, high = limits.get(key, (0.0, 1.0))
                handle = ax.imshow(grid, cmap=STRUCTURE_CMAPS.get(key, st.DENSITY_CMAP),
                                   vmin=low, vmax=high, interpolation="nearest")
                mappables.setdefault(key, handle)
                _mark_targets(ax, targets, grid.shape)
                _annotate_peak(ax, grid)
            ax.set_xticks([])
            ax.set_yticks([])
            if panel_title:
                ax.set_title(panel_title, fontsize=small, pad=4)
            if is_after and grid is not None:
                before = clean.get(structure)
                share = _share_at_targets(before, grid, targets)
                if share is not None:
                    _annotate_share(ax, share, is_check=is_check)
                for side in ax.spines.values():
                    side.set(edgecolor=VERMILLION if is_check else st.INK,
                             linewidth=1.8 if is_check else 0.8)

    bars = spec[n_rows, 1:].subgridspec(1, len(structures), wspace=0.45)
    for index, (key, _) in enumerate(structures):
        if key not in mappables:
            continue
        bar = fig.colorbar(mappables[key], cax=fig.add_subplot(bars[0, index]),
                           orientation="horizontal")
        bar.set_label(STRUCTURE_UNITS.get(key, ""),
                      fontsize=plt.rcParams["legend.fontsize"] - 1.5)
        bar.outline.set_linewidth(0.8)
        bar.ax.tick_params(width=0.8, direction="in",
                           labelsize=plt.rcParams["xtick.labelsize"] - 2)
    if title:
        fig.suptitle(title, y=1 - 0.10 / fig_h, va="top",
                     fontsize=plt.rcParams["axes.titlesize"] + 1)

    caption = (
        "Every ablation on the latent patch grid, one measurement per row. " +
        (f"{unit}. " if unit else "") +
        "Each row asks one question -- take this away, then look at that -- and carries its whole "
        "evidence: the image that was generated, the structure that was taken away as it stands in "
        "the untouched run, the measured structure before, and the measured structure after. Rows "
        "are grouped into blocks by what was taken away, so reading down a block answers \"when "
        "this goes, what goes with it\". Circles mark the register patches, frozen from the "
        "untouched run and never reselected, and the badge on each after panel reports how much of "
        "the measured structure is left there as a share of the untouched value; a red badge and "
        "outline mark a row where a removal is measured against itself, which checks the "
        "manipulation rather than reporting a finding. Each panel is annotated with its own peak. "
        "Panels titled \\emph{recomputed} are bookkeeping: the same quantity recomputed from the "
        "untouched states with the structure masked out of the arithmetic, the model never re-run, "
        "so nothing downstream can react. Panels titled \\emph{model re-run} generate again with "
        "the structure removed, so every later block is free to respond. Which one a row uses is "
        "forced rather than chosen: bookkeeping exists only where the measured quantity is a "
        "function of the residual-stream states and the removal is a mask on those same states, "
        "which attention is not -- it is formed inside the block from queries and keys, and "
        "key-space sink suppression is not a mask on the residual stream at all. Colour scales are "
        "shared per measured quantity, fixed from the untouched run at the 99.9th percentile with "
        "zero as the floor, so a panel going dark means the structure went away rather than that "
        "the scale moved with it.")
    _block_labels(fig, blocks, dict(removals))
    return _finish(fig, caption, path, tight=False)


def _block_labels(fig, blocks: Mapping[str, Sequence[Any]], labels: Mapping[str, str]) -> None:
    """Name each block of rows once, down the left margin, spanning its own rows.

    Placed after layout and in figure coordinates: a block spans several axes, so
    there is no single axis whose label could carry it, and the span is only known
    once the gridspec has been resolved.
    """
    fig.canvas.draw()
    for removal, axes in blocks.items():
        boxes = [ax.get_position() for ax in axes]
        top = max(box.y1 for box in boxes)
        bottom = min(box.y0 for box in boxes)
        left = min(box.x0 for box in boxes)
        x = max(left - 0.040, 0.006)
        fig.text(x, (top + bottom) / 2, f"Removing the {_removal_noun(labels.get(removal, removal))}",
                 rotation=90, ha="center", va="center",
                 fontsize=plt.rcParams["axes.labelsize"], fontweight="bold")
        fig.add_artist(plt.Line2D([x + 0.011, x + 0.011], [bottom, top], transform=fig.transFigure,
                                  color=st.RULE, linewidth=0.9))


def fig_channel_trace(trace: pd.DataFrame, maintenance_layers: Sequence[int] = (),
                      writer_layer: Optional[int] = None, path=None):
    """The dominant channel at the frozen register patches, followed through depth.

    One map at one layer cannot separate two opposite readings of a surviving
    channel: that the suppression never landed, or that it landed and the network
    put the channel back.  This follows the same patches across every observed
    layer, with the layers the suppression acts on shaded, so the two are told
    apart by eye. A line that dips inside the shaded span and climbs out of it
    is regeneration; a line that never dips is an intervention that missed.

    The ordinary patches are drawn alongside as the floor: the register channel
    means nothing without the level it is supposed to tower over.
    """
    from .questions import REMOVALS

    _need(trace, ["removal", "layer", "at_registers"], "dominant-channel trace")
    removals = [(k, label) for k, label in REMOVALS if k in set(trace["removal"])]
    fig, ax = plt.subplots(figsize=(7.4, 4.3))

    span = [int(l) for l in maintenance_layers or ()]
    if span:
        ax.axvspan(min(span) - 0.5, max(span) + 0.5, color="0.92", zorder=0)
    if writer_layer is not None:
        ax.axvline(int(writer_layer), color=st.RULE, linestyle=(0, (1, 2)), linewidth=1.0,
                   zorder=1)
        ax.annotate("writer", xy=(int(writer_layer), 0.0), xycoords=("data", "axes fraction"),
                    xytext=(3, 4), textcoords="offset points", ha="left", va="bottom",
                    color=st.INK_SOFT, fontsize=plt.rcParams["legend.fontsize"] - 1)

    colors = _condition_colors([k for k, _ in removals])
    for key, label in removals:
        subset = trace[trace["removal"] == key]
        if subset.empty:
            continue
        _line_with_band(ax, subset, "layer", "at_registers", color=colors[key], label=label)
    ordinary = trace.groupby("layer", observed=True)["elsewhere"].mean()
    if ordinary.notna().any():
        ax.plot(ordinary.index, ordinary.to_numpy(dtype=float), color=st.INK_SOFT,
                linestyle=(0, (4, 2)), linewidth=1.2, zorder=2,
                label="Ordinary patches, for scale")

    ax.set(xlabel="Layer", ylabel="Activation of the dominant channel")
    _integer_layers(ax)
    # Headroom for the span label, which otherwise lands on the topmost line.
    ax.margins(y=0.16)
    st.light_grid(ax, "y")
    ax.legend(loc="best", frameon=False, fontsize=plt.rcParams["legend.fontsize"] - 1)
    if span:
        ax.annotate("channel held down here", xy=((min(span) + max(span)) / 2.0, 1.0),
                    xycoords=("data", "axes fraction"), xytext=(0, -5),
                    textcoords="offset points", ha="center", va="top", color=st.INK_SOFT,
                    fontsize=plt.rcParams["legend.fontsize"] - 1)
    caption = (
        "The dominant register channel at the frozen register patches, followed through depth "
        "under each removal, with prompt-clustered 95\\% bands. The shaded span marks the layers "
        "where the channel suppression acts, and the dotted rule the writing layer. This "
        "separates two readings that a single layer cannot: a line that falls inside the shaded "
        "span and climbs back out of it is the network rebuilding the channel, while a line that "
        "never falls is an intervention that did not reach what it aimed at. The dashed grey line "
        "is the mean over all other patches, which is the level the register channel is supposed "
        "to tower over.")
    return _finish(fig, caption, path)


def _removal_noun(label: str) -> str:
    """"High-norm tokens removed" -> "high-norm tokens", so a sentence can use it."""
    for suffix in (" removed", " suppressed"):
        if label.endswith(suffix):
            label = label[: -len(suffix)]
    return label[0].lower() + label[1:] if label else label


def _method_phrase(method: Optional[str]) -> str:
    """Name the method in words a reader can act on, never the internal term."""
    return {"subtract": "recomputed", "causal": "model re-run"}.get(method or "", "removed")


def _target_positions(frame: pd.DataFrame) -> List[Tuple[int, int]]:
    """Grid positions of the frozen register patches, from the untouched rows."""
    if "is_frozen_target" not in frame.columns:
        return []
    marked = frame[frame["is_frozen_target"].astype(bool)]
    if marked.empty:
        return []
    return sorted({(int(r), int(c)) for r, c in zip(marked["row"], marked["col"])})


def _mark_targets(ax, targets: Sequence[Tuple[int, int]],
                  shape: Optional[Tuple[int, int]] = None) -> None:
    """Circle the frozen register patches, so the same patches are followed across panels.

    The radius grows with the grid: a circle wide enough to be seen on a 64-patch
    side would swallow half of an 8-patch one.
    """
    side = max(shape) if shape else 32
    radius = max(0.9, 0.022 * float(side))
    for row, col in targets:
        ax.add_patch(plt.Circle((col, row), radius, fill=False, edgecolor="white",
                                linewidth=1.1, alpha=0.9, zorder=3))


def _annotate_peak(ax, grid: np.ndarray) -> None:
    """The panel's own peak, so a dark panel is readable as a number too."""
    finite = grid[np.isfinite(grid)]
    if not finite.size:
        return
    peak = float(finite.max())
    text = f"peak {peak:.3g}" if peak else "all zero"
    ax.text(0.97, 0.03, text, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=plt.rcParams["legend.fontsize"] - 2, color="white",
            bbox=dict(boxstyle="square,pad=0.18", facecolor=st.INK, edgecolor="none", alpha=0.7))


def _share_at_targets(before: Optional[np.ndarray], after: Optional[np.ndarray],
                      targets: Sequence[Tuple[int, int]]) -> Optional[float]:
    """How much of the structure is left on the frozen patches, as a share of before."""
    if before is None or after is None or not targets:
        return None
    rows = [r for r, _ in targets if r < before.shape[0] and r < after.shape[0]]
    cols = [c for (r, c) in targets if r < before.shape[0] and r < after.shape[0]]
    if not rows:
        return None
    reference = float(np.nanmean(before[rows, cols]))
    if not np.isfinite(reference) or abs(reference) < 1e-12:
        return None
    treated = float(np.nanmean(after[rows, cols]))
    return treated / reference


# ======================================  direction-vs-magnitude control surface
# What each grid endpoint is called, and whether the clean value sits at the low or the
# high end. The sense matters for the diverging maps: a quantity whose clean value is a
# midpoint is drawn diverging around it, and one that only grows is drawn sequential.
Q13_ENDPOINTS: Dict[str, Tuple[str, str]] = {
    "selected_alpha_mean": ("mean $x^\\top v^*$ of treated tokens", "diverging"),
    "n_highnorm": ("number of high-norm tokens", "sequential"),
    "selected_sink_strength_mean": ("treated-token sink strength\n($\\times$ uniform share)",
                                    "sequential"),
    "sink_retention": ("affected heads keeping their clean sink", "sequential"),
    "highnorm_retention": ("clean high-norm tokens still high-norm", "sequential"),
    "lpips": ("LPIPS from the clean image", "sequential"),
    "rmse": ("pixel RMSE from the clean image", "sequential"),
    "spectrum_high_over_uniform": ("high-frequency difference energy\n(1.0 = unstructured)",
                                   "sequential"),
    "concentration_ratio": ("difference energy on clean structure\n(1.0 = spread evenly)",
                            "sequential"),
    "clip_prompt_similarity_drop": ("drop in CLIP prompt similarity", "sequential"),
}


def _q13_surface(frame: pd.DataFrame, column: str) -> Optional[pd.DataFrame]:
    """Mean of ``column`` over every unit, as a gamma-by-beta table."""
    if frame is None or frame.empty or column not in frame.columns:
        return None
    rows = frame[frame[column].notna()]
    if rows.empty:
        return None
    grid = rows.groupby(["gamma", "beta"], observed=True)[column].mean().unstack("beta")
    return grid.sort_index().sort_index(axis=1) if not grid.empty else None


def fig_q13_control_surface(population: pd.DataFrame, images: Optional[pd.DataFrame] = None,
                            path=None, *, checkpoint: str = "",
                            endpoints: Sequence[str] = ("selected_alpha_mean", "n_highnorm",
                                                        "selected_sink_strength_mean",
                                                        "lpips", "spectrum_high_over_uniform",
                                                        "concentration_ratio")):
    """The 2D causal control surface: alignment across, magnitude up.

    This is the figure the experiment exists to produce. ``beta`` rescales the component
    along $v^*$ and the token is then renormalised, so it varies alignment **at fixed
    norm**; ``gamma`` scales the whole vector afterwards, so it varies magnitude **at
    fixed direction**. An endpoint that varies along one axis and not the other is
    controlled by that variable alone, and the shape of each panel is the answer.

    ``(beta, gamma) = (1, 1)`` runs the full edit rather than being skipped, so that cell
    is the numerical floor and not a definitional zero.
    """
    tables: List[Tuple[str, Optional[pd.DataFrame]]] = []
    for name in endpoints:
        source = population if (population is not None and name in getattr(
            population, "columns", [])) else images
        tables.append((name, _q13_surface(source, name)))
    if not any(table is not None for _, table in tables):
        raise ValueError("cannot draw the Q13 control surface: no endpoint has any cells")

    columns = min(3, max(1, len(tables)))
    rows = int(math.ceil(len(tables) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(4.3 * columns, 3.6 * rows),
                             squeeze=False)
    # Each panel carries its own colourbar, which sits between it and the next panel's
    # y-axis label, so the gap has to be set explicitly rather than left to tight_layout.
    fig.subplots_adjust(wspace=0.62, hspace=0.45, left=0.07, right=0.94,
                        top=0.93, bottom=0.10)
    flat = [ax for row in axes for ax in row]
    for ax, (name, table) in zip(flat, tables):
        label, sense = Q13_ENDPOINTS.get(name, (name.replace("_", " "), "sequential"))
        if table is None:
            _no_data(ax, f"{label}\nnot measurable in this run")
            continue
        values = table.to_numpy(dtype=float)
        if sense == "diverging":
            # Centred on the clean cell, so the colour says "more aligned than clean" or
            # "less", rather than merely "large".
            centre = float(table.loc[1.0, 1.0]) if (1.0 in table.index
                                                    and 1.0 in table.columns) else 0.0
            limit = float(np.nanmax(np.abs(values - centre))) or 1.0
            mesh = ax.imshow(values, origin="lower", aspect="auto", cmap=st.diverging_cmap(),
                             vmin=centre - limit, vmax=centre + limit)
        else:
            mesh = ax.imshow(values, origin="lower", aspect="auto", cmap=st.DENSITY_CMAP)
        ax.set_xticks(range(len(table.columns)))
        ax.set_xticklabels([f"{float(c):g}" for c in table.columns])
        ax.set_yticks(range(len(table.index)))
        ax.set_yticklabels([f"{float(r):g}" for r in table.index])
        # Short axis labels on purpose: six panels each carrying a colourbar leaves no
        # room for a sentence, and the caption defines both axes in full.
        ax.set_xlabel(r"$\beta$  (alignment)")
        ax.set_ylabel(r"$\gamma$  (norm)")
        # Ring the clean cell: every other cell is read relative to it.
        if 1.0 in table.index and 1.0 in table.columns:
            x = list(table.columns).index(1.0)
            y = list(table.index).index(1.0)
            ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, fill=False,
                                       edgecolor="white", lw=1.6))
            ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, fill=False,
                                       edgecolor=st.INK, lw=0.7, ls=(0, (3, 2))))
        st.colorbar(fig, mesh, ax, label)
        st.open_box(ax)
    for ax in flat[len(tables):]:
        ax.set_visible(False)
    _panel_letters(fig, flat[:len(tables)])
    head = f"{checkpoint}: " if checkpoint else ""
    return _finish(fig, (
        f"{head}the 2D causal control surface. Horizontally, $\\beta$ rescales each treated "
        f"token's component along $v^*$ and the token is renormalised to its original "
        f"length, so alignment varies at fixed norm. Vertically, $\\gamma$ scales the "
        f"whole vector afterwards, so magnitude varies at fixed direction. The outlined "
        f"cell is $(\\beta, \\gamma) = (1, 1)$, which runs the full edit and therefore "
        f"measures the numerical floor rather than a definitional zero. A panel that varies "
        f"left-to-right but not bottom-to-top is controlled by alignment; one that varies "
        f"bottom-to-top but not left-to-right is controlled by magnitude."), path, tight=False)


def fig_q13_separation(token: pd.DataFrame, path=None, *, checkpoint: str = ""):
    """Are norm, projection and sinkhood the same variable in the clean model?

    Both panels are properties of the *clean* run, before any intervention: if the
    three quantities were interchangeable the point cloud would be a line and the sink
    markers would separate at a single cut. Panel (b) is where the threshold hypothesis lives: a thresholded
    relationship shows as sinks appearing only past some projection, rather than
    increasing smoothly with it.
    """
    _need(token, ["alpha", "norm", "is_sink", "sink_strength"], "Q13 separation scatter")
    clean = token[token["intervention"].astype(str).str.startswith("clean")]
    if clean.empty:
        clean = token
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.3))
    sink = clean["is_sink"].fillna(False).astype(bool)
    selected = clean.get("is_selected", pd.Series(False, index=clean.index)).fillna(False).astype(bool)
    for ax, ycolumn, ylabel in (
            (axes[0], "norm", r"residual norm $||x||_2$"),
            (axes[1], "sink_strength", "sink strength  ($\\times$ uniform share)")):
        for mask, color, label, size in ((~sink, SHAM, "not a sink", 9),
                                         (sink, VERMILLION, "attention sink", 22)):
            rows = clean[mask]
            if rows.empty:
                continue
            ax.scatter(rows["alpha"], rows[ycolumn], s=size, color=color, alpha=0.55,
                       linewidths=0.0, label=label, zorder=2 if label == "not a sink" else 3)
        chosen = clean[selected]
        if not chosen.empty:
            ax.scatter(chosen["alpha"], chosen[ycolumn], s=64, facecolor="none",
                       edgecolor=BLUE, linewidths=1.2, label="treated population", zorder=4)
        ax.set_xlabel(r"clean projection $x^\top v^*$")
        ax.set_ylabel(ylabel)
        st.open_box(ax)
        st.light_grid(ax, "y")
    axes[0].legend(frameon=False, fontsize=7.5, loc="best")
    _panel_letters(fig, list(axes))
    head = f"{checkpoint}: " if checkpoint else ""
    return _finish(fig, (
        f"{head}clean-run token population. (a) residual norm against projection onto "
        f"$v^*$; (b) sink strength against the same projection, where sink strength is the "
        f"incoming image-to-image attention mass as a multiple of the uniform share "
        f"$1/N$ -- the project's existing criterion, applied per token. Markers show which "
        f"tokens meet that criterion and which were selected for treatment. If norm, "
        f"projection and sinkhood were one variable, (a) would be a line and (b) would "
        f"separate cleanly at a single cut."), path)


def fig_q13_rescue(population: pd.DataFrame, images: Optional[pd.DataFrame] = None,
                   path=None, *, checkpoint: str = "",
                   effects: Optional[pd.DataFrame] = None):
    """Does restoring the $v^*$-aligned state rescue what ablating the channel destroyed?

    The four conditions are the mediation test. ``channel_ablate`` is the damage;
    ``vstar_rescue`` restores the projection the ablation removed; the orthogonal rescue
    injects the same per-token L2 in the orthogonal complement, so size cannot
    explain a difference between them; and ``channel_restore_only`` puts back the single
    coordinate, which separates "the mechanism is a direction" from "the mechanism is a
    channel".
    """
    _need(population, ["condition", "arm"], "Q13 rescue comparison")
    rescue = population[population["arm"] == "rescue"]
    if "is_downstream" in rescue.columns:
        downstream = rescue[rescue["is_downstream"].fillna(True).astype(bool)]
        rescue = downstream if not downstream.empty else rescue
    if rescue.empty:
        raise ValueError("cannot draw the Q13 rescue comparison: no rescue rows")
    panels = [("selected_alpha_recovery", "$v^*$ projection, as a fraction of clean"),
              ("highnorm_retention", "clean high-norm tokens still high-norm"),
              ("sink_retention", "affected heads keeping their clean sink")]
    if images is not None and not images.empty:
        panels.append(("__image__", "image distance from clean"))
    fig, axes = plt.subplots(1, len(panels), figsize=(3.6 * len(panels), 4.0), squeeze=False)
    flat = [ax for row in axes for ax in row]
    order = [c for c in ("clean", "channel_ablate", "channel_ablate_vstar_rescue",
                         "channel_ablate_orthogonal_rescue", "channel_restore_only")
             if c in set(rescue["condition"])]
    kind = "standard error across units"
    short = {"clean": "clean", "channel_ablate": "channel\nablated",
             "channel_ablate_vstar_rescue": "+ $v^*$\nrescue",
             "channel_ablate_orthogonal_rescue": "+ orthogonal\n(L2-matched)",
             "channel_restore_only": "+ channel\nonly"}
    for ax, (column, label) in zip(flat, panels):
        if column == "__image__":
            metric = next((m for m in ("lpips", "rmse") if m in images.columns
                           and images[m].notna().any()), None)
            rows = images[images["arm"] == "rescue"] if "arm" in images.columns else images
            if metric is None or rows.empty:
                _no_data(ax, "no image distances\nin this run")
                continue
            source, value, label = rows, metric, Q13_ENDPOINTS.get(metric, (label, ""))[0]
        elif column in rescue.columns and rescue[column].notna().any():
            source, value = rescue, column
        else:
            _no_data(ax, f"{label}\nnot measurable in this run")
            continue
        means = source.groupby("condition", observed=True)[value].mean()
        # Prefer the project's prompt-clustered bootstrap interval over a standard error
        # across units: it is paired within unit, so prompt-to-prompt variation cancels
        # rather than inflating the bar, and it clusters by prompt so two seeds of one
        # prompt are not two independent observations. Falls back to the SEM, labelled as
        # such, when the run has too few clusters for the bootstrap to report an interval.
        interval = None
        if effects is not None and not effects.empty:
            rows = effects[(effects["metric"] == value) & effects["has_interval"]]
            if not rows.empty:
                anchor = float(means.get(str(rows["reference"].iloc[0]), 0.0))
                interval = {str(r["condition"]): (anchor + float(r["ci_low"]),
                                                  anchor + float(r["ci_high"]))
                            for _, r in rows.iterrows()}
                kind = (f"prompt-clustered bootstrap, {int(rows['n_clusters'].max())} "
                        "clusters, paired against the ablation")
        spread = source.groupby("condition", observed=True)[value].sem()
        present = [c for c in order if c in means.index]
        colors = [Q11_ROLE_COLOR.get("reference" if c == "clean" else
                                     ("control" if "orthogonal" in c else "condition"), BLUE)
                  for c in present]
        positions = range(len(present))
        if interval is not None:
            low = [means[c] - interval[c][0] if c in interval else 0.0 for c in present]
            high = [interval[c][1] - means[c] if c in interval else 0.0 for c in present]
            errors = np.abs(np.vstack([low, high]))
        else:
            errors = np.array([[0.0 if np.isnan(spread.get(c, np.nan)) else spread[c]
                                for c in present]] * 2)
        ax.errorbar(list(positions), [means[c] for c in present], yerr=errors,
                    fmt="none", ecolor=st.RULE, elinewidth=1.0, capsize=3, zorder=2)
        ax.scatter(list(positions), [means[c] for c in present], s=70, c=colors,
                   edgecolor="white", linewidth=0.8, zorder=3)
        if value in ("selected_alpha_recovery", "highnorm_retention", "sink_retention"):
            ax.axhline(1.0, color=SHAM, lw=1.0, ls=(0, (2, 2)), zorder=1)
            ax.annotate("clean level", (len(present) - 1, 1.0), textcoords="offset points",
                        xytext=(0, 5), ha="right", fontsize=7, color=st.INK_SOFT)
        ax.set_xticks(list(positions))
        ax.set_xticklabels([short.get(c, c) for c in present], fontsize=7.5)
        ax.set_ylabel(label)
        ax.set_xlim(-0.6, len(present) - 0.4)
        st.open_box(ax)
        st.light_grid(ax, "y")
    _panel_letters(fig, flat)
    head = f"{checkpoint}: " if checkpoint else ""
    return _finish(fig, (
        f"{head}the rescue arm, averaged over prompt-seed units at layers the edit can "
        f"reach. Error bars: {kind}. The rescue target is the projection "
        f"measured on the paired clean trajectory, never on the ablated run. The "
        f"orthogonal condition injects the same per-token L2 as the $v^*$ rescue in the "
        f"orthogonal complement, so a difference between them is about direction rather "
        f"than about perturbation size. Restoring only the dominant coordinate separates a "
        f"channel mechanism from a distributed one."), path)


def fig_q13_contact_sheet(images: pd.DataFrame, root, path=None, *, checkpoint: str = "",
                          prompt_id: int = 0, seed: int = 0, layer_scope: str = "write",
                          show_difference: bool = True):
    """Representative conditions side by side, with amplified difference maps beneath.

    The amplification factor is written into each difference panel's title, because an
    amplified difference shown without its factor is not interpretable.
    """
    _need(images, ["intervention", "image_path"], "Q13 contact sheet")
    from PIL import Image

    root = Path(root)
    wanted = [("clean", "clean"), ("beta0_gamma1", r"$\beta=0$"), ("beta2_gamma1", r"$\beta=2$"),
              ("beta1_gamma0.5", r"$\gamma=0.5$"), ("beta1_gamma1.5", r"$\gamma=1.5$"),
              ("channel_ablate", "channel ablated"),
              ("channel_ablate_vstar_rescue", "+ $v^*$ rescue"),
              ("channel_ablate_orthogonal_rescue", "+ orthogonal"),
              ("channel_restore_only", "+ channel only")]
    rows = images[(images["prompt_id"] == prompt_id) & (images["seed"] == seed)]
    scoped = rows[rows["layer_scope"] == layer_scope] if "layer_scope" in rows.columns else rows
    lookup = {str(r["intervention"]): r for _, r in (scoped if not scoped.empty else rows).iterrows()}
    def load(value):
        if not isinstance(value, str) or not value:
            return None
        candidate = root / value
        try:
            return Image.open(candidate).convert("RGB") if candidate.exists() else None
        except Exception:
            return None

    panels = [(key, label, lookup[key]) for key, label in wanted if key in lookup]
    if not panels:
        raise ValueError("cannot draw the Q13 contact sheet: none of the named conditions "
                         "has a saved image for this unit")
    if not any(load(row.get("image_path")) is not None for _, _, row in panels):
        # A sheet of "no image" panels looks like a result and is not one. Refusing lets
        # the notebook say the images are unavailable, which is the actual fact.
        raise ValueError("cannot draw the Q13 contact sheet: no panel has a readable image "
                         "on disk -- was the run made with save_images=False, or on a "
                         "checkpoint with no decoder?")

    band = 2 if show_difference else 1
    fig, axes = plt.subplots(band, len(panels), figsize=(1.75 * len(panels), 1.95 * band),
                             squeeze=False)
    for column, (key, label, row) in enumerate(panels):
        image = load(row.get("image_path"))
        ax = axes[0][column]
        if image is None:
            _no_data(ax, "no image")
        else:
            ax.imshow(np.asarray(image))
        ax.set_title(label, fontsize=7.5)
        ax.set_xticks([]); ax.set_yticks([])
        for side in ax.spines.values():
            side.set_visible(False)
        if band == 1:
            continue
        ax_difference = axes[1][column]
        difference = load(row.get("difference_map"))
        if difference is None or key == "clean":
            _no_data(ax_difference, "--" if key == "clean" else "no map")
        else:
            ax_difference.imshow(np.asarray(difference), cmap="gray", vmin=0, vmax=255)
            factor = row.get("difference_amplification")
            ax_difference.set_title(f"difference $\\times${float(factor):g}" if factor
                                    else "difference", fontsize=7.0)
        ax_difference.set_xticks([]); ax_difference.set_yticks([])
        for side in ax_difference.spines.values():
            side.set_visible(False)
    head = f"{checkpoint}: " if checkpoint else ""
    factors = sorted({float(r.get("difference_amplification")) for _, _, r in panels
                      if r.get("difference_amplification")})
    note = (f" Difference maps are amplified {factors[0]:g}$\\times$ about mid grey"
            if len(factors) == 1 else "")
    return _finish(fig, (
        f"{head}representative conditions for prompt {prompt_id}, seed {seed}, "
        f"{layer_scope} scope. Every panel uses the same prompt, seed, sampler, step count "
        f"and guidance, and differs only in the installed edit.{note}, so mid grey is no "
        f"change and the sign of the change is visible; the factor is stated because an "
        f"amplified difference read without it is not interpretable."), path)


# ==========================================  intervention-depth comparison
# One hue per depth, in the order the register lives through them, so a reader can see
# "earlier is more controllable" as a spread between lines rather than as three tables.
Q13_DEPTH_ORDER = ("birth", "boundary", "established")
Q13_DEPTH_COLOR = {"birth": st.OKABE_ITO["green"], "boundary": BLUE,
                   "established": st.OKABE_ITO["purple"]}
Q13_DEPTH_LABEL = {"birth": "birth (writer range)", "boundary": "boundary (register start)",
                   "established": "established (register + 4)"}


def fig_q13_depth_response(depth: pd.DataFrame, path=None, *, checkpoint: str = "",
                           gamma: float = 1.0):
    r"""The same edit at three depths, swept over $\beta$.

    Three questions, one figure, and each panel is the direct measurement rather than a
    proxy.

    **(a) Was there room to move?** The dashed rules are the *clean* alignment at each
    depth and the solid lines are what the edit achieved. Since
    $\cos \to 1$ asymptotically in $\beta$, a depth whose clean state already sits near
    $|\cos| = 1$ has almost nothing a positive $\beta$ can add, so a flat response there
    is a **ceiling**, not an ineffective edit, and the two are only distinguishable with
    the clean value drawn beside it.

    **(b) Is the achieved state one the model could have produced?** The share of the
    treated tokens' energy lying inside the clean population's own subspace at that layer,
    as a ratio to a typical clean token's share. Near 1 the state is on the manifold; well
    below 1 it is off it, whatever its norm. This is what separates a meaningful
    opposite-$v^*$ state at negative $\beta$ from corruption of the same magnitude.

    **(c) What reached the image?** If the earlier depth gives a larger and more
    monotone image response at matched $\beta$, the writer phase affords more control than
    the boundary does. On a checkpoint with no decoder there is no image distance to draw,
    and the panel falls back to the treated tokens' own projection, which the caption
    says, because a reader would otherwise take the third panel for an image claim.
    """
    _need(depth, ["layer_scope", "beta"], "Q13 depth response")
    rows = depth[depth["layer_scope"].isin(Q13_DEPTH_ORDER)]
    if rows.empty:
        raise ValueError("cannot draw the Q13 depth response: no depth scopes present")
    image_metric = next((m for m in ("lpips", "rmse") if m in rows.columns
                         and rows[m].notna().any()), None)
    panels = [("realised_cosine", r"signed alignment $\cos(x, v^*)$ after the edit", False),
              ("onmanifold_ratio_to_clean",
               "on-manifold energy\n(1.0 = as deep as a clean token)", True),
              (image_metric or "selected_alpha_mean",
               Q13_ENDPOINTS.get(image_metric, ("mean $x^\\top v^*$ of treated tokens",))[0]
               if image_metric else "mean $x^\\top v^*$ of treated tokens", False)]

    fig, axes = plt.subplots(1, len(panels), figsize=(4.7 * len(panels), 4.3), squeeze=False)
    flat = [ax for row in axes for ax in row]
    clean_marks: List[Tuple[float, str]] = []
    for ax, (column, label, unit_line) in zip(flat, panels):
        if column not in rows.columns or not rows[column].notna().any():
            _no_data(ax, f"{label}\nnot measurable in this run")
            continue
        for scope in Q13_DEPTH_ORDER:
            series = rows[rows["layer_scope"] == scope].sort_values("beta")
            if series.empty or not series[column].notna().any():
                continue
            layer = int(series["layer"].iloc[0]) if "layer" in series.columns else -1
            ax.plot(series["beta"], series[column], color=Q13_DEPTH_COLOR[scope], lw=2.0,
                    marker="o", markersize=4.5, markeredgecolor="white",
                    markeredgewidth=0.6, zorder=3,
                    label=f"{Q13_DEPTH_LABEL[scope]}" + (f"  ·  layer {layer}"
                                                         if layer >= 0 else ""))
            # The clean alignment each depth started from: without it, a flat line is
            # ambiguous between "no room left" and "no effect".
            if column == "realised_cosine" and "clean_cosine" in series.columns:
                # The SIGNED clean alignment, so it is commensurable with the solid line:
                # the two coincide at beta = 1, which is what makes the panel readable.
                clean = float(series["clean_cosine"].mean())
                ax.axhline(clean, color=Q13_DEPTH_COLOR[scope], lw=0.9, ls=(0, (2, 2)),
                           alpha=0.75, zorder=1)
                clean_marks.append((clean, Q13_DEPTH_COLOR[scope]))
        if unit_line:
            ax.axhline(1.0, color=SHAM, lw=1.0, ls=(0, (4, 2)), zorder=1)
        ax.axvline(1.0, color=st.GRID, lw=0.8, zorder=0)
        ax.annotate("unmodified", (1.0, ax.get_ylim()[0]), textcoords="offset points",
                    xytext=(3, 3), fontsize=7, color=st.INK_SOFT)
        if (rows["beta"] < 0).any():
            ax.axvline(0.0, color=st.GRID, lw=0.8, ls=(0, (1, 2)), zorder=0)
        ax.set_xlabel(r"$\beta$  (alignment multiplier, norm held fixed)")
        ax.set_ylabel(label, fontsize=8.5)
        st.open_box(ax)
        st.light_grid(ax, "y")
        if column == "realised_cosine" and clean_marks:
            right = float(rows["beta"].max())
            left = float(rows["beta"].min()) - 0.12
            # Gap measured against the AXIS range, not the marks' own range. The clean
            # levels can sit within a thousandth of each other while the axis spans two,
            # and a gap scaled to their spread would then be smaller than the text.
            low, high = ax.get_ylim()
            gap = (high - low) * 0.062
            levels = [value for value, _ in clean_marks]
            if len(levels) > 1 and max(levels) - min(levels) < gap:
                # The depths' clean alignments are indistinguishable at this scale, so
                # their dashed lines lie on top of one another. Three leader lines to one
                # location would say nothing, so one label states the range instead --
                # and the fact that the three depths start from the same place is itself
                # the thing a reader needs to know before reading panel (a).
                ax.set_xlim(left, right + 0.16)
                span = (f"{min(levels):.3f}" if max(levels) - min(levels) < 5e-4
                        else f"{min(levels):.3f}-{max(levels):.3f}")
                level = float(np.mean(levels))
                # No leader line is needed, so the label goes inside the panel against
                # its right edge, and hangs away from the line it names, downward from
                # a level in the upper half and upward from one in the lower, so it stays
                # on the panel wherever the clean alignment happens to sit.
                upper = level > (low + high) / 2.0
                ax.annotate(f"clean {span}\n(all {len(levels)} depths)",
                            (0.988, level), xycoords=("axes fraction", "data"),
                            textcoords="offset points", xytext=(0, -4 if upper else 4),
                            fontsize=7.0, color=st.INK_SOFT, ha="right",
                            va="top" if upper else "bottom")
            else:
                ax.set_xlim(left, right + 0.62)
                # Spread the anchors, then re-centre the block on the marks and give the
                # axis room for it. Pushing labels upward from the lowest one, which is
                # what _spread_labels does, can place the top label outside the axes --
                # where matplotlib draws it over the panel letter or clips it away.
                anchors = _spread_labels(levels, gap)
                shift = float(np.mean(levels)) - float(np.mean(anchors))
                anchors = [anchor + shift for anchor in anchors]
                ax.set_ylim(min(low, min(anchors) - gap * 0.6),
                            max(high, max(anchors) + gap * 0.6))
                for (value, color), anchor in zip(clean_marks, anchors):
                    ax.annotate(f"clean {value:.3f}", (right, value),
                                xytext=(right + 0.14, anchor), fontsize=7.0, color=color,
                                va="center", arrowprops=dict(arrowstyle="-", color=color,
                                                             lw=0.6, alpha=0.5, shrinkA=0,
                                                             shrinkB=2))
    # Below the axes, so it cannot land on the first panel's data or its letter.
    flat[0].legend(frameon=False, fontsize=7.5, loc="upper center",
                   bbox_to_anchor=(0.5, -0.155), ncol=1, handletextpad=0.5)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.93, bottom=0.30, wspace=0.28)
    _panel_letters(fig, flat)
    head = f"{checkpoint}: " if checkpoint else ""
    negative = "including negative values, which invert the sign of the projection, " \
        if (rows["beta"] < 0).any() else ""
    third = ("What reached the image." if image_metric else
             "No image distances were available on this run -- a checkpoint with no "
             "decoder measures the internal state only -- so the third panel is the "
             "treated tokens' own projection, not an image response.")
    return _finish(fig, (
        f"{head}the same edit installed at three depths in the register's life and swept "
        f"over $\\beta$ {negative}at $\\gamma$ = {gamma:g}. Each depth is a single layer, "
        f"named from the frozen layer ranges rather than chosen by hand. "
        f"(a) The solid line is the alignment the edit achieved; the dashed line of the "
        f"same colour is the clean alignment it started from. Because $\\cos$ approaches 1 "
        f"asymptotically in $\\beta$, a depth already near 1 has no room left, and a flat "
        f"response there is a ceiling rather than an ineffective edit -- a distinction "
        f"that cannot be made without the clean value drawn beside it. "
        f"(b) The share of the treated tokens' energy inside the clean population's own "
        f"subspace at that layer, as a ratio to a typical clean token's share; the "
        f"operators fix norm and alignment by construction, so this is the only quantity "
        f"here that can say whether a state is one the model could have produced. "
        f"(c) {third} A larger, more monotone response at the earlier "
        f"depth is what 'the writer phase affords more control' would look like."),
        path, tight=False)


# =========================================  writer-phase stage probe (edit site)
# The order the forward pass reaches each stage depends on where the edit is installed,
# so the two sites do not share an x axis and are never drawn on one.
CS_STAGE_ORDER = ("before_mlp", "after_mlp_write", "after_edit", "after_next_block")
Q13_SITE_COLOR = {"block_input": st.OKABE_ITO["orange"], "block_output": BLUE}
Q13_SITE_TITLE = {"block_input": "edit at the block INPUT",
                  "block_output": "edit AFTER the MLP's residual addition"}
Q13_SITE_LEGEND = {"block_input": "block input (before attention and the MLP)",
                   "block_output": "block output (after the MLP's residual addition)"}
_STAGE_TICK = {"before_mlp": "post-attn\nresidual", "after_mlp_write": "block out\n(as produced)",
               "after_edit": "the hook\n(the edit)", "after_next_block": "after the\nnext block"}


def fig_q13_stage_probe(stages: pd.DataFrame, path=None, *, checkpoint: str = "",
                        beta: float = 0.0):
    r"""Does the block rewrite the edit, or does the edit hold?

    **(a) and (b): the trajectory through one edited block, one panel per edit site.**
    The treated tokens' $x^\top v^*$ at each stage, in the order the forward pass reaches
    it *under that site*: a block-input hook fires before the block runs and a
    block-output hook after, so the two orders differ and a shared axis would read the
    block's response to the edit as its cause. A block-input edit that lands at zero and
    leaves the block near the clean value was **rewritten by that block's feed-forward**,
    which is a fact about the layer and not about the operator.

    **(c) How much survives one more block.** The share of the clean projection still
    removed at the following block's output, measured against the $\beta = 1$ run, which
    is untouched. This is the durability the depth question turns on: an edit the next
    block undoes is not a handle on the state.
    """
    _need(stages, ["stage", "layer_scope", "beta", "alpha_mean", "edit_point"],
          "Q13 stage probe")
    rows = stages[np.isclose(stages["beta"].astype(float), float(beta))]
    if rows.empty:
        raise ValueError(f"cannot draw the Q13 stage probe: no rows at beta = {beta:g}")
    keys = ["layer_scope", "edit_point", "stage", "forward_order"]
    means = rows.groupby(keys, observed=True)["alpha_mean"].mean().reset_index()
    # beta = 1 runs the edit rather than skipping it and reproduces the state exactly, so
    # it is the untouched trajectory and the reference "suppressed" is measured against.
    reference = (stages[np.isclose(stages["beta"].astype(float), 1.0)]
                 .groupby(["layer_scope", "stage"], observed=True)["alpha_mean"].mean())
    sites = [p for p in ("block_input", "block_output")
             if p in set(means["edit_point"].astype(str))]

    fig, axes = plt.subplots(1, len(sites) + 1, figsize=(4.6 * (len(sites) + 1), 4.5),
                             squeeze=False)
    flat = [ax for row in axes for ax in row]
    for ax, site in zip(flat, sites):
        here = means[means["edit_point"].astype(str) == site]
        ticks = (here.dropna(subset=["alpha_mean"])
                 .drop_duplicates("forward_order").sort_values("forward_order"))
        for scope, group in here.groupby("layer_scope", observed=True):
            line = group.dropna(subset=["alpha_mean"]).sort_values("forward_order")
            if line.empty:
                continue
            ax.plot(line["forward_order"], line["alpha_mean"],
                    color=Q13_SITE_COLOR[site], lw=1.8, marker="o", markersize=4.5,
                    markeredgecolor="white", markeredgewidth=0.6, alpha=0.9, zorder=3)
            ax.annotate(str(scope), (float(line["forward_order"].iloc[-1]),
                                     float(line["alpha_mean"].iloc[-1])),
                        textcoords="offset points", xytext=(5, 0), fontsize=6.8,
                        va="center", color=Q13_SITE_COLOR[site])
        # The hook's own stage, marked, so "where the edit happened" is never inferred
        # from the shape of the line.
        hook = here[here["stage"] == "after_edit"]["forward_order"]
        if not hook.empty:
            ax.axvline(float(hook.iloc[0]), color=st.GRID, lw=0.9, zorder=0)
        ax.axhline(0, color=st.INK_SOFT, lw=0.9, zorder=2)
        ax.set_xticks(list(ticks["forward_order"]))
        ax.set_xticklabels([_STAGE_TICK.get(t, t) for t in ticks["stage"]], fontsize=7)
        ax.set_xlim(-0.45, float(ticks["forward_order"].max()) + 0.75)
        ax.set_ylabel(r"mean $x^{\top} v^{*}$ of the treated tokens", fontsize=8.5)
        ax.set_xlabel("stage, in the order this site reaches it", fontsize=8.5)
        ax.set_title(Q13_SITE_TITLE[site], fontsize=8.5, color=Q13_SITE_COLOR[site], pad=8)

    bar_ax = flat[len(sites)]
    bars = []
    for (scope, site), group in means.groupby(["layer_scope", "edit_point"], observed=True):
        nxt = group[group["stage"] == "after_next_block"]["alpha_mean"].dropna()
        clean = float(reference.get((scope, "after_next_block"), float("nan")))
        if nxt.empty or not np.isfinite(clean) or abs(clean) < 1e-12:
            continue
        left = float(nxt.iloc[0])
        bars.append(dict(depth=str(scope).replace("_postmlp", ""), site=str(site),
                         suppressed=1.0 - left / clean,
                         # A sign flip only means something where there is a magnitude to
                         # have a sign. Near total suppression the residue is numerical
                         # noise about zero and its sign is a coin toss, so calling that
                         # "inverted" would report precision as a finding.
                         flipped=bool(np.sign(left) != np.sign(clean)
                                      and abs(left) >= 0.05 * abs(clean))))
    if bars:
        frame = pd.DataFrame(bars)
        depths = sorted(set(frame["depth"]))
        width, x = 0.36, np.arange(len(depths))
        for k, site in enumerate(sites):
            values, flags = [], []
            for depth in depths:
                match = frame[(frame["site"] == site) & (frame["depth"] == depth)]
                values.append(float(match["suppressed"].iloc[0]) if not match.empty
                              else float("nan"))
                flags.append(bool(match["flipped"].iloc[0]) if not match.empty else False)
            offset = (k - (len(sites) - 1) / 2.0) * width
            bar_ax.bar(x + offset, values, width, color=Q13_SITE_COLOR[site],
                       edgecolor="white", linewidth=0.7, zorder=3,
                       label=Q13_SITE_LEGEND[site])
            for xi, value, flag in zip(x + offset, values, flags):
                if not np.isfinite(value):
                    continue
                bar_ax.annotate(f"{value:.0%}" + ("\nsign flipped" if flag else ""),
                                (xi, value), ha="center", fontsize=6.8, va="bottom",
                                xytext=(0, 3), textcoords="offset points")
        bar_ax.set_xticks(x); bar_ax.set_xticklabels(depths, fontsize=7.5)
        bar_ax.axhline(1.0, color=SHAM, lw=1.0, ls=(0, (4, 2)), zorder=1)
        bar_ax.set_ylabel("share of the clean projection removed,\none block later",
                          fontsize=8.5)
        bar_ax.set_xlabel("depth", fontsize=8.5)
        bar_ax.set_title("durability", fontsize=8.5, pad=8)
        bar_ax.legend(frameon=False, fontsize=7.0, loc="upper center",
                      bbox_to_anchor=(0.5, -0.16), ncol=1)
    else:
        _no_data(bar_ax, "no untouched reference at beta = 1\nto measure suppression against")
    for ax in flat:
        st.open_box(ax); st.light_grid(ax, "y")
    fig.subplots_adjust(left=0.07, right=0.98, top=0.88, bottom=0.28, wspace=0.30)
    _panel_letters(fig, flat)
    head = f"{checkpoint}: " if checkpoint else ""
    return _finish(fig, (
        f"{head}the same edit at $\\beta$ = {beta:g} installed at two sites inside the "
        f"block, and what the block does with it. The first panels give the treated "
        f"tokens' projection at each stage, ordered as the forward pass reaches it under "
        f"that site: a block-input hook fires before the block runs and a block-output "
        f"hook after, so the two orders differ and one shared axis would read the block's "
        f"response to the edit as its cause. An edit that lands at zero and leaves the "
        f"block near the clean value was rewritten by that block's own feed-forward. The "
        f"last panel is the share of the clean projection still removed at the following "
        f"block's output, against the $\\beta$ = 1 run, which reproduces the state "
        f"exactly and is therefore the untouched trajectory. A sign flip is marked: it is "
        f"the strongest evidence an edit survived, because no amount of shrinkage "
        f"produces one."), path, tight=False)


# ==============================================  true rotation in the (v*, u) plane
Q13_TARGET_COLOR = {"random_orthogonal": st.OKABE_ITO["orange"],
                    "residual_pc1": BLUE,
                    "semantic_direction": st.OKABE_ITO["green"]}
Q13_TARGET_LABEL = {"random_orthogonal": "random direction orthogonal to $v^*$ (the null)",
                    "residual_pc1": "residual PC1 (the population's own axis)",
                    "semantic_direction": "supplied semantic direction"}


def fig_q13_rotation_plane(plane: pd.DataFrame, path=None, *, checkpoint: str = "",
                           target: str = "", layer: Optional[int] = None):
    r"""Where the treated tokens really sit in the $(v^*, u)$ plane, before and after.

    Measured activations projected onto the two basis vectors, not a schematic. Each
    arrow is one token, from its clean coordinates to its rotated ones, and the dashed
    circle is its own norm: a rotation moves a token **along** that circle, which is
    what "the norm is held fixed exactly" looks like when it is drawn rather than
    asserted.

    The plane shows $\alpha = x^\top v^*$ and $b = x^\top u$ and nothing else, so
    ``out_of_plane`` is stated: it is the length of the token that neither axis sees, it
    is identical before and after by construction, and where it dominates the token's
    norm the visible arrow is a small part of a much longer vector.
    """
    _need(plane, ["alpha_before", "b_before", "alpha_after", "b_after"],
          "Q13 rotation plane")
    rows = plane
    if target and "rotation_target" in rows.columns:
        rows = rows[rows["rotation_target"] == target]
    if layer is not None and "layer" in rows.columns:
        rows = rows[rows["layer"].astype(int) == int(layer)]
    if rows.empty:
        raise ValueError("cannot draw the Q13 rotation plane: no rows for "
                         f"target={target!r} layer={layer!r}")
    angles = sorted(set(rows["theta_deg"].dropna().astype(float))) \
        if "theta_deg" in rows.columns else [0.0]
    extreme = max(angles, key=abs) if angles else 0.0
    turned = rows[np.isclose(rows["theta_deg"].astype(float), extreme)] \
        if "theta_deg" in rows.columns else rows
    one_unit = turned.drop_duplicates("token") if "token" in turned.columns else turned

    fig, ax = plt.subplots(figsize=(6.4, 6.2))
    limit = float(np.nanmax(np.abs(np.concatenate([
        one_unit["alpha_before"].to_numpy(), one_unit["b_before"].to_numpy(),
        one_unit["alpha_after"].to_numpy(), one_unit["b_after"].to_numpy()])))) * 1.22
    limit = limit if np.isfinite(limit) and limit > 0 else 1.0
    for _, row in one_unit.iterrows():
        radius = float(np.hypot(row["alpha_before"], row["b_before"]))
        ax.add_patch(plt.Circle((0, 0), radius, fill=False, color=st.GRID, lw=0.8,
                                ls=(0, (2, 3)), zorder=1))
        # The path along the circle, not the chord between its ends. A straight arrow
        # from the clean point to the rotated one reads as a translation, which is the
        # one thing this operator does not do. It would show the token leaving its own
        # norm circle and returning, and that never happens.
        start = float(np.arctan2(row["b_before"], row["alpha_before"]))
        stop = start + math.radians(float(extreme))
        sweep = np.linspace(start, stop, 64)
        ax.plot(radius * np.cos(sweep), radius * np.sin(sweep), color=BLUE, lw=1.6,
                alpha=0.9, zorder=4, solid_capstyle="round")
        # One arrowhead at the end, over the last step of the arc, so the direction of
        # travel is visible without the head implying a straight path.
        ax.annotate("", xy=(radius * np.cos(sweep[-1]), radius * np.sin(sweep[-1])),
                    xytext=(radius * np.cos(sweep[-6]), radius * np.sin(sweep[-6])),
                    arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.6,
                                    shrinkA=0, shrinkB=0), zorder=4)
    ax.scatter(one_unit["alpha_before"], one_unit["b_before"], s=46, color="white",
               edgecolor=st.INK_SOFT, linewidth=1.3, zorder=5, label="clean")
    ax.scatter(one_unit["alpha_after"], one_unit["b_after"], s=46, color=BLUE,
               edgecolor="white", linewidth=0.9, zorder=5,
               label=f"rotated by {extreme:g}$\\degree$")
    ax.axhline(0, color=st.INK_SOFT, lw=0.9, zorder=2)
    ax.axvline(0, color=st.INK_SOFT, lw=0.9, zorder=2)
    ax.set_xlim(-limit, limit); ax.set_ylim(-limit, limit)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$\alpha = x^{\top} v^{*}$")
    ax.set_ylabel(r"$b = x^{\top} u$")
    name = Q13_TARGET_LABEL.get(target, target or "u")
    head = f"{checkpoint} " if checkpoint else ""
    ax.set_title(f"{head}$(v^*, u)$ plane — $u$ = {name}", fontsize=9.5, pad=10)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    st.open_box(ax)
    hidden = (float(one_unit["out_of_plane_before"].mean())
              if "out_of_plane_before" in one_unit.columns else float("nan"))
    norm = float(one_unit["norm_before"].mean()) if "norm_before" in one_unit.columns \
        else float("nan")
    share = (hidden ** 2) / (norm ** 2) if np.isfinite(hidden) and norm else float("nan")
    drift = (float((one_unit["norm_after"] - one_unit["norm_before"]).abs().max())
             if {"norm_after", "norm_before"} <= set(one_unit.columns) else float("nan"))
    fig.subplots_adjust(left=0.14, right=0.97, top=0.92, bottom=0.10)
    return _finish(fig, (
        f"{head}the treated tokens' real coordinates in the $(v^*, u)$ plane, from the "
        f"measured activations rather than a diagram. Each arrow is one token moving from "
        f"its clean position to its position after a {extreme:g}$\\degree$ rotation; the "
        f"dashed circle through each starting point is that token's own norm, and the "
        f"path runs along it rather than across it, which is what holding the norm fixed "
        f"looks like when it is drawn rather than asserted (largest norm change here: "
        f"{drift:.2e}). Only "
        f"$\\alpha$ and $b$ are shown: a mean {hidden:.3g} of each token's length lies "
        f"outside the plane, {share:.1%} of its squared norm, and a plane rotation cannot "
        f"move any of it -- so the visible arrow is the whole of what changed and a small "
        f"part of where the token is."), path, tight=False)


def fig_q13_rotation_response(turned: pd.DataFrame, path=None, *, checkpoint: str = ""):
    r"""What the angle bought, per target: alignment, sinkhood, the manifold, the image.

    **(a)** The realised alignment against $\theta$. At $\theta = 90^\circ$,
    $\alpha' = -b$ and $b' = \alpha$: the rotation sweeps the whole $v^*$ alignment out
    and puts $b$ in its place. So the legend prints $b^2/\|x\|^2$, what the rotation
    brings *in*, rather than the plane's total share. The share is
    $\alpha^2/\|x\|^2 + b^2/\|x\|^2$ and its first term **is** $\cos^2$, so on a
    register token aligned with $v^*$ it exceeds 0.96 for every $u$ and distinguishes
    none of them.

    **(b)** Sink strength against the realised alignment, which is the arm's question:
    can the register state be moved off $v^*$ and stop acting as a sink?

    **(c)** On-manifold energy and the image distance together. A rotation that suppresses
    sinkhood by leaving the manifold has not shown that direction carries the mechanism;
    it has shown that a broken state is not a sink.
    """
    _need(turned, ["rotation_target", "theta_deg"], "Q13 rotation response")
    if turned.empty:
        raise ValueError("cannot draw the Q13 rotation response: the table is empty")
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.5))
    sink = next((c for c in ("selected_sink_strength_mean", "mean_sink_strength")
                 if c in turned.columns and turned[c].notna().any()), None)
    image = next((c for c in ("lpips", "rmse") if c in turned.columns
                  and turned[c].notna().any()), None)
    for target, group in turned.groupby("rotation_target", observed=True):
        line = group.sort_values("theta_deg")
        colour = Q13_TARGET_COLOR.get(str(target), st.INK_SOFT)
        # b^2/||x||^2, not the plane share. The share is alpha^2 + b^2 over the squared
        # norm, and alpha^2 IS cos^2, so on a register token aligned with v* the share
        # exceeds 0.96 whatever u is, and cannot tell one u from another. What the
        # rotation brings IN to v* at 90 degrees is b, so b^2 is the discriminating
        # quantity and the one worth printing.
        brought = (float(line["b2_share"].mean()) if "b2_share" in line.columns
                   else float("nan"))
        removable = (float(line["alpha2_share"].mean()) if "alpha2_share" in line.columns
                     else float("nan"))
        label = f"{Q13_TARGET_LABEL.get(str(target), str(target))}"
        if np.isfinite(brought):
            label += (f"\n$b^2/|x|^2$ = {brought:.3%}"
                      + (f"  ($\\cos^2$ = {removable:.1%})"
                         if np.isfinite(removable) else ""))
        axes[0].plot(line["theta_deg"], line["cosine_vstar_after"], color=colour, lw=2.0,
                     marker="o", markersize=4.5, markeredgecolor="white",
                     markeredgewidth=0.6, zorder=3, label=label)
        if sink:
            axes[1].plot(line["cosine_vstar_after"], line[sink], color=colour, lw=1.6,
                         marker="o", markersize=4.5, markeredgecolor="white",
                         markeredgewidth=0.6, zorder=3)
            for _, row in line.iterrows():
                if abs(float(row["theta_deg"])) in (0.0, 90.0):
                    axes[1].annotate(f"{row['theta_deg']:g}$\\degree$",
                                     (row["cosine_vstar_after"], row[sink]),
                                     textcoords="offset points", xytext=(4, 4),
                                     fontsize=6.6, color=colour)
        if "onmanifold_ratio_to_clean" in line.columns:
            axes[2].plot(line["theta_deg"], line["onmanifold_ratio_to_clean"],
                         color=colour, lw=1.8, marker="o", markersize=4.0,
                         markeredgecolor="white", markeredgewidth=0.6, zorder=3)
    axes[0].axvline(0, color=st.GRID, lw=0.9, zorder=0)
    axes[0].set_xlabel(r"$\theta$  (rotation in the $(v^*, u)$ plane, degrees)")
    axes[0].set_ylabel(r"realised $\cos(x, v^{*})$ after the rotation", fontsize=8.5)
    axes[0].legend(frameon=False, fontsize=6.8, loc="upper center",
                   bbox_to_anchor=(0.5, -0.17), ncol=1)
    if sink:
        axes[1].set_xlabel(r"realised $\cos(x, v^{*})$")
        axes[1].set_ylabel("sink strength of the treated tokens", fontsize=8.5)
    else:
        _no_data(axes[1], "no sink readout in this run")
    axes[2].axhline(1.0, color=SHAM, lw=1.0, ls=(0, (4, 2)), zorder=1)
    axes[2].set_xlabel(r"$\theta$  (degrees)")
    axes[2].set_ylabel("on-manifold energy\n(1.0 = as deep as a clean token)", fontsize=8.5)
    if image:
        twin = axes[2].twinx()
        for target, group in turned.groupby("rotation_target", observed=True):
            line = group.sort_values("theta_deg")
            twin.plot(line["theta_deg"], line[image],
                      color=Q13_TARGET_COLOR.get(str(target), st.INK_SOFT), lw=1.2,
                      ls=(0, (3, 2)), marker="s", markersize=3.2, alpha=0.8, zorder=2)
        twin.set_ylabel(f"{image.upper()} from the clean image (dashed)", fontsize=8.0)
    for ax in axes:
        st.open_box(ax); st.light_grid(ax, "y")
    fig.subplots_adjust(left=0.06, right=0.94, top=0.93, bottom=0.30, wspace=0.34)
    _panel_letters(fig, list(axes))
    head = f"{checkpoint}: " if checkpoint else ""
    worst = (float(turned["norm_relative_error_max"].max())
             if "norm_relative_error_max" in turned.columns else float("nan"))
    return _finish(fig, (
        f"{head}a true rotation of the register state in the $(v^*, u)$ plane, at a norm "
        f"the operator preserves rather than restores -- the largest relative norm change "
        f"across every angle and target is {worst:.1e}. (a) The alignment each angle "
        f"achieved. At $\\theta = 90^\\circ$ the rotation sets $\\alpha' = -b$, so it "
        f"sweeps the $v^*$ alignment out and substitutes $b$; the legend therefore gives "
        f"$b^2/|x|^2$, the alignment each target can bring in, and $\\cos^2$ "
        f"beside it. The plane's total share is the sum of those two and its first term is "
        f"$\\cos^2$ itself, so on a register token aligned with $v^*$ it exceeds 0.96 "
        f"whatever $u$ is and tells one target from another not at all. (b) Sink strength "
        f"against the alignment actually realised -- the arm's question, since $\\theta$ "
        f"itself is not what the model sees. (c) Whether the rotated state is one the "
        f"clean population could have contained, with the image distance beside it: "
        f"suppressing sinkhood by leaving the manifold shows that a broken state is not a "
        f"sink, not that direction carries the mechanism."), path, tight=False)


# =====================================  rotation strip: the images, and what moved
Q13_ROTATION_TARGET_TITLE = {
    "ordinary_mean": "rotated toward where ORDINARY tokens point",
    "ordinary_pc1": "rotated toward how ordinary tokens vary most",
    "residual_pc1": "rotated toward the residual's leading axis",
    "random_orthogonal": "rotated toward a random direction",
    "semantic_direction": "rotated toward the supplied semantic direction",
}
# The four mechanism readouts, in the order the project usually reads them: how many loud
# aligned tokens survive, what the dominant channel is doing, and whether the sinks hold.
# The four the question asks about. The treated tokens' NORM is left out on purpose: it
# is preserved by the operator at the hook, but every table
# column named `norm` is measured at a block output downstream, so a panel of it would
# vary and read as the operator failing when it is the network's own response.
Q13_STRIP_READOUTS: Tuple[Tuple[str, str], ...] = (
    ("n_highnorm", "high-norm tokens"),
    ("n_highnorm_and_aligned", "high-norm AND aligned"),
    ("dominant_channel_value", "dominant channel value"),
    ("selected_sink_strength_mean", "sink strength, treated"),
    ("n_sinks", "tokens meeting the sink criterion"),
)


def fig_q13_rotation_strip(images: pd.DataFrame, root, readouts: Optional[pd.DataFrame] = None,
                           path=None, *, checkpoint: str = "", target: str = "",
                           max_columns: int = 9):
    r"""What the picture does as the register is rotated, with the mechanism beneath it.

    One column per angle: the generated image on top, and under the whole strip the
    mechanism readouts against the same $\theta$ axis, so a change in the picture can be
    read against what happened to the high-norm population, the dominant channel and the
    sinks at the same angle.

    The images share the strip's own ordering and nothing is rescaled between them --
    they are the generated frames, at the same prompt, seed, sampler and guidance, with
    only the installed rotation different. $\theta = 0$ is generated rather than skipped,
    so the leftmost comparison is against a frame that went through the same operator.
    """
    _need(images, ["theta_deg"], "Q13 rotation strip")
    rows = images[images["theta_deg"].notna()]
    if target and "rotation_target" in rows.columns:
        rows = rows[rows["rotation_target"] == target]
    if rows.empty:
        raise ValueError(f"cannot draw the Q13 rotation strip: no rows for target={target!r}")
    if "image_path" not in rows.columns or not rows["image_path"].notna().any():
        raise ValueError(
            "cannot draw the Q13 rotation strip: no images were saved. This checkpoint "
            "has no decoder, or save_images was False -- the strip is the one figure here "
            "that is entirely about the picture, so it refuses rather than drawing frames "
            "of nothing.")
    frames = (rows.dropna(subset=["image_path"]).drop_duplicates("theta_deg")
              .sort_values("theta_deg"))
    if len(frames) > int(max_columns):
        # Keep the ends and thin the middle: the extremes are the comparison, and a strip
        # wider than the page is unreadable.
        keep = np.unique(np.linspace(0, len(frames) - 1, int(max_columns)).round().astype(int))
        frames = frames.iloc[keep]
    angles = [float(a) for a in frames["theta_deg"]]

    available = ([(c, l) for c, l in Q13_STRIP_READOUTS
                  if readouts is not None and not readouts.empty and c in readouts.columns
                  and readouts[c].notna().any()] if readouts is not None else [])
    height = 3.1 + (2.7 if available else 0.0)
    fig = plt.figure(figsize=(max(1.55 * len(frames) + 0.9, 2.7 * max(len(available), 1)),
                              height))
    outer = fig.add_gridspec(2 if available else 1, 1,
                             height_ratios=[1.0, 0.85] if available else [1.0],
                             hspace=0.46)
    grid = outer[0].subgridspec(1, len(frames), wspace=0.06)
    base = Path(root)
    for column, (_, row) in enumerate(frames.iterrows()):
        ax = fig.add_subplot(grid[0, column])  # noqa: E501 - one image per angle
        picture = base / str(row["image_path"])
        try:
            from PIL import Image

            ax.imshow(np.asarray(Image.open(picture).convert("RGB")))
        except Exception as problem:                                   # pragma: no cover
            _no_data(ax, f"image unreadable\n{type(problem).__name__}")
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(st.INK_SOFT if abs(float(row["theta_deg"])) < 1e-9 else st.GRID)
            spine.set_linewidth(1.6 if abs(float(row["theta_deg"])) < 1e-9 else 0.7)
        unmodified = " (unmodified)" if abs(float(row["theta_deg"])) < 1e-9 else ""
        ax.set_title(f"{row['theta_deg']:g}$\\degree${unmodified}", fontsize=8.0, pad=4)

    if available:
        # One panel per readout, each on its OWN y axis. They are a token count, a signed
        # channel value, a sink strength and a norm: no shared scale exists, and dividing
        # each by its unmodified value inverts the ones whose reference is negative and
        # turns the comparison into an artefact.
        table = (readouts[readouts["rotation_target"] == target]
                 if target and "rotation_target" in readouts.columns else readouts)
        inner = outer[1].subgridspec(1, len(available), wspace=0.42)
        colours = (BLUE, st.OKABE_ITO["orange"], st.OKABE_ITO["green"],
                   st.OKABE_ITO["purple"], st.OKABE_ITO["blue"])
        for position, ((column, label), colour) in enumerate(zip(available, colours)):
            panel = fig.add_subplot(inner[0, position])
            line = (table.dropna(subset=[column]).groupby("theta_deg", observed=True)
                    [column].mean().reset_index().sort_values("theta_deg"))
            if line.empty:
                _no_data(panel, f"{label}\nnot measured")
                continue
            panel.plot(line["theta_deg"], line[column], color=colour, lw=1.8, marker="o",
                       markersize=4.0, markeredgecolor="white", markeredgewidth=0.6,
                       zorder=3)
            reference = line.loc[np.isclose(line["theta_deg"], 0.0), column]
            if not reference.empty:
                # The unmodified level, drawn rather than divided by, so a value that
                # crosses zero stays readable.
                panel.axhline(float(reference.iloc[0]), color=st.INK_SOFT, lw=0.9,
                              ls=(0, (3, 2)), zorder=2)
                panel.annotate(f"unmodified {float(reference.iloc[0]):.3g}",
                               (0.02, 0.03), xycoords="axes fraction", fontsize=6.4,
                               color=st.INK_SOFT)
            if float(line[column].max()) == float(line[column].min()):
                # A readout that never moves is a finding, not a blank panel: give it a
                # visible range so the flat line is drawn rather than filling the axis.
                level = float(line[column].iloc[0])
                panel.set_ylim(level - 1.0, level + 1.0)
            panel.axvline(0.0, color=st.GRID, lw=0.9, zorder=0)
            panel.set_xlim(min(angles) - 6, max(angles) + 6)
            panel.set_xticks(angles)
            panel.set_xticklabels([f"{a:g}" for a in angles], fontsize=6.8)
            panel.set_title(label, fontsize=7.8, pad=5)
            panel.set_xlabel(r"$\theta$ (deg)", fontsize=7.5)
            st.open_box(panel); st.light_grid(panel, "y")

    head = f"{checkpoint}: " if checkpoint else ""
    what = Q13_ROTATION_TARGET_TITLE.get(target, f"rotated toward {target}" if target else
                                         "rotated")
    fig.suptitle(f"{head}{what}", fontsize=10, y=0.985)
    fig.subplots_adjust(left=0.055, right=0.985, top=0.88,
                        bottom=0.14 if available else 0.06)
    return _finish(fig, (
        f"{head}the generated image as the register state is {what}, at a norm the "
        f"rotation preserves exactly. Every frame shares the prompt, seed, sampler, step "
        f"count and guidance and differs only in the angle; $\\theta = 0$ is generated "
        f"through the same operator rather than skipped, so the outlined frame is the "
        f"numerical floor and not a separate clean run. Beneath, the mechanism at the "
        f"same angles, each as a share of its own unmodified value because a token count "
        f"and a sink strength do not share a scale: the high-norm aligned population, the "
        f"dominant channel's magnitude, the treated tokens' sink strength, and how many "
        f"tokens still meet the project's sink criterion, and the treated tokens' norm, "
        f"which the operator holds fixed. Each keeps its own y axis with its unmodified "
        f"level drawn rather than divided out: they are a count, a signed channel value, "
        f"a strength and a length, so no shared scale exists and normalising the ones "
        f"whose reference is negative would invert them. Reading a change in the picture "
        f"against the same angle below it is the point of the shared $\\theta$ axis."),
        path, tight=False)
