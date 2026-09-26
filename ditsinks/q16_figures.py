"""Paper figures for the register-retiming main run.

Every figure is drawn from the pooled unit records (:func:`ditsinks.q16_main.collect_units`)
and the frozen :class:`~ditsinks.q16_main.DepthProtocol` of each checkpoint, in the
repository's publication style (:mod:`ditsinks.style`). Conventions shared by all of them:

- **Colour means the model** in every panel: FLUX.1-dev blue, PixArt-Sigma vermillion
  (Okabe-Ito, validated for colour-vision deficiency on the adjacent and all-pairs lists).
  The unmodified run is gray, and the natural register interval is a light gray band.
- **Intervals are 95% percentile bootstrap intervals clustered by prompt**: a prompt's
  seeds share its layout and register positions, so prompts are resampled whole.
- **Names are formal.** Conditions are named by what was done and where ("Induced at
  blocks 12-16", "Natural register removed"), never by code names.

Each ``fig_*`` function returns a matplotlib figure; :func:`make_all_figures` renders them
all to PDF and PNG and writes a draft caption for each.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib as mpl
import numpy as np

from . import style as S
from .q16_main import DepthProtocol, FORMAL_METRIC_NAMES, condition_catalog

__all__ = [
    "MODEL_NAMES", "MODEL_COLORS", "model_name", "model_color", "fig_design",
    "fig_image_effect_by_depth", "fig_attention_by_depth", "fig_writer_response",
    "fig_removal", "fig_specificity", "fig_examples", "fig_threshold_range",
    "fig_boundaries", "make_all_figures", "summary_table", "example_columns",
    "rank_example_units", "EXAMPLE_PRESETS", "fig_guidance_ablation",
]

MODEL_NAMES = {
    "flux1-dev": "FLUX.1-dev", "flux1-schnell": "FLUX.1-schnell",
    "pixart-sigma-1024": "PixArt-Σ", "pixart-sigma-512": "PixArt-Σ (512 px)",
    "tiny-flux1": "Synthetic FLUX", "tiny-pixart": "Synthetic PixArt",
}
MODEL_COLORS = {
    "flux1-dev": S.OKABE_ITO["blue"], "flux1-schnell": S.OKABE_ITO["blue"],
    "tiny-flux1": S.OKABE_ITO["blue"],
    "pixart-sigma-1024": S.OKABE_ITO["vermillion"], "pixart-sigma-512": S.OKABE_ITO["vermillion"],
    "tiny-pixart": S.OKABE_ITO["vermillion"],
}
REFERENCE_GRAY = "#7a7a7a"
NATURAL_FILL = "#ebebeb"
NATURAL_EDGE = "#bdbdbd"
# Ordered classes of where a head's attention went, most to least of the clean sink left:
# an ordinal gray ramp (ordered categories take a one-hue ramp, not categorical hues).
ATTENTION_GROUPS = (
    ("Clean sink kept", ("kept",), "#262626"),
    ("Another register token", ("other_original_carrier", "relocated_vstar_carrier",
                                "clean_register_elsewhere"), "#6b6b6b"),
    ("A non-register token", ("non_register_token",), "#a8a8a8"),
    ("No dominant key", ("spread_out",), "#dcdcdc"),
)


def model_name(checkpoint: str) -> str:
    return MODEL_NAMES.get(str(checkpoint), str(checkpoint))


def model_color(checkpoint: str) -> str:
    return MODEL_COLORS.get(str(checkpoint), S.OKABE_ITO["green"])


def _ordered(protocols: Mapping[str, DepthProtocol]) -> List[str]:
    order = ["flux1-dev", "flux1-schnell", "pixart-sigma-1024", "pixart-sigma-512",
             "tiny-flux1", "tiny-pixart"]
    return sorted(protocols, key=lambda k: order.index(k) if k in order else len(order))


def _span(protocol: DepthProtocol, name: str) -> Tuple[int, int]:
    a, b = protocol.windows[name]
    return int(a), int(b)


def _range_label(a: int, b: int) -> str:
    return f"{a}–{b}" if a != b else f"{a}"


def _natural_band(ax, protocol: DepthProtocol, *, label: bool = False) -> None:
    ax.axvspan(protocol.formation - 0.5, protocol.natural_end + 0.5, color=NATURAL_FILL,
               zorder=0, lw=0, label="Natural register interval" if label else None)


def _depth_conditions(protocol: DepthProtocol) -> List[str]:
    return ([f"induce_{w}" for w in protocol.early_windows()]
            + (["induce_N"] if "N" in protocol.windows else [])
            + [f"induce_{w}" for w in protocol.late_windows()])


def _clean_axes(ax) -> None:
    S.open_box(ax)
    S.light_grid(ax, "y")


# ================================================================ statistics
def summary_table(frame, value: str, *, by: Sequence[str] = ("checkpoint", "condition"),
                  iterations: int = 2000, seed: int = 0):
    """Mean and prompt-clustered 95% interval of ``value`` per group of ``by``."""
    import pandas as pd
    from .causal_stats import bootstrap_mean

    rows = []
    if frame is None or frame.empty or value not in frame:
        return pd.DataFrame(columns=list(by) + ["mean", "ci_low", "ci_high", "n_units",
                                                "n_prompts"])
    for keys, block in frame.groupby(list(by), sort=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        estimate = bootstrap_mean(block, value, iterations=iterations, seed=seed)
        rows.append(dict(zip(by, keys), mean=estimate.value, ci_low=estimate.ci_low,
                         ci_high=estimate.ci_high, n_units=estimate.n_units,
                         n_prompts=estimate.n_clusters))
    return pd.DataFrame(rows)


def _errorbar(ax, x, row, color, *, marker="o", filled=True, label=None, size=5.5):
    y, lo, hi = row["mean"], row["ci_low"], row["ci_high"]
    has = np.isfinite(lo) and np.isfinite(hi)
    ax.errorbar([x], [y], yerr=[[y - lo], [hi - y]] if has else None, fmt=marker,
                color=color, mfc=color if filled else "white", mec=color, ms=size,
                mew=1.2, elinewidth=1.1, capsize=2.5, zorder=4, label=label)


# ================================================================ Figure 1
def fig_design(protocols: Mapping[str, DepthProtocol], lifecycle=None):
    """The measured natural interval and every window, per model, on the block axis."""
    import matplotlib.pyplot as plt

    S.use("paper")
    models = _ordered(protocols)
    fig, axes = plt.subplots(len(models), 1, figsize=(7.0, 1.85 * len(models) + 0.3),
                             squeeze=False)
    for ax, checkpoint in zip(axes[:, 0], models):
        p = protocols[checkpoint]
        color = model_color(checkpoint)
        _natural_band(ax, p)
        rows = {"windows": 2.0, "removal": 1.0, "extend": 0.0}
        # Register population of the unmodified run, scaled to the band above the rows.
        if lifecycle is not None and not lifecycle.empty:
            ref = lifecycle[(lifecycle["checkpoint"] == checkpoint)
                            & (lifecycle["condition"] == "reference")
                            & (lifecycle["step"] == p.readout_step)]
            if not ref.empty and "n_highnorm_frozen" in ref:
                curve = ref.groupby("layer")["n_highnorm_frozen"].median()
                if curve.max() > 0:
                    y = 3.0 + 0.9 * curve / curve.max()
                    ax.fill_between(curve.index, 3.0, y, step="mid", color=REFERENCE_GRAY,
                                    alpha=0.35, lw=0)
                    ax.step(curve.index, y, where="mid", color=REFERENCE_GRAY, lw=1.0)
        for name in p.early_windows() + (["N"] if "N" in p.windows else []) + p.late_windows():
            a, b = _span(p, name)
            face = "white" if name == "N" else color
            ax.add_patch(plt.Rectangle((a - 0.45, rows["windows"] - 0.32), b - a + 0.9, 0.64,
                                       facecolor=face, edgecolor=color, lw=1.0,
                                       alpha=1.0 if name == "N" else 0.30, zorder=3))
            ax.add_patch(plt.Rectangle((a - 0.45, rows["windows"] - 0.32), b - a + 0.9, 0.64,
                                       fill=False, edgecolor=color, lw=1.0, zorder=4))
            ax.text(0.5 * (a + b), rows["windows"], name, ha="center", va="center",
                    fontsize=7.5, color=S.INK, zorder=5)
        a, b = _span(p, "removal")
        ax.add_patch(plt.Rectangle((a - 0.45, rows["removal"] - 0.32), b - a + 0.9, 0.64,
                                   facecolor="white", edgecolor=REFERENCE_GRAY, hatch="////",
                                   lw=1.0, zorder=3))
        a, b = _span(p, "extend")
        ax.add_patch(plt.Rectangle((a - 0.45, rows["extend"] - 0.32), b - a + 0.9, 0.64,
                                   facecolor="white", edgecolor=color, lw=1.2, zorder=3))
        if p.seam is not None:
            ax.axvline(p.seam - 0.5, color=S.INK_SOFT, lw=0.8, zorder=2)
            ax.text(p.seam - 0.9, 4.2, "\u2190 dual-stream blocks", ha="right", va="bottom",
                    fontsize=7, color=S.INK_SOFT)
            ax.text(p.seam - 0.1, 4.2, "single-stream blocks \u2192", ha="left", va="bottom",
                    fontsize=7, color=S.INK_SOFT)
        ax.set_yticks([0.0, 1.0, 2.0, 3.45])
        ax.set_yticklabels(["Maintained", "Removed", "Induced", "Register\npopulation"],
                           fontsize=8)
        ax.set_ylim(-0.6, 4.7)
        ax.set_xlim(-0.8, p.n_layers - 0.2)
        ax.set_title(model_name(checkpoint), loc="left", fontsize=10)
        ax.tick_params(axis="y", length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.tick_params(top=False, right=False)
    axes[-1, 0].set_xlabel("Transformer block")
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=NATURAL_FILL, edgecolor="none",
                     label="Natural register interval (measured)"),
               Patch(facecolor="#bcd6ea", edgecolor=S.INK_SOFT, label="Induction window"),
               Patch(facecolor="white", edgecolor=S.INK_SOFT,
                     label="Rewritten in place / maintained"),
               Patch(facecolor="white", edgecolor=REFERENCE_GRAY, hatch="////",
                     label="Natural register removed")]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=7.5, frameon=False,
               bbox_to_anchor=(0.5, -0.005))
    fig.tight_layout(h_pad=0.8, rect=(0, 0.05, 1, 1))
    return fig


# ================================================================ Figure 2
def fig_image_effect_by_depth(images, protocols: Mapping[str, DepthProtocol], *,
                              metrics: Sequence[str] = ("lpips",
                                                        "clip_prompt_similarity_change"),
                              iterations: int = 2000):
    """Image change against the depth at which the register state was introduced."""
    import matplotlib.pyplot as plt

    S.use("paper")
    models = [m for m in _ordered(protocols) if images is not None and not images.empty
              and (images["checkpoint"] == m).any()]
    metrics = [m for m in metrics if images is not None and m in images
               and images[m].notna().any()]
    if not models or not metrics:
        return None
    fig, axes = plt.subplots(len(metrics), len(models),
                             figsize=(3.45 * len(models), 2.35 * len(metrics) + 0.35),
                             squeeze=False, sharey="row")
    for j, checkpoint in enumerate(models):
        p = protocols[checkpoint]
        color = model_color(checkpoint)
        block = images[images["checkpoint"] == checkpoint]
        for i, metric in enumerate(metrics):
            ax = axes[i, j]
            _natural_band(ax, p)
            stats = summary_table(block, metric, iterations=iterations).set_index("condition")
            for side in (p.early_windows(), p.late_windows()):
                xs, ys = [], []
                for name in side:
                    key = f"induce_{name}"
                    if key not in stats.index:
                        continue
                    a, b = _span(p, name)
                    xs.append(0.5 * (a + b))
                    ys.append(stats.loc[key, "mean"])
                    _errorbar(ax, xs[-1], stats.loc[key], color)
                if len(xs) > 1:
                    ax.plot(xs, ys, color=color, lw=1.0, zorder=3)
            if "induce_N" in stats.index:
                a, b = _span(p, "N")
                _errorbar(ax, 0.5 * (a + b), stats.loc["induce_N"], color, marker="D",
                          filled=False, size=5)
            if metric != "lpips":
                ax.axhline(0.0, color=S.RULE, lw=0.7, zorder=1)
            ax.set_xlim(-0.8, p.n_layers - 0.2)
            _clean_axes(ax)
            if i == 0:
                # Window names along the top: the blocks are in the design figure.
                top = ax.get_xaxis_transform()
                for name in p.early_windows() + ["N"] + p.late_windows():
                    if name in p.windows:
                        ax.text(0.5 * sum(_span(p, name)), 1.01, name, transform=top,
                                ha="center", va="bottom", fontsize=7, color=S.INK_SOFT)
                ax.set_title(model_name(checkpoint), loc="left", fontsize=10, pad=14)
            if j == 0:
                ax.set_ylabel(FORMAL_METRIC_NAMES.get(metric, metric).replace(" to the ", "\nto the ")
                              .replace("Change in ", "Change in\n"), fontsize=9)

    handles = [plt.Line2D([], [], color=S.INK, marker="o", lw=0, ms=5,
                          label="Induced before formation or after the natural end"),
               plt.Line2D([], [], color=S.INK, marker="D", mfc="white", lw=0, ms=5,
                          label="Rewritten in place (natural depth)"),
               plt.Rectangle((0, 0), 1, 1, color=NATURAL_FILL, label="Natural register interval")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.supxlabel("Transformer block at the centre of the induction window", y=0.055,
                  fontsize=mpl.rcParams["axes.labelsize"], fontweight="bold")
    return fig


# ================================================================ Figure 3
def _heat_rows(protocol: DepthProtocol) -> List[str]:
    keys = ["reference"] + _depth_conditions(protocol) + ["remove"]
    keys += [f"remove_induce_{w}" for w in (protocol.primary_early, protocol.primary_late) if w]
    keys += ["extend"]
    return keys


def _generic_label(key: str) -> str:
    """A model-independent row name: the window by its name, defined in the design figure."""
    if key == "reference":
        return "Unmodified"
    if key == "remove":
        return "Natural register removed"
    if key == "extend":
        return "Maintained past the natural end"
    if key == "induce_N":
        return "Rewritten in place (N)"
    if key.startswith("remove_induce_"):
        return f"Removed; induced at {key.split('_')[-1]}"
    if key.startswith("induce_"):
        return f"Induced at {key.split('_')[-1]}"
    return key


def fig_attention_by_depth(lifecycle, protocols: Mapping[str, DepthProtocol], *,
                           step: Optional[int] = None, value: str = "carrier_incoming_mass"):
    """Attention received by the register tokens, block by block, in every condition."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    S.use("paper")
    if lifecycle is None or lifecycle.empty or value not in lifecycle:
        return None
    models = [m for m in _ordered(protocols) if (lifecycle["checkpoint"] == m).any()]
    if not models:
        return None
    rows: List[str] = []
    for checkpoint in models:
        rows += [k for k in _heat_rows(protocols[checkpoint]) if k not in rows]
    matrices = {}
    for checkpoint in models:
        p = protocols[checkpoint]
        at = p.readout_step if step is None else int(step)
        block = lifecycle[(lifecycle["checkpoint"] == checkpoint) & (lifecycle["step"] == at)]
        matrix = np.full((len(rows), p.n_layers), np.nan)
        for r, key in enumerate(rows):
            curve = block[block["condition"] == key].groupby("layer")[value].median()
            for layer, v in curve.items():
                if 0 <= int(layer) < p.n_layers and np.isfinite(v) and v > 0:
                    matrix[r, int(layer)] = v
        matrices[checkpoint] = matrix
    finite = np.concatenate([m[np.isfinite(m)] for m in matrices.values()])
    if finite.size == 0:
        return None
    # One colour scale for both models: the shares are comparable fractions of attention.
    norm = LogNorm(vmin=max(float(np.nanpercentile(finite, 2)), 1e-5),
                   vmax=float(np.nanmax(finite)))
    widths = [protocols[m].n_layers for m in models]
    fig, axes = plt.subplots(1, len(models), figsize=(2.3 + 2.4 * len(models), 3.7),
                             squeeze=False, gridspec_kw=dict(width_ratios=widths))
    image = None
    for k, (ax, checkpoint) in enumerate(zip(axes[0], models)):
        p = protocols[checkpoint]
        catalog = {c.key: c for c in condition_catalog(p)}
        image = ax.imshow(matrices[checkpoint], aspect="auto", cmap=S.PROBABILITY_CMAP,
                          norm=norm, interpolation="nearest")
        for r, key in enumerate(rows):
            condition = catalog.get(key)
            if condition is None or condition.window is None:
                continue
            a, b = _span(p, condition.window)
            ax.add_patch(plt.Rectangle((a - 0.5, r - 0.5), b - a + 1, 1.0, fill=False,
                                       edgecolor="white", lw=1.1))
        ax.axvline(p.formation - 0.5, color="white", lw=0.6, ls=(0, (2, 2)))
        ax.axvline(p.natural_end + 0.5, color="white", lw=0.6, ls=(0, (2, 2)))
        if p.seam is not None:
            ax.axvline(p.seam - 0.5, color="#f0f0f0", lw=0.9)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([_generic_label(r) for r in rows] if k == 0 else [], fontsize=7)
        ax.set_xlabel("Transformer block")
        ax.set_title(model_name(checkpoint), loc="left", fontsize=10)
        ax.tick_params(top=False, right=False)
    fig.tight_layout(rect=(0, 0, 0.9, 1))
    cax = fig.add_axes([0.915, 0.2, 0.014, 0.65])
    bar = fig.colorbar(image, cax=cax)
    bar.set_label("Attention share received by the register tokens", fontsize=8)
    bar.outline.set_linewidth(0.8)
    bar.ax.tick_params(width=0.8, direction="in", labelsize=7)
    return fig


# ================================================================ Figure 4
def fig_writer_response(lifecycle, protocols: Mapping[str, DepthProtocol], *,
                        iterations: int = 2000):
    """What the natural writer does when the register state was already written earlier."""
    import matplotlib.pyplot as plt
    import pandas as pd
    from .q16_main import writer_gain_table

    S.use("paper")
    if lifecycle is None or lifecycle.empty:
        return None
    models = [m for m in _ordered(protocols) if (lifecycle["checkpoint"] == m).any()]
    gains, extra = [], []
    for checkpoint in models:
        p = protocols[checkpoint]
        table = writer_gain_table(lifecycle, p)
        if not table.empty:
            gains.append(table)
        natural = lifecycle[(lifecycle["checkpoint"] == checkpoint)
                            & (lifecycle["step"] == p.readout_step)
                            & lifecycle["layer"].between(p.formation, p.natural_end)]
        if "n_highnorm_new" in natural:
            per_unit = natural.groupby(["checkpoint", "prompt_id", "seed", "condition"])[
                "n_highnorm_new"].mean().reset_index()
            extra.append(per_unit)
    if not gains:
        return None
    gains = pd.concat(gains, ignore_index=True)
    extra = pd.concat(extra, ignore_index=True) if extra else pd.DataFrame()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))
    width = 0.8 / max(len(models), 1)
    for m, checkpoint in enumerate(models):
        p = protocols[checkpoint]
        keys = _depth_conditions(p)
        catalog = {c.key: c for c in condition_catalog(p)}
        color = model_color(checkpoint)
        stats = summary_table(gains[gains["checkpoint"] == checkpoint],
                              "writer_gain_vs_reference", iterations=iterations
                              ).set_index("condition")
        for i, key in enumerate(keys):
            if key in stats.index:
                _errorbar(axes[0], i + (m - (len(models) - 1) / 2) * width, stats.loc[key],
                          color, label=model_name(checkpoint) if i == 0 else None)
        if not extra.empty:
            stats = summary_table(extra[extra["checkpoint"] == checkpoint], "n_highnorm_new",
                                  iterations=iterations).set_index("condition")
            for i, key in enumerate(["reference"] + keys):
                if key in stats.index:
                    _errorbar(axes[1], i + (m - (len(models) - 1) / 2) * width,
                              stats.loc[key], color)
    first = protocols[models[0]]
    names = {c.key: c for c in condition_catalog(first)}
    keys = _depth_conditions(first)
    axes[0].axhline(1.0, color=S.RULE, lw=0.7)
    axes[0].axhline(0.0, color=S.RULE, lw=0.5)
    axes[0].set_xticks(range(len(keys)))
    axes[0].set_xticklabels([names[k].window for k in keys], fontsize=8)
    axes[0].set_ylabel("v* written at the writer block\n(relative to unmodified)", fontsize=9)
    axes[0].set_xlabel("Window of induction")
    axes[1].set_xticks(range(len(keys) + 1))
    axes[1].set_xticklabels(["None"] + [names[k].window for k in keys], fontsize=8)
    axes[1].set_ylabel("New high-norm tokens per block\nin the natural interval", fontsize=9)
    axes[1].set_xlabel("Window of induction")
    for ax, tag in zip(axes, ("(a)", "(b)")):
        _clean_axes(ax)
        S.panel_label(ax, tag, dx=-0.22)
    if len(models) > 1:
        axes[0].legend(fontsize=8, loc="best")
    fig.tight_layout()
    return fig


# ================================================================ Figure 5
def fig_removal(images, attention, protocols: Mapping[str, DepthProtocol], *,
                iterations: int = 2000):
    """Removing the natural register state, and relocating it earlier or later."""
    import matplotlib.pyplot as plt

    S.use("paper")
    models = _ordered(protocols)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw=dict(width_ratios=[1.25, 1]))
    # (a) where attention went, over the natural interval, at the readout step
    ax = axes[0]
    y, ticks, labels = 0.0, [], []
    if attention is not None and not attention.empty:
        for checkpoint in models:
            p = protocols[checkpoint]
            catalog = {c.key: c for c in condition_catalog(p)}
            keys = ["reference", "remove"] + [f"remove_induce_{w}" for w in
                                              (p.primary_early, p.primary_late) if w]
            block = attention[(attention["checkpoint"] == checkpoint)
                              & (attention["step"] == p.readout_step)
                              & attention["layer"].between(p.formation + 1, p.natural_end)]
            header = y + 0.55
            ax.plot([0.012], [header], transform=ax.get_yaxis_transform(), marker="s", ms=5,
                    color=model_color(checkpoint), clip_on=False)
            ax.text(0.03, header, model_name(checkpoint), transform=ax.get_yaxis_transform(),
                    ha="left", va="center", fontsize=8.5, fontweight="bold", color=S.INK)
            y -= 0.45
            for key in keys:
                rows = block[(block["condition"] == key) & (block["n_affected_heads"] > 0)]
                if rows.empty:
                    continue
                weights = rows["n_affected_heads"].to_numpy(float)
                left = 0.0
                for name, classes, shade in ATTENTION_GROUPS:
                    share = sum(np.average(rows[f"affected_{c}"].fillna(0).to_numpy(float),
                                           weights=weights)
                                for c in classes if f"affected_{c}" in rows)
                    ax.barh(y, share, left=left, height=0.62, color=shade,
                            edgecolor="white", lw=1.0,
                            label=name if (checkpoint == models[0] and key == "reference")
                            else None)
                    left += share
                ticks.append(y)
                labels.append(_generic_label(key))
                y -= 1.0
            y -= 0.6
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Share of heads whose clean sink was a register token")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=7,
                  frameon=False)
        S.open_box(ax)
        ax.tick_params(axis="y", length=0)
    S.panel_label(ax, "(a)", dx=-0.02, dy=1.03)
    # (b) the image effect of each, against the unmodified image
    ax = axes[1]
    if images is not None and not images.empty and "lpips" in images and images["lpips"].notna().any():
        width = 0.8 / max(len(models), 1)
        first = protocols[models[0]]
        keys = ["remove"] + [f"remove_induce_{w}" for w in (first.primary_early,
                                                            first.primary_late) if w] + ["extend"]
        for m, checkpoint in enumerate(models):
            stats = summary_table(images[images["checkpoint"] == checkpoint], "lpips",
                                  iterations=iterations).set_index("condition")
            p = protocols[checkpoint]
            local = ["remove"] + [f"remove_induce_{w}" for w in (p.primary_early,
                                                                p.primary_late) if w] + ["extend"]
            for i, key in enumerate(local):
                if key in stats.index:
                    _errorbar(ax, i + (m - (len(models) - 1) / 2) * width, stats.loc[key],
                              model_color(checkpoint),
                              label=model_name(checkpoint) if i == 0 else None)
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels(["Removed", "Removed,\nearly", "Removed,\nlate",
                            "Maintained"][:len(keys)], fontsize=7.5)
        ax.set_ylabel(FORMAL_METRIC_NAMES["lpips"].replace(" to the ", "\nto the "), fontsize=9)
        _clean_axes(ax)
        if len(models) > 1:
            ax.legend(fontsize=7.5, loc="best")
    S.panel_label(ax, "(b)", dx=-0.25, dy=1.03)
    fig.tight_layout()
    return fig


# ================================================================ Figure 6
def fig_specificity(images, protocols: Mapping[str, DepthProtocol], *,
                    metric: str = "lpips", iterations: int = 2000):
    """Is the effect of early induction specific to v* at the register positions?"""
    import matplotlib.pyplot as plt
    from .causal_stats import paired_effect

    S.use("paper")
    if images is None or images.empty or metric not in images or images[metric].isna().all():
        return None
    models = [m for m in _ordered(protocols) if (images["checkpoint"] == m).any()]
    rows = [("primary", "v* state at the register positions"),
            ("control_random_direction", "Random direction, norm-matched"),
            ("control_ordinary_positions", "Same state at non-register positions"),
            ("control_in_distribution", "v* at in-distribution strength"),
            ("induce_N", "Rewritten in place (natural depth)")]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5), sharey=True,
                             gridspec_kw=dict(width_ratios=[1, 1]))
    height = 0.7 / max(len(models), 1)
    for m, checkpoint in enumerate(models):
        p = protocols[checkpoint]
        color = model_color(checkpoint)
        block = images[images["checkpoint"] == checkpoint]
        primary = f"induce_{p.primary_early}"
        stats = summary_table(block, metric, iterations=iterations).set_index("condition")
        for i, (key, _) in enumerate(rows):
            key = primary if key == "primary" else key
            yv = i + (m - (len(models) - 1) / 2) * height
            if key in stats.index:
                row = stats.loc[key]
                has = np.isfinite(row["ci_low"]) and np.isfinite(row["ci_high"])
                axes[0].errorbar([row["mean"]], [yv],
                                 xerr=[[row["mean"] - row["ci_low"]],
                                       [row["ci_high"] - row["mean"]]] if has else None,
                                 fmt="o", color=color, ms=5.5, elinewidth=1.1, capsize=2.5,
                                 label=model_name(checkpoint) if i == 0 else None)
            if key == primary or not (block["condition"] == key).any():
                continue
            estimate = paired_effect(block, metric, treatment=primary, reference=key,
                                     iterations=iterations)
            has = estimate.has_interval
            axes[1].errorbar([estimate.value], [yv],
                             xerr=[[estimate.value - estimate.ci_low],
                                   [estimate.ci_high - estimate.value]] if has else None,
                             fmt="o", color=color, ms=5.5, elinewidth=1.1, capsize=2.5)
    axes[0].set_yticks(range(len(rows)))
    axes[0].set_yticklabels([name for _, name in rows], fontsize=7.5)
    axes[0].invert_yaxis()
    axes[0].set_xlabel(FORMAL_METRIC_NAMES.get(metric, metric), fontsize=9)
    axes[1].axvline(0.0, color=S.RULE, lw=0.8)
    axes[1].set_xlabel("Paired difference: v* state at the register\npositions minus the "
                       "control", fontsize=9)
    for ax in axes:
        S.open_box(ax)
        S.light_grid(ax, "x")
        ax.tick_params(axis="y", length=0)
    S.panel_label(axes[0], "(a)", dx=-0.02, dy=1.04)
    S.panel_label(axes[1], "(b)", dx=-0.02, dy=1.04)
    if len(models) > 1:
        axes[0].legend(fontsize=7.5, loc="lower right")
    fig.tight_layout(w_pad=1.2)
    return fig


# ================================================================ Figure 7
# Named column sets for the example grid. Every set starts with the unmodified image, the
# comparison every other column is read against.
EXAMPLE_PRESETS = ("depth", "key", "all", "largest")
# Metrics on which a smaller value means a larger change.
_SMALLER_IS_MORE_CHANGE = {"psnr", "clip_image_similarity"}
_SHORT_METRIC = {"lpips": "LPIPS", "psnr": "PSNR (dB)", "pixel": "mean absolute pixel difference",
                 "clip_image_similarity": "CLIP image similarity"}


def _all_conditions(protocol: DepthProtocol) -> List[str]:
    return [c.key for c in condition_catalog(protocol, include_check=False)]


def example_columns(protocol: DepthProtocol, conditions="depth") -> List[str]:
    """The condition keys the example grid shows, in generation order.

    ``conditions`` is a preset or an explicit list of keys:

    - ``"depth"``: the depth sweep (every window before formation, in place, after the natural
      end) and removal;
    - ``"key"``: the earliest and the primary early window, in place, the primary and the
      latest late window, removal, removal plus early induction, and the random-direction
      control;
    - ``"all"``: every condition of the run (the hooks-only check is omitted: it is
      bit-identical to the unmodified image by construction);
    - ``"largest"``: filled in by :func:`fig_examples` from the measured change (see
      ``n_columns`` there); here it returns ``"all"``.
    """
    every = _all_conditions(protocol)
    if isinstance(conditions, str):
        if conditions not in EXAMPLE_PRESETS:
            raise ValueError(f"unknown column preset {conditions!r}; use one of "
                             f"{EXAMPLE_PRESETS} or a list of condition keys")
        if conditions == "depth":
            keys = ["reference"] + _depth_conditions(protocol) + ["remove"]
        elif conditions == "key":
            early, late = protocol.early_windows(), protocol.late_windows()
            keys = ["reference"]
            keys += [f"induce_{w}" for w in dict.fromkeys(
                early[:1] + ([protocol.primary_early] if protocol.primary_early else []))]
            keys += ["induce_N"] if "N" in protocol.windows else []
            keys += [f"induce_{w}" for w in dict.fromkeys(
                ([protocol.primary_late] if protocol.primary_late else []) + late[-1:])]
            keys += ["remove", f"remove_induce_{protocol.primary_early}",
                     "control_random_direction"]
        else:
            keys = every
    else:
        keys = ["reference"] + [str(k) for k in conditions if str(k) != "reference"]
    known = set(every) | {"hooks_only"}
    unknown = [k for k in keys if k not in known]
    if unknown:
        raise ValueError(f"not a condition of this run: {unknown}; the conditions are {every}")
    return [k for k in dict.fromkeys(keys) if k in known]


def _short_title(key: str, protocol: DepthProtocol) -> str:
    """A two-line column title; the caption carries the full names."""
    catalog = {c.key: c for c in condition_catalog(protocol)}
    condition = catalog.get(key)
    span = (_range_label(*_span(protocol, condition.window))
            if condition is not None and condition.window in protocol.windows else "")
    if key == "reference":
        return "Unmodified"
    if key == "hooks_only":
        return "Hooks only"
    if key == "remove":
        return "Removed"
    if key.startswith("remove_induce_"):
        return f"Removed +\ninduced {span}"
    if key == "extend":
        return f"Maintained\nto {_span(protocol, 'extend')[1]}"
    if key == "control_random_direction":
        return f"Random\ndirection {span}"
    if key == "control_ordinary_positions":
        return f"Non-register\npositions {span}"
    if key == "control_in_distribution":
        return f"In-distribution\nv* {span}"
    if key == "induce_N":
        return f"In place\n{span}"
    return f"Induced\n{span}"


def _unit_folder(root: Path, prompt_id: int, seed: int) -> Path:
    return Path(root) / "units" / f"p{int(prompt_id):03d}_s{int(seed)}"


def _load(root: Path, prompt_id: int, seed: int, key: str, *, full: bool = False):
    """A unit's image as float RGB in [0, 1]: the full-size PNG when ``full`` and present,
    else the thumbnail."""
    from PIL import Image

    folder = _unit_folder(root, prompt_id, seed)
    for path in ([folder / "images" / f"{key}.png"] if full else []) + [
            folder / "thumbnails" / f"{key}.jpg"]:
        if path.exists():
            return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return None


def _resized(array, size: int):
    from PIL import Image

    image = Image.fromarray((np.clip(array, 0, 1) * 255).astype(np.uint8))
    return np.asarray(image.resize((size, size), Image.BILINEAR), dtype=np.float32) / 255.0


def _pixel_change(root: Path, prompt_id: int, seed: int, key: str, *, size: int = 128):
    """Mean absolute pixel difference to the unit's unmodified image (thumbnails)."""
    reference = _load(root, prompt_id, seed, "reference")
    other = _load(root, prompt_id, seed, key)
    if reference is None or other is None:
        return float("nan")
    return float(np.abs(_resized(other, size) - _resized(reference, size)).mean())


def _prompt_text(root: Path, prompt_id: int, seed: int) -> str:
    import json

    marker = _unit_folder(root, prompt_id, seed) / "unit.json"
    try:
        return str(json.loads(marker.read_text()).get("prompt", ""))
    except Exception:
        return ""


def _available_units(root: Path) -> List[Tuple[int, int]]:
    out = []
    for folder in sorted((Path(root) / "units").glob("p*_s*")):
        if (folder / "thumbnails" / "reference.jpg").exists():
            pid, seed = folder.name.split("_s")
            out.append((int(pid[1:]), int(seed)))
    return out


def rank_example_units(images, checkpoint: str, conditions: Sequence[str], *,
                       root: Optional[Path] = None, metric: str = "lpips",
                       statistic: str = "mean", seeds: Optional[Sequence[int]] = None):
    """Every prompt-seed pair of ``checkpoint``, ranked by how much the shown conditions
    change its image, most first.

    The score is the ``statistic`` (``"mean"`` or ``"max"``) of ``metric`` over
    ``conditions`` (the unmodified column excluded). ``metric`` is a column of the pooled
    ``images`` table (LPIPS by default; for PSNR and CLIP image similarity a lower value
    ranks first) or ``"pixel"``, the mean absolute pixel difference computed from the
    thumbnails under ``root``; the thumbnails are also used when the table has no value. The per-condition values are returned alongside, so the notebook can show
    the table and the figure can be built from any row of it.
    """
    import pandas as pd

    shown = [k for k in conditions if k != "reference"]
    table = None
    if metric != "pixel" and images is not None and not images.empty and metric in images:
        block = images[(images["checkpoint"] == checkpoint) & images["condition"].isin(shown)]
        if not block.empty:
            table = block.pivot_table(index=["prompt_id", "seed"], columns="condition",
                                      values=metric, aggfunc="mean")
    empty = pd.DataFrame(columns=["prompt_id", "seed", "prompt", "score", "metric",
                                  "most_changed_condition"])
    if table is None or table.empty:
        rows = {unit: {k: _pixel_change(root, *unit, k) for k in shown}
                for unit in (_available_units(root) if root is not None else [])}
        if not rows or not shown:
            return empty
        table = pd.DataFrame.from_dict(rows, orient="index")
        table.index = pd.MultiIndex.from_tuples(table.index, names=["prompt_id", "seed"])
        metric = "pixel"
    table = table.reindex(columns=[k for k in shown if k in table.columns])
    if seeds is not None:
        table = table[table.index.get_level_values("seed").isin([int(s) for s in seeds])]
    if table.empty:
        return empty
    values = table.to_numpy(dtype=float)
    sign = -1.0 if metric in _SMALLER_IS_MORE_CHANGE else 1.0
    change = sign * values                        # larger = more change, whatever the metric
    with np.errstate(all="ignore"):
        score = (np.nanmax(change, axis=1) if statistic == "max"
                 else np.nanmean(change, axis=1)) * sign
    out = table.reset_index()
    out.insert(2, "score", score)
    out.insert(3, "metric", metric)
    largest = []
    for row in change:
        largest.append(str(table.columns[int(np.nanargmax(row))])
                       if np.isfinite(row).any() else None)
    out.insert(4, "most_changed_condition", largest)
    out = out.dropna(subset=["score"]).sort_values("score", ascending=sign < 0)
    if root is not None:                          # only units whose images are on disk
        present = set(_available_units(root))
        out = out[[(int(a), int(b)) in present for a, b in zip(out["prompt_id"], out["seed"])]]
    if root is not None:
        out.insert(2, "prompt", [_prompt_text(root, p, s) for p, s in
                                 zip(out["prompt_id"], out["seed"])])
    return out.reset_index(drop=True)


def _normalise_units(units, checkpoint: str, seed: Optional[int]):
    """``units`` as given in the notebook: prompt ids, (prompt, seed) pairs, or a dict of
    either per checkpoint. A bare prompt id takes ``seed``, or its most-changed seed when
    ``seed`` is None (marked by a seed of None here)."""
    if units is None:
        return None
    if isinstance(units, Mapping):
        units = units.get(checkpoint)
        if units is None:
            return None
    out = []
    for entry in units:
        if isinstance(entry, (tuple, list)):
            out.append((int(entry[0]), int(entry[1])))
        else:
            out.append((int(entry), None if seed is None else int(seed)))
    return out


def _select_units(ranking, *, units, select: str, n_rows: int, seed: Optional[int],
                  primary: str, images, checkpoint: str, root: Path):
    """The (prompt, seed) pairs to show and a sentence stating how they were chosen."""
    if units is not None:
        chosen = []
        for prompt_id, s in units:
            if s is None:
                candidates = ranking[ranking["prompt_id"] == prompt_id] if len(ranking) else []
                s = int(candidates["seed"].iloc[0]) if len(candidates) else 0
            chosen.append((prompt_id, s))
        return chosen, "chosen by the authors"
    if select == "most_changed":
        chosen, seen = [], set()
        for _, row in ranking.iterrows():          # one seed per prompt: distinct scenes
            if int(row["prompt_id"]) in seen:
                continue
            seen.add(int(row["prompt_id"]))
            chosen.append((int(row["prompt_id"]), int(row["seed"])))
            if len(chosen) == n_rows:
                break
        used = str(ranking["metric"].iloc[0]) if len(ranking) else "lpips"
        what = _SHORT_METRIC.get(used, used)
        return chosen, (f"the {len(chosen)} prompts whose images change most (by the mean "
                        f"{what} to the unmodified image over the conditions shown, the "
                        "most-changed seed of each); selected for the visibility of the "
                        "change, not as typical cases")
    if select == "percentile":
        s = 0 if seed is None else int(seed)
        chosen: List[Tuple[int, int]] = []
        if images is not None and not images.empty and "lpips" in images:
            block = images[(images["checkpoint"] == checkpoint) & (images["seed"] == s)
                           & (images["condition"] == primary)].dropna(subset=["lpips"])
            ordered = block.sort_values("lpips")["prompt_id"].tolist()
            for q in np.linspace(0.25, 0.75, n_rows) if ordered else []:
                pick = int(ordered[min(int(round(q * (len(ordered) - 1))), len(ordered) - 1)])
                if (pick, s) not in chosen:
                    chosen.append((pick, s))
        return chosen, (f"the prompts at evenly spaced percentiles (25th to 75th) of the "
                        f"effect of induction just before formation, seed {s}")
    if select == "first":
        available = [u for u in _available_units(root) if seed is None or u[1] == int(seed)]
        return available[:n_rows], "the first prompts of the run"
    raise ValueError(f"unknown selection {select!r}: use 'most_changed', 'percentile' or "
                     "'first', or pass units")


def _crop_box(reference, others, *, fraction: float, size: int = 256):
    """The square (x0, y0, side) in [0, 1] units where the shown conditions change the
    image most: the window of side ``fraction`` with the largest summed absolute difference
    to the unmodified image, over every shown condition. One box per unit, so every column
    is cropped at the same place."""
    base = _resized(reference, size)
    heat = np.zeros((size, size), dtype=np.float64)
    for other in others:
        heat += np.abs(_resized(other, size) - base).mean(axis=-1)
    side = max(4, int(round(fraction * size)))
    integral = np.pad(heat.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    sums = (integral[side:, side:] - integral[:-side, side:] - integral[side:, :-side]
            + integral[:-side, :-side])
    y, x = np.unravel_index(int(np.argmax(sums)), sums.shape)
    # Centre the box on the change inside it, so a change at the window's edge is not cut.
    window = heat[y:y + side, x:x + side]
    if window.sum() > 0:
        rows, cols = np.indices(window.shape)
        cy = y + float((rows * window).sum() / window.sum())
        cx = x + float((cols * window).sum() / window.sum())
        y = int(np.clip(round(cy - side / 2), 0, size - side))
        x = int(np.clip(round(cx - side / 2), 0, size - side))
    return x / size, y / size, side / size


def _crop(array, box):
    x0, y0, side = box
    h, w = array.shape[:2]
    a, b = int(round(x0 * w)), int(round(y0 * h))
    n = max(1, int(round(side * min(h, w))))
    return array[b:b + n, a:a + n]


def fig_examples(root_by_model: Mapping[str, Path], protocols: Mapping[str, DepthProtocol],
                 images, *, units=None, select: str = "most_changed", n_rows: int = 3,
                 seed: Optional[int] = None, seeds: Optional[Sequence[int]] = None,
                 conditions="depth", n_columns: int = 6, metric: str = "lpips",
                 detail: Optional[str] = "crop", crop_fraction: float = 0.3,
                 annotate: bool = True, show_prompt: bool = False,
                 panel_inches: float = 0.92):
    """Generations of chosen prompt-seed pairs under chosen conditions, per model.

    **Which units (rows).** ``units`` names them: prompt ids (``[3, 7, 12]``), prompt-seed
    pairs (``[(3, 42), (7, 0)]``), or a dict of either per checkpoint. A bare prompt id is
    shown at ``seed``, or at its most-changed seed when ``seed`` is None. Without ``units``,
    ``select`` chooses ``n_rows`` of them:

    - ``"most_changed"`` (default): the prompts whose images change most over the columns
      shown (:func:`rank_example_units`), one seed per prompt;
    - ``"percentile"``: prompts at the 25th to 75th percentile of the primary early
      window's LPIPS at ``seed`` (typical, not striking, cases);
    - ``"first"``: the first units of the run.

    ``seeds`` restricts the automatic choice to those seeds.

    **Which conditions (columns).** ``conditions`` is a preset of :func:`example_columns`
    (``"depth"``, ``"key"``, ``"all"``) or a list of keys. ``"largest"`` shows the
    ``n_columns`` conditions whose mean change over the chosen rows is largest, in run order.
    The unmodified image is always the first column.

    **Making small changes visible.** ``detail`` adds, under each row, ``"crop"``: the same
    square of every image (side ``crop_fraction`` of the image, cut from the full-size PNG),
    placed where the shown conditions change the unit most and outlined on the row above;
    ``"difference"``: the absolute difference to the unmodified image, on one colour scale
    for the whole figure; ``"both"``; or None. ``annotate`` prints each image's LPIPS to the
    unmodified image in its corner.

    Returns ``{checkpoint: figure}``. Each figure carries ``q16_selection`` (the rows shown
    with their prompt and per-condition change) and ``q16_caption`` (a caption that states
    the selection rule).
    """
    import textwrap
    import warnings

    import matplotlib.pyplot as plt
    import pandas as pd
    from matplotlib.patches import Rectangle

    if detail not in (None, "crop", "difference", "both"):
        raise ValueError("detail is None, 'crop', 'difference' or 'both'")
    S.use("paper")
    models = [m for m in _ordered(protocols) if m in root_by_model]
    figures = {}
    for checkpoint in models:
        p = protocols[checkpoint]
        root = Path(root_by_model[checkpoint])
        primary = f"induce_{p.primary_early}"
        columns = example_columns(p, "all" if conditions == "largest" else conditions)
        ranking = rank_example_units(images, checkpoint, columns, root=root, metric=metric,
                                     seeds=seeds)
        chosen, rule = _select_units(ranking, units=_normalise_units(units, checkpoint, seed),
                                     select=select, n_rows=n_rows, seed=seed,
                                     primary=primary, images=images, checkpoint=checkpoint,
                                     root=root)
        missing = [u for u in chosen
                   if not (_unit_folder(root, *u) / "thumbnails" / "reference.jpg").exists()]
        if missing:
            warnings.warn(f"{model_name(checkpoint)}: no images for prompt-seed pairs "
                          f"{missing} under {root}; they are left out")
        chosen = [u for u in chosen if u not in missing]
        if not chosen:
            continue
        index = ranking.set_index(["prompt_id", "seed"]) if len(ranking) else None
        column_rule = ""
        if conditions == "largest":
            rows = index.loc[[u for u in chosen if u in index.index]] if index is not None \
                else pd.DataFrame()
            means = rows.reindex(columns=columns[1:]).mean(numeric_only=True).dropna()
            keep = set(means.sort_values(ascending=metric in _SMALLER_IS_MORE_CHANGE)
                       .index[:int(n_columns)])
            columns = ["reference"] + [k for k in columns[1:] if k in keep]
            column_rule = (f" Columns: the {len(columns) - 1} conditions with the largest mean "
                           "change over these rows.")
        values = {}
        if images is not None and not images.empty and metric in images:
            block = images[images["checkpoint"] == checkpoint]
            values = {(int(r.prompt_id), int(r.seed), r.condition): getattr(r, metric)
                      for r in block[["prompt_id", "seed", "condition", metric]].itertuples()}

        kinds = ["image"] + (["crop"] if detail in ("crop", "both") else []) \
            + (["difference"] if detail in ("difference", "both") else [])
        n_sub = len(kinds)
        heights = []
        for r in range(len(chosen)):
            heights += [1.0] * n_sub + ([0.12] if r < len(chosen) - 1 else [])
        fig = plt.figure(figsize=(panel_inches * len(columns) + (0.9 if show_prompt else 0.25),
                                  panel_inches * sum(heights) + 0.55))
        grid = fig.add_gridspec(len(heights), len(columns), height_ratios=heights,
                                hspace=0.06, wspace=0.04)

        # Everything a row needs, loaded once; the difference scale is shared by the figure.
        loaded, boxes, diffs = {}, {}, {}
        for unit in chosen:
            full = detail is not None          # full-size PNGs: no JPEG noise in the details
            loaded[unit] = {k: _load(root, *unit, k, full=full) for k in columns}
            reference = loaded[unit]["reference"]
            others = [loaded[unit][k] for k in columns[1:] if loaded[unit][k] is not None]
            if reference is not None and others and "crop" in kinds:
                boxes[unit] = _crop_box(reference, others, fraction=crop_fraction)
            if reference is not None and "difference" in kinds:
                base = _resized(reference, 256)
                diffs[unit] = {k: np.abs(_resized(loaded[unit][k], 256) - base).mean(-1)
                               for k in columns[1:] if loaded[unit][k] is not None}
        every = [d for per in diffs.values() for d in per.values()]
        vmax = max(float(np.percentile(np.concatenate([d.ravel() for d in every]), 99.5)),
                   1e-3) if every else 1.0

        selection = []
        for r, unit in enumerate(chosen):
            prompt_id, s = unit
            top = r * (n_sub + 1)
            prompt = _prompt_text(root, prompt_id, s)
            for sub, kind in enumerate(kinds):
                for c, key in enumerate(columns):
                    ax = fig.add_subplot(grid[top + sub, c])
                    ax.set_xticks([])
                    ax.set_yticks([])
                    for side in ax.spines.values():
                        side.set_linewidth(0.4)
                    picture = loaded[unit][key]
                    if kind == "image" and picture is not None:
                        ax.imshow(_resized(picture, 384) if picture.shape[0] > 384 else picture)
                        if unit in boxes:
                            x0, y0, side = boxes[unit]
                            h, w = ax.get_images()[0].get_array().shape[:2]
                            ax.add_patch(Rectangle(
                                (x0 * w, y0 * h), side * w, side * h, fill=False,
                                edgecolor="white", lw=0.8))
                        value = values.get((prompt_id, s, key))
                        if annotate and key != "reference" and value is not None \
                                and np.isfinite(value):
                            ax.text(0.03, 0.03, f"{value:.2f}", transform=ax.transAxes,
                                    fontsize=5.5, color="white", ha="left", va="bottom",
                                    bbox=dict(boxstyle="square,pad=0.15", fc="black",
                                              ec="none", alpha=0.55))
                    elif kind == "crop" and picture is not None and unit in boxes:
                        ax.imshow(_crop(picture, boxes[unit]))
                    elif kind == "difference":
                        if key == "reference":
                            ax.text(0.5, 0.5, "|Δ| to\nunmodified", ha="center",
                                    va="center", fontsize=6.5, transform=ax.transAxes)
                            for side in ax.spines.values():
                                side.set_visible(False)
                        elif key in diffs.get(unit, {}):
                            ax.imshow(diffs[unit][key], cmap="magma",
                                      norm=mpl.colors.PowerNorm(0.5, vmin=0, vmax=vmax))
                    if r == 0 and sub == 0:
                        ax.set_title(_short_title(key, p), fontsize=6.5, pad=3)
                    if c == 0:
                        if kind == "image":
                            label = f"Prompt {prompt_id}\nseed {s}"
                            if show_prompt and prompt:
                                label = textwrap.shorten(prompt, 60, placeholder="…")
                                label = "\n".join(textwrap.wrap(label, 20)[:3]) + \
                                    f"\n(P{prompt_id}, seed {s})"
                            ax.set_ylabel(label, fontsize=6.5)
                        elif kind == "crop":
                            ax.set_ylabel("Detail", fontsize=6.5)
            ranked = (index.loc[unit] if index is not None and unit in index.index else {})
            selection.append(dict(
                checkpoint=checkpoint, prompt_id=prompt_id, seed=s, prompt=prompt,
                score=ranked.get("score") if len(ranked) else None,
                **{k: (ranked.get(k) if len(ranked) else None) for k in columns[1:]}))
        fig.suptitle(f"{model_name(checkpoint)}", fontsize=8.5, y=0.995)
        fig.subplots_adjust(left=(0.9 if show_prompt else 0.3) / fig.get_figwidth(),
                            right=0.995, bottom=0.01, top=1 - 0.5 / fig.get_figheight())
        fig.q16_selection = pd.DataFrame(selection)
        fig.q16_caption = _examples_caption(rule + "." + column_rule, kinds, annotate,
                                            crop_fraction, vmax, metric)
        figures[checkpoint] = fig
    return figures


def _examples_caption(rule: str, kinds: Sequence[str], annotate: bool, fraction: float,
                      vmax: float, metric: str) -> str:
    text = (f"Generations under the conditions named above each column; every row is one "
            f"prompt-seed pair and shares its initial noise across the row. Rows: {rule}")
    if not text.endswith("."):
        text += "."
    if annotate and metric != "pixel":
        text += f" Corner numbers: {_SHORT_METRIC.get(metric, metric)} to the unmodified image."
    if "crop" in kinds:
        text += (f" Below each row, the same square ({fraction:.0%} of the image side, at full "
                 "resolution) of every image, placed where the shown conditions change that "
                 "prompt-seed pair most; it is outlined in white above.")
    if "difference" in kinds:
        text += (" Below that, the absolute difference to the unmodified image (mean over "
                 "colour channels, pixel values in [0, 1]; one square-root colour scale for "
                 f"the whole figure, 0 to {vmax:.2f}).")
    return text


# ================================================================ Appendix: guidance
def fig_guidance_ablation(images_by_policy: Mapping[str, object], protocol: DepthProtocol, *,
                          metric: str = "lpips", iterations: int = 2000):
    """Where an edit is applied under classifier-free guidance.

    ``images_by_policy`` holds two pooled image tables, the conditional-pass-only run first
    and the both-passes run second, keyed by the formal names the legend prints. Every
    condition's change is shown under both, on the prompt-seed pairs both runs hold (so the
    comparison is paired), with prompt-clustered 95% intervals.
    """
    import matplotlib.pyplot as plt

    S.use("paper")
    checkpoint = protocol.checkpoint
    frames = []
    for name, frame in images_by_policy.items():
        if frame is None or frame.empty or metric not in frame:
            continue
        frame = frame[frame["checkpoint"] == checkpoint]
        if frame[metric].notna().any():
            frames.append((name, frame))
    if len(frames) != 2:
        return None
    shared = set(zip(frames[0][1]["prompt_id"], frames[0][1]["seed"])) & set(
        zip(frames[1][1]["prompt_id"], frames[1][1]["seed"]))
    if not shared:
        return None
    frames = [(name, frame[[(p, s) in shared for p, s in zip(frame["prompt_id"],
                                                             frame["seed"])]])
              for name, frame in frames]
    catalog = [c for c in condition_catalog(protocol, include_check=False)
               if c.group != "reference"]
    keys = [c.key for c in catalog
            if all((frame["condition"] == c.key).any() for _, frame in frames)]
    if not keys:
        return None
    labels = {c.key: c.label(protocol) for c in catalog}
    stats = [summary_table(frame, metric, iterations=iterations).set_index("condition")
             for _, frame in frames]
    color = model_color(checkpoint)
    # The open ring is drawn over the filled dot, so where the two agree both stay visible.
    styles = [dict(mfc="none", mec=REFERENCE_GRAY, color=REFERENCE_GRAY, zorder=4),
              dict(mfc=color, mec=color, color=color, zorder=3)]
    fig, ax = plt.subplots(figsize=(5.8, 0.27 * len(keys) + 1.0))
    for i, key in enumerate(keys):
        means = [float(table.loc[key, "mean"]) for table in stats]
        ax.plot(means, [i, i], color=NATURAL_EDGE, lw=1.0, zorder=1)
        for (name, _), table, style in zip(frames, stats, styles):
            row = table.loc[key]
            has = np.isfinite(row["ci_low"]) and np.isfinite(row["ci_high"])
            ax.errorbar([row["mean"]], [i],
                        xerr=[[row["mean"] - row["ci_low"]], [row["ci_high"] - row["mean"]]]
                        if has else None, fmt="o", ms=5.5, mew=1.2, elinewidth=1.0,
                        capsize=2.0, label=name if i == 0 else None, **style)
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels([labels[k] for k in keys], fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlim(left=0.0)
    ax.set_xlabel(FORMAL_METRIC_NAMES.get(metric, metric), fontsize=9)
    S.open_box(ax)
    S.light_grid(ax, "x")
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, frameon=False,
              fontsize=7.5, handletextpad=0.3, columnspacing=1.2)
    fig.tight_layout()
    guidance = protocol.settings.get("guidance")
    n_prompts = len({p for p, _ in shared})
    fig.q16_caption = (
        f"Classifier-free guidance and the pass an edit is applied to "
        f"({model_name(checkpoint)}). {FORMAL_METRIC_NAMES.get(metric, metric)} for every "
        f"condition with the edit applied to the conditional pass only (open) and to both "
        f"passes of the guided sampler (filled); mean over the {len(shared)} prompt-seed "
        f"pairs run under both ({n_prompts} prompts), with 95% bootstrap intervals "
        f"clustered by prompt. The sampler forms every update as uncond + "
        f"{guidance if guidance is not None else 'g'} × (cond − uncond), so an edit of the "
        f"conditional pass alone enters every update multiplied by the guidance scale.")
    return fig


# ================================================================ Appendix
def fig_threshold_range(sensitivity, protocols: Mapping[str, DepthProtocol]):
    """The range check: each shared threshold moved with the other two held fixed."""
    import matplotlib.pyplot as plt

    S.use("paper")
    if sensitivity is None or sensitivity.empty:
        return None
    panels = [("highnorm_ratio", "carriers_at_peak", "Register tokens at the peak block"),
              ("highnorm_ratio", "natural_end_block", "Measured natural end (block)"),
              ("alignment_quantile", "max_share_of_image_stripped",
               "Largest share of the image\nstripped by the removal rule"),
              ("alignment_quantile", "min_share_of_carriers_caught",
               "Smallest share of register\ntokens caught by the rule"),
              ("sink_threshold", "mean_sink_tokens_per_block", "Sink tokens per block"),
              ("sink_threshold", "share_of_sinks_that_are_carriers",
               "Share of sinks that are\nregister tokens")]
    axis_names = {"highnorm_ratio": "High-norm ratio (× median norm)",
                  "alignment_quantile": "Alignment quantile of ordinary tokens",
                  "sink_threshold": "Sink threshold (× uniform share)"}
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.3))
    models = [m for m in _ordered(protocols) if (sensitivity["checkpoint"] == m).any()]
    for k, (parameter, statistic, ylabel) in enumerate(panels):
        ax = axes[k % 2, k // 2]
        for checkpoint in models:
            block = sensitivity[(sensitivity["checkpoint"] == checkpoint)
                                & (sensitivity["parameter"] == parameter)
                                & (sensitivity["statistic"] == statistic)]
            if block.empty:
                continue
            grouped = block.groupby("value")["result"]
            median, low, high = grouped.median(), grouped.quantile(0.25), grouped.quantile(0.75)
            color = model_color(checkpoint)
            ax.fill_between(median.index, low, high, color=color, alpha=0.15, lw=0)
            ax.plot(median.index, median.values, color=color, lw=1.4, marker="o", ms=3.5,
                    label=model_name(checkpoint))
            chosen = block[block["chosen"]]["value"]
            if len(chosen):
                ax.axvline(float(chosen.iloc[0]), color=S.INK_SOFT, lw=0.8)
        if parameter == "alignment_quantile":
            ax.set_xscale("logit")
            ax.set_xticks([0.99, 0.999, 0.9999])
            ax.set_xticklabels(["0.99", "0.999", "0.9999"])
        elif parameter == "sink_threshold":
            ax.set_xscale("log")
            ax.set_xticks([2, 5, 10, 20, 40])
            ax.set_xticklabels(["2", "5", "10", "20", "40"])
        ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
        ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        ax.set_ylabel(ylabel, fontsize=8)
        if k % 2 == 1:
            ax.set_xlabel(axis_names[parameter], fontsize=8)
        _clean_axes(ax)
    if len(models) > 1:
        axes[0, 0].legend(fontsize=7, loc="best")
    fig.tight_layout()
    return fig


def fig_boundaries(boundaries, protocols: Mapping[str, DepthProtocol]):
    """Where the natural interval was measured on every unit and edited step."""
    import matplotlib.pyplot as plt

    S.use("paper")
    if boundaries is None or boundaries.empty:
        return None
    models = [m for m in _ordered(protocols) if (boundaries["checkpoint"] == m).any()]
    fig, axes = plt.subplots(len(models), 1, figsize=(6.4, 1.5 * len(models) + 0.4),
                             squeeze=False)
    for ax, checkpoint in zip(axes[:, 0], models):
        p = protocols[checkpoint]
        block = boundaries[(boundaries["checkpoint"] == checkpoint) & boundaries["valid"].astype(bool)]
        color = model_color(checkpoint)
        for row_y, column, name in ((1.0, "formation", "Formation"),
                                    (0.0, "natural_end", "Natural end")):
            values = block[column].dropna().astype(int)
            counts = values.value_counts().sort_index()
            if counts.empty:
                continue
            ax.scatter(counts.index, np.full(len(counts), row_y), s=12 + 60 * counts.values / counts.max(),
                       color=color, alpha=0.8, edgecolor="white", lw=0.8, zorder=3)
        ax.axvline(p.formation, color=S.INK_SOFT, lw=0.8)
        ax.axvline(p.natural_end, color=S.INK_SOFT, lw=0.8)
        ax.set_yticks([0.0, 1.0])
        ax.set_yticklabels(["Natural end", "Formation"], fontsize=8)
        ax.set_ylim(-0.6, 1.6)
        ax.set_xlim(-0.8, p.n_layers - 0.2)
        ax.set_title(model_name(checkpoint), loc="left", fontsize=10)
        S.open_box(ax)
    axes[-1, 0].set_xlabel("Transformer block")
    fig.tight_layout()
    return fig


# ================================================================ everything
CAPTIONS = {
    "fig_design": (
        "Depth design. For each model, the natural register interval measured on held-out "
        "calibration prompts (gray band, from formation to the last block at which any "
        "register token passes the register test), the register population of the "
        "unmodified run at the readout step, the equal-length windows at which the register "
        "state is induced (filled: before formation or after the natural end; open: rewritten "
        "in place), the interval cleared when the natural register is removed (hatched), and "
        "the span over which it is maintained past its natural end. On FLUX.1-dev the "
        "vertical rule separates dual-stream from single-stream blocks."),
    "fig_image_effect_by_depth": (
        "Image change against the depth at which the register state is introduced. The "
        "register state -- the natural register's v* projection and norm, as multiples of "
        "each block's median token norm -- is written at the natural register positions over "
        "one window of blocks, during the first third of the denoising trajectory, with the "
        "natural register left in place. Points: mean over {n_units} prompt-seed pairs "
        "({n_prompts} prompts, 5 seeds); bars: 95% bootstrap intervals clustered by prompt. "
        "The open diamond rewrites the state where it already exists and measures the write "
        "itself."),
    "fig_attention_by_depth": (
        "Attention received by the register tokens at every block (median over prompt-seed "
        "pairs, readout step), for the unmodified run and every condition. White boxes mark "
        "the blocks at which each condition edits; dotted lines bound the natural interval."),
    "fig_writer_response": (
        "The natural writer's response to an earlier register state. (a) The change in the "
        "register tokens' v* projection across the formation block, relative to the "
        "unmodified run (1: the writer writes as it naturally does; 0: it adds nothing). "
        "(b) High-norm tokens outside the register positions, per block of the natural "
        "interval. Mean and 95% prompt-clustered bootstrap interval."),
    "fig_removal": (
        "Removing the natural register state and relocating it. (a) For heads whose "
        "strongest image key in the unmodified run is a register token, where that head's "
        "strongest key is after the intervention, over the natural interval. (b) LPIPS to "
        "the unmodified image when the natural state is removed alone, removed and induced "
        "before formation or after the natural end, and maintained past its natural end."),
    "fig_specificity": (
        "Specificity of early induction. (a) LPIPS to the unmodified image for the register "
        "state induced just before formation and for four controls over the same blocks and "
        "steps: a random direction orthogonal to v* with the same projection and norm; the "
        "same v* state at positions that never carry the register; v* at the largest "
        "projection the recipient block already holds; and the state rewritten in place. "
        "(b) Paired differences (induced minus control) with 95% prompt-clustered "
        "intervals."),
    # fig_examples writes its own caption (fig.q16_caption): it states how the rows were
    # chosen, which depends on the notebook's settings.
    "fig_examples": (
        "Generations under the conditions named above each column; every row is one "
        "prompt-seed pair."),
    "fig_threshold_range": (
        "Range check of the three shared thresholds on the unmodified runs: each moved with "
        "the other two held at their chosen values (vertical lines). Median and interquartile "
        "range over prompt-seed pairs."),
    "fig_boundaries": (
        "Measured formation and natural end on every prompt-seed pair and edited step "
        "(marker area: frequency). Vertical lines: the frozen protocol boundaries."),
}


def make_all_figures(pooled, protocols: Mapping[str, DepthProtocol], out_dir, *,
                     root_by_model: Optional[Mapping[str, Path]] = None,
                     iterations: int = 2000,
                     examples: Optional[Mapping[str, object]] = None,
                     ablation: Optional[Mapping[str, Mapping[str, object]]] = None
                     ) -> Dict[str, List[Path]]:
    """Render every figure to PDF and PNG and write ``captions.md``.

    ``examples`` holds the keyword arguments of :func:`fig_examples` (which prompt-seed
    pairs, which conditions, the detail rows); the rows each example figure shows are written
    next to it as ``fig_examples_<model>_rows.csv``. ``ablation`` maps a checkpoint to its
    two pooled image tables, conditional pass only and both passes
    (:func:`fig_guidance_ablation`).
    """
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    images = pooled.get("images")
    units = pooled.get("units")
    n_units = int(len(images.drop_duplicates(["checkpoint", "prompt_id", "seed"]))) \
        if images is not None and not images.empty else 0
    n_prompts = int(images["prompt_id"].nunique()) if images is not None and not images.empty else 0
    built = {
        "fig_design": fig_design(protocols, pooled.get("lifecycle")),
        "fig_image_effect_by_depth": fig_image_effect_by_depth(images, protocols,
                                                               iterations=iterations),
        "fig_attention_by_depth": fig_attention_by_depth(pooled.get("lifecycle"), protocols),
        "fig_writer_response": fig_writer_response(pooled.get("lifecycle"), protocols,
                                                   iterations=iterations),
        "fig_removal": fig_removal(images, pooled.get("attention"), protocols,
                                   iterations=iterations),
        "fig_specificity": fig_specificity(images, protocols, iterations=iterations),
        "fig_threshold_range": fig_threshold_range(pooled.get("sensitivity"), protocols),
        "fig_boundaries": fig_boundaries(pooled.get("boundaries"), protocols),
    }
    for checkpoint, frames in (ablation or {}).items():
        if checkpoint in protocols:
            built[f"fig_guidance_ablation_{checkpoint}"] = fig_guidance_ablation(
                frames, protocols[checkpoint], iterations=iterations)
    if root_by_model:
        for checkpoint, fig in fig_examples(root_by_model, protocols, images,
                                            **dict(examples or {})).items():
            built[f"fig_examples_{checkpoint}"] = fig
    written: Dict[str, List[Path]] = {}
    captions = ["# Q16 figure captions (drafts)\n"]
    for name, fig in built.items():
        if fig is None:
            continue
        paths = []
        for suffix in ("pdf", "png"):
            paths.append(S.savefig(fig, out_dir / f"{name}.{suffix}"))
        selection = getattr(fig, "q16_selection", None)
        if selection is not None:
            selection.to_csv(out_dir / f"{name}_rows.csv", index=False)
        text = getattr(fig, "q16_caption", None)
        plt.close(fig)
        written[name] = paths
        base = "fig_examples" if name.startswith("fig_examples") else name
        if text is None:
            text = CAPTIONS.get(base, "").format(n_units=n_units, n_prompts=n_prompts)
        captions.append(f"## {name}\n\n{text}\n")
    (out_dir / "captions.md").write_text("\n".join(captions))
    return written
