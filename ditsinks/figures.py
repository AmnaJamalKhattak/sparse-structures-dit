"""Publication figures.

Every function returns a matplotlib Figure and attaches a suggested LaTeX
caption as ``fig.dv_caption``; nothing is written to disk unless ``path`` is
given. Methodological detail lives in that caption, never printed on the canvas.

Axis conventions used throughout (t indexes image tokens, c channels, h heads):

    ||x_t||          residual-stream L2 norm of image token t after the block
    a_h(t)           attention head h places on key t, averaged over image queries
    abar(t)          a_h(t) averaged over heads; sums to 1 over image keys
    N * abar(t)      the same in multiples of the uniform share 1/N
    max_t |x_{t,c}|  peak activation of channel c over image tokens
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import LogNorm
from matplotlib.ticker import MaxNLocator

from . import style as st
from .metrics import layer_value_matrix, sink_layer_verdict, sink_profile

# --------------------------------------------------------------- quantities
QUANTITY = {
    "norm": dict(
        phen="highnorm",
        axis="Norm value",
        short="residual-stream norm",
        name="Token norm",
    ),
    "attention": dict(
        phen="sink",
        axis="Incoming attention",
        short="incoming attention mass",
        name="Incoming attention",
    ),
    "attention_ratio": dict(
        phen="sink",
        axis="Incoming attention (relative to uniform)",
        short="incoming attention mass in multiples of the uniform share",
        name="Incoming attention",
    ),
    "channel": dict(
        phen="channel",
        axis="Peak activation value",
        short="per-channel peak activation",
        name="Channel activation",
    ),
}

MODEL_LABEL = {"flux1": "FLUX.1", "flux2": "FLUX.2", "pixart": "PixArt-$\\Sigma$"}


def _save(fig, path):
    return st.savefig(fig, path) if path else None


def _model_name(result) -> str:
    return str(result.meta.get("model", result.cfg.model))


def _last_step(result, step: Optional[int]) -> int:
    return max({r.step for r in result.records.values()}) if step is None else step


def _timestep_of(result, step: int) -> Optional[float]:
    for r in result.records.values():
        if r.step == step and r.timestep is not None:
            return float(r.timestep)
    return None


def _layer_axis(ax, n_layers: int, tick_every: Optional[int] = None) -> None:
    ax.set_xlabel("Layer index")
    ax.set_xlim(-0.5, n_layers - 0.5)
    if tick_every is None:
        tick_every = 1 if n_layers <= 16 else (2 if n_layers <= 40 else 4)
    ax.set_xticks(np.arange(0, n_layers, tick_every))


def _density_matrix(mat: np.ndarray, bins: int, log_y: bool
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """Counts per (value bin, layer), plus the bin edges."""
    finite = mat[np.isfinite(mat)]
    lo, hi = float(np.nanmin(finite)), float(np.nanmax(finite))
    if log_y:
        lo = max(lo, float(finite[finite > 0].min()) if (finite > 0).any() else 1e-12)
        edges = np.geomspace(lo, max(hi, lo * 1.0001), bins + 1)
    else:
        edges = np.linspace(lo, max(hi, lo + 1e-9), bins + 1)
    counts = np.zeros((bins, mat.shape[0]))
    for j in range(mat.shape[0]):
        row = mat[j][np.isfinite(mat[j])]
        counts[:, j], _ = np.histogram(row, bins=edges)
    return counts, edges


def _draw_density(ax, counts, edges, layers, log_y, cmap, vmax=None):
    """Draw the (value bin x layer) count matrix in true data coordinates.

    pcolormesh rather than imshow so the value axis carries real units: the tick
    locator then produces round numbers and a log scale works natively.
    """
    x_edges = np.arange(layers.min() - 0.5, layers.max() + 1.5, 1.0)
    masked = np.ma.masked_less(counts, 1.0)
    mesh = ax.pcolormesh(
        x_edges, edges, masked,
        norm=LogNorm(vmin=1, vmax=max(float(counts.max()), 1.0) if vmax is None else vmax),
        cmap=cmap, shading="flat", rasterized=True,
    )
    if log_y:
        ax.set_yscale("log")
    else:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10]))
        ax.yaxis.set_major_formatter(lambda v, _p: f"{v:,.0f}" if abs(v) >= 1 else f"{v:.3g}")
    ax.set_ylim(edges[0], edges[-1])
    return mesh


def _mark_boundary(ax, x, text: bool = True) -> None:
    """Dashed rule at the dual-stream to single-stream transition."""
    if x is None:
        return
    ax.axvline(x, color="0.45", linewidth=0.9, linestyle=(0, (4, 2)), zorder=6)
    if text:
        ax.annotate("dual $\\rightarrow$ single", xy=(x, 1.0), xycoords=("data", "axes fraction"),
                    xytext=(4, -4), textcoords="offset points", ha="left", va="top",
                    fontsize=plt.rcParams["legend.fontsize"] - 2, color="0.35",
                    bbox=dict(facecolor="white", edgecolor="none", pad=1.0, alpha=0.85))


# ================================================ 1. layer-wise distributions
def fig_layer_distribution(result, quantity: str = "norm", step: Optional[int] = None,
                           bins: int = 400, log_y: bool = False, normalize: str = "none",
                           path=None, title: Optional[str] = None, figsize=(7.2, 4.0)):
    """Layer-wise distribution of a per-token (or per-channel) quantity.

    Counts are binned within each layer and shown on a logarithmic colour scale,
    so the bulk of the distribution reads as a bright ridge and isolated outlier
    tokens remain visible as single cells far above it.
    """
    spec = QUANTITY[quantity]
    theme = st.THEME
    step = _last_step(result, step)
    mat, layers = layer_value_matrix(result, quantity, step=step)
    if mat.size == 0:
        raise ValueError(f"no records for quantity={quantity!r} at step {step}")

    axis_label = spec["axis"]
    if normalize == "median":
        med = np.nanmedian(mat, axis=1, keepdims=True)
        mat = mat / np.clip(med, 1e-12, None)
        log_y = True
        axis_label = axis_label.split(" (")[0] + " (relative to layer median)"
    elif normalize != "none":
        raise ValueError("normalize must be 'none' or 'median'")

    counts, edges = _density_matrix(mat, bins, log_y)

    fig, ax = plt.subplots(figsize=figsize)
    im = _draw_density(ax, counts, edges, layers, log_y, theme.density_cmap)
    st.colorbar(fig, im, ax, "Token count" if quantity != "channel" else "Channel count")

    _layer_axis(ax, len(layers))
    ax.set_ylabel(axis_label)
    b = result.zone_boundary()
    _mark_boundary(ax, b)
    ts = _timestep_of(result, step)
    head = title or (f"Layer-wise {spec['name'].lower()} distribution"
                     + (f", $t={ts:.0f}$" if ts is not None else f", step {step}"))
    ax.set_title(f"{head}  ({_model_name(result)})", loc="center")

    fig.dv_caption = (
        f"Layer-wise distribution of the {spec['short']} in {_model_name(result)}. "
        f"Each column is one transformer layer; colour is the number of "
        f"{'channels' if quantity == 'channel' else 'image tokens'} falling in a bin, on a log scale. "
        + (f"The dashed line marks the dual-stream to single-stream transition. "
           if b is not None else "")
        + f"Denoising step {step}; {len(result.cfg.prompts)} prompts x {len(result.cfg.seeds)} seeds pooled."
    )
    _save(fig, path)
    return fig


# ================================================================== 2. atlas
def fig_layer_atlas(result, step: Optional[int] = None, bins: int = 400,
                    normalize: str = "median", path=None, figsize=(7.2, 7.4)):
    """The three phenomena on one shared layer axis, in the layer-wise-distribution form.

    Panels share the layer axis and the colour scale meaning (log count), so the
    reader compares depth ranges directly: where the norm outliers detach, where
    the attention mass concentrates, and where one channel's peak runs away.

    `normalize="median"` divides the norm and channel panels by their own layer
    median and expresses attention in multiples of the uniform share, which makes
    the three panels comparable and stops the bulk's growth with depth from
    hiding the early layers. `normalize="none"` plots the quantities in the units
    they were measured in, with the per-layer median drawn on top; use it when
    the absolute values are themselves the result.
    """
    theme = st.THEME
    step = _last_step(result, step)
    if normalize not in ("median", "none"):
        raise ValueError("normalize must be 'median' or 'none'")
    if normalize == "median":
        rows = [
            ("norm", "median", "(a)", "High-norm tokens",
             "Norm value\n(relative to median)"),
            ("attention_ratio", "none", "(b)", "Attention sinks",
             "Incoming attention\n(relative to uniform)"),
            ("channel", "median", "(c)", "Massive activation channels",
             "Peak activation\n(relative to median)"),
        ]
    else:
        rows = [
            ("norm", "none", "(a)", "High-norm tokens", "Norm value"),
            ("attention", "none", "(b)", "Attention sinks", "Incoming attention"),
            ("channel", "none", "(c)", "Massive activation channels", "Peak activation value"),
        ]

    fig, axes = plt.subplots(len(rows), 1, figsize=figsize, sharex=True)
    b = result.zone_boundary()
    n_layers = 0
    ims = []
    for ax, (quantity, norm_mode, tag, name, axis_label) in zip(axes, rows):
        mat, layers = layer_value_matrix(result, quantity, step=step)
        n_layers = len(layers)
        if norm_mode == "median":
            med = np.nanmedian(mat, axis=1, keepdims=True)
            mat = mat / np.clip(med, 1e-12, None)
        counts, edges = _density_matrix(mat, bins, log_y=True)
        im = _draw_density(ax, counts, edges, layers, True, theme.density_cmap)
        ims.append(im)
        if normalize == "none":
            # In absolute units the bulk is the reference, so draw it.
            med = np.array([np.nanmedian(mat[j]) for j in range(mat.shape[0])])
            ax.plot(layers, med, color="white", linewidth=1.6, zorder=5)
            ax.plot(layers, med, color="black", linewidth=0.8, zorder=6, label="median")
            if tag == "(a)":
                st.legend_outside(ax, fontsize=plt.rcParams["legend.fontsize"] - 1)
        ax.set_ylabel(axis_label)
        _mark_boundary(ax, b, text=False)
        ax.set_title(name, loc="left", fontsize=plt.rcParams["axes.labelsize"])
        st.panel_label(ax, tag, dx=-0.095, dy=1.01)
        st.colorbar(fig, im, ax, "Count", fraction=0.030, pad=0.012)

    _layer_axis(axes[-1], n_layers)
    for ax in axes[:-1]:
        ax.set_xlabel("")
    ts = _timestep_of(result, step)
    head = ("Layer-wise distributions across the network" if normalize == "median"
            else "Layer-wise distributions in absolute units")
    fig.suptitle(
        f"{head}  ({_model_name(result)}"
        + (f", $t={ts:.0f}$" if ts is not None else f", step {step}") + ")",
        y=0.995,
    )
    fig.subplots_adjust(hspace=0.26, left=0.135, right=0.90, top=0.935, bottom=0.075)

    if normalize == "median":
        units = ("(b) is in multiples of the uniform share $1/N$, and (a) and (c) are divided by "
                 "their own layer median, so depth-dependent growth of the bulk does not obscure "
                 "the outliers.")
    else:
        units = ("All three are in the units they were measured in, with the per-layer median "
                 "drawn in black; the value axes are logarithmic because the bulk grows by orders "
                 "of magnitude with depth.")
    fig.dv_caption = (
        f"Layer-wise distributions of the three phenomena in {_model_name(result)}, "
        "on a shared layer axis. (a) residual-stream token norms, (b) head-mean incoming "
        "image-to-image attention per key token, (c) per-channel peak activation. "
        + units + " Colour is a log-scaled count in every panel."
        + (" Dashed line: dual-stream to single-stream transition." if b is not None else "")
    )
    _save(fig, path)
    return fig


def fig_layer_summary(result, step: Optional[int] = None, path=None, figsize=(7.2, 4.8)):
    """Extremal-to-typical ratios per layer, and the depth ranges where each exceeds threshold.

    Each curve is dimensionless -- the largest value at a layer divided by that
    layer's own reference level -- so all three share one axis without a second
    y-scale.
    """
    from .metrics import layer_table

    theme = st.THEME
    df = layer_table(result)
    if step is not None:
        df = df[df["step"] == step]
    agg = df.groupby("layer", as_index=False).mean(numeric_only=True).sort_values("layer")
    n_layers = int(result.meta.get("n_layers", agg["layer"].max() + 1))
    cfg = result.cfg

    series = [
        ("highnorm", "norm_max_over_median", cfg.highnorm_ratio, "High-norm tokens"),
        ("sink", "sink_ratio_headmax", cfg.sink_ratio_threshold, "Attention sinks"),
        ("channel", "chan_max_over_median", cfg.massive_channel_ratio,
         "Massive activation channels"),
    ]

    fig, (ax, raster) = plt.subplots(
        2, 1, figsize=figsize, sharex=True,
        gridspec_kw=dict(height_ratios=[3.6, 0.9], hspace=0.07))

    for key, col, thr, label in series:
        ax.plot(agg["layer"], agg[col], color=theme.color(key), label=label, zorder=3)
    ax.axhline(1.0, color=st.RULE, linewidth=0.8, zorder=1)
    ax.set_yscale("log")
    ax.set_ylabel("Outlier ratio  (largest / typical)")
    ax.set_ylim(top=ax.get_ylim()[1] * 2.2)
    st.light_grid(ax, "y")
    st.legend_outside(ax, fontsize=plt.rcParams["legend.fontsize"] - 1,
                      handlelength=1.4)

    for i, (key, col, thr, _label) in enumerate(series):
        y = len(series) - 1 - i
        on = agg["layer"][agg[col] >= thr].to_numpy()
        raster.broken_barh([(l - 0.5, 1.0) for l in on], (y + 0.22, 0.56),
                           facecolors=theme.color(key), edgecolor="none")
    raster.set_yticks([i + 0.5 for i in range(len(series))])
    raster.set_yticklabels(["massive channels", "sinks", "high-norm"],
                           fontsize=plt.rcParams["legend.fontsize"] - 1)
    raster.set_ylim(0, len(series))
    raster.set_ylabel("Above\nthreshold", fontsize=plt.rcParams["legend.fontsize"] - 1)
    raster.tick_params(axis="y", length=0)

    b = result.zone_boundary()
    if b is not None:
        for a in (ax, raster):
            a.axvline(b, color=st.RULE, linewidth=0.8, linestyle=(0, (4, 2)), zorder=2)
    _layer_axis(raster, n_layers)
    ax.set_title(f"Depth profile of the three phenomena  ({_model_name(result)})", loc="center")

    fig.dv_caption = (
        f"Extremal-to-typical ratios by layer in {_model_name(result)}. Curves are, "
        r"respectively, $\max_t\|x_t\|/\mathrm{med}_t\|x_t\|$ for high-norm tokens, "
        r"$N\max_t \bar{a}(t)$ for attention sinks, and $\max_c p_c/\mathrm{med}_c p_c$ with "
        r"$p_c=\max_t|x_{t,c}|$ for massive activation channels. A value of 1 "
        "means the largest value at that layer equals its reference level, i.e. nothing is "
        "anomalous. The lower panel marks the layers where each ratio exceeds its threshold: token norm "
        f"$>{cfg.highnorm_ratio:g}\\times$ the layer median, top image key $>"
        f"{cfg.sink_ratio_threshold:g}\\times$ the uniform attention share, channel peak $>"
        f"{cfg.massive_channel_ratio:g}\\times$ the median channel peak. "
        "Curves are averaged over prompts, seeds and captured denoising steps."
    )
    _save(fig, path)
    return fig


def fig_layer_panels(result, step: Optional[int] = None, path=None, figsize=(7.2, 8.6),
                     show_chance: bool = True):
    """Per-phenomenon detail, plus the coupling between high-norm tokens and sinks."""
    from .metrics import layer_table

    theme = st.THEME
    df = layer_table(result)
    if step is not None:
        df = df[df["step"] == step]
    agg = df.groupby("layer", as_index=False).mean(numeric_only=True).sort_values("layer")
    n_layers = int(result.meta.get("n_layers", agg["layer"].max() + 1))
    L = agg["layer"]

    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
    fig.subplots_adjust(hspace=0.22)

    ax = axes[0]
    ax.plot(L, agg["norm_max_over_median"], color=theme.color("highnorm"),
            label="largest / median norm")
    ax.plot(L, agg["n_highnorm"] + 1.0, color=theme.color("highnorm"), linestyle=(0, (2, 1.6)),
            label=r"number of high-norm tokens $+\,1$")
    ax.set_yscale("log")
    ax.set_ylabel("High-norm tokens")

    ax = axes[1]
    ax.plot(L, agg["sink_ratio_headmax"], color=theme.color("sink"),
            label="strongest head")
    ax.plot(L, agg["sink_ratio_headmean"], color=theme.color("sink"), linestyle=(0, (2, 1.6)),
            label=r"head mean")
    ax.axhline(1.0, color=st.RULE, linewidth=0.8)
    ax.set_yscale("log")
    ax.set_ylabel("Attention sinks\n(relative to uniform)")

    ax = axes[2]
    ax.plot(L, agg["chan_max_over_median"], color=theme.color("channel"),
            label="largest / median peak activation")
    ax.plot(L, agg["n_massive_channels"] + 1.0, color=theme.color("channel"),
            linestyle=(0, (2, 1.6)), label=r"number of massive channels $+\,1$")
    ax.set_yscale("log")
    ax.set_ylabel("Massive activation\nchannels")

    ax = axes[3]
    ax.plot(L, agg["top1pct_overlap"], color=theme.color("reference"),
            label="overlap of top-1% by norm and top-1% by attention")
    ax.plot(L, agg["heads_sinking_on_highnorm"], color=theme.color("reference"),
            linestyle=(0, (2, 1.6)), label="heads whose top-1 sink is a high-norm token")
    if show_chance and "jaccard_topk_chance" in agg:
        ax.plot(L, agg["jaccard_topk_chance"], color=st.RULE, linewidth=0.8,
                label="chance level")
    ax.set_ylim(-0.03, 1.15)
    ax.set_ylabel("Coupling  (fraction)")

    for i, ax in enumerate(axes):
        st.light_grid(ax, "y")
        if ax.get_yscale() == "log":
            lo, hi = ax.get_ylim()
            ax.set_ylim(lo, hi * 3.2)          # headroom so the legend clears the curves
        st.legend_outside(ax, fontsize=plt.rcParams["legend.fontsize"] - 0.5)
        st.panel_label(ax, f"({'abcd'[i]})", dx=-0.105, dy=1.02)
        b = result.zone_boundary()
        if b is not None:
            ax.axvline(b, color=st.RULE, linewidth=0.8, linestyle=(0, (4, 2)))
    _layer_axis(axes[-1], n_layers)
    axes[0].set_title(f"Layer-resolved statistics  ({_model_name(result)})", loc="center")

    fig.dv_caption = (
        f"Layer-resolved statistics for {_model_name(result)}. (a)-(c) the three "
        "phenomena separately; (d) their coupling: the fraction of the top 1\\% of "
        "tokens by norm that are also in the top 1\\% by incoming attention, and the "
        "fraction of heads whose strongest image key is a high-norm token. "
        + ("The grey curve is the random-overlap baseline for independent sets of the same size."
           if show_chance else "")
    )
    _save(fig, path)
    return fig


# ============================================== 3. attention, layer by layer
def fig_attention_head_grid(result, layer: int, prompt_id: int = 0, seed: Optional[int] = None,
                            step: Optional[int] = None, max_heads: int = 24, ncols: int = 6,
                            path=None):
    """Per-head attention maps for one layer.

    An attention sink appears as a bright vertical stripe: every query row places
    part of its mass on the same key column.
    """
    theme = st.THEME
    recs = [r for r in result.records.values()
            if r.layer == layer and r.attn_map is not None and r.prompt_id == prompt_id
            and (seed is None or r.seed == seed) and (step is None or r.step == step)]
    if not recs:
        raise ValueError(
            f"No stored attention map for layer {layer}. Add it to SweepConfig.focus_layers "
            f"(currently {sorted(result.cfg.focus_layers)}) and re-run the sweep."
        )
    rec = sorted(recs, key=lambda r: (r.seed, r.step))[-1]
    maps = rec.attn_map.numpy()
    H = min(max_heads, maps.shape[0])
    nrows = math.ceil(H / ncols)
    vmax = float(np.percentile(maps[:H], 99.8))
    sink_tokens = rec.incoming_img2img.argmax(dim=-1).tolist() if rec.incoming_img2img is not None else []
    cell = rec.attn_map_scale or 1
    boundary = rec.attn_map_boundary or 0

    fig, axes = plt.subplots(nrows, ncols, figsize=(1.55 * ncols + 1.0, 1.75 * nrows + 0.9),
                             squeeze=False)
    fig.subplots_adjust(hspace=0.30, wspace=0.08, top=0.90, bottom=0.10, left=0.075, right=0.90)

    im = None
    for h in range(nrows * ncols):
        ax = axes[h // ncols][h % ncols]
        ax.set_xticks([]); ax.set_yticks([])
        if h >= H:
            ax.axis("off")
            continue
        im = ax.imshow(maps[h], cmap=theme.probability_cmap, vmin=0, vmax=vmax, aspect="auto")
        tok = f", sink at token {sink_tokens[h]}" if h < len(sink_tokens) else ""
        ax.set_title(f"head {h}{tok}", fontsize=plt.rcParams["legend.fontsize"] - 1, pad=3)
        if boundary:
            ax.axvline(boundary - 0.5, color="white", linewidth=0.8, linestyle=(0, (3, 2)))
        if h < len(sink_tokens):
            col = boundary + int(sink_tokens[h] / cell)
            if 0 <= col < maps.shape[2]:
                ax.plot([col], [0.0], marker="v", markersize=4.0, color="white",
                        markeredgecolor="black", markeredgewidth=0.5, clip_on=False, zorder=6)
        for s in ax.spines.values():
            s.set_linewidth(0.6)

    fig.supxlabel("Key token index  (text keys first, then image keys)", y=0.028)
    fig.supylabel("Image query token index", x=0.022)
    if im is not None:
        cax = fig.add_axes([0.915, 0.24, 0.014, 0.48])
        cb = fig.colorbar(im, cax=cax)
        cb.set_label("Attention probability")
        cb.outline.set_linewidth(0.8)
        cb.ax.tick_params(width=0.8, direction="in")

    fig.suptitle(
        f"Per-head attention, layer {layer} ({rec.block_name})  "
        f"({_model_name(result)}, prompt {rec.prompt_id}, seed {rec.seed}, step {rec.step})",
        y=0.97,
    )
    pooling = ("no pooling was needed" if cell == 1 else
               f"both axes are mean-pooled at {cell} tokens per cell, which preserves a "
               "one-token-wide sink column that striding would miss")
    keys = "keys ordered [text $|$ image]" if boundary else "image keys"
    fig.dv_caption = (
        f"Per-head image-query attention at layer {layer} ({rec.block_name}) of "
        f"{_model_name(result)}. Rows are image queries, columns are {keys}; {pooling}. "
        + ("The dashed line is the text/image key boundary and the " if boundary else "The ")
        + "marker is each head's strongest image key. A sink is the bright vertical "
        "stripe shared by all query rows."
    )
    _save(fig, path)
    return fig


def fig_sink_profile(result, layer: int, prompt_id: int = 0, seed: Optional[int] = None,
                     step: Optional[int] = None, top_n: int = 5, label_above: float = 2.0,
                     path=None, figsize=(12.8, 3.7)):
    """Incoming attention per image token at one layer, and its spatial arrangement."""
    theme = st.THEME
    recs = [r for r in result.records.values()
            if r.layer == layer and r.prompt_id == prompt_id
            and (seed is None or r.seed == seed) and (step is None or r.step == step)]
    if not recs:
        raise ValueError(f"no record for layer {layer}")
    rec = sorted(recs, key=lambda r: (r.seed, r.step))[-1]
    prof = sink_profile(rec)
    norms = rec.norms.get("post_block")
    n = rec.n_img
    ratio = (prof * n).numpy()
    gh, gw = _grid(result, n)

    # This is intentionally a wide figure.  At the old 7.2-inch width, two
    # vertical colour-bar labels and three panel headings collided in notebook
    # rendering even though the saved PDF was technically valid.
    fig = plt.figure(figsize=figsize, layout="constrained")
    gs = fig.add_gridspec(1, 3, width_ratios=[2.2, 1.0, 1.0], wspace=0.12)

    ax = fig.add_subplot(gs[0])
    ax.plot(np.arange(n), ratio, color=theme.color("sink"), linewidth=0.8)
    ax.axhline(1.0, color=st.RULE, linewidth=0.8)
    ax.set_yscale("log")
    ax.set_xlabel("Image token index")
    ax.set_ylabel("Incoming attention\n(relative to uniform)")
    ax.set_title("(a) Incoming attention by image token", loc="left")
    st.open_box(ax)
    st.light_grid(ax, "y")
    top = [int(t) for t in np.argsort(ratio)[::-1][:top_n] if ratio[t] >= label_above]
    for rank, t in enumerate(top):
        ax.annotate(f"token {t}", xy=(t, ratio[t]),
                    xytext=(0, 5 + 10 * (rank % 2)),
                    textcoords="offset points", ha="center",
                    fontsize=plt.rcParams["legend.fontsize"] - 1)

    ax2 = fig.add_subplot(gs[1])
    im = ax2.imshow(ratio.reshape(gh, gw), cmap=theme.probability_cmap,
                    norm=LogNorm(vmin=max(ratio.min(), 1e-2), vmax=max(ratio.max(), 1.1)))
    ax2.set_xticks([]); ax2.set_yticks([])
    ax2.set_xlabel("Patch column")
    ax2.set_ylabel(f"Patch row  ({gh}x{gw} grid)")
    ax2.set_title("(b) Spatial attention map", loc="left")
    st.colorbar(fig, im, ax2, "Incoming attention\n(relative to uniform)",
                fraction=0.05, pad=0.03)

    ax3 = fig.add_subplot(gs[2])
    if norms is not None:
        nn = norms.numpy() / float(np.median(norms.numpy()))
        im3 = ax3.imshow(nn.reshape(gh, gw), cmap=theme.density_cmap,
                         norm=LogNorm(vmin=0.5, vmax=max(float(nn.max()), 1.1)))
        st.colorbar(fig, im3, ax3, "Token norm\n(relative to median)",
                    fraction=0.05, pad=0.03)
    ax3.set_xticks([]); ax3.set_yticks([])
    ax3.set_xlabel("Patch column")
    ax3.set_ylabel(f"Patch row  ({gh}x{gw} grid)")
    ax3.set_title("(c) Spatial token-norm map", loc="left")

    fig.suptitle(f"Attention sinks at layer {layer} ({rec.block_name})  "
                 f"({_model_name(result)}, prompt {rec.prompt_id}, seed {rec.seed})")
    fig.dv_caption = (
        f"Attention sinks at layer {layer} ({rec.block_name}) of {_model_name(result)}. "
        r"(a) head-mean incoming image-to-image attention per key token, renormalised over "
        r"image keys and expressed in multiples of the uniform share $1/N$; the labelled "
        "tokens are the sinks. (b) the same quantity on the patch grid. (c) residual-stream "
        "norm relative to the layer median, on the same grid, showing that the sinks and the "
        "high-norm tokens occupy the same positions."
    )
    _save(fig, path)
    return fig


def fig_layer_head_sinks(result, step: Optional[int] = None, path=None, figsize=(7.2, 3.6)):
    """Sink strength for every (layer, head) pair."""
    from .metrics import head_table

    theme = st.THEME
    df = head_table(result)
    if df.empty:
        raise ValueError("no attention captured")
    if step is not None:
        df = df[df["step"] == step]
    piv = df.pivot_table(index="head_id", columns="layer",
                         values="sink_ratio_over_uniform", aggfunc="mean")
    vals = piv.to_numpy()

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(vals, aspect="auto", cmap=theme.probability_cmap,
                   norm=LogNorm(vmin=max(np.nanmin(vals), 1.0), vmax=max(np.nanmax(vals), 1.1)),
                   origin="lower",
                   extent=[piv.columns.min() - 0.5, piv.columns.max() + 0.5,
                           -0.5, piv.shape[0] - 0.5])
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Attention head")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    st.colorbar(fig, im, ax, "Sink strength (relative to uniform)", fraction=0.030, pad=0.012)
    b = result.zone_boundary()
    if b is not None:
        ax.axvline(b, color="white", linewidth=1.0, linestyle=(0, (4, 2)))
    ax.set_title(f"Sink strength per head and layer  ({_model_name(result)})", loc="center")

    fig.dv_caption = (
        f"Sink strength for every attention head of {_model_name(result)}. Colour is the "
        r"mass the head's strongest image key receives, in multiples of the uniform share "
        "$1/N$. A solid bright column means every head in that block converges on one key "
        "token; scattered cells mean each head selects its own."
        + (" Dashed line: dual-stream to single-stream transition." if b is not None else "")
    )
    _save(fig, path)
    return fig


def fig_text_vs_image_sink(result, path=None, figsize=(7.2, 3.2)):
    """How the raw attention mass of image queries divides between text and image keys."""
    from .metrics import layer_table

    theme = st.THEME
    df = layer_table(result)
    if "text_attention_mass" not in df or float(df["text_attention_mass"].max()) == 0.0:
        raise ValueError("this model has no text keys in the image attention stream "
                         "(PixArt keeps text in cross-attention)")
    agg = df.groupby("layer", as_index=False).mean(numeric_only=True).sort_values("layer")

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(agg["layer"], 1 - agg["text_attention_mass"], color=theme.color("sink"),
            label="image keys")
    ax.plot(agg["layer"], agg["text_attention_mass"], color=theme.color("reference"),
            label="text keys")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Share of attention mass")
    _layer_axis(ax, int(result.meta.get("n_layers", len(agg))))
    st.open_box(ax)
    st.light_grid(ax, "y")
    st.legend_outside(ax)
    b = result.zone_boundary()
    if b is not None:
        ax.axvline(b, color=st.RULE, linewidth=0.8, linestyle=(0, (4, 2)))
    ax.set_title(f"Division of attention between text and image keys  ({_model_name(result)})",
                 loc="center")

    fig.dv_caption = (
        "Raw attention mass placed by image queries on text keys versus image keys, by layer, "
        f"in {_model_name(result)}. Sink strengths reported elsewhere are renormalised over "
        "image keys; this figure gives the fraction of the total mass those numbers describe, "
        "which bounds how much of a head's behaviour an image sink can account for."
    )
    _save(fig, path)
    return fig


# ================================================================== 4. v-star
def fig_vstar_identity(report, result=None, path=None, figsize=(13.5, 8.0),
                       show_chance: bool = True):
    """Characterisation of the shared register direction $v^*$."""
    theme = st.THEME
    v = report.v.numpy()
    sq = v ** 2                       # squared coordinates; sum to 1
    C = v.size
    order = np.argsort(sq)[::-1]

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.subplots_adjust(hspace=0.72, wspace=0.48, top=0.88, bottom=0.18,
                        left=0.065, right=0.98)

    # (a) coordinate magnitudes by channel
    ax = axes[0][0]
    ax.vlines(np.arange(C), 0, sq, color="0.75", linewidth=0.5)
    top = report.top_channels[:6]
    ax.vlines(top, 0, sq[top], color=theme.color("channel"), linewidth=1.4)
    for c in top[:2]:
        ax.annotate(f"channel {c}", xy=(c, sq[c]), xytext=(0, 4), textcoords="offset points",
                    ha="center", fontsize=plt.rcParams["legend.fontsize"] - 1)
    ax.set_ylim(0, float(sq.max()) * 1.22)
    ax.set_xlabel("Channel index")
    ax.set_ylabel("Channel contribution")
    ax.set_title("Coordinates of $v^*$", loc="center")
    st.open_box(ax)
    st.panel_label(ax, "(a)", dx=-0.26, dy=1.12)

    # (b) cumulative share
    ax = axes[0][1]
    k = min(64, C)
    ax.plot(np.arange(1, k + 1), report.cumulative_energy[:k], color=theme.color("channel"))
    ax.axhline(1.0, color=st.RULE, linewidth=0.8)
    ax.set_xscale("log")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Channels ranked by contribution")
    ax.set_ylabel("Cumulative contribution")
    e1 = report.top_channel_energy[0]
    ax.plot([1], [e1], marker="o", markersize=4.5, color=theme.color("channel"), zorder=4)
    ax.annotate(f"channel {report.top_channel}: {e1:.0%}", xy=(1, e1), xytext=(6, -2),
                textcoords="offset points", fontsize=plt.rcParams["legend.fontsize"] - 1)
    ax.set_title("Concentration in few channels", loc="center")
    st.open_box(ax)
    st.panel_label(ax, "(b)", dx=-0.26, dy=1.12)

    # (c) invariance across prompt and seed
    ax = axes[0][2]
    M = report.condition_cos
    if M.size == 0 or M.shape[0] < 2:
        ax.text(0.5, 0.5, "single condition;\nnot tested", ha="center", va="center",
                fontsize=plt.rcParams["legend.fontsize"], transform=ax.transAxes)
        ax.set_xticks([]); ax.set_yticks([])
    else:
        im = ax.imshow(M, cmap=theme.probability_cmap, vmin=0, vmax=1)
        ax.set_xticks(range(len(report.condition_labels)))
        ax.set_yticks(range(len(report.condition_labels)))
        ax.set_xticklabels(report.condition_labels, rotation=90,
                           fontsize=plt.rcParams["legend.fontsize"] - 2)
        ax.set_yticklabels(report.condition_labels,
                           fontsize=plt.rcParams["legend.fontsize"] - 2)
        if M.shape[0] <= 6:
            for i in range(M.shape[0]):
                for j in range(M.shape[1]):
                    ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                            fontsize=plt.rcParams["legend.fontsize"] - 3,
                            color="black" if M[i, j] > 0.55 else "white")
        st.colorbar(fig, im, ax, r"$|\cos|$", fraction=0.05, pad=0.04)
    ax.set_xlabel("Prompt, seed")
    ax.set_ylabel("Prompt, seed")
    ax.set_title(r"Invariance of $v^*$", loc="center")
    st.panel_label(ax, "(c)", dx=-0.26, dy=1.12)

    # (d) alignment of tokens with v*
    ax = axes[1][0]
    bins = np.linspace(0, 1, 41)
    for vals, color, label, hatch in (
        (np.abs(report.cos_registers), theme.color("highnorm"), "high-norm tokens", None),
        (np.abs(report.cos_controls), "0.45", "ordinary tokens", None),
        (np.abs(report.cos_random), "0.80", "random directions", None),
    ):
        if vals.size:
            ax.hist(vals, bins=bins, color=color, label=label, histtype="stepfilled",
                    alpha=0.9, linewidth=0.6, edgecolor="white", hatch=hatch)
    ax.set_yscale("log")
    ax.set_xlabel("Cosine similarity with $v^*$")
    ax.set_ylabel("Number of tokens")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=1,
              fontsize=plt.rcParams["legend.fontsize"] - 2, frameon=False)
    ax.set_title("Alignment with $v^*$", loc="center")
    st.open_box(ax)
    st.panel_label(ax, "(d)", dx=-0.26, dy=1.12)

    # (e) direction vs magnitude as a sink predictor
    ax = axes[1][1]
    sel = report.selection
    if not sel.empty:
        ax.plot(sel["layer"], sel["hit_direction"], color=theme.color("sink"),
                label="ranked by direction")
        ax.plot(sel["layer"], sel["hit_norm"], color=theme.color("highnorm"),
                label="ranked by norm")
        if show_chance:
            ax.plot(sel["layer"], sel["hit_chance"], color=st.RULE, linewidth=0.8,
                    label="random-overlap baseline")
        ax.set_ylim(-0.03, 1.22)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=1,
                  fontsize=plt.rcParams["legend.fontsize"] - 2, frameon=False)
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Fraction of top 1%\nthat are sinks")
    exact = report.meta.get("selection_is_exact", False)
    ax.set_title("Direction vs. magnitude", loc="center")
    st.open_box(ax)
    st.light_grid(ax, "y")
    st.panel_label(ax, "(e)", dx=-0.26, dy=1.12)

    # (f) emergence with depth
    ax = axes[1][2]
    lp = report.layer_profile
    if not lp.empty:
        ax.plot(lp["layer"], lp["register_cos"], color=theme.color("highnorm"),
                label="high-norm tokens")
        ax.plot(lp["layer"], lp["control_cos"], color="0.45", label="ordinary tokens")
        ax.set_ylim(0, 1.22)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=1,
                  fontsize=plt.rcParams["legend.fontsize"] - 2, frameon=False)
        born = lp[lp["register_cos"] > 0.5]["layer"]
        if len(born):
            ax.axvline(float(born.iloc[0]), color=st.RULE, linewidth=0.8, linestyle=(0, (4, 2)))
            ax.annotate(f"layer {int(born.iloc[0])}", xy=(float(born.iloc[0]), 1.16),
                        xytext=(4, 0), textcoords="offset points",
                        fontsize=plt.rcParams["legend.fontsize"] - 2, va="top")
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Cosine similarity with $v^*$")
    ax.set_title("Emergence with depth", loc="center")
    st.open_box(ax)
    st.light_grid(ax, "y")
    st.panel_label(ax, "(f)", dx=-0.26, dy=1.12)

    model = (result.meta.get("model") if result is not None else report.meta.get("model")) or "model"
    fig.suptitle(rf"The shared register direction $v^*$  ({model})", y=0.975)

    inv = ("" if len(report.condition_labels) < 2 else
           f" Pairwise $|\\cos|$ between per-condition fits is "
           f"{_offdiag(report.condition_cos):.3f}.")
    fig.dv_caption = (
        rf"Characterisation of $v^*$, the direction shared by high-norm tokens in {model}. "
        rf"$v^*$ is the leading right-singular vector of the matrix of unit register directions; "
        rf"{report.explained_variance:.0%} of their variance lies on it. "
        r"(a) squared coordinates $(v^*_c)^2$, which sum to one over channels; "
        rf"(b) cumulative share, with channel {report.top_channel} contributing "
        rf"{report.top_channel_energy[0]:.0%}; (c) cosine similarity between $v^*$ fitted "
        rf"independently for each prompt and seed.{inv} (d) alignment of high-norm tokens with "
        r"$v^*$ against matched ordinary tokens from the same layers and against random unit "
        r"directions; (e) fraction of the top 1\% of tokens under each ranking that are also "
        r"the layer's top 1\% attention sinks; (f) the same alignment as a function of depth."
    )
    _save(fig, path)
    return fig


def fig_vstar_channel_bridge(report, result, layer: Optional[int] = None, path=None,
                             figsize=(7.2, 3.0)):
    """Link the massive channel, the high-norm tokens and the sinks at one layer."""
    theme = st.THEME
    recs = [r for r in result.records.values() if r.channel_top_values is not None]
    if layer is not None:
        recs = [r for r in recs if r.layer == layer]
    if not recs:
        raise ValueError("no channel columns captured")
    rec = sorted(recs, key=lambda r: (r.layer, r.step))[len(recs) // 2 if layer is None else -1]
    vals = rec.channel_top_values.float()[:, 0].abs().numpy()
    norms = rec.norms["post_block"].numpy()
    prof = sink_profile(rec).numpy() * rec.n_img
    ch = int(rec.channel_top_ids[0])
    hot = np.argsort(vals)[::-1][:5]

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.subplots_adjust(wspace=0.34, bottom=0.20, top=0.82, left=0.09, right=0.98)

    ax = axes[0]
    ax.scatter(norms / np.median(norms), vals, s=6, color="0.6", linewidth=0, label="image token")
    ax.scatter((norms / np.median(norms))[hot], vals[hot], s=26, color=theme.color("channel"),
               edgecolor="black", linewidth=0.5, zorder=4,
               label=f"largest in channel {ch}")
    ax.set_xscale("log")
    ax.set_xlabel("Norm value (relative to median)")
    ax.set_ylabel(f"Activation in channel {ch}")
    st.legend_outside(ax, fontsize=plt.rcParams["legend.fontsize"] - 1)
    ax.set_title(f"High-norm tokens carry channel {ch}", loc="center")
    st.open_box(ax)
    st.panel_label(ax, "(a)", dx=-0.19, dy=1.05)

    ax = axes[1]
    ax.scatter(vals, prof, s=6, color="0.6", linewidth=0)
    ax.scatter(vals[hot], prof[hot], s=26, color=theme.color("sink"),
               edgecolor="black", linewidth=0.5, zorder=4, label=f"largest in channel {ch}")
    ax.set_yscale("log")
    ax.set_xlabel(f"Activation in channel {ch}")
    ax.set_ylabel("Incoming attention\n(relative to uniform)")
    st.legend_outside(ax, fontsize=plt.rcParams["legend.fontsize"] - 1)
    ax.set_title("Those tokens receive the attention", loc="center")
    st.open_box(ax)
    st.panel_label(ax, "(b)", dx=-0.19, dy=1.05)

    fig.suptitle(f"Channel {ch}, high-norm tokens and attention sinks at layer {rec.layer} "
                 f"({rec.block_name}, {_model_name(result)})", y=0.98)
    fig.dv_caption = (
        rf"Relationship between the leading massive activation channel $c={ch}$, the high-norm "
        rf"tokens and the attention sinks at layer {rec.layer} ({rec.block_name}) of "
        rf"{_model_name(result)}. (a) tokens with a large activation in channel {ch} are exactly "
        r"the tokens with a large residual norm; (b) the same tokens receive the attention mass."
    )
    _save(fig, path)
    return fig


def fig_model_comparison(results: Dict[str, "object"], path=None, figsize=None):
    """The same three statistics across architectures, against relative depth."""
    from .metrics import layer_table

    theme = st.THEME
    keys = list(results)
    rows = [
        ("highnorm", "norm_max_over_median",
         "High-norm tokens\n(largest / median norm)"),
        ("sink", "sink_ratio_headmax",
         "Attention sinks\n(relative to uniform)"),
        ("channel", "chan_max_over_median",
         "Massive channels\n(largest / median peak)"),
    ]
    figsize = figsize or (2.35 * len(keys) + 1.1, 6.2)
    fig, axes = plt.subplots(len(rows), len(keys), figsize=figsize, squeeze=False, sharey="row")
    fig.subplots_adjust(hspace=0.30, wspace=0.14, top=0.90, bottom=0.09, left=0.16, right=0.98)

    for j, k in enumerate(keys):
        res = results[k]
        df = layer_table(res)
        agg = df.groupby("layer", as_index=False).mean(numeric_only=True).sort_values("layer")
        n = max(int(res.meta.get("n_layers", len(agg))), 1)
        depth = agg["layer"] / max(n - 1, 1)
        b = res.zone_boundary()
        for i, (phen, col, ylabel) in enumerate(rows):
            ax = axes[i][j]
            if col in agg:
                ax.plot(depth, agg[col], color=theme.color(phen))
            ax.axhline(1.0, color=st.RULE, linewidth=0.8)
            ax.set_yscale("log")
            ax.set_xlim(0, 1)
            if b is not None:
                ax.axvline(b / max(n - 1, 1), color=st.RULE, linewidth=0.8, linestyle=(0, (4, 2)))
            st.light_grid(ax, "y")
            if j == 0:
                ax.set_ylabel(ylabel)
            if i == 0:
                ax.set_title(f"{k}\n({res.meta.get('n_layers', '?')} layers)", loc="center")
            if i == len(rows) - 1:
                ax.set_xlabel("Relative depth")
    fig.suptitle("Comparison across architectures", y=0.985)
    fig.dv_caption = (
        "Extremal-to-typical ratios against relative depth for each architecture, so networks "
        "with different layer counts are directly comparable. Rows share a y-axis. A dashed "
        "vertical line marks the dual-stream to single-stream transition where the model has one."
    )
    _save(fig, path)
    return fig


def _grid(result, n: int) -> Tuple[int, int]:
    g = result.meta.get("grid")
    if g and g[0] * g[1] == n:
        return int(g[0]), int(g[1])
    side = int(round(math.sqrt(n)))
    return (side, side) if side * side == n else (1, n)


def _offdiag(m: np.ndarray) -> float:
    if m.size == 0 or m.shape[0] < 2:
        return float("nan")
    iu = np.triu_indices(m.shape[0], k=1)
    return float(np.mean(m[iu]))


# ============================================================================
# Figures carried over from the FLUX-only v4 notebook.
# Same analyses, restyled and made model-agnostic.
# ============================================================================
def fig_spatial_token_maps(result, layer: int, prompt_id: int = 0, seed: Optional[int] = None,
                           step: Optional[int] = None, path=None,
                           figsize=(12.8, 4.0)):
    """Generated image, raw incoming attention, and raw token norm.

    This reproduces the visual language of the original v4 flagship figure:
    ``viridis`` for attention and ``magma`` for residual norm.  Set-overlap
    categories deliberately live in :func:`fig_spatial_overlap_maps`, so this
    descriptive figure does not bake in an arbitrary top-k threshold.
    """
    rec = _pick_record(result, layer, prompt_id, seed, step)
    norms = rec.norms.get("post_block")
    prof = sink_profile(rec)
    if norms is None or prof is None:
        raise ValueError(f"layer {layer} has no norms or attention")
    n = rec.n_img
    gh, gw = _grid(result, n)
    image = result.images.get((rec.prompt_id, rec.seed))

    ncols = 3 if image is not None else 2
    fig, axes = plt.subplots(1, ncols, figsize=figsize, squeeze=False,
                             layout="constrained")
    axes = axes[0]
    i = 0
    if image is not None:
        axes[i].imshow(image)
        axes[i].set_title("Generated image")
        axes[i].set_xticks([]); axes[i].set_yticks([])
        i += 1

    # Raw values and colormaps match the attached v4 figure exactly.
    im = axes[i].imshow(prof.numpy().reshape(gh, gw), cmap="viridis")
    axes[i].set_title("Incoming attention (head mean)")
    st.colorbar(fig, im, axes[i], "Incoming attention mass", fraction=0.05, pad=0.03)
    i += 1

    im = axes[i].imshow(norms.numpy().reshape(gh, gw), cmap="magma")
    axes[i].set_title("Post-block token norm")
    st.colorbar(fig, im, axes[i], "Norm Value", fraction=0.05, pad=0.03)

    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel("Patch column")
        ax.set_ylabel("Patch row")
        ax.set_axis_off()
    fig.suptitle(f"Spatial distribution of attention sinks and high-norm tokens — "
                 f"layer {layer} ({rec.block_name}), {_model_name(result)}, "
                 f"prompt {rec.prompt_id}, seed {rec.seed}")
    fig.dv_caption = (
        f"Spatial layout of the two phenomena at layer {layer} ({rec.block_name}) of "
        f"{_model_name(result)}, on the {gh}x{gw} patch grid. The centre panel is raw head-mean "
        "incoming image-to-image attention mass; the right panel is the raw post-block "
        r"residual-stream norm $\|x_t\|_2$. No top-k threshold is applied in this figure."
    )
    _save(fig, path)
    return fig


def fig_spatial_overlap_maps(result, layer: int, prompt_id: int = 0,
                             seed: Optional[int] = None, step: Optional[int] = None,
                             ks: Sequence[int] = (1, 3, 5, 10), path=None,
                             figsize=(12.8, 3.5)):
    """Spatial membership maps for several top-k overlap definitions."""
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    rec = _pick_record(result, layer, prompt_id, seed, step)
    norms = rec.norms.get("post_block")
    prof = sink_profile(rec)
    if norms is None or prof is None:
        raise ValueError(f"layer {layer} has no norms or attention")
    n = rec.n_img
    gh, gw = _grid(result, n)
    colors = ["#d9d9d9", "#4c72b0", "#dd8452", "#c44e52"]
    cmap = ListedColormap(colors)

    fig, axes = plt.subplots(1, len(ks), figsize=figsize, squeeze=False)
    fig.subplots_adjust(left=0.03, right=0.98, top=0.76, bottom=0.22, wspace=0.24)
    for ax, requested_k in zip(axes[0], ks):
        k = max(1, min(int(requested_k), n))
        high = set(torch.topk(norms, k).indices.tolist())
        sink = set(torch.topk(prof, k).indices.tolist())
        cat = np.zeros(n, dtype=int)
        for token in high:
            cat[token] = 1
        for token in sink:
            cat[token] = 3 if cat[token] == 1 else 2
        ax.imshow(cat.reshape(gh, gw), cmap=cmap, vmin=0, vmax=3,
                  interpolation="nearest")
        ax.set_title(f"Top {k}: $|H_k \\cap S_k|={len(high & sink)}$")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel("Patch column"); ax.set_ylabel("Patch row")
        ax.set_axis_off()

    handles = [Patch(color=colors[1], label="high-norm only"),
               Patch(color=colors[2], label="sink only"),
               Patch(color=colors[3], label="overlap")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.01),
               ncol=3, frameon=False)
    fig.suptitle(f"Spatial overlap of high-norm and attention-selected tokens — "
                 f"layer {layer} ({rec.block_name}), {_model_name(result)}")
    fig.dv_caption = (
        f"Spatial overlap at layer {layer} ({rec.block_name}) of {_model_name(result)} for "
        "top-1, top-3, top-5 and top-10 token sets. Blue denotes tokens selected only by "
        "post-block residual norm, orange tokens selected only by incoming attention, and red "
        "tokens selected by both rankings."
    )
    _save(fig, path)
    return fig


def fig_rank_rank(result, layer: int, step: Optional[int] = None, bins: int = 60,
                  path=None, figsize=(3.6, 3.4)):
    """Rank by norm against rank by incoming attention, pooled over heads.

    A ridge on the diagonal means the two orderings agree across the whole
    distribution, not only in the extreme tail that the top-k statistics see.
    """
    theme = st.THEME
    recs = [r for r in result.records.values()
            if r.layer == layer and (step is None or r.step == step)]
    if not recs:
        raise ValueError(f"no record for layer {layer}")
    xs, ys = [], []
    for rec in recs:
        norms = rec.norms.get("post_block")
        if norms is None or rec.incoming_img2img is None:
            continue
        n = rec.n_img
        nr = np.empty(n)
        nr[np.argsort(-norms.numpy())] = np.arange(1, n + 1)
        for h in range(rec.incoming_img2img.shape[0]):
            ar = np.empty(n)
            ar[np.argsort(-rec.incoming_img2img[h].numpy())] = np.arange(1, n + 1)
            xs.append(nr); ys.append(ar)
    x = np.concatenate(xs); y = np.concatenate(ys)
    n = int(max(x.max(), y.max()))

    fig, ax = plt.subplots(figsize=figsize)
    h = ax.hist2d(x, y, bins=[np.linspace(1, n, bins), np.linspace(1, n, bins)],
                  norm=LogNorm(vmin=1), cmap=theme.density_cmap, cmin=1)
    ax.plot([1, n], [1, n], color=st.RULE, linewidth=0.8, linestyle=(0, (4, 2)))
    ax.set_xlabel("Rank by norm  (1 = largest)")
    ax.set_ylabel("Rank by incoming attention")
    st.colorbar(fig, h[3], ax, "Token count", fraction=0.046, pad=0.02)
    ax.set_title(f"Rank agreement, layer {layer} ({recs[0].block_name})", loc="center")
    fig.dv_caption = (
        f"Joint distribution of rank by residual norm and rank by incoming attention at layer "
        f"{layer} ({recs[0].block_name}) of {_model_name(result)}, pooled over heads, prompts and "
        "seeds. The dashed line is equality. Mass on the diagonal indicates the two orderings "
        "agree across the whole distribution rather than only among the few extreme tokens."
    )
    _save(fig, path)
    return fig


def fig_norm_attention_curves(result, layer: int, head: int = 0, prompt_id: int = 0,
                              seed: Optional[int] = None, step: Optional[int] = None,
                              path=None, figsize=(7.2, 3.4)):
    """Incoming attention and residual norm as functions of token index.

    Two stacked panels rather than one plot with two y-scales: a shared vertical
    scale between unrelated quantities invents a correspondence that is not there.
    """
    theme = st.THEME
    rec = _pick_record(result, layer, prompt_id, seed, step)
    n = rec.n_img
    inc = (rec.incoming_img2img[head] * n).numpy()

    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
    fig.subplots_adjust(hspace=0.12)
    axes[0].plot(np.arange(n), inc, color=theme.color("sink"), linewidth=0.8)
    axes[0].axhline(1.0, color=st.RULE, linewidth=0.8)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Incoming attention\n(relative to uniform)")
    st.open_box(axes[0]); st.light_grid(axes[0], "y")
    st.panel_label(axes[0], "(a)", dx=-0.085, dy=1.03)

    for stage, ls in (("post_block", "-"), ("attn_out", (0, (2, 1.6)))):
        norms = rec.norms.get(stage)
        if norms is not None:
            axes[1].plot(np.arange(n), norms.numpy(), color=theme.color("highnorm"),
                         linestyle=ls, linewidth=0.8,
                         label={"post_block": "block output",
                                "attn_out": "attention output"}[stage])
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Norm value")
    axes[1].set_xlabel("Image token index")
    st.legend_outside(axes[1], fontsize=plt.rcParams["legend.fontsize"] - 1)
    st.open_box(axes[1]); st.light_grid(axes[1], "y")
    st.panel_label(axes[1], "(b)", dx=-0.085, dy=1.03)

    fig.suptitle(f"Token profiles, layer {layer} ({rec.block_name}), head {head}  "
                 f"({_model_name(result)}, prompt {rec.prompt_id}, seed {rec.seed})", y=0.98)
    fig.dv_caption = (
        f"Per-token profiles at layer {layer} ({rec.block_name}) of {_model_name(result)}, head "
        f"{head}. (a) incoming image-to-image attention in multiples of the uniform share; "
        "(b) residual-stream norm at the block output and at the attention output. The same few "
        "token indices carry the spikes in both panels."
    )
    _save(fig, path)
    return fig


def fig_norm_vs_attention_scatter(result, layer: int, head: int = 0, prompt_id: int = 0,
                                  seed: Optional[int] = None, step: Optional[int] = None,
                                  k: int = 5, stage: str = "post_block", path=None,
                                  figsize=(3.8, 3.4)):
    """Norm against incoming attention, one point per token, categorised by top-k membership."""
    theme = st.THEME
    rec = _pick_record(result, layer, prompt_id, seed, step)
    norms = rec.norms.get(stage)
    if norms is None:
        raise ValueError(f"stage {stage!r} not captured at this layer")
    n = rec.n_img
    inc = (rec.incoming_img2img[head] * n).numpy()
    kk = max(1, min(k, n))
    high = set(torch.topk(norms, kk).indices.tolist())
    sink = set(torch.topk(rec.incoming_img2img[head], kk).indices.tolist())

    groups = [
        ("ordinary", "0.72", 5, [t for t in range(n) if t not in high and t not in sink]),
        ("high-norm only", theme.color("highnorm"), 20, sorted(high - sink)),
        ("sink only", theme.color("sink"), 20, sorted(sink - high)),
        ("both", theme.color("reference"), 34, sorted(high & sink)),
    ]
    fig, ax = plt.subplots(figsize=figsize)
    x = norms.numpy()
    for label, color, size, idx in groups:
        if idx:
            ax.scatter(x[idx], inc[idx], s=size, color=color, linewidth=0.4,
                       edgecolor="white" if size > 8 else "none", label=label, zorder=3)
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel("Norm value")
    ax.set_ylabel("Incoming attention\n(relative to uniform)")
    st.legend_outside(ax, fontsize=plt.rcParams["legend.fontsize"] - 1)
    st.open_box(ax); st.light_grid(ax, "y")
    ax.set_title(f"Layer {layer} ({rec.block_name}), head {head}", loc="center")
    fig.dv_caption = (
        f"Residual norm against incoming attention for every image token at layer {layer} "
        f"({rec.block_name}) of {_model_name(result)}, head {head}. Points are coloured by "
        f"membership of the top {kk} by norm, the top {kk} by attention, or both."
    )
    _save(fig, path)
    return fig


def fig_overlap_by_stage(result, ks: Sequence[int] = (1, 5, 10), path=None,
                         figsize=(7.2, 3.2), show_chance: bool = True):
    """Overlap between high-norm tokens and sinks, per residual-stream stage."""
    from .metrics import stage_overlap_table

    theme = st.THEME
    df = stage_overlap_table(result, ks)
    if df.empty:
        raise ValueError("no overlap data")
    agg = df.groupby(["stage", "k"], as_index=False).agg(
        jaccard=("jaccard", "mean"),
        lo=("jaccard", lambda v: np.percentile(v, 25)),
        hi=("jaccard", lambda v: np.percentile(v, 75)),
        chance=("chance", "mean"))
    n_layers = df.groupby("stage")["layer"].nunique()
    stages = [s for s in ("pre_block", "attn_out", "post_attn_residual", "post_block")
              if s in set(agg["stage"])]
    labels = {"pre_block": "block input", "attn_out": "attention output",
              "post_attn_residual": "after attention", "post_block": "block output"}
    width = 0.8 / max(len(ks), 1)

    fig, ax = plt.subplots(figsize=figsize)
    for i, k in enumerate(sorted(agg["k"].unique())):
        sub = agg[agg["k"] == k].set_index("stage").reindex(stages)
        pos = np.arange(len(stages)) + (i - (len(ks) - 1) / 2) * width
        err = np.vstack([(sub["jaccard"] - sub["lo"]).clip(lower=0),
                         (sub["hi"] - sub["jaccard"]).clip(lower=0)])
        ax.bar(pos, sub["jaccard"], width * 0.92, yerr=err, capsize=2,
               error_kw=dict(elinewidth=0.8, capthick=0.8),
               color=plt.rcParams["axes.prop_cycle"].by_key()["color"][i % 4],
               label=f"top {int(k)}", edgecolor="white", linewidth=0.6)
        if show_chance:
            ax.hlines(sub["chance"], pos - width * 0.46, pos + width * 0.46,
                      color=st.RULE, linewidth=1.0, zorder=5)
    ax.set_xticks(np.arange(len(stages)))
    ax.set_xticklabels([f"{labels[s]}\n({int(n_layers[s])} layers)" for s in stages])
    ax.set_xlabel("Residual-stream stage")
    ax.set_ylabel("Overlap  (Jaccard)")
    ax.set_ylim(bottom=0)
    st.legend_outside(ax)
    st.open_box(ax); st.light_grid(ax, "y")
    ax.set_title(f"Overlap between high-norm tokens and attention sinks  ({_model_name(result)})",
                 loc="center")
    fig.dv_caption = (
        "Jaccard overlap between the top-k image tokens by residual norm and the top-k by "
        f"incoming attention in {_model_name(result)}, measured at each residual-stream stage. "
        + ("Grey rules mark the random-overlap baseline for independent sets of the same size. "
           if show_chance else "")
        + "Error bars span the 25th to 75th percentiles over layers, prompts and seeds."
    )
    _save(fig, path)
    return fig


def fig_spearman_heatmap(result, step: Optional[int] = None, path=None, figsize=(7.2, 3.4)):
    """Rank correlation between token norm and incoming attention, per layer and head."""
    from .metrics import head_table

    df = head_table(result)
    if step is not None:
        df = df[df["step"] == step]
    if "spearman_norm_vs_attention" not in df:
        raise ValueError("no rank correlations available")
    piv = df.pivot_table(index="head_id", columns="layer",
                         values="spearman_norm_vs_attention", aggfunc="mean")
    vals = piv.to_numpy()
    vmax = float(np.nanmax(np.abs(vals))) or 1.0

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(vals, aspect="auto", cmap=st.diverging_cmap(), vmin=-vmax, vmax=vmax,
                   origin="lower",
                   extent=[piv.columns.min() - 0.5, piv.columns.max() + 0.5,
                           -0.5, piv.shape[0] - 0.5])
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Attention head")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    st.colorbar(fig, im, ax, "Rank correlation", fraction=0.030, pad=0.012)
    b = result.zone_boundary()
    if b is not None:
        ax.axvline(b, color="black", linewidth=0.8, linestyle=(0, (4, 2)))
    ax.set_title(f"Rank correlation between norm and incoming attention  "
                 f"({_model_name(result)})", loc="center")
    fig.dv_caption = (
        "Spearman rank correlation between residual norm and incoming attention across all image "
        f"tokens, for every head and layer of {_model_name(result)}. Positive values mean loud "
        "tokens receive more attention. Zero is the neutral midpoint of the colour scale."
    )
    _save(fig, path)
    return fig


def fig_sink_rank_histogram(result, path=None, figsize=(3.8, 3.2)):
    """How high in the norm ordering each head's sink token sits."""
    from .metrics import head_table

    theme = st.THEME
    df = head_table(result)
    if df.empty:
        raise ValueError("no head data")
    ranks = df["sink_token_norm_rank"].to_numpy()
    ranks = ranks[ranks > 0]

    fig, ax = plt.subplots(figsize=figsize)
    bins = np.geomspace(1, max(ranks.max(), 2), 40)
    ax.hist(ranks, bins=bins, color=theme.color("sink"), edgecolor="white", linewidth=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("Rank of the sink token by norm  (1 = largest)")
    ax.set_ylabel("Number of heads")
    st.open_box(ax); st.light_grid(ax, "y")
    frac = float(np.mean(ranks <= 10))
    ax.axvline(10, color=st.RULE, linewidth=0.8, linestyle=(0, (4, 2)))
    ax.annotate(f"{frac:.0%} within the\ntop 10 by norm", xy=(10, ax.get_ylim()[1]),
                xytext=(6, -6), textcoords="offset points", va="top",
                fontsize=plt.rcParams["legend.fontsize"] - 1)
    ax.set_title("Norm rank of each head's sink token", loc="center")
    fig.dv_caption = (
        "Distribution over all heads, layers, prompts and seeds of where each head's strongest "
        f"image key sits in that layer's norm ordering, in {_model_name(result)}. Mass near rank 1 "
        "means heads sink on the highest-norm tokens; a uniform spread means the two are unrelated."
    )
    _save(fig, path)
    return fig


def fig_token_trajectories(result, n_tokens: int = 6, prompt_id: int = 0,
                           seed: Optional[int] = None, step: Optional[int] = None,
                           path=None, figsize=(7.2, 3.4)):
    """Norm of individual tokens as a function of depth."""
    theme = st.THEME
    recs = sorted([r for r in result.records.values()
                   if r.prompt_id == prompt_id and (seed is None or r.seed == seed)
                   and (step is None or r.step == step)], key=lambda r: r.layer)
    if not recs:
        raise ValueError("no records")
    seed_used = recs[0].seed
    recs = [r for r in recs if r.seed == seed_used and r.step == recs[-1].step]
    layers = [r.layer for r in recs]
    from .metrics import highnorm_mask

    peak = max(recs, key=lambda r: float(r.norms["post_block"].max()))
    ratio = result.cfg.highnorm_ratio
    is_high = highnorm_mask(peak.norms["post_block"], ratio)
    order = torch.argsort(peak.norms["post_block"], descending=True).tolist()
    top = order[:min(n_tokens, peak.n_img)]
    med = np.array([float(r.norms["post_block"].median()) for r in recs])

    # Colour by what the token turns out to be, not by its index: an index is a
    # nominal label and a colour ramp over it would encode nothing.
    fig, ax = plt.subplots(figsize=figsize)
    drawn_high = drawn_ord = False
    for t in top:
        y = [float(r.norms["post_block"][t]) for r in recs]
        high = bool(is_high[t])
        ax.plot(layers, y, linewidth=1.0,
                color=theme.color("highnorm") if high else "0.62",
                label=("high-norm token" if high and not drawn_high else
                       "ordinary token" if not high and not drawn_ord else None),
                zorder=3 if high else 2)
        drawn_high |= high
        drawn_ord |= not high
        if high:
            ax.annotate(f"token {t}", xy=(layers[-1], y[-1]), xytext=(4, 0),
                        textcoords="offset points", va="center",
                        fontsize=plt.rcParams["legend.fontsize"] - 1,
                        color=theme.color("highnorm"))
    ax.plot(layers, med, color="black", linewidth=1.2, linestyle=(0, (4, 2)),
            label="median token", zorder=4)
    ax.set_yscale("log")
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Norm value")
    st.legend_outside(ax, fontsize=plt.rcParams["legend.fontsize"] - 1)
    st.open_box(ax); st.light_grid(ax, "y")
    b = result.zone_boundary()
    if b is not None:
        ax.axvline(b, color=st.RULE, linewidth=0.8, linestyle=(0, (4, 2)))
    ax.set_title(f"Norm trajectories of individual tokens  ({_model_name(result)}, "
                 f"prompt {prompt_id}, seed {seed_used})", loc="center")
    fig.dv_caption = (
        f"Residual-stream norm of the {len(top)} highest-norm image tokens as a function of depth "
        f"in {_model_name(result)}, against the median token. Tokens are selected at the layer "
        "where the norm peaks and then tracked across every layer, so the figure shows the layer "
        "at which they separate from the bulk and whether they return to it. Colour marks whether "
        f"a token exceeds {ratio:g} times the layer median at the peak layer."
    )
    _save(fig, path)
    return fig


def fig_percentile_sensitivity(result, stage: str = "post_block", path=None,
                               figsize=(7.2, 3.2), show_chance: bool = True):
    """Is the norm/sink overlap an artefact of where the threshold is drawn?"""
    from .metrics import percentile_sweep_table

    theme = st.THEME
    df = percentile_sweep_table(result, stages=(stage,))
    if df.empty:
        raise ValueError("no percentile sweep available")
    agg = df.groupby(["kind", "percentile"], as_index=False).agg(
        jaccard=("jaccard", "mean"), chance=("chance", "mean"))
    fig, ax = plt.subplots(figsize=figsize)
    for i, (kind, g) in enumerate(agg.groupby("kind")):
        ax.plot(g["percentile"], g["jaccard"], marker="o", markersize=3.5,
                color=[theme.color("highnorm"), theme.color("sink"), theme.color("channel")][i % 3],
                label=f"{kind} blocks")
    if show_chance:
        ch = agg.groupby("percentile", as_index=False)["chance"].mean()
        ax.plot(ch["percentile"], ch["chance"], color=st.RULE, linewidth=0.8,
                linestyle=(0, (4, 2)), label="random-overlap baseline")
    ax.set_xlabel("Percentile threshold defining high-norm and sink")
    ax.set_ylabel("Overlap  (Jaccard)")
    st.legend_outside(ax)
    st.open_box(ax); st.light_grid(ax, "y")
    ax.set_title(f"Sensitivity of the overlap to the percentile threshold  "
                 f"({_model_name(result)}, {stage})", loc="center")
    fig.dv_caption = (
        "Overlap between the high-norm set and the sink set as the percentile defining both is "
        f"swept, for {_model_name(result)} measured at the {stage} stage. A curve that stays well "
        + ("above the random-overlap baseline across the sweep shows the coupling is not an "
           "artefact of one threshold." if show_chance else
           "stable across the sweep shows the conclusion is not an artefact of one threshold.")
    )
    _save(fig, path)
    return fig


def fig_norm_ratio_substrate(result, percentile: float = 97.7, path=None, figsize=(7.2, 3.0)):
    """Is there an outlier population at all, or only a percentile that always selects some?

    An empirical percentile selects a fixed fraction of tokens by construction, so
    counts prove nothing. The test is whether the threshold-to-median norm ratio
    exceeds what a Gaussian residual would give.
    """
    from .metrics import percentile_sweep_table

    theme = st.THEME
    df = percentile_sweep_table(result, percentiles=(percentile,))
    if df.empty:
        raise ValueError("no sweep available")
    agg = df.groupby(["layer", "stage"], as_index=False).agg(
        ratio=("norm_ratio", "mean"), null=("gaussian_null_ratio", "mean"))
    piv = agg.pivot(index="stage", columns="layer", values="ratio")
    null = float(agg["null"].mean())

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(piv.to_numpy(), aspect="auto", cmap=theme.density_cmap,
                   norm=LogNorm(vmin=max(np.nanmin(piv.to_numpy()), 1e-2),
                                vmax=max(np.nanmax(piv.to_numpy()), 1.1)),
                   extent=[piv.columns.min() - 0.5, piv.columns.max() + 0.5, -0.5,
                           piv.shape[0] - 0.5], origin="lower")
    ax.set_yticks(range(piv.shape[0]))
    ax.set_yticklabels([{"attn_out": "attention output",
                         "post_block": "block output"}.get(s, s) for s in piv.index])
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Residual-stream stage")
    st.colorbar(fig, im, ax, "Threshold / median norm", fraction=0.035, pad=0.012)
    ax.set_title(f"Outlier population test at the {percentile:g}th percentile  "
                 f"(Gaussian residual would give {null:.2f})", loc="center")
    fig.dv_caption = (
        f"Ratio of the {percentile:g}th-percentile norm to the median norm, per layer and stage, in "
        f"{_model_name(result)}. Under Gaussian residual components the norm follows a chi "
        f"distribution and this ratio would be about {null:.3f}; values far above it indicate a "
        "genuine outlier population rather than the tail that any percentile rule selects."
    )
    _save(fig, path)
    return fig


def fig_head_sink_membership(result, path=None, figsize=(7.2, 2.8)):
    """Fraction of heads whose strongest image key is a high-norm token, by layer."""
    from .metrics import head_sink_membership

    theme = st.THEME
    df = head_sink_membership(result)
    if df.empty:
        raise ValueError("no membership data")
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(df["layer"], df["fraction"], width=0.85, color=theme.color("sink"),
           edgecolor="white", linewidth=0.4)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Fraction of heads")
    _layer_axis(ax, int(result.meta.get("n_layers", len(df))))
    st.open_box(ax); st.light_grid(ax, "y")
    b = result.zone_boundary()
    if b is not None:
        ax.axvline(b, color=st.RULE, linewidth=0.8, linestyle=(0, (4, 2)))
    ax.set_title(f"Heads whose strongest image key is a high-norm token  ({_model_name(result)})",
                 loc="center")
    fig.dv_caption = (
        "Fraction of attention heads whose strongest image key is a high-norm token, by layer, in "
        f"{_model_name(result)}. High-norm means a residual norm above "
        f"{result.cfg.highnorm_ratio:g} times the layer median. This is the head-level form of the "
        "coupling: it counts heads, not set overlap."
    )
    _save(fig, path)
    return fig


def fig_text_attention_per_head(result, layers: Optional[Sequence[int]] = None, path=None,
                                figsize=(7.2, 3.0)):
    """Per-head competition between the text keys and the image sink."""
    from .metrics import head_table

    theme = st.THEME
    df = head_table(result)
    if df.empty or float(df["text_mass"].max()) == 0.0:
        raise ValueError("this model has no text keys in the image attention stream")
    if layers is None:
        cand = sorted(df["layer"].unique())
        layers = [cand[len(cand) // 4], cand[len(cand) // 2], cand[-1]]
    sub = df[df["layer"].isin(layers)]

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.subplots_adjust(wspace=0.34, bottom=0.20, top=0.82, left=0.09, right=0.98)
    colors = [theme.color("highnorm"), theme.color("sink"), theme.color("channel")]
    for i, l in enumerate(layers):
        g = sub[sub["layer"] == l].groupby("head_id", as_index=False)["text_mass"].mean()
        axes[0].plot(g["head_id"], g["text_mass"], marker="o", markersize=3,
                     color=colors[i % 3], label=f"layer {l}")
    axes[0].set_ylim(0, 1)
    axes[0].set_xlabel("Attention head")
    axes[0].set_ylabel("Share of attention on text keys")
    st.legend_outside(axes[0], fontsize=plt.rcParams["legend.fontsize"] - 1)
    st.open_box(axes[0]); st.light_grid(axes[0], "y")
    st.panel_label(axes[0], "(a)", dx=-0.19, dy=1.05)

    rate = df.groupby("layer", as_index=False)["sink_is_text"].mean()
    axes[1].bar(rate["layer"], rate["sink_is_text"], width=0.85, color=theme.color("reference"),
                edgecolor="white", linewidth=0.4)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_xlabel("Layer index")
    axes[1].set_ylabel("Fraction of heads")
    st.open_box(axes[1]); st.light_grid(axes[1], "y")
    axes[1].set_title("Strongest key is a text token", loc="center",
                      fontsize=plt.rcParams["axes.labelsize"])
    st.panel_label(axes[1], "(b)", dx=-0.19, dy=1.05)

    fig.suptitle(f"Competition between text keys and image keys  ({_model_name(result)})", y=0.98)
    fig.dv_caption = (
        f"Competition between text keys and image keys in {_model_name(result)}. (a) share of each "
        "head's raw attention mass that lands on text keys, at three depths. (b) fraction of heads "
        "per layer whose single strongest key over the whole sequence is a text token rather than "
        "an image token; where this is high, the image sink describes only the residual mass."
    )
    _save(fig, path)
    return fig


def fig_channel_landscape(result, layer: Optional[int] = None, n_label: int = 4, path=None,
                          figsize=(7.2, 2.8)):
    """Mean absolute activation per channel: the massive channels against the bulk."""
    theme = st.THEME
    recs = [r for r in result.records.values() if r.channel_mean_abs is not None]
    if layer is not None:
        recs = [r for r in recs if r.layer == layer]
    if not recs:
        raise ValueError("no channel statistics captured")
    prof = torch.stack([r.channel_mean_abs for r in recs]).mean(0).numpy()
    peak = torch.stack([r.channel_absmax for r in recs]).mean(0).numpy()
    order = np.argsort(peak)[::-1][:n_label]

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(np.arange(prof.size), prof, color="0.55", linewidth=0.5, label="mean over tokens")
    ax.plot(np.arange(peak.size), peak, color=theme.color("channel"), linewidth=0.5,
            label="peak over tokens")
    for c in order:
        ax.annotate(f"channel {int(c)}", xy=(c, peak[c]), xytext=(0, 5),
                    textcoords="offset points", ha="center",
                    fontsize=plt.rcParams["legend.fontsize"] - 1)
    ax.set_yscale("log")
    ax.set_xlabel("Channel index")
    ax.set_ylabel("Activation magnitude")
    st.legend_outside(ax, fontsize=plt.rcParams["legend.fontsize"] - 1)
    st.open_box(ax); st.light_grid(ax, "y")
    where = "all layers" if layer is None else f"layer {layer}"
    ax.set_title(f"Activation magnitude per channel, {where}  ({_model_name(result)})", loc="center")
    fig.dv_caption = (
        f"Per-channel activation magnitude in {_model_name(result)}, averaged over {where}, prompts "
        "and seeds: the mean over image tokens and the peak over image tokens. A handful of "
        "channels stand orders of magnitude above the bulk; these are the massive activation "
        "channels the register direction is concentrated in."
    )
    _save(fig, path)
    return fig


def fig_vstar_projection_trajectory(report, result=None, path=None, figsize=(7.2, 3.0)):
    """Magnitude of the projection onto the register direction, by depth."""
    theme = st.THEME
    lp = report.layer_profile
    if lp.empty:
        raise ValueError("no layer profile in the report")
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(lp["layer"], lp["register_proj"], color=theme.color("highnorm"),
            label="high-norm tokens")
    ax.plot(lp["layer"], lp["control_proj"], color="0.45", label="ordinary tokens")
    ax.set_yscale("log")
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Projection onto $v^*$")
    st.legend_outside(ax)
    st.open_box(ax); st.light_grid(ax, "y")
    if result is not None:
        b = result.zone_boundary()
        if b is not None:
            ax.axvline(b, color=st.RULE, linewidth=0.8, linestyle=(0, (4, 2)))
    model = (result.meta.get("model") if result is not None else report.meta.get("model")) or "model"
    ax.set_title(f"Emergence of the register direction with depth  ({model})", loc="center")
    fig.dv_caption = (
        rf"Absolute projection of image tokens onto the register direction $v^*$ as a function of "
        rf"depth in {model}, for high-norm tokens and for matched ordinary tokens from the same "
        r"layers. The layer at which the two curves separate is where $v^*$ is written into the "
        "residual stream; the ordinary-token curve is the baseline any direction would produce."
    )
    _save(fig, path)
    return fig


def fig_register_selection(result, layer: Optional[int] = None, path=None,
                           figsize=(7.2, 2.9), show_chance: bool = True):
    """Which patches become high-norm, and how stable that choice is."""
    from itertools import combinations

    from .metrics import register_positions

    theme = st.THEME
    df = register_positions(result, layer=layer)
    if df.empty:
        raise ValueError("no high-norm tokens found at any layer")
    layer = int(df["layer"].iloc[0])
    df = df[df["step"] == df["step"].max()]
    n = max(r.n_img for r in result.records.values())
    gh, gw = _grid(result, n)

    heat = np.zeros((gh, gw))
    for ids in df["ids"]:
        for t in ids:
            heat[int(t) // gw, int(t) % gw] += 1

    def jac(a, b):
        a, b = set(a), set(b)
        return len(a & b) / len(a | b) if (a | b) else np.nan

    same_prompt, diff_prompt = [], []
    rows = list(df.itertuples())
    for r1, r2 in combinations(rows, 2):
        (same_prompt if r1.prompt_id == r2.prompt_id else diff_prompt).append(jac(r1.ids, r2.ids))
    sizes = df["ids"].apply(len)
    m = max(float(sizes.median()), 1.0)
    chance = (m * m / n) / max(2 * m - m * m / n, 1e-9)

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.subplots_adjust(wspace=0.42, bottom=0.24, top=0.76, left=0.07, right=0.97)
    im = axes[0].imshow(np.ma.masked_less(heat, 1), cmap=theme.density_cmap,
                        vmin=1, vmax=max(heat.max(), 1))
    axes[0].set_xticks([]); axes[0].set_yticks([])
    axes[0].set_xlabel("Patch column")
    axes[0].set_ylabel(f"Patch row  ({gh}x{gw} grid)")
    cb = st.colorbar(fig, im, axes[0], "Times selected", fraction=0.046, pad=0.02)
    cb.locator = MaxNLocator(integer=True)
    cb.update_ticks()
    for sp in axes[0].spines.values():
        sp.set_linewidth(0.8)
    axes[0].set_title("Selected patches", loc="center",
                      fontsize=plt.rcParams["axes.labelsize"])
    st.panel_label(axes[0], "(a)", dx=-0.16, dy=1.14)

    names = ["same prompt,\ndifferent seed", "different\nprompt"]
    vals = [np.nanmean(same_prompt) if same_prompt else np.nan,
            np.nanmean(diff_prompt) if diff_prompt else np.nan]
    colors = [theme.color("highnorm"), theme.color("sink")]
    if show_chance:
        names.append("random-overlap\nbaseline")
        vals.append(chance)
        colors.append("0.6")
    bars = axes[1].bar(names, vals, width=0.6,
                       color=colors,
                       edgecolor="white", linewidth=0.6)
    for rect, v in zip(bars, vals):
        if np.isfinite(v):
            axes[1].annotate(f"{v:.3f}", xy=(rect.get_x() + rect.get_width() / 2, v),
                             xytext=(0, 3), textcoords="offset points", ha="center",
                             fontsize=plt.rcParams["legend.fontsize"] - 1)
    axes[1].set_ylabel("Overlap  (Jaccard)")
    axes[1].set_xlabel("Comparison")
    axes[1].set_ylim(0, max([v for v in vals if np.isfinite(v)] + [1e-3]) * 1.35)
    st.open_box(axes[1]); st.light_grid(axes[1], "y")
    axes[1].set_title("Stability of the choice", loc="center",
                      fontsize=plt.rcParams["axes.labelsize"])
    st.panel_label(axes[1], "(b)", dx=-0.16, dy=1.14)

    fig.suptitle(f"Selection of the high-norm tokens, layer {layer}  ({_model_name(result)})",
                 y=0.98)
    fig.dv_caption = (
        f"Which image patches become high-norm at layer {layer} of {_model_name(result)}. "
        "(a) how often each patch is selected, pooled over prompts and seeds. (b) overlap between "
        "the selected sets across seeds of one prompt and across prompts, against the expected "
        "overlap for independent sets of the same size. High overlap in both means the positions "
        "are fixed by the architecture; high across seeds only means they track image content; "
        "chance-level means the winners are set by fluctuations."
    )
    _save(fig, path)
    return fig


def fig_qk_geometry(result, step: Optional[int] = None, path=None, figsize=(11.8, 3.8)):
    """Why natural register keys are preferred by image queries."""
    from .metrics import qk_geometry_table

    df = qk_geometry_table(result)
    if df.empty:
        raise ValueError("no query-key geometry in this sweep; re-run capture with the updated code")
    if step is None:
        step = int(df["step"].max())
    df = df[df["step"] == step]
    agg = df.groupby("layer", as_index=False).agg(
        median_rank=("register_key_rank", "median"),
        top10_rate=("register_key_topk", "mean"),
        register_cos=("register_key_cosine", "mean"),
        ordinary_cos=("ordinary_key_cosine", "mean"))

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.80, bottom=0.24, wspace=0.38)
    axes[0].plot(agg["layer"], agg["median_rank"], color=st.SERIES["sink"])
    axes[0].set_yscale("log"); axes[0].set_xlabel("Layer index")
    axes[0].set_ylabel("Median register-key rank\n(1 = most aligned)")
    axes[0].set_title("(a) Key rank", loc="left")

    axes[1].plot(agg["layer"], agg["top10_rate"], color=st.SERIES["sink"])
    axes[1].set_ylim(0, 1.05); axes[1].set_xlabel("Layer index")
    axes[1].set_ylabel("Fraction of heads with\nregister key in top 10")
    axes[1].set_title("(b) Top-10 frequency", loc="left")

    axes[2].plot(agg["layer"], agg["register_cos"], color=st.SERIES["sink"], label="register key")
    axes[2].plot(agg["layer"], agg["ordinary_cos"], color="0.55", label="median image key")
    axes[2].set_xlabel("Layer index"); axes[2].set_ylabel("Cosine with mean image query")
    axes[2].set_title("(c) Query-key alignment", loc="left")
    for ax in axes:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
        st.open_box(ax); st.light_grid(ax, "y")
    handles, labels = axes[2].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.015),
               ncol=2, frameon=False, fontsize=plt.rcParams["legend.fontsize"] - 1)
    fig.suptitle(f"Natural register keys align with image queries  ({_model_name(result)}, step {step})")
    fig.dv_caption = (
        f"Observational query-key geometry in {_model_name(result)} at denoising step {step}. "
        "The register candidate is the highest post-block-norm image token. Per-head cosine "
        "similarity is computed between each natural image key and that head's mean image query "
        "after QK normalisation and rotary embedding. This explains the observed preference but "
        "does not establish that the key component induced by v* is sufficient."
    )
    _save(fig, path)
    return fig


def fig_sparse_lifecycle(result, report, step: Optional[int] = None, path=None,
                         figsize=(7.2, 6.4)):
    """Joint depth trajectory of channel, direction, norm, and sink signatures."""
    from .metrics import layer_table

    df = layer_table(result)
    if step is None:
        step = int(df["step"].max())
    df = df[df["step"] == step]
    layer = df.groupby("layer", as_index=False).mean(numeric_only=True).sort_values("layer")
    lp = report.layer_profile
    if "step" in lp:
        lp = lp[lp["step"] == step]
    lp = lp.groupby("layer", as_index=False).mean(numeric_only=True).sort_values("layer")

    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True, layout="constrained")
    rows = [
        ("chan_max_over_median", "Dominant channel", "Peak channel activation\n(relative to median)"),
        (None, "Shared register axis", r"Register $|x^\top v^*|$"),
        ("norm_max_over_median", "High-norm register", "Largest token norm\n(relative to median)"),
        ("sink_ratio_headmean", "Attention sink", "Incoming attention\n(relative to uniform)"),
    ]
    for i, (column, title, ylabel) in enumerate(rows):
        ax = axes[i]
        if column is None:
            ax.plot(lp["layer"], lp["register_proj"], color=st.SERIES["highnorm"])
        else:
            color = st.SERIES[["channel", "highnorm", "sink"][[0, 2, 3].index(i)]]
            ax.plot(layer["layer"], layer[column], color=color)
        ax.set_yscale("log"); ax.set_ylabel(ylabel); ax.set_title(f"({'abcd'[i]}) {title}", loc="left")
        st.open_box(ax); st.light_grid(ax, "y")
        b = result.zone_boundary()
        if b is not None:
            ax.axvline(b, color=st.RULE, linewidth=0.8, linestyle=(0, (4, 2)))
    _layer_axis(axes[-1], int(result.meta.get("n_layers", len(layer))))
    fig.suptitle(f"Lifecycle of the sparse signatures  ({_model_name(result)}, step {step})")
    fig.dv_caption = (
        f"Joint layer trajectory in {_model_name(result)} at denoising step {step}: the largest "
        "channel peak relative to the median channel, register projection onto the fitted shared "
        "axis v*, largest token norm relative to the median token, and head-mean sink strength "
        "relative to uniform attention. Co-emergence is descriptive; the causal roles are tested "
        "by separate interventions."
    )
    _save(fig, path)
    return fig


def _pick_record(result, layer: int, prompt_id: int, seed: Optional[int], step: Optional[int]):
    recs = [r for r in result.records.values()
            if r.layer == layer and r.prompt_id == prompt_id
            and (seed is None or r.seed == seed) and (step is None or r.step == step)]
    if not recs:
        raise ValueError(f"no record for layer {layer}, prompt {prompt_id}")
    return sorted(recs, key=lambda r: (r.seed, r.step))[-1]


# ============================================================================
# Causal interventions
# ============================================================================
def _cond_order(present: Sequence[str]) -> List[str]:
    from .interventions import ABLATION_SUITE, MAGNITUDE_SWEEP_SUITE

    canonical = ABLATION_SUITE + MAGNITUDE_SWEEP_SUITE
    return [c for c in canonical if c in set(present)] + \
           [c for c in present if c not in set(canonical)]


def _cond_color(key: str):
    theme = st.THEME
    if key.startswith("magnitude"):
        base = theme.color("highnorm")
    elif key.startswith("direction_transfer"):
        base = theme.color("channel")
    elif key.startswith("direction") or key.startswith("full"):
        base = theme.color("sink")
    else:
        base = "0.55"
    if key.endswith(("_matched", "_random")):
        return _lighten(base, 0.45 if key.endswith("_matched") else 0.68)
    return base


def _lighten(hex_color: str, amount: float) -> str:
    import matplotlib.colors as mc

    r, g, b = mc.to_rgb(hex_color)
    return mc.to_hex((r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount))


def fig_sink_persistence(results, conditions: Optional[Sequence[str]] = None, path=None,
                         figsize=None):
    """Does the sink stay on the same token when its magnitude, or its direction, is removed?"""
    from .interventions import ABLATION_SUITE, condition_tick, sink_persistence

    df = sink_persistence(results)
    if df.empty:
        raise ValueError("no intervention data")
    wanted = set(conditions) if conditions else set(ABLATION_SUITE)
    df = df[df["condition"].isin(wanted)]
    conds = [c for c in _cond_order(sorted(set(df["condition"]))) if c != "baseline"]
    layers = sorted(set(df["layer"]))
    figsize = figsize or (2.1 * len(layers) + 1.6, 3.9)

    fig, axes = plt.subplots(1, len(layers), figsize=figsize, squeeze=False, sharey=True)
    fig.subplots_adjust(wspace=0.10, bottom=0.36, top=0.86, left=0.14, right=0.98)
    for ax, l in zip(axes[0], layers):
        sub = df[df["layer"] == l].set_index("condition").reindex(conds)
        ax.bar(np.arange(len(conds)), sub["sink_unchanged"], 0.7,
               color=[_cond_color(c) for c in conds], edgecolor="white", linewidth=0.5)
        ax.set_xticks(np.arange(len(conds)))
        ax.set_xticklabels([condition_tick(c) for c in conds], rotation=35, ha="right",
                           rotation_mode="anchor")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"layer {l}", loc="center", fontsize=plt.rcParams["axes.labelsize"])
        st.open_box(ax); st.light_grid(ax, "y")
    axes[0][0].set_ylabel("Fraction of heads with an\nunchanged sink token")
    fig.supxlabel("Condition", y=0.012)
    fig.suptitle("Persistence of the sink token under ablation", y=0.98)
    fig.dv_caption = (
        "Fraction of attention heads whose strongest image key is unchanged after the edit, among "
        "the heads whose unedited sink was one of the edited tokens, at each observed layer. A high "
        "value under magnitude ablation means the sink does not follow the norm; a low value under "
        "direction ablation means it follows the direction. The matched-norm and random-token "
        "controls edit the same number of tokens at the same denoising steps and should leave the "
        "sink in place."
    )
    _save(fig, path)
    return fig


def fig_head_agreement(results, path=None, figsize=(7.2, 3.0)):
    """Does the block-wide consensus on one sink token survive the edit?"""
    from .interventions import condition_tick, head_agreement

    df = head_agreement(results)
    if df.empty:
        raise ValueError("no intervention data")
    conds = _cond_order(sorted(set(df["condition"])))
    layers = sorted(set(df["layer"]))

    fig, ax = plt.subplots(figsize=figsize)
    for i, c in enumerate(conds):
        sub = df[df["condition"] == c].sort_values("layer")
        ax.plot(sub["layer"], sub["agreement"], marker="o", markersize=3.5,
                color=_cond_color(c), label=condition_tick(c))
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Fraction of heads sharing\nthe modal sink token")
    ax.set_xticks(layers)
    st.open_box(ax); st.light_grid(ax, "y")
    st.legend_outside(ax, fontsize=plt.rcParams["legend.fontsize"] - 2)
    ax.set_title("Agreement among heads on a single sink token", loc="center")
    fig.dv_caption = (
        "Fraction of heads in each observed block that place their strongest image key on the same "
        "token, by condition. A drop under an ablation means the block-wide consensus that defines "
        "a register has been destroyed rather than merely moved."
    )
    _save(fig, path)
    return fig


def fig_norm_recovery(results, path=None, figsize=(7.2, 3.0)):
    """After the edit, does the network rebuild the outlier norm downstream?"""
    from .interventions import condition_tick, norm_recovery

    df = norm_recovery(results)
    if df.empty:
        raise ValueError("no layer metrics in the intervention results")
    conds = _cond_order(sorted(set(df["condition"])))

    fig, ax = plt.subplots(figsize=figsize)
    for c in conds:
        sub = df[df["condition"] == c].sort_values("layer")
        ax.plot(sub["layer"], sub["norm_ratio"], marker="o", markersize=3.5,
                color=_cond_color(c), label=condition_tick(c))
    ax.axhline(1.0, color=st.RULE, linewidth=0.8)
    ax.set_yscale("log")
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Outlier ratio  (largest / typical)")
    ax.set_xticks(sorted(set(df["layer"])))
    st.open_box(ax); st.light_grid(ax, "y")
    st.legend_outside(ax, fontsize=plt.rcParams["legend.fontsize"] - 2)
    ax.set_title("Recovery of the outlier norm downstream of the edit", loc="center")
    fig.dv_caption = (
        "Largest residual norm relative to the layer median, at the layers observed after the edit. "
        "A curve that returns towards the unedited level shows the network rebuilds the outlier "
        "downstream; one that stays flat shows the edit persists for the rest of the forward pass."
    )
    _save(fig, path)
    return fig


def fig_attention_reallocation(results, conditions: Optional[Sequence[str]] = None, path=None,
                               figsize=None):
    """Where does the attention go when the image register is removed?"""
    from .interventions import ABLATION_SUITE, attention_reallocation, condition_tick

    df = attention_reallocation(results)
    if df.empty:
        raise ValueError("this model has no text keys in the image attention stream")
    wanted = set(conditions) if conditions else set(ABLATION_SUITE)
    df = df[df["condition"].isin(wanted)]
    conds = _cond_order(sorted(set(df["condition"])))
    layers = sorted(set(df["layer"]))
    figsize = figsize or (2.1 * len(layers) + 1.6, 3.9)

    fig, axes = plt.subplots(1, len(layers), figsize=figsize, squeeze=False, sharey=True)
    fig.subplots_adjust(wspace=0.10, bottom=0.36, top=0.86, left=0.14, right=0.98)
    for ax, l in zip(axes[0], layers):
        sub = df[df["layer"] == l].set_index("condition").reindex(conds)
        ax.bar(np.arange(len(conds)), sub["text_mass"], 0.7,
               color=[_cond_color(c) for c in conds], edgecolor="white", linewidth=0.5)
        ax.set_xticks(np.arange(len(conds)))
        ax.set_xticklabels([condition_tick(c) for c in conds], rotation=35, ha="right",
                           rotation_mode="anchor")
        ax.set_ylim(0, 1.0)
        ax.set_title(f"layer {l}", loc="center", fontsize=plt.rcParams["axes.labelsize"])
        st.open_box(ax); st.light_grid(ax, "y")
    axes[0][0].set_ylabel("Share of attention mass\non text keys")
    fig.supxlabel("Condition", y=0.012)
    fig.suptitle("Reallocation of attention to text keys", y=0.98)
    fig.dv_caption = (
        "Share of the image queries' raw attention mass that lands on text keys, by condition and "
        "observed layer. An increase under register ablation would mean the image register and the "
        "text sink are interchangeable places for a head to park its attention."
    )
    _save(fig, path)
    return fig


def fig_image_change(dist: pd.DataFrame, tests: Optional[pd.DataFrame] = None, path=None,
                     figsize=(7.2, 3.2)):
    """How much does the generated image actually change?"""
    from .interventions import condition_tick

    if dist.empty:
        raise ValueError("no image distances")
    metric = str(dist["metric"].iloc[0])
    conds = _cond_order(sorted(set(dist["condition"])))
    agg = dist.groupby("condition")["distance"].agg(["mean", "std", "count"]).reindex(conds)

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(np.arange(len(conds)), agg["mean"], 0.68,
           yerr=agg["std"].fillna(0.0), capsize=2,
           error_kw=dict(elinewidth=0.8, capthick=0.8),
           color=[_cond_color(c) for c in conds], edgecolor="white", linewidth=0.5)
    for i, c in enumerate(conds):
        pts = dist[dist["condition"] == c]["distance"]
        ax.scatter(np.full(len(pts), i), pts, s=6, color="black", alpha=0.45, zorder=4,
                   linewidth=0)
    ax.set_xticks(np.arange(len(conds)))
    ax.set_xticklabels([condition_tick(c) for c in conds], rotation=35, ha="right",
                       rotation_mode="anchor")
    ax.set_xlabel("Condition")
    ax.set_ylabel(f"Distance from the unedited image\n({metric})")
    ax.set_ylim(bottom=0)
    st.open_box(ax); st.light_grid(ax, "y")
    ax.set_title("Perceptual change in the generated image", loc="center")
    note = ""
    if tests is not None and not tests.empty:
        sig = tests.sort_values("p_value").iloc[0]
        note = (f" Paired comparison, {sig['condition_a']} against {sig['condition_b']}: "
                f"median difference {sig['median_difference']:.3f}, Wilcoxon "
                f"p = {sig['p_value']:.3f} over {int(sig['n_pairs'])} generations.")
    fig.dv_caption = (
        f"{metric} distance between each condition's image and the unedited image, over all "
        "prompts and seeds. Points are individual generations, bars the mean, error bars one "
        "standard deviation. The controls bound how much change comes from editing any tokens at "
        "all, so only the gap between an ablation and its controls is attributable to the "
        "registers." + note
    )
    _save(fig, path)
    return fig


def fig_image_grid(root, cfg, seed: Optional[int] = None, conditions: Optional[Sequence[str]] = None,
                   path=None):
    """The generated images themselves, one column per condition."""
    from PIL import Image

    from .interventions import condition_tick

    root = Path(root)
    seed = cfg.seeds[0] if seed is None else seed
    available = [d.name for d in sorted((root / "images").iterdir()) if d.is_dir()]
    conds = _cond_order([c for c in (conditions or available) if c in available])
    rows = list(range(len(cfg.prompts)))
    if not conds or not rows:
        raise ValueError("no images to show")

    fig, axes = plt.subplots(len(rows), len(conds),
                             figsize=(1.35 * len(conds) + 0.6, 1.45 * len(rows) + 0.7),
                             squeeze=False)
    fig.subplots_adjust(wspace=0.04, hspace=0.06, top=0.86, bottom=0.03, left=0.06, right=0.99)
    for ri, pid in enumerate(rows):
        for ci, c in enumerate(conds):
            ax = axes[ri][ci]
            fp = root / "images" / c / f"prompt{pid}_seed{seed}.png"
            if fp.exists():
                ax.imshow(Image.open(fp).convert("RGB"))
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(0.5)
            if ri == 0:
                ax.set_title(condition_tick(c), fontsize=plt.rcParams["legend.fontsize"] - 1,
                             rotation=30, ha="left", va="bottom", x=0.0)
        axes[ri][0].set_ylabel(f"prompt {pid}", fontsize=plt.rcParams["legend.fontsize"] - 1)
    fig.suptitle(f"Generated images under each condition  (seed {seed})", y=0.99)
    fig.dv_caption = (
        f"Images generated under each condition at seed {seed}, one row per prompt. Read against "
        "the controls: an ablation that changes the image no more than its matched-norm and "
        "random-token controls has not shown the registers matter for generation."
    )
    _save(fig, path)
    return fig


def fig_image_difference(root, cfg, seed: Optional[int] = None,
                         conditions: Optional[Sequence[str]] = None, reference: str = "baseline",
                         path=None):
    """Where in the image the change lands."""
    from PIL import Image

    from .interventions import condition_tick

    theme = st.THEME
    root = Path(root)
    seed = cfg.seeds[0] if seed is None else seed
    available = [d.name for d in sorted((root / "images").iterdir()) if d.is_dir()]
    conds = [c for c in _cond_order([c for c in (conditions or available) if c in available])
             if c != reference]
    rows = list(range(len(cfg.prompts)))
    if not conds:
        raise ValueError("no conditions to compare")

    fig, axes = plt.subplots(len(rows), len(conds),
                             figsize=(1.35 * len(conds) + 0.9, 1.45 * len(rows) + 0.8),
                             squeeze=False)
    fig.subplots_adjust(wspace=0.04, hspace=0.06, top=0.84, bottom=0.03, left=0.08, right=0.90)
    im = None
    for ri, pid in enumerate(rows):
        fn = f"prompt{pid}_seed{seed}.png"
        ref = root / "images" / reference / fn
        if not ref.exists():
            continue
        base = np.array(Image.open(ref).convert("RGB")).astype(np.float32)
        diffs = {}
        for c in conds:
            fp = root / "images" / c / fn
            if fp.exists():
                diffs[c] = np.abs(np.array(Image.open(fp).convert("RGB")).astype(np.float32)
                                  - base).mean(axis=2)
        vmax = max((np.percentile(d, 99.5) for d in diffs.values()), default=1.0) or 1.0
        for ci, c in enumerate(conds):
            ax = axes[ri][ci]
            if c in diffs:
                im = ax.imshow(diffs[c], cmap=theme.density_cmap, vmin=0, vmax=vmax)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(0.5)
            if ri == 0:
                ax.set_title(condition_tick(c), fontsize=plt.rcParams["legend.fontsize"] - 1,
                             rotation=30, ha="left", va="bottom", x=0.0)
        axes[ri][0].set_ylabel(f"prompt {pid}", fontsize=plt.rcParams["legend.fontsize"] - 1)
    if im is not None:
        cax = fig.add_axes([0.915, 0.10, 0.014, 0.62])
        cb = fig.colorbar(im, cax=cax)
        cb.set_label("Absolute pixel change")
        cb.outline.set_linewidth(0.8)
        cb.ax.tick_params(width=0.8, direction="in")
    fig.suptitle(f"Spatial distribution of the change in the image  (seed {seed})", y=0.98)
    fig.dv_caption = (
        "Absolute per-pixel difference from the unedited image, one column per condition and one "
        "row per prompt. Each row shares a colour scale. Change concentrated at the patch that was "
        "edited would indicate a local effect; change spread over the image indicates the edit "
        "propagates through generation."
    )
    _save(fig, path)
    return fig


def fig_transfer_capture(results, path=None, figsize=(7.2, 3.0)):
    """Is the register direction on its own enough to make a new token a sink?"""
    from .interventions import condition_tick, transfer_capture

    theme = st.THEME
    df = transfer_capture(results)
    if df.empty:
        raise ValueError("no transfer conditions in these results")
    df = df[df["condition"] == "direction_transfer"] if "direction_transfer" in set(df["condition"]) \
        else df[df["alpha"].isna() | (df["alpha"] == df["alpha"].max())]
    layers = sorted(set(df["layer"]))
    width = 0.36

    fig, ax = plt.subplots(figsize=figsize)
    for i, which in enumerate(["transfer", "baseline"]):
        sub = df[df["which"] == which].set_index("layer").reindex(layers)
        ax.bar(np.arange(len(layers)) + (i - 0.5) * width, sub["capture_rate"], width * 0.9,
               color=theme.color("channel") if which == "transfer" else "0.6",
               edgecolor="white", linewidth=0.5,
               label="after the direction transfer" if which == "transfer"
                     else "same token without the transfer")
    ax.set_xticks(np.arange(len(layers)))
    ax.set_xticklabels([str(l) for l in layers])
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Fraction of heads sinking\non the recipient token")
    ax.set_ylim(0, 1.05)
    st.open_box(ax); st.light_grid(ax, "y")
    st.legend_outside(ax, fontsize=plt.rcParams["legend.fontsize"] - 1)
    ax.set_title("Sink capture by a transferred direction", loc="center")
    fig.dv_caption = (
        "Fraction of heads whose strongest image key is the ordinary token that received a "
        "register's direction, against the rate at which that same token is the sink without the "
        "transfer. The direction is copied at the recipient's own norm, so no magnitude is "
        "injected: a gap between the two bars means the direction alone is sufficient to create "
        "a sink at a new position."
    )
    _save(fig, path)
    return fig


def fig_transfer_magnitude(results, path=None, figsize=(7.2, 3.0)):
    """Does the transferred direction need magnitude to survive downstream?"""
    from .interventions import transfer_capture

    theme = st.THEME
    df = transfer_capture(results)
    df = df[(df["which"] == "transfer") & df["alpha"].notna()] if not df.empty else df
    if df.empty:
        raise ValueError("no magnitude-sweep conditions in these results")
    layers = sorted(set(df["layer"]))

    fig, ax = plt.subplots(figsize=figsize)
    cmap = plt.get_cmap(theme.probability_cmap)
    for i, l in enumerate(layers):
        sub = df[df["layer"] == l].sort_values("alpha")
        ax.plot(sub["alpha"], sub["capture_rate"], marker="o", markersize=4,
                color=cmap(0.15 + 0.7 * i / max(len(layers) - 1, 1)), label=f"layer {l}")
    ax.set_xlabel("Insertion norm  (multiples of the median token norm)")
    ax.set_ylabel("Fraction of heads sinking\non the recipient token")
    ax.set_ylim(0, 1.05)
    st.open_box(ax); st.light_grid(ax, "y")
    st.legend_outside(ax, fontsize=plt.rcParams["legend.fontsize"] - 1)
    ax.set_title("Dependence of sink capture on insertion magnitude", loc="center")
    fig.dv_caption = (
        "Sink capture at the recipient token as the norm at which the register direction is "
        "inserted is swept. Attention normalises its inputs, so capture at the layer of the edit "
        "should not depend on magnitude; a rise with magnitude at later layers would indicate the "
        "direction needs magnitude to survive the intervening blocks rather than to be selected."
    )
    _save(fig, path)
    return fig
