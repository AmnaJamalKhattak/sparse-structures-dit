r"""3D figures of the measured residual-stream geometry.

Every point in every figure here is a real inner product of a real token activation with
a real basis vector. No schematic, no illustrative coordinates, no decorative geometry.

Each figure states three things about its projection:

1. **The basis.** Which three directions are drawn, and how the third one was fitted.
2. **The retained energy.** ``captured_fraction`` is the share of a token's squared norm
   the three axes keep. Three axes out of a few thousand can be a faithful summary or a
   shadow, and the number tells a reader which.
3. **Degeneracies.** Where ``v*`` is nearly the dominant channel, axis 2 spans a
   vanishingly thin slice, and a flat spread there means "nothing to see" rather than
   "the channel does not matter".

The visual language is spare: white panes, no wire cage, one thin axis line per
direction with an arrowhead, ordinary tokens as a faint translucent haze, the
population of interest as saturated opaque points.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib import patheffects
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from . import style as st
from .geometry import CATEGORY_LABELS, Basis, GeometrySnapshot, token_table

# Okabe-Ito again, so a token category keeps one identity across every figure. Ordinary
# tokens are grey and faint: they are the backdrop the population of interest is read
# against, not a series competing for attention.
CATEGORY_STYLE: Dict[str, Dict[str, Any]] = {
    "ordinary": dict(color="#9aa0a6", size=5.0, alpha=0.20, zorder=2, edge="none"),
    "highnorm_nonsink": dict(color=st.OKABE_ITO["blue"], size=42.0, alpha=0.92, zorder=4,
                             edge="white"),
    "highnorm_sink": dict(color=st.OKABE_ITO["vermillion"], size=78.0, alpha=0.97, zorder=6,
                          edge="white"),
}
VSTAR_COLOR = st.OKABE_ITO["purple"]
CHANNEL_COLOR = st.OKABE_ITO["green"]
RESIDUAL_COLOR = st.OKABE_ITO["sky"]


# ------------------------------------------------------------------ 3D scaffold
def _clean_3d(ax, labels: Sequence[str], *, ticks: int = 4,
              label_size: float = 8.5) -> None:
    """Strip the default 3D chrome down to three labelled directions.

    Matplotlib's 3D default is a wire cage with shaded panes, which reads as a box the
    data sits inside. The data is a projection, not a box, so the panes go, the grid
    drops to the faintest line that still gives depth, and the tick count comes down to
    the few values a reader needs for scale.
    """
    ax.set_facecolor("white")
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        axis.pane.set_edgecolor((1.0, 1.0, 1.0, 0.0))
        axis._axinfo["grid"].update(color="#eeeeee", linewidth=0.45, linestyle="-")
        axis.line.set_color("#c8c8c8")
        axis.line.set_linewidth(0.7)
    ax.set_xlabel(labels[0], labelpad=10, fontsize=label_size, color=st.INK_SOFT)
    ax.set_ylabel(labels[1], labelpad=10, fontsize=label_size, color=st.INK_SOFT)
    ax.set_zlabel(labels[2], labelpad=10, fontsize=label_size, color=st.INK_SOFT)
    ax.tick_params(labelsize=6.5, colors="#8a8a8a", pad=0.5)
    if ticks:
        for locator in (ax.xaxis, ax.yaxis, ax.zaxis):
            locator.set_major_locator(plt.MaxNLocator(ticks, prune="both"))
    try:
        # `zoom` is what actually fills the frame: mplot3d's default leaves roughly a
        # third of the axes empty, which on a scatter reads as a sparse cloud.
        ax.set_box_aspect((1, 1, 0.9), zoom=1.18)
    except TypeError:
        try:
            ax.set_box_aspect((1, 1, 0.9))
        except Exception:
            pass
    except Exception:
        pass


def _equalise(ax, points: np.ndarray, *, margin: float = 1.04) -> float:
    """One shared scale on all three axes, centred on the **origin**.

    Centred on the origin rather than on the cloud's centroid, because in this basis the
    origin is zero activation and the axes are directions through it: a cloud offset from
    the centre of the frame is a real fact about the population, and re-centring on its
    mean would erase it.

    The scale is shared across the three axes on purpose. Unequal axis scales make an
    angle in the picture unrelated to the angle in the data, which is exactly how a
    truthful projection becomes a misleading one.
    """
    if not len(points):
        return 1.0
    span = float(np.abs(points).max()) * margin or 1.0
    for setter in (ax.set_xlim, ax.set_ylim, ax.set_zlim):
        setter(-span, span)
    return span


def _arrow(ax, start, end, *, color: str, label: str = "", lw: float = 2.0,
           head: float = 0.11, fontsize: float = 8.0, label_offset: float = 1.10) -> None:
    """A thin shaft with a cone head, drawn as data-space lines.

    ``quiver`` in mplot3d scales its head in axes units, so arrows change thickness with
    the camera. Drawing the shaft and a small cone in data coordinates keeps the arrow
    the same object whatever the view angle.
    """
    start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    ax.plot(*zip(start, end), color=color, lw=lw, solid_capstyle="round", zorder=8)
    direction = end - start
    length = float(np.linalg.norm(direction))
    if length <= 0:
        return
    unit = direction / length
    base = end - unit * head * length
    # Two perpendiculars, so the cone reads as a head from any camera angle.
    helper = np.array([1.0, 0.0, 0.0]) if abs(unit[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    p1 = np.cross(unit, helper)
    p1 /= np.linalg.norm(p1) or 1.0
    p2 = np.cross(unit, p1)
    radius = head * length * 0.34
    ring = [base + radius * (math.cos(t) * p1 + math.sin(t) * p2)
            for t in np.linspace(0, 2 * math.pi, 13)]
    ax.add_collection3d(Line3DCollection([[tuple(point), tuple(end)] for point in ring],
                                         colors=color, linewidths=0.7, zorder=8))
    if label:
        tip = start + direction * label_offset
        ax.text(*tip, label, color=color, fontsize=fontsize, ha="center", va="center",
                zorder=9, path_effects=[patheffects.withStroke(linewidth=2.2,
                                                               foreground="white")])


def _basis_note(basis: Basis, captured: np.ndarray) -> str:
    """The basis and retained-energy line printed on the figure, not only in the caption."""
    pieces = [f"3D projection of {int(basis.vectors.shape[1])}-d activations"
              f"  ·  {basis.kind} basis",
              f"axes retain {100 * float(np.median(captured)):.0f}% of a median token's "
              f"energy"]
    if np.isfinite(basis.total_variance_explained):
        pieces.append(f"{100 * basis.total_variance_explained:.0f}% of population variance")
    return "      ".join(pieces)


def _stamp(fig, text: str, *, y: float = 0.012) -> None:
    fig.text(0.5, y, text, ha="center", va="bottom", fontsize=7.0, color=st.INK_SOFT)


def _finish(fig, caption: str, path=None):
    fig.dv_caption = caption
    if path:
        st.savefig(fig, Path(path))
    return fig


def _scatter_categories(ax, snapshot: GeometrySnapshot, coordinates: np.ndarray, *,
                        span: Optional[float] = None, scale: float = 1.0,
                        legend: bool = False) -> int:
    """Draw the three token categories, clamping anything outside ``span`` visibly.

    Matplotlib silently drops points outside the axis limits, which on a figure whose
    whole subject is the extreme tail would quietly delete the evidence. Out-of-range
    tokens are therefore pulled to the boundary and drawn as hollow diamonds, and the
    count is returned so the caller can say how many were clamped.
    """
    clamped = 0
    for category in ("ordinary", "highnorm_nonsink", "highnorm_sink"):
        mask = (snapshot.tokens["category"] == category).to_numpy()
        if not mask.any():
            continue
        style = CATEGORY_STYLE[category]
        points = coordinates[mask]
        outside = (np.abs(points).max(axis=1) > span) if span else np.zeros(len(points), bool)
        inside = points[~outside]
        if len(inside):
            ax.scatter(inside[:, 0], inside[:, 1], inside[:, 2], s=style["size"] * scale,
                       c=style["color"], alpha=style["alpha"], depthshade=False,
                       edgecolors=style["edge"],
                       linewidths=0.0 if style["edge"] == "none" else 0.55 * scale,
                       label=(f'{CATEGORY_LABELS[category]}  ({int(mask.sum())})'
                              if legend else None),
                       zorder=style["zorder"])
        if outside.any():
            edge = np.clip(points[outside], -span, span)
            clamped += int(outside.sum())
            ax.scatter(edge[:, 0], edge[:, 1], edge[:, 2], s=style["size"] * scale * 1.1,
                       facecolors="none", edgecolors=style["color"], marker="D",
                       linewidths=0.9 * scale, depthshade=False,
                       zorder=style["zorder"] + 1)
    return clamped



# ================================================= 1. the empirical token cloud
def fig_token_cloud(snapshot: GeometrySnapshot, path=None, *, elev: float = 19.0,
                    azim: float = -58.0, show_extremes: bool = True,
                    arrow_scale: float = 0.85, figsize=(8.6, 7.0)):
    """The measured token population in the interpretable basis.

    Axis 1 is ``v*`` itself, so a token's horizontal coordinate is its projection
    :math:`\\alpha_i = x_i^\\top v^*`, not a proxy for it. Axis 2 is what the dominant
    channel carries beyond ``v*``, and axis 3 is the loudest remaining variation in this
    layer's own tokens.

    A dense grey core of ordinary tokens sits near the origin, a spur of high-norm
    tokens extends along axis 1, and the sinks sit at the far end of that spur rather
    than scattered through it.
    """
    coordinates = snapshot.coordinates
    tokens = snapshot.tokens
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    _clean_3d(ax, snapshot.basis.labels)
    span = _equalise(ax, coordinates)

    clamped = _scatter_categories(ax, snapshot, coordinates, span=span, legend=True)
    if show_extremes:
        extreme = (tokens["is_extreme"] & ~tokens["is_sink"]).to_numpy()
        if extreme.any():
            points = coordinates[extreme]
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=110, facecolors="none",
                       edgecolors=st.INK_SOFT, linewidths=0.9, zorder=5,
                       label=f'{CATEGORY_LABELS["extreme"]}, not a sink '
                             f'({int(extreme.sum())})')

    # The basis directions, drawn from the origin because that is where they pass
    # through: v* is a direction in the residual stream, not an offset from the cloud's
    # centroid. Scaled inside the data's own extent so they read as axes rather than
    # competing with the points for the frame.
    reach = span * arrow_scale
    origin = np.zeros(3)
    _arrow(ax, origin, np.array([reach, 0, 0]), color=VSTAR_COLOR, label=r"$v^*$")
    if not snapshot.basis.degenerate_axis_2:
        _arrow(ax, origin, np.array([0, reach, 0]), color=CHANNEL_COLOR,
               label=f"$e_{{c^*}}$ (ch {snapshot.dominant_channel})", fontsize=7.5)
    _arrow(ax, origin, np.array([0, 0, reach * 0.82]), color=RESIDUAL_COLOR,
           label="residual PC", fontsize=7.5, lw=1.5)

    ax.view_init(elev=elev, azim=azim)
    ax.set_title(f"{snapshot.checkpoint}  ·  layer {snapshot.layer}, step {snapshot.step}"
                 f"  ·  {snapshot.condition}", fontsize=10.5, pad=12, color=st.INK)
    legend = ax.legend(loc="upper left", bbox_to_anchor=(-0.06, 0.98), frameon=False,
                       fontsize=8.0, handletextpad=0.5, borderaxespad=0.0)
    for handle in legend.legend_handles:
        handle.set_alpha(1.0)
    note = _basis_note(snapshot.basis, snapshot.captured)
    if clamped:
        note += f"      {clamped} token(s) beyond the axis range, drawn as hollow diamonds"
    _stamp(fig, note)
    if snapshot.basis.notes:
        fig.text(0.5, 0.045, snapshot.basis.notes.split(". ")[0] + ".", ha="center",
                 va="bottom", fontsize=7.0, color=st.OKABE_ITO["vermillion"], style="italic")
    fig.subplots_adjust(left=0.02, right=0.97, top=0.93, bottom=0.10)

    counts = tokens["category"].value_counts()
    return _finish(fig, (
        f"Measured image-token activations at layer {snapshot.layer}, denoising step "
        f"{snapshot.step} of {snapshot.checkpoint}, projected onto three orthonormal "
        f"directions: (1) the frozen register direction $v^*$, so a token's first "
        f"coordinate is exactly $\\alpha_i = x_i^\\top v^*$; (2) the dominant-channel "
        f"basis vector $e_{{c^*}}$ (channel {snapshot.dominant_channel}) orthogonalised "
        f"against $v^*$, so axis 2 carries only what the channel adds beyond the "
        f"direction; (3) the first principal direction of what remains, fitted on this "
        f"layer's own tokens and defined up to sign. "
        f"{int(counts.get('ordinary', 0))} ordinary tokens, "
        f"{int(counts.get('highnorm_nonsink', 0))} high-norm non-sinks and "
        f"{int(counts.get('highnorm_sink', 0))} high-norm sinks, using the project's own "
        f"thresholds: norm $\\geq$ {snapshot.norm_threshold:.3g} and incoming attention "
        f"$\\geq$ {snapshot.sink_threshold:g}$\\times$ the uniform share. This is a "
        f"projection of {int(snapshot.states.shape[-1])}-dimensional states; the three "
        f"axes retain {100 * float(np.median(snapshot.captured)):.0f}% of a median "
        f"token's squared norm, so distances along the plotted axes are real lower bounds "
        f"on distances in the full space."), path)


# ============================================ 2a. inside a token: 3D decomposition
def fig_token_decomposition(snapshot: GeometrySnapshot, path=None, *,
                            tokens: Optional[Dict[str, Optional[int]]] = None,
                            elev: float = 20.0, azim: float = -60.0):
    r"""$x_i = \alpha_i v^* + r_i$, drawn for one token of each kind, with no loss.

    Each panel uses a basis built for that token: axes 1 and 2 stay the interpretable
    pair ($\hat v^*$ and the orthogonalised dominant channel), and axis 3 is whatever is
    left of this particular token after those two. The token therefore lies exactly in
    the drawn subspace, so every arrow has its true length and the printed norms are the
    plotted ones. A shared basis would not work here: most of an individual residual
    falls outside any three fixed directions, and the $r_i$ leg would be drawn shorter
    than it is.

    The price, stated on the figure, is that axis 3 differs between panels. That is the
    right trade for a decomposition: the quantity of interest is the split *within* a
    token, not the comparison of residual directions *between* tokens.
    """
    from .geometry import exact_token_basis

    chosen = tokens or snapshot.representatives()
    present = [(key, token) for key, token in
               ((k, chosen.get(k)) for k in
                ("ordinary", "highnorm_nonsink", "highnorm_sink")) if token is not None]
    if not present:
        raise ValueError("cannot draw the decomposition: no representative token exists")

    fig = plt.figure(figsize=(4.5 * len(present), 4.35))
    for index, (key, token) in enumerate(present):
        ax = fig.add_subplot(1, len(present), index + 1, projection="3d")
        state = snapshot.states[int(token)].float()
        basis = exact_token_basis(state, snapshot.vstar, snapshot.dominant_channel)
        point = basis.project(state.unsqueeze(0)).numpy()[0]
        _clean_3d(ax, ("$v^*$", "ch $c^*$ axis", "own residual"), ticks=3, label_size=8.0)
        span = float(np.abs(point).max()) * 1.18 or 1.0
        for setter in (ax.set_xlim, ax.set_ylim, ax.set_zlim):
            setter(-span, span)

        alpha = float(point[0])
        parallel = np.array([alpha, 0.0, 0.0])
        residual = point - parallel
        # The two legs of the right triangle, and the hypotenuse that is the token.
        _arrow(ax, (0, 0, 0), tuple(parallel), color=VSTAR_COLOR, lw=2.6,
               label=r"$\alpha_i v^*$", label_offset=1.22, fontsize=9.0)
        _arrow(ax, tuple(parallel), tuple(point), color=RESIDUAL_COLOR, lw=2.2,
               label=r"$r_i$", label_offset=1.06, fontsize=9.0)
        _arrow(ax, (0, 0, 0), tuple(point), color=st.INK, lw=1.6, head=0.09,
               label=r"$x_i$", label_offset=1.14, fontsize=9.0)
        # Guide lines closing the triangle, so the right angle is visible: the two legs
        # are orthogonal because the basis is orthonormal.
        ax.plot(*zip(np.zeros(3), point), color=st.GRID, lw=0.0)
        corner = np.array([alpha, point[1], 0.0])
        for a, b in ((parallel, corner), (corner, point)):
            ax.plot(*zip(a, b), color="#d8d8d8", lw=0.7, ls=(0, (2, 2)), zorder=1)

        energy = max(float(state.pow(2).sum()), 1e-12)
        share = alpha ** 2 / energy
        channel_value = float(state[int(snapshot.dominant_channel)])
        channel_share = channel_value ** 2 / energy
        row = snapshot.tokens.iloc[int(token)]
        colour = CATEGORY_STYLE[key]["color"]
        ax.set_title(f"{CATEGORY_LABELS[key]}", fontsize=9.5, pad=18, color=colour)
        ax.text2D(0.5, 0.965, f"token {int(token)}   ·   $||x_i||$ = {float(row['norm']):.2f}",
                  transform=ax.transAxes, ha="center", fontsize=8.5, color=st.INK)
        ax.text2D(0.5, -0.02,
                  f"$\\alpha_i$ = {alpha:+.2f}      $||r_i||$ = {float(np.linalg.norm(residual)):.2f}\n"
                  f"{100 * share:.0f}% of energy on $v^*$   ·   {100 * channel_share:.0f}% in "
                  f"channel {snapshot.dominant_channel} ({channel_value:+.2f})",
                  transform=ax.transAxes, ha="center", va="top", fontsize=8.0,
                  color=st.INK_SOFT)
        ax.view_init(elev=elev, azim=azim)

    fig.suptitle(r"Inside a token:   $x_i = \alpha_i v^* + r_i$", fontsize=12.0, y=0.985)
    _stamp(fig, "exact per-token basis: each token lies fully in its own drawn subspace, "
                "so every arrow has its true length  ·  axis 3 differs between panels", y=0.004)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.84, bottom=0.17, wspace=0.06)
    return _finish(fig, (
        f"The orthogonal decomposition $x_i = \\alpha_i v^* + r_i$ for one representative "
        f"token of each category at layer {snapshot.layer}, step {snapshot.step} of "
        f"{snapshot.checkpoint}. The ordinary representative is the median-norm ordinary "
        f"token; each high-norm representative is the loudest of its kind. "
        f"Unlike the population figures, each panel uses a basis built for its own token: "
        f"axes 1 and 2 are the same interpretable pair, and axis 3 is what remains of "
        f"that token after them. The token therefore lies exactly in the drawn subspace, "
        f"so the purple leg has length $|\\alpha_i|$, the blue leg has length $||r_i||$, "
        f"and the two meet at a right angle because the basis is orthonormal -- nothing "
        f"is foreshortened and nothing is out of the page. Axis 3 consequently differs "
        f"between panels, which is the correct trade for a figure about the split within "
        f"a token rather than about comparing residual directions between tokens. Both "
        f"printed shares are computed in the full "
        f"{int(snapshot.states.shape[-1])}-dimensional space."), path)


# ======================================= 2b. inside a token: channel decomposition
def fig_channel_decomposition(snapshot: GeometrySnapshot, path=None, *,
                              tokens: Optional[Dict[str, Optional[int]]] = None,
                              top_channels: int = 48):
    r"""The same tokens channel by channel: is the token large because of $c^*$?

    Each panel shows the token's actual per-channel activation, with the $v^*$-aligned
    component $\alpha_i v^*$ overlaid. Where the two agree the channel's value *is* the
    register state; where the bar stands above the overlay, that channel carries
    something else. The dominant channel is marked, so "large because of one channel" and
    "large because of a distributed direction" are visibly different pictures.
    """
    chosen = tokens or snapshot.representatives()
    present = [(key, token) for key, token in
               ((k, chosen.get(k)) for k in ("ordinary", "highnorm_nonsink", "highnorm_sink"))
               if token is not None]
    if not present:
        raise ValueError("cannot draw the channel decomposition: no representative token")

    unit = snapshot.vstar.float()
    channel = int(snapshot.dominant_channel)
    # Rank channels by the loudest representative, so every panel shares one x axis and
    # the panels are directly comparable rather than each sorted to flatter itself.
    reference = snapshot.states[int(present[-1][1])].float()
    order = torch.argsort(reference.abs(), descending=True)[:int(top_channels)]
    if channel not in set(int(c) for c in order):
        order = torch.cat([order[:-1], torch.tensor([channel])])
    positions = np.arange(len(order))

    fig, axes = plt.subplots(len(present), 1, figsize=(9.4, 2.35 * len(present)),
                             sharex=True, squeeze=False)
    flat = [ax for row in axes for ax in row]
    for ax, (key, token) in zip(flat, present):
        state = snapshot.states[int(token)].float()
        alpha = float(state @ unit)
        aligned = (alpha * unit)[order].numpy()
        values = state[order].numpy()
        colors = [CATEGORY_STYLE[key]["color"] if int(c) != channel
                  else st.OKABE_ITO["orange"] for c in order]
        ax.bar(positions, values, width=0.78, color=colors, edgecolor="white",
               linewidth=0.35, zorder=3,
               label="measured activation $x_i$")
        ax.step(positions, aligned, where="mid", color=VSTAR_COLOR, lw=1.5, zorder=4,
                label=r"$v^*$-aligned component $\alpha_i v^*$")
        ax.axhline(0.0, color=st.RULE, lw=0.7, zorder=2)
        mark = int((order == channel).nonzero()[0, 0]) if (order == channel).any() else None
        if mark is not None:
            ax.axvline(mark, color=st.OKABE_ITO["orange"], lw=0.9, ls=(0, (3, 2)), zorder=1)
            ax.annotate(f"$c^*$ = {channel}", (mark, 0.0), xycoords=("data", "axes fraction"),
                        textcoords="offset points", xytext=(4, 3), fontsize=7.5,
                        color=st.OKABE_ITO["orange"], va="bottom")
        row = snapshot.tokens.iloc[int(token)]
        share = alpha ** 2 / max(float(state.pow(2).sum()), 1e-12)
        channel_share = float(state[channel]) ** 2 / max(float(state.pow(2).sum()), 1e-12)
        ax.set_ylabel("activation", fontsize=8.5)
        ax.set_title(f"{CATEGORY_LABELS[key]}  ·  token {int(token)}  ·  "
                     f"$||x_i||$ = {float(row['norm']):.1f}  ·  "
                     f"{100 * channel_share:.0f}% of energy in channel {channel}, "
                     f"{100 * share:.0f}% on $v^*$", fontsize=8.5, loc="left", pad=4)
        st.open_box(ax)
        st.light_grid(ax, "y")
    flat[0].legend(frameon=False, fontsize=7.5, loc="upper left", ncol=2)
    flat[-1].set_xticks(positions[::max(1, len(positions) // 12)])
    flat[-1].set_xticklabels([str(int(order[i])) for i in
                              range(0, len(order), max(1, len(positions) // 12))],
                             fontsize=7)
    flat[-1].set_xlabel(f"channel index, ordered by $|$activation$|$ of the loudest "
                        f"representative (top {len(order)} of "
                        f"{int(snapshot.states.shape[-1])})", fontsize=8.5)
    fig.tight_layout()
    return _finish(fig, (
        f"Channel-wise view of the same representative tokens at layer {snapshot.layer}, "
        f"step {snapshot.step} of {snapshot.checkpoint}. Bars are measured activations; "
        f"the stepped line is the $v^*$-aligned component $\\alpha_i v^*$ evaluated "
        f"channel by channel. Channels are ordered once, by the loudest representative, "
        f"so the three panels share an x axis and can be compared directly. The dominant "
        f"channel $c^*$ = {snapshot.dominant_channel} is marked. Two quantities are "
        f"printed per token, both computed in the full space: the share of its squared "
        f"norm sitting in channel $c^*$ alone, and the share carried by $v^*$. A token "
        f"whose two shares are close is large because of that one channel; a token whose "
        f"$v^*$ share is much the larger is large because of a distributed direction. "
        f"No projection is involved in this figure: the bars are the raw measured "
        f"activations, in the model's own coordinates."), path)


# ================================================ 3. lifecycle across depth/time
def fig_lifecycle_strip(snapshots: Sequence[GeometrySnapshot], path=None, *,
                        columns: Optional[int] = None, elev: float = 18.0,
                        azim: float = -58.0, by: str = "layer",
                        shared_limits: bool = True,
                        span_percentile: Optional[float] = None):
    """The population's geometry across depth or denoising time, on one fixed basis.

    Every frame uses the same basis and, by default, the same axis limits, which is what
    makes the sequence a comparison rather than a slideshow: a cloud that appears to grow
    is growing, not being rescaled. Three transitions to look for: tokens moving out
    along axis 1, the extreme members acquiring sink markers, and the spur collapsing
    back toward the core.
    """
    if not len(snapshots):
        raise ValueError("cannot draw the lifecycle strip: no snapshots")
    frames = sorted(snapshots, key=lambda s: (s.step, s.layer))
    # Choose a column count that leaves no ragged tail where one exists, so the strip
    # reads as a sequence rather than as a grid with holes in it.
    if columns is None:
        columns = next((c for c in range(min(6, len(frames)), 2, -1)
                        if len(frames) % c == 0), min(5, len(frames)))
    rows = int(math.ceil(len(frames) / max(columns, 1)))
    fig = plt.figure(figsize=(3.5 * min(columns, len(frames)), 3.35 * rows))
    limits = None
    if shared_limits:
        stacked = np.abs(np.vstack([s.coordinates for s in frames]))
        limits = (float(np.percentile(stacked, span_percentile)) if span_percentile
                  else float(stacked.max())) * 1.04 or 1.0

    clamped = 0
    for index, snapshot in enumerate(frames):
        ax = fig.add_subplot(rows, min(columns, len(frames)), index + 1, projection="3d")
        _clean_3d(ax, ("$v^*$", "ch $c^*$", "resid"), ticks=0, label_size=7.0)
        ax.tick_params(labelsize=0, length=0)
        coordinates = snapshot.coordinates
        if limits is not None:
            for setter in (ax.set_xlim, ax.set_ylim, ax.set_zlim):
                setter(-limits, limits)
        else:
            _equalise(ax, coordinates)
        clamped += _scatter_categories(ax, snapshot, coordinates, span=limits, scale=0.62)
        ax.view_init(elev=elev, azim=azim)
        loud = int((snapshot.tokens["category"] != "ordinary").sum())
        sinks = int((snapshot.tokens["category"] == "highnorm_sink").sum())
        heading = (f"layer {snapshot.layer}" if by == "layer" else f"step {snapshot.step}")
        ax.set_title(f"{heading}   ·   {loud} high-norm, {sinks} sink",
                     fontsize=8.0, pad=2, color=st.INK)
    handles = [plt.Line2D([], [], marker="o", linestyle="none",
                          markerfacecolor=CATEGORY_STYLE[c]["color"],
                          markeredgecolor="none",
                          markersize=4 + 2 * i, label=CATEGORY_LABELS[c])
               for i, c in enumerate(("ordinary", "highnorm_nonsink", "highnorm_sink"))]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.5, 0.022))
    first = frames[0]
    fig.suptitle(f"{first.checkpoint}  ·  register lifecycle in a fixed $v^*$ basis  ·  "
                 f"{first.condition}", fontsize=11.0, y=0.985)
    strip_note = _basis_note(first.basis, first.captured)
    if span_percentile:
        strip_note += (f"      axes clipped at the {span_percentile:g}th percentile; "
                       f"{clamped} token-frame(s) drawn as hollow diamonds on the boundary")
    _stamp(fig, strip_note, y=0.002)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.085, wspace=0.02,
                        hspace=0.16)
    return _finish(fig, (
        f"The measured token population across {by}s, every frame projected onto the "
        f"*same* basis, fitted once, and drawn on the same axis limits -- so a cloud that "
        f"appears to move is moving. Axis 1 is $v^*$, axis 2 the orthogonalised dominant "
        f"channel, axis 3 the first residual principal direction. Counts above each frame "
        f"use the project's thresholds. What to look for: tokens extending along axis 1 "
        f"as the register state is written, the extreme members of that spur acquiring "
        f"sink markers, and the spur collapsing back toward the core during dissolution."),
        path)


def write_lifecycle_animation(snapshots: Sequence[GeometrySnapshot], path, *,
                              fps: int = 4, elev: float = 18.0, azim: float = -58.0,
                              dpi: int = 110) -> Dict[str, Any]:
    """A GIF of the lifecycle, and the per-frame PNGs whether or not the GIF works.

    Frames are written first and unconditionally. Animation writers are the most
    environment-dependent part of matplotlib, so the frames are the deliverable and the
    GIF is a convenience.
    """
    path = Path(path)
    frames_dir = path.parent / f"{path.stem}_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    frames = sorted(snapshots, key=lambda s: (s.step, s.layer))
    if not frames:
        return {"gif": None, "frames": [], "note": "no snapshots"}

    stacked = np.vstack([s.coordinates for s in frames])
    span = float(np.abs(stacked).max()) * 1.04 or 1.0

    written: List[str] = []
    for snapshot in frames:
        figure = plt.figure(figsize=(6.4, 5.4))
        ax = figure.add_subplot(111, projection="3d")
        _clean_3d(ax, snapshot.basis.labels)
        for setter in (ax.set_xlim, ax.set_ylim, ax.set_zlim):
            setter(-span, span)
        coordinates = snapshot.coordinates
        for category in ("ordinary", "highnorm_nonsink", "highnorm_sink"):
            mask = (snapshot.tokens["category"] == category).to_numpy()
            if not mask.any():
                continue
            style = CATEGORY_STYLE[category]
            points = coordinates[mask]
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=style["size"],
                       c=style["color"], alpha=style["alpha"], depthshade=False,
                       edgecolors=style["edge"],
                       linewidths=0.0 if style["edge"] == "none" else 0.5,
                       zorder=style["zorder"])
        ax.view_init(elev=elev, azim=azim)
        loud = int((snapshot.tokens["category"] != "ordinary").sum())
        sinks = int((snapshot.tokens["category"] == "highnorm_sink").sum())
        ax.set_title(f"{snapshot.checkpoint}  ·  layer {snapshot.layer}, step "
                     f"{snapshot.step}  ·  {loud} high-norm, {sinks} sink",
                     fontsize=10.0, pad=10)
        _stamp(figure, _basis_note(snapshot.basis, snapshot.captured))
        figure.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.09)
        name = frames_dir / f"frame_s{snapshot.step:03d}_l{snapshot.layer:03d}.png"
        figure.savefig(name, dpi=dpi, facecolor="white")
        plt.close(figure)
        written.append(str(name))

    gif, note = None, ""
    try:
        from PIL import Image

        images = [Image.open(name).convert("RGB") for name in written]
        if images:
            images[0].save(path, save_all=True, append_images=images[1:],
                           duration=int(1000 / max(fps, 1)), loop=0)
            gif = str(path)
    except Exception as exc:                                   # pragma: no cover
        note = f"GIF not written ({type(exc).__name__}: {exc}); the frames are complete."
    return {"gif": gif, "frames": written, "fps": fps, "note": note,
            "n_frames": len(written)}


# ============================================= 4. basis diagnostics panel
def fig_basis_diagnostics(snapshots: Sequence[GeometrySnapshot], path=None):
    """What the 3D views are and are not entitled to claim.

    Three panels. (a) How much of each token actually survives the projection, by
    category: the number that decides whether the cloud is a summary or a shadow.
    (b) How much of $v^*$ is the dominant channel, which bounds any claim that axis 1 is
    about geometry rather than about one coordinate. (c) How the retained share varies
    with depth, since a projection can be faithful at one layer and thin at another.
    """
    if not len(snapshots):
        raise ValueError("cannot draw the basis diagnostics: no snapshots")
    table = token_table(snapshots)
    first = snapshots[0]
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.9))

    ax = axes[0]
    groups = [(c, table.loc[table["category"] == c, "captured_fraction"].to_numpy())
              for c in ("ordinary", "highnorm_nonsink", "highnorm_sink")]
    groups = [(c, values) for c, values in groups if len(values)]
    if groups:
        parts = ax.violinplot([values for _, values in groups], showextrema=False,
                              widths=0.8)
        for body, (category, _) in zip(parts["bodies"], groups):
            body.set_facecolor(CATEGORY_STYLE[category]["color"])
            body.set_alpha(0.55)
            body.set_edgecolor("white")
        for index, (category, values) in enumerate(groups, start=1):
            ax.scatter([index], [np.median(values)], s=34, color=st.INK, zorder=4)
            ax.annotate(f"{100 * np.median(values):.0f}%", (index, np.median(values)),
                        textcoords="offset points", xytext=(9, -3), fontsize=8,
                        color=st.INK)
        ax.set_xticks(range(1, len(groups) + 1))
        ax.set_xticklabels([CATEGORY_LABELS[c].replace(" ", "\n") for c, _ in groups],
                           fontsize=7.5)
    ax.set_ylim(0, 1)
    ax.set_ylabel("share of a token's squared norm\nretained by the 3 axes", fontsize=8.5)
    ax.axhline(1.0, color=st.GRID, lw=0.8, ls=(0, (2, 2)))
    st.open_box(ax); st.light_grid(ax, "y")

    ax = axes[1]
    alignment = float(first.basis.channel_alignment)
    independent = float(first.basis.independent_channel_content)
    ax.bar([0, 1], [alignment, independent], width=0.55,
           color=[st.OKABE_ITO["orange"], CHANNEL_COLOR], edgecolor="white", linewidth=0.6)
    for x, value in ((0, alignment), (1, independent)):
        ax.annotate(f"{value:.3f}", (x, value), textcoords="offset points", xytext=(0, 4),
                    ha="center", fontsize=8.5, color=st.INK)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"$|v^*_{{c^*}}|$\nhow much of $v^*$\nIS channel "
                        f"{first.dominant_channel}",
                        "$\\sqrt{1-|v^*_{c^*}|^2}$\nunit length surviving\northogonalisation"],
                       fontsize=7.5)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("length (unit vectors)", fontsize=8.5)
    if first.basis.degenerate_axis_2:
        ax.annotate("axis 2 DEGENERATE:\nsubstituted a data-driven\nresidual direction",
                    (1, independent), textcoords="offset points", xytext=(0, 26),
                    ha="center", fontsize=7.5, color=st.OKABE_ITO["vermillion"])
    st.open_box(ax); st.light_grid(ax, "y")

    ax = axes[2]
    if "layer" in table.columns and table["layer"].nunique() > 1:
        for category in ("ordinary", "highnorm_nonsink", "highnorm_sink"):
            rows = table[table["category"] == category]
            if rows.empty:
                continue
            by_layer = rows.groupby("layer", observed=True)["captured_fraction"].median()
            ax.plot(by_layer.index, by_layer.to_numpy(),
                    color=CATEGORY_STYLE[category]["color"], lw=1.9, marker="o",
                    markersize=3.6, markeredgecolor="white", markeredgewidth=0.5,
                    label=CATEGORY_LABELS[category])
        ax.set_xlabel("transformer block", fontsize=8.5)
        ax.set_ylabel("median retained share", fontsize=8.5)
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.set_ylim(0, 1)
        ax.legend(frameon=False, fontsize=7.5, loc="best")
        st.open_box(ax); st.light_grid(ax, "y")
    else:
        ax.text(0.5, 0.5, "one layer captured;\nno depth profile to draw",
                transform=ax.transAxes, ha="center", va="center", fontsize=9,
                color=st.INK_SOFT)
        ax.set_xticks([]); ax.set_yticks([])
    for index, ax in enumerate(axes):
        ax.set_title(f"({chr(97 + index)})", loc="left", fontsize=9.5, color=st.INK)
    fig.tight_layout()
    return _finish(fig, (
        f"What the 3D projections are entitled to claim, for {first.checkpoint}. "
        f"(a) The share of each token's squared norm the three axes retain, by category: "
        f"a high share means the cloud is a faithful summary of those tokens, a low one "
        f"means it is a thin slice of them. The register population and the ordinary "
        f"population generally differ here, and the figure is about the former. "
        f"(b) How much of $v^*$ is the dominant channel itself. Where the left bar is "
        f"near 1 the right bar is near 0, axis 2 spans almost nothing, and no claim "
        f"separating 'the direction' from 'the channel' can rest on this basis -- which "
        f"is announced rather than left to be inferred from a flat axis. "
        f"(c) The retained share by depth, since a projection can be faithful where the "
        f"register state exists and thin elsewhere."), path)


def save_figure_metadata(snapshot: GeometrySnapshot, directory, *, figure: str,
                         extra: Optional[Dict[str, Any]] = None) -> Path:
    """Write the JSON record for a figure into ``<directory>/<figure>.json``."""
    from .geometry import save_metadata

    return save_metadata(snapshot, Path(directory) / f"{figure}.json", figure=figure,
                         extra=extra)
