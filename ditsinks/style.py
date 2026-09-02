"""Figure style for publication.

Typography follows the NeurIPS body text: a Times-like serif (STIXGeneral, which
ships with matplotlib, so it is available in Colab), STIX math, bold figure text
for legibility at final paper size, and ticks inside a closed axes box.

Colour has two jobs and one rule each.

*Categorical* (which phenomenon a line belongs to) uses three colours from the
Okabe-Ito colour-vision-deficiency-safe palette. They clear every adjacent- and
all-pairs CVD threshold and sit above 3:1 contrast on white, so they survive
greyscale printing and projector washout.

*Sequential* (how many tokens fall in a bin) uses `inferno` everywhere the
quantity is a count, because the colour axis then means the same thing in every
panel. Attention probability -- a different quantity -- uses `cividis`, so the
two are never confused.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler

# ---------------------------------------------------------------- colours
# Okabe & Ito (2008), "Color Universal Design".
OKABE_ITO = {
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "green": "#009E73",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "orange": "#E69F00",
    "yellow": "#F0E442",
    "black": "#000000",
}

PHENOMENA = ("highnorm", "sink", "channel")

SERIES = {
    "highnorm": OKABE_ITO["blue"],
    "sink": OKABE_ITO["vermillion"],
    "channel": OKABE_ITO["green"],
    "reference": OKABE_ITO["purple"],
}

LABELS = {
    "highnorm": "high-norm tokens",
    "sink": "attention sinks",
    "channel": "massive activation channels",
}

INK = "#000000"
INK_SOFT = "#3a3a3a"
RULE = "#555555"
GRID = "#d9d9d9"

DENSITY_CMAP = "inferno"       # counts
PROBABILITY_CMAP = "cividis"   # attention probabilities


def diverging_cmap():
    """Signed quantities (e.g. rank correlations): two opposed hues, neutral middle."""
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(
        "ditsinks_diverging",
        [SERIES["highnorm"], "#7FB3DE", "#EFEFEF", "#F0A87A", SERIES["sink"]],
    )


def legend_outside(ax, *, width_fraction: float = 0.72, **kwargs):
    """Put an axes legend in reserved space to its right, never over data.

    Matplotlib's ordinary ``ax.legend()`` draws inside the plotting rectangle
    and can conceal precisely the outliers a diagnostic figure is meant to
    show.  This helper narrows the axes, then uses the released part of the
    figure as a dedicated legend gutter.  ``bbox_inches='tight'`` in
    :func:`savefig` also preserves the gutter in exported PDFs.
    """
    kwargs.pop("loc", None)
    kwargs.pop("ncol", None)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * width_fraction, box.height])
    return ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0),
                     borderaxespad=0.0, ncol=1, **kwargs)

_CONTEXTS = {
    "paper": dict(base=10.0, dpi=150, save_dpi=400, lw=1.4),
    "talk": dict(base=13.0, dpi=140, save_dpi=250, lw=2.0),
}


@dataclass
class Theme:
    context: str = "paper"

    def color(self, key: str) -> str:
        return SERIES.get(key, INK)

    def label(self, key: str) -> str:
        return LABELS.get(key, key)

    @property
    def density_cmap(self) -> str:
        return DENSITY_CMAP

    @property
    def probability_cmap(self) -> str:
        return PROBABILITY_CMAP

    # kept so older calls keep working
    @property
    def c(self) -> Dict[str, str]:
        return {"ink": INK, "ink2": INK_SOFT, "muted": RULE, "grid": GRID,
                "axis": RULE, "surface": "white", "plane": "#f2f2f2",
                "accent": SERIES["reference"], **SERIES}

    def cmap(self, phenomenon: str = "neutral"):
        return DENSITY_CMAP


THEME = Theme("paper")


def use(context: str = "paper") -> Theme:
    """Install the publication defaults. `context` is "paper" or "talk"."""
    global THEME
    context = {"light": "paper", "dark": "paper"}.get(context, context)
    if context not in _CONTEXTS:
        raise ValueError(f"context must be one of {sorted(_CONTEXTS)}, got {context!r}")
    THEME = Theme(context)
    p = _CONTEXTS[context]
    b = p["base"]
    mpl.rcParams.update({
        # --- typography
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "Times New Roman", "Nimbus Roman",
                       "Liberation Serif", "DejaVu Serif"],
        "font.weight": "bold",
        "mathtext.fontset": "stix",
        "mathtext.default": "bf",
        "font.size": b,
        "axes.titlesize": b + 1,
        "axes.labelsize": b,
        "axes.labelweight": "bold",
        "xtick.labelsize": b - 1,
        "ytick.labelsize": b - 1,
        "legend.fontsize": b - 1,
        "figure.titlesize": b + 2,
        "figure.labelweight": "bold",
        # All figure text must remain legible after paper scaling.
        "axes.titleweight": "bold",
        "figure.titleweight": "bold",
        "axes.titlepad": 7.0,
        "axes.labelpad": 3.5,
        # --- axes and ticks: closed box, ticks inside, as in most ML papers
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "text.color": INK,
        "axes.linewidth": 0.8,
        "axes.grid": False,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "grid.linestyle": "-",
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.major.size": 3.5, "ytick.major.size": 3.5,
        "xtick.minor.size": 2.0, "ytick.minor.size": 2.0,
        "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "xtick.minor.width": 0.6, "ytick.minor.width": 0.6,
        "xtick.minor.visible": False,
        "ytick.minor.visible": False,
        # --- legend: thin boxed frame, no shadow, no rounded corners
        "legend.frameon": True,
        "legend.framealpha": 1.0,
        "legend.facecolor": "white",
        "legend.edgecolor": "0.6",
        "legend.fancybox": False,
        "legend.borderpad": 0.35,
        "legend.labelspacing": 0.3,
        "legend.handlelength": 1.6,
        "legend.handletextpad": 0.5,
        "legend.columnspacing": 1.1,
        # --- marks
        "lines.linewidth": p["lw"],
        "lines.markersize": 4.0,
        "lines.solid_capstyle": "round",
        "patch.linewidth": 0.6,
        # --- output
        "image.cmap": DENSITY_CMAP,
        "image.interpolation": "nearest",
        "figure.dpi": p["dpi"],
        "savefig.dpi": p["save_dpi"],
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "axes.prop_cycle": cycler(color=[SERIES["highnorm"], SERIES["sink"],
                                         SERIES["channel"], SERIES["reference"]]),
    })
    return THEME


def open_box(ax) -> None:
    """Left/bottom spines only -- for line plots, where a full box adds nothing."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(top=False, right=False)


def light_grid(ax, axis: str = "y") -> None:
    ax.grid(True, axis=axis, color=GRID, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)


def panel_label(ax, text: str, dx: float = -0.13, dy: float = 1.04, **kw) -> None:
    """(a), (b), ... in the conventional position above the axes' top-left corner."""
    ax.text(dx, dy, text, transform=ax.transAxes, ha="left", va="baseline",
            fontsize=mpl.rcParams["axes.titlesize"], fontweight="bold", **kw)


def colorbar(fig, mappable, ax, label: str, **kw):
    kw.setdefault("fraction", 0.046)
    kw.setdefault("pad", 0.02)
    cb = fig.colorbar(mappable, ax=ax, **kw)
    cb.set_label(label)
    cb.outline.set_linewidth(0.8)
    cb.ax.tick_params(width=0.8, direction="in")
    return cb


def boundary_line(ax, x, color: str = "white", label: bool = False, ls=(0, (4, 2))) -> None:
    """Mark the dual->single transition inside an image panel."""
    if x is None:
        return
    ax.axvline(x, color=color, linewidth=1.0, linestyle=ls, zorder=5)


def savefig(fig, path, **kw):
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, **kw)
    return path


# Deprecated helpers kept as no-ops so old call sites do not crash.
def tidy(ax, y_grid: bool = True, x_grid: bool = False) -> None:
    open_box(ax)
    if y_grid:
        light_grid(ax, "y")
    if x_grid:
        light_grid(ax, "x")


def footnote(fig, text: str, y: float = 0.0) -> None:      # noqa: ARG001
    """No-op: methodological notes belong in the LaTeX caption, not on the canvas."""
    return None
