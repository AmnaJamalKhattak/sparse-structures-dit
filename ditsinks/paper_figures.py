"""Figures made for the paper itself, drawn only from what the runs measured.

Figure 1 (``fig_teaser``)
    One generation per model, read at one layer and denoising step: the generated image,
    and three maps of the image tokens -- token norm over the layer median, the magnitude of
    the model's massive channel, and the attention each token receives -- with the high-norm
    tokens ringed in all of them. The second row is the distribution of |cos(x, v*)| over the
    same tokens, with the high-norm tokens marked. Every value comes from one traced
    generation (``capture_teaser``); the sentences printed under the maps are computed from
    those values, never typed.

Retiming grid (``fig_retiming_grid``)
    One prompt and seed per model from a Q16 run, with any chosen set of conditions. Column
    labels carry the layers each condition edits, read from the run's frozen protocol, and
    each panel carries the LPIPS its unit recorded.

Both write at the ICLR text width (5.5 in) in the house style of the other paper figures.

Figure 1's caption numbers (``teaser_numbers.tex``, ``write_teaser_tex``)
    Every number the caption states, as ``\\figone{model}{field}`` macros written from the
    same captures the figure is drawn from, as ``q16_numbers.tex`` is for Section 10. Models
    are ``flux``, ``schnell`` and ``pixart``; the fields are listed in ``TEASER_FIELDS``.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = ["TeaserCapture", "capture_teaser", "teaser_stats", "fig_teaser", "step_name",
           "along_across", "reference_image_difference", "TEASER_FIELDS", "latex_teaser_macros",
           "write_teaser_tex", "RetimingPanel", "RetimingRow", "retiming_row_from_unit",
           "fig_retiming_grid", "condition_label", "condition_layers", "resolve_conditions",
           "available_conditions", "unit_register_tokens", "MODEL_NAMES"]

# ---------------------------------------------------------------- style
INK, SOFT, MUTED = "#1a1a1a", "#4d4d4d", "#8a8a8a"
VSTAR = "#0072B2"                       # the high-norm tokens along v* (Okabe-Ito blue)
OTHER_TOKENS = "#8c8c8c"                # every other token: grey context
COLOURS = {"flux": "#0072B2", "pixart": "#D55E00"}
MAP_CMAP = "inferno"
WIDTH = 5.5                             # ICLR text width, inches
LABEL, SMALL = 7.0, 6.2                 # points

MODEL_NAMES = {"flux1-dev": ("FLUX.1-dev", "flux"), "flux1-schnell": ("FLUX.1-schnell", "flux"),
               "pixart-sigma-1024": ("PixArt-Σ", "pixart"),
               "pixart-sigma-512": ("PixArt-Σ", "pixart"),
               "tiny-flux1": ("Synthetic FLUX", "flux"), "tiny-pixart": ("Synthetic PixArt", "pixart")}


def _style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": LABEL, "axes.linewidth": 0.5,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 600,
        # ditsinks.style (the other figures) sets bold text; paper figures ask for bold
        # only where they want it, so a session that drew those first must not leak it here
        "font.weight": "normal", "axes.labelweight": "normal", "axes.titleweight": "normal",
        "figure.labelweight": "normal", "figure.titleweight": "normal",
        "mathtext.default": "it",
        # and must not crop: every paper figure is laid out at its exact printed size
        "savefig.bbox": None,
    })


class _Canvas:
    """A figure laid out in inches from its top-left corner, with an invisible overlay axes
    (``ink``) in the same coordinates for labels, brackets and marks between axes."""

    def __init__(self, width: float, height: float):
        import matplotlib.pyplot as plt

        _style()
        self.w, self.h = width, height
        self.fig = plt.figure(figsize=(width, height))
        self.ink = self.fig.add_axes([0, 0, 1, 1], zorder=10)
        self.ink.set_xlim(0, width)
        self.ink.set_ylim(height, 0)
        self.ink.axis("off")
        self.ink.patch.set_alpha(0)

    def axes(self, x: float, y: float, w: float, h: float, *, bare: bool = True):
        ax = self.fig.add_axes([x / self.w, 1 - (y + h) / self.h, w / self.w, h / self.h])
        if bare:
            ax.set_xticks([])
            ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.4)
            spine.set_color(SOFT)
        return ax

    def text(self, x: float, y: float, s: str, **kw):
        kw.setdefault("fontsize", LABEL)
        kw.setdefault("color", INK)
        return self.ink.text(x, y, s, **kw)

    def line(self, xs, ys, **kw):
        kw.setdefault("color", INK)
        kw.setdefault("lw", 0.5)
        return self.ink.plot(xs, ys, **kw)


def _display(checkpoint: str) -> Tuple[str, str]:
    return MODEL_NAMES.get(str(checkpoint), (str(checkpoint), "flux"))


# ================================================================ Figure 1: capture
@dataclass
class TeaserCapture:
    """What one generation shows at one layer and step, per image token.

    ``attention`` is, for every token, the largest share of image-to-image attention any head
    gives it (renormalised over image keys, averaged over image queries); ``cosine`` is the
    signed cosine of the token's residual state with v*.
    """

    checkpoint: str
    layer: int
    step: int
    prompt_id: int
    seed: int
    prompt: str
    channel: int
    channel_share: float            # (v*_c)^2 / ||v*||^2 for ``channel``
    grid: Tuple[int, int]
    norm: np.ndarray
    channel_values: np.ndarray      # |x_c|
    attention: np.ndarray
    cosine: np.ndarray
    image: Optional[np.ndarray] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def model(self) -> str:
        return _display(self.checkpoint)[0]

    @property
    def colour_key(self) -> str:
        return _display(self.checkpoint)[1]

    def maps(self) -> Dict[str, np.ndarray]:
        rows, cols = self.grid
        median = float(np.median(self.norm))
        return {"norm": (self.norm / max(median, 1e-12)).reshape(rows, cols),
                "channel": np.abs(self.channel_values).reshape(rows, cols),
                "attention": self.attention.reshape(rows, cols)}

    def save(self, path) -> Path:
        path = Path(path)
        meta = dict(checkpoint=self.checkpoint, layer=self.layer, step=self.step,
                    prompt_id=self.prompt_id, seed=self.seed, prompt=self.prompt,
                    channel=self.channel, channel_share=self.channel_share,
                    grid=list(self.grid), **self.meta)
        arrays = dict(norm=self.norm, channel_values=self.channel_values,
                      attention=self.attention, cosine=self.cosine,
                      meta=np.array(json.dumps(meta)))
        if self.image is not None:
            arrays["image"] = self.image
        np.savez_compressed(path, **arrays)
        return path

    @classmethod
    def load(cls, path) -> "TeaserCapture":
        with np.load(Path(path), allow_pickle=False) as data:
            meta = json.loads(str(data["meta"]))
            known = {"checkpoint", "layer", "step", "prompt_id", "seed", "prompt", "channel",
                     "channel_share", "grid"}
            return cls(checkpoint=meta["checkpoint"], layer=int(meta["layer"]),
                       step=int(meta["step"]), prompt_id=int(meta["prompt_id"]),
                       seed=int(meta["seed"]), prompt=str(meta["prompt"]),
                       channel=int(meta["channel"]), channel_share=float(meta["channel_share"]),
                       grid=tuple(int(g) for g in meta["grid"]),
                       norm=np.asarray(data["norm"], dtype=float),
                       channel_values=np.asarray(data["channel_values"], dtype=float),
                       attention=np.asarray(data["attention"], dtype=float),
                       cosine=np.asarray(data["cosine"], dtype=float),
                       image=np.asarray(data["image"]) if "image" in data else None,
                       meta={k: v for k, v in meta.items() if k not in known})


def capture_teaser(ctx, *, prompt_id: int, prompt: str, seed: int, layer: int, step: int,
                   channel: Optional[int] = None) -> TeaserCapture:
    """Run one unmodified generation and read every token at ``layer`` and ``step``.

    ``ctx`` is a :class:`~ditsinks.questions.QuestionContext` built from the model's frozen
    v* artifact, as the main runs build it. The channel defaults to the artifact's dominant
    register channel. Nothing is edited: the tracer only reads.
    """
    import torch

    from .causal_engine import CausalTracer, run_traced_generation

    channel = int(ctx.dominant_channel if channel is None else channel)
    # Only one layer is traced, so the tracer cannot learn the image-token count from the
    # layers before it; without the hint the attention of that layer would go unrecorded.
    tracer = CausalTracer(ctx.driver.adapter, ctx.driver.transformer, direction=ctx.vstar,
                          layers=[int(layer)], steps=[int(step)], channels=[channel],
                          grid=ctx.driver.grid, cfg=ctx.cfg, n_img_hint=ctx.driver.n_img)
    trace, _ = run_traced_generation(ctx.driver, tracer, prompt_id=int(prompt_id),
                                     prompt=prompt, seed=int(seed), save_image=True)
    row = trace.at(int(step), int(layer))
    if row is None or row.norm is None or row.cosine is None:
        raise RuntimeError(f"nothing was recorded at layer {layer}, step {step}: check that "
                           "the layer exists and the step is inside the sampler's range")
    if row.incoming is None:
        raise RuntimeError(f"no attention was recorded at layer {layer}, step {step}")
    v = ctx.vstar.detach().float().cpu()
    share = float(v[channel] ** 2 / v.pow(2).sum().clamp_min(1e-12))
    image = trace.image
    if image is not None and hasattr(image, "convert"):
        image = np.asarray(image.convert("RGB"))
    grid = tuple(int(g) for g in (trace.grid or ctx.driver.grid))
    return TeaserCapture(
        checkpoint=str(ctx.cfg.model), layer=int(layer), step=int(step),
        prompt_id=int(prompt_id), seed=int(seed), prompt=str(prompt), channel=channel,
        channel_share=share, grid=grid,
        norm=row.norm.float().numpy(),
        channel_values=row.channel_values[0].float().abs().numpy(),
        attention=row.incoming.float().max(dim=0).values.numpy(),
        cosine=row.cosine.float().numpy(),
        image=image,
        meta=dict(n_heads=int(row.incoming.shape[0]),
                  guidance=getattr(ctx.cfg, "guidance_scale", None),   # None: model default
                  steps=int(getattr(ctx.cfg, "num_inference_steps", 0) or 0)))


def step_name(step: int, steps: int = 0) -> str:
    """How a figure names a denoising step: 'final step' for the last one, else 'step s of S'
    counted from 1 (the runs count from 0)."""
    if steps and int(step) == int(steps) - 1:
        return "final step"
    return f"step {int(step) + 1} of {int(steps)}" if steps else f"step {int(step) + 1}"


def teaser_stats(capture: TeaserCapture, *, ratio: float = 3.0) -> Dict[str, Any]:
    """The numbers Figure 1 and its caption state, from the captured values alone.

    High-norm tokens are the tokens at least ``ratio`` times the layer's median norm; the
    overlap is how many of them are also among the same number of largest tokens of the
    channel and attention maps.
    """
    ratio_map = capture.norm / max(float(np.median(capture.norm)), 1e-12)
    high = np.flatnonzero(ratio_map >= ratio)
    k = max(len(high), 1)

    def top(values):
        return set(np.argsort(np.nan_to_num(values, nan=-np.inf))[::-1][:k].tolist())

    shared = set(high.tolist()) & top(np.abs(capture.channel_values)) & top(capture.attention)
    abs_cos = np.abs(capture.cosine)
    ordinary = np.setdiff1d(np.arange(abs_cos.size), high)
    uniform = 1.0 / max(capture.attention.size, 1)
    return dict(
        model=capture.model, checkpoint=capture.checkpoint, layer=capture.layer,
        step=capture.step, steps=int(capture.meta.get("steps", 0) or 0),
        when=step_name(capture.step, int(capture.meta.get("steps", 0) or 0)), ratio=float(ratio),
        prompt_id=capture.prompt_id, seed=capture.seed, n_tokens=int(abs_cos.size),
        n_high_norm=int(len(high)), n_shared=int(len(shared)),
        high_norm_tokens=high.tolist(), shared_tokens=sorted(shared),
        max_norm_ratio=float(ratio_map.max()),
        high_norm_abs_cos_median=float(np.median(abs_cos[high])) if len(high) else float("nan"),
        high_norm_abs_cos_min=float(abs_cos[high].min()) if len(high) else float("nan"),
        ordinary_abs_cos_median=float(np.median(abs_cos[ordinary])) if len(ordinary)
        else float("nan"),
        ordinary_abs_cos_q99=float(np.quantile(abs_cos[ordinary], 0.99)) if len(ordinary)
        else float("nan"),
        high_norm_attention_x_uniform=float(np.median(capture.attention[high]) / uniform)
        if len(high) else float("nan"),
        high_norm_ratio_median=float(np.median(ratio_map[high])) if len(high) else float("nan"),
        # the angle to the axis v* spans, as the arrows of row 2 show it (0-90 degrees)
        high_norm_angle_median=_degrees(np.median(abs_cos[high])) if len(high) else float("nan"),
        ordinary_angle_median=_degrees(np.median(abs_cos[ordinary])) if len(ordinary)
        else float("nan"),
        channel=capture.channel, channel_share=float(capture.channel_share))


def _degrees(abs_cos: float) -> float:
    return float(np.degrees(np.arccos(np.clip(float(abs_cos), 0.0, 1.0))))


def along_across(capture: TeaserCapture) -> Tuple[np.ndarray, np.ndarray]:
    """Each token's state split against v*, in units of the layer's median token norm: the
    signed component along v* and the length of the rest of the state. The two are the legs
    of a right triangle whose hypotenuse is the token's norm, so an arrow drawn to
    (along, across) on equal axes has the token's true length and its true angle to v*."""
    median = max(float(np.median(capture.norm)), 1e-12)
    ratio = np.asarray(capture.norm, dtype=float) / median
    cos = np.clip(np.asarray(capture.cosine, dtype=float), -1.0, 1.0)
    return ratio * cos, ratio * np.sqrt(1.0 - cos ** 2)


def reference_image_difference(capture: TeaserCapture, run_roots: Sequence) -> Optional[Dict]:
    """How far the captured image is from the unmodified image of the same prompt and seed in
    a Q16 run (``units/p<P>_s<S>/images/reference.png``): the check that the capture is the
    generation the paper's other numbers were measured on. Pixel values 0-255; None when no
    run holds that unit or the capture has no image."""
    from PIL import Image

    from .q16_main import unit_directory

    if capture.image is None:
        return None
    for root in run_roots:
        path = unit_directory(root, capture.prompt_id, capture.seed) / "images" / "reference.png"
        if path.exists():
            with Image.open(path) as image:
                image = image.convert("RGB")
                ours = np.asarray(capture.image)[..., :3]
                if image.size != (ours.shape[1], ours.shape[0]):
                    image = image.resize((ours.shape[1], ours.shape[0]), Image.LANCZOS)
                diff = np.abs(np.asarray(image, float) - ours.astype(float))
            return dict(path=str(path), mean_abs=float(diff.mean()), max_abs=float(diff.max()))
    return None


# ================================================================ Figure 1: caption numbers
# field -> (stats key, how it is printed)
TEASER_FIELDS = {
    "layer": ("layer", "int"), "step": ("step", "int"), "steps": ("steps", "int"),
    "when": ("when", "text"), "prompt": ("prompt_id", "int"), "seed": ("seed", "int"),
    "ratio": ("ratio", "ratio"),
    "tokens": ("n_tokens", "int"), "highnorm": ("n_high_norm", "int"),
    "shared": ("n_shared", "int"), "maxratio": ("max_norm_ratio", "1"),
    "coshigh": ("high_norm_abs_cos_median", "3"), "coshighmin": ("high_norm_abs_cos_min", "3"),
    "cosother": ("ordinary_abs_cos_median", "3"), "cosotherq": ("ordinary_abs_cos_q99", "3"),
    "attnx": ("high_norm_attention_x_uniform", "0"), "channel": ("channel", "int"),
    "normhigh": ("high_norm_ratio_median", "1"), "anglehigh": ("high_norm_angle_median", "0"),
    "angleother": ("ordinary_angle_median", "0"),
    "share": ("channel_share", "pct"),
}
_TEASER_HEADER = r"""% Generated by ditsinks.paper_figures -- do not edit by hand; rerun notebook cell 3.3.
\makeatletter
\providecommand{\figone}[2]{\@ifundefined{qfig@#1@#2}{\textbf{??}}{\@nameuse{qfig@#1@#2}}}
"""


def _teaser_value(value, how: str) -> str:
    if how == "text":
        return str(value) if value else "n/a"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(value):
        return "n/a"
    if how == "int":
        return str(int(round(value)))
    if how == "pct":
        return f"{100 * value:.1f}\\%"
    if how == "ratio":
        return f"{value:g}"
    return f"{value:.{int(how)}f}"


def latex_teaser_macros(stats: Sequence[Dict[str, Any]]) -> str:
    """Every number of Figure 1 as ``\\figone{model}{field}`` (``TEASER_FIELDS``), from
    ``teaser_stats`` of the captures the figure is drawn from."""
    from .q16_paper import MODEL_KEYS

    lines = [_TEASER_HEADER.rstrip("\n")]
    for s in stats:
        model = MODEL_KEYS.get(str(s["checkpoint"]), str(s["checkpoint"]).replace("_", "-"))
        for name, (key, how) in TEASER_FIELDS.items():
            lines.append(f"\\@namedef{{qfig@{model}@{name}}}{{{_teaser_value(s.get(key), how)}}}")
    lines.append("\\makeatother")
    return "\n".join(lines) + "\n"


def write_teaser_tex(stats: Sequence[Dict[str, Any]], out_dir) -> Path:
    """Write ``teaser_numbers.tex`` (upload it to the Overleaf root beside main.tex)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "teaser_numbers.tex"
    path.write_text(latex_teaser_macros(stats))
    return path


# ================================================================ Figure 1: drawing
def _rings(ax, tokens, grid, *, image_extent: bool, lw: float = 0.45) -> None:
    """Ring each token. The radius is a fixed share of the map's width (never less than a
    token), so the rings read the same on a 64 x 64 map and a coarse one. Three tokens or
    fewer also get an arrow from the map's centre side, so that a single token -- PixArt-Sigma's
    register, in a corner -- is found at a glance."""
    from matplotlib.patches import Circle

    rows, cols = grid
    radius = max(0.6, 0.035 * cols)                  # in tokens
    scale = 1.0 / cols if image_extent else 1.0
    tokens = [int(t) for t in tokens]
    for t in tokens:
        r, c = divmod(t, cols)
        x, y = ((c + 0.5) / cols, (r + 0.5) / rows) if image_extent else (c, r)
        ax.add_patch(Circle((x, y), radius * scale, fill=False, lw=lw, ec="white",
                            alpha=0.95 if image_extent else 0.85))
        if len(tokens) <= 3:
            toward = np.array([cols / 2 - c, rows / 2 - r], dtype=float)
            toward /= max(float(np.linalg.norm(toward)), 1e-9)
            head = np.array([x, y]) + toward * radius * scale * 1.15
            tail = head + toward * 0.22 * cols * scale
            ax.annotate("", xy=tuple(head), xytext=tuple(tail),
                        arrowprops=dict(arrowstyle="-|>", color="white", lw=0.7,
                                        mutation_scale=5, shrinkA=0, shrinkB=0))


def _same_sentence(n_high: int, n_shared: int, ratio: float = 3.0) -> str:
    """What the bracket over the three maps says, from the counts ``teaser_stats`` gives."""
    if n_high == 0:
        return f"no token reaches {ratio:g}× the median norm"
    if n_shared == n_high:
        return ("the same token in all three" if n_high == 1
                else f"the same {n_high} tokens in all three")
    if n_high == 1:
        return "the high-norm token does not lead all three"
    return f"{n_shared} of the {n_high} high-norm tokens lead{'s' if n_shared == 1 else ''} all three"


def _teaser_maps(cv, cap: TeaserCapture, stats: Dict[str, Any], x0: float, top: float,
                 tile: float, gap: float, gamma: float) -> None:
    """The image and the norm, channel and attention maps of one capture, left to right
    from ``x0``; the high-norm tokens ringed in all four."""
    from matplotlib.colors import PowerNorm

    maps = cap.maps()
    high = stats["high_norm_tokens"]
    ax = cv.axes(x0, top, tile, tile)
    if cap.image is not None:
        ax.imshow(cap.image, extent=(0, 1, 1, 0), interpolation="lanczos")
        ax.set_xlim(0, 1)
        ax.set_ylim(1, 0)
        _rings(ax, high, cap.grid, image_extent=True)
    for j, name in enumerate(("norm", "channel", "attention"), start=1):
        ax = cv.axes(x0 + j * (tile + gap), top, tile, tile)
        values = maps[name]
        finite = values[np.isfinite(values)]
        ax.imshow(values, cmap=MAP_CMAP, interpolation="nearest",
                  norm=PowerNorm(gamma, vmin=float(finite.min()), vmax=float(finite.max())))
        _rings(ax, high, cap.grid, image_extent=False)


def _teaser_bracket(cv, stats: Dict[str, Any], left: float, right: float, by: float,
                    ratio: float) -> None:
    """What the three maps share, computed: a bracket under them and one sentence."""
    cv.line([left, left, right, right], [by - 0.03, by, by, by - 0.03], color=SOFT, lw=0.5)
    cv.text((left + right) / 2, by + 0.14, _same_sentence(stats["n_high_norm"],
                                                          stats["n_shared"], ratio),
            ha="center", va="baseline", fontsize=SMALL, fontweight="bold")


def _teaser_arrows(cv, ax, along: np.ndarray, across: np.ndarray, high, *, units: float,
                   x_start: float, width: float, height: float, free_band: float,
                   below_axis: float, xlabel: bool) -> None:
    """Every token as an arrow from one origin to (along v*, the rest), on equal axes at
    ``units`` median token norms per inch: true lengths and true angles to v*. Only the
    high-norm tokens are blue; both axes are dark grey and everything is labelled where it
    is drawn."""
    from matplotlib.collections import LineCollection

    renderer = cv.fig.canvas.get_renderer()
    ordinary = np.setdiff1d(np.arange(along.size), high)
    y_start = -below_axis * units * height
    x_end, y_end = x_start + units * width, y_start + units * height
    ax.set_xlim(x_start, x_end)
    ax.set_ylim(y_start, y_end)
    segments = np.zeros((len(ordinary), 2, 2))
    segments[:, 1, 0], segments[:, 1, 1] = along[ordinary], across[ordinary]
    ax.add_collection(LineCollection(segments, colors=OTHER_TOKENS, linewidths=0.35,
                                     alpha=0.2, rasterized=True, zorder=2))
    for t in high:
        tip = (float(along[t]), float(across[t]))
        ax.plot([0.0, tip[0]], [0.0, tip[1]], color=VSTAR, lw=0.75, zorder=4,
                solid_capstyle="butt")
        ax.annotate("", xy=tip, xytext=(0.9 * tip[0], 0.9 * tip[1]),
                    arrowprops=dict(arrowstyle="-|>", color=VSTAR, lw=0.75,
                                    mutation_scale=5, shrinkA=0, shrinkB=0), zorder=4)
    # the two axes, both dark grey so that only token arrows are blue: v* (labelled at
    # its arrowhead) and the orthogonal rest
    ax.spines["bottom"].set_position(("data", 0.0))
    ax.spines["bottom"].set_color(SOFT)
    ax.spines["bottom"].set_linewidth(0.5)
    ax.annotate("", xy=(x_end, 0.0), xytext=(x_end - 0.02 * units * width, 0.0),
                arrowprops=dict(arrowstyle="-|>", color=SOFT, lw=0.5, mutation_scale=5,
                                shrinkA=0, shrinkB=0), annotation_clip=False, zorder=5)
    ax.text(x_end + 0.015 * units * width, 0.0, "$v^\\star$", ha="left", va="center",
            fontsize=LABEL + 1, color=INK, zorder=6, clip_on=False)
    y_top = y_start + (1.0 - 0.5 * free_band) * (y_end - y_start)
    ax.spines["left"].set_position(("data", 0.0))
    ax.spines["left"].set_bounds(0.0, y_top)
    ax.spines["left"].set_color(SOFT)
    ax.spines["left"].set_linewidth(0.5)
    ax.annotate("", xy=(0.0, y_top), xytext=(0.0, y_top - 0.02 * units * width),
                arrowprops=dict(arrowstyle="-|>", color=SOFT, lw=0.5, mutation_scale=5,
                                shrinkA=0, shrinkB=0), annotation_clip=False, zorder=5)
    ax.text(-0.012 * units * width, y_top, "$\\perp\\!v^\\star$", ha="right", va="center",
            fontsize=SMALL + 0.4, color=SOFT)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_yticks([])
    ax.set_xticks([t for t in range(0, int(x_end) + 1, 5)])
    ax.tick_params(axis="x", labelsize=SMALL - 0.6, length=2, width=0.4, pad=1.2, colors=SOFT)
    if xlabel:
        ax.set_xlabel("along $v^\\star$, in median token norms", fontsize=SMALL, labelpad=1.5)
    # labels where the marks are, instead of a key
    small = SMALL - 0.4
    fan_top = float(np.quantile(across[ordinary], 0.99)) if len(ordinary) else 0.0
    ax.text(0.03 * units * width, fan_top + 0.06 * units * height, "other tokens",
            ha="left", va="bottom", fontsize=small, color=SOFT, zorder=6)
    if len(high):
        name = "high-norm token" if len(high) == 1 else "high-norm tokens"
        tips_x, tips_y = along[high], across[high]
        label = ax.text(0, 0, name, fontsize=small, color=INK, zorder=6)
        size = label.get_window_extent(renderer).width / cv.fig.dpi * units
        reach = float(tips_x.max()) + 0.03 * units * width
        if reach + size <= x_end - 0.02 * units * width:
            # room beyond the longest arrow: label it there, level with its tip
            label.set_position((reach, float(tips_y[np.argmax(tips_x)])))
            label.set_ha("left")
            label.set_va("center")
        else:
            # otherwise above the arrows, over the middle of their tips
            label.set_position((float(np.median(tips_x)),
                                float(tips_y.max()) + 0.07 * units * height))
            label.set_ha("center")
            label.set_va("bottom")


STACKED_WIDTH = 0.535 * WIDTH       # Figure 1's width: the caption takes the rest of the line


def fig_teaser(captures: Sequence[TeaserCapture], *, ratio: float = 3.0, gamma: float = 0.6,
               layout: str = "stacked", scale_with: Sequence[TeaserCapture] = (),
               width: Optional[float] = None):
    """Figure 1: for each model, the image and three maps of the same tokens (first row),
    and every token as an arrow against v* (second row, ``along_across``).

    ``layout='stacked'`` puts the models one above the other (FLUX.1-dev first), each block
    a row of maps over its row of arrows, ``STACKED_WIDTH`` wide so that the caption can sit
    beside it; the arrow panels share one x axis, so lengths compare directly. ``'columns'``
    puts the two blocks side by side across the full text width.

    Each map is scaled from its own minimum (black) to its maximum (yellow), with a power
    norm (``gamma``) so that mid-range values stay visible. The arrows of all models share
    one scale, in median token norms, and equal axes, so lengths and angles are true.
    Colour carries one meaning: blue is the high-norm tokens, grey the other tokens, and
    both axes are dark grey (checked colour-blind-safe against the densest grey). The
    numbers are in the caption (``latex_teaser_macros``).

    ``scale_with``: further captures that set the arrow scale without being drawn, so that a
    figure of one model (Figure 1) and one of the other (the appendix) share one scale.
    ``width`` (stacked only, inches): the figure's width; ``WIDTH`` draws one model across the
    full text width at print size (default ``STACKED_WIDTH``)."""
    captures = list(captures)
    if not captures:
        raise ValueError("fig_teaser needs at least one capture")
    if layout not in ("stacked", "columns"):
        raise ValueError("layout is 'stacked' or 'columns'")
    stats_all = [teaser_stats(c, ratio=ratio) for c in captures]
    geometry = [along_across(c) for c in captures]
    scaled = geometry + [along_across(c) for c in scale_with]
    # Every panel starts its x axis at the same place, so the panels line up.
    lo = min(0.0, min(float(a.min()) for a, _ in scaled))
    free_band, below_axis = 0.25, 0.06              # shares of the arrow panel's height
    gap = 0.035
    n = len(captures)

    # Each model is one block: its title, its image and maps, and its arrows under them.
    # The blocks are the same in both layouts, only placed differently.
    if layout == "stacked":
        width, group_gap, block_gap = width or STACKED_WIDTH, 0.0, 0.06
        group_w = width
    else:
        width, group_gap, block_gap = WIDTH, 0.2, 0.0
        group_w = (width - (n - 1) * group_gap) / n
    tile = (group_w - 3 * gap) / 4
    top0, arrow_left = 0.3, 0.25
    strip0 = top0 + tile + 0.3
    arrow_w, arrow_h = group_w - 0.5, 0.46
    block_h = strip0 + arrow_h + 0.27
    height = block_h if layout == "columns" else n * block_h + (n - 1) * block_gap
    # One scale for every model (median token norms per inch), chosen so that the longest
    # arrow fits and every arrow stays below the band kept free for the labels.
    units = max(max((max(0.0, float(a.max())) - lo) * 1.1 / arrow_w,
                    float(b.max()) / ((1.0 - free_band - below_axis) * arrow_h))
                for a, b in scaled)
    units = max(units, 1e-9)
    x_start = lo - 0.04 * units * arrow_w
    cv = _Canvas(width, height)
    arrows = dict(units=units, x_start=x_start, width=arrow_w, height=arrow_h,
                  free_band=free_band, below_axis=below_axis)

    for g, (cap, stats) in enumerate(zip(captures, stats_all)):
        x0, y0 = ((g * (group_w + group_gap), 0.0) if layout == "columns"
                  else (0.0, g * (block_h + block_gap)))
        top = y0 + top0
        cv.text(x0 + group_w / 2, y0 + 0.09, f"{cap.model}  ·  layer {cap.layer}",
                ha="center", va="baseline", color=INK, fontweight="bold")
        for j, title in enumerate(["image", "token norm", f"channel {cap.channel}",
                                   "attention"]):
            cv.text(x0 + j * (tile + gap) + tile / 2, top - 0.045, title, ha="center",
                    va="baseline", fontsize=SMALL)
        _teaser_maps(cv, cap, stats, x0, top, tile, gap, gamma)
        _teaser_bracket(cv, stats, x0 + tile + gap, x0 + group_w, top + tile + 0.06, ratio)
        ax = cv.axes(x0 + arrow_left, y0 + strip0, arrow_w, arrow_h, bare=False)
        _teaser_arrows(cv, ax, *geometry[g], stats["high_norm_tokens"], xlabel=True,
                       **arrows)
    fig = cv.fig
    fig.teaser_stats = stats_all
    return fig


# ================================================================ retiming grid
CONDITION_LABELS = {
    "reference": "original", "hooks_only": "hooks only", "induce_N": "in place",
    "extend": "kept past its end", "remove": "removed",
    "control_random_direction": "random direction",
    "control_ordinary_positions": "other positions",
    "control_in_distribution": "in-distribution $v^\\star$",
}


def condition_label(key: str) -> str:
    """The short name a grid column prints for a Q16 condition."""
    key = str(key)
    if key in CONDITION_LABELS:
        return CONDITION_LABELS[key]
    if key.startswith("remove_induce_"):
        return "induced"                  # the header's second line gives the layers
    if key.startswith("induce_E"):
        return f"early ({key[len('induce_'):]})"
    if key.startswith("induce_L"):
        return f"late ({key[len('induce_'):]})"
    return key.replace("_", " ")


def _span(a: int, b: int, word: str = "layers") -> str:
    return f"{word} {a}–{b}" if int(a) != int(b) else f"{word[:-1]} {a}"


def condition_layers(protocol, key: str) -> str:
    """The layers a condition edits, from the run's frozen protocol ('' for the reference)."""
    from .q16_main import condition_catalog

    catalog = {c.key: c for c in condition_catalog(protocol, include_check=True)}
    c = catalog.get(str(key))
    if c is None or c.group == "reference":
        return ""
    parts = []
    if c.remove:
        a, b = protocol.windows["removal"]
        parts.append(_span(a, b))
    if c.window and c.window in protocol.windows:
        a, b = protocol.windows[c.window]
        parts.append(("then " if c.remove else "") + _span(a, b))
    return ", ".join(parts)


def condition_header(protocol, key: str) -> str:
    """The second line of a grid column, short enough for a 0.6 in panel: the layers a
    condition writes ('layers 5–8'); 'natural registers' under the removal ('removed');
    and, for a removal followed by a write elsewhere ('induced'), only the layers of the
    write, the removal being the removal column's."""
    from .q16_main import condition_catalog

    key = str(key)
    catalog = {c.key: c for c in condition_catalog(protocol, include_check=True)}
    c = catalog.get(key)
    if c is not None and c.remove:
        if c.window and c.window in protocol.windows:
            return _span(*protocol.windows[c.window])
        return "natural registers"
    return condition_layers(protocol, key)


def _fit_width(text, width: float, *, smallest: float = 5.0) -> None:
    """Shrink a header until it is no wider than ``width`` inches (never below
    ``smallest`` points), so a long custom label cannot run into its neighbours."""
    fig = text.get_figure()
    renderer = getattr(fig.canvas, "get_renderer", lambda: None)()
    if renderer is None:
        return
    size = float(text.get_fontsize())
    while size > smallest and text.get_window_extent(renderer).width / fig.dpi > width:
        size = max(smallest, size - 0.2)
        text.set_fontsize(size)


def resolve_conditions(protocol, names: Sequence[str]) -> List[str]:
    """The run's condition keys for a grid: each name is a condition key (``induce_E1``,
    ``remove``) or a window of the protocol (``E1``, ``N``, ``L2``), which means the
    condition that writes the register in that window."""
    from .q16_main import condition_catalog

    keys = [c.key for c in condition_catalog(protocol, include_check=False)]
    out, unknown = [], []
    for name in names:
        name = str(name)
        key = name if name in keys else f"induce_{name}"
        (out if key in keys else unknown).append(key if key in keys else name)
    if unknown:
        windows = ", ".join(k for k in protocol.windows if k != "removal")
        raise ValueError(f"not conditions of this run: {unknown}. Conditions: {keys}; "
                         f"windows: {windows}")
    return out


def available_conditions(protocol, run_root=None, unit: Optional[Tuple[int, int]] = None
                         ) -> List[Dict[str, Any]]:
    """Every condition of a protocol with its label and layers, and (with a unit) whether its
    image is on disk -- the menu the notebook shows before a grid is chosen."""
    from .q16_main import condition_catalog, unit_directory

    rows = []
    for c in condition_catalog(protocol, include_check=False):
        row = dict(condition=c.key, label=condition_label(c.key),
                   layers=condition_layers(protocol, c.key))
        if run_root is not None and unit is not None:
            folder = unit_directory(run_root, *unit)
            row["on_disk"] = ((folder / "images" / f"{c.key}.png").exists()
                              or (folder / "thumbnails" / f"{c.key}.jpg").exists())
        rows.append(row)
    return rows


def unit_register_tokens(run_root, prompt_id: int, seed: int) -> Optional[int]:
    """How many tokens carry the register in one unit's unmodified run: the median over its
    recorded steps of the natural carriers the main run matched (``register_match.csv``)."""
    import pandas as pd

    from .q16_main import unit_directory

    path = unit_directory(run_root, prompt_id, seed) / "register_match.csv"
    if not path.exists():
        return None
    table = pd.read_csv(path)
    if "n_natural_carriers" not in table or table.empty:
        return None
    keys = [c for c in ("prompt_id", "seed", "step") if c in table]
    per_step = table.drop_duplicates(keys) if keys else table
    values = per_step["n_natural_carriers"].dropna()
    return int(round(float(values.median()))) if len(values) else None


@dataclass
class RetimingPanel:
    key: str
    label: str
    layers: str
    image: np.ndarray
    lpips: Optional[float] = None


@dataclass
class RetimingRow:
    model: str
    colour_key: str
    subtitle: str
    panels: List[RetimingPanel]


def _read_image(folder: Path, key: str, source: str, size: int) -> np.ndarray:
    from PIL import Image

    candidates = ([folder / "images" / f"{key}.png", folder / "thumbnails" / f"{key}.jpg"]
                  if source == "images" else
                  [folder / "thumbnails" / f"{key}.jpg", folder / "images" / f"{key}.png"])
    for path in candidates:
        if path.exists():
            with Image.open(path) as image:
                image = image.convert("RGB")
                if size and max(image.size) > size:
                    image = image.resize((size, size), Image.LANCZOS)
                return np.asarray(image)
    raise FileNotFoundError(f"no image for condition {key!r} in {folder}")


def retiming_row_from_unit(run_root, protocol, *, prompt_id: int, seed: int,
                           conditions: Sequence[str], labels: Optional[Dict[str, str]] = None,
                           subtitle: str = "", image_source: str = "images", size: int = 512,
                           metric: str = "lpips") -> RetimingRow:
    """One model's row of the grid, read from its Q16 unit folder.

    Layers come from the frozen protocol, the metric from the unit's ``images.csv``, the
    images from ``images/`` (full size, downsampled to ``size``) or ``thumbnails/``.
    """
    import pandas as pd

    from .q16_main import unit_directory

    folder = unit_directory(run_root, prompt_id, seed)
    if not folder.exists():
        raise FileNotFoundError(f"no unit folder {folder}")
    values: Dict[str, float] = {}
    table_path = folder / "images.csv"
    if table_path.exists():
        table = pd.read_csv(table_path)
        if metric in table:
            values = {str(k): float(v) for k, v in
                      table.groupby("condition")[metric].median().items()}
    labels = dict(labels or {})
    panels = [RetimingPanel(key=str(k), label=labels.get(k, condition_label(k)),
                            layers=condition_header(protocol, k),
                            image=_read_image(folder, str(k), image_source, size),
                            lpips=values.get(str(k)))
              for k in conditions]
    model, colour = _display(protocol.checkpoint)
    return RetimingRow(model=model, colour_key=colour, subtitle=subtitle, panels=panels)


def fig_retiming_grid(rows: Sequence[RetimingRow], *, layout: str = "auto",
                      metric_name: str = "LPIPS"):
    """The retiming examples: ``layout='groups'`` puts the models side by side in one row,
    ``'rows'`` gives each model its own row; ``'auto'`` picks groups while every panel stays
    at least 0.6 in wide."""
    rows = list(rows)
    if not rows:
        raise ValueError("fig_retiming_grid needs at least one row")
    gap, group_gap = 0.04, 0.16
    counts = [len(r.panels) for r in rows]
    side_tile = (WIDTH - (len(rows) - 1) * group_gap - sum(n - 1 for n in counts) * gap) \
        / max(sum(counts), 1)
    if layout == "auto":
        layout = "groups" if side_tile >= 0.6 else "rows"
    if layout not in ("groups", "rows"):
        raise ValueError("layout is 'auto', 'groups' or 'rows'")
    if layout == "groups":
        tile = side_tile
        header = 0.44
        cv = _Canvas(WIDTH, header + tile + 0.02)
        x0 = 0.0
        placements = []
        for row, n in zip(rows, counts):
            placements.append((row, x0, header))
            x0 += n * tile + (n - 1) * gap + group_gap
    else:
        n_max = max(counts)
        tile = (WIDTH - (n_max - 1) * gap) / n_max
        header = 0.44
        row_h = header + tile + 0.08
        cv = _Canvas(WIDTH, row_h * len(rows))
        placements = [(row, 0.0, i * row_h + header) for i, row in enumerate(rows)]
    for row, x0, top in placements:
        n = len(row.panels)
        group_w = n * tile + (n - 1) * gap
        title = row.model + (f"  ·  {row.subtitle}" if row.subtitle else "")
        cv.text(x0 + group_w / 2, top - 0.33, title, ha="center", va="baseline",
                color=COLOURS.get(row.colour_key, INK), fontweight="bold")
        for j, panel in enumerate(row.panels):
            x = x0 + j * (tile + gap)
            _fit_width(cv.text(x + tile / 2, top - 0.15, panel.label, ha="center",
                               va="baseline", fontsize=LABEL if tile >= 0.6 else SMALL),
                       tile + gap / 2)
            if panel.layers:
                _fit_width(cv.text(x + tile / 2, top - 0.05, panel.layers, ha="center",
                                   va="baseline", fontsize=SMALL - (0.6 if tile < 0.6 else 0),
                                   color=SOFT), tile + gap / 2)
            ax = cv.axes(x, top, tile, tile)
            ax.imshow(panel.image, interpolation="lanczos")
            if panel.lpips is not None and panel.key != "reference":
                cv.text(x + 0.03, top + tile - 0.035, f"{panel.lpips:.2f}", fontsize=SMALL,
                        color="white", va="baseline", ha="left", fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.18,rounding_size=0.08",
                                  fc=(0, 0, 0, 0.72), ec="none"))
    cv.fig.metric_name = metric_name
    return cv.fig


# ================================================================ direction at fixed norm (Q13)
DEPTH_COLOURS = {18: "#8fcfb4", 20: "#2f9e76", 24: "#00553b"}
ROTATION_COLOURS = {"ordinary_mean": "#CC79A7", "random_orthogonal": "#56B4E9",
                    "ordinary_pc1": "#E69F00"}
ROTATION_LABELS = {"ordinary_mean": "toward the ordinary-token mean",
                   "random_orthogonal": "toward a random orthogonal direction",
                   "ordinary_pc1": "toward the ordinary tokens' first PC"}


def rotation_retention_table(population) -> "pd.DataFrame":
    """Per rotation target, angle and prompt: the share of affected heads that keep their
    clean sink (``sink_retention``) and the treated tokens' sink strength, averaged over the
    register-zone layers after the edit, from a Q13 rotation run's ``population_metrics``."""
    import pandas as pd

    rows = population[population["arm"] == "rotation"] if "arm" in population else population
    edited = rows["is_edited_layer"].eq(True)
    if not edited.any():
        raise ValueError("population has no edited layer (is_edited_layer)")
    edit_layer = int(rows.loc[edited, "layer"].min())
    zone = rows[rows["is_register_zone"].eq(True) & (rows["layer"] > edit_layer)]
    keys = ["rotation_target", "theta_deg", "prompt_id", "seed"]
    table = zone.groupby(keys, observed=True).agg(
        sink_retention=("sink_retention", "mean"),
        sink_strength=("selected_sink_strength_mean", "mean"),
        first_layer=("layer", "min"), last_layer=("layer", "max")).reset_index()
    table["edit_layer"] = edit_layer
    return pd.DataFrame(table)


def fig_direction_at_fixed_norm(depth, rotation, *, sink_threshold: float = 10.0):
    """Sinkhood follows alignment with v* when the norm is held fixed.

    (a) ``depth``: a Q13 ``depth_response`` table -- the v* component of the treated
    tokens scaled by beta and the token renormalised to its own norm, one row per
    (layer, beta); sink strength read at the edited layer. (b) ``rotation``:
    ``rotation_retention_table`` -- the state rotated off v* by theta in the plane of v*
    and a target direction, norm exact, read over the register zone downstream."""
    import matplotlib.pyplot as plt

    _style()
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(WIDTH, 1.95),
                                     gridspec_kw=dict(wspace=0.42, left=0.085, right=0.985,
                                                      top=0.86, bottom=0.21))
    # (a) alignment at fixed norm
    for layer, group in depth.groupby("layer"):
        group = group.sort_values("beta")
        colour = DEPTH_COLOURS.get(int(layer), SOFT)
        ax_a.plot(group["beta"], group["sink_strength"], color=colour, lw=1.3, marker="o",
                  ms=3.6, mec="white", mew=0.5, zorder=3, label=f"layer {int(layer)}")
    ax_a.axhline(sink_threshold, color=MUTED, lw=0.7, ls=(0, (3, 2)), zorder=1)
    ax_a.text(2.05, sink_threshold * 1.12, "sink threshold", fontsize=SMALL - 0.6,
              color=MUTED, ha="right", va="bottom")
    ax_a.axvline(1.0, color="#d9d9d9", lw=0.8, zorder=0)
    ax_a.text(1.0, 330, "unmodified", fontsize=SMALL - 0.6, color=MUTED, ha="center",
              va="bottom")
    ax_a.set_yscale("log")
    ax_a.set_ylim(0.5, 320)
    ax_a.set_xticks(sorted(depth["beta"].unique()))
    ax_a.set_xticklabels([f"{b:g}" for b in sorted(depth["beta"].unique())])
    ax_a.set_xlabel(r"$\beta$: $v^\star$ component scaled, norm fixed", fontsize=SMALL,
                    labelpad=1.5)
    ax_a.set_ylabel("sink strength of the\nedited tokens (× uniform)", fontsize=SMALL,
                    labelpad=1.5)
    ax_a.legend(loc="lower right", fontsize=SMALL - 0.6, frameon=False, handlelength=1.6,
                borderaxespad=0.2, labelspacing=0.2, numpoints=1)

    # (b) rotation off v* at fixed norm
    summary = rotation.groupby(["rotation_target", "theta_deg"]).agg(
        mean=("sink_retention", "mean"), low=("sink_retention", "min"),
        high=("sink_retention", "max")).reset_index()
    for target in ("ordinary_pc1", "random_orthogonal", "ordinary_mean"):
        group = summary[summary["rotation_target"] == target].sort_values("theta_deg")
        if group.empty:
            continue
        colour = ROTATION_COLOURS.get(target, SOFT)
        ax_b.fill_between(group["theta_deg"], group["low"], group["high"], color=colour,
                          alpha=0.18, lw=0, zorder=1)
        ax_b.plot(group["theta_deg"], group["mean"], color=colour, lw=1.3, marker="o",
                  ms=3.6, mec="white", mew=0.5, zorder=3,
                  label=ROTATION_LABELS.get(target, target))
    ax_b.plot([0], [1.0], marker="o", ms=3.6, mfc="white", mec=INK, mew=0.7, zorder=4,
              ls="none")
    ax_b.annotate("unmodified", (0, 1.0), xytext=(4, -1), textcoords="offset points",
                  fontsize=SMALL - 0.6, color=MUTED, va="top")
    ax_b.set_xlim(-0.8, 15.8)
    ax_b.set_ylim(-0.03, 1.05)
    ax_b.set_xticks([0, 2.5, 5, 7.5, 10, 12.5, 15])
    ax_b.set_xticklabels(["0", "2.5", "5", "7.5", "10", "12.5", "15"])
    ax_b.set_xlabel(r"rotation off $v^\star$ at fixed norm (degrees)", fontsize=SMALL,
                    labelpad=1.5)
    first, last = int(rotation["first_layer"].min()), int(rotation["last_layer"].max())
    ax_b.set_ylabel(f"heads keeping their sink,\nlayers {first}–{last}", fontsize=SMALL,
                    labelpad=1.5)
    ax_b.legend(loc="upper right", fontsize=SMALL - 0.8, frameon=False, handlelength=1.6,
                borderaxespad=0.1, labelspacing=0.2, numpoints=1)
    for ax, letter in ((ax_a, "a"), (ax_b, "b")):
        ax.tick_params(labelsize=SMALL - 0.6, length=2, width=0.4, pad=1.5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.text(-0.2, 1.1, f"({letter})", transform=ax.transAxes, fontsize=LABEL,
                fontweight="bold", va="bottom")
    return fig
