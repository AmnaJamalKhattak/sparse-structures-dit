r"""Geometric projections of the register token population into three axes.

Every coordinate a figure in this module plots is a real inner product of a
measured token state with a basis vector; nothing here is schematic.

The primary basis is built from the mechanism rather than from generic PCA:

.. math::

    e_1 &= \hat v^* \\
    e_2 &= \frac{e_{c^*} - (e_{c^*}^\top e_1)e_1}{\|\cdot\|}
           \qquad\text{(dominant-channel variation not already in } e_1) \\
    e_3 &= \text{first principal direction of } \;
           x - (x^\top e_1)e_1 - (x^\top e_2)e_2

Axis 1 is v*'s direction, axis 2 is what the dominant channel carries beyond
it, and axis 3 is the leading remaining variation in the supplied data. The
basis is orthonormal by construction, so a coordinate is a length and the
three squared coordinates sum to a share of the token's squared norm.

``captured_fraction`` is that share, and every figure reports it: three axes
out of a few thousand can retain most of a token's energy or almost none of
it, and the value on the figure says which.

If ``v*`` is nearly the dominant channel, which happens on some checkpoints,
:math:`e_{c^*}` is nearly parallel to :math:`e_1`, what survives
orthogonalisation is small, and axis 2 spans only a thin slice of the space.
``independent_channel_content`` is :math:`\sqrt{1 - (\hat v^*_{c^*})^2}`, the
length that survives, and is reported so a flat axis-2 spread is read as
"little to see here" rather than as evidence the channel does not matter.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from .adapters import InterventionPoint
from .causal_engine import (CausalTracer, FrozenTargets, Trace, run_traced_generation,
                            select_frozen_targets)
from .control_surface import sink_readout
from .questions import QuestionContext


# ------------------------------------------------------------------ the basis
CATEGORIES: Tuple[str, ...] = ("ordinary", "highnorm_nonsink", "highnorm_sink", "extreme")
CATEGORY_LABELS: Dict[str, str] = {
    "ordinary": "Ordinary token",
    "highnorm_nonsink": "High-norm non-sink",
    "highnorm_sink": "High-norm sink",
    "extreme": "Top 1% by projection",
}


def _unit(v: torch.Tensor) -> torch.Tensor:
    v = torch.as_tensor(v).float().flatten()
    return v / v.norm().clamp_min(1e-12)


@dataclass
class Basis:
    """Three orthonormal directions, and the diagnostics that say what they are worth."""

    vectors: torch.Tensor                      # [3, C], orthonormal rows
    kind: str                                  # "interpretable" | "pca"
    labels: Tuple[str, str, str]
    dominant_channel: int
    channel_alignment: float = float("nan")    # |v*_{c*}|, how much of v* IS the channel
    independent_channel_content: float = float("nan")   # sqrt(1 - channel_alignment^2)
    residual_variance_explained: float = float("nan")   # axis 3, of the residual
    total_variance_explained: float = float("nan")      # all three, of the population
    degenerate_axis_2: bool = False
    notes: str = ""

    def project(self, states: torch.Tensor) -> torch.Tensor:
        """``[N, C]`` activations to ``[N, 3]`` coordinates in this basis."""
        return states.float() @ self.vectors.t().to(states.device).float()

    def captured_fraction(self, states: torch.Tensor) -> torch.Tensor:
        """Per token, the share of its squared norm the three axes retain.

        Orthonormality is what makes this a fraction rather than a ratio of unrelated
        quantities: the projection can never exceed the norm, so the value lies in
        ``[0, 1]`` and 1.0 would mean the token lies exactly in the drawn subspace.
        """
        coordinates = self.project(states)
        total = states.float().pow(2).sum(-1)
        return coordinates.pow(2).sum(-1) / total.clamp_min(1e-12)

    def as_dict(self) -> Dict[str, Any]:
        row = {k: v for k, v in asdict(self).items() if k != "vectors"}
        row["labels"] = list(self.labels)
        row["width"] = int(self.vectors.shape[1])
        return row


def interpretable_basis(states: torch.Tensor, vstar: torch.Tensor, dominant_channel: int,
                        *, degenerate_threshold: float = 1e-3) -> Basis:
    r"""``v*``, the orthogonalised dominant-channel axis, and a data-driven residual axis.

    Axis 3 is fitted on the *supplied* states, so it is a property of the population being
    drawn rather than a global constant.  Two consequences worth stating on any figure
    that uses it: the axis differs between layers, and it is defined only up to sign.
    """
    states = states.float()
    width = int(states.shape[-1])
    channel = int(dominant_channel) % width
    e1 = _unit(vstar)
    if e1.numel() != width:
        raise ValueError(f"v* has width {e1.numel()} but the states are {width} wide")

    axis = torch.zeros(width)
    axis[channel] = 1.0
    alignment = float(abs(e1[channel]))
    raw = axis - (axis @ e1) * e1
    surviving = float(raw.norm())
    degenerate = surviving <= degenerate_threshold
    if degenerate:
        # v* IS this channel, to numerical precision. There is no "channel variation not
        # already in v*" to draw, and normalising the residue would plot rounding noise
        # on an axis a reader would take seriously. A data-driven direction is substituted
        # and the substitution is recorded rather than hidden.
        pool = states - (states @ e1).unsqueeze(-1) * e1
        e2 = _first_principal_direction(pool)
    else:
        e2 = raw / raw.norm()

    residual = states - (states @ e1).unsqueeze(-1) * e1
    residual = residual - (residual @ e2).unsqueeze(-1) * e2
    e3 = _first_principal_direction(residual)
    explained = _variance_explained(residual, e3)

    vectors = torch.stack([e1, e2, e3])
    total = _subspace_variance_explained(states, vectors)
    labels = ("Projection onto $v^*$",
              "Residual variation axis 1 (data-driven)" if degenerate
              else "Dominant-channel-related axis",
              "Residual variation axis" if not degenerate
              else "Residual variation axis 2 (data-driven)")
    notes = ""
    if degenerate:
        notes = (f"v* is the dominant channel to within {degenerate_threshold:g}: "
                 f"|v*[{channel}]| = {alignment:.6f}, leaving {surviving:.2e} after "
                 "orthogonalisation. Axis 2 is a data-driven residual direction instead, "
                 "and no claim about 'channel variation beyond v*' can be read from it.")
    elif alignment > 0.9:
        notes = (f"v* is nearly the dominant channel: |v*[{channel}]| = {alignment:.4f}, so "
                 f"axis 2 spans only the {surviving:.3f} of unit length that survives "
                 "orthogonalisation. A narrow spread there is expected, and is not "
                 "evidence that the channel is unimportant.")
    return Basis(vectors=vectors, kind="interpretable", labels=labels,
                 dominant_channel=channel, channel_alignment=alignment,
                 independent_channel_content=surviving,
                 residual_variance_explained=explained, total_variance_explained=total,
                 degenerate_axis_2=degenerate, notes=notes)


def pca_basis(states: torch.Tensor, vstar: torch.Tensor, dominant_channel: int) -> Basis:
    """The three leading principal directions of the states: a comparison-only view.

    PCA answers where the variance is, which is not the same question as where the
    mechanism is, so this basis is not used as the primary figure.
    """
    states = states.float()
    centred = states - states.mean(0, keepdim=True)
    _, singular, right = torch.linalg.svd(centred, full_matrices=False)
    vectors = right[:3]
    variance = singular.pow(2)
    share = float(variance[:3].sum() / variance.sum().clamp_min(1e-12))
    e1 = _unit(vstar)
    channel = int(dominant_channel) % int(states.shape[-1])
    # How much of the mechanism's direction the PCA view happens to contain, so a reader
    # can tell whether the two bases are describing the same structure.
    overlap = float((vectors @ e1).pow(2).sum().clamp(max=1.0).sqrt())
    return Basis(vectors=vectors, kind="pca",
                 labels=("PC 1", "PC 2", "PC 3"), dominant_channel=channel,
                 channel_alignment=float(abs(e1[channel])),
                 total_variance_explained=share,
                 notes=(f"Mean-centred PCA of this layer's image tokens. The three PCs "
                        f"contain {overlap:.3f} of v* by length, so the mechanism's "
                        f"direction is {'well' if overlap > 0.7 else 'only partly'} "
                        "represented in this view."))


def _first_principal_direction(x: torch.Tensor) -> torch.Tensor:
    """Leading right singular vector of the mean-centred rows, sign-fixed.

    The sign of a principal direction is arbitrary, which would make a lifecycle sequence
    flip axis 3 between frames for no reason. Fixing it by the largest-magnitude
    coordinate makes consecutive frames comparable.
    """
    centred = x.float() - x.float().mean(0, keepdim=True)
    if centred.shape[0] < 2:
        out = torch.zeros(int(x.shape[-1]))
        out[0] = 1.0
        return out
    _, _, right = torch.linalg.svd(centred, full_matrices=False)
    direction = right[0]
    return direction * float(torch.sign(direction[direction.abs().argmax()]) or 1.0)


def _variance_explained(x: torch.Tensor, direction: torch.Tensor) -> float:
    centred = x.float() - x.float().mean(0, keepdim=True)
    total = float(centred.pow(2).sum())
    if total <= 0:
        return float("nan")
    return float((centred @ _unit(direction)).pow(2).sum() / total)


def _subspace_variance_explained(x: torch.Tensor, vectors: torch.Tensor) -> float:
    centred = x.float() - x.float().mean(0, keepdim=True)
    total = float(centred.pow(2).sum())
    if total <= 0:
        return float("nan")
    return float((centred @ vectors.t().float()).pow(2).sum() / total)


def build_basis(states: torch.Tensor, vstar: torch.Tensor, dominant_channel: int, *,
                kind: str = "interpretable") -> Basis:
    if kind == "interpretable":
        return interpretable_basis(states, vstar, dominant_channel)
    if kind == "pca":
        return pca_basis(states, vstar, dominant_channel)
    raise KeyError(f"unknown basis kind {kind!r}; known: 'interpretable', 'pca'")


def exact_token_basis(state: torch.Tensor, vstar: torch.Tensor,
                      dominant_channel: int) -> Basis:
    r"""Three axes that contain one token exactly, with two of them interpretable.

    A shared 3D basis fits a population figure, but not a decomposition figure: most of
    an individual token's residual lies outside any three fixed directions, so the drawn
    :math:`r_i` leg would be a fraction of the real one.

    Axes 1 and 2 are :math:`\hat v^*` and the orthogonalised dominant channel; axis 3 is
    whatever is left of this token after those two, so the subspace contains the token
    with no residue at all:

    .. math::

        e_3 = \frac{x - (x^\top e_1)e_1 - (x^\top e_2)e_2}
                    {\|x - (x^\top e_1)e_1 - (x^\top e_2)e_2\|}

    so ``captured_fraction`` is 1.0 by construction and every arrow in the figure has its
    true length. The cost is that axis 3 differs per token, which is why this basis is
    used only for single-token figures and never for a cloud.
    """
    state = state.float().flatten()
    width = int(state.numel())
    channel = int(dominant_channel) % width
    e1 = _unit(vstar)
    axis = torch.zeros(width)
    axis[channel] = 1.0
    raw = axis - (axis @ e1) * e1
    surviving = float(raw.norm())
    degenerate = surviving <= 1e-3
    e2 = (raw / raw.norm() if not degenerate
          else _unit(torch.eye(width)[(channel + 1) % width]
                     - (torch.eye(width)[(channel + 1) % width] @ e1) * e1))
    leftover = state - (state @ e1) * e1 - (state @ e2) * e2
    if float(leftover.norm()) <= 1e-9:
        # The token already lies in the plane of axes 1 and 2. Any third orthogonal
        # direction completes the basis and contributes a zero-length leg, which is the
        # truthful picture: there is no residual to draw.
        candidate = torch.eye(width)[(channel + 2) % width]
        candidate = candidate - (candidate @ e1) * e1
        candidate = candidate - (candidate @ e2) * e2
        e3 = _unit(candidate)
    else:
        e3 = leftover / leftover.norm()
    return Basis(vectors=torch.stack([e1, e2, e3]), kind="exact-per-token",
                 labels=("Projection onto $v^*$",
                         "Residual axis (data-driven)" if degenerate
                         else "Dominant-channel-related axis",
                         "This token's own residual direction"),
                 dominant_channel=channel, channel_alignment=float(abs(e1[channel])),
                 independent_channel_content=surviving, degenerate_axis_2=degenerate,
                 notes=("Axis 3 is this token's own leftover direction, so the token lies "
                        "exactly in the drawn subspace and every arrow has its true "
                        "length. Axis 3 therefore differs between panels."))


# ------------------------------------------------------------- token categories
def categorise(states: torch.Tensor, observation, *, vstar: torch.Tensor,
               norm_threshold: float, sink_threshold: float,
               extreme_percentile: float = 99.0) -> pd.DataFrame:
    """One row per image token: its geometry, its category, and why.

    Both thresholds come from the repository's own definitions. The norm bar is
    ``highnorm_ratio x median`` as computed by
    :func:`~ditsinks.causal_engine.select_frozen_targets`, and sinkhood is read through
    :func:`~ditsinks.control_surface.sink_readout`, which applies
    ``SweepConfig.sink_ratio_threshold`` per token. No new definition is introduced here.
    """
    states = states.float()
    unit = _unit(vstar)
    norms = states.norm(dim=-1)
    alpha = states @ unit
    n = int(norms.numel())
    sink = sink_readout(observation, n_tokens=n, threshold=float(sink_threshold))
    is_sink = sink["is_sink"].bool()
    loud = norms >= float(norm_threshold)
    cut = float(torch.quantile(alpha, min(max(extreme_percentile / 100.0, 0.0), 1.0)))
    extreme = alpha >= cut

    category = np.array(["ordinary"] * n, dtype=object)
    category[(loud & ~is_sink).numpy()] = "highnorm_nonsink"
    category[(loud & is_sink).numpy()] = "highnorm_sink"
    return pd.DataFrame({
        "token": np.arange(n),
        "norm": norms.numpy(), "alpha": alpha.numpy(),
        "cosine": (alpha / norms.clamp_min(1e-9)).numpy(),
        "is_highnorm": loud.numpy(), "is_sink": is_sink.numpy(),
        "is_extreme": extreme.numpy(),
        "sink_strength": sink["sink_strength_headmax"].numpy(),
        "n_sink_heads": sink["n_sink_heads"].numpy(),
        "category": category,
        "norm_rank": (-norms).argsort().argsort().numpy() + 1,
        "alpha_rank": (-alpha).argsort().argsort().numpy() + 1,
    })


# ------------------------------------------------------------------- snapshots
@dataclass
class GeometrySnapshot:
    """Everything one (step, layer) contributes to a geometric figure.

    Holds the real ``[N, C]`` activations, so the figures project rather than illustrate.
    """

    step: int
    layer: int
    states: torch.Tensor                       # [N, C] measured image-token activations
    tokens: pd.DataFrame                       # one row per token, from `categorise`
    basis: Basis
    # v* is carried explicitly rather than read off the basis. Axis 1 is v* for the
    # interpretable basis, but it is PC1 for the PCA basis, so a decomposition figure
    # that took axis 1 as v* would decompose along PC1 while labelling it v*.
    vstar: torch.Tensor = field(default_factory=lambda: torch.zeros(0))
    condition: str = "clean"
    prompt_id: int = 0
    prompt: str = ""
    seed: int = 0
    checkpoint: str = ""
    point: str = InterventionPoint.BLOCK_INPUT.value
    norm_threshold: float = 0.0
    sink_threshold: float = 0.0
    dominant_channel: int = 0

    @property
    def coordinates(self) -> np.ndarray:
        return self.basis.project(self.states).numpy()

    @property
    def captured(self) -> np.ndarray:
        return self.basis.captured_fraction(self.states).numpy()

    def category_ids(self, category: str) -> List[int]:
        return [int(t) for t in self.tokens.loc[self.tokens["category"] == category, "token"]]

    def representatives(self) -> Dict[str, Optional[int]]:
        """One token per category, chosen by a stated rule rather than by eye.

        The ordinary representative is the median-norm token, so it is typical rather than
        extreme; each high-norm representative is the loudest of its kind, so the figure
        shows the clearest instance of the thing being described.
        """
        rows = self.tokens
        out: Dict[str, Optional[int]] = {}
        ordinary = rows[rows["category"] == "ordinary"]
        if not ordinary.empty:
            middle = (ordinary["norm"] - ordinary["norm"].median()).abs().idxmin()
            out["ordinary"] = int(ordinary.loc[middle, "token"])
        for category in ("highnorm_nonsink", "highnorm_sink"):
            group = rows[rows["category"] == category]
            out[category] = (int(group.loc[group["norm"].idxmax(), "token"])
                             if not group.empty else None)
        return out

    def metadata(self) -> Dict[str, Any]:
        """The structured record saved beside every figure."""
        counts = self.tokens["category"].value_counts().to_dict()
        return {
            "checkpoint": self.checkpoint, "prompt_id": int(self.prompt_id),
            "prompt": self.prompt, "seed": int(self.seed), "condition": self.condition,
            "timestep": int(self.step), "layer": int(self.layer),
            "intervention_point": self.point,
            "dominant_channel": int(self.dominant_channel),
            "n_image_tokens": int(len(self.tokens)),
            "basis": self.basis.as_dict(),
            "norm_threshold": float(self.norm_threshold),
            "sink_threshold": float(self.sink_threshold),
            "category_counts": {str(k): int(v) for k, v in counts.items()},
            "token_ids_by_category": {c: self.category_ids(c) for c in CATEGORIES[:3]},
            "representative_tokens": self.representatives(),
            "captured_fraction_median": float(np.median(self.captured)),
            "captured_fraction_by_category": {
                str(category): float(np.median(self.captured[
                    self.tokens["category"].to_numpy() == category]))
                for category in self.tokens["category"].unique()},
            "projection_note": (
                "These coordinates are a 3D orthogonal projection of "
                f"{int(self.states.shape[-1])}-dimensional measured activations onto the "
                f"{self.basis.kind} basis named in `basis.labels`. Coordinates are real "
                "inner products; nothing is schematic."),
        }


def snapshot_from_trace(trace: Trace, *, step: int, layer: int, vstar: torch.Tensor,
                        dominant_channel: int, norm_threshold: float, sink_threshold: float,
                        basis_kind: str = "interpretable",
                        basis: Optional[Basis] = None,
                        point: InterventionPoint = InterventionPoint.BLOCK_INPUT,
                        checkpoint: str = "", prompt: str = "",
                        extreme_percentile: float = 99.0) -> Optional[GeometrySnapshot]:
    """Build a snapshot from a traced generation, or ``None`` if states were not kept.

    Passing ``basis`` reuses a basis fitted elsewhere, which is what a lifecycle sequence
    needs: refitting axis 3 per frame would make the camera move with the data and turn a
    real change into an artefact of the projection.
    """
    observation = trace.at(int(step), int(layer))
    # The state the block actually CONSUMED, after any installed edit. This matters only
    # on a treated run, and it matters a great deal there: `StateProbe` registers before
    # `EditInstaller`, so `trace.probe` at the edit layer returns the tensor as it was
    # *entering* the hook. A clean-versus-treated comparison read off that would show the
    # edited layer as unchanged and the effect as appearing one layer late.
    # `q11.consumed_states` already resolves exactly this and is reused rather than
    # reimplemented.
    from .q11 import consumed_states

    states = consumed_states(trace, int(step), int(layer), point)
    if states is None and observation is not None:
        states = observation.states
    if states is None:
        return None
    states = states.float()
    resolved = basis or build_basis(states, vstar, dominant_channel, kind=basis_kind)
    tokens = categorise(states, observation, vstar=vstar, norm_threshold=norm_threshold,
                        sink_threshold=sink_threshold,
                        extreme_percentile=extreme_percentile)
    return GeometrySnapshot(
        step=int(step), layer=int(layer), states=states, tokens=tokens, basis=resolved,
        vstar=_unit(vstar).clone(),
        condition=trace.condition, prompt_id=trace.prompt_id, prompt=prompt,
        seed=trace.seed, checkpoint=checkpoint, point=InterventionPoint(point).value,
        norm_threshold=float(norm_threshold), sink_threshold=float(sink_threshold),
        dominant_channel=int(dominant_channel))


# --------------------------------------------------------------- capture + I/O
def capture_snapshots(ctx: QuestionContext, *, layers: Sequence[int],
                      steps: Optional[Sequence[int]] = None, prompt_id: int = 0,
                      seed: Optional[int] = None, basis_kind: str = "interpretable",
                      basis_layer: Optional[int] = None, basis: Optional[Basis] = None,
                      point: InterventionPoint = InterventionPoint.BLOCK_INPUT,
                      plans: Sequence[Any] = (), condition: str = "clean",
                      targets: Optional[FrozenTargets] = None,
                      extreme_percentile: float = 99.0,
                      ) -> Tuple[List[GeometrySnapshot], FrozenTargets]:
    """One generation, keeping the real activations at every requested layer.

    The tracer normally stores per-token summaries; a ``StateProbe`` keeps the whole
    ``[N, C]`` slice, which is what a geometric figure needs. That costs
    ``N x C x 4`` bytes per layer-step, about 12 MB at FLUX's 512px shapes, so the caller
    names the layers rather than getting all of them.

    ``basis_layer`` fits the basis once, at that layer, and every snapshot reuses it. A
    lifecycle figure that refits axis 3 per frame moves the camera with the data, so a
    population that merely rotated would appear to change shape. Axes 1 and 2 are model
    constants and never move; only axis 3 is fitted, and only once.
    """
    wanted = sorted({int(l) for l in layers})
    # `probes` keeps the [N, C] slice at the intervention point, and `snapshot_from_trace`
    # reads it through `consumed_states`, so `full_state_layers` would store a second copy
    # of every layer for nothing. At FLUX's 512px shapes that second copy is 12 MB per
    # layer-step, 0.27 GB over the register window.
    capture = sorted({int(s) for s in (steps if steps is not None else [ctx.step])})
    # The frozen-target selection below reads the intervention layer, so it has to be
    # probed whether or not the caller asked for a snapshot there. Requesting a single
    # unrelated layer would otherwise fail inside this function rather than at its edge.
    probe_layers = sorted(set(wanted) | {int(ctx.intervention_layer)})
    seed = int(ctx.seeds[0] if seed is None else seed)
    prompt = list(ctx.prompts)[int(prompt_id)] if int(prompt_id) < len(ctx.prompts) else ""
    tracer = CausalTracer(
        ctx.driver.adapter, ctx.driver.transformer, direction=ctx.vstar,
        layers=sorted(set(ctx.observe_layers) | set(probe_layers)), steps=capture,
        channels=ctx.channels, grid=ctx.driver.grid, cfg=ctx.cfg)
    trace, _ = run_traced_generation(
        ctx.driver, tracer, prompt_id=int(prompt_id), prompt=prompt, seed=seed,
        condition=condition, plans=list(plans), targets=targets,
        probes=[(point, probe_layers)], save_image=True)
    frozen = targets or select_frozen_targets(
        trace, layer=int(ctx.intervention_layer), step=int(capture[0]), point=point,
        direction=ctx.vstar, percentile=ctx.percentile, topk=ctx.topk,
        highnorm_ratio=ctx.highnorm_ratio)

    if basis is not None:
        # A treated run compared against a clean one must use the CLEAN basis, or the
        # comparison is between two cameras rather than between two populations.
        shared = basis
        checkpoint = getattr(ctx.cfg.spec, "key", "")
        sink_threshold = float(getattr(ctx.cfg, "sink_ratio_threshold", 10.0))
        return ([s for s in (snapshot_from_trace(
            trace, step=step, layer=layer, vstar=ctx.vstar,
            dominant_channel=ctx.dominant_channel,
            norm_threshold=float(frozen.norm_threshold), sink_threshold=sink_threshold,
            basis=shared, point=point, checkpoint=checkpoint, prompt=prompt,
            extreme_percentile=extreme_percentile)
            for step in capture for layer in wanted) if s is not None], frozen)

    anchor = int(basis_layer if basis_layer is not None else ctx.intervention_layer)
    if anchor not in probe_layers:
        # A basis fitted on a layer that was never captured would silently fall back to
        # per-snapshot bases, which is the one thing a lifecycle figure must not do.
        raise ValueError(f"basis_layer {anchor} was not captured; ask for it in `layers`")
    from .q11 import consumed_states

    anchor_states = consumed_states(trace, int(capture[0]), anchor, point)
    if anchor_states is None:
        observation = trace.at(int(capture[0]), anchor)
        anchor_states = None if observation is None else observation.states
    shared = (build_basis(anchor_states.float(), ctx.vstar, ctx.dominant_channel,
                          kind=basis_kind) if anchor_states is not None else None)

    checkpoint = getattr(ctx.cfg.spec, "key", "")
    sink_threshold = float(getattr(ctx.cfg, "sink_ratio_threshold", 10.0))
    snapshots: List[GeometrySnapshot] = []
    for step in capture:
        for layer in wanted:
            snapshot = snapshot_from_trace(
                trace, step=step, layer=layer, vstar=ctx.vstar,
                dominant_channel=ctx.dominant_channel,
                norm_threshold=float(frozen.norm_threshold),
                sink_threshold=sink_threshold, basis_kind=basis_kind, basis=shared,
                point=point, checkpoint=checkpoint, prompt=prompt,
                extreme_percentile=extreme_percentile)
            if snapshot is not None:
                snapshots.append(snapshot)
    return snapshots, frozen


def save_metadata(snapshot: GeometrySnapshot, path, *, figure: str = "",
                  extra: Optional[Dict[str, Any]] = None) -> Path:
    """Write the structured record for one figure beside the figure itself."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(snapshot.metadata())
    payload["figure"] = figure
    if extra:
        payload.update(extra)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    return path


def token_table(snapshots: Sequence[GeometrySnapshot]) -> pd.DataFrame:
    """Every drawn token's real coordinates, as a table a reader can re-plot from."""
    frames = []
    for snapshot in snapshots:
        coordinates = snapshot.coordinates
        frame = snapshot.tokens.copy()
        frame["axis_1"], frame["axis_2"], frame["axis_3"] = (coordinates[:, 0],
                                                             coordinates[:, 1],
                                                             coordinates[:, 2])
        frame["captured_fraction"] = snapshot.captured
        for key, value in (("checkpoint", snapshot.checkpoint),
                           ("prompt_id", snapshot.prompt_id), ("seed", snapshot.seed),
                           ("condition", snapshot.condition), ("step", snapshot.step),
                           ("layer", snapshot.layer), ("basis", snapshot.basis.kind),
                           ("dominant_channel", snapshot.dominant_channel)):
            frame[key] = value
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def basis_table(snapshots: Sequence[GeometrySnapshot]) -> pd.DataFrame:
    """One row per distinct basis actually used, with its diagnostics."""
    seen, rows = set(), []
    for snapshot in snapshots:
        key = (snapshot.basis.kind, snapshot.layer, snapshot.step)
        if key in seen:
            continue
        seen.add(key)
        rows.append(dict(checkpoint=snapshot.checkpoint, step=snapshot.step,
                         layer=snapshot.layer, **snapshot.basis.as_dict(),
                         captured_fraction_median=float(np.median(snapshot.captured))))
    return pd.DataFrame(rows)
