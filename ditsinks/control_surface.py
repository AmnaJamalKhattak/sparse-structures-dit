r"""The direction vs magnitude experiment: is ``v*`` a causal state variable, or a
correlate of large magnitude?

Everything this project has established about the register population is *correlational
in one specific way*: the tokens were selected by norm, they turn out to be aligned with
``v*``, and the dominant channel carries most of both.  Norm, alignment and sinkhood move
together in the clean model, so no observation of the clean model can separate them.

This module separates them by construction.  For a token state ``x`` and unit ``v``:

.. math::

    \alpha = x^\top v, \qquad r = x - \alpha v, \qquad x = r + \alpha v

**Alignment at fixed norm.**  An alignment multiplier ``beta`` rescales only the
component along ``v``, and the result is renormalised to the token's original length:

.. math::

    y(\beta) = r + \beta \alpha v, \qquad
    \tilde y(\beta) = \|x\|_2 \frac{y(\beta)}{\|y(\beta)\|_2}

so :math:`\cos(\tilde y, v) = \beta\alpha / \sqrt{\|r\|^2 + \beta^2\alpha^2}` rises
monotonically in ``beta`` while :math:`\|\tilde y\| = \|x\|` exactly.  ``beta`` is
therefore a pure alignment knob.

**Magnitude at fixed direction.**  A norm multiplier ``gamma`` then scales the whole
vector, which cannot change the direction ``beta`` produced:

.. math::

    x'(\beta, \gamma) = \gamma\, \tilde y(\beta)

``(beta, gamma) = (1, 1)`` reconstructs ``x`` up to float32 rounding, and that identity
is asserted rather than assumed; see :func:`identity_error` and the tests.

The two axes give a 5x5 grid in which magnitude and direction vary independently, which
is what makes H1 (magnitude) and H2 (direction) distinguishable at all.

**The rescue arm** asks the mediation question.  The dominant channel is necessary for
the high-norm state, but the channel and the direction are confounded in the clean model
too.  If the channel matters *because* it builds the ``v*``-aligned state, then restoring
that state after ablating the channel should restore the phenotype; if the mechanism is
really just one coordinate, restoring the coordinate alone should be enough.

Nothing here decides between those readings.  The module measures; the notebook reports.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from . import endpoints as EP
from . import image_detail as ID
from .adapters import InterventionPoint
from .causal_engine import (CausalTracer, EditContext, EditPlan, FrozenTargets, Trace,
                            run_traced_generation, select_frozen_targets)
from .causal_ops import scale_channel
from .questions import QuestionContext


# ============================================================ the two operators
def _unit(v: torch.Tensor) -> torch.Tensor:
    v = torch.as_tensor(v).float().flatten()
    return v / v.norm().clamp_min(1e-12)


def decompose(x: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split ``[N, C]`` states into ``alpha`` of shape ``[N, 1]`` and the residual ``r``.

    Float32 throughout. ``v`` quantised into the model dtype is not a unit vector any
    more, and a decomposition along a non-unit direction does not reassemble.
    """
    x = x.float()
    unit = _unit(v).to(x.device)
    alpha = (x @ unit).unsqueeze(-1)
    return alpha, x - alpha * unit


def align_scale(x: torch.Tensor, v: torch.Tensor, beta: float, *, seed: int = 0,
                degenerate_ratio: float = 1e-2) -> Tuple[torch.Tensor, int]:
    """``y~(beta)``: change alignment with ``v`` while holding every token's norm fixed.

    Returns the edited states and the number of tokens that hit the degenerate branch.

    **The degenerate case.**  ``y(beta)`` vanishes when a token is collinear with ``v``
    and ``beta = 0``: there is no residual left to point along.  Renormalising numerical
    noise back to full length would amplify the noise and can even restore the original
    direction with its sign flipped.  Below ``degenerate_ratio`` of the original norm the
    direction is therefore replaced by a seeded random direction orthogonal to ``v``,
    which is what "keep the magnitude, remove the alignment" has to mean in that limit.
    This mirrors :func:`ditsinks.causal_ops.remove_direction`, and the count is returned
    so a run can report how often it happened instead of hiding it.
    """
    x = x.float()
    unit = _unit(v).to(x.device)
    alpha, residual = decompose(x, unit)
    norm = x.norm(dim=-1, keepdim=True)
    y = residual + float(beta) * alpha * unit
    length = y.norm(dim=-1, keepdim=True)
    degenerate = length <= float(degenerate_ratio) * norm
    count = int(degenerate.sum())
    if count:
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        noise = torch.randn(y.shape, generator=generator).to(y.device)
        noise = noise - (noise @ unit).unsqueeze(-1) * unit
        y = torch.where(degenerate, noise, y)
        length = y.norm(dim=-1, keepdim=True)
    return norm * y / length.clamp_min(1e-12), count


def control_surface_states(x: torch.Tensor, v: torch.Tensor, beta: float, gamma: float, *,
                           seed: int = 0) -> Tuple[torch.Tensor, int]:
    """``x'(beta, gamma) = gamma * y~(beta)``: the full 2D control applied to ``[N, C]``."""
    aligned, degenerate = align_scale(x, v, beta, seed=seed)
    return float(gamma) * aligned, degenerate


def identity_error(x: torch.Tensor, v: torch.Tensor) -> float:
    """Max absolute reconstruction error of ``(beta, gamma) = (1, 1)``.

    The contract the whole grid rests on: the centre cell must be the clean state, so
    every other cell's effect is attributable to the manipulation rather than to the
    machinery.  Returned as a number so a run can print it rather than trust it.
    """
    rebuilt, _ = control_surface_states(x, v, 1.0, 1.0)
    return float((rebuilt - x.float()).abs().max())


def match_perturbation(base: torch.Tensor, edited: torch.Tensor,
                       target: torch.Tensor) -> torch.Tensor:
    """Rescale ``edited - base`` per token so its L2 norm equals ``target``.

    The energy-matching primitive every control in this module uses.  A control that
    perturbs by a different amount than the condition it controls for is not a control,
    and matching it here, once, on the tensors themselves, is stronger than matching
    a run-level energy total that could be met by a few tokens moving a great deal.
    """
    delta = edited.float() - base.float()
    length = delta.norm(dim=-1, keepdim=True)
    wanted = torch.as_tensor(target, device=delta.device, dtype=delta.dtype).reshape(-1, 1)
    return base.float() + delta * (wanted / length.clamp_min(1e-12))


def orthogonal_direction(v: torch.Tensor, width: int, *, seed: int) -> torch.Tensor:
    r"""A deterministic unit vector orthogonal to ``v``.

    .. math:: u_\perp = u - (u^\top v)v, \qquad u_\perp \leftarrow u_\perp/\|u_\perp\|

    Derived from ``seed`` alone, so the same experiment seed always produces the same
    control direction and it can be regenerated exactly.
    """
    unit = _unit(v)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    u = torch.randn(int(width), generator=generator)
    u = u - (u @ unit) * unit
    return _unit(u)


def random_direction(width: int, *, seed: int) -> torch.Tensor:
    """A deterministic random unit vector, not orthogonalised against anything."""
    generator = torch.Generator(device="cpu").manual_seed(int(seed) + 104_729)
    return _unit(torch.randn(int(width), generator=generator))


# ============================================================ token selection
SELECTION_MODES: Dict[str, str] = {
    "highnorm_and_aligned": "clean high-norm AND strongly v*-aligned (the primary population)",
    "register": "the repository's percentile high-norm rule (FrozenTargets.register_ids)",
    "topk_norm": "top-k by clean residual norm (threshold-free companion)",
    "topk_projection": "top-k by clean x . v*",
    "percentile_projection": "top percentile by clean x . v*",
    "sink_only": "clean tokens that are the argmax key for at least one head",
    "random_tokens": "count-matched ordinary tokens (the random-token control)",
}


@dataclass(frozen=True)
class Treatment:
    """The frozen treatment population for one (prompt, seed), and how it was chosen."""

    mode: str
    token_ids: Tuple[int, ...]
    rule: str
    n_candidates: int
    clean_sink_ids: Tuple[int, ...] = ()
    n_selected_that_were_sinks: int = 0

    def as_dict(self) -> Dict[str, Any]:
        row = asdict(self)
        row["token_ids"] = list(self.token_ids)
        row["clean_sink_ids"] = list(self.clean_sink_ids)
        return row


def clean_sink_ids(targets: FrozenTargets, *, layers: Sequence[int] = ()) -> Tuple[int, ...]:
    """Tokens that are some head's strongest image key on the clean run.

    This is the repository's existing per-token notion of sinkhood.
    ``FrozenTargets.clean_sink_by_head`` holds the per-head argmax recorded by
    :class:`~ditsinks.causal_engine.CausalTracer`, and no new definition is introduced.
    """
    wanted = {int(l) for l in layers} if len(layers) else None
    found: List[int] = []
    for layer, by_head in targets.clean_sink_by_head.items():
        if wanted is not None and int(layer) not in wanted:
            continue
        for token in by_head.values():
            if int(token) not in found:
                found.append(int(token))
    return tuple(sorted(found))


def select_treatment(targets: FrozenTargets, *, mode: str = "highnorm_and_aligned",
                     direction: Optional[torch.Tensor] = None, topk: int = 8,
                     projection_percentile: float = 99.0,
                     sink_layers: Sequence[int] = ()) -> Treatment:
    """Freeze the treatment population from the **clean** pass only.

    Every mode reads ``targets``, which :func:`select_frozen_targets` built from the
    clean trajectory.  No mode may consult a treated run: re-deriving the population
    after the intervention would let the intervention choose its own subjects, and every
    downstream comparison would be between different sets of tokens.
    """
    if mode not in SELECTION_MODES:
        raise KeyError(f"unknown selection mode {mode!r}; known: {sorted(SELECTION_MODES)}")
    states = targets.clean_states
    sinks = clean_sink_ids(targets, layers=sink_layers)
    n = int(targets.n_img or (0 if states is None else states.shape[0]))

    def finish(ids: Sequence[int], rule: str) -> Treatment:
        chosen = tuple(sorted({int(t) for t in ids}))
        return Treatment(mode=mode, token_ids=chosen, rule=rule, n_candidates=n,
                         clean_sink_ids=sinks,
                         n_selected_that_were_sinks=len(set(chosen) & set(sinks)))

    if mode == "register":
        # FrozenTargets normally carries its own rule; a caller that built one by hand
        # would otherwise produce a treatment with no provenance at all, and provenance
        # is the point of freezing the population.
        return finish(targets.register_ids, targets.selection_rule
                      or "FrozenTargets.register_ids (percentile high-norm rule)")
    if mode == "topk_norm":
        return finish(targets.topk_ids, f"top-{len(targets.topk_ids)} by clean residual norm")
    if mode == "random_tokens":
        return finish(targets.random_ids, "count-matched ordinary tokens, seeded from the unit")
    if mode == "sink_only":
        return finish(sinks, "clean per-head argmax image keys at the measured layers")

    if states is None or direction is None:
        raise ValueError(f"mode {mode!r} needs the clean states and v*; "
                         "run select_frozen_targets with a StateProbe at the edit point")
    projection = states.float() @ _unit(direction)
    norms = states.float().norm(dim=-1)
    if mode == "topk_projection":
        order = torch.argsort(projection, descending=True)
        return finish([int(t) for t in order[: min(int(topk), n)]],
                      f"top-{min(int(topk), n)} by clean x . v*")
    if mode == "percentile_projection":
        cut = float(torch.quantile(projection, min(max(projection_percentile / 100.0, 0.0), 1.0)))
        ids = [int(t) for t in torch.argsort(projection, descending=True) if float(projection[t]) >= cut]
        return finish(ids or [int(projection.argmax())],
                      f"top {100 - projection_percentile:g}% by clean x . v*")

    # highnorm_and_aligned: the intersection used for the primary arm.
    # Norm uses the repository's existing threshold (ratio x median); alignment uses the
    # bar select_frozen_targets already computed from the natural register population,
    # so neither half of the intersection is a new definition invented here.
    cosine = projection / norms.clamp_min(1e-9)
    loud = norms >= targets.norm_threshold
    aligned = cosine.abs() >= max(targets.alignment_threshold, 0.0)
    both = torch.nonzero(loud & aligned).flatten().tolist()
    if not both:
        # An empty intersection is a real outcome on some layers, and an empty treatment
        # set would silently turn every condition into a no-op. Fall back to the frozen
        # register set and say so in the rule, rather than producing a run of no-ops.
        return finish(targets.register_ids,
                      "FALLBACK: no token met both bars; used the frozen register set. "
                      f"norm >= {targets.norm_threshold:.4g} and |cos| >= "
                      f"{targets.alignment_threshold:.4g} selected nothing")
    return finish(both, f"norm >= {targets.norm_threshold:.4g} (ratio x median) AND "
                        f"|cos(x, v*)| >= {targets.alignment_threshold:.4g} "
                        "(weakest natural register, floored by the ordinary extreme)")


# ======================================================= the true rotation operator
# A rotation, not a rescale. `align_scale` multiplies the v* component by beta and then
# restores the original length, which moves the state along the v* axis and renormalises;
# the resulting direction is always in the span of {v*, r} where r is the token's own
# residual. That cannot take a state into a *chosen* direction, and it cannot preserve
# norm without the renormalisation step doing the work.
#
# This does. Pick a unit u with u.v = 0, split x = alpha v + b u + q with q orthogonal to
# both, and rotate the (v, u) plane:
#
#     alpha' = alpha cos(theta) - b sin(theta)
#     b'     = alpha sin(theta) + b cos(theta)
#     x'     = alpha' v + b' u + q
#
# ||x'||^2 = alpha'^2 + b'^2 + ||q||^2 = alpha^2 + b^2 + ||q||^2 = ||x||^2, exactly, because
# (alpha, b) -> (alpha', b') is an orthogonal map of R^2 and q is untouched. No
# renormalisation is applied and none is needed; the preserved norm is a property of the
# operator rather than of a correction step, which is the whole point of doing it this way.


def rotate_plane(x: torch.Tensor, v: torch.Tensor, u: torch.Tensor,
                 theta: float) -> torch.Tensor:
    r"""Rotate each row of ``x`` by ``theta`` radians inside the :math:`(v, u)` plane.

    ``v`` and ``u`` must be unit and orthogonal; :func:`plane_basis` builds such a pair
    and records how far from orthogonal the requested target was.  Everything outside
    the plane is carried through untouched, so the norm is preserved by construction and
    :math:`\theta = 0` returns ``x`` bit-for-bit up to the float32 round trip.
    """
    v_unit = _unit(v).to(device=x.device, dtype=torch.float32)
    u_unit = _unit(u).to(device=x.device, dtype=torch.float32)
    rows = x.float()
    alpha = (rows @ v_unit).unsqueeze(-1)
    b = (rows @ u_unit).unsqueeze(-1)
    q = rows - alpha * v_unit - b * u_unit
    cos, sin = math.cos(float(theta)), math.sin(float(theta))
    alpha_new = alpha * cos - b * sin
    b_new = alpha * sin + b * cos
    return q + alpha_new * v_unit + b_new * u_unit


def rotation_norm_error(x: torch.Tensor, rotated: torch.Tensor) -> float:
    """Largest relative change in token norm the rotation produced.

    Reported and asserted on rather than corrected. A rotation that does not preserve
    norm is a bug in the basis (a ``u`` that is not unit, or not orthogonal to ``v``),
    and renormalising the output would hide exactly that.
    """
    before = x.float().norm(dim=-1)
    after = rotated.float().norm(dim=-1)
    return float(((after - before).abs() / before.clamp_min(1e-12)).max())


def plane_basis(vstar: torch.Tensor, target: str, *, width: int, seed: int = 0,
                residuals: Optional[torch.Tensor] = None,
                ordinary: Optional[torch.Tensor] = None,
                semantic: Optional[torch.Tensor] = None) -> Dict[str, Any]:
    r"""The orthonormal pair :math:`(v, u)` the rotation turns in.

    ``v`` is always the frozen :math:`v^*`, normalised.  ``u`` is the *target* direction,
    projected off ``v`` and renormalised, so ``u.v = 0`` holds to float precision however
    the target was obtained.  The returned record carries the raw alignment the target had
    with :math:`v^*` before orthogonalisation, because a target that was nearly parallel
    to :math:`v^*` leaves almost nothing behind and the rotation then turns in a plane
    chosen mostly by numerical noise.

    ``random_orthogonal``  a seeded random unit direction. The null: any direction of the
                           same length, chosen without reference to the data.
    ``residual_pc1``       the leading principal direction of the clean tokens' residuals
                           after :math:`v^*` is removed: the direction the population
                           itself varies in most, once the register axis is taken out.
    ``ordinary_mean``      where the *ordinary* tokens point: the mean of their unit
                           directions, which is a direction question rather than a
                           length one, so the longest token does not decide it. Rotating
                           the register toward this asks: what happens if the register
                           state points where an ordinary token points, at the
                           register's own norm?
    ``ordinary_pc1``       how the ordinary tokens *vary* most, rather than where they
                           point. The two differ whenever the ordinary population is
                           spread around its own mean, and both are worth having.
    ``semantic_direction`` a direction supplied by the caller. Nothing is derived or
                           guessed: without one the mode refuses, and says so, rather
                           than quietly falling back to a random direction and reporting
                           a semantic result.
    """
    v = _unit(vstar).float()
    notes: List[str] = []
    if target == "random_orthogonal":
        raw = random_direction(width, seed=seed)
    elif target == "residual_pc1":
        if residuals is None or residuals.numel() == 0:
            raise ValueError(
                "residual_pc1 needs the clean token states at this (step, layer): it is "
                "the first principal direction of their residuals after v* is removed, "
                "and there is nothing to fit without them")
        rows = residuals.float()
        alpha = (rows @ v).unsqueeze(-1)
        residual = rows - alpha * v
        centred = residual - residual.mean(dim=0, keepdim=True)
        if centred.shape[0] < 2:
            raise ValueError(
                "residual_pc1 needs at least two clean tokens to have a principal "
                "direction; one token defines no variance")
        _, _, right = torch.linalg.svd(centred, full_matrices=False)
        raw = right[0]
        # SVD fixes a direction only up to sign. Pin it by the largest coordinate so the
        # same population gives the same u on every run and +theta means one thing.
        raw = raw * float(torch.sign(raw[int(raw.abs().argmax())]).item() or 1.0)
    elif target in ("ordinary_mean", "ordinary_pc1"):
        if ordinary is None or ordinary.numel() == 0:
            raise ValueError(
                f"{target} needs the ORDINARY tokens' states at this (step, layer) -- "
                "every image token that is neither a frozen register token nor in the "
                "high-norm top-k. The runner supplies them; there is nothing to fit "
                "without them.")
        rows = ordinary.float()
        if int(rows.shape[0]) < 2:
            raise ValueError(f"{target} needs at least two ordinary tokens, got "
                             f"{int(rows.shape[0])}")
        if target == "ordinary_mean":
            # The mean of UNIT directions, not of the raw vectors. "Where they point" is
            # a question about direction, and a mean of raw vectors is dominated by
            # whichever ordinary token happens to be longest.
            units = rows / rows.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            raw = units.mean(dim=0)
            # How coherently they point at all. Near 1 the ordinary tokens share a
            # direction; near 0 they are spread over the sphere and their mean is mostly
            # cancellation, so "where ordinary tokens point" is not a well-posed target
            # and the rotation would be toward an artefact of the averaging.
            notes.append(
                f"ordinary-token direction concentration ||mean of unit vectors|| = "
                f"{float(raw.norm()):.4f}. Near 1 they share a direction; near 0 they are "
                "spread over the sphere and this target is not well posed.")
        else:
            centred = rows - rows.mean(dim=0, keepdim=True)
            _, _, right = torch.linalg.svd(centred, full_matrices=False)
            raw = right[0]
            raw = raw * float(torch.sign(raw[int(raw.abs().argmax())]).item() or 1.0)
    elif target == "semantic_direction":
        if semantic is None:
            raise ValueError(
                "semantic_direction has no direction to use. Pass one through "
                "ControlSurfaceConfig.semantic_direction -- a vector of the model's "
                "width, in the same basis as v*. This mode deliberately does not derive "
                "or guess one: a random direction reported as semantic would be a "
                "fabricated result, and the interface exists so the direction can be "
                "supplied when you have one.")
        raw = semantic.float().reshape(-1)
        if int(raw.numel()) != int(width):
            raise ValueError(f"semantic_direction has width {int(raw.numel())}, but this "
                             f"checkpoint's residual stream is {int(width)}")
    else:
        raise KeyError(f"unknown rotation target {target!r}; known: "
                       + ", ".join(repr(k) for k in ROTATION_TARGETS))
    raw = raw.float().to(v.device)
    raw_unit = _unit(raw)
    alignment = float((raw_unit @ v).item())
    perpendicular = raw_unit - alignment * v
    surviving = float(perpendicular.norm().item())
    if surviving <= 1e-6:
        raise ValueError(
            f"the {target!r} direction is parallel to v* (|cos| = {abs(alignment):.6f}), so "
            "nothing survives orthogonalisation and there is no plane to rotate in")
    u = perpendicular / surviving
    if surviving < 0.1:
        notes.append(
            f"only {surviving:.3f} of the {target} direction survives orthogonalisation "
            f"against v* (|cos| = {abs(alignment):.3f}); u is mostly whatever was left, "
            "so read the plane as 'v* against a near-degenerate complement'")
    return dict(target=target, v=v, u=u,
                target_cosine_with_vstar=alignment,
                independent_target_content=surviving,
                u_dot_v=float((u @ v).item()),
                seed=int(seed), notes=notes)


ROTATION_ANGLES_DEG: Tuple[float, ...] = (-90.0, -60.0, -30.0, -15.0, 0.0,
                                          15.0, 30.0, 60.0, 90.0)
ROTATION_TARGETS: Tuple[str, ...] = ("random_orthogonal", "residual_pc1",
                                     "ordinary_mean", "ordinary_pc1",
                                     "semantic_direction")
# Which targets need which clean population. `residual_pc1` is fitted on the treated
# tokens' own slice; the ordinary targets are fitted on everything that is neither a
# frozen register token nor high-norm, which is a different population entirely.
TARGETS_NEEDING_ORDINARY: Tuple[str, ...] = ("ordinary_mean", "ordinary_pc1")


def rotation_conditions(angles: Sequence[float] = ROTATION_ANGLES_DEG,
                        targets: Sequence[str] = ("random_orthogonal", "residual_pc1"),
                        ) -> Tuple[Condition, ...]:
    """One condition per (target, angle). ``theta = 0`` is the reference in every target.

    It is generated rather than skipped, for the same reason the grid runs (1, 1): it is
    the numerical floor every other angle in that target is read against, and a cell that
    was never generated cannot supply one.
    """
    out: List[Condition] = []
    for target in targets:
        for angle in angles:
            out.append(Condition(
                key=f"rotate_{target}_{angle:g}deg",
                label=f"Rotate {angle:g}deg in the (v*, {target}) plane",
                arm="rotation", edit_direction="rotation", theta_deg=float(angle),
                rotation_target=str(target),
                role="reference" if abs(float(angle)) < 1e-12 else "condition"))
    return tuple(out)

def clean_subspace(states: torch.Tensor, rank: int = 32) -> Dict[str, Any]:
    r"""The affine subspace the clean image tokens actually occupy at one layer.

    Fitted as the leading right singular vectors of the mean-centred clean slice, so
    "on the manifold" means "inside the space this layer's own tokens span", measured
    against this layer's own population rather than against an assumption.

    Returned with the clean population's *own* energy share inside that subspace, because
    the treated share is only readable beside it: a rank-32 subspace of a 3072-wide space
    might capture 85% of a clean token or 40%, and which it is decides what a treated
    token's 30% means.
    """
    states = states.float()
    centre = states.mean(0, keepdim=True)
    centred = states - centre
    rank = int(min(max(rank, 1), min(centred.shape) - 1)) if min(centred.shape) > 1 else 1
    try:
        _, singular, right = torch.linalg.svd(centred, full_matrices=False)
    except Exception:                                          # pragma: no cover
        return {}
    basis = right[:rank]
    inside = (centred @ basis.t()).pow(2).sum(-1)
    total = centred.pow(2).sum(-1).clamp_min(1e-12)
    share = (inside / total)
    width = int(states.shape[-1])
    return {"centre": centre, "basis": basis, "rank": rank,
            "clean_share_median": float(share.median()),
            "clean_share_p10": float(torch.quantile(share, 0.10)),
            "variance_explained": float(singular[:rank].pow(2).sum()
                                        / singular.pow(2).sum().clamp_min(1e-12)),
            # Whether the subspace is a genuine RESTRICTION is a question about its
            # dimension, not about the variance it happens to explain: rank 11 of 12
            # explains only ~0.97 and still leaves nowhere to be off-manifold, while rank
            # 32 of 3072 explains far less and is a real constraint. The fraction is what
            # decides it, and both are reported.
            "rank_fraction": rank / max(width, 1),
            "is_discriminative": bool(rank / max(width, 1) < 0.5)}


def onmanifold_energy(states: torch.Tensor, subspace: Dict[str, Any]) -> float:
    r"""Share of the treated tokens' energy lying inside the clean subspace.

    The operators preserve or rescale a token's *norm* by construction, so norm cannot
    say whether a state is one the model ever produces. This can: a token pushed to a
    negative ``v*`` projection at unchanged length either still lies in the space the
    clean population spans (a state the model could have reached) or it does not, and
    is then off-manifold corruption wearing the right magnitude.
    """
    if not subspace or "basis" not in subspace:
        return float("nan")
    centred = states.float() - subspace["centre"].to(states.device).float()
    basis = subspace["basis"].to(states.device).float()
    inside = (centred @ basis.t()).pow(2).sum(-1)
    total = centred.pow(2).sum(-1).clamp_min(1e-12)
    return float((inside / total).mean())


# ================================================================== conditions
@dataclass(frozen=True)
class Condition:
    """One generated counterfactual: what was done, to which tokens, and why.

    ``arm`` decides the output subdirectory; ``role`` marks whether a row is the thing
    being tested or the control it is checked against.
    """

    key: str
    label: str
    arm: str                                   # clean | grid | rescue | controls
    beta: float = 1.0
    gamma: float = 1.0
    ablate_channel: bool = False
    rescue: str = "none"                       # none | vstar | orthogonal | channel
    edit_direction: str = "vstar"              # vstar | orthogonal | random | channel
    token_group: str = "treatment"             # treatment | random_tokens | sink_only
    role: str = "condition"                    # condition | reference | control
    match_energy_to: Optional[str] = None      # key of the condition whose L2 to match
    # A causal intervention edits the conditional branch, the convention the causal-
    # mechanism experiments use and what every other arm here uses. On a CFG-batched
    # pipeline that choice is not
    # neutral: PixArt concatenates [negative, positive] and combines them as
    # `u + s(t - u)`, so an edit confined to the conditional row enters only the second
    # term and its effect on the prediction is scaled by the guidance strength s. Editing
    # both rows puts the edit in both terms, where s largely cancels. The `cfg` arm runs
    # a few conditions both ways so that factor is measured rather than assumed.
    all_batch_rows: bool = False
    # The true-rotation arm: an angle in the (v*, u) plane and which u. `beta` and
    # `gamma` stay at their identity values there: a rotation is not a rescale, and
    # composing the two would make neither measurable.
    theta_deg: float = 0.0
    rotation_target: str = ""

    @property
    def directory(self) -> str:
        if self.arm == "rotation":
            return f"rotation/{self.rotation_target}/theta_{self.theta_deg:g}"
        if self.arm == "grid":
            return f"grid/beta_{self.beta:g}_gamma_{self.gamma:g}"
        if self.arm == "cfg":
            return f"cfg/{self.key}"
        if self.arm == "clean":
            return "clean"
        return f"{self.arm}/{self.key}"


RESCUE_CONDITIONS: Tuple[Condition, ...] = (
    Condition("clean", "Clean", "rescue", role="reference"),
    Condition("channel_ablate", "Dominant channel ablated", "rescue",
              ablate_channel=True, role="condition"),
    Condition("channel_ablate_vstar_rescue", "Channel ablated + v* rescue", "rescue",
              ablate_channel=True, rescue="vstar", role="condition"),
    Condition("channel_ablate_orthogonal_rescue", "Channel ablated + norm-matched orthogonal rescue",
              "rescue", ablate_channel=True, rescue="orthogonal", role="control"),
    Condition("channel_restore_only", "Channel ablated + dominant coordinate restored",
              "rescue", ablate_channel=True, rescue="channel", role="condition"),
)


def grid_conditions(betas: Sequence[float], gammas: Sequence[float]) -> List[Condition]:
    """The 5x5 causal control surface, centre cell included as a round-trip test."""
    return [Condition(f"beta{beta:g}_gamma{gamma:g}",
                      f"beta={beta:g}, gamma={gamma:g}", "grid", beta=float(beta),
                      gamma=float(gamma),
                      role="reference" if (beta == 1.0 and gamma == 1.0) else "condition")
            for gamma in gammas for beta in betas]


def control_conditions(betas: Sequence[float]) -> List[Condition]:
    """Four controls, each matched to a named grid cell rather than to a nominal size.

    ``random_direction`` and ``orthogonal`` answer "was it this direction, or any
    direction of that size"; ``random_tokens`` answers "was it these tokens, or any
    tokens"; ``sink_only`` asks whether the sink subset carries the effect that the
    broader aligned population carries.
    """
    extremes = [b for b in betas if b != 1.0]
    low = min(extremes) if extremes else 0.0
    high = max(extremes) if extremes else 2.0
    out: List[Condition] = []
    for beta in (low, high):
        # The comparison that decides whether the beta axis tests geometry or one channel.
        # Energy-matched against the same beta along v*, so a difference is about WHICH
        # direction was rescaled and not about how hard.
        out.append(Condition(f"channel_axis_beta{beta:g}",
                             f"beta={beta:g} along the dominant channel axis, L2-matched",
                             "controls", beta=float(beta), edit_direction="channel",
                             role="condition", match_energy_to=f"beta{beta:g}_gamma1"))
        out.append(Condition(f"random_direction_beta{beta:g}",
                             f"Random direction, beta={beta:g}, L2-matched", "controls",
                             beta=float(beta), edit_direction="random", role="control",
                             match_energy_to=f"beta{beta:g}_gamma1"))
        out.append(Condition(f"orthogonal_beta{beta:g}",
                             f"Orthogonal to v*, beta={beta:g}, L2-matched", "controls",
                             beta=float(beta), edit_direction="orthogonal", role="control",
                             match_energy_to=f"beta{beta:g}_gamma1"))
        out.append(Condition(f"random_tokens_beta{beta:g}",
                             f"Same edit on ordinary tokens, beta={beta:g}", "controls",
                             beta=float(beta), token_group="random_tokens", role="control"))
        out.append(Condition(f"sink_only_beta{beta:g}",
                             f"Same edit on clean sinks only, beta={beta:g}", "controls",
                             beta=float(beta), token_group="sink_only", role="condition"))
    return out


def cfg_conditions(betas: Sequence[float], gammas: Sequence[float]) -> List[Condition]:
    """A few strong settings run on BOTH batch rows, to measure the guidance factor.

    Small on purpose: the twins of a few named cells, not a second 5x5 grid. Their
    conditional-only counterparts already exist in the ``grid`` and ``rescue`` arms, so
    the comparison is a join rather than a duplication. Widen it only if this diagnostic
    shows the two branch choices behaving qualitatively differently, rather than
    differing by a scale factor.

    **This arm is readable on images only, and that is not a limitation of the
    implementation.** The tracer records the conditional row, and batch rows do not mix
    inside the transformer: each is an independent element of the same forward pass. So
    the internal readouts of a twin and its conditional-only partner are *identical by
    construction*, whatever the guidance strength. The two choices diverge in the
    **sampler**, where ``u + s(t - u)`` combines the branches: a conditional-only edit
    enters only the second term and its effect on the prediction is scaled by ``s``,
    while an edit on both rows enters both terms and the scaling largely cancels. A
    reader who compared ``population_metrics`` between the twins would find no difference
    and conclude the branch choice does not matter, which is exactly backwards.
    """
    extremes = [b for b in betas if b != 1.0]
    low = min(extremes) if extremes else 0.0
    high = max(extremes) if extremes else 2.0
    out = [Condition(f"allrows_beta{beta:g}_gamma1",
                     f"beta={beta:g} applied to BOTH CFG rows", "cfg", beta=float(beta),
                     role="control", all_batch_rows=True) for beta in (low, high)]
    quiet = min(gammas) if len(gammas) else 0.5
    if float(quiet) != 1.0:
        # A gamma twin at gamma=1 would be a no-op twin of a no-op cell: two generations
        # spent reproducing the clean image on both branches.
        out.append(Condition(f"allrows_beta1_gamma{quiet:g}",
                             f"gamma={quiet:g} applied to BOTH CFG rows", "cfg",
                             gamma=float(quiet), role="control", all_batch_rows=True))
    out.append(Condition("allrows_channel_ablate",
                         "Dominant channel ablated on BOTH CFG rows", "cfg",
                         ablate_channel=True, role="control", all_batch_rows=True))
    return out


# The conditional-only partner of each cfg twin, for the join the notebook reports.
CFG_PARTNERS: Dict[str, str] = {"allrows_channel_ablate": "channel_ablate"}


def cfg_partner(key: str) -> str:
    """The conditional-only condition a ``cfg`` twin should be compared against."""
    if key in CFG_PARTNERS:
        return CFG_PARTNERS[key]
    return key.replace("allrows_", "", 1) if key.startswith("allrows_") else key


# ===================================================================== the edit
# How much relative norm drift is tolerable before the (v*, u) basis is declared broken.
# A float32 plane rotation on FLUX's width lands around 1e-7; 1e-4 leaves four orders of
# headroom for a bf16 residual stream and still catches a basis that is not orthonormal.
_ROTATION_NORM_TOLERANCE = 1e-4


def _basis_like(v: torch.Tensor, u: torch.Tensor, states: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
    """The (v, u) pair as float32 on the same device as ``states``.

    The plane is built once per (step, layer) from the CLEAN probe states, which the
    tracer keeps on the CPU so a long run does not hold activations in GPU memory. The
    edit then applies it to the LIVE tensor, which is on the model's device. Every
    matmul between the two has to cross that boundary, and doing it in one place rather
    than at each call site is the only way to be sure none was missed. A device
    mismatch here is not a wrong number, it is a crash mid-run.
    """
    return (_unit(v).to(device=states.device, dtype=torch.float32),
            _unit(u).to(device=states.device, dtype=torch.float32))


def _angle_from_vstar_deg(states: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Each token's actual angle to v*, in degrees: the quantity theta acts on.

    This is not theta. theta is how far the plane was turned; this is where the token
    ended up, and the two differ because a token starts at its own angle to v* and only
    the (v*, u) component of it moves.
    """
    rows = states.float()
    unit = _unit(v).to(device=rows.device, dtype=torch.float32)
    cosine = (rows @ unit) / rows.norm(dim=-1).clamp_min(1e-12)
    return torch.rad2deg(torch.acos(cosine.clamp(-1.0, 1.0)))


def _rotation_note(before: torch.Tensor, after: torch.Tensor, plane: Dict[str, Any],
                   condition: Condition, norm_error: float) -> Dict[str, Any]:
    """Everything the rotation arm records at the hook, per (step, layer).

    Norms, both projections and both angles, before and after, so the claim "the norm
    was held fixed exactly while the direction changed" is a pair of numbers in the
    table rather than a property of the code a reader has to trust.
    """
    first, second = before.float(), after.float()
    v, u = _basis_like(plane["v"], plane["u"], first)
    angle_before = _angle_from_vstar_deg(first, v)
    angle_after = _angle_from_vstar_deg(second, v)
    note: Dict[str, Any] = dict(
        theta_deg=float(condition.theta_deg),
        rotation_target=str(condition.rotation_target),
        rotation_u_dot_v=float(plane["u_dot_v"]),
        rotation_u_norm=float(u.norm()),
        rotation_target_cosine_with_vstar=float(plane["target_cosine_with_vstar"]),
        rotation_independent_target_content=float(plane["independent_target_content"]),
        rotation_norm_relative_error=float(norm_error),
        norm_before=float(first.norm(dim=-1).mean()),
        norm_after=float(second.norm(dim=-1).mean()),
        alpha_vstar_before=float((first @ v).mean()),
        alpha_vstar_after=float((second @ v).mean()),
        projection_u_before=float((first @ u).mean()),
        projection_u_after=float((second @ u).mean()),
        cosine_vstar_before=float(((first @ v) / first.norm(dim=-1).clamp_min(1e-12)).mean()),
        cosine_vstar_after=float(((second @ v) / second.norm(dim=-1).clamp_min(1e-12)).mean()),
        cosine_u_before=float(((first @ u) / first.norm(dim=-1).clamp_min(1e-12)).mean()),
        cosine_u_after=float(((second @ u) / second.norm(dim=-1).clamp_min(1e-12)).mean()),
        angle_from_vstar_before_deg=float(angle_before.mean()),
        angle_from_vstar_after_deg=float(angle_after.mean()),
        # Signed, so it can be compared against theta: a token whose whole length lies in
        # the plane moves by exactly theta, and one with a large out-of-plane component
        # moves by less. The gap between them is how much of the state the plane holds.
        angle_change_deg=float((angle_after - angle_before).mean()),
    )
    note.update(_plane_shares(first, v, u))
    return note


def _plane_shares(states: torch.Tensor, v: torch.Tensor, u: torch.Tensor
                  ) -> Dict[str, Any]:
    r"""The two halves of the plane's share of the token, and what each one bounds.

    .. math:: \frac{\alpha^2}{\|x\|^2} + \frac{b^2}{\|x\|^2}
              = \text{in\_plane\_energy\_share}

    They are reported apart because they bound *different* things, and the sum alone
    invites the wrong reading:

    ``alpha2_share`` is :math:`\cos^2(x, v^*)`. It is the alignment the rotation can
    **remove**: at :math:`\theta = 90^\circ`, :math:`\alpha' = -b`, so a token whose
    length is mostly :math:`\alpha` has its :math:`v^*` alignment almost entirely
    swept out.

    ``b2_share`` is what the rotation can **bring in** to :math:`v^*`: at
    :math:`\theta = 90^\circ`, :math:`\alpha'` *is* :math:`-b`, so this is the
    alignment the token ends up with.

    A high-alignment register token therefore has ``in_plane_energy_share`` of at least
    :math:`\cos^2 \approx 0.96` for **any** choice of :math:`u`, which is why the sum
    says almost nothing about whether a given :math:`u` is interesting, and why
    ``b2_share`` is the column that does.
    """
    rows = states.float()
    energy = rows.pow(2).sum(dim=-1).clamp_min(1e-12)
    unit_v, unit_u = _basis_like(v, u, rows)
    alpha_share = (rows @ unit_v).pow(2) / energy
    b_share = (rows @ unit_u).pow(2) / energy
    total = alpha_share + b_share
    # cos^2 computed independently of the decomposition: against a v that is normalised
    # here rather than assumed to be, and against each token's own norm. Comparing the
    # share to THIS is a real check; comparing it to its own first term would not be,
    # since the share is defined as that term plus a square and the inequality would
    # hold by algebra whatever v and u were.
    cosine = (rows @ unit_v) / rows.norm(dim=-1).clamp_min(1e-12)
    cos_squared = cosine.pow(2)
    slack = float((cos_squared - total).max())
    if slack > 1e-4:
        raise AssertionError(
            f"the in-plane share came out {slack:.3e} BELOW cos^2(x, v*), which is "
            "impossible when the decomposition is sound: cos^2 is the share's first "
            f"term. The (v, u) basis is not orthonormal -- u.v = "
            f"{float(unit_u @ unit_v):.3e}, |v| = {float(v.float().norm()):.6f}, "
            f"|u| = {float(u.float().norm()):.6f} -- so alpha and b are not the "
            "coordinates of one orthogonal decomposition of the same state.")
    # The other half of the same soundness condition, which the inequality above cannot
    # see: a u that is parallel to v double-counts alpha and inflates the share past 1.
    overflow = float((total - 1.0).max())
    if overflow > 1e-4:
        raise AssertionError(
            f"the in-plane share came out {1.0 + overflow:.6f}, above 1. Two orthonormal "
            f"coordinates cannot account for more than the whole token, so u is not "
            f"orthogonal to v: u.v = {float(unit_u @ unit_v):.3e}.")
    return dict(
        alpha2_share=float(alpha_share.mean()),
        b2_share=float(b_share.mean()),
        in_plane_energy_share=float(total.mean()),
        # The mean of per-token cos^2, which is what alpha2_share is. Recorded beside the
        # mean of per-token cos so the two are never confused: a population split in sign
        # has a near-zero mean cosine and a large mean cos^2.
        cos_squared_mean=float(cos_squared.mean()),
        in_plane_share_at_least_cos_squared=bool(slack <= 1e-4),
        # 1.0 would mean the token lies entirely in the plane and the rotation moves all
        # of it; below that, the out-of-plane part rides along unchanged.
        out_of_plane_share=float((1.0 - total).clamp_min(0.0).mean()))


def _plane_coordinates(before: torch.Tensor, after: torch.Tensor,
                       plane: Dict[str, Any], token_ids: Sequence[int], *,
                       layer: int, step: int) -> List[Dict[str, Any]]:
    """Each treated token's real coordinates in the (v*, u) plane, before and after.

    These are the actual activations projected onto the two basis vectors, not a
    schematic: ``(alpha, b)`` is where the token is, and ``out_of_plane`` is how much of
    it neither axis sees. A figure drawn from this is a faithful 2D view of a rotation
    that happened in the full width, and the out-of-plane length is what tells a reader
    how much of the token the picture leaves out.
    """
    rows: List[Dict[str, Any]] = []
    first, second = before.float(), after.float()
    v, u = _basis_like(plane["v"], plane["u"], first)
    for position, token in enumerate(token_ids):
        x0, x1 = first[position], second[position]
        a0, b0 = float(x0 @ v), float(x0 @ u)
        a1, b1 = float(x1 @ v), float(x1 @ u)
        rows.append(dict(
            token=int(token), layer=int(layer), step=int(step),
            alpha_before=a0, b_before=b0, alpha_after=a1, b_after=b1,
            norm_before=float(x0.norm()), norm_after=float(x1.norm()),
            # What the plane does not show. sqrt(||x||^2 - alpha^2 - b^2), and it is
            # identical before and after by construction, because a plane rotation
            # cannot move anything out of the plane.
            out_of_plane_before=float((x0.pow(2).sum() - a0 ** 2 - b0 ** 2)
                                      .clamp_min(0.0).sqrt()),
            out_of_plane_after=float((x1.pow(2).sum() - a1 ** 2 - b1 ** 2)
                                     .clamp_min(0.0).sqrt())))
    return rows


def make_edit(condition: Condition, *, direction: torch.Tensor, token_ids: Sequence[int],
              dominant_channel: int, clean_states: Dict[Tuple[int, int], torch.Tensor],
              seed: int, diary: Optional[List[Dict[str, Any]]] = None,
              subspaces: Optional[Dict[Tuple[int, int], Dict[str, Any]]] = None,
              planes: Optional[Dict[Tuple[int, int], Dict[str, Any]]] = None,
              plane_rows: Optional[List[Dict[str, Any]]] = None) -> Callable:
    r"""Build the single token edit for one condition.

    **Hook ordering is not left to chance.**  Ablation and rescue are composed inside
    *one* edit function, applied in a fixed order:

    1. zero the dominant coordinate at the frozen tokens (``ablate_channel``);
    2. then apply the rescue to the result of step 1.

    So there is no dependence on the order PyTorch happens to fire two hooks, and no
    way for a future caller to install them the other way round.  The rescue target
    always comes from the **paired clean trajectory** at the same ``(step, layer)``,
    never from the ablated run: ``alpha`` measured after the ablation is exactly the
    quantity the ablation destroyed, and reusing it would rescue nothing.
    """
    ids = [int(t) for t in token_ids]
    unit = _unit(direction)
    width = int(unit.numel())
    # `EditInstaller` calls the edit once per batch row when `all_batch_rows` is set, and
    # the conditional row is always first. Without an index the diary would carry two
    # notes per (step, layer) for the CFG twins and their reported perturbation L2 would
    # be double-counted in any mean. The index is recorded so a summary can filter to the
    # conditional row rather than the count being silently wrong.
    calls: Dict[Tuple[int, int], int] = {}
    if condition.edit_direction == "channel":
        # The pure dominant-channel axis e_{c*}. Running the beta operator along this
        # instead of along v* is what separates "the mechanism is a direction" from "the
        # mechanism is channel c*": where v* is nearly axis-aligned the two edits are
        # almost the same operation, and the difference between them is exactly the part
        # of v* that is NOT the dominant channel.
        axis = torch.zeros(width)
        axis[int(dominant_channel) % width] = 1.0
        edit_vector = axis
    elif condition.edit_direction == "rotation":
        # The rotation arm has no single edit vector: it turns the (v*, u) plane, and u
        # comes from `planes` per (step, layer) because `residual_pc1` is fitted on the
        # clean population there. v* is kept for the rescue and diagnostic paths.
        edit_vector = unit
    else:
        edit_vector = {"vstar": unit,
                       "orthogonal": orthogonal_direction(unit, width, seed=seed),
                       "random": random_direction(width, seed=seed)}[condition.edit_direction]

    @torch.no_grad()
    def edit(x: torch.Tensor, ctx: EditContext) -> torch.Tensor:
        if not ids:
            return x
        out = x.float().clone()
        index = torch.as_tensor(ids, device=out.device, dtype=torch.long)
        index = index[index < out.shape[0]]
        if not index.numel():
            return x
        rows = [int(t) for t in index.tolist()]
        before = out[index].clone()
        seen = calls.get((int(ctx.step), int(ctx.layer)), 0)
        calls[(int(ctx.step), int(ctx.layer))] = seen + 1
        note: Dict[str, Any] = dict(layer=int(ctx.layer), step=int(ctx.step),
                                    condition=condition.key, n_tokens=len(rows),
                                    batch_row_call=seen,
                                    is_conditional_row=bool(seen == 0))

        # ---- step 1: ablation, if this condition ablates
        if condition.ablate_channel:
            out = scale_channel(out, int(dominant_channel), 0.0, rows)
            note["channel_absmax_after_ablation"] = float(
                out[index, int(dominant_channel)].abs().max())
        # The state the rescue starts from, kept so the rescue can be measured against
        # the damage rather than against the incoming state.
        after_ablation = out[index].clone()

        # ---- step 1b: the true rotation, if this condition rotates
        #
        # Exclusive with the beta/gamma branch below. Rotating and rescaling
        # in one condition would leave neither measurable: a norm change and an angle
        # change both move cos(x, v*), and this arm varies the angle at a norm that is
        # fixed by the operator rather than by a correction.
        if condition.edit_direction == "rotation":
            plane = (planes or {}).get((int(ctx.step), int(ctx.layer)))
            if plane is None:
                note["rotation_skipped"] = (
                    "no (v*, u) plane was built for this (step, layer)")
            else:
                theta = math.radians(float(condition.theta_deg))
                v_plane, u_plane = plane["v"], plane["u"]
                incoming = out[index].clone()
                rotated = rotate_plane(incoming, v_plane, u_plane, theta)
                # Asserted, not corrected. A norm that moves means u was not unit or not
                # orthogonal to v, and renormalising the output would hide the bug that
                # makes every angle in the arm wrong.
                error = rotation_norm_error(incoming, rotated)
                if error > _ROTATION_NORM_TOLERANCE:
                    raise AssertionError(
                        f"the rotation changed the token norm by {error:.3e} relative, "
                        f"above the {_ROTATION_NORM_TOLERANCE:.0e} tolerance. A plane "
                        f"rotation preserves norm exactly, so this means the (v*, u) "
                        f"basis is not orthonormal: u.v = {plane['u_dot_v']:.3e}, "
                        f"|u| = {float(plane['u'].norm()):.6f}. Not corrected by "
                        "renormalising, because that would leave the angles wrong and "
                        "the norms right.")
                out[index] = rotated
                note.update(_rotation_note(incoming, rotated, plane, condition, error))
                if plane_rows is not None and seen == 0:
                    plane_rows.extend(_plane_coordinates(
                        incoming, rotated, plane, rows, layer=int(ctx.layer),
                        step=int(ctx.step)))

        # ---- step 2: the beta/gamma control surface, if this condition moves it
        elif condition.beta != 1.0 or condition.gamma != 1.0:
            edited, degenerate = control_surface_states(
                out[index], edit_vector, condition.beta, condition.gamma, seed=seed)
            note["degenerate_tokens"] = degenerate
            if condition.match_energy_to is not None:
                # Match against what the v* edit would do to *this* tensor, not to the
                # clean state. Under a windowed intervention the trajectory has already
                # diverged by the second edited layer, so a target precomputed from the
                # clean pass would drift; recomputing it on the live state keeps the
                # control exactly matched at every layer it fires at.
                reference, _ = control_surface_states(
                    out[index], unit, condition.beta, condition.gamma, seed=seed)
                target = (reference - out[index]).norm(dim=-1)
                edited = match_perturbation(out[index], edited, target)
                note["energy_matched_to"] = condition.match_energy_to
                note["match_target_l2"] = float(target.pow(2).sum().sqrt())
            out[index] = edited

        # ---- step 2': the rescue, always after the ablation
        clean = clean_states.get((int(ctx.step), int(ctx.layer)))
        if condition.rescue != "none":
            if clean is None:
                note["rescue_skipped"] = "no clean state probed at this (step, layer)"
            else:
                # ``clean`` is intentionally allowed to live on CPU while ``out`` follows
                # the model device.  PyTorch requires an index tensor to be on CPU or on
                # the same device as the tensor being indexed, so keep separate indices
                # and direction copies for the cached clean state and the live state.
                clean_index = index.to(device=clean.device)
                unit_clean = unit.to(device=clean.device, dtype=torch.float32)
                unit_out = unit.to(device=out.device, dtype=torch.float32)

                if condition.rescue == "vstar":
                    # Set x . v* back to the clean value, leaving the orthogonal part alone.
                    clean_alpha = (
                        clean[clean_index].float() @ unit_clean
                    ).unsqueeze(-1).to(out.device)
                    live_alpha = (
                        out[index].float() @ unit_out
                    ).unsqueeze(-1)
                    delta_alpha = clean_alpha - live_alpha
                    injection = delta_alpha * unit_out
                    out[index] = out[index] + injection

                    # Keep the legacy field for compatibility, and also record diagnostics
                    # that test the rescue at the edit hook itself. Downstream layers are
                    # allowed to transform either perturbation and are scientific outcomes,
                    # not implementation-validity checks.
                    post_alpha = (
                        out[index].float() @ unit_out
                    ).unsqueeze(-1)
                    injection_l2 = float(injection.pow(2).sum().sqrt())
                    note["rescue_alpha_l2"] = float(delta_alpha.abs().pow(2).sum())
                    note["rescue_injection_l2"] = injection_l2
                    note["rescue_projection_change_l2"] = float(
                        (post_alpha - live_alpha).pow(2).sum().sqrt())
                    note["rescue_target_alpha_error_l2"] = float(
                        (post_alpha - clean_alpha).pow(2).sum().sqrt())
                    note["rescue_target_alpha_rel_error"] = float(
                        (post_alpha - clean_alpha).pow(2).sum().sqrt()
                        / clean_alpha.pow(2).sum().sqrt().clamp_min(1e-12))

                elif condition.rescue == "orthogonal":
                    # Rotate the *signed* per-token v* rescue coefficients into one
                    # deterministic direction orthogonal to v*. This preserves the
                    # token-wise sign pattern while matching the v* rescue L2 exactly.
                    clean_alpha = (
                        clean[clean_index].float() @ unit_clean
                    ).unsqueeze(-1).to(out.device)
                    live_alpha = (
                        out[index].float() @ unit_out
                    ).unsqueeze(-1)
                    delta_alpha = clean_alpha - live_alpha
                    perp = orthogonal_direction(unit, width, seed=seed).to(
                        device=out.device, dtype=torch.float32)
                    injection = delta_alpha * perp
                    out[index] = out[index] + injection

                    post_alpha = (
                        out[index].float() @ unit_out
                    ).unsqueeze(-1)
                    injection_l2 = float(injection.pow(2).sum().sqrt())
                    note["rescue_alpha_l2"] = float(delta_alpha.abs().pow(2).sum())
                    note["rescue_injection_l2"] = injection_l2
                    note["rescue_projection_change_l2"] = float(
                        (post_alpha - live_alpha).pow(2).sum().sqrt())
                    note["rescue_target_alpha_error_l2"] = float(
                        (post_alpha - clean_alpha).pow(2).sum().sqrt())
                    note["rescue_projection_change_rel_to_injection"] = float(
                        (post_alpha - live_alpha).pow(2).sum().sqrt()
                        / max(injection_l2, 1e-12))

                elif condition.rescue == "channel":
                    clean_channel = clean[
                        clean_index, int(dominant_channel)
                    ].to(device=out.device, dtype=out.dtype)
                    out[index, int(dominant_channel)] = clean_channel
                    note["restored_channel"] = int(dominant_channel)


        note["perturbation_l2"] = float((out[index] - before).pow(2).sum().sqrt())
        note["perturbation_energy"] = float((out[index] - before).pow(2).sum())

        # Is the edited state one this layer's population could have contained? The
        # operators control norm and alignment by construction, so neither can answer
        # that; the clean subspace can.
        subspace = (subspaces or {}).get((int(ctx.step), int(ctx.layer)))
        if subspace:
            note["onmanifold_energy"] = onmanifold_energy(out[index], subspace)
            note["onmanifold_energy_before"] = onmanifold_energy(before, subspace)
            note["onmanifold_clean_median"] = subspace["clean_share_median"]
            note["onmanifold_clean_p10"] = subspace["clean_share_p10"]
            note["onmanifold_rank"] = subspace["rank"]
            # The diagnostic only discriminates when the subspace is a genuine restriction.
            # Where the requested rank approaches the width (a narrow model, or a rank set
            # too high) it captures everything and every state looks on-manifold. The
            # variance it explains is recorded so that case is visible rather than read as
            # a result.
            note["onmanifold_variance_explained"] = subspace["variance_explained"]
            note["onmanifold_rank_fraction"] = subspace["rank_fraction"]
            note["onmanifold_is_discriminative"] = subspace["is_discriminative"]
            # 1.0 means the treated tokens sit as deep in the clean subspace as a typical
            # clean token; well below 1.0 means the edit left it.
            note["onmanifold_ratio_to_clean"] = (
                note["onmanifold_energy"] / subspace["clean_share_median"]
                if subspace["clean_share_median"] > 1e-12 else float("nan"))

        # How far the treated state sits from the clean one, before and after whatever
        # this condition did. This is the number that separates a rescue from a
        # displacement, and its absence is why an L2-matched orthogonal control can look
        # like a suspiciously effective rescue downstream.
        #
        # Matching the *injection* L2 is the right way to control for effort, but it does
        # not control for the resulting distance from clean: the v* rescue spends its
        # budget moving back toward the clean state, while the orthogonal control spends
        # the same budget moving sideways. Starting a distance d from clean and adding an
        # orthogonal step of length d lands at sqrt(2).d, further from clean than the
        # ablation alone. A condition with ratio > 1 is not a matched null; it is a larger
        # perturbation than the damage it is meant to control for, and any downstream
        # "recovery" it shows has to be read in that light.
        if clean is not None:
            reference = clean[index.to(device=clean.device)].float().to(out.device)
            gap = lambda state: float((state - reference).pow(2).sum().sqrt())
            incoming, damaged, final = gap(before), gap(after_ablation), gap(out[index])
            note["clean_state_l2"] = float(reference.pow(2).sum().sqrt())
            # Zero at the first edited layer, where the incoming state IS the clean one.
            # Nonzero at later layers of a windowed edit, because upstream edits have
            # already moved the trajectory: a fact worth having rather than an error.
            note["distance_from_clean_incoming_l2"] = incoming
            note["distance_from_clean_after_ablation_l2"] = damaged
            note["distance_from_clean_after_rescue_l2"] = final
            # The ratio compares a rescue against the damage it is meant to undo, so it
            # is defined only where there is damage to undo. For a grid cell or a control
            # that never ablates, `damaged` is just the incoming state, zero at the
            # first edited layer, and the ratio would be a division by nothing dressed
            # up as a finding. Those conditions get the absolute gap and no ratio.
            #
            # Below 1: the rescue moved the state back toward the clean trajectory.
            # Above 1: it moved further away, so it is a larger perturbation than the
            # damage it controls for; sqrt(2) is the signature of an orthogonal step of
            # the gap's own length.
            if condition.ablate_channel and condition.rescue != "none" and damaged > 1e-12:
                note["rescue_closed_distance_ratio"] = final / damaged
            elif condition.ablate_channel and condition.rescue == "none":
                note["rescue_closed_distance_ratio"] = 1.0        # the damage itself
        if diary is not None:
            diary.append(note)
        return out

    return edit


def vstar_rescue_energy(clean: torch.Tensor, ablated: torch.Tensor, direction: torch.Tensor,
                        token_ids: Sequence[int]) -> torch.Tensor:
    """Per-token L2 the ``v*`` rescue must inject, for the matched orthogonal control.

    ``clean`` and ``ablated`` need not live on the same device.  The result is returned on
    ``ablated.device`` so callers can use it directly with the live intervention tensor.
    """
    unit = _unit(direction)
    ids = [int(t) for t in token_ids]
    clean_index = torch.as_tensor(ids, device=clean.device, dtype=torch.long)
    ablated_index = torch.as_tensor(ids, device=ablated.device, dtype=torch.long)
    clean_alpha = clean[clean_index].float() @ unit.to(
        device=clean.device, dtype=torch.float32)
    ablated_alpha = ablated[ablated_index].float() @ unit.to(
        device=ablated.device, dtype=torch.float32)
    return (clean_alpha.to(ablated.device) - ablated_alpha).abs()


# ================================================================ measurement
def _ranks(values: torch.Tensor) -> torch.Tensor:
    """Rank 1 = largest, so "rank 1 by norm" reads the way a reader expects."""
    order = torch.argsort(values, descending=True)
    ranks = torch.empty_like(order)
    ranks[order] = torch.arange(1, order.numel() + 1, device=order.device)
    return ranks


def sink_readout(observation, *, n_tokens: int, threshold: float,
                 absolute: bool = False) -> Dict[str, torch.Tensor]:
    """Per-token sinkhood, using the repository's existing criterion.

    ``ditsinks.config.SweepConfig.sink_ratio_threshold`` defines a sink as absorbing at
    least ``threshold`` times the uniform share ``1/N`` of the image->image attention
    mass, and ``metrics.layer_table`` applies it to a layer's head-max.  The same rule is
    applied here per token, so ``is_sink`` means exactly what it means everywhere else in
    this project. ``n_sink_heads`` is the repository's other notion (how many heads
    take this token as their strongest image key), and both are reported because they
    are not the same question.

    ``absolute=True`` scores against uniform over the **whole** key sequence instead,
    undoing the image-only renormalisation.  The two answer different questions, and the
    difference matters as soon as a condition moves attention off the image.  The default
    is a share of the image-directed mass, so it is *exactly* invariant to a virtual
    register drawing mass proportionally: the quantity it divides out is precisely the
    mass the register took, which makes it unable to show a register taking sink duty
    away from the tokens that held it.  ``absolute=True`` registers that loss.  Neither
    is inflated by the register's presence; the relative one simply cannot see it.
    Needs ``observation.image_mass`` and ``observation.n_keys``, recorded from the pass;
    a trace from before those existed raises rather than silently scoring relatively.
    """
    empty = torch.zeros(n_tokens)
    if observation is None or observation.incoming is None:
        return {"sink_strength_headmax": empty.clone() * float("nan"),
                "sink_strength_headmean": empty.clone() * float("nan"),
                "n_sink_heads": empty.clone(), "is_sink": empty.clone().bool(),
                "incoming_headmean": empty.clone() * float("nan"),
                "qk_score": empty.clone() * float("nan"),
                "qk_rank": empty.clone().long()}
    incoming = observation.incoming.float()                     # [heads, tokens]
    uniform = 1.0 / max(int(incoming.shape[-1]), 1)
    if absolute:
        mass, n_keys = observation.image_mass, observation.n_keys
        if mass is None or not n_keys:
            raise ValueError(
                "absolute sinkhood needs the pre-renormalisation image mass and the full "
                "key length, which this observation does not carry. Traces recorded "
                "before LayerObservation.image_mass existed cannot be rescored; re-run "
                "the pass, or pass absolute=False and correct for the shrinking pie by "
                "hand (subtract the register-only condition's own contribution)."
            )
        # Undo the renormalisation, then score against uniform over the whole sequence,
        # registers included. The result no longer depends on how much mass left the image.
        incoming = incoming * mass.float().to(incoming).unsqueeze(-1)
        uniform = 1.0 / max(int(n_keys), 1)
    head_argmax = incoming.argmax(dim=-1)
    counts = torch.zeros(int(incoming.shape[-1]))
    for token in head_argmax.tolist():
        counts[int(token)] += 1.0
    headmax = incoming.max(dim=0).values / uniform
    qk = (observation.qk_cosine.float().mean(dim=0) if observation.qk_cosine is not None
          else torch.full((int(incoming.shape[-1]),), float("nan")))
    return {"sink_strength_headmax": headmax,
            "sink_strength_headmean": incoming.mean(dim=0) / uniform,
            "n_sink_heads": counts,
            "is_sink": headmax >= float(threshold),
            "incoming_headmean": incoming.mean(dim=0),
            "qk_score": qk,
            "qk_rank": _ranks(torch.nan_to_num(qk, nan=-1e9))}


def token_rows(trace: Trace, clean: Trace, treatment: Treatment, *, condition: Condition,
               layers: Sequence[int], steps: Sequence[int], direction: torch.Tensor,
               dominant_channel: int, sink_threshold: float, edit_layers: Sequence[int] = (),
               all_tokens: bool = False) -> List[Dict[str, Any]]:
    """Per-token representation and sink rows.

    ``all_tokens`` widens the readout from the treatment set to the whole image slice.
    It is used for the clean run only: the population scatters (projection against norm,
    projection against sink strength) are statements about the clean model's structure,
    and writing every token for all 60-odd conditions would multiply the table by the
    grid for no additional evidence.
    """
    unit = _unit(direction)
    selected = set(treatment.token_ids)
    sinks = set(treatment.clean_sink_ids)
    edited = {int(l) for l in edit_layers}
    first_edited = min(edited) if edited else None
    rows: List[Dict[str, Any]] = []
    for step in steps:
        for layer in layers:
            observation = trace.at(int(step), int(layer))
            if observation is None or observation.norm is None:
                continue
            norms = observation.norm.float()
            alpha = (observation.projection.float() if observation.projection is not None
                     else torch.full_like(norms, float("nan")))
            cosine = (observation.cosine.float() if observation.cosine is not None
                      else alpha / norms.clamp_min(1e-9))
            n = int(norms.numel())
            norm_rank, alpha_rank = _ranks(norms), _ranks(torch.nan_to_num(alpha, nan=-1e9))
            channel = (observation.channel_values.float()[0] if observation.channel_values
                       is not None and observation.channel_values.shape[0] else
                       torch.full_like(norms, float("nan")))
            sink = sink_readout(observation, n_tokens=n, threshold=sink_threshold)
            keys = observation.keys
            clean_observation = clean.at(int(step), int(layer))
            clean_norm = (clean_observation.norm.float() if clean_observation is not None
                          and clean_observation.norm is not None else None)
            clean_alpha = (clean_observation.projection.float() if clean_observation is not None
                           and clean_observation.projection is not None else None)
            wanted = range(n) if all_tokens else sorted(t for t in selected if t < n)
            for token in wanted:
                rows.append(dict(
                    condition=condition.key, arm=condition.arm, role=condition.role,
                    beta=condition.beta, gamma=condition.gamma,
                    intervention=condition.key, step=int(step), layer=int(layer),
                    is_edited_layer=bool(int(layer) in edited),
                    is_downstream=(None if first_edited is None
                                   else bool(int(layer) >= first_edited)),
                    token=int(token), is_selected=bool(token in selected),
                    was_clean_sink=bool(token in sinks),
                    norm=float(norms[token]), alpha=float(alpha[token]),
                    cosine=float(cosine[token]),
                    dominant_channel_value=float(channel[token]),
                    norm_rank=int(norm_rank[token]), alpha_rank=int(alpha_rank[token]),
                    sink_strength=float(sink["sink_strength_headmax"][token]),
                    sink_strength_headmean=float(sink["sink_strength_headmean"][token]),
                    n_sink_heads=int(sink["n_sink_heads"][token]),
                    is_sink=bool(sink["is_sink"][token]),
                    incoming_attention=float(sink["incoming_headmean"][token]),
                    qk_score=float(sink["qk_score"][token]),
                    qk_rank=int(sink["qk_rank"][token]),
                    key_norm=(float(keys[:, token, :].float().norm(dim=-1).mean())
                              if keys is not None and token < keys.shape[1] else float("nan")),
                    clean_norm=None if clean_norm is None else float(clean_norm[token]),
                    clean_alpha=None if clean_alpha is None else float(clean_alpha[token]),
                ))
    return rows


def population_rows(trace: Trace, clean: Trace, treatment: Treatment, *,
                    condition: Condition, layers: Sequence[int], steps: Sequence[int],
                    norm_threshold: float, sink_threshold: float,
                    alignment_threshold: float = 0.0,
                    edit_layers: Sequence[int] = (),
                    register_layers: Sequence[int] = (),
                    treatment_targets: Optional[FrozenTargets] = None,
                    recovery_floor: float = 1.0) -> List[Dict[str, Any]]:
    """Whole-population readouts: does the high-norm state still exist at all?

    ``highnorm_retention`` is the fraction of the *clean* high-norm tokens that are still
    high-norm under treatment: a paired quantity, so a condition that destroys the
    population and a condition that relocates it do not score the same.
    """
    rows: List[Dict[str, Any]] = []
    edited = {int(l) for l in edit_layers}
    register = {int(l) for l in register_layers}
    first_edited = min(edited) if edited else None
    if treatment_targets is None:
        # `ordinary_projection_spread` excludes the register groups when it measures the
        # ordinary population, so it needs the frozen selection rather than a bare id list.
        treatment_targets = FrozenTargets(prompt_id=0, seed=0, step=0, layer=0,
                                          register_ids=tuple(treatment.token_ids))
    for step in steps:
        for layer in layers:
            observation, reference = trace.at(int(step), int(layer)), clean.at(int(step), int(layer))
            if observation is None or observation.norm is None:
                continue
            norms = observation.norm.float()
            alpha = (observation.projection.float() if observation.projection is not None
                     else torch.full_like(norms, float("nan")))
            loud = norms >= float(norm_threshold)
            ids = [t for t in treatment.token_ids if t < int(norms.numel())]
            sink = sink_readout(observation, n_tokens=int(norms.numel()), threshold=sink_threshold)
            aligned_bar = float(alignment_threshold)
            cosine_all = alpha / norms.clamp_min(1e-9)
            both = loud & (cosine_all.abs() >= aligned_bar) if aligned_bar > 0 else loud
            row = dict(
                condition=condition.key, arm=condition.arm, role=condition.role,
                beta=condition.beta, gamma=condition.gamma, intervention=condition.key,
                step=int(step), layer=int(layer), n_tokens=int(norms.numel()),
                is_edited_layer=bool(int(layer) in edited),
                is_downstream=(None if first_edited is None
                               else bool(int(layer) >= first_edited)),
                # Where the register state is supposed to exist. Outside it, a recovery
                # ratio is not a meaningful quantity however carefully it is computed.
                is_register_zone=(None if not register else bool(int(layer) in register)),
                norm_threshold=float(norm_threshold),
                n_highnorm=int(loud.sum()),
                # `n_highnorm` is pinned along the beta axis by construction: beta
                # renormalises every treated token, so no count defined by a norm
                # threshold alone can move. That makes it a positive control rather than
                # an endpoint. This is the population count that CAN move, and it is the
                # one the project actually cares about: loud AND aligned.
                n_highnorm_and_aligned=int(both.sum()),
                alignment_threshold=aligned_bar,
                norm_median=float(norms.median()), norm_p99=float(torch.quantile(norms, 0.99)),
                norm_max=float(norms.max()),
                alpha_mean=float(alpha.mean()), alpha_p99=float(torch.quantile(alpha, 0.99)),
                alpha_top1=float(alpha.max()),
                selected_norm_mean=float(norms[ids].mean()) if ids else float("nan"),
                selected_alpha_mean=float(alpha[ids].mean()) if ids else float("nan"),
                selected_cosine_mean=float((alpha[ids] / norms[ids].clamp_min(1e-9)).mean())
                if ids else float("nan"),
                n_sinks=int(sink["is_sink"].sum()),
                mean_sink_strength=float(sink["sink_strength_headmax"].mean()),
                max_sink_strength=float(sink["sink_strength_headmax"].max()),
                selected_sink_strength_mean=float(sink["sink_strength_headmax"][ids].mean())
                if ids else float("nan"),
                n_selected_are_sinks=int(sink["is_sink"][ids].sum()) if ids else 0,
            )
            if reference is not None and reference.norm is not None:
                clean_loud = reference.norm.float() >= float(norm_threshold)
                row["n_highnorm_clean"] = int(clean_loud.sum())
                if reference.cosine is not None and aligned_bar > 0:
                    clean_both = clean_loud & (reference.cosine.float().abs() >= aligned_bar)
                    row["n_highnorm_and_aligned_clean"] = int(clean_both.sum())
                    row["aligned_population_retention"] = (
                        float((both & clean_both).sum() / clean_both.sum())
                        if int(clean_both.sum()) else float("nan"))
                row["highnorm_retention"] = (float((loud & clean_loud).sum() / clean_loud.sum())
                                             if int(clean_loud.sum()) else float("nan"))
                union = int((loud | clean_loud).sum())
                row["highnorm_jaccard"] = (float((loud & clean_loud).sum() / union)
                                           if union else float("nan"))
                if reference.projection is not None:
                    clean_alpha = reference.projection.float()
                    row["clean_alpha_top1"] = float(clean_alpha.max())
                    clean_mean = float(clean_alpha[ids].mean()) if ids else float("nan")
                    row["clean_selected_alpha_mean"] = clean_mean
                    # The natural unit for a projection change, and the same one
                    # `endpoints.measure_layer` uses for `vstar_projection_change`: how
                    # many ordinary-token standard deviations the register moved. Defined
                    # everywhere, including where the register state has dissolved.
                    # How much room beta has left to rotate the state toward v*.
                    # cos -> 1 asymptotically in beta, so where the clean state already
                    # sits at |cos| ~ 0.99 there is almost nothing for a positive beta to
                    # add, and an ineffective edit there is a ceiling rather than a null
                    # result. This is the quantity that distinguishes the two.
                    if reference.cosine is not None and ids:
                        clean_cosine = reference.cosine.float()[ids]
                        # SIGNED, to match `selected_cosine_mean`, which is also signed.
                        # The two are compared directly: at beta = gamma = 1 they must be
                        # equal, since that cell is a no-op, and mixing a mean of
                        # absolute values with a mean of signed ones breaks that identity
                        # wherever the register population has mixed signs. The sign is
                        # also the whole content of a negative beta, so taking |cos| here
                        # would hide the effect the depth sweep is meant to measure.
                        row["clean_cosine_signed"] = float(clean_cosine.mean())
                        # ABSOLUTE, for the ceiling: |cos| approaches 1 asymptotically in
                        # beta, so the room left to move is 1 - |cos| regardless of sign.
                        row["clean_cosine_abs"] = float(clean_cosine.abs().mean())
                        row["clean_cosine_headroom"] = 1.0 - float(clean_cosine.abs().mean())
                        # Whether the signed mean means anything: a population split
                        # between aligned and anti-aligned tokens averages to near zero
                        # and neither the comparison nor the ceiling is then readable.
                        signs = torch.sign(clean_cosine)
                        modal = torch.sign(signs.sum()) if float(signs.sum()) != 0 else 1.0
                        row["clean_cosine_sign_agreement"] = float(
                            (signs == modal).float().mean())
                    spread = EP.ordinary_projection_spread(clean, int(step), int(layer),
                                                           treatment_targets)
                    row["ordinary_alpha_spread"] = spread
                    row["selected_alpha_change_in_spreads"] = (
                        (float(alpha[ids].mean()) - clean_mean) / spread
                        if ids and spread > 0 else float("nan"))
                    # A RATIO to the clean mean is only a summary where that mean is
                    # resolvable. In the dissolution zone the register projection
                    # collapses toward zero by construction, so the denominator vanishes
                    # and a mean-of-ratios over depth is dominated by whichever layer
                    # happened to divide by the smallest number, which can turn a large
                    # perturbation into a spurious "400% recovery". The ratio is therefore
                    # reported only where the clean mean clears the ordinary spread, and
                    # `recovery_denominator_resolvable` says where that was.
                    resolvable = bool(ids and spread > 0
                                      and abs(clean_mean) >= float(recovery_floor) * spread)
                    row["recovery_denominator_resolvable"] = resolvable
                    row["selected_alpha_recovery"] = (
                        float(alpha[ids].mean() / clean_mean) if resolvable else float("nan"))
                # Sink recovery: of the heads whose clean argmax was a treated token, how
                # many still choose a treated token. Uses the same per-head argmax record
                # that FrozenTargets froze on the clean run.
                if reference.incoming is not None and observation.incoming is not None:
                    chosen = set(ids)
                    clean_pick = reference.incoming.float().argmax(dim=-1)
                    live_pick = observation.incoming.float().argmax(dim=-1)
                    affected = [h for h in range(int(clean_pick.numel()))
                                if int(clean_pick[h]) in chosen]
                    row["n_affected_heads"] = len(affected)
                    row["sink_retention"] = (
                        float(np.mean([1.0 if int(live_pick[h]) == int(clean_pick[h]) else 0.0
                                       for h in affected])) if affected else float("nan"))
                    row["sink_still_on_target"] = (
                        float(np.mean([1.0 if int(live_pick[h]) in chosen else 0.0
                                       for h in affected])) if affected else float("nan"))
            rows.append(row)
    return rows


def _load_lpips():
    """LPIPS if installed and its weights are reachable, else ``None``.

    Same graceful degradation as :mod:`ditsinks.q11`: a perceptual metric is worth
    having and is not worth failing a GPU run over.
    """
    try:
        import lpips

        return lpips.LPIPS(net="alex", verbose=False)
    except Exception:
        return None


def _load_clip(name: Optional[str]):
    """A CLIP model for prompt/image similarity, only when explicitly configured.

    Returns ``None`` unless ``name`` is given.  Left off by default on purpose: CLIP
    weights are a download, and a metrics table that silently fetches a few hundred
    megabytes in the middle of a run is not a metrics table anybody asked for.
    """
    if not name:
        return None
    try:
        from transformers import CLIPModel, CLIPProcessor

        return CLIPModel.from_pretrained(name).eval(), CLIPProcessor.from_pretrained(name)
    except Exception:
        return None


def image_metrics(clean_image, treated_image, *, prompt: str = "", lpips_net=None,
                  clip=None, detail_quantile: float = 0.8) -> Dict[str, float]:
    """Paired image distance plus the three local-detail readouts.

    Paired means same prompt, same seed, same sampler: any distance is attributable to
    the intervention rather than to sampling.
    """
    from .q11 import _image_fidelity

    row: Dict[str, float] = dict(_image_fidelity(clean_image, treated_image, lpips_net))
    row.update(ID.detail_metrics(clean_image, treated_image, quantile=detail_quantile))
    if clip is not None and clean_image is not None and treated_image is not None:
        model, processor = clip
        try:
            with torch.no_grad():
                batch = processor(text=[prompt or ""], images=[clean_image, treated_image],
                                  return_tensors="pt", padding=True, truncation=True)
                out = model(**batch)
                images = torch.nn.functional.normalize(out.image_embeds, dim=-1)
                text = torch.nn.functional.normalize(out.text_embeds, dim=-1)
            row["clip_image_similarity"] = float(images[0] @ images[1])
            row["clip_prompt_similarity_clean"] = float(images[0] @ text[0])
            row["clip_prompt_similarity_treated"] = float(images[1] @ text[0])
            row["clip_prompt_similarity_drop"] = (row["clip_prompt_similarity_clean"]
                                                  - row["clip_prompt_similarity_treated"])
        except Exception:
            pass
    return row


# ===================================================================== config
@dataclass
class ControlSurfaceConfig:
    """Every knob the experiment exposes, with the repository's values as defaults.

    Nothing here is read from a generated outcome.  ``dominant_channel``, the layer
    window and the denoising step all default to the frozen discovery artifact via
    :class:`~ditsinks.questions.QuestionContext`, and are overridable only explicitly.
    """

    experiment_root: Path
    betas: Tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0)
    gammas: Tuple[float, ...] = (0.5, 0.75, 1.0, 1.25, 1.5)
    # "write" edits the first register layer only (one write); "window" edits the whole
    # register range (a maintained state). Running both makes that an ablation rather
    # than an assumption about which one the mechanism needs.
    layer_scopes: Tuple[str, ...] = ("write", "window")
    steps: Optional[Tuple[int, ...]] = None            # None -> the context's capture step
    point: InterventionPoint = InterventionPoint.BLOCK_INPUT
    selection_mode: str = "highnorm_and_aligned"
    topk: int = 8
    projection_percentile: float = 99.0
    dominant_channel: Optional[int] = None             # None -> artifact's channel
    run_grid: bool = True
    run_rescue: bool = True
    run_controls: bool = True
    # Only meaningful on a CFG-batched pipeline. On FLUX the transformer runs at batch 1
    # with guidance as an embedding, so "edit every row" and "edit the conditional row"
    # are literally the same tensor operation and the arm is skipped with a note.
    run_cfg_diagnostic: bool = True
    detail_quantile: float = 0.8
    amplification: float = 10.0
    # A recovery ratio is suppressed only where the clean projection is *vanishing*
    # relative to the ordinary population's own spread. A tenth of a spread is a genuinely
    # degenerate denominator; a whole spread would also discard weakly separated
    # populations, where the ratio is noisy but still the quantity of interest. The real
    # defence against a mean-of-ratios blowing up is restricting the summary to the
    # register zone, which `is_register_zone` supports and the notebook does.
    recovery_floor: float = 0.1
    # The on-manifold diagnostic: is an edited state one the clean population could have
    # contained? Costs one SVD of the clean slice per (step, layer) per unit, which is
    # negligible beside a generation, and is the only measurement here that can tell a
    # meaningful opposite-v* state from off-manifold corruption of the same magnitude.
    measure_manifold: bool = True
    manifold_rank: int = 32
    # The writer-phase stage probe: read the treated tokens' projection at four points
    # around the edited block, so "the edit was overwritten by the MLP" and "the edit
    # was applied and the state stayed suppressed" become different numbers rather than
    # two readings of one. Off by default, so the main grid is unchanged.
    measure_stages: bool = False
    # The true-rotation arm. Off by default: it is a separate experiment with its own
    # angles and its own reference cell, and turning it on inside the main grid's run
    # would mix two operators in one table.
    run_rotation: bool = False
    rotation_angles_deg: Tuple[float, ...] = ROTATION_ANGLES_DEG
    rotation_targets: Tuple[str, ...] = ("random_orthogonal", "residual_pc1")
    # The `semantic_direction` target's vector, in the model's width and in v*'s basis.
    # There is no default and none is derived: the mode refuses without one rather than
    # substituting a random direction and reporting it as semantic.
    semantic_direction: Optional[torch.Tensor] = None
    # Storing the attention keys is the only way to report a key norm, and it is far more
    # expensive than it sounds: the tracer's key_layers path keeps the keys, the queries,
    # AND the full attention probabilities and logits, which are O(heads x tokens x
    # sequence). At FLUX's 512px shapes that is ~312 MB *per layer-step*, 6.6 GB over
    # the register window, on top of the model's own 24 GB. It is opt-in, and
    # `store_keys_layers` caps how many layers it applies to so a stray True cannot OOM
    # a run. qk_score and qk_rank need none of this; they come from the cheap
    # [heads, tokens] qk_cosine the tracer always records.
    store_keys: bool = False
    store_keys_layers: int = 1
    clip_model: Optional[str] = None
    save_images: bool = True
    save_difference_maps: bool = True
    resume: bool = True

    def layers_for(self, ctx: QuestionContext, scope: str) -> List[int]:
        """Intervention layers for a named scope, taken from the frozen artifact.

        The three single-layer depth scopes place the *same* edit at three points in the
        register's life, so a difference between them is about when the state is touched
        rather than about what was done to it. Every offset is derived from the frozen
        layer ranges, so a checkpoint with a different geometry gets the corresponding
        layers rather than FLUX's numbers:

        ``birth``        one layer inside the writer range, before it ends; the state is
                         still being written. FLUX 18, PixArt-Sigma 9.
        ``boundary``     the first register layer, which is what ``write`` already uses and
                         where the main grid was run. FLUX 20, PixArt-Sigma 10.
        ``established``  four layers into the register zone, where the state is maintained
                         rather than formed. FLUX 24, PixArt-Sigma 14.
        """
        last = int(ctx.driver.n_layers) - 1
        base, _ = split_scope(scope)
        if base == "write":
            return [int(ctx.intervention_layer)]
        if base == "window":
            return [int(l) for l in ctx.register_layers]
        if base in _DEPTH_SCOPES:
            return [_depth_layer(ctx, base, last)]
        raise KeyError(f"unknown layer scope {scope!r}; known: 'write', 'window', "
                       + ", ".join(repr(k) for k in _DEPTH_SCOPES)
                       + ", each optionally suffixed '_postmlp'")

    def point_for(self, scope: str) -> InterventionPoint:
        """Where in the block this scope's edit lands.

        The suffix wins where there is one, so a run can put the same edit at the block
        input and at the block output and compare them; every scope without a suffix
        gets the configuration's own point, which is what every existing scope does.
        """
        _, point = split_scope(scope)
        return self.point if point is None else point

    def points_in_use(self) -> List[InterventionPoint]:
        """Every distinct intervention point this configuration's scopes will hook."""
        seen: List[InterventionPoint] = []
        for scope in (self.point, *(self.point_for(s) for s in self.layer_scopes)):
            if scope not in seen:
                seen.append(scope)
        return seen

    def resolved_steps(self, ctx: QuestionContext) -> List[int]:
        return [int(s) for s in (self.steps if self.steps is not None else [ctx.step])]

    def as_dict(self, ctx: Optional[QuestionContext] = None) -> Dict[str, Any]:
        row = {k: v for k, v in asdict(self).items()}
        row["experiment_root"] = str(self.experiment_root)
        row["point"] = self.point.value
        if ctx is not None:
            row["resolved"] = dict(
                checkpoint=getattr(ctx.cfg.spec, "key", ""),
                repo_id=getattr(ctx.cfg.spec, "repo_id", ""),
                dominant_channel=int(self.dominant_channel if self.dominant_channel
                                     is not None else ctx.dominant_channel),
                intervention_layer=int(ctx.intervention_layer),
                register_layers=[int(l) for l in ctx.register_layers],
                steps=self.resolved_steps(ctx),
                num_inference_steps=int(ctx.cfg.num_inference_steps),
                guidance_scale=float(ctx.cfg.guidance_scale or 0.0),
                height=int(ctx.cfg.height), width=int(ctx.cfg.width),
                dtype=str(ctx.cfg.dtype), prompts=list(ctx.prompts),
                seeds=[int(s) for s in ctx.seeds],
                norm_threshold_rule=f"{ctx.highnorm_ratio:g} x median residual norm",
                sink_criterion=(f"incoming image->image mass >= "
                                f"{getattr(ctx.cfg, 'sink_ratio_threshold', 10.0):g} x uniform "
                                "share (SweepConfig.sink_ratio_threshold)"),
                vstar_source=dict(ctx.vstar_metadata),
            )
        return row


# The depth comparison: one edit, three points in the register's life. Offsets rather than
# layer numbers, so the frozen ranges decide the layers on every checkpoint.
_DEPTH_SCOPES: Dict[str, str] = {
    "birth": "inside the writer range, while the state is still being written",
    "boundary": "the first register layer -- where the main grid was run",
    "established": "four layers into the register zone, where the state is maintained",
    # The writer range block by block. Editing "the writer phase" at one layer says
    # nothing about which writer layer matters, and on FLUX the range spans the
    # dual/single boundary: 17 and 18 are MMDiT blocks, 19 is a fused single block,
    # so the three are not interchangeable even in architecture.
    "writer+0": "the first writer layer",
    "writer+1": "the second writer layer",
    "writer+2": "the third writer layer",
}
_ESTABLISHED_OFFSET = 4
# The three life-stage depths, which `depth_layer_table` lists by default.
_LIFE_STAGE_SCOPES: Tuple[str, ...] = ("birth", "boundary", "established")

# A scope may name where in the block the edit lands as well as which layer. The suffix
# is part of the scope name so it reaches the cache key, the output directory and every
# table without a second parallel argument to thread through the runner.
#
#   "birth"            -> the config's point, which is BLOCK_INPUT: the tensor the block
#                         receives, BEFORE attention and before the feed-forward.
#   "birth_postmlp"    -> BLOCK_OUTPUT: the block's returned hidden states, which for a
#                         transformer block is x + attn + mlp, i.e. AFTER the
#                         feed-forward's residual addition. An edit here cannot be
#                         rewritten by this block's own MLP, because the MLP has run.
_SCOPE_POINTS: Dict[str, InterventionPoint] = {"postmlp": InterventionPoint.BLOCK_OUTPUT}


def split_scope(scope: str) -> Tuple[str, Optional[InterventionPoint]]:
    """Separate a scope name from the intervention point its suffix names.

    Returns ``(base, point)`` where ``point`` is None when the scope does not override
    the configuration's own point, which is every scope that existed before the
    writer-phase diagnostic, so their behaviour is unchanged by construction.
    """
    for suffix, point in _SCOPE_POINTS.items():
        tail = "_" + suffix
        if str(scope).endswith(tail):
            return str(scope)[: -len(tail)], point
    return str(scope), None


def _depth_layer(ctx: QuestionContext, scope: str, last: int) -> int:
    """Resolve one depth scope against the frozen layer ranges, clamped to the model."""
    register_start = int(ctx.register_layers[0])
    if scope == "birth":
        # One before the writer range ends, which lands inside it on both checkpoints:
        # FLUX's writer is 17-19 so this is 18, PixArt's is 8-10 so this is 9.
        candidate = int(ctx.writer_layer) - 1
    elif scope == "boundary":
        candidate = register_start
    elif scope.startswith("writer+"):
        # Offsets from the writer range's FIRST layer, so the three name consecutive
        # writer blocks: FLUX 17/18/19, PixArt-Sigma 8/9/10.
        start = int(ctx.writer_range[0]) if ctx.writer_range else int(ctx.writer_layer)
        candidate = start + int(scope.split("+", 1)[1])
    else:
        candidate = register_start + _ESTABLISHED_OFFSET
    return int(min(max(candidate, 0), last))


# The four stages around one edited block. Only one of them is the edit; the other three
# say what the block did with it.
STAGE_ORDER = ("before_mlp", "after_mlp_write", "after_edit", "after_next_block")
STAGE_MEANING: Dict[str, str] = {
    "before_mlp": "post-attention residual, before the feed-forward (norm2's input)",
    "after_mlp_write": "the block's output, after the feed-forward's residual addition, "
                       "as the block produced it -- a BLOCK_OUTPUT probe registers "
                       "before the edit installer, so at a post-MLP edit this is the "
                       "value the edit is about to overwrite",
    "after_edit": "the tensor the edit returned, inside the hook",
    "after_next_block": "the output of the following block",
}
# Where the edit sits in the forward pass decides which stages precede it, and reading
# the table in the wrong order inverts its meaning: at BLOCK_INPUT the hook fires first
# and everything else is the block's RESPONSE to the edit; at BLOCK_OUTPUT the hook fires
# last and the two earlier stages are the untouched clean trajectory.
_FORWARD_ORDER: Dict[str, Dict[str, int]] = {
    InterventionPoint.BLOCK_INPUT.value: {
        "after_edit": 0, "before_mlp": 1, "after_mlp_write": 2, "after_next_block": 3},
    InterventionPoint.BLOCK_OUTPUT.value: {
        "before_mlp": 0, "after_mlp_write": 1, "after_edit": 2, "after_next_block": 3},
}


def _projection(states: Optional[torch.Tensor], ids: Sequence[int],
                unit: torch.Tensor) -> Tuple[float, float]:
    """Mean x.v* and mean signed cos(x, v*) over the treated tokens, or NaN."""
    if states is None or not len(ids):
        return float("nan"), float("nan")
    keep = [int(t) for t in ids if int(t) < int(states.shape[0])]
    if not keep:
        return float("nan"), float("nan")
    x = states[keep].float()
    alpha = x @ unit
    return float(alpha.mean()), float((alpha / x.norm(dim=-1).clamp_min(1e-9)).mean())


def stage_rows(trace: Trace, *, direction: torch.Tensor, token_ids: Sequence[int],
               layer: int, step: int, point: InterventionPoint,
               n_layers: Optional[int] = None) -> List[Dict[str, Any]]:
    """The treated tokens' alignment at four stages around one edited block.

    This exists because an edit installed at ``BLOCK_INPUT`` and a readout taken at the
    block *output* are separated by the whole block, feed-forward included, so a writer
    layer can rewrite the edited component and the readout returns the clean value while
    the edit was applied perfectly. The four stages separate the two:

    ``before_mlp``        the post-attention residual, from a ``PRE_MLP_RESIDUAL`` probe.
                          Absent where the block has no such stage (FLUX single blocks).
    ``after_mlp_write``   the block output as the block produced it. The probe registers
                          before the edit installer, so at a ``BLOCK_OUTPUT`` edit this
                          is the pre-edit value and the pair with ``after_edit`` is the
                          edit's effect at that site with nothing else moving.
    ``after_edit``        the installer's own ``x_post_hook`` record: the tensor the edit
                          returned, whichever site it was installed at.
    ``after_next_block``  the following block's output: how much of the edit survives
                          one more block of processing.
    """
    unit = direction.float() / direction.float().norm().clamp_min(1e-12)
    ids = [int(t) for t in token_ids]
    found: Dict[str, Optional[torch.Tensor]] = {
        "before_mlp": trace.probe(step, layer, InterventionPoint.PRE_MLP_RESIDUAL),
        "after_mlp_write": trace.probe(step, layer, InterventionPoint.BLOCK_OUTPUT),
    }
    edited = [value for key, value in trace.diagnostics.items()
              if key[0] == int(step) and key[1] == int(layer) and key[3] == "x_post_hook"]
    found["after_edit"] = edited[0] if edited else None
    following = trace.at(int(step), int(layer) + 1)
    rows: List[Dict[str, Any]] = []
    for name in STAGE_ORDER:
        if name == "after_next_block":
            if following is None or following.projection is None or following.norm is None:
                alpha = cosine = float("nan")
            else:
                keep = [t for t in ids if t < int(following.norm.numel())]
                alpha = (float(following.projection[keep].float().mean()) if keep
                         else float("nan"))
                cosine = (float((following.projection[keep].float()
                                 / following.norm[keep].float().clamp_min(1e-9)).mean())
                          if keep else float("nan"))
        else:
            alpha, cosine = _projection(found.get(name), ids, unit)
        order = _FORWARD_ORDER.get(point.value, {}).get(name)
        rows.append(dict(stage=name, stage_index=STAGE_ORDER.index(name),
                         # The order the forward pass actually reaches this stage under
                         # THIS edit point. Sort on it, not on stage_index, or the
                         # block's response to the edit reads as its cause.
                         forward_order=order if order is not None else -1,
                         is_after_the_edit=(None if order is None else
                                            bool(order >= _FORWARD_ORDER[point.value]
                                                 ["after_edit"])),
                         stage_meaning=STAGE_MEANING[name], layer=int(layer),
                         step=int(step), edit_point=point.value,
                         alpha_mean=alpha, cosine_mean=cosine,
                         measured=bool(alpha == alpha)))
    return rows


def _supported_layers(ctx: QuestionContext, point: InterventionPoint,
                      layers: Sequence[int]) -> List[int]:
    """Layers whose block actually advertises this point.

    FLUX's writer range straddles the dual/single boundary: 17 and 18 are MMDiT blocks
    and 19 is a fused single block with no post-attention stage before the feed-forward,
    so ``pre_mlp_residual`` exists at two of the three. Asking for it at the third would
    raise; a stage table with one column missing and a reason is the better answer.
    """
    refs = {int(r.index): r for r in ctx.driver.adapter.layers(ctx.driver.transformer)}
    keep = []
    for layer in layers:
        ref = refs.get(int(layer))
        if ref is None:
            continue
        try:
            if ctx.driver.adapter.intervention_capability(ref, point).supported:
                keep.append(int(layer))
        except Exception:
            continue
    return keep


def depth_layer_table(ctx: QuestionContext,
                      scopes: Sequence[str] = ()) -> pd.DataFrame:
    """What the depth scopes resolve to here, and whether each lands where it claims.

    Printed before a depth run because the offsets are only meaningful if they fall in the
    phase they are named after. A checkpoint whose register zone is shorter than the
    established offset would silently put two scopes on the same layer, and the run would
    compare a layer against itself.
    """
    last = int(ctx.driver.n_layers) - 1
    register = set(int(l) for l in ctx.register_layers)
    writer_span = (set(range(int(ctx.writer_range[0]), int(ctx.writer_range[1]) + 1))
                   if ctx.writer_range else {int(ctx.writer_layer)})
    # Default to the three life-stage scopes, which is what this table has always
    # listed. The writer-block offsets are a separate diagnostic and are listed only
    # when a caller names them, so the notebook's "two scopes on one layer" hard stop
    # keeps meaning what it meant.
    rows = []
    for scope in (scopes if scopes else _LIFE_STAGE_SCOPES):
        base, override = split_scope(scope)
        if base not in _DEPTH_SCOPES:
            continue
        layer = _depth_layer(ctx, base, last)
        phase = writer_span if base.startswith("writer") or base == "birth" else register
        point = InterventionPoint.BLOCK_INPUT if override is None else override
        rows.append(dict(scope=scope, layer=layer, point=point.value,
                         description=_DEPTH_SCOPES[base],
                         edit_lands=("after the feed-forward's residual addition"
                                     if point is InterventionPoint.BLOCK_OUTPUT
                                     else "at the block input, before attention and MLP"),
                         lands_in_named_phase=bool(layer in phase)))
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["distinct_layers"] = int(frame["layer"].nunique())
    frame["distinct_sites"] = int(frame[["layer", "point"]].drop_duplicates().shape[0])
    return frame


@dataclass
class ControlSurfaceResult:
    """Every table the experiment produced, plus where it was written."""

    root: Path
    token_metrics: pd.DataFrame
    population_metrics: pd.DataFrame
    attention_metrics: pd.DataFrame
    image_metrics: pd.DataFrame
    treatments: pd.DataFrame
    edit_diary: pd.DataFrame
    rescue_effects: pd.DataFrame = field(default_factory=pd.DataFrame)
    displacement: pd.DataFrame = field(default_factory=pd.DataFrame)
    depth_response: pd.DataFrame = field(default_factory=pd.DataFrame)
    # Four readings around one edited block, present only when `measure_stages` is on.
    # It is the table that tells an edit the block overwrote from an edit that held.
    stage_probe: pd.DataFrame = field(default_factory=pd.DataFrame)
    # One row per (rotation target, angle), present only when `run_rotation` is on.
    rotation: pd.DataFrame = field(default_factory=pd.DataFrame)
    # Each treated token's real (v*, u) coordinates before and after the rotation, which
    # is what the plane figure is drawn from: measured activations, not a schematic.
    rotation_plane: pd.DataFrame = field(default_factory=pd.DataFrame)
    verdict: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)
    images: Dict[str, Any] = field(default_factory=dict, repr=False)

    def save(self) -> Dict[str, Path]:
        directory = Path(self.root) / "metrics"
        directory.mkdir(parents=True, exist_ok=True)
        written: Dict[str, Path] = {}
        for name in ("token_metrics", "population_metrics", "attention_metrics",
                     "image_metrics", "treatments", "edit_diary", "rescue_effects",
                     "displacement", "depth_response", "stage_probe", "rotation",
                     "rotation_plane"):
            path = directory / f"{name}.csv"
            getattr(self, name).to_csv(path, index=False)
            written[name] = path
        payload = {"verdict": self.verdict, "meta": _jsonable(self.meta)}
        written["verdict"] = directory / "verdict.json"
        written["verdict"].write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return written


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def _add_floor_columns(images: pd.DataFrame) -> pd.DataFrame:
    """Express each image distance as a multiple of the (1, 1) cell's own distance.

    The centre cell runs the full edit, so its distance from the clean image is the
    model's sensitivity to a round trip through the operator rather than an effect. A
    raw distance is not interpretable without it: a cell at 1.2x the floor has done
    almost nothing, whatever its absolute value looks like. Computed per unit, because
    the floor is a property of the prompt and seed.
    """
    if images.empty or "rmse" not in images.columns:
        return images
    keys = [c for c in ("prompt_id", "seed", "layer_scope") if c in images.columns]
    centre = images[(images.get("arm") == "grid")
                    & np.isclose(images["beta"].astype(float), 1.0)
                    & np.isclose(images["gamma"].astype(float), 1.0)]
    if centre.empty or not keys:
        return images
    out = images.copy()
    # Mapped by key rather than merged: assigning a merge result back by position is only
    # safe while the merge preserves order and multiplicity, which is an assumption about
    # pandas rather than about the data. A lookup cannot silently misalign.
    unit_key = list(zip(*(out[k] for k in keys))) if len(keys) > 1 else list(out[keys[0]])
    for metric in ("rmse", "lpips"):
        if metric not in out.columns or not out[metric].notna().any():
            continue
        floor = centre.groupby(keys, observed=True)[metric].mean()
        denominator = pd.Series([floor.get(k, float("nan")) for k in unit_key],
                                index=out.index, dtype=float)
        out[f"{metric}_over_floor"] = (out[metric] / denominator).where(
            denominator.abs() > 1e-12)
    return out


# ===================================================================== runner
def _cell_id(scope: str, condition: Condition, prompt_id: int, seed: int) -> str:
    return f"{scope}|{condition.key}|p{int(prompt_id)}|s{int(seed)}"


def _analysis_code_digest() -> str:
    """Hash the source of the modules that produce the recorded numbers.

    A cached cell is only reusable if the code that wrote it would write the same thing
    today.  Adding a diagnostic to the edit hook, changing how a population row is
    computed, or fixing a rescue operator all leave the config identical and the numbers
    different, and the cache would serve the old rows, silently, with columns a new
    analysis cell expects to be there. That is how a run can report "diagnostics are
    missing" while the code that emits them is sitting in the checkout.

    Hashed by *source text*, not by git SHA: a commit that touches only a notebook or a
    doc must not throw away GPU hours, and a detached Colab clone has no useful SHA of its
    own. The three modules named here are the ones that decide what a cell records.
    """
    import hashlib

    digest = hashlib.sha256()
    for name in ("control_surface", "endpoints", "causal_ops"):
        path = Path(__file__).with_name(f"{name}.py")
        try:
            digest.update(path.read_bytes())
        except OSError:                                   # pragma: no cover
            digest.update(name.encode())
    return digest.hexdigest()[:16]


def experiment_fingerprint(ctx: QuestionContext, config: "ControlSurfaceConfig") -> str:
    """A hash of everything that would change a cell's numbers.

    The resume cache is keyed by scope, condition, prompt and seed, which is what
    identifies a *cell*, not what identifies the *experiment*. Widening the grid, changing
    the treatment rule, moving the edit point or pointing at a different channel all
    produce the same cell ids with different meanings, and a resumed run would mix them
    without a word. Every cell therefore records this fingerprint, and a cell whose
    fingerprint does not match is recomputed.

    Excludes the things that cannot change a number: output paths, whether
    images are saved, the difference-map amplification, and the resume flag itself.  It
    *includes* a digest of the analysis code, because a code change is the one way a cell
    can become stale while every setting still matches.
    """
    import hashlib

    material = {
        "betas": [float(b) for b in config.betas],
        "gammas": [float(g) for g in config.gammas],
        "layer_scopes": list(config.layer_scopes),
        "steps": config.resolved_steps(ctx),
        "point": config.point.value,
        "selection_mode": config.selection_mode,
        "topk": int(config.topk),
        "projection_percentile": float(config.projection_percentile),
        "dominant_channel": int(config.dominant_channel if config.dominant_channel
                                is not None else ctx.dominant_channel),
        "detail_quantile": float(config.detail_quantile),
        "clip_model": config.clip_model or "",
        "arms": [config.run_grid, config.run_rescue, config.run_controls,
                 config.run_cfg_diagnostic],
        "checkpoint": getattr(ctx.cfg.spec, "key", ""),
        "prompts": list(ctx.prompts),
        "seeds": [int(s) for s in ctx.seeds],
        "intervention_layer": int(ctx.intervention_layer),
        "register_layers": [int(l) for l in ctx.register_layers],
        "num_inference_steps": int(ctx.cfg.num_inference_steps),
        "guidance_scale": float(ctx.cfg.guidance_scale or 0.0),
        "height": int(ctx.cfg.height), "width": int(ctx.cfg.width),
        "dtype": str(ctx.cfg.dtype),
        "percentile": float(ctx.percentile),
        "highnorm_ratio": float(ctx.highnorm_ratio),
        "vstar_sha": hashlib.sha256(
            ctx.vstar.detach().float().cpu().numpy().tobytes()).hexdigest()[:16],
        "analysis_code": _analysis_code_digest(),
    }
    blob = json.dumps(material, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def _save_image(image, root: Path, condition: Condition, prompt_id: int, seed: int,
                scope: str) -> Optional[Path]:
    if image is None:
        return None
    directory = Path(root) / condition.directory
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{scope}_p{int(prompt_id)}_s{int(seed)}.png"
    try:
        image.save(path)
    except Exception:
        return None
    return path


def run_control_surface(ctx: QuestionContext, config: ControlSurfaceConfig
                        ) -> ControlSurfaceResult:
    """The 2D control surface, the rescue arm and the controls, one generation per cell.

    The order of operations per (prompt, seed) is the whole methodological contract:

    1. **One clean generation.**  It supplies the frozen treatment population, the clean
       ``alpha`` every rescue targets, and the paired image every distance is measured
       against.  It is re-run on resume rather than serialised, because a ``Trace`` holds
       the activations and the tables do not.
    2. **Freeze the treatment set** from that clean pass, and write it out.  No condition
       may re-derive it.
    3. **Every counterfactual** starts from the same prompt, seed, sampler, step count,
       guidance and model configuration, and differs only in the installed edit.

    Completed cells are cached under ``<root>/cells``, so an interrupted Colab session
    resumes without repeating a generation.
    """
    root = Path(config.experiment_root)
    (root / "cells").mkdir(parents=True, exist_ok=True)
    fingerprint = experiment_fingerprint(ctx, config)
    stale = 0
    channel = int(config.dominant_channel if config.dominant_channel is not None
                  else ctx.dominant_channel)
    steps = config.resolved_steps(ctx)
    sink_threshold = float(getattr(ctx.cfg, "sink_ratio_threshold", 10.0))
    scopes = [s for s in config.layer_scopes]
    cfg_batched = bool(getattr(ctx.cfg.spec, "cfg_batched", False))
    if config.run_cfg_diagnostic and not cfg_batched:
        ctx.say("  [Q13] skipping the CFG branch diagnostic: this pipeline runs the "
                "transformer at batch 1, so editing 'every row' is the same operation "
                "as editing the conditional row.")
    edit_layers = sorted({l for s in scopes for l in config.layers_for(ctx, s)})
    # A scope may name where in the block its edit lands, so the clean pass has to probe
    # each point that will be hooked. Reading a block output and writing it into a
    # block input would transplant a state across a residual update.
    layers_by_point: Dict[InterventionPoint, List[int]] = {}
    for scope in scopes:
        layers_by_point.setdefault(config.point_for(scope), []).extend(
            config.layers_for(ctx, scope))
    # The frozen treatment population is always selected at the intervention layer, at
    # the configuration's own point, whatever the scopes edit. A run whose scopes all
    # name some other layer or site would otherwise have no state to select from.
    layers_by_point.setdefault(config.point, []).extend(
        list(edit_layers) + [int(ctx.intervention_layer)])
    layers_by_point = {point: sorted(set(int(l) for l in layers))
                       for point, layers in layers_by_point.items()}
    # The stage probe needs the post-attention residual as well, where the block has one.
    stage_points = ([InterventionPoint.PRE_MLP_RESIDUAL, InterventionPoint.BLOCK_OUTPUT]
                    if config.measure_stages else [])
    observe = sorted(set(ctx.observe_layers) | set(edit_layers)
                     | {l + 1 for l in edit_layers if l + 1 < int(ctx.driver.n_layers)})
    lpips_net, clip = _load_lpips(), _load_clip(config.clip_model)
    # Cap the key-storage layers rather than trusting the flag: the cost is ~312 MB per
    # layer-step at FLUX's shapes, so applying it to a 20-layer window would need 6.6 GB.
    key_layers = (edit_layers[: max(int(config.store_keys_layers), 1)]
                  if config.store_keys else ())
    if config.store_keys and len(key_layers) < len(edit_layers):
        ctx.say(f"  [Q13] store_keys limited to layer(s) {list(key_layers)} of "
                f"{len(edit_layers)} -- the full attention tensors cost ~312 MB per "
                f"layer-step at 512px. Raise store_keys_layers deliberately.")

    (root / "config.json").write_text(
        json.dumps(_jsonable({**config.as_dict(ctx), "fingerprint": fingerprint}),
                   indent=2, sort_keys=True) + "\n")

    tokens: List[Dict[str, Any]] = []
    population: List[Dict[str, Any]] = []
    attention: List[Dict[str, Any]] = []
    stages: List[Dict[str, Any]] = []
    plane_points: List[Dict[str, Any]] = []
    images_meta: List[Dict[str, Any]] = []
    treatments: List[Dict[str, Any]] = []
    diary_rows: List[Dict[str, Any]] = []
    kept_images: Dict[str, Any] = {}
    identity_errors: List[float] = []

    def tracer() -> CausalTracer:
        return CausalTracer(
            ctx.driver.adapter, ctx.driver.transformer, direction=ctx.vstar, layers=observe,
            steps=steps, channels=ctx.channels, grid=ctx.driver.grid, cfg=ctx.cfg,
            key_layers=key_layers)

    for prompt_id, prompt, seed in ctx.units():
        ctx.say(f"  [Q13] prompt {prompt_id} seed {seed}: clean pass")
        probe_specs = [(point, layers) for point, layers in layers_by_point.items()]
        probe_specs += [(point, _supported_layers(ctx, point, edit_layers))
                        for point in stage_points
                        if _supported_layers(ctx, point, edit_layers)]
        clean_trace, _ = run_traced_generation(
            ctx.driver, tracer(), prompt_id=prompt_id, prompt=prompt, seed=seed,
            condition="clean", probes=probe_specs, save_image=True)
        targets = select_frozen_targets(
            clean_trace, layer=int(ctx.intervention_layer), step=int(steps[0]),
            point=config.point, direction=ctx.vstar, percentile=ctx.percentile,
            topk=ctx.topk, highnorm_ratio=ctx.highnorm_ratio)
        treatment = select_treatment(
            targets, mode=config.selection_mode, direction=ctx.vstar, topk=config.topk,
            projection_percentile=config.projection_percentile, sink_layers=edit_layers)
        treatments.append(dict(model=getattr(ctx.cfg.spec, "key", ""),
                               prompt_id=prompt_id, prompt=prompt, seed=seed,
                               **treatment.as_dict(),
                               norm_threshold=float(targets.norm_threshold),
                               alignment_threshold=float(targets.alignment_threshold)))
        ctx.say(f"    treatment: {len(treatment.token_ids)} tokens by {treatment.mode} "
                f"({treatment.n_selected_that_were_sinks} were clean sinks)")

        states_by_point: Dict[InterventionPoint, Dict[Tuple[int, int], torch.Tensor]] = {}
        subspaces_by_point: Dict[InterventionPoint, Dict[Tuple[int, int], Any]] = {}
        for point, layers in layers_by_point.items():
            found = {(int(st), int(l)): clean_trace.probe(int(st), int(l), point)
                     for st in steps for l in layers
                     if clean_trace.probe(int(st), int(l), point) is not None}
            states_by_point[point] = found
            # One SVD per (step, layer) per unit, reused by every condition: the subspace
            # is a property of the clean pass, so refitting it per condition would be both
            # wasteful and wrong: a treated run would be judged against its own
            # distortion.
            subspaces_by_point[point] = (
                {key: clean_subspace(state, rank=config.manifold_rank)
                 for key, state in found.items()} if config.measure_manifold else {})
        clean_states = states_by_point.get(config.point, {})
        # One (v*, u) plane per target per (step, layer), fitted on the CLEAN states so
        # every angle in a target turns in the same plane and the reference cell at
        # theta = 0 is the same state the other angles started from. `residual_pc1` in
        # particular must not be refitted per condition: a treated run would then be
        # rotated in a plane its own distortion chose.
        planes_by_target: Dict[str, Dict[InterventionPoint,
                                         Dict[Tuple[int, int], Dict[str, Any]]]] = {}
        if config.run_rotation:
            width = int(ctx.vstar.numel())
            # The ordinary population, by the project's own definition: every image token
            # that is neither a frozen register token nor in the high-norm top-k. This is
            # what `ordinary_mean` and `ordinary_pc1` are fitted on, and it is a different
            # population from the treated slice `residual_pc1` uses.
            excluded = set(int(t) for t in targets.register_ids) | \
                set(int(t) for t in targets.topk_ids)
            ordinary_states: Dict[InterventionPoint, Dict[Tuple[int, int], torch.Tensor]] = {}
            for point, found in states_by_point.items():
                per_key: Dict[Tuple[int, int], torch.Tensor] = {}
                for key, state in found.items():
                    keep = [t for t in range(int(state.shape[0])) if t not in excluded]
                    if len(keep) >= 2:
                        per_key[key] = state[keep]
                ordinary_states[point] = per_key
            if any(t in TARGETS_NEEDING_ORDINARY for t in config.rotation_targets):
                sample = next((v for v in ordinary_states.get(config.point, {}).values()),
                              None)
                ctx.say(f"  [Q13] ordinary population for the rotation targets: "
                        f"{0 if sample is None else int(sample.shape[0])} tokens "
                        f"({len(excluded)} excluded as register or high-norm top-k)")
            for target in config.rotation_targets:
                per_point: Dict[InterventionPoint, Dict[Tuple[int, int], Dict[str, Any]]] = {}
                for point, found in states_by_point.items():
                    built: Dict[Tuple[int, int], Dict[str, Any]] = {}
                    for key, state in found.items():
                        try:
                            built[key] = plane_basis(
                                ctx.vstar, target, width=width, seed=int(seed),
                                residuals=state,
                                ordinary=ordinary_states.get(point, {}).get(key),
                                semantic=config.semantic_direction)
                        except (ValueError, KeyError) as problem:
                            ctx.say(f"  [Q13] rotation target {target!r} unavailable at "
                                    f"step {key[0]} layer {key[1]}: {problem}")
                    per_point[point] = built
                planes_by_target[target] = per_point
            for target, per_point in planes_by_target.items():
                for point, built in per_point.items():
                    for key, plane in sorted(built.items()):
                        for problem in plane["notes"]:
                            ctx.say(f"  [Q13] {target} at step {key[0]} layer {key[1]}: "
                                    f"{problem}")
                        break          # one note per (target, point) is enough
        if clean_states:
            identity_errors.append(max(identity_error(state, ctx.vstar)
                                       for state in clean_states.values()))
        clean_image = clean_trace.image
        _save_image(clean_image, root, Condition("clean", "Clean", "clean"),
                    prompt_id, seed, "clean")
        kept_images.setdefault(f"clean_p{prompt_id}_s{seed}", clean_image)

        plan: List[Tuple[str, Condition]] = []
        for scope in scopes:
            if config.run_grid:
                plan += [(scope, c) for c in grid_conditions(config.betas, config.gammas)]
            if config.run_rescue:
                plan += [(scope, c) for c in RESCUE_CONDITIONS]
            if config.run_controls:
                plan += [(scope, c) for c in control_conditions(config.betas)]
            if config.run_cfg_diagnostic and cfg_batched:
                plan += [(scope, c) for c in cfg_conditions(config.betas, config.gammas)]
            if config.run_rotation:
                plan += [(scope, c) for c in rotation_conditions(
                    config.rotation_angles_deg, config.rotation_targets)]

        for scope, condition in plan:
            cell = _cell_id(scope, condition, prompt_id, seed)
            cache = root / "cells" / f"{cell.replace('|', '__')}.json"
            frames = {name: root / "cells" / f"{cell.replace('|', '__')}_{name}.csv"
                      for name in ("tokens", "population", "attention", "stages")}
            identifiers = dict(
                model=getattr(ctx.cfg.spec, "key", ""), prompt_id=prompt_id, prompt=prompt,
                seed=seed, layer_scope=scope, intervention=condition.key,
                condition_label=condition.label, arm=condition.arm, role=condition.role,
                beta=condition.beta, gamma=condition.gamma,
                # The rotation arm's coordinates, on every table, so its image distances
                # and downstream sink readouts join back to the angle that produced them.
                theta_deg=condition.theta_deg if condition.arm == "rotation" else None,
                rotation_target=condition.rotation_target or None,
                edit_direction=condition.edit_direction, token_group=condition.token_group,
                dominant_channel=channel, selection_mode=treatment.mode,
                n_treated_tokens=len(treatment.token_ids),
                edited_batch_rows="all" if condition.all_batch_rows else "conditional",
                cfg_batched=cfg_batched,
                guidance_scale=float(getattr(ctx.cfg, "guidance_scale", 0.0) or 0.0))

            if config.resume and cache.exists():
                payload = json.loads(cache.read_text())
                if payload.get("fingerprint") != fingerprint:
                    # Same cell id, different experiment. Recompute rather than mix.
                    stale += 1
                    for path in [cache, *frames.values()]:
                        try:
                            path.unlink()
                        except FileNotFoundError:
                            pass
                else:
                    images_meta.append(payload["image"])
                    for name, sink in (("tokens", tokens), ("population", population),
                                       ("attention", attention), ("stages", stages)):
                        if frames[name].exists():
                            sink.extend(pd.read_csv(frames[name]).to_dict("records"))
                    diary_rows.extend(payload.get("diary", []))
                    plane_points.extend(payload.get("plane", []))
                    continue

            layers = config.layers_for(ctx, scope)
            group = {"treatment": treatment.token_ids,
                     "random_tokens": targets.random_ids,
                     "sink_only": treatment.clean_sink_ids}[condition.token_group]
            diary: List[Dict[str, Any]] = []
            turn_rows: List[Dict[str, Any]] = []
            plans: List[EditPlan] = []
            scope_point = config.point_for(scope)
            if condition.key != "clean":
                plans = [EditPlan(
                    edit=make_edit(condition, direction=ctx.vstar, token_ids=group,
                                   dominant_channel=channel,
                                   clean_states=states_by_point.get(scope_point, {}),
                                   seed=int(seed), diary=diary,
                                   subspaces=subspaces_by_point.get(scope_point, {}),
                                   planes=planes_by_target.get(
                                       condition.rotation_target, {}).get(scope_point, {}),
                                   plane_rows=turn_rows),
                    point=scope_point, layers=layers, steps=steps,
                    all_batch_rows=condition.all_batch_rows,
                    label=f"{scope}:{condition.key}")]
            ctx.say(f"    [{scope}] {condition.label}")
            trace, stats = run_traced_generation(
                ctx.driver, tracer(), prompt_id=prompt_id, prompt=prompt, seed=seed,
                condition=condition.key, plans=plans, targets=targets,
                save_image=config.save_images,
                probes=probe_specs if config.measure_stages else ())

            path = _save_image(trace.image, root, condition, prompt_id, seed, scope)
            row = dict(identifiers, **image_metrics(
                clean_image, trace.image, prompt=prompt, lpips_net=lpips_net, clip=clip,
                detail_quantile=config.detail_quantile), **stats.as_dict(),
                image_path=None if path is None else str(path.relative_to(root)))
            if config.save_difference_maps and trace.image is not None:
                array = ID.amplified_difference(clean_image, trace.image, config.amplification)
                if array is not None:
                    try:
                        from PIL import Image

                        directory = root / condition.directory
                        directory.mkdir(parents=True, exist_ok=True)
                        name = f"{scope}_p{prompt_id}_s{seed}_diff_x{config.amplification:g}.png"
                        Image.fromarray(array).save(directory / name)
                        row["difference_map"] = str((directory / name).relative_to(root))
                        row["difference_amplification"] = float(config.amplification)
                    except Exception:
                        pass
            images_meta.append(row)

            cell_tokens = [dict(identifiers, **r) for r in token_rows(
                trace, clean_trace, treatment, condition=condition, layers=observe,
                steps=steps, direction=ctx.vstar, dominant_channel=channel,
                sink_threshold=sink_threshold, edit_layers=layers, all_tokens=False)]
            cell_population = [dict(identifiers, **r) for r in population_rows(
                trace, clean_trace, treatment, condition=condition, layers=observe,
                steps=steps, norm_threshold=float(targets.norm_threshold),
                sink_threshold=sink_threshold,
                alignment_threshold=float(targets.alignment_threshold), edit_layers=layers,
                register_layers=ctx.register_layers, treatment_targets=targets,
                recovery_floor=config.recovery_floor)]
            cell_attention: List[Dict[str, Any]] = []
            for step in steps:
                for layer in observe:
                    for measurement in EP.measure_layer(
                            clean_trace.at(int(step), int(layer)),
                            trace.at(int(step), int(layer)), targets, layer=int(layer),
                            grid=ctx.driver.grid, token_ids=treatment.token_ids,
                            channels=ctx.channels):
                        cell_attention.append(dict(identifiers, step=int(step),
                                                   **asdict(measurement)))
            cell_stages: List[Dict[str, Any]] = []
            if config.measure_stages:
                for step in steps:
                    for layer in layers:
                        cell_stages += [dict(identifiers, **r) for r in stage_rows(
                            trace, direction=ctx.vstar, token_ids=treatment.token_ids,
                            layer=int(layer), step=int(step), point=scope_point,
                            n_layers=int(ctx.driver.n_layers))]
            for name, batch, sink in (("tokens", cell_tokens, tokens),
                                      ("population", cell_population, population),
                                      ("attention", cell_attention, attention),
                                      ("stages", cell_stages, stages)):
                if batch:
                    pd.DataFrame(batch).to_csv(frames[name], index=False)
                sink.extend(batch)
            diary_rows.extend(dict(identifiers, **note) for note in diary)
            plane_points.extend(dict(identifiers, **row) for row in turn_rows)
            cache.write_text(json.dumps(_jsonable(
                {"fingerprint": fingerprint, "image": row,
                 "diary": [dict(identifiers, **n) for n in diary],
                 "plane": [dict(identifiers, **r) for r in turn_rows]}),
                indent=2, sort_keys=True) + "\n")
            if condition.arm != "grid" or (condition.beta, condition.gamma) in (
                    (1.0, 1.0), (min(config.betas), 1.0), (max(config.betas), 1.0),
                    (1.0, min(config.gammas)), (1.0, max(config.gammas))):
                kept_images[f"{scope}_{condition.key}_p{prompt_id}_s{seed}"] = trace.image

        # The clean population scatters are properties of the clean model, so they are
        # written once per unit over every token rather than once per condition.
        clean_identifiers = dict(
            model=getattr(ctx.cfg.spec, "key", ""), prompt_id=prompt_id, prompt=prompt,
            seed=seed, layer_scope="clean", condition_label="Clean (all tokens)",
            dominant_channel=channel, selection_mode=treatment.mode,
            n_treated_tokens=len(treatment.token_ids))
        tokens.extend(dict(clean_identifiers, **r)
            for r in token_rows(clean_trace, clean_trace, treatment,
                                condition=Condition("clean_all_tokens", "Clean (all tokens)",
                                                    "clean", role="reference"),
                                layers=observe, steps=steps, direction=ctx.vstar,
                                dominant_channel=channel, sink_threshold=sink_threshold,
                                edit_layers=edit_layers, all_tokens=True))

    frames = {name: pd.DataFrame(rows) for name, rows in
              (("token_metrics", tokens), ("population_metrics", population),
               ("attention_metrics", attention), ("image_metrics", images_meta),
               ("treatments", treatments), ("edit_diary", diary_rows))}
    stage_probe = pd.DataFrame(stages)
    rotation_plane = pd.DataFrame(plane_points)
    frames["image_metrics"] = _add_floor_columns(frames["image_metrics"])
    if stale:
        ctx.say(f"  [Q13] discarded {stale} cached cell(s) computed under a different "
                f"configuration or an older version of the analysis code; they were "
                f"recomputed under fingerprint {fingerprint}.")
    meta = dict(
        fingerprint=fingerprint, stale_cells_discarded=stale,
        checkpoint=getattr(ctx.cfg.spec, "key", ""), dominant_channel=channel,
        edit_layers=edit_layers, observed_layers=observe, steps=steps,
        layer_scopes=scopes, selection_mode=config.selection_mode,
        sink_threshold=sink_threshold,
        identity_error_max=max(identity_errors) if identity_errors else float("nan"),
        lpips_available=lpips_net is not None, clip_available=clip is not None,
        vstar_source=dict(ctx.vstar_metadata),
        measurement_note=("(beta, gamma) = (1, 1) runs the edit rather than skipping it, so "
                          "its image distance is the numerical floor every other cell is "
                          "read against"))
    effects = rescue_effects(frames["population_metrics"], frames["image_metrics"])
    moved = displacement_report(frames["edit_diary"])
    if not moved.empty:
        away = moved[~moved["is_matched_null"].fillna(True)]
        for _, row in away.iterrows():
            ctx.say(f"  [Q13] {row['intervention']} ends "
                    f"{float(row['distance_ratio']):.2f}x further from the clean state "
                    "than the ablation does; it is a larger perturbation, not a matched "
                    "null, and its downstream behaviour must be read that way.")
    clusters = int(effects["n_clusters"].max()) if not effects.empty else 0
    meta["n_prompt_clusters"] = clusters
    meta["clustered_intervals_available"] = bool(
        not effects.empty and effects["has_interval"].any())
    if clusters and clusters < 3:
        ctx.say(f"  [Q13] {clusters} prompt cluster(s): below the three this project's "
                "percentile bootstrap requires, so no clustered interval is available. "
                "Raise the prompt count before quoting an effect with error bars.")
    meta["displacement_controls_exceeding_the_damage"] = (
        [] if moved.empty else
        sorted(set(moved.loc[~moved["is_matched_null"].fillna(True), "intervention"])))
    depth = depth_response(frames["population_metrics"], frames["image_metrics"],
                           frames["edit_diary"])
    turned = rotation_response(frames["edit_diary"], frames["population_metrics"],
                               frames["image_metrics"])
    if not turned.empty:
        worst = float(turned["norm_relative_error_max"].max())
        ctx.say(f"  [Q13] rotation arm: largest relative norm change across every angle "
                f"and target is {worst:.2e} (tolerance {_ROTATION_NORM_TOLERANCE:.0e}); "
                "the norm was held by the operator, not by a correction.")
        meta["rotation_norm_relative_error_max"] = worst
        meta["rotation_targets_run"] = sorted(set(turned["rotation_target"]))
    meta["depth_scopes"] = [s for s in scopes if split_scope(s)[0] in _DEPTH_SCOPES]
    meta["scope_points"] = {s: config.point_for(s).value for s in scopes}
    result = ControlSurfaceResult(root=root, verdict=verdict(frames, meta), meta=meta,
                                  images=kept_images, rescue_effects=effects,
                                  displacement=moved, depth_response=depth,
                                  stage_probe=stage_probe, rotation=turned,
                                  rotation_plane=rotation_plane, **frames)
    result.save()
    return result


def gap_closed(means: pd.Series, *, damaged: str = "channel_ablate",
               clean: str = "clean") -> pd.Series:
    r"""Fraction of the damage each rescue undid, rather than its absolute level.

    .. math:: \text{closed} = \frac{m_{\text{rescue}} - m_{\text{damaged}}}
                                     {m_{\text{clean}} - m_{\text{damaged}}}

    The mediation question is "how much of what the ablation destroyed came back", and
    the absolute recovery level cannot answer it: where the ablation happens to do little
    damage, every condition sits near the clean value and a threshold on the absolute
    number rejects a perfectly good rescue.  Normalising by the damage makes the
    comparison independent of how hard the ablation bit, which differs by checkpoint,
    by layer and by how much of ``v*`` the dominant channel actually carries.

    1.0 means fully rescued, 0.0 means the rescue did nothing, and the denominator is
    reported as ``nan`` when the ablation did no damage, which is a fact about the
    ablation, not a failure of the rescue.
    """
    if damaged not in means.index or clean not in means.index:
        return pd.Series(dtype=float)
    span = float(means[clean]) - float(means[damaged])
    if abs(span) < 1e-9:
        return pd.Series(float("nan"), index=means.index)
    return (means - float(means[damaged])) / span


def rescue_effects(population: pd.DataFrame, images: Optional[pd.DataFrame] = None, *,
                   reference: str = "channel_ablate",
                   iterations: int = 2000) -> pd.DataFrame:
    """Each rescue against the ablation, with the project's prompt-clustered interval.

    Uses :func:`ditsinks.causal_stats.paired_effect` rather than a standard error across
    units, for two reasons.  It is paired within unit, so prompt-to-prompt variation,
    which is large and uninteresting, cancels instead of inflating the spread.  And it
    clusters by prompt, so two seeds of one prompt are not counted as two independent
    observations.

    The reference is the **ablation**, not the clean run: the mediation question is how
    much of the damage each condition undid, and an interval on the difference from clean
    would answer a different one.

    ``bootstrap_mean`` declines to produce an interval below three clusters, and that
    refusal is passed through here rather than papered over: an underpowered run must
    not be able to buy a confidence interval.
    """
    from . import causal_stats as CST

    rows: List[Dict[str, Any]] = []
    sources = [("population", population)]
    if images is not None and not images.empty:
        sources.append(("image", images))
    for source_name, frame in sources:
        if frame is None or frame.empty or "condition" not in frame.columns:
            continue
        arm = frame[frame["arm"] == "rescue"] if "arm" in frame.columns else frame
        if "is_downstream" in arm.columns:
            keep = arm[arm["is_downstream"].fillna(True).astype(bool)]
            arm = keep if not keep.empty else arm
        if "is_register_zone" in arm.columns and arm["is_register_zone"].any():
            # The register zone is where the state the rescue targets actually exists.
            # Beyond it the projection collapses by design, so a ratio to the clean mean
            # divides by a quantity on its way to zero and a mean over depth is dominated
            # by whichever layer divided by the smallest number.
            zone = arm[arm["is_register_zone"].fillna(False).astype(bool)]
            arm = zone if not zone.empty else arm
        if arm.empty or reference not in set(arm["condition"]):
            continue
        # `selected_alpha_change_in_spreads` leads because it is defined at every layer;
        # `selected_alpha_recovery` follows and is NaN wherever its denominator was not
        # resolvable, so the paired bootstrap simply drops those layers instead of being
        # dominated by them.
        metrics = [c for c in ("selected_alpha_change_in_spreads", "selected_alpha_mean",
                               "selected_alpha_recovery", "n_highnorm_and_aligned",
                               "highnorm_retention", "sink_retention",
                               "selected_sink_strength_mean", "lpips", "rmse",
                               "concentration_ratio", "spectrum_high_over_uniform")
                   if c in arm.columns and arm[c].notna().any()]
        for condition in sorted(set(arm["condition"]) - {reference}):
            for metric in metrics:
                estimate = CST.paired_effect(arm, metric, treatment=condition,
                                             reference=reference, iterations=iterations)
                rows.append(dict(source=source_name, condition=condition,
                                 reference=reference, metric=metric,
                                 effect=estimate.value, ci_low=estimate.ci_low,
                                 ci_high=estimate.ci_high, n_units=estimate.n_units,
                                 n_clusters=estimate.n_clusters,
                                 has_interval=estimate.has_interval,
                                 excludes_zero=estimate.excludes_zero))
    return pd.DataFrame(rows)


def displacement_report(diary: pd.DataFrame) -> pd.DataFrame:
    """Did each condition move the state toward the clean trajectory, or away from it?

    Read this before any rescue comparison.  Matching the *injection* L2 between a rescue
    and its orthogonal control matches the effort spent, which is the right way to control
    for effort, but it does not match the resulting distance from the clean state.  A
    rescue spends its budget moving back; an orthogonal step of the same length spends it
    moving sideways and lands at ``sqrt(2)`` times the gap, further away than the damage
    it controls for.

    A condition with ``distance_ratio`` above 1 is therefore not a matched null. Whatever
    it shows downstream is the response to a larger perturbation, and cannot be read as
    evidence that the direction did not matter.
    """
    if diary is None or diary.empty:
        return pd.DataFrame()
    columns = {"rescue_closed_distance_ratio", "distance_from_clean_after_ablation_l2",
               "distance_from_clean_after_rescue_l2"}
    if not columns.issubset(set(diary.columns)):
        return pd.DataFrame()
    rows = diary[diary["rescue_closed_distance_ratio"].notna()]
    if rows.empty:
        return pd.DataFrame()
    if "is_conditional_row" in rows.columns:
        # One note per (step, layer); an `all_batch_rows` condition writes one per row.
        kept = rows[rows["is_conditional_row"].fillna(True).astype(bool)]
        rows = kept if not kept.empty else rows
    keys = [k for k in ("layer_scope", "arm", "intervention") if k in rows.columns]
    grouped = rows.groupby(keys, observed=True).agg(
        gap_after_ablation=("distance_from_clean_after_ablation_l2", "mean"),
        gap_after_rescue=("distance_from_clean_after_rescue_l2", "mean"),
        distance_ratio=("rescue_closed_distance_ratio", "mean"),
        injection_l2=("rescue_injection_l2", "mean")
        if "rescue_injection_l2" in rows.columns else
        ("distance_from_clean_after_rescue_l2", "size"),
    ).reset_index()
    grouped["moved_toward_clean"] = grouped["distance_ratio"] < 1.0
    grouped["is_matched_null"] = grouped["distance_ratio"] <= 1.0 + 1e-6
    return grouped


def depth_response(population: pd.DataFrame, images: Optional[pd.DataFrame] = None,
                   diary: Optional[pd.DataFrame] = None, *,
                   gamma: float = 1.0) -> pd.DataFrame:
    """The beta response at each intervention depth, on one row per (scope, beta).

    Built for the depth comparison and answering its three questions in one table:

    ``clean_cosine_headroom``   how much room beta had at that depth before the edit.
                                A depth where the clean state is already at |cos| ~ 1 has
                                none, so a flat response there is a ceiling and not a
                                null result.
    ``realised_cosine``         what the edit actually achieved. Compared against the
                                headroom, this separates "the edit did nothing" from
                                "there was nothing left to do".
    ``onmanifold_ratio_to_clean``  whether the achieved state is one the clean population
                                could have contained. This is what distinguishes a
                                meaningful opposite-``v*`` state at negative beta from
                                off-manifold corruption of the same magnitude.
    """
    if population is None or population.empty:
        return pd.DataFrame()
    grid = population[population["arm"] == "grid"] if "arm" in population.columns else population
    if "is_edited_layer" in grid.columns:
        at_edit = grid[grid["is_edited_layer"].fillna(False).astype(bool)]
        grid = at_edit if not at_edit.empty else grid
    grid = grid[np.isclose(grid["gamma"].astype(float), float(gamma))]
    if grid.empty:
        return pd.DataFrame()
    keys = ["layer_scope", "beta"]
    aggregates = {"layer": ("layer", "first")}
    for column, name in (("selected_cosine_mean", "realised_cosine"),
                         ("clean_cosine_signed", "clean_cosine"),
                         ("clean_cosine_abs", "clean_cosine_abs"),
                         ("clean_cosine_sign_agreement", "clean_cosine_sign_agreement"),
                         ("clean_cosine_headroom", "clean_cosine_headroom"),
                         ("selected_alpha_mean", "selected_alpha_mean"),
                         ("n_highnorm_and_aligned", "n_highnorm_and_aligned"),
                         ("selected_sink_strength_mean", "sink_strength")):
        if column in grid.columns:
            aggregates[name] = (column, "mean")
    table = grid.groupby(keys, observed=True).agg(**aggregates).reset_index()

    for source, columns in ((images, ("lpips", "rmse", "concentration_ratio",
                                      "spectrum_high_over_uniform")),
                            (diary, ("onmanifold_energy", "onmanifold_ratio_to_clean",
                                     "onmanifold_clean_median",
                                     "onmanifold_variance_explained", "perturbation_l2"))):
        if source is None or source.empty:
            continue
        rows = source
        if "arm" in rows.columns:
            rows = rows[rows["arm"] == "grid"]
        if "is_conditional_row" in rows.columns:
            kept = rows[rows["is_conditional_row"].fillna(True).astype(bool)]
            rows = kept if not kept.empty else rows
        if rows.empty or not {"layer_scope", "beta", "gamma"} <= set(rows.columns):
            continue
        rows = rows[np.isclose(rows["gamma"].astype(float), float(gamma))]
        present = [c for c in columns if c in rows.columns and rows[c].notna().any()]
        if rows.empty or not present:
            continue
        table = table.merge(rows.groupby(keys, observed=True)[present].mean().reset_index(),
                            on=keys, how="left")

    if {"realised_cosine", "clean_cosine"} <= set(table.columns):
        # Both signed, so this is zero at beta = 1 by construction: the check
        # that the two quantities are commensurable at all.
        table["cosine_gain"] = table["realised_cosine"] - table["clean_cosine"]
        if "clean_cosine_abs" in table.columns:
            # The gain as a share of the room that was available. Near 1 means the edit
            # used up the headroom; near 0 at a depth that HAD headroom means the edit
            # genuinely failed rather than hit a ceiling.
            room = (1.0 - table["clean_cosine_abs"]).clip(lower=1e-9)
            table["gain_over_headroom"] = (table["realised_cosine"].abs()
                                           - table["clean_cosine_abs"]) / room
    return table.sort_values(["layer_scope", "beta"]).reset_index(drop=True)


def rotation_response(diary: pd.DataFrame, population: Optional[pd.DataFrame] = None,
                      images: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """One row per (target, angle): what the rotation did, and what followed from it.

    The hook columns come from the diary, because they are the only place the state is
    read at the moment the edit produced it. The sink and on-manifold columns come from
    the same diary rows; the image distances and the detail metrics are joined from the
    image table, and the downstream sink readouts from the population table restricted
    to the register zone, where the state the rotation moved actually exists.

    ``norm_relative_error_max`` is carried through to the front of the table on purpose.
    Every claim in this arm rests on the norm having been held fixed, and a reader should
    not have to go looking for the number that says whether it was.
    """
    if diary is None or diary.empty or "theta_deg" not in diary.columns:
        return pd.DataFrame()
    rows = diary[diary["arm"] == "rotation"] if "arm" in diary.columns else diary
    if "is_conditional_row" in rows.columns:
        kept = rows[rows["is_conditional_row"].fillna(True).astype(bool)]
        rows = kept if not kept.empty else rows
    rows = rows.dropna(subset=["theta_deg"])
    if rows.empty:
        return pd.DataFrame()
    keys = ["rotation_target", "theta_deg"]
    aggregates: Dict[str, Tuple[str, str]] = {
        "layer": ("layer", "first"),
        "norm_relative_error_max": ("rotation_norm_relative_error", "max"),
        "u_dot_v_absmax": ("rotation_u_dot_v", lambda v: float(v.abs().max())),
    }
    for column in ("norm_before", "norm_after", "alpha_vstar_before", "alpha_vstar_after",
                   "projection_u_before", "projection_u_after", "cosine_vstar_before",
                   "cosine_vstar_after", "cosine_u_before", "cosine_u_after",
                   "angle_from_vstar_before_deg", "angle_from_vstar_after_deg",
                   "angle_change_deg", "in_plane_energy_share", "alpha2_share",
                   "b2_share", "cos_squared_mean", "out_of_plane_share",
                   "in_plane_share_at_least_cos_squared", "perturbation_l2",
                   "onmanifold_energy", "onmanifold_ratio_to_clean",
                   "onmanifold_is_discriminative",
                   "rotation_target_cosine_with_vstar",
                   "rotation_independent_target_content"):
        if column in rows.columns:
            aggregates[column] = (column, "mean")
    table = rows.groupby(keys, observed=True).agg(**aggregates).reset_index()
    # The norm claim, as one boolean a reader can sort on.
    table["norm_held_fixed"] = table["norm_relative_error_max"] <= _ROTATION_NORM_TOLERANCE

    for source, columns in (
            (images, ("lpips", "rmse", "concentration_ratio", "sobel_change",
                      "spectrum_high_over_uniform", "spectrum_low_over_uniform")),
            (population, ("selected_sink_strength_mean", "n_selected_are_sinks",
                          "n_sinks", "mean_sink_strength", "n_highnorm_and_aligned"))):
        if source is None or source.empty:
            continue
        joined = source
        if "arm" in joined.columns:
            joined = joined[joined["arm"] == "rotation"]
        if "theta_deg" not in joined.columns or "rotation_target" not in joined.columns:
            continue
        if source is population and "is_register_zone" in joined.columns:
            # The rotation moved a register state, so its downstream sink behaviour is
            # only a fact about that state where the state exists.
            zone = joined[joined["is_register_zone"].fillna(True).astype(bool)]
            joined = zone if not zone.empty else joined
        present = [c for c in columns if c in joined.columns and joined[c].notna().any()]
        if joined.empty or not present:
            continue
        table = table.merge(
            joined.groupby(keys, observed=True)[present].mean().reset_index(),
            on=keys, how="left")

    if {"cosine_vstar_after", "selected_sink_strength_mean"} <= set(table.columns):
        # The arm's actual question: does moving the state off v* cost sinkhood, and does
        # it cost image quality? Reported as a pair, per target, so a rotation that buys
        # one with the other is visible rather than averaged away.
        reference = table[np.isclose(table["theta_deg"], 0.0)].set_index("rotation_target")
        for name, column in (("sink_share_of_unrotated", "selected_sink_strength_mean"),
                             ("lpips_over_unrotated", "lpips")):
            if column not in table.columns:
                continue
            base = table["rotation_target"].map(reference[column]) \
                if column in reference.columns else np.nan
            table[name] = table[column] / pd.Series(base, index=table.index).replace(
                0.0, np.nan)
    return table.sort_values(["rotation_target", "theta_deg"]).reset_index(drop=True)


# ==================================================================== verdict
def _axis_span(frame: pd.DataFrame, column: str, *, axis: str) -> float:
    """Range of ``column`` along one grid axis with the other held at its neutral value.

    This is the direct comparison the 2D grid makes: how much the outcome moves
    when alignment varies at fixed magnitude, against how much it moves when magnitude
    varies at fixed alignment.  Both are read off the same surface, in the same units.
    """
    if frame.empty or column not in frame.columns:
        return float("nan")
    held = "gamma" if axis == "beta" else "beta"
    line = frame[np.isclose(frame[held].astype(float), 1.0)]
    if line.empty:
        return float("nan")
    values = line.groupby(axis, observed=True)[column].mean().dropna()
    return float(values.max() - values.min()) if len(values) > 1 else float("nan")


def verdict(frames: Dict[str, pd.DataFrame], meta: Dict[str, Any]) -> str:
    """Report the contrasts, in numbers, without choosing between the hypotheses.

    Every sentence here is a measurement with its value attached.  Which of H1--H5 the
    numbers favour is a judgement for the writeup, and coding that judgement into the
    runner would make the experiment unable to come out the other way.
    """
    lines: List[str] = []
    images, population, token = (frames.get("image_metrics", pd.DataFrame()),
                                 frames.get("population_metrics", pd.DataFrame()),
                                 frames.get("token_metrics", pd.DataFrame()))
    floor = float("nan")
    if not images.empty and "rmse" in images.columns:
        centre = images[(images["arm"] == "grid") & np.isclose(images["beta"], 1.0)
                        & np.isclose(images["gamma"], 1.0)]
        if not centre.empty:
            floor = float(centre["rmse"].mean())
        if np.isfinite(floor):
            lines.append(
                f"Numerical floor: the (beta=1, gamma=1) cell runs the full edit and lands "
                f"{floor:.4g} RMSE from its own clean image. Every other cell must be read "
                f"against that, not against zero.")
        else:
            lines.append("No image distances are available on this run -- a checkpoint with "
                         "no decoder measures the internal state only -- so the numerical "
                         "floor of the (1, 1) cell could not be established from images.")
    identity = float(meta.get("identity_error_max", float("nan")))
    if np.isfinite(identity):
        consequence = ("so the floor above is the model's sensitivity rather than an error in "
                       "the operator" if np.isfinite(floor) else
                       "so the operator itself is a faithful round trip, whatever the images "
                       "would have shown")
        lines.append(f"The (1, 1) reconstruction is exact to {identity:.3g} in float32 at the "
                     f"edit point, {consequence}.")

    def downstream(frame: pd.DataFrame) -> pd.DataFrame:
        """Rows the edit could possibly have reached.

        Averaging a recovery fraction over layers *before* the intervention pulls every
        condition toward 1, because nothing upstream can have moved. The untouched layers
        stay in the saved tables (they are the placebo check below) but no summary may
        average over them.
        """
        if frame.empty or "is_downstream" not in frame.columns:
            return frame
        keep = frame[frame["is_downstream"].fillna(True).astype(bool)]
        return keep if not keep.empty else frame

    upstream = pd.DataFrame()
    if not population.empty and "is_downstream" in population.columns:
        upstream = population[~population["is_downstream"].fillna(True).astype(bool)]
    if not upstream.empty and "selected_alpha_mean" in upstream.columns:
        # Layers before the edit must be bit-identical across conditions. A nonzero
        # spread here is a leak, not a result, so it is reported as a gate rather than
        # left for a reader to notice.
        spread = upstream.groupby(["step", "layer"], observed=True)["selected_alpha_mean"] \
            .agg(lambda values: float(values.max() - values.min()))
        plural = "layer-step" if len(spread) == 1 else "layer-steps"
        lines.append(f"Placebo gate: across the {len(spread)} {plural} upstream of the edit, "
                     f"the treated tokens' projection varies by at most "
                     f"{float(spread.max()):.3g} between conditions -- it must be ~0, since "
                     "no condition can reach them.")

    grid = downstream(images[images["arm"] == "grid"]) if not images.empty else pd.DataFrame()
    for column, name in (("rmse", "image RMSE"), ("concentration_ratio",
                                                  "difference energy on clean structure")):
        beta_span, gamma_span = (_axis_span(grid, column, axis="beta"),
                                 _axis_span(grid, column, axis="gamma"))
        if np.isfinite(beta_span) and np.isfinite(gamma_span):
            lines.append(
                f"Direction against magnitude on {name}: varying alignment at fixed norm "
                f"spans {beta_span:.4g}; varying norm at fixed direction spans "
                f"{gamma_span:.4g}.")

    grid_population = (downstream(population[population["arm"] == "grid"])
                       if not population.empty else pd.DataFrame())
    for column, name in (("selected_alpha_mean", "the treated tokens' v* projection"),
                         ("n_highnorm", "the high-norm population size"),
                         ("selected_sink_strength_mean", "treated-token sink strength")):
        beta_span, gamma_span = (_axis_span(grid_population, column, axis="beta"),
                                 _axis_span(grid_population, column, axis="gamma"))
        if np.isfinite(beta_span) and np.isfinite(gamma_span):
            lines.append(f"On {name}: alignment axis spans {beta_span:.4g}, magnitude axis "
                         f"spans {gamma_span:.4g}.")

    rescue = (downstream(population[population["arm"] == "rescue"]) if not population.empty
              else pd.DataFrame())
    if not rescue.empty:
        for column, name in (("selected_alpha_recovery", "v* projection recovered"),
                             ("highnorm_retention", "clean high-norm tokens still high-norm"),
                             ("sink_retention", "affected heads keeping their clean sink")):
            if column not in rescue.columns:
                continue
            by = rescue.groupby("condition", observed=True)[column].mean().dropna()
            if by.empty:
                continue
            ordered = ", ".join(f"{key} {float(value):.3g}" for key, value in by.items())
            lines.append(f"Rescue, {name}: {ordered}.")
            # The absolute level cannot answer the mediation question where the ablation
            # did little damage, so the share of the damage each condition undid is
            # reported beside it.
            closed = gap_closed(by)
            if not closed.empty and closed.notna().any():
                share = ", ".join(f"{key} {float(value):.0%}" for key, value in closed.items()
                                  if key not in ("clean", "channel_ablate")
                                  and np.isfinite(value))
                if share:
                    lines.append(f"  -- share of that damage undone: {share}.")
            elif not closed.empty:
                lines.append("  -- the ablation did no measurable damage to this endpoint, "
                             "so there was nothing for a rescue to undo.")

    if not token.empty and {"is_sink", "alpha_rank", "is_selected"} <= set(token.columns):
        clean = token[token["intervention"] == "clean_all_tokens"]
        if not clean.empty:
            cut = max(1, int(0.01 * clean["alpha_rank"].max()))
            top, rest = clean[clean["alpha_rank"] <= cut], clean[clean["alpha_rank"] > cut]
            if not clean["is_sink"].any():
                # Reporting "0% against 0%" invites reading a separation into a run where
                # the criterion simply never fired, which is a different fact and has to
                # be said as one.
                lines.append(
                    f"No token on the clean run meets the project's sink criterion "
                    f"({meta.get('sink_threshold')}x the uniform share), so sinkhood cannot be "
                    "compared along the projection on this run.")
            elif not top.empty and not rest.empty:
                lines.append(
                    f"Sinkhood along the projection: {float(top['is_sink'].mean()):.0%} of the "
                    f"top 1% by clean v* projection meet the project's sink criterion, against "
                    f"{float(rest['is_sink'].mean()):.0%} of the rest.")

    displaced = meta.get("displacement_controls_exceeding_the_damage") or []
    if displaced:
        lines.append(
            f"Displacement warning: {', '.join(displaced)} "
            f"{'ends' if len(displaced) == 1 else 'end'} further from the clean "
            "state than the ablation does. Matching injection L2 matches effort, not the "
            "resulting distance from clean -- an orthogonal step of the gap's own length "
            "lands at sqrt(2) times the gap. Any downstream recovery such a condition "
            "shows is the response to a larger perturbation and is not evidence about "
            "direction specificity.")

    if meta.get("n_prompt_clusters") and not meta.get("clustered_intervals_available"):
        lines.append(
            f"No clustered confidence interval is available: this run has "
            f"{int(meta['n_prompt_clusters'])} prompt cluster(s) and this project's "
            "percentile bootstrap requires three. Every number above is a point estimate "
            "with no error bar, which is enough to decide whether to run further and not "
            "enough to quote.")
    elif meta.get("clustered_intervals_available"):
        lines.append(f"Effects are reported with prompt-clustered bootstrap intervals over "
                     f"{int(meta.get('n_prompt_clusters', 0))} prompt cluster(s) in "
                     "`rescue_effects`, paired within unit against the ablation.")

    if not lines:
        return "Q13 produced no measurable cells; the control surface was not measured."
    head = (f"Q13 on {meta.get('checkpoint', 'the checkpoint')}, channel "
            f"{meta.get('dominant_channel')}, layers {meta.get('edit_layers')}, steps "
            f"{meta.get('steps')}, treatment by {meta.get('selection_mode')}.")
    tail = ("These are the measured contrasts. Which of H1 (magnitude), H2 (direction), "
            "H3 (thresholded sinkhood), H4 (functional separation) or H5 (mediation by v*) "
            "they support is a judgement for the writeup; the runner does not make it.")
    return " ".join([head] + lines + [tail])
