r"""Q16 -- is the register's *depth* causal, or only its presence?

The register state forms in a bounded interval of transformer depth, persists, and
dissolves. This module asks whether that interval matters: can the state be induced
earlier or later, and if it can, does it do the same job?

**Four outcomes, deliberately not collapsed into one.** The vocabulary here exists to stop
the last from being read backwards into the first:

``L1`` induced a v\*-aligned residual state at the target depth;
``L2`` that state is also high-norm and survives the whole intervention window;
``L3`` it additionally acquires favourable query-key geometry and draws attention;
``L4`` the downstream computation and the image change.

L4 without L3 is a perturbation with a visible effect, not a retimed register. L1 without
L2 is a direction written into a token that the block immediately discards.

**And three kinds of "persistence", which are different findings:**

- *maintained* -- the state exists only because the hook reinserts it at every site;
- *carried* -- it survives into later blocks that were never patched;
- *reconstructed* -- the network rebuilds or amplifies it on its own.

:func:`persistence_class` labels which of these a run produced, from measurements rather
than from the schedule that was configured.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Mapping, Optional, Sequence, Tuple

import torch

__all__ = [
    "Window", "windows_from_ranges", "extension_bridge", "unrelated_direction",
    "induce_alpha",
    "InductionCalibration", "calibrate_alpha", "EditRecord", "lifecycle_edit",
    "LifecycleCondition", "CONDITIONS", "achieved_lifecycle", "persistence_class",
    # Natural-register-matched induction (primary) and its direction-only control.
    "direction_only_twin", "induce_matched", "RegisterTarget", "RegisterTargets",
    "calibrate_register_match", "maintenance_summary", "maintenance_profile",
    # Suppression judged by what each block receives; four separate results; the
    # measured depth schedule; and the induction-site pilot (Q1/Q4-informed revision).
    "BlockBars", "CleanStateReference", "AlignmentRule", "received_state_rows",
    "received_state_summary", "regrowth_relocation_rows", "regrowth_relocation_summary",
    "ATTENTION_CLASSES", "attention_relocation_rows", "attention_relocation_summary",
    "induction_written", "measured_formation_onset", "LifecycleSchedule",
    "lifecycle_schedule", "schedule_conflicts", "transplant_edit", "align_edit",
    "site_pilot_rows",
    "RegisterStateRule", "rule_selectivity", "SuppressionSite", "Suppressor",
    "suppression_verdict", "classify_achieved_lifetime", "carrier_census",
    # What the next computation actually receives, and the schedule that guards it.
    "DeliveryProbe", "delivered_rows", "delivery_verdict", "delivery_coverage",
    "coverage_recommendation", "SuppressionSchedule", "suppression_schedule",
    "measured_dissolution_onset", "continuity_check",
    # Does anything frozen at one denoising step still hold at the others?
    "carrier_drift", "drift_verdict",
    # Across denoising time, and whether the image changes sit on the edited tokens.
    "denoising_phase_steps", "lifecycle_over_time", "thin_trace", "save_trace_compact",
    "suppressed_positions", "artifact_colocation", "structure_extent",
    # The natural interval by the register test, and the thresholds' range check.
    "register_test_counts", "measured_formation_by_count", "measured_natural_end",
    "threshold_sensitivity",
]


def _unit(v: torch.Tensor) -> torch.Tensor:
    v = torch.as_tensor(v).float().flatten()
    return v / v.norm().clamp_min(1e-12)


def _is_conditional(ctx) -> bool:
    """False when an edit is applied to a non-conditional CFG row (see ``EditContext``)."""
    return getattr(ctx, "branch", "conditional") == "conditional"


def _conditional_row(x: torch.Tensor) -> torch.Tensor:
    """The [N, C] slice the analysis reads; classifier-free guidance puts it last."""
    return x if x.ndim == 2 else x.flatten(0, -3)[-1]


# ------------------------------------------------------------------------ windows
@dataclass(frozen=True)
class Window:
    """A closed block interval, declared before any intervention is run."""

    name: str
    first: int
    last: int

    @property
    def layers(self) -> Tuple[int, ...]:
        return tuple(range(int(self.first), int(self.last) + 1))

    @property
    def length(self) -> int:
        return len(self.layers)

    def crosses(self, boundary: Optional[int]) -> bool:
        """Does this window straddle an architectural boundary at ``boundary``?

        On FLUX.1 the dual-stream blocks end and the single-stream blocks begin at a fixed
        index. A window that crosses it is not comparable to one that does not, and an
        early-versus-late difference across that seam is confounded with the architecture.
        """
        if boundary is None:
            return False
        return int(self.first) < int(boundary) <= int(self.last)

    def row(self, boundary: Optional[int] = None) -> Dict[str, Any]:
        return dict(window=self.name, first=self.first, last=self.last,
                    n_sites=self.length, crosses_architecture_boundary=self.crosses(boundary))


def windows_from_ranges(ranges, *, length: Optional[int] = None,
                        n_layers: Optional[int] = None,
                        late_first: Optional[int] = None) -> Dict[str, Window]:
    r"""Early / natural / late windows from a checkpoint's frozen ``LayerRanges``.

    The natural window is the artifact's own register range. Early is the interval ending
    just before the writer range begins. ``length`` makes early and late **equal in
    size**, which matters: an early window with more intervention sites than the late one
    would confound depth with dose.

    **The late window never overlaps the natural one.** It is where the state exists in
    the conditions that put it LATE, so any block it shares with the natural range is a
    block where "late" and "natural" cannot be told apart, and a late-condition effect
    stops being attributable to lateness. ``late_first`` may move the window further out;
    it may not move it in, and a value inside the natural range is refused.

    Extension -- keeping the natural state alive past its usual end -- needs the state
    maintained through its natural decline, which begins inside the natural range. That
    is not the late window's job. It is a separate BRIDGE (:func:`extension_bridge`), used
    only by the condition that keeps the natural window intact, and reported as such.

    Returns only the windows that exist. An early window is impossible when the writer
    range starts at block 0, and saying so beats silently shifting it.
    """
    writer = tuple(int(v) for v in ranges.writer)
    register = tuple(int(v) for v in ranges.register)
    if register[1] < register[0]:
        raise ValueError(
            f"the artifact's register range is {register}, which contains no block. "
            "There is no natural lifecycle to retime on this checkpoint, and that is the "
            "result rather than something to work around by widening the range.")
    natural = Window("natural", register[0], register[1])
    span = int(length) if length else max(1, min(natural.length, 3))

    out: Dict[str, Window] = {"natural": natural}
    early_last = writer[0] - 1
    early_first = early_last - span + 1
    if early_last >= 0 and early_first >= 0:
        out["early"] = Window("early", early_first, early_last)

    if late_first is not None and int(late_first) <= register[1]:
        raise ValueError(
            f"late_first={late_first} is inside the natural range {register}: the late "
            "window would share blocks with the natural one, and a late-condition effect "
            "could no longer be attributed to lateness. Extension through the natural "
            "decline is extension_bridge()'s job, not the late window's.")
    late_first = int(register[1] + 1 if late_first is None else late_first)
    late_last = late_first + span - 1
    if n_layers is not None:
        late_last = min(late_last, int(n_layers) - 1)
    if n_layers is None or late_first <= int(n_layers) - 1:
        if late_last >= late_first:
            out["late"] = Window("late", late_first, late_last)
    return out


def extension_bridge(natural: Window, late: Window,
                     dissolution_onset: Optional[int]) -> Optional[Window]:
    r"""The blocks that carry the natural state from its decline into the late window.

    NATURAL + LATE asks whether the register's lifetime can be *extended*, and extension
    has to be continuous: the state must never lapse between its natural existence and
    the late window. It naturally starts to dissolve before the natural range ends -- on
    FLUX.1-schnell the carriers decline from block 35 and are gone at 40 -- so an
    induction that only begins at the late window finds nothing to extend and creates a
    second state after a gap.

    The bridge maintains the state from the measured dissolution onset up to the block
    before the late window. It lies inside the natural range by necessity, which is why
    it is a separate window with its own name rather than an extension of LATE: the late
    window stays strictly after the natural one, and the bridge is used ONLY by the
    condition that keeps the natural window intact. A condition that suppresses the
    natural window has nothing to bridge.

    Returns ``None`` when no onset was measured, or when the onset leaves no room before
    the late window -- there is then nothing to bridge, and whether the run was continuous
    is still decided by :func:`continuity_check` rather than assumed.
    """
    if dissolution_onset is None:
        return None
    first = max(int(dissolution_onset), int(natural.first))
    last = int(late.first) - 1
    if first > last:
        return None
    return Window("bridge", first, last)


def unrelated_direction(vstar: torch.Tensor, *, seed: int = 0) -> torch.Tensor:
    """A matched control direction: unit length, orthogonal to ``v*``, fixed by seed.

    Projected off ``v*`` rather than merely sampled, so "unrelated" is exact instead of
    approximate -- at 3072 dimensions a random draw is nearly orthogonal anyway, but
    nearly is not a control.
    """
    v = _unit(vstar)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    draw = torch.randn(v.shape, generator=generator, dtype=torch.float32)
    draw = draw - (draw @ v) * v
    return draw / draw.norm().clamp_min(1e-12)


# --------------------------------------------------------------------- the operator
def induce_alpha(x: torch.Tensor, tokens: Sequence[int], alpha_target: float,
                 direction: torch.Tensor) -> torch.Tensor:
    r"""Set the component along ``direction`` to ``alpha_target``, leaving ``r`` alone.

    .. math:: x' = r + \alpha_{\text{target}} \hat v

    **This does not preserve norm, and it must not.** The natural register *is* a
    high-norm state; an induction that held the recipient at its original length would be
    testing something else. The resulting norm is recorded so the perturbation can be
    compared against the recipient layer's own distribution rather than assumed reasonable.

    Implemented as ``x += (alpha_target - alpha) * v``, which is algebraically the same and
    leaves the orthogonal complement invariant *by construction* -- subtracting a large
    ``alpha v`` and adding another back would lose precision in ``r`` exactly where the
    claim "only the v* component moved" is made.
    """
    out = x.clone()
    v = _unit(direction).to(device=out.device)
    row = _conditional_row(out)
    for token in tokens:
        token = int(token)
        if token >= int(row.shape[0]):
            continue
        alpha = float(row[token].float() @ v)
        out[..., token, :] = out[..., token, :] + (
            (float(alpha_target) - alpha) * v.to(out.dtype))
    return out


@dataclass
class InductionCalibration:
    """How the target coefficient was chosen, and whether it is in distribution."""

    mode: str
    alpha_target: float
    natural_alpha: float
    natural_median_norm: float
    recipient_median_norm: float
    recipient_max_abs_projection: float
    out_of_distribution_ratio: float
    note: str = ""

    @property
    def out_of_distribution(self) -> bool:
        return self.out_of_distribution_ratio > 1.0

    def row(self) -> Dict[str, Any]:
        return dict(calibration_mode=self.mode, alpha_target=self.alpha_target,
                    natural_alpha=self.natural_alpha,
                    natural_median_norm=self.natural_median_norm,
                    recipient_median_norm=self.recipient_median_norm,
                    recipient_max_abs_projection=self.recipient_max_abs_projection,
                    out_of_distribution_ratio=self.out_of_distribution_ratio,
                    out_of_distribution=self.out_of_distribution, note=self.note)


def calibrate_alpha(natural_projection: torch.Tensor, natural_norm: torch.Tensor,
                    recipient_projection: torch.Tensor, recipient_norm: torch.Tensor,
                    tokens: Sequence[int], *, mode: str = "ratio") -> InductionCalibration:
    r"""Choose the target coefficient from clean measurements at both depths.

    ``ratio`` (default) carries the natural register's **dimensionless** alignment --
    :math:`\alpha / \mathrm{median}\|x\|` at the natural window -- onto the recipient
    layer's own median norm. Residual scale drifts with depth, so copying a raw
    coefficient across depths would confound "same state" with "same number".

    ``absolute`` copies the natural coefficient unchanged, which is the right control when
    the question is whether that *particular magnitude* is reachable at all.

    The returned ``out_of_distribution_ratio`` is the target divided by the largest
    absolute projection any token reaches at the recipient layer in the clean run. Above
    1.0, the induction is asking for a state the layer never naturally contains, and an
    image change from such a perturbation is not evidence that a register acquired a new
    function. It is reported, never silently clipped.
    """
    if mode not in ("ratio", "absolute", "recipient_outlier", "cosine_matched"):
        raise ValueError("mode must be 'ratio', 'absolute', 'recipient_outlier' or "
                         "'cosine_matched'")
    ids = [int(t) for t in tokens if int(t) < int(natural_projection.shape[0])]
    if not ids:
        raise ValueError("calibration needs at least one natural carrier token")
    index = torch.as_tensor(ids)
    natural_alpha = float(natural_projection.float()[index].mean())
    natural_median = float(natural_norm.float().median())
    recipient_median = float(recipient_norm.float().median())
    recipient_max = float(recipient_projection.float().abs().max())

    if mode == "ratio":
        target = natural_alpha / max(natural_median, 1e-9) * recipient_median
        note = ("the natural carrier's alpha/median-norm ratio, applied at the recipient "
                "layer's own scale")
    elif mode == "absolute":
        target = natural_alpha
        note = "the natural carrier's raw alpha, copied across depth unscaled"
    elif mode == "cosine_matched":
        # SECONDARY, and never the primary anchor. It sets alpha so the induced token's
        # cos(x, v*) equals the natural carrier's -- which on FLUX, where QK-norm makes
        # the key scale-invariant, means matching the only quantity that reaches the
        # attention logit. That sounds attractive and is the wrong default: the question
        # is whether the SAME KIND of v*-aligned residual state has different
        # consequences at different depths, and depth-dependent query-key preference is
        # one of the outcomes being measured. Matching on it would calibrate away the
        # mechanism under study. Use it as a mechanistic control, to ask "if the key
        # direction is held equal, does the attention still differ", never to set the
        # strength the image conditions run at.
        # The natural carriers' actual cos(x, v*). This used to be alpha / MEDIAN norm --
        # a ratio (13x on FLUX), not a cosine -- clamped to 0.999, so the mode resolved
        # to 0.999 whatever the register's real alignment was.
        per_token = (natural_projection.float()[index] /
                     natural_norm.float()[index].clamp_min(1e-9))
        cosine = float(per_token.mean())
        cosine = max(min(cosine, 0.999), -0.999)
        target = cosine * recipient_median / max((1.0 - cosine ** 2) ** 0.5, 1e-6)
        note = ("SECONDARY CONTROL: alpha chosen so cos(x, v*) matches the natural "
                "carrier's. Not a primary anchor -- it normalises away the "
                "depth-dependent query-key preference that is itself an outcome")
    else:
        # PRIMARY. The recipient layer's own clean residual scale on the v* axis: the
        # largest absolute projection it already holds. A multiple then means "n times
        # what this layer has", which is predeclarable without reference to any outcome.
        # The other two modes reproduce the natural population's outlier wherever they
        # are applied, so at a register-free depth every strength is far out of
        # distribution and every criterion passes for arithmetic reasons.
        target = recipient_max
        note = ("PRIMARY: the recipient layer's own maximum clean projection; multiples "
                "of this are multiples of what the layer already holds")
    return InductionCalibration(
        mode=mode, alpha_target=float(target), natural_alpha=natural_alpha,
        natural_median_norm=natural_median, recipient_median_norm=recipient_median,
        recipient_max_abs_projection=recipient_max,
        out_of_distribution_ratio=abs(float(target)) / max(recipient_max, 1e-9), note=note)


# ------------------------------------------------------------------- edit callables
@dataclass
class EditRecord:
    """What one edit did at one site, for the achieved-perturbation table."""

    layer: int
    step: int
    kind: str
    token: int
    alpha_before: float
    alpha_after: float
    norm_before: float
    norm_after: float
    cosine_after: float
    perturbation_l2: float
    crossed_highnorm_threshold: bool

    def row(self) -> Dict[str, Any]:
        return dict(layer=self.layer, step=self.step, kind=self.kind, token=self.token,
                    alpha_before=self.alpha_before, alpha_after=self.alpha_after,
                    norm_before=self.norm_before, norm_after=self.norm_after,
                    cosine_after=self.cosine_after,
                    perturbation_l2=self.perturbation_l2,
                    crossed_highnorm_threshold=self.crossed_highnorm_threshold)


def lifecycle_edit(kind: str, tokens, direction: torch.Tensor, *,
                   alpha_target=0.0, highnorm_threshold: float = 0.0,
                   records: Optional[List[EditRecord]] = None,
                   register_targets: Optional["RegisterTargets"] = None):
    r"""The edit callable for one window.

    ``induce``
        :math:`x' = r + \alpha_{\text{target}} \hat v`. Not norm preserving, by design.
        Sets the v* projection only: the DIRECTION-ONLY injection, now the control.
    ``induce_matched``
        natural-register-matched induction, the primary operator: the v* projection AND
        the total norm are both set to the natural register's, scaled to the recipient
        block (:func:`induce_matched`). Its sizes come from ``register_targets``
        (:func:`calibrate_register_match`), per denoising step and block, not from
        ``alpha_target``.
    ``suppress``
        the repository's norm-preserving removal: :math:`y = r`, then
        :math:`x' = \|x\| \, y/\|y\|`. Identical to the validated beta/gamma operator at
        :math:`\beta = 0, \gamma = 1`, which a test asserts rather than assumes -- the
        whole comparison between conditions D, E and F depends on all three using the
        *same* suppression, and "we reimplemented it the same way" is not a guarantee.
    ``sham``
        runs the pathway and returns the tensor unchanged.

    ``tokens`` and ``alpha_target`` each accept a **mapping keyed by denoising step** as
    well as a single value. That exists because the frozen selection and the frozen
    coefficient are both read from the clean run at *one* step, and a schedule that fires
    at every step then applies them where they may no longer describe anything --
    positions the register has left, or a magnitude that was typical at step 4 and is
    out of distribution at step 18. :func:`carrier_drift` measures whether that happens on
    a given trajectory and :func:`drift_verdict` says whether freezing is defensible;
    where it is not, pass ``{step: ...}`` and each step gets its own. A step missing from
    the mapping is left untouched, which is deliberate: silently falling back to another
    step's positions is the failure this parameter exists to prevent.

    Every site appends an :class:`EditRecord`, so the achieved perturbation is measured
    rather than inferred from the configuration.
    """
    if kind not in ("induce", "induce_matched", "suppress", "sham"):
        raise ValueError(f"unknown lifecycle edit kind: {kind!r}")
    if kind == "induce_matched":
        return _matched_edit(tokens, direction, register_targets, records,
                             highnorm_threshold)
    per_step_tokens = isinstance(tokens, Mapping)
    per_step_alpha = isinstance(alpha_target, Mapping)
    if not per_step_tokens:
        tokens = [int(t) for t in tokens]

    def edit(image: torch.Tensor, ctx) -> torch.Tensor:
        if int(direction.numel()) != int(image.shape[-1]):
            return image.clone()
        step = int(getattr(ctx, "step", -1))
        ids = ([int(t) for t in tokens.get(step, ())] if per_step_tokens
               else list(tokens))
        target = (float(alpha_target.get(step, float("nan"))) if per_step_alpha
                  else float(alpha_target))
        if per_step_alpha and target != target:
            return image.clone()          # this step was not calibrated; leave it alone
        v = _unit(direction).to(device=image.device)
        before = _conditional_row(image)
        out = image.clone()
        for token in ids:
            if token >= int(before.shape[0]):
                continue
            x = before[token].float()
            alpha, norm = float(x @ v), float(x.norm())
            if kind == "induce":
                out[..., token, :] = out[..., token, :] + (
                    (target - alpha) * v.to(out.dtype))
            elif kind == "suppress":
                residual = x - alpha * v
                length = float(residual.norm())
                # Relative, not absolute: a token that is very nearly a pure multiple of
                # v* has a residual made of rounding error, and rescaling THAT back to
                # ||x|| would hand the block an amplified-noise vector of full register
                # magnitude. Refusing leaves the token as it is and says so in the
                # record, which is the honest outcome -- there is no norm-preserving
                # removal of a direction from a vector that is only that direction.
                if length > 1e-4 * max(norm, 1e-12):
                    out[..., token, :] = (residual * (norm / length)).to(out.dtype)
            if records is not None and _is_conditional(ctx):
                after = _conditional_row(out)[token].float()
                records.append(EditRecord(
                    layer=int(getattr(ctx, "layer", -1)), step=int(getattr(ctx, "step", -1)),
                    kind=kind, token=token, alpha_before=alpha,
                    alpha_after=float(after @ v), norm_before=norm,
                    norm_after=float(after.norm()),
                    cosine_after=float(after @ v / after.norm().clamp_min(1e-12)),
                    perturbation_l2=float((after - x).norm()),
                    crossed_highnorm_threshold=bool(
                        float(after.norm()) >= float(highnorm_threshold))))
        return out

    return edit


# ------------------------------------------------------------------------ conditions
@dataclass(frozen=True)
class LifecycleCondition:
    """One intended schedule. The schedule is a plan, never a result."""

    key: str
    label: str
    early: str = "none"       # "none" | "induce"
    natural: str = "none"     # "none" | "suppress"
    late: str = "none"        # "none" | "induce"
    role: str = "test"
    note: str = ""
    # HOW an "induce" is carried out. 'matched' -- the primary -- reproduces the natural
    # register's relative norm AND v* projection at the recipient block. 'direction_only'
    # sets the v* projection alone, leaving the token near ordinary norm: the earlier,
    # lower-strength injection, kept as a control.
    induction: str = "matched"

    def schedule(self) -> Dict[str, str]:
        return {"early": self.early, "natural": self.natural, "late": self.late}

    def induces(self) -> bool:
        return "induce" in (self.early, self.late)


CONDITIONS: Tuple[LifecycleCondition, ...] = (
    LifecycleCondition("A_clean", "Clean", role="reference",
                       note="the natural lifecycle, unmodified"),
    LifecycleCondition("B_early_natural_intact", "Early induction, natural window intact",
                       early="induce",
                       note="does an early state survive, and does it change the natural "
                            "window even though nothing is hooked there?"),
    LifecycleCondition("C_late_natural_intact", "Late induction, natural window intact",
                       late="induce",
                       note="is the state extended, or is a separate late state created?"),
    LifecycleCondition("D_late_natural_suppressed", "Late induction, natural suppressed",
                       natural="suppress", late="induce",
                       note="read only against F, never against clean alone"),
    LifecycleCondition("E_early_natural_suppressed", "Early induction, natural suppressed",
                       early="induce", natural="suppress",
                       note="read only against F"),
    LifecycleCondition("F_suppression_only", "Natural window suppressed only",
                       natural="suppress", role="control",
                       note="without F, D and E cannot be separated from the cost of "
                            "removing the natural interval"),
    LifecycleCondition("G_sham", "Sham", role="control",
                       note="the same hooks, no tensor changed"),
    # The diagnostic arm. It suppresses ONLY the frozen clean positions, which is what
    # keeps relocation and reconstruction elsewhere visible -- a state-based rule would
    # chase them and hide exactly the behaviour this arm exists to measure. Its image is
    # NOT comparable with the three state-based conditions, because it edits a different
    # number of positions for a different number of hooks.
    LifecycleCondition("X_fixed_carrier_suppression",
                       "Fixed-carrier suppression (diagnostic, not an image condition)",
                       natural="suppress", role="diagnostic",
                       note="frozen positions only, from the writer onward; measures "
                            "whether the model relocates or rebuilds the state"),
)

def direction_only_twin(condition: LifecycleCondition) -> LifecycleCondition:
    """The same schedule with the direction-only injection: the matched arm's control.

    Everything but the induction operator is identical -- same windows, same
    suppression, same steps. The control sets a SMALLER v* projection (the declared
    calibration times its strength) and leaves the rest of the token at its own size, so
    a difference between the twins is what the natural-register-sized state adds over
    that lower-strength injection: projection size and norm TOGETHER, not norm alone.
    How much of the matched norm the projection brings by itself is reported by
    :func:`calibrate_register_match` (``projection_only_norm_ratio``).
    """
    from dataclasses import replace
    return replace(condition, key=f"{condition.key}__dironly",
                   label=f"{condition.label} (direction-only control)",
                   induction="direction_only", role="control",
                   note="v* projection set at ordinary norm; the matched arm's control")


# Predeclared BEFORE any image. Same schedule as B, a lower strength, so that the
# question "does early induction change the image at all" can be asked at a dose that
# leaves natural formation approximately as the clean run leaves it. Its strength rule is
# declared in the notebook and recorded; it is not retuned after seeing a result.
# Direction-only by definition: its strength is a multiple from the direction-only sweep,
# and the matched operator has no strength to lower.
WEAK_EARLY = LifecycleCondition(
    "B_early_weak", "Early induction at a natural-formation-preserving strength",
    early="induce", role="test", induction="direction_only",
    note="predeclared weak arm: the largest swept multiple that relocated no carriers")


CONTROL_CONDITIONS: Tuple[LifecycleCondition, ...] = (
    LifecycleCondition("B_unrelated_direction", "Early induction along an unrelated axis",
                       early="induce", role="control",
                       note="matched magnitude, orthogonal direction"),
    LifecycleCondition("B_ordinary_tokens", "Early induction at ordinary positions",
                       early="induce", role="control",
                       note="matched magnitude and direction, non-carrier positions"),
)


# ---------------------------------------------------------------- achieved lifecycle
def achieved_lifecycle(trace, *, step: int, layers: Sequence[int],
                       carriers: Sequence[int], sink_threshold: float,
                       highnorm_ratio: float = 3.0,
                       condition: str = "") -> List[Dict[str, Any]]:
    """Per-block measurement of what actually happened, for every block in ``layers``.

    Measured outside the intervention windows as well as inside them, because the
    question "did the induced state persist" can only be answered where nothing was
    hooked. Newly emerged high-norm carriers are counted **separately** from the frozen
    clean ones and never merged into them.
    """
    from .control_surface import sink_readout

    frozen = {int(t) for t in carriers}
    rows: List[Dict[str, Any]] = []
    for layer in layers:
        obs = trace.at(int(step), int(layer))
        if obs is None:
            continue
        row: Dict[str, Any] = dict(condition=condition, layer=int(layer), step=int(step))
        if obs.projection is not None:
            projection = obs.projection.float()
            index = torch.as_tensor(sorted(frozen & set(range(int(projection.shape[0])))))
            row["carrier_projection"] = (float(projection[index].mean())
                                         if index.numel() else float("nan"))
            row["all_token_projection_p99"] = float(projection.quantile(0.99))
            row["all_token_projection_max"] = float(projection.max())
        if obs.norm is not None:
            norms = obs.norm.float()
            median = float(norms.median())
            index = torch.as_tensor(sorted(frozen & set(range(int(norms.shape[0])))))
            row["carrier_norm"] = float(norms[index].mean()) if index.numel() else float("nan")
            row["median_norm"] = median
            # The two statistics natural-register-matched induction targets, read back at
            # every block: relative norm and alignment. Relative, because the matched
            # state is defined against each block's own scale.
            row["carrier_norm_ratio"] = (row["carrier_norm"] / max(median, 1e-9)
                                         if index.numel() else float("nan"))
            # Against the ORDINARY tokens of this block specifically -- not the carriers,
            # and not any other high-norm token -- so a condition that inflates many
            # tokens cannot flatter its own carriers by raising the reference.
            ordinary = norms < highnorm_ratio * median
            if index.numel():
                ordinary[index] = False
            ordinary_median = (float(norms[ordinary].median()) if int(ordinary.sum())
                               else median)
            row["ordinary_median_norm"] = ordinary_median
            row["carrier_norm_vs_ordinary"] = (row["carrier_norm"] / max(ordinary_median, 1e-9)
                                               if index.numel() else float("nan"))
            if obs.cosine is not None and index.numel():
                row["carrier_cosine"] = float(obs.cosine.float()[index].mean())
            high = torch.nonzero(norms > highnorm_ratio * median).flatten().tolist()
            row["n_highnorm"] = len(high)
            row["n_highnorm_frozen"] = len([t for t in high if int(t) in frozen])
            row["n_highnorm_new"] = len([t for t in high if int(t) not in frozen])
            new_ids = sorted(int(t) for t in high if int(t) not in frozen)
            row["highnorm_new_ids"] = new_ids[:12]          # display only
            # The FULL sets, because the per-block counts are token-LAYER observations:
            # summing them cannot tell 10 tokens crossing at 40 blocks from 400 tokens
            # crossing once. `carrier_census` needs the identities, not the counts.
            row["highnorm_new_id_set"] = tuple(new_ids)
            row["highnorm_frozen_id_set"] = tuple(
                sorted(int(t) for t in high if int(t) in frozen))
        if obs.top_channel_value is not None and obs.top_channel is not None:
            index = torch.as_tensor(sorted(frozen & set(range(int(obs.top_channel.shape[0])))))
            if index.numel():
                row["carrier_dominant_channel"] = int(
                    torch.mode(obs.top_channel[index].flatten()).values)
                row["carrier_dominant_value"] = float(obs.top_channel_value[index].float().mean())
        if obs.incoming is not None:
            n_tokens = int(obs.incoming.shape[-1])
            # ABSOLUTE attention: a share of the whole key sequence, text included, so a
            # condition that moves mass off the image is not flattered by renormalisation.
            absolute = obs.image_mass is not None and obs.n_keys
            readout = sink_readout(obs, n_tokens=n_tokens, threshold=sink_threshold,
                                   absolute=bool(absolute))
            row["attention_scale"] = "absolute" if absolute else "image_renormalised"
            index = torch.as_tensor(sorted(frozen & set(range(n_tokens))))
            if index.numel():
                # Absolute share of the whole distribution the carriers receive, which is
                # a different readout from "x uniform" and is the one that says how much
                # attention actually went there.
                incoming = obs.incoming.float()
                mass = incoming[:, index].sum(-1)
                if obs.image_mass is not None:
                    mass = mass * obs.image_mass.float().to(mass.device)
                row["carrier_incoming_mass"] = float(mass.mean())
                row["carrier_sink_strength"] = float(readout["sink_strength_headmax"][index].mean())
                # `any` is too weak to be a level: one token out of sixteen clearing the
                # bar satisfies it, and at a depth just after dissolution some carriers
                # are ALREADY sinks in the clean run -- so an `any`-based L3 fires at a
                # near-no-op perturbation and reports baseline sinkhood as induction.
                # The fraction is a population statistic and can be compared with clean.
                row["carrier_sink_fraction"] = float(readout["is_sink"][index].float().mean())
                row["carrier_any_sink"] = bool(readout["is_sink"][index].any())
                row["carrier_n_sink_heads"] = float(readout["n_sink_heads"][index].sum())
                row["carrier_is_sink"] = bool(row["carrier_sink_strength"] >= sink_threshold)
            sinks = {int(t) for t in torch.nonzero(readout["is_sink"]).flatten().tolist()}
            row["n_sinks"] = len(sinks)
            row["n_sinks_new"] = len(sinks - frozen)
            row["new_sink_ids"] = sorted(sinks - frozen)[:12]
            if obs.qk_cosine is not None and index.numel():
                qk = obs.qk_cosine.float().mean(dim=0)
                row["carrier_qk_cosine"] = float(qk[index].mean())
                order = torch.argsort(qk, descending=True).tolist()
                row["carrier_qk_rank"] = float(
                    sum(order.index(int(t)) for t in index.tolist()) / index.numel())
        rows.append(row)
    return rows


def persistence_class(rows: Sequence[Dict[str, Any]], *, window: Window,
                      metric: str = "carrier_projection",
                      retained: float = 0.5,
                      baseline: Optional[Sequence[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Was the state maintained, carried, or reconstructed? Measured, not configured.

    - **maintained**: strong inside the patched window, gone immediately after it;
    - **carried**: still at least ``retained`` of its in-window level after the hooks stop;
    - **reconstructed**: *higher* after the window than inside it, so the network added to
      it rather than merely failing to erase it.

    ``baseline`` is the same measurement from the **clean** run, and passing it switches
    the quantity from the absolute level to the **excess over clean**. For an early
    window this is not optional. The blocks after an early window are exactly where the
    natural register forms, so an absolute reading there measures the model doing its
    ordinary job and reports it as the induced state persisting -- every early row comes
    back "carried" or "reconstructed" whether or not anything survived.

    Returns ``"not measured"`` rather than a guess when there are no blocks after the
    window to read, which is the honest answer for a late window at the end of the stack.
    """
    def series(source):
        return {int(r["layer"]): r[metric] for r in source
                if r.get(metric) is not None and r[metric] == r[metric]}

    treated = series(rows)
    if baseline is not None:
        clean = series(baseline)
        values = {layer: abs(value - clean.get(layer, 0.0))
                  for layer, value in treated.items()}
    else:
        values = {layer: abs(value) for layer, value in treated.items()}
    inside = [v for layer, v in values.items() if window.first <= layer <= window.last]
    after = [v for layer, v in values.items() if layer > window.last]
    if not inside:
        return dict(persistence="not measured", reason="nothing measured inside the window")
    peak = max(inside)
    scale = "excess over clean" if baseline is not None else "absolute"
    if not after:
        return dict(persistence="not measured", in_window_peak=peak, scale=scale,
                    reason="no unpatched block after the window to read")
    tail = max(after)
    ratio = tail / max(peak, 1e-9)
    label = ("reconstructed" if ratio > 1.05 else
             "carried" if ratio >= retained else "maintained")
    return dict(persistence=label, in_window_peak=peak, after_window_peak=tail,
                retention_ratio=ratio, metric=metric, scale=scale)


# ===================================================================== suppression
@dataclass(frozen=True)
class RegisterStateRule:
    r"""When is a token carrying the sparse register state, judged at one hook?

    The same conjunction the frozen selection uses -- **high norm AND strong ``v*``
    projection** -- evaluated on the tensor at the current hook and nothing else. Both
    thresholds are relative to that block's own distribution, so the rule is
    model-specific and depth-specific without importing anything from another layer or
    from a later block.

    The conjunction is the point. A projection percentile *alone* selects whatever
    fraction of the image stream the percentile names -- at p99 on 4096 tokens that is
    41 tokens at **every** block whether or not a register exists there, because a
    percentile always selects. Ordinary tokens carry weaker ``v*`` components too, and
    erasing them all would remove the direction from the image stream rather than
    preventing the sparse state. Requiring high norm as well is what makes the rule
    select ~0 where no register-like state is present. :func:`rule_selectivity` measures
    that on the clean run, and the rule must be frozen on those numbers before any image
    is looked at.
    """

    highnorm_ratio: float = 3.0
    projection_percentile: float = 99.0
    min_projection: float = 0.0

    def select_from_stats(self, norms: torch.Tensor,
                          projections: torch.Tensor) -> List[int]:
        """The criterion, from per-token norm and ``v*`` projection alone.

        Split out from :meth:`select` because the pre-run audit runs over every block of
        the suppression interval, and the full residual slice is ~50 MB per block on real
        FLUX while these two vectors are a few KB. The tracer already records both for
        every layer, so the audit costs nothing.
        """
        norms, projections = norms.float(), projections.float()
        high = norms > float(self.highnorm_ratio) * norms.median().clamp_min(1e-9)
        if not bool(high.any()):
            return []
        bar = torch.quantile(projections, float(self.projection_percentile) / 100.0)
        floor = torch.tensor(float(self.min_projection), device=projections.device)
        aligned = projections > torch.maximum(bar, floor)
        return [int(t) for t in torch.nonzero(high & aligned).flatten().tolist()]

    def select(self, states: torch.Tensor, direction: torch.Tensor) -> List[int]:
        """Token ids meeting the criterion in ``states`` ([N, C]), current hook only."""
        row = _conditional_row(states).float()
        v = _unit(direction).to(row.device)
        return self.select_from_stats(row.norm(dim=-1), row @ v)

    def row(self) -> Dict[str, Any]:
        return dict(highnorm_ratio=self.highnorm_ratio,
                    projection_percentile=self.projection_percentile,
                    min_projection=self.min_projection)


def _rule_select(rule, states, direction, *, step: int, layer: int) -> List[int]:
    """A rule's selection at one hook; a clean-referenced rule also needs WHERE it is."""
    if getattr(rule, "needs_context", False):
        return rule.select(states, direction, step=step, layer=layer)
    return rule.select(states, direction)


def _rule_select_stats(rule, norms, projections, *, step: int, layer: int) -> List[int]:
    if getattr(rule, "needs_context", False):
        return rule.select_from_stats(norms, projections, step=step, layer=layer)
    return rule.select_from_stats(norms, projections)


def rule_selectivity(trace, rule: RegisterStateRule, *, step: int, layers: Sequence[int],
                     direction: torch.Tensor,
                     carriers: Sequence[int] = ()) -> List[Dict[str, Any]]:
    """How many tokens would this rule touch, per block, on the **clean** run?

    The audit that has to happen before a state-based rule is frozen. A rule that selects
    hundreds of tokens per block is not detecting the sparse register state, it is
    erasing the direction from the image stream, and its image effect would be
    uninterpretable. Reports the count, the share of the image, and how much of it is the
    known clean carrier population.
    """
    frozen = {int(t) for t in carriers}
    rows: List[Dict[str, Any]] = []
    for layer in layers:
        obs = trace.at(int(step), int(layer))
        if obs is None or obs.norm is None or obs.projection is None:
            rows.append(dict(layer=int(layer), selected="not measured",
                             reason="no per-token norm/projection captured at this block"))
            continue
        # The OUTPUT of this block is what the hook at the next block's input sees.
        chosen = _rule_select_stats(rule, obs.norm, obs.projection, step=int(step),
                                    layer=int(layer) + 1)
        n_tokens = int(obs.norm.shape[0])
        rows.append(dict(
            layer=int(layer), selected=len(chosen),
            share_of_image=len(chosen) / max(n_tokens, 1),
            overlap_with_frozen_carriers=len(set(chosen) & frozen),
            outside_frozen_carriers=len(set(chosen) - frozen),
            ids=sorted(chosen)[:16]))
    return rows


@dataclass
class SuppressionSite:
    """What the suppressor found and did at one block, before it changed anything."""

    layer: int
    step: int
    mode: str
    n_selected: int
    newly_targeted: List[int]
    max_projection_before: float
    n_highnorm_aligned_before: float
    intervention_l2: float
    max_projection_after: float
    operator: str = "norm_preserving"
    # How many tokens meet the register criterion at this hook, before the edit -- the
    # REGROWTH measure. Equal to n_selected in state mode; in fixed mode n_selected is
    # just the mask size, which says nothing about whether the state came back.
    n_register_like: Optional[int] = None
    # Mean ||r|| / median token norm over the tokens this hook edited, where r is the
    # token with its v* component removed. It is what SUBTRACTIVE removal leaves behind,
    # and it says how far the NORM-PRESERVING operator had to scale that remainder up.
    residual_to_median: float = float("nan")
    # WHICH tokens the hook edited, split by whether they belong to the original clean
    # carrier population. Q1 found that removing the register is followed, in most runs,
    # by ANOTHER token becoming the carrier; a single "regrowth" count cannot tell the
    # original carriers coming back from the state moving to a new position.
    chosen_ids: Tuple[int, ...] = ()
    n_original: Optional[int] = None
    n_new: Optional[int] = None
    new_ids: Tuple[int, ...] = ()
    # What the block RECEIVED -- measured on the tensor the hook hands on, directly, not
    # inferred from the rule. ``received_bar`` is the clean ordinary alignment ceiling the
    # rule enforces at this block; NaN for a rule that has none.
    received_bar: float = float("nan")
    received_max_cosine: float = float("nan")
    received_n_above_bar: Optional[int] = None
    # More aligned than every ordinary token of the clean run at this input: the count (a)
    # reads. Zero whenever the removal bar is at or below the ordinary maximum.
    received_n_register_like: Optional[int] = None
    received_carrier_cosine: float = float("nan")
    received_max_projection_over_median: float = float("nan")
    share_of_image: float = float("nan")

    def row(self) -> Dict[str, Any]:
        return dict(layer=self.layer, step=self.step, mode=self.mode,
                    operator=self.operator,
                    n_selected=self.n_selected,
                    n_register_like=self.n_register_like,
                    n_original=self.n_original, n_new=self.n_new,
                    new_ids=list(self.new_ids)[:16],
                    n_newly_targeted=len(self.newly_targeted),
                    newly_targeted=list(self.newly_targeted)[:16],
                    max_projection_before=self.max_projection_before,
                    n_highnorm_aligned_before=self.n_highnorm_aligned_before,
                    intervention_l2=self.intervention_l2,
                    max_projection_after=self.max_projection_after,
                    residual_to_median=self.residual_to_median,
                    received_bar=self.received_bar,
                    received_max_cosine=self.received_max_cosine,
                    received_n_above_bar=self.received_n_above_bar,
                    received_n_register_like=self.received_n_register_like,
                    received_carrier_cosine=self.received_carrier_cosine,
                    received_max_projection_over_median=self.received_max_projection_over_median,
                    share_of_image=self.share_of_image)


class Suppressor:
    r"""Prevent the sparse ``v*``-aligned state, by fixed mask or by state-based rule.

    One class, two selection modes, **the same operator and the same schedule** -- which
    is what lets the suppression conditions be compared with each other. The operator is
    a constructor argument: ``subtractive`` is the primary, ``norm_preserving`` the
    sink-identity control (see ``operator`` below).

    ``mode="fixed"``
        the frozen clean-run carrier positions, every hook. A mechanistic diagnostic:
        because it never targets anything else, relocation and reconstruction elsewhere
        remain visible.
    ``mode="state"``
        :class:`RegisterStateRule` re-evaluated at every hook on that hook's own tensor.
        What EARLY ONLY, LATE ONLY and SUPPRESSION ONLY use.

    **Regrowth is read off the rule itself.** At the first hook the natural state exists,
    so the rule selects the carrier population. At every later hook, anything it selects
    is a register-like state that re-formed since the previous block -- so
    ``n_selected`` after the first hook *is* the regrowth measure, with no extra
    threshold to choose. ``newly_targeted`` names positions never touched before, which
    separates "the same tokens keep coming back" from "the state is moving".
    """

    def __init__(self, direction: torch.Tensor, *, mode: str = "state",
                 tokens: Sequence[int] = (), rule=None,
                 sites: Optional[List[SuppressionSite]] = None,
                 operator: str = "subtractive", original: Sequence[int] = ()):
        if mode not in ("fixed", "state"):
            raise ValueError("mode must be 'fixed' or 'state'")
        # WHICH counterfactual token replaces a register token. The two are not variants
        # of one another; they answer different questions.
        #
        # ``norm_preserving``  x' = ||x|| r / ||r||. The validated beta=0, gamma=1 operator:
        #     direction changes, magnitude does not. For a register token -- whose norm is
        #     over ten times the median and almost all of it along v* -- this scales the
        #     remainder r up to register magnitude. The token keeps a register-sized norm
        #     pointing along its ordinary content, which later blocks, writing
        #     contributions of ordinary size, can barely move. That is not "a register
        #     that never formed"; it is a different anomalous state.
        # ``subtractive``      x' = r. The token as it would be without its v* component:
        #     ordinary content at roughly ordinary size, which later blocks can write
        #     into normally. The closer counterfactual for "prevent the state".
        if operator not in ("norm_preserving", "subtractive"):
            raise ValueError("operator must be 'norm_preserving' or 'subtractive'")
        self.operator = operator
        if mode == "fixed" and not len(tokens):
            raise ValueError("fixed-carrier suppression needs the frozen clean positions")
        self.per_step_tokens = isinstance(tokens, Mapping)
        if mode == "state" and rule is None:
            raise ValueError("state-based suppression needs a frozen RegisterStateRule")
        # The original clean carriers, to split every selection into "came back" and
        # "somewhere new". In fixed mode they default to the mask itself.
        self.original = {int(t) for t in (original if len(original) else
                                          ([] if isinstance(tokens, Mapping) else tokens))}
        self.direction = direction
        self.mode = mode
        self.tokens = (dict(tokens) if self.per_step_tokens else [int(t) for t in tokens])
        self.rule = rule
        self.sites: List[SuppressionSite] = [] if sites is None else sites
        # Per DENOISING STEP. Once the schedule fires at every step, a single set would
        # mark every token "already seen" after the first step and `newly_targeted` would
        # read zero for the rest of the run -- turning the relocation measure off exactly
        # when there is most to measure.
        self.seen: Dict[int, set] = {}

    def edit(self, image: torch.Tensor, ctx) -> torch.Tensor:
        if int(self.direction.numel()) != int(image.shape[-1]):
            return image.clone()
        v = _unit(self.direction).to(image.device)
        before = _conditional_row(image).float()
        projection = before @ v
        norms = before.norm(dim=-1)
        high = norms > (self.rule.highnorm_ratio if self.rule else 3.0) * \
            norms.median().clamp_min(1e-9)

        fixed = (self.tokens.get(int(getattr(ctx, "step", -1)), ())
                 if self.per_step_tokens else self.tokens)
        step, layer = int(getattr(ctx, "step", -1)), int(getattr(ctx, "layer", -1))
        register_like = (_rule_select(self.rule, image, self.direction, step=step,
                                      layer=layer)
                         if self.rule is not None else None)
        chosen = ([int(t) for t in fixed] if self.mode == "fixed" else register_like)
        chosen = [t for t in chosen if t < int(before.shape[0])]
        conditional = _is_conditional(ctx)
        seen = self.seen.setdefault(step, set())
        new = sorted(set(chosen) - seen)
        if conditional:                    # the records describe the conditional row only
            seen.update(chosen)

        out = image.clone()
        moved = 0.0
        median = float(norms.median().clamp_min(1e-9))
        remainders: List[float] = []
        for token in chosen:
            x = before[token]
            alpha = float(x @ v)
            residual = x - alpha * v
            length = float(residual.norm())
            # See lifecycle_edit: the floor is relative to ||x||, because rescaling a
            # rounding-error residual back to the token's norm would replace the state
            # with amplified noise at full register magnitude.
            if length <= 1e-4 * max(float(x.norm()), 1e-12):
                continue
            remainders.append(length / median)
            replacement = (residual if self.operator == "subtractive"
                           else residual * (float(x.norm()) / length))
            moved += float((replacement - x).norm())
            out[..., token, :] = replacement.to(out.dtype)

        if not conditional:
            return out
        received = _conditional_row(out).float()
        after = received @ v
        received_cos = after / received.norm(dim=-1).clamp_min(1e-9)
        bars = (self.rule.bars(step, layer)
                if self.rule is not None and hasattr(self.rule, "bars") else None)
        bar = float(bars.alignment_bar) if bars is not None else float("nan")
        # What the rule SELECTED, split: the original carriers coming back versus the
        # state appearing at a position that never carried it in the clean run.
        selected = register_like if register_like is not None else chosen
        original = [t for t in selected if t in self.original]
        fresh = sorted(t for t in selected if t not in self.original)
        carriers = [t for t in sorted(self.original) if t < int(received.shape[0])]
        self.sites.append(SuppressionSite(
            layer=layer, step=step,
            mode=self.mode, n_selected=len(chosen), newly_targeted=new,
            max_projection_before=float(projection.max()),
            n_highnorm_aligned_before=float((high & (projection > 0)).sum()),
            intervention_l2=moved, max_projection_after=float(after.max()),
            operator=self.operator,
            residual_to_median=(sum(remainders) / len(remainders)
                                if remainders else float("nan")),
            n_register_like=(len(register_like) if register_like is not None else None),
            chosen_ids=tuple(sorted(chosen)), n_original=len(original), n_new=len(fresh),
            new_ids=tuple(fresh),
            received_bar=bar, received_max_cosine=float(received_cos.max()),
            received_n_above_bar=(int((received_cos >= bar).sum()) if bar == bar else None),
            received_n_register_like=(int((received_cos > bars.ordinary_max_cosine).sum())
                                      if bars is not None
                                      and bars.ordinary_max_cosine == bars.ordinary_max_cosine
                                      else None),
            received_carrier_cosine=(float(received_cos[carriers].mean())
                                     if carriers else float("nan")),
            received_max_projection_over_median=float(after.max()) / median,
            share_of_image=len(chosen) / max(int(before.shape[0]), 1)))
        return out


def suppression_verdict(sites: Sequence[SuppressionSite], *, window: Window,
                        clean_carrier_projection: float,
                        step: Optional[int] = None,
                        n_steps_total: Optional[int] = None,
                        delivery: Optional[Dict[str, Any]] = None,
                        lifecycle_rows: Optional[Sequence[Dict[str, Any]]] = None,
                        baseline_rows: Optional[Sequence[Dict[str, Any]]] = None
                        ) -> Dict[str, Any]:
    r"""Did the schedule prevent the state from being USED, and how hard did it regrow?

    Two questions, reported separately, because only one of them can invalidate a
    condition.

    **Regrowth** -- how often the rule fires again after the first hook. A hook at
    ``BLOCK_INPUT`` of block *k* cleans the stream block *k* then computes from; block
    *k* writes some of the state back, so at block *k+1*'s hook the rule finds it again.
    That is the model rebuilding the state, and it is a **measurement of how strongly it
    does so**, not a failure. An earlier version of this function gated on it, which was
    wrong twice over: it would call a perfectly-guarded interval incomplete merely
    because the network kept trying, and it invited escalating the intervention until the
    number looked right -- towards exactly the indiscriminate erasure the rule exists to
    avoid.

    **Delivery** -- whether any attention or feed-forward inside the interval actually
    read a register-like state. This is the gate, and it comes from
    :func:`delivery_verdict` measured at the normalised tensor the QKV projection
    receives. Between block *k*'s output and block *k+1*'s consumers stands block *k+1*'s
    hook, so regrowth at a hook is consistent with nothing ever being consumed. The one
    place that is not true is the interval's own terminal boundary, which
    :func:`suppression_schedule` closes with an extra hook and
    :func:`delivery_verdict` scores separately.

    **The raw ``v*`` projection is reported but does not gate.** Ordinary tokens carry
    weaker ``v*`` components, and the goal is to prevent the sparse register state rather
    than erase every trace of the direction.

    A run whose delivery verdict is ``delivered`` is **not** a valid EARLY ONLY or LATE
    ONLY result and its image must be reported as incomplete suppression rather than as
    an achieved schedule. Its image is still produced and still data.
    """
    steps_hooked = sorted({int(s.step) for s in sites})
    # ``step`` selects ONE denoising step's hooks. Without it a schedule that fires at
    # every step would interleave 20 sweeps down the stack, and "the rule re-fired after
    # the first hook" would be counting the next step's first hook as regrowth.
    inside = [s for s in sites if window.first <= s.layer <= window.last
              and (step is None or int(s.step) == int(step))]
    if not inside:
        return dict(suppression="not measured", reason="no hook fired inside the window",
                    steps_hooked=len(steps_hooked))
    inside = sorted(inside, key=lambda s: s.layer)
    later = inside[1:]

    def regrowth(site):
        value = getattr(site, "n_register_like", None)
        return int(site.n_selected if value is None else value)

    regrew = [s for s in later if regrowth(s) > 0]
    peak = max((s.max_projection_before for s in later), default=0.0)
    share = peak / max(abs(float(clean_carrier_projection)), 1e-9)

    # Did the SUPPRESSED POPULATION regain sink behaviour? Not "does the image contain
    # sinks" -- it always does; a clean FLUX run carries 130-200 sinks per block, so
    # counting any sink outside the frozen carriers marks every condition as failed,
    # including clean. The question is whether the tokens whose v* component was removed
    # are attracting attention again.
    sink_blocks: List[int] = []
    relocated: List[int] = []
    clean_new = {int(r.get("layer", -1)): int(r.get("n_sinks_new", 0) or 0)
                 for r in (baseline_rows or [])}
    for row in (lifecycle_rows or []):
        layer = int(row.get("layer", -1))
        if not (window.first <= layer <= window.last):
            continue
        if bool(row.get("carrier_is_sink")):
            sink_blocks.append(layer)
        # Relocation is reported, not gated: new sinks BEYOND what the clean run already
        # has at that block. Suppression is meant to prevent the sparse register state,
        # and the model rearranging its ordinary sinks is a different phenomenon.
        here = int(row.get("n_sinks_new", 0) or 0)
        if here > 1.5 * max(clean_new.get(layer, 0), 1):
            relocated.append(layer)

    delivered = str((delivery or {}).get("delivery", "not measured"))
    boundary_open = bool((delivery or {}).get("terminal_received")) \
        if (delivery or {}).get("terminal_received") is not None else None
    # Positive evidence of failure wins over an absent measurement: carriers that are
    # still sinks, or a leaking boundary, make a run incomplete whether or not delivery
    # could be scored. Only with no such evidence does an unmeasured delivery leave the
    # verdict at "not measured" rather than "complete".
    if delivered == "delivered":
        verdict = "incomplete"
        note = ((delivery or {}).get("note", "") +
                ". Report this image as INCOMPLETE SUPPRESSION, not as an achieved "
                "early-only or late-only schedule.")
    elif sink_blocks:
        verdict = "incomplete"
        note = (f"the suppressed carriers were attention sinks at {len(sink_blocks)} "
                f"block(s) (first {sink_blocks[0]}) although their v* component was "
                "removed -- so their sinkhood is not carried by v* alone. Report as "
                "INCOMPLETE SUPPRESSION.")
    elif boundary_open:
        verdict = "incomplete"
        note = ("the interval itself was clean but its terminal boundary leaked: the "
                "first block after it consumed a regrown state. Install the terminal "
                "cleanup hook from suppression_schedule().")
    elif delivered == "not measured":
        verdict = "not measured"
        note = ("whether any computation inside the interval consumed the state is "
                "UNKNOWN -- " + str((delivery or {}).get(
                    "reason", "no delivery probe was run")) + " Regrowth alone cannot "
                "answer it.")
    else:
        verdict = "complete"
        note = ("no attention or feed-forward inside the interval, nor at its terminal "
                "boundary, received a register-like state, and the suppressed carriers "
                f"never regained sink behaviour. The rule re-fired at {len(regrew)} of "
                f"{len(later)} later hooks, which measures how hard the model rebuilds "
                "the state -- it is a finding, not a failure.")
    return dict(
        suppression=verdict,
        delivery=delivered,
        n_hooks=len(inside),
        # The TEMPORAL dose, reported beside the depth dose. A schedule that fires at one
        # denoising step out of twenty leaves the sampler nineteen steps in which to
        # repair the image, and an absent image effect then says nothing about the
        # register -- only about the dose.
        steps_hooked=len(steps_hooked),
        steps_total=(int(n_steps_total) if n_steps_total else None),
        temporal_coverage=(len(steps_hooked) / int(n_steps_total)
                           if n_steps_total else None),
        hooks_all_steps=len([s for s in sites
                             if window.first <= s.layer <= window.last]),
        # Regrowth: reported, never a gate. See the docstring.
        hooks_where_state_regrew=len(regrew),
        regrowth_rate=len(regrew) / max(len(later), 1),
        first_regrowth_layer=regrew[0].layer if regrew else None,
        mean_tokens_rebuilt=(sum(regrowth(s) for s in later) / max(len(later), 1)),
        blocks_where_carriers_were_sinks=len(sink_blocks),
        first_sink_layer=sink_blocks[0] if sink_blocks else None,
        blocks_with_sink_relocation=len(relocated),
        terminal_boundary_open=boundary_open,
        peak_delivered_register_share=(delivery or {}).get(
            "peak_delivered_register_share"),
        # Reported, deliberately NOT a gate -- see the docstring.
        peak_projection_after_first_hook=peak,
        peak_as_share_of_clean_carrier=share,
        total_newly_targeted=len({t for s in inside for t in s.newly_targeted}),
        total_intervention_l2=float(sum(s.intervention_l2 for s in inside)),
        note=note)


# ======================================================== what the lifetime actually was
def classify_achieved_lifetime(rows: Sequence[Dict[str, Any]],
                               baseline: Sequence[Dict[str, Any]], *,
                               induction: Window, natural: Window,
                               collapse: float = 0.25,
                               displacement: float = 0.5) -> Dict[str, Any]:
    r"""Which of three different things did this run actually produce?

    The configured hooks say what was *attempted*. The primary image comparisons have to
    be grouped by what was *achieved*, because these are not the same experiment:

    ``continuous_extension``
        the induced state is still present, above the clean trajectory, in the blocks
        between the induction window and the natural window -- so availability really was
        extended rather than pulsed.
    ``pulse_then_natural``
        the excess over clean collapses before the natural window, and natural formation
        proceeds roughly as in the clean run. **An early pulse followed by ordinary
        natural formation, not a retimed lifecycle**, and must be reported as such.
    ``displacement``
        natural formation is substantially altered -- the frozen carriers lose their
        high-norm state and/or the population relocates. Whatever the image shows, the
        natural lifecycle did not run.

    Deliberately measured as **excess over the clean run** at the carrier positions, so
    the natural register's own behaviour is not mistaken for the induced state surviving.
    ``collapse`` is the fraction of the in-window excess below which the state counts as
    gone; ``displacement`` is the fraction of the clean high-norm carrier count below
    which natural formation counts as disrupted. Both are predeclared, and neither is
    chosen by looking at an image.
    """
    def by_layer(source, key):
        return {int(r["layer"]): r[key] for r in source
                if r.get(key) is not None and r[key] == r[key]}

    treated_p = by_layer(rows, "carrier_projection")
    clean_p = by_layer(baseline, "carrier_projection")
    if not treated_p or not clean_p:
        return dict(achieved="not measured",
                    reason="no carrier projection recorded for this condition")

    excess = {layer: abs(value - clean_p.get(layer, 0.0))
              for layer, value in treated_p.items()}
    inside = [v for layer, v in excess.items()
              if induction.first <= layer <= induction.last]
    if not inside or max(inside) <= 0:
        return dict(achieved="no_induction",
                    reason="the induction window shows no excess over clean")
    peak = max(inside)

    # Natural formation: did the frozen carriers keep their high-norm state?
    treated_high = by_layer(rows, "n_highnorm_frozen")
    clean_high = by_layer(baseline, "n_highnorm_frozen")
    in_natural = [layer for layer in clean_high
                  if natural.first <= layer <= natural.last]
    kept = float("nan")
    if in_natural:
        clean_total = sum(clean_high[layer] for layer in in_natural)
        treated_total = sum(treated_high.get(layer, 0) for layer in in_natural)
        kept = treated_total / max(clean_total, 1e-9)
    new_carriers = sum(r.get("n_highnorm_new", 0) or 0 for r in rows
                       if natural.first <= int(r["layer"]) <= natural.last)

    # The bridge: blocks strictly between the induction window and the natural window,
    # plus the natural window's own first block when they are adjacent.
    if induction.last < natural.first:
        bridge = [layer for layer in excess
                  if induction.last < layer <= natural.first]
    else:
        bridge = [layer for layer in excess if layer > induction.last]
    bridge_excess = [excess[layer] for layer in sorted(bridge)]
    survived = (min(bridge_excess) / peak) if bridge_excess else float("nan")

    if in_natural and kept == kept and kept < float(displacement):
        achieved, note = "displacement", (
            f"only {kept:.0%} of the clean high-norm carrier presence survives in the "
            "natural window, so natural formation did not run as it does in the clean "
            "run; this is not an intact or continuously extended lifecycle")
    elif not bridge_excess:
        achieved, note = "not measured", (
            "no block between the induction and natural windows was observed, so a "
            "pulse cannot be told from an extension")
    elif survived >= float(collapse):
        achieved, note = "continuous_extension", (
            f"{survived:.0%} of the in-window excess is still present where the two "
            "windows meet, so availability was extended rather than pulsed")
    else:
        achieved, note = "pulse_then_natural", (
            f"the excess falls to {survived:.0%} of its in-window peak before the "
            "natural window; this is an EARLY PULSE followed by ordinary natural "
            "formation, not a retimed lifecycle")
    return dict(achieved=achieved, in_window_excess_peak=peak,
                excess_at_bridge=(min(bridge_excess) if bridge_excess else float("nan")),
                survived_fraction=survived,
                natural_carrier_presence_kept=kept,
                new_carriers_in_natural_window=int(new_carriers),
                note=note)


def carrier_census(rows: Sequence[Dict[str, Any]], *, window: Window) -> Dict[str, Any]:
    r"""Unique token positions, not token-layer observations.

    ``n_highnorm_new`` is a **per-block count**, so summing it over a window gives
    token-layer observations: 395 can be ten positions crossing the threshold at forty
    blocks, or three hundred and ninety-five positions crossing once. Those are opposite
    answers to "did the intervention relocate register formation?" -- the first is the
    same population flickering around the bar, the second is genuine relocation.

    Returns the unique counts and the persistence of each position (how many blocks of
    the window it was high-norm at), so the two can be told apart:

    ``unique_new`` small with high ``median_blocks_per_new`` -- a stable relocated
    population.  ``unique_new`` large with ``median_blocks_per_new`` near 1 -- threshold
    flicker, and the cumulative count is an artefact of the bar, not a finding.
    """
    from collections import Counter

    new_hits: "Counter[int]" = Counter()
    frozen_hits: "Counter[int]" = Counter()
    blocks = 0
    for row in rows:
        layer = int(row.get("layer", -1))
        if not (window.first <= layer <= window.last):
            continue
        blocks += 1
        new_hits.update(row.get("highnorm_new_id_set") or ())
        frozen_hits.update(row.get("highnorm_frozen_id_set") or ())

    def summarise(counter, label):
        if not counter:
            return {f"unique_{label}": 0, f"observations_{label}": 0,
                    f"median_blocks_per_{label}": 0.0,
                    f"{label}_present_throughout": 0}
        counts = sorted(counter.values())
        middle = counts[len(counts) // 2]
        return {f"unique_{label}": len(counter),
                f"observations_{label}": int(sum(counts)),
                f"median_blocks_per_{label}": float(middle),
                f"{label}_present_throughout": int(sum(1 for c in counts if c == blocks))}

    out: Dict[str, Any] = dict(blocks_in_window=blocks)
    out.update(summarise(new_hits, "new"))
    out.update(summarise(frozen_hits, "frozen"))
    if out["unique_new"]:
        out["interpretation"] = (
            "a stable relocated population"
            if out["median_blocks_per_new"] >= 0.5 * max(blocks, 1)
            else "threshold flicker rather than relocation: most new positions are "
                 "high-norm at only a few blocks, so the cumulative count reflects the "
                 "bar rather than a moved register")
    else:
        out["interpretation"] = "no position outside the frozen set became high-norm"
    out["top_new_positions"] = [t for t, _ in new_hits.most_common(8)]
    return out


# ================================ what the next computation actually receives
#
# A suppression hook at ``BLOCK_INPUT`` removes the state from the residual stream, and
# the block it guards then writes some of it back, so the rule fires again at the next
# block. That re-firing is REGROWTH. It is not, by itself, a failure: the question the
# experiment asks is whether any computation ever *consumed* a register-like state, and
# between one block's output and the next block's consumers stands the next suppression
# hook.
#
# The two are separated here because they are different measurements at different
# tensors, and only one of them can invalidate a condition.
#
# The consuming operation is reached through an adaptive norm, and the two families put
# the modulation in different places -- which decides where this has to be measured.
#
#   FLUX single    norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
#                  -> self.attn(hidden_states=norm_hidden_states)   (QKV projection)
#                  -> self.proj_mlp(norm_hidden_states)             (feed-forward)
#   FLUX dual      norm_hidden_states, ... = self.norm1(hidden_states, emb=temb)
#                  -> self.attn(hidden_states=norm_hidden_states)
#   PixArt         norm_hidden_states = self.norm1(hidden_states)          <- PLAIN LN
#                  norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
#                  -> self.attn1(norm_hidden_states, ...)
#
# ``AdaLayerNormZero``/``AdaLayerNormZeroSingle`` apply their scale and shift **inside**
# the module, so on FLUX the norm module's output is exactly what the QKV projection
# receives. ``BasicTransformerBlock`` with ``norm_type="ada_norm_single"`` applies them
# **in the block body**, so on PixArt it is not: a per-channel scale stands in between,
# and a per-channel scale is not a rotation -- it changes each token's alignment with a
# direction by a different amount and can reorder which token is most aligned.
#
# So this probe reads ``InterventionPoint.ATTENTION_INPUT``: the tensor the attention
# module is CALLED with. That is the same quantity on both families, after all
# modulation and after any block-level positional embedding, and it is the only site at
# which "what the computation received" is a measurement rather than an approximation.
#
# What survives that norm is the open question, and it is answered per run rather than
# asserted: LayerNorm divides token magnitude out, but the modulation that follows can
# put some back. ``consumed_norm_spread`` is reported for exactly this reason. Where it
# is ~1 the register's magnitude never reaches a weight and what the computation
# receives is a DIRECTION -- the same fact as FLUX's ``norm_k = RMSNorm`` making the key
# norm constant, one stage earlier -- and delivery is then an alignment question. Where
# it is not ~1, alignment remains necessary but stops being sufficient, and the run says
# so instead of the analysis assuming otherwise.

_DELIVERY_KEYS = ("consumed_norm", "consumed_projection", "consumed_cosine",
                  "supplied_norm", "supplied_projection", "supplied_cosine")


class DeliveryProbe:
    r"""Record what each block SUPPLIES to its normaliser and what it CONSUMES from it.

    A context manager, entered around a generation exactly like
    :class:`~ditsinks.causal_engine.StateProbe`, but it reduces to per-token scalars at
    the hook instead of keeping the state. That is not an optimisation detail: the full
    slice is ~50 MB per block on real FLUX, and this probe runs at every block of the
    suppression interval in every condition, which would be gigabytes of residual stream
    held for three numbers per token.

    Both sides of the same module are recorded, so the two are exactly aligned:

    ``supplied_*``  the residual entering the block -- the tensor the suppression rule is
                    evaluated on, and the one a ``BLOCK_INPUT`` edit rewrites. This probe
                    is entered around the generation, so its hook registers before the
                    edit installer's and reads the state **as the block received it**,
                    which is what makes it the regrowth reading at the rule's own tensor.
    ``consumed_*``  the tensor the image self-attention is CALLED with, after the adaptive
                    norm and after its scale and shift. What the QKV projection reads.

    The pair is what makes the coverage audit possible: the suppression rule is evaluated
    on ``supplied_*`` and its effect has to be judged on ``consumed_*``.

    **Why the consumed side is the attention's input and not the norm module's output.**
    They are the same tensor on FLUX and different tensors on PixArt, whose
    ``BasicTransformerBlock`` applies ``* (1 + scale_msa) + shift_msa`` outside ``norm1``.
    Reading the module output would therefore measure a pre-modulation tensor on PixArt
    and quietly report it as what the computation received.
    """

    def __init__(self, adapter, transformer, tracer, layers: Sequence[int], *,
                 direction: torch.Tensor, point=None, supplied_point=None):
        from .adapters import InterventionPoint
        from .causal_engine import hook_site, image_span

        self._hook_site, self._image_span = hook_site, image_span
        self.adapter = adapter
        self.tracer = tracer
        self.point = (InterventionPoint.ATTENTION_INPUT if point is None
                      else InterventionPoint(point))
        self.supplied_point = (InterventionPoint.BLOCK_INPUT if supplied_point is None
                               else InterventionPoint(supplied_point))
        self.v = _unit(direction)
        self.refs = {r.index: r for r in adapter.layers(transformer)}
        self.layers = [int(l) for l in layers if int(l) in self.refs]
        self.rows: Dict[Tuple[int, int], Dict[str, torch.Tensor]] = {}
        self.unsupported: Dict[int, str] = {}
        self._handles: List[Any] = []

    def __enter__(self) -> "DeliveryProbe":
        for layer in self.layers:
            for point, prefix in ((self.supplied_point, "supplied"),
                                  (self.point, "consumed")):
                try:
                    module, side = self._hook_site(self.adapter, self.refs[layer], point)
                except (NotImplementedError, KeyError) as exc:
                    self.unsupported[layer] = str(exc)
                    continue
                hook = (self._make_pre(layer, prefix) if side == "pre"
                        else self._make_post(layer, prefix))
                register = (module.register_forward_pre_hook if side == "pre"
                            else module.register_forward_hook)
                self._handles.append(register(hook, with_kwargs=True))
        return self

    def __exit__(self, *exc):
        for handle in self._handles:
            try:
                handle.remove()
            except Exception:
                pass
        self._handles.clear()
        return False

    # -------------------------------------------------------------- recording
    def _reduce(self, value, layer: int, prefix: str) -> None:
        tensors = ([value] if torch.is_tensor(value)
                   else [t for t in (value if isinstance(value, (tuple, list)) else [])
                         if torch.is_tensor(t)])
        for tensor in tensors:
            span = self._image_span(
                tensor, self.tracer.n_img or (tensor.shape[1] if tensor.ndim == 3 else 0),
                self.tracer.n_txt)
            if span is None:
                continue
            row = int(tensor.shape[0]) - 1
            image = tensor[row, span, :].detach().float()
            v = self.v.to(image.device)
            norm = image.norm(dim=-1)
            projection = image @ v
            store = self.rows.setdefault((int(self.tracer.step), int(layer)), {})
            store[f"{prefix}_norm"] = norm.cpu()
            store[f"{prefix}_projection"] = projection.cpu()
            store[f"{prefix}_cosine"] = (projection / norm.clamp_min(1e-9)).cpu()
            return

    def _make_pre(self, layer: int, prefix: str):
        # Both call conventions in use: FLUX passes `hidden_states=` as a keyword to its
        # attention and to its blocks; PixArt passes the stream positionally to both.
        @torch.no_grad()
        def pre_hook(module, args, kwargs):
            if self.tracer.recording:
                value = kwargs.get("hidden_states", kwargs.get("x")) \
                    if isinstance(kwargs, dict) else None
                self._reduce(value if torch.is_tensor(value) else tuple(args),
                             layer, prefix)
            return None
        return pre_hook

    def _make_post(self, layer: int, prefix: str):
        @torch.no_grad()
        def post_hook(module, args, kwargs, out):
            if self.tracer.recording:
                self._reduce(out, layer, prefix)
            return None
        return post_hook

    # ------------------------------------------------------------- read-back
    def at(self, step: int, layer: int) -> Optional[Dict[str, torch.Tensor]]:
        return self.rows.get((int(step), int(layer)))

    @classmethod
    def from_rows(cls, rows: Dict[Tuple[int, int], Dict[str, torch.Tensor]], *,
                  unsupported: Optional[Dict[int, str]] = None) -> "DeliveryProbe":
        """A probe over statistics already recorded, for reanalysis without the model.

        The per-token vectors are small enough to save beside a run, so the delivery
        analysis can be redone -- at a different tolerance, against a different carrier
        set -- without another generation.
        """
        probe = cls.__new__(cls)
        probe.rows = {(int(s), int(l)): dict(v) for (s, l), v in rows.items()}
        probe.unsupported = dict(unsupported or {})
        probe.layers = sorted({l for _, l in probe.rows})
        probe.v = None
        probe._handles = []
        return probe


def delivered_rows(probe: DeliveryProbe, *, step: int, layers: Sequence[int],
                   carriers: Sequence[int] = (),
                   rule: Optional[RegisterStateRule] = None) -> List[Dict[str, Any]]:
    r"""Per block: what the consuming operation received, and from whom.

    ``ordinary_*`` are computed over the image tokens that are **not** frozen carriers,
    and they are the reference the gate uses. The point of scoring against the clean
    run's own ordinary ceiling rather than against zero is that the direction ``v*`` is
    not absent from an ordinary token -- it is a real axis of the residual stream and
    every token has some component along it. Suppression is meant to prevent the sparse
    register state, not to erase the axis, so "no register was delivered" has to mean
    "nothing arrived more aligned than the model's own ordinary tokens".

    ``rule`` widens "register" from the frozen carriers to **every token the rule selects
    at that block's input**. Without it, a frozen set smaller than the real population
    leaves genuine register tokens in the "ordinary" pool, the ordinary ceiling is then
    set by a register, and the scale the gate divides by collapses toward zero. Pass it
    for the clean reference; it is harmless on an intervened run, whose rows the gate
    reads only for ``image_cos_max``.
    """
    frozen = {int(t) for t in carriers}
    rows: List[Dict[str, Any]] = []
    for layer in layers:
        store = probe.at(step, int(layer))
        if not store or "consumed_cosine" not in store:
            rows.append(dict(layer=int(layer), delivered="not measured",
                             reason=probe.unsupported.get(int(layer),
                                                          "no hook fired at this block")))
            continue
        cos = store["consumed_cosine"].float()
        norm = store["consumed_norm"].float()
        projection = store["consumed_projection"].float()
        n = int(cos.shape[0])
        register = set(frozen)
        if rule is not None and "supplied_norm" in store:
            register |= {int(t) for t in _rule_select_stats(
                rule, store["supplied_norm"], store["supplied_projection"],
                step=int(step), layer=int(layer))}
        mask = torch.ones(n, dtype=torch.bool)
        ids = torch.tensor(sorted(t for t in register if t < n), dtype=torch.long)
        if ids.numel():
            mask[ids] = False
        ordinary = cos[mask]
        # Projections, beside the cosines. The adaptive norm adds ONE shift vector to every
        # token; it cancels inside attention's softmax (a common key offset adds the same
        # logit to every key), but it does not cancel inside a cosine -- a large shift
        # pulls every token's cosine toward cos(shift, v*) and can flatten real
        # differences to nothing. Differences of projections are shift-invariant.
        ordinary_projection = projection[mask]
        o_med = float(ordinary_projection.median()) if ordinary_projection.numel() else float("nan")
        o_mad = (float((ordinary_projection - o_med).abs().median())
                 if ordinary_projection.numel() else float("nan"))
        row = dict(
            layer=int(layer), n_tokens=n,
            # The LayerNorm proof, read off this run rather than off the source: if this
            # is ~1.0 the consuming operation cannot see token magnitude at all.
            consumed_norm_spread=float(norm.max() / norm.median().clamp_min(1e-9)),
            supplied_norm_spread=(
                float(store["supplied_norm"].max() /
                      store["supplied_norm"].median().clamp_min(1e-9))
                if "supplied_norm" in store else float("nan")),
            image_cos_max=float(cos.max()),
            image_proj_max=float(projection.max()),
            ordinary_cos_max=float(ordinary.max()) if ordinary.numel() else float("nan"),
            ordinary_cos_p99=(float(torch.quantile(ordinary, 0.99))
                              if ordinary.numel() else float("nan")),
            ordinary_cos_median=(float(ordinary.median())
                                 if ordinary.numel() else float("nan")),
            ordinary_proj_max=(float(ordinary_projection.max())
                               if ordinary_projection.numel() else float("nan")),
            ordinary_proj_median=o_med, ordinary_proj_mad=o_mad)
        if ids.numel():
            carrier_cos = cos.index_select(0, ids)
            row.update(n_carriers=int(ids.numel()),
                       n_rule_selected=int(ids.numel()) - len(frozen & set(range(n))),
                       carrier_cos_max=float(carrier_cos.max()),
                       carrier_cos_mean=float(carrier_cos.mean()),
                       carrier_cos_min=float(carrier_cos.min()),
                       carrier_proj_max=float(projection.index_select(0, ids).max()),
                       carrier_proj_median=float(projection.index_select(0, ids).median()),
                       # How far the typical register token stands out along v* from
                       # the ordinary population, in robust SDs, shift-invariant.
                       register_vs_ordinary_mads=(
                           (float(projection.index_select(0, ids).median()) - o_med)
                           / max(o_mad, 1e-9)))
        rows.append(row)
    return rows


def delivery_verdict(rows: Sequence[Dict[str, Any]],
                     baseline_rows: Sequence[Dict[str, Any]], *,
                     window: Window, terminal: Optional[int] = None,
                     tolerance: float = 0.25,
                     min_headroom: float = 0.10,
                     min_scored_fraction: float = 0.5) -> Dict[str, Any]:
    r"""Did any consuming operation inside the interval receive a register-like state?

    The gate the suppression conditions are judged on. For each block, the excess
    alignment delivered over the clean run's ordinary ceiling at that same block, as a
    share of what the clean run's register delivers there:

    .. math::
        s_k = \frac{\max_t \cos(\tilde x^{(k)}_t, v^*) - c^{\text{ord}}_k}
                   {\cos^{\text{carrier}}_k - c^{\text{ord}}_k}

    where :math:`\tilde x` is the normalised tensor the QKV projection reads,
    :math:`c^{\text{ord}}_k` is the clean run's maximum over NON-carrier tokens, and the
    denominator is the clean run's carrier maximum. :math:`s_k \le 0` means nothing
    arrived more aligned than an ordinary token; :math:`s_k = 1` means the consumer
    received exactly what it receives with the natural register in place.

    Scale-free by construction, so it can be compared across depth even though alignment
    itself drifts with depth. ``tolerance`` is declared before the run.

    ``min_headroom`` is the smallest gap, in cosine, between the clean register and the
    clean ordinary ceiling at which a block is scored at all. Below it the share is a
    ratio with a near-zero denominator: the maximum over four thousand tokens moves by a
    few hundredths of a cosine between any two trajectories, and divided by a gap of a
    few hundredths that reads as "twice the natural register". It happens exactly where
    the clean model has not yet written the register into the block's input -- the writer
    blocks, at the start of the suppression interval -- so there is nothing there to
    deliver in the clean run either. Such blocks are reported and not scored.

    ``terminal`` is the block **after** the suppression interval, which is the one place
    a regrown state can be consumed: the last hook guards ``window.last``, and nothing
    stands between that block's output and ``window.last + 1``'s QKV projection unless a
    terminal cleanup hook is installed there. It is scored separately and reported
    whether or not it passes, because a suppression interval that leaks only at its own
    boundary is a different finding from one that leaks throughout.
    """
    clean = {int(r.get("layer", -1)): r for r in baseline_rows}
    per_block: List[Dict[str, Any]] = []
    for row in rows:
        layer = int(row.get("layer", -1))
        reference = clean.get(layer)
        if row.get("delivered") == "not measured" or reference is None:
            continue
        ceiling = float(reference.get("ordinary_cos_max", float("nan")))
        register = float(reference.get("carrier_cos_max", float("nan")))
        headroom = register - ceiling
        scorable = bool(headroom == headroom and headroom >= float(min_headroom))
        share = ((float(row["image_cos_max"]) - ceiling) / headroom
                 if scorable else float("nan"))
        per_block.append(dict(
            layer=layer, image_cos_max=float(row["image_cos_max"]),
            clean_ordinary_ceiling=ceiling, clean_carrier_cos=register,
            clean_headroom=headroom, scored=scorable,
            delivered_register_share=share,
            received=bool(share == share and share > float(tolerance)),
            role=("terminal boundary" if terminal is not None and layer == int(terminal)
                  else "inside interval" if window.first <= layer <= window.last
                  else "outside")))

    inside = [b for b in per_block if b["role"] == "inside interval"]
    boundary = next((b for b in per_block if b["role"] == "terminal boundary"), None)
    if not inside:
        return dict(delivery="not measured",
                    reason="no delivery probe fired inside the interval",
                    per_block=per_block)
    # Every block's share is measured against the gap between the clean run's carrier
    # alignment and its ordinary ceiling. Where that gap is not positive there is no
    # scale to measure on, and calling the result "prevented" would be claiming success
    # from an absent measurement rather than from a clean one.
    scaled = [b for b in inside
              if b["delivered_register_share"] == b["delivered_register_share"]]
    unscored = [int(b["layer"]) for b in inside if not b["scored"]]
    if not scaled:
        return dict(delivery="not measured", blocks_without_register=unscored,
                    reason=("at no block does the clean run's carrier deliver more v* "
                            "alignment than its most-aligned ordinary token, so there is "
                            "no register-delivery scale to score against. Either the "
                            "carriers are not a register at this depth, or the frozen "
                            "direction is not the one they carry."),
                    n_blocks=len(inside), per_block=per_block)
    # A verdict on the interval needs most of the interval. "Prevented, 0 of 1 scored
    # blocks (of 23)" is not evidence of prevention: 22 blocks were never measured.
    if len(scaled) < float(min_scored_fraction) * len(inside):
        return dict(delivery="not measured", blocks_without_register=unscored,
                    n_blocks=len(inside), n_blocks_scored=len(scaled),
                    reason=(f"only {len(scaled)} of {len(inside)} blocks have a clean "
                            "register that stands out at the consumer, below the declared "
                            f"{min_scored_fraction:.0%}. A verdict from so few blocks would "
                            "describe them, not the interval."),
                    per_block=per_block)
    received = [b for b in scaled if b["received"]]
    peak = max(b["delivered_register_share"] for b in scaled)
    verdict = "prevented" if not received else "delivered"
    note = {
        "prevented": ("no consuming operation inside the interval received a state more "
                      f"v*-aligned than {tolerance:.0%} of the way from this block's "
                      "ordinary ceiling to its natural register"),
        "delivered": (f"{len(received)} of {len(scaled)} scored blocks handed their attention "
                      "and feed-forward a register-like state; this interval did not "
                      "achieve the schedule it declares"),
    }[verdict]
    out = dict(
        delivery=verdict, n_blocks=len(inside), n_blocks_scored=len(scaled),
        n_blocks_received=len(received),
        # Blocks where the clean register does not stand out at the consumer: typically
        # the writer blocks at the start of the interval. Reported, not scored.
        blocks_without_register=unscored, min_headroom=float(min_headroom),
        peak_layer=(max(scaled, key=lambda b: b["delivered_register_share"])["layer"]
                    if scaled else None),
        first_receiving_layer=received[0]["layer"] if received else None,
        peak_delivered_register_share=peak, tolerance=float(tolerance), note=note,
        per_block=per_block)
    if boundary is not None:
        out.update(
            terminal_layer=int(boundary["layer"]),
            terminal_delivered_register_share=boundary["delivered_register_share"],
            terminal_received=bool(boundary["received"]),
            terminal_note=(
                "the first block after the interval consumed a regrown state: its "
                "attention saw the register the interval was meant to remove"
                if boundary["received"] else
                "the first block after the interval received nothing register-like, so "
                "the boundary is closed"))
    elif terminal is not None:
        out.update(terminal_layer=int(terminal), terminal_delivered_register_share=None,
                   terminal_received=None,
                   terminal_note="the boundary block was not probed, so the one place a "
                                 "regrown state can be consumed is UNMEASURED")
    return out


def delivery_coverage(probe: DeliveryProbe, rule: RegisterStateRule, *,
                      step: int, layers: Sequence[int],
                      carriers: Sequence[int] = ()) -> List[Dict[str, Any]]:
    r"""Does the high-norm AND aligned rule catch everything that DELIVERS the state?

    The audit that answers "is the conjunction missing a precursor?", and it has to be
    run on the **clean** trajectory, before any suppression, because the question is
    about the rule rather than about the intervention.

    The concern is specific and the geometry makes it real. The rule is evaluated on the
    residual (``supplied_*``), where the register is both large and aligned. The consumer
    reads the normalised tensor (``consumed_*``), where **magnitude has been divided
    out**. A token that is strongly aligned with ``v*`` but has not yet crossed the
    high-norm bar -- a register in the course of being written, at the block where the
    writer is depositing into it -- is invisible to the rule and yet can arrive at the
    QKV projection pointing very nearly where the finished register points.

    So for every block: evaluate the rule on what the block was supplied, then ask how
    many UNSELECTED tokens delivered an alignment at or above the weakest alignment the
    rule did catch. Those are the misses, and ``bar_to_cover`` is the high-norm ratio
    that would have caught them. Both are measurements; neither changes the rule. A rule
    is broadened only if this says it must be, and then to the measured bar rather than
    to every token carrying some ``v*``.
    """
    frozen = {int(t) for t in carriers}
    rows: List[Dict[str, Any]] = []
    for layer in layers:
        store = probe.at(step, int(layer))
        if not store or "consumed_cosine" not in store or "supplied_norm" not in store:
            rows.append(dict(layer=int(layer), coverage="not measured",
                             reason=probe.unsupported.get(
                                 int(layer), "both sides of the normaliser are needed")))
            continue
        supplied_norm = store["supplied_norm"].float()
        supplied_projection = store["supplied_projection"].float()
        consumed_cos = store["consumed_cosine"].float()
        n = int(consumed_cos.shape[0])
        selected = [t for t in _rule_select_stats(rule, supplied_norm, supplied_projection,
                                                  step=int(step), layer=int(layer))
                    if t < n]
        row = dict(layer=int(layer), n_tokens=n, n_selected=len(selected),
                   overlap_with_frozen_carriers=len(set(selected) & frozen))
        if not selected:
            # Nothing to under-cover: the rule finds no register-like state here, which
            # is the expected answer outside the register's own interval.
            row.update(n_missed=0, floor_consumed_cos=float("nan"),
                       max_missed_consumed_cos=float(consumed_cos.max()),
                       bar_to_cover=float("nan"),
                       note="rule selects nothing at this block")
            rows.append(row)
            continue
        ids = torch.tensor(sorted(selected), dtype=torch.long)
        floor = float(consumed_cos.index_select(0, ids).min())
        mask = torch.ones(n, dtype=torch.bool)
        mask[ids] = False
        missed = torch.nonzero((consumed_cos >= floor) & mask).flatten()
        ratios = (supplied_norm / supplied_norm.median().clamp_min(1e-9))
        row.update(
            floor_consumed_cos=floor,
            n_missed=int(missed.numel()),
            missed_ids=[int(t) for t in missed[:16].tolist()],
            max_missed_consumed_cos=(float(consumed_cos.index_select(0, missed).max())
                                     if missed.numel() else float("nan")),
            # What the high-norm bar would have to be to catch them. Reported per block;
            # the recommendation is the minimum over the interval.
            bar_to_cover=(float(ratios.index_select(0, missed).min())
                          if missed.numel() else float("nan")),
            missed_max_norm_ratio=(float(ratios.index_select(0, missed).max())
                                   if missed.numel() else float("nan")),
            note=("every token delivering as much v* alignment as the weakest selected "
                  "one was selected" if not missed.numel() else
                  f"{int(missed.numel())} token(s) below the norm bar delivered at least "
                  "as much alignment as the weakest token the rule caught"))
        rows.append(row)
    return rows


def coverage_recommendation(rows: Sequence[Dict[str, Any]], rule: RegisterStateRule,
                            *, window: Optional[Window] = None) -> Dict[str, Any]:
    """Summarise :func:`delivery_coverage`: is the rule sufficient, and if not, what bar?

    Deliberately conservative about the answer "broaden it". A handful of misses at one
    block, at an alignment barely above the floor, is not the same finding as a
    systematic precursor the rule cannot see, and the two are separated here by counting
    the blocks at which any miss occurred rather than by totalling the misses.
    """
    measured = [r for r in rows if r.get("coverage") != "not measured"
                and (window is None or window.first <= int(r["layer"]) <= window.last)]
    firing = [r for r in measured if int(r.get("n_selected", 0) or 0) > 0]
    with_misses = [r for r in firing if int(r.get("n_missed", 0) or 0) > 0]
    bars = [float(r["bar_to_cover"]) for r in with_misses
            if r.get("bar_to_cover") == r.get("bar_to_cover")]
    recommended = min(bars) if bars else None
    sufficient = not with_misses
    return dict(
        blocks_measured=len(measured), blocks_where_rule_fires=len(firing),
        blocks_with_misses=len(with_misses),
        total_missed_token_blocks=sum(int(r.get("n_missed", 0) or 0)
                                      for r in with_misses),
        worst_block=(max(with_misses, key=lambda r: int(r["n_missed"]))["layer"]
                     if with_misses else None),
        current_highnorm_ratio=float(rule.highnorm_ratio),
        recommended_highnorm_ratio=recommended,
        rule_is_sufficient=sufficient,
        note=("the conjunction caught every token that delivered register-like alignment "
              "at every block where it fires, so no precursor escapes it and there is no "
              "case for broadening it"
              if sufficient else
              f"a lower high-norm bar of x{recommended:.2f} (from x{rule.highnorm_ratio:g}) "
              f"would have covered the misses at {len(with_misses)} block(s). Adopt it "
              "only after re-running rule_selectivity at that bar: a bar that selects "
              "hundreds of tokens per block erases the direction from the image stream "
              "rather than preventing the sparse state, and that trade is the decision, "
              "not the coverage number alone"))


@dataclass(frozen=True)
class SuppressionSchedule:
    """The blocks a suppressor guards, terminal boundary included."""

    window: Window
    layers: Tuple[int, ...]
    terminal: Optional[int]
    stopped_before: Optional[int] = None
    note: str = ""

    @property
    def hook_layers(self) -> List[int]:
        """Every block whose input is cleaned, in order."""
        return list(self.layers) + ([self.terminal] if self.terminal is not None else [])

    def row(self) -> Dict[str, Any]:
        return dict(window=self.window.name, first=self.window.first,
                    last=self.window.last, n_hooks=len(self.hook_layers),
                    terminal_cleanup=self.terminal, stopped_before=self.stopped_before,
                    note=self.note)


def suppression_schedule(window: Window, *, n_layers: int,
                         stop_before: Optional[int] = None) -> SuppressionSchedule:
    r"""Turn an interval into the hook list a suppressor actually needs.

    Two things the naive ``list(window.layers)`` gets wrong.

    **The terminal boundary.** A hook at ``BLOCK_INPUT`` of block *k* protects block *k*'s
    own attention and feed-forward. The last hook is at ``window.last``, so block
    ``window.last``'s output -- which may carry a state its own writer just rebuilt --
    enters ``window.last + 1`` with nothing between. One more hook there closes it. It is
    part of the schedule rather than an analysis step, because there is no way to measure
    the leak away afterwards.

    **A following induction.** When an induction window begins immediately after, the
    suppression interval must stop before it or the two operators fight at every shared
    block. ``stop_before`` truncates the interval, and the terminal cleanup then lands on
    the induction's first block, where it runs first and the induction writes the
    intended state on top of a cleaned stream -- which is exactly what "the state exists
    only from here on" means.
    """
    last = int(window.last)
    stopped = None
    if stop_before is not None and int(stop_before) - 1 < last:
        last, stopped = int(stop_before) - 1, int(stop_before)
    if last < window.first:
        raise ValueError(
            f"suppression interval {window.first}-{window.last} is empty once truncated "
            f"before block {stop_before}: the two schedules overlap completely")
    terminal = last + 1 if last + 1 <= int(n_layers) - 1 else None
    if terminal is None:
        note = ("the interval runs to the last block, so there is no following block "
                "for a regrown state to reach")
    elif stopped is not None:
        note = (f"truncated at {last} so it does not overlap the induction beginning at "
                f"{stopped}; the terminal cleanup at {terminal} runs BEFORE that "
                "induction, on the same block input")
    else:
        note = (f"terminal cleanup at block {terminal} closes the boundary: without it "
                f"block {last}'s output reaches block {terminal}'s attention unguarded")
    return SuppressionSchedule(window=window, layers=tuple(range(int(window.first), last + 1)),
                               terminal=terminal, stopped_before=stopped, note=note)


# ============================= continuous extension, or a second state after a gap
def measured_dissolution_onset(baseline_rows: Sequence[Dict[str, Any]], *,
                               natural: Window,
                               metric: str = "carrier_projection",
                               fraction: float = 0.5) -> Dict[str, Any]:
    r"""Where does the natural register actually START to go, on this run?

    Not where the frozen artifact's ``dissolution`` range says it does. The artifact
    declares a range fitted over a whole prompt population; a single trajectory has its
    own onset, and on FLUX.1-schnell the two differ by five blocks -- the declared
    dissolution begins at 40 while the carrier population is already in decline from 35.

    The distinction is the whole of the NATURAL + LATE condition. An induction that
    begins after the state is gone cannot extend a lifetime; it creates a second state.
    To test extension the maintenance has to begin while there is still something to
    maintain, which is the first block where the plateau has broken but the state is
    still present.

    Returns the first block inside ``natural`` at which ``metric`` has fallen below
    ``fraction`` of its plateau (its maximum over the window), together with the block
    where it disappears entirely, so a caller can see how much room the transition has.
    """
    inside = [(int(r["layer"]), float(r.get(metric, float("nan"))))
              for r in baseline_rows
              if natural.first <= int(r.get("layer", -1)) <= natural.last
              and r.get(metric) is not None]
    inside = sorted((l, v) for l, v in inside if v == v)
    if not inside:
        return dict(onset=None, reason=f"{metric} was not measured inside the window")
    plateau = max(v for _, v in inside)
    peak_layer = next(l for l, v in inside if v == plateau)
    bar = float(fraction) * plateau
    decline = [l for l, v in inside if l > peak_layer and v < bar]
    gone = [l for l, v in inside if l > peak_layer and v < 0.1 * plateau]
    onset = decline[0] if decline else None
    return dict(
        onset=onset, plateau_layer=peak_layer, plateau_value=plateau,
        threshold=bar, fraction=float(fraction), metric=metric,
        disappears_at=gone[0] if gone else None,
        blocks_of_transition=((gone[0] - onset) if (onset is not None and gone) else None),
        note=("the register is still present at this block and already declining, which "
              "is where maintenance has to start for the test to be about extension"
              if onset is not None else
              "the metric never falls below the threshold inside the window, so this run "
              "shows no dissolution to extend through"))


def continuity_check(rows: Sequence[Dict[str, Any]],
                     baseline_rows: Sequence[Dict[str, Any]], *,
                     first: int, last: int,
                     metric: str = "carrier_projection",
                     presence: float = 0.5) -> Dict[str, Any]:
    r"""Was the state present at EVERY block from ``first`` to ``last``, or did it lapse?

    The measurement that decides whether NATURAL + LATE produced what it is named after.
    Presence at a block is the condition's own value of ``metric`` against a bar set from
    the **clean plateau** -- the same absolute bar at every depth, because a bar that
    followed the clean run's decline would call the clean run itself continuous and could
    never distinguish extension from anything.

    A single absent block between the natural plateau and the late window means the state
    disappeared and a different one was created afterwards. That is
    ``late_re_induction``, and it is a legitimate result reported under its own name; it
    is not lifetime extension and must not be described as such.
    """
    clean = {int(r.get("layer", -1)): float(r.get(metric, float("nan")))
             for r in baseline_rows if r.get(metric) is not None}
    plateau = max((v for l, v in clean.items() if v == v and l <= int(last)),
                  default=float("nan"))
    if not (plateau == plateau) or plateau <= 0:
        return dict(continuity="not measured", clean_plateau=plateau,
                    reason=(f"the clean run's peak {metric} at or before block {last} is "
                            f"{plateau}, which is not a positive level presence can be "
                            "measured against. v* is sign-conventioned so a register "
                            "projects positively; a non-positive peak means the carriers "
                            "are not carrying it here."))
    bar = float(presence) * plateau
    series = sorted((int(r["layer"]), float(r.get(metric, float("nan"))))
                    for r in rows
                    if int(first) <= int(r.get("layer", -1)) <= int(last)
                    and r.get(metric) is not None)
    if not series:
        return dict(continuity="not measured",
                    reason=f"{metric} was not measured between blocks {first} and {last}")
    present = [l for l, v in series if v == v and v >= bar]
    gaps = [l for l, v in series if not (v == v and v >= bar)]
    continuous = not gaps
    return dict(
        continuity="continuous" if continuous else "interrupted",
        label="continuous_extension" if continuous else "late_re_induction",
        first=int(first), last=int(last), metric=metric,
        presence_bar=bar, clean_plateau=plateau, presence_fraction=float(presence),
        n_blocks=len(series), n_present=len(present), gap_blocks=gaps[:16],
        n_gap_blocks=len(gaps),
        first_gap=gaps[0] if gaps else None,
        note=("the state was above the presence bar at every block from the natural "
              "plateau through the late window, so the late schedule extended one "
              "lifetime rather than starting a second"
              if continuous else
              f"the state fell below the presence bar at {len(gaps)} block(s) "
              f"(first at {gaps[0]}) before the late window re-established it. This is "
              "LATE RE-INDUCTION -- a second state after a gap -- and reporting it as a "
              "longer lifetime would be wrong."))


# ====================== does anything frozen at one step still hold at the others?
def carrier_drift(trace, *, layer: int, steps: Sequence[int], direction: torch.Tensor,
                  reference_step: int, rule: Optional[RegisterStateRule] = None,
                  topk: int = 8) -> List[Dict[str, Any]]:
    r"""Re-select the carriers at every denoising step and compare with the frozen set.

    Two things get frozen from one step and then applied at all of them, and they fail
    differently.

    **Positions.** The induction arms write at token ids chosen from the clean run at one
    step. If the register sits on different image tokens at step 2 and step 18, those arms
    spend most of the trajectory inducing at positions that carry nothing -- an
    intervention on ordinary tokens wearing the label of a register experiment. (The
    state-based suppression rule does not have this problem: it re-derives its selection
    at every hook, so it follows the population through depth *and* time. Only the fixed
    mask arm and the induction arms are exposed.)

    **Scale.** ``calibrate_alpha`` anchors the induced coefficient to the recipient
    layer's clean residual scale *at one step*. An early denoising step is far noisier
    than a late one, so the same absolute alpha can be unremarkable at one end of the
    trajectory and wildly out of distribution at the other. This is usually the larger of
    the two effects and the easier one to miss, because nothing about it looks wrong.

    Both are read off the clean run, before any condition is built, and both are cheap:
    per-token norm and projection at one block, which the tracer already records.
    """
    rule = rule or RegisterStateRule()
    v = _unit(direction)

    def select(step):
        obs = trace.at(int(step), int(layer))
        if obs is None or obs.norm is None or obs.projection is None:
            return None, None
        chosen = _rule_select_stats(rule, obs.norm, obs.projection, step=int(step),
                                    layer=int(layer) + 1)
        if chosen:
            order = sorted(chosen, key=lambda t: -float(obs.projection[t]))
            chosen = order[:int(topk)]
        return set(int(t) for t in chosen), obs

    reference, ref_obs = select(reference_step)
    rows: List[Dict[str, Any]] = []
    for step in steps:
        here, obs = select(step)
        if here is None:
            rows.append(dict(step=int(step), carriers="not measured",
                             reason=f"no clean observation at block {layer}"))
            continue
        row = dict(step=int(step), layer=int(layer), n_carriers=len(here),
                   carrier_ids=sorted(here),          # the full set, for reuse
                   ids=sorted(here)[:12],             # truncated, for display
                   median_norm=float(obs.norm.float().median()),
                   max_projection=float(obs.projection.float().max()))
        row["n_reference_carriers"] = len(reference or ())
        if reference or here:
            union = here | reference
            row.update(
                n_shared=len(here & reference),
                # Jaccard, so a step that finds ten carriers where the reference found
                # two cannot score as agreement.
                position_overlap=(len(here & reference) / len(union)) if union else 1.0,
                is_reference=bool(int(step) == int(reference_step)))
        if ref_obs is not None:
            ref_scale = float(ref_obs.projection.float().max())
            row["scale_ratio_to_reference"] = (
                float(obs.projection.float().max()) / ref_scale
                if abs(ref_scale) > 1e-9 else float("nan"))
        rows.append(row)
    return rows


def drift_verdict(rows: Sequence[Dict[str, Any]], *, position_overlap: float = 0.5,
                  scale_ratio: float = 2.0) -> Dict[str, Any]:
    """Are frozen positions and a frozen coefficient defensible on this trajectory?

    Reports the two separately, because the remedies are different: drifting positions
    need per-step re-selection, a drifting scale needs per-step calibration, and a run
    can need one without the other.
    """
    measured = [r for r in rows if r.get("carriers") != "not measured"
                and not r.get("is_reference")]
    if not measured:
        return dict(drift="not measured",
                    reason="no step other than the reference produced a clean selection")
    # No carrier ANYWHERE is not drift -- it is an absent measurement, and calling it
    # drift would send a run to per-step re-selection of nothing.
    if not any(int(r.get("n_carriers", 0) or 0) for r in measured) and \
            not any(int(r.get("n_reference_carriers", 0) or 0) for r in measured):
        return dict(drift="not measured", positions_stable=None, scale_stable=None,
                    n_steps=len(measured),
                    reason=("the criterion selects no carrier at any denoising step, at "
                            "the reference step included, so there is no population "
                            "whose stability could be measured. Either this block holds "
                            "no register on this trajectory, or the rule is too tight."))
    overlaps = [float(r["position_overlap"]) for r in measured
                if r.get("position_overlap") is not None]
    ratios = [float(r["scale_ratio_to_reference"]) for r in measured
              if r.get("scale_ratio_to_reference") == r.get("scale_ratio_to_reference")]
    worst_overlap = min(overlaps) if overlaps else float("nan")
    worst_ratio = max((max(r, 1 / r) for r in ratios if abs(r) > 1e-9),
                      default=float("nan"))
    positions_ok = bool(overlaps) and worst_overlap >= float(position_overlap)
    scale_ok = bool(ratios) and worst_ratio <= float(scale_ratio)
    notes = []
    if not positions_ok:
        notes.append(
            f"the carrier set at the worst step shares only {worst_overlap:.0%} of its "
            "positions with the frozen set, so the induction arms would spend part of "
            "the trajectory writing into tokens that carry nothing. Re-select per step, "
            "or restrict the schedule to the steps that agree.")
    if not scale_ok:
        notes.append(
            f"the recipient scale moves by up to {worst_ratio:.1f}x across the "
            "trajectory, so one frozen coefficient is unremarkable at one end and out of "
            "distribution at the other. Calibrate per step.")
    return dict(
        drift="stable" if (positions_ok and scale_ok) else "drifts",
        positions_stable=positions_ok, scale_stable=scale_ok,
        worst_position_overlap=worst_overlap, worst_scale_ratio=worst_ratio,
        n_steps=len(measured),
        steps_below_overlap=[int(r["step"]) for r in measured
                             if r.get("position_overlap") is not None
                             and float(r["position_overlap"]) < float(position_overlap)],
        note=(" ".join(notes) if notes else
              "positions and scale both hold across the trajectory, so freezing them at "
              "one step is defensible here -- measured, not assumed."))


# ========================================== across denoising time, not only depth
def denoising_phase_steps(phase, n_steps: int) -> List[int]:
    """Resolve an ``INTERVENTION_STEPS`` value to the step indices it names.

    ``'all'``, ``'early'``, ``'mid'`` and ``'late'`` (thirds of the trajectory), or an
    explicit list. Thirds rather than fixed indices so a 4-step schnell run and a 28-step
    dev run are split at the same fractions of the trajectory. That is what makes the
    phases comparable across checkpoints: step N/3 of N sits at roughly the same noise
    level for any N.
    """
    n = int(n_steps)
    if isinstance(phase, str):
        if phase == "all":
            return list(range(n))
        cut1, cut2 = round(n / 3), round(2 * n / 3)
        spans = {"early": range(0, cut1), "mid": range(cut1, cut2), "late": range(cut2, n)}
        if phase not in spans:
            raise ValueError("phase must be 'all', 'early', 'mid', 'late', 'capture' or a "
                             "list of step indices")
        return list(spans[phase])
    return sorted(int(s) for s in phase)


def lifecycle_over_time(trace, *, steps: Sequence[int], layers: Sequence[int],
                        carriers: Sequence[int], sink_threshold: float,
                        highnorm_ratio: float = 3.0, condition: str = "",
                        channels: Sequence[int] = ()) -> List[Dict[str, Any]]:
    """:func:`achieved_lifecycle` at every recorded step: the lifecycle in depth AND time.

    One row per (step, block), plus the recorded register channels' carrier mean when the
    tracer kept them -- channel 154 on FLUX, 293 on PixArt -- so whether the downstream
    structures move with ``v*`` is read from the same table as ``v*`` itself.

    Computed immediately after each run, so :func:`thin_trace` can then drop the heavy
    per-step attention tensors without losing anything this table needs.
    """
    frozen = sorted(int(t) for t in carriers)
    rows: List[Dict[str, Any]] = []
    for step in steps:
        for row in achieved_lifecycle(trace, step=int(step), layers=layers,
                                      carriers=carriers, sink_threshold=sink_threshold,
                                      highnorm_ratio=highnorm_ratio, condition=condition):
            obs = trace.at(int(step), int(row["layer"]))
            values = getattr(obs, "channel_values", None) if obs is not None else None
            if values is not None and frozen:
                ids = torch.as_tensor([t for t in frozen if t < int(values.shape[-1])])
                for i, channel in enumerate(channels):
                    if i < int(values.shape[0]) and ids.numel():
                        row[f"carrier_ch{int(channel)}"] = float(
                            values[i].float().index_select(0, ids).abs().mean())
                        row[f"median_ch{int(channel)}"] = float(
                            values[i].float().abs().median())
            for key in ("highnorm_new_id_set", "highnorm_frozen_id_set", "highnorm_new_ids"):
                row.pop(key, None)
            rows.append(dict(row, step=int(step)))
    return rows


_HEAVY_FIELDS = ("incoming", "qk_cosine", "attention_probs", "attention_logits",
                 "keys", "queries", "values", "states")


def thin_trace(trace, *, keep_steps: Sequence[int]) -> int:
    """Drop the heavy per-step tensors everywhere except ``keep_steps``; return bytes freed.

    Recording every denoising step at every block costs ~1.3 GB of attention summaries
    per run at 1024px on FLUX. Every existing analysis reads only the reference step, and
    :func:`lifecycle_over_time` has already reduced the rest to a table -- so the tensors
    at the other steps can go once that table exists, and memory stays at one run's
    worth however many conditions are run.
    """
    keep = {int(s) for s in keep_steps}
    freed = 0
    for (step, _), obs in list(trace.rows.items()):
        if int(step) in keep:
            continue
        for field_name in _HEAVY_FIELDS:
            value = getattr(obs, field_name, None)
            if torch.is_tensor(value):
                freed += value.element_size() * value.numel()
                setattr(obs, field_name, None)
    return freed


def save_trace_compact(trace, path, *, half: bool = True) -> int:
    """Write every recorded tensor of a trace to ``path``; return the bytes written.

    Stored as ``{(step, layer): {field: tensor}}`` plus the trace's identifying metadata,
    in float16 by default -- the per-token statistics need no more precision than that,
    and it halves a run's footprint on Drive. Call it before :func:`thin_trace` to keep
    the full record.
    """
    blob: Dict[Any, Any] = {"meta": dict(prompt_id=trace.prompt_id, seed=trace.seed,
                                         condition=trace.condition, n_img=trace.n_img,
                                         grid=tuple(trace.grid),
                                         channels=tuple(trace.channels))}
    written = 0
    for key, obs in trace.rows.items():
        entry = {}
        for field_name, value in vars(obs).items():
            if torch.is_tensor(value):
                value = value.half() if (half and value.is_floating_point()) else value
                entry[field_name] = value
                written += value.element_size() * value.numel()
        blob[tuple(int(k) for k in key)] = entry
    torch.save(blob, path)
    return written


# ================================ are the artifacts where the edits were made?
def suppressed_positions(sites: Sequence[SuppressionSite]) -> Dict[int, int]:
    """For every image token ever suppressed: in how many denoising steps it was.

    ``newly_targeted`` records each token the first time it is chosen within a step, so
    the union over a step's sites is exactly that step's suppressed set, and counting
    steps per token needs nothing the sites do not already hold.
    """
    per_step: Dict[int, set] = {}
    for site in sites:
        per_step.setdefault(int(site.step), set()).update(int(t) for t in site.newly_targeted)
    counts: Dict[int, int] = {}
    for tokens in per_step.values():
        for t in tokens:
            counts[t] = counts.get(t, 0) + 1
    return counts


def artifact_colocation(image, reference, *, grid: Tuple[int, int],
                        suppressed: Dict[int, int],
                        top_fraction: float = 0.01) -> Dict[str, Any]:
    r"""Do the image changes sit ON the suppressed tokens, or somewhere else?

    The question that decides what the point artifacts are. Each image token covers a
    square of pixels; the per-token artifact score is the mean absolute difference from
    the reference over that square.

    - Changes concentrated on the suppressed tokens point at the edit itself -- the token
      that was altered is the token that renders differently.
    - Changes elsewhere point at the network: the register's absence altered how OTHER
      tokens were computed, through attention.

    Reported three ways, because each can mislead alone: the mean score on suppressed
    versus never-suppressed tokens; the share of the most-changed tokens that were ever
    suppressed, against the share of all tokens that were (the chance level); and the
    rank correlation between how many steps a token was suppressed and its score.
    """
    import numpy as np

    a = np.asarray(image, dtype=np.float64)
    b = np.asarray(reference, dtype=np.float64)
    if a.shape != b.shape or a.ndim < 2:
        return dict(colocation="not measured", reason="images missing or of different size")
    diff = np.abs(a - b)
    if diff.ndim == 3:
        diff = diff.mean(axis=2)
    rows, cols = int(grid[0]), int(grid[1])
    if rows <= 0 or cols <= 0:
        return dict(colocation="not measured", reason="token grid unknown")
    ph, pw = diff.shape[0] // rows, diff.shape[1] // cols
    if ph == 0 or pw == 0:
        return dict(colocation="not measured", reason="image smaller than the token grid")
    score = diff[:rows * ph, :cols * pw].reshape(rows, ph, cols, pw).mean(axis=(1, 3)).ravel()
    n = score.size
    counts = np.zeros(n)
    for t, c in suppressed.items():
        if 0 <= int(t) < n:
            counts[int(t)] = c
    hit = counts > 0
    if not hit.any():
        return dict(colocation="not measured", reason="no token was ever suppressed")
    k = max(1, int(round(float(top_fraction) * n)))
    top = np.argsort(-score)[:k]
    chance = float(hit.mean())
    in_top = float(hit[top].mean())
    order = lambda x: np.argsort(np.argsort(x))
    rho = float(np.corrcoef(order(counts), order(score))[0, 1]) if n > 2 else float("nan")
    ratio = float(score[hit].mean() / max(score[~hit].mean(), 1e-9)) if (~hit).any() \
        else float("nan")
    local = in_top > 3 * chance and ratio > 2
    return dict(
        colocation="on the suppressed tokens" if local else "not concentrated on them",
        n_tokens=n, n_suppressed=int(hit.sum()), chance_share=chance,
        top_share_suppressed=in_top, top_k=k,
        score_ratio_suppressed_vs_rest=ratio, rank_correlation=rho,
        note=(f"{in_top:.0%} of the {k} most-changed tokens were suppressed, against "
              f"{chance:.0%} by chance; suppressed tokens changed {ratio:.1f}x as much. The "
              "artifacts sit where the edit was made."
              if local else
              f"{in_top:.0%} of the {k} most-changed tokens were suppressed, against "
              f"{chance:.0%} by chance. The change is carried to tokens the edit never "
              "touched -- through the network, not at the edit site."),
        token_scores=score.reshape(rows, cols), token_counts=counts.reshape(rows, cols))


def structure_extent(rows: Sequence[Dict[str, Any]], baseline: Sequence[Dict[str, Any]], *,
                     metric: str, fraction: float = 0.5) -> List[Dict[str, Any]]:
    r"""Per denoising step: the first and last block at which a structure is present.

    The measurement behind "does moving v* move the other structures?". Presence at a
    block is ``metric`` at or above ``fraction`` of the CLEAN run's peak over blocks at
    the same step -- one absolute bar per step, from the clean run, so an intervention
    that halves a structure everywhere reads as a shorter extent rather than as the same
    extent at a lower level.

    Run it on ``carrier_projection`` (v* itself), then on the dominant register channel,
    ``n_highnorm`` and ``carrier_sink_strength``. If suppressing or inducing v* shifts
    the other structures' onset and offset by the same blocks, the lifecycle is coupled;
    if v* moves and they do not, it is not.
    """
    def by_step(source):
        out: Dict[int, Dict[int, float]] = {}
        for r in source:
            value = r.get(metric)
            if value is None or value != value:
                continue
            out.setdefault(int(r["step"]), {})[int(r["layer"])] = float(value)
        return out

    clean, treated = by_step(baseline), by_step(rows)
    result: List[Dict[str, Any]] = []
    for step in sorted(treated):
        reference = clean.get(step)
        # A clean peak that is not positive gives no presence scale: "at least half of
        # it" would then be satisfied by nearly anything. Skip rather than report noise.
        if not reference or max(reference.values()) <= 0:
            continue
        bar = float(fraction) * max(reference.values())
        present = sorted(l for l, v in treated[step].items() if v >= bar)
        clean_present = sorted(l for l, v in reference.items() if v >= bar)
        result.append(dict(
            step=step, metric=metric, bar=bar,
            onset=present[0] if present else None,
            offset=present[-1] if present else None,
            n_blocks=len(present),
            clean_onset=clean_present[0] if clean_present else None,
            clean_offset=clean_present[-1] if clean_present else None,
            clean_n_blocks=len(clean_present),
            onset_shift=(present[0] - clean_present[0]
                         if present and clean_present else None),
            offset_shift=(present[-1] - clean_present[-1]
                          if present and clean_present else None)))
    return result


# ========================== natural-register-matched induction (the primary operator)
def induce_matched(x: torch.Tensor, tokens: Sequence[int], alpha_target,
                   norm_target, direction: torch.Tensor) -> torch.Tensor:
    r"""Give each token the natural register's v* projection AND its total norm.

    A token is :math:`x = \alpha \hat v + r`. Setting only :math:`\alpha` leaves the
    token near ordinary norm; the natural register is over ten times the median norm.
    Matching both also fixes the size of the remainder, since
    :math:`\|x\|^2 = \alpha^2 + \|r\|^2`:

    .. math::
        x' = \alpha' \hat v + \sqrt{\|x\|'^2 - \alpha'^2}\; \hat r

    **What is matched:** those two scalars. **What is not:** the direction :math:`\hat r`
    of the remainder, which stays the recipient token's own content rather than a
    natural register's; anything else a natural register carries in its orthogonal
    complement; and the token's history -- how it got there, and what earlier blocks
    wrote alongside it. A matched state that fails to behave like a register is
    therefore not by itself evidence that depth matters: the unmatched parts are live
    alternatives, and :func:`calibrate_register_match` measures the first of them.

    ``alpha_target`` / ``norm_target`` are a number each, or ``{token: value}``. A batched
    tensor has each row edited with ITS OWN remainder rather than one row's content
    copied into another. (Inside the engine an edit is handed the conditional row's
    [N, C] slice only, so the unconditional branch is not edited -- as for every Q16
    operator.) A token whose remainder is numerically zero has no direction to scale and
    is left alone, as :func:`lifecycle_edit`'s suppression does.
    """
    v = _unit(direction).to(device=x.device)
    out = x.clone()
    rows = [out] if out.ndim == 2 else list(out.reshape(-1, *out.shape[-2:]))
    for row in rows:
        for token in tokens:
            token = int(token)
            if token >= int(row.shape[0]):
                continue
            a_t = float(alpha_target[token] if isinstance(alpha_target, Mapping)
                        else alpha_target)
            n_t = float(norm_target[token] if isinstance(norm_target, Mapping)
                        else norm_target)
            current = row[token].float()
            alpha = float(current @ v)
            residual = current - alpha * v
            length = float(residual.norm())
            if length <= 1e-6 * max(float(current.norm()), 1e-12):
                continue
            remainder = max(n_t * n_t - a_t * a_t, 0.0) ** 0.5
            row[token] = (a_t * v + residual * (remainder / length)).to(row.dtype)
    return out


def _matched_perturbation(norm: float, alpha: float, alpha_target: float,
                          norm_target: float) -> float:
    r""":math:`\|x' - x\|` of the matched edit, from a token's norm and projection alone.

    :math:`x' - x = (\alpha' - \alpha)\hat v + (\|r'\| - \|r\|)\hat r`, both terms
    orthogonal, so the size of the edit on the CLEAN tensor needs no full state.
    """
    remainder = max(norm * norm - alpha * alpha, 0.0) ** 0.5
    target_remainder = max(norm_target * norm_target - alpha_target * alpha_target, 0.0) ** 0.5
    return ((alpha_target - alpha) ** 2 + (target_remainder - remainder) ** 2) ** 0.5


@dataclass(frozen=True)
class RegisterTarget:
    """The natural register's size at ONE recipient block: per token, with a fallback.

    ``per_token`` holds a token's OWN natural statistics when it is one of the natural
    carriers, so the oracle positions keep the population's heterogeneity rather than
    all being set to its mean. Any other token -- an ordinary-position control, say --
    gets the population values.
    """

    alpha: float
    norm: float
    per_token: Dict[int, Tuple[float, float]] = field(default_factory=dict)

    def at(self, token: int) -> Tuple[float, float]:
        return self.per_token.get(int(token), (self.alpha, self.norm))


class RegisterTargets:
    """Targets by (denoising step, block).

    Relative statistics are what is carried across depth and time, so each block of a
    window, and each denoising step, gets targets scaled to ITS OWN median norm. A
    fallback for steps that were not calibrated separately exists only if it is
    registered explicitly with ``step=None``; without one, an uncalibrated step is left
    alone -- the same rule :func:`lifecycle_edit` applies to per-step coefficients.
    """

    def __init__(self):
        self._targets: Dict[Tuple[Optional[int], int], RegisterTarget] = {}

    def add(self, step: Optional[int], layer: int, target: RegisterTarget) -> None:
        self._targets[(None if step is None else int(step), int(layer))] = target

    def at(self, step: int, layer: int) -> Optional[RegisterTarget]:
        exact = self._targets.get((int(step), int(layer)))
        return exact if exact is not None else self._targets.get((None, int(layer)))

    def steps(self) -> List[int]:
        return sorted({s for s, _ in self._targets if s is not None})

    def layers(self) -> List[int]:
        return sorted({l for _, l in self._targets})

    def __len__(self) -> int:
        return len(self._targets)


def _matched_edit(tokens, direction, register_targets, records, highnorm_threshold):
    if register_targets is None:
        raise ValueError("induce_matched needs RegisterTargets from calibrate_register_match")
    per_step_tokens = isinstance(tokens, Mapping)
    fixed = None if per_step_tokens else [int(t) for t in tokens]

    def edit(image: torch.Tensor, ctx) -> torch.Tensor:
        if int(direction.numel()) != int(image.shape[-1]):
            return image.clone()
        step, layer = int(getattr(ctx, "step", -1)), int(getattr(ctx, "layer", -1))
        target = register_targets.at(step, layer)
        if target is None:
            return image.clone()          # no calibration for this block and step
        ids = [int(t) for t in tokens.get(step, ())] if per_step_tokens else fixed
        before = _conditional_row(image).float()
        ids = [t for t in ids if t < int(before.shape[0])]
        out = induce_matched(image, ids, {t: target.at(t)[0] for t in ids},
                             {t: target.at(t)[1] for t in ids}, direction)
        if records is not None and _is_conditional(ctx):
            v = _unit(direction).to(before.device)
            after = _conditional_row(out).float()
            for t in ids:
                norm_after = float(after[t].norm())
                records.append(EditRecord(
                    layer=layer, step=step, kind="induce_matched", token=t,
                    alpha_before=float(before[t] @ v), alpha_after=float(after[t] @ v),
                    norm_before=float(before[t].norm()), norm_after=norm_after,
                    cosine_after=float(after[t] @ v) / max(norm_after, 1e-12),
                    perturbation_l2=float((after[t] - before[t]).norm()),
                    crossed_highnorm_threshold=bool(
                        norm_after >= float(highnorm_threshold))))
        return out

    return edit


def calibrate_register_match(trace, *, step: int, natural_layers,
                             recipient_layers: Sequence[int], carriers: Sequence[int],
                             induce_tokens: Optional[Sequence[int]] = None,
                             natural_states: Optional[torch.Tensor] = None,
                             recipient_states: Optional[Mapping[int, torch.Tensor]] = None,
                             direction: Optional[torch.Tensor] = None
                             ) -> Tuple[Dict[int, RegisterTarget], List[Dict[str, Any]]]:
    r"""Scale the clean natural register's statistics to each recipient block.

    **Reference.** The carriers at the INPUT of each block in ``natural_layers`` (read as
    the previous block's output) on the clean run at ``step`` -- the natural register as
    the natural window's computations receive it. Per carrier, averaged over those
    blocks: :math:`\rho_t = \|x_t\| / \mathrm{median}` and
    :math:`\pi_t = \alpha_t / \mathrm{median}`. Relative, because block scales differ by
    an order of magnitude across the stack and a copied absolute number would mean a
    different thing at every depth.

    **Targets.** At recipient block L the targets are :math:`\rho_t m_L` and
    :math:`\pi_t m_L`, with :math:`m_L` the median norm of L's INPUT -- the tensor the
    edit writes into.

    **Is it unusually large for the recipient?** One report row per block: the target
    against the largest norm and projection ANY clean token holds there, the target's
    percentile among them, the size of the edit on the clean tensor in units of that
    block's median, and ``projection_only_norm_ratio`` -- the norm a token would reach if
    only its projection were set -- which says how much of the norm match the
    projection already brings.

    **What is not matched.** With ``natural_states`` (a full [N, C] slice at a natural
    block's input) and ``recipient_states`` (``{block: [N, C]}`` at recipient inputs) it
    measures the one unmatched quantity a single slice can show: the direction of the
    remainder -- the part of the token orthogonal to v*, which the operator keeps from
    the recipient. Every number has an ORDINARY-token baseline beside it, because the
    residual stream has a direction every token shares, and a raw cosine would report
    that as the register's:

    - ``natural_remainder_coherence`` / ``ordinary_remainder_coherence`` -- how much the
      carriers' remainders point one way, against how much any token's do. A clear excess
      is a shared component the register carries beyond v*.
    - ``remainder_vs_natural_cosine`` / ``ordinary_remainder_baseline`` -- how close each
      carrier's remainder at the recipient is to ITS OWN remainder as a natural register
      (same position), against how close an ordinary token's is to itself across the same
      two depths. A clear deficit is register-specific content the matched state lacks.
    """
    try:
        natural_layers = [int(l) for l in natural_layers]
    except TypeError:
        natural_layers = [int(natural_layers)]
    references = [trace.at(int(step), l - 1) for l in natural_layers]
    references = [o for o in references
                  if o is not None and o.norm is not None and o.projection is not None]
    if not references:
        raise ValueError(f"no clean statistics at the input of blocks {natural_layers}, "
                         f"step {step}")
    n_tokens = min(int(o.norm.shape[0]) for o in references)
    carriers = [int(t) for t in carriers if int(t) < n_tokens]
    if not carriers:
        raise ValueError("calibrate_register_match needs at least one natural carrier")
    rho = {t: 0.0 for t in carriers}
    pi = {t: 0.0 for t in carriers}
    for obs in references:
        median = float(obs.norm.float().median().clamp_min(1e-9))
        for t in carriers:
            rho[t] += float(obs.norm[t]) / median / len(references)
            pi[t] += float(obs.projection[t]) / median / len(references)
    rho_bar = sum(rho.values()) / len(rho)
    pi_bar = sum(pi.values()) / len(pi)
    cos_bar = sum(pi[t] / max(rho[t], 1e-9) for t in carriers) / len(carriers)
    induce_tokens = [int(t) for t in (induce_tokens if induce_tokens is not None else carriers)]

    def unit_remainders(states):
        v = _unit(direction)
        remainder = states.float() - (states.float() @ v)[:, None] * v[None, :]
        return remainder / remainder.norm(dim=-1, keepdim=True).clamp_min(1e-9)

    natural_unit = ordinary = None
    remainder_report: Dict[str, float] = {}
    if natural_states is not None and direction is not None:
        natural_unit = unit_remainders(natural_states)
        in_carriers = set(carriers)
        ordinary = [t for t in range(int(natural_unit.shape[0])) if t not in in_carriers]
        remainder_report["natural_remainder_coherence"] = float(
            natural_unit[carriers].mean(0).norm())
        remainder_report["ordinary_remainder_coherence"] = (
            float(natural_unit[ordinary].mean(0).norm()) if ordinary else float("nan"))

    targets: Dict[int, RegisterTarget] = {}
    rows: List[Dict[str, Any]] = []
    for layer in recipient_layers:
        recipient = trace.at(int(step), int(layer) - 1)
        if recipient is not None and recipient.norm is not None \
                and recipient.projection is not None:
            r_norm, r_proj = recipient.norm.float(), recipient.projection.float()
        elif (recipient_states is not None and int(layer) in recipient_states
              and direction is not None):
            # The trace records block OUTPUTS, so block 0's input -- the embedding -- is
            # never in it. A probed input slice, where one exists, is the same tensor.
            states = recipient_states[int(layer)].float()
            r_norm, r_proj = states.norm(dim=-1), states @ _unit(direction)
        else:
            continue
        m = float(r_norm.median().clamp_min(1e-9))
        per_token = {t: (pi[t] * m, rho[t] * m) for t in induce_tokens if t in rho}
        target = RegisterTarget(alpha=pi_bar * m, norm=rho_bar * m, per_token=per_token)
        targets[int(layer)] = target
        ids = [t for t in induce_tokens if t < int(r_norm.shape[0])]
        edits, alone = [], []
        for t in ids:
            a_t, n_t = target.at(t)
            norm_t, alpha_t = float(r_norm[t]), float(r_proj[t])
            edits.append(_matched_perturbation(norm_t, alpha_t, a_t, n_t))
            alone.append((a_t * a_t + max(norm_t * norm_t - alpha_t * alpha_t, 0.0)) ** 0.5)
        norm_max, proj_max = float(r_norm.max()), float(r_proj.abs().max())
        row = dict(
            step=int(step), layer=int(layer), recipient_median=m,
            natural_blocks=f"{min(natural_layers)}-{max(natural_layers)}",
            n_natural_carriers=len(carriers),
            natural_norm_ratio=rho_bar, natural_projection_ratio=pi_bar,
            natural_cosine=cos_bar,
            target_norm=target.norm, target_alpha=target.alpha,
            recipient_max_norm=norm_max, recipient_max_projection=proj_max,
            recipient_p99_norm=float(r_norm.quantile(0.99)),
            norm_vs_recipient_max=target.norm / max(norm_max, 1e-9),
            alpha_vs_recipient_max=abs(target.alpha) / max(proj_max, 1e-9),
            target_norm_percentile=float((r_norm < target.norm).float().mean() * 100),
            # How far OUTSIDE the recipient's clean distribution, in robust units (median
            # absolute deviations from the median) -- a scale on which "unusually large"
            # has a size, not only a yes or no.
            norm_robust_z=(target.norm - m) / max(float((r_norm - m).abs().median()), 1e-9),
            alpha_robust_z=((target.alpha - float(r_proj.median()))
                            / max(float((r_proj - r_proj.median()).abs().median()), 1e-9)),
            clean_perturbation=(sum(edits) / len(edits)) if edits else float("nan"),
            perturbation_vs_median=((sum(edits) / len(edits)) / m) if edits else float("nan"),
            projection_only_norm_ratio=((sum(alone) / len(alone)) / m) if alone
            else float("nan"))
        # Beyond anything a clean token holds at this block -- in norm, or along v*.
        row["norm_unusually_large"] = bool(row["norm_vs_recipient_max"] > 1.0)
        row["projection_unusually_large"] = bool(row["alpha_vs_recipient_max"] > 1.0)
        row["unusually_large"] = bool(row["norm_unusually_large"]
                                      or row["projection_unusually_large"])
        if (natural_unit is not None and recipient_states is not None
                and int(layer) in recipient_states):
            own = unit_remainders(recipient_states[int(layer)])
            n_common = min(int(own.shape[0]), int(natural_unit.shape[0]))
            matched = [t for t in ids if t in rho and t < n_common]
            others = [t for t in ordinary if t < n_common]
            row.update(remainder_report)
            if matched:
                row["remainder_vs_natural_cosine"] = float(
                    (own[matched] * natural_unit[matched]).sum(-1).mean())
            if others:
                row["ordinary_remainder_baseline"] = float(
                    (own[others] * natural_unit[others]).sum(-1).mean())
        rows.append(row)
    return targets, rows


def _finite(value) -> bool:
    return value is not None and value == value


def maintenance_summary(rows: Sequence[Dict[str, Any]], baseline: Sequence[Dict[str, Any]],
                        *, window: Window, reference_layers: Sequence[int],
                        persist_fraction: float = 0.5) -> Dict[str, Any]:
    r"""Inside a maintained window and after it: did the state look and act like a register?

    Four things per condition, each against the clean natural register over
    ``reference_layers`` (its plateau) as the reference:

    - **alignment** -- carrier cos(x, v*) at block outputs inside the window;
    - **norm** -- carrier norm / that block's median, inside the window;
    - **attention-sink behaviour** -- carrier sink strength, and the share of window
      blocks where the carriers are sinks;
    - **persistence** -- after the window, how many consecutive blocks keep at least
      ``persist_fraction`` of the excess over clean the carriers had at the window's last
      block, for the v* projection, the norm ratio and the sink strength separately.
      Nothing is hooked there, so maintenance guarantees none of it.

    Reads block OUTPUTS -- what survives each block's own processing, which is where a
    register-sized norm could matter if it steadies v* against a block's writes. The
    sink readout of a block is the attention computed INSIDE it, on the maintained input.
    """
    clean = {int(r["layer"]): r for r in baseline}
    here = {int(r["layer"]): r for r in rows}

    def mean(source, layers, key):
        values = [float(source[l][key]) for l in layers
                  if l in source and _finite(source[l].get(key))]
        return sum(values) / len(values) if values else float("nan")

    inside = [l for l in window.layers if l in here]
    reference = [int(l) for l in reference_layers]
    out = dict(
        window=window.name, first=window.first, last=window.last,
        cosine=mean(here, inside, "carrier_cosine"),
        norm_ratio=mean(here, inside, "carrier_norm_ratio"),
        norm=mean(here, inside, "carrier_norm"),
        norm_vs_ordinary=mean(here, inside, "carrier_norm_vs_ordinary"),
        incoming_mass=mean(here, inside, "carrier_incoming_mass"),
        sink_strength=mean(here, inside, "carrier_sink_strength"),
        sink_block_share=(sum(bool(here[l].get("carrier_is_sink")) for l in inside)
                          / len(inside)) if inside else float("nan"),
        natural_cosine=mean(clean, reference, "carrier_cosine"),
        natural_norm_ratio=mean(clean, reference, "carrier_norm_ratio"),
        natural_norm=mean(clean, reference, "carrier_norm"),
        natural_norm_vs_ordinary=mean(clean, reference, "carrier_norm_vs_ordinary"),
        natural_incoming_mass=mean(clean, reference, "carrier_incoming_mass"),
        natural_sink_strength=mean(clean, reference, "carrier_sink_strength"))
    for key in ("cosine", "norm_ratio", "norm", "norm_vs_ordinary", "incoming_mass",
                "sink_strength"):
        natural = out[f"natural_{key}"]
        out[f"{key}_vs_natural"] = (out[key] / natural
                                    if _finite(out[key]) and _finite(natural) and natural
                                    else float("nan"))
    for metric, label in (("carrier_projection", "projection"),
                          ("carrier_norm_ratio", "norm_ratio"),
                          ("carrier_sink_strength", "sink_strength")):
        last = window.last
        if last not in here or last not in clean:
            continue
        excess0 = (float(here[last].get(metric, float("nan")))
                   - float(clean[last].get(metric, float("nan"))))
        blocks = 0
        if _finite(excess0) and excess0 > 0:
            layer = last + 1
            while layer in here and layer in clean:
                excess = (float(here[layer].get(metric, float("nan")))
                          - float(clean[layer].get(metric, float("nan"))))
                if not (_finite(excess) and excess >= persist_fraction * excess0):
                    break
                blocks += 1
                layer += 1
        out[f"excess_{label}_at_window_end"] = excess0
        out[f"{label}_persists_blocks"] = blocks
        # Whether there was anything after the window to read at all: a late window
        # at the end of the stack has 0 blocks of persistence because nothing follows.
        out["blocks_after_window_measured"] = sum(
            1 for l in here if l > last and l in clean)
    return out


def maintenance_profile(rows: Sequence[Dict[str, Any]], baseline: Sequence[Dict[str, Any]],
                        *, window: Window, records: Sequence[EditRecord] = (),
                        step: Optional[int] = None, after: int = 6) -> List[Dict[str, Any]]:
    """Block by block, through a maintained window and ``after`` blocks past it.

    Two readings of every window block, because they answer different questions:
    what the operator WROTE at the block's input (its :class:`EditRecord`, which the
    operator controls) and what the block HANDED ON at its output (which it does not).
    Past the window only the second exists -- that is persistence. The clean run's
    value at every block is beside it, since "excess over clean" is the only reading
    that means anything after an early window, where the natural register forms anyway.
    """
    clean = {int(r["layer"]): r for r in baseline}
    here = {int(r["layer"]): r for r in rows}
    written: Dict[int, List[EditRecord]] = {}
    for record in records:
        if record.kind not in ("induce", "induce_matched"):
            continue
        if step is not None and int(record.step) != int(step):
            continue
        written.setdefault(int(record.layer), []).append(record)

    def avg(values):
        values = [float(v) for v in values if _finite(v)]
        return sum(values) / len(values) if values else float("nan")

    out: List[Dict[str, Any]] = []
    for layer in range(int(window.first), int(window.last) + int(after) + 1):
        if layer not in here:
            continue
        r, c, recs = here[layer], clean.get(layer, {}), written.get(layer, [])
        row = dict(layer=layer, in_window=window.first <= layer <= window.last,
                   written_alpha=avg(x.alpha_after for x in recs),
                   written_norm=avg(x.norm_after for x in recs),
                   written_cosine=avg(x.cosine_after for x in recs),
                   perturbation_l2=avg(x.perturbation_l2 for x in recs))
        for key in ("carrier_projection", "carrier_cosine", "carrier_norm_ratio",
                    "carrier_norm", "carrier_norm_vs_ordinary", "carrier_incoming_mass",
                    "carrier_sink_strength", "carrier_sink_fraction", "carrier_is_sink",
                    "carrier_dominant_channel", "carrier_dominant_value", "median_norm"):
            row[key] = r.get(key, float("nan"))
            row[f"clean_{key}"] = c.get(key, float("nan"))
        excess = (float(row["carrier_projection"]) - float(row["clean_carrier_projection"])
                  if _finite(row["carrier_projection"])
                  and _finite(row["clean_carrier_projection"]) else float("nan"))
        row["excess_projection"] = excess
        row["handed_on_fraction"] = (float(row["carrier_projection"]) / row["written_alpha"]
                                     if _finite(row["written_alpha"]) and row["written_alpha"]
                                     and _finite(row["carrier_projection"]) else float("nan"))
        out.append(row)
    return out


# ========================== suppression judged by what each block RECEIVES (Q1-informed)
@dataclass(frozen=True)
class BlockBars:
    """What "carrying the register state" means at one (denoising step, block OUTPUT).

    Read from the CLEAN run of the same generation, with Q1's definitions: a token is
    **v*-aligned** when its ``cos(x, v*)`` reaches the ``alignment_quantile`` of the
    ORDINARY tokens at this block (neither frozen carriers nor high-norm), and it meets
    the **register criterion** when it is also high-norm (``norm >= highnorm_ratio x
    median``). ``above_bar`` lists every clean token already over the alignment bar -- the
    frozen carriers and the ordinary tail the quantile leaves -- so a treated token is
    called NEW only when the paired clean run did not have it there. Q1 could not prove
    that novelty ("another carrier wins, novelty unproven"); a paired per-block reference
    can.
    """

    step: int
    layer: int
    median_norm: float
    norm_bar: float
    alignment_bar: float
    projection_ceiling: float
    above_bar: Tuple[int, ...] = ()
    # The strict bar the RESULTS use: more aligned than EVERY ordinary token of the clean
    # run here. The removal bar (``alignment_bar``, a quantile) leaves ~0.1% of ordinary
    # tokens above it by construction, so counting "received" or "new" against it would
    # report the ordinary tail of the clean run itself as a v* state.
    ordinary_max_cosine: float = float("nan")
    register_like: Tuple[int, ...] = ()      # clean tokens above the ordinary maximum
    carriers: Tuple[int, ...] = ()           # clean tokens meeting the register criterion
    carrier_projection: float = float("nan")  # the FROZEN carriers' clean means here
    carrier_cosine: float = float("nan")
    carrier_norm: float = float("nan")
    head_sinks: Optional[Tuple[int, ...]] = None    # clean strongest image key per head
    head_share: Optional[Tuple[float, ...]] = None  # its incoming share, per head

    @property
    def concentration(self) -> float:
        if not self.head_share:
            return float("nan")
        return sum(self.head_share) / len(self.head_share)

    def row(self) -> Dict[str, Any]:
        return dict(step=self.step, layer=self.layer, median_norm=self.median_norm,
                    norm_bar=self.norm_bar, alignment_bar=self.alignment_bar,
                    ordinary_max_cosine=self.ordinary_max_cosine,
                    projection_ceiling=self.projection_ceiling,
                    n_above_bar=len(self.above_bar), n_register_like=len(self.register_like),
                    n_register_criterion=len(self.carriers),
                    carrier_projection=self.carrier_projection,
                    carrier_cosine=self.carrier_cosine, carrier_norm=self.carrier_norm,
                    clean_concentration=self.concentration)


class CleanStateReference:
    r"""Per-(step, block) bars and per-head clean sinks, built once from the clean run.

    Must be built BEFORE :func:`thin_trace` drops the per-step attention tensors, because
    the clean sink of every head at every step is what attention relocation is judged
    against. Everything else it holds survives thinning.

    ``at(step, layer)`` describes block ``layer``'s OUTPUT; ``for_input_of(step, layer)``
    describes what arrives at block ``layer``'s INPUT, which is the previous block's
    output -- the tensor a ``BLOCK_INPUT`` hook edits. A step that was not recorded falls
    back to the nearest recorded step at the same block, and the bars say which step
    they came from (``BlockBars.step``) so a fallback is visible.
    """

    def __init__(self, *, highnorm_ratio: float = 3.0, alignment_quantile: float = 0.999,
                 frozen: Sequence[int] = ()):
        self.highnorm_ratio = float(highnorm_ratio)
        self.alignment_quantile = float(alignment_quantile)
        self.frozen = tuple(sorted(int(t) for t in frozen))
        self._bars: Dict[Tuple[int, int], BlockBars] = {}

    def add(self, trace, *, steps: Sequence[int], layers: Sequence[int]) -> int:
        frozen = set(self.frozen)
        added = 0
        for step in steps:
            for layer in layers:
                obs = trace.at(int(step), int(layer))
                if obs is None or obs.norm is None or obs.projection is None:
                    continue
                norm, projection = obs.norm.float(), obs.projection.float()
                cosine = (obs.cosine.float() if obs.cosine is not None
                          else projection / norm.clamp_min(1e-9))
                n = int(norm.shape[0])
                median = float(norm.median().clamp_min(1e-9))
                norm_bar = self.highnorm_ratio * median
                ordinary = norm < norm_bar
                ids = [t for t in frozen if t < n]
                if ids:
                    ordinary[ids] = False
                if int(ordinary.sum()) < 2:
                    ordinary = torch.ones(n, dtype=torch.bool)
                bar = float(torch.quantile(cosine[ordinary], self.alignment_quantile))
                ceiling = float(cosine[ordinary].max())
                above = cosine >= bar
                index = torch.as_tensor(ids, dtype=torch.long)
                sinks = share = None
                if obs.incoming is not None:
                    incoming = obs.incoming.float()
                    sinks = tuple(int(t) for t in incoming.argmax(-1).tolist())
                    share = tuple(float(x) for x in incoming.max(-1).values.tolist())
                self._bars[(int(step), int(layer))] = BlockBars(
                    step=int(step), layer=int(layer), median_norm=median,
                    norm_bar=norm_bar, alignment_bar=bar,
                    projection_ceiling=float(projection[ordinary].max()),
                    above_bar=tuple(int(t) for t in torch.nonzero(above).flatten().tolist()),
                    ordinary_max_cosine=ceiling,
                    register_like=tuple(int(t) for t in torch.nonzero(
                        cosine > ceiling).flatten().tolist()),
                    carriers=tuple(int(t) for t in torch.nonzero(
                        above & (norm >= norm_bar)).flatten().tolist()),
                    carrier_projection=(float(projection[index].mean()) if ids
                                        else float("nan")),
                    carrier_cosine=float(cosine[index].mean()) if ids else float("nan"),
                    carrier_norm=float(norm[index].mean()) if ids else float("nan"),
                    head_sinks=sinks, head_share=share)
                added += 1
        return added

    def at(self, step: int, layer: int) -> Optional[BlockBars]:
        exact = self._bars.get((int(step), int(layer)))
        if exact is not None:
            return exact
        candidates = [s for (s, l) in self._bars if l == int(layer)]
        if not candidates:
            return None
        nearest = min(candidates, key=lambda s: (abs(s - int(step)), s))
        return self._bars[(nearest, int(layer))]

    def for_input_of(self, step: int, layer: int) -> Optional[BlockBars]:
        return self.at(step, int(layer) - 1) if int(layer) >= 1 else None

    def steps(self) -> List[int]:
        return sorted({s for s, _ in self._bars})

    def layers(self) -> List[int]:
        return sorted({l for _, l in self._bars})

    def rows(self) -> List[Dict[str, Any]]:
        return [b.row() for _, b in sorted(self._bars.items())]

    def __len__(self) -> int:
        return len(self._bars)


@dataclass(frozen=True)
class AlignmentRule:
    r"""Remove v* from ANY token more v*-aligned than the clean run's ordinary tokens HERE.

    Evaluated at every hook, over **every image token**, against the bar of that exact
    (denoising step, block) in the paired clean run. Three properties the conjunction
    rule (:class:`RegisterStateRule`) did not have:

    - **no norm bar.** A token being written into at the writer stage, or a new carrier
      that has not yet grown large, is aligned before it is high-norm; after the adaptive
      norm divides magnitude out, what a block's attention reads is that alignment. A norm
      bar lets exactly those escape.
    - **a fixed, clean-referenced bar.** A percentile of the edited tensor always selects
      its share of tokens, register or not; a bar read from the clean run selects only
      what is more aligned than any ordinary token there -- ~0.1% of ordinary tokens at
      the default quantile, plus the carriers. 4.4 audits that count on the clean run.
    - **it follows the state wherever it goes.** Q1 found that removing the register at
      its original positions is followed, in most runs, by another token becoming the
      carrier. A frozen mask would leave that new carrier untouched.
    """

    reference: CleanStateReference
    needs_context: ClassVar[bool] = True

    @property
    def highnorm_ratio(self) -> float:
        return self.reference.highnorm_ratio

    def bars(self, step: int, layer: int) -> Optional[BlockBars]:
        """The bar a hook at block ``layer``'s INPUT enforces."""
        return self.reference.for_input_of(step, layer)

    def select_from_stats(self, norms: torch.Tensor, projections: torch.Tensor, *,
                          step: int, layer: int) -> List[int]:
        bars = self.bars(step, layer)
        if bars is None:
            return []
        cosine = projections.float() / norms.float().clamp_min(1e-9)
        return [int(t) for t in torch.nonzero(cosine >= bars.alignment_bar).flatten().tolist()]

    def select(self, states: torch.Tensor, direction: torch.Tensor, *, step: int,
               layer: int) -> List[int]:
        row = _conditional_row(states).float()
        v = _unit(direction).to(row.device)
        return self.select_from_stats(row.norm(dim=-1), row @ v, step=step, layer=layer)

    def row(self) -> Dict[str, Any]:
        return dict(rule="clean_alignment_ceiling",
                    alignment_quantile=self.reference.alignment_quantile,
                    highnorm_ratio=self.reference.highnorm_ratio,
                    note="cos(x, v*) >= the clean ordinary quantile at this (step, block); "
                         "every image token; no norm bar")


# ---------------------------------------------------------------- (a) the received state
def received_state_rows(trace, sites: Sequence[SuppressionSite], reference: CleanStateReference,
                        *, step: int, layers: Sequence[int],
                        carriers: Sequence[int]) -> List[Dict[str, Any]]:
    r"""(a) The v* state each block RECEIVED at its input -- the suppression test itself.

    Direct residual-state measurement, at every block in ``layers``:

    - at a HOOKED block, the suppressor's own measurement of the tensor it handed on
      (:class:`SuppressionSite` ``received_*``), taken after the edit;
    - at an UNHOOKED block -- the one after the terminal cleanup, say -- the previous
      block's output, which is exactly what it receives.

    ``received_n_register_like`` counts tokens more v*-aligned than EVERY ordinary token
    of the paired clean run at that input. Zero at every consuming block is what "the
    natural v* state was suppressed" means here. ``received_max_cosine_vs_natural`` is
    how close the most aligned token a block received came to the natural register, as
    a share of the clean carriers' alignment there -- the continuous version of the same
    question. Neither says anything about attention sinks, which are (c): a non-register
    sink that remains does not put a v* state back into the stream.

    ``register_like_before_hook`` is what the previous block HANDED ON -- regrowth, which
    the hook then removed. It is (b)'s business and reported here only for context.
    ``received_n_above_bar`` counts against the (lower) removal bar, for reference.
    """
    by_layer = {int(s.layer): s for s in sites if int(s.step) == int(step)}
    frozen = [int(t) for t in carriers]
    rows: List[Dict[str, Any]] = []
    for layer in layers:
        layer = int(layer)
        bars = reference.for_input_of(step, layer)
        pre = trace.at(int(step), layer - 1) if layer >= 1 else None
        site = by_layer.get(layer)
        row: Dict[str, Any] = dict(step=int(step), layer=layer, hooked=site is not None)
        if bars is None or pre is None or pre.norm is None or pre.projection is None:
            row.update(received="not measured",
                       reason="no clean bar or no recorded input for this block")
            rows.append(row)
            continue
        norm, projection = pre.norm.float(), pre.projection.float()
        cosine = projection / norm.clamp_min(1e-9)
        n = int(norm.shape[0])
        ids = [t for t in frozen if t < n]
        ceiling = bars.ordinary_max_cosine
        row.update(bar=bars.alignment_bar, ordinary_max_cosine=ceiling, bars_from_step=bars.step,
                   register_like_before_hook=int((cosine > ceiling).sum()),
                   clean_register_like=len(bars.register_like),
                   clean_carrier_cosine=bars.carrier_cosine)
        if site is not None and site.received_n_register_like is not None:
            row.update(n_removed=len(site.chosen_ids),
                       received_max_cosine=site.received_max_cosine,
                       received_n_register_like=int(site.received_n_register_like),
                       received_n_above_bar=site.received_n_above_bar,
                       received_carrier_cosine=site.received_carrier_cosine,
                       share_of_image_edited=site.share_of_image,
                       measured="directly, on the tensor the hook handed on")
        else:
            row.update(n_removed=0 if site is None else len(site.chosen_ids),
                       received_max_cosine=float(cosine.max()),
                       received_n_register_like=int((cosine > ceiling).sum()),
                       received_n_above_bar=int((cosine >= bars.alignment_bar).sum()),
                       received_carrier_cosine=(float(cosine[ids].mean()) if ids
                                                else float("nan")),
                       measured="the previous block's output (no hook here)")
        natural = bars.carrier_cosine
        row["received_max_cosine_vs_natural"] = (
            row["received_max_cosine"] / natural
            if natural == natural and natural > 0 else float("nan"))
        rows.append(row)
    return rows


def received_state_summary(rows: Sequence[Dict[str, Any]], *, interval: Window,
                           terminal: Optional[int]) -> Dict[str, Any]:
    """(a), summarised for one step: did any consuming block receive a v*-aligned token?

    The interval, its terminal cleanup, and the first UNHOOKED block after it are read
    separately -- the last one is the only consuming site no hook protects.
    """
    measured = [r for r in rows if isinstance(r.get("received_n_register_like"), int)]
    inside = [r for r in measured if interval.first <= r["layer"] <= interval.last]
    term = [r for r in measured if terminal is not None and r["layer"] == terminal]
    after = [r for r in measured if terminal is not None and r["layer"] == terminal + 1]
    guarded = inside + term
    leaked = [r["layer"] for r in guarded if r["received_n_register_like"] > 0]
    boundary = after[0] if after else None
    boundary_leak = bool(boundary and boundary["received_n_register_like"] > 0)
    if not guarded:
        result = "not measured"
    elif leaked:
        result = "v* state received inside the interval"
    elif boundary_leak:
        # The interval and its cleanup held; the block after it received a state its
        # predecessor rebuilt from a cleaned input -- regrowth AFTER the interval.
        result = "v* state received after the interval"
    else:
        result = "no v*-aligned state received"
    return dict(
        result=result, blocks_measured=len(guarded),
        blocks_received=leaked,
        # How close any consuming block came to receiving the natural register: the most
        # aligned token it received, as a share of the clean carriers' alignment there.
        max_received_cosine_vs_natural=max(
            (r["received_max_cosine_vs_natural"] for r in guarded
             if r["received_max_cosine_vs_natural"] == r["received_max_cosine_vs_natural"]),
            default=float("nan")),
        carriers_received_cosine_max=max((r["received_carrier_cosine"] for r in guarded
                                          if r["received_carrier_cosine"] ==
                                          r["received_carrier_cosine"]),
                                         default=float("nan")),
        boundary_block=(terminal + 1) if terminal is not None else None,
        boundary_received=(boundary["received_n_register_like"] if boundary else None),
        boundary_clean_register_like=(boundary.get("clean_register_like") if boundary
                                      else None),
        max_share_of_image_edited=max((r.get("share_of_image_edited", 0.0) or 0.0
                                       for r in guarded), default=0.0),
        note=("no block from the start of the interval through its terminal cleanup -- nor "
              "the first unhooked block after it -- received a token more v*-aligned than "
              "every ordinary token of the paired clean run at that block"
              if result == "no v*-aligned state received" else
              f"a consuming block inside the interval received a v*-aligned token, at "
              f"blocks {leaked[:8]}"
              if result == "v* state received inside the interval" else
              "the interval and its cleanup held, but the first block after them received "
              "a v*-aligned token its predecessor rebuilt"
              if result == "v* state received after the interval" else
              "no block of the interval was measured"))


# ---------------------------------------------------------- (b) regrowth and relocation
def regrowth_relocation_rows(trace, reference: CleanStateReference, *, step: int,
                             layers: Sequence[int],
                             carriers: Sequence[int]) -> List[Dict[str, Any]]:
    r"""(b) At every block OUTPUT: did the ORIGINAL carriers come back, and did the state
    appear SOMEWHERE ELSE?

    Kept apart because Q1 found both after removal -- the original carriers regaining the
    register criterion in about a third of runs, another carrier winning in most of the
    rest -- and a single "regrowth" count cannot tell them apart. A block's output is what
    the next hook sees and removes, so this is what the model rebuilt, not what was
    consumed; (a) is the consumption test.

    "Aligned" means more v*-aligned than every ordinary token of the paired clean run at
    this (step, block). "New" means aligned here, not an original carrier, and not a
    register-like token of the clean run at this same position -- novelty relative to
    clean, which Q1 could not establish. The register criterion adds Q1's norm bar.
    """
    frozen = {int(t) for t in carriers}
    rows: List[Dict[str, Any]] = []
    for layer in layers:
        obs = trace.at(int(step), int(layer))
        bars = reference.at(step, int(layer))
        if obs is None or bars is None or obs.norm is None or obs.projection is None:
            continue
        norm, projection = obs.norm.float(), obs.projection.float()
        cosine = projection / norm.clamp_min(1e-9)
        n = int(norm.shape[0])
        like = cosine > bars.ordinary_max_cosine
        aligned = set(int(t) for t in torch.nonzero(like).flatten().tolist())
        criterion = set(int(t) for t in torch.nonzero(
            like & (norm >= bars.norm_bar)).flatten().tolist())
        clean_like = set(bars.register_like)
        new_aligned = sorted(aligned - frozen - clean_like)
        new_criterion = sorted(criterion - frozen - clean_like)
        ids = [t for t in sorted(frozen) if t < n]
        mean_projection = float(projection[ids].mean()) if ids else float("nan")
        rows.append(dict(
            step=int(step), layer=int(layer), n_original=len(ids),
            original_aligned=len(aligned & frozen),
            original_register_criterion=len(criterion & frozen),
            original_cosine=float(cosine[ids].mean()) if ids else float("nan"),
            clean_original_cosine=bars.carrier_cosine,
            original_projection_share=(mean_projection / bars.carrier_projection
                                       if bars.carrier_projection == bars.carrier_projection
                                       and bars.carrier_projection > 0 else float("nan")),
            new_aligned=len(new_aligned), new_register_criterion=len(new_criterion),
            new_aligned_ids=tuple(new_aligned), new_criterion_ids=tuple(new_criterion),
            max_new_cosine=(float(cosine[new_aligned].max()) if new_aligned
                            else float("nan"))))
    return rows


def regrowth_relocation_summary(rows: Sequence[Dict[str, Any]], *,
                                window: Window) -> Dict[str, Any]:
    """(b), summarised: original-carrier recovery and new carriers, kept separate."""
    inside = sorted((r for r in rows if window.first <= r["layer"] <= window.last),
                    key=lambda r: r["layer"])
    if not inside:
        return dict(regrowth="not measured", relocation="not measured")
    regained = [r["layer"] for r in inside if r["original_register_criterion"] > 0]
    shares = [r["original_projection_share"] for r in inside
              if r["original_projection_share"] == r["original_projection_share"]]
    per_position: Dict[int, int] = {}
    for r in inside:
        for t in r["new_aligned_ids"]:
            per_position[int(t)] = per_position.get(int(t), 0) + 1
    new_criterion = sorted({int(t) for r in inside for t in r["new_criterion_ids"]})
    blocks = sorted(per_position.values())
    first_new = next((r["layer"] for r in inside if r["new_register_criterion"] > 0), None)
    first_regained = regained[0] if regained else None
    # Q1's outcome vocabulary, with novelty now established against the paired clean run.
    if first_regained is not None and (first_new is None or first_regained <= first_new):
        outcome = "original carriers regain the register criterion"
    elif first_new is not None:
        outcome = "another carrier wins (new relative to the paired clean run)"
    else:
        outcome = "no token meets the register criterion"
    return dict(
        outcome=outcome,
        blocks_original_regained=regained,
        max_original_projection_share=max(shares) if shares else float("nan"),
        mean_original_projection_share=(sum(shares) / len(shares)) if shares else float("nan"),
        unique_new_aligned_positions=len(per_position),
        unique_new_register_positions=len(new_criterion),
        median_blocks_per_new_position=(blocks[len(blocks) // 2] if blocks else 0),
        first_new_register_block=first_new,
        blocks_measured=len(inside))


# --------------------------------------------- (c) original-sink persistence vs relocation
ATTENTION_CLASSES = ("kept", "other_original_carrier", "relocated_vstar_carrier",
                     "clean_register_elsewhere", "non_register_token", "spread_out")


def attention_relocation_rows(trace, reference: CleanStateReference, *, step: int,
                              layers: Sequence[int], carriers: Sequence[int],
                              diffuse_ratio: float = 0.5) -> List[Dict[str, Any]]:
    r"""(c) Per head: does its clean sink persist, and if not, where did its attention go?

    Q1's two panels, per (step, block): the share of heads whose strongest image key is
    still the clean one (all heads, and the AFFECTED heads whose clean sink was an
    original carrier), and the strongest incoming share (concentration) against clean.
    Every affected head is then classed by where its attention went:

    ``kept``                    the same token as in clean;
    ``other_original_carrier``  another token of the original carrier population;
    ``relocated_vstar_carrier`` a token that carried a NEW v*-aligned state into this
                                block (above every clean ordinary token at its input, and
                                not register-like there in clean) -- whether or not a hook
                                then stripped it, so sinkhood that survives the stripping
                                is visible;
    ``clean_register_elsewhere`` a token that is register-like at this input in the CLEAN
                                run but is not one of ``carriers`` -- the natural register at
                                a position the carrier set does not name (at a denoising step
                                other than the one the set was read at, say). Without this
                                class such a head was called ``non_register_token``;
    ``non_register_token``      anything else: a sink that is not a v* carrier;
    ``spread_out``              the head's strongest share fell below ``diffuse_ratio`` of
                                its clean value -- no single anchor took over.

    A non-register sink is a finding about routing. It does not put the v* state back
    into the stream and does not, by itself, contradict (a).
    """
    frozen = {int(t) for t in carriers}
    rows: List[Dict[str, Any]] = []
    for layer in layers:
        obs = trace.at(int(step), int(layer))
        bars = reference.at(step, int(layer))
        if (obs is None or obs.incoming is None or bars is None
                or bars.head_sinks is None or bars.head_share is None):
            continue
        incoming = obs.incoming.float()
        heads = int(incoming.shape[0])
        if heads != len(bars.head_sinks):
            continue
        top = incoming.argmax(-1).tolist()
        share = incoming.max(-1).values.tolist()
        relocated: set = set()
        pre = trace.at(int(step), int(layer) - 1) if int(layer) >= 1 else None
        in_bars = reference.for_input_of(step, int(layer))
        clean_register = (set(int(t) for t in in_bars.register_like) - frozen
                          if in_bars is not None else set())
        if pre is not None and in_bars is not None and pre.norm is not None \
                and pre.projection is not None:
            cos_in = pre.projection.float() / pre.norm.float().clamp_min(1e-9)
            relocated = (set(int(t) for t in torch.nonzero(
                cos_in > in_bars.ordinary_max_cosine).flatten().tolist())
                - frozen - set(in_bars.register_like))
        counts = {c: 0 for c in ATTENTION_CLASSES}
        kept_all = affected = 0
        for h in range(heads):
            clean_sink, clean_share = int(bars.head_sinks[h]), float(bars.head_share[h])
            token, value = int(top[h]), float(share[h])
            kept_all += int(token == clean_sink)
            if clean_sink not in frozen:
                continue
            affected += 1
            if token == clean_sink:
                counts["kept"] += 1
            elif clean_share > 0 and value < diffuse_ratio * clean_share:
                counts["spread_out"] += 1
            elif token in frozen:
                counts["other_original_carrier"] += 1
            elif token in relocated:
                counts["relocated_vstar_carrier"] += 1
            elif token in clean_register:
                counts["clean_register_elsewhere"] += 1
            else:
                counts["non_register_token"] += 1
        row = dict(step=int(step), layer=int(layer), n_heads=heads,
                   share_heads_kept_clean_sink=kept_all / max(heads, 1),
                   n_affected_heads=affected,
                   concentration=sum(share) / max(heads, 1),
                   clean_concentration=bars.concentration)
        row["concentration_vs_clean"] = (row["concentration"] / row["clean_concentration"]
                                         if row["clean_concentration"] > 0 else float("nan"))
        for c in ATTENTION_CLASSES:
            row[f"affected_{c}"] = (counts[c] / affected) if affected else float("nan")
        rows.append(row)
    return rows


def attention_relocation_summary(rows: Sequence[Dict[str, Any]], *,
                                 window: Window) -> Dict[str, Any]:
    """(c), summarised over a window: persistence of the original sinks vs redistribution."""
    inside = [r for r in rows if window.first <= r["layer"] <= window.last]
    if not inside:
        return dict(attention="not measured")
    with_affected = [r for r in inside if r["n_affected_heads"]]

    def mean(key, source):
        values = [float(r[key]) for r in source if r[key] == r[key]]
        return sum(values) / len(values) if values else float("nan")

    classes = {c: mean(f"affected_{c}", with_affected) for c in ATTENTION_CLASSES}
    moved = {c: v for c, v in classes.items() if c != "kept" and v == v}
    return dict(
        share_heads_kept_clean_sink=mean("share_heads_kept_clean_sink", inside),
        affected_heads_kept=classes["kept"],
        where_affected_attention_went=(max(moved, key=moved.get) if moved else None),
        **{f"affected_{c}": v for c, v in classes.items()},
        concentration_vs_clean=mean("concentration_vs_clean", inside),
        blocks_measured=len(inside), blocks_with_affected_heads=len(with_affected))


# ------------------------------------------------ (d) the induced state, at every step
def induction_written(records: Sequence[EditRecord], target, *, window: Window,
                      tolerance: float = 0.02) -> List[Dict[str, Any]]:
    """(d), per denoising step: was the intended state written at every hook of the window?

    ``target(record)`` returns ``(alpha_target, norm_target)`` for one write, ``norm_target``
    ``None`` when only the projection is controlled (the direction-only arm). One row per
    step: how many writes, how many were within ``tolerance`` on every controlled number,
    and the worst relative error -- so "established at every step" is read off the
    records rather than assumed from the schedule.
    """
    by_step: Dict[int, List[float]] = {}
    for record in records:
        if record.kind not in ("induce", "induce_matched"):
            continue
        if not (window.first <= int(record.layer) <= window.last):
            continue
        wanted = target(record)
        if wanted is None:
            continue
        alpha_t, norm_t = wanted
        error = abs(record.alpha_after - alpha_t) / max(abs(alpha_t), 1e-9)
        if norm_t is not None:
            error = max(error, abs(record.norm_after - norm_t) / max(abs(norm_t), 1e-9))
        by_step.setdefault(int(record.step), []).append(error)
    return [dict(step=s, writes=len(e), within=sum(1 for x in e if x <= tolerance),
                 worst_error=max(e), established=all(x <= tolerance for x in e))
            for s, e in sorted(by_step.items())]


# ====================================================== the depth schedule, as measured
def measured_formation_onset(baseline_rows: Sequence[Dict[str, Any]], *, natural: Window,
                             metric: str = "carrier_projection",
                             fraction: float = 0.5) -> Dict[str, Any]:
    r"""Where does the natural register actually FORM on this run?

    The first block whose OUTPUT carries at least ``fraction`` of the clean plateau -- the
    block whose feed-forward wrote it. Read from the clean trajectory rather than from the
    artifact's declared writer range, which is fitted over a prompt population.
    """
    series = sorted((int(r["layer"]), float(r.get(metric, float("nan"))))
                    for r in baseline_rows
                    if r.get(metric) is not None and int(r.get("layer", -1)) <= natural.last)
    series = [(l, v) for l, v in series if v == v]
    plateau = max((v for l, v in series if natural.first <= l <= natural.last),
                  default=float("nan"))
    if not (plateau == plateau) or plateau <= 0:
        return dict(onset=None, reason=f"no positive clean {metric} plateau to form towards")
    bar = float(fraction) * plateau
    onset = next((l for l, v in series if v >= bar), None)
    return dict(onset=onset, plateau_value=plateau, threshold=bar, fraction=float(fraction),
                metric=metric,
                note=(f"block {onset} is the first whose output carries the register "
                      f"({fraction:.0%} of the plateau): its feed-forward wrote it"
                      if onset is not None else "the register never forms"))


@dataclass(frozen=True)
class LifecycleSchedule:
    """Every window of the retiming design, derived from MEASURED boundaries."""

    early: Optional[Window]
    suppression: Window
    terminal: Optional[int]
    bridge: Optional[Window]
    extension: Optional[Window]
    late_only: Optional[Window]
    formation_onset: int
    natural_end: int
    dissolution_onset: Optional[int]
    notes: Tuple[str, ...] = ()

    def induction_windows(self) -> Dict[str, Window]:
        return {n: w for n, w in (("early", self.early), ("bridge", self.bridge),
                                  ("extension", self.extension), ("late", self.late_only))
                if w is not None}

    def rows(self) -> List[Dict[str, Any]]:
        out = []
        for name, window, used_by in (
                ("early", self.early, "B, E: induction"),
                ("suppression", self.suppression, "D, E, F, X: the SAME hooks"),
                ("bridge", self.bridge, "C: maintenance from before the decline"),
                ("extension", self.extension, "C: beyond the natural end"),
                ("late", self.late_only, "D: replacement after the suppressed interval")):
            if window is not None:
                out.append(dict(window=name, first=window.first, last=window.last,
                                blocks=window.length, used_by=used_by))
        if self.terminal is not None:
            out.append(dict(window="terminal cleanup", first=self.terminal,
                            last=self.terminal, blocks=1, used_by="D, E, F, X"))
        return out


def lifecycle_schedule(*, n_layers: int, formation_onset: int, natural_end: int,
                       dissolution_onset: Optional[int] = None,
                       early: Optional[Tuple[int, int]] = (14, 16),
                       window_length: int = 3, bridge_lead: int = 2,
                       extension_blocks: int = 3) -> LifecycleSchedule:
    r"""Place every window from the measured formation and dissolution boundaries.

    - **suppression** starts at the earlier of the measured formation onset and the
      block after the early window, and runs through the natural end; one terminal
      cleanup hook follows it. D, E and F share it exactly, so their images compare.
    - **early** (B, E) ends before the natural state forms.
    - **bridge** (C) starts ``bridge_lead`` blocks BEFORE the measured decline, while the
      naturally formed state is still at plateau, and runs through the natural end.
    - **extension** (C) continues into ``extension_blocks`` genuinely later blocks, past
      the measured natural end.
    - **late** (D) is the replacement, placed AFTER the suppressed interval AND its
      terminal cleanup, so no block carries both a suppression hook and an induction
      hook. C's extension and D's replacement are separate windows, each where its own
      logic puts it -- not forced into one three-block slot.
    """
    notes: List[str] = []
    early_w = Window("early", int(early[0]), int(early[1])) if early else None
    if early_w is not None and early_w.last >= int(formation_onset):
        raise ValueError(f"the early window {early_w.first}-{early_w.last} must end before "
                         f"the measured formation onset {formation_onset}")
    first = (min(early_w.last + 1, int(formation_onset)) if early_w is not None
             else int(formation_onset))
    if int(natural_end) < first:
        raise ValueError(f"the natural end {natural_end} precedes the suppression start {first}")
    suppression = Window("suppression", first, int(natural_end))
    terminal = int(natural_end) + 1 if int(natural_end) + 1 <= int(n_layers) - 1 else None
    if early_w is not None and first < int(formation_onset):
        notes.append(f"suppression starts at {first}, before the formation onset "
                     f"{formation_onset}, so an early-induced state is removed before any "
                     "natural-window block consumes it; D, E and F share this start")
    bridge = None
    if dissolution_onset is not None:
        start = max(int(formation_onset), int(dissolution_onset) - int(bridge_lead))
        if start <= int(natural_end):
            bridge = Window("bridge", start, int(natural_end))
    extension = None
    if int(natural_end) + 1 <= int(n_layers) - 1 and extension_blocks > 0:
        extension = Window("extension", int(natural_end) + 1,
                           min(int(n_layers) - 1, int(natural_end) + int(extension_blocks)))
    late_only = None
    if terminal is not None and terminal + 1 <= int(n_layers) - 1:
        late_only = Window("late", terminal + 1,
                           min(int(n_layers) - 1, terminal + int(window_length)))
        if late_only.length < int(window_length):
            notes.append(f"the late-only window was clipped to {late_only.length} block(s) "
                         "by the end of the stack")
    return LifecycleSchedule(early=early_w, suppression=suppression, terminal=terminal,
                             bridge=bridge, extension=extension, late_only=late_only,
                             formation_onset=int(formation_onset),
                             natural_end=int(natural_end),
                             dissolution_onset=(None if dissolution_onset is None
                                                else int(dissolution_onset)),
                             notes=tuple(notes))


def schedule_conflicts(induction_layers: Sequence[int],
                       suppression_hooks: Sequence[int]) -> List[int]:
    """Blocks where one condition would both suppress and induce -- they would fight."""
    return sorted(set(int(l) for l in induction_layers) & set(int(l) for l in suppression_hooks))


# ============================ the induction SITE: block input vs the feed-forward's input
def transplant_edit(tokens: Sequence[int], source_states, direction: torch.Tensor, *,
                    scale: str = "recipient", records: Optional[List[EditRecord]] = None):
    r"""Replace each token's vector at the hooked site with a SOURCE state (Q4's transfer).

    ``source_states`` is ``{token: [C]}`` or ``{step: {token: [C]}}``; a step or token
    without a source is left alone. ``scale='recipient'`` keeps the recipient's own norm,
    so only the direction transfers -- which is all that survives a LayerNorm without
    affine parameters, as at FLUX's and PixArt's feed-forward input, and makes the edit
    as small as the transfer allows. ``'none'`` copies the vector as is.

    At ``PRE_MLP_RESIDUAL`` on FLUX dual blocks (and PixArt) the hook sits on ``norm2``'s
    input: it changes what the feed-forward READS and nothing else -- the residual stream
    that continues past the block is untouched, so whatever changes downstream is what
    the feed-forward WROTE from that input. That is the writer computation Q4 found
    decisive at the writer block, and why this is a different intervention from writing
    a state into the block input.
    """
    per_step = any(isinstance(value, Mapping) for value in source_states.values())
    wanted = [int(t) for t in tokens]

    def edit(image: torch.Tensor, ctx) -> torch.Tensor:
        step = int(getattr(ctx, "step", -1))
        sources = source_states.get(step, {}) if per_step else source_states
        out = image.clone()
        before = _conditional_row(image).float()
        v = _unit(direction).to(before.device)
        rows = [out] if out.ndim == 2 else list(out.reshape(-1, *out.shape[-2:]))
        for token in wanted:
            source = sources.get(token)
            if source is None or token >= int(before.shape[0]):
                continue
            source = torch.as_tensor(source).float().to(before.device)
            if int(source.numel()) != int(before.shape[-1]):
                continue
            for row in rows:
                current = row[token].float()
                value = (source * (float(current.norm()) / max(float(source.norm()), 1e-12))
                         if scale == "recipient" else source)
                row[token] = value.to(row.dtype)
            if records is not None:
                after = _conditional_row(out)[token].float()
                x = before[token]
                records.append(EditRecord(
                    layer=int(getattr(ctx, "layer", -1)), step=step, kind="transplant",
                    token=token, alpha_before=float(x @ v), alpha_after=float(after @ v),
                    norm_before=float(x.norm()), norm_after=float(after.norm()),
                    cosine_after=float(after @ v) / max(float(after.norm()), 1e-12),
                    perturbation_l2=float((after - x).norm()),
                    crossed_highnorm_threshold=False))
        return out

    return edit


def align_edit(tokens: Sequence[int], cosine_target, direction: torch.Tensor, *,
               records: Optional[List[EditRecord]] = None):
    r"""Set each token's ``cos(x, v*)`` to a target, keeping its norm and its remainder.

    .. math:: x' = \|x\| \left(c\, \hat v + \sqrt{1 - c^2}\; \hat r\right)

    The v*-ONLY transfer at a site where magnitude does not survive (a LayerNorm input):
    the v* part of a source state, carried as alignment. ``cosine_target`` is a number or
    ``{token: c}`` (or ``{step: {token: c}}``).
    """
    per_step = isinstance(cosine_target, Mapping) and any(
        isinstance(value, Mapping) for value in cosine_target.values())
    wanted = [int(t) for t in tokens]

    def edit(image: torch.Tensor, ctx) -> torch.Tensor:
        step = int(getattr(ctx, "step", -1))
        targets = (cosine_target.get(step, {}) if per_step else cosine_target)
        v = _unit(direction).to(image.device)
        before = _conditional_row(image).float()
        out = image.clone()
        rows = [out] if out.ndim == 2 else list(out.reshape(-1, *out.shape[-2:]))
        for token in wanted:
            c = (targets.get(token) if isinstance(targets, Mapping) else targets)
            if c is None or token >= int(before.shape[0]):
                continue
            c = max(-1.0, min(1.0, float(c)))
            for row in rows:
                current = row[token].float()
                residual = current - float(current @ v) * v
                length = float(residual.norm())
                if length <= 1e-6 * max(float(current.norm()), 1e-12):
                    continue
                new = float(current.norm()) * (c * v + (1 - c * c) ** 0.5 * residual / length)
                row[token] = new.to(row.dtype)
            if records is not None:
                after = _conditional_row(out)[token].float()
                x = before[token]
                records.append(EditRecord(
                    layer=int(getattr(ctx, "layer", -1)), step=step, kind="align",
                    token=token, alpha_before=float(x @ v), alpha_after=float(after @ v),
                    norm_before=float(x.norm()), norm_after=float(after.norm()),
                    cosine_after=float(after @ v) / max(float(after.norm()), 1e-12),
                    perturbation_l2=float((after - x).norm()),
                    crossed_highnorm_threshold=False))
        return out

    return edit


def site_pilot_rows(trace, clean, *, step: int, layers: Sequence[int],
                    carriers: Sequence[int], sink_threshold: float,
                    highnorm_ratio: float = 3.0, arm: str = "") -> List[Dict[str, Any]]:
    """The post-block state and sink behaviour of one site-pilot arm, block by block.

    ``write_along_vstar`` is how much the block itself added to the carriers' v*
    projection (output minus input), against clean -- the quantity that says whether a
    block's feed-forward WROTE a register-like state, which a block-input edit and a
    feed-forward-input edit produce by different routes.
    """
    rows = []
    treated = {int(r["layer"]): r for r in achieved_lifecycle(
        trace, step=step, layers=[l for l in layers] + [min(layers) - 1],
        carriers=carriers, sink_threshold=sink_threshold, highnorm_ratio=highnorm_ratio)}
    base = {int(r["layer"]): r for r in achieved_lifecycle(
        clean, step=step, layers=[l for l in layers] + [min(layers) - 1],
        carriers=carriers, sink_threshold=sink_threshold, highnorm_ratio=highnorm_ratio)}
    for layer in sorted(layers):
        here, before = treated.get(layer), treated.get(layer - 1)
        c_here, c_before = base.get(layer), base.get(layer - 1)
        if here is None or c_here is None:
            continue
        write = (here.get("carrier_projection", float("nan"))
                 - (before or {}).get("carrier_projection", float("nan")))
        clean_write = (c_here.get("carrier_projection", float("nan"))
                       - (c_before or {}).get("carrier_projection", float("nan")))
        rows.append(dict(
            arm=arm, layer=layer,
            carrier_projection=here.get("carrier_projection"),
            carrier_cosine=here.get("carrier_cosine"),
            carrier_norm_ratio=here.get("carrier_norm_ratio"),
            carrier_sink_strength=here.get("carrier_sink_strength"),
            carrier_is_sink=here.get("carrier_is_sink"),
            write_along_vstar=write, clean_write_along_vstar=clean_write,
            clean_carrier_projection=c_here.get("carrier_projection"),
            clean_carrier_sink_strength=c_here.get("carrier_sink_strength")))
    return rows


# ================================ the natural interval by the register test, and the
# range check that backs the three thresholds every table here depends on
def register_test_counts(reference: CleanStateReference, *, step: int,
                         layers: Sequence[int],
                         carriers: Optional[Sequence[int]] = None) -> Dict[int, int]:
    r"""Per block OUTPUT: how many tokens pass the register test in the clean run.

    The register test is Q1's criterion, the one every table here uses: norm at least
    ``highnorm_ratio`` x the block's median AND alignment with v* above the
    ``alignment_quantile`` of that block's ordinary tokens. With ``carriers`` only those
    positions are counted; without them every token is, a count that needs no carrier
    set and so can place formation before any carrier has been chosen.

    A block recorded only at another step is left out rather than filled in from the
    nearest step: a boundary read off a different noise level is a different boundary.
    """
    wanted = None if carriers is None else {int(t) for t in carriers}
    counts: Dict[int, int] = {}
    for layer in layers:
        bars = reference.at(int(step), int(layer))
        if bars is None or int(bars.step) != int(step):
            continue
        counts[int(layer)] = (len(bars.carriers) if wanted is None
                              else sum(1 for t in bars.carriers if int(t) in wanted))
    return counts


def measured_formation_by_count(counts: Mapping[int, int], *,
                                fraction: float = 0.5) -> Dict[str, Any]:
    r"""The first block whose output holds ``fraction`` of the peak register population.

    Half-maximum is the usual onset convention for a rise. On FLUX.1-dev the population
    jumps from a handful of tokens to its plateau in one block (the writer), so every
    fraction between roughly 0.1 and 0.9 names the same block -- which the range check
    (:func:`threshold_sensitivity`) reports rather than asserts.
    """
    series = sorted((int(l), int(c)) for l, c in counts.items())
    if not series or max(c for _, c in series) <= 0:
        return dict(onset=None, peak_layer=None, peak_count=0, fraction=float(fraction),
                    note="no block holds a token that passes the register test")
    peak_count = max(c for _, c in series)
    peak_layer = next(l for l, c in series if c == peak_count)
    bar = float(fraction) * peak_count
    onset = next(l for l, c in series if c >= bar)
    return dict(onset=onset, peak_layer=peak_layer, peak_count=peak_count,
                fraction=float(fraction), threshold=bar,
                note=f"block {onset} is the first whose output holds {fraction:.0%} of the "
                     f"peak register population ({peak_count} tokens at block {peak_layer})")


def measured_natural_end(counts: Mapping[int, int], *, after: int) -> Dict[str, Any]:
    r"""The last block of the natural interval, by the register test.

    The block before the first block after ``after`` (the plateau) at which no carrier
    passes the register test. It replaces "the carriers' v* projection fell below 10% of
    its peak", which never fires on FLUX.1-dev: after the register dissolves, its former
    carriers keep cos(v*) near 0.5, so their projection bottoms out around 15% of the peak
    however far down the stack it is read. A run on which that rule never fired fell back
    to a declared range without saying so; this one either measures the end or returns
    ``None`` with the reason.
    """
    later = sorted(int(l) for l in counts if int(l) > int(after))
    if not later:
        return dict(natural_end=None, first_empty_block=None, rule="register_test",
                    note=f"no block after {after} was measured")
    gone = next((l for l in later if int(counts[l]) == 0), None)
    if gone is None:
        return dict(natural_end=None, first_empty_block=None, rule="register_test",
                    last_block_measured=later[-1],
                    note=f"a carrier still passes the register test at block {later[-1]}, "
                         "the last one measured: the register never ends on this run")
    return dict(natural_end=gone - 1, first_empty_block=gone, rule="register_test",
                last_block_measured=later[-1],
                note=f"block {gone} is the first after {after} at which no carrier passes "
                     f"the register test, so the natural interval ends at {gone - 1}")


def threshold_sensitivity(trace, *, step: int, layers: Sequence[int],
                          carriers: Sequence[int],
                          highnorm_ratio: float = 3.0, alignment_quantile: float = 0.999,
                          sink_threshold: float = 10.0,
                          highnorm_ratios: Sequence[float] = (1.5, 2.0, 2.5, 3.0, 4.0, 5.0,
                                                              6.0, 8.0),
                          alignment_quantiles: Sequence[float] = (0.99, 0.995, 0.999,
                                                                  0.9995, 0.9999),
                          sink_thresholds: Sequence[float] = (2.0, 5.0, 10.0, 20.0, 40.0),
                          interval: Optional[Tuple[int, int]] = None
                          ) -> List[Dict[str, Any]]:
    r"""Does any conclusion depend on the exact value of the three shared thresholds?

    Each threshold is moved over a range with the other two held at their chosen values,
    and the quantities it decides are re-measured on the SAME clean run -- no new
    generation. A value that sits on a plateau of these curves is defensible without an
    argument about why that value: the answer does not change around it.

    - ``highnorm_ratio`` decides the carrier set and, through it, the natural interval:
      the carrier count at the peak block, its overlap with the chosen carriers, and the
      measured formation block and natural end.
    - ``alignment_quantile`` decides what the removal rule strips on the clean run: the
      largest share of the image stripped at any block of ``interval``, and the fewest of
      the chosen carriers above the bar there (the rule must still catch the register).
    - ``sink_threshold`` decides which tokens count as sinks: over ``interval``, the mean
      number of sink tokens per block and the share of them that are carriers.

    ``interval`` defaults to the measured natural interval at the chosen values. One row
    per (parameter, value, statistic); ``chosen`` marks the value the run uses.
    """
    import numpy as np

    layers = sorted(int(l) for l in layers)
    chosen_set = {int(t) for t in carriers}
    rows: List[Dict[str, Any]] = []

    def emit(parameter, value, chosen, **stats):
        for name, number in stats.items():
            rows.append(dict(parameter=parameter, value=float(value), chosen=bool(chosen),
                             statistic=name,
                             result=float(number) if number is not None else float("nan")))

    def reference_for(ratio, quantile, frozen):
        ref = CleanStateReference(highnorm_ratio=ratio, alignment_quantile=quantile,
                                  frozen=frozen)
        ref.add(trace, steps=[int(step)], layers=layers)
        return ref

    # The interval the other two checks read, at the chosen values.
    base = reference_for(highnorm_ratio, alignment_quantile, sorted(chosen_set))
    if interval is None:
        population = register_test_counts(base, step=step, layers=layers)
        formation = measured_formation_by_count(population)
        end = (measured_natural_end(register_test_counts(base, step=step, layers=layers,
                                                         carriers=sorted(chosen_set)),
                                    after=formation["peak_layer"])
               if formation["onset"] is not None else dict(natural_end=None))
        if formation["onset"] is not None:
            interval = (int(formation["onset"]),
                        int(end["natural_end"] if end["natural_end"] is not None
                            else layers[-1]))
    inside = ([l for l in layers if interval[0] <= l <= interval[1]]
              if interval is not None else [])

    # ---- high-norm ratio: the carrier set and the natural interval it implies
    for ratio in highnorm_ratios:
        ref = reference_for(float(ratio), alignment_quantile, ())
        population = register_test_counts(ref, step=step, layers=layers)
        formation = measured_formation_by_count(population)
        peak = formation["peak_layer"]
        selected = (set(ref.at(step, peak).carriers) if peak is not None else set())
        end = (measured_natural_end(register_test_counts(ref, step=step, layers=layers,
                                                         carriers=sorted(selected)),
                                    after=peak)
               if peak is not None else dict(natural_end=None))
        union = selected | chosen_set
        emit("highnorm_ratio", ratio, abs(float(ratio) - float(highnorm_ratio)) < 1e-9,
             carriers_at_peak=len(selected),
             overlap_with_chosen_carriers=(len(selected & chosen_set) / len(union)
                                           if union else float("nan")),
             formation_block=formation["onset"],
             natural_end_block=end.get("natural_end"))

    # ---- alignment quantile: what the removal rule strips on the clean run
    for quantile in alignment_quantiles:
        ref = reference_for(highnorm_ratio, float(quantile), sorted(chosen_set))
        shares, caught = [], []
        for layer in inside:
            bars = ref.at(step, layer)
            if bars is None or int(bars.step) != int(step):
                continue
            n = int(trace.at(step, layer).norm.shape[0])
            shares.append(len(bars.above_bar) / max(n, 1))
            if chosen_set:
                caught.append(len(chosen_set & set(bars.above_bar)) / len(chosen_set))
        emit("alignment_quantile", quantile,
             abs(float(quantile) - float(alignment_quantile)) < 1e-12,
             max_share_of_image_stripped=max(shares) if shares else None,
             mean_tokens_stripped_per_block=(float(np.mean(shares)) * int(
                 trace.at(step, inside[0]).norm.shape[0]) if shares else None),
             min_share_of_carriers_caught=min(caught) if caught else None)

    # ---- sink threshold: which tokens count as sinks inside the natural interval
    from .control_surface import sink_readout
    for threshold in sink_thresholds:
        counts, carrier_share = [], []
        for layer in inside:
            obs = trace.at(step, layer)
            if obs is None or getattr(obs, "incoming", None) is None:
                continue
            n = int(obs.incoming.shape[-1])
            sinks = sink_readout(obs, n_tokens=n, threshold=float(threshold))["is_sink"]
            ids = set(torch.nonzero(sinks).flatten().tolist())
            counts.append(len(ids))
            if ids:
                carrier_share.append(len(ids & chosen_set) / len(ids))
        emit("sink_threshold", threshold,
             abs(float(threshold) - float(sink_threshold)) < 1e-9,
             mean_sink_tokens_per_block=float(np.mean(counts)) if counts else None,
             share_of_sinks_that_are_carriers=(float(np.mean(carrier_share))
                                               if carrier_share else None))
    for row in rows:
        row["step"] = int(step)
        row["interval_first"] = interval[0] if interval is not None else None
        row["interval_last"] = interval[1] if interval is not None else None
    return rows
