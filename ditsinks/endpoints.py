"""Turn paired clean/treated traces into the predeclared endpoints these experiments use.

Two endpoints are primary and everything else is a mechanistic diagnostic:

``head_retention``     the fraction of *affected* heads whose strongest image key
                       is still the token it was in the clean run.  Affected
                       means the head's clean sink was one of the frozen target
                       tokens; a head that never looked at a register cannot
                       report anything about removing one.
``vstar_projection_change``  the change in projection of the target tokens on the
                       frozen direction, in units of the clean ordinary spread,
                       so it is comparable across layers and checkpoints.

Everything here reads frozen target identities.  No endpoint is computed from a
set re-selected out of a treated activation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .causal_engine import FrozenTargets, LayerObservation, Trace
from .causal_records import LayerHeadMeasurement


@dataclass(frozen=True)
class LayerReference:
    """What "large" and "aligned" mean at one layer, read from the clean run."""

    layer: int
    spread: float                 # ordinary-token projection spread, the unit of an effect
    median_norm: float
    norm_threshold: float         # the bar a token clears to count as high-norm here
    alignment_threshold: float    # the bar it clears to count as register-aligned here


@dataclass(frozen=True)
class CleanReference:
    """Per-layer bars for a whole trajectory.

    Freezing one bar at the intervention layer and applying it thirty layers later
    compares a token against a population it is no longer part of: residual norms
    grow with depth and register alignment decays, so a stale bar makes a late
    effect look enormous and a late recovery look trivial.  Every bar here comes
    from the clean run of the *same* generation, at the *same* layer.
    """

    layers: Dict[int, LayerReference] = field(default_factory=dict)
    fallback: Optional[LayerReference] = None

    def at(self, layer: int) -> Optional[LayerReference]:
        return self.layers.get(int(layer), self.fallback)


def build_clean_reference(clean: Trace, targets: FrozenTargets, *, layers: Sequence[int],
                          step: int, highnorm_ratio: float = 3.0,
                          alignment_quantile: float = 0.999) -> CleanReference:
    """Read the high-norm bar, the alignment bar and the spread at every layer.

    The alignment bar is a null: what the *ordinary* tokens of the clean run reach
    at this layer.  A treated token clears it only by being more register-aligned
    than any ordinary token ever gets there, which is the claim the endpoint is
    meant to test.
    """
    exclude = set(int(t) for t in targets.register_ids) | set(int(t) for t in targets.topk_ids)
    built: Dict[int, LayerReference] = {}
    for layer in sorted({int(l) for l in layers}):
        observation = clean.at(step, layer)
        if observation is None or observation.norm is None:
            continue
        count = int(observation.norm.numel())
        ordinary = [t for t in range(count) if t not in exclude] or list(range(count))
        median = float(observation.norm.median())
        spread = 1.0
        if observation.projection is not None and len(ordinary) > 1:
            spread = float(max(observation.projection[ordinary].float().std(), 1e-6))
        alignment = 0.0
        if observation.cosine is not None and ordinary:
            values = observation.cosine.float().abs()[ordinary]
            alignment = float(max(torch.quantile(values, alignment_quantile), 1e-3))
        built[layer] = LayerReference(layer=layer, spread=spread, median_norm=median,
                                      norm_threshold=float(highnorm_ratio) * median,
                                      alignment_threshold=alignment)
    fallback = LayerReference(layer=-1, spread=1.0, median_norm=targets.median_norm,
                              norm_threshold=targets.norm_threshold,
                              alignment_threshold=targets.alignment_threshold)
    return CleanReference(layers=built, fallback=fallback)


def _grid_distance(a: Optional[int], b: Optional[int], grid: Tuple[int, int]) -> Optional[float]:
    """Euclidean distance in patch units, which is what "the sink moved" means."""
    if a is None or b is None:
        return None
    rows, cols = grid
    if rows <= 0 or cols <= 0:
        return float(abs(int(a) - int(b)))
    ar, ac = divmod(int(a), cols)
    br, bc = divmod(int(b), cols)
    return float(math.hypot(ar - br, ac - bc))


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if len(values) else float("nan")


def key_rank(qk_cosine: torch.Tensor, head: int, tokens: Sequence[int]) -> Optional[int]:
    """Best rank (1 = highest) held by any frozen target among the image keys."""
    if qk_cosine is None or not tokens:
        return None
    row = qk_cosine[head]
    order = torch.argsort(row, descending=True).tolist()
    positions = [order.index(int(t)) + 1 for t in tokens if int(t) < len(order)]
    return int(min(positions)) if positions else None


def query_key_advantage(qk_cosine: torch.Tensor, head: int, tokens: Sequence[int]) -> Optional[float]:
    """How much better a target key aligns with the mean image query than average."""
    if qk_cosine is None or not tokens:
        return None
    row = qk_cosine[head].float()
    valid = [int(t) for t in tokens if int(t) < row.numel()]
    if not valid:
        return None
    return float(row[valid].max() - row.mean())


def measure_layer(clean: Optional[LayerObservation], treated: Optional[LayerObservation],
                  targets: FrozenTargets, *, layer: int, grid: Tuple[int, int],
                  token_ids: Sequence[int], channels: Sequence[int] = (),
                  reference: Optional[LayerReference] = None) -> List[LayerHeadMeasurement]:
    """All per-head rows plus one layer-level (``head == -1``) rollup."""
    if treated is None:
        return []
    ids = [int(t) for t in token_ids if int(t) < (treated.norm.numel() if treated.norm is not None else 0)]
    rows: List[LayerHeadMeasurement] = []
    clean_sinks = targets.clean_sink_by_head.get(int(layer), {})

    retained, affected, on_target, retained_all = [], 0, [], []
    if treated.incoming is not None:
        for head in range(int(treated.incoming.shape[0])):
            incoming = treated.incoming[head].float()
            sink = int(incoming.argmax())
            clean_sink = clean_sinks.get(head)
            was_affected = clean_sink is not None and clean_sink in set(ids)
            keeps = None if clean_sink is None else bool(sink == clean_sink)
            if was_affected:
                affected += 1
                retained.append(1.0 if keeps else 0.0)
            if keeps is not None:
                retained_all.append(1.0 if keeps else 0.0)
            on_target.append(1.0 if sink in set(ids) else 0.0)
            rows.append(LayerHeadMeasurement(
                layer=int(layer), head=int(head), sink_token_id=sink,
                clean_sink_token_id=clean_sink,
                head_was_affected=was_affected,
                new_sink_token_id=None if keeps in (None, True) else sink,
                sink_position=None if grid[1] <= 0 else (sink // grid[1], sink % grid[1]),
                original_sink_retained=keeps,
                sink_displacement=_grid_distance(sink, clean_sink, grid),
                sink_is_target=bool(sink in set(ids)),
                attention_concentration=float(incoming.max()),
                attention_entropy=None if treated.entropy is None else float(treated.entropy[head]),
                attention_to_targets=float(incoming[ids].sum()) if ids else None,
                key_rank=key_rank(treated.qk_cosine, head, ids),
                query_key_advantage=query_key_advantage(treated.qk_cosine, head, ids),
            ))

    layer_row = LayerHeadMeasurement(layer=int(layer), head=-1)
    if treated.projection is not None and ids:
        alpha = float(treated.projection[ids].mean())
        layer_row.alpha = alpha
        layer_row.cosine = float(treated.cosine[ids].mean()) if treated.cosine is not None else None
        norm = float(treated.norm[ids].pow(2).mean()) if treated.norm is not None else 0.0
        layer_row.perpendicular_norm = float(math.sqrt(max(norm - alpha ** 2, 0.0)))
        if clean is not None and clean.projection is not None:
            spread = reference.spread if reference and reference.spread > 0 else 1.0
            layer_row.vstar_projection_change = (alpha - float(clean.projection[ids].mean())) / spread
    if treated.cosine is not None:
        absolute = treated.cosine.float().abs()
        layer_row.max_cosine = float(absolute.max())
        layer_row.max_cosine_token = int(absolute.argmax())
    if treated.norm is not None:
        median = float(treated.norm.median())
        layer_row.target_norm_ratio = float(treated.norm[ids].mean() / max(median, 1e-9)) if ids else None
        norm_bar = reference.norm_threshold if reference else targets.norm_threshold
        alignment_bar = reference.alignment_threshold if reference else targets.alignment_threshold
        present = (treated.norm >= norm_bar)
        if treated.cosine is not None and alignment_bar > 0:
            present = present & (treated.cosine.float().abs() >= alignment_bar)
        layer_row.register_present = bool(present.any())
    if treated.channel_values is not None and len(channels) >= 1 and ids:
        values = treated.channel_values.float()
        layer_row.dominant_channel_value = float(values[0][ids].abs().mean())
        if values.shape[0] > 1:
            layer_row.competitor_channel_value = float(values[1][ids].abs().mean())
            denominator = max(abs(layer_row.dominant_channel_value), 1e-9)
            layer_row.channel_takeover = layer_row.competitor_channel_value / denominator
    if treated.top_channel is not None and ids:
        modal = torch.mode(treated.top_channel[ids]).values
        layer_row.takeover_channel = int(modal)
    layer_row.head_retention = _mean(retained) if affected else None
    layer_row.affected_heads = int(affected)
    layer_row.sink_retention_all = _mean(retained_all) if retained_all else None
    layer_row.sink_on_target_fraction = _mean(on_target) if on_target else None
    if treated.incoming is not None:
        layer_row.attention_concentration = float(treated.incoming.float().max(dim=-1).values.mean())
        layer_row.attention_to_targets = float(treated.incoming.float()[:, ids].sum(-1).mean()) if ids else None
    if treated.entropy is not None:
        layer_row.attention_entropy = float(treated.entropy.float().mean())
    rows.append(layer_row)
    return rows


def ordinary_projection_spread(trace: Trace, step: int, layer: int,
                               targets: FrozenTargets) -> float:
    """Standard deviation of the ordinary tokens' projection: the natural unit.

    Reporting a projection change in raw units makes layers and checkpoints
    incomparable; dividing by the clean ordinary spread expresses it as "how
    many ordinary-token standard deviations the register moved".
    """
    observation = trace.at(step, layer)
    if observation is None or observation.projection is None:
        return 1.0
    exclude = set(targets.register_ids) | set(targets.topk_ids)
    keep = [t for t in range(int(observation.projection.numel())) if t not in exclude]
    if len(keep) < 2:
        return 1.0
    return float(max(observation.projection[keep].float().std(), 1e-6))


def measure_trace(clean: Trace, treated: Trace, targets: FrozenTargets, *,
                  layers: Sequence[int], step: int, token_ids: Optional[Sequence[int]] = None,
                  channels: Sequence[int] = (),
                  reference: Optional[CleanReference] = None) -> List[LayerHeadMeasurement]:
    """Endpoints for every observed layer of one treated trajectory.

    ``reference`` carries the clean run's per-layer bars.  A caller running many
    conditions against one clean pass should build it once and pass it in; when it
    is omitted it is derived here from ``clean``, which is equivalent but repeats
    the work for every condition.
    """
    ids = list(token_ids if token_ids is not None else targets.register_ids)
    reference = reference or build_clean_reference(clean, targets, layers=layers, step=step)
    out: List[LayerHeadMeasurement] = []
    for layer in layers:
        out.extend(measure_layer(clean.at(step, layer), treated.at(step, layer), targets,
                                 layer=int(layer), grid=treated.grid or clean.grid,
                                 token_ids=ids, channels=channels,
                                 reference=reference.at(int(layer))))
    return out


# ---------------------------------------------------------------- verdicts
RECOVERY_KINDS = ("same_position", "relocated", "none")
FATES = ("same_position", "relocated", "reserve_takeover", "diffuse", "none")


def recovery_outcome(treated: Trace, targets: FrozenTargets, *, layers: Sequence[int],
                     step: int, reference: Optional[CleanReference] = None
                     ) -> Tuple[Optional[int], Optional[int], str]:
    """First layer *after* the intervention at which a token is register-aligned again.

    The intervened layer is excluded here rather than by each caller: a register
    that the edit did not fully strip is still above the bar in the very block we
    edited, and counting that as regeneration would report maintenance the network
    never performed.
    """
    original = set(int(t) for t in targets.register_ids)
    for layer in sorted(int(l) for l in layers if int(l) > int(targets.layer)):
        observation = treated.at(step, layer)
        if observation is None or observation.cosine is None:
            continue
        bars = reference.at(layer) if reference else None
        norm_bar = bars.norm_threshold if bars else targets.norm_threshold
        alignment_bar = bars.alignment_threshold if bars else targets.alignment_threshold
        aligned = observation.cosine.float().abs()
        if observation.norm is not None:
            aligned = aligned * (observation.norm >= norm_bar).float()
        best = int(aligned.argmax())
        if float(aligned[best]) >= alignment_bar > 0:
            return layer, best, ("same_position" if best in original else "relocated")
    return None, None, "none"


def sink_fate(rows: Sequence[LayerHeadMeasurement], targets: FrozenTargets, *,
              clean_concentration: float, diffuse_ratio: float = 0.5,
              recovery_kind: str = "none") -> str:
    """One mutually exclusive label for what happened to the routing anchor.

    The order matters: regeneration at the original positions is checked before
    relocation, and both before the diffuse verdict, because a run that
    regenerates its register also concentrates attention again.
    """
    if recovery_kind == "same_position":
        return "same_position"
    if recovery_kind == "relocated":
        return "relocated"
    head_rows = [r for r in rows if r.head >= 0 and r.attention_concentration is not None]
    if not head_rows:
        return "none"
    concentration = _mean([r.attention_concentration for r in head_rows])
    if clean_concentration > 0 and concentration < diffuse_ratio * clean_concentration:
        return "diffuse"
    reserve = _mean([1.0 if (r.sink_is_target is False and r.original_sink_retained is False) else 0.0
                     for r in head_rows if r.original_sink_retained is not None])
    if reserve >= 0.5:
        return "reserve_takeover"
    return "none"


def register_lifetime(trace: Trace, targets: FrozenTargets, *, layers: Sequence[int],
                      step: int, reference: Optional[CleanReference] = None) -> Optional[int]:
    """Last observed layer at which the frozen registers are still registers.

    Lifetime is measured on the frozen token identities, not on whatever token
    happens to be loudest after treatment, so an intervention cannot extend the
    lifetime merely by promoting a different token.
    """
    last = None
    for layer in sorted(int(l) for l in layers):
        observation = trace.at(step, layer)
        if observation is None or observation.norm is None or observation.cosine is None:
            continue
        ids = [int(t) for t in targets.register_ids if int(t) < observation.norm.numel()]
        if not ids:
            continue
        bars = reference.at(layer) if reference else None
        norm_bar = bars.norm_threshold if bars else targets.norm_threshold
        alignment_bar = bars.alignment_threshold if bars else targets.alignment_threshold
        median = float(observation.norm.median())
        loud = float(observation.norm[ids].max()) >= norm_bar if median > 0 else False
        aligned = float(observation.cosine.float().abs()[ids].max()) >= alignment_bar
        if loud and aligned:
            last = layer
    return last


def register_lifetimes(trace: Trace, targets: FrozenTargets, *, layers: Sequence[int],
                       step: int, reference: CleanReference,
                       projection_threshold: Optional[float] = None) -> Dict[str, Optional[int]]:
    """Four independent lifecycle endpoints on the same frozen carrier IDs."""
    last = {"cosine_lifetime": None, "projection_lifetime": None,
            "high_norm_lifetime": None, "sink_lifetime": None}
    baseline = trace.at(step, int(targets.layer))
    ids0 = list(targets.register_ids)
    projection_bar = projection_threshold
    if projection_bar is None:
        projection_bar = (float(baseline.projection[ids0].abs().mean()) * 0.5
                          if baseline is not None and baseline.projection is not None and ids0 else None)
    for layer in sorted(int(value) for value in layers):
        observation = trace.at(step, layer)
        if observation is None:
            continue
        ids = [token for token in ids0 if observation.norm is None or token < observation.norm.numel()]
        if not ids:
            continue
        bars = reference.at(layer)
        if observation.cosine is not None and bars is not None and \
                float(observation.cosine[ids].abs().max()) >= bars.alignment_threshold:
            last["cosine_lifetime"] = layer
        if observation.projection is not None and projection_bar is not None and \
                float(observation.projection[ids].abs().max()) >= projection_bar:
            last["projection_lifetime"] = layer
        if observation.norm is not None and bars is not None and \
                float(observation.norm[ids].max()) >= bars.norm_threshold:
            last["high_norm_lifetime"] = layer
        if observation.incoming is not None:
            sinks = observation.incoming.argmax(dim=-1).tolist()
            if any(int(sink) in set(ids) for sink in sinks):
                last["sink_lifetime"] = layer
    return last


def alignment_half_life(trace: Trace, targets: FrozenTargets, *, layers: Sequence[int],
                        step: int) -> Optional[float]:
    """Depth at which the registers' alignment with v* has fallen by half.

    The threshold-based lifetime is the predeclared endpoint, but it is undefined
    whenever no layer clears the high-norm bar.  This continuous companion is
    always defined once the registers are aligned at all, is linearly interpolated
    between layers, and moves in the same direction, so a dissolution claim never
    rests on a threshold that happened not to be crossed.
    """
    ordered = sorted(int(l) for l in layers)
    series: List[Tuple[int, float]] = []
    for layer in ordered:
        observation = trace.at(step, layer)
        if observation is None or observation.cosine is None:
            continue
        ids = [int(t) for t in targets.register_ids if int(t) < observation.cosine.numel()]
        if ids:
            series.append((layer, float(observation.cosine.float().abs()[ids].mean())))
    if len(series) < 2 or series[0][1] <= 0:
        return None
    half = series[0][1] / 2.0
    for (previous_layer, previous_value), (layer, value) in zip(series, series[1:]):
        if value <= half:
            span = previous_value - value
            fraction = 0.0 if span <= 0 else (previous_value - half) / span
            return float(previous_layer + fraction * (layer - previous_layer))
    return float(series[-1][0])


def clean_head_concentration(clean: Trace, *, layers: Sequence[int], step: int) -> float:
    """Mean clean top-1 incoming attention share, the reference for "diffuse"."""
    values = []
    for layer in layers:
        observation = clean.at(step, int(layer))
        if observation is not None and observation.incoming is not None:
            values.append(float(observation.incoming.float().max(dim=-1).values.mean()))
    return _mean(values) if values else 0.0


def clean_capture_rate(clean: Trace, targets: FrozenTargets, *, layers: Sequence[int],
                       step: int) -> float:
    """Fraction of heads whose clean sink is a natural register, the reference rate for the residual-to-key transplant ladder."""
    ids = set(int(t) for t in targets.register_ids)
    values = []
    for layer in layers:
        observation = clean.at(step, int(layer))
        if observation is None or observation.incoming is None:
            continue
        sinks = observation.incoming.float().argmax(dim=-1).tolist()
        values.extend(1.0 if int(s) in ids else 0.0 for s in sinks)
    return _mean(values) if values else 0.0


def capture_rate(treated: Trace, recipient: int, *, layers: Sequence[int], step: int) -> float:
    """Fraction of heads whose strongest image key becomes the recipient token."""
    values = []
    for layer in layers:
        observation = treated.at(step, int(layer))
        if observation is None or observation.incoming is None:
            continue
        sinks = observation.incoming.float().argmax(dim=-1).tolist()
        values.extend(1.0 if int(s) == int(recipient) else 0.0 for s in sinks)
    return _mean(values) if values else 0.0


def token_capture_rate(trace: Trace, token: int, *, layer: int, step: int) -> float:
    """Top-1 capture for one exact token at one exact attention operation."""
    observation = trace.at(step, int(layer))
    if observation is None or observation.incoming is None:
        return float("nan")
    sinks = observation.incoming.float().argmax(dim=-1)
    return float((sinks == int(token)).float().mean())


def capture_profile(trace: Trace, recipient: int, *, intervention_layer: int,
                    layers: Sequence[int], step: int) -> List[Dict[str, object]]:
    """Keep same-operation capture separate from downstream persistence."""
    ordered = sorted({int(layer) for layer in layers if int(layer) >= int(intervention_layer)})
    last = ordered[-1] if ordered else int(intervention_layer)
    labels = {int(intervention_layer): "same_operation",
              int(intervention_layer) + 1: "plus_1_block",
              int(intervention_layer) + 2: "plus_2_blocks",
              int(intervention_layer) + 3: "plus_3_blocks",
              last: "end_of_zone"}
    return [dict(layer=layer, temporal_endpoint=labels.get(layer, f"plus_{layer-intervention_layer}_blocks"),
                 capture_rate=token_capture_rate(trace, recipient, layer=layer, step=step))
            for layer in ordered]
