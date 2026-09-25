"""Q11.1 -- lifecycle-aware register-preserving quantization.

The question this runner answers is narrow and stated in one line:

    At the same effective activation bit budget, does preserving the causally
    identified direction ``v*`` outperform preserving large activations?

and its lifecycle companion:

    Does protecting ``v*`` only through the layers where the register state exists
    match protecting it everywhere, while beating equally sized wrong windows?

Both are comparisons *at matched budget*, so the budget is computed and exported
alongside every result (:mod:`ditsinks.quantization`) rather than asserted in prose.

Scope, stated once so no downstream figure has to repeat it: this is QDQ simulation of
an activation allocation policy on the image-token residual stream.  Weights are not
quantized and no low-bit kernel is involved, so **end-to-end latency is not claimed**.

Two cost claims *are* supported and are computed in :mod:`ditsinks.cost_model`: the
activation footprint, which is exact arithmetic over the bit allocation, and the ceiling
on any speedup an activation policy could reach, which is Amdahl over memory traffic.  The
second is what keeps the first honest -- at batch one a diffusion transformer is heavily
weight-bound, so the memory result is real while the latency result would be near zero
however good the kernels were.
"""
from __future__ import annotations

import dataclasses
import json
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from . import endpoints as EP
from . import quantization as QZ
from .adapters import InterventionPoint
from .causal_engine import (EditPlan, Trace, run_traced_generation, select_frozen_targets)
from .questions import QuestionContext, QuestionResult, _resumable, _rows_to_frame


# ------------------------------------------------------------- lifecycle windows
def _clamp_window(start: int, width: int, n_layers: int) -> frozenset:
    start = max(0, min(int(start), max(n_layers - 1, 0)))
    stop = min(start + max(int(width), 1), n_layers)
    return frozenset(range(start, stop))


def lifecycle_windows(ctx: QuestionContext, *,
                      measured: Optional[Sequence[int]] = None) -> Dict[str, frozenset]:
    """The register window, and the equally sized wrong windows it must beat.

    ``preregistered`` is the discovery artifact's register range, chosen before any of
    these runs existed.  ``measured`` is the empirically observed lifetime when Q6 has
    supplied one; it is reported beside the preregistered window rather than replacing
    it, because a window fitted on the same runs it is evaluated on invites exactly the
    selection objection this comparison is meant to answer.

    The two wrong windows have the *same width* and sit before and after the register
    zone.  Width is what buys precision, so a wrong window of a different width would
    confound position with budget.
    """
    n_layers = int(ctx.driver.n_layers)
    register = sorted(int(l) for l in ctx.register_layers)
    width = max(len(register), 1)
    windows = {
        "all": frozenset(range(n_layers)),
        "preregistered": frozenset(register),
        "early": _clamp_window(register[0] - width, width, n_layers),
        "late": _clamp_window(register[-1] + 1, width, n_layers),
    }
    if measured:
        windows["measured"] = frozenset(int(l) for l in measured if 0 <= int(l) < n_layers)
    return windows


def measured_window_from_q6(q6: Optional[QuestionResult], ctx: QuestionContext) -> List[int]:
    """The empirically active register window implied by Q6's clean lifetimes.

    Q6 records, per unit, the last layer at which the frozen register carriers are still
    both high-norm and aligned.  The window runs from the first register layer to the
    median of that endpoint, which is a summary of where the state demonstrably exists
    rather than where it was predeclared to exist.  Returns ``[]`` when Q6 has not run,
    so the caller can say the measured window is unavailable instead of inventing one.
    """
    if q6 is None:
        return []
    lifetime = (q6.tables or {}).get("lifetime")
    if lifetime is None or lifetime.empty:
        return []
    for column in ("clean_high_norm_lifetime", "clean_lifetime", "clean_cosine_lifetime"):
        if column in lifetime.columns and lifetime[column].notna().any():
            last = int(np.nanmedian(pd.to_numeric(lifetime[column], errors="coerce").dropna()))
            start = int(ctx.register_layers[0])
            return list(range(start, max(last, start) + 1))
    return []


# -------------------------------------------------------- low-rank calibration
class _RangeSketch:
    """Randomized range finder for the dominant subspace of each layer's activations.

    The SVDQuant-style baseline needs a per-layer low-rank basis that absorbs outliers.
    Materialising every layer's activations, or even its Gram matrix, does not fit in a
    Colab session for a 57-block model at 3072 channels.  A random sketch ``Y = XᵀΩ``
    does: it costs ``C x (r+p)`` floats per layer, and orthonormalising ``Y`` recovers
    the dominant range to within the usual randomized-SVD guarantees, which is more than
    enough to define a protection subspace.
    """

    def __init__(self, rank: int, oversample: int = 4, seed: int = 0):
        self.rank = int(rank)
        self.oversample = int(oversample)
        self.seed = int(seed)
        self._sketch: Dict[int, torch.Tensor] = {}

    def observe(self, layer: int, states: torch.Tensor) -> None:
        x = states.detach().float().cpu()
        width = x.shape[-1]
        generator = torch.Generator().manual_seed(self.seed + 1000 * int(layer))
        omega = torch.randn(x.shape[0], self.rank + self.oversample, generator=generator)
        update = x.T @ omega                                  # [C, r+p]
        current = self._sketch.get(int(layer))
        self._sketch[int(layer)] = update if current is None else current + update

    def bases(self) -> Dict[int, torch.Tensor]:
        out: Dict[int, torch.Tensor] = {}
        for layer, sketch in self._sketch.items():
            q, _ = torch.linalg.qr(sketch)
            out[int(layer)] = q[:, :self.rank].T.contiguous()   # [r, C]
        return out


# ------------------------------------------------------------------- conditions
def build_schemes(ctx: QuestionContext, *, bits: int = 4,
                  windows: Optional[Dict[str, frozenset]] = None,
                  bases: Optional[Dict[int, torch.Tensor]] = None,
                  rank: int = 1, random_seed: int = 17,
                  include_lowrank: bool = True) -> List[QZ.QuantScheme]:
    """Every condition the comparison needs, at one matched budget.

    Each protected scheme keeps exactly ``rank`` high-precision scalars per token per
    active layer, so the bulk quantizer is handed the same problem in all of them and
    the only thing that varies is *which* information was spared.
    """
    windows = windows or lifecycle_windows(ctx)
    width = int(ctx.vstar.numel())
    generator = torch.Generator().manual_seed(random_seed)
    random_direction = torch.randn(width, generator=generator)

    schemes = [
        QZ.QuantScheme("clean", "Unquantized reference", bits=16, role="reference",
                       note="the paired baseline every fidelity metric is measured against"),
        QZ.QuantScheme(f"uniform_int{bits}", f"Uniform INT{bits}", bits=bits, role="reference",
                       note="no protection: the floor the protected schemes must beat"),
        QZ.QuantScheme("dominant_channel", "INT4 + dominant channel kept", bits=bits,
                       protect="coordinates", channels=(int(ctx.dominant_channel),),
                       role="condition",
                       note="protects one coordinate; converges on v* protection when v* is axis-aligned"),
        QZ.QuantScheme("vstar_all_layers", "INT4 + v* preserved, all layers", bits=bits,
                       protect="direction", direction=ctx.vstar, role="condition",
                       note="the geometry, everywhere"),
        QZ.QuantScheme("vstar_lifecycle", "INT4 + v* preserved, register window", bits=bits,
                       protect="direction", direction=ctx.vstar,
                       layers=windows["preregistered"], window_label="preregistered register zone",
                       role="condition", note="the lifecycle-aware condition"),
        QZ.QuantScheme("magnitude_topk", f"INT4 + {rank} largest coordinate(s) kept", bits=bits,
                       protect="largest", count=rank, role="control",
                       note="equal-budget magnitude-aware baseline; chooses per token, so it "
                            "also pays index bits"),
        QZ.QuantScheme("random_direction", "INT4 + random direction preserved", bits=bits,
                       protect="direction", direction=random_direction, role="control",
                       note="equal-budget, exactly matched cost: isolates the choice of direction"),
        QZ.QuantScheme("vstar_early_window", "INT4 + v* preserved, early window", bits=bits,
                       protect="direction", direction=ctx.vstar, layers=windows["early"],
                       window_label="wrong window (before the register zone)", role="control",
                       note="same width as the register zone, wrong position"),
        QZ.QuantScheme("vstar_late_window", "INT4 + v* preserved, late window", bits=bits,
                       protect="direction", direction=ctx.vstar, layers=windows["late"],
                       window_label="wrong window (after the register zone)", role="control",
                       note="same width as the register zone, wrong position"),
    ]
    if include_lowrank:
        # The basis is calibrated from the clean pass, which has not happened when a
        # notebook builds this list to show its budgets. The scheme is created anyway, with
        # an empty basis the runner fills in, so the prior-art control appears in the budget
        # table and in the fairness gate instead of silently never running.
        schemes.append(QZ.QuantScheme(
            "lowrank_absorption", f"INT4 + rank-{rank} absorption branch", bits=bits,
            protect="subspace", basis_by_layer=dict(bases or {}), count=rank, role="control",
            note="SVDQuant-inspired analogue: a calibrated low-rank branch absorbs the "
                 "outlier subspace. Not the published system; an equal-budget stand-in."))
    if "measured" in windows and windows["measured"]:
        schemes.append(QZ.QuantScheme(
            "vstar_measured_window", "INT4 + v* preserved, measured lifetime", bits=bits,
            protect="direction", direction=ctx.vstar, layers=windows["measured"],
            window_label="Q6-measured register lifetime", role="condition",
            note="empirical window; reported beside the preregistered one, not instead of it"))
    return schemes


# --------------------------------------------------------------------- metrics
def _image_array(image) -> Optional[np.ndarray]:
    if image is None:
        return None
    return np.asarray(image).astype(np.float32)


def _image_fidelity(clean_image, treated_image, lpips_net=None) -> Dict[str, float]:
    """Paired fidelity of a treated generation against its own clean generation.

    Paired, same seed, same prompt: the only difference is the precision policy, so a
    distance here is attributable to quantization rather than to sampling noise.
    """
    a, b = _image_array(clean_image), _image_array(treated_image)
    if a is None or b is None or a.shape != b.shape:
        return {"rmse": float("nan"), "psnr": float("nan"), "lpips": float("nan"),
                "max_abs_error": float("nan")}
    error = a - b
    rmse = float(np.sqrt(np.mean(error ** 2)))
    out = {"rmse": rmse,
           "psnr": float(20 * math.log10(255.0 / rmse)) if rmse > 0 else float("inf"),
           "max_abs_error": float(np.abs(error).max()),
           "lpips": float("nan")}
    if lpips_net is not None:
        with torch.no_grad():
            ta = torch.from_numpy(a).permute(2, 0, 1)[None] / 127.5 - 1
            tb = torch.from_numpy(b).permute(2, 0, 1)[None] / 127.5 - 1
            out["lpips"] = float(lpips_net(ta, tb))
    return out


def _load_lpips():
    """LPIPS when it is installed and its weights are reachable, else ``None``.

    A perceptual metric is worth having and is not worth failing a GPU run over, so this
    degrades to the pixel metrics and the caller reports which metrics exist.
    """
    try:
        import lpips

        return lpips.LPIPS(net="alex", verbose=False)
    except Exception:
        return None


def consumed_states(trace: Trace, step: int, layer: int,
                    point: InterventionPoint) -> Optional[torch.Tensor]:
    """The activation the block actually consumed, after any installed edit.

    A :class:`StateProbe` is registered before the edit installer, so on a treated run
    ``trace.probe`` returns the tensor *before* that layer's edit -- verified, not
    assumed.  The installer's own ``x_post_hook`` record is the tensor the module
    received, which is what a precision policy has to be judged on.  The clean run
    installs no edit, so it falls through to the probe.
    """
    edited = [value for key, value in trace.diagnostics.items()
              if key[0] == int(step) and key[1] == int(layer)
              and key[2] == point.value and key[3] == "x_post_hook"]
    return edited[0] if edited else trace.probe(step, layer, point)


def _mechanism_at_layer(ctx: QuestionContext, clean: Trace, treated: Trace,
                        targets, layer: int) -> Dict[str, float]:
    """Does the circuit survive the precision policy, not merely the picture?

    The application's argument is that generic outlier protection can keep the largest
    value while still rotating ``v*`` enough to reroute the sinks.  That is only testable
    if direction, magnitude, routing and key geometry are measured apart from one
    another, which is what these columns do.
    """
    out: Dict[str, float] = {}
    ids = list(targets.register_ids)
    unit = ctx.vstar.float() / ctx.vstar.float().norm().clamp_min(1e-12)

    clean_states = consumed_states(clean, ctx.step, layer, InterventionPoint.BLOCK_INPUT)
    treated_states = consumed_states(treated, ctx.step, layer, InterventionPoint.BLOCK_INPUT)
    if clean_states is not None and treated_states is not None and ids:
        keep = [t for t in ids if t < clean_states.shape[0] and t < treated_states.shape[0]]
        if keep:
            a, b = clean_states[keep].float(), treated_states[keep].float()
            alpha_clean, alpha_treated = a @ unit, b @ unit
            out["vstar_projection_error"] = float(
                ((alpha_treated - alpha_clean).abs()
                 / alpha_clean.abs().clamp_min(1e-9)).mean())
            out["vstar_cosine"] = float(
                torch.nn.functional.cosine_similarity(a, b, dim=-1).mean())
            out["register_norm_ratio"] = float(
                (b.norm(dim=-1) / a.norm(dim=-1).clamp_min(1e-9)).mean())
            out["state_relative_error"] = float(
                ((b - a).norm(dim=-1) / a.norm(dim=-1).clamp_min(1e-9)).mean())

    clean_obs, treated_obs = clean.at(ctx.step, layer), treated.at(ctx.step, layer)
    if clean_obs is not None and treated_obs is not None:
        if clean_obs.incoming is not None and treated_obs.incoming is not None:
            clean_sinks = clean_obs.incoming.argmax(dim=-1)
            treated_sinks = treated_obs.incoming.argmax(dim=-1)
            out["sink_agreement"] = float((clean_sinks == treated_sinks).float().mean())
            if ids:
                out["sink_on_register"] = float(np.mean(
                    [1.0 if int(s) in set(ids) else 0.0 for s in treated_sinks.tolist()]))
        if clean_obs.norm is not None and treated_obs.norm is not None and ids:
            k = min(len(ids), int(clean_obs.norm.numel()))
            clean_top = set(clean_obs.norm.topk(k).indices.tolist())
            treated_top = set(treated_obs.norm.topk(k).indices.tolist())
            union = clean_top | treated_top
            out["highnorm_jaccard"] = float(len(clean_top & treated_top) / len(union)) \
                if union else float("nan")
        if treated_obs.qk_cosine is not None and ids:
            ranks = [EP.key_rank(treated_obs.qk_cosine, head, ids)
                     for head in range(int(treated_obs.qk_cosine.shape[0]))]
            valid = [r for r in ranks if r is not None]
            out["register_key_rank"] = float(np.mean(valid)) if valid else float("nan")
    return out


def _mechanism_profile(ctx: QuestionContext, clean: Trace, treated: Trace, targets,
                       layers: Sequence[int]) -> Tuple[List[Dict[str, float]], Dict[str, float]]:
    """Per-layer mechanism fidelity, and its mean over the register zone.

    Measuring at a single layer cannot separate a lifecycle window from protection
    everywhere: the two policies are identical up to the first layer where they diverge,
    so the consumed state there is identical too.  The window comparison only becomes
    visible across depth, which is why this returns a profile rather than a scalar.  The
    headline aggregate is the mean over the register zone, because that is where the
    state the method claims to protect actually exists.
    """
    register = {int(l) for l in ctx.register_layers}
    rows, zone = [], []
    for layer in sorted({int(l) for l in layers}):
        values = _mechanism_at_layer(ctx, clean, treated, targets, layer)
        if not values:
            continue
        rows.append(dict(layer=layer, in_register_zone=layer in register, **values))
        if layer in register:
            zone.append(values)
    if not zone:                      # no observed layer inside the zone: fall back to all
        zone = [{k: v for k, v in row.items()
                 if k not in ("layer", "in_register_zone")} for row in rows]
    keys = sorted({k for row in zone for k in row})
    aggregate = {k: float(np.nanmean([row[k] for row in zone if k in row])) for k in keys} \
        if zone else {}
    return rows, aggregate


# --------------------------------------------------------------------- runner
@_resumable("q11")
def run_q11(ctx: QuestionContext, *, bits: int = 4, rank: int = 1,
            q6: Optional[QuestionResult] = None,
            schemes: Optional[Sequence[QZ.QuantScheme]] = None,
            calibrate_lowrank: bool = True) -> QuestionResult:
    """Run every quantization condition on every unit and measure both outcomes.

    One clean generation per unit serves three purposes: it is the fidelity reference,
    it supplies the frozen register selection every condition must reuse verbatim, and
    it is the calibration set for the low-rank baseline.  Reusing it is what keeps the
    comparison paired.
    """
    point = InterventionPoint.BLOCK_INPUT
    n_layers = int(ctx.driver.n_layers)
    all_layers = list(range(n_layers))
    width = int(ctx.vstar.numel())
    lpips_net = _load_lpips()

    measured = measured_window_from_q6(q6, ctx)
    windows = lifecycle_windows(ctx, measured=measured)

    rows: List[Dict[str, Any]] = []
    mechanism_rows: List[Dict[str, Any]] = []
    frames: List[pd.DataFrame] = []
    images: Dict[str, Any] = {}
    resolved_schemes: Optional[List[QZ.QuantScheme]] = None

    for prompt_id, prompt, seed in ctx.units():
        ctx.say(f"  [Q11] prompt {prompt_id} seed {seed}: clean pass")
        tracer = ctx.tracer()
        clean, _ = run_traced_generation(
            ctx.driver, tracer, prompt_id=prompt_id, prompt=prompt, seed=seed,
            condition="clean", probes=[(point, all_layers if calibrate_lowrank
                                        else list(ctx.observe_layers))],
            save_image=True)
        targets = select_frozen_targets(
            clean, layer=ctx.intervention_layer, step=ctx.step, point=point,
            direction=ctx.vstar, percentile=ctx.percentile, topk=ctx.topk,
            highnorm_ratio=ctx.highnorm_ratio)

        bases = None
        if calibrate_lowrank:
            sketch = _RangeSketch(rank=rank, seed=seed)
            for layer in all_layers:
                states = clean.probe(ctx.step, layer, point)
                if states is not None:
                    sketch.observe(layer, states)
            bases = sketch.bases()

        if resolved_schemes is None:
            resolved_schemes = list(schemes) if schemes else build_schemes(
                ctx, bits=bits, windows=windows, bases=bases, rank=rank)
        # A caller that built the scheme list up front could not have calibrated the
        # low-rank basis: it comes from this unit's clean pass. Fill it in rather than
        # letting the prior-art control degrade to plain quantization while still being
        # charged for protection it never applied.
        if bases:
            resolved_schemes = [dataclasses.replace(s, basis_by_layer=bases)
                                if s.protect == "subspace" and not s.basis_by_layer else s
                                for s in resolved_schemes]
        active = resolved_schemes

        for scheme in active:
            if scheme.key == "clean":
                treated, stats_image = clean, clean.image
            else:
                ctx.say(f"    [Q11] {scheme.key}")
                plan = EditPlan(edit=scheme.edit(), point=point, layers=all_layers,
                                steps=None, label=scheme.key, all_batch_rows=True)
                treated, _ = run_traced_generation(
                    ctx.driver, ctx.tracer(), prompt_id=prompt_id, prompt=prompt, seed=seed,
                    condition=scheme.key, plans=[plan], targets=targets, save_image=True)
                stats_image = treated.image
            if prompt_id == 0 and seed == int(list(ctx.seeds)[0]) and stats_image is not None:
                images[scheme.key] = stats_image

            budget = scheme.budget(width, n_layers)
            fidelity = _image_fidelity(clean.image, stats_image, lpips_net)
            profile, mechanism = _mechanism_profile(
                ctx, clean, treated, targets, ctx.observe_layers)
            base = dict(question="q11", condition=scheme.key, condition_label=scheme.label,
                        role=scheme.role, window=scheme.window_label,
                        protect=scheme.protect or "none", bulk_bits=scheme.bits,
                        bits_per_coordinate=budget.bits_per_coordinate,
                        bits_per_token=budget.bits_per_token,
                        prompt_id=prompt_id, seed=seed, step=ctx.step,
                        target_token_ids=json.dumps(list(targets.register_ids)))
            rows.append({**base, **fidelity, **mechanism})
            mechanism_rows.extend({**base, **layer_row} for layer_row in profile)

            if scheme.key != "clean":
                measurements = EP.measure_trace(
                    clean, treated, targets, layers=ctx.downstream_layers, step=ctx.step,
                    token_ids=list(targets.register_ids), channels=ctx.channels,
                    reference=EP.build_clean_reference(
                        clean, targets, layers=ctx.downstream_layers, step=ctx.step,
                        highnorm_ratio=ctx.highnorm_ratio))
                frames.append(_rows_to_frame(
                    measurements, question="q11", condition=scheme.key,
                    condition_label=scheme.label, role=scheme.role,
                    window=scheme.window_label, prompt_id=prompt_id, seed=seed, step=ctx.step,
                    bits_per_coordinate=budget.bits_per_coordinate))

    summary = pd.DataFrame(rows)
    tidy = pd.concat([f for f in frames if not f.empty], ignore_index=True) if frames \
        else pd.DataFrame()
    budgets = QZ.budget_table(resolved_schemes or [], width=width, n_layers=n_layers)
    return QuestionResult(
        "q11", tidy if not tidy.empty else summary,
        {"summary": summary, "mechanism": pd.DataFrame(mechanism_rows), "budget": budgets,
         "pareto": _pareto_table(summary), "windows": _window_table(windows, n_layers)},
        _q11_verdict(summary, budgets),
        meta=dict(checkpoint=getattr(ctx.cfg.spec, "key", ""), bits=bits, rank=rank,
                  n_layers=n_layers, width=width, intervention_layer=ctx.intervention_layer,
                  register_layers=list(ctx.register_layers),
                  measured_window=list(measured),
                  windows={k: sorted(v) for k, v in windows.items()},
                  lpips_available=lpips_net is not None,
                  images_available=bool(images),
                  image_metric_note=("image fidelity requires a decoder; synthetic "
                                     "checkpoints produce no image and those columns are "
                                     "reported as unavailable rather than as zero"),
                  equal_budget_groups=QZ.equal_budget_groups(budgets),
                  simulation="QDQ activation fake-quantization; weights untouched; no "
                             "low-bit kernel, so end-to-end latency is not claimed. Activation "
                             "memory is exact arithmetic over the bit allocation and IS claimed; "
                             "see ditsinks.cost_model for the footprint and the traffic-share "
                             "ceiling that bounds any speedup."),
        images=images)


def _window_table(windows: Dict[str, frozenset], n_layers: int) -> pd.DataFrame:
    return pd.DataFrame([
        dict(window=name, n_layers=len(layers), fraction_of_depth=len(layers) / max(n_layers, 1),
             first_layer=min(layers) if layers else None,
             last_layer=max(layers) if layers else None,
             layers=json.dumps(sorted(int(l) for l in layers)))
        for name, layers in windows.items()])


def _pareto_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Cost against both outcomes, one row per condition: the frontier to be plotted."""
    if summary.empty:
        return pd.DataFrame()
    metrics = [c for c in ("lpips", "rmse", "psnr", "vstar_cosine", "vstar_projection_error",
                           "sink_agreement", "highnorm_jaccard", "state_relative_error",
                           "register_key_rank")
               if c in summary.columns]
    grouped = summary.groupby(["condition", "condition_label", "role", "window"],
                              dropna=False, observed=True)
    table = grouped[metrics + ["bits_per_coordinate"]].mean().reset_index()
    return table.sort_values("bits_per_coordinate")


def _q11_verdict(summary: pd.DataFrame, budgets: pd.DataFrame) -> str:
    """State the comparison the question asks for, or say why it cannot be stated.

    An unmeasurable quantity is reported in words.  A verdict that prints a bare ``nan``
    reads as a number and is worse than one that admits the metric is unavailable.
    """
    if summary.empty:
        return "Q11 produced no runs."
    mean = summary.groupby("condition", observed=True).mean(numeric_only=True)
    needed = ("vstar_lifecycle", "magnitude_topk")
    missing = [c for c in needed if c not in mean.index]
    if missing:
        return ("Q11: the headline comparison cannot be stated because these conditions "
                f"did not run: {', '.join(missing)}.")

    # Prefer a perceptual image metric; fall back to pixel error, then to the mechanism.
    # Whichever is used is named in the sentence, so no reader has to guess.
    for column, label, lower_is_better in (("lpips", "LPIPS", True), ("rmse", "pixel RMSE", True),
                                           ("state_relative_error", "register state relative error",
                                            True)):
        if column in mean.columns and mean[column].loc[list(needed)].notna().all():
            metric, metric_label, lower = column, label, lower_is_better
            break
    else:
        return ("Q11 ran, but neither an image metric nor a register-state metric was "
                "measurable for both headline conditions, so no comparison is claimed.")

    cost = budgets.set_index("condition")["bits_per_coordinate"]
    lifecycle, magnitude = float(mean.loc["vstar_lifecycle", metric]), \
        float(mean.loc["magnitude_topk", metric])
    wins = lifecycle < magnitude if lower else lifecycle > magnitude
    mine = float(cost.get("vstar_lifecycle", float("nan")))
    theirs = float(cost.get("magnitude_topk", float("nan")))
    gap = abs(mine - theirs)
    # Which way the budget gap runs decides how the comparison may be read. A cheaper
    # mechanism-aware scheme that also wins is a conservative result; a dearer one that
    # wins is not a matched-budget claim at all, and the sentence has to say so.
    if gap != gap:
        budget_note = "budgets could not be compared"
    elif gap < 1e-9:
        budget_note = "budgets matched exactly"
    elif mine < theirs:
        budget_note = (f"{gap:.4f} bits/coordinate cheaper than the magnitude control, "
                       f"so the comparison is conservative")
    else:
        budget_note = (f"{gap:.4f} bits/coordinate dearer than the magnitude control, "
                       f"so this is not a matched-budget claim")

    parts = [
        f"Q11: at {cost.get('vstar_lifecycle', float('nan')):.3f} bits/coordinate "
        f"({budget_note}), lifecycle-gated v* protection reaches {metric_label} "
        f"{lifecycle:.4f} against {magnitude:.4f} for equal-budget magnitude protection -- "
        f"{'better' if wins else 'not better'} ({'lower' if lower else 'higher'} is better)."]

    if "vstar_all_layers" in mean.index and mean.loc["vstar_all_layers", metric] == \
            mean.loc["vstar_all_layers", metric]:
        everywhere = float(mean.loc["vstar_all_layers", metric])
        parts.append(f"Protecting v* at every layer reaches {everywhere:.4f} at "
                     f"{cost.get('vstar_all_layers', float('nan')):.3f} bits/coordinate, so the "
                     f"lifecycle window "
                     f"{'matches or beats it more cheaply' if (lifecycle <= everywhere) == lower else 'costs quality'}.")

    wrong = [c for c in ("vstar_early_window", "vstar_late_window") if c in mean.index]
    if wrong and all(mean.loc[c, metric] == mean.loc[c, metric] for c in wrong):
        worst = max((float(mean.loc[c, metric]) for c in wrong)) if lower \
            else min((float(mean.loc[c, metric]) for c in wrong))
        beaten = (lifecycle < worst) if lower else (lifecycle > worst)
        parts.append(f"Against equally sized wrong windows it is "
                     f"{'better' if beaten else 'no better'} ({worst:.4f} at worst).")

    if "vstar_cosine" in mean.columns and mean.loc[list(needed), "vstar_cosine"].notna().all():
        parts.append(
            f"On the mechanism, v* alignment in the register zone is "
            f"{float(mean.loc['vstar_lifecycle', 'vstar_cosine']):.4f} under lifecycle "
            f"protection against {float(mean.loc['magnitude_topk', 'vstar_cosine']):.4f} "
            f"under magnitude protection.")

    parts.append(
        "QDQ simulation of an activation policy: the quality comparison is evidence about "
        "numerical robustness, the activation footprint is exact arithmetic over the bit "
        "allocation, and end-to-end latency is not claimed because it needs real low-bit "
        "kernels.")
    return " ".join(parts)
