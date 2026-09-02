"""Identify v*: the direction the register tokens share.

The claim under test is specific, and each part gets its own number:

  1. registers point ONE way          -> variance explained by the top singular
                                         direction of the sign-free register matrix
  2. that way is prompt/seed-invariant-> cosine between v* fitted separately per
                                         (prompt, seed), against a matched
                                         ordinary-token null
  3. one channel carries most of it   -> energy share of v*'s top channel, and
                                         whether it is the model's top massive
                                         activation channel
  4. selection is by direction, not   -> hit rate of sink tokens when ranking all
     magnitude                          tokens by |x . v*| vs by ||x||
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from .capture import LayerRecord
from .metrics import highnorm_mask, sink_profile


@dataclass
class VStarReport:
    v: torch.Tensor                       # [C], unit
    n_vectors: int
    explained_variance: float             # share of register-direction variance on v*
    singular_spectrum: np.ndarray
    sign_positive_fraction: float
    top_channels: List[int]
    top_channel_energy: List[float]
    cumulative_energy: np.ndarray
    participation_ratio: float
    massive_channel: Optional[int]
    massive_channel_rank_in_vstar: Optional[int]
    cos_registers: np.ndarray
    cos_controls: np.ndarray
    cos_random: np.ndarray
    condition_labels: List[str]
    condition_cos: np.ndarray             # [K, K] cosine between per-condition v*
    layer_profile: pd.DataFrame
    selection: pd.DataFrame               # direction vs magnitude, per layer
    coverage: float                       # v* energy inside the stored channel subset
    layers_used: List[int]
    meta: Dict[str, object] = field(default_factory=dict)

    @property
    def top_channel(self) -> int:
        return int(self.top_channels[0])

    def summary(self) -> str:
        lines = []
        if not self.meta.get("fitted_on_highnorm", True):
            lines.append(
                "CAVEAT: no token anywhere cleared the high-norm threshold, so v* was fitted on "
                "the loudest tokens per layer instead. That makes this 'the direction the loudest "
                "tokens share', which is a much weaker claim than a register direction."
            )
        lines += [
            f"v* fitted on {self.n_vectors} register directions from layers "
            f"{min(self.layers_used)}-{max(self.layers_used)}",
            f"  one direction?        {self.explained_variance:.1%} of the register-direction "
            f"variance lies on v* (random unit vectors would give ~{1.0 / max(self.v.numel(), 1):.2%})",
            f"  sign consistency      {self.sign_positive_fraction:.1%} of registers point the same way along v*",
            (f"  prompt/seed invariant cos between per-condition v* = "
             f"{_offdiag_mean(self.condition_cos):.3f} (n={len(self.condition_labels)} conditions)"
             if len(self.condition_labels) >= 2 else
             "  prompt/seed invariant  NOT TESTED - only one (prompt, seed) condition in this sweep; "
             "add prompts or seeds to test invariance"),
            f"  registers vs ordinary |cos(x, v*)| = {np.abs(self.cos_registers).mean():.3f} "
            f"vs {np.abs(self.cos_controls).mean():.3f} (random: {np.abs(self.cos_random).mean():.3f})",
            f"  top channel           {self.top_channel} carries {self.top_channel_energy[0]:.1%} "
            f"of v*'s energy; top-4 carry {self.cumulative_energy[3]:.1%}",
        ]
        if self.massive_channel is not None:
            same = self.massive_channel == self.top_channel
            lines.append(
                f"  massive channel       model's loudest channel is {self.massive_channel}; "
                + ("it IS v*'s top channel" if same else
                   f"it sits at rank {self.massive_channel_rank_in_vstar} in v*")
            )
        if not self.selection.empty:
            s = self.selection.mean(numeric_only=True)
            lines.append(
                f"  direction vs magnitude sink hit-rate in the top 1% ranked by |x.v*| = "
                f"{s.get('hit_direction', float('nan')):.2f} vs by ||x|| = {s.get('hit_norm', float('nan')):.2f} "
                f"(chance {s.get('hit_chance', float('nan')):.2f})"
            )
        return "\n".join(lines)


# ------------------------------------------------------------------- fitting
def _register_rows(result, layers: Optional[Sequence[int]], ratio: float
                   ) -> Tuple[List[torch.Tensor], List[dict], List[torch.Tensor]]:
    regs, meta, ctrls = [], [], []
    for rec in result.records.values():
        if layers is not None and rec.layer not in set(layers):
            continue
        if rec.register_vecs is None:
            continue
        norms = rec.norms.get("post_block")
        keep = None
        if norms is not None and rec.register_ids is not None:
            hn = highnorm_mask(norms, ratio)
            keep = [i for i, t in enumerate(rec.register_ids.tolist()) if bool(hn[int(t)])]
        if keep is None:
            keep = list(range(rec.register_vecs.shape[0]))
        for i in keep:
            v = rec.register_vecs[i].float()
            if float(v.norm()) <= 0:
                continue
            regs.append(v)
            meta.append(dict(prompt_id=rec.prompt_id, seed=rec.seed, step=rec.step,
                             layer=rec.layer, kind=rec.kind, token=int(rec.register_ids[i]),
                             norm=float(v.norm())))
        if rec.control_vecs is not None:
            for j in range(rec.control_vecs.shape[0]):
                c = rec.control_vecs[j].float()
                if float(c.norm()) > 0:
                    ctrls.append(c)
    return regs, meta, ctrls


def _fit_direction(vectors: Sequence[torch.Tensor]) -> Tuple[torch.Tensor, float, np.ndarray]:
    """Top right-singular vector of the unit-vector matrix.

    SVD is the sign-free way to ask "is there one shared axis?" -- it maximises
    the mean squared cosine, so a population split into +v and -v still resolves
    to one axis instead of averaging to zero.
    """
    U = torch.stack([v / v.norm().clamp_min(1e-9) for v in vectors])     # [M, C]
    U = U - 0.0                                                          # no centring: direction, not spread
    try:
        _, S, Vh = torch.linalg.svd(U, full_matrices=False)
    except Exception:
        _, S, Vh = torch.svd_lowrank(U, q=min(8, min(U.shape)))
        Vh = Vh.T
    v = Vh[0]
    v = v / v.norm().clamp_min(1e-9)
    spectrum = (S ** 2).numpy()
    explained = float(spectrum[0] / max(spectrum.sum(), 1e-12))
    if float((U @ v).sum()) < 0:                                          # fix the global sign
        v = -v
    return v, explained, spectrum


def fit_vstar(result, layers: Optional[Sequence[int]] = None, ratio: Optional[float] = None,
              n_top_channels: int = 12, rng_seed: int = 0, fallback: bool = True) -> VStarReport:
    ratio = result.cfg.highnorm_ratio if ratio is None else ratio
    if layers is None:
        layers = _layers_with_registers(result, ratio)
    regs, meta, ctrls = _register_rows(result, layers, ratio)
    fell_back = False
    if len(regs) < 3 and fallback:
        # No token clears the high-norm bar anywhere. Fit on the loudest tokens
        # instead so the rest of the analysis still runs -- but say so, loudly:
        # "the loudest tokens share a direction" is a much weaker statement.
        fell_back = True
        layers = sorted({r.layer for r in result.records.values()})
        regs, meta, ctrls = _register_rows(result, layers, 0.0)
    if len(regs) < 3:
        raise ValueError(
            "Not enough tokens to fit a direction -- the sweep captured no register vectors. "
            "Check that the sweep ran and that register_topk > 0."
        )

    v, explained, spectrum = _fit_direction(regs)
    C = int(v.numel())
    meta_df = pd.DataFrame(meta)

    U = torch.stack([r / r.norm().clamp_min(1e-9) for r in regs])
    cos_registers = (U @ v).numpy()
    if ctrls:
        Uc = torch.stack([c / c.norm().clamp_min(1e-9) for c in ctrls])
        cos_controls = (Uc @ v).numpy()
    else:
        cos_controls = np.zeros(0, dtype=np.float32)
    g = torch.Generator().manual_seed(rng_seed)
    R = torch.randn(max(len(regs), 256), C, generator=g)
    R = R / R.norm(dim=-1, keepdim=True)
    cos_random = (R @ v).numpy()

    energy = (v ** 2).numpy()
    order = np.argsort(energy)[::-1]
    top_channels = [int(c) for c in order[:n_top_channels]]
    top_energy = [float(energy[c]) for c in top_channels]
    cumulative = np.cumsum(energy[order])
    participation = float(1.0 / np.sum(energy ** 2))

    massive_channel = _modal_massive_channel(result, layers)
    massive_rank = None
    if massive_channel is not None:
        massive_rank = int(np.where(order == massive_channel)[0][0]) + 1

    labels, cond_cos = _condition_invariance(result, layers, ratio)
    layer_profile = _layer_profile(result, v, ratio)
    selection, coverage = _direction_vs_magnitude(result, v, ratio)

    return VStarReport(
        v=v, n_vectors=len(regs), explained_variance=explained, singular_spectrum=spectrum,
        sign_positive_fraction=float((cos_registers > 0).mean()),
        top_channels=top_channels, top_channel_energy=top_energy, cumulative_energy=cumulative,
        participation_ratio=participation, massive_channel=massive_channel,
        massive_channel_rank_in_vstar=massive_rank,
        cos_registers=cos_registers, cos_controls=cos_controls, cos_random=cos_random,
        condition_labels=labels, condition_cos=cond_cos, layer_profile=layer_profile,
        selection=selection, coverage=coverage, layers_used=sorted(set(meta_df["layer"])),
        meta=dict(model=result.meta.get("model"), family=result.meta.get("family"),
                  d_model=C, registers=meta_df, fitted_on_highnorm=not fell_back),
    )


def _layers_with_registers(result, ratio: float) -> List[int]:
    keep = []
    for rec in result.records.values():
        norms = rec.norms.get("post_block")
        if norms is None:
            continue
        if bool(highnorm_mask(norms, ratio).any()):
            keep.append(rec.layer)
    return sorted(set(keep)) or sorted({r.layer for r in result.records.values()})


def _modal_massive_channel(result, layers: Sequence[int]) -> Optional[int]:
    votes: Dict[int, int] = {}
    for rec in result.records.values():
        if rec.layer not in set(layers) or rec.channel_absmax is None:
            continue
        c = int(torch.argmax(rec.channel_absmax))
        votes[c] = votes.get(c, 0) + 1
    if not votes:
        return None
    return max(votes.items(), key=lambda kv: kv[1])[0]


def _condition_invariance(result, layers: Sequence[int], ratio: float) -> Tuple[List[str], np.ndarray]:
    """Fit v* independently per (prompt, seed) and compare."""
    groups: Dict[Tuple[int, int], List[torch.Tensor]] = {}
    for rec in result.records.values():
        if rec.layer not in set(layers) or rec.register_vecs is None:
            continue
        norms = rec.norms.get("post_block")
        hn = highnorm_mask(norms, ratio) if norms is not None else None
        for i in range(rec.register_vecs.shape[0]):
            t = int(rec.register_ids[i])
            if hn is not None and not bool(hn[t]):
                continue
            groups.setdefault((rec.prompt_id, rec.seed), []).append(rec.register_vecs[i].float())
    keys = sorted(k for k, v in groups.items() if len(v) >= 3)
    if len(keys) < 2:
        return [], np.zeros((0, 0))
    vs = []
    for k in keys:
        v, _, _ = _fit_direction(groups[k])
        vs.append(v)
    V = torch.stack(vs)
    M = (V @ V.T).numpy()
    # Directions are defined up to sign; compare axes.
    M = np.abs(M)
    labels = [f"p{p} s{s}" for p, s in keys]
    return labels, M


def _layer_profile(result, v: torch.Tensor, ratio: float) -> pd.DataFrame:
    rows = []
    for rec in result.records.values():
        norms = rec.norms.get("post_block")
        if rec.register_vecs is None or norms is None:
            continue
        hn = highnorm_mask(norms, ratio)
        reg_cos, reg_proj = [], []
        for i in range(rec.register_vecs.shape[0]):
            t = int(rec.register_ids[i])
            if not bool(hn[t]):
                continue
            x = rec.register_vecs[i].float()
            reg_cos.append(float(x @ v / x.norm().clamp_min(1e-9)))
            reg_proj.append(float(abs(x @ v)))
        ctrl_cos, ctrl_proj = [], []
        if rec.control_vecs is not None:
            for j in range(rec.control_vecs.shape[0]):
                x = rec.control_vecs[j].float()
                ctrl_cos.append(float(x @ v / x.norm().clamp_min(1e-9)))
                ctrl_proj.append(float(abs(x @ v)))
        rows.append(dict(
            layer=rec.layer, kind=rec.kind, prompt_id=rec.prompt_id, seed=rec.seed, step=rec.step,
            n_registers=len(reg_cos),
            register_cos=float(np.mean(np.abs(reg_cos))) if reg_cos else np.nan,
            register_proj=float(np.mean(reg_proj)) if reg_proj else np.nan,
            control_cos=float(np.mean(np.abs(ctrl_cos))) if ctrl_cos else np.nan,
            control_proj=float(np.mean(ctrl_proj)) if ctrl_proj else np.nan,
        ))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.groupby(["layer", "kind"], as_index=False).mean(numeric_only=True).sort_values("layer")


def direction_score(rec: LayerRecord, v: torch.Tensor) -> Tuple[Optional[torch.Tensor], float]:
    """Approximate x . v* for every token, using the stored loud-channel columns.

    Returns (scores [N], coverage) where coverage is the share of v*'s energy
    inside those channels -- report it, do not hide it.
    """
    if rec.channel_top_ids is None or rec.channel_top_values is None:
        return None, 0.0
    ids = rec.channel_top_ids.long()
    sub = v[ids]
    coverage = float((sub ** 2).sum())
    scores = rec.channel_top_values.float() @ sub
    return scores, coverage


def _direction_vs_magnitude(result, v: torch.Tensor, ratio: float) -> Tuple[pd.DataFrame, float]:
    """Does |x.v*| pick out the sinks better than ||x|| does?"""
    rows, coverages = [], []
    for rec in result.records.values():
        norms = rec.norms.get("post_block")
        prof = sink_profile(rec)
        if norms is None or prof is None:
            continue
        scores, cov = direction_score(rec, v)
        if scores is None:
            continue
        coverages.append(cov)
        n = norms.numel()
        k = max(1, n // 100)
        sink_set = set(torch.topk(prof, k).indices.tolist())
        by_norm = set(torch.topk(norms, k).indices.tolist())
        by_dir = set(torch.topk(scores.abs(), k).indices.tolist())
        by_cos = set(torch.topk(scores.abs() / norms.clamp_min(1e-9), k).indices.tolist())
        rows.append(dict(
            layer=rec.layer, kind=rec.kind, prompt_id=rec.prompt_id, seed=rec.seed, step=rec.step,
            k=k,
            hit_norm=len(by_norm & sink_set) / k,
            hit_direction=len(by_dir & sink_set) / k,
            hit_cosine=len(by_cos & sink_set) / k,
            hit_chance=k / n,
        ))
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.groupby(["layer", "kind"], as_index=False).mean(numeric_only=True).sort_values("layer")
    return df, float(np.mean(coverages)) if coverages else 0.0


def _offdiag_mean(m: np.ndarray) -> float:
    if m.size == 0 or m.shape[0] < 2:
        return float("nan")
    iu = np.triu_indices(m.shape[0], k=1)
    return float(np.mean(m[iu]))


def exact_selection_table(result, projection_rows: Dict[Tuple[int, int, int, int], Dict[str, torch.Tensor]]
                          ) -> pd.DataFrame:
    """Direction vs magnitude, using exact projections from `run_projection_sweep`.

    For each layer: rank all image tokens by |x . v*|, by ||x||, and by the
    cosine, then ask how many of each ranking's top 1% are the layer's top-1%
    attention sinks. Direction wins => sinks are picked by where a token points.
    """
    rows = []
    for key, rec in result.records.items():
        got = projection_rows.get(key)
        prof = sink_profile(rec)
        if got is None or prof is None:
            continue
        proj = got["proj"].float()
        norms = got["norm"].float()
        n = int(norms.numel())
        if n != int(prof.numel()):
            continue
        k = max(1, n // 100)
        sink_set = set(torch.topk(prof, k).indices.tolist())
        by_norm = set(torch.topk(norms, k).indices.tolist())
        by_dir = set(torch.topk(proj.abs(), k).indices.tolist())
        by_cos = set(torch.topk(proj.abs() / norms.clamp_min(1e-9), k).indices.tolist())
        rows.append(dict(
            layer=rec.layer, kind=rec.kind, prompt_id=rec.prompt_id, seed=rec.seed, step=rec.step,
            k=k,
            hit_norm=len(by_norm & sink_set) / k,
            hit_direction=len(by_dir & sink_set) / k,
            hit_cosine=len(by_cos & sink_set) / k,
            hit_chance=k / n,
            max_abs_cos=float((proj.abs() / norms.clamp_min(1e-9)).max()),
        ))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.groupby(["layer", "kind"], as_index=False).mean(numeric_only=True).sort_values("layer")


def attach_exact_selection(report: VStarReport, result, projection_rows) -> VStarReport:
    df = exact_selection_table(result, projection_rows)
    if not df.empty:
        report.selection = df
        report.coverage = 1.0
        report.meta["selection_is_exact"] = True
    return report
