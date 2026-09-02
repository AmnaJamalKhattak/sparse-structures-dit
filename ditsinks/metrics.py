"""Turn LayerRecords into tidy tables.

Three phenomena, one row per (prompt, seed, step, layer):

high-norm tokens          a few image tokens whose residual norm is orders of
                          magnitude above the bulk
attention sinks           image keys that absorb a disproportionate share of the
                          image->image attention mass
massive activation channels  channels whose peak |activation| dwarfs every other
                          channel's

...plus the coupling between them, which is the actual question.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from .capture import LayerRecord


# --------------------------------------------------------------- definitions
def highnorm_mask(norms: torch.Tensor, ratio: float) -> torch.Tensor:
    """Tokens whose norm exceeds `ratio` x the median norm of that layer."""
    med = norms.median().clamp_min(1e-9)
    return norms > ratio * med


def mad_outlier_mask(norms: torch.Tensor, z: float = 6.0) -> torch.Tensor:
    med = norms.median()
    mad = (norms - med).abs().median().clamp_min(1e-9)
    return norms > med + z * 1.4826 * mad


def massive_channel_mask(channel_absmax: torch.Tensor, ratio: float) -> torch.Tensor:
    """Channels whose peak |activation| exceeds `ratio` x the median channel peak."""
    med = channel_absmax.median().clamp_min(1e-9)
    return channel_absmax > ratio * med


def sink_profile(rec: LayerRecord) -> Optional[torch.Tensor]:
    """Head-mean incoming image->image attention per image token, [N]."""
    if rec.incoming_img2img is None:
        return None
    return rec.incoming_img2img.mean(dim=0)


def expected_jaccard(k: int, n: int) -> float:
    """Jaccard of two independent uniform k-subsets of n items."""
    if k <= 0 or n <= 0:
        return float("nan")
    ei = (k * k) / n
    return ei / max(2 * k - ei, 1e-9)


def _jaccard(a: set, b: set) -> float:
    u = a | b
    return len(a & b) / len(u) if u else float("nan")


def _topk_set(x: torch.Tensor, k: int) -> set:
    k = max(1, min(int(k), x.numel()))
    return set(torch.topk(x, k).indices.tolist())


# ------------------------------------------------------------- layer summary
def layer_table(result, ratio: Optional[float] = None) -> pd.DataFrame:
    """One row per (prompt, seed, step, layer): the layer atlas in numbers."""
    cfg = result.cfg
    ratio = cfg.highnorm_ratio if ratio is None else ratio
    rows: List[dict] = []

    for rec in sorted(result.records.values(), key=lambda r: (r.prompt_id, r.seed, r.step, r.layer)):
        norms = rec.norms.get("post_block")
        if norms is None or norms.numel() == 0:
            continue
        n = int(norms.numel())
        med = float(norms.median())
        hn = highnorm_mask(norms, ratio)
        row = dict(
            model=result.meta.get("model", cfg.model),
            family=result.meta.get("family", cfg.family),
            prompt_id=rec.prompt_id, seed=rec.seed, step=rec.step, timestep=rec.timestep,
            layer=rec.layer, kind=rec.kind, local_id=rec.local_id, block_name=rec.block_name,
            n_img=n, n_heads=rec.n_heads, d_model=rec.d_model,
            # ---- high-norm tokens
            norm_median=med,
            norm_max=float(norms.max()),
            norm_p99=float(np.percentile(norms.numpy(), 99)),
            norm_max_over_median=float(norms.max()) / max(med, 1e-9),
            n_highnorm=int(hn.sum()),
            highnorm_frac=float(hn.float().mean()),
            n_mad_outliers=int(mad_outlier_mask(norms).sum()),
        )

        # ---- massive activation channels
        cam = rec.channel_absmax
        if cam is not None and cam.numel():
            mc = massive_channel_mask(cam, cfg.massive_channel_ratio)
            ch_med = float(cam.median())
            row.update(
                chan_absmax=float(cam.max()),
                chan_median_absmax=ch_med,
                chan_max_over_median=float(cam.max()) / max(ch_med, 1e-9),
                n_massive_channels=int(mc.sum()),
                top_channel=int(torch.argmax(cam)),
                top_channel_absmax=float(cam.max()),
            )

        # ---- attention sinks
        prof = sink_profile(rec)
        if prof is not None:
            uniform = 1.0 / max(n, 1)
            per_head_sink = rec.incoming_img2img.argmax(dim=-1)
            per_head_strength = rec.incoming_img2img.amax(dim=-1)
            modal, modal_count = torch.mode(per_head_sink)
            k1pct = max(1, n // 100)
            top1pct_mass = float(torch.topk(prof, k1pct).values.sum())
            ent = rec.attn_entropy
            row.update(
                sink_strength_headmean=float(per_head_strength.mean()),
                sink_strength_headmax=float(per_head_strength.max()),
                sink_ratio_headmean=float(per_head_strength.mean()) / uniform,
                sink_ratio_headmax=float(per_head_strength.max()) / uniform,
                sink_mass_top1=float(prof.max()),
                sink_mass_top1pct=top1pct_mass,
                sink_mass_top1pct_over_uniform=top1pct_mass / (k1pct * uniform),
                sink_token_headmean=int(torch.argmax(prof)),
                n_unique_head_sinks=int(torch.unique(per_head_sink).numel()),
                head_sink_consensus=float(modal_count) / max(rec.n_heads, 1),
                modal_sink_token=int(modal),
                attn_entropy=float(ent.mean()) if ent is not None else float("nan"),
                attn_entropy_norm=float(ent.mean()) / math.log(max(n, 2)) if ent is not None else float("nan"),
                text_attention_mass=float(rec.text_mass.mean()) if rec.text_mass is not None else 0.0,
                text_sink_strength=float(rec.text_sink_strength.mean()) if rec.text_sink_strength is not None else 0.0,
            )
            row["has_sinks"] = bool(row["sink_ratio_headmax"] >= cfg.sink_ratio_threshold)

            # ---- coupling: are the loud tokens the sinks?
            k = max(1, min(cfg.sink_topk, n))
            hn_set = _topk_set(norms, k)
            sk_set = _topk_set(prof, k)
            row["jaccard_topk"] = _jaccard(hn_set, sk_set)
            row["jaccard_topk_chance"] = expected_jaccard(k, n)
            k1 = max(1, n // 100)
            row["top1pct_overlap"] = len(_topk_set(norms, k1) & _topk_set(prof, k1)) / k1
            hn_ids = set(torch.nonzero(hn).flatten().tolist())
            row["heads_sinking_on_highnorm"] = (
                float(np.mean([int(t) in hn_ids for t in per_head_sink.tolist()])) if hn_ids else 0.0
            )
            norm_rank = torch.empty(n, dtype=torch.long)
            norm_rank[torch.argsort(norms, descending=True)] = torch.arange(n)
            row["median_sink_norm_rank"] = float(norm_rank[per_head_sink].float().median()) + 1.0
            row["spearman_norm_vs_attention"] = _spearman(norms.numpy(), prof.numpy())

            # ---- coupling: is the sink token loud in the massive channel?
            if rec.channel_top_values is not None and rec.channel_top_ids is not None:
                v = rec.channel_top_values.float()[:, 0].abs()      # loudest channel, per token
                row["top_channel_spike_token"] = int(torch.argmax(v))
                row["sink_is_top_channel_spike"] = bool(int(torch.argmax(v)) == int(torch.argmax(prof)))
                row["highnorm_is_top_channel_spike"] = bool(int(torch.argmax(v)) == int(torch.argmax(norms)))
                k1 = max(1, n // 100)
                row["top_channel_spike_overlap_highnorm"] = len(_topk_set(v, k1) & _topk_set(norms, k1)) / k1

        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["prompt_id", "seed", "step", "layer"]).reset_index(drop=True)
    return df


def layer_summary(df: pd.DataFrame, by: Sequence[str] = ("layer",)) -> pd.DataFrame:
    """Average the layer table over prompts/seeds/steps (the atlas backbone)."""
    if df.empty:
        return df
    keep = [c for c in df.columns if df[c].dtype.kind in "biufc"]
    firsts = ["kind", "block_name"]
    g = df.groupby(list(by), as_index=False)
    out = g[keep].mean(numeric_only=True)
    meta = g[[c for c in firsts if c in df.columns]].first()
    for c in firsts:
        if c in meta.columns:
            out[c] = meta[c].values
    return out.sort_values(list(by)).reset_index(drop=True)


# ------------------------------------------------- matrices for 2-D histograms
def layer_value_matrix(result, quantity: str, step: Optional[int] = None,
                       prompt_id: Optional[int] = None, seed: Optional[int] = None
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """Stack a per-token (or per-channel) quantity into a [n_layers, n_values] array.

    `quantity` is one of:
      "norm"            residual-stream token norms          -> high-norm tokens
      "attention"       head-mean incoming attention mass    -> attention sinks
      "attention_ratio" the same, in multiples of uniform    -> attention sinks
      "channel"         per-channel peak |activation|        -> massive channels
    """
    picks: Dict[int, List[torch.Tensor]] = {}
    for rec in result.records.values():
        if step is not None and rec.step != step:
            continue
        if prompt_id is not None and rec.prompt_id != prompt_id:
            continue
        if seed is not None and rec.seed != seed:
            continue
        v = _quantity(rec, quantity)
        if v is None:
            continue
        picks.setdefault(rec.layer, []).append(v)

    layers = sorted(picks)
    if not layers:
        return np.zeros((0, 0)), np.array([])
    width = max(sum(t.numel() for t in picks[l]) for l in layers)
    mat = np.full((len(layers), width), np.nan, dtype=np.float32)
    for i, l in enumerate(layers):
        vals = torch.cat(picks[l]).numpy()
        mat[i, : vals.size] = vals
    return mat, np.array(layers)


def _quantity(rec: LayerRecord, quantity: str) -> Optional[torch.Tensor]:
    if quantity == "norm":
        return rec.norms.get("post_block")
    if quantity in ("attention", "attention_ratio"):
        prof = sink_profile(rec)
        if prof is None:
            return None
        return prof * rec.n_img if quantity == "attention_ratio" else prof
    if quantity == "channel":
        return rec.channel_absmax
    raise KeyError(f"unknown quantity {quantity!r}")


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 3:
        return float("nan")
    try:
        from scipy import stats

        r, _ = stats.spearmanr(a, b)
        return float(r)
    except Exception:
        ra = pd.Series(a).rank().to_numpy()
        rb = pd.Series(b).rank().to_numpy()
        if ra.std() == 0 or rb.std() == 0:
            return float("nan")
        return float(np.corrcoef(ra, rb)[0, 1])


# ------------------------------------------------------------ head-level table
def head_table(result) -> pd.DataFrame:
    """One row per head: which token it sinks on, how hard, and how loud that token is."""
    rows: List[dict] = []
    for rec in result.records.values():
        if rec.incoming_img2img is None:
            continue
        norms = rec.norms.get("post_block")
        prof = sink_profile(rec)
        n = rec.n_img
        hn = highnorm_mask(norms, result.cfg.highnorm_ratio) if norms is not None else None
        norm_rank = None
        if norms is not None:
            norm_rank = torch.empty(n, dtype=torch.long)
            norm_rank[torch.argsort(norms, descending=True)] = torch.arange(n)
        raw_img_max = (rec.incoming_raw_img.amax(dim=-1)
                       if rec.incoming_raw_img is not None else None)
        for h in range(rec.incoming_img2img.shape[0]):
            inc = rec.incoming_img2img[h]
            tok = int(inc.argmax())
            rows.append(dict(
                prompt_id=rec.prompt_id, seed=rec.seed, step=rec.step, layer=rec.layer,
                kind=rec.kind, block_name=rec.block_name, head_id=h, n_img=n,
                sink_token=tok,
                sink_strength=float(inc.max()),
                sink_ratio_over_uniform=float(inc.max()) * n,
                sink_token_norm=float(norms[tok]) if norms is not None else float("nan"),
                sink_token_norm_rank=int(norm_rank[tok]) + 1 if norm_rank is not None else -1,
                sink_is_highnorm=bool(hn[tok]) if hn is not None else False,
                text_mass=float(rec.text_mass[h]) if rec.text_mass is not None else 0.0,
                sink_is_text=bool(rec.text_sink_strength is not None and raw_img_max is not None
                                  and float(rec.text_sink_strength[h]) > float(raw_img_max[h])),
                spearman_norm_vs_attention=(_spearman(norms.numpy(), inc.numpy())
                                            if norms is not None else float("nan")),
                entropy=float(rec.attn_entropy[h]) if rec.attn_entropy is not None else float("nan"),
                headmean_sink_token=int(prof.argmax()) if prof is not None else -1,
            ))
    return pd.DataFrame(rows)


def sink_layer_verdict(df: pd.DataFrame, threshold: Optional[float] = None) -> pd.DataFrame:
    """Per layer: does this layer have attention sinks at all, and how strong?"""
    if df.empty:
        return df
    agg = df.groupby(["layer", "kind", "block_name"], as_index=False).agg(
        sink_ratio=("sink_ratio_headmax", "mean"),
        sink_ratio_headmean=("sink_ratio_headmean", "mean"),
        head_consensus=("head_sink_consensus", "mean"),
        mass_top1pct=("sink_mass_top1pct", "mean"),
        entropy_norm=("attn_entropy_norm", "mean"),
        n_highnorm=("n_highnorm", "mean"),
        n_massive_channels=("n_massive_channels", "mean"),
        norm_ratio=("norm_max_over_median", "mean"),
        jaccard=("jaccard_topk", "mean"),
        jaccard_chance=("jaccard_topk_chance", "mean"),
    )
    thr = threshold if threshold is not None else 10.0
    agg["has_sinks"] = agg["sink_ratio"] >= thr
    agg["has_highnorm"] = agg["n_highnorm"] > 0
    agg["has_massive_channels"] = agg["n_massive_channels"] > 0
    return agg.sort_values("layer").reset_index(drop=True)


def answer_where_are_the_sinks(result, df: Optional[pd.DataFrame] = None) -> str:
    """Plain-English answer to 'in which layers do attention sinks exist?'."""
    df = layer_table(result) if df is None else df
    v = sink_layer_verdict(df, threshold=result.cfg.sink_ratio_threshold)
    if v.empty:
        return "No layers captured."
    name = {int(r.layer): r.block_name for r in v.itertuples()}

    def _span(mask_col: str) -> str:
        on = v[v[mask_col]]["layer"].tolist()
        if not on:
            return "none"
        runs, start, prev = [], on[0], on[0]
        for l in on[1:]:
            if l != prev + 1:
                runs.append((start, prev))
                start = l
            prev = l
        runs.append((start, prev))
        return ", ".join(
            f"L{a}-L{b} ({name[a]}..{name[b]})" if a != b else f"L{a} ({name[a]})" for a, b in runs
        )

    peak = v.loc[v["sink_ratio"].idxmax()]
    model = result.meta.get("model", result.cfg.model)
    lines = [
        f"{model}: {len(v)} layers captured"
        + (f", dual->single boundary after L{int(result.zone_boundary() - 0.5)}"
           if result.zone_boundary() is not None else " (single-stream model)"),
        f"  attention sinks           {_span('has_sinks')}",
        f"  high-norm tokens          {_span('has_highnorm')}",
        f"  massive activation chans  {_span('has_massive_channels')}",
        f"  strongest sink            L{int(peak['layer'])} ({peak['block_name']}): the top image key takes "
        f"{peak['sink_ratio']:,.0f}x its uniform share; {peak['head_consensus']:.0%} of heads agree on it",
    ]
    coupled = v[v["has_sinks"]]
    if not coupled.empty:
        lines.append(
            f"  coupling in sink layers   top-k high-norm and top-k sink sets overlap "
            f"{coupled['jaccard'].mean():.2f} (chance {coupled['jaccard_chance'].mean():.3f})"
        )
    return "\n".join(lines)


def qk_geometry_table(result, topk: int = 10) -> pd.DataFrame:
    """Natural register-key alignment with the mean image query, per head.

    The statistic is observational.  It uses the top post-block norm token as
    the register candidate and never claims that ``W_K v*`` is sufficient.
    Older saved sweeps predate this compact capture and return an empty table.
    """
    rows = []
    for rec in result.records.values():
        cos = getattr(rec, "qk_mean_cosine", None)
        if cos is None or rec.register_ids is None:
            continue
        token = int(rec.register_ids[0])
        if token >= cos.shape[1]:
            continue
        rank = (cos > cos[:, token:token + 1]).sum(dim=1) + 1
        ordinary = cos.median(dim=1).values
        for head in range(cos.shape[0]):
            rows.append(dict(
                layer=rec.layer, kind=rec.kind, block_name=rec.block_name,
                prompt_id=rec.prompt_id, seed=rec.seed, step=rec.step,
                head_id=head, register_token=token,
                register_key_cosine=float(cos[head, token]),
                ordinary_key_cosine=float(ordinary[head]),
                register_key_rank=int(rank[head]),
                register_key_topk=bool(rank[head] <= min(topk, cos.shape[1])),
                n_image_keys=int(cos.shape[1]),
            ))
    return pd.DataFrame(rows)


# ------------------------------------------------- stage- and threshold-sweeps
def stage_overlap_table(result, ks: Sequence[int] = (1, 5, 10)) -> pd.DataFrame:
    """Overlap between the top-k tokens by norm and the top-k attention sinks.

    Computed for every residual-stream stage the architecture exposes, because
    where the high-norm token is measured changes the answer: a token can be
    marked by attention and only amplified by the MLP one sub-layer later.
    """
    rows: List[dict] = []
    for rec in result.records.values():
        prof = sink_profile(rec)
        if prof is None:
            continue
        n = rec.n_img
        for stage, norms in rec.norms.items():
            if norms is None or norms.numel() != n:
                continue
            for k in ks:
                kk = max(1, min(int(k), n))
                hn = _topk_set(norms, kk)
                sk = _topk_set(prof, kk)
                rows.append(dict(
                    prompt_id=rec.prompt_id, seed=rec.seed, step=rec.step,
                    layer=rec.layer, kind=rec.kind, block_name=rec.block_name,
                    stage=stage, k=kk, n_img=n,
                    jaccard=_jaccard(hn, sk),
                    chance=expected_jaccard(kk, n),
                    any_overlap=float(len(hn & sk) > 0),
                ))
    return pd.DataFrame(rows)


def percentile_sweep_table(result, percentiles: Sequence[float] = (90.0, 95.0, 97.7, 99.0, 99.9),
                           stages: Sequence[str] = ("attn_out", "post_block")) -> pd.DataFrame:
    """Sensitivity of the norm/sink overlap to the percentile that defines "high-norm".

    Also reports the threshold-to-median norm ratio against the Gaussian null: if
    residual components were Gaussian the norm would follow sigma*chi(d), so a
    ratio close to the chi prediction means there is no outlier population to
    find, however many tokens a percentile rule happens to select.
    """
    try:
        from scipy.stats import chi as chi_dist
    except Exception:
        chi_dist = None

    rows: List[dict] = []
    for rec in result.records.values():
        prof = sink_profile(rec)
        if prof is None:
            continue
        inc = prof.numpy()
        d = max(int(rec.d_model), 1)
        med_chi = chi_dist.ppf(0.5, d) if chi_dist is not None else None
        for stage in stages:
            norms_t = rec.norms.get(stage)
            if norms_t is None or norms_t.numel() != rec.n_img:
                continue
            norms = norms_t.numpy()
            med = float(np.median(norms))
            ids = np.arange(rec.n_img)
            for p in percentiles:
                thr_n = float(np.percentile(norms, p))
                thr_a = float(np.percentile(inc, p))
                hn = set(ids[norms > thr_n].tolist())
                sk = set(ids[inc > thr_a].tolist())
                rows.append(dict(
                    prompt_id=rec.prompt_id, seed=rec.seed, step=rec.step,
                    layer=rec.layer, kind=rec.kind, block_name=rec.block_name,
                    stage=stage, percentile=float(p), n_img=rec.n_img,
                    n_highnorm=len(hn), n_sink=len(sk),
                    jaccard=_jaccard(hn, sk),
                    chance=expected_jaccard(max(len(hn), 1), rec.n_img),
                    norm_ratio=thr_n / max(med, 1e-9),
                    gaussian_null_ratio=(float(chi_dist.ppf(p / 100.0, d) / med_chi)
                                         if chi_dist is not None else np.nan),
                ))
    return pd.DataFrame(rows)


def head_sink_membership(result, ratio: Optional[float] = None) -> pd.DataFrame:
    """Fraction of heads whose strongest image key is a high-norm token, per layer."""
    ratio = result.cfg.highnorm_ratio if ratio is None else ratio
    rows: List[dict] = []
    for rec in result.records.values():
        norms = rec.norms.get("post_block")
        if rec.incoming_img2img is None or norms is None:
            continue
        hn = set(torch.nonzero(highnorm_mask(norms, ratio)).flatten().tolist())
        sinks = rec.incoming_img2img.argmax(dim=-1).tolist()
        rows.append(dict(
            layer=rec.layer, kind=rec.kind, block_name=rec.block_name,
            prompt_id=rec.prompt_id, seed=rec.seed, step=rec.step,
            fraction=float(np.mean([int(t) in hn for t in sinks])) if hn else 0.0,
            n_heads=len(sinks),
        ))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.groupby(["layer", "kind", "block_name"], as_index=False)["fraction"].mean()


def register_positions(result, layer: Optional[int] = None,
                       ratio: Optional[float] = None) -> pd.DataFrame:
    """Which token positions are high-norm, per (prompt, seed, step)."""
    ratio = result.cfg.highnorm_ratio if ratio is None else ratio
    if layer is None:
        counts: Dict[int, int] = {}
        for rec in result.records.values():
            norms = rec.norms.get("post_block")
            if norms is not None:
                counts[rec.layer] = counts.get(rec.layer, 0) + int(highnorm_mask(norms, ratio).sum())
        layer = max(counts, key=counts.get) if counts else 0
    rows = []
    for rec in result.records.values():
        if rec.layer != layer:
            continue
        norms = rec.norms.get("post_block")
        if norms is None:
            continue
        ids = torch.nonzero(highnorm_mask(norms, ratio)).flatten().tolist()
        rows.append(dict(prompt_id=rec.prompt_id, seed=rec.seed, step=rec.step,
                         layer=layer, ids=tuple(sorted(int(i) for i in ids))))
    return pd.DataFrame(rows)
