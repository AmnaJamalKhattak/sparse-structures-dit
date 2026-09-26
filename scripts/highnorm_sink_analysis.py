#!/usr/bin/env python3
"""High-norm token vs. attention-sink analysis for FLUX DiT models.

This script instruments FLUX transformer blocks, captures image-token residual norms
and image-to-image attention probabilities, then writes head-level/token-level CSVs
and diagnostic figures under ``outputs/highnorm_sink_analysis``.

The primary sink definition is token identity based: for each head, the sink token
is the key/source image token with largest mean incoming attention over all image
queries after row-renormalising the image-to-image attention slice.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except Exception:  # pragma: no cover
    plt = None
    sns = None

TEXT_LENGTH = 512


@dataclass
class ExperimentConfig:
    model_name: str = "black-forest-labs/FLUX.1-schnell"
    prompts: List[str] = field(default_factory=lambda: [
        "A close-up portrait of an astronaut in a reflective helmet, cinematic lighting, ultra detailed"
    ])
    seeds: List[int] = field(default_factory=lambda: [42])
    num_inference_steps: int = 4
    capture_timesteps: List[int] = field(default_factory=lambda: [3])
    dual_blocks: List[int] = field(default_factory=lambda: [18])
    single_blocks: List[int] = field(default_factory=lambda: [0])
    text_length: int = TEXT_LENGTH
    topk: List[int] = field(default_factory=lambda: [1, 5, 10])
    highnorm_method: str = "rank_and_mad"
    mad_z: float = 6.0
    output_dir: str = "outputs/highnorm_sink_analysis"
    guidance_scale: float = 0.0
    height: int = 1024
    width: int = 1024
    dtype: str = "bfloat16"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_heatmap_tokens: int = 512
    validate_hooked_generation: bool = False


@dataclass
class BlockCapture:
    block_type: str
    block_id: int
    timestep_index: int
    timestep_value: Optional[float] = None
    pre_attention: Optional[torch.Tensor] = None       # [B, N_img, C]
    post_attention: Optional[torch.Tensor] = None      # [B, N_img, C]
    post_mlp: Optional[torch.Tensor] = None            # [B, N_img, C]
    attn_img2img: Optional[torch.Tensor] = None        # [B, H, Q_img, K_img]
    key: Optional[torch.Tensor] = None                 # [B, H, K_img, D]
    value: Optional[torch.Tensor] = None               # [B, H, K_img, D]
    attn_output: Optional[torch.Tensor] = None         # [B, N_img, C]

    @property
    def name(self) -> str:
        return f"{self.block_type}_block{self.block_id}"


class CaptureStore:
    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg
        self.current_timestep_index = -1
        self.current_timestep_value: Optional[float] = None
        self.records: Dict[str, BlockCapture] = {}
        self.handles: List[Any] = []

    def want_timestep(self) -> bool:
        return self.current_timestep_index in set(self.cfg.capture_timesteps)

    def key(self, block_type: str, block_id: int) -> str:
        return f"{block_type}_block{block_id}"

    def rec(self, block_type: str, block_id: int) -> BlockCapture:
        k = self.key(block_type, block_id)
        if k not in self.records:
            self.records[k] = BlockCapture(block_type, block_id, self.current_timestep_index, self.current_timestep_value)
        rec = self.records[k]
        rec.timestep_index = self.current_timestep_index
        rec.timestep_value = self.current_timestep_value
        return rec

    def close(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()


def detach_cpu(x: torch.Tensor) -> torch.Tensor:
    return x.detach().float().cpu()


def image_slice(x: torch.Tensor, block_type: str, text_length: int) -> torch.Tensor:
    """Return image-token slice for FLUX block outputs shaped [B, N, C]."""
    if block_type == "single" and x.shape[1] > text_length:
        return x[:, text_length:, :]
    return x


def rank_desc(values: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(values, descending=True)
    ranks = torch.empty_like(order)
    ranks[order] = torch.arange(1, values.numel() + 1, device=values.device)
    return ranks


def compute_incoming_attention_mass(attn_img2img: torch.Tensor) -> torch.Tensor:
    """Row-renormalise image-to-image attention and average over query tokens.

    Args:
        attn_img2img: [B, H, Q_img, K_img]
    Returns:
        [B, H, K_img] incoming attention mass per source token and head.
    """
    denom = attn_img2img.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    attn_norm = attn_img2img / denom
    return attn_norm.mean(dim=-2)


def identify_sink_tokens(attn_img2img: torch.Tensor, topk: Sequence[int] = (1, 5, 10)) -> Dict[str, torch.Tensor]:
    incoming = compute_incoming_attention_mass(attn_img2img)
    max_k = max(topk)
    vals, idx = torch.topk(incoming, k=max_k, dim=-1)
    return {"incoming": incoming, "top_values": vals, "top_indices": idx, "sink_token": idx[..., 0], "sink_strength": vals[..., 0]}


def compute_highnorm_tokens(norms: torch.Tensor, topk: Sequence[int], mad_z: float = 6.0) -> Dict[str, Any]:
    max_k = max(topk)
    vals, idx = torch.topk(norms, k=max_k, dim=-1)
    med = norms.median(dim=-1, keepdim=True).values
    mad = (norms - med).abs().median(dim=-1, keepdim=True).values.clamp(min=1e-8)
    robust_mask = norms > (med + mad_z * 1.4826 * mad)
    return {"top_values": vals, "top_indices": idx, "ranks": rank_desc(norms), "robust_mask": robust_mask}


def spearmanr_torch(x: torch.Tensor, y: torch.Tensor) -> float:
    rx, ry = rank_desc(x.float()).float(), rank_desc(y.float()).float()
    rx, ry = rx - rx.mean(), ry - ry.mean()
    den = rx.norm() * ry.norm()
    return float((rx @ ry / den).item()) if den > 0 else float("nan")


def head_entropy(attn_img2img: torch.Tensor) -> torch.Tensor:
    attn = attn_img2img / attn_img2img.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    return -(attn.clamp(min=1e-12).log() * attn).sum(dim=-1).mean(dim=-1)


def attention_weighted_value_contribution(attn_img2img: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    attn = attn_img2img / attn_img2img.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    contrib = attn.mean(dim=-2).unsqueeze(-1) * value.norm(dim=-1)
    return contrib


def pct_attention_to_highnorm(incoming: torch.Tensor, top_ids: torch.Tensor) -> float:
    return float(incoming[top_ids].sum().item() / incoming.sum().clamp(min=1e-8).item())


def ensure_output_dirs(root: Path) -> None:
    for d in ["anchor_attention_heatmaps", "token_norm_plots", "incoming_attention_plots", "scatter_norm_vs_attention",
              "overlap_across_blocks", "sinkiness_heatmaps", "token_trajectory_plots"]:
        (root / d).mkdir(parents=True, exist_ok=True)


def validate_capture(rec: BlockCapture) -> None:
    tensors = [rec.pre_attention, rec.post_attention, rec.post_mlp, rec.attn_img2img]
    assert all(t is not None for t in tensors), f"Incomplete capture for {rec.name}"
    n = rec.pre_attention.shape[1]
    assert rec.attn_img2img.shape[-1] == n and rec.attn_img2img.shape[-2] == n, f"Token mismatch in {rec.name}"
    for name, t in [("pre", rec.pre_attention), ("post_attn", rec.post_attention), ("post_mlp", rec.post_mlp), ("attn", rec.attn_img2img)]:
        assert torch.isfinite(t).all(), f"NaN/Inf in {rec.name}:{name}"
    row_sums = (rec.attn_img2img / rec.attn_img2img.sum(dim=-1, keepdim=True).clamp(min=1e-8)).sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4), f"Renormalisation failed for {rec.name}"


def build_metric_rows(cfg: ExperimentConfig, records: Mapping[str, BlockCapture], prompt_id: int, prompt: str, seed: int) -> Tuple[List[dict], List[dict]]:
    head_rows: List[dict] = []
    token_rows: List[dict] = []
    for rec in records.values():
        validate_capture(rec)
        pre = rec.pre_attention[0].norm(dim=-1); post = rec.post_attention[0].norm(dim=-1); mlp = rec.post_mlp[0].norm(dim=-1)
        norms = {"pre_attention": pre, "post_attention": post, "post_mlp": mlp}
        hn = {k: compute_highnorm_tokens(v, cfg.topk, cfg.mad_z) for k, v in norms.items()}
        sinks = identify_sink_tokens(rec.attn_img2img, cfg.topk)
        incoming_all = sinks["incoming"][0]
        entropy = head_entropy(rec.attn_img2img)[0]
        key_norm = rec.key[0].norm(dim=-1) if rec.key is not None else torch.full_like(incoming_all, float("nan"))
        value_norm = rec.value[0].norm(dim=-1) if rec.value is not None else torch.full_like(incoming_all, float("nan"))
        contrib = attention_weighted_value_contribution(rec.attn_img2img, rec.value)[0] if rec.value is not None else torch.full_like(incoming_all, float("nan"))
        H, N = incoming_all.shape
        for h in range(H):
            incoming = incoming_all[h]
            sink_token = int(sinks["sink_token"][0, h])
            sink_top = {k: set(map(int, sinks["top_indices"][0, h, :k].tolist())) for k in cfg.topk}
            row = dict(model_name=cfg.model_name, prompt_id=prompt_id, prompt=prompt, seed=seed,
                       timestep_index=rec.timestep_index, sigma_or_timestep_value_if_available=rec.timestep_value,
                       block_id=rec.block_id, block_type=rec.block_type, head_id=h, num_image_tokens=N,
                       sink_token_id=sink_token, sink_strength=float(sinks["sink_strength"][0, h]))
            for lname, vals in norms.items():
                top_ids = hn[lname]["top_indices"][:max(cfg.topk)]
                row[f"sink_rank_by_norm_{lname}"] = int(hn[lname]["ranks"][sink_token])
                row[f"top1_norm_token_{lname}"] = int(top_ids[0])
                for k in cfg.topk:
                    high = set(map(int, top_ids[:k].tolist()))
                    row[f"top{k}_overlap_{lname}"] = int(len(high & sink_top.get(k, set())) > 0)
                row[f"spearman_norm_vs_incoming_attention_{lname}"] = spearmanr_torch(vals, incoming)
                row[f"pct_attention_mass_to_top5_highnorm_{lname}"] = pct_attention_to_highnorm(incoming, top_ids[:5])
            row.update(key_norm_sink=float(key_norm[h, sink_token]), value_norm_sink=float(value_norm[h, sink_token]),
                       value_contribution_sink=float(contrib[h, sink_token]), attention_entropy=float(entropy[h]))
            head_rows.append(row)
            incoming_rank = rank_desc(incoming)
            norm_ranks = {lname: rank_desc(vals) for lname, vals in norms.items()}
            for tok in range(N):
                tr = dict(model_name=cfg.model_name, prompt_id=prompt_id, seed=seed, timestep_index=rec.timestep_index,
                          block_id=rec.block_id, block_type=rec.block_type, head_id=h, token_id=tok,
                          pre_attention_norm=float(pre[tok]), post_attention_norm=float(post[tok]), post_mlp_norm=float(mlp[tok]),
                          incoming_attention_mass=float(incoming[tok]), incoming_attention_rank=int(incoming_rank[tok]),
                          norm_rank_pre_attention=int(norm_ranks["pre_attention"][tok]),
                          norm_rank_post_attention=int(norm_ranks["post_attention"][tok]),
                          norm_rank_post_mlp=int(norm_ranks["post_mlp"][tok]),
                          is_sink_top1=tok == sink_token, is_sink_top5=tok in sink_top.get(5, set()))
                for lname in norms:
                    ids = hn[lname]["top_indices"]
                    for k in cfg.topk:
                        tr[f"is_highnorm_top{k}_{lname}"] = tok in set(map(int, ids[:k].tolist()))
                token_rows.append(tr)
    return head_rows, token_rows


def transition_metrics(cfg: ExperimentConfig, token_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, g in token_df.groupby(["model_name", "prompt_id", "seed", "timestep_index", "block_type", "block_id", "head_id"]):
        sink = g[g.is_sink_top1].iloc[0]
        nons = g[~g.is_sink_top1].copy()
        if nons.empty: continue
        nons["norm_dist"] = (nons.pre_attention_norm - sink.pre_attention_norm).abs()
        ctrl = nons.sort_values("norm_dist").iloc[0]
        gs = sink.post_attention_norm - sink.pre_attention_norm
        gc = ctrl.post_attention_norm - ctrl.pre_attention_norm
        rows.append(dict(zip(["model_name","prompt_id","seed","timestep_index","block_type","block_id","head_id"], keys),
                         mean_attention_norm_growth_sink=gs, mean_attention_norm_growth_non_sink=gc,
                         paired_difference=gs-gc, effect_size=(gs-gc)/(nons.post_attention_norm.sub(nons.pre_attention_norm).std() or np.nan),
                         fraction_of_runs_where_sink_grows_more=float(gs > gc), control_type="norm_matched_non_sink"))
    # Dual18 post-MLP -> Single0 later sink tracking.
    for keys, g in token_df.groupby(["model_name", "prompt_id", "seed", "timestep_index"]):
        src = g[(g.block_type == "dual") & (g.block_id == 18) & (g.head_id == g.head_id.min())]
        tgt = g[(g.block_type == "single") & (g.block_id.isin([0, 1]))]
        if src.empty or tgt.empty: continue
        for tok in src.sort_values("norm_rank_post_mlp").head(10).token_id.astype(int):
            tt = tgt[tgt.token_id == tok]
            if tt.empty: continue
            rows.append(dict(model_name=keys[0], prompt_id=keys[1], seed=keys[2], timestep_index=keys[3],
                             source_block="dual_block18", target_block="single_block0/1", token_id=tok,
                             source_post_mlp_norm_rank=int(src[src.token_id == tok].norm_rank_post_mlp.iloc[0]),
                             target_best_incoming_attention_rank=int(tt.incoming_attention_rank.min()),
                             num_heads_where_token_is_top1_sink=int(tt.is_sink_top1.sum()),
                             num_heads_where_token_is_top5_sink=int(tt.is_sink_top5.sum()), control_type="later_sink_tracking"))
    return pd.DataFrame(rows)

# Plotting helpers intentionally consume CSV-friendly data so figures can be regenerated without the model.
def plot_outputs(cfg: ExperimentConfig, head_df: pd.DataFrame, token_df: pd.DataFrame, records: Mapping[str, BlockCapture]) -> None:
    if plt is None: return
    root = Path(cfg.output_dir); ensure_output_dirs(root)
    for rec in records.values():
        attn = (rec.attn_img2img[0] / rec.attn_img2img[0].sum(dim=-1, keepdim=True).clamp(min=1e-8)).numpy()
        H, N, _ = attn.shape; step = max(1, N // cfg.max_heatmap_tokens)
        cols = min(4, H); rows = math.ceil(H / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3.5*rows), squeeze=False)
        for h in range(H):
            ax = axes[h//cols][h%cols]
            ax.imshow(attn[h, ::step, ::step], aspect="auto", origin="lower", cmap="magma")
            ax.set_title(f"{rec.name} head {h}"); ax.set_xlabel("key token"); ax.set_ylabel("query token")
        fig.tight_layout(); fig.savefig(root/"anchor_attention_heatmaps"/f"{rec.name}_heads.png", dpi=180); plt.close(fig)
    # Aggregate figures.
    if not head_df.empty:
        plt.figure(figsize=(10,4)); sns.boxplot(data=head_df, x="block_type", y="sink_strength"); plt.tight_layout(); plt.savefig(root/"sinkiness_heatmaps"/"sinkiness_by_block_type.png", dpi=180); plt.close()
        overlap_cols = [c for c in head_df.columns if c.startswith("top") and "overlap" in c]
        if overlap_cols:
            odf = head_df.melt(id_vars=["block_type","block_id"], value_vars=overlap_cols, var_name="metric", value_name="overlap")
            plt.figure(figsize=(12,5)); sns.barplot(data=odf, x="block_id", y="overlap", hue="metric"); plt.tight_layout(); plt.savefig(root/"overlap_across_blocks"/"overlap_by_block.png", dpi=180); plt.close()
    for (bt,bid,h), g in token_df.groupby(["block_type","block_id","head_id"]):
        if h != 0: continue
        name=f"{bt}_block{bid}_head{h}"
        plt.figure(figsize=(11,4));
        plt.plot(g.token_id, g.incoming_attention_mass, label="incoming attention")
        plt.plot(g.token_id, g.pre_attention_norm/g.pre_attention_norm.max()*g.incoming_attention_mass.max(), label="pre norm scaled")
        plt.plot(g.token_id, g.post_attention_norm/g.post_attention_norm.max()*g.incoming_attention_mass.max(), label="post-attn norm scaled")
        plt.legend(); plt.tight_layout(); plt.savefig(root/"incoming_attention_plots"/f"{name}.png", dpi=180); plt.close()
        for col in ["pre_attention_norm","post_attention_norm","post_mlp_norm"]:
            cls = np.where(g[f"is_highnorm_top5_{col.replace('_norm','')}"] & g.is_sink_top5, "overlap", np.where(g[f"is_highnorm_top5_{col.replace('_norm','')}"], "high-norm only", np.where(g.is_sink_top5, "sink only", "ordinary")))
            plt.figure(figsize=(6,5)); sns.scatterplot(x=g[col], y=g.incoming_attention_mass, hue=cls, s=12, linewidth=0); plt.tight_layout(); plt.savefig(root/"scatter_norm_vs_attention"/f"{name}_{col}.png", dpi=180); plt.close()


def write_readme(cfg: ExperimentConfig, root: Path) -> None:
    (root / "README.md").write_text(f"""# High-norm token vs attention-sink analysis

Model: `{cfg.model_name}`. Prompts: {len(cfg.prompts)}. Seeds: {cfg.seeds}. Capture timesteps: {cfg.capture_timesteps}.
Dual blocks: {cfg.dual_blocks}; single blocks: {cfg.single_blocks}.

## Tensor locations
The script captures image-token residual states before attention, after attention/pre-MLP, and after MLP/block output. For FLUX single blocks the first `{cfg.text_length}` text tokens are removed so token IDs in all metrics refer to image-token IDs.

## Definitions
High-norm tokens are always computed by rank (top-k={cfg.topk}) and additionally by a median+MAD outlier threshold when requested. Attention sinks are source/key image tokens with maximal mean incoming image-to-image attention per head after row-renormalising the image-key slice.

## Outputs
`head_level_metrics.csv` has one row per block/head; `token_level_metrics.csv` has one row per token/head; `transition_metrics.csv` includes sink→norm-growth and Dual18→Single later-sink tracking. Figure folders contain heatmaps, incoming-attention curves, norm/attention scatters, overlap summaries, and sinkiness summaries.

## Rerun
```bash
python scripts/highnorm_sink_analysis.py --config config.json
```
Change `model_name` to `black-forest-labs/FLUX.1-dev` for the main model if compute allows.
""")

# FLUX instrumentation is best-effort across diffusers versions.
def run_flux(cfg: ExperimentConfig) -> Dict[str, BlockCapture]:
    from diffusers import FluxPipeline
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[cfg.dtype]
    pipe = FluxPipeline.from_pretrained(cfg.model_name, torch_dtype=dtype).to(cfg.device)
    store = CaptureStore(cfg)

    def add_block_hooks(blocks: Sequence[torch.nn.Module], block_type: str, wanted: Sequence[int]) -> None:
        for i in wanted:
            block = blocks[i]
            def pre_hook(mod, inp, i=i, block_type=block_type):
                if store.want_timestep(): store.rec(block_type, i).pre_attention = detach_cpu(image_slice(inp[0], block_type, cfg.text_length))
            def post_hook(mod, inp, out, i=i, block_type=block_type):
                if not store.want_timestep(): return
                x = out[0] if isinstance(out, tuple) else out
                store.rec(block_type, i).post_mlp = detach_cpu(image_slice(x, block_type, cfg.text_length))
            store.handles += [block.register_forward_pre_hook(pre_hook), block.register_forward_hook(post_hook)]
            # post-attention/pre-MLP: hook first feed-forward or norm2 input when available.
            for name in ["norm2", "ff", "ff_context", "feed_forward"]:
                sub = getattr(block, name, None)
                if sub is not None:
                    def pre_mlp_hook(mod, inp, i=i, block_type=block_type):
                        if store.want_timestep(): store.rec(block_type, i).post_attention = detach_cpu(image_slice(inp[0], block_type, cfg.text_length))
                    store.handles.append(sub.register_forward_pre_hook(pre_mlp_hook)); break

    transformer = pipe.transformer
    add_block_hooks(transformer.transformer_blocks, "dual", cfg.dual_blocks)
    add_block_hooks(transformer.single_transformer_blocks, "single", cfg.single_blocks)
    # Force eager attention probabilities if supported.
    if hasattr(transformer, "set_attn_processor"):
        # Native attention processors do not expose probabilities through a forward
        # hook. Install a processor here that records attention weights, key and
        # value on rec.attn_img2img/key/value; missing captures raise below.
        pass
    def cb(pipe, step_index, timestep, callback_kwargs):
        store.current_timestep_index = int(step_index); store.current_timestep_value = float(timestep) if torch.is_tensor(timestep) else float(timestep)
        return callback_kwargs
    for prompt_id, prompt in enumerate(cfg.prompts):
        for seed in cfg.seeds:
            gen = torch.Generator(device=cfg.device).manual_seed(seed)
            pipe(prompt=prompt, height=cfg.height, width=cfg.width, num_inference_steps=cfg.num_inference_steps,
                 guidance_scale=cfg.guidance_scale, generator=gen, callback_on_step_begin=cb)
    store.close()
    missing = [r.name for r in store.records.values() if r.attn_img2img is None]
    if missing:
        raise RuntimeError("Residual hooks ran, but attention probabilities were not captured. Install/adapt the notebook's AttentionCapturingProcessor for this diffusers version and assign rec.attn_img2img/key/value in the processor. Missing: " + ", ".join(missing))
    return store.records


def run_synthetic(cfg: ExperimentConfig) -> Dict[str, BlockCapture]:
    torch.manual_seed(0); records = {}
    for bt, bids in [("dual", cfg.dual_blocks), ("single", cfg.single_blocks)]:
        for bid in bids:
            B,H,N,C,D = 1,4,128,64,16
            rec=BlockCapture(bt,bid,cfg.capture_timesteps[0],0.0)
            rec.pre_attention=torch.randn(B,N,C); rec.post_attention=rec.pre_attention+0.05*torch.randn(B,N,C); rec.post_mlp=rec.post_attention+0.05*torch.randn(B,N,C)
            sink=7 if bt=="single" else 31
            attn=torch.rand(B,H,N,N)*0.01
            if bt=="single": attn[:,:,:,sink]+=0.25
            attn=attn/attn.sum(-1,keepdim=True)
            rec.attn_img2img=attn; rec.key=torch.randn(B,H,N,D); rec.value=torch.randn(B,H,N,D); rec.attn_output=torch.randn(B,N,C)
            records[rec.name]=rec
    return records


def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--config"); p.add_argument("--synthetic", action="store_true", help="Run fast synthetic validation without downloading FLUX.")
    args=p.parse_args(); cfg=ExperimentConfig()
    if args.config:
        data=json.loads(Path(args.config).read_text()); cfg=ExperimentConfig(**{**asdict(cfg), **data})
    root=Path(cfg.output_dir); root.mkdir(parents=True, exist_ok=True); ensure_output_dirs(root)
    (root/"config.json").write_text(json.dumps(asdict(cfg), indent=2))
    records = run_synthetic(cfg) if args.synthetic else run_flux(cfg)
    all_head=[]; all_tok=[]
    # Current runner stores one prompt/seed minimal captures; metrics schema supports expansion.
    for pid, prompt in enumerate(cfg.prompts[:1]):
        for seed in cfg.seeds[:1]:
            h,t=build_metric_rows(cfg, records, pid, prompt, seed); all_head+=h; all_tok+=t
    head_df=pd.DataFrame(all_head); token_df=pd.DataFrame(all_tok); trans_df=transition_metrics(cfg, token_df)
    head_df.to_csv(root/"head_level_metrics.csv", index=False); token_df.to_csv(root/"token_level_metrics.csv", index=False); trans_df.to_csv(root/"transition_metrics.csv", index=False)
    plot_outputs(cfg, head_df, token_df, records); write_readme(cfg, root)
    print("Validation: captured tensors finite; token counts aligned; image attention row-renormalisation asserted.")
    print(f"Wrote {len(head_df)} head rows, {len(token_df)} token rows, {len(trans_df)} transition rows to {root}")

if __name__ == "__main__":
    main()
