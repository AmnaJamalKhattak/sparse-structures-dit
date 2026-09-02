"""Layer-sweep capture: one generation in, per-layer statistics out.

The design constraint is that we want *every* layer, not a handful, so nothing
of size O(N^2) or O(N*C) may be kept. Attention is reduced to its incoming-mass
profile inside the tap; activations are reduced to per-channel magnitude
profiles inside the hook. What survives per layer per step is a few [H, N] and
[C] vectors, which is small enough to hold a 57-layer model in memory.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

from .adapters import LayerRef, ModelAdapter, extract_image_tokens, image_slice
from .attention_patch import AttentionTap, to_bhsd

STAGES = ("pre_block", "attn_out", "post_attn_residual", "post_block")


@dataclass
class LayerRecord:
    """Compact per-(generation, step, layer) summary."""

    prompt_id: int
    seed: int
    step: int
    timestep: Optional[float]
    layer: int
    kind: str
    local_id: int
    n_img: int
    n_txt: int
    n_heads: int
    d_model: int

    norms: Dict[str, torch.Tensor] = field(default_factory=dict)          # stage -> [N]
    channel_absmax: Optional[torch.Tensor] = None                         # [C]
    channel_mean_abs: Optional[torch.Tensor] = None                       # [C]
    channel_top_ids: Optional[torch.Tensor] = None                        # [M] loudest channels
    channel_top_values: Optional[torch.Tensor] = None                     # [N, M] per-token values

    incoming_img2img: Optional[torch.Tensor] = None                       # [H, N] renormalised
    incoming_raw_img: Optional[torch.Tensor] = None                       # [H, N] raw
    text_mass: Optional[torch.Tensor] = None                              # [H]
    text_sink_strength: Optional[torch.Tensor] = None                     # [H]
    attn_entropy: Optional[torch.Tensor] = None                           # [H] nats
    qk_mean_cosine: Optional[torch.Tensor] = None                         # [H, N], mean image q vs image k
    cross_incoming: Optional[torch.Tensor] = None                         # [H, S_text] PixArt attn2
    cross_entropy: Optional[torch.Tensor] = None                          # [H] nats

    register_ids: Optional[torch.Tensor] = None                           # [k] loudest tokens
    register_vecs: Optional[torch.Tensor] = None                          # [k, C]
    control_ids: Optional[torch.Tensor] = None                            # [k] ordinary tokens
    control_vecs: Optional[torch.Tensor] = None                           # [k, C]

    attn_map: Optional[torch.Tensor] = None                               # [H, P, P] pooled
    attn_map_boundary: Optional[int] = None                               # pooled index of text|image split
    attn_map_scale: Optional[int] = None                                  # tokens per pooled cell

    @property
    def block_name(self) -> str:
        return f"{self.kind}{self.local_id}"

    @property
    def key(self) -> Tuple[int, int, int, int]:
        return (self.prompt_id, self.seed, self.step, self.layer)


class SweepCapture:
    """Attaches to a transformer and records a LayerRecord per layer per step."""

    def __init__(self, adapter: ModelAdapter, cfg, transformer, capture_pass: int = 0):
        self.adapter = adapter
        self.cfg = cfg
        self.transformer = transformer
        self.capture_pass = capture_pass
        self.refs: List[LayerRef] = adapter.layers(transformer)
        self.refs_by_attn: Dict[int, Tuple[LayerRef, str]] = {}
        for ref in self.refs:
            for role, mod in ref.attns.items():
                self.refs_by_attn[id(mod)] = (ref, role)

        self.records: Dict[Tuple[int, int, int, int], LayerRecord] = {}
        self._handles: List[Any] = []
        self._tap: Optional[AttentionTap] = None

        # Per-generation state.
        self.prompt_id = 0
        self.seed = 0
        self.step = -1
        self.pass_idx = 0
        self._last_timestep = None
        self.timestep_value: Optional[float] = None
        self.n_img: Optional[int] = None
        self.n_txt: int = 0
        self._active: Optional[Tuple[LayerRef, str, int]] = None

    # -------------------------------------------------------------- lifecycle
    def begin_generation(self, prompt_id: int, seed: int) -> None:
        self.prompt_id, self.seed = int(prompt_id), int(seed)
        self.step = -1
        self.pass_idx = 0
        self._last_timestep = None
        self.n_img = None
        self.n_txt = 0

    def __enter__(self) -> "SweepCapture":
        self._install()
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def close(self) -> None:
        for h in self._handles:
            try:
                h.remove()
            except Exception:
                pass
        self._handles.clear()
        if self._tap is not None:
            self._tap.__exit__(None, None, None)
            self._tap = None

    # ---------------------------------------------------------------- wiring
    def _resolve_focus_layers(self) -> None:
        if self.cfg.focus_layers or not getattr(self.cfg, "focus_layer_fractions", None):
            return
        n = len(self.refs)
        self.cfg.focus_layers = sorted({
            max(0, min(n - 1, int(round(f * (n - 1))))) for f in self.cfg.focus_layer_fractions
        })

    def _install(self) -> None:
        self._resolve_focus_layers()
        tr = self.transformer
        self._handles.append(tr.register_forward_pre_hook(self._transformer_pre, with_kwargs=True))

        for ref in self.refs:
            if not self.cfg.wants_layer(ref.index):
                continue
            self._handles.append(
                ref.block.register_forward_pre_hook(self._make_block_pre(ref), with_kwargs=True)
            )
            self._handles.append(
                ref.block.register_forward_hook(self._make_block_post(ref), with_kwargs=True)
            )
            norm2 = getattr(ref.block, "norm2", None)
            if norm2 is not None:
                self._handles.append(norm2.register_forward_pre_hook(self._make_norm2_pre(ref)))
            for role, mod in ref.attns.items():
                self._handles.append(
                    mod.register_forward_pre_hook(self._make_attn_pre(ref, role), with_kwargs=True)
                )
                self._handles.append(mod.register_forward_hook(self._make_attn_post(ref, role), with_kwargs=True))

        self._tap = AttentionTap(self.adapter.family, self._on_attention)
        self._tap.__enter__()

    # ------------------------------------------------------------- callbacks
    def _transformer_pre(self, module, args, kwargs):
        ts = kwargs.get("timestep", None)
        tv = _to_float(ts)
        if tv is None or tv != self._last_timestep:
            self.step += 1
            self.pass_idx = 0
            self._last_timestep = tv
        else:
            self.pass_idx += 1
        self.timestep_value = tv

        merged = dict(kwargs)
        if args and "hidden_states" not in merged:
            merged["hidden_states"] = args[0]
        n_img = self.adapter.num_image_tokens(merged, self.cfg)
        if n_img is not None:
            self.n_img = n_img
        self.n_txt = self.adapter.num_text_tokens(merged, self.cfg)
        return None

    @property
    def _recording(self) -> bool:
        return self.cfg.wants_step(self.step) and self.pass_idx == self.capture_pass

    def _batch_index(self, x: torch.Tensor) -> int:
        # With classifier-free guidance the pipeline stacks [uncond, cond];
        # the conditional half is the one that made the image.
        return int(x.shape[0]) - 1

    def _rec(self, ref: LayerRef, n_heads: int = 0, d_model: int = 0) -> LayerRecord:
        key = (self.prompt_id, self.seed, self.step, ref.index)
        rec = self.records.get(key)
        if rec is None:
            rec = LayerRecord(
                prompt_id=self.prompt_id, seed=self.seed, step=self.step,
                timestep=self.timestep_value, layer=ref.index, kind=ref.kind,
                local_id=ref.local_id, n_img=int(self.n_img or 0), n_txt=int(self.n_txt),
                n_heads=n_heads, d_model=d_model,
            )
            self.records[key] = rec
        if n_heads and not rec.n_heads:
            rec.n_heads = n_heads
        if d_model and not rec.d_model:
            rec.d_model = d_model
        return rec

    def _make_block_pre(self, ref: LayerRef):
        def hook(mod, args, kwargs):
            if not self._recording:
                return None
            x = kwargs.get("hidden_states", args[0] if args else None)
            if x is None or not torch.is_tensor(x) or x.ndim != 3:
                return None
            if self.n_img is None:
                self.n_img = int(x.shape[1])
            img = extract_image_tokens(x, self.n_img)
            if img is not None:
                self._store_stage(ref, "pre_block", img)
            return None
        return hook

    def _make_block_post(self, ref: LayerRef):
        def hook(mod, args, kwargs, out):
            if not self._recording:
                return None
            img = extract_image_tokens(out, self.n_img or 0)
            if img is None:
                return None
            rec = self._store_stage(ref, "post_block", img)
            h = img[self._batch_index(img)].float()
            rec.d_model = int(h.shape[-1])
            absh = h.abs()
            rec.channel_absmax = absh.amax(dim=0).cpu()
            rec.channel_mean_abs = absh.mean(dim=0).cpu()
            m = min(int(self.cfg.channel_topk), absh.shape[-1])
            ch_top = torch.topk(rec.channel_absmax, m).indices
            rec.channel_top_ids = ch_top
            rec.channel_top_values = h[:, ch_top.to(h.device)].to(torch.float16).cpu()

            norms = rec.norms["post_block"]
            k = min(self.cfg.register_topk, norms.numel())
            order = torch.argsort(norms, descending=True)
            top = order[:k]
            rec.register_ids = top.cpu()
            rec.register_vecs = h[top.to(h.device)].cpu()

            # Matched ordinary tokens: the null for "do registers share a direction?".
            # Deterministic, drawn from the middle of the norm ranking so they are
            # neither outliers nor the quietest tokens.
            pool = order[k:]
            if pool.numel():
                stride = max(1, pool.numel() // k)
                ctrl = pool[(pool.numel() // 4) :: stride][:k]
                if ctrl.numel():
                    rec.control_ids = ctrl.cpu()
                    rec.control_vecs = h[ctrl.to(h.device)].cpu()
            return None
        return hook

    def _make_norm2_pre(self, ref: LayerRef):
        def hook(mod, inp):
            if not self._recording or not inp:
                return None
            x = inp[0]
            if not torch.is_tensor(x) or x.ndim != 3:
                return None
            img = extract_image_tokens(x, self.n_img or 0)
            if img is not None:
                self._store_stage(ref, "post_attn_residual", img)
            return None
        return hook

    def _make_attn_pre(self, ref: LayerRef, role: str):
        def hook(mod, args, kwargs):
            heads = int(getattr(mod, "heads", 0) or 0)
            self._active = (ref, role, heads)
            return None
        return hook

    def _make_attn_post(self, ref: LayerRef, role: str):
        def hook(mod, args, kwargs, out):
            if self._recording and role == self.adapter.image_attn_role:
                img = extract_image_tokens(out, self.n_img or 0)
                if img is not None:
                    self._store_stage(ref, "attn_out", img)
            self._active = None
            return None
        return hook

    def _store_stage(self, ref: LayerRef, stage: str, img: torch.Tensor) -> LayerRecord:
        rec = self._rec(ref)
        b = self._batch_index(img)
        rec.norms[stage] = img[b].float().norm(dim=-1).cpu()
        return rec

    # ------------------------------------------------------- attention stats
    def _on_attention(self, query, key, value, kwargs=None):
        if self._active is None or not self._recording:
            return
        ref, role, heads = self._active
        if not self.cfg.wants_layer(ref.index):
            return
        try:
            self._attention_stats(ref, role, heads, query, key, kwargs or {})
        finally:
            pass

    @torch.no_grad()
    def _attention_stats(self, ref: LayerRef, role: str, heads: int, query, key, kwargs):
        n_img = int(self.n_img or 0)
        if n_img <= 0:
            return
        q = to_bhsd(query, heads)
        k = to_bhsd(key, heads)
        b = self._batch_index(q)
        q = q[b].float()                         # [H, S_q, D]
        k = k[b].float()                         # [H, S_k, D]
        H, S_q, D = q.shape
        S_k = k.shape[1]

        q_slice = image_slice(S_q, n_img)
        if q_slice is None:
            return                                # queries are not image tokens
        k_slice = image_slice(S_k, n_img)

        scale = kwargs.get("scale", None)
        if scale is None:
            scale = 1.0 / math.sqrt(D)
        mask_row = _key_mask_row(kwargs.get("attn_mask", None), S_k, b)

        qi = q[:, q_slice, :]
        n_q = qi.shape[1]
        chunk = max(1, int(self.cfg.attn_chunk))
        acc = torch.zeros(H, S_k, dtype=torch.float32, device=q.device)
        ent = torch.zeros(H, dtype=torch.float32, device=q.device)
        keep_map = role == self.adapter.image_attn_role and ref.index in set(self.cfg.focus_layers)
        map_rows: List[torch.Tensor] = []

        for start in range(0, n_q, chunk):
            qc = qi[:, start:start + chunk, :]
            scores = torch.matmul(qc, k.transpose(-1, -2)) * scale
            if mask_row is not None:
                scores = scores + mask_row.to(scores.device).view(1, 1, -1)
            probs = torch.softmax(scores, dim=-1)
            acc += probs.sum(dim=1)
            ent += -(probs.clamp_min(1e-12).log() * probs).sum(dim=-1).sum(dim=-1)
            if keep_map:
                map_rows.append(_pool_rows(probs, self.cfg.focus_map_max_tokens, S_k))
            del scores, probs

        incoming_raw = (acc / max(n_q, 1)).cpu()          # [H, S_k]
        entropy = (ent / max(n_q, 1)).cpu()
        rec = self._rec(ref, n_heads=H)

        if role == self.adapter.image_attn_role and k_slice is not None:
            img_in = incoming_raw[:, k_slice]
            rec.incoming_raw_img = img_in
            rec.incoming_img2img = img_in / img_in.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            rec.attn_entropy = entropy
            # Post-QK-normalisation and post-RoPE geometry, exactly as supplied
            # to the model's attention kernel.  This O(H*N) summary is enough
            # to rank natural register keys without retaining q or k tensors.
            q_mean = torch.nn.functional.normalize(qi.mean(dim=1), dim=-1)
            image_keys = torch.nn.functional.normalize(k[:, k_slice, :], dim=-1)
            rec.qk_mean_cosine = torch.einsum("hd,hnd->hn", q_mean, image_keys).cpu()
            n_text_keys = S_k - n_img
            if n_text_keys > 0:
                txt = incoming_raw[:, :n_text_keys]
                rec.text_mass = txt.sum(dim=-1)
                rec.text_sink_strength = txt.amax(dim=-1)
            else:
                rec.text_mass = torch.zeros(H)
                rec.text_sink_strength = torch.zeros(H)
            if keep_map and map_rows:
                pooled_q = _pool_query_axis(torch.cat(map_rows, dim=1), self.cfg.focus_map_max_tokens)
                rec.attn_map = pooled_q
                cell = max(1, math.ceil(S_k / self.cfg.focus_map_max_tokens))
                rec.attn_map_scale = cell
                rec.attn_map_boundary = int((S_k - n_img) // cell) if S_k > n_img else 0
        elif role == "cross":
            # Cross-attention keys are text tokens: this is where a PixArt head
            # can park its attention instead of on an image register.
            rec.cross_incoming = incoming_raw
            rec.cross_entropy = entropy


def _key_mask_row(mask, s_k: int, batch_index: int) -> Optional[torch.Tensor]:
    """Reduce an attention mask to one additive row over keys, or give up.

    diffusers hands SDPA a `[B, H, Q_or_1, S_k]` additive mask for the masked
    cross-attention path. Only key-padding masks (constant over queries) can be
    folded into a per-key row; anything query-dependent is refused rather than
    silently mis-applied, since a wrong mask would quietly bias every number here.
    """
    if not torch.is_tensor(mask):
        return None
    m = mask
    if m.ndim == 4:
        m = m[min(batch_index, m.shape[0] - 1)]          # [H, Q_or_1, S_k]
    if m.ndim == 3:
        m = m[0] if m.shape[0] >= 1 else m               # masks are shared across heads
    if m.ndim == 2:
        if m.shape[-1] != s_k:
            return None
        if m.shape[0] != 1 and not bool((m == m[0]).all()):
            return None                                   # query-dependent: refuse
        m = m[0]
    if m.ndim != 1 or m.shape[0] != s_k:
        return None
    if m.dtype == torch.bool:
        return torch.where(m, torch.zeros_like(m, dtype=torch.float32),
                           torch.full_like(m, float("-inf"), dtype=torch.float32))
    return m.float()


def _pool_rows(probs: torch.Tensor, target: int, s_k: int) -> torch.Tensor:
    """Mean-pool the key axis of [H, c, S_k] down to <= `target` columns."""
    cell = max(1, math.ceil(s_k / target))
    if cell == 1:
        return probs.cpu()
    pad = (-s_k) % cell
    if pad:
        probs = torch.nn.functional.pad(probs, (0, pad))
    H, c, S = probs.shape
    return probs.view(H, c, S // cell, cell).mean(dim=-1).cpu()


def _pool_query_axis(m: torch.Tensor, target: int) -> torch.Tensor:
    """Mean-pool the query axis of [H, Q, P] down to <= `target` rows."""
    H, Q, P = m.shape
    cell = max(1, math.ceil(Q / target))
    if cell == 1:
        return m
    pad = (-Q) % cell
    if pad:
        m = torch.nn.functional.pad(m, (0, 0, 0, pad))
    Q2 = m.shape[1]
    return m.view(H, Q2 // cell, cell, P).mean(dim=2)


def _to_float(x) -> Optional[float]:
    if x is None:
        return None
    if torch.is_tensor(x):
        if x.numel() == 0:
            return None
        return float(x.detach().flatten()[0].cpu())
    try:
        return float(x)
    except Exception:
        return None


class ProjectionCapture:
    """Second, cheap pass: once v* is known, record every token's exact x . v*.

    The atlas sweep can only approximate the projection from the loud-channel
    columns it stores. This pass computes it exactly, at every layer, for the
    cost of one dot product per block -- so the "direction, not magnitude" claim
    is measured rather than estimated.
    """

    def __init__(self, adapter: ModelAdapter, cfg, transformer, direction: torch.Tensor):
        self.adapter = adapter
        self.cfg = cfg
        self.transformer = transformer
        self.v = direction.float().flatten()
        self.v = self.v / self.v.norm().clamp_min(1e-9)
        self.refs = adapter.layers(transformer)
        self._handles: List[Any] = []
        self.rows: Dict[Tuple[int, int, int, int], Dict[str, torch.Tensor]] = {}
        self.prompt_id = 0
        self.seed = 0
        self.step = -1
        self.pass_idx = 0
        self._last_timestep = None
        self.n_img: Optional[int] = None

    def begin_generation(self, prompt_id: int, seed: int) -> None:
        self.prompt_id, self.seed = int(prompt_id), int(seed)
        self.step = -1
        self.pass_idx = 0
        self._last_timestep = None
        self.n_img = None

    def __enter__(self) -> "ProjectionCapture":
        tr = self.transformer
        self._handles.append(tr.register_forward_pre_hook(self._pre, with_kwargs=True))
        for ref in self.refs:
            self._handles.append(ref.block.register_forward_hook(self._make_post(ref), with_kwargs=True))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            try:
                h.remove()
            except Exception:
                pass
        self._handles.clear()
        return False

    def _pre(self, module, args, kwargs):
        tv = _to_float(kwargs.get("timestep", None))
        if tv is None or tv != self._last_timestep:
            self.step += 1
            self.pass_idx = 0
            self._last_timestep = tv
        else:
            self.pass_idx += 1
        merged = dict(kwargs)
        if args and "hidden_states" not in merged:
            merged["hidden_states"] = args[0]
        n_img = self.adapter.num_image_tokens(merged, self.cfg)
        if n_img is not None:
            self.n_img = n_img
        return None

    def _make_post(self, ref: LayerRef):
        def hook(mod, args, kwargs, out):
            if not (self.cfg.wants_step(self.step) and self.pass_idx == 0):
                return None
            if not self.cfg.wants_layer(ref.index):
                return None
            if self.n_img is None:
                # PixArt hands the transformer 4-D latents, so the token count is
                # only visible once a block has run.
                first = out[1] if isinstance(out, (tuple, list)) and len(out) >= 2 else out
                if torch.is_tensor(first) and first.ndim == 3:
                    self.n_img = int(first.shape[1])
            img = extract_image_tokens(out, self.n_img or 0)
            if img is None:
                return None
            h = img[int(img.shape[0]) - 1].float()
            v = self.v.to(h.device)
            self.rows[(self.prompt_id, self.seed, self.step, ref.index)] = dict(
                proj=(h @ v).cpu(),
                norm=h.norm(dim=-1).cpu(),
            )
            return None
        return hook
