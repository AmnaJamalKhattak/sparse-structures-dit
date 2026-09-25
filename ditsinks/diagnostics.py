"""Numerical acceptance gates for causal manipulations."""
from __future__ import annotations

from typing import Dict, Sequence

import torch


def _unit(value: torch.Tensor) -> torch.Tensor:
    return value.float() / value.float().norm(dim=-1, keepdim=True).clamp_min(1e-12)


def assert_norm_clamp(pre: torch.Tensor, post: torch.Tensor, vstar: torch.Tensor,
                      token_ids: Sequence[int], target_norm: float,
                      *, atol: float = 2e-5) -> Dict[str, float]:
    ids = torch.as_tensor(token_ids, dtype=torch.long, device=pre.device)
    v = _unit(vstar.to(pre).reshape(1, -1))[0]
    before_cos = _unit(pre[ids]) @ v
    after_cos = _unit(post[ids]) @ v
    cosine_error = float((before_cos - after_cos).abs().max())
    norm_error = float((post[ids].float().norm(dim=-1) - target_norm).abs().max())
    if cosine_error > atol or norm_error > atol * max(abs(float(target_norm)), 1.0):
        raise AssertionError(f"norm clamp invariant failed: cosine={cosine_error:g}, norm={norm_error:g}")
    return {"max_cosine_error": cosine_error, "max_norm_error": norm_error}


def assert_direction_removed(post: torch.Tensor, vstar: torch.Tensor,
                             token_ids: Sequence[int], *, atol: float = 2e-4) -> float:
    ids = torch.as_tensor(token_ids, dtype=torch.long, device=post.device)
    v = _unit(vstar.to(post).reshape(1, -1))[0]
    projection = float((post[ids].float() @ v).abs().max())
    scale = max(float(post[ids].float().norm(dim=-1).max()), 1.0)
    if projection > atol * scale:
        raise AssertionError(f"direction removal invariant failed: projection={projection:g}")
    return projection


def assert_channel_zero(post: torch.Tensor, channel: int, token_ids: Sequence[int],
                        *, atol: float = 0.0) -> float:
    ids = torch.as_tensor(token_ids, dtype=torch.long, device=post.device)
    maximum = float(post[ids, int(channel)].abs().max())
    if maximum > atol:
        raise AssertionError(f"channel {channel} is not zero after intervention: {maximum:g}")
    return maximum


def assert_identity(clean: torch.Tensor, treated: torch.Tensor, *, name: str,
                    atol: float = 1e-6, rtol: float = 1e-5) -> float:
    if clean.shape != treated.shape:
        raise AssertionError(f"{name} shape differs: {tuple(clean.shape)} != {tuple(treated.shape)}")
    error = float((clean.float() - treated.float()).abs().max())
    if not torch.allclose(clean.float(), treated.float(), atol=atol, rtol=rtol):
        raise AssertionError(f"{name} identity failed: max error={error:g}")
    return error


def token_index_table(*, sequence_length: int, image_tokens: int, text_tokens: int,
                      grid_cols: int) -> list[dict]:
    if sequence_length not in (image_tokens, image_tokens + text_tokens):
        raise ValueError("sequence length is neither image-only nor text+image")
    offset = sequence_length - image_tokens
    rows = []
    for global_id in range(sequence_length):
        image_id = global_id - offset if global_id >= offset else None
        rows.append(dict(global_sequence_index=global_id,
                         token_type="image" if image_id is not None else "text",
                         image_local_index=image_id,
                         spatial_row=None if image_id is None or grid_cols <= 0 else image_id // grid_cols,
                         spatial_col=None if image_id is None or grid_cols <= 0 else image_id % grid_cols))
    return rows

