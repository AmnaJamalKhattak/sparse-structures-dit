"""Tiny real transformers for smoke-testing the sweep without weights.

These are the *actual* diffusers block classes, instantiated at toy width and
driven with random inputs, so hook signatures, sequence layouts and the
attention tap are exercised for real. Only the weights are fake.

A few image tokens are pushed hard along a fixed direction that is concentrated
on one channel, and the layers are wired so that direction survives. That gives
the figures a known ground truth: if the analysis cannot recover the planted
direction here, it will not find a real one.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch


def _dtype(name: str) -> torch.dtype:
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[name]


@dataclass
class TinyModelBundle:
    """Stands in for a diffusers pipeline in the sweep runner."""

    transformer: Any
    family: str
    n_img: int
    n_txt: int
    grid: Tuple[int, int]
    steps: int
    call: Any                       # (transformer, seed, step, register_dir) -> None
    d_model: int
    n_heads: int


def build_tiny(
    family: str,
    steps: int = 2,
    grid: int = 8,
    n_txt: int = 6,
    n_dual: int = 2,
    n_single: int = 3,
    heads: int = 2,
    head_dim: int = 8,
    seed: int = 0,
    device: str = "cpu",
) -> TinyModelBundle:
    torch.manual_seed(seed)
    n_img = grid * grid
    dim = heads * head_dim

    if family == "flux1":
        from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel

        tr = FluxTransformer2DModel(
            patch_size=1, in_channels=dim, num_layers=n_dual, num_single_layers=n_single,
            attention_head_dim=head_dim, num_attention_heads=heads,
            joint_attention_dim=dim, pooled_projection_dim=dim,
            axes_dims_rope=(head_dim // 4 * 2, head_dim // 4, head_dim // 4),
        )
        n_layers = n_dual + n_single

        def call(transformer, gen_seed, step, register_dir):
            g = torch.Generator().manual_seed(gen_seed * 1000 + step)
            hs = torch.randn(1, n_img, dim, generator=g)
            _plant(hs, register_dir, gen_seed)
            ehs = torch.randn(1, n_txt, dim, generator=g)
            img_ids = _flux_ids(grid)
            txt_ids = torch.zeros(n_txt, 3)
            transformer(
                hidden_states=hs, encoder_hidden_states=ehs,
                pooled_projections=torch.randn(1, dim, generator=g),
                timestep=torch.tensor([1.0 - step / max(steps, 1)]),
                img_ids=img_ids, txt_ids=txt_ids, return_dict=False,
            )

    elif family == "flux2":
        from diffusers.models.transformers.transformer_flux2 import Flux2Transformer2DModel

        tr = Flux2Transformer2DModel(
            patch_size=1, in_channels=dim, num_layers=n_dual, num_single_layers=n_single,
            attention_head_dim=head_dim, num_attention_heads=heads,
            joint_attention_dim=dim, timestep_guidance_channels=16, mlp_ratio=2.0,
            axes_dims_rope=(head_dim // 4, head_dim // 4, head_dim // 4, head_dim // 4),
            rope_theta=2000, guidance_embeds=True,
        )
        n_layers = n_dual + n_single

        def call(transformer, gen_seed, step, register_dir):
            g = torch.Generator().manual_seed(gen_seed * 1000 + step)
            hs = torch.randn(1, n_img, dim, generator=g)
            _plant(hs, register_dir, gen_seed)
            ehs = torch.randn(1, n_txt, dim, generator=g)
            transformer(
                hidden_states=hs, encoder_hidden_states=ehs,
                timestep=torch.tensor([1.0 - step / max(steps, 1)]),
                img_ids=_flux2_ids(grid), txt_ids=torch.zeros(n_txt, 4),
                guidance=torch.tensor([1.0]), return_dict=False,
            )

    elif family == "pixart":
        from diffusers.models.transformers.pixart_transformer_2d import PixArtTransformer2DModel

        n_layers = n_dual + n_single
        tr = PixArtTransformer2DModel(
            num_attention_heads=heads, attention_head_dim=head_dim, in_channels=4,
            out_channels=8, num_layers=n_layers, cross_attention_dim=dim,
            sample_size=grid * 2, patch_size=2, caption_channels=dim,
            norm_type="ada_norm_single", num_embeds_ada_norm=1000,
        )

        def call(transformer, gen_seed, step, register_dir):
            g = torch.Generator().manual_seed(gen_seed * 1000 + step)
            lat = torch.randn(2, 4, grid * 2, grid * 2, generator=g)   # CFG-style batch of 2
            enc = torch.randn(2, n_txt, dim, generator=g)
            transformer(
                hidden_states=lat, encoder_hidden_states=enc,
                timestep=torch.tensor([100.0 - 10 * step] * 2),
                added_cond_kwargs={"resolution": None, "aspect_ratio": None},
                encoder_attention_mask=torch.ones(2, n_txt),
                return_dict=False,
            )

    else:
        raise KeyError(f"unknown family {family!r}")

    tr = tr.to(device).eval()
    for p in tr.parameters():
        p.requires_grad_(False)

    return TinyModelBundle(
        transformer=tr, family=family, n_img=n_img, n_txt=(0 if family == "pixart" else n_txt),
        grid=(grid, grid), steps=steps, call=call, d_model=dim, n_heads=heads,
    )


def planted_direction(d_model: int, channel: int = 3, seed: int = 7) -> torch.Tensor:
    """A fixed unit vector whose energy is dominated by one channel."""
    g = torch.Generator().manual_seed(seed)
    v = 0.15 * torch.randn(d_model, generator=g)
    v[channel % d_model] = 3.0
    return v / v.norm()


def _plant(hs: torch.Tensor, direction: Optional[torch.Tensor], gen_seed: int, n: int = 3) -> None:
    """Make a few tokens loud along `direction` (the synthetic 'registers')."""
    if direction is None:
        return
    n_img = hs.shape[1]
    idx = [(7 + 11 * i + gen_seed) % n_img for i in range(n)]
    for t in idx:
        hs[0, t, :] = direction.to(hs.dtype) * 12.0


def _flux_ids(grid: int) -> torch.Tensor:
    yy, xx = torch.meshgrid(torch.arange(grid), torch.arange(grid), indexing="ij")
    ids = torch.zeros(grid * grid, 3)
    ids[:, 1] = yy.reshape(-1).float()
    ids[:, 2] = xx.reshape(-1).float()
    return ids


def _flux2_ids(grid: int) -> torch.Tensor:
    yy, xx = torch.meshgrid(torch.arange(grid), torch.arange(grid), indexing="ij")
    ids = torch.zeros(grid * grid, 4)
    ids[:, 2] = yy.reshape(-1).float()
    ids[:, 3] = xx.reshape(-1).float()
    return ids
