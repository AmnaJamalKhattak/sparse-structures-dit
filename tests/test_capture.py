"""Correctness tests that need no weights and no GPU.

They instantiate the real diffusers block classes at toy width, so a signature
change in diffusers breaks these tests rather than a 40-minute GPU run.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ditsinks import SweepConfig, run_sweep                    # noqa: E402
from ditsinks.adapters import get_adapter, image_slice          # noqa: E402
from ditsinks.attention_patch import AttentionTap, to_bhsd      # noqa: E402
from ditsinks.registry import resolve_model                     # noqa: E402
from ditsinks.synthetic import build_tiny, planted_direction    # noqa: E402

FAMILIES = ["flux1", "flux2", "pixart"]


def _cfg(family: str, **kw) -> SweepConfig:
    base = dict(model=f"tiny-{family}", prompts=["a", "b"], seeds=[0, 1],
                num_inference_steps=2, capture_steps=[0, 1], height=128, width=128,
                focus_layers=[0])
    base.update(kw)
    return SweepConfig(**base)


# --------------------------------------------------------------------- basics
def test_registry_aliases():
    assert resolve_model("flux2-schnell").key == "flux2-klein"
    assert resolve_model("black-forest-labs/FLUX.1-schnell").family == "flux1"
    with pytest.raises(KeyError):
        resolve_model("not-a-model")


def test_image_slice_rules():
    assert image_slice(64, 64) == slice(0, 64)      # image-only stream
    assert image_slice(70, 64) == slice(6, 70)      # [text, image]
    assert image_slice(6, 64) is None               # text-only key axis


def test_to_bhsd_layouts():
    bshd = torch.zeros(1, 40, 4, 8)                 # FLUX layout
    bhsd = torch.zeros(1, 4, 40, 8)                 # Attention layout
    assert to_bhsd(bshd, heads=4).shape == (1, 4, 40, 8)
    assert to_bhsd(bhsd, heads=4).shape == (1, 4, 40, 8)


# ------------------------------------------------------------- the tap is inert
@pytest.mark.parametrize("family", FAMILIES)
def test_attention_tap_does_not_change_the_model(family):
    """The whole design rests on this: observing must not perturb generation."""
    bundle = build_tiny(family, steps=1, grid=6)
    v = planted_direction(bundle.d_model)
    outs = []
    for use_tap in (False, True):
        captured = []
        torch.manual_seed(0)
        if use_tap:
            with AttentionTap(family, lambda q, k, val, kw=None: captured.append(1)):
                with torch.no_grad():
                    out = _forward_once(bundle, v)
            assert captured, "tap never fired"
        else:
            with torch.no_grad():
                out = _forward_once(bundle, v)
        outs.append(out)
    assert torch.equal(outs[0], outs[1]), "the attention tap changed the model output"


def _forward_once(bundle, v) -> torch.Tensor:
    grabbed = {}

    def hook(mod, args, kwargs, out):
        grabbed["out"] = out[1] if isinstance(out, (tuple, list)) else out

    blocks = bundle.transformer.transformer_blocks
    h = blocks[-1].register_forward_hook(hook, with_kwargs=True)
    try:
        bundle.call(bundle.transformer, 0, 0, v)
    finally:
        h.remove()
    return grabbed["out"].detach().clone()


# ------------------------------------------------------------------- capture
@pytest.mark.parametrize("family", FAMILIES)
def test_sweep_captures_every_layer_and_step(family):
    cfg = _cfg(family)
    res = run_sweep(cfg, progress=False)
    n_layers = res.meta["n_layers"]
    expected = len(cfg.prompts) * len(cfg.seeds) * len(cfg.capture_steps) * n_layers
    assert len(res.records) == expected
    assert res.meta["attention_tap"], "no attention entry point was patched"

    for rec in res.records.values():
        assert rec.n_img > 0 and rec.n_heads > 0 and rec.d_model > 0
        # every layer must expose the residual stream on both sides
        assert "pre_block" in rec.norms and "post_block" in rec.norms
        assert rec.norms["post_block"].shape == (rec.n_img,)
        assert rec.incoming_img2img is not None
        assert rec.incoming_img2img.shape == (rec.n_heads, rec.n_img)
        sums = rec.incoming_img2img.sum(-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4)
        assert rec.channel_absmax.shape == (rec.d_model,)
        assert rec.register_vecs.shape[1] == rec.d_model
        assert torch.isfinite(rec.incoming_img2img).all()
        assert torch.isfinite(rec.channel_absmax).all()


@pytest.mark.parametrize("family", FAMILIES)
def test_text_mass_is_consistent_with_family(family):
    res = run_sweep(_cfg(family), progress=False)
    rec = next(iter(res.records.values()))
    if family == "pixart":
        # PixArt image self-attention never sees text keys.
        assert float(rec.text_mass.max()) == 0.0
        assert rec.cross_incoming is not None and rec.cross_incoming.shape[0] == rec.n_heads
    else:
        assert 0.0 < float(rec.text_mass.mean()) < 1.0
        raw_total = rec.incoming_raw_img.sum(-1) + rec.text_mass
        assert torch.allclose(raw_total, torch.ones_like(raw_total), atol=2e-3)


@pytest.mark.parametrize("family", FAMILIES)
def test_focus_layer_attention_maps(family):
    cfg = _cfg(family, focus_layers=[0, 2], focus_map_max_tokens=32)
    res = run_sweep(cfg, progress=False)
    maps = [r for r in res.records.values() if r.attn_map is not None]
    assert maps, "no focus attention map was stored"
    assert {r.layer for r in maps} == {0, 2}
    for r in maps:
        assert r.attn_map.ndim == 3 and r.attn_map.shape[0] == r.n_heads
        assert r.attn_map.shape[1] <= cfg.focus_map_max_tokens
        assert r.attn_map.shape[2] <= cfg.focus_map_max_tokens
        if r.n_txt:
            assert r.attn_map_boundary > 0


def test_batch_index_reads_the_conditional_half():
    """PixArt runs real CFG as [uncond, cond]; the analysis must read `cond`."""
    from ditsinks.capture import SweepCapture

    bundle = build_tiny("pixart", steps=1, grid=6)
    cap = SweepCapture(get_adapter("pixart"), _cfg("pixart"), bundle.transformer)
    x = torch.zeros(2, 10, 4)
    assert cap._batch_index(x) == 1


# --------------------------------------------------------------- mask handling
def test_key_mask_row_folds_padding_masks_and_refuses_causal_ones():
    from ditsinks.capture import _key_mask_row

    additive = torch.zeros(2, 4, 1, 6)
    additive[1, :, :, 4:] = -10000.0
    row = _key_mask_row(additive, 6, batch_index=1)
    assert row.tolist() == [0.0, 0.0, 0.0, 0.0, -10000.0, -10000.0]

    boolean = torch.tensor([True, True, False])
    assert _key_mask_row(boolean, 3, 0).tolist() == [0.0, 0.0, float("-inf")]

    # A query-dependent mask cannot be folded into one row, so it must be refused
    # rather than silently applied to every query.
    causal = torch.tril(torch.ones(5, 5)).unsqueeze(0).unsqueeze(0)
    assert _key_mask_row(causal, 5, 0) is None
    assert _key_mask_row(None, 5, 0) is None
    assert _key_mask_row(torch.zeros(2, 4, 1, 9), 6, 0) is None   # wrong key length
