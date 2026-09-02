"""A synthetic SweepResult that *looks like* a real FLUX/PixArt sweep.

Purpose: render and sanity-check every figure without a GPU, and let a reader
see what the analysis produces before spending an hour of compute. The numbers
here are invented -- shaped to match published FLUX behaviour (registers born in
the late dual blocks, sinks switching on at the dual->single boundary, one
dominant activation channel) -- and are never used for any claim.

`ditsinks.synthetic` is the opposite tool: real diffusers blocks, real hooks,
fake weights. Use that one to test the capture code.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

from .capture import LayerRecord
from .config import SweepConfig
from .runner import SweepResult

MOCK_BANNER = "MOCK DATA - shapes and figures only, not a measurement"


def make_mock_result(
    n_dual: int = 19,
    n_single: int = 38,
    n_img: int = 1024,
    d_model: int = 3072,
    n_heads: int = 24,
    n_txt: int = 512,
    prompts: Sequence[str] = ("mock prompt A", "mock prompt B"),
    seeds: Sequence[int] = (0, 1),
    steps: Sequence[int] = (3,),
    register_channel: int = 154,
    birth_layer: int = 17,
    sink_layer: int = 19,
    decay_layer: int = 40,
    focus_layers: Sequence[int] = (30,),
    seed: int = 0,
) -> SweepResult:
    rng = np.random.default_rng(seed)
    n_layers = n_dual + n_single
    grid = int(round(np.sqrt(n_img)))

    v_star = _make_direction(d_model, register_channel, rng)

    cfg = SweepConfig(
        model="flux1-schnell", prompts=list(prompts), seeds=list(seeds),
        num_inference_steps=max(steps) + 1, capture_steps=list(steps),
        height=grid * 16, width=grid * 16, focus_layers=list(focus_layers),
        save_images=False,
    )
    result = SweepResult(cfg=cfg)
    result.meta.update(
        model="mock-flux1", repo_id="mock", family="flux1", mock=True, banner=MOCK_BANNER,
        n_layers=n_layers, grid=(grid, grid), planted_direction=torch.tensor(v_star, dtype=torch.float32),
        register_channel=register_channel,
        layer_names=[f"dual{i}" for i in range(n_dual)] + [f"single{i}" for i in range(n_single)],
        layer_kinds=["dual"] * n_dual + ["single"] * n_single,
    )

    for pid in range(len(prompts)):
        for sd in seeds:
            reg_ids = _register_ids(rng, n_img, pid, sd)
            for st in steps:
                for layer in range(n_layers):
                    result.records[(pid, int(sd), int(st), layer)] = _mock_layer(
                        rng, pid, int(sd), int(st), layer, n_dual, n_img, d_model, n_heads,
                        n_txt, reg_ids, v_star, register_channel, birth_layer, sink_layer,
                        decay_layer, focus_layers,
                    )
    return result


# ------------------------------------------------------------------ internals
def _make_direction(d_model: int, ch: int, rng) -> np.ndarray:
    v = 0.02 * rng.standard_normal(d_model)
    for c, w in ((ch, 1.0), ((ch + 1292) % d_model, 0.42), ((ch + 1677) % d_model, 0.31),
                 ((ch + 2412) % d_model, 0.22)):
        v[c] = w
    return v / np.linalg.norm(v)


def _register_ids(rng, n_img: int, pid: int, sd: int) -> np.ndarray:
    grid = int(round(np.sqrt(n_img)))
    base = np.array([3 * grid + 2, 17 * grid + grid - 3, 28 * grid + 15])
    return (base + 7 * pid + 3 * sd) % n_img


def _envelope(layer: int, birth: int, decay: int) -> float:
    """0 before birth, ramps up, plateaus, fades after `decay`."""
    if layer < birth:
        return 0.0
    ramp = min(1.0, (layer - birth + 1) / 5.0)
    fade = 1.0 if layer <= decay else max(0.05, 1.0 - (layer - decay) / 12.0)
    return ramp * fade


def _mock_layer(rng, pid, sd, st, layer, n_dual, n_img, d_model, n_heads, n_txt,
                reg_ids, v_star, ch, birth, sink_layer, decay, focus_layers) -> LayerRecord:
    kind = "dual" if layer < n_dual else "single"
    local = layer if kind == "dual" else layer - n_dual
    env = _envelope(layer, birth, decay)

    # ---- bulk residual norms grow smoothly with depth (matches published curves)
    bulk = 900.0 * np.exp(0.045 * layer) * (1.0 + 0.06 * rng.standard_normal(n_img))
    norms = np.clip(bulk, 1.0, None)
    norms[reg_ids] = norms[reg_ids] * (1.0 + 28.0 * env)
    norms_t = torch.tensor(norms, dtype=torch.float32)

    pre = torch.tensor(norms * (0.92 + 0.02 * rng.standard_normal(n_img)), dtype=torch.float32)
    attn_out = torch.tensor(np.abs(norms * 0.12 * (1 + 0.3 * rng.standard_normal(n_img))), dtype=torch.float32)

    # ---- channels: one massive channel switches on with the registers
    cam = np.abs(rng.standard_normal(d_model)) * 12.0 * np.exp(0.03 * layer) + 20.0
    cam[ch] += 60.0 * env * np.exp(0.05 * layer) * 12.0
    for extra in ((ch + 1292) % d_model, (ch + 1677) % d_model):
        cam[extra] += 18.0 * env * np.exp(0.05 * layer) * 12.0
    cam_t = torch.tensor(cam, dtype=torch.float32)
    top_ids = torch.topk(cam_t, 16).indices
    ch_vals = torch.tensor(rng.standard_normal((n_img, 16)) * 8.0, dtype=torch.float32)
    for j, c in enumerate(top_ids.tolist()):
        if c == ch or c in ((ch + 1292) % d_model, (ch + 1677) % d_model):
            spike = float(cam[c]) * (0.85 + 0.1 * rng.standard_normal(len(reg_ids)))
            ch_vals[torch.tensor(reg_ids, dtype=torch.long), j] = torch.tensor(spike, dtype=torch.float32)

    # ---- attention: sinks switch on at the dual->single boundary
    sink_env = _envelope(layer, sink_layer, decay + 8)
    inc = rng.random((n_heads, n_img)) * (0.6 / n_img) + 0.4 / n_img
    head_strength = 0.55 * sink_env * (0.5 + 0.5 * rng.random(n_heads))
    for h in range(n_heads):
        share = head_strength[h]
        if share > 0:
            inc[h] *= (1.0 - share)
            weights = rng.random(len(reg_ids)) + 0.6
            weights = weights / weights.sum()
            for j, t in enumerate(reg_ids):
                inc[h, t] += share * weights[j]
    inc = inc / inc.sum(axis=1, keepdims=True)
    inc_t = torch.tensor(inc, dtype=torch.float32)
    qk = rng.normal(0.0, 0.08, (n_heads, n_img))
    for t in reg_ids:
        qk[:, int(t)] += 0.75 * sink_env
    qk_t = torch.tensor(qk, dtype=torch.float32)

    text_frac = 0.35 * (1.0 - 0.7 * sink_env) if kind == "single" else 0.45
    text_mass = torch.tensor(np.clip(text_frac + 0.05 * rng.standard_normal(n_heads), 0, 0.95),
                             dtype=torch.float32)
    ent = torch.tensor(np.log(n_img) * (0.98 - 0.35 * sink_env) + 0.02 * rng.standard_normal(n_heads),
                       dtype=torch.float32)

    # ---- register vectors point along v* once the direction exists
    k = min(8, n_img)
    order = torch.topk(norms_t, k).indices
    vecs = torch.tensor(rng.standard_normal((k, d_model)) * 30.0, dtype=torch.float32)
    for i, t in enumerate(order.tolist()):
        if t in set(int(x) for x in reg_ids) and env > 0:
            mag = float(norms[t])
            purity = 0.55 + 0.42 * min(1.0, env)
            noise = rng.standard_normal(d_model)
            noise /= np.linalg.norm(noise)
            direction = purity * v_star + (1 - purity) * noise
            direction /= np.linalg.norm(direction)
            vecs[i] = torch.tensor(direction * mag, dtype=torch.float32)

    ctrl_ids = torch.topk(norms_t, 40).indices[-8:]
    ctrl_vecs = torch.tensor(rng.standard_normal((8, d_model)), dtype=torch.float32)
    ctrl_vecs = ctrl_vecs / ctrl_vecs.norm(dim=-1, keepdim=True) * norms_t[ctrl_ids].unsqueeze(-1)

    rec = LayerRecord(
        prompt_id=pid, seed=sd, step=st, timestep=float(1000 - 200 * st), layer=layer,
        kind=kind, local_id=local, n_img=n_img, n_txt=n_txt, n_heads=n_heads, d_model=d_model,
        norms={"pre_block": pre, "attn_out": attn_out, "post_block": norms_t},
        channel_absmax=cam_t, channel_mean_abs=cam_t / 40.0,
        channel_top_ids=top_ids, channel_top_values=ch_vals.to(torch.float16),
        incoming_img2img=inc_t, incoming_raw_img=inc_t * (1 - text_mass).unsqueeze(-1),
        qk_mean_cosine=qk_t,
        text_mass=text_mass, text_sink_strength=text_mass / max(n_txt, 1) * 40,
        attn_entropy=ent, register_ids=order, register_vecs=vecs,
        control_ids=ctrl_ids, control_vecs=ctrl_vecs,
    )
    if kind == "dual":
        rec.norms["post_attn_residual"] = norms_t * 0.96

    if layer in set(focus_layers):
        rec.attn_map, rec.attn_map_boundary, rec.attn_map_scale = _mock_attn_map(
            rng, n_heads, n_img, n_txt, reg_ids, sink_env, target=128
        )
    return rec


def _mock_attn_map(rng, n_heads, n_img, n_txt, reg_ids, sink_env, target=128):
    s_k = n_txt + n_img
    cell = max(1, int(np.ceil(s_k / target)))
    p = int(np.ceil(s_k / cell))
    q_cell = max(1, int(np.ceil(n_img / target)))
    q = int(np.ceil(n_img / q_cell))
    m = rng.random((n_heads, q, p)) * 0.15 / p + 0.2 / p
    boundary = int(n_txt // cell)
    for h in range(n_heads):
        for i in range(q):                                   # local/diagonal structure
            j = boundary + int(i * q_cell / cell)
            if 0 <= j < p:
                m[h, i, j] += 1.4 / p
        for t in reg_ids:                                    # the sink columns
            j = boundary + int(t / cell)
            if 0 <= j < p:
                m[h, :, j] += (3.5 + 5 * rng.random()) * sink_env / p
        m[h, :, : max(boundary, 1)] += 0.9 / p               # text/padding sink band
        m[h] /= m[h].sum(axis=1, keepdims=True)
    return torch.tensor(m, dtype=torch.float32), boundary, cell
