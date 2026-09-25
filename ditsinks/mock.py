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

import math
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
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


# ---------------------------------------------------------------- causal mock
CAUSAL_MOCK_BANNER = ("MOCK CAUSAL RESULT - shapes, figures and statistics only. "
                      "These numbers are invented and support no claim.")


def make_mock_question_results(prompts: int = 8, seeds: int = 3, first_layer: int = 20,
                               last_layer: int = 40, rng_seed: int = 0):
    """Q1--Q6 result bundles with the structure of a real run, but invented numbers.

    Two uses: render and check every causal figure without a GPU, and let a reader
    see the shape of the answer before spending compute.  The effect sizes below
    are drawn from what the workshop paper already reports, so the mock looks like
    a plausible result rather than noise -- which is exactly why it must never be
    mistaken for one.  Every returned bundle is tagged in ``meta['mock']``.
    """
    from .questions import (Q1_CONDITIONS, Q2_TARGET_LABELS, Q5_POSITION_LABELS, Q5_STAGES,
                            Q6_CONDITIONS, QuestionResult, _q1_verdict, _q2_verdict, _q3_verdict,
                            _q4_verdict, _q5_verdict, _q6_verdict)

    rng = np.random.default_rng(rng_seed)
    layers = list(range(first_layer, last_layer + 1, 2))
    units = [(p, s) for p in range(prompts) for s in range(seeds)]
    noise = lambda scale=0.03: float(rng.normal(0, scale))

    # ---- Q1 -----------------------------------------------------------------
    retention = {"sham": 0.95, "direction_removal": 0.24, "matched_ordinary_state": 0.19,
                 "ordinary_norm_clamp": 0.86, "state_zeroed": 0.12,
                 "random_tokens_zeroed": 0.93, "norm_matched_tokens_zeroed": 0.90,
                 "offregister_direction_removal": 0.91,
                 "normmatched_direction_removal": 0.88}
    projection = {"sham": 0.0, "direction_removal": -2.4, "matched_ordinary_state": -2.6,
                  "ordinary_norm_clamp": -0.3, "state_zeroed": -3.1,
                  "random_tokens_zeroed": -0.1, "norm_matched_tokens_zeroed": -0.15,
                  "offregister_direction_removal": -0.2,
                  "normmatched_direction_removal": -0.25}
    fate_mix = {"sham": ("same_position", 0.95), "direction_removal": ("relocated", 0.62),
                "matched_ordinary_state": ("relocated", 0.55),
                "ordinary_norm_clamp": ("same_position", 0.88),
                "state_zeroed": ("diffuse", 0.66), "random_tokens_zeroed": ("same_position", 0.94),
                "norm_matched_tokens_zeroed": ("same_position", 0.92),
                "offregister_direction_removal": ("same_position", 0.9),
                "normmatched_direction_removal": ("same_position", 0.87)}
    # (alignment, norm ratio) relative to a clean register, so each condition lands
    # in the quadrant that describes what it actually did.
    geometry = {'sham': (0.94, 6.2), 'direction_removal': (0.06, 6.1),
                'matched_ordinary_state': (0.11, 1.0), 'ordinary_norm_clamp': (0.93, 1.0),
                'state_zeroed': (0.03, 0.1), 'random_tokens_zeroed': (0.92, 6.1),
                'norm_matched_tokens_zeroed': (0.91, 6.0),
                'offregister_direction_removal': (0.93, 6.1),
                'normmatched_direction_removal': (0.90, 6.0)}
    rows, fates = [], []
    for condition in Q1_CONDITIONS:
        for prompt_id, seed in units:
            offset = rng.normal(0, 0.04)
            leading, share = fate_mix[condition.key]
            fate = leading if rng.random() < share else str(rng.choice(
                [f for f in ("same_position", "relocated", "reserve_takeover", "diffuse", "none")
                 if f != leading]))
            fates.append(dict(question="q1", condition=condition.key,
                              condition_label=condition.label, role=condition.role,
                              selection_rule="percentile", prompt_id=prompt_id, seed=seed,
                              fate=fate, recovery_kind="same_position" if fate == "same_position"
                              else ("relocated" if fate == "relocated" else "none"),
                              first_recovery_layer=float(rng.choice(layers)) if fate != "none" else None,
                              head_retention=retention[condition.key] + offset))
            for layer in layers:
                decay = (layer - first_layer) / max(last_layer - first_layer, 1)
                rows.append(dict(question="q1", condition=condition.key,
                                 condition_label=condition.label, role=condition.role,
                                 edit=condition.edit, token_group=condition.group,
                                 selection_rule="percentile", prompt_id=prompt_id, seed=seed,
                                 layer=layer, head=-1, fate=fate,
                                 head_retention=float(np.clip(retention[condition.key] + offset
                                                              + 0.1 * decay + noise(), 0, 1)),
                                 sink_retention_all=float(np.clip(retention[condition.key] + offset, 0, 1)),
                                 affected_heads=12,
                                 attention_concentration=float(np.clip(
                                     0.34 * (0.4 + 0.6 * retention[condition.key]) + noise(0.01), 0, 1)),
                                 vstar_projection_change=projection[condition.key] * (1 - 0.35 * decay)
                                 + noise(0.12),
                                 max_cosine=float(np.clip(0.92 * retention[condition.key] + 0.05
                                                          + noise(0.02), 0, 1)),
                                 cosine=geometry[condition.key][0] + noise(0.015),
                                 target_norm_ratio=geometry[condition.key][1] + noise(0.05)))
    q1 = QuestionResult("q1", pd.DataFrame(rows), {"fates": pd.DataFrame(fates)})
    q1.verdict = _q1_verdict(q1.tables["fates"], q1.tidy)

    # ---- Q2 -----------------------------------------------------------------
    dose_rows = []
    for target in Q2_TARGET_LABELS:
        gammas = ([0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5] if target == "dominant_channel" else [0.0])
        strength = {"dominant_channel": -2.5, "competing_channel": -0.35,
                    "unspecific_channel": -0.2, "random_channel": -0.05,
                    "matched_energy_direction": -0.4, "ordinary_positions": -0.08}[target]
        for gamma in gammas:
            for scope in ("writer_only", "maintenance"):
                factor = 1.0 if scope == "writer_only" else 1.4
                for prompt_id, seed in units:
                    change = strength * factor * (1.0 - gamma) + rng.normal(0, 0.12)
                    dose_rows.append(dict(
                        question="q2", condition=f"{target}__gamma{gamma:g}__{scope}",
                        gamma=gamma, scope=scope, control=target,
                        control_label=Q2_TARGET_LABELS[target], prompt_id=prompt_id, seed=seed,
                        vstar_projection_change=change, immediate_vstar_change=change * 1.2,
                        original_sink_retained=float(np.clip(0.22 + 0.72 * min(gamma, 1.0)
                                                             + noise(0.03), 0, 1)),
                        key_rank=float(np.clip(1 + 40 * (1 - min(gamma, 1.0)), 1, 60)),
                        query_key_advantage=0.42 * min(gamma, 1.0) + noise(0.02),
                        channel_takeover=0.3 + 0.5 * (1 - min(gamma, 1.0))))
    q2 = QuestionResult("q2", pd.DataFrame(dose_rows), {"dose_response": pd.DataFrame(dose_rows)})
    q2.verdict = _q2_verdict(q2.tables["dose_response"])

    # ---- Q3 -----------------------------------------------------------------
    q3_rows, recovery_rows = [], []
    schedules = {"single_shot": ("same_position", 0.58), "repeated": ("none", 0.7)}
    for scope, (leading, share) in schedules.items():
        for prompt_id, seed in units:
            kind = leading if rng.random() < share else str(rng.choice(
                [k for k in ("same_position", "relocated", "none") if k != leading]))
            first = float(rng.choice(layers[1:6])) if kind != "none" else None
            recovery_rows.append(dict(question="q3", condition=f"direction_destroyed__{scope}",
                                      role="intervention", scope=scope, prompt_id=prompt_id,
                                      seed=seed, recovery_kind=kind, first_recovery_layer=first,
                                      recovered_at_original_position=kind == "same_position"))
            for layer in layers:
                progress = (layer - first_layer) / max(last_layer - first_layer, 1)
                recovered = kind != "none" and first is not None and layer >= first
                q3_rows.append(dict(question="q3", condition=f"direction_destroyed__{scope}",
                                    condition_label=f"Direction destroyed, {scope.replace('_', ' ')}",
                                    role="intervention", scope=scope, prompt_id=prompt_id,
                                    seed=seed, layer=layer, head=-1, recovery_kind=kind,
                                    first_recovery_layer=first,
                                    max_cosine=float(np.clip((0.72 if recovered else 0.2)
                                                             + 0.1 * progress + noise(0.03), 0, 1)),
                                    head_retention=float(np.clip(0.2 + 0.5 * recovered + noise(), 0, 1))))
    q3 = QuestionResult("q3", pd.DataFrame(q3_rows), {"recovery": pd.DataFrame(recovery_rows)})
    q3.verdict = _q3_verdict(q3.tables["recovery"])

    # ---- Q4 -----------------------------------------------------------------
    features = {"pre_mlp_residual": ("Pre-feed-forward residual", 0.46, -0.41, 0.86),
                "feedforward_activation": ("Feed-forward hidden activation", 0.31, -0.28, 0.79),
                "modulated_stream": ("Timestep-modulated normalised stream", 0.05, -0.03, 0.52),
                "preexisting_direction": ("Pre-existing register-direction component", 0.18,
                                          -0.15, 0.68)}
    patch_rows, separation_rows = [], []
    endpoints = {"capture_rate": "Becomes the attention sink",
                 "dominant_channel_value": "Dominant-channel magnitude",
                 "cosine": "Alignment with register direction"}
    for key, (label, transfer, prevent, auc) in features.items():
        for prompt_id, seed in units:
            separation_rows.append(dict(question="q4", feature=key, feature_label=label,
                                        supported=True, prompt_id=prompt_id, seed=seed,
                                        separation=float(np.clip(auc + noise(0.03), 0, 1)),
                                        token_dependence=0.8 if key != "modulated_stream" else 0.02,
                                        reason=""))
            for endpoint, endpoint_label in endpoints.items():
                scale = 1.0 if endpoint == "capture_rate" else 2.2
                for direction, effect in (("register_to_ordinary", transfer),
                                          ("ordinary_to_register", prevent)):
                    patch_rows.append(dict(
                        question="q4", patch=key, patch_label=label, direction=direction,
                        direction_label=("Transfer into an ordinary token"
                                         if direction == "register_to_ordinary"
                                         else "Prevent at an eventual register"),
                        endpoint=endpoint, endpoint_label=endpoint_label, prompt_id=prompt_id,
                        seed=seed, effect=effect * scale + noise(0.04)))
    q4 = QuestionResult("q4", pd.DataFrame(patch_rows),
                        {"patch_effects": pd.DataFrame(patch_rows),
                         "separation": pd.DataFrame(separation_rows)})
    q4.verdict = _q4_verdict(q4.tables["patch_effects"], q4.tables["separation"], {})

    # ---- Q5 -----------------------------------------------------------------
    rates = {"direction_at_ordinary_norm": 0.09, "direction_at_register_norm": 0.17,
             "full_residual_state": 0.38, "normalised_residual_state": 0.58,
             "key_before_position": float("nan"), "final_key": 0.84}
    ladder_rows = []
    for stage in Q5_STAGES:
        supported = not math.isnan(rates[stage.key])
        for position, factor in (("natural_register", 1.0), ("adjacent_patch", 0.72),
                                 ("random_ordinary", 0.55)):
            # Once the final key itself is transplanted, position stops mattering --
            # which is the whole point of the last rung, so the mock reflects it.
            factor = 0.97 if stage.key == "final_key" else factor
            for prompt_id, seed in units:
                base = rates[stage.key]
                value = float("nan") if math.isnan(base) else float(
                    np.clip(base * factor + noise(0.03), 0, 1))
                # The repaired Q5 schema: an exact-recipient clean baseline
                # (`matched_clean_rate`) is what a single-token transplant can be
                # compared against, while the any-register rate is context only --
                # they have different numerators and mixing them was the audit's
                # critical Q5 finding. `temporal_endpoint` separates the operation
                # itself from later-layer persistence.
                for transfer_mode in (("self_patch",) if position == "natural_register"
                                      else ("copy", "move")):
                    ladder_rows.append(dict(
                        question="q5", stage=stage.key, stage_label=stage.label,
                        position=position, position_label=Q5_POSITION_LABELS[position],
                        transfer_mode=transfer_mode, temporal_endpoint="same_operation",
                        prompt_id=prompt_id, seed=seed, supported=supported,
                        reason="" if supported else
                        "not a separable module boundary in this architecture",
                        capture_rate=value,
                        matched_clean_rate=float(np.clip(0.18 + noise(0.02), 0, 1)),
                        any_register_clean_rate=float(np.clip(0.88 + noise(0.02), 0, 1))))
    q5 = QuestionResult("q5", pd.DataFrame(ladder_rows), {})
    q5.verdict = _q5_verdict(q5.tidy, {s.key: (not math.isnan(rates[s.key]),
                                               "" if not math.isnan(rates[s.key]) else
                                               "not exposed as a separable module")
                                       for s in Q5_STAGES})

    # ---- Q6 -----------------------------------------------------------------
    shifts = {"sham": 0.0, "competing_channel_suppressed": 2.6,
              "competing_channel_amplified_early": -1.9, "direction_refreshed": 0.7,
              "random_channel_suppressed": 0.1, "matched_energy_removed": 0.2}
    q6_rows, lifetime_rows, trajectory_rows = [], [], []
    for condition in (("clean", "Clean run", "clean"),) + tuple(
            (c.key, c.label, c.role) for c in Q6_CONDITIONS):
        key, label, role = condition
        for prompt_id, seed in units:
            shift = shifts.get(key, 0.0) + noise(0.4)
            if key != "clean":
                lifetime_rows.append(dict(question="q6", condition=key, condition_label=label,
                                          role=role, prompt_id=prompt_id, seed=seed,
                                          clean_lifetime=last_layer - 6,
                                          register_lifetime=last_layer - 6 + shift,
                                          lifetime_shift=shift, half_life_shift=shift * 0.8))
            for layer in layers:
                progress = (layer - first_layer) / max(last_layer - first_layer, 1)
                survival = math.exp(-3.0 * max(progress - 0.15 - 0.05 * shifts.get(key, 0.0), 0))
                trajectory_rows.append(dict(
                    question="q6", condition=key, condition_label=label, prompt_id=prompt_id,
                    seed=seed, layer=layer, alpha=42.0 * survival + noise(0.5),
                    perpendicular_norm=11.0 + 6.0 * progress + noise(0.4),
                    cosine=float(np.clip(0.93 * survival + noise(0.02), 0, 1)),
                    dominant_channel=38.0 * survival + noise(0.5),
                    competing_channel=6.0 + 26.0 * progress * (0.3 if key ==
                                                               "competing_channel_suppressed" else 1.0),
                    sink_strength=float(np.clip(0.55 * survival + noise(0.02), 0, 1)),
                    angular_velocity=0.04 * progress + noise(0.005)))
                if key != "clean":
                    q6_rows.append(dict(question="q6", condition=key, condition_label=label,
                                        role=role, prompt_id=prompt_id, seed=seed, layer=layer,
                                        head=-1, lifetime_shift=shift,
                                        alpha=42.0 * survival, perpendicular_norm=11.0 + 6.0 * progress,
                                        sink_retention=float(np.clip(0.55 * survival + noise(0.02), 0, 1)),
                                        head_retention=float(np.clip(0.55 * survival + noise(0.02), 0, 1))))
    q6 = QuestionResult("q6", pd.DataFrame(q6_rows),
                        {"lifetime": pd.DataFrame(lifetime_rows),
                         "trajectory": pd.DataFrame(trajectory_rows)})
    q6.verdict = _q6_verdict(q6.tables["lifetime"])

    results = {"q1": q1, "q2": q2, "q3": q3, "q4": q4, "q5": q5, "q6": q6}
    for result in results.values():
        result.meta = {"mock": True, "banner": CAUSAL_MOCK_BANNER}
    return results
