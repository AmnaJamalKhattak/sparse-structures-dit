"""A synthetic pooled Q16 dataset shaped like the real one, for exercising the figures.

The numbers are invented and only plausible in shape -- a FLUX-like 57-block stack and a
PixArt-like 28-block one -- so every figure path runs and can be looked at offline.
"""
import numpy as np
import pandas as pd

from ditsinks import q16_main as Q

MODELS = {
    "flux1-dev": dict(n_layers=57, formation=18, natural_end=39, decline=31, seam=19,
                      size=1024, steps=28),
    "pixart-sigma-1024": dict(n_layers=28, formation=10, natural_end=20, decline=15, seam=None,
                              size=1024, steps=20),
}
EFFECT = {"induce_E1": 0.05, "induce_E2": 0.07, "induce_E3": 0.12, "induce_N": 0.03,
          "induce_L1": 0.10, "induce_L2": 0.06, "induce_L3": 0.035, "remove": 0.15,
          "remove_induce_E3": 0.18, "remove_induce_L1": 0.14, "extend": 0.08,
          "control_random_direction": 0.04, "control_ordinary_positions": 0.06,
          "control_in_distribution": 0.02, "hooks_only": 0.0}


def protocol_for(checkpoint):
    spec = MODELS[checkpoint]
    settings = Q.MainSettings(checkpoint=checkpoint, size=spec["size"], steps=spec["steps"])
    rows = [dict(step=s, prompt_id=p, seed=0, formation=spec["formation"], peak_layer=spec["formation"] + 2,
                 peak_count=30, natural_end=spec["natural_end"], decline=spec["decline"],
                 n_carriers=30, valid=True, note="")
            for p in range(3) for s in settings.edit_step_list()]
    return Q.protocol_from_calibration(rows, n_layers=spec["n_layers"], seam=spec["seam"],
                                       checkpoint=checkpoint, settings=settings)


def pooled(n_prompts=8, seeds=(0, 42, 1234, 777, 3407), seed=0):
    rng = np.random.default_rng(seed)
    protocols = {m: protocol_for(m) for m in MODELS}
    images, lifecycle, attention, sensitivity, boundaries = [], [], [], [], []
    for checkpoint, p in protocols.items():
        catalog = Q.condition_catalog(p)
        scale = 1.0 if checkpoint == "flux1-dev" else 0.8
        for prompt in range(n_prompts):
            prompt_effect = rng.normal(0, 0.015)
            for s in seeds:
                keys = dict(checkpoint=checkpoint, prompt_id=prompt, seed=s)
                for c in catalog:
                    if c.key == "reference":
                        continue
                    lp = max(0.0, scale * EFFECT.get(c.key, 0.05) + prompt_effect
                             + rng.normal(0, 0.01)) if c.key != "hooks_only" else 0.0
                    images.append(dict(keys, condition=c.key, label=c.label(p), group=c.group,
                                       window=c.window, lpips=lp,
                                       clip_image_similarity=1 - 0.6 * lp,
                                       clip_prompt_similarity_change=-0.08 * lp
                                       + rng.normal(0, 0.002)))
                step = p.readout_step
                for c in catalog:
                    if c.key == "hooks_only":
                        continue
                    for layer in range(p.n_layers):
                        natural = p.formation <= layer <= p.natural_end
                        single = p.seam is None or layer >= p.seam
                        mass = 0.004 + (0.14 if natural and single else 0.01 if natural else 0.0)
                        projection = (25000 if natural else 800) * (1 + 0.01 * layer)
                        pop = 30 if natural else 0
                        if c.remove and natural:
                            mass, projection, pop = 0.004, 600, 0
                        if c.window:
                            a, b = p.windows[c.window]
                            if a <= layer <= b:
                                mass = 0.13 if single else 0.009
                                projection = 12000 * (1 + 0.02 * layer)
                        if c.key.startswith("induce_E") and layer == p.formation:
                            projection = 11000
                        lifecycle.append(dict(keys, condition=c.key, step=step, layer=layer,
                                              carrier_incoming_mass=mass * rng.uniform(0.8, 1.2),
                                              carrier_projection=projection,
                                              n_highnorm_frozen=pop,
                                              n_highnorm_new=(3 if c.key.startswith("induce_E")
                                                              and natural else 0)))
                        if c.key in ("reference", "remove", f"remove_induce_{p.primary_early}",
                                     f"remove_induce_{p.primary_late}"):
                            kept = 1.0 if c.key == "reference" else 0.03
                            attention.append(dict(
                                keys, condition=c.key, step=step, layer=layer,
                                n_affected_heads=20 if natural else 0, affected_kept=kept,
                                affected_other_original_carrier=0.0 if kept == 1 else 0.1,
                                affected_relocated_vstar_carrier=0.0,
                                affected_clean_register_elsewhere=0.0,
                                affected_non_register_token=0.0 if kept == 1 else 0.2,
                                affected_spread_out=0.0 if kept == 1 else 0.67))
                for value in (1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0):
                    sensitivity.append(dict(keys, parameter="highnorm_ratio", value=value,
                                            chosen=value == 3.0, statistic="carriers_at_peak",
                                            result=30 + (40 if value < 2.5 else 0)
                                            - (8 if value > 6 else 0) + rng.normal(0, 1)))
                    sensitivity.append(dict(keys, parameter="highnorm_ratio", value=value,
                                            chosen=value == 3.0, statistic="natural_end_block",
                                            result=p.natural_end + (6 if value < 2.5 else 0)))
                for value in (0.99, 0.995, 0.999, 0.9995, 0.9999):
                    sensitivity.append(dict(keys, parameter="alignment_quantile", value=value,
                                            chosen=value == 0.999,
                                            statistic="max_share_of_image_stripped",
                                            result=(1 - value) + 0.008))
                    sensitivity.append(dict(keys, parameter="alignment_quantile", value=value,
                                            chosen=value == 0.999,
                                            statistic="min_share_of_carriers_caught",
                                            result=1.0 if value < 0.9999 else 0.9))
                for value in (2.0, 5.0, 10.0, 20.0, 40.0):
                    sensitivity.append(dict(keys, parameter="sink_threshold", value=value,
                                            chosen=value == 10.0,
                                            statistic="mean_sink_tokens_per_block",
                                            result=40 / value + 25))
                    sensitivity.append(dict(keys, parameter="sink_threshold", value=value,
                                            chosen=value == 10.0,
                                            statistic="share_of_sinks_that_are_carriers",
                                            result=min(1.0, 0.6 + value / 40)))
                for s2 in p.edit_steps:
                    boundaries.append(dict(keys, step=s2,
                                           formation=p.formation + int(rng.integers(0, 2)),
                                           natural_end=p.natural_end - int(rng.integers(0, 3)),
                                           valid=True))
    frames = dict(images=pd.DataFrame(images), lifecycle=pd.DataFrame(lifecycle),
                  attention=pd.DataFrame(attention), sensitivity=pd.DataFrame(sensitivity),
                  boundaries=pd.DataFrame(boundaries))
    frames["units"] = frames["images"][["checkpoint", "prompt_id", "seed"]].drop_duplicates()
    return frames, protocols
