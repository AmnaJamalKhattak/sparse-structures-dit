# Generation interventions

Paired, same-seed image generations in FLUX.1 and PixArt-Sigma that intervene on natural
high-norm register tokens, a single dominant residual channel, and per-head attention sinks.
Implemented in `src/experiments/generation_function.py`.

## What is intervened on

For each (prompt, seed) pair, a clean run first generates the image once while recording,
at every denoising step and block, which image tokens are natural registers and which key
each attention head treats as its natural sink. Interventions then reuse this clean trace
instead of re-detecting targets in the perturbed run.

A register mask marks tokens whose L2 norm exceeds `register_threshold` (default 3.0) times
the median token norm at that step and block, keeping at most `max_registers` (default 8)
tokens, largest norm first, ties broken toward the lower index. A per-head sink is the image
key that receives the most incoming attention under an image-key-renormalized softmax
(computed exactly, not sampled).

Six conditions, applied only inside the requested phase (a step range) and zone (a block
range):

- `baseline`: no edit.
- `remove_vstar`: at register tokens only, subtract the projection onto a unit direction
  `v*` fit ahead of time from clean register vectors.
- `suppress_channel`: zero one fixed channel (154 for FLUX.1-dev by default, 293 has been
  used for PixArt-Sigma) across every image token, not just registers.
- `suppress_sink`: mask each head's clean-traced sink key out of the image-query attention
  distribution, recomputed at full precision and re-dispatched through the model's own
  attention backend and dtype. Text-query rows and the residual stream are left untouched.
  For PixArt this only touches image self-attention (`attn1`). The T5 cross-attention
  (`attn2`) is not touched.
- `remove_top_registers`: zero the whole residual vector at register-token positions.
- `norm_only`: rescale register-token vectors to the median norm of the non-register tokens
  in the same step and block, preserving their direction. This is the magnitude-matched
  control for the other register edits.

`v*` is fit once per config, from the top right singular vector of unit-normalized register
vectors pooled across the calibration prompts, seeds, steps, and the layers covered by the
configured zones (sign chosen so the mean projection is positive).

Phase and zone ranges are read from the config JSON. If omitted, phases default to early
(steps 0-8), middle (9-18), and late (19-27) for a 28-step schedule, and zones default to
writer (block 18), early_register (19-22), mid_register (23-34), and dissolution (35-39).
FLUX.1 has 57 blocks (0-56). PixArt-Sigma has 28 image-only DiT blocks (0-27). The tests use
an example PixArt configuration with the writer at block 13 and a register zone at 14-20.
Because PixArt uses classifier-free guidance, interventions edit only the conditional (last)
row of the `[uncond, cond]` batch. The unconditional row and cross-attention are untouched.

## What is measured

`--evaluate` reads the run manifest and computes, per generated pair: LPIPS (if the `lpips`
package is installed), CLIP image-text similarity for the clean and edited image via OpenCLIP
ViT-H-14 (`laion2b_s32b_b79k`) if `open_clip` is installed, ImageReward score (`ImageReward-v1.0`)
if the `ImageReward` package is installed, and low/high spatial-frequency RMS distance between
clean and edited RGB (Gaussian blur at `sigma=8` pixels splits the bands). A missing optional
package leaves its columns blank rather than substituting a different metric. Externally
computed GenEval-style scores can be merged in by exact run-cell identity via
`--structured-scores`.

`summary.csv` bootstraps (2000 resamples, 95% percentile interval) the mean of each metric
within every (condition, phase, zone) group.

## Controls

- `baseline` (no edit) and `norm_only` (magnitude-matched, direction-preserving) bound the
  register-edit conditions.
- `remove_vstar` and `remove_top_registers` isolate a single fitted direction versus the
  full register vector.
- After each generation, the intervention hooks self-check (`audit()`): the total forward
  count per block must equal the number of denoising steps, and the number of edits actually
  fired must equal exactly the steps-in-phase times blocks-in-zone. A mismatch raises before
  the run is recorded, which is how a target that silently failed to fire gets caught.
- `validate_model_layout` (aliased as `validate_flux1_layout`) checks before generation that
  the discovered blocks match the configured model family, every zone layer exists, and the
  channel index fits the residual width.

## How to run

Everything reads a JSON config deserialized into `Q7Config` (`--config path.json`,
required). With no action flag the paired grid runs directly, which needs a compatible `v*`
already on disk:

```
python -m src.experiments.generation_function --config config.json --calibrate-vstar
python -m src.experiments.generation_function --config config.json
```

`--calibrate-and-run` does both with a single model load and reuses prompt conditioning
across calibration and generation. `--smoke` truncates the run to the first (prompt, seed)
pair and the first phase/zone cell, for a quick end-to-end check. `--evaluate` computes
`paired_metrics.csv` and `summary.csv` (`--structured-scores file.csv` merges external
scores). `--plot` redraws the figures from an existing `paired_metrics.csv` without
recomputing metrics. `--calibrate-vstar`, `--evaluate`, `--plot`, and `--calibrate-and-run`
are mutually exclusive.

## Outputs

All paths are under `output_dir` from the config.

- `images/`: one clean PNG per (prompt, seed) and one edited PNG per (prompt, seed,
  condition, phase, zone).
- `runs.jsonl`: one appended record per generated pair (prompt, seed, condition, phase,
  zone, image paths, config hash, run identity, generation parameters, intervention audit).
  A run is resumed at the level of a whole (prompt, seed) scenario or an individual cell,
  keyed on run identity so a changed scheduler config or `v*` file does not silently reuse
  stale images.
- `vstar.npy` and `vstar.json`: the fitted direction and its calibration metadata (model,
  vector and scenario counts, calibration layers, config hash, file hash).
- `paired_metrics.csv`, `summary.csv`: per-pair metrics and the bootstrapped summary above.
- `figures/q7_causal_map.png`, `q7_frequency_profile.png`: effects by phase and zone.
- `figures/q7_prompt_fidelity.png`: paired prompt-fidelity deltas with bootstrap CIs, when
  any such metric is present.
- `figures/q7_vstar_loadings.png`: the 12 largest `v*` loadings by magnitude, with the
  configured channel marked.
- `figures/q7_representative_contact_sheet*.png`: up to three sheets, one per prompt, each
  showing clean/intervened/amplified-difference images for one representative seed/phase/zone.
