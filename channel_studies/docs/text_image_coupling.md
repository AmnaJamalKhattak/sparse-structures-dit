# Text to image coupling

Whether specific text-stream token states in FLUX causally seed, maintain, or are read back
from the image-register circuit. Implemented across `src/experiments/text_image_coupling.py`
(config, CLI, and shared numeric utilities), `q9_runtime.py` (calibration and per-job model
execution), `q9_report.py` (aggregation, figures, export), and `q9_colab.py` (notebook config
builder). Only `flux1-dev` and `flux-schnell` are supported. PixArt has no live DiT text
stream to read back from.

## What is intervened on

Candidate text positions are chosen per prompt from real (non-batch-padding) token
positions by norm, by attention, or both: `norm_candidates` flags tokens whose L2 norm
exceeds `text_norm_threshold` times the median. `sink_mask` flags a text key if, averaged
over image-query heads, its share of incoming attention exceeds `sink_enrichment` divided by
the sequence length and its absolute mass exceeds `sink_min_mass`. `candidate_source`
selects norm, sink, their union, or their intersection, restricted to `candidate_classes`
(content, eos, pad, special, by default eos and pad). Candidate image positions
(registers) use the same norm criterion as the generation intervention study, at
`image_norm_threshold`, capped at `image_max_registers`.

Each `site` (-1 for the projected T5 output before block 0, or 0-55 for a DiT block) crossed
with each denoising `step` gets its own branched, single forward from the frozen clean
latent, conditioning, and timestep at that step. `readout_layer` (a layer after every site)
is where the causal-map summary is read.

Text-state edits (applied at candidate positions, or at `matched_positions`-selected control
positions for the `ordinary_*` variants): `remove_direction` projects out a fitted unit
direction `v`. `norm_matched` keeps the token's own direction but rescales it to the norm
that direction removal would leave, isolating the norm change from the direction change.
`random_direction` mixes the token's direction with a random orthogonal one so the edit
matches `remove_direction` in both final norm and edit size, without removing the fitted
direction. `suppress_channel` zeros one fitted channel. `zero`/`ordinary_zero` zero the whole
token, at candidate or matched-control positions. `donor_swap` replaces candidate states with
the same positions from a tokenizer-length-matched donor prompt, holding this prompt's image
latent and pooled CLIP conditioning fixed. `image_remove_direction`, `image_zero`, and
`image_ordinary_zero` apply the same three edits to the image-register positions instead, to
test read-back from image to text.

Six attention-edge methods measure or edit one query-to-key pathway instead of editing a
state: `image_reads_text_*` (image queries, candidate text keys), `text_reads_register_*`
(candidate text queries, image-register keys), `text_reads_content_*` (candidate text
queries, ordinary content-text keys). The `_score` variant blocks the target keys out of the
softmax so the remaining keys renormalize. The `_value` variant leaves the attention weights
alone and subtracts only the weighted value contribution of the target keys from the output,
separating routing competition from value content.

Four rescue kinds run only in `confirm` mode, only alongside `remove_direction`, applied once
at `rescue_layer` (which must sit after every intervention site): `projection` restores only
the clean projection onto the fitted image direction at the register positions.
`ordinary_projection` does the same at matched control positions. `state` restores the full
clean value at the register positions. `sham` is a no-op patch used as the rescue-path
control.

## What is measured

`attention_reductions` computes an exact (not sampled) all-key softmax per query population
(text, image) and head: mass onto text keys and image keys, per-token-class mass, the image
and text sink positions and masses, and entropy, without keeping a full attention matrix.
`state_metrics` reports, per token class and for both the candidate/register positions and
the clean run's frozen ("fixed") positions: token count, norm, and, where a fitted direction
or channel exists, `alignment_squared` (mean squared projection onto the fitted direction,
as a fraction of squared norm) and `channel_energy` (mean squared value in one channel, as a
fraction of squared norm). For text this channel is the calibration-fitted dominant channel
at that site. For image it is the configured `image_channel` (154 by default). A `positions`
row records the Jaccard overlap between the fixed and freshly redetected register or
candidate sets. `birth_summary` records, per denoising step, the first layer with any
norm-outlier register, the first layer where that register is channel-dominated
(`channel_energy` >= 0.5), and the full list of channel-dominated layers. Every edited-run
metric is paired against the same metric
at the same (stage, step, layer, stream, population) in the clean run to get a delta. Deltas
are aggregated by `cluster_summary`, a bootstrap over prompts (seeds and repeated
sites/steps within a prompt are not treated as independent samples).

In `confirm` mode, full trajectories are also scored with the same optional LPIPS, OpenCLIP,
and ImageReward metrics as the generation intervention study, and externally computed
GenEval-style or ImageReward scores can be merged in by exact job identity.

When `equivalence_bound` is set, `equivalence.csv` reports, per condition/site/step, whether
the bootstrap CI of the image-side `channel_energy` delta at `readout_layer` falls entirely
inside `[-bound, bound]`.

## Controls

- `ordinary_zero` / `image_ordinary_zero`: the same lesion applied to a same-class,
  nearest-norm matched control position instead of the candidate, chosen without
  replacement. A candidate with no eligible match records `no_matched_control` rather than
  being silently skipped.
- `norm_matched` and `random_direction` are magnitude- and edit-size-matched controls for
  `remove_direction` that do not remove the fitted direction itself.
- `sham` is a no-op control for the `projection`/`state` rescues.
- Calibration prompts and evaluation prompts must be disjoint (`Q9Config.validate` raises
  otherwise).
- A saved clean-run snapshot is replayed and its prediction is checked against the original
  clean forward (`rtol=1e-3`, `atol=1e-3`) before any probe or donor swap reuses it.
- `Reservoir` caps the number of activation rows kept per (prompt, site) during calibration
  at `reservoir_size`, the same budget for every prompt.

## How to run

```
python -m src.experiments.text_image_coupling --config config.json
python -m src.experiments.text_image_coupling --config config.json --plot
python -m src.experiments.text_image_coupling --config config.json --export-compact /path/to/export
```

`--config` (required) is a JSON file matching `Q9Config`. With no other flag this runs
calibration (skipped if a matching calibration is already cached) followed by the configured
jobs. `discovery` mode calibrates and reports without running any jobs. `--plot` rebuilds the
report from an existing run without recomputing anything. `--export-compact DIR` copies the
run's small csv/json files and figures into `DIR/<run-id>/`, refusing above a 250 MiB budget.
It does not copy raw activations or the image grid. `output_dir` must not resolve under a
path containing `/drive/`. Run locally and export the compact bundle to Drive separately.

`q9_colab.py` wraps this for a notebook: `build_config` turns form fields into a `Q9Config`
(a `pilot` workload shrinks prompts/seeds/sites/steps/methods for a quick pass. `full` keeps
the preset grid), `save_config` writes it to disk, `run_experiment` launches the same CLI in
a subprocess and streams its log, `show_results` displays the run's figures inline, and
`run_budget` reports the planned number of trajectories and forwards for a config before it
runs.

## Outputs

Under `output_dir/runs/<identity>/`:

- `config.json`, `calibration.json`, `provenance.json`, `tokens.json`, `t5.json`: run
  configuration, fitted directions with stability checks, environment and model metadata,
  per-prompt token classification, and T5-layer summaries.
- `p{prompt_id:03d}_s{seed}/baseline.json` (+ `baseline.png` in confirm mode) and one
  `{job_id}.json` (+ optional `.png`) per (method, site, step, rescue) job, where `job_id` is
  a hash of that tuple.
- `direction_stability.csv`, `t5_direction_stability.csv`, `directions.csv`, and
  `figures/q9_direction_energy.png`: calibration diagnostics.
- `manifest.json`, `audits.csv`, `birth.csv`: one row per job, per edit/rescue audit, and per
  (job or baseline, denoising step) register-birth summary.
- `paired_metrics.csv.gz`: every measured (layer, stream, population, metric) delta, for
  every job.
- `summary.csv`: the bootstrapped mean and CI of each delta, by condition, site, step,
  rescue, layer, stream, population, and metric.
- `figures/q9_causal_map.png`: the `readout_layer` image `channel_energy` change, one panel
  per method, over the step x site grid.
- `figures/q9_text_to_image.png`, `q9_image_to_text.png`: example downstream-layer response
  curves for one site/step, forward (text edit, image response) and reverse (image edit,
  text response).
- `equivalence.csv`: only written when `equivalence_bound` is set.
- `image_metrics.csv`, `image_summary.csv`: per-job perceptual and prompt-fidelity metrics
  and their cluster summary, for confirm-mode trajectories.
- `clean_p*_s*.csv`, `example_clean_atlas.csv`, `figures/q9_example_pairs.png`,
  `q9_text_norm_atlas.png`, `q9_routing.png`: baseline token-class/layer atlases from one
  example trajectory and example clean/edited image pairs.
- `report_status.json`: job and status counts, which calibrated directions had a leading
  energy fraction below 0.5, and notes that no-candidate/no-control jobs are excluded rather
  than treated as null results and that summary CIs are clustered by prompt.
