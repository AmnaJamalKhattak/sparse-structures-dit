# Localization baseline

Reproduces the channel-mask localization protocol of Turri et al. (arXiv:2605.13974)
for FLUX.2-klein: a small number of massive-activation channels in a diffusion
transformer are claimed to localize the image subject.

## What is computed

For each layer, the top-k channels are selected by channel score
`mean(abs(activation))` over tokens (abs then mean). Their per-token activations are
min-max normalized per channel, then clustered with K-means (K=2) on the resulting
k-dim per-token vectors. The cluster with the higher mean of
`s[n] = normalized[n,:].sum()` (the heatmap) is the foreground, giving a binary mask.
The same procedure runs for bottom-k channels (lowest score) and for random-k channels
(several trials) as controls.

Only image-stream tokens at the last denoising timestep are used. The text/image split
is derived at runtime from the packed sequence length, not hard-coded.

For evaluation, BiRefNet segments the decoded RGB image into a pseudo-ground-truth
foreground mask. Each layer's stored latent-resolution mask is upsampled to the image
resolution and scored with IoU against this mask. Averaging over all cached prompts
gives mIoU per (layer, strategy); random-k IoU is averaged over its trials per prompt
first.

## Decision rule

None. This produces a descriptive mIoU curve per layer and strategy, not a hypothesis
test. An optional `--reference-check` compares the curve against approximate published
values (top-k dominates every layer, bottom-k flat near 0.2, random-k in between, top-k
peak near 0.5 around layer 10) and prints any differences as warnings. A different
curve is not treated as an error.

## Layout

```
configs/default.yaml            # config (see precedence below)
src/common/config.py            # config loading and validation
src/common/model_utils.py       # pipeline and BiRefNet loading, capture hooks (lazy torch)
src/common/clustering.py        # normalize -> KMeans(2) -> mask
src/common/io.py                # reduced-cache shards, prompt loading, IoU, upsampling
src/stage1_generate_and_cache.py
src/stage2_channel_ranking.py
src/stage3_mask_construction.py
src/stage4_evaluate_figure3d.py
scripts/run_pipeline.sh         # resumable end-to-end wrapper
outputs/  cache/                # runtime only
```

## Install

```bash
uv sync                    # core + test deps (numpy, scikit-learn)
uv sync --extra fig3       # + torch/diffusers/transformers/matplotlib for a real run
```

The numeric core (`config`, `clustering`, `io`, ranking) is pure numpy/scikit-learn and
importable without a GPU. torch, diffusers, transformers, and matplotlib are imported
lazily inside the model-touching stages, so the tests run anywhere.

## Configure

Fill the five required keys in `configs/default.yaml` (or override them; the loader
fails loudly if any is empty, since this runs unattended):

| key | example |
|---|---|
| `model_ckpt` | `black-forest-labs/FLUX.2-klein-4B` (ungated) or `-9B` (gated), HF id or local dir |
| `prompt_source` | 1,600 GenAI-Bench prompts, bundled at `data/genai_prompts.jsonl`, or another `.txt`/`.json`/`.jsonl`/`.parquet` file, or an HF dataset id |
| `birefnet_weights` | `ZhengPeng7/BiRefNet` |
| `output_dir` | where CSV, plots, qualitative dumps, and `run_metadata.json` land |
| `activation_cache_dir` | where reduced cache shards land |

Override precedence, highest wins: CLI flag, then `FIG3_*` env var, then YAML.

```bash
FIG3_TOP_K=12 FIG3_SEED=1 python -m src.stage1_generate_and_cache --config configs/default.yaml
python -m src.stage1_generate_and_cache --config configs/default.yaml --set seed=1 --set device=cuda
```

## Run

```bash
python -m src.stage1_generate_and_cache --config configs/default.yaml --fused
python -m src.stage4_evaluate_figure3d  --config configs/default.yaml

# or the resumable wrapper (skips already-cached prompts)
scripts/run_pipeline.sh configs/default.yaml
```

`FLUX.2-klein-4B` is about 16 GB in half precision (Qwen3-4B text encoder plus 4B
transformer), so it needs a GPU of at least 24 GB to load fully. On a 16 GB GPU set
`offload: true` to enable `enable_model_cpu_offload` (fits, slower).

Fused mode (default) runs stages 2 and 3 in-process right after each capture and
persists only small reduced artifacts (scores, channel indices, binary masks, decoded
RGB, qualitative PNGs). The full `[N_I, D]` per-layer tensor is discarded, since at
FLUX hidden width across nearly all layers and 1,600 prompts it would be hundreds of
GB.

To debug the stages standalone on a handful of cached prompts:

```bash
python -m src.stage1_generate_and_cache --config cfg.yaml --no-fused --limit 4  # persist full acts
python -m src.stage2_channel_ranking    --config cfg.yaml --prompt-id 0
python -m src.stage3_mask_construction  --config cfg.yaml --prompt-id 0 --qualitative
python -m src.stage4_evaluate_figure3d  --config cfg.yaml
```

The study launcher runs the same evaluation under a neutral name:

```bash
python -m src.experiments localization --config configs/default.yaml
```

This writes `localization_results.csv` and `localization_curve.png` instead of the
`figure3d_*` names `stage4_evaluate_figure3d` writes directly. Both accept
`--artifact-prefix localization|figure3d` and `--reference-check`.

## Outputs

- `outputs/run_metadata.json`: resolved config, package versions, prompt-source hash,
  and logged assumptions (random-k trial count, K-means init/seed defaults, capture
  conventions)
- `outputs/figure3d_results.csv`: `layer, strategy, mean_miou, std_miou, n`
- `outputs/figure3d_curve.png`: the three mIoU curves
- `outputs/qualitative/prompt_XXXXX/`: heatmap and mask PNGs for the first
  `num_example_prompts` prompts
