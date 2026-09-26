# Diffusion Activation Studies

This folder measures how activation magnitude concentrates in a few channels and a
few tokens in diffusion transformers, and how text-stream and image-stream interventions
affect generation. It runs these studies on FLUX.1 and PixArt-Sigma, including a
reproduction of the channel-mask localization baseline of Turri et al. (arXiv:2605.13974).

## Layout

- `src/common/`: model-agnostic helpers shared across studies (capture hooks, config
  loading, channel ranking, clustering, spatial maps, IO). Pure numpy/scikit-learn where
  possible, so most of it imports and unit-tests without a GPU.
- `src/stage1_generate_and_cache.py` through `src/stage4_evaluate_figure3d.py`: the
  localization baseline pipeline (generate and cache activations, build masks, evaluate
  mIoU against pseudo-labels).
- `src/experiments/`: the study drivers, plus the `__main__.py` launcher.
- `configs/`: YAML configs for the studies that take one.
- `data/`: prompt sets used by the studies.
- `docs/`: method notes for individual studies.
- `scripts/`: the localization pipeline shell wrapper.
- `tests/`: the pytest suite.
- `Figure3_Colab.ipynb`: a Colab notebook that runs every study with a GPU runtime. Copy
  this folder (`channel_studies/`) to `/content/massive-activations-fig3` before running its
  setup cell.

## Install

Requires Python >= 3.11. From this folder (`channel_studies/`):

```bash
pip install -e ".[fig3]"
pip install -e ".[q9]"
```

`fig3` installs the dependencies for the localization baseline, channel stability, token
norm decomposition, cross-model, text stream, and generation intervention studies. `q9`
installs the dependencies for the text to image coupling study. Model runs need a GPU.
The test suite runs on CPU with `pytest`, `numpy`, `scikit-learn` and `pyyaml`. Tests that
need torch are skipped when it isn't installed.

## Studies

Run commands from this folder. Most studies go through the shared launcher,
`python -m src.experiments <study> --config <path>`; pass `<study> --help` for a driver's
full option list. The generation intervention and text to image coupling studies are run
as their own modules instead, since they build their JSON config from the notebook's
model presets rather than a static YAML file.

- Localization baseline: channel masks and layer-wise mIoU against BiRefNet pseudo-labels.
- Channel stability: whether the largest-magnitude channels for a generation change across
  prompts, seeds, and denoising steps.
- Token norm decomposition: whether high-norm image tokens stay high-norm once the massive
  channels are excluded from the norm, with effect sizes and null comparisons.
- Token norm decomposition, qualitative panels: the same question with one figure per
  prompt and no statistics.
- Cross-model comparison: the token norm decomposition repeated across models, one model
  per figure row.
- Text stream: the same channel and norm analysis applied to text tokens rather than image
  tokens.
- Generation intervention study: paired interventions on the register, channel, and sink
  circuit during generation, scored on the resulting images.
- Text to image coupling study: causal interventions on the text stream, testing whether
  they change the image-side register circuit.

| Study | Module | Command | Config | Method doc |
|---|---|---|---|---|
| Localization baseline | `src.stage4_evaluate_figure3d` | `python -m src.experiments localization --config configs/default.yaml` | `configs/default.yaml` | `docs/localization_baseline.md` |
| Channel stability | `src.experiments.channel_stability` | `python -m src.experiments stability --config configs/channel_stability.yaml` | `configs/channel_stability.yaml` | |
| Token norm decomposition | `src.experiments.highnorm_tokens` | `python -m src.experiments norms --config configs/highnorm_tokens.yaml` | `configs/highnorm_tokens.yaml` | `docs/highnorm_tokens.md` |
| Token norm decomposition, qualitative panels | `src.experiments.highnorm_qualitative` | `python -m src.experiments norm-panels --config configs/highnorm_tokens.yaml` | `configs/highnorm_tokens.yaml` | `docs/highnorm_tokens.md` |
| Cross-model comparison | `src.experiments.highnorm_crossmodel` | `python -m src.experiments cross-model --config configs/highnorm_crossmodel.yaml` | `configs/highnorm_crossmodel.yaml` | |
| Text stream | `src.experiments.text_stream_qualitative` | `python -m src.experiments text --config configs/highnorm_tokens.yaml --layers all` | `configs/highnorm_tokens.yaml` | |
| Generation intervention study | `src.experiments.generation_function` | `python -m src.experiments.generation_function --config <config.json> --calibrate-and-run` | generated at runtime, see `Figure3_Colab.ipynb` | `docs/generation_interventions.md` |
| Text to image coupling study | `src.experiments.text_image_coupling` | `python -m src.experiments.text_image_coupling --config <config.json>` | generated at runtime, see `Figure3_Colab.ipynb` | `docs/text_image_coupling.md` |

The localization baseline also has a two-stage pipeline wrapper:

```bash
scripts/run_pipeline.sh configs/default.yaml
```

This runs `src.stage1_generate_and_cache` (generate and cache activations, resumable) and
then `src.stage4_evaluate_figure3d` (evaluate mIoU and write the results curve).

Channel exclusion in the token norm decomposition and cross-model studies is post-hoc
attribution on captured activations, not an intervention in the model's forward pass.
`docs/highnorm_tokens.md` states the selectivity and elevation effect sizes and the H1/H2/H3
decision rule those studies use.

The localization baseline loads its config through `src/common/config.py`, which resolves
settings with a fixed precedence: a `--set key=value` CLI override wins over a `FIG3_*`
environment variable, which wins over the value in `configs/default.yaml`. Required paths
and model ids are left blank in that file on purpose; the loader fails immediately if one
is still empty when a run starts, rather than falling back to a silent default. The other
studies load their config with a separate, per-study loader.

## Tests

```bash
pytest
```

Run from this folder; `pyproject.toml` sets the test path and Python path so no
extra configuration is needed. The suite covers the CPU-testable core: config loading,
channel ranking, clustering and mask construction, IoU and upsampling, and the CLI and
config-building paths of every study, without needing a GPU or the model weights.

## Data

`data/genai_prompts.jsonl` holds the 1,600 GenAI-Bench prompts used for the localization
baseline's full run. Each line is a JSON object with a single `prompt` field.
