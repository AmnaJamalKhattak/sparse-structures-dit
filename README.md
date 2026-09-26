# Sparse structures in Diffusion Transformers

Code for measuring and intervening on high-norm tokens, massive-activation channels and
attention sinks in FLUX.1-dev, FLUX.1-schnell and PixArt-Sigma.

The repository has three independent parts. Each has its own dependencies and is run on
its own.

| Part | Path | Contents |
|---|---|---|
| `ditsinks` package and notebooks | repository root (`ditsinks/`, `notebooks/`, `scripts/`, `results/`, `docs/`, `tests/`) | the main experiments: sinks, shared direction, causal interventions, direction vs magnitude, register retiming. Described in the rest of this file. |
| Channel studies | `channel_studies/` | channel stability, token norm decomposition, cross-model and text stream studies, generation interventions, and a reproduction of a channel-mask localization baseline. See `channel_studies/README.md`. |
| Channel recurrence and register interventions | `register_interventions/` | six standalone Colab notebooks (FLUX.1-dev and PixArt-Sigma) for channel recurrence across scenarios, register dissolution and restoration. See `register_interventions/README.md`. |

A high-norm token is an image token whose residual-stream norm is much larger than the
layer's typical norm. A massive-activation channel is a single hidden coordinate whose
absolute value is unusually large across tokens. An attention sink is an image token that
receives far more incoming attention than a uniform share would predict. The notebooks
measure where these three line up, fit the shared direction the high-norm tokens carry
(`v*`), and then intervene on norm, direction and timing to test which of them is doing
the causal work.

Every notebook in `notebooks/` uses the `ditsinks` package for capture hooks, metrics,
interventions and figures. Each notebook is meant to run in Colab on a GPU. Copy this
repository to `/content/ditsinks-code` (the notebooks that set `CLONE_DIR`)
or open the notebook from the repository root; the setup cell then installs its own
dependencies. Results are written to Google Drive so a later session can pick up where an
earlier one left off.

## What the package does

`ditsinks` is organised around a few stages that most notebooks share:

- **capture**: forward hooks that record per-token residual norms, per-channel peak
  activations, and image-to-image attention, at chosen blocks and denoising steps.
- **discovery**: from a capture, finds the register tokens, fits the shared direction
  `v*` they carry, and picks the dominant channel and the layer ranges where the register
  state forms, is maintained, and dissolves. This is saved as a frozen artifact so later
  causal experiments compare against the same reference.
- **interventions**: edits applied at a hook, at every denoising step, such as rescaling a
  token to the median norm, replacing its direction, or rotating it in the plane of `v*`.
  Every intervention runs alongside matched controls (a random-token control, a
  next-highest-norm control) so an effect can be attributed to the register tokens
  specifically.
- **metrics, statistics, figures**: turn captured or intervened runs into tidy tables,
  prompt-clustered confidence intervals, and the plots that read them.

## Layout of the `ditsinks` part

| Path | Contents |
|---|---|
| `ditsinks/` | the package: capture hooks for token norms, channels and attention; fitting `v*`; interventions; statistics; figures |
| `notebooks/` | one notebook per experiment (see the table below) |
| `scripts/` | figures and analyses that run from saved results, without a GPU |
| `results/q13/` | the two CSV tables that `scripts/make_fig4_direction_at_fixed_norm.py` reads, with their provenance in `results/q13/PROVENANCE.md` |
| `docs/` | design notes for the retiming and direction-vs-magnitude experiments |
| `tests/` | CPU tests that need no model weights |
| `channel_studies/` | a separate part with its own `README.md`, `pyproject.toml` and tests |
| `register_interventions/` | a separate part with its own `README.md` |

## Install

```bash
pip install -r requirements.txt
pytest
```

`pytest` from the repository root runs only `tests/` (set in `pytest.ini`). The channel
studies have their own install and test commands, run from `channel_studies/`.

The test suite needs no GPU and no downloaded weights; it runs against synthetic data and
takes a few minutes on a CPU.

Running a notebook itself needs a GPU. A Colab A100 handles every notebook here; a few
cells work on a smaller GPU (an L4 or T4 is enough for FLUX.1-schnell or PixArt-Sigma at
512 px, and for the `tiny-flux1` synthetic checkpoint used in dry runs). Each notebook's
first cells install the package versions it needs, which can differ from
`requirements.txt`. The diffusers version matters most:

| Notebook | diffusers |
|---|---|
| `causal_mechanism`, `direction_vs_magnitude`, `register_retiming`, `paper_figures` | `>=0.36` |
| `highnorm_sinks_massive_channels` | `==0.40.0` |
| `highnorm_tokens_vs_attention_sinks_flux`, `sparse_structures_repro` | `==0.35.1` |

## Notebooks

| Notebook | What it runs | Model(s) | GPU | Writes |
|---|---|---|---|---|
| `highnorm_tokens_vs_attention_sinks_flux.ipynb` | Captures explicit attention and residual norms per block and head, and measures whether the highest-norm image tokens are also the attention sinks. Includes a seed-clamp and a direction-ablation intervention on the register tokens, and a channel-identity pass that fits `v*` from captured register vectors. | FLUX.1-schnell, FLUX.1-dev | yes | head/token/stage/transition CSVs, spatial and attention figures, `vstar.pt` |
| `highnorm_sinks_massive_channels.ipynb` | The same analysis (sink layers, shared direction, sink shape) generalised across model families through adapters, plus overlap/robustness checks and a causal ablation suite (magnitude clamp, direction replacement, transfer). | FLUX.1-dev, FLUX.1-schnell, FLUX.2-dev, FLUX.2-klein, PixArt-Sigma (512 and 1024) | yes | head/token/stage CSVs, `vstar.pt`, per-model figures |
| `sparse_structures_repro.ipynb` | A from-scratch reproduction: register census, the core direction-vs-magnitude ablation, an angular-drift prediction, analytic query-key alignment, an MLP neuron search for the writer circuit, text-stream census and padding-sink masking, plus channel-ablation and register-control-law experiments. | FLUX.1-schnell, FLUX.1-dev, PixArt-Sigma-XL-2-1024-MS | yes | per-model output folders with CSVs, images and figures |
| `causal_mechanism.ipynb` | Six causal experiments on the register mechanism: natural-register removal, live channel suppression, direction regeneration, writer localization, a sufficiency ladder, and what ends the register state. Needs a frozen discovery artifact (fits one first if none is saved). | FLUX.1, PixArt-Sigma | yes | discovery artifacts, per-experiment result tables and figures |
| `direction_vs_magnitude.ipynb` | Separates norm from `v*`-alignment by construction: an alignment multiplier at fixed norm, a magnitude multiplier at fixed direction, a rescue arm that restores one or the other, and a true rotation in the plane of `v*`. Reads its frozen `v*` and layer ranges from the discovery artifact rather than fitting its own. | FLUX.1-schnell | yes | depth-response and rotation-retention CSVs, figures |
| `register_retiming.ipynb` | Writes, removes or holds the natural register state at other depths of the network (before its natural formation, in place, after its natural end, or past where it would normally decline), with matched controls, and measures the effect on the generated image and on downstream computation. | FLUX.1-dev, FLUX.1-schnell, PixArt-Sigma | yes | a protocol file and per-unit result folders |
| `paper_figures.ipynb` | Redraws the retiming grid from a register-retiming run's saved outputs (no GPU needed), and builds a teaser figure of one generation's image, token maps and per-token arrows against `v*` (needs a GPU, once per model). | reads outputs from the other notebooks; the teaser figure needs FLUX.1-dev or PixArt-Sigma | Part A: no; Part B: yes | figures and LaTeX caption macros |

Each notebook is self-contained: it names the run outputs or artifacts it needs (a
protocol file, a discovery artifact, a set of CSVs) rather than assuming another notebook
ran in the same session.

## Scripts

`scripts/make_fig4_direction_at_fixed_norm.py` redraws the alignment-at-fixed-norm figure
from saved results, without a GPU:

```bash
python scripts/make_fig4_direction_at_fixed_norm.py [--data results/q13] [--out figures]
```

It reads the two CSVs in `results/q13/` (a depth sweep and a rotation run, both from
`direction_vs_magnitude.ipynb` on FLUX.1-schnell) and writes
`fig_direction_at_fixed_norm.pdf` and `.png` to `--out`.

`scripts/highnorm_sink_analysis.py` is a standalone script version of the high-norm-token
vs. attention-sink capture and analysis: it instruments FLUX transformer blocks, captures
residual norms and image-to-image attention, and writes head-level and token-level CSVs
plus diagnostic figures under `outputs/highnorm_sink_analysis`. It needs a GPU and
downloaded FLUX weights, and takes command-line arguments for model, resolution, prompts
and seeds; run it with `--help` for the full list.

## Tests

`tests/` covers the `ditsinks` package against synthetic data, so nothing needs a GPU or a
downloaded checkpoint:

- capture hooks and attention patching (`test_capture.py`, `test_attention_append.py`)
- the causal engine, operators, records and statistics (`test_causal_engine.py`,
  `test_causal_ops.py`, `test_causal_records.py`, `test_causal_stats.py`)
- discovery of `v*` and the register layer ranges (`test_discovery.py`)
- the direction-vs-magnitude control surface and its geometry (`test_control_surface.py`,
  `test_geometry.py`)
- the retiming lifecycle, end to end (`test_lifecycle.py`,
  `test_lifecycle_delivery_e2e.py`, `test_q16_main.py`, `test_q16_main_e2e.py`)
- quantization, prompt sets, and every figure-drawing function (`test_quantization.py`,
  `test_prompt_sets.py`, `test_causal_figures.py`, `test_paper_figures_module.py`,
  `test_q16_figures.py`)

Run the full suite with `pytest` (or `pytest tests`), or a single file with
`pytest tests/test_name.py`.
`test_analysis.py` also checks that every figure follows the shared plotting conventions:
labelled axes, a fixed colour-vision-safe palette, and no legend covering data.
