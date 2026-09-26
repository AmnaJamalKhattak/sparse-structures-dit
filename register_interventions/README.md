# Channel recurrence and register interventions

Six self-contained Colab notebooks, one FLUX.1-dev and one PixArt-Sigma version of each
of three experiments. They don't import any package from this repository. Each installs
its own dependencies, mounts Google Drive and writes everything to one Drive folder per
model: `MyDrive/flux_ma_channels` for FLUX.1-dev and `MyDrive/pixart_ma_channels` for
PixArt-Sigma.

| Notebook | What it runs |
|---|---|
| `channel_recurrence_flux.ipynb`, `channel_recurrence_pixart_sigma.ipynb` | Ranks the channels of the image hidden state by magnitude in each of 100 scenarios (20 prompts by 5 seeds) at one block and denoising step, chosen by fixed rules on a five-scenario pilot, and measures how often the same top channels recur across scenarios. |
| `register_dissolution_flux.ipynb`, `register_dissolution_pixart_sigma.ipynb` | Paired interventions that test what ends the high-norm register state at the end of the band: suppressing or amplifying the catching-up channel, reprojecting the register onto its direction `v*`, and matched controls. |
| `restoration_mediation_flux.ipynb`, `restoration_mediation_pixart_sigma.ipynb` | Lesion and rescue experiments on the register token at one block and step: after a lesion, one candidate mediator (the `v*` projection, the register key, or the dominant channel) is restored to its clean value and the attention sink is measured. |

## Order

The notebooks of one model share its Drive folder and build on each other.

1. Run the channel recurrence notebook first. It writes the prompt list, the pilot
   decision, the prompt-embedding cache and the full tensors used to fit `v*`.
2. The register dissolution notebook reads those files and freezes its own decision
   and `v*`.
3. The restoration notebook reads the channel recurrence outputs. If the dissolution
   notebook's frozen decision and `v*` are present it reuses them, otherwise it refits
   `v*` with the same definition.

Every notebook stops with a message if a file it needs is missing. Each opens with a
"How to run" cell that lists its inputs, settings and outputs.

## Requirements

- A Colab GPU runtime. An A100 is recommended and is needed for the bf16 FLUX.1-dev path.
- For FLUX.1-dev, access to the gated model on Hugging Face and a read token stored as
  the Colab secret `HF_TOKEN`. PixArt-Sigma is not gated.

Approximate cost on an A100:

| Experiment | FLUX.1-dev | PixArt-Sigma |
|---|---|---|
| Channel recurrence (pilot and main run) | about 1.5 to 2 hours, 4.5 GiB | under 1 hour, 1.5 GiB |
| Register dissolution (8 conditions, 100 scenarios) | about 5 hours, 10 to 25 GiB | about 1 hour |
| Restoration (12 conditions, 100 scenarios) | about 3 hours, 9 to 20 GiB | about 1 hour, 5 GiB |

All runs can be resumed. After a disconnect, run the notebook again from the top and
finished work is skipped.
