# Q16 main run — protocol

Notebook: `notebooks/iclr_q16_main.ipynb`. Library code: `ditsinks/q16_main.py` (protocol, units,
pooling) and `ditsinks/q16_figures.py` (paper figures). Every value and its basis:
[`PARAMETERS.md`](PARAMETERS.md).

## Question

What happens to the generated image, and to the computation that produces it, when the
natural register state exists at a different depth of the transformer than it naturally
does? The register state here is the sparse set of high-norm, v\*-aligned image tokens that
act as attention sinks.

## Conditions (per prompt–seed pair)

| Key | Formal name | What is done |
|---|---|---|
| `reference` | Unmodified | the reference generation |
| `induce_E1..E3` | Induced at blocks *a*–*b* | the register state written before natural formation; the natural register is retained |
| `induce_N` | Rewritten in place | the same write inside the natural plateau: the write's own effect |
| `induce_L1..L3` | Induced at blocks *a*–*b* | the register state written after the natural end; the natural register is retained |
| `remove` | Natural register removed | v\* removed from every token more aligned than the unmodified run's ordinary tokens, over the natural interval |
| `remove_induce_E3` / `_L1` | Removed; induced at blocks *a*–*b* | removal, plus the state written just before formation / just after the natural end |
| `extend` | Maintained through block *b* | the natural state held from before its decline through L1 |
| `control_random_direction` | Random direction, norm-matched | the E3 write along a random direction orthogonal to v\* |
| `control_ordinary_positions` | Non-register positions | the E3 write at positions that never carry the register |
| `control_in_distribution` | v\* at in-distribution strength | a v\* write no larger than the recipient block already holds |
| `hooks_only` | Hooks installed, no modification | the first unit only; must be bit-identical to Unmodified |

Every intervention fires at every hooked block during the first third of the denoising
trajectory. At each edited step, the positions and sizes are those of the natural
register at that same step.

## Two stages

1. **Calibration (stage 1).** Unmodified generations on 4 held-out calibration prompts
   (seed 0), recording residual states at the edited steps. The natural interval is
   measured on each (formation, natural end, decline) and every window is placed by rule
   (`plan_depth_windows`). All of it is frozen into `q16_main/<model>/protocol.json`. The
   notebook refuses to overwrite a frozen protocol, or to run stage 2 against one
   calibrated for different edited steps.
2. **Main run (stage 2).** For every prompt–seed pair, in seed-major order, the notebook
   runs every condition and writes compact records. It marks the unit complete, so an
   interrupted session resumes where it stopped.

A unit whose unmodified run has no register at any edited step is a **declared
exclusion**. It is recorded, listed with its reason, never retried, and never pooled.

## Records per unit (`q16_main/<model>/units/pNNN_sS/`)

| File | Contents |
|---|---|
| `images.csv` | LPIPS, PSNR, CLIP image and image–text similarity against the unit's unmodified image |
| `lifecycle.csv` | per condition, recorded step and block: carrier v\* projection, alignment, norm (absolute, relative to the median and to ordinary tokens), high-norm counts (original and new), attention received, sink strength |
| `attention.csv` | per condition, attention step and block: for heads whose clean sink was a register token, where the sink went (kept / another register token / a clean register token outside the carrier set / a non-register token / spread out) |
| `removal.csv` | per removal condition and edited step: whether any consuming block received the natural v\* state, regrowth at the original or new positions |
| `induction.csv` | per inducing condition and edited step: whether the intended state was written exactly |
| `edits.csv` | hook calls, writes and the mean perturbation per condition |
| `boundaries.csv`, `sensitivity.csv`, `rule_audit.csv`, `register_match.csv` | the unit's own natural interval, the threshold range check, the removal rule's footprint on the unmodified run, and the matched targets |
| `images/`, `thumbnails/` | every generation at full size, and at 384 px for figures |

## Analysis (declared before the run)

- **Effects:** each condition's metric is a paired difference from the unit's unmodified
  image. The mean has a 95% bootstrap interval that resamples prompts with all their seeds
  (`effect_table`).
- **Contrasts** (`contrasts.csv`):
  - each depth window against the in-place rewrite (the effect of depth beyond the write
    itself);
  - E3 against each of the three controls (specificity to v\* at the register positions);
  - each relocation against removal alone.
- **State verification before any image is read:** the fraction of edited steps at which
  the induced state was written exactly; the fraction at which no consuming block
  received the natural state under removal; and the removal rule's footprint.

## Figures (`q16_main/figures/`, PDF and PNG, draft captions in `captions.md`)

| Figure | Shows |
|---|---|
| `fig_design` | the measured natural interval, the register population, every window, per model |
| `fig_image_effect_by_depth` | LPIPS and the change in CLIP image–text similarity against the depth of induction |
| `fig_attention_by_depth` | attention received by the register tokens at every block, every condition |
| `fig_writer_response` | what the natural writer adds at the carriers after an earlier write, and new high-norm tokens |
| `fig_removal` | where attention goes after removal; the image effect of removal, relocation and extension |
| `fig_specificity` | E3 against the three controls and the in-place rewrite, with paired differences |
| `fig_examples_<model>` | generations of chosen prompt–seed pairs under chosen conditions, with a full-resolution crop and/or difference map under each row (section 7.1; see below) |
| `fig_threshold_range` (appendix) | the range check of the three shared thresholds |
| `fig_boundaries` (appendix) | the measured formation and natural end on every unit and step |

**The example figure (notebook 7.1).** Most conditions change small details, so the rows and
columns are chosen in 7.1 rather than fixed.

- **Rows:** the prompt–seed pairs you name (`EXAMPLE_UNITS`), or chosen by a rule
  (`EXAMPLE_SELECT`):
  - `most_changed`: the prompts with the largest mean LPIPS over the columns shown;
  - `percentile`: typical cases.
- **Columns** (`EXAMPLE_CONDITIONS`):
  - `depth`: the 7 windows plus removal;
  - `key`;
  - `all`: all 14 conditions;
  - `largest`;
  - or a list of keys.
- **Under each row:** a full-resolution crop of the most-changed region, and/or the
  difference to the unmodified image.

7.1 prints and saves the ranking of every pair (`figures/example_ranking_<model>.csv`). The
rows shown go to `fig_examples_<model>_rows.csv`, and the caption states the selection rule.

## How to run

1. **Merge the branch into `main`** (the setup cell clones `main`), or set `REPO_REF` in 1.1
   to the branch.
2. **FLUX.1-dev session:** set `MODEL = 'flux1-dev'` in 2.1.
   - If the frozen v\* was fitted under a different sampler setting, 3.1 says so. Move the
     old artifact aside and set `RUN_DISCOVERY = True` once.
   - Run through section 4 (stage 1). Look at the protocol table and the design figure.
   - Set `RUN_MAIN = True` and rerun section 5 in as many sessions as it takes;
     `MAX_UNITS_THIS_SESSION` stops cleanly before a runtime limit.
3. **PixArt-Σ session:** set `MODEL = 'pixart-sigma-1024'` and `RUN_DISCOVERY = True` the
   first time, since there is no 1024 px artifact yet. Then do the same as for FLUX.
4. Sections 6–8 pool every complete unit of every model found on Drive and write the
   figures and appendix tables. They can be run at any point.

**Cost:** each unit is about 15 generations with a light trace (residual states at about 11
steps, attention at 3). The notebook prints the measured time per unit and the time left
after every unit.

## Inducing faithfully on PixArt-Σ: both passes of the sampler (notebook section 9)

The induction is the same on both models: v* written at matched high norm, or without high
norm (in-distribution strength), over equal-length windows before formation, in place and
after the natural end. What differs is the sampler. PixArt-Σ runs the model twice per step,
with and without the prompt, and forms each update as uncond + 4.5 × (cond − uncond).
FLUX.1-dev runs it once. So on PixArt-Σ the state is written in **both** passes
(`edit_branches='both'`, 2.1).

The first PixArt-Σ run wrote the conditional pass only. That multiplies every edit by 4.5,
and the early windows collapsed (LPIPS 0.8–1.0). Section 9.2 re-ran three prompt–seed pairs
with both passes edited, and with guidance off:

- the early and in-place writes became small (0.04–0.11), as on FLUX.1-dev;
- the unconditional pass held its one-token register at the same position and size as the
  conditional pass at every edited step.

That first run stays on Drive (`q16_main/pixart-sigma-1024/units/`) and is not pooled into
the results. The both-passes run lives in `both_passes/` under the same frozen protocol.
Section 9.1 reads each model's register profile and, per condition, the state after the
natural end (`register_profile.csv`, `late_state.csv`).

## Corrections this run incorporates

- **The natural end is measured by the register test.** The old "10% of the peak
  projection" rule never fired on FLUX.1-dev, and the end silently fell back to a declared
  range.
- **Neither FLUX nor PixArt lets attention see a token's residual norm.** Both read every
  token through a LayerNorm first. The norm-keeping removal therefore hands the next block
  the same input as the subtractive one; it is an operator control, not a test of whether
  size alone makes a sink. PixArt-Σ tests depth apart from block type. It does not test
  norm.
- **Attention classes include "a clean register token outside the carrier set",** so a
  head that moves to the natural register at a position the carrier set does not name is
  no longer counted as "non-register".

## What it cannot establish

- **Depth and block type are confounded on FLUX.1-dev** (dual-stream before block 19),
  so an early-versus-late difference there is also a block-type difference. PixArt-Σ has
  one block type throughout.
- **A matched state is not a transplanted register.** Only the v\* projection and norm
  are matched; the rest of the token and its history are not.
- **No mediation.** An attention change and an image change in the same condition do not
  show that one causes the other.
- **Positions are read from the unmodified run** (oracle positions). This is a diagnostic
  design, not a deployable procedure.
- **Two passes, one set of positions.** On PixArt-Σ both passes are edited at the
  conditional pass's register positions and sizes; `branch_overlap.csv` reports, per unit,
  whether the unconditional pass holds its register there.
