# Register retiming

Is the register's depth causal, or only its presence? The register state is a sparse
set of high-norm, `v*`-aligned image tokens that act as attention sinks and normally
forms in a bounded interval of transformer depth. This experiment asks whether that
interval matters: can the same state be induced earlier or later, or removed, and if so
does it do the same job.

Library code: `ditsinks/lifecycle.py` (windows, operators, measurement), `ditsinks/q16_main.py`
(protocol, units, pooling) and `ditsinks/q16_figures.py` (figures). Every value used and
its basis: [`parameters.md`](parameters.md). Notebook: `notebooks/register_retiming.ipynb`.

## Conditions (per prompt-seed pair)

| Key | What is done |
|---|---|
| `reference` | the unmodified generation |
| `induce_E1..E3` | the register state written before natural formation, at increasing depth; the natural register is retained |
| `induce_N` | the same write inside the natural plateau: the write's own effect |
| `induce_L1..L3` | the register state written after the natural end, at increasing depth; the natural register is retained |
| `remove` | the natural register removed, over its natural interval |
| `remove_induce_E3` / `_L1` | removal, plus the state written just before formation or just after the natural end |
| `extend` | the natural state held from before its decline through the first late window |
| `control_random_direction` | the E3 write along a random direction orthogonal to `v*`, norm-matched |
| `control_ordinary_positions` | the E3 write at positions that never carry the register |
| `control_in_distribution` | a `v*` write no larger than the recipient block already holds |
| `hooks_only` | hooks installed, no modification; must be bit-identical to `reference` |

Every intervention fires at every hooked block during the first third of the denoising
trajectory. At each edited step, the positions and sizes used are those of the natural
register at that same step.

## How a state is induced

The primary operator (`induce_matched`) gives each carrier the clean natural register's
`v*` projection and norm, both expressed as multiples of the recipient block's median
token norm, so only depth moves and the size stays natural for that block. A
direction-only control (`induce_alpha`) sets the `v*` projection alone and leaves the
token near ordinary norm. Removal is subtractive: the `v*` component is cleared from
every image token more aligned than the unmodified run's ordinary tokens, at every hook
of the removal interval, with a terminal cleanup hook past its end.

Matching the projection and norm is not the same as transplanting the register: the rest
of the token's direction and its history are not matched.

## Two stages

1. **Calibration.** Unmodified generations on held-out calibration prompts measure the
   natural interval (formation, natural end, decline) on each one, and every window is
   placed by rule (`plan_depth_windows`). The result is frozen to `protocol.json` before
   any conditioned image is generated. The run refuses to overwrite a frozen protocol or
   to run stage 2 against one calibrated for different edited steps.
2. **Main run.** For every prompt-seed pair, every condition is run and compact records
   are written. A unit is marked complete, so an interrupted run resumes where it
   stopped. A unit whose unmodified run has no register at any edited step is a declared
   exclusion: recorded, listed with its reason, never retried or pooled.

## Records per unit (`q16_main/<model>/units/pNNN_sS/`)

| File | Contents |
|---|---|
| `images.csv` | LPIPS, PSNR, CLIP image and image-text similarity against the unit's unmodified image |
| `lifecycle.csv` | per condition, step and block: carrier `v*` projection, alignment, norm, high-norm counts, attention received, sink strength |
| `attention.csv` | per condition, step and block: for heads whose clean sink was a register token, where the sink went |
| `removal.csv` | per removal condition and edited step: whether any consuming block received the natural `v*` state, and regrowth |
| `induction.csv` | per inducing condition and edited step: whether the intended state was written exactly |
| `edits.csv` | hook calls, writes and the mean perturbation per condition |
| `boundaries.csv`, `sensitivity.csv`, `rule_audit.csv`, `register_match.csv` | the unit's own natural interval, the threshold range check, the removal rule's footprint, and the matched targets |
| `images/`, `thumbnails/` | every generation at full size and at reduced size for figures |

## Analysis

Each condition's metric is a paired difference from the unit's own unmodified image, with
a 95% bootstrap interval resampling prompts with all their seeds. Pre-declared contrasts
(`contrasts.csv`): each depth window against the in-place rewrite, E3 against each
control, and each relocation against removal alone. Before any image is read, the run
also checks that the induced state was actually written and that removal actually kept
the natural state out of every consuming block; these checks and the removal rule's
footprint are recorded per unit.

## Figures (`q16_main/figures/`)

`fig_design` (the measured natural interval and windows per model), `fig_image_effect_by_depth`,
`fig_attention_by_depth`, `fig_writer_response`, `fig_removal`, `fig_specificity`,
`fig_examples_<model>` (generations under chosen conditions, with a crop and/or
difference map under each row), and two appendix figures: `fig_threshold_range` (the
range check of the three shared thresholds) and `fig_boundaries` (the measured formation
and natural end on every unit and step).

For `fig_examples`, rows (prompt-seed pairs) and columns (conditions) are chosen by rule
in the notebook rather than fixed; the ranking used is written to
`figures/example_ranking_<model>.csv` and the rows shown to `figures/example_ranking_<model>_rows.csv`.

## How to run

1. Set the model and run stage 1 (calibration) first; the notebook prints the protocol
   table and the design figure so it can be checked before any main-run image is
   generated.
2. Set `RUN_MAIN = True` and run stage 2 in as many sessions as needed; a session limit
   on units stops cleanly before a runtime limit, and a unit already marked complete for
   the frozen protocol is skipped.
3. Repeat for the other model. Sections that pool results and write figures can be run
   at any point once some units are complete.

Each unit is about 15 generations with a light trace (residual states at about 11 steps,
attention at 3 steps). The notebook prints the measured time per unit and the time left.

On PixArt-Sigma the sampler runs the model twice per step, with and without the prompt,
and forms the update as `uncond + guidance x (cond - uncond)`. Writing the induced state
into the conditional pass alone multiplies its effect by the guidance scale, so both
passes are edited (`edit_branches='both'`), each with its own remainder, at the register
positions and sizes of the conditional pass. `branch_overlap.csv` checks, per unit,
whether the unconditional pass holds its register at those same positions.

## What it cannot establish

- **Depth and block type are confounded on FLUX.1-dev.** The early window lands in
  dual-stream blocks and the late window in single-stream blocks, so an early-versus-late
  difference there is also a block-type difference. PixArt-Sigma is uniform single-stream
  and is the cleaner test of timing.
- **A matched state is not a transplanted register.** Only the `v*` projection and norm
  are matched; the rest of the token and its history are not.
- **No mediation.** An attention change and an image change in the same condition do not
  show that one causes the other.
- **Positions are read from the unmodified run** (oracle positions), which is a
  diagnostic design, not a deployable procedure.
