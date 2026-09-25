# Q16 — how the experiment is implemented

Companion to `EXPERIMENT_DESIGN.md`, which says *what* the experiment is and *why*. This
one says *how the code works*: where each piece lives, the order things happen in, the
exact arithmetic, and which choices are enforced by the program rather than by discipline.

**Status: built and verified offline. No real model has been run for Q16.** Every number
quoted below as "measured" comes from FLUX.1-schnell calibration and sweep runs that
produce **no images**; the image pilot is still gated.

---

## 1. Where it lives

| file | lines | role |
|---|---|---|
| `ditsinks/lifecycle.py` | 2743 | every Q16-specific operator, rule and classifier |
| `tests/test_lifecycle.py` | 1661 | 103 tests; all of them pin a claim the write-up makes |
| `notebooks/iclr_q16_lifecycle_retiming.ipynb` | 26 code cells | the runnable experiment |
| `experiments/q16_lifecycle_retiming/` | — | design, this file, and the run records |

Nothing in Q13, Q14 or Q15 was modified. Q16 reuses their engine and adds one module.

### What is reused rather than rebuilt

`run_traced_generation` and `EditPlan` (paired generation, and the ability to compose
several plans at different layer ranges in one pass — which is what makes a *schedule*
expressible at all); `CausalTracer` (per-block projection, cosine, norm, dominant channel,
incoming attention, `qk_cosine`, `image_mass`, `n_keys`); `control_surface_states`
(the β/γ operator); `sink_readout`; `select_frozen_targets` and `select_treatment`;
`CS.image_metrics`; the contact-sheet conventions.

---

## 2. The order things happen in

```
1.1  clone/refresh the repo (Colab secret only, never printed)
1.2  run the test suite BEFORE any weights download
2.1  the only cell you edit: prompts, windows, thresholds, gates
2.2  mount Drive; refuse to proceed on Colab without it
3.1  load the frozen v* for THIS checkpoint, or say plainly none exists
3.2  [gated] fit and freeze a v* on DiffusionDB, disjoint from the pilot prompts
3.3  resolve PROVISIONAL windows from the frozen LayerRanges; detect the seam
4.1  clean run across EVERY block, with a DeliveryProbe; freeze the carriers; build
     the clean reference (Q1 bars and every head's clean sink, every step) before
     thinning; probe the feed-forward input for the site pilot
4.1b what the consuming operations receive (secondary); MEASURE formation, decline and
     end, and place every window on them (lifecycle_schedule)
4.1c re-select the carriers at EVERY denoising step and compare with the frozen
     set; decide per-step induction from that, before any condition exists
4.2  calibrate the direction-only CONTROL's coefficient, and the natural-register-
     matched targets per (step, block) for the PRIMARY arms; report whether they are
     unusually large for the recipient, and what they do not match
4.3  [gated] feasibility sweep — internal signatures only, NO images
4.4  FREEZE the dynamic alignment rule and audit it on clean at every step; build ONE
     suppression schedule; hard-stop on any block with both kinds of hook
4.5  [gated] induction-site pilot: block input vs the feed-forward input (Q4)
5.1  [gated] the conditions, with images and a DeliveryProbe each
5.2  validity gates (sham, one schedule, no overlapping hooks, writes at the hook)
5.3  THE FOUR RESULTS: (a) v* state at consuming sites, (b) regrowth vs relocation,
     (c) sink persistence vs redistribution, (d) induction written; depth plots;
     which comparisons are interpretable
6.1  achieved lifecycle, every condition, every block
6.2  levels + achieved-lifetime classification
6.3  indirect effects on a window that was never hooked
6.4  SECONDARY: the consumer-side cosine, with how many blocks it could score
6.5  NATURAL + LATE: continuous extension, or a second state after a gap
6.6  the lifecycle in depth AND denoising time
6.7  matched vs direction-only: alignment, norm, sinkhood, persistence, block by block
7.1  paired image metrics
7.2  contact sheet + the achieved-lifecycle figure
7.3  the Stage-5 decision
```

### The dose has two axes, and one of them was nearly zero

`EditPlan.steps` decides which **denoising** steps an edit fires at, and the first pilot
passed `[STEP]` — one step out of twenty. Measured on the synthetic checkpoint with
everything else identical, that is 4 edit calls where `steps=None` gives 32; on the real
run, 24 against 480. A diffusion sampler repairs what it is given room to repair, so the
pilot's images came back visually identical to clean while the sham was *exactly* clean
and every real arm was non-zero and rank-ordered by hook count — a dose-response at a
dose too small to see.

`INTERVENTION_STEPS` (2.1) resolves to `EDIT_STEPS` in 3.3 and reaches every `EditPlan`:
`'all'` (the default, `steps=None`), `'capture'` (`[STEP]`, the mechanistic readout), or
an explicit list. 3.3 prints the coverage and warns below 50%; 5.1 records `edit_calls`
and perturbation energy per condition to `dose.csv`.

Two pieces of accounting had to follow it. `suppression_verdict` takes `step` — without
it, a schedule firing at every step interleaves twenty sweeps down the stack and the next
step's first hook reads as regrowth after the previous step's last. `Suppressor.seen` is
keyed by step for the same reason: one pooled set reports zero relocation from step 1 on.
Both are pinned by tests.

Three gates, all `False` by default: `RUN_DISCOVERY`, `RUN_SWEEP`, `PILOT_PROCEED`. The
sweep is separated from the pilot deliberately — it answers "is induction possible at all"
for the cost of image-free generations, before any image study.

---

## 3. Windows

`windows_from_ranges(ranges, length, n_layers)` derives early / natural / late from the
checkpoint's **frozen** `LayerRanges`, before anything runs:

- **early** = the `length` blocks ending immediately before the writer range begins;
- **natural** = the register range;
- **late** = the `length` blocks starting one block after the natural range. It may be
  moved further out with `late_first`, never in: the library refuses a value inside the
  natural range, and 3.3 hard-stops if either window touches the natural lifetime.

Early and late are the **same length by construction** — an early window with more
intervention sites than the late one would confound depth with dose, and every
early-vs-late comparison would be uninterpretable.

A window that cannot exist is **omitted, never shifted**. A writer range starting at block
0 leaves no room before it; a late window past the end of the stack is clipped or dropped;
a register range containing no block raises. Conditions needing a missing window are
**skipped and reported**, not approximated at another depth.

**Extension gets a bridge, not a moved late window.** Cell 4.1b runs
`measured_dissolution_onset` on the clean trajectory (`DISSOLUTION_FRACTION = 0.9`: the first
block where the carriers' projection falls below 90% of its plateau) and
`extension_bridge(natural, late, onset)` turns it into the blocks from the onset to the block
before the late window. 5.1 induces over the bridge *then* the late window, but only for the
condition that keeps the natural window intact (C). D suppresses the natural window and has
nothing to bridge. The bridge is calibrated at its own first block, and its label, its
retention report in 5.2 and its shading in 6.6 and 7.2 all read `bridge`. The first build
moved the late window itself to the onset, into the natural range, which made "late"
unattributable to lateness.

`continuity_check` (6.5) then decides which of the two actually happened, against one
absolute bar set from the **clean plateau** — a bar tracking the clean decline would call
dissolution itself continuous. A gap anywhere between the plateau and the late window's end
labels the run `late_re_induction`, and 7.3 carries that label into the image table.

`Window.crosses(boundary)` flags a window straddling FLUX's dual/single seam, and cell 3.3
prints an explicit confound warning when early and late land in different architectures —
which on FLUX they always do, since early must precede the writer at 17 and the seam is at
19.

**The suppression interval is separate from the natural window.** It defaults to the
**writer onset** (`artifact.layer_ranges.writer[0]`), not the register range, and a hard
stop fires if it starts later. An earlier build defaulted to the register range, so the
executed schedule suppressed 20–39 while the writer operated at 17–19 — the state was
written in full and only then removed, which is a different experiment.

---

## 4. The operators

### Induction, primary — `induce_matched`

```
x' = a_t · v̂ + sqrt(max(n_t² − a_t², 0)) · r/‖r‖      a_t = π_t·m_L,  n_t = ρ_t·m_L
```

- **Per batch row.** A batched tensor has each row edited with its own remainder. Inside
  the engine an edit is handed only the conditional row's `[N, C]` slice (as for every Q16
  operator), so the unconditional branch is not edited.
- **Edge cases, left alone rather than invented.** A token whose remainder is numerically
  zero has no direction to scale, so it is skipped. A norm target below the projection
  target is clamped: the token becomes pure `a_t · v̂`.
- **Size on the clean tensor.** `_matched_perturbation` gives the size of the edit from a
  token's norm and projection alone:
  `‖x' − x‖² = (a_t − α)² + (‖r'‖ − ‖r‖)²`. A test checks it against the real edit.
- **`RegisterTarget` / `RegisterTargets`.** A block's population targets plus per-token
  `(a_t, n_t)` for the natural carriers, keyed by `(step, block)`. A fallback exists only
  where one is registered with `step=None`; otherwise an uncalibrated step or block is
  left untouched. That is the same rule `lifecycle_edit` applies to per-step coefficients.
- **`calibrate_register_match`.**
  - It reads the natural plateau's **inputs** (the previous block's output in the trace)
    and each recipient's input.
  - Block 0's input is the embedding, which the trace never records. There it falls back
    to a probed input slice.
  - It returns one report row per block. The row carries the targets, the recipient's
    largest clean norm and projection, `norm_unusually_large`,
    `projection_unusually_large`, the target's percentile, `perturbation_vs_median` and
    `projection_only_norm_ratio`.
  - Given full slices, the row also carries the remainder check: the carriers' remainder
    coherence against ordinary tokens', and the same-position remainder cosine against an
    ordinary-token baseline.
- **Records.** Every write is recorded as an `EditRecord` of kind `induce_matched`. Its
  `crossed_highnorm_threshold` is measured against the recipient's bar, not asserted.

### Induction, control — `induce_alpha`

```
x' = r + α_target · v̂        implemented as   x += (α_target − α) · v̂
```

**Not norm-preserving, by design.** The natural register *is* a high-norm state; holding
the recipient at its original length would test something else. Written as an addition
because subtracting a large `α·v̂` and adding another back loses precision in `r` exactly
where the claim "only the `v*` component moved" is made — adding a multiple of `v̂` leaves
the orthogonal complement invariant *by construction*, which a test checks.

### Suppression — the repository's operator, asserted not assumed

```
y = r,   x' = ‖x‖ · y/‖y‖
```

A test asserts this is **bit-identical** to `control_surface_states(β=0, γ=1)`, and a
second asserts that operator's `identity_error` at `(1,1)` is below 1e-5. The three
suppression conditions only compare if all three suppress the same way, and
"reimplemented identically" is not a guarantee.

Explicitly *not* used, and not silently substituted: zeroing whole token vectors,
suppressing channel 154/293 globally, or masking incoming attention.

### Maintenance

Every operator installs as an `EditPlan` over **every block in its window** at the
declared denoising steps, not as a one-shot edit. A single edit the model repairs would
otherwise score as a lifecycle shift.

Two library functions read maintenance back. Both read block **outputs**, against the
clean natural register on its plateau.

- **`maintenance_summary`** reports, per window: alignment, the norm ratio, sink strength,
  the share of blocks that are sinks, and each of those against its natural value. It
  also counts how many unhooked blocks after the window keep at least half the excess
  over clean. That count is taken separately for the projection, the norm ratio and the
  sink strength.
- **`maintenance_profile`** gives the per-block view. It puts what the operator **wrote**
  at the hook (from the `EditRecord`s at one step) next to what the block **handed on**,
  with the clean value beside both.

---

## 5. Suppression: one class, one schedule, a dynamic clean-referenced rule

`Suppressor` has one operator per condition and one schedule; only the selection differs.

**The operator.** `operator="subtractive"` is the default and the primary:
`x' = x − (x·v̂)v̂`, and the token is not rescaled back to register size.
`operator="norm_preserving"` is the validated β=0, γ=1 operator. A test pins it
bit-identical to `control_surface_states`. It runs as `F_norm_preserving`, an operator
control (not a sink-identity test: every block reads its input through a LayerNorm, so it
hands the next block the same input as the subtractive operator). The notebook knob is
`SUPPRESSION_OPERATOR`.

**`mode="state"`, with `AlignmentRule`: what the image conditions use.** At every hook,
over **every image token**, remove `v*` from any token whose `cos(x, v*)` reaches the
clean run's ordinary 0.999 quantile *at that same (denoising step, block)*. The bar comes
from `CleanStateReference`, built from the clean run in `after_run` before the per-step
attention is thinned. `for_input_of(step, L)` reads block L−1's output, which is what a
hook at L edits. A step that was not recorded falls back to the nearest one, and says so.

The rule has no norm bar, and its bar is not a percentile of the edited tensor. Both
choices close escape routes the earlier conjunction rule (`RegisterStateRule`, kept for
the drift audit) left open:

- aligned-but-small precursors or new carriers slipped under its norm bar;
- its percentile always selected its share of tokens, register or not.

The rule follows the state wherever it goes; Q1 found that another carrier takes over
after removal in most runs.

**`mode="fixed"`: the diagnostic arm X.** The frozen clean positions only, every hook.
Because it never targets anything else, relocation elsewhere stays visible. Its image is
not comparable with the state-based conditions.

**What each hook records** (`SuppressionSite`):

- `chosen_ids`;
- the selection split into `n_original` (the original clean carriers coming back) and
  `n_new` / `new_ids` (positions that never carried it);
- **what the block received**, measured directly on the tensor the hook handed on:
  `received_max_cosine`, `received_n_above_bar`, `received_n_register_like` and
  `received_carrier_cosine`;
- `share_of_image`, the share edited at that hook in the intervened run.

### The audit that precedes the freeze

`rule_selectivity` runs the rule over the clean trajectory at every recorded step, for
every hook of the interval, terminal cleanup included. It reports the count, the share of
the image, and the overlap with the frozen carriers. The notebook refuses a rule touching
more than `RULE_MAX_SHARE_OF_IMAGE` at any hook, and records the frozen rule, operator and
schedules in `records/suppression_rule.json`. At the 0.999 quantile the rule touches the
register plus about 0.1% of ordinary tokens per hook.

---

## 6. Verification

### Four separate results (cell 5.3)

Each result is computed for every suppressing condition, at every recorded denoising
step, by one function each.

- **(a) `received_state_rows` / `received_state_summary`: the natural `v*` state at
  consuming sites.**
  - The rows cover every block from the start of the interval through the first unhooked
    block after the terminal cleanup.
  - At a hooked block they read the suppressor's direct measurement. At an unhooked block
    they read the previous block's output.
  - `received_n_register_like` counts tokens more aligned than *every* ordinary token of
    the paired clean run at that input (`BlockBars.ordinary_max_cosine`).
  - `received_max_cosine_vs_natural` is how close the most aligned received token came to
    the clean carriers' alignment.
  - The result is one of `no v*-aligned state received`,
    `v* state received inside the interval`, or `v* state received after the interval`.
    The last means the interval held and the first block after it received a state its
    predecessor rebuilt.
- **(b) `regrowth_relocation_rows` / `regrowth_relocation_summary`.** This reads block
  outputs, which is what the next hook sees.
  - It counts the original carriers above the clean ordinary maximum, and those also
    meeting Q1's register criterion.
  - It counts new aligned and new register-like positions, excluding any token that is
    register-like in the paired clean run at that block.
  - It reports unique positions, blocks per position, and an outcome in Q1's vocabulary,
    with novelty established.
- **(c) `attention_relocation_rows` / `attention_relocation_summary`.** Per head against
  the clean sink of the same head, step and block, stored in `CleanStateReference`. The
  rows are computed in `after_run` before thinning, into `ATTENTION_ROWS`.
  - Reported: the share of heads keeping the clean sink, and concentration against clean.
  - For each *affected* head (its clean sink was an original carrier) the class is one
    of: `kept`, `other_original_carrier`, `relocated_vstar_carrier`,
    `non_register_token` or `spread_out`.
- **(d) `induction_written`.** Per step, whether every write in a window hit its target.
  A matched write must hit projection *and* norm; a direction-only write, projection
  only. What the blocks handed on is read from the per-step lifecycle table.

5.3 also draws the clean-versus-intervention depth plots and the per-step table. It then
marks E vs F, D vs F and C vs A interpretable only where (a) held at every step on the
suppressing sides and (d) was established at every step.

### The induction site pilot (cell 4.5)

- `transplant_edit` replaces a token's vector at the hooked site with a source state. By
  default it keeps the recipient's norm, which is all a LayerNorm input can carry anyway.
- `align_edit` sets a token's `cos(x, v*)`, keeping its norm and its remainder direction.
- At `PRE_MLP_RESIDUAL` both change only what the feed-forward reads: the hook is on
  `norm2`'s input. An e2e test checks, on FLUX and PixArt, that only the edited token's
  block output changes.
- `site_pilot_rows` reports the post-block state, each block's own write along `v*`
  against clean, and sink strength.

### Two tensors, two questions

> The delivery half of this section now describes the *secondary* consumer-side view
> (6.4). Suppression is reported by the four results above.


The suppressor acts on each block's **input**. Two different things follow from that, and
the implementation keeps them apart because only one of them can invalidate a condition.

**Regrowth — `SuppressionSite`, at the hook.** What the rule sees at hook *k+1* is what
block *k* produced, so `n_selected` after the first hook **is** the regrowth measure, with
no extra threshold to choose. `newly_targeted` separates "the same tokens keep coming back"
from "the state is moving". Reported as a rate, a first block and a mean number of tokens
rebuilt per block. **It gates nothing.**

**Delivery — `DeliveryProbe`, at two points.** It registers a pre-hook at `BLOCK_INPUT`
(`supplied_*`, the residual the rule is evaluated on and a `BLOCK_INPUT` edit rewrites) and
a pre-hook at `ATTENTION_INPUT` (`consumed_*`, the tensor the image self-attention is
*called* with). Both reduce to three per-token vectors at the hook — a few kilobytes per
block against the ~50 MB a full residual slice would cost at every block of every
condition. The probe is entered around the generation, so its `BLOCK_INPUT` hook registers
before the edit installer's and reads the state as the block received it, which makes
`supplied_*` the regrowth reading at the rule's own tensor. `from_rows` rebuilds a probe
from saved statistics, so the delivery analysis can be redone at a different tolerance
without another generation.

`ATTENTION_INPUT` is a new intervention point, and it exists because the norm module's
output is the wrong tensor on one of the two families. FLUX's
`AdaLayerNormZero`/`AdaLayerNormZeroSingle` apply scale and shift inside the module;
PixArt's `BasicTransformerBlock` computes `norm1(x) * (1 + scale_msa) + shift_msa` in the
block body. The first build probed the module output and so measured a pre-modulation
tensor on PixArt — delivered cosines off by up to 0.16, norm spread reading 1.00× against a
true 1.07–1.24×. Two e2e tests now require the two tensors to be identical on FLUX and to
differ on PixArt.

`delivered_rows` turns the statistics into per-block rows (secondary view, 6.4). `consumed_norm_spread` is the
one to read first: LayerNorm divides token magnitude out and the modulation can put some
back, so whether the consumer sees magnitude is measured per run. Measured on the live
model: consumed 1.04–1.15× against a 2.4× residual on tiny-flux1, 1.07–1.24× against 1.3×
on tiny-pixart — low enough that delivery is an alignment question on both.

`delivery_verdict` scores each block as the excess alignment over the clean run's **ordinary
ceiling at that same block**, as a share of what the clean register delivers there
(`DELIVERY_TOLERANCE`, 0.25). The terminal boundary block is scored separately and always
reported; an unprobed one is `UNMEASURED`, not closed. Where the clean carrier delivers no
more alignment than the clean ordinary ceiling, the verdict is `not measured`.

Cell 6.4 reports both, side by side, per condition and per block across the whole interval
**from the intervened runs**: selected at hook, projection entering and after the hook,
carrier projection at the block *output*, high-norm counts — beside the delivered cosine,
the clean ordinary ceiling and the delivered register share. Cell 4.4 is the *clean-run*
audit of what the rule would select; they answer different questions and the notebook says
so.

### `suppression_schedule` — the terminal boundary

A hook at block *k*'s input protects block *k*'s consumers. The last hook is at the
interval's last block, so that block's output reaches the next block's attention with
nothing between. `suppression_schedule(window, n_layers, stop_before)` returns
`hook_layers` = the interval **plus one terminal cleanup block**, and truncates the interval
when an induction begins inside it (`stop_before`), landing the cleanup on the induction's
first block. Plan order is execution order for two `BLOCK_INPUT` pre-hooks, and 5.1 builds
suppression first, so that block is cleaned and then written. A fully overlapped interval
raises rather than silently emptying.

### `delivery_coverage` — does the rule miss a precursor?

Run on the **clean** trajectory in 4.4. At every block it applies the rule to `supplied_*`
and counts the unselected tokens whose `consumed_*` alignment is at or above the weakest
the rule caught — the precursor case, a token strongly aligned but still below the norm
bar, which LayerNorm renders indistinguishable at the consumer. `bar_to_cover` per block,
`coverage_recommendation` for the interval. `BROADEN_RULE_TO_COVERAGE` is **False** by
default; set True, the notebook re-audits selectivity at the proposed bar and hard-stops if
it exceeds `RULE_MAX_SHARE_OF_IMAGE`.

### `suppression_verdict` (superseded)

No longer called by the notebook; the four results above replace it. It is kept in the
library for reanalysis of older runs. It returned `complete` only when delivery is `prevented`, the terminal boundary is closed, **and** the
suppressed carriers never regain sink behaviour. Four deliberate choices:

- **Regrowth does not gate.** An earlier build failed a condition on it. That would call a
  perfectly guarded interval incomplete merely because the network kept trying, and would
  invite escalating the intervention until the number looked right.
- **The raw `v*` projection is reported and does not gate.** A test pins that a 60%
  projection residue with delivery prevented is still `complete`.
- **Sink presence is not evidence of failure.** An earlier build counted any sink outside
  the frozen carriers, and a clean FLUX block carries 130–200 — every condition, clean
  included, was marked failed by construction. The check now asks whether the *suppressed
  carriers* became sinks again; relocation beyond the clean level is reported separately.
- **`incomplete` classifies and continues.** It does not abort. The image is still
  produced and labelled, because "explicitly report that complete suppression was not
  achieved" requires the report to exist. No delivery measurement gives `not measured`,
  never `complete`.

---

## 7. Calibration

| mode | anchor | status |
|---|---|---|
| `recipient_outlier` | the recipient layer's largest clean `v*` projection | **PRIMARY** |
| `ratio` | the natural carrier's `α / median‖x‖`, rescaled | reachability question |
| `absolute` | the natural `α`, unscaled | reachability question |
| `cosine_matched` | `α` such that `cos(x, v*)` matches the natural carrier's | **SECONDARY CONTROL ONLY** |

This table sizes the **direction-only control** (and the sweep and weak arm). The
**primary** arms are sized by `calibrate_register_match` (§4). It takes the natural
register's projection **and** norm as ratios to the median, averages them over the clean
plateau at the same denoising step, and scales them to each recipient block's input
median. Its report says, per block, whether the result is beyond anything a clean token
holds there. Nothing, primary or control, is calibrated on attention.

Cosine matching is deliberately *not* the primary anchor: depth-dependent query–key
preference is one of the outcomes being measured, and anchoring strength on it would
calibrate away the mechanism under study. Its own `note` string says so, and a test asserts
that text is present so it cannot quietly become a default.

`out_of_distribution_ratio` = target ÷ the largest absolute projection the recipient layer
reaches in the clean run. Above 1.0 the induction asks for a state the layer never
naturally contains. It is **reported, never clipped**.

*Why the primary anchor changed:* a first sweep in `ratio` mode passed every level at
every strength across an 8× range — not feasibility but **saturation**. The natural
register's `α` is 13.3× the natural-window median norm, so `ratio` reproduced that outlier
wherever applied; at block 14 it asked for 15× the loudest projection the layer holds.
With `recipient_outlier` a threshold appears.

---

## 8. What is measured, and how it is classified

### Per block, every condition, every block — `achieved_lifecycle`

Carrier `v*` projection; all-token projection p99 and max; carrier norm; median norm;
high-norm counts **split frozen vs new**, with the full id sets; dominant channel and its
value; carrier **incoming attention mass** (absolute share of the distribution) and sink
strength; sink counts split frozen vs new; `qk_cosine` and carrier rank.

Attention is scored **absolutely** — against uniform over the whole key sequence, text
included — using `sink_readout(absolute=True)`. The image-renormalised default would
divide out exactly the mass a retiming condition moves. The scale used is written into
every row, so a fallback is visible rather than silent.

### Four levels, never a ladder

`L1` a `v*`-aligned state stands out at that depth (vs the clean p99 *there*) ·
`L2` it is also high-norm (vs that layer's **own** bar) ·
`L3` it draws attention (population **mean** sink strength, not `any`) ·
`L4` the image changes.

**On FLUX, L2 is not a step toward L3.** `norm_k = RMSNorm(dim_head)` is applied to the
key before RoPE and before the dispatch, so a token's residual magnitude cannot reach the
attention logit at all. Measured on a real forward pass: residual norms spread 3.46× across
tokens while key norms spread **1.00×** — exactly constant. Norm still reaches the *value*
path, so L2 is on the causal path to the **image** but not to the **sink**. PixArt has no
`norm_q`/`norm_k` at all, so magnitude *does* reach its logits — the same intervention is
not the same experiment in the two models.

`L3` uses the population mean because `any` is satisfied by one token of sixteen, and at a
depth just after dissolution some carriers are *already* sinks in the clean run — an
`any`-based level fired at a near-no-op perturbation and reported baseline sinkhood as
successful induction.

### `classify_achieved_lifetime` — what the image comparisons are grouped by

The configured hooks say what was attempted. This says what happened, measured as **excess
over clean** at the carrier positions:

| outcome | condition |
|---|---|
| `continuous_extension` | excess still present where the induction and natural windows meet |
| `pulse_then_natural` | excess collapses before natural formation, which then proceeds as in clean |
| `displacement` | frozen carriers lose their high-norm presence and/or the population relocates |
| `no_induction` | no excess in the induction window |

The image table carries an `achieved` column beside a `suppression` column, and prints:
*conditions whose achieved lifetime differs are not the same experiment and must not be
averaged together, however similar their configured schedules look.*

### `carrier_census` — unique positions, not token-layer observations

`n_highnorm_new` is a **per-block count**, so a sum across a window is token-**layer**
observations: 395 can be ten positions crossing the bar at forty blocks or 395 crossing
once — opposite answers to "did early induction relocate register formation?". The census
reports unique positions plus the median number of blocks each stayed high-norm. Small
unique count with high persistence is a **stable relocated population**; large unique count
with a median near 1 is **threshold flicker**, and the cumulative number reflects the bar
rather than a moved register.

### `persistence_class` — with the clean baseline, which is not optional

`maintained` / `carried` / `reconstructed`, and `not measured` when no unpatched block
follows the window. **It must be given the clean rows.** The blocks after an early window
are exactly where the natural register forms, so an absolute reading there measures the
model's ordinary behaviour and reports it as the induced state persisting — every early
condition came back "carried" or "reconstructed" whether or not anything survived.

---

## 9. Conditions

| id | early | natural | late | role |
|---|---|---|---|---|
| `A_clean` | — | — | — | reference |
| `B_early_natural_intact` | induce | — | — | test |
| `C_late_natural_intact` | — | — | induce | test |
| `D_late_natural_suppressed` | — | suppress | induce | read against F |
| `E_early_natural_suppressed` | induce | suppress | — | read against F |
| `F_suppression_only` | — | suppress | — | the control that makes D and E readable |
| `G_sham` | hooks run, nothing changes | | | must reproduce A |
| `X_fixed_carrier_suppression` | — | suppress (fixed mask) | — | **diagnostic, not an image condition** |
| `B_early_weak` | induce (weak) | — | — | predeclared weak arm |
| `<B,C,D,E>__dironly` | as the base condition | | | direction-only control twin |

`LifecycleCondition.induction` is `'matched'` for B, C, D and E: natural-register-matched,
the primary. `direction_only_twin` makes the control from a condition by copying it,
changing only the operator. `B_early_weak` is `'direction_only'` because its strength is a
multiple from the direction-only sweep. `DIRECTION_ONLY_CONTROLS` in 2.1 lists which
twins run. In 5.1 each twin runs right after its matched condition.

A test asserts D, E and F declare the **same** natural-window action; if they diverge the
comparisons that interpret D and E are meaningless.

**`B_early_natural_intact` does not suppress the natural window and does not require it to
match clean.** Strength is frozen first; whatever happens to natural formation is a
measured outcome and the classifier labels it.

**`B_early_weak`** resolves its strength by a declared *rule* — the largest swept multiple
that relocated zero carriers — read off the frozen sweep before any image exists. If no
swept row satisfies the rule the arm is **skipped, not guessed**.

---

## 10. Gates

Cell 5.2 raises and stops the run on any of these (and 4.4 already hard-stops before any
run if the suppressing conditions do not share one schedule, or if a condition would
suppress and induce at the same block):

- a direction of the wrong width, or not unit length;
- `G_sham` differing from clean at any probed block;
- the operator failing to write its target at the hook. For a matched arm, every
  token-write must carry its **own** target projection **and** norm within 2%;
- conditions using different denoising steps.

Cell 5.1 also stops before any run if a block of a matched window has no target at the
reference step.

It **classifies without stopping** on: incomplete suppression (labelled, imaged, reported);
the diagnostic arm's regrowth (its result).

Cell 7.3 applies the Stage-5 stopping rules. Per the brief, **failing to become a sink does
not stop the imaging** — an induced state that reaches L1 and never reaches L3 but still
changes the picture is a dissociation to report, and 7.3 says so explicitly.

---

## 11. Declared limitations

1. **No frozen `v*` exists in this repository for FLUX.1-dev or PixArt-Σ.** It cannot be
   borrowed across checkpoints (the loader refuses on a tag mismatch) and must not be
   fitted on the evaluation prompts.
2. **On FLUX the early window is dual-stream and the late window single-stream.**
   Structural, detected and printed, not correctable by design. PixArt-Σ is uniform and is
   the cleaner test of timing.
3. **Oracle position selection.** Early-window carriers are positions read from a *later*
   clean block — valid for mechanistic diagnosis, not a deployable procedure, and stated
   wherever it appears.
4. **Model revisions are not pinned** (`ModelSpec` records `repo_id` only).
5. **One prompt, one seed, one denoising step** in the pilot.
6. **The writer hook is `BLOCK_INPUT`, not `WRITER_RESIDUAL`.** The write lands in that
   block's output and is removed at the next block's input, before any consuming operation
   reads it — `delivery_coverage` is what establishes whether that is sufficient here, and
   `delivery_verdict` is what confirms it on the run. Editing the writer's own contribution
   would stop the write itself, but that tensor is a contribution rather than a token state
   and the validated operator does not carry over to it unchanged.
7. **Suppression controls what is *delivered*, not what is written.** Within the interval
   the state is removed before any attention or feed-forward reads it, and the terminal
   cleanup closes the one boundary where that would otherwise fail. It is not removed from
   the residual stream between blocks, and the regrowth numbers say so plainly. A claim
   that the state "never existed" would be wrong; the claim the design supports is that no
   computation inside the interval consumed it.
8. **Delivery is measured as alignment, because on these checkpoints magnitude cannot
   reach a weight.** If a checkpoint's adaptive norm did pass token magnitude through, the
   gate would still be necessary but no longer sufficient — 4.1b prints a warning when the
   consumed spread exceeds 1.5×.
9. **Two quantities are frozen at one denoising step: the carrier positions and the
   induction coefficient.** Cell 4.1c measures whether either drifts across the
   trajectory (`carrier_drift` / `drift_verdict`) and `PER_STEP_INDUCTION` re-selects and
   re-calibrates per step when it does. The suppression arms are not exposed — their rule
   re-derives at every hook. Where the audit says `not measured`, nothing is inferred.
10. **This design does not test mediation.** Even with L3 and L4 both present, it does not
   establish that the attention change *causes* the image change.
11. **Natural-register-matched is not transplanted.** Two statistics are set. The
   remainder's direction, anything beyond `v*`, and the state's history are not. 4.2
   measures the first of these against an ordinary-token baseline.
12. **The matched state is usually far outside what an early block holds.** This is
   reported per block and never clipped. The direction-only twin differs from it in
   projection size and norm together, so a twin difference is not "norm alone".

---

## 12. Verification status

`pytest tests/test_lifecycle.py -q` → **103 passing**, no weights, no GPU, no cost.
`pytest tests/test_lifecycle_delivery_e2e.py -q` → **9 passing** on synthetic weights.
Full suite **605 passing**.

The delivery claim is the one that had to be checked against the running model rather than
against the diffusers source, and the end-to-end tests do exactly that on both families:
that `DeliveryProbe` records both sides of the module the QKV projection reads, that the
consumed spread collapses to ~1 while the supplied spread does not, and that a
`BLOCK_INPUT` edit is upstream of it — a planted aligned state arrives aligned, and the
norm-preserving removal leaves the carrier no more aligned than an ordinary token of the
same block.

Both pipelines and both transformers were read against the installed diffusers (0.40.0)
and against upstream `main`, which agree in every part Q16 hooks. Verified there: FLUX
concatenates `[text, image]` inside each single block so the image tokens are the tail;
FLUX runs true CFG as two separate forward passes with the conditional first, and PixArt
duplicates the batch as `[negative, conditional]` with the conditional last — so the
conditional row is the last row in both, by different mechanisms, and `pass_idx == 0`
selects the conditional pass on FLUX; PixArt's `attn1` is pure image self-attention with
no mask and no image RoPE; FLUX carries `norm_q`/`norm_k = RMSNorm` and PixArt carries
neither, which is why magnitude reaches PixArt's attention logits and not FLUX's.

Verified end to end offline on `tiny-flux1` and `tiny-pixart`. All 20 runnable cells
from 3.3 onward execute, including the new 6.7, under four configurations: FLUX, PixArt,
a forced bridge (C's bridge + late path), and the feasibility sweep with its matched rows.
In those runs:

- all validity gates pass, and every matched arm writes its per-token projection and norm
  at the hook;
- the sham reproduces clean at every probed block;
- every suppression schedule closes its terminal boundary;
- the delivery verdict reports `not measured` rather than `prevented` where the synthetic
  carriers deliver no register, which is the correct degradation;
- the continuity classifier populates.

The matched edit is also checked through the real hooks on both families
(`test_the_matched_state_is_exactly_what_each_maintained_block_receives`). What each
maintained block's computation receives carries the target projection and norm, and the
targets follow each block's own scale.

Synthetic checkpoints have random weights and no registers, so **nothing scientific follows
from those runs** — they verify the machinery only.
