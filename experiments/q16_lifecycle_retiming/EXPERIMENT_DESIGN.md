# Q16 — Is the register's *depth* causal, or only its presence?

**Stage 1–3 deliverable.** No model has been run for this experiment. Nothing below is a
result.

Architecture, hook sites, frozen-`v*` handling and the FLUX/PixArt differences are
documented in `../q15_register_address_transfer/EXPERIMENT_DESIGN.md` §1–3 and are not
repeated. This file records what is **specific to retiming**.

---

## 0. The Q1/Q4 revision: what changed, what Q1 and Q4 support, and what stays a hypothesis

The central question is whether changing the depth at which a comparable register state
exists changes its effect on the image. Two things have to be established on the run
itself before any image difference is read:

- the natural state was genuinely suppressed;
- the early or late replacement was genuinely established.

This revision rebuilds the experiment around those two measurements. It uses the Q1
register-removal and Q4 writer-selection results from
`notebooks/iclr_q1_q6_causal_mechanism.ipynb`.

### What Q1 and Q4 showed (FLUX)

**Q1** removed the frozen register population once, at one block's input, then followed
the carriers and the sinks downstream. All numbers below are read off the figure.

- **Outcome after removal.** With the register direction removed (norm-preserving),
  about 31% of prompt-seed runs have the original carriers regain the register criterion.
  About 69% end with *another carrier winning*. The label says "novelty unproven",
  because Q1 did not test that token against the clean run. Replacement by a matched
  ordinary state and zeroing give the same roughly 30/70 split. Clamping to ordinary norm
  keeps the direction, and more runs regain at the same position (about 46%).
- **Controls.** The sham and matched-magnitude removal off the register regain
  everywhere. Zeroing high-norm non-register tokens, or random tokens, still lets another
  carrier win in about a third of runs.
- **Sink identity.** Register-targeting removal leaves the clean sink in a minority of
  heads, and zeroing or matched replacement leaves it in almost none. The off-register
  direction-removal controls keep it in about three quarters of heads, the sham in all.
- **Direction versus magnitude.** Direction removal drops the targets' alignment with
  `v*` by about 0.40, with little change in norm. Zeroing, replacement and clamping drop
  alignment by about 0.17 and norm by about 2–2.6 layer medians.

**Q4** patched candidate features at the writer block, 18 on FLUX (the last dual block of
the writer range).

- **Before the write,** only the feed-forward hidden activation separates eventual
  registers from matched ordinary tokens (about 0.9). The pre-existing `v*` component
  does not.
- **Transfer.** Transferring the pre-feed-forward residual, or the feed-forward hidden
  activation, into an ordinary token makes it a sink (+0.10 and +0.08 against the sham).
  Transferring only the `v*` component of that residual does nothing (about 0).
- **Prevention.** Preventing through those two features at an eventual register removes
  sinkhood (−0.17). Preventing through the `v*` component barely does (−0.05).
- **What the site is.** The pre-feed-forward "residual" hook is `norm2`'s input. It
  changes only what the feed-forward reads, not the residual stream. A Q16 test checks
  this on the running model, on both families.

### The changes, and what supports each

| change | directly supported by | what it does not establish |
|---|---|---|
| Suppression is **dynamic over every image token**, re-evaluated at every hook | Q1: after removal another carrier wins in most runs | that relocation also happens under *maintained* suppression; Q1's removal was one-shot |
| **Original-carrier recovery, new `v*`-aligned carriers and attention relocation are three separate results** | Q1: regrowth (~30%) and relocation (~70%) both occur, and the clean sink is lost in most heads | that a remaining sink carries `v*` |
| New carriers are judged **against the paired clean run** at the same step and block | Q1 could not establish novelty | — |
| **Subtractive** removal is primary; norm-preserving is kept as an **operator control** (it cannot test sink identity: both models read every token through a LayerNorm, so it hands the next block the same input as the subtractive operator) | Q1: the two kinds of removal perturb different properties (alignment versus magnitude), so they answer different questions | that subtractive removal is *the* counterfactual, or that it removes the point artifacts (hypothesis; our dev pilot suggests fewer regrowth hooks and relocations) |
| The removal rule has **no norm bar** | Q4: before the write, an eventual register is not marked by a large `v*` component; our coverage audit found aligned-but-small tokens escaping a norm bar | — |
| Suppression is judged on the **residual state each block receives** and on **per-head attention** | our dev pilot: the consumer-side cosine could score 1 of 23 blocks | — |
| The **induction site is piloted**: block input versus the feed-forward's input | Q4: at the writer, the pre-FF site and the `v*` component are not equivalent | that an early block's feed-forward performs the writer computation (hypothesis, tested by the pilot) |
| **Matched** induction stays primary; direction-only is its control | the natural register's size, from our clean runs | that matching projection and norm reproduces the register (hypothesis; 4.2 measures the unmatched remainder) |
| Windows on the **measured** formation and dissolution; C maintained from **before** the decline and past the natural end; D **after** the suppressed interval; one shared suppression schedule | the clean dev lifecycle: forms at 18, plateau to 29, declines from 30, gone by 40 | — |

**Still Q16 hypotheses, and not treated as proven anywhere:**

- that a matched state at another depth behaves like the natural register;
- that high norm stabilises `v*` across blocks, or drives regrowth. Q1's norm-clamp
  result shows the *direction* left in place lets the originals re-qualify; it does not
  show that norm drives regrowth;
- that the subtractive operator removes the point artifacts rather than moving them;
- that transferring `v*`, or `v*` plus norm, transfers the writer computation;
- that any image difference is caused by the change in depth.

### Four results replace COMPLETE / INCOMPLETE (cell 5.3)

Each result is measured at every recorded denoising step.

- **(a) The natural `v*` state at consuming sites.** Does any block of the interval, its
  terminal cleanup, or the first unhooked block after them receive a token more
  `v*`-aligned than *every* ordinary token of the paired clean run at that block? At a
  hooked block this is measured directly on the tensor the hook handed on. At an
  unhooked block it is the previous block's output. Two readings go with it: how close
  the most aligned received token came to the natural register, and the boundary block
  reported separately.
- **(b) Regrowth versus relocation, at any position.** Do the original carriers regain
  the register criterion in what each block hands on? And which new positions become
  `v*`-aligned or register-like, new relative to the paired clean run? Unique positions
  and their persistence are counted, and the outcome uses Q1's vocabulary.
- **(c) Original-sink persistence versus attention redistribution.** For each head whose
  clean sink was an original carrier: did it keep that sink? If not, did its attention
  go to another original carrier, a relocated `v*` carrier, a non-register token, or
  spread out? Concentration is reported against clean. A non-register sink that remains
  does not contradict (a).
- **(d) The induced states.** Was the intended state written at every hook of its window
  at every step? What did the blocks hand on, and was it a sink?

Two different bars are used, deliberately:

- **Removal:** the clean ordinary 0.999 quantile of `cos(x, v*)` (Q1's alignment bar),
  applied to every token at every hook.
- **Results:** the clean ordinary maximum. The quantile leaves about 0.1% of ordinary
  tokens above it by construction, so counting against it would report the clean run's
  own ordinary tail as a `v*` state.

The consumer-side cosine that used to be the gate is kept as a secondary view (6.4),
reported with how many blocks it could score.

### The schedule on FLUX.1-dev (measured boundaries)

| window | blocks | used by |
|---|---|---|
| early | 14–16 | B, E |
| suppression | 17–39, with the terminal cleanup at 40 | D, E, F, F_norm_preserving, X: **one schedule** |
| bridge | 28–39 (two blocks before the decline at ~30) | C |
| extension | 40–42 (past the natural end at 39) | C |
| late (the replacement) | 41–43 (after the cleanup) | D |

No block carries both a suppression hook and an induction hook. The notebook hard-stops
if one would, or if the suppressing conditions did not share one schedule. C's extension
and D's replacement are separate windows. They answer different questions and are not
forced into one three-block slot.

### Before any large image run

1. Run the paired pilot on one prompt (5.1).
2. Read 5.3: the depth plots against clean, the four results, and the per-step table.
3. Treat only the comparisons 5.3 marks *interpretable* as depth results in 7.3: E vs F
   (early-only), D vs F (late-only), and C vs A (natural + late). Each is interpretable
   only if (a) held at every step on both sides that suppress, and (d) was established at
   every step.
4. Only then run the larger image study, with the same schedules.

---

## 1. The question, and the four outcomes it must not collapse

The register forms in a bounded depth interval, persists, and dissolves. Does that
interval matter?

| level | what it means | what it does **not** license |
|---|---|---|
| **L1** | the target tokens acquire a substantial `v*` projection at the target depth | calling it a register |
| **L2** | that state is also high-norm and lasts the whole window | calling it a sink |
| **L3** | it acquires favourable query–key geometry and draws attention | calling it functional |
| **L4** | downstream computation and the image change | calling the lifecycle retimed |

**L4 without L3 is a perturbation with a visible effect.** L1 without L2 is a direction
written into a token the block discards. Each level is measured and reported separately;
the verdict cell refuses to summarise them into one.

### Three kinds of persistence, which are different findings

- **maintained** — the state exists only because the hook reinserts it at every site;
- **carried** — it survives into later blocks that were never patched;
- **reconstructed** — it is *stronger* after the window than inside it, so the network
  added to it rather than merely failing to erase it.

`lifecycle.persistence_class` labels which occurred, **from the measured trajectory**, and
returns `"not measured"` when no unpatched block follows the window — the honest answer
for a late window at the end of the stack.

**It must be given the clean baseline, and for an early window that is not optional.** The
blocks after an early window are exactly where the natural register forms, so an absolute
reading there measures the model doing its ordinary job and reports it as the induced
state persisting: every early condition comes back `carried` or `reconstructed` whether or
not anything survived. With `baseline=` the quantity becomes the **excess over clean**, and
a state that vanished when the hooks stopped is correctly labelled `maintained`. Pinned by
a test.

---

## 2. Windows: predeclared, equal length, boundary-aware

`lifecycle.windows_from_ranges(ranges, length, n_layers)` derives them from each
checkpoint's **frozen** `LayerRanges`, before any intervention runs:

- **natural** = the artifact's own register range;
- **early** = the `length` blocks ending just before the writer range begins;
- **late** = the `length` blocks starting one block **after** the natural range. It
  never overlaps the natural lifetime: `late_first` may move it further out, and a value
  inside the natural range is refused by the library and by a hard stop in 3.3.

Early and late are **equal length by construction**. This is not cosmetic: an early window
with more intervention sites than the late one confounds depth with dose, and every
early-vs-late comparison would be uninterpretable.

A window that cannot exist is **omitted, not shifted** — a writer range starting at block 0
leaves no room before it, and a late window past the end of the stack is clipped or
dropped. A register range containing no block is refused outright. Tested.

### Late never overlaps natural; extension gets a bridge

**The late window is where the state exists in the late conditions**, so it must lie
strictly after the natural lifetime. A block it shared with the natural range would be one
where "late" and "natural" cannot be told apart, and no late-condition effect could be
attributed to lateness. An earlier build got this wrong. Asked to start *maintenance* at the
measured dissolution onset so that extension could not lapse, it moved the whole late window
there instead — to 35–37, inside the natural range 20–39. The library now refuses a
`late_first` inside the natural range, and 3.3 hard-stops if either window touches the
natural lifetime, writer stage included.

**Extension is a different thing, and gets its own window.** NATURAL + LATE (C) asks
whether the lifetime can be extended, and extension has to be continuous. The natural state
starts to dissolve *before* the natural range ends, so an induction that begins only at the
late window finds nothing to extend and creates a second state after a gap.
`lifecycle.extension_bridge` returns the blocks from the measured dissolution onset to the
block before the late window. **Only C uses it.** D suppresses the natural window, so it has
nothing to bridge, and its state exists only in the late window. The late window is
identical in C and D.

The onset is `measured_dissolution_onset` on the clean trajectory, at
`DISSOLUTION_FRACTION = 0.9`: the first block where the carriers' `v*` projection falls
below 90% of its plateau. On the dev pilot the rule-selected carrier count holds at 23–27
through block 29 and then falls from block 30 (21 → 14 → 9 → … → 1 at 39). A fraction of 0.5
would start the bridge around 35, by which point 15 of 24 carriers have already dissolved,
so "maintaining" them would be re-inducing them. 0.9 was declared before any bridged image
existed. The cost is stated rather than hidden: C is induced over bridge + late blocks,
against three for B. That is what extension means, not a dose mismatch to correct. The
dose-matched early-versus-late comparison is **D against E**.

Placing the bridge does not make the result continuous, and the design does not assume it
does. `lifecycle.continuity_check` measures presence at **every** block from the natural
plateau through the end of the late window, against one absolute bar set from the *clean
plateau*. A bar that followed the clean run's own decline would call dissolution itself
continuous and could never separate the two outcomes; a test pins that the clean run is
classified `late_re_induction` under this bar. One block below the bar between the plateau
and the late window means the state disappeared and a different one was created. The run is
then reported as **late re-induction** — a legitimate finding under its own name, never as a
longer lifetime.

### The architectural confound, reported rather than absorbed

On FLUX.1, blocks 0–18 are dual-stream and 19–56 single-stream. With the prior schnell
ranges (writer 17–19, register 20–39) the early window lands at **14–16 (dual)** and the
late window at **40–42 (single)**. These are *different attention architectures*.

`Window.crosses(boundary)` flags a window that straddles the seam, and every window's row
in `records/windows.csv` carries `crosses_architecture_boundary`. **Any early-vs-late
difference on FLUX is confounded with dual-vs-single stream and must be reported that
way.** PixArt-Σ is uniform single-stream and does not have this confound, which makes it
the cleaner test of timing — a reason to weight it, not to drop FLUX.

Windows are frozen before the run and are never reselected after seeing an image.

---

## 3. The operators

### Induction, primary — natural-register-matched (`lifecycle.induce_matched`)

Injecting `v*` at ordinary token norm does not reproduce the natural register. The natural
register sits at over ten times the median norm, and that size may be part of what keeps
`v*` stable from block to block, even if direction alone is enough for a block to select
the token as a sink. The primary arms therefore set **both** statistics the natural
register has, its `v*` projection and its norm, at every maintained block:

```
x' = π_t · m_L · v̂  +  sqrt(ρ_t² − π_t²) · m_L · r̂
```

- `ρ_t = ‖x_t‖ / median` and `π_t = α_t / median` are carrier `t`'s own ratios on the
  **clean** run. They are averaged over the natural **plateau**: from the natural window's
  first block to the block before the measured dissolution onset (20–29 on the dev pilot).
  They are read at those blocks' **inputs**, at the **same denoising step** as the edit.
  A token that is not a natural carrier gets the population mean.
- `m_L` is the median norm of recipient block `L`'s **input** at that step, which is the
  tensor the edit writes into. The sizes are relative because block scales differ by an
  order of magnitude across the stack, so a copied absolute number would mean something
  different at every depth.
- `r̂` is the recipient token's **own** remainder direction. It is rescaled, never
  rotated.

`calibrate_register_match` builds the targets per `(step, block)` into a `RegisterTargets`.
The edit (`lifecycle_edit('induce_matched', …)`) fires at every block of the window and at
every edited step, so the state is **maintained** there. Because the reference is the same
step, the induced state follows the natural register through denoising time and only its
depth moves. With frozen positions, a step the clean trace cannot calibrate falls back to
the reference step's size, which is registered explicitly with `step=None`. With per-step
positions, such a step is left alone.

**Verified, not assumed.** 5.2 gates on what was written **at the hook**: every
token-write must carry its own target projection and norm, each within 2%. 6.7 then reads
the window and six unhooked blocks after it, block by block. At each block it records the
carrier's `cos(x, v*)`, its norm over the block's median, and its sink strength (the
attention computed inside the block, on the maintained input). It also records the excess
over clean. All of these are compared with the clean natural register on its plateau
(`maintenance_summary`, `maintenance_profile`).

**What is matched, and what is not.** Two scalars are matched. Three things are not:

- the direction of the rest of the token, since the carrier keeps its own content;
- anything else the natural register carries beyond `v*`;
- its history: how it got there, and what earlier blocks wrote alongside it.

4.2 measures the first of these against an ordinary-token baseline. The baseline is
needed because the residual stream has a direction every token shares, and a raw cosine
would credit that to the register. 4.2 asks two questions:

- Do the natural carriers' remainders share a direction more than ordinary tokens' do?
- Is a carrier's remainder at the recipient further from its own natural-register
  remainder than an ordinary token's remainder is from itself across the same two depths?

**A matched state that does not behave like a register is therefore not evidence that
depth matters on its own. One that does behave like a register shows that those
statistics are enough for that behaviour at that depth. It does not show that the
mechanism moved.**

**Is it unusually large for the recipient?** Usually yes at an early block, which never
holds a register. 4.2 reports these per block:

- the target against the largest norm and the largest `v*` projection any clean token
  holds there (`norm_unusually_large`, `projection_unusually_large`);
- the target's percentile among the block's clean tokens;
- the size of the edit on the clean tensor, in units of the block's median;
- `projection_only_norm_ratio`: the norm the tokens would reach from the projection
  alone.

The target is **not clipped**. PixArt's image attention has no query/key norm, so on
PixArt a state this large can become a sink by size alone. On FLUX the keys are
RMS-normalised, so only the direction reaches the logits.

**The control.** The earlier direction-only injection (next section) runs as a twin
(`<key>__dironly`) of B, C, D and E. The twin has the same windows, suppression and steps;
only the operator differs. Its projection is smaller (the recipient-outlier anchor times
the strength) and its norm stays near ordinary. **A twin difference is therefore what the
register-sized state adds over that lower-strength injection: projection size and norm
together, not norm alone.** On a checkpoint whose natural register lies almost entirely
along `v*` (a high natural `cos`), "high norm" and "large projection" are nearly the same
thing. 4.2 says whether that is the case, using `projection_only_norm_ratio`.

### Induction, control — direction-only (`lifecycle.induce_alpha`)

The control arm for the matched induction above. The weak early arm and the feasibility
sweep's strength multiples also use it.

```
x' = r + α_target · v̂          implemented as   x += (α_target − α) · v̂
```

**Deliberately not norm-preserving.** The natural register *is* a high-norm state; an
induction that held the recipient at its original length would be testing something else.
The achieved norm is recorded, so the perturbation is compared against the recipient
layer's own distribution rather than assumed reasonable.

Written as an addition because subtracting a large `α·v̂` and adding another back loses
precision in `r` exactly where the claim "only the `v*` component moved" is made. Adding a
multiple of `v̂` leaves the orthogonal complement invariant **by construction**.

### Suppression — subtractive primary; the validated norm-preserving operator as a control

```
primary (subtractive):              x' = x − (x·v̂) v̂  =  r
operator control (preserving):      x' = ‖x‖ · r/‖r‖
```

The primary removes the `v*` component and leaves the token at the size that remains; it
is **not** rescaled back to register norm (§0). The norm-preserving operator keeps a
register-sized token pointing along its ordinary content, and runs as
`F_norm_preserving`, an operator control. It cannot answer whether a large token without
`v*` stays a sink: every block reads its input through a LayerNorm (FLUX and PixArt
alike), so it hands the next block the same input as the subtractive operator and differs
only in how far later updates can turn the stripped token. For that control: **A test asserts this is bit-identical to
`control_surface.control_surface_states(β=0, γ=1)`**, and a second asserts that operator's
`identity_error` at `(1,1)` is below 1e-5. D, E and F only compare if all three suppress
identically; "we reimplemented it the same way" is not a guarantee, so it is checked.

Explicitly **not** used, and not silently swapped in: zeroing whole token vectors,
suppressing channel 154/293 globally, or masking incoming attention to sinks. Those are
different interventions.

### Two suppression experiments, and why they are not interchangeable

**1. Fixed-carrier suppression (`X_fixed_carrier_suppression`) — a diagnostic, not an
image condition.** The frozen clean positions, every hook from the writer onward. Because
it never targets anything else, relocation and reconstruction elsewhere stay **visible** —
a state-based rule would chase them and hide exactly the behaviour this arm measures. Its
image is not comparable with the state-based conditions: it edits a different number of
positions for a different number of hooks, so any image difference would mix the absence
of the state with the cost of suppressing its regeneration.

**2. State-based suppression — what EARLY ONLY, LATE ONLY and SUPPRESSION ONLY all use.**
`RegisterStateRule` re-evaluated at every hook on that hook's own tensor. Same operator,
same rule, same schedule in all three, so their image differences remain interpretable.

### The rule, and the audit that has to precede it

The criterion is the conjunction the frozen selection already uses: **high norm AND strong
`v*` projection**, both relative to that block's own distribution.

**The conjunction is load-bearing, and a projection percentile alone is not usable.** A
percentile *always* selects its fraction — at p99 on 4096 tokens that is 41 tokens at
every block whether or not a register exists there. Ordinary tokens carry weaker `v*`
components, so a percentile-only rule erases the direction from the image stream rather
than preventing the sparse state. Requiring high norm as well makes the rule select ~0
where nothing register-like is present, which a test pins by stripping the register state
from a synthetic scene and checking the rule goes silent.

`rule_selectivity` audits the rule on the **clean** run, per block, before it is frozen:
selected count, share of the image, and how much falls outside the known carriers. The
notebook **refuses** a rule that would touch more than a declared ceiling (default 2% of
the image) at any block, and writes the frozen rule to `records/suppression_rule.json`
with `frozen_before_any_image: true`. It runs off the per-block norm and projection the
tracer already records — not the full residual slice, which would be ~900 MB over an
18-block interval on real FLUX.

### The suppression interval covers the writer, and the default now enforces it

An earlier build documented suppression from the writer onset but **defaulted to the
artifact's register range**, so the executed schedule was blocks **20–39** while the
measured writer operates at **17–19**. The state was written in full and only then
removed — "let it form, then delete it", a different experiment from preventing its
emergence. The guard meant to catch this compared `SUPPRESS.first` against the *register*
range, so it could never fire.

`SUPPRESSION_FIRST = None` now resolves to `artifact.layer_ranges.writer[0]`, and a
**hard stop** fires if suppression starts after the writer onset.

**Which hook at the writer stage.** Suppression runs at `BLOCK_INPUT` throughout. The
write at block 18 therefore lands in block 18's *output* and is removed at block 19's
input, before any later block reads it — so it never propagates, which is what "prevent
its emergence and persistence" has to mean operationally. Editing the writer's own
contribution (`WRITER_RESIDUAL`) would stop the write itself, but that tensor is an
attention/feed-forward contribution rather than a token state, so the validated
norm-preserving `v*` removal does not carry over to it unchanged and would need its own
validation. Recorded as a deliberate choice, with the per-block achieved table as the
evidence of whether it worked.

### Suppression effectiveness is read from the intervened trajectory

Cell 4.4's selectivity table is the **clean-run** audit — what the rule *would* select
before anything is done. Cell 6.4 now also reports, per condition and per block across the
whole suppressed interval and **from the intervened runs**: tokens selected at the hook,
newly targeted, `v*` projection entering the block and after the hook, carrier projection
and norm at the block output, median norm, high-norm counts split frozen vs new, and
carrier sink strength. Written to `records/suppression_achieved.csv`.

### Cumulative counts cannot answer the relocation question

`n_highnorm_new` is a **per-block count**, so the sweep's `new_carriers_after` is a sum of
token-**layer** observations. 395 can be ten positions crossing the bar at forty blocks or
three hundred and ninety-five positions crossing once — opposite answers to "did early
induction relocate register formation?".

`carrier_census` reports **unique positions** alongside the observation count, plus the
median number of blocks each position was high-norm. A small unique count with high
per-position persistence is a **stable relocated population**; a large unique count with a
median near 1 is **threshold flicker**, and the cumulative number reflects the bar rather
than a moved register. A test constructs both with the same cumulative total of 40 and
asserts they are distinguished.

### Regrowth is a measurement; delivery is the gate

> **Superseded by §0.** The consumer-side cosine gate described here is now a secondary
> view (6.4). It could score only a handful of blocks, because of the adaptive norm's
> shared shift. Suppression is reported as the four separate results of §0 (cell 5.3),
> measured on the residual state each block receives and on per-head attention. The
> reasoning below about why regrowth is not failure still holds.

Two questions, at two different tensors, and only one of them can invalidate a condition.

**Regrowth.** The suppressor acts on each block's **input**, so what it sees at hook *k+1*
is what block *k* produced. `n_selected` after the first hook therefore *is* the regrowth
measure, with no extra threshold to choose, and `newly_targeted` separates "the same tokens
keep coming back" from "the state is moving". It is a measurement of **how hard the model
rebuilds the state** — a finding about the model — and it gates nothing. An earlier version
of `suppression_verdict` failed a condition on it, which was wrong twice over: it would
call a perfectly guarded interval incomplete merely because the network kept trying, and it
invited escalating the intervention until the number looked right, towards exactly the
indiscriminate erasure this design forbids.

**Delivery.** Whether any attention or feed-forward inside the interval actually *read* a
register-like state. Between block *k*'s output and block *k+1*'s consumers stands block
*k+1*'s hook, so regrowth at a hook is entirely consistent with nothing ever being
consumed. This is the gate.

#### What the consuming operation receives, and where it has to be measured

The consumer is reached through an adaptive norm, and **the two families put the modulation
in different places**:

```
FLUX single    norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
                 -> self.attn(hidden_states=norm_hidden_states)    # the QKV projection
                 -> self.proj_mlp(norm_hidden_states)              # the feed-forward
FLUX dual      norm_hidden_states, ... = self.norm1(hidden_states, emb=temb)
                 -> self.attn(hidden_states=norm_hidden_states)
PixArt         norm_hidden_states = self.norm1(hidden_states)            # PLAIN LayerNorm
               norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
                 -> self.attn1(norm_hidden_states, ...)
```

`AdaLayerNormZero`/`AdaLayerNormZeroSingle` apply their scale and shift **inside** the
module, so on FLUX the norm module's output is exactly what the QKV projection receives.
`BasicTransformerBlock` with `norm_type="ada_norm_single"` applies them **in the block
body**, so on PixArt it is not — and a per-channel scale is not a rotation, so it changes
each token's alignment with a direction by a different amount and can reorder which token
is most aligned.

Delivery is therefore probed at `InterventionPoint.ATTENTION_INPUT`: **the tensor the
attention module is called with**, after all modulation and after any block-level
positional embedding. That is the same quantity on both families and it is the only site at
which "what the computation received" is a measurement rather than an approximation.
`delivery_coverage` needs the residual as well, so `DeliveryProbe` records both — the block
input (`supplied_*`, the tensor the rule is evaluated on) and the attention input
(`consumed_*`).

*This was got wrong first.* The probe originally read the norm **module's output**
(`PRE_KEY_NORM_RESIDUAL`). That is bit-identical to the attention input on FLUX and is
**not** on PixArt, where it reports a pre-modulation tensor: measured on the synthetic
checkpoints, the delivered cosines differ by up to **0.16** and the norm spread reads 1.00×
against a true 1.07–1.24×. On a PixArt main study that would have mis-scored the gate
silently. Two end-to-end tests now pin the asymmetry — the two tensors must be identical on
FLUX and must differ on PixArt — so a diffusers change in either direction fails a test
rather than changing a result.

What survives the norm is then a measurement, not an assertion. LayerNorm divides token
magnitude out; the modulation that follows can put some back.
`consumed_norm_spread` is reported for exactly that reason. Where it is ~1, the register's
magnitude never reaches a weight — the same fact as FLUX's `norm_k = RMSNorm` making the
key norm constant, one stage earlier — and **what the computation receives is a
direction**, so delivery is an alignment question. Where it is not, alignment stays
necessary and stops being sufficient, and 4.1b says so above 1.5× instead of the analysis
assuming otherwise.

#### The gate

For each block, the excess alignment delivered over the **clean run's ordinary ceiling at
that same block**, as a share of what the clean run's register delivers there:

```
s_k = (max_t cos(x̃_t, v*) − c_ord_k) / (cos_carrier_k − c_ord_k)
```

`s_k ≤ 0` means nothing arrived more aligned than an ordinary token; `s_k = 1` means the
consumer received exactly what it receives with the natural register in place. Scoring
against the ordinary ceiling rather than against zero is necessary and not conservatism:
`v*` is a real axis of the residual stream and every token has some component along it, and
the adaptive norm's shift is a token-independent vector that offsets every token's cosine
identically — it cancels in a comparison against the ordinary population and does not
cancel against zero. `DELIVERY_TOLERANCE` (0.25) is declared before the run.

Where the clean carrier delivers no more alignment than the clean ordinary ceiling there is
no scale to score on, and the verdict is **`not measured`**, never `prevented`: claiming
success from an absent measurement is worse than reporting nothing. Tested.

#### The terminal boundary

A hook at `BLOCK_INPUT` of block *k* protects block *k*'s own attention and feed-forward.
The last hook is at the interval's last block, so **that block's output reaches the next
block's QKV projection with nothing in between** — on the schnell configuration, block 39's
output entering block 40. `lifecycle.suppression_schedule` therefore extends every
suppression interval by **one terminal cleanup hook** past its end. It is part of the
schedule rather than an analysis step, because there is no way to measure the leak away
afterwards. `delivery_verdict` scores the boundary block separately and reports it whether
or not it passes: an interval that leaks only at its own boundary is a different finding
from one that leaks throughout, and a boundary block that was not probed is reported as
`UNMEASURED` rather than closed.

Where a suppression interval is immediately followed by an induction window — LATE ONLY,
whose suppression now ends where the late window begins — the interval is truncated to stop
before it and the terminal cleanup lands **on the induction's first block**. Plan order is
execution order for two `BLOCK_INPUT` pre-hooks on one block, and suppression is built
first, so the stream is cleaned and the induction then writes the intended state on top of
it. That is precisely what "the state exists only from here on" has to mean.

#### What the verdict says

`suppression_verdict` returns **complete** only when delivery is `prevented`, the terminal
boundary is closed, and the suppressed carriers never regained sink behaviour. Three
deliberate choices:

- **Regrowth does not gate**, as above. It is reported as a rate, a first block, and a mean
  number of tokens rebuilt per block.
- **The raw `v*` projection is reported and does not gate.** A test pins that a 60%
  projection residue with delivery prevented is still `complete`.
- **The criterion coming back clean while a sink returns is a genuine failure** — attention
  is being routed by something `v*` does not describe — and is classified `incomplete`.
- **No delivery measurement means `not measured`**, not `complete`.

The notebook also reports the gap between the two vantage points: the criterion at block
**inputs** versus the carrier projection at block **outputs**. A projection that returns
while the criterion stays clean means the block rebuilt the *direction* without rebuilding
the sparse high-norm *state*, which is the distinction this whole design rests on.

### What the first full-dose dev pilot showed, and the confound in it

Three prompts, 28/28 steps edited, sham exactly zero on all three. Within every prompt
**D, E and F are nearly identical** (mean |diff| 10.4–14.1) while **B and C sit near
clean** (0.2–1.7). Composition is untouched everywhere. The suppression conditions add
**point artifacts on low-information background** — sparkles on a flat wall, bokeh on a
studio backdrop, dark specks on a dark wall — plus texture and tone changes on objects.
X, which suppresses only the frozen carriers, produces far fewer points.

So the image effect is entirely suppression-driven, and the induced states add nothing:
neither a new state at another depth (B, C) nor a replacement for the removed one (D, E
versus F). The paper cannot say that yet, for three reasons.

**The operator may be manufacturing the artifacts.** Norm-preserving removal sets
`x' = ||x|| r/||r||`. A register token's norm is over ten times the median and nearly all
of it lies along `v*`, so this scales the remainder `r` *up* to register size. The token
keeps a register-sized norm, now pointing along its ordinary content, and later blocks —
which write ordinary-sized contributions — can barely move it. The natural dissolution
machinery removes `v*` and so has nothing to remove. That is not "a register that never
formed"; it is a different anomalous token. Three observations fit it: the artifacts sit
where registers live (low-information background), they look like isolated point
features, and their number tracks how many distinct positions were suppressed — few for
X's frozen set, many for the state rule, which re-selects at every block and every step.
`SUPPRESSION_OPERATOR` now offers `subtractive` (`x' = r`, the token without its `v*`
component at ordinary size), `RUN_OPERATOR_CONTROL` adds suppression-only with the other
operator in every run, and 7.2b measures whether the most-changed image tokens are the
suppressed ones (`artifact_colocation`). If the stars vanish under subtractive removal
and sit on suppressed tokens, they are the operator's; if they persist and sit
elsewhere, they are the register's absence acting through attention.

**The early induction was never register-sized.** `recipient_outlier` × 1 anchored
alpha to the loudest clean token at block 14: about 1,000 against a natural register of
about 26,000–30,000. Because the consumer reads a LayerNorm'd direction, what matters is
alignment, and alpha ≈ 1,000 on a token of ordinary norm gives cos(x, v\*) of roughly 0.6
against the natural register's ~0.99. B and E induced a half-aligned token, not a
register. Separately, `cosine_matched` was broken: it divided alpha by the *median norm*
(a ratio, 13× on FLUX) and clamped the result to 0.999, so it always returned the same
target. It now uses the carriers' actual cos(x, v\*).

**Three blocks is a pulse, not a lifetime.** The natural state lives about 20 blocks.
Early and late windows of three blocks test "a brief state at another depth", not "the
lifetime moved". See §8 for the translation design that tests the latter.

### Recording every denoising step

`MEASURE_STEPS='all'` records the tracer at every step — separate from
`INTERVENTION_STEPS`, which decides where edits fire. What makes it affordable: the
full-tensor state probe and the edit installers' manipulation checks stay at the
reference step only (`StateProbe(steps=...)`, `CausalTracer.diagnostic_steps`); each run is
reduced to a depth-by-time table (`lifecycle_over_time`) and saved whole in float16 to
Drive (`save_trace_compact`, ~0.7 GB per run at 1024px); then the per-step attention
tensors are freed (`thin_trace`), keeping the reference step intact for every existing
table. Cell 6.6 draws the lifecycle as block × step heatmaps and uses `structure_extent`
to ask whether the dominant channel, the high-norm tokens and the sinks shift their
onset and offset with `v*` — the coupling claim behind the whole question.

### Two gates that read the wrong tensor (first dev pilot)

The first FLUX.1-dev run with the full temporal dose hard-stopped 5.2 on the late
induction arms and marked every suppression condition "delivered, peak 1.97 of the
natural register". Neither verdict was about what it claimed to measure.

**The induction gate read the block's response, not the operator's write.** It compared
the carrier projection at the block **output** with the target. The operator writes at
the block **input** — `EditRecord.alpha_after`, recorded at the hook — so the gate was
really measuring how much of the induced state survives one block. On the early window
that happens to be ~90%, so the check passed by accident. On the late window, which now
starts at the measured dissolution onset, each block erases 60–75% of what it is handed
(30,357 written, 11,789 handed on in C, 7,630 in D). That is the dissolution machinery
working. It is a finding, and a good one, not a failed intervention. The gate now reads
`alpha_after` at the hook, within 2%. The survival fraction is reported per window to
`induction_retention.csv` and never gates. Maintenance at the block input still
guarantees what every block *inside* the window **receives**. What leaves the window is
6.5's question.

**The delivery gate was dividing by a register that was not there yet.** Its denominator
is the clean register's delivered cosine minus the clean ordinary ceiling at that block.
The suppression interval opens at the writer blocks (17–18). There the clean model has not
yet written the register into the block's input, so that gap is a few hundredths of a
cosine. The maximum over 4,096 tokens moves by that much between any two trajectories,
so ordinary jitter divided by a near-zero gap reads as "twice the natural register". The
tell was that D, E and F all reported almost exactly the same peak (1.96–1.98). A real
delivery would not be that uniform across three schedules. Two fixes:

- `DELIVERY_MIN_HEADROOM` (0.10). A block is scored only where the clean register clearly
  stands out at its consumer; blocks below the floor are reported as
  `blocks_without_register` and not scored. In the clean run there is nothing to deliver
  there either.
- `delivered_rows(rule=...)` widens "register" at each block from the frozen carriers to
  **every token the rule selects at that block's input**. A frozen set smaller than the
  real population otherwise leaves genuine register tokens in the "ordinary" pool, where
  they set the ceiling and shrink the gap from the other side.

Both are pinned by tests, including one that reproduces the false "delivered" without the
floor.

### Does the conjunction miss a precursor at the writer?

The rule is evaluated on the **residual**, where the register is both large and aligned. The
consumer reads the **normalised** tensor, where magnitude has been divided out. That gap is
exactly where a precursor could hide: a token being written into at the writer stage can be
strongly aligned, still below the high-norm bar, and arrive at the QKV projection pointing
very nearly where a finished register points.

`lifecycle.delivery_coverage` measures it directly on the **clean** trajectory, before any
suppression, because the question is about the rule rather than about the intervention. At
every block it applies the rule to what the block was supplied, then counts the
**unselected** tokens that delivered an alignment at or above the weakest alignment the rule
did catch. `bar_to_cover` is the high-norm ratio that would have caught them;
`coverage_recommendation` reports the minimum over the interval and separates *how many
blocks* under-cover from *how many token-blocks* did, because a handful of misses at one
block barely above the floor is not the same finding as a systematic precursor.

Nothing is broadened automatically. `BROADEN_RULE_TO_COVERAGE` defaults to **False**: the
under-coverage is reported as a limitation and the delivery measurement is what says whether
it mattered. Set True, the bar drops to the measured level and only then — and the notebook
**hard-stops** if that bar would touch more than `RULE_MAX_SHARE_OF_IMAGE` of the image,
because covering a precursor by erasing the direction from the image stream is not
prevention. The recommendation's own note carries that condition, and a test asserts the
note carries it.

A run classified `incomplete` is **not** a valid EARLY ONLY or LATE ONLY result. Its image
is **still produced and labelled** as an incomplete-suppression result — the brief allows
"explicitly report that complete suppression was not achieved", and aborting the run would
make that report impossible. The classification is carried into the image table as a
`suppression` column beside `achieved`. The diagnostic arm is exempt entirely, because
regrowth there is its result rather than a failure.

**Two corrections made after the first real suppression run:**

*The sink check was firing on every condition, including clean.* It counted any sink
outside the frozen carriers as evidence that suppression failed — but a clean FLUX block
carries 130–200 sinks, so this marked all 20 blocks of every condition as failed. The
check now asks whether the **suppressed carriers regained sink behaviour**, which is the
question. Sink *relocation* beyond the clean level at that block is reported separately
and does not gate: the model rearranging its ordinary sinks is a different phenomenon
from the sparse register state coming back.

*Incomplete suppression aborted the run.* It now classifies and continues.

### Maintenance schedule

Both operators are installed as an `EditPlan` over **every block in the window**, at the
declared denoising step(s) — not one-shot. The network can rewrite `v*` after a single
edit, so a one-shot suppression that the model repairs would be scored as a lifecycle
shift when it is a failed suppression. §9's per-block verification is what catches this.

The intervention mask is **frozen**: the same clean-run token ids at every site. Newly
emerging carriers are recorded separately and are never added to the mask.

### Calibration — `lifecycle.calibrate_alpha`

The high-norm bar is taken from the **recipient layer's own median**, not the natural
window's. "Norm > ratio × the median of that layer" is how high-norm is defined everywhere
else in this project, and block 14's median is well below block 20's — importing the
natural window's bar silently made L2 harder there by that factor. The natural-window bar
is reported alongside, because "as high-norm as the natural register" is a different and
also meaningful question.

`PILOT_STRENGTH` accepts one value **or one per window**. When the two depths have very
different L3 thresholds, a single number either fails to induce at one depth or
over-drives the other; matching each window at its own threshold is the fairer comparison,
and both perturbation sizes are recorded so the reader can see they are not identical.

### The direction-only control's anchor is the recipient layer's clean residual scale — not QK cosine

This sets the **control's** size. The primary arms take their size from the natural
register itself (§3). Matching projection and norm reproduces the natural register's
residual `cos(x, v*)` as a consequence. Like the control, the primary arm is never
calibrated on attention, so query–key preference stays an outcome.

An earlier draft proposed matching `cos(x, v*)` across depths, on the grounds that FLUX's
QK-norm makes it the only quantity reaching the attention logit. **That was the wrong
default and is now demoted to a secondary control.** The question is whether the *same
kind* of `v*`-aligned residual state has different consequences at different depths, and
**depth-dependent query–key preference is one of the outcomes being measured**. Anchoring
strength on it would calibrate away the mechanism under study.

So: strength is set from the **recipient layer's own clean residual scale on the `v*`
axis** (`recipient_outlier`), predeclared, and the following are *reported* rather than
matched — achieved `v*` projection, norm, perturbation L2, QK cosine, attention mass, sink
strength, and the strength multiple used. `cosine_matched` remains available, labelled
`SECONDARY CONTROL` in its own note, for the mechanistic question "if the key direction is
held equal, does the attention still differ". The same principle applies to PixArt, which
has **no QK-norm** at all, so magnitude reaches its logits and its secondary control needs
norm as well as cosine.

### The primary comparisons are grouped by the ACHIEVED lifetime

The configured hooks say what was attempted. `classify_achieved_lifetime` says what
happened, from the measured trajectory as **excess over clean** at the carrier positions:

| outcome | meaning |
|---|---|
| `continuous_extension` | the excess is still present where the induction and natural windows meet — availability really was extended |
| `pulse_then_natural` | the excess collapses before natural formation, which then proceeds as in clean — **an early pulse, not a retimed lifecycle** |
| `displacement` | the frozen carriers lose their high-norm presence and/or the population relocates — natural formation did not run |
| `no_induction` | no excess in the induction window at all |

Conditions whose achieved lifetime differs **are not the same experiment** and are never
averaged together, however similar their configured schedules look. The image table is
printed with an `achieved` column beside every row.

### EARLY + NATURAL does not require an intact natural window

The natural window is **not** suppressed in this condition and is **not** required to
match clean numerically. A defensible strength is frozen first; whatever happens to
natural formation is a measured outcome, labelled by the classifier. If early induction
displaces the natural state, the run is reported as `displacement` rather than as an
intact or continuously extended lifecycle.

A separately predeclared **weak early arm** (`B_early_weak`) asks whether early induction
changes the image at a dose that leaves natural formation approximately as clean leaves
it. Its strength is resolved by a declared *rule* — the largest swept multiple that
relocated **zero** carriers — read off the frozen sweep before any image exists, never
retuned afterwards. If no swept row satisfies the rule the arm is **skipped**, not guessed.

### Early window: 14–16 for the pilot

Minimal, immediately before onset. `9–16` is a separate longer-duration condition for
after the pilot, not a dose-matching device. The achieved state is tracked through the
writer and the natural window, and if it disappears before natural formation the run is
reported as an early pulse.

### The achieved-state criteria, in one place

Every one of these is declared before the pilot and measured afterwards; none is a
restatement of the schedule.

| what is claimed | how it is decided | where |
|---|---|---|
| **L1** a `v*`-aligned state exists at the target depth | the carrier projection at the block output exceeds the clean run's p99 **at that same depth** | 6.2 |
| **L2** it is high-norm and survives the window | carrier norm > `HIGHNORM_RATIO` × the **recipient layer's own** median | 6.2 |
| **L3** it draws attention | the carrier population's **mean** sink strength exceeds `SINK_THRESHOLD`; `carrier_any_sink` reported separately | 6.2 |
| **L4** the image changes | LPIPS/RMSE against the same clean reference | 7.1 |
| **suppression achieved** | delivery `prevented` **and** terminal boundary closed **and** carriers never regained sink behaviour | 5.2 / 6.4 |
| **state delivered to a consumer** | `s_k > DELIVERY_TOLERANCE` at the normalised tensor the QKV projection reads | 6.4 |
| **regrowth** | tokens the rule re-selects after the first hook — reported, gates nothing | 6.4 |
| **rule coverage** | unselected tokens delivering at or above the weakest selected one, on the clean run | 4.4 |
| **persistence** | maintained / carried / reconstructed, as **excess over clean** | 6.2 |
| **achieved lifetime** | continuous_extension / pulse_then_natural / displacement / no_induction | 6.2 |
| **continuity** | present at every block from the natural plateau to the late window's end, against one absolute bar from the clean plateau | 6.5 |

### A failure to become a sink does not stop the imaging

Perturbing the `v*`-aligned residual state and suppressing sink attention are known to
have different image consequences. An induced state that reaches L1 and never becomes a
sink but still changes the picture is a **dissociation worth reporting**, and 7.3 says so
explicitly rather than treating it as a failed experiment.

| mode | target | when it is the right question |
|---|---|---|
| **`recipient_outlier` (default)** | the largest absolute projection the recipient layer already holds in the clean run | the feasibility sweep — a multiple then means "n times what this layer already has" |
| `ratio` | natural `α / median‖x‖` carried onto the recipient layer's median norm | is the natural *ratio* reachable here? |
| `absolute` | the natural `α`, unscaled | is that particular *magnitude* reachable here? |

**Why the default changed, measured on FLUX.1-schnell.** The natural register's `α` is
**26,188** against a natural-window median norm of **1,968** — a ratio of **13.3×**.
`ratio` mode reproduces that outlier wherever it is applied, which at block 14 (max clean
projection **1,163**) meant asking for `α = 17,761`, i.e. **15× the loudest projection the
layer holds**. A first sweep at multiples 0.5–4.0 then spanned `α` from 8,880 to 324,318,
and **every row passed L1, L2 and L3 at every strength** — not feasibility but saturation:
a key is linear in the residual and the attention logit scales with `‖k‖`, so a token far
louder than its neighbours must dominate attention. `recipient_outlier` anchors on what
the layer already holds, so the multiples are interpretable and a threshold can appear.

*(An earlier draft of this section said the natural register sits at ~100× the median
token norm. The measured figure is 13.3×; the saturation argument is unchanged, the
number was wrong.)*

**With the anchor fixed and the bar made layer-relative, the levels separate cleanly**
(schnell, one prompt, one seed, 8 multiples from ×0.06 to ×8, carrier population):

| window | clean sink strength | L1 | L2 | **L3 (bar 10× uniform)** |
|---|---|---|---|---|
| early 14–16 | **4.04×** | ×1.0 | ×4.0 | **never — peaks at 7.83 at ×8** |
| late 40–42 | **6.97×** | ×0.5 | ×2.0 | **×0.5** (crosses between 0.25 and 0.5) |

**Early induction reaches L1 and L2 but never L3.** That is the Level-2 outcome this
design exists to name: a register-like state that is not recognised as an attention sink
at that depth. Early sink strength is also **non-monotonic** — it falls from 4.04 to 3.39
up to ×1.0 before recovering, so a weak `v*` induction makes those tokens slightly *less*
preferred before it starts helping.

Persistence, measured as excess over clean, shows every early row as **reconstructed**
with retention 10–19× — the excess at blocks 18+ reaches ~25,000 at ×2, the same order as
the natural register's own `α` of 26,188.

### The control resolved it, and L3 had to be redefined to read it

`carrier_is_sink` was `any(is_sink)` over the carrier set. One token out of sixteen
clearing the bar satisfied it — and at a depth just after dissolution some carriers are
**already** sinks in the clean run, so an `any`-based L3 fired at ×0.06 and reported
baseline sinkhood as successful induction. The level now follows the **population mean**,
with the clean value and the sink fraction reported beside it. Pinned by a test.

**Read against the mean, on schnell, one prompt, one seed:**

| population | window | clean | peak | crosses the 10× bar? |
|---|---|---|---|---|
| carriers | early | 4.04 | **7.83** | **no** |
| carriers | late | 6.97 | 87.72 | yes, ×0.5 |
| ordinary | early | 2.60 | **8.77** | **no** |
| **ordinary** | **late** | **1.70** | **85.22** | **yes** |

Ordinary tokens at block 40 — no register history, sink strength 1.70× in the clean run —
reach **85×**. So the re-excitation confound does not hold: **depth 40 supports
register-like attention routing on tokens that were never registers.** Early fails on
*both* populations, so its failure is a property of the depth rather than of carrier
selection.

### Early induction disrupts the natural register rather than being amplified

Excess over clean after an early window grows for two opposite reasons, and
`new_carriers_after` against `frozen_still_highnorm_after` separates them:

| multiple | new high-norm tokens after | frozen carriers still high-norm |
|---|---|---|
| ×0.5 | 0 | 262 |
| ×1.0 | 90 | 231 |
| ×2.0 | 323 | **9** |
| ×4.0 | 327 | **0** |

At ×2–4 the natural register **collapses at its own positions and relocates to ~325
different tokens**. The earlier "reconstructed, retention 18.8×" reading was that
relocation, not the model amplifying the induced state.

**This constrains the pilot.** Condition B is "early induction, natural window *intact*",
and at ×2 and above it is not intact — B would silently become B-plus-suppression. The
largest early strength that reaches L1 while leaving the register standing is **×1.0**
(frozen 231 of 262 still high-norm). The threshold summary now prints this table for the
early carrier arm so the constraint is visible before the pilot rather than after.

*(Caveat on the ordinary arm's version of these two columns: "frozen" there means the
ordinary positions, so its `new_carriers_after` ≈ 266 is dominated by the natural register
doing its ordinary job and is not comparable with the carrier arm's.)*

### The confound that gated the pilot, and the control that resolved it

**The late carriers are the same tokens that were registers at 20–39.** They have
dissolved but retain residual query–key structure, which is why their *clean* sink
strength at block 40 (6.97×) already exceeds their clean strength at block 14 (4.04×). So
"late induction reaches L3" may mean **a recently-dissolved register can be re-excited**
rather than **this depth supports a register**. Those are different claims, and reporting
the first as the second is the central false conclusion available in this experiment.

The same ambiguity applies to the early persistence result: excess over clean growing to
~25,000 after the window is consistent both with the model *amplifying* the induced state
and with the induction *disrupting* the natural register into looking different.

The sweep therefore runs **two populations** — the frozen carriers and the matched
ordinary positions selected in 4.1 — and records `new_carriers_after` against
`frozen_still_highnorm_after` so amplification and disruption are distinguishable. The
threshold summary prints both populations side by side. Neither costs an image.

The sweep now also prints `perturbation_over_median_norm` and warns explicitly when every
row passes every level.

`out_of_distribution_ratio` = target ÷ the largest absolute projection any token reaches at
the recipient layer in the clean run. **Above 1.0 the induction asks for a state the layer
never naturally contains.** It is reported, never clipped, and the notebook prints a
warning rather than proceeding quietly. An image change from such a perturbation is not
evidence that a register acquired a new function.

Strength is **not** tuned per condition to make each one produce a desired result. One
small predeclared sweep establishes feasibility; the pilot then uses one frozen value.

---

## 4. Conditions

| id | early | natural | late | read against |
|---|---|---|---|---|
| **A** clean | — | — | — | reference |
| **B** early, natural intact | induce | — | — | A |
| **C** late, natural intact | — | — | induce | A |
| **D** late, natural suppressed | — | suppress | induce | **F**, and C |
| **E** early, natural suppressed | induce | suppress | — | **F**, and B |
| **F** suppression only | — | suppress | — | the control that makes D and E readable |
| **G** sham | hooks run, nothing changes | | | must reproduce A |
| **X** fixed-carrier suppression | — | suppress (frozen mask) | — | diagnostic; not an image condition |

**The schedules that actually run.** These are for FLUX.1-dev with its measured
boundaries (forms at 18, declines from ~30, gone by 40), `WINDOW_LENGTH = 3`,
`BRIDGE_LEAD = 2` and `EXTENSION_BLOCKS = 3` (§0):

| id | induction hooks | suppression hooks | terminal cleanup |
|---|---|---|---|
| **B** | early 14–16 | — | — |
| **C** | bridge 28–39, then extension 40–42 | — | — |
| **D** | late 41–43 | 17–39 | 40 |
| **E** | early 14–16 | 17–39 | 40 |
| **F** | — | 17–39 | 40 |
| **F_norm_preserving** | — | 17–39 (norm-preserving operator) | 40 |
| **X** | — | 17–39 (frozen mask) | 40 |

What this table makes explicit:

- **Neither early nor late-only touches the natural lifetime.** Early ends at 16, before
  suppression starts at 17 and before the state forms at 18. D's replacement starts at
  41, after the interval and its cleanup.
- **Suppression starts before the formation onset.** It starts at the earlier of the
  formation onset and the block after the early window. The state is therefore prevented,
  never allowed to form and then deleted, and an early-induced state is removed before any
  natural-window block consumes it. D, E and F share exactly this schedule.
- **Every interval carries one hook past its end.** A hook at a block's input protects
  that block's consumers. Nothing else would stand between the interval's last output and
  the next block's attention.
- **No block carries both kinds of hook.** D's replacement no longer shares block 40 with
  the cleanup, so the two cannot cancel. The notebook hard-stops if any condition would
  suppress and induce at the same block.
- **C and D use different late windows on purpose.** C maintains the naturally formed
  state from before its decline, then extends it past the natural end (`bridge`, then
  `extension`). D induces a replacement after a suppressed interval (`late`). Neither is
  forced into the other's window.

**Every inducing condition runs twice.** B, C, D and E use natural-register-matched
induction (§3). Their `__dironly` twins use the direction-only injection with the same
schedule. The twins are listed in `DIRECTION_ONLY_CONTROLS` in 2.1, and each runs right
after its matched condition. The weak early arm `B_early_weak` is direction-only by
definition, because its strength is a multiple from the direction-only sweep. The
required comparisons gain B, C, D and E against their own twins.

Matched controls, run for whichever condition shows an effect: an **unrelated direction**
(unit length, projected exactly orthogonal to `v*`, fixed by seed) and **ordinary-token
positions** at matched magnitude.

A test asserts D, E and F declare the *same* natural-window action; if they ever diverge
the comparisons that interpret D and E are meaningless.

**Interpretation rules carried into RESULTS.md:** D differing from F is not a successful
relocation; E resembling A is not a successful early replacement; and suppression may
cause damage no later intervention can reverse even when the later state is
representationally identical.

---

## 5. What is measured, in every condition, at every block

Not only inside the windows — the persistence question can only be answered where nothing
was hooked. `lifecycle.achieved_lifecycle` records per block:

`carrier_projection`, the all-token projection p99 and max, `carrier_norm`, `median_norm`,
`carrier_norm_ratio` and `carrier_cosine` (the two statistics matched induction sets, read
back at every block),
`n_highnorm` split into **frozen vs new** with the new ids, the dominant channel and its
value, `carrier_sink_strength`, `carrier_is_sink`, `carrier_n_sink_heads`, `n_sinks` split
into frozen vs new, `carrier_qk_cosine` and `carrier_qk_rank`.

**Attention is scored absolutely** — against uniform over the whole key sequence, text
included — using `sink_readout(absolute=True)` with the `image_mass`/`n_keys` the tracer
records. §9 forbids normalising away mass diverted to text, and the image-renormalised
default would do exactly that. The scale used is written into every row as
`attention_scale`, so a run that falls back is visible rather than silently different.

---

## 5b. The temporal dose — why the first pilot had no image effect

The first FLUX.1-dev pilot produced images visually identical to clean in every
condition. It was not a null result about the register, and it was not a broken
intervention. **The edits fired at one denoising step out of twenty.**

Three things in that pilot's own contact sheet say so, and they say it together:

1. **G (sham) was exactly black** — the hook machinery is inert when it is supposed to
   be, so nothing was wired wrong.
2. **Every real condition was non-zero** — the edits reached the forward pass.
3. **The differences rank-ordered by the number of hooks** — sham 0 < B/C (3 sites) <
   D/E/X < F (24 sites). A dose-response, at a dose too small to see.

A diffusion sampler is a contraction toward the data manifold. Perturb one step and the
remaining nineteen steps of *unedited* sampling pull the trajectory back. Measured on the
synthetic checkpoint with everything else held identical, `steps=[STEP]` delivers 4 edit
calls where `steps=None` delivers 32 — on the real 20-step run, 24 against 480. **We were
spending 5% of the available dose and reading the result as a property of the register.**

`INTERVENTION_STEPS` makes the schedule explicit and defaults to `'all'`:

| value | what it is for |
|---|---|
| `'all'` | every denoising step. What an **image-level** readout needs. |
| `'capture'` | the single traced step. The **mechanistic** readout — one clean sweep down the stack, which is exactly what the delivery and regrowth tables describe. Expect little or no image change, and do not report that as evidence. |
| `[8, 9, 10]` | an explicit list. |

**This is a dose, not a confound.** Depth is still the only thing that differs *between*
conditions; every condition gets the same temporal treatment. What changes is that the
image is given a reason to move. Below 50% coverage the notebook prints a standing
warning that an absent image effect is a statement about the dose, and 5.1 records
`edit_calls` and perturbation energy per condition to `dose.csv` so the two axes of dose
— depth and time — are always visible beside the result.

Two consequences the accounting had to absorb. `suppression_verdict` now takes `step`:
without it, a schedule firing at every step interleaves twenty sweeps down the stack and
the next step's *first* hook is counted as regrowth after the previous step's last.
`Suppressor.seen` is now per step for the same reason — one pooled set would report zero
relocation from step 1 onward.

### What freezing at one step actually commits to

Firing at every step exposes something the one-step schedule hid: **two quantities are
read from the clean run at a single step and then applied at all of them.**

- **Positions.** `CARRIERS` comes from the clean run at one block, one step. If the
  register sits on different image tokens at step 2 and step 18, the induction arms spend
  most of the trajectory writing into tokens that carry nothing — an intervention on
  ordinary tokens wearing the label of a register experiment.
- **Scale.** `calibrate_alpha` anchors the coefficient to the recipient layer's clean
  residual scale *at that same step*. An early denoising step is far noisier than a late
  one, so one frozen alpha can be unremarkable at one end of the trajectory and out of
  distribution at the other. This is usually the larger of the two and the easier to
  miss, because nothing about it looks wrong.

**Only the induction arms are exposed.** The state-based suppression rule re-derives its
selection at every hook, so it follows the population through depth *and* through time;
D, E and F need no fix. What is exposed is B, C, the induce half of D and E, and the X
diagnostic's fixed mask (by design — but now across every step, which changes what that
diagnostic means).

`lifecycle.carrier_drift` re-selects with the **same conjunction** at every denoising step
on the clean run and reports the Jaccard overlap with the frozen set and the scale ratio
to the reference step; `drift_verdict` judges them **separately**, because the remedies
differ and a run can need one without the other. Cell 4.1c runs it before any condition is
built, image-free, and `PER_STEP_INDUCTION` (default: decided by the measurement) switches
`lifecycle_edit` to per-step positions and coefficients. A step the clean run cannot
calibrate gets **no entry**, and the operator then skips that step rather than borrowing
another step's number — silently falling back is the failure the parameter exists to
prevent.

No carrier at any step, the reference included, is reported as **`not measured`**, never
as drift: calling an absent population unstable would send a run to per-step re-selection
of nothing and would read as a finding about the register.

### And the contact sheet was hiding what was there

The difference panels shared one scale taken from the **absolute maximum** across
conditions. A single saturated pixel at 250 set the range, and real differences of
10–30 grey levels rendered as black. The scale is now the 99.5th percentile over all
conditions — still shared, so the panels stay comparable — with the true maxima printed
and a per-condition table of mean, p99, max and the share of pixels moving more than two
grey levels written to `image_difference.csv`. A figure that makes a small effect
indistinguishable from no effect is not a conservative choice; it is a wrong one.

---

## 6. Depth is the variable; denoising time is the dose

The **depth windows** are what differ between conditions. The **denoising schedule** is
identical across conditions and is set by `INTERVENTION_STEPS` (§5b), which defaults to
every step because an image-level readout needs the dose. Both are recorded in every row
and in `dose.csv`.

The two must not be conflated in a claim. "Suppressing the register at blocks 17–39
changes the image" is a depth claim only if the temporal schedule was the same in the
conditions being compared — which it is, by construction. "One denoising step is enough"
is a *different* experiment: run `INTERVENTION_STEPS='capture'` against `'all'` at the
same depths and compare, rather than reading a weak one-step image effect as a weak
register.

---

## 7. Declared limitations

1. **Frozen `v*` is absent for both main-study models** in this repository (see Q15 §3).
   The Q16 notebook loads the same artifacts and refuses without one.
2. **The FLUX early/late confound** with dual vs single stream (§2). Structural, not
   fixable by design — only reportable.
3. **Oracle position selection.** Early-window carriers are the positions identified from
   a *later* clean block. Valid for mechanistic diagnosis, **not a deployable procedure**,
   and stated as such wherever it appears.
4. **Model revisions are not pinned** (`ModelSpec` records `repo_id` only).
5. **One prompt, one seed, one step** in the pilot. Nothing generalises from it.
6. Early FLUX windows may simply never support a register-sized projection; §3's
   out-of-distribution ratio is how that is detected rather than forced.
7. **L1/L2/L3 are cheap to pass by brute force.** At a large enough perturbation all
   three are arithmetic consequences rather than findings. The sweep reports the
   perturbation in units of the recipient layer's median norm alongside every level, and
   the smallest multiple reaching L3 — not the largest effect — is what the pilot uses.
8. **The measured schnell lifecycle disagrees with the frozen artifact.** The fitting
   population shows onset at 17 (n=7), the write at **18** (n=398, norm 30,887), a
   plateau to **34**, decline **35–39**, and effectively nothing from 40. The artifact
   declares `register (20,39)` and `dissolution (40,56)`, so dissolution actually begins
   at **35**, five blocks earlier than declared. The windows are kept as frozen for
   comparability with Q13/Q14; the measured curve is reported rather than the declared
   one. Note also that the plateau count is **censored by `register_topk = 8`** (400 =
   10 prompts × 5 seeds × 8), so its true height is unknown; the decline is real because
   the count falls below the cap. **The late window is placed on the measured onset, not
   the declared one** (§2); the natural and suppression windows are kept as frozen for
   comparability with Q13/Q14.
9. **Suppression controls what is *delivered*, not what is written.** Inside the interval
   the state is removed before any attention or feed-forward reads it, and the terminal
   cleanup closes the one boundary where that would otherwise fail — but it is not
   removed from the residual stream *between* blocks, and the regrowth numbers say so
   plainly. A claim that the state "never existed" would be wrong. What this design
   supports is that no computation inside the interval consumed it.
10. **Delivery is measured as alignment, because on these checkpoints magnitude cannot
    reach a weight.** If a checkpoint's adaptive norm passed token magnitude through, the
    alignment gate would remain necessary but stop being sufficient; 4.1b prints a warning
    when the consumed spread exceeds 1.5×, and that warning would have to be carried into
    the write-up.
11. **Rule coverage is audited, not guaranteed.** `delivery_coverage` measures whether the
    high-norm-AND-aligned conjunction misses a precursor at the writer. Where it does and
    the rule is left as frozen, the under-coverage is a stated limitation of that run, not
    something the delivery verdict silently absorbs.
12. **Natural-register-matched is not transplanted.** The primary arms set the natural
    register's `v*` projection and relative norm. They do not set the rest of the token's
    direction, anything the register carries beyond `v*`, or its history. 4.2 measures
    the remainder against an ordinary-token baseline, and the verdicts are worded so that
    matching two statistics is never described as moving the mechanism.
13. **The matched state is usually far outside what an early block holds.** 4.2 reports
    it per block (norm and projection against the block's largest clean token), and it
    is not clipped. On PixArt, magnitude reaches the attention logits, so matched
    sinkhood at an early block is weaker evidence there than on FLUX.
14. **Result (a) holds by construction at hooked blocks.** The removal bar sits at or
    below the results bar, so a hooked block cannot receive a register-like token. The
    informative parts of (a) are the first unhooked block after the interval, and how
    close the most aligned received token came to the natural register. The informative
    parts of suppression overall are (b) and (c).
15. **The removal rule touches a few ordinary tokens.** At the 0.999 quantile it strips
    `v*` from roughly 0.1% of ordinary tokens per hook, besides the register state. D, E
    and F share this exactly, so their comparisons are unaffected. The clean-run audit
    (4.4) reports the count.
16. **The site pilot is one denoising step and one prompt.** It says whether the two
    sites produce the same post-block state. It does not choose the site automatically.

---

## 8. Next design: translate the lifetime instead of pulsing and wiping it

The current conditions remove the whole natural lifetime and add a three-block pulse
elsewhere. That confounds "the lifetime moved" with "the state was wiped" — and the
wipe dominates the image. The research question asks what happens when the lifetime is
*shifted*, which is a translation. Its edges can be moved independently:

| manipulation | how | blocks touched |
|---|---|---|
| onset earlier by k | induce at the carriers over the k blocks before the writer | k |
| onset later by k | suppress the first k blocks of the natural lifetime | k |
| offset earlier by k | suppress the last k blocks before dissolution | k |
| offset later by k | maintain the state for k blocks past the dissolution onset | k |

A 2 × 2 of onset shift × offset shift at k ∈ {3, 6} factorises emergence time from
dissolution time — the two events the question names — and manipulates only a few
blocks each, so the edits stay small and the operator confound shrinks with them.
Condition C is already "offset later by 3". Induction should be natural-register-matched
(§3, now the primary), at the measured carrier population
(`N_CARRIERS` from the rule-selectivity audit), and the suppressed edges should use the
operator 7.2b favours.

## 9. Stages

| stage | content | gate |
|---|---|---|
| 1–3 | design, operators, tests, sham equivalence | ✅ no weights, no cost |
| 4 | small FLUX pilot, conditions A–G | **user approval + a frozen `v*`** |
| 5 | inspect achieved trajectories and paired images | **stop here if no induction** |
| 6 | main study: FLUX.1-dev and PixArt-Σ independently | only if 5 is interpretable |
| 7–8 | replication, figures, RESULTS.md | — |

**Before reading any stopping rule:** check `dose.csv`. A condition that edited a small
fraction of the denoising steps has not tested anything at the image level.

**Stopping rule.** If neither early nor late induction produces the internal signatures,
the experiment reports that and does not proceed to an image-level study. The failure is
then diagnosed as one of: weak query–key geometry, rapid overwriting, insufficient or
excessive strength, wrong site, or instrumentation — not searched around.
