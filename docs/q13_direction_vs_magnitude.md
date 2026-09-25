# Q13 — Is `v*` a causal state variable, or a correlate of large magnitude?

## The problem this experiment exists to solve

Everything established so far about the register population is correlational in one
specific way. The tokens were **selected by norm**; they turn out to be aligned with `v*`;
the dominant channel carries most of both. Norm, `v*`-alignment and sinkhood move together
in the clean model, so **no observation of the clean model can separate them**.

A reviewer's objection writes itself: these are just the loud tokens, `v*` is what loud
tokens look like, and nothing here needed a direction.

## Experiment A — the 2D control surface

For a token state `x` and unit `v = v*/||v*||`:

```
alpha = x . v        r = x - alpha v        x = r + alpha v
```

**Alignment at fixed norm.** An alignment multiplier `beta` rescales only the component
along `v`, and the token is then renormalised to its original length:

```
y(beta)      = r + beta alpha v
y~(beta)     = ||x|| y(beta) / ||y(beta)||
```

so `cos(y~, v) = beta alpha / sqrt(||r||^2 + beta^2 alpha^2)` rises monotonically in `beta`
while `||y~|| = ||x||` **exactly**.

**Magnitude at fixed direction.** A norm multiplier `gamma` then scales the whole vector,
which cannot change the direction `beta` produced:

```
x'(beta, gamma) = gamma y~(beta)
```

`beta in {0, 0.5, 1, 1.5, 2}` against `gamma in {0.5, 0.75, 1, 1.25, 1.5}` gives a 5x5 grid
in which the two variables vary independently. That independence is the whole contribution,
and it is asserted rather than assumed: `tests/test_control_surface.py` holds `beta` to
exact norm preservation and `gamma` to exact direction preservation.

### The centre cell is the numerical floor, not a definitional zero

`(beta, gamma) = (1, 1)` reconstructs `x` to ~1e-7 relative in float32. The grid
deliberately **runs the edit** at the centre rather than skipping it, so that cell measures
the model's own sensitivity to a round trip through the operator. Every other cell is read
against that number. Skipping it would have produced a clean zero that means nothing.

### The degenerate case, stated

`y(beta)` vanishes when a token is collinear with `v` and `beta = 0`: there is no residual
left to point along. Renormalising the numerical residue would amplify noise and can even
restore the original direction with its sign flipped. Below 1% of the original norm the
direction is therefore replaced by a **seeded** random direction orthogonal to `v`, and the
count is reported per condition rather than hidden. This mirrors `causal_ops.remove_direction`,
which faced the same limit.

## Experiment B — the causal rescue

The dominant channel is necessary for the high-norm state, but the channel and the
direction are confounded in the clean model too. Five conditions separate them:

| Condition | Role | What it isolates |
|---|---|---|
| `clean` | reference | the paired baseline |
| `channel_ablate` | condition | the damage |
| `channel_ablate_vstar_rescue` | condition | `x'.v` set back to `alpha_clean` |
| `channel_ablate_orthogonal_rescue` | **control** | the **same per-token L2**, orthogonal to `v` |
| `channel_restore_only` | condition | only the dominant coordinate restored |

`channel_restore_only` is the comparison that decides how much of this is geometry: if
restoring one coordinate suffices, the mechanism is axis-aligned and should be described as
a channel; if the full `v*` rescue does substantially better, the distributed direction
matters beyond the single massive coordinate.

### Two things that would have invalidated it

**The rescue target must come from the paired clean trajectory.** `alpha` measured *after*
the ablation is exactly the quantity the ablation destroyed; a rescue that reused it would
restore nothing while looking like a rescue.
`test_the_rescue_target_comes_from_the_clean_trajectory_not_the_ablated_one` is the guard.

**Hook ordering must not be a question.** Ablation and rescue are composed inside **one**
edit function, ablation first, so nothing depends on the order PyTorch happens to fire two
hooks. `channel_restore_only` is the case that would silently degrade to a bare ablation if
the order inverted, and it has its own test.

### The mediation number is a share of the damage, not an absolute level

How hard the ablation bites varies by checkpoint, by layer and by how much of `v*` the
dominant channel actually carries. `gap_closed` reports

```
closed = (m_rescue - m_ablated) / (m_clean - m_ablated)
```

so 1.0 means fully rescued and 0.0 means nothing was undone, independent of the damage. An
absolute threshold would reject a perfectly good rescue wherever the ablation happened to
be gentle — a miscalibration this experiment's own validation gate caught before any GPU
time was spent.

## The treatment population

Frozen from the clean pass and reused verbatim by every condition. Seven modes, all reading
`select_frozen_targets`' clean-run output:

| Mode | Rule |
|---|---|
| `highnorm_and_aligned` | **primary**: the repository's norm threshold AND its alignment bar |
| `register` | the repository's percentile high-norm rule |
| `topk_norm` | top-k by clean residual norm |
| `topk_projection` | top-k by clean `x.v*` |
| `percentile_projection` | top percentile by clean `x.v*` |
| `sink_only` | clean tokens that are some head's strongest image key |
| `random_tokens` | count-matched ordinary tokens |

Neither half of the primary intersection is a new definition: the norm bar is
`highnorm_ratio x median` from `SweepConfig`, the alignment bar is the one
`select_frozen_targets` already derives from the natural register population. An empty
intersection **falls back loudly** — the string `FALLBACK` appears in the saved rule and the
notebook's validation gate fails on it — because a silent fallback would turn every
condition into a no-op.

Which selected tokens were attention sinks on the clean run is recorded, so sinkhood can be
analysed as an outcome rather than assumed as a property.

## Sinkhood uses the existing criterion

`SweepConfig.sink_ratio_threshold` defines a sink as absorbing at least 10x the uniform
share `1/N` of the image-to-image attention mass, and `metrics.layer_table` applies it to a
layer's head-max. `sink_readout` applies the same rule **per token**, so `is_sink` means
what it means everywhere else in the project. `n_sink_heads` — how many heads take the
token as their strongest image key — is reported beside it, because the two are not the
same question. No new sink definition is introduced.

## Image-level measurements

Paired against the same prompt and seed, so any distance is attributable to the edit.
`q11._image_fidelity` supplies RMSE, PSNR and LPIPS (degrading gracefully when LPIPS is not
installed); CLIP is opt-in via `clip_model` and off by default, because a metrics table that
silently downloads several hundred megabytes mid-run is not one anybody asked for.

`image_detail.py` adds three transparent local-detail readouts:

1. **Fourier band split** into low / mid / high, reported **beside each band's own area**,
   because "30% of the energy is high frequency" is not readable as high or low without the
   white-noise expectation. `spectrum_<band>_over_uniform` is that comparison, and it is
   1.0 for an unstructured difference.
2. **Sobel magnitude change**, so a softening can be told from a different scene.
3. **High-gradient concentration** — the share of difference energy landing on the top 20%
   of the *clean* image's Sobel magnitude, divided by that mask's area. Above 1 means the
   effect concentrates on contours and texture. The mask is rank-based rather than
   value-based: a generated image often has large flat regions where a value threshold sits
   at zero and the mask silently becomes the whole frame.

Every quantity is a mean of squares over an explicit mask, computed in numpy, so any number
can be re-derived from the saved images.

## Controls

| Control | Matched on | Question it answers |
|---|---|---|
| `random_direction_beta*` | per-token L2 | was it this direction, or any direction of that size? |
| `orthogonal_beta*` | per-token L2 | same, with the direction provably not `v*` |
| `random_tokens_beta*` | edit and count | was it these tokens, or any tokens? |
| `sink_only_beta*` | edit | does the sink subset carry what the aligned population carries? |

Energy matching is computed **on the live tensor** at every layer, against what the `v*`
edit would have done to that same tensor — not against a target precomputed from the clean
pass. Under a windowed intervention the trajectory has already diverged by the second
edited layer, so a precomputed target would drift and the control would stop being matched.

## The placebo gate

Layers upstream of the edit cannot be reached by any condition, so their readouts must be
identical across conditions. The spread is computed and reported on every run, and the
notebook's validation fails if it is not ~0. A leak there is a bug, not a finding.

## Resumability

Completed cells are cached under `<root>/cells` as one JSON plus three CSVs each, keyed by
`scope|condition|prompt|seed`. An interrupted Colab session re-runs only the clean passes —
which supply the frozen population and the paired `alpha`, and hold activations that the
tables do not — and then continues. Verified by
`test_a_resumed_run_regenerates_nothing_and_returns_the_same_numbers`.

## What may and may not be claimed

**The measurements can distinguish:**

| | reads as |
|---|---|
| H1 magnitude | endpoints vary up `gamma` and are flat across `beta` |
| H2 direction | endpoints vary across `beta` and are flat up `gamma` |
| H3 thresholded sinks | sinkhood appears only past some clean projection, not smoothly |
| H4 functional separation | image detail moves with `beta` while sink retention does not |
| H5 mediation by `v*` | the `v*` rescue undoes damage the L2-matched orthogonal one does not |

The verdict reports every contrast with its number and **does not choose between them**.
Coding a preference in would have made the experiment unable to come out the other way.

**Not claimable:**

- **Anything from the `beta` axis where `v*` is one coordinate.** The notebook prints the
  participation ratio. Near 1, `beta` and a single-channel rescale are close to the same
  operation, and this is a result about a channel rather than about geometry. PixArt-Sigma
  is where the distinction becomes non-trivial, which is why the channel is configurable
  (FLUX 154, PixArt-Sigma 293) rather than hard-coded.
- **A norm-threshold count as evidence about `beta`.** `beta` preserves the norm exactly, so
  `n_highnorm` *cannot* move along that axis. It is a positive control for the operator, not
  an endpoint — and it is asserted as one in the tests.
- **Image effects at a cell near the `(1, 1)` floor.** That is sampling sensitivity.
- **Sinkhood conclusions where no token meets the criterion.** The verdict says so when it
  happens rather than printing a vacuous `0% against 0%`.
- **Anything at a step the population was not frozen at**, if `steps` is widened beyond the
  capture step: the frozen token identity drifts as the trajectory moves.

## The caveat fixes (second pass)

Five of the eight caveats from the first pass turned out to need code, not prose.

**The population count `beta` can move.** `n_highnorm` is pinned along the `beta` axis by
construction -- `beta` renormalises every treated token, so no count defined by a norm
threshold alone can move -- which makes it a positive control rather than an endpoint.
`n_highnorm_and_aligned` is the count that *can* move, and it is the population the project
actually cares about: loud AND aligned. Rotating a token away from `v*` at fixed norm removes
it from that set, and `test_the_aligned_population_count_is_the_one_beta_can_move` asserts
both halves.

**Every cell is now readable against the centre.** `rmse_over_floor` and `lpips_over_floor`
express each distance as a multiple of the `(1, 1)` cell's own distance, computed per
prompt-seed-scope because the floor is a property of the trajectory. A cell at 1.2x the floor
has done almost nothing, whatever its absolute value looks like.

**The `beta` axis now has a channel control.** Where `v*` is nearly axis-aligned, `beta`
along `v*` and `beta` along `e_{c*}` are almost the same operation. `channel_axis_beta*`
runs the operator along the pure dominant-channel axis at matched per-token L2, so the
difference between the two is exactly the part of `v*` that is not the dominant channel.
This is the `beta`-axis counterpart of `channel_restore_only` on the rescue arm, and it
answers the axis-alignment question empirically rather than by recollection.

**The PixArt CFG amplification, verified rather than assumed.** Diffusers' PixArt-Sigma
pipeline concatenates `[negative_prompt_embeds, prompt_embeds]` and combines the two branches
as `u + s(t - u)`, so the conditional row is last -- which is what `_row()` reads -- and an
edit confined to it enters only the second term. Its effect on the prediction is therefore
scaled by the guidance strength `s` (4.5 at the PixArt default). FLUX does not batch CFG at
all: guidance arrives as an embedding and the transformer runs at batch 1, so the same edit
is unamplified there. **Cross-checkpoint comparison of image-effect magnitudes is confounded
by this factor**, and it was the opposite of the first-pass caveat, which had described the
conditional-only edit as understated.

The primary arm stays conditional-only, to keep Q13 consistent with Q1--Q6. A small named
`cfg` arm runs four strong settings on both rows -- `beta` low, `beta` high, `gamma` low and
the channel ablation -- and `cfg_partner` names each twin's conditional-only counterpart so
the comparison is a join rather than a second 5x5 grid. On a non-batched pipeline the arm is
skipped with a printed note, because editing "every row" of a batch-1 tensor is the same
tensor operation and running it would reproduce the conditional-only numbers exactly.

**Per-token sinkhood needed no change.** All three candidate quantities were already columns
in `token_metrics.csv` -- `sink_strength_headmax` (the uniform-share ratio),
`is_sink` (that ratio against `SweepConfig.sink_ratio_threshold`), and `n_sink_heads` (how
many heads take the token as their strongest image key) -- alongside `incoming_attention`,
`qk_score`, `qk_rank` and `key_norm`. The analysis chooses; the run records all of them.

## Statistical power, and the floor the project already set

`causal_stats.bootstrap_mean` declines to report an interval below **three** prompt
clusters, on the stated ground that two clusters can only draw three distinct resamples and
the interval would be an artefact. Q13 now uses that machinery rather than a standard error
across units, through `rescue_effects`:

- **paired within unit** against the ablation, so prompt-to-prompt variation -- which is
  large and uninteresting -- cancels rather than inflating the bar;
- **clustered by prompt**, so two seeds of one prompt are not two independent observations;
- referenced to the **ablation**, not to clean, because the mediation question is how much of
  the damage each condition undid.

The refusal propagates. `ControlSurfaceResult.meta` carries `n_prompt_clusters` and
`clustered_intervals_available`, the verdict says in words when no interval exists, the
notebook's gate prints the cluster count beside the generation budget, and
`fig_q13_rescue` names which error bar it drew. An underpowered run cannot buy a confidence
interval by falling back to a SEM.

The default is **five prompt-seed units**: 405 generations over both layer scopes, which
clears the floor with margin.

## Real-scale problems that only appear off the synthetic checkpoint

`tiny-flux1` is 16 channels and 64 tokens; FLUX at 512px is 3072 and 1024. Six things
changed meaning across that gap, and all six are now closed:

| | |
|---|---|
| **`store_keys` cost** | The tracer's `key_layers` path keeps the keys, the queries **and** the full attention probabilities and logits -- ~312 MB per layer-step at 512px, **6.6 GB** over the register window, on top of the model's own 24 GB. The old comment said "heads x tokens x head_dim" (12 MB), a 500x understatement. Now stated honestly and capped by `store_keys_layers`, which defaults to 1. `qk_score` and `qk_rank` never needed it. |
| **Duplicate state storage** | `full_state_layers` in both the runner and the geometry capture stored a second copy of every layer that nothing read -- 0.24 GB per trace and 0.27 GB per capture. Removed; the probe is the single source. |
| **`v*` read off the basis** | `fig_token_decomposition` and `fig_channel_decomposition` took `basis.vectors[0]` as `v*`, which is PC1 on the PCA basis. They would have decomposed a token along PC1 while labelling the leg `alpha_i v*`. Snapshots now carry `v*` explicitly. |
| **Diary double-counting** | `all_batch_rows` calls the edit once per batch row, so each CFG twin wrote two notes and any mean over `perturbation_l2` counted both. Notes now carry `batch_row_call` and `is_conditional_row`. |
| **Positional merge** | `_add_floor_columns` assigned a merge result back by position, which is safe only while pandas preserves order and multiplicity. Now a key lookup. |
| **Unfingerprinted resume cache** | The cache key identifies a *cell* -- scope, condition, prompt, seed -- not an *experiment*. Widening the grid or changing the treatment rule produced the same cell ids with different meanings, and a resumed run would have mixed them silently. `experiment_fingerprint` hashes everything that can change a number (including `v*` itself) and excludes everything that cannot (paths, image saving, amplification); a mismatched cell is deleted and recomputed, and the count is reported. |

## The CFG arm is an image-level diagnostic, by construction

Worth stating separately because the natural way to read it is wrong. The tracer records the
conditional row, and batch rows do not mix inside the transformer -- each is an independent
element of one forward pass. So a twin and its conditional-only partner have **identical**
internal readouts whatever the guidance strength; verified on `tiny-pixart`, where every
`selected_alpha_mean` matched to four decimals.

The two choices diverge in the **sampler**, where `eps = u + s(t - u)` combines the branches.
A reader who compared `population_metrics` between the twins would find no difference and
conclude the branch choice is irrelevant, which is the opposite of the truth. Notebook § 7.6
compares images only and says why.

## What the first real FLUX validation run exposed

Two flaws that the synthetic checkpoint could not show.

### The resume cache was not keyed on the code

The validation reported *"edit-hook rescue diagnostics are missing"* while the code that
emits them was in the checkout. `experiment_fingerprint` covered the config and the context
but not the analysis code, so a commit that added diagnostics left every setting identical
and the cache served the old rows -- without the new columns. It now includes
`_analysis_code_digest()`, a hash of the **source text** of `control_surface.py`,
`endpoints.py` and `causal_ops.py`: the three modules that decide what a cell records.
Source text rather than git SHA, so a commit touching only a notebook does not discard GPU
hours, and a detached Colab clone still gets a stable digest.

### The orthogonal control was never a matched null

The run reported the orthogonal control closing **407%** of the ablation gap, with a mean
`v*` projection 3.8x clean. The missing diagnostic was the distance from the clean state,
now recorded at the edit hook:

| condition | gap after ablation | gap after rescue | ratio |
|---|---|---|---|
| `channel_ablate` | 736.7 | 736.7 | 1.000 |
| `channel_ablate_vstar_rescue` | 736.7 | 36.8 | **0.050** |
| `channel_ablate_orthogonal_rescue` | 736.7 | 1037.1 | **1.408 ≈ √2** |
| `channel_restore_only` | 736.7 | 0.0 | 0.000 |

Matching the **injection L2** matches the effort spent, which is the right way to control
for effort -- but it does not match the **resulting distance from clean**. The rescue spends
its budget moving back toward the clean trajectory; an orthogonal step of the gap's own
length spends it moving sideways and lands at `sqrt(2)` times the gap, *further* from clean
than the damage it controls for. The original comment in the code claimed "size cannot
explain a difference between them", and that was wrong.

So a condition with `rescue_closed_distance_ratio > 1` is not a matched null, and whatever
it shows downstream is at least partly the network responding to a larger perturbation.
`displacement_report` records it, the runner logs it per condition, the verdict states it,
and notebook § 4.2 prints it above the rescue table. It is **reported, not gated**: you
cannot reduce the distance from clean by moving orthogonally, so `sqrt(2)` is the
mathematical content of the comparison rather than a bug to fix.

### And a metric that could have manufactured the same number

`selected_alpha_recovery` is a ratio to the clean projection mean, averaged over depth.
Beyond the register zone that projection collapses toward zero **by design** -- that is what
dissolution means -- so the denominator vanishes and a mean over layers is dominated by
whichever layer divided by the smallest number. A large perturbation can then read as a
several-hundred-percent recovery from arithmetic alone.

Three changes: the ratio is suppressed where the clean mean does not clear a tenth of the
ordinary-token spread (`recovery_denominator_resolvable` says where);
`selected_alpha_change_in_spreads` reports the same movement in the repository's own unit
(`endpoints.ordinary_projection_spread`, the unit `measure_layer` already uses for
`vstar_projection_change`), which is defined everywhere including after dissolution; and
`is_register_zone` lets every summary restrict to where the targeted state exists, which
`rescue_effects` and § 4.2 now do.

On the synthetic checkpoint the corrected register-zone summary reads: `v*` rescue 0.995 of
clean (98% of the damage undone, 0.05 spreads of residual movement), channel-restore 1.000,
orthogonal 0.796 (0% undone) -- the shape the design intends.

## The intervention-depth arm — does *when* the edit lands matter?

Everything above installs the edit at one layer. This arm installs the **same** edit at
three points in the register's life, so any difference between them is about when the state
was touched. It is additive: its own directory, its own fingerprinted cache, the same frozen
`v*`, the same token selection, the same `BLOCK_INPUT` point. The main 5×5 grid and the
rescue arm are unchanged and their outputs are not rewritten.

### The layers come from the frozen ranges, not from FLUX's numbers

| Scope | Offset | FLUX | PixArt-Σ |
|---|---|---|---|
| `birth` | `writer_layer - 1` | 18 | 9 |
| `boundary` | `register_layers[0]` | 20 | 10 |
| `established` | `register_layers[0] + 4` | 24 | 14 |

`boundary` is the layer the main grid already used, so it is the anchor the other two are
read against. The offsets are resolved against `ctx.writer_layer` and `ctx.register_layers`
and clamped to the model's depth, so a checkpoint with a different geometry gets the
corresponding layers rather than FLUX's.

`depth_layer_table` is printed before anything is spent and reports two things the run
depends on: whether the three scopes resolve to **distinct** layers, and whether each lands
inside the phase it is named after. The notebook hard-stops on the first — two scopes on one
layer would compare a layer against itself — and warns on the second, because the layers are
still distinct and the comparison still runs; only the phase *labels* become unwarranted.
On the synthetic checkpoint `established` clamps outside the register range and the warning
is what fires, which is the behaviour to expect on any short model.

### The cheaper grid, with a negative arm

β ∈ {−0.25, 0, 0.5, 1, 2} against γ ∈ {0.5, 1.0, 1.5, 2.0}: 20 cells rather than 25, plus
the 5 rescue conditions and the 10 controls anchored on the sweep's extremes — 35 per depth
per unit. At 3 depths and 5 prompt×seed units that is **530 generations**, and the notebook
prints the figure before the gate.

Only `boundary` shares cells with section 6, and only at β ∈ {0, 0.5, 1, 2},
γ ∈ {0.5, 1, 1.5}. Effect magnitudes are therefore not comparable against the main grid
outside that overlap.

β < 0 is new here: it inverts the sign of the projection at unchanged norm, which the main
grid never does. `cos(ỹ, v) = βα/√(‖r‖² + β²α²)` is odd in β, so β = −0.25 lands at a
negative cosine of smaller magnitude, not at a mirror of β = +0.25.

### Two diagnostics the grid alone could not supply

The three questions this arm exists to answer each need a measurement the existing tables
did not carry.

**Is negative β a meaningful opposite-`v*` state, or off-manifold corruption?** Neither norm
nor alignment can answer it: the operator fixes both by construction, so an inverted state
and a corrupted one of the same length are indistinguishable on every column the main grid
records. `clean_subspace` takes the leading right singular vectors of the mean-centred clean
slice at that layer, and `onmanifold_energy` is the share of a treated token's centred energy
lying inside it, reported as `onmanifold_ratio_to_clean` against a typical clean token's own
share. Near 1 the state is one the clean population could have contained; well below 1 it is
not, whatever its norm.

This diagnostic is only discriminative when the subspace is a genuine restriction, so
`onmanifold_rank_fraction = rank / width` is recorded and `onmanifold_is_discriminative` is
gated on it being below 0.5. FLUX at rank 32 of 3072 is 0.010 ✓; the synthetic checkpoint at
15 of 16 is 0.94 ✗ — and there cell 10.4 **refuses to answer** rather than printing a
reassuring 1.0. Variance explained is the wrong key for this: a rank-11-of-12 subspace
explains 0.97 of the variance and restricts nothing.

**Is a flat positive-β response a ceiling or a failure?** `cos` approaches 1 asymptotically
in β, so a depth whose clean state already sits at |cos| ≈ 1 has nothing left for a positive
β to add. `clean_cosine_headroom = 1 - |cos_clean|` is that room, measured on the clean pass,
and `gain_over_headroom` is the share of it the edit used. Without the headroom beside it, a
flat line is ambiguous between the two, and they are opposite conclusions. On the synthetic
checkpoint the headroom is 0.018 and β = 2 uses 0.72 of it — a ceiling, stated as one.

`clean_cosine` is the **signed** mean and `clean_cosine_abs` the absolute one. They were one
column at first, which put a signed realised alignment beside an absolute clean one and made
`cosine_gain` read −0.055 at β = 1, where it must be exactly zero because that cell is a
no-op. Separating them makes the β = 1 identity exact, and cell 10.3 asserts it.
`clean_cosine_sign_agreement` says whether the register population is one-signed at all,
since a signed mean over a split population is not readable.

**Does birth afford more control?** The β-response slope and monotonicity at each depth, on
the image where a decoder exists and on the treated tokens' own projection where it does not
— `fig_q13_depth_response`'s third panel and its caption both say which of the two they drew.

### What it cannot settle

It cannot attribute a depth difference to the *phase* rather than to the layer: `birth` and
`established` are six layers apart on FLUX and any two layers differ in more than their
phase. The phase labels come from the frozen ranges, which is the best available warrant and
not a proof. It also cannot separate "more control" from "more damage" on its own — a larger
image response at birth could be either, and the on-manifold column and the paired
image-detail metrics are what distinguish them.


### In the notebook

Section 10.7–10.11 runs the site arm: `writer+0/1/2` and `boundary`, each at both sites,
grid cells only (no rescue, no controls — those are section 10's and are not repeated), at
β ∈ {−0.25, 0, 1} and γ = 1. Twenty-four generations per unit.

Cell 10.10 is the gate. At β = 0 the projection at the hook must be negligible, and the
test is **relative to the unmodified projection** rather than an absolute epsilon: the hook
records the tensor after it has been cast back to the model's dtype, so on a bf16
checkpoint a genuinely zeroed component reads as a few thousandths of `‖x‖` — which an
absolute threshold would call a failure and stop a working run. The cell requires the
residue below 1% of the β = 1 value and hard-stops otherwise, because if the edit is not
reaching the tensor the block consumes, nothing downstream of it means anything.

`fig_q13_stage_probe` draws one trajectory panel **per site**, never one shared axis, plus
the durability bars. A sign flip is marked only where the residue is at least 5% of the
clean projection: under near-total suppression the leftover is noise about zero and its
sign is a coin toss, so flagging it would publish float precision as a finding.

## Assumptions I could not settle from the repository

1. **Per-token sinkhood.** The repository's criterion is defined on a layer's head-max. I
   applied the identical ratio per token, and report `n_sink_heads` beside it. If the
   intended per-token rule is different, `sink_readout` is the one place to change.
2. **The band edges** in the Fourier split are thirds of the Nyquist radius. Any other split
   invites the suspicion that the boundaries were chosen once the answer was known; the band
   areas are reported so the choice is auditable.
3. **`detail_quantile = 0.8`.** Configurable. The concentration *ratio* divides by the mask
   area, so the headline number is insensitive to it by construction.
4. **PixArt CFG batching** -- settled above by reading the diffusers pipeline, and handled
   by the `cfg` diagnostic arm rather than left as a caveat.

## Files

- `ditsinks/control_surface.py` — the operators, the conditions, the selection modes, the
  measurement tables, the resumable runner, the verdict
- `ditsinks/image_detail.py` — the three local-detail readouts
- `ditsinks/causal_figures.py` — `fig_q13_control_surface`, `fig_q13_separation`,
  `fig_q13_rescue`, `fig_q13_contact_sheet`, `fig_q13_depth_response`,
  `fig_q13_stage_probe`, `fig_q13_rotation_response`, `fig_q13_rotation_plane`
- `notebooks/iclr_q13_direction_vs_magnitude.ipynb` — the gated run, Drive-backed
- `tests/test_control_surface.py`

## The writer-phase stage probe, and why the depth arm needed a second edit site

The first real depth run read `realised_cosine` ≈ +0.98 at layer 18 for β = 0 and
β = −0.25 — impossible immediately after an edit that sets the `v*` component to zero.
It was not a bug in the intervention. It was a bug in where the number is read.

### The two sites, named

`CausalTracer._make_block_post` (`causal_engine.py:311`) is a `register_forward_hook` on
the block, so **`trace.at(step, layer)` is the block's OUTPUT** — `x + attn + mlp`, i.e.
after the feed-forward's residual addition. `population_rows` reads exactly that, so every
`is_edited_layer` row was the post-block state. The edit, meanwhile, was installed at
`BLOCK_INPUT`, a forward **pre**-hook on the same module. The whole block, feed-forward
included, sits between them.

So on a **writer** layer — whose function is to write the register state along `v*` — the
block simply rewrites the component the edit removed, and the readout returns the clean
value while the edit was applied perfectly. Measured at the birth layer, γ = 1:

| stage | β = −0.25 | β = 0 | β = 1 (no-op) |
|---|---|---|---|
| at the hook | +0.026 | **0.000** | −0.105 |
| post-attention residual | −0.007 | −0.033 | −0.139 |
| block output — *what the tables recorded* | −0.049 | −0.076 | −0.180 |

The block rebuilt 42% of the clean projection from a state where it was exactly zero, and
restored the clean *sign* at β = −0.25. **The β = 1 identity check cannot catch this**,
because β = 1 is a no-op at every site; it tests commensurability, not siting.

### `_postmlp`: the same edit, after the MLP has run

A scope may now name the site as well as the layer. `birth` installs at the configuration's
point; `birth_postmlp` installs the identical edit at `BLOCK_OUTPUT` — after the
feed-forward's residual addition, where this block's MLP can no longer rewrite it. The
suffix is part of the scope name, so it reaches the cache key, the output directory and
every table without a second argument threaded through the runner, and **every scope
without a suffix behaves exactly as before**.

`writer+0`, `writer+1`, `writer+2` resolve the writer range block by block — FLUX 17/18/19,
PixArt-Σ 8/9/10. On FLUX that range straddles the architecture boundary: 17 and 18 are
MMDiT dual blocks, 19 is a fused single block, so `pre_mlp_residual` exists at two of the
three and `_supported_layers` drops the third rather than raising.

### The four stages

`stage_rows` records the treated tokens' `x·v*` and signed cosine at:

1. `before_mlp` — the post-attention residual (`PRE_MLP_RESIDUAL`, norm2's input).
2. `after_mlp_write` — the block output as the block produced it. A `StateProbe` registers
   before the `EditInstaller`, so at a post-MLP edit this is the value the edit is about
   to overwrite.
3. `after_edit` — the installer's own `x_post_hook`, whichever site it was installed at.
4. `after_next_block` — how much survives one more block.

`forward_order` is point-aware, because at `BLOCK_INPUT` the hook fires **first** and the
other stages are the block's *response* to the edit, while at `BLOCK_OUTPUT` it fires last
and the earlier stages are the untouched clean trajectory. Sorting both on one order
inverts the meaning. `measure_stages` is off by default, so the main grid is unchanged.

### What the validation showed

Three prompts, both sites, tiny-flux1. At β = 0 the projection at the hook is
7e-9 … 5e-8 at all four scopes — the operator does zero it. One block downstream:

| scope | site | β = 0 suppressed | β = −0.25 sign |
|---|---|---|---|
| `birth` | block input | 50% | restored to clean |
| `birth_postmlp` | **block output** | **85%** | **inverted, survives** |
| `boundary` | block input | 77% | restored to clean |
| `boundary_postmlp` | **block output** | **90%** | **inverted, survives** |

A post-MLP edit is roughly twice as durable at the writer layer, and it is the only one of
the two at which the negative-β sign inversion still exists one block later. That is the
distinction the arm was built to make: *input edit → the MLP rewrites `v*`* against
*post-MLP edit → the state is genuinely suppressed*.

## The true rotation arm — a different direction, not a larger one

`beta` rescales the `v*` coefficient and then restores the original length:
`y(β) = r + βα v̂`, then `ỹ = ‖x‖ y/‖y‖`. Two consequences follow from the form and were
the reason for this arm. The edited direction always lies in `span{v̂, r}`, where `r` is the
token's *own* residual, so no `β` can put weight on a **chosen** direction; and the norm is
held by the renormalisation step rather than by the operator, so "at fixed norm" is a
property of a correction applied afterwards.

### The operator

For `v = v*/‖v*‖`, pick a unit `u` with `uᵀv = 0` and split each treated token as
`x = αv + bu + q` with `q ⊥ v` and `q ⊥ u`. Rotate only the `(v, u)` plane:

```
α' = α cos θ − b sin θ
b' = α sin θ + b cos θ
x' = α' v + b' u + q
```

`‖x'‖² = α'² + b'² + ‖q‖² = α² + b² + ‖q‖² = ‖x‖²` **exactly**, because `(α, b) → (α', b')`
is an orthogonal map of ℝ² and `q` is never touched. No renormalisation is applied and none
is needed.

`rotate_plane` **raises** if the measured norm drifts past `1e-4` relative, rather than
correcting it. A norm that moves means `u` was not unit or not orthogonal to `v`;
renormalising the output would leave the norms right and every reported angle wrong, which
is the failure mode worth being loud about. Measured on FLUX's width in float32 the drift
is ~1e-7, and θ = 0 returns the activation to ~1e-7 — both asserted in cell 11.3 and in
`test_the_rotation_preserves_the_norm_by_construction_not_by_correction`.

### Where it runs, and why

At `writer+1_postmlp` — FLUX layer 18, after the feed-forward's residual addition. The
stage probe showed a pre-MLP edit at a writer layer is partly rewritten by that block's own
feed-forward, so an edit at the block input would have its angle undone before anything
downstream saw it. This is the first arm that depends on the post-MLP site existing.

### The three targets

| target | what it is | its role |
|---|---|---|
| `random_orthogonal` | seeded random unit direction ⟂ `v*` | the null: a direction of the same length, chosen without reference to the data |
| `residual_pc1` | leading principal direction of the clean tokens' residuals after `v*` is removed | the axis the population itself varies in most, once the register direction is taken out |
| `semantic_direction` | supplied by the caller | an interface. It **refuses** without one rather than substituting a random direction and reporting a semantic result |

`u` is projected off `v` and renormalised whatever the target, so `uᵀv = 0` holds to float
precision however the target was obtained. `plane_basis` records the target's raw alignment
with `v*` before orthogonalisation, refuses a target parallel to `v*` (nothing survives, so
there is no plane), and announces one where under 10% survives — there `u` is mostly
whatever was left and the plane is chosen largely by noise. `residual_pc1`'s sign is pinned
by its largest coordinate so `+θ` means one thing across runs.

### The plane share, decomposed — and a claim of mine that was wrong

`in_plane_energy_share = (α² + b²)/‖x‖²`. Its **first term is exactly `cos²(x, v*)`**, so
for a register token aligned with `v*` at `cos ≈ 0.98` the share is at least 0.96 **for
every choice of `u`** and therefore distinguishes no target from any other. An earlier
version of this document and of the figure legend claimed the share bounds the response and
cited "0.4% on FLUX's width" for `random_orthogonal`. That number was measured on a
*different* run — the synthetic checkpoint, where the register-selected token happened to
have `cos = −0.035` with the planted `v*`, making `α²/‖x‖² = 0.0012` and
`b²/‖x‖² = 0.0030`. The arithmetic was correct for that token and the attribution was not.
The formula was never wrong; the reading built on it was.

The two halves are now recorded apart, because they bound different things:

| column | equals | bounds |
|---|---|---|
| `alpha2_share` | `cos²(x, v*)` | the alignment the rotation can **remove**: at θ = 90°, `α' = −b` |
| `b2_share` | `(x·u)²/‖x‖²` | the alignment it **brings in**: at θ = 90°, `α'` *is* `−b` |

`b2_share` is the column that tells one target from another. On FLUX's width a random
orthogonal `u` has `b²/‖x‖² ≈ (1 − cos²)/3072 ≈ 1.3e-5`, so rotating by 90° takes
`cos(x, v*)` from +0.98 to about **−0.004** — the alignment is removed almost entirely, and
replaced by essentially nothing. `residual_pc1` removes the same alignment and substitutes
the population's own largest residual axis. **Both targets remove `v*`; they differ in what
takes its place**, which is a cleaner statement of the comparison than the one this arm
originally made.

`_plane_shares` asserts `share ≥ cos²` against a `cos²` computed independently — from a `v`
normalised in place and each token's own norm — and separately that the share does not
exceed 1. Both failures mean the same thing, that `(v, u)` is not orthonormal, and neither
is reachable by algebra alone: an assertion comparing the share to its own first term would
hold whatever `v` and `u` were, which is the vacuous check the first version shipped.

### What is recorded

Per (target, angle) in `rotation.csv`: the realised angle from `v*` before and after, the
cosine with `v*` and with `u`, the token norm before and after, `x·v*`, `x·u`,
`in_plane_energy_share`, `perturbation_l2`, the on-manifold score with its discriminative
flag, sink strength and sink counts from the register zone, LPIPS/RMSE, and the existing
Sobel / high-frequency / concentration metrics — joined on `(rotation_target, theta_deg)`,
which every table now carries.

`rotation_plane.csv` holds each treated token's **real** `(α, b)` coordinates before and
after, plus the out-of-plane length, which a plane rotation cannot change and which cell
11.5 checks. `fig_q13_rotation_plane` draws from it: one path per token, running **along**
its own norm circle rather than across it, because a straight arrow between the endpoints
would read as a translation — the one thing this operator never does.

### First measurement

One prompt on the synthetic checkpoint, at `writer+1_postmlp`. The operator holds: norm
drift 9.6e-8, `|u·v*|` 9.3e-10, θ = 0 perturbation 1.7e-7. Rotating toward `residual_pc1`
(`b² = 0.956` there) took `cos(x, v*)` from −0.043 to ±0.978 at fixed norm, and **sink
strength tracked it monotonically**: 0.908 at `cos = −0.978`, 1.056 unrotated, 1.234 at
`cos = +0.978`. `random_orthogonal` (`b² = 0.0015`) moved the alignment by 0.08 in total,
because on that token both `α` and `b` are small and the rotation exchanged two small
coordinates.

On-manifold energy fell to 0.86 at ±90°, which is exactly the caveat the third panel exists
to surface: a rotation that suppresses sinkhood by leaving the clean subspace has shown that
an unreachable state is not a sink, not that direction carries the mechanism. At ±90° the
perturbation L2 is 9.74 against a token norm of ~5, so these are large edits by every
measure except norm — `perturbation_l2` is on the table beside the effect for that reason.

**`residual_pc1` is not the application result.** It is the population's largest residual
axis: a data-defined direction with no interpretation attached, useful as a demonstration
that the operator can move the state and that sinkhood follows. The semantic question —
whether rotating toward a *meaningful* subspace raises semantic payload while reducing
sinkhood and preserving image quality — needs a genuine semantic direction and a
semantic-information metric, neither of which this arm supplies. `semantic_direction`
refuses until one is provided, and no run should be read as a semantic result until it is.

**The β grid is unchanged.** The rotation arm is additive: `run_rotation` defaults to False,
it has its own root, its own conditions and its own reference cell at θ = 0, and the main
5×5 grid, the rescue arm, the frozen `v*` and the token selection are untouched.
