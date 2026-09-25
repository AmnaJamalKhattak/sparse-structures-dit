# Q16 — every parameter and its basis

This is the reviewer-facing record of every value the Q16 main run uses
(`notebooks/iclr_q16_main.ipynb`, `ditsinks/q16_main.py`). Each value has one of three
kinds of basis:

1. **Measured** from the unmodified generation of the same prompt and seed. There is
   nothing to tune.
2. **Shared definitions** used by every table of the paper, defined once in
   `ditsinks.config.SweepConfig` and inherited by the atlas, the workshop paper and Q1–Q6.
3. **Conventions with a stated rationale.** Each is either shown not to change the result
   (the range check below), or is a design constraint whose purpose is stated.

Everything in this file is frozen in `protocol.json` before any conditioned image is
generated (stage 1 of the main run). Nothing here is chosen after seeing a result.

## 1. Measured from the unmodified run (no free parameter)

| Quantity | Rule | Notes |
|---|---|---|
| Register population at a block | Tokens passing the **register test** at the block's output: norm ≥ 3 × the block's median token norm, and cos(x, v\*) ≥ the 99.9th percentile of that block's ordinary tokens | This is Q1's criterion. "Ordinary" means below the norm bar and not a carrier. |
| Formation block *F* | First block whose output holds half of the peak register population (half-maximum) | FLUX.1-dev pilot: block 18. The carriers' v\* projection jumps from about 7% to 89% of its peak between blocks 17 and 18, so any fraction from about 0.08 to 0.89 names the same block. |
| Carriers at step *t* | Every token passing the register test at the selection block, at step *t* | These are per step. There is no single anchor step (§3). |
| Natural end *E* | The block before the first post-peak block at which **no carrier** passes the register test | This replaces "the carriers' v\* projection fell below 10% of its peak". On FLUX.1-dev that rule never fires: after dissolution the former carriers keep cos(v\*) near 0.5, so their projection bottoms out near 14% of its peak. The pilot then silently fell back to a range declared from earlier FLUX.1-schnell work. |
| Decline block *D* | First block after the peak at which the carriers' mean v\* projection falls below 90% of its maximum | Used only to start the extension before the decline and to end the in-place window within the plateau. |
| Matched targets | For each carrier and each (step, block): ρ = ‖x‖ / median and π = (x·v\*) / median, averaged over the natural plateau at the same step, then multiplied by the recipient block's median | The induced state therefore has the natural register's own size, relative to each block's scale. The rest of the token is kept from the recipient; this is stated as a limitation. |
| Removal bar | The 99.9th percentile of cos(x, v\*) over the unmodified run's ordinary tokens, at the same (step, block) | The removal is dynamic: it applies to every image token at every hook. |
| Model-level boundaries (main run) | *F* = the **earliest** and *E* = the **latest** boundary over the calibration prompts and the edited steps. Selection block = the most frequent peak block. Decline = the median. | These are conservative: no early window reaches a formed register, and no late window starts while one is present, in any calibration generation. The per-unit boundaries are recorded (`boundaries.csv`, appendix figure). |

## 2. Shared definitions

| Parameter | Value | Basis | Range check (on every unit) |
|---|---|---|---|
| `highnorm_ratio` | 3.0 × median | Registers sit at 6–14 × the median token norm on FLUX.1-dev. At block 29 the 99th percentile of all tokens is 1.2 × the median, and at early blocks even the largest token is under 1.5 ×. 3 lies in the empty gap. | Carrier count at the peak block, overlap with the chosen set, formation and natural end, for ratios 1.5–8 |
| `sink_threshold` | 10 × the uniform share 1/N | A sink is defined relative to 1/N, so the definition means the same at 512 px and 1024 px. Carrier tokens receive 2–3 × uniform before they become registers; the natural register receives 60–77 ×. | Sink tokens per block, and the share of sinks that are carriers, for thresholds 2–40 |
| `alignment_quantile` | 0.999 | Q1's alignment bar. By construction it also removes about 0.1% of ordinary tokens (about 4 of 4,096 per block at 1024 px), a stated false-positive rate. | Largest share of the image stripped, and smallest share of carriers caught, for 0.99–0.9999 |
| `projection_pct` (pilot) | 99 | The ICLR plan's Q1 register definition: the top 1% of tokens by norm | — |

The range check (`LC.threshold_sensitivity`) moves one threshold at a time, holding the
other two at their chosen values, and re-measures what it decides on the **same**
unmodified run. It needs no new generation. It is reported as the appendix figure
`fig_threshold_range` and the table `appendix/threshold_range.csv`.

## 3. Design choices with a stated rationale

| Choice | Value | Rationale |
|---|---|---|
| Windows per side | 3 | Three depths before formation and three after the natural end: enough to show a trend within each region. |
| Window length *L* | The largest *L* that fits 3 windows on both sides (FLUX.1-dev: 5 blocks) | Every window holds the same number of hooked blocks, so depth is not confounded with dose. |
| Window placement | Flush against the natural interval, one free block between (E3 ends at *F* − 2; L1 starts at *E* + 2), stacked outward, starting no earlier than block 1 | The free block before formation is where the writer starts. The free block after the natural end carries the removal's terminal hook. No block carries both an induction and a removal hook. Block 0 reads the patch embedding. |
| In-place window N | *F* + 1 to *F* + *L*, ending before *D* | Rewrites the state where it already exists: the effect of the write itself, and the baseline every depth is compared against. |
| Removal interval | *F* − 1 to *E*, plus a terminal hook at *E* + 1 | It starts where the writer starts. The terminal hook protects the first block after the interval. |
| Extension | From *D* − 2 through the last block of L1 | The same late blocks are covered by a continued lifetime (extension) and a re-created one (L1). The two-block lead keeps the start on the plateau. |
| Edited denoising steps | The first third (FLUX.1-dev: steps 0–8 of 28; PixArt-Σ: 0–6 of 20) | This is the high-noise phase in which the global composition is set. On the pilot, edits at every step produced high-frequency artefacts: writing an out-of-distribution state into the low-noise steps damages texture, which is a known failure mode of activation editing, not evidence about register function. The phase is a declared factor (`EDIT_PHASE`), and a second phase can be run as an ablation. |
| Anchor step | None | At every edited step the positions and sizes are those of the natural register at that same step. One-step figures show the middle edited step. Attention is recorded at the middle and last edited steps and at the trajectory midpoint. |
| Random-direction control | A unit vector orthogonal to v\* (seed 0), with the same π and ρ | The same size and the same positions; only the direction differs. |
| Non-register-position control | Count-matched tokens that are never carriers at any edited step, below 2 × the median norm, and not more aligned than any ordinary token (seeded per unit and step) | The same state at positions that never carry the register. |
| In-distribution control | v\* projection set to 1.0 × the largest projection any token of the recipient block already holds | The strongest v\* write that stays inside the recipient block's own clean range. |
| `min_carriers` | 1 | The register test alone defines a register. PixArt-Σ's register is a single token, so any higher floor would exclude every PixArt-Σ unit. FLUX.1-dev's register has about 30 tokens, so the floor does not bind there. Each frozen `protocol.json` records the value its run used. A unit with no register at any edited step is a **declared exclusion**, counted and reported, never silently a no-op. |
| `rule_max_share_of_image` | 2% | On FLUX.1-dev the register is about 0.8% of the image (32 of 4,096 tokens), and 2% is 2.5 × that; on PixArt-Σ it is one token. A rule touching more would be removing a direction, not a register. It is audited on every unit (`rule_audit.csv`). |
| `edit_branches` | FLUX.1-dev: `conditional` (its only pass); PixArt-Σ: `both` | The induction has to change the model's state in every pass the sampler makes. FLUX.1-dev runs the model once per step (guidance is an input). PixArt-Σ runs it twice, with and without the prompt, and forms each update as uncond + 4.5 × (cond − uncond). Writing the state into one pass only would put an edited pass and an unedited one into that difference, multiplying the edit by 4.5. Both passes are therefore edited, each with its own remainder, at the register positions and sizes of the conditional pass. On every PixArt-Σ unit, `branch_overlap.csv` checks that the unconditional pass holds its register at those positions. |

## 4. Sampling, prompts, seeds, statistics

| Item | Value | Basis |
|---|---|---|
| FLUX.1-dev | 1024 × 1024, 28 steps, guidance 3.5, bf16 | The reference pipeline defaults at the native resolution |
| PixArt-Σ-XL-2-1024-MS | 1024 × 1024, 20 steps, CFG 4.5, bf16 | The reference pipeline defaults at the native resolution |
| v\* | Fitted on 24 DiffusionDB prompts (seed-0 draw) at the trajectory midpoint | The frozen discovery artifact, per checkpoint |
| Prompts | DiffusionDB: a seeded (seed 1), diversity-filtered draw. Exact and near duplicates of the discovery prompts are removed (content-word overlap ≥ 0.5). 4 calibration prompts plus the evaluation prompts, frozen to `prompts.json`. | Real user prompts, held out from the direction's fit. The same prompts are used for both models. |
| Seeds | 0, 42, 1234, 777, 3407 | Fixed in advance and reported. Every condition of a prompt–seed pair shares the initial noise. |
| Unit and cluster | Unit = (prompt, seed); cluster = prompt | A prompt's seeds share its layout and register positions. |
| Intervals | 95% percentile bootstrap, 2,000 resamples, resampling prompts with all their seeds | No interval is reported with fewer than 3 prompts. |
| Metrics | LPIPS (AlexNet) and PSNR against the unit's unmodified image; CLIP ViT-L/14 image–image similarity and image–text similarity change | Paired: each is a within-unit comparison |
| Pre-declared contrasts | Each depth window against the in-place rewrite; E3 against each control; each relocation against removal alone | `contrasts.csv`, section 6 of the notebook |
