# ICLR causal pipeline repair and validation status

## Scope and stop condition

This change repairs the causal pipeline and adds numerical acceptance gates. It
does **not** claim a new mechanistic result. Existing artifacts under
`iclr results/figures/` are untouched. Corrected runs must use
`QuestionContext.create_run_directory`, which creates a new, non-overwriting
`results/iclr_causal/<run_id>/` tree containing configuration, provenance,
diagnostics, per-question tables, and figures.

The current execution environment has no PyTorch/NumPy/pandas/matplotlib and
network package installation is blocked. Therefore real and synthetic inference
could not be run here. The success gate remains closed: no Q1--Q6 scientific
interpretation or corrected figure is certified by this commit.

## Files and repairs

| Area | Repair |
|---|---|
| Provenance | Added a refusal-to-overwrite run layout, `config.json`, `provenance.json`, git commit, Python/package versions, checkpoint, prompts/seeds, layers, channels and `v*` metadata. |
| Hook diagnostics | `Trace` now retains lossless `x_pre_hook`, `x_post_hook`, and `x_next_module_input` tensors at the recorded step. Q/K and reconstructed attention probabilities are retained at requested key layers. |
| Acceptance gates | Added fail-loud checks for norm-clamp direction/norm, direction removal, channel zero, tensor identity, and explicit text/image/grid token indexing. Q1/Q2/Q3/Q5/Q6 invoke the relevant checks. |
| NeurIPS compatibility | Added `run_neurips_compatibility`: live high-norm selection at every denoising step, old random-direction replacement at unchanged norm, median norm clamp, same-layer top-1 endpoint, affected/all-head macro, affected-head pooled, and affected count. |
| Q1 | Saves scalar hook diagnostics, explicit temporal labels, per-head affected flags, and separate affected/all-head macro and pooled summaries. |
| Q2 | Numerically asserts gamma-zero at the hook, includes current writer output in the measurement horizon, saves reconstruction diagnostics, temporal labels, and separate retention summaries. |
| Q3 | Renamed original repeated edit to `fixed_carrier_repeated`; added `dynamic_register_state`, which reselects current carriers under preregistered clean bars and asserts removal; added token-level pre/post/clean/treated histories and a strict `newly_regenerated` flag. Rescue remains separate. |
| Q5 | Replaced the mismatched clean reference with exact-recipient clean capture plus a separately labelled any-register context rate; emits same-operation, +1/+2/+3 and end-zone rows; runs COPY and MOVE; records recipient/source capture, key similarity, rank and inserted cosine; exact final-key and full-residual self-patches are identity-gated. |
| Structure maps | Exports distinct `survival_immediate` and `survival_propagated` tables and records numerator, denominator, target population, endpoint and aggregation definitions. |
| Q6 | Reports mean/mean-absolute/RMS projection and literal mean/RMS per-token perpendicular norm; adds cosine, projection, high-norm and sink lifetimes; keeps the old edit as `pure_vstar_replacement`; adds an orthogonal-component-preserving refresh; adds early matched-energy, unspecific-channel and norm-preserving competitor controls; asserts late suppression and refresh invariants. |

## Validation ledger

| Gate | Static/unit implementation | Runtime result in this checkout |
|---|---|---|
| NeurIPS direction-vs-magnitude reproduction | Compatibility runner implemented | **NOT RUN — STOP** |
| Sham residual/Q/K/probability/sink/image identity | Generic identity gate and captured Q/K/probabilities implemented | **NOT RUN — STOP** |
| Norm clamp | Fail-loud cosine and norm checks wired into Q1 | **NOT RUN** |
| Direction removal | Fail-loud projection check wired into Q1 and dynamic Q3 | **NOT RUN** |
| Channel zero | Fail-loud exact coordinate check wired into Q2/Q6 | **NOT RUN** |
| Final-key self-patch | Same-token K and probability identity gate wired into Q5 | **NOT RUN — Q5 STOP** |
| Full-residual self-patch | Downstream-key identity gate wired into Q5 | **NOT RUN — Q5 STOP** |
| Token/head indexing | Explicit mapping helper; existing two-layout head tests plus MOVE test | **NOT RUN** |
| Q3 paired-clean novelty | Token history and strict boolean implemented | **NOT RUN** |
| Q6 magnitude controls | Three early controls and separate pure/context-preserving refreshes implemented | **NOT RUN** |

## Required execution order

1. Run the unit suite with scientific dependencies installed.
2. Create a unique run with `ctx.create_run_directory(<run_id>, root=<repo-or-drive>)`.
3. Run `run_neurips_compatibility` for one prompt/seed. Stop unless direction
   retention is much lower than norm-clamp retention.
4. Run the one-prompt/one-seed Q1/Q2/Q3/Q5/Q6 diagnostic subset with resume
   disabled. Any assertion failure is a hard stop.
5. Inspect CSV diagnostics, not figures. Then run one prompt across 2--3 seeds,
   followed by two prompts across two seeds.
6. Only after all identity/manipulation gates pass may the full confirmatory run
   and corrected figures be generated.

## Artifact disposition

All PDFs currently under `iclr results/figures/` remain historical and are
invalid for corrected causal claims. No corrected figure is marked trustworthy
yet because no post-repair inference run was possible. In particular, the old
Q3 “regeneration,” Q5 sufficiency ladder, Q6 dissolution panel, mixed-time
structure summary, and Q1/Q2 headline comparisons must not be reused as evidence.

## Scientific result

There is deliberately no revised causal conclusion. Whether any contradiction
with the NeurIPS direction-versus-magnitude mechanism remains is **unknown**
until compatibility mode and all identity/manipulation gates pass on actual
model tensors.

