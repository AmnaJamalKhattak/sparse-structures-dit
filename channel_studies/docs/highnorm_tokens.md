# High-norm token overlap

Checks whether massive-activation outlier tokens are the same tokens that Darcet et al.
(arXiv:2309.16588) call high-norm tokens or registers.

## What is computed

At one FLUX block (default layer 18), the top `base_k` channels by `mean(abs(activation))`
over tokens are the massive channels `C_k`. `m[n] = sum_{c in C_k} |X[n,c]|` is the massive
score; its top `outlier_frac` tokens are the outlier tokens, the speckles visible when only
those channels are kept.

Because `‖x[n,:]‖² = sum_d x[n,d]²`, a token with a massive value in one channel has its L2
norm dominated by that channel, so comparing the massive score directly against the full
norm `N_full` is circular. Every overlap statistic instead uses `N_ex`, the norm with the
massive channels excluded (`token_norms(X, exclude=C_k)` in `src/common/highnorm.py`).
`N_full` is also reported to show the size of that confound.

Two effect sizes drive the verdict, both medians over prompts:
- `selectivity` = median `m`[outlier] / median `m`[typical]: is the channel token-sparse?
- `elevation` = median `N_ex`[outlier] / median `N_ex`[typical]: do outlier tokens stay
  high-norm once the massive channels are excluded?

Set overlap (IoU, AUROC, Spearman) against a scale-matched random-channel null and a
token-permutation null is also computed and reported, but does not drive the verdict.
`ρ[n] = ‖X[n,C_k]‖² / ‖X[n,:]‖²` cannot separate H1 from H3, since both give `ρ` near 1.
IoU against the scale-matched null cannot detect H2, since a scale-matched channel
reproduces nearly the same overlap when outlier tokens are elevated across every channel.

A layer sweep (every `norm_profile_stride`-th block, plus the first, target, and final)
tracks the full-norm percentiles and bimodality coefficient across depth, and a
neighbor-cosine-similarity check (Darcet et al., Fig. 5a) tests whether outlier tokens are
redundant with their grid neighbors at the earliest captured block.

## Decision rule

`summarize` in `src/experiments/highnorm_tokens.py` applies this from the median
`selectivity` and `elevation` over prompts:

| Verdict | Signature |
|---|---|
| H1, one mechanism | selectivity >= `SELECTIVITY_MIN` (10), elevation < `ELEVATION_H2` (1.5) |
| H2, co-located but distinct | selectivity >= `SELECTIVITY_MIN`, elevation >= `ELEVATION_H2` |
| H3, premise fails | selectivity < `SELECTIVITY_MIN` |

Only NaN is inconclusive; +inf passes through the threshold comparisons as a valid
unbounded signal. Residual elevation alone does not establish a functional register role,
and channel exclusion here is a post-hoc decomposition, not a causal intervention.

## Deviations from Darcet et al.

Logged to `summary.json`, not silently picked.

- Neighbor redundancy uses the layer-0 block output as the early-representation proxy;
  FLUX has no ViT patch-embedding layer with the same semantics.
- The high-norm threshold is derived per run by a 1-D 2-means split on log-norms rather
  than the hard-coded 150, which is DINOv2-specific and stated in the paper to vary
  across models.
- Only the last denoising step is captured, inherited from the localization baseline's
  capture convention.

## Run

```bash
python -m src.experiments.highnorm_tokens --config configs/highnorm_tokens.yaml
```

## Outputs

- `per_prompt.csv`, `summary.json`
- `fig_variance_explained.png`, `fig_overlap.png`, `fig_norm_profile.png`,
  `fig_norm_hist.png`, and, when spatial panels or position-stability data are
  collected, `fig_spatial_panels.png` and `fig_position_stability.png`
