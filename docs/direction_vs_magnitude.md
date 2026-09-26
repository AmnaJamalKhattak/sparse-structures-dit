# Direction vs magnitude

For a token state `x` and unit `v = v*/||v*||`, decompose `x = r + alpha v` with
`alpha = x . v`.

## Control surface (`ditsinks/control_surface.py`)

- **beta** (alignment multiplier): `y(beta) = r + beta * alpha * v`, then renormalised to
  `||x||`. Changes `cos(x, v)` monotonically while preserving the token's norm exactly.
- **gamma** (magnitude multiplier): `x' = gamma * y~(beta)`. Scales the whole vector
  without changing the direction `beta` produced.
- Grid: `beta` in `{0, 0.5, 1, 1.5, 2}` against `gamma` in `{0.5, 0.75, 1, 1.25, 1.5}`,
  25 cells. The centre cell `(1, 1)` is run rather than skipped: it reconstructs `x` to
  about 1e-7 relative in float32 and is the numerical floor every other cell is read
  against.
- Degenerate case: if a token is collinear with `v` and `beta = 0`, there is no residual
  to point along. Below 1% of the original norm the direction is replaced by a seeded
  random direction orthogonal to `v` (mirrors `causal_ops.remove_direction`), and the
  count of affected tokens is reported per condition.

## Rescue arm (mediation)

Five conditions: `clean` (reference), `channel_ablate` (the dominant channel zeroed),
`channel_ablate_vstar_rescue` (`x . v` set back to the clean `alpha`),
`channel_ablate_orthogonal_rescue` (control: the same per-token L2, applied orthogonal to
`v`), `channel_restore_only` (only the dominant coordinate restored).

Ablation and rescue are composed inside one edit function, ablation always first, so the
result does not depend on hook firing order. The rescue target is read from the paired
clean trajectory at the same (step, layer), never from the ablated run.

`gap_closed = (m_rescue - m_ablated) / (m_clean - m_ablated)`: 1.0 means the damage was
fully undone, 0.0 means none of it was, independent of how much damage the ablation did.

## Treatment population (`select_treatment`)

Frozen from the clean pass only; no mode reads a treated run. Seven modes:
`highnorm_and_aligned` (primary: the repository's norm threshold AND its alignment bar),
`register` (the repository's percentile high-norm rule), `topk_norm`, `topk_projection`,
`percentile_projection`, `sink_only` (clean per-head argmax image keys), `random_tokens`
(count-matched ordinary tokens). `highnorm_and_aligned` uses `highnorm_ratio x median`
for the norm bar and the alignment bar `select_frozen_targets` already derives; neither
is a new definition. An empty intersection falls back to the frozen register set and
records `FALLBACK` in the saved selection rule.

## Sinkhood

Uses the repository's existing criterion: `SweepConfig.sink_ratio_threshold` defines a
sink as absorbing at least 10x the uniform share of image-to-image attention mass.
`sink_readout` applies the same rule per token. `n_sink_heads` counts how many heads take
the token as their strongest image key.

## Image-level measurements

RMSE, PSNR and LPIPS come from `q11._image_fidelity` (LPIPS degrades gracefully if not
installed); CLIP is optional and off by default. `ditsinks/image_detail.py` adds three
readouts, each a mean of squares over an explicit mask, computed in numpy:

- Fourier band split into low/mid/high thirds of the Nyquist radius, each reported
  beside its own area share (`spectrum_<band>_over_uniform`, 1.0 for an unstructured
  difference).
- Sobel-magnitude change (`gradient_energy_ratio`; below 1 means less edge energy in the
  treated image).
- High-gradient concentration: the share of difference energy landing on the top 20% of
  the *clean* image's Sobel magnitude, divided by that mask's area. The mask is chosen by
  rank rather than by value, so a large flat region cannot pin the threshold at zero.

## Controls

`random_direction_beta*` and `orthogonal_beta*` (L2-matched, different directions),
`random_tokens_beta*` (same edit, ordinary tokens instead of the treatment population),
`sink_only_beta*` (same edit, clean sinks only). Energy matching is computed on the live
tensor at each layer against what the v* edit would do to that same tensor, not against a
target precomputed from the clean pass, since a windowed edit has already diverged from
clean by the second edited layer.

## Placebo gate

Layers upstream of the edit cannot be reached by any condition, so their readouts must be
identical across conditions. The spread is computed and reported on every run.

## Resumability

Completed cells are cached under `<root>/cells` as one JSON plus three CSVs, keyed by
`scope|condition|prompt|seed`. An `experiment_fingerprint` hashes everything that can
change a cell's numbers (configuration, `v*`, and the source text of
`control_surface.py`, `endpoints.py` and `causal_ops.py`), excluding paths and image
saving. A cell computed under a different fingerprint is discarded and recomputed.

## Rotation arm (`plane_basis`, `rotate_plane`)

A rotation rather than a rescale: `beta` always keeps the edited direction in
`span{v*, r}`, so it cannot move a token toward a chosen direction. Pick a unit `u`
orthogonal to `v*` from one of `random_orthogonal` (seeded random), `residual_pc1`
(leading principal direction of the clean tokens' residuals after `v*` is removed),
`ordinary_mean` or `ordinary_pc1` (fitted on tokens that are neither register nor
high-norm), or a supplied `semantic_direction` (refuses without one). Split
`x = alpha*v + b*u + q` and rotate the `(v, u)` plane by `theta`:

```
alpha' = alpha cos(theta) - b sin(theta)
b'     = alpha sin(theta) + b cos(theta)
x'     = alpha' v + b' u + q
```

`q` is untouched and `(alpha, b) -> (alpha', b')` is an orthogonal map of R^2, so the norm
is preserved exactly with no renormalisation step. `rotate_plane` raises if the measured
norm drift exceeds 1e-4 relative rather than correcting it, since a drift means `u` was
not unit or not orthogonal to `v`.

`in_plane_energy_share = (alpha^2 + b^2) / ||x||^2` is split and reported as
`alpha2_share = cos^2(x, v*)` (the alignment the rotation can remove) and `b2_share`
(the alignment it can bring in), each checked against an independently computed
`cos^2(x, v*)`.

`clean_subspace` / `onmanifold_energy` measure whether an edited state lies in the affine
subspace the clean population spans at that layer (the leading right singular vectors of
the mean-centred clean slice), reported against the clean population's own share of that
subspace. The diagnostic is only discriminative where the subspace is a genuine
restriction (`rank / width < 0.5`).

## CFG diagnostic

A few strong settings are run on both batch rows to measure how much the guidance
sampler's combination (`u + s(t - u)`) scales an edit confined to the conditional row
versus one applied to both. Skipped, with a note, on a pipeline that does not batch CFG
(for example FLUX at batch 1), where the two choices are the same tensor operation.

## How to run

`notebooks/direction_vs_magnitude.ipynb`. Configuration is `ControlSurfaceConfig` in
`ditsinks/control_surface.py`.

Files: `ditsinks/control_surface.py` (operators, conditions, selection modes,
measurement tables, resumable runner, verdict text), `ditsinks/image_detail.py` (local
detail readouts), `tests/test_control_surface.py`.
