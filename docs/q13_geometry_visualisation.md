# The geometry module — real activations, honest projections

## What this is

A 3D picture of the register population, built from the measured `[N, C]` image-token
activations. Every point is a genuine inner product of a real token state with a real basis
vector. Nothing here is schematic, and no figure contains illustrative coordinates.

## The interpretable basis

A residual stream is thousands of dimensions wide, so any 3D view is a projection, and the
choice of projection is where such a figure is honest or not. Generic PCA answers *where is
the variance*, which is a fact about the data but not about the mechanism.

```
e1 = v*/||v*||
e2 = (e_c* - (e_c* . e1) e1) / ||.||     # what channel c* carries BEYOND v*
e3 = first principal direction of  x - (x.e1)e1 - (x.e2)e2
```

| Axis | Label on the figure | Property |
|---|---|---|
| 1 | Projection onto `v*` | the coordinate **is** `alpha_i = x_i . v*`, asserted in the tests |
| 2 | Dominant-channel-related axis | orthogonal to `v*`, so axes 1 and 2 never double-count |
| 3 | Residual variation axis | fitted on the anchor layer's own tokens, sign-pinned |

The basis is orthonormal, so a plotted coordinate is an honest length and the three squared
coordinates are a genuine share of the token's squared norm.

## The number that keeps it honest

**`captured_fraction`** — the share of a token's squared norm the three axes retain — is
printed on every figure, per category. Three axes out of three thousand can be a faithful
summary or a shadow, and only the data decides which. On the synthetic checkpoint the split
came out at 17% for ordinary tokens and **86% for the sink token**: the projection is a
faithful account of the population the figure is about, and a thin slice of everything else.
That is the honest framing, and it is only available because the number is computed.

Because the basis is orthonormal, a low fraction has a precise consequence: **distances in
the picture are lower bounds on distances in the full space.** Tokens that look far apart
*are* far apart; tokens that look close may not be.

## Two degeneracies, both announced

**`v*` is nearly the dominant channel.** Then `e_c*` is almost parallel to axis 1 and almost
nothing survives the orthogonalisation. The basis reports `channel_alignment = |v*_c*|` and
`independent_channel_content = sqrt(1 - |v*_c*|^2)`, and the figure prints a warning, so a
flat axis-2 spread reads as *no room on that axis* rather than *the channel does not matter*.
Above 0.9 alignment the note appears; at full collapse the basis substitutes a data-driven
direction, relabels axis 2, sets `degenerate_axis_2`, and says so in `notes`.

**The token's residual does not fit in three fixed axes.** This is why the decomposition
figure uses a *different* basis from the cloud — see below.

## The five figures

| Figure | What it answers | Basis |
|---|---|---|
| `fig_token_cloud` | **Main figure.** How the population sits relative to `v*`, and whether sinks are the tip of the high-norm spur or scattered through it | interpretable, shared |
| `fig_token_decomposition` | `x_i = alpha_i v* + r_i` for one token of each kind | **exact per-token** |
| `fig_channel_decomposition` | Is the token large because of channel `c*`, or because of a distributed direction? | **none — raw activations** |
| `fig_lifecycle_strip` | How the geometry evolves across depth or denoising time | interpretable, shared and fixed |
| `fig_basis_diagnostics` | What the 3D views are entitled to claim | — |

Plus `write_lifecycle_animation`, which writes per-frame PNGs **unconditionally** and then
attempts a GIF. Animation writers are the most environment-dependent part of matplotlib, and
a fragile GIF must not be able to cost a run its figures.

### Why the decomposition uses a different basis

In a shared 3D basis most of an individual token's residual falls outside the three
directions, so the drawn `r_i` leg would be a fraction of the real one — a figure whose
entire subject is the split between `alpha_i v*` and `r_i` would understate the second term.

The decomposition therefore keeps axes 1 and 2 and takes axis 3 to be *whatever is left of
that token*:

```
e3 = (x - (x.e1)e1 - (x.e2)e2) / ||.||
```

The token then lies in the drawn subspace **exactly** — `captured_fraction` is 1.0 by
construction, verified in `test_the_per_token_basis_contains_its_token_exactly` — so the
purple leg has length `|alpha_i|`, the blue leg has length `||r_i||`, and the right angle is
real. The cost, stated on the figure, is that axis 3 differs between panels, so residual
directions must not be compared across them. That is the correct trade for a figure about
the split *within* a token.

## What is reused

Nothing about the project's definitions is re-invented for a picture:

- **Activations**: `CausalTracer(full_state_layers=...)` plus `StateProbe`, unchanged.
- **The consumed state**: `q11.consumed_states`, which resolves the post-edit tensor. This
  matters — see below.
- **High-norm**: `select_frozen_targets`' `norm_threshold` (= `highnorm_ratio x median`).
- **Sinkhood**: `control_surface.sink_readout`, which applies
  `SweepConfig.sink_ratio_threshold` per token.
- **`v*`, the dominant channel, the layer window**: the frozen discovery artifact via
  `QuestionContext`.
- **Palette and theme**: `style.py`, so a token category keeps one identity project-wide.

### The bug this uncovered

`StateProbe` registers *before* `EditInstaller`, so `trace.probe` at the edit layer returns
the tensor as it was **entering** the hook. A clean-versus-treated geometry figure read off
that would have shown the edited layer as unchanged and the effect as appearing one layer
late — a plausible picture that is wrong. Snapshots now resolve the consumed state through
`q11.consumed_states`, and the fix is pinned by
`test_a_snapshot_reads_the_state_the_block_consumed_not_the_pre_edit_one`, which asserts all
three properties at once: nothing upstream of the edit moves, the ablation is visible at the
layer it happens, and the consequence propagates downstream.

## Styling decisions, and why

- **Panes and the wire cage removed.** Matplotlib's 3D default reads as a box the data sits
  inside. The data does not sit inside anything — it is a projection.
- **Arrows start at the origin, not the centroid.** `v*` is a direction through the origin;
  drawing it from the cloud's mean would be geometrically wrong. Limits are centred on the
  origin for the same reason: a cloud offset from the frame's centre is a real fact.
- **One shared scale on all three axes.** Unequal scales make an angle in the picture
  unrelated to the angle in the data. This is why a cloud can look small — one extreme token
  sets the range, and on a register population that token *is* the subject.
- **Nothing silently vanishes.** Matplotlib drops points outside the axis limits, which on a
  figure about the extreme tail would delete the evidence. Out-of-range tokens are pulled to
  the boundary, drawn as hollow diamonds, and counted on the figure.
- **Ordinary tokens are a faint grey haze; the population of interest is saturated and
  opaque.** The backdrop should not compete with the subject.

## Configurability

Model, prompt id, seed, timestep, layers, basis kind, dominant channel, token-selection
thresholds, extreme percentile, camera angles, and every output path. `GEOMETRY_COMPARE` in
the notebook draws a treated run beside the clean one **on the same basis** — a comparison
between two bases would be a comparison between two cameras.

## Structured metadata

`save_metadata` writes one JSON per figure with the model, prompt id, prompt text, seed,
timestep, layer, condition, dominant channel, intervention point, basis kind and all its
diagnostics, per-category token ids, category counts, representative token ids, both
thresholds, the captured fraction overall and per category, and an explicit
`projection_note`. `token_table` exports every drawn point's real coordinates so a reader can
re-plot the figure without the model, and `basis_table` exports the diagnostics per layer.

## Caveats in interpreting the 3D geometry

1. **Low `captured_fraction` means the picture is a shadow.** Distances are lower bounds.
2. **Axis 3 is fitted and meaningless in sign.** Do not read "positive on axis 3" as
   anything. It is reused across a sequence precisely so frames stay comparable.
3. **Axis 2 can be nearly empty.** Check `independent_channel_content` before reading it.
4. **The clouds are one prompt and one seed.** An existence picture, not a population
   statistic. The quantitative claims live in the control-surface tables.
5. **The decomposition panels use per-token bases.** Compare the split within a token, never
   residual directions between tokens.
6. **A treated cloud shows the consumed state.** At the edit layer that is post-edit by
   design; upstream layers are the placebo check and must be identical.

## Files

- `ditsinks/geometry.py` — bases, projection, categorisation, capture, metadata, tables
- `ditsinks/geometry_figures.py` — the five figures and the animation writer
- `notebooks/iclr_q13_direction_vs_magnitude.ipynb` § 9
- `tests/test_geometry.py`
