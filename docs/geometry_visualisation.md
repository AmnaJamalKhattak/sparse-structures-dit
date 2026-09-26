# Geometry figures

## What this is

3D figures of the register population, built from the measured `[N, C]` image-token
activations. Every point is an inner product of a real token state with a real basis
vector; no figure uses illustrative coordinates.

## The interpretable basis

```
e1 = v*/||v*||
e2 = (e_c* - (e_c* . e1) e1) / ||.||     # what channel c* carries beyond v*
e3 = first principal direction of  x - (x.e1)e1 - (x.e2)e2
```

| Axis | Label on the figure | Property |
|---|---|---|
| 1 | Projection onto `v*` | the coordinate is `alpha_i = x_i . v*` |
| 2 | Dominant-channel-related axis | orthogonal to `v*`, so axes 1 and 2 never double-count |
| 3 | Residual variation axis | fitted on the anchor layer's own tokens, sign-pinned |

The basis is orthonormal, so a plotted coordinate is a real length and the three squared
coordinates sum to a share of the token's squared norm.

## `captured_fraction`

The share of a token's squared norm the three axes retain, printed on every figure per
category. Because the basis is orthonormal, distances in the picture are lower bounds on
distances in the full space: tokens that look far apart are far apart, tokens that look
close may not be.

## Two degeneracies

**`v*` is nearly the dominant channel.** Then `e_c*` is almost parallel to axis 1 and
almost nothing survives orthogonalisation. The basis reports `channel_alignment =
|v*_c*|` and `independent_channel_content = sqrt(1 - |v*_c*|^2)`, and the figure prints a
note above 0.9 alignment. At full collapse the basis substitutes a data-driven direction
for axis 2, sets `degenerate_axis_2`, and records the substitution in `notes`.

**A token's residual does not fit in three fixed axes.** This is why the per-token
decomposition figure uses a different basis from the population cloud (below).

## The five figures

| Figure | Draws | Basis |
|---|---|---|
| `fig_token_cloud` | The population relative to `v*`: sinks against ordinary and other high-norm tokens | interpretable, shared |
| `fig_token_decomposition` | `x_i = alpha_i v* + r_i` for one token of each category | exact per-token |
| `fig_channel_decomposition` | The token's per-channel activation against its `v*`-aligned component, channel `c*` marked | none, raw activations |
| `fig_lifecycle_strip` | The geometry across depth or denoising time | interpretable, shared and fixed |
| `fig_basis_diagnostics` | `captured_fraction` by category, `v*`'s alignment with the dominant channel, retained share by depth | none |

`write_lifecycle_animation` writes per-frame PNGs unconditionally, then attempts a GIF.

### Why the decomposition uses a different basis

In the shared 3D basis most of an individual token's residual falls outside the three
directions, so the drawn `r_i` leg would be a fraction of the real one. The decomposition
keeps axes 1 and 2 and takes axis 3 to be whatever is left of that token:

```
e3 = (x - (x.e1)e1 - (x.e2)e2) / ||.||
```

The token then lies in the drawn subspace exactly (`captured_fraction` is 1.0 by
construction), so the purple leg has length `|alpha_i|`, the blue leg has length
`||r_i||`, and the right angle is real. Axis 3 differs between panels, so residual
directions must not be compared across them.

## What is reused

- Activations: `CausalTracer(full_state_layers=...)` plus `StateProbe`.
- The consumed state: `q11.consumed_states`, which resolves the tensor a block actually
  received after any installed edit (`StateProbe` registers before `EditInstaller`, so a
  raw probe at the edit layer would return the pre-edit tensor).
- High-norm threshold: `select_frozen_targets`'s `norm_threshold`
  (`highnorm_ratio x median`).
- Sinkhood: `control_surface.sink_readout`, applying `SweepConfig.sink_ratio_threshold`
  per token.
- `v*`, the dominant channel and the layer window: the frozen discovery artifact via
  `QuestionContext`.
- Palette and theme: `style.py`, so a token category keeps one identity across figures.

## Styling

- Panes and the wire cage are removed; the data is a projection, not a box to sit inside.
- Arrows for `v*` and the channel axis start at the origin, not the cloud's centroid,
  since both are directions through the origin. Axis limits are centred on the origin for
  the same reason.
- All three axes share one scale, so an angle in the picture matches the angle in the
  data.
- Matplotlib drops points outside the axis limits by default; out-of-range tokens are
  instead pulled to the boundary, drawn as hollow diamonds, and counted on the figure.
- Ordinary tokens are drawn as a faint grey haze; the population of interest is
  saturated and opaque.

## Configurability

Model, prompt id, seed, timestep, layers, basis kind, dominant channel, token-selection
thresholds, extreme percentile, camera angles, and every output path. Comparing a treated
run against a clean one uses the same (clean) basis for both, since comparing across
different bases would compare two projections rather than two populations.

## Structured metadata

`save_metadata` writes one JSON per figure with the model, prompt id, prompt text, seed,
timestep, layer, condition, dominant channel, intervention point, basis kind and its
diagnostics, per-category token ids, category counts, representative token ids, both
thresholds, and the captured fraction overall and per category. `token_table` exports
every drawn point's coordinates; `basis_table` exports the basis diagnostics per layer.

## Reading the figures

1. A low `captured_fraction` means the picture is a partial view; distances are lower
   bounds only.
2. Axis 3's sign is arbitrary. It is reused across a sequence so frames stay comparable,
   not because the sign carries meaning.
3. Check `independent_channel_content` before reading axis 2: it can be nearly empty.
4. Population clouds are drawn from one prompt and one seed each.
5. Decomposition panels use a per-token basis: compare the split within a token, not
   residual directions between tokens.
6. A treated cloud shows the consumed (post-edit) state at the edit layer; upstream
   layers should be identical to the clean run.

## Files

- `ditsinks/geometry.py`: bases, projection, categorisation, capture, metadata, tables.
- `ditsinks/geometry_figures.py`: the five figures and the animation writer.
- `notebooks/direction_vs_magnitude.ipynb`
- `tests/test_geometry.py`
