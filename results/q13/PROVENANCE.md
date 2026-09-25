# Q13 result tables behind Figure 4

Both tables come from `notebooks/iclr_q13_direction_vs_magnitude.ipynb`, run on FLUX.1-schnell
(1024 x 1024, denoising step 2 of 4, five DiffusionDB prompts at seed 0). They are the only inputs
of `scripts/make_fig4_direction_at_fixed_norm.py`.

| File | Run output it was taken from | What it is |
|---|---|---|
| `q13_depth_response_flux1-schnell.csv` | `confirmatory/flux1-schnell/V1_q13_depth/metrics/depth_response.csv` | Alignment sweep: the treated tokens' $v^\star$ component is scaled by $\beta$ and each token is rescaled to its own norm. Values are read at the edited layer (18, 20 or 24) and averaged over the 5 prompts. `sink_strength` is the largest share of image attention that any head gives a treated token, times the number of image tokens. |
| `q13_rotation_retention_flux1-schnell.csv` | reduced from `confirmatory/flux1-schnell/q13_rotation/metrics/population_metrics.csv` by `ditsinks.paper_figures.rotation_retention_table` | Rotation: the register state at layer 18 is rotated by $\theta$ in the plane of $v^\star$ and a target direction, with its norm kept exact. One row per target, angle and prompt; `sink_retention` is the share of affected heads that keep their unmodified sink, averaged over layers 20 to 39. |
