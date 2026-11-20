# Bundled measurements

Seeds 0, 1, 2 and 3 cover `gemma`, `qwen`, `llama`, `qwen7` and `qwen32b` on
TruthfulQA, HaluEval, ParaDetox and RTP. The 32B comparison has dense and `ours`;
the other four models have all eight steered comparison methods and both norm
variants. Coordinate, stability, capability and proportion results cover the
four smaller models on the first three tasks.

The layout is the runner's `$OUTPUT_DIR` layout:
`seed<k>/<model>/<task>/`. Scores are percentages; NLL ratios, repetition,
coordinate overlaps and coefficient masses are fractions. Capability summaries
and paired intervals use accuracy fractions or nats per token.

| File | Measurements |
|---|---|
| `test.json` | Unsteered reference, strength grids and best feasible selections |
| `mask.json` | Selected coordinates and available rerun/selection diagnostics |
| `selectors.json` | Available comparison-selector supports and statistics |
| `coordinate_statistics.json` | Selected coefficient mass and coordinate overlap |
| `stability.json` | Rerun pair overlaps and per-step overlap with the final mask |
| `capability.json` | Capability summaries and paired bootstrap intervals |
| `proportion/rho10.json`, `rho40.json`, `rho80.json` | Proportion grids and selections (20% is `test.json`) |
| `proportion/mask-rho10.json`, `mask-rho40.json`, `mask-rho80.json` | Available proportion-mask coordinates |

Missing fields are omitted. Archived points do not uniformly contain
`strength_index`, `alpha`, `alpha_status`, `alpha_source_method`,
`additive_multiplier_mean`, `completion_nll`, `questions`, `coordinates` or
`status`; an unavailable point can have null measurements. Seed-0 proportion
masks and selector supports, and the seed-0 32B mask, are absent. Seed-0 masks
lack final-epoch rerun masks and selection metrics; some later masks lack
`step_mask_count` or training seeds. Some stability files lack rerun coordinates,
epochs or definition strings; seed-0 trajectories also include the recorded
step 0. Capability records do not uniformly contain full summaries, configs,
selected-point diagnostics or bootstrap seed/resample metadata. Seed-0
capability `selected` strengths identify the archived evaluated endpoints,
which can differ from the selection under the current 1.15 NLL limit.
Feasibility and grid selections use the current limits. Nominal amplitudes are
not substituted for measured `alpha`. Fitted direction vectors, per-item data,
execution metadata, paths, hashes and timing are excluded.

Method keys:

| Key | Method |
|---|---|
| `ours` | Optimized sparsification with constrained normalization |
| `sparsification_only` | Optimized sparsification without normalization |
| `sparsification_global_norm` | Optimized sparsification with global normalization |
| `dense` | Dense baseline (CAA), the mean-difference direction |
| `normalized_dense` | Dense baseline with global normalization |
| `pca_dense`, `pca_sparse` | PCA direction, dense and sparse |
| `magnitude`, `gradient`, `ie` | Coordinates selected by magnitude, gradient and IE |
| `unsteered` | The model without steering |

## Rebuilding the paper's tables and figures

These commands run on CPU and download nothing. Rebuild the four-seed comparison, coordinate and
capability tables (`comparison.csv`, `coordinates.csv`, `capability.csv`):

```bash
python -m reproduce.average --results results --output outputs/tables
```

Redraw the paper's seven data figures. Figures 2 and 3 are diagrams and use no data. Matplotlib
3.10.9 reproduces the paper's rendering.

```bash
python -m reproduce.paper_figures --results results --output outputs/figures
```

Rebuild one seed's tables and figures (repeat for seeds 1, 2 and 3):

```bash
python -m reproduce.summarize --results results --output outputs/seed0 --seed 0
```

Averaging selects the best feasible score independently in each seed, then
averages those scores. Coordinate tables use percentages. Capability Mean is
the mean over seeds of the per-seed mean absolute change; minimum and maximum
are signed extrema over all model, task and seed measurements. Accuracy changes
are percentage points and WikiText changes are nats per token. The unsteered
capability row reports the mean baseline measurement.
