# Surpassing Dense Steering in LLMs through Optimized Sparsification and Constrained Normalization

Code and results for the paper. The method adds a steering vector on a learned subset of
residual-stream coordinates (optimized sparsification) and then rescales only those coordinates so
that their norm returns to its value before the edit (constrained normalization).

The repository supports two uses:

1. **Rebuild the paper's tables and figures** from the bundled results. CPU only, no model download.
2. **Rerun the experiments** on a GPU.

Both start from the same installation (Linux, Python 3.11):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 1. Rebuild the tables and figures

`results/` holds the measurements behind every number in the paper: five models, four tasks and
seeds 0 to 3. Its files are described in [results/README.md](results/README.md).

```bash
python -m reproduce.average --results results --output outputs/tables
python -m reproduce.paper_figures --results results --output outputs/figures
```

| Output | Paper item |
|---|---|
| `tables/comparison.csv` | Main comparison table |
| `tables/coordinates.csv` | Coordinate overlap table |
| `tables/capability.csv` | General capability table |
| `figures/main_result.pdf` | Figure 1 |
| `figures/selection_overlap.pdf` | Figure 4 |
| `figures/norm_strength.pdf` | Normalization and strength figure |
| `figures/proportion_sweeps.pdf` | Proportion figure |
| `figures/dense_margin.pdf` | Margin over the dense baseline |
| `figures/selected_coordinate_statistics.pdf`, `figures/mask_seed_overlap.pdf` | Appendix figures |

Values are means over the four seeds. The main comparison takes each seed's best feasible score
first and then averages. Figures 2 and 3 are diagrams and use no data. Matplotlib 3.10.9 reproduces
the paper's rendering.

## 2. Rerun the experiments

### Models

A CUDA GPU with native bf16 support is required.

| Model key | Hugging Face identifier | Steered layer | GPU memory |
|---|---|---:|---|
| `gemma` | `google/gemma-2-2b` | 18 | 24 GB |
| `qwen` | `Qwen/Qwen2.5-3B-Instruct` | 24 | 24 GB |
| `llama` | `NousResearch/Llama-2-7b-chat-hf` | 21 | 40 to 80 GB |
| `qwen7` | `Qwen/Qwen2.5-7B-Instruct` | 19 | 40 to 80 GB |
| `qwen32b` | `Qwen/Qwen2.5-32B-Instruct` | 43 | 80 GB for grids, 2 × 80 GB for fitting |

### Prepare

```bash
export SPARSE_STEERING_RUN_ROOT=/path/to/storage
source launch/environment.sh
python -m reproduce.download_models --model gemma
python -m reproduce.prepare
python -m reproduce.capability_panels
```

Model weights, datasets and fitted objects are stored below `SPARSE_STEERING_RUN_ROOT`. Results are
written to `$OUTPUT_DIR` in the same layout as `results/`. Log in to the Hugging Face Hub first if a
model is gated.

### Run one case

A case is one model, task and seed. Tasks are `truthfulqa`, `halueval`, `paradetox` and `rtp`.

```bash
# 1. Fit the direction, the comparison methods and the coordinate mask
python -m reproduce.fit --seed 0 --model gemma --task truthfulqa

# 2. Evaluate every method on the strength grid
python -m reproduce.grid --seed 0 --model gemma --task truthfulqa

# 3. Evaluate general capability at each method's best feasible setting
python -m reproduce.capability --seed 0 --model gemma --task truthfulqa
```

Repeat for every model, task and seed 0 to 3. Once all four seeds of a model are fitted, compute
its coordinate overlap and mask stability:

```bash
python -m reproduce.postprocess --model gemma
```

Then build the tables and figures as in section 1, with `--results "$OUTPUT_DIR"`.

- **RealToxicityPrompts.** `rtp` reuses the ParaDetox direction and mask of the same seed. Run only
  its `grid`, after the ParaDetox `fit`.
- **Proportion of coordinates.** Add `--fraction 0.1`, `0.4` or `0.8` to both `fit` and `grid`.
  The default is `0.2`.
- **One GPU type per case.** Run a case, including its unsteered reference, on a single GPU type.
  The text-quality limit is a ratio to that reference.
- **Resuming.** A repeated command skips the units that are already complete. Run each case in a
  single process.
- **Launchers.** `bash launch/run.sh fit ...` runs a command in the shared environment, and
  `launch/run.sbatch` is a Slurm template. Each command accepts `--help`.

## Code

| Path | Contents |
|---|---|
| `reproduce/` | The commands above |
| `sparse_steering/method/` | Mask training, sparse edit and constrained normalization |
| `sparse_steering/baselines/` | Dense and PCA steering; magnitude, gradient and IE coordinate selection |
| `sparse_steering/tasks/` | Answer-selection and detoxification tasks, data sources and splits |
| `sparse_steering/evaluation/` | Strength grid, best feasible selection, capability and confidence intervals |
| `sparse_steering/runtime/` | Model loading, hooks, storage and multi-GPU placement |
| `sparse_steering/analysis/` | Tables and figures |
| `protocol/` | Experiment settings (`experiment.json`) and the fixed splits of each seed |
| `configs/` | Dataset revisions, library versions and capability item identifiers |

`protocol/experiment.json` is the single definition of the experiments: model revisions, layers,
strength grid, training settings and text-quality limits.
[protocol/PROTOCOL.md](protocol/PROTOCOL.md) explains each part.

The paper's expressions are implemented in these files, relative to `sparse_steering/`:

| Paper expression | Implementation |
|---|---|
| Mean-difference direction and activation scale | `runtime/model_loading.py::fit_direction` |
| Exact-K mask and straight-through surrogate | `method/mask_training/masks.py` |
| Answer-selection loss | `method/mask_training/objectives.py::reference_loss` |
| Full-response loss | `method/mask_training/objectives.py::preference_loss` |
| Sparse edit, h_tilde = h + alpha * v_M | `method/sparse_edit.py::realize_batch` |
| Constrained normalization, beta * h_tilde | `method/constrained_normalization.py` |
| Global normalization | `method/global_normalization.py` |
| Best feasible score under the text-quality limits | `evaluation/text_quality.py` |

Result files and tables name the methods with these keys:

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

The best feasible setting of a method is the strength with the highest task score whose NLL ratio
is at most 1.15 and whose repetition is within the task's limit.

## Tests

```bash
python -m unittest discover
```

The 34 tests run on CPU and download nothing.
