# Individual commands

`run_experiment.sh` in the repository root runs the commands below for one case (one model, task
and seed). Use them directly to run only part of a case. Each command accepts `--help`.

A CUDA GPU with native bf16 support is required.

| Model key | Hugging Face identifier | Steered layer | GPU memory |
|---|---|---:|---|
| `gemma` | `google/gemma-2-2b` | 18 | 24 GB |
| `qwen` | `Qwen/Qwen2.5-3B-Instruct` | 24 | 24 GB |
| `llama` | `NousResearch/Llama-2-7b-chat-hf` | 21 | 40 to 80 GB |
| `qwen7` | `Qwen/Qwen2.5-7B-Instruct` | 19 | 40 to 80 GB |
| `qwen32b` | `Qwen/Qwen2.5-32B-Instruct` | 43 | 80 GB for grids, 2 × 80 GB for fitting |

## Prepare

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

export SPARSE_STEERING_RUN_ROOT=/path/to/storage
export HF_TOKEN=<token>             # only for gated models such as Gemma
source launch/environment.sh        # sets $OUTPUT_DIR and the cache locations
python -m reproduce.download_models --model gemma
python -m reproduce.prepare
python -m reproduce.capability_panels
```

Model weights, datasets and fitted objects are stored below `SPARSE_STEERING_RUN_ROOT`. Results are
written to `$OUTPUT_DIR` in the same layout as `results/`.

## Run one case

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

Then build the tables and figures as in [results/README.md](../results/README.md), with
`--results "$OUTPUT_DIR"`.

- **RealToxicityPrompts.** `rtp` reuses the ParaDetox direction and mask of the same seed. Run only
  its `grid`, after the ParaDetox `fit`.
- **Proportion of coordinates.** Add `--fraction 0.1`, `0.4` or `0.8` to both `fit` and `grid`.
  The default is `0.2`.
- **One GPU type per case.** Run a case, including its unsteered reference, on a single GPU type.
  The text-quality limit is a ratio to that reference.
- **Resuming.** A repeated command skips the units that are already complete. Run each case in a
  single process.
- **Launchers.** `bash launch/run.sh fit ...` runs a command in the shared environment, and
  `launch/run.sbatch` is a Slurm template.

## Code

| Path | Contents |
|---|---|
| `sparse_steering/method/` | Mask training, sparse edit and constrained normalization |
| `sparse_steering/baselines/` | Dense and PCA steering; magnitude, gradient and IE coordinate selection |
| `sparse_steering/tasks/` | Answer-selection and detoxification tasks, data sources and splits |
| `sparse_steering/evaluation/` | Strength grid, best feasible selection, capability and confidence intervals |
| `sparse_steering/runtime/` | Model loading, hooks, storage and multi-GPU placement |
| `sparse_steering/analysis/` | Tables and figures |
| `protocol/` | Experiment settings (`experiment.json`) and the fixed splits of each seed |
| `configs/` | Dataset revisions, library versions and capability item identifiers |

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

## Tests

```bash
python -m unittest discover
```

The 34 tests run on CPU and download nothing.
