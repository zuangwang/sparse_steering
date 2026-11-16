# Surpassing Dense Steering in LLMs through Optimized Sparsification and Constrained Normalization

Code for the experiments in the paper. The method steers a language model by adding a steering vector on a learned subset of residual-stream coordinates and rescaling the remaining coordinates so that the activation norm is preserved. The repository contains the method, all baselines, the frozen experimental protocol, and the scripts that produce every table and figure.

## Contents

| Path | Contents |
|---|---|
| `reproduce/` | Command-line entry points for every experiment (start here) |
| `multi_seed/` | Frozen protocol: model revisions and layers, strength grids (`CASE_GRIDS.json`), data splits for seeds 0 to 3 (`splits/`), protocol description (`PROTOCOL.md`) |
| `full_support_sparse_steering/`, `learned_sparse_steering/` | Mask training and the steering operators of our method |
| `faithful_baselines/`, `table1_completion/`, `pca_energy_controls/` | Baselines: dense steering, normalized dense steering, PCA (dense and sparse), Magnitude, Gradient, and indirect-effect (IE) coordinate selection |
| `toxicity_bipo/`, `qa_reference_bipo/` | Task-specific fitting and scoring for detoxification and question answering |
| `general_capability_eval/`, `paper_evaluation/` | Capability evaluation (ARC-Challenge, HellaSwag, WikiText-103) and paired bootstrap |
| `mask_training_diagnostics/`, `mask_step_trajectories/` | Selected-coordinate overlap and mask-stability analyses |
| `qwen32b_table1_et/`, `multi_seed_q32/`, `q32mag/` | Multi-GPU runtime for the 32B model |
| `analysis/` | Result readers and plotting code |
| `launch/` | Shell and Slurm templates |
| `configs/` | Capability item identifiers, dataset revisions, recorded library versions |

The remaining top-level packages are shared helpers imported by the packages above.

## Requirements

- Linux, Python 3.11, a CUDA GPU with native bf16 support.
- `torch==2.8.0+cu128`, `transformers==4.55.0`, `numpy==1.26.4`; other dependencies are listed in `requirements.txt`.

| Model key | Hugging Face identifier | Steered layer | GPU memory |
|---|---|---:|---|
| `gemma` | `google/gemma-2-2b` | 18 | 24 GB |
| `qwen` | `Qwen/Qwen2.5-3B-Instruct` | 24 | 24 GB |
| `llama` | `NousResearch/Llama-2-7b-chat-hf` | 21 | 40 to 80 GB |
| `qwen7` | `Qwen/Qwen2.5-7B-Instruct` | 19 | 40 to 80 GB |
| `qwen32b` | `Qwen/Qwen2.5-32B-Instruct` | 43 | 80 GB for inference; 2 × 80 GB for mask training |

Model revisions are pinned in `multi_seed/CASE_GRIDS.json`. Toxicity is scored with `s-nlp/roberta_toxicity_classifier` at a pinned revision. Keep all runs of one model, task and seed on one GPU type, including the unsteered reference, because the text-quality limit is a ratio to that reference.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

export SPARSE_STEERING_RUN_ROOT=/path/to/storage   # outside this repository
source launch/environment.sh
```

`launch/environment.sh` places model weights, datasets, caches and results below `SPARSE_STEERING_RUN_ROOT`. Results are written to `$OUTPUT_DIR` (default `$SPARSE_STEERING_RUN_ROOT/results`). Log in to the Hugging Face Hub first if a model is gated.

```bash
python -m reproduce.download_models --model gemma   # once per model
python -m reproduce.prepare                         # datasets and committed splits
python -m reproduce.capability_panels               # fixed capability panels
```

## Running the experiments

Each case is one model, one task and one seed. Tasks are `truthfulqa`, `halueval`, `paradetox` and `rtp`. RealToxicityPrompts (`rtp`) reuses the ParaDetox direction and mask of the same seed, so it has a `grid` step but no `fit` step.

```bash
# 1. Fit the steering direction, the baselines' coordinate selections and the mask
python -m reproduce.runner fit  --seed 0 --model gemma --task truthfulqa

# 2. Test sweep over the strength grid for all methods
python -m reproduce.runner grid --seed 0 --model gemma --task truthfulqa

# 3. Selected-coordinate and mask-stability analyses (CPU, after the fits of a model)
python -m reproduce.postprocess --model gemma

# 4. Capability evaluation at each method's selected setting (after the test sweep)
python -m reproduce.capability --seed 0 --model gemma --task truthfulqa

# 5. Tables and figures
python -m reproduce.summarize --results "$OUTPUT_DIR" --output "$OUTPUT_DIR/figures" --seed 0
```

Runs are resumable: a rerun skips the cells that are already saved. Do not run the same case in two processes at once.

### Paper items

| Paper item | Commands | Output of `reproduce.summarize` |
|---|---|---|
| Table 1 and the dense-margin figure (4 models × 4 tasks × 9 methods) | `fit`, `grid` for `gemma`, `qwen`, `llama`, `qwen7` on all tasks | `table1.csv`, `dense_margin.csv`, `dense_margin.pdf` |
| Main-result figure (our method and the dense baseline, including the 32B model) | `fit`, `grid`; add `--model qwen32b` | `main_result.pdf` |
| Norm-strength figure (five normalization variants) | same `fit`, `grid` runs | `norm_strength.pdf` |
| Proportion sweeps (10, 20, 40, 80% of coordinates) | `fit`, `grid` with `--fraction 0.1`, `0.4`, `0.8` after the default 20% run | `proportion_sweeps.pdf` |
| Selected-coordinate overlap | `postprocess` | `coordinates.csv`, `selected_coordinates.pdf` |
| Mask stability | `postprocess` | `stability.csv`, `mask_stability.pdf` |
| Capability table | `capability` for the four smaller models × `truthfulqa`, `halueval`, `paradetox` | `capability.csv` |
| Multiple seeds | repeat with `--seed 1`, `2`, `3` | one summary per seed |

### Protocol

- **Splits.** `multi_seed/splits/seed<k>/<task>.json` list the item identifiers of the training, mask-selection and test sets. Seed 0 for RealToxicityPrompts is rebuilt by a deterministic, checksum-verified rule.
- **Strength grid.** Each case uses the grid in `multi_seed/CASE_GRIDS.json`: twelve strengths, plus four larger strengths used by the main-result and norm-strength figures.
- **Decoding.** Greedy, 32 new tokens, steering applied at every generated position.
- **Selection.** For each method, the reported score is the best test score among the settings that satisfy the text-quality limits (negative log-likelihood ratio to the unsteered model at most 1.15, and the repetition limit of the task).
- **Seeds.** A seed fixes the data splits, everything fitted on the training set, and the initialization and data order of mask training. Each mask is trained three times, and the rerun is chosen on the mask-selection set.

## Running on a cluster

`launch/` contains generic templates. In the Slurm scripts, replace `<ACCOUNT>`, `<PARTITION>` and `<GPU_TYPE>`, set `SS_MODULE` to the module to run, and pass its arguments after the script name. The templates locate the repository root themselves and use the storage variables from `launch/environment.sh`.

## Tests

```bash
python -m compileall -q .
python -m unittest discover
```

The unit tests run on CPU and download nothing.
