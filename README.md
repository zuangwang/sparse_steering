# Surpassing Dense Steering in LLMs through Optimized Sparsification and Constrained Normalization

Code for the experiments in the paper. The method steers a language model by adding a steering
vector on a learned subset of residual-stream coordinates and rescaling only these selected
coordinates so that their norm equals their norm before the edit, which leaves all other coordinates
unchanged. The repository contains the method, all baselines, the frozen experimental protocol, and
the scripts that produce every table and figure.

## Contents

The library lives in `sparse_steering/`; its rows below omit that prefix. Other rows are
relative to the source root.

| Path | Contents |
|---|---|
| `reproduce/` | Preparation, fits, grids, capability, postprocessing and summaries |
| `protocol/experiment.json` | Model revisions, layers, grids, training and quality limits |
| `protocol/splits/` | Fixed task membership for seeds 0 to 3 |
| `protocol/PROTOCOL.md` | Objectives, scoring and selection rules |
| `configs/` | Capability identifiers, dataset revisions and library versions |
| `launch/` | Storage environment, shell launcher and Slurm template |
| `method/mask_training/` | Answer-selection and open-generation objectives and trainers |
| `method/sparse_edit.py` | Sparse addition and bf16 displacement matching |
| `method/constrained_normalization.py` | Norm preservation on selected coordinates |
| `method/global_normalization.py` | Norm preservation on the full activation |
| `baselines/` | Dense, PCA and `coordinate_selectors/` comparison methods |
| `tasks/` | Answer selection, detoxification, sources and fixed splits |
| `evaluation/strength_sweep.py` | Common strength grid and method configuration |
| `evaluation/text_quality.py` | Best feasible setting under quality limits |
| `evaluation/sweep_checks.py` | Displacement, support and sweep consistency checks |
| `evaluation/capability.py` | Capability orchestration and panel validation |
| `evaluation/choice_scoring.py` | ARC and HellaSwag choice likelihoods |
| `evaluation/language_modeling.py` | WikiText continuation likelihoods |
| `evaluation/bootstrap.py` | Paired confidence intervals |
| `runtime/` | Model loading, hooks, storage, runner and multi-GPU placement |
| `analysis/` | Coordinate overlap, mask stability, tables and figures |
| `tests/` | CPU tests for code reached by the reproduce commands |

### Where each equation lives

All paths in this table are relative to `sparse_steering/`.

| Paper expression | Implementation |
|---|---|
| Mean-difference direction and activation scale | `runtime/model_loading.py::fit_direction` |
| Exact-K mask and straight-through surrogate | `method/mask_training/masks.py` |
| Answer-selection loss, gamma | `method/mask_training/objectives.py::reference_loss` |
| Full-response loss, gamma | `method/mask_training/objectives.py::preference_loss` |
| Sparse edit, h_tilde = h + alpha * v_M | `method/sparse_edit.py::realize_batch` |
| Constrained normalization, beta * h_tilde | `method/constrained_normalization.py` |
| Global normalization comparison | `method/global_normalization.py` |
| Best feasible score under NLL and repetition limits | `evaluation/text_quality.py` |

The constrained operator implements sparse addition and rescaling together. Its multiplier search
checks the final bf16 displacement against the dense edit requested at the same strength. The loss
scale `gamma` is distinct from the normalization factor `beta`.

## Requirements

- Linux, Python 3.11, a CUDA GPU with native bf16 support.
- `torch==2.8.0+cu128`, `transformers==4.55.0`, `numpy==1.26.4`; other dependencies are listed in
  `requirements.txt`.

| Model key | Hugging Face identifier | Steered layer | GPU memory |
|---|---|---:|---|
| `gemma` | `google/gemma-2-2b` | 18 | 24 GB |
| `qwen` | `Qwen/Qwen2.5-3B-Instruct` | 24 | 24 GB |
| `llama` | `NousResearch/Llama-2-7b-chat-hf` | 21 | 40 to 80 GB |
| `qwen7` | `Qwen/Qwen2.5-7B-Instruct` | 19 | 40 to 80 GB |
| `qwen32b` | `Qwen/Qwen2.5-32B-Instruct` | 43 | See below |

The 32B model needs 80 GB for inference and two 80 GB GPUs for mask training.

Model revisions are pinned in `protocol/experiment.json`. Toxicity is scored with
`s-nlp/roberta_toxicity_classifier` at a pinned revision. Keep all runs of one model, task and seed
on one GPU type, including the unsteered reference, because the text-quality limit is a ratio to
that reference.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

export SPARSE_STEERING_RUN_ROOT=/path/to/storage
source launch/environment.sh
python -m reproduce.download_models --model gemma
python -m reproduce.prepare
python -m reproduce.capability_panels
```

The environment places model weights, datasets, caches and fitted objects below
`SPARSE_STEERING_RUN_ROOT`, outside the source folder. Results are written to `$OUTPUT_DIR`. Log in
to the Hugging Face Hub first if a model is gated.

## Running the experiments

Each case is one model, task and seed. Tasks are `truthfulqa`, `halueval`, `paradetox` and `rtp`.
RealToxicityPrompts reuses the ParaDetox direction and mask of the same seed, so run its `grid`
command after fitting ParaDetox.

```bash
# Fit the direction, comparison methods and coordinate mask
python -m reproduce.fit --seed 0 --model gemma --task truthfulqa

# Evaluate the strength grid
python -m reproduce.grid --seed 0 --model gemma --task truthfulqa

# Compute coordinate overlap and mask stability after fitting a model
python -m reproduce.postprocess --model gemma

# Evaluate capability at each method's best feasible setting
python -m reproduce.capability --seed 0 --model gemma --task truthfulqa

# Build tables and figures
python -m reproduce.summarize --results "$OUTPUT_DIR" --output "$OUTPUT_DIR/figures" --seed 0
```

Repeat with `--seed 1`, `2` and `3`. To change the selected fraction, add `--fraction 0.1`, `0.4` or
`0.8` to both `fit` and `grid`; the default is `0.2`. `python -m reproduce.runner fit ...` and
`python -m reproduce.runner grid ...` are equivalent forms. `bash launch/run.sh fit ...` uses the
shared environment, and `launch/run.sbatch` provides a generic submission template.

The 32B model uses two visible GPUs for fitting and one for grid inference. `runtime/multi_gpu.py`
places its intervention layer, embeddings and logits on GPU 0. A completed unit is skipped when its
command is repeated; keep each case in a single process.

### Paper items

Run `fit` and `grid` for each model, task and seed, then `summarize`. Include `qwen32b` for the
main-result figure and the fraction settings for the proportion figure. Run `postprocess` for
coordinate and stability outputs, and `capability` for the capability table.

| Paper item | Summary output |
|---|---|
| Main comparison | `comparison.csv` |
| Dense margin | `dense_margin.csv`, `dense_margin.pdf` |
| Main result | `main_result.pdf` |
| Norm and strength | `norm_strength.pdf` |
| Proportion sweeps | `proportion_sweeps.pdf` |
| Coordinate overlap | `coordinates.csv`, `selected_coordinates.pdf` |
| Mask stability | `stability.csv`, `mask_stability.pdf` |
| General capability | `capability.csv` |

Repeating the commands with seeds 1, 2 and 3 produces one summary per seed.

### Protocol

- **Settings.** `protocol/experiment.json` is the single experiment definition.
  `protocol/PROTOCOL.md` explains every section.
- **Splits.** `protocol/splits/seed<k>/<task>.json` fixes training, mask-selection and test
  membership. Seed 0 for RealToxicityPrompts uses a deterministic, checksum-verified reconstruction
  rule.
- **Strength.** `strength_index` identifies the 0 to 15 grid; five cases also have a half-step
  identified as `"1/2"`. `alpha` is the multiplier of the unnormalized added vector. The internal
  `edit_norm_ratio_squared` sets the requested squared displacement relative to the
  training-activation scale.
- **Methods.** Keys are `dense`, `normalized_dense`, `pca_dense`, `pca_sparse`, `magnitude`,
  `gradient`, `ie`, `ours`, `sparsification_only` and `sparsification_global_norm`. The reference is
  `unsteered`. Our method combines optimized sparsification with constrained normalization. The
  dense baseline (CAA) uses the mean-difference direction. Mask training adapts the
  preference objective from BiPO while freezing the direction and language model.
- **Selection.** The best feasible setting has the highest task score within the NLL and repetition
  limits. NLL ratio is at most 1.15; task-specific repetition limits are defined in the protocol.
- **Seeds.** Each experiment seed fixes data splits and fitted objects. Three mask-training reruns
  differ in initialization and data order; select one on the separate mask-selection set.

## Tests

```bash
python -m compileall -q .
python -m unittest discover
```

The 28 tests run on CPU and download nothing. Each reproduce command accepts `--help`.
