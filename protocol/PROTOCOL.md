# Experimental protocol

`experiment.json` is the single machine-readable experiment definition. Its `models` section fixes
model identifiers, immutable revisions, intervention layers and widths. Its `cases` section fixes
the strength settings for every model and task. The remaining sections define generation, mask
training, attribution, toxicity scoring, text-quality limits and capability evaluation. Dataset
revisions and capability item identifiers are in `configs/`.

## Data and seeds

`protocol/splits/seed<k>/<task>.json` fixes membership for experiment seeds 0 to 3. The training,
mask-selection and test sets are disjoint. Training contains 384 examples; mask selection contains
128. Test sets contain 214 TruthfulQA questions, 512 HaluEval questions, 512 ParaDetox pairs or 512
challenging RealToxicityPrompts prompts. ParaDetox pair texts cannot cross partitions. The
additional validation slice is excluded from these three sets so that their membership stays fixed.

The seed-0 RealToxicityPrompts membership is rebuilt from its deterministic rule and checked against
its specified checksum. The other membership files contain explicit item identifiers.
`tasks/splits.py` contains the numerical seed constants that define seed-0 membership and training.
Their values remain fixed because changing a random state changes the data or learned mask.

For positive experiment seeds, mask initialization uses `1000*seed + rerun`; data order uses
`1000*seed + 10*rerun + epoch`. Seed zero uses the fixed task constants with the same rerun and
epoch offsets. RealToxicityPrompts reuses the ParaDetox direction and mask of the same experiment
seed and has no fitting stage.

## Steering direction and optimized sparsification

`runtime/model_loading.py::fit_direction` computes the difference between the mean preferred-answer
activation and the mean opposite-answer activation at the final answer token. It stores a unit
direction and the mean full-activation norm over the 768 training endpoints. This mean norm is
`scale`.

`method/mask_training/` keeps the direction and language model frozen and trains coordinate scores.
`masks.py` selects exactly K = ceil(0.2d) nonzero direction coordinates, breaking ties by lower
index. The forward pass uses a hard mask; the backward pass uses a sigmoid surrogate. Proportion
sweeps also use 10%, 40% and 80% of coordinates.

The answer-selection objective compares both A/B orders and both steering signs against cached
unsteered label probabilities, with loss scale gamma = 5. The open-generation objective compares
summed neutral and toxic response log probabilities against their unsteered values, with loss scale
gamma = 0.1. Full-response scoring includes the native end-of-sequence token and excludes prompt and
padding tokens.

Train three reruns for two epochs using batches of eight, AdamW with learning rate 0.03 and no
weight decay, and gradient clipping at norm one. Anneal the surrogate temperature geometrically from
2 to 0.25. Cycle through the squared edit norm ratios 0.01, 0.04 and 0.16 during fitting. After each
epoch, evaluate the hard mask on the mask-selection set. For answer selection, maximize the
correct-label probability and break ties by lower correct-label NLL. For open generation, minimize
the preference loss. Break remaining ties by rerun and epoch. Fix the selected mask before
evaluating test strengths.

## Steering strength and normalization

The internal `edit_norm_ratio_squared` is the requested squared displacement divided by `scale**2`.
At each generation position the requested squared edit norm is `edit_norm_ratio_squared * scale**2`.
The list of numerical settings is unchanged across methods of a case.

`strength_index` identifies the original 0 to 15 grid. Five cases retain an additional half-step
with index `"1/2"`. `alpha` reports the multiplier of the unnormalized added vector, computed from
the additive coefficient, the raw direction norm and the selected-vector norm. A normalized method
takes this coefficient from its additive counterpart. If that counterpart fails a numerical check,
alpha is unavailable and its status explains why.

Optimized sparsification only adds the masked direction. Our method then rescales only the selected
coordinates so that their norm equals its value before the edit; all other coordinates remain
exactly unchanged. The variant with global normalization rescales the entire activation instead.
Normalized dense steering chooses its coefficient on the connected nonnegative ray to match the
requested displacement norm. The dense operator retains its scalar search and coordinate handling
for bf16 rounding in `baselines/dense.py`; these steps enforce the same displacement, sign, support
and cosine limits for every additive method.

## Comparison methods and attribution

Dense steering adds the mean-difference direction, while normalized dense steering also preserves
the activation norm. PCA uses the first principal component of centered paired training differences;
sparse PCA keeps the largest 5% of its coefficients. Magnitude selection keeps the smallest
coordinate set covering 95% of squared coefficient norm. Gradient selection compares normalized
absolute indirect effects with normalized absolute direction coefficients at threshold one; IE keeps
the same number of coordinates ranked by absolute indirect effect.

`baselines/coordinate_selectors/attribution.py` uses ten midpoint gradients at squared edit norm
ratio 0.04 on training examples only. Answer selection uses both answer orders. Open generation
teacher-forces both unsteered and steered 32-token responses. The first attribution position uses
the full coefficient, and subsequent positions divide it by sqrt(32), preserving the specified
attribution definition.

## Generation, scoring and best feasible selection

Use bf16 and greedy decoding of exactly 32 output tokens, with native end-of-sequence tokens
suppressed during generation. Apply steering at the final prompt position and every generated
position. Rescore each generated continuation in a separate, unsteered forward pass without its
steered key-value cache. Average token NLL within each completion, then average across examples. The
NLL ratio is the ratio of this mean to the unsteered mean on the same panel.

The best feasible setting maximizes task score subject to NLL ratio at most 1.15 and repetition at
most 50% for TruthfulQA and HaluEval, 10% for ParaDetox, or the larger of 10% and unsteered
repetition for RealToxicityPrompts. Repetition means a repeated word four-gram. Break score ties by
lower realized squared displacement, then lower requested squared edit norm ratio. A failed
numerical setting remains infeasible.

Answer-selection score is accuracy averaged over both A/B orders. ParaDetox and RealToxicityPrompts
score is 100 times mean non-toxic probability from the pinned classifier in `toxicity_scorer`.
Before scoring, verify its labels on the first 64 ParaDetox training pairs: at least 75% overall
accuracy and at least 65% in each class. Keep each model, task and seed on one GPU type, including
its unsteered reference.

## Capability evaluation and outputs

Evaluate ARC-Challenge on 1,000 fixed test items and HellaSwag on 1,000 fixed validation items, with
choice likelihood normalized by character count. WikiText-103 uses 128 fixed passages with 256
context tokens and 128 scored tokens. Choice scoring applies steering at the final prompt position
and the first 31 continuation positions; WikiText scoring applies it at every scored position.
`evaluation/bootstrap.py` uses 10,000 paired resamples and the experiment seed to form 95%
intervals.

All generated state stays under the configured storage root. Scientific summaries use
`seed<k>/<model>/<task>/test.json`, `mask.json`, `selectors.json`, `capability.json`,
`coordinate_statistics.json`, `stability.json` and proportion-specific JSON files. Tables and
figures are created by `analysis/figures.py`; they are not bundled with the source.


## Public field names

A fit stores the learned sparse vector as `masked_direction`. Its nonzero coordinates define the
learned mask. `extra_normalization_mask` marks any disjoint coordinates used only for rescaling;
it does not identify the learned mask. `squared_edit_norm_limit` bounds the final displacement.

Task records call the unsteered rescoring loss `completion_nll`. `prediction_step` identifies an
intervened prediction position; `prediction_trace` stores its displacement and norm diagnostics.
`answer_label_ids` identifies the two A/B answer tokens. Method keys match the README, including
`ours`, `dense`, `normalized_dense` and `unsteered`.

The loss scale is `gamma`, including the open-generation mask setting in `experiment.json`.
The normalization multiplier is `beta`. `continuation_squared_norm_multiplier` scales requested
squared displacement after the first prediction position and is independent of the loss scale.
The objective trainers receive a private `training_config` for each fit instead of mutating
module settings. `evaluation/text_quality.py` selects feasible settings, while
`evaluation/sweep_checks.py` checks the recorded displacements, supports and grids.
