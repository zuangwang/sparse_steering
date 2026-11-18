"""Fit steering directions and masks, then evaluate strength sweeps."""

import argparse
import fcntl
import gzip
import json
import math
import time
import types

from sparse_steering.evaluation.text_quality import best_feasible
from sparse_steering.runtime.storage import CODE, guard, metadata, path, read, save, sha
from sparse_steering.tasks.splits import training_seeds

METHODS = (
    "dense",
    "normalized_dense",
    "pca_dense",
    "pca_sparse",
    "magnitude",
    "gradient",
    "ie",
    "ours",
    "sparsification_only",
    "sparsification_global_norm",
)
GRIDS = read(CODE / "protocol/experiment.json")


def initialize(model, task, *, two_gpus=False):
    from sparse_steering.runtime import model_loading

    spec = dict(
        model=model,
        dataset=task,
        kind="qa" if task in ("truthfulqa", "halueval") else "toxicity",
        model_spec=GRIDS["models"][model],
    )
    if task == "rtp":

        def prompt(row):
            return row.get("prompt", "Question: " + row["question"] + "\nAnswer:")

        model_loading.generation_prompt = prompt
    if model == "qwen32b":
        from sparse_steering.runtime.multi_gpu import bind

        runtime, torch, np, audit, *_ = bind(spec, two_gpus=two_gpus)
    else:
        runtime, torch, np, audit, *_ = model_loading.bind(spec)
    return spec, runtime, torch, np, audit, runtime.make_hook


def loc(variable, seed, model, task, *parts):
    return path(variable, f"seed{seed}", model, task, *parts)


def fit(seed, model, task, fraction=0.2):
    guard(gpu=True)
    started = time.time()
    folder = loc("LLM_ARTIFACT_ROOT", seed, model, task)
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / ("fits.json" if fraction == 0.2 else f"fits-rho{round(100 * fraction)}.json")
    if target.exists():
        return
    spec, runtime, torch, np, _, _ = initialize(model, task, two_gpus=model == "qwen32b")
    panels = read(path("LLM_EXPERIMENT_CACHE_ROOT", f"panels/seed{seed}/{task}.json"))
    direction = folder / "direction"
    direction.mkdir(exist_ok=True)
    if not (direction / "fit_states.npz").exists():
        from sparse_steering.runtime.model_loading import fit_direction

        fit_direction(runtime, panels["train"], direction)
    with np.load(direction / "fit_states.npz") as stored:
        dense, scale, states = (
            stored["dense"].copy(),
            float(stored["scale"]),
            stored["states"].copy(),
        )
    maskfolder = folder / f"rho{round(100 * fraction)}"
    maskfolder.mkdir(exist_ok=True)
    from sparse_steering.method.mask_training.training import fit_mask

    mask = fit_mask(
        runtime, panels, dense, scale, maskfolder, seed=seed, task=task, fraction=fraction
    )
    learned = dense.astype(np.float64) * mask
    learned /= np.linalg.norm(learned)
    sparse = dict(
        scale=scale,
        dense=dense.tolist(),
        masked_direction=learned.tolist(),
        extra_normalization_mask=[0] * len(dense),
        selected_mask=mask.astype(int).tolist(),
        support_policy="entire_saved_learned_mask",
    )
    raw = np.split(states.astype(np.float64), 2)
    raw_norm = float(np.linalg.norm(raw[0].mean(0) - raw[1].mean(0)))
    fits = {m: sparse for m in ("ours", "sparsification_only", "sparsification_global_norm")}
    fits["dense"] = dict(scale=scale, dense=dense.tolist())
    if task == "paradetox":
        from sparse_steering.tasks.detoxification import ToxicityScorer, calibrate

        calibrate(ToxicityScorer(), panels, folder)
    if fraction == 0.2:
        from sparse_steering.baselines.pca import fit_arrays

        pca, diagnostics, check, check_scale = fit_arrays(
            states, len(panels["train"]), len(dense), False
        )
        if not np.allclose(check, dense, rtol=1e-05, atol=1e-07) or check_scale != scale:
            raise ValueError("Paired PCA activation ordering or scale mismatch.")
        fits.update(
            dense=dict(scale=scale, dense=dense.tolist()),
            normalized_dense=dict(
                scale=scale, dense=dense.tolist(), normalized_dense=dense.tolist()
            ),
        )
        fits.update({name: dict(scale=scale, dense=vector) for name, vector in pca.items()})
        from sparse_steering.baselines.coordinate_selectors.attribution import compute

        attrfolder = folder / "attribution"
        attrfolder.mkdir(exist_ok=True)
        effect = compute(
            runtime, torch, panels["train"], dense.tolist(), scale, spec["kind"], attrfolder
        )
        from sparse_steering.baselines.coordinate_selectors.selectors import fits as selector_fits

        selected = selector_fits(dense.tolist(), effect["indirect_effect"])
        fits.update({name: dict(value, scale=scale) for name, value in selected.items()})
        save(loc("OUTPUT_DIR", seed, model, task, "selectors.json"), selected)
        save(folder / "pca.json", diagnostics)
    for method, value in fits.items():
        value["raw_direction_norm"] = raw_norm
        value["alpha_divisor"] = (
            float(np.linalg.norm(dense * mask))
            if method in ("ours", "sparsification_only", "sparsification_global_norm")
            else (
                float(np.linalg.norm(value["direction"]))
                if "direction" in value
                else float(np.linalg.norm(value["dense"])) if method == "pca_sparse" else 1.0
            )
        )
    candidates = read(maskfolder / "mask_candidates.json")
    selected = read(maskfolder / "selected_mask.json")
    reruns = []
    for rerun in (0, 1, 2):
        final = next((c for c in candidates if c["rerun"] == rerun and c["epoch"] == 2))
        reruns.append(
            dict(
                rerun=rerun,
                coordinates=[i for i, x in enumerate(final["mask"]) if x],
                seeds=[training_seeds(seed, task, rerun, e) for e in (0, 1)],
            )
        )
    result = dict(
        selected_rerun=selected["rerun"],
        selected_epoch=selected["epoch"],
        coordinates=mask.nonzero()[0].tolist(),
        reruns=reruns,
        selection_metrics={k: v for k, v in selected.items() if k not in ("mask", "scores")},
        step_mask_count=len(list((maskfolder / "step_masks").glob("*.json.gz"))),
        step_masks_remote=str(maskfolder / "step_masks"),
        metadata=metadata(started, [f"seed{seed}/{model}/{task}/rho{round(100 * fraction)}"]),
    )
    result["metadata"]["split_sha256"] = sha(CODE / f"protocol/splits/seed{seed}/{task}.json")
    save(
        loc(
            "OUTPUT_DIR",
            seed,
            model,
            task,
            "mask.json" if fraction == 0.2 else f"proportion/mask-rho{round(100 * fraction)}.json",
        ),
        result,
    )
    save(target, fits)
    print("Fit completed.", seed, model, task, fraction, flush=True)


def configure(runtime, torch, original, method, edit_norm_ratio_squared, fit):
    if method == "sparsification_global_norm" and edit_norm_ratio_squared:
        from sparse_steering.method.global_normalization import make_hook

        def global_hook(self, direction, target_norm=0.0, *, matched=True, signs=None):
            assert matched and signs is None
            return make_hook(self, torch, direction)

        runtime.make_hook = types.MethodType(global_hook, runtime)
        config = dict(
            method=method,
            edit_norm_ratio_squared=edit_norm_ratio_squared,
            continuation_squared_norm_multiplier=1.0,
        )
        runtime.configure(config, fit["scale"])
        return config, torch.tensor(fit["masked_direction"], device="cuda", dtype=torch.float64)
    from sparse_steering.evaluation.strength_sweep import configure as parent

    return parent(
        runtime,
        torch,
        original,
        method if method != "sparsification_global_norm" else "ours",
        edit_norm_ratio_squared,
        fit,
    )


def measure(
    spec, runtime, torch, audit, original, rows, fits, method, edit_norm_ratio_squared, target
):
    if target.exists():
        return read(target)["summary"]
    started = time.time()
    fit = fits[method]
    config, direction = configure(runtime, torch, original, method, edit_norm_ratio_squared, fit)
    values = {}
    prediction_steps = {}
    for kind in ("binary", "generation") if spec["kind"] == "qa" else ("generation",):
        runtime.prediction_trace = []
        if kind == "binary":
            values[kind] = runtime.evaluate_binary(rows, direction)
        else:
            values[kind] = []
            for i in range(0, len(rows), runtime.batch_size):
                values[kind].extend(runtime.generate(rows[i : i + runtime.batch_size], direction))
        prediction_steps[kind] = runtime.prediction_trace
        if method in (
            "magnitude",
            "gradient",
            "ie",
            "sparsification_only",
            "ours",
            "sparsification_global_norm",
        ):
            from sparse_steering.evaluation.sweep_checks import raw_audit

            auditfit = (
                dict(
                    fit,
                    masked_direction=fit["direction"],
                    extra_normalization_mask=[0] * len(fit["direction"]),
                )
                if method in ("magnitude", "gradient", "ie")
                else fit
            )
            auditconfig = config
            if method == "sparsification_global_norm" and edit_norm_ratio_squared:
                auditconfig = dict(config, method="ours")
                auditfit = dict(
                    fit, extra_normalization_mask=[v == 0 for v in fit["masked_direction"]]
                )
            raw_audit(values[kind], prediction_steps[kind], rows, kind, auditconfig, auditfit)
        else:
            audit(values[kind], prediction_steps[kind], rows, kind, config, fit["scale"])
    generated = values["generation"]
    if spec["kind"] == "qa":
        scored = values["binary"]
        score = sum((v["accuracy"] for v in scored)) / len(scored)
    else:
        from sparse_steering.tasks.detoxification import ToxicityScorer

        if not hasattr(runtime, "_sweep_scorer"):
            runtime._sweep_scorer = ToxicityScorer()
        values["scores"] = runtime._sweep_scorer.score(
            [dict(id=v["id"], candidate=v["text"]) for v in generated]
        )
        if [v["id"] for v in values["scores"]] != [r["id"] for r in rows]:
            raise ValueError("Scorer panel mismatch.")
        score = sum((v["score"] for v in values["scores"])) / len(rows)
        scored = generated
    vector = fit.get("direction", fit.get("masked_direction", fit.get("dense")))
    point = dict(
        method=method,
        edit_norm_ratio_squared=edit_norm_ratio_squared,
        score=100 * score,
        actual_squared_edit_norm=sum((v["realized_squared_edit_norm"] for v in scored))
        / (len(scored) * fit["scale"] ** 2),
        completion_nll=sum((v["completion_nll"] for v in generated)) / len(rows),
        repeated_4gram=sum((v["repeated_4gram"] for v in generated)) / len(rows),
        coordinates=(
            0
            if not edit_norm_ratio_squared
            else (
                len(vector)
                if method == "sparsification_global_norm"
                else sum((x != 0 for x in vector))
            )
        ),
        questions=len(rows),
        seconds=time.time() - started,
    )
    if edit_norm_ratio_squared and method in (
        "dense",
        "pca_dense",
        "pca_sparse",
        "magnitude",
        "gradient",
        "ie",
        "sparsification_only",
    ):
        generation_prediction_steps = prediction_steps["generation"]
        point["additive_multiplier_mean"] = math.fsum(
            (
                a.get("alpha", math.sqrt(a["realized_squared_edit_norm"]))
                for a in generation_prediction_steps
            )
        ) / len(generation_prediction_steps)
    target.parent.mkdir(parents=True, exist_ok=True)
    raw = target.with_suffix(".raw.json.gz")
    with gzip.open(raw, "wt") as stream:
        json.dump(dict(values=values, prediction_steps=prediction_steps), stream, allow_nan=False)
    save(
        target, dict(summary=point, metadata=metadata(started, [str(target)]), raw_sha256=sha(raw))
    )
    print(
        json.dumps(dict(unit=str(target), seconds=point["seconds"], score=point["score"])),
        flush=True,
    )
    return point


def grid(seed, model, task, fraction=0.2):
    guard(gpu=True)
    started = time.time()
    dest = loc(
        "OUTPUT_DIR",
        seed,
        model,
        task,
        "test.json" if fraction == 0.2 else f"proportion/rho{round(100 * fraction)}.json",
    )
    if dest.exists():
        return
    spec, runtime, torch, np, audit, original = initialize(model, task)
    rows = read(path("LLM_EXPERIMENT_CACHE_ROOT", f"panels/seed{seed}/{task}.json"))["test"]
    fit_task = "paradetox" if task == "rtp" else task
    fits = read(
        loc(
            "LLM_ARTIFACT_ROOT",
            seed,
            model,
            fit_task,
            "fits.json" if fraction == 0.2 else f"fits-rho{round(100 * fraction)}.json",
        )
    )
    base_target = loc("LLM_EXPERIMENT_CACHE_ROOT", seed, model, task, "units/unsteered.json")
    unsteered = measure(
        spec, runtime, torch, audit, original, rows, fits, "dense", 0.0, base_target
    )
    unsteered = dict(unsteered, strength_index=0, alpha=0.0, nll_ratio=1.0)
    repetition_limit = (
        0.5
        if spec["kind"] == "qa"
        else max(0.1, unsteered["repeated_4gram"]) if task == "rtp" else 0.1
    )
    unsteered.update(
        repetition_limit=repetition_limit,
        feasible=unsteered["repeated_4gram"] <= repetition_limit + 1e-12,
    )
    grids = {}
    base = (
        0.0,
        0.0001,
        0.0004,
        0.001,
        0.0025,
        0.005,
        0.01,
        0.0225,
        0.04,
        0.09,
        0.16,
        0.36,
        0.49,
        0.64,
        1.0,
        4.0,
    )
    methods = METHODS if task != "rtp" else METHODS[:9]
    if model == "qwen32b":
        methods = ("dense", "sparsification_only", "ours")
    if fraction != 0.2:
        methods = ("ours", "sparsification_only")
    from sparse_steering.evaluation.strength_sweep import NUMERICAL_FAILURES

    for method in methods:
        grids[method] = [dict(unsteered, method=method)]
        for edit_norm_ratio_squared in GRIDS["cases"][model + "/" + task][
            "edit_norm_ratios_squared"
        ]:
            if not edit_norm_ratio_squared:
                continue
            target = loc(
                "LLM_EXPERIMENT_CACHE_ROOT",
                seed,
                model,
                task,
                (
                    f"units/rho{round(100 * fraction)}/{method}/"
                    f"edit_norm_ratio_squared_{edit_norm_ratio_squared:.12g}.json"
                ),
            )
            try:
                point = measure(
                    spec,
                    runtime,
                    torch,
                    audit,
                    original,
                    rows,
                    fits,
                    method,
                    edit_norm_ratio_squared,
                    target,
                )
            except ValueError as e:
                if str(e) not in NUMERICAL_FAILURES | {
                    "Constrained normalization failed its scalar, norm and displacement gates.",
                    "The globally normalized edit violates a displacement, norm, "
                    "cosine or finite-value limit.",
                }:
                    raise
                point = dict(
                    method=method,
                    edit_norm_ratio_squared=edit_norm_ratio_squared,
                    score=None,
                    completion_nll=None,
                    repeated_4gram=None,
                    actual_squared_edit_norm=None,
                    coordinates=None,
                    feasible=False,
                    status="numerical_gate_failure",
                    error=str(e),
                )
                save(target, dict(summary=point, metadata=metadata(started, [str(target)])))
            point = dict(
                point,
                strength_index=(
                    "1/2"
                    if edit_norm_ratio_squared == 2.5e-05
                    else base.index(edit_norm_ratio_squared)
                ),
                repetition_limit=repetition_limit,
                alpha=None,
            )
            if point.get("score") is not None:
                point["nll_ratio"] = point["completion_nll"] / unsteered["completion_nll"]
                point["feasible"] = (
                    point["nll_ratio"] <= 1.15 + 1e-12
                    and point["repeated_4gram"] <= repetition_limit + 1e-12
                )
            grids[method].append(point)
        save(dest.with_name(dest.stem + "_partial.json"), dict(unsteered=unsteered, grids=grids))
    counterparts = dict(
        normalized_dense="dense",
        ours="sparsification_only",
        sparsification_global_norm="sparsification_only",
    )
    for method, points in grids.items():
        counterpart = counterparts.get(method, method)
        for point in points:
            point["alpha_source_method"] = counterpart
            if not point["edit_norm_ratio_squared"]:
                continue
            source = next(
                (
                    p
                    for p in grids.get(counterpart, [])
                    if p["edit_norm_ratio_squared"] == point["edit_norm_ratio_squared"]
                ),
                None,
            )
            if source and source.get("additive_multiplier_mean") is not None:
                point["alpha"] = (
                    source["additive_multiplier_mean"]
                    / fits[method]["raw_direction_norm"]
                    / fits[method]["alpha_divisor"]
                )
                point["alpha_status"] = "complete"
            else:
                point["alpha_status"] = "additive_counterpart_numerical_failure"
    selected = {method: best_feasible(points) for method, points in grids.items()}
    prov = metadata(started, [f"seed{seed}/{model}/{task}/grid/rho{round(100 * fraction)}"])
    prov["split_sha256"] = sha(CODE / f"protocol/splits/seed{seed}/{task}.json")
    save(dest, dict(unsteered=unsteered, grids=grids, selected=selected, metadata=prov))
    save(
        loc(
            "OUTPUT_DIR",
            seed,
            model,
            task,
            (
                "metadata.json"
                if fraction == 0.2
                else f"proportion/metadata-rho{round(100 * fraction)}.json"
            ),
        ),
        prov,
    )


def main(phase=None):
    p = argparse.ArgumentParser()
    if phase is None:
        p.add_argument("phase", choices=("fit", "grid"))
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--model", choices=("gemma", "qwen", "llama", "qwen7", "qwen32b"), required=True)
    p.add_argument("--task", choices=("truthfulqa", "halueval", "paradetox", "rtp"), required=True)
    p.add_argument("--fraction", type=float, default=0.2)
    a = p.parse_args()
    if phase is not None:
        a.phase = phase
    guard(gpu=True)
    lockpath = path("OUTPUT_DIR", "case.lock")
    with lockpath.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (fit if a.phase == "fit" else grid)(a.seed, a.model, a.task, a.fraction)


if __name__ == "__main__":
    main()
