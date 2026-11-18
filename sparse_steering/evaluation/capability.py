"""Evaluate the selected steering settings on fixed capability panels."""

import math
from types import SimpleNamespace

from sparse_steering.evaluation.strength_sweep import configure


def native_prompt(tokenizer, text, model_id):
    if model_id == "google/gemma-2-2b":
        return tokenizer.encode(text)
    system = "You are a helpful assistant."
    if tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            [dict(role="system", content=system), dict(role="user", content=text)],
            tokenize=True,
            add_generation_prompt=True,
        )
    if model_id == "NousResearch/Llama-2-7b-chat-hf":
        return tokenizer.encode(
            "[INST] <<SYS>>\n" + system + "\n<</SYS>>\n\n" + text.strip() + " [/INST]",
            add_special_tokens=True,
        )
    raise ValueError("Missing native instruction template.")


def panels(benchmark):
    path = Path(os.environ["DATA_DIR"]) / "capability_panels" / (benchmark + ".json")
    expected = {
        "mmlu": ("1cbc32a7d3ed34a4ab11c53244822560844aa916542a78eeb9846d82e341ca9d", 1140),
        "wikitext": ("c19c75a5184b2cb11ecd8562fe446c508d6d1548a7db915ff4aacdb29de502a9", 128),
    }
    rows = read(path)
    if benchmark in expected:
        digest, count = expected[benchmark]
        if sha(path) != digest or len(rows) != count:
            raise ValueError("Frozen benchmark panel differs: " + benchmark)
    else:
        manifest = read(path.with_name("MANIFEST.json"))
        if sha(path) != manifest["files_sha256"][path.name] or len(rows) != 1000:
            raise ValueError("Frozen ARC/HellaSwag panel differs.")
    return rows, sha(path)


def view(runtime, torch, original, item, config):
    cell, direction = configure(
        runtime,
        torch,
        original,
        config["method"],
        config["edit_norm_ratio_squared"],
        item["fits"][config["method"]],
    )
    from sparse_steering.evaluation.capability import native_prompt

    wrapper = SimpleNamespace(
        original=runtime,
        torch=torch,
        model=runtime.model,
        tokenizer=runtime.tokenizer,
        spec=runtime.spec,
        direction=direction,
        config=dict(
            zero_steering=not config["edit_norm_ratio_squared"],
            runtime_method=cell["method"],
            edit_norm_ratio_squared=config["edit_norm_ratio_squared"],
            continuation_squared_norm_multiplier=cell["continuation_squared_norm_multiplier"],
            config_id=config["id"],
        ),
    )
    wrapper.forward = runtime.forward
    wrapper.prompt = lambda text: native_prompt(runtime.tokenizer, text, runtime.spec["id"])

    def begin(keys):
        runtime._contexts = [dict(id=str(key)) for key in keys]
        runtime._prediction_step = 0
        runtime.prediction_trace = []
        if direction is None:
            return None
        hook, records = runtime.make_hook(direction)
        return runtime.layer.register_forward_hook(hook)

    wrapper.begin = begin
    return wrapper


def batches(wrapper, benchmark, rows, scorers):
    if benchmark == "mmlu":
        for i in range(0, len(rows), 2):
            yield (i, rows[i : i + 2])
    elif benchmark == "wikitext":
        sequences = [wrapper.tokenizer.encode(r["text"])[:384] for r in rows]
        if any((len(s) != 384 for s in sequences)):
            raise ValueError("WikiText cannot fill its fixed window.")
        for i in range(0, len(rows), 2):
            yield (i, ([r["id"] for r in rows[i : i + 2]], sequences[i : i + 2]))
    else:
        flat = scorers[benchmark].encoded_panel(wrapper, rows)
        yield from enumerate(scorers[benchmark].batches(flat))


def summary(benchmark, values):
    if benchmark == "mmlu":
        subjects = sorted({v["subject"] for v in values})
        scores = {
            s: sum((v["accuracy"] for v in values if v["subject"] == s))
            / sum((v["subject"] == s for v in values))
            for s in subjects
        }
        return dict(
            accuracy=sum(scores.values()) / len(scores),
            subject_accuracy=scores,
            n_items=len(values),
        )
    if benchmark == "wikitext":
        nll = sum((v["nll_sum"] for v in values)) / sum((v["scored_tokens"] for v in values))
        return dict(
            nll=nll,
            perplexity=math.exp(nll),
            n_items=len(values),
            scored_tokens=sum((v["scored_tokens"] for v in values)),
        )
    return dict(accuracy=sum((v["accuracy"] for v in values)) / len(values), n_items=len(values))


import argparse
import fcntl
import gzip
import hashlib
import json
import os
import time
from pathlib import Path

from sparse_steering.runtime.runner import initialize, loc
from sparse_steering.runtime.storage import CODE, guard, metadata, path, read, save, sha

BENCHMARKS = ("arc_challenge", "hellaswag", "wikitext")
METHODS = (
    "dense",
    "normalized_dense",
    "pca_dense",
    "pca_sparse",
    "magnitude",
    "gradient",
    "ie",
    "ours",
)


def prepare_panels():
    from sparse_steering.evaluation.capability import panels

    return {benchmark: panels(benchmark) for benchmark in BENCHMARKS}


def condition(
    runtime, torch, original, fits, fit_sha, config, benchmark, rows, panel_sha, scorers, folder
):
    from sparse_steering.evaluation.capability import batches, summary, view
    from sparse_steering.evaluation.strength_sweep import NUMERICAL_FAILURES

    folder.mkdir(parents=True, exist_ok=True)
    target = folder / "complete.json"
    if target.exists():
        record = read(target)
        if record["status"] == "complete":
            with gzip.open(folder / "items.json.gz", "rt") as stream:
                return record, json.load(stream)
        return record, None
    started = time.time()
    item = dict(fits=fits)
    wrapper = view(runtime, torch, original, item, config)
    scientific_identity = dict(
        config=config,
        benchmark=benchmark,
        panel_sha256=panel_sha,
        fits_sha256=fit_sha,
        scorer_sha256=sha(scorers[benchmark].__file__),
    )
    identity = hashlib.sha256(json.dumps(scientific_identity, sort_keys=True).encode()).hexdigest()
    values = []
    try:
        for index, batch in batches(wrapper, benchmark, rows, scorers):
            target_batch = folder / f"batch{index:05d}.json"
            if target_batch.exists():
                saved = read(target_batch)
                if saved["identity"] != identity:
                    raise ValueError("Capability batch inputs changed.")
                scored = saved["values"]
            else:
                if benchmark == "wikitext":
                    scored, trace, _ = scorers[benchmark].score(wrapper, *batch)
                else:
                    scored = scorers[benchmark].score_batch(wrapper, batch)
                    trace = []
                if trace:
                    with gzip.open(folder / f"batch{index:05d}.raw.json.gz", "wt") as stream:
                        json.dump(trace, stream, allow_nan=False)
                save(target_batch, dict(identity=identity, values=scored))
            values.extend(scored)
        if benchmark != "wikitext":
            values = scorers[benchmark].aggregate_choices(rows, values)
        if len(values) != len(rows):
            raise ValueError("Incomplete capability panel.")
        with gzip.open(folder / "items.json.gz", "wt") as stream:
            json.dump(values, stream, allow_nan=False)
        record = dict(
            status="complete",
            config=config,
            benchmark=benchmark,
            summary=summary(benchmark, values),
            panel_sha256=panel_sha,
            items_sha256=sha(folder / "items.json.gz"),
            seconds=time.time() - started,
            metadata=metadata(started, [config["id"] + "/" + benchmark]),
        )
    except ValueError as error:
        if str(error) not in NUMERICAL_FAILURES:
            raise
        record = dict(
            status="failed",
            failure_kind="numerical_gate_failure",
            error=str(error),
            config=config,
            benchmark=benchmark,
            panel_sha256=panel_sha,
            seconds=time.time() - started,
            metadata=metadata(started, [config["id"] + "/" + benchmark]),
        )
        values = None
    save(target, record)
    print(
        json.dumps(
            dict(
                config=config["id"],
                benchmark=benchmark,
                status=record["status"],
                seconds=record["seconds"],
            )
        ),
        flush=True,
    )
    return record, values


def run(seed, model, task):
    guard(gpu=True)
    started = time.time()
    dest = loc("OUTPUT_DIR", seed, model, task, "capability.json")
    if dest.exists():
        return
    selected_path = loc("OUTPUT_DIR", seed, model, task, "test.json")
    selected = read(selected_path)["selected"]
    fit_path = loc("LLM_ARTIFACT_ROOT", seed, model, task, "fits.json")
    fits = read(fit_path)
    fit_sha = sha(fit_path)
    spec, runtime, torch, np, _, original = initialize(model, "truthfulqa")
    from sparse_steering.evaluation.bootstrap import paired
    from sparse_steering.evaluation.capability import scorer_modules

    panels = prepare_panels()
    scorers = scorer_modules()
    result = dict(
        seed=seed,
        model=model,
        task=task,
        selection_source_sha256=sha(selected_path),
        methods={},
        unsteered={},
    )
    for benchmark in BENCHMARKS:
        rows, panel_sha = panels[benchmark]
        baseconfig = dict(id=model + "__unsteered", method="dense", edit_norm_ratio_squared=0.0)
        basefolder = path("LLM_EXPERIMENT_CACHE_ROOT", f"capability/{model}/unsteered/{benchmark}")
        unsteered, basevalues = condition(
            runtime,
            torch,
            original,
            fits,
            "unsteered",
            baseconfig,
            benchmark,
            rows,
            panel_sha,
            scorers,
            basefolder,
        )
        if basevalues is None:
            raise ValueError("Unsteered capability evaluation failed.")
        result["unsteered"][benchmark] = unsteered
        unsteered_zero = paired(basevalues, basevalues, benchmark, seed=seed)
        result["methods"].setdefault("unsteered", {})[benchmark] = dict(
            unsteered, paired=unsteered_zero
        )
        for method in METHODS:
            choice = selected[method]
            if choice is None:
                record = dict(
                    status="unavailable", reason="No feasible test setting", selected=None
                )
            elif not choice["edit_norm_ratio_squared"]:
                record = dict(
                    unsteered, reused_unsteered=True, selected=choice, paired=unsteered_zero
                )
            else:
                config = dict(
                    id=f"seed{seed}__{model}__{task}__{method}",
                    method=method,
                    edit_norm_ratio_squared=choice["edit_norm_ratio_squared"],
                )
                folder = loc(
                    "LLM_EXPERIMENT_CACHE_ROOT", seed, model, task, "capability", method, benchmark
                )
                record, values = condition(
                    runtime,
                    torch,
                    original,
                    fits,
                    fit_sha,
                    config,
                    benchmark,
                    rows,
                    panel_sha,
                    scorers,
                    folder,
                )
                record = dict(record, selected=choice)
                if values is not None:
                    record["paired"] = paired(values, basevalues, benchmark, seed=seed)
            result["methods"].setdefault(method, {})[benchmark] = record
        save(dest.with_name("capability_partial.json"), result)
    result["metadata"] = metadata(started, [f"seed{seed}/{model}/{task}/capability"])
    result["metadata"]["split_sha256"] = sha(CODE / f"protocol/splits/seed{seed}/{task}.json")
    save(dest, result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--model", choices=("gemma", "qwen", "llama", "qwen7", "qwen32b"), required=True
    )
    parser.add_argument("--task", choices=("truthfulqa", "halueval", "paradetox"), required=True)
    args = parser.parse_args()
    guard(gpu=True)
    with path("OUTPUT_DIR", "case.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args.seed, args.model, args.task)


def scorer_modules():
    from sparse_steering.evaluation import choice_scoring, language_modeling

    return dict(arc_challenge=choice_scoring, hellaswag=choice_scoring, wikitext=language_modeling)
