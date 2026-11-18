"""Rebuild the paper's four-seed tables from saved measurements (CPU only)."""

import argparse
import json
import math
from pathlib import Path
from statistics import mean

from sparse_steering.analysis.figures import METHODS, MODELS, TASKS, table
from sparse_steering.evaluation.text_quality import best_feasible

SEEDS = (0, 1, 2, 3)
BENCHMARKS = ("arc_challenge", "hellaswag", "wikitext")


def read_case(results, seed, model, task, filename):
    path = results / f"seed{seed}" / model / task / filename
    if not path.is_file():
        raise ValueError(f"Missing measurement: {path}")
    return json.loads(path.read_text())


def finite(value):
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"Expected a finite measurement, got {value!r}")
    return value


def comparison(results):
    """Select independently for every seed before averaging the scores."""
    rows = []
    models = list(MODELS)
    if any((results / f"seed{s}" / "qwen32b").exists() for s in SEEDS):
        models.append("qwen32b")
    for model in models:
        methods = METHODS if model != "qwen32b" else ("dense", "ours")
        for task in TASKS:
            scores = {m: [] for m in ("unsteered", *methods)}
            for seed in SEEDS:
                data = read_case(results, seed, model, task, "test.json")
                scores["unsteered"].append(finite(data["unsteered"]["score"]))
                for method in methods:
                    winner = best_feasible(data["grids"][method])
                    if winner is None:
                        raise ValueError(f"No feasible setting: seed{seed}/{model}/{task}/{method}")
                    scores[method].append(finite(winner["score"]))
            for method, values in scores.items():
                rows.append(dict(model=model, task=task, method=method, score=mean(values)))
    return rows


def coordinates(results):
    """Report overlap and squared-coefficient mass as percentages."""
    rows = []
    for model in MODELS:
        for task in TASKS[:3]:
            data = [read_case(results, s, model, task, "coordinate_statistics.json") for s in SEEDS]
            if len({(d["k"], d["width"]) for d in data}) != 1:
                raise ValueError(f"Coordinate counts differ across seeds: {model}/{task}")
            row = dict(model=model, task=task, k=data[0]["k"], width=data[0]["width"])
            for key in ("selected_squared_coefficient_mass", "top_k_squared_coefficient_mass"):
                row[key] = 100 * mean(finite(d[key]) for d in data)
            for key in ("top_k_coefficients", "top_k_ie", "gradient", "ie"):
                row[key] = 100 * mean(finite(d["overlaps"][key]) for d in data)
            rows.append(row)
    return rows


def capability(results):
    """Average per-seed absolute changes; take signed ranges over all cases."""
    deltas = {(m, b): {s: [] for s in SEEDS} for m in METHODS for b in BENCHMARKS}
    baseline = {b: [] for b in BENCHMARKS}
    for seed in SEEDS:
        for model in MODELS:
            for task in TASKS[:3]:
                data = read_case(results, seed, model, task, "capability.json")
                for benchmark in BENCHMARKS:
                    scale = 1 if benchmark == "wikitext" else 100
                    metric = "nll" if benchmark == "wikitext" else "accuracy"
                    baseline[benchmark].append(
                        scale * finite(data["unsteered"][benchmark]["summary"][metric])
                    )
                    for method in METHODS:
                        record = data["methods"][method][benchmark]
                        if record["status"] != "complete":
                            raise ValueError(f"Incomplete capability measurement: {seed}/{model}/{task}/{method}/{benchmark}")
                        deltas[method, benchmark][seed].append(scale * finite(record["paired"]["delta"]))
    rows = []
    for benchmark in BENCHMARKS:
        rows.append(dict(method="unsteered", benchmark=benchmark, mean=mean(baseline[benchmark]), minimum=None, maximum=None))
    for method in METHODS:
        for benchmark in BENCHMARKS:
            per_seed = deltas[method, benchmark]
            values = [v for seed in SEEDS for v in per_seed[seed]]
            rows.append(dict(method=method, benchmark=benchmark,
                             mean=mean(mean(abs(v) for v in per_seed[s]) for s in SEEDS),
                             minimum=min(values), maximum=max(values)))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # Validate all inputs before writing any table.
    tables = {"comparison.csv": comparison(args.results),
              "coordinates.csv": coordinates(args.results),
              "capability.csv": capability(args.results)}
    args.output.mkdir(parents=True, exist_ok=True)
    for filename, rows in tables.items():
        table(args.output / filename, rows)
    print("Wrote four-seed tables to", args.output)


if __name__ == "__main__":
    main()
