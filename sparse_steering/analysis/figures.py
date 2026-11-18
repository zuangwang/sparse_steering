"""Build comparison tables and figures from strength-sweep outputs."""

import argparse
import csv
import json
from pathlib import Path

NAMES = dict(
    dense="dense baseline",
    normalized_dense="Normalized dense baseline",
    pca_dense="PCA (dense)",
    pca_sparse="PCA (sparse)",
    magnitude="Magnitude",
    gradient="Gradient",
    ie="IE",
    ours="our method",
    sparsification_only="Optimized sparsification only",
    sparsification_global_norm="Optimized sparsification with global normalization",
)
METHODS = tuple(NAMES)[:8]
MODELS = ("gemma", "qwen", "llama", "qwen7")
TASKS = ("truthfulqa", "halueval", "paradetox", "rtp")


def read(p):
    return json.loads(p.read_text())


def table(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    source = a.results / f"seed{a.seed}"
    cases = {(f.parent.parent.name, f.parent.name): read(f) for f in source.glob("*/*/test.json")}
    if not cases:
        raise ValueError("No test.json files under " + str(source))
    rows = []
    margins = []
    for (model, task), data in sorted(cases.items()):
        rows.append(
            dict(
                model=model,
                task=task,
                method="Unsteered",
                score=data["unsteered"]["score"],
                edit_norm_ratio_squared=0,
            )
        )
        dense = max(
            (
                data["selected"][m]["score"]
                for m in ("dense", "normalized_dense", "pca_dense")
                if data["selected"].get(m)
            ),
            default=None,
        )
        for method in METHODS:
            v = data["selected"].get(method)
            if v:
                rows.append(
                    dict(
                        model=model,
                        task=task,
                        method=NAMES[method],
                        score=v["score"],
                        edit_norm_ratio_squared=v["edit_norm_ratio_squared"],
                    )
                )
                if dense is not None:
                    margins.append(
                        dict(
                            model=model, task=task, method=NAMES[method], margin=v["score"] - dense
                        )
                    )
    table(a.output / "comparison.csv", rows)
    table(a.output / "dense_margin.csv", margins)
    for stem, models, tasks, methods, xkey, ykey in [
        (
            "main_result",
            ("gemma", "qwen", "llama", "qwen32b"),
            TASKS,
            ("dense", "ours"),
            "score",
            "nll_ratio",
        ),
        (
            "norm_strength",
            MODELS,
            TASKS[:3],
            (
                "dense",
                "sparsification_only",
                "normalized_dense",
                "sparsification_global_norm",
                "ours",
            ),
            "alpha",
            "nll_ratio",
        ),
    ]:
        fig, axes = plt.subplots(len(tasks), len(models), figsize=(13, 9), squeeze=False)
        for i, task in enumerate(tasks):
            for j, model in enumerate(models):
                ax = axes[i, j]
                d = cases.get((model, task))
                ax.set_title(model + " / " + task)
                ax.set_xlabel(xkey)
                ax.set_ylabel("NLL ratio")
                ax.set_yscale("log")
                ax.axhline(1.15, color="grey", linestyle="-.")
                if d:
                    for method in methods:
                        pts = [
                            v
                            for v in d["grids"].get(method, [])
                            if v.get(xkey) is not None
                            and v.get(ykey) is not None
                            and (xkey != "alpha" or v[xkey] > 0)
                        ]
                        ax.plot([v[xkey] for v in pts], [v[ykey] for v in pts], label=NAMES[method])
                if xkey == "alpha":
                    ax.set_xscale("log")
        handles, labels = next(
            (
                ax.get_legend_handles_labels()
                for ax in axes.flat
                if ax.get_legend_handles_labels()[0]
            ),
            ([], []),
        )
        fig.legend(handles, labels, loc="lower center", ncol=3)
        fig.tight_layout(rect=(0, 0.08, 1, 1))
        fig.savefig(a.output / (stem + ".pdf"))
        plt.close(fig)
    fig, axes = plt.subplots(1, 4, figsize=(13, 3), squeeze=False)
    for ax, task in zip(axes.flat, TASKS):
        subset = [
            r
            for r in margins
            if r["task"] == task
            and r["method"]
            in [NAMES[m] for m in ("pca_sparse", "magnitude", "gradient", "ie", "ours")]
        ]
        for r in subset:
            ax.scatter(r["method"], r["margin"], label=r["model"])
        ax.axhline(0, color="grey")
        ax.set_title(task)
        ax.tick_params(axis="x", labelrotation=60)
    fig.tight_layout()
    fig.savefig(a.output / "dense_margin.pdf")
    plt.close(fig)
    fig, axes = plt.subplots(3, 4, figsize=(13, 9))
    for i, task in enumerate(TASKS[:3]):
        for j, model in enumerate(MODELS):
            ax = axes[i, j]
            ax.set_title(model + " / " + task)
            ax.set_yscale("log")
            ax.axhline(1.15, color="grey", linestyle="-.")
            for rho in (10, 20, 40, 80):
                f = (
                    source
                    / model
                    / task
                    / (f"proportion/rho{rho}.json" if rho != 20 else "test.json")
                )
                if not f.exists():
                    continue
                pts = [
                    v
                    for v in read(f)["grids"]["ours"]
                    if v.get("score") is not None and v.get("nll_ratio") is not None
                ]
                ax.plot([v["score"] for v in pts], [v["nll_ratio"] for v in pts], label=f"{rho}%")
            ax.set_xlabel("Score (%)")
            ax.set_ylabel("NLL ratio")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(a.output / "proportion_sweeps.pdf")
    plt.close(fig)
    capability = []
    coordinates = []
    stability = []
    for f in source.glob("*/*/capability.json"):
        d = read(f)
        for method, benchmarks in d["methods"].items():
            for benchmark, v in benchmarks.items():
                pair = v.get("paired", {})
                capability.append(
                    dict(
                        model=f.parent.parent.name,
                        task=f.parent.name,
                        method=method,
                        benchmark=benchmark,
                        status=v["status"],
                        paired=json.dumps(pair, sort_keys=True),
                    )
                )
    for f in source.glob("*/*/coordinate_statistics.json"):
        d = read(f)
        coordinates.append(
            dict(
                model=f.parent.parent.name,
                task=f.parent.name,
                k=d["k"],
                selected_mass=d["selected_squared_coefficient_mass"],
                top_k_mass=d["top_k_squared_coefficient_mass"],
                **d["overlaps"],
            )
        )
    for f in source.glob("*/*/stability.json"):
        d = read(f)
        stability.append(
            dict(
                model=f.parent.parent.name,
                task=f.parent.name,
                mean=d["pairwise_mean"],
                minimum=d["pairwise_range"][0],
                maximum=d["pairwise_range"][1],
            )
        )
    table(a.output / "capability.csv", capability)
    table(a.output / "coordinates.csv", coordinates)
    table(a.output / "stability.csv", stability)
    if coordinates:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        labels = [r["model"] + "/" + r["task"] for r in coordinates]
        for key in ("top_k_coefficients", "top_k_ie", "gradient", "ie"):
            axes[0].plot(labels, [r[key] for r in coordinates], marker="o", label=key)
        for key in ("selected_mass", "top_k_mass"):
            axes[1].plot(labels, [r[key] for r in coordinates], marker="o", label=key)
        for ax in axes:
            ax.tick_params(axis="x", labelrotation=90)
            ax.legend()
        fig.tight_layout()
        fig.savefig(a.output / "selected_coordinates.pdf")
        plt.close(fig)
    if stability:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.errorbar(
            [r["model"] + "/" + r["task"] for r in stability],
            [r["mean"] for r in stability],
            yerr=[
                [r["mean"] - r["minimum"] for r in stability],
                [r["maximum"] - r["mean"] for r in stability],
            ],
            fmt="o",
        )
        ax.tick_params(axis="x", labelrotation=90)
        fig.tight_layout()
        fig.savefig(a.output / "mask_stability.pdf")
        plt.close(fig)
    print("Wrote tables and figures to", a.output)


if __name__ == "__main__":
    main()
