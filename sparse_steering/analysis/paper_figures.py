"""Load four-seed measurements and render the paper's seven data figures.

Only saved JSON measurements are read. Scores remain percentages here; the
main-result renderer receives fractions to match its percentage-axis geometry.
"""

import math
from collections import defaultdict
from pathlib import Path

from reproduce.average import SEEDS, comparison, read_case
from .figures import MODELS, TASKS

BUDGETS = (
    0,
    1e-4,
    4e-4,
    1e-3,
    2.5e-3,
    5e-3,
    1e-2,
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
NORM_METHODS = (
    "dense",
    "normalized_dense",
    "sparsification_global_norm",
    "sparsification_only",
    "ours",
)


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def mean(values):
    values = list(values)
    if not values:
        raise ValueError("Cannot average an empty set of measurements")
    return sum(values) / len(values)


def curve_points(data, method):
    """Index finite, available measurements on the paper's common strength grid."""
    points = {}
    for point in data["grids"][method]:
        budget = point["edit_norm_ratio_squared"]
        if not finite(budget):
            continue
        key = min(BUDGETS, key=lambda b: abs(b - budget))
        if abs(key - budget) >= 1e-9:
            continue
        if point.get("status", "complete") != "complete" or not point.get(
            "available", True
        ):
            continue
        if not all(finite(point.get(k)) for k in ("score", "nll_ratio")):
            continue
        points[key] = (point["score"], point["nll_ratio"], point.get("alpha"))
    baseline = data["unsteered"]["score"]
    if not finite(baseline):
        raise ValueError("Nonfinite unsteered score")
    points[0] = (baseline, 1.0, 0.0)
    return points


def average_curve(seed_curves):
    """Mean each grid point over its available seeds; never interpolate gaps.

    Alpha is averaged over finite alpha measurements among those same seeds.
    The count records curve coverage, independently of alpha availability.
    """
    out = []
    for budget in BUDGETS:
        have = [c[budget] for c in seed_curves if budget in c]
        have = [p for p in have if finite(p[0]) and finite(p[1])]
        if not have:
            continue
        alphas = [p[2] for p in have if finite(p[2])]
        out.append(
            dict(
                budget=budget,
                n=len(have),
                score=mean(p[0] for p in have),
                nll_ratio=mean(p[1] for p in have),
                alpha=mean(alphas) if alphas else None,
            )
        )
    return out


def average_trajectories(seed_records):
    """Average per (model, task, rerun, step), retaining steps 1 through 96."""
    groups = defaultdict(list)
    for records in seed_records:
        seen = set()
        for row in records:
            step = int(row["step"])
            if not 1 <= step <= 96:
                continue
            key = row["model"], row["dataset"], int(row["rerun"]), step
            if key in seen:
                raise ValueError(f"Duplicate trajectory point in one seed: {key}")
            seen.add(key)
            value = row["overlap_final"]
            if not finite(value) or not 0 <= value <= 1:
                raise ValueError(f"Invalid trajectory overlap: {key}")
            groups[key].append(value)
    return [
        dict(model=m, dataset=t, restart=r, step=s, overlap_final=mean(v))
        for (m, t, r, s), v in groups.items()
    ]


def load_data(results):
    """Read the bundled layout, including the additional 32B main-result cases."""
    results = Path(results)
    selected = {
        (r["model"], r["task"], r["method"]): r["score"] for r in comparison(results)
    }
    curves, proportions, coordinates = {}, {}, {}
    stability, trajectories = [], [[] for _ in SEEDS]
    for model in (*MODELS, "qwen32b"):
        for task in TASKS:
            name = f"{model}/{task}"
            seeds = [read_case(results, s, model, task, "test.json") for s in SEEDS]
            methods = (
                ("ours", "dense")
                if model == "qwen32b" or task == "rtp"
                else NORM_METHODS
            )
            curves[name] = {
                m: average_curve([curve_points(d, m) for d in seeds]) for m in methods
            }
            if model == "qwen32b" or task == "rtp":
                continue
            proportions[name] = {}
            for rho in (10, 20, 40, 80):
                data = (
                    seeds
                    if rho == 20
                    else [
                        read_case(results, s, model, task, f"proportion/rho{rho}.json")
                        for s in SEEDS
                    ]
                )
                proportions[name][rho] = [
                    p
                    for p in average_curve([curve_points(d, "ours") for d in data])
                    if p["budget"] <= 0.36
                ]
            stats = [
                read_case(results, s, model, task, "coordinate_statistics.json")
                for s in SEEDS
            ]
            if len({(d["k"], d["width"]) for d in stats}) != 1:
                raise ValueError(f"Coordinate counts differ across seeds: {name}")
            coordinates[name] = dict(
                mag=mean(100 * d["overlaps"]["top_k_coefficients"] for d in stats),
                ie=mean(100 * d["overlaps"]["top_k_ie"] for d in stats),
                selected_squared_mass_pct=mean(
                    100 * d["selected_squared_coefficient_mass"] for d in stats
                ),
                top_k_squared_mass_pct=mean(
                    100 * d["top_k_squared_coefficient_mass"] for d in stats
                ),
            )
            values = defaultdict(list)
            for seed in SEEDS:
                data = read_case(results, seed, model, task, "stability.json")
                pairs = [p["overlap"] for p in data["pairs"]]
                if len(pairs) != 3 or not all(finite(v) and 0 <= v <= 1 for v in pairs):
                    raise ValueError(f"Expected three rerun-pair overlaps: {name}")
                for key, value in [
                    ("mean_overlap", mean(pairs)),
                    ("min_overlap", min(pairs)),
                    ("max_overlap", max(pairs)),
                ]:
                    values[key].append(value)
                trajectories[seed].extend(
                    dict(r, model=model, dataset=task) for r in data["trajectories"]
                )
            stability.append(
                dict(
                    case=name,
                    chance=stats[0]["k"] / stats[0]["width"],
                    **{k: mean(v) for k, v in values.items()},
                )
            )
    return (
        curves,
        proportions,
        coordinates,
        dict(cases=stability),
        average_trajectories(trajectories),
        selected,
    )


def render_all(results, output):
    """Write seven vector PDFs, with no model loading or GPU dependencies."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pymupdf
    from . import paper_coordinates, paper_main, paper_margin, paper_markers
    from . import paper_math, paper_norm, paper_proportion, paper_style

    curves, proportions, coordinates, stability, trajectory, selected = load_data(
        results
    )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    def save(name, fig):
        target = output / f"{name}.pdf"
        try:
            fig.savefig(target, metadata=paper_style.PDF_METADATA)
        finally:
            plt.close(fig)
        return target

    def main_point(p):
        return dict(p, score=p["score"] / 100)

    main = {
        n: dict(
            baseline=main_point(c["ours"][0]),
            grids={m: [main_point(p) for p in c[m]] for m in ("ours", "dense")},
        )
        for n, c in curves.items()
        if n.split("/")[0] in paper_main.MODELS
    }
    choices = {
        n: {
            m: dict(score=selected[(*n.split("/"), m)] / 100) for m in ("ours", "dense")
        }
        for n in main
    }
    target = save("main_result", paper_main.render(main, choices))
    with pymupdf.open(target) as doc:
        if paper_markers.legend_cross(doc, doc[0]):
            data = doc.tobytes(garbage=3, deflate=True)
        else:
            data = None
    if data:
        target.write_bytes(data)

    norm = {
        n: dict(grids={m: {p["budget"]: p for p in c[m]} for m in NORM_METHODS})
        for n, c in curves.items()
        if n in proportions
    }

    def strength(name, method, budget):
        return norm[name]["grids"][method][budget]["alpha"] or math.nan

    target = save("norm_strength", paper_norm.render(norm, strength))
    with pymupdf.open(target) as doc:
        paper_markers.filled_markers(doc, doc[0])
        data = doc.tobytes(garbage=3, deflate=True)
    target.write_bytes(data)
    paper_math.retypeset("norm_strength", target)

    save("proportion_sweeps", paper_proportion.render(proportions))
    labels = {
        "PCA (sparse)": "pca_sparse",
        "PCA (dense)": "pca_dense",
        "Magnitude": "magnitude",
        "Gradient": "gradient",
        "IE": "ie",
        "Our method": "ours",
        "Dense baseline": "dense",
    }
    # Margins between the scores as printed in Table 1 (two decimals).
    scores = {
        label: [round(selected[m, t, method], 2) for m in MODELS for t in TASKS]
        for label, method in labels.items()
    }
    fig = paper_margin.draw(paper_margin.margins(scores))
    with plt.rc_context(paper_style.rc_params()):
        save("dense_margin", fig)
    save("selection_overlap", paper_coordinates.render_selection(coordinates))
    target = save(
        "selected_coordinate_statistics", paper_coordinates.render_mass(coordinates)
    )
    paper_math.retypeset("selected_coordinate_statistics", target)
    target = save(
        "mask_seed_overlap", paper_coordinates.render_stability(stability, trajectory)
    )
    paper_math.retypeset("mask_seed_overlap", target)
