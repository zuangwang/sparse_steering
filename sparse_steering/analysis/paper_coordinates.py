"""Coordinate overlap, coefficient mass and mask stability panels."""

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from . import paper_style as fs

MODELS = ("gemma", "qwen", "llama", "qwen7")
TASKS = DATASETS = tuple(fs.DATASET_NAMES)
PINS = "ours"
STYLES = {PINS: fs.PRIMARY_STYLE[PINS]}
COLORS = fs.TASK_COLORS
MAG, ATT = "#9a803c", "#4c8587"


def render_selection(coordinates):
    fs.apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.5), sharey=True)
    fig.subplots_adjust(left=0.105, right=0.99, bottom=0.25, top=0.72, wspace=0.18)
    for ax, task in zip(axes, TASKS):
        for i, m in enumerate(MODELS):
            for key, off, col in (("mag", -0.17, MAG), ("ie", 0.17, ATT)):
                v = coordinates[f"{m}/{task}"][key]
                ax.bar(i + off, v, width=0.3, color=col, linewidth=0)
        ax.axhline(20, color=fs.REFERENCE, linestyle="--", linewidth=0.8, zorder=3)
        ax.set_xticks(range(4), [fs.MODEL_NAMES[m] for m in MODELS])
        plt.setp(ax.get_xticklabels(), rotation=50, ha="center")
        ax.set_xlim(-0.5, 3.5)
        ax.set_ylim(0, 32)
        ax.set_yticks([0, 10, 20, 30])
        ax.tick_params(axis="x", length=0)
        ax.grid(axis="y", linewidth=0.5)
        fs.panel_title(ax, fs.DATASET_NAMES[task], fontsize=8)
    fig.supylabel("Coordinate overlap with our method (%)", x=0.03, fontsize=8)
    fig.legend(
        handles=[
            Patch(facecolor=MAG, label="Also selected by magnitude selection"),
            Patch(facecolor=ATT, label="Also selected by IE selection"),
            Line2D(
                [],
                [],
                color=fs.REFERENCE,
                ls="--",
                lw=0.8,
                label="Random selection (20%)",
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=3,
        handlelength=1.5,
        handletextpad=0.5,
        borderaxespad=0,
    )
    return fig


def render_mass(coordinates):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    fs.apply_style()
    fig, axes = plt.subplots(
        1, 3, figsize=(fs.TEXT_WIDTH, 2.75), sharex=True, sharey=True
    )
    fig.subplots_adjust(
        left=0.14, right=0.985, top=0.84, bottom=53 / (2.75 * 72), wspace=0.15
    )
    bar_values = {}
    for index, (dataset, ax) in enumerate(zip(DATASETS, axes)):
        rows = [coordinates[f"{model}/{dataset}"] for model in MODELS]
        for key, offset, color, hatch in (
            ("selected_squared_mass_pct", -0.18, STYLES[PINS]["color"], None),
            (
                "top_k_squared_mass_pct",
                0.18,
                fs.METHOD_STYLE["magnitude"]["color"],
                None,
            ),
        ):
            values = [row[key] for row in rows]
            bars = ax.barh(
                [i + offset for i in range(len(MODELS))],
                values,
                height=0.31,
                color=color,
                edgecolor=fs.INK if hatch else color,
                linewidth=0.5,
                hatch=hatch,
            )
            assert [bar.get_width() for bar in bars] == values
            bar_values[f"{dataset}/{key}"] = values
        ax.set_yticks(range(len(MODELS)), [fs.MODEL_NAMES[m] for m in MODELS])
        ax.set_ylim(len(MODELS) - 0.5, -0.5)
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 20, 50, 75, 100])
        ax.axvline(20, color=fs.REFERENCE, linestyle="--", linewidth=0.6, zorder=1)
        ax.grid(axis="x")
        fs.panel_title(ax, fs.DATASET_NAMES[dataset], chr(97 + index))
    fig.supxlabel(r"Share of $\|v\|^2$ (%)", y=23 / (2.75 * 72), fontsize=fs.LABEL_SIZE)
    legend = fig.legend(
        handles=[
            Patch(facecolor=STYLES[PINS]["color"], label="Our coordinates"),
            Patch(
                facecolor=fs.METHOD_STYLE["magnitude"]["color"],
                label=r"Largest $K$ coefficients",
            ),
            Line2D(
                [],
                [],
                color=fs.REFERENCE,
                ls="--",
                lw=0.6,
                label="Random K coordinates (expected)",
            ),
        ],
        loc="lower center",
        bbox_to_anchor=(0.53, 3 / (2.75 * 72)),
        ncol=3,
    )
    return fig


def render_stability(summary, trajectory):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    fs.apply_style()
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )
    records = {r["case"]: r for r in summary["cases"]}
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.5), sharey=False)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.73, bottom=0.21, wspace=0.60)
    for i, model in enumerate(MODELS):
        for j, task in enumerate(TASKS):
            row = records[model + "/" + task]
            x = i + (j - 1) * 0.24
            value, low, high = (
                row[k] for k in ("mean_overlap", "min_overlap", "max_overlap")
            )
            axes[0].bar(
                x,
                value,
                width=0.20,
                color=COLORS[task],
                edgecolor=fs.INK,
                linewidth=0.5,
                alpha=0.85,
                zorder=2,
            )
            axes[0].errorbar(
                x,
                value,
                yerr=[[value - low], [high - value]],
                fmt="none",
                ecolor=fs.INK,
                elinewidth=0.75,
                capsize=1.8,
                zorder=3,
            )
    axes[0].set_xticks(range(4), [fs.MODEL_NAMES[m] for m in MODELS])
    axes[0].set_xlim(-0.5, 3.5)
    axes[0].set_ylabel("Overlap |A∩B|/K", fontsize=8)
    fs.panel_title(axes[0], "Overlap between reruns", "a", fontsize=8)
    # Model colors are separate from the task palette. Dash patterns also
    # identify models when the lines meet near the final mask.
    model_styles = {
        "gemma": ("#2f3440", "-"),
        "qwen": ("#596170", "--"),
        "llama": ("#7b8494", "-."),
        "qwen7": ("#a1a8b3", ":"),
    }
    for model in MODELS:
        rows = [r for r in trajectory if r["model"] == model]
        by_key = {(r["dataset"], int(r["restart"]), int(r["step"])): r for r in rows}
        expected = {
            (task, seed, step)
            for task in TASKS
            for seed in range(3)
            for step in range(1, 97)
        }
        assert set(by_key) == expected and len(rows) == len(expected)
        means = []
        for step in range(1, 97):
            values = [
                float(by_key[task, seed, step]["overlap_final"])
                for task in TASKS
                for seed in range(3)
            ]
            assert all(0 <= v <= 1 for v in values)
            means.append(np.mean(values))
        assert abs(means[-1] - 1) < 1e-9
        color, style = model_styles[model]
        axes[1].plot(
            range(1, 97),
            means,
            color=color,
            ls=style,
            lw=1.25,
            label=fs.MODEL_NAMES[model],
            zorder=3,
        )
    # ceil(.2*d)/d differs by less than .0002 across the four widths. A single
    # reference is the equally weighted mean of the exact case-specific K/d.
    chance = float(np.mean([r["chance"] for r in records.values()]))
    for ax in axes:
        ax.axhline(chance, color=fs.REFERENCE, ls="--", lw=0.8, zorder=1)
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        ax.grid(axis="y")
    secondary = axes[0].secondary_yaxis(
        "right", functions=(lambda y: y / chance, lambda y: y * chance)
    )
    secondary.set_ylabel("× chance", fontsize=8, labelpad=3)
    secondary.set_yticks([1, 2, 3, 4], ["1×", "2×", "3×", "4×"])
    secondary.tick_params(labelsize=8, pad=2)
    axes[1].set_ylabel("Share of final coordinates selected", fontsize=8, labelpad=3)
    axes[1].axvline(48, color=fs.REFERENCE, lw=0.5, zorder=1)
    axes[1].text(
        50, 0.06, "end of epoch 1", ha="left", va="bottom", fontsize=7, color=fs.MUTED
    )
    axes[1].set_xlim(0, 96)
    axes[1].set_xticks([0, 24, 48, 72, 96])
    axes[1].set_xlabel("Training step", fontsize=8)
    fs.panel_title(axes[1], "Overlap with final mask", "b", fontsize=8)
    tasks = [
        Patch(facecolor=COLORS[t], alpha=0.85, label=fs.DATASET_NAMES[t]) for t in TASKS
    ]
    axes[0].legend(
        handles=tasks,
        loc="lower left",
        bbox_to_anchor=(-0.04, 1.18),
        ncol=3,
        fontsize=8,
        columnspacing=0.8,
        handlelength=1.1,
        borderaxespad=0,
    )
    axes[1].legend(
        loc="lower left",
        bbox_to_anchor=(-0.01, 1.18),
        ncol=2,
        fontsize=8,
        columnspacing=1.1,
        handlelength=1.7,
        borderaxespad=0,
    )
    from matplotlib.lines import Line2D

    fig.legend(
        handles=[
            Line2D(
                [], [], color=fs.REFERENCE, ls="--", lw=0.8, label="Chance K/d ≈ 20%"
            )
        ],
        loc="lower center",
        bbox_to_anchor=(0.25, 0.012),
        fontsize=8,
    )
    return fig
