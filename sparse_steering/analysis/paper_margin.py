"""Sparse methods' margins over their corresponding dense methods."""

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from .paper_style import MODEL_NAMES, rc_params


MODELS = ["gemma", "qwen", "llama", "qwen7"]
TASKS = ["TruthfulQA", "HaluEval", "ParaDetox", "RTP"]
# (row label in Table 1, tick label, row label of its corresponding dense method)
SPARSE = [
    ("PCA (sparse)", "PCA (sparse)", "PCA (dense)"),
    ("Magnitude", "Magnitude", "Dense baseline"),
    ("Gradient", "Gradient", "Dense baseline"),
    ("IE", "IE", "Dense baseline"),
    ("Our method", "Ours", "Dense baseline"),
]
MARKERS = dict(zip(MODELS, "os^D"))
GREY, BLUE, DARK, GRID = "#7d838a", "#2a78d6", "#1f2328", "#e6e6e3"
WIDTH_PT, HEIGHT_PT = 468, 162
AXIS_LABEL = "Score \u2212 corresponding dense method (%)"
YLIM, XLIM = (-4, 18.6), (-0.6, 4.6)
OFFSETS = (-0.195, -0.065, 0.065, 0.195)


def margins(scores):
    """{tick label: {task: [margin per model]}} in percentage points."""
    out = {}
    for name, label, dense in SPARSE:
        diff = [a - b for a, b in zip(scores[name], scores[dense])]
        out[label] = {
            task: [diff[4 * m + t] for m in range(4)] for t, task in enumerate(TASKS)
        }
    return out


def draw(data):
    style = rc_params()
    style.update(
        {
            "text.color": "black",
            "axes.labelcolor": "black",
            "axes.facecolor": "white",
            "axes.edgecolor": "black",
            "axes.linewidth": 0.6,
            "xtick.color": "black",
            "ytick.color": "black",
            "xtick.major.size": 3.5,
            "ytick.major.size": 3.5,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
        }
    )
    with plt.rc_context(style):
        fig = plt.figure(figsize=(WIDTH_PT / 72, HEIGHT_PT / 72))
        for index, task in enumerate(TASKS):
            ax = fig.add_axes(
                [
                    (37.2 + 113.3 * index) / WIDTH_PT,
                    (HEIGHT_PT - 114.3) / HEIGHT_PT,
                    81.8 / WIDTH_PT,
                    81 / HEIGHT_PT,
                ]
            )
            for level in (0, 10):
                ax.axhline(level, color=GRID, linewidth=0.5, zorder=1)
            ax.axhline(0, color=DARK, linewidth=0.7, linestyle=(0, (4, 3)), zorder=1.5)
            for position, (_, label, _) in enumerate(SPARSE):
                values = data[label][task]
                colour = BLUE if label == "Ours" else GREY
                ax.plot(
                    [position - 0.3, position + 0.3],
                    [sum(values) / 4] * 2,
                    linewidth=1.6,
                    color=BLUE if label == "Ours" else DARK,
                    solid_capstyle="butt",
                    zorder=2,
                )
                for model, offset, value in zip(MODELS, OFFSETS, values):
                    ax.plot(
                        position + offset,
                        value,
                        linestyle="none",
                        marker=MARKERS[model],
                        markersize=4.2,
                        markerfacecolor=colour,
                        markeredgecolor="white",
                        markeredgewidth=0.6,
                        zorder=3,
                        clip_on=False,
                    )
            ax.set_xlim(XLIM)
            ax.set_ylim(YLIM)
            ax.set_xticks(
                range(len(SPARSE)),
                [row[1] for row in SPARSE],
                fontsize=7.5,
                rotation=40,
                ha="right",
            )
            ours = ax.get_xticklabels()[-1]
            ours.set_color(BLUE)
            ours.set_fontweight("bold")
            ax.set_yticks([0, 10])
            ax.tick_params(axis="y", labelsize=8.5, labelleft=index == 0)
            ax.set_title(task, fontsize=9, loc="center", pad=6)
            if index == 0:
                ax.set_ylabel(AXIS_LABEL, fontsize=7.5, labelpad=3)
        handles = [
            Line2D(
                [],
                [],
                linestyle="none",
                marker=MARKERS[m],
                markersize=5,
                markerfacecolor=GREY,
                markeredgecolor="white",
                markeredgewidth=1,
                label=MODEL_NAMES[m],
            )
            for m in MODELS
        ]
        handles.append(
            Line2D([], [], color=DARK, linewidth=1.6, label="Mean over models")
        )
        fig.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.006),
            ncol=5,
            fontsize=7.5,
            handlelength=1.9,
            handletextpad=0.6,
            columnspacing=2.3,
            borderaxespad=0,
            frameon=False,
        )
    return fig
