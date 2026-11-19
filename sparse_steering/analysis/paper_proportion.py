"""Selected-coordinate proportion sweeps, with a common print layout."""

import math
from . import paper_style as fs

MODELS = ("gemma", "qwen", "llama", "qwen7")
TASKS = tuple(fs.DATASET_NAMES)
PROPORTIONS = (10, 20, 40, 80)
MARK_ORDER = (20, 10, 40, 80)
STYLES = {
    10: dict(marker="o"),
    20: dict(marker="^"),
    40: dict(marker="s"),
    80: dict(marker="D"),
}
SIZE_PT = (468, 270)
TOP_PT, ROW_GAP_PT, BOTTOM_PT, XLABEL_DROP_PT = 28.8, 34.1, 30.0, 22
X_LABEL = {
    "truthfulqa": "Accuracy (%)",
    "halueval": "Accuracy (%)",
    "paradetox": "Non-toxic rate (%)",
}
MARKER_SIZE = {"^": 2.6, "o": 2.3, "s": 2.3, "D": 2.33}
MAIN_WIDTH, OTHER_WIDTH = 1.6, 0.9
COLOR = {10: "#57a943", 20: fs.PINS_BLUE, 40: "#a17ee4", 80: "#ef869b"}


def render(sweeps):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator, NullFormatter, NullLocator

    fs.apply_style()
    # Symbols use the paper's math font (Computer Modern), not the STIX default of the style.
    matplotlib.rcParams["mathtext.fontset"] = "cm"
    fig, axes = plt.subplots(3, 4, figsize=tuple(v / 72 for v in SIZE_PT), dpi=220)
    panel_pt = (SIZE_PT[1] - TOP_PT - BOTTOM_PT - 2 * ROW_GAP_PT) / 3
    fig.subplots_adjust(
        left=0.13,
        right=0.985,
        top=1 - TOP_PT / SIZE_PT[1],
        bottom=BOTTOM_PT / SIZE_PT[1],
        wspace=0.42,
        hspace=ROW_GAP_PT / panel_pt,
    )
    for index, ax in enumerate(axes.flat):
        task, model = TASKS[index // 4], MODELS[index % 4]
        name = f"{model}/{task}"
        fs.panel_frame(ax)
        series = sweeps[name]
        points = [p for grid in series.values() for p in grid]
        low, high = min(p["nll_ratio"] for p in points), max(
            p["nll_ratio"] for p in points
        )
        span = max(math.log(high / low), math.log(1.25))
        ax.set_yscale("log")
        ax.set_ylim(low * math.exp(-0.13 * span), high * math.exp(0.16 * span))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_minor_formatter(NullFormatter())
        for rho in (10, 40, 80, 20):
            style, grid = STYLES[rho], series[rho]
            ax.plot(
                [p["score"] for p in grid],
                [p["nll_ratio"] for p in grid],
                color=COLOR[rho],
                linewidth=MAIN_WIDTH if rho == 20 else OTHER_WIDTH,
                zorder=4 if rho == 20 else 3,
            )
        low, high = min(p["score"] for p in points), max(p["score"] for p in points)
        span = max(high - low, 5)
        ax.set_xlim(low - 0.13 * span, high + 0.16 * span)
        low, high = ax.get_xlim()
        ax.set_xticks(
            [
                v
                for v in MaxNLocator(nbins=3, steps=[1, 2, 2.5, 5, 10]).tick_values(
                    low, high
                )
                if low <= v <= high
            ]
        )
        occupied = []
        for rho in MARK_ORDER:
            style, grid = STYLES[rho], series[rho]
            keep = fs.spaced_markers(
                ax,
                [p["score"] for p in grid],
                [p["nll_ratio"] for p in grid],
                occupied=occupied,
            )
            # Markers as in Figure 1: filled in the series colour over a white rim.
            for edge, width in (("white", 0.9), (COLOR[rho], 0.45)):
                ax.plot(
                    [grid[i]["score"] for i in keep],
                    [grid[i]["nll_ratio"] for i in keep],
                    linestyle="none",
                    color=COLOR[rho],
                    marker=style["marker"],
                    markersize=MARKER_SIZE[style["marker"]],
                    markeredgecolor=edge,
                    markeredgewidth=width,
                    markerfacecolor=COLOR[rho],
                    zorder=5,
                )
        ax.grid(axis="y")
        ax.tick_params(pad=2)
        ax.set_ylabel("NLL ratio", fontsize=7.5, labelpad=1.5)
        if index < 4:
            fs.panel_title(ax, fs.MODEL_NAMES[model], fontsize=8)
        if index % 4 == 0:
            box = ax.get_position()
            fig.text(
                0.018,
                (box.y0 + box.y1) / 2,
                fs.DATASET_NAMES[task],
                rotation=90,
                ha="center",
                va="center",
                fontsize=8,
            )
            # One x-axis label per row, placed as in Figure 1.
            fig.text(
                0.55,
                box.y0 - XLABEL_DROP_PT / SIZE_PT[1],
                X_LABEL[task],
                ha="center",
                va="center",
                fontsize=8,
            )
        fs.plain_log_ticks(ax.yaxis)
    handles = [
        Line2D(
            [],
            [],
            label=rf"$\rho = {r}\%$",
            color=COLOR[r],
            marker=STYLES[r]["marker"],
            linewidth=MAIN_WIDTH if r == 20 else OTHER_WIDTH,
            markersize=3.0,
            markeredgewidth=0.7,
            markerfacecolor=COLOR[r],
        )
        for r in PROPORTIONS
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.54, 0.99),
        ncol=4,
        handlelength=1.4,
        columnspacing=1.5,
        handletextpad=0.5,
        borderaxespad=0,
    )
    return fig
