"""NLL versus mean raw strength, for five normalization variants."""

import math
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter, NullLocator
from . import paper_style as fs

MODELS = ("gemma", "qwen", "llama", "qwen7")
DATASETS = tuple(fs.DATASET_NAMES)
PINS, GLOBAL = "ours", "sparsification_global_norm"
METHODS = ("dense", "normalized_dense", GLOBAL, "sparsification_only", PINS)
MAIN_STYLES = {
    GLOBAL: dict(
        label="Global normalization", color="#c7ad38", marker="s", linestyle="-."
    )
}
LABELS = dict(
    zip(
        METHODS,
        (
            "Dense baseline",
            "Normalized dense baseline",
            "Optimized sparsification + global normalization",
            "Optimized sparsification only",
            "Ours",
        ),
    )
)
SIZE_PT = (468, 280)
X_LABEL = r"Strength $\alpha$"
NLL_RATIO_LIMIT = 1.15


def plot_budgets(name, method, grid):
    return sorted(b for b in grid if b)


def log_ticks(low, high, count=3):
    """Readable ordinary decimal ticks without offset or minor labels."""
    powers = list(range(math.ceil(math.log10(low)), math.floor(math.log10(high)) + 1))
    if len(powers) > count:
        powers = [
            powers[round(i * (len(powers) - 1) / (count - 1))] for i in range(count)
        ]
    ticks = [10.0**e for e in powers]
    return ticks, [
        f"{v:g}" if v < 1e6 else f"{v/1e6:g}M" if v < 1e9 else f"{v/1e9:g}B"
        for v in ticks
    ]


def x_geometry(name, grids, strength):
    xs = [strength(name, m, b) for m, grid in grids.items() for b in grid if b]
    xs = [x for x in xs if math.isfinite(x) and x > 0]
    low, high = min(xs), max(xs)
    span = math.log(high / low)
    return (
        low * math.exp(-0.08 * span),
        high * math.exp(0.32 * span),
        high * math.exp(0.10 * span),
        high * math.exp(0.15 * span),
    )


def ax_line(ax, name, method):
    (curve,) = [line for line in ax.lines if line.get_gid() == f"{name}:{method}:curve"]
    return curve


def marked(ax, name):
    """Marker indices of each curve, ours first: crowded points keep only the line."""
    occupied = []
    return {
        method: fs.spaced_markers(
            ax,
            ax_line(ax, name, method).get_xdata(),
            ax_line(ax, name, method).get_ydata(),
            occupied=occupied,
        )
        for method in (PINS, *[m for m in METHODS if m != PINS])
    }


def render(cases, strength):
    """The main figure contains all twelve cases; no separate appendix plot."""
    fs.apply_style()
    fig, axes = plt.subplots(
        3, 4, figsize=tuple(v / 72 for v in SIZE_PT), dpi=220, sharex=False, sharey=True
    )
    # Figure 1 layout: an NLL ratio label on every panel and a strength label under every row.
    fig.subplots_adjust(
        left=0.13,
        right=0.955,
        bottom=30 / SIZE_PT[1],
        top=0.845,
        wspace=0.38,
        hspace=0.70,
    )
    all_ratios = [
        p["nll_ratio"]
        for c in cases.values()
        for g in c["grids"].values()
        for p in g.values()
    ]
    low, high = min(all_ratios), max(all_ratios)
    log_span = math.log(high / low)
    # Physical padding keeps the full marker envelopes inside the short panels.
    ylim = (low * math.exp(-0.08 * log_span), high * math.exp(0.08 * log_span))
    styles = {
        m: dict(fs.PRIMARY_STYLE[m] if m != GLOBAL else MAIN_STYLES[m], label=LABELS[m])
        for m in METHODS
    }
    for row, task in enumerate(DATASETS):
        for col, model in enumerate(MODELS):
            ax = axes[row, col]
            name = f"{model}/{task}"
            grids = cases[name]["grids"]
            fs.panel_frame(ax)
            for method in METHODS:  # PinS last, with figure_style's thicker stroke.
                budgets = plot_budgets(name, method, grids[method])
                (curve,) = ax.plot(
                    [strength(name, method, b) for b in budgets],
                    [
                        (
                            grids[method][b]["nll_ratio"]
                            if b in grids[method]
                            else math.nan
                        )
                        for b in budgets
                    ],
                    **styles[method],
                    **fs.method_line(method),
                )
                curve.set_gid(f"{name}:{method}:curve")
            ax.set_xscale("log")
            ax.set_yscale("log")
            xmin, xmax, bracket_x, label_x = x_geometry(name, grids, strength)
            ax.set_xlim(xmin, xmax)
            ax.tick_params(labelbottom=True)
            ax.set_ylim(*ylim)
            ax.set_xticks(*log_ticks(xmin, xmax))
            ax.set_yticks([1, 3, 10, 30], ["1", "3", "10", "30"])
            for axis in (ax.xaxis, ax.yaxis):
                axis.set_minor_locator(NullLocator())
                axis.set_minor_formatter(NullFormatter())
            ax.tick_params(pad=2, labelleft=True)
            ax.set_ylabel("NLL ratio", fontsize=7.5, labelpad=1.5)
            for method, shown in marked(ax, name).items():
                ax_line(ax, name, method).set_markevery(shown)
            ax.grid(axis="y")
            limit = ax.axhline(
                NLL_RATIO_LIMIT,
                color=fs.REFERENCE,
                linewidth=0.65,
                linestyle="-.",
                zorder=1,
            )
            limit.set_gid(f"{name}:limit")
            if row == 0:
                fs.panel_title(ax, fs.MODEL_NAMES[model], fontsize=8)
            if col == 0:
                box = ax.get_position()
                label = fig.text(
                    0.018,
                    (box.y0 + box.y1) / 2,
                    fs.DATASET_NAMES[task],
                    rotation=90,
                    ha="center",
                    va="center",
                    fontsize=8,
                )
                label.set_gid(f"{task}:row_label")
                fig.text(
                    0.5425,
                    box.y0 - 22 / SIZE_PT[1],
                    X_LABEL,
                    ha="center",
                    va="center",
                    fontsize=8,
                ).set_gid(f"{task}:strength_label")
            low = grids[PINS][4.0]["nll_ratio"]
            high = min(grids[m][4.0]["nll_ratio"] for m in METHODS if m != PINS)
            assert high > low
            x, cap = bracket_x, bracket_x * 0.08
            (bracket,) = ax.plot(
                [x - cap, x, x, x - cap],
                [low, low, high, high],
                color=fs.INK,
                linewidth=0.6,
                zorder=7,
            )
            bracket.set_gid(f"{name}:bracket")
            # Multiplicative separation: geometric midpoint on the log axis.
            label = ax.text(
                label_x,
                math.sqrt(low * high),
                f"{high/low:.1f}x",
                fontsize=7,
                ha="left",
                va="center",
                color=fs.INK,
            )
            label.set_gid(f"{name}:bracket_label")
    handles = [
        Line2D(
            [],
            [],
            **styles[m],
            linewidth=fs.method_line(m)["linewidth"],
            markersize=3,
            markerfacecolor="white",
        )
        for m in METHODS
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.54, 0.995),
        ncol=3,
        fontsize=7.5,
        columnspacing=1.1,
        handlelength=1.7,
        borderaxespad=0,
    )
    return fig
