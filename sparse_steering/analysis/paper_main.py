"""Main comparison: averaged curves and independently selected per-seed scores."""

import math
from decimal import Decimal
from . import paper_style as fs

MODELS = ("gemma", "qwen", "llama", "qwen32b")
NAMES = ("Gemma 2B", "Qwen 3B", "Llama 7B", "Qwen 32B")
METHODS = ("ours", "dense")
TASK_NAMES = dict(fs.DATASET_NAMES, rtp="RTP")
CASES = tuple((m, t) for t in (*fs.DATASET_NAMES, "rtp") for m in MODELS)
SIZE_PT = (468, 360)
STYLES = {
    "ours": dict(fs.PRIMARY_STYLE["ours"], label="Ours"),
    "dense": dict(fs.PRIMARY_STYLE["dense"], label="Dense baseline"),
}


def gap_label(pins, control):
    gap = Decimal(f'{100 * pins["score"]:.2f}') - Decimal(
        f'{100 * control["score"]:.2f}'
    )
    return f"{gap:+.2f}%".replace("-", "−")


def place_attached_gap_labels(fig, notes):
    """Place labels within 4 pt of their horizontal score brackets."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    px = fig.dpi / 72
    obstacles = fs.plot_obstacles(fig)
    boxes = [t.get_window_extent(renderer) for t in fs.rendered_texts(fig)]
    placements = []
    for note in notes:
        ax = note["ax"]
        midpoint, y = ax.transData.transform((note["midpoint"], note["y"]))
        lower = ax.transData.transform((note["lower"], note["y"]))[0]
        upper = ax.transData.transform((note["upper"], note["y"]))[0]
        text = fig.text(0, 0, note["label"], fontsize=7.5, color=fs.INK, zorder=11)
        text.set_gid(f'{note["case"]}:gap_label')
        chosen = None
        for dx in [0] + [sign * step / 2 for step in range(1, 201) for sign in (1, -1)]:
            x = midpoint + dx * px
            if not lower <= x <= upper:
                continue
            for gap in (2.5, 3, 2, 3.5, 3.75, 4):
                for side in ("below", "above"):
                    anchor = y + (gap if side == "above" else -gap) * px
                    text.set_va("bottom" if side == "above" else "top")
                    for alignment in ("center", "right", "left"):
                        text.set_ha(alignment)
                        text.set_position(
                            (x / fig.bbox.width, anchor / fig.bbox.height)
                        )
                        box = text.get_window_extent(renderer)
                        inner = ax.bbox.padded(-px)
                        if not (
                            inner.contains(box.x0, box.y0)
                            and inner.contains(box.x1, box.y1)
                        ):
                            continue
                        if any(box.padded(px).overlaps(other) for other in boxes):
                            continue
                        if fs.text_data_collisions(box, obstacles, px):
                            continue
                        chosen = dict(
                            case=note["case"],
                            label=note["label"],
                            side=side,
                            horizontal_offset_pt=dx,
                            label_bracket_distance_pt=gap,
                            axes_fraction=list(
                                ax.transAxes.inverted().transform((x, anchor))
                            ),
                        )
                        boxes.append(box)
                        break
                    if chosen:
                        break
                if chosen:
                    break
            if chosen:
                break
        assert chosen, ("No attached score-gap label position", note["case"])
        placements.append(chosen)
    return placements


def render(data, selections):
    """Return live artists for an independent check; do not write output here."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator, NullLocator, NullFormatter

    fs.apply_style()
    fig, axes = plt.subplots(4, 4, figsize=tuple(v / 72 for v in SIZE_PT), dpi=220)
    height = SIZE_PT[1]
    fig.subplots_adjust(
        left=0.13, right=0.985, top=0.91, bottom=30 / height, wspace=0.38, hspace=0.70
    )
    notes = []
    for i, ((model, task), ax) in enumerate(zip(CASES, axes.flat)):
        name = f"{model}/{task}"
        d = data[name]
        series = {
            m: [
                p
                for p in ps
                if m == "ours" or p["budget"] not in (0.49, 0.64, 1.0, 4.0)
            ]
            for m, ps in d["grids"].items()
        }
        chosen = selections[name]
        fs.panel_frame(ax)
        ratios = [p["nll_ratio"] for ps in series.values() for p in ps]
        low, high = min(ratios), max(ratios)
        log_span = max(math.log(high / low), math.log(1.25))
        ymin, ymax = low * math.exp(-0.12 * log_span), high * math.exp(0.65 * log_span)
        ax.set_yscale("log")
        ax.set_ylim(ymin, ymax)
        ax.yaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_minor_formatter(NullFormatter())
        fs.plain_log_ticks(
            ax.yaxis,
            maximum_ticks=3,
            candidates=(1, 1.1, 1.3, 1.5, 2, 3, 5, 10, 15, 20, 30, 50, 100),
        )
        for method, points in series.items():
            style = STYLES[method]
            line = fs.method_line(method)
            xx = [100 * p["score"] for p in points]
            yy = [p["nll_ratio"] for p in points]
            (artist,) = ax.plot(
                xx,
                yy,
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=line["linewidth"],
                alpha=line["alpha"],
                zorder=line["zorder"],
            )
            artist.set_gid(f"{name}:{method}:curve")
            (artist,) = ax.plot(
                xx,
                yy,
                color=style["color"],
                marker=style["marker"],
                linestyle="none",
                markersize=2.6 if method == METHODS[0] else 2.3,
                markeredgewidth=0.45,
                markerfacecolor=style["color"],
                alpha=line["alpha"],
                clip_on=False,
                zorder=5 if method == METHODS[0] else 4,
            )
            artist.set_gid(f"{name}:{method}:grid")
            artist.set_path_effects(
                [pe.Stroke(linewidth=0.9, foreground="white"), pe.Normal()]
            )
        p = d["baseline"]
        (artist,) = ax.plot(
            [100 * p["score"]],
            [p["nll_ratio"]],
            marker="x",
            color=fs.INK,
            linestyle="none",
            markersize=3.2,
            markeredgewidth=0.7,
            zorder=6,
        )
        artist.set_gid(f"{name}:baseline")
        values = [100 * p["score"] for ps in series.values() for p in ps]
        span = max(max(values) - min(values), 5)
        ax.set_xlim(min(values) - 0.14 * span, max(values) + 0.16 * span)
        lo, hi = ax.get_xlim()
        ax.set_xticks(
            [
                v
                for v in MaxNLocator(nbins=3, steps=[1, 2, 2.5, 5, 10]).tick_values(
                    lo, hi
                )
                if lo <= v <= hi
            ]
        )
        # Crowded grid points keep only the line; ours claims its markers first.
        occupied = []
        for method in sorted(series, key=lambda m: m != METHODS[0]):
            mark = [a for a in ax.lines if a.get_gid() == f"{name}:{method}:grid"][0]
            mark.set_markevery(
                fs.spaced_markers(
                    ax, mark.get_xdata(), mark.get_ydata(), occupied=occupied
                )
            )
        if i % 4 == 0:
            box = ax.get_position()
            y = (box.y0 + box.y1) / 2
            fig.text(
                0.018,
                y,
                TASK_NAMES[task],
                rotation=90,
                ha="center",
                va="center",
                fontsize=8,
            ).set_gid(f"{task}:row_label")
            label = (
                "Non-toxic rate (%)" if task in ("paradetox", "rtp") else "Accuracy (%)"
            )
            fig.text(
                0.55, box.y0 - 22 / height, label, ha="center", va="center", fontsize=8
            ).set_gid(f"{task}:score_label")
        ax.grid(axis="y")
        ax.tick_params(pad=2)
        ax.set_ylabel("NLL ratio", fontsize=7.5, labelpad=1.5)
        if i < 4:
            fs.panel_title(ax, NAMES[i], fontsize=8)
        control = max(
            (m for m in series if m != METHODS[0]), key=lambda m: chosen[m]["score"]
        )
        pins, other = chosen[METHODS[0]], chosen[control]
        x0, x1 = sorted((100 * pins["score"], 100 * other["score"]))
        bracket_y = high * math.exp(0.47 * log_span)
        bracket = ax.annotate(
            "",
            xy=(x1, bracket_y),
            xytext=(x0, bracket_y),
            arrowprops=dict(
                arrowstyle="|-|",
                color=fs.INK,
                linewidth=0.6,
                mutation_scale=1.5,
                shrinkA=0,
                shrinkB=0,
            ),
            zorder=10,
        )
        bracket.set_gid(f"{name}:bracket")
        if x0 == x1:
            (cap,) = ax.plot(
                [x0],
                [bracket_y],
                marker="|",
                linestyle="none",
                markersize=3,
                markeredgewidth=0.6,
                color=fs.INK,
                zorder=10,
            )
            cap.set_gid(f"{name}:bracket_zero")
        label = gap_label(pins, other)
        notes.append(
            dict(
                ax=ax,
                case=name,
                y=bracket_y,
                midpoint=(x0 + x1) / 2,
                lower=x0,
                upper=x1,
                label=label,
            )
        )
    handles = [
        Line2D(
            [],
            [],
            **STYLES[m],
            linewidth=fs.method_line(m)["linewidth"],
            markersize=3,
            markerfacecolor=STYLES[m]["color"],
        )
        for m in METHODS
    ]
    handles.append(
        Line2D([], [], color=fs.MUTED, marker="x", ls="none", ms=4.4, label="unsteered")
    )
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.55, 0.995),
        ncol=5,
        handlelength=1.05,
        columnspacing=1.1,
        handletextpad=0.35,
        borderaxespad=0,
    )
    placements = place_attached_gap_labels(fig, notes)
    return fig
