"""Presentation constants for the current paper, separate from archived evidence."""

TEXT_WIDTH = 6.5
FONT_FAMILY = "STIXGeneral"
TITLE_SIZE = 9
LABEL_SIZE = 8
TICK_SIZE = 7.5
INK = "#25262a"
MUTED = "#68656a"
GRID = "#e9e8e5"
REFERENCE = "#8b8c90"
FRAME = "#c4c5c3"
PANEL = "#fdfbf8"
HEADER = "#3b404b"
PINS_BLUE = "#326f9d"
ADDITION_BLUE = "#95b6d0"
CAA_BROWN = "#a26132"
NEUTRAL_SLATES = ("#2f3440", "#7b8494", "#c3c8d0")
TASK_COLORS = dict(zip(("truthfulqa", "halueval", "paradetox"), NEUTRAL_SLATES))


def rc_params():
    """One print-scale system, independent of earlier builders' global state.

    STIX text and mathematics match the author's training/deployment figures.
    Matplotlib supplies the font files; embed them as TrueType.
    """
    import matplotlib as mpl
    from matplotlib.font_manager import FontProperties, findfont

    findfont(FontProperties(family=FONT_FAMILY), fallback_to_default=False)
    rc = dict(mpl.rcParamsDefault)
    rc.update(
        {
            "font.family": FONT_FAMILY,
            "font.size": LABEL_SIZE,
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.titlesize": TITLE_SIZE,
            "axes.titleweight": "normal",
            "axes.titlelocation": "left",
            "axes.titlepad": 6,
            "axes.labelsize": LABEL_SIZE,
            "axes.labelpad": 4,
            "xtick.labelsize": TICK_SIZE,
            "ytick.labelsize": TICK_SIZE,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#b0b2b3",
            "axes.linewidth": 0.5,
            "axes.facecolor": PANEL,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.5,
            "lines.linewidth": 1.1,
            "lines.markersize": 3.2,
            "lines.markeredgewidth": 0.7,
            "legend.fontsize": TICK_SIZE,
            "legend.frameon": False,
            "legend.handlelength": 1.8,
            "legend.handletextpad": 0.5,
            "legend.columnspacing": 1.2,
            "legend.labelspacing": 0.45,
            "savefig.facecolor": "white",
        }
    )
    return rc


def apply_style():
    import matplotlib.pyplot as plt

    plt.rcParams.update(rc_params())


def plain_log_ticks(
    axis,
    maximum_ticks=4,
    minimum_gap_pt=4,
    candidates=(1, 1.1, 1.5, 2, 3, 5, 10, 15, 20, 30, 50, 100),
):
    """Choose plain-number log ticks using rendered label spacing."""
    from itertools import combinations
    import math

    ax = axis.axes
    low, high = axis.get_view_interval()
    values = [v for v in candidates if low <= v <= high]
    axis.set_ticks(values, labels=[f"{v:g}" for v in values])
    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    boxes = [t.get_window_extent(renderer) for t in axis.get_ticklabels()]
    for count in range(min(maximum_ticks, len(values)), 1, -1):
        choices = []
        for indices in combinations(range(len(values)), count):
            if indices[0] != 0 or indices[-1] != len(values) - 1:
                continue
            gaps = [
                (
                    (boxes[b].y0 - boxes[a].y1)
                    if axis is ax.yaxis
                    else (boxes[b].x0 - boxes[a].x1)
                )
                * 72
                / ax.figure.dpi
                for a, b in zip(indices, indices[1:])
            ]
            if min(gaps) >= minimum_gap_pt:
                distances = [
                    math.log(values[b] / values[a])
                    for a, b in zip(indices, indices[1:])
                ]
                choices.append((sum(d * d for d in distances), indices))
        if choices:
            ticks = [values[i] for i in min(choices)[1]]
            axis.set_ticks(ticks, labels=[f"{v:g}" for v in ticks])
            return
    raise AssertionError(("No readable log ticks", low, high))


def panel_frame(ax):
    """A rounded 0.5 pt frame, without changing data limits or plot geometry."""
    from matplotlib.patches import FancyBboxPatch

    if getattr(ax, "_paper_panel_frame", None) is None:
        frame = FancyBboxPatch(
            (0, 0),
            1,
            1,
            transform=ax.transAxes,
            boxstyle="round,pad=0,rounding_size=0.025",
            facecolor="none",
            edgecolor=FRAME,
            linewidth=0.5,
            clip_on=False,
            zorder=2,
        )
        ax.add_artist(frame)
        ax._paper_panel_frame = frame
        for spine in ax.spines.values():
            spine.set_visible(False)
    return ax._paper_panel_frame


def panel_title(ax, title, letter=None, fontsize=TITLE_SIZE):
    """Plain left-aligned title: bold first line, optional second line in muted type."""
    panel_frame(ax)
    lines = title.split("\n")
    head = f"({letter}) {lines[0]}" if letter else lines[0]

    def put(dy, text, **style):
        return ax.annotate(
            text,
            xy=(0, 1),
            xycoords="axes fraction",
            xytext=(0, dy),
            textcoords="offset points",
            ha="left",
            va="bottom",
            fontsize=fontsize,
            annotation_clip=False,
            zorder=11,
            **style,
        )

    dy = 3
    if len(lines) > 1:
        put(dy, lines[1], color=MUTED)
        dy += fontsize * 1.2
    return put(dy, head, fontweight="bold", color=INK)


def method_line(method):
    """Redundant silhouettes/dashes, with SS visually ahead of controls."""
    ours = method == "ours"
    return dict(
        linewidth=1.6 if ours else 0.9,
        markersize=3.3 if ours else 2.8,
        markerfacecolor="white",
        markeredgewidth=0.7 if ours else 0.6,
        alpha=1 if ours else 0.75,
        zorder=4 if ours else 3,
    )


PRIMARY_STYLE = {
    "dense": dict(label="CAA", color=CAA_BROWN, marker="o", linestyle="--"),
    "normalized_dense": dict(
        label="Normalized CAA", color="#648477", marker="s", linestyle="-."
    ),
    "sparsification_only": dict(
        label="SS: addition", color=ADDITION_BLUE, marker="D", linestyle="--"
    ),
    "ours": dict(
        label="SS: scaled norm preservation", color=PINS_BLUE, marker="^", linestyle="-"
    ),
}
# Paired coordinate-set estimates share one stroke, including endpoint caps.
PAIRED_INTERVAL_STYLE = dict(
    color=PRIMARY_STYLE["sparsification_only"]["color"],
    linewidth=1.2,
    linestyle="solid",
    capstyle="butt",
    alpha=1,
    zorder=3,
)
MODEL_NAMES = {
    "gemma": "Gemma 2B",
    "qwen": "Qwen 3B",
    "llama": "Llama 7B",
    "qwen7": "Qwen 7B",
}
DATASET_NAMES = {
    "truthfulqa": "TruthfulQA",
    "halueval": "HaluEval",
    "paradetox": "ParaDetox",
}
METHOD_STYLE = {"magnitude": dict(color="#9a803c")}
PDF_METADATA = {"CreationDate": None, "ModDate": None, "Author": None, "Creator": None}


def feasibility_facecolor(feasible, color):
    """Fill settings that pass both supplied limits; leave failures hollow."""
    return color if feasible else "white"


MARKER_GAP_PT = 5.5


def spaced_markers(ax, xs, ys, keep=(), occupied=None, gap_pt=MARKER_GAP_PT):
    """Indices of the sweep points that receive a marker.

    The curve itself passes through every point; markers are dropped where they
    would crowd, so that no drawn marker lies within gap_pt of another marker of
    the same curve or of a position in occupied (markers of curves drawn
    earlier in the panel), to which the kept positions are appended.  The
    endpoints and the indices in keep are always marked.  Call this only after
    the axis limits and the figure layout are final."""
    import numpy as np

    xy = ax.transData.transform(np.column_stack((xs, ys))) * 72 / ax.figure.dpi
    occupied = [] if occupied is None else occupied
    finite = [i for i in range(len(xs)) if np.isfinite(xy[i]).all()]
    chosen = sorted({finite[0], finite[-1], *keep})
    taken = [xy[j] for j in chosen] + list(occupied)
    for i in finite:
        if i not in chosen and all(np.hypot(*(xy[i] - q)) >= gap_pt for q in taken):
            chosen.append(i)
            taken.append(xy[i])
    occupied.extend(xy[i] for i in chosen)
    return sorted(chosen)


def plot_obstacles(fig):
    """Actual strokes, marker envelopes, bars and bands in display coordinates.

    Axis background/feasibility shading and panel frames are not measurements.
    Dashed strokes are checked continuously, which is conservative about gaps.
    """
    from matplotlib.collections import LineCollection, PathCollection, PolyCollection
    from matplotlib.container import BarContainer
    from matplotlib.path import Path
    from matplotlib.text import Annotation
    from matplotlib.transforms import Bbox
    import numpy as np

    obstacles = []
    px = fig.dpi / 72

    def add_path(path, kind, padding=0, filled=False):
        if len(path.vertices):
            obstacles.append((kind, path.get_extents(), path, padding, filled))

    for ax in fig.axes:
        for line in ax.lines:
            if not line.get_visible():
                continue
            path = line.get_path().transformed(line.get_transform())
            if line.get_linestyle() not in ("None", "", " "):
                reference = len(path.vertices) == 2 and (
                    np.ptp(path.vertices[:, 0]) == 0 or np.ptp(path.vertices[:, 1]) == 0
                )
                add_path(
                    path,
                    "reference_line" if reference else "curve",
                    line.get_linewidth() * px / 2,
                )
            if line.get_marker() not in ("None", None, "", " "):
                radius = (line.get_markersize() + line.get_markeredgewidth()) * px / 2
                for x, y in path.vertices:
                    if np.isfinite([x, y]).all():
                        obstacles.append(
                            (
                                "marker",
                                Bbox.from_extents(
                                    x - radius, y - radius, x + radius, y + radius
                                ),
                                None,
                                0,
                                True,
                            )
                        )
        for collection in ax.collections:
            if not collection.get_visible():
                continue
            if isinstance(collection, PathCollection):
                points = collection.get_offset_transform().transform(
                    collection.get_offsets()
                )
                sizes = collection.get_sizes()
                widths = collection.get_linewidths()
                for i, (x, y) in enumerate(points):
                    radius = (
                        (np.sqrt(sizes[i % len(sizes)]) + widths[i % len(widths)])
                        * px
                        / 2
                    )
                    obstacles.append(
                        (
                            "marker",
                            Bbox.from_extents(
                                x - radius, y - radius, x + radius, y + radius
                            ),
                            None,
                            0,
                            True,
                        )
                    )
            elif isinstance(collection, LineCollection):
                for segment in collection.get_segments():
                    add_path(
                        Path(segment).transformed(collection.get_transform()),
                        "error_bar",
                        max(collection.get_linewidths()) * px / 2,
                    )
            elif isinstance(collection, PolyCollection):
                for path in collection.get_paths():
                    add_path(
                        path.transformed(collection.get_transform()),
                        "band",
                        filled=True,
                    )
        for container in ax.containers:
            if isinstance(container, BarContainer):
                for bar in container.patches:
                    obstacles.append(
                        (
                            "bar",
                            bar.get_window_extent(),
                            None,
                            bar.get_linewidth() * px / 2,
                            True,
                        )
                    )
        for text in ax.texts:
            if isinstance(text, Annotation) and text.arrow_patch is not None:
                patch = text.arrow_patch
                add_path(
                    patch.get_path().transformed(patch.get_transform()),
                    "annotation_line",
                    patch.get_linewidth() * px / 2,
                )
    return obstacles


def text_data_collisions(box, obstacles, clearance):
    """Return obstacle classes touched by a text box plus physical clearance."""
    hits = []
    for kind, bounds, path, padding, filled in obstacles:
        test = box.padded(clearance + padding)
        if bounds.overlaps(test) and (
            path is None or path.intersects_bbox(test, filled=filled)
        ):
            hits.append(kind)
    return hits


def rendered_texts(fig):
    """Exclude cached tick labels outside the view: Matplotlib does not draw them."""
    from matplotlib.text import Text

    hidden = set()
    for ax in [a for parent in fig.axes for a in (parent, *parent.child_axes)]:
        for axis in (ax.xaxis, ax.yaxis):
            low, high = sorted(axis.get_view_interval())
            for tick in [*axis.get_major_ticks(), *axis.get_minor_ticks()]:
                if not low <= tick.get_loc() <= high:
                    hidden.update((id(tick.label1), id(tick.label2)))
    return [
        t
        for t in fig.findobj(Text)
        if t.get_visible() and t.get_text().strip() and id(t) not in hidden
    ]
