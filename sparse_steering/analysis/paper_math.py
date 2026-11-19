"""Match math-label ink anchors using Matplotlib's bundled Computer Modern fonts.

PDF labels are replaced after export to retain the original STIX text geometry.
No TeX installation or external font files are needed.
"""

import io
import re
import matplotlib
import matplotlib.pyplot as plt
import pymupdf
from .paper_style import PDF_METADATA, apply_style


PROBE = 100.0  # where a label is drawn when its ink is measured

# Each entry: the exact text of the old label, the new text, and which point of
# the old label stays fixed.
FIGURES = {
    "norm_strength": dict(
        labels=[
            dict(text="Strength α", new=r"Strength $\alpha$", anchor="center"),
        ]
    ),
    "selected_coordinate_statistics": dict(
        labels=[
            dict(
                text="Share of ‖v‖2 (%)", new=r"Share of $\|v\|^2$ (%)", anchor="center"
            ),
            dict(text="Largest K coefficients", new=r"Largest $K$ coefficients"),
            dict(
                text="Random K coordinates (expected)",
                new=r"Random $K$ coordinates (expected)",
            ),
        ]
    ),
    "mask_seed_overlap": dict(
        labels=[
            dict(text="Overlap |A∩B|/K", new=r"Overlap $|A\cap B|/K$", anchor="center"),
            dict(text="Chance K/d ≈ 20%", new=r"Chance $K/d\approx 20\%$"),
        ]
    ),
}


def page_lines(page):
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            spans = [span for span in line["spans"] if span["text"].strip()]
            if spans:
                yield "".join(
                    span["text"] for span in line["spans"]
                ).strip(), spans, line["dir"]


def geometry(spans, direction=(1.0, 0.0)):
    """Ink box, main baseline and look of a label made of these text pieces."""
    size = max(span["size"] for span in spans)
    main = next(span for span in spans if span["size"] == size)
    box = pymupdf.Rect()
    for span in spans:
        box |= pymupdf.Rect(span["bbox"])
    return dict(
        box=box,
        origin=main["origin"],
        size=size,
        color="#%06x" % main["color"],
        bold=any(span["font"].endswith("Bold") for span in spans),
        # Computer Modern pieces mean that this label was converted before.
        converted=any("Cm" in span["font"] for span in spans),
        vertical=abs(direction[1]) > 0.5,
        spans=spans,
    )


def find(page, label):
    return [
        geometry(spans, direction)
        for text, spans, direction in page_lines(page)
        if text == label["text"]
    ]


def draw(ax, x, y, old, new, rotation=0):
    return ax.text(
        x,
        y,
        new,
        fontsize=old["size"],
        color=old["color"],
        fontweight="bold" if old["bold"] else "normal",
        ha="left",
        va="baseline",
        rotation=rotation,
        rotation_mode="anchor",
    )


def canvas(width, height):
    fig = plt.figure(figsize=(width / 72, height / 72))
    fig.patch.set_alpha(0)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set(xlim=(0, width), ylim=(height, 0))  # PDF page coordinates, y downwards
    ax.set_axis_off()
    return fig, ax


def as_pdf(fig):
    buffer = io.BytesIO()
    fig.savefig(
        buffer, format="pdf", transparent=True, bbox_inches=None, metadata=PDF_METADATA
    )
    plt.close(fig)
    return pymupdf.open(stream=buffer.getvalue(), filetype="pdf")


def measure(old, new):
    """Offsets of the drawn ink from the requested anchor, for horizontal text."""
    fig, ax = canvas(900, 220)
    draw(ax, PROBE, PROBE, old, new)
    ink = geometry(
        [span for _, spans, _ in page_lines(as_pdf(fig)[0]) for span in spans]
    )
    return ink["box"].x0 - PROBE, ink["box"].x1 - PROBE, ink["origin"][1] - PROBE


def placement(old, label):
    left, right, drop = measure(old, label["new"])
    anchor, box = label.get("anchor", "left"), old["box"]
    if old["vertical"]:
        # Read upwards: the text runs towards smaller y and its baseline is a vertical line.
        along = dict(left=box.y1, center=(box.y0 + box.y1) / 2, right=box.y0)[anchor]
        along = label.get("at", along)
        shift = dict(left=left, center=(left + right) / 2, right=right)[anchor]
        return old["origin"][0] - drop, along + shift, 90
    target = dict(left=box.x0, center=(box.x0 + box.x1) / 2, right=box.x1)[anchor]
    target = label.get("at", target)
    shift = dict(left=left, center=(left + right) / 2, right=right)[anchor]
    return target - shift, old["origin"][1] - drop, 0


def remove_labels(document, page, labels):
    """Delete the labels from the page's own code, leaving everything else as it is.

    Matplotlib writes each label as "q [colour] matrix cm BT ... ET Q", where the
    matrix holds the label's anchor. Editing that code directly avoids redaction,
    which rewrites the whole page and was found to change marker outlines.
    """
    block = re.compile(
        r"q\n(?:[^\n]*rg\n)?[-\d.]+ [-\d.]+ [-\d.]+ [-\d.]+ ([-\d.]+) ([-\d.]+) cm\n"
        r"BT\n.*?\nET\n(?:q [-\d. ]+ re f Q\n)*Q\n",
        re.S,
    )
    streams = {
        xref: document.xref_stream(xref).decode("latin1")
        for xref in page.get_contents()
    }
    blocks = [
        (
            xref,
            m.start(),
            m.end(),
            float(m.group(1)),
            page.rect.height - float(m.group(2)),
        )
        for xref, stream in streams.items()
        for m in block.finditer(stream)
    ]
    cut = set()
    for old in labels:
        x, y = old["spans"][0]["origin"]
        xref, start, end, bx, by = min(
            blocks, key=lambda b: abs(b[3] - x) + abs(b[4] - y)
        )
        assert abs(bx - x) < 2.5 and abs(by - y) < 2.5, (
            old["spans"][0]["text"],
            x,
            y,
            bx,
            by,
        )
        cut.add((xref, start, end))
    assert len(cut) == len(labels)
    for xref, start, end in sorted(cut, reverse=True):
        streams[xref] = streams[xref][:start] + streams[xref][end:]
    for xref in {c[0] for c in cut}:
        document.update_stream(xref, streams[xref].encode("latin1"))
    return document.reload_page(page)


def retypeset(name, target):
    spec = FIGURES[name]
    document = pymupdf.open(target)
    page = document[0]
    apply_style()
    matplotlib.rcParams["mathtext.fontset"] = "cm"
    fig, ax = canvas(page.rect.width, page.rect.height)
    olds = []
    for label in spec["labels"]:
        for old in find(page, label):
            if old["converted"]:
                continue
            x, y, rotation = placement(old, label)
            draw(ax, x, y, old, label["new"], rotation=rotation)
            olds.append(old)
    if not olds:
        plt.close(fig)
        return 0
    page = remove_labels(document, page, olds)
    page.show_pdf_page(page.rect, as_pdf(fig), 0)
    data = document.tobytes(garbage=3, deflate=True)
    document.close()
    target.write_bytes(data)
    return len(olds)
