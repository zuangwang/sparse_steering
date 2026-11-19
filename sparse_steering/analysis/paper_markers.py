"""PDF marker finishing for the paper's filled symbols and legend cross.

Work on the exported vector paths so their sizes and stroke order agree with
other panels independently of the plotting backend's marker definitions.
"""

import re
import pymupdf


INK = (0x25 / 255, 0x26 / 255, 0x2A / 255)
OURS = "#326f9d"
# Marker extents in Figure 1, in points: triangle 2.6 and circle 2.3. Squares take
# the circle's size and the diamond is scaled by the same factor.
TARGET = dict(triangle=2.6, circle=2.3, square=2.3, diamond=3.3)
MITER, ROUND = 0, 1


def hexa(colour):
    return (
        None
        if colour is None
        else "#%02x%02x%02x" % tuple(round(v * 255) for v in colour)
    )


def edit_stream(document, page, edit):
    """Apply `edit` to the page's drawing code and return the refreshed page."""
    first, *others = page.get_contents()
    # A page may hold its code in several parts; they are read as one text.
    code = "\n".join(
        document.xref_stream(xref).decode("latin1") for xref in (first, *others)
    )
    document.update_stream(first, edit(code).encode("latin1"))
    for xref in others:
        document.update_stream(xref, b" ")
    return document.reload_page(page)  # the page object must see the edited code


def legend_cross(document, page):
    """Figure 1: redraw the legend cross with the colour and size of the panel crosses."""
    crosses = [
        d
        for d in page.get_drawings()
        if len(d["items"]) == 2
        and all(i[0] == "l" for i in d["items"])
        and d["rect"].width > 2
        and abs(d["rect"].width - d["rect"].height) < 0.01
    ]
    legend = [d for d in crosses if d["rect"].y1 < 15]
    panel = [d for d in crosses if d["rect"].y1 >= 15]
    assert (
        len(legend) == 1
        and len(panel) == 16
        and len({hexa(d["color"]) for d in panel}) == 1
    )
    old, new = legend[0], panel[0]
    if hexa(old["color"]) == hexa(new["color"]):
        return False
    height, box, half = page.rect.height, old["rect"], new["rect"].width / 2
    centre = ((box.x0 + box.x1) / 2, height - (box.y0 + box.y1) / 2)
    colour = " ".join(f"{v:.10f}" for v in new["color"])
    # The cross is two strokes in the page's own code: colour, width, then "m l m l B".
    pattern = re.compile(
        r"[\d.]+ [\d.]+ [\d.]+ rg [\d.]+ w [\d.]+ [\d.]+\s+[\d.]+ RG [\d.]+ [\d.]+ [\d.]+ rg\s+"
        r"([\d.]+) ([\d.]+) m\s+([\d.]+) ([\d.]+) l\s+[\d.]+ [\d.]+ m\s+[\d.]+ [\d.]+ l\s+B\n"
    )

    def is_legend(match):
        x0, y0, x1, y1 = map(float, match.groups())
        return (
            abs(x0 - box.x0) < 0.01
            and abs(x1 - box.x1) < 0.01
            and abs(min(y0, y1) - (height - box.y1)) < 0.01
            and abs(max(y0, y1) - (height - box.y0)) < 0.01
        )

    def edit(stream):
        (match,) = [m for m in pattern.finditer(stream) if is_legend(m)]
        x0, x1, y0, y1 = (
            centre[0] - half,
            centre[0] + half,
            centre[1] - half,
            centre[1] + half,
        )
        cross = (
            f'{colour} rg {new["width"]:.2f} w {colour} RG {colour} rg\n\n'
            f"{x0:.5f} {y0:.5f} m\n{x1:.5f} {y1:.5f} l\n{x0:.5f} {y1:.5f} m\n{x1:.5f} {y0:.5f} l\n\nB\n"
        )
        return stream[: match.start()] + cross + stream[match.end() :]

    edit_stream(document, page, edit)
    return True


def kind(drawing):
    items = "".join(item[0] for item in drawing["items"])
    return {"lll": "triangle", "llll": "diamond", "re": "square"}.get(items, "circle")


def outline(drawing, scale):
    """Corner points of a marker, scaled about its centre (a circle gives centre and radius)."""
    box = drawing["rect"]
    centre = pymupdf.Point((box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2)
    if kind(drawing) == "circle":
        return centre, box.width / 2 * scale
    if kind(drawing) == "square":
        corners = [box.tl, box.tr, box.br, box.bl]
    else:
        corners = [item[1] for item in drawing["items"]]
    return [centre + (corner - centre) * scale for corner in corners], None


def marker(shape, drawing, scale, **style):
    points, radius = outline(drawing, scale)
    if radius is None:
        shape.draw_polyline(points)
    else:
        shape.draw_circle(points, radius)
    shape.finish(
        closePath=True, lineJoin=ROUND if radius is not None else MITER, **style
    )


def filled_markers(document, page):
    """Figure 6: replace the hollow markers by filled markers over a white rim."""
    drawings = page.get_drawings()
    hollow = [
        d
        for d in drawings
        if d["type"] == "fs"
        and hexa(d["fill"]) == "#ffffff"
        and max(d["rect"].width, d["rect"].height) < 6
    ]
    if not hollow:
        return False
    panels = [
        d["rect"]
        for d in drawings
        if d["type"] == "f" and d["rect"].width > 40 and d["rect"].y0 > 25
    ]
    our_lines = [
        d
        for d in drawings
        if d["type"] == "s"
        and hexa(d["color"]) == OURS
        and d["width"] > 1.5
        and all(item[0] == "l" for item in d["items"])
    ]
    assert all(
        d["rect"].y1 < 25 or any(panel.contains(d["rect"]) for panel in panels)
        for d in our_lines
    )
    # Most markers are small forms placed with "Do"; dropping those calls removes them.
    page = edit_stream(document, page, lambda stream: re.sub(r"/M\d+ Do", "", stream))
    # The others are closed paths written out in the page's code and painted with "B".
    height = page.rect.height
    left = [
        d["rect"]
        for d in page.get_drawings()
        if d["type"] == "fs"
        and hexa(d["fill"]) == "#ffffff"
        and max(d["rect"].width, d["rect"].height) < 6
    ]
    path = re.compile(
        r"((?:[-\d.]+ [-\d.]+ [ml]\n|[-\d.]+ [-\d.]+ [-\d.]+ [-\d.]+ [-\d.]+ [-\d.]+ c\n)+h\n\n)B\n"
    )

    def unpaint(match):
        numbers = [
            [float(v) for v in line.split()[:-1]][-2:]
            for line in match.group(1).splitlines()[:-2]
        ]
        xs, ys = [n[0] for n in numbers], [height - n[1] for n in numbers]
        box = pymupdf.Rect(min(xs), min(ys), max(xs), max(ys))
        is_marker = any(
            abs(box.x0 - r.x0) < 0.05
            and abs(box.y0 - r.y0) < 0.05
            and abs(box.x1 - r.x1) < 0.05
            and abs(box.y1 - r.y1) < 0.05
            for r in left
        )
        return match.group(1) + ("n\n" if is_marker else "B\n")

    page = edit_stream(document, page, lambda stream: path.sub(unpaint, stream))
    assert not [
        d
        for d in page.get_drawings()
        if d["type"] == "fs"
        and hexa(d["fill"]) == "#ffffff"
        and max(d["rect"].width, d["rect"].height) < 6
    ]
    assert len(page.get_drawings()) == len(drawings) - len(hollow)

    shape = page.new_shape()
    in_legend = lambda d: d["rect"].y1 < 25

    def draw(d):
        colour, opacity = d["color"], d["stroke_opacity"]
        if in_legend(d):
            marker(shape, d, 1, fill=colour, color=colour, width=0.7)
            return
        scale = TARGET[kind(d)] / d["rect"].width
        marker(
            shape,
            d,
            scale,
            fill=colour,
            color=(1, 1, 1),
            width=0.9,
            fill_opacity=opacity,
            stroke_opacity=1,
        )
        marker(
            shape,
            d,
            scale,
            fill=colour,
            color=colour,
            width=0.45,
            fill_opacity=opacity,
            stroke_opacity=opacity,
        )

    for d in hollow:
        if hexa(d["color"]) != OURS:
            draw(d)
    for d in our_lines:
        points = [d["items"][0][1]] + [item[2] for item in d["items"]]
        shape.draw_polyline(points)
        shape.finish(
            color=d["color"],
            width=d["width"],
            closePath=False,
            lineJoin=ROUND,
            lineCap=2,
        )
    for d in hollow:
        if hexa(d["color"]) == OURS:
            draw(d)
    shape.commit()
    return True
