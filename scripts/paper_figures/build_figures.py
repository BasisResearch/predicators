"""Compose paper figures from archived simulator renders, recorded runs, and
verified results."""
import base64
import functools
import hashlib
import io
import json
import math
import os
import random
import re
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Sequence, \
    Tuple, Union

# Optional figure-authoring dependencies, separate from benchmark runtime.
import cairocffi  # type: ignore[import-not-found] # pylint: disable=import-error
import cairosvg  # type: ignore[import-not-found] # pylint: disable=import-error
from PIL import Image, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parent
PAPER = Path(
    os.environ.get("EMPIRIC_PAPER_ROOT",
                   str(ROOT.parents[2] / "sim-predicator-paper"))).resolve()
FIG = ROOT / "figures"
OUTPUT = PAPER / "figures"
DOMAINS = ["Domino", "Bridge", "Balloons", "Boil", "Fan"]
INK, MUTED = "#203744", "#5b6e79"
TEAL, RUST, GREEN = "#087f8c", "#bd5929", "#397957"
PANEL, EDGE = "#f7f9fa", "#d9e2e7"
NS = "http://www.w3.org/2000/svg"
USED_IMAGES = set()
RENDERER = "gui"
ET.register_namespace("", NS)
# Crops retain task objects and their surroundings in the Cycles renders,
# 900 pixels square except Balloons at 1280 by 800. Fan frames the arena, and
# Balloons keeps the air above the table, where lifted balloons float, up to
# the burst cap over the chute.
CROPS = {
    "boil": (190, 185, 890, 830),
    "domino": (0, 145, 710, 850),
    "fan": (190, 185, 820, 723),
    "bridge": (220, 200, 880, 850),
    "balloons": (312, 70, 968, 631),
}
# The original Bridge illustrations and trajectory panels used this tighter
# crop. Keep it shared with prepare_figma_assets.py so the paper and Figma use
# identical framing.
BRIDGE_FOCUS_CROP = (270, 280, 810, 860)
# Figure 1's two-block lift panels retain the gripper as context. The solved
# panel is wider and therefore uses its own crop.
BRIDGE_PAIR_CROP = (150, 60, 830, 700)
BRIDGE_SOLVED_CROP = (210, 220, 820, 560)
# Figure 2's act panel shows the robot lowering a block, glued on the end
# that faces the row, toward the row's glued end (a 900 by 540 render). The
# crop centres the two glued ends and trims the gripper body.
BRIDGE_ACT_CROP = (60, 0, 810, 450)
# An illustrative glue program in the paper's terms. Wet faces are observed,
# so they select the glued joints; the hidden recurrent state x_res counts
# each joint's cure progress while its faces meet; and a joint past the cure
# time t_cure in theta_res becomes an engine weld through the Attach residual
# command. The recorded run's program tracked dab wetness instead and welded
# on contact.
GLUE_CODE = [
    "class Glue(BaseSimulator):",
    "  def step(self, action):",
    "    for j in wet_joints():",
    "      if faces_meet(j):",
    "        x_res.cure[j] += 1",
    "      if x_res.cure[j] > t_cure:",
    "        self.attach(j)",
]
Segment = Union[str, Tuple[str, str]]
# Python token colors from the Visual Studio Code light theme.
PY_STORAGE, PY_CONTROL, PY_TYPE = "#0000ff", "#af00db", "#267f99"
PY_FUNCTION, PY_VARIABLE, PY_NUMBER = "#795e26", "#001080", "#098658"
_PY_TOKEN = re.compile(r"\s+|[A-Za-z_]\w*|\d+(?:\.\d+)?|\.\.\.|\S")
_PY_STORAGE = {"class", "def", "lambda", "self", "None", "True", "False"}
_PY_CONTROL = {
    "if", "elif", "else", "for", "while", "in", "not", "and", "or", "is",
    "return", "pass", "break", "continue", "import", "from", "as", "with"
}


def _python_tokens(line: str) -> List[Tuple[str, str]]:
    """Split a code line into tokens colored as a Python editor would.

    A leading "+" or "-" is a diff marker and keeps the diff colors.
    """
    tokens = _PY_TOKEN.findall(line)
    words = [t for t in tokens if not t.isspace()]
    header = bool(words) and words[0] == "class"
    colored: List[Tuple[str, str]] = []
    previous = ""
    for i, token in enumerate(tokens):
        if token.isspace():
            colored.append((token, INK))
            continue
        following = next((t for t in tokens[i + 1:] if not t.isspace()), "")
        if i == 0 and token in "+-":
            color = GREEN if token == "+" else RUST
        elif token in _PY_STORAGE:
            color = PY_STORAGE
        elif token in _PY_CONTROL:
            color = PY_CONTROL
        elif token[0].isalpha() or token[0] == "_":
            color = (PY_TYPE if header else PY_FUNCTION
                     if previous == "def" or following == "(" else PY_VARIABLE)
        elif token[0].isdigit():
            color = PY_NUMBER
        else:
            color = INK
        colored.append((token, color))
        previous = token
    return colored


class _DatedPDFSurface(cairosvg.surface.PDFSurface):
    """A PDF surface with a fixed creation date.

    Cairo otherwise stamps the build time into each PDF, so rebuilding
    an unchanged figure would write new bytes.
    """

    def _create_surface(self, width: float,
                        height: float) -> Tuple[Any, float, float]:
        surface, width, height = super()._create_surface(width, height)
        surface.set_metadata(cairocffi.PDF_METADATA_CREATE_DATE,
                             "2026-09-24T00:00:00Z")
        return surface, width, height


class Drawing:
    """SVG composition with raster panels and PDF/PNG export."""

    def __init__(self, height: float) -> None:
        self.height = height
        # Half a panel stroke of margin keeps strokes on the canvas edge as
        # wide as the others.
        pad = 0.4
        w, h = 528 + 2 * pad, height + 2 * pad
        self.root = ET.Element(
            f"{{{NS}}}svg", {
                "width": f"{w:g}",
                "height": f"{h:g}",
                "viewBox": f"{-pad:g} {-pad:g} {w:g} {h:g}"
            })
        self.root.append(
            ET.Comment(
                " Generated by scripts/build_figures.py; do not edit. "))
        self.add("rect", x=-pad, y=-pad, width=w, height=h, fill="white")

    def add(self, tag: str, **kw: Any) -> ET.Element:
        """Append an SVG element."""
        return ET.SubElement(
            self.root, f"{{{NS}}}{tag}",
            {k.replace("_", "-"): str(v)
             for k, v in kw.items()})

    def text(self,
             x: float,
             y: float,
             text: str,
             size: float = 10.5,
             color: str = INK,
             weight: str = "normal",
             anchor: str = "start",
             mono: bool = False,
             italic: bool = False) -> ET.Element:
        """Place one text label."""
        e = self.add("text",
                     x=x,
                     y=y,
                     font_family="DejaVu Sans Mono" if mono else "DejaVu Sans",
                     font_size=size,
                     fill=color,
                     font_weight=weight,
                     text_anchor=anchor)
        if italic:
            e.set("font-style", "italic")
        if mono:
            e.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
        e.text = text
        return e

    def rich(self,
             x: float,
             y: float,
             parts: Sequence[Segment],
             size: float = 8.1,
             color: str = INK,
             weight: str = "normal") -> ET.Element:
        """Place a left-aligned line with italic ("i") and code ("m") parts.

        CairoSVG misplaces anchored text that is split into spans, so
        these lines always start at x; _advance() measures them for
        centering.
        """
        e = self.text(x, y, "", size, color, weight)
        e.text = None
        for part in parts:
            content, style = (part, "") if isinstance(part, str) else part
            span = ET.SubElement(e, f"{{{NS}}}tspan")
            if style == "i":
                span.set("font-style", "italic")
            elif style == "m":
                span.set("font-family", "DejaVu Sans Mono")
            span.text = content
        return e

    def code(self, x: float, y: float, line: str, size: float) -> None:
        """Place one line of Python with standard syntax colors.

        Spaces become no-break spaces so SVG importers keep the
        indentation.
        """
        e = self.text(x, y, "", size, INK, mono=True)
        e.text = None
        for token, color in _python_tokens(line):
            span = ET.SubElement(e, f"{{{NS}}}tspan", {"fill": color})
            span.text = token.replace(" ", "\u00a0")

    def rect(self,
             x: float,
             y: float,
             w: float,
             h: float,
             fill: str = "#f4f7f8",
             stroke: str = "#d3dde1",
             radius: float = 5,
             dash: Optional[str] = None,
             width: float = .8) -> None:
        """Draw a rounded rectangle."""
        kw: Dict[str, Any] = dict(x=x,
                                  y=y,
                                  width=w,
                                  height=h,
                                  rx=radius,
                                  fill=fill,
                                  stroke=stroke,
                                  stroke_width=width)
        if dash:
            kw["stroke_dasharray"] = dash
        self.add("rect", **kw)

    def arrow(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        color: str = MUTED,
        dash: Optional[str] = None,
        via: Sequence[Tuple[float, float]] = ()
    ) -> None:
        """Draw a directed connector, optionally through corner points."""
        points = [(x1, y1), *via, (x2, y2)]
        (x1, y1), (x2, y2) = points[-2:]
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        ux, uy = dx / length, dy / length
        # The shaft stops inside the head; a shaft running to the tip pokes
        # out of the narrow point and blunts it.
        shaft = [*points[:-1], (x2 - 4 * ux, y2 - 4 * uy)]
        kw: Dict[str,
                 Any] = dict(d="M" + " L".join(f"{x} {y}" for x, y in shaft),
                             fill="none",
                             stroke=color,
                             stroke_width=1.3)
        if dash:
            kw["stroke_dasharray"] = dash
        self.add("path", **kw)
        head = [(x2, y2), (x2 - 5 * ux - 2.5 * uy, y2 - 5 * uy + 2.5 * ux),
                (x2 - 5 * ux + 2.5 * uy, y2 - 5 * uy - 2.5 * ux)]
        self.add("polygon",
                 points=" ".join(f"{x},{y}" for x, y in head),
                 fill=color)

    def image(self,
              source: Path,
              x: float,
              y: float,
              w: float,
              h: float,
              crop: Optional[Sequence[int]] = None,
              contain: bool = False) -> None:
        """Embed one raster file, cropped and fitted to its box."""
        USED_IMAGES.add(source)
        im = Image.open(source).convert("RGB")
        if crop is not None:
            left, top, right, bottom = crop
            im = im.crop((left, top, right, bottom))
        size = (round(w * 4), round(h * 4))
        if contain:
            im = ImageOps.pad(im,
                              size,
                              method=Image.Resampling.LANCZOS,
                              color="#f4f7f8")
        else:
            im = ImageOps.fit(im, size, method=Image.Resampling.LANCZOS)
        stream = io.BytesIO()
        im.save(stream, format="PNG")
        self.add("image",
                 x=x,
                 y=y,
                 width=w,
                 height=h,
                 href="data:image/png;base64," +
                 base64.b64encode(stream.getvalue()).decode())

    def photo(self,
              name: str,
              x: float,
              y: float,
              w: float,
              h: float,
              crop: Optional[Tuple[int, int, int, int]] = None,
              contain: bool = False,
              gui_key: Optional[str] = None) -> None:
        """Embed a cropped source photograph."""
        gui_manifest = ROOT / "data/gui-figure-manifest.json"
        gui_panel = None
        if gui_manifest.exists():
            gui_panel = json.loads(gui_manifest.read_text())["lookup"].get(
                gui_key or f"{self.height}:{x}:{y}")
        cycles_name = name.replace("balloons_refined_", "balloons_")
        if cycles_name == "bridge_exec_04_done":
            cycles_name = "bridge_cycles_win"
        elif cycles_name.rsplit("_", 1)[-1] in {"start", "win"}:
            domain, state = cycles_name.rsplit("_", 1)
            cycles_name = f"{domain}_cycles_{state}"
        cycles_source = FIG / "sources" / f"{cycles_name}.png"
        if RENDERER == "cycles" and cycles_source.exists():
            sources = [cycles_source]
        elif gui_panel:
            sources = [ROOT / gui_panel]
            crop = None
        else:
            sources = [
                p for p in (FIG / "sources").glob(f"{name}.*")
                if p.suffix.lower() in {'.png', '.jpg', '.jpeg'}
            ]
        # Local GUI re-renders can replace pixels without changing panel IDs,
        # archived evidence, or the layout. Never silently overwrite sources.
        overrides = ROOT / "data/figure-render-overrides.json"
        if overrides.exists():
            USED_IMAGES.add(overrides)
            replacements = json.loads(overrides.read_text())
            replacement = replacements.get(gui_key) or replacements.get(name)
            if replacement:
                sources = [ROOT / replacement["path"]]
                crop = replacement.get("crop")
        assert len(sources) == 1, (name, sources)
        self.image(sources[0], x, y, w, h, crop, contain)

    def recorded(self,
                 name: str,
                 x: float,
                 y: float,
                 w: float,
                 h: float,
                 crop: Optional[Sequence[int]] = None) -> None:
        """Embed the Cycles render of a recorded or constructed state."""
        source = FIG / "sources" / f"{name}_cycles.png"
        assert source.exists(), (
            f"{source} is missing; export and render the Figure 3 scenes "
            "first (docs/RENDERING.md).")
        self.image(source, x, y, w, h, crop)

    def scene(self,
              domain: str,
              state: str,
              x: float,
              y: float,
              w: float,
              h: float,
              gui_key: Optional[str] = None) -> None:
        """Embed a domain render, including the updated Balloons assets."""
        if domain == "Balloons":
            self.photo(f"balloons_refined_{state}",
                       x,
                       y,
                       w,
                       h,
                       crop=CROPS["balloons"],
                       gui_key=gui_key)
        else:
            self.photo(f"{domain.lower()}_{state}",
                       x,
                       y,
                       w,
                       h,
                       crop=CROPS[domain.lower()],
                       gui_key=gui_key)

    def save(self, name: str) -> None:
        """Write vector, PDF, and raster versions."""
        raw = ET.tostring(self.root, encoding="utf-8", xml_declaration=True)
        (OUTPUT / f"{name}.svg").write_bytes(raw)
        _DatedPDFSurface.convert(bytestring=raw,
                                 write_to=str(OUTPUT / f"{name}.pdf"))
        cairosvg.svg2png(bytestring=raw,
                         write_to=str(OUTPUT / f"{name}.png"),
                         scale=2)


def teaser() -> None:
    """Render the overview teaser."""
    # The five simulated domains and, past a divider, the real robot share
    # one pane below the method panels; the canvas ends just below the pane.
    left, gap, divider_gap, top = 16.0, 4.0, 9.0, 146.0
    w = (520 - left - 4 * gap - divider_gap) / 6
    h = w * 82 / 96
    d = Drawing(round(top + 2 * h + 35))
    for x, title, color in [(0, "Observe missing physics", RUST),
                            (184, "Program a simulator", TEAL),
                            (368, "Plan and solve", GREEN)]:
        d.rect(x, 1, 160, 127, fill=PANEL, stroke=EDGE)
        d.text(x + 8, 19, title, 10.4, color, "bold")
    # Illustrations built from the recorded training level: lifting one block
    # of a glued pair leaves its partner behind without a glue model.
    d.recorded("bridge_pair_predicted", 8, 28, 68, 64, BRIDGE_PAIR_CROP)
    d.recorded("bridge_pair_observed", 84, 28, 68, 64, BRIDGE_PAIR_CROP)
    d.text(42, 106, "Predicted", 9.5, RUST, anchor="middle")
    d.text(118, 106, "Observed", 9.5, GREEN, anchor="middle")
    d.text(80, 120, "The base simulator lacks glue.", 8.5, anchor="middle")
    d.text(192, 34, "Illustrative simulator extension", 8.1, MUTED)
    # The same editor colors as Figure 2's listing.
    for i, line in enumerate(GLUE_CODE):
        d.code(192, 48 + 9.6 * i, line, 7.4)
    d.text(192, 120, "Fit the program to observations.", 8.5)
    d.photo("bridge_exec_04_done",
            376,
            28,
            144,
            78,
            crop=BRIDGE_SOLVED_CROP,
            gui_key="304:370:57")
    d.text(448,
           120,
           "Rehearse, then complete the bridge.",
           8.5,
           anchor="middle")
    d.arrow(163, 66, 181, 66)
    d.arrow(347, 66, 365, 66)
    mechanisms = [
        "Contact & friction", "Glue & bonds", "Lift & damping",
        "Filling & heat", "Wind-driven motion"
    ]
    source_order = ["Boil", "Domino", "Fan", "Bridge", "Balloons"]
    d.rect(0, top - 8, 528, 2 * h + 42, fill=PANEL, stroke=EDGE)
    for center, label in [(top + h / 2, "Initial"),
                          (top + 1.5 * h + 1, "Final")]:
        d.text(0, 0, label, 9.6, MUTED, anchor="middle")
        d.root[-1].set("transform", f"translate(12 {center}) rotate(-90)")

    # Each column is named above its mechanism.
    def names(col: float, domain: str, mechanism: str) -> None:
        d.text(col + w / 2,
               top + 2 * h + 14,
               domain,
               10.2,
               TEAL,
               "bold",
               anchor="middle")
        d.text(col + w / 2,
               top + 2 * h + 25,
               mechanism,
               8,
               MUTED,
               anchor="middle")

    for i, (domain, mechanism) in enumerate(zip(DOMAINS, mechanisms)):
        col = left + i * (w + gap)
        source_x = source_order.index(domain) * 107
        d.scene(domain, "start", col, top, w, h, gui_key=f"281:{source_x}:18")
        d.scene(domain,
                "win",
                col,
                top + h + 1,
                w,
                h,
                gui_key=f"281:{source_x}:136")
        names(col, domain, mechanism)
    real = left + 5 * w + 4 * gap + divider_gap
    d.add("path",
          d=f"M{real - divider_gap / 2} {top} "
          f"L{real - divider_gap / 2} {top + 2 * h + 27}",
          stroke=EDGE,
          stroke_width=1)
    for row, state in enumerate(("start", "win")):
        d.image(FIG / "sources" / f"real_fan_domino_{state}.png", real,
                top + row * (h + 1), w, h)
    names(real, "Real robot", "Wind & mass")
    d.save("fig1_residual")


def _panel(d: Drawing,
           x: float,
           y: float,
           w: float,
           h: float,
           title: Sequence[Segment],
           color: str,
           caption: Sequence[Sequence[Segment]],
           size: float = 8.1,
           title_size: float = 9.8) -> None:
    """Draw one step of the method loop with its caption at the bottom."""
    d.rect(x, y, w, h, fill=PANEL, stroke=EDGE)
    d.rich(x + 8, y + 16, title, title_size, color, "bold")
    leading = size + 2.9
    for i, parts in enumerate(caption):
        d.rich(x + 8, y + h - 8 - leading * (len(caption) - 1 - i), parts,
               size)


# The glue program and its revision of the first contact test.
METHOD_CODE = GLUE_CODE[:3] + [
    "-     if centres_close(j):", "+     if faces_meet(j):"
] + GLUE_CODE[4:]
# Predicates the agent writes with the program; the monitor checks them.
PREDICATE_CODE = ["def Attached(a, b): ...", "def SeatedOn(s, l): ..."]
# Predicates checked after a skill, the last of which fails.
CHECKS = [("✓", "AtSite(leg0, site0)", GREEN),
          ("✓", "Attached(span1, span3)", GREEN),
          ("✗", "SeatedOn(span3, leg1)", RUST)]
# Figure 2 geometry: panel height, bottom-row offset, and canvas height.
PH = 140
YB = PH + 20
GRID_HEIGHT = YB + PH + 1


def _code(d: Drawing,
          x: float,
          y: float,
          w: float,
          size: float = 6.9,
          leading: float = 8.0) -> None:
    """Draw the program listing, its revised line, and its predicates."""
    gap = 4
    lines = len(METHOD_CODE) + len(PREDICATE_CODE)
    d.rect(x,
           y,
           w,
           leading * lines + gap + 5.2,
           fill="white",
           stroke=EDGE,
           radius=3)
    # Symbols tag the program and the predicate definitions at the right.
    d.text(x + w - 5, y + 9, "P", 8, MUTED, anchor="end", italic=True)
    d.text(x + w - 5,
           y + 9 + leading * len(METHOD_CODE) + gap,
           "Φ",
           8,
           MUTED,
           anchor="end",
           italic=True)
    for i, line in enumerate(METHOD_CODE):
        base = y + 9 + leading * i
        if line[0] in "+-":
            d.add("rect",
                  x=x + 1,
                  y=base - 0.92 * size,
                  width=w - 2,
                  height=leading,
                  fill="#fbe9e1" if line[0] == "-" else "#e3f1e7")
        d.code(x + 4, base, line, size)
    rule = y + 9 + leading * (len(METHOD_CODE) - 1) + 3 + gap / 2
    d.add("path",
          d=f"M{x + 4} {rule} L{x + w - 4} {rule}",
          stroke=EDGE,
          stroke_width=0.8,
          stroke_dasharray="2 2")
    for j, line in enumerate(PREDICATE_CODE):
        base = y + 9 + leading * (len(METHOD_CODE) + j) + gap
        d.code(x + 4, base, line, size)


def _parameter_draws() -> List[float]:
    """Return schematic equal-weight draws from the parameter belief."""
    rng = random.Random(11)
    return [min(0.95, max(0.05, rng.gauss(0.58, 0.095))) for _ in range(20)]


def _posterior(d: Drawing, px: float, py: float, pw: float, ph: float) -> None:
    """Draw the parameter belief and equal-weight draws from it."""
    oy = py + ph
    d.add("path",
          d=f"M{px} {py} L{px} {oy} L{px + pw + 4} {oy}",
          fill="none",
          stroke=MUTED,
          stroke_width=0.8)
    # The running example's parameter, the glue's cure time, labels the axis
    # from below so the rightmost draws stay clear.
    d.text(px + pw + 4, oy + 9, "t_cure", 7.4, MUTED, anchor="end", mono=True)
    curve = [(px + pw * k / 60,
              oy - 0.9 * ph * math.exp(-(k / 60 - 0.58)**2 / 0.018))
             for k in range(61)]
    outline = " L".join(f"{x:.2f} {y:.2f}" for x, y in curve)
    d.add("path",
          d=f"M{px} {oy} L{outline} L{px + pw} {oy} Z",
          fill=TEAL,
          fill_opacity=0.1,
          stroke="none")
    d.add("path", d="M" + outline, fill="none", stroke=TEAL, stroke_width=1.3)
    # Equal-weight draws sit on the axis like a rug, dense where the belief
    # is high.
    rng = random.Random(5)
    for theta in _parameter_draws():
        d.add("circle",
              cx=round(px + theta * pw, 2),
              cy=round(oy - 5 + rng.uniform(-2.2, 2.2), 2),
              r=1.9,
              fill=TEAL,
              fill_opacity=0.75)
    d.rich(px + 6, py + 6, [("q", "i"), "(", ("θ", "i"), ")"], 8.2, TEAL)


def _estimate(d: Drawing, px: float, py: float, pw: float, ph: float) -> None:
    """Draw the state belief for the running example as two stacked plots.

    A block's pose is read through sensor noise, so its draws sit close
    together. The joint's cure progress in x_res has no readings; it is
    replayed from the recorded contacts, and each draw's t_cure decides
    whether the joint has bonded by step t.
    """
    step = (pw - 15.8) / 17
    end = px + 6 + step * 17
    top = py + 12
    th = 0.47 * (ph - 22)
    low = top + th + 10
    lh = py + ph - low
    for y0, h, label in ((top, th, "pose"), (low, lh, "cure")):
        d.add("path",
              d=f"M{px} {y0} L{px} {y0 + h} L{px + pw + 4} {y0 + h}",
              fill="none",
              stroke=MUTED,
              stroke_width=0.8)
        d.text(0, 0, label, 7.4, MUTED, anchor="middle")
        d.root[-1].set("transform",
                       f"translate({px - 4} {y0 + h / 2}) rotate(-90)")
    d.text(px + pw + 2, low + lh - 4, "t", 9, MUTED, anchor="end", italic=True)
    rng = random.Random(3)
    truth = [
        top + 0.55 * th - 0.18 * th * math.tanh((k - 8) / 4) for k in range(18)
    ]
    for k, level in enumerate(truth):
        d.add("circle",
              cx=round(px + 6 + step * k, 2),
              cy=round(level + rng.gauss(0, 0.09 * th), 2),
              r=1.5,
              fill=RUST,
              fill_opacity=0.8)
    d.add("path",
          d="M" + " L".join(f"{px + 6 + step * k:.2f} {level:.2f}"
                            for k, level in enumerate(truth)),
          fill="none",
          stroke=TEAL,
          stroke_width=1.4)
    # The pose belief at step t: the rest-window mean with its shrunken
    # spread, and a few draws from it.
    spread = 0.07 * th
    pose_draws = [truth[-1] + z * spread for z in (-1.4, -0.6, 0.1, 0.7, 1.5)]
    # The cure progress is the same under every draw: it is replayed from the
    # recorded contacts and grows while the wet faces meet, from step 6 on.
    cure = [low + lh - 4 - 0.057 * lh * max(k - 6, 0) for k in range(18)]
    d.add("path",
          d="M" + " L".join(f"{px + 6 + step * k:.2f} {level:.2f}"
                            for k, level in enumerate(cure)),
          fill="none",
          stroke=TEAL,
          stroke_width=1.4)
    # Each draw's t_cure sits at its own height at step t. The progress has
    # passed the lower ones, so those draws hold the joint bonded (green, as
    # a passing check) and the higher ones do not yet (rust).
    thresholds = [cure[-1] + offset for offset in (13, 8, 4, -4, -8)]
    # The draws of both parts sit at the same step t, one column per plot.
    for draws in (pose_draws, thresholds):
        upper, lower = min(draws) - 3, max(draws) + 3
        d.add("rect",
              x=f"{end - 3.5:.2f}",
              y=f"{upper:.2f}",
              width=7,
              height=f"{lower - upper:.2f}",
              rx=3.5,
              fill=TEAL,
              fill_opacity=0.16)
    for level in pose_draws:
        d.add("circle",
              cx=f"{end:.2f}",
              cy=f"{level:.2f}",
              r=1.6,
              fill=TEAL,
              fill_opacity=0.8)
    for level in thresholds:
        d.add("circle",
              cx=f"{end:.2f}",
              cy=f"{level:.2f}",
              r=1.6,
              fill=GREEN if level > cure[-1] else RUST)
    d.text(px + 6, top + 7, "noisy oₜ", 7.6, RUST, italic=True)
    d.rich(px + 6, low + 7, ["hidden ", ("x", "i"), "ᵣₑₛ"], 7.6, MUTED)
    d.text(px + 6, low + 16, "no readings", 7.2, MUTED)


def _rehearse(d: Drawing, x: float, y: float, w: float) -> None:
    """Draw rollouts from joint draws of the belief."""
    sx0, sy0 = x + 16, y + 66
    gx = x + w - 40
    d.rect(gx, y + 42, 26, 40, fill="#e3f1e7", stroke=GREEN, radius=2)
    offsets = (34, 46, 51, 56, 61, 66, 71, 76, 80, 92)
    # Each rollout starts from its own state draw.
    starts = (-3.6, 2.4, -1.2, 3.6, 0.0, -2.4, 1.2, -4.4, 4.4, -0.4)
    for offset, start in zip(offsets, starts):
        end = y + offset
        inside = y + 42 <= end <= y + 82
        begin = sy0 + start
        d.add("path",
              d=f"M{sx0} {begin} Q{x + 0.49 * w:.2f} {begin - 22} "
              f"{gx + 13} {end}",
              fill="none",
              stroke=GREEN if inside else RUST,
              stroke_width=0.9,
              stroke_opacity=0.75)
        d.add("circle",
              cx=gx + 13,
              cy=end,
              r=1.8,
              fill=GREEN if inside else RUST)
        d.add("circle", cx=sx0, cy=begin, r=1.6, fill=TEAL, fill_opacity=0.85)
    d.text(sx0, sy0 + 16, "xₜ⁽ⁱ⁾", 9, TEAL, "bold", "middle", italic=True)
    d.rich(x + 8, y + 32,
           ["(1/", ("K", "i"), ") Σᵢ ",
            ("R", "i"), "(τ⁽ⁱ⁾) = 0.8"], 7.8, GREEN)


# Titles and captions of the loop's steps, which use the running glue example.
CODE_TITLE: List[Segment] = [
    "Write or revise ", ("P", "i"), " and ", ("Φ", "i")
]
# Inference takes two steps: a belief over the parameters, then one over the
# current state, whose hidden part depends on each parameter draw.
THETA_TITLE: List[Segment] = ["Infer ", ("θ", "i")]
STATE_TITLE: List[Segment] = ["Infer ", ("xₜ", "i")]
PLAN_TITLE: List[Segment] = [
    "Plan under (", ("θ", "i"), "⁽ⁱ⁾, ", ("xₜ", "i"), "⁽ⁱ⁾)"
]
MONITOR_TITLE = "Monitor with predicates"
ACT_CAPTION: List[List[Segment]] = [["The agent executes a plan or an"],
                                    [
                                        "experiment; observations join ",
                                        ("D", "i"), "."
                                    ]]
CODE_CAPTION: List[List[Segment]] = [["The agent writes the glue physics"],
                                     ["and predicates for skill outcomes."]]
THETA_CAPTION: List[List[Segment]] = [[
    "The data ", ("D", "i"), " give a belief ", ("q", "i"), "(", ("θ", "i"),
    ") over"
], ["the cure time; the dots are draws."]]
STATE_CAPTION: List[List[Segment]] = [
    ["Noisy readings pin down the pose;"],
    ["the joint's cure progress is hidden."],
    ["Short cure times bond it (green)."],
]
PLAN_CAPTION: List[List[Segment]] = [["Rehearse plans under the draws;"],
                                     ["run the likeliest, or experiment."]]
MONITOR_CAPTION: List[List[Segment]] = [[
    "The predicates ", ("Φ", "i"), " check each skill's"
], ["outcome; a failure stops the plan."]]


def _numbered(number: int, title: Sequence[Segment]) -> List[Segment]:
    return [f"{number}  ", *title]


def method() -> None:
    """Figure 2: the learning and planning loop with the method's internals.

    Six equal panels run clockwise from acting, through writing the
    program, inferring its parameters and the current state, and
    planning, to monitoring; the act and monitor steps hand each skill
    between them.
    """
    d = Drawing(GRID_HEIGHT)
    _panel(d, 0, 0, 160, PH, _numbered(1, ["Act"]), RUST, ACT_CAPTION)
    d.recorded("trajectory_bridge_train_1", 8, 22, 144, 86, BRIDGE_ACT_CROP)
    # Monitoring is numbered last, so readers meet the predicates in step 2
    # before seeing them checked.
    _panel(d, 0, YB, 160, PH, _numbered(6, [MONITOR_TITLE]), RUST,
           MONITOR_CAPTION)
    d.text(8, YB + 38, "after Place(span3, …)", 7.2, MUTED, mono=True)
    for i, (mark, atom, color) in enumerate(CHECKS):
        d.text(10, YB + 52 + 11.5 * i, mark, 8.6, color, "bold")
        d.text(22, YB + 52 + 11.5 * i, atom, 7.2, INK, mono=True)
    d.text(22, YB + 87, "→ stop: refit, replan, or revise", 7.2, RUST)
    # The monitor runs after each skill, and the plan continues only when
    # every check passes; a failure stops it inside the monitor panel.
    gap = (PH + YB) / 2
    d.arrow(70, PH + 3, 70, YB - 3)
    d.text(64, gap + 2.7, "after each skill", 7.4, MUTED, anchor="end")
    d.arrow(90, YB - 3, 90, PH + 3, color=GREEN)
    d.text(96, gap + 2.7, "if checks pass", 7.4, GREEN)
    _panel(d, 184, 0, 160, PH, _numbered(2, CODE_TITLE), TEAL, CODE_CAPTION)
    _code(d, 191, 22, 146)
    # Parameters come before the state because each parameter draw sets the
    # hidden part of its state draw.
    _panel(d, 368, 0, 160, PH, _numbered(3, THETA_TITLE), TEAL, THETA_CAPTION)
    _posterior(d, 382, 30, 128, 66)
    _panel(d, 368, YB, 160, PH, _numbered(4, STATE_TITLE), TEAL, STATE_CAPTION)
    _estimate(d, 382, YB + 14, 128, 82)
    _panel(d, 184, YB, 160, PH, _numbered(5, PLAN_TITLE), GREEN, PLAN_CAPTION)
    _rehearse(d, 184, YB + 6, 160)
    mid = PH / 2
    d.arrow(163, mid, 181, mid)
    d.text(172, mid - 6, "D", 8.2, INK, anchor="middle", italic=True)
    d.arrow(347, mid, 365, mid)
    d.arrow(448, PH + 3, 448, YB - 3)
    d.arrow(365, YB + mid, 347, YB + mid)
    d.arrow(181, YB + mid, 163, YB + mid, color=GREEN)
    d.text(172, YB + mid - 6, "plan", 7.8, GREEN, anchor="middle")
    d.save("fig2_method")


# Figure 3 and its appendix companion show one recorded run per domain
# (data/trajectories/stripes.json), from its experiments to the solved test.
MAIN_STRIPES = ["Bridge", "Balloons"]
APPENDIX_STRIPES = ["Domino", "Boil", "Fan"]
# Crops of the Cycles stripe renders, 900 pixels square except Balloons at
# 1280 by 800. Balloons keeps the ceiling and the chute base, so a balloon
# holding the box low stays in view.
STRIPE_CROPS = {
    "domino": (200, 300, 850, 808),
    "bridge": (260, 280, 760, 671),
    "balloons": (400, 40, 1170, 640),
    "boil": (240, 240, 830, 700),
    "fan": (150, 170, 800, 678),
}
# Frames are 92 wide on a 103 pitch; simulated frames share one height and
# the robot's photos keep their own aspect.
STRIPE_FH, ROBOT_FH = 72, 64
StripeFrame = Tuple[Path, Optional[Sequence[int]], str, str, bool]


class Stripe(NamedTuple):
    """One titled row of frames and the gap after which the agent learns."""
    title: str
    frames: List[StripeFrame]
    height: float
    learn_after: int
    learned: Sequence[str]


@functools.lru_cache(maxsize=None)
def _face(weight: str) -> Any:
    """The DejaVu Sans face that CairoSVG draws with, at 100 pixels."""
    pattern = "DejaVu Sans:bold" if weight == "bold" else "DejaVu Sans"
    path = subprocess.run(["fc-match", "--format=%{file}", pattern],
                          check=True,
                          capture_output=True,
                          text=True).stdout
    return ImageFont.truetype(path, 100)  # type: ignore[no-untyped-call]


def _advance(parts: Sequence[Segment], size: float, weight: str) -> float:
    """The width of a rich() line in DejaVu Sans.

    Its oblique faces share the upright advances, so italic parts are
    measured upright.
    """
    text = "".join(p if isinstance(p, str) else p[0] for p in parts)
    return float(_face(weight).getlength(text)) * size / 100


def _learning_bar(d: Drawing, x: float, y: float, h: float,
                  learned: Sequence[str]) -> None:
    """A teal bar in a frame gap, labelled with what the agent learns."""
    d.rect(x, y, 9, h, fill=TEAL, stroke="none", radius=3)
    parts: List[Segment] = ["learns "]
    for k, symbol in enumerate(learned):
        if k:
            parts.append(", ")
        parts.append((symbol, "i"))
    size = 6.2
    e = d.rich(-_advance(parts, size, "bold") / 2, 0, parts, size, "white",
               "bold")
    # Rotated to read upwards; the baseline sits right of the bar's center
    # so the lowercase letters center across the bar.
    e.set("transform",
          f"translate({x + 4.5 + 0.36 * size:g} {y + h / 2:g}) rotate(-90)")


def _stripe(d: Drawing, y: float, stripe: Stripe) -> None:
    """Draw one titled row of five captioned frames joined by arrows."""
    fh = stripe.height
    e = d.text(0, 0, stripe.title, 9.6, TEAL, "bold", "middle")
    e.set("transform", f"translate(12 {y + fh / 2}) rotate(-90)")
    for i, (source, crop, first, second, model) in enumerate(stripe.frames):
        x = 24 + i * 103
        if model:
            # A state from the agent's own model, not from the environment:
            # the image sits inside the frame box and the dashed border
            # runs along the box, with a clear margin between them.
            d.image(source, x + 2.2, y + 2.2, 87.6, fh - 4.4, crop)
        else:
            d.image(source, x, y, 92, fh, crop)
        if model:
            d.rect(x + 0.5,
                   y + 0.5,
                   91,
                   fh - 1,
                   fill="none",
                   stroke=TEAL,
                   radius=2,
                   dash="2.5 1.5",
                   width=1)
            d.rect(x + 3.5,
                   y + 3.5,
                   25,
                   9,
                   fill="white",
                   stroke="none",
                   radius=1.5)
            d.text(x + 16, y + 10.3, "model", 6.4, TEAL, "bold", "middle")
        d.text(x + 46, y + fh + 10, first, 7.3, anchor="middle")
        d.text(x + 46, y + fh + 19, second, 6.9, MUTED, anchor="middle")
        if i == stripe.learn_after:
            _learning_bar(d, x + 93, y, fh, stripe.learned)
        elif i < len(stripe.frames) - 1:
            d.arrow(x + 94, y + fh / 2, x + 101, y + fh / 2)


def _domain_stripes(domains: Sequence[str]) -> List[Stripe]:
    """Cycles frames and captions of the selected domains' stripes."""
    spec_path = ROOT / "data/trajectories/stripes.json"
    USED_IMAGES.add(spec_path)
    rows = {
        row["domain"]: row
        for row in json.loads(spec_path.read_text())["rows"]
    }
    stripes = []
    for domain in domains:
        key, row = domain.lower(), rows[domain]
        frames: List[StripeFrame] = [
            (FIG / "sources" / f"trajectory_{key}_stripe_{k}_cycles.png",
             STRIPE_CROPS[key], frame["caption"], frame["detail"], "states"
             in frame) for k, frame in enumerate(row["frames"])
        ]
        stripes.append(
            Stripe(domain, frames, STRIPE_FH, row["learn_after"],
                   row["learned"]))
    return stripes


def _robot_stripe() -> Stripe:
    """The real-robot run: two probes, then the test after the patch moved.

    Between them the agent writes its wind program and fits the masses,
    friction and wind parameters.
    """
    robot_archive = ROOT / "data/trajectories/real_fan_domino.json"
    USED_IMAGES.add(robot_archive)
    robot = json.loads(robot_archive.read_text())
    slides = {m["episode"]: m for m in robot["measured"]}
    predicted = robot["test_plan"]["predicted_slide_cm"]
    seconds = [
        f"slides {slides[1]['slide_cm']:.1f} cm",
        f"slides {slides[2]['slide_cm']:.1f} cm, flat",
        "patch moved to 0.51 m",
        "one gust",
        f"{slides[3]['slide_cm']:.1f} cm; pred. "
        f"{predicted[0]:.1f}±{predicted[1]:.1f}",
    ]
    frames: List[StripeFrame] = [
        (FIG / "sources" / f"{frame['name']}.png", None, frame["label"],
         second, False) for frame, second in zip(robot["frames"], seconds)
    ]
    return Stripe("Real robot", frames, ROBOT_FH, 1, ("P", "θ"))


def _stripes_figure(name: str, stripes: Sequence[Stripe]) -> None:
    """Stack stripes, each a frame row plus its two caption lines."""
    d = Drawing(sum(stripe.height + 30 for stripe in stripes) - 6)
    y = 0.0
    for stripe in stripes:
        _stripe(d, y, stripe)
        y += stripe.height + 30
    d.save(name)


def trajectories() -> None:
    """Figure 3: recorded runs in the main-text domains and on the robot."""
    _stripes_figure("fig3_trajectories",
                    _domain_stripes(MAIN_STRIPES) + [_robot_stripe()])


def appendix_trajectories() -> None:
    """Appendix figure: recorded runs in the remaining domains."""
    _stripes_figure("figA_trajectories", _domain_stripes(APPENDIX_STRIPES))


def build_overview_figures() -> None:
    """Rebuild Figures 1 to 3 and the appendix trajectories from archived
    sources."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    USED_IMAGES.clear()
    teaser()
    method()
    trajectories()
    appendix_trajectories()
    inputs = USED_IMAGES | {
        Path(__file__), ROOT / "data/gui-figure-manifest.json"
    }
    outputs = [
        OUTPUT / f"{name}.{ext}"
        for name in ("fig1_residual", "fig2_method", "fig3_trajectories",
                     "figA_trajectories") for ext in ("pdf", "svg", "png")
    ]

    def hashes(paths: Iterable[Path], base: Path) -> Dict[str, str]:
        return {
            str(p.relative_to(base)):
            hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(paths)
        }

    (ROOT / "data/overview-figure-build-manifest.json").write_text(
        json.dumps(
            {
                "generated_by":
                "scripts/paper_figures/build_figures.py --overview; "
                "do not edit manually",
                "inputs":
                hashes(inputs, ROOT),
                "outputs":
                hashes(outputs, OUTPUT)
            },
            indent=2) + "\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overview", action="store_true", required=True)
    parser.add_argument("--renderer", choices=("gui", "cycles"), default="gui")
    args = parser.parse_args()
    RENDERER = args.renderer
    build_overview_figures()
