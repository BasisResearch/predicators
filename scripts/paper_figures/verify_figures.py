"""Verify migrated figure inputs, outputs, and execution-frame provenance."""
import hashlib
import itertools
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List

# Optional figure-authoring dependency.
import pymupdf  # type: ignore[import-not-found] # pylint: disable=import-error

ROOT = Path(__file__).resolve().parent
PAPER = Path(
    os.environ.get("EMPIRIC_PAPER_ROOT",
                   str(ROOT.parents[2] / "sim-predicator-paper"))).resolve()
OUTPUT = PAPER / "figures"
SVG = "{http://www.w3.org/2000/svg}"


def _outlined_boxes(svg: ET.ElementTree, width: float) -> List[pymupdf.Rect]:
    """Return a figure's outlined rectangles in PDF points."""
    left, top, view_width, _ = map(float,
                                   svg.getroot().get("viewBox", "").split())
    scale = width / view_width
    boxes = []
    for rect in svg.iter(f"{SVG}rect"):
        if rect.get("stroke", "none") == "none":
            continue
        x, y = float(rect.get("x", 0)), float(rect.get("y", 0))
        w, h = float(rect.get("width", 0)), float(rect.get("height", 0))
        boxes.append(
            pymupdf.Rect((x - left) * scale, (y - top) * scale,
                         (x + w - left) * scale, (y + h - top) * scale))
    return boxes


def main() -> None:
    """Check render provenance and PDF bounds against the saved manifest."""
    manifest = json.loads(
        (ROOT / "data/overview-figure-build-manifest.json").read_text())
    for group, base in (("inputs", ROOT), ("outputs", OUTPUT)):
        for name, digest in manifest[group].items():
            assert hashlib.sha256(
                (base / name).read_bytes()).hexdigest() == digest, name
    for name, count in (("fig1_residual", 15), ("fig2_method", 1),
                        ("fig3_trajectories", 15), ("figA_trajectories", 15)):
        svg = ET.parse(OUTPUT / f"{name}.svg")
        assert len(svg.findall(f".//{SVG}image")) == count
        with pymupdf.open(OUTPUT / f"{name}.pdf") as doc:
            boxes = _outlined_boxes(svg, doc[0].rect.width)
            words = doc[0].get_text("words")
            for word in words:
                assert doc[0].rect.contains(pymupdf.Rect(word[:4])), (name,
                                                                      word)
                # Text sits wholly inside or outside each outlined box; the
                # inset ignores the line spacing around the glyphs.
                inner = pymupdf.Rect(word[:4]) + (0.5, 0.5, -0.5, -0.5)
                for box in boxes:
                    assert box.contains(inner) or not box.intersects(inner), (
                        name, word[4], box)
            # No two words overlap, so neighbouring captions stay apart.
            glyphs = [(pymupdf.Rect(w[:4]) + (0.5, 0.5, -0.5, -0.5), w[4])
                      for w in words]
            for (a, text_a), (b, text_b) in itertools.combinations(glyphs, 2):
                assert not a.intersects(b), (name, text_a, text_b)
    archive = json.loads((ROOT / "data/trajectories/figure3.json").read_text())
    robot = json.loads(
        (ROOT / "data/trajectories/real_fan_domino.json").read_text())
    frames = [frame for row in archive["rows"] for frame in row["frames"]]
    for row in archive["rows"]:
        assert row["frames"][-1]["event"]["state"] == "WIN"
    # Mid-skill states have no GUI render; they are archived by step alone.
    for frame in ([f for f in frames if f["sha256"]] + robot["frames"] +
                  robot["teaser_frames"]):
        source = ROOT / "figures/sources" / (frame["name"] + ".png")
        assert hashlib.sha256(
            source.read_bytes()).hexdigest() == frame["sha256"], frame["name"]
    print("PASS: input/output hashes, image counts, text bounds, text inside "
          "boxes, no overlapping words, and frame provenance")


if __name__ == "__main__":
    main()
