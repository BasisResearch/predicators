"""Verify migrated figure inputs, outputs, and execution-frame provenance."""
import hashlib
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

# Optional figure-authoring dependency.
import pymupdf  # type: ignore[import-not-found] # pylint: disable=import-error

ROOT = Path(__file__).resolve().parent
PAPER = Path(
    os.environ.get("EMPIRIC_PAPER_ROOT",
                   str(ROOT.parents[2] / "sim-predicator-paper"))).resolve()
OUTPUT = PAPER / "figures"


def main() -> None:
    """Check render provenance and PDF bounds against the saved manifest."""
    manifest = json.loads(
        (ROOT / "data/overview-figure-build-manifest.json").read_text())
    for group, base in (("inputs", ROOT), ("outputs", OUTPUT)):
        for name, digest in manifest[group].items():
            assert hashlib.sha256(
                (base / name).read_bytes()).hexdigest() == digest, name
    for name, count in (("fig1_residual", 15), ("fig2_method", 1),
                        ("fig3_trajectories", 14)):
        svg = ET.parse(OUTPUT / f"{name}.svg")
        assert len(
            svg.findall(".//{http://www.w3.org/2000/svg}image")) == count
        with pymupdf.open(OUTPUT / f"{name}.pdf") as doc:
            for word in doc[0].get_text("words"):
                assert doc[0].rect.contains(pymupdf.Rect(word[:4])), (name,
                                                                      word)
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
    print("PASS: input/output hashes, image counts, text bounds, "
          "and frame provenance")


if __name__ == "__main__":
    main()
