"""Prepare cropped Cycles panels for the editable Figma figures.

This script applies the same crops used by the paper compositor and
writes a manifest that maps every output image to its Figma node.
"""
import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

from build_figures import BRIDGE_MECHANISM_CROP, BRIDGE_SOLVED_CROP, CROPS
from PIL import Image

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "figures/sources"
Crop = Optional[Tuple[int, int, int, int]]

# Figure 2 uses wide image boxes. These crops preserve each action's gripper
# context while matching the final panel aspect ratio closely enough that
# Figma's FILL mode does not zoom the image a second time.
FIGURE2_CROPS = {
    "explore": (210, 260, 820, 600),
    "hypothesize": (210, 220, 820, 560),
    "design": (210, 230, 820, 570),
    "execute": (210, 170, 820, 510),
    "solve": (210, 220, 820, 560),
}

# The order is visual only. The node IDs are stable identifiers in the EMPIRIC
# Figma file, not positions in an upload batch.
ASSETS = {
    "fig1_bridge_wet.png":
    ("117:71", "bridge_wet_lift_cycles.png", BRIDGE_MECHANISM_CROP),
    "fig1_bridge_bonded.png":
    ("117:72", "bridge_bonded_lift_cycles.png", BRIDGE_MECHANISM_CROP),
    "fig1_bridge_solved.png":
    ("117:84", "trajectory_bridge_4_cycles.png", BRIDGE_SOLVED_CROP),
    "fig1_boil.png": ("117:91", "boil_cycles_start.png", CROPS["boil"]),
    "fig1_domino.png": ("117:94", "domino_cycles_start.png", CROPS["domino"]),
    "fig1_fan.png": ("117:97", "fan_cycles_start.png", CROPS["fan"]),
    "fig1_bridge.png": ("117:100", "bridge_cycles_start.png", CROPS["bridge"]),
    "fig1_balloons.png":
    ("117:103", "balloons_cycles_start.png", (240, 260, 1050, 790)),
    "fig2_explore.png": ("134:3", "trajectory_bridge_0_cycles.png",
                         FIGURE2_CROPS["explore"]),
    "fig2_hypothesize.png": ("134:6", "trajectory_bridge_1_cycles.png",
                             FIGURE2_CROPS["hypothesize"]),
    "fig2_design.png": ("134:9", "trajectory_bridge_2_cycles.png",
                        FIGURE2_CROPS["design"]),
    "fig2_execute.png": ("134:12", "bridge_wet_lift_cycles.png",
                         FIGURE2_CROPS["execute"]),
    "fig2_solve.png": ("134:19", "trajectory_bridge_4_cycles.png",
                       FIGURE2_CROPS["solve"]),
}


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """Write reproducible, already-cropped Figma image fills."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest: Dict[str, object] = {
        "generated_by": "scripts/paper_figures/prepare_figma_assets.py",
        "file_key": "PgS1btsW3SH28tvW52xjrk",
        "assets": {},
    }
    entries = manifest["assets"]
    assert isinstance(entries, dict)
    for output_name, (node_id, source_name, crop) in ASSETS.items():
        source = SOURCE / source_name
        destination = args.output_dir / output_name
        image = Image.open(source).convert("RGB")
        if crop is not None:
            image = image.crop(crop)
        image.save(destination, format="PNG", optimize=True)
        entries[output_name] = {
            "node_id": node_id,
            "source": str(source.relative_to(ROOT)),
            "source_sha256": _digest(source),
            "crop": crop,
            "output_sha256": _digest(destination),
            "size": list(image.size),
        }
    (args.output_dir /
     "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
