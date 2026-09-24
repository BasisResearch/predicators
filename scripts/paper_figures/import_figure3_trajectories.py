"""Archive the selected recorded Bridge frames and their source events for
Figure 3."""
import hashlib
import json
import pickle
import shutil
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parent
LOGS = ROOT.parents[1] / "logs/agent_continual"
RUN = "bridge-mb_opus_span_transfer_r2/seed0/run_20260916_190710"
# Each row follows one level of the same run. Steps are within the level.
# Steps 122 (mid-dip) and 1290 (mid-carry) fall inside a skill, so they have
# no GUI render and are archived by their recorded state index alone.
# Both levels share the camera, so one crop keeps the table in the same place
# in both rows. It spans the tabletop and the gripper above it and drops most
# of the table's front face.
CROP = (250, 224, 820, 670)
SELECTION = [
    ("bridge_train", "L01", "train",
     ((0, "Initial scene"), (122, "Dip a block end"),
      (1290, "Row lifts as one"), (1362, "Level solved")), CROP),
    ("bridge_test", "L02", "test",
     ((0, "New task"), (563, "Apply glue"), (1652, "Re-seat joint"),
      (1702, "Lift assembly"), (1940, "Bridge solved")), CROP),
]


def main() -> None:
    """Archive the selected frames, their events, and the scorecard."""
    archive = ROOT / "data/trajectories"
    archive.mkdir(parents=True, exist_ok=True)
    source = LOGS / RUN
    shutil.copy2(source / "scorecard.json", archive / "bridge-scorecard.json")
    rows = []
    for key, level, split, frames, crop in SELECTION:
        events = [
            json.loads(line)
            for line in (source / level /
                         "index.jsonl").read_text().splitlines()
        ]
        start = next(e for e in events if e["event"] == "level_start")
        offset = start["run_steps"]
        recording = source / level / "episodes.pkl"
        recording_bytes = recording.read_bytes()
        episodes = pickle.loads(recording_bytes)  # Trusted local record.
        episode = next(ep for ep in episodes if ep["end"] == "win")
        row: Dict[str, Any] = dict(
            key=key,
            domain="Bridge",
            run=RUN,
            level=level,
            split=split,
            seed=0,
            crop=crop,
            recording=str(recording.relative_to(LOGS)),
            recording_sha256=hashlib.sha256(recording_bytes).hexdigest(),
            frames=[])
        for i, (step, label) in enumerate(frames):
            assert step < len(episode["states"]), (key, step)
            matches = [
                e for e in events if e.get("render") and (
                    (step == 0 and e["event"] == "level_start") or
                    (step > 0 and e["event"] == "invoke"
                     and e.get("level_steps") == step))
            ]
            assert len(matches) <= 1, (key, step, matches)
            name = f"trajectory_{key}_{i}"
            frame: Dict[str, Any] = dict(name=name,
                                         label=label,
                                         level_step=step,
                                         run_step=offset + step,
                                         event=None,
                                         source=None,
                                         sha256=None)
            if matches:
                event = matches[0]
                original = (source / level / "renders" /
                            Path(event["render"]).name)
                dest = ROOT / "figures/sources" / f"{name}.png"
                shutil.copy2(original, dest)
                frame.update(event=event,
                             source=str(original.relative_to(LOGS)),
                             sha256=hashlib.sha256(
                                 dest.read_bytes()).hexdigest())
            row["frames"].append(frame)
        assert row["frames"][-1]["event"]["state"] == "WIN"
        rows.append(row)
    archive_record = dict(
        generated_by="scripts/paper_figures/import_figure3_trajectories.py;"
        " do not edit manually",
        description="Recorded Bridge states from one run, not simulated "
        "predictions; level_step counts within a level and run_step counts "
        "environment steps from the start of the run.",
        rows=rows)
    (archive /
     "figure3.json").write_text(json.dumps(archive_record, indent=2) + "\n")


if __name__ == "__main__":
    main()
