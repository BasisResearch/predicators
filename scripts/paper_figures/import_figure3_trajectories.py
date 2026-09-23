"""Archive selected real execution frames and their source events for Figure
3."""
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parent
LOGS = ROOT.parents[1] / "logs/agent_continual"
SELECTION = [
    ("Bridge", "bridge-mb_opus_span_transfer_r2/seed0/run_20260916_190710",
     "L02", (0, 563, 1652, 1702,
             1940), ("Initial scene", "Apply glue", "Re-seat joint",
                     "Lift assembly", "Bridge solved"), (270, 280, 810, 860)),
    ("Balloons", "balloons-mb_opus_compose_r2/seed0/run_20260917_082044",
     "L03", (0, 124, 241, 264,
             348), ("Initial scene", "One attached", "Two attached",
                    "Release third", "Target reached"), (290, 170, 760, 720)),
]


def main() -> None:
    """Archive the selected source frames and their scorecards."""
    archive = ROOT / "data/trajectories"
    archive.mkdir(parents=True, exist_ok=True)
    rows = []
    for domain, run, level, steps, labels, crop in SELECTION:
        source = LOGS / run
        events = [
            json.loads(line)
            for line in (source / level /
                         "index.jsonl").read_text().splitlines()
        ]
        shutil.copy2(source / "scorecard.json",
                     archive / f"{domain.lower()}-scorecard.json")
        row: Dict[str, Any] = dict(domain=domain,
                                   run=run,
                                   level=level,
                                   seed=0,
                                   crop=crop,
                                   frames=[])
        for i, (step, label) in enumerate(zip(steps, labels)):
            matches = [
                e for e in events if e.get("render") and (
                    (step == 0 and e["event"] == "level_start") or
                    (step > 0 and e["event"] == "invoke"
                     and e.get("level_steps") == step))
            ]
            assert len(matches) == 1, (domain, step, matches)
            event = matches[0]
            original = source / level / "renders" / Path(event["render"]).name
            name = f"trajectory_{domain.lower()}_{i}"
            dest = ROOT / "figures/sources" / f"{name}.png"
            shutil.copy2(original, dest)
            row["frames"].append(
                dict(name=name,
                     label=label,
                     level_step=step,
                     run_step=event["run_steps"],
                     event=event,
                     source=str(original.relative_to(LOGS)),
                     sha256=hashlib.sha256(dest.read_bytes()).hexdigest()))
        assert row["frames"][-1]["event"]["state"] == "WIN"
        rows.append(row)
    (archive / "figure3.json").write_text(
        json.dumps(dict(
            generated_by=
            "scripts/import_figure3_trajectories.py; do not edit manually",
            description="Recorded execution images, not simulated predictions; "
            "step counts are within each test level.",
            rows=rows),
                   indent=2) + "\n")


if __name__ == "__main__":
    main()
