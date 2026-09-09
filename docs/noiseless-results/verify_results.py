"""Verify the archived noiseless sweep without access to experiment logs."""
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path


def main() -> None:
    """Check archive integrity and recompute the published statistics."""
    root = Path(__file__).resolve().parent
    provenance = json.loads((root / "provenance.json").read_text())
    for name, expected in provenance["artifacts_sha256"].items():
        actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
        assert actual == expected, name
    snapshots = json.loads((root / "scorecard_snapshots.json").read_text())
    manifest = json.loads((root / "manifest.json").read_text())
    with (root / "per_seed.csv").open(newline="") as stream:
        seeds = list(csv.DictReader(stream))
    with (root / "summary.csv").open(newline="") as stream:
        summaries = list(csv.DictReader(stream))
    domains = {"Balloons", "Boil", "Bridge", "Domino", "Fan"}
    expected_keys = {(domain, arm, seed) for domain in domains
                     for arm in ("MB", "MF") for seed in range(3)}
    assert len(seeds) == len(snapshots) == len(manifest["runs"]) == 30
    assert {(row["domain"], row["arm"], int(row["seed"]))
            for row in seeds} == expected_keys
    assert [entry["selection"] for entry in snapshots] == manifest["runs"]
    cards = {entry["selection"]["scorecard"]: entry["scorecard"]
             for entry in snapshots}
    for row in seeds:
        card = cards[row["scorecard"]]
        assert card["seed"] == int(row["seed"])
        assert card["finished_at"] and card["end_reason"] != "crash"
        assert card["end_reason"] == row["end_reason"]
        assert all(card.get(key, 0) == 0 for key in (
            "obs_noise_position", "obs_noise_orientation", "obs_noise_scalar"))
        levels = card["levels"]
        assert len(levels) == (3 if row["domain"] == "Balloons" else 2)
        assert math.isclose(float(row["solve_rate"]),
                            sum(bool(level["won"]) for level in levels) /
                            len(levels))
        for metric in ("steps", "resets"):
            total = sum(level[metric] for level in levels)
            assert total == int(row[metric]) == card["totals"]["total_" + metric]
    assert len(summaries) == 10
    assert {(row["domain"], row["arm"]) for row in summaries} == {
        (domain, arm) for domain in domains for arm in ("MB", "MF")}
    for summary in summaries:
        selected = [row for row in seeds if row["domain"] == summary["domain"]
                    and row["arm"] == summary["arm"]]
        assert len(selected) == int(summary["n_seeds"]) == 3
        for metric in ("solve_rate", "steps", "resets"):
            values = [float(row[metric]) for row in selected]
            assert math.isclose(statistics.mean(values),
                                float(summary[metric + "_mean"]))
            assert math.isclose(statistics.stdev(values),
                                float(summary[metric + "_sd"]))
    print("Verified 30 noiseless runs and all 10 aggregate rows.")


if __name__ == "__main__":
    main()
