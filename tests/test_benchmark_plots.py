"""Keep repaired Oracle cohorts distinct in benchmark plots and reports."""
import json
from pathlib import Path
from typing import Any

from scripts.plotting import monitor_benchmark_arms as monitor
from scripts.plotting import plot_benchmark_arms as plot


def test_oracle_r2_layout_and_scope() -> None:
    """Both Oracle entries lead the list, but r2 covers only its pilot."""
    assert plot.ARMS[:2] == ["oracle_dynamics", "oracle_dynamics_r2"]
    assert plot.LABELS[:2] == ["Oracle dynamics", "Oracle dynamics r2"]
    assert len(plot.ARMS) == len(plot.LABELS) == len(plot.COLORS)
    assert [i for _, start, end in plot.GROUPS
            for i in range(start, end)] == list(range(len(plot.ARMS)))
    for domain, dirs in plot.DOMAINS:
        assert len(dirs["MB_r2"]) == 2
        assert all("mb_opus_benchmark_r2/seed" in d for d in dirs["MB_r2"])
        assert {d[-1] for d in dirs["MB_r2"]} == {"3", "4"}
        assert all("benchmark_r1/" in d for d in dirs["oracle_dynamics"])
        new = dirs["oracle_dynamics_r2"]
        assert len(new) == (3 if domain in ("Domino",
                                            "Bridge (4-span)") else 0)
        assert all("benchmark_r2/" in d for d in new)


def test_oracle_r2_finished_only_and_report(tmp_path: Path,
                                            monkeypatch: Any) -> None:
    """An unfinished r2 neither replaces r1 nor enters the denominator."""
    monkeypatch.setattr(plot, "LOGS", str(tmp_path))
    dirs = dict(plot.DOMAINS)["Domino"]
    for arm, finished in (("oracle_dynamics", True), ("oracle_dynamics_r2",
                                                      False)):
        run = tmp_path / dirs[arm][0] / "run_20260919"
        run.mkdir(parents=True)
        (run / "scorecard.json").write_text(json.dumps({
            "end_reason":
            "completed" if finished else None,
            "levels": [{
                "won": True
            }],
            "totals": {
                "total_steps": 100,
                "total_resets": 0
            }
        }),
                                            encoding="utf-8")
    rows = plot.records()
    assert len(rows) == 1 and rows[0]["arm"] == "oracle_dynamics"
    original = monitor.REPORT.read_text(encoding="utf-8")
    report = monitor.report_text(original, rows, vars(plot), "test")
    assert "Oracle dynamics r2: 0/6 seeds finished" in report
    unfinished = report.split("## Unfinished runs", 1)[1]
    assert unfinished.count("Oracle dynamics r2") == 6
    assert "| Boil (two-jug) | 2. Oracle dynamics r2" not in report
    assert monitor.report_text(report, rows, vars(plot), "test") == report
