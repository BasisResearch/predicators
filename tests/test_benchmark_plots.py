"""Keep repaired Oracle cohorts distinct in benchmark plots and reports."""
import json
from pathlib import Path
from typing import Any

from scripts.plotting import monitor_benchmark_arms as monitor
from scripts.plotting import plot_benchmark_arms as plot


def test_assets_pilot_selection() -> None:
    """Ten intended seeds, never cancelled maze runs or paper records."""
    paths = [(domain, path) for domain, dirs in plot.DOMAINS
             for path in dirs["from_assets"]]
    assert len(paths) == 10
    assert {d
            for d, _ in paths} == {
                "Domino", "Bridge (4-span)", "Balloons (composition)",
                "Boil (2-jug)", "Fan (ramp transfer)"
            }
    assert all("from_assets_opus_pilot_r1/seed" in p for _, p in paths)
    assert not any("/fan-from_assets" in p for _, p in paths)
    assert "from_assets" not in plot.PAPER_ARMS
    report = monitor.report_text(monitor.REPORT.read_text(), [], vars(plot),
                                 "test")
    assert "EMPIRIC from assets: 0/10 seeds finished" in report
    assert "Cancelled Fan maze pilots are excluded" in report
    assert report.count("## EMPIRIC from assets development") == 1


def test_paper_cohort_selection() -> None:
    """Replace whole Oracle cohorts without pooling rounds or changing
    input."""
    rows = [
        dict(domain=d, arm=a, seed=s, source=f"{d}/{a}/{s}")
        for d, dirs in plot.DOMAINS for a, paths in dirs.items()
        for s in [int(path.rsplit("seed", 1)[1]) for path in paths]
    ]
    selected = plot.paper_records(rows)
    assert len(selected) == 5 * 7 * 5
    assert len({(r["domain"], r["arm"], r["seed"]) for r in selected}) == 175
    assert {r["arm"] for r in selected} == set(plot.PAPER_ARMS)
    for row in selected:
        expected = row["arm"]
        assert row["source_arm"] == expected
        assert row["source"] == f"{row['domain']}/{expected}/{row['seed']}"
    assert all("source_arm" not in row for row in rows)


def test_current_fan_is_only_ramp() -> None:
    """Both figures use ramp, never fallback or pool historical variants."""
    assert plot.PAPER_DOMAINS == ("Domino", "Bridge (4-span)",
                                  "Balloons (composition)", "Boil (2-jug)",
                                  "Fan (ramp transfer)")
    assert plot.DISPLAY_TITLE["Fan (ramp transfer)"] == "Fan"


def test_benchmark_fan_is_the_ramp_transfer() -> None:
    """The benchmark's Fan setting is the reviewed ramp layout the paper
    figures plot."""
    from scripts.cluster_utils import \
        parse_configs  # pylint: disable=import-outside-toplevel
    config = next(parse_configs("empiric/envs.yaml"))
    fan = config["ENVS"]["fan"]["FLAGS"]
    assert fan["fan_ramp_transfer"]
    assert fan["fan_inertial_transfer"]
    assert fan["fan_exposed_transfer"]
    assert fan["fan_ramp_rise"] == 0.003
    assert fan["fan_ramp_landing_extension"] == 0.10


def test_oracle_r2_layout_and_scope() -> None:
    """Each paper domain selects one five-seed Oracle cohort."""
    assert plot.ARMS[:2] == ["oracle_dynamics", "MB"]
    assert plot.LABELS[:2] == ["Oracle dynamics", "EMPIRIC"]
    assert len(plot.ARMS) == len(plot.LABELS) == len(plot.COLORS)
    assert [i for _, start, end in plot.GROUPS
            for i in range(start, end)] == list(range(len(plot.ARMS)))
    for domain, dirs in plot.DOMAINS:
        if domain not in plot.PAPER_DOMAINS:
            continue
        assert "MB_r2" not in dirs
        assert len(dirs["MB"]) == 5
        assert {d[-1] for d in dirs["MB"]} == set("01234")
        if domain == "Fan (ramp transfer)":
            assert len(dirs["oracle_dynamics"]) == 5
            assert all("fan_prompt_r2" in p for p in dirs["oracle_dynamics"])
            assert all("ramp_skill_repair_r1" in p for p in dirs["MB"])
            continue
        assert all("mb_opus_benchmark_r2/seed" in d for d in dirs["MB"][3:])
        if domain in ("Boil (2-jug)", "Balloons (composition)"):
            assert dirs["MB"][2].endswith("mb_opus_benchmark_r2/seed2")
        oracle = dirs["oracle_dynamics"]
        assert len(oracle) == 5
        assert "oracle_dynamics_r2" not in dirs
        for seed, directory in enumerate(oracle):
            round_id = "r2" if domain in (
                "Domino", "Bridge (4-span)") or seed >= 3 else "r1"
            assert f"benchmark_{round_id}/seed{seed}" in directory


def test_oracle_r2_finished_only_and_report(tmp_path: Path,
                                            monkeypatch: Any) -> None:
    """An unfinished r2 neither replaces r1 nor enters the denominator."""
    monkeypatch.setattr(plot, "LOGS", str(tmp_path))
    dirs = dict(plot.DOMAINS)["Domino"]
    for round_id, finished in (("r1", True), ("r2", False)):
        directory = dirs["oracle_dynamics"][0].replace(
            "benchmark_r2/", f"benchmark_{round_id}/")
        run = tmp_path / directory / "run_20260919"
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
    assert not rows
    original = monitor.REPORT.read_text(encoding="utf-8")
    report = monitor.report_text(original, rows, vars(plot), "test")
    assert "Oracle dynamics: 0/25 seeds finished" in report
    unfinished = report.split("## Unfinished runs", 1)[1]
    assert unfinished.count("Oracle dynamics") == 35
    assert "| Boil (two-jug) | 1. Oracle dynamics" in report
    assert monitor.report_text(report, rows, vars(plot), "test") == report


def test_empiric_r2_seed_ids_and_completion_count(tmp_path: Path,
                                                  monkeypatch: Any) -> None:
    """Pool the prospective seeds while retaining their source cohort."""
    monkeypatch.setattr(plot, "LOGS", str(tmp_path))
    directory = dict(plot.DOMAINS)["Domino"]["MB"][3]
    run = tmp_path / directory / "run_20260919"
    run.mkdir(parents=True)
    (run / "scorecard.json").write_text(json.dumps({
        "end_reason": "all_levels_won",
        "levels": [{
            "won": True
        }],
        "totals": {
            "total_steps": 265,
            "total_resets": 0
        }
    }),
                                        encoding="utf-8")
    rows = plot.records()
    assert len(rows) == 1 and rows[0]["seed"] == 3
    assert rows[0]["arm"] == "MB"
    assert rows[0]["source_arm"] == "MB_r2"
    assert plot.paper_records(rows)[0]["source_arm"] == "MB_r2"
    report = monitor.report_text(monitor.REPORT.read_text(), rows, vars(plot),
                                 "test")
    averages = report.split("## Per-seed results", 1)[0]
    assert "| 265 (n=1) | 0 | 1/5 |" in averages
    details, unfinished = report.split("## Unfinished runs", 1)
    assert "| 2. EMPIRIC | 3 | 1/1 | 265 |" in details
    assert "| Domino (high-friction turn) | 2. EMPIRIC | 4 |" in unfinished
    assert "| Domino (high-friction turn) | 2. EMPIRIC | 3 |" not in unfinished
    assert "EMPIRIC r2" not in plot.LABELS


def test_fan_transfer_is_separate_and_excluded_from_paper() -> None:
    """The pilot has two matched arms without changing paper cohorts."""
    assert len(plot.PAPER_DOMAINS) == 5
    assert plot.FAN_TRANSFER in dict(plot.DOMAINS)
    directories = dict(plot.DOMAINS)[plot.FAN_TRANSFER]
    assert set(directories) == set(plot.ARMS)
    for arm, paths in directories.items():
        assert len(paths) == (2 if arm in ("MB", "MF") else 0)
        assert all("transfer_pilot_r1" in path for path in paths)
    pilot = dict(domain=plot.FAN_TRANSFER,
                 arm="MB",
                 seed=0,
                 won=2,
                 levels=2,
                 steps=450,
                 resets=0,
                 source="pilot")
    assert not plot.paper_records([pilot])
    report = monitor.report_text(monitor.REPORT.read_text(), [pilot],
                                 vars(plot), "test")
    assert "EMPIRIC: 0/25 seeds finished" in report
    assert "Fan transfer pilot: EMPIRIC 1/2 and Direct agent 0/2" in report
    assert "| Fan (exposed transfer) | 2. EMPIRIC | 1/1 (100%)" in report
    unfinished = report.split("## Unfinished runs", 1)[1]
    assert unfinished.count("Fan (exposed transfer)") == 3
    assert monitor.report_text(report, [pilot], vars(plot), "test") == report


def test_fan_variants_have_matched_five_seed_cohorts() -> None:
    """Development variants pool pilots and confirmation, never paper data."""
    for domain in plot.FAN_VARIANTS:
        directories = dict(plot.DOMAINS)[domain]
        for arm, paths in directories.items():
            if arm == "from_assets":
                assert len(paths) == (2 if domain == "Fan (ramp transfer)" else
                                      0)
                continue
            launched = {"MB", "MF"}
            if domain in plot.FAN_VARIANTS:
                launched.update({
                    "oracle_dynamics", "mf_scene_package", "standalone",
                    "no_fitting", "no_uncertainty"
                })
                expected = 5 if arm in launched else 0
                assert len(paths) == expected
            for seed, path in enumerate(paths):
                assert path.endswith(f"/seed{seed}")
                if domain == "Fan (ramp transfer)":
                    expected_round = ("fan_prompt_r2"
                                      if arm == "oracle_dynamics" else
                                      "_opus_ramp_skill_repair_r1")
                    assert expected_round in path
                elif arm in ("MB", "MF"):
                    assert ("pilot_r1"
                            if seed < 2 else "confirmation_r1") in path
                else:
                    assert f"{arm}_opus_inertial_r1" in path
        row = dict(domain=domain,
                   arm="MB",
                   seed=0,
                   won=2,
                   levels=2,
                   steps=100,
                   resets=0,
                   source="development")
        assert bool(plot.paper_records(
            [row])) == (domain == "Fan (ramp transfer)")
        report = monitor.report_text(monitor.REPORT.read_text(), [row],
                                     vars(plot), "test")
        status = next(line for line in report.splitlines()
                      if line.startswith(f"- {domain}:"))
        assert "EMPIRIC 1/1 solved, 1/5 finished" in status
        assert monitor.report_text(report, [row], vars(plot), "test") == report


def test_inertial_baseline_completion_stays_out_of_paper(
        tmp_path: Path, monkeypatch: Any) -> None:
    """Publish finished development seeds without changing paper counts."""
    monkeypatch.setattr(plot, "LOGS", str(tmp_path))
    domain = "Fan (inertial transfer)"
    directory = dict(plot.DOMAINS)[domain]["oracle_dynamics"][0]
    run = tmp_path / directory / "run_20260921"
    run.mkdir(parents=True)
    (run / "scorecard.json").write_text(json.dumps({
        "end_reason":
        "all_levels_won",
        "levels": [{
            "won": True
        }, {
            "won": True
        }],
        "totals": {
            "total_steps": 100,
            "total_resets": 0
        },
    }),
                                        encoding="utf-8")
    rows = plot.records()
    assert len(rows) == 1 and rows[0]["domain"] == domain
    assert not plot.paper_records(rows)
    report = monitor.report_text(monitor.REPORT.read_text(), rows, vars(plot),
                                 "test")
    assert "Oracle dynamics: 0/25 seeds finished" in report
    assert f"{domain}: Oracle dynamics 1/1 solved, 1/5 finished" in report
    row = "| Fan (inertial transfer) | 1. Oracle dynamics | 1/1 (100%)"
    assert row in report
