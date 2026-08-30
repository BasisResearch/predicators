"""Tests for the info.log parser and the resume-lineage rows of
scripts/log_viewer.py."""

from pathlib import Path
from typing import Any, Dict

from scripts.log_viewer import _explore_results, _misc_chip, _parse_info_log, \
    chain_summary, resume_chains

_SAVE = ("INFO: Saved local sandbox query/response to logs/x/sandbox/"
         "session_logs/{name}.md")
_RESUME = ("INFO: --auto_resume: checkpoint(s) found at saved_approaches/x"
           ".saved_* (last completed cycle: {last}); resuming with "
           "load_approach + restart_learning, skip_until_cycle={skip}.")
_CATCH_UP = ("INFO: Resumed past cycle {c} whose test never ran; testing it "
             "now before continuing.")


def _write_log(tmp_path: Path, lines: list) -> str:
    path = tmp_path / "info.log"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def test_resumed_catch_up_test_belongs_to_its_cycle(tmp_path: Path) -> None:
    """A resumed run re-tests its last completed cycle before its first banner;
    those sessions are that cycle's, not the offline phase's."""
    path = _write_log(tmp_path, [
        _RESUME.format(last=0, skip=1),
        _CATCH_UP.format(c=0),
        _SAVE.format(name="001_test_task0_20260827_171624"),
        "INFO: Tasks solved: 1 / 1",
        "ONLINE LEARNING CYCLE 1",
        _SAVE.format(name="002_explore_20260827_181820"),
        "INFO: Interaction episode on train task 0: reward=1.00, "
        "terminated=True, accepted=True",
        _SAVE.format(name="003_learn_20260827_205835"),
        _SAVE.format(name="004_test_task0_20260827_232442"),
        "INFO: Tasks solved: 0 / 1",
    ])
    parsed = _parse_info_log(path)
    assert parsed["resume_cycle"] == 1
    assert parsed["session_cycles"] == {1: 0, 2: 1, 3: 1, 4: 1}
    assert parsed["round_cycles"] == [0, 1]
    assert parsed["explore"][2]["cycle"] == 1
    assert parsed["last_cycle"] == 1


def test_fresh_run_pre_banner_sessions_stay_offline(tmp_path: Path) -> None:
    """Without the catch-up line, sessions saved before the first banner are
    the offline learn / pre-loop test (cycle None)."""
    path = _write_log(tmp_path, [
        _SAVE.format(name="001_learn_20260827_121109"),
        _SAVE.format(name="002_test_task0_20260827_130000"),
        "INFO: Tasks solved: 0 / 1",
        "ONLINE LEARNING CYCLE 0",
        _SAVE.format(name="003_explore_20260827_140000"),
    ])
    parsed = _parse_info_log(path)
    assert parsed["resume_cycle"] is None
    assert parsed["session_cycles"] == {1: None, 2: None, 3: 0}
    assert parsed["round_cycles"] == [None]
    assert parsed["last_cycle"] == 0


def test_catch_up_line_never_overrides_a_real_banner(tmp_path: Path) -> None:
    """The catch-up line only stands in for a banner while none has been seen;
    a later occurrence cannot rewind the cycle."""
    path = _write_log(tmp_path, [
        "ONLINE LEARNING CYCLE 2",
        _CATCH_UP.format(c=0),
        _SAVE.format(name="001_test_task0_20260827_171624"),
        "INFO: Tasks solved: 1 / 1",
    ])
    parsed = _parse_info_log(path)
    assert parsed["session_cycles"] == {1: 2}
    assert parsed["round_cycles"] == [2]


def _run(name: str, seed: str = "seed0") -> dict:
    return {
        "exp": "fam/exp",
        "seed": seed,
        "name": name,
        "rel": f"fam/exp/{seed}/{name}",
        "mtime": 0.0,
        "activity": 0.0,
    }


def test_resume_chains_join_resumed_runs_to_their_predecessor() -> None:
    """A resumed run joins the chain of its chronological predecessor of the
    same seed; fresh runs and other seeds open their own chains."""
    a, b, c = _run("run_20260827_121109"), _run("run_20260827_150302"), _run(
        "run_20260827_171610")
    fresh = _run("run_20260828_090000")
    other = _run("run_20260827_121111", seed="seed1")
    summaries: Dict[str, Dict[str, Any]] = {
        a["rel"]: {
            "resume_cycle": None
        },
        b["rel"]: {
            "resume_cycle": 0
        },
        c["rel"]: {
            "resume_cycle": 1
        },
        fresh["rel"]: {
            "resume_cycle": None
        },
        other["rel"]: {
            "resume_cycle": 2
        },  # nothing before it: own chain
    }
    chains = resume_chains([c, other, a, fresh, b], summaries)
    assert [[r["name"] for r in ch] for ch in chains] == [
        [a["name"], b["name"], c["name"]],
        [fresh["name"]],
        [other["name"]],
    ]


def test_chain_summary_merges_episodes_by_cycle_and_tests_by_newest() -> None:
    """Episodes re-bucket by true cycle across runs; the newest run wins a
    cycle's test result; costs sum."""
    a, b = _run("run_20260827_121109"), _run("run_20260827_150302")
    summaries: Dict[str, Dict[str, Any]] = {
        a["rel"]: {
            "episodes": [
                {
                    "num": 1,
                    "kind": "learn",
                    "task": None,
                    "round": 0,
                    "cycle_tag": "cycleNone"
                },
                {
                    "num": 2,
                    "kind": "explore",
                    "task": None,
                    "round": 1,
                    "cycle_tag": "cycle0",
                    "env_reward": 0.0,
                    "env_accepted": False
                },
                {
                    "num": 3,
                    "kind": "test",
                    "task": 0,
                    "round": 1,
                    "cycle_tag": "cycle0"
                },  # killed mid-test: no verdict
            ],
            "test_results": [],
            "test_cycles": [0],
            "total_cost":
            1.5,
            "resume_cycle":
            None,
            "done":
            False,
        },
        b["rel"]: {
            "episodes": [
                {
                    "num": 1,
                    "kind": "test",
                    "task": 0,
                    "round": 0,
                    "cycle_tag": "cycle0",
                    "env_solved": True
                },
                {
                    "num": 2,
                    "kind": "explore",
                    "task": None,
                    "round": 1,
                    "cycle_tag": "cycle1",
                    "env_reward": 1.0,
                    "env_accepted": True
                },
            ],
            "test_results": [(1, 1, 1.0)],
            "test_cycles": [0],
            "total_cost":
            2.0,
            "resume_cycle":
            1,
            "done":
            False,
        },
    }
    merged = chain_summary([a, b], summaries)
    rows = [(ep["run_id"], ep["num"], ep["round"], ep["superseded"])
            for ep in merged["episodes"]]
    # off -> row 0, c0 -> row 1 (a's killed test and b's re-test share
    # it), c1 -> row 2.
    assert rows == [
        ("20260827_121109", 1, 0, True),
        ("20260827_121109", 2, 1, True),
        ("20260827_121109", 3, 1, True),
        ("20260827_150302", 1, 1, False),
        ("20260827_150302", 2, 2, False),
    ]
    assert merged["test_results"] == [(1, 1, 1.0)]
    assert merged["test_cycles"] == [0]
    assert merged["explore_results"] == [(0, 0, 1, 0.0), (1, 1, 1, 1.0)]
    assert merged["total_cost"] == 3.5
    assert merged["resume_cycle"] == 1


def test_chain_summary_single_run_keeps_rows() -> None:
    """A one-run chain keeps the run's own grid rows."""
    a = _run("run_20260827_121109")
    summaries: Dict[str, Dict[str, Any]] = {
        a["rel"]: {
            "episodes": [{
                "num": 1,
                "kind": "learn",
                "task": None,
                "round": 4
            }],
            "test_results": [],
            "test_cycles": [],
            "total_cost": 0.0,
        }
    }
    merged = chain_summary([a], summaries)
    assert merged["episodes"][0]["round"] == 4
    assert merged["episodes"][0]["superseded"] is False


_CERTIFIED = ("INFO: agent_bilevel explorer: the agent's tool-validated plan "
              "passed the belief's capture gate (validation: 8/8 rollouts "
              "ok); executing it verbatim as this episode's solve attempt "
              "(mental model solved the goal).")
_VERDICT = ("INFO: Interaction episode on train task 0: reward={r}, "
            "terminated={t}, accepted={a}")


def test_certified_session_is_marked(tmp_path: Path) -> None:
    """The capture-gate line marks the newest explore session as certified; the
    cycle's second request is an ordinary session with its own transcript and
    verdict."""
    path = _write_log(tmp_path, [
        "ONLINE LEARNING CYCLE 2",
        _SAVE.format(name="018_explore_20260830_090000"),
        _CERTIFIED,
        _SAVE.format(name="019_explore_20260830_093000"),
        _VERDICT.format(r="1.00", t="True", a="True"),
        _VERDICT.format(r="0.00", t="False", a="False"),
    ])
    ex = _parse_info_log(path)["explore"]
    assert ex[18]["certified"] is True and ex[18]["accepted"] is True
    assert "certified" not in ex[19] and ex[19]["accepted"] is False


def test_certified_chip_mark() -> None:
    """A certified session's chip carries the diamond and its tooltip; the
    tally is unchanged."""
    ep = {
        "num": 18,
        "kind": "explore",
        "env_reward": 1.0,
        "env_terminated": True,
        "env_accepted": True,
        "env_msg": "",
        "certified": True,
    }
    html = _misc_chip(ep)
    assert "018 explore ✓ 1.00 ◆" in html
    assert "belief-certified" in html
    assert _explore_results([dict(ep, cycle_tag="cycle2")]) == [(2, 1, 1, 1.0)]
