"""Tests for synthesis-prompt formatter helpers.

These are pure-Python staticmethods (or `self`-less methods) on
``AgentSimLearningApproach`` and ``AgentSimPredicateInventionApproach``
that render parts of the agent's first synthesis message. They were
added so the agent (a) knows the provenance of each interaction
trajectory and (b) gets reminded about prior-cycle files in the
sandbox.
"""
# pylint: disable=protected-access
from __future__ import annotations

import numpy as np
import pytest

# Bootstrap circular imports before pulling from predicators.approaches.
import predicators.utils  # noqa: F401
from predicators.structs import Action, LowLevelTrajectory, State, Type


@pytest.fixture(name="approach_cls")
def _approach_cls():
    """Late-import the class so test collection is cheap."""
    from predicators.approaches.agent_sim_learning_approach import \
        AgentSimLearningApproach
    return AgentSimLearningApproach


def _mk_traj(is_demo, task_idx, sim_v=None, preds_v=None):
    """Build a 1-action trajectory with the given provenance tags."""
    cup_type = Type("cup_type", ["f"])
    cup = cup_type("cup")
    states = [State({cup: [0.0]}), State({cup: [1.0]})]
    actions = [Action(np.array([0.5]))]
    return LowLevelTrajectory(
        states,
        actions,
        _is_demo=is_demo,
        _train_task_idx=task_idx,
        _source_simulator_version=sim_v,
        _source_predicates_version=preds_v,
    )


# ── _format_trajectory_listing ──────────────────────────────────────


def test_trajectory_listing_empty(approach_cls):
    """Empty trajectory list short-circuits to an empty string."""
    assert approach_cls._format_trajectory_listing([]) == ""


def test_trajectory_listing_demo_has_no_provenance_tail(approach_cls):
    """Demo trajectories never carry provenance — even if the tags are
    set, the listing should still render them as plain demos for
    consistency with the offline-data semantics."""
    trajs = [_mk_traj(is_demo=True, task_idx=0)]
    out = approach_cls._format_trajectory_listing(trajs)
    assert "[0] demo, task 0" in out
    assert "generated using" not in out


def test_trajectory_listing_interaction_with_provenance(approach_cls):
    """Interaction trajectories with provenance show the sim/preds tags."""
    trajs = [
        _mk_traj(is_demo=False,
                 task_idx=2,
                 sim_v="cycle_001_vers_004",
                 preds_v="cycle_001_vers_003"),
    ]
    out = approach_cls._format_trajectory_listing(trajs)
    assert "[0] interaction, task 2" in out
    assert "sim cycle_001_vers_004" in out
    assert "predicates cycle_001_vers_003" in out


def test_trajectory_listing_partial_provenance(approach_cls):
    """A trajectory with only ``source_simulator_version`` set should
    list only the sim tag — no stray ``, `` from a missing pair."""
    trajs = [_mk_traj(is_demo=False, task_idx=1, sim_v="cycle_001_vers_007")]
    out = approach_cls._format_trajectory_listing(trajs)
    line = [l for l in out.splitlines() if l.startswith("  [0]")][0]
    assert "sim cycle_001_vers_007" in line
    assert "predicates" not in line


# ── _format_prior_state_block ────────────────────────────────────────


def test_prior_state_block_empty_when_no_files(approach_cls, tmp_path):
    """Neither simulator.py nor predicates.py exists → empty block."""
    out = approach_cls._format_prior_state_block(None, str(tmp_path))
    assert out == ""


def test_prior_state_block_simulator_only(approach_cls, tmp_path):
    """Only simulator.py exists → block mentions it and not predicates.py."""
    (tmp_path / "simulator.py").write_text("# sim")
    out = approach_cls._format_prior_state_block(None, str(tmp_path))
    assert "`./simulator.py`" in out
    assert "`./predicates.py`" not in out
    # Always points at the versioned-snapshot dirs for cross-reference.
    assert "./simulator_versions/" in out


def test_prior_state_block_both_files(approach_cls, tmp_path):
    """Both files exist → block lists them joined with ' and '."""
    (tmp_path / "simulator.py").write_text("# sim")
    (tmp_path / "predicates.py").write_text("LEARNED_PREDICATES = []")
    out = approach_cls._format_prior_state_block(None, str(tmp_path))
    assert "`./simulator.py` and `./predicates.py`" in out
    # Soft language so the agent isn't forbidden from a fresh rewrite.
    assert "fresh rewrite is fine" in out


# ── _format_goal_nl_block (predicate-invention subclass) ────────────


def test_goal_nl_block_empty_when_no_tasks_have_goal_nl():
    """No ``goal_nl`` populated → empty block (no header)."""
    from predicators.approaches.agent_sim_predicate_invention_approach import \
        AgentSimPredicateInventionApproach
    fake_self = type(
        "_FakeApproach",
        (),
        {
            "_train_tasks": [type("_T", (), {"goal_nl": None})()] * 2,
        },
    )()
    out = AgentSimPredicateInventionApproach._format_goal_nl_block(fake_self)
    assert out == ""


def test_goal_nl_block_dedups_identical_goals():
    """Same NL goal across tasks shows up once, with the single-task header."""
    from predicators.approaches.agent_sim_predicate_invention_approach import \
        AgentSimPredicateInventionApproach
    fake_task = type("_T", (), {"goal_nl": "boil the water"})
    fake_self = type(
        "_FakeApproach",
        (),
        {
            "_train_tasks": [fake_task() for _ in range(3)],
        },
    )()
    out = AgentSimPredicateInventionApproach._format_goal_nl_block(fake_self)
    assert out.startswith("Goal (natural language): boil the water")
    # Trailing blank line separates from the next paragraph in the prompt.
    assert out.endswith("\n\n")


def test_goal_nl_block_multiple_distinct_goals():
    """Distinct goals across tasks render as a bulleted list."""
    from predicators.approaches.agent_sim_predicate_invention_approach import \
        AgentSimPredicateInventionApproach
    tasks = [
        type("_T1", (), {"goal_nl": "boil the water"})(),
        type("_T2", (), {"goal_nl": "stack the cups"})(),
    ]
    fake_self = type("_FakeApproach", (), {"_train_tasks": tasks})()
    out = AgentSimPredicateInventionApproach._format_goal_nl_block(fake_self)
    assert "Goals across train tasks (natural language):" in out
    assert "  - boil the water" in out
    assert "  - stack the cups" in out
