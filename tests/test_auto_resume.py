"""Tests for --auto_resume checkpoint discovery (main.discover_resume_cycles).

The discovery function is pure: given the config's checkpoint load path,
it reports whether any checkpoint exists and the highest completed
online-learning cycle, from which _maybe_auto_resume derives
load_approach / restart_learning / skip_until_cycle.
"""
import os

from predicators.main import discover_resume_cycles


def _touch(path):
    with open(path, "w", encoding="utf-8") as f:
        f.write("x")


def test_no_checkpoints(tmp_path):
    """An empty approach dir means a fresh start."""
    load_path = str(tmp_path / "cfg.saved")
    assert discover_resume_cycles(load_path) == (False, None)


def test_only_post_offline_checkpoint(tmp_path):
    """Only the _None checkpoint: resume from cycle 0."""
    load_path = str(tmp_path / "cfg.saved")
    _touch(load_path + "_None.AgentSimPredInv")
    assert discover_resume_cycles(load_path) == (True, None)


def test_max_cycle_wins(tmp_path):
    """The highest completed cycle across suffixes is reported."""
    load_path = str(tmp_path / "cfg.saved")
    _touch(load_path + "_None.AgentSimPredInv")
    for cycle in (0, 2, 1):
        _touch(load_path + f"_{cycle}.AgentSimPredInv")
    assert discover_resume_cycles(load_path) == (True, 2)


def test_foreign_and_malformed_files_ignored(tmp_path):
    """Files of other configs or malformed names never count."""
    load_path = str(tmp_path / "cfg.saved")
    # A different config's checkpoints share the dir but not the prefix.
    _touch(str(tmp_path / "othercfg.saved_5.AgentSimPredInv"))
    # Malformed cycle token.
    _touch(load_path + "_abc.AgentSimPredInv")
    assert discover_resume_cycles(load_path) == (False, None)
    _touch(load_path + "_3.AgentPlanner")
    assert discover_resume_cycles(load_path) == (True, 3)
