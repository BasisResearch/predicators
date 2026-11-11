"""Tests for versioned-snapshot helpers in ``predicators.agent_sdk.tools``.

Covers the plumbing of the file-driven simulator / predicates synthesis
pipeline:

* ``finalize_versioned_snapshot`` — the "take one more snapshot if the
  live file changed" helper run after the agent session closes.
* ``make_write_snapshot_hook`` — the PostToolUse hook that snapshots
  ``simulator.py`` / ``predicates.py`` after every Write/Edit/MultiEdit.
* ``_ArtifactSnapshotter`` - the snapshot each synthesis tool call
  (``sim.fit``, ``sim.validate``, ...) takes of the file it loads.

All are pure-Python and side-effect on the filesystem only; no agent
SDK calls are made.
"""
# pylint: disable=protected-access,unused-import
import asyncio
from types import SimpleNamespace

# Bootstrap circular imports before pulling from predicators.agent_sdk.
import predicators.utils  # noqa: F401 — required for import side effects
from predicators.agent_sdk.tools import snapshots as snapshots_mod
from predicators.agent_sdk.tools.snapshots import _ArtifactSnapshotter, \
    _SnapshotTarget, finalize_versioned_snapshot, make_write_snapshot_hook, \
    restored_version

# ── finalize_versioned_snapshot ──────────────────────────────────────


def test_finalize_versioned_snapshot_missing_live_file(tmp_path):
    """Returns ``None`` and writes nothing when the live file is absent."""
    versions = tmp_path / "simulator_versions"
    versions.mkdir()
    tag = finalize_versioned_snapshot(
        str(tmp_path / "simulator.py"),
        str(versions),
        cycle_idx=1,
        artifact_name="simulator",
    )
    assert tag is None
    assert not list(versions.iterdir())


def test_finalize_versioned_snapshot_creates_first_snapshot(tmp_path):
    """First call writes ``cycle_001_vers_001`` and returns its tag."""
    live = tmp_path / "simulator.py"
    versions = tmp_path / "simulator_versions"
    live.write_text("# v1\n")
    tag = finalize_versioned_snapshot(str(live),
                                      str(versions),
                                      cycle_idx=1,
                                      artifact_name="simulator")
    assert tag == "cycle_001_vers_001"
    snapshots = sorted(p.name for p in versions.iterdir())
    assert snapshots == ["cycle_001_vers_001_simulator.py"]
    assert (versions /
            "cycle_001_vers_001_simulator.py").read_text() == "# v1\n"


def test_finalize_versioned_snapshot_offline_label(tmp_path):
    """A negative cycle index (the offline pass) renders as ``offline``,
    keeping it distinct from cycle 0's online snapshots."""
    live = tmp_path / "simulator.py"
    versions = tmp_path / "simulator_versions"
    live.write_text("# offline\n")
    tag = finalize_versioned_snapshot(str(live),
                                      str(versions),
                                      cycle_idx=-1,
                                      artifact_name="simulator")
    assert tag == "cycle_offline_vers_001"
    assert sorted(p.name for p in versions.iterdir()) == [
        "cycle_offline_vers_001_simulator.py"
    ]
    # Cycle 0's online pass gets its own prefix and version counter.
    live.write_text("# cycle 0\n")
    tag = finalize_versioned_snapshot(str(live),
                                      str(versions),
                                      cycle_idx=0,
                                      artifact_name="simulator")
    assert tag == "cycle_000_vers_001"


def test_finalize_versioned_snapshot_dedup_on_unchanged_file(tmp_path):
    """A no-op finalize on unchanged content reuses the prior tag."""
    live = tmp_path / "predicates.py"
    versions = tmp_path / "predicates_versions"
    live.write_text("LEARNED_PREDICATES = []\n")
    first = finalize_versioned_snapshot(str(live),
                                        str(versions),
                                        cycle_idx=2,
                                        artifact_name="predicates")
    second = finalize_versioned_snapshot(str(live),
                                         str(versions),
                                         cycle_idx=2,
                                         artifact_name="predicates")
    assert first == second == "cycle_002_vers_001"
    assert len(list(versions.iterdir())) == 1


def test_finalize_versioned_snapshot_bumps_on_change(tmp_path):
    """Changed content increments ``vers_YYY`` within the same cycle."""
    live = tmp_path / "simulator.py"
    versions = tmp_path / "simulator_versions"
    live.write_text("# v1\n")
    finalize_versioned_snapshot(str(live),
                                str(versions),
                                cycle_idx=1,
                                artifact_name="simulator")
    live.write_text("# v2\n")
    tag = finalize_versioned_snapshot(str(live),
                                      str(versions),
                                      cycle_idx=1,
                                      artifact_name="simulator")
    assert tag == "cycle_001_vers_002"
    names = sorted(p.name for p in versions.iterdir())
    assert names == [
        "cycle_001_vers_001_simulator.py",
        "cycle_001_vers_002_simulator.py",
    ]


def test_finalize_versioned_snapshot_new_cycle_restarts_vers_yyy(tmp_path):
    """A new cycle starts at ``vers_001`` even when other cycles populated the
    same directory."""
    live = tmp_path / "simulator.py"
    versions = tmp_path / "simulator_versions"
    live.write_text("# v1\n")
    finalize_versioned_snapshot(str(live),
                                str(versions),
                                cycle_idx=1,
                                artifact_name="simulator")
    # Mutate and finalize as cycle 2; same content as cycle 1 still gets
    # a fresh cycle_002 entry because cycle 2 has no prior snapshots.
    live.write_text("# v2\n")
    tag = finalize_versioned_snapshot(str(live),
                                      str(versions),
                                      cycle_idx=2,
                                      artifact_name="simulator")
    assert tag == "cycle_002_vers_001"
    names = sorted(p.name for p in versions.iterdir())
    assert names == [
        "cycle_001_vers_001_simulator.py",
        "cycle_002_vers_001_simulator.py",
    ]


def test_finalize_versioned_snapshot_other_artifact_ignored(tmp_path):
    """Existing files for a *different* ``artifact_name`` don't influence the
    version count."""
    live = tmp_path / "predicates.py"
    versions = tmp_path / "shared_versions"
    versions.mkdir()
    # Sibling simulator snapshot for the same cycle — must not affect
    # the predicates counter.
    (versions / "cycle_001_vers_007_simulator.py").write_text("sim")
    live.write_text("preds")
    tag = finalize_versioned_snapshot(str(live),
                                      str(versions),
                                      cycle_idx=1,
                                      artifact_name="predicates")
    assert tag == "cycle_001_vers_001"
    assert (versions / "cycle_001_vers_001_predicates.py").exists()


# ── make_write_snapshot_hook ────────────────────────────────────────


def _run_hook(hook, tool_name, file_path):
    """Synchronously invoke the async hook with the input the SDK passes: the
    CLI's JSON parsed into a dict (``PostToolUseHookInput``)."""
    hook_input = {
        "hook_event_name": "PostToolUse",
        "tool_name": tool_name,
        "tool_input": {
            "file_path": file_path
        },
    }
    return asyncio.run(hook(hook_input, "toolu_1", {"signal": None}))


def _make_hook(tmp_path, cycle_idx=1):
    sandbox = tmp_path
    sim = sandbox / "simulator.py"
    preds = sandbox / "predicates.py"
    sim_vd = sandbox / "simulator_versions"
    preds_vd = sandbox / "predicates_versions"
    targets = [
        _SnapshotTarget(str(sim), str(sim_vd), "simulator", lambda: cycle_idx),
        _SnapshotTarget(str(preds), str(preds_vd), "predicates",
                        lambda: cycle_idx),
    ]
    return make_write_snapshot_hook(targets, sandbox_dir=str(sandbox)), {
        "sim": sim,
        "preds": preds,
        "sim_vd": sim_vd,
        "preds_vd": preds_vd,
    }


def test_write_hook_snapshots_simulator_on_write(tmp_path):
    """Write tool with the simulator path produces a new snapshot."""
    hook, paths = _make_hook(tmp_path)
    paths["sim"].write_text("# rules\n")
    _run_hook(hook, "Write", "./simulator.py")
    snapshots = sorted(p.name for p in paths["sim_vd"].iterdir())
    assert snapshots == ["cycle_001_vers_001_simulator.py"]


def test_write_hook_ignores_unrelated_tools(tmp_path):
    """Read / Bash / Grep firing on the simulator path don't snapshot."""
    hook, paths = _make_hook(tmp_path)
    paths["sim"].write_text("# rules\n")
    for tool in ("Read", "Bash", "Grep", "Glob", "NotebookEdit"):
        _run_hook(hook, tool, "./simulator.py")
    assert not paths["sim_vd"].exists() or not list(paths["sim_vd"].iterdir())


def test_write_hook_dedup_on_no_op_edit(tmp_path):
    """Edit producing identical content does not append a new snapshot."""
    hook, paths = _make_hook(tmp_path)
    paths["sim"].write_text("body\n")
    _run_hook(hook, "Write", "./simulator.py")
    _run_hook(hook, "Edit", "./simulator.py")
    _run_hook(hook, "MultiEdit", "./simulator.py")
    snapshots = list(paths["sim_vd"].iterdir())
    assert len(snapshots) == 1


def test_write_hook_resolves_absolute_and_relative_paths(tmp_path):
    """A relative ``./predicates.py`` and an absolute path resolve to the same
    target — both trigger snapshots, but dedup means only one file."""
    hook, paths = _make_hook(tmp_path)
    paths["preds"].write_text("LEARNED_PREDICATES = []\n")
    _run_hook(hook, "Write", "./predicates.py")
    _run_hook(hook, "Edit", str(paths["preds"]))  # same content, absolute
    snapshots = list(paths["preds_vd"].iterdir())
    assert len(snapshots) == 1
    assert snapshots[0].name == "cycle_001_vers_001_predicates.py"


def test_write_hook_ignores_files_outside_target_list(tmp_path):
    """A write to some random file in the sandbox does not snapshot."""
    hook, paths = _make_hook(tmp_path)
    other = tmp_path / "scratch.py"
    other.write_text("print('hi')\n")
    _run_hook(hook, "Write", "./scratch.py")
    assert not paths["sim_vd"].exists() or not list(paths["sim_vd"].iterdir())
    assert (not paths["preds_vd"].exists()
            or not list(paths["preds_vd"].iterdir()))


def test_write_hook_swallows_exceptions(tmp_path):
    """A snapshot failure must not propagate — hooks failing should never break
    the agent's edit loop."""
    hook, _paths = _make_hook(tmp_path)
    # Missing file_path is one quiet failure path; a non-string is another,
    # and so is an input that is not the SDK's dict.
    asyncio.run(hook({"tool_name": "Write", "tool_input": {}}, None, None))
    asyncio.run(hook({"tool_name": "Edit", "tool_input": None}, None, None))
    asyncio.run(hook(SimpleNamespace(tool_name="Write"), None, None))
    # Inputs that look valid but the snapshot helper trips on (unwritable
    # versions dir) should also not raise — point a target at a path that
    # cannot be created, fire the hook, expect no exception.
    bad_target = _SnapshotTarget(
        live_file=str(tmp_path / "simulator.py"),
        versions_dir="/dev/null/cannot/create",
        artifact_name="simulator",
        cycle_index_provider=lambda: 1,
    )
    bad_hook = make_write_snapshot_hook([bad_target],
                                        sandbox_dir=str(tmp_path))
    (tmp_path / "simulator.py").write_text("body")
    _run_hook(bad_hook, "Write", "./simulator.py")


def test_write_hook_uses_cycle_provider_at_call_time(tmp_path):
    """The cycle index is read each time the hook fires, not captured up front,
    so consecutive cycles land in different filenames."""
    sandbox = tmp_path
    sim = sandbox / "simulator.py"
    sim_vd = sandbox / "simulator_versions"
    cycle = [1]
    target = _SnapshotTarget(str(sim), str(sim_vd), "simulator",
                             lambda: cycle[0])
    hook = make_write_snapshot_hook([target], sandbox_dir=str(sandbox))

    sim.write_text("# c1\n")
    _run_hook(hook, "Write", "./simulator.py")
    cycle[0] = 2
    sim.write_text("# c2\n")
    _run_hook(hook, "Edit", "./simulator.py")

    snapshots = sorted(p.name for p in sim_vd.iterdir())
    assert snapshots == [
        "cycle_001_vers_001_simulator.py",
        "cycle_002_vers_001_simulator.py",
    ]


# ── restored_version ─────────────────────────────────────────────────


def test_restored_version_keeps_a_matching_checkpoint_tag(tmp_path):
    """A checkpoint tag whose snapshot holds the restored file is kept, under
    any current cycle, and nothing new is written."""
    versions = tmp_path / "simulator_versions"
    versions.mkdir()
    (versions / "cycle_003_vers_002_simulator.py").write_text("SIM")
    live = tmp_path / "simulator.py"
    live.write_text("SIM")
    tag = restored_version(str(live), str(versions), 0, "simulator",
                           "cycle_003_vers_002")
    assert tag == "cycle_003_vers_002"
    assert sorted(p.name for p in versions.iterdir()) == [
        "cycle_003_vers_002_simulator.py"
    ]


def test_restored_version_names_a_file_the_checkpoint_did_not_hold(tmp_path):
    """A checkpoint taken before the file existed, or before its last edit,
    gets the file snapshotted under the current cycle."""
    versions = tmp_path / "simulator_versions"
    live = tmp_path / "simulator.py"
    live.write_text("SIM")
    assert restored_version(str(live), str(versions), 2, "simulator",
                            None) == "cycle_002_vers_001"
    live.write_text("EDITED")
    assert restored_version(str(live), str(versions), 2, "simulator",
                            "cycle_002_vers_001") == "cycle_002_vers_002"
    assert restored_version(str(tmp_path / "missing.py"), str(versions), 2,
                            "simulator", "cycle_002_vers_001") is None


# ── _ArtifactSnapshotter ─────────────────────────────────────────────


def _snapshotter(tmp_path, cycle):
    """A simulator snapshotter whose cycle is read from ``cycle[0]``."""
    return _ArtifactSnapshotter(str(tmp_path / "simulator.py"),
                                str(tmp_path / "simulator_versions"),
                                "simulator", lambda: cycle[0])


def test_snapshotter_numbers_after_the_versions_on_disk(tmp_path):
    """A new snapshotter numbers its snapshots after the cycle's highest
    version on disk.

    A restarted run re-issues the round it was cut off in, so the cycle
    already holds the killed process's snapshots, and none of them is
    written over (run_20261003_073316 of the Domino fixes_r1 round lost
    its first round's ``cycle_000_vers_001`` to the relaunch).
    """
    versions = tmp_path / "simulator_versions"
    versions.mkdir()
    (versions / "cycle_000_vers_004_simulator.py").write_text("OLD")
    (versions / "cycle_001_vers_001_simulator.py").write_text("A")
    (versions / "cycle_001_vers_002_simulator.py").write_text("B")
    (tmp_path / "simulator.py").write_text("C")
    snapshotter = _snapshotter(tmp_path, [1])
    assert snapshotter.snapshot() == (b"C", "cycle_001_vers_003", None)
    # Unchanged content keeps its tag and writes nothing.
    assert snapshotter.snapshot() == (b"C", "cycle_001_vers_003", None)
    assert {p.name: p.read_text()
            for p in versions.iterdir()} == {
                "cycle_000_vers_004_simulator.py": "OLD",
                "cycle_001_vers_001_simulator.py": "A",
                "cycle_001_vers_002_simulator.py": "B",
                "cycle_001_vers_003_simulator.py": "C",
            }


def test_snapshotter_and_write_hook_share_the_numbering(tmp_path):
    """Snapshots the write hook took between two tool calls keep their
    numbers: the next tool call tags the file with the hook's version when
    the content matches, and numbers a new version after it otherwise."""
    hook, paths = _make_hook(tmp_path, cycle_idx=1)
    snapshotter = _snapshotter(tmp_path, [1])
    paths["sim"].write_text("A")
    assert snapshotter.snapshot()[1] == "cycle_001_vers_001"
    paths["sim"].write_text("B")
    _run_hook(hook, "Edit", "./simulator.py")
    paths["sim"].write_text("C")
    _run_hook(hook, "Edit", "./simulator.py")
    assert snapshotter.snapshot()[1] == "cycle_001_vers_003"
    paths["sim"].write_text("D")
    assert snapshotter.snapshot()[1] == "cycle_001_vers_004"
    assert [p.read_text() for p in sorted(paths["sim_vd"].iterdir())
            ] == ["A", "B", "C", "D"]


def test_snapshotter_tag_names_a_snapshot_of_the_current_cycle(tmp_path):
    """After the cycle advances, an unchanged file is snapshotted under the new
    cycle, so the returned tag always names a file that holds the content."""
    cycle = [0]
    snapshotter = _snapshotter(tmp_path, cycle)
    (tmp_path / "simulator.py").write_text("A")
    assert snapshotter.snapshot()[1] == "cycle_000_vers_001"
    cycle[0] = 1
    assert snapshotter.snapshot()[1] == "cycle_001_vers_001"
    assert (tmp_path / "simulator_versions" /
            "cycle_001_vers_001_simulator.py").read_text() == "A"


def test_snapshot_never_writes_over_a_version(tmp_path, monkeypatch):
    """A version another writer creates between the scan and the write is kept,
    and the snapshot takes the next number."""
    versions = tmp_path / "simulator_versions"
    live = tmp_path / "simulator.py"
    live.write_text("MINE")
    scan = snapshots_mod._latest_snapshot
    scans = []

    def racing_scan(*args):
        scans.append(args)
        if len(scans) == 1:
            # The first scan finds the cycle empty; the other writer's
            # file lands right after it.
            versions.mkdir(exist_ok=True)
            (versions / "cycle_001_vers_001_simulator.py").write_text("THEIRS")
            return 0, None
        return scan(*args)

    monkeypatch.setattr(snapshots_mod, "_latest_snapshot", racing_scan)
    assert finalize_versioned_snapshot(str(live), str(versions), 1,
                                       "simulator") == "cycle_001_vers_002"
    assert len(scans) == 2
    assert {p.name: p.read_text()
            for p in versions.iterdir()} == {
                "cycle_001_vers_001_simulator.py": "THEIRS",
                "cycle_001_vers_002_simulator.py": "MINE",
            }
