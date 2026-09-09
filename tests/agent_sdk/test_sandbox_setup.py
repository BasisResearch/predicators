"""Tests for the sandbox's python guard, data export and test-phase rollback
(predicators.agent_sdk.sandbox_setup)."""
import os
import pickle
import shutil
import subprocess
import sys
from typing import List

import numpy as np
import pytest

from predicators.agent_sdk.sandbox_setup import export_trajectories, \
    find_repo_root, git_commit_all, pyguard_env, rollback_sandbox, \
    snapshot_sandbox, trajectories_path, write_pyguard
from predicators.structs import LowLevelTrajectory


def _run(code: str, sandbox: str) -> subprocess.CompletedProcess:
    env = {**os.environ, **pyguard_env(sandbox)}
    return subprocess.run([sys.executable, "-c", code],
                          cwd=sandbox,
                          env=env,
                          capture_output=True,
                          text=True,
                          check=False,
                          timeout=120)


def test_pyguard_blocks_hidden_modules_and_sources(tmp_path) -> None:
    """A python started from the sandbox imports the public package but cannot
    import the hidden modules, open their source or the run artifacts, read
    predicators source outside the import system, or spawn an unguarded child
    interpreter."""
    sandbox = str(tmp_path)
    repo = str(find_repo_root())
    write_pyguard(sandbox, repo)
    (tmp_path / "mine.txt").write_text("hello")
    ok = _run(
        "import numpy, predicators.structs, predicators.utils\n"
        "print(open('mine.txt').read())", sandbox)
    assert ok.returncode == 0, ok.stderr
    assert "hello" in ok.stdout
    env_init = os.path.join(repo, "predicators", "envs", "__init__.py")
    cases: List[str] = [
        "import predicators.envs",
        "import importlib; importlib.import_module('predicators.envs')",
        f"open({env_init!r})",
        f"open({os.path.join(repo, 'logs', 'x.txt')!r})",
        "import inspect, predicators.structs as s; inspect.getsource(s)",
        "import predicators.structs as s; open(s.__file__).read()",
        "import predicators.structs as s; "
        "s.__loader__.get_source('predicators.structs')",
        "import importlib.util as u; "
        f"spec = u.spec_from_file_location('x', {env_init!r}); "
        "spec.loader.exec_module(u.module_from_spec(spec))",
        "import subprocess, sys; "
        "subprocess.run([sys.executable, '-S', '-c', '1'])",
        "import subprocess, sys; "
        "subprocess.run([sys.executable, '-c', '1'], env={})",
    ]
    for code in cases:
        res = _run(code, sandbox)
        assert res.returncode != 0, code
        # inspect.getsource swallows the guard's OSError into its own.
        assert ("sandbox guard" in res.stderr
                or "could not get source code" in res.stderr), (code,
                                                                res.stderr)
    # A guarded child inherits the guard and keeps working.
    res = _run(
        "import subprocess, sys; "
        "subprocess.run([sys.executable, '-c', 'import predicators.structs'],"
        " check=True)", sandbox)
    assert res.returncode == 0, res.stderr


def test_pyguard_classifies_by_inode_when_realpath_cannot(tmp_path) -> None:
    """A spelling of the repo that realpath cannot map onto the guard's
    prefixes (a bind mount, simulated here by a symlink with realpath disabled)
    is still classified: the hidden dirs deny, the sandbox allows."""
    repo = tmp_path / "repo"
    (repo / "predicators" / "envs").mkdir(parents=True)
    (repo / "predicators" / "envs" / "secret.py").write_text("x = 1\n")
    (repo / "logs" / "sb").mkdir(parents=True)
    (repo / "logs" / "sb" / "mine.txt").write_text("hello\n")
    alias = tmp_path / "alias"
    alias.symlink_to(repo, target_is_directory=True)
    sandbox = str(repo / "logs" / "sb")
    write_pyguard(sandbox, str(repo))
    prelude = ("import os, sitecustomize as g; "
               "g._resolve = lambda p: os.fsdecode(p) if isinstance(p, bytes)"
               " else os.fspath(p); ")
    hidden = str(alias / "predicators" / "envs" / "secret.py")
    res = _run(prelude + f"open({hidden!r}).read()", sandbox)
    assert res.returncode != 0 and "sandbox guard" in res.stderr, res.stderr
    mine = str(alias / "logs" / "sb" / "mine.txt")
    res = _run(prelude + f"print(open({mine!r}).read())", sandbox)
    assert res.returncode == 0, res.stderr
    assert "hello" in res.stdout


def test_pyguard_learns_other_spellings_from_the_mount_table(tmp_path) -> None:
    """Two mount points of one source are two spellings of one tree: the
    guard's prefixes cover the second spelling (with realpath disabled and the
    inode fallback emptied, only the mount table can do it)."""
    mnt_a, mnt_b = tmp_path / "mnt_a", tmp_path / "mnt_b"
    repo = mnt_a / "repo"
    (repo / "predicators" / "envs").mkdir(parents=True)
    (repo / "predicators" / "envs" / "secret.py").write_text("x = 1\n")
    (repo / "logs" / "sb").mkdir(parents=True)
    (repo / "logs" / "sb" / "mine.txt").write_text("hello\n")
    mnt_b.symlink_to(mnt_a, target_is_directory=True)
    mounts = tmp_path / "mounts"
    mounts.write_text(f"nfs:/export {mnt_a} nfs4 rw 0 0\n"
                      f"nfs:/export {mnt_b} nfs4 rw 0 0\n"
                      "tmpfs /run tmpfs rw 0 0\n")
    sandbox = str(repo / "logs" / "sb")
    write_pyguard(sandbox, str(repo), mounts_file=str(mounts))
    prelude = ("import os, sitecustomize as g; "
               "g._resolve = lambda p: os.fsdecode(p) if isinstance(p, bytes)"
               " else os.fspath(p); g.IDENTS[:] = []; ")
    res = _run(prelude + "print(g.ROOTS)", sandbox)
    assert res.returncode == 0, res.stderr
    assert str(mnt_b / "repo") in res.stdout
    hidden = str(mnt_b / "repo" / "predicators" / "envs" / "secret.py")
    res = _run(prelude + f"open({hidden!r}).read()", sandbox)
    assert res.returncode != 0 and "sandbox guard" in res.stderr, res.stderr
    mine = str(mnt_b / "repo" / "logs" / "sb" / "mine.txt")
    res = _run(prelude + f"print(open({mine!r}).read())", sandbox)
    assert res.returncode == 0, res.stderr
    assert "hello" in res.stdout


def test_pyguard_allows_a_sandbox_under_the_repo_logs(tmp_path) -> None:
    """The production layout puts the sandbox under <repo>/logs/, which is
    otherwise hidden: the sandbox's own files stay readable, its run dir
    (archived test-phase files) does not."""
    repo = find_repo_root()
    run_dir = repo / "logs" / f"_pyguard_test_{os.getpid()}"
    sandbox = run_dir / "sandbox"
    sandbox.mkdir(parents=True)
    try:
        write_pyguard(str(sandbox), str(repo))
        (sandbox / "mine.txt").write_text("hello")
        (run_dir / "journal_eval_cycle0.md").write_text("leak")
        res = _run("print(open('mine.txt').read())", str(sandbox))
        assert res.returncode == 0 and "hello" in res.stdout, res.stderr
        res = _run("open('../journal_eval_cycle0.md').read()", str(sandbox))
        assert res.returncode != 0 and "sandbox guard" in res.stderr
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)
    del tmp_path


def test_pyguard_lets_the_interpreter_open_a_script(tmp_path) -> None:
    """``python3 script.py`` opens the script from C with no Python frame on
    the stack; the guard must let that read through (the sandbox's PreToolUse
    hook and every agent-written script start that way)."""
    sandbox = str(tmp_path)
    write_pyguard(sandbox, str(find_repo_root()))
    (tmp_path / "script.py").write_text("print('script ok')")
    env = {**os.environ, **pyguard_env(sandbox)}
    res = subprocess.run([sys.executable, "script.py"],
                         cwd=sandbox,
                         env=env,
                         capture_output=True,
                         text=True,
                         check=False)
    assert res.returncode == 0 and "script ok" in res.stdout, res.stderr


def _traj(n: int) -> LowLevelTrajectory:
    # pylint: disable=import-outside-toplevel
    from predicators.structs import Action, State
    states = [
        State({}, simulator_state=object(), privileged={"secret": 1})
        for _ in range(n + 1)
    ]
    actions = [Action(np.zeros(1, dtype=np.float32)) for _ in range(n)]
    return LowLevelTrajectory(states, actions)


def _init_sandbox(tmp_path) -> str:
    sandbox = str(tmp_path)
    subprocess.run(["git", "init", "-q"], cwd=sandbox, check=True)
    (tmp_path / "CLAUDE.md").write_text("rules")
    git_commit_all(sandbox, "init", check=True)
    return sandbox


def test_export_trajectories_writes_once_per_change(tmp_path) -> None:
    """The pickle is rewritten only when the trajectories changed and is
    loadable with the public structs."""
    sandbox = _init_sandbox(tmp_path)
    trajs = [_traj(2), _traj(3)]
    assert export_trajectories(sandbox, trajs)
    assert not export_trajectories(sandbox, trajs)
    with open(trajectories_path(sandbox), "rb") as f:
        loaded = pickle.load(f)
    assert [len(t["actions"]) for t in loaded] == [2, 3]
    assert [len(t["states"]) for t in loaded] == [3, 4]
    # Hidden ground truth and simulator bookkeeping never reach the disk.
    for traj in loaded:
        for state in traj["states"]:
            assert state.privileged is None
            assert state.simulator_state is None
        for action in traj["actions"]:
            assert action["option"] is None
            assert action["arr"].shape == (1, )
    assert export_trajectories(sandbox, trajs + [_traj(1)])
    tracked = subprocess.run(["git", "ls-files"],
                             cwd=sandbox,
                             capture_output=True,
                             text=True,
                             check=True)
    assert "data/trajectories.pkl" in tracked.stdout


def test_snapshot_and_rollback_restore_the_sandbox(tmp_path) -> None:
    """Files added, changed or committed after the snapshot are archived
    outside the sandbox and the tree returns to the snapshot."""
    sandbox = _init_sandbox(tmp_path)
    (tmp_path / "notes.md").write_text("learning notes")
    rev = snapshot_sandbox(sandbox)
    assert rev
    # The test phase edits a tracked file, commits a log, adds a script.
    (tmp_path / "notes.md").write_text("learning notes + test leak")
    (tmp_path / "session_logs").mkdir()
    (tmp_path / "session_logs" / "004_test_task0.md").write_text("log")
    git_commit_all(sandbox,
                   "log query 4",
                   paths=["session_logs/004_test_task0.md"])
    (tmp_path / "plan.py").write_text("print(1)")
    archive = tmp_path.parent / (tmp_path.name + "_archive")
    archived = rollback_sandbox(sandbox, rev, str(archive))
    assert set(archived) == {
        "notes.md", "session_logs/004_test_task0.md", "plan.py"
    }
    assert (archive / "notes.md").read_text() == "learning notes + test leak"
    assert (tmp_path / "notes.md").read_text() == "learning notes"
    assert not (tmp_path / "plan.py").exists()
    assert not (tmp_path / "session_logs" / "004_test_task0.md").exists()
    assert snapshot_sandbox(None) is None
    with pytest.raises(subprocess.CalledProcessError):
        rollback_sandbox(sandbox, "not-a-rev", str(archive))
