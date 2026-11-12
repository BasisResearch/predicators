"""Tests for the sandbox's python guard, data export and test-phase rollback
(predicators.agent_sdk.sandbox_setup)."""
import json
import os
import pickle
import re
import shutil
import subprocess
import sys
from typing import Dict, List

import numpy as np
import pytest

from predicators.agent_sdk.sandbox_setup import GeneratedReference, \
    export_trajectories, find_repo_root, git_commit_all, pyguard_env, \
    rollback_sandbox, setup_sandbox_directory, snapshot_sandbox, \
    trajectories_path, write_pyguard
from predicators.agent_sdk.tools.sandbox_guard import \
    SANDBOX_ALLOWED_MODULES, SANDBOX_ALLOWED_PACKAGE_TREES, \
    SANDBOX_PUBLIC_MODULES
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


def test_reopened_sandbox_removes_retired_controller_source(tmp_path) -> None:
    """Reopening removes previously exported private controller references."""
    sandbox = tmp_path / "sandbox"
    ref = sandbox / "reference"
    (ref / "skill_factories").mkdir(parents=True)
    (ref / "options.py").write_text("hidden attachment implementation")
    (ref / "skill_factories" /
     "base.py").write_text("private collision geometry")
    setup_sandbox_directory(
        sandbox_dir=str(sandbox),
        repo_root=str(find_repo_root()),
        extra_reference_files={
            "skills.md": "predicators/agent_sdk/prompts/public_skills.md"
        },
        claude_md_content="public contract",
        system_prompt="public contract",
        log_dir=str(tmp_path))
    assert sorted(p.name for p in ref.iterdir()) == ["skills.md"]
    assert (ref / "skills.md").read_text().startswith("# Skill API")


def test_generated_references_are_written_into_the_sandbox(tmp_path) -> None:
    """A generated reference lands in the sandbox as its text, beside the
    copied ones, and nothing is staged in the log directory."""
    sandbox = tmp_path / "sandbox"
    log_dir = tmp_path / "log"
    setup_sandbox_directory(
        sandbox_dir=str(sandbox),
        repo_root=str(find_repo_root()),
        extra_reference_files={
            "skills.md":
            "predicators/agent_sdk/prompts/public_skills.md",
            "scene/scene_manifest.json":
            GeneratedReference('{"bodies": []}\n', "a test scene"),
        },
        claude_md_content="public contract",
        system_prompt="public contract",
        log_dir=str(log_dir))
    ref = sandbox / "reference"
    assert (ref / "scene" / "scene_manifest.json").read_text() == \
        '{"bodies": []}\n'
    assert (ref / "skills.md").read_text().startswith("# Skill API")
    assert sorted(p.name
                  for p in log_dir.iterdir()) == ["full_system_prompt.md"]


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
        "from predicators.code_sim_learning import base_simulator",
        "import predicators.code_sim_learning.continual_oracle",
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


# Modules a sandboxed python could import before the guard held an
# allowlist: the harness's fit, its belief, the scene loader, motion
# planning, the approaches and the agent tooling.
_UNAVAILABLE_MODULES = [
    "predicators.code_sim_learning.physical_sysid",
    "predicators.code_sim_learning.parameter_belief",
    "predicators.code_sim_learning.scene_loader",
    "predicators.code_sim_learning.fit_space",
    "predicators.code_sim_learning.commands",
    "predicators.code_sim_learning.rollout_env",
    "predicators.pybullet_helpers.motion_planning",
    "predicators.pybullet_helpers.controllers",
    "predicators.pybullet_helpers.objects",
    "predicators.pybullet_helpers.robots",
    "predicators.approaches",
    "predicators.agent_sdk.belief_probe",
    "predicators.observation_noise",
]
_UNAVAILABLE_MESSAGE = re.compile(
    r"^ImportError: sandbox guard: (predicators[\w.]*) is not available in "
    r"the sandbox; use \./reference/ and the tools$")
# Tries every module in every spelling inside one guarded python, and
# prints what each attempt raised.
_IMPORT_PROBE = """
import importlib, json, sys
names = json.loads(sys.argv[1])
out = {}
for name in names:
    for spelling, load in (
            ("statement", lambda n: exec("import " + n, {})),
            ("from", lambda n: exec(
                "from %s import %s" % tuple(n.rsplit(".", 1))
                if "." in n else "import " + n, {})),
            ("__import__", __import__),
            ("importlib", importlib.import_module)):
        try:
            load(name)
            out[name + " " + spelling] = "loaded"
        except BaseException as e:
            out[name + " " + spelling] = type(e).__name__ + ": " + str(e)
        loaded = [m for m in sys.modules if m == name]
        if loaded:
            out[name + " " + spelling] += " (in sys.modules)"
print("PROBE " + json.dumps(out) + " END")
"""


def _probe_imports(names: List[str], sandbox: str) -> Dict[str, str]:
    env = {**os.environ, **pyguard_env(sandbox)}
    res = subprocess.run(
        [sys.executable, "-c", _IMPORT_PROBE,
         json.dumps(names)],
        cwd=sandbox,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=300)
    assert res.returncode == 0, res.stderr[-2000:]
    # PyBullet prints to stdout too, so the report sits between markers.
    found = re.search(r"PROBE (\{.*\}) END", res.stdout)
    assert found is not None, res.stdout[-2000:]
    return json.loads(found.group(1))


def test_pyguard_refuses_predicators_modules_off_the_list(tmp_path) -> None:
    """A python started from the sandbox imports no predicators module off the
    allowlist, in any spelling, whoever imports it.

    The refusal names the module and says nothing about what it is for.
    ``importlib.import_module`` raises no import event, so the read of
    the module's file is what stops it.
    """
    sandbox = str(tmp_path)
    write_pyguard(sandbox, str(find_repo_root()))
    results = _probe_imports(_UNAVAILABLE_MODULES, sandbox)
    assert len(results) == 4 * len(_UNAVAILABLE_MODULES)
    for attempt, outcome in results.items():
        said = _UNAVAILABLE_MESSAGE.match(outcome)
        assert said is not None, (attempt, outcome)
        assert said.group(1) == attempt.split()[0], (attempt, outcome)
    # The hidden modules keep their own message.
    hidden = _probe_imports([
        "predicators.envs", "predicators.ground_truth_models",
        "predicators.code_sim_learning.base_simulator"
    ], sandbox)
    for attempt, outcome in hidden.items():
        assert "sandbox guard" in outcome, (attempt, outcome)
        assert "is not available in the sandbox" not in outcome, attempt
        assert "(in sys.modules)" not in outcome, attempt
        if not attempt.endswith(" importlib"):
            name = attempt.split()[0]
            assert outcome == (
                f"ImportError: sandbox guard: importing {name} would expose "
                "the hidden environment implementation; use ./reference/ "
                "and the tools"), attempt
    # A file of the package loaded under another name is still that module.
    lm_path = os.path.join(str(find_repo_root()), "predicators",
                           "code_sim_learning", "lm.py")
    res = _run(
        "import importlib.util as u; "
        f"spec = u.spec_from_file_location('mine', {lm_path!r}); "
        "spec.loader.exec_module(u.module_from_spec(spec))", sandbox)
    assert res.returncode != 0
    assert ("sandbox guard: predicators.code_sim_learning.lm is not "
            "available in the sandbox") in res.stderr, res.stderr


def test_pyguard_keeps_what_agents_are_told_to_use(tmp_path) -> None:
    """The public modules, the libraries, the recorded trajectories and a
    guarded child of a guarded python all keep working under the allowlist."""
    sandbox = _init_sandbox(tmp_path)
    write_pyguard(sandbox, str(find_repo_root()))
    results = _probe_imports([
        "predicators.structs", "predicators.utils", "predicators.settings",
        "predicators.pybullet_helpers.geometry",
        "predicators.third_party.fast_downward_translator.pddl", "numpy",
        "scipy.optimize", "pybullet"
    ], sandbox)
    for attempt, outcome in results.items():
        assert outcome == "loaded (in sys.modules)", (attempt, outcome)
    res = _run(
        "from predicators.structs import Object, State, Type\n"
        "from predicators import utils\n"
        "import numpy as np, pybullet as p\n"
        "client = p.connect(p.DIRECT)\n"
        "p.stepSimulation(physicsClientId=client)\n"
        "p.disconnect(client)\n"
        "ball = Object('ball0', Type('ball', ['x']))\n"
        "state = State({ball: np.array([1.0])})\n"
        "print('OK', float(state.get(ball, 'x')),\n"
        "      utils.PyBulletState.__name__)", sandbox)
    assert res.returncode == 0 and "OK 1.0 PyBulletState" in res.stdout, \
        res.stderr
    # The recorded data loads as the sandbox's CLAUDE.md shows.
    assert export_trajectories(sandbox, [_object_traj(2), _object_traj(3)])
    res = _run(
        "import pickle\n"
        "with open('./data/trajectories.pkl', 'rb') as f:\n"
        "    episodes = pickle.load(f)\n"
        "state = episodes[1]['states'][-1]\n"
        "print('LOADED', [len(e['states']) for e in episodes],\n"
        "      type(state).__module__, type(state).__name__,\n"
        "      [(o.name, o.type.name, float(state.get(o, 'x')))\n"
        "       for o in state])", sandbox)
    assert res.returncode == 0, res.stderr
    assert ("LOADED [3, 4] predicators.structs State "
            "[('ball0', 'ball', 3.0)]") in res.stdout, res.stdout
    # A guarded child of a guarded python keeps both: the public modules
    # load in it, and the allowlist still holds.
    res = _run(
        "import subprocess, sys\n"
        "subprocess.run([sys.executable, '-c',\n"
        "                'import predicators.structs, predicators.utils'],\n"
        "               check=True)\n"
        "inner = subprocess.run(\n"
        "    [sys.executable, '-c', 'import predicators.approaches'],\n"
        "    capture_output=True, text=True)\n"
        "assert inner.returncode != 0, 'the child lost the guard'\n"
        "print(inner.stderr.strip().splitlines()[-1])", sandbox)
    assert res.returncode == 0, res.stderr
    assert ("ImportError: sandbox guard: predicators.approaches is not "
            "available in the sandbox") in res.stdout, res.stdout


def _loaded_predicators_modules(guard_sandbox: str = "") -> List[str]:
    """The predicators modules a fresh python holds after it imports the public
    modules; under the guard of ``guard_sandbox`` when one is given."""
    code = ("import json, sys\n"
            f"import {', '.join(SANDBOX_PUBLIC_MODULES)}\n"
            "names = sorted(m for m in sys.modules if m == 'predicators'\n"
            "               or m.startswith('predicators.'))\n"
            "print('LOADED ' + json.dumps(names) + ' END')\n")
    env = dict(os.environ)
    if guard_sandbox:
        env.update(pyguard_env(guard_sandbox))
    res = subprocess.run([sys.executable, "-c", code],
                         env=env,
                         capture_output=True,
                         text=True,
                         check=False,
                         timeout=300)
    assert res.returncode == 0, (
        "the public modules " + ", ".join(SANDBOX_PUBLIC_MODULES) +
        (" do not load under the sandbox's python guard; a module they "
         "import is missing from SANDBOX_ALLOWED_MODULES "
         "(predicators/agent_sdk/tools/sandbox_guard.py):\n"
         if guard_sandbox else " do not load:\n") + res.stderr[-2000:])
    # PyBullet prints to stdout too, so the list sits between markers.
    found = re.search(r"LOADED (\[.*\]) END", res.stdout)
    assert found is not None, res.stdout[-2000:]
    return json.loads(found.group(1))


def test_pyguard_allowlist_is_what_the_public_modules_load(tmp_path) -> None:
    """The allowlist is exactly the set of predicators modules that importing
    the public modules loads.

    A new import in ``predicators.structs`` or ``predicators.utils``
    would otherwise break every agent script at its first line, and a
    module the allowlist keeps for nothing is a module an agent can
    import for nothing.
    """
    loaded = _loaded_predicators_modules()
    listed = [
        name for name in loaded
        if not any(name == tree or name.startswith(tree + ".")
                   for tree in SANDBOX_ALLOWED_PACKAGE_TREES)
    ]
    assert len(listed) < len(loaded), "the vendored tree no longer loads"
    missing = sorted(set(listed) - set(SANDBOX_ALLOWED_MODULES))
    assert not missing, (
        f"{', '.join(SANDBOX_PUBLIC_MODULES)} now load {missing}, which "
        "SANDBOX_ALLOWED_MODULES (predicators/agent_sdk/tools/"
        "sandbox_guard.py) does not list. The sandbox's python guard "
        "refuses every predicators module off that list, so agent scripts "
        "that import the public modules would fail. Remove the new import "
        "from the public modules, or add these modules to the list if "
        "every arm's agent may import them.")
    stale = sorted(set(SANDBOX_ALLOWED_MODULES) - set(listed))
    assert not stale, (
        f"SANDBOX_ALLOWED_MODULES lists {stale}, which "
        f"{', '.join(SANDBOX_PUBLIC_MODULES)} no longer load. Remove them "
        "from the list (predicators/agent_sdk/tools/sandbox_guard.py): an "
        "agent could import them for no reason.")
    # The same set loads under the guard itself.
    sandbox = str(tmp_path)
    write_pyguard(sandbox, str(find_repo_root()))
    assert _loaded_predicators_modules(sandbox) == loaded


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


def _object_traj(n: int) -> LowLevelTrajectory:
    """A recording of one object, as the agent's data holds them."""
    # pylint: disable=import-outside-toplevel
    from predicators.structs import Action, Object, State, Type
    ball = Object("ball0", Type("ball", ["x"]))
    states = [
        State({ball: np.array([float(i)])},
              simulator_state=object(),
              privileged={"secret": 1}) for i in range(n + 1)
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

    def _dirty() -> bool:
        status = subprocess.run(["git", "status", "--porcelain"],
                                cwd=sandbox,
                                capture_output=True,
                                text=True,
                                check=True)
        return "data/trajectories.pkl" in status.stdout

    # A refresh between the agent's turns rewrites the tracked file
    # without a commit; the next committing refresh settles it even when
    # the bytes have not changed since.
    more = trajs + [_traj(1), _traj(4)]
    assert export_trajectories(sandbox, more, commit=False)
    assert _dirty()
    assert not export_trajectories(sandbox, more)
    assert not _dirty()


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
