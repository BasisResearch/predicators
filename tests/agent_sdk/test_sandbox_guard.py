"""Tests for the sandbox escape guard and oversize-output spilling.

Covers ``_screen_text_for_sandbox_escape`` (used in-process by
``run_python``), ``_make_spilling_text_result`` (oversize tool output
spilled into the sandbox instead of dumped to ``~/.claude/...``), and
the self-contained ``VALIDATE_SANDBOX_SCRIPT`` Bash/file-path hook.
"""
# pylint: disable=protected-access
from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys

from predicators.agent_sdk.sandbox_setup import VALIDATE_SANDBOX_SCRIPT, \
    sandbox_settings
from predicators.agent_sdk.tools.results import _make_spilling_text_result
from predicators.agent_sdk.tools.sandbox_guard import \
    SANDBOX_ALLOWED_MODULES, SANDBOX_ALLOWED_MODULES_PATTERN, \
    SANDBOX_ALLOWED_PACKAGES, SANDBOX_EXTRA_MODULES, \
    SANDBOX_MODEL_FILE_MODULES, _screen_text_for_sandbox_escape, \
    agent_code_import_refusal, allowed_modules_pattern, \
    model_file_import_refusal

# (text, should_block) — exercised against both the in-process screen and
# the live hook script (as a Bash command).
_ALLOW = [
    "from predicators.structs import State",
    "python3 my_experiment.py",
    "exec(open('./simulator.py').read())",
    'print("loading...")',
    "x = 10 / 2",
    'grep -rn "pattern" ./session_logs',
    'print("/done")',  # absolute-looking but not a real system root
    'print("see https://example.com/a/b")',  # URL, not a path
]
_BLOCK = [
    "import inspect; inspect.getsource(State)",
    "inspect.getfile(State)",
    'open("/etc/passwd").read()',
    "cat /Users/me/.claude/projects/x/tool-results/y.txt",
    "open('../../secret.txt')",
    "grep x /usr/lib/python3/site-packages/foo.py",
    # Hidden-implementation imports (run_20260717_182040 seed1 turn 22
    # pulled real domino constants this way); the public authoring
    # surface (predicators.structs, in _ALLOW) stays importable.
    "from predicators.envs.pybullet_domino import PyBulletDominoEnv",
    "import predicators.ground_truth_models.domino as gt",
    "from predicators.code_sim_learning.base_simulator import "
    "oracle_base_simulator_class",
    "from predicators.code_sim_learning.continual_oracle import oracle_source",
    "import predicators.code_sim_learning.boil_oracle as oracle",
    "import predicators.code_sim_learning.bridge_oracle as oracle",
    "from predicators.code_sim_learning import base_simulator",
]


def test_screen_allows_in_sandbox_paths(tmp_path) -> None:
    """Legitimate relative/in-sandbox code is not flagged."""
    sandbox = str(tmp_path)
    inside = tmp_path / "reference" / "structs.py"
    assert _screen_text_for_sandbox_escape(f"open('{inside}')",
                                           sandbox) is None
    for text in _ALLOW:
        assert _screen_text_for_sandbox_escape(text, sandbox) is None, text


def test_screen_blocks_escapes(tmp_path) -> None:
    """Absolute/``..`` escapes and source introspection are flagged."""
    sandbox = str(tmp_path)
    for text in _BLOCK:
        assert _screen_text_for_sandbox_escape(text, sandbox) is not None, text


def test_screen_blocks_hidden_state_access(tmp_path) -> None:
    """Executed code cannot reach hidden env ground truth: recorded states'
    ``privileged`` channel, the probe's ``_ctx`` (whose ``env`` is the live
    real env), the full probe a facade ``sim`` holds in ``_probe``, or the
    hidden-dynamics kill switch (exploits observed in run_20260819_183137 seed1
    and run_20260819_183252 seed2).

    Python-only: the Bash hook does not screen these tokens (a shell
    command cannot reach in-process objects, and grepping old session
    logs for the words is legitimate), so they stay out of ``_BLOCK``.
    """
    sandbox = str(tmp_path)
    for text in [
            "print(s.privileged)",
            "e = sim._ctx.env",
            "getattr(sim, '_ctx').env.cure_threshold",
            "sim._probe.belief()",
            "sys.modules[type(sim).__module__]._PROBES[sim]",
            "type(sim).run.__globals__",
            "e._skip_domain_specific_dynamics = False",
    ]:
        assert _screen_text_for_sandbox_escape(text, sandbox) is not None, text
    # Word boundaries: harness-styled identifiers that merely end in
    # ``ctx`` (or mention privilege in prose) are untouched.
    for text in [
            "budget_ctx = 3",
            "print('privilege separation')",
            "ctx = make_ctx()",
            "my_probe = 1",
    ]:
        assert _screen_text_for_sandbox_escape(text, sandbox) is None, text


def test_screen_without_sandbox_dir_still_guards_hidden_state() -> None:
    """``sandbox_dir=None`` must not disable screening.

    An in-process session with no sandbox copy still execs into the live
    namespace, so the hidden-state, introspection, and import screens
    apply regardless. Only the path screen, which has no boundary to
    enforce, is skipped.
    """
    for text in [
            "print(s.privileged)",
            "e = sim._ctx.env",
            "inspect.getsource(sim)",
            "from predicators.envs.pybullet_bridge import PyBulletBridgeEnv",
    ]:
        assert _screen_text_for_sandbox_escape(text, None) is not None, text
    assert _screen_text_for_sandbox_escape("open('/etc/hosts')", None) is None


# Agent Python that names a predicators module off the sandbox's list:
# (code, the name the refusal carries). Python only: the Bash hook leaves
# these to the python guard every ``python3`` loads, which refuses the
# import itself.
_UNAVAILABLE = [
    ("import predicators.code_sim_learning.physical_sysid",
     "predicators.code_sim_learning.physical_sysid"),
    ("import predicators.code_sim_learning.physical_sysid as sysid",
     "predicators.code_sim_learning.physical_sysid"),
    ("from predicators.code_sim_learning.parameter_belief import "
     "BeliefConfig", "predicators.code_sim_learning.parameter_belief"),
    ("from predicators.code_sim_learning import scene_loader",
     "predicators.code_sim_learning.scene_loader"),
    ("from predicators.code_sim_learning.fit_space import ParamSpec",
     "predicators.code_sim_learning.fit_space"),
    ("from predicators.code_sim_learning.commands import ApplyForce",
     "predicators.code_sim_learning.commands"),
    ("from predicators.pybullet_helpers.motion_planning import "
     "run_motion_planning", "predicators.pybullet_helpers.motion_planning"),
    ("from predicators.pybullet_helpers import motion_planning",
     "predicators.pybullet_helpers.motion_planning"),
    ("from predicators.pybullet_helpers import (\n    geometry,\n"
     "    motion_planning,\n)",
     "predicators.pybullet_helpers.motion_planning"),
    ("from predicators.pybullet_helpers import \\\n    controllers",
     "predicators.pybullet_helpers.controllers"),
    ("import predicators.approaches", "predicators.approaches"),
    ("from predicators import approaches", "predicators.approaches"),
    ("from predicators import utils, approaches as a",
     "predicators.approaches"),
    ("import numpy, predicators.agent_sdk.belief_probe",
     "predicators.agent_sdk.belief_probe"),
    ("from predicators.agent_sdk.belief_probe import BeliefProbe",
     "predicators.agent_sdk.belief_probe"),
    ("if True:\n    import predicators.observation_noise",
     "predicators.observation_noise"),
    ("import importlib\n"
     "m = importlib.import_module('predicators.code_sim_learning.fitting')",
     "predicators.code_sim_learning.fitting"),
    ('m = __import__("predicators.code_sim_learning.lm")',
     "predicators.code_sim_learning.lm"),
    ('import sys\n'
     'm = sys.modules["predicators.code_sim_learning.physical_sysid"]',
     "predicators.code_sim_learning.physical_sysid"),
    ("import sys\n"
     "m = sys.modules.get('predicators.code_sim_learning.orchestrator')",
     "predicators.code_sim_learning.orchestrator"),
    ("import predicators\n"
     "sse = predicators.code_sim_learning.fitting.compute_sse",
     "predicators.code_sim_learning.fitting.compute_sse"),
    ("exec('import predicators.option_model')", "predicators.option_model"),
]
# Agent Python that stays on the list.
_AVAILABLE = [
    "import predicators",
    "import predicators.structs",
    "import predicators.utils as utils",
    "from predicators import utils",
    "from predicators import structs, utils",
    "from predicators.structs import State, Type",
    "from predicators.structs import (\n    Object,\n    State,\n)",
    "from predicators.utils import PyBulletState",
    "from predicators.settings import CFG",
    "from predicators.pybullet_helpers.geometry import Pose",
    "from predicators.pybullet_helpers import geometry, joint",
    "import predicators.third_party.fast_downward_translator.pddl as pddl",
    "import importlib\nm = importlib.import_module('predicators.structs')",
    "import sys\nm = sys.modules['predicators.utils']",
    "s = predicators.structs.State({})",
    "import pybullet, numpy, scipy.optimize",
    "import pickle\n"
    "with open('./data/trajectories.pkl', 'rb') as f:\n"
    "    episodes = pickle.load(f)",
    "open('./notes/predicators.txt').read()",
    "print('the predicators package')",
]


def test_screen_refuses_predicators_modules_off_the_list(tmp_path) -> None:
    """``run_python`` code runs in the harness process, where every module is
    loaded, so its text is the barrier: any spelling that names a predicators
    module off the allowlist is refused, with a reason that names the module
    and says nothing about what it is for."""
    sandbox = str(tmp_path)
    for text, name in _UNAVAILABLE:
        for sandbox_dir in (sandbox, None):
            reason = _screen_text_for_sandbox_escape(text, sandbox_dir)
            assert reason == (f"{name} is not available in the sandbox; use "
                              "./reference/ and the tools"), text
    for text in _AVAILABLE:
        assert _screen_text_for_sandbox_escape(text, sandbox) is None, text
        assert agent_code_import_refusal(text) is None, text
    # The hidden modules keep their own reason, in every spelling.
    for text in [
            "from predicators import envs",
            "m = __import__('predicators.envs.pybullet_fan')",
            "import sys\nm = sys.modules['predicators.ground_truth_models']",
            "import importlib\nm = importlib.import_module("
            "'predicators.code_sim_learning.base_simulator')",
    ]:
        reason = _screen_text_for_sandbox_escape(text, sandbox)
        assert reason is not None and "the sandbox hides" in reason, text


def test_allowed_modules_pattern_is_exact() -> None:
    """The shared pattern admits the listed modules and the vendored tree, and
    nothing beside them: no sibling of a listed module, and no name that only
    starts like one."""
    allowed = re.compile(SANDBOX_ALLOWED_MODULES_PATTERN)
    for name in SANDBOX_ALLOWED_MODULES + SANDBOX_EXTRA_MODULES + (
            "predicators.third_party",
            "predicators.third_party.fast_downward_translator.pddl.actions",
    ):
        assert allowed.fullmatch(name), name
    for name in [
            "predicators.structs2",
            "predicators.utils.extra",
            "predicators.pybullet_helpers.motion_planning",
            "predicators.pybullet_helpers.robots",
            "predicators.code_sim_learning.fit_space",
            "predicators.code_sim_learning.commands",
            "predicators.code_sim_learning.scene_loader",
            "predicators.agent_sdk.belief_probe",
            "predicators.third_party_extra",
            "predicators.approaches",
            "predicatorsXstructs",
    ]:
        assert not allowed.fullmatch(name), name
    # The text screen needs to know which listed names are packages.
    for name in SANDBOX_ALLOWED_MODULES + SANDBOX_EXTRA_MODULES:
        spec = importlib.util.find_spec(name)
        assert spec is not None, name
        is_package = spec.submodule_search_locations is not None
        assert is_package == (name in SANDBOX_ALLOWED_PACKAGES), name
    # An arm's own extra modules are admitted exactly as well.
    wider = re.compile(
        allowed_modules_pattern(["predicators.pybullet_helpers.link"]))
    assert wider.fullmatch("predicators.pybullet_helpers.link")
    assert not wider.fullmatch("predicators.pybullet_helpers.links")
    assert not allowed.fullmatch("predicators.pybullet_helpers.link")


def test_model_file_screen_admits_the_two_imports_agents_are_shown() -> None:
    """A model file may import ``ParamSpec`` and the physics commands, as the
    model contract and the engine wrapper show; every other module is refused
    as in ``run_python``."""
    for text in [
            "from predicators.code_sim_learning.fit_space import ParamSpec",
            "from predicators.code_sim_learning.commands import ApplyForce, "
            "Attach, SetVelocity",
            "from predicators.code_sim_learning import commands",
            "import predicators.code_sim_learning.commands as commands",
            "from predicators.structs import State",
    ]:
        assert model_file_import_refusal(text) is None, text
    for module in SANDBOX_MODEL_FILE_MODULES:
        assert agent_code_import_refusal(f"import {module}") is not None
    for text, name in [
        ("from predicators.code_sim_learning import fitting",
         "predicators.code_sim_learning.fitting"),
        ("from predicators.code_sim_learning.physical_sysid import x",
         "predicators.code_sim_learning.physical_sysid"),
        ("from predicators.code_sim_learning.scene_loader import LoadedScene",
         "predicators.code_sim_learning.scene_loader"),
        ("import predicators.pybullet_helpers.motion_planning",
         "predicators.pybullet_helpers.motion_planning"),
    ]:
        assert model_file_import_refusal(text) == (
            f"{name} is not available in the sandbox; use ./reference/ and "
            "the tools"), text
    hidden = model_file_import_refusal(
        "from predicators.envs.pybullet_env import PyBulletEnv")
    assert hidden is not None and "the sandbox hides" in hidden


# Bash-only: forms that would start a python without the sitecustomize
# guard. ``run_python`` runs in-process, so its screen ignores them.
_BLOCK_BASH_ONLY = [
    "PYTHONPATH=/x python3 script.py",
    "export PYTHONPATH=.",
    "unset PYTHONPATH; python3 script.py",
    "env -i python3 script.py",
    "python3 -S script.py",
    "python3 -I -c 'print(1)'",
    "python -Es script.py",
    "/usr/bin/python3 -S script.py",
    # Hidden imports inside a quoted python -c body, or spelled through
    # __import__ / importlib, are screened wherever they appear.
    "python3 -c 'import predicators.envs.pybullet_boil'",
    "python3 -c 'from predicators.envs.pybullet_boil import PyBulletBoilEnv'",
    "python3 -c \"__import__('predicators.ground_truth_models.boil')\"",
]
_ALLOW_BASH = [
    "python3 -c 'import sys; print(sys.path)'",
    "python3 -m pytest -q",
    "python3 -W ignore -u script.py",
    "python3 -B -O script.py",
    "./venv/bin/python3 script.py",
]


def _run_hook(tmp_path, tool_name, tool_input) -> bool:
    """Run the generated hook script; return True if it denied the call."""
    script = tmp_path / "validate_sandbox.py"
    script.write_text(VALIDATE_SANDBOX_SCRIPT)
    payload = json.dumps({
        "tool_name": tool_name,
        "tool_input": tool_input,
    })
    out = subprocess.run([sys.executable, "validate_sandbox.py"],
                         cwd=str(tmp_path),
                         input=payload,
                         capture_output=True,
                         text=True,
                         check=True)
    return '"deny"' in out.stdout


def test_hook_screens_bash(tmp_path) -> None:
    """The hook script blocks escaping Bash commands and allows safe ones."""
    for text in _ALLOW:
        assert not _run_hook(tmp_path, "Bash", {"command": text}), text
    for text in _BLOCK + _BLOCK_BASH_ONLY:
        assert _run_hook(tmp_path, "Bash", {"command": text}), text
    for text in _ALLOW_BASH:
        assert not _run_hook(tmp_path, "Bash", {"command": text}), text
        assert _screen_text_for_sandbox_escape(text, str(tmp_path)) is None


def test_hook_denies_real_paths_under_unlisted_roots(tmp_path) -> None:
    """An absolute path that exists on disk is a real location even when its
    mount is not in SYSTEM_ROOTS (bind mounts such as /orcd), so a ``cat`` of
    it is denied; printed data that only looks like a path is still allowed."""
    script = tmp_path / "validate_sandbox.py"
    from predicators.agent_sdk.tools.sandbox_guard import \
        SANDBOX_SYSTEM_ROOTS  # pylint: disable=import-outside-toplevel
    stripped = VALIDATE_SANDBOX_SCRIPT.replace(
        "SYSTEM_ROOTS = " + repr(SANDBOX_SYSTEM_ROOTS), "SYSTEM_ROOTS = ()")
    assert stripped != VALIDATE_SANDBOX_SCRIPT

    def run(command: str) -> bool:
        script.write_text(stripped)
        payload = json.dumps({
            "tool_name": "Bash",
            "tool_input": {
                "command": command
            }
        })
        out = subprocess.run([sys.executable, "validate_sandbox.py"],
                             cwd=str(tmp_path),
                             input=payload,
                             capture_output=True,
                             text=True,
                             check=True)
        return '"deny"' in out.stdout

    outside = os.path.join(sys.prefix, "lib")  # exists, not under /tmp
    assert not outside.startswith(str(tmp_path))
    assert run(f"cat {outside}/x.py")
    assert run(f"ls {os.path.dirname(sys.prefix)}")
    # The sandbox's own mount root is real even with no roots listed.
    assert run(f"cat {tmp_path.parent}/other.txt")
    assert not run('print("/done")')
    assert not run("cat /no_such_root_xyz/file.txt")
    assert not run(f"cat {tmp_path}/mine.txt")


def test_hook_reads_other_spellings_from_the_mount_table(tmp_path) -> None:
    """With a mount table naming a second mount of the sandbox's source, the
    sandbox under that spelling is inside; a hidden path under it is still
    outside."""
    mnt_a, mnt_b = tmp_path / "mnt_a", tmp_path / "mnt_b"
    sandbox = mnt_a / "sb"
    sandbox.mkdir(parents=True)
    (mnt_a / "secret.txt").write_text("x")
    mnt_b.symlink_to(mnt_a, target_is_directory=True)
    mounts = tmp_path / "mounts"
    mounts.write_text(f"nfs:/export {mnt_a} nfs4 rw 0 0\n"
                      f"nfs:/export {mnt_b} nfs4 rw 0 0\n")
    script = sandbox / "validate_sandbox.py"
    script.write_text(
        VALIDATE_SANDBOX_SCRIPT.replace('MOUNTS_FILE = "/proc/self/mounts"',
                                        f"MOUNTS_FILE = {str(mounts)!r}"))
    # Run the script's definitions with the request stubbed out (the
    # tool dispatch after the "# File-path tools" marker is dropped).
    probe = ("import json, os, sys; sys.argv = ['x']; "
             "g = {'__file__': os.path.abspath('validate_sandbox.py')}; "
             "exec(open('validate_sandbox.py').read().replace("
             "'data = json.load(sys.stdin)', 'data = {}').split("
             "'\\n# File-path tools')[0], g); "
             "print(json.dumps({'spellings': g['SANDBOX_SPELLINGS'], "
             f"'inside': g['within']({str(mnt_b / 'sb' / 'a.txt')!r}), "
             f"'outside': g['within']({str(mnt_b / 'secret.txt')!r})}}))")
    out = subprocess.run([sys.executable, "-c", probe],
                         cwd=str(sandbox),
                         capture_output=True,
                         text=True,
                         check=True)
    got = json.loads(out.stdout.strip().splitlines()[-1])
    assert str(mnt_b / "sb") in got["spellings"]
    assert got["inside"] and not got["outside"]


def test_hook_validates_file_paths(tmp_path) -> None:
    """The hook blocks file tools targeting paths outside the sandbox."""
    inside = str(tmp_path / "x.txt")
    assert not _run_hook(tmp_path, "Read", {"file_path": inside})
    assert not _run_hook(tmp_path, "Read", {"file_path": "./y.txt"})
    assert _run_hook(tmp_path, "Read", {"file_path": "/etc/passwd"})
    assert _run_hook(tmp_path, "Grep", {"path": "/usr/lib"})
    # Tools the hook does not screen are allowed through.
    assert not _run_hook(tmp_path, "WebFetch", {"url": "http://x"})


def test_spilling_inline_small_and_spills_large(tmp_path) -> None:
    """Small output stays inline; large output spills into the sandbox."""
    text = _make_spilling_text_result(str(tmp_path))
    small = text("hello")
    assert small == {"content": [{"type": "text", "text": "hello"}]}
    assert not (tmp_path / "tool_outputs").exists()

    big = "\n".join(f"line {i} " + "x" * 100 for i in range(2000))
    res = text(big)
    preview = res["content"][0]["text"]
    assert "output too large to inline" in preview
    assert "./tool_outputs/result_0001.txt" in preview
    spilled = tmp_path / "tool_outputs" / "result_0001.txt"
    assert spilled.exists() and spilled.read_text() == big
    # Counter advances on the next oversize result.
    assert "result_0002.txt" in text(big)["content"][0]["text"]


def test_spilling_noop_without_sandbox() -> None:
    """With no sandbox dir, output is always returned inline."""
    text = _make_spilling_text_result(None)
    big = "z" * 50000
    assert text(big) == {"content": [{"type": "text", "text": big}]}


def test_hook_survives_a_cd_into_a_subdirectory(tmp_path) -> None:
    """The settings name the guard by absolute path and the guard keeps the
    sandbox as its root, so a session that cd'ed into a subdirectory (a from-
    assets agent ran ``cd reference/base_sim``) can still use its tools and cd
    back, and still cannot leave."""
    sandbox = tmp_path / "sandbox"
    (sandbox / ".claude").mkdir(parents=True)
    (sandbox / ".claude" /
     "validate_sandbox.py").write_text(VALIDATE_SANDBOX_SCRIPT)
    subdir = sandbox / "reference" / "base_sim"
    subdir.mkdir(parents=True)
    (tmp_path / "outside.txt").write_text("x")
    command = sandbox_settings(
        str(sandbox))["hooks"]["PreToolUse"][0]["hooks"][0]["command"]

    def denied(tool_name, tool_input) -> bool:
        payload = json.dumps({
            "tool_name": tool_name,
            "tool_input": tool_input,
            "cwd": str(subdir),
        })
        # A hook command runs through the shell in the session's directory.
        out = subprocess.run(command,
                             shell=True,
                             cwd=str(subdir),
                             input=payload,
                             capture_output=True,
                             text=True,
                             check=True)
        return '"deny"' in out.stdout

    assert not denied("Bash", {"command": "cd ../.. && ls"})
    assert not denied("Read", {"file_path": "../../simulator.py"})
    assert not denied("Read", {"file_path": str(sandbox / "simulator.py")})
    assert denied("Read", {"file_path": "../../../outside.txt"})
    assert denied("Bash", {"command": f"cat {tmp_path / 'outside.txt'}"})
