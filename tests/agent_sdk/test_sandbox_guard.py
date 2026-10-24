"""Tests for the sandbox escape guard and oversize-output spilling.

Covers ``_screen_text_for_sandbox_escape`` (used in-process by
``run_python``), ``_make_spilling_text_result`` (oversize tool output
spilled into the sandbox instead of dumped to ``~/.claude/...``), and
the self-contained ``VALIDATE_SANDBOX_SCRIPT`` Bash/file-path hook.
"""
# pylint: disable=protected-access
from __future__ import annotations

import json
import os
import subprocess
import sys

from predicators.agent_sdk.sandbox_setup import VALIDATE_SANDBOX_SCRIPT, \
    sandbox_settings
from predicators.agent_sdk.tools.results import _make_spilling_text_result
from predicators.agent_sdk.tools.sandbox_guard import \
    _screen_text_for_sandbox_escape

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
    real env), or the hidden-dynamics kill switch (exploits observed in
    run_20260819_183137 seed1 and run_20260819_183252 seed2).

    Python-only: the Bash hook does not screen these tokens (a shell
    command cannot reach in-process objects, and grepping old session
    logs for the words is legitimate), so they stay out of ``_BLOCK``.
    """
    sandbox = str(tmp_path)
    for text in [
            "print(s.privileged)",
            "e = sim._ctx.env",
            "getattr(sim, '_ctx').env.cure_threshold",
            "e._skip_domain_specific_dynamics = False",
    ]:
        assert _screen_text_for_sandbox_escape(text, sandbox) is not None, text
    # Word boundaries: harness-styled identifiers that merely end in
    # ``ctx`` (or mention privilege in prose) are untouched.
    for text in [
            "budget_ctx = 3",
            "print('privilege separation')",
            "ctx = make_ctx()",
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
