"""Sandbox directory scaffolding shared by the session managers.

Creates and populates the per-run sandbox (reference files, CLAUDE.md,
PreToolUse hook, git repo for Glob visibility) and owns the embedded
hook script. Prompt *text* lives in :mod:`sandbox_prompts`; this module
is filesystem and git.
"""
import json
import logging
import os
import pickle
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from predicators.agent_sdk.tools.sandbox_guard import \
    SANDBOX_HIDDEN_MODULES_PATTERN, SANDBOX_INTROSPECTION, \
    SANDBOX_PYTHON_BYPASS_PATTERN, SANDBOX_SYSTEM_ROOTS

logger = logging.getLogger(__name__)

# Timeout for the git calls that make sandbox files Glob-visible.
GIT_TIMEOUT_S = 5

# ---------------------------------------------------------------------------
# Hook script that blocks Read/Write/Edit/Glob/Grep outside the sandbox.
# Python imports (via the PYTHONPATH mount) are NOT affected by this hook
# since they go through the Python interpreter, not Claude's built-in tools.
# ---------------------------------------------------------------------------

# NOTE: raw string so the embedded regex backslashes (\\s, \\.) survive
# verbatim. The Bash screening mirrors ``tools._screen_text_for_sandbox
# _escape`` (which guards ``run_python``); the shared constants are
# injected below so the two guards cannot drift on WHAT they block, only
# on how they anchor it (Python source vs. shell command strings).
_VALIDATE_SANDBOX_TEMPLATE = r'''#!/usr/bin/env python3
"""Sandbox PreToolUse guard.

Blocks built-in file tools (Read/Write/Edit/Glob/Grep) whose target path
resolves outside the sandbox, and heuristically screens Bash commands for
out-of-sandbox reads / predicators-source introspection. Best effort: a
determined script can still escape (env vars, subprocess, computed paths);
OS-level isolation is the hard boundary. Kept dependency-free so it stays
cheap to run on every tool call.
"""
import json
import os
import re
import sys

SYSTEM_ROOTS = __SYSTEM_ROOTS__
INTROSPECTION = __INTROSPECTION__
# Interpreter forms that would start a python without the sandbox's
# sitecustomize guard (see sandbox_setup.write_pyguard).
PYTHON_BYPASS_RE = re.compile(__PYTHON_BYPASS_PATTERN__)
PATH_RE = re.compile(r"""(?:^|(?<=[\s'"`(=]))((?:/|\.\.)[^\s'"`)<>|;:,]*)""")
# Hidden-implementation imports (e.g. inside `python -c`); the public
# authoring surface (predicators.structs / predicators.utils) stays
# importable. The module alternation is injected from tools.py.
HIDDEN_IMPORT_RE = re.compile(
    r"(?:^|[\s;])(?:from|import)\s+" + __HIDDEN_MODULES_PATTERN__)

data = json.load(sys.stdin)
tool_name = data.get("tool_name", "")
tool_input = data.get("tool_input", {})
sandbox = os.path.realpath(os.getcwd())


def within(path):
    resolved = os.path.realpath(
        path if os.path.isabs(path) else os.path.join(sandbox, path))
    return resolved == sandbox or resolved.startswith(sandbox + os.sep)


def deny(reason):
    json.dump({
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        }
    }, sys.stdout)
    sys.exit(0)


# File-path tools: validate the single target path (empty -> cwd, allow).
if tool_name in ("Read", "Write", "Edit", "Glob", "Grep"):
    key = "file_path" if tool_name in ("Read", "Write", "Edit") else "path"
    file_path = tool_input.get(key, "")
    if file_path and not within(file_path):
        deny("Blocked: " + file_path +
             " resolves outside the sandbox directory")
    sys.exit(0)

# Bash: heuristically screen the command string for escapes.
if tool_name == "Bash":
    command = tool_input.get("command", "")
    for needle in INTROSPECTION:
        if needle in command:
            deny("Blocked: '" + needle + "' may read predicators source "
                 "outside the sandbox; use ./reference/ and the MCP tools")
    if HIDDEN_IMPORT_RE.search(command):
        deny("Blocked: importing predicators env/ground-truth modules "
             "would expose implementation the sandbox hides; use "
             "./reference/ and the MCP tools")
    if PYTHON_BYPASS_RE.search(command):
        deny("Blocked: PYTHONPATH edits, env -i and the -S/-I/-E "
             "interpreter flags would start a python without the "
             "sandbox guard; run python3 plainly")
    for match in PATH_RE.finditer(command):
        token = match.group(1)
        if within(token):
            continue
        if token.startswith("/") and not any(
                token == r or token.startswith(r + "/")
                for r in SYSTEM_ROOTS):
            # Absolute but not a real filesystem path (printed data) — skip.
            continue
        deny("Blocked: path '" + token +
             "' in the command resolves outside the sandbox directory")
    sys.exit(0)

# Anything else: allow.
sys.exit(0)
'''

VALIDATE_SANDBOX_SCRIPT = (_VALIDATE_SANDBOX_TEMPLATE.replace(
    "__SYSTEM_ROOTS__", repr(SANDBOX_SYSTEM_ROOTS)).replace(
        "__INTROSPECTION__", repr(SANDBOX_INTROSPECTION)).replace(
            "__HIDDEN_MODULES_PATTERN__",
            repr(SANDBOX_HIDDEN_MODULES_PATTERN)).replace(
                "__PYTHON_BYPASS_PATTERN__",
                repr(SANDBOX_PYTHON_BYPASS_PATTERN)))

# ---------------------------------------------------------------------------
# Python-level guard: a ``sitecustomize`` module that every python the
# agent starts from the sandbox loads through PYTHONPATH (the session's
# env, see ``pyguard_env``). Where the Bash hook screens command strings,
# this vetoes the interpreter's own import and open events, so a script
# cannot reach the hidden predicators modules, their source, or the
# harness's run artifacts through computed paths or ``importlib``. Still
# best effort: OS-level isolation (the docker sandbox) is the hard line.
# ---------------------------------------------------------------------------

PYGUARD_DIRNAME = os.path.join(".claude", "pyguard")

# Repo-relative directories a sandboxed python may never open: the hidden
# implementation, and where runs, checkpoints and datasets live.
HIDDEN_REPO_DIRS = (
    os.path.join("predicators", "envs"),
    os.path.join("predicators", "ground_truth_models"),
    "logs",
    "saved_approaches",
    "saved_datasets",
    "scripts",
    "tests",
    "docs",
)

_PYGUARD_TEMPLATE = r'''
"""Sandbox python guard, loaded as ``sitecustomize`` via PYTHONPATH.

Vetoes imports of the hidden predicators modules, reads of their source
and of the harness's run artifacts, reads of any predicators source
outside the import system, and child interpreters started without this
guard. Raises ImportError / PermissionError at the offending call.
"""
import importlib.util
import os
import re
import sys

HIDDEN_MODULE_RE = re.compile(__HIDDEN_MODULES_PATTERN__)
REPO_ROOT = __REPO_ROOT__
HIDDEN_DIRS = __HIDDEN_DIRS__
SANDBOX_DIR = __SANDBOX_DIR__
GUARD_DIR = __GUARD_DIR__
PYTHON_BYPASS_RE = re.compile(__PYTHON_BYPASS_PATTERN__)


def _root_spellings():
    # The repo can be reachable under several spellings (bind mounts,
    # symlinks realpath cannot see through): the one the harness ran
    # from, and the one the installed package resolves to here.
    roots = {REPO_ROOT}
    try:
        spec = importlib.util.find_spec("predicators")
        locations = list(spec.submodule_search_locations or []) if spec \
            else []
    except Exception:  # pylint: disable=broad-except
        locations = []
    for loc in locations:
        roots.add(os.path.dirname(os.path.realpath(loc)))
    return sorted(roots)


def _under(path, prefix):
    return path == prefix or path.startswith(prefix + os.sep)


def _spellings(path):
    out = {path}
    for a in ROOTS:
        if _under(path, a):
            for b in ROOTS:
                out.add(b + path[len(a):])
    return sorted(out)


ROOTS = _root_spellings()
SANDBOX_PREFIXES = _spellings(SANDBOX_DIR)
DENY_PREFIXES = [
    p for d in HIDDEN_DIRS for p in _spellings(os.path.join(REPO_ROOT, d))
]
SOURCE_PREFIXES = _spellings(os.path.join(REPO_ROOT, "predicators"))


def _resolve(path):
    if isinstance(path, bytes):
        path = os.fsdecode(path)
    elif not isinstance(path, str):
        path = os.fspath(path)
    return os.path.realpath(path)


def _classify(path):
    # 'sandbox', 'deny', 'source' or None; the sandbox (which lives under
    # the repo's logs/ tree) wins over the directories it sits in.
    for group, label in ((SANDBOX_PREFIXES, "sandbox"),
                         (DENY_PREFIXES, "deny"),
                         (SOURCE_PREFIXES, "source")):
        for prefix in group:
            if _under(path, prefix):
                return label
    return None


def _check_read(raw_path, importing=False):
    # Veto reads of the hidden directories, and of any predicators source
    # unless ``importing`` (the import machinery loading a public module).
    try:
        path = _resolve(raw_path)
    except (TypeError, ValueError):
        return
    kind = _classify(path)
    if kind == "deny":
        raise PermissionError("sandbox guard: " + path + " is outside the "
                              "sandbox and hidden from the agent")
    if kind == "source" and not importing:
        raise PermissionError(
            "sandbox guard: reading predicators source (" + path +
            ") is not allowed; use ./reference/ and the tools")


def _importing():
    # True only for importlib's get_code -> get_data read of a module
    # (``io.open_code`` raises the "open" event straight from get_data).
    # loader.get_source, linecache and inspect reach get_data by other
    # routes and stay refused, so source lines cached for warnings or
    # tracebacks never pass either.
    caller = sys._getframe(2)
    outer = caller.f_back
    return ("importlib._bootstrap_external" in caller.f_code.co_filename
            and caller.f_code.co_name == "get_data" and outer is not None
            and outer.f_code.co_name == "get_code")


def _guard(event, args):
    if event == "import":
        name = args[0]
        if isinstance(name, str) and HIDDEN_MODULE_RE.match(name):
            raise ImportError(
                "sandbox guard: importing " + name + " would expose the "
                "hidden environment implementation; use ./reference/ and "
                "the tools")
    elif event == "open":
        _check_read(args[0], _importing())
    elif event == "subprocess.Popen":
        argv, env = args[1], args[3]
        if isinstance(argv, (list, tuple)):
            argv = " ".join(os.fsdecode(a) if isinstance(a, bytes)
                            else str(a) for a in argv)
        if isinstance(argv, str) and PYTHON_BYPASS_RE.search(argv):
            raise PermissionError(
                "sandbox guard: child interpreters must keep the sandbox "
                "guard (no -S/-I/-E, env -i or PYTHONPATH edits)")
        if env is not None:
            child_path = env.get("PYTHONPATH", "") if hasattr(
                env, "get") else ""
            if GUARD_DIR not in str(child_path).split(os.pathsep):
                raise PermissionError(
                    "sandbox guard: child processes must inherit "
                    "PYTHONPATH (it carries the sandbox guard)")


sys.addaudithook(_guard)
'''


def pyguard_dir(sandbox_dir: str) -> str:
    """Where the sandbox's ``sitecustomize`` guard lives."""
    return os.path.join(os.path.realpath(sandbox_dir), PYGUARD_DIRNAME)


def pyguard_env(sandbox_dir: str) -> Dict[str, str]:
    """Environment overrides that load the guard into every python the agent
    starts: the guard directory first on ``PYTHONPATH``."""
    guard = pyguard_dir(sandbox_dir)
    existing = os.environ.get("PYTHONPATH", "")
    parts = [guard] + [p for p in existing.split(os.pathsep) if p]
    return {"PYTHONPATH": os.pathsep.join(parts)}


def write_pyguard(sandbox_dir: str, repo_root: str) -> str:
    """Write the ``sitecustomize`` guard for ``sandbox_dir``; returns its
    directory (the value ``pyguard_env`` puts on PYTHONPATH)."""
    root = os.path.realpath(repo_root)
    guard = pyguard_dir(sandbox_dir)
    os.makedirs(guard, exist_ok=True)
    substitutions = {
        "__HIDDEN_MODULES_PATTERN__": SANDBOX_HIDDEN_MODULES_PATTERN,
        "__REPO_ROOT__": root,
        "__HIDDEN_DIRS__": list(HIDDEN_REPO_DIRS),
        "__SANDBOX_DIR__": os.path.realpath(sandbox_dir),
        "__GUARD_DIR__": guard,
        "__PYTHON_BYPASS_PATTERN__": SANDBOX_PYTHON_BYPASS_PATTERN,
    }
    script = _PYGUARD_TEMPLATE
    for marker, value in substitutions.items():
        script = script.replace(marker, repr(value))
    with open(os.path.join(guard, "sitecustomize.py"), "w",
              encoding="utf-8") as f:
        f.write(script)
    return guard


SANDBOX_SETTINGS: Dict[str, Any] = {
    "hooks": {
        "PreToolUse": [{
            "matcher":
            "Read|Write|Edit|Glob|Grep|Bash",
            "hooks": [{
                "type": "command",
                "command": "python3 .claude/validate_sandbox.py",
            }],
        }]
    }
}


def find_repo_root() -> Path:
    """Return the repository root by locating ``setup.py`` upward."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "setup.py").exists():
            return parent
    raise RuntimeError(
        "Could not find predicators repo root: no setup.py found in any "
        f"parent of {__file__}")


def git_commit_all(sandbox_dir: str,
                   message: str,
                   paths: Optional[list] = None,
                   check: bool = False) -> None:
    """Stage ``paths`` (default: everything) and commit as the sandbox user.

    Claude Code indexes git-tracked files at session startup, so files
    must be committed before the session starts to be Glob-visible.
    """
    add_target = paths if paths is not None else ["-A"]
    subprocess.run(["git", "add", *add_target],
                   cwd=sandbox_dir,
                   capture_output=True,
                   timeout=GIT_TIMEOUT_S,
                   check=check)
    subprocess.run(
        [
            "git", "commit", "-q", "-m", message, "--author",
            "sandbox <sandbox@local>"
        ],
        cwd=sandbox_dir,
        capture_output=True,
        timeout=GIT_TIMEOUT_S,
        check=check,
        env={
            **os.environ, "GIT_COMMITTER_NAME": "sandbox",
            "GIT_COMMITTER_EMAIL": "sandbox@local"
        },
    )


def setup_sandbox_directory(
    sandbox_dir: str,
    repo_root: str,
    extra_reference_files: Dict[str, str],
    claude_md_content: str,
    system_prompt: str,
    log_dir: str,
    seed_scratchpad: bool = True,
    phase: Optional[str] = None,
) -> None:
    """Create and populate a sandbox directory for the agent.

    Sets up:
    - ``reference/`` with curated files copied from the host repo
    - ``CLAUDE.md`` with agent instructions
    - ``.claude/settings.json`` with PreToolUse hooks
    - ``.claude/validate_sandbox.py`` hook script
    - ``.git/`` marker so Claude CLI treats the sandbox as project root
    - ``session_logs/``, ``test_images/`` subdirectories
    - ``full_system_prompt[_{phase}].md`` in *log_dir* for easy inspection

    Args:
        sandbox_dir: Absolute path to the sandbox directory.
        repo_root: Absolute path to the predicators repository root.
        extra_reference_files: Mapping of destination paths (relative to
            ``sandbox/reference/``) to source paths (relative to repo root).
        claude_md_content: Content for the ``CLAUDE.md`` file.
        system_prompt: Full system prompt to log for inspection.
        log_dir: Directory for host-visible logs.
        phase: Optional phase tag (e.g. ``"solve"``, ``"synthesis"``). When
            provided, the logged prompt is suffixed so solve and synthesis
            prompts don't overwrite each other across phase switches.
    """
    os.makedirs(sandbox_dir, exist_ok=True)
    sandbox = Path(sandbox_dir)
    logger.info("Setting up sandbox directory: %s", sandbox_dir)

    # 1. Copy reference files from host repo
    registry = dict(extra_reference_files)
    ref_dir = sandbox / "reference"
    for dest_rel, src_rel in registry.items():
        src = Path(repo_root) / src_rel
        dest = ref_dir / dest_rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if src.exists():
            shutil.copy2(str(src), str(dest))
        else:
            logger.warning("Reference file not found: %s", src)

    # 2. Real git repo so Claude CLI treats sandbox as project root
    #    (A plain .git file isn't recognised; Glob/Read resolve from the
    #    project root, so we need an actual repo here.)
    git_dir = sandbox / ".git"
    need_initial_commit = False
    if not git_dir.is_dir():
        if git_dir.exists():
            git_dir.unlink()  # remove old marker file
        subprocess.run(["git", "init", "-q"], cwd=str(sandbox), check=True)
        need_initial_commit = True

    # 3. .claude/settings.json with PreToolUse hooks
    #    Every write below is explicitly utf-8: these strings carry em dashes,
    #    and bare write_text() encodes with the locale's preferred encoding,
    #    so on a C/POSIX locale the whole sandbox setup dies with
    #    "'ascii' codec can't encode character '—'" -- and because setup
    #    runs once per query, every solve attempt fails identically.
    claude_dir = sandbox / ".claude"
    claude_dir.mkdir(exist_ok=True)
    (claude_dir / "settings.json").write_text(
        json.dumps(SANDBOX_SETTINGS, indent=2) + "\n", encoding="utf-8")

    # 4. validate_sandbox.py hook script, and the python-level guard
    #    (loaded through the session env, see pyguard_env).
    (claude_dir / "validate_sandbox.py").write_text(VALIDATE_SANDBOX_SCRIPT,
                                                    encoding="utf-8")
    write_pyguard(str(sandbox), repo_root)

    # 5. CLAUDE.md
    (sandbox / "CLAUDE.md").write_text(claude_md_content, encoding="utf-8")

    # 6. Create subdirectories and seed files
    for subdir in ("session_logs", "test_images"):
        (sandbox / subdir).mkdir(exist_ok=True)
    # Seed empty scratchpad if enabled
    if seed_scratchpad:
        notes_path = sandbox / "notes.md"
        if not notes_path.exists():
            notes_path.write_text("", encoding="utf-8")

    # 7. Log full system prompt to main log dir for easy inspection.
    #    Suffix with the phase tag when provided so solve and synthesis
    #    prompts don't overwrite each other across phase switches.
    os.makedirs(log_dir, exist_ok=True)
    prompt_filename = ("full_system_prompt.md"
                       if not phase else f"full_system_prompt_{phase}.md")
    with open(os.path.join(log_dir, prompt_filename), "w",
              encoding="utf-8") as f:
        f.write(system_prompt)

    # 8. Initial commit so files are git-tracked (Glob-visible).
    if need_initial_commit:
        git_commit_all(str(sandbox), "sandbox init", check=True)

    logger.info(
        "Sandbox directory ready: %d reference files copied",
        sum(1 for d, s in registry.items() if (Path(repo_root) / s).exists()),
    )


# ---------------------------------------------------------------------------
# Data on disk: the agent's own scripts (Bash ``python3``) read the same
# trajectories the prompts and probe namespace expose.
# ---------------------------------------------------------------------------

DATA_DIRNAME = "data"
TRAJECTORIES_FILENAME = "trajectories.pkl"


def trajectories_path(sandbox_dir: str) -> str:
    """Where the sandbox's pickled trajectory list lives."""
    return os.path.join(sandbox_dir, DATA_DIRNAME, TRAJECTORIES_FILENAME)


def _option_label(action: Any) -> Optional[str]:
    if not action.has_option():
        return None
    option = action.get_option()
    objects = ", ".join(o.name for o in option.objects)
    params = ", ".join(f"{float(v):.4g}" for v in option.params)
    return f"{option.name}({objects})[{params}]"


def sanitize_trajectory(trajectory: Any) -> Dict[str, Any]:
    """A plain record of ``trajectory`` that is safe to hand to the agent.

    States keep only their observable ``data`` (no ``privileged`` ground
    truth, no simulator bookkeeping). Actions become ``{"arr", "option"}``
    dicts: the array and the grounded option as a text label, since the
    live ``Action``/``_Option`` objects close over the harness and do not
    pickle.
    """
    # pylint: disable-next=import-outside-toplevel
    from predicators.structs import State

    # pylint: disable-next=protected-access
    task_idx = trajectory._train_task_idx
    return {
        "states": [State(dict(s.data)) for s in trajectory.states],
        "actions": [{
            "arr": a.arr,
            "option": _option_label(a)
        } for a in trajectory.actions],
        "is_demo":
        trajectory.is_demo,
        "train_task_idx":
        task_idx,
    }


def export_trajectories(sandbox_dir: str, trajectories: Sequence[Any]) -> bool:
    """Pickle sanitized copies of ``trajectories`` (see
    :func:`sanitize_trajectory`) into the sandbox's ``data/`` directory.

    Rewrites (and git-commits, for Glob visibility) only when the bytes
    changed. Returns whether the file was rewritten.
    """
    path = trajectories_path(sandbox_dir)
    payload = pickle.dumps([sanitize_trajectory(t) for t in trajectories],
                           protocol=pickle.HIGHEST_PROTOCOL)
    if os.path.isfile(path):
        with open(path, "rb") as f:
            if f.read() == payload:
                return False
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(payload)
    os.replace(tmp, path)
    if os.path.isdir(os.path.join(sandbox_dir, ".git")):
        git_commit_all(sandbox_dir, "refresh data", paths=[path])
    return True


# ---------------------------------------------------------------------------
# Test-phase isolation: the whole sandbox rolls back to a snapshot taken
# when the test phase began, so nothing written while solving test tasks
# (session logs, scene images, notes, scripts, plan files) is readable
# from later learning or from the next evaluation. The pre-test tree is
# committed, everything the phase added is archived outside the sandbox,
# then the tree is reset.
# ---------------------------------------------------------------------------


def _git(sandbox_dir: str, *args: str) -> str:
    out = subprocess.run(["git", *args],
                         cwd=sandbox_dir,
                         capture_output=True,
                         text=True,
                         timeout=GIT_TIMEOUT_S,
                         check=True)
    return out.stdout


def snapshot_sandbox(sandbox_dir: Optional[str]) -> Optional[str]:
    """Commit the sandbox as it stands and return the commit id, or None when
    there is no git sandbox to snapshot."""
    if not sandbox_dir or not os.path.isdir(os.path.join(sandbox_dir, ".git")):
        return None
    git_commit_all(sandbox_dir, "pre-test snapshot")
    return _git(sandbox_dir, "rev-parse", "HEAD").strip()


def rollback_sandbox(sandbox_dir: str, rev: str,
                     archive_dir: str) -> List[str]:
    """Archive every file added or changed since ``rev`` into ``archive_dir``
    (same relative layout), then restore the sandbox tree to ``rev``.

    Returns the archived relative paths.
    """
    changed = _git(sandbox_dir, "diff", "--name-only", "--diff-filter=ACMR",
                   rev).split("\n")
    untracked = _git(sandbox_dir, "ls-files", "--others").split("\n")
    archived: List[str] = []
    for rel in sorted({p for p in changed + untracked if p}):
        src = os.path.join(sandbox_dir, rel)
        if not os.path.isfile(src):
            continue
        dst = os.path.join(archive_dir, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        archived.append(rel)
    _git(sandbox_dir, "reset", "-q", "--hard", rev)
    _git(sandbox_dir, "clean", "-qfdx")
    return archived
