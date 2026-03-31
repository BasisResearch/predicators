"""Shared prompt templates and sandbox scaffolding for agent session managers.

Both ``LocalSandboxSessionManager`` and ``DockerSessionManager`` use
these constants and builder functions so that prompt text stays in sync.
"""
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict

from predicators.agent_sdk.tools import BUILTIN_TOOLS

logger = logging.getLogger(__name__)

_MAX_PARAM_LEN = 120


def truncate(value: Any, max_len: int = _MAX_PARAM_LEN) -> str:
    """Return a short string repr of *value*, truncating if needed."""
    s = repr(value)
    if len(s) > max_len:
        return s[:max_len] + "..."
    return s


def find_repo_root() -> Path:
    """Return the repository root by locating ``setup.py`` upward."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "setup.py").exists():
            return parent
    raise RuntimeError(
        "Could not find predicators repo root: no setup.py found in any "
        f"parent of {__file__}")


# ---------------------------------------------------------------------------
# Hook script that blocks Read/Write/Edit/Glob/Grep outside the sandbox.
# Python imports (via the PYTHONPATH mount) are NOT affected by this hook
# since they go through the Python interpreter, not Claude's built-in tools.
# ---------------------------------------------------------------------------

VALIDATE_SANDBOX_SCRIPT = """\
#!/usr/bin/env python3
import json
import os
import sys

data = json.load(sys.stdin)
tool_name = data.get("tool_name", "")
tool_input = data.get("tool_input", {})

# Determine the file/directory path based on tool type.
if tool_name in ("Read", "Write", "Edit"):
    file_path = tool_input.get("file_path", "")
elif tool_name in ("Glob", "Grep"):
    file_path = tool_input.get("path", "")
else:
    sys.exit(0)

if not file_path:
    # No path specified — defaults to cwd (sandbox), allow.
    sys.exit(0)

sandbox = os.path.realpath(os.getcwd())
resolved = os.path.realpath(file_path)

if resolved == sandbox or resolved.startswith(sandbox + os.sep):
    sys.exit(0)

json.dump({
    "hookSpecificOutput": {
        "hookEventName": "PreToolUse",
        "permissionDecision": "deny",
        "permissionDecisionReason": (
            f"Blocked: {file_path} resolves outside the sandbox directory"
        ),
    }
}, sys.stdout)
"""

SANDBOX_SETTINGS: Dict[str, Any] = {
    "hooks": {
        "PreToolUse": [{
            "matcher":
            "Read|Write|Edit|Glob|Grep",
            "hooks": [{
                "type": "command",
                "command": "python3 .claude/validate_sandbox.py",
            }],
        }]
    }
}

# ---------------------------------------------------------------------------
# Prompt template builders
# ---------------------------------------------------------------------------

_BUILTIN_TOOLS_STR = ", ".join(BUILTIN_TOOLS)


def build_claude_md(log_prefix: str = "query") -> str:
    """Build the CLAUDE.md content written into the sandbox directory.

    Args:
        log_prefix: Prefix for log filenames shown in examples
            (e.g. ``"local_sandbox_query"`` or ``"docker_query"``).
    """
    return f"""\
# Predicators Agent Sandbox

## Working Directory
Your working directory is the sandbox. All files you create MUST stay here.
Always use relative paths (e.g., `./my_script.py`).

## Python
The Python interpreter is `python3`. The predicators package is available
for import in your scripts:

    python3 -c "from predicators.structs import State, Type; print('OK')"

You can write and run test scripts in the sandbox:

    python3 my_experiment.py

## Reference Files
Curated source files are available in ./reference/ for you to read.
Read these to understand the APIs before writing code.

## Session Logs
Your past session queries and tool results are in ./session_logs/. Use Glob and
Read to review your earlier attempts when debugging:

    Glob ./session_logs/*.md
    Read ./session_logs/{log_prefix}_001_*.md

## Scene Images
`test_option_plan` automatically saves scene images to ./test_images/
after each step. You can Read them to inspect the spatial state of
the environment.

## Proposed Code
All proposal code and option source code is saved to ./proposed_code/.
Proposals are numbered (e.g. `001_propose_options_Pick.py`); option source
saved via `inspect_options` uses the option name (e.g. `Pick.py`):

    Glob ./proposed_code/*.py
    Read ./proposed_code/001_propose_options_Pick.py

## Debugging Strategy
- **Use visualize_state liberally** — it's free (no physics, no failure
  modes). When stuck on a step, STOP testing and visualize the object at
  several candidate positions and orientations to find the right region
  before spending more test_option_plan calls.
- **Vary all parameters** — orientation and other non-position params
  affect both the outcome and whether the action succeeds.
- **Search coarse-to-fine** — spread initial attempts across the full
  parameter range. After 3 failures in a small neighborhood, jump to a
  different region.

## Rules
- Do NOT attempt to read or browse files outside the sandbox directory
- Do NOT modify files in ./reference/ — they are for reading only
- Write all your code, experiments, and tests in the sandbox
- Do NOT inspect predicators source code (e.g. via `inspect.getsource()`,
  `inspect.getfile()`, reading `.py` files from site-packages, or any other
  method). Use the MCP tools and reference files instead.
"""


def build_sandbox_system_prompt(
    env_description: str = "a local sandbox environment",
    workspace_description: str = "the current directory",
    ref_path: str = "./reference/",
    log_prefix: str = "query",
) -> str:
    """Build the system prompt suffix appended for sandbox sessions.

    Args:
        env_description: Short description of the sandbox environment.
        workspace_description: How the workspace directory is described.
        ref_path: Path to reference files shown in examples.
        log_prefix: Prefix for log filenames shown in examples.
    """
    return f"""

## Sandbox Environment
You are running in {env_description}. You have the following
built-in tools available: {_BUILTIN_TOOLS_STR}.

Your workspace is {workspace_description}. All file operations (Read, Write,
Edit, Glob, Grep) are restricted to this directory.

### Writing and Running Code
You can write Python scripts and execute them with `python3`:
```
python3 my_script.py
```
The predicators package is importable in your scripts:
```python
from predicators.structs import State, Type, Object, Predicate
from predicators.structs import ParameterizedOption, Action
```

### Reference Files
Curated API reference files are in {ref_path}.
Read these files to understand the system APIs before writing code.

### Session Logs
Your past queries and tool results are saved in ./session_logs/ as markdown
files. Use Glob and Read to review your previous attempts:
```
Glob ./session_logs/*.md
Read ./session_logs/{log_prefix}_001_*.md
```

### Scene Images
`test_option_plan` automatically saves scene images to ./test_images/
after each plan step for later review.

### Proposed Code
All proposal code and option source code is saved to ./proposed_code/.
Proposals are numbered (e.g. `001_propose_options_Pick.py`); option source
saved via `inspect_options` uses the option name (e.g. `Pick.py`):
```
Glob ./proposed_code/*.py
Read ./proposed_code/001_propose_options_Pick.py
```

### Rules
- Do NOT try to read or browse files outside the sandbox directory
- Do NOT modify files in ./reference/
- Do NOT inspect predicators source code via `inspect.getsource()`,
  `inspect.getfile()`, or by reading `.py` files from site-packages.
  Use MCP tools and reference files instead.
"""


# ---------------------------------------------------------------------------
# Shared sandbox directory setup
# ---------------------------------------------------------------------------


def setup_sandbox_directory(
    sandbox_dir: str,
    repo_root: str,
    extra_reference_files: Dict[str, str],
    claude_md_content: str,
    system_prompt: str,
    log_dir: str,
    seed_scratchpad: bool = True,
) -> None:
    """Create and populate a sandbox directory for the agent.

    Sets up:
    - ``reference/`` with curated files copied from the host repo
    - ``CLAUDE.md`` with agent instructions
    - ``.claude/settings.json`` with PreToolUse hooks
    - ``.claude/validate_sandbox.py`` hook script
    - ``.git/`` marker so Claude CLI treats the sandbox as project root
    - ``session_logs/``, ``test_images/``, ``proposed_code/`` subdirectories
    - ``full_system_prompt.md`` in *log_dir* for easy inspection

    Args:
        sandbox_dir: Absolute path to the sandbox directory.
        repo_root: Absolute path to the predicators repository root.
        extra_reference_files: Mapping of destination paths (relative to
            ``sandbox/reference/``) to source paths (relative to repo root).
        claude_md_content: Content for the ``CLAUDE.md`` file.
        system_prompt: Full system prompt to log for inspection.
        log_dir: Directory for host-visible logs.
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
        import subprocess  # pylint: disable=import-outside-toplevel
        subprocess.run(["git", "init", "-q"], cwd=str(sandbox), check=True)
        need_initial_commit = True

    # 3. .claude/settings.json with PreToolUse hooks
    claude_dir = sandbox / ".claude"
    claude_dir.mkdir(exist_ok=True)
    (claude_dir / "settings.json"
     ).write_text(json.dumps(SANDBOX_SETTINGS, indent=2) + "\n")

    # 4. validate_sandbox.py hook script
    (claude_dir / "validate_sandbox.py").write_text(VALIDATE_SANDBOX_SCRIPT)

    # 5. CLAUDE.md
    (sandbox / "CLAUDE.md").write_text(claude_md_content)

    # 6. Create subdirectories and seed files
    for subdir in ("session_logs", "test_images", "proposed_code"):
        (sandbox / subdir).mkdir(exist_ok=True)
    # Seed empty scratchpad if enabled
    if seed_scratchpad:
        notes_path = sandbox / "notes.md"
        if not notes_path.exists():
            notes_path.write_text("")

    # 7. Log full system prompt to main log dir for easy inspection
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, "full_system_prompt.md"),
              "w",
              encoding="utf-8") as f:
        f.write(system_prompt)

    # 8. Initial commit so files are git-tracked.
    #    Claude Code indexes committed files at session startup; any file
    #    not committed before the session starts is invisible to Glob.
    if need_initial_commit:
        import subprocess  # pylint: disable=import-outside-toplevel
        subprocess.run(["git", "add", "-A"], cwd=str(sandbox), check=True)
        subprocess.run(
            [
                "git", "commit", "-q", "-m", "sandbox init", "--author",
                "sandbox <sandbox@local>"
            ],
            cwd=str(sandbox),
            check=True,
            env={
                **os.environ, "GIT_COMMITTER_NAME": "sandbox",
                "GIT_COMMITTER_EMAIL": "sandbox@local"
            },
        )

    logger.info(
        "Sandbox directory ready: %d reference files copied",
        sum(1 for d, s in registry.items() if (Path(repo_root) / s).exists()),
    )
