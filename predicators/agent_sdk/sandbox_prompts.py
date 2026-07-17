"""Shared prompt templates and sandbox scaffolding for agent session managers.

Both ``LocalSandboxSessionManager`` and ``DockerSessionManager`` use
these constants and builder functions so that prompt text stays in sync.
"""
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

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

# NOTE: raw string so the embedded regex backslashes (\\s, \\.) survive
# verbatim. The screening logic for Bash mirrors
# ``tools._screen_text_for_sandbox_escape`` (which guards ``run_python``);
# keep the two in sync.
VALIDATE_SANDBOX_SCRIPT = r'''#!/usr/bin/env python3
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

SYSTEM_ROOTS = (
    "/Users", "/home", "/root", "/etc", "/usr", "/opt", "/var", "/private",
    "/tmp", "/bin", "/sbin", "/lib", "/sys", "/proc", "/dev", "/mnt", "/srv",
)
INTROSPECTION = ("getsource", "inspect.getfile", "site-packages")
PATH_RE = re.compile(r"""(?:^|(?<=[\s'"`(=]))((?:/|\.\.)[^\s'"`)<>|;:,]*)""")

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

# ---------------------------------------------------------------------------
# Prompt template builders
# ---------------------------------------------------------------------------

_BUILTIN_TOOLS_STR = ", ".join(BUILTIN_TOOLS)

_CLAUDE_MD_HEADER = """\
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
Your past session queries and tool results are in ./session_logs/. Files are
named `<NNN>_<kind>_<timestamp>.md` where `<NNN>` is a run-wide counter and
`<kind>` is the query phase (e.g. `learn`, `test`, `explore`). The counter
comes first so alphabetical sort matches chronological order. Use Glob and
Read to review your earlier attempts when debugging:

    Glob ./session_logs/*.md
    Read ./session_logs/001_learn_*.md

## Scene Images
`evaluate_option_plan` automatically saves scene images to ./test_images/
after each step. You can Read them to inspect the spatial state of
the environment.

## Proposed Code
All proposal code and option source code is saved to ./proposed_code/.
Proposals are numbered (e.g. `001_propose_options_Pick.py`); option source
saved via `inspect_options` uses the option name (e.g. `Pick.py`):

    Glob ./proposed_code/*.py
    Read ./proposed_code/001_propose_options_Pick.py
"""

_CLAUDE_MD_RULES = """\

## Rules
- Do NOT attempt to read or browse files outside the sandbox directory.
  This is enforced for the file tools AND for Bash and the Python
  execution tools (run_python / explore_python): commands or code
  containing absolute or `../` paths that leave the sandbox (or source
  introspection) are blocked. Use relative paths inside the sandbox.
- Do NOT modify files in ./reference/ — they are for reading only
- Write all your code, experiments, and tests in the sandbox
- Do NOT inspect predicators source code (e.g. via `inspect.getsource()`,
  `inspect.getfile()`, reading `.py` files from site-packages, or any other
  method). Use the MCP tools and reference files instead.
"""

_CLAUDE_MD_SOLVE_STRATEGY = """\

## Debugging Strategy
- **Visualize liberally** — {visualize_hint} It's free (no physics, no
  failure modes). When stuck on a step, STOP testing and visualize the
  object at several candidate positions and orientations to find the
  right region before spending more evaluate_option_plan calls.
- **Vary all parameters** — orientation and other non-position params
  affect both the outcome and whether the action succeeds.
- **Search coarse-to-fine** — spread initial attempts across the full
  parameter range. After 3 failures in a small neighborhood, jump to a
  different region.
"""

# The solve strategy's visualization pointer depends on which tool the
# session actually has (see agent_planner_explore_python_keep_replaced
# _tools): explore_python subsumes visualize_state when it replaces it.
_VISUALIZE_HINT_TOOL = "use visualize_state."
_VISUALIZE_HINT_PROBE = ("use explore_python (`sim.reset(mods={...})`, "
                         "then `sim.render(...)`).")
_VISUALIZE_HINT_GENERIC = ("render candidate layouts with whatever "
                           "visualization your tools provide.")

_CLAUDE_MD_SYNTHESIS_STRATEGY = """\

## Model-Learning Strategy

Trajectory numbers are evidence, not ground truth. Two states with nearly
identical recorded coordinates can be geometrically very different — an
object's recorded pose origin often does not coincide with the part that
actually drives the rule (a body center vs. an outlet on its side, a
joint base vs. an end-effector tip, a container origin vs. its opening,
a switch housing vs. its handle). Before encoding any geometric
threshold, render the scene and check what's actually where.

**Threshold-fitting protocol** — follow this whenever a predicate or rule
condition compares a recorded feature against a learned cutoff:

1. Bucket trajectory steps by whether the downstream effect actually
   occurred (the rule-relevant feature advanced, the goal-relevant
   quantity changed, etc.). Compute your candidate quantity at each step.
2. Inspect the two buckets' value ranges. They must separate by a clear
   margin. If they overlap, or the gap is narrower than roughly 5% of
   the value range, STOP — a knife-edge separator is a symptom, not a
   fit, and a threshold flush against the data boundary is rejected.
   The candidate quantity is measuring against the wrong reference
   point; do not widen the threshold to absorb the gap.
3. For any two-body geometric gate, default to a learned anchor offset
   in the fixture's LOCAL frame, rotated into the world frame by the
   fixture's `rot` (origin + R(rot) @ (local_dx, local_dy)), with
   local_dx/local_dy declared as ParamSpecs and shared between the rule
   and its gating predicate — not a raw origin-distance threshold. To
   find the offset, call `visualize_state` at one representative state
   from each bucket and use `annotate_scene` to overlay, on one render,
   the recorded object origin and the positions where the effect did
   vs. did not fire. The gap between the origin and the effect-firing
   cluster is the offset.
4. Re-derive the candidate quantity using the anchored reference and
   refit. Only commit once the buckets separate by a comfortable margin
   (well past the 5% knife-edge). If the fit drives local_dx/local_dy to
   ~0, the origin was the functional point after all — fine, keep them.

**Other times to render the scene:**
- A new predicate is proposed: render a state where it should be true
  and one where it should be false to sanity-check the definition.
- A predicate's classifier looks right numerically but downstream signal
  (refinement success, residual reduction, plan completion) doesn't
  follow — the predicate is firing in the wrong places.
- You're choosing between candidate reference points (body center vs.
  contact surface, frame origin vs. tool tip, etc.).

`visualize_state` and `annotate_scene` are free (no physics, no failure
modes). Reach for them before, not after, you commit a numeric fit.
"""


def build_claude_md(phase: Optional[str] = None) -> str:
    """Build the CLAUDE.md content written into the sandbox directory.

    Args:
        phase: ``"synthesis"`` selects the model-learning strategy block;
            anything else (including ``None`` and ``"solve"``) selects the
            solve-time debugging block. The choice is reflected in the file
            written into the sandbox so the agent reads phase-appropriate
            guidance every turn.
    """
    # pylint: disable-next=import-outside-toplevel
    from predicators.settings import CFG
    if phase == "synthesis":
        strategy = _CLAUDE_MD_SYNTHESIS_STRATEGY
    else:
        from predicators.agent_sdk.tools import explore_python_replaces_tools
        if explore_python_replaces_tools():
            hint = _VISUALIZE_HINT_PROBE
        elif CFG.agent_planner_use_visualize_state:
            hint = _VISUALIZE_HINT_TOOL
        else:
            hint = _VISUALIZE_HINT_GENERIC
        strategy = _CLAUDE_MD_SOLVE_STRATEGY.format(visualize_hint=hint)
    return _CLAUDE_MD_HEADER + strategy + _CLAUDE_MD_RULES


def build_sandbox_system_prompt(
    env_description: str = "a local sandbox environment",
    workspace_description: str = "the current directory",
    ref_path: str = "./reference/",
) -> str:
    """Build the system prompt suffix appended for sandbox sessions.

    Args:
        env_description: Short description of the sandbox environment.
        workspace_description: How the workspace directory is described.
        ref_path: Path to reference files shown in examples.
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
files named `<NNN>_<kind>_<timestamp>.md` (e.g. `001_learn_...md`,
`002_test_...md`). The counter comes first so alphabetical sort matches
chronological order. Use Glob and Read to review previous attempts:
```
Glob ./session_logs/*.md
Read ./session_logs/001_learn_*.md
```

### Scene Images
`evaluate_option_plan` automatically saves scene images to ./test_images/
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
    phase: Optional[str] = None,
) -> None:
    """Create and populate a sandbox directory for the agent.

    Sets up:
    - ``reference/`` with curated files copied from the host repo
    - ``CLAUDE.md`` with agent instructions
    - ``.claude/settings.json`` with PreToolUse hooks
    - ``.claude/validate_sandbox.py`` hook script
    - ``.git/`` marker so Claude CLI treats the sandbox as project root
    - ``session_logs/``, ``test_images/``, ``proposed_code/`` subdirectories
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

    # 7. Log full system prompt to main log dir for easy inspection.
    #    Suffix with the phase tag when provided so solve and synthesis
    #    prompts don't overwrite each other across phase switches.
    os.makedirs(log_dir, exist_ok=True)
    prompt_filename = ("full_system_prompt.md"
                       if not phase else f"full_system_prompt_{phase}.md")
    with open(os.path.join(log_dir, prompt_filename), "w",
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
