"""Docker-sandboxed agent session manager.

Runs ``ClaudeSDKClient`` inside a Docker container so that the agent's
built-in tools (Bash, Read, Write, Edit, Glob, Grep) all execute in an
isolated environment.  Custom predicator MCP tools are created in-process
inside the container via the same ``create_mcp_tools()`` code used on
the host.

The host predicators source tree is mounted read-only at
``/opt/predicators`` for Python imports (``PYTHONPATH``).  PreToolUse
hooks block the agent's built-in tools (Read, Write, Edit, Glob, Grep)
from accessing anything outside ``/sandbox/``, so the agent cannot
browse environment source code or ground truth models directly.  Curated
reference files are copied into ``/sandbox/reference/`` for the agent to
read.  The agent can write and run Python scripts in ``/sandbox/``, and
``from predicators.structs import State`` works via the mount.

Shared data (pickled context and results) passes through ``/data``.

Usage
-----
When ``CFG.agent_sdk_use_docker_sandbox`` is ``True``, the
``AgentSessionMixin`` creates a ``DockerSessionManager`` in place of the
normal ``AgentSessionManager``.  The interface is identical::

    manager = DockerSessionManager(...)
    responses = await manager.query("Solve this task...")
    await manager.close()

Build the image first::

    bash docker/build.sh
"""
import datetime
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import dill as pkl

from predicators.agent_sdk.tools import ToolContext
from predicators.settings import CFG

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Reference file registry
# ---------------------------------------------------------------------------
# Maps destination paths (relative to /sandbox/reference/) to source paths
# (relative to the repo root).  These files are copied fresh from the host
# at session start so they always reflect the latest code.
REFERENCE_FILE_REGISTRY: Dict[str, str] = {
    # # Core API
    # "core/structs.py": "predicators/structs.py",
    # "core/settings.py": "predicators/settings.py",
    # # Planning
    # "core/planning.py": "predicators/planning.py",
    # "core/planning_with_processes.py":
    #     "predicators/planning_with_processes.py",
    # # Agent SDK
    # "agent_sdk/tools.py": "predicators/agent_sdk/tools.py",
    # "agent_sdk/option_builder.py": "predicators/agent_sdk/option_builder.py",
    # # Skill factories
    # "skill_factories/base.py":
    #     "predicators/ground_truth_models/skill_factories/base.py",
    # "skill_factories/pick.py":
    #     "predicators/ground_truth_models/skill_factories/pick.py",
    # "skill_factories/move_to.py":
    #     "predicators/ground_truth_models/skill_factories/move_to.py",
    # "skill_factories/place.py":
    #     "predicators/ground_truth_models/skill_factories/place.py",
    # "skill_factories/push.py":
    #     "predicators/ground_truth_models/skill_factories/push.py",
    # "skill_factories/pour.py":
    #     "predicators/ground_truth_models/skill_factories/pour.py",
    # "skill_factories/wait.py":
    #     "predicators/ground_truth_models/skill_factories/wait.py",
    # # Examples
    # "examples/api_oo_state.py": "prompts/api_oo_state.py",
    # "examples/api_sym_predicate.py": "prompts/api_sym_predicate.py",
}

# ---------------------------------------------------------------------------
# Sandbox scaffolding (mirrors robocode's sandbox.py pattern)
# ---------------------------------------------------------------------------

# Hook script that blocks Read/Write/Edit/Glob/Grep outside /sandbox/.
# Python imports (via the PYTHONPATH mount) are NOT affected by this hook
# since they go through the Python interpreter, not Claude's built-in tools.
_VALIDATE_SANDBOX_SCRIPT = """\
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

_SANDBOX_SETTINGS: Dict[str, Any] = {
    "hooks": {
        "PreToolUse": [
            {
                "matcher": "Read|Write|Edit|Glob|Grep",
                "hooks": [
                    {
                        "type": "command",
                        "command": "python3 .claude/validate_sandbox.py",
                    }
                ],
            }
        ]
    }
}

_CLAUDE_MD_TEMPLATE = """\
# Predicators Agent Sandbox

## Working Directory
Your working directory is /sandbox/. All files you create MUST stay here.
Always use relative paths (e.g., `./my_script.py`, not `/sandbox/my_script.py`).

## Python
The Python interpreter is `python3`. The predicators package is available
for import in your scripts:

    python3 -c "from predicators.structs import State, Type; print('OK')"

You can write and run test scripts in the sandbox:

    python3 my_experiment.py

## Reference Files
Curated source files are available in ./reference/ for you to read:

    reference/core/             - structs.py, settings.py, planning.py,
                                  planning_with_processes.py
    reference/agent_sdk/        - tools.py (MCP tool definitions),
                                  option_builder.py (option construction)
    reference/skill_factories/  - base.py, pick.py, move_to.py, place.py,
                                  push.py, pour.py, wait.py
    reference/examples/         - api_oo_state.py, api_sym_predicate.py

Read these to understand the APIs before writing code.

## MCP Tools
You have custom predicator MCP tools available:
- inspect_* tools: inspect types, predicates, options, trajectories, tasks
- propose_* tools: propose new types, predicates, processes, options
- test_* tools: test predicates on states, test planning, test option plans
- planning tools: generate abstract and bilevel plans

Use these tools as your primary interface for interacting with the system.

## Rules
- Do NOT attempt to read or browse files outside /sandbox/
- Do NOT modify files in ./reference/ — they are for reading only
- Write all your code, experiments, and tests in /sandbox/
- Do NOT inspect predicators source code (e.g. via `inspect.getsource()`,
  `inspect.getfile()`, reading `.py` files from site-packages, or any other
  method). Use the MCP tools and reference files instead.
"""

# Instructions appended to the agent's system prompt when running in Docker.
_SANDBOX_SYSTEM_PROMPT = """\

## Sandbox Environment
You are running inside an isolated Docker sandbox. You have the following
built-in tools available: Bash, Read, Write, Edit, Glob, Grep.

Your workspace is /sandbox/. All file operations (Read, Write, Edit,
Glob, Grep) are restricted to this directory.

### Writing and Running Code
You can write Python scripts in /sandbox/ and execute them with `python3`:
```
python3 my_script.py
```
The predicators package is importable in your scripts:
```python
from predicators.structs import State, Type, Object, Predicate
from predicators.structs import ParameterizedOption, Action
```

### Reference Files
Curated API reference files are in /sandbox/reference/:
- reference/core/ — core data structures (structs.py), settings, planners
- reference/agent_sdk/ — MCP tool definitions, option builder helpers
- reference/skill_factories/ — reusable skill patterns (pick, place, etc.)
- reference/examples/ — example predicate and state API usage

Read these files to understand the system APIs before writing code.

### What to Use
- Use MCP tools (inspect_*, propose_*, test_*) as your primary interface
- Read reference files when you need to understand API details
- Write test scripts in /sandbox/ to validate your ideas experimentally

### What NOT to Do
- Do NOT try to read or browse files outside /sandbox/
- Do NOT modify files in /sandbox/reference/
- Do NOT inspect predicators source code via `inspect.getsource()`,
  `inspect.getfile()`, or by reading `.py` files from site-packages.
  Use MCP tools and reference files instead.
"""


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _find_repo_root() -> Path:
    """Return the repository root by locating ``setup.py`` upward."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "setup.py").exists():
            return parent
    raise RuntimeError(
        "Could not find predicators repo root: no setup.py found in any "
        f"parent of {__file__}")


def _get_claude_oauth_token() -> Optional[str]:
    """Extract the Claude Code OAuth access token from the macOS Keychain.

    Returns ``None`` on non-macOS platforms or when the token cannot be
    found.  On macOS, ``claude login`` stores credentials under the
    service name ``"Claude Code-credentials"``.
    """
    if sys.platform != "darwin":
        return None
    try:
        result = subprocess.run(
            ["security", "find-generic-password",
             "-s", "Claude Code-credentials", "-w"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        if result.returncode != 0:
            return None
        import json as _json
        creds = _json.loads(result.stdout.strip())
        return creds.get("claudeAiOauth", {}).get("accessToken")
    except (subprocess.SubprocessError, json.JSONDecodeError, KeyError):
        return None


class DockerSessionManager:
    """Runs ClaudeSDKClient inside Docker with built-in + custom MCP tools.

    Matches the ``AgentSessionManager`` interface so that all agent-based
    approaches work unchanged.  Each ``query()`` call:

    1. Serializes ``ToolContext`` + message to pickle in a temp directory.
    2. Runs ``docker run ...`` with the predicators source mounted at
       ``/opt/predicators:ro`` (for Python imports) and a curated sandbox
       at ``/sandbox`` (for agent file operations).
    3. Inside Docker, the runner script creates ``ClaudeSDKClient`` with
       both built-in tools AND custom MCP tools, queries the agent, and
       pickles back responses + mutated proposals.
    4. Host reads back the pickled results.

    PreToolUse hooks restrict the agent's built-in tools (Read, Write,
    Edit, Glob, Grep) to ``/sandbox/`` only.  Python imports via
    ``PYTHONPATH`` are unaffected.
    """

    def __init__(
        self,
        system_prompt: str,
        log_dir: str,
        model_name: str,
        tool_context: ToolContext,
        tool_names: Optional[List[str]] = None,
        image: str = "predicators-sandbox",
    ) -> None:
        # Append sandbox instructions to the system prompt
        self._system_prompt = system_prompt + _SANDBOX_SYSTEM_PROMPT
        self._log_dir = log_dir
        self._model_name = model_name
        self._tool_context = tool_context
        self._tool_names = tool_names
        self._image = image
        self._repo_root = str(_find_repo_root())

        self._total_cost_usd: float = 0.0
        self._total_turns: int = 0
        self._query_count: int = 0
        self._session_id: Optional[str] = None
        self._conversation_log: List[Dict[str, Any]] = []

        # Persistent sandbox directory (created lazily, cleaned up on close)
        self._sandbox_dir: Optional[str] = None

    # -- Properties matching AgentSessionManager interface --

    @property
    def session_id(self) -> Optional[str]:
        return self._session_id

    @session_id.setter
    def session_id(self, value: Optional[str]) -> None:
        self._session_id = value

    @property
    def tool_names(self) -> List[str]:
        """Return short tool names (without MCP prefix)."""
        from predicators.agent_sdk.tools import MCP_SERVER_NAME
        prefix = f"mcp__{MCP_SERVER_NAME}__"
        from predicators.agent_sdk.docker_agent_runner import BUILTIN_TOOLS
        names = list(BUILTIN_TOOLS)
        if self._tool_names:
            names += self._tool_names
        return [
            t[len(prefix):] if t.startswith(prefix) else t
            for t in names
        ]

    @property
    def conversation_log(self) -> List[Dict[str, Any]]:
        """Return the in-memory log of all query/response pairs."""
        return self._conversation_log

    # -- Sandbox setup --

    def _ensure_sandbox_dir(self) -> None:
        """Create and populate the sandbox directory if it doesn't exist.

        Sets up:
        - ``reference/`` with curated files from the host repo
        - ``CLAUDE.md`` with agent instructions
        - ``.claude/settings.json`` with PreToolUse hooks blocking
          Read/Write/Edit/Glob/Grep outside ``/sandbox/``
        - ``.claude/validate_sandbox.py`` hook script
        - ``.git/`` so Claude CLI treats ``/sandbox`` as project root
        """
        if self._sandbox_dir is not None:
            return

        self._sandbox_dir = os.path.abspath(
            os.path.join(self._log_dir, "sandbox"))
        os.makedirs(self._sandbox_dir, exist_ok=True)
        sandbox = Path(self._sandbox_dir)
        logger.info("Setting up sandbox directory: %s", self._sandbox_dir)

        # 1. Copy reference files from host repo
        registry = dict(REFERENCE_FILE_REGISTRY)
        extra = getattr(CFG, "agent_sdk_sandbox_extra_reference_files", {})
        if extra:
            registry.update(extra)

        ref_dir = sandbox / "reference"
        for dest_rel, src_rel in registry.items():
            src = Path(self._repo_root) / src_rel
            dest = ref_dir / dest_rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            if src.exists():
                shutil.copy2(str(src), str(dest))
            else:
                logger.warning("Reference file not found: %s", src)

        # 2. Create a minimal .git marker so Claude CLI treats /sandbox
        #    as a project root.  A plain file (gitdir marker) is enough
        #    and avoids VS Code detecting a nested git repo on the host.
        git_marker = sandbox / ".git"
        if not git_marker.exists():
            git_marker.write_text("# marker for Claude CLI project root\n")

        # 3. Create .claude/settings.json with PreToolUse hooks
        claude_dir = sandbox / ".claude"
        claude_dir.mkdir(exist_ok=True)
        (claude_dir / "settings.json").write_text(
            json.dumps(_SANDBOX_SETTINGS, indent=2) + "\n"
        )

        # 4. Create validate_sandbox.py hook script
        (claude_dir / "validate_sandbox.py").write_text(
            _VALIDATE_SANDBOX_SCRIPT
        )

        # 5. Write CLAUDE.md
        (sandbox / "CLAUDE.md").write_text(_CLAUDE_MD_TEMPLATE)

        logger.info(
            "Sandbox directory ready: %d reference files copied",
            sum(1 for d, s in registry.items()
                if (Path(self._repo_root) / s).exists()),
        )

    # -- Session lifecycle --

    async def start_session(self) -> None:
        """No-op: each query() is a fresh docker run."""
        pass

    async def query(self, message: str) -> List[Dict[str, Any]]:
        """Run the agent in Docker and return collected response messages.

        Returns the same ``List[Dict[str, Any]]`` format as
        ``AgentSessionManager.query()``.
        """
        self._query_count += 1

        # Ensure sandbox is set up (lazy init, persists across queries)
        self._ensure_sandbox_dir()

        # 1. Create temp directory for data exchange
        tmp_dir = tempfile.mkdtemp(prefix="pred-docker-")
        input_path = os.path.join(tmp_dir, "query_input.pkl")
        output_path = os.path.join(tmp_dir, "query_output.pkl")
        incremental_log_path = os.path.join(tmp_dir, "query_log.json")

        try:
            # 2. Pickle QueryInput
            query_input = {
                "tool_context": self._tool_context,
                "message": message,
                "system_prompt": self._system_prompt,
                "model_name": self._model_name,
                "max_turns": CFG.agent_sdk_max_agent_turns_per_iteration,
                "tool_names": self._tool_names,
                "cfg_snapshot": dict(CFG.__dict__),
                "log_path": "/data/query_log.json",
            }
            with open(input_path, "wb") as f:
                pkl.dump(query_input, f)

            logger.info(
                "Docker query %d: message length=%d, model=%s",
                self._query_count, len(message), self._model_name,
            )

            # 3. Build docker run command
            container_name = f"pred-sandbox-{uuid.uuid4().hex[:8]}"
            docker_cmd = self._build_docker_command(
                container_name, tmp_dir)

            # 4. Run Docker container
            logger.info(
                "Starting Docker sandbox: container=%s image=%s",
                container_name, self._image,
            )
            env = self._build_env()

            proc = subprocess.Popen(
                docker_cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            # Stream stderr in real-time so tool calls / agent messages
            # appear on the host terminal as they happen.
            stderr_lines: List[str] = []
            try:
                timeout_sec = CFG.agent_sdk_agent_timeout + 120
                import threading

                def _stream_stderr() -> None:
                    assert proc.stderr is not None
                    for line in proc.stderr:
                        line = line.rstrip("\n")
                        stderr_lines.append(line)
                        logger.info("%s", line)

                stderr_thread = threading.Thread(
                    target=_stream_stderr, daemon=True)
                stderr_thread.start()

                # Wait for stdout (captured for error reporting)
                stdout_data = proc.stdout.read() if proc.stdout else ""
                proc.wait(timeout=timeout_sec)
                stderr_thread.join(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
                logger.error("Docker container timed out after %ds",
                             timeout_sec)
                stdout_data = ""

            if proc.returncode != 0:
                logger.error(
                    "Docker container exited with code %d.\nstdout: %s\n"
                    "stderr (last 2000 chars): %s",
                    proc.returncode,
                    stdout_data[-2000:] if stdout_data else "(empty)",
                    "\n".join(stderr_lines)[-2000:]
                    if stderr_lines else "(empty)",
                )
            else:
                logger.info("Docker container exited successfully.")

            # 5. Load query output
            if os.path.exists(output_path):
                with open(output_path, "rb") as f:
                    query_output = pkl.load(f)

                responses = query_output.get("responses", [])
                proposals = query_output.get("iteration_proposals")

                # 6. Merge proposals back into host ToolContext
                if proposals is not None:
                    self._tool_context.iteration_proposals = proposals

                # Track costs/turns
                for resp in responses:
                    if resp.get("type") == "result":
                        cost = resp.get("total_cost_usd")
                        turns = resp.get("num_turns")
                        if cost is not None:
                            self._total_cost_usd += cost
                        if turns is not None:
                            self._total_turns += turns
            else:
                logger.error(
                    "No output pickle found at %s. Container may have "
                    "crashed.", output_path)
                responses = [{
                    "type": "error",
                    "error": (
                        f"Docker container failed (exit code "
                        f"{proc.returncode}). "
                        f"stderr: {''.join(stderr_lines[-20:])}"
                    ),
                }]

            # 7. Save query log — copy the incremental log written by
            # the Docker agent runner (already flushed per-message).
            # Fall back to writing a batch log if no incremental file.
            if os.path.exists(incremental_log_path) and self._log_dir:
                os.makedirs(self._log_dir, exist_ok=True)
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                dest = os.path.join(
                    self._log_dir,
                    f"docker_query_{self._query_count:03d}_"
                    f"{timestamp}.json")
                # Enrich the container's log with host metadata
                try:
                    with open(incremental_log_path) as lf:
                        log_data = json.load(lf)
                    log_data.update({
                        "query_number": self._query_count,
                        "timestamp": timestamp,
                        "query": message,
                        "session_id": self._session_id,
                        "docker_image": self._image,
                    })
                    with open(dest, "w") as lf:
                        json.dump(log_data, lf, indent=2, default=str)
                    logger.info("Saved docker query/response to %s", dest)
                except Exception:
                    # Fallback: just copy the raw file
                    shutil.copy2(incremental_log_path, dest)
            else:
                self._save_query_response_log(message, responses)

            # Track in-memory for conversation replay
            self._conversation_log.append({
                "query": message,
                "response": responses,
            })

            return responses

        finally:
            # Cleanup temp data directory (sandbox persists across queries)
            shutil.rmtree(tmp_dir, ignore_errors=True)

    async def close(self) -> None:
        """No-op: sandbox directory is kept for inspection."""
        self._sandbox_dir = None

    async def _recover_session(self, last_message: str) -> None:
        """No-op: each query is independent."""
        pass

    def save_session_info(self) -> None:
        """Save session metadata to log directory."""
        os.makedirs(self._log_dir, exist_ok=True)
        info = {
            "session_type": "docker",
            "session_id": self._session_id,
            "total_cost_usd": self._total_cost_usd,
            "total_turns": self._total_turns,
            "model": self._model_name,
            "docker_image": self._image,
        }
        path = os.path.join(self._log_dir, "session_info.json")
        with open(path, "w") as f:
            json.dump(info, f, indent=2)
        logger.info("Saved session info to %s", path)

    # -- Internal helpers --

    def _build_docker_command(self, container_name: str,
                              tmp_dir: str) -> List[str]:
        """Build the ``docker run`` command."""
        cmd = [
            "docker", "run", "--rm",
            "--name", container_name,
            "--cap-add=NET_ADMIN",
            "--cap-add=NET_RAW",
        ]

        # Authentication: prefer ANTHROPIC_API_KEY, fall back to OAuth
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if api_key:
            cmd += ["-e", "ANTHROPIC_API_KEY"]
        else:
            oauth_token = _get_claude_oauth_token()
            if oauth_token:
                cmd += ["-e", "CLAUDE_CODE_OAUTH_TOKEN"]
                # We'll add this to env in _build_env()
            else:
                # Fall back to bind-mounting ~/.claude
                claude_cfg = Path(
                    os.environ.get("CLAUDE_CONFIG_DIR",
                                   str(Path.home() / ".claude")))
                cmd += ["-v", f"{claude_cfg}:/home/node/.claude"]

        # Mount predicators source for Python imports (hidden from agent
        # tools by the PreToolUse hook — only Python's import system can
        # read these files).
        cmd += ["-v", f"{self._repo_root}:/opt/predicators:ro"]
        cmd += ["-e", "PYTHONPATH=/opt/predicators"]

        # Mount curated sandbox directory
        assert self._sandbox_dir is not None
        cmd += ["-v", f"{self._sandbox_dir}:/sandbox"]

        # Mount data exchange directory
        cmd += ["-v", f"{tmp_dir}:/data"]

        # Working directory
        cmd += ["-w", "/sandbox"]

        # Image
        cmd.append(self._image)

        # Command: run the agent runner script from the mounted source
        cmd += [
            "python3", "-u",
            "/opt/predicators/predicators/agent_sdk/docker_agent_runner.py",
            "/data/query_input.pkl",
            "/data/query_output.pkl",
        ]

        return cmd

    def _build_env(self) -> Dict[str, str]:
        """Build environment dict for the docker subprocess."""
        # Pass through host env, stripping CLAUDECODE* vars
        env = {
            k: v for k, v in os.environ.items()
            if not k.startswith("CLAUDECODE")
        }

        # Ensure ANTHROPIC_API_KEY is passed through if set
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if api_key:
            env["ANTHROPIC_API_KEY"] = api_key
        else:
            # Try OAuth token
            oauth_token = _get_claude_oauth_token()
            if oauth_token:
                env["CLAUDE_CODE_OAUTH_TOKEN"] = oauth_token

        return env

    def _save_query_response_log(self, query: str,
                                 response: List[Dict[str, Any]]) -> None:
        """Save query and response to a timestamped JSON file."""
        if not self._log_dir:
            return

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = (f"docker_query_{self._query_count:03d}_"
                    f"{timestamp}.json")
        filepath = os.path.join(self._log_dir, filename)

        log_data = {
            "query_number": self._query_count,
            "timestamp": timestamp,
            "query": query,
            "response": response,
            "session_id": self._session_id,
            "docker_image": self._image,
        }

        os.makedirs(self._log_dir, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(log_data, f, indent=2, default=str)

        logger.info("Saved docker query/response to %s", filepath)
