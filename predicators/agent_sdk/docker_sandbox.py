"""Docker-sandboxed agent session manager.

Runs ``ClaudeSDKClient`` inside a Docker container so that the agent's
built-in tools (Bash, Read, Write, Edit, Glob, Grep, Task*) all execute
in an isolated environment.  Custom predicator MCP tools are created in-process
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

from predicators.agent_sdk.sandbox_prompts import build_claude_md, \
    build_sandbox_system_prompt, find_repo_root, setup_sandbox_directory
from predicators.agent_sdk.tools import ToolContext
from predicators.settings import CFG

logger = logging.getLogger(__name__)

# Build Docker-specific prompts from shared templates.
_CLAUDE_MD_TEMPLATE = build_claude_md(log_prefix="docker_query")
_SANDBOX_SYSTEM_PROMPT = build_sandbox_system_prompt(
    env_description="an isolated Docker sandbox",
    workspace_description="/sandbox/",
    ref_path="/sandbox/reference/",
    log_prefix="docker_query",
)

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _get_claude_oauth_token() -> Optional[str]:
    """Extract the Claude Code OAuth access token from the macOS Keychain.

    Returns ``None`` on non-macOS platforms or when the token cannot be
    found.  On macOS, ``claude login`` stores credentials under the
    service name ``"Claude Code-credentials"``.
    """
    if sys.platform != "darwin":
        return None
    try:  # type: ignore[unreachable]
        result = subprocess.run(
            [
                "security", "find-generic-password", "-s",
                "Claude Code-credentials", "-w"
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode != 0:
            return None
        import json as _json  # pylint: disable=reimported,import-outside-toplevel
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
        extra_reference_files: Optional[Dict[str, str]] = None,
    ) -> None:
        # Append sandbox instructions to the system prompt
        self._system_prompt = system_prompt + _SANDBOX_SYSTEM_PROMPT
        self._log_dir = log_dir
        self._model_name = model_name
        self._tool_context = tool_context
        self._tool_names = tool_names
        self._image = image
        self._extra_reference_files = extra_reference_files or {}
        self._repo_root = str(find_repo_root())

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
        """Return the current session ID."""
        return self._session_id

    @session_id.setter
    def session_id(self, value: Optional[str]) -> None:
        self._session_id = value

    @property
    def tool_names(self) -> List[str]:
        """Return short tool names (without MCP prefix)."""
        # pylint: disable=import-outside-toplevel
        from predicators.agent_sdk.tools import BUILTIN_TOOLS, MCP_SERVER_NAME

        # pylint: enable=import-outside-toplevel
        prefix = f"mcp__{MCP_SERVER_NAME}__"
        names = list(BUILTIN_TOOLS)
        if self._tool_names:
            names += self._tool_names
        return [t[len(prefix):] if t.startswith(prefix) else t for t in names]

    @property
    def conversation_log(self) -> List[Dict[str, Any]]:
        """Return the in-memory log of all query/response pairs."""
        return self._conversation_log

    # -- Sandbox setup --

    def _ensure_sandbox_dir(self) -> None:
        """Create and populate the sandbox directory if it doesn't exist."""
        if self._sandbox_dir is not None:
            return

        self._sandbox_dir = os.path.abspath(
            os.path.join(self._log_dir, "sandbox"))

        setup_sandbox_directory(
            sandbox_dir=self._sandbox_dir,
            repo_root=self._repo_root,
            extra_reference_files=self._extra_reference_files,
            claude_md_content=_CLAUDE_MD_TEMPLATE,
            system_prompt=self._system_prompt,
            log_dir=self._log_dir,
            seed_scratchpad=CFG.agent_planner_use_scratchpad,
        )

        # Set sandbox paths on tool context
        # (In Docker, these are host paths; the container maps them to
        # /sandbox/ via the volume mount.)
        self._tool_context.image_save_dir = str(
            os.path.join(self._sandbox_dir, "test_images"))
        self._tool_context.sandbox_dir = self._sandbox_dir

    # -- Session lifecycle --

    async def start_session(self) -> None:
        """No-op: each query() is a fresh docker run."""

    async def query(self, message: str) -> List[Dict[str, Any]]:
        """Run the agent in Docker and return collected response messages.

        Returns the same ``List[Dict[str, Any]]`` format as
        ``AgentSessionManager.query()``.
        """
        self._query_count += 1
        self._tool_context.turn_id = self._query_count

        # Ensure sandbox is set up (lazy init, persists across queries)
        self._ensure_sandbox_dir()

        # 1. Create temp directory for data exchange
        tmp_dir = tempfile.mkdtemp(prefix="pred-docker-")
        input_path = os.path.join(tmp_dir, "query_input.pkl")
        output_path = os.path.join(tmp_dir, "query_output.pkl")

        # Compute final log filename upfront so the container can write
        # directly to the log directory (incremental updates visible on host).
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = (f"docker_query_{self._query_count:03d}_"
                        f"{timestamp}.md")
        if self._log_dir:
            os.makedirs(self._log_dir, exist_ok=True)
            incremental_log_path = os.path.join(self._log_dir, log_filename)
        else:
            incremental_log_path = os.path.join(tmp_dir, "query_log.md")

        try:
            # 2. Pickle QueryInput
            # Tell the container where to write the incremental log.
            # If _log_dir is set, it's mounted at /log inside the container.
            container_log_path = (f"/log/{log_filename}"
                                  if self._log_dir else "/data/query_log.md")
            query_input = {
                "tool_context": self._tool_context,
                "message": message,
                "system_prompt": self._system_prompt,
                "model_name": self._model_name,
                "max_turns": CFG.agent_sdk_max_agent_turns_per_iteration,
                "tool_names": self._tool_names,
                "cfg_snapshot": dict(CFG.__dict__),
                "log_path": container_log_path,
            }
            with open(input_path, "wb") as f:
                pkl.dump(query_input, f)

            logger.info(
                "Docker query %d: message length=%d, model=%s",
                self._query_count,
                len(message),
                self._model_name,
            )

            # 3. Build docker run command
            container_name = f"pred-sandbox-{uuid.uuid4().hex[:8]}"
            docker_cmd = self._build_docker_command(container_name, tmp_dir)

            # 4. Run Docker container
            logger.info(
                "Starting Docker sandbox: container=%s image=%s",
                container_name,
                self._image,
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
                import threading  # pylint: disable=import-outside-toplevel

                def _stream_stderr() -> None:
                    assert proc.stderr is not None
                    for line in proc.stderr:
                        line = line.rstrip("\n")
                        stderr_lines.append(line)
                        logger.info("%s", line)

                stderr_thread = threading.Thread(target=_stream_stderr,
                                                 daemon=True)
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
                with open(output_path, "rb") as f_in:
                    query_output = pkl.load(f_in)

                responses = query_output.get("responses", [])
                proposals = query_output.get("iteration_proposals")

                # 6. Merge proposals back into host ToolContext
                if proposals is not None:
                    logger.info(
                        "Docker proposals: proposed_options=%s, "
                        "retract=%s",
                        [o.name for o in proposals.proposed_options],
                        sorted(proposals.retract_option_names),
                    )
                    self._tool_context.iteration_proposals = proposals
                    # Sync proposed/retracted options into ctx.options so
                    # the host-side parser can find them.
                    self._tool_context.options |= proposals.proposed_options
                    if proposals.retract_option_names:
                        self._tool_context.options = {
                            o
                            for o in self._tool_context.options
                            if o.name not in proposals.retract_option_names
                        }
                    logger.info(
                        "After Docker sync: tool_context.options=%s",
                        sorted(o.name for o in self._tool_context.options),
                    )
                else:
                    logger.warning(
                        "Docker output has iteration_proposals=None; "
                        "no proposals synced.")

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
                    "type":
                    "error",
                    "error": (f"Docker container failed (exit code "
                              f"{proc.returncode}). "
                              f"stderr: {''.join(stderr_lines[-20:])}"),
                }]

            # 7. Finalize query log — the incremental log was written
            # directly to _log_dir as markdown (updated per-message).
            # Prepend host metadata header now that the container is done.
            if os.path.exists(incremental_log_path) and self._log_dir:
                try:
                    with open(incremental_log_path, encoding="utf-8") as lf:
                        existing = lf.read()
                    header_lines = [
                        f"- **Query:** {self._query_count}",
                        f"- **Timestamp:** {timestamp}",
                        f"- **Session:** {self._session_id}",
                        f"- **Image:** {self._image}",
                        "",
                        "",
                    ]
                    with open(incremental_log_path, "w",
                              encoding="utf-8") as lf:
                        lf.write("\n".join(header_lines) + existing)
                    logger.info("Finalized docker query/response at %s",
                                incremental_log_path)
                except Exception:  # pylint: disable=broad-except
                    logger.warning("Failed to enrich log at %s",
                                   incremental_log_path,
                                   exc_info=True)
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
        with open(path, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2)
        logger.info("Saved session info to %s", path)

    # -- Internal helpers --

    def _build_docker_command(self, container_name: str,
                              tmp_dir: str) -> List[str]:
        """Build the ``docker run`` command."""
        cmd = [
            "docker",
            "run",
            "--rm",
            "--name",
            container_name,
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

        # Mount log directory for incremental log updates visible on host
        if self._log_dir:
            log_dir_abs = os.path.abspath(self._log_dir)
            cmd += ["-v", f"{log_dir_abs}:/log"]

        # Working directory
        cmd += ["-w", "/sandbox"]

        # Image
        cmd.append(self._image)

        # Command: run the agent runner script from the mounted source
        cmd += [
            "python3",
            "-u",
            "/opt/predicators/predicators/agent_sdk/docker_agent_runner.py",
            "/data/query_input.pkl",
            "/data/query_output.pkl",
        ]

        return cmd

    def _build_env(self) -> Dict[str, str]:
        """Build environment dict for the docker subprocess."""
        # Pass through host env, stripping CLAUDECODE* vars
        env = {
            k: v
            for k, v in os.environ.items() if not k.startswith("CLAUDECODE")
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
        """Save query and response to a timestamped markdown file."""
        if not self._log_dir:
            return

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = (f"docker_query_{self._query_count:03d}_"
                    f"{timestamp}.md")
        filepath = os.path.join(self._log_dir, filename)

        lines = [
            f"- **Query:** {self._query_count}",
            f"- **Timestamp:** {timestamp}",
            f"- **Session:** {self._session_id}",
            f"- **Image:** {self._image}",
            "",
            "# Docker Query",
            "",
            "## Prompt",
            "",
            query,
            "",
            "## Response",
            "",
        ]
        for entry in response:
            lines.append(
                f"```json\n{json.dumps(entry, indent=2, default=str)}\n```")
            lines.append("")

        os.makedirs(self._log_dir, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        logger.info("Saved docker query/response to %s", filepath)
