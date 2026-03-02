"""Docker-sandboxed agent session manager.

Runs ``ClaudeSDKClient`` inside a Docker container so that the agent's
built-in tools (Bash, Read, Write, Edit, Glob, Grep) all execute in an
isolated environment.  Custom predicator MCP tools are created in-process
inside the container via the same ``create_mcp_tools()`` code used on
the host.

The host predicators source tree is mounted read-only at ``/workspace``.
A writable scratch area is at ``/sandbox``.  Shared data (pickled context
and results) passes through ``/data``.

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
    2. Runs ``docker run ... python3 docker_agent_runner.py``.
    3. Inside Docker, the runner script creates ``ClaudeSDKClient`` with
       both built-in tools AND custom MCP tools, queries the agent, and
       pickles back responses + mutated proposals.
    4. Host reads back the pickled results.
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
        self._system_prompt = system_prompt
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

        # 1. Create temp directory for data exchange
        tmp_dir = tempfile.mkdtemp(prefix="pred-docker-")
        input_path = os.path.join(tmp_dir, "query_input.pkl")
        output_path = os.path.join(tmp_dir, "query_output.pkl")

        try:
            # 2. Pickle QueryInput
            query_input = {
                "tool_context": self._tool_context,
                "message": message,
                "system_prompt": self._system_prompt,
                "model_name": self._model_name,
                "max_turns": CFG.agent_sdk_max_agent_turns_per_iteration,
                "tool_names": self._tool_names,
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

            proc = subprocess.run(
                docker_cmd,
                env=env,
                capture_output=True,
                text=True,
                timeout=CFG.agent_sdk_agent_timeout + 120,  # extra buffer
            )

            if proc.returncode != 0:
                logger.error(
                    "Docker container exited with code %d.\nstdout: %s\n"
                    "stderr: %s",
                    proc.returncode,
                    proc.stdout[-2000:] if proc.stdout else "(empty)",
                    proc.stderr[-2000:] if proc.stderr else "(empty)",
                )
            else:
                logger.info("Docker container exited successfully.")
                if proc.stdout:
                    logger.debug("stdout: %s", proc.stdout[-500:])

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
                        f"stderr: {proc.stderr[-500:] if proc.stderr else ''}"
                    ),
                }]

            # 7. Save query log
            self._save_query_response_log(message, responses)

            return responses

        finally:
            # Cleanup temp directory
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)

    async def close(self) -> None:
        """No-op: no persistent container to close."""
        pass

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
        runner_script = (
            "/workspace/predicators/agent_sdk/docker_agent_runner.py")

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

        # Mount predicators source read-only
        cmd += ["-v", f"{self._repo_root}:/workspace:ro"]

        # Mount data exchange directory
        cmd += ["-v", f"{tmp_dir}:/data"]

        # Working directory
        cmd += ["-w", "/sandbox"]

        # Image
        cmd.append(self._image)

        # Command: run the agent runner script
        cmd += [
            "python3", "-u",
            runner_script,
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
