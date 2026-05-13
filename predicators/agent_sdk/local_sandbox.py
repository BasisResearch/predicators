"""Local-sandboxed agent session manager.

Runs ``ClaudeSDKClient`` in-process with ``cwd`` set to a local sandbox
directory.  The agent gets built-in tools (Bash, Read, Write, Edit, Glob,
Grep, Task*) plus custom MCP tools, but PreToolUse hooks restrict built-in
file tools to the sandbox directory.

Unlike ``DockerSessionManager``, no Docker container is used -- the agent
runs directly on the host but is confined to the sandbox via hooks.

Curated reference files are copied into ``sandbox/reference/`` for the
agent to read.  The agent can write and run Python scripts in the sandbox.

Usage
-----
When ``CFG.agent_sdk_use_local_sandbox`` is ``True``, the
``AgentSessionMixin`` creates a ``LocalSandboxSessionManager`` in place
of the normal ``AgentSessionManager``::

    manager = LocalSandboxSessionManager(...)
    responses = await manager.query("Solve this task...")
    await manager.close()
"""
import datetime
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

from predicators.agent_sdk.log_formatter import format_conversation_markdown
from predicators.agent_sdk.response_parser import parse_message
from predicators.agent_sdk.sandbox_prompts import build_claude_md, \
    build_sandbox_system_prompt, find_repo_root, setup_sandbox_directory, \
    truncate
from predicators.agent_sdk.tools import BUILTIN_TOOLS, ToolContext
from predicators.settings import CFG

logger = logging.getLogger(__name__)

# Build local-sandbox-specific prompts from shared templates.
# CLAUDE.md is built per-instance with the phase tag so the agent reads
# phase-appropriate strategy guidance every turn (see build_claude_md).
_LOCAL_SANDBOX_SYSTEM_PROMPT = build_sandbox_system_prompt(
    env_description="a local sandbox environment",
    workspace_description="the current directory",
    ref_path="./reference/",
)


class LocalSandboxSessionManager:
    """Runs ClaudeSDKClient locally with cwd set to a sandbox directory.

    Matches the ``AgentSessionManager`` / ``DockerSessionManager``
    interface so that all agent-based approaches work unchanged.
    """

    def __init__(
        self,
        system_prompt: str,
        log_dir: str,
        model_name: str,
        tool_context: ToolContext,
        tool_names: Optional[List[str]] = None,
        extra_reference_files: Optional[Dict[str, str]] = None,
        phase: Optional[str] = None,
    ) -> None:
        self._system_prompt = system_prompt + _LOCAL_SANDBOX_SYSTEM_PROMPT
        self._log_dir = log_dir
        self._model_name = model_name
        self._tool_context = tool_context
        self._tool_names = tool_names
        self._extra_reference_files = extra_reference_files or {}
        self._repo_root = str(find_repo_root())
        self._phase = phase

        self._total_cost_usd: float = 0.0
        self._total_turns: int = 0
        self._query_count: int = 0
        self._session_id: Optional[str] = None
        self._conversation_log: List[Dict[str, Any]] = []
        # Sandbox path is deterministic from log_dir; expose it on the
        # tool context eagerly so callers that build sandbox-relative
        # paths before the first query() see the right value. Directory
        # creation + file copying still happen lazily in
        # ``_ensure_sandbox_dir`` on first query.
        self._sandbox_dir: Optional[str] = os.path.abspath(
            os.path.join(self._log_dir, "sandbox"))
        self._tool_context.sandbox_dir = self._sandbox_dir
        self._tool_context.image_save_dir = str(
            os.path.join(self._sandbox_dir, "test_images"))
        self._sandbox_populated = False
        self._client: Any = None
        self._started = False
        self._sandbox_log_path: Optional[str] = None
        self._current_log_meta: Dict[str, Any] = {}
        self._query_count_seeded: bool = False

    # -- Properties matching session manager interface --

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
        from predicators.agent_sdk.tools import \
            MCP_SERVER_NAME  # pylint: disable=import-outside-toplevel
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
        """Create and populate the sandbox directory if it doesn't exist.

        The path itself is set in ``__init__`` (so callers can use it
        before the first query); this method handles dir creation and
        seeding, which is idempotent across calls but only needs to run
        once per session.
        """
        if self._sandbox_populated:
            return
        assert self._sandbox_dir is not None  # set in __init__

        setup_sandbox_directory(
            sandbox_dir=self._sandbox_dir,
            repo_root=self._repo_root,
            extra_reference_files=self._extra_reference_files,
            claude_md_content=build_claude_md(phase=self._phase),
            system_prompt=self._system_prompt,
            log_dir=self._log_dir,
            seed_scratchpad=CFG.agent_planner_use_scratchpad,
            phase=self._phase,
        )
        self._sandbox_populated = True

    # -- Session lifecycle --

    async def start_session(self) -> None:
        """Create ClaudeSDKClient with cwd set to the sandbox directory."""
        # pylint: disable=import-outside-toplevel
        from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient, \
            create_sdk_mcp_server

        from predicators.agent_sdk.tools import create_mcp_tools, \
            get_allowed_tool_list

        # pylint: enable=import-outside-toplevel

        self._ensure_sandbox_dir()

        # Create MCP tools (closures over tool_context, in-process)
        tools = create_mcp_tools(self._tool_context,
                                 tool_names=self._tool_names)
        mcp_server = create_sdk_mcp_server(
            name="predicator_tools",
            version="1.0.0",
            tools=tools,
        )

        # Built-in tools + custom MCP tools
        mcp_tool_list = get_allowed_tool_list(self._tool_names)
        allowed_tools = BUILTIN_TOOLS + mcp_tool_list

        extra_hooks = dict(self._tool_context.extra_session_hooks or {})
        options = ClaudeAgentOptions(
            allowed_tools=allowed_tools,
            mcp_servers={"predicator_tools": mcp_server},
            permission_mode="bypassPermissions",
            system_prompt=self._system_prompt,
            model=self._model_name,
            max_turns=CFG.agent_sdk_max_agent_turns_per_iteration,
            cwd=self._sandbox_dir,
            setting_sources=["project", "local"],
            hooks=(extra_hooks
                   if extra_hooks else None),  # type: ignore[arg-type]
        )

        self._client = ClaudeSDKClient(options=options)
        await self._client.connect()
        self._started = True
        logger.info("Local sandbox session started (cwd=%s)",
                    self._sandbox_dir)

    async def query(self,
                    message: str,
                    kind: str = "query") -> List[Dict[str, Any]]:
        """Send a message to the agent and collect all response messages.

        ``kind`` is a short tag (e.g. ``learn``, ``test``, ``explore``)
        that becomes the prefix of the saved log filename.
        """
        # Continue numbering across sessions in the same run by seeding the
        # counter from any existing log files in _log_dir on first use.
        self._seed_query_count_from_log_dir()
        self._query_count += 1
        self._tool_context.turn_id = self._query_count
        collected: List[Dict[str, Any]] = []

        # Ensure sandbox exists before creating the log file.
        self._ensure_sandbox_dir()

        # Create and commit the log file BEFORE starting the session so that
        # Claude Code's Glob (which indexes files at session startup) can
        # discover it.
        log_path = self._init_incremental_log(message, kind=kind)

        if not self._started:
            await self.start_session()

        try:
            await self._client.query(message)
            async for msg in self._client.receive_response():
                entry = parse_message(msg)
                if entry is None:
                    continue
                collected.append(entry)

                # Log side-effects
                if entry["type"] == "assistant":
                    for block in entry.get("content", []):
                        if block.get("type") == "text":
                            logging.debug("Agent: %s...", block["text"][:200])
                        elif block.get("type") == "tool_use":
                            params = block.get("input") or {}
                            param_summary = ", ".join(
                                f"{k}={truncate(v)}"
                                for k, v in params.items())
                            logging.debug("Agent tool call: %s(%s)",
                                          block["name"], param_summary)
                elif entry["type"] == "result":
                    cost = entry.get("total_cost_usd")
                    turns = entry.get("num_turns")
                    if cost is not None:
                        self._total_cost_usd += cost
                    if turns is not None:
                        self._total_turns += turns
                    logging.info(
                        "Local sandbox iteration complete. "
                        "Turns: %s, Cost: $%s", turns or '?', cost or '?')

                # Flush log after each message
                if log_path:
                    self._flush_log(log_path, collected)

        except Exception as e:  # pylint: disable=broad-except
            logging.error("Local sandbox session error: %s", e)
            collected.append({"type": "error", "error": str(e)})
            await self._recover_session(message)

        # Final flush
        if log_path:
            self._flush_log(log_path, collected)
            logging.info("Saved local sandbox query/response to %s", log_path)

        # Log proposals (matches Docker sandbox logging)
        proposals = self._tool_context.iteration_proposals
        if proposals.proposed_options or proposals.retract_option_names:
            logger.info(
                "Local sandbox proposals: proposed_options=%s, "
                "retract=%s",
                [o.name for o in proposals.proposed_options],
                sorted(proposals.retract_option_names),
            )
            logger.info(
                "After local sandbox query: tool_context.options=%s",
                sorted(o.name for o in self._tool_context.options),
            )

        self._conversation_log.append({
            "query": message,
            "response": collected,
        })

        return collected

    async def close(self) -> None:
        """Close the agent session."""
        if self._client is not None:
            try:
                await self._client.disconnect()
            except Exception as e:  # pylint: disable=broad-except
                logging.warning("Error closing local sandbox session: %s", e)
            finally:
                self._client = None
                self._started = False

    async def _recover_session(self, _last_message: str) -> None:
        """Attempt to recover from a session error."""
        logging.warning("Attempting local sandbox session recovery...")
        try:
            if self._client is not None:
                try:
                    await self._client.disconnect()
                except Exception:  # pylint: disable=broad-except
                    pass
            self._started = False
            await self.start_session()
            logging.info("Local sandbox session recovered.")
        except Exception as e:  # pylint: disable=broad-except
            logging.error("Local sandbox session recovery failed: %s", e)

    def save_session_info(self) -> None:
        """Save session metadata to log directory."""
        os.makedirs(self._log_dir, exist_ok=True)
        info = {
            "session_type": "local_sandbox",
            "session_id": self._session_id,
            "total_cost_usd": self._total_cost_usd,
            "total_turns": self._total_turns,
            "model": self._model_name,
            "sandbox_dir": self._sandbox_dir,
        }
        path = os.path.join(self._log_dir, "session_info.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2)
        logging.info("Saved session info to %s", path)

    # -- Logging helpers --

    _LOG_FILENAME_RE = re.compile(r"^[a-z][a-z_]*_(\d{3})_\d{8}_\d{6}\.md$")

    def _seed_query_count_from_log_dir(self) -> None:
        """Make the per-session counter continuous across the run.

        On first use, scan ``_log_dir`` for prior log files matching
        ``<kind>_NNN_<ts>.md`` and pick up where the last session left
        off. Without this, every fresh session would restart at 001.
        """
        if self._query_count_seeded:
            return
        self._query_count_seeded = True
        if not self._log_dir or not os.path.isdir(self._log_dir):
            return
        max_n = 0
        for name in os.listdir(self._log_dir):
            m = self._LOG_FILENAME_RE.match(name)
            if m:
                max_n = max(max_n, int(m.group(1)))
        self._query_count = max_n

    def _init_incremental_log(self,
                              query: str,
                              kind: str = "query") -> Optional[str]:
        """Initialize log file for incremental writing.

        Writes to both the sandbox ``session_logs/`` dir (so the agent
        can read its own logs) and the main ``_log_dir`` (for the host).
        """
        if not self._log_dir:
            return None

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{kind}_{self._query_count:03d}_{timestamp}.md"
        # Primary: main log dir (host-visible)
        filepath = os.path.join(self._log_dir, filename)
        os.makedirs(self._log_dir, exist_ok=True)

        # Also write to sandbox/session_logs/ so the agent can read its own logs
        self._sandbox_log_path = None  # type: ignore[no-redef]
        if self._sandbox_dir is not None:
            sandbox_logs = os.path.join(self._sandbox_dir, "session_logs")
            os.makedirs(sandbox_logs, exist_ok=True)
            self._sandbox_log_path = os.path.join(sandbox_logs, filename)

        self._current_log_meta = {
            "query_number": self._query_count,
            "kind": kind,
            "timestamp": timestamp,
            "query": query,
            "session_id": self._session_id,
        }
        self._flush_log(filepath, [])

        # Commit the log file so Claude Code's Glob can discover it.
        # Claude Code indexes git-tracked files at session startup, so the
        # file must be committed before start_session() is called.
        if self._sandbox_log_path and self._sandbox_dir:
            try:
                import subprocess  # pylint: disable=import-outside-toplevel
                subprocess.run(
                    ["git", "add", self._sandbox_log_path],
                    cwd=self._sandbox_dir,
                    capture_output=True,
                    timeout=5,
                    check=False,
                )
                subprocess.run(
                    [
                        "git", "commit", "-q", "-m",
                        f"log query {self._query_count}", "--author",
                        "sandbox <sandbox@local>"
                    ],
                    cwd=self._sandbox_dir,
                    capture_output=True,
                    timeout=5,
                    check=False,
                    env={
                        **os.environ, "GIT_COMMITTER_NAME": "sandbox",
                        "GIT_COMMITTER_EMAIL": "sandbox@local"
                    },
                )
            except Exception:  # pylint: disable=broad-except
                pass
        return filepath

    def _flush_log(self, filepath: str, response: List[Dict[str,
                                                            Any]]) -> None:
        """Write current conversation state as markdown to the log file."""
        try:
            log_content = format_conversation_markdown(
                response,
                title="Local Sandbox Query",
                meta=self._current_log_meta,
            )
            with open(filepath, "w", encoding="utf-8") as lf:
                lf.write(log_content)
            # Also write to sandbox/session_logs/ for agent access
            sandbox_path = getattr(self, '_sandbox_log_path', None)
            if sandbox_path:
                with open(sandbox_path, "w", encoding="utf-8") as lf:
                    lf.write(log_content)
        except Exception:  # pylint: disable=broad-except
            pass  # Don't let logging errors break the agent
