"""Local-sandboxed agent session manager.

Runs ``ClaudeSDKClient`` in-process with ``cwd`` set to a local sandbox
directory.  The agent gets built-in tools (Bash, Read, Write, Edit, Glob,
Grep, Task*) plus custom MCP tools, but PreToolUse hooks restrict built-in
file tools to the sandbox directory.

Unlike ``DockerSessionManager``, no Docker container is used -- the agent
runs directly on the host but is confined to the sandbox via hooks.

Curated reference files are copied into ``sandbox/reference/`` for the
agent to read.  The agent can write and run Python scripts in the sandbox.

Behavioral notes relative to the shared base
(:mod:`predicators.agent_sdk.session_base`):

- Query logs are markdown, dual-written to the host ``_log_dir`` and to
  ``sandbox/session_logs/`` so the agent can read its own logs, and
  git-committed before session start for Glob discovery.
- The per-session query counter is seeded from existing log files so
  numbering stays continuous across sessions in the same run (this is
  deliberately local-only).
- A wall-clock deadline interrupt rides on the receive-loop's per-entry
  callback (see ``query``).

Usage
-----
When the ``agent_sdk_use_local_sandbox`` flag is ``True``, the
``AgentSessionMixin`` creates a ``LocalSandboxSessionManager`` in place
of the normal ``AgentSessionManager``::

    manager = LocalSandboxSessionManager(...)
    responses = await manager.query("Solve this task...")
    await manager.close()
"""
import datetime
import logging
import os
import time
from typing import Any, Dict, List, Optional

from predicators.agent_sdk.config import SessionConfig
from predicators.agent_sdk.log_formatter import format_conversation_markdown
from predicators.agent_sdk.sandbox_prompts import build_sandbox_system_prompt
from predicators.agent_sdk.sandbox_setup import export_trajectories, \
    git_commit_all, pyguard_env
from predicators.agent_sdk.session_base import SandboxSessionManagerBase, \
    build_agent_options, build_sandbox_mcp, max_session_log_number
from predicators.agent_sdk.tools import ToolContext, session_log_filename

logger = logging.getLogger(__name__)

# The CLI's wall-clock limit per MCP tool call, in milliseconds, unless
# the environment sets MCP_TOOL_TIMEOUT: six hours, room for a learning
# session run inside one tool call.
MCP_TOOL_TIMEOUT_MS = 6 * 3600 * 1000

# Grace period past the solve-attempt deadline before interrupting a
# still-streaming agent turn (cooperative tool refusals normally end
# the turn well before this).
_DEADLINE_INTERRUPT_SLACK_S = 180

# Build local-sandbox-specific prompts from shared templates.
# CLAUDE.md (sandbox mechanics only; see build_claude_md) is written
# into the sandbox when it is populated.
_LOCAL_SANDBOX_SYSTEM_PROMPT = build_sandbox_system_prompt(
    env_description="a local sandbox environment",
    workspace_description="the current directory",
    ref_path="./reference/",
)


class LocalSandboxSessionManager(SandboxSessionManagerBase):
    """Runs ClaudeSDKClient locally with cwd set to a sandbox directory.

    Matches the ``AgentSessionManager`` / ``DockerSessionManager``
    interface so that all agent-based approaches work unchanged.
    """

    _log_label = "Local sandbox"

    def __init__(
        self,
        system_prompt: str,
        log_dir: str,
        model_name: str,
        tool_context: ToolContext,
        tool_names: Optional[List[str]] = None,
        extra_reference_files: Optional[Dict[str, str]] = None,
        phase: Optional[str] = None,
        config: Optional[SessionConfig] = None,
        query_count_floor: int = 0,
    ) -> None:
        super().__init__(system_prompt=system_prompt +
                         _LOCAL_SANDBOX_SYSTEM_PROMPT,
                         log_dir=log_dir,
                         model_name=model_name,
                         tool_context=tool_context,
                         tool_names=tool_names,
                         extra_reference_files=extra_reference_files,
                         phase=phase,
                         config=config)
        self._sandbox_log_path: Optional[str] = None
        self._query_count_seeded: bool = False
        # Lowest transcript number this session may hand out minus one:
        # an auto-resumed run passes the count its predecessor reached
        # (recorded in the checkpoint), so ids continue across the
        # lineage instead of restarting at 001 in the new run dir.
        self._query_count_floor = int(query_count_floor)

    # -- Session lifecycle --

    async def start_session(self) -> None:
        """Create ClaudeSDKClient with cwd set to the sandbox directory."""
        from claude_agent_sdk import \
            ClaudeSDKClient  # pylint: disable=import-outside-toplevel

        self._ensure_sandbox_dir()

        # Create MCP tools (closures over tool_context, in-process) and
        # the combined built-in + custom allowed-tool list.
        mcp_server, allowed_tools = build_sandbox_mcp(self._tool_context,
                                                      self._tool_names)

        extra_hooks = dict(self._tool_context.extra_session_hooks or {})
        options = build_agent_options(
            system_prompt=self._system_prompt,
            model_name=self._model_name,
            allowed_tools=allowed_tools,
            mcp_server=mcp_server,
            max_turns=self._config.max_turns,
            max_buffer_size=self._config.max_buffer_size,
            reasoning_effort=self._config.reasoning_effort,
            cwd=self._sandbox_dir,
            setting_sources=["project", "local"],
            hooks=extra_hooks,
            # Every python the agent starts loads the sandbox's
            # sitecustomize guard (sandbox_setup.write_pyguard); a tool
            # call may run a whole learning session (continual play's
            # learn_run), so the CLI's per-call timeout is raised.
            env={
                "MCP_TOOL_TIMEOUT":
                os.environ.get("MCP_TOOL_TIMEOUT", str(MCP_TOOL_TIMEOUT_MS)),
                **pyguard_env(self._sandbox_dir),
            },
            resume=self.resume_session_id,
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

        # Ensure sandbox exists before creating the log file.
        self._ensure_sandbox_dir()
        self._export_data()

        # Create and commit the log file BEFORE starting the session so that
        # Claude Code's Glob (which indexes files at session startup) can
        # discover it.
        log_path = self._init_incremental_log(message, kind=kind)

        if not self._started:
            await self.start_session()

        # Wall-clock backstop for the solve attempt deadline: the probe
        # and run_python enforce it cooperatively (tool calls refuse
        # past the deadline), so normally the agent wraps up on its own;
        # interrupt only if the turn stream is still going long after.
        # The approach clears attempt_deadline before its final-submission
        # nudge, so the submission query is never interrupted.
        interrupt_sent = False

        async def _maybe_interrupt_on_deadline(_entry: Dict[str, Any]) -> None:
            nonlocal interrupt_sent
            deadline = getattr(self._tool_context, "attempt_deadline", None)
            if (interrupt_sent or deadline is None or time.monotonic() <=
                    deadline + _DEADLINE_INTERRUPT_SLACK_S):
                return
            interrupt_sent = True
            logger.warning(
                "Solve-attempt wall clock exceeded by >%ds mid-query; "
                "interrupting the agent turn.", _DEADLINE_INTERRUPT_SLACK_S)
            try:
                await self._client.interrupt()
            except Exception as e:  # pylint: disable=broad-except
                logger.warning("Interrupt failed: %s", e)

        collected = await self._run_streamed_query(
            message,
            log_path=log_path,
            kind=kind,
            on_entry=_maybe_interrupt_on_deadline)

        return collected

    def _session_info_extras(self) -> Dict[str, Any]:
        """Extra session-info keys: manager type + sandbox location."""
        return {
            "session_type": "local_sandbox",
            "sandbox_dir": self._sandbox_dir,
        }

    # -- Logging helpers --

    # Matches the new ``NNN_kind[_taskN]_ts.md`` layout and the legacy
    # ``kind_NNN_ts.md`` layout so resuming across the migration is
    # lossless. The counter is always captured in group 1 or 2; the
    # optional ``_task<idx>`` segment tags test queries with their task.
    def _seed_query_count_from_log_dir(self) -> None:
        """Make the per-session counter continuous across the run.

        On first use, scan ``_log_dir`` for prior log files matching
        ``NNN_<kind>_<ts>.md`` (or the legacy ``<kind>_NNN_<ts>.md``)
        and pick up where the last session left off, never below the
        resumed lineage's floor. Without this, every fresh session would
        restart at 001.
        """
        if self._query_count_seeded:
            return
        self._query_count_seeded = True
        self._query_count = max(max_session_log_number(self._log_dir),
                                self._query_count_floor)

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
        # Counter-first layout: alphabetical sort matches chronological
        # order across mixed ``learn``/``test``/``explore`` phases. Test
        # queries also carry a ``_task<idx>`` segment for attribution.
        filename = session_log_filename(
            self._query_count, kind, timestamp,
            getattr(self._tool_context, "test_task_idx", None))
        # Primary: main log dir (host-visible)
        filepath = os.path.join(self._log_dir, filename)
        os.makedirs(self._log_dir, exist_ok=True)

        # Also write to sandbox/session_logs/ so the agent can read its own logs
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
        try:
            git_commit_all(self._sandbox_dir,
                           f"log query {self._query_count}",
                           paths=[self._sandbox_log_path])
        except Exception as e:  # pylint: disable=broad-except
            # A failed commit breaks the agent's Glob discovery of its
            # own logs, so it is worth a visible warning.
            logger.warning("git commit of session log failed: %s", e)
        return filepath

    def _export_data(self) -> None:
        """Refresh ``data/trajectories.pkl`` from the tool context so the
        agent's own scripts read the same training data the prompts and tool
        namespace expose."""
        ctx = self._tool_context
        trajectories = list(getattr(ctx, "offline_trajectories", []) or []) + \
            list(getattr(ctx, "online_trajectories", []) or [])
        try:
            if export_trajectories(self._sandbox_dir, trajectories):
                logger.info("Sandbox data refreshed: %d trajectories.",
                            len(trajectories))
        except Exception as e:  # pylint: disable=broad-except
            logger.warning("Sandbox data export failed: %s", e)

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
            if self._sandbox_log_path:
                with open(self._sandbox_log_path, "w", encoding="utf-8") as lf:
                    lf.write(log_content)
        except Exception as e:  # pylint: disable=broad-except
            logger.debug("Session-log flush failed: %s", e)
            # Don't let logging errors break the agent.
