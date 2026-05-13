"""Mixin providing shared agent session infrastructure.

Extracts common code for ToolContext initialization, lazy
AgentSessionManager creation, async-to-sync bridging, and agent explorer
creation from AgentPlannerApproach and AgentAbstractionLearningApproach.
"""
import asyncio
import logging
import os
from typing import Any, Dict, List, Optional, Set, Union

from predicators.agent_sdk.session_manager import AgentSessionManager, \
    run_query_sync
from predicators.agent_sdk.tools import ALL_TOOL_NAMES, ToolContext, \
    create_mcp_tools, get_allowed_tool_list
from predicators.explorers import create_explorer
from predicators.explorers.base_explorer import BaseExplorer
from predicators.settings import CFG
from predicators.structs import ParameterizedOption, Predicate, Task, Type

logger = logging.getLogger(__name__)


class AgentSessionMixin:
    """Mixin that provides shared agent session infrastructure.

    Subclasses must override:
      - _get_agent_system_prompt()

    And may optionally override:
      - _get_solve_tool_names()         -- complete tool surface for
        solve / explore sessions. May mix static MCP tool names with
        names of dynamic ``SdkMcpTool`` instances. ``None`` = all
        static MCP tools, ``[]`` = none.
      - _get_synthesis_tool_names()     -- complete tool surface for
        synthesis sessions (``_learning_mode=True``). Same shape /
        semantics as the solve hook, independent value.

    Dynamic ``SdkMcpTool`` instances are supplied by the approach
    directly: it assigns them to ``ctx.extra_mcp_tools`` before
    opening a synthesis session and clears the field afterwards. The
    mixin asserts the instance names line up with the names declared
    in :meth:`_get_synthesis_tool_names`.
    """

    _log_subdir: str = "agent"  # fallback; _get_log_dir prefers get_name()

    # ------------------------------------------------------------------ #
    # Initialization
    # ------------------------------------------------------------------ #

    def _init_agent_session_state(
        self,
        types: Set[Type],
        predicates: Set[Predicate],
        options: Set[ParameterizedOption],
        train_tasks: List[Task],
    ) -> None:
        """Initialize ToolContext and lazy agent session placeholders."""
        self._tool_context = ToolContext(
            types=types,
            predicates=predicates,
            options=options,
            train_tasks=train_tasks,
        )
        self._agent_session: Optional[Union[
            AgentSessionManager, Any]] = None  # or DockerSessionManager
        self._agent_session_id: Optional[str] = None

    # ------------------------------------------------------------------ #
    # Customization hooks (override in subclasses)
    # ------------------------------------------------------------------ #

    def _get_agent_system_prompt(self) -> str:
        """Return the system prompt for the agent session."""
        raise NotImplementedError

    def _get_solve_tool_names(self) -> Optional[List[str]]:
        """Return the complete tool surface for solve / explore sessions.

        May mix static MCP tool names with names of dynamic
        ``SdkMcpTool`` instances. ``None`` means "all static MCP tools";
        override to subset.
        """
        return None

    def _get_synthesis_tool_names(self) -> Optional[List[str]]:
        """Return the complete tool surface for the synthesis session.

        Selected when ``_learning_mode`` is True. Independent of the
        solve list — the two phases may share names or be disjoint. Each
        name must back either a static MCP tool (member of
        ``ALL_TOOL_NAMES``) or a dynamic ``SdkMcpTool`` instance the
        approach attaches via ``ctx.extra_mcp_tools``. Default ``[]``
        means no tools (approaches with no synthesis phase need not
        override).
        """
        return []

    def _get_sandbox_reference_files(self) -> Dict[str, str]:
        """Return extra reference files for the docker sandbox.

        Maps destination paths (relative to ``/sandbox/reference/``) to
        source paths (relative to the repo root).  Override in
        subclasses to provide approach-specific reference material.
        """
        return {}

    # ------------------------------------------------------------------ #
    # Shared implementations
    # ------------------------------------------------------------------ #

    def _ensure_agent_session(self) -> None:
        """Create the agent session manager if needed.

        When ``CFG.agent_sdk_use_docker_sandbox`` is ``True``, creates a
        ``DockerSessionManager`` that runs ``ClaudeSDKClient`` inside a
        Docker container with full built-in tools (Bash, Read, Write,
        …). Otherwise creates the normal in-process
        ``AgentSessionManager``.
        """
        if self._agent_session is not None:
            return

        # Pick the declared tool surface by phase. ``_learning_mode`` is
        # the same signal the system-prompt branch reads, so tools and
        # prompt stay in sync. Each approach declares its solve and
        # synthesis tool sets independently — they may be disjoint.
        # ``tool_names`` is the *complete* declared list (may mix static
        # MCP names with names of dynamic SdkMcpTool instances).
        if getattr(self, "_learning_mode", False):
            tool_names = self._get_synthesis_tool_names()  # pylint: disable=assignment-from-none
        else:
            tool_names = self._get_solve_tool_names()  # pylint: disable=assignment-from-none

        # Sanity: every dynamic name in the declared list must have a
        # backing tool attached to ``ctx.extra_mcp_tools``. Static MCP
        # names (``ALL_TOOL_NAMES``) are excluded — they're materialized
        # downstream by ``create_mcp_tools``. Catches typos and missing
        # builder hooks before the agent silently fails to invoke a
        # declared-but-missing tool.
        declared = set(tool_names or ())
        dynamic_declared = declared - set(ALL_TOOL_NAMES)
        if dynamic_declared:
            attached = list(self._tool_context.extra_mcp_tools or ())
            built = {getattr(t, "name", "") for t in attached}
            missing = dynamic_declared - built
            phase_for_msg = ("synthesis" if getattr(self, "_learning_mode",
                                                    False) else "solve")
            assert not missing, (
                f"Dynamic tool name(s) {sorted(missing)} declared in "
                f"_get_{phase_for_msg}_tool_names but no matching tool "
                f"attached to ctx.extra_mcp_tools — add them to the "
                f"builder or drop the names.")

        phase = "synthesis" if getattr(self, "_learning_mode",
                                       False) else "solve"
        approach_name = getattr(type(self), "get_name",
                                lambda: type(self).__name__)()
        if tool_names is None:
            logger.info(
                "[%s] %s session tool surface: ALL static MCP tools "
                "(no subset declared).", approach_name, phase)
        else:
            static = sorted(n for n in tool_names if n in set(ALL_TOOL_NAMES))
            dynamic = sorted(n for n in tool_names
                             if n not in set(ALL_TOOL_NAMES))
            logger.info(
                "[%s] %s session tool surface (%d total): "
                "static=%s dynamic=%s", approach_name, phase, len(tool_names),
                static, dynamic)

        if CFG.agent_sdk_use_docker_sandbox:
            from predicators.agent_sdk.docker_sandbox import \
                DockerSessionManager  # pylint: disable=import-outside-toplevel
            self._agent_session = DockerSessionManager(
                system_prompt=self._get_agent_system_prompt(),
                log_dir=self._get_log_dir(),
                model_name=CFG.agent_sdk_model_name,
                tool_context=self._tool_context,
                tool_names=tool_names,
                image=CFG.agent_sdk_docker_image,
                extra_reference_files=self._get_sandbox_reference_files(),
                phase=phase,
            )
        elif CFG.agent_sdk_use_local_sandbox:
            from predicators.agent_sdk.local_sandbox import \
                LocalSandboxSessionManager  # pylint: disable=import-outside-toplevel
            self._agent_session = LocalSandboxSessionManager(
                system_prompt=self._get_agent_system_prompt(),
                log_dir=self._get_log_dir(),
                model_name=CFG.agent_sdk_model_name,
                tool_context=self._tool_context,
                tool_names=tool_names,
                extra_reference_files=self._get_sandbox_reference_files(),
                phase=phase,
            )
        else:
            from claude_agent_sdk import \
                create_sdk_mcp_server  # pylint: disable=import-outside-toplevel

            tools = create_mcp_tools(self._tool_context, tool_names=tool_names)
            mcp_server = create_sdk_mcp_server(
                name="predicator_tools",
                version="1.0.0",
                tools=tools,
            )

            self._agent_session = AgentSessionManager(
                system_prompt=self._get_agent_system_prompt(),
                mcp_server=mcp_server,
                log_dir=self._get_log_dir(),
                model_name=CFG.agent_sdk_model_name,
                allowed_tools=get_allowed_tool_list(tool_names),
                tool_context=self._tool_context,
            )

        if self._agent_session_id is not None:
            sess = self._agent_session
            sess.session_id = (  # type: ignore[attr-defined,union-attr]
                self._agent_session_id)

        # Save system prompt to log directory. Suffix with the phase tag
        # so solve and synthesis prompts don't overwrite each other across
        # phase switches.
        log_dir = self._get_log_dir()
        os.makedirs(log_dir, exist_ok=True)
        prompt_path = os.path.join(log_dir, f"system_prompt_{phase}.md")
        with open(prompt_path, "w", encoding="utf-8") as f:
            f.write(self._get_agent_system_prompt())

    def _get_log_dir(self) -> str:
        """Return the log directory, using the approach name."""
        if hasattr(CFG, 'log_file') and CFG.log_file:
            return CFG.log_file
        name = (
            self.get_name()  # type: ignore[attr-defined]
            if hasattr(self, 'get_name') else self._log_subdir)
        return os.path.join("logs", name)

    def _close_agent_session(self) -> None:
        """Close and discard the current agent session, if one exists."""
        if self._agent_session is None:
            return
        session = self._agent_session
        self._agent_session = None
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import nest_asyncio  # type: ignore[import-untyped,import-not-found]  # pylint: disable=import-outside-toplevel
                nest_asyncio.apply()
                loop.run_until_complete(session.close())
            else:
                loop.run_until_complete(session.close())
        except RuntimeError:
            asyncio.run(session.close())
        except Exception:  # pylint: disable=broad-except
            pass

    def _query_agent_sync(self, message: str,
                          **query_kwargs: Any) -> List[Dict[str, Any]]:
        """Synchronous wrapper for async agent query.

        Extra kwargs (e.g. ``kind="learn"``) are forwarded to the
        session's ``query`` method for log-file tagging.
        """
        self._ensure_agent_session()
        assert self._agent_session is not None
        return run_query_sync(self._agent_session, message, **query_kwargs)

    def _create_agent_explorer(
        self,
        predicates: Set[Predicate],
        options: Set[ParameterizedOption],
        name: str = "agent_plan",
    ) -> BaseExplorer:
        """Create an agent explorer with tool_context and agent_session."""
        self._ensure_agent_session()
        return create_explorer(
            name,
            predicates,
            options,
            self._types,  # type: ignore[attr-defined]
            self._action_space,  # type: ignore[attr-defined]
            self._train_tasks,  # type: ignore[attr-defined]
            tool_context=self._tool_context,
            agent_session=self._agent_session,
        )
