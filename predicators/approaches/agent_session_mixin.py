"""Mixin providing shared agent session infrastructure.

Extracts common code for ToolContext initialization, lazy
AgentSessionManager creation, async-to-sync bridging, and agent explorer
creation from AgentOpenLoopApproach and OnlineAgentProcessPlanningApproach.
"""
import asyncio
import os
from typing import Any, Dict, List, Optional, Set

from predicators.agent_sdk.session_manager import AgentSessionManager
from predicators.agent_sdk.tools import ToolContext
from predicators.explorers import create_explorer
from predicators.explorers.base_explorer import BaseExplorer
from predicators.settings import CFG
from predicators.structs import ParameterizedOption, Predicate, Task, Type


class AgentSessionMixin:
    """Mixin that provides shared agent session infrastructure.

    Subclasses must override the abstract hooks:
      - _get_agent_model_name()
      - _get_agent_system_prompt()
      - _create_agent_mcp_tools()

    And may optionally override:
      - _get_agent_allowed_tools()
    """

    _log_subdir: str = "agent"

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
        self._agent_session: Optional[AgentSessionManager] = None
        self._agent_session_id: Optional[str] = None

    # ------------------------------------------------------------------ #
    # Customization hooks (override in subclasses)
    # ------------------------------------------------------------------ #

    def _get_agent_model_name(self) -> str:
        """Return the model name setting for the agent session."""
        raise NotImplementedError

    def _get_agent_system_prompt(self) -> str:
        """Return the system prompt for the agent session."""
        raise NotImplementedError

    def _create_agent_mcp_tools(self) -> list:
        """Return the MCP tools list for the agent session."""
        raise NotImplementedError

    def _get_agent_allowed_tools(self) -> Optional[List[str]]:
        """Return optional tool whitelist. None means allow all."""
        return None

    # ------------------------------------------------------------------ #
    # Shared implementations
    # ------------------------------------------------------------------ #

    def _ensure_agent_session(self) -> None:
        """Create the agent session manager if needed."""
        if self._agent_session is not None:
            return

        from claude_agent_sdk import create_sdk_mcp_server

        system_prompt = self._get_agent_system_prompt()
        tools = self._create_agent_mcp_tools()
        mcp_server = create_sdk_mcp_server(
            name="predicator_tools",
            version="1.0.0",
            tools=tools,
        )
        log_dir = self._get_log_dir()
        allowed_tools = self._get_agent_allowed_tools()

        kwargs: Dict[str, Any] = dict(
            system_prompt=system_prompt,
            mcp_server=mcp_server,
            log_dir=log_dir,
            model_name=self._get_agent_model_name(),
        )
        if allowed_tools is not None:
            kwargs["allowed_tools"] = allowed_tools

        self._agent_session = AgentSessionManager(**kwargs)
        if self._agent_session_id is not None:
            self._agent_session.session_id = self._agent_session_id

    def _get_log_dir(self) -> str:
        """Return the log directory, using ``_log_subdir`` class attribute."""
        base = CFG.log_file if hasattr(CFG, 'log_file') and CFG.log_file \
            else "logs"
        return os.path.join(base, self._log_subdir)

    def _query_agent_sync(self, message: str) -> List[Dict[str, Any]]:
        """Synchronous wrapper for async agent query."""
        self._ensure_agent_session()
        assert self._agent_session is not None
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import nest_asyncio
                nest_asyncio.apply()
                return loop.run_until_complete(
                    self._agent_session.query(message))
            else:
                return loop.run_until_complete(
                    self._agent_session.query(message))
        except RuntimeError:
            return asyncio.run(self._agent_session.query(message))

    def _create_agent_explorer(
        self,
        predicates: Set[Predicate],
        options: Set[ParameterizedOption],
    ) -> BaseExplorer:
        """Create an agent explorer with tool_context and agent_session."""
        self._ensure_agent_session()
        return create_explorer(
            "agent",
            predicates,
            options,
            self._types,  # type: ignore[attr-defined]
            self._action_space,  # type: ignore[attr-defined]
            self._train_tasks,  # type: ignore[attr-defined]
            tool_context=self._tool_context,
            agent_session=self._agent_session,
        )
