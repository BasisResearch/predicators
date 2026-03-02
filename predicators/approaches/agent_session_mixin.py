"""Mixin providing shared agent session infrastructure.

Extracts common code for ToolContext initialization, lazy
AgentSessionManager creation, async-to-sync bridging, and agent explorer
creation from AgentPlannerApproach and AgentAbstractionLearningApproach.
"""
import asyncio
import os
from typing import Any, Dict, List, Optional, Set

from predicators.agent_sdk.session_manager import AgentSessionManager
from predicators.agent_sdk.tools import ToolContext, create_mcp_tools, \
    get_allowed_tool_list
from predicators.explorers import create_explorer
from predicators.explorers.base_explorer import BaseExplorer
from predicators.settings import CFG
from predicators.structs import ParameterizedOption, Predicate, Task, Type


class AgentSessionMixin:
    """Mixin that provides shared agent session infrastructure.

    Subclasses must override:
      - _get_agent_model_name()
      - _get_agent_system_prompt()

    And may optionally override:
      - _get_agent_tool_names()  -- return a subset of ALL_TOOL_NAMES (None = all)
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

    def _get_agent_tool_names(self) -> Optional[List[str]]:
        """Return tool name filter.

        None means all tools; override to subset.
        """
        return None

    # ------------------------------------------------------------------ #
    # Shared implementations
    # ------------------------------------------------------------------ #

    def _ensure_agent_session(self) -> None:
        """Create the agent session manager if needed."""
        if self._agent_session is not None:
            return

        from claude_agent_sdk import create_sdk_mcp_server

        tool_names = self._get_agent_tool_names()
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
            model_name=self._get_agent_model_name(),
            allowed_tools=get_allowed_tool_list(tool_names),
        )
        if self._agent_session_id is not None:
            self._agent_session.session_id = self._agent_session_id

    def _get_log_dir(self) -> str:
        """Return the log directory, using ``_log_subdir`` class attribute."""
        if hasattr(CFG, 'log_file') and CFG.log_file:
            return CFG.log_file
        return os.path.join("logs", self._log_subdir)

    def _close_agent_session(self) -> None:
        """Close and discard the current agent session, if one exists."""
        if self._agent_session is None:
            return
        session = self._agent_session
        self._agent_session = None
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import nest_asyncio  # type: ignore[import-not-found]
                nest_asyncio.apply()
                loop.run_until_complete(session.close())
            else:
                loop.run_until_complete(session.close())
        except RuntimeError:
            asyncio.run(session.close())
        except Exception:
            pass

    def _query_agent_sync(self, message: str) -> List[Dict[str, Any]]:
        """Synchronous wrapper for async agent query."""
        self._ensure_agent_session()
        assert self._agent_session is not None
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import nest_asyncio  # type: ignore[import-not-found]
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
