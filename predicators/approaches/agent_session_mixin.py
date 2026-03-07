"""Mixin providing shared agent session infrastructure.

Extracts common code for ToolContext initialization, lazy
AgentSessionManager creation, async-to-sync bridging, and agent explorer
creation from AgentPlannerApproach and AgentAbstractionLearningApproach.
"""
import asyncio
import os
from typing import Any, Dict, List, Optional, Set, Union

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
      - _get_agent_system_prompt()

    And may optionally override:
      - _get_agent_tool_names()  -- return a subset of ALL_TOOL_NAMES (None = all)
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

    def _get_agent_tool_names(self) -> Optional[List[str]]:
        """Return tool name filter.

        None means all tools; override to subset.
        """
        return None

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

        tool_names = self._get_agent_tool_names()

        if CFG.agent_sdk_use_docker_sandbox:
            from predicators.agent_sdk.docker_sandbox import \
                DockerSessionManager
            self._agent_session = DockerSessionManager(
                system_prompt=self._get_agent_system_prompt(),
                log_dir=self._get_log_dir(),
                model_name=CFG.agent_sdk_model_name,
                tool_context=self._tool_context,
                tool_names=tool_names,
                image=CFG.agent_sdk_docker_image,
                extra_reference_files=self._get_sandbox_reference_files(),
            )
        elif CFG.agent_sdk_use_local_sandbox:
            from predicators.agent_sdk.local_sandbox import \
                LocalSandboxSessionManager
            self._agent_session = LocalSandboxSessionManager(
                system_prompt=self._get_agent_system_prompt(),
                log_dir=self._get_log_dir(),
                model_name=CFG.agent_sdk_model_name,
                tool_context=self._tool_context,
                tool_names=tool_names,
                extra_reference_files=self._get_sandbox_reference_files(),
            )
        else:
            from claude_agent_sdk import create_sdk_mcp_server

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
            )

        if self._agent_session_id is not None:
            self._agent_session.session_id = self._agent_session_id

        # Save system prompt to log directory
        log_dir = self._get_log_dir()
        os.makedirs(log_dir, exist_ok=True)
        prompt_path = os.path.join(log_dir, "system_prompt.txt")
        with open(prompt_path, "w") as f:
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
