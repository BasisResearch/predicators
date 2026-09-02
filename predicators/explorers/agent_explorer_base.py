"""Shared plumbing for the agent explorers.

Both agent explorers query a Claude agent session for a plan and roll it
out in the real environment; they differ in what the agent is given (the
model-based explorer exposes the learned belief simulator through tools,
the model-free one only the task description). This base class holds the
common session wiring: construction, the trajectory summary shown to the
agent, final-text extraction, and the random-options fallback.
"""

from typing import Any, Dict, List, Optional, Set

from gym.spaces import Box

from predicators import utils
from predicators.agent_sdk.learn_prompts import render_world_model_notes_block
from predicators.agent_sdk.response_parser import extract_final_text
from predicators.agent_sdk.session_manager import SessionManagerProtocol
from predicators.agent_sdk.sketch_prompts import summarize_trajectories
from predicators.agent_sdk.tools import ToolContext
from predicators.explorers.base_explorer import BaseExplorer
from predicators.structs import Action, ExplorationStrategy, \
    ParameterizedOption, Predicate, State, Task, Type


class AgentExplorerBase(BaseExplorer):
    """Base class for explorers that query a Claude agent session."""

    def __init__(self, predicates: Set[Predicate],
                 options: Set[ParameterizedOption], types: Set[Type],
                 action_space: Box, train_tasks: List[Task],
                 max_steps_before_termination: int, tool_context: ToolContext,
                 agent_session: SessionManagerProtocol) -> None:
        super().__init__(predicates, options, types, action_space, train_tasks,
                         max_steps_before_termination)
        self._tool_context = tool_context
        self._agent_session = agent_session

    def _agent_tool_names(self) -> Optional[List[str]]:
        """Return tool names exposed by the current session, if any."""
        return getattr(self._agent_session, "tool_names", None)

    def _random_options_fallback(self) -> ExplorationStrategy:
        """Fall back to random option sampling."""

        def fallback_policy(state: State) -> Action:
            del state
            raise utils.RequestActPolicyFailure(
                "Random option sampling failed!")

        policy = utils.create_random_option_policy(self._options, self._rng,
                                                   fallback_policy)
        return policy, lambda _: False

    def _build_trajectory_summary(self) -> str:
        """Summarize trajectory data for the agent."""
        all_trajs = (self._tool_context.offline_trajectories +
                     self._tool_context.online_trajectories)
        return summarize_trajectories(all_trajs, self._predicates)

    def _world_model_notes_block(self) -> str:
        """The natural-language world model quoted into the explore prompt
        (empty unless the approach keeps one)."""
        return render_world_model_notes_block(
            self._tool_context.world_model_notes,
            self._tool_context.world_model_notes_path)

    @staticmethod
    def _extract_option_plan_text(responses: List[Dict[str, Any]]) -> str:
        """Extract plan text from the last assistant text response."""
        return extract_final_text(responses)
