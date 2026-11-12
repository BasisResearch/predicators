"""Continual agents that construct their own low-level controllers.

Both arms receive observations, the actuator interface, goals, recorded
experience and a persistent coding sandbox. Neither receives options,
predicates, helper objects or a domain simulator. The model-based arm
additionally receives an importable copy of the generic PyBulletEnv
class and its abstract BaseEnv dependency, and builds its own simulator
in Python.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional, Set

from gym.spaces import Box

from predicators.agent_sdk.play_prompts import build_minimal_play_system_prompt
from predicators.agent_sdk.primitive_policy import primitive_observation
from predicators.agent_sdk.sandbox_setup import GeneratedReference, \
    ReferenceFiles
from predicators.agent_sdk.tools.continual_tools import PRIMITIVE_TOOL_NAMES
from predicators.approaches.agent_continual_approach import \
    AgentContinualModelFreeApproach
from predicators.structs import ParameterizedOption, Predicate, Task, Type

if TYPE_CHECKING:
    from predicators.run.continual import ProtocolSession

# The predicators modules the engine wrapper loads beside the ones every
# sandbox's python may import (sandbox_guard.SANDBOX_ALLOWED_MODULES).
# The model-based arm's agent is told to import the wrapper in its own
# python (prompts/play_minimal.md), so its sandbox's guard admits them,
# and no other arm's does. The arm's tests recompute the list.
ENGINE_WRAPPER_MODULES = (
    "predicators.code_sim_learning.commands",
    "predicators.code_sim_learning.fit_space",
    "predicators.code_sim_learning.model_state",
    "predicators.code_sim_learning.utils",
    "predicators.pybullet_helpers.camera",
    "predicators.pybullet_helpers.ikfast",
    "predicators.pybullet_helpers.ikfast.load",
    "predicators.pybullet_helpers.ikfast.utils",
    "predicators.pybullet_helpers.inverse_kinematics",
    "predicators.pybullet_helpers.link",
    "predicators.pybullet_helpers.objects",
    "predicators.pybullet_helpers.real_robot_bridge",
    "predicators.pybullet_helpers.robots",
    "predicators.pybullet_helpers.robots.fetch",
    "predicators.pybullet_helpers.robots.mobile_fetch",
    "predicators.pybullet_helpers.robots.panda",
    "predicators.pybullet_helpers.robots.single_arm",
    "predicators.pybullet_helpers.settle",
    "predicators.pybullet_helpers.studio_visuals",
    "predicators.pybullet_helpers.world_gap",
)


class AgentContinualModelFreeMinimalApproach(AgentContinualModelFreeApproach):
    """Choose raw actions from experience, optionally with code as policy."""

    _save_suffix = "AgentContinualModelFreeMinimal"

    def __init__(self, initial_predicates: Set[Predicate],
                 initial_options: Set[ParameterizedOption], types: Set[Type],
                 action_space: Box, train_tasks: List[Task],
                 **kwargs: Any) -> None:
        # Remove scaffolding before the parent initializes its context. The
        # arm's knowledge boundary must hold even with conflicting CFG flags.
        del initial_predicates, initial_options
        kwargs.pop("option_model", None)
        super().__init__(set(), set(), types, action_space, train_tasks,
                         **kwargs)

    @classmethod
    def get_name(cls) -> str:
        return "agent_continual_model_free_minimal"

    def _use_gt_helpers(self) -> bool:
        return False

    def _continual_tool_names(self) -> List[str]:
        return list(PRIMITIVE_TOOL_NAMES)

    def _get_sandbox_reference_files(self) -> ReferenceFiles:
        return {}

    def _sync_tool_context(self) -> None:
        super()._sync_tool_context()
        self._tool_context.show_option_source = False
        self._tool_context.gt_options_ref_path = None

    def _play_system_prompt(self) -> str:
        return build_minimal_play_system_prompt(model_based=False)

    def _model_status(self, session: ProtocolSession) -> str:
        episodes, steps = self._episode_counts(session)
        return (f"Recorded episodes so far: {episodes} ({steps} steps), in "
                "`./data/trajectories.pkl`. Your Python files and journal "
                "persist across rounds and levels.")

    # The minimal arms keep the query heading and the sandbox CLAUDE.md
    # that every arm shares.

    def _model_status_heading(self) -> str:
        return "Model and data"

    def _get_sandbox_claude_md(self) -> Optional[str]:
        return None

    def _build_query(self, session: ProtocolSession, kind: str) -> str:
        query = super()._build_query(session, kind)
        return query + "\n\nLow-level control observation (also the input " \
            "to get_action):\n" + json.dumps(primitive_observation(session))


class AgentContinualModelBasedMinimalApproach(
        AgentContinualModelFreeMinimalApproach):
    """Build an entire simulator from the generic PyBullet base class."""

    _save_suffix = "AgentContinualModelBasedMinimal"

    @classmethod
    def get_name(cls) -> str:
        return "agent_continual_model_based_minimal"

    def _play_system_prompt(self) -> str:
        return build_minimal_play_system_prompt(model_based=True)

    def _get_sandbox_reference_files(self) -> ReferenceFiles:
        # Import the copied BaseEnv, avoiding predicators.envs' registry,
        # which imports every domain class. The sandbox guard stays intact.
        package = Path(__file__).resolve().parents[1]
        source = (package / "envs" /
                  "pybullet_env.py").read_text(encoding="utf-8")
        original = "from predicators.envs import BaseEnv\n"
        if source.count(original) != 1:
            raise ValueError("PyBulletEnv's BaseEnv import changed; update "
                             "the standalone reference binding")
        source = source.replace(
            original, "from reference.base_sim.base_env import BaseEnv\n")
        return {
            "base_sim/pybullet_env.py":
            GeneratedReference(source,
                               "predicators/envs/pybullet_env.py, its BaseEnv "
                               "import pointed at reference/base_sim",
                               imports=ENGINE_WRAPPER_MODULES),
            "base_sim/base_env.py":
            str(package / "envs" / "base_env.py"),
        }
