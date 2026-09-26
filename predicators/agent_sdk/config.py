"""Typed configuration surfaces for the agent_sdk package.

Each dataclass groups the ``CFG`` flags one agent-sdk concern reads,
under clean field names. ``from_cfg()`` is called at USE time (handler
entry / session construction), never cached at import time, so tests
that mutate settings via ``utils.reset_config`` keep working.

The experiment flags keep their historical names (``agent_bilevel_*``,
``agent_planner_*``, ...) for experiment-yaml compatibility; the
``from_cfg()`` classmethods below are the single place those old flag
names map to the clean field names.
"""
from dataclasses import dataclass

from predicators.settings import CFG


@dataclass(frozen=True)
class SessionConfig:
    """Agent session construction: model, budgets, and sandbox choice.

    Consumed by the three session managers and by
    ``AgentSessionMixin._ensure_agent_session``, which builds one
    instance per session and passes it to whichever manager it
    constructs.
    """
    model_name: str
    reasoning_effort: str
    max_turns: int
    max_buffer_size: int
    agent_timeout: int
    use_local_sandbox: bool
    use_scratchpad: bool

    @classmethod
    def from_cfg(cls) -> "SessionConfig":
        """Read the session flags from the live ``CFG``."""
        # Flags keep their names for experiment-yaml compatibility.
        return cls(
            model_name=CFG.agent_sdk_model_name,
            reasoning_effort=CFG.agent_sdk_reasoning_effort,
            max_turns=CFG.agent_sdk_max_agent_turns_per_iteration,
            max_buffer_size=CFG.agent_sdk_max_buffer_size,
            agent_timeout=CFG.agent_sdk_agent_timeout,
            use_local_sandbox=CFG.agent_sdk_use_local_sandbox,
            use_scratchpad=CFG.agent_planner_use_scratchpad,
        )


@dataclass(frozen=True)
class RefinementConfig:
    """Plan-sketch refinement settings the probe's ``refine`` reads."""
    ground_samplers: bool
    max_samples_per_step: int

    @classmethod
    def from_cfg(cls) -> "RefinementConfig":
        """Read the refinement flags from the live ``CFG``."""
        # Flags keep their names for experiment-yaml compatibility.
        return cls(
            ground_samplers=CFG.agent_bilevel_ground_samplers,
            max_samples_per_step=CFG.agent_bilevel_max_samples_per_step,
        )


@dataclass(frozen=True)
class ToolSurfaceConfig:
    """Which optional tools a session offers, and their surface knobs.

    Consumed by the tool builders (descriptions baked at build time) and
    image sizing.
    """
    use_base_simulator: bool
    python_call_timeout: float
    image_max_px: int

    @classmethod
    def from_cfg(cls) -> "ToolSurfaceConfig":
        """Read the tool-surface flags from the live ``CFG``."""
        # Flags keep their names for experiment-yaml compatibility.
        return cls(
            use_base_simulator=CFG.agent_planner_use_base_simulator,
            python_call_timeout=(CFG.agent_sdk_python_call_timeout),
            image_max_px=CFG.agent_sdk_image_max_px,
        )
