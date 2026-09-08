"""Shared interface for the RL tools (model-free now, model-based later)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol

import numpy as np

RewardFn = Callable[
    [Dict[str, np.ndarray], Dict[str, np.ndarray], np.ndarray, np.ndarray, float],
    float]

ACTION_MODES = ("xyz", "xyz_yaw")


@dataclass
class RLRequest:
    """What the agent asked for, plus per-run defaults from config."""
    reward_fn: RewardFn
    reward_source: str
    budget_interactions: int
    episode_length: int = 50
    action_mode: str = "xyz"
    control_gripper: bool = False
    workspace_half_extent: float = 0.15
    max_step_translation: float = 0.01
    max_step_yaw_deg: float = 5.0
    points_per_object: int = 32
    object_names: Optional[List[str]] = None
    out_dir: Optional[Path] = None
    seed: int = 0
    algo: str = "sac"
    algo_kwargs: Dict[str, Any] = field(default_factory=dict)
    early_stop_successes: int = 5
    early_stop_window: int = 8
    # Abort when, after 40% of the budget, neither the best per-episode max
    # reward nor the mean return improved over the last stagnation_episodes
    # episodes and no episode succeeded (typical reset-free failure: the
    # object was pushed out of the workspace box). 0 disables.
    stagnation_episodes: int = 40
    stagnation_eps: float = 0.02
    final_exec_attempts: int = 3
    reward_clip: tuple = (-10.0, 1.0)
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None


@dataclass
class RLResult:
    """What the agent gets back."""
    success: bool
    episodes: int
    interactions_used: int
    successes_during_training: int
    early_stopped: bool
    budget_exhausted: bool
    best_episode_return: float
    last_episode_return: float
    final_exec_rewards: List[float]
    final_exec_success: bool
    out_dir: Optional[str]
    policy_path: Optional[str]
    message: str = ""
    dropped: bool = False
    stagnated: bool = False
    episode_returns: List[float] = field(default_factory=list)
    episode_max_rewards: List[float] = field(default_factory=list)

    def summary(self) -> str:
        """Text for the tool result."""
        lines = [
            f"RL finished: {self.episodes} episodes, {self.interactions_used} "
            f"env interactions used by this call.",
            f"Episodes reaching reward >= 1 during training: "
            f"{self.successes_during_training}.",
            f"Best episode return {self.best_episode_return:.3f}; last "
            f"episode return {self.last_episode_return:.3f}.",
            "Early stopping triggered." if self.early_stopped else
            "Early stopping did not trigger.",
        ]
        if self.final_exec_rewards:
            lines.append(
                "Final execution of the learned policy (max reward per attempt): "
                + ", ".join(f"{r:.3f}" for r in self.final_exec_rewards)
                + (" -> reached reward 1." if self.final_exec_success else
                   " -> did not reach reward 1."))
        if self.dropped:
            lines.append("The held object was DROPPED during training; the call was aborted.")
        if self.stagnated:
            lines.append("Training STAGNATED (no reward improvement for many episodes); the call was aborted.")
        if self.budget_exhausted:
            lines.append("The overall interaction budget is exhausted.")
        if self.policy_path:
            lines.append(f"Policy saved to {self.policy_path}.")
        if self.message:
            lines.append(self.message)
        return "\n".join(lines)


class RLBackend(Protocol):
    """Anything that can optimise an agent-written particle reward."""

    def run(self, session, request: RLRequest) -> RLResult:
        """Run the algorithm in the live session and leave the sim wherever
        the final policy execution ends."""
