"""The continual protocol's scorecard: base metrics per level and per run.

See ``docs/continual-protocol.md`` section 4.4. The card records raw
counts only. Nothing here is normalised, capped, or weighted: how the
metrics aggregate into a headline number is decided later, from the
data, and is deliberately not encoded in this module.

The card is rewritten atomically after every skill invocation, reset
and level event, so a run that dies at any point leaves a valid partial
card on disk that a requeue resumes from.
"""
from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

SCHEMA_VERSION = 1


@dataclass
class EpisodeRecord:
    """One episode on a level: from a reset to a win, a game over, the next
    reset, or the end of the run."""
    index: int
    # Steps charged within this episode (the reset that opened it is
    # charged to the level, not to the episode).
    steps: int = 0
    # "in_progress" | "win" | "game_over:<reason>" | "reset" |
    # "harness_reset" | "interrupted"
    end: str = "in_progress"
    # The env evaluator's verdict on the episode, when the env defines
    # one and the episode ended in a terminal state.
    reward: Optional[float] = None
    terminated: Optional[bool] = None
    rejected: Optional[bool] = None


@dataclass
class LevelCard:
    """The base metrics of one level (section 4.4)."""
    index: int
    split: str  # "train" | "test"
    task_idx: int
    goal: List[str]
    goal_nl: str = ""
    attempted: bool = False
    won: bool = False
    won_at_step: Optional[int] = None
    # The level ended in GAME_OVER with no reset available (a test level
    # unless continual_allow_test_resets): it can no longer be won.
    lost: bool = False
    # Low-level env steps charged on the level, including reset steps.
    steps: int = 0
    resets: int = 0
    skill_invocations: int = 0
    failed_skill_invocations: int = 0
    game_overs: List[str] = field(default_factory=list)
    divergences: int = 0
    # Active seconds on the level (queue time never enters this), and
    # the part of it spent inside env calls.
    wall_clock: float = 0.0
    wall_clock_env: float = 0.0
    # Sandbox usage: sim_rollouts, fits, learn_sessions, sessions,
    # turns, llm_cost_usd. Free in step terms, recorded for the record.
    sandbox: Dict[str, float] = field(default_factory=dict)
    episodes: List[EpisodeRecord] = field(default_factory=list)
    steps_before_first_win: Optional[int] = None
    resets_before_first_win: Optional[int] = None
    # Recovery bookkeeping (section 6.6), kept apart from the agent's
    # own counts.
    preemptions: int = 0
    resumes: int = 0
    downtime: float = 0.0
    harness_resets: int = 0
    interrupted_invocations: int = 0
    started_at: Optional[float] = None
    finished_at: Optional[float] = None

    @property
    def wall_clock_sandbox(self) -> float:
        """Active seconds not spent inside env calls."""
        return max(0.0, self.wall_clock - self.wall_clock_env)

    @property
    def current_episode(self) -> Optional[EpisodeRecord]:
        """The open episode, if any."""
        if self.episodes and self.episodes[-1].end == "in_progress":
            return self.episodes[-1]
        return None

    def add_sandbox(self, key: str, delta: float) -> None:
        """Accumulate one sandbox-usage counter."""
        self.sandbox[key] = self.sandbox.get(key, 0.0) + delta


def _git_sha() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short=12", "HEAD"],
                             capture_output=True,
                             text=True,
                             timeout=10,
                             check=False)
        return out.stdout.strip() if out.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


@dataclass
class RunCard:
    """One env run's scorecard: its levels plus run metadata."""
    run_id: str
    env: str
    seed: int
    arm: str
    levels: List[LevelCard]
    step_cap: int
    wall_clock_cap: float
    config: str = ""
    git_sha: str = field(default_factory=_git_sha)
    schema: int = SCHEMA_VERSION
    started_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    # "all_levels_won" | "step_cap" | "wall_clock_cap" | "agent_ended" |
    # "level_not_won" | "level_lost" | "crash"
    end_reason: Optional[str] = None
    end_note: str = ""

    # -- Derived totals (computed, never stored as inputs) -------------

    @property
    def levels_total(self) -> int:
        """Number of levels in the run."""
        return len(self.levels)

    @property
    def levels_completed(self) -> int:
        """Number of levels won."""
        return sum(1 for lv in self.levels if lv.won)

    @property
    def total_steps(self) -> int:
        """Steps charged across all levels."""
        return sum(lv.steps for lv in self.levels)

    @property
    def total_resets(self) -> int:
        """Agent resets across all levels."""
        return sum(lv.resets for lv in self.levels)

    @property
    def total_skill_invocations(self) -> int:
        """Skill invocations across all levels."""
        return sum(lv.skill_invocations for lv in self.levels)

    @property
    def total_wall_clock(self) -> float:
        """Active seconds across all levels."""
        return sum(lv.wall_clock for lv in self.levels)

    @property
    def total_downtime(self) -> float:
        """Queue seconds across all levels."""
        return sum(lv.downtime for lv in self.levels)

    @property
    def total_llm_cost(self) -> float:
        """LLM spend in USD across all levels."""
        return sum(lv.sandbox.get("llm_cost_usd", 0.0) for lv in self.levels)

    @property
    def steps_remaining(self) -> int:
        """Steps left under the pooled cap."""
        return max(0, self.step_cap - self.total_steps)

    @property
    def is_finished(self) -> bool:
        """Whether the run has ended."""
        return self.end_reason is not None

    def current_level_index(self) -> Optional[int]:
        """The first level not yet won, or ``None`` when all are won."""
        for lv in self.levels:
            if not lv.won:
                return lv.index
        return None

    # -- Serialisation ---------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """A JSON-ready dict: the stored fields plus the derived totals."""
        d = dataclasses.asdict(self)
        d["totals"] = {
            "levels_total": self.levels_total,
            "levels_completed": self.levels_completed,
            "total_steps": self.total_steps,
            "total_resets": self.total_resets,
            "total_skill_invocations": self.total_skill_invocations,
            "total_wall_clock": self.total_wall_clock,
            "total_downtime": self.total_downtime,
            "total_llm_cost": self.total_llm_cost,
            "steps_remaining": self.steps_remaining,
        }
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RunCard":
        """Inverse of :meth:`to_dict` (the derived block is ignored)."""
        levels = []
        for lv in d["levels"]:
            episodes = [EpisodeRecord(**ep) for ep in lv.get("episodes", [])]
            lv = dict(lv)
            lv["episodes"] = episodes
            levels.append(LevelCard(**lv))
        fields = {f.name for f in dataclasses.fields(cls)}
        kwargs = {k: v for k, v in d.items() if k in fields and k != "levels"}
        return cls(levels=levels, **kwargs)

    def save(self, path: str) -> None:
        """Atomically write the card as JSON."""
        self.updated_at = time.time()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, sort_keys=True)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str) -> "RunCard":
        """Read a card written by :meth:`save`."""
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
