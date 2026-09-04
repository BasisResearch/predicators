"""The continual protocol loop: one agent, one environment, a scorecard.

See ``docs/continual-protocol.md``. The run plays an env's levels in
order (its train tasks, then its test tasks). The only primitive is the
low-level env step; ``env.reset()`` is a step and a reset; skills are an
agent-side library invoked through the same session. Nothing in the
sandbox is charged. The harness never judges intent: it counts, records
and enforces the caps, and the ``RunCard`` is the result.

``ProtocolSession`` is the API a controller or an agent tool may call
(section 5.1). ``ContinualRun`` owns the level list, the episode runner,
the scorecard, the recordings and the resume path (section 6.6).
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from predicators import utils
from predicators.approaches import ApproachFailure, ApproachTimeout, \
    BaseApproach
from predicators.envs import BaseEnv
from predicators.run.episode import EpisodeOver, EpisodeRunner, EpisodeState, \
    InvocationOutcome, StepOutcome
from predicators.run.recording import LevelRecording, states_close
from predicators.run.scorecard import EpisodeRecord, LevelCard, RunCard
from predicators.settings import CFG
from predicators.structs import Action, Dataset, EnvironmentTask, \
    EpisodeEvaluation, GroundAtom, ParameterizedOption, Predicate, State, \
    Task, _Option

LEVEL_ORDERS = ("train_then_test", "train_only", "test_only")


class RunEnded(Exception):
    """The run is over: a cap was hit, or the controller ended it."""

    def __init__(self, reason: str, note: str = "") -> None:
        super().__init__(reason)
        self.reason = reason
        self.note = note


class LevelOver(Exception):
    """A charged call was made on a level that is over (won or lost)."""


class LevelAlreadyWon(LevelOver):
    """A charged call was made on a level that is already won."""


class LevelLost(LevelOver):
    """A charged call was made on a level that ended in GAME_OVER with no reset
    available (section 4.6)."""


class ResetUnavailable(Exception):
    """The agent asked to reset a level that has no resets (section 4.6).

    Nothing is charged.
    """


@dataclass(frozen=True)
class LevelSpec:
    """One level: a task of the env, with the goal the arm is shown."""
    index: int
    split: str
    task_idx: int
    env_task: EnvironmentTask
    task: Task

    @property
    def goal_strs(self) -> List[str]:
        """The goal atoms as sorted strings."""
        return sorted(str(a) for a in self.task.goal)


@dataclass(frozen=True)
class Ledger:
    """The pacing signal every tool result carries (section 5.1)."""
    level_index: int
    levels_total: int
    level_steps: int
    level_resets: int
    run_steps: int
    run_resets: int
    step_cap: int
    active_seconds: float
    wall_clock_cap_seconds: float
    # False on a level without resets (a test level, unless
    # continual_allow_test_resets): GAME_OVER ends it, lost.
    resets_allowed: bool = True

    @property
    def steps_remaining(self) -> int:
        """Steps left under the pooled cap."""
        return max(0, self.step_cap - self.run_steps)

    def footer(self) -> str:
        """One line for tool results."""
        hours = self.active_seconds / 3600.0
        cap_hours = self.wall_clock_cap_seconds / 3600.0
        return (f"[ledger] level {self.level_index + 1}/{self.levels_total}; "
                f"steps {self.level_steps} this level, {self.run_steps} "
                f"this run, {self.steps_remaining} remaining; resets "
                f"{self.level_resets} this level, {self.run_resets} this "
                f"run{'' if self.resets_allowed else ' (none on this level)'}"
                f"; active {hours:.2f}/{cap_hours:.0f} h")


@dataclass(frozen=True)
class ProtocolObservation:
    """The analogue of ARC's frame (section 5.2)."""
    state: EpisodeState
    reason: str
    level: LevelSpec
    frame: State
    atoms: Set[GroundAtom]
    evaluation: Optional[EpisodeEvaluation]
    ledger: Ledger
    skills: List[ParameterizedOption]


@dataclass(frozen=True)
class InvocationResult:
    """A skill invocation plus the protocol's view of it."""
    outcome: InvocationOutcome
    expected: Set[GroundAtom]
    missing: Set[GroundAtom]
    render: Optional[str]
    # Atoms the caller expected to be absent that hold afterwards.
    present: Set[GroundAtom] = field(default_factory=set)

    @property
    def diverged(self) -> bool:
        """Whether the observed outcome differs from the expected one."""
        return bool(self.missing) or bool(self.present)

    @property
    def status(self) -> str:
        """The controller's termination status."""
        return self.outcome.status


@dataclass(frozen=True)
class PolicyRunOutcome:
    """What running a closed-loop policy through the session produced."""
    steps: int
    invocations: int
    # "succeeded" (the policy ran out of plan), "failed" (it raised) or
    # "interrupted" (the episode ended: WIN or GAME_OVER).
    status: str
    reason: str
    episode_state: EpisodeState


def build_levels(env: BaseEnv, arm: str) -> List[LevelSpec]:
    """The env's levels in protocol order (section 4.1)."""
    order = CFG.continual_levels
    assert order in LEVEL_ORDERS, order
    specs: List[LevelSpec] = []
    if order in ("train_then_test", "train_only"):
        for i, env_task in enumerate(env.get_train_tasks()):
            specs.append(
                LevelSpec(len(specs), "train", i, env_task, env_task.task))
    if order in ("train_then_test", "test_only"):
        for i, env_task in enumerate(env.get_test_tasks()):
            # The oracle plans on the env's own goal; every other arm
            # sees the alternative goal, as run_testing does.
            shown = env_task if arm == "oracle" else \
                env_task.replace_goal_with_alt_goal()
            specs.append(LevelSpec(len(specs), "test", i, env_task,
                                   shown.task))
    return specs


class ProtocolSession:
    """The env-facing API of the protocol (section 5.1).

    Every charged call counts, records, checks the caps and may raise
    ``RunEnded``; ``LevelOver`` (``LevelAlreadyWon``, ``LevelLost``),
    ``ResetUnavailable`` and ``EpisodeOver`` are the protocol errors a
    caller must handle.
    """

    def __init__(self, run: "ContinualRun") -> None:
        self._run = run

    # -- env.* ---------------------------------------------------------------

    def observe(self) -> ProtocolObservation:
        """The current observation.

        Free.
        """
        return self._run.observation()

    def step(self, action: Action) -> StepOutcome:
        """One primitive action.

        One step.
        """
        return self._run.step(action)

    def reset(self, note: str = "") -> ProtocolObservation:
        """Restart the current level.

        One step plus one reset. Refused with ``ResetUnavailable`` on a
        level without resets (see ``resets_allowed``); nothing is
        charged then.
        """
        return self._run.reset(note)

    def end_run(self, note: str = "") -> None:
        """End the run for this env."""
        raise RunEnded("agent_ended", note)

    # -- skills.* ------------------------------------------------------------

    def list_skills(self) -> List[ParameterizedOption]:
        """The skill library."""
        return self._run.skills

    def invoke(
            self,
            option: _Option,
            expected: Optional[Set[GroundAtom]] = None,
            note: str = "",
            expected_absent: Optional[Set[GroundAtom]] = None
    ) -> InvocationResult:
        """One skill invocation."""
        return self._run.invoke(option, expected or set(), note,
                                expected_absent or set())

    def execute_plan(
        self,
        plan: Sequence[_Option],
        expected: Optional[Sequence[Set[GroundAtom]]] = None,
        stop_on_divergence: bool = True,
        note: str = "",
        expected_absent: Optional[Sequence[Set[GroundAtom]]] = None
    ) -> List[InvocationResult]:
        """Execute a plan by invoking its skills in order (section 5.1)."""
        results: List[InvocationResult] = []
        for i, option in enumerate(plan):
            exp = set(expected[i]) if expected is not None else set()
            absent = set(expected_absent[i]) if expected_absent else set()
            result = self.invoke(option, exp, note, absent)
            results.append(result)
            if result.status != "succeeded":
                break
            if stop_on_divergence and result.diverged:
                break
        return results

    # -- Level data for the arm ---------------------------------------------

    @property
    def levels(self) -> List[LevelSpec]:
        """The run's level list."""
        return self._run.levels

    @property
    def level_index(self) -> int:
        """The index of the level in progress."""
        return self._run.level_index

    @property
    def resets_allowed(self) -> bool:
        """Whether ``reset`` is available on the level in progress: always on a
        train level, on a test level only under ``continual_allow_test_resets``
        (section 4.6)."""
        return self._run.resets_allowed()

    def level_card(self) -> LevelCard:
        """The scorecard of the level in progress."""
        return self._run.level_card()

    def level_episodes(self) -> List[Dict[str, Any]]:
        """The episodes of the level in progress: ``states``, ``actions``,
        ``end``, ``reward``, ``terminated``, ``index``."""
        return self._run.level_episodes()

    def previous_level_episodes(self,
                                level_index: int) -> List[Dict[str, Any]]:
        """A finished level's episodes, read from its recording."""
        return self._run.previous_level_episodes(level_index)

    def index_entries(self) -> List[Dict[str, Any]]:
        """The recording index of the level in progress."""
        return self._run.index_entries()

    def run_policy(self,
                   policy: Callable[[State], Action],
                   note: str = "") -> PolicyRunOutcome:
        """Step a closed-loop policy until it ends or the episode does.

        Skill invocations are detected from the option each action
        carries, so a policy built from an option plan is recorded
        exactly as the same plan sent through ``execute_plan``.
        """
        return self._run.run_policy(policy, note)

    # -- Bookkeeping hooks for the sandbox side ------------------------------

    def record_sandbox(self, key: str, delta: float = 1.0) -> None:
        """Accumulate a sandbox-usage counter on the current level."""
        self._run.record_sandbox(key, delta)

    def render(self, tag: str) -> Optional[str]:
        """Save a render of the current state; returns its path."""
        return self._run.render(tag)

    @property
    def abstract_predicates(self) -> Set[Predicate]:
        """Predicates used for abstraction and divergence checks."""
        return self._run.predicates

    @abstract_predicates.setter
    def abstract_predicates(self, preds: Set[Predicate]) -> None:
        self._run.predicates = preds


class ContinualRun:
    """One env run under the continual protocol."""

    def __init__(
            self,
            env: BaseEnv,
            approach: BaseApproach,
            controller: Any,
            skills: Optional[Sequence[ParameterizedOption]] = None) -> None:
        self._env = env
        self._approach = approach
        self._controller = controller
        self._arm = approach.get_name()
        self._run_id = utils.get_config_path_str()
        self._levels = build_levels(env, self._arm)
        if skills is None:
            skills = _skill_library(env, approach)
        self._skills = sorted(skills, key=lambda o: o.name)
        self._predicates: Set[Predicate] = set(env.predicates)
        self._card_path = os.path.join(CFG.continual_scorecards_dir,
                                       f"{self._run_id}.json")
        self._rec_root = os.path.join(CFG.continual_recordings_dir,
                                      self._run_id)
        self._card = self._load_or_new_card()
        self._session = ProtocolSession(self)
        self._runner: Optional[EpisodeRunner] = None
        self._level_env: Optional[BaseEnv] = None
        self._fresh_env: Optional[BaseEnv] = None
        self._recording: Optional[LevelRecording] = None
        self._level_episodes: List[Dict[str, Any]] = []
        self._current: Optional[int] = None
        self._replaying = False
        # While a skill runs, terminal index entries wait until after the
        # invocation's own entry so the index reads in order.
        self._in_invocation = False
        self._pending_terminal: Optional[Dict[str, Any]] = None
        self._tick = time.time()
        self._env_seconds_at_tick = 0.0
        self._steps_since_flush = 0

    # -- Public state ----------------------------------------------------

    @property
    def card(self) -> RunCard:
        """The scorecard."""
        return self._card

    @property
    def card_path(self) -> str:
        """Where the scorecard is written."""
        return self._card_path

    @property
    def recordings_dir(self) -> str:
        """The run's recordings root."""
        return self._rec_root

    @property
    def session(self) -> ProtocolSession:
        """The session handed to the controller."""
        return self._session

    @property
    def skills(self) -> List[ParameterizedOption]:
        """The skill library."""
        return list(self._skills)

    @skills.setter
    def skills(self, options: Sequence[ParameterizedOption]) -> None:
        self._skills = sorted(options, key=lambda o: o.name)

    @property
    def predicates(self) -> Set[Predicate]:
        """Predicates used for abstraction and divergence checks."""
        return set(self._predicates)

    @predicates.setter
    def predicates(self, preds: Set[Predicate]) -> None:
        self._predicates = set(preds)
        if self._runner is not None:
            self._runner.predicates = self._predicates

    @property
    def levels(self) -> List[LevelSpec]:
        """The level list."""
        return list(self._levels)

    # -- The loop --------------------------------------------------------

    def run(self) -> RunCard:
        """Play the levels in order until the run ends."""
        logging.info(
            "[Continual] run %s: %d levels, step cap %d, wall-clock cap "
            "%.0f h, scorecard %s", self._run_id, len(self._levels),
            self._card.step_cap, self._card.wall_clock_cap / 3600.0,
            self._card_path)
        try:
            while True:
                k = self._card.current_level_index()
                if k is None:
                    raise RunEnded("all_levels_won")
                self._check_caps()
                self._begin_level(k)
                try:
                    self._controller.play_level(self._session)
                finally:
                    self._end_level(k)
                lv = self._card.levels[k]
                if lv.lost:
                    raise RunEnded(
                        "level_lost", f"level {k + 1} ended in GAME_OVER "
                        "with no reset available")
                if not lv.won:
                    raise RunEnded(
                        "level_not_won",
                        "the controller returned without winning the level")
        except RunEnded as e:
            self._finish(e.reason, e.note)
        except BaseException:
            # Leave the card resumable: no end reason, latest counts.
            self._flush()
            raise
        return self._card

    # -- Session operations ---------------------------------------------

    def observation(self) -> ProtocolObservation:
        """The protocol observation of the current level."""
        runner, lv = self._require_level()
        frame = runner.observation()
        evaluation = None
        if runner.episode_state is not EpisodeState.NOT_FINISHED:
            evaluation = runner.evaluate()
        return ProtocolObservation(
            state=runner.episode_state,
            reason=runner.reason,
            level=self._levels[lv.index],
            frame=frame,
            atoms=utils.abstract(frame, self._env.predicates),
            evaluation=evaluation,
            ledger=self.ledger(),
            skills=self.skills,
        )

    def ledger(self) -> Ledger:
        """The current ledger."""
        _, lv = self._require_level()
        return Ledger(
            level_index=lv.index,
            levels_total=self._card.levels_total,
            level_steps=lv.steps,
            level_resets=lv.resets,
            run_steps=self._card.total_steps,
            run_resets=self._card.total_resets,
            step_cap=self._card.step_cap,
            active_seconds=self._active_seconds(),
            wall_clock_cap_seconds=self._card.wall_clock_cap,
            resets_allowed=self.resets_allowed(),
        )

    def step(self, action: Action) -> StepOutcome:
        """One charged primitive step."""
        runner, _ = self._require_open_level()
        # The runner's step listener charges and records the step.
        outcome = runner.step(action)
        if self._steps_since_flush >= CFG.continual_flush_every_steps:
            self._flush()
        self._check_caps()
        return outcome

    def reset(self, note: str = "", by: str = "agent") -> ProtocolObservation:
        """A charged reset of the current level."""
        runner, lv = self._require_open_level()
        if by == "agent" and not self.resets_allowed():
            raise ResetUnavailable(
                f"level {lv.index + 1} ({self._levels[lv.index].split}) has "
                "no resets: the level ends with its episode")
        self._close_episode("reset" if by == "agent" else by)
        runner.finish()
        if by == "agent":
            lv.steps += 1
            lv.resets += 1
        else:
            lv.harness_resets += 1
        spec = self._levels[lv.index]
        runner.reset(spec.split, spec.task_idx)
        self._open_episode(by)
        render = self.render(f"ep{self._episode_index():03d}_reset")
        self._index({
            "event": "reset",
            "by": by,
            "episode": self._episode_index(),
            "note": note,
            "state": runner.episode_state.value,
            "level_steps": lv.steps,
            "render": render,
        })
        self._flush()
        self._check_caps()
        return self.observation()

    def invoke(
            self,
            option: _Option,
            expected: Set[GroundAtom],
            note: str,
            expected_absent: Optional[Set[GroundAtom]] = None
    ) -> InvocationResult:
        """One charged skill invocation with an optional expected outcome."""
        runner, lv = self._require_open_level()
        self._in_invocation = True
        try:
            outcome = runner.run_option(option)
        finally:
            self._in_invocation = False
        lv.skill_invocations += 1
        if outcome.status == "failed":
            lv.failed_skill_invocations += 1
        atoms_after = runner.abstract(outcome.observation)
        missing = set(expected) - atoms_after
        present = set(expected_absent or set()) & atoms_after
        if missing or present:
            lv.divergences += 1
        render = self.render(f"ep{self._episode_index():03d}_s{lv.steps:06d}_"
                             f"{option.name}")
        self._index({
            "event":
            "invoke",
            "episode":
            self._episode_index(),
            "skill":
            option.simple_str(),
            "params": [float(v) for v in option.params],
            "status":
            outcome.status,
            "reason":
            outcome.reason,
            "steps":
            outcome.steps,
            "start_step":
            outcome.start_step,
            "end_step":
            outcome.end_step,
            "level_steps":
            lv.steps,
            "expected":
            sorted(str(a) for a in expected),
            "expected_absent":
            sorted(str(a) for a in expected_absent or ()),
            "missing":
            sorted(str(a) for a in missing),
            "present":
            sorted(str(a) for a in present),
            "atoms":
            sorted(str(a) for a in atoms_after),
            "env_atoms":
            sorted(
                str(a) for a in utils.abstract(outcome.observation,
                                               self._env.predicates)),
            "state":
            outcome.episode_state.value,
            "episode_reason":
            outcome.reason,
            "note":
            note,
            "render":
            render,
        })
        if self._pending_terminal is not None:
            self._index(self._pending_terminal)
            self._pending_terminal = None
        self._flush()
        self._check_caps()
        return InvocationResult(outcome, set(expected), missing, render,
                                present)

    def run_policy(self, policy: Callable[[State], Action],
                   note: str) -> PolicyRunOutcome:
        """Charged execution of a closed-loop policy (see the session)."""
        runner, lv = self._require_open_level()
        if runner.episode_state is not EpisodeState.NOT_FINISHED:
            raise EpisodeOver(f"episode is {runner.episode_state.value} "
                              f"({runner.reason}); only reset is valid")
        run_start = lv.steps
        invocations = 0
        current: Optional[_Option] = None
        inv_start = lv.steps
        status, reason = "succeeded", ""
        self._in_invocation = True
        try:
            while _episode_open(runner):
                state = runner.observation()
                try:
                    act = policy(state)
                except (utils.OptionExecutionFailure, ApproachFailure,
                        ApproachTimeout) as e:
                    info = getattr(e, "info", None) or {}
                    exhausted = bool(info.get("plan_exhausted")) or \
                        "exhausted" in str(e).lower()
                    status = "succeeded" if exhausted else "failed"
                    reason = str(e.args[0]) if e.args else ""
                    break
                option = act.get_option() if act.has_option() else None
                if option is not current:
                    if current is not None:
                        self._policy_invocation_entry(current, "succeeded", "",
                                                      inv_start, lv.steps,
                                                      note)
                    current = option
                    inv_start = lv.steps
                    invocations += 1
                    lv.skill_invocations += 1
                runner.step(act)
        finally:
            self._in_invocation = False
        if not _episode_open(runner):
            status, reason = "interrupted", runner.reason
        if current is not None:
            last_status = status
            if status == "interrupted" and \
                    current.terminal(runner.observation()):
                last_status = "succeeded"
            if last_status == "failed":
                lv.failed_skill_invocations += 1
            self._policy_invocation_entry(current, last_status, reason,
                                          inv_start, lv.steps, note)
        if self._pending_terminal is not None:
            self._index(self._pending_terminal)
            self._pending_terminal = None
        self._flush()
        self._check_caps()
        return PolicyRunOutcome(lv.steps - run_start, invocations, status,
                                reason, runner.episode_state)

    def _policy_invocation_entry(self, option: _Option, status: str,
                                 reason: str, start: int, end: int,
                                 note: str) -> None:
        runner, lv = self._require_level()
        render = self.render(
            f"ep{self._episode_index():03d}_s{lv.steps:06d}_{option.name}")
        self._index({
            "event":
            "invoke",
            "via":
            "policy",
            "episode":
            self._episode_index(),
            "skill":
            option.simple_str(),
            "params": [float(v) for v in option.params],
            "status":
            status,
            "reason":
            reason,
            "steps":
            end - start,
            "start_step":
            start,
            "end_step":
            end,
            "level_steps":
            lv.steps,
            "expected": [],
            "missing": [],
            "atoms":
            sorted(str(a) for a in runner.abstract(runner.observation())),
            "state":
            runner.episode_state.value,
            "episode_reason":
            runner.reason,
            "note":
            note,
            "render":
            render,
        })

    def record_sandbox(self, key: str, delta: float) -> None:
        """Accumulate a sandbox-usage counter on the current level and write
        the card: sandbox work happens between env events, and the card on disk
        is what a viewer reads to tell a working run from a stalled one."""
        _, lv = self._require_level()
        lv.add_sandbox(key, delta)
        self._card.save(self._card_path)

    @property
    def level_index(self) -> int:
        """The index of the level in progress."""
        _, lv = self._require_level()
        return lv.index

    def level_card(self) -> LevelCard:
        """The scorecard of the level in progress."""
        _, lv = self._require_level()
        return lv

    def level_episodes(self) -> List[Dict[str, Any]]:
        """The in-memory episodes of the level in progress, joined with their
        scorecard records."""
        _, lv = self._require_level()
        records = {ep.index: ep for ep in lv.episodes}
        out = []
        for ep in self._level_episodes:
            rec = records.get(ep["episode"])
            out.append({
                "index":
                ep["episode"],
                "states":
                list(ep["states"]),
                "actions":
                list(ep["actions"]),
                "end":
                ep.get("end", "in_progress"),
                "reward":
                rec.reward if rec is not None else None,
                "terminated":
                rec.terminated if rec is not None else None,
            })
        return out

    def previous_level_episodes(self,
                                level_index: int) -> List[Dict[str, Any]]:
        """A finished level's episodes from its recording (actions lose their
        skill labels; prefer the arm's own memory when it has it)."""
        path = os.path.join(self._rec_root, f"L{level_index + 1:02d}")
        if not os.path.isdir(path):
            return []
        rec = LevelRecording(path)
        try:
            episodes = rec.read_episodes()
        finally:
            rec.close()
        records = {
            ep.index: ep
            for ep in self._card.levels[level_index].episodes
        }
        out = []
        for ep in episodes:
            r = records.get(ep["episode"])
            out.append({
                "index":
                ep["episode"],
                "states":
                list(ep["states"]),
                "actions": [
                    Action(np.array(a["arr"], dtype=np.float32))
                    for a in ep["actions"]
                ],
                "end":
                ep.get("end", "in_progress"),
                "reward":
                r.reward if r is not None else None,
                "terminated":
                r.terminated if r is not None else None,
            })
        return out

    def index_entries(self) -> List[Dict[str, Any]]:
        """The recording index of the level in progress."""
        if self._recording is None:
            return []
        return self._recording.read_index()

    def render(self, tag: str) -> Optional[str]:
        """Save a render of the current state if rendering is on."""
        if not CFG.continual_render or self._recording is None or \
                self._level_env is None:
            return None
        try:
            frames = self._level_env.render()
        except Exception as e:  # pylint: disable=broad-except
            logging.debug("[Continual] render skipped: %s", e)
            return None
        if not frames:
            return None
        return self._recording.save_render(tag, frames[0])

    # -- Level lifecycle -------------------------------------------------

    def _begin_level(self, k: int) -> None:
        spec = self._levels[k]
        lv = self._card.levels[k]
        self._current = k
        self._level_env = self._env
        self._fresh_env = None
        if CFG.test_fresh_env_per_episode:
            fresh = self._env.make_fresh_test_instance()
            if fresh is not None:
                self._fresh_env = fresh
                self._level_env = fresh
        self._runner = EpisodeRunner(
            self._level_env,
            horizon=CFG.horizon,
            max_option_steps=CFG.max_num_steps_option_rollout,
            predicates=self._predicates)
        self._runner.add_step_listener(self._on_runner_step)
        self._recording = LevelRecording(
            os.path.join(self._rec_root, f"L{k + 1:02d}"))
        self._level_episodes = []
        self._tick = time.time()
        self._env_seconds_at_tick = 0.0
        if lv.attempted and self._recording.load_checkpoint() is not None:
            self._restore(k)
            return
        lv.attempted = True
        lv.started_at = time.time()
        logging.info("[Continual] level %d/%d begins: %s task %d, goal %s",
                     k + 1, len(self._levels), spec.split, spec.task_idx,
                     spec.goal_strs)
        self._runner.reset(spec.split, spec.task_idx)
        lv.episodes = []
        self._open_episode("level_start")
        render = self.render(f"ep{self._episode_index():03d}_start")
        self._index({
            "event": "level_start",
            "episode": 0,
            "split": spec.split,
            "task_idx": spec.task_idx,
            "goal": spec.goal_strs,
            "goal_nl": spec.task.goal_nl or "",
            "state": self._runner.episode_state.value,
            "render": render,
        })
        self._flush()

    def _restore(self, k: int) -> None:
        """Rebuild the level's live episode from its recording (6.6)."""
        assert self._runner is not None and self._recording is not None
        spec = self._levels[k]
        lv = self._card.levels[k]
        ckpt = self._recording.load_checkpoint()
        assert ckpt is not None
        episode, action_arrs = self._recording.read_current_episode()
        now = time.time()
        lv.resumes += 1
        lv.preemptions += 1
        lv.downtime += max(0.0, now - float(ckpt["ts"]))
        logging.info(
            "[Continual] level %d resumes: replaying %d recorded steps of "
            "episode %d (downtime %.0f s)", k + 1, len(action_arrs), episode,
            now - float(ckpt["ts"]))
        self._replaying = True
        try:
            self._runner.reset(spec.split, spec.task_idx)
            actions = [
                Action(np.array(a, dtype=np.float32)) for a in action_arrs
            ]
            self._runner.replay(actions)
        finally:
            self._replaying = False
        replayed = self._runner.num_steps
        state_ok = states_close(self._runner.observation(), ckpt["state"])
        ok = replayed == len(action_arrs) and state_ok
        # Rebuild the in-memory episode list from the pickle so the
        # level's data stays complete across the resume.
        self._level_episodes = self._reload_episodes()
        if ok:
            logging.info(
                "[Continual] replay verified: %d steps, state "
                "matches the checkpoint", replayed)
            self._index({
                "event": "resume",
                "episode": episode,
                "replayed_steps": replayed,
                "verified": True,
                "downtime": lv.downtime,
                "state": self._runner.episode_state.value,
            })
            self._flush()
            return
        logging.warning(
            "[Continual] replay diverged (replayed %d of %d steps, state "
            "match %s): harness reset of level %d", replayed, len(action_arrs),
            state_ok, k + 1)
        self._index({
            "event": "resume",
            "episode": episode,
            "replayed_steps": replayed,
            "verified": False,
            "downtime": lv.downtime,
        })
        self.reset("replay diverged after a resume", by="harness_reset")

    def _end_level(self, k: int) -> None:
        lv = self._card.levels[k]
        if (lv.won or lv.lost) and lv.finished_at is None:
            lv.finished_at = time.time()
        self._flush()
        if self._recording is not None:
            self._recording.close()
        if self._fresh_env is not None:
            self._fresh_env.dispose()
        self._fresh_env = None
        self._level_env = None
        self._runner = None
        self._recording = None
        self._current = None

    def _finish(self, reason: str, note: str) -> None:
        self._card.end_reason = reason
        self._card.end_note = note
        self._card.finished_at = time.time()
        self._card.save(self._card_path)
        logging.info(
            "[Continual] run ended: %s%s. Levels won %d/%d, steps %d, "
            "resets %d, active %.0f s, downtime %.0f s.", reason,
            f" ({note})" if note else "", self._card.levels_completed,
            self._card.levels_total, self._card.total_steps,
            self._card.total_resets, self._card.total_wall_clock,
            self._card.total_downtime)

    # -- Step accounting -------------------------------------------------

    def _on_runner_step(self, action: Action, outcome: StepOutcome) -> None:
        """Runner listener: every applied step is logged; charged unless it is
        a replay."""
        if self._replaying:
            return
        self._after_step(action, outcome)

    def _after_step(self, action: Action, outcome: StepOutcome) -> None:
        runner, lv = self._require_level()
        ep = lv.current_episode
        if ep is None:
            # A step outside an open episode cannot happen through the
            # session; guard anyway so the card never goes inconsistent.
            ep = EpisodeRecord(self._episode_index())
            lv.episodes.append(ep)
        lv.steps += 1
        ep.steps += 1
        self._steps_since_flush += 1
        assert self._recording is not None
        self._recording.append_step(ep.index, ep.steps - 1, action)
        self._level_episodes[-1]["actions"].append(action)
        self._level_episodes[-1]["states"].append(outcome.observation)
        if outcome.state is EpisodeState.WIN:
            self._on_win(runner, lv)
        elif outcome.state is EpisodeState.GAME_OVER:
            self._on_game_over(runner, lv)

    def _on_win(self, runner: EpisodeRunner, lv: LevelCard) -> None:
        lv.won = True
        lv.won_at_step = lv.steps
        if lv.steps_before_first_win is None:
            lv.steps_before_first_win = lv.steps
            lv.resets_before_first_win = lv.resets
        self._close_episode("win", runner.evaluate())
        render = self.render(f"ep{self._episode_index():03d}_win")
        self._terminal_index({
            "event": "win",
            "episode": self._episode_index(),
            "level_steps": lv.steps,
            "render": render,
        })
        logging.info("[Continual] level %d WON after %d steps and %d resets",
                     lv.index + 1, lv.steps, lv.resets)
        self._flush()

    def _on_game_over(self, runner: EpisodeRunner, lv: LevelCard) -> None:
        reason = runner.reason
        lv.game_overs.append(reason)
        self._close_episode(f"game_over:{reason}", runner.evaluate())
        # With no reset available the level is over: lost (4.6).
        lost = not self.resets_allowed()
        if lost:
            lv.lost = True
            lv.finished_at = time.time()
        render = self.render(f"ep{self._episode_index():03d}_game_over")
        self._terminal_index({
            "event": "game_over",
            "episode": self._episode_index(),
            "reason": reason,
            "level_steps": lv.steps,
            "level_over": lost,
            "render": render,
        })
        logging.info(
            "[Continual] level %d GAME_OVER (%s) at level step %d%s",
            lv.index + 1, reason, lv.steps,
            "; the level has no resets, so it is lost" if lost else "")
        self._flush()

    # -- Episode records -------------------------------------------------

    def _episode_index(self) -> int:
        _, lv = self._require_level()
        return lv.episodes[-1].index if lv.episodes else 0

    def _open_episode(self, by: str) -> None:
        runner, lv = self._require_level()
        assert self._recording is not None
        index = (lv.episodes[-1].index + 1) if lv.episodes else 0
        lv.episodes.append(EpisodeRecord(index))
        self._recording.begin_episode(index, by)
        self._level_episodes.append({
            "episode": index,
            "end": "in_progress",
            "states": [runner.observation()],
            "actions": [],
        })
        if runner.episode_state is EpisodeState.WIN:
            self._on_win(runner, lv)
        elif runner.episode_state is EpisodeState.GAME_OVER:
            self._on_game_over(runner, lv)

    def _close_episode(self,
                       end: str,
                       evaluation: Optional[EpisodeEvaluation] = None) -> None:
        _, lv = self._require_level()
        ep = lv.current_episode
        if ep is None:
            return
        ep.end = end
        if evaluation is not None:
            ep.reward = float(evaluation.reward)
            ep.terminated = bool(evaluation.terminated)
            ep.rejected = bool(evaluation.rejected)
        if self._level_episodes:
            self._level_episodes[-1]["end"] = end

    def _reload_episodes(self) -> List[Dict[str, Any]]:
        """The level's episodes as recorded, for continuing the pickle."""
        assert self._recording is not None and self._runner is not None
        episodes: List[Dict[str, Any]] = []
        if os.path.isfile(self._recording.episodes_path):
            for ep in self._recording.read_episodes():
                episodes.append({
                    "episode":
                    ep["episode"],
                    "end":
                    ep["end"],
                    "states":
                    list(ep["states"]),
                    "actions": [
                        Action(np.array(a["arr"], dtype=np.float32))
                        for a in ep["actions"]
                    ],
                })
        if episodes and episodes[-1]["end"] == "in_progress":
            # Replace the live episode with the replayed one: same
            # actions, states straight from the env.
            states, actions = self._runner.trajectory()
            episodes[-1]["states"] = states
            episodes[-1]["actions"] = actions
        elif not episodes:
            states, actions = self._runner.trajectory()
            episodes.append({
                "episode": self._episode_index(),
                "end": "in_progress",
                "states": states,
                "actions": actions,
            })
        return episodes

    # -- Persistence -----------------------------------------------------

    def _flush(self) -> None:
        self._tick_clock()
        self._card.save(self._card_path)
        if self._recording is not None and self._runner is not None and \
                self._level_episodes:
            _, lv = self._require_level()
            self._recording.flush(self._level_episodes,
                                  self._runner.observation(),
                                  self._episode_index(), lv.steps)
        self._steps_since_flush = 0

    def _terminal_index(self, entry: Dict[str, Any]) -> None:
        """A win/game-over entry: written now, or after the invocation that is
        running."""
        if self._in_invocation:
            self._pending_terminal = entry
        else:
            self._index(entry)

    def _index(self, entry: Dict[str, Any]) -> None:
        if self._recording is not None:
            entry = dict(entry)
            entry.setdefault("run_steps", self._card.total_steps)
            self._recording.append_index(entry)

    def _tick_clock(self) -> None:
        now = time.time()
        if self._current is not None:
            lv = self._card.levels[self._current]
            lv.wall_clock += now - self._tick
            if self._runner is not None:
                env_seconds = self._runner.env_seconds
                lv.wall_clock_env += env_seconds - self._env_seconds_at_tick
                self._env_seconds_at_tick = env_seconds
        self._tick = now

    def _active_seconds(self) -> float:
        return self._card.total_wall_clock + (time.time() - self._tick)

    def _check_caps(self) -> None:
        if self._card.total_steps >= self._card.step_cap:
            raise RunEnded("step_cap")
        if self._active_seconds() >= self._card.wall_clock_cap:
            raise RunEnded("wall_clock_cap")

    def _require_level(self) -> Tuple[EpisodeRunner, LevelCard]:
        if self._current is None or self._runner is None:
            raise EpisodeOver("no level is in progress")
        return self._runner, self._card.levels[self._current]

    def _require_open_level(self) -> Tuple[EpisodeRunner, LevelCard]:
        """The level in progress, for a charged call: it must not be over."""
        runner, lv = self._require_level()
        if lv.won:
            raise LevelAlreadyWon(f"level {lv.index + 1} is already won")
        if lv.lost:
            raise LevelLost(f"level {lv.index + 1} is lost: its episode "
                            "ended in GAME_OVER and the level has no resets")
        return runner, lv

    def resets_allowed(self) -> bool:
        """Whether the agent may reset the level in progress: always on a train
        level, on a test level only under ``continual_allow_test_resets``
        (section 4.6)."""
        _, lv = self._require_level()
        return self._levels[lv.index].split == "train" or \
            bool(CFG.continual_allow_test_resets)

    def _load_or_new_card(self) -> RunCard:
        if getattr(CFG, "auto_resume", False) and \
                os.path.isfile(self._card_path):
            card = RunCard.load(self._card_path)
            if card.is_finished:
                logging.info(
                    "[Continual] --auto_resume: %s already ended (%s); "
                    "starting a fresh run over it.", self._card_path,
                    card.end_reason)
            elif len(card.levels) != len(self._levels):
                logging.warning(
                    "[Continual] --auto_resume: %s has %d levels but the "
                    "env now has %d; starting fresh.", self._card_path,
                    len(card.levels), len(self._levels))
            else:
                logging.info(
                    "[Continual] --auto_resume: resuming %s at "
                    "level %s", self._card_path, card.current_level_index())
                return card
        levels = [
            LevelCard(index=s.index,
                      split=s.split,
                      task_idx=s.task_idx,
                      goal=s.goal_strs,
                      goal_nl=s.task.goal_nl or "") for s in self._levels
        ]
        return RunCard(
            run_id=self._run_id,
            env=CFG.env,
            seed=CFG.seed,
            arm=self._arm,
            levels=levels,
            step_cap=int(CFG.continual_steps_per_level * len(levels)),
            wall_clock_cap=float(CFG.continual_wall_clock_hours * 3600.0),
            config=CFG.experiment_id,
        )


def _episode_open(runner: EpisodeRunner) -> bool:
    """Whether the runner's episode is still in progress (a call, so mypy does
    not narrow the property across the stepping loop)."""
    return runner.episode_state is EpisodeState.NOT_FINISHED


def run_continual(env: BaseEnv,
                  approach: BaseApproach,
                  offline_dataset: Optional[Dataset] = None) -> RunCard:
    """Entry point from ``run_pipeline``: build the controller and play.

    Under ``--auto_resume`` (``maybe_auto_resume`` found a checkpoint
    and set ``load_approach``) the approach is loaded from its latest
    checkpoint before the run resumes. A learning arm gets the offline
    dataset through ``prepare_for_continual`` without a learning
    session: when to learn is the arm's decision.
    """
    # pylint: disable-next=import-outside-toplevel
    from predicators.run.controllers import create_controller
    if CFG.load_approach and approach.is_learning_based:
        cycle = CFG.skip_until_cycle - 1 if CFG.skip_until_cycle > 0 \
            else None
        approach.load(cycle)
    prepare = getattr(approach, "prepare_for_continual", None)
    if prepare is not None:
        prepare(
            offline_dataset if offline_dataset is not None else Dataset([]))
    controller = create_controller(env, approach)
    run = ContinualRun(env, approach, controller)
    return run.run()


def _skill_library(env: BaseEnv,
                   approach: BaseApproach) -> List[ParameterizedOption]:
    """The parameterised options the arm may invoke."""
    options = getattr(approach, "_initial_options", None)
    if options:
        return sorted(options, key=lambda o: o.name)
    # pylint: disable-next=import-outside-toplevel
    from predicators.ground_truth_models import get_gt_options
    return sorted(get_gt_options(env.get_name()), key=lambda o: o.name)


def level_summary(card: RunCard) -> str:
    """One line per level for logs."""
    lines = []
    for lv in card.levels:
        status = "won" if lv.won else ("lost" if lv.lost else
                                       ("not won" if lv.attempted else "-"))
        lines.append(f"L{lv.index + 1} {lv.split}[{lv.task_idx}] {status}: "
                     f"steps {lv.steps}, resets {lv.resets}, invocations "
                     f"{lv.skill_invocations}, game overs "
                     f"{len(lv.game_overs)}")
    return "\n".join(lines)
