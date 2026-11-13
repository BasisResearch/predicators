"""Exploration probe API exposed to agents via ``run_python``.

``BeliefProbe`` is a thin facade over shared machinery -
``parse_sketch_from_text`` (plan grammar), ``execute_plan_forward``
(forward executor over the option model), the tools' state-modification
and rendering helpers. What it adds is composability: the agent can set
the sim to any task state (or a modified copy), read full-precision
features, run partial plans, render, snapshot/restore, and write sweep
loops in one ``run_python`` call instead of one tool round-trip per
experiment.

Nothing the probe executes acts in the environment: plans reach the
robot only through the play tools (``skills_execute_plan``). The task
evaluator is reachable, but only as a read-only preview:
``run(trials>=2, solved=True)`` and ``refine(require_solved=True)``
score rollouts with it (see ``_require_solved_evaluator``).

In synthesis sessions the same facade probes the CANDIDATE simulator
(the ``simulator.py`` under edit, freshly fitted - see
``BeliefProbe._option_model``), ``fit`` exposes the fitting stack
(``ctx.probe_fit_provider``), and validation is hand-composed:
``fit()`` then ``refine`` then a continuous ``run`` of the refined
plan (the forward pass; ``run`` reports the first step whose subgoal
annotation failed to hold).
"""

from __future__ import annotations

import dataclasses
import functools
import json
import logging
import os
import time
from typing import TYPE_CHECKING, Any, Callable, Dict, FrozenSet, List, \
    Optional, Sequence, Set, Tuple, Union

from predicators import utils
from predicators.agent_sdk.config import RefinementConfig
from predicators.agent_sdk.parallel_rollouts import prefetch_parallel
from predicators.agent_sdk.probe_results import ProbeRefineResult, \
    ProbeResult, ProbeTrialsResult, _StrLikeResult
from predicators.agent_sdk.rollout_probe import _RUN_MEMORY_LEAD, \
    ProbeBudgetExceeded, _check_time_budget, _count_rollout, _failed_skills, \
    _fmt_option, _fmt_params, _grounded_steps, _line_states, _memory_written, \
    _motion_failure_note, _motion_failures, _RolloutCore, _skill_record
from predicators.agent_sdk.tools.context import LiveRehearsal, \
    absolute_rollout_seed, decorrelated_rollout_seed
from predicators.agent_sdk.tools.verdicts import _EvalStateCollector, \
    _policy_source_path, evaluate_states_with, make_solved_check, \
    task_evaluator
from predicators.code_sim_learning.model_state import memory_written_note
from predicators.structs import State, Task, excluded_object_type_names

if TYPE_CHECKING:
    from predicators.agent_sdk.tools import ToolContext


@dataclasses.dataclass(repr=False)
class ProbeBeliefResult(_StrLikeResult):
    """Outcome of one ``BeliefProbe.run(..., belief_draws=K)`` call.

    ``draws`` holds one dict per draw of the belief (``goal_reached``,
    ``num_actions``, ``failure``, ``max_shift`` - the largest feature
    move of the drawn start state from the probe's current state);
    ``successes`` counts goal-reaching draws. Each draw is where the
    objects may really be given the observation and the declared noise,
    rolled on a fresh env at the base planner seed, so a failing draw
    is a pose the plan does not tolerate, not execution variability.
    The current state is NOT advanced.
    """
    draws: List[Dict[str, Any]]
    successes: int
    from_belief: bool
    notes: List[str] = dataclasses.field(default_factory=list)

    def __repr__(self) -> str:
        n = len(self.draws)
        source = ("the belief shown in the last observation (its smoothed "
                  "frame and spread)" if self.from_belief else
                  "the current state with the declared sigma on every "
                  "noisy feature")
        lines = [
            f"Belief draws: {self.successes}/{n} start states reached the "
            f"goal (each draw is one place the objects may really be, "
            f"from {source}; fresh env + base planner seed per draw)"
        ]
        for k, d in enumerate(self.draws):
            head = f"  draw {k} (max feature shift {d['max_shift']:.4f})"
            if d["failure"]:
                lines.append(f"{head}: FAILED - {d['failure']}")
            elif d["goal_reached"]:
                lines.append(f"{head}: goal reached ({d['num_actions']} "
                             "actions)")
            else:
                lines.append(f"{head}: goal NOT reached "
                             f"({d['num_actions']} actions)")
        if 0 < self.successes < n:
            lines.append(
                "  the plan depends on a pose the observation cannot pin "
                "down: it succeeds on some plausible poses and fails on "
                "others. Either look again from rest (the belief narrows "
                "with every frame an object rests through) or choose "
                "parameters with margin over the whole spread before "
                "spending real steps.")
        elif self.successes == 0 and n:
            lines.append("  no plausible pose lets this plan reach the goal.")
        lines.extend(f"NOTE: {n_}" for n_ in self.notes)
        return "\n".join(lines)


_UNSETTLED_ENGINES: Set[str] = set()


def _settled(env: Any, state: State) -> State:
    """``state`` after its engine's quasi-static settle, which makes a draw of
    the state belief consistent with contacts and attachments (see
    ``PyBulletEnv.settle_state``); unchanged when the engine has none."""
    settle = getattr(env, "settle_state", None)
    if settle is None:
        name = type(env).__name__
        if name not in _UNSETTLED_ENGINES:
            _UNSETTLED_ENGINES.add(name)
            logging.warning(
                "State draws are not settled: the %s engine has no "
                "settle_state, so rollouts start from the draws as drawn.",
                name)
        return state
    try:
        return settle(state)
    except Exception as e:  # pylint: disable=broad-except
        logging.warning(
            "Settling a state draw failed, so its rollout starts "
            "from the draw: %s", e)
        return state


# How many live-observation rehearsals the context keeps for the
# rehearsal gate; older ones were from earlier observations anyway.
_LIVE_REHEARSALS_KEPT = 32


def _log_rehearsal(ctx: "ToolContext", plan_text: str,
                   result: "ProbeJointResult") -> None:
    """Append one rehearsal to ``<log_dir>/rehearsals.jsonl``.

    The record (time, plan, P-hat and each draw's outcome) is what a
    calibration check joins with the executed plans and their real
    outcomes: whether plans rehearsed at P-hat p succeed about a fraction
    p of the time.
    """
    if not ctx.log_dir:
        return
    keys = ("params", "success", "solved", "goal_reached", "lost", "lost_line")
    draws = [{k: d.get(k) for k in keys} for d in result.draws]
    record = {
        "time": time.time(),
        "call": int(ctx.test_call_id),
        "plan": plan_text,
        "p_hat": result.p_hat,
        "stderr": result.stderr,
        "from_observation": result.from_observation,
        "draws": draws,
        "mean": result.mean_verdict,
    }
    try:
        with open(os.path.join(ctx.log_dir, "rehearsals.jsonl"),
                  "a",
                  encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except OSError as e:
        logging.warning("Could not log the rehearsal: %s", e)


def _planning_values() -> str:
    """What a report calls the values the planning model runs at: the fit's
    point estimate, or the declared values in the arm whose text names no
    harness fit (``agent_sim_learn_fit_invisible``)."""
    # pylint: disable-next=import-outside-toplevel
    from predicators.settings import CFG
    if CFG.agent_sim_learn_fit_invisible:
        return "the declared values"
    return "the fit's point estimate"


def _mean_disagreement_note(result: "ProbeJointResult") -> str:
    """Under ``continual_rehearsal_mean_verdict``, the note for a mean rollout
    whose verdict no scored draw shares: it ends the episode in GAME_OVER and
    no draw does, or it solves the task and no draw does; "" otherwise.

    The mean rollout runs at the fit's point estimate, the values the
    planning model runs at, while the draws spread over the belief; a
    point estimate at the edge of the belief can predict a loss that
    every draw misses.
    """
    verdict = result.mean_verdict
    scored = [d for d in result.draws if d.get("solved") is not None]
    if verdict is None or not scored:
        return ""
    lead = (f"The rollout from the belief mean, at {_planning_values()} "
            "(the values the planning model runs at), ")
    if verdict["lost"] and not any(d.get("lost") for d in scored):
        return (f"{lead}ends the episode in GAME_OVER, but none of the "
                f"{len(scored)} draws the task's evaluator scored does: the "
                "planning model predicts a loss the draws miss, so they may "
                "understate its risk.")
    if verdict["solved"] and not any(d.get("solved") for d in scored):
        return (f"{lead}solves the task, but none of the {len(scored)} "
                "draws the task's evaluator scored does: the planning model "
                "predicts a success the draws do not support.")
    return ""


def _fmt_draw_params(params: Dict[str, float], limit: int = 4) -> str:
    """A draw's parameters, shortened for a one-line report."""
    items = sorted(params.items())
    text = ", ".join(f"{k}={v:.4g}" for k, v in items[:limit])
    if len(items) > limit:
        text += f", +{len(items) - limit} more"
    return text


def _separating_ranges(draws: List[Dict[str, Any]]) -> List[str]:
    """One line per parameter whose failing and passing draws do not
    overlap: the side of the parameter on which the plan fails."""
    passing = [d for d in draws if d["success"]]
    failing = [d for d in draws if not d["success"]]
    if not passing or not failing:
        return []
    lines = []
    for name in sorted(passing[0]["params"]):
        good = [d["params"][name] for d in passing if name in d["params"]]
        bad = [d["params"][name] for d in failing if name in d["params"]]
        if not good or not bad:
            continue
        if max(bad) < min(good):
            lines.append(f"{name}: every failing draw has {name} <= "
                         f"{max(bad):.4g}; passing draws span "
                         f"[{min(good):.4g}, {max(good):.4g}]")
        elif min(bad) > max(good):
            lines.append(f"{name}: every failing draw has {name} >= "
                         f"{min(bad):.4g}; passing draws span "
                         f"[{min(good):.4g}, {max(good):.4g}]")
    return lines


# A lead is a parameter whose failing draws sit mostly on one side of its
# passing draws: the AUC below is at least this far from 0.5. Chance
# alone reaches it for some of many parameters, so the report says how
# often shuffled outcomes do.
_LEAD_STRENGTH = 0.3
_LEAD_SHUFFLES = 400
# A rollout value counts as past the recordings when it lies beyond their
# range, and beyond its own start, by more than _SUPPORT_SIGMAS of the
# feature's declared observation noise: a recorded extreme is one noisy
# reading. A feature read exactly counts _SUPPORT_EXACT of its recorded
# span as its sigma. The support note names at most _SUPPORT_NAMED.
_SUPPORT_SIGMAS = 2.0
_SUPPORT_EXACT = 0.01
_SUPPORT_NAMED = 3


def _unreconstructed(env: Any) -> List[str]:
    """``object.feature`` for each feature ``env`` read back differently from
    the state it was last set to, the robot's aside: its joints come back
    through inverse kinematics, which is lossy by design."""
    lossy = getattr(env, "_last_unreconstructible_features", None) or []
    return [
        f"{obj.name}.{feature}" for obj, feature in lossy
        if obj.type.name != "robot"
    ]


def _beyond_recordings(ctx: "ToolContext",
                       states: Sequence[State]) -> Optional[str]:
    """Under ``continual_rehearsal_support_note``, the scored features the
    rollout from the belief mean takes past the range every recorded state
    shows, and past where the rollout starts, by more than twice their declared
    observation noise; None when there are none.

    The margin is the data's own: a recorded extreme is one noisy
    reading, so a value within two of its sigmas is about as well
    tested. The rollout's start counts as recorded, so a new level's
    layout is not named for being new.
    """
    # pylint: disable=import-outside-toplevel
    from predicators.observation_noise import ObservationNoise
    from predicators.settings import CFG

    # pylint: enable=import-outside-toplevel
    if not CFG.continual_rehearsal_support_note or not states or \
            ctx.recorded_ranges_provider is None:
        return None
    ranges = ctx.recorded_ranges_provider() or {}
    noise = ObservationNoise.from_cfg()
    start = states[0]
    worst: Dict[Tuple[str, str], Tuple[float, float]] = {}
    for state in states[1:]:
        for obj in state:
            if obj.type.name == "robot" or obj not in start:
                continue
            for feature in obj.type.feature_names:
                span = ranges.get((obj.type.name, feature))
                if span is None:
                    continue
                first = float(start.get(obj, feature))
                lo, hi = min(span[0], first), max(span[1], first)
                sigma = (noise.feature_sigma(obj.type, feature)
                         if noise.declared else 0.0)
                if sigma <= 0.0:
                    sigma = _SUPPORT_EXACT * max(span[1] - span[0], 1e-9)
                value = float(state.get(obj, feature))
                past = max(value - hi, lo - value) / sigma
                key = (obj.name, feature)
                if past > _SUPPORT_SIGMAS and past > worst.get(
                        key, (0.0, 0.0))[0]:
                    worst[key] = (past, value)
    if not worst:
        return None
    named = sorted(worst.items(), key=lambda kv: -kv[1][0])[:_SUPPORT_NAMED]
    parts = []
    for (name, feature), (_past, value) in named:
        type_name = next(o.type.name for o in start if o.name == name)
        lo, hi = ranges[(type_name, feature)]
        parts.append(f"{name}.{feature} to {value:.4g} (recorded "
                     f"{type_name} {feature}: {lo:.4g} to {hi:.4g})")
    return ("The rollout from the belief mean takes " + "; ".join(parts) +
            ". No recording tests the model there, so its prediction there "
            "can be off by more than the draws' spread.")


def _faster_than_recorded(ctx: "ToolContext",
                          states: Sequence[State]) -> Optional[str]:
    """Under ``continual_rehearsal_speed_note``, the scored bodies the rollout
    from the belief mean moves faster than any recording shows them moving
    under the same switch pattern; None when there are none.

    A pattern no recording shows counts as a top speed of zero. The
    margin is the data's own: a speed read from two noisy positions is
    about as well tested within two of its sigmas.
    """
    # pylint: disable=import-outside-toplevel
    from predicators.agent_sdk.speed_support import SPEED_WINDOW, \
        scored_bodies, window_speeds
    from predicators.observation_noise import ObservationNoise

    # pylint: enable=import-outside-toplevel
    if ctx.recorded_speeds_provider is None or \
            ctx.scored_features_provider is None or not states:
        return None
    recorded = ctx.recorded_speeds_provider()
    features = ctx.scored_features_provider()
    if recorded is None or not features:
        return None
    switches, top = recorded
    noise = ObservationNoise.from_cfg()
    worst: Dict[Tuple[str, FrozenSet[str]], Tuple[float, float, float]] = {}
    bodies = scored_bodies(features)
    for obj, pattern, speed in window_speeds(states, bodies, switches):
        sigma = (noise.feature_sigma(obj.type, bodies[obj.type.name][0])
                 if noise.declared else 0.0)
        record = top.get((obj.type.name, pattern), 0.0)
        margin = max(_SUPPORT_SIGMAS * sigma * 2.0**0.5 / SPEED_WINDOW,
                     _SUPPORT_EXACT * record)
        key = (obj.name, pattern)
        if speed > record + margin and speed - record > worst.get(
                key, (0.0, 0.0, 0.0))[0]:
            worst[key] = (speed - record, speed, record)
    if not worst:
        return None
    named = sorted(worst.items(), key=lambda kv: -kv[1][0])[:_SUPPORT_NAMED]
    parts = []
    for (name, pattern), (_gap, speed, record) in named:
        switched = ", ".join(sorted(pattern)) or "no switch"
        type_name = next(o.type.name for o in states[0] if o.name == name)
        seen = (f"the top recorded speed is {record:.4g}" if
                (type_name, pattern) in top else "no recording has this")
        parts.append(f"{name} at {speed:.4g} per step with {switched} on "
                     f"({seen})")
    # pylint: disable-next=import-outside-toplevel
    from predicators.settings import CFG
    then = ("check the model against what that plan records"
            if CFG.agent_sim_learn_fit_invisible else "refit")
    return ("The rollout from the belief mean moves " + "; ".join(parts) +
            ". No recording tests the model at that speed, so its prediction "
            "there can be off by more than the draws' spread. If a failure "
            "there cannot be undone, first run a plan that stays safe if the "
            f"model is wrong at that speed, then {then}.")


def _counts_game_over() -> bool:
    """Whether rehearsal reports count the draws that end in GAME_OVER."""
    # pylint: disable-next=import-outside-toplevel
    from predicators.settings import CFG
    return bool(CFG.continual_rehearsal_game_over)


def _outcome_leads(draws: List[Dict[str, Any]],
                   undeclared: Set[str]) -> List[str]:
    """The parameters the failing draws lean on when none separates them
    outright, and how often shuffled outcomes lean as far.

    Needs at least three failing and three passing draws; returns no
    lines when no parameter's AUC is ``_LEAD_STRENGTH`` from 0.5.
    """
    # pylint: disable=import-outside-toplevel
    import numpy as np

    def _auc(values: Any, failing: Any) -> float:
        """The chance that a failing draw's value exceeds a passing draw's,
        ties counting half: the Mann-Whitney statistic."""
        bad = values[failing][:, None]
        good = values[~failing][None, :]
        wins = float(np.sum(bad > good)) + 0.5 * float(np.sum(bad == good))
        return wins / (bad.size * good.size)

    failing = np.array([not d["success"] for d in draws])
    if failing.sum() < 3 or (~failing).sum() < 3:
        return []
    names = set.intersection(*(set(d["params"]) for d in draws))
    table = {
        name: np.array([float(d["params"][name]) for d in draws])
        for name in sorted(names)
    }
    table = {k: v for k, v in table.items() if np.ptp(v) > 0}
    if not table:
        return []
    aucs = {k: _auc(v, failing) for k, v in table.items()}
    ranked = sorted(aucs, key=lambda k: -abs(aucs[k] - 0.5))
    top = [k for k in ranked[:3] if abs(aucs[k] - 0.5) >= _LEAD_STRENGTH]
    if not top:
        return []
    best = abs(aucs[top[0]] - 0.5)
    rng = np.random.default_rng(0)
    hits = 0
    for _ in range(_LEAD_SHUFFLES):
        shuffled = rng.permutation(failing)
        if max(abs(_auc(v, shuffled) - 0.5) for v in table.values()) >= best:
            hits += 1
    lines = [
        "The failing draws lean on these parameters (AUC: the chance that "
        "a failing draw's value exceeds a passing draw's):"
    ]
    for k in top:
        side = "higher" if aucs[k] > 0.5 else "lower"
        star = "*" if k in undeclared else ""
        lines.append(f"  {k}{star}: AUC {aucs[k]:.2f}, failing draws "
                     f"{side} (median {np.median(table[k][failing]):.4g} "
                     f"against {np.median(table[k][~failing]):.4g})")
    if any(k in undeclared for k in top):
        # pylint: disable-next=import-outside-toplevel
        from predicators.settings import CFG
        if CFG.belief_material_lines:
            lines.append("  * not declared in simulator.py: the fit keeps "
                         "your scene's value, and the draws move it only "
                         "as far as the recordings allow with the other "
                         "parameters at their fitted values; declare it "
                         "to fit it together with them.")
        else:
            lines.append("  * not declared in simulator.py: every "
                         "rehearsal draws it over its plausible range; "
                         "declare it and the recordings fit it.")
    lines.append(
        f"  Shuffling which draws failed leans some parameter this far in "
        f"{hits / _LEAD_SHUFFLES:.0%} of {_LEAD_SHUFFLES} shuffles: these "
        "are leads to test before you rely on them.")
    return lines


@dataclasses.dataclass(repr=False)
class ProbeJointResult(_StrLikeResult):
    """Outcome of ``BeliefProbe.run(plan)`` under the joint belief.

    ``draws`` holds one dict per joint draw ``(theta_i, x_t_i)``:
    ``params`` (theta_i), ``goal_reached``, ``solved`` (the task
    evaluator's verdict on the episode so far followed by the draw's
    rollout, None when it could not score the draw), ``lost`` (whether
    that evaluator ends the episode there unsolved, as a real episode
    ends in GAME_OVER; None when unscored), ``success`` (solved when
    scored, else goal reached), ``num_actions``, ``failure`` and
    ``planner_seed``. ``p_hat`` is the fraction of successful draws and
    ``stderr`` its standard error; ``mean`` is the step-by-step rollout
    from the belief mean at the values the planning model runs at (the
    point estimate). Only the mean rollout advances the current state,
    and its fields (``final_state``, ``steps``, ``goal_reached``,
    ``final_atoms``) read through, as on a single rollout's result.
    """
    draws: List[Dict[str, Any]]
    p_hat: float
    stderr: float
    from_observation: bool
    mean: Optional[Any] = None
    notes: List[str] = dataclasses.field(default_factory=list)
    # The report's lines on the parameters the failing draws lean on
    # (``_outcome_leads``), when none separates them outright.
    leads: List[str] = dataclasses.field(default_factory=list)
    # Under ``continual_rehearsal_mean_verdict``, the task evaluator's
    # verdict on the mean rollout as on a draw (``solved``, ``lost``); None
    # when it was not scored.
    mean_verdict: Optional[Dict[str, bool]] = None

    @property
    def successes(self) -> int:
        """How many draws succeeded."""
        return sum(1 for d in self.draws if d["success"])

    def __getattr__(self, name: str) -> Any:
        # String methods stay on this report; other unknown attributes
        # read the mean rollout's result.
        if not name.startswith("_") and not hasattr(str, name):
            mean = self.__dict__.get("mean")
            if mean is not None and hasattr(mean, name):
                return getattr(mean, name)
        return super().__getattr__(name)

    def _game_over_note(self) -> str:
        """Under ``continual_rehearsal_game_over``, how many of the failing
        draws the task's evaluator scored end the episode in GAME_OVER."""
        failing = [
            d for d in self.draws
            if d["solved"] is False and d.get("lost") is not None
        ]
        if not failing or not _counts_game_over():
            return ""
        lost = sum(1 for d in failing if d["lost"])
        if not lost:
            return (f"; none of the {len(failing)} failing draws the task's "
                    "evaluator scored ends the episode in GAME_OVER")
        return (f"; {lost} of the {len(failing)} failing draws the task's "
                "evaluator scored end the episode in GAME_OVER, as a real "
                "episode ends when it is lost")

    def _failed_skill_note(self) -> str:
        """Under ``continual_rehearsal_game_over``, how many draws stopped at a
        skill whose controller failed, when some draw failed."""
        if not _counts_game_over() or self.successes == len(self.draws):
            return ""
        failed = sum(1 for d in self.draws if d["failure"])
        if not failed:
            return "; no draw stopped at a failed skill"
        return f"; {failed} draw(s) stopped at a failed skill"

    def __repr__(self) -> str:
        n = len(self.draws)
        scored = sum(1 for d in self.draws if d["solved"] is not None)
        source = ("each draw pairs a parameter draw with a state draw "
                  "and the memory those parameters imply"
                  if self.from_observation else
                  "the probe is not on the current observation, so each "
                  "draw varies the parameters from its current state")
        # Draws whose goal atoms held but which the task's evaluator did
        # not count: said in the headline, so a reader of that line alone
        # does not take them for rollouts that missed the goal.
        held = sum(1 for d in self.draws
                   if d["solved"] is False and d["goal_reached"])
        held_note = (f"; the goal atoms held on {held} of the failing draws, "
                     "which the task's evaluator did not count as solved"
                     if held else "")
        lines = [
            f"Rehearsal on {n} joint draws of the belief: P-hat = "
            f"{self.p_hat:.2f} +- {self.stderr:.2f} ({self.successes}/{n} "
            f"succeeded{self._failed_skill_note()}{held_note}"
            f"{self._game_over_note()}; {source}; fresh env and planner seed "
            "per draw)"
        ]
        if scored < n:
            lines.append(
                f"  {n - scored} draw(s) could not be scored by the task "
                "evaluator and count by whether the goal atoms held.")
        for k, d in enumerate(self.draws):
            head = f"  draw {k} [{_fmt_draw_params(d['params'])}]"
            ended = (", GAME_OVER"
                     if d.get("lost") and _counts_game_over() else "")
            if d["failure"] and d["success"]:
                # The evaluator counted the episode solved before the
                # step failed, as a real episode ends there.
                lines.append(f"{head}: solved ({d['num_actions']} actions), "
                             f"then FAILED - {d['failure']}")
            elif d["failure"]:
                lines.append(f"{head}: FAILED{ended} - {d['failure']}")
            elif d["solved"] is False:
                lines.append(
                    f"{head}: goal atoms "
                    f"{'held' if d['goal_reached'] else 'did not hold'}"
                    f", NOT solved{ended} ({d['num_actions']} actions)")
            elif d["success"]:
                lines.append(f"{head}: solved ({d['num_actions']} actions)")
            else:
                lines.append(f"{head}: goal NOT reached "
                             f"({d['num_actions']} actions)")
        separating = _separating_ranges(self.draws)
        if separating:
            lines.append("Where the draws fail:")
            lines.extend(f"  {line}" for line in separating)
        elif self.leads:
            lines.extend(self.leads)
        elif 0 < self.successes < n:
            lines.append("  no single parameter separates the failing draws "
                         "from the passing ones: the state draws, or "
                         "combinations of parameters, decide them.")
        # The mean rollout's report repeats its own notes.
        mean_notes = getattr(self.mean, "notes", None) or []
        lines.extend(f"NOTE: {n_}" for n_ in self.notes
                     if n_ not in mean_notes)
        if self.mean is not None:
            lines.append("")
            lines.append("Step-by-step rollout from the belief mean at "
                         f"{_planning_values()} (the values the planning "
                         "model runs at):")
            lines.append(repr(self.mean))
        return "\n".join(lines)


@dataclasses.dataclass(repr=False)
class ProbeBeliefState(_StrLikeResult):
    """Outcome of ``BeliefProbe.belief()``: the execution-time belief at the
    probe's current state.

    ``objects`` holds one line per object with noisy features (value and
    spread per feature, frames averaged); ``fractions`` maps each atom
    of the session's predicates that holds on any belief draw to the
    fraction of draws it holds on; ``from_observation`` says whether the
    belief is the one the last real observation showed (the probe still
    sits on it) or the current state with the declared sigma.
    """
    objects: List[str]
    fractions: Dict[str, float]
    from_observation: bool
    num_draws: int

    def __repr__(self) -> str:
        source = ("the last real observation's belief"
                  if self.from_observation else
                  "the current state with the declared sigma on every noisy "
                  "feature")
        lines = [
            f"Belief at the current state (from {source}; atoms scored on "
            f"{self.num_draws} draws):"
        ]
        lines.extend(f"  {line}" for line in self.objects)
        unsure = sorted((a for a, f in self.fractions.items() if 0 < f < 1),
                        key=lambda a: (abs(self.fractions[a] - 0.5), a))
        sure = sorted(a for a, f in self.fractions.items() if f >= 1)
        if unsure:
            lines.append("  unsure atoms (fraction of draws): " +
                         ", ".join(f"{a} {self.fractions[a]:.2f}"
                                   for a in unsure))
        lines.append("  atoms holding on every draw: " +
                     (", ".join(sure) if sure else "(none)"))
        return "\n".join(lines)


def _reference_point() -> str:
    """A physics sweep's name for its reference point: the fitted values, or
    the declared values in the arm whose text names no harness fit."""
    # pylint: disable-next=import-outside-toplevel
    from predicators.settings import CFG
    if CFG.agent_sim_learn_fit_invisible:
        return "declared values"
    return "fitted params"


@dataclasses.dataclass(repr=False)
class ProbeSweepResult(_StrLikeResult):
    """Outcome of one ``BeliefProbe.run(..., physics_sweep=True)`` call.

    ``points`` holds one dict per sweep point in ascending-sigma order
    (``params`` - the physical-parameter override dict, ``None`` for
    the reference point at the planning model's values, run first;
    ``goal_reached``,
    ``num_actions``, ``failure``). ``successes`` counts goal-reaching
    points. Each point runs once on a fresh env at the BASE
    motion-planner seed, so its outcome is a deterministic measurement
    of the plan at those physical parameters - a failing point is a
    hole in the design's success band, not noise. The current state is
    NOT advanced.
    """
    points: List[Dict[str, Any]]
    successes: int
    notes: List[str] = dataclasses.field(default_factory=list)

    def __repr__(self) -> str:
        n = len(self.points)
        lines = [
            f"Physics sweep: {self.successes}/{n} points reached the goal "
            "(fresh env + base planner seed per point - each point is a "
            "deterministic measurement, not a sample)"
        ]
        for p in self.points:
            desc = (_reference_point() if p["params"] is None else ", ".join(
                f"{k}={v:.4g}" for k, v in sorted(p["params"].items())))
            if p["failure"]:
                line = f"  {desc}: FAILED - {p['failure']}"
            elif p["goal_reached"]:
                line = f"  {desc}: goal reached ({p['num_actions']} actions)"
            else:
                line = (f"  {desc}: goal NOT reached "
                        f"({p['num_actions']} actions)")
            lines.append(line)
        if self.successes < n:
            # pylint: disable-next=import-outside-toplevel
            from predicators.settings import CFG
            owner = ("declared"
                     if CFG.agent_sim_learn_fit_invisible else "identified")
            lines.append(
                f"  the plan fails INSIDE the {owner}-parameter "
                "uncertainty range. The real environment may sit at any "
                "of these points (success can be non-monotonic: passing "
                "neighbors do NOT cover the points between them) - add "
                "design margin until every point passes before executing "
                "the plan.")
        lines.extend(f"NOTE: {n_}" for n_ in self.notes)
        return "\n".join(lines)


@dataclasses.dataclass(repr=False)
class ProbeSuggestResult(_StrLikeResult):
    """Outcome of ``BeliefProbe.suggest_probes``: per annotated step, the
    agent's own parameters' ensemble disagreement and the top alternatives."""
    suggestions: List[Any]
    notes: List[str]
    # "disagreement" (the legacy ensemble score) or "information" (the
    # joint belief's predicate information, paper Eq. 4).
    measure: str = "disagreement"

    def __repr__(self) -> str:
        if self.measure == "information":
            return self._repr_information()
        lines = [
            "Probe suggestions - ensemble disagreement on each step's "
            "subgoal atoms (0 = every posterior member agrees on the "
            "outcome, 1 = the members split; higher = a measurement that "
            "teaches the model more). These are SUGGESTIONS: your "
            "submitted parameters execute exactly as written, so adopt an "
            "alternative only by writing it into your sketch, and only on "
            "a step whose failure the episode can afford."
        ]
        for s in self.suggestions:
            objs = ", ".join(s.objects)
            head = f"  step {s.step_idx} {s.option_name}({objs}) -> " \
                f"{{{', '.join(s.subgoal_atoms)}}}:"
            if s.nominal_params is None:
                head += " no proposal;"
            elif not s.nominal_feasible:
                head += (f" proposed {_fmt_params(s.nominal_params)} does "
                         "NOT establish the subgoal;")
            else:
                per = ", ".join(f"{a}={v:.2f}"
                                for a, v in s.nominal_per_atom.items())
                head += (f" proposed {_fmt_params(s.nominal_params)} "
                         f"disagreement {s.nominal_score:.2f} ({per});")
            head += f" {s.n_feasible}/{s.n_draws} draws feasible"
            lines.append(head)
            if not s.candidates:
                lines.append("      no feasible alternative found")
            for rank, (params, score, per_atom) in enumerate(s.candidates, 1):
                per = ", ".join(f"{a}={v:.2f}" for a, v in per_atom.items())
                lines.append(f"      alt {rank}: {_fmt_params(params)} "
                             f"disagreement {score:.2f} ({per})")
        if not self.suggestions:
            lines.append("  (no annotated step with continuous parameters)")
        for note in self.notes:
            lines.append(f"  NOTE: {note}")
        return "\n".join(lines)

    def _repr_information(self) -> str:
        lines = [
            "Experiment suggestions - the information (bits) a noisy "
            "reading of each step's subgoal atoms carries about the "
            "parameters, over the joint draws of the belief (higher = the "
            "outcome depends more on what the belief is unsure of). These "
            "are SUGGESTIONS: adopt an alternative only by writing it into "
            "your plan, and weigh the information against the environment "
            "steps it costs."
        ]
        for s in self.suggestions:
            objs = ", ".join(s.objects)
            head = f"  step {s.step_idx} {s.option_name}({objs}) -> " \
                f"{{{', '.join(s.subgoal_atoms)}}}:"
            if s.nominal_params is None:
                head += " no proposal;"
            elif not s.nominal_feasible:
                head += (f" proposed {_fmt_params(s.nominal_params)} does "
                         "NOT establish the subgoal in the model;")
            else:
                per = ", ".join(f"{a}={v:.2f}"
                                for a, v in s.nominal_per_atom.items())
                head += (f" proposed {_fmt_params(s.nominal_params)} "
                         f"information {s.nominal_score:.2f} ({per});")
            head += f" {s.n_feasible}/{s.n_draws} alternatives feasible"
            lines.append(head)
            if not s.candidates:
                lines.append("      no feasible alternative found")
            for rank, (params, score, per_atom) in enumerate(s.candidates, 1):
                per = ", ".join(f"{a}={v:.2f}" for a, v in per_atom.items())
                lines.append(f"      alt {rank}: {_fmt_params(params)} "
                             f"information {score:.2f} ({per})")
        if not self.suggestions:
            lines.append("  (no annotated step with continuous parameters)")
        for note in self.notes:
            lines.append(f"  NOTE: {note}")
        return "\n".join(lines)


def _belief_frames_note(ctx: Any, grounded: Sequence[Any]) -> Optional[str]:
    """How few frames the state belief has of the bodies a plan acts on (its
    steps' objects, else every noisy body), when fewer than its window.

    Each frame is a fresh noise draw, so a body that stays at rest
    narrows as sigma / sqrt(frames), up to ``continual_belief_window``
    frames: at the start of an episode the belief has one frame, its
    draws spread a whole sigma, and a short Wait sharpens them (four
    times the frames halve the spread). None
    without the belief the last observation showed, or when every such
    body rests on a full window.
    """
    # pylint: disable=import-outside-toplevel
    from predicators.observation_noise import ObservationNoise
    from predicators.settings import CFG

    # pylint: enable=import-outside-toplevel
    found = _fewest_frames(ctx, grounded)
    if found is None:
        return None
    bodies, fewest = found
    window = max(int(CFG.continual_belief_window), 1)
    sigma = ObservationNoise.from_cfg().position
    if fewest >= window or sigma <= 0.0:
        return None
    count = f"{fewest} frame{'s' if fewest != 1 else ''}"
    their = "its" if len(bodies) == 1 else "their"
    # Four times the frames halve the spread.
    halve = (f" (a Wait of {3 * fewest} steps halves the spread)"
             if 4 * fewest <= window else "")
    return (f"The state belief has {count} of {', '.join(bodies)}, so "
            f"{their} position spread is {sigma / fewest**0.5:.2g} m. While a "
            "body stays at rest, each new frame narrows that spread as 1 / "
            f"sqrt(frames), to {sigma / window**0.5:.2g} m at {window} "
            "frames, so a Wait before this request makes these rollouts plan "
            f"on positions nearer the true ones{halve}.")


def _fewest_frames(ctx: Any,
                   grounded: Sequence[Any]) -> Optional[Tuple[List[str], int]]:
    """The bodies a plan acts on (its steps' objects, else every noisy body)
    that the state belief has the fewest frames of, and how many; None without
    the belief the last observation showed."""
    used = getattr(getattr(ctx, "current_belief", None), "frames_used", None)
    if not used:
        return None
    names = [
        obj.name for option in grounded
        for obj in getattr(option, "objects", ()) if obj.name in used
    ] or list(used)
    fewest = min(used[name] for name in names)
    return sorted({n for n in names if used[n] == fewest}), fewest


def _belief_spread(ctx: Any,
                   grounded: Sequence[Any]) -> Optional[Tuple[float, float]]:
    """The position and orientation spread the state belief keeps of the bodies
    a plan acts on: one frame's noise over the square root of the fewest frames
    it has of them, up to ``continual_belief_window``.

    None without the belief the last observation showed.
    """
    # pylint: disable=import-outside-toplevel
    from predicators.observation_noise import ObservationNoise
    from predicators.settings import CFG

    # pylint: enable=import-outside-toplevel
    found = _fewest_frames(ctx, grounded)
    if found is None:
        return None
    frames = max(1, min(found[1], max(int(CFG.continual_belief_window), 1)))
    noise = ObservationNoise.from_cfg()
    return noise.position / frames**0.5, noise.orientation / frames**0.5


# The core of the probe classes: every call but the harness fit's entry
# points, which BeliefProbe adds. Its state control, readers and single
# rollout are the rollout core's. An agent can read the docstrings of the
# public calls of both (help(sim)), so they name no harness fit.
class _ProbeCore(_RolloutCore):
    """Stateful exploration handle over the belief simulator.

    Typical loop::

        print(sim.task())      # the current level's goal, objects and
                               # init details (no state change)
        sim.reset()                                   # the level's start
        sim.reset(mods={"domino_1": {"x": 0.46}})     # modified copy
        sid = sim.snapshot()
        out = sim.run("Push(robot:robot, domino_0:domino)[0.05, 0.05]\\n"
                      "Wait(robot:robot)[]")
        print(out)             # per-step outcomes
        sim.state("domino_1")  # full-precision features
        sim.render("after_push")
        sim.restore(sid)
        # Search params for a suffix from here (simulation only):
        print(sim.refine(
            "Place(robot:robot)[0.46, 1.32, 0.55, -1.0] ~ [0.05, 0.05, "
            "0.0, 0.5] -> {SomeSubgoal(domino_1:domino)}"))

    The "current state" is just a ``State`` object; ``run`` executes from
    it (the option model resets the sim env from that state) and advances
    it to the rollout's final state.
    """

    # Distinct deterministic rng streams per instance (see refine).
    _next_instance_id = 0

    def __init__(self, ctx: "ToolContext"):
        super().__init__(ctx)
        self._refine_calls = 0
        self._belief_draw_calls = 0
        self._instance_id = _ProbeCore._next_instance_id
        _ProbeCore._next_instance_id += 1

    # ── Introspection ────────────────────────────────────────────

    def check_restore(self) -> Dict[str, Any]:
        """Check current candidate memory/poses across fresh worlds, no ticks.

        This checks reconstruction, not whether the inferred state is
        true. Use after model edits and on held assemblies before
        trusting lifts. The robot's pose comes back through inverse
        kinematics, approximately by design, so its deltas are listed
        apart (``robot_deltas``) and do not reject the check.
        """
        # pylint: disable-next=import-outside-toplevel
        from predicators.agent_sdk.restoration import check_restore
        return check_restore(self._ctx, self._require_state())

    def task(self, task_idx: Optional[int] = None) -> str:
        """Describe a task: goal (NL preferred), initial atoms, objects, and
        initial-state details.

        ``task_idx`` indexes ``train_tasks``, the levels reached so far;
        ``None`` describes the current level. Purely informational -
        does not change the probe's current state. Use
        ``reset(task_idx)`` + ``render()`` for the scene image.
        """
        # pylint: disable-next=import-outside-toplevel
        from predicators.agent_sdk.tools.digests import render_task_digest
        ctx = self._ctx
        task = self._task_at(task_idx)
        # The current level is the last of train_tasks in continual play;
        # its index is what task_idx, is_goal_state and train_tasks take.
        index: Optional[int] = task_idx
        if index is None:
            for i, candidate in enumerate(ctx.train_tasks):
                if candidate is task:
                    index = i
                    break
        label: Union[int, str] = "(current level)" if index is None else index
        # The is_goal_state/train_tasks query hint only makes sense in
        # sessions whose exec namespace binds those names, for a task in
        # train_tasks.
        return render_task_digest(
            task,
            label,
            ctx.predicates,
            include_goal_query_hint=(index is not None
                                     and ctx.probe_option_model_provider
                                     is not None),
            current_level=task_idx is None)

    def score(self,
              traj_idxs: Optional[List[int]] = None,
              num_particles: Optional[int] = None,
              path: Optional[str] = None) -> str:
        """Score the candidate world model on the recorded trajectories.

        Program-world-model synthesis sessions only. The particle-filter
        kernel pseudo-likelihood of the current ``world_model.py`` (or
        ``path``) over the hidden state, with per-feature errors and the
        worst transitions; ``traj_idxs`` restricts the data,
        ``num_particles`` overrides the belief size.
        """
        ctx = self._ctx
        _check_time_budget(ctx)
        provider = ctx.probe_score_provider
        if provider is None:
            raise RuntimeError(
                "sim.score is unavailable in this session: scoring happens "
                "in a program-world-model learning session.")
        return provider(path=path,
                        traj_idxs=traj_idxs,
                        num_particles=num_particles)

    def validate(self,
                 traj_idxs: Optional[List[int]] = None,
                 params: Optional[Dict[str, float]] = None) -> str:
        """Replay recorded actions at the current model's deployed parameters.

        It changes no parameter value, drops no recording and takes no
        real action. Returns errors for every selected full trajectory.
        ``traj_idxs`` indexes the available recordings, never unseen
        tasks. ``params`` optionally overrides named rule/physical
        values for this diagnostic only, e.g. estimates from a subset of
        the recordings. Keep validation data out of parameter estimation
        and model design to call it held out. Errors are not a success
        certificate for a future plan.
        """
        ctx = self._ctx
        _check_time_budget(ctx)
        provider = ctx.probe_validation_provider
        if provider is None:
            raise RuntimeError("sim.validate is unavailable in this session: "
                               "there is no candidate replay workbench.")
        # Resolve edited files exactly as sim.run does, without an implicit fit.
        self._option_model()
        return provider(traj_idxs=traj_idxs, params=params)

    def predicates(self,
                   max_trajectories: int = 10,
                   max_groundings_per_predicate: int = 4) -> str:
        """Reload ``predicates.py`` and report milestone behaviour.

        Predicate-invention synthesis sessions only. Loads
        ``LEARNED_PREDICATES`` fresh from the file (snapshotting it into
        ``predicates_versions/``), validates each entry, installs the
        set so ``run`` / ``refine`` abstract with the draft, and
        reports, per predicate x grounding over the recorded
        trajectories: coverage (ever-true / ever-false), flip counts,
        and monotonicity (a milestone flips False->True once and
        stays true). Call it after every edit of ``predicates.py``.
        """
        self._require_available("predicates")
        loader = self._artifact_loader("predicates")
        return loader(
            max_trajectories=max_trajectories,
            max_groundings_per_predicate=max_groundings_per_predicate)

    def _artifact_loader(self, name: str) -> Callable[..., str]:
        ctx = self._ctx
        _check_time_budget(ctx)
        loader = ctx.probe_artifact_loaders.get(name)
        if loader is None:
            raise RuntimeError(
                f"sim.{name} is unavailable in this session: it has no "
                f"{name}.py surface to load.")
        return loader

    def _residuals(self,
                   max_transitions: int = 100,
                   abs_tol: float = 1e-4,
                   rel_tol: float = 1e-3,
                   num_worst_examples: int = 3,
                   fit_params: bool = False,
                   path: Optional[str] = None,
                   rollout: bool = False,
                   sweep_num_points: int = 6,
                   sweep_params: Optional[Union[str, List[str]]] = None,
                   phys_params: Optional[Dict[str, float]] = None) -> str:
        """The report behind ``residuals``, for every argument a probe class
        offers."""
        ctx = self._ctx
        _check_time_budget(ctx)
        provider = ctx.probe_residuals_provider
        if provider is None:
            raise RuntimeError(
                "sim.residuals is unavailable in this session: residual "
                "reports are a learning diagnostic; there is no candidate "
                "simulator to score at solve time.")
        return provider(max_transitions=max_transitions,
                        abs_tol=abs_tol,
                        rel_tol=rel_tol,
                        num_worst_examples=num_worst_examples,
                        fit_params=fit_params,
                        path=path,
                        rollout=rollout,
                        sweep_num_points=sweep_num_points,
                        sweep_params=sweep_params,
                        phys_params=phys_params)

    # ── Execution ────────────────────────────────────────────────

    def belief(self, draws: Optional[int] = None) -> ProbeBeliefState:
        """The execution-time belief at the probe's current state: each
        object's smoothed features with their spread, and the fraction of
        belief draws on which each atom of the session's predicates holds.

        The belief is the one the last real observation showed when the
        probe still sits on that observation, else the current state
        with the declared sigma on every noisy feature (one frame's
        worth of uncertainty). ``draws`` defaults to the run's
        ``continual_belief_draws``. Needs a declared observation-noise
        channel.
        """
        self._require_available("belief")
        # pylint: disable=import-outside-toplevel
        import numpy as np

        from predicators.observation_belief import BeliefFrame, \
            atom_fractions, smooth_frames
        from predicators.observation_noise import ObservationNoise
        from predicators.settings import CFG

        # pylint: enable=import-outside-toplevel
        ctx = self._ctx
        _check_time_budget(ctx)
        if not CFG.continual_uncertainty_decisions:
            raise ValueError("Explicit uncertainty is disabled; use "
                             "sim.reset(current=True) for the point estimate.")
        noise = ObservationNoise.from_cfg()
        if not (noise.enabled and noise.declared):
            raise ValueError(
                "sim.belief() needs a declared observation-noise channel: "
                "without one the observation is exact and the belief is "
                "the state itself.")
        current = self._require_state()
        shown = getattr(ctx, "current_belief", None)
        from_observation = (isinstance(shown, BeliefFrame)
                            and ctx.current_observation is not None
                            and current.allclose(ctx.current_observation))
        belief = shown if from_observation else smooth_frames(
            [current], noise, 1, float(CFG.continual_belief_sigmas))
        assert isinstance(belief, BeliefFrame)
        num_draws = int(
            draws if draws is not None else CFG.continual_belief_draws)
        self._belief_draw_calls += 1
        rng = np.random.default_rng(CFG.seed + 15485863 *
                                    (self._instance_id + 1) +
                                    self._belief_draw_calls)
        fractions = atom_fractions(belief, set(ctx.predicates), num_draws, rng)
        hidden = excluded_object_type_names()
        objects = [
            belief.object_line(obj)
            for obj in sorted(belief.frame, key=lambda o: o.name)
            if obj.name in belief.frames_used and obj.type.name not in hidden
        ]
        return ProbeBeliefState(objects,
                                {str(a): f
                                 for a, f in fractions.items()},
                                from_observation, num_draws)

    def _run_belief_draws(self, probe_task: Task, grounded: List[Any],
                          sketch_steps: Any, all_predicates: Any,
                          num_draws: int, seed: Optional[int],
                          notices: List[str]) -> ProbeBeliefResult:
        """The ``belief_draws`` mode of :meth:`run`."""
        # pylint: disable=import-outside-toplevel
        import contextlib

        import numpy as np

        from predicators.agent_sdk import bilevel_sketch
        from predicators.observation_belief import BeliefFrame, belief_draw
        from predicators.observation_noise import ObservationNoise
        from predicators.settings import CFG

        # pylint: enable=import-outside-toplevel
        ctx = self._ctx
        noise = ObservationNoise.from_cfg()
        if not (noise.enabled and noise.declared):
            raise ValueError(
                "belief_draws needs a declared observation-noise channel: "
                "without one the start state is exact and there is nothing "
                "to draw.")
        belief = getattr(ctx, "current_belief", None)
        current = probe_task.init
        from_belief = (isinstance(belief, BeliefFrame)
                       and ctx.current_observation is not None
                       and current.allclose(ctx.current_observation))
        fresh_scope = self._fresh_scope()
        if fresh_scope is None:
            notices.append("no fresh-env scope in this session: the draws "
                           "ran on the shared env, so they are correlated.")
        self._belief_draw_calls += 1
        rng = np.random.default_rng(CFG.seed + 104729 *
                                    (self._instance_id + 1) +
                                    self._belief_draw_calls)
        starts: List[State] = []
        for _ in range(num_draws):
            if from_belief:
                assert isinstance(belief, BeliefFrame)
                starts.append(belief_draw(belief, rng))
            else:
                starts.append(noise.perturb(current, rng))

        def _shift(start: State) -> float:
            by_name = {o.name: o for o in current}
            worst = 0.0
            for obj in start:
                base = by_name.get(obj.name)
                if base is None:
                    continue
                worst = max(worst,
                            float(np.max(np.abs(start[obj] - current[base]))))
            return worst

        def _one_draw(k: int) -> Dict[str, Any]:
            task_k = dataclasses.replace(probe_task, init=starts[k])
            with (fresh_scope() if fresh_scope is not None else
                  contextlib.nullcontext()), absolute_rollout_seed(seed):
                model = self._option_model()
                r = bilevel_sketch.execute_plan_forward(
                    task_k,
                    grounded,
                    model,
                    predicates=all_predicates,
                    sketch=sketch_steps,
                    stop_on_failure=True)
            failure: Optional[str] = None
            if r.first_failure_idx is not None:
                fs = r.steps[r.first_failure_idx]
                failure = (f"step {r.first_failure_idx} "
                           f"({_fmt_option(fs.option)}): "
                           f"{fs.failure_reason or 'not initiable'}")
            return {
                "goal_reached": r.goal_reached,
                "num_actions": sum(s.num_actions for s in r.steps),
                "failure": failure,
                "max_shift": _shift(starts[k]),
            }

        # Resolve the model before forking children that build their worlds
        # from it (see _run_joint_draws).
        self._option_model()
        prefetched = prefetch_parallel(
            [functools.partial(_one_draw, k) for k in range(num_draws)],
            "probe belief draws")
        draws: List[Dict[str, Any]] = []
        try:
            for k in range(num_draws):
                _check_time_budget(ctx)
                _count_rollout(ctx)
                pre = prefetched[k]
                draws.append(pre if pre is not None else _one_draw(k))
        except ProbeBudgetExceeded as e:
            if not draws:
                raise
            notices.append(f"time budget expired after {len(draws)}/"
                           f"{num_draws} draws - the remaining draws were "
                           f"skipped ({e})")
        successes = sum(1 for d in draws if d["goal_reached"])
        return ProbeBeliefResult(draws, successes, from_belief, notices)

    def _on_current_observation(self, state: State) -> bool:
        """Whether ``state`` is the frame the last real observation showed."""
        current = self._ctx.current_observation
        return current is not None and state.allclose(current)

    def _belief_mean_start(self, state: State) -> Optional[State]:
        """The mean of the belief over the current state, when ``state`` is the
        last real observation and the session carries a belief, else None.

        The noisy frame is one draw around the true state; the belief's
        mean frame is the estimate a rollout under the joint belief
        starts from.
        """
        if not self._on_current_observation(state):
            return None
        mean_frame = getattr(getattr(self._ctx, "current_belief", None),
                             "frame", None)
        return mean_frame.copy() if mean_frame is not None else None

    def _run_joint_draws(
        self,
        probe_task: Task,
        grounded: List[Any],
        sketch_steps: Any,
        all_predicates: Any,
        num_draws: int,
        seed: Optional[int],
        notices: List[str],
        fresh: bool = False,
        from_observation: Optional[bool] = None,
        return_final: bool = False,
        score: bool = True,
        points: Optional[Sequence[Dict[str,
                                       float]]] = None) -> ProbeJointResult:
        """Rehearse the plan on ``num_draws`` joint draws of the belief.

        Each draw runs on a fresh env at its own physical parameters,
        with its rule parameters in scope, from its own start state and
        memory, at its own planner seed, so the estimate also covers
        execution variability. The draws are the session's, fixed per
        decision point (common random numbers). A draw's success is the
        task evaluator's verdict on the episode so far followed by its
        rollout, else whether the goal atoms held. ``fresh`` uses new
        draws instead of the common ones; ``from_observation`` overrides
        the check that the probe task starts at the current observation
        (a search that starts from the belief mean still rehearses the
        current state). ``return_final`` adds each draw's final state,
        sanitized, as ``final``; ``score`` False skips the task
        evaluator.

        ``points`` replaces the belief's draws with given physical
        parameter values (see :meth:`run_points`): each runs from the
        probe's own state at the same planner seed, so two points differ
        in their parameters alone.
        """
        # pylint: disable=import-outside-toplevel
        import contextlib

        import numpy as np

        from predicators.agent_sdk import bilevel_sketch
        from predicators.observation_noise import ObservationNoise
        from predicators.run.recording import sanitize_state
        from predicators.settings import CFG

        # pylint: enable=import-outside-toplevel
        ctx = self._ctx
        # Resolve the model in this process before the draws. The draws come
        # from its belief, and each forked child builds its world from it.
        # So a missing or broken simulator.py refuses once, before any draw
        # or child, and the draws follow the latest edit of the file.
        self._option_model()
        if from_observation is None:
            from_observation = self._on_current_observation(probe_task.init)
        if points is None:
            provider = ctx.joint_draws_provider
            assert provider is not None
            draws = list(provider(num_draws, fresh=fresh))
        else:
            draws = [(dict(theta), probe_task.init.copy()) for theta in points]
        # A draw of the state belief under a noisy channel starts after a
        # quasi-static settle in its own engine (_settled).
        settle = from_observation and ObservationNoise.from_cfg().enabled
        if not from_observation:
            # A hypothetical state (after reset(mods=...) or a partial
            # rollout): the parameters still vary, the state is the
            # probe's own.
            draws = [(theta, probe_task.init.copy()) for theta, _ in draws]
        fresh_scope = self._fresh_scope()
        if fresh_scope is None:
            raise ValueError(
                "The rehearsal needs the session's fresh-env scope "
                "(unavailable here); run the plan once with "
                "sim.run(plan, draws=0).")
        names_provider = ctx.physical_param_names_provider
        physical = set(names_provider() if names_provider else ())
        if points is not None:
            physical.update(name for theta in points for name in theta)
        evaluator: Any = None
        if score and ctx.probe_engine_available:
            evaluator = task_evaluator(self._base_task)
        prefix_states, prefix_labels = (self._episode_prefix(from_observation)
                                        if evaluator is not None else ([], []))
        base_seed = seed if seed is not None else CFG.seed
        draw_scope = ctx.joint_draw_scope

        def _one_draw(k: int) -> Dict[str, Any]:
            theta, start = draws[k]
            seed_index = k if points is None else 0
            phys = {n: v for n, v in theta.items() if n in physical}
            rule = {n: v for n, v in theta.items() if n not in physical}
            solved: Optional[bool] = None
            lost: Optional[bool] = None
            lost_line: Optional[int] = None
            lines_run: List[int] = []
            moving_lines: List[int] = []
            unreconstructed: List[str] = []
            memory_written: Dict[str, str] = {}
            with (draw_scope(rule) if (rule and draw_scope is not None)
                  else contextlib.nullcontext()), \
                    (fresh_scope(physical_overrides=phys)
                     if phys else fresh_scope()), \
                    absolute_rollout_seed(seed), \
                    decorrelated_rollout_seed(seed_index):
                model = self._option_model()
                if settle:
                    start = _settled(getattr(model, "sim_env", None), start)
                task_k = dataclasses.replace(probe_task, init=start)
                collector = (_EvalStateCollector(model, start)
                             if evaluator is not None else None)

                def _on_step(i: int, outcome: Any) -> None:
                    # The first step resets the fresh world to the draw's
                    # start; a feature the model reads back otherwise
                    # means the draw ran from another state.
                    if i == 0:
                        unreconstructed.extend(
                            _unreconstructed(getattr(model, "sim_env", None)))
                    if collector is not None:
                        collector.on_step(i, outcome)

                r = bilevel_sketch.execute_plan_forward(
                    task_k,
                    grounded,
                    model,
                    predicates=all_predicates,
                    sketch=sketch_steps,
                    on_step=_on_step,
                    stop_on_failure=True)
                memory_written = _memory_written(
                    getattr(model, "sim_env", None))
                if (collector is not None and not collector.coarse
                        and len(collector.states) > 1):
                    try:
                        # A real episode stops at its first terminal step,
                        # won or lost, so the draw is judged on the states
                        # up to it: a win there is not undone by later
                        # lines.
                        end = collector.first_terminal(evaluator,
                                                       prefix_states)
                        upto = len(collector.states) if end is None else end
                        verdict = evaluate_states_with(
                            evaluator,
                            prefix_states + collector.states[:upto],
                            prefix_labels + collector.labels[:upto - 1],
                            sim_env=getattr(model, "sim_env", None))
                        solved = bool(verdict["solved"])
                        # The evaluator ends the episode without counting
                        # it, as a real one ends in GAME_OVER.
                        lost = bool(verdict["terminated"]) and not solved
                        if lost and end is not None:
                            # Where it was lost: the rehearsal gate weighs
                            # only the losses a request's own lines risk; a
                            # draw lost at its start charges no line.
                            lost_line = collector.line_of(end)
                        # Where the scene was still moving: the gate asks
                        # that a request's rehearsal run until it rests.
                        lines_run = [line for line, _ in collector.line_ends]
                        moving_lines = [
                            line for line, count in collector.line_ends
                            if collector.moving_at(count)
                        ]
                    except Exception as e:  # pylint: disable=broad-except
                        # The report says the draw went unscored; the log
                        # keeps why.
                        logging.warning("Joint-draw verdict failed: %r", e)
                        solved = lost = None
                        lost_line = None
            failure: Optional[str] = None
            if r.first_failure_idx is not None:
                fs = r.steps[r.first_failure_idx]
                failure = (f"step {r.first_failure_idx} "
                           f"({_fmt_option(fs.option)}): "
                           f"{fs.failure_reason or 'not initiable'}")
            final: Optional[State] = None
            if return_final:
                last = start
                for st in r.steps:
                    if st.post_state is not None:
                        last = st.post_state
                final = sanitize_state(last)
            # What a request that starts this plan should leave behind,
            # line by line: the predictive check scores its real outcome
            # on these.
            line_states = (_line_states(r.steps) if from_observation else [])
            success = (bool(solved) if solved is not None else
                       bool(r.goal_reached) and failure is None)
            return {
                "final": final,
                "line_states": line_states,
                "params": dict(theta),
                "goal_reached": bool(r.goal_reached),
                "solved": solved,
                "lost": lost,
                "lost_line": lost_line,
                "lines_run": lines_run,
                "moving_lines": moving_lines,
                "success": success,
                "num_actions": sum(st.num_actions for st in r.steps),
                "failure": failure,
                "planner_seed": base_seed + seed_index,
                "unreconstructed": sorted(set(unreconstructed)),
                "memory_written": memory_written,
            }

        prefetched = prefetch_parallel(
            [functools.partial(_one_draw, k) for k in range(len(draws))],
            "probe joint draws" if points is None else "probe points")
        results: List[Dict[str, Any]] = []
        try:
            for k in range(len(draws)):
                _check_time_budget(ctx)
                _count_rollout(ctx)
                pre = prefetched[k]
                results.append(pre if pre is not None else _one_draw(k))
        except ProbeBudgetExceeded as e:
            if not results:
                raise
            notices.append(f"time budget expired after {len(results)}/"
                           f"{len(draws)} draws - the remaining draws were "
                           f"skipped ({e})")
        n = len(results)
        p_hat = sum(1 for d in results if d["success"]) / n
        stderr = float(np.sqrt(p_hat * (1.0 - p_hat) / n))
        off_start = [d for d in results if d.get("unreconstructed")]
        if off_start:
            features = ", ".join(
                sorted({f
                        for d in off_start for f in d["unreconstructed"]}))
            notices.append(
                f"{len(off_start)} of {n} draws did not start from the state "
                f"they were given: after setting it, simulator.py reads back "
                f"other values of {features}, so those draws ran from another "
                "state. Make the model set and report these features exactly, "
                "or this rehearsal predicts from the wrong start.")
        wrote = [d for d in results if d.get("memory_written")]
        if wrote:
            notices.append(
                memory_written_note(
                    f"In {len(wrote)} of {n} draws", {
                        key: value
                        for d in wrote
                        for key, value in d["memory_written"].items()
                    }))
        budget = ctx.execution_step_budget()
        over = [
            d for d in results if d["success"] and d["num_actions"] > budget
        ]
        if over:
            notices.append(
                f"{len(over)} successful draw(s) exceeded the remaining "
                f"execution step budget ({budget} low-level steps) - the "
                "real executor would run out of steps.")
        stalled = _motion_failures([d["failure"] for d in results])
        if stalled:
            motion = _motion_failure_note(
                f"In {stalled} of the {n} draws a skill's controller failed.",
                _failed_skills([d["failure"] for d in results]),
                _skill_record(ctx),
                _belief_frames_note(ctx, grounded)
                if from_observation else None,
                _belief_spread(ctx, grounded) if from_observation else None)
            if motion is not None:
                notices.append(motion)
        undeclared_provider = ctx.undeclared_param_names_provider
        undeclared = set(undeclared_provider() if undeclared_provider else ())
        leads = ([] if points is not None or _separating_ranges(results) else
                 _outcome_leads(results, undeclared))
        return ProbeJointResult(results,
                                p_hat,
                                stderr,
                                from_observation,
                                notes=notices,
                                leads=leads)

    def _require_solved_evaluator(self, flag: str) -> Any:
        """Precondition gate shared by ``run(solved=True)`` and
        ``refine(require_solved=True)``, so the two evaluator-scored surfaces
        can never drift apart.

        Returns the rule that judges the task (``task_evaluator``: its
        evaluator, else its goal atoms). Raises when the current state
        is not the task's unmodified initial state (the evaluator's
        rules reference the true init, so any other start would give
        silently wrong verdicts) or when the task has no goal atoms to
        judge.
        """
        if not self._ctx.probe_engine_available:
            # The standalone model's session: the refusal names the flag
            # the caller passed, and nothing the session lacks.
            raise ValueError(
                f"{flag} is not part of `sim` in this session: the "
                "run_python description lists what `sim` offers here.")
        if not self._pristine:
            raise ValueError(
                f"{flag} needs the task's unmodified initial state (call "
                "reset() with no mods first, before any state-advancing "
                "run()): the evaluator's rules reference the true init, so "
                "a verdict from a modified or advanced state would be "
                "silently wrong.")
        assert self._base_task is not None
        evaluator = task_evaluator(self._base_task)
        if evaluator is None:
            raise ValueError(f"{flag}: this task has no goal atoms to judge.")
        return evaluator

    def _episode_prefix(
            self, from_observation: bool) -> Tuple[List[State], List[Any]]:
        """The recorded episode before the probe's start, with its labels.

        A rollout from the current observation is scored after these
        states; its own start replaces the episode's last one. Empty
        when the probe is not on the current observation.
        """
        provider = self._ctx.episode_prefix_provider
        if not from_observation or provider is None:
            return [], []
        states, labels = provider()
        return list(states)[:-1], list(labels)

    def _score_rollout(self, rollout: ProbeResult,
                       from_observation: bool) -> Optional[Dict[str, bool]]:
        """Score ``rollout`` as a rehearsal draw is scored.

        The task evaluator judges the episode so far followed by the
        rollout, up to where the episode first ends. None when it cannot
        be scored (no engine, no evaluator, per-option states only).
        """
        if (not self._ctx.probe_engine_available or rollout.coarse
                or len(rollout.states) < 2
                or len(rollout.labels) != len(rollout.states) - 1):
            return None
        evaluator = task_evaluator(self._base_task)
        if evaluator is None:
            return None
        prefix_states, prefix_labels = self._episode_prefix(from_observation)
        collector = _EvalStateCollector(None, rollout.states[0])
        collector.states = list(rollout.states)
        collector.labels = list(rollout.labels)
        try:
            end = collector.first_terminal(evaluator, prefix_states)
            upto = len(collector.states) if end is None else end
            verdict = evaluate_states_with(
                evaluator,
                prefix_states + collector.states[:upto],
                prefix_labels + collector.labels[:upto - 1],
                sim_env=getattr(self._option_model(), "sim_env", None))
        except Exception as e:  # pylint: disable=broad-except
            # As for a draw: the report says nothing, the log keeps why.
            logging.warning("Mean-rollout verdict failed: %r", e)
            return None
        solved = bool(verdict["solved"])
        return {
            "solved": solved,
            "lost": bool(verdict["terminated"]) and not solved
        }

    def run_points(self,
                   plan_text: str,
                   points: Sequence[Dict[str, float]],
                   seed: Optional[int] = None) -> List[Dict[str, Any]]:
        """Run ``plan_text`` from the current state once per point of physical
        parameter values, each on a fresh env at those values with every other
        value at the planning model's.

        The runs share the start state and the planner seed, so two of
        them differ in their parameters alone. Returns one record per
        point, a joint draw's (``params``, ``success``, ``solved``,
        ``lost``, ``lost_line``, ``failure``); fewer when the time
        budget ends the runs early.
        """
        _check_time_budget(self._ctx)
        probe_task, sketch_steps, all_predicates, _ = \
            self._parse_sketch(plan_text)
        result = self._run_joint_draws(probe_task,
                                       _grounded_steps(sketch_steps),
                                       sketch_steps,
                                       all_predicates,
                                       len(points),
                                       seed, [],
                                       points=points)
        return list(result.draws)

    def _run_rehearsal(self, plan_text: str, render: bool,
                       seed: Optional[int]) -> ProbeJointResult:
        """``run(plan)`` under the joint belief: the K joint draws, then the
        rollout from the belief mean (see :meth:`run`)."""
        # pylint: disable-next=import-outside-toplevel
        from predicators.settings import CFG
        ctx = self._ctx
        _check_time_budget(ctx)
        ctx.test_call_id += 1
        probe_task, sketch_steps, all_predicates, notices = \
            self._parse_sketch(plan_text)
        result = self._run_joint_draws(probe_task,
                                       _grounded_steps(sketch_steps),
                                       sketch_steps, all_predicates,
                                       int(CFG.belief_joint_draws), seed,
                                       notices)
        # The step-by-step rollout starts from the belief mean when the
        # probe sits on the current observation (run moves it there).
        result.mean = self.run(plan_text,
                               render=render,
                               contacts=bool(ctx.probe_engine_available),
                               seed=seed,
                               draws=0)
        if isinstance(result.mean, ProbeResult):
            if any(d.get("memory_written") for d in result.draws):
                result.mean.notes[:] = [
                    note for note in result.mean.notes
                    if not note.startswith(_RUN_MEMORY_LEAD)
                ]
            beyond = _beyond_recordings(ctx, result.mean.states)
            if beyond:
                result.notes.append(beyond)
            faster = _faster_than_recorded(ctx, result.mean.states)
            if faster:
                result.notes.append(faster)
            if CFG.continual_rehearsal_mean_verdict:
                result.mean_verdict = self._score_rollout(
                    result.mean, result.from_observation)
                disagreement = _mean_disagreement_note(result)
                if disagreement:
                    result.notes.append(disagreement)
        if ctx.reach_note_provider is not None:
            reach = ctx.reach_note_provider()
            if reach:
                result.notes.append(reach)
        if ctx.bound_note_provider is not None:
            bound = ctx.bound_note_provider()
            if bound:
                result.notes.append(bound)
        _log_rehearsal(ctx, plan_text, result)
        if result.from_observation and ctx.current_observation is not None:
            # What the rehearsal gate lets through: this plan, from this
            # observation (the bound keeps a long session's list small).
            basis = (ctx.prediction_basis_provider()
                     if ctx.prediction_basis_provider is not None else None)
            mean_states = (list(result.mean.line_states) if isinstance(
                result.mean, ProbeResult) else [])
            ctx.live_rehearsals.append(
                LiveRehearsal(
                    ctx.current_observation,
                    plan_text,
                    float(result.p_hat),
                    [list(d.get("line_states") or []) for d in result.draws],
                    basis,
                    mean_states, [d.get("lost_line") for d in result.draws],
                    scored=[d.get("lost") is not None for d in result.draws],
                    lines_run=[
                        list(d.get("lines_run") or []) for d in result.draws
                    ],
                    moving_lines=[
                        list(d.get("moving_lines") or []) for d in result.draws
                    ],
                    notes=list(result.notes)))
            del ctx.live_rehearsals[:-_LIVE_REHEARSALS_KEPT]
        return result

    # More modes and results than the rollout core's ``run``: its arguments
    # sit among the modes', and each mode returns its own result class.
    def run(  # type: ignore[override]  # pylint: disable=arguments-renamed
        self,
        plan_text: str,
        render: bool = True,
        trials: int = 1,
        solved: bool = False,
        contacts: bool = False,
        physics_sweep: bool = False,
        seed: Optional[int] = None,
        fresh: bool = False,
        belief_draws: int = 0,
        draws: Optional[int] = None,
    ) -> Union[ProbeResult, ProbeTrialsResult, ProbeSweepResult,
               ProbeBeliefResult, ProbeJointResult]:
        """Execute an option plan from the current state.

        Under the joint belief (``belief_joint_draws`` K > 0), a plain
        ``run(plan)`` rehearses the plan on the K joint draws of the
        belief (a parameter draw, a state draw and the memory it implies,
        a fresh env and a planner seed per draw) and reports the success
        estimate P-hat with its standard error, each draw's outcome and
        the parameter ranges on which draws fail, followed by a
        step-by-step rollout from the belief mean at the planning
        model's values, with contacts, which advances the current state.
        ``draws=0`` runs only that single rollout: from the belief mean
        when the probe sits on the last real observation, else from the
        probe's state. ``draws=K`` is the rehearsal, and any other count
        is refused. The ``belief_draws``,
        ``trials`` and ``solved`` modes below apply only without the
        joint belief (``belief_joint_draws`` 0); under it they raise,
        since the rehearsal replaces them. ``physics_sweep``,
        ``contacts`` and ``fresh`` keep their meaning.

        Without the joint belief, ``belief_draws=K`` (K > 0, its own
        mode) rolls the plan once from each of K draws of the
        execution-time belief - where the objects may really be given
        the observation and the declared noise (the smoothed frame and
        spread the last observation showed when the probe still sits on
        it, else the current state with the declared sigma) - on a fresh
        env at the base planner seed, and returns a
        ``ProbeBeliefResult``. This is how a
        placement is certified over the target's plausible poses before
        real steps are spent on it. Needs a declared observation-noise
        channel.

        ``plan_text`` uses the same grammar as ``skills_execute_plan``:
        one option per line, ``Option(obj:type, ...)[params]`` with
        exact continuous params (``[]`` for none); ``-> {atoms}``
        subgoal annotations are optional but CHECKED - each step's
        report lists any annotated atoms that did not hold in its
        post-state, which makes a single continuous ``run`` of a
        refined plan the forward-validation pass (``refine`` resets
        state between options and resamples, so a refine-pass that
        diverges here means a rule is more permissive than the env).
        Advances the current state
        to the rollout's final state (``restore`` a snapshot to rewind).
        Each step's post-state is rendered to a saved image whose path
        lands in the step report; pass ``render=False`` inside tight
        sweep loops to skip that. Exploratory only: nothing runs in the
        environment.

        Without the joint belief, ``trials=N`` (N > 1) runs the SAME
        plan N times and returns a ``ProbeTrialsResult`` with the
        per-trial outcomes and success count - use it to estimate a
        plan's reliability instead of hand-rolled repeat loops. Each
        trial runs on a freshly constructed physics env when the session
        provides one (repeats on the shared env - including
        snapshot/restore replays - share solver state and velocity
        residuals, so they are correlated and read optimistic). With
        ``trials > 1`` nothing is rendered and the current state is NOT
        advanced.

        Without the joint belief, ``solved=True`` (needs ``trials`` >= 2
        and the task's UNMODIFIED initial state - plain ``reset()``, no
        rollout since) additionally scores every trial with the TASK
        EVALUATOR, reporting per-trial ``solved``/``reward``. Reaching
        the goal atoms is NOT the same as being scored a solve - the
        evaluator can reject a goal-reaching route - so check ``solved``
        counts here BEFORE executing the plan in the environment.

        ``contacts=True`` (single-run mode only, ``trials=1``) records
        every physical contact during the rollout and reports, per step,
        which robot links touched which objects and which object pairs
        came into contact, with low-level action spans. Use it to check
        WHAT actually caused motion - e.g. whether a topple was driven
        by the object you pushed or by the robot's body brushing the
        scene.

        ``physics_sweep=True`` (its own mode - incompatible with
        ``trials``/``solved``/``contacts``) re-runs the plan once at
        each stress point - under the joint belief, each physical
        parameter at the ends of its 95% interval with the others at
        their most likely values; without it, a grid spanning the
        +-1-sigma uncertainty range of the identified physical
        parameters - each on
        a fresh env at the base motion-planner seed, plus once at the
        planning model's values, and returns a ``ProbeSweepResult`` with
        per-point outcomes. Those
        values carry real uncertainty and the true environment
        may sit anywhere in the range - and success can be
        NON-MONOTONIC in a physical parameter (a cascade that topples
        just above and just below some friction can fail exactly at
        it), so ``trials=`` at those values CANNOT see this. A
        design near a feasibility boundary (e.g. the minimal block
        count) is exactly where such holes live: sweep it and add
        margin until EVERY point passes before executing it for real.
        Rollouts are deterministic per point, so each point costs one
        rollout and its outcome is a measurement, not a sample. The
        current state is NOT advanced and nothing is rendered.

        ``seed=S`` overrides the base motion-planner seed for this call.
        Trials report the planner seed each ran at (trial ``i`` runs at
        ``S + i``; without ``seed=`` at ``base + i``). A single run
        (``trials=1``) executes entirely at ``S``; a physics sweep runs
        every point at ``S`` instead of the base.

        SUBSTRATE: a default single run executes on the WARM shared
        session env from the probe's current state - a feature for
        mid-exploration probing (after ``reset(mods=...)`` or a partial
        rollout), but a different physics substrate from trials, physics
        sweeps and the real episode, which all run on a freshly
        constructed env. A warm pass at a seed where a fresh
        trial failed is evidence of shared-env optimism, not seed luck.
        ``fresh=True`` (single-run mode only) reproduces a failed
        trial's substrate exactly: ``run(plan, seed=<reported seed>,
        fresh=True)`` runs once on a fresh env from the parsed start
        state, without advancing the shared session state (per-step
        renders are skipped - the renderer draws the shared env).
        """
        # pylint: disable-next=import-outside-toplevel
        import contextlib

        # pylint: disable-next=import-outside-toplevel
        from predicators.agent_sdk import bilevel_sketch
        # pylint: disable-next=import-outside-toplevel
        from predicators.settings import CFG
        joint = (int(CFG.belief_joint_draws) > 0
                 and self._ctx.joint_draws_provider is not None
                 and CFG.continual_uncertainty_decisions)
        if draws is not None and not joint:
            raise ValueError("draws= needs the joint belief, which this "
                             "session does not carry.")
        if joint and draws not in (None, 0):
            # The draws are fixed per decision point: a count other than
            # theirs would quietly mean something else.
            if draws != int(CFG.belief_joint_draws):
                raise ValueError(
                    f"draws={draws!r}: the rehearsal runs on the "
                    f"{int(CFG.belief_joint_draws)} joint draws fixed at "
                    "this decision point. Leave draws out to rehearse on "
                    "them, or pass draws=0 for one rollout from the belief "
                    "mean.")
            draws = None
        if joint and (trials > 1 or solved or belief_draws > 0):
            raise ValueError(
                "trials, solved and belief_draws are replaced by the "
                "rehearsal on the joint belief: sim.run(plan) reports the "
                "success estimate over K joint draws (parameters, state and "
                "memory), scored by the task evaluator.")
        if joint and draws is None and not (contacts or fresh
                                            or physics_sweep):
            return self._run_rehearsal(plan_text, render, seed)
        if not self._ctx.probe_engine_available and (solved or contacts
                                                     or physics_sweep):
            # As above: the refusal names the flags the caller passed.
            raise ValueError(
                "sim.run takes no `solved`, `contacts` or `physics_sweep` "
                "flag in this session: the run_python description lists "
                "what `sim` offers here.")
        if trials < 1:
            raise ValueError(f"trials must be >= 1, got {trials}")
        if trials > 1 and "trials" in self._ctx.probe_disabled:
            raise RuntimeError(
                "sim.run runs a plan once per call in this session: leave "
                "`trials` out.")
        if belief_draws > 0:
            self._require_available("belief")
        if "render" in self._ctx.probe_disabled:
            # Per-step scene images come from the real engine; an arm
            # without one gets text-only rollouts.
            render = False
        if (not CFG.continual_uncertainty_decisions
                and (physics_sweep or belief_draws > 0)):
            raise ValueError("Explicit uncertainty is disabled; rehearse at "
                             "the point estimate without uncertainty sweeps.")
        if physics_sweep and (trials > 1 or solved or contacts):
            raise ValueError(
                "physics_sweep=True is its own mode: it varies the PHYSICS "
                "per rollout (one deterministic rollout per parameter "
                "point), while trials/solved/contacts measure the plan at "
                f"the {_reference_point()}. Run them as separate calls.")
        if belief_draws < 0:
            raise ValueError(f"belief_draws must be >= 0, got {belief_draws}")
        if belief_draws > 0 and (trials > 1 or solved or contacts
                                 or physics_sweep or fresh):
            raise ValueError(
                "belief_draws=K is its own mode: it varies the START STATE "
                "over the belief (one rollout per draw); run trials, "
                "solved, contacts, physics_sweep and fresh as separate "
                "calls.")
        if solved and trials < 2:
            raise ValueError(
                "solved=True needs trials >= 2: a single shared-env rollout "
                "is a correlated sample, so a one-off verdict would read "
                "optimistic. Use trials>=2 (fresh env per trial), or "
                "refine(..., require_solved=True) to search for "
                "evaluator-solved params.")
        if contacts and trials > 1:
            raise ValueError(
                "contacts=True is only available in single-run mode "
                "(trials=1): per-contact recording is a diagnostic for one "
                "rollout, not a statistic.")
        if fresh and (trials > 1 or physics_sweep):
            raise ValueError(
                "fresh=True applies to single runs only: trials and "
                "physics_sweep already run every rollout on a freshly "
                "constructed env.")
        if fresh and contacts:
            raise ValueError(
                "fresh=True cannot record contacts: contact recording "
                "binds to the shared session env. Run contacts=True "
                "without fresh (the warm substrate), or diagnose the "
                "fresh-substrate failure via the step report alone.")
        ctx = self._ctx
        _check_time_budget(ctx)
        ctx.test_call_id += 1
        probe_task, sketch_steps, all_predicates, notices = \
            self._parse_sketch(plan_text)
        mean_start: Optional[State] = None
        if joint and not (fresh or physics_sweep):
            # A single run under the joint belief starts where the
            # rehearsal's rollout does; the probe's state moves only when
            # the run completes.
            mean_start = self._belief_mean_start(probe_task.init)
            if mean_start is not None:
                probe_task = dataclasses.replace(probe_task, init=mean_start)
        # Ground via the shared helper so an annotated Wait waits for
        # its annotated atoms here exactly as in refine and real
        # execution.
        grounded = _grounded_steps(sketch_steps)

        evaluator: Any = None
        if solved:
            evaluator = self._require_solved_evaluator("solved=True")

        if belief_draws > 0:
            return self._run_belief_draws(probe_task, grounded, sketch_steps,
                                          all_predicates, belief_draws, seed,
                                          notices)

        if physics_sweep:
            fresh_scope = self._fresh_scope()
            if fresh_scope is None:
                raise ValueError(
                    "physics_sweep=True needs the session's fresh-env "
                    "scope (unavailable here): perturbing the shared "
                    "session env would leak the perturbation into every "
                    "later call.")
            # Resolve the model before reading its stress points and forking
            # children that build their worlds from it (see
            # _run_joint_draws).
            self._option_model()
            provider = ctx.physics_margin_provider
            sweep_points = list(provider() or []) if provider else []
            if not sweep_points and CFG.agent_sim_learn_fit_invisible:
                raise ValueError(
                    "physics_sweep=True, but the model declares no "
                    "physical parameter with a range to sweep. Rehearse "
                    "the plan with sim.run(plan) instead.")
            if not sweep_points:
                raise ValueError(
                    "physics_sweep=True, but no identified physical "
                    "parameters with nonzero posterior width are deployed "
                    "this cycle, so there is no uncertainty range to "
                    "sweep. Use trials= to measure execution reliability "
                    "at the current physics instead.")
            point_dicts: List[Dict[str, Any]] = []
            all_points: List[Optional[Dict[str, float]]] = \
                [None] + sweep_points
            # Rebound so the nested function sees the narrowed
            # (non-Optional) type - the None-check above cannot
            # propagate into a closure.
            sweep_scope = fresh_scope

            def _one_point(
                    point: Optional[Dict[str, float]]) -> Dict[str, Any]:
                # Base planner seed at every point (no
                # decorrelated_rollout_seed): with the seed held fixed, an
                # outcome flip between points is attributable to the
                # physics perturbation alone.
                with (sweep_scope() if point is None else sweep_scope(
                        physical_overrides=point)), \
                        absolute_rollout_seed(seed):
                    model = self._option_model()
                    r = bilevel_sketch.execute_plan_forward(
                        probe_task,
                        grounded,
                        model,
                        predicates=all_predicates,
                        sketch=sketch_steps,
                        stop_on_failure=True)
                point_failure: Optional[str] = None
                if r.first_failure_idx is not None:
                    fs = r.steps[r.first_failure_idx]
                    point_failure = (f"step {r.first_failure_idx} "
                                     f"({_fmt_option(fs.option)}): "
                                     f"{fs.failure_reason or 'not initiable'}")
                return {
                    "params": point,
                    "goal_reached": r.goal_reached,
                    "num_actions": sum(s.num_actions for s in r.steps),
                    "failure": point_failure,
                }

            point_prefetched = prefetch_parallel(
                [functools.partial(_one_point, pt) for pt in all_points],
                "probe physics sweep")
            try:
                for point_idx, point in enumerate(all_points):
                    _check_time_budget(ctx)
                    _count_rollout(ctx)
                    pre = point_prefetched[point_idx]
                    point_dicts.append(
                        pre if pre is not None else _one_point(point))
            except ProbeBudgetExceeded as e:
                # Same salvage rule as trials mode: completed points are
                # sim time living in the return value, not stdout.
                if not point_dicts:
                    raise
                notices.append(
                    f"time budget expired after {len(point_dicts)}/"
                    f"{len(all_points)} sweep points - the remaining "
                    f"points were skipped ({e})")
            budget = ctx.execution_step_budget()
            sweep_over = [
                p for p in point_dicts
                if p["goal_reached"] and p["num_actions"] > budget
            ]
            if sweep_over:
                notices.append(
                    f"{len(sweep_over)} goal-reaching sweep point(s) "
                    f"exceeded the remaining execution step budget ({budget} "
                    "low-level steps) - the real executor would run out "
                    "of steps.")
            sweep_successes = sum(1 for p in point_dicts if p["goal_reached"])
            if CFG.code_sim_learning_interval_belief:
                # pylint: disable-next=import-outside-toplevel
                from predicators.code_sim_learning.identifiability import \
                    straddle_summary

                # Certification over the belief interval: a mixed sweep
                # means the interval straddles the plan's success
                # boundary, which is the probe trigger - one narrowing
                # experiment beats more planning, so adaptive
                # info-seeking is armed here, not only by a refusal.
                interval_pts = [
                    p for p in point_dicts if p["params"] is not None
                ]
                n_ok = sum(1 for p in interval_pts if p["goal_reached"])
                if interval_pts and 0 < n_ok < len(interval_pts):
                    straddle = straddle_summary(
                        [p["params"] for p in interval_pts],
                        [bool(p["goal_reached"]) for p in interval_pts])
                    notices.append(
                        "the belief interval straddles this plan's success "
                        f"boundary ({n_ok}/{len(interval_pts)} interval "
                        "points passed" +
                        (f"; {straddle}" if straddle else "") + "). One "
                        "real experiment that narrows that parameter is "
                        "worth more than more planning here: "
                        "sim.suggest_probes(plan_text) ranks probes on your "
                        "sketch (adaptive info-seeking is now armed), or "
                        "find a design that holds across the whole "
                        "interval.")
                    ctx.param_sensitive_refusal_pending = True
            return ProbeSweepResult(point_dicts, sweep_successes, notices)

        if trials > 1:
            # The candidate-aware scope preserves the deployed model while
            # replacing physics for every trial, including synthesis.
            fresh_scope = self._fresh_scope()
            trial_dicts: List[Dict[str, Any]] = []
            base_planner_seed = seed if seed is not None else CFG.seed

            def _one_trial(trial_idx: int) -> Dict[str, Any]:
                trial_solved: Optional[bool] = None
                trial_reward: Optional[float] = None
                trial_note = ""
                evaluation_status = ("unavailable" if evaluator is not None
                                     else "not_requested")
                evaluation_error = ("per-step states missing"
                                    if evaluator is not None else "")
                coarse = False
                # decorrelated_rollout_seed: a fresh env alone gives
                # bit-identical repeats (motion planning reads the
                # constant CFG.seed), so without it N trials are one
                # effective sample. Entered inside the fresh scope so
                # env construction keeps the base seed.
                trial_inexact: List[str] = []
                with (fresh_scope() if fresh_scope is not None else
                      contextlib.nullcontext()), \
                        absolute_rollout_seed(seed), \
                        decorrelated_rollout_seed(trial_idx):
                    model = self._option_model()
                    collector = (_EvalStateCollector(model, probe_task.init)
                                 if evaluator is not None else None)

                    def _trial_on_step(i: int, outcome: Any) -> None:
                        # The first option's execution resets the
                        # (fresh) env to the parsed start state; any
                        # features that reset could not round-trip
                        # are a start-state reconstruction error, a
                        # failure source distinct from plan margin.
                        if i == 0:
                            env = getattr(model, "sim_env", None)
                            lossy = getattr(
                                env, "_last_unreconstructible_features",
                                None) or []
                            trial_inexact.extend(f"{obj.name}.{feat}"
                                                 for obj, feat in lossy)
                        if collector is not None:
                            collector.on_step(i, outcome)

                    r = bilevel_sketch.execute_plan_forward(
                        probe_task,
                        grounded,
                        model,
                        predicates=all_predicates,
                        sketch=sketch_steps,
                        on_step=_trial_on_step,
                        stop_on_failure=True)
                    # Score INSIDE the scope: the evaluator's
                    # certificate probes at the (fresh) env the
                    # rollout ran on.
                    if collector is not None:
                        coarse = collector.coarse
                    if (collector is not None and not coarse
                            and len(collector.states) > 1):
                        try:
                            verdict = evaluate_states_with(
                                evaluator,
                                collector.states,
                                collector.labels,
                                sim_env=getattr(model, "sim_env", None))
                            # Reward first: it is the only fallible
                            # conversion, so a malformed verdict is
                            # caught below with BOTH fields still None
                            # instead of crashing the loop and losing
                            # the completed trials.
                            trial_reward = float(verdict["reward"])
                            trial_solved = bool(verdict["solved"])
                            trial_note = str(verdict.get("note") or "")
                            evaluation_status = ("accepted" if trial_solved
                                                 else "rejected")
                            evaluation_error = ""
                        except Exception as e:  # pylint: disable=broad-except
                            logging.debug("Trial evaluator verdict failed: %s",
                                          e)
                            trial_solved = None
                            trial_reward = None
                            evaluation_status = "error"
                            # Exception messages can contain private task
                            # rules or values. Keep details in host logs.
                            evaluation_error = type(e).__name__
                failure: Optional[str] = None
                if r.first_failure_idx is not None:
                    fs = r.steps[r.first_failure_idx]
                    failure = (f"step {r.first_failure_idx} "
                               f"({_fmt_option(fs.option)}): "
                               f"{fs.failure_reason or 'not initiable'}")
                total = sum(s.num_actions for s in r.steps)
                return {
                    "goal_reached": r.goal_reached,
                    "num_actions": total,
                    "failure": failure,
                    "solved": trial_solved,
                    "reward": trial_reward,
                    "note": trial_note,
                    "verdict_coarse": coarse,
                    "evaluation_status": evaluation_status,
                    "evaluation_error": evaluation_error,
                    "planner_seed": base_planner_seed + trial_idx,
                    "inexact_start_features": sorted(set(trial_inexact)),
                }

            # Resolve the model before forking children that build their
            # worlds from it (see _run_joint_draws).
            self._option_model()
            trial_prefetched = prefetch_parallel(
                [functools.partial(_one_trial, k) for k in range(trials)],
                "probe trials")
            try:
                for trial_idx in range(trials):
                    _check_time_budget(ctx)
                    _count_rollout(ctx)
                    pre = trial_prefetched[trial_idx]
                    trial_dicts.append(
                        pre if pre is not None else _one_trial(trial_idx))
            except ProbeBudgetExceeded as e:
                # Completed trials are minutes of sim time and live in the
                # RETURN VALUE, not stdout - discarding them on a mid-loop
                # budget stop would force re-simulating them. Return the
                # partial result instead; only an empty result re-raises.
                if not trial_dicts:
                    raise
                notices.append(
                    f"time budget expired after {len(trial_dicts)}/{trials} "
                    f"trials - the remaining trials were skipped ({e})")
            successes = sum(1 for t in trial_dicts if t["goal_reached"])
            if evaluator is not None:
                if any(t["verdict_coarse"] for t in trial_dicts):
                    notices.append(
                        "per-step states were unavailable for some trials; "
                        "their evaluator checks are unavailable, NOT passes.")
                if any(t["evaluation_status"] == "error" for t in trial_dicts):
                    notices.append(
                        "the evaluator errored on some trials; these are "
                        "NOT certified, even if the goal atoms held.")
            budget = ctx.execution_step_budget()
            over = [
                t for t in trial_dicts
                if t["goal_reached"] and t["num_actions"] > budget
            ]
            if over:
                notices.append(
                    f"{len(over)} goal-reaching trial(s) exceeded the "
                    "remaining "
                    f"execution step budget ({budget} low-level steps) - "
                    "the real executor would run out of steps.")
            inexact = [t for t in trial_dicts if t["inexact_start_features"]]
            if inexact:
                feats = sorted(
                    {f
                     for t in inexact for f in t["inexact_start_features"]})
                shown = ", ".join(
                    feats[:8]) + ("..." if len(feats) > 8 else "")
                notices.append(
                    f"{len(inexact)}/{len(trial_dicts)} trials started from "
                    f"an inexactly reconstructed state (features: {shown}) "
                    "- failures may partly reflect start-state "
                    "reconstruction error, not plan margin.")
            return ProbeTrialsResult(trial_dicts, successes, fresh_scope
                                     is not None, notices)

        # The plain mode: one rollout, the rollout core's. Under the joint
        # belief it started at the belief mean, and a controller failure's
        # note then says how well the belief places the plan's bodies.
        return self._run_once(
            probe_task,
            grounded,
            sketch_steps,
            all_predicates,
            notices,
            render=render,
            contacts=contacts,
            seed=seed,
            fresh=fresh,
            motion_context=((lambda: (_belief_frames_note(ctx, grounded),
                                      _belief_spread(ctx, grounded)))
                            if mean_start is not None else None))

    def _rollouts_in(self, run_kwargs: Dict[str, Any]) -> int:
        return max(1, int(run_kwargs.get("trials", 1)))

    def run_policy(
        self,
        trials: int = 1,
        seed: Optional[int] = None,
        source: Optional[str] = None,
    ) -> Union[ProbeResult, ProbeTrialsResult]:
        """Run ./policy.py closed-loop FROM THE CURRENT STATE.

        Policy-mode counterpart of ``run``: loads the sandbox's
        ``policy.py`` fresh (or executes ``source`` directly) and drives
        its ``get_option(state, memory)`` through the belief model -
        option failures land in ``memory['last_failure']`` and the
        policy is asked again; get_option bugs end the episode.

        Starting from the CURRENT probe state is the point: perturb or
        advance the state first (``reset(mods=...)``, a partial ``run``)
        and check that the policy RECOVERS from off-nominal states, not
        just the initial one. ``trials=N`` repeats the rollout from the
        SAME current state on fresh envs (fresh policy memory per
        trial). Nothing runs in the environment. Also
        available in learn sessions (probing a candidate simulator);
        there it runs against the candidate model.
        """
        self._require_available("run_policy")
        # pylint: disable=import-outside-toplevel
        import contextlib

        from predicators.agent_sdk.policy_execution import \
            build_policy_option_fn, execute_policy_forward
        from predicators.settings import CFG

        # pylint: enable=import-outside-toplevel
        ctx = self._ctx
        _check_time_budget(ctx)
        ctx.test_call_id += 1
        if trials < 1:
            raise ValueError(f"trials must be >= 1, got {trials}")
        if source is None:
            path = _policy_source_path(ctx)
            if path is None or not os.path.isfile(path):
                raise ValueError(
                    "No ./policy.py found (and no `source` given). Write "
                    "`def get_option(state, memory): ...` there first.")
            with open(path, "r", encoding="utf-8") as f:
                source = f.read()
        cur = self._require_state()
        assert self._base_task is not None
        probe_task = dataclasses.replace(self._base_task,
                                         init=cur,
                                         evaluator=None)
        all_options = ctx.options
        all_predicates = ctx.predicates
        types = set(ctx.types)
        for opt in all_options:
            types.update(opt.types)
        for pred in all_predicates:
            types.update(pred.types)
        types.update(o.type for o in cur)

        def _fresh_fn() -> Any:
            fn, err = build_policy_option_fn(source,
                                             probe_task,
                                             predicates=all_predicates,
                                             options=all_options,
                                             types=types)
            if err is not None or fn is None:
                raise ValueError(err or "policy failed to load")
            return fn

        max_opts = CFG.agent_policy_max_options
        notices: List[str] = []

        def _one_rollout(fn: Any) -> Any:
            model = self._option_model()
            return execute_policy_forward(probe_task,
                                          fn,
                                          model,
                                          predicates=all_predicates,
                                          max_policy_options=max_opts)

        if trials > 1:
            fresh_scope = self._fresh_scope()
            trial_dicts: List[Dict[str, Any]] = []
            base_planner_seed = seed if seed is not None else CFG.seed
            try:
                for trial_idx in range(trials):
                    _check_time_budget(ctx)
                    _count_rollout(ctx)
                    with (fresh_scope() if fresh_scope is not None else
                          contextlib.nullcontext()), \
                            absolute_rollout_seed(seed), \
                            decorrelated_rollout_seed(trial_idx):
                        r = _one_rollout(_fresh_fn())
                    failure: Optional[str] = None
                    if r.policy_error is not None:
                        failure = f"policy error: {r.policy_error}"
                    elif r.first_failure_idx is not None and \
                            not r.goal_reached:
                        fs = r.steps[r.first_failure_idx]
                        failure = (f"step {r.first_failure_idx} "
                                   f"({_fmt_option(fs.option)}): "
                                   f"{fs.failure_reason}")
                    trial_dicts.append({
                        "goal_reached": r.goal_reached,
                        "num_actions": r.total_actions,
                        "failure": failure,
                        "planner_seed": base_planner_seed + trial_idx,
                        "inexact_start_features": [],
                    })
            except ProbeBudgetExceeded as e:
                # Same salvage rule as ``run``: completed trials live in
                # the return value, so a mid-loop budget stop returns the
                # partial result instead of discarding them.
                if not trial_dicts:
                    raise
                notices.append(
                    f"time budget expired after {len(trial_dicts)}/{trials} "
                    f"trials - the remaining trials were skipped ({e})")
            successes = sum(1 for t in trial_dicts if t["goal_reached"])
            return ProbeTrialsResult(trial_dicts, successes, fresh_scope
                                     is not None, notices)

        _count_rollout(ctx)
        step_dicts: List[Dict[str, Any]] = []

        def _on_step(_i: int, outcome: Any) -> None:
            failure = outcome.failure_reason
            if failure is not None:
                failure = (f"{failure} [surfaced to the policy as "
                           "memory['last_failure']]")
            added: List[str] = []
            deleted: List[str] = []
            if outcome.post_state is not None:
                before = utils.abstract(outcome.pre_state, ctx.predicates)
                after = utils.abstract(outcome.post_state, ctx.predicates)
                added = [str(a) for a in sorted(after - before)]
                deleted = [str(a) for a in sorted(before - after)]
            step_dicts.append({
                "option": _fmt_option(outcome.option),
                "num_actions": outcome.num_actions,
                "failure": failure,
                "added": added,
                "deleted": deleted,
                "subgoals_missing": [],
                "image": None,
            })

        with absolute_rollout_seed(seed):
            result = execute_policy_forward(probe_task,
                                            _fresh_fn(),
                                            self._option_model(),
                                            predicates=all_predicates,
                                            max_policy_options=max_opts,
                                            on_step=_on_step)
        if result.policy_error is not None:
            notices.append(f"POLICY ERROR ended the episode: "
                           f"{result.policy_error}")
        # A closed-loop probe advances the session state like `run`.
        self._state = result.final_state
        self._pristine = False
        final_atoms = [
            str(a)
            for a in sorted(utils.abstract(result.final_state, ctx.predicates))
        ]
        return ProbeResult(step_dicts, result.goal_reached, final_atoms,
                           result.final_state, notices)

    # ── Probe ranking and plan search ────────────────────────────

    def suggest_probes(self,
                       sketch_text: Optional[str] = None,
                       max_draws: Optional[int] = None,
                       top_k: int = 3) -> "ProbeSuggestResult":
        """Rank alternative continuous parameters by what they would teach.

        Rolls ``sketch_text`` forward FROM THE CURRENT STATE on your own
        parameters and, at every step annotated with ``-> {subgoals}``
        that has continuous parameters, draws up to ``max_draws``
        alternatives, keeps the ones that still establish the subgoal,
        and ranks them by the learned model's ensemble disagreement on
        those atoms (the posterior members' split on whether the
        subgoal holds). Returns the top ``top_k`` per step next to your
        proposal's own score. Pure advice: nothing you submit is
        changed by this - write an alternative into your sketch to run
        it. Needs the info-seeking ensemble (solve/explore sessions
        with ``agent_explorer_info_seeking``); otherwise the result
        says so.

        Under the joint belief (``belief_joint_draws`` K > 0) a feasible
        alternative is instead scored by the information its subgoal
        readings carry about the parameters (paper Eq. 4): it is rolled
        out, with the plan prefix, on each of the K joint draws; each
        subgoal predicate is read, at the most likely parameters, on
        ``belief_info_noise_samples`` noisy observations of each draw's
        final state; and the score is the mean over subgoals of
        ``H(mean_i r_ij) - mean_i H(r_ij)``. ``max_draws`` then defaults
        to 8 alternatives per step, since each costs K rollouts.
        """
        self._require_available("suggest_probes")
        if sketch_text is None:
            raise ValueError(
                "sim.suggest_probes(plan_text) needs the plan whose "
                "experiment you want ranked: annotate the step you would "
                "vary with its expected outcome, e.g. "
                "'Place(robot:robot, b:block)[0.6, 0.3, 0.5, 0.0] -> "
                "{OnTable(b:block)}'; its continuous parameters are the "
                "candidates.")
        # pylint: disable=import-outside-toplevel
        import numpy as np

        from predicators.agent_sdk import bilevel_sketch
        from predicators.settings import CFG

        # pylint: enable=import-outside-toplevel
        ctx = self._ctx
        _check_time_budget(ctx)
        if not CFG.continual_uncertainty_decisions:
            raise ValueError("Disagreement-based probing is disabled.")
        if (int(CFG.belief_joint_draws) > 0
                and ctx.joint_draws_provider is not None):
            return self._suggest_on_joint_draws(
                sketch_text, 8 if max_draws is None else int(max_draws), top_k)
        if max_draws is None:
            max_draws = 20
        scorer = ctx.atom_disagreement_fn
        if scorer is None:
            return ProbeSuggestResult([], [
                "no ensemble disagreement scorer is available in this "
                "session (info-seeking ensemble not installed), so "
                "alternatives cannot be ranked."
            ])
        if not ctx.info_seeking_active():
            # Adaptive info-seeking: hold probing back until a physics
            # sweep has actually found a fragile plan. Spending real steps
            # to reduce uncertainty before then is the step tax this mode
            # removes on easy levels.
            return ProbeSuggestResult([], [
                "adaptive info-seeking: no physics sweep has found a "
                "parameter-sensitive plan yet, so probing is not worth "
                "real steps. Stress-test your best plan with "
                "sim.run(plan, physics_sweep=True); if a parameter's "
                "uncertainty threatens the goal, the sweep names that "
                "parameter and this call will then rank the probes that "
                "reduce it."
            ])
        probe_task, sketch_steps, all_predicates, notices = \
            self._parse_sketch(sketch_text)
        self._refine_calls += 1
        rng = np.random.default_rng(CFG.seed + 100003 *
                                    (self._instance_id + 1) +
                                    self._refine_calls)

        def _on_rollout() -> None:
            _check_time_budget(ctx)
            _count_rollout(ctx)

        suggestions, notes = bilevel_sketch.suggest_probes(
            probe_task,
            sketch_steps,
            self._option_model(),
            predicates=all_predicates,
            info_scorer=scorer,
            rng=rng,
            max_draws=max(1, int(max_draws)),
            top_k=max(1, int(top_k)),
            on_rollout=_on_rollout)
        return ProbeSuggestResult(suggestions, list(notices) + notes)

    def _suggest_on_joint_draws(self, sketch_text: str, max_draws: int,
                                top_k: int) -> "ProbeSuggestResult":
        """``suggest_probes`` under the joint belief (see its docstring)."""
        # pylint: disable=import-outside-toplevel
        import numpy as np

        from predicators.agent_sdk import bilevel_sketch
        from predicators.code_sim_learning.active_experiment import \
            noisy_read_information
        from predicators.observation_noise import ObservationNoise
        from predicators.settings import CFG

        # pylint: enable=import-outside-toplevel
        ctx = self._ctx
        probe_task, sketch_steps, all_predicates, notices = \
            self._parse_sketch(sketch_text)
        on_observation = self._on_current_observation(probe_task.init)
        noise = ObservationNoise.from_cfg()
        num = int(CFG.belief_joint_draws)
        samples = max(1, int(CFG.belief_info_noise_samples))

        def plan_scorer(plan: List[Any],
                        atoms: Any) -> Tuple[float, Dict[str, float]]:
            result = self._run_joint_draws(probe_task,
                                           plan,
                                           sketch_steps[:len(plan)],
                                           all_predicates,
                                           num,
                                           None, [],
                                           from_observation=on_observation,
                                           return_final=True,
                                           score=False)
            atom_list = sorted(atoms, key=str)
            rows = []
            for i, draw in enumerate(result.draws):
                final = draw["final"]
                # Common random numbers: draw i reads the same noise for
                # every candidate.
                rng = np.random.default_rng([int(CFG.seed), 7919, i])
                reads = np.zeros(len(atom_list))
                for _ in range(samples):
                    view = noise.perturb(final, rng) if noise.enabled \
                        else final.copy()
                    # The predicates read the observation only.
                    view.latent = {}
                    reads += [float(a.holds(view)) for a in atom_list]
                rows.append(reads / samples)
            matrix = np.asarray(rows, dtype=float)
            per_atom = {
                str(a): noisy_read_information(matrix[:, [j]])
                for j, a in enumerate(atom_list)
            }
            return noisy_read_information(matrix), per_atom

        self._refine_calls += 1
        rng = np.random.default_rng(CFG.seed + 100003 *
                                    (self._instance_id + 1) +
                                    self._refine_calls)
        suggestions, notes = bilevel_sketch.suggest_probes(
            probe_task,
            sketch_steps,
            self._option_model(),
            predicates=all_predicates,
            info_scorer=None,
            rng=rng,
            max_draws=max(1, max_draws),
            top_k=max(1, int(top_k)),
            on_rollout=lambda: _check_time_budget(ctx),
            plan_scorer=plan_scorer)
        notices.append(
            f"scores are the information (bits) one reading of each step's "
            f"subgoals carries about the parameters, over the {num} joint "
            f"draws with {samples} noisy observations each; 0 means every "
            "draw reads the subgoals the same way.")
        return ProbeSuggestResult(suggestions,
                                  list(notices) + notes,
                                  measure="information")

    def refine(self,
               sketch_text: str,
               timeout: float = 60.0,
               max_samples_per_step: Optional[int] = None,
               require_goal: bool = False,
               require_solved: bool = False) -> "ProbeRefineResult":
        """Backtracking parameter search for a sketch FROM THE CURRENT STATE.

        Same grammar as ``skills_execute_plan``, and composable:
        refine a plan *suffix* from a snapshot where the
        prefix already executed, so the search budget goes to the step
        that matters instead of re-descending through the whole plan.
        Annotate each step's ``-> {subgoals}`` - success means every
        step established its annotation (set ``require_goal=True`` to
        also demand the task goal at the last step; the result's
        ``Verdict`` line states which of these a SUCCESS certifies).
        Each step's ``[params]`` seed the search (tried first, then
        sampled around); ``~ [w]`` half-width regions confine the
        sampling ONLY when ground samplers are enabled in the session
        config - otherwise the annotation is accepted but ignored (a
        NOTE says so) and sampling stays uniform. Does not advance the
        current state. Returns best-found params (also on TIMEOUT - the
        refined prefix is reported as far as it got) plus per-step
        sample counts and the deepest near-miss. Exploratory only:
        nothing runs in the environment.

        ``require_solved=True`` (implies ``require_goal``) additionally
        gates final-step acceptance on the TASK EVALUATOR's public
        ``solved`` verdict, so the search rejects parameters that reach
        the goal atoms via a route the evaluator scores as a non-solve
        and keeps searching (the near-miss records such rejections).
        Only valid when the current state is the task's UNMODIFIED
        initial state (plain ``reset()``, no rollout since): the
        evaluator's rules reference the true init, so any other start
        would give silently-wrong verdicts.
        """
        self._require_available("refine")
        # pylint: disable=import-outside-toplevel
        import numpy as np

        from predicators.agent_sdk import bilevel_sketch
        from predicators.settings import CFG

        # pylint: enable=import-outside-toplevel
        ctx = self._ctx
        _check_time_budget(ctx)
        probe_task, sketch_steps, all_predicates, notices = \
            self._parse_sketch(sketch_text)
        joint = (int(CFG.belief_joint_draws) > 0
                 and ctx.joint_draws_provider is not None
                 and CFG.continual_uncertainty_decisions)
        on_observation = self._on_current_observation(probe_task.init)
        if joint and on_observation:
            # Under the joint belief the search starts from the belief
            # mean (paper Appendix B.4).
            mean_frame = getattr(getattr(ctx, "current_belief", None), "frame",
                                 None)
            if mean_frame is not None:
                probe_task = dataclasses.replace(probe_task,
                                                 init=mean_frame.copy())
        solved_check: Optional[Callable[[List[State], List[Any], bool],
                                        Tuple[bool, str]]] = None
        gate_ran = [False]
        gate_called = [False]
        if require_solved:
            require_goal = True
            evaluator = self._require_solved_evaluator("require_solved")
            # An unavailable certificate must not become a successful
            # refinement, even when the goal predicates hold.
            inner_check = make_solved_check(
                evaluator, getattr(self._option_model(), "sim_env", None))

            def gated_solved_check(states: List[State], labels: List[Any],
                                   coarse: bool) -> Tuple[bool, str]:
                # Track whether the gate was consulted at all vs. with a
                # real (non-coarse) rollout: "never consulted" means no
                # candidate reached the goal atoms (the blocker is
                # upstream of scoring), while "only coarse" means the
                # option model exposed no per-step trajectories - the
                # two need different explanations (a shared message here
                # read as a system failure in run_20260717 audits).
                gate_called[0] = True
                if not coarse:
                    gate_ran[0] = True
                return inner_check(states, labels, coarse)

            solved_check = gated_solved_check

        if max_samples_per_step is None:
            max_samples_per_step = \
                RefinementConfig.from_cfg().max_samples_per_step
        self._refine_calls += 1
        # Deterministic but distinct from the solver's CFG.seed + attempt
        # streams and from other probe instances, so
        # "try a different random search" does not replay failed draws.
        rng = np.random.default_rng(CFG.seed + 100003 *
                                    (self._instance_id + 1) +
                                    self._refine_calls)
        outcome = bilevel_sketch.refine_sketch(
            probe_task,
            sketch_steps,
            self._option_model(),
            predicates=all_predicates,
            timeout=timeout,
            rng=rng,
            max_samples_per_step=max_samples_per_step,
            check_subgoals=True,
            check_final_goal=require_goal,
            run_id="probe",
            strip_latent_wait_targets=not ctx.latent_tracking_available,
            solved_check=solved_check)
        refined_plan, success = outcome.plan, outcome.success
        total_samples = outcome.total_samples
        step_samples = outcome.step_samples_cumulative
        reason = outcome.termination_reason or ("success"
                                                if success else "failure")
        joint_note = ""
        if joint and success:
            refined_plan, joint_note, extra = self._select_on_joint_draws(
                probe_task, sketch_steps, all_predicates, refined_plan,
                timeout, rng, max_samples_per_step, require_goal, solved_check,
                on_observation)
            total_samples += extra
        plan_lines: List[str] = []
        for i, st in enumerate(sketch_steps):
            opt = refined_plan[i] if i < len(refined_plan) else None
            objs = ", ".join(f"{o.name}:{o.type.name}" for o in st.objects)
            params = _fmt_params(opt.params) if opt is not None else "[?]"
            atoms = sorted(str(a) for a in (st.subgoal_atoms or set()))
            atoms += sorted(f"NOT {a}"
                            for a in (st.subgoal_neg_atoms or set()))
            suffix = f" -> {{{', '.join(atoms)}}}" if atoms else ""
            plan_lines.append(f"{st.option.name}({objs}){params}{suffix}")
        near_miss: Optional[Dict[str, Any]] = None
        if outcome.deepest_failure is not None:
            df = outcome.deepest_failure
            near_miss = {
                "step_idx": df.step_idx,
                "option": _fmt_option(df.option),
                "reason": df.fail_reason,
            }
        note_parts = list(notices)
        if joint_note:
            note_parts.append(joint_note)
        if require_solved and not gate_ran[0]:
            if not gate_called[0]:
                note_parts.append(
                    "require_solved: no candidate reached the goal atoms at "
                    "the final step, so the task evaluator was never "
                    "consulted - the blocker is upstream of scoring (see "
                    "the near-miss).")
            else:
                note_parts.append(
                    "require_solved: the option model exposed no per-step "
                    "trajectories, so the evaluator gate only saw coarse "
                    "rollouts and this result is NOT certified.")
        # What a SUCCESS actually certifies, so it is never read as more
        # than it means (a bare "SUCCESS" was disproven with 5 fresh
        # rollouts in run_20260717_154753 seed2 - it only meant
        # "executed").
        if not success:
            verdict = ""
        elif require_solved and gate_ran[0]:
            verdict = ("evaluator-solved - the task evaluator scored a "
                       "full rollout of these params as a solve.")
        elif require_goal:
            verdict = ("goal-reached - the task's goal atoms held at the "
                       "final step")
            # Recommend the evaluator gate only when the task has a rule to
            # judge it by: sessions without one wasted a call per session
            # following this advice into a refusal (run_20260830, three
            # sessions).
            if task_evaluator(self._base_task) is not None:
                verdict += ("; no evaluator verdict (use "
                            "require_solved=True from a pristine reset() "
                            "for that).")
            else:
                verdict += (" (this task has no goal atoms an evaluator "
                            "could judge, so goal-reached is the strongest "
                            "verdict available).")
        else:
            verdict = ("executed - every step established its subgoal "
                       "annotation; the task goal was NOT checked (set "
                       "require_goal=True to demand it).")
        return ProbeRefineResult(success, reason, total_samples, step_samples,
                                 plan_lines, near_miss, " ".join(note_parts),
                                 verdict)

    def _select_on_joint_draws(
            self, probe_task: Task, sketch_steps: Any, all_predicates: Any,
            first_plan: List[Any], timeout: float, rng: Any,
            max_samples_per_step: int, require_goal: bool, solved_check: Any,
            on_observation: bool) -> Tuple[List[Any], str, int]:
        """Collect up to ``belief_refine_candidates`` proposals, score them on
        the common joint draws, and re-estimate the best on fresh draws.

        The first proposal is the search's own result; the others come
        from further searches without the sketch's proposed parameters,
        sharing the caller's timeout. Returns the chosen plan, a note
        for the report and the samples the extra searches spent.
        """
        # pylint: disable=import-outside-toplevel
        import numpy as np

        from predicators.agent_sdk import bilevel_sketch
        from predicators.settings import CFG

        # pylint: enable=import-outside-toplevel
        ctx = self._ctx
        wanted = max(int(CFG.belief_refine_candidates), 1)
        proposals = [list(first_plan)]
        extra_samples = 0
        open_steps = [
            dataclasses.replace(st, initial_params=None, ground_sampler=None)
            for st in sketch_steps
        ]
        deadline = time.monotonic() + timeout
        while len(proposals) < wanted and time.monotonic() < deadline:
            _check_time_budget(ctx)
            left = deadline - time.monotonic()
            outcome = bilevel_sketch.refine_sketch(
                probe_task,
                open_steps,
                self._option_model(),
                predicates=all_predicates,
                timeout=max(1.0, left / max(1, wanted - len(proposals))),
                rng=rng,
                max_samples_per_step=max_samples_per_step,
                check_subgoals=True,
                check_final_goal=require_goal,
                run_id="probe",
                strip_latent_wait_targets=not ctx.latent_tracking_available,
                solved_check=solved_check)
            extra_samples += outcome.total_samples
            if not outcome.success:
                continue
            plan = list(outcome.plan)
            duplicate = any(
                all(
                    np.allclose(a.params, b.params)
                    for a, b in zip(plan, other)) for other in proposals)
            if not duplicate:
                proposals.append(plan)
        num = int(CFG.belief_joint_draws)
        scores = []
        for plan in proposals:
            result = self._run_joint_draws(probe_task,
                                           plan,
                                           sketch_steps,
                                           all_predicates,
                                           num,
                                           None, [],
                                           from_observation=on_observation)
            scores.append(result.p_hat)
        best = int(np.argmax(scores))
        fresh = self._run_joint_draws(probe_task,
                                      proposals[best],
                                      sketch_steps,
                                      all_predicates,
                                      num,
                                      None, [],
                                      fresh=True,
                                      from_observation=on_observation)
        listed = ", ".join(f"{p:.2f}" for p in scores)
        note = (f"Scored {len(proposals)} proposal(s) on the {num} common "
                f"joint draws (P-hat {listed}) and kept proposal {best + 1}; "
                f"its P-hat on {num} fresh draws is {fresh.p_hat:.2f} +- "
                f"{fresh.stderr:.2f}, an estimate its selection does not "
                "bias.")
        return proposals[best], note, extra_samples

    # ── Internals ────────────────────────────────────────────────

    def __getattribute__(self, name: str) -> Any:
        # A method the session's arm leaves out (ToolContext.probe_absent)
        # does not exist on the probe.
        if not name.startswith("_"):
            ctx = object.__getattribute__(self, "__dict__").get("_ctx")
            if name in getattr(ctx, "probe_absent", ()):
                raise AttributeError(f"{type(self).__name__!r} object has no "
                                     f"attribute {name!r}")
        return object.__getattribute__(self, name)

    def __dir__(self) -> List[str]:
        absent = getattr(self._ctx, "probe_absent", ())
        return [name for name in super().__dir__() if name not in absent]


class BeliefProbe(_ProbeCore):
    """The probe with the harness fit's entry points: ``fit``, and the
    ``fit_params`` and ``sweep_params`` modes of ``residuals``."""

    def fit(self,
            traj_idxs: Optional[List[int]] = None,
            fixed: Optional[Dict[str, float]] = None,
            path: Optional[str] = None) -> str:
        """Fit the candidate simulator's parameters and return the report.

        Synthesis sessions only (at solve time the deployed belief
        model is fixed). With no arguments this is the CANONICAL fit -
        the same fit the probe deploys for ``run``/``refine``, on the
        full data - and, when ``simulator.py`` declares
        ``AGENT_PARAM_SPECS`` on its subclass, the identified values are applied
        to the planning base env. Passing ``traj_idxs`` (fit only those
        trajectories' data) or ``fixed`` (pin parameters at given
        values) makes the fit EXPLORATORY: a diagnostic report only -
        nothing is published or applied, and the probe keeps running
        the canonical fit. On the system-ID path ``traj_idxs`` is a
        cross-trajectory consistency check (subset fits that disagree
        mean heterogeneous data); ``fixed`` is rejected there - pin by
        narrowing the param's declared bounds. Reports SSE at
        init vs post-fit, fitted values with deltas, and (system-ID
        path) per-parameter identifiability. The fit is the expensive
        probe call; use deliberately.
        """
        ctx = self._ctx
        _check_time_budget(ctx)
        provider = ctx.probe_fit_provider
        if provider is None:
            raise RuntimeError(
                "sim.fit is unavailable in this session: fitting happens "
                "during learning; the solve-time belief model is fixed.")
        return provider(path=path, traj_idxs=traj_idxs, fixed=fixed)

    def residuals(self,
                  max_transitions: int = 100,
                  abs_tol: float = 1e-4,
                  rel_tol: float = 1e-3,
                  num_worst_examples: int = 3,
                  fit_params: bool = False,
                  path: Optional[str] = None,
                  rollout: bool = False,
                  sweep_num_points: int = 6,
                  sweep_params: Optional[Union[str, List[str]]] = None,
                  phys_params: Optional[Dict[str, float]] = None) -> str:
        """Per-feature residual report for the current simulator rules.

        Synthesis sessions only. Loads the current simulator fresh from
        ``simulator.py`` and reports, per feature in RESIDUAL_FEATURES,
        mismatch counts, mean/max abs error, improvement over the
        no-rule baseline (negative = the rules hurt), and the worst-N
        example transitions - the fast inner loop for finding where the
        rules disagree with the data. Uses init_value params unless
        ``fit_params=True`` (LM-fits first; diagnostic only, nothing
        published). Tolerance: ``|pred - obs| > rel_tol * |obs| +
        abs_tol``. Snapshots the simulator file and tags the report
        ``[cycle_XXX_vers_YYY]``. ``path`` scores a DIFFERENT simulator
        file than the canonical one - use it to compare rule-set
        candidates side by side.

        ``rollout=True`` switches to the OPEN-LOOP report: free-running
        replay of each recorded trajectory (errors compound, which the
        teacher-forced default structurally cannot see). It is the only
        residual view that can implicate or exonerate a physical
        parameter; consult it before deciding the parameter
        declaration either way. Two mutually exclusive opt-ins probe
        the parameters themselves:

        * ``sweep_params`` - a list of env-registry parameter names, or
          ``"all"`` - sweeps each named parameter alone across its
          plausible range (``sweep_num_points`` candidates each) and
          reports whether the data is explained materially better at
          another value. Slow: one fresh-env rollout per candidate per
          motion segment, so name the suspects (or pass ``"all"`` once
          before the declaration decision) rather than sweeping in a
          loop.
        * ``phys_params`` - ``{name: value}`` - scores the data at ONE
          hypothesized point and reports the SSE ratio vs the baseline;
          the cheap composable primitive for your own targeted sweeps.
        """
        return self._residuals(max_transitions=max_transitions,
                               abs_tol=abs_tol,
                               rel_tol=rel_tol,
                               num_worst_examples=num_worst_examples,
                               fit_params=fit_params,
                               path=path,
                               rollout=rollout,
                               sweep_num_points=sweep_num_points,
                               sweep_params=sweep_params,
                               phys_params=phys_params)


# The probe under agent_sim_learn_fit_invisible: the core's calls, with a
# ``residuals`` that takes the core's arguments alone. It has no ``fit``,
# and an agent can read its docstrings, so they name none.
class DeclaredValuesProbe(_ProbeCore):
    """Stateful exploration handle over your simulator, which runs at the
    values its declarations give."""

    def residuals(self,
                  max_transitions: int = 100,
                  abs_tol: float = 1e-4,
                  rel_tol: float = 1e-3,
                  num_worst_examples: int = 3,
                  path: Optional[str] = None,
                  rollout: bool = False,
                  phys_params: Optional[Dict[str, float]] = None) -> str:
        """Residual report for the current simulator at its declared values.

        Loads the current simulator fresh from ``simulator.py`` and
        reports where it disagrees with the recorded data. Snapshots the
        simulator file and tags the report ``[cycle_XXX_vers_YYY]``.
        ``path`` scores a DIFFERENT simulator file than the canonical
        one - use it to compare candidates side by side.

        A model that is a ``RESIDUAL_ENV`` subclass gets the OPEN-LOOP
        report: free-running replay of each recorded trajectory, where
        errors compound. ``rollout=True`` asks for it by name.
        ``phys_params`` - ``{name: value}`` - scores the data at ONE
        hypothesized point and reports the SSE ratio vs the declared
        values; the cheap composable primitive for your own targeted
        sweeps. For a rule-form model the default report is per
        transition: per feature in RESIDUAL_FEATURES, mismatch counts,
        mean/max abs error and the worst-N example transitions, with the
        tolerance ``|pred - obs| > rel_tol * |obs| + abs_tol``.
        """
        return self._residuals(max_transitions=max_transitions,
                               abs_tol=abs_tol,
                               rel_tol=rel_tol,
                               num_worst_examples=num_worst_examples,
                               fit_params=False,
                               path=path,
                               rollout=rollout,
                               sweep_num_points=6,
                               sweep_params=None,
                               phys_params=phys_params)


# The agent knows its probe as a BeliefProbe, whichever class serves it, so
# the class's own errors (an unknown attribute or keyword) name that.
DeclaredValuesProbe.__name__ = DeclaredValuesProbe.__qualname__ = "BeliefProbe"
DeclaredValuesProbe.residuals.__qualname__ = "BeliefProbe.residuals"


def probe_class_name() -> str:
    """The name the session's agent knows its probe class by: it is bound in
    the ``run_python`` namespace and makes extra independent instances."""
    # pylint: disable-next=import-outside-toplevel
    from predicators.settings import CFG
    if CFG.continual_point_values_only:
        # pylint: disable-next=import-outside-toplevel
        from predicators.agent_sdk.point_probe import PROBE_NAME
        return PROBE_NAME
    return "BeliefProbe"


def build_probe_namespace(ctx: "ToolContext") -> Dict[str, Any]:
    """The persistent ``run_python`` namespace (solve sessions).

    The probe facade, numpy, and the collected REAL trajectories as
    read-only evidence (``trajectories`` plus a ``describe_trajectory``
    digest helper) - so the agent can check the belief model against
    what actually happened. Deliberately NOT here: anything
    evaluator-shaped (``evaluate_trajectory`` scores arbitrary state
    sequences, which in a solve session is a free oracle for searching
    over sequences the evaluator likes without paying for physics;
    evaluator access stays behind the gated ``run(solved=True)`` /
    ``refine(require_solved=True)`` paths) and authoring bindings
    (``Predicate``, ``ParamSpec`` - synthesis-role, see
    ``_build_synthesis_exec_ns``).
    """
    # pylint: disable-next=import-outside-toplevel
    import numpy as np

    # pylint: disable-next=import-outside-toplevel
    from predicators.agent_sdk.tools.digests import render_trajectory_digest
    # pylint: disable-next=import-outside-toplevel
    from predicators.settings import CFG
    all_trajs = list(ctx.offline_trajectories) + list(ctx.online_trajectories)
    probe_cls: Callable[["ToolContext"],
                        Any] = (DeclaredValuesProbe
                                if CFG.agent_sim_learn_fit_invisible else
                                BeliefProbe)
    if CFG.continual_point_values_only:
        # The arm at one value per parameter holds a probe without the
        # calls built on a belief.
        # pylint: disable-next=import-outside-toplevel
        from predicators.agent_sdk.point_probe import SimProbe
        probe_cls = SimProbe

    def describe_trajectory(traj_idx: int,
                            include_states: bool = True,
                            include_atoms: bool = False,
                            max_timesteps: int = 10) -> str:
        return render_trajectory_digest(all_trajs,
                                        ctx.train_tasks,
                                        ctx.predicates,
                                        traj_idx,
                                        include_states=include_states,
                                        include_atoms=include_atoms,
                                        max_timesteps=max_timesteps)

    return {
        "sim": probe_cls(ctx),
        probe_class_name(): lambda: probe_cls(ctx),
        "np": np,
        "trajectories": all_trajs,
        "describe_trajectory": describe_trajectory,
    }
