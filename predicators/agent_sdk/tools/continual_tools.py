"""The env, skill, learning and session tools of a continual-protocol play
session (docs/protocol/design.md, section 5.1).

The tools translate agent commands into typed requests for the run's
``InteractionExecutor`` and format its results. The driver owns physical
execution, including the inner skill-plan and primitive-policy loops.
The protocol counts, records and enforces caps. Every tool result
carries the ledger. Give-up is deferred until the conversation has saved
its notes.
"""
from __future__ import annotations

import dataclasses
import json
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, AsyncIterator, Callable, Dict, \
    Iterable, List, NamedTuple, Optional, Sequence, Set, Tuple

import numpy as np

from predicators import utils
from predicators.agent_sdk.predictive_check import check_prediction, \
    point_predicts
from predicators.agent_sdk.preflight_audit import audit_safely
from predicators.agent_sdk.primitive_policy import open_primitive_policy, \
    primitive_action, primitive_observation
from predicators.agent_sdk.sketch_parsing import parse_sketch_from_text
from predicators.agent_sdk.tools.context import LiveRehearsal, ToolContext
from predicators.agent_sdk.tools.digests import render_options_digest
from predicators.agent_sdk.tools.python_exec import _resolve_sandbox_file
from predicators.agent_sdk.tools.results import _error_result, _text_result
from predicators.agent_sdk.tools.reward_model import PREDICTION_STARTS, \
    level_task, load_prediction_file, score_prediction
from predicators.observation_belief import atom_fractions, \
    describe_fractions, uncertain_atoms
from predicators.observation_noise import ObservationNoise
from predicators.run.episode import EpisodeOver, EpisodeState
from predicators.run.interaction import ActionSelector, ExecutePlan, \
    ExecutePolicy, ExecuteSkill, ExecutionObserver, ExecutionProgress, \
    ExecutionRequest, GiveUp, PrimitiveAction, RequestReset, SkillExecution
from predicators.run.recording import states_close
from predicators.settings import CFG
from predicators.structs import Action, GroundAtom, Predicate, State, Task, \
    _Option, excluded_object_type_names, step_option_labels

if TYPE_CHECKING:  # pragma: no cover - import cycle through approaches
    from predicators.run.continual import InvocationResult, \
        ProtocolObservation, ProtocolSession

CONTINUAL_TOOL_NAMES = [
    "env_observe",
    "env_step",
    "env_reset",
    "give_up",
    "skills_list",
    "skills_invoke",
    "skills_execute_plan",
]
PRIMITIVE_TOOL_NAMES = [
    "env_observe", "env_step", "env_reset", "give_up", "env_run_policy"
]
# The tools of continual_claim_check.
CLAIM_TOOL_NAMES = ["claims_record", "claims_resolve"]
# The tool of continual_provide_reward_model on an arm with no run_python.
REWARD_MODEL_TOOL_NAMES = ["evaluate_trajectory"]
ALL_CONTINUAL_TOOL_NAMES = list(
    dict.fromkeys(CONTINUAL_TOOL_NAMES + PRIMITIVE_TOOL_NAMES +
                  CLAIM_TOOL_NAMES + REWARD_MODEL_TOOL_NAMES))
# The tools that drive the robot below the skill library.
RAW_CONTROL_TOOL_NAMES = ("env_step", "env_run_policy")


def play_tool_names(names: Sequence[str]) -> List[str]:
    """A skill agent's tool list under ``CFG.continual_raw_control``: the given
    names, minus the raw-control tools when the flag is off."""
    if CFG.continual_raw_control:
        return list(names)
    return [n for n in names if n not in RAW_CONTROL_TOOL_NAMES]


# How a refusal opens on a reset training level.
PRACTISE_HEAD = "This training level has been reset without a win:"

GRAMMAR = (
    "One skill per line: `Skill(obj:type, ...)[p1, p2] -> {Atom(obj:type), "
    "NOT Other(obj:type)}`. Exact params in `[]` (`[]` when the skill "
    "has none); the `-> {...}` expected outcome is optional.")


@dataclass
class PlayState:
    """What the tools record for the arm to act on after the query."""
    pending_give_up: Optional[GiveUp] = None
    run_ended: Optional[Tuple[str, str]] = None
    charged_calls: int = 0


# ── Formatting ─────────────────────────────────────────────────────


def visible_atoms(ctx: ToolContext, frame: State) -> Set[GroundAtom]:
    """The atoms of ``frame`` under the arm's predicate vocabulary.

    Under the protocol an arm starts with no env predicate (the
    allowlist ``agent_sim_learn_kept_predicates_names`` can hand it
    some) and must not see the env's full atom set through the
    observation; the protocol's own view of the env atoms stays in the
    recording for analysis.
    """
    return utils.abstract(frame, set(ctx.predicates))


def _identities(predicates: Iterable[Predicate]) -> Set[int]:
    """The predicate objects themselves, for membership by identity.

    ``Predicate`` compares by name and argument types, so a predicate
    the arm invents under an env predicate's name would otherwise pass
    as the env's: the goal atoms would surface the moment the arm
    guessed the name (domino m3, 2026-09-05, invented ``Toppled`` and
    was shown ``Toppled(domino_1)`` from then on) and its own classifier
    would be listed as the environment's. An env predicate the allowlist
    keeps is the env's own object, so identity keeps it env-origin.
    """
    return {id(p) for p in predicates}


def _split_atoms(
    ctx: ToolContext,
    atoms: Set[GroundAtom],
    env_predicates: Optional[Iterable[Predicate]] = None
) -> Tuple[List[str], List[str]]:
    """(env-origin atoms, invented atoms) as sorted strings.

    ``env_predicates`` are the env's own predicate objects (the session
    knows them); without them the simulator env on the context is
    asked, and an arm with neither sees every atom as its own. An atom
    is the env's only when its predicate IS one of them (see
    :func:`_identities`).
    """
    if env_predicates is None:
        env = ctx.env
        env_predicates = env.predicates if env is not None else set()
    env_ids = _identities(env_predicates)
    env_origin = sorted(str(a) for a in atoms if id(a.predicate) in env_ids)
    invented = sorted(str(a) for a in atoms if id(a.predicate) not in env_ids)
    return env_origin, invented


def visible_goal(ctx: ToolContext, task: Task) -> List[str]:
    """The goal atoms the arm's vocabulary can express, as strings.

    Expressible means the goal atom's predicate object is in the arm's
    vocabulary (an env predicate the allowlist kept), never a predicate
    of the arm's own that shares the name: the goal stays its
    description until then.
    """
    own = _identities(ctx.predicates)
    return sorted(str(a) for a in task.goal if id(a.predicate) in own)


def context_status(ctx: ToolContext) -> str:
    """The ``[context]`` line: what the run's conversation holds.

    Shown next to the ledger on every tool result and in every query, so
    the agent can journal ahead of a compaction instead of after one.
    """
    if ctx.context_tokens is None:
        size = "size not reported yet"
    else:
        size = f"~{ctx.context_tokens / 1000:.0f}k tokens"
        if ctx.context_window_tokens:
            size += f" of {ctx.context_window_tokens / 1000:.0f}k"
    return (f"[context] {size}; {ctx.context_turns} turns this run; "
            f"compacted {ctx.context_compactions}x")


def format_observation(obs: "ProtocolObservation",
                       ctx: ToolContext,
                       *,
                       with_state: bool,
                       render_path: Optional[str],
                       env_predicates: Optional[Iterable[Predicate]] = None,
                       with_goal: bool = True,
                       with_budget: bool = True) -> str:
    """The observation as text (section 5.2); ``env_predicates`` as in
    :func:`_split_atoms`. Queries can omit goal and budget fields already
    supplied by their template; tool results include them by default.
    """
    lines = []
    if obs.state is EpisodeState.GAME_OVER and not obs.ledger.resets_allowed:
        lines.append(f"[episode] GAME_OVER ({obs.reason}); this level has "
                     "no resets, so it is over and lost. Write your notes "
                     "and stop.")
    elif obs.state is EpisodeState.GAME_OVER:
        lines.append(f"[episode] GAME_OVER ({obs.reason}); only env_reset "
                     "is valid now.")
    elif obs.state is EpisodeState.WIN:
        lines.append("[episode] WIN: the level is won. Write your notes "
                     "and stop.")
    else:
        lines.append("[episode] NOT_FINISHED")
    spec = obs.level
    goal = visible_goal(ctx, spec.task)
    goal_text = ", ".join(goal) if goal else (
        "(not expressible in your predicates; the goal description is "
        "the goal)")
    resets_note = "" if obs.ledger.resets_allowed else ", no resets"
    lines.append(f"[level] {spec.index + 1}/{obs.ledger.levels_total} "
                 f"({spec.split} task {spec.task_idx}{resets_note})")
    if with_goal:
        lines[-1] += f"; goal atoms: {goal_text}"
    if with_goal and spec.task.goal_nl:
        lines.append(f"[goal] {spec.task.goal_nl}")
    noise = ObservationNoise.from_cfg()
    if noise.enabled and noise.declared:
        lines.append(f"[noise] {noise.summary()}")
    if obs.evaluation is not None:
        lines.append(f"[evaluation] reward {obs.evaluation.reward:.3f}, "
                     f"terminated {obs.evaluation.terminated}")
    try:
        env_origin, invented = _split_atoms(ctx, visible_atoms(ctx, obs.frame),
                                            env_predicates)
        note = ""
    except Exception as e:  # pylint: disable=broad-except
        env_origin, invented = [], []
        note = f"(atoms could not be evaluated: {e})"
    lines.append("[atoms] " + (", ".join(env_origin) or note or "(none)"))
    if invented:
        lines.append("[your predicates] " + ", ".join(invented))
    if obs.belief is not None and CFG.continual_uncertainty_decisions:
        try:
            if ctx.joint_fractions_provider is not None:
                fractions = ctx.joint_fractions_provider()
            else:
                fractions = atom_fractions(
                    obs.belief, set(ctx.predicates),
                    int(CFG.continual_belief_draws),
                    np.random.default_rng(CFG.seed +
                                          7919 * int(obs.ledger.run_steps)))
            unsure = uncertain_atoms(fractions)
        except Exception as e:  # pylint: disable=broad-except
            fractions, unsure = {}, []
            lines.append(f"[atoms under the belief] (could not be "
                         f"evaluated: {e})")
        if unsure:
            lines.append(
                "[atoms under the belief] unsure: " +
                describe_fractions(fractions, unsure) +
                " (the fraction of belief draws on which the atom holds; "
                "atoms at 0 or 1 read as listed above)")
    if with_state:
        lines.append("[objects]")
        lines.append(obs.frame.dict_str(indent=2, num_decimal_points=4))
        if (obs.belief is not None and obs.belief.frames_used
                and CFG.continual_uncertainty_decisions):
            lines.append("[belief] each object smoothed over the frames it "
                         "rested through (value+-spread):")
            lines.extend(belief_object_lines(obs.frame, obs.belief))
    if render_path:
        lines.append(f"[render] {render_path}")
    if with_budget:
        lines.append(obs.ledger.footer())
        lines.append(context_status(ctx))
    return "\n".join(lines)


def belief_object_lines(frame: State, belief: Any) -> List[str]:
    """The ``[belief]`` block's line for each object the belief smoothed,
    leaving out the types every state listing hides
    (``excluded_object_type_names``)."""
    hidden = excluded_object_type_names()
    return [
        "  " + belief.object_line(obj)
        for obj in sorted(frame, key=lambda o: o.name)
        if obj.name in belief.frames_used and obj.type.name not in hidden
    ]


def _atoms_change_line(before: Set[GroundAtom], after: Set[GroundAtom]) -> str:
    added = sorted(str(a) for a in after - before)
    removed = sorted(str(a) for a in before - after)
    if not added and not removed:
        return "  atoms: no change"
    parts = []
    if added:
        parts.append("+" + ", +".join(added))
    if removed:
        parts.append("-" + ", -".join(removed))
    return "  atoms changed: " + "; ".join(parts)


# ── Plan parsing ───────────────────────────────────────────────────


def parse_plan_lines(
        text: str, ctx: ToolContext,
        task: Task) -> List[Tuple[_Option, Set[GroundAtom], Set[GroundAtom]]]:
    """Parse plan text into grounded options with expected outcomes.

    Raises ``ValueError`` naming the first line that does not parse or a
    parametrised skill given no exact parameters.
    """
    steps = parse_sketch_from_text(text,
                                   task,
                                   predicates=set(ctx.predicates),
                                   options=set(ctx.options),
                                   types=set(ctx.types),
                                   parse_continuous_params=True,
                                   strict=True,
                                   parse_ground_samplers=False)
    if not steps:
        raise ValueError("no skill line parsed. " + GRAMMAR)
    out = []
    for step in steps:
        dim = int(np.prod(step.option.params_space.shape))
        if step.initial_params is None:
            if dim > 0:
                raise ValueError(f"{step.option.name} takes {dim} "
                                 "parameter(s); give exact values in `[]`")
            params = np.zeros(0, dtype=np.float32)
        else:
            params = np.asarray(step.initial_params, dtype=np.float32)
            if params.shape[0] != dim:
                raise ValueError(f"{step.option.name} takes {dim} "
                                 f"parameter(s), got {params.shape[0]}")
        option = step.option.ground(list(step.objects), params)
        expected = set(step.subgoal_atoms or set())
        absent = set(step.subgoal_neg_atoms or set())
        if option.name == "Wait":
            if expected:
                option.memory["wait_target_atoms"] = set(expected)
            if absent:
                option.memory["wait_target_neg_atoms"] = set(absent)
        out.append((option, expected, absent))
    return out


# Two plan lines name the same request when their parameters agree to
# this tolerance (a value printed to four decimals and parsed back).
_SAME_PARAM_TOL = 1e-4


def _same_line(option: _Option, wanted: _Option) -> bool:
    """Whether two parsed plan lines name the same request: the same skill on
    the same objects with the same parameters."""
    return (option.name == wanted.name
            and [o.name
                 for o in option.objects] == [o.name for o in wanted.objects]
            and option.params.shape == wanted.params.shape and np.allclose(
                option.params, wanted.params, rtol=0.0, atol=_SAME_PARAM_TOL))


def _shared_lines(
        plan: List[Tuple[_Option, Set[GroundAtom], Set[GroundAtom]]],
        request: List[Tuple[_Option, Set[GroundAtom],
                            Set[GroundAtom]]]) -> int:
    """How many leading lines of the parsed ``request`` the parsed ``plan``
    begins with, in order."""
    count = 0
    for (option, _, _), (wanted, _, _) in zip(plan, request):
        if not _same_line(option, wanted):
            break
        count += 1
    return count


def _motion_lines(
        parsed: List[Tuple[_Option, Set[GroundAtom], Set[GroundAtom]]]) -> int:
    """How many leading lines of a parsed request a rehearsal must repeat:
    every line up to its last line other than Wait.

    Wait commands no motion, so the request runs as those lines and then
    a request of Wait lines alone, which passes the rehearsal gate. A
    request of Wait lines alone needs no rehearsal (0).
    """
    for index in range(len(parsed), 0, -1):
        if parsed[index - 1][0].name != "Wait":
            return index
    return 0


def _wait_steps(
    lines: List[Tuple[_Option, Set[GroundAtom], Set[GroundAtom]]]
) -> Optional[int]:
    """How many steps the Wait lines in ``lines`` ask for in all; None when one
    of them names no step count."""
    total = 0
    for option, _, _ in lines:
        if option.name != "Wait":
            continue
        if not option.params.size:
            return None
        total += int(round(float(option.params[0])))
    return total


class _RehearsalMatch(NamedTuple):
    """A rehearsal that let a request through, and how many of the request's
    leading lines its plan repeated, from its line ``start`` on (0 for a
    request that begins the plan; more for one that continues it)."""
    rehearsal: LiveRehearsal
    lines: int
    start: int = 0


class _PlanCursor:
    """Where the last scored request left its rehearsal's plan: the match, and
    the run's position (level, steps, resets) when the request ended; both None
    after any other request or action."""

    def __init__(self) -> None:
        self.match: Optional[_RehearsalMatch] = None
        self.position: Optional[Tuple[int, int, int]] = None

    def clear(self) -> None:
        """Forget the place."""
        self.match = None
        self.position = None


# ── Tools ──────────────────────────────────────────────────────────


def _preflight_note(what: str, subject: str) -> str:
    """The skill tools' rehearsal sentence, present only when the skill
    preflight (``CFG.continual_skill_preflight``) is on: without it no arm
    rehearses a request before running it."""
    if not CFG.continual_skill_preflight:
        return ""
    return (f" The model arm first rehearses the {what} in `sim` from the "
            f"last observation: {subject} whose controller fails there is "
            "refused, charging nothing, with the controller's diagnostic.")


def _force_property(what: str) -> Dict[str, Any]:
    """The skill tools' ``force`` argument, offered only with the skill
    preflight it overrides."""
    if not CFG.continual_skill_preflight:
        return {}
    return {
        "force": {
            "type":
            "boolean",
            "description":
            f"run the {what} even when its rehearsal in "
            "`sim` fails (default false)",
        }
    }


def build_continual_tools(
    ctx: ToolContext,
    session: ProtocolSession,
    state: PlayState,
    *,
    save_render: Callable[[str], Optional[str]],
    tool_names: Optional[Sequence[str]] = None,
) -> List[Any]:
    """The ``SdkMcpTool`` instances of a play session.

    ``save_render(tag)`` saves a render of the current state into the
    sandbox and returns its sandbox-relative path, or ``None``. The
    model tools (``run_python`` with the ``sim`` probe) are attached by
    the arm, not built here.
    """
    # pylint: disable=import-outside-toplevel
    from claude_agent_sdk import tool

    # pylint: enable=import-outside-toplevel
    wanted = set(tool_names) if tool_names is not None else set(
        CONTINUAL_TOOL_NAMES)

    def _ended() -> Optional[Dict[str, Any]]:
        if session.executor.busy:
            return _error_result("A policy is running; wait for its call to "
                                 "finish before issuing another environment "
                                 "command." + _footer())
        if state.run_ended is not None:
            reason, note = state.run_ended
            return _error_result(f"The run has ended ({reason}"
                                 f"{': ' + note if note else ''}). Stop.")
        return None

    def _footer() -> str:
        try:
            return ("\n\n" + session.observe().ledger.footer() + "\n" +
                    context_status(ctx))
        except EpisodeOver:
            return ""

    def _gated() -> Optional[Dict[str, Any]]:
        """The arm's skill gate, when it installed one: a refusal that charges
        nothing, or None when the skill may run."""
        if ctx.skill_gate is None:
            return None
        reason = ctx.skill_gate()
        if reason is None:
            return None
        return _error_result(reason + " Nothing was charged." + _footer())

    def _practising() -> bool:
        """Whether the level is a reset training level under
        ``continual_practise_after_reset``."""
        return ctx.practise_gate is not None and ctx.practise_gate()

    def _raw_gated() -> Optional[Dict[str, Any]]:
        """Raw control on a reset training level waits for the model too, so it
        is no way around the skill gate."""
        return _gated() if _practising() else None

    def _matching_rehearsal(
        parsed: List[Tuple[_Option, Set[GroundAtom], Set[GroundAtom]]]
    ) -> Optional[_RehearsalMatch]:
        """The newest rehearsal on the joint belief from the current
        observation whose plan begins with the request's lines up to its last
        line other than Wait (its first line, for a request of Wait lines
        alone), if any."""
        if not ctx.live_rehearsals:
            return None
        needed = max(_motion_lines(parsed), 1)
        # The live frame: a skill step does not refresh
        # ctx.current_observation, so that copy can be a step behind. The
        # comparison reads the observed features only, never the memory a
        # model infers.
        current = session.observe().frame
        for rehearsal in reversed(ctx.live_rehearsals):
            if not states_close(rehearsal.observation, current):
                continue
            try:
                rehearsed = parse_plan_lines(rehearsal.plan_text, ctx,
                                             _level_task())
            except ValueError:
                continue
            shared = _shared_lines(rehearsed, parsed)
            if shared >= needed:
                return _RehearsalMatch(rehearsal, shared)
        return None

    plan_cursor = _PlanCursor()

    def _position() -> Tuple[int, int, int]:
        ledger = session.observe().ledger
        return (ledger.level_index, ledger.run_steps, ledger.run_resets)

    def _continuing_rehearsal(
        parsed: List[Tuple[_Option, Set[GroundAtom], Set[GroundAtom]]]
    ) -> Optional[_RehearsalMatch]:
        """The rehearsal whose plan the request continues: the last scored
        request ran its lines up to here, nothing has run since, and the
        request repeats its next lines (a Wait that lets a released body
        settle, say).

        The request is then scored against that rehearsal's draws at the
        line it reaches, as if the plan had run as one request.
        """
        previous = plan_cursor.match
        if previous is None or plan_cursor.position is None:
            return None
        try:
            if _position() != plan_cursor.position:
                return None
            rehearsed = parse_plan_lines(previous.rehearsal.plan_text, ctx,
                                         _level_task())
        except (EpisodeOver, ValueError):
            return None
        start = previous.start + previous.lines
        shared = _shared_lines(rehearsed[start:], parsed)
        if shared < max(_motion_lines(parsed), 1):
            return None
        return _RehearsalMatch(previous.rehearsal, shared, start)

    def _advance_cursor(match: Optional[_RehearsalMatch], num_lines: int,
                        progress: ExecutionProgress) -> None:
        """Keep the request's place in its rehearsal's plan when the rehearsal
        repeated all of it and all of it ran; else forget it."""
        plan_cursor.clear()
        if match is None or match.lines != num_lines or \
                len(progress.skills) != num_lines or \
                any(skill.result.status != "succeeded"
                    for skill in progress.skills):
            return
        try:
            plan_cursor.position = _position()
        except EpisodeOver:
            return
        plan_cursor.match = match

    def _unrehearsed(
            parsed: List[Tuple[_Option, Set[GroundAtom], Set[GroundAtom]]],
            match: Optional[_RehearsalMatch]) -> Optional[Dict[str, Any]]:
        """The arm's rehearsal gate, when it installed one and it applies now:
        a refusal that charges nothing, unless a rehearsal on the joint belief
        from the current observation began with the request's lines up to its
        last line other than Wait (``match``).

        A request of Wait lines alone commands no motion and passes.
        """
        practising = _practising()
        if not practising and (ctx.rehearsal_gate is None
                               or not ctx.rehearsal_gate()):
            return None
        if _motion_lines(parsed) == 0 or match is not None:
            return None
        return _error_result(
            f"{PRACTISE_HEAD if practising else 'Test level:'} rehearse "
            "this on the joint belief first. In "
            "run_python, print(sim.reset(current=True).run(plan)) with a "
            "plan whose first lines are this request, best followed by the "
            "rest of your plan: a rehearsal on the joint draws from the "
            "current observation lets the request through, and its P-hat is "
            "yours to weigh. A request of Wait lines alone needs none, and "
            "the Wait lines that end a request need not match the rehearsed "
            "plan's; runs at the point estimate (draws=0) and runs from "
            "staged states do not count. Nothing was charged." + _footer())

    def _risky(parsed: List[Tuple[_Option, Set[GroundAtom], Set[GroundAtom]]],
               match: Optional[_RehearsalMatch]) -> Optional[Dict[str, Any]]:
        """Under ``continual_rehearsal_game_over_gate``, while the rehearsal
        gate applies: a refusal that charges nothing when more than that share
        of the matching rehearsal's scored draws end the episode in GAME_OVER
        during the request's lines or the Wait lines that directly follow them
        in the rehearsed plan, where the scene moves on its own, or, with the
        draws whose scene was still moving when those lines ended, could.

        Those Wait lines must cover the request's own: a request that
        waits longer than its rehearsal did is refused, since the draws
        say nothing about the steps beyond. A request of Wait lines
        alone commands no motion and passes: what it waits through is
        already under way.
        """
        limit = float(CFG.continual_rehearsal_game_over_gate)
        if (limit <= 0.0 or match is None or ctx.rehearsal_gate is None
                or not ctx.rehearsal_gate()):
            return None
        motion = _motion_lines(parsed)
        if motion == 0:
            return None
        rehearsal = match.rehearsal
        try:
            rehearsed = parse_plan_lines(rehearsal.plan_text, ctx,
                                         _level_task())
        except ValueError:
            rehearsed = []
        horizon = motion
        while (horizon < len(rehearsed)
               and rehearsed[horizon][0].name == "Wait"):
            horizon += 1
        asked = _wait_steps(parsed[motion:])
        covered = _wait_steps(rehearsed[motion:horizon])
        if asked is not None and covered is not None and asked > covered:
            return _error_result(
                f"Test level: this request waits {asked} steps after its "
                f"last motion line, and the rehearsal that let it through "
                f"waited {covered}; its draws say nothing about the steps "
                "beyond, where a GAME_OVER can still come. Rehearse the plan "
                "with at least that much Wait after that line. Nothing was "
                "charged." + _footer())
        count = len(rehearsal.lost_lines)
        flags = rehearsal.scored or [True] * count
        moving = rehearsal.moving_lines or [[]] * count
        draws = lost = unsettled = 0
        for line, done, moved in zip(rehearsal.lost_lines, flags, moving):
            if not done:
                continue
            draws += 1
            if line is not None and line < horizon:
                lost += 1
            elif horizon - 1 in moved:
                unsettled += 1
        if not draws:
            return None
        if lost > limit * draws:
            return _error_result(
                f"Test level: {lost} of the {draws} scored draws of the "
                "rehearsal that let this request through end the episode in "
                "GAME_OVER during it or the Wait lines that follow it, more "
                f"than the {limit:.0%} of draws a test-level step may risk; a "
                "GAME_OVER ends a test level. Rehearse a step whose draws "
                "avoid it, or, if your model is wrong there, repair it and "
                "rehearse again. Nothing was charged." + _footer())
        if lost + unsettled > limit * draws:
            return _error_result(
                f"Test level: in {unsettled} of the {draws} scored draws of "
                "the rehearsal that let this request through, the scene was "
                "still moving when the Wait lines after it ended, so they do "
                "not show how it ends; with the "
                f"{lost} lost in it, more than the {limit:.0%} of draws a "
                "test-level step may risk could still end in GAME_OVER, and "
                "what happens until the scene rests follows from this request. "
                "Rehearse it with more Wait after it, until the scene rests. "
                "Nothing was charged." + _footer())
        return None

    def _scene_moving() -> bool:
        """Whether the belief sees a body moving (the velocities
        ``observation_belief.with_motion`` records)."""
        sim_state = getattr(session.observe().frame, "simulator_state", None)
        if not isinstance(sim_state, dict):
            return False
        velocities = sim_state.get("body_velocities")
        return isinstance(velocities, dict) and any(
            any(linear) for linear, _ in velocities.values())

    def _lost_draws(match: _RehearsalMatch) -> int:
        """The draws of the match's rehearsal lost at its next line or
        later."""
        return sum(line is not None and line >= match.start
                   for line in match.rehearsal.lost_lines)

    def _lost_waiting() -> Optional[int]:
        """The lost draws of the newest rehearsal of Wait lines alone from the
        current observation, or None without one."""
        current = session.observe().frame
        for rehearsal in reversed(ctx.live_rehearsals):
            if not states_close(rehearsal.observation, current):
                continue
            try:
                rehearsed = parse_plan_lines(rehearsal.plan_text, ctx,
                                             _level_task())
            except ValueError:
                continue
            if rehearsed and _motion_lines(rehearsed) == 0:
                return sum(line is not None for line in rehearsal.lost_lines)
        return None

    def _safer_than_waiting(match: Optional[_RehearsalMatch]) -> bool:
        """Whether the scene moves and the request loses fewer draws than doing
        nothing: then acting is the safer choice."""
        waiting = _lost_waiting()
        return (match is not None and waiting is not None and _scene_moving()
                and _lost_draws(match) < waiting)

    def _unsafe_plan(
            parsed: List[Tuple[_Option, Set[GroundAtom], Set[GroundAtom]]],
            match: Optional[_RehearsalMatch]) -> Optional[Dict[str, Any]]:
        """Under ``continual_rehearsal_safe_plan``, while the rehearsal gate
        applies: a refusal that charges nothing when a draw of the request's
        rehearsal ends the episode in GAME_OVER at any later line of its plan.

        A request of Wait lines alone commands no motion and passes.
        """
        if (not CFG.continual_rehearsal_safe_plan or match is None
                or ctx.rehearsal_gate is None or not ctx.rehearsal_gate()
                or _motion_lines(parsed) == 0):
            return None
        lost = _lost_draws(match)
        if lost == 0 or _safer_than_waiting(match):
            return None
        return _error_result(
            f"Test level: {lost} of the {len(match.rehearsal.lost_lines)} "
            "draws of this plan's rehearsal end the episode in GAME_OVER, "
            "and a GAME_OVER ends a test level. A lost draw is a world the "
            "data still allow. Find a plan no draw loses, or settle what "
            "separates the lost draws: repair the model if it is wrong "
            "there, or collect data that rules their parameters out. While "
            "the scene moves, a plan that loses fewer draws than a rehearsal "
            "of Wait lines alone passes. Nothing was charged." + _footer())

    still_refused: Dict[str, int] = {}

    def _moving_start(
        parsed: List[Tuple[_Option, Set[GroundAtom], Set[GroundAtom]]],
        match: Optional[_RehearsalMatch],
        continuing: Optional[_RehearsalMatch],
    ) -> Optional[Dict[str, Any]]:
        """Under ``continual_rehearsal_still_start``, while the rehearsal gate
        applies: a refusal that charges nothing when a request starts a new
        plan while the belief sees a body moving.

        A request of Wait lines alone commands no motion and passes.
        """
        if (not CFG.continual_rehearsal_still_start
                or ctx.rehearsal_gate is None or not ctx.rehearsal_gate()
                or _motion_lines(parsed) == 0):
            return None
        if not _scene_moving():
            still_refused.clear()
            return None
        if continuing is not None or _safer_than_waiting(match):
            return None
        steps = session.observe().ledger.run_steps
        first = still_refused.setdefault("first", steps)
        waited = steps - first
        if waited >= CFG.continual_rehearsal_still_wait:
            return None
        return _error_result(
            "Test level: a body is moving, so this new plan would start from "
            "a speed estimated from noisy frames. Send Wait lines until the "
            "scene rests, and rehearse the whole plan from rest. If waiting "
            "itself is the danger, rehearse Wait lines alone from here: a "
            "plan that loses fewer draws than they do passes. After "
            f"{CFG.continual_rehearsal_still_wait} steps from this first "
            f"refusal ({waited} so far), any plan passes, for a scene that "
            "never rests. Nothing was charged." + _footer())

    def _unseen_notes(
            match: Optional[_RehearsalMatch]) -> Optional[Dict[str, Any]]:
        """Under ``continual_rehearsal_notes_gate``, while the rehearsal gate
        applies: a refusal that charges nothing when the rehearsal that let
        the request through ended with notes no output given to the agent
        has shown in full.

        The refusal shows them, so the request sent again runs.
        """
        practising = _practising()
        if (not CFG.continual_rehearsal_notes_gate or match is None
                or (not practising and
                    (ctx.rehearsal_gate is None or not ctx.rehearsal_gate()))):
            return None
        rehearsal = match.rehearsal
        unseen = [n for n in rehearsal.notes if n not in ctx.shown_notes]
        if not unseen:
            return None
        ctx.shown_notes.update(unseen)
        notes = "\n".join(f"NOTE: {note}" for note in unseen)
        return _error_result(
            f"{PRACTISE_HEAD if practising else 'Test level:'} the "
            "rehearsal that let this request through "
            f"(P-hat {rehearsal.p_hat:.2f}) ended with notes no output of "
            f"yours has shown:\n{notes}\nWeigh them before this step; send "
            "the request again to run it. Nothing was charged." + _footer())

    def _undeclared_materials(
            parsed: List[Tuple[_Option, Set[GroundAtom], Set[GroundAtom]]],
            match: Optional[_RehearsalMatch]) -> Optional[Dict[str, Any]]:
        """Under ``belief_undeclared_material_screen``, while the rehearsal
        gate applies: a refusal that charges nothing when the outcome of the
        rehearsed plan changes with a material the model does not declare,
        which no draw of the rehearsal varied.

        The screen (``ctx.material_screen``) runs once per rehearsal, on
        a request that begins the rehearsal's plan from its observation.
        Its note joins the rehearsal's notes as one the agent has seen,
        so the request sent again runs.
        """
        practising = _practising()
        if (ctx.material_screen is None or match is None or match.start
                or not _motion_lines(parsed)
                or (not practising and
                    (ctx.rehearsal_gate is None or not ctx.rehearsal_gate()))):
            return None
        rehearsal = match.rehearsal
        if rehearsal.screened:
            return None
        rehearsal.screened = True
        note = ctx.material_screen(rehearsal.plan_text)
        if note is None:
            return None
        rehearsal.notes.append(note)
        ctx.shown_notes.add(note)
        return _error_result(
            f"{PRACTISE_HEAD if practising else 'Test level:'} the "
            "rehearsal that let this request through "
            f"(P-hat {rehearsal.p_hat:.2f}) does not cover every value the "
            f"plan depends on.\nNOTE: {note}\nWeigh this before the step; "
            "send the request again to run it as it is. Nothing was "
            "charged." + _footer())

    def _unclaimed(
            parsed: List[Tuple[_Option, Set[GroundAtom], Set[GroundAtom]]],
            match: Optional[_RehearsalMatch]) -> Optional[Dict[str, Any]]:
        """Under ``continual_claim_check``: a refusal that charges nothing when
        a record of the belief behind this request is due.

        A request of Wait lines alone commands no motion and passes.
        """
        check = ctx.claim_check
        if check is None or _motion_lines(parsed) == 0:
            return None
        obs = session.observe()
        p_hat = None if match is None else match.rehearsal.p_hat
        trigger = check.due(obs.level.split, obs.ledger.level_index,
                            obs.ledger.level_resets, p_hat)
        if trigger is not None:
            return _error_result(
                check.refusal(trigger, obs.ledger.level_resets, p_hat) +
                " Nothing was charged." + _footer())
        # A test level has no resets, so a step there comes after the
        # experiment; a training level's experiment is itself a step.
        if obs.level.split == "test" and check.open_ids():
            return _error_result(
                "Claim check: call claims_resolve for the open records "
                f"before a step on a test level: {check.open_ids()}. Nothing "
                "was charged." + _footer())
        return None

    def _prediction_note(match: Optional[_RehearsalMatch],
                         progress: ExecutionProgress) -> str:
        """The predictive check of a request, under the arm's
        continual_predictive_check: its outcome after the last line the
        rehearsal that let it through repeated, scored against that
        rehearsal's draws there.

        A surprise is reported and raises the fitted program's
        temperature floor; an outcome within the draws adds nothing. The
        step has run by now, so a failure of the check is reported in
        place of the check and never hides the step's result.
        """
        try:
            return _scored_prediction(match, progress)
        except Exception as e:  # pylint: disable=broad-except
            logging.exception("Prediction check failed")
            return f"\nPrediction check: could not be evaluated: {e!r}"

    def _scored_prediction(match: Optional[_RehearsalMatch],
                           progress: ExecutionProgress) -> str:
        """:func:`_prediction_note` without its guard."""
        provider = ctx.scored_features_provider
        if provider is None or match is None:
            return ""
        rehearsal, num_lines, start = match
        ran = progress.skills[:num_lines]
        # A plan that stopped at or before the last shared line, by a
        # divergence or the end of the episode, is not scored.
        if (not rehearsal.line_states or len(ran) < num_lines or
            (progress.stop_reason and len(progress.skills) <= num_lines)):
            return ""
        # A controller that stopped short ran a different step from the one
        # the draws rehearsed.
        if any(skill.result.status != "succeeded" for skill in ran):
            return ""
        outcome = ran[-1].result.outcome
        if outcome.episode_state is not EpisodeState.NOT_FINISHED:
            return ""
        predicted: List[State] = []
        line = start + num_lines
        for states in rehearsal.line_states:
            last = states[line - 1] if len(states) >= line else None
            if last is not None:
                predicted.append(last)
        features = provider()
        if len(predicted) < 2 or not features:
            return ""
        frame = ran[-1].after
        noise = ObservationNoise.from_cfg()
        widths = (ctx.feature_width_provider()
                  if ctx.feature_width_provider is not None else None)
        check = check_prediction(frame, predicted, features, noise, widths)
        if check is not None:
            # Every check, surprising or not, so a run's log shows how
            # calibrated its rehearsals were.
            name, feat = check.worst
            logging.info(
                "Prediction check: worst %s.%s at z %.3g against critical "
                "%.3g over %d features and %d draws; factor %.3g.", name, feat,
                check.z_max, check.critical, check.num_features,
                check.num_draws, check.needed)
        if check is None or not check.surprising:
            return ""
        name, feat = check.worst
        # Wait lines past the rehearsed ones leave the draws no prediction
        # of the request's end, so the check scores the last line they
        # predicted.
        if start:
            which = (f"this outcome, which continues the rehearsed plan from "
                     f"its line {start + 1} to its line {line},")
        elif len(progress.skills) == num_lines:
            which = "this outcome"
        else:
            which = (f"the outcome of line {num_lines}, the last the "
                     "rehearsal repeated,")
        text = (
            f"\nPrediction check: {which} lies outside the rehearsal's "
            f"draws: {name}.{feat} is {check.observed:.4g} where the "
            f"{check.num_draws} draws gave {check.mean:.4g} +- "
            f"{check.spread:.2g}, {check.z_max:.1f} predictive spreads away "
            f"(a calibrated prediction stays within {check.critical:.1f} "
            f"over these {check.num_features} features).")
        point = (rehearsal.mean_states[line - 1]
                 if len(rehearsal.mean_states) >= line else None)
        if point is not None and point_predicts(
                frame, point, check.missed, noise, check.num_features, widths):
            sampled = ("" if CFG.belief_material_lines else
                       " and samples over its plausible range")
            return text + (
                " The rehearsal's rollout at the belief's mean predicted "
                "it, so the draws spread the wrong way, through a factor "
                "this outcome rules out (a material the model leaves "
                f"undeclared{sampled}, say), and the temperature stays.")
        if check.out_of_reach:
            missed = ", ".join(f"{n}.{f}" for n, f in check.out_of_reach)
            text += (f" The parameters the belief allows barely move {missed} "
                     "toward the outcome at any temperature: the model's "
                     "form, not its parameters' uncertainty, misses it there.")
        if check.inflation <= 1.0:
            return text + " The belief's temperature stays as it was."
        return text + _widen(rehearsal.basis, check.inflation)

    def _widen(basis: Optional[Tuple[str, float]], inflation: float) -> str:
        """Raise the temperature floor of the fitted program the surprised
        rehearsal drew from (``basis``) by ``inflation``, and say so."""
        if basis is None:
            return (" The draws came from the prior, before any fit of this "
                    "simulator.py, so the belief does not change.")
        provider = ctx.prediction_basis_provider
        current = provider() if provider is not None else None
        if provider is None or current is None or current[0] != basis[0]:
            return (" The rehearsal ran an earlier simulator.py, so the "
                    "belief does not change.")
        program, temperature = basis
        if ctx.predictive_program != program:
            ctx.predictive_program, ctx.predictive_temperature = program, 1.0
        ctx.predictive_temperature = max(ctx.predictive_temperature,
                                         temperature * inflation)
        after = provider()
        now = after[1] if after is not None else temperature
        return (" The fit understates this program's error on steps its "
                "recordings did not show, so its joint draws now come from "
                f"the belief at temperature {now:.3g} (this rehearsal drew "
                f"at {temperature:.3g}); a refit keeps that temperature as a "
                "floor, and an edit of simulator.py starts over.")

    def _preflight(plan_text: str,
                   args: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """The arm's skill preflight, when it installed one: the request
        rehearsed in its sim from the last real observation.

        A refusal charges nothing and carries the sim's diagnostic;
        ``force`` on the request skips the rehearsal.
        """
        if ctx.skill_preflight is None or bool(args.get("force", False)):
            return None
        reason = ctx.skill_preflight(plan_text)
        if reason is None:
            return None
        return _error_result(reason + " Nothing was charged; pass "
                             "force=true to run it anyway." + _footer())

    def _resets_allowed() -> bool:
        try:
            return session.resets_allowed
        except EpisodeOver:
            return True

    def _protocol_error(e: Exception) -> Dict[str, Any]:
        # pylint: disable-next=import-outside-toplevel
        from predicators.run.continual import LevelAlreadyWon, LevelLost, \
            ResetUnavailable, RunEnded, StepRefused
        if isinstance(e, LevelAlreadyWon):
            return _error_result("The level is already won; nothing more "
                                 "can be charged on it. Write your notes "
                                 "and stop." + _footer())
        if isinstance(e, LevelLost):
            return _error_result("The level is lost: its episode ended in "
                                 "GAME_OVER and this level has no resets, "
                                 "so nothing more can be charged on it. "
                                 "Write your notes and stop." + _footer())
        if isinstance(e, ResetUnavailable):
            return _error_result(f"{e}. Nothing was charged. Continue the "
                                 "episode; if it ends in GAME_OVER, write "
                                 "your notes and stop." + _footer())
        if isinstance(e, StepRefused):
            return _error_result(f"{e}" + _footer())
        if isinstance(e, EpisodeOver):
            if _resets_allowed():
                return _error_result(f"{e}. Call env_reset to start a new "
                                     "episode." + _footer())
            return _error_result(f"{e}. This level has no resets, so it "
                                 "is over. Write your notes and stop." +
                                 _footer())
        if isinstance(e, RunEnded):
            state.run_ended = (e.reason, e.note)
            return _error_result(f"RUN ENDED: {e.reason}"
                                 f"{': ' + e.note if e.note else ''}. No "
                                 "further environment interaction is "
                                 "possible. Stop.")
        logging.exception("[continual tools] unexpected error")
        if isinstance(e, (ValueError, TypeError)):
            # The agent's own request was malformed (a wrong action shape,
            # an unparseable plan line): the message is about its input,
            # not about the environment, so it is safe and useful to show.
            return _error_result(f"Error: {type(e).__name__}: {e}" + _footer())
        # Anything else comes from inside the harness or a controller and
        # may carry private geometry; the details stay in the host log.
        return _error_result("The request could not be completed. Inspect "
                             "the observation before continuing." + _footer())

    env_predicates = list(session.env_predicates)

    async def _execute(request: ExecutionRequest,
                       progress: ExecutionProgress,
                       observer: Optional[ExecutionObserver] = None) -> None:

        def attempted() -> None:
            state.charged_calls += 1

        if ctx.before_real_action is not None:
            ctx.before_real_action()
        observer = observer or ExecutionObserver()
        observer.on_attempt = attempted
        audit = ctx.execution_audit
        prediction = audit_safely(audit, "before", request) if audit else None
        try:
            await session.executor.execute(request, progress, observer)
        finally:
            if audit is not None and prediction is not None:
                audit_safely(audit, "after", prediction, progress)

    def _skill_reporter(lines: List[str], total: int = 0) -> ExecutionObserver:
        before: Set[GroundAtom] = set()

        def started(frame: State) -> None:
            nonlocal before
            before = visible_atoms(ctx, frame)

        def finished(invocation: SkillExecution) -> None:
            after = visible_atoms(ctx, invocation.after)
            text = _format_result(invocation.result, before, after)
            prefix = f"[{len(lines) + 1}/{total}] " if total else ""
            lines.append(prefix + text)

        return ExecutionObserver(before_skill=started, after_skill=finished)

    def _observe_text(with_state: bool, tag: str) -> str:
        obs = session.observe()
        ctx.current_observation = obs.frame
        ctx.current_belief = obs.belief
        render = save_render(tag)
        text = format_observation(obs,
                                  ctx,
                                  with_state=with_state,
                                  render_path=render,
                                  env_predicates=env_predicates)
        control = primitive_observation(session)
        if "env_run_policy" not in wanted:
            # Skill agents need current proprioception too, including before
            # the first action and after reset. Avoid repeating object data.
            control = {
                k: control[k]
                for k in ("joint_positions", "action_space")
            }
        text = "[control] " + json.dumps(control) + "\n" + text
        return text

    def _level_task() -> Task:
        obs = session.observe()
        return dataclasses.replace(obs.level.task, init=obs.frame)

    @tool("env_observe",
          "The current observation: episode state, level and goal, the "
          "environment's atoms, your predicates' atoms, every object's "
          "features, current robot joint positions and action-space order, "
          "a render of the scene, and the ledger. Free.", {
              "type": "object",
              "properties": {},
          })
    async def env_observe(args: Dict[str, Any]) -> Dict[str, Any]:
        del args
        ended = _ended()
        if ended is not None:
            return ended
        try:
            return _text_result(
                _observe_text(True, f"observe_{state.charged_calls:04d}"))
        except Exception as e:  # pylint: disable=broad-except
            return _protocol_error(e)

    # The protocol refuses a primitive step past the robot's speed.
    speed_rule = (" An action that moves a joint farther than its "
                  "max_change_per_step in the action space is refused, "
                  "uncharged." if CFG.continual_raw_step_limit else "")

    @tool(
        "env_step",
        "Apply ONE primitive action: a low-level action vector of the "
        "environment's action space. Counts one step." + speed_rule, {
            "type": "object",
            "properties": {
                "action": {
                    "type": "array",
                    "items": {
                        "type": "number"
                    },
                    "description": "the action vector"
                }
            },
            "required": ["action"],
        })
    async def env_step(args: Dict[str, Any]) -> Dict[str, Any]:
        ended = _ended()
        if ended is not None:
            return ended
        gated = _raw_gated()
        if gated is not None:
            return gated
        try:
            action = primitive_action(args.get("action", []),
                                      session.action_space)
            progress = ExecutionProgress()
            await _execute(PrimitiveAction(action), progress)
            outcome = progress.step_outcome
            assert outcome is not None
            text = (f"step applied; episode {outcome.state.value}"
                    f"{' (' + outcome.reason + ')' if outcome.reason else ''}"
                    "\n" +
                    _observe_text(False, f"step_{state.charged_calls:04d}"))
            return _text_result(text)
        except Exception as e:  # pylint: disable=broad-except
            return _protocol_error(e)

    @tool(
        "env_run_policy",
        "Execute a sandbox Python file defining get_action(observation, "
        "memory). It returns one low-level vector or None to stop. Each "
        "action costs one step. Receives the current observable JSON frame, "
        "joint positions, action-space contract, goal and step counts. "
        "Memory and module globals persist within this call only. Stops on "
        "None, max_steps, WIN, GAME_OVER, a cap, or an error." + speed_rule, {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string"
                },
                "max_steps": {
                    "type": "integer",
                    "minimum": 1
                },
            },
            "required": ["path", "max_steps"],
        })
    async def env_run_policy(args: Dict[str, Any]) -> Dict[str, Any]:
        ended = _ended()
        if ended is not None:
            return ended
        gated = _raw_gated()
        if gated is not None:
            return gated
        progress = ExecutionProgress()

        @asynccontextmanager
        async def open_controller() -> AsyncIterator[ActionSelector]:
            if not ctx.sandbox_dir:
                raise ValueError("policy execution requires a sandbox")
            obs = session.observe()
            seconds = min(
                CFG.agent_sdk_python_call_timeout,
                obs.ledger.wall_clock_cap_seconds - obs.ledger.active_seconds)
            async with open_primitive_policy(str(args.get("path", "")),
                                             ctx.sandbox_dir,
                                             seconds) as policy:

                async def get_action(
                        observation: ProtocolObservation) -> Optional[Action]:
                    values = await policy.action(
                        primitive_observation(session, observation))
                    if values is None:
                        return None
                    return primitive_action(values, session.action_space)

                yield get_action

        try:
            await _execute(
                ExecutePolicy(open_controller, args.get("max_steps", 0)),
                progress)
            return _text_result(
                f"policy applied {progress.steps} step(s)\n" +
                _observe_text(True, f"policy_{state.charged_calls:04d}"))
        except Exception as e:  # pylint: disable=broad-except
            # A step that reaches a run cap is recorded before step() raises.
            result = _protocol_error(e)
            result["content"][0]["text"] = (
                f"policy applied {progress.steps} step(s) before stopping\n" +
                result["content"][0]["text"])
            return result

    @tool(
        "env_reset",
        "Restart the current level from its initial state. Counts one "
        "step and one reset; a last resort, not a retry button (see the "
        "rules). The only valid action after GAME_OVER on a level with "
        "resets; on a level the observation marks 'no resets' (test "
        "levels by default) it is refused and GAME_OVER ends the level.", {
            "type": "object",
            "properties": {
                "note": {
                    "type": "string",
                    "description": "why you are resetting (recorded)"
                }
            },
        })
    async def env_reset(args: Dict[str, Any]) -> Dict[str, Any]:
        ended = _ended()
        if ended is not None:
            return ended
        try:
            await _execute(RequestReset(str(args.get("note", ""))),
                           ExecutionProgress())
            return _text_result(
                "reset done\n" +
                _observe_text(True, f"reset_{state.charged_calls:04d}"))
        except Exception as e:  # pylint: disable=broad-except
            return _protocol_error(e)

    give_up_refused = (" Refused on a training level, which can always be "
                       "reset."
                       if CFG.continual_refuse_training_give_up else "")

    @tool(
        "give_up",
        "Give up: end the run for this environment and forfeit every "
        "remaining level. Takes effect when you stop. A last resort." +
        give_up_refused, {
            "type": "object",
            "properties": {
                "note": {
                    "type": "string",
                    "description": "why (recorded)"
                }
            },
            "required": ["note"],
        })
    async def give_up(args: Dict[str, Any]) -> Dict[str, Any]:
        ended = _ended()
        if ended is not None:
            return ended
        if CFG.continual_refuse_training_give_up and \
                session.observe().level.split == "train":
            return _error_result(
                "Refused: this is a training level. It can always be reset "
                "(env_reset, one step), and a reset keeps every file and "
                "recording, while giving up forfeits every remaining level, "
                "the test levels included. Reset and try another approach; "
                "the run ends by itself when its steps or time run out." +
                _footer())
        state.pending_give_up = GiveUp(
            str(args.get("note", "")) or "agent gave up")
        return _text_result("Give-up recorded. It takes effect when you "
                            "stop and forfeits every remaining level: "
                            "write your notes and stop.")

    @tool("skills_list",
          "The skill library: typed signatures, parameter meanings and "
          "ranges, and the plan-line grammar. Free.", {
              "type": "object",
              "properties": {},
          })
    async def skills_list(args: Dict[str, Any]) -> Dict[str, Any]:
        del args
        digest = render_options_digest(
            session.list_skills(), gt_options_ref_path=ctx.gt_options_ref_path)
        return _text_result(digest + "\n\n" + GRAMMAR + _footer())

    @tool(
        "skills_invoke",
        "Invoke ONE skill from one plan line and run it to termination. "
        "Counts the steps it took. Annotate the expected outcome with "
        "`-> {atoms}` so a divergence is recorded." +
        _preflight_note("line", "a skill"), {
            "type": "object",
            "properties": {
                "skill": {
                    "type": "string",
                    "description": "one plan line"
                },
                "note": {
                    "type": "string",
                    "description": "what this invocation tests (recorded)"
                },
                **_force_property("skill"),
            },
            "required": ["skill"],
        })
    async def skills_invoke(args: Dict[str, Any]) -> Dict[str, Any]:
        ended = _ended()
        if ended is not None:
            return ended
        gated = _gated()
        if gated is not None:
            return gated
        try:
            parsed = parse_plan_lines(str(args.get("skill", "")), ctx,
                                      _level_task())
        except ValueError as e:
            return _error_result(f"Could not parse the skill line: {e}" +
                                 _footer())
        if len(parsed) != 1:
            return _error_result("skills_invoke takes exactly one line; use "
                                 "skills_execute_plan for several." +
                                 _footer())
        match = _matching_rehearsal(parsed)
        # Under the still start, the lines of a running plan go on the
        # rehearsal that started it, from rest.
        continuing = (_continuing_rehearsal(parsed)
                      if CFG.continual_rehearsal_still_start else None)
        unrehearsed = _unrehearsed(parsed, match or continuing)
        if unrehearsed is not None:
            return unrehearsed
        moving = _moving_start(parsed, match, continuing)
        if moving is not None:
            return moving
        unsafe = _unsafe_plan(parsed, match or continuing)
        if unsafe is not None:
            return unsafe
        risky = _risky(parsed, match)
        if risky is not None:
            return risky
        materials = _undeclared_materials(parsed, match)
        if materials is not None:
            return materials
        unseen = _unseen_notes(match)
        if unseen is not None:
            return unseen
        unclaimed = _unclaimed(parsed, match)
        if unclaimed is not None:
            return unclaimed
        refused = _preflight(str(args.get("skill", "")), args)
        if refused is not None:
            return refused
        option, expected, absent = parsed[0]
        try:
            progress = ExecutionProgress()
            lines: List[str] = []
            scored = match or _continuing_rehearsal(parsed)
            plan_cursor.clear()
            await _execute(
                ExecuteSkill(option, expected, str(args.get("note", "")),
                             absent), progress, _skill_reporter(lines))
            text = lines[0] + _prediction_note(scored, progress)
            _advance_cursor(scored, len(parsed), progress)
            render = save_render(f"invoke_{state.charged_calls:04d}")
            if render:
                text += f"\n[render] {render}"
            return _text_result(text + _footer())
        except Exception as e:  # pylint: disable=broad-except
            return _protocol_error(e)

    @tool(
        "skills_execute_plan",
        "Execute a plan: one skill per line, in order. Stops at a failed "
        "skill, at a divergence from an annotated expected outcome "
        "(unless stop_on_divergence is false), at WIN or at GAME_OVER. "
        "Counts the steps taken." + _preflight_note("plan", "a plan"), {
            "type": "object",
            "properties": {
                "plan": {
                    "type": "string",
                    "description": "plan text, one skill per line"
                },
                "stop_on_divergence": {
                    "type": "boolean",
                    "description": "default true"
                },
                "note": {
                    "type": "string",
                    "description": "what this plan tests (recorded)"
                },
                **_force_property("plan"),
            },
            "required": ["plan"],
        })
    async def skills_execute_plan(args: Dict[str, Any]) -> Dict[str, Any]:
        ended = _ended()
        if ended is not None:
            return ended
        gated = _gated()
        if gated is not None:
            return gated
        try:
            parsed = parse_plan_lines(str(args.get("plan", "")), ctx,
                                      _level_task())
        except ValueError as e:
            return _error_result(f"Could not parse the plan: {e}" + _footer())
        match = _matching_rehearsal(parsed)
        # Under the still start, the lines of a running plan go on the
        # rehearsal that started it, from rest.
        continuing = (_continuing_rehearsal(parsed)
                      if CFG.continual_rehearsal_still_start else None)
        unrehearsed = _unrehearsed(parsed, match or continuing)
        if unrehearsed is not None:
            return unrehearsed
        moving = _moving_start(parsed, match, continuing)
        if moving is not None:
            return moving
        unsafe = _unsafe_plan(parsed, match or continuing)
        if unsafe is not None:
            return unsafe
        risky = _risky(parsed, match)
        if risky is not None:
            return risky
        materials = _undeclared_materials(parsed, match)
        if materials is not None:
            return materials
        unseen = _unseen_notes(match)
        if unseen is not None:
            return unseen
        unclaimed = _unclaimed(parsed, match)
        if unclaimed is not None:
            return unclaimed
        refused = _preflight(str(args.get("plan", "")), args)
        if refused is not None:
            return refused
        stop = bool(args.get("stop_on_divergence", True))
        note = str(args.get("note", ""))
        progress = ExecutionProgress()
        lines: List[str] = []

        try:
            request = ExecutePlan(
                tuple(
                    ExecuteSkill(option, expected, note, absent)
                    for option, expected, absent in parsed), stop)
            scored = match or _continuing_rehearsal(parsed)
            plan_cursor.clear()
            await _execute(request, progress,
                           _skill_reporter(lines, len(parsed)))
            count = len(progress.skills)
            if progress.stop_reason == "divergence":
                lines.append(f"plan stopped after skill {count}: divergence")
            elif progress.stop_reason not in ("", "terminal"):
                lines.append(f"plan stopped: skill {count} "
                             f"{progress.stop_reason}")
            checked = _prediction_note(scored, progress)
            _advance_cursor(scored, len(parsed), progress)
            if checked:
                lines.append(checked.lstrip("\n"))
            render = save_render(f"plan_{state.charged_calls:04d}")
            if render:
                lines.append(f"[render] {render}")
            return _text_result("\n".join(lines) + _footer())
        except Exception as e:  # pylint: disable=broad-except
            if lines:
                partial = "\n".join(lines) + "\n"
                err = _protocol_error(e)
                err["content"][0]["text"] = partial + err["content"][0]["text"]
                return err
            return _protocol_error(e)

    def _claim_position() -> Dict[str, int]:
        ledger = session.observe().ledger
        return {
            "run_steps": ledger.run_steps,
            "rehearsals": len(ctx.live_rehearsals),
        }

    @tool(
        "claims_record",
        "Record the belief your next step leans on most, when a claim check "
        "asks for it. direct_tests lists the recordings that tested it "
        "directly. Free.", {
            "type":
            "object",
            "properties": {
                "claim": {
                    "type": "string",
                    "description": "the belief"
                },
                "rules_out": {
                    "type": "string",
                    "description": "the option or warning it sets aside"
                },
                "direct_tests": {
                    "type": "array",
                    "description": "each {recording: index as in "
                    "sim.fit(traj_idxs=...), what: what it showed}",
                    "items": {
                        "type": "object"
                    },
                },
                "what_varied": {
                    "type":
                    "string",
                    "description":
                    "what differed between those tests; say nothing "
                    "varied if so"
                },
                "rival": {
                    "type": "string",
                    "description": "another explanation of the same evidence"
                },
                "next_experiment": {
                    "type":
                    "string",
                    "description":
                    "the cheapest run that separates the claim from the "
                    "rival, real or rehearsed"
                },
                "predictions": {
                    "type":
                    "object",
                    "description":
                    "{claim: what it shows if the claim "
                    "holds, rival: what it shows if the rival does}",
                },
            },
            "required": [
                "claim", "rules_out", "direct_tests", "what_varied", "rival",
                "next_experiment", "predictions"
            ],
        })
    async def claims_record(args: Dict[str, Any]) -> Dict[str, Any]:
        ended = _ended()
        if ended is not None:
            return ended
        if ctx.claim_check is None or ctx.recording_count is None:
            return _error_result("No claim check runs here." + _footer())
        ledger = session.observe().ledger
        accepted, reply = ctx.claim_check.record(
            args,
            recordings=ctx.recording_count(),
            level=ledger.level_index,
            resets=ledger.level_resets,
            **_claim_position())
        return (_text_result if accepted else _error_result)(reply + _footer())

    @tool(
        "claims_resolve",
        "Close a claim record with what its experiment showed. Free.", {
            "type": "object",
            "properties": {
                "id": {
                    "type": "integer",
                    "description": "the record"
                },
                "verdict": {
                    "type": "string",
                    "description": "supported, refuted, unresolved or not_run"
                },
                "result": {
                    "type":
                    "string",
                    "description":
                    "what the experiment showed, or why it "
                    "did not run"
                },
            },
            "required": ["id", "verdict", "result"],
        })
    async def claims_resolve(args: Dict[str, Any]) -> Dict[str, Any]:
        ended = _ended()
        if ended is not None:
            return ended
        if ctx.claim_check is None:
            return _error_result("No claim check runs here." + _footer())
        closed, reply = ctx.claim_check.resolve(args, **_claim_position())
        return (_text_result if closed else _error_result)(reply + _footer())

    def _prediction_start(start: str) -> Tuple[List[State], List[Any]]:
        """The recorded states a prediction continues, with their labels; none
        for the level's initial state."""
        if start == "initial":
            return [], []
        if start == "episode":
            episodes = session.level_episodes()
            if episodes:
                episode = episodes[-1]
                return (list(episode["states"]),
                        list(step_option_labels(episode["actions"])))
        return [session.observe().frame], []

    @tool(
        "evaluate_trajectory",
        "Score a state sequence that you predict with a level's reward "
        "model. Free. The sequence is a start state followed by your "
        "predicted states. `states` holds one entry per predicted state: "
        "{object_name: {feature: value}} for the features that change from "
        "the state before it, {} for no change. `start` picks the start "
        "state: `observation` (the default) is the latest observation, "
        "`episode` is the current episode's recorded states and skill "
        "labels up to that observation, `initial` is the level's initial "
        "state. `actions` is optional: one label per predicted state, "
        "[\"Skill\", [\"object\", ...], [param, ...]], or null for a "
        "transition you do not attribute to a skill. `path` names a JSON "
        "file in the sandbox holding {\"states\": [...], \"actions\": "
        "[...]} in place of the two arguments. `task_idx` indexes the "
        "levels reached so far and scores from that level's initial state; "
        "omit it for the current level. Returns {\"reward\", \"solved\", "
        "\"note\"}. No simulator backs this tool. A rule that replays "
        "physics is not checked: the verdict then has solved false and no "
        "success bonus, and `note` says so.", {
            "type": "object",
            "properties": {
                "states": {
                    "type":
                    "array",
                    "description":
                    "one {object_name: {feature: value}} entry per "
                    "predicted state",
                    "items": {
                        "type": "object"
                    },
                },
                "actions": {
                    "type": "array",
                    "description": "one label or null per predicted state",
                },
                "path": {
                    "type":
                    "string",
                    "description":
                    "a JSON file in the sandbox with `states` and, "
                    "optionally, `actions`",
                },
                "start": {
                    "type": "string",
                    "description": "observation (default), episode or initial"
                },
                "task_idx": {
                    "type":
                    "integer",
                    "description":
                    "a level reached so far; omit for the "
                    "current level"
                },
            },
        })
    async def evaluate_trajectory(args: Dict[str, Any]) -> Dict[str, Any]:
        ended = _ended()
        if ended is not None:
            return ended
        try:
            changes, labels = args.get("states"), args.get("actions")
            path = args.get("path") or ""
            if path:
                if changes is not None or labels is not None:
                    raise ValueError("Pass `path`, or `states` with optional "
                                     "`actions`.")
                if not ctx.sandbox_dir:
                    raise ValueError("`path` requires a sandbox.")
                resolved, err = _resolve_sandbox_file(path, ctx.sandbox_dir)
                if err is not None:
                    raise ValueError(err)
                changes, labels = load_prediction_file(resolved)
            task_idx = args.get("task_idx")
            if task_idx is not None:
                task_idx = int(task_idx)
            start = args.get("start") or ("observation"
                                          if task_idx is None else "initial")
            if start not in PREDICTION_STARTS:
                raise ValueError("`start` must be one of " +
                                 ", ".join(PREDICTION_STARTS) + ".")
            task = level_task(ctx.train_tasks, ctx.current_task, task_idx)
            if (start != "initial" and ctx.current_task is not None
                    and task is not ctx.current_task):
                raise ValueError("Another level is scored from its initial "
                                 "state: omit `start` or pass `initial`.")
            states, recorded = _prediction_start(start)
            verdict = score_prediction(
                task,
                states or [task.init],
                recorded,
                changes if changes is not None else [],
                labels,
                which="(current level)" if task_idx is None else task_idx)
        except ValueError as e:
            # The request was malformed: the message is about its input.
            return _error_result(f"{e}" + _footer())
        except Exception as e:  # pylint: disable=broad-except
            return _protocol_error(e)
        return _text_result(json.dumps(verdict) + _footer())

    all_tools = {
        "evaluate_trajectory": evaluate_trajectory,
        "claims_record": claims_record,
        "claims_resolve": claims_resolve,
        "env_run_policy": env_run_policy,
        "env_observe": env_observe,
        "env_step": env_step,
        "env_reset": env_reset,
        "give_up": give_up,
        "skills_list": skills_list,
        "skills_invoke": skills_invoke,
        "skills_execute_plan": skills_execute_plan,
    }
    return [all_tools[n] for n in ALL_CONTINUAL_TOOL_NAMES if n in wanted]


def _format_result(result: InvocationResult, before: Set[GroundAtom],
                   after: Set[GroundAtom]) -> str:
    o = result.outcome
    params = ", ".join(f"{float(p):.4g}" for p in o.option.params)
    head = (f"{o.option.simple_str()}[{params}]: {o.status} after "
            f"{o.steps} steps" + (f" ({o.reason})" if o.reason else ""))
    lines = [head]
    if o.episode_state is EpisodeState.WIN:
        lines.append("  episode: WIN")
    elif o.episode_state is EpisodeState.GAME_OVER:
        lines.append(f"  episode: GAME_OVER ({o.reason})")
    if result.expected or result.present or result.missing:
        if result.diverged:
            parts = []
            if result.missing:
                parts.append("expected but absent: " +
                             ", ".join(sorted(str(a) for a in result.missing)))
            if result.present:
                parts.append("expected absent but present: " +
                             ", ".join(sorted(str(a) for a in result.present)))
            lines.append("  DIVERGENCE: " + "; ".join(parts))
        else:
            lines.append("  expected outcome held")
        if result.fractions:
            lines.append(
                "  under the belief (fraction of draws): " +
                describe_fractions(result.fractions, list(result.fractions)))
    lines.append(_atoms_change_line(before, after))
    return "\n".join(lines)
