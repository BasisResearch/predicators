"""The env, skill, learning and session tools of a continual-protocol play
session (docs/continual-protocol.md, section 5.1).

The tools are thin text adapters over ``ProtocolSession``: every charged
call goes through the session, which counts, records and enforces the
caps. The tools never judge intent; they report what happened and end
every result with the ledger line.

Two things a tool cannot do from inside a running SDK session, ending
the session and running a learning sub-session, are recorded on the
shared ``PlayState`` and acted on by the arm after the query returns.
"""
from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, \
    Optional, Sequence, Set, Tuple

import numpy as np

from predicators import utils
from predicators.agent_sdk.sketch_parsing import parse_sketch_from_text
from predicators.agent_sdk.tools.context import ToolContext
from predicators.agent_sdk.tools.digests import render_options_digest
from predicators.agent_sdk.tools.results import _error_result, _text_result
from predicators.observation_belief import atom_fractions, \
    describe_fractions, uncertain_atoms
from predicators.observation_noise import ObservationNoise
from predicators.run.episode import EpisodeOver, EpisodeState
from predicators.settings import CFG
from predicators.structs import Action, GroundAtom, Predicate, State, Task, \
    _Option

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

GRAMMAR = (
    "One skill per line: `Skill(obj:type, ...)[p1, p2] -> {Atom(obj:type), "
    "NOT Other(obj:type)}`. Exact params in `[]` (`[]` when the skill "
    "has none); the `-> {...}` expected outcome is optional.")


@dataclass
class PlayState:
    """What the tools record for the arm to act on after the query."""
    pending_give_up: Optional[str] = None
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


def format_observation(
        obs: "ProtocolObservation",
        ctx: ToolContext,
        *,
        with_state: bool,
        render_path: Optional[str],
        env_predicates: Optional[Iterable[Predicate]] = None) -> str:
    """The observation as text (section 5.2); ``env_predicates`` as in
    :func:`_split_atoms`."""
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
                 f"({spec.split} task {spec.task_idx}{resets_note}); goal "
                 f"atoms: {goal_text}")
    if spec.task.goal_nl:
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
    if obs.belief is not None:
        try:
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
        if obs.belief is not None and obs.belief.frames_used:
            lines.append("[belief] each object smoothed over the frames it "
                         "rested through (value+-spread):")
            for obj in sorted(obs.frame, key=lambda o: o.name):
                if obj.name in obs.belief.frames_used:
                    lines.append("  " + obs.belief.object_line(obj))
    if render_path:
        lines.append(f"[render] {render_path}")
    lines.append(obs.ledger.footer())
    lines.append(context_status(ctx))
    return "\n".join(lines)


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


# ── Tools ──────────────────────────────────────────────────────────


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

    def _resets_allowed() -> bool:
        try:
            return session.resets_allowed
        except EpisodeOver:
            return True

    def _protocol_error(e: Exception) -> Dict[str, Any]:
        # pylint: disable-next=import-outside-toplevel
        from predicators.run.continual import LevelAlreadyWon, LevelLost, \
            ResetUnavailable, RunEnded
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
        return _error_result(f"Error: {type(e).__name__}: {e}" + _footer())

    env_predicates = list(session.env_predicates)

    def _observe_text(with_state: bool, tag: str) -> str:
        obs = session.observe()
        ctx.current_observation = obs.frame
        ctx.current_belief = obs.belief
        render = save_render(tag)
        return format_observation(obs,
                                  ctx,
                                  with_state=with_state,
                                  render_path=render,
                                  env_predicates=env_predicates)

    def _level_task() -> Task:
        obs = session.observe()
        return dataclasses.replace(obs.level.task, init=obs.frame)

    @tool("env_observe",
          "The current observation: episode state, level and goal, the "
          "environment's atoms, your predicates' atoms, every object's "
          "features, a render of the scene, and the ledger. Free.", {
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

    @tool(
        "env_step",
        "Apply ONE primitive action: a low-level action vector of the "
        "environment's action space. Counts one step. Prefer skills; "
        "this is the raw primitive.", {
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
        try:
            arr = np.asarray(args.get("action", []), dtype=np.float32)
            shape = tuple(session.action_space.shape)
            if arr.shape != shape:
                return _error_result(
                    f"action must have shape {shape}, got {arr.shape}" +
                    _footer())
            state.charged_calls += 1
            outcome = session.step(Action(arr))
            text = (f"step applied; episode {outcome.state.value}"
                    f"{' (' + outcome.reason + ')' if outcome.reason else ''}"
                    "\n" +
                    _observe_text(False, f"step_{state.charged_calls:04d}"))
            return _text_result(text)
        except Exception as e:  # pylint: disable=broad-except
            return _protocol_error(e)

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
            state.charged_calls += 1
            session.reset(str(args.get("note", "")))
            return _text_result(
                "reset done\n" +
                _observe_text(True, f"reset_{state.charged_calls:04d}"))
        except Exception as e:  # pylint: disable=broad-except
            return _protocol_error(e)

    @tool(
        "give_up",
        "Give up: end the run for this environment and forfeit every "
        "remaining level. Takes effect when you stop. A last resort.", {
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
        state.pending_give_up = str(args.get("note", "")) or "agent gave up"
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
        "`-> {atoms}` so a divergence is recorded.", {
            "type": "object",
            "properties": {
                "skill": {
                    "type": "string",
                    "description": "one plan line"
                },
                "note": {
                    "type": "string",
                    "description": "what this invocation tests (recorded)"
                }
            },
            "required": ["skill"],
        })
    async def skills_invoke(args: Dict[str, Any]) -> Dict[str, Any]:
        ended = _ended()
        if ended is not None:
            return ended
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
        option, expected, absent = parsed[0]
        try:
            before = visible_atoms(ctx, session.observe().frame)
            state.charged_calls += 1
            result = session.invoke(option, expected,
                                    str(args.get("note", "")), absent)
            after = visible_atoms(ctx, session.observe().frame)
            text = _format_result(result, before, after)
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
        "Counts the steps taken.", {
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
                }
            },
            "required": ["plan"],
        })
    async def skills_execute_plan(args: Dict[str, Any]) -> Dict[str, Any]:
        ended = _ended()
        if ended is not None:
            return ended
        try:
            parsed = parse_plan_lines(str(args.get("plan", "")), ctx,
                                      _level_task())
        except ValueError as e:
            return _error_result(f"Could not parse the plan: {e}" + _footer())
        stop = bool(args.get("stop_on_divergence", True))
        note = str(args.get("note", ""))
        lines: List[str] = []
        try:
            for i, (option, expected, absent) in enumerate(parsed):
                before = visible_atoms(ctx, session.observe().frame)
                state.charged_calls += 1
                result = session.invoke(option, expected, note, absent)
                after = visible_atoms(ctx, session.observe().frame)
                lines.append(f"[{i + 1}/{len(parsed)}] " +
                             _format_result(result, before, after))
                if result.status != "succeeded":
                    lines.append(f"plan stopped: skill {i + 1} "
                                 f"{result.status}")
                    break
                if stop and result.diverged:
                    lines.append(f"plan stopped after skill {i + 1}: "
                                 "divergence")
                    break
                if result.outcome.episode_state is not \
                        EpisodeState.NOT_FINISHED:
                    break
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

    all_tools = {
        "env_observe": env_observe,
        "env_step": env_step,
        "env_reset": env_reset,
        "give_up": give_up,
        "skills_list": skills_list,
        "skills_invoke": skills_invoke,
        "skills_execute_plan": skills_execute_plan,
    }
    return [all_tools[n] for n in CONTINUAL_TOOL_NAMES if n in wanted]


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
