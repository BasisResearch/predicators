"""The supplied scene simulator of the No learning arm.

The arm's ``run_python`` binds ``sim`` to a :class:`Simulator`: the
planning calls of EMPIRIC from assets' probe (resets, rehearsals on the
joint draws of the belief, single rollouts, plan refinement, the state
belief) over a world the harness builds by itself, the domain-general
``SceneBase`` with each level's scene loaded from its manifest and asset
files (``supplied_scene_class``). No agent file is loaded into that
world, and the arm has no parameters, so a joint draw is a draw of the
start state with its own planner seed.

The agent holds the :class:`Simulator` and never the probe behind it, so
the calls the arm does not have (a fit, residuals, validation, model
loading, a physics sweep, physical values per rollout) do not exist on
``sim``. Everything here that the agent can read, the docstrings, the
tool description, the reports and their notes, describes the simulator
as it is and names nothing else.
"""
from __future__ import annotations

import dataclasses
import logging
import re
import weakref
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np

from predicators.agent_sdk.belief_probe import ProbeBeliefState, \
    ProbeJointResult, _counts_game_over, _ProbeCore
from predicators.agent_sdk.probe_results import ProbeGatherResult, \
    ProbeRefineResult, ProbeResult, _StrLikeResult
from predicators.agent_sdk.rollout_probe import Modifications
from predicators.agent_sdk.tools.context import ToolContext
from predicators.agent_sdk.tools.python_exec import _make_python_exec_tool
from predicators.agent_sdk.tools.results import _make_coercing_tool, \
    _make_spilling_text_result, _region_syntax_blurb
from predicators.agent_sdk.tools.reward_model import level_task, \
    score_trajectory
from predicators.settings import CFG
from predicators.structs import Action, State, step_option_labels

# The probe core's calls that the probe behind this arm's ``sim`` does not
# have either: the reports on a model file and the calls that load one,
# the experiment ranking over parameters, policies and runs at given
# physical values. (The core has no harness fit.)
SCENE_SIMULATOR_PROBE_ABSENT = frozenset({
    "score", "validate", "predicates", "check_restore", "suggest_probes",
    "run_policy", "run_points"
})

# Passages of the probe core's notes that speak of a model the agent
# could change, each with this arm's wording.
_NOTE_WORDING = (
    # A Wait that ran to its step cap.
    (" or revise the modeled dynamics", ""),
    # The verdict of the rollout from the belief mean, which no draw shares.
    (", at the declared values (the values the planning model runs at), ", " "
     ),
    (", at the fit's point estimate (the values the planning model runs "
     "at), ", " "),
    ("the planning model predicts", "that rollout predicts"),
    # A draw whose start the world read back differently.
    ("simulator.py reads back", "the simulator reads back"),
    (" Make the model set and report these features exactly, or this "
     "rehearsal predicts from the wrong start.",
     " For those features this rehearsal predicts from that other start."),
)
# What a note of this arm never names. A note that still does after its
# rewording is left out of the report and logged for the harness.
_NOT_IN_A_NOTE = re.compile(
    r"simulator\.py|declar|\b(?:re)?fit|parameter|PARAMS|mechanism|"
    r"\b(?:the|your) model\b|learn", re.IGNORECASE)
# What a draw of a rehearsal report keeps of the probe core's record.
_DRAW_FIELDS = ("success", "solved", "lost", "lost_line", "goal_reached",
                "num_actions", "failure", "planner_seed", "line_states")


def arm_notes(notes: Sequence[str]) -> List[str]:
    """``notes`` in this arm's wording (``_NOTE_WORDING``), without any that
    still names what the arm does not have."""
    kept: List[str] = []
    for note in notes:
        for old, new in _NOTE_WORDING:
            note = note.replace(old, new)
        if _NOT_IN_A_NOTE.search(note):
            logging.warning("A probe note was left out of the report: %s",
                            note)
            continue
        kept.append(note)
    return kept


@dataclasses.dataclass(repr=False)
class Rehearsal(_StrLikeResult):
    """Outcome of ``sim.run(plan)``: the plan on the joint draws of the belief,
    then one rollout from the belief mean.

    ``draws`` holds one dict per draw: ``success`` (``solved`` when the
    task evaluator scored the draw, else ``goal_reached``), ``solved``
    (the evaluator's verdict on the episode so far followed by the
    draw's rollout, None when it could not score the draw), ``lost``
    (whether that evaluator ends the episode there unsolved, as a real
    episode ends in GAME_OVER; None when unscored), ``lost_line``,
    ``goal_reached``, ``num_actions``, ``failure``, ``planner_seed`` and
    ``line_states`` (the draw's state after each plan line, None for a
    line that did not run cleanly). ``p_hat`` is the fraction of
    successful draws and ``stderr`` its standard error. ``mean`` is the
    step-by-step rollout from the belief mean. Only that rollout
    advances the current state, and its fields (``states``, ``labels``,
    ``line_states``, ``final_state``, ``steps``, ``goal_reached``,
    ``final_atoms``) read through, as on a single rollout's result.
    """
    draws: List[Dict[str, Any]]
    p_hat: float
    stderr: float
    from_observation: bool
    mean: Optional[Any] = None
    notes: List[str] = dataclasses.field(default_factory=list)
    # The task evaluator's verdict on the rollout from the belief mean
    # (``solved``, ``lost``); None when it was not scored.
    mean_verdict: Optional[Dict[str, bool]] = None

    @property
    def successes(self) -> int:
        """How many draws succeeded."""
        return sum(1 for d in self.draws if d["success"])

    def __getattr__(self, name: str) -> Any:
        # String methods stay on this report; other unknown attributes
        # read the rollout from the belief mean.
        if not name.startswith("_") and not hasattr(str, name):
            mean = self.__dict__.get("mean")
            if mean is not None and hasattr(mean, name):
                return getattr(mean, name)
        return super().__getattr__(name)

    def _game_over_note(self) -> str:
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
        if not _counts_game_over() or self.successes == len(self.draws):
            return ""
        failed = sum(1 for d in self.draws if d["failure"])
        if not failed:
            return "; no draw stopped at a failed skill"
        return f"; {failed} draw(s) stopped at a failed skill"

    def __repr__(self) -> str:
        n = len(self.draws)
        scored = sum(1 for d in self.draws if d["solved"] is not None)
        source = ("each draw is a start state the observations allow"
                  if self.from_observation else
                  "the simulator is not on the current observation, so every "
                  "draw starts from its current state")
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
            head = f"  draw {k}"
            ended = (", GAME_OVER"
                     if d.get("lost") and _counts_game_over() else "")
            if d["failure"] and d["success"]:
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
        if 0 < self.successes < n:
            lines.append("  the draws differ in their start state and their "
                         "planner seed: a failing draw is a start, or a "
                         "path, that this plan does not tolerate.")
        mean_notes = getattr(self.mean, "notes", None) or []
        lines.extend(f"NOTE: {n_}" for n_ in self.notes
                     if n_ not in mean_notes)
        if self.mean is not None:
            lines.append("")
            lines.append("Step-by-step rollout from the belief mean:")
            lines.append(repr(self.mean))
        return "\n".join(lines)


@dataclasses.dataclass(repr=False)
class Gathered(ProbeGatherResult):
    """Outcome of ``sim.gather``: one status line per handle.

    The full reports stay on the handles (``print(handle.result)`` for a
    done handle, ``handle.error`` for a failed one). ``pending`` handles
    are still running: keep them and gather again later.
    """

    def __repr__(self) -> str:
        lines = [
            f"Gather: {len(self.done)} done, {len(self.pending)} pending."
        ]
        for h in self.done:
            head = f"  handle {h.index}"
            if not h.ok:
                first = str(h.error).splitlines()[0] if h.error else "unknown"
                lines.append(f"{head}: FAILED - {first}")
                continue
            r = h.result
            if isinstance(r, Rehearsal):
                summary = f"P-hat {r.p_hat:.2f}"
            elif isinstance(r, ProbeResult):
                summary = f"goal_reached={r.goal_reached}"
            else:
                summary = "done"
            lines.append(f"{head}: {summary} - print(handle.result) for the "
                         "full report")
        for h in self.pending:
            lines.append(f"  handle {h.index}: still running")
        return "\n".join(lines)


class SceneSimulatorProbe(_ProbeCore):
    """The probe behind this arm's ``sim``, over the supplied world.

    The world is the one behind the session's option model, which the
    harness built with no agent file. ``run`` is the probe core's
    rehearsal and single rollout with no mode that names a physical
    value, and its reports carry this arm's wording. The agent holds a
    :class:`Simulator` over it, never the probe.
    """

    def _scene(self) -> Any:
        """The supplied world."""
        scene = getattr(self._option_model(), "sim_env", None)
        if scene is None:
            raise RuntimeError("The simulator is not loaded.")
        return scene

    def _parse_sketch(self, plan_text: str) -> Any:
        # The probe core notes which values a model file runs at; this
        # arm's world has none to name.
        self._ctx.probe_param_status = None
        return super()._parse_sketch(plan_text)

    def _model_tag(self) -> Optional[str]:
        # The world never changes within a run, so no rollout is stale.
        return None

    def render(
            self,
            label: str = "probe",
            annotations: Optional[List[Dict[str,
                                            Any]]] = None) -> Optional[str]:
        # The world is built on first use; an image needs it.
        self._option_model()
        return super().render(label, annotations)

    def _run_rehearsal(self, plan_text: str, render: bool,
                       seed: Optional[int]) -> ProbeJointResult:
        ctx = self._ctx
        result = super()._run_rehearsal(plan_text, render, seed)
        result.notes[:] = arm_notes(result.notes)
        # The record the rehearsal gate holds shows the same notes.
        if (result.from_observation and ctx.live_rehearsals
                and ctx.live_rehearsals[-1].plan_text == plan_text):
            ctx.live_rehearsals[-1].notes = list(result.notes)
        return result

    # A narrower signature than the probe core's: a rehearsal on the joint
    # draws or one rollout, and none of the modes that vary a physical
    # value.
    def run(  # type: ignore[override]  # pylint: disable=arguments-differ
            self,
            plan_text: str,
            render: bool = True,
            draws: Optional[int] = None,
            contacts: bool = False,
            seed: Optional[int] = None) -> Union[ProbeResult, Rehearsal]:
        """Rehearse a skill plan from the current state on the joint draws of
        the belief, or with ``draws=0`` run it once."""
        result = super().run(plan_text,
                             render=render,
                             draws=draws,
                             contacts=contacts,
                             seed=seed)
        if isinstance(result, ProbeJointResult):
            return Rehearsal(
                draws=[{key: draw.get(key)
                        for key in _DRAW_FIELDS} for draw in result.draws],
                p_hat=result.p_hat,
                stderr=result.stderr,
                from_observation=result.from_observation,
                mean=result.mean,
                notes=list(result.notes),
                mean_verdict=result.mean_verdict)
        assert isinstance(result, ProbeResult)
        result.notes[:] = arm_notes(result.notes)
        return result

    def refine(self,
               sketch_text: str,
               timeout: float = 60.0,
               max_samples_per_step: Optional[int] = None,
               require_goal: bool = False,
               require_solved: bool = False) -> ProbeRefineResult:
        result = super().refine(sketch_text,
                                timeout=timeout,
                                max_samples_per_step=max_samples_per_step,
                                require_goal=require_goal,
                                require_solved=require_solved)
        if _NOT_IN_A_NOTE.search(result.note):
            logging.warning("A refine note was left out of the report: %s",
                            result.note)
            result.note = ""
        return result


# The probe behind each Simulator the agent holds.
_PROBES: "weakref.WeakKeyDictionary[Simulator, SceneSimulatorProbe]" = \
    weakref.WeakKeyDictionary()


class Simulator:
    """The simulator of the current scene: the robot, the skills with their
    real controllers, and the scene's bodies as loaded from the scene manifest
    and the asset files.

    It holds a current state. ``reset`` sets it, ``run`` rehearses a
    skill plan from it and moves it to where the rollout from the belief
    mean ends, ``refine`` searches a plan's skill parameters, and
    ``state`` reads it. Nothing here acts in the environment.
    """

    def __init__(self, probe: SceneSimulatorProbe) -> None:
        # The probe is kept off the instance: ``dir(sim)`` and ``vars(sim)``
        # list the arm's calls and nothing behind them.
        _PROBES[self] = probe

    def reset(self,
              task_idx: Optional[int] = None,
              mods: Optional[Modifications] = None,
              current: bool = False) -> "Simulator":
        """Set the current state.

        The default is the initial observation of a level (``task_idx``
        indexes ``train_tasks``, the levels reached so far; None is the
        current level). ``current=True`` is the latest real observation.
        ``mods`` overrides features of that state (``{'obj': {'x':
        1.05}}``). Returns ``self`` so calls chain.
        """
        _PROBES[self].reset(task_idx=task_idx, mods=mods, current=current)
        return self

    def task(self, task_idx: Optional[int] = None) -> str:
        """Describe a level: its goal, objects, initial atoms and initial
        state.

        ``task_idx`` indexes ``train_tasks``; None is the current level.
        The current state does not change.
        """
        return _PROBES[self].task(task_idx)

    def run(self,
            plan_text: str,
            render: bool = True,
            draws: Optional[int] = None,
            contacts: bool = False,
            seed: Optional[int] = None) -> Union[ProbeResult, Rehearsal]:
        """Rehearse a skill plan from the current state.

        ``plan_text`` has the grammar of ``skills_execute_plan``. A
        plain ``run(plan)`` runs the plan on the joint draws of the
        belief: each draw is a start state the observations allow, on a
        fresh copy of the simulator with its own motion-planner seed. It
        reports the success estimate P-hat with its standard error and
        each draw's outcome, scored by the task evaluator on the episode
        so far followed by the draw's rollout. Then it runs the plan
        step by step from the belief mean, with contacts, and moves the
        current state to where that rollout ends. When the simulator is
        not on the latest observation (after ``mods`` or a rollout),
        every draw starts from the current state.

        ``draws=0`` runs only the one rollout: from the belief mean when
        the simulator is on the latest observation, else from the
        current state. Its result prints the per-step outcomes;
        ``states`` holds the predicted state after each low-level step
        (the start state first), ``labels`` the skill of each
        transition, ``line_states`` the state after each plan line (None
        for a line that failed) and ``final_state`` the last one. Any
        other count than 0 and the draws' own is refused.

        ``render`` saves an image per step of the one rollout,
        ``contacts`` (with ``draws=0``) reports what touched what, and
        ``seed`` sets the motion planner's seed.
        """
        return _PROBES[self].run(plan_text,
                                 render=render,
                                 draws=draws,
                                 contacts=contacts,
                                 seed=seed)

    def refine(self,
               sketch_text: str,
               timeout: float = 60.0,
               require_goal: bool = False,
               require_solved: bool = False) -> ProbeRefineResult:
        """Search the skill parameters of a plan from the current state.

        ``sketch_text`` has the grammar of ``skills_execute_plan``. Each
        step's ``[params]`` seed the search: they are tried first, then
        values around them. A step succeeds when it runs and its ``->
        {subgoals}`` annotation holds; ``require_goal`` also asks for
        the goal atoms at the last step, and ``require_solved`` (from an
        unmodified ``reset()``) for the task evaluator's solve. On the
        latest observation the search starts from the belief mean, the
        successful proposals are scored on the joint draws, and the best
        comes back with its P-hat on fresh draws. The current state does
        not change. The result reports the plan with the values found,
        the samples each step took and the deepest near-miss.
        """
        return _PROBES[self].refine(sketch_text,
                                    timeout=timeout,
                                    require_goal=require_goal,
                                    require_solved=require_solved)

    def belief(self, draws: Optional[int] = None) -> ProbeBeliefState:
        """The state belief at the current state: each object's smoothed
        features with their spread, and the fraction of ``draws`` belief draws
        on which each atom holds."""
        return _PROBES[self].belief(draws)

    def run_async(self,
                  plan_text: str,
                  draws: Optional[int] = None,
                  contacts: bool = False,
                  seed: Optional[int] = None) -> Any:
        """Launch ``run(plan_text, ...)`` in a forked child and return a handle
        at once.

        The current state does not advance and nothing is rendered.
        Collect the handles with ``gather``; a done handle has
        ``.result``, a failed one ``.error``.
        """
        return _PROBES[self].run_async(plan_text,
                                       draws=draws,
                                       contacts=contacts,
                                       seed=seed)

    def gather(self,
               handles: Sequence[Any],
               timeout: Optional[float] = None) -> Gathered:
        """Wait for ``run_async`` handles, up to ``timeout`` seconds, and
        summarize them; a pending handle is still running."""
        result = _PROBES[self].gather(handles, timeout=timeout)
        return Gathered(done=result.done,
                        pending=result.pending,
                        stale=[],
                        current_tag=None)

    def state(self, obj_name: Optional[str] = None) -> Any:
        """The features of the current state at full precision:

        ``{"name:type": {feature: value}}``, or one object's ``{feature:
        value}`` for ``state("name")``.
        """
        return _PROBES[self].state(obj_name)

    def atoms(self) -> List[str]:
        """The atoms of the supplied predicates that hold in the current
        state."""
        return _PROBES[self].atoms()

    def render(
            self,
            label: str = "state",
            annotations: Optional[List[Dict[str,
                                            Any]]] = None) -> Optional[str]:
        """Save an image of the current state and return its path.

        ``annotations`` overlays marker, line and rectangle dicts:
        ``{'type': 'marker', 'position': [x, y, z], 'color': [r, g, b],
        'size': s}``; a line has ``from`` and ``to``, a rectangle
        ``min_corner`` and ``max_corner``.
        """
        return _PROBES[self].render(label, annotations)

    def snapshot(self) -> int:
        """Save a copy of the current state; the id that ``restore`` takes."""
        return _PROBES[self].snapshot()

    def restore(self, snapshot_id: int) -> "Simulator":
        """Set the current state back to a snapshot."""
        _PROBES[self].restore(snapshot_id)
        return self

    def drop(self, snapshot_id: int) -> "Simulator":
        """Free a snapshot."""
        _PROBES[self].drop(snapshot_id)
        return self

    def clear_snapshots(self) -> "Simulator":
        """Free every snapshot."""
        _PROBES[self].clear_snapshots()
        return self


def _make_evaluate_trajectory(
        ctx: ToolContext,
        probe: SceneSimulatorProbe) -> Callable[..., Dict[str, Any]]:
    """The level's reward model for this arm's ``run_python``.

    A rule that replays physics replays in the supplied world.
    """

    def evaluate_trajectory(states: Sequence[State],
                            actions: Optional[Sequence[Any]] = None,
                            task_idx: Optional[int] = None) -> Dict[str, Any]:
        """Score ``states`` with a level's reward model.

        ``task_idx`` indexes ``train_tasks``, the levels reached so far;
        omit it for the current level. ``states``: the sequence,
        ``states[t]`` before action ``t``. ``actions``: one entry per
        transition, a recorded ``Action``, a label ``(skill_name,
        (object_name, ...), (param, ...))`` or ``None``; omit for no
        labels. Returns ``{"reward", "solved", "note"}``. A rule that
        replays physics replays in the simulator, and ``note`` says what
        it replayed ("" when nothing was replayed).
        """
        labels: Optional[List[Any]] = None
        if actions is not None:
            labels = [
                step_option_labels([entry])[0]
                if isinstance(entry, Action) else entry for entry in actions
            ]
        task = level_task(ctx.train_tasks, ctx.current_task, task_idx)
        # pylint: disable-next=protected-access
        scene = probe._scene()
        return score_trajectory(
            task,
            states,
            labels,
            sim_env=scene,
            which="(current level)" if task_idx is None else task_idx)

    return evaluate_trajectory


def scene_simulator_description() -> str:
    """The description of this arm's ``run_python``: every call of ``sim`` and
    the other names bound beside it."""
    limit = float(CFG.agent_sdk_synthesis_python_call_timeout)
    num = int(CFG.belief_joint_draws)
    return (
        "Execute Python code (`code`, or `path` to a .py file you wrote in "
        "the sandbox). The namespace lasts for the round: variables survive "
        "across calls, and code you want to keep belongs in sandbox files. "
        "Nothing run here acts in the environment. Bound: `sim`, the "
        "simulator of the current scene: the robot, the skills with their "
        "real controllers, and the scene's bodies as loaded from the scene "
        "manifest and the asset files. A body loaded from a URDF file has "
        "the file's masses, a moving body built from shapes has the mass of "
        "its volume at 1000 kg/m^3, and the other material values are the "
        "engine's defaults where no file sets one. The simulator reads each "
        "body's pose, colour and held flag from the engine; every other "
        "feature (a switch reading, a level, a flag) keeps the value of the "
        "state a rollout starts from. "
        "`sim.reset(task_idx=None, mods=None, current=False)` sets the "
        "current state: a level's initial observation by default (`task_idx` "
        "indexes `train_tasks`, the levels reached so far; None is the "
        "current level), or the latest real observation with `current=True`; "
        "`mods={'obj': {'x': 1.05}}` overrides features of that state. "
        "`sim.task(task_idx=None)` describes a level (goal, objects, initial "
        "atoms and state; the current level by default) without touching the "
        "current state. "
        "`sim.run(plan_text, render=True, draws=None, contacts=False, "
        "seed=None)` rehearses a skill plan FROM THE CURRENT STATE (same "
        f"grammar as skills_execute_plan) on the {num} joint draws of the "
        "belief - each draw is a start state the observations allow, on a "
        "fresh env with its own planner seed - and reports the success "
        "estimate P-hat with its standard error, scored by the TASK "
        "EVALUATOR on the episode so far followed by each draw's rollout "
        "(reaching the goal atoms is NOT the same as being scored a solve), "
        "and each draw's outcome, then a step-by-step rollout from the "
        "belief mean with contacts and saved per-step scene-image paths "
        "(view them with the Read tool), which advances the current state; "
        "`-> {subgoals}` annotations are CHECKED in that rollout; draws=0 "
        "runs only that rollout (a run at the point estimate), from the "
        "belief mean when the simulator sits on the last observation, else "
        "from its current state (pass render=False inside loops), and any "
        "count but 0 and the draws' own is refused; after `mods` or a "
        "rollout the simulator no longer sits on the last observation, and "
        "every draw then starts from its current state; contacts=True with "
        "draws=0 reports, per step, which robot links touched which objects "
        "and which object pairs touched, with action spans; `seed` sets the "
        "motion planner's seed. A rollout's result has `states` (the "
        "predicted state after each low-level step, the start state first), "
        "`labels` (the skill of each transition), `line_states` (the state "
        "after each plan line, None for a line that failed) and "
        "`final_state`; a rehearsal's result reads them from its rollout "
        "from the belief mean and adds `p_hat`, `stderr` and `draws`. "
        "`sim.belief()` lists the state belief with the atoms it is unsure "
        "about. "
        "`sim.refine(sketch_text, timeout=60, require_goal=False, "
        "require_solved=False)` runs backtracking search over the skill "
        "parameters from the belief mean (same grammar as "
        f"skills_execute_plan{_region_syntax_blurb()}; each step's "
        "`[params]` seed the search; success = each step runs and "
        "establishes its `-> {subgoals}` annotation; require_goal also asks "
        "for the goal atoms at the last step, and require_solved, from an "
        "unmodified `sim.reset()`, for the task evaluator's solve), scores "
        f"up to {int(CFG.belief_refine_candidates)} successful proposals on "
        "the common joint draws, and returns the best with its P-hat on "
        "fresh draws, an estimate its selection does not bias; the result "
        "reports per-step sample counts and the deepest near-miss, and the "
        "current state does not change. "
        "`sim.run_async(plan_text, ...)` launches the same run in a forked "
        "child and returns a handle at once (the current state does not "
        "advance; no rendering; a rehearsal run this way does not open a "
        "skill request), and `sim.gather(handles, timeout=None)` waits and "
        "summarizes (read each handle's `.result` or `.error`). "
        "`sim.state()` / `sim.state('obj')` give the full-precision "
        "features of the current state; `sim.atoms()`; `sim.render(label, "
        "annotations=None)` saves an image of the current state and "
        "returns its path, with optional marker, line and rectangle "
        "overlays (`{'type': 'marker', 'position': [x, y, z], 'color': [r, "
        "g, b], 'size': s}`; lines use `from`/`to`, rectangles "
        "`min_corner`/`max_corner`); `sim.snapshot()` / `sim.restore(id)` "
        "save and restore the current state. Also bound: `np`; "
        "`trajectories` (the recorded real episodes, the episode in "
        "progress included, read-only; each has `is_demo`, "
        "`train_task_idx`, `states`, `actions`); `train_tasks` (the levels "
        "reached so far; each has `init`, `goal`, `goal_holds(state)`); "
        "`describe_trajectory(traj_idx, include_states=True, "
        "include_atoms=False, max_timesteps=10)`, a per-timestep digest of "
        "one recorded episode; `is_goal_state(state, task_idx=None)`, "
        "whether the goal atoms hold in one state (reaching them does not "
        "by itself mean solved); `evaluate_trajectory(states, actions=None, "
        "task_idx=None)` -> {reward, solved, note}, a level's reward model "
        "on a state sequence (`states[t]` before action `t`; `actions` "
        "holds one entry per transition, a recorded action, a label "
        "`(skill, (objects, ...), (params, ...))` as in a result's "
        "`labels`, or None); a rule that replays physics replays in the "
        "simulator, and `note` says what it replayed. print() output is "
        "returned; output over ~30k characters is saved under "
        "`tool_outputs/run_python/` with a preview. " +
        (f"Each call has a {limit:.0f} s wall-clock limit; output printed "
         "before a stop is returned." if limit > 0 else ""))


def build_scene_simulator_tool(
        ctx: ToolContext,
        namespace: Dict[str, Any],
        *,
        sandbox_dir: Optional[str],
        sandbox_dir_for_agent: Optional[str],
        output_observer: Optional[Callable[[str], None]] = None) -> Any:
    """This arm's ``run_python`` for one round.

    ``namespace`` holds the recording and the level helpers; this adds
    ``sim``, ``np`` and the level's reward model, and returns the tool
    that runs code in it.
    """
    # pylint: disable-next=import-outside-toplevel
    from claude_agent_sdk import tool as _sdk_tool
    probe = SceneSimulatorProbe(ctx)
    namespace.update({
        "sim":
        Simulator(probe),
        "np":
        np,
        "evaluate_trajectory":
        _make_evaluate_trajectory(ctx, probe),
    })
    return _make_python_exec_tool(
        _make_coercing_tool(_sdk_tool),
        name="run_python",
        description=scene_simulator_description(),
        exec_ns=namespace,
        sandbox_dir=sandbox_dir,
        sandbox_dir_for_agent=sandbox_dir_for_agent,
        text_result=_make_spilling_text_result(
            sandbox_dir, agent_prefix=sandbox_dir_for_agent),
        # The cap the model arms' run_python has.
        call_timeout_s=CFG.agent_sdk_synthesis_python_call_timeout,
        output_observer=output_observer,
    )
