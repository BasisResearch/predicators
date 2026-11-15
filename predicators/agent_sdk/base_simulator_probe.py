"""The supplied base simulator of the Direct + base simulator arm.

The arm's ``run_python`` binds ``sim`` to a :class:`BaseSimulator`: the
simulation calls of the probe the other arms have, over a world that the
harness builds by itself, the domain-general ``SceneBase`` with each
level's scene loaded from its manifest and asset files. No agent file is
loaded into that world. ``sim`` offers resets, skill rollouts with the
real controllers and the states those rollouts predict, and a rollout
can run at physical values the caller names. It has no fit, no belief,
no plan search and no call that adds a mechanism, and its own
documentation names none.
"""
from __future__ import annotations

import contextlib
import math
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, \
    Optional, Sequence
from typing import Type as TypingType

import numpy as np

from predicators.agent_sdk.belief_probe import Modifications, \
    ProbeGatherResult, ProbeResult, _check_time_budget, _ProbeCore
from predicators.agent_sdk.tools.context import ToolContext
from predicators.agent_sdk.tools.python_exec import _make_python_exec_tool
from predicators.agent_sdk.tools.results import _make_coercing_tool, \
    _make_spilling_text_result
from predicators.agent_sdk.tools.reward_model import level_task, \
    score_trajectory, with_features
from predicators.agent_sdk.tools.scene import apply_state_modifications
from predicators.code_sim_learning.scene_base import SceneBase, \
    scene_base_class
from predicators.settings import CFG
from predicators.structs import Action, LowLevelTrajectory, State, Type, \
    step_option_labels

# The probe core's calls that the probe behind this arm's ``sim`` does not
# have either: the reports on a model file and the calls that load one,
# the belief and its draws, plan search and policies. (The core has no
# harness fit.)
BASE_SIMULATOR_PROBE_ABSENT = frozenset({
    "score", "validate", "predicates", "check_restore", "task", "belief",
    "suggest_probes", "refine", "run_policy", "run_points"
})
# The ``run`` modes that probe refuses.
BASE_SIMULATOR_PROBE_DISABLED = frozenset({"trials", "belief"})

# A rollout note's advice to change a model: this arm has none.
_MODEL_ADVICE = " or revise the modeled dynamics"


def supplied_scene_class(env_name: str, types: Iterable[Type],
                         asset_dir: str) -> TypingType[SceneBase]:
    """The class of the supplied world: the ``SceneBase`` bound to the
    deployment and to the scene files under ``asset_dir``.

    It has no mechanism and no parameter, and each body keeps what the
    loader gives it. Its one addition: a world held at other physical
    values (``BaseSimulatorProbe`` holds it so for one call) reports
    them as its values, so a copy of it that a reward model builds to
    replay physics starts at them too.
    """
    base = scene_base_class(env_name, types, asset_dir)

    class SuppliedScene(base):  # type: ignore[misc,valid-type]
        """The scene as loaded, at the physical values it is held at."""

        def get_physical_param_info(self) -> Dict[str, Dict]:
            info = super().get_physical_param_info()
            for name, value in self._sampled_materials.items():
                if name in info:
                    info[name] = dict(info[name], default=value)
            return info

    return SuppliedScene


class BaseSimulatorProbe(_ProbeCore):
    """The probe behind this arm's ``sim``, over the supplied world.

    The world is the one behind the session's option model, which the
    harness built (:func:`supplied_scene_class`). ``reset`` also takes a
    recorded state, ``run`` is one rollout that can name physical
    values, and ``params`` lists the values the loaded scene has. The
    agent holds a :class:`BaseSimulator` over it, never the probe.
    """

    def _scene(self) -> Any:
        """The supplied world."""
        scene = getattr(self._option_model(), "sim_env", None)
        if scene is None:
            raise RuntimeError("The base simulator is not loaded.")
        return scene

    def reset(self,
              task_idx: Optional[int] = None,
              mods: Optional[Modifications] = None,
              current: bool = False,
              state: Optional[State] = None) -> "BaseSimulatorProbe":
        """Set the current state.

        The default is the initial observation of a level (``task_idx``
        indexes the levels reached so far; None is the current level).
        ``current=True`` is the latest real observation. ``state`` is a
        recorded state, such as ``trajectories[i].states[t]``;
        ``task_idx`` then names the level it belongs to (the
        trajectory's ``train_task_idx``; None is the current level).
        ``mods`` overrides features of a copy (``{'obj': {'x':
        1.05}}``). Returns ``self`` so calls chain.
        """
        if state is None:
            super().reset(task_idx=task_idx, mods=mods, current=current)
            return self
        if current:
            raise ValueError("Pass `state` or `current=True`, not both.")
        if not isinstance(state, State):
            raise ValueError("`state` must be a State, such as "
                             "trajectories[i].states[t].")
        _check_time_budget(self._ctx)
        task = self._task_at(task_idx)
        start = state.copy()
        if mods:
            start, _, err = apply_state_modifications(
                start, self._normalize_mods(mods))
            if err:
                raise ValueError(err)
        self._state = start
        self._base_task = task
        self._tracking_current_task = task_idx is None
        self._pristine = False
        return self

    def params(self) -> Dict[str, Dict[str, Any]]:
        """The physical values of the loaded scene.

        ``{name: {"value", "description"}}``: per object type with a
        body, its mass, frictions, restitution and damping, and the
        contact values of the static bodies that no observation names
        (``support``). A value is the one the loaded scene gives the
        type's first body. ``run(plan, phys_params={name: value})``
        runs one rollout at other values.
        """
        scene = self._scene()
        # Binding the current state's objects names their types' values.
        scene.own_material_specs(list(self._require_state()))
        return {
            name: {
                "value": float(entry["default"]),
                "description": str(entry["description"]),
            }
            for name, entry in sorted(scene.get_physical_param_info().items())
        }

    @contextlib.contextmanager
    def _at_values(
            self, phys_params: Optional[Mapping[str,
                                                float]]) -> Iterator[None]:
        """Hold the supplied world at ``phys_params`` inside the block, and at
        the values the loaded scene has after it."""
        if not phys_params:
            yield
            return
        loaded = self.params()
        unknown = sorted(set(phys_params) - set(loaded))
        if unknown:
            raise ValueError(f"Unknown physical values {unknown}. "
                             f"Available: {sorted(loaded)}")
        values = {name: float(value) for name, value in phys_params.items()}
        for name, value in values.items():
            if not math.isfinite(value) or value < 0.0 or (
                    name.endswith("_mass") and value == 0.0):
                raise ValueError(
                    f"{name}={value!r}: a physical value must be finite "
                    "and not negative, and a mass must be positive.")
        scene = self._scene()
        ctx = self._ctx
        scene.apply_physical_param_overrides(values)
        ctx.probe_param_status = "set for this call: " + ", ".join(
            f"{name}={value:g}" for name, value in sorted(values.items()))
        try:
            yield
        finally:
            ctx.probe_param_status = None
            # A value at the scene's own gives each body its own back.
            scene.apply_physical_param_overrides(
                {name: loaded[name]["value"]
                 for name in values})

    # A narrower signature than the probe core's: this arm has one
    # rollout per call, and none of the trial, sweep or draw modes.
    def run(  # type: ignore[override]  # pylint: disable=arguments-differ
            self,
            plan_text: str,
            render: bool = True,
            contacts: bool = False,
            seed: Optional[int] = None,
            phys_params: Optional[Mapping[str, float]] = None) -> ProbeResult:
        """Run a skill plan from the current state, once.

        ``plan_text`` has the grammar of ``skills_execute_plan``. The
        rollout runs the real skill controllers in the supplied world
        and advances the current state to where it ends. The result
        prints the per-step outcomes; ``states`` holds the predicted
        state after each low-level step (the start state first),
        ``labels`` the skill of each transition, ``line_states`` the
        state after each plan line and ``final_state`` the last one.
        ``render`` saves an image per step, ``contacts`` reports what
        touched what, ``seed`` sets the motion planner's seed, and
        ``phys_params`` (``{name: value}``, names from :meth:`params`)
        runs this rollout at other physical values.
        """
        with self._at_values(phys_params):
            result = super().run(plan_text,
                                 render=render,
                                 contacts=contacts,
                                 seed=seed)
        assert isinstance(result, ProbeResult)
        result.notes[:] = [
            note.replace(_MODEL_ADVICE, "") for note in result.notes
        ]
        return result


class BaseSimulator:
    """The supplied base simulator: the robot, the skills with their real
    controllers, and the bodies of each level's scene as the harness loads them
    from the scene manifest and the asset files.

    It holds a current state. ``reset`` sets it, ``run`` runs a skill
    plan from it and moves it to where the rollout ends, and ``state``
    reads it. Nothing here acts in the environment.
    """

    def __init__(self, probe: BaseSimulatorProbe) -> None:
        self._probe = probe

    def reset(self,
              task_idx: Optional[int] = None,
              mods: Optional[Modifications] = None,
              current: bool = False,
              state: Optional[State] = None) -> "BaseSimulator":
        """Set the current state.

        The default is the initial observation of a level (``task_idx``
        indexes the levels reached so far; None is the current level).
        ``current=True`` is the latest real observation. ``state`` is a
        recorded state, such as ``trajectories[i].states[t]``; for a
        state of an earlier level also pass
        ``task_idx=trajectories[i].train_task_idx``. ``mods`` overrides
        features of that state (``{'obj': {'x': 1.05}}``). Returns
        ``self`` so calls chain.
        """
        self._probe.reset(task_idx=task_idx,
                          mods=mods,
                          current=current,
                          state=state)
        return self

    def run(self,
            plan_text: str,
            render: bool = True,
            contacts: bool = False,
            seed: Optional[int] = None,
            phys_params: Optional[Mapping[str, float]] = None) -> ProbeResult:
        """Run a skill plan from the current state, once.

        ``plan_text`` has the grammar of ``skills_execute_plan``. The
        rollout runs the real skill controllers and moves the current
        state to where it ends. Print the result for the per-step
        outcomes. Its ``states`` hold the predicted state after each
        low-level step (the start state first), ``labels`` the skill of
        each transition, ``line_states`` the state after each plan line
        (None for a line that failed) and ``final_state`` the last one.
        ``render`` saves an image per step, ``contacts`` reports what
        touched what, ``seed`` sets the motion planner's seed, and
        ``phys_params`` (``{name: value}``, names from ``params()``)
        runs this rollout at other physical values.
        """
        return self._probe.run(plan_text,
                               render=render,
                               contacts=contacts,
                               seed=seed,
                               phys_params=phys_params)

    def run_async(self,
                  plan_text: str,
                  contacts: bool = False,
                  seed: Optional[int] = None,
                  phys_params: Optional[Mapping[str, float]] = None) -> Any:
        """Launch ``run(plan_text, ...)`` in a forked child and return a handle
        at once.

        The current state does not advance and nothing is rendered.
        Collect the handles with ``gather``; a done handle has
        ``.result``, a failed one ``.error``.
        """
        return self._probe.run_async(plan_text,
                                     contacts=contacts,
                                     seed=seed,
                                     phys_params=phys_params)

    def gather(self,
               handles: Sequence[Any],
               timeout: Optional[float] = None) -> ProbeGatherResult:
        """Wait for ``run_async`` handles, up to ``timeout`` seconds, and
        summarize them; a pending handle is still running."""
        return self._probe.gather(handles, timeout=timeout)

    def params(self) -> Dict[str, Dict[str, Any]]:
        """The physical values of the loaded scene, as ``{name: {"value",
        "description"}}``: mass, frictions, restitution and damping per object
        type with a body, and the contact values of the static bodies that no
        observation names (``support``)."""
        return self._probe.params()

    def state(self, obj_name: Optional[str] = None) -> Any:
        """The features of the current state at full precision:

        ``{"name:type": {feature: value}}``, or one object's ``{feature:
        value}`` for ``state("name")``.
        """
        return self._probe.state(obj_name)

    def atoms(self) -> List[str]:
        """The atoms of the supplied predicates that hold in the current
        state."""
        return self._probe.atoms()

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
        return self._probe.render(label, annotations)

    def snapshot(self) -> int:
        """Save a copy of the current state; the id that ``restore`` takes."""
        return self._probe.snapshot()

    def restore(self, snapshot_id: int) -> "BaseSimulator":
        """Set the current state back to a snapshot."""
        self._probe.restore(snapshot_id)
        return self

    def drop(self, snapshot_id: int) -> "BaseSimulator":
        """Free a snapshot."""
        self._probe.drop(snapshot_id)
        return self

    def clear_snapshots(self) -> "BaseSimulator":
        """Free every snapshot."""
        self._probe.clear_snapshots()
        return self


def _make_evaluate_trajectory(
        ctx: ToolContext,
        probe: BaseSimulatorProbe) -> Callable[..., Dict[str, Any]]:
    """The level's reward model for this arm's ``run_python``.

    A rule that replays physics replays in the supplied world.
    """

    def evaluate_trajectory(
            states: Sequence[State],
            actions: Optional[Sequence[Any]] = None,
            task_idx: Optional[int] = None,
            phys_params: Optional[Mapping[str,
                                          float]] = None) -> Dict[str, Any]:
        """Score ``states`` with a level's reward model.

        ``task_idx`` indexes the levels reached so far; omit it for the
        current level. ``states``: the sequence, ``states[t]`` before
        action ``t``. ``actions``: one entry per transition, a recorded
        ``Action``, a label ``(skill_name, (object_name, ...), (param,
        ...))`` or ``None``; omit for no labels. Returns ``{"reward",
        "solved", "note"}``. A rule that replays physics replays in the
        supplied base simulator, at ``phys_params`` when given, and
        ``note`` says what it replayed ("" when nothing was replayed).
        """
        labels: Optional[List[Any]] = None
        if actions is not None:
            labels = [
                step_option_labels([entry])[0]
                if isinstance(entry, Action) else entry for entry in actions
            ]
        task = level_task(ctx.train_tasks, ctx.current_task, task_idx)
        # pylint: disable=protected-access
        with probe._at_values(phys_params):
            return score_trajectory(
                task,
                states,
                labels,
                sim_env=probe._scene(),
                which="(current level)" if task_idx is None else task_idx)

    return evaluate_trajectory


def base_simulator_description(reward_model: bool) -> str:
    """The description of this arm's ``run_python``: every call of ``sim`` and
    the other names bound beside it."""
    limit = float(CFG.agent_sdk_synthesis_python_call_timeout)
    reward = (
        "`evaluate_trajectory(states, actions=None, task_idx=None, "
        "phys_params=None)` -> {reward, solved, note}, a level's reward "
        "model on a state sequence (`states[t]` before action `t`; "
        "`actions` holds one entry per transition, a recorded action, a "
        "label `(skill, (objects, ...), (params, ...))` as in a result's "
        "`labels`, or None; `task_idx` indexes the levels reached so far); "
        "a rule that replays physics replays in the supplied simulator, at "
        "`phys_params` when given, and `note` says what it replayed; "
        "`with_features(state, {'obj': {'feature': value}})`, a changed "
        "copy of a state; " if reward_model else "")
    return (
        "Execute Python code (`code`, or `path` to a .py file you wrote in "
        "the sandbox). The namespace lasts for the round: variables survive "
        "across calls, and code you want to keep belongs in sandbox files. "
        "Nothing run here acts in the environment. Bound: `sim`, the "
        "supplied base simulator: the robot, the skills with their real "
        "controllers, and the bodies of each level's scene as the harness "
        "loads them from the scene manifest and the asset files. It holds "
        "no hidden mechanism, it loads no file of yours, and no call adds "
        "forces, attachments, state or step code to it. "
        "`sim.reset(task_idx=None, mods=None, current=False, state=None)` "
        "sets the current state: a level's initial observation by default "
        "(`task_idx` indexes the levels reached so far; None is the "
        "current level), the latest real observation with `current=True`, "
        "or a recorded state with `state=trajectories[i].states[t]` (for "
        "a state of an earlier level, also pass "
        "`task_idx=trajectories[i].train_task_idx`); "
        "`mods={'obj': {'x': 1.05}}` overrides features of that state. "
        "`sim.run(plan_text, render=True, contacts=False, seed=None, "
        "phys_params=None)` runs a skill plan from the current state, once "
        "(same grammar as skills_execute_plan), and advances the current "
        "state to where the rollout ends; print the result for the "
        "per-step outcomes and the saved per-step scene-image paths (view "
        "them with the Read tool; pass render=False inside loops); the "
        "result has `states` (the predicted state after each low-level "
        "step, the start state first), `labels` (the skill of each "
        "transition), `line_states` (the state after each plan line, None "
        "for a line that failed) and `final_state`; `-> {subgoals}` "
        "annotations are checked; contacts=True reports, per step, which "
        "robot links touched which objects and which object pairs touched; "
        "`seed` sets the motion planner's seed. `sim.params()` lists the "
        "physical values of the loaded scene as {name: {value, "
        "description}}: mass, frictions, restitution and damping per "
        "object type, and the contact values of the static support; "
        "`phys_params={name: value}` runs that one rollout at other "
        "values, and the simulator returns to the loaded values after it. "
        "`sim.run_async(plan_text, ...)` launches the same run in a forked "
        "child and returns a handle at once (the current state does not "
        "advance; no rendering), and `sim.gather(handles, timeout=None)` "
        "waits and summarizes (read each handle's `.result` or `.error`). "
        "`sim.state()` / `sim.state('obj')` give the full-precision "
        "features of the current state; `sim.atoms()`; `sim.render(label, "
        "annotations=None)` saves an image of the current state and "
        "returns its path, with optional marker, line and rectangle "
        "overlays (`{'type': 'marker', 'position': [x, y, z], 'color': [r, "
        "g, b], 'size': s}`; lines use `from`/`to`, rectangles "
        "`min_corner`/`max_corner`); `sim.snapshot()` / `sim.restore(id)` "
        "save and restore the current state. The simulator reads each "
        "body's pose, colour and held flag from the engine; every other "
        "feature (a switch reading, a level, a flag) keeps the value of "
        "the state a rollout starts from. Also bound: `np`; `trajectories` "
        "(the recorded real episodes, the episode in progress included, "
        "read-only; each has `train_task_idx`, `states`, `actions`); " +
        reward + "print() output is returned; output over ~30k characters "
        "is saved under `tool_outputs/run_python/` with a preview. " +
        (f"Each call has a {limit:.0f} s wall-clock limit; output printed "
         "before a stop is returned." if limit > 0 else ""))


def build_base_simulator_tool(ctx: ToolContext,
                              trajectories: List[LowLevelTrajectory], *,
                              sandbox_dir: Optional[str],
                              sandbox_dir_for_agent: Optional[str],
                              reward_model: bool) -> Any:
    """This arm's ``run_python`` for one round: a new namespace with ``sim``,
    ``np``, the recording (``trajectories``, kept current in place by the
    caller) and, with ``reward_model``, the level's reward model."""
    # pylint: disable-next=import-outside-toplevel
    from claude_agent_sdk import tool as _sdk_tool
    probe = BaseSimulatorProbe(ctx)
    namespace: Dict[str, Any] = {
        "sim": BaseSimulator(probe),
        "np": np,
        "trajectories": trajectories,
    }
    if reward_model:
        namespace["evaluate_trajectory"] = _make_evaluate_trajectory(
            ctx, probe)
        namespace["with_features"] = with_features
    return _make_python_exec_tool(
        _make_coercing_tool(_sdk_tool),
        name="run_python",
        description=base_simulator_description(reward_model),
        exec_ns=namespace,
        sandbox_dir=sandbox_dir,
        sandbox_dir_for_agent=sandbox_dir_for_agent,
        text_result=_make_spilling_text_result(
            sandbox_dir, agent_prefix=sandbox_dir_for_agent),
        # The cap the model arms' run_python has.
        call_timeout_s=CFG.agent_sdk_synthesis_python_call_timeout,
    )
