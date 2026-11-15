"""The rollout core of the probe classes: one deterministic rollout.

``_RolloutCore`` holds a current state and runs a skill plan from it
once, on the session's option model. It owns the state control (reset,
snapshots), the readers (``state``, ``atoms``, ``render``), the plan
parser and the forked single rollouts (``run_async``, ``gather``). It
has no belief, no draw, no trial, no sweep and no plan search:
``belief_probe._ProbeCore`` adds those on top of it, and the supplied
base simulator (``base_simulator_probe``) builds on this class alone.
"""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import logging
import os
import re
import time
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, \
    Sequence, Tuple, TypeVar, Union

from predicators import utils
from predicators.agent_sdk.config import RefinementConfig, ToolSurfaceConfig
from predicators.agent_sdk.probe_results import ProbeGatherResult, \
    ProbeResult, _BareNameFallbackDict
from predicators.agent_sdk.tools.context import absolute_rollout_seed
from predicators.agent_sdk.tools.scene import apply_state_modifications, \
    draw_pybullet_annotation, render_pybullet_image, render_scene_image
from predicators.agent_sdk.tools.verdicts import _EvalStateCollector, \
    load_ground_sampler_fns
from predicators.code_sim_learning.model_state import memory_written_note
from predicators.structs import State, Task, excluded_object_type_names

if TYPE_CHECKING:
    from predicators.agent_sdk.plan_execution import StepOutcome
    from predicators.agent_sdk.tools import ToolContext

# Modification format accepted by ``reset``: either the tools' list form
# ``[{"object": name, "features": {feat: val}}]`` or the terser dict form
# ``{name: {feat: val}}``.
Modifications = Union[List[Dict[str, Any]], Dict[str, Dict[str, float]]]


class ProbeBudgetExceeded(BaseException):
    """A probe call ran past a wall-clock budget.

    Raised cooperatively at probe checkpoints (every sim call) when the
    run_python per-call limit has expired, and injected asynchronously
    by the budget watchdog. ``run_python`` catches it specially: the
    code's printed output so far is returned with the budget message
    appended, so a stopped sweep still hands the agent its partial
    results.

    It derives from ``BaseException``, as ``KeyboardInterrupt`` does, so
    the ``except Exception`` handlers that guard individual steps of a
    fit do not swallow it: the watchdog fires once, and a fit that caught
    it ran on past its limit (fixes_r13 Bridge seed 1: 4534 s against a
    3600 s limit).
    """


def _check_time_budget(ctx: "ToolContext") -> None:
    """Raise :class:`ProbeBudgetExceeded` when the run_python call's time limit
    is up."""
    now = time.monotonic()
    call_dl = ctx.python_call_deadline
    if call_dl is not None and now > call_dl:
        call_timeout = ToolSurfaceConfig.from_cfg().python_call_timeout
        raise ProbeBudgetExceeded(
            f"this run_python call exceeded its "
            f"{call_timeout:.0f}s time limit "
            "and was stopped between sim calls; output printed so far is "
            "returned above. Large sweeps are expensive - narrow the "
            "candidate set (coarse-to-fine, fewer perturbations per "
            "candidate) and split work across calls so each returns "
            "within the limit.")


def _count_rollout(ctx: "ToolContext", n: int = 1) -> None:
    """Meter full-plan rollouts for the attempt budget footer."""
    ctx.attempt_rollout_count += n


def _fmt_params(params: Any) -> str:
    """``[p1, p2]`` rendering used everywhere the probe prints params, so
    values the agent copies out of one report parse back identically in
    another."""
    return "[" + ", ".join(f"{float(p):.4g}" for p in params) + "]"


def _fmt_option(option: Any) -> str:
    """``Name(obj1, obj2)[p1, p2]`` signature for reports."""
    objs = ", ".join(o.name for o in option.objects)
    return f"{option.name}({objs}){_fmt_params(option.params)}"


# Contact-pair lines reported per step before eliding the rest: enough
# for a busy step's story, small enough to not drown the step report.
_MAX_CONTACT_LINES_PER_STEP = 12


def _attach_step_contacts(events: List[Dict[str, Any]],
                          step_dicts: List[Dict[str, Any]]) -> None:
    """Bucket recorded contact events into per-option-step summaries.

    ``events`` come from ``stop_contact_recording`` (step-ordered, one
    env step per low-level action, so cumulative ``num_actions`` gives
    the option-step boundaries). Each step dict gains ``contacts``: one
    line per contact pair with its 0-based action span within the step,
    robot links listed first. The LAST step's bucket absorbs any events
    past the counted boundary: a partially-executed failing option
    reports ``num_actions=0`` while the env genuinely stepped, and
    dropping those events would blank the contacts of exactly the step
    under diagnosis.
    """
    idx = 0
    start = 0
    for i, s in enumerate(step_dicts):
        is_last = i == len(step_dicts) - 1
        end = float("inf") if is_last else start + s["num_actions"]
        # Insertion-ordered dict doubles as the report order.
        spans: Dict[Tuple[str, str], Tuple[int, int]] = {}
        while idx < len(events) and events[idx]["step"] <= end:
            ev = events[idx]
            idx += 1
            if ev["step"] <= start:
                continue
            a, b = ev["a"], ev["b"]
            # Canonicalize the unordered pair: robot link first, else
            # lexicographic, so the two orientations merge into one span.
            if a.startswith("robot:") != b.startswith("robot:"):
                if b.startswith("robot:"):
                    a, b = b, a
            elif a > b:
                a, b = b, a
            key = (a, b)
            rel = ev["step"] - start - 1
            if key not in spans:
                spans[key] = (rel, rel)
            else:
                spans[key] = (spans[key][0], rel)
        lines = []
        for key, (lo, hi) in spans.items():
            span = f"action {lo}" if lo == hi else f"actions {lo}-{hi}"
            lines.append(f"{key[0]} <-> {key[1]} ({span})")
        if len(lines) > _MAX_CONTACT_LINES_PER_STEP:
            extra = len(lines) - _MAX_CONTACT_LINES_PER_STEP
            lines = lines[:_MAX_CONTACT_LINES_PER_STEP] + [
                f"... +{extra} more pairs"
            ]
        s["contacts"] = lines
        start += s["num_actions"]


def _grounded_steps(sketch_steps: Any) -> List[Any]:
    """Each step of a parsed sketch grounded at its own parameters."""
    # pylint: disable-next=import-outside-toplevel
    import numpy as np

    # pylint: disable-next=import-outside-toplevel
    from predicators.agent_sdk import bilevel_sketch
    grounded: List[Any] = []
    for st in sketch_steps:
        params = (st.initial_params if st.initial_params is not None else
                  np.array([], dtype=np.float32))
        grounded.append(
            bilevel_sketch.ground_step(st, np.asarray(params,
                                                      dtype=np.float32)))
    return grounded


def _memory_written(env: Any) -> Dict[str, str]:
    """The model_state entries ``env``'s ``_domain_specific_step`` changed,
    each with the value it left (see
    ``PyBulletEnv.memory_written_by_step``)."""
    return dict(getattr(env, "memory_written_by_step", None) or {})


# How a single run's memory note starts: a rehearsal drops it from its
# rollout from the belief mean when its draws' note already says the same.
_RUN_MEMORY_LEAD = "While this model ran"


def _line_states(steps: Sequence[StepOutcome]) -> List[Optional[State]]:
    """Each plan line's state after it ran in a rehearsal draw, or None where
    the line did not run cleanly there (not initiable, no actions, a controller
    or env failure): such a draw did not rehearse the line, so its state says
    nothing about what the line does when it runs."""
    # pylint: disable-next=import-outside-toplevel
    from predicators.run.recording import sanitize_state
    return [
        sanitize_state(st.post_state)
        if st.post_state is not None and st.failure_reason is None else None
        for st in steps
    ]


# A step report lists at most this many changed flags.
_MAX_FLAG_CHANGES = 8


def _flag_changes(pre: State, post: State) -> List[str]:
    """The whole-number features (a switch's state, a flag, a count) of the
    objects other than the robot that a step changed, as ``name.feature old.

    -> new``: what a step did, for a model with no predicates to say it.
    """
    before = {obj.name: obj for obj in pre}
    out: List[str] = []
    for obj in sorted(post, key=lambda o: o.name):
        old_obj = before.get(obj.name)
        if old_obj is None or obj.type.name == "robot":
            continue
        for feature in obj.type.feature_names:
            if feature not in old_obj.type.feature_names:
                continue
            old = float(pre.get(old_obj, feature))
            new = float(post.get(obj, feature))
            if old != new and old.is_integer() and new.is_integer():
                out.append(f"{obj.name}.{feature} {old:g} -> {new:g}")
    if len(out) > _MAX_FLAG_CHANGES:
        out = out[:_MAX_FLAG_CHANGES] + [
            f"and {len(out) - _MAX_FLAG_CHANGES} more"
        ]
    return out


# A skill phase's controller failure names its skill and phase first.
_PHASE_FAILURE = re.compile(r"\[([A-Za-z_]\w*)/[^\]]+\]")


def _motion_failures(failures: Sequence[Optional[str]]) -> int:
    """How many of ``failures`` are a skill controller's own (an IK stall, no
    motion plan: ``[Skill/Phase] ...``), not a refusal to start."""
    return sum(1 for f in failures if f and _PHASE_FAILURE.search(f))


def _failed_skills(failures: Sequence[Optional[str]]) -> List[str]:
    """The skills whose controllers failed in ``failures``, in order."""
    names: List[str] = []
    for failure in failures:
        found = _PHASE_FAILURE.search(failure or "")
        if found and found.group(1) not in names:
            names.append(found.group(1))
    return names


def _motion_failure_note(
        lead: str,
        skills: Sequence[str] = (),
        record: Optional[Dict[str, Tuple[int, int]]] = None,
        frames: Optional[str] = None,
        spread: Optional[Tuple[float, float]] = None) -> Optional[str]:
    """``lead`` plus a caveat on a skill controller's failure in a rollout
    under observation noise: the real controller plans on the true scene, this
    one on observed or drawn positions, and a few millimetres can change its
    plan. With this run's real ``record`` (per skill, calls that succeeded and
    failed), the note gives each of ``skills``'s; ``frames`` (see
    :func:`_belief_frames_note`) says how a Wait would sharpen the start, and
    ``spread`` (see :func:`_belief_spread`) how near the state belief puts the
    plan's bodies, which is far nearer than one frame's noise once they have
    rested a while.

    None when the observations are exact. Under
    ``continual_point_values_only`` the note is ``lead`` with the real
    record alone, and None without a record: that arm's text names no
    noise.
    """
    # pylint: disable-next=import-outside-toplevel
    from predicators.observation_noise import ObservationNoise
    # pylint: disable-next=import-outside-toplevel
    from predicators.settings import CFG
    noise = ObservationNoise.from_cfg()
    if noise.position <= 0.0 and noise.orientation <= 0.0:
        return None
    real = [
        f"{name} {record[name][0]} of {sum(record[name])}" for name in skills
        if record and sum(record.get(name, (0, 0)))
    ]
    history = ("This run's real calls of it succeeded in: " + ", ".join(real) +
               ". " if real else "")
    if CFG.continual_point_values_only:
        return f"{lead} {history}".rstrip() if real else None
    sharpen = f"{frames} " if frames else ""
    if spread is None:
        start = ("these rollouts planned on positions with the observation "
                 f"noise ({noise.position:g} m, {noise.orientation:g} rad)")
    else:
        start = ("these rollouts planned on positions the state belief "
                 "estimates from noisy frames "
                 f"({noise.position:g} m, {noise.orientation:g} rad of noise "
                 f"each) to within about {spread[0]:.2g} m and "
                 f"{spread[1]:.2g} rad for the bodies this plan acts on")
    return (f"{lead} The real controller plans on the true scene; {start}, "
            "and a few millimetres can change the arm's plan. "
            f"{history}{sharpen}Weigh such a failure against the skill's "
            "real record and against starts a few millimetres away "
            "(reset(current=True, mods=...)) before you conclude the real "
            "skill would fail.")


def _skill_record(ctx: Any) -> Optional[Dict[str, Tuple[int, int]]]:
    """This run's real skill record from the session, if it has one."""
    provider = getattr(ctx, "skill_record_provider", None)
    if provider is None:
        return None
    try:
        return provider()
    except Exception:  # pylint: disable=broad-except
        logging.warning("Could not read the run's skill record.",
                        exc_info=True)
        return None


_Probe = TypeVar("_Probe", bound="_RolloutCore")


class _RolloutCore:
    """A current state and single rollouts of skill plans from it.

    The "current state" is just a ``State`` object; ``run`` executes
    from it (the option model resets the sim env from that state) and
    advances it to the rollout's final state.
    """

    def __init__(self, ctx: "ToolContext"):
        self._ctx = ctx
        self._state: Optional[State] = None
        self._base_task: Optional[Task] = None
        # True when the base task is the solve-time "current" task (a
        # plain reset()/first use) rather than an explicit train
        # task_idx. The instance outlives individual solve queries, so
        # current-task probes must follow ctx.current_task when the
        # harness re-points it (see _require_state).
        self._tracking_current_task = False
        self._snapshots: Dict[int, Tuple[State, bool]] = {}
        self._next_snapshot_id = 1
        # True while the current state IS the task's unmodified initial
        # state (no mods, no rollout since reset). Gates require_solved:
        # the task evaluator's staging rules reference the true init, so
        # a verdict from any other start would be silently wrong.
        self._pristine = False
        # Lazy fork registry for run_async (created on first use).
        self._async_registry: Optional[Any] = None

    # ── State control ────────────────────────────────────────────

    def _option_model(self) -> Any:
        """The option model probes execute against.

        Solve sessions bind the deployed belief model via
        ``ctx.option_model``. Synthesis sessions install
        ``ctx.probe_option_model_provider`` instead - a lazy builder
        over the candidate ``simulator.py`` the agent is editing (fresh
        LM fit, cached until the file changes) - so probes always
        exercise the latest belief model, never the stale pre-synthesis
        one (real physics on cycle 1: a live-env leak).
        """
        provider = self._ctx.probe_option_model_provider
        if provider is not None:
            return provider()
        return self._ctx.option_model

    def _fresh_scope(self) -> Optional[Callable[..., Any]]:
        """Select isolation for the model this probe actually executes."""
        # pylint: disable-next=import-outside-toplevel
        from predicators.settings import CFG
        if not CFG.agent_plan_validation_fresh_env:
            return None
        ctx = self._ctx
        if ctx.probe_option_model_provider is not None:
            return ctx.probe_validation_env_scope
        return ctx.validation_env_scope

    def _task_at(self, task_idx: Optional[int]) -> Task:
        """The task at ``task_idx`` in ``train_tasks`` (the levels reached so
        far), or the current level's task for None."""
        ctx = self._ctx
        if task_idx is None:
            if ctx.current_task is None:
                raise ValueError("No task_idx given and no current task set.")
            return ctx.current_task
        if not 0 <= task_idx < len(ctx.train_tasks):
            raise ValueError(
                f"Invalid task_idx {task_idx}. Available: "
                f"0-{len(ctx.train_tasks) - 1}, the levels reached so far; "
                "omit it for the current level.")
        return ctx.train_tasks[task_idx]

    def reset(self: _Probe,
              task_idx: Optional[int] = None,
              mods: Optional[Modifications] = None,
              current: bool = False) -> _Probe:
        """Set the current state to a task's initial state, optionally with
        object-feature overrides applied to a copy.

        ``task_idx`` indexes ``train_tasks``, the levels reached so far
        (0 is the first, the last is the current level); ``None`` uses
        the current level. ``current=True`` starts instead from the last
        real observation the session recorded (every env tool result
        refreshes it), with inferred memory refreshed for the current
        model, so a rollout begins where the environment is now; the
        level's goal stays the task, and evaluator-scored modes
        (``solved``, ``require_solved``) are off, since they reference
        the task's true initial state. Returns ``self`` so calls chain.
        """
        ctx = self._ctx
        _check_time_budget(ctx)
        if current:
            if task_idx is not None:
                raise ValueError("current=True starts from the last real "
                                 "observation; task_idx does not apply.")
            if ctx.current_observation_provider is not None:
                ctx.current_observation = ctx.current_observation_provider()
            if ctx.current_observation is None:
                raise ValueError(
                    "No real observation has been recorded in this session: "
                    "call env_observe first, or reset() for the level's "
                    "initial state.")
            if ctx.current_task is None:
                raise ValueError("No current task set.")
            state = ctx.current_observation
            if mods:
                state, _, err = apply_state_modifications(
                    state, self._normalize_mods(mods))
                if err:
                    raise ValueError(err)
            else:
                state = state.copy()
            self._state = state
            self._base_task = ctx.current_task
            self._tracking_current_task = True
            self._pristine = False
            return self
        # Continual play binds ctx.current_task to the level in progress
        # at every level start, so a plain reset() stages that level: on a
        # test level, a forced task_idx=0 staged the first training level.
        task = self._task_at(task_idx)
        state = task.init
        if mods:
            mod_list = self._normalize_mods(mods)
            state, _, err = apply_state_modifications(state, mod_list)
            if err:
                raise ValueError(err)
        else:
            state = state.copy()
        self._state = state
        self._base_task = task
        self._tracking_current_task = task_idx is None
        self._pristine = not mods
        return self

    @staticmethod
    def _normalize_mods(mods: Modifications) -> List[Dict[str, Any]]:
        if isinstance(mods, dict):
            return [{
                "object": name,
                "features": feats
            } for name, feats in mods.items()]
        return list(mods)

    def snapshot(self) -> int:
        """Bank a copy of the current state; returns an id for restore.

        Snapshots rewind the ABSTRACT state only: replays from a
        restored snapshot still share the session env's solver state and
        velocity residuals, so repeated restore-and-run trials are
        correlated with each other (and read optimistic vs. a fresh
        env). For honest reliability estimates pass ``trials=N`` to
        ``run``, which uses a fresh physics env per trial when the
        session provides one.
        """
        sid = self._next_snapshot_id
        self._next_snapshot_id += 1
        self._snapshots[sid] = (self._require_state().copy(), self._pristine)
        return sid

    def restore(self: _Probe, snapshot_id: int) -> _Probe:
        """Set the current state back to a snapshot."""
        self._require_state()  # follow a task change before restoring
        if snapshot_id not in self._snapshots:
            raise ValueError(f"Unknown snapshot id {snapshot_id}. "
                             f"Available: {sorted(self._snapshots)}")
        state, pristine = self._snapshots[snapshot_id]
        self._state = state.copy()
        self._pristine = pristine
        return self

    def drop(self: _Probe, snapshot_id: int) -> _Probe:
        """Free a banked snapshot (they otherwise live for the session)."""
        self._snapshots.pop(snapshot_id, None)
        return self

    def clear_snapshots(self: _Probe) -> _Probe:
        """Free every banked snapshot."""
        self._snapshots.clear()
        return self

    # ── Introspection ────────────────────────────────────────────

    def state(
        self,
        obj_name: Optional[str] = None
    ) -> Union[Dict[str, Dict[str, float]], Dict[str, float]]:
        """Full-precision feature dict of the current state.

        ``state()`` -> ``{"name:type": {feat: value}}`` for all objects
        - keys use the same ``name:type`` form as atoms, plan lines, and
        the prompt, so a key read anywhere else indexes here directly;
        ``state("domino_1")`` and ``state("domino_1:domino")`` both ->
        that object's ``{feat: value}``. The full dict also resolves
        BARE-name lookups (``state()["domino_1"]``) via a fallback, so
        agent helpers persisted from before the ``name:type`` keying
        keep working across a resume.
        """
        cur = self._require_state()
        # Hidden types (CFG.excluded_objects_in_state_str) are absent
        # from every listing the agent sees, this one included.
        hidden = excluded_object_type_names()
        listed = [obj for obj in cur if obj.type.name not in hidden]

        def _features(obj: Any) -> Dict[str, float]:
            return {
                feat: float(cur.get(obj, feat))
                for feat in obj.type.feature_names
            }

        if obj_name is not None:
            # Sweep loops call the single-object form per iteration;
            # keep it O(one object), not O(scene). Accept both the bare
            # name and the name:type form every other surface prints
            # (bare-only lookups cost a KeyError per session:
            # run_20260830 hit it in four sessions).
            bare = obj_name.split(":", 1)[0]
            for obj in listed:
                if obj.name == bare:
                    return _features(obj)
            raise ValueError(f"Unknown object '{obj_name}'. Available: "
                             f"{sorted(str(o) for o in listed)}")
        return _BareNameFallbackDict(
            {str(obj): _features(obj)
             for obj in sorted(listed, key=str)})

    def atoms(self) -> List[str]:
        """Sorted ground atoms true in the current state."""
        cur = self._require_state()
        preds = self._ctx.predicates
        return [str(a) for a in sorted(utils.abstract(cur, preds))]

    def render(
            self,
            label: str = "probe",
            annotations: Optional[List[Dict[str,
                                            Any]]] = None) -> Optional[str]:
        """Render the current state; returns the saved image path.

        ``annotations`` overlays temporary geometry for this render
        only: dicts with ``type`` of ``marker``
        (``position`` [x, y, z]), ``line`` (``from``/``to``), or
        ``rectangle`` (``min_corner``/``max_corner``), plus optional
        ``color`` [r, g, b], ``size``, and ``label`` text. Use it to
        mark candidate positions, offsets, and reference points on the
        staged scene.
        """
        self._require_available("render")
        # pylint: disable=import-outside-toplevel
        import pybullet as pb

        # pylint: enable=import-outside-toplevel
        ctx = self._ctx
        cur = self._require_state()
        ctx.test_call_id += 1
        if ctx.env is None:
            raise ValueError("No environment available for rendering.")
        if annotations:
            ctx.env._set_state(cur)  # pylint: disable=protected-access
            physics_id = ctx.env._physics_client_id  # pylint: disable=protected-access
            debug_ids: List[int] = []
            try:
                for ann in annotations:
                    debug_ids.extend(draw_pybullet_annotation(ann, physics_id))
                img = render_pybullet_image(ctx, f"probe_{label}")
            finally:
                # Remove only the drawn bodies (never the env's own),
                # also on a bad-annotation error mid-draw.
                for body_id in debug_ids:
                    try:
                        pb.removeBody(body_id, physicsClientId=physics_id)
                    except Exception:  # pylint: disable=broad-except
                        pass
        else:
            img = render_pybullet_image(ctx, f"probe_{label}", state=cur)
        saved = img.get("saved_path") if img else None
        # The saved filename is prefixed (iter/task/test counters), so a
        # guessed path never matches - print the real one even when the
        # agent doesn't print the return value (audited runs lost turns
        # Read-ing guessed names).
        if saved:
            print(f"Saved scene image (view with Read): {saved}")
        return saved

    def _parse_sketch(self, plan_text: str) -> Any:
        """Parse ``plan_text`` against the current state.

        Returns ``(probe_task, sketch_steps, all_predicates, notices)``;
        the probe task starts at the current state and carries no
        evaluator, and ``notices`` lists parse caveats to surface (e.g.
        region annotations ignored because ground samplers are off).
        Same grammar and parser as ``skills_execute_plan``, plus ``~
        [w]`` search regions.
        """
        # pylint: disable=import-outside-toplevel
        from predicators.agent_sdk import bilevel_sketch

        # pylint: enable=import-outside-toplevel
        ctx = self._ctx
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

        gs_fns, gs_err = load_ground_sampler_fns(ctx)
        if gs_err is not None:
            raise ValueError(gs_err)
        notices: List[str] = []
        sketch_steps = bilevel_sketch.parse_sketch_from_text(
            plan_text,
            probe_task,
            predicates=all_predicates,
            options=all_options,
            types=types,
            parse_continuous_params=True,
            strict=True,
            parse_ground_samplers=RefinementConfig.from_cfg().ground_samplers,
            ground_sampler_fns=gs_fns or None,
            notices=notices)
        if not sketch_steps:
            raise ValueError(
                "Parsed empty plan. Each line must be "
                "`Option(obj:type, ...)[params]` with a known option, typed "
                "object refs, and exact params in `[]`.")
        status = ctx.probe_param_status
        if status and not status.startswith("fitted ("):
            notices.append(f"PARAMS {status}.")
        return probe_task, sketch_steps, all_predicates, notices

    # ── Execution ────────────────────────────────────────────────

    def run(self,
            plan_text: str,
            render: bool = True,
            contacts: bool = False,
            seed: Optional[int] = None) -> ProbeResult:
        """Run a skill plan from the current state, once, and advance the
        current state to where the rollout ends."""
        ctx = self._ctx
        _check_time_budget(ctx)
        ctx.test_call_id += 1
        probe_task, sketch_steps, all_predicates, notices = \
            self._parse_sketch(plan_text)
        return self._run_once(probe_task,
                              _grounded_steps(sketch_steps),
                              sketch_steps,
                              all_predicates,
                              notices,
                              render=render,
                              contacts=contacts,
                              seed=seed)

    def _rollouts_in(self, run_kwargs: Dict[str, Any]) -> int:
        """How many rollouts a ``run`` call with these arguments makes."""
        del run_kwargs
        return 1

    def _run_once(
        self,
        probe_task: Task,
        grounded: List[Any],
        sketch_steps: Any,
        all_predicates: Any,
        notices: List[str],
        *,
        render: bool,
        contacts: bool,
        seed: Optional[int],
        fresh: bool = False,
        motion_context: Optional[Callable[[], Tuple[
            Optional[str], Optional[Tuple[float, float]]]]] = None
    ) -> ProbeResult:
        """One rollout of a parsed plan: the single implementation behind every
        probe class's plain ``run``.

        ``probe_task`` starts at the rollout's start state, ``grounded``
        holds the plan's steps at their parameters and ``notices`` the
        notes so far, which the result takes over. The rollout runs on
        the warm session env and advances the current state; with
        ``fresh`` it runs on a freshly constructed env and leaves the
        current state alone. ``motion_context`` gives what a caller with
        a state belief adds to the note on a controller failure (see
        :func:`_motion_failure_note`: its ``frames`` and ``spread``).
        """
        # pylint: disable-next=import-outside-toplevel
        from predicators.agent_sdk import bilevel_sketch
        ctx = self._ctx
        report_preds = ctx.predicates

        def _horizon_note(total_actions: int) -> Optional[str]:
            budget = ctx.execution_step_budget()
            if total_actions > budget:
                return (f"the rollout used {total_actions} low-level steps, "
                        f"more than the remaining execution step budget "
                        f"({budget}) - the real executor would run out "
                        "of steps, so shorten or speed up the plan.")
            return None

        run_fresh_scope = None
        if fresh:
            run_fresh_scope = self._fresh_scope()
            if run_fresh_scope is None:
                raise ValueError(
                    "fresh=True needs the session's fresh-env scope "
                    "(unavailable for this model). Rerun without "
                    "fresh=True.")
            if render:
                render = False
                notices.append(
                    "fresh=True: per-step renders skipped (the renderer "
                    "draws the shared session env, which this rollout does "
                    "not touch).")
            notices.append(
                "fresh=True: rollout ran on a freshly constructed env from "
                "the parsed start state; the shared session state was NOT "
                "advanced.")

        step_dicts: List[Dict[str, Any]] = []
        # The rollout's states for the result (bound once the option
        # model exists below; the collector reads its per-step
        # trajectory when it records one).
        collectors: List[_EvalStateCollector] = []

        def _on_step(i: int, outcome: Any) -> None:
            for c in collectors:
                c.on_step(i, outcome)
            sig = _fmt_option(outcome.option)
            added: List[str] = []
            deleted: List[str] = []
            failure = outcome.failure_reason
            if not outcome.initiable:
                failure = failure or "not initiable"
            changed: List[str] = []
            partial_actions: Optional[int] = None
            if outcome.post_state is not None:
                before = utils.abstract(outcome.pre_state, report_preds)
                after = utils.abstract(outcome.post_state, report_preds)
                added = [str(a) for a in sorted(after - before)]
                deleted = [str(a) for a in sorted(before - after)]
                changed = _flag_changes(outcome.pre_state, outcome.post_state)
                if failure is not None:
                    partial = getattr(model, "last_partial_trajectory", None)
                    if partial is not None and partial.actions:
                        partial_actions = len(partial.actions)
            if i == 0 and fresh:
                env = getattr(model, "sim_env", None)
                lossy = getattr(env, "_last_unreconstructible_features",
                                None) or []
                if lossy:
                    feats = ", ".join(
                        sorted({f"{obj.name}.{feat}"
                                for obj, feat in lossy}))
                    notices.append(
                        "the fresh env could not reconstruct the start "
                        f"state exactly (features: {feats}) - a failure "
                        "here may partly reflect start-state "
                        "reconstruction error, not plan margin.")
            # Per-step audit image; the env already sits at the
            # post-step state here.
            img = render_scene_image(
                ctx,
                f"probe_step_{i}_{outcome.option.name}") if render else None
            if (outcome.option.name == "Wait" and failure is None
                    and outcome.num_actions >= utils.wait_rollout_step_cap()
                    and
                    not (outcome.option.params.size == 1 and
                         0 < outcome.option.params[0] <= outcome.num_actions)):
                notices.append(
                    f"step {i} (Wait) ran to its step cap "
                    f"({outcome.num_actions} actions). Inspect the final "
                    "state and subgoal report to decide whether to wait "
                    "longer or revise the modeled dynamics.")
            step_dicts.append({
                "option":
                sig,
                "num_actions":
                outcome.num_actions,
                "failure":
                failure,
                "added":
                added,
                "deleted":
                deleted,
                "subgoals_missing":
                [str(a) for a in sorted(outcome.subgoal_missing or set())],
                "image":
                img.get("saved_path") if img else None,
                "changed":
                changed,
                "partial_actions":
                partial_actions,
            })

        _count_rollout(ctx)
        model = self._option_model()
        collectors.append(_EvalStateCollector(model, probe_task.init))
        # Contact recording rides on the physics env the option model
        # steps (its ``sim_env``, the same binding certificate probes
        # use); a candidate-simulator model without one degrades to a
        # notice instead of an error.
        contact_env: Any = None
        if contacts:
            env = getattr(model, "sim_env", None)
            if env is None or not hasattr(env, "start_contact_recording"):
                notices.append(
                    "contact recording unavailable: this session's option "
                    "model exposes no physics env.")
            else:
                contact_env = env
                contact_env.start_contact_recording()
        contact_events: List[Dict[str, Any]] = []
        memory_written: Dict[str, str] = {}
        if seed is not None:
            notices.append(f"rollout ran at planner seed {seed} (base seed "
                           f"overridden for this call)")
        try:
            with (run_fresh_scope() if run_fresh_scope is not None else
                  contextlib.nullcontext()), absolute_rollout_seed(seed):
                result = bilevel_sketch.execute_plan_forward(
                    probe_task,
                    grounded,
                    model,
                    predicates=all_predicates,
                    sketch=sketch_steps,
                    on_step=_on_step,
                    stop_on_failure=True)
                memory_written = _memory_written(
                    getattr(model, "sim_env", None))
        finally:
            if contact_env is not None:
                contact_events = contact_env.stop_contact_recording()
        if memory_written:
            notices.append(
                memory_written_note(_RUN_MEMORY_LEAD, memory_written))
        if contact_env is not None:
            counted = sum(s["num_actions"] for s in step_dicts)
            recorded = contact_env.contact_steps_recorded
            if recorded != counted:
                notices.append(
                    f"contact recording saw {recorded} env steps but the "
                    f"step report counts {counted} actions - a partially "
                    "executed failing step reports 0 actions, so its "
                    "contacts are attached to the last step and spans "
                    "there are relative to that step's start.")
            _attach_step_contacts(contact_events, step_dicts)
        # A step that failed partway leaves the world where it stopped, as
        # a real skill's failure does: an effect it had before it failed
        # (a switch it flipped) stays.
        stopped = (result.final_state
                   if result.stopped_state is None else result.stopped_state)
        if not fresh:
            # A fresh-env rollout's final state is not the warm session
            # state; advancing to it would silently mix substrates.
            self._state = stopped
            self._pristine = False
        final_atoms = [
            str(a) for a in sorted(utils.abstract(stopped, report_preds))
        ]
        # A real episode ends the moment its goal holds, so a goal the
        # failed step reached before it stopped counts as reached.
        goal_reached = result.goal_reached or (
            stopped is not result.final_state
            and probe_task.goal_holds(stopped))
        hn = _horizon_note(
            sum(s["num_actions"] + (s.get("partial_actions") or 0)
                for s in step_dicts))
        if hn is not None:
            notices.append(hn)
        if _motion_failures([s["failure"] for s in step_dicts]):
            skills = _failed_skills([s["failure"] for s in step_dicts])
            record = _skill_record(ctx)
            frames: Optional[str] = None
            spread: Optional[Tuple[float, float]] = None
            if motion_context is not None:
                frames, spread = motion_context()
            motion = _motion_failure_note(
                "A skill's controller failed in this rollout.", skills, record,
                frames, spread)
            if motion is not None:
                notices.append(motion)
        return ProbeResult(step_dicts,
                           goal_reached,
                           final_atoms,
                           stopped,
                           notices,
                           states=list(collectors[0].states),
                           line_states=_line_states(result.steps),
                           labels=list(collectors[0].labels),
                           coarse=collectors[0].coarse)

    # ── Async rollouts (launch now, gather later) ────────────────

    def _model_tag(self) -> Optional[str]:
        """Identity of the current belief model, stamped on handles.

        The sandbox ``simulator.py`` mtime: cheap, monotone under the
        agent's edits, and enough to flag results that predate the
        current file.
        """
        sandbox = getattr(self._ctx, "sandbox_dir", None)
        if not sandbox:
            return None
        path = os.path.join(sandbox, "simulator.py")
        try:
            return f"simulator.py@{os.path.getmtime(path):.3f}"
        except OSError:
            return None

    def run_async(self, plan_text: str, **kwargs: Any) -> Any:
        """Launch ``run(plan_text, ...)`` in a forked child; return a handle
        immediately.

        Launch several independent runs, keep working (edit the model,
        think - even across ``run_python`` calls, handles survive
        in the namespace), then collect with ``gather``. Two semantic
        differences from ``run``: the session state does NOT advance
        (the child executes on a forked copy - a measurement, not
        navigation), and rendering is unavailable (``render`` is forced
        off). The child snapshots the session AND the belief model at
        launch, so results describe the model as of the launch;
        ``gather`` flags results whose ``simulator.py`` is older than
        the current file. When parallel workers are disabled the run
        executes inline (under a snapshot, so the no-advance semantics
        hold) and the returned handle is already done.
        """
        self._require_available("run_async")
        # pylint: disable-next=import-outside-toplevel
        from predicators.agent_sdk.async_rollouts import \
            AsyncRolloutRegistry, CompletedRollout, async_rollouts_available
        # pylint: disable-next=import-outside-toplevel
        from predicators.settings import CFG
        ctx = self._ctx
        _check_time_budget(ctx)
        kwargs["render"] = False
        _count_rollout(ctx, self._rollouts_in(kwargs))
        tag = self._model_tag()
        workers = int(getattr(CFG, "agent_validation_parallel_workers", 0))
        if workers <= 1 or not async_rollouts_available():
            sid = self.snapshot()
            try:
                return CompletedRollout(result=self.run(plan_text, **kwargs),
                                        tag=tag)
            except Exception as e:  # pylint: disable=broad-except
                return CompletedRollout(error=f"{type(e).__name__}: {e}",
                                        tag=tag)
            finally:
                self.restore(sid)
                self.drop(sid)
        if self._async_registry is None:
            self._async_registry = AsyncRolloutRegistry(max_workers=workers)
        job = functools.partial(self.run, plan_text, **kwargs)
        return self._async_registry.launch(job, tag=tag)

    def gather(self,
               handles: Sequence[Any],
               timeout: Optional[float] = None) -> ProbeGatherResult:
        """Wait for ``run_async`` handles and summarize their statuses.

        Blocks up to ``timeout`` seconds (default 1500, under the
        session's per-call watchdog) while pumping queued launches.
        Details stay on the handles: ``print(handle.result)`` for a
        done one, ``handle.error`` for a failed one. Pending handles
        are still running - keep them and gather again later.
        """
        self._require_available("gather")
        # pylint: disable-next=import-outside-toplevel
        from predicators.agent_sdk.async_rollouts import AsyncRollout
        _check_time_budget(self._ctx)
        live = [h for h in handles if isinstance(h, AsyncRollout)]
        if live and self._async_registry is not None:
            self._async_registry.gather(
                live, timeout=1500.0 if timeout is None else timeout)
        done = [h for h in handles if h.done()]
        pending = [h for h in handles if not h.done()]
        current = self._model_tag()
        stale = [
            h for h in done
            if h.tag is not None and current is not None and h.tag != current
        ]
        return ProbeGatherResult(done=done,
                                 pending=pending,
                                 stale=stale,
                                 current_tag=current)

    # ── Internals ────────────────────────────────────────────────

    def _require_available(self, call: str) -> None:
        """Refuse a ``sim`` call this session's arm withholds."""
        if call in self._ctx.probe_disabled:
            raise RuntimeError(
                f"sim.{call} is unavailable in this session: the run_python "
                "description lists what `sim` offers here.")

    def _require_state(self) -> State:
        """Current state, following the harness's task pointer.

        The probe instance outlives individual solve queries while the
        harness re-points ``ctx.current_task`` per task; silently
        probing the PREVIOUS task's layout (identical object names, so
        plans would parse and run) is the stale-current-task bug class.
        When the tracked current task changes, reset to the new task's
        init and drop the old task's snapshots (restoring one would mix
        tasks).
        """
        if self._state is None:
            self.reset()
        elif (self._tracking_current_task
              and self._ctx.current_task is not None
              and self._ctx.current_task is not self._base_task):
            self._snapshots.clear()
            self.reset()
        assert self._state is not None
        return self._state
