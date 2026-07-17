"""Exploration probe API exposed to solve-phase agents via ``explore_python``.

``ProbeSim`` is a thin facade over the machinery the curated tools
already use - ``parse_sketch_from_text`` (plan grammar),
``execute_plan_forward`` (forward executor over the option model), the
tools' state-modification and rendering helpers - so probe rollouts
behave identically to ``evaluate_option_plan`` rollouts. What it adds is
composability: the agent can set the sim to any task state (or a
modified copy), read full-precision features, run partial plans, render,
snapshot/restore, and write sweep loops in one ``explore_python`` call
instead of one tool round-trip per experiment.

By construction the probe carries NO scoring surface: it never touches a
task's evaluator, and nothing it executes can be captured as the
answer - submission happens only through ``evaluate_option_plan`` on the
true initial state.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional, Tuple, Union

from predicators import utils
from predicators.structs import State, Task

# Modification format accepted by ``reset``: either the tools' list form
# ``[{"object": name, "features": {feat: val}}]`` or the terser dict form
# ``{name: {feat: val}}``.
Modifications = Union[List[Dict[str, Any]], Dict[str, Dict[str, float]]]


def _fmt_params(params: Any) -> str:
    """``[p1, p2]`` rendering used everywhere the probe prints params, so
    values the agent copies out of one report parse back identically in
    another."""
    return "[" + ", ".join(f"{float(p):.4g}" for p in params) + "]"


def _fmt_option(option: Any) -> str:
    """``Name(obj1, obj2)[p1, p2]`` signature for reports."""
    objs = ", ".join(o.name for o in option.objects)
    return f"{option.name}({objs}){_fmt_params(option.params)}"


@dataclasses.dataclass(repr=False)
class ProbeResult:
    """Outcome of one ``ProbeSim.run`` call.

    Attributes mirror the ``evaluate_option_plan`` report: ``steps`` is
    a list of per-step dicts (``option``, ``num_actions``, ``failure``,
    ``added``, ``deleted``, ``image`` - the saved post-step scene
    image path, if rendering is available), plus ``goal_reached`` and
    ``final_atoms``. ``print(result)`` renders the same step-by-step
    summary the tool prints.
    """
    steps: List[Dict[str, Any]]
    goal_reached: bool
    final_atoms: List[str]
    final_state: State

    def __repr__(self) -> str:
        lines = []
        for i, s in enumerate(self.steps):
            line = f"Step {i}: {s['option']} ({s['num_actions']} actions)"
            if s["failure"]:
                line += f"\n  FAILURE: {s['failure']}"
            if s["added"] or s["deleted"]:
                line += (f"\n  Added: {{{', '.join(s['added'])}}}"
                         f"\n  Deleted: {{{', '.join(s['deleted'])}}}")
            lines.append(line)
        lines.append(f"Goal reached: {self.goal_reached}")
        lines.append(f"Final atoms: {{{', '.join(self.final_atoms)}}}")
        images = [s["image"] for s in self.steps if s.get("image")]
        if images:
            lines.append("Saved images (view with Read):")
            lines.extend(f"  {p}" for p in images)
        return "\n".join(lines)


@dataclasses.dataclass(repr=False)
class ProbeRefineResult:
    """Outcome of one ``ProbeSim.refine`` call.

    ``plan_lines`` holds one line per sketch step with the refined
    params filled in (``[?]`` for steps the search never refined) -
    paste them into ``sim.run`` or ``evaluate_option_plan``.
    ``near_miss`` is the deepest validation failure (step index, the
    exact params that got furthest, and why they failed), also populated
    on timeout/exhaustion. ``note`` carries caveats (e.g. the
    require_solved gate having been skipped on coarse rollouts).
    """
    success: bool
    reason: str
    total_samples: int
    step_samples: List[int]
    plan_lines: List[str]
    near_miss: Optional[Dict[str, Any]]
    note: str = ""

    def __repr__(self) -> str:
        lines = [
            f"Refinement {'SUCCESS' if self.success else 'FAILURE'} "
            f"({self.reason}): {self.total_samples} samples, per-step "
            f"{self.step_samples}"
        ]
        lines.append("Plan (refined params; [?] = never refined):")
        lines.extend(f"  {l}" for l in self.plan_lines)
        if self.near_miss is not None:
            lines.append(f"Deepest near-miss: step "
                         f"{self.near_miss['step_idx']} "
                         f"{self.near_miss['option']} - "
                         f"{self.near_miss['reason']}")
        if self.note:
            lines.append(f"NOTE: {self.note}")
        return "\n".join(lines)


class ProbeSim:
    """Stateful exploration handle over the belief simulator.

    Typical loop::

        sim.reset()                                   # true task init
        sim.reset(mods={"domino_1": {"x": 0.46}})     # modified copy
        sid = sim.snapshot()
        out = sim.run("Push(robot:robot, domino_0:domino)[0.05, 0.05]\\n"
                      "Wait(robot:robot)[]")
        print(out)             # per-step outcomes
        sim.state("domino_1")  # full-precision features
        sim.render("after_push")
        sim.restore(sid)
        # Search params for a suffix from here (nothing is captured):
        print(sim.refine(
            "Place(robot:robot)[0.46, 1.32, 0.55, -1.0] ~ [0.05, 0.05, "
            "0.0, 0.5] -> {SomeSubgoal(domino_1:domino)}"))

    The "current state" is just a ``State`` object; ``run`` executes from
    it (the option model resets the sim env from that state, exactly as
    ``evaluate_option_plan`` does from a task init) and advances it to
    the rollout's final state.
    """

    # Distinct deterministic rng streams per instance (see refine).
    _next_instance_id = 0

    def __init__(self, ctx: Any):
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
        self._refine_calls = 0
        self._instance_id = ProbeSim._next_instance_id
        ProbeSim._next_instance_id += 1
        # True while the current state IS the task's unmodified initial
        # state (no mods, no rollout since reset). Gates require_solved:
        # the task evaluator's staging rules reference the true init, so
        # a verdict from any other start would be silently wrong.
        self._pristine = False

    # ── State control ────────────────────────────────────────────

    def reset(self,
              task_idx: Optional[int] = None,
              mods: Optional[Modifications] = None) -> "ProbeSim":
        """Set the current state to a task's initial state, optionally with
        object-feature overrides applied to a copy.

        ``task_idx`` indexes the train tasks; ``None`` uses the current
        solve-time task. Returns ``self`` so calls chain.
        """
        # pylint: disable-next=import-outside-toplevel
        from predicators.agent_sdk.tools import _apply_state_modifications
        ctx = self._ctx
        if task_idx is not None:
            if not 0 <= task_idx < len(ctx.train_tasks):
                raise ValueError(f"Invalid task_idx {task_idx}. Available: "
                                 f"0-{len(ctx.train_tasks) - 1}")
            task = ctx.train_tasks[task_idx]
        elif ctx.current_task is not None:
            task = ctx.current_task
        else:
            raise ValueError("No task_idx given and no current task set.")
        state = task.init
        if mods:
            mod_list = self._normalize_mods(mods)
            state, _, err = _apply_state_modifications(state, mod_list)
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
        """Bank a copy of the current state; returns an id for restore."""
        sid = self._next_snapshot_id
        self._next_snapshot_id += 1
        self._snapshots[sid] = (self._require_state().copy(), self._pristine)
        return sid

    def restore(self, snapshot_id: int) -> "ProbeSim":
        """Set the current state back to a snapshot."""
        self._require_state()  # follow a task change before restoring
        if snapshot_id not in self._snapshots:
            raise ValueError(f"Unknown snapshot id {snapshot_id}. "
                             f"Available: {sorted(self._snapshots)}")
        state, pristine = self._snapshots[snapshot_id]
        self._state = state.copy()
        self._pristine = pristine
        return self

    def drop(self, snapshot_id: int) -> "ProbeSim":
        """Free a banked snapshot (they otherwise live for the session)."""
        self._snapshots.pop(snapshot_id, None)
        return self

    def clear_snapshots(self) -> "ProbeSim":
        """Free every banked snapshot."""
        self._snapshots.clear()
        return self

    # ── Introspection ────────────────────────────────────────────

    def state(
        self,
        obj_name: Optional[str] = None
    ) -> Union[Dict[str, Dict[str, float]], Dict[str, float]]:
        """Full-precision feature dict of the current state.

        ``state()`` -> ``{obj: {feat: value}}`` for all objects;
        ``state("domino_1")`` -> that object's ``{feat: value}``.
        """
        cur = self._require_state()

        def _features(obj: Any) -> Dict[str, float]:
            return {
                feat: float(cur.get(obj, feat))
                for feat in obj.type.feature_names
            }

        if obj_name is not None:
            # Sweep loops call the single-object form per iteration;
            # keep it O(one object), not O(scene).
            for obj in cur:
                if obj.name == obj_name:
                    return _features(obj)
            raise ValueError(f"Unknown object '{obj_name}'. Available: "
                             f"{sorted(o.name for o in cur)}")
        return {obj.name: _features(obj) for obj in sorted(cur, key=str)}

    def atoms(self) -> List[str]:
        """Sorted ground atoms true in the current state."""
        cur = self._require_state()
        preds = (self._ctx.predicates
                 | self._ctx.iteration_proposals.proposed_predicates)
        return [str(a) for a in sorted(utils.abstract(cur, preds))]

    def render(
            self,
            label: str = "probe",
            annotations: Optional[List[Dict[str,
                                            Any]]] = None) -> Optional[str]:
        """Render the current state; returns the saved image path.

        ``annotations`` overlays temporary geometry for this render only
        (the annotate_scene format): dicts with ``type`` of ``marker``
        (``position`` [x, y, z]), ``line`` (``from``/``to``), or
        ``rectangle`` (``min_corner``/``max_corner``), plus optional
        ``color`` [r, g, b], ``size``, and ``label`` text. Use it to
        mark candidate positions, offsets, and reference points on the
        staged scene.
        """
        # pylint: disable=import-outside-toplevel
        import pybullet as pb

        from predicators.agent_sdk.tools import _draw_pybullet_annotation, \
            _render_pybullet_image

        # pylint: enable=import-outside-toplevel
        ctx = self._ctx
        cur = self._require_state()
        ctx.test_call_id += 1
        if annotations:
            if ctx.env is None:
                raise ValueError("No environment available for rendering.")
            ctx.env._set_state(cur)  # pylint: disable=protected-access
            physics_id = ctx.env._physics_client_id  # pylint: disable=protected-access
            debug_ids: List[int] = []
            try:
                for ann in annotations:
                    debug_ids.extend(_draw_pybullet_annotation(
                        ann, physics_id))
                img = _render_pybullet_image(ctx, f"probe_{label}")
            finally:
                # Remove only the drawn bodies (never the env's own),
                # also on a bad-annotation error mid-draw.
                for body_id in debug_ids:
                    try:
                        pb.removeBody(body_id, physicsClientId=physics_id)
                    except Exception:  # pylint: disable=broad-except
                        pass
        else:
            img = _render_pybullet_image(ctx, f"probe_{label}", state=cur)
        # Same handoff visualize_state provides: a later annotate_scene
        # call (when offered) draws on the state the agent just staged
        # and rendered, not on the pristine task init.
        ctx.visualized_state = cur
        return img.get("saved_path") if img else None

    # ── Execution ────────────────────────────────────────────────

    def _parse_sketch(self, plan_text: str) -> Any:
        """Parse ``plan_text`` against the current state.

        Returns ``(probe_task, sketch_steps, all_predicates)``; the
        probe task starts at the current state and carries no evaluator.
        Same grammar and parser as ``evaluate_option_plan`` /
        ``refine_plan_sketch`` (``~ [w]`` search regions included).
        """
        # pylint: disable=import-outside-toplevel
        from predicators.agent_sdk import bilevel_sketch
        from predicators.agent_sdk.tools import _load_ground_sampler_fns
        from predicators.settings import CFG

        # pylint: enable=import-outside-toplevel
        ctx = self._ctx
        cur = self._require_state()
        assert self._base_task is not None
        probe_task = dataclasses.replace(self._base_task,
                                         init=cur,
                                         evaluator=None)

        all_options = ctx.options | ctx.iteration_proposals.proposed_options
        all_predicates = (ctx.predicates
                          | ctx.iteration_proposals.proposed_predicates)
        types = set(ctx.types)
        for opt in all_options:
            types.update(opt.types)
        for pred in all_predicates:
            types.update(pred.types)
        types.update(o.type for o in cur)

        gs_fns, gs_err = _load_ground_sampler_fns(ctx)
        if gs_err is not None:
            raise ValueError(gs_err)
        sketch_steps = bilevel_sketch.parse_sketch_from_text(
            plan_text,
            probe_task,
            predicates=all_predicates,
            options=all_options,
            types=types,
            parse_continuous_params=True,
            strict=True,
            parse_ground_samplers=CFG.agent_bilevel_ground_samplers,
            ground_sampler_fns=gs_fns or None)
        if not sketch_steps:
            raise ValueError(
                "Parsed empty plan. Each line must be "
                "`Option(obj:type, ...)[params]` with a known option, typed "
                "object refs, and exact params in `[]`.")
        return probe_task, sketch_steps, all_predicates

    def run(self, plan_text: str, render: bool = True) -> ProbeResult:
        """Execute an option plan from the current state.

        ``plan_text`` uses the same grammar as ``evaluate_option_plan``:
        one option per line, ``Option(obj:type, ...)[params]`` with
        exact continuous params (``[]`` for none); ``-> {atoms}``
        subgoals are accepted but optional. Advances the current state
        to the rollout's final state (``restore`` a snapshot to rewind).
        Like ``evaluate_option_plan``, each step's post-state is
        rendered to a saved image whose path lands in the step report;
        pass ``render=False`` inside tight sweep loops to skip that.
        Exploratory only: results are never captured and carry no
        evaluator verdict.
        """
        # pylint: disable-next=import-outside-toplevel
        import numpy as np

        # pylint: disable-next=import-outside-toplevel
        from predicators.agent_sdk import bilevel_sketch
        # pylint: disable-next=import-outside-toplevel
        from predicators.agent_sdk.tools import _render_scene_image
        ctx = self._ctx
        ctx.test_call_id += 1
        probe_task, sketch_steps, all_predicates = \
            self._parse_sketch(plan_text)
        grounded: List[Any] = []
        for st in sketch_steps:
            params = (st.initial_params if st.initial_params is not None else
                      np.array([], dtype=np.float32))
            grounded.append(
                st.option.ground(list(st.objects),
                                 np.asarray(params, dtype=np.float32)))

        report_preds = ctx.predicates
        step_dicts: List[Dict[str, Any]] = []

        def _on_step(i: int, outcome: Any) -> None:
            sig = _fmt_option(outcome.option)
            added: List[str] = []
            deleted: List[str] = []
            failure = outcome.failure_reason
            if not outcome.initiable:
                failure = failure or "not initiable"
            if outcome.post_state is not None:
                before = utils.abstract(outcome.pre_state, report_preds)
                after = utils.abstract(outcome.post_state, report_preds)
                added = [str(a) for a in sorted(after - before)]
                deleted = [str(a) for a in sorted(before - after)]
            # Same per-step audit image evaluate_option_plan saves; the
            # env already sits at the post-step state here.
            img = _render_scene_image(
                ctx,
                f"probe_step_{i}_{outcome.option.name}") if render else None
            step_dicts.append({
                "option": sig,
                "num_actions": outcome.num_actions,
                "failure": failure,
                "added": added,
                "deleted": deleted,
                "image": img.get("saved_path") if img else None,
            })

        result = bilevel_sketch.execute_plan_forward(probe_task,
                                                     grounded,
                                                     ctx.option_model,
                                                     predicates=all_predicates,
                                                     sketch=sketch_steps,
                                                     on_step=_on_step,
                                                     stop_on_failure=True)
        self._state = result.final_state
        self._pristine = False
        final_atoms = [
            str(a)
            for a in sorted(utils.abstract(result.final_state, report_preds))
        ]
        return ProbeResult(step_dicts, result.goal_reached, final_atoms,
                           result.final_state)

    def refine(self,
               sketch_text: str,
               timeout: float = 60.0,
               max_samples_per_step: Optional[int] = None,
               require_goal: bool = False,
               require_solved: bool = False) -> "ProbeRefineResult":
        """Backtracking parameter search for a sketch FROM THE CURRENT STATE.

        Same grammar and search core as ``refine_plan_sketch``, but
        composable: refine a plan *suffix* from a snapshot where the
        prefix already executed, so the search budget goes to the step
        that matters instead of re-descending through the whole plan.
        Annotate each step's ``-> {subgoals}`` - success means every
        step established its annotation (set ``require_goal=True`` to
        also demand the task goal at the last step). Each step's
        ``[params]`` seed the search (tried first, then sampled around);
        add ``~ [w]`` half-width regions to confine the sampling. Does
        not advance the current state. Returns best-found params (also
        on TIMEOUT - the refined prefix is reported as far as it got)
        plus per-step sample counts and the deepest near-miss.
        Exploratory only: nothing is captured.

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
        # pylint: disable=import-outside-toplevel
        import numpy as np

        from predicators.agent_sdk import bilevel_sketch
        from predicators.agent_sdk.tools import make_solved_check
        from predicators.settings import CFG

        # pylint: enable=import-outside-toplevel
        ctx = self._ctx
        probe_task, sketch_steps, all_predicates = \
            self._parse_sketch(sketch_text)
        solved_check = None
        gate_ran = [False]
        if require_solved:
            require_goal = True
            if not self._pristine:
                raise ValueError(
                    "require_solved needs the task's unmodified initial "
                    "state (call reset() with no mods and refine before "
                    "any run()): the evaluator's rules reference the true "
                    "init, so a verdict from a modified or advanced state "
                    "would be silently wrong.")
            assert self._base_task is not None
            evaluator = self._base_task.evaluator
            if evaluator is None:
                raise ValueError("require_solved: this task defines no "
                                 "task evaluator.")
            # Same gate (and therefore same accept policy: coarse and
            # evaluator errors never block, non-terminated never blocks)
            # as refine_plan_sketch, so identical params can't get
            # contradictory verdicts across the two surfaces.
            inner_check = make_solved_check(
                evaluator, getattr(ctx.option_model, "sim_env", None))

            def solved_check(states: List[State], labels: List[Any],
                             coarse: bool) -> Tuple[bool, str]:
                # Track whether any non-coarse verdict actually ran, so
                # a SUCCESS whose gate was silently skipped (an option
                # model without per-step trajectories) is reported as
                # ungated instead of implying certification.
                if not coarse:
                    gate_ran[0] = True
                return inner_check(states, labels, coarse)

        if max_samples_per_step is None:
            max_samples_per_step = CFG.agent_bilevel_max_samples_per_step
        self._refine_calls += 1
        # Deterministic but distinct from refine_plan_sketch's
        # CFG.seed + attempt streams and from other probe instances, so
        # "try a different random search" does not replay failed draws.
        rng = np.random.default_rng(CFG.seed + 100003 *
                                    (self._instance_id + 1) +
                                    self._refine_calls)
        step_samples: List[int] = [0] * len(sketch_steps)
        termination_reason: List[str] = []
        deepest_failure: List[Any] = []
        refined_plan, success, total_samples = bilevel_sketch.refine_sketch(
            probe_task,
            sketch_steps,
            ctx.option_model,
            predicates=all_predicates,
            timeout=timeout,
            rng=rng,
            max_samples_per_step=max_samples_per_step,
            check_subgoals=True,
            check_final_goal=require_goal,
            run_id="probe",
            step_samples_cumulative=step_samples,
            termination_reason=termination_reason,
            deepest_failure_holder=deepest_failure,
            parameterized_samplers=ctx.parameterized_samplers or None,
            solved_check=solved_check)
        reason = termination_reason[0] if termination_reason else (
            "success" if success else "failure")
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
        if deepest_failure:
            df = deepest_failure[0]
            near_miss = {
                "step_idx": df.step_idx,
                "option": _fmt_option(df.option),
                "reason": df.fail_reason,
            }
        note = ""
        if require_solved and not gate_ran[0]:
            note = ("require_solved was requested but no per-step "
                    "trajectories were available, so the evaluator gate "
                    "never ran - this result is NOT certified.")
        return ProbeRefineResult(success, reason, total_samples, step_samples,
                                 plan_lines, near_miss, note)

    # ── Internals ────────────────────────────────────────────────

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


def build_probe_namespace(ctx: Any) -> Dict[str, Any]:
    """The persistent ``explore_python`` namespace for solve-phase sessions.

    Deliberately small: the probe facade, numpy, and nothing else - the
    single-verdict-surface principle. The agent gets a ready ``sim``
    plus the class for extra independent instances.
    """
    # pylint: disable-next=import-outside-toplevel
    import numpy as np
    return {"sim": ProbeSim(ctx), "ProbeSim": lambda: ProbeSim(ctx), "np": np}
