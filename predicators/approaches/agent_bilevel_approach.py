"""Agent bilevel approach: agent produces plan sketch, search refines params.

The agent generates a plan sketch — a sequence of parameterized skills with
object bindings but without continuous parameters, plus optional subgoal
atoms after each step.  A backtracking search then samples continuous
parameters and validates each step via the option model.

Example command::

    python predicators/main.py --env pybullet_domino \
        --approach agent_bilevel --seed 0 \
        --num_train_tasks 1 --num_test_tasks 1 \
        --num_online_learning_cycles 1 --explorer agent_plan
"""
import logging
import time
from typing import Any, Callable, List, Optional, Sequence, Set, Tuple

import numpy as np

from predicators import utils
from predicators.agent_sdk import bilevel_sketch
from predicators.agent_sdk.bilevel_sketch import SketchStep as _SketchStep
from predicators.approaches import ApproachFailure
from predicators.approaches.agent_planner_approach import AgentPlannerApproach
from predicators.execution_monitoring.subgoal_annotations_monitor import \
    SubgoalExecutionStatus
from predicators.settings import CFG
from predicators.structs import Action, GroundAtom, Object, \
    ParameterizedOption, Predicate, State, Task, _Option


class AgentBilevelApproach(AgentPlannerApproach):
    """Bilevel planning: agent proposes discrete skeleton, search refines
    continuous parameters.

    Extends AgentPlannerApproach — reuses agent session, tools,
    trajectory management, exploration, save/load.  Overrides solving to
    separate discrete planning from continuous refinement.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if CFG.agent_bilevel_max_execution_replans > 0 and \
                CFG.execution_monitor != "subgoal_annotations":
            raise ValueError(
                "agent_bilevel_max_execution_replans > 0 requires "
                "--execution_monitor subgoal_annotations (got "
                f"{CFG.execution_monitor!r}): divergence detection lives "
                "in the execution monitor, so without it test execution "
                "is silently open-loop.")
        # Live status of the currently executing annotated plan, exported
        # to the subgoal_annotations execution monitor. None whenever no
        # monitored plan is active (exploration, replanning disabled).
        self._exec_status: Optional[SubgoalExecutionStatus] = None
        # Per-episode replan budget, refreshed by reset_for_new_episode.
        self._exec_replans_left = 0

    @classmethod
    def get_name(cls) -> str:
        return "agent_bilevel"

    # ------------------------------------------------------------------ #
    # Execution monitoring (closed-loop test execution)
    # ------------------------------------------------------------------ #

    def reset_for_new_episode(self) -> None:
        super().reset_for_new_episode()
        self._exec_status = None
        self._exec_replans_left = CFG.agent_bilevel_max_execution_replans

    def get_execution_monitoring_info(self) -> List[Any]:
        if self._exec_status is None:
            return []
        return [self._exec_status]

    # ------------------------------------------------------------------ #
    # Agent session hooks
    # ------------------------------------------------------------------ #

    def _get_synthesis_tool_names(self) -> Optional[List[str]]:
        """No synthesis phase in this approach — declare an empty set."""
        return []

    # ------------------------------------------------------------------ #
    # System prompt (simplified — no parameter tuning workflow)
    # ------------------------------------------------------------------ #

    def _get_agent_system_prompt(self) -> str:
        return (
            "You are a planning agent. You observe task environments through "
            "inspection tools and generate plan sketches to achieve goals. "
            "You have access to read-only tools to inspect predicates, "
            "options, trajectories, and training tasks.\n\n"
            "Your job is to produce a DISCRETE plan sketch: the sequence of "
            "skills (parameterized options) and their object arguments, plus "
            "optional subgoal atoms that should hold after each step. You do "
            "NOT need to specify continuous parameters — those will be found "
            "automatically by a search procedure.\n\n"
            "Some effects may not be immediate — if an action triggers a "
            "delayed process (e.g. gradual accumulation, propagation "
            "through contacting objects, a sensor catching up to an "
            "actuator), insert a Wait after it so the effect has time "
            "to occur before the next action.\n\n"
            "## Subgoal Annotations\n"
            "After each step, annotate which predicate atoms should hold "
            "after that step succeeds. Use the format:\n"
            "  OptionName(obj1:type1, obj2:type2) -> {Pred(obj1:type1), "
            "Pred2(obj1:type1, obj2:type2)}\n"
            "Always use typed references (obj:type) in subgoal atoms.\n"
            "Annotate EVERY step whose effect the predicates can express. "
            "Annotations are not just search hints: refinement validates "
            "each annotated step, and at execution time they are checked "
            "against the real state, so a step that diverges can be "
            "detected and replanned instead of silently dooming the rest "
            "of the plan. Prefer atoms that NEWLY hold (or stop holding) "
            "because of the step — atoms that were already true beforehand "
            "cannot reveal divergence. A step you cannot annotate is a "
            "blind spot for both search and recovery.\n"
            "For Wait steps, the annotation also specifies exactly when the "
            "Wait should terminate. Use `NOT Pred(...)` for atoms that should "
            "become false (e.g. `Wait(robot:robot) -> "
            "{Ready(widget:widget)}`).")

    # ------------------------------------------------------------------ #
    # Solve prompt (no continuous params, subgoal format)
    # ------------------------------------------------------------------ #

    def _build_solve_prompt(self, task: Task) -> str:
        """Build prompt asking for a plan sketch without continuous params."""
        return bilevel_sketch.build_solve_prompt(
            task,
            all_predicates=self._get_all_predicates(),
            all_options=self._get_all_options(),
            trajectory_summary=self._build_trajectory_summary(),
            tool_names=self._get_solve_tool_names(),
        )

    # ------------------------------------------------------------------ #
    # Solving
    # ------------------------------------------------------------------ #

    def _solve(self, task: Task, timeout: int) -> Callable[[State], Action]:
        replan_policy = self._maybe_replan_from_divergence(task, timeout)
        if replan_policy is not None:
            return replan_policy
        max_sketch_retries = CFG.agent_bilevel_max_retries
        max_refine_retries = CFG.agent_bilevel_max_refine_retries
        self._sync_tool_context()
        self._tool_context.current_task = task
        start = time.perf_counter()

        for sketch_attempt in range(max_sketch_retries):
            if timeout - (time.perf_counter() - start) <= 0:
                break
            try:
                sketch = self._query_agent_for_plan_sketch(task)
            except Exception as e:  # pylint: disable=broad-except
                logging.warning("Sketch query failed (attempt %d): %s",
                                sketch_attempt, e)
                continue

            sketch_lines = []
            for i, s in enumerate(sketch):
                objs = ", ".join(o.name for o in s.objects)
                line = f"  {i}: {s.option.name}({objs})"
                if s.subgoal_atoms:
                    atoms = ", ".join(str(a) for a in s.subgoal_atoms)
                    line += f" -> {{{atoms}}}"
                sketch_lines.append(line)
            logging.info("[%s] Sketch (attempt %d):\n%s", self._run_id,
                         sketch_attempt, "\n".join(sketch_lines))

            # Resample continuous params with a fresh seed before paying
            # for another agent query: a sketch that refines but fails
            # forward validation is a continuous-params problem, not a
            # wrong skeleton, and re-querying rarely changes the skeleton
            # while always costing an LLM call.
            for refine_attempt in range(max_refine_retries):
                remaining = timeout - (time.perf_counter() - start)
                if remaining <= 0:
                    break
                # Flatten the two loop indices so every (sketch, refine)
                # pair draws a unique seed in _refine_sketch.
                seed_offset = (sketch_attempt * max_refine_retries +
                               refine_attempt)
                plan, success = self._refine_sketch(task,
                                                    sketch,
                                                    remaining,
                                                    attempt=seed_offset)
                if not success:
                    logging.info(
                        f"Refinement failed (sketch "
                        f"{sketch_attempt}, refine {refine_attempt}), "
                        f"{len(sketch)} steps.")
                    continue

                plan_strs = []
                for i, o in enumerate(plan):
                    obj_s = ", ".join(obj.name for obj in o.objects)
                    par_s = ", ".join(f"{p:.4f}" for p in o.params)
                    plan_strs.append(f"  {i}: {o.name}({obj_s})"
                                     f"[{par_s}]")
                plan_str = "\n".join(plan_strs)
                logging.info(f"[{self._run_id}] Refinement succeeded (sketch "
                             f"{sketch_attempt}, refine {refine_attempt}), "
                             f"{len(plan)} steps:\n{plan_str}")

                # Forward validation: verify the plan works in
                # continuous execution (no state resets between steps).
                # Catches refinement/execution drift from option-model
                # state-reset noise (see pybullet_env.py:506 warning).
                # Pass the original sketch so per-step subgoal divergence
                # is logged with the specific atom that went missing.
                assert self._option_model is not None, \
                    "agent_bilevel requires a simulator " \
                    "(agent_planner_use_simulator=True)."
                ok, reason = bilevel_sketch.validate_plan_forward(
                    task,
                    plan,
                    self._option_model,
                    predicates=self._get_all_predicates(),
                    sketch=sketch,
                    run_id=self._run_id,
                )
                if ok:
                    return self._plan_to_policy(plan, sketch=sketch)
                logging.info(f"[{self._run_id}] Forward validation failed "
                             f"(sketch {sketch_attempt}, refine "
                             f"{refine_attempt}): {reason}")
                # Fall through to the next seed on the same sketch.

        raise ApproachFailure(
            f"Bilevel solve failed after {max_sketch_retries} sketches.")

    # ------------------------------------------------------------------ #
    # Plan sketch extraction
    # ------------------------------------------------------------------ #

    def _query_agent_for_plan_sketch(self, task: Task) -> List[_SketchStep]:
        """Query agent for a plan sketch and parse it."""
        sketch_file = CFG.agent_bilevel_plan_sketch_file
        if sketch_file:
            filepath = utils.get_path_to_predicators_root() + \
                f"/scripts/{CFG.agent_bilevel_plan_sketch_dir}/{sketch_file}"
            with open(filepath, "r", encoding="utf-8") as f:
                plan_text = f.read().strip()
            logging.info("Loaded plan sketch from file: %s", sketch_file)
        else:
            prompt = self._build_solve_prompt(task)
            responses = self._query_agent_sync(prompt, kind="test")
            plan_text = self._extract_option_plan_text(responses)

        if not plan_text:
            raise ApproachFailure("Agent returned empty plan text.")

        sketch = bilevel_sketch.parse_sketch_from_text(
            plan_text,
            task,
            predicates=self._get_all_predicates(),
            options=self._get_all_options(),
            types=self._types,
        )

        if not sketch:
            option_names = sorted(o.name for o in self._get_all_options())
            raise ApproachFailure(f"Parsed empty plan sketch from agent.\n"
                                  f"  Plan text:\n{plan_text}\n"
                                  f"  Available option names: {option_names}")

        logging.info(f"[{self._run_id}] Agent produced sketch with "
                     f"{len(sketch)} steps, "
                     f"{sum(1 for s in sketch if s.subgoal_atoms)} "
                     f"with subgoals.")
        return sketch

    # ------------------------------------------------------------------ #
    # Backtracking refinement
    # ------------------------------------------------------------------ #

    def _refine_sketch(
        self,
        task: Task,
        sketch: List[_SketchStep],
        timeout: float,
        attempt: int = 0,
    ) -> Tuple[List[_Option], bool]:
        """Backtracking search over continuous parameters for a plan sketch.

        Returns ``(plan, success)``.  On success, ``plan`` is a list of
        grounded options that achieves the task goal.  On failure,
        ``plan`` is the longest partial refinement found.

        ``attempt`` perturbs the RNG so retries explore different
        samples — without it, refinement is deterministic in
        ``CFG.seed`` and a forward-validation failure would loop on
        the identical plan.

        Delegates to ``bilevel_sketch.refine_sketch``. The task is
        first passed through :meth:`_attach_initial_latent` so that
        partially-observable approaches can seed
        ``task.init.latent`` with the initial latent block; the default
        implementation returns ``task`` unchanged.
        """
        task = self._attach_initial_latent(task)
        assert self._option_model is not None, \
            "agent_bilevel requires a simulator " \
            "(agent_planner_use_simulator=True)."
        plan, success, _ = bilevel_sketch.refine_sketch(
            task,
            sketch,
            self._option_model,
            predicates=self._get_all_predicates(),
            timeout=timeout,
            rng=np.random.default_rng(CFG.seed + attempt),
            max_samples_per_step=CFG.agent_bilevel_max_samples_per_step,
            check_subgoals=CFG.agent_bilevel_check_subgoals,
            log_state=CFG.agent_bilevel_log_state,
            run_id=self._run_id,
            option_samplers=self._get_all_samplers(),
        )
        return plan, success

    def _attach_initial_latent(self, task: Task) -> Task:
        """Hook for partial-observability approaches to seed the latent.

        Subclasses that thread a ``latent`` state block through the
        simulator (e.g. ``AgentPOSimPredicateInventionApproach``)
        override this to attach an initial latent to
        ``task.init.latent`` before refinement begins. The default
        returns ``task`` unchanged — fully-observable approaches need do
        nothing.
        """
        return task

    def _sample_params(self, option: ParameterizedOption, _state: State,
                       rng: np.random.Generator) -> np.ndarray:
        """Sample continuous parameters for an option."""
        return bilevel_sketch.sample_params(option, rng)

    def _parse_subgoal_annotations(
        self,
        text: str,
        predicates: Set[Predicate],
        objects: Sequence[Object],
    ) -> List[Optional[Tuple[Set[GroundAtom], Set[GroundAtom]]]]:
        """Shim over ``bilevel_sketch.parse_subgoal_annotations``."""
        option_names = {o.name for o in self._get_all_options()}
        return bilevel_sketch.parse_subgoal_annotations(
            text, predicates, objects, option_names)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _maybe_replan_from_divergence(
            self, task: Task,
            timeout: int) -> Optional[Callable[[State], Action]]:
        """Handle a mid-episode re-solve triggered by the subgoal_annotations
        execution monitor.

        CogMan calls solve() identically at episode start and on a
        monitor-triggered replan; the two are distinguished by
        ``_exec_status``, which is non-None only while a monitored plan
        is executing (``reset_for_new_episode`` clears it at episode
        start). On a replan, ``task.init`` is the real state in which
        the just-finished step's annotation failed. Divergence is
        usually a continuous-execution problem (a sampled parameter
        whose real outcome differed from the option-model rollout), not
        a wrong skeleton, so we first try to resume a suffix of the
        executed sketch (cheap — no agent query; see
        :meth:`_replan_suffix`). Returns None to fall through to a
        fresh agent sketch, and raises ApproachFailure once the
        per-episode replan budget is exhausted so the episode fails
        fast instead of burning the horizon open-loop.
        """
        status = self._exec_status
        if status is None or status.steps_initiated == 0:
            return None
        self._exec_status = None
        failed_idx = status.steps_initiated - 1
        steps = list(status.sketch)
        failed_name = steps[failed_idx].option.name
        if self._exec_replans_left <= 0:
            raise ApproachFailure(
                f"Subgoal divergence after step {failed_idx} "
                f"({failed_name}). No execution replans left.")
        self._exec_replans_left -= 1
        logging.info(
            "Subgoal divergence after step %d (%s). Replanning from the "
            "current state (%d execution replans left).", failed_idx,
            failed_name, self._exec_replans_left)
        policy = self._replan_suffix(task.init, task, steps, failed_idx,
                                     timeout)
        if policy is None:
            # No suffix of the executed skeleton is refinable from here —
            # fall through to pay for a fresh agent sketch.
            logging.info("Suffix replan failed; querying the agent for a "
                         "fresh sketch.")
        return policy

    def _plan_to_policy(
        self,
        plan: List[_Option],
        sketch: Optional[List[_SketchStep]] = None,
    ) -> Callable[[State], Action]:
        """Wrap a grounded option plan into a step-by-step policy.

        With ``CFG.agent_bilevel_max_execution_replans > 0`` and a full
        per-step sketch, the policy also publishes a live
        ``SubgoalExecutionStatus`` (via
        ``get_execution_monitoring_info``) that the subgoal_annotations
        execution monitor reads to check, at each option boundary, that
        the just-finished step's annotation holds in the REAL state. On
        divergence the monitor makes CogMan re-invoke solve(), which
        lands in :meth:`_maybe_replan_from_divergence`.
        """
        predicates = self._get_all_predicates()

        def _abstract(s: State) -> Set[GroundAtom]:
            return utils.abstract(s, predicates)

        monitored = (CFG.agent_bilevel_max_execution_replans > 0
                     and sketch is not None and len(sketch) == len(plan))

        queue = list(plan)
        total = len(queue)
        status: Optional[SubgoalExecutionStatus] = None
        if monitored:
            assert sketch is not None
            status = SubgoalExecutionStatus(sketch=list(sketch))
            self._exec_status = status

        def _option_policy(state: State) -> _Option:
            del state  # unused
            if not queue:
                logging.info("Option plan exhausted after %d options.", total)
                raise utils.OptionExecutionFailure("Option plan exhausted!")
            option = queue.pop(0)
            num_done = total - len(queue)
            if status is not None:
                status.steps_initiated = num_done
                status.current_option = option
            next_option = None if not queue else queue[0].simple_str()
            logging.info("Executing option %d/%d: %s (remaining=%d, next=%s)",
                         num_done, total, option.simple_str(), len(queue),
                         next_option)
            return option

        inner = utils.option_policy_to_policy(_option_policy,
                                              abstract_function=_abstract)

        def _policy(s: State) -> Action:
            try:
                return inner(s)
            except utils.OptionExecutionFailure as e:
                raise ApproachFailure(e.args[0], e.info)

        return _policy

    def _replan_suffix(
        self,
        state: State,
        task: Task,
        sketch: List[_SketchStep],
        failed_idx: int,
        timeout: int,
    ) -> Optional[Callable[[State], Action]]:
        """Cheap-first recovery: re-refine a suffix of the current sketch.

        Divergence is usually a continuous-execution problem (a sampled
        parameter whose real outcome differed from the option-model
        rollout), not a wrong skeleton, so before paying for a fresh
        agent sketch we retry the one we have. Candidate resume points
        run from the failed step backward to just after the latest
        earlier annotated step whose subgoals still hold in the current
        state. The holds-check only bounds the walk-back — annotations
        are optional and can hold coincidentally (e.g. a final
        SwitchOff's {Off} atom holds before the switch was ever touched)
        — so every candidate suffix must still refine AND forward-
        validate from the current state before we trust it. Returns None
        when no suffix candidate validates.
        """
        assert self._option_model is not None
        sub_task = Task(state, task.goal)
        resume_floor = 0
        for j in range(failed_idx - 1, -1, -1):
            step = sketch[j]
            if step.subgoal_atoms is None and step.subgoal_neg_atoms is None:
                continue
            pos_ok = all(a.holds(state) for a in (step.subgoal_atoms or set()))
            neg_ok = not any(
                a.holds(state) for a in (step.subgoal_neg_atoms or set()))
            if pos_ok and neg_ok:
                resume_floor = j + 1
                break
        start = time.perf_counter()
        for j in range(failed_idx, resume_floor - 1, -1):
            remaining = timeout - (time.perf_counter() - start)
            if remaining <= 0:
                break
            suffix = list(sketch[j:])
            plan, success = self._refine_sketch(sub_task,
                                                suffix,
                                                remaining,
                                                attempt=j)
            if not success:
                logging.info(
                    "Suffix replan: refinement failed resuming at "
                    "step %d.", j)
                continue
            ok, reason = bilevel_sketch.validate_plan_forward(
                sub_task,
                plan,
                self._option_model,
                predicates=self._get_all_predicates(),
                sketch=suffix,
                run_id=self._run_id,
            )
            if ok:
                logging.info(
                    "Suffix replan: resuming executed sketch at step %d "
                    "(%d steps).", j, len(plan))
                return self._plan_to_policy(plan, sketch=suffix)
            logging.info(
                "Suffix replan: forward validation failed resuming at "
                "step %d: %s", j, reason)
        return None
