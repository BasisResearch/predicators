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
from typing import Callable, List, Optional, Sequence, Set, Tuple

import numpy as np

from predicators import utils
from predicators.agent_sdk import bilevel_sketch
from predicators.agent_sdk.bilevel_sketch import SketchStep as _SketchStep
from predicators.approaches import ApproachFailure
from predicators.approaches.agent_planner_approach import AgentPlannerApproach
from predicators.planning import run_backtracking_refinement
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

    @classmethod
    def get_name(cls) -> str:
        return "agent_bilevel"

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
            "delayed process (e.g. water filling, dominoes cascading, "
            "heating), insert a Wait after it so the effect has time to "
            "occur before the next action.\n\n"
            "## Subgoal Annotations\n"
            "After each step you can annotate which predicate atoms should "
            "hold after that step succeeds. This helps the search procedure "
            "verify progress. Use the format:\n"
            "  OptionName(obj1:type1, obj2:type2) -> {Pred(obj1:type1), "
            "Pred2(obj1:type1, obj2:type2)}\n"
            "Always use typed references (obj:type) in subgoal atoms.\n"
            "Subgoal annotations are optional but improve search efficiency.\n"
            "For Wait steps, the annotation also specifies exactly when the "
            "Wait should terminate. Use `NOT Pred(...)` for atoms that should "
            "become false (e.g. `Wait(robot:Robot) -> "
            "{Boiled(water:water_type)}`).")

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
            tool_names=self._get_agent_tool_names(),
        )

    # ------------------------------------------------------------------ #
    # Solving
    # ------------------------------------------------------------------ #

    def _solve(self, task: Task, timeout: int) -> Callable[[State], Action]:
        max_retries = CFG.agent_bilevel_max_retries
        self._sync_tool_context()
        self._tool_context.current_task = task
        start = time.perf_counter()

        for attempt in range(max_retries):
            remaining = timeout - (time.perf_counter() - start)
            if remaining <= 0:
                break
            try:
                sketch = self._query_agent_for_plan_sketch(task)
            except Exception as e:  # pylint: disable=broad-except
                logging.warning("Sketch query failed (attempt %d): %s",
                                attempt, e)
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
                         attempt, "\n".join(sketch_lines))

            plan, success = self._refine_sketch(task, sketch, remaining)
            if success:
                plan_strs = []
                for i, o in enumerate(plan):
                    obj_s = ", ".join(obj.name for obj in o.objects)
                    par_s = ", ".join(f"{p:.4f}" for p in o.params)
                    plan_strs.append(f"  {i}: {o.name}({obj_s})"
                                     f"[{par_s}]")
                plan_str = "\n".join(plan_strs)
                logging.info(
                    f"[{self._run_id}] Refinement succeeded "
                    f"(attempt {attempt}), {len(plan)} steps:\n{plan_str}")

                # Forward validation: verify the plan works in
                # continuous execution (no state resets between steps).
                # if self._validate_plan_forward(task, plan):
                return self._plan_to_policy(plan)
                # logging.info("Forward validation failed; retrying.")
            logging.info(f"Refinement failed (attempt {attempt}), "
                         f"{len(sketch)} steps.")

        raise ApproachFailure(
            f"Bilevel solve failed after {max_retries} attempts.")

    # ------------------------------------------------------------------ #
    # Plan sketch extraction
    # ------------------------------------------------------------------ #

    def _query_agent_for_plan_sketch(self, task: Task) -> List[_SketchStep]:
        """Query agent for a plan sketch and parse it."""
        sketch_file = CFG.agent_bilevel_plan_sketch_file
        if sketch_file:
            with open(sketch_file, "r", encoding="utf-8") as f:
                plan_text = f.read().strip()
            logging.info("Loaded plan sketch from file: %s", sketch_file)
        else:
            prompt = self._build_solve_prompt(task)
            responses = self._query_agent_sync(prompt)
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
    ) -> Tuple[List[_Option], bool]:
        """Backtracking search over continuous parameters for a plan sketch.

        Returns ``(plan, success)``.  On success, ``plan`` is a list of
        grounded options that achieves the task goal.  On failure,
        ``plan`` is the longest partial refinement found.

        Delegates to ``bilevel_sketch.refine_sketch``.
        """
        plan, success, _ = bilevel_sketch.refine_sketch(
            task,
            sketch,
            self._option_model,
            predicates=self._get_all_predicates(),
            timeout=timeout,
            rng=np.random.default_rng(CFG.seed),
            max_samples_per_step=CFG.agent_bilevel_max_samples_per_step,
            check_subgoals=CFG.agent_bilevel_check_subgoals,
            log_state=CFG.agent_bilevel_log_state,
            run_id=self._run_id,
        )
        return plan, success

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
    # Forward validation
    # ------------------------------------------------------------------ #

    def _validate_plan_forward(
        self,
        task: Task,
        plan: List[_Option],
    ) -> bool:
        """Re-execute the plan continuously in the option model.

        Runs all options sequentially so that state carries forward
        naturally — matching how the real env will execute.

        Returns True if the plan reaches the goal, False otherwise.
        """
        n = len(plan)
        if n == 0:
            return task.goal_holds(task.init)

        def sample_fn(i: int, _s: State, _r: np.random.Generator) -> _Option:
            return plan[i]

        def validate_fn(i: int, _s: State, _o: _Option, post: State,
                        _n: int) -> Tuple[bool, str]:
            if i == n - 1 and not task.goal_holds(post):
                return False, "goal not reached"
            return True, ""

        _, success, _ = run_backtracking_refinement(
            init_state=task.init,
            option_model=self._option_model,
            n_steps=n,
            max_tries=[1] * n,
            sample_fn=sample_fn,
            validate_fn=validate_fn,
            rng=np.random.default_rng(0),
            timeout=float('inf'),
        )
        return success

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _plan_to_policy(
        self,
        plan: List[_Option],
    ) -> Callable[[State], Action]:
        """Wrap a grounded option plan into a step-by-step policy."""
        predicates = self._get_all_predicates()
        policy = utils.option_plan_to_policy(
            plan, abstract_function=lambda s: utils.abstract(s, predicates))

        def _policy(s: State) -> Action:
            try:
                return policy(s)
            except utils.OptionExecutionFailure as e:
                raise ApproachFailure(e.args[0], e.info)

        return _policy
