"""Agent bilevel approach: agent produces plan sketch, search refines params.

The agent generates a plan sketch — a sequence of parameterized skills with
object bindings but without continuous parameters, plus optional subgoal
atoms after each step.  A backtracking search then samples continuous
parameters and validates each step via the option model.

Example command::

    python predicators/main.py --env pybullet_domino \
        --approach agent_bilevel --seed 0 \
        --num_train_tasks 1 --num_test_tasks 1 \
        --num_online_learning_cycles 1 --explorer agent
"""
import dataclasses
import logging
import re
import time
from typing import Any, Callable, List, Optional, Sequence, Set, Tuple, cast

import numpy as np

from predicators import utils
from predicators.approaches import ApproachFailure
from predicators.approaches.agent_planner_approach import AgentPlannerApproach
from predicators.settings import CFG
from predicators.structs import Action, GroundAtom, Object, \
    ParameterizedOption, Predicate, State, Task, _Option


@dataclasses.dataclass
class _SketchStep:
    """One step in an agent-produced plan sketch."""
    option: ParameterizedOption
    objects: Sequence[Object]
    subgoal_atoms: Optional[Set[GroundAtom]]  # None = no subgoal constraint
    # Atoms that must be FALSE after this step.
    subgoal_neg_atoms: Optional[Set[GroundAtom]] = None


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
        init_state = task.init
        objects = list(init_state)

        # Objects
        obj_strs = []
        for obj in sorted(objects, key=lambda o: o.name):
            obj_strs.append(f"  {obj.name}: {obj.type.name}")

        # Goal
        goal_strs = [str(a) for a in sorted(task.goal, key=str)]

        # Options (show params_space info so agent understands what's tunable)
        option_strs = []
        for opt in sorted(self._get_all_options(), key=lambda o: o.name):
            type_sig = ", ".join(t.name for t in opt.types)
            params_dim = opt.params_space.shape[0]
            if params_dim > 0:
                low = opt.params_space.low.tolist()
                high = opt.params_space.high.tolist()
                if opt.params_description:
                    desc = ", ".join(opt.params_description)
                    param_info = (f"  [auto-searched params: {desc}, "
                                  f"range {low} to {high}]")
                else:
                    param_info = (f"  [auto-searched: {params_dim}d, "
                                  f"range {low} to {high}]")
            else:
                param_info = ""
            option_strs.append(f"  {opt.name}({type_sig}){param_info}")

        # Current atoms
        atoms = utils.abstract(init_state, self._get_all_predicates())
        atom_strs = [str(a) for a in sorted(atoms, key=str)]

        # Trajectory summary
        traj_summary = self._build_trajectory_summary()

        # State features
        state_str = init_state.dict_str(indent=2)

        # Available tools
        tool_names = self._get_agent_tool_names()
        tools_str = ""
        if tool_names:
            tool_list = "\n".join(f"  - {t}" for t in tool_names)
            tools_str = f"\n## Available Tools\n{tool_list}\n"

        # Natural language goal
        goal_nl_section = ""
        if task.goal_nl:
            goal_nl_section = f"\n## Goal Description\n{task.goal_nl}\n"

        # Available predicates for subgoal annotations
        pred_strs = []
        for pred in sorted(self._get_all_predicates(), key=lambda p: p.name):
            type_sig = ", ".join(t.name for t in pred.types)
            pred_strs.append(f"  {pred.name}({type_sig})")

        prompt = f"""You are solving a task. \
Generate a plan sketch to achieve the goal.
{goal_nl_section}
## Goal Atoms
{chr(10).join(goal_strs)}

## Initial State Atoms
{chr(10).join(atom_strs)}

## Initial State Features
{state_str}

## Objects
{chr(10).join(obj_strs)}

## Available Options
{chr(10).join(option_strs)}

## Available Predicates (for subgoal annotations)
{chr(10).join(pred_strs)}
{traj_summary}{tools_str}
## Instructions
Use your available tools to inspect the environment before producing the plan.

Generate a plan SKETCH — the sequence of options with object arguments, but \
WITHOUT continuous parameters. Continuous parameters will be found \
automatically by a backtracking search procedure.

Optionally annotate subgoal atoms that should hold after each step. This \
helps the search verify progress. Use `-> {{atoms}}` after each step.

After any action whose desired subgoal depends on a delayed process (e.g. \
water filling, dominoes cascading, heating), insert a Wait action. For Wait \
steps, annotate with the atoms the process should produce — this tells the \
system exactly when the Wait should end rather than terminating on any \
incidental atom change. Use `NOT Pred(...)` for atoms that should become false.

Output the plan sketch with one option per line in this format:
  OptionName(obj1:type1, obj2:type2) -> \
{{Pred(obj1:type1), Pred2(obj1:type1, obj2:type2)}}
  Wait(robot:Robot) -> {{Boiled(water:water_type)}}
  Wait(robot:Robot) -> {{NOT Touching(a:block, b:block)}}

Always use typed references (obj:type) in both option arguments AND subgoal \
atoms. The `-> {{atoms}}` part is optional. If you omit it, the search will \
only check that the option executed successfully (non-zero actions).

Output ONLY the plan sketch lines at the end, after any analysis."""

        return prompt

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
        prompt = self._build_solve_prompt(task)
        responses = self._query_agent_sync(prompt)
        plan_text = self._extract_option_plan_text(responses)

        if not plan_text:
            n_responses = len(responses)
            types = [r.get("type") for r in responses]
            raise ApproachFailure(
                f"Agent returned empty plan text. "
                f"Got {n_responses} responses with types: {types}")

        cleaned_text = self._strip_code_fences(plan_text)

        # Phase 1: parse options + objects (no continuous params)
        objects = list(task.init)
        parsed = utils.parse_model_output_into_option_plan(
            cleaned_text,
            objects,
            self._types,
            self._get_all_options(),
            parse_continuous_params=False)

        if not parsed:
            option_names = sorted(o.name for o in self._get_all_options())
            raise ApproachFailure(f"Parsed empty plan sketch from agent.\n"
                                  f"  Plan text:\n{plan_text}\n"
                                  f"  Available option names: {option_names}")

        # Phase 2: parse subgoal annotations from raw text
        subgoals = self._parse_subgoal_annotations(cleaned_text,
                                                   self._get_all_predicates(),
                                                   objects)

        # Zip into sketch steps
        sketch = []
        for i, (option, objs, _) in enumerate(parsed):
            sg = subgoals[i] if i < len(subgoals) else None
            if sg is not None:
                pos, neg = sg
                sketch.append(
                    _SketchStep(option=option,
                                objects=objs,
                                subgoal_atoms=pos if pos else None,
                                subgoal_neg_atoms=neg if neg else None))
            else:
                sketch.append(
                    _SketchStep(option=option,
                                objects=objs,
                                subgoal_atoms=None))

        logging.info(f"[{self._run_id}] Agent produced sketch with "
                     f"{len(sketch)} steps, "
                     f"{sum(1 for s in sketch if s.subgoal_atoms)} "
                     f"with subgoals.")
        return sketch

    def _parse_subgoal_annotations(
        self,
        text: str,
        predicates: Set[Predicate],
        objects: Sequence[Object],
    ) -> List[Optional[Tuple[Set[GroundAtom], Set[GroundAtom]]]]:
        """Parse ``-> {Pred(...), NOT Pred(...)}`` annotations from plan text.

        Returns a list parallel to the option lines.  Entries are None
        for lines without annotations.  Each non-None entry is
        ``(positive_atoms, negative_atoms)``.
        """
        pred_map = {p.name: p for p in predicates}
        obj_map = {o.name: o for o in objects}

        # Regex: match -> { ... } after the option line
        subgoal_re = re.compile(r'->\s*\{([^}]*)\}')
        # Regex: match individual atoms, optionally prefixed with NOT
        atom_re = re.compile(r'(NOT\s+)?(\w+)\(([^)]*)\)')

        results: List[Optional[Tuple[Set[GroundAtom], Set[GroundAtom]]]] = []
        option_names = {o.name for o in self._get_all_options()}

        for line in text.split('\n'):
            stripped = line.strip()
            if not stripped:
                continue
            # Check if this line starts with a valid option name
            first_token = stripped.split('(')[0]
            if first_token not in option_names:
                continue

            # This is an option line — check for subgoal annotation
            sg_match = subgoal_re.search(stripped)
            if not sg_match:
                results.append(None)
                continue

            atoms_text = sg_match.group(1)
            pos_atoms: Set[GroundAtom] = set()
            neg_atoms: Set[GroundAtom] = set()
            for atom_match in atom_re.finditer(atoms_text):
                is_neg = atom_match.group(1) is not None
                pred_name = atom_match.group(2)
                # Handle both "obj" and "obj:type" formats
                obj_names = [
                    n.strip().split(':')[0]
                    for n in atom_match.group(3).split(',')
                ]

                if pred_name not in pred_map:
                    logging.warning(f"Unknown predicate in subgoal: "
                                    f"{pred_name}")
                    continue
                pred = pred_map[pred_name]
                try:
                    objs = [obj_map[n] for n in obj_names]
                except KeyError as e:
                    logging.warning(f"Unknown object in subgoal: {e}")
                    continue
                if len(objs) != len(pred.types):
                    logging.warning(
                        f"Arity mismatch for {pred_name}: expected "
                        f"{len(pred.types)}, got {len(objs)}")
                    continue
                atom = GroundAtom(pred, objs)
                if is_neg:
                    neg_atoms.add(atom)
                else:
                    pos_atoms.add(atom)

            if pos_atoms or neg_atoms:
                results.append((pos_atoms, neg_atoms))
            else:
                results.append(None)

        return results

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
        """
        if not sketch:
            return [], False

        rng = np.random.default_rng(CFG.seed)
        max_samples = CFG.agent_bilevel_max_samples_per_step
        check_subgoals = CFG.agent_bilevel_check_subgoals
        start_time = time.perf_counter()

        n = len(sketch)
        cur_idx = 0
        num_tries = [0] * n
        max_tries = [
            max_samples if step.option.params_space.shape[0] > 0 else 1
            for step in sketch
        ]
        plan: List[Optional[_Option]] = [None] * n
        traj: List[Optional[State]] = [task.init] + [None] * n

        total_samples = 0

        while cur_idx < n:
            elapsed = time.perf_counter() - start_time
            if elapsed > timeout:
                logging.info(
                    f"Sketch refinement timed out after {elapsed:.1f}s "
                    f"at step {cur_idx}/{n}, {total_samples} total samples.")
                return [p for p in plan if p is not None], False

            step = sketch[cur_idx]
            num_tries[cur_idx] += 1
            total_samples += 1
            step_name = (f"{step.option.name}"
                         f"({', '.join(o.name for o in step.objects)})")

            # Optionally log state before sampling
            cur_state = traj[cur_idx]
            assert cur_state is not None, f"traj[{cur_idx}] should not be None"

            if CFG.agent_bilevel_log_state:
                logging.debug(f"  State before {step_name}:\n"
                              f"{cur_state.pretty_str()}")

            # Sample continuous parameters and ground option
            params = self._sample_params(step.option, cur_state, rng)
            grounded = step.option.ground(step.objects, params)
            # Inject Wait target atoms from sketch annotations
            if grounded.name == "Wait":
                if step.subgoal_atoms is not None:
                    grounded.memory["wait_target_atoms"] = step.subgoal_atoms
                if step.subgoal_neg_atoms is not None:
                    grounded.memory["wait_target_neg_atoms"] = \
                        step.subgoal_neg_atoms
            plan[cur_idx] = grounded

            state = cur_state
            can_continue = False
            fail_reason = "not initiable"

            if grounded.initiable(state):
                try:
                    next_state, num_actions = \
                        self._option_model.get_next_state_and_num_actions(
                            state, grounded)
                except utils.EnvironmentFailure as e:
                    fail_reason = f"env failure: {e}"
                else:
                    if num_actions == 0:
                        model = self._option_model
                        fail_reason = (
                            getattr(  # type: ignore[attr-defined]
                                model, "last_execution_failure", None)
                            or "0 actions")
                    else:
                        traj[cur_idx + 1] = next_state
                        # Check subgoals if specified
                        if (check_subgoals and step.subgoal_atoms is not None):
                            current_atoms = utils.abstract(
                                next_state, self._get_all_predicates())
                            if step.subgoal_atoms.issubset(current_atoms):
                                can_continue = True
                            else:
                                missing = step.subgoal_atoms - current_atoms
                                fail_reason = (
                                    f"subgoal missing: "
                                    f"{{{', '.join(str(a) for a in missing)}}}"
                                )
                        else:
                            can_continue = True
                        # Final step: also check task goal
                        if can_continue and cur_idx == n - 1:
                            if not task.goal_holds(next_state):
                                can_continue = False
                                fail_reason = "goal not reached"

            if can_continue:
                logging.info(
                    f"  Step {cur_idx}/{n} {step_name} OK "
                    f"(sample {num_tries[cur_idx]}/{max_tries[cur_idx]})\n")
                if CFG.agent_bilevel_log_state:
                    next_st = traj[cur_idx + 1]
                    assert next_st is not None
                    logging.debug(f"  State after {step_name}:\n"
                                  f"{next_st.pretty_str()}")
                cur_idx += 1
            else:
                logging.debug(
                    f"  Step {cur_idx}/{n} {step_name} FAIL "
                    f"(sample {num_tries[cur_idx]}/{max_tries[cur_idx]})"
                    f": {fail_reason}")
                # Backtrack: re-try current step or go back further
                while num_tries[cur_idx] >= max_tries[cur_idx]:
                    bt_objs = ", ".join(o.name
                                        for o in sketch[cur_idx].objects)
                    bt_name = (f"{sketch[cur_idx].option.name}"
                               f"({bt_objs})")
                    logging.info(f"  Step {cur_idx}/{n} {bt_name} exhausted "
                                 f"{max_tries[cur_idx]} samples, backtracking")
                    num_tries[cur_idx] = 0
                    plan[cur_idx] = None
                    traj[cur_idx + 1] = None
                    cur_idx -= 1
                    if cur_idx < 0:
                        logging.info(f"Sketch refinement exhausted after "
                                     f"{total_samples} total samples.")
                        return [], False

        # All steps succeeded
        assert all(p is not None for p in plan)
        logging.info(f"Refinement complete: {total_samples} total samples "
                     f"for {n} steps.")
        return cast(List[_Option], plan), True

    def _sample_params(self, option: ParameterizedOption, _state: State,
                       rng: np.random.Generator) -> np.ndarray:
        """Sample continuous parameters for an option.

        Currently uniform random; hook point for future learned
        samplers.
        """
        if option.params_space.shape[0] == 0:
            return np.array([], dtype=np.float32)
        low = option.params_space.low
        high = option.params_space.high
        return rng.uniform(low, high).astype(np.float32)

    # ------------------------------------------------------------------ #
    # Forward validation
    # ------------------------------------------------------------------ #

    def _validate_plan_forward(
        self,
        task: Task,
        plan: List[_Option],
    ) -> bool:
        """Re-execute the plan continuously in the option model's env.

        Unlike refinement (which resets state between steps via
        ``_reset_state``), this runs all options sequentially so that the
        physics state carries forward naturally — matching how the main
        env will execute during the real episode.

        Returns True if the plan reaches the goal, False otherwise.
        """
        state = task.init
        option_names = cast(  # pylint: disable=protected-access
            Any, self._option_model)._name_to_parameterized_option
        predicates = self._get_all_predicates()
        total_actions = 0

        for i, grounded in enumerate(plan):
            # Create a fresh option copy (same as the option model does).
            env_param_opt = option_names.get(grounded.parent.name,
                                             grounded.parent)
            option_copy = env_param_opt.ground(grounded.objects,
                                               grounded.params.copy())
            # Propagate Wait target atoms through re-grounding
            for key in ("wait_target_atoms", "wait_target_neg_atoms"):
                if key in grounded.memory:
                    option_copy.memory[key] = grounded.memory[key]

            if not option_copy.initiable(state):
                logging.info(f"Forward validation: step {i} "
                             f"({option_copy.name}) not initiable.")
                return False

            # Build a terminal condition that mirrors the option model:
            # 1. The option's own terminal
            # 2. terminate_on_repeat (stuck detection)
            # 3. wait_option_terminate_on_atom_change
            last_state_ref: List[Optional[State]] = [None]
            abstract_fn = lambda s, _p=predicates: utils.abstract(s, _p)

            def _terminal(  # pylint: disable=cell-var-from-loop
                    s: State,
                    oc: _Option = option_copy,
                    _abs: Callable = abstract_fn) -> bool:
                if oc.terminal(s):
                    return True
                prev = last_state_ref[0]
                if prev is not None:
                    if (CFG.option_model_terminate_on_repeat
                            and prev.allclose(s)):
                        raise utils.OptionExecutionFailure(
                            f"Option '{oc.name}' got stuck.")
                    if (CFG.wait_option_terminate_on_atom_change
                            and oc.name == "Wait"):
                        result = utils.check_wait_target_atoms(oc, s, _abs)
                        if result is True:
                            last_state_ref[0] = s
                            return True
                        if result is None:
                            cur_atoms = _abs(s)
                            prev_atoms = _abs(prev)
                            if cur_atoms != prev_atoms:
                                last_state_ref[0] = s
                                return True
                last_state_ref[0] = s
                return False

            try:
                sim = cast(  # pylint: disable=protected-access
                    Any, self._option_model)._simulator
                traj = utils.run_policy_with_simulator(
                    option_copy.policy,
                    sim,
                    state,
                    _terminal,
                    max_num_steps=CFG.max_num_steps_option_rollout)
            except (utils.OptionExecutionFailure,
                    utils.EnvironmentFailure) as e:
                logging.info(f"Forward validation: step {i} "
                             f"({option_copy.name}) failed: {e}")
                return False

            if len(traj.actions) == 0:
                logging.info(f"Forward validation: step {i} "
                             f"({option_copy.name}) produced 0 actions.")
                return False

            total_actions += len(traj.actions)
            state = traj.states[-1]
            atoms = utils.abstract(state, predicates)
            logging.debug(
                f"Forward validation: step {i} "
                f"({option_copy.name}) OK, {len(traj.actions)} actions. "
                f"Atoms: {sorted(str(a) for a in atoms)}")

        if not task.goal_holds(state):
            atoms = utils.abstract(state, predicates)
            goal_atoms = task.goal
            missing = goal_atoms - atoms
            logging.info(
                f"Forward validation: goal not reached. "
                f"Missing: {{{', '.join(str(a) for a in sorted(missing))}}}. "
                f"State:\n{state.pretty_str()}")
            return False

        logging.info(f"Forward validation succeeded: {total_actions} "
                     f"actions from {len(plan)} steps.")
        return True

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
