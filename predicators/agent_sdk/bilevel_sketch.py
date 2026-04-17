"""Shared helpers for bilevel plan-sketch construction and refinement.

Extracted from ``AgentBilevelApproach`` so both the approach (at solve
time) and ``AgentBilevelExplorer`` (at exploration time) can build plan
sketches, parse subgoal annotations, and run backtracking refinement
against an arbitrary ``_OptionModelBase``.

The helpers are pure module-level functions — they take their
dependencies (option_model, predicates, rng, settings) explicitly so
neither approaches nor explorers need to subclass one another.
"""
import dataclasses
import logging
import re
from typing import Callable, List, Optional, Sequence, Set, Tuple, cast

import numpy as np

from predicators import utils
from predicators.option_model import _OptionModelBase
from predicators.planning import run_backtracking_refinement
from predicators.structs import GroundAtom, Object, ParameterizedOption, \
    Predicate, State, Task, Type, _Option


@dataclasses.dataclass
class SketchStep:
    """One step in an agent-produced plan sketch.

    ``subgoal_atoms`` / ``subgoal_neg_atoms`` are optional: ``None``
    means "no subgoal constraint at this step"; an empty set means "the
    annotation was present but contained no atoms of that polarity".
    """
    option: ParameterizedOption
    objects: Sequence[Object]
    subgoal_atoms: Optional[Set[GroundAtom]]
    subgoal_neg_atoms: Optional[Set[GroundAtom]] = None


def strip_code_fences(text: str) -> str:
    """Strip markdown code fences wrapping plan text."""
    lines = text.split('\n')
    while lines and lines[0].strip().startswith('```'):
        lines.pop(0)
    while lines and lines[-1].strip().startswith('```'):
        lines.pop()
    return '\n'.join(lines)


def sample_params(option: ParameterizedOption,
                  rng: np.random.Generator) -> np.ndarray:
    """Sample continuous parameters uniformly from the option's box."""
    if option.params_space.shape[0] == 0:
        return np.array([], dtype=np.float32)
    low = option.params_space.low
    high = option.params_space.high
    return rng.uniform(low, high).astype(np.float32)


def build_solve_prompt(
    task: Task,
    *,
    all_predicates: Set[Predicate],
    all_options: Set[ParameterizedOption],
    trajectory_summary: str = "",
    tool_names: Optional[Sequence[str]] = None,
) -> str:
    """Build the bilevel solve/explore prompt asking for a plan sketch.

    Mirrors ``AgentBilevelApproach._build_solve_prompt`` but takes
    dependencies explicitly so explorers can reuse it.
    """
    init_state = task.init
    objects = list(init_state)

    obj_strs = []
    for obj in sorted(objects, key=lambda o: o.name):
        obj_strs.append(f"  {obj.name}: {obj.type.name}")

    goal_strs = [str(a) for a in sorted(task.goal, key=str)]

    option_strs = []
    for opt in sorted(all_options, key=lambda o: o.name):
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

    atoms = utils.abstract(init_state, all_predicates)
    atom_strs = [str(a) for a in sorted(atoms, key=str)]

    state_str = init_state.dict_str(indent=2)

    tools_str = ""
    if tool_names:
        tool_list = "\n".join(f"  - {t}" for t in tool_names)
        tools_str = f"\n## Available Tools\n{tool_list}\n"

    goal_nl_section = ""
    if task.goal_nl:
        goal_nl_section = f"\n## Goal Description\n{task.goal_nl}\n"

    pred_strs = []
    for pred in sorted(all_predicates, key=lambda p: p.name):
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
{trajectory_summary}{tools_str}
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


def parse_subgoal_annotations(
    text: str,
    predicates: Set[Predicate],
    objects: Sequence[Object],
    option_names: Set[str],
) -> List[Optional[Tuple[Set[GroundAtom], Set[GroundAtom]]]]:
    """Parse ``-> {Pred(...), NOT Pred(...)}`` annotations from plan text.

    Returns a list parallel to the option lines in ``text``. Each entry
    is ``None`` for a line with no annotation, or ``(positive_atoms,
    negative_atoms)`` otherwise.
    """
    pred_map = {p.name: p for p in predicates}
    obj_map = {o.name: o for o in objects}

    subgoal_re = re.compile(r'->\s*\{([^}]*)\}')
    atom_re = re.compile(r'(NOT\s+)?(\w+)\(([^)]*)\)')

    results: List[Optional[Tuple[Set[GroundAtom], Set[GroundAtom]]]] = []

    for line in text.split('\n'):
        stripped = line.strip()
        if not stripped:
            continue
        first_token = stripped.split('(')[0]
        if first_token not in option_names:
            continue

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
            obj_names = [
                n.strip().split(':')[0] for n in atom_match.group(3).split(',')
            ]

            if pred_name not in pred_map:
                logging.warning(f"Unknown predicate in subgoal: {pred_name}")
                continue
            pred = pred_map[pred_name]
            try:
                objs = [obj_map[n] for n in obj_names]
            except KeyError as e:
                logging.warning(f"Unknown object in subgoal: {e}")
                continue
            if len(objs) != len(pred.types):
                logging.warning(f"Arity mismatch for {pred_name}: expected "
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


def parse_sketch_from_text(
    plan_text: str,
    task: Task,
    *,
    predicates: Set[Predicate],
    options: Set[ParameterizedOption],
    types: Set[Type],
) -> List[SketchStep]:
    """Parse plan-sketch text into ``SketchStep``s.

    Applies ``strip_code_fences`` first, then delegates option-plan
    parsing to ``utils.parse_model_output_into_option_plan`` and subgoal
    annotation parsing to ``parse_subgoal_annotations``.
    """
    cleaned_text = strip_code_fences(plan_text)
    objects = list(task.init)
    option_names = {o.name for o in options}

    parsed = utils.parse_model_output_into_option_plan(
        cleaned_text, objects, types, options, parse_continuous_params=False)

    if not parsed:
        return []

    subgoals = parse_subgoal_annotations(cleaned_text, predicates, objects,
                                         option_names)

    sketch: List[SketchStep] = []
    for i, (option, objs, _) in enumerate(parsed):
        sg = subgoals[i] if i < len(subgoals) else None
        if sg is not None:
            pos, neg = sg
            sketch.append(
                SketchStep(option=option,
                           objects=objs,
                           subgoal_atoms=pos if pos else None,
                           subgoal_neg_atoms=neg if neg else None))
        else:
            sketch.append(
                SketchStep(option=option, objects=objs, subgoal_atoms=None))
    return sketch


def refine_sketch(
    task: Task,
    sketch: List[SketchStep],
    option_model: _OptionModelBase,
    *,
    predicates: Set[Predicate],
    timeout: float,
    rng: np.random.Generator,
    max_samples_per_step: int,
    check_subgoals: bool,
    check_final_goal: bool = True,
    truncate_on_subgoal_fail: bool = False,
    log_state: bool = False,
    run_id: str = "bilevel",
    on_step_fail: Optional[Callable[[int, List[Optional[_Option]], str],
                                    None]] = None,
) -> Tuple[List[_Option], bool, int]:
    """Backtracking search over continuous parameters for a plan sketch.

    Returns ``(refined_plan, success, total_samples)``. On success the
    plan is fully refined; on failure it is the longest prefix of
    refined options (``None`` entries dropped).

    ``check_subgoals`` gates per-step subgoal-atom validation.
    ``check_final_goal`` gates the task-goal check on the final step.
    ``truncate_on_subgoal_fail`` (explorer mode) lets backtracking run
    to exhaustion with subgoal checks enabled, then — if the search
    fails — returns the consistent plan prefix captured at the deepest
    subgoal failure seen during backtracking (inclusive of the failing
    step). Use this to build *experiment* plans that probe a single
    mental-model disagreement: upstream steps get their standard
    backtracking retries, but once the deepest unresolvable subgoal is
    identified, subsequent sketch steps are dropped (they would be
    built on a false mental-model state).

    Wait steps inject ``wait_target_atoms`` / ``wait_target_neg_atoms``
    from the sketch's subgoal annotations into ``grounded.memory`` so
    that ``WaitOption`` terminates on the intended atom change rather
    than the first incidental one.
    """
    if not sketch:
        return [], False, 0

    n = len(sketch)
    max_tries = [
        max_samples_per_step if step.option.params_space.shape[0] > 0 else 1
        for step in sketch
    ]
    # Snapshot of the deepest subgoal failure seen during backtracking.
    # Tracks (idx, plan_prefix_snapshot). Updated whenever on_step_fail
    # reports a subgoal failure at a strictly deeper index than before.
    # The snapshot is taken at the moment of failure, so it is a
    # *consistent* trajectory: run_backtracking_refinement has already
    # written plan[idx] for that attempt and the prefix plan[:idx+1]
    # reflects the exact grounded options that led to this failure.
    deepest_subgoal_fail_idx: List[int] = [-1]
    deepest_subgoal_fail_prefix: List[List[Optional[_Option]]] = [[]]

    def sample_fn(idx: int, state: State,
                  rng_: np.random.Generator) -> _Option:
        step = sketch[idx]
        if log_state:
            step_name = (f"{step.option.name}"
                         f"({', '.join(o.name for o in step.objects)})")
            logging.debug(f"[{run_id}]  State before {step_name}:\n"
                          f"{state.pretty_str()}")
        params = sample_params(step.option, rng_)
        grounded = step.option.ground(list(step.objects), params)
        if grounded.name == "Wait":
            if step.subgoal_atoms is not None:
                grounded.memory["wait_target_atoms"] = step.subgoal_atoms
            if step.subgoal_neg_atoms is not None:
                grounded.memory["wait_target_neg_atoms"] = \
                    step.subgoal_neg_atoms
        return grounded

    def validate_fn(idx: int, _pre_state: State, _option: _Option,
                    post_state: State, _num_actions: int) -> Tuple[bool, str]:
        step = sketch[idx]
        if check_subgoals and step.subgoal_atoms is not None:
            current_atoms = utils.abstract(post_state, predicates)
            if not step.subgoal_atoms.issubset(current_atoms):
                missing = step.subgoal_atoms - current_atoms
                return False, (f"subgoal missing: "
                               f"{{{', '.join(str(a) for a in missing)}}}")
        if check_final_goal and idx == n - 1:
            if not task.goal_holds(post_state):
                return False, "goal not reached"
        return True, ""

    def wrapped_on_step_fail(idx: int, cur_plan: List[Optional[_Option]],
                             fail_reason: str) -> None:
        # run_backtracking_refinement calls this BEFORE clearing
        # plan[idx] (planning.py lines 592-599), so cur_plan[0..idx] is
        # still populated with the grounded options that produced this
        # exact failure trajectory. Record the deepest subgoal failure
        # seen so far along with a consistent snapshot of the prefix.
        if (truncate_on_subgoal_fail
                and fail_reason.startswith("subgoal missing")
                and idx > deepest_subgoal_fail_idx[0]):
            deepest_subgoal_fail_idx[0] = idx
            deepest_subgoal_fail_prefix[0] = list(cur_plan[:idx + 1])
        if on_step_fail is not None:
            on_step_fail(idx, cur_plan, fail_reason)

    plan, success, total_samples = run_backtracking_refinement(
        init_state=task.init,
        option_model=option_model,
        n_steps=n,
        max_tries=max_tries,
        sample_fn=sample_fn,
        validate_fn=validate_fn,
        rng=rng,
        timeout=timeout,
        on_step_fail=wrapped_on_step_fail,
    )

    logging.info(
        f"[{run_id}] Refinement {'succeeded' if success else 'failed'}: "
        f"{total_samples} samples for {n} steps.")

    if (truncate_on_subgoal_fail and not success
            and deepest_subgoal_fail_idx[0] >= 0):
        snapshot = deepest_subgoal_fail_prefix[0]
        refined = [p for p in snapshot if p is not None]
        logging.info(f"[{run_id}] Truncating at deepest subgoal failure "
                     f"(step {deepest_subgoal_fail_idx[0]}): "
                     f"{len(refined)}/{n} steps in experiment plan.")
        return cast(List[_Option], refined), False, total_samples

    refined = [p for p in plan if p is not None]
    if success:
        return cast(List[_Option], refined), True, total_samples
    return refined, False, total_samples
