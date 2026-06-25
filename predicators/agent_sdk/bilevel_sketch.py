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
from typing import Callable, Collection, Dict, List, Optional, Sequence, Set, \
    Tuple, cast

import numpy as np

from predicators import utils
from predicators.option_model import _OptionModelBase
from predicators.planning import run_backtracking_refinement
from predicators.structs import GroundAtom, Object, OptionSampler, \
    ParameterizedOption, Predicate, State, Task, Type, _Option

# Signature of an info-gain scorer: given a candidate post-state and the
# atoms whose truth the step is meant to establish, return a scalar where
# larger means more informative about the learned model (e.g. ensemble
# disagreement on those atoms). Used to turn refinement from
# feasibility-seeking into information-seeking.
InfoScorer = Callable[[State, Collection[GroundAtom]], float]


def _fmt_params(opt: _Option) -> str:
    """Compact one-line dump of a grounded option's parameters."""
    return np.array2string(np.asarray(opt.params, dtype=float),
                           precision=4,
                           separator=", ")


@dataclasses.dataclass
class _FeasiblePool:
    """Ranked stock of feasible candidates at one search node.

    A search node is a step under a fixed prefix of upstream choices —
    equivalently one attempt cycle of ``run_backtracking_refinement``
    (the step's try counter and its pre-state both change only when the
    step exhausts and an upstream step re-chooses). ``pre_state`` is the
    exact ``State`` object the pool was drawn from; holding the
    reference keeps the object alive, so ``is``-identity in
    ``_sample_info_seeking`` detects precisely when an upstream re-
    choice rewrote ``traj[idx]`` (new node ⇒ stale stock, fresh budget).
    ``spent`` counts pool rollouts charged against the node's budget;
    ``ranked`` holds the not-yet-proposed feasible candidates as
    ``(info_score, option)``, most informative first.
    """
    pre_state: State
    spent: int
    ranked: List[Tuple[float, _Option]]


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
    # Optional LLM-proposed continuous parameters for this step. ``None``
    # means "no proposal" (the search samples from the start); otherwise the
    # refinement tries these first (clipped to the option's box) on the first
    # arrival at this step, then falls back to the sampler/uniform draw.
    initial_params: Optional[np.ndarray] = None


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
    experiment_guidance: str = "",
    prior_failures: str = "",
    initial_image_section: str = "",
    propose_params: bool = False,
    require_tool_validation: bool = False,
) -> str:
    """Build the bilevel solve/explore prompt asking for a plan sketch.

    Mirrors ``AgentBilevelApproach._build_solve_prompt`` but takes
    dependencies explicitly so explorers can reuse it.

    ``prior_failures`` is a pre-formatted block summarizing earlier
    sketch attempts that the backtracking search could not refine (with a
    pointer to the full per-step log in the sandbox). Injected so a
    re-query produces a *different* skeleton instead of re-emitting the
    dead one.

    ``propose_params`` switches the prompt from "param-free sketch, search
    finds all continuous params" to "propose your best continuous params in
    ``[...]`` per step; the search refines them and samples on failure".

    ``require_tool_validation`` tells the agent it MUST drive
    ``refine_plan_sketch`` to a SUCCESS on the current task (its captured,
    forward-validated plan is the only output) — used when the approach has
    no refinement fallback. When False, validation is merely encouraged.
    """
    init_state = task.init
    objects = list(init_state)

    obj_strs = []
    for obj in sorted(objects, key=lambda o: o.name):
        obj_strs.append(f"  {obj.name}: {obj.type.name}")

    # Only expose goal atoms whose predicate is in the agent's current
    # predicate set. Approaches that strip env predicates (e.g.
    # agent_sim_predicate_invention) rely on goal_nl to communicate the
    # goal; leaking unfiltered task.goal atoms would expose predicates the
    # agent is supposed to invent for itself.
    goal_strs = [
        str(a) for a in sorted(task.goal, key=str)
        if a.predicate in all_predicates
    ]

    option_strs = []
    for opt in sorted(all_options, key=lambda o: o.name):
        type_sig = ", ".join(t.name for t in opt.types)
        params_dim = opt.params_space.shape[0]
        if params_dim > 0:
            low = opt.params_space.low.tolist()
            high = opt.params_space.high.tolist()
            label = "params" if propose_params else "auto-searched params"
            if opt.params_description:
                desc = ", ".join(opt.params_description)
                param_info = (f"  [{label}: {desc}, "
                              f"range {low} to {high}]")
            else:
                kind = f"{params_dim}d"
                param_info = (f"  [{label}: {kind}, "
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

    experiment_section = ""
    if experiment_guidance:
        experiment_section = (f"\n## Experiment Guidance\n"
                              f"{experiment_guidance}\n")

    prior_failures_section = ""
    if prior_failures:
        prior_failures_section = (
            "\n## Previous Sketch Attempts (FAILED — do NOT repeat them)\n"
            "Each block below is a sketch you already tried and the "
            "backtracking search could NOT refine, with where it got stuck "
            "and a pointer to the full per-step refinement log (read it with "
            "`Read` for details). Produce a DIFFERENT skeleton that avoids "
            "the failure — change the step that got stuck (object choice, "
            "ordering, an intermediate step, or its subgoal annotation).\n"
            f"{prior_failures}\n")

    goal_nl_section = ""
    if task.goal_nl:
        goal_nl_section = f"\n## Goal Description\n{task.goal_nl}\n"

    goal_atoms_section = ""
    if goal_strs:
        goal_atoms_section = (f"\n## Goal Atoms\n{chr(10).join(goal_strs)}\n")

    pred_strs = []
    for pred in sorted(all_predicates, key=lambda p: p.name):
        type_sig = ", ".join(t.name for t in pred.types)
        line = f"  {pred.name}({type_sig})"
        if pred.natural_language_assertion is not None:
            names = [t.name for t in pred.types]
            line += f" — {pred.natural_language_assertion(names)}"
        pred_strs.append(line)

    # Advice for a step the search reports stuck (SAMPLE_EXHAUSTED). When the
    # agent proposes params it tunes that step's values; otherwise it can only
    # change the skeleton.
    deep_tune_advice = (
        "deep-tune just that step (it needs precise values from you), then "
        "re-test it. When deep-tuning a step with `evaluate_option_plan`:\n"
        "- Inspect the rendered images in `./test_images/` to see what "
        "actually happened.\n"
        "- For a failure like an IK error or collision, use the image and "
        "object poses to reason about WHY and adjust params directionally — "
        "don't try random nearby values.\n"
        "- Use `visualize_state` to move objects to candidate positions and "
        "orientations for free (no physics) and find the right region "
        "visually before testing.\n"
        "- Vary ALL parameters, not just position — orientation and others "
        "affect both the outcome and whether the action succeeds.\n"
        "- Search coarse-to-fine: spread attempts across the full range; if "
        "several nearby values fail the same way, jump to a different region "
        "instead of continuing to tweak.")
    revise_sketch_advice = (
        "revise the sketch — try different objects, a different ordering, an "
        "added intermediate step, or a corrected subgoal annotation — then "
        "re-test.")

    if propose_params:
        sketch_kind_guidance = (
            "Generate a plan — the sequence of options with object arguments "
            "and continuous parameters in `[...]` per step (see each option's "
            "params and range above; use `[]` for options with no "
            "parameters).\n\n"
            "Spend effort on parameters in proportion to difficulty:\n"
            "- Where a WIDE range of values works, any reasonable value is "
            "fine — don't over-tune these.\n"
            "- Where good values are hard to hit — tight tolerances or exact "
            "relative placements (e.g. positioning one object at a precise "
            "offset from another) — use `refine_plan_sketch` to search for a "
            "working value (it's slower) and read the value it found.")
        format_block = (
            "Output the plan with one option per line in this format:\n"
            "  OptionName(obj1:type1, obj2:type2)[param1, param2] -> "
            "{Pred(obj1:type1), Pred2(obj1:type1, obj2:type2)}\n"
            "  Wait(robot:robot)[] -> "
            "{Pred3(obj1:type1), NOT Pred4(obj1:type1, obj2:type2)}")
    else:
        sketch_kind_guidance = (
            "Generate a plan sketch — the sequence of options with object "
            "arguments, WITHOUT continuous parameters; a backtracking search "
            "finds them for you.")
        format_block = (
            "Output the plan sketch with one option per line in this "
            "format:\n"
            "  OptionName(obj1:type1, obj2:type2) -> "
            "{Pred(obj1:type1), Pred2(obj1:type1, obj2:type2)}\n"
            "  Wait(robot:robot) -> "
            "{Pred3(obj1:type1), NOT Pred4(obj1:type1, obj2:type2)}")

    if require_tool_validation:
        stuck_advice = (deep_tune_advice
                        if propose_params else revise_sketch_advice)
        submit_guidance = (
            "SUBMIT via `evaluate_option_plan`: pass your full plan as text "
            "(one option per line, `Option(obj:type)[params] -> {subgoals}`, "
            "with EXACT params) and run it on the CURRENT task (omit "
            "task_idx). When it reaches the goal, that plan is captured as "
            "your answer, so do NOT finish until evaluate_option_plan reaches "
            "the goal. It runs your EXACT parameters with no sampling. To find "
            "working parameters you MAY use `refine_plan_sketch` (it searches "
            "but is slower); read the parameters it reports and submit them "
            "via evaluate_option_plan. If a step does not reach its subgoal, "
            + stuck_advice)
    elif propose_params:
        submit_guidance = (
            "You may validate with `refine_plan_sketch` (it tries your "
            "parameters first, then samples) and deep-tune any step it "
            "reports stuck before finishing.")
    else:
        submit_guidance = (
            "You may vet a sketch with `refine_plan_sketch` before finishing; "
            "the backtracking search will find continuous parameters.")

    prompt = f"""You are solving a task. \
Generate a plan sketch to achieve the goal.
{goal_nl_section}{goal_atoms_section}{experiment_section}
## Initial State Atoms
{chr(10).join(atom_strs)}

## Initial State Features
{state_str}
{initial_image_section}
## Objects
{chr(10).join(obj_strs)}

## Available Options
{chr(10).join(option_strs)}

## Available Predicates (for subgoal annotations)
{chr(10).join(pred_strs)}
{trajectory_summary}{tools_str}{prior_failures_section}
## Instructions
Use your available tools to inspect the environment before producing the plan.

{sketch_kind_guidance}

{submit_guidance}

Annotate subgoal atoms after EVERY step whose effect your predicates can \
express, using `-> {{atoms}}`. Prefer atoms that NEWLY hold (or stop \
holding) because of the step — atoms that were already true beforehand \
reveal nothing. Annotations are load-bearing: the search validates each \
annotated step, and during execution they are checked against the real \
state so a diverged step triggers replanning instead of silently dooming \
the rest of the plan.

After any action whose desired subgoal depends on a delayed process (e.g. \
water filling, dominoes cascading, heating), insert a Wait action. For Wait \
steps, annotate with the atoms the process should produce — this tells the \
system exactly when the Wait should end rather than terminating on any \
incidental atom change. Use `NOT Pred(...)` for atoms that should become false.

{format_block}

Always use typed references (obj:type) in both option arguments AND subgoal \
atoms. If you omit `-> {{atoms}}` on a step, the search only checks that the \
option executed (non-zero actions) and execution monitoring is blind there — \
omit it only when no available predicate can express the step's effect.

Output ONLY the plan sketch lines at the end, after any analysis."""

    return prompt


# Matches an atom like ``Pred(a:t, b:t)`` or ``NOT Pred(a)`` in subgoal text.
_ATOM_RE = re.compile(r'(NOT\s+)?(\w+)\(([^)]*)\)')


def parse_atoms(
    atoms_text: str,
    predicates: Set[Predicate],
    objects: Sequence[Object],
) -> Tuple[Set[GroundAtom], Set[GroundAtom]]:
    """Parse atoms like ``Pred(a:t, b:t)`` / ``NOT Pred(a)`` from a string.

    Returns ``(positive_atoms, negative_atoms)``. Any number of atoms
    may appear in ``atoms_text`` (separated by commas or anything else —
    the regex finds each ``Pred(...)``). Atoms with an unknown predicate
    or object, or the wrong arity, are skipped with a warning.
    """
    pred_map = {p.name: p for p in predicates}
    obj_map = {o.name: o for o in objects}
    pos_atoms: Set[GroundAtom] = set()
    neg_atoms: Set[GroundAtom] = set()
    for atom_match in _ATOM_RE.finditer(atoms_text):
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
        (neg_atoms if is_neg else pos_atoms).add(GroundAtom(pred, objs))
    return pos_atoms, neg_atoms


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
    subgoal_re = re.compile(r'->\s*\{([^}]*)\}')
    results: List[Optional[Tuple[Set[GroundAtom], Set[GroundAtom]]]] = []

    for line in text.split('\n'):
        # Mirror the enumeration-prefix tolerance in the option-plan
        # parser so the per-line subgoal results stay index-parallel with
        # the parsed options (a numbered "0: Pick(...)" line must be seen
        # as an option line here too, else annotations misalign).
        stripped = utils.strip_enumeration_prefix(line.strip())
        if not stripped:
            continue
        first_token = stripped.split('(')[0]
        if first_token not in option_names:
            continue

        sg_match = subgoal_re.search(stripped)
        if not sg_match:
            results.append(None)
            continue

        pos_atoms, neg_atoms = parse_atoms(sg_match.group(1), predicates,
                                           objects)
        if pos_atoms or neg_atoms:
            results.append((pos_atoms, neg_atoms))
        else:
            results.append(None)

    return results


# A `-> {atoms}` subgoal annotation appended to a sketch step line. Stripped
# before the canonical option-plan parser reads the `[params]` block, so a
# `{...}` brace is never mistaken for params text.
_SUBGOAL_ANNOTATION_RE = re.compile(r'\s*->\s*\{[^}]*\}')


def strip_subgoal_annotations(text: str) -> str:
    """Remove ``-> {atoms}`` subgoal annotations from every line."""
    return _SUBGOAL_ANNOTATION_RE.sub('', text)


def parse_sketch_from_text(
    plan_text: str,
    task: Task,
    *,
    predicates: Set[Predicate],
    options: Set[ParameterizedOption],
    types: Set[Type],
    parse_continuous_params: bool = False,
) -> List[SketchStep]:
    """Parse plan-sketch text into ``SketchStep``s.

    Applies ``strip_code_fences`` first, then delegates option-plan
    parsing to ``utils.parse_model_output_into_option_plan`` and subgoal
    annotation parsing to ``parse_subgoal_annotations``.

    When ``parse_continuous_params`` is set, each step's ``[p0, p1, ...]``
    block is parsed by the SAME canonical parser the open-loop planner
    uses (``parse_model_output_into_option_plan`` with
    ``parse_continuous_params=True``) and stored as ``initial_params`` for
    the refinement to try first. Sketch lines also carry ``-> {subgoal}``
    annotations, which that parser would misread as params text, so they
    are stripped before parsing and read separately from the original.
    """
    cleaned_text = strip_code_fences(plan_text)
    objects = list(task.init)
    option_names = {o.name for o in options}

    # Strip subgoal annotations only when parsing params, so the `[params]`
    # extraction in the canonical parser isn't confused by a `{...}` brace.
    parse_text = (strip_subgoal_annotations(cleaned_text)
                  if parse_continuous_params else cleaned_text)
    parsed = utils.parse_model_output_into_option_plan(
        parse_text,
        objects,
        types,
        options,
        parse_continuous_params=parse_continuous_params)

    if not parsed:
        return []

    subgoals = parse_subgoal_annotations(cleaned_text, predicates, objects,
                                         option_names)

    sketch: List[SketchStep] = []
    for i, (option, objs, params) in enumerate(parsed):
        sg = subgoals[i] if i < len(subgoals) else None
        ip = (np.asarray(params, dtype=np.float32)
              if parse_continuous_params else None)
        if sg is not None:
            pos, neg = sg
            sketch.append(
                SketchStep(option=option,
                           objects=objs,
                           subgoal_atoms=pos if pos else None,
                           subgoal_neg_atoms=neg if neg else None,
                           initial_params=ip))
        else:
            sketch.append(
                SketchStep(option=option,
                           objects=objs,
                           subgoal_atoms=None,
                           initial_params=ip))
    # Coverage diagnostic: unannotated steps are invisible to per-step
    # refinement validation, execution monitoring, and suffix replanning.
    unannotated = [
        f"{i}: {s.option.name}" for i, s in enumerate(sketch)
        if s.subgoal_atoms is None and s.subgoal_neg_atoms is None
    ]
    if unannotated:
        logging.info("Sketch subgoal coverage: %d/%d steps unannotated (%s).",
                     len(unannotated), len(sketch), ", ".join(unannotated))
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
    step_samples_cumulative: Optional[List[int]] = None,
    termination_reason: Optional[List[str]] = None,
    elapsed_holder: Optional[List[float]] = None,
    info_scorer: Optional[InfoScorer] = None,
    info_n_feasible_target: int = 1,
    option_samplers: Optional[Dict[str, OptionSampler]] = None,
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
    validation failure seen during backtracking (inclusive of the
    failing step). "Validation failure" covers both an unmet subgoal
    atom and, when ``check_final_goal`` is on, an unreached task goal at
    the final step; the latter captures the *whole* plan as the
    experiment (run it in reality and observe — a goal the mental model
    predicts won't hold is exactly the disagreement worth collecting).
    Use this to build *experiment* plans that probe a mental-model
    disagreement: upstream steps get their standard backtracking
    retries, but once the deepest unresolvable step is identified,
    subsequent sketch steps are dropped (they would be built on a false
    mental-model state).

    ``max_samples_per_step`` is a per-step rollout budget per *search
    node* (the step under a fixed prefix of upstream choices;
    backtracking past the step and re-descending with a new upstream
    choice starts a new node). Plain steps spend it the classic way, one
    sampled rollout per attempt. Info-seeking steps spend it pooling
    candidates at the node, and the pooled feasible candidates double as
    a ranked retry stock — the budget is spent once, never multiplied.

    With ``info_scorer`` set and ``info_n_feasible_target > 1``,
    parameter sampling at subgoal-annotated steps with continuous
    parameters becomes information-seeking: candidates are drawn until
    ``info_n_feasible_target`` feasible ones are pooled (bounded by the
    node's rollout budget) and proposed most-informative-first, one per
    attempt, with no re-drawing while the stock lasts (a retry after a
    downstream collapse or a final-goal miss pops the next-best for
    free). The step's attempt cap equals ``info_n_feasible_target``, so
    it exhausts exactly when every pooled candidate has been tried. See
    ``_sample_info_seeking``.

    Wait steps inject ``wait_target_atoms`` / ``wait_target_neg_atoms``
    from the sketch's subgoal annotations into ``grounded.memory`` so
    that ``WaitOption`` terminates on the intended atom change rather
    than the first incidental one.

    ``option_samplers`` maps an option name to a per-skill sampler
    ``(state, subgoal_atoms, rng, objects) -> params`` (the NSRTSampler
    signature, with the step subgoal in the atoms slot), used on both
    plain and info-seeking draws to aim that option's parameters at the
    subgoal instead of drawing uniformly. The return is clipped to the
    option's box; a missing or misbehaving sampler falls back to uniform
    sampling.
    """
    if not sketch:
        return [], False, 0

    n = len(sketch)
    # Snapshot of the deepest validation failure seen during backtracking
    # (an unmet subgoal atom, or — with check_final_goal — an unreached
    # task goal at the final step). Tracks (idx, plan_prefix_snapshot),
    # updated whenever on_step_fail reports such a failure at a strictly
    # deeper index than before. The snapshot is taken at the moment of
    # failure, so it is a *consistent* trajectory: run_backtracking_refinement
    # has already written plan[idx] for that attempt and the prefix
    # plan[:idx+1] reflects the exact grounded options that led to it.
    deepest_fail_idx: List[int] = [-1]
    deepest_fail_prefix: List[List[Optional[_Option]]] = [[]]

    # Options whose synthesized sampler already misbehaved once — so the
    # per-draw fallback warning fires at most once per option, not on every
    # one of the (potentially thousands of) draws during backtracking.
    _sampler_warned: Set[str] = set()

    # Step indices whose LLM-proposed initial_params have already been used --
    # tried directly on the plain path, or seeded into the info-seeking pool.
    # One-shot per step: on later attempts (resample after a failed subgoal
    # check, or re-descent after an upstream backtrack) the guess is not
    # re-proposed/re-seeded and selection falls to the sampler/uniform/
    # info-seeking path.
    _llm_params_tried: Set[int] = set()

    def _draw_params(step: SketchStep, state: State,
                     rng_: np.random.Generator) -> np.ndarray:
        """Draw continuous params for a step's option.

        Uses a registered per-skill sampler (keyed by option name) when
        present, else falls back to uniform ``sample_params`` — also on
        a sampler error or wrong-shaped return.
        """
        sampler = (option_samplers.get(step.option.name)
                   if option_samplers else None)
        if sampler is not None:
            box = step.option.params_space
            expected = box.shape[0]
            try:
                raw = sampler(state, step.subgoal_atoms or set(), rng_,
                              list(step.objects))
                params = np.asarray(raw, dtype=np.float32).reshape(-1)
                if params.shape == (expected, ):
                    return np.clip(params, box.low, box.high)
                reason = (f"returned shape {params.shape}, "
                          f"expected ({expected},)")
            except Exception as e:  # pylint: disable=broad-except
                reason = f"raised {type(e).__name__}: {e}"
            if step.option.name not in _sampler_warned:
                _sampler_warned.add(step.option.name)
                logging.warning(
                    "[%s] synthesized sampler for %s %s; falling back to "
                    "uniform sampling for this option.", run_id,
                    step.option.name, reason)
        return sample_params(step.option, rng_)

    def _ground(step: SketchStep, params: np.ndarray) -> _Option:
        grounded = step.option.ground(list(step.objects), params)
        if grounded.name == "Wait":
            if step.subgoal_atoms is not None:
                grounded.memory["wait_target_atoms"] = step.subgoal_atoms
            if step.subgoal_neg_atoms is not None:
                grounded.memory["wait_target_neg_atoms"] = \
                    step.subgoal_neg_atoms
        return grounded

    def _info_seeking_applies(step: SketchStep) -> bool:
        # Pooled selection only helps when there are continuous params to
        # choose among AND subgoal atoms whose truth the ensemble can
        # disagree about. Parameter-free steps (e.g. Wait) and unannotated
        # steps fall through to the plain single-sample path unchanged.
        return (info_scorer is not None and info_n_feasible_target > 1
                and step.option.params_space.shape[0] > 0
                and step.subgoal_atoms is not None)

    # Per-step attempt caps. Plain steps spend their whole budget as
    # attempts: one sampled rollout per attempt, max_samples_per_step
    # attempts (unchanged semantics). Info-seeking steps get exactly
    # info_n_feasible_target attempts: the pooled feasible candidates
    # double as the node's retry stock, one proposed per attempt, so the
    # step exhausts precisely when every pooled candidate has been tried
    # (with 1-draw fillers for attempts left over when the pool came up
    # short of the target).
    def _is_deterministic(step: SketchStep) -> bool:
        # A sampler may flag itself as returning constant params (ignoring
        # state/rng); re-drawing it yields the identical option, so its step
        # gets a single attempt -- backtracking then skips straight past it
        # instead of wasting the full budget re-descending through it.
        sampler = (option_samplers.get(step.option.name)
                   if option_samplers else None)
        return bool(getattr(sampler, "deterministic", False))

    max_tries = []
    for _step in sketch:
        if _step.option.params_space.shape[0] == 0:
            max_tries.append(1)
        elif _is_deterministic(_step):
            max_tries.append(1)
        elif _info_seeking_applies(_step):
            max_tries.append(info_n_feasible_target)
        else:
            max_tries.append(max_samples_per_step)

    # Node-scoped pools for info-seeking steps: step_pools[idx] holds
    # the ranked feasible stock and rollout spend for the step's current
    # search node (see _FeasiblePool for the node-identity mechanism).
    # total_pool_rollouts accumulates across the whole search for the
    # completion log, since run_backtracking_refinement's total_samples
    # only counts attempts.
    step_pools: List[Optional[_FeasiblePool]] = [None] * n
    total_pool_rollouts = [0]

    def _sample_info_seeking(step: SketchStep, state: State,
                             rng_: np.random.Generator, idx: int) -> _Option:
        """Propose the most informative not-yet-tried feasible candidate for
        the step's current search node.

        When the step carries LLM-proposed ``initial_params`` (and they
        have not been used yet), they are evaluated as the FIRST candidate
        of the node's pool, so the max-disagreement selection chooses among
        {LLM guess} ∪ sampled draws rather than the guess short-circuiting
        the probe.

        The first attempt at a node draws candidates — each rolled
        forward through the same option_model the backtracking loop uses
        — until ``info_n_feasible_target`` feasible ones are pooled or
        the node's rollout budget (``max_samples_per_step``) is spent,
        then proposes the max-disagreement one and banks the rest as a
        ranked stock. Later attempts at the same node (the loop retries
        after a final-goal miss or after downstream steps collapse back
        onto this one) pop the next-best from the stock with NO new
        rollouts: the candidates were already rolled out and
        subgoal-checked, and the pre-state is fixed within a node, so
        for a deterministic learned model they stay valid. (With a
        stochastic model a popped candidate may still fail the loop's
        re-execution — it just consumes an attempt, like any failure.)

        Candidates that aren't initiable, produce no actions, or fail
        to establish the subgoal consume budget but never enter the
        stock. If a draw round finds nothing feasible, the first sample
        is returned so the loop records the validation failure
        (explorer-mode truncation relies on it); an attempt arriving
        with both stock and budget exhausted gets a 1-draw minimum so
        it can fail fast until the attempt cap
        (= ``info_n_feasible_target``) exhausts the step.

        Node identity: ``traj[idx]`` is rewritten only when an upstream
        step re-executes, which can only happen after this step
        exhausts, so comparing the pre-state *object* (``is``) flips
        exactly at node boundaries — stale stock is dropped and the
        budget refreshed.
        """
        assert info_scorer is not None and step.subgoal_atoms is not None
        # Narrowed locals so the nested _consider closure (below) keeps the
        # non-None types mypy can't carry across the function boundary.
        scorer = info_scorer
        subgoal_atoms = step.subgoal_atoms
        objs = ", ".join(o.name for o in step.objects)
        pool = step_pools[idx]
        if pool is None or pool.pre_state is not state:
            pool = _FeasiblePool(pre_state=state, spent=0, ranked=[])
            step_pools[idx] = pool
        if pool.ranked:
            score, grounded = pool.ranked.pop(0)
            logging.info(
                "[%s] info-seeking %s(%s): proposing next-ranked stock "
                "candidate params %s (disagreement %.4f, %d left in "
                "stock) — no new rollouts.", run_id, step.option.name, objs,
                _fmt_params(grounded), score, len(pool.ranked))
            return grounded
        # Stock empty: first attempt at this node, or every pooled
        # candidate has been proposed. Draw from the node's remaining
        # budget (>=1 so the attempt can still fail fast when spent).
        draw_cap = max(max_samples_per_step - pool.spent, 1)
        best_score = -float("inf")
        best_nxt: Optional[State] = None
        scored: List[Tuple[float, _Option]] = []
        # Score of the first feasible candidate — what plain (greedy
        # first-feasible) backtracking would have accepted; logged as the
        # baseline so a run shows what boundary-probing bought. With LLM
        # params on, the seeded guess (below) is that first candidate.
        first_feasible_score: Optional[float] = None
        first_candidate: Optional[_Option] = None
        n_draws = 0

        def _consider(grounded: _Option) -> None:
            # Roll one grounded candidate forward and, when it establishes
            # the subgoal, fold it into the scored pool and running argmax.
            # Shared by the LLM-guess seed and the sampler/uniform draws so
            # the two stay in lockstep.
            nonlocal best_score, best_nxt, first_feasible_score
            nonlocal first_candidate
            if first_candidate is None:
                first_candidate = grounded
            if not grounded.initiable(state):
                return
            try:
                nxt, num_actions = \
                    option_model.get_next_state_and_num_actions(
                        state, grounded)
            except Exception:  # pylint: disable=broad-except
                # Scoring rollout is best-effort; a model failure on this
                # candidate just removes it from contention.
                return
            if num_actions == 0:
                return
            post_atoms = utils.abstract(nxt, predicates)
            if not subgoal_atoms.issubset(post_atoms):
                return  # infeasible: subgoal not established
            score = scorer(nxt, subgoal_atoms)
            scored.append((score, grounded))
            if first_feasible_score is None:
                first_feasible_score = score
            if score > best_score:
                best_score = score
                best_nxt = nxt

        # Seed the LLM-proposed params (once per step) as the FIRST pool
        # candidate, so the disagreement argmax chooses among
        # {LLM guess} ∪ sampled draws instead of the guess short-circuiting
        # the probe. Clipping mirrors the plain branch; arity is already
        # validated by the option-plan parser.
        if step.initial_params is not None and idx not in _llm_params_tried:
            _llm_params_tried.add(idx)
            box = step.option.params_space
            llm_grounded = _ground(
                step,
                np.clip(np.asarray(step.initial_params, dtype=np.float32),
                        box.low, box.high).astype(np.float32))
            n_draws += 1
            n_pooled_before = len(scored)
            _consider(llm_grounded)
            logging.info(
                "[%s] info-seeking %s(%s): seeded LLM-proposed params %s "
                "(%s) into the candidate pool.", run_id, step.option.name,
                objs, _fmt_params(llm_grounded), "feasible" if
                len(scored) > n_pooled_before else "infeasible — not pooled")

        while len(scored) < info_n_feasible_target and n_draws < draw_cap:
            grounded = _ground(step, _draw_params(step, state, rng_))
            n_draws += 1
            _consider(grounded)
        pool.spent += n_draws
        total_pool_rollouts[0] += n_draws
        # Log every pick at INFO (not gated on log_state) — active-learning
        # visibility into where boundary-probing engaged and what it found.
        # All-zero scores ⇒ ensemble agrees here (uninformative).
        if not scored:
            assert first_candidate is not None
            logging.info(
                "[%s] info-seeking %s(%s): 0 feasible candidates after "
                "%d draws (%d/%d node budget spent; target %d); falling "
                "back to first sample (no boundary probe).", run_id,
                step.option.name, objs, n_draws, pool.spent,
                max_samples_per_step, info_n_feasible_target)
            return first_candidate
        # Stable sort: ties keep draw order, so among equally informative
        # candidates the first-drawn (what plain backtracking would have
        # taken) is proposed first.
        scored.sort(key=lambda t: t[0], reverse=True)
        _, best = scored[0]
        pool.ranked = scored[1:]
        # Per-atom disagreement of the chosen candidate, so the log shows
        # which subgoal atoms carry the uncertainty rather than only the
        # aggregate (mean) the selection maximized.
        assert best_nxt is not None
        assert first_feasible_score is not None
        per_atom = ", ".join(f"{a}={info_scorer(best_nxt, {a}):.4f}"
                             for a in sorted(step.subgoal_atoms, key=str))
        logging.info(
            "[%s] info-seeking %s(%s): picked params %s with disagreement "
            "%.4f vs first-feasible %.4f (%d/%d feasible in %d draws, "
            "%d banked, %d/%d node budget; per-atom: %s).", run_id,
            step.option.name, objs, _fmt_params(best), best_score,
            first_feasible_score, len(scored), info_n_feasible_target, n_draws,
            len(pool.ranked), pool.spent, max_samples_per_step, per_atom)
        return best

    def sample_fn(idx: int, state: State,
                  rng_: np.random.Generator) -> _Option:
        step = sketch[idx]
        if log_state:
            step_name = (f"{step.option.name}"
                         f"({', '.join(o.name for o in step.objects)})")
            logging.debug(f"[{run_id}]  State before {step_name}:\n"
                          f"{state.pretty_str()}")
        # Info-seeking (when on) owns param selection for eligible steps and
        # folds any LLM-proposed params into its scored candidate pool (see
        # _sample_info_seeking), so the disagreement argmax chooses among
        # {LLM guess} ∪ sampled draws instead of the guess pre-empting it.
        if _info_seeking_applies(step):
            return _sample_info_seeking(step, state, rng_, idx)
        # Plain path: on the first arrival at this step, try the LLM-proposed
        # params (if any) before any sampling. Clipping avoids ground()'s
        # out-of-box ValueError; arity is already validated by the parser.
        if step.initial_params is not None and idx not in _llm_params_tried:
            _llm_params_tried.add(idx)
            box = step.option.params_space
            params = np.clip(np.asarray(step.initial_params, dtype=np.float32),
                             box.low, box.high).astype(np.float32)
            logging.debug("[%s] step %d %s: trying LLM-proposed params %s",
                          run_id, idx, step.option.name, params.tolist())
            return _ground(step, params)
        return _ground(step, _draw_params(step, state, rng_))

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
        # exact failure trajectory. Record the deepest validation failure
        # (unmet subgoal, or unreached task goal at the final step) seen so
        # far along with a consistent snapshot of the prefix. A final-goal
        # failure is at idx==n-1, so its snapshot is the full plan — the
        # experiment we want to execute in reality.
        if (truncate_on_subgoal_fail
                and (fail_reason.startswith("subgoal missing")
                     or fail_reason == "goal not reached")
                and idx > deepest_fail_idx[0]):
            deepest_fail_idx[0] = idx
            deepest_fail_prefix[0] = list(cur_plan[:idx + 1])
        if on_step_fail is not None:
            on_step_fail(idx, cur_plan, fail_reason)

    # One-line eligibility summary: if info-seeking is requested but no
    # step qualifies (a step needs continuous params + a subgoal
    # annotation), the per-step probe silently never fires — say so.
    if info_scorer is not None and info_n_feasible_target > 1:
        eligible = [
            i for i, s in enumerate(sketch) if _info_seeking_applies(s)
        ]
        logging.info(
            "[%s] info-seeking eligible steps: %s of %d (target %d, "
            "node budget %d).", run_id, eligible or "none", n,
            info_n_feasible_target, max_samples_per_step)

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
        step_samples_cumulative=step_samples_cumulative,
        termination_reason=termination_reason,
        elapsed_holder=elapsed_holder,
    )

    # total_samples counts attempts only; pool rollouts are the real
    # model-call cost of info-seeking steps, so surface them alongside.
    pool_note = (f" (+{total_pool_rollouts[0]} info-seeking pool rollouts)"
                 if total_pool_rollouts[0] else "")
    logging.info(
        f"[{run_id}] Refinement {'succeeded' if success else 'failed'}: "
        f"{total_samples} samples for {n} steps{pool_note}.")

    if (truncate_on_subgoal_fail and not success and deepest_fail_idx[0] >= 0):
        snapshot = deepest_fail_prefix[0]
        refined = [p for p in snapshot if p is not None]
        logging.info(f"[{run_id}] Truncating at deepest validation failure "
                     f"(step {deepest_fail_idx[0]}): "
                     f"{len(refined)}/{n} steps in experiment plan.")
        return cast(List[_Option], refined), False, total_samples

    refined = [p for p in plan if p is not None]
    if success:
        return cast(List[_Option], refined), True, total_samples
    return refined, False, total_samples


def _fmt_state_features(state: State) -> str:
    """Compact one-line dump of every object's features.

    Used by ``validate_plan_forward`` to trace how the continuous
    rollout's state drifts step by step.
    """
    parts = []
    for obj in sorted(state, key=lambda o: o.name):
        feats = ", ".join(f"{f}={state.get(obj, f):.4f}"
                          for f in obj.type.feature_names)
        parts.append(f"{obj.name}[{feats}]")
    return " ".join(parts)


@dataclasses.dataclass
class StepOutcome:
    """Result of executing one option in ``execute_plan_forward``.

    ``post_state`` is ``None`` when the option was not initiable or
    raised (no state to continue from). ``failure_reason`` is ``None``
    on a clean step, else the reason (``"not initiable"`` / ``"0
    actions"`` / the option model's last failure / ``"env failure:
    ..."``). ``subgoal_missing`` holds the step's positive subgoal atoms
    that did NOT hold afterwards (only set when a sketch is supplied).
    """
    option: _Option
    pre_state: State
    post_state: Optional[State]
    num_actions: int
    initiable: bool
    failure_reason: Optional[str]
    subgoal_missing: Optional[Set[GroundAtom]]


@dataclasses.dataclass
class ForwardResult:
    """Outcome of executing a grounded plan forward through the option
    model."""
    steps: List[StepOutcome]
    final_state: State
    goal_reached: bool
    # First step that failed to execute (not initiable / 0 actions / env
    # failure), or None if every step executed.
    first_failure_idx: Optional[int]
    # First step whose positive subgoal atoms diverged, or None.
    first_subgoal_divergence_idx: Optional[int]
    # Total low-level actions executed across all options.
    total_actions: int = 0
    # First step index whose post-state satisfies the goal, or None if the
    # goal was never reached along the way.
    goal_step_idx: Optional[int] = None
    # Cumulative low-level actions through ``goal_step_idx`` (the count the
    # real, horizon-capped executor would spend to first reach the goal), or
    # None if the goal was never reached.
    actions_to_goal: Optional[int] = None

    @property
    def executed_all(self) -> bool:
        """True iff every option executed (initiable and >0 actions)."""
        return self.first_failure_idx is None

    @property
    def success(self) -> bool:
        """True iff every option executed AND the goal was reached."""
        return self.executed_all and self.goal_reached

    @property
    def clean_to_goal(self) -> bool:
        """True iff the goal is reached with no hard option failure at or
        before the step that first reaches it.

        Mirrors the real closed-loop executor, which aborts at the first
        failing option: a 0-action / not-initiable failure *after* the goal
        already holds is harmless, but one before it dooms the rollout.
        ``execute_plan_forward`` itself continues past such failures (the
        option model returns an unchanged post-state), so this guards
        against capturing a plan whose goal atoms only hold because forward
        simulation pressed on through a collision the real env would abort
        on.
        """
        if not self.goal_reached or self.goal_step_idx is None:
            return False
        return (self.first_failure_idx is None
                or self.first_failure_idx > self.goal_step_idx)


def execute_plan_forward(
    task: Task,
    plan: List[_Option],
    option_model: _OptionModelBase,
    *,
    predicates: Set[Predicate],
    sketch: Optional[List[SketchStep]] = None,
    on_step: Optional[Callable[[int, StepOutcome], None]] = None,
    stop_on_failure: bool = False,
) -> ForwardResult:
    """Execute a fully-grounded plan step by step through the option model.

    Shared forward-execution core behind ``validate_plan_forward`` (used
    by ``refine_plan_sketch``) and the ``evaluate_option_plan`` tool.
    State carries forward across options — matching how the real env
    executes. Per step it mirrors ``run_backtracking_refinement``'s
    fixed-plan path: check ``initiable``, call
    ``get_next_state_and_num_actions`` (catching
    ``EnvironmentFailure``), treat 0 actions as a failure, and — when a
    sketch is given — check the step's positive ``subgoal_atoms``
    against the post-state. ``on_step(i, outcome)`` is called after each
    step for callers that emit per-step reporting.

    Execution always stops early when a step is not initiable or raises
    (no post-state to continue from). When ``stop_on_failure`` is True it
    *also* stops at a 0-action step — mirroring the real closed-loop
    executor, which aborts at the first failing option rather than
    pressing on. When False (the default), a 0-action step is recorded as
    a failure but execution continues from the model's returned state
    (kept for ``validate_plan_forward``'s per-step diagnostics).
    """
    state = task.init
    steps: List[StepOutcome] = []
    first_failure_idx: Optional[int] = None
    first_div_idx: Optional[int] = None
    total_actions = 0
    goal_step_idx: Optional[int] = None
    actions_to_goal: Optional[int] = None

    for i, option in enumerate(plan):
        pre = state
        initiable = option.initiable(pre)
        post: Optional[State] = None
        num_actions = 0
        failure_reason: Optional[str] = None

        if not initiable:
            failure_reason = "not initiable"
        else:
            try:
                post, num_actions = \
                    option_model.get_next_state_and_num_actions(pre, option)
            except utils.EnvironmentFailure as e:
                failure_reason = f"env failure: {e}"
                post = None
            except Exception as e:  # pylint: disable=broad-except
                failure_reason = f"execution error: {type(e).__name__}: {e}"
                post = None
            else:
                if num_actions == 0:
                    failure_reason = (getattr(option_model,
                                              "last_execution_failure", None)
                                      or "0 actions")

        subgoal_missing: Optional[Set[GroundAtom]] = None
        if post is not None and sketch is not None and i < len(sketch):
            step = sketch[i]
            if step.subgoal_atoms:
                cur_atoms = utils.abstract(post, predicates)
                missing = step.subgoal_atoms - cur_atoms
                if missing:
                    subgoal_missing = missing
                    if first_div_idx is None:
                        first_div_idx = i

        outcome = StepOutcome(option=option,
                              pre_state=pre,
                              post_state=post,
                              num_actions=num_actions,
                              initiable=initiable,
                              failure_reason=failure_reason,
                              subgoal_missing=subgoal_missing)
        steps.append(outcome)
        if on_step is not None:
            on_step(i, outcome)

        if failure_reason is not None and first_failure_idx is None:
            first_failure_idx = i
        if post is None:
            break  # cannot continue without a post-state
        if failure_reason is not None and stop_on_failure:
            break  # mirror the real executor: abort at the first failure
        state = post
        total_actions += num_actions
        # Record when the goal *first* holds, plus the cumulative actions to
        # get there — the budget the real horizon-capped executor would
        # spend. A plan whose goal only holds after more steps than the
        # episode horizon allows is not real-executable.
        if goal_step_idx is None and task.goal_holds(state):
            goal_step_idx = i
            actions_to_goal = total_actions

    return ForwardResult(
        steps=steps,
        final_state=state,
        goal_reached=task.goal_holds(state),
        first_failure_idx=first_failure_idx,
        first_subgoal_divergence_idx=first_div_idx,
        total_actions=total_actions,
        goal_step_idx=goal_step_idx,
        actions_to_goal=actions_to_goal,
    )


def validate_plan_forward(
    task: Task,
    plan: List[_Option],
    option_model: _OptionModelBase,
    *,
    predicates: Set[Predicate],
    sketch: Optional[List[SketchStep]] = None,
    run_id: str = "bilevel",
) -> Tuple[bool, str]:
    """Re-execute a refined plan continuously, checking goal at the end.

    Runs all options sequentially with state carrying forward — matching
    how the real env will execute, and exposing accumulated state drift
    that refinement's per-step resets hide.

    When ``sketch`` is provided, also checks each step's ``subgoal_atoms``
    against the post-state and logs the first divergence with the missing
    atoms. Without ``sketch``, only the final goal is checked.

    Returns ``(success, diagnosis)``. ``diagnosis`` is a one-line summary
    of why validation failed (or ``""`` on success), suitable for surface
    in synthesis-tool output. The full failure context (state features,
    missing atoms, last option model error) is logged at INFO level.

    Single-shot per option (no resampling) — surfaces stochasticity-
    sensitive plans that refinement's resampling hides. Delegates the
    execution to ``execute_plan_forward``; this wrapper adds the INFO
    logging (per-step subgoal divergence, final state) and the one-line
    diagnosis, with priority execution-failure > goal-not-reached >
    subgoal-divergence.
    """
    n = len(plan)
    if n == 0:
        if task.goal_holds(task.init):
            return True, ""
        return False, "empty plan; init state does not satisfy goal"

    if sketch is not None and len(sketch) != n:
        logging.warning(
            "[%s] validate_plan_forward: sketch length %d != plan length %d; "
            "ignoring sketch (no per-step subgoal diagnostics).", run_id,
            len(sketch), n)
        sketch = None

    result = execute_plan_forward(task,
                                  plan,
                                  option_model,
                                  predicates=predicates,
                                  sketch=sketch)

    # Per-step subgoal divergence is a *signal*, not a hard failure (the
    # plan may establish a subgoal earlier, have it temporarily violated,
    # then re-establish it). Log each; remember the first for the diagnosis.
    first_div_msg = ""
    for i, outcome in enumerate(result.steps):
        if not outcome.subgoal_missing or outcome.post_state is None:
            continue
        step = sketch[i] if sketch is not None else None
        missing_strs = sorted(str(a) for a in outcome.subgoal_missing)
        opt_str = (f"{outcome.option.name}"
                   f"({', '.join(o.name for o in outcome.option.objects)})")
        logging.info(
            "[%s] Forward-validate subgoal divergence at step %d (%s):\n"
            "  expected:  %s\n"
            "  missing:   %s\n"
            "  full features: %s", run_id, i, opt_str,
            sorted(str(a)
                   for a in (step.subgoal_atoms or set())) if step else [],
            missing_strs, _fmt_state_features(outcome.post_state))
        if not first_div_msg:
            first_div_msg = (f"step {i} ({opt_str}): subgoals not satisfied "
                             f"after option (missing {missing_strs})")

    # Final-state log — only when every step executed (matches the old
    # behavior, where it ran at the last step's validation).
    if result.executed_all:
        final = result.final_state
        held = sorted(str(a) for a in task.goal if a.holds(final))
        missing = sorted(str(a) for a in task.goal if not a.holds(final))
        abstract_atoms = sorted(
            str(a) for a in utils.abstract(final, predicates))
        logging.info(
            "[%s] Forward-validate FINAL state%s:\n"
            "  goal atoms held:    %s\n"
            "  goal atoms MISSING: %s\n"
            "  abstract state:     %s\n"
            "  full features:      %s\n"
            "  full state:\n%s", run_id,
            " (goal reached)" if result.goal_reached else " (GOAL NOT "
            "REACHED)", held or "(none)", missing or "(none)", abstract_atoms,
            _fmt_state_features(final), final.pretty_str())

    if result.success:
        return True, ""

    # Diagnosis priority: execution failure > goal-not-reached > divergence.
    if result.first_failure_idx is not None:
        i = result.first_failure_idx
        outcome = result.steps[i]
        opt_str = (f"{outcome.option.name}"
                   f"({', '.join(o.name for o in outcome.option.objects)})")
        reason = outcome.failure_reason or "unknown reason"
        logging.info(
            "[%s] Forward-validate option failure at step %d (%s): %s", run_id,
            i, opt_str, reason)
        return False, (f"option execution failed at step {i} ({opt_str}): "
                       f"{reason}")
    if not result.goal_reached:
        goal_missing = sorted(
            str(a) for a in task.goal if not a.holds(result.final_state))
        return False, (f"goal not reached at final step "
                       f"(missing {goal_missing or '(none)'})")
    return False, first_div_msg or "validation failed"


def resolve_refine_timeout(
    timeout: Optional[float],
    n_steps: int,
    *,
    per_step: float,
    minimum: float,
) -> Tuple[float, str]:
    """Resolve a refinement timeout, auto-scaling by sketch length.

    When ``timeout`` is None it auto-scales as
    ``max(minimum, per_step * n_steps)`` so longer sketches get more
    budget. Returns ``(timeout_seconds, source)`` where ``source`` is
    ``"auto"`` or ``"explicit"``. Config defaults are passed in (not read
    from ``CFG``) to keep this module settings-free.
    """
    if timeout is None:
        return float(max(minimum, per_step * n_steps)), "auto"
    return float(timeout), "explicit"


def refine_and_validate_report(
    task: Task,
    sketch: List[SketchStep],
    option_model: _OptionModelBase,
    *,
    predicates: Set[Predicate],
    timeout: float,
    rng: np.random.Generator,
    max_samples_per_step: int,
    check_subgoals: bool,
    log_state: bool = False,
    option_samplers: Optional[Dict[str, OptionSampler]] = None,
    run_id: str = "refine",
    timeout_source: str = "explicit",
    extra_summary_lines: Optional[List[str]] = None,
) -> Tuple[bool, str, List[_Option]]:
    """Refine a sketch, forward-validate on success, return a report.

    Runs ``refine_sketch`` (backtracking search over continuous params)
    and, when refinement succeeds, ``validate_plan_forward`` (continuous
    re-execution). Returns ``(overall_success, human_readable_report,
    plan)`` where ``overall_success`` is True only if both refinement and
    forward validation pass, and ``plan`` is the refined grounded-option
    plan (the longest refined prefix on failure). The report names the
    verdict (SUCCESS / TIMEOUT / SAMPLE_EXHAUSTED /
    FORWARD_VALIDATION_FAILED), per-step sample counts, the stuck step on
    failure, and the forward-validation outcome.

    ``extra_summary_lines`` are appended verbatim after the time line
    (e.g. a caller-specific ``Post-fit SSE`` line). Config-derived knobs
    (``timeout``, ``max_samples_per_step``, ``check_subgoals``,
    ``log_state``) are passed explicitly so this module stays free of
    ``CFG``; callers read them from settings.
    """
    step_samples_cumulative: List[int] = [0] * len(sketch)
    termination_reason: List[str] = []
    elapsed_holder: List[float] = []
    plan, success, n_samples = refine_sketch(
        task,
        sketch,
        option_model,
        predicates=predicates,
        timeout=timeout,
        rng=rng,
        max_samples_per_step=max_samples_per_step,
        check_subgoals=check_subgoals,
        log_state=log_state,
        run_id=run_id,
        step_samples_cumulative=step_samples_cumulative,
        termination_reason=termination_reason,
        elapsed_holder=elapsed_holder,
        option_samplers=option_samplers,
    )

    reason = termination_reason[0] if termination_reason else (
        "success" if success else "exhausted")
    elapsed = elapsed_holder[0] if elapsed_holder else 0.0
    if success:
        verdict = "SUCCESS"
    elif reason == "timeout":
        verdict = "FAILURE: TIMEOUT"
    elif reason == "exhausted":
        verdict = "FAILURE: SAMPLE_EXHAUSTED"
    else:
        verdict = "FAILURE"

    lines = [
        verdict,
        f"  Sketch: {len(sketch)} steps  Refined: {len(plan)} steps  "
        f"Samples: {n_samples} total",
        f"  Per-step samples: {step_samples_cumulative}  "
        f"(cap {max_samples_per_step}/step)",
        f"  Time: {elapsed:.1f}s used / {timeout:.1f}s allotted "
        f"(timeout source: {timeout_source})",
    ]
    if extra_summary_lines:
        lines.extend(extra_summary_lines)
    if not success and len(plan) < len(sketch):
        stuck_idx = len(plan)
        stuck = sketch[stuck_idx]
        objs = ", ".join(f"{o.name}:{o.type.name}" for o in stuck.objects)
        lines.append(f"  Stuck at step {stuck_idx}: "
                     f"{stuck.option.name}({objs})")
        if stuck.subgoal_atoms:
            atoms = ", ".join(str(a) for a in stuck.subgoal_atoms)
            lines.append(f"    subgoals: {atoms}")

    # Forward validation: re-execute the refined plan continuously (state
    # carries forward across all options). Refinement's per-step resets
    # and resampling can mask drift the real env will hit at test time.
    if success:
        try:
            fv_ok, fv_reason = validate_plan_forward(
                task,
                plan,
                option_model,
                predicates=predicates,
                sketch=sketch,
                run_id=run_id,
            )
        except Exception as e:  # pylint: disable=broad-except
            fv_ok = False
            fv_reason = f"forward validation raised: {e}"
        if fv_ok:
            lines.append("  Forward validation: SUCCESS")
        else:
            # Demote the headline verdict: refinement passed but the plan
            # does not survive continuous execution, which is what the
            # real env will see at test time.
            success = False
            lines[0] = "FAILURE: FORWARD_VALIDATION_FAILED"
            lines.append(f"  Forward validation: FAIL — {fv_reason}")
            lines.append(
                "    (Refinement resets state between options and "
                "resamples up to the per-step cap; forward validation "
                "runs the same plan once continuously. A divergence here "
                "means the refined plan does not survive continuous "
                "execution — accumulated drift, or (when the model is "
                "learned) a rule/threshold more permissive than the env's "
                "effective behavior. See the INFO log for the step-by-step "
                "divergence.)")

    return success, "\n".join(lines), plan
