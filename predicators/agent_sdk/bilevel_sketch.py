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

    experiment_section = ""
    if experiment_guidance:
        experiment_section = (f"\n## Experiment Guidance\n"
                              f"{experiment_guidance}\n")

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

    prompt = f"""You are solving a task. \
Generate a plan sketch to achieve the goal.
{goal_nl_section}{goal_atoms_section}{experiment_section}
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

Output the plan sketch with one option per line in this format:
  OptionName(obj1:type1, obj2:type2) -> \
{{Pred(obj1:type1), Pred2(obj1:type1, obj2:type2)}}
  Wait(robot:Robot) -> {{Boiled(water:water_type)}}
  Wait(robot:Robot) -> {{NOT Touching(a:block, b:block)}}

Always use typed references (obj:type) in both option arguments AND subgoal \
atoms. If you omit `-> {{atoms}}` on a step, the search only checks that the \
option executed (non-zero actions) and execution monitoring is blind there — \
omit it only when no available predicate can express the step's effect.

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
        # Score of the first feasible draw — what plain (non-info-seeking)
        # backtracking would have accepted; logged as the baseline so a run
        # shows what boundary-probing bought over greedy first-feasible.
        first_feasible_score: Optional[float] = None
        first_candidate: Optional[_Option] = None
        n_draws = 0
        while len(scored) < info_n_feasible_target and n_draws < draw_cap:
            grounded = _ground(step, _draw_params(step, state, rng_))
            n_draws += 1
            if first_candidate is None:
                first_candidate = grounded
            if not grounded.initiable(state):
                continue
            try:
                nxt, num_actions = \
                    option_model.get_next_state_and_num_actions(
                        state, grounded)
            except Exception:  # pylint: disable=broad-except
                # Scoring rollout is best-effort; a model failure on this
                # candidate just removes it from contention.
                continue
            if num_actions == 0:
                continue
            post_atoms = utils.abstract(nxt, predicates)
            if not step.subgoal_atoms.issubset(post_atoms):
                continue  # infeasible: subgoal not established
            score = info_scorer(nxt, step.subgoal_atoms)
            scored.append((score, grounded))
            if first_feasible_score is None:
                first_feasible_score = score
            if score > best_score:
                best_score = score
                best_nxt = nxt
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
        if _info_seeking_applies(step):
            return _sample_info_seeking(step, state, rng_, idx)
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

    Differences from ``refine_sketch``:
      * ``max_tries=[1]`` per step — single shot at each option, no
        backtracking. Surfaces stochasticity-sensitive plans that
        refinement's resampling hides.
      * ``rng=np.random.default_rng(0)`` — sample_fn ignores it anyway
        (returns ``plan[i]``).
      * Per-step subgoal logging when ``sketch`` is given.
      * Disables the refinement progress bar so per-step DEBUG logs from
        ``run_backtracking_refinement`` remain visible.
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

    diagnosis_holder: List[str] = [""]

    def sample_fn(i: int, _s: State, _r: np.random.Generator) -> _Option:
        return plan[i]

    def _log_subgoal_divergence(i: int, post: State,
                                step: SketchStep) -> Optional[str]:
        """If ``step.subgoal_atoms`` aren't all in ``post``, log + return a
        one-line summary of what's missing; else return None."""
        if step.subgoal_atoms is None or not step.subgoal_atoms:
            return None
        cur_atoms = utils.abstract(post, predicates)
        missing = step.subgoal_atoms - cur_atoms
        if not missing:
            return None
        missing_strs = sorted(str(a) for a in missing)
        objs_str = ", ".join(o.name for o in plan[i].objects)
        opt_str = f"{plan[i].name}({objs_str})"
        logging.info(
            "[%s] Forward-validate subgoal divergence at step %d (%s):\n"
            "  expected:  %s\n"
            "  missing:   %s\n"
            "  full features: %s", run_id, i, opt_str,
            sorted(str(a) for a in step.subgoal_atoms), missing_strs,
            _fmt_state_features(post))
        return (f"step {i} ({opt_str}): subgoals not satisfied after "
                f"option (missing {missing_strs})")

    def validate_fn(i: int, _pre: State, _opt: _Option, post: State,
                    _n: int) -> Tuple[bool, str]:
        # Per-step subgoal divergence is a *signal*, not a hard failure
        # (the refined plan may have established a subgoal earlier and
        # had it temporarily violated then re-established). We capture
        # the first divergence as the leading-edge diagnosis but keep
        # going so we still get the final-state log.
        if sketch is not None:
            div = _log_subgoal_divergence(i, post, sketch[i])
            if div is not None and not diagnosis_holder[0]:
                diagnosis_holder[0] = div

        if i == n - 1:
            goal_ok = task.goal_holds(post)
            held = sorted(str(a) for a in task.goal if a.holds(post))
            missing = sorted(str(a) for a in task.goal if not a.holds(post))
            abstract_atoms = sorted(
                str(a) for a in utils.abstract(post, predicates))
            logging.info(
                "[%s] Forward-validate FINAL state%s:\n"
                "  goal atoms held:    %s\n"
                "  goal atoms MISSING: %s\n"
                "  abstract state:     %s\n"
                "  full features:      %s\n"
                "  full state:\n%s", run_id,
                " (goal reached)" if goal_ok else " (GOAL NOT REACHED)", held
                or "(none)", missing or "(none)", abstract_atoms,
                _fmt_state_features(post), post.pretty_str())
            if not goal_ok:
                # Final-state goal failure wins over any earlier subgoal
                # divergence as the headline reason.
                diagnosis_holder[0] = (f"goal not reached at final step "
                                       f"(missing {missing or '(none)'})")
                return False, "goal not reached"
        return True, ""

    # progress_bar=False keeps INFO/DEBUG logs from
    # run_backtracking_refinement (the "Step X/N FAIL: <reason>" lines)
    # visible — critical for diagnosing why an option's
    # get_next_state_and_num_actions returned 0 actions.
    plan_result, success, _ = run_backtracking_refinement(
        init_state=task.init,
        option_model=option_model,
        n_steps=n,
        max_tries=[1] * n,
        sample_fn=sample_fn,
        validate_fn=validate_fn,
        rng=np.random.default_rng(0),
        timeout=float('inf'),
        progress_bar=False,
    )

    if success:
        return True, ""

    # Validation reached `success=False` for one of:
    #   1. validate_fn returned False at the final step (goal not reached)
    #   2. an earlier step's option failed (initiable=False, 0 actions,
    #      or env failure) — run_backtracking_refinement backtracks until
    #      cur_idx<0 with max_tries=1
    # Identify which by checking how far the plan progressed.
    completed = sum(1 for p in plan_result if p is not None)
    if completed < n and not diagnosis_holder[0]:
        # Failure happened during option execution at step `completed`.
        # Pull whatever the option model recorded as the last failure
        # reason so the caller knows it's an execution problem, not a
        # subgoal-divergence one.
        last_err = getattr(option_model, "last_execution_failure", None)
        opt = plan[completed]
        opt_str = f"{opt.name}({', '.join(o.name for o in opt.objects)})"
        diagnosis_holder[0] = (f"option execution failed at step "
                               f"{completed} ({opt_str}): "
                               f"{last_err or 'unknown reason'}")
        logging.info(
            "[%s] Forward-validate option failure at step %d (%s): %s", run_id,
            completed, opt_str, last_err or "unknown reason")

    return False, diagnosis_holder[0] or "validation failed"
