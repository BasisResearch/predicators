"""Synthesis-time validation hooks for the agent sim-learning approach.

These helpers run inside an active synthesis-agent session: they need
approach state (base env, train tasks, predicates, options) but never
re-enter the agent — no sketch-prompt query, no new session — so they
can be invoked from a synthesis tool without disturbing the live
session's prompt or tool set. They live here (rather than on the
approach class) to keep the approach module focused on orchestration and
to group them with the other ``code_sim_learning`` simulation / fitting
primitives.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from predicators.code_sim_learning.training import ParamSpec
from predicators.code_sim_learning.utils import LearnedSimulator, apply_rules
from predicators.settings import CFG
from predicators.structs import Action, State, Task

logger = logging.getLogger(__name__)


def run_refinement_for_synthesis(
    approach: Any,
    rules: List,
    specs: List[ParamSpec],
    process_features: Dict[str, List[str]],
    base_pred_triples: List[Tuple[State, Action, State]],
    task_idx: int,
    timeout: Optional[float] = None,
    plan_text: str = "",
) -> str:
    """Validate that the candidate simulator supports plan refinement.

    MCMC-fits parameters from ``specs``, builds a combined option
    model from ``rules`` + the fitted params, parses ``plan_text``
    into a sketch via ``bilevel_sketch.parse_sketch_from_text``, and
    runs ``bilevel_sketch.refine_sketch`` on it. Always fits before
    refinement: the candidate's deployed behaviour is the *fitted*
    simulator, so refining against init_value params would test the
    wrong model.

    ``timeout`` is wall-clock seconds for refinement only (MCMC
    fitting is not subject to it). When ``None``, it auto-scales with
    sketch length:
    ``max(CFG.agent_bilevel_refinement_timeout_min,
         CFG.agent_bilevel_refinement_timeout_per_step * len(sketch))``
    so longer plans automatically get more budget.

    Returns a human-readable report. On failure the report includes a
    termination reason (``timeout`` vs ``exhausted``), per-step
    cumulative sample counts, wall-clock used vs allotted, and a hint
    on whether to raise the timeout or revisit the rules. The hint
    branches on whether the stuck step exhausted its per-step sample
    cap (rule problem) or not (likely budget problem).
    """
    # pylint: disable=import-outside-toplevel,protected-access
    from predicators.agent_sdk import bilevel_sketch

    if task_idx < 0 or task_idx >= len(approach._train_tasks):
        return (f"Error: task_idx {task_idx} out of range "
                f"[0, {len(approach._train_tasks)}).")

    try:
        params, fit_sse = approach._fit_parameters(rules, specs,
                                                   base_pred_triples,
                                                   process_features)
    except Exception as e:  # pylint: disable=broad-except
        return f"Error: param fitting failed:\n{e}"

    learned = LearnedSimulator(
        step_fn=lambda s, _r=rules, _p=params:  # type: ignore[misc]
        apply_rules(s, _r, _p),
        name="agent_in_session")
    combined_sim = approach._build_combined_simulator(learned)
    candidate_om = approach._build_option_model(combined_sim)

    if not plan_text or not plan_text.strip():
        return ("Error: `plan` is required. Pass an option-skeleton plan "
                "(one option call per line, typed `obj:type` references, "
                "every argument supplied) — there is no oracle/file "
                "fallback. See the tool description for the format.")

    task = approach._train_tasks[task_idx]
    try:
        sketch = bilevel_sketch.parse_sketch_from_text(
            plan_text.strip(),
            task,
            predicates=approach._get_all_predicates(),
            options=approach._get_all_options(),
            types=approach._types,
        )
    except Exception as e:  # pylint: disable=broad-except
        return f"Error: could not parse plan:\n{e}"
    if not sketch:
        return ("Error: parsed empty plan sketch from `plan`. Check that "
                "every line names a known option with typed `obj:type` "
                "arguments matching what the inspect tools report.")

    if timeout is None:
        timeout = float(
            max(CFG.agent_bilevel_refinement_timeout_min,
                CFG.agent_bilevel_refinement_timeout_per_step * len(sketch)))
        timeout_source = "auto"
    else:
        timeout = float(timeout)
        timeout_source = "explicit"
    assert timeout is not None

    logger.info(
        "Refining plan sketch (task %d, %d steps, timeout=%.0fs/%s):",
        task_idx, len(sketch), timeout, timeout_source)
    for i, step in enumerate(sketch):
        objs = ", ".join(f"{o.name}:{o.type.name}" for o in step.objects)
        line = f"  {i}: {step.option.name}({objs})"
        if step.subgoal_atoms:
            atoms = ", ".join(str(a) for a in step.subgoal_atoms)
            line += f"  [subgoals: {atoms}]"
        logger.info(line)

    step_samples_cumulative: List[int] = [0] * len(sketch)
    termination_reason: List[str] = []
    elapsed_holder: List[float] = []
    plan, success, n_samples = bilevel_sketch.refine_sketch(
        task,
        sketch,
        candidate_om,
        predicates=approach._get_all_predicates(),
        timeout=timeout,
        rng=np.random.default_rng(CFG.seed),
        max_samples_per_step=CFG.agent_bilevel_max_samples_per_step,
        check_subgoals=CFG.agent_bilevel_check_subgoals,
        log_state=CFG.agent_bilevel_log_state,
        run_id=f"{getattr(approach, '_run_id', 'sim_learn')}_validate",
        step_samples_cumulative=step_samples_cumulative,
        termination_reason=termination_reason,
        elapsed_holder=elapsed_holder,
    )

    reason = termination_reason[0] if termination_reason else (
        "success" if success else "exhausted")
    elapsed = elapsed_holder[0] if elapsed_holder else 0.0
    cap = CFG.agent_bilevel_max_samples_per_step
    if success:
        verdict = "SUCCESS"
    elif reason == "timeout":
        verdict = "FAILURE: TIMEOUT"
    elif reason == "exhausted":
        verdict = "FAILURE: SAMPLE_EXHAUSTED"
    else:
        verdict = "FAILURE"

    lines = [
        f"Task {task_idx}: {verdict}",
        f"  Sketch: {len(sketch)} steps  Refined: {len(plan)} steps  "
        f"Samples: {n_samples} total",
        f"  Per-step samples: {step_samples_cumulative}  (cap "
        f"{cap}/step)",
        f"  Time: {elapsed:.1f}s used / {timeout:.1f}s allotted "
        f"(timeout source: {timeout_source})",
        f"  Post-fit SSE: {fit_sse:.6f}",
    ]
    if not success and len(plan) < len(sketch):
        stuck_idx = len(plan)
        stuck = sketch[stuck_idx]
        objs = ", ".join(f"{o.name}:{o.type.name}" for o in stuck.objects)
        lines.append(f"  Stuck at step {stuck_idx}: "
                     f"{stuck.option.name}({objs})")
        if stuck.subgoal_atoms:
            atoms = ", ".join(str(a) for a in stuck.subgoal_atoms)
            lines.append(f"    subgoals: {atoms}")
    return "\n".join(lines)


def get_or_build_sketch(
    approach: Any,
    task: Task,
    plan_text: str = "",
) -> Tuple[List, str]:
    """Return ``(sketch, source_label)`` for ``task``.

    Resolution order (first non-empty wins):
      1. ``plan_text`` — agent-proposed plan, parsed via
         ``parse_sketch_from_text``. This is the primary path.
      2. ``CFG.agent_bilevel_plan_sketch_file`` — fall-through for
         pre-baked sketches.
      3. Oracle task planning over the env's GT NSRTs — last-resort
         cold-start fallback.
    """
    # pylint: disable=import-outside-toplevel,protected-access
    from predicators.agent_sdk import bilevel_sketch
    from predicators.ground_truth_models import get_gt_nsrts
    from predicators.planning import run_task_plan_once

    if plan_text and plan_text.strip():
        sketch_from_agent = bilevel_sketch.parse_sketch_from_text(
            plan_text.strip(),
            task,
            predicates=approach._get_all_predicates(),
            options=approach._get_all_options(),
            types=approach._types,
        )
        return sketch_from_agent, "agent_proposed"

    sketch_file = CFG.agent_bilevel_plan_sketch_file
    if sketch_file:
        with open(sketch_file, "r", encoding="utf-8") as f:
            file_text = f.read().strip()
        sketch_from_file = bilevel_sketch.parse_sketch_from_text(
            file_text,
            task,
            predicates=approach._get_all_predicates(),
            options=approach._get_all_options(),
            types=approach._types,
        )
        return sketch_from_file, f"file:{sketch_file}"

    nsrts = get_gt_nsrts(CFG.env, approach._initial_predicates,
                         approach._initial_options)
    # Symbolic-only; 10 s is plenty for any env with GT NSRTs and
    # decouples this step from the refinement timeout.
    plan, atoms_seq, _ = run_task_plan_once(
        task,
        nsrts,
        approach._initial_predicates,
        approach._types,
        timeout=10.0,
        seed=CFG.seed,
        task_planning_heuristic=CFG.sesame_task_planning_heuristic,
    )
    sketch: List = []
    for i, gnsrt in enumerate(plan):
        delta = (atoms_seq[i + 1] -
                 atoms_seq[i] if i + 1 < len(atoms_seq) else set())
        sketch.append(
            bilevel_sketch.SketchStep(
                option=gnsrt.option,
                objects=list(gnsrt.option_objs),
                subgoal_atoms=delta if delta else None,
            ))
    return sketch, "oracle_task_plan"
