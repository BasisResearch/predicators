"""The explore_python solve-phase exploration tool."""
from typing import Any, Callable, Dict

from predicators.agent_sdk.config import ToolSurfaceConfig
from predicators.agent_sdk.tools.context import ToolContext
from predicators.agent_sdk.tools.python_exec import _make_python_exec_tool
from predicators.agent_sdk.tools.results import _region_syntax_blurb


def _build_exploration_tools(ctx: ToolContext, _text_result: Callable,
                             tool: Callable) -> Dict[str, Any]:
    """Solve-phase ``explore_python`` over the ProbeSim exploration facade.

    The namespace is deliberately tiny (probe facade + numpy): the probe
    reuses the exact machinery behind ``evaluate_option_plan`` (same
    plan grammar, same option-model executor, same renderer) but carries
    no scoring surface - nothing run here can be captured as the answer,
    so it is safe to hand the agent as a freely composable physics
    probe. Built only when the session's config opts in: the
    ``tool_names=None`` legacy surface would otherwise grant every
    default-configured session an in-process exec tool.
    """
    surface_cfg = ToolSurfaceConfig.from_cfg()
    if not surface_cfg.use_explore_python:
        return {}
    # pylint: disable-next=import-outside-toplevel
    from predicators.agent_sdk.probe_api import build_probe_namespace

    # Synthesis sessions install a candidate-simulator provider before
    # opening the session (see ToolContext.probe_option_model_provider);
    # its presence at build time is exactly what flips the probe's
    # semantics, so the description follows it.
    synthesis_probe = ctx.probe_option_model_provider is not None
    if synthesis_probe:
        sim_desc = (
            "`sim` (a ProbeSim over the CANDIDATE simulator: your current "
            "simulator.py with freshly MCMC-fitted params, rebuilt "
            "automatically when the file changes - so probes always "
            "exercise what you just wrote; errors until a loadable "
            "simulator.py exists)")
        reset_desc = (
            "`sim.reset(task_idx, mods=None)` sets the current state "
            "to a train task's init (task_idx is required in this "
            "session), optionally with "
            "feature overrides (`mods={'obj': {'x': 1.05}}`); ")
        submit_desc = (
            "EXPLORATORY "
            "ONLY: nothing run here is captured and no task-evaluator "
            "verdict is computed - validate the simulator itself via "
            "evaluate_plan_refinement.")
    else:
        sim_desc = "`sim` (a ProbeSim over the belief simulator)"
        reset_desc = (
            "`sim.reset(task_idx=None, mods=None)` sets the current state "
            "to a task's init (current task by default), optionally with "
            "feature overrides (`mods={'obj': {'x': 1.05}}`); ")
        submit_desc = (
            "EXPLORATORY "
            "ONLY: nothing run here is captured as your answer and no "
            "task-evaluator verdict is computed - validate and submit the "
            "final plan via evaluate_option_plan from the true initial "
            "state.")
    explore_python = _make_python_exec_tool(
        tool,
        name="explore_python",
        description=(
            "Execute Python code for cheap physics/geometry exploration in "
            "a persistent namespace (variables survive across calls - "
            "define helpers and sweep loops once, reuse them). Available: "
            f"{sim_desc}, `ProbeSim()` "
            "(extra independent instances), `np`. ProbeSim API: "
            f"{reset_desc}"
            "`sim.run(plan_text, render=True, trials=1)` executes an option "
            "plan FROM THE CURRENT "
            "STATE (same grammar as evaluate_option_plan; print the result "
            "for per-step outcomes incl. saved per-step scene-image paths - "
            "view them with the Read tool; pass render=False inside tight "
            "sweep loops) and advances the state; trials=N repeats the plan "
            "N times (fresh physics per trial when available) and returns "
            "the per-trial outcomes + success count WITHOUT advancing the "
            "state - use it for reliability estimates instead of "
            "hand-rolled repeat loops (restore/rerun repeats share solver "
            "state and read optimistic); "
            "`sim.state()` / "
            "`sim.state('obj')` full-precision features; `sim.atoms()`; "
            "`sim.render(label, annotations=None)` saves an image "
            "(returns its path; Read it to view), "
            "optionally overlaying marker/line/rectangle dicts "
            "(`{'type': 'marker', 'position': [x, y, z], 'color': "
            "[r, g, b], 'size': s}`; lines use `from`/`to`, rectangles "
            "`min_corner`/`max_corner`) to check offsets and reference "
            "points visually; `sim.snapshot()` / "
            "`sim.restore(id)` bank and rewind states (use to re-try "
            "different actions from one setup, or resume after a fixed "
            "plan prefix without re-running it); "
            "`sim.refine(sketch_text, timeout=60, require_goal=False, "
            "require_solved=False)` runs "
            "backtracking parameter search FROM THE CURRENT STATE (same "
            "grammar/search as refine_plan_sketch"
            f"{_region_syntax_blurb()}; "
            "success = each step establishes its `-> {subgoals}` "
            "annotation, and the result's Verdict line states what it "
            "certifies) - refine a plan SUFFIX from a snapshot so the "
            "budget goes to the step that matters; the result reports "
            "best-found params even on timeout, per-step sample counts, and "
            "the deepest near-miss. require_solved=True (only from an "
            "unmodified reset() state) additionally requires the task "
            "evaluator to score the final rollout solved=True, rejecting "
            "goal-reaching-but-unscored candidates during the search. "
            "print() output is "
            "returned; oversize output is spilled to "
            "`tool_outputs/explore_python/` (Read/Grep it back). " +
            (f"Each call has a "
             f"{surface_cfg.explore_python_call_timeout:.0f}s "
             "wall-clock limit (checked between sim calls, plus a hard stop "
             "for sim-free code; printed output up to the stop is "
             "returned): budget sweeps accordingly - "
             "prefer coarse-to-fine over exhaustive grids, and print "
             "intermediate bests so partial results survive a stop. "
             if surface_cfg.explore_python_call_timeout > 0
             and not synthesis_probe else "") + f"{submit_desc}"),
        exec_ns=build_probe_namespace(ctx),
        sandbox_dir=ctx.sandbox_dir,
        text_result=_text_result,
        budget_ctx=ctx,
    )
    return {"explore_python": explore_python}
