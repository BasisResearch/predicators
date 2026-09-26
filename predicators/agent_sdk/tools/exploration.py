"""The solve-phase ``run_python`` tool over the belief probe."""
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from predicators.agent_sdk.config import ToolSurfaceConfig
from predicators.agent_sdk.tools.context import ToolContext
from predicators.agent_sdk.tools.python_exec import _make_python_exec_tool
from predicators.agent_sdk.tools.results import _region_syntax_blurb


@dataclass(frozen=True)
class ProbeSurface:
    """What the ``sim`` probe of a session can do, for its description.

    The tool description must offer exactly what the probe accepts: a
    comparison arm that refuses fitting, model edits, alternative
    parameter values or uncertainty sweeps must not be invited to use
    them. ``fit``: ``sim.fit``, ``fit_params`` and ``sweep_params``.
    ``edit_model``: the agent writes ``simulator.py`` (False when the
    harness supplies it). ``sealed``: the agent's model is sealed at the
    first real action (the zero-shot arm). ``alt_params``: scoring
    alternative parameter values (``validate(params=...)``,
    ``residuals(phys_params=...)``). ``uncertainty``: belief draws,
    ``sim.belief``, ``sim.suggest_probes`` and physics sweeps.
    """
    fit: bool = True
    edit_model: bool = True
    sealed: bool = False
    alt_params: bool = True
    uncertainty: bool = True


def belief_probe_blurb(synthesis_probe: bool,
                       surface: Optional[ProbeSurface] = None) -> str:
    """The BeliefProbe surface description, shared by every prompt/tool surface
    that offers the probe.

    Solve sessions bind it as ``sim`` in ``run_python``'s namespace over
    the deployed belief model; synthesis sessions bind the same facade
    over the candidate simulator. One renderer so the two descriptions
    cannot drift. ``synthesis_probe`` selects the candidate-simulator
    wording (task_idx-required resets, the model file, and the
    validation protocol); ``surface`` trims the calls a comparison arm
    refuses (:class:`ProbeSurface`).
    """
    surface = surface or ProbeSurface()
    if synthesis_probe:
        if not surface.edit_model:
            sim_desc = (
                "`sim` (a BeliefProbe over the SUPPLIED simulator, fixed "
                "for the run and not exposed as source; no call fits or "
                "changes its parameters)")
        elif surface.sealed:
            sim_desc = (
                "`sim` (a BeliefProbe over your simulator.py, rebuilt "
                "automatically when the file changes until your first real "
                "action seals it; afterwards it runs the sealed code at its "
                "declared values, and no call fits or changes them; errors "
                "until a loadable simulator.py exists)")
        elif not surface.fit:
            sim_desc = (
                "`sim` (a BeliefProbe over the CANDIDATE simulator: your "
                "current simulator.py, rebuilt automatically when the file "
                "changes, at the values written in its declarations - the "
                "harness fits nothing; errors until a loadable simulator.py "
                "exists)")
        else:
            sim_desc = (
                "`sim` (a BeliefProbe over the CANDIDATE simulator: your "
                "current simulator.py, rebuilt automatically when the file "
                "changes, at the params of your last `sim.fit()` - it never "
                "fits on its own, and results carry a PARAMS UNFITTED "
                "notice until you fit the current file; errors until a "
                "loadable simulator.py exists)")
        reset_desc = (
            "`sim.reset(task_idx, mods=None)` sets the current state "
            "to a train task's init (task_idx is required in this "
            "session), optionally with "
            "feature overrides (`mods={'obj': {'x': 1.05}}`); ")
        task_desc = ("`sim.task(task_idx)` describes a train task (goal, "
                     "objects, initial atoms and state) without touching "
                     "the current state; ")
        if surface.fit:
            task_desc += (
                "`sim.fit(traj_idxs=None, fixed=None)` fits "
                "PARAM_SPECS (loaded fresh from simulator.py) against "
                "the recorded data and returns the report (SSE "
                "init->fit, fitted values, identifiability when "
                "PHYSICAL_PARAM_SPECS is declared). No arguments = the "
                "CANONICAL fit the probe deploys (system-ID values "
                "applied to the planning env); traj_idxs (subset of "
                "trajectories; on the system-ID path a "
                "cross-trajectory consistency check) or fixed "
                "({name: value} pins; rule params only) = "
                "EXPLORATORY diagnostic, nothing published. Expensive "
                "- call after meaningful rule edits, not in loops; "
                "`sim.validate(traj_idxs=None, params=None)` replays "
                "recorded actions at deployed values without fitting or "
                "dropping recordings; explicit params are diagnostic "
                "overrides, never deployed. Use it to compare model "
                "structures on identical data; ")
        elif surface.alt_params:
            task_desc += (
                "`sim.validate(traj_idxs=None, params=None)` replays "
                "recorded actions at the declared values without dropping "
                "recordings; explicit params are diagnostic overrides, "
                "never deployed. Use it to compare candidate values or "
                "model structures on identical data; ")
        else:
            task_desc += (
                "`sim.validate(traj_idxs=None)` replays recorded actions "
                "under the fixed model without dropping recordings; ")
        fit_arg = ", fit_params=False" if surface.fit else ""
        purpose = ("the fast inner loop for finding WHICH rule to fix"
                   if surface.edit_model else
                   "the fast view of where the fixed model disagrees with "
                   "the recordings")
        task_desc += (
            "`sim.residuals(max_transitions=100, abs_tol=1e-4, "
            f"rel_tol=1e-3, num_worst_examples=3{fit_arg})` per-feature "
            "residual report for the " +
            ("current simulator.py rules"
             if surface.edit_model else "supplied model") + " (mismatch "
            "counts, mean/max abs error, vs-no-rule-baseline improvement, "
            f"worst-N example transitions) - {purpose}. It is "
            "teacher-forced (each step predicted from the RECORDED state), "
            "so it CANNOT rule out a mis-set physical parameter: "
            "compounding errors reset every step. ")
        rollout_args = ["rollout=True"]
        if surface.fit:
            rollout_args.append("sweep_params=None")
        if surface.alt_params:
            rollout_args.append("phys_params=None")
        if surface.fit:
            rollout_args.append("sweep_num_points=6")
        task_desc += (f"`sim.residuals({', '.join(rollout_args)})` is the "
                      "OPEN-LOOP counterpart: replays each recorded "
                      "trajectory's actions free-running and reports the "
                      "divergence at the current baselines")
        if surface.fit:
            task_desc += (
                ". sweep_params=[names] (or 'all') additionally sweeps "
                "each named env-registry physical parameter across "
                "its plausible range ('this data is explained Nx "
                "better at a different friction')")
        if surface.alt_params:
            task_desc += (
                "; phys_params={name: value} " +
                ("instead " if surface.fit else "") +
                "scores ONE hypothesized point and reports the SSE ratio "
                "vs the baseline (the cheap primitive for your own targeted "
                "sweeps)")
        if surface.fit:
            task_desc += (
                ". A sweep is slow (one fresh-env rollout per "
                "candidate per segment, minutes for the full "
                "registry) but it is the ONLY residual view that can "
                "see physical-parameter error - run one (e.g. "
                "sweep_params='all') BEFORE deciding whether to "
                "declare PHYSICAL_PARAM_SPECS, in either direction")
        task_desc += "; "
    else:
        sim_desc = "`sim` (a BeliefProbe over the belief simulator)"
        reset_desc = (
            "`sim.reset(task_idx=None, mods=None)` sets the current state "
            "to a task's init (current task by default), optionally with "
            "feature overrides (`mods={'obj': {'x': 1.05}}`); ")
        task_desc = ("`sim.task(task_idx=None)` describes a task - goal, "
                     "objects, initial atoms and state (current task by "
                     "default) - without touching the current state; ")
    # pylint: disable-next=import-outside-toplevel
    from predicators.settings import CFG
    joint = bool(surface.uncertainty) and int(CFG.belief_joint_draws) > 0
    if joint:
        belief_desc = ("`sim.belief()` lists the state belief with the atoms "
                       "it is unsure about; ")
        probes_desc = (
            "`sim.suggest_probes(plan_text, max_draws=8, top_k=3)` rolls "
            "your plan out on the joint draws of the belief and, per `-> "
            "{subgoals}`-annotated step with continuous params, ranks "
            "feasible alternatives by the information a noisy reading of "
            "the subgoals carries about the parameters (advice only: what "
            "you execute runs as written); ")
        run_desc = (
            "`sim.run(plan_text, render=True, draws=None, contacts=False, "
            "physics_sweep=False, seed=None)` rehearses an option plan "
            "FROM THE CURRENT STATE (same grammar as submit_plan) on the "
            f"{int(CFG.belief_joint_draws)} joint draws of the belief - a "
            "parameter draw, a state draw and the memory those parameters "
            "imply, each on a fresh env with its own planner seed - and "
            "reports the success estimate P-hat with its standard error, "
            "scored by the TASK EVALUATOR on the episode so far followed "
            "by each draw's rollout (reaching the goal atoms is NOT the "
            "same as being scored a solve), each draw's outcome and the "
            "parameter ranges on which draws fail, then a step-by-step "
            "rollout from the belief mean at the most likely parameters "
            "with contacts and saved per-step scene-image paths (view them "
            "with the Read tool), which advances the state; `-> "
            "{subgoals}` annotations are CHECKED in that rollout; draws=0 "
            "runs only the rollout from the current state (pass "
            "render=False inside tight sweep loops); physics_sweep=True "
            "stress-tests the plan at the ends of each parameter's 95% "
            "interval (it locates failure boundaries and carries no "
            f"probability); {belief_desc}"
            "contacts=True with draws=0 reports, per ")
        refine_desc = (
            "`sim.refine(sketch_text, timeout=60, require_goal=False, "
            "require_solved=False)` runs "
            "backtracking parameter search from the belief mean (same "
            "grammar as submit_plan"
            f"{_region_syntax_blurb()}; "
            "success = each step establishes its `-> {subgoals}` "
            "annotation), scores up to "
            f"{int(CFG.belief_refine_candidates)} successful proposals on "
            "the common joint draws, and returns the best with its P-hat "
            "on fresh draws, an estimate its selection does not bias; the "
            "result reports per-step sample counts and the deepest "
            "near-miss.")
    else:
        belief_desc = (
            "under a declared observation-noise channel "
            "belief_draws=K rolls the plan from K draws of where the "
            "objects may really be (the belief the last observation "
            "showed) and `sim.belief()` lists that belief with the atoms "
            "it is unsure about; " if surface.uncertainty else "")
        probes_desc = (
            "`sim.suggest_probes(sketch_text, max_draws=20, top_k=3)` rolls "
            "your sketch forward on your own parameters and, per `-> "
            "{subgoals}`-annotated step with continuous params, ranks "
            "feasible alternatives by the learned model's ensemble "
            "disagreement on those atoms (advice only: what you submit "
            "runs as written); " if surface.uncertainty else "")
        run_desc = (
            "`sim.run(plan_text, render=True, trials=1, solved=False, "
            "contacts=False)` executes an option "
            "plan FROM THE CURRENT "
            "STATE (same grammar as submit_plan; print the result "
            "for per-step outcomes incl. saved per-step scene-image paths - "
            "view them with the Read tool; pass render=False inside tight "
            "sweep loops) and advances the state; `-> {subgoals}` "
            "annotations are CHECKED - each step's report lists annotated "
            "atoms that did not hold in its post-state, so one continuous "
            "run of a refined plan is the forward-validation pass (a "
            "refine-pass that diverges here means a rule is more "
            "permissive than the env); trials=N repeats the plan "
            "N times (fresh physics per trial when available) and returns "
            "the per-trial outcomes + success count WITHOUT advancing the "
            "state - use it for reliability estimates instead of "
            "hand-rolled repeat loops (restore/rerun repeats share solver "
            "state and read optimistic); solved=True (trials>=2, from an "
            "unmodified reset() state) also scores each trial with the "
            "TASK EVALUATOR (per-trial solved/reward) - reaching the goal "
            "atoms is NOT the same as being scored a solve, so check this "
            f"BEFORE submitting; {belief_desc}"
            "contacts=True (single run) reports, per ")
        refine_desc = (
            "`sim.refine(sketch_text, timeout=60, require_goal=False, "
            "require_solved=False)` runs "
            "backtracking parameter search FROM THE CURRENT STATE (same "
            "grammar as submit_plan"
            f"{_region_syntax_blurb()}; "
            "success = each step establishes its `-> {subgoals}` "
            "annotation, and the result's Verdict line states what it "
            "certifies) - refine a plan SUFFIX from a snapshot so the "
            "budget goes to the step that matters; the result reports "
            "best-found params even on timeout, per-step sample counts, and "
            "the deepest near-miss. require_solved=True (only from an "
            "unmodified reset() state) additionally requires the task "
            "evaluator to score the final rollout solved=True, rejecting "
            "goal-reaching-but-unscored candidates during the search.")
    keep_working = "edit/fit/think" if surface.fit else "think/plan"
    return (f"{sim_desc}, `BeliefProbe()` "
            "(extra independent instances). BeliefProbe API: "
            f"{reset_desc}{task_desc}"
            f"{run_desc}"
            "step, which robot links touched which objects and which "
            "object pairs touched, with action spans - use it to verify "
            "WHAT caused motion (e.g. an intended push vs. the arm "
            "brushing the scene); "
            "`sim.run_async(plan_text, ...)` launches the same run in a "
            "forked child and returns a handle IMMEDIATELY (the session "
            "state does NOT advance; rendering unavailable) - a plain "
            "python for-loop over sim.run executes ONE rollout at a "
            "time, so for independent rollouts (plan variants, seeds, "
            "mods sweeps) launch them all with run_async, keep working "
            f"({keep_working} - handles survive across calls), then "
            "`sim.gather(handles, timeout=None)` waits and summarizes "
            "(read each handle's `.result`/`.error`; ~Nx faster at N "
            "workers) - wait with gather, NOT a `.done()` sleep loop: "
            "gather bounds the wait and flags stale results. Results "
            "reflect the model AS OF launch - gather "
            "flags results that ran under an older "
            f"{'simulator.py' if surface.edit_model else 'model state'}; "
            "adaptive loops (next params chosen from the last result) "
            "stay sequential by nature - use plain sim.run there; "
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
            f"plan prefix without re-running it); {probes_desc}"
            f"{refine_desc}")


def _build_exploration_tools(ctx: ToolContext, _text_result: Callable,
                             tool: Callable) -> Dict[str, Any]:
    """Solve-phase ``run_python`` over the BeliefProbe exploration facade.

    The namespace is the probe facade, numpy, and the collected real
    trajectories as read-only evidence (see ``build_probe_namespace`` -
    nothing evaluator-shaped beyond the probe's gated paths): the probe
    reuses the exact machinery behind ``submit_plan`` (same
    plan grammar, same option-model executor, same renderer) but
    carries no scoring surface - nothing run here can be captured as
    the answer, so it is safe to hand the agent as a freely composable
    physics probe. Synthesis sessions attach their own ``run_python``
    (fit data + the candidate-simulator probe in one namespace; see
    ``_get_synthesis_tool_names``), and ``create_mcp_tools`` skips this
    instance when one is attached.
    """
    surface_cfg = ToolSurfaceConfig.from_cfg()
    # pylint: disable-next=import-outside-toplevel
    from predicators.agent_sdk.belief_probe import build_probe_namespace

    submit_desc = (
        "EXPLORATORY "
        "ONLY: nothing run here is captured as your answer - preview "
        "the evaluator's verdict with sim.run(solved=True), then "
        "validate and submit the final plan via submit_plan "
        "from the true initial state.")
    run_python = _make_python_exec_tool(
        tool,
        name="run_python",
        description=(
            "Execute Python code (`code`, or `path` to a .py file you "
            "wrote in the sandbox) for cheap physics/geometry exploration "
            "in a persistent namespace (variables survive across calls - "
            "define helpers and sweep loops once, reuse them). Available: " +
            belief_probe_blurb(synthesis_probe=False) +
            " Also bound: `np`; `trajectories` (the recorded REAL "
            "offline+online trajectories, read-only evidence - use them to "
            "check the belief model against what actually happened; each "
            "has `is_demo`, `train_task_idx`, `states`, `actions`) and "
            "`describe_trajectory(traj_idx, include_states=True, "
            "include_atoms=False, max_timesteps=10)` for a per-timestep "
            "digest of one of them. "
            "print() output is "
            "returned; oversize output is spilled to "
            "`tool_outputs/run_python/` (Read/Grep it back). " +
            (f"Each call has a "
             f"{surface_cfg.python_call_timeout:.0f}s "
             "wall-clock limit (checked between sim calls, plus a hard stop "
             "for sim-free code; printed output up to the stop is "
             "returned): budget sweeps accordingly - "
             "prefer coarse-to-fine over exhaustive grids, and print "
             "intermediate bests so partial results survive a stop. " if
             surface_cfg.python_call_timeout > 0 else "") + f"{submit_desc}"),
        exec_ns=build_probe_namespace(ctx),
        sandbox_dir=ctx.sandbox_dir,
        text_result=_text_result,
        budget_ctx=ctx,
    )
    return {"run_python": run_python}
