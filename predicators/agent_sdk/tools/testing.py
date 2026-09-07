"""Testing tools, including the submit_plan capture surface."""
import contextlib
import functools
import logging
import os
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np

from predicators import utils
from predicators.agent_sdk import bilevel_sketch
from predicators.agent_sdk.config import RefinementConfig, ValidationConfig
from predicators.agent_sdk.parallel_rollouts import \
    prefetch_parallel as _prefetch_parallel
from predicators.agent_sdk.tools.budget import _budget_footer
from predicators.agent_sdk.tools.capture import BestEffortReason, \
    CaptureDecision, _decide_capture
from predicators.agent_sdk.tools.context import ToolContext, \
    _capture_task_key, decorrelated_rollout_seed
from predicators.agent_sdk.tools.results import _error_result
from predicators.agent_sdk.tools.scene import format_object_poses, \
    render_pybullet_image
from predicators.agent_sdk.tools.tasks import _resolve_task
from predicators.agent_sdk.tools.verdicts import _EvalStateCollector, \
    _format_evaluator_verdict, _resolve_task_evaluator, _sandbox_base, \
    evaluate_states_with, load_ground_sampler_fns
from predicators.settings import CFG
from predicators.structs import GroundAtom, State, Task

# Ceiling on agent-requested validation rollouts per submission
# (validation_rollouts): the agent pays for rollouts from its budget, but
# a typo'd request should not silently torch it.
_MAX_REQUESTED_ROLLOUTS = 25


def _missing_goal_atoms(task: Task, state: State) -> Set[GroundAtom]:
    """Goal atoms that do NOT hold in ``state`` by their own classifiers.

    Evaluated per atom with the goal predicates' own classifiers (the
    same ones ``goal_holds`` runs), never by abstracting the state with
    the agent's predicate set: the env's goal predicates are not in
    that set under predicate invention, so every goal atom then read
    as missing whenever the goal was not reached - including the ones
    that held - and one agent concluded the goal atoms could never be
    made True in the belief and abandoned a working route
    (2026-08-27 bridge policy seed 0, cycle 3).
    """
    return {a for a in task.goal if not a.holds(state)}


def _policy_source_path(ctx: ToolContext) -> Optional[str]:
    """Host path of the agent-editable ``policy.py`` (policy mode)."""
    base = _sandbox_base(ctx)
    if not base:
        return None
    return os.path.join(base, "policy.py")


def _parameter_margin_sweep(
        ctx: ToolContext, validation_cfg: ValidationConfig,
        fresh_scope: Callable[..., Any], rollout: Callable[[], Tuple[bool,
                                                                     str]],
        subject: str) -> Tuple[List[str], Optional[str], str]:
    """Margin sweep over BOTH parameter-uncertainty sources of one.

    capture-eligible submission - the single code path behind the
    physics-margin and rule-parameter gates of ``submit_plan``
    and ``submit_policy``.

    The execution repeats before this all run AT the fitted parameters,
    so they cannot see a submission whose success band excludes the
    fit's own error (run_20260723_091108: a capture validated 8/8 at
    fitted lateral_friction 0.5319 failed deterministically at true
    0.5). Two sources express that error:

    * identified PHYSICAL params: the fit posterior's sigma grid,
      applied as construction overrides on a fresh env (perturbing the
      shared env would leak into later tool calls), at the BASE planner
      seed so a failure is attributable to the perturbation alone;
    * learned RULE params: the calibrated posterior ensemble (the same
      members info-seeking exploration scores with), applied by
      swapping the live fitted-params view - entered BEFORE the fresh
      env so values bound at construction also see the member.

    ``rollout`` runs one validation rollout and returns ``(ok, why)``.
    Returns ``(outcome lines, param-sensitive detail or None, suffix
    for the validation note)``; any failing point sets the detail,
    which rejects the submission as PARAM-SENSITIVE.
    """
    outcomes: List[str] = []
    detail: Optional[str] = None
    note = ""
    if (validation_cfg.physics_margin
            and ctx.physics_margin_provider is not None):
        points = ctx.physics_margin_provider() or []

        def _physics_rollout(point: Dict[str, float]) -> Tuple[bool, str]:
            with fresh_scope(physical_overrides=point):
                return rollout()

        prefetched = _prefetch_parallel(
            [functools.partial(_physics_rollout, point) for point in points],
            f"{subject} physics margin")
        for point_idx, point in enumerate(points):
            ctx.attempt_rollout_count += 1
            pre = prefetched[point_idx]
            ok, why = pre if pre is not None else _physics_rollout(point)
            desc = ", ".join(f"{k}={v:.4g}" for k, v in sorted(point.items()))
            if ok:
                outcomes.append(f"physics point ({desc}): goal reached")
            else:
                outcomes.append(f"physics point ({desc}): FAILED - {why}")
                if detail is None:
                    detail = f"at {desc}: {why}"
        if points and detail is None:
            note += (
                f" Physics-margin check passed: the {subject} also reached "
                f"the goal at all {len(points)} grid points spanning +-1 "
                "sigma of the identified physical parameters.")
    if (validation_cfg.rule_param_margin and detail is None
            and ctx.rule_param_margin_provider is not None
            and ctx.rule_param_override_scope is not None):
        rule_points = ctx.rule_param_margin_provider() or []
        override_scope = ctx.rule_param_override_scope

        def _member_rollout(point: Dict[str, float]) -> Tuple[bool, str]:
            assert override_scope is not None
            with override_scope(point), fresh_scope():
                return rollout()

        # Prefetching runs every member even though the sequential loop
        # below still breaks at the first failure - the extra results
        # are discarded, keeping the report identical with the flag on
        # or off (failures are rare enough that the prepaid tail is
        # cheaper than serializing the common all-pass case).
        member_prefetched = _prefetch_parallel(
            [functools.partial(_member_rollout, pt) for pt in rule_points],
            f"{subject} rule-param margin")
        for member_idx, point in enumerate(rule_points):
            ctx.attempt_rollout_count += 1
            pre = member_prefetched[member_idx]
            ok, why = pre if pre is not None else _member_rollout(point)
            desc = (f"rule-param ensemble member "
                    f"{member_idx + 1}/{len(rule_points)}")
            if ok:
                outcomes.append(f"{desc}: goal reached")
            else:
                shown = ", ".join(f"{k}={v:.4g}"
                                  for k, v in sorted(point.items())[:8])
                if len(point) > 8:
                    shown += ", ..."
                outcomes.append(f"{desc}: FAILED - {why}")
                detail = f"under {desc} ({shown}): {why}"
                break
        if rule_points and detail is None:
            note += (
                f" Rule-parameter margin check passed: the {subject} also "
                f"reached the goal under all {len(rule_points)} calibrated "
                "posterior members of the learned rule parameters.")
    return outcomes, detail, note


def _build_testing_tools(ctx: ToolContext, _text_result: Callable,
                         tool: Callable) -> Dict[str, Any]:
    """Evaluation tools (option plans / policies against tasks)."""

    # Tool descriptions bake config values at BUILD time (session open);
    # the handlers below re-read config at CALL time.
    _gs_eval_doc = (
        "Runs your exact params with NO sampling (a `~` ground-sampler "
        "annotation - `~ [w1, w2]` region or `~ my_sampler` - is accepted "
        "but IGNORED here; only `sim.refine` uses it). "
        if RefinementConfig.from_cfg().ground_samplers else
        "Runs your exact params with NO sampling. ")

    @tool(
        "submit_plan",
        "SUBMIT a fully-specified plan as your answer for the CURRENT task. "
        "`plan` is text - one option per line, same grammar as `sim.run` / "
        "`sim.refine`: `Option(obj1:type1, obj2:type2)[param1, param2] -> "
        "{Atom(obj:type), ...}` (typed object refs; EXACT continuous params "
        "in `[]`, `[]` for none; optional `-> {atoms}` subgoals, prefix NOT "
        "to require false). " + _gs_eval_doc +
        "The plan is rolled out from the task's TRUE initial state through "
        "the belief model and reported step by step (include_states/"
        "include_atoms control the report). If it reaches the goal it is "
        "captured as your answer, and the per-step subgoals make it execute "
        "closed-loop (monitored, with replan-on-divergence). Capture is "
        "gated: a goal-reaching plan is re-run several times (simulation "
        "varies across runs; each rollout reports the motion-planner seed "
        "it ran at) and a FLAKY plan is reported instead of captured. The "
        "gate's rollout set is exactly what `sim.run(plan, trials=N)` runs "
        "(fresh env per rollout, same planner seeds), so measure "
        "reliability there BEFORE submitting, and reproduce one failed "
        "rollout with `sim.run(plan, seed=S, fresh=True)`; then add "
        "margin and resubmit. `validation_rollouts` "
        "requests a STRICTER gate for this submission (more rollouts; never "
        "fewer than configured). This is the ONLY path that captures an "
        "answer: explore (other tasks, modified states, partial plans, "
        "parameter sweeps, seeded reproductions) with `sim` in run_python, "
        "then submit the final plan here. "
        "When identified physical parameters are active, it is also re-run "
        "at a grid of perturbations spanning +-1 sigma of those parameters "
        "(the physics fit's own uncertainty); a PARAM-SENSITIVE plan is "
        "reported instead of captured - add design margin so it succeeds "
        "across the whole range. "
        "When the task has an evaluator, a goal-reaching plan the evaluator "
        "still scores as a non-solve (no success credit in its reward) is "
        "NOT captured (the real env applies the same scoring, so it could "
        "never count as a solve).",
        {
            "type": "object",
            "properties": {
                "plan": {
                    "type":
                    "string",
                    "description":
                    "Plan text, one option per line: "
                    "`Option(obj1:type1, obj2:type2)[p1, p2] -> "
                    "{Atom(obj:type), ...}` (exact params in `[]`; `[]` for "
                    "none; optional `-> {atoms}` subgoals, NOT-prefix to "
                    "require false).",
                },
                "include_states": {
                    "type":
                    "boolean",
                    "description":
                    "Include the full low-level state feature dict after each "
                    "step",
                    "default":
                    True
                },
                "include_atoms": {
                    "type": "boolean",
                    "description":
                    "Include atoms added/deleted after each step",
                    "default": True
                },
                "validation_rollouts": {
                    "type":
                    "integer",
                    "description":
                    "Request a stricter capture gate: total validation "
                    "rollouts a goal-reaching submission must pass. The "
                    "effective count is max(configured gate, this) - it can "
                    "raise the gate but never lower it. Use before "
                    "committing a plan you suspect is marginal.",
                },
            },
            "required": ["plan"],
        },
    )
    async def submit_plan(args: Dict[str, Any]) -> Dict[str, Any]:
        refine_cfg = RefinementConfig.from_cfg()
        validation_cfg = ValidationConfig.from_cfg()
        ctx.test_call_id += 1
        # Snapshot for the [budget] footer's per-call delta; this handler
        # increments the counter itself (initial rollout + validation
        # repeats), and without the snapshot the footer reports the
        # attempt's cumulative total as "+N this call".
        rollouts_before = ctx.attempt_rollout_count

        if ctx.option_model is None:
            return _error_result("No option model available in ToolContext.")

        all_options = ctx.options
        opt_map = {o.name: o for o in all_options}
        model = ctx.option_model
        model._name_to_parameterized_option = (  # type: ignore[attr-defined]  # pylint: disable=protected-access
            opt_map)

        plan_text = (args.get("plan") or "").strip()
        include_states = args.get("include_states", False)
        include_atoms = args.get("include_atoms", True)
        requested_rollouts = args.get("validation_rollouts")
        if requested_rollouts is not None and (not isinstance(
                requested_rollouts, int) or requested_rollouts < 1):
            return _error_result(
                "validation_rollouts must be a positive integer.")

        # Always the CURRENT task from its true initial state: this is
        # the submission path, and exploration on other tasks or from
        # modified states lives on the probe (sim.run).
        resolved, task_err = _resolve_task(ctx, None)
        if task_err is not None:
            return task_err
        assert resolved is not None
        task = resolved.task
        task_label = resolved.label

        lines = [f"Testing option plan on task {task_label}:"]
        saved_image_paths: List[str] = []

        all_predicates = ctx.predicates

        if not plan_text:
            return _error_result("`plan` is required (option plan text).")
        # Parse the text plan into a sketch (options + objects + exact params +
        # subgoals) using the SAME grammar/parser as sim.refine.
        types = set(ctx.types)
        for opt in all_options:
            types.update(opt.types)
        for pred in all_predicates:
            types.update(pred.types)
        types.update(o.type for o in task.init)
        try:
            # strict: the `plan` argument is pure plan text, so a line that
            # fails to parse is an error the agent must see - silently
            # dropping it (the freeform default) executes a different plan
            # than the agent asked for.
            gs_fns, gs_err = load_ground_sampler_fns(ctx)
            if gs_err is not None:
                return _error_result(gs_err)
            parse_notices: List[str] = []
            sketch_steps = bilevel_sketch.parse_sketch_from_text(
                plan_text,
                task,
                predicates=all_predicates,
                options=all_options,
                types=types,
                parse_continuous_params=True,
                strict=True,
                parse_ground_samplers=refine_cfg.ground_samplers,
                ground_sampler_fns=gs_fns or None,
                notices=parse_notices)
        except Exception as e:  # pylint: disable=broad-except
            return _error_result(f"Could not parse plan: {e}")
        lines.extend(f"NOTE: {n}" for n in parse_notices)
        if not sketch_steps:
            return _error_result(
                "Parsed empty plan. Each line must be "
                "`Option(obj:type, ...)[params] -> {subgoals}` with a known "
                "option, typed object refs, and exact params in `[]`.")
        # Ground each step with its parsed exact params, via the same
        # helper the refine path uses: an annotated Wait gets its
        # wait_target_atoms installed, so it waits for the annotated
        # atoms here exactly as in refine and in real execution -
        # grounding directly made the same Wait terminate on the first
        # incidental atom change in this rollout but wait for its
        # targets in refine, two different durations for one plan.
        grounded_plan: List[Any] = []
        for step_idx, st in enumerate(sketch_steps):
            params = (st.initial_params if st.initial_params is not None else
                      np.array([], dtype=np.float32))
            try:
                grounded_plan.append(
                    bilevel_sketch.ground_step(
                        st, np.asarray(params, dtype=np.float32)))
            except Exception as e:  # pylint: disable=broad-except
                return _error_result(f"Failed to ground step {step_idx} "
                                     f"({st.option.name}): {e}")

        # Per-low-level-step states + option labels for the task-evaluator
        # verdict below (see _EvalStateCollector for why per-step states).
        eval_collector = _EvalStateCollector(model, task.init)

        # Per-step report callback, driven by the shared forward executor.
        def _report_step(i: int, outcome: Any) -> None:
            eval_collector.collect(outcome)
            opt = outcome.option
            sig = f"{opt.name}({[o.name for o in opt.objects]})"
            if not outcome.initiable:
                atoms = utils.abstract(outcome.pre_state, ctx.predicates)
                atoms_str = ", ".join(str(a) for a in sorted(atoms))
                lines.append(f"Step {i}: {sig} - NOT INITIABLE\n"
                             f"  Current atoms: {{{atoms_str}}}\n"
                             f"  Object poses at failure:\n"
                             f"{format_object_poses(outcome.pre_state)}")
                return
            step_line = f"Step {i}: {sig} ({outcome.num_actions} actions)"
            if (opt.name == "Wait" and outcome.failure_reason is None
                    and outcome.num_actions >= utils.wait_rollout_step_cap()):
                step_line += (
                    "\n  NOTE: this Wait ran to its step cap - "
                    "its wait-target atoms never became true in the "
                    "belief (and no other atom changed). Check whether "
                    "the awaited change is modeled, or drop the Wait.")
            if outcome.failure_reason is not None:
                step_line += (f"\n  FAILURE REASON: {outcome.failure_reason}"
                              "\n  Object poses at failure:\n"
                              f"{format_object_poses(outcome.pre_state)}")
            post = outcome.post_state
            if post is not None and include_atoms:
                before = utils.abstract(outcome.pre_state, ctx.predicates)
                after = utils.abstract(post, ctx.predicates)
                added_s = ", ".join(str(a) for a in sorted(after - before))
                del_s = ", ".join(str(a) for a in sorted(before - after))
                step_line += (f"\n  Added:   {{{added_s}}}"
                              f"\n  Deleted: {{{del_s}}}")
            if post is not None and include_states:
                step_line += ("\n  State:\n" +
                              post.dict_str(indent=4, num_decimal_points=4))
            lines.append(step_line)
            # Render from the outcome's state explicitly: the rollout
            # runs on the gate's fresh env (see fresh_scope below) while
            # the renderer draws the shared session env, so without the
            # state the images would show a stale scene.
            img_block = render_pybullet_image(
                ctx,
                f"step_{i}_{opt.name}",
                state=post if post is not None else outcome.pre_state)
            if img_block and img_block.get("saved_path"):
                saved_image_paths.append(img_block["saved_path"])

        # One substrate for the WHOLE gate: rollout 1 (the capture
        # rollout) runs on the same freshly constructed env as the
        # validation repeats, at the base planner seed. This makes the
        # gate reproducible from inside the session - sim.run(plan,
        # trials=N) runs the identical rollout set (fresh env per trial,
        # planner seeds base..base+N-1) - and stops a submission from
        # advancing the shared session env. Rollout 1 on the warm shared
        # env was a different physics substrate from every repeat: the
        # 2026-08-30 bridge runs tuned plans to 27/27 on one substrate
        # that then scored 1/10 on the other, with no way to reproduce
        # the gate's rollouts.
        fresh_scope = (ctx.validation_env_scope
                       if validation_cfg.fresh_env else None)
        # Execute exactly like the real closed-loop executor: abort at the
        # first failing option (0-action collision / not-initiable / env
        # failure) instead of pressing on. Otherwise forward simulation can
        # continue past a collision and report a goal that the real rollout —
        # which ends the episode at that failed option — never reaches.
        ctx.attempt_rollout_count += 1
        with (fresh_scope()
              if fresh_scope is not None else contextlib.nullcontext()):
            result = bilevel_sketch.execute_plan_forward(
                task,
                grounded_plan,
                ctx.option_model,
                predicates=all_predicates,
                sketch=sketch_steps,
                on_step=_report_step,
                stop_on_failure=True)
            final_atoms = utils.abstract(result.final_state, ctx.predicates)
            # Task-evaluator verdict on this belief-sim rollout, computed
            # BEFORE capture and INSIDE the scope (certificate probes must
            # judge on the env the rollout ran on): the real evaluator
            # applies the same certificate, so a goal-reaching but
            # illegitimate plan can never count as a solve and must not be
            # captured as the answer (run_20260712_173955 tasks 1-2:
            # flagged-illegitimate captures stood all session and were
            # executed only to be rejected). Failure-tolerant: verdict
            # stays None when the task has no evaluator or nothing
            # executed. A coarse verdict (option-boundary states only) can
            # falsely reject a legitimate cascade, so it never blocks
            # capture.
            evaluator = _resolve_task_evaluator(ctx, task_label)
            verdict: Optional[Dict[str, Any]] = None
            if evaluator is not None and len(eval_collector.states) > 1:
                try:
                    verdict = evaluate_states_with(evaluator,
                                                   eval_collector.states,
                                                   eval_collector.labels,
                                                   sim_env=getattr(
                                                       ctx.option_model,
                                                       "sim_env", None))
                except Exception as e:  # pylint: disable=broad-except
                    logging.debug("Task-evaluator verdict failed: %s", e)
        # Use the env's goal-check (its own classifiers); robust to invented
        # predicates that don't reuse env names.
        goal_reached = result.goal_reached
        # One more real-executor constraint the option model doesn't enforce:
        # the episode is capped at the phase's step budget (the horizon, or
        # the interaction-request cap for explore episodes). A plan whose
        # goal is reached only after more steps than that will time out in
        # real rollout, so don't count it as achieved/captured.
        horizon = utils.real_episode_step_budget(ctx.phase)
        within_horizon = (result.actions_to_goal is not None
                          and result.actions_to_goal <= horizon)
        goal_achieved = (goal_reached and result.clean_to_goal
                         and within_horizon)
        evaluator_rejected = (verdict is not None and not verdict["legitimate"]
                              and not eval_collector.coarse)
        # An evaluator rejection only disqualifies a capture when the goal
        # atoms actually hold via an illegitimate route - a reward hack (e.g.
        # the agent knocked the target over directly). An honest shortfall,
        # where the rollout simply fails to reach the goal, is ALSO
        # legitimate=False (there is no genuine cascade to certify), but that
        # is exactly what a best-effort submission is meant to capture, so it
        # must not be conflated with a reward hack.
        reward_hack = (evaluator_rejected and verdict is not None
                       and verdict["terminated"])

        # Multi-rollout validation of a capture candidate. The shared sim
        # env is nondeterministic across repeats (motion-planner sampling,
        # physics-solver state), which is the same variability the real
        # rollout will sample - a plan that only sometimes succeeds here is
        # a margin-free plan that will likely fail on the real env
        # (run_20260712_192457 task 1: a sim-validated 2-hop relay died on a
        # ~9mm placement drift). So a goal-reaching plan is captured only
        # after every one of validation_cfg.rollouts total
        # rollouts succeeds; a flaky repeat is reported to the agent, who
        # still has the session to add margin and resubmit.
        def _validation_rollout() -> Tuple[bool, str, List[Optional[State]]]:
            """One extra rollout of the exact plan.

            Returns ``(ok, failure detail, per-step post-states)``; the
            post-state list is padded with ``None`` to the plan length
            so a truncated (failed) rollout still indexes safely.
            Passing rollouts' post-states feed the captured-annotation
            intersection filter.
            """
            v_collector = _EvalStateCollector(model, task.init)
            r = bilevel_sketch.execute_plan_forward(
                task,
                grounded_plan,
                model,
                predicates=all_predicates,
                sketch=sketch_steps,
                on_step=v_collector.on_step,
                stop_on_failure=True)
            posts: List[Optional[State]] = [s.post_state for s in r.steps]
            posts += [None] * (len(grounded_plan) - len(posts))
            if r.first_failure_idx is not None:
                fr = r.steps[r.first_failure_idx].failure_reason
                opt = r.steps[r.first_failure_idx].option
                return False, (f"step {r.first_failure_idx} "
                               f"({opt.name}) failed: {fr}"), posts
            if not r.goal_reached:
                missing = _missing_goal_atoms(task, r.final_state)
                missing_str = ", ".join(str(a) for a in sorted(missing))
                detail = f" (missing: {{{missing_str}}})" if missing else ""
                return False, f"goal not reached{detail}", posts
            if not (r.actions_to_goal is not None
                    and r.actions_to_goal <= horizon):
                return False, (f"goal reached only after "
                               f"{r.actions_to_goal} low-level steps, past "
                               f"the episode horizon ({horizon})"), posts
            # Same legitimacy rule as the first rollout: a non-coarse
            # illegitimate verdict fails the validation.
            if (evaluator is not None and len(v_collector.states) > 1
                    and not v_collector.coarse):
                try:
                    v = evaluate_states_with(evaluator,
                                             v_collector.states,
                                             v_collector.labels,
                                             sim_env=getattr(
                                                 ctx.option_model, "sim_env",
                                                 None))
                    if not v["legitimate"]:
                        return False, (
                            "this rollout reached the goal atoms but the "
                            "task evaluator scored it as a non-solve "
                            f"(solved=False, reward={v['reward']:.2f})"), posts
                except Exception as e:  # pylint: disable=broad-except
                    logging.debug("Validation-rollout verdict failed: %s", e)
            return True, "", posts

        flaky_detail: Optional[str] = None
        validation_note = ""
        n_rollouts = max(1, validation_cfg.rollouts)
        # Escalated gate once this task has produced a FLAKY rejection: the
        # agent is provably tuning in a marginal region, where a lucky
        # streak passes the base gate and dies on the single real episode
        # (run_20260717_182321: a 20/20-swept relay placement validated 3/3,
        # then missed the target for real).
        capture_task_key = _capture_task_key(ctx)
        if capture_task_key in ctx.flaky_capture_task_keys:
            n_rollouts = max(n_rollouts, validation_cfg.rollouts_after_flaky)
        # The agent may request a STRICTER gate for this submission (a
        # plan it suspects is marginal); it can never lower the
        # configured gate - that would let a lucky draw bypass it.
        capped_request: Optional[int] = None
        if requested_rollouts is not None:
            capped_request = min(requested_rollouts, _MAX_REQUESTED_ROLLOUTS)
            if capped_request < requested_rollouts:
                lines.append(
                    f"NOTE: validation_rollouts={requested_rollouts} capped "
                    f"at {_MAX_REQUESTED_ROLLOUTS}.")
            n_rollouts = max(n_rollouts, capped_request)
        # fresh_scope (computed above, shared with rollout 1): repeats on
        # the shared env are correlated (its reset cannot reconstruct
        # state exactly), so only fresh envs sample the same distribution
        # the real episode will.
        rollout_outcomes: List[str] = []
        # Per-step post-states of PASSING validation rollouts, for the
        # captured-annotation intersection filter. Failing rollouts are
        # excluded on purpose: they are off-track by definition, so their
        # post-states are not evidence about what holds on a successful
        # execution (using them would prune annotations that hold in
        # every on-track run). Physics-margin rollouts are likewise
        # excluded: they run under deliberately perturbed physics.
        passing_validation_posts: List[List[Optional[State]]] = []
        base_planner_seed = CFG.seed
        if (ctx.capture_goal_reaching_plans and goal_achieved
                and not evaluator_rejected and grounded_plan
                and n_rollouts > 1):
            # Run ALL validation rollouts even after a failure: the
            # per-rollout outcome list distinguishes failure modes (a
            # physics-tail fizzle vs. an IK stall vs. a certificate
            # rejection) and yields a reliability estimate - a bare
            # "rollout k FAILED" left agents guessing which
            # (run_20260717_182040 seed0 turn 214).
            # decorrelated_rollout_seed: a fresh env alone gives
            # bit-identical repeats (motion planning reads the
            # constant CFG.seed at call time), so without it the
            # validation repeats re-run the capture rollout verbatim
            # and detect nothing. The capture rollout itself keeps
            # the base seed; repeats sample execution variability.
            def _repeat_rollout(
                    repeat_idx: int
            ) -> Tuple[bool, str, List[Optional[State]]]:
                with (fresh_scope() if fresh_scope is not None else
                      contextlib.nullcontext()), \
                        decorrelated_rollout_seed(repeat_idx - 1):
                    return _validation_rollout()

            repeat_indices = list(range(2, n_rollouts + 1))
            repeat_prefetched = _prefetch_parallel([
                functools.partial(_repeat_rollout, k) for k in repeat_indices
            ], "capture repeat rollouts")
            for pos, repeat_idx in enumerate(repeat_indices):
                ctx.attempt_rollout_count += 1
                pre = repeat_prefetched[pos]
                ok, why, repeat_posts = (pre if pre is not None else
                                         _repeat_rollout(repeat_idx))
                repeat_seed = base_planner_seed + repeat_idx - 1
                if ok:
                    passing_validation_posts.append(repeat_posts)
                    rollout_outcomes.append(
                        f"rollout {repeat_idx} (planner seed "
                        f"{repeat_seed}): goal reached")
                else:
                    rollout_outcomes.append(
                        f"rollout {repeat_idx} (planner seed "
                        f"{repeat_seed}): FAILED - {why}")
                    if flaky_detail is None:
                        flaky_detail = (f"rollout {repeat_idx}/{n_rollouts} "
                                        f"(planner seed {repeat_seed}) "
                                        f"FAILED: {why}")
            if flaky_detail is None:
                fresh_note = (", each on a freshly constructed simulator "
                              "instance" if fresh_scope is not None else "")
                validation_note = (
                    f" Validated {n_rollouts}/{n_rollouts} rollouts "
                    f"(planner seeds {base_planner_seed}-"
                    f"{base_planner_seed + n_rollouts - 1}; the "
                    "simulator's motion planning and physics stepping vary "
                    "across runs; repeats sample that execution "
                    f"variability{fresh_note}; sim.run(plan, "
                    f"trials={n_rollouts}) reruns this exact rollout set).")

        # Parameter-margin gates (see _parameter_margin_sweep): the
        # execution repeats above all run AT the fitted parameters, so
        # they cannot see a plan whose success band excludes the fit's
        # own error - in the identified physical params or the learned
        # rule constants.
        param_sensitive_detail: Optional[str] = None
        margin_outcomes: List[str] = []
        if (fresh_scope is not None and ctx.capture_goal_reaching_plans
                and goal_achieved and not evaluator_rejected and grounded_plan
                and flaky_detail is None):
            margin_outcomes, param_sensitive_detail, margin_note = \
                _parameter_margin_sweep(
                    ctx, validation_cfg, fresh_scope,
                    lambda: _validation_rollout()[:2], "plan")
            validation_note += margin_note

        def _stash_uncaptured_submission() -> None:
            """Remember the best refused submission of this attempt.

            The journal auto-entry records it at attempt end, so the
            plan (and its honest evaluator reward) survives the fresh-
            context restart and the final best-effort nudge can resubmit
            it instead of the attempt's work vanishing with its context.
            """
            reward = float(verdict["reward"]) if verdict is not None else None
            prev = ctx.best_uncaptured_reward
            if ctx.best_uncaptured_plan_lines is not None and (
                    reward is None or (prev is not None and reward <= prev)):
                return
            ctx.best_uncaptured_reward = reward
            ctx.best_uncaptured_plan_lines = list(
                bilevel_sketch.format_plan_lines(grounded_plan))

        # The capture decision itself is pure (see _decide_capture, which
        # also documents the best-effort-mode semantics); the branches
        # below apply its ctx mutations and format its messages.
        capture_outcome = _decide_capture(
            # In policy mode the deliverable is policy.py (via
            # submit_policy); this tool remains a probe but can no
            # longer capture the answer.
            capture_enabled=(ctx.capture_goal_reaching_plans
                             and not ctx.policy_capture_mode),
            is_current_task=True,
            have_plan=bool(grounded_plan),
            goal_achieved=goal_achieved,
            evaluator_rejected=evaluator_rejected,
            reward_hack=reward_hack,
            flaky=flaky_detail is not None,
            best_effort_mode=ctx.capture_best_effort_plan,
            have_validated_capture=bool(ctx.solved_plan_reached_goal),
            param_sensitive=param_sensitive_detail is not None)
        decision = capture_outcome.decision
        captured = capture_outcome.captured
        if captured:
            # Capture the plan with a sketch that keeps only the subgoals
            # that actually held (so the closed-loop monitor won't flag a
            # spurious divergence on a wrong annotation). An annotation
            # must hold in rollout 1 AND in every PASSING validation
            # rollout: an atom that held once by luck under the sim's own
            # nondeterminism would otherwise survive into the executed
            # sketch and kill the real episode on a spurious divergence.
            # (With zero passing repeats this reduces to the rollout-1
            # filter.)
            validated_solve = decision is CaptureDecision.VALIDATED_CAPTURE
            captured_sketch = []

            def _held_in_passing_repeats(atom: Any, i: int,
                                         want_held: bool) -> bool:
                for posts in passing_validation_posts:
                    post_i = posts[i] if i < len(posts) else None
                    # bool(): classifiers may return numpy bools, which
                    # fail identity checks against Python bools.
                    if post_i is None or bool(atom.holds(post_i)) != want_held:
                        return False
                return True

            # Execution-verifiability probe. The closed-loop monitor
            # evaluates the captured annotations on REAL observations,
            # which carry no latent (``State.latent`` is None outside
            # belief rollouts). An atom whose truth in the certifying
            # post-state depends on the belief latent therefore reads
            # false at execution no matter what physically happens, and
            # a single such annotation aborts a healthy episode (a
            # latent-only SeamBonded killed two runs whose bonds had in
            # fact formed). Certify each surviving positive atom on the
            # same post-state with the latent stripped and drop the
            # ones that fail; a classifier that RAISES without a latent
            # would crash the monitor, so it is dropped from either
            # polarity the same way (a stripped negative atom that
            # merely evaluates is kept - it cannot fire spuriously).
            unverifiable_dropped: List[str] = []

            def _probe_without_latent(atom: Any,
                                      post: State) -> Optional[bool]:
                stripped = State(post.data)
                try:
                    return bool(atom.holds(stripped))
                except Exception:  # pylint: disable=broad-except
                    return None

            for i, st in enumerate(sketch_steps):
                post = (result.steps[i].post_state
                        if i < len(result.steps) else None)
                if post is not None:
                    after = utils.abstract(post, all_predicates)
                    pos_held = {
                        a
                        for a in (st.subgoal_atoms or set())
                        if a in after and _held_in_passing_repeats(a, i, True)
                    }
                    neg_held = {
                        a
                        for a in (st.subgoal_neg_atoms or set())
                        if a not in after
                        and _held_in_passing_repeats(a, i, False)
                    }
                    pos_drop = {
                        a
                        for a in pos_held
                        if _probe_without_latent(a, post) is not True
                    }
                    neg_drop = {
                        a
                        for a in neg_held
                        if _probe_without_latent(a, post) is None
                    }
                    unverifiable_dropped.extend(
                        f"step {i} ({st.option.name}): {a}"
                        for a in sorted(pos_drop | neg_drop, key=str))
                    pos_held -= pos_drop
                    neg_held -= neg_drop
                else:
                    pos_held, neg_held = set(), set()
                captured_sketch.append(
                    bilevel_sketch.SketchStep(option=st.option,
                                              objects=st.objects,
                                              subgoal_atoms=pos_held or None,
                                              subgoal_neg_atoms=neg_held
                                              or None))
            # Re-align each Wait's target atoms with the FILTERED
            # sketch: the real executor waits on exactly the monitored
            # (execution-verifiable) atoms. A latent-only target (e.g. a
            # belief Bonded) reads false on every real observation, so
            # leaving it in the grounded option's memory would stall the
            # real Wait to its step-cap backstop no matter what happens.
            # With every target filtered away the Wait falls back to
            # any-atom-change, the same rule the belief rollout then
            # shares.
            for g_opt, cap_step in zip(grounded_plan, captured_sketch):
                if g_opt.name != "Wait":
                    continue
                g_opt.memory.pop("wait_target_atoms", None)
                g_opt.memory.pop("wait_target_neg_atoms", None)
                if cap_step.subgoal_atoms:
                    g_opt.memory["wait_target_atoms"] = cap_step.subgoal_atoms
                if cap_step.subgoal_neg_atoms:
                    g_opt.memory["wait_target_neg_atoms"] = \
                        cap_step.subgoal_neg_atoms
            ctx.solved_plan = grounded_plan
            ctx.solved_sketch = captured_sketch
            ctx.solved_plan_reached_goal = validated_solve
            ctx.solved_plan_eval_reward = (float(verdict["reward"])
                                           if verdict is not None else None)
            summary_bits = [
                f"validation: "
                f"{1 + sum(1 for o in rollout_outcomes if 'FAILED' not in o)}"
                f"/{1 + len(rollout_outcomes)} rollouts ok"
            ]
            if flaky_detail is not None:
                summary_bits.append(f"first failure: {flaky_detail}")
            if margin_outcomes:
                n_margin_ok = sum(1 for o in margin_outcomes
                                  if "FAILED" not in o)
                summary_bits.append(f"physics margin: {n_margin_ok}/"
                                    f"{len(margin_outcomes)} points ok")
            ctx.solved_plan_validation_summary = "; ".join(summary_bits)
            n_annot = sum(1 for s in captured_sketch
                          if s.subgoal_atoms or s.subgoal_neg_atoms)
            reason = capture_outcome.best_effort_reason
            if reason is None:
                best_effort_note = ""
            elif reason is BestEffortReason.GOAL_NOT_REACHED:
                best_effort_note = (" (best-effort: goal NOT reached, "
                                    "accepted because the attempt budget is "
                                    "exhausted; it executes for its honest "
                                    "reward but will not count as a solve)")
            elif reason is BestEffortReason.REWARD_HACK:
                best_effort_note = (" (best-effort: the rollout reaches the "
                                    "goal atoms but the task evaluator "
                                    "scores it as a non-solve, and the real "
                                    "env applies the same scoring; accepted "
                                    "because the attempt budget is exhausted "
                                    "- it executes for its honest reward but "
                                    "will not count as a solve)")
            elif reason is BestEffortReason.FLAKY:
                best_effort_note = (f" (best-effort: {flaky_detail}; "
                                    "accepted because the attempt budget is "
                                    "exhausted - it executes for its honest "
                                    "reward but may not reproduce its "
                                    "solve)")
            else:
                assert reason is BestEffortReason.PARAM_SENSITIVE
                best_effort_note = (" (best-effort: failed "
                                    f"{param_sensitive_detail}; accepted "
                                    "because the attempt budget is exhausted "
                                    "- it executes for its honest reward but "
                                    "may fail under the true physics)")
            unverifiable_note = ""
            if unverifiable_dropped:
                dropped_lines = "\n".join(f"  {d}"
                                          for d in unverifiable_dropped)
                unverifiable_note = (
                    f"\nNOTE: {len(unverifiable_dropped)} annotation(s) "
                    "cannot be verified from a real observation (their "
                    "truth here depends on the belief latent, which real "
                    "env states do not carry) and were excluded from "
                    "closed-loop monitoring - the plan still executes "
                    "them, they just cannot trigger a replan:\n"
                    f"{dropped_lines}\n"
                    "Prefer annotating with predicates whose classifiers "
                    "read observable features.")
                logging.info(
                    "Capture: excluded %d execution-unverifiable "
                    "annotation(s) from the monitored sketch:\n%s",
                    len(unverifiable_dropped), dropped_lines)
            lines.append(f"Captured as the current answer{best_effort_note}: "
                         f"{len(grounded_plan)} steps, "
                         f"{n_annot} with subgoal annotations for closed-loop "
                         f"monitoring.{validation_note}{unverifiable_note}")
        elif decision is CaptureDecision.FLAKY_NO_CAPTURE:
            # Record the task so later submissions face the escalated
            # gate - flakiness here is evidence the whole parameter
            # region is marginal, not just this point.
            ctx.flaky_capture_task_keys.add(capture_task_key)
            _stash_uncaptured_submission()
            escalated_n = max(max(1, validation_cfg.rollouts),
                              validation_cfg.rollouts_after_flaky)
            n_ok = 1 + sum(1 for o in rollout_outcomes if "FAILED" not in o)
            per_rollout = "\n".join(f"  {o}" for o in rollout_outcomes)
            lines.append(
                f"FLAKY (plan NOT captured): the plan reached the goal on "
                f"rollout 1 but {flaky_detail}. Per-rollout outcomes "
                f"(estimated reliability {n_ok}/{n_rollouts}):\n"
                f"  rollout 1 (planner seed {base_planner_seed}): "
                f"goal reached\n{per_rollout}\n"
                "The simulator's motion "
                "planning and physics stepping vary across runs, and the "
                "real environment samples the same variability - a plan "
                "that only sometimes succeeds in simulation will likely "
                "fail for real. This gate is reproducible in run_python: "
                f"sim.run(plan, trials={n_rollouts}) runs the identical "
                "rollout set (fresh env per trial, same planner seeds), "
                "and sim.run(plan, seed=<the failed rollout's planner "
                "seed>, fresh=True) re-runs one failed rollout exactly "
                "with full per-step reporting (without fresh=True the "
                "warm session env is a different, optimistic substrate). "
                "Then add margin (e.g. tighter spacing, aim "
                "impacts closer to the middle of the fall path) and "
                "resubmit. Because this task has now produced a flaky "
                f"submission, captures require {escalated_n}/{escalated_n} "
                "successful rollouts: fix the margin rather than "
                "resubmitting near-identical parameters.")
        elif decision is CaptureDecision.PARAM_SENSITIVE_NO_CAPTURE:
            _stash_uncaptured_submission()
            per_point = "\n".join(f"  {o}" for o in margin_outcomes)
            lines.append(
                "PARAM-SENSITIVE (plan NOT captured): the plan passed "
                "execution validation at the fitted physical parameters "
                f"but FAILED {param_sensitive_detail}.\n"
                "Physics-margin rollouts (a grid spanning +-1 sigma of "
                "the identified physical parameters, the sysID fit's "
                f"own uncertainty):\n{per_point}\n"
                "The fitted values are uncertain at this scale and the "
                "real environment may sit anywhere in that range - "
                "including BETWEEN passing points: success can be "
                "non-monotonic in a physical parameter, so a design must "
                "hold across the whole range, not just at the values you "
                "tuned at. Add margin to the DESIGN (not the execution) - "
                "e.g. tighter spacing or impacts nearer the middle of the "
                "fall path - then resubmit.")
        elif decision is CaptureDecision.REWARD_HACK_NO_CAPTURE:
            assert verdict is not None
            _stash_uncaptured_submission()
            lines.append(
                "NOT CAPTURED: the rollout reaches the goal atoms but the "
                "task evaluator scores it as a non-solve (solved=False, "
                f"reward={verdict['reward']:.2f}). The real env applies the "
                "same scoring, so executing this plan cannot count as a "
                "solve. Find a plan whose rollout the evaluator scores "
                "solved=True.")
        if result.first_failure_idx is not None:
            fr = result.steps[result.first_failure_idx].failure_reason
            lines.append(
                f"\nPlan FAILED at step {result.first_failure_idx}: {fr}")
        final_atoms_str = ", ".join(str(a) for a in sorted(final_atoms))
        lines.append(f"\nFinal atoms: {{{final_atoms_str}}}")
        if task.goal_nl:
            lines.append(f"Goal (natural language): {task.goal_nl}")
        else:
            goal_str = ", ".join(str(g) for g in sorted(task.goal))
            lines.append(f"Goal: {{{goal_str}}}")
        lines.append(f"Goal achieved: {goal_achieved}")
        # Task-evaluator verdict line (verdict computed above, before the
        # capture decision it gates). On a FLAKY rejection this verdict is
        # rollout 1's only - printing it unlabeled next to a failing
        # rollout's non-solve read as two contradictory verdicts in one
        # message (run_20260717_182040 seed1 turn 96).
        if verdict is not None:
            vline = _format_evaluator_verdict(verdict,
                                              coarse=eval_collector.coarse)
            if flaky_detail is not None and not captured:
                vline += (" [rollout 1 only - NOT the operative outcome; "
                          "this submission was rejected as FLAKY above]")
            lines.append(vline)
        # Goal atoms hold but the plan needs more low-level steps than the
        # episode horizon allows: say so and that it was NOT captured, so the
        # agent shortens the plan instead of stopping on a false positive.
        # (A best-effort capture still happens above; then only warn.)
        if goal_reached and not within_horizon and not captured:
            lines.append(
                f"NOT EXECUTABLE (plan was NOT captured): reaching the goal "
                f"takes {result.actions_to_goal} low-level steps but the "
                f"episode horizon is {horizon}. The real executor will run "
                f"out of steps — shorten the plan (fewer or quicker steps) "
                f"before resubmitting.")
        elif goal_reached and not within_horizon:
            lines.append(
                f"WARNING: reaching the goal takes {result.actions_to_goal} "
                f"low-level steps but the episode horizon is {horizon}, so "
                f"the real executor will run out of steps before the goal.")
        # Print the missing goal atoms even when the goal is stated in
        # natural language: "Goal achieved: False" with no per-atom
        # diagnosis left agents unable to tell a near-miss from a
        # non-starter, and validation-rollout failures already name the
        # missing atoms - this just makes rollout 1 report the same way.
        if not goal_reached:
            missing = _missing_goal_atoms(task, result.final_state)
            missing_str = ", ".join(str(a) for a in sorted(missing))
            lines.append(f"Missing goal atoms: {{{missing_str}}}")

        # Append image save paths to text output
        if saved_image_paths:
            lines.append("\nSaved images:")
            for p in saved_image_paths:
                lines.append(f"  {p}")

        # Build result with text only (images are saved to disk)
        return _text_result("\n".join(lines) +
                            _budget_footer(ctx, rollouts_before))

    @tool(
        "submit_policy",
        "Validate ./policy.py - your closed-loop `get_option(state, memory)` "
        "program - on the CURRENT task and capture it as your answer. The "
        "policy source is SNAPSHOTTED at call time (later edits need a new "
        "call). Each rollout runs the policy closed-loop through the belief "
        "model: get_option is called at every option boundary with the "
        "actual current state; option failures (not initiable, motion-"
        "planning refusal, 0 actions) do NOT end the episode - the failure "
        "text arrives in memory['last_failure'] and get_option is asked "
        "again, so RECOVERY is your policy's job; exceptions in get_option, "
        "unparsable/ungroundable lines, re-issuing one identical "
        "failing line repeatedly (the stuck-loop guard), and re-issuing "
        "one identical line that keeps completing with no state change "
        "(its no-op livelock twin) DO end it. Capture "
        "is gated like "
        "submit_plan: the goal-reaching rollout is repeated "
        "several times (fresh simulator env + varied planner seed per "
        "repeat, fresh memory per episode) and a FLAKY policy is reported "
        "instead of captured; physics-margin perturbations apply too. "
        "`validation_rollouts` requests a stricter gate. Test recovery "
        "behavior first with sim.run_policy() in "
        "run_python, which runs ./policy.py from the CURRENT probe "
        "state (including perturbed or mid-plan states).",
        {
            "type": "object",
            "properties": {
                "validation_rollouts": {
                    "type":
                    "integer",
                    "description":
                    "Request a stricter capture gate: total validation "
                    "rollouts a goal-reaching policy must pass (effective "
                    "count is max(configured, this); never fewer).",
                },
                "include_atoms": {
                    "type": "boolean",
                    "description":
                    "Include atoms added/deleted after each step",
                    "default": True
                },
            },
        },
    )
    async def submit_policy(args: Dict[str, Any]) -> Dict[str, Any]:
        # pylint: disable-next=import-outside-toplevel
        from predicators.agent_sdk.policy_execution import \
            build_policy_option_fn, execute_policy_forward
        validation_cfg = ValidationConfig.from_cfg()
        ctx.test_call_id += 1
        rollouts_before = ctx.attempt_rollout_count
        if not ctx.policy_capture_mode:
            return _error_result(
                "submit_policy is only available in policy mode "
                "(agent_solve_policy_mode); submit plans via "
                "submit_plan instead.")
        if ctx.option_model is None:
            return _error_result("No option model available in ToolContext.")
        all_options = ctx.options
        model = ctx.option_model
        model._name_to_parameterized_option = (  # type: ignore[attr-defined]  # pylint: disable=protected-access
            {o.name: o
             for o in all_options})
        requested_rollouts = args.get("validation_rollouts")
        include_atoms = args.get("include_atoms", True)
        if requested_rollouts is not None and (not isinstance(
                requested_rollouts, int) or requested_rollouts < 1):
            return _error_result(
                "validation_rollouts must be a positive integer.")

        resolved, task_err = _resolve_task(ctx, None)
        if task_err is not None:
            return task_err
        assert resolved is not None
        task = resolved.task
        task_label = resolved.label

        policy_path = _policy_source_path(ctx)
        if policy_path is None or not os.path.isfile(policy_path):
            return _error_result(
                "No ./policy.py found. Write your closed-loop policy there "
                "first: `def get_option(state, memory): ...` returning one "
                "plan line (sketch grammar) or None for DONE.")
        with open(policy_path, "r", encoding="utf-8") as f:
            policy_source = f.read()

        all_predicates = ctx.predicates
        types = set(ctx.types)
        for opt in all_options:
            types.update(opt.types)
        for pred in all_predicates:
            types.update(pred.types)
        types.update(o.type for o in task.init)

        def _fresh_option_fn() -> Tuple[Optional[Any], Optional[str]]:
            # Fresh instance per episode: memory must reset per rollout.
            return build_policy_option_fn(policy_source,
                                          task,
                                          predicates=all_predicates,
                                          options=all_options,
                                          types=types)

        option_fn, load_err = _fresh_option_fn()
        if load_err is not None or option_fn is None:
            return _error_result(load_err or "policy.py failed to load.")
        max_opts = CFG.agent_policy_max_options
        horizon = utils.real_episode_step_budget(ctx.phase)

        lines = [f"Testing policy.py on task {task_label}:"]
        saved_image_paths: List[str] = []
        eval_collector = _EvalStateCollector(model, task.init)

        def _report_step(i: int, outcome: Any) -> None:
            eval_collector.collect(outcome)
            opt = outcome.option
            sig = f"{opt.name}({[o.name for o in opt.objects]})"
            step_line = f"Step {i}: {sig} ({outcome.num_actions} actions)"
            if outcome.failure_reason is not None:
                step_line += (
                    f"\n  OPTION FAILURE (surfaced to the policy as "
                    f"memory['last_failure']): {outcome.failure_reason}")
            post = outcome.post_state
            if post is not None and include_atoms:
                before = utils.abstract(outcome.pre_state, ctx.predicates)
                after = utils.abstract(post, ctx.predicates)
                added_s = ", ".join(str(a) for a in sorted(after - before))
                del_s = ", ".join(str(a) for a in sorted(before - after))
                step_line += (f"\n  Added:   {{{added_s}}}"
                              f"\n  Deleted: {{{del_s}}}")
            lines.append(step_line)
            # Explicit state: the rollout runs on the gate's fresh env
            # while the renderer draws the shared session env (see
            # submit_plan's _report_step).
            img_block = render_pybullet_image(
                ctx,
                f"policy_step_{i}_{opt.name}",
                state=post if post is not None else outcome.pre_state)
            if img_block and img_block.get("saved_path"):
                saved_image_paths.append(img_block["saved_path"])

        # One substrate for the whole gate, as in submit_plan: rollout 1
        # runs on the same fresh env as the validation repeats, at the
        # base planner seed, so the gate is reproducible in-session.
        fresh_scope = (ctx.validation_env_scope
                       if validation_cfg.fresh_env else None)
        ctx.attempt_rollout_count += 1
        with (fresh_scope()
              if fresh_scope is not None else contextlib.nullcontext()):
            result = execute_policy_forward(task,
                                            option_fn,
                                            model,
                                            predicates=all_predicates,
                                            max_policy_options=max_opts,
                                            on_step=_report_step)
            # Verdict INSIDE the scope: certificate probes must judge on
            # the env the rollout ran on.
            evaluator = _resolve_task_evaluator(ctx, task_label)
            verdict: Optional[Dict[str, Any]] = None
            if evaluator is not None and len(eval_collector.states) > 1:
                try:
                    verdict = evaluate_states_with(evaluator,
                                                   eval_collector.states,
                                                   eval_collector.labels,
                                                   sim_env=getattr(
                                                       ctx.option_model,
                                                       "sim_env", None))
                except Exception as e:  # pylint: disable=broad-except
                    logging.debug("Task-evaluator verdict failed: %s", e)

        goal_reached = result.goal_reached
        within_horizon = (result.actions_to_goal is not None
                          and result.actions_to_goal <= horizon)
        if result.policy_error is not None:
            lines.append(f"POLICY ERROR (ended the episode): "
                         f"{result.policy_error}")
        n_surfaced = sum(1 for s in result.steps
                         if s.failure_reason is not None)
        if n_surfaced:
            lines.append(
                f"{n_surfaced} option failure(s) were surfaced to the "
                "policy during this rollout (recovery attempts included "
                "above).")
        # Closed-loop: recovered option failures do NOT disqualify - the
        # policy handling them is the point. Only the goal, the horizon,
        # and policy-code errors gate.
        goal_achieved = (goal_reached and within_horizon
                         and result.policy_error is None)
        evaluator_rejected = (verdict is not None and not verdict["legitimate"]
                              and not eval_collector.coarse)
        reward_hack = (evaluator_rejected and verdict is not None
                       and verdict["terminated"])

        def _policy_validation_rollout() -> Tuple[bool, str]:
            fn, err = _fresh_option_fn()
            if err is not None or fn is None:
                return False, f"policy failed to load: {err}"
            v_collector = _EvalStateCollector(model, task.init)
            r = execute_policy_forward(task,
                                       fn,
                                       model,
                                       predicates=all_predicates,
                                       max_policy_options=max_opts,
                                       on_step=v_collector.on_step)
            if r.policy_error is not None:
                return False, f"policy error: {r.policy_error}"
            if not r.goal_reached:
                missing = _missing_goal_atoms(task, r.final_state)
                missing_str = ", ".join(str(a) for a in sorted(missing))
                detail = f" (missing: {{{missing_str}}})" if missing else ""
                return False, f"goal not reached{detail}"
            if not (r.actions_to_goal is not None
                    and r.actions_to_goal <= horizon):
                return False, (f"goal reached only after {r.actions_to_goal} "
                               f"low-level steps, past the episode horizon "
                               f"({horizon})")
            if (evaluator is not None and len(v_collector.states) > 1
                    and not v_collector.coarse):
                try:
                    v = evaluate_states_with(evaluator,
                                             v_collector.states,
                                             v_collector.labels,
                                             sim_env=getattr(
                                                 ctx.option_model, "sim_env",
                                                 None))
                    if not v["legitimate"]:
                        return False, (
                            "this rollout reached the goal atoms but the "
                            "task evaluator scored it as a non-solve "
                            f"(solved=False, reward={v['reward']:.2f})")
                except Exception as e:  # pylint: disable=broad-except
                    logging.debug("Validation-rollout verdict failed: %s", e)
            return True, ""

        flaky_detail: Optional[str] = None
        validation_note = ""
        n_rollouts = max(1, validation_cfg.rollouts)
        capture_task_key = _capture_task_key(ctx)
        if capture_task_key in ctx.flaky_capture_task_keys:
            n_rollouts = max(n_rollouts, validation_cfg.rollouts_after_flaky)
        if requested_rollouts is not None:
            n_rollouts = max(n_rollouts,
                             min(requested_rollouts, _MAX_REQUESTED_ROLLOUTS))
        # fresh_scope computed above, shared with rollout 1.
        rollout_outcomes: List[str] = []
        base_planner_seed = CFG.seed
        if (ctx.capture_goal_reaching_plans and goal_achieved
                and not evaluator_rejected and n_rollouts > 1):

            def _policy_repeat_rollout(repeat_idx: int) -> Tuple[bool, str]:
                with (fresh_scope() if fresh_scope is not None else
                      contextlib.nullcontext()), \
                        decorrelated_rollout_seed(repeat_idx - 1):
                    return _policy_validation_rollout()

            repeat_indices = list(range(2, n_rollouts + 1))
            repeat_prefetched = _prefetch_parallel([
                functools.partial(_policy_repeat_rollout, k)
                for k in repeat_indices
            ], "policy repeat rollouts")
            for pos, repeat_idx in enumerate(repeat_indices):
                ctx.attempt_rollout_count += 1
                pre = repeat_prefetched[pos]
                ok, why = (pre if pre is not None else
                           _policy_repeat_rollout(repeat_idx))
                repeat_seed = base_planner_seed + repeat_idx - 1
                if ok:
                    rollout_outcomes.append(
                        f"rollout {repeat_idx} (planner seed "
                        f"{repeat_seed}): goal reached")
                else:
                    rollout_outcomes.append(
                        f"rollout {repeat_idx} (planner seed "
                        f"{repeat_seed}): FAILED - {why}")
                    if flaky_detail is None:
                        flaky_detail = (f"rollout {repeat_idx}/{n_rollouts} "
                                        f"(planner seed {repeat_seed}) "
                                        f"FAILED: {why}")
            if flaky_detail is None:
                validation_note = (
                    f" Validated {n_rollouts}/{n_rollouts} rollouts "
                    f"(planner seeds {base_planner_seed}-"
                    f"{base_planner_seed + n_rollouts - 1}; fresh env and "
                    "fresh policy memory per rollout).")

        # Parameter-margin gates, mirroring submit_plan (one
        # shared code path: see _parameter_margin_sweep).
        param_sensitive_detail: Optional[str] = None
        margin_outcomes: List[str] = []
        if (fresh_scope is not None and ctx.capture_goal_reaching_plans
                and goal_achieved and not evaluator_rejected
                and flaky_detail is None):
            margin_outcomes, param_sensitive_detail, margin_note = \
                _parameter_margin_sweep(ctx, validation_cfg, fresh_scope,
                                        _policy_validation_rollout, "policy")
            validation_note += margin_note

        capture_outcome = _decide_capture(
            capture_enabled=(ctx.capture_goal_reaching_plans
                             and ctx.policy_capture_mode),
            is_current_task=True,
            have_plan=True,
            goal_achieved=goal_achieved,
            evaluator_rejected=evaluator_rejected,
            reward_hack=reward_hack,
            flaky=flaky_detail is not None,
            best_effort_mode=ctx.capture_best_effort_plan,
            have_validated_capture=bool(ctx.solved_plan_reached_goal),
            param_sensitive=param_sensitive_detail is not None)
        decision = capture_outcome.decision
        captured = capture_outcome.captured
        if captured:
            validated_solve = decision is CaptureDecision.VALIDATED_CAPTURE
            ctx.solved_plan = None
            ctx.solved_sketch = None
            ctx.solved_policy_source = policy_source
            ctx.solved_plan_reached_goal = validated_solve
            ctx.solved_plan_eval_reward = (float(verdict["reward"])
                                           if verdict is not None else None)
            summary_bits = [
                f"validation: "
                f"{1 + sum(1 for o in rollout_outcomes if 'FAILED' not in o)}"
                f"/{1 + len(rollout_outcomes)} rollouts ok"
            ]
            if flaky_detail is not None:
                summary_bits.append(f"first failure: {flaky_detail}")
            if margin_outcomes:
                n_margin_ok = sum(1 for o in margin_outcomes
                                  if "FAILED" not in o)
                summary_bits.append(f"physics margin: {n_margin_ok}/"
                                    f"{len(margin_outcomes)} points ok")
            ctx.solved_plan_validation_summary = "; ".join(summary_bits)
            reason = capture_outcome.best_effort_reason
            best_effort_note = ""
            if reason is not None:
                best_effort_note = (
                    " (best-effort: accepted because the attempt budget is "
                    "exhausted; it executes for its honest reward but may "
                    "not count as a solve)")
            lines.append(
                f"Captured policy.py as the current answer{best_effort_note}"
                f": {len(result.steps)} option(s) in the capture rollout."
                f"{validation_note}")
        elif decision is CaptureDecision.FLAKY_NO_CAPTURE:
            ctx.flaky_capture_task_keys.add(capture_task_key)
            per_rollout = "\n".join(f"  {o}" for o in rollout_outcomes)
            n_ok = 1 + sum(1 for o in rollout_outcomes if "FAILED" not in o)
            lines.append(
                f"FLAKY (policy NOT captured): rollout 1 reached the goal "
                f"but {flaky_detail}. Per-rollout outcomes (estimated "
                f"reliability {n_ok}/{n_rollouts}):\n{per_rollout}\n"
                "A closed-loop policy that cannot recover in some rollouts "
                "needs better feedback handling - inspect the failing "
                "seeds with rollout_seed and strengthen the recovery "
                "branches.")
        elif decision is CaptureDecision.PARAM_SENSITIVE_NO_CAPTURE:
            per_point = "\n".join(f"  {o}" for o in margin_outcomes)
            lines.append(f"PARAM-SENSITIVE (policy NOT captured): failed "
                         f"{param_sensitive_detail}. Per-point outcomes:\n"
                         f"{per_point}")
        elif decision is CaptureDecision.REWARD_HACK_NO_CAPTURE:
            lines.append(
                "NOT captured: the rollout reaches the goal atoms but the "
                "task evaluator scores it as a non-solve, and the real env "
                "applies the same scoring.")

        lines.append(f"Goal achieved: {goal_reached}")
        if verdict is not None:
            lines.append(
                _format_evaluator_verdict(verdict,
                                          coarse=eval_collector.coarse))
        if goal_reached and not within_horizon:
            lines.append(
                f"NOT EXECUTABLE: reaching the goal takes "
                f"{result.actions_to_goal} low-level steps but the episode "
                f"horizon is {horizon}.")
        if not goal_reached:
            missing = _missing_goal_atoms(task, result.final_state)
            missing_str = ", ".join(str(a) for a in sorted(missing))
            lines.append(f"Missing goal atoms: {{{missing_str}}}")
        if saved_image_paths:
            lines.append("\nSaved images:")
            lines.extend(f"  {p}" for p in saved_image_paths)
        return _text_result("\n".join(lines) +
                            _budget_footer(ctx, rollouts_before))

    return {
        "submit_plan": submit_plan,
        "submit_policy": submit_policy,
    }
