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
import os
import time
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from predicators import utils
from predicators.agent_sdk import bilevel_sketch
from predicators.agent_sdk.bilevel_sketch import SketchStep as _SketchStep
from predicators.agent_sdk.tools import BUILTIN_TOOLS
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
        # Whether the most recent sketch query ended because the agent hit
        # agent_sdk_max_agent_turns_per_iteration. Set by
        # _query_agent_for_plan_sketch, read by _solve to decide between
        # retrying (a real error) and accepting the nudged best-effort
        # submission (budget exhaustion).
        self._last_sketch_query_hit_turn_cap = False

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
        # Optionally give each test solve a fresh agent conversation. reset()
        # fires once per test task (not on mid-episode replans, which go
        # through step()); the next query lazily rebuilds the session with the
        # same sandbox + artifacts but empty chat context. Test-phase only, so
        # exploration episodes keep their shared session.
        if CFG.agent_fresh_session_per_test_task and self._in_test_phase:
            self._close_agent_session()

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

    def _get_solve_tool_names(self) -> Optional[List[str]]:
        # Bilevel solving hands continuous refinement to a search, so the
        # agent also gets refine_plan_sketch (backtracking refinement +
        # forward validation on a param-free sketch). Needs a simulator.
        tools = list(super()._get_solve_tool_names() or [])
        if CFG.agent_planner_use_simulator:
            tools.append("refine_plan_sketch")
        return tools

    # ------------------------------------------------------------------ #
    # System prompt (simplified — no parameter tuning workflow)
    # ------------------------------------------------------------------ #

    def _get_agent_system_prompt(self) -> str:
        propose = CFG.agent_bilevel_use_llm_initial_params
        # When True the approach skips its own (post-agent) backtracking
        # refinement, so the agent must itself deliver a plan it validated via
        # refine_plan_sketch. (The agent still triggers a backtracking search
        # inside that tool; what's skipped is the approach's separate one.)
        skip_final_backtracking_search = not CFG.agent_bilevel_refine_fallback
        # What a sketch step consists of (shared between modes).
        if propose:
            sketch_desc = (
                "a sequence of skills (parameterized options) with object "
                "arguments, subgoal atoms after each step, and the continuous "
                "parameters for each step")
        else:
            sketch_desc = (
                "a sequence of skills (parameterized options) with object "
                "arguments and subgoal atoms after each step, plus continuous "
                "parameters you find with refine_plan_sketch")

        if skip_final_backtracking_search:
            # The deliverable is a plan that WORKS IN THE SIMULATOR, submitted
            # by evaluate_option_plan; refine_plan_sketch is only a (slower)
            # aid for finding parameters while reasoning.
            job = (
                "Your job is to produce a plan that WORKS IN THE SIMULATOR — "
                + sketch_desc +
                " that reaches the goal. You DELIVER it by running "
                "evaluate_option_plan with per-step subgoals on the current "
                "task until it reaches the goal — that captured plan is your "
                "ONLY output, so do not finish until evaluate_option_plan "
                "reaches the goal. It runs your EXACT parameters with no "
                "sampling, so every parameter must be right. To find working "
                "values you MAY use refine_plan_sketch while reasoning (it "
                "searches for parameters but is slower); read the parameters "
                "it reports and submit them via evaluate_option_plan. Use "
                "whatever tools help (inspection, visualize_state).")
            if propose:
                job += (
                    " Where many values work, any reasonable parameter is "
                    "fine; where good values are hard to hit (tight "
                    "tolerances), use refine_plan_sketch to search for one, "
                    "and confine its search near your estimate by appending "
                    "a region `~ [w1, w2]` of per-parameter half-widths "
                    "after a step's `[params]`.")
        else:
            # Fallback mode: the agent hands off a sketch and the approach's
            # backtracking search refines the continuous parameters.
            job = "Your job is to produce a plan sketch: " + sketch_desc + "."
            if propose:
                job += (
                    " A backtracking search refines the parameters, trying "
                    "yours first and sampling for any step where they don't "
                    "work, so propose precise values only where sampling would "
                    "struggle. You may validate with refine_plan_sketch and "
                    "deep-tune any step it reports stuck.")

        # Keep responses short: the model's deliberation is the main driver of
        # the output-token overflow, and testing is often faster than deriving.
        brevity = (
            " Keep your reasoning concise: prefer making a concrete attempt "
            "and testing it with refine_plan_sketch / evaluate_option_plan to "
            "let the simulator tell you what's wrong.")
        params_clause = job + brevity + "\n\n"
        # Keep the subgoal-annotation template's option format consistent with
        # the solve prompt: show the [params] slot iff the agent proposes them.
        param_slot = "[param1, param2]" if propose else ""
        wait_slot = "[]" if propose else ""
        return (
            "You are a planning agent. You observe task environments through "
            "inspection tools and generate plan sketches to achieve goals. "
            "You have access to read-only tools to inspect predicates, "
            "options, trajectories, and training tasks.\n\n"
            f"{params_clause}"
            "Some effects may not be immediate — if an action triggers a "
            "delayed process (e.g. gradual accumulation, propagation "
            "through contacting objects, a sensor catching up to an "
            "actuator), insert a Wait after it so the effect has time "
            "to occur before the next action.\n\n"
            "## Subgoal Annotations\n"
            "After each step, annotate which predicate atoms should hold "
            "after that step succeeds. Use the format:\n"
            f"  OptionName(obj1:type1, obj2:type2){param_slot} -> "
            "{Pred(obj1:type1), Pred2(obj1:type1, obj2:type2)}\n"
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
            f"become false (e.g. `Wait(robot:robot){wait_slot} -> "
            "{Ready(widget:widget)}`).")

    # ------------------------------------------------------------------ #
    # Solve prompt (no continuous params, subgoal format)
    # ------------------------------------------------------------------ #

    def _build_solve_prompt(self,
                            task: Task,
                            prior_failures: Optional[List[str]] = None) -> str:
        """Build prompt asking for a plan sketch without continuous params."""
        failures_text = "\n\n".join(prior_failures) if prior_failures else ""
        return bilevel_sketch.build_solve_prompt(
            task,
            all_predicates=self._get_all_predicates(),
            all_options=self._get_all_options(),
            trajectory_summary=self._build_trajectory_summary(),
            tool_names=self._solve_prompt_tool_names(),
            prior_failures=failures_text,
            initial_image_section=self._initial_image_section(),
            propose_params=CFG.agent_bilevel_use_llm_initial_params,
            require_tool_validation=not CFG.agent_bilevel_refine_fallback,
        )

    def _solve_prompt_tool_names(self) -> Optional[List[str]]:
        """Tool list advertised in the solve prompt's "Available Tools".

        Mirrors what the explore prompt lists (the explorer renders
        ``agent_session.tool_names``): the same MCP subset *plus* the
        sandbox's built-in tools (Bash/Read/Write/...). The built-ins are
        only actually granted under the local or docker sandbox -- which
        is exactly when ``LocalSandboxSessionManager.tool_names`` prepends
        them -- so they are advertised only then. Without a sandbox the
        list is the bare MCP subset, unchanged.
        """
        names = self._get_solve_tool_names()
        if names is None:
            return None
        if CFG.agent_sdk_use_local_sandbox or CFG.agent_sdk_use_docker_sandbox:
            return list(BUILTIN_TOOLS) + names
        return names

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
        # Let refine_plan_sketch / evaluate_option_plan record a goal-reaching
        # plan on this task into solved_plan/solved_sketch (consumed below).
        self._tool_context.capture_goal_reaching_plans = True
        # Render the initial state so the agent can see the scene layout.
        self._render_initial_state_image(task)
        start = time.perf_counter()
        # Exclude the (minutes-long) LLM sketch query from the refinement
        # budget, else a slow query overruns `timeout` and starves the
        # refine loop -- failing the solve without ever refining.
        llm_query_time = 0.0

        def _refine_remaining() -> float:
            elapsed = time.perf_counter() - start - llm_query_time
            return timeout - elapsed

        sketches_tried = 0
        # Pre-formatted summaries of earlier sketches the search could not
        # refine; threaded into the next sketch query so the agent revises
        # the dead skeleton instead of re-emitting it.
        prior_failures: List[str] = []
        for sketch_attempt in range(max_sketch_retries):
            if _refine_remaining() <= 0:
                break
            # Clear any prior capture so we only act on this query's result.
            self._tool_context.solved_plan = None
            self._tool_context.solved_sketch = None
            self._tool_context.solved_plan_reached_goal = None
            self._last_sketch_query_hit_turn_cap = False
            query_start = time.perf_counter()
            try:
                sketch = self._query_agent_for_plan_sketch(
                    task, prior_failures=prior_failures)
            except Exception as e:  # pylint: disable=broad-except
                llm_query_time += time.perf_counter() - query_start
                # The agent may have validated a working plan via
                # refine_plan_sketch even if its final text didn't parse.
                policy = self._consume_validated_plan()
                if policy is not None:
                    return policy
                # On output-token overflow, tell the next attempt (same
                # conversation) to be concise. The overflow is one over-long
                # RESPONSE; the SDK compacts context itself, so only brevity,
                # not compaction, prevents a repeat.
                if "output token maximum" in str(e):
                    prior_failures.append(
                        "Your previous response hit the output-token limit and "
                        "was discarded, so no plan was produced. Your extended "
                        "thinking is capped — do not spend a whole response on "
                        "long derivations.")
                logging.warning("Sketch query failed (attempt %d): %s",
                                sketch_attempt, e)
                hit_cap = self._last_sketch_query_hit_turn_cap
                nudge_start = time.perf_counter()
                policy = self._nudge_final_submission(
                    accept_best_effort=hit_cap)
                llm_query_time += time.perf_counter() - nudge_start
                if policy is not None:
                    return policy
                if hit_cap:
                    # The turn cap is a budget end, not a retryable error:
                    # a fresh full-budget attempt re-explores from scratch
                    # at full cost with no new information. The nudge above
                    # already accepted the agent's best-effort submission;
                    # with nothing captured even then, give up on this task.
                    break
                continue
            llm_query_time += time.perf_counter() - query_start
            sketches_tried += 1

            # Fast path: the agent already refined + forward-validated a plan
            # on this task via refine_plan_sketch — return it directly instead
            # of re-refining the (possibly different) final-text sketch.
            policy = self._consume_validated_plan()
            if policy is not None:
                return policy

            if not CFG.agent_bilevel_refine_fallback:
                # Default: no approach-side fallback. The agent must itself
                # reach a confirmed evaluate_option_plan capture (consumed
                # above) so we never execute a plan it didn't verify; if it
                # ended without one, re-query with feedback instead of
                # refining its unvalidated sketch.
                logging.info(
                    "[%s] Attempt %d ended without a validated plan; "
                    "re-querying the agent.", self._run_id, sketch_attempt)
                hit_cap = self._last_sketch_query_hit_turn_cap
                nudge_start = time.perf_counter()
                policy = self._nudge_final_submission(
                    accept_best_effort=hit_cap)
                llm_query_time += time.perf_counter() - nudge_start
                if policy is not None:
                    return policy
                if hit_cap:
                    # See the identical break in the exception path: turn-cap
                    # exhaustion is not retryable, and the best-effort nudge
                    # already captured whatever the agent could submit.
                    break
                prior_failures.append(
                    "You finished without a validated plan. You MUST run "
                    "evaluate_option_plan on the current task (omit task_idx) "
                    "until it confirms a capture; that captured run is your "
                    "submitted answer. A plan given only as text is "
                    "discarded, and refine_plan_sketch does NOT submit - it "
                    "only finds parameters for you to submit via "
                    "evaluate_option_plan.")
                continue

            sketch_lines = bilevel_sketch.format_sketch_lines(sketch)
            logging.info("[%s] Sketch (attempt %d):\n%s", self._run_id,
                         sketch_attempt, "\n".join(sketch_lines))

            # Aggregate per-step failures across this sketch's refine
            # retries (same skeleton, so the obstruction is the same):
            # deepest step the search reached, and a tally of the distinct
            # failure reasons it hit there and earlier.
            record_fail, fail_state = self._make_step_fail_recorder()

            # Resample continuous params with a fresh seed before paying
            # for another agent query: a sketch that refines but fails
            # forward validation is a continuous-params problem, not a
            # wrong skeleton, and re-querying rarely changes the skeleton
            # while always costing an LLM call.
            for refine_attempt in range(max_refine_retries):
                remaining = _refine_remaining()
                if remaining <= 0:
                    break
                # Flatten the two loop indices so every (sketch, refine)
                # pair draws a unique seed in _refine_sketch.
                seed_offset = (sketch_attempt * max_refine_retries +
                               refine_attempt)
                plan, success = self._refine_sketch(task,
                                                    sketch,
                                                    remaining,
                                                    attempt=seed_offset,
                                                    on_step_fail=record_fail)
                if not success:
                    reason_msg = ""
                    if fail_state["deepest_idx"] >= 0:
                        reason_msg = (
                            f" (stuck at step {fail_state['deepest_idx']}: "
                            f"{fail_state['deepest_reason']})")

                    logging.info(
                        f"Refinement failed (sketch "
                        f"{sketch_attempt}, refine {refine_attempt}), "
                        f"{len(sketch)} steps{reason_msg}.")
                    continue

                plan_str = "\n".join(bilevel_sketch.format_plan_lines(plan))
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

            # Every refine retry for this skeleton failed: save a full
            # per-step refinement log to the sandbox and add a preview +
            # pointer so the next sketch query revises this dead skeleton.
            preview = self._record_refinement_failure(
                sketch_attempt, sketch_lines, sketch,
                fail_state["deepest_idx"], fail_state["deepest_reason"],
                fail_state["counts"])
            if preview:
                prior_failures.append(preview)

        raise ApproachFailure(
            f"Bilevel solve failed after {sketches_tried} sketch(es) "
            f"(LLM query time {llm_query_time:.1f}s excluded from the "
            f"{timeout}s refinement budget).")

    # ------------------------------------------------------------------ #
    # Plan sketch extraction
    # ------------------------------------------------------------------ #

    def _query_agent_for_plan_sketch(
            self,
            task: Task,
            prior_failures: Optional[List[str]] = None) -> List[_SketchStep]:
        """Query agent for a plan sketch and parse it.

        ``prior_failures`` carries preview+pointer blocks for earlier
        sketches the search could not refine; they are injected into the
        prompt so the re-query revises the dead skeleton.
        """
        sketch_file = CFG.agent_bilevel_plan_sketch_file
        if sketch_file:
            # An absolute path is used as-is; a bare filename is resolved
            # against the configured plan-sketch directory under scripts/.
            if os.path.isabs(sketch_file):
                filepath = sketch_file
            else:
                filepath = (
                    f"{utils.get_path_to_predicators_root()}/scripts/"
                    f"{CFG.agent_bilevel_plan_sketch_dir}/{sketch_file}")
            with open(filepath, "r", encoding="utf-8") as f:
                plan_text = f.read().strip()
            logging.info("Loaded plan sketch from file: %s", sketch_file)
        else:
            prompt = self._build_solve_prompt(task,
                                              prior_failures=prior_failures)
            responses = self._query_agent_sync(prompt, kind="test")
            # Record cap-exhaustion before parsing: a capped session usually
            # has no final text, so the "empty plan text" failure below must
            # still be attributable to the turn cap by _solve.
            self._last_sketch_query_hit_turn_cap = \
                self._responses_hit_turn_cap(responses)
            plan_text = self._extract_option_plan_text(responses)

        if not plan_text:
            raise ApproachFailure("Agent returned empty plan text.")

        sketch = bilevel_sketch.parse_sketch_from_text(
            plan_text,
            task,
            predicates=self._get_all_predicates(),
            options=self._get_all_options(),
            types=self._types,
            parse_continuous_params=CFG.agent_bilevel_use_llm_initial_params,
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

    @staticmethod
    def _responses_hit_turn_cap(responses: List[Dict[str, Any]]) -> bool:
        """Whether a query's response stream ended on the SDK turn cap.

        The SDK reports the cap as result subtype ``error_max_turns``;
        the num_turns comparison is a fallback for backends whose result
        entries lack the subtype field.
        """
        max_turns = CFG.agent_sdk_max_agent_turns_per_iteration
        for entry in responses:
            if entry.get("type") != "result":
                continue
            if entry.get("subtype") == "error_max_turns":
                return True
            num_turns = entry.get("num_turns")
            if num_turns is not None and num_turns >= max_turns:
                return True
        return False

    @staticmethod
    def _make_step_fail_recorder(
    ) -> Tuple[Callable[[int, List[Optional[_Option]], str], None], "dict"]:
        """Build an ``on_step_fail`` callback and its accumulator state.

        Returns ``(callback, state)`` where ``state`` is a dict with
        keys ``deepest_idx`` (the deepest step index the search reached
        before failing), ``deepest_reason`` (the failure reason there),
        and ``counts`` (a ``Counter`` over ``(step_idx, reason)``).
        Built as a factory so the closure captures fresh per-sketch
        state instead of loop variables.
        """
        state: dict = {
            "deepest_idx": -1,
            "deepest_reason": "",
            "counts": Counter(),
        }

        def _record(idx: int, _plan: List[Optional[_Option]],
                    reason: str) -> None:
            state["counts"][(idx, reason)] += 1
            if idx > state["deepest_idx"]:
                state["deepest_idx"] = idx
                state["deepest_reason"] = reason

        return _record, state

    def _record_refinement_failure(
        self,
        attempt_idx: int,
        sketch_lines: List[str],
        sketch: List[_SketchStep],
        deepest_idx: int,
        deepest_reason: str,
        reason_counts: "Counter[Tuple[int, str]]",
    ) -> str:
        """Persist a full refinement-failure log to the sandbox and return a
        preview+pointer block for the next sketch prompt.

        Writes ``<sandbox>/refinement_logs/sketch_<NN>_refine.md`` with the
        tried skeleton, where backtracking got stuck (deepest step), and a
        per-step tally of the distinct failure reasons. The returned block
        embeds a short preview and a relative pointer to that file so the
        agent can ``Read`` the detail. Returns ``""`` if there is nothing
        to report (no recorded failures).
        """
        if not reason_counts:
            return ""

        def _step_desc(idx: int) -> str:
            if 0 <= idx < len(sketch):
                objs = ", ".join(o.name for o in sketch[idx].objects)
                return f"step {idx}: {sketch[idx].option.name}({objs})"
            return f"step {idx}"

        total_fail = sum(reason_counts.values())
        deepest_desc = _step_desc(deepest_idx)

        full_lines = [
            f"# Refinement failure — sketch attempt {attempt_idx}",
            "",
            "## Sketch (could not be refined)",
            *sketch_lines,
            "",
            "## Outcome",
            f"FAILED. Deepest step the search reached: {deepest_desc}.",
            f"Dominant failure there: {deepest_reason}",
            f"Total failed samples: {total_fail}.",
            "",
            "## Per-step failure reasons (count)",
        ]
        for (idx, reason), cnt in sorted(reason_counts.items(),
                                         key=lambda kv: (kv[0][0], -kv[1])):
            full_lines.append(f"- {_step_desc(idx)}: {cnt}x  {reason}")
        full_text = "\n".join(full_lines) + "\n"

        # Prefer the agent-visible sandbox cwd so the pointer is a valid
        # relative path for the agent; fall back to the run log dir.
        sandbox = getattr(self._tool_context, "sandbox_dir", None) \
            or self._get_log_dir()
        rel_dir = "refinement_logs"
        out_dir = os.path.join(sandbox, rel_dir)
        os.makedirs(out_dir, exist_ok=True)
        fname = f"sketch_{attempt_idx:02d}_refine.md"
        try:
            with open(os.path.join(out_dir, fname), "w",
                      encoding="utf-8") as f:
                f.write(full_text)
            pointer = f"./{rel_dir}/{fname}"
        except OSError as e:  # pragma: no cover - best-effort logging
            logging.warning("Could not write refinement log: %s", e)
            pointer = "(refinement log unavailable)"

        preview = "\n".join([
            f"### Attempt {attempt_idx} (FAILED)",
            *sketch_lines,
            f"  -> Refinement FAILED. Deepest step reached: {deepest_desc}. "
            f"Dominant failure: {deepest_reason} "
            f"({total_fail} failed samples).",
            f"  Full per-step refinement log: {pointer}",
        ])
        return preview

    # ------------------------------------------------------------------ #
    # Backtracking refinement
    # ------------------------------------------------------------------ #

    def _refine_sketch(
        self,
        task: Task,
        sketch: List[_SketchStep],
        timeout: float,
        attempt: int = 0,
        on_step_fail: Optional[Callable[[int, List[Optional[_Option]], str],
                                        None]] = None,
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
            on_step_fail=on_step_fail,
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
        monitor-triggered replan; ``_exec_status`` distinguishes them
        (non-None only while a monitored plan executes;
        reset_for_new_episode clears it at episode start). On a replan
        ``task.init`` is the real state where the just-finished step's
        annotation failed. Divergence is usually a continuous-execution
        problem (a sampled parameter whose real outcome differed from
        the option-model rollout), not a wrong skeleton, so we first try
        to resume a suffix of the executed sketch (cheap, no agent
        query; see :meth:`_replan_suffix`). Returns None to fall through
        to a fresh agent sketch; raises ApproachFailure when the
        episode's replan budget is exhausted so the episode fails fast
        instead of running the horizon open-loop.
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
            # No suffix of the executed skeleton refines from here; fall
            # through to pay for a fresh agent sketch.
            logging.info("Suffix replan failed; querying the agent for a "
                         "fresh sketch.")
        return policy

    _FINAL_SUBMIT_NUDGE = (
        "You are out of exploration budget for this attempt. Do NOT explore "
        "further. In as few tool calls as possible, submit your single best "
        "plan NOW via evaluate_option_plan on the current task (omit "
        "task_idx), using the best parameters you have already validated. "
        "If it reaches the goal it is captured as your answer; then finish.")

    _FINAL_SUBMIT_NUDGE_BEST_EFFORT = (
        "You are out of exploration budget for this attempt. Do NOT explore "
        "further. In as few tool calls as possible, submit your single best "
        "plan NOW via evaluate_option_plan on the current task (omit "
        "task_idx), using the best parameters you have already validated. "
        "It is captured as your answer even if it does not fully reach the "
        "goal; then finish.")

    def _nudge_final_submission(
        self,
        accept_best_effort: bool = False,
    ) -> Optional[Callable[[State], Action]]:
        """One short follow-up query after an attempt ended with no captured
        plan: tell the agent to submit its best plan now.

        A session that hits the turn cap mid-iteration contributes
        nothing, even when it has a near-working plan in context; this
        converts that dead end into a submission attempt at the cost of a
        few turns.

        With ``accept_best_effort`` (set when the attempt ended on the
        turn cap rather than an error) the submitted plan is captured
        and executed even if its belief rollout does not reach the goal:
        the budget is spent, and a partial plan beats forfeiting the
        task after more full-budget retries.
        """
        nudge = (self._FINAL_SUBMIT_NUDGE_BEST_EFFORT
                 if accept_best_effort else self._FINAL_SUBMIT_NUDGE)
        self._tool_context.capture_best_effort_plan = accept_best_effort
        try:
            self._query_agent_sync(nudge, kind="test")
        except Exception as e:  # pylint: disable=broad-except
            logging.warning("Final-submission nudge failed: %s", e)
        finally:
            self._tool_context.capture_best_effort_plan = False
        policy = self._consume_validated_plan()
        if policy is not None:
            logging.info(
                "[%s] Final-submission nudge produced a validated plan.",
                self._run_id)
        return policy

    def _consume_validated_plan(self) -> Optional[Callable[[State], Action]]:
        """Return a policy from an agent-validated plan, or None.

        ``evaluate_option_plan`` records a captured (goal-reaching,
        validated) plan on the current solve task into the tool context.
        Returning that exact simulator-verified plan guarantees the
        agent's tool-validated answer is what executes, and avoids a
        fresh refinement that with a different seed might not reproduce
        it.
        """
        plan = self._tool_context.solved_plan
        sketch = self._tool_context.solved_sketch
        reached_goal = self._tool_context.solved_plan_reached_goal
        self._tool_context.solved_plan = None
        self._tool_context.solved_sketch = None
        self._tool_context.solved_plan_reached_goal = None
        if not plan:
            return None
        # Log the full validated plan (options + continuous params + subgoal
        # annotations), mirroring the per-step plan log the approach-side
        # refinement path emits.
        lines = bilevel_sketch.format_plan_lines(plan, sketch=sketch)
        verdict = ("simulator-verified" if reached_goal is not False else
                   "best-effort: belief rollout did NOT reach the goal")
        logging.info(
            "[%s] Using agent-validated plan from refine_plan_sketch "
            "(%d steps, %s):\n%s", self._run_id, len(plan), verdict,
            "\n".join(lines))
        return self._plan_to_policy(plan, sketch=sketch)

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

        inner = utils.option_policy_to_policy(
            _option_policy,
            max_option_steps=CFG.max_num_steps_option_rollout,
            abstract_function=_abstract)
        return self._wrap_option_failures(inner)

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
