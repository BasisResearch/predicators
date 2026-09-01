"""Agent bilevel explorer: sketch, refine against mental model, execute real.

Produces a plan *sketch* via a Claude agent, runs backtracking refinement
against the approach's currently-learned option model (from
``tool_context.option_model``), then rolls the refined plan out for real.
When the mental model disagrees with reality (e.g. a subgoal atom it
expected after a Wait doesn't actually hold), the trajectory is a targeted
learning signal for online simulator synthesis.

Parallels ``AgentPlanExplorer`` for session plumbing and
``AgentModelBasedApproach`` for the sketch/refine workflow.
"""

import logging
import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Set

import numpy as np
from gym.spaces import Box

from predicators import utils
from predicators.agent_sdk import bilevel_sketch
from predicators.agent_sdk.rendering import save_task_state_image
from predicators.agent_sdk.session_base import AgentSessionFatalError, \
    query_fatal_error
from predicators.agent_sdk.session_manager import SessionManagerProtocol, \
    run_query_sync
from predicators.agent_sdk.tools import PlanCapture, ToolContext, \
    agent_render_resolution, load_ground_sampler_fns
from predicators.explorers.base_explorer import BaseExplorer
from predicators.settings import CFG
from predicators.structs import Action, ExplorationStrategy, \
    ParameterizedOption, Predicate, State, Task, Type


class AgentBilevelExplorer(BaseExplorer):
    """Queries a Claude agent for a plan sketch, refines it, and executes."""

    def __init__(self, predicates: Set[Predicate],
                 options: Set[ParameterizedOption], types: Set[Type],
                 action_space: Box, train_tasks: List[Task],
                 max_steps_before_termination: int, tool_context: ToolContext,
                 agent_session: SessionManagerProtocol) -> None:
        super().__init__(predicates, options, types, action_space, train_tasks,
                         max_steps_before_termination)
        self._tool_context = tool_context
        self._agent_session = agent_session

    @classmethod
    def get_name(cls) -> str:
        return "agent_bilevel"

    # ------------------------------------------------------------------ #
    # Exploration strategy
    # ------------------------------------------------------------------ #

    def _get_exploration_strategy(self, train_task_idx: int,
                                  timeout: int) -> ExplorationStrategy:
        task = self._train_tasks[train_task_idx]
        # The approach syncs tool_context.option_model right before building
        # this explorer, so reading here picks up the latest learned model.
        option_model = self._tool_context.option_model
        assert option_model is not None, \
            "agent_bilevel explorer needs a synced option_model"

        # Reset the per-request mental-model verdict so a stale value can't
        # leak if refinement below throws or falls back to random before
        # producing one.
        self._tool_context.last_mental_model_solved = None

        # A plan this cycle already certified on this task (see the
        # capture branch below) is replayed for the cycle's remaining
        # requests without a new query: the train-driven early-stop rule
        # needs EVERY attempt of the cycle to solve, and a second real
        # execution of the certified plan is the cheapest evidence.
        certified = self._tool_context.cycle_certified_plans.get(
            train_task_idx)
        if certified is not None and \
                CFG.agent_explorer_replay_certified_plan:
            logging.info(
                "agent_bilevel explorer: replaying this cycle's "
                "belief-certified plan for train task %d (%d steps) "
                "without a new query.", train_task_idx, len(certified))
            self._tool_context.last_mental_model_solved = True
            return self._certified_plan_strategy(certified)

        # Point the agent's interactive tools (refine_plan_sketch,
        # evaluate_option_plan, the sim probe) at the EXPLORE task. They
        # default to ctx.current_task when the agent omits task_idx, and
        # test-time _solve leaves current_task on the last TEST task.
        # Without this the agent tunes/validates its exploration plan against
        # the wrong task (e.g. a test goal referencing objects this task
        # lacks), so parameter search is meaningless and only tasks solvable
        # without tuning get solved.
        #
        # Enable the capture path too (keyed to current_task == this explore
        # task): the agent often submits + simulator-validates a goal-reaching
        # plan via evaluate_option_plan / refine_plan_sketch but ends with a
        # prose summary whose final text doesn't parse into a sketch. Without
        # capture that productive solve is lost to the random-options fallback;
        # with it we recover the captured plan below (see _sketch_from_capture)
        # and feed its continuous params into the info-gain search. Clear any
        # stale capture first; the next test _solve re-points current_task and
        # clears capture again, so an exploration plan can't leak into a test
        # solve.
        self._tool_context.current_task = task
        self._tool_context.capture_goal_reaching_plans = True
        # Exploration delivers plan sketches even in policy-mode configs:
        # make sure a solve attempt's policy_capture_mode never bleeds
        # into this query (it would disable the plan capture gate).
        self._tool_context.policy_capture_mode = False
        self._tool_context.clear_plan_capture()

        try:
            prompt = bilevel_sketch.build_solve_prompt(
                task,
                all_predicates=self._predicates,
                all_options=self._options,
                trajectory_summary=self._build_trajectory_summary(),
                tool_names=self._agent_tool_names(),
                experiment_guidance=self._build_experiment_guidance(),
                # Plans generated by this cycle's earlier requests: ask for a
                # complementary plan instead of the identical one repeated.
                scheduled_plans=list(self._tool_context.cycle_scheduled_plans),
                initial_image_section=self._initial_image_section(
                    task, train_task_idx),
                propose_params=CFG.agent_bilevel_use_llm_initial_params,
                # The explorer refines its own sketch for exploration; it does
                # not use the approach's tool-validated capture path.
                require_tool_validation=False,
                # Explore contract: the sketch is a real-env experiment, and
                # the belief model may lack goal-critical dynamics, so a
                # simulator-failing sketch is a valid deliverable.
                explore_mode=True,
                ground_samplers=CFG.agent_bilevel_ground_samplers,
            )
            responses = run_query_sync(self._agent_session,
                                       prompt,
                                       kind="explore")
            dead = query_fatal_error(responses)
            if dead is not None:
                # The session backend refused the query (usage limit,
                # auth, transport): a random-options episode here would
                # be junk data that the cycle then learns from (2026-08-28
                # run_20260827_171610 cycle 3). Terminate the run instead;
                # the relaunch re-explores this cycle.
                raise AgentSessionFatalError(
                    "explore query died without the agent doing any work "
                    f"({dead}); not falling back to random exploration.")
            plan_text = self._extract_option_plan_text(responses)
            # The session's tool capture: a goal-reaching plan the agent
            # validated in the belief through the capture gate
            # (evaluate_option_plan / refine_plan_sketch, N fresh
            # rollouts). ``reached_goal`` is the gate's verdict.
            capture = self._tool_context.take_plan_capture()
            if CFG.agent_explorer_replay_certified_plan and capture.plan \
                    and capture.reached_goal is True:
                # Certified: the mental model solves the task with THIS
                # plan, so run it verbatim as a solve attempt instead of
                # re-searching (or boundary-probing) its parameters. A
                # real success now counts for early stopping.
                plan = list(capture.plan)
                logging.info(
                    "agent_bilevel explorer: the agent's tool-validated "
                    "plan passed the belief's capture gate (%s); executing "
                    "it verbatim as this episode's solve attempt (mental "
                    "model solved the goal).", capture.validation_summary
                    or "goal reached")
                if capture.sketch:
                    self._tool_context.last_sketch_subgoals = [
                        (s.subgoal_atoms, s.subgoal_neg_atoms)
                        for s in capture.sketch
                    ]
                    self._tool_context.last_sketch_options = [
                        (s.option.name, [o.name for o in s.objects])
                        for s in capture.sketch
                    ]
                self._tool_context.last_mental_model_solved = True
                self._tool_context.cycle_certified_plans[train_task_idx] = plan
                self._tool_context.cycle_scheduled_plans.append(
                    self._format_plan(plan) +
                    "\n  NOTE: belief-certified; executes verbatim as a "
                    "solve attempt and is replayed for this cycle's "
                    "remaining episodes.")
                return self._certified_plan_strategy(plan)
            if not plan_text and not capture.plan:
                raise ValueError("agent returned empty plan text")

            gs_fns, gs_err = load_ground_sampler_fns(self._tool_context)
            if gs_err is not None:
                logging.warning("[explore] %s", gs_err)
            sketch = bilevel_sketch.parse_sketch_from_text(
                plan_text,
                task,
                predicates=self._predicates,
                options=self._options,
                types=self._types,
                parse_continuous_params=CFG.
                agent_bilevel_use_llm_initial_params,
                parse_ground_samplers=CFG.agent_bilevel_ground_samplers,
                ground_sampler_fns=gs_fns or None,
            ) if plan_text else []
            if not sketch:
                # Final message didn't parse into a sketch, but the agent may
                # have submitted + simulator-validated a goal-reaching plan via
                # evaluate_option_plan / refine_plan_sketch (captured into
                # solved_plan / solved_sketch). Recover it as the sketch,
                # carrying its continuous params as initial_params so they seed
                # the info-gain search below rather than replaying verbatim.
                # Mirrors the test solver's preference for the tool-validated
                # capture over the final text.
                sketch = self._sketch_from_capture(capture) or []
            if not sketch:
                raise ValueError("parsed empty plan sketch")

            self._tool_context.last_sketch_subgoals = [
                (s.subgoal_atoms, s.subgoal_neg_atoms) for s in sketch
            ]
            self._tool_context.last_sketch_options = [
                (s.option.name, [o.name for o in s.objects]) for s in sketch
            ]

            # Log the sketch + subgoal annotations the learner will refine
            # (mirrors the solver's sketch log). Subgoal-annotated steps are
            # the ones info-seeking can turn into boundary probes.
            sketch_lines = []
            for i, s in enumerate(sketch):
                objs = ", ".join(o.name for o in s.objects)
                line = f"  {i}: {s.option.name}({objs})"
                if s.initial_params is not None and len(s.initial_params):
                    par = ", ".join(f"{p:.4f}" for p in s.initial_params)
                    line += f"[{par}]"
                if s.subgoal_atoms:
                    atoms = ", ".join(str(a) for a in s.subgoal_atoms)
                    line += f" -> {{{atoms}}}"
                sketch_lines.append(line)
            logging.info(
                "agent_bilevel explorer: refining sketch for train task %d "
                "(%d steps):\n%s", train_task_idx, len(sketch),
                "\n".join(sketch_lines))
            # Record this request's plan so the cycle's NEXT explore query
            # (generated before anything executes) can differ from it.
            self._tool_context.cycle_scheduled_plans.append(
                "\n".join(sketch_lines))

            # Explorer mode: keep BOTH subgoal and final-goal validation ON so
            # the mental model reports the deepest step it cannot predict - a
            # per-step subgoal it can't establish, or (at the final step) the
            # task goal it predicts won't hold. On failure the returned plan
            # keeps the searched prefix (the failing step runs with the exact
            # params the model rejected) and CONTINUES with the sketch's
            # seeded params for the suffix: exploration exists because the
            # belief model is known-wrong, so its inability to certify later
            # steps is a reason to collect the data, not to drop the tail of
            # the designed experiment (a truncation here once discarded the
            # blocking bond test of a bridge episode and cost the next learn
            # session hours of belief-sim guesswork). `success` honestly
            # reflects whether the mental model could reach the goal, so a
            # model that merely executes-but-mispredicts is distinguishable
            # from one that truly solves the task.
            # Active-experiment design: when info-seeking is on, hand
            # refinement the ensemble-disagreement scorer so it picks the most
            # *informative* feasible continuous parameters (those straddling
            # the learned model's decision boundaries) rather than the first
            # feasible sample. Sampling pools feasible candidates within the
            # step's per-node rollout budget (max_samples_per_step) and
            # proposes them best-first across backtracking retries (the ranked
            # remainder is replayed with no new rollouts), so hard-to-satisfy
            # subgoals yield a real argmax without multiplying the budget. Off
            # -> info_scorer is None and refinement behaves as before.
            info_scorer = None
            info_n_feasible_target = 1
            if CFG.agent_explorer_info_seeking:
                info_scorer = self._tool_context.atom_disagreement_fn
                info_n_feasible_target = \
                    CFG.agent_explorer_info_n_feasible_target
                n_annotated = sum(1 for s in sketch
                                  if s.subgoal_atoms is not None)
                logging.info(
                    "agent_bilevel explorer: info-seeking ON "
                    "(pool %d feasible candidates/step within the "
                    "%d-rollout step budget, ensemble size %d) — %d/%d "
                    "steps are subgoal-annotated and eligible for boundary "
                    "probing.%s", info_n_feasible_target,
                    CFG.agent_bilevel_explorer_max_samples_per_step,
                    CFG.agent_explorer_info_ensemble_size, n_annotated,
                    len(sketch), "" if info_scorer is not None else
                    " WARNING: no ensemble scorer wired (atom_disagreement_fn "
                    "is None) — probing disabled.")

            outcome = bilevel_sketch.refine_sketch(
                task,
                sketch,
                option_model,
                predicates=self._predicates,
                timeout=float(timeout),
                rng=self._rng,
                max_samples_per_step=CFG.
                agent_bilevel_explorer_max_samples_per_step,
                check_subgoals=True,
                check_final_goal=True,
                truncate_on_subgoal_fail=True,
                strip_latent_wait_targets=(
                    not self._tool_context.latent_tracking_available),
                log_state=CFG.agent_bilevel_log_state,
                run_id="agent_bilevel_explorer",
                info_scorer=info_scorer,
                info_n_feasible_target=info_n_feasible_target,
                parameterized_samplers=self._tool_context.
                parameterized_samplers,
                pin_proposed_params=CFG.agent_explorer_pin_proposed_params,
                pinned_step_retries=CFG.agent_explorer_pinned_step_retries,
            )
            plan, success = outcome.plan, outcome.success
            # Record the honest verdict so get_interaction_requests can stamp
            # it onto this request: early stopping must not treat a task as
            # solved when the mental model couldn't reach its goal, even if
            # real-env execution of the experiment happens to.
            self._tool_context.last_mental_model_solved = success
            mm_status = ("solved the goal" if success else
                         "did NOT reach the goal — running as experiment")
            logging.info(
                f"agent_bilevel explorer: sketch has {len(sketch)} steps, "
                f"refined {len(plan)} (mental model {mm_status}).")
            seeded_from = outcome.seeded_only_from
            if plan:
                plan_strs = []
                for i, opt in enumerate(plan):
                    obj_s = ", ".join(o.name for o in opt.objects)
                    par_s = ", ".join(f"{p:.4f}" for p in opt.params)
                    mark = (" [seeded-only]" if seeded_from is not None
                            and i >= seeded_from else "")
                    plan_strs.append(
                        f"  {i}: {opt.name}({obj_s})[{par_s}]{mark}")
                logging.info("agent_bilevel explorer: experiment plan:\n%s",
                             "\n".join(plan_strs))
            # Keep the scheduled-plan record honest for the cycle's next
            # explore query: say which steps run without belief-model
            # certification, and whether any sketch tail was dropped for
            # lack of seeds.
            record_notes = []
            if seeded_from is not None:
                record_notes.append(
                    f"steps {seeded_from}..{len(plan) - 1} execute on the "
                    "sketch's seeded params without belief-model "
                    "certification")
            if len(plan) < len(sketch):
                record_notes.append(
                    f"only the first {len(plan)}/{len(sketch)} sketch steps "
                    "execute (later steps lacked seeded params)")
            if record_notes:
                self._tool_context.cycle_scheduled_plans[-1] += (
                    "\n  NOTE: " + "; ".join(record_notes) + ".")

            if plan:
                policy = utils.option_plan_to_policy(
                    plan,
                    abstract_function=lambda s: utils.abstract(
                        s, self._predicates))
                return self._wrap_policy(policy), lambda _: False

            logging.info("agent_bilevel explorer: refinement produced zero "
                         "steps, falling back to random.")
        except AgentSessionFatalError:
            # A random fallback would hide the broken session backend;
            # re-raise so the run terminates.
            raise
        except Exception as e:  # pylint: disable=broad-except
            logging.warning(f"agent_bilevel explorer failed: {e}. "
                            "Falling back to random options.")

        if not CFG.agent_explorer_fallback_to_random:
            raise utils.RequestActPolicyFailure(
                "agent_bilevel explorer failed and fallback disabled.")
        return self._random_options_fallback()

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _sketch_from_capture(
            self,
            capture: PlanCapture) -> Optional[List[bilevel_sketch.SketchStep]]:
        """Rebuild a sketch from a captured, tool-validated plan, or None.

        ``evaluate_option_plan`` / ``refine_plan_sketch`` stash a
        forward-validated, goal-reaching plan on the explore task into
        ``solved_plan`` (grounded options with continuous params) and
        ``solved_sketch`` (the option skeleton plus the subgoals that
        actually held). We reconstruct a sketch from that skeleton and
        graft each captured option's continuous params onto the step's
        ``initial_params``, so the info-gain refinement below seeds them
        as the first candidate in each step's pool (see
        ``_sample_info_seeking``) rather than replaying them verbatim.
        The capture was already taken (consumed) by the caller.
        """
        plan = capture.plan
        captured_sketch = capture.sketch
        if not plan or not captured_sketch:
            return None
        seeded: List[bilevel_sketch.SketchStep] = []
        for i, step in enumerate(captured_sketch):
            params = None
            if i < len(plan):
                params = np.asarray(plan[i].params, dtype=np.float32)
            seeded.append(
                bilevel_sketch.SketchStep(
                    option=step.option,
                    objects=step.objects,
                    subgoal_atoms=step.subgoal_atoms,
                    subgoal_neg_atoms=step.subgoal_neg_atoms,
                    initial_params=params))
        logging.info(
            "agent_bilevel explorer: final text didn't parse, recovered the "
            "agent's tool-validated plan from capture (%d steps); seeding its "
            "continuous params into the info-gain search.", len(seeded))
        return seeded

    @staticmethod
    def _format_plan(plan: Sequence[Any]) -> str:
        """One indented ``i: Option(objs)[params]`` line per grounded
        option."""
        lines = []
        for i, opt in enumerate(plan):
            obj_s = ", ".join(o.name for o in opt.objects)
            par_s = ", ".join(f"{p:.4f}" for p in opt.params)
            lines.append(f"  {i}: {opt.name}({obj_s})[{par_s}]")
        return "\n".join(lines)

    def _certified_plan_strategy(self,
                                 plan: Sequence[Any]) -> ExplorationStrategy:
        """Execute a belief-certified grounded plan verbatim."""
        logging.info("agent_bilevel explorer: certified plan:\n%s",
                     self._format_plan(plan))
        policy = utils.option_plan_to_policy(
            list(plan),
            abstract_function=lambda s: utils.abstract(s, self._predicates))
        return self._wrap_policy(policy), lambda _: False

    def _wrap_policy(
            self, policy: Callable[[State],
                                   Action]) -> Callable[[State], Action]:
        """Convert OptionExecutionFailure into RequestActPolicyFailure.

        Lets the main loop cleanly terminate the episode when the
        refined plan finishes or fails mid-execution (which is exactly
        the disagreement signal we want to collect).
        """

        def _wrapped(state: State) -> Action:
            try:
                return policy(state)
            except utils.OptionExecutionFailure as e:
                raise utils.RequestActPolicyFailure(e.args[0], e.info) from e

        return _wrapped

    def _random_options_fallback(self) -> ExplorationStrategy:
        """Fall back to random option sampling."""

        def fallback_policy(state: State) -> Action:
            del state
            raise utils.RequestActPolicyFailure(
                "Random option sampling failed!")

        policy = utils.create_random_option_policy(self._options, self._rng,
                                                   fallback_policy)
        return policy, lambda _: False

    def _agent_tool_names(self) -> Optional[List[str]]:
        """Return tool names exposed by the current session, if any."""
        return getattr(self._agent_session, "tool_names", None)

    def _initial_image_section(self, task: Task, train_task_idx: int) -> str:
        """Render the explore task's initial state and return a prompt section
        pointing at it, mirroring what test-time solves get.

        Saved as ``train_task{N:03d}_initial_state.png`` so train-task
        scenes are inspectable alongside the test-task init images.
        Empty string when rendering is unavailable (e.g. the sandbox
        isn't created yet, so ``image_save_dir`` is unset).
        """
        env = self._tool_context.env
        save_dir = self._tool_context.image_save_dir
        if env is None or save_dir is None:
            return ""
        img_name = f"train_task{train_task_idx:03d}_initial_state.png"
        with agent_render_resolution():
            saved = save_task_state_image(env, task, save_dir, img_name)
        if saved is None:
            return ""
        # cwd of the agent is the sandbox root, so reference test_images/.
        return ("\n## Initial State Image\n"
                "A rendering of the initial scene has been saved to "
                f"`./test_images/{img_name}`. **Read this image first** to "
                "understand the spatial layout before planning.\n")

    def _build_experiment_guidance(self) -> str:
        """LLM-proposal half of active-experiment design.

        Always injects the learn phase's open-questions ledger (the
        ranked experiment specs it wrote for exploration to run) when
        one exists in the sandbox. When info-seeking is on, additionally
        tell the agent that refinement will turn each annotated step
        into a boundary-probing experiment, and - when an ensemble
        scorer is wired - point it at the predicates the learned model
        is currently most internally uncertain about.
        """
        parts = []
        ledger = self._read_open_questions()
        if ledger:
            parts.append(
                "The learning phase left this ranked ledger of OPEN "
                "QUESTIONS - uncertainties it could not settle from the "
                "data collected so far, each with the experiment that "
                "would settle it. Settling ledger entries is this "
                "episode's highest-value use; design the episode to "
                "cover as many as its step budget allows:\n" + ledger)
        if CFG.agent_explorer_info_seeking:
            parts.append(
                "Your explicit continuous parameters execute exactly as "
                "written. To find the parameters a step could be run at "
                "to teach the model most, call "
                "`sim.suggest_probes(plan_text)`: it rolls your sketch "
                "forward on your own parameters and, per annotated step, "
                "ranks feasible alternatives by the learned model's "
                "ensemble disagreement on the step's subgoal atoms. Adopt "
                "one by writing it into your sketch, only on a step whose "
                "failure the episode can afford; annotate steps with the "
                "geometry/timing you are least sure the model has right.")
            disagreement = self._build_disagreement_summary()
            if disagreement:
                parts.append(disagreement)
        # System-ID gaps from the previous learn phase (synced by the
        # sim-learning approach): what the collected data could NOT
        # support, phrased as experiment objectives. Exploration is the
        # only place those gaps can be filled.
        sysid = getattr(self._tool_context, "sysid_diagnostics", None)
        if sysid:
            parts.append(
                "The previous system-identification fit left gaps that "
                "only new interaction data can close:\n" + sysid)
        return "\n\n".join(parts)

    _MAX_OPEN_QUESTIONS_CHARS = 4000

    def _read_open_questions(self) -> str:
        """The learn phase's ./open_questions.md ledger, or "".

        The ledger is ranked, so on overflow the head is kept. Read
        directly from the sandbox instead of asking the session to go
        find it: the explore query should START from the ledger, not
        spend its budget rediscovering it.
        """
        sandbox_dir = self._tool_context.sandbox_dir
        if not sandbox_dir:
            return ""
        path = os.path.join(sandbox_dir, "open_questions.md")
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read().strip()
        except OSError:
            return ""
        if len(text) > self._MAX_OPEN_QUESTIONS_CHARS:
            text = (text[:self._MAX_OPEN_QUESTIONS_CHARS] +
                    "\n[... ledger truncated; lower-ranked entries omitted]")
        return text

    def _build_disagreement_summary(self) -> str:
        """Name the predicates the ensemble disagrees most about.

        Scans a bounded sample of recent-trajectory states, scores each
        abstract atom's ensemble disagreement via the wired scorer, and
        reports the highest-disagreement predicates. Grounded in the
        actual ensemble, so it points the agent at genuinely-uncertain
        dynamics rather than guesses. Empty when no scorer/trajectories.
        """
        fn = self._tool_context.atom_disagreement_fn
        if fn is None:
            return ""
        all_trajs = (self._tool_context.offline_trajectories +
                     self._tool_context.online_trajectories)
        if not all_trajs:
            return ""
        recent = all_trajs[-CFG.agent_sdk_max_trajectories_in_context:]
        states: List[State] = []
        for traj in recent:
            n = len(traj.states)
            if n == 0:
                continue
            stride = max(1, n // 6)  # <= ~6 states/trajectory to bound cost
            states.extend(traj.states[::stride])
        best: Dict[str, float] = {}
        for s in states:
            for atom in utils.abstract(s, self._predicates):
                try:
                    d = float(fn(s, {atom}))
                except Exception:  # pylint: disable=broad-except
                    continue
                name = atom.predicate.name
                if d > best.get(name, 0.0):
                    best[name] = d
        # One log line with the full ranking (scope note: abstract() yields
        # true atoms only, so a predicate absent here was never measured, not
        # necessarily agreed-upon). All values <= 0.05 -> no guidance: the
        # ensemble is internally confident (or too tight) everywhere.
        all_ranked = sorted(((v, k) for k, v in best.items()), reverse=True)
        logging.info(
            "agent_bilevel explorer: per-predicate max ensemble disagreement "
            "over %d states — %s.", len(states),
            ", ".join(f"{k}={v:.4f}" for v, k in all_ranked) or "(none)")
        ranked = [(v, k) for v, k in all_ranked if v > 0.05][:4]
        if not ranked:
            return ""
        named = ", ".join(f"{k} (disagreement {v:.2f})" for v, k in ranked)
        return ("Across recent trajectories, the learned model is most "
                f"internally uncertain about: {named}. A sketch that puts "
                "these predicates on the critical path will be most "
                "informative.")

    def _build_trajectory_summary(self) -> str:
        """Summarize trajectory data for the agent."""
        all_trajs = (self._tool_context.offline_trajectories +
                     self._tool_context.online_trajectories)
        if not all_trajs:
            return ""

        max_trajs = CFG.agent_sdk_max_trajectories_in_context
        recent = all_trajs[-max_trajs:]
        lines = [
            f"\n## Trajectory Summary ({len(all_trajs)} total, "
            f"showing last {len(recent)})"
        ]

        for i, traj in enumerate(recent):
            n_steps = len(traj.actions)
            init_atoms = utils.abstract(traj.states[0], self._predicates)
            final_atoms = utils.abstract(traj.states[-1], self._predicates)
            new_atoms = final_atoms - init_atoms
            lost_atoms = init_atoms - final_atoms
            lines.append(f"\nTrajectory {i}: {n_steps} steps")
            if new_atoms:
                lines.append(
                    "  Gained: " +
                    f"{', '.join(str(a) for a in sorted(new_atoms, key=str))}")
            if lost_atoms:
                lines.append(
                    "  Lost: " +
                    f"{', '.join(str(a) for a in sorted(lost_atoms, key=str))}"
                )

        return "\n".join(lines)

    def _extract_option_plan_text(self, responses: List[Dict[str,
                                                             Any]]) -> str:
        """Extract plan text from the last assistant text response."""
        last_text_parts: List[str] = []
        for resp in responses:
            if resp.get("type") == "assistant":
                parts = [
                    block.get("text", "") for block in resp.get("content", [])
                    if isinstance(block, dict) and block.get("type") == "text"
                ]
                if parts:
                    last_text_parts = parts
        return "\n".join(last_text_parts)
