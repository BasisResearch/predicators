"""Agent sim-learning approach: learns a simulator program online.

Extends AgentBilevelApproach to learn process dynamics via an
agent-synthesized step-level simulator with parameterized process
rules. Parameters are fitted via emcee ensemble MCMC (training.py).

The approach creates a base oracle (PyBullet with process
dynamics disabled) and composes it with the learned step-level
dynamics into a single simulator function, plugged into a standard
_OracleOptionModel for true per-step interleaving.

Example command::

    python predicators/main.py --env pybullet_boil \
        --approach agent_sim_learning --seed 0 \
        --num_train_tasks 10 --num_test_tasks 5 \
        --num_online_learning_cycles 5 --explorer agent_plan
"""

import inspect
import logging
import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pybullet
from gym.spaces import Box

from predicators import utils
from predicators.agent_sdk.tools import SYNTHESIS_TOOL_NAMES, \
    _SnapshotTarget, create_synthesis_tools, finalize_versioned_snapshot, \
    make_write_snapshot_hook
from predicators.approaches.agent_bilevel_approach import AgentBilevelApproach
from predicators.code_sim_learning.training import ParamSpec, compute_sse, \
    fit_params, log_sse_breakdown
from predicators.code_sim_learning.utils import LearnedSimulator, \
    apply_rules, iter_feature_residuals, merge_updates, \
    read_simulator_components
from predicators.envs import create_new_env
from predicators.ground_truth_models import get_gt_simulator
from predicators.option_model import _OptionModelBase, _OracleOptionModel
from predicators.settings import CFG
from predicators.structs import Action, Dataset, InteractionResult, \
    LowLevelTrajectory, ParameterizedOption, Predicate, State, Task, Type

logger = logging.getLogger(__name__)

# ── Approach ─────────────────────────────────────────────────────


class AgentSimLearningApproach(AgentBilevelApproach):
    """Bilevel planning with a learned step-level simulator.

    During online learning:
    1. Collect trajectories (inherited from AgentBilevelApproach)
    2. Segment into option-level transitions
    3. Synthesize parameterized process rules via Claude agent
    4. Fit rule parameters via emcee ensemble MCMC
    5. Compose with base oracle into a combined simulator
    6. Build _OracleOptionModel with the combined simulator

    During solving:
    - Uses the learned model for plan validation in backtracking
      refinement.
    """

    def __init__(self,
                 initial_predicates: Set[Predicate],
                 initial_options: Set[ParameterizedOption],
                 types: Set[Type],
                 action_space: Box,
                 train_tasks: List[Task],
                 *args: Any,
                 option_model: Optional[_OptionModelBase] = None,
                 **kwargs: Any) -> None:
        # Build the base env and pass the option model in so the parent
        # __init__ doesn't spin up its own full-process env, which
        # would fight this one for the PyBullet GUI client.
        self._base_env = create_new_env(CFG.env,
                                        do_cache=False,
                                        use_gui=CFG.option_model_use_gui,
                                        skip_process_dynamics=True)
        if option_model is None:
            option_model = _OracleOptionModel(initial_options,
                                              self._base_env.simulate)
        super().__init__(initial_predicates,
                         initial_options,
                         types,
                         action_space,
                         train_tasks,
                         *args,
                         option_model=option_model,
                         **kwargs)
        self._learned_simulator: Optional[LearnedSimulator] = None
        # Loss-scope mask for parameter fitting (compute_sse).
        self._process_features: Dict[str, List[str]] = {}
        self._process_rules: Optional[List] = None
        # Always the same dict object — fits update it in place via
        # clear()+update() so _ParamsView (held by invented predicate
        # classifiers) picks up new values without holding a reference
        # to ``self``. Truthy iff a fit has populated it.
        self._fitted_params: Dict[str, float] = {}
        self._fit_sse: float = float("inf")
        self._learning_mode: bool = False
        # Snapshot tags of the most recent simulator / predicates files
        # committed by the synthesis agent — used to stamp newly
        # collected online trajectories with their source-version
        # provenance (consumed in the next learn-phase prompt).
        self._current_simulator_version: Optional[str] = None
        self._current_predicates_version: Optional[str] = None

    @classmethod
    def get_name(cls) -> str:
        return "agent_sim_learning"

    # ── Agent session hooks ──────────────────────────────────────

    def _get_agent_system_prompt(self) -> str:
        if self._learning_mode:
            return self._build_synthesis_system_prompt()
        return super()._get_agent_system_prompt()

    def _get_synthesis_tool_names(self) -> Optional[List[str]]:
        """Complete tool surface for the synthesis agent.

        Combines the static MCP tools the agent may call (the inspect
        family — used to read off option/predicate/type signatures when
        writing rules) with the names of the dynamic synthesis callables
        (``run_python``, ``evaluate_step_fit``, ``report_residuals``,
        ``evaluate_plan_refinement``) attached to
        ``ctx.extra_mcp_tools`` inside :meth:`_synthesize_with_agent`.
        The mixin asserts the attached instances and this list agree.
        """
        return ["inspect_types", "inspect_options", "inspect_trajectories"] +\
            list(SYNTHESIS_TOOL_NAMES)

    # ── Subclass hooks ──────────────────────────────────────────
    # Default implementations are no-ops so subclasses can add
    # predicate-invention (or other) extensions without copying
    # _synthesize_with_agent.

    def _learning_cycle_index(self) -> int:
        """1-indexed cycle number used in versioned snapshot filenames.

        Offline learning is cycle 1; ``_online_learning_cycle`` is
        incremented before each online learn call, so adding 1 keeps the
        offline pass and the first online pass on different indices.
        """
        return self._online_learning_cycle + 1

    def _compute_extra_synthesis_paths(self, base: str) -> Dict[str, str]:
        """Return extra path bindings for the synthesis sandbox."""
        del base
        return {}

    def _extra_synthesis_tools(
        self,
        exec_ns: Dict[str, Any],
        base_pred_triples: List[Tuple[State, Action, State]],
        inferred_hint: Dict[str, List[str]],
        extra_paths: Dict[str, str],
    ) -> List[Any]:
        """Return additional MCP tools to append to the synthesis tool list."""
        del exec_ns, base_pred_triples, inferred_hint, extra_paths
        return []

    def _extra_synthesis_message(self, extra_paths: Dict[str, str]) -> str:
        """Return text to append to the agent's first synthesis message."""
        del extra_paths
        return ""

    def _extra_synthesis_system_prompt(self) -> str:
        """Return text to append to the synthesis system prompt."""
        return ""

    def _post_synthesis_loading(
        self,
        extra_paths: Dict[str, str],
        specs: List[ParamSpec],
    ) -> None:
        """Hook run after the simulator file is loaded post-session.

        ``specs`` are the just-loaded ``PARAM_SPECS``; subclasses may
        seed ``self._fitted_params`` from their ``init_value``s before
        the proper fit runs (useful when loading other artifacts that
        close over ``params``).
        """
        del extra_paths, specs

    def _build_write_snapshot_targets(
        self,
        simulator_file: str,
        versions_dir: str,
        extra_paths: Dict[str, str],
    ) -> List[_SnapshotTarget]:
        """Files the PostToolUse snapshot hook should watch.

        Defaults to just the simulator. Subclasses (e.g. predicate
        invention) may append their own artifacts. ``extra_paths`` is
        the same dict returned by ``_compute_extra_synthesis_paths``.
        """
        del extra_paths
        return [
            _SnapshotTarget(
                live_file=simulator_file,
                versions_dir=versions_dir,
                artifact_name="simulator",
                cycle_index_provider=self._learning_cycle_index,
            ),
        ]

    @staticmethod
    def _build_synthesis_session_hooks(
        targets: List[_SnapshotTarget],
        sandbox_dir: str,
    ) -> Dict[str, list]:
        """Wrap snapshot targets in a Claude Agent SDK ``HookMatcher``.

        Returns the dict suitable for assignment to
        ``ToolContext.extra_session_hooks``. Falls back to an empty dict
        if the SDK ``HookMatcher`` isn't importable (so the approach
        still works against older SDK versions).
        """
        if not targets:
            return {}
        try:
            from claude_agent_sdk import \
                HookMatcher  # pylint: disable=import-outside-toplevel
        except ImportError:
            logger.warning("claude_agent_sdk.HookMatcher unavailable; "
                           "write-time snapshots disabled.")
            return {}
        hook = make_write_snapshot_hook(targets, sandbox_dir=sandbox_dir)
        return {
            "PostToolUse": [
                HookMatcher(matcher="Write|Edit|MultiEdit", hooks=[hook]),
            ],
        }

    # ── Learning ────────────────────────────────────────────────

    def learn_from_offline_dataset(self, dataset: Dataset) -> None:
        super().learn_from_offline_dataset(dataset)
        self._learn_simulator(self._get_all_trajectories())

    def learn_from_interaction_results(
            self, results: Sequence[InteractionResult]) -> None:
        super().learn_from_interaction_results(results)
        self._learn_simulator(self._get_all_trajectories())

    def _learn_simulator(self, trajectories: List[LowLevelTrajectory]) -> None:
        """Synthesize rules, fit parameters, and build the option model."""
        # Two parallel triple lists drive the rest of this method:
        # * obs_triples       — raw (s_t, a, s_{t+1}) from the data.
        # * base_pred_triples — same triples but s_t replaced by the
        #   base sim's one-step prediction. The rules run on top of
        #   that prediction; SSE compares against s_{t+1}.
        obs_triples = self._extract_obs_triples(trajectories)
        if not obs_triples:
            logger.warning("No step transitions; skipping simulator learning.")
            return
        # Headless env for the pre-compute: reusing the GUI base_env
        # corrupts its visual-shape state after a few hundred steps.
        fit_env = create_new_env(CFG.env,
                                 do_cache=False,
                                 use_gui=False,
                                 skip_process_dynamics=True)
        logger.info("Pre-computing base states for %d transitions.",
                    len(obs_triples))
        base_pred_triples = self._compute_base_pred_triples(
            obs_triples, fit_env)
        inferred_hint = self._infer_process_features_from_residuals(
            obs_triples, base_pred_triples)
        logger.info("Process features (data-driven hint): %s", inferred_hint)

        self._synthesize_with_agent(trajectories, obs_triples,
                                    base_pred_triples, inferred_hint)

        if self._process_rules is not None and self._fitted_params:
            rules, params = self._process_rules, self._fitted_params
            self._learned_simulator = LearnedSimulator(
                step_fn=lambda s, _r=rules, _p=params:  # type: ignore[misc]
                apply_rules(s, _r, _p),
                name="agent_synthesized")
        elif self._learned_simulator is None:
            logger.warning("Synthesis produced no simulator, skipping.")
            return

        combined_sim = self._build_combined_simulator(self._learned_simulator)
        self._option_model = self._build_option_model(combined_sim)
        logger.info("Built learned option model (SSE: %.6f).", self._fit_sse)

    def _build_option_model(
        self,
        simulator_fn: Callable[[State, Action], State],
    ) -> _OracleOptionModel:
        """Wrap a simulator function in an OracleOptionModel.

        Uses ``self._get_all_options()`` rather than
        ``get_gt_options(CFG.env)`` to avoid spawning a second cached
        PyBullet env via ``get_or_create_env``.
        """
        model = _OracleOptionModel(self._get_all_options(), simulator_fn)
        if CFG.wait_option_terminate_on_atom_change:
            model._abstract_function = (  # pylint: disable=protected-access
                lambda s: utils.abstract(s, self._get_all_predicates()))
        return model

    # ── Agent-based synthesis ────────────────────────────────────

    def _synthesize_with_agent(
        self,
        trajectories: List[LowLevelTrajectory],
        obs_triples: List[Tuple[State, Action, State]],
        base_pred_triples: List[Tuple[State, Action, State]],
        inferred_hint: Dict[str, List[str]],
    ) -> None:
        """Synthesize PROCESS_RULES, PARAM_SPECS, PROCESS_FEATURES via agent.

        ``inferred_hint`` is passed to the agent as a starting point and
        used as the eval/test scope until it declares its own
        ``PROCESS_FEATURES``. CFG flags
        ``agent_sim_learn_oracle_sim_program`` and
        ``agent_sim_learn_oracle_sim_params`` short-circuit the agent
        and/or MCMC by loading the GT simulator instead.
        """

        if CFG.agent_sim_learn_oracle_sim_program:
            rules, specs, process_features = get_gt_simulator(CFG.env)
            self._log_feature_set_diff(inferred_hint, process_features,
                                       "inferred", "oracle")
            if not CFG.agent_sim_learn_oracle_sim_params:
                rng = np.random.default_rng(CFG.seed)
                noise_scale = CFG.agent_sim_learn_oracle_sim_param_noise_scale
                if noise_scale < 0.0:
                    raise ValueError(
                        "agent_sim_learn_oracle_sim_param_noise_scale must "
                        "be non-negative.")
                perturbed = []
                for s in specs:
                    val = float(
                        np.clip(
                            s.init_value * (1.0 + rng.normal(0, noise_scale)),
                            s.lo, s.hi))
                    perturbed.append(ParamSpec(s.name, val, lo=s.lo, hi=s.hi))
                specs = perturbed
            logger.info("Loaded oracle sim program (%d rules, %d params).",
                        len(rules), len(specs))
        else:
            # Resolve sandbox_dir without depending on a live session
            # manager. LocalSandboxSessionManager does set this on
            # tool_context in __init__, but it isn't constructed until
            # _ensure_agent_session() runs further below.
            if CFG.agent_sdk_use_local_sandbox:
                sandbox_dir: Optional[str] = os.path.abspath(
                    os.path.join(self._get_log_dir(), "sandbox"))
            else:
                sandbox_dir = self._tool_context.sandbox_dir

            base = sandbox_dir or self._get_log_dir()
            simulator_file = os.path.join(base, "simulator.py")
            versions_dir = os.path.join(base, "simulator_versions")
            extra_paths = self._compute_extra_synthesis_paths(base)

            # Path the agent sees: cwd-relative for local-sandbox (the
            # validation hook resolves against cwd and rejects literal
            # ``/sandbox/...`` paths), docker mount point for docker,
            # absolute host path otherwise.
            if CFG.agent_sdk_use_local_sandbox:
                simulator_file_for_agent = "./simulator.py"
                sandbox_dir_for_agent: Optional[str] = "."
            elif sandbox_dir:
                simulator_file_for_agent = "/sandbox/simulator.py"
                sandbox_dir_for_agent = "/sandbox"
            else:
                simulator_file_for_agent = simulator_file
                sandbox_dir_for_agent = None

            exec_ns: Dict[str, Any] = {
                "trajectories":
                trajectories,
                "train_tasks":
                self._train_tasks,
                "is_goal_state":
                lambda state, task_idx: self._train_tasks[task_idx].goal_holds(
                    state),
                "np":
                np,
                "ParamSpec":
                ParamSpec,
            }

            # Build dynamic synthesis tools and attach them to the
            # tool context *before* opening the session. The attached
            # set is filtered against ``_get_synthesis_tool_names`` so
            # that method is the single source of truth for what the
            # agent sees — anything a builder constructs but the names
            # list omits is dropped here. The ``finally`` block below
            # clears the attachment.
            tools = create_synthesis_tools(
                exec_ns,
                base_pred_triples,
                inferred_hint,
                simulator_file=simulator_file,
                versions_dir=versions_dir,
                approach=self,
                sandbox_dir=base,
                sandbox_dir_for_agent=sandbox_dir_for_agent,
                cycle_index_provider=self._learning_cycle_index,
            )
            tools.extend(
                self._extra_synthesis_tools(exec_ns, base_pred_triples,
                                            inferred_hint, extra_paths))
            declared = set(self._get_synthesis_tool_names() or ())
            self._tool_context.extra_mcp_tools = [
                t for t in tools if getattr(t, "name", "") in declared
            ]
            self._learning_mode = True

            # PostToolUse hook: snapshot simulator.py / predicates.py on
            # every successful Write/Edit/MultiEdit, so the version
            # history covers everything the agent committed to file
            # (not just states that happened to coincide with an eval
            # call). Only active for this synthesis session.
            snapshot_targets = self._build_write_snapshot_targets(
                simulator_file, versions_dir, extra_paths)
            self._tool_context.extra_session_hooks = (
                self._build_synthesis_session_hooks(snapshot_targets, base))

            # Fresh session so the synthesis prompt + tools take effect.
            self._close_agent_session()
            self._ensure_agent_session()

            structs_ref = self._write_structs_reference()

            n_trajs = len(trajectories)
            n_demos = sum(1 for t in trajectories if t.is_demo)
            n_interaction = n_trajs - n_demos
            predicate_listing = self._format_predicate_signatures(
                self._get_all_predicates())
            trajectory_listing = self._format_trajectory_listing(trajectories)
            prior_state_block = self._format_prior_state_block(base)
            message = f"""\
Synthesize a process dynamics simulator for this environment. \
There are {n_trajs} trajectories ({len(obs_triples)} step \
transitions) available: {n_demos} oracle demonstration(s) (goal \
reached by construction) and {n_interaction} interaction \
trajectory/ies (collected during online learning; some may have \
failed to reach the goal).

{trajectory_listing}
Each trajectory carries a `train_task_idx`. You can query the \
ground-truth goal-check (a black-box binary reward) by calling \
`is_goal_state(state, task_idx)`. Equivalently \
`train_tasks[task_idx].goal_holds(state)`. Use this to (1) confirm \
which trajectories reached the goal and (2) treat failed \
interaction trajectories as counterexamples — places where your \
predicate or rule said "this should work" but the env disagreed.

{prior_state_block}Data-structure source code is at: {structs_ref}

A residual scan between the base simulator's prediction and the \
observed next state suggests these features carry process dynamics \
(starting hint, may include base-sim jitter — refine as you go):
{inferred_hint}

## Available Predicates (for subgoal annotations)
{predicate_listing}

Subgoal annotations in your plans for `evaluate_plan_refinement` \
must reference these predicate names with matching arity and types. \
Any threshold or condition you bake into a rule must be consistent \
with what the predicate's classifier actually checks, or refinement \
will reject parameter samples that look correct on paper.

Read the data-structures file first, then explore the trajectory \
data with `run_python` (variables: `trajectories`, `train_tasks`, \
`is_goal_state`, `np`, `ParamSpec`). Write your simulator to \
`{simulator_file_for_agent}` — define PROCESS_RULES, PARAM_SPECS, \
and PROCESS_FEATURES there. Every successful Write/Edit of \
`{simulator_file_for_agent}` is snapshotted to `simulator_versions/` as \
`cycle_XXX_vers_YYY_simulator.py` (deduped by content); the synthesis \
tools (evaluate_step_fit, report_residuals, evaluate_plan_refinement) \
load that file fresh on every call and report the version tag \
[cycle_XXX_vers_YYY] in their output. Iterate with `Edit` and re-run \
the tools."""

            extra_message = self._extra_synthesis_message(extra_paths)
            if extra_message:
                message = message + "\n\n" + extra_message

            try:
                self._query_agent_sync(message, kind="learn")
            finally:
                self._tool_context.extra_session_hooks = {}
                self._tool_context.extra_mcp_tools = []
                self._learning_mode = False
                self._close_agent_session()

            final_sim_tag = finalize_versioned_snapshot(
                simulator_file,
                versions_dir,
                cycle_idx=self._learning_cycle_index(),
                artifact_name="simulator",
            )
            if final_sim_tag is not None:
                self._current_simulator_version = final_sim_tag
                logger.info("Final simulator snapshot: %s", final_sim_tag)

            rules, specs, declared_features = (
                self._load_simulator_from_module_file(simulator_file,
                                                      trajectories))
            if rules is None or specs is None:
                return
            assert declared_features is not None, (
                "Agent did not declare PROCESS_FEATURES; "
                "synthesis output is incomplete.")
            process_features = declared_features
            self._log_feature_set_diff(inferred_hint, process_features,
                                       "inferred", "declared")
            logger.info("Agent synthesized %d rules, %d params.", len(rules),
                        len(specs))
            self._post_synthesis_loading(extra_paths, specs)

        self._process_rules = rules
        self._process_features = process_features

        _noise_sigma = 0.05  # matches fit_params default
        if CFG.agent_sim_learn_oracle_sim_params:
            self._fitted_params.clear()
            self._fitted_params.update({s.name: s.init_value for s in specs})
            oracle_sim_fn = lambda s, a, p: apply_rules(  # noqa: E731
                s, rules, p)
            self._fit_sse = compute_sse(oracle_sim_fn, base_pred_triples,
                                        self._fitted_params, process_features)
            fit_ll = -0.5 * self._fit_sse / (_noise_sigma**2)
            logger.info("Oracle params — SSE: %.6f  log-likelihood: %.2f",
                        self._fit_sse, fit_ll)
            for name, val in sorted(self._fitted_params.items()):
                logger.info("  %-30s  %.4f", name, val)
            log_sse_breakdown(oracle_sim_fn,
                              base_pred_triples,
                              self._fitted_params,
                              process_features,
                              label="oracle")
        else:
            new_params, self._fit_sse = self._fit_parameters(
                rules, specs, base_pred_triples, process_features)
            self._fitted_params.clear()
            self._fitted_params.update(new_params)
            if CFG.code_sim_learning_num_mcmc_steps == 0:
                logger.info("Skipped MCMC; using %d initial params.",
                            len(specs))
            else:
                logger.info("Fitted %d params.", len(specs))

    # ── Parameter fitting ────────────────────────────────────────

    @staticmethod
    def _fit_parameters(
        rules: List,
        specs: List[ParamSpec],
        base_pred_triples: List[Tuple[State, Action, State]],
        process_features: Dict[str, List[str]],
    ) -> Tuple[Dict[str, float], float]:
        """Fit parameters for the synthesized rules via MCMC.

        ``base_pred_triples`` must already have the base step applied;
        precomputing avoids re-running it inside the MCMC inner loop.
        """

        def sim_fn(state: State, _action: Action, params: Dict[str,
                                                               float]) -> Dict:
            return apply_rules(state, rules, params)

        noise_sigma = 0.05  # matches fit_params default
        init_params = {s.name: s.init_value for s in specs}
        pre_sse = compute_sse(sim_fn, base_pred_triples, init_params,
                              process_features)
        pre_ll = -0.5 * pre_sse / (noise_sigma**2)
        logger.info("Before fitting — SSE: %.6f  log-likelihood: %.2f",
                    pre_sse, pre_ll)
        log_sse_breakdown(sim_fn,
                          base_pred_triples,
                          init_params,
                          process_features,
                          label="before")

        result = fit_params(
            simulator_fn=sim_fn,
            transitions=base_pred_triples,
            param_specs=specs,
            process_features=process_features,
        )

        fitted_params = result.point_estimate
        post_sse = compute_sse(sim_fn, base_pred_triples, fitted_params,
                               process_features)
        post_ll = -0.5 * post_sse / (noise_sigma**2)
        logger.info("After fitting  — SSE: %.6f  log-likelihood: %.2f",
                    post_sse, post_ll)
        log_sse_breakdown(sim_fn,
                          base_pred_triples,
                          fitted_params,
                          process_features,
                          label="after")

        for name in sorted(fitted_params):
            init_val = init_params[name]
            fit_val = fitted_params[name]
            delta = fit_val - init_val
            pct = (delta / init_val * 100) if init_val != 0 else float("nan")
            logger.info("  %-30s  %.4f -> %.4f  (Δ=%.4f, %+.1f%%)", name,
                        init_val, fit_val, delta, pct)

        return fitted_params, post_sse

    # ── Process-feature inference ────────────────────────────────

    @staticmethod
    def _compute_base_pred_triples(
        obs_triples: List[Tuple[State, Action, State]],
        base_env: Any,
    ) -> List[Tuple[State, Action, State]]:
        """Replace each ``s_t`` with the base sim's one-step prediction."""
        return [(base_env.simulate(s, a), a, s_next)
                for s, a, s_next in obs_triples]

    @staticmethod
    def _infer_process_features_from_residuals(
        obs_triples: List[Tuple[State, Action, State]],
        base_pred_triples: List[Tuple[State, Action, State]],
        abs_tol: float = 1e-4,
        rel_tol: float = 1e-3,
        min_hits: int = 3,
    ) -> Dict[str, List[str]]:
        """Features whose base-sim prediction diverges from observation.

        Flags ``(type, feat)`` if ``|pred - obs| > rel_tol*|obs| + abs_tol``
        on at least ``min_hits`` triples. The ``min_hits`` floor keeps
        one-off PyBullet jitter from leaking base-handled features into the set.
        """
        del obs_triples  # objects are identical across both triple lists
        pairs = [(s_base, s_obs) for s_base, _, s_obs in base_pred_triples]
        hits: Dict[Tuple[str, str], int] = {}
        for _, _, tn, feat, pred, obs in iter_feature_residuals(pairs):
            if abs(pred - obs) > rel_tol * abs(obs) + abs_tol:
                hits[(tn, feat)] = hits.get((tn, feat), 0) + 1
        out: Dict[str, List[str]] = {}
        for (t, f), n in hits.items():
            if n >= min_hits:
                out.setdefault(t, []).append(f)
        return {t: sorted(fs) for t, fs in out.items()}

    @staticmethod
    def _log_feature_set_diff(
        a: Dict[str, List[str]],
        b: Dict[str, List[str]],
        a_label: str,
        b_label: str,
    ) -> None:
        """Log set-difference between two {type: [feats]} maps."""
        a_pairs = {(t, f) for t, fs in a.items() for f in fs}
        b_pairs = {(t, f) for t, fs in b.items() for f in fs}
        only_a = sorted(a_pairs - b_pairs)
        only_b = sorted(b_pairs - a_pairs)
        common = a_pairs & b_pairs
        logger.info(
            "Feature-set diff: %s vs %s (%d common, %d only-%s, %d only-%s)",
            a_label, b_label, len(common), len(only_a), a_label, len(only_b),
            b_label)
        if only_a:
            logger.info("  only in %s: %s", a_label, only_a)
        if only_b:
            logger.info("  only in %s: %s", b_label, only_b)

    @staticmethod
    def _format_predicate_signatures(predicates: Set[Predicate]) -> str:
        """Pretty-print predicates as ``Name(type1, type2)`` lines.

        Mirrors the ``## Available Predicates`` block in
        ``bilevel_sketch.build_solve_prompt``.
        """
        lines = []
        for pred in sorted(predicates, key=lambda p: p.name):
            type_sig = ", ".join(t.name for t in pred.types)
            lines.append(f"  {pred.name}({type_sig})")
        return "\n".join(lines)

    @staticmethod
    def _format_trajectory_listing(
            trajectories: List[LowLevelTrajectory]) -> str:
        """Render a per-trajectory listing with provenance tags.

        Each interaction trajectory shows the simulator / predicates
        snapshot used to generate the plan that collected it (if
        tracked). Demo trajectories list as ``demo``. Listed in the same
        order the agent sees them via the ``trajectories`` var.
        """
        if not trajectories:
            return ""
        lines = ["Trajectory roster (matches the `trajectories` list):"]
        for idx, traj in enumerate(trajectories):
            kind = "demo" if traj.is_demo else "interaction"
            try:
                task_str = f"task {traj.train_task_idx}"
            except AssertionError:
                task_str = "task ?"
            provenance: List[str] = []
            sim_v = traj.source_simulator_version
            preds_v = traj.source_predicates_version
            if sim_v:
                provenance.append(f"sim {sim_v}")
            if preds_v:
                provenance.append(f"predicates {preds_v}")
            tail = (f" — generated using {', '.join(provenance)}"
                    if provenance else "")
            lines.append(f"  [{idx}] {kind}, {task_str}{tail}")
        return "\n".join(lines) + "\n"

    def _format_prior_state_block(self, base: str) -> str:
        """Tell the agent about any simulator/predicates left over from a
        previous learning cycle.

        Returns a paragraph the agent can act on (read the files first
        and treat this cycle as incremental refinement) or an empty
        string if no prior state exists. The base sandbox dir is scanned
        for ``simulator.py`` / ``predicates.py``.
        """
        prior: List[str] = []
        sim_path = os.path.join(base, "simulator.py")
        preds_path = os.path.join(base, "predicates.py")
        if os.path.isfile(sim_path):
            prior.append("`./simulator.py`")
        if os.path.isfile(preds_path):
            prior.append("`./predicates.py`")
        if not prior:
            return ""
        joined = " and ".join(prior)
        return f"""\
Prior cycle state: {joined} already exist in the sandbox from a previous \
learning cycle. Read them first — they are the previous cycle's committed \
result and a reasonable starting point for incremental refinement (though \
a fresh rewrite is fine if the prior approach looks fundamentally wrong). \
Earlier versions are in `./simulator_versions/` and \
`./predicates_versions/` (named `cycle_XXX_vers_YYY_*.py`); \
cross-reference the trajectory roster's provenance tags against those \
files to see exactly which rules and predicates produced each failed plan.

"""

    @staticmethod
    def _load_simulator_from_module_file(
        path: str,
        trajectories: Optional[List[LowLevelTrajectory]] = None,
    ) -> Tuple[Optional[List], Optional[List[ParamSpec]], Optional[Dict[
            str, List[str]]]]:
        """Load PROCESS_RULES, PARAM_SPECS, PROCESS_FEATURES from one file.

        Execs ``path`` once in a fresh namespace. Returns ``(None, None,
        None)`` on missing file, exec failure, or if either
        ``PROCESS_RULES`` or ``PARAM_SPECS`` is absent; ``features`` may
        be ``None`` independently, in which case the caller asserts
        (``PROCESS_FEATURES`` is required from the agent).
        """
        if not os.path.isfile(path):
            logger.warning("No simulator file at %s.", path)
            return None, None, None

        ns: Dict[str, Any] = {
            "np": np,
            "ParamSpec": ParamSpec,
            "trajectories": trajectories or [],
        }
        with open(path, "r", encoding="utf-8") as f:
            code = f.read()
        try:
            exec(code, ns)  # pylint: disable=exec-used
        except Exception:  # pylint: disable=broad-except
            logger.warning("Failed to exec %s.", path, exc_info=True)
            return None, None, None

        rules, specs, features = read_simulator_components(ns)
        if rules is None:
            logger.warning("Simulator file %s missing PROCESS_RULES.", path)
            return None, None, None
        if specs is None:
            logger.warning("Simulator file %s missing PARAM_SPECS.", path)
            return None, None, None

        logger.info("Loaded %d rules, %d param specs from %s.", len(rules),
                    len(specs), path)
        return rules, specs, features

    # ── Static helpers ───────────────────────────────────────────

    def _write_structs_reference(self) -> str:
        """Write key struct sources to the sandbox; return the agent-visible
        path."""
        # pylint: disable=import-outside-toplevel,reimported
        from predicators.structs import Action as _Action
        from predicators.structs import LowLevelTrajectory as _LLT
        from predicators.structs import Object as _Object
        from predicators.structs import State as _State
        from predicators.structs import Type as _Type

        source = "\n\n".join(
            inspect.getsource(cls)
            for cls in [_Type, _Object, _State, _Action, _LLT])

        base = self._tool_context.sandbox_dir or self._get_log_dir()
        ref_dir = os.path.join(base, "reference")
        os.makedirs(ref_dir, exist_ok=True)
        ref_path = os.path.join(ref_dir, "structs.py")
        with open(ref_path, "w", encoding="utf-8") as f:
            f.write(source)

        # Path the agent sees: relative to its cwd in local-sandbox mode
        # (the sandbox-validation hook resolves against cwd and rejects
        # any literal ``/sandbox/...`` path), the docker mount point in
        # docker mode, or the absolute host path otherwise.
        if CFG.agent_sdk_use_local_sandbox:
            return "./reference/structs.py"
        if self._tool_context.sandbox_dir:
            return "/sandbox/reference/structs.py"
        return ref_path

    @staticmethod
    def _extract_obs_triples(
        trajectories: List[LowLevelTrajectory],
    ) -> List[Tuple[State, Action, State]]:
        """Extract observed (s_t, action_t, s_{t+1}) triples."""
        triples: List[Tuple[State, Action, State]] = []
        for traj in trajectories:
            for i in range(len(traj.actions)):
                triples.append(
                    (traj.states[i], traj.actions[i], traj.states[i + 1]))
        return triples

    def _recreate_base_env(self) -> None:
        """Reconnect after a PyBullet physics-server crash."""
        try:
            client_id = self._base_env._physics_client_id  # type: ignore[attr-defined]  # pylint: disable=protected-access
            pybullet.disconnect(client_id)
        except Exception:  # pylint: disable=broad-except  # client may already be dead
            pass
        logging.warning(
            "PyBullet physics client crashed; recreating base env "
            "(use_gui=%s).", CFG.option_model_use_gui)
        self._base_env = create_new_env(CFG.env,
                                        do_cache=False,
                                        use_gui=CFG.option_model_use_gui,
                                        skip_process_dynamics=True)

    def _build_combined_simulator(
        self,
        learned_simulator: LearnedSimulator,
    ) -> Callable[[State, Action], State]:
        """Compose base env with learned step-level dynamics.

        Captures ``self`` so the closure can recreate ``_base_env`` and
        retry once on a PyBullet crash (common on macOS Metal + GUI).
        """

        def combined_simulate(state: State, action: Action) -> State:
            try:
                base_state = self._base_env.simulate(state, action)
            except pybullet.error as e:
                logging.warning(
                    "PyBullet error in combined_simulate (%s); "
                    "recreating base env and retrying.", e)
                self._recreate_base_env()
                base_state = self._base_env.simulate(state, action)
            updates = learned_simulator.predict_step(base_state)
            if not updates:
                return base_state
            return merge_updates(base_state, updates)

        return combined_simulate

    def _build_synthesis_system_prompt(self) -> str:
        """Build the system prompt for the synthesis agent."""
        base_prompt = """\
You are synthesizing a parameterized process-dynamics simulator for a \
robotic manipulation environment.

A separate PyBullet base sim handles robot movement, grasping, and rigid- \
body physics. Your simulator handles **process dynamics** — features \
that change due to physical or causal processes (gradual level changes, \
accumulation, propagation between contacting objects, sensor readouts \
that lag actuators, etc.) that the base sim doesn't model.

## What you produce

One file `simulator.py` (path given in the first message) defining three \
top-level names:

```python
PROCESS_RULES:    List[Callable]            # rule functions (see signature below)
PARAM_SPECS:      List[ParamSpec]           # learnable parameters
PROCESS_FEATURES: Dict[str, List[str]]      # {type_name: [feature_names]} your rules predict
```

`PROCESS_FEATURES` defines both the loss scope and the test-time overwrite \
scope: only the listed `(type, feature)` pairs are scored against \
observations, and only those are written on top of the base sim at test \
time. Be honest — listing features your rules don't actually update \
inflates the loss without giving MCMC anything to optimise.

### Rule signature

```python
def rule(state, updates, params):
    # state:   the current env State
    # updates: Dict[Object, Dict[str, float]] accumulated from prior rules
    # params:  Dict[str, float], one entry per ParamSpec
    #
    # Accumulate, don't replace:
    #     updates.setdefault(obj, {})[feat] = new_value
    # Return the same dict.
    ...
```

### Timing

Each rule fires once per step:

```
state[t] ──base_sim──▶ draft state[t+1] ──your rules──▶ final state[t+1]
                                               ^^^^^^^
                        (only PROCESS_FEATURES are overwritten)
```

Rules see `state[t]`. They cannot see actions, the base sim's draft, or \
`state[t+2]`. If a feature changes one step *after* its gating event \
(e.g. an action toggles a gating flag at `t`, but the feature it drives \
only starts changing at `t+1`), that's an inherent 1-step lag in the \
data — accept the single boundary residual or model the delay with an \
extra parameter rather than chasing it with ever-stricter conditions.

### Geometric gates

If a rule's firing condition depends on the relative position of two \
bodies, do **not** gate on the raw distance between their recorded \
poses. `obj.x, obj.y` is the recorded pose origin — usually a body's \
base or frame center — while the point that actually drives the \
physics (a contact surface, an outlet on the body's side, an \
end-effector tip, a container opening, a handle) is typically offset \
from it. That offset lives in the body's **local frame**, so it \
rotates with the body's `rot` feature; gating on raw origin distance \
silently bakes in one task's orientation and breaks on any task where \
the fixture is rotated differently.

**Default to a learned, rotation-aware anchor offset.** Express every \
two-body geometric gate as a distance to an *anchored* point — the \
fixture origin plus a local-frame offset rotated into the world frame \
by the fixture's `rot` — with the offset declared as learnable params:

```python
PARAM_SPECS = [
    # Functional point offset, in the fixture's LOCAL frame:
    ParamSpec("fixture_local_dx",       0.0,  lo=-0.3, hi=0.3),
    ParamSpec("fixture_local_dy",       0.0,  lo=-0.3, hi=0.3),
    ParamSpec("widget_at_fixture_dist", 0.10, lo=0.0,  hi=0.4),
]

# `fixture`, `widget`: the relevant object pair (bind as your rule needs).
def process_rule(state, updates, params):
    rot = state.get(fixture, "rot")
    cos_r, sin_r = np.cos(rot), np.sin(rot)
    rot_mat = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
    local_offset = np.array([params["fixture_local_dx"],
                             params["fixture_local_dy"]])
    origin = np.array([state.get(fixture, "x"), state.get(fixture, "y")])
    anchor = origin + rot_mat @ local_offset  # world-frame point
    widget_xy = np.array([state.get(widget, "x"), state.get(widget, "y")])
    if np.linalg.norm(widget_xy - anchor) < params["widget_at_fixture_dist"]:
        ...  # fire
```

If the functional point really does coincide with the recorded origin, \
the fit drives the offsets to ~0 — no harm done. A threshold-only gate \
(no offset) is the exception: use one only after you have positively \
confirmed the recorded origin *is* the functional point. Share the \
offset and distance params with the gating predicate so the rule and \
predicate anchor to the same point.

**Required check before committing a geometric gate.** Bucket the \
trajectory steps by whether the gated effect actually fired, compute \
your gate quantity at each step, and confirm the two buckets separate \
by a clear margin. If they overlap, or separate only by a knife-edge \
gap (~5% of the value range or narrower), the gate references the \
wrong point — a threshold flush against the data boundary is a \
rejected fit, not a fit. Do **not** nudge the threshold to paper over \
it: add or refit the anchor offset and re-bucket. To find the offset, \
call `visualize_state` on a representative state from each bucket and \
use `annotate_scene` to overlay, on one render, the recorded origin \
and the positions where the effect did vs. did not fire; the gap \
between the origin and the effect-firing cluster is the offset.

### ParamSpec

```python
ParamSpec(name: str, init_value: float,
          lo: Optional[float] = None, hi: Optional[float] = None)
```

Bounds shape both the MCMC prior and the warm-start clamp. Set `lo=0.0` \
for non-negative rates, etc.

### Pre-injected when `simulator.py` is exec'd

`numpy as np`, `ParamSpec`. Import anything else at the top of the file. \
The data classes (`State`, `Object`, `Action`, ...) come from \
`predicators.structs`; source is in the reference file linked in the \
first message.

## Tools

`Write` / `Edit` `simulator.py` is your normal coding loop. Every \
successful write is snapshotted to \
`simulator_versions/cycle_XXX_vers_YYY_simulator.py` (deduped by \
content; ``XXX`` is the current cycle, ``YYY`` resets per cycle). The \
synthesis tools below load the file fresh on every call and prefix \
their output with `[cycle_XXX_vers_YYY]` so you and reviewers can diff \
iterations.

- `run_python(code)` — ad-hoc data exploration. `trajectories`, `np`, \
`ParamSpec` in scope. **Does not** define rules.
- `evaluate_step_fit` — per-step prediction accuracy: SSE on the step \
transitions at `init_value` params, plus post-fit SSE and fitted \
parameters from a parameter fit. Cheap; the inner-loop signal.
- `report_residuals` — per-feature breakdown: mismatch counts, mean / \
max abs error, vs-baseline improvement (negative ⇒ rules are adding \
error), worst-N example transitions. Diagnostic for *which* rule to fix.
- `evaluate_plan_refinement(plan, task_idx)` — per-task planning \
success: MCMC-fits, builds the combined simulator, runs backtracking \
refinement against a plan **you propose** (see "Plan format" below). \
Reports success or the step that got stuck. Slow; the gate before \
declaring done.

`evaluate_step_fit` and `evaluate_plan_refinement` test complementary \
things — pointwise accuracy vs. goal reachability. A rule can have \
ε-small SSE and still get a saturation threshold or alignment cap *just* \
wrong enough that refinement can't satisfy a subgoal. Use step-fit + \
residuals as the fast inner loop and plan-refinement as the slow \
goal-relevant gate.
__SYNTHESIS_PROMPT_EXTRA__
## Plan format for `evaluate_plan_refinement`

One option call per line, **with every option argument supplied and using \
typed object references** (`obj:type`), matching exactly what the inspect \
tools report. Use the inspect tools (or `run_python` over a trajectory) to \
read off the right names and arities — the parser is strict and silently \
omitting an argument will not be auto-filled. Example:

```
PickWidget(robot:robot, widget0:widget)
Place(robot:robot) -> {WidgetAtFixture(widget0:widget, fixture0:fixture)}
ActivateFixture(robot:robot, fixture0:fixture)
Wait(robot:robot) -> {WidgetReady(widget0:widget)}
...
```

(The names above are illustrative — use whatever options, types, and \
predicates the inspect tools actually report for your task.) Insert a \
`Wait` after any action that triggers a delayed process (gradual \
accumulation, propagation, sensor catch-up) so your rules have steps to \
fire on.

**Subgoal annotations** (`-> {Atom(obj:type, ...)}` after a step) are \
optional in general but **effectively required after open-ended skills \
like `Place`**. Without one the backtracking search has no preference for \
*where* to put the object, so a `Place; Wait` pair will refine cleanly \
but skip past the relevant target location and your rules never fire — \
the run looks like a rule bug but is actually a missing subgoal. For \
`Wait`, the annotation also specifies when the wait should terminate; \
prefix an atom with `NOT` if it should become false.

## Workflow

1. Explore data with `run_python` — what features change per step, \
which ones aren't explained by the base sim.
2. `Write` `simulator.py`; `Edit` to iterate.
3. Score with `evaluate_step_fit`, then `report_residuals` to find \
diverging features. Negative `vs base` ⇒ a rule is actively hurting — \
usually a wrong gate or sign.
4. When SSE is plausible, propose an option-skeleton plan and call \
`evaluate_plan_refinement(plan="...", task_idx=i)`. A stuck step means \
the rules gating its subgoal atoms are too tight or too loose; fix and \
re-validate.
"""
        extra = self._extra_synthesis_system_prompt()
        if extra:
            return base_prompt.replace("__SYNTHESIS_PROMPT_EXTRA__",
                                       "\n" + extra.rstrip() + "\n")
        return base_prompt.replace("__SYNTHESIS_PROMPT_EXTRA__", "")
