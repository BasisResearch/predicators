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
from predicators.agent_sdk.tools import create_synthesis_tools
from predicators.approaches.agent_bilevel_approach import AgentBilevelApproach
from predicators.code_sim_learning.training import ParamSpec, compute_sse, \
    fit_params, log_sse_breakdown
from predicators.code_sim_learning.utils import LearnedSimulator, \
    apply_rules, merge_updates, read_simulator_components
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
        self._fitted_params: Optional[Dict[str, float]] = None
        self._fit_sse: float = float("inf")
        self._learning_mode: bool = False

    @classmethod
    def get_name(cls) -> str:
        return "agent_sim_learning"

    # ── Agent session hooks ──────────────────────────────────────

    def _get_agent_system_prompt(self) -> str:
        if self._learning_mode:
            return self._build_synthesis_system_prompt()
        return super()._get_agent_system_prompt()

    # ── Learning ────────────────────────────────────────────────

    def learn_from_offline_dataset(self, dataset: Dataset) -> None:
        super().learn_from_offline_dataset(dataset)
        self._learn_simulator(dataset.trajectories)

    def learn_from_interaction_results(
            self, results: Sequence[InteractionResult]) -> None:
        super().learn_from_interaction_results(results)
        self._learn_simulator(self._online_trajectories)

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

        if self._process_rules is not None and self._fitted_params is not None:
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
                    val = s.init_value * (1.0 +
                                          float(rng.normal(0, noise_scale)))
                    if s.lo is not None:
                        val = max(s.lo, val)
                    if s.hi is not None:
                        val = min(s.hi, val)
                    perturbed.append(ParamSpec(s.name, val, lo=s.lo, hi=s.hi))
                specs = perturbed
            logger.info("Loaded oracle sim program (%d rules, %d params).",
                        len(rules), len(specs))
        else:
            base = self._tool_context.sandbox_dir or self._get_log_dir()
            save_dir = os.path.join(base, "simulator_code")

            exec_ns: Dict[str, Any] = {
                "trajectories": trajectories,
                "np": np,
                "ParamSpec": ParamSpec,
            }

            tools = create_synthesis_tools(exec_ns,
                                           base_pred_triples,
                                           inferred_hint,
                                           save_dir=save_dir)
            self._tool_context.extra_mcp_tools = tools
            self._learning_mode = True

            # Fresh session so the synthesis prompt + tools take effect.
            self._close_agent_session()
            self._ensure_agent_session()

            structs_ref = self._write_structs_reference()

            n_trajs = len(trajectories)
            message = f"""\
Synthesize a process dynamics simulator for this environment. \
There are {n_trajs} trajectories ({len(obs_triples)} step \
transitions) available.

Data-structure source code is at: {structs_ref}

A residual scan between the base simulator's prediction and the \
observed next state suggests these features carry process dynamics \
(starting hint, may include base-sim jitter — refine as you go):
{inferred_hint}

Read the data-structures file first, then explore the trajectory \
data with `run_python` and define PROCESS_RULES, PARAM_SPECS, and \
PROCESS_FEATURES."""

            try:
                self._query_agent_sync(message)
            finally:
                self._tool_context.extra_mcp_tools = []
                self._learning_mode = False
                self._close_agent_session()

            rules, specs, declared = self._load_simulator_from_file(
                save_dir, trajectories)
            if rules is None or specs is None:
                return
            assert declared is not None, (
                "Agent did not declare PROCESS_FEATURES; "
                "synthesis output is incomplete.")
            process_features = declared
            self._log_feature_set_diff(inferred_hint, process_features,
                                       "inferred", "declared")
            logger.info("Agent synthesized %d rules, %d params.", len(rules),
                        len(specs))

        self._process_rules = rules
        self._process_features = process_features

        _noise_sigma = 0.05  # matches fit_params default
        if CFG.agent_sim_learn_oracle_sim_params:
            self._fitted_params = {s.name: s.init_value for s in specs}
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
            self._fitted_params, self._fit_sse = self._fit_parameters(
                rules, specs, base_pred_triples, process_features)
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
        hits: Dict[Tuple[str, str], int] = {}
        for (s_t, _, _), (s_base, _, s_obs) in zip(obs_triples,
                                                   base_pred_triples):
            for obj in s_t:
                for feat in obj.type.feature_names:
                    pred = float(s_base.get(obj, feat))
                    obs = float(s_obs.get(obj, feat))
                    if abs(pred - obs) > rel_tol * abs(obs) + abs_tol:
                        key = (obj.type.name, feat)
                        hits[key] = hits.get(key, 0) + 1
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
    def _load_simulator_from_file(
        save_dir: str,
        trajectories: Optional[List[LowLevelTrajectory]] = None,
    ) -> Tuple[Optional[List], Optional[List[ParamSpec]], Optional[Dict[
            str, List[str]]]]:
        """Load PROCESS_RULES, PARAM_SPECS, PROCESS_FEATURES from saved files.

        Execs all ``NNN_run_python.py`` files in ``save_dir`` in order
        into one namespace. Returns ``(None, None, None)`` if rules or
        specs are missing; ``features`` may be ``None`` independently,
        in which case the caller asserts (PROCESS_FEATURES is required
        from the agent).
        """
        if not os.path.isdir(save_dir):
            logger.warning("No simulator code dir at %s.", save_dir)
            return None, None, None

        files = sorted(f for f in os.listdir(save_dir)
                       if f.endswith(".py") and f[0].isdigit())
        if not files:
            logger.warning("No code files in %s.", save_dir)
            return None, None, None

        ns: Dict[str, Any] = {
            "np": np,
            "ParamSpec": ParamSpec,
            "trajectories": trajectories or [],
        }
        for fname in files:
            fpath = os.path.join(save_dir, fname)
            with open(fpath, "r", encoding="utf-8") as f:
                code = f.read()
            try:
                exec(code, ns)  # pylint: disable=exec-used
            except Exception:  # pylint: disable=broad-except
                logger.warning("Failed to exec %s, skipping.",
                               fpath,
                               exc_info=True)

        rules, specs, features = read_simulator_components(ns)
        if rules is None:
            logger.warning("Saved code did not define PROCESS_RULES.")
            return None, None, None
        if specs is None:
            logger.warning("Saved code did not define PARAM_SPECS.")
            return None, None, None

        logger.info("Loaded %d rules, %d param specs from %d files in %s.",
                    len(rules), len(specs), len(files), save_dir)
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

        # Agent sees the sandbox-mounted path, not the host path.
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

    @staticmethod
    def _build_synthesis_system_prompt() -> str:
        """Build the system prompt for the synthesis agent."""
        return """\
You are synthesizing a parameterized process dynamics simulator for a \
robotic manipulation environment.

A separate base physics engine (PyBullet) handles robot movement, grasping, \
and rigid body physics. Your simulator handles **process dynamics**: features \
that change due to ongoing physical or causal processes (e.g., water filling, \
heat transfer) that the base sim doesn't model.

## Tools

- `run_python(code)` — execute Python in a persistent namespace. `print()` \
output is returned. The namespace persists across calls.
- `evaluate_simulator` — fit parameters using PROCESS_RULES and PARAM_SPECS \
from the namespace. Reports SSE.
- `test_simulator` — test predictions vs observations on step transitions. \
Shows mismatches.

### Pre-loaded variables

- `trajectories`: List[LowLevelTrajectory] — the collected trajectory data
- `np`, `ParamSpec` — standard imports

### Data structures

The trajectory data uses classes from `predicators.structs` (Type, Object, \
State, Action, LowLevelTrajectory). Their source code is provided as a \
reference file — Read the path given in the first message.

## Goal

Define three variables in the `run_python` namespace:

- `PROCESS_RULES`: list of rule functions
- `PARAM_SPECS`: list of ParamSpec objects
- `PROCESS_FEATURES`: `Dict[str, List[str]]` — for each object type, \
the feature names your rules predict. This is treated as the truth: \
the loss only penalises mismatches on these features, and at test \
time the learned simulator only overwrites these features on top of \
the base sim's prediction. Be honest — listing features your rules \
don't actually update will inflate the loss without giving MCMC \
anything to optimise.

Parameters are fitted automatically after the session ends.

### Process rule signature

```python
def rule(state, updates, params):
    \"\"\"Apply one process for a single simulation step.

    Args:
        state: Current env state.
        updates: Dict[Object, Dict[str, value]] accumulated from prior rules.
        params: Dict[str, float] of learned parameters.

    Returns:
        The (possibly modified) updates dict.
    \"\"\"
```

### ParamSpec

```python
ParamSpec(name: str, init_value: float)
```

## Workflow

1. Explore the trajectory data with `run_python`: types, features, \
state changes over time
2. Identify which features change due to process dynamics (not the base sim)
3. Define `PROCESS_RULES` and `PARAM_SPECS` in the namespace via `run_python`
4. Call `evaluate_simulator` to fit parameters and check SSE
5. Call `test_simulator` to see prediction mismatches
6. Iterate if needed

## Tips

- Each trajectory is a sequence of states from one episode. Compare \
consecutive states to see per-step changes.
- Group objects by type: \
`groups = {}; for o in state: groups.setdefault(o.type.name, []).append(o)`
- Accumulate updates: `updates.setdefault(obj, {})[feat] = new_value`
"""
