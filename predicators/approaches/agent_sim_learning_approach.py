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
from predicators.code_sim_learning.training import ParamSpec, compute_mse, \
    fit_params
from predicators.code_sim_learning.utils import LearnedSimulator, \
    apply_rules, merge_updates
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
        # Build the base env BEFORE super().__init__ and pass
        # the resulting option model in via option_model=. This stops
        # AgentPlannerApproach.__init__ from spinning up its own full-
        # process env (which would conflict with this one over PyBullet
        # GUI connections) and is the only env this approach holds.
        # learn_from_interaction_results later wraps a kin+learned
        # combined simulator around the same env.
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
        self._simulator: Optional[LearnedSimulator] = None
        self._process_features: Dict[str, List[str]] = {
            t.name: list(t.feature_names)
            for t in types if t.feature_names
        }
        # Persistent state across learning cycles.
        self._process_rules: Optional[List] = None
        self._fitted_params: Optional[Dict[str, float]] = None
        self._fit_mse: float = float("inf")
        # True during simulator synthesis (learning); False during
        # plan generation (decision-making).
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
        """Synthesize rules, fit parameters, and build the option model.

        Shared by ``learn_from_offline_dataset`` and
        ``learn_from_interaction_results``.
        """
        self._synthesize_with_agent(self._process_features, trajectories)

        # Build learned simulator.
        if self._process_rules is not None and self._fitted_params is not None:
            rules, params = self._process_rules, self._fitted_params
            self._simulator = LearnedSimulator(
                step_fn=lambda s, _r=rules, _p=params:  # type: ignore[misc]
                apply_rules(s, _r, _p),
                name="agent_synthesized")
        elif self._simulator is None:
            logger.warning("Synthesis produced no simulator, skipping.")
            return

        # Build combined simulator.
        combined_sim = self._build_combined_simulator(self._simulator,
                                                      self._process_features)

        # Build learned option model
        self._option_model = self._build_option_model(combined_sim)
        logger.info("Built learned option model (MSE: %.6f).", self._fit_mse)

    def _build_option_model(
        self,
        simulator_fn: Callable[[State, Action], State],
    ) -> _OracleOptionModel:
        """Wrap a simulator function in an OracleOptionModel.

        Plumbs ``_abstract_function`` for Wait-target atom-change
        termination so the model behaves identically whether it's
        wrapping the bare base simulator (init) or the learned
        kin+process combined simulator (post learn_from_interaction).
        Uses ``self._get_all_options()`` rather than
        ``get_gt_options(CFG.env)`` to avoid spawning a second cached
        PyBullet env via ``get_or_create_env``.
        """
        model = _OracleOptionModel(self._get_all_options(), simulator_fn)
        if CFG.wait_option_terminate_on_atom_change:
            preds = self._get_all_predicates()
            model._abstract_function = (  # pylint: disable=protected-access
                lambda s, _p=preds: utils.abstract(s, _p))
        return model

    # ── Agent-based synthesis ────────────────────────────────────

    def _synthesize_with_agent(
        self,
        process_features: Dict[str, List[str]],
        trajectories: List[LowLevelTrajectory],
    ) -> None:
        """Synthesize parameterized process rules via a Claude agent.

        Provides ``run_python``, ``evaluate_simulator``, and
        ``test_simulator`` tools.  The agent explores trajectory data
        via ``run_python`` (which has a persistent namespace with
        ``trajectories`` pre-loaded), then defines ``PROCESS_RULES``
        and ``PARAM_SPECS``.  Each ``run_python`` call appends code
        to a saved file; after the session we reload from that file.

        - ``agent_sim_learn_oracle_sim_program``: skip agent synthesis
          and load GT rules/specs instead (init_values perturbed so
          MCMC has non-trivial work).
        - ``agent_sim_learn_oracle_sim_param_noise_scale``: adjust the
          magnitude of the perturbation applied to oracle init_values.
        - ``agent_sim_learn_oracle_sim_params``: skip MCMC fitting and
          use the GT parameter values directly.
        """
        step_transitions = self._extract_step_transitions(trajectories)

        # ── Obtain rules + specs ────────────────────────────────
        if CFG.agent_sim_learn_oracle_sim_program:
            rules, specs = get_gt_simulator(CFG.env)
            if not CFG.agent_sim_learn_oracle_sim_params:
                rng = np.random.default_rng(CFG.seed)
                noise_scale = CFG.agent_sim_learn_oracle_sim_param_noise_scale
                if noise_scale < 0.0:
                    raise ValueError(
                        "agent_sim_learn_oracle_sim_param_noise_scale must "
                        "be non-negative.")
                perturbed = []
                for s in specs:
                    val = s.init_value * (
                        1.0 + float(rng.normal(0, noise_scale)))
                    if s.lo is not None:
                        val = max(s.lo, val)
                    if s.hi is not None:
                        val = min(s.hi, val)
                    perturbed.append(
                        ParamSpec(s.name, val, lo=s.lo, hi=s.hi))
                specs = perturbed
            logger.info("Loaded oracle sim program (%d rules, %d params).",
                        len(rules), len(specs))
        else:
            # Directory for saving simulator source code.
            base = self._tool_context.sandbox_dir or self._get_log_dir()
            save_dir = os.path.join(base, "simulator_code")

            # Persistent exec namespace — the agent's "scratch-pad".
            exec_ns: Dict[str, Any] = {
                "trajectories": trajectories,
                "np": np,
                "ParamSpec": ParamSpec,
            }

            # Build synthesis tools (run_python, evaluate, test).
            tools = create_synthesis_tools(exec_ns,
                                           step_transitions,
                                           process_features,
                                           self._base_env,
                                           save_dir=save_dir)
            self._tool_context.extra_mcp_tools = tools
            self._learning_mode = True

            # Force a fresh session so the synthesis system prompt and
            # tool set take effect.
            self._close_agent_session()
            self._ensure_agent_session()

            # Write data-structure reference for the agent to Read.
            structs_ref = self._write_structs_reference()

            n_trajs = len(trajectories)
            message = f"""\
Synthesize a process dynamics simulator for this environment. \
There are {n_trajs} trajectories ({len(step_transitions)} step \
transitions) available.

Data-structure source code is at: {structs_ref}
Read that file first, then explore the trajectory data with \
`run_python` and define PROCESS_RULES and PARAM_SPECS."""

            try:
                self._query_agent_sync(message)
            finally:
                self._tool_context.extra_mcp_tools = []
                self._learning_mode = False
                self._close_agent_session()

            # Load results from saved versioned files.
            rules, specs = self._load_simulator_from_file(
                save_dir, trajectories)
            if rules is None or specs is None:
                return

            logger.info("Agent synthesized %d rules, %d params.", len(rules),
                        len(specs))

        self._process_rules = rules

        # ── Obtain fitted parameters ────────────────────────────
        # Use a headless env for fitting.
        fit_env = create_new_env(CFG.env,
                                 do_cache=False,
                                 use_gui=False,
                                 skip_process_dynamics=True)
        _noise_sigma = 0.05  # matches fit_params default
        if CFG.agent_sim_learn_oracle_sim_params:
            self._fitted_params = {s.name: s.init_value for s in specs}
            self._fit_mse = compute_mse(
                lambda s, a, p: apply_rules(  # type: ignore[misc]
                    fit_env.simulate(s, a), rules, p),
                step_transitions,
                self._fitted_params,
                process_features)
            fit_ll = -0.5 * self._fit_mse / (_noise_sigma**2)
            logger.info("Oracle params — MSE: %.6f  log-likelihood: %.2f",
                        self._fit_mse, fit_ll)
            for name, val in sorted(self._fitted_params.items()):
                logger.info("  %-30s  %.4f", name, val)
        else:
            self._fitted_params, self._fit_mse = self._fit_parameters(
                rules, specs, step_transitions, process_features, fit_env)
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
        step_transitions: List[Tuple[State, Action, State]],
        process_features: Dict[str, List[str]],
        base_env: Any,
    ) -> Tuple[Dict[str, float], float]:
        """Fit parameters for the synthesized rules via MCMC.

        Args:
            base_env: Base environment. base_env.simulate(s, a) handles the
                first half of each transition, leaving only the learned
                process-rule updates for the MCMC loop to evaluate.

        Returns:
            (fitted_params, mse) tuple.
        """
        assert base_env is not None, "base_env required"
        # base_env.simulate(s, a) is param-independent, so pre-compute it
        # once here rather than inside every MCMC log-posterior call
        # (num_walkers × num_steps × len(transitions) invocations).
        # The MCMC loop then only evaluates the cheap apply_rules step.
        logger.info("Pre-computing base states for %d transitions.",
                    len(step_transitions))
        base_transitions: List[Tuple[State, Action, State]] = [
            (base_env.simulate(s, a), a, s_next)
            for s, a, s_next in step_transitions
        ]

        def sim_fn(state: State, action: Action,
                   params: Dict[str, float]) -> Dict:
            return apply_rules(state, rules, params)

        noise_sigma = 0.05  # matches fit_params default
        init_params = {s.name: s.init_value for s in specs}
        pre_mse = compute_mse(sim_fn, base_transitions, init_params,
                              process_features)
        pre_ll = -0.5 * pre_mse / (noise_sigma**2)
        logger.info("Before fitting — MSE: %.6f  log-likelihood: %.2f",
                    pre_mse, pre_ll)

        result = fit_params(
            simulator_fn=sim_fn,
            transitions=base_transitions,
            param_specs=specs,
            process_features=process_features,
        )

        fitted_params = result.point_estimate
        post_mse = compute_mse(sim_fn, base_transitions, fitted_params,
                               process_features)
        post_ll = -0.5 * post_mse / (noise_sigma**2)
        logger.info("After fitting  — MSE: %.6f  log-likelihood: %.2f",
                    post_mse, post_ll)

        for name in sorted(fitted_params):
            init_val = init_params[name]
            fit_val = fitted_params[name]
            delta = fit_val - init_val
            pct = (delta / init_val * 100) if init_val != 0 else float("nan")
            logger.info("  %-30s  %.4f -> %.4f  (Δ=%.4f, %+.1f%%)", name,
                        init_val, fit_val, delta, pct)

        return fitted_params, post_mse

    @staticmethod
    def _load_simulator_from_file(
        save_dir: str,
        trajectories: Optional[List[LowLevelTrajectory]] = None,
    ) -> Tuple[Optional[List], Optional[List[ParamSpec]]]:
        """Load PROCESS_RULES and PARAM_SPECS from versioned code files.

        Executes all ``NNN_run_python.py`` files in ``save_dir`` in
        order, accumulating into a single namespace.

        Returns (rules, specs), either of which may be None on failure.
        """
        if not os.path.isdir(save_dir):
            logger.warning("No simulator code dir at %s.", save_dir)
            return None, None

        files = sorted(f for f in os.listdir(save_dir)
                       if f.endswith(".py") and f[0].isdigit())
        if not files:
            logger.warning("No code files in %s.", save_dir)
            return None, None

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

        rules = ns.get("PROCESS_RULES")
        specs = ns.get("PARAM_SPECS")
        if not isinstance(rules, list) or not rules:
            logger.warning("Saved code did not define PROCESS_RULES.")
            return None, None
        if not isinstance(specs, list) or not specs:
            logger.warning("Saved code did not define PARAM_SPECS.")
            return None, None

        logger.info("Loaded %d rules, %d param specs from %d files in %s.",
                    len(rules), len(specs), len(files), save_dir)
        return rules, specs

    # ── Static helpers ───────────────────────────────────────────

    def _write_structs_reference(self) -> str:
        """Write extracted source of key structs to the sandbox.

        Returns the path the agent should Read.
        """
        # pylint: disable=import-outside-toplevel,reimported
        from predicators.structs import Action as _Action
        from predicators.structs import LowLevelTrajectory as _LLT
        from predicators.structs import Object as _Object
        from predicators.structs import State as _State
        from predicators.structs import Type as _Type

        source = "\n\n".join(
            inspect.getsource(cls)
            for cls in [_Type, _Object, _State, _Action, _LLT])

        # Write into sandbox reference dir if available, else log dir.
        base = self._tool_context.sandbox_dir or self._get_log_dir()
        ref_dir = os.path.join(base, "reference")
        os.makedirs(ref_dir, exist_ok=True)
        ref_path = os.path.join(ref_dir, "structs.py")
        with open(ref_path, "w", encoding="utf-8") as f:
            f.write(source)

        # In Docker sandbox the agent sees /sandbox/reference/structs.py.
        if self._tool_context.sandbox_dir:
            return "/sandbox/reference/structs.py"
        return ref_path

    @staticmethod
    def _extract_step_transitions(
        trajectories: List[LowLevelTrajectory],
    ) -> List[Tuple[State, Action, State]]:
        """Extract consecutive (s_t, action_t, s_{t+1}) triples."""
        triples: List[Tuple[State, Action, State]] = []
        for traj in trajectories:
            for i in range(len(traj.actions)):
                triples.append(
                    (traj.states[i], traj.actions[i], traj.states[i + 1]))
        return triples

    def _recreate_base_env(self) -> None:
        """Reconnect after a PyBullet physics-server crash.

        Disconnects the dead client (best-effort), then spins up a fresh
        env with the same settings so subsequent simulate() calls work.
        """
        try:
            pybullet.disconnect(self._base_env._physics_client_id)
        except Exception:  # client may already be dead
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
        simulator: LearnedSimulator,
        process_features: Dict[str, List[str]],
    ) -> Callable[[State, Action], State]:
        """Compose base env with learned step-level dynamics.

        Captures ``self`` so that if the PyBullet physics server crashes
        (common on macOS Metal with GUI mode after many simulation steps),
        the closure can recreate ``self._base_env`` and retry once.
        """

        def combined_simulate(state: State, action: Action) -> State:
            try:
                kin_state = self._base_env.simulate(state, action)
            except pybullet.error as e:
                logging.warning(
                    "PyBullet error in combined_simulate (%s); "
                    "recreating base env and retrying.", e)
                self._recreate_base_env()
                kin_state = self._base_env.simulate(state, action)
            updates = simulator.predict_step(kin_state)
            if not updates:
                return kin_state
            return merge_updates(kin_state, updates, process_features)

        return combined_simulate

    @staticmethod
    def _build_synthesis_system_prompt() -> str:
        """Build the system prompt for the synthesis agent."""
        return """\
You are synthesizing a parameterized process dynamics simulator for a \
robotic manipulation environment.

A separate physics engine (PyBullet) handles kinematics (robot movement, \
grasping, rigid body physics). Your simulator handles **process dynamics**: \
non-kinematic features that change due to ongoing physical or causal processes.

## Tools

- `run_python(code)` — execute Python in a persistent namespace. `print()` \
output is returned. The namespace persists across calls.
- `evaluate_simulator` — fit parameters using PROCESS_RULES and PARAM_SPECS \
from the namespace. Reports MSE.
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

Define two variables in the `run_python` namespace:

- `PROCESS_RULES`: list of rule functions
- `PARAM_SPECS`: list of ParamSpec objects

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
2. Identify which features change due to process dynamics (not kinematics)
3. Define `PROCESS_RULES` and `PARAM_SPECS` in the namespace via `run_python`
4. Call `evaluate_simulator` to fit parameters and check MSE
5. Call `test_simulator` to see prediction mismatches
6. Iterate if needed

## Tips

- Each trajectory is a sequence of states from one episode. Compare \
consecutive states to see per-step changes.
- Group objects by type: \
`groups = {}; for o in state: groups.setdefault(o.type.name, []).append(o)`
- Accumulate updates: `updates.setdefault(obj, {})[feat] = new_value`
"""
