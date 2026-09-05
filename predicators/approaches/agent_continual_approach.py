"""The LLM agent arms of the continual protocol (docs/continual-protocol.md).

Both arms mix :class:`ContinualPlayMixin` (the play loop) in front of
the phased approach class that holds their machinery:

* ``AgentContinualApproach`` (``agent_continual``) is C1's learner
  (hybrid simulator synthesis, parameter fit, predicate invention) on
  ``AgentSimPredicateInventionApproach``. There is no separate learning
  conversation: every round carries the model workbench, so the same
  session that acts in the environment also writes ``simulator.py`` and
  ``predicates.py``, fits them with ``sim.fit`` and validates plans with
  ``sim.run`` / ``sim.refine``. The ``sim`` probe reads the current
  files, so an edit is live on the next call; after the session the arm
  reloads the files, deploys the fit and installs the invented
  predicates.
* ``AgentContinualModelFreeApproach`` (``agent_continual_model_free``)
  is the model-free baseline on ``AgentModelFreeApproach``: the same env
  and skill tools, sandbox and journal, but no belief model, no ``sim``
  and no ``run_python``. What it knows comes from the recorded data and
  the real environment.

Neither arm starts with a predicate: the observation is the object
features and a render, the goal is its natural-language description,
and the model-based arm invents predicates as it learns. The allowlist
``agent_sim_learn_kept_predicates_names`` hands either arm env
predicates when an experiment wants that.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, FrozenSet, List, Optional, Set, \
    Tuple

from predicators.agent_sdk import journal as journal_mod
from predicators.agent_sdk.tools.continual_tools import CONTINUAL_TOOL_NAMES
from predicators.agent_sdk.tools.synthesis import create_synthesis_tools
from predicators.approaches.agent_model_free_approach import \
    AgentModelFreeApproach
from predicators.approaches.agent_sim_learning_approach import \
    resolve_kept_predicate_names
from predicators.approaches.agent_sim_predicate_invention_approach import \
    AgentSimPredicateInventionApproach
from predicators.approaches.continual_play_mixin import ContinualPlayMixin
from predicators.code_sim_learning.rollout_env import dispose_env
from predicators.code_sim_learning.utils import LearnedSimulator, apply_rules
from predicators.envs import create_new_env
from predicators.option_model import _OptionModelBase
from predicators.settings import CFG
from predicators.structs import LowLevelTrajectory, Predicate, State

if TYPE_CHECKING:  # pragma: no cover - the run package imports approaches
    from predicators.run.continual import ProtocolSession

# The env predicates an arm starts with unless the CFG allowlist hands
# it some: none.
NO_ENV_PREDICATES: FrozenSet[str] = frozenset()

# What one round stashes for its post-round finalize.
_RoundModel = Tuple[List[LowLevelTrajectory], List[Any], List[Any],
                    Dict[str, List[str]], Any, Dict[str, str]]


class AgentContinualApproach(ContinualPlayMixin,
                             AgentSimPredicateInventionApproach):
    """C1's learner (hybrid simulator, parameter fit, predicate invention)
    playing under the continual protocol, modelling in the rounds it plays."""

    _save_suffix = "AgentContinual"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Per-round workbench state, set in _round_extra_tools and
        # consumed in _after_round.
        self._round_model: Optional[_RoundModel] = None
        self._fit_version_before: Optional[str] = None
        self._episodes_at_last_fit = 0
        self._last_round_modelled = False

    @classmethod
    def get_name(cls) -> str:
        return "agent_continual"

    # -- The arm's declarations -------------------------------------------

    def _continual_tool_names(self) -> List[str]:
        return ["run_python"] + list(CONTINUAL_TOOL_NAMES)

    def _resolve_kept_names(self) -> Optional[FrozenSet[str]]:
        """No env predicate by default under the protocol; the CFG allowlist
        can hand the agent some."""
        return resolve_kept_predicate_names(NO_ENV_PREDICATES)

    def _learning_cycle_index(self) -> int:
        """Snapshot files are tagged by round number, so
        ``simulator_versions/`` reads chronologically across the run."""
        return self._rounds_played

    def _play_system_prompt(self) -> str:
        from predicators.agent_sdk.play_prompts import \
            build_play_system_prompt  # pylint: disable=import-outside-toplevel
        return build_play_system_prompt(
            self._continual_tool_names(),
            base_sim_refs=self._base_sim_reference_paths())

    def _model_status(self, session: ProtocolSession) -> str:
        n_eps, n_steps = self._episode_counts(session)
        data = f"Recorded episodes so far: {n_eps} ({n_steps} steps)."
        if self._current_simulator_version is None:
            return ("No model yet: `sim` is the base simulator, the visible "
                    "physics with none of the environment's hidden "
                    "mechanisms. Build `./simulator.py` and `./predicates.py` "
                    "in `run_python` and call `sim.fit()`. " + data)
        new = max(0, n_eps - self._episodes_at_last_fit)
        refit = (f" {new} episode(s) recorded since your last fit; refit with "
                 "`sim.fit()` before you rely on the model." if new else "")
        return (
            f"Your model: `simulator.py` {self._current_simulator_version}"
            f", `predicates.py` {self._current_predicates_version or 'none'}"
            f". Last fit: {self._fit_status_text()}. {data}{refit}")

    def _round_was_productive(self, session: ProtocolSession, state: Any,
                              steps_before: int, steps_after: int) -> bool:
        del session, state
        return steps_after > steps_before or self._last_round_modelled

    def _round_record_extra(self, state: Any) -> str:
        del state
        if not self._last_round_modelled:
            return ""
        return (f"Model updated: simulator.py "
                f"{self._current_simulator_version}, predicates.py "
                f"{self._current_predicates_version or 'none'}; fit "
                f"{self._fit_status_text()}.")

    # -- The model workbench, installed per round -----------------

    def _round_extra_tools(self, session: ProtocolSession) -> List[Any]:
        """Build the model workbench for this round: the synthesis
        ``run_python`` over the recorded data and the ``sim`` probe on the
        agent's own ``simulator.py`` / ``predicates.py``.

        The probe reads the files fresh each call (``sim.fit`` publishes
        the fit, ``sim.run`` / ``sim.refine`` roll the candidate
        forward), so the agent models and validates in the conversation
        it acts in. :meth:`_after_round` deploys what it wrote.
        """
        del session
        # pylint: disable-next=import-outside-toplevel
        from predicators.agent_sdk.belief_probe import _check_time_budget, \
            build_probe_namespace
        trajectories = self._get_all_trajectories()
        obs_triples, base_pred_triples, inferred_hint = \
            self._prepare_model_data(trajectories)
        paths = self._resolve_synthesis_paths()
        extra_paths = self._compute_extra_synthesis_paths(paths.base)
        self._round_model = (trajectories, obs_triples, base_pred_triples,
                             inferred_hint, paths, extra_paths)
        self._fit_version_before = self._probe_fit_state().get("version")

        exec_ns = self._build_synthesis_exec_ns(trajectories)
        ctx = self._tool_context
        ctx.learn_cycle_index = self._learning_cycle_index()
        toolkit = create_synthesis_tools(
            exec_ns,
            base_pred_triples,
            inferred_hint,
            simulator_file=paths.simulator_file,
            versions_dir=paths.versions_dir,
            approach=self,
            sandbox_dir=paths.base,
            sandbox_dir_for_agent=paths.sandbox_dir_for_agent,
            cycle_index_provider=self._learning_cycle_index,
            budget_check=lambda: _check_time_budget(self._tool_context),
        )
        self._install_extra_synthesis_surfaces(exec_ns, base_pred_triples,
                                               inferred_hint, extra_paths)
        ctx.probe_option_model_provider = \
            self._make_candidate_probe_model_provider(
                paths.simulator_file, trajectories, base_pred_triples,
                inferred_hint)
        ctx.probe_fit_provider = toolkit.fit_runner
        ctx.probe_residuals_provider = toolkit.residuals_runner
        probe_ns = build_probe_namespace(ctx)
        exec_ns["sim"] = probe_ns["sim"]
        exec_ns["BeliefProbe"] = probe_ns["BeliefProbe"]
        declared = set(self._get_synthesis_tool_names() or ())
        return [t for t in toolkit.tools if getattr(t, "name", "") in declared]

    def _round_hooks(self, session: ProtocolSession) -> Dict[str, list]:
        del session
        if self._round_model is None:
            return {}
        _, _, _, _, paths, extra_paths = self._round_model
        targets = self._build_write_snapshot_targets(paths.simulator_file,
                                                     paths.versions_dir,
                                                     extra_paths)
        return self._build_synthesis_session_hooks(targets, paths.base)

    def _after_round(self, session: ProtocolSession, state: Any) -> None:
        """Deploy whatever the round wrote: load the model files, fit and build
        the option model, install the invented predicates."""
        del state
        self._last_round_modelled = False
        if self._round_model is None:
            return
        trajectories, obs_triples, base_pred_triples, inferred_hint, paths, \
            extra_paths = self._round_model
        del obs_triples
        try:
            self._deploy_session_model(session, trajectories,
                                       base_pred_triples, inferred_hint, paths,
                                       extra_paths)
        except Exception as e:  # pylint: disable=broad-except
            logging.exception("[Continual agent] deploying the session's "
                              "model failed")
            self._append_model_journal(paths, f"deploy failed: {e}")
        finally:
            self._clear_probe_providers()
            self._round_model = None

    def _deploy_session_model(self, session: ProtocolSession,
                              trajectories: List[LowLevelTrajectory],
                              base_pred_triples: List[Any],
                              inferred_hint: Dict[str, List[str]], paths: Any,
                              extra_paths: Dict[str, str]) -> None:
        loaded = self._load_synthesis_artifacts(trajectories, inferred_hint,
                                                paths, extra_paths, {})
        if loaded is None:
            # No loadable simulator.py this session: the prior model, if
            # any, still stands.
            return
        rules, specs, residual_features = loaded
        self._residual_rules = rules
        self._residual_features = residual_features
        self._fit_params_after_synthesis(rules, specs, base_pred_triples,
                                         residual_features)
        if self._residual_rules is not None and self._fitted_params:
            _rules, _params = self._residual_rules, self._fitted_params

            def _step_fn(s: State, c: Any) -> Any:
                return apply_rules(s, _rules, _params, cmds=c)

            self._learned_simulator = LearnedSimulator(
                step_fn=_step_fn, name="agent_synthesized")
            combined = self._build_combined_simulator(self._learned_simulator)
            self._option_model = self._build_option_model(combined)
        self._last_round_modelled = True
        self._episodes_at_last_fit = len(self._online_trajectories)
        session.record_sandbox("fits", 1)
        # The invented predicates the runner abstracts with (Wait
        # targets, divergence checks) follow the model.
        self._sync_tool_context()
        session.abstract_predicates = set(self._get_all_predicates())
        self._append_model_journal(
            paths, f"simulator {self._current_simulator_version}, predicates "
            f"{self._current_predicates_version}, {self._fit_status_text()}")

    def _clear_probe_providers(self) -> None:
        ctx = self._tool_context
        ctx.probe_option_model_provider = None
        ctx.probe_fit_provider = None
        ctx.probe_residuals_provider = None
        ctx.probe_param_status = None
        ctx.probe_artifact_loaders.clear()
        ctx.learn_cycle_index = None
        ctx.extra_session_hooks = {}

    def _append_model_journal(self, paths: Any, outcome: str) -> None:
        journal_mod.append_entry(
            self._tool_context.sandbox_dir or self._get_log_dir(),
            f"Model after round {self._rounds_played + 1}",
            outcome,
            filename=journal_mod.ATTEMPTS_FILENAME)
        del paths

    def _prepare_model_data(
        self, trajectories: List[LowLevelTrajectory]
    ) -> Tuple[List[Any], List[Any], Dict[str, List[str]]]:
        """The recorded transitions, their base-sim predictions and the
        residual-feature hint, over every episode so far."""
        obs_triples = self._extract_obs_triples(trajectories)
        if not obs_triples:
            return [], [], {}
        fit_env = create_new_env(CFG.env,
                                 do_cache=False,
                                 use_gui=False,
                                 skip_residual_dynamics=True)
        try:
            base_pred_triples = self._compute_base_pred_triples(
                obs_triples, fit_env)
        finally:
            dispose_env(fit_env)
        inferred_hint = self._infer_residual_features_from_scan(
            obs_triples, base_pred_triples)
        return obs_triples, base_pred_triples, inferred_hint

    def _fit_status_text(self) -> str:
        """The last fit as one line for the prompt: the point estimate per
        parameter and the posterior sample count, never the raw result (its
        Jacobian dump is noise to the agent)."""
        result = getattr(self, "_last_fit_result", None)
        if result is None:
            return "no fit result"
        try:
            estimate = dict(result.point_estimate)
            n_samples = int(result.samples.shape[0])
        except (AttributeError, TypeError, ValueError, IndexError):
            return str(result)[:300]
        params = ", ".join(f"{k}={v:.4g}" for k, v in estimate.items())
        return (f"fitted {len(estimate)} parameter(s) from {n_samples} "
                f"posterior sample(s): {params}")[:600]

    # -- Parent-specific overrides ---------------------------------------

    def _get_agent_system_prompt(self) -> str:
        return self._play_system_prompt()

    def _extra_save_state(self) -> Dict[str, Any]:
        state = super()._extra_save_state()
        state.update(self._continual_save_state())
        state["episodes_at_last_fit"] = self._episodes_at_last_fit
        return state

    def _load_extra_save_state(self, save_dict: Dict[str, Any]) -> None:
        super()._load_extra_save_state(save_dict)
        self._load_continual_save_state(save_dict)
        self._episodes_at_last_fit = int(
            save_dict.get("episodes_at_last_fit", 0))


class AgentContinualModelFreeApproach(ContinualPlayMixin,
                                      AgentModelFreeApproach):
    """The model-free baseline of the continual protocol: the env and skill
    tools, the sandbox and the journal, and nothing else.

    No belief model, no ``sim``, no ``run_python`` and no learning
    session; the agent's own code in the sandbox reads the recorded
    data. Run with ``agent_planner_use_simulator`` off, as the phased
    ``agent_model_free`` arm is, so no simulator is built at all.
    """

    _save_suffix = "AgentContinualModelFree"

    @classmethod
    def get_name(cls) -> str:
        return "agent_continual_model_free"

    # -- The arm's declarations -------------------------------------------

    def _continual_tool_names(self) -> List[str]:
        return list(CONTINUAL_TOOL_NAMES)

    def _get_all_predicates(self) -> Set[Predicate]:
        """The arm's fixed vocabulary: the env predicates the allowlist keeps,
        none by default; it invents nothing."""
        names = resolve_kept_predicate_names(NO_ENV_PREDICATES)
        if names is None:
            return set(self._initial_predicates)
        return {p for p in self._initial_predicates if p.name in names}

    def _create_planner_option_model(self) -> Optional[_OptionModelBase]:
        """No simulator, whatever ``agent_planner_use_simulator`` says: the arm
        has no ``run_python`` and must never hold a model of the env."""
        return None

    # -- Parent-specific overrides ---------------------------------------

    def _get_agent_system_prompt(self) -> str:
        return self._play_system_prompt()

    def _extra_save_state(self) -> Dict[str, Any]:
        state = super()._extra_save_state()
        state.update(self._continual_save_state())
        return state

    def _load_extra_save_state(self, save_dict: Dict[str, Any]) -> None:
        super()._load_extra_save_state(save_dict)
        self._load_continual_save_state(save_dict)
