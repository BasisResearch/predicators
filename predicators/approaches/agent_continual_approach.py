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
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, FrozenSet, List, Optional, Set, \
    Tuple

from predicators.agent_sdk import journal as journal_mod
from predicators.agent_sdk.fit_status import format_fit_status
from predicators.agent_sdk.tools.continual_tools import CONTINUAL_TOOL_NAMES
from predicators.agent_sdk.tools.sandbox_guard import \
    _screen_text_for_sandbox_escape
from predicators.agent_sdk.tools.synthesis import create_synthesis_tools
from predicators.approaches.agent_model_free_approach import \
    AgentModelFreeApproach
from predicators.approaches.agent_sim_learning_approach import \
    count_residual_hits, residual_hint_from_hits, \
    resolve_kept_predicate_names
from predicators.approaches.agent_sim_predicate_invention_approach import \
    AgentSimPredicateInventionApproach
from predicators.approaches.continual_play_mixin import ContinualPlayMixin
from predicators.code_sim_learning.rollout_env import dispose_env
from predicators.code_sim_learning.utils import LearnedSimulator, apply_rules
from predicators.envs import create_new_env
from predicators.option_model import _OptionModelBase
from predicators.settings import CFG
from predicators.structs import Action, LowLevelTrajectory, Predicate, State

if TYPE_CHECKING:  # pragma: no cover - the run package imports approaches
    from predicators.run.continual import ProtocolSession

# The env predicates an arm starts with unless the CFG allowlist hands
# it some: none.
NO_ENV_PREDICATES: FrozenSet[str] = frozenset()

# The agent's own helpers around the probe, loaded into the run_python
# namespace at every round start so they survive compaction and resume.
PROBE_EXTENSION_FILE = "probe_ext.py"

_Triple = Tuple[State, Action, State]


@dataclass
class _EpisodeTriples:
    """One recorded episode's transitions, their base-sim predictions and
    residual hits, extended as the episode grows.

    ``first_state`` names the episode: the recording keeps its state
    objects across syncs, so the first one identifies the episode and
    pins it against id reuse.
    """
    first_state: State
    obs: List[_Triple] = field(default_factory=list)
    base: List[_Triple] = field(default_factory=list)
    hits: Dict[Tuple[str, str], int] = field(default_factory=dict)


@dataclass
class _Workbench:
    """The model workbench's data over the run.

    The lists are what the ``run_python`` namespace, the synthesis
    toolkit and the probe hold by reference; every charged env call
    extends them in place (:meth:`AgentContinualApproach.
    _refresh_workbench`), so a fit, a rollout or the agent's own code
    sees the recording as it stands. The base env predicts the new
    transitions; it opens on first use and is released with the round.
    """
    trajectories: List[LowLevelTrajectory] = field(default_factory=list)
    obs_triples: List[_Triple] = field(default_factory=list)
    base_pred_triples: List[_Triple] = field(default_factory=list)
    inferred_hint: Dict[str, List[str]] = field(default_factory=dict)
    episodes: Dict[int, _EpisodeTriples] = field(default_factory=dict)
    env: Optional[Any] = None
    # The recording's shape (one action count per episode) at the last
    # refresh: a change invalidates the engine's memoized fits.
    fingerprint: Tuple[int, ...] = ()


# What one round stashes for its post-round finalize: the workbench, the
# synthesis paths, the extra artifact paths.
_RoundModel = Tuple[_Workbench, Any, Dict[str, str]]


class AgentContinualApproach(ContinualPlayMixin,
                             AgentSimPredicateInventionApproach):
    """C1's learner (hybrid simulator, parameter fit, predicate invention)
    playing under the continual protocol, modelling in the rounds it plays."""

    _save_suffix = "AgentContinual"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # The workbench's data lives for the run (its base-sim
        # predictions are computed once per transition); the round's
        # paths are set in _round_extra_tools and consumed in
        # _after_round.
        self._workbench = _Workbench()
        self._round_model: Optional[_RoundModel] = None
        self._fit_version_before: Optional[str] = None
        self._episodes_at_last_fit = 0
        # What loading ./probe_ext.py did at the round's start, for the
        # query's model line.
        self._probe_ext_status = ""
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
        # pylint: disable-next=import-outside-toplevel
        from predicators.agent_sdk.play_prompts import build_model_contract, \
            build_play_system_prompt

        # The contract of the model files (docs/continual-protocol.md,
        # 5): the rule signature follows CFG.partially_observable, the
        # system-identification menu is the base env's.
        contract = build_model_contract(
            partially_observable=CFG.partially_observable,
            physical_params_section=self._physical_params_prompt_section(),
            declared_params_only=CFG.agent_sim_learn_declared_params_only)
        return build_play_system_prompt(
            self._continual_tool_names(),
            base_sim_refs=self._base_sim_reference_paths(),
            model_contract=contract)

    def _model_status(self, session: ProtocolSession) -> str:
        n_eps, n_steps = self._episode_counts(session)
        data = f"Recorded episodes so far: {n_eps} ({n_steps} steps)."
        ext = f" {self._probe_ext_status}" if self._probe_ext_status else ""
        if self._current_simulator_version is None:
            return ("No model yet: `sim` is the base simulator, the visible "
                    "physics with none of the environment's hidden "
                    "mechanisms. Build `./simulator.py` and `./predicates.py` "
                    "in `run_python` and call `sim.fit()`. " + data + ext)
        new = max(0, n_eps - self._episodes_at_last_fit)
        refit = (f" {new} episode(s) recorded since your last fit; refit with "
                 "`sim.fit()` before you rely on the model." if new else "")
        return (
            f"Your model: `simulator.py` {self._current_simulator_version}"
            f", `predicates.py` {self._current_predicates_version or 'none'}"
            f". Last fit: {self._fit_status_text()}. {data}{refit}{ext}")

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
        bench = self._workbench
        self._refresh_workbench()
        trajectories = bench.trajectories
        base_pred_triples = bench.base_pred_triples
        inferred_hint = bench.inferred_hint
        paths = self._resolve_synthesis_paths()
        extra_paths = self._compute_extra_synthesis_paths(paths.base)
        self._round_model = (bench, paths, extra_paths)
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
        ctx.probe_validation_provider = toolkit.validation_runner
        ctx.probe_residuals_provider = toolkit.residuals_runner
        probe_ns = build_probe_namespace(ctx)
        exec_ns["sim"] = probe_ns["sim"]
        exec_ns["BeliefProbe"] = probe_ns["BeliefProbe"]
        self._load_probe_extension(exec_ns, paths.base)
        declared = set(self._get_synthesis_tool_names() or ())
        return [t for t in toolkit.tools if getattr(t, "name", "") in declared]

    def _load_probe_extension(self, exec_ns: Dict[str, Any],
                              sandbox_dir: str) -> None:
        """Run the agent's ``./probe_ext.py`` in the ``run_python`` namespace.

        The namespace is rebuilt every round and lost on a resume, so
        helpers the agent wraps around ``sim`` (sweeps, scoring loops,
        layout builders) vanished with the context unless it re-ran them
        by hand (domino m2 and m3, 2026-09-05, both rebuilt their
        helpers from the session log). The file's top-level definitions
        land next to ``sim``, ``trajectories`` and the rest, under the
        same sandbox screen as ``run_python`` code; what happened is
        reported in the query's model line (:meth:`_model_status`).
        """
        path = os.path.join(sandbox_dir, PROBE_EXTENSION_FILE)
        self._probe_ext_status = ""
        if not os.path.isfile(path):
            return
        with open(path, encoding="utf-8") as f:
            code = f.read()
        reason = _screen_text_for_sandbox_escape(code, sandbox_dir)
        if reason is not None:
            self._probe_ext_status = (
                f"`./{PROBE_EXTENSION_FILE}` was NOT loaded: the sandbox "
                f"guard blocked it ({reason}).")
            return
        before = set(exec_ns)
        prev_cwd = os.getcwd()
        try:
            os.chdir(sandbox_dir)
            exec(compile(code, f"./{PROBE_EXTENSION_FILE}", "exec"), exec_ns)  # pylint: disable=exec-used
        except Exception as e:  # pylint: disable=broad-except
            logging.warning("[Continual agent] %s failed to load: %s",
                            PROBE_EXTENSION_FILE, e)
            self._probe_ext_status = (
                f"`./{PROBE_EXTENSION_FILE}` failed to load "
                f"({type(e).__name__}: {e}); its definitions are missing "
                "from `run_python` until you fix it.")
            return
        finally:
            os.chdir(prev_cwd)
        names = sorted(n for n in set(exec_ns) - before
                       if not n.startswith("_"))
        listed = f" ({', '.join(names)})" if names else ""
        self._probe_ext_status = (
            f"`./{PROBE_EXTENSION_FILE}` loaded into `run_python`{listed}.")

    def _round_hooks(self, session: ProtocolSession) -> Dict[str, list]:
        del session
        if self._round_model is None:
            return {}
        _, paths, extra_paths = self._round_model
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
        bench, paths, extra_paths = self._round_model
        try:
            self._deploy_session_model(session, bench.trajectories,
                                       bench.base_pred_triples,
                                       bench.inferred_hint, paths, extra_paths)
        except Exception as e:  # pylint: disable=broad-except
            logging.exception("[Continual agent] deploying the session's "
                              "model failed")
            self._append_model_journal(paths, f"deploy failed: {e}")
        finally:
            self._clear_probe_providers()
            self._round_model = None
            self._release_workbench_env()

    def _refresh_arm_data(self, session: ProtocolSession) -> None:
        del session
        self._refresh_workbench()

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
        ctx.probe_validation_provider = None
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

    # -- The workbench's data -----------------------------------------------

    def _refresh_workbench(self) -> None:
        """Bring the workbench's lists up to the recording, in place: the
        trajectory list, the transitions with their base-sim predictions
        (computed for the new transitions only), the residual-feature hint.

        Called when a round opens and after every charged env call
        inside it, so a fit, a rollout or the agent's own code over
        ``trajectories`` sees the episode in progress. The per-episode
        cache is keyed by the episode's first state object, which the
        recording keeps; an episode that comes back as new objects (a
        level reloaded from its pickle after a resume) is predicted once
        more.
        """
        bench = self._workbench
        bench.trajectories[:] = self._get_all_trajectories()
        # The engine slices the flat triples back into per-episode
        # groups by these lengths (a latent block threads within an
        # episode, never across); the list object keeps them current.
        self._fit_trajectories = bench.trajectories
        fingerprint = tuple(len(t.actions) for t in bench.trajectories)
        if fingerprint != bench.fingerprint:
            # New data invalidates the memoized whole fits and
            # explainability verdicts, as the phased learn hook does
            # when its data arrives (the caches key trajectories by
            # segment lengths, so a grown episode must not answer from
            # its shorter self).
            bench.fingerprint = fingerprint
            self._explainability_cache.clear()
            self._sysid_fit_cache.clear()
        seen: Set[int] = set()
        obs_all: List[_Triple] = []
        base_all: List[_Triple] = []
        for traj in bench.trajectories:
            if not traj.actions:
                continue
            first = traj.states[0]
            key = id(first)
            ep = bench.episodes.get(key)
            if ep is None or ep.first_state is not first:
                ep = _EpisodeTriples(first)
                bench.episodes[key] = ep
            seen.add(key)
            done = len(ep.obs)
            if len(traj.actions) > done:
                new_obs = [(traj.states[i], traj.actions[i],
                            traj.states[i + 1])
                           for i in range(done, len(traj.actions))]
                new_base = self._base_predictions(new_obs)
                ep.obs.extend(new_obs)
                ep.base.extend(new_base)
                count_residual_hits(new_base, ep.hits)
            obs_all.extend(ep.obs)
            base_all.extend(ep.base)
        for key in [k for k in bench.episodes if k not in seen]:
            del bench.episodes[key]
        bench.obs_triples[:] = obs_all
        bench.base_pred_triples[:] = base_all
        hits: Dict[Tuple[str, str], int] = {}
        for ep in bench.episodes.values():
            for pair, n in ep.hits.items():
                hits[pair] = hits.get(pair, 0) + n
        hint = residual_hint_from_hits(hits)
        bench.inferred_hint.clear()
        bench.inferred_hint.update(hint)

    def _base_predictions(self, obs_triples: List[_Triple]) -> List[_Triple]:
        """The base sim's one-step prediction of each transition, on the
        workbench's own env (the visible physics with no hidden mechanism),
        opened on first use."""
        if not obs_triples:
            return []
        bench = self._workbench
        if bench.env is None:
            bench.env = create_new_env(CFG.env,
                                       do_cache=False,
                                       use_gui=False,
                                       skip_residual_dynamics=True)
        return self._compute_base_pred_triples(obs_triples, bench.env)

    def _release_workbench_env(self) -> None:
        bench = self._workbench
        if bench.env is not None:
            dispose_env(bench.env)
            bench.env = None

    def _fit_status_text(self) -> str:
        """The last fit as one line for the prompt: the point estimate per
        parameter and the posterior sample count, never the raw result (its
        Jacobian dump is noise to the agent)."""
        published = self._probe_fit_state()
        if published:
            return format_fit_status(published)
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
