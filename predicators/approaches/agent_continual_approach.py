"""The LLM agent arms of the continual protocol (docs/protocol/design.md).

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

import hashlib
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, FrozenSet, List, \
    Optional, Set, Tuple

from predicators.agent_sdk import journal as journal_mod
from predicators.agent_sdk.fit_status import format_fit_status
from predicators.agent_sdk.tools.continual_tools import CONTINUAL_TOOL_NAMES, \
    play_tool_names
from predicators.agent_sdk.tools.exploration import ProbeSurface
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
from predicators.code_sim_learning.scene_manifest import \
    build_scene_manifest, write_scene_manifest
from predicators.code_sim_learning.utils import LearnedSimulator, apply_rules
from predicators.envs import create_new_env
from predicators.observation_noise import ObservationNoise
from predicators.option_model import _OptionModelBase, _OracleOptionModel
from predicators.run.episode import EpisodeOver
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
        # (bodies, asset files) of the last scene manifest, for the prompt.
        self._scene_package_summary: Tuple[int, int] = (0, 0)
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
        return ["run_python"] + play_tool_names(CONTINUAL_TOOL_NAMES)

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
        from predicators.agent_sdk.play_prompts import build_play_system_prompt
        return build_play_system_prompt(
            self._continual_tool_names(),
            base_sim_refs=self._base_sim_reference_paths(),
            model_contract=self._play_model_contract(),
            **self._play_prompt_options())

    def _play_model_contract(self, **options: Any) -> str:
        """The contract of the model files (docs/protocol/design.md, 5).

        Model memory follows CFG.partially_observable, the system-
        identification menu is the base env's. ``options`` are the
        frozen arms' keyword arguments for build_model_contract.
        """
        # pylint: disable-next=import-outside-toplevel
        from predicators.agent_sdk.play_prompts import build_model_contract
        return build_model_contract(
            partially_observable=CFG.partially_observable,
            physical_params_section=self._physical_params_prompt_section(),
            declared_params_only=CFG.agent_sim_learn_declared_params_only,
            **options)

    def _play_prompt_options(self) -> Dict[str, Any]:
        """Extra keyword arguments for build_play_system_prompt.

        The frozen arms pass their arm statement here.
        """
        if CFG.continual_provide_scene_package:
            return {"scene_package": True}
        return {}

    def _probe_surface(self) -> ProbeSurface:
        """What this arm's ``sim`` probe accepts (``run_python``'s
        description)."""
        return ProbeSurface(fit=self._fit_available(),
                            uncertainty=CFG.continual_uncertainty_decisions)

    def _no_model_section(self) -> str:
        """The play_query section shown while no model file exists."""
        return "no_model"

    def _fit_available(self) -> bool:
        """Whether the harness fits parameters in this arm.

        Frozen and no-fitting arms return False, so the query never asks
        for a refit that the probe would refuse.
        """
        return True

    def _model_status(self, session: ProtocolSession) -> str:
        n_eps, n_steps = self._episode_counts(session)
        # pylint: disable-next=import-outside-toplevel
        from predicators.agent_sdk.prompt_templates import render
        if self._current_simulator_version is None:
            status = render("play_query",
                            self._no_model_section(),
                            n_episodes=str(n_eps),
                            n_steps=str(n_steps))
        else:
            new = max(0, n_eps - self._episodes_at_last_fit)
            refit_note = (" Refit with `sim.fit()` before you rely on the "
                          "model." if new and self._fit_available() else "")
            status = render("play_query",
                            "model_status",
                            simulator_version=self._current_simulator_version,
                            predicates_version=self._current_predicates_version
                            or "none",
                            fit_status=self._fit_status_text(),
                            n_episodes=str(n_eps),
                            n_steps=str(n_steps),
                            new_episodes=str(new),
                            refit_note=refit_note)
        return status + (f" {self._probe_ext_status}"
                         if self._probe_ext_status else "")

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
            probe_surface=self._probe_surface(),
        )
        self._install_extra_synthesis_surfaces(exec_ns, base_pred_triples,
                                               inferred_hint, extra_paths)
        candidate_provider = self._make_candidate_probe_model_provider(
            paths.simulator_file, trajectories, base_pred_triples,
            inferred_hint)

        def probe_model() -> _OptionModelBase:
            # Before a candidate exists the probe runs the real skill
            # controllers over the visible base physics (hidden
            # mechanisms disabled), as the prompt says: reach, grasp
            # and collision feasibility rehearse from the first round.
            # Bridge seed 2 of the Sept 17, 2026 Sonnet pilots never
            # wrote a model and never rehearsed a pick; a `sim` that
            # raised without a file gave it no reason to.
            if os.path.isfile(paths.simulator_file):
                return candidate_provider()
            ctx.probe_param_status = (
                "no model yet: the visible base physics with hidden "
                "mechanisms disabled, running the real skill controllers")
            return self._base_physics_probe_model()

        ctx.probe_option_model_provider = probe_model
        ctx.probe_fit_provider = toolkit.fit_runner
        ctx.probe_validation_provider = toolkit.validation_runner
        ctx.probe_residuals_provider = toolkit.residuals_runner

        def current_observation() -> State:
            # Load a present candidate at its carried/declared values before
            # replaying observed memory. This never fits or takes a real step.
            if os.path.isfile(paths.simulator_file):
                assert ctx.probe_option_model_provider is not None
                ctx.probe_option_model_provider()
            obs = session.observe()
            ctx.current_belief = obs.belief
            return obs.frame

        ctx.current_observation_provider = current_observation
        ctx.skill_gate = (self._make_skill_gate(session, paths.simulator_file,
                                                trajectories)
                          if CFG.continual_require_model_on_test else None)
        probe_ns = build_probe_namespace(ctx)
        exec_ns["sim"] = probe_ns["sim"]
        exec_ns["BeliefProbe"] = probe_ns["BeliefProbe"]
        ctx.skill_preflight = (self._make_skill_preflight(
            session, probe_ns["BeliefProbe"], paths.simulator_file)
                               if CFG.continual_skill_preflight else None)
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
        """Deploy the round's files and parameter values without fitting."""
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
        if self._residual_env_cls is not None or (
                self._residual_rules is not None and self._fitted_params):
            _rules, _params = self._residual_rules, self._fitted_params

            def _step_fn(s: State, c: Any) -> Any:
                return apply_rules(s, _rules, _params, cmds=c)

            self._learned_simulator = LearnedSimulator(
                step_fn=_step_fn, name="agent_synthesized")
            combined = self._build_combined_simulator(self._learned_simulator)
            self._option_model = self._build_option_model(combined)
        self._last_round_modelled = True
        fit_version = self._probe_fit_state().get("version")
        if fit_version is not None and fit_version != self._fit_version_before:
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
        ctx.current_observation_provider = None
        ctx.skill_gate = None
        ctx.skill_preflight = None
        ctx.probe_param_status = None
        ctx.probe_artifact_loaders.clear()
        ctx.learn_cycle_index = None
        ctx.extra_session_hooks = {}

    # -- The test-level model gate ----------------------------------

    def _make_skill_gate(self, session: ProtocolSession, simulator_file: str,
                         trajectories: List[LowLevelTrajectory]) -> Any:
        """The ``ToolContext.skill_gate`` of this round under
        ``continual_require_model_on_test``: on a test level the skill
        tools refuse until :meth:`_model_readiness` is satisfied; on a
        train level (evidence collection) and once the level is over
        they never refuse."""

        def gate() -> Optional[str]:
            try:
                split = session.observe().level.split
            except EpisodeOver:
                return None
            if split != "test":
                return None
            return self._model_readiness(simulator_file, trajectories)

        return gate

    def _model_readiness(
            self, simulator_file: str,
            trajectories: List[LowLevelTrajectory]) -> Optional[str]:
        """Why the sandbox's ``simulator.py`` could not be deployed right now,
        or None when it could.

        Mirrors :meth:`_deploy_session_model`'s requirements so an
        artifact the gate accepts is one the round's end deploys: the
        file exists and execs, and it declares ``RESIDUAL_FEATURES``
        (the deploy asserts it; Boil and Fan Sonnet runs on Sept 16,
        2026 wrote models without it and ran on base physics). Fitting
        is the agent's call: a round deploys an unfitted model at its
        carried or declared values. Loads are cached by content digest
        so a sweep of ``skills_invoke`` calls execs the file once.
        """
        head = "Test level: this arm acts through its model. "
        if not os.path.isfile(simulator_file):
            return (head + "Write `./simulator.py` (RESIDUAL_ENV with "
                    "AGENT_PARAM_SPECS and RESIDUAL_FEATURES) in run_python "
                    "and rehearse the plan before invoking a skill.")
        with open(simulator_file, "rb") as f:
            digest = hashlib.sha256(f.read()).hexdigest()
        cache = getattr(self, "_readiness_cache_store", None)
        if cache is None:
            cache = {}
            setattr(self, "_readiness_cache_store", cache)
        if cache.get("digest") != digest:
            rules, specs, features, _ns = \
                self._load_simulator_from_module_file(
                    simulator_file, trajectories)
            cache.clear()
            cache.update(digest=digest,
                         loadable=rules is not None and specs is not None,
                         features=features is not None)
        if not cache["loadable"]:
            return (head + "`./simulator.py` does not load (run "
                    "`sim.reset(current=True)` in run_python to see the "
                    "error); fix it first.")
        if not cache["features"]:
            return (head + "`./simulator.py` declares no RESIDUAL_FEATURES, "
                    "so it cannot be deployed. Declare RESIDUAL_FEATURES on "
                    "the subclass (the observation features your dynamics "
                    "own; `{}` if none) first.")
        return None

    def _base_physics_probe_model(self) -> _OracleOptionModel:
        """The probe's model before a candidate ``simulator.py`` exists: the
        real skill controllers over the stock planning base env, the visible
        physics with hidden mechanisms disabled.

        A subclass model an earlier file installed is cleared first so
        the env is the stock one. Cached per base env instance.
        """
        self._install_residual_env_cls(None)
        cache = getattr(self, "_base_physics_model_cache", None)
        if cache is None or cache[0] is not self._base_env:
            model = _OracleOptionModel(self._initial_options,
                                       self._base_env.simulate)
            model.sim_env = self._base_env
            cache = (self._base_env, model)
            setattr(self, "_base_physics_model_cache", cache)
        return cache[1]

    def _make_skill_preflight(
            self, session: ProtocolSession, probe_factory: Callable[[], Any],
            simulator_file: str) -> Callable[[str], Optional[str]]:
        """The ``ToolContext.skill_preflight`` of this round under
        ``continual_skill_preflight``: the request's plan text rehearsed
        on a private probe from the last real observation, against the
        agent's candidate ``simulator.py``.

        The sim runs the real skill controllers, so a controller failure
        there (a grasp pose in contact, no collision-free path, a lift
        that leaves the object behind) is the refusal, carrying the
        controller's diagnostic that the real env withholds. When the
        observation channel is noisy and uncertainty decisions are on,
        the request is also rolled from ``continual_skill_preflight_draws``
        plausible poses; failing on more than half refuses it too.

        Before a candidate exists the request runs unrehearsed. The
        probe's fallback, the base physics with the hidden mechanisms
        disabled, is not the real env: the Opus Bridge runs of Sept 17,
        2026 had no model, and every refusal there was false, a welded
        partner rehearsed as a loose block. A rehearsal that cannot run
        (no observation yet, the probe's budget spent, a broken
        candidate) never blocks the request either: the failure is
        logged and the skill runs.
        """
        ctx = self._tool_context

        def _fails(step: Dict[str, Any]) -> bool:
            failure = step.get("failure")
            # "0 actions" alone is an option that terminated at once, not
            # a controller failure.
            return bool(failure) and failure != "0 actions"

        def preflight(plan_text: str) -> Optional[str]:
            if not os.path.isfile(simulator_file):
                return None
            try:
                session.observe()
            except EpisodeOver:
                return None
            try:
                probe = probe_factory()
                probe.reset(current=True)
                result = probe.run(plan_text, render=False)
            except Exception as e:  # pylint: disable=broad-except
                logging.warning(
                    "[Continual agent] skill preflight skipped: %s: %s",
                    type(e).__name__, e)
                return None
            model = ctx.probe_param_status or "the current model"
            head = f"Rehearsed in `sim` ({model}) from the last observation: "
            for i, step in enumerate(result.steps):
                if _fails(step):
                    return (head + f"skill {i + 1} ({step['option']}) fails "
                            f"there: {step['failure']}")
            draws = int(CFG.continual_skill_preflight_draws)
            noise = ObservationNoise.from_cfg()
            if not (draws > 0 and CFG.continual_uncertainty_decisions
                    and noise.enabled and noise.declared):
                return None
            try:
                probe.reset(current=True)
                belief = probe.run(plan_text, render=False, belief_draws=draws)
            except Exception as e:  # pylint: disable=broad-except
                logging.warning(
                    "[Continual agent] skill preflight belief draws "
                    "skipped: %s: %s",
                    type(e).__name__, e)
                return None
            failing = [d for d in belief.draws if _fails(d)]
            if len(failing) * 2 > len(belief.draws):
                return (head + f"the request fails on {len(failing)} of "
                        f"{len(belief.draws)} plausible poses of the "
                        f"objects (first: {failing[0]['failure']}). Look "
                        "again from rest so the belief narrows, or choose "
                        "parameters with margin over the spread.")
            return None

        return preflight

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
        return self._compute_base_pred_triples(obs_triples,
                                               self._workbench_env())

    def _workbench_env(self) -> Any:
        """The workbench's own base-sim world (the visible physics with no
        hidden mechanism), opened on first use and released with the round."""
        bench = self._workbench
        if bench.env is None:
            bench.env = create_new_env(CFG.env,
                                       do_cache=False,
                                       use_gui=False,
                                       skip_residual_dynamics=True)
        return bench.env

    def _release_workbench_env(self) -> None:
        bench = self._workbench
        if bench.env is not None:
            dispose_env(bench.env)
            bench.env = None

    # -- The scene package (engine, manifest, assets) -----------------------

    def _scene_state(self) -> State:
        """The initial state of the level being played (the run's first train
        task before any level starts): what the manifest describes."""
        task = self._tool_context.current_task
        if task is None:
            task = self._train_tasks[0]
        return task.init

    @staticmethod
    def _standalone_source(name: str, directory: Path) -> Path:
        """A copy of one engine module whose imports point at the copies beside
        it, so the reference reads as a self-contained package."""
        package = Path(__file__).resolve().parents[1]
        sources = {
            "pybullet_env.py": package / "envs" / "pybullet_env.py",
            "scene_base.py": package / "code_sim_learning" / "scene_base.py",
        }
        text = sources[name].read_text(encoding="utf-8")
        rebinds = {
            "from predicators.envs import BaseEnv\n":
            "from reference.base_sim.base_env import BaseEnv\n",
            "from predicators.envs.pybullet_env import PyBulletEnv\n":
            "from reference.base_sim.pybullet_env import PyBulletEnv\n",
        }
        for original, replacement in rebinds.items():
            if text.count(original) == 1:
                text = text.replace(original, replacement)
        target = directory / name
        target.write_text(text, encoding="utf-8")
        return target

    def _scene_package_files(self) -> Dict[str, str]:
        """Sandbox reference path -> source file of the engine wrapper, the
        scene manifest of the level being played and the asset files its bodies
        were loaded from (what the agentic real-to-sim arm builds its scene
        from)."""
        package = Path(__file__).resolve().parents[1]
        directory = Path(self._get_log_dir()) / "reference_sources"
        directory.mkdir(parents=True, exist_ok=True)
        files = {
            "base_sim/base_env.py":
            str(package / "envs" / "base_env.py"),
            "base_sim/pybullet_env.py":
            str(self._standalone_source("pybullet_env.py", directory)),
        }
        manifest, assets = build_scene_manifest(self._workbench_env(),
                                                self._scene_state())
        files["scene/scene_manifest.json"] = write_scene_manifest(
            manifest, str(directory / "scene_manifest.json"))
        files.update(assets)
        self._scene_package_summary = (len(manifest["bodies"]), len(assets))
        return files

    def _scene_package_paths(self) -> List[str]:
        """Agent-visible paths of :meth:`_scene_package_files`."""
        bodies, assets = self._scene_package_summary
        return [
            "./reference/base_sim/pybullet_env.py",
            "./reference/base_sim/base_env.py",
            f"./reference/scene/scene_manifest.json ({bodies} bodies)",
            f"./reference/assets/ ({assets} URDF and mesh files, named in "
            "the manifest)",
        ]

    def _get_sandbox_reference_files(self) -> Dict[str, str]:
        files = super()._get_sandbox_reference_files()
        if CFG.continual_provide_scene_package:
            files.update(self._scene_package_files())
        return files

    def _base_sim_reference_paths(self) -> List[str]:
        paths = super()._base_sim_reference_paths()
        if not CFG.continual_provide_scene_package:
            return paths
        package = self._scene_package_paths()
        names = {os.path.basename(path.split(" ")[0]) for path in package}
        # The twin's own core modules first, then the package; the
        # engine wrapper is listed once.
        return [path for path in paths if os.path.basename(path) not in names
                ] + package

    def _fit_status_text(self) -> str:
        """The last fit as one line for the prompt: the point estimate per
        parameter and the posterior sample count, never the raw result (its
        Jacobian dump is noise to the agent)."""
        result = getattr(self, "_last_fit_result", None)
        if result is None and getattr(self, "_param_specs", []):
            return ("UNFITTED for the current simulator.py; using carried "
                    "or declared parameter values. Call sim.fit() to fit")
        if result is None and getattr(self, "_residual_env_cls", None):
            return "no learnable parameters"
        published = self._probe_fit_state()
        if published:
            return format_fit_status(published)
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
        return play_tool_names(CONTINUAL_TOOL_NAMES)

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
