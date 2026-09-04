"""The LLM agent arms of the continual protocol (docs/continual-protocol.md).

Both arms mix :class:`ContinualPlayMixin` (the play loop) in front of
the phased approach class that holds their machinery:

* ``AgentContinualApproach`` (``agent_continual``) is C1's learner
  (hybrid simulator synthesis, parameter fit, predicate invention) on
  ``AgentSimPredicateInventionApproach``. Its sessions also get
  ``run_python`` with the belief probe ``sim`` and ``learn_run``, which
  runs a learning session over every recorded episode inside the play
  session: the play session's manager is parked and its clock paused
  while the learning session runs in a manager of its own, and the
  refit model is behind ``sim`` when the call returns.
* ``AgentContinualModelFreeApproach`` (``agent_continual_model_free``)
  is the model-free baseline on ``AgentModelFreeApproach``: the same env
  and skill tools, sandbox and journal, but no belief model, no ``sim``
  and no learning session. What it knows comes from the recorded data
  and the real environment.

Neither arm starts with a predicate: the observation is the object
features and a render, the goal is its natural-language description,
and the model-based arm invents predicates as it learns. The allowlist
``agent_sim_learn_kept_predicates_names`` hands either arm env
predicates when an experiment wants that.
"""
from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, Callable, Dict, FrozenSet, List, \
    Optional, Set

from predicators.agent_sdk import journal as journal_mod
from predicators.agent_sdk.play_prompts import render_learning_status
from predicators.agent_sdk.session_base import AgentSessionFatalError
from predicators.agent_sdk.tools.continual_tools import CONTINUAL_TOOL_NAMES
from predicators.approaches.agent_model_free_approach import \
    AgentModelFreeApproach
from predicators.approaches.agent_sim_learning_approach import \
    resolve_kept_predicate_names
from predicators.approaches.agent_sim_predicate_invention_approach import \
    AgentSimPredicateInventionApproach
from predicators.approaches.continual_play_mixin import ContinualPlayMixin
from predicators.option_model import _OptionModelBase
from predicators.structs import Predicate

if TYPE_CHECKING:  # pragma: no cover - the run package imports approaches
    from predicators.agent_sdk.session_manager import SessionManagerProtocol
    from predicators.run.continual import ProtocolSession

# The env predicates an arm starts with unless the CFG allowlist hands
# it some: none.
NO_ENV_PREDICATES: FrozenSet[str] = frozenset()


class AgentContinualApproach(ContinualPlayMixin,
                             AgentSimPredicateInventionApproach):
    """C1's learner (hybrid simulator, parameter fit, predicate invention)
    playing under the continual protocol."""

    _save_suffix = "AgentContinual"

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

    def _learning_status(self, session: ProtocolSession) -> str:
        n_eps, n_steps = self._episode_counts(session)
        return render_learning_status(
            n_learn=self._learn_runs,
            sim_version=self._current_simulator_version,
            pred_version=self._current_predicates_version,
            fit_status=self._last_fit_status,
            n_episodes=n_eps,
            n_steps=n_steps,
            n_new_episodes=max(0, n_eps - self._episodes_at_last_learn))

    def _learn_callable(
            self, session: ProtocolSession) -> Optional[Callable[[str], str]]:
        return lambda note: self._learn_now(session, note)

    # -- Parent-specific overrides ---------------------------------------

    def _get_agent_system_prompt(self) -> str:
        if self._learning_mode:  # a learning session inside a play session
            return super()._get_agent_system_prompt()
        return self._play_system_prompt()

    def _extra_save_state(self) -> Dict[str, Any]:
        state = super()._extra_save_state()
        state.update(self._continual_save_state())
        return state

    def _load_extra_save_state(self, save_dict: Dict[str, Any]) -> None:
        super()._load_extra_save_state(save_dict)
        self._load_continual_save_state(save_dict)

    # -- Learning -----------------------------------------------------------

    def _learn_now(self, session: ProtocolSession, note: str) -> str:
        """Run a learning session inside the play session and return its
        summary for the agent.

        The learning session is an SDK session of its own, so the play
        session's manager is parked, not closed (its CLI is waiting on
        this tool call), and the attempt clock is paused for the
        learning's duration. The recorded episodes so far, including
        this session's, are the data.
        """
        t0 = time.monotonic()
        with self._tool_context.attempt_clock_paused(), \
                self._parked_agent_session() as parked:
            self._sync_level_trajectories(session)
            outcome = self._run_learn(session, note)
        if parked is not None:
            self._refresh_play_session(parked)
        n_eps, n_steps = self._episode_counts(session)
        minutes = (time.monotonic() - t0) / 60.0
        return (f"Learning session {self._learn_runs} {outcome} in "
                f"{minutes:.0f} min over {n_eps} recorded episode(s), "
                f"{n_steps} steps. Belief model version: "
                f"{self._current_simulator_version}; predicates version: "
                f"{self._current_predicates_version}; fit: "
                f"{self._last_fit_status}. `sim` now serves this model in "
                "your run_python namespace, and ./data/trajectories.pkl is "
                "refreshed.")

    @staticmethod
    def _refresh_play_session(manager: SessionManagerProtocol) -> None:
        """After a learning session: ``session_info.json`` names the play
        session again, and the sandbox's data pickle carries the episodes so
        far (the local sandbox manager exports it; others have no pickle)."""
        export_data = getattr(manager, "_export_data", None)
        for name, fn in (("save_session_info", manager.save_session_info),
                         ("_export_data", export_data)):
            if fn is None:
                continue
            try:
                fn()
            except Exception as e:  # pylint: disable=broad-except
                logging.warning(
                    "[Continual agent] %s after learning failed: "
                    "%s", name, e)

    def _run_learn(self, session: ProtocolSession, note: str) -> str:
        """Run the inherited learning over every recorded episode; returns the
        outcome (``completed`` or ``failed: ...``)."""
        trajectories = self._get_all_trajectories()
        logging.info(
            "[Continual agent] learning session %d requested (%s) over %d "
            "trajectories", self._learn_runs + 1, note, len(trajectories))
        t0 = time.time()
        try:
            self._learn_simulator(trajectories)
            self._last_fit_status = self._fit_status_text()
            outcome = "completed"
        except AgentSessionFatalError:
            raise
        except Exception as e:  # pylint: disable=broad-except
            logging.exception("[Continual agent] learning failed")
            self._last_fit_status = f"last learning failed: {e}"
            outcome = f"failed: {e}"
        self._learn_runs += 1
        n_eps, _ = self._episode_counts(session)
        self._episodes_at_last_learn = n_eps
        session.record_sandbox("learn_sessions", 1)
        session.record_sandbox("fits", 1)
        # The learner may have installed predicates: the runner's
        # abstraction (Wait targets, divergence checks) follows.
        self._sync_tool_context()
        session.abstract_predicates = set(self._get_all_predicates())
        journal_mod.append_entry(
            self._tool_context.sandbox_dir or self._get_log_dir(),
            f"Learning session {self._learn_runs} ({outcome}, "
            f"{time.time() - t0:.0f} s)",
            f"Requested with note: {note or '(none)'}. Trajectories: "
            f"{len(trajectories)}. Simulator version: "
            f"{self._current_simulator_version}; predicates version: "
            f"{self._current_predicates_version}.",
            filename=journal_mod.ATTEMPTS_FILENAME)
        self.save(session.level_index)
        return outcome

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
        return [n for n in CONTINUAL_TOOL_NAMES if n != "learn_run"]

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
