"""Standalone program dynamics inside the shared continual conversation."""
from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any, Callable, Dict, FrozenSet, List, \
    Optional

import numpy as np

from predicators.agent_sdk.play_prompts import render_tool_list
from predicators.agent_sdk.prompt_templates import render
from predicators.agent_sdk.tools.continual_tools import CONTINUAL_TOOL_NAMES
from predicators.agent_sdk.tools.program_synthesis import \
    STANDALONE_PROBE_DISABLED, STANDALONE_RUN_PYTHON_DESCRIPTION, \
    CandidateLoader
from predicators.approaches.agent_program_world_model_approach import \
    AgentProgramWorldModelApproach
from predicators.approaches.agent_sim_learning_approach import \
    resolve_kept_predicate_names
from predicators.approaches.continual_play_mixin import ContinualPlayMixin
from predicators.code_sim_learning.program_world_model import \
    ProgramOptionModel
from predicators.observation_noise import ObservationNoise
from predicators.settings import CFG
from predicators.structs import LowLevelTrajectory, State

if TYPE_CHECKING:
    from predicators.run.continual import ProtocolSession


class AgentContinualProgramWorldModelApproach(ContinualPlayMixin,
                                              AgentProgramWorldModelApproach):
    """Learn skill transitions without a supplied physics simulator."""

    _save_suffix = "AgentContinualProgramWM"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._program_trajectories: List[LowLevelTrajectory] = []
        self._program_round: Optional[Any] = None
        # A program must be written before any simulated prediction.
        self._option_model = None
        self._tool_context.probe_engine_available = False

    @classmethod
    def get_name(cls) -> str:
        return "agent_continual_program_world_model"

    def _resolve_kept_names(self) -> Optional[FrozenSet[str]]:
        return resolve_kept_predicate_names(frozenset())

    def _continual_tool_names(self) -> List[str]:
        return ["run_python"] + list(CONTINUAL_TOOL_NAMES)

    def _learning_cycle_index(self) -> int:
        return self._rounds_played

    def _program_tool_overrides(self) -> Dict[str, Any]:
        return {"run_python_description": STANDALONE_RUN_PYTHON_DESCRIPTION}

    def _program_probe_disabled(self) -> FrozenSet[str]:
        # Close to WorldCoder: score the program on the data and roll a
        # plan through it once; plan search, repeated trials, predicate
        # scoring and engine renders of predicted states are withheld.
        return STANDALONE_PROBE_DISABLED

    def _get_sandbox_reference_files(self) -> Dict[str, str]:
        return {
            key: value
            for key, value in super()._get_sandbox_reference_files().items()
            if not key.startswith("base_sim/")
        }

    def _play_system_prompt(self) -> str:
        sections = [
            render("play_system", "identity_model_free"),
            render("play_system", "protocol"),
            render("play_system", "observations")
        ]
        noise = ObservationNoise.from_cfg()
        if noise.enabled and noise.declared:
            sections.append(
                render("play_system",
                       "observation_noise",
                       noise_line=noise.describe() + "."))
        sections += [
            render("play_program", "workflow"),
            render("play_system",
                   "tools",
                   tool_list=render_tool_list(CONTINUAL_TOOL_NAMES)),
            render("play_system", "grammar"),
            render("play_system",
                   "sandbox",
                   model_files=
                   "Keep world_model.py, predicates.py, and journal.md."),
            render("play_system", "journal_model_free"),
            render("play_system", "context"),
            render("play_program", "contract"),
        ]
        return "\n\n".join(sections)

    def _get_agent_system_prompt(self) -> str:
        return self._play_system_prompt()

    def _model_status(self, session: ProtocolSession) -> str:
        n_eps, n_steps = self._episode_counts(session)
        if self._program is None:
            return render("play_query",
                          "no_world_model",
                          n_episodes=str(n_eps),
                          n_steps=str(n_steps))
        return render("play_query",
                      "world_model_status",
                      world_model_version=self._current_simulator_version
                      or "unversioned",
                      n_episodes=str(n_eps),
                      n_steps=str(n_steps))

    def _round_extra_tools(self, session: ProtocolSession) -> List[Any]:
        self._refresh_arm_data(session)
        paths = self._resolve_synthesis_paths()
        wm_paths = self._world_model_paths(paths)
        extra_paths = self._compute_extra_synthesis_paths(paths.base)
        self._program_round = (wm_paths, extra_paths)
        namespace = self._build_synthesis_exec_ns(self._program_trajectories)
        # The inherited evaluator can replay engine dynamics. Do not offer
        # that side channel in the standalone-model comparison.
        namespace.pop("evaluate_trajectory", None)
        self._attach_program_session_state(namespace,
                                           self._program_trajectories, paths,
                                           wm_paths, extra_paths)
        self._tool_context.current_observation_provider = \
            lambda: self._current_program_observation(session)
        return list(self._tool_context.extra_mcp_tools)

    def _make_candidate_program_provider(
            self, load_candidate: CandidateLoader
    ) -> Callable[[], ProgramOptionModel]:
        provider = super()._make_candidate_program_provider(load_candidate)

        def current_model() -> ProgramOptionModel:
            model = provider()
            # Predicate materialization and particle draws must follow the
            # same candidate as planning, including edits within a round.
            self._program = model.program
            self._program_model = model
            self._option_model = model
            self._tool_context.option_model = model
            return model

        return current_model

    def _current_program_observation(self, session: ProtocolSession) -> State:
        """Replay this episode's observed skill prefix under the candidate.

        Recompute from the initial observation so model edits and
        checkpoint restoration cannot retain memory inferred under a
        different model. Recorded frames remain untouched and each
        transition starts from its actual noisy pre-observation, not a
        model-predicted trajectory.
        """
        provider = self._tool_context.probe_option_model_provider
        assert provider is not None
        candidate = provider()
        assert isinstance(candidate, ProgramOptionModel)
        obs = session.observe()
        self._tool_context.current_belief = obs.belief
        episode = session.level_episodes()[-1]
        states, actions = episode["states"], episode["actions"]
        if any(not action.has_option() for action in actions):
            raise ValueError(
                "The skill-level program cannot reconstruct memory after "
                "primitive actions. Use an explicit observed start with "
                "your inferred latent, or rehearse from the level start.")
        # Use a private RNG/model: asking for the same current frame twice
        # must not consume the planning model's stochastic rollout stream.
        replay = ProgramOptionModel(candidate.program, seed=CFG.seed)
        latent = replay.initial_latent(states[0],
                                       rng=np.random.default_rng(CFG.seed))
        index = 0
        while index < len(actions):
            option = actions[index].get_option()
            start = states[index].copy()
            start.latent = latent
            predicted, _ = replay.get_next_state_and_num_actions(
                start, copy.deepcopy(option))
            assert predicted.latent is not None
            latent = predicted.latent
            index += 1
            while (index < len(actions)
                   and actions[index].get_option() is option):
                index += 1
        current = obs.frame.copy()
        current.latent = latent
        return current

    def _round_hooks(self, session: ProtocolSession) -> Dict[str, list]:
        del session
        return self._tool_context.extra_session_hooks

    def _refresh_arm_data(self, session: ProtocolSession) -> None:
        del session
        self._program_trajectories[:] = self._get_all_trajectories()
        self._fit_trajectories = self._program_trajectories

    def _after_round(self, session: ProtocolSession, state: Any) -> None:
        del state
        try:
            if self._program_round is not None:
                program = self._load_program_artifacts(*self._program_round)
                if program is not None:
                    self._install_program(program)
                    self._sync_tool_context()
                    session.abstract_predicates = set(
                        self._get_all_predicates())
        finally:
            ctx = self._tool_context
            ctx.probe_option_model_provider = None
            ctx.probe_score_provider = None
            ctx.probe_fit_provider = None
            ctx.probe_validation_provider = None
            ctx.probe_residuals_provider = None
            ctx.probe_param_status = None
            ctx.current_observation_provider = None
            ctx.learn_cycle_index = None
            ctx.extra_session_hooks = {}
            self._program_round = None
            self._learning_mode = False

    def _extra_save_state(self) -> Dict[str, Any]:
        state = super()._extra_save_state()
        state.update(self._continual_save_state())
        return state

    def _load_extra_save_state(self, save_dict: Dict[str, Any]) -> None:
        super()._load_extra_save_state(save_dict)
        self._load_continual_save_state(save_dict)
