"""The base of the agent arms: the agent session, its tool context, the
recorded trajectories and the checkpoints.

The arms themselves (``agent_continual_approach`` and its siblings) add
the play loop of the continual protocol through ``ContinualPlayMixin``;
this class and the simulator-learning classes between it and the arms
carry the machinery they share. None of them solves a task on its own.
"""
import datetime
import logging
import os
from typing import Any, Callable, Dict, List, Optional, Set, cast

import dill as pkl
from gym.spaces import Box

from predicators import utils
from predicators.approaches import ApproachFailure
from predicators.approaches.agent_session_mixin import AgentSessionMixin
from predicators.approaches.base_approach import BaseApproach
from predicators.option_model import _OptionModelBase, create_option_model
from predicators.settings import CFG
from predicators.structs import Action, Dataset, LowLevelTrajectory, \
    ParameterizedOption, Predicate, State, Task, Type


class AgentModelFreeApproach(AgentSessionMixin, BaseApproach):
    """The agent session, tool context, trajectories and checkpoints the agent
    arms share."""

    def __init__(self,
                 initial_predicates: Set[Predicate],
                 initial_options: Set[ParameterizedOption],
                 types: Set[Type],
                 action_space: Box,
                 train_tasks: List[Task],
                 *args: Any,
                 option_model: Optional[_OptionModelBase] = None,
                 **kwargs: Any) -> None:
        super().__init__(initial_predicates, initial_options, types,
                         action_space, train_tasks, *args, **kwargs)
        self._offline_dataset = Dataset([])
        self._online_trajectories: List[LowLevelTrajectory] = []
        self._option_model: Optional[_OptionModelBase] = (
            option_model if option_model is not None else
            self._create_planner_option_model())
        # Terminate Wait on atom change using the approach's predicates (which
        # may include invented ones), looked up lazily so the lambda picks up
        # predicates invented after __init__.
        if self._option_model is not None and \
                CFG.wait_option_terminate_on_atom_change:
            cast(  # pylint: disable=protected-access
                Any, self._option_model)._abstract_function = (
                    lambda s: utils.abstract(s, self._get_all_predicates()))
        self._run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Initializes _tool_context and _agent_session_id (see mixin).
        self._init_agent_session_state(self._types, self._initial_predicates,
                                       initial_options, train_tasks)

        # Capture the underlying env once at construction. The initial option
        # model wraps ``env.simulate`` (a bound method), so ``__self__`` is the
        # env. A later model may rebuild ``_option_model`` with a plain
        # learned simulator that has no ``__self__``; pinning the env
        # reference here keeps scene rendering (the probe's sim.render)
        # working in every round.
        env_self = getattr(getattr(self._option_model, '_simulator', None),
                           '__self__', None)
        if env_self is not None:
            self._tool_context.env = env_self

    @classmethod
    def get_name(cls) -> str:
        return "agent_model_free"

    @property
    def is_learning_based(self) -> bool:
        return True

    def _get_log_dir(self) -> str:
        """Return per-run log directory (created by configure_logging)."""
        log_dir = super()._get_log_dir()
        os.makedirs(log_dir, exist_ok=True)
        logging.info("Logging agent queries/responses to: %s", log_dir)
        return log_dir

    # ------------------------------------------------------------------ #
    # Overridable helpers (for subclass customisation)
    # ------------------------------------------------------------------ #

    def _get_all_options(self) -> Set[ParameterizedOption]:
        """Return the full set of options available for planning."""
        return self._initial_options

    def _get_all_predicates(self) -> Set[Predicate]:
        """Return the full set of predicates for abstraction."""
        return self._initial_predicates

    def _get_all_trajectories(self) -> List[LowLevelTrajectory]:
        """Return all trajectories (offline + online)."""
        return self._offline_dataset.trajectories + self._online_trajectories

    def _create_planner_option_model(self) -> Optional[_OptionModelBase]:
        """Build the option model the tools roll plans out through.

        Honors two CFG knobs:

        * ``agent_planner_use_simulator`` -- when False, returns ``None``
          so the agent has no simulator to roll plans out in.
        * ``agent_planner_use_base_simulator`` -- when True (and a
          simulator is used), wraps the *base* env
          (``skip_residual_dynamics=True``), denying the planner the delayed
          ``_domain_specific_step`` dynamics; otherwise wraps the real env.
        """
        if not CFG.agent_planner_use_simulator:
            return None
        return create_option_model(
            CFG.option_model_name,
            skip_residual_dynamics=CFG.agent_planner_use_base_simulator)

    def _get_sandbox_reference_files(self) -> Dict[str, str]:
        """Document public control semantics without exporting
        implementation."""
        return {"skills.md": "predicators/agent_sdk/prompts/public_skills.md"}

    def _solve(self, task: Task, timeout: int) -> Callable[[State], Action]:
        raise ApproachFailure(
            f"{self.get_name()} has no task solver: the agent arms play "
            "levels under the continual protocol "
            "(ContinualPlayMixin.play_level).")

    def _sync_tool_context(self) -> None:
        """Push current approach state into the shared ToolContext.

        The MCP tools (run_python and the play tools) read from the
        ToolContext dataclass, not the approach directly. This keeps
        them in sync after mutations (e.g. new trajectories collected,
        options added). Subclasses should call super() and then set
        additional fields (e.g. skill_factory_context).
        """
        self._tool_context.types = self._types
        # The agent's predicate vocabulary, not the raw env set: tools
        # abstract states, list predicates, and parse plan annotations
        # from this, so stripped predicates (agent_sim_learning
        # allowlist) must not leak in and invented ones must appear.
        self._tool_context.predicates = self._get_all_predicates()
        self._tool_context.options = self._initial_options
        self._tool_context.show_option_source = False
        self._tool_context.gt_options_ref_path = None
        self._tool_context.train_tasks = self._train_tasks
        self._tool_context.offline_trajectories = \
            self._offline_dataset.trajectories
        self._tool_context.online_trajectories = self._online_trajectories
        self._tool_context.log_dir = self._get_log_dir()
        self._tool_context.option_model = self._option_model
        # Wire the active-experiment info-gain scorer when a learning subclass
        # exposes one and info-seeking exploration is on. Syncing the bound
        # method (not a snapshot) keeps it pointed at the latest fit/ensemble.
        # getattr guard: non-learning approaches lack it.
        if CFG.agent_explorer_info_seeking:
            self._tool_context.atom_disagreement_fn = getattr(
                self, "score_atom_disagreement", None)
        else:
            self._tool_context.atom_disagreement_fn = None
        all_trajs = (self._offline_dataset.trajectories +
                     self._online_trajectories)
        if all_trajs:
            self._tool_context.example_state = all_trajs[0].states[0]

        # Refresh env from the option model only if extraction succeeds. After
        # sim learning, ``_simulator`` may be a plain lambda with no
        # ``__self__``; don't clobber the env reference seeded in ``__init__``
        # in that case.
        if self._option_model is not None and \
                hasattr(self._option_model, '_simulator'):
            env_self = getattr(
                self._option_model._simulator,  # pylint: disable=protected-access
                '__self__',
                None)
            if env_self is not None:
                self._tool_context.env = env_self

    # ------------------------------------------------------------------ #
    # Save / Load
    # ------------------------------------------------------------------ #

    # Filename suffix for the pickled approach state. Subclasses that
    # persist extra fields override this so their saves don't collide
    # with the base planner's.
    _save_suffix: str = "AgentPlanner"

    def _extra_save_state(self) -> Dict[str, Any]:
        """Subclass hook: extra (key -> value) pairs to persist.

        Merged into the base save dict; restored by the matching
        :meth:`_load_extra_save_state`.
        """
        return {}

    def _load_extra_save_state(self, save_dict: Dict[str, Any]) -> None:
        """Subclass hook: restore fields written by _extra_save_state.

        Called after the base fields are restored and ``_run_id`` has
        been refreshed, but before the tool context is re-synced.
        """

    def save(self, online_learning_cycle: Optional[int] = None) -> None:
        """Save approach state to disk; the continual runner names the
        checkpoint after the level it closes (``online_learning_cycle``)."""
        save_path = utils.get_approach_save_path_str()
        path = f"{save_path}_{online_learning_cycle}.{self._save_suffix}"
        save_dict = {
            "offline_dataset":
            self._offline_dataset,
            "online_trajectories":
            self._online_trajectories,
            "run_id":
            self._run_id,
            "agent_session_id":
            (self._agent_session.session_id if self._agent_session else None),
            **self._extra_save_state(),
        }
        with open(path, "wb") as f:
            pkl.dump(save_dict, f)
        logging.info("[Run %s] Saved approach to %s", self._run_id, path)

    def load(self, online_learning_cycle: Optional[int] = None) -> None:
        save_path = utils.get_approach_load_path_str()
        path = f"{save_path}_{online_learning_cycle}.{self._save_suffix}"
        with open(path, "rb") as f:
            save_dict = pkl.load(f)

        self._offline_dataset = save_dict["offline_dataset"]
        self._online_trajectories = save_dict["online_trajectories"]
        # pylint: disable=attribute-defined-outside-init
        # (_agent_session_id is initialized via the agent-session mixin.)
        self._agent_session_id = save_dict.get("agent_session_id")

        # New run_id for continued execution (each run gets its own dir), but
        # log the original run_id for reference.
        original_run_id = save_dict.get("run_id", "unknown")
        self._run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        self._load_extra_save_state(save_dict)

        # Re-sync tool context (subclass fields are restored first).
        self._sync_tool_context()

        logging.info(
            "[Run %s] Loaded from previous run %s: %d offline, %d online "
            "trajectories", self._run_id, original_run_id,
            len(self._offline_dataset.trajectories),
            len(self._online_trajectories))
