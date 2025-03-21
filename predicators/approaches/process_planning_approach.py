import abc
import logging
from typing import Any, Callable, List, Optional, Set, Tuple

from gym.spaces import Box

from predicators import utils
from predicators.approaches import ApproachFailure, ApproachTimeout, \
    BaseApproach
from predicators.approaches.bilevel_planning_approach import \
    BilevelPlanningApproach
from predicators.option_model import _OptionModelBase, create_option_model
from predicators.planning import PlanningFailure, PlanningTimeout
from predicators.planning_with_processes import \
    run_task_plan_with_processes_once
from predicators.settings import CFG
from predicators.structs import Action, CausalProcess, GroundAtom, \
    Metrics, ParameterizedOption, Predicate, State, Task, Type, \
    _GroundEndogenousProcess, _Option


class BilevelProcessPlanningApproach(BilevelPlanningApproach):
    """A bilevel planning approach that doesn't use the nsrt world model but
    uses the process world model."""

    def __init__(self,
                 initial_predicates: Set[Predicate],
                 initial_options: Set[ParameterizedOption],
                 types: Set[Type],
                 action_space: Box,
                 train_tasks: List[Task],
                 task_planning_heuristic: str = "default",
                 max_skeletons_optimized: int = -1,
                 bilevel_plan_without_sim: Optional[bool] = None,
                 option_model: Optional[_OptionModelBase] = None) -> None:
        super().__init__(initial_predicates,
                         initial_options,
                         types,
                         action_space,
                         train_tasks,
                         task_planning_heuristic,
                         max_skeletons_optimized,
                         bilevel_plan_without_sim,
                         option_model=option_model)
        self._last_option_plan: List[_Option] = []  # used if plan WITH sim

    @abc.abstractmethod
    def _get_current_processes(self) -> Set[CausalProcess]:
        """Get the current set of Processes."""
        raise NotImplementedError("Override me!")

    def _solve(self, task: Task, timeout: int) -> Callable[[State], Action]:
        self._num_calls += 1
        # ensure random over successive
        seed = self._seed + self._num_calls
        processes = self._get_current_processes()
        preds = self._get_current_predicates()

        # Run task planning only and then greedily sample
        # and execute in the policy.
        if self._plan_without_sim:
            process_plan, atoms_seq, metrics = self._run_task_plan(
                task, processes, preds, timeout, seed)
            self._last_process_plan = process_plan
            self._last_atoms_seq = atoms_seq
            policy = utils.process_plan_to_greedy_policy(
                process_plan,
                task.goal,
                self._rng,
                noop_option_terminate_on_atom_change=True,
                abstract_function=lambda s: utils.abstract(s, preds))
            logging.debug("Current Task Plan:")
            for process in process_plan:
                logging.debug(process.name)
        else:
            ...

        self._save_metrics(metrics, processes, preds)

        def _policy(s: State) -> Action:
            try:
                return policy(s)
            except utils.OptionExecutionFailure as e:
                raise ApproachFailure(e.args[0], e.info)

        return _policy

    def _run_task_plan(
        self, task: Task, processes: Set[CausalProcess], preds: Set[Predicate],
        timeout: int, seed: int, **kwargs: Any
    ) -> Tuple[List[_GroundEndogenousProcess], List[Set[GroundAtom]], Metrics]:
        try:
            plan, atoms_seq, metrics = run_task_plan_with_processes_once(
                task,
                processes,
                preds,
                self._types,
                timeout,
                seed,
                task_planning_heuristic=self._task_planning_heuristic,
                max_horizon=float(CFG.horizon),
                **kwargs)
        except PlanningFailure as e:
            raise ApproachFailure(e.args[0], e.info)
        except PlanningTimeout as e:
            raise ApproachTimeout(e.args[0], e.info)

        return plan, atoms_seq, metrics

    def _save_metrics(self, metrics, processes, predicates):
        pass
