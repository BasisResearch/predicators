from typing import Set

from predicators.approaches.process_planning_approach import \
    BilevelProcessPlanningApproach
from predicators.ground_truth_models import get_gt_processes
from predicators.settings import CFG
from predicators.structs import NSRT, CausalProcess, Dataset


class ParamLearningBilevelProcessPlanningApproach(
        BilevelProcessPlanningApproach):
    """A bilevel planning approach that uses hand-specified processes."""

    def __init__(self,
                 initial_predicates,
                 initial_options,
                 types,
                 action_space,
                 train_tasks,
                 task_planning_heuristic="default",
                 max_skeletons_optimized=-1,
                 bilevel_plan_without_sim=None,
                 processes=None,
                 option_model=None):
        super().__init__(initial_predicates,
                         initial_options,
                         types,
                         action_space,
                         train_tasks,
                         task_planning_heuristic,
                         max_skeletons_optimized,
                         bilevel_plan_without_sim,
                         option_model=option_model)
        if processes is None:
            processes = get_gt_processes(CFG.env, self._initial_predicates,
                                         self._initial_options)
        self._processes = processes

    @classmethod
    def get_name(cls):
        return "param_learning_process_planning"

    @property
    def is_learning_based(self):
        return True

    def _get_current_processes(self) -> Set[CausalProcess]:
        return self._processes

    def _get_current_nsrts(self) -> Set[NSRT]:
        """Get the current set of NSRTs."""
        return set()

    def learn_from_offline_dataset(self, dataset: Dataset):
        pass
