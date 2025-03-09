from typing import List, Optional, Set

from gym.spaces import Box

from predicators.approaches.dynamic_bilevel_planning_approach import \
    DynamicBilevelPlanningApproach
from predicators.ground_truth_models import get_gt_processes
from predicators.option_model import _OptionModelBase
from predicators.settings import CFG
from predicators.structs import NSRT, CausalProcess, ParameterizedOption, \
    Predicate, Task, Type


class OracleDynamicBilevelPlanningApproach(DynamicBilevelPlanningApproach):
    """A bilevel planning approach that uses hand-specified processes."""

    def __init__(self,
                 initial_predicates: Set[Predicate],
                 initial_options: Set[ParameterizedOption],
                 types: Set[Type],
                 action_space: Box,
                 train_tasks: List[Task],
                 task_planning_heuristic: str = "default",
                 max_skeletons_optimized: int = -1,
                 bilevel_plan_without_sim: Optional[bool] = None,
                 processes: Optional[Set[CausalProcess]] = None,
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
        if processes is None:
            processes = get_gt_processes(CFG.env, self._initial_predicates,
                                         self._initial_options)
        self._processes = processes

    @classmethod
    def get_name(cls) -> str:
        return "oracle_process_model"

    @property
    def is_learning_based(self) -> bool:
        return False

    def _get_current_processes(self) -> Set[CausalProcess]:
        return self._processes

    def _get_current_nsrts(self) -> Set[NSRT]:
        """Get the current set of NSRTs."""
        return set()
