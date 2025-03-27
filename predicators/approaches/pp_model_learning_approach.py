import logging
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Set

import numpy as np
from gym.spaces import Box
from scipy.optimize import minimize
from tqdm.auto import tqdm

from predicators import planning, utils
from predicators.approaches.pp_param_learning_approach import \
    ParamLearningBilevelProcessPlanningApproach
from predicators.ground_truth_models import get_gt_processes
from predicators.option_model import _OptionModelBase
from predicators.settings import CFG
from predicators.structs import NSRT, AtomOptionTrajectory, CausalProcess, \
    Dataset, GroundAtom, ParameterizedOption, Predicate, Task, Type, \
    _GroundCausalProcess

class ProcessLearningBilevelProcessPlanningApproach(
        ParamLearningBilevelProcessPlanningApproach):
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
                 option_model: Optional[_OptionModelBase] = None):
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
        self._processes: List[CausalProcess] = sorted(processes)

    @classmethod
    def get_name(cls):
        return "process_learning_process_planning"

    @property
    def is_learning_based(self):
        return True

    def _get_current_processes(self) -> Set[CausalProcess]:
        return set(self._processes)

    def _get_current_nsrts(self) -> Set[NSRT]:
        """Get the current set of NSRTs."""
        return set()

    def learn_from_offline_dataset(self,
                                   dataset: Dataset,
                                   guide_per_process: bool = False) -> None:
        """Learn models from the offline datasets."""
        ...
    
    def learn_from_interaction_results(self, interaction_results) -> None:
        """Learn models from the interaction results."""
        ...
