from typing import Any, List, Optional, Set

from predicators import utils
from predicators.nsrt_learning.process_learning.base_process_learner import \
    BaseProcessLearner
from predicators.settings import CFG
from predicators.structs import PAPAD, CausalProcess, ExogenousProcess, \
    LowLevelTrajectory, Predicate, Segment, Task

__all__ = ["BaseProcessLearner"]

# Import submodules to register them.
utils.import_submodules(__path__, __name__)


def learn_exogenous_processes(trajectories: List[LowLevelTrajectory],
                              train_tasks: List[Task],
                              predicates: Set[Predicate],
                              segmented_trajs: List[List[Segment]],
                              verify_harmlessness: bool,
                              annotations: Optional[List[Any]],
                              verbose: bool = True) -> List[ExogenousProcess]:
    """Learn exogenous processes on the given data segments."""
    for cls in utils.get_all_subclasses(BaseProcessLearner):
        if not cls.__abstractmethods__ and \
            cls.get_name() == CFG.exogenous_process_learner:
            learner = cls(...)
