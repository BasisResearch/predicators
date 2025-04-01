from typing import List, Set, Optional, Any

from predicators import utils
from predicators.nsrt_learning.process_learning.base_process_learner import (
    BaseProcessLearner,
)
from predicators.structs import LowLevelTrajectory, Task, Predicate, Segment, \
    CausalProcess, PAPAD

__all__ = ["BaseProcessLearner"]

# Import submodules to register them.
utils.import_submodules(__path__, __name__)

def learn_processes(trajectories: List[LowLevelTrajectory],
                           train_tasks: List[Task],
                           predicates: Set[Predicate],
                           segmented_trajs: List[List[Segment]],
                           verify_harmlessness: bool,
                           annotations: Optional[List[Any]],
                           verbose: bool = True) -> List[PAPAD]:
    pass