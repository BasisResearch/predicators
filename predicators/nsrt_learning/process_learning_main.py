import logging
from typing import List, Optional, Set, Any

from gym.spaces import Box

from predicators.settings import CFG
from predicators.nsrt_learning.segmentation import segment_trajectory
from predicators.structs import LowLevelTrajectory, Task, Predicate, \
    ParameterizedOption, GroundAtomTrajectory, CausalProcess
from predicators.nsrt_learning.strips_learning import learn_strips_operators
from predicators.nsrt_learning.process_learning import learn_processes
from predicators.nsrt_learning.nsrt_learning_main import _learn_pnad_options, \
    _learn_pnad_samplers



def learn_processes_from_data(
        trajectories: List[LowLevelTrajectory], train_tasks: List[Task],
        predicates: Set[Predicate], known_options: Set[ParameterizedOption],
        action_space: Box, 
        ground_atom_dataset: Optional[List[GroundAtomTrajectory]],
        sampler_learner: str, annotations: Optional[List[Any]]
        ) -> Set[CausalProcess]:
    """Learn CausalProcesses from the given dataset of low-level transitions,
    using the given set of predicates.
    """
    logging.info(f"\nLearning CausalProcesses on {len(trajectories)} "
                 "trajectories...")

    # We will probably learn endogenous and exogenous processes separately.
    # -- Learn the endogenous processes --
    # STEP 1: Segment the trajectory by options. (don't currently consider
    #         segmenting by predicates).
    #         Segment each trajectory in the dataset based on changes in
    #         either predicates or options. If we are doing option learning,
    #         then the data will not contain options, so this segmenting
    #         procedure only uses the predicates.
    #         If we know the option segmentations this is pretty similar to 
    #         learning NSRTs.
    if ground_atom_dataset is None:
        segmented_trajs = [
            segment_trajectory(traj, predicates) for traj in trajectories
        ]
    else:
        segmented_trajs = [
            segment_trajectory(traj, predicates, ground_atom_dataset[i])
            for i, traj in enumerate(trajectories)
        ]

    # STEP 2: Learn STRIPS operators on the given data segments as for NSRTs.
    pnads = learn_strips_operators(
            trajectories,
            train_tasks,
            predicates,
            segmented_trajs,
            verify_harmlessness=True,
            verbose=(CFG.option_learner != "no_learning"),
            annotations=annotations)

    # STEP 3: Learn options and update PNADs
    if CFG.strips_learner != "oracle" or CFG.sampler_learner != "oracle" or \
       CFG.option_learner != "no_learning":
        # Updates the endo_papads in-place.
        _learn_pnad_options(pnads, known_options, action_space)

    # STEP 4 (currently skipped): Learn samplers and update PNADs
    _learn_pnad_samplers(pnads, sampler_learner)
    breakpoint()

    # STEP 5: Convert PNADs to endogenous processes. (Maybe also make rough
    #         parameter estimates.)
    # TODO: potentially fit the endogenous-process parameters here.
    endogenous_processes = []
    for pnad in pnads:
        proc = pnad.make_endogenous_process()
        endogenous_processes.append(proc)

    # Get the segments that are not explained by the endogenous processes.
    # # STEP 2: Learn the exogenous processes.
    # ...

    # Segment again by atom_changes for inventing exogenous processes.
    # CFG.segmenter = "atom_changes"
    # segmented_trajs = [
    #     segment_trajectory(traj, predicates) for traj in trajectories
    # ]

    exogenous_processes = []
    # STEP 6: Make, log, and return the endogenous and exogenous processes.
    processes = endogenous_processes + exogenous_processes
    logging.info(f"\nLearned CausalProcesses:")
    for proc in processes:
        logging.info(proc)
    logging.info("")
    breakpoint()

    return set(processes)