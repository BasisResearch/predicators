import logging
from typing import List, Optional, Set, Any

from gym.spaces import Box

from predicators.settings import CFG
from predicators.nsrt_learning.segmentation import segment_trajectory
from predicators.structs import LowLevelTrajectory, Task, Predicate, PNAD,\
    ParameterizedOption, GroundAtomTrajectory, CausalProcess, Segment
from predicators.nsrt_learning.strips_learning import learn_strips_operators
from predicators.nsrt_learning.process_learning import learn_exogenous_processes
from predicators.nsrt_learning.nsrt_learning_main import _learn_pnad_options, \
    _learn_pnad_samplers
from predicators import utils



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
    # -- Learn the endogenous processes ---
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

    # STEP 5: Convert PNADs to endogenous processes. (Maybe also make rough
    #         parameter estimates.)
    endogenous_processes = [pnad.make_endogenous_process() for pnad in pnads]

    # --- Learn the exogenous processes. ---
    # STEP 1: Segment the trajectory by atom_changes, and filter out the ones
    #         that are explained by the endogenous processes.
    CFG.segmenter = "atom_changes"
    CFG.strips_learner = CFG.exogenous_process_learner

    segmented_trajs = [
        segment_trajectory(traj, predicates) for traj in trajectories
    ]
    # filtering out explained segments
    filtered_segmented_trajs = filter_explained_segment(segmented_trajs, pnads)
    
    # STEP 2: Learn the exogenous processes based on unexplained processes.
    #         This is different from STRIPS/endogenous processes, where these 
    #         don't have options and samplers.
    # Let's start with just the STRIPS learner for now.
    # exogenous_processes = learn_exogenous_processes()

    # TODO: remove any atoms with robot in them? Because in most cases the 
    #       robot's state shouldn't matter (there are certainly cases where it).
    exogenous_processes_pnad = learn_strips_operators(
                                trajectories,
                                train_tasks,
                                predicates,
                                filtered_segmented_trajs,
                                verify_harmlessness=False,
                                verbose=(CFG.option_learner != "no_learning"),
                                annotations=annotations)
    exogenous_processes = [pnad.make_exogenous_process() for pnad in 
                           exogenous_processes_pnad]

    # STEP 6: Make, log, and return the endogenous and exogenous processes.
    processes = endogenous_processes + exogenous_processes
    logging.info(f"\nLearned CausalProcesses:")
    for proc in processes:
        logging.info(proc)
    logging.info("")
    breakpoint()

    return set(processes)

def filter_explained_segment(segmented_trajs: List[List[Segment]],
                             pnads: List[PNAD]
                             ) -> List[List[Segment]]:
    """Filter out segments that are explained by the given PNADs."""
    logging.debug(f"Num of unfiltered segments: {len(segmented_trajs[0])}\n")
    filtered_trajs = []
    for traj in segmented_trajs:
        objects = set(traj[0].trajectory.states[0])
        filtered_segments = []
        for segment in traj:
            # TODO: is this kind of like "cover"?
            relevant_ops = [pnad.op for pnad in pnads if 
                            segment.get_option().parent == pnad.option_spec[0]]
            add_atoms = segment.add_effects
            delete_atoms = segment.delete_effects
            # if not explained by any
            if not any([add_atoms.issubset(g_op.add_effects) and 
                        delete_atoms.issubset(g_op.delete_effects) for op 
                    in relevant_ops for g_op 
                    in utils.all_ground_operators(op, objects)]):
                filtered_segments.append(segment)
        filtered_trajs.append(filtered_segments)

    logging.debug(f"Num of filtered segments: {len(filtered_trajs[0])}")
    for seg_traj in filtered_trajs:
        for i, seg in enumerate(seg_traj):
            logging.debug(f"Segment {i}: Add atoms: {seg.add_effects}; "
                          f"Delete atoms: {seg.delete_effects}; ")
    return filtered_trajs