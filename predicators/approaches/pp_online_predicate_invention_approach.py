import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import dill as pkl
import numpy as np
from gym.spaces import Box
from scipy.optimize import minimize

from predicators.approaches.pp_predicate_invention_approach import \
    PredicateInventionProcessPlanningApproach
from predicators.approaches.pp_online_process_learning_approach import \
    OnlineProcessLearningAndPlanningApproach
from predicators.option_model import _OptionModelBase
from predicators.settings import CFG
from predicators.structs import Dataset, ParameterizedOption, Predicate, \
    Task, Type, InteractionResult, LowLevelTrajectory, ExogenousProcess, \
    State, GroundAtom
from predicators.nsrt_learning.process_learning_main import \
    filter_explained_segment

from predicators.planning import task_plan_grounding
from predicators.nsrt_learning.segmentation import segment_trajectory


class OnlinePredicateInventionProcessPlanningApproach(
        PredicateInventionProcessPlanningApproach,
        OnlineProcessLearningAndPlanningApproach):
    """A bilevel planning approach that invent predicates."""

    def __init__(self,
                 initial_predicates: Set[Predicate],
                 initial_options: Set[ParameterizedOption],
                 types: Set[Type],
                 action_space: Box,
                 train_tasks: List[Task],
                 task_planning_heuristic: str = "default",
                 max_skeletons_optimized: int = -1,
                 bilevel_plan_without_sim: Optional[bool] = None,
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

    @classmethod
    def get_name(cls):
        return "online_predicate_invention_and_process_planning"

    def learn_from_interaction_results(self, 
                                results: Sequence[InteractionResult]) -> None:
        for result in results:
            traj = LowLevelTrajectory(result.states, result.actions)
            self._online_dataset.append(traj)

        # Learn from the dataset
        annotations = None
        if self._online_dataset.has_annotations:
            annotations = self._online_dataset.annotations  # pragma: no cover
        
        # --- Invent predicates based on the dataset

        # Method 1: Find each state, if it satisfies the condition of an 
        #   exogenous process, check later that its effect did take place, save
        #   it if not.
        #   Then for each exogenous process, compare the above negative state
        #   with positive states where the effect took place (e.g. in the demo).
        # Maybe this will mirror the planner.
        CFG.segmenter = "option_changes"
        segmented_trajs = [
            segment_trajectory(traj, self._get_current_predicates())
            for traj in self._online_dataset.trajectories
        ]
        exogenous_processes = self._get_current_exogenous_processes()
        
        # Step 1: Find the negative examples
        # map from ground_exogenous_process to a list of tuples of 
        # init state, unrealized add/del effects.
        # TODO: currently assume each exogenous_process only happen once
        unexpected_process_dict: Dict[ExogenousProcess, 
                List[Tuple[State, List[GroundAtom], List[GroundAtom]]]
            ] = defaultdict(list)
        for segmented_traj in segmented_trajs:
            # Checking each segmented trajectory
            objects = list(segmented_traj[0].trajectory.states[0])
            ground_exogenous_processes, _ = task_plan_grounding(
                    set(), objects, exogenous_processes, 
                    allow_noops=True, compute_reachable_atoms=False)
            for g_exo_process in ground_exogenous_processes:
                g_exo_process_activate_state = []
                g_exo_process_exp_add_effects = []
                g_exo_process_exp_del_effects = []

                for i, segment in enumerate(segmented_traj):
                    satisfy_condition =\
                        g_exo_process.condition_at_start.issubset(
                            segment.init_atoms)
                    first_state_or_prev_state_doesnt_satisfy = i == 0 or\
                        not g_exo_process.condition_at_start.issubset(
                            segmented_traj[i - 1].init_atoms)
                    if satisfy_condition and \
                        first_state_or_prev_state_doesnt_satisfy:
                        g_exo_process_activate_state.append(
                            segment.trajectory.states[0])
                        g_exo_process_exp_add_effects.extend(
                            g_exo_process.add_effects)
                        g_exo_process_exp_del_effects.extend(
                            g_exo_process.delete_effects)
                    # Remove from the expected effects the ones that actually
                    # took place
                    for atom in segment.add_effects:
                        if atom in g_exo_process_exp_add_effects:
                            g_exo_process_exp_add_effects.remove(atom)
                    for atom in segment.delete_effects:
                        if atom in g_exo_process_exp_del_effects:
                            g_exo_process_exp_del_effects.remove(atom)
                if len(g_exo_process_exp_add_effects) > 0 or \
                    len(g_exo_process_exp_del_effects) > 0:
                    unexpected_process_dict[g_exo_process].append(
                        (g_exo_process_activate_state,
                            g_exo_process_exp_add_effects,
                            g_exo_process_exp_del_effects))
        
        # Step 2: Find the positive examples
        # For each expected effect that did not take place, find in the demo
        #  the initial state where it did take place, and save it as a positive
        #  example.
        CFG.segmenter = "atom_changes"
        segmented_trajs = [
            segment_trajectory(traj, self._get_current_predicates())
            for traj in self._offline_dataset.trajectories
        ]
        # Filter out segments explained by endogenous processes.
        filtered_segmented_trajs = filter_explained_segment(segmented_trajs,
                                    self._get_current_endogenous_processes(),
                                    remove_options=True)
        positive_examples: Dict[ExogenousProcess, 
                                List[State]] = defaultdict(list)
        for g_exo_process in unexpected_process_dict.keys():
            for segmented_traj in filtered_segmented_trajs:
                # Checking each segmented trajectory
                for segment in segmented_traj:
                    # Check if the segment is a positive example for any
                    # exogenous process
                    if g_exo_process.condition_at_start.issubset(
                            segment.init_atoms) and \
                        g_exo_process.add_effects.issubset(
                            segment.add_effects) and \
                        g_exo_process.delete_effects.issubset(
                            segment.delete_effects):
                        positive_examples[g_exo_process].append(
                            segment.trajectory.states[0])
                        
        # Step 3: Prompt VLM to invent predicates
        # TODO: prepare the prompt
        # TODO: implement the prompt and parse logic
        # TODO: make sure the images are labeled with object ids
        breakpoint()