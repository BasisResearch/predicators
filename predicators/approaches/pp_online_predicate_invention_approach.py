import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, FrozenSet,\
    Iterator, Tuple
import time

import dill as pkl
import numpy as np
from gym.spaces import Box

from predicators.approaches.pp_predicate_invention_approach import \
    PredicateInventionProcessPlanningApproach
from predicators.approaches.pp_online_process_learning_approach import \
    OnlineProcessLearningAndPlanningApproach
from predicators.option_model import _OptionModelBase
from predicators.settings import CFG
from predicators.structs import Dataset, ParameterizedOption, Predicate, \
    Task, Type, InteractionResult, LowLevelTrajectory, ExogenousProcess, \
    State, GroundAtom, GroundAtomTrajectory
from predicators.nsrt_learning.process_learning_main import \
    filter_explained_segment
from predicators.planning import task_plan_grounding
from predicators.nsrt_learning.segmentation import segment_trajectory
from predicators.approaches.grammar_search_invention_approach import \
    _create_grammar, _GivenPredicateGrammar
from predicators.envs import create_new_env
from predicators.predicate_search_score_functions import \
    _PredicateSearchScoreFunction, create_score_function
from predicators import utils

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
        # just used for oracle predicate proposal or learned predicate
        self._oracle_predicates = create_new_env(CFG.env, use_gui=False
                                                 ).predicates
        self.base_prim_candidates: Set[Predicate] = initial_predicates.copy()
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
        # remember to reset at the end
        initial_segmenter_method = CFG.segmenter
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
        CFG.segmenter = initial_segmenter_method
                        
        # Step 3: Prompt VLM to invent predicates
        # TODO: prepare the prompt
        # TODO: implement the prompt and parse logic
        proposed_predicates = self._get_predicate_proposals()
        logging.info(f"Done: created {len(proposed_predicates)} predicates")

        # Step 4: Select the predicates to keep
        self._learned_predicates = self._select_proposed_predicates(
            ite=self._online_learning_cycle,
            all_trajs=self._offline_dataset.trajectories,
            proposed_predicates=proposed_predicates,
            train_tasks=self._train_tasks)
        logging.debug(f"Learned predicates: {self._get_current_predicates()}")
        breakpoint()

        # Step 5: Learn processes & parameters
        annotations = None
        if self._online_dataset.has_annotations:
            annotations = self._online_dataset.annotations  # pragma: no cover
        self._learn_processes(
            self._online_dataset.trajectories,
            online_learning_cycle=self._online_learning_cycle,
            annotations=annotations)

        if CFG.learn_process_parameters:
            self._learn_process_parameters(self._online_dataset)

        self._online_learning_cycle += 1
    
    def _get_predicate_proposals(self) -> Set[Predicate]:
        if CFG.vlm_predicator_oracle_base_predicates:
            prim_predicates = self._oracle_predicates - self._initial_predicates
        else:
            # TODO: remove env dependency
            prim_predicates = self._get_proposals_from_vlm(...)
        return prim_predicates
    
    def _get_proposals_from_vlm(self,) -> Set[Predicate]:
        pass

    def _select_proposed_predicates(self, 
                                    ite: int,
                                    all_trajs: List[LowLevelTrajectory],
                                    proposed_predicates: Set[Predicate],
                                    train_tasks: List[Task] = None,
            ) -> Set[Predicate]:
        pass
        if CFG.vlm_predicator_oracle_learned_predicates:
            selected_preds = proposed_predicates
        else:
            self.base_prim_candidates |= proposed_predicates
            
            all_candidates: Dict[Predicate, float] = {}
            if CFG.vlm_predicator_use_grammar:
                grammar = _create_grammar(dataset=Dataset(all_trajs),
                                          given_predicates=\
                            self.base_prim_candidates|self._initial_predicates)
            else:
                grammar = _GivenPredicateGrammar(
                            self.base_prim_candidates|self._initial_predicates)
            all_candidates.update(grammar.generate(
                max_num=CFG.grammar_search_max_predicates))
            
            atom_dataset: List[GroundAtomTrajectory] =\
                        utils.create_ground_atom_dataset(all_trajs,
                                                        set(all_candidates))
            # select predicates
            logging.info("[Start] Predicate search.")
            score_function = create_score_function(
                                CFG.grammar_search_score_function,
                                self._initial_predicates,
                                atom_dataset,
                                all_candidates,
                                train_tasks,
                                current_processes=self._get_current_processes(),
                                use_processes=True)
            start_time = time.perf_counter()
            selected_preds = self._select_predicates_by_score_optimization(
                ite,
                all_candidates,
                score_function,
                self._initial_predicates)
            logging.info("[Finished] Predicate search.")
            logging.info("Total search time "
                         f"{time.perf_counter() - start_time:.2f}s")
        return selected_preds
    
    def _select_predicates_by_score_optimization(
        self,
        ite: int,
        candidates: Dict[Predicate, float],
        score_function: _PredicateSearchScoreFunction,
        initial_predicates: Set[Predicate] = set(),
    ) -> Set[Predicate]:
        """Perform a greedy search over predicate sets."""

        def _check_goal(s: FrozenSet[Predicate]) -> bool:
            del s  # unused
            return False

        # Successively consider larger predicate sets.
        def _get_successors(
            s: FrozenSet[Predicate]
        ) -> Iterator[Tuple[None, FrozenSet[Predicate], float]]:
            for predicate in sorted(set(candidates) - s):  # determinism
                # Actions not needed. Frozensets for hashing. The cost of
                # 1.0 is irrelevant because we're doing GBFS / hill
                # climbing and not A* (because we don't care about the
                # path).
                yield (None, frozenset(s | {predicate}), 1.0)

        # Start the search with no candidates.
        # Don't need to include the initial predicates here because its
        init: FrozenSet[Predicate] = frozenset(initial_predicates)

        # calculate the number of total combinations of all sizes
        num_combinations = 2**len(set(candidates))

        # Greedy local hill climbing search.
        if CFG.grammar_search_search_algorithm == "hill_climbing":
            path, _, heuristics = utils.run_hill_climbing(
                init,
                _check_goal,
                _get_successors,
                score_function.evaluate,
                enforced_depth=CFG.grammar_search_hill_climbing_depth,
                parallelize=CFG.grammar_search_parallelize_hill_climbing)
            logging.info("\nHill climbing summary:")
            for i in range(1, len(path)):
                new_additions = path[i] - path[i - 1]
                assert len(new_additions) == 1
                new_addition = next(iter(new_additions))
                h = heuristics[i]
                prev_h = heuristics[i - 1]
                logging.info(f"\tOn step {i}, added {new_addition}, with "
                             f"heuristic {h:.3f} (an improvement of "
                             f"{prev_h - h:.3f} over the previous step)")
        elif CFG.grammar_search_search_algorithm == "gbfs":
            path, _ = utils.run_gbfs(
                init,
                _check_goal,
                _get_successors,
                score_function.evaluate,
                max_evals=CFG.grammar_search_gbfs_num_evals,
            )
        else:
            raise NotImplementedError(
                "Unrecognized grammar_search_search_algorithm: "
                f"{CFG.grammar_search_search_algorithm}.")
        kept_predicates = path[-1]
        # The total number of predicate sets evaluated is just the
        # ((number of candidates selected) + 1) * total number of candidates.
        # However, since 'path' always has length one more than the
        # number of selected candidates (since it evaluates the empty
        # predicate set first), we can just compute it as below.
        assert self._metrics.get("total_num_predicate_evaluations") is None
        self._metrics["total_num_predicate_evaluations"] = len(path) * len(
            candidates)

        # # Filter out predicates that don't appear in some operator
        # # preconditions.
        # logging.info("\nFiltering out predicates that don't appear in "
        #              "preconditions...")
        # preds = kept_predicates | initial_predicates
        # pruned_atom_data = utils.prune_ground_atom_dataset(atom_dataset, preds)
        # segmented_trajs = [
        #     segment_trajectory(ll_traj, set(preds), atom_seq=atom_seq)
        #     for (ll_traj, atom_seq) in pruned_atom_data
        # ]
        # low_level_trajs = [ll_traj for ll_traj, _ in pruned_atom_data]
        # preds_in_preconds = set()
        # for pnad in learn_strips_operators(low_level_trajs,
        #                                    train_tasks,
        #                                    set(kept_predicates
        #                                        | initial_predicates),
        #                                    segmented_trajs,
        #                                    verify_harmlessness=False,
        #                                    annotations=None,
        #                                    verbose=False):
        #     for atom in pnad.op.preconditions:
        #         preds_in_preconds.add(atom.predicate)
        # kept_predicates &= preds_in_preconds

        logging.info(
            f"\n[ite {ite}] Selected {len(kept_predicates)} predicates"
            f" out of {len(candidates)} candidates:")
        for pred in kept_predicates:
            logging.info(f"\t{pred}")
        score_function.evaluate(kept_predicates)  # log useful numbers

        return set(kept_predicates)