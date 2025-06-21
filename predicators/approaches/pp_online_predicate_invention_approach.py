import logging
import time
from collections import defaultdict
from typing import Any, Dict, FrozenSet, Iterator, List, Optional, Sequence, \
    Set, Tuple

from gym.spaces import Box

from predicators import utils
from predicators.approaches.grammar_search_invention_approach import \
    _create_grammar, _GivenPredicateGrammar
from predicators.approaches.pp_online_process_learning_approach import \
    OnlineProcessLearningAndPlanningApproach
from predicators.approaches.pp_predicate_invention_approach import \
    PredicateInventionProcessPlanningApproach
from predicators.envs import create_new_env
from predicators.nsrt_learning.process_learning_main import \
    filter_explained_segment
from predicators.nsrt_learning.segmentation import segment_trajectory
from predicators.option_model import _OptionModelBase
from predicators.planning_with_processes import process_task_plan_grounding
from predicators.predicate_search_score_functions import \
    _PredicateSearchScoreFunction, create_score_function
from predicators.settings import CFG
from predicators.structs import Dataset, ExogenousProcess, GroundAtom, \
    GroundAtomTrajectory, InteractionResult, LowLevelTrajectory, \
    ParameterizedOption, Predicate, Segment, State, Task, Type, \
    _GroundExogenousProcess


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
        self._oracle_predicates = create_new_env(CFG.env,
                                                 use_gui=False).predicates
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
    def get_name(cls) -> str:
        return "online_predicate_invention_and_process_planning"

    def learn_from_offline_dataset(self, dataset: Dataset) -> None:
        # Just store the dataset, don't learn from it yet.
        self._offline_dataset = dataset

    def learn_from_interaction_results(
            self, results: Sequence[InteractionResult]) -> None:
        # --- Process the interaction results ---
        for result in results:
            traj = LowLevelTrajectory(result.states, result.actions)
            self._online_dataset.append(traj)

        proposed_predicates = self._get_predicate_proposals()
        logging.info(f"Done: created {len(proposed_predicates)} predicates")

        # --- Select the predicates to keep ---
        self._learned_predicates = self._select_proposed_predicates(
            ite=self._online_learning_cycle,
            all_trajs=self._online_dataset.trajectories + \
                        self._offline_dataset.trajectories,
            proposed_predicates=proposed_predicates,
            train_tasks=self._train_tasks)
        logging.debug(f"Learned predicates: "
                      f"{self._learned_predicates-self._initial_predicates}")

        # --- Learn processes & parameters ---
        self._learn_processes(
            self._offline_dataset.trajectories + \
                self._online_dataset.trajectories,
            online_learning_cycle=self._online_learning_cycle)
        # breakpoint()

        if CFG.learn_process_parameters:
            self._learn_process_parameters(self._online_dataset)

        self._online_learning_cycle += 1

    def _get_predicate_proposals(self) -> Set[Predicate]:
        if CFG.vlm_predicator_oracle_base_predicates:
            prim_predicates = self._oracle_predicates - self._initial_predicates
        else:
            # --- Invent predicates based on the dataset

            # Method 1: Find each state, if it satisfies the condition of an
            #   exogenous process, check later that its effect did take place, save
            #   it if not.
            #   Then for each exogenous process, compare the above negative state
            #   with positive states where the effect took place (e.g. in the demo).
            # Maybe this will mirror the planner.
            # Remember to reset at the end

            # Step 1: Find the false positive examples
            exogenous_processes = list(self._get_current_exogenous_processes())
            false_positive_process_state = get_false_positive_states(
                self._online_dataset.trajectories,
                self._get_current_predicates(), exogenous_processes)

            # Step 2: Find the true positive examples
            # For each expected effect that did not take place, find in the demo
            #  the initial state where it did take place, and save it as a positive
            #  example.
            true_positive_process_state = get_true_positive_process_states(
                self._get_current_predicates(), exogenous_processes,
                list(false_positive_process_state.keys()),
                self._offline_dataset.trajectories)

            # Step 3: Prompt VLM to invent predicates
            # TODO: prepare the prompt
            # TODO: implement the prompt and parse logic
            prim_predicates = self._get_proposals_from_vlm(...)
        return prim_predicates

    def _get_proposals_from_vlm(self, ) -> Set[Predicate]:
        pass

    def _select_proposed_predicates(
        self,
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
                grammar = _GivenPredicateGrammar(self.base_prim_candidates
                                                 | self._initial_predicates)
            all_candidates.update(
                grammar.generate(max_num=CFG.grammar_search_max_predicates))

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
                ite, all_candidates, score_function, self._initial_predicates)
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


def get_false_positive_states_from_seg_trajs(
    segmented_trajs: List[List[Segment]],
    exogenous_processes: List[ExogenousProcess],
) -> Dict[_GroundExogenousProcess, List[State]]:

    # Map from ground_exogenous_process to a list of init states where the
    # condition is satisfied.
    false_positive_process_state: Dict[_GroundExogenousProcess, List[State]] = \
        defaultdict(list)

    # Cache for ground_exogenous_processes to avoid recomputation
    objects_to_ground_processes = {}

    for segmented_traj in segmented_trajs:
        # Checking each segmented trajectory
        objects = frozenset(segmented_traj[0].trajectory.states[0])
        # Only recompute if objects are different
        if objects not in objects_to_ground_processes:
            ground_exogenous_processes, _ = process_task_plan_grounding(
                set(),
                objects,
                exogenous_processes,
                allow_noops=True,
                compute_reachable_atoms=False)
            objects_to_ground_processes[objects] = ground_exogenous_processes
        else:
            ground_exogenous_processes = objects_to_ground_processes[objects]

        # Pre-compute segment init_atoms for efficiency
        segment_init_atoms = [segment.init_atoms for segment in segmented_traj]

        for g_exo_process in ground_exogenous_processes:
            condition = g_exo_process.condition_at_start  # Cache reference
            add_effects = g_exo_process.add_effects
            delete_effects = g_exo_process.delete_effects

            for i, segment in enumerate(segmented_traj):
                satisfy_condition = condition.issubset(segment_init_atoms[i])
                first_state_or_prev_state_doesnt_satisfy = i == 0 or \
                    not condition.issubset(segment_init_atoms[i - 1])

                if satisfy_condition and first_state_or_prev_state_doesnt_satisfy:
                    false_positive_process_state[g_exo_process].append(
                        # segment.trajectory.states[0])
                        segment.init_atoms)

                # Check for removal condition
                if (add_effects.issubset(segment.add_effects)
                        and delete_effects.issubset(segment.delete_effects)):
                    if false_positive_process_state[g_exo_process]:
                        # TODO: we don't really know which one to remove, pop
                        # the first one is a bias.
                        false_positive_process_state[g_exo_process].pop(0)
    return false_positive_process_state


def get_false_positive_states(
    trajectories: List[LowLevelTrajectory],
    predicates: Set[Predicate],
    exogenous_processes: List[ExogenousProcess],
) -> Dict[_GroundExogenousProcess, List[State]]:
    """Get the false positive states for each exogenous process.

    Return:
        ground_exogenous_process ->
            Tuple[List[State], List[GroundAtom], List[GroundAtom]] per
            trajectory where List[State] is the list of states where the
            process is activated in the trajectory.
    """
    initial_segmenter_method = CFG.segmenter
    # TODO: use option_changes allows for creating a segment for the noop option
    # in the end, but would cause problem if the start and end of option
    # execution doesn't satisfy the condition but somewhere in the middle does
    # it. The same problem exists for the effects.
    #
    # The fix for the atom_changes segmenter would be to create a segment in
    # the end if there is still sttes after the last atom change.
    CFG.segmenter = "atom_changes"
    segmented_trajs = [
        segment_trajectory(traj, predicates, verbose=False)
        for traj in trajectories
    ]
    CFG.segmenter = initial_segmenter_method

    return get_false_positive_states_from_seg_trajs(segmented_trajs,
                                                    exogenous_processes)


def get_true_positive_process_states(
    predicates: Set[Predicate],
    exogenous_processes: List[ExogenousProcess],
    ground_exogenous_processes: List[_GroundExogenousProcess],
    trajectories: List[LowLevelTrajectory],
) -> Dict[_GroundExogenousProcess, List[State]]:
    """Get the true positive states for each exogenous process."""
    initial_segmenter_method = CFG.segmenter
    CFG.segmenter = "atom_changes"
    segmented_trajs = [
        segment_trajectory(traj, predicates) for traj in trajectories
    ]
    CFG.segmenter = initial_segmenter_method

    # Filter out segments explained by endogenous processes.
    filtered_segmented_trajs = filter_explained_segment(segmented_trajs,
                                                        exogenous_processes,
                                                        remove_options=True)
    true_positive_process_state: Dict[_GroundExogenousProcess,
                                      List[State]] = defaultdict(list)
    for g_exo_process in ground_exogenous_processes:
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
                    true_positive_process_state[g_exo_process].append(
                        segment.trajectory.states[0])
    return true_positive_process_state
