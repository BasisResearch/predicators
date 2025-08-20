from __future__ import annotations

import heapq as hq
import logging
import os
import re
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from itertools import islice
from pprint import pformat
from typing import Callable, Collection, Dict, FrozenSet, Iterator, List, \
    Optional, Set, Tuple

import numpy as np

from predicators import utils
from predicators.planning import PlanningFailure, _MaxSkeletonsFailure, \
    _SkeletonSearchTimeout
from predicators.settings import CFG
from predicators.structs import AbstractPolicy, CausalProcess, DefaultState, \
    DerivedPredicate, DummyOption, EndogenousProcess, GroundAtom, Metrics, \
    Object, Predicate, Task, Type, _GroundCausalProcess, \
    _GroundEndogenousProcess, _GroundExogenousProcess
from predicators.utils import _TaskPlanningHeuristic


def process_task_plan_grounding(
    init_atoms: Set[GroundAtom],
    objects: Set[Object],
    nsrts: Collection[CausalProcess],
    allow_noops: bool = True,
    compute_reachable_atoms: bool = False,
) -> Tuple[List[_GroundCausalProcess], Set[GroundAtom]]:
    """Ground all operators for task planning into dummy _GroundNSRTs,
    filtering out ones that are unreachable or have empty effects.

    Also return the set of reachable atoms, which is used by task
    planning to quickly determine if a goal is unreachable.

    See the task_plan docstring for usage instructions.
    """
    ground_nsrts = []
    for nsrt in sorted(nsrts):
        for ground_nsrt in utils.all_ground_nsrts(nsrt, objects):
            if allow_noops or (ground_nsrt.add_effects
                               | ground_nsrt.delete_effects):
                ground_nsrts.append(ground_nsrt)
    if compute_reachable_atoms:
        reachable_atoms = utils.get_reachable_atoms(ground_nsrts, init_atoms)
    else:
        reachable_atoms = set()

    reachable_nsrts = ground_nsrts
    return reachable_nsrts, reachable_atoms


@dataclass(repr=False, eq=False)
class _ProcessPlanningNode():
    """
    Args:
        state_history: a finegrained, per-step history of the state trajectory
            compared to atoms_sequence which is segmented by action.
        action_history: a finegrained, per-step history of the action trajectory
            compared to skeleton which is segmented by action.
    """
    atoms: Set[GroundAtom]  # per big step state
    skeleton: List[_GroundEndogenousProcess]  # per big step action
    atoms_sequence: List[Set[GroundAtom]]  # expected state sequence
    parent: Optional[_ProcessPlanningNode]
    cumulative_cost: float
    state_history: List[Set[GroundAtom]]  # per small step state
    action_history: List[
        Optional[_GroundEndogenousProcess]]  # per small step action
    scheduled_events: Dict[int, List[Tuple[_GroundCausalProcess, int]]]


class ProcessWorldModel:

    def __init__(
        self,
        ground_processes: List[_GroundCausalProcess],
        state: Set[GroundAtom],
        state_history: List[Set[GroundAtom]] = [],
        action_history: List[Optional[_GroundEndogenousProcess]] = [],
        scheduled_events: Dict[int, List[Tuple[_GroundCausalProcess,
                                               int]]] = {},
        t: int = 0,
        derived_predicates: Set[DerivedPredicate] = set(),
        objects: Set[Object] = set()
    ) -> None:

        self.ground_processes = ground_processes
        self.state = state
        # if len(state_history) == 0:
        #     state_history.append(state)
        self.state_history = state_history
        self.current_action: Optional[_GroundEndogenousProcess] = None
        self.action_history = action_history
        self.scheduled_events: Dict[int, List[Tuple[_GroundCausalProcess,
                                                    int]]] = scheduled_events
        self.t = t
        self.derived_predicates = derived_predicates
        self.objects = objects

    def small_step(
            self,
            small_step_action: Optional[_GroundEndogenousProcess] = None
    ) -> None:
        """Will keep the current action as a class variable for now, as opposed
        to a part of the state variable as in the demo code."""
        initial_state = self.state.copy()

        # 1. self.current_action is set to an action when this small_step is
        # first called. And is set back to None when `duration` timesteps
        # sampled from its distribution passes.
        # `small_step_action` is not None in the first call but becomes None in
        # subsequent calls.
        if small_step_action is not None:
            self.current_action = small_step_action.copy()
            # logging.debug(f"At time {self.t}, start performing "
            #               f"{self.current_action.name}")
        self.action_history.append(self.current_action.copy() if self.
                                   current_action is not None else None)

        # 2. Process effects scheduled for this timestep.
        if self.t in self.scheduled_events:
            for g_process, start_time in self.scheduled_events[self.t]:
                # If it's the end of an endogenous process (an action), then
                # change self.current_action back to None.
                if (all(
                        g_process.condition_overall.issubset(s)
                        for s in self.state_history[start_time + 1:])
                        and g_process.condition_at_end.issubset(self.state)):
                    # logging.debug(f"At time {self.t}:")
                    for atom in g_process.delete_effects:
                        self.state.discard(atom)
                        # logging.debug(f"Deleting {atom}")
                    for atom in g_process.add_effects:
                        self.state.add(atom)
                        # logging.debug(f"Adding   {atom}")
                    # The second clause seems redundant because small_step_action
                    # is always None in the subsequent steps of small_step, i.e.,
                    # which is at least 1 timestep after the process is scheduled.
                    if isinstance(g_process, _GroundEndogenousProcess) and\
                        small_step_action is None:
                        self.current_action = None
            del self.scheduled_events[self.t]

            # Remove all the previous derived predicates before adding new
            # ones.
            if len(self.derived_predicates) > 0:
                self.state = {
                    atom
                    for atom in self.state
                    if not isinstance(atom.predicate, DerivedPredicate)
                }
                # for atom in self.state:
                #     if atom.predicate.name == "JugFilled":
                #         # it has a tuple of types instead of a list..
                #         breakpoint()
                #         break
                self.state |= utils.abstract_with_derived_predicates(
                    self.state, self.derived_predicates, self.objects)

        # 3. Schedule new events whose condition are met
        for g_process in self.ground_processes:
            satisfy_condition_at_start = g_process.condition_at_start.issubset(
                self.state)
            # Only schedule when it's previously unsatisfied to avoid repeated
            # scheduling.
            first_state_or_prev_state_doesnt_satisfy = (
                len(self.state_history) == 0
                or not g_process.condition_at_start.issubset(
                    self.state_history[-1]))
            is_exogenous = isinstance(g_process, _GroundExogenousProcess)
            # if is_exogenous:
            #     logging.debug(f"process {g_process} ")
            #     breakpoint()
            # Action. Here we shouldn't require it was previous unsatisfied.
            is_endogenous = isinstance(g_process, _GroundEndogenousProcess)
            first_step_running_action = small_step_action is not None and \
                                        g_process == small_step_action
            if is_endogenous:
                not_noop = g_process.parent.option.name != 'NoOp'
            # logging.debug(f"Condition at start: {satisfy_condition_at_start} "
            #               f"no prev state or prev doesnt satisfy: {no_prev_state_or_prev_doesnt_satisfy} "
            #               f"Is endogenous: {is_endogenous} "
            #               f"Is exogenous: {is_exogenous} "
            #               f"First step running action: {first_step_running_action}")
            if (satisfy_condition_at_start and
                ((is_exogenous and first_state_or_prev_state_doesnt_satisfy) or
                 (is_endogenous and first_step_running_action and not_noop))):
                delay = g_process.delay_distribution.sample()
                delay = max(1, delay)  # Ensure delay is at least 1.
                schedued_time = self.t + delay
                # logging.debug(f"At time {self.t}, scheduling "
                #               f"{g_process.name_and_objects_str()} "
                #               f"for time {schedued_time}")
                if schedued_time not in self.scheduled_events:
                    self.scheduled_events[schedued_time] = []
                self.scheduled_events[schedued_time].append(
                    (g_process, self.t))
                # logging.debug(f"current scheduled_events: {self.scheduled_events.keys()}")

        self.state_history.append(self.state.copy())

        # if the action has finished and set to None.
        if self.current_action is None:
            return
        self.t += 1

    def big_step(self,
                 action_process: _GroundEndogenousProcess,
                 max_num_steps: int = 50) -> Set[GroundAtom]:
        """current_action is set to an action in the first call to small_step
        and is set to None when 1) the action reaches the end of its duration
        2) some aspects of the state changes; removing this because this can
        cause action to stop before the end of its duration 3) reaches
        max_num_steps."""
        initial_state = self.state.copy()
        num_steps = 0
        action_not_finished = True

        while action_not_finished and num_steps < max_num_steps:
            self.small_step(action_process)
            num_steps += 1

            if action_process is not None:
                action_process = None

            action_not_finished = self.current_action is not None

            # if currently executing NoOp and state has changed, then break
            if self.current_action is not None and \
                self.current_action.parent.option.name == 'NoOp' and \
                self.state != initial_state:
                break
        return self.state


def _skeleton_generator_with_processes(
    task: Task,
    ground_processes: List[_GroundCausalProcess],
    init_atoms: Set[GroundAtom],
    heuristic: _TaskPlanningHeuristic,
    seed: int,
    timeout: float,
    metrics: Metrics,
    max_skeletons_optimized: int,
    abstract_policy: Optional[AbstractPolicy] = None,
    sesame_max_policy_guided_rollout: int = 0,
    use_visited_state_set: bool = False,
    log_sucessful_small_steps: bool = False,
    log_heuristic: bool = False,
    time_heuristic: bool = True,
    derived_predicates: Set[DerivedPredicate] = set(),
    objects: Set[Object] = set(),
) -> Iterator[Tuple[List[_GroundEndogenousProcess], List[Set[GroundAtom]]]]:

    # Filter out all the action from processes
    # zero heuristic
    objects = objects.copy()
    ground_action_processes = [
        p for p in ground_processes if isinstance(p, _GroundEndogenousProcess)
    ]
    start_time = time.perf_counter()
    queue: List[Tuple[float, float, _ProcessPlanningNode]] = []
    root_node = _ProcessPlanningNode(
        atoms=init_atoms,
        skeleton=[],
        atoms_sequence=[init_atoms],
        parent=None,
        cumulative_cost=0,
        state_history=[],
        action_history=[],
        scheduled_events={},
    )
    metrics["num_nodes_created"] += 1
    rng_prio = np.random.default_rng(seed)
    if time_heuristic:
        heuristic_call_count = 0
        total_heuristic_time = 0.0
        heuristic_start_time = time.perf_counter()
        h = heuristic(root_node.atoms)
        heuristic_end_time = time.perf_counter()
        heuristic_call_count += 1
        total_heuristic_time += (heuristic_end_time - heuristic_start_time)
    else:
        h = heuristic(root_node.atoms)
    if log_heuristic:
        logging.debug(f"Root heuristic: {h}")
    hq.heappush(queue, (h, rng_prio.uniform(), root_node))
    # Initialize with empty skeleton for root.
    # We want to keep track of the visited skeletons so that we avoid
    # repeatedly outputting the same faulty skeletons.
    visited_skeletons: Set[Tuple[_GroundCausalProcess, ...]] = set()
    visited_skeletons.add(tuple(root_node.skeleton))
    if use_visited_state_set:
        # This set will maintain (frozen) atom sets that have been fully
        # expanded already, and ensure that we never expand redundantly.
        visited_atom_sets = set()
    # Start search.
    while queue and (time.perf_counter() - start_time < timeout):
        if int(metrics["num_skeletons_optimized"]) == max_skeletons_optimized:
            raise _MaxSkeletonsFailure(
                "Planning reached max_skeletons_optimized!")
        _, _, node = hq.heappop(queue)
        if use_visited_state_set:
            frozen_atoms = frozenset(node.atoms)
            visited_atom_sets.add(frozen_atoms)
        # Good debug point #1: print out the skeleton here to see what
        # the high-level search is doing. You can accomplish this via:
        # for act in node.skeleton:
        #     logging.info(f"{act.name} {act.objects}")
        # logging.info("")
        if task.goal.issubset(node.atoms):
            # If this skeleton satisfies the goal, yield it.
            metrics["num_skeletons_optimized"] += 1
            time_taken = time.perf_counter() - start_time
            logging.debug(f"\n[Task Planner] Found Plan of length "
                          f"{len(node.skeleton)} in {time_taken:.2f}s:")
            for process in node.skeleton:
                logging.debug(process.name_and_objects_str())
            logging.debug("")

            if log_sucessful_small_steps:
                prev_state = None
                for i, (state, action) in enumerate(
                        zip(node.state_history, node.action_history)):
                    if prev_state is None:
                        logging.debug(f"State {i}: {sorted(state)}")
                    else:
                        logging.debug(
                            f"State {i}: "
                            f"Add atoms: {sorted(state - prev_state)} "
                            f"Del atoms: {sorted(prev_state - state)}")
                    action_str = action.name_and_objects_str() \
                                    if action is not None else None
                    logging.info(f"Action {i}: {action_str}\n")
                    prev_state = state
                logging.debug(
                    f"State {len(node.state_history)}: "
                    f"Add atoms: {sorted(node.state_history[-1] - prev_state)} "
                    f"Del atoms: {sorted(prev_state - node.state_history[-1])}"
                )

            # Log heuristic timing stats when a solution is found
            if time_heuristic:
                average_heuristic_time = total_heuristic_time / heuristic_call_count if heuristic_call_count > 0 else 0.0
                logging.info(
                    f"Heuristic timing stats - Calls: {heuristic_call_count}, Total time: {total_heuristic_time:.4f}s, Average time: {average_heuristic_time:.4f}s"
                )

            yield node.skeleton, node.atoms_sequence
        else:
            # Generate successors.
            metrics["num_nodes_expanded"] += 1
            # Skip abstract policy support...
            applicable_actions = list(
                utils.get_applicable_operators(ground_action_processes,
                                               node.atoms))

            # Domain-specific pruning for domino environment
            if CFG.env == "pybullet_domino" and CFG.domino_prune_actions:
                # Filter out backwards placements and redundant picks
                filtered_actions = []
                placed_dominos = set()  # Track which dominos have been placed

                # First pass: identify already placed dominos
                for prev_action in node.skeleton:
                    if prev_action.parent.name == "PlaceDomino":
                        # The domino being placed is the second argument
                        if len(prev_action.objects) > 1:
                            placed_dominos.add(prev_action.objects[1])

                for action in applicable_actions:
                    # Always keep NoOp and Push actions
                    if action.parent.name in ["NoOp", "PushStartBlock"]:
                        filtered_actions.append(action)
                    # For Pick, only pick dominos that haven't been placed yet
                    elif action.parent.name == "PickDomino":
                        domino_to_pick = action.objects[1] if len(
                            action.objects) > 1 else None
                        if domino_to_pick and domino_to_pick not in placed_dominos:
                            filtered_actions.append(action)
                    # For Place, apply heuristics
                    elif action.parent.name == "PlaceDomino":
                        # Keep all place actions for now, but could add more pruning
                        # E.g., only place in forward direction, avoid cycles, etc.
                        filtered_actions.append(action)
                    else:
                        filtered_actions.append(action)

                # If pruning removed all actions, fall back to unpruned
                if filtered_actions:
                    applicable_actions = filtered_actions

            for action_process in applicable_actions:

                # --- Run the action process on the world model
                world_model = ProcessWorldModel(
                    ground_processes=ground_processes.copy(),
                    state=node.atoms.copy(),
                    state_history=node.state_history.copy(),
                    action_history=node.action_history.copy(),
                    scheduled_events=deepcopy(node.scheduled_events),
                    t=len(node.state_history),
                    derived_predicates=derived_predicates,
                    objects=objects)

                assert isinstance(action_process, _GroundEndogenousProcess)
                # plan_so_far = [p.name for p in node.skeleton]
                # plan_so_far = [p.name_and_objects_str() for p in node.skeleton]
                # logging.debug(f"Expand after plan {plan_so_far}:")
                # applicable_actions = list(utils.get_applicable_operators(
                #     ground_action_processes, node.atoms))
                # num_applicable_actions = len(applicable_actions)
                # logging.debug(f"Num applicable actions: {num_applicable_actions}")
                # logging.debug(f"Taking action: {action_process.name_and_objects_str()}")
                # action_names = [p.name_and_objects_str() for p in node.skeleton]
                # # action_names = [p.name for p in node.skeleton]
                # # target_action_names = ['PickJugFromOutsideFaucetAndBurner',
                # #                        'PlaceUnderFaucet',
                # #                        'SwitchFaucetOn',
                # #                        'SwitchBurnerOn',
                # #                        'SwitchFaucetOff',
                # #                        'PickJugFromFaucet',
                # #                        'PlaceOnBurner',
                # #                        'PickJugFromOutsideFaucetAndBurner',
                # #                        'PlaceUnderFaucet',
                # #                        'SwitchFaucetOn',
                # #                        'SwitchBurnerOn',
                # #                        ]
                # target_action_names = [
                #                     'PickDomino(robot:robot, domino_2:domino, pos_y0_x2:loc, rot_0:rot)',
                #                     # 'PlaceDomino(robot:robot, domino_2:domino, domino_3:domino, pos_y0_x2:loc, rot_135:rot)',
                #                     # 'PickDomino(robot:robot, domino_1:domino, pos_y0_x0:loc, rot_0:rot)',
                #                     # 'PlaceDomino(robot:robot, domino_1:domino, domino_0:domino, pos_y1_x2:loc, rot_180:rot',
                #                     ]
                # if action_names == target_action_names and \
                #     action_process.name_and_objects_str() == 'PlaceDomino(robot:robot, domino_2:domino, domino_3:domino, pos_y0_x2:loc, rot_135:rot)':
                # # if action_names == target_action_names:
                #     breakpoint()
                world_model.big_step(action_process)
                child_atoms = world_model.state.copy()
                # --- End

                # Same as standard skeleton generator
                if use_visited_state_set:
                    frozen_atoms = frozenset(child_atoms)
                    if frozen_atoms in visited_atom_sets:
                        continue
                child_skeleton = node.skeleton + [action_process]
                child_skeleton_tup = tuple(child_skeleton)
                if child_skeleton_tup in visited_skeletons:  # pragma: no cover
                    continue
                visited_skeletons.add(child_skeleton_tup)
                # Action costs are unitary.
                if action_process.option.name == 'NoOp':
                    action_cost = 0.5
                else:
                    action_cost = 1.0
                child_cost = node.cumulative_cost + action_cost
                child_node = _ProcessPlanningNode(
                    atoms=child_atoms,
                    skeleton=child_skeleton.copy(),
                    atoms_sequence=node.atoms_sequence + [child_atoms],
                    parent=node,
                    cumulative_cost=child_cost,
                    state_history=world_model.state_history.copy(),
                    action_history=world_model.action_history.copy(),
                    scheduled_events=deepcopy(world_model.scheduled_events))
                metrics["num_nodes_created"] += 1
                # priority is g [cost] plus h [heuristic]
                if time_heuristic:
                    heuristic_start_time = time.perf_counter()
                    h = heuristic(child_node.atoms)
                    heuristic_end_time = time.perf_counter()
                    heuristic_call_count += 1
                    total_heuristic_time += (heuristic_end_time -
                                             heuristic_start_time)
                else:
                    h = heuristic(child_node.atoms)
                priority = (child_node.cumulative_cost + h)
                if log_heuristic:
                    logging.debug(
                        f"Heuristic: {h}, g: {child_node.cumulative_cost}")
                hq.heappush(queue, (priority, rng_prio.uniform(), child_node))
                if time.perf_counter() - start_time >= timeout:
                    break
    if time_heuristic:
        average_heuristic_time = total_heuristic_time / heuristic_call_count if heuristic_call_count > 0 else 0.0
        logging.info(
            f"Heuristic timing stats - Calls: {heuristic_call_count}, "
            f"Total time: {total_heuristic_time:.4f}s, "
            f"Average time: {average_heuristic_time:.4f}s, "
            f"Num_nodes_created: {metrics['num_nodes_created']}, "
            f"Num_nodes_expanded: {metrics['num_nodes_expanded']}")

    if not queue:
        raise _MaxSkeletonsFailure("Planning ran out of skeletons!")
    assert time.perf_counter() - start_time >= timeout
    raise _SkeletonSearchTimeout


def task_plan_from_task(
    task: Task,
    predicates: Set[Predicate],
    processes: Set[CausalProcess],
    seed: int,
    timeout: float,
    max_skeletons_optimized: int,
    use_visited_state_set: bool = True,
) -> Iterator[Tuple[List[_GroundEndogenousProcess], List[Set[GroundAtom]],
                    Metrics]]:
    # TODO: Expand the concept predicates to include all dependencies
    if isinstance(predicates, FrozenSet):
        predicates = set(predicates)
    assert isinstance(predicates, set), \
        f"Expected predicates to be a set, got {type(predicates)}"
    all_predicates = utils.add_in_auxiliary_predicates(predicates)
    derived_predicates = utils.get_derived_predicates(all_predicates)

    init_atoms = utils.abstract(task.init, all_predicates)
    logging.debug("[Task Planner] Task init atoms: "
                  f"{pformat(sorted(init_atoms))}")
    goal = task.goal
    objects = set(task.init)
    ground_processes, reachable_atoms = process_task_plan_grounding(
        init_atoms,
        objects,
        processes,
        allow_noops=True,
        compute_reachable_atoms=False)

    use_derived_predicates = True
    if CFG.sesame_task_planning_heuristic == "goal_count":
        heuristic = utils.create_task_planning_heuristic(
            CFG.sesame_task_planning_heuristic, init_atoms, goal,
            ground_processes, all_predicates, objects)
    elif CFG.sesame_task_planning_heuristic == "lm_cut":
        heuristic = create_lm_cut_heuristic(
            goal,
            ground_processes,
            derived_predicates,
            objects,
            use_derived_predicates=use_derived_predicates)
    elif CFG.sesame_task_planning_heuristic == "h_max":
        heuristic = create_h_max_heuristic(
            goal,
            ground_processes,
            derived_predicates,
            objects,
            use_derived_predicates=use_derived_predicates)

    elif CFG.sesame_task_planning_heuristic == "h_ff":
        heuristic = create_ff_heuristic(
            goal,
            ground_processes,
            derived_predicates,
            objects,
            use_derived_predicates=use_derived_predicates)
    else:
        raise ValueError(
            f"Unrecognized sesame_task_planning_heuristic: {CFG.sesame_task_planning_heuristic}"
        )

    return task_plan(
        init_atoms,
        goal,
        ground_processes,
        reachable_atoms,
        heuristic,
        seed,
        timeout,
        max_skeletons_optimized,
        use_visited_state_set=use_visited_state_set,
        derived_predicates=derived_predicates,
        objects=objects,
    )


def task_plan(
    init_atoms: Set[GroundAtom],
    goal: Set[GroundAtom],
    ground_processes: List[_GroundCausalProcess],
    reachable_atoms: Set[GroundAtom],
    heuristic: _TaskPlanningHeuristic,
    seed: int,
    timeout: float,
    max_skeletons_optimized: int,
    use_visited_state_set: bool = True,
    derived_predicates: Set[DerivedPredicate] = set(),
    objects: Set[Object] = set(),
) -> Iterator[Tuple[List[_GroundEndogenousProcess], List[Set[GroundAtom]],
                    Metrics]]:
    """Run only the task planning portion of SeSamE. A* search is run, and
    skeletons that achieve the goal symbolically are yielded. Specifically,
    yields a tuple of (skeleton, atoms sequence, metrics dictionary).

    This method is NOT used by SeSamE, but is instead provided as a
    convenient wrapper around _skeleton_generator below (which IS used
    by SeSamE) that takes in only the minimal necessary arguments.

    This method is tightly coupled with task_plan_grounding -- the reason they
    are separate methods is that it is sometimes possible to ground only once
    and then plan multiple times (e.g. from different initial states, or to
    different goals). To run task planning once, call task_plan_grounding to
    get ground_nsrts and reachable_atoms; then create a heuristic using
    utils.create_task_planning_heuristic; then call this method. See the tests
    in tests/test_planning for usage examples.
    """
    if CFG.planning_check_dr_reachable and not goal.issubset(reachable_atoms):
        logging.info(f"Detected goal unreachable. Goal: {goal}")
        logging.info(f"Initial atoms: {init_atoms}")
        raise PlanningFailure(f"Goal {goal} not dr-reachable")
    dummy_task = Task(DefaultState, goal)
    metrics: Metrics = defaultdict(float)
    # logging.debug(f"init_atoms: {init_atoms}")
    generator = _skeleton_generator_with_processes(
        dummy_task,
        ground_processes,
        init_atoms,
        heuristic,
        seed,
        timeout,
        metrics,
        max_skeletons_optimized,
        use_visited_state_set=use_visited_state_set,
        derived_predicates=derived_predicates,
        objects=objects,
    )

    # Note that we use this pattern to avoid having to catch an exception
    # when _skeleton_generator runs out of skeletons to optimize.
    for skeleton, atoms_sequence in islice(generator, max_skeletons_optimized):
        yield skeleton, atoms_sequence, metrics.copy()


def run_task_plan_with_processes_once(
    task: Task,
    processes: Set[CausalProcess],
    preds: Set[Predicate],
    types: Set[Type],
    timeout: float,
    seed: int,
    task_planning_heuristic: str,
    max_horizon: float = np.inf,
    compute_reachable_atoms: bool = False,
) -> Tuple[List[_GroundEndogenousProcess], List[Set[GroundAtom]], Metrics]:
    """Get a single abstract plan for a task.

    The sequence of ground atom sets returned represent NECESSARY atoms.
    """

    start_time = time.perf_counter()

    if CFG.sesame_task_planner == "astar":
        duration = time.perf_counter() - start_time
        timeout -= duration
        plan, atoms_seq, metrics = next(
            task_plan_from_task(
                task,
                preds,
                processes,
                seed,
                timeout,
                max_skeletons_optimized=1,
            ))
        if len(plan) > max_horizon:
            raise PlanningFailure(
                "Skeleton produced by A-star exceeds horizon!")
    else:
        raise ValueError("Unrecognized sesame_task_planner: "
                         f"{CFG.sesame_task_planner}")

    # comment out for now
    # necessary_atoms_seq = utils.compute_necessary_atoms_seq(
    #     plan, atoms_seq, goal)
    necessary_atoms_seq: List[Set[GroundAtom]] = []

    return plan, necessary_atoms_seq, metrics


def create_ff_heuristic(
    goal: Set[GroundAtom],
    ground_processes: List[_GroundCausalProcess],
    derived_predicates: Set[DerivedPredicate] = set(),
    objects: Set[Object] = set(),
    use_derived_predicates: bool = True,
    debug_log: bool = False,
) -> Callable[[Set[GroundAtom]], float]:
    """Creates a callable FF heuristic function with efficient RPG
    generation."""

    adds_map: Dict[GroundAtom, List[_GroundCausalProcess]] = defaultdict(list)
    for process in ground_processes:
        for atom in process.add_effects:
            adds_map[atom].append(process)

    # --- CHANGE START: Use pre-computation for the shared function ---
    dep_to_derived_preds: Dict[Predicate,
                               List[DerivedPredicate]] = defaultdict(list)
    if use_derived_predicates:
        for der_pred in derived_predicates:
            for aux_pred in der_pred.auxiliary_predicates:
                dep_to_derived_preds[aux_pred].append(der_pred)
    # --- CHANGE END ---

    def _ff_heuristic(atoms: Set[GroundAtom]) -> float:
        """The FF heuristic using incremental RPG generation."""
        if goal.issubset(atoms):
            return 0.0

        # --- 1. Build the Relaxed Planning Graph (RPG) ---
        initial_facts = atoms.copy()
        if use_derived_predicates:
            # The first layer must be a full, non-incremental computation.
            initial_facts.update(
                utils.abstract_with_derived_predicates(initial_facts,
                                                       derived_predicates,
                                                       objects))

        fact_layers: List[Set[GroundAtom]] = [initial_facts]
        process_layers: List[Set[_GroundCausalProcess]] = []

        if debug_log:
            count = 0
            logging.debug(f"Initial facts: {sorted(initial_facts)}")
        while not goal.issubset(fact_layers[-1]):
            if debug_log:
                logging.debug(f"Calculating heuristic layer {count}...")
                count += 1
            current_facts = fact_layers[-1]

            # Find all processes whose preconditions are met in the current layer.
            applicable_processes: Set[_GroundCausalProcess] = set()
            for process in ground_processes:
                if process.condition_at_start.issubset(current_facts):
                    applicable_processes.add(process)

            process_layers.append(applicable_processes)

            # --- Incremental Fact Generation ---
            # a) Collect all new primitive facts from applicable processes.
            primitive_add_effects = set()
            for process in applicable_processes:
                primitive_add_effects.update(process.add_effects)

            newly_added_primitive_facts = primitive_add_effects - current_facts
            if debug_log:
                logging.debug(
                    f"Newly added primitive facts: {sorted(newly_added_primitive_facts)}"
                )

            # b) Incrementally compute new derived facts.
            newly_derived_facts = set()
            if use_derived_predicates:
                # --- CHANGE START: Call the shared function ---
                newly_derived_facts = _run_incremental_derived_predicate_logic(
                    newly_added_primitive_facts,
                    current_facts,
                    objects,
                    dep_to_derived_preds,
                )
                # --- CHANGE END ---
                if debug_log:
                    logging.debug(
                        f"Newly derived facts: {sorted(newly_derived_facts)}\n"
                    )

            next_facts = current_facts | newly_added_primitive_facts | newly_derived_facts

            # If the new layer is identical to the old one, we've stagnated.
            if next_facts == current_facts:
                return float('inf')

            fact_layers.append(next_facts)

        # --- 2. Extract a Relaxed Plan (Backward Search through the RPG) ---
        relaxed_plan_actions: Set[_GroundEndogenousProcess] = set()
        subgoals_to_achieve = goal.copy()

        for i in range(len(fact_layers) - 1, 0, -1):
            unachieved_subgoals = subgoals_to_achieve.copy()
            for subgoal in unachieved_subgoals:
                # If the subgoal appeared for the first time in this layer...
                if subgoal in fact_layers[i] and subgoal not in fact_layers[i -
                                                                            1]:
                    best_supporter = None
                    # Find a process from the previous layer that achieves it.
                    for process in adds_map.get(subgoal, []):
                        if process in process_layers[i - 1]:
                            best_supporter = process
                            break

                    if best_supporter:
                        # Only agent actions (endogenous) contribute to the plan cost.
                        if isinstance(best_supporter,
                                      _GroundEndogenousProcess):
                            relaxed_plan_actions.add(best_supporter)

                        # Add the supporter's preconditions to our set of subgoals.
                        subgoals_to_achieve.update(
                            best_supporter.condition_at_start)
                        subgoals_to_achieve.discard(subgoal)

        return float(len(relaxed_plan_actions))

    return _ff_heuristic


def create_lm_cut_heuristic(
    goal: Set[GroundAtom],
    ground_processes: List[_GroundCausalProcess],
    derived_predicates: Set[DerivedPredicate] = set(),
    objects: Set[Object] = set(),
    use_derived_predicates: bool = True,
) -> Callable[[Set[GroundAtom]], float]:
    """Creates a callable LM-cut heuristic function.

    This heuristic iteratively finds landmarks by computing a relaxed
    plan, calculating its cost, and then assuming its effects have been
    achieved before solving for the next landmark. This is a practical
    implementation of the LM-cut principle. It also correctly handles
    exogenous processes and derived predicates (axioms) as zero-cost
    events.
    """

    # --- Pre-computation to speed up sub-problems ---
    adds_map: Dict[GroundAtom, List[_GroundCausalProcess]] = defaultdict(list)
    for process in ground_processes:
        for atom in process.add_effects:
            adds_map[atom].append(process)

    # --- CHANGE START: Use pre-computation for the shared function ---
    dep_to_derived_preds: Dict[Predicate,
                               List[DerivedPredicate]] = defaultdict(list)
    if use_derived_predicates:
        for der_pred in derived_predicates:
            for aux_pred in der_pred.auxiliary_predicates:
                dep_to_derived_preds[aux_pred].append(der_pred)
    # --- CHANGE END ---

    def _calculate_relaxed_plan(
        current_atoms: Set[GroundAtom], current_goal: Set[GroundAtom]
    ) -> Tuple[float, Set[_GroundCausalProcess]]:
        """Helper that computes one relaxed plan (our landmark) from a given
        state."""
        initial_facts = current_atoms.copy()
        if use_derived_predicates:
            initial_facts.update(
                utils.abstract_with_derived_predicates(initial_facts,
                                                       derived_predicates,
                                                       objects))

        if current_goal.issubset(initial_facts):
            return 0.0, set()

        fact_layers: List[Set[GroundAtom]] = [initial_facts]
        process_layers: List[Set[_GroundCausalProcess]] = []

        while not current_goal.issubset(fact_layers[-1]):
            current_facts = fact_layers[-1]

            applicable_processes: Set[_GroundCausalProcess] = set()
            for process in ground_processes:
                if process.condition_at_start.issubset(current_facts):
                    applicable_processes.add(process)

            process_layers.append(applicable_processes)

            primitive_add_effects = set()
            for process in applicable_processes:
                primitive_add_effects.update(process.add_effects)
            newly_added_primitive_facts = primitive_add_effects - current_facts

            newly_derived_facts = set()
            if use_derived_predicates:
                # --- CHANGE START: Call the shared function ---
                newly_derived_facts = _run_incremental_derived_predicate_logic(
                    newly_added_primitive_facts,
                    current_facts,
                    objects,
                    dep_to_derived_preds,
                )
                # --- CHANGE END ---

            next_facts = current_facts | newly_added_primitive_facts | newly_derived_facts

            if next_facts == current_facts:
                return float('inf'), set()

            fact_layers.append(next_facts)

        # 2. Extract one relaxed plan via backward search.
        relaxed_plan: Set[_GroundCausalProcess] = set()
        subgoals_to_achieve = current_goal.copy()

        for i in range(len(fact_layers) - 1, 0, -1):
            for subgoal in subgoals_to_achieve.copy():
                if subgoal in fact_layers[i] and subgoal not in fact_layers[i -
                                                                            1]:
                    best_supporter = None
                    for process in adds_map.get(subgoal, []):
                        if process in process_layers[i - 1]:
                            best_supporter = process
                            break

                    if best_supporter:
                        relaxed_plan.add(best_supporter)
                        subgoals_to_achieve.update(
                            best_supporter.condition_at_start)
                        subgoals_to_achieve.discard(subgoal)

        # 3. Calculate the cost of the relaxed plan.
        cost = 0.0
        for process in relaxed_plan:
            # Endogenous processes (agent actions) have a cost.
            if isinstance(process, _GroundEndogenousProcess):
                # Use axiom_cost if it's a derived predicate axiom, otherwise default to 1.
                cost += getattr(process, 'axiom_cost', 1.0)

        return cost, relaxed_plan

    def _lm_cut_heuristic(atoms: Set[GroundAtom]) -> float:
        """The main heuristic function.

        It iteratively calls the relaxed plan solver to find and sum the
        costs of landmarks.
        """
        total_cost = 0.0
        current_atoms = atoms.copy()

        # Loop until the goal is satisfied in our simulated state.
        while not goal.issubset(current_atoms):
            # Find the cost and plan for the next landmark.
            landmark_cost, landmark_plan = _calculate_relaxed_plan(
                current_atoms, goal)

            # If a landmark is infinitely costly, the goal is unreachable.
            if landmark_cost == float('inf'):
                return float('inf')

            # If we found a plan with no cost (e.g., only free events),
            # but haven't reached the goal, we must force progress by adding
            # at least one real action. A cost of 1 is the minimum.
            if landmark_cost == 0.0:
                total_cost += 1.0

            total_cost += landmark_cost

            # "Apply" the landmark by adding the effects of its plan to our state.
            if not landmark_plan:
                # Should not be reachable if cost is not inf, but as a safeguard...
                return float('inf')

            for process in landmark_plan:
                current_atoms.update(process.add_effects)

        return total_cost

    return _lm_cut_heuristic


def create_h_max_heuristic(
    goal: Set[GroundAtom],
    ground_processes: List[_GroundCausalProcess],
    derived_predicates: Set[DerivedPredicate] = set(),
    objects: Set[Object] = set(),
    use_derived_predicates: bool = True,
) -> Callable[[Set[GroundAtom]], float]:
    """Creates a callable h_max heuristic function.

    This heuristic is compatible with exogenous processes (zero-cost)
    and derived predicates (zero-cost). It works by building a Relaxed
    Planning Graph (RPG) and finding the maximum cost to achieve any
    single atom in the goal set. The cost of an atom is the cost of the
    cheapest process that achieves it, where the cost of a process is
    the maximum cost of any of its preconditions plus its own cost (1
    for actions, 0 otherwise).
    """

    # Pre-computation for derived predicate dependencies.
    dep_to_derived_preds: Dict[Predicate,
                               List[DerivedPredicate]] = defaultdict(list)
    if use_derived_predicates:
        for der_pred in derived_predicates:
            for aux_pred in der_pred.auxiliary_predicates:
                dep_to_derived_preds[aux_pred].append(der_pred)

    def _h_max_heuristic(atoms: Set[GroundAtom]) -> float:
        """The h_max heuristic function."""
        if goal.issubset(atoms):
            return 0.0

        # Initialize costs: 0 for initial atoms, infinity otherwise.
        atom_costs = defaultdict(lambda: float('inf'))
        for atom in atoms:
            atom_costs[atom] = 0.0

        # Iteratively relax costs until a fixed point is reached.
        while True:
            costs_changed = False

            # --- 1. Propagate costs through primitive processes ---
            for process in ground_processes:
                # Cost of preconditions is the max cost of any single precond.
                precond_cost = max(
                    [atom_costs[p] for p in process.condition_at_start]
                    or [0.0])

                if precond_cost == float('inf'):
                    continue

                # Actions (endogenous) have cost 1, others (exogenous) have cost 0.
                process_cost = 1.0 if isinstance(
                    process, _GroundEndogenousProcess) else 0.0
                total_cost = precond_cost + process_cost

                # Update costs of effects if we found a cheaper way to achieve them.
                for effect in process.add_effects:
                    if total_cost < atom_costs[effect]:
                        atom_costs[effect] = total_cost
                        costs_changed = True

            # --- 2. Propagate costs through derived predicates (zero-cost) ---
            if use_derived_predicates:
                # We need to loop here to handle chains of derived predicates.
                while True:
                    derived_costs_changed = False
                    # This logic is a simplified version of the incremental approach,
                    # adapted for h_max's cost propagation.
                    current_facts_for_eval = {
                        a
                        for a, c in atom_costs.items() if c != float('inf')
                    }

                    # Check all derived predicates whose inputs might have changed.
                    derived_atoms = utils._abstract_with_derived_predicates(
                        current_facts_for_eval, derived_predicates, objects)

                    for derived_atom in derived_atoms:
                        # To determine the cost, we need to find the specific
                        # atoms that make this derived predicate true. This is
                        # complex, so we approximate by taking the max cost
                        # of any atom in the current state. This is a safe
                        # over-approximation for the preconditions. A more
                        # precise implementation would require inspecting the
                        # logic inside the 'holds' function. For now, we
                        # find the cost of the supporter atoms.
                        # NOTE: This is a simplification. A fully correct h_max
                        # would need to know the specific atoms that satisfy
                        # the 'holds' condition. We find the supporters by
                        # checking the auxiliary predicates.
                        supporter_atoms = set()
                        for p in derived_atom.predicate.auxiliary_predicates:
                            supporter_atoms.update(
                                a for a in current_facts_for_eval
                                if a.predicate == p)

                        if not supporter_atoms: continue

                        derived_cost = max(
                            [atom_costs[a] for a in supporter_atoms] or [0.0])

                        if derived_cost < atom_costs[derived_atom]:
                            atom_costs[derived_atom] = derived_cost
                            derived_costs_changed = True
                            costs_changed = True

                    if not derived_costs_changed:
                        break

            # If no costs were updated in a full pass, we've reached a fixed point.
            if not costs_changed:
                break

        # The heuristic value is the max cost of any goal atom.
        goal_costs = [atom_costs[g] for g in goal]

        # If any goal atom is infinitely costly, the goal is unreachable.
        if not goal_costs or max(goal_costs) == float('inf'):
            return float('inf')

        return max(goal_costs)

    return _h_max_heuristic


def _run_incremental_derived_predicate_logic(
    newly_added_facts: Set[GroundAtom],
    existing_facts: Set[GroundAtom],
    objects: Set[Object],
    dep_to_derived_preds: Dict[Predicate, List[DerivedPredicate]],
) -> Set[GroundAtom]:
    """Incrementally compute the fixed point of derived predicate atoms."""
    all_newly_derived_facts: Set[GroundAtom] = set()
    facts_for_next_iter = newly_added_facts.copy()

    while facts_for_next_iter:
        derived_preds_to_check: Set[DerivedPredicate] = set()
        for fact in facts_for_next_iter:
            if fact.predicate in dep_to_derived_preds:
                derived_preds_to_check.update(
                    dep_to_derived_preds[fact.predicate])

        if not derived_preds_to_check:
            break

        current_state_for_eval = existing_facts | all_newly_derived_facts | newly_added_facts
        potential_new_atoms = utils._abstract_with_derived_predicates(
            current_state_for_eval, derived_preds_to_check, objects)

        truly_new_atoms = potential_new_atoms - (existing_facts
                                                 | all_newly_derived_facts)

        if not truly_new_atoms:
            break

        all_newly_derived_facts.update(truly_new_atoms)
        facts_for_next_iter = truly_new_atoms

    return all_newly_derived_facts


if __name__ == "__main__":
    from predicators.envs.pybullet_boil import PyBulletBoilEnv
    from predicators.ground_truth_models import get_gt_options, \
        get_gt_processes
    args = utils.parse_args()
    utils.update_config(args)
    str_args = " ".join(sys.argv)
    utils.configure_logging()
    CFG.seed = 0
    CFG.env = "pybullet_boil"
    CFG.planning_filter_unreachable_nsrt = False
    CFG.planning_check_dr_reachable = False

    env = PyBulletBoilEnv()
    # objects
    robot = env._robot
    faucet = env._faucet
    jug1 = env._jugs[0]
    burner1 = env._burners[0]

    # Processes
    options = get_gt_options(env.get_name())
    processes = get_gt_processes(env.get_name(), env.predicates, options)
    action_processes = [
        p for p in processes if isinstance(p, EndogenousProcess)
    ]
    pick = [p for p in action_processes if p.name == 'PickJugFromFaucet'][0]
    # place = [p for p in action_processes if p.name == 'PlaceUnderFaucet'][0]
    switch_on = [p for p in action_processes if p.name == 'SwitchFaucetOn'][0]
    switch_off = [p for p in action_processes
                  if p.name == 'SwitchFaucetOff'][0]
    noop = [p for p in action_processes if p.name == 'NoOp'][0]

    plan = [
        switch_on.ground([robot, faucet]),
        switch_off.ground([robot, faucet]),
        noop.ground([robot]),
        noop.ground([robot])
    ]

    # Predicates
    predicates = env.predicates

    def policy():
        global plan
        if len(plan) > 0:
            return plan.pop(0)
        else:
            return None

    # Task
    rng = np.random.default_rng(CFG.seed)
    task = env._make_tasks(1, rng)[0]
    ground_processes, _ = process_task_plan_grounding(
        init_atoms=task.init,
        objects=set(task.init),
        nsrts=processes,
        allow_noops=True,
        compute_reachable_atoms=False)

    world_model = ProcessWorldModel(ground_processes=ground_processes,
                                    state=utils.abstract(
                                        task.init, predicates),
                                    state_history=[],
                                    action_history=[],
                                    scheduled_events={},
                                    t=0)
    for _ in range(100):
        action = policy()
        if action is not None:
            world_model.big_step(action)
        else:
            break
