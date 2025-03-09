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
from typing import Any, Collection, Dict, FrozenSet, Iterator, List, \
    Optional, Sequence, Set, Tuple

import numpy as np

from predicators import utils
from predicators.option_model import _OptionModelBase
from predicators.planning import PlanningFailure, PlanningTimeout, \
    _MaxSkeletonsFailure, _SkeletonSearchTimeout, task_plan_grounding
from predicators.refinement_estimators import BaseRefinementEstimator
from predicators.settings import CFG
from predicators.structs import NSRT, AbstractPolicy, CausalProcess, \
    DefaultState, DummyOption, EndogenousProcess, ExogenousProcess, \
    GroundAtom, Metrics, Object, OptionSpec, ParameterizedOption, Predicate, \
    State, STRIPSOperator, Task, Type, _GroundCausalProcess, \
    _GroundEndogenousProcess, _GroundExogenousProcess, _GroundNSRT, \
    _GroundSTRIPSOperator, _Option
from predicators.utils import EnvironmentFailure, _TaskPlanningHeuristic


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

    def __init__(self,
                 ground_processes: List[_GroundCausalProcess],
                 state: Set[GroundAtom],
                 state_history: List[Set[GroundAtom]] = [],
                 action_history: List[Optional[_GroundEndogenousProcess]] = [],
                 scheduled_events: Dict[int, List[Tuple[_GroundCausalProcess,
                                                        int]]] = {},
                 t: int = 0) -> None:

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

    def small_step(
            self,
            small_step_action: Optional[_GroundEndogenousProcess] = None
    ) -> None:
        """Will keep the current action as a class variable for now, as opposed
        to a part of the state variable as in the demo code."""
        initial_state = self.state.copy()

        # 1. current_action is set to an action when this small_step is first
        # called. `small_step_action` will be None in all subsequent calls until
        # some aspects of the state changes, where the big_step loop will break.
        # This is set to None
        if small_step_action is not None:
            # In the original implementation, this corresponds to adding an atom
            # to the initial_state.
            self.current_action = small_step_action.copy()
            logging.debug(f"At time {self.t}, start performing "
                          f"{self.current_action.name}")

        # 2. Process events scheduled for this timestep.
        if self.t in self.scheduled_events:
            for g_process, start_time in self.scheduled_events[self.t]:
                # If it's the end of an endogenous process, i.e. an action,
                # should change the current action back to None.
                # if process.condition_overall(self.history[start_time+1:]) and\
                #         process.condition_at_end(self.state):

                if (all(
                        g_process.condition_overall.issubset(s)
                        for s in self.state_history[start_time + 1:])
                        and g_process.condition_at_end.issubset(self.state)):
                    logging.debug(f"At time {self.t}:")
                    for atom in g_process.delete_effects:
                        self.state.discard(atom)
                        logging.debug(f"Discarding {atom}")
                    for atom in g_process.add_effects:
                        self.state.add(atom)
                        logging.debug(f"Adding {atom}")
                    if isinstance(g_process, _GroundEndogenousProcess) and\
                        small_step_action is None:
                        self.current_action = None
            del self.scheduled_events[self.t]

        # 3. Schedule new events whose processes are met
        # TODO: should the scheduling be before processing the effects in step 2
        # or after? Because in the current order, if, at state 0, the agent
        # takes an action that takes 1 step to take effect,
        for g_process in self.ground_processes:
            # logging.debug(f"At time {self.t}, trying to schedule {g_process.name}")
            satisfy_condition_at_start = g_process.condition_at_start.issubset(
                self.state)
            no_prev_state_or_prev_doesnt_satisfy = (
                len(self.state_history) == 0
                or not g_process.condition_at_start.issubset(
                    self.state_history[-1]))
            is_endogenous = isinstance(g_process, _GroundEndogenousProcess)
            is_exogenous = isinstance(g_process, _GroundExogenousProcess)
            first_step_running_action = small_step_action is not None and \
                                        g_process == small_step_action
            # logging.debug(f"Condition at start: {satisfy_condition_at_start} "
            #               f"no prev state or prev doesnt satisfy: {no_prev_state_or_prev_doesnt_satisfy} "
            #               f"Is endogenous: {is_endogenous} "
            #               f"Is exogenous: {is_exogenous} "
            #               f"First step running action: {first_step_running_action}")
            if (satisfy_condition_at_start and
                ((is_exogenous and no_prev_state_or_prev_doesnt_satisfy) or
                 (is_endogenous and first_step_running_action))):
                delay = g_process.delay_distribution.sample()
                schedued_time = self.t + delay
                logging.debug(f"At time {self.t}, scheduling "
                              f"{g_process.name_and_objects_str()} "
                              f"for time {schedued_time}")
                if schedued_time not in self.scheduled_events:
                    self.scheduled_events[schedued_time] = []
                self.scheduled_events[schedued_time].append(
                    (g_process, self.t))

        # 4. Check if state changes -- for printing and deactivating the wait
        # action
        # if self.state != initial_state:
        #     # Can log the change here
        #     # logging.debug(...)

        #     for t in list(self.scheduled_events.keys()):
        #         for process, start_time in self.scheduled_events[t]:
        #             if process.name == 'NoOp':
        #                 self.scheduled_events[t].remove((process, start_time))

        # This is moved from before step 3 to after, because other wise at t=0,
        # there will be two states in the state_history buffer.
        self.action_history.append(self.current_action.copy() if self.
                                   current_action is not None else None)
        self.state_history.append(self.state.copy())
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
                # initial_state = self.state.copy()
                action_process = None
                # effect_have_not_occurred = True
            # else:
            action_not_finished = self.current_action is not None

            # if NoOp is scheduled to end,
            wait_end = False
            if self.t in self.scheduled_events:
                for g_process, start_time in self.scheduled_events[self.t]:
                    if g_process.name == 'NoOp':
                        wait_end = True
                        for t in list(self.scheduled_events.keys()):
                            for process, start_time in self.scheduled_events[
                                    t]:
                                if process.name == 'NoOp':
                                    self.scheduled_events[t].remove(
                                        (process, start_time))
                        break
            if wait_end:
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
    use_visited_state_set: bool = False
) -> Iterator[Tuple[List[_GroundEndogenousProcess], List[Set[GroundAtom]]]]:

    # Filter out all the action from processes
    # zero heuristic
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
    hq.heappush(queue,
                (heuristic(root_node.atoms), rng_prio.uniform(), root_node))
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
            logging.debug(f"\nGot Plan:")
            for process in node.skeleton:
                logging.debug(process.name_and_objects_str())
            logging.debug("")
            for i, (state, action) in enumerate(
                    zip(node.state_history, node.action_history)):
                logging.debug(f"State {i}: {state}")
                logging.debug(
                    f"Action {i}: {action.name_and_objects_str() if action is not None else None}"
                )
            logging.debug(
                f"State {len(node.state_history)}: {node.state_history[-1]}")
            breakpoint()
            yield node.skeleton, node.atoms_sequence
        else:
            # Generate successors.
            metrics["num_nodes_expanded"] += 1
            # Skip abstract policy support...
            for action_process in utils.get_applicable_operators(
                    ground_action_processes, node.atoms):

                # --- Run the action process on the world model
                world_model = ProcessWorldModel(
                    ground_processes=ground_processes.copy(),
                    state=node.atoms.copy(),
                    state_history=node.state_history.copy(),
                    action_history=node.action_history.copy(),
                    scheduled_events=deepcopy(node.scheduled_events),
                    t=len(node.state_history))

                assert isinstance(action_process, _GroundEndogenousProcess)
                plan_so_far = [p.name_and_objects_str() for p in node.skeleton]
                # logging.info(f"Expand after plan {plan_so_far}:")
                # if len(node.skeleton) > 2 and \
                #     node.skeleton[0].name == 'PickJug' and \
                #     node.skeleton[1].name == 'PlaceUnderFaucet' and \
                #     node.skeleton[2].name == 'SwitchFaucetOn' and \
                #     action_process.name == 'NoOp':

                # if len(node.skeleton) == 5 and \
                #     node.skeleton[0].name == 'NoOp' and \
                #     node.skeleton[1].name == 'SwitchFaucetOn' and \
                #     node.skeleton[2].name == 'NoOp' and \
                #     node.skeleton[3].name == 'NoOp' and \
                #     node.skeleton[4].name == 'NoOp' and \
                #     action_process.name == 'PickJugFromFaucet':
                #     breakpoint()
                # if len(node.skeleton) == 0 and \
                #     action_process.name == 'SwitchFaucetOn':
                #     breakpoint()
                # if len(node.skeleton) == 1 and \
                #     node.skeleton[0].name == 'SwitchFaucetOn' and \
                #     action_process.name == 'NoOp':
                #     breakpoint()
                # if len(node.skeleton) == 2 and \
                #     node.skeleton[0].name == 'SwitchFaucetOn' and \
                #     node.skeleton[1].name == 'NoOp' and \
                #     action_process.name == 'SwitchBurnerOn':
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
                child_cost = node.cumulative_cost + 1.0
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
                priority = (child_node.cumulative_cost +
                            heuristic(child_node.atoms))
                hq.heappush(queue, (priority, rng_prio.uniform(), child_node))
                if time.perf_counter() - start_time >= timeout:
                    break
    if not queue:
        raise _MaxSkeletonsFailure("Planning ran out of skeletons!")
    assert time.perf_counter() - start_time >= timeout
    raise _SkeletonSearchTimeout


def task_plan(
    init_atoms: Set[GroundAtom],
    goal: Set[GroundAtom],
    ground_processes: List[_GroundCausalProcess],
    reachable_atoms: Set[GroundAtom],
    heuristic: _TaskPlanningHeuristic,
    seed: int,
    timeout: float,
    max_skeletons_optimized: int,
    use_visited_state_set: bool = False,
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
    generator = _skeleton_generator_with_processes(
        dummy_task,
        ground_processes,
        init_atoms,
        heuristic,
        seed,
        timeout,
        metrics,
        max_skeletons_optimized,
        use_visited_state_set=use_visited_state_set)

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

    init_atoms = utils.abstract(task.init, preds)
    goal = task.goal
    objects = set(task.init)

    start_time = time.perf_counter()

    if CFG.sesame_task_planner == "astar":
        ground_processes, reachable_atoms = task_plan_grounding(
            init_atoms,
            objects,
            processes,
            allow_noops=True,
            compute_reachable_atoms=compute_reachable_atoms)
        # TODO: is this applicable here?
        # assert task_planning_heuristic is not None
        # heuristic = lambda x: 0
        heuristic = utils.create_task_planning_heuristic(
            task_planning_heuristic, init_atoms, goal, ground_processes, preds,
            objects)
        duration = time.perf_counter() - start_time
        timeout -= duration
        plan, atoms_seq, metrics = next(
            task_plan(
                init_atoms,
                goal,
                ground_processes,
                reachable_atoms,
                heuristic,
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

    # processes
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

    def policy():
        global plan
        if len(plan) > 0:
            return plan.pop(0)
        else:
            return None

    # Task
    rng = np.random.default_rng(CFG.seed)
    task = env._make_tasks(1, rng)[0]
    ground_processes, _ = task_plan_grounding(init_atoms=task.init,
                                              objects=set(task.init),
                                              nsrts=processes,
                                              allow_noops=True,
                                              compute_reachable_atoms=False)

    breakpoint()
    world_model = ProcessWorldModel(ground_processes=ground_processes,
                                    state=utils.abstract(
                                        task.init, env.predicates),
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
