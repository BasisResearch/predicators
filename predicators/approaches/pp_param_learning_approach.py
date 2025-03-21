import logging
from itertools import chain, combinations
from typing import Dict, List, Optional, Sequence, Set

import numpy as np
from gym.spaces import Box
from scipy.optimize import minimize
from tqdm.auto import tqdm

from predicators import planning, utils
from predicators.approaches.process_planning_approach import \
    BilevelProcessPlanningApproach
from predicators.ground_truth_models import get_gt_processes
from predicators.option_model import _OptionModelBase
from predicators.settings import CFG
from predicators.structs import NSRT, AtomOptionTrajectory, CausalProcess, \
    Dataset, GroundAtom, ParameterizedOption, Predicate, Task, Type


def powerset(iterable):
    """Return an iterator of all possible subsets of the iterable."""
    s = list(iterable)
    powerset = chain.from_iterable(
        combinations(s, r) for r in range(len(s) + 1))
    return (set(x) for x in powerset)


class ParamLearningBilevelProcessPlanningApproach(
        BilevelProcessPlanningApproach):
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
        return "param_learning_process_planning"

    @property
    def is_learning_based(self):
        return True

    def _get_current_processes(self) -> Set[CausalProcess]:
        return set(self._processes)

    def _get_current_nsrts(self) -> Set[NSRT]:
        """Get the current set of NSRTs."""
        return set()

    def learn_from_offline_dataset(self, dataset: Dataset) -> None:
        """Learn parameters of processes from the offline datasets.

        This is currently achieved by optimizing the marginal data
        likelihood.
        """
        # TODO: relax the assumption of one datapoint
        # Q: should there only be a guide for the exogenous processes?
        # A: No, we are also interested in the delay for actions.

        # For now, assume there is a guide for each lifted process.

        # 1. Initialize the parameters.
        # The parameters are organized as follows:
        # - 1 param for the frame axiom, and
        # - 3 parameters for each process (weight, mean, std)
        # - num_processes * trajectory_length for the variational distribution
        atom_option_dataset = utils.create_ground_atom_option_dataset(
            dataset.trajectories, self._get_current_predicates())
        traj_len = len(atom_option_dataset[0].states)
        num_processes = len(self._processes)
        num_proc_params = 1 + 3 * num_processes
        num_q_params = num_processes * traj_len
        num_parameters = num_proc_params + num_q_params

        init_guess = [1] * num_parameters
        bounds = [(0.01, 100)] * num_parameters
        proc_params = init_guess[:num_proc_params]
        guide_params = init_guess[num_proc_params:]
        guide: Dict[CausalProcess, List[float]] = {
            proc: guide_params[i * traj_len:(i + 1) * traj_len]
            for i, proc in enumerate(self._processes)
        }

        # self._set_process_parameters(proc_params[1:])  # skip frame weight
        # pre_learn_elbo = self.elbo(atom_option_dataset,
        #                            self._processes,
        #                            guide,
        #                            frame_strength=proc_params[0],
        #                            predicates=self._get_current_predicates())
        # logging.info(f"Likelihood bound before optimization: {pre_learn_elbo}")

        # Keep track of iterations for progress display
        iteration_count = 0
        progress_bar = tqdm(desc="Optimizing parameters", unit="iter")

        # 2. Define objective and optimize
        def objective(params):
            """Objective function for scipy.optimize.minimize to minimize.

            It does some preparation and then calls the -ELBO function.
            """
            nonlocal iteration_count
            iteration_count += 1
            progress_bar.update(1)

            self._set_process_parameters(params[1:num_proc_params])
            guide_params = params[num_proc_params:]
            guide: Dict[CausalProcess, List[float]] = {
                proc: guide_params[i * traj_len:(i + 1) * traj_len]
                for i, proc in enumerate(self._processes)
            }

            elbo_val = self.elbo(atom_option_dataset,
                                 self._processes,
                                 guide,
                                 frame_strength=params[0],
                                 predicates=self._get_current_predicates())

            progress_bar.set_postfix(elbo=-elbo_val)
            return -elbo_val

        result = minimize(objective,
                          init_guess,
                          bounds=bounds,
                          options={
                              "disp": True,
                              "maxiter": 1000,
                              "pgtol": 1e-9
                          },
                          method="L-BFGS-B")
        progress_bar.close()
        logging.info(f"Best likelihood bound: {-result.fun}")

        # 3. Set the optimized parameters
        self._set_process_parameters(result.x[1:num_proc_params])

    @staticmethod
    def elbo(atom_option_dataset: List[AtomOptionTrajectory],
             processes: List[CausalProcess], guide: Dict[CausalProcess,
                                                         List[float]],
             frame_strength: float, predicates: Set[Predicate]) -> float:
        """Compute the ELBO of the dataset under the model.

        Args:
            atom_option_dataset: ...
            processes: A set of processes. (Our Model)
            guide: A list of variational distributions. (Our Guide)
            frame_strength: The strength of the frame axiom.
        Note that different trajectories could have different objects.
        """
        # Assume there is only one trajectory in the dataset
        assert len(atom_option_dataset) == 1
        trajectory = atom_option_dataset[0]
        num_time_steps = len(trajectory.states)
        objects = set(trajectory._low_level_states[0])
        # ground_processes = utils.all_ground_nsrts(sorted(processes), objects)
        ground_processes = planning.task_plan_grounding(
            init_atoms=set(),
            objects=objects,
            nsrts=processes,
            allow_noops=True,
            compute_reachable_atoms=False)[0]
        assert len(processes) == len(guide)

        # start time per ground process
        # for endogenous processes cause_triggered should be true when the
        # action is taken
        start_times = [[
            t for t in range(num_time_steps)
            if gp.cause_triggered(trajectory.states[:t +
                                                    1], trajectory.actions[:t +
                                                                           1])
        ] for gp in ground_processes]
        # PickJugFromFaucet and FromOutside are counted twice each because we
        # only check options not processes equivalence

        # Current issue: if we record the start time of actions to be when the
        # action.parent (_option) is the same as the ground process's option
        # then multiple processes with the same option will have shared start
        # time.
        # If we record the start time as when the start_condition is also
        # satisfied, some start times are not recorded, presumbly because the
        # start_condition is satisfied during planning, but not in the data
        # derived from the demonstration.
        # -> This generally shouldn't happen because the processes are learned
        # from the same data.

        # TODO: extend to multiple occurrences
        for start_time_list in start_times:
            assert len(start_time_list) <= 1

        # TODO: think more about what to do with the guide for processes that
        # never occur.
        guide = {proc: np.exp(q_i) for proc, q_i in guide.items()}
        for gp, start_time in zip(ground_processes, start_times):
            proc_guide = guide[gp.parent]
            if len(start_time) > 0:
                proc_guide[:start_time[0] + 1] = 0
            # For processes that are not triggered at all, the guide is uniform
            # over time.
            proc_guide /= np.sum(proc_guide)

        # 1. Sum of effect factors for processes
        # 2. Normalization constant per time step
        ll = 0  # Log likelihood

        # all true/false possible predicates-object com
        # TODO: leverage the factored state to reduce complexity
        all_possible_atoms = utils.all_possible_ground_atoms(
            trajectory._low_level_states[0], predicates)

        def tuple_sorted(set_obj):
            return tuple(sorted(set_obj))

        # possible_states = powerset(all_possible_atoms)
        for t in range(1, num_time_steps):
            x_t = trajectory.states[t]

            factor = {
                tuple_sorted(possible_x): 0
                for possible_x in powerset(all_possible_atoms)
            }
            Z = 0

            for possible_x in powerset(all_possible_atoms):

                # Factor from frame axiom
                if tuple_sorted(possible_x) == tuple_sorted(
                        trajectory.states[t - 1]):
                    factor[tuple_sorted(possible_x)] += frame_strength

                # Factor from other processes
                for gp in ground_processes:
                    factor[tuple_sorted(possible_x)] += guide[gp.parent][t] *\
                        gp.effect_factor(possible_x)

                # The first term is -inf
                Z += np.exp(
                    sum(
                        np.log(guide[gp.parent][t] *
                               np.exp(gp.effect_factor(possible_x)) +
                               (1 - guide[gp.parent][t]))
                        for gp in ground_processes) + frame_strength *
                    (tuple_sorted(possible_x) == tuple_sorted(
                        trajectory.states[t - 1])))

                # if tuple_sorted(possible_x) == tuple_sorted(x_t):
                #     logging.debug(f"Factor={factor[tuple_sorted(possible_x)]}")

            logZ = np.log(Z)
            ll += factor[tuple_sorted(x_t)] - logZ
            # logging.debug(f"\tp(x_{t})={np.exp(factor[tuple_sorted(x_t)] - logZ)}, ")

        # 3. Sum of delay probabilities
        # TODO: update for potentially multiple occurrences
        for start_time, gp in zip(start_times, ground_processes):
            if len(start_time) > 0:
                for t in range(start_time[0] + 1, num_time_steps):
                    delay_prob = gp.delay_distribution.probability(
                        t - start_time[0])
                    if delay_prob > 1e-6:
                        ll += guide[gp.parent][t] * np.log(delay_prob)

        # 4. Entropy of the variational distributions
        H = 0
        for q_i in guide.values():
            for p in q_i:
                if p > 1e-6:
                    H -= p * np.log(p)

        logging.debug(f"ELBO={ll + H}")
        return ll + H

    def _set_process_parameters(self, parameters: Sequence[float]) -> None:
        assert len(parameters) == 3 * len(self._processes)

        # Loop through the parameters 3 at a time
        for i in range(0, len(parameters), 3):
            self._processes[i // 3]._set_parameters(parameters[i:i + 3])
