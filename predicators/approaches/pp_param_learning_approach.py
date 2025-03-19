import logging
from typing import Dict, List, Optional, Sequence, Set

import numpy as np
from gym.spaces import Box
from scipy.optimize import minimize

from predicators import utils
from predicators.approaches.process_planning_approach import \
    BilevelProcessPlanningApproach
from predicators.ground_truth_models import get_gt_processes
from predicators.option_model import _OptionModelBase
from predicators.settings import CFG
from predicators.structs import NSRT, CausalProcess, Dataset, \
    ParameterizedOption, Predicate, Task, Type


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

    def learn_from_offline_dataset(
            self,
            dataset: Dataset,
            share_guide_across_groundings: bool = False):
        """Learn parameters of processes from the offline datasets.

        This is currently achieved by optimizing the marginal data
        likelihood.
        """
        # TODO: relax the assumption of one datapoint
        # Q: should there only be a guide for the exogenous processes?
        # A: No, we are also interested in the delay for actions.

        # TODO: figure out if there should be a guide for each lifted process
        # or ground process.
        # For now, assume there is a guide for each lifted process.

        # 1. Initialize the parameters.
        # The parameters are organized as follows:
        # - 1 param for the frame axiom, and
        # - 3 parameters for each process (weight, mean, std)
        # - num_processes * trajectory_length for the variational distribution
        traj_len = len(dataset.trajectories[0].states)
        num_processes = len(self._processes)
        num_proc_params = 1 + 3 * num_processes
        num_q_params = num_processes * traj_len
        num_parameters = num_proc_params + num_q_params

        init_guess = [1] * num_parameters
        bounds = [(0.01, 100)] * num_parameters
        proc_params = init_guess[:num_proc_params]
        guide_params = init_guess[num_proc_params:]
        # TODO: make this a mapping from process to q? 1
        guide: Dict[CausalProcess, List[float]] = {
            proc: guide_params[i * traj_len:(i + 1) * traj_len]
            for i, proc in enumerate(self._processes)
        }

        self._set_process_parameters(proc_params[1:])  # skip frame weight
        pre_learn_elbo = self.elbo(dataset,
                                   self._processes,
                                   guide,
                                   frame_strength=proc_params[0])
        logging.info(f"Likelihood bound before optimization: {pre_learn_elbo}")

        # 2. Define objective and optimize
        def objective(params):
            """Objective function for scipy.optimize.minimize to minimize.

            It does some preparation and then calls the -ELBO function.
            """
            self._set_process_parameters(proc_params[1:num_proc_params])
            guide_params = params[num_proc_params:]
            guide: Dict[CausalProcess, List[float]] = {
                proc: guide_params[i * traj_len:(i + 1) * traj_len]
                for i, proc in enumerate(self._processes)
            }

            return -self.elbo(dataset,
                              self._get_current_processes,
                              guide,
                              frame_strength=params[0])

        result = minimize(objective,
                          init_guess,
                          bounds=bounds,
                          options={
                              "disp": True,
                              "maxiter": 1000,
                              "pgtol": 1e-9
                          },
                          methods="L-BFGS-B")
        logging.info(f"Best likelihood bound: {result.fun}")

        # 3. Set the optimized parameters
        self._set_process_parameters(result.x[1:num_proc_params])

    @staticmethod
    def elbo(dataset: Dataset,
             processes: List[CausalProcess],
             guide: Dict[CausalProcess, List[float]],
             frame_strength: float = 1.0) -> float:
        """Compute the ELBO of the dataset under the model.

        Args:
            dataset: A dataset.
            processes: A set of processes. (Our Model)
            guide: A list of variational distributions. (Our Guide)
            frame_strength: The strength of the frame axiom.
        Note that different trajectories could have different objects.
        """
        # TODO: update the q usage here as it's now a dictionary
        # Assume there is only one trajectory in the dataset
        trajectory = dataset.trajectories[0]
        num_time_steps = len(trajectory.states)
        objects = set(trajectory.states[0])
        ground_processes = utils.all_ground_nsrts(sorted(processes), objects)
        assert len(processes) == len(guide)

        # start time per ground process
        # TODO: for endogenous processes, cause_triggered should be true when
        # the action is taken
        start_times = [[
            t for t in range(num_time_steps)
            if gp.cause_triggered(trajectory.states[:t + 1])
        ] for gp in ground_processes]
        guide = {proc: np.exp(q_i) for proc, q_i in guide.items()}

        for gp, start_time in zip(ground_processes, start_times):
            proc_guide = guide[gp.parent]
            proc_guide[:start_time[0] + 1] = 0
            proc_guide /= np.sum(proc_guide)

        # 1. Sum of effect factors for processes
        # 2. Normalization constant per time step
        ll = 0  # Log likelihood

        # all true/false possible predicates-object combinations
        possible_states = ...  # TODO: implement this
        for t in range(1, num_time_steps):
            x_t = trajectory.states[t]

            factor = {tuple(possible_x): 0 for possible_x in possible_states}
            Z = 0

            for possible_x in possible_states:

                # Factor from frame axiom
                if tuple(possible_x) == tuple(trajectory[t - 1]):
                    factor[tuple(possible_x)] += frame_strength

                # Factor from other processes
                for gp in ground_processes:
                    factor[tuple(possible_x)] += guide[gp.parent][t] *\
                        gp.effect_factor(possible_x)

                Z += np.exp(
                    sum(
                        np.log(guide[gp.parent][t]) * np.exp(
                            gp.effect_factor(possible_x) +
                            (1 - guide[gp.parent][t]))
                        for gp in ground_processes) + frame_strength *
                    (tuple(possible_x) == tuple(trajectory[t - 1])))

                if tuple(possible_x) == tuple(x_t):
                    logging.debug(f"Factor={factor[tuple(possible_x)]}")

            logZ = np.log(Z)
            ll += factor[tuple(x_t)] - logZ
            logging.debug(f"\tp(x_{t})={np.exp(factor[tuple(x_t)] - logZ)}, ")

        # 3. Sum of delay probabilities
        # TODO: update for potentially multiple occurrences
        for start_time, gp in zip(start_times, ground_processes):
            for t in range(start_time[0] + 1, num_time_steps):
                ll += guide[gp.parent][t] * np.log(gp.delay_distribution(t - 
                                                                start_time[0]))

        # 4. Entropy of the variational distributions
        H = 0
        for q_i in guide.values():
            for p in q_i:
                if p > 1e-6:
                    H -= p * np.log(p)

        return ll + H

    def _set_process_parameters(self, parameters: Sequence[float]) -> None:
        assert len(parameters) == 3 * len(self._processes)

        # Loop through the parameters 3 at a time
        for i in range(0, len(parameters), 3):
            self._processes[i // 3]._set_parameters(parameters[i:i + 3])
