import logging
from collections import defaultdict
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
    Dataset, GroundAtom, ParameterizedOption, Predicate, Task, Type, \
    _GroundCausalProcess

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
    
    def learn_from_offline_dataset(self,
                                   dataset: Dataset,
                                   guide_per_process: bool = False) -> None:
        """Learn parameters of processes from the offline datasets.

        This is currently achieved by optimizing the marginal data
        likelihood.
        """
        self._learn_process_parameters(dataset, guide_per_process)

    def _learn_process_parameters(self,
                                  dataset: Dataset,
                                  guide_per_process: bool = False) -> None:
        """Learn parameters of processes from the online dataset."""
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
        trajectory = atom_option_dataset[0]
        traj_len = len(trajectory.states)
        objects = set(trajectory._low_level_states[0])
        self.ground_processes = planning.task_plan_grounding(
            init_atoms=set(),
            objects=objects,
            nsrts=self._processes,
            allow_noops=True,
            compute_reachable_atoms=False)[0]
        num_time_steps = len(trajectory.states)
        start_times = [[
            t for t in range(num_time_steps)
            if gp.cause_triggered(trajectory.states[:t +
                                                    1], trajectory.actions[:t +
                                                                           1])
        ] for gp in self.ground_processes]
        all_possible_atoms = utils.all_possible_ground_atoms(
            trajectory._low_level_states[0], self._get_current_predicates())
        num_processes = len(self._processes)
        num_ground_processes = len(self.ground_processes)
        num_proc_params = 1 + 3 * num_processes
        # New: a guide for each ground process
        if guide_per_process:
            num_q_params = num_processes * traj_len
        else:
            num_q_params = num_ground_processes * traj_len
        num_parameters = num_proc_params + num_q_params

        init_guess = np.random.rand(num_parameters) # rand init -- kevin
        bounds = [(-100, 100)] * num_parameters # allow negative -- tom
        # init_guess, bounds = self._initialize_parameters(
        #     num_q_params, num_processes)

        # Keep track of iterations for progress display
        iteration_count = 0
        progress_bar = tqdm(desc="Optim. params.", unit="iter")

        # 2. Define objective and optimize
        def objective(params):
            """Objective function for scipy.optimize.minimize to minimize.

            It does some preparation and then calls the -ELBO function.
            """
            nonlocal iteration_count
            nonlocal guide_per_process
            nonlocal start_times
            nonlocal all_possible_atoms

            iteration_count += 1
            progress_bar.update(1)

            self._set_process_parameters(params[1:num_proc_params])
            guide_params = params[num_proc_params:]
            if guide_per_process:
                keys = self._processes
            else:
                keys = self.ground_processes
            guide: Dict[_GroundCausalProcess, List[float]] = {
                proc: guide_params[i * traj_len:(i + 1) * traj_len]
                for i, proc in enumerate(keys)
            }

            elbo_val = self.elbo(atom_option_dataset,
                                 self._processes,
                                 self.ground_processes,
                                 guide,
                                 frame_strength=params[0],
                                 predicates=self._get_current_predicates(),
                                 guide_per_process=guide_per_process,
                                 start_times=start_times,
                                 all_possible_atoms=all_possible_atoms)
            return -elbo_val

        result = minimize(
            objective,
            init_guess,
            bounds=bounds,
            # defaul params work ok
            options={
                "disp": True,
            #     "maxiter": 10000,
            #     "pgtol": 1e-9
            }, 
            method="L-BFGS-B")  # terminate in 19464iter
        progress_bar.close()
        logging.info(f"Best likelihood bound: {-result.fun}")

        # 3. Set the optimized parameters
        self._set_process_parameters(result.x[1:num_proc_params])

    @staticmethod
    def elbo(
        atom_option_dataset: List[AtomOptionTrajectory],
        processes: List[CausalProcess],
        ground_processes: List[_GroundCausalProcess],
        guide: Dict[CausalProcess, List[float]],
        frame_strength: float,
        predicates: Set[Predicate],
        guide_per_process: bool,
        start_times: List[List[int]],
        all_possible_atoms: List[GroundAtom],
    ) -> float:
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
        if guide_per_process:
            assert len(processes) == len(guide)
        else:
            assert len(ground_processes) == len(guide)
        trajectory = atom_option_dataset[0]
        num_time_steps = len(trajectory.states)

        # start time per ground process
        # for endogenous processes cause_triggered should be true when the
        # action is taken

        # TODO: extend to multiple occurrences
        for start_time_list in start_times:
            assert len(start_time_list) <= 1

        # TODO: think more about what to do with the guide for processes that
        # never occur.
        guide = {proc: np.exp(q_i) for proc, q_i in guide.items()}
        for gp, start_time in zip(ground_processes, start_times):
            proc_guide = guide[gp]

            if len(start_time) > 0:
                proc_guide[:start_time[0] + 1] = 0
                # TODO: check what to do for processes that never occur
            proc_guide /= np.sum(proc_guide)

        # 1. Sum of effect factors for processes
        # 2. Normalization constant per time step
        ll = 0  # Log likelihood

        # possible_states = powerset(all_possible_atoms)
        for t in range(1, num_time_steps):
            x_t = trajectory.states[t]
            for j in range(len(all_possible_atoms)):
                factor = defaultdict(float)  # Default value of 0.0
                factor_atom = all_possible_atoms[j]
                x_tj: Optional[GroundAtom] = factor_atom in x_t

                # Factor from frame axiom: if atom didnt change
                if (factor_atom in x_t) == (factor_atom in \
                                                    trajectory.states[t - 1]):
                    factor[x_tj] += frame_strength

                # Factor from other processes
                for gp in ground_processes:
                    if guide_per_process:
                        key = gp.parent
                    else:
                        key = gp
                    factor[x_tj] += guide[key][t] *\
                        gp.factored_effect_factor(x_tj, factor_atom)

                Z = 0
                for atom_value in [True, False]:
                    atom_didnt_change = atom_value == (factor_atom in \
                                                    trajectory.states[t - 1])
                    if guide_per_process:
                        Z += np.exp(
                            sum(
                                np.log(guide[gp.parent][t] * np.exp(
                                    gp.factored_effect_factor(
                                        atom_value, factor_atom)) +
                                       (1 - guide[gp.parent][t]))
                                for gp in ground_processes) +
                            frame_strength * atom_didnt_change)
                    else:
                        Z += np.exp(
                            sum(
                                np.log(guide[gp][t] * np.exp(
                                    gp.factored_effect_factor(
                                        atom_value, factor_atom)) +
                                       (1 - guide[gp][t]))
                                for gp in ground_processes) +
                            frame_strength * atom_didnt_change)

                logZ = np.log(Z)
                ll += factor[x_tj] - logZ

        # 3. Sum of delay probabilities
        # TODO: update for potentially multiple occurrences
        for start_time, gp in zip(start_times, ground_processes):
            if len(start_time) > 0:
                for t in range(start_time[0] + 1, num_time_steps):
                    delay_prob = gp.delay_distribution.probability(
                        t - start_time[0])
                    if delay_prob > 1e-9:
                        if guide_per_process:
                            ll += guide[gp.parent][t] * np.log(delay_prob)
                        else:
                            ll += guide[gp][t] * np.log(delay_prob)

        # 4. Entropy of the variational distributions
        H = 0
        for q_i in guide.values():
            # skipping the guide whose process never occurs
            if q_i[0] == 0:
                for p in q_i:
                    if p > 1e-6:
                        H -= p * np.log(p)

        logging.debug(f"H={H:.4f}, ELBO={ll + H:.4f}")
        return ll + H

    def _set_process_parameters(self, parameters: Sequence[float]) -> None:
        assert len(parameters) == 3 * len(self._processes)

        # Loop through the parameters 3 at a time
        for i in range(0, len(parameters), 3):
            self._processes[i // 3]._set_parameters(parameters[i:i + 3])

    def _initialize_parameters(self, num_q_params: int, num_processes: int):
        """Build an initial guess vector and corresponding bounds for: 1) frame
        axiom strength (1 scalar) 2) per-process parameters:

            - strength_i >= 0
            - lambda_i in (0, 1)
            - nu_i >= 0
        3) variational logits (unbounded real)
        """
        init_guess = []
        bounds = []

        # 1) Frame axiom strength
        #    For example, random in [0, 5], but adjust as needed.
        fa_strength = np.random.uniform(0.0, 10.0)
        init_guess.append(fa_strength)
        # If you want no upper bound, use (0, None).
        # If you prefer a big but finite upper bound, do (0, 1e2) or similar.
        bounds.append((0, 100))

        # 2) For each process, we have three parameters: strength, lambda, nu.
        for _ in range(num_processes):
            # strength_i, e.g. random in [0,5]
            strength_i = np.random.uniform(0.0, 10.0)
            init_guess.append(strength_i)
            bounds.append((0, 100))

            # lambda_i, e.g. random in (0,1)
            lambda_i = np.random.uniform(0.01, 0.99)
            init_guess.append(lambda_i)
            bounds.append((1e-6, 1 - 1e-6))

            # nu_i, e.g. random in [0.01, 5.0]
            # If your use case requires strictly positive or equals 0, adjust accordingly.
            nu_i = np.random.uniform(0.01, 5.0)
            init_guess.append(nu_i)
            bounds.append((1e-6, 100))

        # 3) Variational distribution logits
        #    These can be any real number, so we often have no bounds.
        #    Initialize them near 0 or within [-1, 1].
        for _ in range(num_q_params):
            logit = np.random.uniform(-1.0, 1.0)
            init_guess.append(logit)
            # No bounds on logits => (None, None).
            bounds.append((-100, 100))
        return init_guess, bounds
