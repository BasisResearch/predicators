import logging
import os
import time
from collections import defaultdict
from pprint import pformat
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
from gym.spaces import Box
from torch import Tensor
from torch.optim import LBFGS, Adam
from tqdm.auto import tqdm

from predicators import planning, utils
from predicators.approaches.process_planning_approach import \
    BilevelProcessPlanningApproach
from predicators.ground_truth_models import get_gt_processes
from predicators.option_model import _OptionModelBase
from predicators.settings import CFG
from predicators.structs import NSRT, AtomOptionTrajectory, CausalProcess, \
    Dataset, EndogenousProcess, ExogenousProcess, GroundAtom, \
    ParameterizedOption, Predicate, Task, Type, _GroundCausalProcess

torch.set_default_dtype(torch.double)


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
        self._offline_dataset = Dataset([])

    @classmethod
    def get_name(cls) -> str:
        return "param_learning_process_planning"

    @property
    def is_learning_based(self) -> bool:
        return True

    def _get_current_processes(self) -> Set[CausalProcess]:
        return set(self._processes)

    def _get_current_exogenous_processes(self) -> Set[ExogenousProcess]:
        """Get the current set of exogenous processes."""
        return {p for p in self._processes if isinstance(p, ExogenousProcess)}

    def _get_current_endogenous_processes(self) -> Set[EndogenousProcess]:
        """Get the current set of endogenous processes."""
        return {p for p in self._processes if isinstance(p, EndogenousProcess)}

    def _get_current_nsrts(self) -> Set[NSRT]:
        """Get the current set of NSRTs."""
        return set()

    def learn_from_offline_dataset(self, dataset: Dataset) -> None:
        """Learn parameters of processes from the offline datasets.

        This is currently achieved by optimizing the marginal data
        likelihood.
        """
        self._learn_process_parameters(dataset)

    def _learn_process_parameters(
        self,
        dataset: Dataset,
        num_steps: int = 500,
        lr: float = 5e-2,
        use_lbfgs: bool = True,
    ) -> None:
        """Learn parameters of processes from the online dataset."""
        # TODO: relax the assumption of one datapoint

        torch.manual_seed(CFG.seed)

        atom_option_dataset = utils.create_ground_atom_option_dataset(
            dataset.trajectories, self._get_current_predicates())
        trajectory = atom_option_dataset[0]
        traj_len = len(trajectory.states)
        objects = set(trajectory._low_level_states[0])
        self.ground_processes, _ = planning.task_plan_grounding(
            init_atoms=set(),
            objects=objects,
            nsrts=self._processes,
            allow_noops=True,
            compute_reachable_atoms=False)

        # Cache for normalization constants calculation
        atom_to_val_to_gps: Dict[GroundAtom, Dict[
            bool,
            Set[_GroundCausalProcess]]] = defaultdict(lambda: defaultdict(set))
        for gp in self.ground_processes:
            for atom in gp.add_effects:
                atom_to_val_to_gps[atom][True].add(gp)
            for atom in gp.delete_effects:
                atom_to_val_to_gps[atom][False].add(gp)

        num_time_steps = len(trajectory.states)
        start_times = [[
            t for t in range(num_time_steps)
            if gp.cause_triggered(trajectory.states[:t +
                                                    1], trajectory.actions[:t +
                                                                           1])
        ] for gp in self.ground_processes]
        all_possible_atoms = utils.all_possible_ground_atoms(
            trajectory._low_level_states[0], self._get_current_predicates())
        # 1. Initialize the parameters.
        # The parameters are organized as follows:
        # - 1 param for the frame axiom, and
        # - 3 parameters for each process (weight, mean, std)
        # - num_processes * trajectory_length for the variational distribution
        num_processes = len(self._processes)
        num_ground_processes = len(self.ground_processes)
        num_proc_params = 1 + 3 * num_processes
        # New: a guide for each ground process
        num_q_params = num_ground_processes * traj_len
        total_params = num_proc_params + num_q_params

        # -----------------------------------------------------------------
        # 2.  Set up a single flattened torch Parameter vector
        # -----------------------------------------------------------------
        init = torch.rand(total_params, requires_grad=True)
        params = torch.nn.Parameter(init)

        # convenience slices -----------------------------------------------------
        def _split(param_vec: Tensor) -> Tuple[float, Tensor, Tensor]:
            frame_strength = param_vec[0]
            proc_params = param_vec[1:num_proc_params]
            guide_params = param_vec[num_proc_params:]
            return frame_strength, proc_params, guide_params

        # -----------------------------------------------------------------
        # 3.  Choose an optimiser
        # -----------------------------------------------------------------
        if use_lbfgs:
            optimiser = LBFGS([params],
                              max_iter=num_steps,
                              line_search_fn="strong_wolfe")
        else:
            optimiser = Adam([params], lr=lr)

        # -----------------------------------------------------------------
        # 4.  Optimisation loop
        # -----------------------------------------------------------------
        if use_lbfgs:
            # For LBFGS, num_steps is max_iter (max closure evaluations)
            pbar = tqdm(total=num_steps, desc="Training (LBFGS)")
        else:
            # For Adam, num_steps is the number of optimization steps (epochs)
            pbar = tqdm(range(num_steps), desc="Training (Adam)")

        best_elbo = -np.inf
        iteration_count = 0
        training_curve = {
            'iterations': [],
            'elbos': [],
            'best_elbos': [],
            'wall_time': []
        }
        start_time_for_curve = time.time()

        def _closure() -> Tensor:
            nonlocal best_elbo, iteration_count
            optimiser.zero_grad(set_to_none=True)
            frame_strength, proc_params, guide_params = _split(params)

            # push process parameters into the model ----------------------
            self._set_process_parameters(proc_params.detach())

            # construct guide dictionary (softmax per ground process) -----
            guide: Dict[_GroundCausalProcess, Tensor] = {}
            for i, gp in enumerate(self.ground_processes):
                raw = guide_params[i * traj_len:(i + 1) * traj_len]

                mask = torch.ones(traj_len, dtype=raw.dtype)
                if len(start_times[i]) > 0:
                    mask[:start_times[i][0] + 1] = 0.0
                # Zero‑out impossible positions and renormalise
                stable_exp = torch.exp(raw) * mask
                probs = stable_exp / (stable_exp.sum() + 1e-12)
                guide[gp] = probs

            elbo = self.elbo_torch(
                atom_option_dataset,
                self.ground_processes,
                guide,
                frame_strength,
                start_times,
                set(all_possible_atoms),
                atom_to_val_to_gps,
            )

            loss = -elbo  # maximise ELBO == minimise negative ELBO
            loss.backward()
            if elbo > best_elbo:
                best_elbo = elbo.detach().item()
            training_curve['iterations'].append(iteration_count)
            training_curve['elbos'].append(elbo.detach().item())
            training_curve['best_elbos'].append(best_elbo)
            training_curve['wall_time'].append(time.time() -
                                               start_time_for_curve)

            if pbar:
                pbar.set_postfix({
                    "ELBO": elbo.detach().item(),
                    "best_ELBO": best_elbo
                })
                if use_lbfgs:
                    pbar.update(1)

            iteration_count += 1
            return loss

        # LBFGS needs the closure inside step; Adam uses its own loop
        if use_lbfgs:
            optimiser.step(_closure)
        else:
            for _ in pbar:  # type: ignore[arg-type]
                loss = _closure()
                optimiser.step()

        if pbar:
            pbar.close()

        # -----------------------------------------------------------------
        # 5.  Save learned parameters back to the model state
        # -----------------------------------------------------------------
        frame_strength, proc_params, _ = _split(params.detach())
        self._set_process_parameters(proc_params)
        self._plot_training_curve(training_curve)
        logging.debug("Learned processes:")
        for proc in self._processes:
            logging.debug(pformat(proc))
        breakpoint()

    @staticmethod
    def elbo_torch(
        atom_option_dataset: List[AtomOptionTrajectory],
        ground_processes: List[_GroundCausalProcess],
        guide: Dict[_GroundCausalProcess, Tensor],
        frame_strength: Tensor,
        start_times: List[List[int]],
        all_possible_atoms: Set[GroundAtom],
        atom_to_val_to_gps: Dict[GroundAtom, Dict[bool,
                                                  Set[_GroundCausalProcess]]],
    ) -> Tensor:
        """*Differentiable* ELBO computation.
        """
        assert len(atom_option_dataset) == 1
        trajectory = atom_option_dataset[0]
        num_time_steps = len(trajectory.states)

        ll = torch.tensor(0.0, dtype=frame_strength.dtype)
        yt_prev = trajectory.states[0]

        # -----------------------------------------------------------------
        # 1.  Transition factors (refactored formulation)
        # -----------------------------------------------------------------
        for t in range(1, num_time_steps):
            yt = trajectory.states[t]

            # --- expected effect terms + frame axiom -----------------------
            E_log_Zt = torch.tensor(0.0, dtype=frame_strength.dtype)
            for atom, val_to_gps in atom_to_val_to_gps.items():
                # iterate over (val=True, False) pairs that appear in some law
                sum_ytj = torch.tensor(0.0, dtype=frame_strength.dtype)
                for val in (True, False):
                    gps = val_to_gps[val]
                    # expected effect factor for *observed* assignment
                    if val == (atom in yt):
                        ll = ll + sum(guide[gp][t] * gp.factored_effect_factor(
                            val, atom)  # type: ignore[index]
                                      for gp in gps)
                    # normalisation contribution ---------------------------
                    prod = torch.tensor(1.0, dtype=frame_strength.dtype)
                    for gp in gps:
                        prod = prod * (guide[gp][t] * torch.exp(
                            torch.tensor(gp.factored_effect_factor(val, atom)))
                                       + (1 - guide[gp][t]))
                    sum_ytj = sum_ytj + prod * torch.exp(frame_strength *
                                                         (val ==
                                                          (atom in yt_prev)))
                E_log_Zt = E_log_Zt + torch.log(sum_ytj + 1e-12)

            # atoms not referenced in any process law -----------------------
            del_atoms = yt - yt_prev
            add_atoms = yt_prev - yt
            atoms_unchanged = all_possible_atoms - add_atoms - del_atoms
            atoms_in_law_effects = set(atom_to_val_to_gps)
            atoms_unchanged_not_in_law = atoms_unchanged - atoms_in_law_effects
            atoms_changed_not_in_law = (add_atoms
                                        | del_atoms) - atoms_in_law_effects

            ll = ll + frame_strength * len(atoms_unchanged)
            # Atoms unchanged but not described by the processes
            E_log_Zt = E_log_Zt + len(atoms_unchanged_not_in_law) * torch.log(
                1 + torch.exp(frame_strength))
            # Atoms changed and not described by the processes
            E_log_Zt = E_log_Zt + len(atoms_changed_not_in_law) * torch.log(
                torch.tensor(2.0))

            ll = ll - E_log_Zt
            yt_prev = yt

        # -----------------------------------------------------------------
        # 2.  Delay probabilities
        # -----------------------------------------------------------------
        for starts, gp in zip(start_times, ground_processes):
            if len(starts) > 0:
                s0 = starts[0]
                for t in range(s0 + 1, num_time_steps):
                    delay_prob = gp.delay_distribution.probability(t - s0)
                    if delay_prob > 1e-9:
                        ll = ll + guide[gp][t] * torch.log(
                            torch.tensor(delay_prob))

        # -----------------------------------------------------------------
        # 3.  Entropy of the variational distributions
        # -----------------------------------------------------------------
        H = torch.tensor(0.0, dtype=frame_strength.dtype)
        for probs in guide.values():
            mask = probs > 1e-6
            if mask.any():
                H = H - torch.sum(probs[mask] * torch.log(probs[mask]))

        return ll + H

    def _set_process_parameters(self, parameters: Sequence[float]) -> None:
        assert len(parameters) == 3 * len(self._processes)

        # Loop through the parameters 3 at a time
        for i in range(0, len(parameters), 3):
            self._processes[i // 3]._set_parameters(parameters[i:i + 3])

    def _plot_training_curve(self, training_curve: Dict) -> None:
        """Plot the training curve showing ELBO over iterations."""
        import matplotlib.pyplot as plt

        iterations = training_curve['iterations']
        elbos = training_curve['elbos']
        best_elbos = training_curve['best_elbos']
        wall_time = training_curve['wall_time']

        plt.figure(figsize=(18, 6))  # Adjusted figure size for three plots

        # Plot current ELBO vs Iteration
        plt.subplot(1, 2, 1)
        plt.plot(iterations, elbos, 'b-', alpha=0.7, label='Current ELBO')
        plt.plot(iterations, best_elbos, 'r-', linewidth=2, label='Best ELBO')
        plt.xlabel('Iteration')
        plt.ylabel('ELBO')
        plt.title('ELBO vs Iteration')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot ELBO vs Wall Time
        plt.subplot(1, 2, 2)
        plt.plot(wall_time, elbos, 'b-', alpha=0.7, label='Current ELBO')
        plt.plot(wall_time, best_elbos, 'r-', linewidth=2, label='Best ELBO')
        plt.xlabel('Wall Time (s)')
        plt.ylabel('ELBO')
        plt.title('ELBO vs Wall Time')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save the plot
        filename = f"training_curve_seed_{CFG.seed}.png"
        plt.savefig(os.path.join(CFG.image_dir, filename))
        logging.info(f"Training curve saved to {filename}")
        plt.close()
