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
        num_steps: int = 200,
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
        # Filter for CausalProcess instances as task_plan_grounding can return NSRTs too
        ground_causal_processes: List[_GroundCausalProcess] = [
            gp for gp in self.ground_processes
            if isinstance(gp, _GroundCausalProcess)
        ]

        for gp in ground_causal_processes:
            for atom in gp.add_effects:
                atom_to_val_to_gps[atom][True].add(gp)
            for atom in gp.delete_effects:
                atom_to_val_to_gps[atom][False].add(gp)

        num_time_steps = len(trajectory.states)
        # start_times_per_gp[gp_idx] is a list of start times for causal_ground_processes[gp_idx]
        start_times_per_gp = [[
            t for t in range(num_time_steps)
            if gp.cause_triggered(trajectory.states[:t +
                                                    1], trajectory.actions[:t +
                                                                           1])
        ] for gp in ground_causal_processes]

        # Create a flat list of all process instances (gp, s_i)
        # Each instance will have its own variational distribution.
        process_instances: List[Tuple[_GroundCausalProcess, int]] = []
        for gp_idx, gp in enumerate(ground_causal_processes):
            for s_i in start_times_per_gp[gp_idx]:
                process_instances.append((gp, s_i))

        all_possible_atoms = utils.all_possible_ground_atoms(
            trajectory._low_level_states[0], self._get_current_predicates())
        # 1. Initialize the parameters.
        # The parameters are organized as follows:
        # - 1 param for the frame axiom, and
        # - 3 parameters for each CausalProcess (defined by the process type)
        # - Variational parameters for q(z_t | gp, s_i) for each instance.
        num_processes = len(self._processes)  # Number of CausalProcess *types*
        num_activated_process_instances = len(process_instances)
        num_proc_params = 1 + 3 * num_processes  # Params for frame axiom + CausalProcess types
        # Each active instance (gp, s_i) gets a variational distribution over traj_len.
        num_q_params = num_activated_process_instances * traj_len
        total_params = num_proc_params + num_q_params

        # -----------------------------------------------------------------
        # 2.  Set up a single flattened torch Parameter vector
        # -----------------------------------------------------------------
        init = torch.rand(total_params, requires_grad=True)
        params = torch.nn.Parameter(init)

        # convenience slices -----------------------------------------------------
        def _split(
            param_vec: Tensor
        ) -> Tuple[Tensor, Tensor, Tensor]:  # frame_strength is a Tensor now
            frame_strength = param_vec[0]
            proc_params = param_vec[1:num_proc_params]
            guide_q_params = param_vec[num_proc_params:]
            return frame_strength, proc_params, guide_q_params

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
        training_curve: Dict[str, List] = {
            'iterations': [],
            'elbos': [],
            'best_elbos': [],
            'wall_time': []
        }
        start_time_for_curve = time.time()

        def _closure() -> float:  # LBFGS expects a float return
            nonlocal best_elbo, iteration_count
            optimiser.zero_grad(set_to_none=True)
            frame_strength, proc_params, guide_params = _split(params)

            # push CausalProcess parameters into the model ----------------------
            self._set_process_parameters(proc_params.detach())

            # construct guide dictionary (softmax per ground process instance) -----
            guide: Dict[_GroundCausalProcess, Dict[int,
                                                   Tensor]] = defaultdict(dict)
            current_q_param_idx = 0
            for gp_instance_tuple in process_instances:
                gp, start_t = gp_instance_tuple
                raw_q_for_instance = guide_params[
                    current_q_param_idx:current_q_param_idx + traj_len]
                current_q_param_idx += traj_len

                mask = torch.ones(traj_len, dtype=raw_q_for_instance.dtype)
                # q(z_t | gp, s_i) is 0 if t <= s_i (effect must be after start)
                mask[:start_t + 1] = 0.0

                stable_exp = torch.exp(raw_q_for_instance) * mask
                probs_sum = stable_exp.sum()

                if probs_sum < 1e-12:
                    # Ensure probs has same dtype and device as stable_exp
                    probs = torch.zeros_like(stable_exp)
                else:
                    probs = stable_exp / probs_sum
                guide[gp][start_t] = probs

            elbo = self.elbo_torch(
                atom_option_dataset,
                ground_causal_processes,
                start_times_per_gp,
                guide,
                frame_strength,
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
            return loss.item()  # LBFGS expects a float

        # LBFGS needs the closure inside step; Adam uses its own loop
        if use_lbfgs:
            optimiser.step(_closure)
        else:
            for _ in pbar:
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
        ground_processes: List[
            _GroundCausalProcess],  # All potential ground causal processes
        start_times_per_gp: List[List[
            int]],  # start_times_per_gp[gp_idx] is list of s_i for ground_processes[gp_idx]
        guide: Dict[_GroundCausalProcess,
                    Dict[int, Tensor]],  # Variational params q(z_t ; gp, s_i)
        frame_strength: Tensor,
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
                        ll = ll + sum(
                            q[t] * gp.factored_effect_factor(val, atom)
                            for gp in gps
                            for st, q in guide[gp].items() if st < t)
                    # normalisation contribution ---------------------------
                    prod = torch.tensor(1.0, dtype=frame_strength.dtype)
                    for gp in gps:
                        for st, q in guide[gp].items():
                            if st < t:
                                # q(z_t | gp, s_i) * exp(factor)
                                prod = prod * (q[t] * torch.exp(
                                    torch.tensor(
                                        gp.factored_effect_factor(val, atom)))
                                               + (1 - q[t]))
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
                torch.tensor(2.0, dtype=frame_strength.dtype))

            ll = ll - E_log_Zt
            yt_prev = yt

        # -----------------------------------------------------------------
        # 2.  Delay probabilities
        # -----------------------------------------------------------------
        # Iterate through each ground process type and its list of start times
        for gp_idx, gp_obj in enumerate(ground_processes):
            for s_i in start_times_per_gp[
                    gp_idx]:  # s_i is a specific start time for gp_obj
                # This instance is (gp_obj, s_i)
                # Effects can manifest at t = s_i + d, where d >= 1 (delay)
                if s_i + 1 < num_time_steps:  # Check if any delay is possible
                    # Delays d = 1, 2, ..., (num_time_steps - 1 - s_i)
                    delay_values = torch.arange(1,
                                                num_time_steps - s_i,
                                                dtype=torch.long)
                    if delay_values.numel() == 0:
                        continue

                    # t_indices are the time steps where effects manifest: s_i+1, ..., num_time_steps-1
                    t_indices_for_guide = s_i + delay_values

                    # Get log prob of these delays P(d | gp_obj's params)
                    all_delay_log_probs = gp_obj.delay_distribution.log_prob(
                        delay_values)  # type: ignore

                    # Get q(z_t | gp_obj, s_i) for t in t_indices_for_guide
                    q_dist_for_instance = guide.get(gp_obj).get(s_i, None)
                    if q_dist_for_instance is None:
                        continue

                    guide_slice_for_delays = q_dist_for_instance[
                        t_indices_for_guide]

                    # Mask for valid log probs (not -inf) and non-zero guide probs
                    valid_mask = ~torch.isneginf(all_delay_log_probs) & (
                        guide_slice_for_delays > 1e-9)

                    if valid_mask.any():
                        ll += torch.sum(guide_slice_for_delays[valid_mask] * \
                                        all_delay_log_probs[valid_mask])

        # -----------------------------------------------------------------
        # 3.  Entropy of the variational distributions
        # -----------------------------------------------------------------
        H = torch.tensor(0.0, dtype=frame_strength.dtype)
        for start_time_q_map in guide.values(
        ):  # Each value is a Tensor for one (gp,s_i)
            for q_dist_for_instance in start_time_q_map.values():
                mask = q_dist_for_instance > 1e-9
                if mask.any():
                    H -= torch.sum(q_dist_for_instance[mask] *
                                   torch.log(q_dist_for_instance[mask]))
        return ll + H

    def _set_process_parameters(self, parameters: Tensor) -> None:
        # Parameters are for the CausalProcess types, not ground instances.
        # Assumes 3 parameters per CausalProcess type (e.g., for its delay distribution)
        num_causal_process_types = len(self._processes)
        expected_len = 3 * num_causal_process_types
        assert len(parameters) == expected_len, \
            f"Expected {expected_len} params, got {len(parameters)}"

        # Loop through the CausalProcess types
        for i in range(num_causal_process_types):
            param_slice = parameters[i * 3:(i + 1) * 3]
            self._processes[i]._set_parameters(
                param_slice.tolist())  # Pass list of floats

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
