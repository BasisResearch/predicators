import logging
import os
import time
from collections import defaultdict
from pprint import pformat
from typing import Dict, List, Optional, Sequence, Set, Tuple, Any
import random

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
        num_steps: int = 1,
        use_lbfgs: bool = True,
        batch_size: int = 100,
        inner_lbfgs_max_iter: int = 50,
    ) -> None:
        """Stochastic (mini-batch) optimisation of process parameters.

        """

        torch.manual_seed(CFG.seed)

        # -------------------------------------------------------------- #
        # 0.  Cache per-trajectory data & build a global param layout     #
        # -------------------------------------------------------------- #
        atom_option_dataset = utils.create_ground_atom_option_dataset(
            dataset.trajectories, self._get_current_predicates())

        per_traj: List[Dict[str, Any]] = []
        inst_slice: Dict[Tuple[int, _GroundCausalProcess, int],
                         Tuple[int, int]] = {}

        num_proc_params = 1 + 3 * len(self._processes)  # frame + process-type
        q_offset = 0  # running index in the guide-param block

        for traj_id, traj in enumerate(atom_option_dataset):
            
            tlen = len(traj.states)
            objs = set(traj._low_level_states[0])

            gp_all, _ = planning.task_plan_grounding(
                init_atoms=set(),
                objects=objs,
                nsrts=self._processes,
                allow_noops=True,
                compute_reachable_atoms=False)

            gpcs = [
                gp for gp in gp_all if isinstance(gp, _GroundCausalProcess)
            ]

            atom2gps: Dict[GroundAtom, Dict[bool, Set[_GroundCausalProcess]]] \
                = defaultdict(lambda: defaultdict(set))
            for gp in gpcs:
                for a in gp.add_effects:
                    atom2gps[a][True].add(gp)
                for a in gp.delete_effects:
                    atom2gps[a][False].add(gp)

            starts = [[
                t for t in range(tlen)
                if gp.cause_triggered(traj.states[:t + 1], traj.actions[:t +
                                                                        1])
            ] for gp in gpcs]

            slice_map: Dict[Tuple[_GroundCausalProcess, int], Tuple[int,
                                                                    int]] = {}
            for gp_idx, gp in enumerate(gpcs):
                for s_i in starts[gp_idx]:
                    lo, hi = q_offset, q_offset + tlen
                    slice_map[(gp, s_i)] = (lo, hi)
                    inst_slice[(traj_id, gp, s_i)] = (lo, hi)
                    q_offset = hi

            per_traj.append({
                "trajectory": traj,
                "traj_len": tlen,
                "ground_causal_processes": gpcs,
                "start_times_per_gp": starts,
                "atom2gps": atom2gps,
                "all_atoms": utils.all_possible_ground_atoms(
                    traj._low_level_states[0], self._get_current_predicates()),
                "slice_map": slice_map,
            })

        total_params = num_proc_params + q_offset
        params = torch.nn.Parameter(
            torch.rand(total_params, dtype=torch.double))

        # ---------------------- helpers -------------------------------- #
        def _split(vec: torch.Tensor):
            frame = vec[0]
            proc = vec[1:num_proc_params]
            guide = vec[num_proc_params:]
            return frame, proc, guide

        # ------------------- progress bar -------------------------- #

        if use_lbfgs:
            # show one tick *per closure evaluation* (i.e. per function/grad call)
            pbar = tqdm(total=num_steps * inner_lbfgs_max_iter,
                        desc="Training (mini‑batch LBFGS)")
        else:
            # Adam: one tick per optimisation step
            pbar = tqdm(range(num_steps), desc="Training (Adam)")

        best_elbo = -float("inf")
        curve = {"iterations": [], "elbos": [], "best_elbos": [], 
                 "wall_time": []}
        training_start_time = time.time()
        if not use_lbfgs:
            optim = Adam([params])

        # ------------------- training loop ----------------------------- #
        iteration = 0          # counts closure evaluations
        for outer_step in range(num_steps):
            if use_lbfgs:
                optim = LBFGS([params],
                            max_iter=inner_lbfgs_max_iter,
                            line_search_fn="strong_wolfe")
            # random mini‑batch
            batch_ids = random.sample(range(len(per_traj)),
                                      k=min(batch_size, len(per_traj)))

            def closure() -> float:
                """Compute –ELBO for the current mini‑batch; do pbar & logging."""
                nonlocal best_elbo, iteration
                optim.zero_grad(set_to_none=True)

                frame, proc, guide_flat = _split(params)
                self._set_process_parameters(proc.detach())

                elbo = torch.tensor(0.0, dtype=frame.dtype)
                for tidx in batch_ids:
                    td = per_traj[tidx]
                    guide_dict = defaultdict(dict)
                    for (gp, s_i), (lo, hi) in td["slice_map"].items():
                        raw = guide_flat[lo:hi]
                        mask = torch.ones(td["traj_len"], dtype=raw.dtype)
                        mask[:s_i + 1] = 0
                        probs = torch.softmax(raw + torch.log(mask + 1e-12), dim=0)
                        # stable_exp = torch.exp(raw) * mask
                        # probs = stable_exp / stable_exp.sum()
                        guide_dict[gp][s_i] = probs

                    elbo += self.elbo_torch(
                        [td["trajectory"]],
                        td["ground_causal_processes"],
                        td["start_times_per_gp"],
                        guide_dict,
                        frame,
                        set(td["all_atoms"]),
                        td["atom2gps"],
                    )
                loss = -(elbo / len(batch_ids))
                loss.backward()

                if elbo > best_elbo:
                    best_elbo = elbo.detach().item()

                # --- progress‑bar & bookkeeping --------------------------------
                curve["iterations"].append(iteration)
                curve["elbos"].append(elbo.detach().item())
                curve["best_elbos"].append(best_elbo)
                curve["wall_time"].append(time.time() - training_start_time)
                if pbar:
                    pbar.set_postfix(ELBO=elbo.detach().item(), best=best_elbo)
                if use_lbfgs:          # tick every closure evaluation
                    pbar.update(1)

                iteration += 1
                return loss.item()

            if use_lbfgs:
                optim.step(closure)          # LBFGS calls closure internally
            else:
                loss = closure()             # one call for Adam
                optim.step()

        pbar.close()

        # ---------------- persist results & plot ------------------------ #
        _, proc_params, _ = _split(params.detach())
        self._set_process_parameters(proc_params)
        self._plot_training_curve(curve)
        logging.debug("Learned processes:")
        for p in self._processes:
            logging.debug(pformat(p))
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
