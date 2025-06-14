import logging
import os
import random
import time
from collections import defaultdict
from pprint import pformat
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
from gym.spaces import Box
from torch import Tensor
from torch.optim import LBFGS, Adam
from tqdm.auto import tqdm

from predicators import utils
from predicators.approaches.process_planning_approach import \
    BilevelProcessPlanningApproach
from predicators.planning_with_processes import process_task_plan_grounding
from predicators.ground_truth_models import get_gt_processes
from predicators.option_model import _OptionModelBase
from predicators.settings import CFG
from predicators.structs import NSRT, AtomOptionTrajectory, CausalProcess, \
    Dataset, EndogenousProcess, ExogenousProcess, GroundAtom, \
    ParameterizedOption, Predicate, Task, Type, _GroundCausalProcess, \
    LowLevelTrajectory

# torch.set_default_dtype(torch.double)


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
        self._processes: Set[CausalProcess] = processes
        self._offline_dataset = Dataset([])

    @classmethod
    def get_name(cls) -> str:
        return "param_learning_process_planning"

    @property
    def is_learning_based(self) -> bool:
        return True

    def _get_current_processes(self) -> Set[CausalProcess]:
        return self._processes

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
        use_lbfgs: bool = False,
    ) -> None:
        """Stochastic (mini-batch) optimisation of process parameters."""
        processes = sorted(self._get_current_processes())
        learn_process_parameters(dataset.trajectories,
                self._get_current_predicates(),
                processes,
                use_lbfgs=use_lbfgs,
            )
        logging.debug("Learned processes:")
        for p in processes:
            logging.debug(pformat(p))
        return

def learn_process_parameters(
        trajectories: List[LowLevelTrajectory],
        predicates: Set[Predicate],
        processes: Sequence[CausalProcess],
        use_lbfgs: bool = False,
        plot_training_curve: bool = True,
        lbfgs_max_iter: int = 200,
        seed: int = 0,
        display_progress: bool = True,
    ) -> Tuple[Sequence[CausalProcess], float]:
    if use_lbfgs:
        num_steps = 1
        batch_size = 100
        inner_lbfgs_max_iter = lbfgs_max_iter
    else:
        num_steps = 200
        batch_size = 16

    torch.manual_seed(seed)

    # -------------------------------------------------------------- #
    # 0.  Cache per-trajectory data & build a global param layout     #
    # -------------------------------------------------------------- #
    max_traj_len = max(len(traj.states) for traj in trajectories)\
        if len(trajectories) > 0 else 0
    per_traj_data, params, num_proc_params =\
        _prepare_training_data_and_model_params(
            predicates,
            processes,
            trajectories)
    init_frame, init_proc_param, init_guide_flat = _split_params_tensor(
        params, num_proc_params)
    _set_process_parameters(processes, init_proc_param,
                                    **{'max_k': max_traj_len})
    # logging.debug(f"Init sum of frame strength: {init_frame.item()}, "
    #                 f"process params: {init_proc_param.sum().item()}, "
    #                 f"guide params: {init_guide_flat.max().item()}")
    # logging.debug("Learned processes:")
    # for p in self._processes:
    #     logging.debug(pformat(p))

    # ------------------- progress bar -------------------------- #
    if use_lbfgs:
        # show one tick *per closure evaluation*
        pbar_total = num_steps * inner_lbfgs_max_iter
        desc = "Training (mini‑batch LBFGS)"
    else:
        # Adam: one tick per optimisation step
        pbar_total = num_steps
        desc = "Training (Adam)"
    if display_progress:
        pbar = tqdm(total=pbar_total, desc=desc)
    else:
        pbar = None

    best_elbo = -float("inf")
    curve: Dict = {
        "iterations": [],
        "elbos": [],
        "best_elbos": [],
        "wall_time": []
    }
    training_start_time = time.time()

    optim: Optional[torch.optim.Optimizer] = None
    if use_lbfgs:
        # LBFGS is re-initialized per outer step or initialized once here.
        # optim = LBFGS([params], max_iter=inner_lbfgs_max_iter,
        # line_search_fn="strong_wolfe")
        pass  # Will be initialized in the loop
    else:
        optim = Adam([params], lr=1e-1)

    # ------------------- training loop ----------------------------- #
    iteration = 0  # counts closure evaluations
    for outer_step in range(num_steps):
        if use_lbfgs:
            # Initialize LBFGS optimizer for the current step/batch
            # current_optim = optim
            current_optim = LBFGS([params],
                                    max_iter=inner_lbfgs_max_iter,
                                    line_search_fn="strong_wolfe")
        else:
            current_optim = optim  # Should be Adam optimizer

        assert current_optim is not None, "Optimizer not initialized"

        # random mini‑batch
        batch_ids = random.sample(range(len(per_traj_data)),
                                    k=min(batch_size, len(per_traj_data)))

        def closure() -> float:
            """Compute –ELBO for the current mini‑batch; do pbar &
            logging."""
            nonlocal best_elbo, iteration  # iteration is modified here

            current_optim.zero_grad(set_to_none=True)

            frame, proc_param, guide_flat = _split_params_tensor(
                params, num_proc_params)
            _set_process_parameters(processes, 
                                    proc_param)

            elbo = torch.tensor(0.0,
                                dtype=frame.dtype,
                                device=params.device)
            for tidx in batch_ids:
                td = per_traj_data[tidx]
                # Pass traj_len explicitly
                guide_dict = _create_guide_dict_for_trajectory(
                    td, guide_flat, td["traj_len"])

                elbo += elbo_torch(
                    [td["trajectory"]],
                    td["ground_causal_processes"],
                    td["start_times_per_gp"],
                    guide_dict,
                    frame,
                    set(td["all_atoms"]),
                    td["atom_to_val_to_gps"],
                )

            # Ensure loss is on the same device as params for backward()
            loss = -(elbo / len(batch_ids))
            loss.backward()  # type: ignore

            detached_elbo_item = elbo.detach().item()
            if detached_elbo_item > best_elbo:
                best_elbo = detached_elbo_item

            # --- progress‑bar & bookkeeping --------------------------------
            curve["iterations"].append(iteration)
            curve["elbos"].append(detached_elbo_item)
            curve["best_elbos"].append(best_elbo)
            curve["wall_time"].append(time.time() - training_start_time)
            if pbar:
                pbar.set_postfix(ELBO=detached_elbo_item, best=best_elbo)
                pbar.update(1)

            iteration += 1
            return loss.item()

        if use_lbfgs:
            current_optim.step(closure)  # LBFGS calls closure internally
        else:
            _ = closure()  # Adam: closure computes loss and gradients
            current_optim.step()

    if pbar:
        pbar.close()

    # ---------------- persist results & plot ------------------------ #
    # Use _split_params_tensor here as well
    frame, proc_params, guide_flat = _split_params_tensor(
        params.detach(), num_proc_params)
    _set_process_parameters(processes, proc_params)
    if plot_training_curve:
        _plot_training_curve(curve)
    return processes, best_elbo


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

def _set_process_parameters(processes: Sequence[CausalProcess],
                            parameters: Tensor,
                            **kwargs: Dict) -> None:
    # Parameters are for the CausalProcess types, not ground instances.
    # Assumes 3 parameters per CausalProcess type (e.g., for its delay distribution)
    num_causal_process_types = len(processes)
    expected_len = 3 * num_causal_process_types
    assert len(parameters) == expected_len, \
        f"Expected {expected_len} params, got {len(parameters)}"

    # Loop through the CausalProcess types
    for i in range(num_causal_process_types):
        param_slice = parameters[i * 3:(i + 1) * 3]
        processes[i]._set_parameters(param_slice, **kwargs)

def _split_params_tensor(
        vec: torch.Tensor,
        num_proc_params: int) -> Tuple[Tensor, Tensor, Tensor]:
    """Helper to split the flat parameter tensor."""
    frame = vec[0]
    proc = vec[1:num_proc_params]
    guide = vec[num_proc_params:]
    return frame, proc, guide

def _prepare_training_data_and_model_params(
    predicates: Set[Predicate],
    processes: Sequence[CausalProcess],
    trajectories: List[LowLevelTrajectory],

) -> Tuple[List[Dict[str, Any]], torch.nn.Parameter, int]:
    """Cache per-trajectory data, build global param layout, and init
    params."""
    atom_option_dataset = utils.create_ground_atom_option_dataset(
        trajectories, predicates)

    per_traj_data: List[Dict[str, Any]] = []
    num_proc_params = 1 + 3 * len(processes)  # frame + process-type
    q_offset = 0  # running index in the guide-param block

    for traj_id, traj in enumerate(atom_option_dataset):
        traj_len = len(traj.states)
        objs = set(traj._low_level_states[0])

        _ground_processes, _ = process_task_plan_grounding(
            init_atoms=set(),
            objects=objs,
            nsrts=processes,
            allow_noops=True,
            compute_reachable_atoms=False,
            )
        ground_processes = [
            gp for gp in _ground_processes
            if isinstance(gp, _GroundCausalProcess)
        ]

        atom_to_val_to_gps: Dict[GroundAtom, Dict[
            bool, Set[_GroundCausalProcess]]] = defaultdict(
                lambda: defaultdict(set))
        for gp in ground_processes:
            for a in gp.add_effects:
                atom_to_val_to_gps[a][True].add(gp)
            for a in gp.delete_effects:
                atom_to_val_to_gps[a][False].add(gp)

        start_times = [[
            t for t in range(traj_len)
            if gp.cause_triggered(traj.states[:t + 1], traj.actions[:t +
                                                                    1])
        ] for gp in ground_processes]

        gp_qparam_id_map: Dict[Tuple[_GroundCausalProcess, int],
                                Tuple[int, int]] = {}
        for gp_idx, gp in enumerate(ground_processes):
            for s_i in start_times[gp_idx]:
                lo, hi = q_offset, q_offset + traj_len
                gp_qparam_id_map[(gp, s_i)] = (lo, hi)
                q_offset = hi

        per_traj_data.append({
            "trajectory":
            traj,
            "traj_len":
            traj_len,
            "ground_causal_processes":
            ground_processes,
            "start_times_per_gp":
            start_times,
            "atom_to_val_to_gps":
            atom_to_val_to_gps,
            "all_atoms":
            utils.all_possible_ground_atoms(
                traj._low_level_states[0], predicates),
            "gp_qparam_id_map":
            gp_qparam_id_map,
        })

    total_params_len = num_proc_params + q_offset
    model_params = torch.nn.Parameter(
        torch.rand(total_params_len, 
                #    dtype=torch.double
                   ))

    return per_traj_data, model_params, num_proc_params

def _create_guide_dict_for_trajectory(
        td: Dict[str, Any], guide_flat: Tensor,
        traj_len: int) -> Dict[_GroundCausalProcess, Dict[int, Tensor]]:
    """Helper to create the guide distribution dictionary for a single
    trajectory."""
    guide_dict: Dict[_GroundCausalProcess,
                        Dict[int, Tensor]] = defaultdict(dict)
    for (gp, s_i), (lo, hi) in td["gp_qparam_id_map"].items():
        raw = guide_flat[lo:hi]
        # Ensure mask is on the same device as raw and has the correct dtype
        mask = torch.ones(traj_len, dtype=raw.dtype, device=raw.device)
        mask[:s_i + 1] = 0
        probs = torch.softmax(raw + torch.log(mask + 1e-20), dim=0)
        guide_dict[gp][s_i] = probs
    return guide_dict

def _plot_training_curve(training_curve: Dict,
                         image_dir: str = "images") -> None:
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
    filename = f"training_curve.png"
    plt.savefig(os.path.join(image_dir, filename))
    logging.info(f"Training curve saved to {filename}")
    plt.close()
