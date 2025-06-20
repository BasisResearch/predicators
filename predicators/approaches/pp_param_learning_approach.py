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
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm.auto import tqdm

from predicators import utils
from predicators.approaches.process_planning_approach import \
    BilevelProcessPlanningApproach
from predicators.ground_truth_models import get_gt_processes
from predicators.option_model import _OptionModelBase
from predicators.planning_with_processes import process_task_plan_grounding
from predicators.settings import CFG
from predicators.structs import NSRT, AtomOptionTrajectory, CausalProcess, \
    Dataset, EndogenousProcess, ExogenousProcess, GroundAtom, \
    LowLevelTrajectory, ParameterizedOption, Predicate, Task, Type, \
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
        _, scores = learn_process_parameters(
            dataset.trajectories[:1],
            self._get_current_predicates(),
            processes,
            use_lbfgs=use_lbfgs,
            lbfgs_max_iter=CFG.process_param_learning_num_steps,
            adam_num_steps=CFG.process_param_learning_num_steps,
            learn_frame_strength=True,
            learn_process_strength=True,
            learn_guide=True,
        )
        logging.debug(f"ELBO: {scores[0]}, exp_state: {scores[1]}, "
                      f"exp_delay: {scores[2]}, entropy: {scores[3]}")
        logging.debug("Learned processes:")
        for p in processes:
            logging.debug(pformat(p))
        logging.debug(f"Log frame strength: {scores[4]}")
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
    adam_num_steps: int = 200,
    std_regularization: Optional[int] = 50,
    learn_guide: bool = False,
    learn_frame_strength: bool = False,
    learn_process_strength: bool = False,
    early_stopping_patience: Optional[int] = 20,
    early_stopping_tolerance: float = 1e-4,
) -> Tuple[Sequence[CausalProcess], Tuple[float, float, float, float, float]]:
    if use_lbfgs:
        num_steps = 1
        batch_size = 100
        inner_lbfgs_max_iter = lbfgs_max_iter
    else:
        num_steps = adam_num_steps
        batch_size = 16

    torch.manual_seed(seed)
    random.seed(seed)

    # -------------------------------------------------------------- #
    # 0.  Cache per-trajectory data & build a global param layout     #
    # -------------------------------------------------------------- #
    max_traj_len = max(len(traj.states) for traj in trajectories)\
        if len(trajectories) > 0 else 0
        
    per_traj_data, proc_and_guide_params_full, num_proc_params = \
        _prepare_training_data_and_model_params(
            predicates,
            processes,
            trajectories
        )

    # --- Separate parameter tensor into logical, learnable components ---
    
    # All process parameters (strength + delay) from the initial tensor
    proc_params_full = proc_and_guide_params_full[:num_proc_params]
    
    learnable_params_for_optim = []

    # Handle guide parameters based on the new flag
    if learn_guide:
        guide_params = torch.nn.Parameter(proc_and_guide_params_full[num_proc_params:])
        learnable_params_for_optim.append(guide_params)
    else:
        # Guide is not learnable; detach it from the computation graph.
        guide_params = proc_and_guide_params_full[num_proc_params:].detach()
    
    fixed_strengths = None
    learnable_proc_params = None
    
    if learn_process_strength:
        learnable_proc_params = torch.nn.Parameter(proc_params_full)
        learnable_params_for_optim.append(learnable_proc_params)
    else:
        num_processes = len(processes)
        strength_indices = [i * 3 for i in range(num_processes)]
        delay_indices = [i * 3 + j for i in range(num_processes) for j in [1, 2]]
        fixed_strengths = torch.full((num_processes,), 0.0) # log_strength=0.0 -> strength=1.0
        learnable_proc_params = torch.nn.Parameter(proc_params_full[delay_indices])
        learnable_params_for_optim.append(learnable_proc_params)

    if learn_frame_strength:
        frame_param = torch.nn.Parameter(torch.randn(1) * 0.01)
        learnable_params_for_optim.append(frame_param)
    else:
        frame_param = torch.tensor([2.5])  # exp(2.5) ≈ 12.2

    init_proc_param = proc_params_full.detach()
    _set_process_parameters(processes, init_proc_param,
                            **{'max_k': max_traj_len})

    # ------------------- progress bar -------------------------- #
    if use_lbfgs:
        pbar_total = num_steps * inner_lbfgs_max_iter
        desc = "Training (mini‑batch LBFGS)"
    else:
        pbar_total = num_steps
        desc = "Training (Adam)"
    if display_progress:
        pbar = tqdm(total=pbar_total, desc=desc)
    else:
        pbar = None

    best_elbo = -float("inf")
    exp_state_at_best, exp_delay_at_best, entropy_at_best = 0.0, 0.0, 0.0
    curve: Dict = {
        "iterations": [],
        "elbos": [],
        "best_elbos": [],
        "wall_time": []
    }
    training_start_time = time.time()

    # --- Early stopping setup ---
    patience_counter = early_stopping_patience
    best_params_state = None
    optim: Optional[torch.optim.Optimizer] = None
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
    if use_lbfgs:
        # LBFGS is re-initialized per outer step or initialized once here.
        # optim = LBFGS([params], max_iter=inner_lbfgs_max_iter,
        # line_search_fn="strong_wolfe")
        pass  # Will be initialized in the loop
    else:
        optim = Adam(learnable_params_for_optim, lr=1e-1)
        # scheduler = ReduceLROnPlateau(optim,
        #                               mode='min',
        #                               factor=0.1,
        #                               patience=10,
        #                               verbose=True,)

    # ------------------- training loop ----------------------------- #
    iteration = 0
    for outer_step in range(num_steps):
        if use_lbfgs:
            current_optim = LBFGS(learnable_params_for_optim,
                                  max_iter=inner_lbfgs_max_iter,
                                  line_search_fn="strong_wolfe")
        else:
            current_optim = optim

        assert current_optim is not None, "Optimizer not initialized"

        remaining_ids = list(range(1, len(per_traj_data)))
        additional_samples = min(batch_size - 1, len(remaining_ids))
        batch_ids = [0] + random.sample(remaining_ids, k=additional_samples)

        def closure() -> float:
            nonlocal best_elbo, iteration, exp_state_at_best, exp_delay_at_best, entropy_at_best
            # --- Early stopping nonlocals ---
            nonlocal patience_counter, best_params_state

            if current_optim:
                current_optim.zero_grad(set_to_none=True)

            if learn_process_strength:
                proc_param = learnable_proc_params
            else:
                proc_param = torch.zeros_like(proc_params_full)
                proc_param[strength_indices] = fixed_strengths.to(proc_param.device)
                proc_param[delay_indices] = learnable_proc_params
            
            _set_process_parameters(processes, proc_param)
            
            guide_flat = guide_params
            frame = frame_param

            elbo = torch.tensor(0.0, dtype=frame.dtype, device=frame.device)
            exp_state = torch.tensor(0.0, dtype=frame.dtype, device=frame.device)
            exp_delay = torch.tensor(0.0, dtype=frame.dtype, device=frame.device)
            entropy = torch.tensor(0.0, dtype=frame.dtype, device=frame.device)

            for tidx in batch_ids:
                td = per_traj_data[tidx]
                guide_dict = _create_guide_dict_for_trajectory(
                    td, guide_flat, td["traj_len"], learn_guide)

                data_elbo, data_exp_state, data_exp_delay, data_entropy = elbo_torch(
                    [td["trajectory"]],
                    td["ground_causal_processes"],
                    td["start_times_per_gp"],
                    guide_dict,
                    frame,
                    set(td["all_atoms"]),
                    td["atom_to_val_to_gps"],
                )
                elbo = elbo + data_elbo
                exp_state = exp_state + data_exp_state
                exp_delay = exp_delay + data_exp_delay
                entropy = entropy + data_entropy

            loss = -(elbo / len(batch_ids))
            if std_regularization and learnable_params_for_optim:
                if learn_process_strength:
                    loss = loss + std_regularization * (proc_param[2::3].sum())
                else:
                    # learnable_proc_params contains only delay params. The std is the 2nd of every 2.
                    loss = loss + std_regularization * (learnable_proc_params[1::2].sum())

            if learnable_params_for_optim:
                loss.backward() # type: ignore

            # ... (logging and bookkeeping is unchanged) ...
            detached_elbo_item = elbo.detach().item()
            
            # --- Early stopping check ---
            if early_stopping_patience is not None:
                if detached_elbo_item > best_elbo + early_stopping_tolerance:
                    # Improvement found, save state and reset patience
                    best_elbo = detached_elbo_item
                    best_params_state = [p.clone().detach() for p in learnable_params_for_optim]
                    exp_state_at_best = exp_state.detach().item()
                    exp_delay_at_best = exp_delay.detach().item()
                    entropy_at_best = entropy.detach().item()
                    patience_counter = early_stopping_patience
                else:
                    # No improvement
                    patience_counter -= 1
            elif detached_elbo_item > best_elbo:
                # Regular tracking of best ELBO if early stopping is off
                 best_elbo = detached_elbo_item
                 exp_state_at_best = exp_state.detach().item()
                 exp_delay_at_best = exp_delay.detach().item()
                 entropy_at_best = entropy.detach().item()

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
            current_optim.step(closure)
        else:
            closure()
            current_optim.step()
            if scheduler:
                scheduler.step(loss)

        # --- Trigger early stop if patience has run out ---
        if early_stopping_patience is not None and patience_counter <= 0:
            print(f"\nEarly stopping triggered after {iteration} iterations.")
            break

    if pbar:
        pbar.close()

    # ---------------- persist results & plot ------------------------ #
    if learn_process_strength:
        proc_params = learnable_proc_params.detach()
    else:
        proc_params = torch.zeros_like(proc_params_full)
        proc_params[strength_indices] = fixed_strengths.to(proc_params.device)
        proc_params[delay_indices] = learnable_proc_params.detach()

    _set_process_parameters(processes, proc_params)
    
    if plot_training_curve:
        _plot_training_curve(curve)
        
    return processes, (best_elbo, exp_state_at_best, exp_delay_at_best,
                       entropy_at_best, frame_param.detach().item())

def elbo_torch(
    atom_option_dataset: List[AtomOptionTrajectory],
    ground_processes: List[
        _GroundCausalProcess],  # All potential ground causal processes
    start_times_per_gp: List[List[
        int]],  # start_times_per_gp[gp_idx] is list of s_i for ground_processes[gp_idx]
    guide: Dict[_GroundCausalProcess,
                Dict[int, Tensor]],  # Variational params q(z_t ; gp, s_i)
    log_frame_strength: Tensor,
    all_possible_atoms: Set[GroundAtom],
    atom_to_val_to_gps: Dict[GroundAtom, Dict[bool,
                                              Set[_GroundCausalProcess]]],
) -> Tensor:
    """*Differentiable* ELBO computation.
    """
    assert len(atom_option_dataset) == 1
    trajectory = atom_option_dataset[0]
    num_time_steps = len(trajectory.states)

    ll = torch.tensor(0.0, dtype=log_frame_strength.dtype)
    yt_prev = trajectory.states[0]

    # -----------------------------------------------------------------
    # 1.  Expected log state probabilities
    # -----------------------------------------------------------------
    exp_state_prob = torch.tensor(0.0, dtype=log_frame_strength.dtype)
    for t in range(1, num_time_steps):
        yt = trajectory.states[t]

        # --- expected effect terms + frame axiom, and normalization
        # contribution from atoms described by the processes
        E_log_Zt = torch.tensor(0.0, dtype=log_frame_strength.dtype)
        for atom, val_to_gps in atom_to_val_to_gps.items():
            # iterate over (val=True, False) pairs that appear in some law
            sum_ytj = torch.tensor(0.0, dtype=log_frame_strength.dtype)
            for val in (True, False):
                gps = val_to_gps[val]
                # expected effect factor for *observed* assignment
                if val == (atom in yt):
                    exp_state_prob = exp_state_prob + sum(
                        q[t] * gp.factored_effect_factor(val, atom)
                        for gp in gps for st, q in guide[gp].items() if st < t)
                # normalisation contribution ---------------------------
                prod = torch.tensor(1.0, dtype=log_frame_strength.dtype)
                for gp in gps:
                    for st, q in guide[gp].items():
                        if st < t:
                            prod = prod * (q[t] * torch.exp(
                                gp.factored_effect_factor(val, atom)) +
                                           (1 - q[t]))
                sum_ytj = sum_ytj + prod * torch.exp(log_frame_strength *
                                                     (val ==
                                                      (atom in yt_prev)))
            E_log_Zt = E_log_Zt + torch.log(sum_ytj + 1e-12)

        # Atoms not referenced in any process law, and
        add_atoms = yt - yt_prev
        del_atoms = yt_prev - yt
        atoms_unchanged = all_possible_atoms - del_atoms - add_atoms
        exp_state_prob = exp_state_prob + log_frame_strength * len(
            atoms_unchanged)

        # Normalization contribution from atoms not described by the processes
        atoms_in_law_effects = set(atom_to_val_to_gps)
        atoms_not_in_law_effects = all_possible_atoms - atoms_in_law_effects
        E_log_Zt = E_log_Zt + len(atoms_not_in_law_effects) * torch.log(
            1 + torch.exp(log_frame_strength))

        exp_state_prob = exp_state_prob - E_log_Zt
        yt_prev = yt
    ll = ll + exp_state_prob

    # -----------------------------------------------------------------
    # 2.  Expected Delay probabilities
    # -----------------------------------------------------------------
    # Iterate through each ground process type and its list of start times
    exp_delay_prob = torch.tensor(0.0, dtype=log_frame_strength.dtype)
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
                    raise Exception(
                        f"Guide distribution not found for {gp_obj} at s_i={s_i}"
                    )

                guide_slice_for_delays = q_dist_for_instance[
                    t_indices_for_guide]

                # Mask for valid log probs (not -inf) and non-zero guide probs
                valid_mask = ~torch.isneginf(all_delay_log_probs) & (
                    guide_slice_for_delays > 1e-9)

                if valid_mask.any():
                    exp_delay_prob = exp_delay_prob + torch.sum(
                                        guide_slice_for_delays[valid_mask] * \
                                    all_delay_log_probs[valid_mask])
    ll = ll + exp_delay_prob

    # -----------------------------------------------------------------
    # 3.  Entropy of the variational distributions
    # -----------------------------------------------------------------
    entropy = torch.tensor(0.0, dtype=log_frame_strength.dtype)
    for start_time_q_map in guide.values(
    ):  # Each value is a Tensor for one (gp,s_i)
        for q_dist_for_instance in start_time_q_map.values():
            mask = q_dist_for_instance > 1e-9
            if mask.any():
                entropy -= torch.sum(q_dist_for_instance[mask] *
                                     torch.log(q_dist_for_instance[mask]))
    elbo = ll + entropy
    return elbo, exp_state_prob, exp_delay_prob, entropy


def _set_process_parameters(processes: Sequence[CausalProcess],
                            parameters: Tensor, **kwargs: Dict) -> None:
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
    """Cache per-trajectory data, build global param layout for process
    and guide parameters, and initialize them."""
    atom_option_dataset = utils.create_ground_atom_option_dataset(
        trajectories, predicates)

    per_traj_data: List[Dict[str, Any]] = []
    # num_proc_params is now just the number of process parameters
    num_proc_params = 3 * len(processes)
    q_offset = 0

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
            bool,
            Set[_GroundCausalProcess]]] = defaultdict(lambda: defaultdict(set))
        for gp in ground_processes:
            for a in gp.add_effects:
                atom_to_val_to_gps[a][True].add(gp)
            for a in gp.delete_effects:
                atom_to_val_to_gps[a][False].add(gp)

        start_times = [[
            t for t in range(traj_len)
            if gp.cause_triggered(traj.states[:t + 1], traj.actions[:t + 1])
        ] for gp in ground_processes]

        gp_qparam_id_map: Dict[Tuple[_GroundCausalProcess, int],
                               Tuple[int, int]] = {}
        for gp_idx, gp in enumerate(ground_processes):
            for s_i in start_times[gp_idx]:
                lo, hi = q_offset, q_offset + traj_len
                gp_qparam_id_map[(gp, s_i)] = (lo, hi)
                q_offset = hi

        per_traj_data.append({
            "trajectory": traj,
            "traj_len": traj_len,
            "ground_causal_processes": ground_processes,
            "start_times_per_gp": start_times,
            "atom_to_val_to_gps": atom_to_val_to_gps,
            "all_atoms": utils.all_possible_ground_atoms(
                traj._low_level_states[0], predicates),
            "gp_qparam_id_map": gp_qparam_id_map,
        })

    # Total parameters for processes and the guide ONLY
    total_params_len = num_proc_params + q_offset
    model_params = torch.nn.Parameter(torch.randn(total_params_len) * 0.01)

    return per_traj_data, model_params, num_proc_params


def _create_guide_dict_for_trajectory(
    td: Dict[str, Any], 
    guide_flat: Tensor,
    traj_len: int,
    learn_guide: bool  # New parameter
) -> Dict[_GroundCausalProcess, Dict[int, Tensor]]:
    """Helper to create the guide distribution dictionary for a single trajectory."""
    guide_dict: Dict[_GroundCausalProcess, Dict[int, Tensor]] = defaultdict(dict)
    for (gp, s_i), (lo, hi) in td["gp_qparam_id_map"].items():
        # Create the causality mask to prevent effects from occurring at or before the cause
        mask = torch.ones(traj_len, dtype=torch.float32, device=guide_flat.device)
        mask[:s_i + 1] = 0

        if learn_guide:
            # Current behavior: softmax over learnable logits
            raw = guide_flat[lo:hi]
            probs = torch.softmax(raw + torch.log(mask + 1e-20), dim=0)
        else:
            # New behavior: fixed uniform distribution over valid time steps
            num_valid_steps = mask.sum()
            if num_valid_steps > 0:
                probs = mask / num_valid_steps
            else:
                # If there are no valid future steps, the distribution is all zeros
                probs = torch.zeros_like(mask)

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
