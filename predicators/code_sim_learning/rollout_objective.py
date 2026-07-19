"""The free-running rollout objective for physical system identification.

SSE / residual-vector / per-trajectory-RMS evaluations of the joint
physical+rule forward model, and the rollout Levenberg-Marquardt MAP
fit built on them. See the
:mod:`predicators.code_sim_learning.physical_sysid` module docstring
for why the objective free-runs the base sim.
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from predicators.code_sim_learning.fit_space import ParamSpec, to_fit_space
from predicators.code_sim_learning.lm import solve_lm
from predicators.code_sim_learning.rollout_env import RolloutTrajectory, \
    rollout_states
from predicators.code_sim_learning.trajectory_prep import ResidualScaling
from predicators.structs import Action, State

# Relative finite-difference step for the rollout LM Jacobian. scipy's
# default (~sqrt(machine eps) ~ 1e-8) is below the contact solver's
# sensitivity: the rollout comes back bitwise identical and the Jacobian
# is identically zero, stalling LM at init. Swept empirically on the
# domino friction-recovery smoke test (true 0.35, init 0.8): 0.01
# stalls at 0.446 (noise-dominated Jacobian), 0.05 is flaky (0.43),
# while 0.02 recovers 0.353-0.356 in ~21-30 evals. Contact-rich
# landscapes are rough at multiple scales — re-sweep this if a new
# domain's LM fit stalls well above the MCMC-quality SSE.
_ROLLOUT_LM_DIFF_STEP = 2e-2


def compute_rollout_sse(
    base_env: Any,
    trajectories: List[RolloutTrajectory],
    params: Dict[str, float],
    process_features: Dict[str, List[str]],
    physical_names: Sequence[str],
    rules: Sequence[Any] = (),
    latent_init: Any = None,
    scaling: Optional[ResidualScaling] = None,
) -> float:
    """Total per-step SSE between free-running rollouts and observations.

    ``params`` is the *joint* parameter dict (physical and rule params in one
    namespace); the ``physical_names`` subset is pushed into the env via
    ``apply_physical_param_overrides`` while the full dict is handed to the
    rules, mirroring how ``compute_sse``/``compute_sse_recurrent`` pass params.
    When ``rules`` are present they are applied on top of each *rolled-out*
    base state (latents threaded per trajectory when declared), and a rule's
    predicted feature overrides the base sim's — the same precedence
    ``merge_updates`` uses at plan time.

    Observed and simulated states may carry different ``Object`` instances
    (e.g. from separately-constructed envs / real-env trajectories), so
    features are matched by object name.

    With ``scaling``, residuals are wrapped (angular features) and
    normalized per feature (see :class:`ResidualScaling`); the SSE is
    then dimensionless. All evaluations belonging to one fit must share
    the same ``scaling`` object or their SSEs are incomparable.
    """
    return sum(r * r for r in _iter_rollout_residual_terms(
        base_env, trajectories, params, process_features, physical_names,
        rules, latent_init, scaling))


def compute_rollout_residuals(
    base_env: Any,
    trajectories: List[RolloutTrajectory],
    params: Dict[str, float],
    process_features: Dict[str, List[str]],
    physical_names: Sequence[str],
    rules: Sequence[Any] = (),
    latent_init: Any = None,
    scaling: Optional[ResidualScaling] = None,
) -> np.ndarray:
    """Rollout residuals (predicted - observed, scaled) as a flat vector.

    Levenberg-Marquardt counterpart of :func:`compute_rollout_sse` (same
    prediction pipeline, same iteration order — the sim is deterministic,
    so the same theta yields the same vector, as finite-difference
    Jacobians require).
    """
    return np.asarray(list(
        _iter_rollout_residual_terms(base_env, trajectories, params,
                                     process_features, physical_names, rules,
                                     latent_init, scaling)),
                      dtype=float)


def _iter_rollout_residual_terms(
    base_env: Any,
    trajectories: List[RolloutTrajectory],
    params: Dict[str, float],
    process_features: Dict[str, List[str]],
    physical_names: Sequence[str],
    rules: Sequence[Any],
    latent_init: Any,
    scaling: Optional[ResidualScaling] = None,
) -> Iterator[float]:
    """Yield per-feature residuals for the joint forward model.

    Shared prediction pipeline behind :func:`compute_rollout_sse` and
    :func:`compute_rollout_residuals`: free-run the base sim at the
    physical slice of ``params``, apply rules (latents threaded) on each
    rolled-out state, and score every in-scope feature against its
    observation. Deterministic iteration order. Without ``scaling`` the
    residual is the raw ``pred - obs`` difference (legacy objective).
    """
    # pylint: disable=import-outside-toplevel
    from predicators.code_sim_learning.utils import apply_rules, \
        apply_rules_with_latent, has_latent_rules, init_latent

    # pylint: enable=import-outside-toplevel
    physical = {n: params[n] for n in physical_names if n in params}
    rules_list = list(rules)
    latent_mode = bool(rules_list) and has_latent_rules(rules_list)

    for states, actions in trajectories:
        sim_states = rollout_states(base_env, states[0], actions, physical)
        latent: Dict[str, Any] = (init_latent(latent_init, params)
                                  if latent_mode else {})
        history: List[Tuple[State, Optional[Action]]] = []
        for i, sim_state in enumerate(sim_states):
            obs_state = states[i + 1]
            updates: Dict[Any, Dict[str, Any]] = {}
            if rules_list:
                if latent_mode:
                    history.append((sim_state, actions[i]))
                    updates = apply_rules_with_latent(sim_state, latent,
                                                      history, rules_list,
                                                      params)
                else:
                    updates = apply_rules(sim_state, rules_list, params)
            obs_by_name = {o.name: o for o in obs_state}
            for obj in sim_state:
                feats = process_features.get(obj.type.name, [])
                if not feats:
                    continue
                obs_obj = obs_by_name.get(obj.name)
                if obs_obj is None:
                    continue
                feat_updates = updates.get(obj, {})
                for feat in feats:
                    pred = feat_updates.get(feat, sim_state.get(obj, feat))
                    pred_val = float(
                        pred.item() if hasattr(pred, "item") else pred)
                    obs_val = float(obs_state.get(obs_obj, feat))
                    if scaling is not None:
                        yield scaling.residual(obj.type.name, feat, pred_val,
                                               obs_val)
                    else:
                        yield pred_val - obs_val


def fit_map_lm_rollout(
    base_env: Any,
    trajectories: List[RolloutTrajectory],
    physical_specs: Sequence[ParamSpec],
    process_features: Dict[str, List[str]],
    rules: Sequence[Any] = (),
    rule_specs: Sequence[ParamSpec] = (),
    latent_init: Any = None,
    max_nfev: int = 200,
    scaling: Optional[ResidualScaling] = None,
    prior_centers: Optional[np.ndarray] = None,
    prior_sigmas: Optional[np.ndarray] = None,
    noise_sigma: float = 0.05,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """MAP estimate of the joint physical+rule theta via Levenberg-Marquardt.

    Rollout counterpart of :func:`training.fit_map_lm`, built on
    :func:`compute_rollout_residuals` and the shared bound-aware
    ``solve_lm`` core. Uses a coarse relative finite-difference step
    (``_ROLLOUT_LM_DIFF_STEP``) because simulation residuals are flat
    under scipy's default ~1e-8 perturbations. Returns ``(theta_map,
    jacobian_at_optimum)``; the Jacobian feeds both the identifiability
    report and the Laplace exploration ensemble.

    When ``prior_centers``/``prior_sigmas`` (FIT-space, one per joint
    param) are given, the Gaussian prior is folded into the LM
    objective as extra residual rows ``noise_sigma * (z - c) / sigma``,
    so the point estimate is a true MAP: a direction the data leaves
    flat stays exactly at its anchor instead of drifting on rollout
    noise (a flat likelihood's finite-difference gradient is pure
    noise). The prior rows are stripped from the returned Jacobian -
    its consumers (Laplace ensemble, Hessian diagnostic) add the prior
    term themselves and would otherwise double-count it.
    """
    all_specs = list(physical_specs) + list(rule_specs)
    names = [s.name for s in all_specs]
    physical_names = [s.name for s in physical_specs]
    use_prior = prior_centers is not None and prior_sigmas is not None

    def residuals_fn(theta: np.ndarray) -> np.ndarray:
        params = {n: float(theta[i]) for i, n in enumerate(names)}
        res = compute_rollout_residuals(base_env, trajectories, params,
                                        process_features, physical_names,
                                        rules, latent_init, scaling)
        if not use_prior:
            return res
        z = to_fit_space(all_specs, theta)
        prior_res = noise_sigma * (z - prior_centers) / prior_sigmas
        return np.concatenate([res, prior_res])

    theta_map, jac = solve_lm(residuals_fn,
                              all_specs,
                              max_nfev,
                              "rollout",
                              diff_step=_ROLLOUT_LM_DIFF_STEP)
    if use_prior and jac is not None and jac.shape[0] > len(all_specs):
        jac = jac[:-len(all_specs)]
    return theta_map, jac


def per_trajectory_rms(
    base_env: Any,
    trajectories: List[RolloutTrajectory],
    params: Dict[str, float],
    process_features: Dict[str, List[str]],
    physical_names: Sequence[str],
    rules: Sequence[Any] = (),
    latent_init: Any = None,
    scaling: Optional[ResidualScaling] = None,
) -> List[float]:
    """RMS rollout residual of each trajectory separately at ``params``.

    The per-trajectory analogue of :func:`compute_rollout_sse`: how well
    can the model explain THIS trajectory at these parameters? Without
    ``scaling`` the RMS is in the scored features' native units (meters
    / radians per residual); with it, a dimensionless fraction of
    typical motion.
    """
    out: List[float] = []
    for traj in trajectories:
        res = compute_rollout_residuals(base_env, [traj], params,
                                        process_features, physical_names,
                                        rules, latent_init, scaling)
        out.append(float(np.sqrt(np.mean(res**2))) if res.size else 0.0)
    return out
