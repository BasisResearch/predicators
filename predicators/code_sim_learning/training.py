"""Training utilities for the sim-learning approach.

Parameter fitting via emcee (affine-invariant ensemble MCMC).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from predicators.settings import CFG
from predicators.structs import Action, State

logger = logging.getLogger(__name__)

# Step-level simulator: (State, Action, params_dict) -> {Object: {feat: val}}
StepSimulatorFn = Callable[[State, Action, Dict[str, float]], Dict]


@dataclass
class ParamSpec:
    """Specification for a single learnable parameter."""

    name: str
    init_value: float
    lo: Optional[float] = None
    hi: Optional[float] = None


@dataclass
class FitResult:
    """Result of parameter fitting."""

    names: List[str]
    samples: np.ndarray  # (num_samples, num_params)
    log_probs: np.ndarray  # (num_samples,)

    @property
    def point_estimate(self) -> Dict[str, float]:
        """MAP (sample with highest log-probability)."""
        best_idx = int(np.argmax(self.log_probs))
        return {
            n: float(self.samples[best_idx, i])
            for i, n in enumerate(self.names)
        }


def compute_sse(
    simulator_fn: StepSimulatorFn,
    transitions: List[Tuple[State, Action, State]],
    params: Dict[str, float],
    process_features: Dict[str, List[str]],
) -> float:
    """Sum of squared errors between predicted and observed process features.

    Returns the total (un-normalized) SSE so that the Gaussian
    log-likelihood ``-0.5 * SSE / noise_sigma**2`` is the correct
    iid-observation form. Dividing by count would silently rescale the
    per-observation noise by sqrt(count), making the chain insensitive
    to parameter changes.
    """
    total_se = 0.0

    for s_t, action, s_next_obs in transitions:
        updates = simulator_fn(s_t, action, params)

        for obj, feat_dict in updates.items():
            type_name = obj.type.name
            allowed_feats = process_features.get(type_name, [])
            for feat_name, pred_val in feat_dict.items():
                if feat_name not in allowed_feats:
                    continue
                v = pred_val.item() if hasattr(pred_val, 'item') else pred_val
                obs_val = float(s_next_obs.get(obj, feat_name))
                total_se += (v - obs_val)**2

        # Penalize unpredicted features (model predicts no change).
        for obj in s_t:
            type_name = obj.type.name
            for feat_name in process_features.get(type_name, []):
                if obj in updates and feat_name in updates[obj]:
                    continue
                pred_val = float(s_t.get(obj, feat_name))
                obs_val = float(s_next_obs.get(obj, feat_name))
                total_se += (pred_val - obs_val)**2

    return total_se


def compute_residuals(
    simulator_fn: StepSimulatorFn,
    transitions: List[Tuple[State, Action, State]],
    params: Dict[str, float],
    process_features: Dict[str, List[str]],
) -> np.ndarray:
    """Per-feature residuals (predicted - observed) as a flat vector.

    Used by Levenberg-Marquardt, which needs the residual *vector*
    rather than scalar SSE so it can build J = dr/dtheta. Iteration
    order is deterministic so the same theta produces the same vector
    across calls (required for finite-difference Jacobians).
    """
    residuals: List[float] = []
    for s_t, action, s_next_obs in transitions:
        updates = simulator_fn(s_t, action, params)
        for obj in s_t:
            type_name = obj.type.name
            for feat_name in process_features.get(type_name, []):
                if obj in updates and feat_name in updates[obj]:
                    raw = updates[obj][feat_name]
                    pred = raw.item() if hasattr(raw, 'item') else float(raw)
                else:
                    pred = float(s_t.get(obj, feat_name))
                obs = float(s_next_obs.get(obj, feat_name))
                residuals.append(pred - obs)
    return np.asarray(residuals, dtype=float)


def log_sse_breakdown(
    simulator_fn: StepSimulatorFn,
    transitions: List[Tuple[State, Action, State]],
    params: Dict[str, float],
    process_features: Dict[str, List[str]],
    label: str = "",
) -> None:
    """Log per-(type, feature) SSE so we can see which features dominate.

    Splits each feature's residual into two buckets:
      * ``pred``    — transitions where the rule produced an update
                      (residual is sim's prediction error)
      * ``no_pred`` — transitions where no rule fired
                      (residual is whatever the env changed on its own;
                      large values here mean the model is missing a
                      process for this feature)
    """
    bucket: Dict[Tuple[str, str], Dict[str, float]] = {}

    def _slot(key: Tuple[str, str]) -> Dict[str, float]:
        if key not in bucket:
            bucket[key] = {
                "sse_pred": 0.0,
                "n_pred": 0,
                "sse_no_pred": 0.0,
                "n_no_pred": 0,
                "max_abs_err": 0.0,
            }
        return bucket[key]

    for s_t, action, s_next_obs in transitions:
        updates = simulator_fn(s_t, action, params)

        for obj, feat_dict in updates.items():
            type_name = obj.type.name
            allowed_feats = process_features.get(type_name, [])
            for feat_name, pred_val in feat_dict.items():
                if feat_name not in allowed_feats:
                    continue
                v = pred_val.item() if hasattr(pred_val, 'item') else pred_val
                obs_val = float(s_next_obs.get(obj, feat_name))
                err = float(v) - obs_val
                slot = _slot((type_name, feat_name))
                slot["sse_pred"] += err * err
                slot["n_pred"] += 1
                slot["max_abs_err"] = max(slot["max_abs_err"], abs(err))

        for obj in s_t:
            type_name = obj.type.name
            for feat_name in process_features.get(type_name, []):
                if obj in updates and feat_name in updates[obj]:
                    continue
                pred_val = float(s_t.get(obj, feat_name))
                obs_val = float(s_next_obs.get(obj, feat_name))
                err = pred_val - obs_val
                slot = _slot((type_name, feat_name))
                slot["sse_no_pred"] += err * err
                slot["n_no_pred"] += 1
                slot["max_abs_err"] = max(slot["max_abs_err"], abs(err))

    if not bucket:
        return

    total = sum(s["sse_pred"] + s["sse_no_pred"] for s in bucket.values())
    header = f"SSE breakdown{(' — ' + label) if label else ''} " \
             f"(total {total:.4f}):"
    logger.info(header)
    logger.info("  %-22s  %10s  %6s  %10s  %6s  %10s", "type.feature",
                "sse_pred", "n_pred", "sse_no_pred", "n_nop", "max|err|")
    rows = sorted(
        bucket.items(),
        key=lambda kv: -(kv[1]["sse_pred"] + kv[1]["sse_no_pred"]),
    )
    for (type_name, feat_name), s in rows:
        logger.info(
            "  %-22s  %10.4f  %6d  %10.4f  %6d  %10.4f",
            f"{type_name}.{feat_name}",
            s["sse_pred"],
            int(s["n_pred"]),
            s["sse_no_pred"],
            int(s["n_no_pred"]),
            s["max_abs_err"],
        )


def fit_map_lm(
    simulator_fn: StepSimulatorFn,
    transitions: List[Tuple[State, Action, State]],
    param_specs: List[ParamSpec],
    process_features: Dict[str, List[str]],
    max_nfev: int = 200,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Find a MAP estimate via Levenberg-Marquardt (trust-region reflective).

    Returns ``(theta_map, jacobian_at_optimum)``. Jacobian is ``None``
    only if the residual vector is empty or LM raises; in those cases
    callers should treat the diagnostic as unavailable.

    How LM finds the MAP here:
      * ``compute_residuals`` returns r(theta) = (s_{t+1}_obs - sim(s_t, a;
        theta)) flattened over transitions and the features named in
        ``process_features``. Minimizing 0.5 * ||r||^2 is exactly MLE
        under iid Gaussian observation noise; with the broad Gaussian
        prior used elsewhere in this module being effectively flat near
        init, the least-squares minimizer coincides with the MAP.
      * ``scipy.optimize.least_squares(method='trf')`` runs a
        Levenberg-Marquardt step inside a trust region with box
        constraints (``lo``/``hi`` from ``param_specs``). At each step
        it numerically estimates the Jacobian J = dr/dtheta, solves the
        damped normal equations (J^T J + lambda I) dtheta = -J^T r, and
        adapts lambda based on whether the step reduces SSE.
      * On exit, ``result.x`` is theta_map and ``result.jac`` is J at
        the optimum. J^T J / sigma^2 is the Gauss-Newton approximation
        to the negative log-likelihood Hessian — the input
        ``log_hessian_identifiability`` eigendecomposes to flag flat
        directions.

    Two callers (see ``fit_simulator_params``):
      * Hessian identifiability diagnostic — eigendecompose J^T J.
      * MCMC warm start — center emcee walkers on theta_map (and short-
        circuit to it directly when ``num_mcmc_steps == 0``).
    """
    from scipy.optimize import \
        least_squares  # pylint: disable=import-outside-toplevel

    names = [s.name for s in param_specs]
    init = np.array([s.init_value for s in param_specs], dtype=float)
    lo = np.array([s.lo if s.lo is not None else 1e-6 for s in param_specs])
    hi = np.array([s.hi if s.hi is not None else np.inf for s in param_specs])
    # Nudge init strictly into the interior so trf doesn't reject it.
    init = np.maximum(init, lo + 1e-9)
    safe_hi = np.where(np.isfinite(hi), hi - 1e-9, np.inf)
    init = np.minimum(init, safe_hi)

    def residuals_fn(theta: np.ndarray) -> np.ndarray:
        params = {n: float(theta[i]) for i, n in enumerate(names)}
        return compute_residuals(simulator_fn, transitions, params,
                                 process_features)

    init_residuals = residuals_fn(init)
    if init_residuals.size == 0:
        logger.warning("No residuals to fit (empty process_features); "
                       "skipping LM diagnostic.")
        return init, None

    sse_init = float(np.sum(init_residuals**2))

    try:
        result = least_squares(residuals_fn,
                               init,
                               method='trf',
                               bounds=(lo, hi),
                               max_nfev=max_nfev)
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("LM diagnostic raised %s; skipping Hessian log.", exc)
        return init, None

    sse_lm = float(2.0 * result.cost)
    delta = {names[i]: float(result.x[i] - init[i]) for i in range(len(names))}
    logger.info(
        "LM diagnostic fit: SSE %.4f -> %.4f in %d fn-evals (status=%d, %s).",
        sse_init, sse_lm, result.nfev, result.status,
        "converged" if result.success else "max-evals")
    logger.info("LM theta_map - init: %s",
                {k: f"{v:+.4f}"
                 for k, v in delta.items()})

    jac = np.asarray(result.jac, dtype=float)
    if jac.size == 0:
        return np.asarray(result.x, dtype=float), None
    return np.asarray(result.x, dtype=float), jac


def log_hessian_identifiability(
    jacobian: np.ndarray,
    param_names: List[str],
    noise_sigma: float,
    prior_sigma: np.ndarray,
    top_k: int = 3,
) -> None:
    """Eigendecompose the Hessian at the MAP and log identifiability.

    Under a Laplace approximation, the Hessian of the negative
    log-posterior is the inverse posterior covariance. Its eigenvectors
    are *combinations* of parameters (not individual params), and the
    eigenvalues say how tightly the data constrains each combination:

      * Large eigenvalue -> stiff direction: data pins this down.
      * Small eigenvalue -> sloppy direction: data is silent here.

    Sloppy directions point to parameter combinations no optimizer can
    recover from the current data — typically structural rule-pair
    degeneracy or under-excited input trajectories. The Gauss-Newton
    approximation H ~= J^T J / sigma^2 + diag(1/prior_sigma^2) reuses
    the LM Jacobian, so this analysis costs effectively nothing once
    LM has run.
    """
    H_data = jacobian.T @ jacobian / (noise_sigma**2)
    H_prior = np.diag(1.0 / prior_sigma**2)
    H = H_data + H_prior

    eigvals, eigvecs = np.linalg.eigh(H)  # ascending

    cond = float(eigvals[-1] / max(eigvals[0], 1e-30))
    logger.info("Hessian eigenanalysis (cond %.2e, %d params):", cond,
                len(param_names))

    def _format(vec: np.ndarray) -> str:
        order = np.argsort(-np.abs(vec))
        parts = []
        for j in order[:4]:
            if abs(vec[j]) < 0.05:
                break
            parts.append(f"{vec[j]:+.2f} {param_names[j]}")
        return "  ".join(parts) if parts else "(uniform)"

    n = len(eigvals)
    k = min(top_k, n)
    stiff_idx = list(range(n - 1, n - 1 - k, -1))
    stiff_set = set(stiff_idx)
    sloppy_idx = [i for i in range(k) if i not in stiff_set]

    logger.info("  Stiff (well-constrained):")
    for i in stiff_idx:
        logger.info("    lambda = %10.3e :  %s", eigvals[i],
                    _format(eigvecs[:, i]))

    if sloppy_idx:
        logger.info("  Sloppy (under-constrained):")
        for i in sloppy_idx:
            logger.info("    lambda = %10.3e :  %s", eigvals[i],
                        _format(eigvecs[:, i]))


def fit_params(
    simulator_fn: StepSimulatorFn,
    transitions: List[Tuple[State, Action, State]],
    param_specs: List[ParamSpec],
    process_features: Dict[str, List[str]],
    num_walkers: int = 32,
    num_steps: Optional[int] = None,
    burn_in: int = 200,
    noise_sigma: float = 0.05,
    prior_sigma_scale: float = 1.0,
) -> FitResult:
    """Fit simulator parameters via emcee ensemble MCMC.

    Gradient-free — handles all parameter types (rates, thresholds,
    capacities) uniformly. Returns full posterior with uncertainty.

    Args:
        simulator_fn: Simulator(state, action, params_dict) -> updates.
            Should run the base sim internally if needed.
        transitions: List of (s_t, action, s_{t+1}_obs) triples.
        param_specs: Parameter specifications (name, init_value).
        process_features: {type_name: [feat_names]} to fit.
        num_walkers: Number of ensemble walkers (>= 2*ndim).
        num_steps: Total MCMC steps per walker. If None, defaults to
            CFG.code_sim_learning_num_mcmc_steps. If 0, skip training and
            use initial parameter values directly.
        burn_in: Steps to discard as burn-in.
        noise_sigma: Observation noise std dev for likelihood.
        prior_sigma_scale: Prior width as multiple of init_value.

    Returns:
        FitResult with posterior samples and log-probabilities.
    """
    names = [s.name for s in param_specs]
    init_values = np.array([s.init_value for s in param_specs])
    if num_steps is None:
        num_steps = CFG.code_sim_learning_num_mcmc_steps
    if num_steps < 0:
        raise ValueError("code_sim_learning_num_mcmc_steps must be "
                         "non-negative.")
    prior_sigma = init_values * prior_sigma_scale

    # Optional one-shot LM fit. Two independent uses:
    #   * Hessian diagnostic — eigendecompose J^T J at the MAP.
    #   * Warm start — center MCMC walkers on theta_map (and short-circuit
    #     to it directly when num_steps == 0).
    walker_center = init_values
    if (CFG.code_sim_learning_log_hessian_identifiability
            or CFG.code_sim_learning_warm_start_with_lm):
        theta_map, jac = fit_map_lm(simulator_fn, transitions, param_specs,
                                    process_features)
        if (CFG.code_sim_learning_log_hessian_identifiability
                and jac is not None and jac.size > 0):
            log_hessian_identifiability(jac, names, noise_sigma, prior_sigma)
        if CFG.code_sim_learning_warm_start_with_lm:
            walker_center = np.asarray(theta_map, dtype=float)
            logger.info("Warm-starting MCMC walkers from LM MAP estimate.")
            lm_params = {
                n: float(walker_center[i])
                for i, n in enumerate(names)
            }
            lm_sse = compute_sse(simulator_fn, transitions, lm_params,
                                 process_features)
            lm_ll = -0.5 * lm_sse / (noise_sigma**2)
            logger.info(
                "After LM warm start — SSE: %.6f  log-likelihood: %.2f",
                lm_sse, lm_ll)
            log_sse_breakdown(simulator_fn,
                              transitions,
                              lm_params,
                              process_features,
                              label="lm-warm-start")

    if num_steps == 0:
        if CFG.code_sim_learning_warm_start_with_lm:
            logger.info("Skipping emcee; using LM warm-start parameters.")
        else:
            logger.info("Skipping emcee; using initial parameter values.")
        return FitResult(names, walker_center[None, :], np.zeros(1))

    import emcee  # type: ignore[import-untyped]  # pylint: disable=import-outside-toplevel

    ndim = len(param_specs)
    num_walkers = max(num_walkers, 2 * ndim + 2)
    burn_in = min(burn_in, max(num_steps - 1, 0))

    def log_posterior(theta: np.ndarray) -> float:
        # Reject negative values
        if np.any(theta <= 0):
            return -np.inf
        params = {n: float(theta[i]) for i, n in enumerate(names)}
        # Broad Gaussian prior centered on init values
        log_prior = -0.5 * np.sum(((theta - init_values) / prior_sigma)**2)
        # Likelihood
        sse = compute_sse(simulator_fn, transitions, params, process_features)
        return log_prior + (-0.5 * sse / (noise_sigma**2))

    # Initialize walkers across the prior support (sigma = half the prior
    # width). A tight ball around init traps the chain on flat plateaus
    # of the likelihood (e.g., when threshold-based rules don't fire),
    # because emcee stretch moves scale with the swarm's spread.
    p0 = walker_center + 0.5 * prior_sigma * np.random.randn(num_walkers, ndim)
    p0 = np.clip(p0, 1e-6, None)

    sampler = emcee.EnsembleSampler(num_walkers, ndim, log_posterior)

    logger.info("Running emcee: %d walkers, %d steps, %d burn-in.",
                num_walkers, num_steps, burn_in)

    # Run with periodic progress reports.
    report_interval = max(1, num_steps // 5)
    report_interval = 100
    for i, _result in enumerate(sampler.sample(p0, iterations=num_steps),
                                start=1):
        if i % report_interval == 0 or i == num_steps:
            best_lp = sampler.get_log_prob()[:i].max()
            logger.info("  emcee step %d/%d  (best log-prob: %.2f)", i,
                        num_steps, best_lp)
            for h in logger.handlers + logging.getLogger().handlers:
                h.flush()

    # Discard burn-in, flatten chains.
    samples = sampler.get_chain(discard=burn_in, flat=True)
    log_probs = sampler.get_log_prob(discard=burn_in, flat=True)

    result = FitResult(names=names, samples=samples, log_probs=log_probs)

    logger.info("emcee done. Posterior mean: %s",
                {k: f"{v:.4f}"
                 for k, v in result.point_estimate.items()})

    return result
