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
        """Posterior mean."""
        mean = self.samples.mean(axis=0)
        return {n: float(mean[i]) for i, n in enumerate(self.names)}


def compute_mse(
    simulator_fn: StepSimulatorFn,
    transitions: List[Tuple[State, Action, State]],
    params: Dict[str, float],
    process_features: Dict[str, List[str]],
) -> float:
    """Compute MSE between predicted and observed process features."""
    total_se = 0.0
    count = 0

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
                count += 1

        # Penalize unpredicted features (model predicts no change).
        for obj in s_t:
            type_name = obj.type.name
            for feat_name in process_features.get(type_name, []):
                if obj in updates and feat_name in updates[obj]:
                    continue
                pred_val = float(s_t.get(obj, feat_name))
                obs_val = float(s_next_obs.get(obj, feat_name))
                total_se += (pred_val - obs_val)**2
                count += 1

    if count == 0:
        return 0.0
    return total_se / count


def fit_params(
    simulator_fn: StepSimulatorFn,
    transitions: List[Tuple[State, Action, State]],
    param_specs: List[ParamSpec],
    process_features: Dict[str, List[str]],
    num_walkers: int = 32,
    num_steps: Optional[int] = None,
    burn_in: int = 200,
    noise_sigma: float = 0.05,
    prior_sigma_scale: float = 2.0,
) -> FitResult:
    """Fit simulator parameters via emcee ensemble MCMC.

    Gradient-free — handles all parameter types (rates, thresholds,
    capacities) uniformly. Returns full posterior with uncertainty.

    Args:
        simulator_fn: Simulator(state, action, params_dict) -> updates.
            Should run kinematics internally if needed.
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
    if num_steps == 0:
        logger.info("Skipping emcee; using initial parameter values.")
        return FitResult(names, init_values[None, :], np.zeros(1))

    import emcee  # type: ignore[import-untyped]  # pylint: disable=import-outside-toplevel

    ndim = len(param_specs)
    num_walkers = max(num_walkers, 2 * ndim + 2)
    prior_sigma = init_values * prior_sigma_scale
    burn_in = min(burn_in, max(num_steps - 1, 0))

    def log_posterior(theta: np.ndarray) -> float:
        # Reject negative values
        if np.any(theta <= 0):
            return -np.inf
        params = {n: float(theta[i]) for i, n in enumerate(names)}
        # Broad Gaussian prior centered on init values
        log_prior = -0.5 * np.sum(((theta - init_values) / prior_sigma)**2)
        # Likelihood
        mse = compute_mse(simulator_fn, transitions, params, process_features)
        return log_prior + (-0.5 * mse / (noise_sigma**2))

    # Initialize walkers in a small ball around init values.
    p0 = init_values * (1.0 + 0.01 * np.random.randn(num_walkers, ndim))

    sampler = emcee.EnsembleSampler(num_walkers, ndim, log_posterior)

    logger.info("Running emcee: %d walkers, %d steps, %d burn-in.",
                num_walkers, num_steps, burn_in)

    # Run with periodic progress reports.
    report_interval = max(1, num_steps // 5)
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
