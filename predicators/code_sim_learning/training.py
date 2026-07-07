"""Training utilities for the sim-learning approach.

Parameter fitting via emcee (affine-invariant ensemble MCMC).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from predicators.settings import CFG
from predicators.structs import Action, State

logger = logging.getLogger(__name__)

# Step-level simulator: (State, Action, params_dict) -> {Object: {feat: val}}
StepSimulatorFn = Callable[[State, Action, Dict[str, float]], Dict]

# Per-trajectory list of (base_state, action, next_obs) triples.
# `base_state` is the base sim applied to the previous *real* observation,
# matching the shape used by `compute_sse` but grouped by trajectory so
# the latent block can be threaded across steps within each one.
TrajectoryTriples = List[Tuple[State, Action, State]]


@dataclass
class ParamSpec:
    """Specification for a single learnable parameter.

    ``scale`` selects the fitting parameterization. ``"linear"`` (the
    default) fits theta directly. ``"log"`` fits ``z = log(theta)`` —
    the right choice for positive scale-like parameters (friction,
    mass) whose behavioral effect is multiplicative: the grid sweep
    becomes geometric (equal resolution per decade instead of piling
    every point at the high end), LM finite-difference steps become
    relative, and the Gaussian prior in z is a log-normal that treats
    "4x smaller" and "4x larger" as equally plausible. Everything
    simulator- and caller-facing stays in linear units; only the
    optimizer's internal coordinates change.
    """

    name: str
    init_value: float
    lo: Optional[float] = None
    hi: Optional[float] = None
    scale: str = "linear"

    def __post_init__(self) -> None:
        if self.scale not in ("linear", "log"):
            raise ValueError(f"ParamSpec scale must be 'linear' or 'log', "
                             f"got {self.scale!r} for {self.name!r}.")
        if self.scale == "log":
            if self.init_value <= 0:
                raise ValueError(f"log-scale param {self.name!r} needs a "
                                 f"positive init_value, got "
                                 f"{self.init_value}.")
            if self.lo is not None and self.lo <= 0:
                raise ValueError(f"log-scale param {self.name!r} needs a "
                                 f"positive lo bound, got {self.lo}.")


@dataclass
class FitResult:
    """Result of parameter fitting.

    The optional ``jacobian``/``noise_sigma``/``prior_sigma`` fields are a
    Laplace bundle, attached by both :func:`fit_params` and
    :func:`fit_params_recurrent` whenever their Levenberg-Marquardt fit
    ran (info-seeking exploration or the Hessian/warm-start flags). They
    let a caller build a calibrated posterior covariance
    ``(J^T J / sigma^2 + diag(1/prior^2))^-1`` around the MAP without
    re-deriving it. They stay ``None`` when LM was skipped or failed —
    e.g. MCMC-only runs, where ``samples`` already carries the posterior.

    Space conventions: ``samples`` (and therefore ``point_estimate``)
    are always in EXTERNAL (linear, simulator-facing) units, while
    ``jacobian`` and ``prior_sigma`` live in the FIT space — for a
    log-scale parameter that means d(residual)/d(log theta) and a
    log-space prior width. ``scales`` records each column's ``ParamSpec
    .scale`` so consumers (identifiability probe, Laplace ensemble) can
    map between the two; ``None`` means all-linear (legacy results).
    """

    names: List[str]
    samples: np.ndarray  # (num_samples, num_params) — EXTERNAL units
    log_probs: np.ndarray  # (num_samples,)
    jacobian: Optional[np.ndarray] = None  # (num_residuals, num_params) at MAP
    noise_sigma: Optional[float] = None  # observation-noise sigma used in fit
    prior_sigma: Optional[
        np.ndarray] = None  # (num_params,) Gaussian-prior std, FIT space
    scales: Optional[List[str]] = None  # per-param ParamSpec.scale

    @property
    def point_estimate(self) -> Dict[str, float]:
        """MAP (sample with highest log-probability)."""
        best_idx = int(np.argmax(self.log_probs))
        return {
            n: float(self.samples[best_idx, i])
            for i, n in enumerate(self.names)
        }


def _param_bounds(
        param_specs: List[ParamSpec]) -> Tuple[np.ndarray, np.ndarray]:
    """Per-parameter (lo, hi) box from the ParamSpecs, in EXTERNAL units.

    An unspecified bound defaults to a small positive floor (lo) or +inf
    (hi). A parameter that declares a negative ``lo`` -- e.g. a signed
    local offset whose true value is negative -- is therefore fit over
    its real range, while a parameter that declares no bounds keeps the
    historical positivity assumption. Shared by the LM and emcee paths
    so they constrain to the same box.
    """
    lo = np.array([s.lo if s.lo is not None else 1e-6 for s in param_specs])
    hi = np.array([s.hi if s.hi is not None else np.inf for s in param_specs])
    return lo, hi


def _is_log(spec: ParamSpec) -> bool:
    """Whether ``spec`` fits in log-space (tolerates legacy instances)."""
    return getattr(spec, "scale", "linear") == "log"


def _to_internal(param_specs: List[ParamSpec], values: Any) -> np.ndarray:
    """Map external (linear) parameter values into the fit space."""
    return np.array([
        np.log(v) if _is_log(s) else float(v)
        for s, v in zip(param_specs, values)
    ],
                    dtype=float)


def _to_external(param_specs: List[ParamSpec], values: Any) -> np.ndarray:
    """Map fit-space parameter values back to external (linear) units."""
    return np.array([
        np.exp(v) if _is_log(s) else float(v)
        for s, v in zip(param_specs, values)
    ],
                    dtype=float)


def _rows_to_external(param_specs: List[ParamSpec],
                      arr: np.ndarray) -> np.ndarray:
    """Map a (num_rows, num_params) fit-space array to external units."""
    out = np.array(arr, dtype=float, copy=True)
    for j, spec in enumerate(param_specs):
        if _is_log(spec):
            out[:, j] = np.exp(out[:, j])
    return out


def _internal_bounds(
        param_specs: List[ParamSpec]) -> Tuple[np.ndarray, np.ndarray]:
    """The `_param_bounds` box mapped into the fit space."""
    lo, hi = _param_bounds(param_specs)
    lo_int = np.array(
        [np.log(l) if _is_log(s) else l for s, l in zip(param_specs, lo)],
        dtype=float)
    hi_int = np.array(
        [np.log(h) if _is_log(s) else h for s, h in zip(param_specs, hi)],
        dtype=float)
    return lo_int, hi_int


def _prior_widths(param_specs: List[ParamSpec], scale: float) -> np.ndarray:
    """Positive Gaussian-prior width (sigma) per parameter, in FIT space.

    Linear parameters scale by ``|init|`` so a signed (negative-init)
    parameter gets a positive width, falling back to half the (finite)
    bound range when ``init`` is ~0 so a zero-centred parameter still
    gets a finite prior and walker spread instead of a degenerate zero-
    width one. Log parameters get a constant width of ``scale`` in log-
    space — a log-normal prior whose one-sigma band spans the same
    multiplicative factor (e.g. 0.75 => x/2.1 .. x2.1) at every init,
    matching how scale-like physics parameters actually behave.
    """
    lo, hi = _param_bounds(param_specs)
    init_values = np.array([s.init_value for s in param_specs], dtype=float)
    sigma = np.abs(init_values) * scale
    finite = np.isfinite(lo) & np.isfinite(hi)
    fallback = np.where(finite, 0.5 * (hi - lo), 1.0)
    linear_sigma = np.where(sigma > 1e-9, sigma, fallback)
    is_log = np.array([_is_log(s) for s in param_specs], dtype=bool)
    return np.where(is_log, float(scale), linear_sigma)


def _lm_prefit(
    lm_fit_fn: Callable[[], Tuple[np.ndarray, Optional[np.ndarray]]],
    sse_fn: Callable[[Dict[str, float]], float],
    names: List[str],
    init_values: np.ndarray,
    noise_sigma: float,
    prior_sigma: np.ndarray,
    label: str,
    warm_start_breakdown_fn: Optional[Callable[[Dict[str, float]],
                                               None]] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Optional one-shot LM fit shared by both MCMC entry points.

    Three independent uses, each behind its own CFG flag (the fit runs
    once if any is set):

      * Hessian diagnostic — eigendecompose J^T J at the MAP
        (``code_sim_learning_log_hessian_identifiability``).
      * Warm start — center the MCMC walkers on theta_map
        (``code_sim_learning_warm_start_with_lm``).
      * Laplace ensemble — info-seeking exploration reuses J at the MAP
        for a calibrated posterior covariance, attached to the
        ``FitResult`` (``agent_explorer_info_seeking``).

    Returns ``(walker_center, lm_theta, lm_jac)``: the MCMC walker
    center (the LM MAP when warm-starting, else ``init_values``), the
    LM MAP itself, and the Jacobian at the MAP (the latter two ``None``
    when LM didn't run or failed). ``lm_fit_fn`` and ``sse_fn`` carry
    the per-transition vs recurrent specifics; the optional
    ``warm_start_breakdown_fn`` lets the per-transition caller add its
    ``log_sse_breakdown`` to the warm-start log.
    """
    walker_center = init_values
    lm_theta: Optional[np.ndarray] = None
    lm_jac: Optional[np.ndarray] = None
    if not (CFG.code_sim_learning_log_hessian_identifiability
            or CFG.code_sim_learning_warm_start_with_lm
            or CFG.agent_explorer_info_seeking):
        return walker_center, lm_theta, lm_jac
    theta_map, jac = lm_fit_fn()
    lm_theta = np.asarray(theta_map, dtype=float)
    if jac is not None and jac.size > 0:
        lm_jac = np.asarray(jac, dtype=float)
        if CFG.code_sim_learning_log_hessian_identifiability:
            log_hessian_identifiability(jac, names, noise_sigma, prior_sigma)
    if CFG.code_sim_learning_warm_start_with_lm:
        walker_center = lm_theta
        logger.info("Warm-starting %s MCMC walkers from LM MAP estimate.",
                    label)
        lm_params = {n: float(lm_theta[i]) for i, n in enumerate(names)}
        lm_sse = sse_fn(lm_params)
        logger.info(
            "After %s LM warm start — SSE: %.6f  log-likelihood: "
            "%.2f", label, lm_sse, -0.5 * lm_sse / (noise_sigma**2))
        if warm_start_breakdown_fn is not None:
            warm_start_breakdown_fn(lm_params)
    return walker_center, lm_theta, lm_jac


def _lm_point_fit_result(
    walker_center: np.ndarray,
    lm_theta: Optional[np.ndarray],
    lm_jac: Optional[np.ndarray],
    names: List[str],
    noise_sigma: float,
    prior_sigma: np.ndarray,
    label: str,
    scales: Optional[List[str]] = None,
) -> FitResult:
    """Single-point ``FitResult`` for the ``num_steps == 0`` short-circuit.

    Picks the point estimate the skipped-emcee run reports: the LM MAP
    when one is available and either warm-start or info-seeking asked
    for it (so the Laplace covariance is anchored where the data places
    it, not at init), else the initial parameter values. Carries the
    Laplace bundle through.
    """
    point = walker_center
    if (not CFG.code_sim_learning_warm_start_with_lm
            and CFG.agent_explorer_info_seeking and lm_theta is not None):
        point = lm_theta
        logger.info("Skipping emcee; using %s LM MAP for Laplace ensemble.",
                    label)
    elif CFG.code_sim_learning_warm_start_with_lm and lm_theta is not None:
        logger.info("Skipping emcee; using %s LM warm-start parameters.",
                    label)
    else:
        logger.info("Skipping emcee; using initial parameter values.")
    return FitResult(names,
                     point[None, :],
                     np.zeros(1),
                     jacobian=lm_jac,
                     noise_sigma=noise_sigma,
                     prior_sigma=prior_sigma,
                     scales=scales)


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


def compute_sse_recurrent(
    rules: List,
    trajectories: List[TrajectoryTriples],
    params: Dict[str, float],
    latent_init: Any,
    process_features: Dict[str, List[str]],
) -> float:
    """SSE on observables, with the ``latent`` block threaded per trajectory.

    Counterpart to :func:`compute_sse` for the recurrent
    (partially-observable) approach. Each input trajectory is a list
    of ``(base_state, action, next_obs)`` triples — the same shape
    individual transitions take in :func:`compute_sse`, but grouped
    so the latent block can carry across steps within a trajectory.

    For each trajectory:

    * Build an initial ``latent`` dict from ``latent_init`` (constants
      and any ``ParamSpec``-valued entries resolve from ``params``).
    * Roll forward step-by-step: call
      :func:`apply_rules_with_latent` with the running latent and the
      history prefix; merge the predicted observable feature updates;
      compare to the real next-step observation.
    * The "filter" step is implicit — ``base_state`` is the base sim
      applied to the *real* previous observation, so we re-ground
      observables each step automatically. Only ``latent`` propagates
      across step boundaries within a trajectory.

    Returns the total un-normalised SSE so the Gaussian log-likelihood
    ``-0.5 * SSE / noise_sigma**2`` is the correct iid form.
    """
    # pylint: disable=import-outside-toplevel
    from predicators.code_sim_learning.utils import apply_rules_with_latent, \
        init_latent

    # pylint: enable=import-outside-toplevel

    total_se = 0.0
    for traj in trajectories:
        latent: Dict[str, Any] = init_latent(latent_init, params)
        history: List[Tuple[State, Optional[Action]]] = []
        for state_base, action, state_obs in traj:
            history.append((state_base, action))
            updates = apply_rules_with_latent(state_base, latent, history,
                                              rules, params)

            for obj, feat_dict in updates.items():
                type_name = obj.type.name
                allowed_feats = process_features.get(type_name, [])
                for feat_name, pred_val in feat_dict.items():
                    if feat_name not in allowed_feats:
                        continue
                    v = pred_val.item() if hasattr(pred_val,
                                                   'item') else pred_val
                    obs_val = float(state_obs.get(obj, feat_name))
                    total_se += (v - obs_val)**2

            # Penalize unpredicted features (model predicts no change).
            for obj in state_base:
                type_name = obj.type.name
                for feat_name in process_features.get(type_name, []):
                    if obj in updates and feat_name in updates[obj]:
                        continue
                    pred_val = float(state_base.get(obj, feat_name))
                    obs_val = float(state_obs.get(obj, feat_name))
                    total_se += (pred_val - obs_val)**2

    return total_se


def fit_params_recurrent(
    rules: List,
    trajectories: List[TrajectoryTriples],
    param_specs: List[ParamSpec],
    latent_init: Any,
    process_features: Dict[str, List[str]],
    num_walkers: int = 32,
    num_steps: Optional[int] = None,
    burn_in: int = 200,
    noise_sigma: float = 0.05,
    prior_sigma_scale: float = 1.0,
) -> FitResult:
    """Fit recurrent-sim parameters via emcee MCMC.

    Mirror of :func:`fit_params` for the recurrent (latent-threaded)
    rollout used by the partial-observability approach. Differences
    from :func:`fit_params`:

    * Likelihood = :func:`compute_sse_recurrent` (per-trajectory
      rollout with latent carry) instead of per-transition
      :func:`compute_sse`.
    * Uses a recurrent LM warm-start / Hessian diagnostic / Laplace
      bundle (:func:`fit_map_lm_recurrent`, built on the rollout residual
      vector :func:`compute_residuals_recurrent`) under the same CFG flags
      as the FO path, in place of the per-transition :func:`fit_map_lm`.
      The Jacobian at the MAP is attached to the returned ``FitResult`` so
      callers can build the Laplace ensemble (see
      ``active_experiment.laplace_ensemble``).
    """
    names = [s.name for s in param_specs]
    scales = [getattr(s, "scale", "linear") for s in param_specs]
    init_values = np.array([s.init_value for s in param_specs])
    if num_steps is None:
        num_steps = CFG.code_sim_learning_num_mcmc_steps
    if num_steps < 0:
        raise ValueError("code_sim_learning_num_mcmc_steps must be "
                         "non-negative.")
    lo, hi = _internal_bounds(param_specs)
    init_int = _to_internal(param_specs, init_values)
    prior_sigma = _prior_widths(param_specs, prior_sigma_scale)

    # Optional one-shot recurrent LM fit (see _lm_prefit for its three
    # uses). Each residual eval here is a full set of per-trajectory
    # rollouts, so it is only paid when one of the gating flags is set.
    walker_center, lm_theta, lm_jac = _lm_prefit(
        lambda: fit_map_lm_recurrent(rules, trajectories, param_specs,
                                     latent_init, process_features),
        lambda p: compute_sse_recurrent(rules, trajectories, p, latent_init,
                                        process_features), names, init_values,
        noise_sigma, prior_sigma, "recurrent")

    if num_steps == 0:
        return _lm_point_fit_result(walker_center,
                                    lm_theta,
                                    lm_jac,
                                    names,
                                    noise_sigma,
                                    prior_sigma,
                                    "recurrent",
                                    scales=scales)

    import emcee  # type: ignore[import-untyped]  # pylint: disable=import-outside-toplevel

    ndim = len(param_specs)
    num_walkers = max(num_walkers, 2 * ndim + 2)
    burn_in = min(burn_in, max(num_steps - 1, 0))

    def log_posterior(theta: np.ndarray) -> float:
        # theta lives in the FIT space (log for log-scale params).
        if np.any(theta < lo) or np.any(theta > hi):
            return -np.inf
        ext = _to_external(param_specs, theta)
        params = {n: float(ext[i]) for i, n in enumerate(names)}
        log_prior = -0.5 * np.sum(((theta - init_int) / prior_sigma)**2)
        sse = compute_sse_recurrent(rules, trajectories, params, latent_init,
                                    process_features)
        return log_prior + (-0.5 * sse / (noise_sigma**2))

    p0 = _to_internal(param_specs, walker_center) + \
        0.5 * prior_sigma * np.random.randn(num_walkers, ndim)
    p0 = np.clip(p0, lo, hi)
    sampler = emcee.EnsembleSampler(num_walkers, ndim, log_posterior)
    logger.info("Running emcee (recurrent): %d walkers, %d steps, %d burn-in.",
                num_walkers, num_steps, burn_in)
    report_interval = 100
    for i, _result in enumerate(sampler.sample(p0, iterations=num_steps),
                                start=1):
        if i % report_interval == 0 or i == num_steps:
            best_lp = sampler.get_log_prob()[:i].max()
            logger.info("  emcee step %d/%d  (best log-prob: %.2f)", i,
                        num_steps, best_lp)
            for h in logger.handlers + logging.getLogger().handlers:
                h.flush()
    samples = _rows_to_external(param_specs,
                                sampler.get_chain(discard=burn_in, flat=True))
    log_probs = sampler.get_log_prob(discard=burn_in, flat=True)
    result = FitResult(names=names,
                       samples=samples,
                       log_probs=log_probs,
                       jacobian=lm_jac,
                       noise_sigma=noise_sigma,
                       prior_sigma=prior_sigma,
                       scales=scales)
    logger.info("emcee (recurrent) done. Posterior mean: %s",
                {k: f"{v:.4f}"
                 for k, v in result.point_estimate.items()})
    return result


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


def compute_residuals_recurrent(
    rules: List,
    trajectories: List[TrajectoryTriples],
    params: Dict[str, float],
    latent_init: Any,
    process_features: Dict[str, List[str]],
) -> np.ndarray:
    """Per-feature residuals (predicted - observed) for the recurrent rollout.

    Vector counterpart to :func:`compute_sse_recurrent`, written in the
    object x feature iteration order of :func:`compute_residuals` (not the
    predicted-then-unpredicted order of the SSE) so the flat vector keeps a
    fixed length and position across theta perturbations even when a hard
    gate flips which rule fires -- required for the finite-difference
    Jacobian LM builds. By construction
    ``sum(compute_residuals_recurrent(...)**2)`` equals
    ``compute_sse_recurrent(...)``, so minimizing ``0.5 * ||r||^2`` with LM
    targets the same MAP the recurrent MCMC samples around, and yields the
    Jacobian for the Hessian diagnostic and the Laplace ensemble.

    Each call is a full set of per-trajectory rollouts, so an LM
    finite-difference Jacobian costs ``O(num_params)`` of these.
    """
    # pylint: disable=import-outside-toplevel
    from predicators.code_sim_learning.utils import apply_rules_with_latent, \
        init_latent

    # pylint: enable=import-outside-toplevel

    residuals: List[float] = []
    for traj in trajectories:
        latent: Dict[str, Any] = init_latent(latent_init, params)
        history: List[Tuple[State, Optional[Action]]] = []
        for state_base, action, state_obs in traj:
            history.append((state_base, action))
            updates = apply_rules_with_latent(state_base, latent, history,
                                              rules, params)
            for obj in state_base:
                type_name = obj.type.name
                for feat_name in process_features.get(type_name, []):
                    if obj in updates and feat_name in updates[obj]:
                        raw = updates[obj][feat_name]
                        pred = raw.item() if hasattr(raw,
                                                     'item') else float(raw)
                    else:
                        pred = float(state_base.get(obj, feat_name))
                    obs = float(state_obs.get(obj, feat_name))
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

    Three callers (see ``fit_simulator_params``):
      * Hessian identifiability diagnostic — eigendecompose J^T J.
      * MCMC warm start — center emcee walkers on theta_map (and short-
        circuit to it directly when ``num_mcmc_steps == 0``).
      * Laplace ensemble — reuse J at the MAP for a calibrated posterior
        covariance (see ``active_experiment.laplace_ensemble``).
    """
    names = [s.name for s in param_specs]

    def residuals_fn(theta: np.ndarray) -> np.ndarray:
        params = {n: float(theta[i]) for i, n in enumerate(names)}
        return compute_residuals(simulator_fn, transitions, params,
                                 process_features)

    return _solve_lm(residuals_fn, param_specs, max_nfev, "per-transition")


def _solve_lm(
    residuals_fn: Callable[[np.ndarray], np.ndarray],
    param_specs: List[ParamSpec],
    max_nfev: int,
    label: str,
    diff_step: Optional[float] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Shared Levenberg-Marquardt core for the per-transition and recurrent MAP
    fits.

    Solves ``min_theta 0.5 * ||residuals_fn(theta)||^2`` with
    ``scipy.optimize.least_squares(method='trf')`` under the
    ``param_specs`` box, and returns ``(theta_map, jacobian_at_optimum)``.
    The Jacobian is ``None`` when the residual vector is empty or LM
    raises. ``label`` only tags the log lines (e.g. ``per-transition`` vs
    ``recurrent``). The single residual-vector seam is what lets the
    recurrent fit reuse this unchanged.

    ``diff_step`` overrides the relative finite-difference step for the
    numerical Jacobian (scipy's default ~sqrt(machine eps)). Residuals
    produced by a physics *simulation* need a much coarser step: a 1e-8
    relative perturbation of e.g. a friction coefficient is below the
    contact solver's sensitivity, the rollout comes back bitwise
    identical, and the Jacobian is identically zero — LM would stall at
    init. Analytic-formula residuals (the per-transition and recurrent
    fits) leave this ``None``.

    The optimizer runs in the FIT space (``z = log(theta)`` for
    log-scale params): ``residuals_fn`` still receives external theta,
    the returned MAP is external, but the returned Jacobian is
    ``dr/dz`` — consistent with the fit-space ``prior_sigma`` every
    downstream consumer (Hessian diagnostic, Laplace ensemble,
    identifiability probe) pairs it with. In fit space the relative
    ``diff_step`` also becomes a *multiplicative* theta perturbation
    for log params, so the finite-difference gradient stays equally
    informative across decades instead of vanishing at the low end.
    """
    from scipy.optimize import \
        least_squares  # pylint: disable=import-outside-toplevel

    names = [s.name for s in param_specs]
    init_ext = np.array([s.init_value for s in param_specs], dtype=float)
    init = _to_internal(param_specs, init_ext)
    lo, hi = _internal_bounds(param_specs)
    # Nudge init strictly into the interior so trf doesn't reject it.
    init = np.maximum(init, lo + 1e-9)
    safe_hi = np.where(np.isfinite(hi), hi - 1e-9, np.inf)
    init = np.minimum(init, safe_hi)

    def internal_residuals(z: np.ndarray) -> np.ndarray:
        return residuals_fn(_to_external(param_specs, z))

    init_residuals = internal_residuals(init)
    if init_residuals.size == 0:
        logger.warning(
            "No residuals to fit (empty process_features); "
            "skipping %s LM fit.", label)
        return _to_external(param_specs, init), None

    sse_init = float(np.sum(init_residuals**2))

    try:
        result = least_squares(internal_residuals,
                               init,
                               method='trf',
                               bounds=(lo, hi),
                               diff_step=diff_step,
                               max_nfev=max_nfev)
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("%s LM fit raised %s; skipping.", label, exc)
        return _to_external(param_specs, init), None

    sse_lm = float(2.0 * result.cost)
    x_ext = _to_external(param_specs, result.x)
    delta = {
        names[i]: float(x_ext[i] - init_ext[i])
        for i in range(len(names))
    }
    logger.info("%s LM fit: SSE %.4f -> %.4f in %d fn-evals (status=%d, %s).",
                label, sse_init, sse_lm, result.nfev, result.status,
                "converged" if result.success else "max-evals")
    logger.info("%s LM theta_map - init: %s", label,
                {k: f"{v:+.4f}"
                 for k, v in delta.items()})

    jac = np.asarray(result.jac, dtype=float)
    if jac.size == 0:
        return x_ext, None
    return x_ext, jac


def fit_map_lm_recurrent(
    rules: List,
    trajectories: List[TrajectoryTriples],
    param_specs: List[ParamSpec],
    latent_init: Any,
    process_features: Dict[str, List[str]],
    max_nfev: int = 200,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Levenberg-Marquardt MAP fit for the recurrent (latent-threaded) sim.

    Recurrent counterpart to :func:`fit_map_lm`, sharing the same
    :func:`_solve_lm` core; only the residual vector differs — here it
    comes from :func:`compute_residuals_recurrent` (a full latent rollout
    per evaluation) rather than the per-transition residuals. Returns
    ``(theta_map, jacobian-at-optimum)`` under the ParamSpec ``[lo, hi]``
    box; the Jacobian is ``None`` when residuals are empty or LM raises.

    Same smoothness caveat as the FO path: the finite-difference Jacobian
    is only informative where the likelihood is smooth. A hard-gated
    parameter with no boundary-crossing data has a near-zero column in J,
    so LM leaves it at init -- but the Hessian diagnostic then surfaces it
    as a flat (unidentifiable) direction rather than a confident wrong
    value.

    Cost note: every residual evaluation is a full set of per-trajectory
    rollouts, so the finite-difference Jacobian costs ``O(num_params)``
    rollouts per LM iteration; for large param sets prefer MCMC. And
    because latent threading correlates residuals across steps, ``J^T J``
    ignores that coupling, making the recurrent Laplace covariance a
    slightly looser approximation than the per-transition one (MCMC at
    ``num_mcmc_steps > 0`` remains the gold path).
    """
    names = [s.name for s in param_specs]

    def residuals_fn(theta: np.ndarray) -> np.ndarray:
        params = {n: float(theta[i]) for i, n in enumerate(names)}
        return compute_residuals_recurrent(rules, trajectories, params,
                                           latent_init, process_features)

    return _solve_lm(residuals_fn, param_specs, max_nfev, "recurrent")


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
    scales = [getattr(s, "scale", "linear") for s in param_specs]
    init_values = np.array([s.init_value for s in param_specs])
    if num_steps is None:
        num_steps = CFG.code_sim_learning_num_mcmc_steps
    if num_steps < 0:
        raise ValueError("code_sim_learning_num_mcmc_steps must be "
                         "non-negative.")
    lo, hi = _internal_bounds(param_specs)
    init_int = _to_internal(param_specs, init_values)
    prior_sigma = _prior_widths(param_specs, prior_sigma_scale)

    # Optional one-shot LM fit (see _lm_prefit for its three uses).
    walker_center, lm_theta, lm_jac = _lm_prefit(
        lambda: fit_map_lm(simulator_fn, transitions, param_specs,
                           process_features),
        lambda p: compute_sse(simulator_fn, transitions, p, process_features),
        names,
        init_values,
        noise_sigma,
        prior_sigma,
        "per-transition",
        warm_start_breakdown_fn=lambda p: log_sse_breakdown(simulator_fn,
                                                            transitions,
                                                            p,
                                                            process_features,
                                                            label=
                                                            "lm-warm-start"))

    if num_steps == 0:
        return _lm_point_fit_result(walker_center,
                                    lm_theta,
                                    lm_jac,
                                    names,
                                    noise_sigma,
                                    prior_sigma,
                                    "per-transition",
                                    scales=scales)

    import emcee  # type: ignore[import-untyped]  # pylint: disable=import-outside-toplevel

    ndim = len(param_specs)
    num_walkers = max(num_walkers, 2 * ndim + 2)
    burn_in = min(burn_in, max(num_steps - 1, 0))

    def log_posterior(theta: np.ndarray) -> float:
        # theta lives in the FIT space (log for log-scale params).
        # Reject samples outside the per-parameter [lo, hi] box.
        if np.any(theta < lo) or np.any(theta > hi):
            return -np.inf
        ext = _to_external(param_specs, theta)
        params = {n: float(ext[i]) for i, n in enumerate(names)}
        # Broad Gaussian prior centered on init values
        log_prior = -0.5 * np.sum(((theta - init_int) / prior_sigma)**2)
        # Likelihood
        sse = compute_sse(simulator_fn, transitions, params, process_features)
        return log_prior + (-0.5 * sse / (noise_sigma**2))

    # Initialize walkers across the prior support (sigma = half the prior
    # width). A tight ball around init traps the chain on flat plateaus
    # of the likelihood (e.g., when threshold-based rules don't fire),
    # because emcee stretch moves scale with the swarm's spread.
    p0 = _to_internal(param_specs, walker_center) + \
        0.5 * prior_sigma * np.random.randn(num_walkers, ndim)
    p0 = np.clip(p0, lo, hi)

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

    # Discard burn-in, flatten chains (back to external units).
    samples = _rows_to_external(param_specs,
                                sampler.get_chain(discard=burn_in, flat=True))
    log_probs = sampler.get_log_prob(discard=burn_in, flat=True)

    result = FitResult(names=names,
                       samples=samples,
                       log_probs=log_probs,
                       jacobian=lm_jac,
                       noise_sigma=noise_sigma,
                       prior_sigma=prior_sigma,
                       scales=scales)

    logger.info("emcee done. Posterior mean: %s",
                {k: f"{v:.4f}"
                 for k, v in result.point_estimate.items()})

    return result
