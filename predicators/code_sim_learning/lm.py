"""Shared Levenberg-Marquardt core and its optional pre-fit wrappers.

``solve_lm`` minimizes a residual vector under a ``ParamSpec`` box in
the fit space; ``lm_prefit`` / ``lm_point_fit_result`` implement the
CFG-gated "LM before (or instead of) MCMC" flow shared by the three fit
entry points in :mod:`fitting` and :mod:`physical_sysid`;
``log_hessian_identifiability`` is the eigenanalysis diagnostic that
reuses the LM Jacobian.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from predicators.code_sim_learning.fit_space import FitResult, ParamSpec, \
    fit_space_bounds, from_fit_space, scalar_from_fit_space, to_fit_space
from predicators.settings import CFG

logger = logging.getLogger(__name__)


def lm_prefit(
    lm_fit_fn: Callable[[], Tuple[np.ndarray, Optional[np.ndarray]]],
    sse_fn: Callable[[Dict[str, float]], float],
    names: List[str],
    init_values: np.ndarray,
    noise_sigma: float,
    prior_sigma: np.ndarray,
    label: str,
    warm_start_breakdown_fn: Optional[Callable[[Dict[str, float]],
                                               None]] = None,
    precomputed: Optional[Tuple[np.ndarray, Optional[np.ndarray]]] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Optional one-shot LM fit shared by the three fit entry points (per-
    transition, recurrent, and rollout).

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

    ``precomputed`` short-circuits the LM run with an already-computed
    ``(theta_map, jac)`` from an identical earlier fit (same objective:
    rules, specs, data). The Hessian diagnostic and warm-start SSE logs
    are skipped too - the earlier fit already emitted them. LM residual
    evals are full rollout passes at recurrent scale, so a caller that
    just ran the fit must not pay for it twice.
    """
    walker_center = init_values
    lm_theta: Optional[np.ndarray] = None
    lm_jac: Optional[np.ndarray] = None
    if not (CFG.code_sim_learning_log_hessian_identifiability
            or CFG.code_sim_learning_warm_start_with_lm
            or CFG.agent_explorer_info_seeking):
        return walker_center, lm_theta, lm_jac
    if precomputed is not None:
        theta_map, jac = precomputed
    else:
        theta_map, jac = lm_fit_fn()
    lm_theta = np.asarray(theta_map, dtype=float)
    if jac is not None and jac.size > 0:
        lm_jac = np.asarray(jac, dtype=float)
        if (precomputed is None
                and CFG.code_sim_learning_log_hessian_identifiability):
            log_hessian_identifiability(jac, names, noise_sigma, prior_sigma)
    if CFG.code_sim_learning_warm_start_with_lm:
        walker_center = lm_theta
        if precomputed is not None:
            logger.info(
                "Reusing the earlier LM MAP as the %s MCMC warm start "
                "(LM refit skipped).", label)
        else:
            logger.info("Warm-starting %s MCMC walkers from LM MAP estimate.",
                        label)
            lm_params = {n: float(lm_theta[i]) for i, n in enumerate(names)}
            lm_sse = sse_fn(lm_params)
            logger.info(
                "After %s LM warm start - SSE: %.6f  log-likelihood: "
                "%.2f", label, lm_sse, -0.5 * lm_sse / (noise_sigma**2))
            if warm_start_breakdown_fn is not None:
                warm_start_breakdown_fn(lm_params)
    return walker_center, lm_theta, lm_jac


def lm_point_fit_result(
    walker_center: np.ndarray,
    lm_theta: Optional[np.ndarray],
    lm_jac: Optional[np.ndarray],
    names: List[str],
    noise_sigma: float,
    prior_sigma: np.ndarray,
    label: str,
    scales: Optional[List[str]] = None,
    lm_notes: Optional[List[str]] = None,
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
                     scales=scales,
                     lm_notes=list(lm_notes or []))


# Bracket search for zero-gradient (threshold/gate) parameters: grid
# points across the box per parameter, golden-section refinements
# between the best point's neighbours, and the relative SSE change
# below which a box counts as flat / a move as no improvement.
_GATE_GRID_POINTS = 9
_GATE_REFINE_ITERS = 6
_GATE_MIN_REL_IMPROVEMENT = 1e-6


def zero_jacobian_columns(jac: np.ndarray) -> List[int]:
    """Indices of parameters whose Jacobian column is identically zero."""
    if jac.ndim != 2 or jac.size == 0:
        return []
    return [j for j in range(jac.shape[1]) if not np.any(jac[:, j] != 0.0)]


def bracket_search_zero_gradient_params(
    sse_fn: Callable[[np.ndarray], float],
    z: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    cols: List[int],
    param_specs: List[ParamSpec],
    label: str,
) -> Tuple[np.ndarray, float, List[str]]:
    """Coordinate-wise bracket search (fit space) for the parameters in
    ``cols``, holding every other parameter fixed.

    A threshold or gate parameter (a bond-gap tolerance, a contact
    distance, a cure delay) changes the residuals only when it crosses
    a data point, so its finite-difference LM gradient is exactly zero
    almost everywhere and LM "converges" in one evaluation without
    moving it - on the 2026-08-27 bridge runs every parameter of every
    cycle's fit stayed at its init this way, and all calibration fell
    to hand-bracketing by the agent. This search evaluates the SSE on
    a grid across the parameter's box, refines around the best grid
    point by golden section, and keeps a move only when it lowers the
    SSE. Returns ``(z, sse, notes)`` with one note per parameter
    saying what happened, in external units, for the agent-facing
    report.
    """
    z = np.array(z, dtype=float)
    sse = float(sse_fn(z))
    notes: List[str] = []
    for j in cols:
        spec = param_specs[j]
        init_ext = scalar_from_fit_space(spec, float(z[j]))
        if not (np.isfinite(lo[j]) and np.isfinite(hi[j])) or hi[j] <= lo[j]:
            notes.append(f"{spec.name}: LM gradient is zero and its box is "
                         "unbounded, so no bracket search ran; kept at "
                         f"{init_ext:.4g} (NOT fit from data).")
            continue
        grid = np.linspace(lo[j], hi[j], _GATE_GRID_POINTS)
        vals = []
        for g in grid:
            zz = z.copy()
            zz[j] = g
            vals.append(float(sse_fn(zz)))
        span = max(vals) - min(vals)
        if span <= _GATE_MIN_REL_IMPROVEMENT * max(sse, 1e-12):
            notes.append(f"{spec.name}: SSE is flat across its whole box "
                         f"({_GATE_GRID_POINTS} points), so the data do not "
                         f"constrain it; kept at {init_ext:.4g} (NOT fit "
                         "from data).")
            continue
        best = int(np.argmin(vals))
        best_z, best_sse = float(grid[best]), vals[best]
        a = float(grid[max(best - 1, 0)])
        b = float(grid[min(best + 1, _GATE_GRID_POINTS - 1)])
        phi = (np.sqrt(5.0) - 1.0) / 2.0
        x1 = b - phi * (b - a)
        x2 = a + phi * (b - a)

        def _eval(x: float, col: int = j) -> float:
            zz = z.copy()
            zz[col] = x
            return float(sse_fn(zz))

        f1, f2 = _eval(x1), _eval(x2)
        for _ in range(_GATE_REFINE_ITERS):
            if f1 < f2:
                b, x2, f2 = x2, x1, f1
                x1 = b - phi * (b - a)
                f1 = _eval(x1)
            else:
                a, x1, f1 = x1, x2, f2
                x2 = a + phi * (b - a)
                f2 = _eval(x2)
        for x, f in ((x1, f1), (x2, f2)):
            if f < best_sse:
                best_z, best_sse = x, f
        if best_sse < sse * (1.0 - _GATE_MIN_REL_IMPROVEMENT):
            new_ext = scalar_from_fit_space(spec, best_z)
            notes.append(f"{spec.name}: LM gradient is zero (threshold-like); "
                         f"bracket search over its box moved it "
                         f"{init_ext:.4g} -> {new_ext:.4g} (SSE {sse:.4g} -> "
                         f"{best_sse:.4g}).")
            z[j] = best_z
            sse = best_sse
        else:
            notes.append(f"{spec.name}: LM gradient is zero; bracket search "
                         f"over its box found nothing better than "
                         f"{init_ext:.4g} (SSE {sse:.4g}); kept.")
    logger.info("%s bracket search on %d zero-gradient parameter(s):\n  %s",
                label, len(cols), "\n  ".join(notes))
    return z, sse, notes


def solve_lm(
    residuals_fn: Callable[[np.ndarray], np.ndarray],
    param_specs: List[ParamSpec],
    max_nfev: int,
    label: str,
    diff_step: Optional[float] = None,
    notes_out: Optional[List[str]] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Shared Levenberg-Marquardt core for the per-transition, recurrent, and
    rollout (``physical_sysid``) MAP fits.

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

    Parameters whose Jacobian column at the LM optimum is identically
    zero (threshold/gate parameters, whose finite-difference gradient
    exists only where a data point is crossed) get a coordinate-wise
    bracket search over their box (:func:`bracket_search_zero_gradient_
    params`); if any moves, LM is re-run from the new point so the
    smooth parameters re-adapt. ``notes_out`` collects one line per
    such parameter for the agent-facing fit report.
    """
    from scipy.optimize import \
        least_squares  # pylint: disable=import-outside-toplevel

    names = [s.name for s in param_specs]
    init_ext = np.array([s.init_value for s in param_specs], dtype=float)
    init = to_fit_space(param_specs, init_ext)
    lo, hi = fit_space_bounds(param_specs)
    # Nudge init strictly into the interior so trf doesn't reject it.
    init = np.maximum(init, lo + 1e-9)
    safe_hi = np.where(np.isfinite(hi), hi - 1e-9, np.inf)
    init = np.minimum(init, safe_hi)

    def internal_residuals(z: np.ndarray) -> np.ndarray:
        return residuals_fn(from_fit_space(param_specs, z))

    init_residuals = internal_residuals(init)
    if init_residuals.size == 0:
        logger.warning(
            "No residuals to fit (empty residual_features); "
            "skipping %s LM fit.", label)
        return from_fit_space(param_specs, init), None

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
        return from_fit_space(param_specs, init), None

    sse_lm = float(2.0 * result.cost)
    logger.info("%s LM fit: SSE %.4f -> %.4f in %d fn-evals (status=%d, %s).",
                label, sse_init, sse_lm, result.nfev, result.status,
                "converged" if result.success else "max-evals")

    jac = np.asarray(result.jac, dtype=float)
    zero_cols = zero_jacobian_columns(jac)
    if zero_cols:

        def _sse(z: np.ndarray) -> float:
            return float(np.sum(internal_residuals(z)**2))

        z_new, sse_new, notes = bracket_search_zero_gradient_params(
            _sse, result.x, lo, hi, zero_cols, param_specs, label)
        if notes_out is not None:
            notes_out.extend(notes)
        if sse_new < sse_lm:
            # The moved gates may have given the smooth parameters a
            # gradient: polish from the new point. A failure here keeps
            # the searched point (better than the LM one by construction).
            try:
                polished = least_squares(internal_residuals,
                                         z_new,
                                         method='trf',
                                         bounds=(lo, hi),
                                         diff_step=diff_step,
                                         max_nfev=max_nfev)
                if float(2.0 * polished.cost) <= sse_new:
                    result = polished
                    jac = np.asarray(result.jac, dtype=float)
                else:
                    result.x = z_new
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning(
                    "%s LM polish after bracket search raised "
                    "%s; keeping the searched point.", label, exc)
                result.x = z_new
            sse_lm = min(float(2.0 * result.cost), sse_new)
            logger.info("%s LM fit after bracket search: SSE %.4f.", label,
                        sse_lm)

    x_ext = from_fit_space(param_specs, result.x)
    delta = {
        names[i]: float(x_ext[i] - init_ext[i])
        for i in range(len(names))
    }
    logger.info("%s LM theta_map - init: %s", label,
                {k: f"{v:+.4f}"
                 for k, v in delta.items()})
    if jac.size == 0:
        return x_ext, None
    return x_ext, jac


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
