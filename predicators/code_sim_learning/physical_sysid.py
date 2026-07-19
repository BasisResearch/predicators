"""System identification of PyBullet physical parameters by rollout matching.

The process-rule fitting in :mod:`predicators.code_sim_learning.fitting` is
*teacher-forced and single-step*: it resets the base sim to each observed
``s_t`` and predicts one step. That is correct for slow process features
(heating, filling) but wrong for **momentum-driven** dynamics such as a domino
cascade: the :class:`~predicators.structs.State` carries pose but no velocity,
so resetting to a mid-cascade state discards the angular momentum that produced
the next step, and the one-step prediction systematically under-rotates. No
friction/restitution value can repair that mismatch — physical parameters are
*invisible* to the teacher-forced objective.

This stack fits such parameters by matching a **free-running rollout**
instead: reset *once* to a trajectory's at-rest initial state, roll the base
sim forward under the recorded action sequence (so momentum accrues in-sim),
and compare the full per-step pose trajectory. Because the base sim and the
real environment are the same engine differing only in the fitted scalars, the
sum-of-squared-errors is ~0 at the true parameters, giving a well-specified
identification problem.

Design notes (mirroring MuJoCo's official ``mujoco.sysid`` toolbox):

* The agent declares a **sparse subset** of the parameters the env reveals
  (``env.get_physical_param_info()``) as ``PHYSICAL_PARAMS`` — never "all
  params by default". Undeclared parameters keep the env's built-in values.
* Physical parameters and learned-rule parameters are fit **jointly** in one
  posterior (one theta vector, one MCMC run) so rules cannot silently absorb
  physics error and vice versa. With no rules the fit degenerates to pure
  physical identification; rule-only artifacts keep using the (cheaper)
  teacher-forced / recurrent objectives in ``fitting.py``.
* Non-identifiability is *reported*, not regularized away: the posterior
  contraction per parameter (:func:`identifiability_report`) is surfaced to
  the agent so it can drop null parameters from its declaration.

The interface deliberately mirrors :func:`training.fit_params` (declare an
initialization via :class:`ParamSpec`, a Gaussian prior around it, a Gaussian
likelihood, emcee MCMC) so the agent-facing flow is unchanged: pick an init,
let the solver refine it.

The stack is split across subsystem modules; this module keeps the fit
orchestrators and re-exports the public API so existing importers keep
working:

* :mod:`.config` - :class:`SysIdConfig`, the frozen snapshot of the
  ``code_sim_learning_*`` flags (resolved from ``CFG`` at entry time).
* :mod:`.rollout_env` - env-facing rollout plumbing:
  ``RolloutTrajectory``, :func:`rollout_states`, :func:`dispose_env`,
  :func:`physical_param_anchors`.
* :mod:`.trajectory_prep` - :func:`truncate_settled_tail`,
  :func:`split_at_rest_points`, :class:`ResidualScaling`,
  :func:`compute_residual_scaling`.
* :mod:`.rollout_objective` - :func:`compute_rollout_sse`,
  :func:`compute_rollout_residuals`, :func:`per_trajectory_rms`,
  :func:`fit_map_lm_rollout`.
* :mod:`.grid_seed` - the coordinate grid sweeps (LM-seed relocation and
  the :func:`min_explainable_rms` explainability sweep).
* :mod:`.identifiability` - :func:`identifiability_report`,
  :func:`select_trustworthy_params`, :func:`format_identifiability`.
* This module - :func:`fit_params_rollout` (grid seed + LM MAP + optional
  emcee posterior) and :func:`fit_params_rollout_trimmed` (explainability
  trimming + consistency loop).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from predicators.code_sim_learning.config import SysIdConfig
from predicators.code_sim_learning.fit_space import FitResult, ParamSpec, \
    fit_space_bounds, prior_widths, to_fit_space
from predicators.code_sim_learning.fitting import run_emcee_posterior
from predicators.code_sim_learning.grid_seed import \
    _grid_seed_physical_specs, min_explainable_rms
from predicators.code_sim_learning.identifiability import NOISE_FLOOR_EVALS, \
    format_identifiability, identifiability_report, \
    select_trustworthy_params
from predicators.code_sim_learning.lm import lm_point_fit_result, lm_prefit
from predicators.code_sim_learning.rollout_env import RolloutTrajectory, \
    dispose_env, physical_param_anchors, rollout_states
from predicators.code_sim_learning.rollout_objective import \
    compute_rollout_residuals, compute_rollout_sse, fit_map_lm_rollout, \
    per_trajectory_rms
from predicators.code_sim_learning.trajectory_prep import ResidualScaling, \
    compute_residual_scaling, split_at_rest_points, truncate_settled_tail

logger = logging.getLogger(__name__)

# Re-exported public API of the split modules (see the module map in the
# docstring), so existing importers of this module keep working.
__all__ = [
    "RolloutTrajectory",
    "ResidualScaling",
    "SysIdConfig",
    "compute_residual_scaling",
    "compute_rollout_residuals",
    "compute_rollout_sse",
    "dispose_env",
    "fit_map_lm_rollout",
    "fit_params_rollout",
    "fit_params_rollout_trimmed",
    "format_identifiability",
    "identifiability_report",
    "min_explainable_rms",
    "per_trajectory_rms",
    "physical_param_anchors",
    "rollout_states",
    "select_trustworthy_params",
    "split_at_rest_points",
    "truncate_settled_tail",
]

# Prior width as a fraction of each param's box; shared by the rollout
# fit default and the pinned-at-init fallback result so the two cannot
# silently diverge.
_ROLLOUT_PRIOR_SIGMA_SCALE = 0.75

# Absolute RMS slack in the trim-consistency test, so exact ties and
# floating-point jitter never count as violations.
_CONSISTENCY_RMS_EPS = 1e-3


def fit_params_rollout(
    base_env: Any,
    trajectories: List[RolloutTrajectory],
    physical_specs: Sequence[ParamSpec],
    process_features: Dict[str, List[str]],
    rules: Sequence[Any] = (),
    rule_specs: Sequence[ParamSpec] = (),
    latent_init: Any = None,
    num_walkers: int = 8,
    num_steps: Optional[int] = None,
    burn_in: int = 40,
    noise_sigma: float = 0.05,
    prior_sigma_scale: float = _ROLLOUT_PRIOR_SIGMA_SCALE,
    scaling: Optional[ResidualScaling] = None,
    anchors: Optional[Dict[str, float]] = None,
    config: Optional[SysIdConfig] = None,
) -> FitResult:
    """Jointly identify physical + rule params against rollout SSE.

    Mirrors :func:`training.fit_params` end to end: Gaussian prior,
    Gaussian likelihood ``-0.5*SSE/noise^2``, shared bounds/prior-width
    conventions, the same LM-prefit flow (Hessian diagnostic / warm
    start / Laplace bundle behind the usual CFG flags), and the same
    ``num_steps == 0`` short-circuit to the **LM point fit** — which is
    the experiment default (``CFG.code_sim_learning_rollout_num_mcmc_
    steps = 0``, matching the configs' ``code_sim_learning_num_mcmc_
    steps: 0``); MCMC is opt-in. The forward model is a base-sim
    rollout (:func:`compute_rollout_sse`) rather than the per-step
    process rules, and theta concatenates ``physical_specs`` with
    ``rule_specs``.

    The Gaussian prior is centered on each param's ``anchors`` entry
    (the env-registry baseline, see :func:`physical_param_anchors`)
    when given, else on the declared init, and - unlike the historical
    likelihood-only LM - is folded into the LM objective itself
    (:func:`fit_map_lm_rollout`), so data-flat directions stay at their
    anchors instead of drifting on rollout noise.

    When the grid sweep ran, each physical param's sweep diagnostics
    (SSE span, same-theta noise floor from 3 repeated evaluations at
    the anchor point, and the data-equivalent ``flat_interval``) are
    attached as ``FitResult.sensitivity``. With
    ``code_sim_learning_rollout_sensitivity_factor`` > 0 they also
    drive the pre-fit sensitivity screen: a param whose span does not
    clear the noise floor is "insensitive" and its fitted value is
    withheld by :func:`select_trustworthy_params`.

    Serial only: a shared ``base_env`` instance is mutated per
    evaluation (and a factory-built fresh env lives inside one
    :func:`rollout_states` call), so no multiprocessing pool may be
    used.
    """
    config = config or SysIdConfig.from_cfg()
    all_specs = list(physical_specs) + list(rule_specs)
    assert all_specs, "fit_params_rollout needs at least one ParamSpec."
    physical_names = [s.name for s in physical_specs]
    names = [s.name for s in all_specs]
    scales = [getattr(s, "scale", "linear") for s in all_specs]
    init_values = np.array([s.init_value for s in all_specs], dtype=float)
    if num_steps is None:
        num_steps = config.rollout_num_mcmc_steps
    if num_steps < 0:
        raise ValueError("code_sim_learning_rollout_num_mcmc_steps must be "
                         "non-negative.")
    lo, hi = fit_space_bounds(all_specs)
    anchors = anchors or {}
    center_values = np.array(
        [anchors.get(s.name, s.init_value) for s in all_specs], dtype=float)
    # Prior widths are computed at the CENTER values (the anchor spec),
    # so a linear param's width scales with its anchor, not with
    # whatever init the agent declared this call.
    center_specs = [
        ParamSpec(s.name,
                  float(c),
                  lo=s.lo,
                  hi=s.hi,
                  scale=getattr(s, "scale", "linear"))
        for s, c in zip(all_specs, center_values)
    ]
    center_int = to_fit_space(all_specs, center_values)
    prior_sigma = prior_widths(center_specs, prior_sigma_scale)

    # Coarse grid sweep to place the LM start in the right basin (see
    # _grid_seed_physical_specs for why LM alone can stall). Also
    # yields the per-param SSE spans and data-equivalent flat intervals
    # the sensitivity screen and the identifiability report consume.
    lm_physical_specs = list(physical_specs)
    sensitivity: Optional[Dict[str, Dict[str, Any]]] = None
    if (config.grid_seed_points > 0 and trajectories):
        # Same-theta noise floor at the anchor point (3 repeated evals).
        # Fresh-env rollouts are deterministic so this is normally 0.0;
        # it bounds the sweep's flat tolerance and the sensitivity
        # screen from below if nondeterminism ever returns.
        anchor_point = {
            s.name: anchors.get(s.name, s.init_value)
            for s in all_specs
        }
        floor_evals = [
            compute_rollout_sse(base_env, trajectories, anchor_point,
                                process_features, physical_names, rules,
                                latent_init, scaling)
            for _ in range(NOISE_FLOOR_EVALS)
        ]
        noise_floor = float(np.max(floor_evals) - np.min(floor_evals))
        lm_physical_specs, sweep_info = _grid_seed_physical_specs(
            base_env,
            trajectories,
            physical_specs,
            process_features,
            rules,
            rule_specs,
            latent_init,
            scaling=scaling,
            anchors=anchors,
            noise_floor=noise_floor,
            config=config)
        sens_factor = config.sensitivity_factor
        sensitivity = {}
        for name, info in sweep_info.items():
            entry: Dict[str, Any] = {
                "sse_span": info["span"],
                "noise_floor": noise_floor,
                "flat_interval": info["flat_interval"],
            }
            if sens_factor > 0:
                entry["sensitive"] = (info["span"] >
                                      sens_factor * max(noise_floor, 1e-12))
            sensitivity[name] = entry
        insensitive = sorted(n for n, d in sensitivity.items()
                             if not d.get("sensitive", True))
        if insensitive:
            logger.info(
                "Rollout sysID sensitivity screen: rollouts do not "
                "respond to %s anywhere in their boxes on this data "
                "(SSE span vs %.1f x noise floor %.4g); their fitted "
                "values will not be applied.", insensitive, sens_factor,
                noise_floor)

    # One-shot rollout LM fit (see lm_prefit for its three uses). With
    # the default rollout MCMC budget of 0 this LM MAP *is* the fit.
    walker_center, lm_theta, lm_jac = lm_prefit(
        lambda: fit_map_lm_rollout(base_env,
                                   trajectories,
                                   lm_physical_specs,
                                   process_features,
                                   rules,
                                   rule_specs,
                                   latent_init,
                                   scaling=scaling,
                                   prior_centers=center_int,
                                   prior_sigmas=prior_sigma,
                                   noise_sigma=noise_sigma), lambda p:
        compute_rollout_sse(base_env, trajectories, p, process_features,
                            physical_names, rules, latent_init, scaling),
        names, init_values, noise_sigma, prior_sigma, "rollout")

    if num_steps == 0:
        result = lm_point_fit_result(walker_center,
                                     lm_theta,
                                     lm_jac,
                                     names,
                                     noise_sigma,
                                     prior_sigma,
                                     "rollout",
                                     scales=scales)
        result.sensitivity = sensitivity
        return result

    logger.info(
        "Rollout sysID emcee: %d walkers, %d steps, %d burn-in "
        "(%d physical + %d rule params, %d trajectories).",
        max(num_walkers, 2 * len(all_specs) + 2), num_steps,
        min(burn_in, max(num_steps - 1, 0)), len(physical_names),
        len(list(rule_specs)), len(trajectories))
    samples, log_probs = run_emcee_posterior(
        list(all_specs),
        lambda p:
        compute_rollout_sse(base_env, trajectories, p, process_features,
                            physical_names, rules, latent_init, scaling),
        walker_center,
        center_int,
        prior_sigma,
        lo,
        hi,
        noise_sigma,
        num_walkers,
        num_steps,
        burn_in,
        label="rollout",
        # Rollout evaluations are ~100x costlier than analytic SSEs, so
        # report much more often.
        report_interval=25)
    result = FitResult(names=names,
                       samples=samples,
                       log_probs=log_probs,
                       jacobian=lm_jac,
                       noise_sigma=noise_sigma,
                       prior_sigma=prior_sigma,
                       scales=scales,
                       sensitivity=sensitivity)
    logger.info("Rollout sysID done. MAP: %s",
                {k: f"{v:.4f}"
                 for k, v in result.point_estimate.items()})
    return result


def _init_point_fit_result(all_specs: Sequence[ParamSpec],
                           noise_sigma: float) -> FitResult:
    """A single-sample FitResult pinned at the declared init values.

    Returned when trimming rejects every trajectory: no explainable data
    exists, so no fitted value — not even the pooled fit's — may leak to
    the caller. The identifiability probe on chaotic data can falsely
    report "identified" (chaos responds to everything at prior scale),
    so the guard alone is not sufficient protection; pinning the point
    estimate at the inits makes application a no-op regardless of the
    verdicts.
    """
    init_values = np.array([s.init_value for s in all_specs], dtype=float)
    prior_sigma = prior_widths(list(all_specs), _ROLLOUT_PRIOR_SIGMA_SCALE)
    return FitResult(names=[s.name for s in all_specs],
                     samples=init_values[None, :],
                     log_probs=np.zeros(1),
                     jacobian=None,
                     noise_sigma=noise_sigma,
                     prior_sigma=prior_sigma,
                     scales=[getattr(s, "scale", "linear") for s in all_specs])


def _explainability_cache_key(
    physical_specs: Sequence[ParamSpec],
    rule_specs: Sequence[ParamSpec],
    trajectories: List[RolloutTrajectory],
    anchors: Dict[str, float],
    scaling: Optional[ResidualScaling],
    grid_seed_points: int,
) -> Tuple:
    """Cache key for :func:`min_explainable_rms` verdicts within one phase.

    Everything the candidate grid and the scored data depend on: spec
    boxes/scales, anchors, grid resolution, the residual scaling, and
    the segment shapes. Rule-spec INIT values matter (rule params are
    not swept, so candidates hold them at their anchor-or-init).
    Trajectory identity is approximated by per-segment lengths - exact
    within one learn phase, where the underlying recordings are fixed
    and segmentation is deterministic.
    """
    return (
        tuple((s.name, s.lo, s.hi, getattr(s, "scale", "linear"))
              for s in physical_specs),
        tuple((s.name, s.init_value) for s in rule_specs),
        tuple(sorted(anchors.items())),
        grid_seed_points,
        scaling.signature() if scaling is not None else None,
        tuple(len(actions) for _states, actions in trajectories),
    )


def fit_params_rollout_trimmed(
    base_env: Any,
    trajectories: List[RolloutTrajectory],
    physical_specs: Sequence[ParamSpec],
    process_features: Dict[str, List[str]],
    rules: Sequence[Any] = (),
    rule_specs: Sequence[ParamSpec] = (),
    latent_init: Any = None,
    num_steps: Optional[int] = None,
    noise_sigma: float = 0.05,
    scaling: Optional[ResidualScaling] = None,
    anchors: Optional[Dict[str, float]] = None,
    rms_cache: Optional[Dict[Tuple, List[float]]] = None,
    config: Optional[SysIdConfig] = None,
) -> Tuple[FitResult, List[RolloutTrajectory], List[float]]:
    """Drop trajectories no parameters can explain, fit on the rest.

    Goodness-of-fit trimming: a chaotic recording (e.g. a center-height
    scraping push whose 500-step outcome is contact-chaos) has a large
    rollout residual at EVERY parameter value — it is unexplainable, and
    pooling it with clean recordings lets its parameter-independent
    noise outvote their signal (measured on run_20260706_111805: one
    such trajectory dragged the pooled friction fit to 0.34 vs the true
    0.1 that the clean trajectory alone recovers). Each trajectory's
    best-achievable RMS over a candidate param grid
    (:func:`min_explainable_rms` — judged against its own best params,
    NOT a pooled fit, which chaos poisons) is compared against
    ``noise_sigma`` scaled by
    ``CFG.code_sim_learning_rollout_trim_rms_factor``; unexplainable
    recordings are dropped before the fit ever sees them.

    ``scaling`` defaults to :func:`compute_residual_scaling` over the
    input trajectories, so the trimming threshold compares
    dimensionless RMS values. ``rms_cache`` (a caller-owned dict, e.g.
    per learn phase) memoizes the explainability sweep - the most
    expensive part of repeated ``evaluate_step_fit`` calls - under a
    key that captures everything the verdict depends on, which both
    saves rollouts and pins the verdict for identical inputs.

    Returns ``(fit_result, surviving_trajectories, per_trajectory_rms)``
    — callers must compute their post-fit SSE / identifiability probe on
    the SURVIVORS, or a rejected trajectory's noise re-poisons the
    verdicts. If nothing survives, the survivor list is empty and the
    returned fit is pinned at the declared inits (see
    :func:`_init_point_fit_result`), so applying it is a no-op.
    """
    config = config or SysIdConfig.from_cfg()
    factor = config.trim_rms_factor
    all_specs = list(physical_specs) + list(rule_specs)
    if scaling is None:
        scaling = compute_residual_scaling(trajectories,
                                           process_features,
                                           config=config)
    anchors = anchors or {}
    if not trajectories or factor <= 0:
        result = fit_params_rollout(base_env,
                                    trajectories,
                                    physical_specs,
                                    process_features,
                                    rules=rules,
                                    rule_specs=rule_specs,
                                    latent_init=latent_init,
                                    num_steps=num_steps,
                                    noise_sigma=noise_sigma,
                                    scaling=scaling,
                                    anchors=anchors,
                                    config=config)
        return result, list(trajectories), []
    cache_key: Optional[Tuple] = None
    rms: Optional[List[float]] = None
    if rms_cache is not None:
        cache_key = _explainability_cache_key(physical_specs, rule_specs,
                                              trajectories, anchors, scaling,
                                              config.grid_seed_points)
        rms = rms_cache.get(cache_key)
        if rms is not None:
            logger.info(
                "Rollout sysID trimming: reusing cached explainability "
                "verdicts for this declaration/data signature.")
    if rms is None:
        rms = min_explainable_rms(base_env,
                                  trajectories,
                                  physical_specs,
                                  process_features,
                                  rules=rules,
                                  rule_specs=rule_specs,
                                  latent_init=latent_init,
                                  scaling=scaling,
                                  anchors=anchors,
                                  config=config)
        if rms_cache is not None and cache_key is not None:
            rms_cache[cache_key] = rms
    threshold = factor * noise_sigma
    survivors = [t for t, r in zip(trajectories, rms) if r <= threshold]
    if len(survivors) < len(trajectories):
        logger.info(
            "Rollout sysID trimming: per-trajectory best RMS %s vs "
            "threshold %.4f (%g x noise %.3f) — dropping %d of %d "
            "unexplainable trajectories.", [f"{r:.4g}" for r in rms],
            threshold, factor, noise_sigma,
            len(trajectories) - len(survivors), len(trajectories))
    if not survivors:
        logger.warning(
            "Rollout sysID trimming: NO trajectory is explainable at any "
            "candidate params; skipping the fit and pinning the result at "
            "the declared inits.")
        return _init_point_fit_result(all_specs, noise_sigma), [], rms

    # Consistency loop. Explainability alone is not enough: a chaotic
    # recording can be ACCIDENTALLY explainable at wrong params (measured:
    # a quiet center-height shove reached best-RMS 0.034 at friction
    # ~1.34 while the clean topple's best sits at friction ~0.1), and
    # pooling two explainable-but-disagreeing recordings drags the fit
    # to a compromise nobody supports. Invariant on exit: every survivor
    # fits the FINAL params nearly as well as its own best params. When
    # violated, the least trustworthy survivor (largest best-achievable
    # RMS) is dropped and the fit reruns — anchoring the answer on the
    # cleanest data rather than the loudest.
    consistency = config.consistency_factor
    physical_names = [s.name for s in physical_specs]
    best = [r for r in rms if r <= threshold]
    while True:
        result = fit_params_rollout(base_env,
                                    survivors,
                                    physical_specs,
                                    process_features,
                                    rules=rules,
                                    rule_specs=rule_specs,
                                    latent_init=latent_init,
                                    num_steps=num_steps,
                                    noise_sigma=noise_sigma,
                                    scaling=scaling,
                                    anchors=anchors,
                                    config=config)
        if len(survivors) <= 1 or consistency <= 0:
            break
        fit_rms = per_trajectory_rms(base_env, survivors,
                                     result.point_estimate, process_features,
                                     physical_names, rules, latent_init,
                                     scaling)
        violated = [
            i for i in range(len(survivors))
            if fit_rms[i] > consistency * best[i] + _CONSISTENCY_RMS_EPS
        ]
        if not violated:
            break
        drop_idx = max(range(len(survivors)), key=lambda i: best[i])
        logger.info(
            "Rollout sysID consistency: survivors disagree (RMS at joint "
            "fit %s vs own best %s); dropping the least trustworthy "
            "(index %d, best RMS %.4g) and refitting.",
            [f"{r:.4g}" for r in fit_rms], [f"{r:.4g}" for r in best],
            drop_idx, best[drop_idx])
        survivors.pop(drop_idx)
        best.pop(drop_idx)
    return result, survivors, rms
