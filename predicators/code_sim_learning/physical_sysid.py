"""System identification of PyBullet physical parameters by rollout matching.

The process-rule fitting in :mod:`predicators.code_sim_learning.training` is
*teacher-forced and single-step*: it resets the base sim to each observed
``s_t`` and predicts one step. That is correct for slow process features
(heating, filling) but wrong for **momentum-driven** dynamics such as a domino
cascade: the :class:`~predicators.structs.State` carries pose but no velocity,
so resetting to a mid-cascade state discards the angular momentum that produced
the next step, and the one-step prediction systematically under-rotates. No
friction/restitution value can repair that mismatch — physical parameters are
*invisible* to the teacher-forced objective.

This module fits such parameters by matching a **free-running rollout**
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
  teacher-forced / recurrent objectives in ``training.py``.
* Non-identifiability is *reported*, not regularized away: the posterior
  contraction per parameter (:func:`identifiability_report`) is surfaced to
  the agent so it can drop null parameters from its declaration.

The interface deliberately mirrors :func:`training.fit_params` (declare an
initialization via :class:`ParamSpec`, a Gaussian prior around it, a Gaussian
likelihood, emcee MCMC) so the agent-facing flow is unchanged: pick an init,
let the solver refine it.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, \
    Tuple

import numpy as np
import pybullet as p

from predicators.code_sim_learning.training import FitResult, ParamSpec, \
    _lm_point_fit_result, _lm_prefit, _param_bounds, _prior_widths, \
    _solve_lm
from predicators.settings import CFG
from predicators.structs import Action, State

logger = logging.getLogger(__name__)

# (states, actions) with len(states) == len(actions) + 1 and states[0] at rest.
RolloutTrajectory = Tuple[List[State], List[Action]]

# Posterior/prior-width ratio thresholds for the identifiability verdicts.
_IDENTIFIED_CONTRACTION = 0.3
_WEAK_CONTRACTION = 0.7

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


def _zero_all_velocities(base_env: Any) -> None:
    """Zero linear+angular velocity of every body in the env's client.

    ``_set_state`` rewrites poses but leaves velocities untouched, so without
    this a rollout would inherit residual momentum from a previous rollout and
    no longer start at rest. Fixed-base bodies (tables, walls, the robot base)
    ignore the reset, so blanket-zeroing is safe.
    """
    pcid = base_env._physics_client_id  # pylint: disable=protected-access
    for i in range(p.getNumBodies(physicsClientId=pcid)):
        bid = p.getBodyUniqueId(i, physicsClientId=pcid)
        p.resetBaseVelocity(bid, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
                            physicsClientId=pcid)


def rollout_states(base_env: Any, init_state: State, actions: List[Action],
                   physical_params: Dict[str, float]) -> List[State]:
    """Free-run the base sim from ``init_state`` under ``actions``.

    Resets once to ``init_state`` (zeroing velocities so the rollout begins at
    rest, matching how a recorded cascade starts), applies the candidate
    physics in place, then steps WITHOUT resetting so momentum accrues in-sim.
    Returns the post-step state after each action (length == ``len(actions)``).
    """
    base_env.apply_physical_param_overrides(physical_params)
    base_env._set_state(init_state)  # pylint: disable=protected-access
    _zero_all_velocities(base_env)
    # Re-apply after _set_state in case a reset path ever touches dynamics.
    base_env.apply_physical_param_overrides(physical_params)
    out: List[State] = []
    for action in actions:
        out.append(base_env.step(action))
    return out


def compute_rollout_sse(
        base_env: Any,
        trajectories: List[RolloutTrajectory],
        params: Dict[str, float],
        process_features: Dict[str, List[str]],
        physical_names: Sequence[str],
        rules: Sequence[Any] = (),
        latent_init: Any = None,
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
    """
    return sum((pred - obs)**2 for pred, obs in _iter_rollout_pred_obs(
        base_env, trajectories, params, process_features, physical_names,
        rules, latent_init))


def compute_rollout_residuals(
        base_env: Any,
        trajectories: List[RolloutTrajectory],
        params: Dict[str, float],
        process_features: Dict[str, List[str]],
        physical_names: Sequence[str],
        rules: Sequence[Any] = (),
        latent_init: Any = None,
) -> np.ndarray:
    """Rollout residuals (predicted - observed) as a flat vector.

    Levenberg-Marquardt counterpart of :func:`compute_rollout_sse` (same
    prediction pipeline, same iteration order — the sim is deterministic,
    so the same theta yields the same vector, as finite-difference
    Jacobians require).
    """
    return np.asarray([
        pred - obs for pred, obs in _iter_rollout_pred_obs(
            base_env, trajectories, params, process_features, physical_names,
            rules, latent_init)
    ],
                      dtype=float)


def _iter_rollout_pred_obs(
    base_env: Any,
    trajectories: List[RolloutTrajectory],
    params: Dict[str, float],
    process_features: Dict[str, List[str]],
    physical_names: Sequence[str],
    rules: Sequence[Any],
    latent_init: Any,
) -> Iterator[Tuple[float, float]]:
    """Yield (predicted, observed) feature pairs for the joint forward model.

    Shared prediction pipeline behind :func:`compute_rollout_sse` and
    :func:`compute_rollout_residuals`: free-run the base sim at the
    physical slice of ``params``, apply rules (latents threaded) on each
    rolled-out state, and pair every in-scope feature with its
    observation. Deterministic iteration order.
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
                    pred_val = pred.item() if hasattr(pred, "item") else pred
                    yield float(pred_val), float(obs_state.get(obs_obj, feat))


def fit_map_lm_rollout(
    base_env: Any,
    trajectories: List[RolloutTrajectory],
    physical_specs: Sequence[ParamSpec],
    process_features: Dict[str, List[str]],
    rules: Sequence[Any] = (),
    rule_specs: Sequence[ParamSpec] = (),
    latent_init: Any = None,
    max_nfev: int = 200,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """MAP estimate of the joint physical+rule theta via Levenberg-Marquardt.

    Rollout counterpart of :func:`training.fit_map_lm`, built on
    :func:`compute_rollout_residuals` and the shared bound-aware
    ``_solve_lm`` core. Uses a coarse relative finite-difference step
    (``_ROLLOUT_LM_DIFF_STEP``) because simulation residuals are flat
    under scipy's default ~1e-8 perturbations. Returns ``(theta_map,
    jacobian_at_optimum)``; the Jacobian feeds both the identifiability
    report and the Laplace exploration ensemble.
    """
    all_specs = list(physical_specs) + list(rule_specs)
    names = [s.name for s in all_specs]
    physical_names = [s.name for s in physical_specs]

    def residuals_fn(theta: np.ndarray) -> np.ndarray:
        params = {n: float(theta[i]) for i, n in enumerate(names)}
        return compute_rollout_residuals(base_env, trajectories, params,
                                         process_features, physical_names,
                                         rules, latent_init)

    return _solve_lm(residuals_fn,
                     all_specs,
                     max_nfev,
                     "rollout",
                     diff_step=_ROLLOUT_LM_DIFF_STEP)


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
    prior_sigma_scale: float = 0.75,
) -> FitResult:
    """Jointly identify physical + rule params against rollout SSE.

    Mirrors :func:`training.fit_params` end to end: Gaussian prior
    centered on each ParamSpec init, Gaussian likelihood
    ``-0.5*SSE/noise^2``, shared bounds/prior-width conventions, the
    same LM-prefit flow (Hessian diagnostic / warm start / Laplace
    bundle behind the usual CFG flags), and the same ``num_steps == 0``
    short-circuit to the **LM point fit** — which is the experiment
    default (``CFG.code_sim_learning_rollout_num_mcmc_steps = 0``,
    matching the configs' ``code_sim_learning_num_mcmc_steps: 0``); MCMC
    is opt-in. The forward model is a base-sim rollout
    (:func:`compute_rollout_sse`) rather than the per-step process
    rules, and theta concatenates ``physical_specs`` with
    ``rule_specs``.

    Serial only: the shared ``base_env`` is mutated per evaluation, so no
    multiprocessing pool may be used.
    """
    all_specs = list(physical_specs) + list(rule_specs)
    assert all_specs, "fit_params_rollout needs at least one ParamSpec."
    physical_names = [s.name for s in physical_specs]
    names = [s.name for s in all_specs]
    init_values = np.array([s.init_value for s in all_specs], dtype=float)
    if num_steps is None:
        num_steps = CFG.code_sim_learning_rollout_num_mcmc_steps
    if num_steps < 0:
        raise ValueError("code_sim_learning_rollout_num_mcmc_steps must be "
                         "non-negative.")
    lo, hi = _param_bounds(all_specs)
    prior_sigma = _prior_widths(init_values, lo, hi, prior_sigma_scale)

    # One-shot rollout LM fit (see _lm_prefit for its three uses). With
    # the default rollout MCMC budget of 0 this LM MAP *is* the fit.
    walker_center, lm_theta, lm_jac = _lm_prefit(
        lambda: fit_map_lm_rollout(
            base_env, trajectories, physical_specs, process_features, rules,
            rule_specs, latent_init), lambda p: compute_rollout_sse(
                base_env, trajectories, p, process_features, physical_names,
                rules, latent_init), names, init_values, noise_sigma,
        prior_sigma, "rollout")

    if num_steps == 0:
        return _lm_point_fit_result(walker_center, lm_theta, lm_jac, names,
                                    noise_sigma, prior_sigma, "rollout")

    import emcee  # type: ignore[import-untyped]  # pylint: disable=import-outside-toplevel

    ndim = len(all_specs)
    num_walkers = max(num_walkers, 2 * ndim + 2)
    burn_in = min(burn_in, max(num_steps - 1, 0))

    def log_posterior(theta: np.ndarray) -> float:
        if np.any(theta < lo) or np.any(theta > hi):
            return -np.inf
        params = {n: float(theta[i]) for i, n in enumerate(names)}
        log_prior = -0.5 * np.sum(((theta - init_values) / prior_sigma)**2)
        sse = compute_rollout_sse(base_env, trajectories, params,
                                  process_features, physical_names, rules,
                                  latent_init)
        return float(log_prior - 0.5 * sse / (noise_sigma**2))

    p0 = walker_center + 0.5 * prior_sigma * np.random.randn(num_walkers, ndim)
    p0 = np.clip(p0, lo, hi)
    sampler = emcee.EnsembleSampler(num_walkers, ndim, log_posterior)
    logger.info(
        "Rollout sysID emcee: %d walkers, %d steps, %d burn-in "
        "(%d physical + %d rule params, %d trajectories).", num_walkers,
        num_steps, burn_in, len(physical_names), len(list(rule_specs)),
        len(trajectories))
    report_interval = 25
    for i, _result in enumerate(sampler.sample(p0, iterations=num_steps),
                                start=1):
        if i % report_interval == 0 or i == num_steps:
            best_lp = sampler.get_log_prob()[:i].max()
            logger.info("  rollout emcee step %d/%d  (best log-prob: %.2f)", i,
                        num_steps, best_lp)
            for h in logger.handlers + logging.getLogger().handlers:
                h.flush()

    samples = sampler.get_chain(discard=burn_in, flat=True)
    log_probs = sampler.get_log_prob(discard=burn_in, flat=True)
    result = FitResult(names=names,
                       samples=samples,
                       log_probs=log_probs,
                       jacobian=lm_jac,
                       noise_sigma=noise_sigma,
                       prior_sigma=prior_sigma)
    logger.info("Rollout sysID done. MAP: %s",
                {k: f"{v:.4f}"
                 for k, v in result.point_estimate.items()})
    return result


def identifiability_report(
    result: FitResult,
    sse_fn: Optional[Callable[[Dict[str, float]], float]] = None,
    param_specs: Optional[Sequence[ParamSpec]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Per-parameter posterior-vs-prior contraction from a rollout fit.

    ``contraction = posterior_std / prior_std``: ~1 means the data did not
    constrain the parameter at all (the posterior is just the prior — a
    *null* parameter whose MAP value is arbitrary and should not be
    trusted), while values well below 1 mean the trajectories pinned it
    down. This is the analogue of ``mujoco.sysid``'s post-fit confidence
    intervals: non-identifiability is diagnosed and reported, never
    regularized away silently.

    Posterior widths come from the MCMC chain when one ran. For the
    default LM point fit (single-sample result), pass ``sse_fn`` (the
    rollout SSE at a params dict) and ``param_specs`` (for bounds): the
    widths then come from a prior-scale SSE **curvature probe** around
    the MAP — two rollout evals per parameter (see
    :func:`_probe_posterior_widths`). The Laplace covariance from the
    LM Jacobian is deliberately NOT used: finite-difference Jacobians
    of contact-rich rollouts are noise-dominated, and (measured on the
    domino smoke test) declare every parameter identified — the exact
    failure this report exists to catch. The curvature probe reproduces
    the MCMC ground truth (friction identified / restitution null).
    Without ``sse_fn`` a single-sample result reports "unknown".
    """
    if result.samples.shape[0] > 1:
        post_std = result.samples.std(axis=0)
    elif sse_fn is not None:
        post_std = _probe_posterior_widths(result, sse_fn, param_specs)
    else:
        post_std = np.full(len(result.names), np.nan)
    report: Dict[str, Dict[str, Any]] = {}
    for i, name in enumerate(result.names):
        prior = (float(result.prior_sigma[i])
                 if result.prior_sigma is not None else float("nan"))
        post = float(post_std[i])
        contraction = (post / prior
                       if np.isfinite(prior) and prior > 0 else float("nan"))
        if np.isnan(contraction):
            verdict = "unknown"
        elif contraction < _IDENTIFIED_CONTRACTION:
            verdict = "identified"
        elif contraction < _WEAK_CONTRACTION:
            verdict = "weakly identified"
        else:
            verdict = "NOT identified (posterior ~= prior; MAP arbitrary)"
        report[name] = {
            "posterior_std": post,
            "prior_std": prior,
            "contraction": contraction,
            "verdict": verdict,
        }
    return report


def _probe_posterior_widths(
    result: FitResult,
    sse_fn: Callable[[Dict[str, float]], float],
    param_specs: Optional[Sequence[ParamSpec]],
) -> np.ndarray:
    """Posterior widths via a prior-scale SSE curvature probe at the MAP.

    For each parameter, perturb it by ±prior_sigma (clipped to its box)
    with all others held at the MAP, and turn the mean SSE increase into
    a quadratic-approximation posterior width:
    ``SSE(x±d) - SSE(x) ≈ c·d²`` ⇒ posterior precision ``c/noise²`` ⇒
    ``post_std = noise/sqrt(c)``. A flat direction (no SSE increase over
    the whole prior scale) yields ``inf`` ⇒ contraction ≥ 1 ⇒ "NOT
    identified". Robust to contact-solver micro-chaos because the probe
    step is the *prior* scale, not a finite-difference scale.
    """
    point = result.point_estimate
    noise = result.noise_sigma if result.noise_sigma else 0.05
    assert result.prior_sigma is not None
    if param_specs:
        lo, hi = _param_bounds(list(param_specs))
        bounds = {
            s.name: (float(lo[i]), float(hi[i]))
            for i, s in enumerate(param_specs)
        }
    else:
        bounds = {}
    sse0 = sse_fn(dict(point))
    widths: List[float] = []
    for i, name in enumerate(result.names):
        sigma = float(result.prior_sigma[i])
        x = point[name]
        lo_i, hi_i = bounds.get(name, (-np.inf, np.inf))
        curvatures: List[float] = []
        for sgn in (1.0, -1.0):
            room = (hi_i - x) if sgn > 0 else (x - lo_i)
            delta = min(sigma, room)
            if delta <= 1e-9:
                continue
            pert = dict(point)
            pert[name] = x + sgn * delta
            d_sse = max(sse_fn(pert) - sse0, 0.0)
            curvatures.append(d_sse / delta**2)
        c = float(np.mean(curvatures)) if curvatures else 0.0
        widths.append(noise / np.sqrt(c) if c > 0 else float("inf"))
    return np.asarray(widths, dtype=float)


def format_identifiability(report: Dict[str, Dict[str, Any]]) -> str:
    """Human/agent-readable rendering of :func:`identifiability_report`."""
    lines = []
    for name, info in report.items():
        lines.append(f"  {name:<28} posterior_std={info['posterior_std']:.4g}"
                     f"  prior_std={info['prior_std']:.4g}"
                     f"  contraction={info['contraction']:.2f}"
                     f"  -> {info['verdict']}")
    return "\n".join(lines)
