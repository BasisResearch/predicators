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
    _internal_bounds, _is_log, _lm_point_fit_result, _lm_prefit, \
    _param_bounds, _prior_widths, _rows_to_external, _solve_lm, _to_external, \
    _to_internal
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
    """Zero every velocity in the env's client: base velocities of all bodies
    AND joint velocities of articulated bodies (the robot arm).

    ``_set_state`` rewrites poses but leaves velocities untouched, and
    its per-component diff skips joints whose positions already match
    the requested state, so without this a rollout inherits residual
    momentum from the previous rollout and no longer starts at rest. The
    joint pass matters as much as the base pass: measured up to ~1.8
    rad/s residual arm-joint velocity between rollouts, which chaotic
    contact amplifies into a 20-40% same-theta SSE jitter
    (run_20260705_203314). Fixed-base bodies ignore the base reset and
    fixed joints ignore the joint reset, so blanket-zeroing is safe.
    """
    pcid = base_env._physics_client_id  # pylint: disable=protected-access
    for i in range(p.getNumBodies(physicsClientId=pcid)):
        bid = p.getBodyUniqueId(i, physicsClientId=pcid)
        p.resetBaseVelocity(bid, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
                            physicsClientId=pcid)
        for j in range(p.getNumJoints(bid, physicsClientId=pcid)):
            pos = p.getJointState(bid, j, physicsClientId=pcid)[0]
            p.resetJointState(bid,
                              j,
                              pos,
                              targetVelocity=0.0,
                              physicsClientId=pcid)


def truncate_settled_tail(
    trajectory: RolloutTrajectory,
    process_features: Dict[str, List[str]],
    motion_tol: Optional[float] = None,
    margin: Optional[int] = None,
) -> RolloutTrajectory:
    """Cut a recorded trajectory once its scored features have settled.

    Scans the OBSERVED per-step deltas of the ``process_features`` and
    keeps everything up to the last step where any of them moved by more
    than ``motion_tol``, plus a ``margin`` of settle steps (so the
    rollout is still scored on coming to rest at the right pose). The
    static remainder is dropped: it contains no physics signal — the
    scored bodies no longer move — but re-scores whatever pose
    divergence the free-running rollout has accumulated on every
    remaining step, which is exactly the chaos-amplification term that
    drowned the friction signal in run_20260705_203314. Intermediate
    still phases are safe: the cut is anchored to the LAST motion, so a
    push -> settle -> second push trajectory keeps both pushes.

    A trajectory whose scored features never move carries no signal at
    all; it is truncated to the first ``margin`` steps (kept non-empty
    so callers' trajectory counts stay meaningful) and logged.
    """
    if motion_tol is None:
        motion_tol = CFG.code_sim_learning_rollout_settle_tol
    if margin is None:
        margin = CFG.code_sim_learning_rollout_settle_margin
    states, actions = trajectory
    last_active = -1
    for i in range(len(actions)):
        s_prev, s_next = states[i], states[i + 1]
        prev_by_name = {o.name: o for o in s_prev}
        for obj in s_next:
            feats = process_features.get(obj.type.name, [])
            prev_obj = prev_by_name.get(obj.name)
            if not feats or prev_obj is None:
                continue
            if any(
                    abs(
                        float(s_next.get(obj, f)) -
                        float(s_prev.get(prev_obj, f))) > motion_tol
                    for f in feats):
                last_active = i
                break
    if last_active < 0:
        logger.warning(
            "truncate_settled_tail: no scored feature ever moved more than "
            "%g in a %d-step trajectory; keeping only the first %d steps "
            "(the trajectory carries no physical-parameter signal).",
            motion_tol, len(actions), margin)
    keep = min(len(actions), last_active + 1 + margin)
    if keep >= len(actions):
        return trajectory
    return states[:keep + 1], actions[:keep]


def rollout_states(base_env: Any, init_state: State, actions: List[Action],
                   physical_params: Dict[str, float]) -> List[State]:
    """Free-run the base sim from ``init_state`` under ``actions``.

    Resets once to ``init_state`` (zeroing velocities so the rollout
    begins at rest, matching how a recorded cascade starts), applies the
    candidate physics in place, then steps WITHOUT resetting so momentum
    accrues in-sim. Returns the post-step state after each action
    (length == ``len(actions)``).
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


def _grid_candidates(spec: ParamSpec, num_points: int) -> np.ndarray:
    """Sweep values across ``spec``'s box, evenly spaced in its FIT space.

    Linear params get ``linspace``; log params get ``geomspace`` — equal
    resolution per decade. This matters when the box spans orders of
    magnitude: ``linspace(0.01, 2.0, 8)`` puts 7 of 8 points above 0.29
    and NOTHING between 0.01 and 0.29, so a true friction of 0.1 has no
    nearby candidate and the sweep jumps to the 0.01 endpoint (measured
    on run_20260706_171526: fitted 0.0114 vs true 0.1). ``geomspace``
    puts a candidate at ~0.098.
    """
    assert spec.lo is not None and spec.hi is not None
    if _is_log(spec):
        return np.geomspace(spec.lo, spec.hi, num_points)
    return np.linspace(spec.lo, spec.hi, num_points)


def _grid_seed_physical_specs(
    base_env: Any,
    trajectories: List[RolloutTrajectory],
    physical_specs: Sequence[ParamSpec],
    process_features: Dict[str, List[str]],
    rules: Sequence[Any],
    rule_specs: Sequence[ParamSpec],
    latent_init: Any,
) -> List[ParamSpec]:
    """Relocate each physical param's LM start via a coarse grid sweep.

    The rollout SSE landscape can be flat around the declared init — for
    the domino env, topple reach saturates above friction ~0.5, so from
    an init of 0.5 the LM finite differences see no gradient and the fit
    stalls even on clean, informative data (verified 2026-07-06). One
    coarse sweep per physical parameter (greedy, in declaration order,
    other params held at their current values, the declared init always
    among the candidates) relocates the LM start into the best-scoring
    basin; LM then polishes locally. This only changes the optimizer's
    starting point — the Gaussian prior stays centered on the declared
    init, so the posterior and identifiability semantics are unchanged.
    Rule params are not swept (they fit fine locally and gridding them
    would explode combinatorially).
    """
    num_points = CFG.code_sim_learning_rollout_grid_seed_points
    physical_names = [s.name for s in physical_specs]
    current = {
        s.name: s.init_value
        for s in list(physical_specs) + list(rule_specs)
    }
    seeded: List[ParamSpec] = []
    for spec in physical_specs:
        assert spec.lo is not None and spec.hi is not None, \
            "Grid seeding needs bounded physical params."
        candidates = [float(v) for v in _grid_candidates(spec, num_points)]
        candidates.append(float(spec.init_value))
        best_val, best_sse = float(spec.init_value), float("inf")
        for cand in candidates:
            trial = dict(current)
            trial[spec.name] = cand
            sse = compute_rollout_sse(base_env, trajectories, trial,
                                      process_features, physical_names, rules,
                                      latent_init)
            if sse < best_sse:
                best_sse, best_val = sse, cand
        if best_val != spec.init_value:
            logger.info(
                "Rollout grid seed: %s LM start %.4f -> %.4f "
                "(best of %d candidates, SSE %.4f).", spec.name,
                spec.init_value, best_val, len(candidates), best_sse)
        current[spec.name] = best_val
        seeded.append(
            ParamSpec(spec.name,
                      best_val,
                      lo=spec.lo,
                      hi=spec.hi,
                      scale=spec.scale))
    return seeded


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
    scales = [getattr(s, "scale", "linear") for s in all_specs]
    init_values = np.array([s.init_value for s in all_specs], dtype=float)
    if num_steps is None:
        num_steps = CFG.code_sim_learning_rollout_num_mcmc_steps
    if num_steps < 0:
        raise ValueError("code_sim_learning_rollout_num_mcmc_steps must be "
                         "non-negative.")
    lo, hi = _internal_bounds(all_specs)
    init_int = _to_internal(all_specs, init_values)
    prior_sigma = _prior_widths(all_specs, prior_sigma_scale)

    # Coarse grid sweep to place the LM start in the right basin (the
    # prior stays centered on the declared inits — see
    # _grid_seed_physical_specs for why LM alone can stall).
    lm_physical_specs = list(physical_specs)
    if (CFG.code_sim_learning_rollout_grid_seed_points > 0 and trajectories):
        lm_physical_specs = _grid_seed_physical_specs(base_env, trajectories,
                                                      physical_specs,
                                                      process_features, rules,
                                                      rule_specs, latent_init)

    # One-shot rollout LM fit (see _lm_prefit for its three uses). With
    # the default rollout MCMC budget of 0 this LM MAP *is* the fit.
    walker_center, lm_theta, lm_jac = _lm_prefit(
        lambda: fit_map_lm_rollout(
            base_env, trajectories, lm_physical_specs, process_features, rules,
            rule_specs, latent_init), lambda p: compute_rollout_sse(
                base_env, trajectories, p, process_features, physical_names,
                rules, latent_init), names, init_values, noise_sigma,
        prior_sigma, "rollout")

    if num_steps == 0:
        return _lm_point_fit_result(walker_center,
                                    lm_theta,
                                    lm_jac,
                                    names,
                                    noise_sigma,
                                    prior_sigma,
                                    "rollout",
                                    scales=scales)

    import emcee  # type: ignore[import-untyped]  # pylint: disable=import-outside-toplevel

    ndim = len(all_specs)
    num_walkers = max(num_walkers, 2 * ndim + 2)
    burn_in = min(burn_in, max(num_steps - 1, 0))

    def log_posterior(theta: np.ndarray) -> float:
        # theta lives in the FIT space (log for log-scale params).
        if np.any(theta < lo) or np.any(theta > hi):
            return -np.inf
        ext = _to_external(all_specs, theta)
        params = {n: float(ext[i]) for i, n in enumerate(names)}
        log_prior = -0.5 * np.sum(((theta - init_int) / prior_sigma)**2)
        sse = compute_rollout_sse(base_env, trajectories, params,
                                  process_features, physical_names, rules,
                                  latent_init)
        return float(log_prior - 0.5 * sse / (noise_sigma**2))

    p0 = _to_internal(all_specs, walker_center) + \
        0.5 * prior_sigma * np.random.randn(num_walkers, ndim)
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

    samples = _rows_to_external(all_specs,
                                sampler.get_chain(discard=burn_in, flat=True))
    log_probs = sampler.get_log_prob(discard=burn_in, flat=True)
    result = FitResult(names=names,
                       samples=samples,
                       log_probs=log_probs,
                       jacobian=lm_jac,
                       noise_sigma=noise_sigma,
                       prior_sigma=prior_sigma,
                       scales=scales)
    logger.info("Rollout sysID done. MAP: %s",
                {k: f"{v:.4f}"
                 for k, v in result.point_estimate.items()})
    return result


def per_trajectory_rms(
        base_env: Any,
        trajectories: List[RolloutTrajectory],
        params: Dict[str, float],
        process_features: Dict[str, List[str]],
        physical_names: Sequence[str],
        rules: Sequence[Any] = (),
        latent_init: Any = None,
) -> List[float]:
    """RMS rollout residual of each trajectory separately at ``params``.

    The per-trajectory analogue of :func:`compute_rollout_sse`: how well
    can the model explain THIS trajectory at these parameters, in the
    scored features' native units (meters / radians per residual)?
    """
    out: List[float] = []
    for traj in trajectories:
        res = compute_rollout_residuals(base_env, [traj], params,
                                        process_features, physical_names,
                                        rules, latent_init)
        out.append(float(np.sqrt(np.mean(res**2))) if res.size else 0.0)
    return out


def min_explainable_rms(
        base_env: Any,
        trajectories: List[RolloutTrajectory],
        physical_specs: Sequence[ParamSpec],
        process_features: Dict[str, List[str]],
        rules: Sequence[Any] = (),
        rule_specs: Sequence[ParamSpec] = (),
        latent_init: Any = None,
        extra_candidates: Sequence[Dict[str, float]] = (),
) -> List[float]:
    """Best achievable RMS of each trajectory over a candidate param grid.

    Explainability must be judged per trajectory against its OWN best
    parameters, not against a pooled fit: a poisoned pooled fit makes
    even a clean recording look unexplainable (measured: the clean push
    scored RMS 0.158 at a chaos-dragged fit vs 3e-4 at the true
    friction). The candidate set is the declared inits plus the same
    coordinate sweep the grid seeding uses (one physical param varied at
    a time) plus any ``extra_candidates`` (e.g. the pooled fit); the
    minimum RMS over it answers "can ANY reasonable parameter setting
    explain this recording?" — chaos cannot be explained by any, a clean
    topple is explained near-perfectly by the right one.
    """
    base = {
        s.name: s.init_value
        for s in list(physical_specs) + list(rule_specs)
    }
    candidates: List[Dict[str, float]] = [dict(base)]
    num_points = CFG.code_sim_learning_rollout_grid_seed_points
    if num_points > 0:
        for spec in physical_specs:
            assert spec.lo is not None and spec.hi is not None
            for val in _grid_candidates(spec, num_points):
                cand = dict(base)
                cand[spec.name] = float(val)
                candidates.append(cand)
    candidates.extend(dict(c) for c in extra_candidates)
    physical_names = [s.name for s in physical_specs]
    best = [float("inf")] * len(trajectories)
    for params in candidates:
        rms = per_trajectory_rms(base_env, trajectories, params,
                                 process_features, physical_names, rules,
                                 latent_init)
        best = [min(b, r) for b, r in zip(best, rms)]
    return best


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
    prior_sigma = _prior_widths(list(all_specs), 0.75)
    return FitResult(names=[s.name for s in all_specs],
                     samples=init_values[None, :],
                     log_probs=np.zeros(1),
                     jacobian=None,
                     noise_sigma=noise_sigma,
                     prior_sigma=prior_sigma,
                     scales=[getattr(s, "scale", "linear") for s in all_specs])


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

    Returns ``(fit_result, surviving_trajectories, per_trajectory_rms)``
    — callers must compute their post-fit SSE / identifiability probe on
    the SURVIVORS, or a rejected trajectory's noise re-poisons the
    verdicts. If nothing survives, the survivor list is empty and the
    returned fit is pinned at the declared inits (see
    :func:`_init_point_fit_result`), so applying it is a no-op.
    """
    factor = CFG.code_sim_learning_rollout_trim_rms_factor
    all_specs = list(physical_specs) + list(rule_specs)
    if not trajectories or factor <= 0:
        result = fit_params_rollout(base_env,
                                    trajectories,
                                    physical_specs,
                                    process_features,
                                    rules=rules,
                                    rule_specs=rule_specs,
                                    latent_init=latent_init,
                                    num_steps=num_steps,
                                    noise_sigma=noise_sigma)
        return result, list(trajectories), []
    rms = min_explainable_rms(base_env,
                              trajectories,
                              physical_specs,
                              process_features,
                              rules=rules,
                              rule_specs=rule_specs,
                              latent_init=latent_init)
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
    consistency = CFG.code_sim_learning_rollout_consistency_factor
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
                                    noise_sigma=noise_sigma)
        if len(survivors) <= 1 or consistency <= 0:
            break
        fit_rms = per_trajectory_rms(base_env, survivors,
                                     result.point_estimate, process_features,
                                     physical_names, rules, latent_init)
        violated = [
            i for i in range(len(survivors))
            if fit_rms[i] > consistency * best[i] + 1e-3
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
    scales = _result_scales(result, param_specs)
    if result.samples.shape[0] > 1:
        # Widths in FIT space (log for log-scale params), so the
        # contraction against the fit-space prior width is meaningful.
        arr = np.array(result.samples, dtype=float, copy=True)
        for j, scale in enumerate(scales):
            if scale == "log":
                arr[:, j] = np.log(np.maximum(arr[:, j], 1e-300))
        post_std = arr.std(axis=0)
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


def _result_scales(result: FitResult,
                   param_specs: Optional[Sequence[ParamSpec]]) -> List[str]:
    """Per-column fit scales for ``result``, from specs or the result."""
    if param_specs:
        return [getattr(s, "scale", "linear") for s in param_specs]
    if result.scales is not None:
        return list(result.scales)
    return ["linear"] * len(result.names)


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
    identified".

    The probe is noise-aware: chaotic contact rollouts are not perfectly
    repeatable (residual sub-tolerance state between evaluations gets
    amplified), so the SSE at the MAP is evaluated three times and the
    observed same-theta spread is subtracted from every perturbation's
    SSE increase before it counts as curvature. Without this, the jitter
    itself reads as curvature and every parameter is declared identified
    — the failure mode of run_20260705_203314, where a ~5k prior-scale
    d_sse sat inside a ~±8k same-theta noise floor yet reported
    contraction 0.00 on all params.
    """
    point = result.point_estimate
    noise = result.noise_sigma if result.noise_sigma else 0.05
    assert result.prior_sigma is not None
    scales = _result_scales(result, param_specs)
    if param_specs:
        lo, hi = _param_bounds(list(param_specs))
        bounds = {
            s.name: (float(lo[i]), float(hi[i]))
            for i, s in enumerate(param_specs)
        }
    else:
        bounds = {}
    sse0_evals = [sse_fn(dict(point)) for _ in range(3)]
    sse0 = float(np.median(sse0_evals))
    noise_floor = float(np.max(sse0_evals) - np.min(sse0_evals))
    if noise_floor > 0:
        logger.info(
            "Identifiability probe: same-theta SSE noise floor %.4f "
            "(MAP evals: %s); curvature below it is discounted.", noise_floor,
            [f"{v:.4f}" for v in sse0_evals])
    widths: List[float] = []
    for i, name in enumerate(result.names):
        sigma = float(result.prior_sigma[i])
        x = point[name]
        lo_i, hi_i = bounds.get(name, (-np.inf, np.inf))
        # Probe in the FIT space: for a log param the step is
        # multiplicative (x * e^±delta) and the curvature is measured
        # against the log-space delta, matching the log-space prior
        # width. This also stops a bound-hugging MAP from reading as
        # sharp curvature merely because the linear box ends there —
        # the flat low-friction basin of run_20260706_171526 spans a
        # 10x range that a linear probe sees as 4.5% of the box.
        if scales[i] == "log":
            x_fit = float(np.log(x))
            lo_fit = float(np.log(lo_i)) if lo_i > 0 else -np.inf
            hi_fit = float(np.log(hi_i)) if np.isfinite(hi_i) else np.inf
        else:
            x_fit, lo_fit, hi_fit = x, lo_i, hi_i
        curvatures: List[float] = []
        for sgn in (1.0, -1.0):
            room = (hi_fit - x_fit) if sgn > 0 else (x_fit - lo_fit)
            delta = min(sigma, room)
            if delta <= 1e-9:
                continue
            pert = dict(point)
            pert_fit = x_fit + sgn * delta
            pert[name] = (float(np.exp(pert_fit))
                          if scales[i] == "log" else pert_fit)
            d_sse = max(sse_fn(pert) - sse0 - noise_floor, 0.0)
            curvatures.append(d_sse / delta**2)
        c = float(np.mean(curvatures)) if curvatures else 0.0
        widths.append(noise / np.sqrt(c) if c > 0 else float("inf"))
    return np.asarray(widths, dtype=float)


def select_trustworthy_params(
    fitted: Dict[str, float],
    declared_inits: Dict[str, float],
    physical_names: Sequence[str],
    report: Dict[str, Dict[str, Any]],
) -> Dict[str, float]:
    """Pick which fitted physical values are safe to apply to the planner.

    A parameter whose posterior did not contract ("NOT identified" /
    "unknown") has an arbitrary MAP — on uninformative data the grid
    seed and LM land wherever the rollout noise happened to be lowest —
    so applying it would move the planner's belief randomly, possibly
    further from the truth than the declared init. Keep the declared
    init for those; apply the fitted value only for parameters the data
    actually constrained (identified / weakly identified).
    """
    applied: Dict[str, float] = {}
    for name in physical_names:
        verdict = report.get(name, {}).get("verdict", "unknown")
        if verdict in ("identified", "weakly identified"):
            applied[name] = fitted[name]
        else:
            applied[name] = declared_inits[name]
            if fitted[name] != declared_inits[name]:
                logger.info(
                    "Rollout sysID: NOT applying %s=%.4f (verdict: %s); "
                    "keeping the declared init %.4f.", name, fitted[name],
                    verdict, declared_inits[name])
    return applied


def format_identifiability(report: Dict[str, Dict[str, Any]]) -> str:
    """Human/agent-readable rendering of :func:`identifiability_report`."""
    lines = []
    for name, info in report.items():
        lines.append(f"  {name:<28} posterior_std={info['posterior_std']:.4g}"
                     f"  prior_std={info['prior_std']:.4g}"
                     f"  contraction={info['contraction']:.2f}"
                     f"  -> {info['verdict']}")
    return "\n".join(lines)
