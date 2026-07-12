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
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, FrozenSet, Iterator, List, Optional, \
    Sequence, Set, Tuple

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


@dataclass
class ResidualScaling:
    """Per-(type, feature) residual semantics for the rollout objective.

    ``angular`` features (declared on their :class:`~predicators.structs
    .Type` via ``angular_features``) have their prediction errors
    wrapped to [-pi, pi] before scoring, so equivalent orientations
    (a settled domino at roll -pi vs +pi) do not read as a (2*pi)^2
    error per step. Every residual is then divided by its feature's
    ``scales`` entry, making residuals dimensionless fractions of
    typical motion: without this, radians and meters share one implicit
    unit and rotation errors drown position information (measured on
    run_20260711_141026: roll+yaw carried 83% of the post-fit SSE, with
    max errors of exactly 2*pi and pi - pure representation artifacts).
    """

    angular: FrozenSet[Tuple[str, str]]
    scales: Dict[Tuple[str, str], float] = field(default_factory=dict)

    def residual(self, type_name: str, feat: str, pred: float,
                 obs: float) -> float:
        """Scaled (and, for angular features, wrapped) ``pred - obs``."""
        key = (type_name, feat)
        diff = pred - obs
        if key in self.angular:
            diff = (diff + np.pi) % (2.0 * np.pi) - np.pi
        return diff / self.scales.get(key, 1.0)

    def signature(self) -> Tuple:
        """Hashable identity for caching explainability verdicts."""
        return (tuple(sorted(self.angular)),
                tuple(sorted(
                    (k, round(v, 12)) for k, v in self.scales.items())))


def compute_residual_scaling(
    trajectories: List[RolloutTrajectory],
    process_features: Dict[str, List[str]],
) -> Optional[ResidualScaling]:
    """Data-derived :class:`ResidualScaling` for a fit's trajectory set.

    Angular features come from each Type's declared ``angular_features``
    metadata (the states carry their types, so no env handle is
    needed). Scales: angular features get a constant ``pi`` (the
    largest possible wrapped error, so a full topple-direction mistake
    scores ~0.5); linear features get their observed span (max - min)
    across ALL observed states of the fit data, floored at
    ``CFG.code_sim_learning_rollout_feature_scale_floor`` so static
    features do not amplify sensor noise. Computed from observations
    only, so it is deterministic per dataset and MUST be shared across
    every SSE/RMS evaluation of one fit - per-trajectory scales would
    make trimming verdicts incomparable.

    Returns ``None`` when ``code_sim_learning_rollout_scale_residuals``
    is off (raw, unwrapped residuals - the legacy objective).
    """
    if not CFG.code_sim_learning_rollout_scale_residuals:
        return None
    floor = CFG.code_sim_learning_rollout_feature_scale_floor
    angular: Set[Tuple[str, str]] = set()
    lo: Dict[Tuple[str, str], float] = {}
    hi: Dict[Tuple[str, str], float] = {}
    for states, _actions in trajectories:
        for state in states:
            for obj in state:
                feats = process_features.get(obj.type.name, [])
                if not feats:
                    continue
                type_angular = set(getattr(obj.type, "angular_features", ()))
                for feat in feats:
                    key = (obj.type.name, feat)
                    if feat in type_angular:
                        angular.add(key)
                        continue
                    val = float(state.get(obj, feat))
                    lo[key] = min(lo.get(key, val), val)
                    hi[key] = max(hi.get(key, val), val)
    scales = {key: max(hi[key] - lo_val, floor) for key, lo_val in lo.items()}
    for key in angular:
        scales[key] = float(np.pi)
    return ResidualScaling(angular=frozenset(angular), scales=scales)


def split_at_rest_points(
    trajectory: RolloutTrajectory,
    process_features: Dict[str, List[str]],
    motion_tol: Optional[float] = None,
    min_rest_steps: Optional[int] = None,
    margin: Optional[int] = None,
) -> List[RolloutTrajectory]:
    """Split a recording into independently-scored rest-anchored segments.

    Multiple shooting for chaotic contact dynamics: free-running an
    entire manipulation trajectory lets small early divergence compound
    across phases, which shifts the SSE minimum away from the true
    parameters (replay-divergence bias - observed pulling the fitted
    friction both above and below truth on different runs). Cutting at
    rest points (every scored feature quiescent for at least
    ``min_rest_steps`` consecutive steps) bounds the compounding
    horizon while keeping each segment's zero-velocity re-anchor exact:
    the observed anchor state genuinely is at rest, so no momentum is
    discarded (the failure mode that rules out per-step teacher
    forcing, see module docstring).

    Each segment spans from the at-rest state just before its first
    motion to ``margin`` steps after its last motion (so it is still
    scored on settling at the right pose). Fully-static stretches
    between segments are dropped: they carry no parameter signal but
    would re-score accumulated divergence every step. A trajectory with
    no scored motion at all yields ``[]``.

    Trimming/consistency then operate per segment, so one chaotic phase
    (e.g. a scraping robot push) no longer discards the clean cascade
    recorded seconds later in the same episode.
    """
    if motion_tol is None:
        motion_tol = CFG.code_sim_learning_rollout_settle_tol
    if min_rest_steps is None:
        min_rest_steps = CFG.code_sim_learning_rollout_segment_min_rest_steps
    if margin is None:
        margin = CFG.code_sim_learning_rollout_settle_margin
    states, actions = trajectory
    num_steps = len(actions)
    active: List[int] = []
    for i in range(num_steps):
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
                active.append(i)
                break
    if not active:
        return []
    runs: List[Tuple[int, int]] = []
    run_start = active[0]
    prev = active[0]
    for i in active[1:]:
        if i - prev > min_rest_steps:
            runs.append((run_start, prev))
            run_start = i
        prev = i
    runs.append((run_start, prev))
    segments: List[RolloutTrajectory] = []
    for a, b in runs:
        end = min(num_steps, b + 1 + margin)
        segments.append((states[a:end + 1], actions[a:end]))
    return segments


def physical_param_anchors(
        base_env: Any,
        physical_specs: Sequence[ParamSpec]) -> Dict[str, float]:
    """Env-registry baseline values for the declared physical params.

    ``get_physical_param_info()`` defaults report the env's believed
    baseline WITHOUT any fit (the value the sysID revert path restores
    to), which makes them the right anchor for everything that must not
    drift with the agent's per-call declarations: the Gaussian prior
    center, the held-at values of grid sweeps, and the fallback applied
    for parameters the data does not constrain. Anchoring these at the
    agent's declared inits instead lets a re-declared init (a) change
    the explainability candidate grid call-to-call, flipping trimming
    verdicts on identical data, and (b) smuggle an unsupported
    hypothesis into the planner when the fit does not contract (e.g. a
    declared restitution of 0.15 surviving as "kept init" against a
    baseline of 0.02). Params the env does not reveal are absent from
    the result (callers fall back to the declared init).
    """
    getter = getattr(base_env, "get_physical_param_info", None)
    info = getter() if callable(getter) else {}
    return {
        s.name: float(info[s.name]["default"])
        for s in physical_specs if s.name in info
    }


def _pin_all_physical_params(base_env: Any,
                             physical_params: Dict[str, float]) -> None:
    """Apply ``physical_params`` with every OTHER registry param pinned to its
    env default.

    The env-side override is sticky per param, so on any env that
    outlives one evaluation (a caller-owned instance rather than a per-
    rollout factory build) a fit declaring a SUBSET of the params (e.g.
    only rolling_friction after an earlier fit touched lateral_friction)
    would silently inherit stale values from previous evaluations.
    Pinning also anchors undeclared params at the env's believed
    baseline on fresh builds, so every rollout evaluates the same
    nuisance physics regardless of env lifetime.
    """
    info: Dict[str, Dict] = getattr(base_env, "get_physical_param_info",
                                    lambda: {})()
    full = {name: float(spec["default"]) for name, spec in info.items()}
    full.update(physical_params)
    base_env.apply_physical_param_overrides(full)


def _dispose_env(env: Any) -> None:
    """Free a fresh rollout env by disconnecting its PyBullet client."""
    p.disconnect(env._physics_client_id)  # pylint: disable=protected-access


def rollout_states(base_env: Any, init_state: State, actions: List[Action],
                   physical_params: Dict[str, float]) -> List[State]:
    """Free-run the base sim from ``init_state`` under ``actions``.

    ``base_env`` is either an env instance or a zero-arg FACTORY: a
    factory is invoked to build a fresh env for this single rollout and
    the fresh env's PyBullet client is disconnected before returning.
    Fresh per-rollout worlds are what make repeated evaluations of the
    same theta deterministic - state-level resets on a shared env leave
    history-dependent residuals (near-matching bodies skipped by the
    reconstruction diff, auxiliary robot joints no reset touches), and
    even a bit-identical ``p.restoreState`` world diverges after a
    heavy-contact rollout via solver-internal state. Measured on
    run_20260708_213258: same-theta SSE alternated 0.15/78 on a shared
    env, which corrupted the grid seed and floored the identifiability
    probe (noise floor 82.9 -> every param "NOT identified").

    Resets once to ``init_state`` (zeroing velocities so the rollout
    begins at rest, matching how a recorded cascade starts), applies the
    candidate physics in place (undeclared registry params pinned to env
    defaults, see :func:`_pin_all_physical_params`), then steps WITHOUT
    resetting so momentum accrues in-sim. Returns the post-step state
    after each action (length == ``len(actions)``).
    """
    env = base_env() if callable(base_env) else base_env
    try:
        _pin_all_physical_params(env, physical_params)
        env._set_state(init_state)  # pylint: disable=protected-access
        _zero_all_velocities(env)
        # Re-apply after _set_state in case a reset path ever touches
        # dynamics.
        _pin_all_physical_params(env, physical_params)
        out: List[State] = []
        for action in actions:
            out.append(env.step(action))
        return out
    finally:
        if env is not base_env:
            _dispose_env(env)


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
    ``_solve_lm`` core. Uses a coarse relative finite-difference step
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
        z = _to_internal(all_specs, theta)
        prior_res = noise_sigma * (z - prior_centers) / prior_sigmas
        return np.concatenate([res, prior_res])

    theta_map, jac = _solve_lm(residuals_fn,
                               all_specs,
                               max_nfev,
                               "rollout",
                               diff_step=_ROLLOUT_LM_DIFF_STEP)
    if use_prior and jac is not None and jac.shape[0] > len(all_specs):
        jac = jac[:-len(all_specs)]
    return theta_map, jac


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


def _fit_space(spec: ParamSpec, value: float) -> float:
    """``value`` in ``spec``'s fit space (log for log-scale params)."""
    if _is_log(spec):
        return float(np.log(max(value, 1e-300)))
    return float(value)


def _from_fit_space(spec: ParamSpec, z: float) -> float:
    """Inverse of :func:`_fit_space`."""
    return float(np.exp(z)) if _is_log(spec) else float(z)


def _flat_candidates(
    pool: Sequence[Tuple[float, float]],
    noise_floor: float,
) -> Tuple[List[Tuple[float, float]], float, float]:
    """Split a ``(value, SSE)`` pool into its data-equivalent flat set.

    Candidates whose SSE is within ``max(noise_floor, flat_frac *
    best_SSE)`` of the best candidate are indistinguishable on this
    data: the tolerance is relative to the best achievable SSE, so a
    sharp basin (tiny best SSE) admits only true equals while a
    misfit-dominated landscape (large best SSE) treats its whole
    saturated shelf as one plateau. Returns ``(flat_members, best_sse,
    tolerance)``; with ``flat_frac`` and ``noise_floor`` both 0 the
    flat set degenerates to the exact argmin.
    """
    flat_frac = CFG.code_sim_learning_rollout_grid_flat_frac
    best_sse = min(sse for _v, sse in pool)
    tol = max(noise_floor, flat_frac * best_sse)
    flat = [(v, sse) for v, sse in pool if sse <= best_sse + tol]
    return flat, best_sse, tol


def _closest_to_anchor(spec: ParamSpec, flat: Sequence[Tuple[float, float]],
                       anchor: float) -> float:
    """The flat-set member nearest the anchor in fit space (ties: lower SSE).

    Among data-equivalent values the anchor-nearest one is the MAP
    choice: the data expresses no preference within the flat set, so the
    prior (centered on the anchor) decides. This is also what keeps the
    fit stable across cycles - a jitter-argmin wanders around the
    plateau as segments come and go, the anchor-side edge does not.
    """
    z_anchor = _fit_space(spec, anchor)

    def _rank(entry: Tuple[float, float]) -> Tuple[float, float]:
        value, sse = entry
        return (abs(_fit_space(spec, value) - z_anchor), sse)

    return min(flat, key=_rank)[0]


def _refine_flat_edge(spec: ParamSpec, pool: List[Tuple[float, float]],
                      anchor: float, noise_floor: float, refine_evals: int,
                      sse_for: Callable[[ParamSpec, float], float]) -> float:
    """Bisect the anchor-side edge of ``spec``'s flat set to sub-grid
    resolution.

    The coarse grid quantizes the seed to ~one grid gap
    (``geomspace(0.01, 2, 7)`` has 2.4x spacing), and LM cannot repair
    that on a chaotic replay landscape whose fine-scale gradient is
    noise (measured on run_20260711_224624: LM moved the seeded lateral
    friction by <0.3% and "converged", so the fit reported the 0.827
    grid point every cycle against a true 0.5 sitting mid-gap). Each
    iteration evaluates the fit-space midpoint between the current
    chosen value (the anchor-nearest flat member) and the nearest
    evaluated non-flat value on its anchor side - every evaluated value
    strictly between the anchor and the chosen one is non-flat, else it
    would itself be chosen. A flat midpoint moves the edge toward the
    anchor; a non-flat one tightens the bracket. The flat set is
    recomputed from the full pool each iteration, so a midpoint that
    reveals a genuinely better basin re-anchors everything on it.
    Appends its evaluations to ``pool`` and returns the refined choice.
    """
    z_anchor = _fit_space(spec, anchor)
    for _ in range(refine_evals):
        flat, _best, _tol = _flat_candidates(pool, noise_floor)
        chosen = _closest_to_anchor(spec, flat, anchor)
        z_chosen = _fit_space(spec, chosen)
        if z_chosen == z_anchor:
            break  # The anchor itself became data-equivalent.
        lo_z, hi_z = sorted((z_anchor, z_chosen))
        between = [
            _fit_space(spec, v) for v, _s in pool
            if lo_z < _fit_space(spec, v) < hi_z
        ]
        if z_chosen > z_anchor:
            z_far = max(between) if between else z_anchor
        else:
            z_far = min(between) if between else z_anchor
        if abs(z_chosen - z_far) <= 1e-3:
            break
        mid = _from_fit_space(spec, 0.5 * (z_chosen + z_far))
        pool.append((mid, sse_for(spec, mid)))
    flat, _best, _tol = _flat_candidates(pool, noise_floor)
    return _closest_to_anchor(spec, flat, anchor)


def _grid_seed_physical_specs(
    base_env: Any,
    trajectories: List[RolloutTrajectory],
    physical_specs: Sequence[ParamSpec],
    process_features: Dict[str, List[str]],
    rules: Sequence[Any],
    rule_specs: Sequence[ParamSpec],
    latent_init: Any,
    scaling: Optional[ResidualScaling] = None,
    anchors: Optional[Dict[str, float]] = None,
    noise_floor: float = 0.0,
) -> Tuple[List[ParamSpec], Dict[str, Dict[str, Any]]]:
    """Relocate each physical param's LM start via coordinate grid sweeps.

    The rollout SSE landscape can be flat around the declared init — for
    the domino env, topple reach saturates above friction ~0.5, so from
    an init of 0.5 the LM finite differences see no gradient and the fit
    stalls even on clean, informative data (verified 2026-07-06). The
    sweep is greedy per pass (declaration order, other params held at
    their current values; the grid plus the declared init, the anchor
    and the incumbent are always candidates), and runs up to
    ``code_sim_learning_rollout_grid_sweep_passes`` passes so a param
    swept early is re-examined after its neighbors moved, stopping as
    soon as a full pass moves nothing. Rule params are not swept (they
    fit fine locally and gridding them would explode combinatorially).

    Candidate selection is NOT the raw argmin: candidates whose SSE is
    data-equivalent to the best (:func:`_flat_candidates`) form a flat
    set, and the anchor-nearest member wins. This is the MAP choice
    within the data's resolution, and it prevents the two measured
    pathologies of argmin seeding (run_20260711_224624): a param
    compensating another's quantization error for an insignificant SSE
    gain (spinning_friction 0.5 -> 0.024, true 0.5, for 1.6%), and a
    saturated landscape reported as whichever interior grid point the
    replay chaos favored. Each moved param's anchor-side flat edge is
    then bisected to sub-grid resolution (:func:`_refine_flat_edge`),
    which is what lets a true value sitting mid-gap (lateral friction
    0.5 between grid candidates 0.342 and 0.827) be expressed at all.

    All rollout evaluations within one call are memoized (fresh-env
    rollouts are deterministic), so a converged extra pass costs no new
    rollouts. Pools from the final pass back both the returned per-param
    sweep info dict (``span`` — the sensitivity screen's raw material —
    and ``flat_interval``, the data-equivalent range in external units)
    and the refinement; a pool's held-at context can be one refinement
    tolerance stale for params refined before it, which is within the
    flat set's own resolution.
    """
    num_points = CFG.code_sim_learning_rollout_grid_seed_points
    num_passes = max(1, CFG.code_sim_learning_rollout_grid_sweep_passes)
    refine_evals = CFG.code_sim_learning_rollout_grid_refine_evals
    physical_names = [s.name for s in physical_specs]
    all_specs = list(physical_specs) + list(rule_specs)
    names = [s.name for s in all_specs]
    anchors = anchors or {}
    anchor_of = {
        s.name: float(anchors.get(s.name, s.init_value))
        for s in physical_specs
    }
    current = {
        s.name: float(anchors.get(s.name, s.init_value))
        for s in all_specs
    }

    memo: Dict[Tuple[float, ...], float] = {}

    def _sse_for(spec: ParamSpec, value: float) -> float:
        trial = dict(current)
        trial[spec.name] = float(value)
        key = tuple(trial[n] for n in names)
        if key not in memo:
            memo[key] = compute_rollout_sse(base_env, trajectories, trial,
                                            process_features, physical_names,
                                            rules, latent_init, scaling)
        return memo[key]

    pools: Dict[str, List[Tuple[float, float]]] = {}
    for pass_idx in range(num_passes):
        moved_any = False
        for spec in physical_specs:
            assert spec.lo is not None and spec.hi is not None, \
                "Grid seeding needs bounded physical params."
            values = [float(v) for v in _grid_candidates(spec, num_points)]
            values += [
                float(spec.init_value), anchor_of[spec.name],
                current[spec.name]
            ]
            pool = [(v, _sse_for(spec, v)) for v in dict.fromkeys(values)]
            pools[spec.name] = pool
            flat, _best, tol = _flat_candidates(pool, noise_floor)
            chosen = _closest_to_anchor(spec, flat, anchor_of[spec.name])
            sse_of = dict(pool)
            if chosen != current[spec.name]:
                logger.info(
                    "Rollout grid sweep (pass %d): %s %.4g -> %.4g "
                    "(SSE %.4g -> %.4g over %d candidates, flat tol %.3g).",
                    pass_idx + 1, spec.name, current[spec.name], chosen,
                    sse_of[current[spec.name]], sse_of[chosen], len(pool), tol)
                current[spec.name] = chosen
                moved_any = True
            else:
                raw_best = min(pool, key=lambda vs: vs[1])[0]
                if raw_best != chosen:
                    logger.info(
                        "Rollout grid sweep (pass %d): %s stays at %.4g - "
                        "best candidate %.4g is data-equivalent (SSE %.4g "
                        "vs %.4g, flat tol %.3g), so the anchor-nearest "
                        "value wins.", pass_idx + 1, spec.name, chosen,
                        raw_best, sse_of[raw_best], sse_of[chosen], tol)
        if not moved_any:
            break

    if refine_evals > 0:
        for spec in physical_specs:
            if current[spec.name] == anchor_of[spec.name]:
                continue
            refined = _refine_flat_edge(spec, pools[spec.name],
                                        anchor_of[spec.name], noise_floor,
                                        refine_evals, _sse_for)
            if refined != current[spec.name]:
                logger.info(
                    "Rollout grid refine: %s %.4g -> %.4g (anchor-side "
                    "flat edge at sub-grid resolution).", spec.name,
                    current[spec.name], refined)
                current[spec.name] = refined

    seeded: List[ParamSpec] = []
    sweep_info: Dict[str, Dict[str, Any]] = {}
    for spec in physical_specs:
        pool = pools[spec.name]
        sses = [sse for _v, sse in pool]
        flat, _best, _tol = _flat_candidates(pool, noise_floor)
        flat_values = [v for v, _sse in flat]
        sweep_info[spec.name] = {
            "span": float(max(sses) - min(sses)),
            "flat_interval":
            (float(min(flat_values)), float(max(flat_values))),
        }
        seeded.append(
            ParamSpec(spec.name,
                      current[spec.name],
                      lo=spec.lo,
                      hi=spec.hi,
                      scale=spec.scale))
    return seeded, sweep_info


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
    scaling: Optional[ResidualScaling] = None,
    anchors: Optional[Dict[str, float]] = None,
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
    center_int = _to_internal(all_specs, center_values)
    prior_sigma = _prior_widths(center_specs, prior_sigma_scale)

    # Coarse grid sweep to place the LM start in the right basin (see
    # _grid_seed_physical_specs for why LM alone can stall). Also
    # yields the per-param SSE spans and data-equivalent flat intervals
    # the sensitivity screen and the identifiability report consume.
    lm_physical_specs = list(physical_specs)
    sensitivity: Optional[Dict[str, Dict[str, Any]]] = None
    if (CFG.code_sim_learning_rollout_grid_seed_points > 0 and trajectories):
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
                                latent_init, scaling) for _ in range(3)
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
            noise_floor=noise_floor)
        sens_factor = CFG.code_sim_learning_rollout_sensitivity_factor
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

    # One-shot rollout LM fit (see _lm_prefit for its three uses). With
    # the default rollout MCMC budget of 0 this LM MAP *is* the fit.
    walker_center, lm_theta, lm_jac = _lm_prefit(
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
        result = _lm_point_fit_result(walker_center,
                                      lm_theta,
                                      lm_jac,
                                      names,
                                      noise_sigma,
                                      prior_sigma,
                                      "rollout",
                                      scales=scales)
        result.sensitivity = sensitivity
        return result

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
        log_prior = -0.5 * np.sum(((theta - center_int) / prior_sigma)**2)
        sse = compute_rollout_sse(base_env, trajectories, params,
                                  process_features, physical_names, rules,
                                  latent_init, scaling)
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
                       scales=scales,
                       sensitivity=sensitivity)
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


def min_explainable_rms(
    base_env: Any,
    trajectories: List[RolloutTrajectory],
    physical_specs: Sequence[ParamSpec],
    process_features: Dict[str, List[str]],
    rules: Sequence[Any] = (),
    rule_specs: Sequence[ParamSpec] = (),
    latent_init: Any = None,
    extra_candidates: Sequence[Dict[str, float]] = (),
    scaling: Optional[ResidualScaling] = None,
    anchors: Optional[Dict[str, float]] = None,
) -> List[float]:
    """Best achievable RMS of each trajectory over a candidate param grid.

    Explainability must be judged per trajectory against its OWN best
    parameters, not against a pooled fit: a poisoned pooled fit makes
    even a clean recording look unexplainable (measured: the clean push
    scored RMS 0.158 at a chaos-dragged fit vs 3e-4 at the true
    friction). The candidate set is the anchor point (env-registry
    baselines where revealed, declared inits otherwise) plus the same
    coordinate sweep the grid seeding uses (one physical param varied
    at a time, others held at the anchors) plus any
    ``extra_candidates``; the minimum RMS over it answers "can ANY
    reasonable parameter setting explain this recording?" — chaos
    cannot be explained by any, a clean topple is explained
    near-perfectly by the right one.

    The candidate set deliberately does NOT include the declared inits:
    anchoring it on the (per-phase stable) registry baselines makes the
    explainability verdict a function of the DATA alone, where an
    init-dependent grid flipped the same recording between explainable
    and unexplainable as the agent re-declared inits across calls
    (observed on run_20260711_141026, cycle 2).
    """
    anchors = anchors or {}
    base = {
        s.name: anchors.get(s.name, s.init_value)
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
                                 latent_init, scaling)
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


def _explainability_cache_key(
    physical_specs: Sequence[ParamSpec],
    rule_specs: Sequence[ParamSpec],
    trajectories: List[RolloutTrajectory],
    anchors: Dict[str, float],
    scaling: Optional[ResidualScaling],
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
        CFG.code_sim_learning_rollout_grid_seed_points,
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
    factor = CFG.code_sim_learning_rollout_trim_rms_factor
    all_specs = list(physical_specs) + list(rule_specs)
    if scaling is None:
        scaling = compute_residual_scaling(trajectories, process_features)
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
                                    anchors=anchors)
        return result, list(trajectories), []
    cache_key: Optional[Tuple] = None
    rms: Optional[List[float]] = None
    if rms_cache is not None:
        cache_key = _explainability_cache_key(physical_specs, rule_specs,
                                              trajectories, anchors, scaling)
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
                                  anchors=anchors)
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
                                    noise_sigma=noise_sigma,
                                    scaling=scaling,
                                    anchors=anchors)
        if len(survivors) <= 1 or consistency <= 0:
            break
        fit_rms = per_trajectory_rms(base_env, survivors,
                                     result.point_estimate, process_features,
                                     physical_names, rules, latent_init,
                                     scaling)
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
    num_explainable: Optional[int] = None,
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

    Two verdict overrides guard the probe's blind spots:

    * ``num_explainable`` (how many trimmed-in segments back the fit):
      the probe measures local *precision*, which a single clean
      recording can make arbitrarily sharp while the value is still
      biased; "identified" on n < 2 segments is downgraded to weakly
      identified with an explicit note.
    * ``result.sensitivity`` (pre-fit screen): a param whose grid-sweep
      SSE span sat inside the same-theta noise floor does not affect
      the rollouts at all on this data; chaos can still hand the probe
      apparent curvature for it, so the screen's verdict wins
      ("insensitive", fitted value not applied).
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
    at_bound = _params_at_bound(result, param_specs, scales)
    sensitivity = result.sensitivity or {}
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
        if (verdict == "identified" and num_explainable is not None
                and num_explainable < 2):
            verdict = ("weakly identified (sharp posterior, but only "
                       f"{num_explainable} explainable segment(s) back it)")
        if name in at_bound:
            # A MAP pinned at its box edge means the optimizer ran out
            # of box: the data pushes the parameter outside its
            # physically-plausible range (usually a weakly-informed
            # direction absorbing model error). The one-sided curvature
            # probe reads the wall as sharpness (measured: mass fit to
            # the 1.0 hi bound with posterior_std 5e-11 on
            # run_20260711_141026 replay data), so the probe verdict
            # cannot be trusted there.
            verdict = (f"at box {at_bound[name]} bound (data pushes it out "
                       "of its plausible range; fitted value NOT applied)")
        sens = sensitivity.get(name)
        if sens is not None and not sens.get("sensitive", True):
            verdict = ("insensitive (rollouts do not respond to this param "
                       "anywhere in its box on this data; fitted value is "
                       "noise and was NOT applied)")
        report[name] = {
            "posterior_std": post,
            "prior_std": prior,
            "contraction": contraction,
            "verdict": verdict,
        }
        if sens is not None:
            for key in ("sse_span", "noise_floor"):
                if key in sens:
                    report[name][key] = sens[key]
            interval = sens.get("flat_interval")
            if interval is not None:
                report[name]["flat_interval"] = interval
                # The probe measures LOCAL curvature at the chosen point;
                # when the sweep's data-equivalent interval is much wider
                # than the claimed posterior width, the point estimate is
                # representative of a plateau, not a unique optimum -
                # surface that instead of false precision.
                if scales[i] == "log":
                    width = float(
                        np.log(max(interval[1], 1e-300)) -
                        np.log(max(interval[0], 1e-300)))
                else:
                    width = float(interval[1] - interval[0])
                report[name]["flat_wide"] = bool(
                    np.isfinite(post) and width > 2.0 * post)
    return report


def _params_at_bound(
    result: FitResult,
    param_specs: Optional[Sequence[ParamSpec]],
    scales: Sequence[str],
) -> Dict[str, str]:
    """Params whose MAP sits on its box edge, mapped to "lo" / "hi".

    Proximity is judged in FIT space (log for log-scale params) within
    0.1% of the box width, so "at the low end of a decades-spanning
    box" is measured multiplicatively. Without ``param_specs`` (no
    bounds known) nothing is flagged.
    """
    if not param_specs:
        return {}
    point = result.point_estimate
    lo, hi = _param_bounds(list(param_specs))
    out: Dict[str, str] = {}
    for i, spec in enumerate(param_specs):
        name = spec.name
        if name not in point:
            continue
        x, lo_i, hi_i = float(point[name]), float(lo[i]), float(hi[i])
        if scales[i] == "log":
            x = float(np.log(max(x, 1e-300)))
            lo_i = float(np.log(lo_i)) if lo_i > 0 else -np.inf
            hi_i = float(np.log(hi_i)) if np.isfinite(hi_i) else np.inf
        if np.isfinite(lo_i) and np.isfinite(hi_i):
            tol = 1e-3 * (hi_i - lo_i)
        else:
            tol = 1e-6
        if np.isfinite(lo_i) and x - lo_i <= tol:
            out[name] = "lo"
        elif np.isfinite(hi_i) and hi_i - x <= tol:
            out[name] = "hi"
    return out


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
    anchors: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Pick which fitted physical values are safe to apply to the planner.

    A parameter whose posterior did not contract ("NOT identified" /
    "unknown") or that failed the sensitivity screen ("insensitive") has
    an arbitrary MAP — on uninformative data the grid seed and LM land
    wherever the rollout noise happened to be lowest — so applying it
    would move the planner's belief randomly, possibly further from the
    truth. For those, keep the env-registry ANCHOR when one is known
    (falling back to the declared init): the anchor is the env's
    standing baseline belief, whereas the declared init is this call's
    agent hypothesis, which unsupported data must not smuggle into the
    planner (observed: a declared restitution init of 0.15 surviving as
    "kept init" against a 0.02 baseline). Apply the fitted value only
    for parameters the data actually constrained (identified / weakly
    identified, including annotated variants).
    """
    anchors = anchors or {}
    applied: Dict[str, float] = {}
    for name in physical_names:
        verdict = report.get(name, {}).get("verdict", "unknown")
        if verdict.startswith(("identified", "weakly identified")):
            applied[name] = fitted[name]
        else:
            fallback = anchors.get(name, declared_inits[name])
            applied[name] = fallback
            if fitted[name] != fallback:
                logger.info(
                    "Rollout sysID: NOT applying %s=%.4f (verdict: %s); "
                    "keeping the %s %.4f.", name, fitted[name], verdict,
                    "registry anchor" if name in anchors else "declared init",
                    fallback)
    return applied


def format_identifiability(report: Dict[str, Dict[str, Any]]) -> str:
    """Human/agent-readable rendering of :func:`identifiability_report`."""
    lines = []
    for name, info in report.items():
        lines.append(f"  {name:<28} posterior_std={info['posterior_std']:.4g}"
                     f"  prior_std={info['prior_std']:.4g}"
                     f"  contraction={info['contraction']:.2f}"
                     f"  -> {info['verdict']}")
        interval = info.get("flat_interval")
        if interval is not None and interval[0] != interval[1]:
            note = (" - the fitted value is the edge of this interval "
                    "nearest the baseline belief, not a unique optimum"
                    if info.get("flat_wide") else "")
            lines.append(f"      data-equivalent over [{interval[0]:.4g}, "
                         f"{interval[1]:.4g}]{note}")
    return "\n".join(lines)
