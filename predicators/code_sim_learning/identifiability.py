"""Post-fit identifiability verdicts for the rollout system-ID stack.

Posterior-vs-prior contraction reporting (MCMC widths or the noise-
aware SSE curvature probe), at-bound detection, trust selection of the
fitted values, and the human/agent-readable rendering. See the
:mod:`predicators.code_sim_learning.physical_sysid` module docstring
for why non-identifiability is reported rather than regularized away.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from predicators.code_sim_learning.fit_space import LOG_FLOOR, FitResult, \
    ParamSpec, param_bounds

logger = logging.getLogger(__name__)

# Posterior/prior-width ratio thresholds for the identifiability verdicts.
_IDENTIFIED_CONTRACTION = 0.3
_WEAK_CONTRACTION = 0.7

# Same-theta SSE evaluations used to estimate the nondeterminism noise
# floor (grid sweep and identifiability probe use the same count).
NOISE_FLOOR_EVALS = 3

# A fitted value within this fraction of the box width of a bound is
# reported "at bound" (untrustworthy: the optimizer hit the wall).
_AT_BOUND_BOX_FRAC = 1e-3


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
                arr[:, j] = np.log(np.maximum(arr[:, j], LOG_FLOOR))
        post_std = arr.std(axis=0)
    elif sse_fn is not None:
        post_std = _probe_posterior_widths(result, sse_fn, param_specs)
    else:
        post_std = np.full(len(result.names), np.nan)
    at_bound = _params_at_bound(result, param_specs, scales)
    sensitivity = result.sensitivity or {}
    ablation = getattr(result, "anchor_ablation", None) or {}
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
        abl = ablation.get(name)
        if abl is not None:
            # The probe's local curvature at a co-adapted MAP is real,
            # so it cannot see that the move was compensatory; the
            # ablation refit's global data-equivalence verdict wins.
            verdict = ("anchored (a refit with this param at its baseline "
                       "is data-equivalent; the fitted move was "
                       "compensatory and the baseline is applied)")
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
                        np.log(max(interval[1], LOG_FLOOR)) -
                        np.log(max(interval[0], LOG_FLOOR)))
                else:
                    width = float(interval[1] - interval[0])
                report[name]["flat_wide"] = bool(
                    np.isfinite(post) and width > 2.0 * post)
        if abl is not None:
            report[name]["anchor_ablation"] = dict(abl)
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
    lo, hi = param_bounds(list(param_specs))
    out: Dict[str, str] = {}
    for i, spec in enumerate(param_specs):
        name = spec.name
        if name not in point:
            continue
        x, lo_i, hi_i = float(point[name]), float(lo[i]), float(hi[i])
        if scales[i] == "log":
            x = float(np.log(max(x, LOG_FLOOR)))
            lo_i = float(np.log(lo_i)) if lo_i > 0 else -np.inf
            hi_i = float(np.log(hi_i)) if np.isfinite(hi_i) else np.inf
        if np.isfinite(lo_i) and np.isfinite(hi_i):
            tol = _AT_BOUND_BOX_FRAC * (hi_i - lo_i)
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
        lo, hi = param_bounds(list(param_specs))
        bounds = {
            s.name: (float(lo[i]), float(hi[i]))
            for i, s in enumerate(param_specs)
        }
    else:
        bounds = {}
    sse0_evals = [sse_fn(dict(point)) for _ in range(NOISE_FLOOR_EVALS)]
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
        abl = info.get("anchor_ablation")
        if abl is not None:
            lines.append(f"      anchor ablation: SSE {abl['sse_pinned']:.4g} "
                         f"with it refit-pinned at baseline "
                         f"{abl['anchor']:.4g} vs {abl['sse_map']:.4g} at the "
                         f"joint MAP (tol {abl['tol']:.4g})")
    return "\n".join(lines)
