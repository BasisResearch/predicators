"""Fit-space primitives shared by every parameter-fitting path.

``ParamSpec`` declares a learnable parameter (bounds and linear/log
scale); ``FitResult`` is the common result bundle. The transform
helpers map between EXTERNAL units (linear, simulator-facing) and the
FIT space the optimizers run in (``z = log(theta)`` for log-scale
params). This module is a dependency-free leaf: it may import nothing
from the rest of the package, so ``fitting``, ``physical_sysid``,
``active_experiment``, ``utils``, and the approaches can all share it
without import cycles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Floor before taking logs of nonnegative values, to keep log(0) finite.
LOG_FLOOR = 1e-300


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

    ``discrete`` marks a parameter whose value is an index or a count
    that the simulator rounds before use (a wiring slot, a selector).
    Its behavioural effect is a staircase, so a small perturbation is
    either a no-op or a jump to a different structure; uncertainty
    jitter around a point estimate (see ``perturbation_ensemble``)
    leaves it alone rather than manufacturing structural alternatives
    that no data supports.
    """

    name: str
    init_value: float
    lo: Optional[float] = None
    hi: Optional[float] = None
    scale: str = "linear"
    discrete: bool = False

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
    Laplace bundle, attached by :func:`fit_params`,
    :func:`fit_params_recurrent`, and
    :func:`physical_sysid.fit_params_rollout` whenever their
    Levenberg-Marquardt fit ran (info-seeking exploration or the
    Hessian/warm-start flags). They
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
    # One line per parameter whose LM Jacobian column was identically
    # zero (threshold/gate parameters the finite-difference fit cannot
    # move): what the bracket search did about it (see lm.solve_lm).
    # Rendered to the agent by the fit tools so "converged" never
    # passes for "fit from data" on such a parameter.
    lm_notes: List[str] = field(default_factory=list)
    # Pre-fit sensitivity screen (physical_sysid): per-param
    # {"sse_span": ..., "noise_floor": ..., "sensitive": bool}. A param
    # whose grid-sweep SSE span stays within the same-theta noise floor
    # does not affect the rollouts at all on this data; its fitted value
    # is noise and must not be applied. None = screen not run.
    sensitivity: Optional[Dict[str, Dict[str, Any]]] = None
    # Post-fit anchor-ablation verdicts (physical_sysid): per reverted
    # param {"anchor": ..., "sse_map": ..., "sse_pinned": ..., "tol":
    # ...}. A listed param's MAP move was compensatory - a refit with it
    # pinned at its env-registry anchor is data-equivalent - so its
    # ``point_estimate`` entry IS the anchor and the identifiability
    # verdict reports it as anchored rather than identified. None =
    # ablation not run or nothing reverted.
    anchor_ablation: Optional[Dict[str, Dict[str, float]]] = None

    @property
    def point_estimate(self) -> Dict[str, float]:
        """MAP (sample with highest log-probability)."""
        best_idx = int(np.argmax(self.log_probs))
        return {
            n: float(self.samples[best_idx, i])
            for i, n in enumerate(self.names)
        }


def param_bounds(
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


def is_log(spec: ParamSpec) -> bool:
    """Whether ``spec`` fits in log-space (tolerates legacy instances)."""
    return getattr(spec, "scale", "linear") == "log"


def to_fit_space(param_specs: List[ParamSpec], values: Any) -> np.ndarray:
    """Map external (linear) parameter values into the fit space."""
    return np.array([
        np.log(v) if is_log(s) else float(v)
        for s, v in zip(param_specs, values)
    ],
                    dtype=float)


def from_fit_space(param_specs: List[ParamSpec], values: Any) -> np.ndarray:
    """Map fit-space parameter values back to external (linear) units."""
    return np.array([
        np.exp(v) if is_log(s) else float(v)
        for s, v in zip(param_specs, values)
    ],
                    dtype=float)


def rows_from_fit_space(param_specs: List[ParamSpec],
                        arr: np.ndarray) -> np.ndarray:
    """Map a (num_rows, num_params) fit-space array to external units."""
    out = np.array(arr, dtype=float, copy=True)
    for j, spec in enumerate(param_specs):
        if is_log(spec):
            out[:, j] = np.exp(out[:, j])
    return out


def fit_space_bounds(
        param_specs: List[ParamSpec]) -> Tuple[np.ndarray, np.ndarray]:
    """The `param_bounds` box mapped into the fit space."""
    lo, hi = param_bounds(param_specs)
    lo_int = np.array(
        [np.log(l) if is_log(s) else l for s, l in zip(param_specs, lo)],
        dtype=float)
    hi_int = np.array(
        [np.log(h) if is_log(s) else h for s, h in zip(param_specs, hi)],
        dtype=float)
    return lo_int, hi_int


def prior_widths(param_specs: List[ParamSpec], scale: float) -> np.ndarray:
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
    lo, hi = param_bounds(param_specs)
    init_values = np.array([s.init_value for s in param_specs], dtype=float)
    sigma = np.abs(init_values) * scale
    finite = np.isfinite(lo) & np.isfinite(hi)
    fallback = np.where(finite, 0.5 * (hi - lo), 1.0)
    linear_sigma = np.where(sigma > 1e-9, sigma, fallback)
    log_mask = np.array([is_log(s) for s in param_specs], dtype=bool)
    return np.where(log_mask, float(scale), linear_sigma)


def scalar_to_fit_space(spec: ParamSpec, value: float) -> float:
    """``value`` in ``spec``'s fit space (log for log-scale params)."""
    if is_log(spec):
        return float(np.log(max(value, LOG_FLOOR)))
    return float(value)


def scalar_from_fit_space(spec: ParamSpec, z: float) -> float:
    """Inverse of :func:`scalar_to_fit_space`."""
    return float(np.exp(z)) if is_log(spec) else float(z)


def declared_interval_report(
        param_specs: List[ParamSpec]) -> Dict[str, Dict[str, Any]]:
    """A physics-margin ``report`` whose hull is each param's declared box.

    Consumed by :func:`identifiability.physics_sigma_points` when no fit
    ran (``agent_sim_learn_declared_params_only``): the
    ``candidate_values`` are the declared ``lo`` / ``hi`` bounds, so the
    margin sweep spans the agent's plausible interval instead of a
    posterior width. A param that declares no finite bound contributes
    nothing on that side; with no finite bound on either side its
    interval is unknown and the margin check is honestly vacuous for it.
    """
    report: Dict[str, Dict[str, Any]] = {}
    for spec in param_specs:
        cands = [
            float(b) for b in (spec.lo, spec.hi)
            if b is not None and np.isfinite(b)
        ]
        report[spec.name] = {
            "posterior_std": float("nan"),
            "candidate_values": cands,
        }
    return report


def declared_interval_fit_result(param_specs: List[ParamSpec],
                                 num_samples: int,
                                 rng: np.random.Generator) -> FitResult:
    """A ``FitResult`` standing in for a fit that never ran.

    Sample 0 is the declared init (the MAP: the only sample with
    log-prob 0); the remaining ``num_samples - 1`` rows are uniform
    draws over each param's declared box in FIT space (geometric for
    log-scale params), so the ensemble machinery (posterior
    subsampling) yields members spread over the agent's plausible
    intervals rather than around a fitted point. A param without a
    finite box on both sides stays at its init in every sample: an
    interval it never declared is not something to sample from.
    """
    assert num_samples >= 1
    names = [s.name for s in param_specs]
    inits = np.array([s.init_value for s in param_specs], dtype=float)
    samples = np.tile(inits, (num_samples, 1))
    if num_samples > 1 and param_specs:
        lo_z, hi_z = fit_space_bounds(param_specs)
        boxed = np.isfinite(lo_z) & np.isfinite(hi_z) & (hi_z > lo_z)
        draws_z = rng.uniform(lo_z[boxed],
                              hi_z[boxed],
                              size=(num_samples - 1, int(boxed.sum())))
        boxed_specs = [s for s, b in zip(param_specs, boxed) if b]
        samples[1:, boxed] = rows_from_fit_space(boxed_specs, draws_z)
    log_probs = np.full(num_samples, -1.0)
    log_probs[0] = 0.0
    return FitResult(names=names,
                     samples=samples,
                     log_probs=log_probs,
                     scales=[s.scale for s in param_specs])
