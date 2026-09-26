"""The parameter factor ``q(theta)`` of the agent's belief.

The agent's belief over the parameters and the current model state
factors as ``q(theta) q(x_t | H_t) delta(z_t - Z_theta(H_t))`` (paper
Section 3.2 and Appendix C). This module builds ``q(theta)`` from a
finished rollout fit and draws from it.

``q(theta)`` approximates the posterior of the fit's own model: the
Gaussian prior the fit folded in, and the replay likelihood with its
noise variance treated as unknown, bounded below by the declared noise,
at its maximum-likelihood value. That value scales the declared variance
by ``lambda = max(1, SSE_min / (N sigma_n^2))``, so a program that cannot
fit the recordings to within its noise model gets a proportionally wider
posterior, and recordings no single parameter setting reconciles widen it
through the same term.

Each parameter's factor is the posterior along a line through the MAP,
the other parameters held at the MAP, evaluated on a grid refined where
the posterior has mass. For a Gaussian posterior with precision
``Lambda`` this conditional has variance ``1 / Lambda_jj``, the
mean-field solution, so the product of lines is a stated mean-field
approximation. Local derivatives of contact-rich replays are unreliable,
which is why the lines are evaluated rather than read from a Jacobian.
The grid stays inside the declared bounds, so every draw does too.

A discrete parameter's line is its set of values: the target is
evaluated at each value with the other parameters at the MAP and
normalized, and draws take values in proportion to it. A discrete
parameter with unbounded or too many values is held at the MAP.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt

from predicators.code_sim_learning.fit_space import ParamSpec, \
    fit_space_bounds, scalar_from_fit_space, scalar_to_fit_space

logger = logging.getLogger(__name__)

# Step reductions allowed when the first probe of a line already lies
# beyond the cutoff (a posterior much sharper than the prior).
_MAX_SHRINKS = 6
# Refinement stops once no grid interval holds more than this fraction
# of the line's mass.
_MAX_INTERVAL_MASS = 0.1
# Points per evaluated interval on which a line's density is stored.
_SUBDIVISIONS = 16

# One pass of the fit's objective: the residual vector at params.
ResidualsFn = Callable[[Dict[str, float]], np.ndarray]


@dataclass(frozen=True)
class BeliefConfig:
    """Settings of the parameter belief (the ``belief_*`` flags)."""

    num_draws: int = 16
    line_cutoff: float = 12.5
    line_max_evals: int = 24

    @classmethod
    def from_cfg(cls) -> BeliefConfig:
        """Snapshot the current global flags."""
        # pylint: disable-next=import-outside-toplevel
        from predicators.settings import CFG
        return cls(num_draws=int(CFG.belief_joint_draws),
                   line_cutoff=float(CFG.belief_line_cutoff),
                   line_max_evals=int(CFG.belief_line_max_evals))


@dataclass(frozen=True)
class LinePosterior:
    """A normalized piecewise-linear density on a sorted grid.

    A single grid point is a point mass.
    """

    grid: np.ndarray
    density: np.ndarray

    @classmethod
    def from_neg_log(cls, grid: npt.ArrayLike,
                     neg_log: npt.ArrayLike) -> LinePosterior:
        """Normalize ``exp(-neg_log)`` over ``grid`` (any order).

        The negative log density is interpolated linearly between the
        evaluated points (a piecewise-exponential density) and resampled
        onto a finer grid: interpolating the density itself between the
        widely spaced tail probes would overstate the tails.
        """
        z = np.asarray(grid, dtype=float)
        g = np.asarray(neg_log, dtype=float)
        order = np.argsort(z, kind="stable")
        z, g = z[order], g[order]
        keep = np.concatenate([[True], np.diff(z) > 0])
        z, g = z[keep], g[keep]
        best = int(np.argmin(g))
        if z.size < 2:
            return cls(z[best:best + 1], np.ones(1))
        fine = np.concatenate([
            np.linspace(a, b, _SUBDIVISIONS, endpoint=False)
            for a, b in zip(z[:-1], z[1:])
        ] + [z[-1:]])
        g = np.interp(fine, z, g)
        z = fine
        dens = np.exp(-(g - np.min(g)))
        mass = float(np.trapz(dens, z))
        if not np.isfinite(mass) or mass <= 0.0:
            return cls(z[best:best + 1], np.ones(1))
        return cls(z, dens / mass)

    @property
    def is_point(self) -> bool:
        """Whether all mass sits on one value."""
        return self.grid.size < 2

    def _pieces(
        self
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        a, b = self.grid[:-1], self.grid[1:]
        fa, fb = self.density[:-1], self.density[1:]
        return a, b, fa, fb, (b - a) * (fa + fb) / 2.0

    def mean(self) -> float:
        """The exact mean of the piecewise-linear density."""
        if self.is_point:
            return float(self.grid[0])
        a, b, fa, fb, _ = self._pieces()
        return float(
            np.sum((b - a) / 6.0 * (fa * (2 * a + b) + fb * (a + 2 * b))))

    def variance(self) -> float:
        """The exact variance of the piecewise-linear density."""
        if self.is_point:
            return 0.0
        a, b, fa, fb, _ = self._pieces()
        second = np.sum(
            (b - a) / 12.0 * (fa * (3 * a * a + 2 * a * b + b * b) + fb *
                              (a * a + 2 * a * b + 3 * b * b)))
        return max(0.0, float(second) - self.mean()**2)

    def _invert(self, u: np.ndarray) -> np.ndarray:
        """Map cumulative masses ``u`` in [0, 1] to grid values."""
        a, b, fa, fb, mass = self._pieces()
        cum = np.cumsum(mass)
        u = np.clip(u, 0.0, 1.0) * cum[-1]
        idx = np.clip(np.searchsorted(cum, u, side="right"), 0, mass.size - 1)
        width = b[idx] - a[idx]
        # The mass needed inside the chosen piece, per unit width: solve
        # fa t + (fb - fa) t^2 / 2 = c for t in [0, 1], in the form that
        # stays stable as the slope vanishes.
        c = np.clip(u - (cum[idx] - mass[idx]), 0.0, None) / width
        slope = fb[idx] - fa[idx]
        root = np.sqrt(np.clip(fa[idx]**2 + 2.0 * slope * c, 0.0, None))
        denom = fa[idx] + root
        safe = np.where(denom > 0.0, denom, 1.0)
        t = np.where(denom > 0.0, 2.0 * c / safe, 0.0)
        return a[idx] + np.clip(t, 0.0, 1.0) * width

    def sample(self, rng: np.random.Generator, size: int) -> np.ndarray:
        """Draw ``size`` values by inverse-CDF sampling."""
        if self.is_point:
            return np.full(size, float(self.grid[0]))
        return self._invert(rng.uniform(0.0, 1.0, size))

    def quantile(self, q: float) -> float:
        """The ``q`` quantile."""
        if self.is_point:
            return float(self.grid[0])
        return float(self._invert(np.array([q]))[0])


@dataclass(frozen=True)
class DiscretePosterior:
    """A normalized distribution over a discrete parameter's values."""

    values: np.ndarray
    probs: np.ndarray

    @classmethod
    def from_neg_log(cls, values: npt.ArrayLike,
                     neg_log: npt.ArrayLike) -> DiscretePosterior:
        """Normalize ``exp(-neg_log)`` over ``values`` (any order)."""
        v = np.asarray(values, dtype=float)
        g = np.asarray(neg_log, dtype=float)
        order = np.argsort(v, kind="stable")
        v, g = v[order], g[order]
        weights = np.exp(-(g - np.min(g)))
        return cls(v, weights / np.sum(weights))

    @property
    def is_point(self) -> bool:
        """Whether all mass sits on one value."""
        return self.values.size < 2

    def mean(self) -> float:
        """The mean value."""
        return float(np.dot(self.values, self.probs))

    def variance(self) -> float:
        """The variance of the value."""
        return max(0.0,
                   float(np.dot(self.values**2, self.probs)) - self.mean()**2)

    def sample(self, rng: np.random.Generator, size: int) -> np.ndarray:
        """Draw ``size`` values in proportion to their probabilities."""
        return rng.choice(self.values, size=size, p=self.probs)

    def quantile(self, q: float) -> float:
        """The smallest value whose cumulative probability reaches ``q``."""
        cum = np.cumsum(self.probs)
        idx = int(np.searchsorted(cum, min(max(q, 0.0), 1.0) - 1e-12))
        return float(self.values[min(idx, self.values.size - 1)])


@dataclass
class ParameterBelief:
    """``q(theta)``: independent line posteriors through the MAP, with draws.

    ``lines`` live in FIT coordinates (log for log-scale parameters);
    ``map_estimate`` and ``draws`` are in external units. ``discrete``
    holds the value distributions of discrete parameters, and ``held``
    parameters (discrete ones that could not be enumerated) keep the MAP
    in every draw.
    """

    names: List[str]
    scales: List[str]
    map_estimate: Dict[str, float]
    lines: Dict[str, LinePosterior]
    noise_scale: float
    draws: np.ndarray
    held: List[str] = field(default_factory=list)
    evaluations: int = 0
    discrete: Dict[str, DiscretePosterior] = field(default_factory=dict)

    @property
    def num_draws(self) -> int:
        """How many joint parameter draws the belief carries."""
        return int(self.draws.shape[0])

    def draw_dicts(self) -> List[Dict[str, float]]:
        """The draws as parameter dicts, in draw order."""
        return [{name: float(row[j])
                 for j, name in enumerate(self.names)} for row in self.draws]

    def sample(self, rng: np.random.Generator,
               num: int) -> List[Dict[str, float]]:
        """``num`` new joint draws (fresh, independent of ``draws``).

        Used to re-estimate a plan chosen on the stored draws, where the
        selection biases the estimate on those same draws.
        """
        rows: List[Dict[str, float]] = [{} for _ in range(num)]
        for j, name in enumerate(self.names):
            if name in self.held or (name not in self.lines
                                     and name not in self.discrete):
                column = np.full(num, float(self.map_estimate[name]))
            elif name in self.discrete:
                column = self.discrete[name].sample(rng, num)
            else:
                z = self.lines[name].sample(rng, num)
                column = np.exp(z) if self.scales[j] == "log" else z
            for row, value in zip(rows, column):
                row[name] = float(value)
        return rows

    def interval(self,
                 name: str,
                 coverage: float = 0.68) -> Tuple[float, float]:
        """The central ``coverage`` interval of ``name``'s factor.

        Computed from the line posterior (external units), not from the
        finite draws.
        """
        tail = (1.0 - coverage) / 2.0
        if name in self.discrete:
            values = self.discrete[name]
            return values.quantile(tail), values.quantile(1.0 - tail)
        line = self.lines.get(name)
        if line is None:
            value = self.map_estimate[name]
            return value, value
        scale = self.scales[self.names.index(name)]
        ends = (line.quantile(tail), line.quantile(1.0 - tail))
        if scale == "log":
            return float(np.exp(ends[0])), float(np.exp(ends[1]))
        return float(ends[0]), float(ends[1])

    def describe(self) -> List[str]:
        """One readable line per parameter, plus the noise-level note."""
        lines = []
        if self.noise_scale > 1.0:
            lines.append(
                f"The program misfits the recordings {self.noise_scale:.3g}x "
                "beyond the declared noise, so the posterior uses that "
                "estimated noise level (widths grow by its square root).")
        for name in self.names:
            value = self.map_estimate[name]
            if name in self.held:
                lines.append(f"  {name}: {value:.4g} (discrete, held)")
                continue
            if name in self.discrete:
                dist = self.discrete[name]
                top = np.argsort(-dist.probs, kind="stable")[:4]
                shares = ", ".join(f"{dist.values[i]:.4g}: "
                                   f"{dist.probs[i]:.2f}" for i in top)
                lines.append(f"  {name}: discrete, posterior {shares}")
                continue
            lo, hi = self.interval(name)
            lines.append(f"  {name}: most likely {value:.4g}; 68% "
                         f"posterior interval [{lo:.4g}, {hi:.4g}]")
        return lines

    def to_dict(self) -> Dict[str, Any]:
        """A picklable, JSON-friendly snapshot for checkpoints."""
        return {
            "names": list(self.names),
            "scales": list(self.scales),
            "map_estimate": dict(self.map_estimate),
            "lines": {
                n: [lp.grid.tolist(), lp.density.tolist()]
                for n, lp in self.lines.items()
            },
            "noise_scale": self.noise_scale,
            "draws": self.draws.tolist(),
            "held": list(self.held),
            "evaluations": self.evaluations,
            "discrete": {
                n: [d.values.tolist(), d.probs.tolist()]
                for n, d in self.discrete.items()
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ParameterBelief:
        """Inverse of :meth:`to_dict`."""
        return cls(names=list(data["names"]),
                   scales=list(data["scales"]),
                   map_estimate=dict(data["map_estimate"]),
                   lines={
                       n: LinePosterior(np.asarray(g, dtype=float),
                                        np.asarray(d, dtype=float))
                       for n, (g, d) in data["lines"].items()
                   },
                   noise_scale=float(data["noise_scale"]),
                   draws=np.asarray(data["draws"], dtype=float).reshape(
                       -1, len(data["names"])),
                   held=list(data.get("held", [])),
                   evaluations=int(data.get("evaluations", 0)),
                   discrete={
                       n: DiscretePosterior(np.asarray(v, dtype=float),
                                            np.asarray(p, dtype=float))
                       for n, (v, p) in data.get("discrete", {}).items()
                   })


def stable_seed(*parts: Any) -> int:
    """A 63-bit seed from ``parts``, stable across processes."""
    text = "|".join(repr(p) for p in parts).encode()
    return int.from_bytes(hashlib.sha256(text).digest()[:8], "little") >> 1


def build_parameter_belief(
    specs: Sequence[ParamSpec],
    map_params: Dict[str, float],
    residuals: ResidualsFn,
    noise_sigma: float,
    prior_centers: Dict[str, float],
    prior_sigmas: Dict[str, float],
    config: BeliefConfig,
    seed: int,
) -> ParameterBelief:
    """Build ``q(theta)`` around ``map_params`` and draw from it.

    ``residuals`` is the fit's objective (scaled residuals over the
    segments it pooled) and ``noise_sigma`` the Gaussian width it scores
    them with; ``prior_centers`` / ``prior_sigmas`` (fit space) are the
    Gaussian prior it folded in.
    """
    assert noise_sigma > 0.0, noise_sigma
    at_map = np.asarray(residuals(dict(map_params)), dtype=float)
    sse_min = float(np.dot(at_map, at_map))
    noise_scale = 1.0
    if at_map.size:
        noise_scale = max(1.0, sse_min / (at_map.size * noise_sigma**2))
    # Negative log likelihood per unit of summed squared residual, at the
    # estimated noise level.
    k = 1.0 / (2.0 * noise_scale * noise_sigma**2)
    evaluations = 0
    lines: Dict[str, LinePosterior] = {}
    discrete: Dict[str, DiscretePosterior] = {}
    held: List[str] = []
    for spec in specs:
        center = prior_centers.get(spec.name)
        sigma_p = prior_sigmas.get(spec.name)
        prior: Optional[Tuple[float, float]] = None
        if (center is not None and sigma_p is not None and np.isfinite(sigma_p)
                and sigma_p > 0):
            prior = (float(center), float(sigma_p))
        line = _LineEvaluator(spec, map_params, residuals, sse_min, k, prior)
        if spec.discrete:
            values = _discrete_values(spec, config.line_max_evals)
            if values is None:
                held.append(spec.name)
                continue
            neg_log = [
                line(scalar_to_fit_space(spec, float(v))) for v in values
            ]
            evaluations += line.count
            discrete[spec.name] = DiscretePosterior.from_neg_log(
                values, neg_log)
            continue
        lo_arr, hi_arr = fit_space_bounds([spec])
        _trace_line(line, float(lo_arr[0]), float(hi_arr[0]), config)
        evaluations += line.count
        grid = line.points()
        lines[spec.name] = LinePosterior.from_neg_log(grid,
                                                      [line(z) for z in grid])

    belief = _with_draws(specs, map_params, lines, held, noise_scale,
                         evaluations, config, seed, discrete)
    logger.info(
        "Parameter belief: %d draws over %d parameters from %d line "
        "evaluations (noise level %.3gx the declared one).", belief.num_draws,
        len(specs), evaluations, noise_scale)
    return belief


def prior_belief(
    specs: Sequence[ParamSpec],
    centers: Dict[str, float],
    prior_sigmas: Dict[str, float],
    config: BeliefConfig,
    seed: int,
) -> ParameterBelief:
    """``q(theta)`` before any fit of the current program: the prior itself.

    ``centers`` are in external units and ``prior_sigmas`` in fit space,
    the same Gaussian prior a fit would fold in, restricted to the
    declared bounds.
    """
    lines: Dict[str, LinePosterior] = {}
    discrete: Dict[str, DiscretePosterior] = {}
    held: List[str] = []
    for spec in specs:
        center = scalar_to_fit_space(spec, float(centers[spec.name]))
        sigma = float(prior_sigmas[spec.name])
        if spec.discrete:
            values = _discrete_values(spec, config.line_max_evals)
            if values is None:
                held.append(spec.name)
                continue
            zs = np.array(
                [scalar_to_fit_space(spec, float(v)) for v in values])
            neg_log = (0.5 * ((zs - center) / sigma)**2 if np.isfinite(sigma)
                       and sigma > 0 else np.zeros_like(zs))
            discrete[spec.name] = DiscretePosterior.from_neg_log(
                values, neg_log)
            continue
        lo_arr, hi_arr = fit_space_bounds([spec])
        lo = max(float(lo_arr[0]), center - 6.0 * sigma)
        hi = min(float(hi_arr[0]), center + 6.0 * sigma)
        if not hi > lo:
            lines[spec.name] = LinePosterior(np.array([center]), np.ones(1))
            continue
        grid = np.linspace(lo, hi, 129)
        lines[spec.name] = LinePosterior.from_neg_log(
            grid, 0.5 * ((grid - center) / sigma)**2)
    return _with_draws(specs, centers, lines, held, 1.0, 0, config, seed,
                       discrete)


def _with_draws(
    specs: Sequence[ParamSpec],
    map_params: Dict[str, float],
    lines: Dict[str, LinePosterior],
    held: List[str],
    noise_scale: float,
    evaluations: int,
    config: BeliefConfig,
    seed: int,
    discrete: Optional[Dict[str,
                            DiscretePosterior]] = None) -> ParameterBelief:
    """Assemble the belief and draw ``config.num_draws`` joint samples."""
    discrete = discrete or {}
    rng = np.random.default_rng(seed)
    draws = np.empty((max(config.num_draws, 0), len(specs)))
    for j, spec in enumerate(specs):
        if spec.name in held:
            draws[:, j] = float(map_params[spec.name])
            continue
        if spec.name in discrete:
            draws[:, j] = discrete[spec.name].sample(rng, draws.shape[0])
            continue
        column = lines[spec.name].sample(rng, draws.shape[0])
        draws[:, j] = [scalar_from_fit_space(spec, z) for z in column]
    return ParameterBelief(
        names=[s.name for s in specs],
        scales=[s.scale for s in specs],
        map_estimate={s.name: float(map_params[s.name])
                      for s in specs},
        lines=lines,
        noise_scale=float(noise_scale),
        draws=draws,
        held=held,
        evaluations=evaluations,
        discrete=discrete)


def _discrete_values(spec: ParamSpec, limit: int) -> Optional[np.ndarray]:
    """The integer values a discrete parameter can take within its bounds.

    None when a bound is missing or the set has more than ``limit``
    values (each costs one objective pass).
    """
    if spec.lo is None or spec.hi is None:
        return None
    lo, hi = int(np.ceil(spec.lo)), int(np.floor(spec.hi))
    if hi < lo or hi - lo + 1 > max(limit, 1):
        return None
    return np.arange(lo, hi + 1, dtype=float)


class _LineEvaluator:
    """The negative log posterior along one parameter's line.

    Values are relative to the MAP; each evaluation runs one objective
    pass with only this parameter moved.
    """

    def __init__(self, spec: ParamSpec, map_params: Dict[str, float],
                 residuals: ResidualsFn, sse_min: float, k: float,
                 prior: Optional[Tuple[float, float]]) -> None:
        self.spec = spec
        self.map_params = map_params
        self.residuals = residuals
        self.sse_min = sse_min
        self.k = k
        self.prior = prior
        self.z0 = scalar_to_fit_space(spec, float(map_params[spec.name]))
        self.sigma_p = prior[1] if prior is not None else None
        self.sse: Dict[float, float] = {self.z0: sse_min}
        self.count = 0

    def _prior(self, z: float) -> float:
        if self.prior is None:
            return 0.0
        center, sigma = self.prior
        return 0.5 * ((z - center) / sigma)**2

    def __call__(self, z: float) -> float:
        z = float(z)
        if z not in self.sse:
            params = dict(self.map_params)
            params[self.spec.name] = scalar_from_fit_space(self.spec, z)
            res = np.asarray(self.residuals(params), dtype=float)
            self.sse[z] = float(np.dot(res, res))
            self.count += 1
        return ((self.sse[z] - self.sse_min) * self.k + self._prior(z) -
                self._prior(self.z0))

    def points(self) -> List[float]:
        """The evaluated grid, sorted."""
        return sorted(self.sse)


def _trace_line(line: _LineEvaluator, lo: float, hi: float,
                config: BeliefConfig) -> None:
    """Evaluate a parameter's line until its posterior mass is resolved.

    Each side is probed outward from the MAP, first shrinking the step
    until a probe falls inside the cutoff, then doubling it until the
    cutoff or the declared bound is reached. The remaining budget
    bisects the grid interval holding the most mass.
    """
    z0 = line.z0
    if line.sigma_p is not None:
        h0 = 0.25 * line.sigma_p
    else:
        h0 = 0.1 * max(1.0, abs(z0))
    if np.isfinite(hi - lo):
        h0 = min(h0, (hi - lo) / 8.0)
    budget = config.line_max_evals
    cutoff = config.line_cutoff
    for side in (-1.0, 1.0):
        edge = lo if side < 0 else hi
        if (edge - z0) * side <= 0.0:
            continue

        def probe(step: float,
                  edge: float = edge,
                  side: float = side) -> Tuple[float, float]:
            z = z0 + side * step
            if (z - edge) * side > 0.0:
                z = edge
            return z, line(z)

        step = h0
        z, g = probe(step)
        shrinks = 0
        while g > cutoff and shrinks < _MAX_SHRINKS and line.count < budget:
            step /= 4.0
            z, g = probe(step)
            shrinks += 1
        while g <= cutoff and z != edge and line.count < budget:
            step *= 2.0
            z, g = probe(step)
    while line.count < budget:
        grid = np.array(line.points())
        if grid.size < 2:
            return
        neg_log = np.array([line(z) for z in grid])
        dens = np.exp(-(neg_log - np.min(neg_log)))
        mass = np.diff(grid) * np.maximum(dens[:-1], dens[1:])
        total = float(np.sum(mass))
        if total <= 0.0:
            return
        widest = int(np.argmax(mass))
        if mass[widest] / total <= _MAX_INTERVAL_MASS:
            return
        mid = 0.5 * (grid[widest] + grid[widest + 1])
        if mid in (grid[widest], grid[widest + 1]):
            return
        line(float(mid))
