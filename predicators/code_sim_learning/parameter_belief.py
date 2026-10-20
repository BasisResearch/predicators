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
import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Collection, Dict, Generator, Iterator, \
    List, Optional, Sequence, Set, Tuple

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
# A free parameter's line (see build_parameter_belief) stops after its
# outward probes when none of them moves the likelihood by more than this
# (in nats): the recordings do not test it, and its prior alone shapes it.
_FLAT_LIKELIHOOD = 0.05

# One pass of the fit's objective: the residual vector at params.
ResidualsFn = Callable[[Dict[str, float]], np.ndarray]
# Scores a list of parameter dicts together, each as a ResidualsFn would.
BatchResidualsFn = Callable[[List[Dict[str, float]]], List[np.ndarray]]


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

    def cdf(self, values: npt.ArrayLike) -> np.ndarray:
        """The mass at or below each of ``values``."""
        z = np.asarray(values, dtype=float)
        if self.is_point:
            return (z >= self.grid[0]).astype(float)
        a, b, fa, fb, mass = self._pieces()
        before = np.concatenate([[0.0], np.cumsum(mass)])
        idx = np.clip(
            np.searchsorted(self.grid, z, side="right") - 1, 0, a.size - 1)
        width = b[idx] - a[idx]
        t = np.clip((z - a[idx]) / width, 0.0, 1.0)
        inside = width * t * (fa[idx] + (fb[idx] - fa[idx]) * t / 2.0)
        return np.clip((before[idx] + inside) / before[-1], 0.0, 1.0)

    def transport(self, values: npt.ArrayLike,
                  target: LinePosterior) -> np.ndarray:
        """``values`` drawn from this density, each moved to the value that
        holds the same mass below it under ``target``: draws of ``target`` in
        the same order."""
        return target.quantiles(self.cdf(values))

    def tempered(self, factor: float) -> LinePosterior:
        """This density at ``factor`` times its temperature: raised to the
        power ``1 / factor`` on the same grid and normalized again."""
        if self.is_point or factor == 1.0:
            return self
        dens = np.power(self.density / np.max(self.density), 1.0 / factor)
        return LinePosterior(self.grid,
                             dens / float(np.trapz(dens, self.grid)))

    def quantiles(self, q: npt.ArrayLike) -> np.ndarray:
        """The quantile at each of ``q``."""
        u = np.asarray(q, dtype=float)
        if self.is_point:
            return np.full(u.shape, float(self.grid[0]))
        return self._invert(u)

    def quantile(self, q: float) -> float:
        """The ``q`` quantile."""
        return float(self.quantiles(np.array([q]))[0])

    def mode(self) -> float:
        """The value of highest density."""
        return float(self.grid[int(np.argmax(self.density))])

    def highest_density_interval(self, coverage: float) -> Tuple[float, float]:
        """The span of the densest pieces that together hold at least
        ``coverage`` of the mass.

        For a unimodal density this is the shortest interval with that
        mass, and it always contains the mode; a central interval
        excludes a mode that sits at a bound.
        """
        if self.is_point:
            return float(self.grid[0]), float(self.grid[0])
        a, b, fa, fb, mass = self._pieces()
        order = np.argsort(-(fa + fb), kind="stable")
        cum = np.cumsum(mass[order])
        count = int(np.searchsorted(cum, coverage * cum[-1])) + 1
        chosen = order[:min(count, order.size)]
        return float(np.min(a[chosen])), float(np.max(b[chosen]))


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

    def mode(self) -> float:
        """The most probable value."""
        return float(self.values[int(np.argmax(self.probs))])


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
    # A mixture over basins (mix_beliefs): each draw comes from one
    # component, picked by weight, and the factors above are the
    # components' weighted mixtures. Empty for a single basin.
    components: List["ParameterBelief"] = field(default_factory=list)
    weights: List[float] = field(default_factory=list)
    # The fit's misfit (SSE above the declared noise's expected SSE)
    # when the temperature counts it as one persistent error
    # (build_parameter_belief); 0 when it does not.
    misfit: float = 0.0
    # The declared noise channel's expected SSE over the fitted
    # recordings, which the misfit is measured above; 0 when unknown.
    noise_sse: float = 0.0
    # The factor the temperature is multiplied by beyond the fit's
    # (inflated): the lines are tempered by it and the draws moved with
    # them; 1 for the belief as fitted.
    inflation: float = 1.0
    # The parameters whose lines the fit's deadline cut short
    # (build_parameter_belief's deadline): each keeps its prior, or a
    # declared parameter the part of its line that was traced.
    cut_short: Tuple[str, ...] = ()
    # How many of the fit's basins the deadline left untraced: the mixture
    # holds the others, and their draws make up the shares of these.
    untraced_basins: int = 0

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
        selection biases the estimate on those same draws. A mixture
        picks a component per draw and samples it jointly.
        """
        if self.components:
            picks = rng.choice(len(self.components), size=num, p=self.weights)
            mixed: List[Dict[str, float]] = [{} for _ in range(num)]
            for k, component in enumerate(self.components):
                chosen = np.flatnonzero(picks == k)
                for i, row in zip(chosen,
                                  component.sample(rng, int(chosen.size))):
                    mixed[int(i)] = row
            return mixed
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

    def inflated(self, factor: float) -> ParameterBelief:
        """This belief at ``factor`` times its temperature.

        Each parameter's line density is raised to the power ``1 /
        factor`` on the range its fit traced (a Gaussian factor widens
        by ``sqrt(factor)``, up to that range), and each draw moves to
        the value that holds the same cumulative mass under the tempered
        line, so the draws keep their order and their pairing across
        parameters. A mixture tempers each component's factors and its
        draws. Discrete and held parameters keep their values. A factor
        of 1 or less returns the belief itself.
        """
        if factor <= 1.0:
            return self
        if not self.components:
            lines = {
                name: line.tempered(factor)
                for name, line in self.lines.items()
            }
            return replace(self,
                           lines=lines,
                           draws=_transported(self, self.draws, lines),
                           inflation=self.inflation * factor)
        parts = [c.inflated(factor) for c in self.components]
        # The draws run through the components in order (mix_beliefs),
        # each block moving under its own component's tempered lines.
        draws = np.array(self.draws, dtype=float, copy=True)
        start = 0
        for old, new in zip(self.components, parts):
            stop = start + old.num_draws
            draws[start:stop] = _transported(old, draws[start:stop], new.lines)
            start = stop
        lines = {
            name: _mix_lines([c.lines[name] for c in parts], self.weights)
            for name in self.lines if all(name in c.lines for c in parts)
        }
        return replace(self,
                       components=parts,
                       lines=lines,
                       draws=draws,
                       inflation=self.inflation * factor)

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

    def most_likely(self, name: str) -> float:
        """The mode of ``name``'s factor (external units); the fit's point
        estimate for a held parameter."""
        if name in self.discrete:
            return self.discrete[name].mode()
        line = self.lines.get(name)
        if line is None:
            return float(self.map_estimate[name])
        mode = line.mode()
        if self.scales[self.names.index(name)] == "log":
            return float(np.exp(mode))
        return float(mode)

    def highest_density_interval(self,
                                 name: str,
                                 coverage: float = 0.68
                                 ) -> Tuple[float, float]:
        """The highest-density ``coverage`` interval of ``name``'s factor
        (external units), which contains its mode."""
        line = self.lines.get(name)
        if line is None or name in self.discrete:
            return self.interval(name, coverage)
        lo, hi = line.highest_density_interval(coverage)
        if self.scales[self.names.index(name)] == "log":
            return float(np.exp(lo)), float(np.exp(hi))
        return float(lo), float(hi)

    def describe(self) -> List[str]:
        """One readable line per parameter, plus the noise-level note.

        Each line gives the factor's own most likely value and its 68%
        highest-density interval; the fit's point estimate, which the
        planning model runs at, is named when it lies outside that
        interval.
        """
        lines = []
        if len(self.components) > 1:
            lines.append(
                f"The recordings fit {len(self.components)} distinct "
                "parameter sets about as well as the model fits them at "
                "all; the belief mixes them, and each draw comes from one "
                "of them.")
        if self.noise_scale > 1.0 and self.misfit > 0.0:
            multiple = ""
            if self.noise_sse > 0.0:
                multiple = (
                    f" ({self.misfit / self.noise_sse:.3g} times the "
                    f"{self.noise_sse:.3g} the declared noise alone leaves; "
                    "a model with every mechanism right leaves about the "
                    "noise's own SSE, so a large multiple points to a "
                    "missing or wrong mechanism)")
            lines.append(
                f"The program's best fit leaves an SSE of {self.misfit:.3g} "
                f"above what the declared noise explains{multiple}, and that "
                "error persists through each recording, so the posterior "
                "counts it as one error: parameter values whose SSE is "
                "within about that much of the best fit's are about as "
                f"likely (temperature {self.noise_scale:.3g}).")
        elif self.noise_scale > 1.0:
            lines.append(
                f"The program misfits the recordings {self.noise_scale:.3g}x "
                "beyond the declared noise, so the posterior uses that "
                "estimated noise level (widths grow by its square root).")
        if self.untraced_basins:
            lines.append(
                "The fit's time ran out before it traced the lines around "
                f"{self.untraced_basins} more of the parameter sets that fit "
                "the recordings about as well; the belief mixes the "
                f"{len(self.components)} it traced.")
        if self.cut_short:
            lines.append(
                "The fit's time ran out before it finished the lines of "
                f"{', '.join(self.cut_short)}: each keeps its prior, or the "
                "part of its line that was traced when that part brackets "
                "the point estimate.")
        if self.inflation > 1.0:
            lines.append(
                "A real step's outcome lay outside the draws its rehearsal "
                "predicted, so the belief runs at "
                f"{self.inflation:.3g} times the fit's temperature: each "
                "parameter's density is flattened over the range its fit "
                "traced.")
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
            lo, hi = self.highest_density_interval(name)
            line = (f"  {name}: most likely {self.most_likely(name):.4g}; 68% "
                    f"posterior interval [{lo:.4g}, {hi:.4g}]")
            if not lo <= value <= hi:
                line += (f"; the fit's point estimate {value:.4g} lies "
                         "outside it")
            lines.append(line)
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
            "components": [c.to_dict() for c in self.components],
            "weights": list(self.weights),
            "misfit": self.misfit,
            "noise_sse": self.noise_sse,
            "inflation": self.inflation,
            "cut_short": list(self.cut_short),
            "untraced_basins": self.untraced_basins,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ParameterBelief:
        """Inverse of :meth:`to_dict`."""
        return cls(
            names=list(data["names"]),
            scales=list(data["scales"]),
            map_estimate=dict(data["map_estimate"]),
            lines={
                n: LinePosterior(np.asarray(g, dtype=float),
                                 np.asarray(d, dtype=float))
                for n, (g, d) in data["lines"].items()
            },
            noise_scale=float(data["noise_scale"]),
            draws=np.asarray(data["draws"],
                             dtype=float).reshape(-1, len(data["names"])),
            held=list(data.get("held", [])),
            evaluations=int(data.get("evaluations", 0)),
            discrete={
                n: DiscretePosterior(np.asarray(v, dtype=float),
                                     np.asarray(p, dtype=float))
                for n, (v, p) in data.get("discrete", {}).items()
            },
            components=[cls.from_dict(c) for c in data.get("components", [])],
            weights=[float(w) for w in data.get("weights", [])],
            misfit=float(data.get("misfit", 0.0)),
            noise_sse=float(data.get("noise_sse", 0.0)),
            inflation=float(data.get("inflation", 1.0)),
            cut_short=tuple(data.get("cut_short", ())),
            untraced_basins=int(data.get("untraced_basins", 0)))


def _transported(belief: ParameterBelief, rows: np.ndarray,
                 lines: Dict[str, LinePosterior]) -> np.ndarray:
    """``rows`` (draws of ``belief`` in external units, one column per
    parameter), each continuous value moved from ``belief``'s line to ``lines``
    at the same cumulative mass."""
    out = np.array(rows, dtype=float, copy=True)
    for j, name in enumerate(belief.names):
        line = belief.lines.get(name)
        if (name in belief.held or name in belief.discrete or line is None
                or line.is_point or name not in lines):
            continue
        log = belief.scales[j] == "log"
        z = np.log(out[:, j]) if log else out[:, j]
        moved = line.transport(z, lines[name])
        out[:, j] = np.exp(moved) if log else moved
    return out


def _mix_lines(lines: Sequence[LinePosterior],
               weights: Sequence[float]) -> LinePosterior:
    """The weighted mixture of line densities on the union of their grids; a
    point mass becomes a narrow triangle."""
    parts = []
    for line, weight in zip(lines, weights):
        if line.is_point:
            value = float(line.grid[0])
            eps = 1e-9 * max(1.0, abs(value))
            parts.append((np.array([value - eps, value, value + eps]),
                          np.array([0.0, 1.0 / eps, 0.0]), weight))
        else:
            parts.append((line.grid, line.density, weight))
    grid = np.unique(np.concatenate([g for g, _d, _w in parts]))
    density = np.zeros_like(grid)
    for g, d, weight in parts:
        density += weight * np.interp(grid, g, d, left=0.0, right=0.0)
    mass = float(np.trapz(density, grid))
    if not np.isfinite(mass) or mass <= 0.0:
        return lines[0]
    return LinePosterior(grid, density / mass)


def _mix_discrete(parts: Sequence[DiscretePosterior],
                  weights: Sequence[float]) -> DiscretePosterior:
    """The weighted mixture of discrete posteriors over their values."""
    values = np.unique(np.concatenate([p.values for p in parts]))
    probs = np.zeros_like(values)
    for part, weight in zip(parts, weights):
        for value, prob in zip(part.values, part.probs):
            probs[int(np.searchsorted(values, value))] += weight * prob
    return DiscretePosterior(values, probs / np.sum(probs))


def mix_beliefs(components: Sequence[ParameterBelief],
                weights: Sequence[float]) -> ParameterBelief:
    """The mixture of beliefs over the same parameters, one per basin.

    A draw picks a component by weight and samples it jointly, so it
    never pairs one basin's value of a parameter with another basin's
    value of the next. Each parameter's factor, which the report reads,
    is the weighted mixture of the components' factors, and a parameter
    every component holds at different values becomes discrete over
    them. The stored draws are the components' own, and the point
    estimate is the first component's.
    """
    first = components[0]
    total = float(sum(weights))
    shares = [float(w) / total for w in weights]
    lines = {
        name: _mix_lines([c.lines[name] for c in components], shares)
        for name in first.lines if all(name in c.lines for c in components)
    }
    discrete = {
        name: _mix_discrete([c.discrete[name] for c in components], shares)
        for name in first.discrete
        if all(name in c.discrete for c in components)
    }
    held = []
    for name in first.held:
        values = [float(c.map_estimate[name]) for c in components]
        if len(set(values)) == 1:
            held.append(name)
        else:
            discrete[name] = _mix_discrete([
                DiscretePosterior(np.array([v]), np.array([1.0]))
                for v in values
            ], shares)
    return ParameterBelief(
        names=list(first.names),
        scales=list(first.scales),
        map_estimate=dict(first.map_estimate),
        lines=lines,
        noise_scale=first.noise_scale,
        draws=np.vstack([c.draws for c in components]),
        held=held,
        evaluations=sum(c.evaluations for c in components),
        discrete=discrete,
        components=list(components),
        weights=shares,
        misfit=first.misfit,
        noise_sse=first.noise_sse,
        inflation=first.inflation,
        cut_short=tuple(
            sorted({name
                    for c in components for name in c.cut_short})))


def join_beliefs(first: ParameterBelief,
                 second: ParameterBelief) -> ParameterBelief:
    """The product of two beliefs over disjoint parameters, draw ``i`` of each
    paired into joint draw ``i``.

    ``second`` must carry at least ``first``'s number of draws.
    """
    overlap = sorted(set(first.names) & set(second.names))
    if overlap:
        raise ValueError(f"Joined beliefs share parameters {overlap}.")
    num = first.num_draws
    if second.num_draws < num:
        raise ValueError(f"Joining {num} draws needs as many from each "
                         f"factor; got {second.num_draws}.")
    if second.components:
        raise ValueError("Only the first joined belief may be a mixture.")
    return ParameterBelief(
        names=list(first.names) + list(second.names),
        scales=list(first.scales) + list(second.scales),
        map_estimate={
            **first.map_estimate,
            **second.map_estimate
        },
        lines={
            **first.lines,
            **second.lines
        },
        noise_scale=first.noise_scale,
        draws=np.hstack([first.draws, second.draws[:num]]),
        held=list(first.held) + list(second.held),
        evaluations=first.evaluations + second.evaluations,
        discrete={
            **first.discrete,
            **second.discrete
        },
        components=[join_beliefs(c, second) for c in first.components],
        weights=list(first.weights),
        misfit=first.misfit,
        noise_sse=first.noise_sse,
        inflation=first.inflation,
        cut_short=tuple(first.cut_short) + tuple(second.cut_short),
        untraced_basins=first.untraced_basins)


def redrawn(belief: ParameterBelief, num_draws: int,
            rng: np.random.Generator) -> ParameterBelief:
    """``belief`` with ``num_draws`` fresh joint draws in place of its own
    (:meth:`ParameterBelief.sample`)."""
    rows = belief.sample(rng, num_draws)
    draws = np.array([[row[name] for name in belief.names] for row in rows],
                     dtype=float).reshape(-1, len(belief.names))
    return replace(belief, draws=draws)


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
    batch_residuals: Optional[BatchResidualsFn] = None,
    misfit: float = 0.0,
    noise_sse: float = 0.0,
    free_names: Collection[str] = (),
    deadline: Optional[float] = None,
) -> ParameterBelief:
    """Build ``q(theta)`` around ``map_params`` and draw from it.

    ``residuals`` is the fit's objective (scaled residuals over the
    segments it pooled) and ``noise_sigma`` the Gaussian width it scores
    them with; ``prior_centers`` / ``prior_sigmas`` (fit space) are the
    Gaussian prior it folded in. ``batch_residuals``, when given, scores
    a list of parameter dicts together, each as ``residuals`` would; the
    lines are then traced in lockstep (:func:`_trace_lines`), with the
    same result.

    The temperature is the estimated noise level, at least 1: ``SSE_min
    / (N noise_sigma^2)`` spreads the misfit over the ``N`` residuals as
    independent errors. ``misfit`` (SSE units), when given, is the fit's
    error above the declared noise counted as ONE error that persists
    through the recordings: a replay that drifts off its recording, or a
    contact that resolves differently, moves many residuals together,
    and a parameter change can trade that error for a lower SSE. SSE
    differences up to the misfit are then no evidence, so the
    temperature is at least ``misfit / noise_sigma^2``. ``noise_sse``
    (SSE units) is the declared noise's expected SSE the misfit is
    measured above; the description states the misfit as a multiple of
    it. ``free_names`` are parameters the fit held at ``map_params``
    rather than estimated: a line of theirs that the recordings leave
    flat stops after its outward probes (``_line_points``).

    ``deadline`` (``time.monotonic()``) is when the fit call must have
    returned: tracing stops before a round of replays that would end
    after it, judged by the longest round so far. A free parameter whose
    line was cut short keeps its prior; a declared one keeps the part of
    its line that was traced when that part brackets the point estimate,
    else its prior. The belief names them (``cut_short``).
    """
    assert noise_sigma > 0.0, noise_sigma
    at_map = np.asarray(residuals(dict(map_params)), dtype=float)
    sse_min = float(np.dot(at_map, at_map))
    noise_scale = 1.0
    if at_map.size:
        noise_scale = max(1.0, sse_min / (at_map.size * noise_sigma**2))
    noise_scale = max(noise_scale, misfit / noise_sigma**2)
    # Negative log likelihood per unit of summed squared residual, at the
    # estimated noise level.
    k = 1.0 / (2.0 * noise_scale * noise_sigma**2)
    evaluations = 0
    lines: Dict[str, LinePosterior] = {}
    discrete: Dict[str, DiscretePosterior] = {}
    held: List[str] = []
    traced: List[Tuple[ParamSpec, _LineEvaluator, float, float]] = []
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
        traced.append((spec, line, float(lo_arr[0]), float(hi_arr[0])))
    free = set(free_names)
    if batch_residuals is not None:
        unfinished = _trace_lines([(line, lo, hi, spec.name in free)
                                   for spec, line, lo, hi in traced], config,
                                  batch_residuals, deadline)
    else:
        unfinished = set()
        for index, (spec, line, lo, hi) in enumerate(traced):
            if not _trace_line(line, lo, hi, config, spec.name in free,
                               deadline):
                unfinished.add(index)
    cut_short: List[str] = []
    for index, (spec, line, lo, hi) in enumerate(traced):
        evaluations += line.count
        grid = line.points()
        if index in unfinished:
            cut_short.append(spec.name)
            brackets = grid[0] < line.z0 < grid[-1]
            if spec.name in free or not brackets:
                lines[spec.name] = _prior_line(line, lo, hi)
                continue
        lines[spec.name] = LinePosterior.from_neg_log(grid,
                                                      [line(z) for z in grid])
    if cut_short:
        logger.warning(
            "Parameter belief: the fit's deadline cut short the lines of "
            "%s.", cut_short)

    belief = replace(_with_draws(specs, map_params, lines, held, noise_scale,
                                 evaluations, config, seed, discrete),
                     misfit=max(misfit, 0.0),
                     noise_sse=max(noise_sse, 0.0),
                     cut_short=tuple(cut_short))
    logger.info(
        "Parameter belief: %d draws over %d parameters from %d line "
        "evaluations (temperature %.3g).", belief.num_draws, len(specs),
        evaluations, noise_scale)
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

    def params_at(self, z: float) -> Dict[str, float]:
        """The MAP with only this parameter moved to fit-space ``z``."""
        params = dict(self.map_params)
        params[self.spec.name] = scalar_from_fit_space(self.spec, float(z))
        return params

    def record(self, z: float, residuals: np.ndarray) -> None:
        """Store the objective's residuals at ``z``, however they were
        scored."""
        res = np.asarray(residuals, dtype=float)
        self.sse[float(z)] = float(np.dot(res, res))
        self.count += 1

    def __call__(self, z: float) -> float:
        z = float(z)
        if z not in self.sse:
            self.record(z, self.residuals(self.params_at(z)))
        return ((self.sse[z] - self.sse_min) * self.k + self._prior(z) -
                self._prior(self.z0))

    def points(self) -> List[float]:
        """The evaluated grid, sorted."""
        return sorted(self.sse)


def _line_points(line: _LineEvaluator,
                 lo: float,
                 hi: float,
                 config: BeliefConfig,
                 flat_stop: bool = False) -> Iterator[float]:
    """The points :func:`_trace_line` evaluates, as a generator.

    It yields each point the line has not scored yet and resumes once the
    point is scored (``line.record``), so driving it alone
    (:func:`_trace_line`) or together with other lines
    (:func:`_trace_lines`) evaluates the same points.

    Each side is probed outward from the MAP, first shrinking the step
    until a probe falls inside the cutoff, then doubling it until the
    cutoff or the declared bound is reached. The remaining budget
    bisects the grid interval holding the most mass. With
    ``flat_stop`` the line ends after the outward probes when none of
    them moves the likelihood by more than ``_FLAT_LIKELIHOOD``.
    """

    def value(z: float) -> Generator[float, None, float]:
        if float(z) not in line.sse:
            yield float(z)
        return line(z)

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

        def probe(
                step: float,
                edge: float = edge,
                side: float = side
        ) -> Generator[float, None, Tuple[float, float]]:
            z = z0 + side * step
            if (z - edge) * side > 0.0:
                z = edge
            g = yield from value(z)
            return z, g

        step = h0
        z, g = yield from probe(step)
        shrinks = 0
        while g > cutoff and shrinks < _MAX_SHRINKS and line.count < budget:
            step /= 4.0
            z, g = yield from probe(step)
            shrinks += 1
        while g <= cutoff and z != edge and line.count < budget:
            step *= 2.0
            z, g = yield from probe(step)
    if flat_stop and all((sse - line.sse_min) * line.k <= _FLAT_LIKELIHOOD
                         for sse in line.sse.values()):
        return
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
        yield from value(float(mid))


def _prior_line(line: _LineEvaluator, lo: float, hi: float) -> LinePosterior:
    """A parameter's prior alone, on its bounds (fit space): the Gaussian the
    fit folded in, within six of its widths, as :func:`prior_belief` builds it;
    uniform on finite bounds without one; the point estimate without either."""
    if line.prior is None:
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            return LinePosterior.from_neg_log(np.linspace(lo, hi, 129),
                                              np.zeros(129))
        return LinePosterior(np.array([line.z0]), np.ones(1))
    center, sigma = line.prior
    low = max(lo, center - 6.0 * sigma)
    high = min(hi, center + 6.0 * sigma)
    if not high > low:
        return LinePosterior(np.array([min(max(center, lo), hi)]), np.ones(1))
    grid = np.linspace(low, high, 129)
    return LinePosterior.from_neg_log(grid, 0.5 * ((grid - center) / sigma)**2)


def _trace_line(line: _LineEvaluator,
                lo: float,
                hi: float,
                config: BeliefConfig,
                flat_stop: bool = False,
                deadline: Optional[float] = None) -> bool:
    """Evaluate a parameter's line until its posterior mass is resolved
    (:func:`_line_points`), one point at a time; False when ``deadline``
    stopped it first (before a point that, by the slowest point so far, would
    end after it)."""
    slowest = 0.0
    for z in _line_points(line, lo, hi, config, flat_stop):
        if deadline is not None and time.monotonic() + slowest > deadline:
            return False
        start = time.monotonic()
        line(z)
        slowest = max(slowest, time.monotonic() - start)
    return True


def _trace_lines(traced: Sequence[Tuple[_LineEvaluator, float, float, bool]],
                 config: BeliefConfig,
                 batch_residuals: BatchResidualsFn,
                 deadline: Optional[float] = None) -> Set[int]:
    """Trace several lines in lockstep: each round scores every unfinished
    line's next point in one ``batch_residuals`` call.

    The lines are independent (each moves one parameter off the MAP), so
    each evaluates exactly the points it would alone, while a round
    costs one batch where tracing line by line costs a call per point.
    Returns the indices of the lines ``deadline`` stopped before they
    finished: no round starts that, by the slowest round so far, would
    end after it.
    """
    generators = [
        _line_points(line, lo, hi, config, flat_stop)
        for line, lo, hi, flat_stop in traced
    ]
    pending: Dict[int, float] = {}
    for i, generator in enumerate(generators):
        z = next(generator, None)
        if z is not None:
            pending[i] = z
    slowest = 0.0
    while pending:
        if deadline is not None and time.monotonic() + slowest > deadline:
            return set(pending)
        order = list(pending)
        start = time.monotonic()
        results = batch_residuals(
            [traced[i][0].params_at(pending[i]) for i in order])
        slowest = max(slowest, time.monotonic() - start)
        for i, residuals in zip(order, results):
            traced[i][0].record(pending.pop(i), residuals)
            z = next(generators[i], None)
            if z is not None:
                pending[i] = z
    return set()
