"""Laplace evidence of a rollout system-ID fit (docs/continual-uncertainty.md,
section 3.4).

The model-comparison level of a Model Discovery Agent weights model
structures by marginal likelihood, and integrating over the parameters
is what penalises a model whose extra parameters only fit noise. The
cheap version needs nothing the fit does not already compute: the MAP
and the Jacobian at the MAP give a Laplace approximation to the
evidence, so every fit can report its log evidence next to its SSE, and
a simulator version that adds parameters has to win on evidence, not on
residual.

With residuals ``r(z)`` in fit space scored under a Gaussian of width
``nu`` (the fit's ``noise_sigma``) and a Gaussian prior ``N(c, S^2)``
per parameter, the log evidence is approximated at the MAP ``z*`` by

    log L(z*) + log p(z*) + (d / 2) log 2 pi - (1 / 2) log det A

with ``A = J^T J / nu^2 + diag(1 / S^2)`` the Gauss-Newton curvature of
the negative log posterior. The approximation is exact for a linear
residual model, which is what :mod:`tests.code_sim_learning.test_evidence`
checks against quadrature.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from predicators.code_sim_learning.fit_space import FitResult, ParamSpec, \
    to_fit_space


@dataclass(frozen=True)
class LaplaceEvidence:
    """One fit's log evidence and the terms it is made of."""

    log_evidence: float
    num_residuals: int
    num_params: int
    log_likelihood: float
    log_prior: float
    # (d / 2) log 2 pi - (1 / 2) log det A: the volume of parameter
    # space the data leave, the Occam term.
    log_occam: float

    def comparable_with(self, other: "LaplaceEvidence") -> bool:
        """Whether the two evidences score the same residual set (same count);
        a different scope or survivor set makes the delta meaningless."""
        return self.num_residuals == other.num_residuals

    def summary(self) -> str:
        """One line for logs."""
        return (f"log evidence {self.log_evidence:.1f} on "
                f"{self.num_residuals} residuals with {self.num_params} "
                f"parameter(s) (log likelihood {self.log_likelihood:.1f}, "
                f"log prior {self.log_prior:.1f}, Occam {self.log_occam:.1f})")

    def as_dict(self) -> Dict[str, float]:
        """A plain record for checkpoints and histories."""
        return {
            "log_evidence": float(self.log_evidence),
            "num_residuals": float(self.num_residuals),
            "num_params": float(self.num_params),
            "log_likelihood": float(self.log_likelihood),
            "log_prior": float(self.log_prior),
            "log_occam": float(self.log_occam),
        }

    @classmethod
    def from_dict(cls, record: Dict[str, float]) -> "LaplaceEvidence":
        """Inverse of :meth:`as_dict`."""
        return cls(log_evidence=float(record["log_evidence"]),
                   num_residuals=int(record["num_residuals"]),
                   num_params=int(record["num_params"]),
                   log_likelihood=float(record["log_likelihood"]),
                   log_prior=float(record["log_prior"]),
                   log_occam=float(record["log_occam"]))


def laplace_log_evidence(
        result: FitResult, sse: float, prior_centers: Dict[str, float],
        param_specs: Sequence[ParamSpec]) -> Optional[LaplaceEvidence]:
    """The Laplace evidence of ``result`` scored at ``sse``, or None when the
    fit carries no Jacobian (Levenberg-Marquardt did not run, or the anchor
    ablation pinned a parameter and dropped it).

    ``sse`` is the fit's SSE at the MAP over the residuals the Jacobian
    rows score; ``prior_centers`` are the prior centres in external
    units (the anchors, or the declared inits), converted to fit space
    with ``param_specs`` like the prior widths the result carries.
    """
    jac = result.jacobian
    if (jac is None or jac.size == 0 or result.noise_sigma is None
            or result.prior_sigma is None or not np.isfinite(sse)):
        return None
    jac = np.asarray(jac, dtype=float)
    if jac.ndim != 2 or jac.shape[1] != len(result.names):
        return None
    nu = float(result.noise_sigma)
    widths = np.asarray(result.prior_sigma, dtype=float)
    specs = {s.name: s for s in param_specs}
    if any(n not in specs for n in result.names):
        return None
    ordered = [specs[n] for n in result.names]
    z_map = to_fit_space(ordered,
                         [result.point_estimate[n] for n in result.names])
    z_centre = to_fit_space(
        ordered, [prior_centers.get(s.name, s.init_value) for s in ordered])
    num_residuals, num_params = jac.shape
    log_likelihood = (-sse / (2.0 * nu**2) -
                      num_residuals * np.log(nu * np.sqrt(2.0 * np.pi)))
    log_prior = float(-0.5 * np.sum(((z_map - z_centre) / widths)**2) -
                      np.sum(np.log(widths * np.sqrt(2.0 * np.pi))))
    curvature = jac.T @ jac / nu**2 + np.diag(1.0 / widths**2)
    sign, logdet = np.linalg.slogdet(curvature)
    if sign <= 0 or not np.isfinite(logdet):
        return None
    log_occam = float(0.5 * num_params * np.log(2.0 * np.pi) - 0.5 * logdet)
    return LaplaceEvidence(log_evidence=float(log_likelihood + log_prior +
                                              log_occam),
                           num_residuals=int(num_residuals),
                           num_params=int(num_params),
                           log_likelihood=float(log_likelihood),
                           log_prior=log_prior,
                           log_occam=log_occam)


def format_evidence_lines(
        evidence: Optional[LaplaceEvidence],
        previous: Optional[Dict[str, LaplaceEvidence]]) -> List[str]:
    """The fit report's evidence lines.

    ``previous`` maps the previous simulator version's tag to its
    evidence (one entry, or None when this is the first canonical fit).
    The delta is quoted only when both score the same residual set.
    """
    if evidence is None:
        return [
            "Log evidence (Laplace at the MAP): unavailable for this fit "
            "(no Jacobian at the MAP: Levenberg-Marquardt did not run, or "
            "the anchor ablation pinned a parameter)."
        ]
    line = (f"Log evidence (Laplace at the MAP): {evidence.log_evidence:.1f} "
            f"on {evidence.num_residuals} residuals with "
            f"{evidence.num_params} parameter(s); Occam term "
            f"{evidence.log_occam:.1f}")
    if previous:
        (tag, prev), = previous.items()
        if evidence.comparable_with(prev):
            delta = evidence.log_evidence - prev.log_evidence
            winner = ("this version wins" if delta > 0 else
                      "the previous version wins" if delta < 0 else "a tie")
            line += (f". Against {tag}: {prev.log_evidence:.1f}, delta "
                     f"{delta:+.1f} ({winner} on evidence)")
        else:
            line += (f". {tag} scored {prev.log_evidence:.1f} on "
                     f"{prev.num_residuals} residuals: a different residual "
                     "set (scope or survivors), so the two are not "
                     "comparable")
    line += "."
    return [
        line,
        "A version that adds parameters has to win on evidence, not on "
        "residual: the evidence integrates over the parameters, so a "
        "parameter that only fits noise lowers it even as the SSE falls.",
    ]
