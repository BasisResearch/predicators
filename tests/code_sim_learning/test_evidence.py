"""Tests for the Laplace evidence of a rollout fit
(``code_sim_learning_fit_evidence``, docs/continual-uncertainty.md 3.4)."""

# pylint: disable=protected-access

import numpy as np
import pytest

from predicators.approaches.agent_sim_learning_approach import \
    AgentSimLearningApproach
from predicators.code_sim_learning.evidence import LaplaceEvidence, \
    format_evidence_lines, laplace_log_evidence
from predicators.code_sim_learning.fit_space import FitResult, ParamSpec


def _linear_fit(jac_column, z_map, centre, prior_width, nu):
    """A one-parameter linear residual model at its posterior MAP.

    ``r(z) = J (z - z_map) + r_hat`` with ``r_hat`` chosen so the
    posterior gradient vanishes at ``z_map`` (``J^T r_hat / nu^2 +
    (z_map - centre) / width^2 = 0``); the Laplace approximation is
    exact for it.
    """
    jac = np.asarray(jac_column, dtype=float)[:, None]
    r_hat = -(nu**2 / prior_width**2) * (z_map - centre) * jac[:, 0] / float(
        jac[:, 0] @ jac[:, 0])
    result = FitResult(names=["gain"],
                       samples=np.array([[z_map]]),
                       log_probs=np.zeros(1),
                       jacobian=jac,
                       noise_sigma=nu,
                       prior_sigma=np.array([prior_width]))

    def sse_at(z):
        r = jac[:, 0] * (z - z_map) + r_hat
        return float(r @ r)

    return result, sse_at


def test_laplace_matches_quadrature_for_a_linear_model():
    """The formula equals the exact marginal likelihood of a linear model."""
    nu, width, centre, z_map = 0.05, 0.75, 0.0, 0.4
    result, sse_at = _linear_fit(np.linspace(0.5, 2.0, 12), z_map, centre,
                                 width, nu)
    spec = ParamSpec("gain", 1.0, lo=-10.0, hi=10.0)
    evidence = laplace_log_evidence(result, sse_at(z_map), {"gain": centre},
                                    [spec])
    assert evidence is not None
    assert evidence.num_residuals == 12
    assert evidence.num_params == 1
    # log int L(z) p(z) dz by quadrature on a fine grid.
    grid = np.linspace(z_map - 6.0, z_map + 6.0, 200001)
    step = grid[1] - grid[0]
    log_lik = np.array([-sse_at(z) / (2 * nu**2)
                        for z in grid]) - 12 * np.log(nu * np.sqrt(2 * np.pi))
    log_prior = (-0.5 * ((grid - centre) / width)**2 -
                 np.log(width * np.sqrt(2 * np.pi)))
    integrand = log_lik + log_prior
    peak = integrand.max()
    log_z = peak + np.log(np.sum(np.exp(integrand - peak)) * step)
    assert evidence.log_evidence == pytest.approx(log_z, abs=1e-3)
    assert evidence.summary().startswith("log evidence ")
    assert LaplaceEvidence.from_dict(evidence.as_dict()) == evidence


def test_occam_term_penalises_a_parameter_the_data_constrain_for_nothing():
    """A data-flat extra parameter leaves the evidence unchanged; an extra
    parameter the data constrain without lowering the SSE lowers it."""
    nu, width = 0.05, 0.75
    base, sse_at = _linear_fit(np.linspace(0.5, 2.0, 12), 0.4, 0.0, width, nu)
    specs = [
        ParamSpec("gain", 1.0, lo=-10.0, hi=10.0),
        ParamSpec("extra", 0.0, lo=-10.0, hi=10.0)
    ]
    one = laplace_log_evidence(base, sse_at(0.4), {"gain": 0.0}, specs[:1])
    assert one is not None
    flat_jac = np.hstack([base.jacobian, np.zeros((12, 1))])
    flat = FitResult(names=["gain", "extra"],
                     samples=np.array([[0.4, 0.0]]),
                     log_probs=np.zeros(1),
                     jacobian=flat_jac,
                     noise_sigma=nu,
                     prior_sigma=np.array([width, width]))
    two_flat = laplace_log_evidence(flat, sse_at(0.4), {
        "gain": 0.0,
        "extra": 0.0
    }, specs)
    assert two_flat is not None
    assert two_flat.log_evidence == pytest.approx(one.log_evidence, abs=1e-9)
    informative = FitResult(names=["gain", "extra"],
                            samples=np.array([[0.4, 0.0]]),
                            log_probs=np.zeros(1),
                            jacobian=np.hstack(
                                [base.jacobian,
                                 np.full((12, 1), 3.0)]),
                            noise_sigma=nu,
                            prior_sigma=np.array([width, width]))
    two = laplace_log_evidence(informative, sse_at(0.4), {
        "gain": 0.0,
        "extra": 0.0
    }, specs)
    assert two is not None
    assert two.log_evidence < one.log_evidence
    assert two.log_occam < one.log_occam


def test_no_jacobian_means_no_evidence():
    """Without a Jacobian (LM skipped, ablation pinned) there is nothing to
    approximate with."""
    result = FitResult(names=["gain"],
                       samples=np.array([[0.4]]),
                       log_probs=np.zeros(1),
                       jacobian=None,
                       noise_sigma=0.05,
                       prior_sigma=np.array([0.75]))
    spec = ParamSpec("gain", 1.0, lo=-10.0, hi=10.0)
    assert laplace_log_evidence(result, 1.0, {"gain": 0.0}, [spec]) is None
    # A name the specs do not carry, or a nan SSE, likewise.
    with_jac = FitResult(names=["gain"],
                         samples=np.array([[0.4]]),
                         log_probs=np.zeros(1),
                         jacobian=np.ones((3, 1)),
                         noise_sigma=0.05,
                         prior_sigma=np.array([0.75]))
    assert laplace_log_evidence(with_jac, 1.0, {}, []) is None
    assert laplace_log_evidence(with_jac, float("nan"), {}, [spec]) is None


def _evidence(log_evidence, num_residuals, num_params=1):
    return LaplaceEvidence(log_evidence=log_evidence,
                           num_residuals=num_residuals,
                           num_params=num_params,
                           log_likelihood=log_evidence,
                           log_prior=0.0,
                           log_occam=0.0)


def test_report_lines_quote_the_delta_only_on_the_same_residual_set():
    """The delta names the winner; a different residual set says so; no
    Jacobian says why."""
    now = _evidence(-100.0, 500, 2)
    same = format_evidence_lines(now, {"vers_002": _evidence(-108.5, 500)})
    assert "-100.0 on 500 residuals with 2 parameter(s)" in same[0]
    assert "Against vers_002: -108.5, delta +8.5 (this version wins" in same[0]
    worse = format_evidence_lines(now, {"vers_002": _evidence(-90.0, 500)})
    assert "delta -10.0 (the previous version wins" in worse[0]
    other = format_evidence_lines(now, {"vers_002": _evidence(-90.0, 640)})
    assert "not comparable" in other[0]
    first = format_evidence_lines(now, None)
    assert "Against" not in first[0]
    assert "has to win on evidence, not on residual" in first[1]
    missing = format_evidence_lines(None, None)
    assert len(missing) == 1 and "unavailable" in missing[0]


def test_approach_keeps_the_history_per_version():
    """The previous version's record is the newest one with another tag."""
    approach = object.__new__(AgentSimLearningApproach)
    approach._fit_evidence_history = {}
    assert approach.previous_fit_evidence("vers_001") is None
    approach.note_fit_evidence("vers_001", _evidence(-120.0, 500))
    assert approach.previous_fit_evidence("vers_001") is None
    approach.note_fit_evidence("vers_002", _evidence(-110.0, 500))
    previous = approach.previous_fit_evidence("vers_003")
    assert previous is not None
    (tag, record), = previous.items()
    assert tag == "vers_002" and record.log_evidence == -110.0
    # Re-fitting the same version compares against the one before it.
    (tag2, _), = approach.previous_fit_evidence("vers_002").items()
    assert tag2 == "vers_001"
