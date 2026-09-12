"""Whole-candidate rejection preserves the declared conditional measure."""
import numpy as np
import pytest

from predicators.code_sim_learning.inference_data import content_digest
from predicators.code_sim_learning.inference_feasibility import draw_feasible

DIGEST = content_digest(b"fixed base and triangular support reference")


def test_joint_constraint_induces_the_expected_dependence():
    """Uniform points in a triangle have mean 1/3 and covariance -1/36."""
    result = draw_feasible(DIGEST,
                           DIGEST,
                           lambda rng: tuple(rng.uniform(size=2)),
                           lambda p: sum(p) < 1,
                           count=4096,
                           max_draws=12000,
                           seed=0)
    assert result.status == "complete"
    samples = np.asarray(result.samples)
    np.testing.assert_allclose(samples.mean(axis=0), 1 / 3, atol=.012)
    assert np.cov(samples.T)[0, 1] == pytest.approx(-1 / 36, abs=.004)
    assert np.all(samples.sum(axis=1) < 1)


def test_global_rejection_reweights_mixture_cases():
    """Equal base case weights become 1:4 after different feasibility rates."""
    result = draw_feasible(DIGEST,
                           DIGEST,
                           lambda rng:
                           (int(rng.integers(2)), float(rng.random())),
                           lambda p: p[1] < (.2 if p[0] == 0 else .8),
                           count=4096,
                           max_draws=12000,
                           seed=1)
    assert result.status == "complete"
    assert np.mean([p[0] for p in result.samples]) == pytest.approx(.8,
                                                                    abs=.02)


def test_exhaustion_and_errors_do_not_manufacture_feasible_samples():
    """No accepted sample is a finite-search result, not proof of
    impossibility."""
    result = draw_feasible(DIGEST,
                           DIGEST,
                           lambda rng: rng.random(),
                           lambda _: False,
                           count=2,
                           max_draws=4,
                           seed=0)
    assert result.status == "budget_exhausted"
    assert result.draws == 4 and result.accepted == 0 and not result.samples
    with pytest.raises(TypeError, match="boolean"):
        draw_feasible(DIGEST,
                      DIGEST,
                      lambda rng: rng.random(),
                      lambda _: 1.,
                      count=1,
                      max_draws=2,
                      seed=0)

    def broken(_):
        """A setup failure must propagate."""
        raise RuntimeError("scene initialization failed")

    with pytest.raises(RuntimeError, match="scene initialization"):
        draw_feasible(DIGEST,
                      DIGEST,
                      lambda rng: rng.random(),
                      broken,
                      count=1,
                      max_draws=2,
                      seed=0)
