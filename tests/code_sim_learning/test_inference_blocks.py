"""Declared block kernels target the same joint distributions."""
import math
from dataclasses import replace
from typing import Union

import numpy as np
import pytest

from predicators.code_sim_learning.inference_data import InferenceIdentity, \
    content_digest
from predicators.code_sim_learning.inference_sampling import BoxPrior, \
    ConditionedPrior, PriorPoint, SamplerConfig, sample_batch


@pytest.mark.parametrize("conditional", [False, True])
def test_joint_block_target_against_grid(conditional: bool) -> None:
    """Correlated coordinates and uninformed marginals survive blocked
    moves."""
    box = BoxPrior(tuple(f"x{i}" for i in range(6)), ((-1., 1.), ) * 6)
    digest = content_digest(b"block reference")
    prior: Union[BoxPrior, ConditionedPrior] = ConditionedPrior(
        box.names, digest, digest, box) if conditional else box
    identity = InferenceIdentity(digest, digest, digest, prior.digest, digest)

    def likelihood(point: np.ndarray) -> float:
        return -.5 * (((point[0] + point[1] - .3) / .12)**2 +
                      ((point[0] - point[1] - .1) / .4)**2)

    def conditioning(point: np.ndarray) -> PriorPoint:
        return PriorPoint(tuple(point), .8 * float(point[0]))

    config = SamplerConfig(particles=1600,
                           temperatures=24,
                           moves=10,
                           proposal_scale=.08,
                           max_evaluations=400000,
                           proposal_blocks=((0, 1), (2, ), (3, ), (4, ),
                                            (5, )))
    result = sample_batch(prior,
                          identity,
                          likelihood,
                          config,
                          14,
                          condition=conditioning if conditional else None)
    assert result.status == "complete"
    axis = np.linspace(-1., 1., 401)
    a, b = np.meshgrid(axis, axis, indexing="ij")
    log_weight = -.5 * (((a + b - .3) / .12)**2 + ((a - b - .1) / .4)**2)
    if conditional:
        log_weight += .8 * a
    weights = np.exp(log_weight - log_weight.max())
    weights /= weights.sum()
    expected = np.array([(weights * a).sum(), (weights * b).sum()])
    samples = np.asarray(result.samples)
    mean = np.average(samples, axis=0, weights=result.weights)
    np.testing.assert_allclose(mean[:2], expected, atol=.02)
    assert np.all(abs(mean[2:]) < .1)
    variance = np.average((samples - mean)**2, axis=0, weights=result.weights)
    np.testing.assert_allclose(variance[2:], np.full(4, 1 / 3), atol=.05)
    expected_cov = (weights * (a - expected[0]) * (b - expected[1])).sum()
    covariance = np.average(
        (samples[:, 0] - mean[0]) * (samples[:, 1] - mean[1]),
        weights=result.weights)
    assert covariance == pytest.approx(expected_cov, abs=.008)


def test_block_partition_and_budget_contract() -> None:
    """Malformed partitions cannot leave coordinates permanently frozen."""
    box = BoxPrior(("x", "y"), ((0., 1.), ) * 2)
    digest = content_digest(b"block validation")
    identity = InferenceIdentity(digest, digest, digest, box.digest, digest)
    for blocks in (((0, ), (0, 1)), ((), ), ((-1, ), ), ((True, ), )):
        with pytest.raises(ValueError, match="distinct nonnegative"):
            SamplerConfig(proposal_blocks=blocks)
    for blocks in (((0, ), ), ((0, 1, 2), )):
        with pytest.raises(ValueError, match="partition all"):
            sample_batch(box, identity, lambda _: 0.,
                         SamplerConfig(proposal_blocks=blocks), 0)
    config = SamplerConfig(particles=10,
                           temperatures=2,
                           moves=2,
                           max_evaluations=10,
                           proposal_blocks=((0, ), (1, )))
    result = sample_batch(box, identity, lambda _: 0., config, 0)
    assert result.status == "budget_exhausted"
    assert not result.samples
    assert not result.weights
    assert result.evaluations == 10
    plain = replace(config, max_evaluations=100, proposal_blocks=())
    result = sample_batch(box, identity, lambda _: 0., plain, 1)
    assert result == sample_batch(box, identity, lambda _: 0.,
                                  replace(plain, proposal_blocks=()), 1)
    assert math.isclose(sum(result.weights), 1.)
