"""Global block moves recover uncertainty lost through finite-population
drift."""
import math
from dataclasses import replace
from typing import Callable, List, Tuple, Union

import numpy as np
import pytest

from predicators.code_sim_learning.inference_checkpoint import \
    SamplerCheckpoint
from predicators.code_sim_learning.inference_data import InferenceIdentity, \
    content_digest
from predicators.code_sim_learning.inference_evaluation import BatchedTarget, \
    TargetEvaluation
from predicators.code_sim_learning.inference_sampling import BoxPrior, \
    ConditionedPrior, PriorPoint, SamplerConfig, sample_batch


def _identity(prior: BoxPrior) -> InferenceIdentity:
    digest = content_digest(b"refresh numerical reference")
    return InferenceIdentity(digest, digest, digest, prior.digest, digest)


@pytest.mark.parametrize("seed", [7, 19])
@pytest.mark.parametrize("batch", [False, True])
def test_uninformed_coordinate_after_weight_collapse(seed: int,
                                                     batch: bool) -> None:
    """A sharply observed nuisance coordinate cannot identify its independent
    companion."""
    prior = BoxPrior(("observed", "unused"), ((0., 1.), (0., 1.)))
    config = SamplerConfig(particles=512,
                           temperatures=1,
                           moves=12,
                           proposal_scale=1e-6,
                           max_evaluations=6656,
                           proposal_blocks=((0, ), (1, )))

    def likelihood(value: np.ndarray) -> float:
        return float(-.5 * ((value[0] - .613) / 1e-6)**2)

    def mapped(
            points: Tuple[Tuple[float, ...],
                          ...]) -> Tuple[TargetEvaluation, ...]:
        return tuple(
            TargetEvaluation(p, p, 0., likelihood(np.asarray(p)))
            for p in points)

    target: Union[Callable[[np.ndarray], float], BatchedTarget] = \
        BatchedTarget(mapped) if batch else likelihood
    local = sample_batch(prior, _identity(prior), target, config, seed)
    refreshed = sample_batch(prior, _identity(prior), target,
                             replace(config, refresh_probability=1.), seed)
    assert local.status == refreshed.status == "complete"
    assert local.surviving_ancestors == 1
    local_unused = np.asarray(local.samples)[:, 1]
    assert np.var(local_unused) < 1e-8
    unused = np.asarray(refreshed.samples)[:, 1]
    assert np.average(unused,
                      weights=refreshed.weights) == pytest.approx(.5, abs=.04)
    assert np.average((unused - .5)**2,
                      weights=refreshed.weights) == pytest.approx(1 / 12,
                                                                  abs=.015)
    # Recovery of this independent coordinate does not certify the sharply
    # concentrated observed-coordinate approximation.


@pytest.mark.parametrize("refresh", [.4, 1.])
@pytest.mark.parametrize("blocked", [False, True])
def test_conditional_measure_and_checkpoint(refresh: float,
                                            blocked: bool) -> None:
    """Refreshes retain nonlinear coordinate factors and exact replay."""
    box = BoxPrior(("u", ), ((0., 1.), ))
    digest = content_digest(b"conditional refresh")
    prior = ConditionedPrior(("x", ), digest, digest, box)
    identity = replace(_identity(box), prior=prior.digest)
    config = SamplerConfig(particles=1200,
                           temperatures=4,
                           moves=4,
                           refresh_probability=refresh,
                           proposal_blocks=((0, ), ) if blocked else ())

    def condition(value: np.ndarray) -> PriorPoint:
        return PriorPoint((value[0]**2, ), math.log(2 * value[0]))

    def likelihood(value: np.ndarray) -> float:
        return math.log(value[0]) if value[0] >= .2 else -math.inf

    saved: List[SamplerCheckpoint] = []
    result = sample_batch(prior,
                          identity,
                          likelihood,
                          config,
                          9,
                          condition=condition,
                          checkpoint=saved.append)
    assert result.status == "complete"
    x = np.asarray(result.samples)[:, 0]
    expected = 2 / 3 * (1 - .2**3) / (1 - .2**2)
    assert np.average(x, weights=result.weights) == pytest.approx(expected,
                                                                  abs=.018)
    assert np.min(x[np.asarray(result.weights) > 0]) >= .2
    assert sample_batch(prior,
                        identity,
                        likelihood,
                        config,
                        9,
                        condition=condition,
                        resume=saved[2]) == result
    with pytest.raises(ValueError, match="Checkpoint differs"):
        sample_batch(prior,
                     identity,
                     likelihood,
                     replace(config, refresh_probability=0.),
                     9,
                     condition=condition,
                     resume=saved[2])


def test_refresh_invalid_probability_and_budget() -> None:
    """A global move is charged normally and cannot publish a partial fit."""
    for invalid in (-.01, 1.01, math.nan, math.inf):
        with pytest.raises(ValueError, match="Refresh probability"):
            SamplerConfig(refresh_probability=invalid)
    prior = BoxPrior(("x", ), ((0., 1.), ))
    config = SamplerConfig(particles=16,
                           max_evaluations=17,
                           refresh_probability=1.)
    result = sample_batch(prior, _identity(prior), lambda _: 0., config, 0)
    assert result.status == "budget_exhausted"
    assert result.evaluations == 17
    assert not result.samples and not result.weights
