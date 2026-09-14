"""Joint support rejection preserves target mass and recovery semantics."""
import math
from dataclasses import replace
from typing import List, Tuple, cast

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


def _setup() -> Tuple[ConditionedPrior, InferenceIdentity]:
    digest = content_digest(b"triangular joint support reference")
    prior = ConditionedPrior(("parameter", "state", "uninformed"),
                             digest, digest,
                             BoxPrior(("u", "v", "w"), ((0., 1.), ) * 3))
    return prior, InferenceIdentity(digest, digest, digest, prior.digest,
                                    digest)


def _target(point: Tuple[float, ...], temper_all: bool) -> TargetEvaluation:
    x, y, _ = point
    if not 0 < y < x:
        return TargetEvaluation(point, point, -math.inf, -math.inf)
    return TargetEvaluation(point, point, 0. if temper_all else math.log(x),
                            math.log(x) if temper_all else 0.)


@pytest.mark.parametrize("seed", [18, 29])
@pytest.mark.parametrize("temper_all", [False, True])
def test_parameter_dependent_support(seed: int, temper_all: bool) -> None:
    """For density x on 0<y<x<1, E[x]=3/4 and E[y]=3/8.

    Redrawing only y at fixed x would remove the support-volume factor x
    and bias these expectations. Initial joint rejection instead has
    E[x]=2/3, E[y]=1/3 before applying any finite target weights.
    """
    prior, identity = _setup()
    saved: List[SamplerCheckpoint] = []

    def evaluate(
            points: Tuple[Tuple[float, ...],
                          ...]) -> Tuple[TargetEvaluation, ...]:
        return tuple(_target(point, temper_all) for point in points)

    config = SamplerConfig(particles=1800,
                           temperatures=8,
                           moves=2,
                           max_evaluations=40000,
                           initialize_on_support=True)
    result = sample_batch(prior,
                          identity,
                          BatchedTarget(evaluate),
                          config,
                          seed,
                          checkpoint=saved.append)
    assert result.status == "complete"
    initial = saved[0].unpack()
    assert initial["initial_finite"] == config.particles
    assert config.particles < initial["evaluations"] < 3 * config.particles
    np.testing.assert_allclose(np.mean(initial["particles"], axis=0),
                               [2 / 3, 1 / 3, .5],
                               atol=.025)
    values = np.asarray(result.samples)
    assert np.all((values[:, 1] > 0) & (values[:, 1] < values[:, 0]))
    np.testing.assert_allclose(np.average(values,
                                          axis=0,
                                          weights=result.weights),
                               [.75, .375, .5],
                               atol=.025)
    assert np.average(values[:, 0]**2,
                      weights=result.weights) == pytest.approx(.6, abs=.025)
    assert np.average(values[:, 1]**2,
                      weights=result.weights) == pytest.approx(.2, abs=.025)
    assert np.average((values[:, 2] - .5)**2,
                      weights=result.weights) == pytest.approx(1 / 12, abs=.01)
    for boundary in (saved[0], saved[3]):
        assert sample_batch(prior,
                            identity,
                            BatchedTarget(evaluate),
                            config,
                            seed,
                            resume=boundary) == result


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("possible", [False, True])
def test_initialization_budget_is_not_a_posterior(batched: bool,
                                                  possible: bool) -> None:
    """Count every rejection, without emitting partial initialization."""
    prior, identity = _setup()
    saved: List[SamplerCheckpoint] = []
    calls = []

    def target(point: Tuple[float, ...]) -> TargetEvaluation:
        calls.append(point)
        if not possible:
            return TargetEvaluation(point, point, -math.inf, -math.inf)
        return _target(point, True)

    def batch(
            points: Tuple[Tuple[float, ...],
                          ...]) -> Tuple[TargetEvaluation, ...]:
        return tuple(target(point) for point in points)

    def condition(point: np.ndarray) -> PriorPoint:
        row = target(tuple(point))
        return PriorPoint(row.joint, row.log_base)

    config = SamplerConfig(particles=20,
                           max_evaluations=13,
                           initialize_on_support=True)
    if batched:
        result = sample_batch(prior,
                              identity,
                              BatchedTarget(batch),
                              config,
                              4,
                              checkpoint=saved.append)
    else:
        result = sample_batch(prior,
                              identity,
                              lambda p: math.log(p[0]),
                              config,
                              4,
                              condition=condition,
                              checkpoint=saved.append)
    assert result.status == "budget_exhausted"
    assert result.evaluations == len(calls) == 13
    assert not saved and not result.samples and not result.weights
    assert result.completed_temperature == 0
    expected = sum(
        math.isfinite(_target(point, True).log_base)
        for point in calls) if possible else 0
    assert result.initial_finite == expected


def test_support_checkpoint_contract() -> None:
    """Initialization mode is part of checkpoint identity, with hard
    support."""
    prior, identity = _setup()
    config = SamplerConfig(particles=16,
                           temperatures=2,
                           moves=1,
                           initialize_on_support=True)
    saved: List[SamplerCheckpoint] = []

    def evaluate(
            points: Tuple[Tuple[float, ...],
                          ...]) -> Tuple[TargetEvaluation, ...]:
        return tuple(_target(point, True) for point in points)

    target = BatchedTarget(evaluate)
    sample_batch(prior, identity, target, config, 9, checkpoint=saved.append)
    with pytest.raises(ValueError, match="Checkpoint differs"):
        sample_batch(prior,
                     identity,
                     target,
                     replace(config, initialize_on_support=False),
                     9,
                     resume=saved[0])
    with pytest.raises(ValueError, match="must be boolean"):
        replace(config, initialize_on_support=cast(bool, 1))
