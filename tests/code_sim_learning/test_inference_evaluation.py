"""Batched targets preserve conditional densities, budgets and
replayability."""
import math
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from typing import List, Tuple

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


def _target(proposal: Tuple[float, ...]) -> TargetEvaluation:
    """Uniform x via x=u², then a truncated likelihood proportional to x."""
    u = proposal[0]
    x = u**2
    return TargetEvaluation(proposal, (x, 1 - x),
                            math.log(2 * u) if u >= .2 else -math.inf,
                            math.log(x) if u >= .4 else -math.inf)


def _mapped(
        proposals: Tuple[Tuple[float, ...],
                         ...]) -> Tuple[TargetEvaluation, ...]:
    return tuple(_target(proposal) for proposal in proposals)


def _setup() -> Tuple[ConditionedPrior, InferenceIdentity, SamplerConfig]:
    digest = content_digest(b"batched conditional density reference")
    prior = ConditionedPrior(("x", "complement"), digest, digest,
                             BoxPrior(("u", ), ((0., 1.), )))
    identity = InferenceIdentity(digest, digest, digest, prior.digest, digest)
    return prior, identity, SamplerConfig(particles=48,
                                          temperatures=5,
                                          moves=3,
                                          proposal_scale=.3,
                                          proposal_blocks=((0, ), ))


def test_process_scheduling_and_resume() -> None:
    """Process isolation and checkpoint recovery give exactly the same run."""
    prior, identity, config = _setup()
    saved: List[SamplerCheckpoint] = []
    expected = sample_batch(prior,
                            identity,
                            BatchedTarget(_mapped),
                            config,
                            41,
                            checkpoint=saved.append)
    assert expected.status == "complete"
    assert expected.accepted_moves > 0
    assert expected.evaluations < config.particles * (
        1 + config.moves * config.temperatures)
    with ProcessPoolExecutor(
            max_workers=2,
            mp_context=multiprocessing.get_context("spawn")) as executor:

        def parallel(
            proposals: Tuple[Tuple[float, ...], ...]
        ) -> Tuple[TargetEvaluation, ...]:
            return tuple(executor.map(_target, proposals))

        actual = sample_batch(prior, identity, BatchedTarget(parallel), config,
                              41)
        assert actual == expected
        for checkpoint in (saved[0], saved[2], saved[-1]):
            assert sample_batch(prior,
                                identity,
                                BatchedTarget(parallel),
                                config,
                                41,
                                resume=checkpoint) == expected

    with pytest.raises(ValueError, match="Checkpoint differs"):
        sample_batch(prior,
                     identity,
                     lambda _: 0.,
                     config,
                     41,
                     condition=lambda x: PriorPoint((x[0], 1 - x[0]), 0.),
                     resume=saved[0])
    with pytest.raises(ValueError, match="coordinate map"):
        sample_batch(prior,
                     identity,
                     BatchedTarget(_mapped),
                     config,
                     41,
                     condition=lambda x: PriorPoint(tuple(x), 0.))


@pytest.mark.parametrize("seed", [7, 19])
def test_conditional_reference_moments(seed: int) -> None:
    """Against analytic truncated Beta(2,1), including the coordinate
    factor."""
    prior, identity, config = _setup()
    config = replace(config,
                     particles=1200,
                     temperatures=12,
                     moves=4,
                     proposal_scale=.15)
    result = sample_batch(prior, identity, BatchedTarget(_mapped), config,
                          seed)
    assert result.status == "complete"
    x = np.asarray(result.samples)[:, 0]
    a = .4**2
    expected = 2 / 3 * (1 - a**3) / (1 - a**2)
    second = .5 * (1 - a**4) / (1 - a**2)
    # Zero-weight rows remain in the population when resampling is skipped.
    assert np.min(x[np.asarray(result.weights) > 0]) >= a
    assert np.max(x) < 1
    assert np.average(x, weights=result.weights) == pytest.approx(expected,
                                                                  abs=.018)
    assert np.average(x**2, weights=result.weights) == pytest.approx(second,
                                                                     abs=.018)
    np.testing.assert_allclose(np.sum(result.samples, axis=1), 1.)


@pytest.mark.parametrize("budget", [3, 51, 150])
def test_budget_reservation_and_interrupted_batches(budget: int) -> None:
    """Never dispatch extra work or publish an unfinished temperature."""
    prior, identity, config = _setup()
    config = replace(config, max_evaluations=budget)
    calls = 0
    saved: List[SamplerCheckpoint] = []

    def counted(
        proposals: Tuple[Tuple[float, ...],
                         ...]) -> Tuple[TargetEvaluation, ...]:
        nonlocal calls
        calls += len(proposals)
        assert calls <= budget
        return _mapped(proposals)

    result = sample_batch(prior,
                          identity,
                          BatchedTarget(counted),
                          config,
                          41,
                          checkpoint=saved.append)
    assert result.status == "budget_exhausted"
    assert result.evaluations == calls == budget
    assert not result.samples and not result.weights
    if budget < config.particles:
        assert not saved
    else:
        checkpoint = saved[-1]
        calls = checkpoint.unpack()["evaluations"]
        assert sample_batch(prior,
                            identity,
                            BatchedTarget(counted),
                            config,
                            41,
                            resume=checkpoint) == result


def test_worker_failure_keeps_last_complete_stage() -> None:
    """A batch exception is an infrastructure error, not zero likelihood."""
    prior, identity, config = _setup()
    saved: List[SamplerCheckpoint] = []
    calls = 0

    def interrupted(
        proposals: Tuple[Tuple[float, ...],
                         ...]) -> Tuple[TargetEvaluation, ...]:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("worker interrupted")
        return _mapped(proposals)

    with pytest.raises(RuntimeError, match="worker interrupted"):
        sample_batch(prior,
                     identity,
                     BatchedTarget(interrupted),
                     config,
                     41,
                     checkpoint=saved.append)
    assert len(saved) == 1
    resumed = sample_batch(prior,
                           identity,
                           BatchedTarget(_mapped),
                           config,
                           41,
                           resume=saved[-1])
    assert resumed == sample_batch(prior, identity, BatchedTarget(_mapped),
                                   config, 41)


def test_bad_batch_results_are_rejected() -> None:
    """A reordered or incomplete map cannot silently corrupt particle
    weights."""
    proposals = ((.3, ), (.7, ))
    cases = (
        (lambda p: _mapped(p)[:-1], "number of rows"),
        (lambda p: _mapped(p)[::-1], "identity/order"),
        (lambda p: tuple(replace(row, joint=(.1, ))
                         for row in _mapped(p)), "joint dimension"),
    )
    for callback, message in cases:
        with pytest.raises(ValueError, match=message):
            BatchedTarget(callback).evaluate(proposals, 2, True)
    with pytest.raises(ValueError, match="Box target"):
        BatchedTarget(_mapped).evaluate(proposals, 2, False)
    for name in ("log_base", "log_likelihood"):
        for invalid in (math.nan, math.inf):
            with pytest.raises(ValueError, match="NaN or positive infinity"):
                if name == "log_base":
                    replace(_target((.7, )), log_base=invalid)
                else:
                    replace(_target((.7, )), log_likelihood=invalid)
    with pytest.raises(ValueError, match="zero target support"):
        replace(_target((.7, )), log_base=-math.inf)
    with pytest.raises(ValueError, match="finite"):
        replace(_target((.7, )), joint=(math.nan, 0.))


def test_box_target_and_zero_support() -> None:
    """Unconditioned batches retain the box prior and failed-result
    semantics."""
    _, identity, config = _setup()
    prior = BoxPrior(("x", ), ((0., 1.), ))
    identity = replace(identity, prior=prior.digest)
    config = replace(config, particles=1200)

    def constant(
        proposals: Tuple[Tuple[float, ...],
                         ...]) -> Tuple[TargetEvaluation, ...]:
        return tuple(TargetEvaluation(p, p, 0., 0.) for p in proposals)

    result = sample_batch(prior, identity, BatchedTarget(constant), config, 7)
    assert result.status == "complete"
    x = np.asarray(result.samples)[:, 0]
    assert np.average(x, weights=result.weights) == pytest.approx(.5, abs=.025)
    assert np.average(x**2, weights=result.weights) == pytest.approx(1 / 3,
                                                                     abs=.025)

    def unsupported(
        proposals: Tuple[Tuple[float, ...],
                         ...]) -> Tuple[TargetEvaluation, ...]:
        return tuple(TargetEvaluation(p, p, 0., -math.inf) for p in proposals)

    saved: List[SamplerCheckpoint] = []
    failed = sample_batch(prior,
                          identity,
                          BatchedTarget(unsupported),
                          config,
                          7,
                          checkpoint=saved.append)
    assert failed.status == "no_particle_support"
    assert failed.evaluations == config.particles
    assert not failed.samples and not saved
