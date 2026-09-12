"""Geometric conditioning must preserve the declared parameter prior law."""
import math
from dataclasses import replace
from typing import Callable, Tuple

import numpy as np
import pytest

from predicators.code_sim_learning.inference_conditioning import \
    AffineConditioning
from predicators.code_sim_learning.inference_data import InferenceIdentity, \
    content_digest
from predicators.code_sim_learning.inference_feasibility import \
    FeasibleConditioning
from predicators.code_sim_learning.inference_sampling import BoxPrior, \
    ConditionedPrior, PriorPoint, SamplerConfig, sample_batch

DIGEST = content_digest(b"feasible batch analytic reference")


def _reference(
    observed: float = .5
) -> Tuple[FeasibleConditioning, Callable[[np.ndarray], PriorPoint]]:
    """The state fits inside a theta-wide region; observe theta*x exactly."""
    original = BoxPrior(("theta", "x", "unused"),
                        ((1., 4.), (0., 4.), (-1., 1.)))
    chart = AffineConditioning(original, ("x", ), (observed, ), DIGEST)
    base = ConditionedPrior(original.names, original.digest, chart.digest,
                            BoxPrior(chart.free_names, chart.free_bounds))

    def condition(free: np.ndarray) -> PriorPoint:
        point = chart.lift(free, np.array([[free[0]]]), np.zeros(1))
        return PriorPoint(point.joint, point.log_base_weight)

    return FeasibleConditioning(base, DIGEST, "global_joint"), condition


def test_support_normalization_changes_the_parameter_posterior() -> None:
    """Ignoring Z(theta) changes 1/theta**2 into 1/theta, a different law."""
    global_law, condition = _reference()
    state_law = replace(global_law,
                        normalization="conditional_state",
                        normalizer_identity=DIGEST)
    grid = 1 + (np.arange(2048) + .5) * 3 / 2048
    global_weights = []
    state_weights = []
    for theta in grid:
        free = np.array([theta, .1])
        whole = global_law.lift(free, condition, lambda x: x[1] <= x[0])
        state = state_law.lift(free,
                               condition,
                               lambda x: x[1] <= x[0],
                               log_normalizer=lambda x: math.log(x[0] / 4))
        assert whole.joint == state.joint
        global_weights.append(math.exp(whole.log_weight))
        state_weights.append(math.exp(state.log_weight))
    assert np.average(grid,
                      weights=global_weights) == pytest.approx(3 / math.log(4),
                                                               abs=1e-6)
    assert np.average(grid, weights=state_weights) == pytest.approx(
        math.log(4) / .75, abs=1e-6)
    assert global_law.prior.original_prior != state_law.prior.original_prior
    new_data, _ = _reference(4.)
    assert new_data.prior.original_prior == global_law.prior.original_prior
    assert new_data.prior.digest != global_law.prior.digest


def test_exact_conditioning_and_support_reach_the_same_sampler() -> None:
    """Noisy evidence is applied once after the supported base correction."""
    law, condition = _reference(4.)
    law = replace(law,
                  normalization="conditional_state",
                  normalizer_identity=DIGEST)

    def lift(free: np.ndarray) -> PriorPoint:
        return law.lift(free,
                        condition,
                        lambda x: x[1] <= x[0],
                        log_normalizer=lambda x: math.log(x[0] / 4))

    def likelihood(joint: np.ndarray) -> float:
        return -.5 * ((joint[1] - 1.6) / .3)**2

    identity = InferenceIdentity(DIGEST, DIGEST, DIGEST, law.prior.digest,
                                 DIGEST)
    result = sample_batch(law.prior,
                          identity,
                          likelihood,
                          SamplerConfig(particles=2048,
                                        temperatures=12,
                                        moves=3,
                                        max_evaluations=100000),
                          19,
                          condition=lift)
    assert result.status == "complete"
    samples = np.asarray(result.samples)
    positive = np.asarray(result.weights) > 0
    assert np.all(samples[positive, 1] <= samples[positive, 0])
    np.testing.assert_allclose(samples[:, 0] * samples[:, 1], 4., atol=1e-15)
    grid = 2 + (np.arange(10000) + .5) / 5000
    weights = np.exp(-.5 * ((4 / grid - 1.6) / .3)**2) / grid**2
    assert np.average(samples[:, 0], weights=result.weights) == \
        pytest.approx(np.average(grid, weights=weights), abs=.04)
    assert abs(np.average(samples[:, 2], weights=result.weights)) < .06


def test_missing_normalizers_and_failed_search_are_not_posteriors() -> None:
    """Unavailable normalization raises; finite support search stays
    distinct."""
    law, condition = _reference()
    with pytest.raises(ValueError, match="normalizer identity"):
        replace(law, normalization="conditional_state")
    state_law = replace(law,
                        normalization="conditional_state",
                        normalizer_identity=DIGEST)
    with pytest.raises(ValueError, match="log normalizer"):
        state_law.lift(np.array([2., 0.]), condition, lambda _: True)
    with pytest.raises(ValueError, match="log normalizer"):
        law.lift(np.array([2., 0.]),
                 condition,
                 lambda _: True,
                 log_normalizer=lambda _: 0.)
    for bad in (-math.inf, math.inf, math.nan, .1):

        def invalid_normalizer(_: np.ndarray, value: float = bad) -> float:
            """Supply an invalid probability without a loop-variable
            closure."""
            return value

        with pytest.raises(ValueError, match="support probability"):
            state_law.lift(np.array([2., 0.]),
                           condition,
                           lambda _: True,
                           log_normalizer=invalid_normalizer)
    prior = law.prior
    identity = InferenceIdentity(DIGEST, DIGEST, DIGEST, prior.digest, DIGEST)
    result = sample_batch(
        prior,
        identity,
        lambda _: 0.,
        SamplerConfig(particles=8, temperatures=2, moves=1),
        0,
        condition=lambda x: law.lift(x, condition, lambda _: False))
    assert result.status == "no_particle_support"
    assert not result.samples


def test_callback_errors_and_mutation_do_not_change_candidates() -> None:
    """Support checks use owned arrays and propagate setup failures."""
    law, condition = _reference()
    free = np.array([2., 0.])

    def mutate(joint: np.ndarray) -> bool:
        joint[:] = -99
        return True

    point = law.lift(free, condition, mutate)
    assert point.joint == (2., .25, 0.)
    np.testing.assert_array_equal(free, [2., 0.])

    def broken(_: np.ndarray) -> bool:
        raise RuntimeError("scene failed to initialize")

    with pytest.raises(RuntimeError, match="initialize"):
        law.lift(free, condition, broken)
    with pytest.raises(TypeError, match="boolean"):
        law.lift(free, condition,
                 lambda _: 1.)  # type: ignore[arg-type,return-value]
    for invalid in (PriorPoint((math.nan, .25, 0.),
                               0.), PriorPoint((2., .25, 0.), math.inf)):

        def invalid_point(_: np.ndarray,
                          value: PriorPoint = invalid) -> PriorPoint:
            """Supply a malformed base point."""
            return value

        with pytest.raises(ValueError, match="Base map"):
            law.lift(free, invalid_point, broken)
    rejected = law.lift(free, lambda _: PriorPoint((2., .25, 0.), -math.inf),
                        broken)
    assert rejected.log_weight == -math.inf
