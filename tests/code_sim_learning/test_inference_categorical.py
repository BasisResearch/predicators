"""Exact finite-case references for prior-preserving discrete guidance."""
import math

import numpy as np
import pytest

from predicators.code_sim_learning.inference_categorical import \
    CategoricalProposal
from predicators.code_sim_learning.inference_conditioning import \
    ConditioningNumericalError


@pytest.mark.parametrize("strength", [0., 1., 10000.])
def test_guidance_preserves_prior_and_likelihood(strength: float) -> None:
    """Exact enumeration recovers prior moments and a separate posterior."""
    prior = (.1, .2, .3, .4)
    guide = CategoricalProposal(prior,
                                (0., -strength, -2 * strength, -math.inf), .25)
    likelihood = (.8, .1, .5, .2)
    evidence, moment, first, second = 0., 0., 0., 0.
    previous = 0.
    for i, probability in enumerate(guide.probabilities):
        point = guide.transform(previous + probability / 2)
        assert point.index == i
        mass = probability * math.exp(point.log_weight)
        first += mass * i
        second += mass * i**2
        evidence += mass * likelihood[i]
        moment += mass * likelihood[i] * i
        assert probability >= .25 * prior[i] - 1e-15
        previous = math.fsum(guide.probabilities[:i + 1])
    assert first == pytest.approx(sum(i * p for i, p in enumerate(prior)))
    assert second == pytest.approx(sum(i**2 * p for i, p in enumerate(prior)))
    expected = sum(p * y for p, y in zip(prior, likelihood))
    assert evidence == pytest.approx(expected)
    assert moment / evidence == pytest.approx(
        sum(i * p * y
            for i, (p, y) in enumerate(zip(prior, likelihood))) / expected)


def test_boundaries_reference_probabilities_and_identity() -> None:
    """Intervals agree with a directly normalized categorical Bayes guide."""
    prior, likelihood = (.2, .3, .5), (.8, .4, .1)
    guide = CategoricalProposal(prior, tuple(math.log(x) for x in likelihood))
    mass = sum(p * y for p, y in zip(prior, likelihood))
    expected = tuple(.25 * p + .75 * p * y / mass
                     for p, y in zip(prior, likelihood))
    assert guide.probabilities == pytest.approx(expected)
    assert guide.transform(0.).index == 0
    assert guide.transform(1.).index == 2
    for i in (0, 1):
        boundary = math.fsum(guide.probabilities[:i + 1])
        assert guide.transform(np.nextafter(boundary, 0.)).index == i
        assert guide.transform(boundary).index == i + 1
    assert guide.digest == CategoricalProposal(prior, guide.log_guide).digest
    changed = CategoricalProposal(prior, (0., 0., 0.))
    assert changed.digest != guide.digest
    assert changed.prior == guide.prior
    assert changed.probabilities == pytest.approx(prior)
    for unit in [math.nan, math.inf, -.1, 1.1]:
        with pytest.raises(ValueError):
            guide.transform(unit)


@pytest.mark.parametrize("prior,log_guide,mass", [((), (), .25),
                                                  ((1., ), (), .25),
                                                  ((0., 1.), (0., 0.), .25),
                                                  ((.2, .3), (0., 0.), .25),
                                                  ((math.nan, ), (0., ), .25),
                                                  ((1., ), (-math.inf, ), .25),
                                                  ((1., ), (math.inf, ), .25),
                                                  ((1., ), (math.nan, ), .25),
                                                  ((1., ), (0., ), 0.),
                                                  ((1., ), (0., ), 1.),
                                                  ((1., ), (0., ), math.nan)])
def test_invalid_cases(prior: tuple, log_guide: tuple, mass: float) -> None:
    """Invalid guides fail explicitly instead of discarding prior cases."""
    with pytest.raises(ValueError):
        CategoricalProposal(prior, log_guide, mass)


def test_unrepresentable_interval_is_not_silently_dropped() -> None:
    """Positive prior cases require representable sampling intervals."""
    with pytest.raises(ConditioningNumericalError):
        CategoricalProposal((1., 1e-100), (0., 0.))
