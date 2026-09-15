"""Independent quadrature checks for observation-guided box proposals."""
import math

import numpy as np
import pytest
from scipy.integrate import quad

from predicators.code_sim_learning.inference_box_guidance import \
    GaussianBoxProposal
from predicators.code_sim_learning.inference_sampling import BoxPrior


def test_joint_mixture_density_and_normalization() -> None:
    """The joint density uses one mixture indicator for the whole vector."""
    prior = BoxPrior(("x", "y"), ((-1., 2.), (.1, .8)))
    guide = GaussianBoxProposal(prior, (.2, .9), (.3, .4), .2)
    point = (.1, .5)
    normal = 1.
    for value, center, scale, (lo, hi) in zip(point, guide.centers,
                                              guide.scales, prior.bounds):
        mass = .5 * (math.erf(
            (hi - center) / (scale * math.sqrt(2))) - math.erf(
                (lo - center) / (scale * math.sqrt(2))))
        normal *= math.exp(-.5 * ((value - center) / scale)**2) / (
            scale * math.sqrt(2 * math.pi) * mass)
    expected = .2 / (3 * .7) + .8 * normal
    assert math.exp(guide.log_density(point)) == pytest.approx(expected)
    integral = quad(
        lambda x: quad(lambda y: math.exp(guide.log_density(
            (x, y))), .1, .8)[0], -1., 2.)[0]
    assert integral == pytest.approx(1., abs=1e-10)
    assert guide.log_density((-1.1, .5)) == -math.inf


@pytest.mark.parametrize("center", [-.8, .5, 2.5])
def test_corrected_proposal_recovers_original_prior_and_posterior(
        center: float) -> None:
    """Integrate both proposal branches and recover fixed-prior evidence.

    The likelihood deliberately resembles the guide. Replacing the prior
    with that guide or omitting the correction changes these integrals.
    """
    prior = BoxPrior(("location", ), ((-1., 2.), ))
    guide = GaussianBoxProposal(prior, (center, ), (.4, ), .2)

    def expectation(power: int, with_likelihood: bool) -> float:

        def integrand(u: float, branch: float) -> float:
            point = guide.transform((u, branch))
            x = point.joint[0]
            likelihood = math.exp(-.5 * ((x - .3) / .6)**2) \
                if with_likelihood else 1.
            return math.exp(point.log_weight) * x**power * likelihood

        return .2 * quad(
            lambda u: integrand(u, 0.), 0., 1., epsabs=1e-9)[0] + .8 * quad(
                lambda u: integrand(u, 1.), 0., 1., epsabs=1e-9)[0]

    assert expectation(0, False) == pytest.approx(1., abs=2e-8)
    assert expectation(1, False) == pytest.approx(.5, abs=2e-8)
    assert expectation(2, False) == pytest.approx(1., abs=2e-8)
    for power in (0, 1, 2):
        expected = quad(
            lambda x, exponent=power: x**exponent * math.exp(-.5 * (
                (x - .3) / .6)**2) / 3.,
            -1.,
            2.)[0]
        assert expectation(power, True) == pytest.approx(expected, abs=2e-8)


def test_endpoints_determinism_support_and_identity() -> None:
    """The defensive branch covers the original support with bounded weight."""
    prior = BoxPrior(("x", "y"), ((-.3, 1.1), (2., 5.)))
    guide = GaussianBoxProposal(prior, (.3, 4.), (.01, .02), .1)
    for branch in (0., 1.):
        assert guide.transform((0., 1., branch)).joint == (-.3, 5.)
    rng = np.random.default_rng(821)
    for _ in range(50):
        unit = tuple(float(v) for v in rng.uniform(size=3))
        point = guide.transform(unit)
        assert point == guide.transform(unit)
        assert point.log_weight <= -math.log(.1) + 1e-12
    other = GaussianBoxProposal(prior, (.4, 3.), (.1, .2), .3)
    assert other.prior.digest == guide.prior.digest
    assert other.digest != guide.digest
    assert GaussianBoxProposal(prior, (.3, 4.), (.01, .02), .1) == guide


def test_invalid_guides_and_coordinates() -> None:
    """Invalid input cannot masquerade as an ordinary rejected candidate."""
    prior = BoxPrior(("x", ), ((0., 1.), ))
    for centers, scales, mass in [((0., ), (0., ), .1),
                                  ((math.inf, ), (1., ), .1),
                                  ((0., ), (1., ), 0.), ((0., ), (1., ), 1.),
                                  ((0., ), (1., ), math.nan),
                                  ((0., 1.), (1., ), .1),
                                  ((1e308, ), (1e-308, ), .1)]:
        with pytest.raises(ValueError):
            GaussianBoxProposal(prior, centers, scales, mass)
    guide = GaussianBoxProposal(prior, (.5, ), (.1, ))
    for unit in [(), (.5, ), (.5, math.nan), (-.1, .5), (.5, 1.1)]:
        with pytest.raises(ValueError):
            guide.transform(unit)
    for point in [(), (math.inf, ), (.1, .2)]:
        with pytest.raises(ValueError):
            guide.log_density(point)
