"""Nonlinear tempering retains the conditional target and default behavior."""
from dataclasses import replace

import numpy as np
import pytest

from predicators.code_sim_learning.inference_data import InferenceIdentity, \
    content_digest
from predicators.code_sim_learning.inference_sampling import BoxPrior, \
    ConditionedPrior, PriorPoint, SamplerConfig, sample_batch


def test_nonlinear_schedule_against_gaussian_reference() -> None:
    """A narrow informed marginal and an uninformed prior both survive."""
    box = BoxPrior(("theta", "unused"), ((-1., 1.), ) * 2)
    digest = content_digest(b"nonlinear temperature reference")
    prior = ConditionedPrior(box.names, digest, digest, box)
    identity = InferenceIdentity(digest, digest, digest, prior.digest, digest)
    config = SamplerConfig(particles=1200,
                           temperatures=32,
                           moves=8,
                           max_evaluations=320000,
                           proposal_scale=.025,
                           proposal_blocks=((0, ), (1, )),
                           temperature_schedule=tuple(
                               (stage / 32)**3 for stage in range(1, 33)))
    result = sample_batch(prior,
                          identity,
                          lambda x: -.5 * ((x[0] - .27) / .025)**2,
                          config,
                          45,
                          condition=lambda x: PriorPoint(tuple(x), .8 * x[0]))
    assert result.status == "complete"
    samples = np.asarray(result.samples)
    mean = np.average(samples, axis=0, weights=result.weights)
    variance = np.average((samples - mean)**2, axis=0, weights=result.weights)
    # Exponentially tilting a Gaussian shifts its mean by tilt * variance.
    # The [-1, 1] truncation is over 29 standard deviations from this mean.
    assert mean[0] == pytest.approx(.27 + .8 * .025**2, abs=.003)
    assert variance[0] == pytest.approx(.025**2, rel=.15)
    assert mean[1] == pytest.approx(0., abs=.08)
    assert variance[1] == pytest.approx(1 / 3, abs=.05)


def test_schedule_validation_and_default_parity() -> None:
    """Explicit linear stages exactly reproduce the original default."""
    for schedule in ((.5, ), (0., 1.), (.4, .9), (.8, .7), (1., 1.),
                     (float("nan"), 1.), (float("inf"), 1.), (True, 1.)):
        with pytest.raises(ValueError, match="Temperature schedule"):
            SamplerConfig(temperatures=2, temperature_schedule=schedule)
    box = BoxPrior(("theta", ), ((-1., 1.), ))
    digest = content_digest(b"default schedule parity")
    identity = InferenceIdentity(digest, digest, digest, box.digest, digest)
    config = SamplerConfig(particles=40, temperatures=4, moves=3)
    explicit = replace(config, temperature_schedule=(.25, .5, .75, 1.))
    first = sample_batch(box, identity, lambda x: -20 * x[0]**2, config, 12)
    second = sample_batch(box, identity, lambda x: -20 * x[0]**2, explicit, 12)
    assert first == replace(second, config=config)
