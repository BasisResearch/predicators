"""Joint forecasts retain particle dependence, posterior mass and
provenance."""
import math
from dataclasses import replace

import numpy as np
import pytest

from predicators.code_sim_learning.inference_assessment import \
    AssessmentProtocol, InferenceCheck, assess_inference
from predicators.code_sim_learning.inference_conditioning import \
    UnsupportedConditioning
from predicators.code_sim_learning.inference_data import EpisodeData, \
    InferenceData, InferenceIdentity, Observation, SensorFeature, \
    SensorModel, content_digest
from predicators.code_sim_learning.inference_observation import \
    OutputObservationModel
from predicators.code_sim_learning.inference_prediction import JointForecast
from predicators.code_sim_learning.inference_sampling import BatchPosterior, \
    BoxPrior, SamplerConfig

POSITION = ("body", "body", "x")
EVENT = ("body", "body", "attached")


def _reference():
    """Specify an empirical joint measure with two anticorrelated modes."""
    model = OutputObservationModel(
        SensorModel((SensorFeature(POSITION, 1.), SensorFeature(EVENT, 0.))))
    episode = EpisodeData("episode", (),
                          (Observation(0, ((POSITION, 0.), (EVENT, 0.))), ))
    data = InferenceData((episode, ))
    prior = BoxPrior(("velocity", "episode.start"), ((-5., 5.), (-5., 5.)))
    digest = content_digest(b"specified empirical measure")
    identity = InferenceIdentity(data.digest, model.digest, digest,
                                 prior.digest, digest)
    candidate = BatchPosterior(identity, prior, SamplerConfig(particles=4), 0,
                               "complete", ((4., 4.), (1., -1.), (-1., 1.)),
                               (0., .25, .75), 3, 1., 2, (2., ), 0, 0, 2, 0)
    protocol = AssessmentProtocol(digest, ("specified", ))
    checks = (InferenceCheck("specified", "pass", "Enumerated mixture",
                             digest), )
    predictive = (InferenceCheck("future", "fail", "Model error retained",
                                 digest), )
    assessment = assess_inference(candidate, protocol, checks, predictive)
    return assessment, data, model


def _replay(row, episode, actions):
    assert row["velocity"] == -row["episode.start"]
    assert len(episode.observations) == 1
    return tuple(
        Observation(i, ((POSITION, row["episode.start"] + i * row["velocity"]),
                        (EVENT, float(row["velocity"] > 0) if i else 0.)))
        for i in range(1 + len(episode.actions) + len(actions)))


def _forecast():
    assessment, data, model = _reference()
    return JointForecast.replay(assessment, data, "episode", model,
                                ((0., ), (0., )), _replay)


def test_complete_history_mixture_matches_enumerated_reference():
    """Likelihoods mix once per whole future, never independently by time."""
    forecast = _forecast()
    # Omit the event readings to compare the Gaussian mixture analytically.
    future = (Observation(1, ((POSITION, .2), )),
              Observation(2, ((POSITION, .7), )))
    left = -.5 * (.2**2 + (.7 - 1)**2) - math.log(2 * math.pi)
    right = -.5 * (.2**2 + (.7 + 1)**2) - math.log(2 * math.pi)
    expected = math.log(.25 * math.exp(left) + .75 * math.exp(right))
    assert forecast.log_likelihood(future) == pytest.approx(expected)
    assert forecast.log_likelihood(future) != pytest.approx(.25 * left +
                                                            .75 * right)
    assert forecast.assessment.predictive_checks[0].status == "fail"
    # Each mode requires its own consistent event sequence. A per-time
    # mixture would incorrectly assign positive mass to this switch.
    switched = (Observation(1,
                            ((EVENT, 1.), )), Observation(2, ((EVENT, 0.), )))
    assert forecast.log_likelihood(switched) == -math.inf
    supported = tuple(Observation(i, ((EVENT, 1.), )) for i in (1, 2))
    assert forecast.log_likelihood(supported) == pytest.approx(math.log(.25))
    # Very small component densities must be mixed without underflow.
    remote = tuple(Observation(i, ((POSITION, 1e4), )) for i in (1, 2))
    assert math.isfinite(forecast.log_likelihood(remote))


def test_draws_keep_whole_joint_modes_and_their_probabilities():
    """One source particle controls the entire sampled history."""
    forecast = _forecast()
    draws = forecast.sample(3000, 41)
    assert draws == forecast.sample(3000, 41)
    assert draws != forecast.sample(3000, 42)
    assert set(i for i, _ in draws) == {1, 2}
    assert np.mean([i == 1 for i, _ in draws]) == pytest.approx(.25, abs=.025)
    for index, history in draws:
        assert [o.step for o in history] == [1, 2]
        assert [dict(o.values)[EVENT] for o in history] == \
            [float(index == 1)] * 2
    assert forecast.assessment.predictive_checks[0].status == "fail"
    for count, seed in ((0, 1), (True, 1), (1, -1), (1, True)):
        with pytest.raises(ValueError):
            forecast.sample(count, seed)


def test_replay_receives_owned_complete_rows_and_no_future_readings():
    """The callback gets all joint coordinates and only fitted observations."""
    assessment, data, model = _reference()
    calls = []

    def replay(row, episode, actions):
        calls.append((dict(row), episode, actions))
        result = _replay(row, episode, actions)
        row["episode.start"] = 999.
        return result

    forecast = JointForecast.replay(assessment, data, "episode", model,
                                    ((0., ), (0., )), replay)
    assert len(calls) == 2
    assert [row for row, _, _ in calls] == [{
        "velocity": 1.,
        "episode.start": -1.
    }, {
        "velocity": -1.,
        "episode.start": 1.
    }]
    assert all(episode == data.episodes[0] for _, episode, _ in calls)
    assert all(actions == ((0., ), (0., )) for _, _, actions in calls)
    assert forecast.assessment == _reference()[0]


@pytest.mark.parametrize("availability", ["unevaluated", "numerical_failure"])
def test_unavailable_assessment_cannot_start_replay(availability):
    """Numerical checks gate prediction artifacts before simulation starts."""
    assessment, data, model = _reference()
    unavailable = replace(assessment,
                          availability=availability,
                          posterior=None)
    with pytest.raises(ValueError, match=availability):
        JointForecast.replay(unavailable, data, "episode", model, (), _replay)
    forged = replace(unavailable, availability="available")
    with pytest.raises(ValueError, match="needs a posterior"):
        JointForecast.replay(forged, data, "episode", model, (), _replay)


def test_ledger_model_episode_and_action_mismatches_reject_before_replay():
    """A suffix included in fitting cannot masquerade as unseen
    observations."""
    assessment, data, model = _reference()
    changed = InferenceData((replace(data.episodes[0], actions=((0., ), )), ))
    other = OutputObservationModel(SensorModel((SensorFeature(POSITION,
                                                              2.), )))
    for ledger, output in ((changed, model), (data, other)):
        with pytest.raises(ValueError, match="identity differs"):
            JointForecast.replay(assessment, ledger, "episode", output, (),
                                 _replay)
    with pytest.raises(ValueError, match="fitted reset episode"):
        JointForecast.replay(assessment, data, "unknown", model, (), _replay)
    for actions in (((math.nan, ), ), ((0., ), (0., 1.))):
        with pytest.raises(ValueError):
            JointForecast.replay(assessment, data, "episode", model, actions,
                                 _replay)


def test_replay_failures_and_missing_particles_are_not_renormalized():
    """No partial ensemble is returned after an exception or lost particle."""
    assessment, data, model = _reference()

    def fail(row, episode, actions):
        if row["velocity"] < 0:
            raise RuntimeError("native replay failed")
        return _replay(row, episode, actions)

    with pytest.raises(RuntimeError, match="native replay failed"):
        JointForecast.replay(assessment, data, "episode", model, ((0., ), ),
                             fail)
    forecast = _forecast()
    with pytest.raises(ValueError, match="positive-weight joint row"):
        replace(forecast, histories=forecast.histories[:1])
    with pytest.raises(ValueError, match="every action"):
        replace(forecast, histories=tuple(h[:-1] for h in forecast.histories))
    bad_prefix = Observation(0, ((POSITION, 0.), (EVENT, 1.)))
    with pytest.raises(UnsupportedConditioning,
                       match="zero-likelihood prefix"):
        replace(forecast,
                histories=tuple(
                    (bad_prefix, ) + h[1:] for h in forecast.histories))


def test_empty_future_and_missing_initial_reading():
    """Empty futures retain unit probability; omitted reads stay omitted."""
    assessment, data, model = _reference()
    forecast = JointForecast.replay(assessment, data, "episode", model, (),
                                    _replay)
    assert forecast.log_likelihood(()) == pytest.approx(0.)
    assert all(history == () for _, history in forecast.sample(10, 3))
    missing = InferenceData((replace(data.episodes[0], observations=()), ))
    assert assessment.posterior is not None
    identity = replace(assessment.identity, data=missing.digest)
    assessment = replace(assessment,
                         identity=identity,
                         posterior=replace(assessment.posterior,
                                           identity=identity))

    def replay(row, episode, actions):
        assert episode.observations == ()
        del row
        return tuple(
            Observation(i, ((POSITION, 0.), (EVENT, 0.)))
            for i in range(len(actions) + 1))

    forecast = JointForecast.replay(assessment, missing, "episode", model, (),
                                    replay)
    assert forecast.log_likelihood(()) == pytest.approx(0.)
