"""Offline likelihood checks against the real observation injector."""
import dataclasses
import math

import numpy as np
import pytest
from scipy.stats import norm

from predicators.code_sim_learning.inference_data import EpisodeData, \
    InferenceData, InferenceIdentity, Observation, SensorFeature, \
    SensorModel, content_digest
from predicators.observation_noise import ObservationNoise, step_rng
from predicators.structs import Object, State, Type


def _state() -> State:
    obj = Object(
        "box",
        Type("box", ["x", "yaw", "pressure", "attached"],
             sensor_features=["pressure"]))
    robot = Object("arm", Type("robot", ["x"]))
    return State({
        obj: np.array([0.2, math.pi, 1.0, 1.0]),
        robot: np.array([0.0])
    })


def test_injector_likelihood_and_exact_constraints() -> None:
    """Raw angles/readings match scipy's density; exact errors have no
    floor."""
    truth = _state()
    noise = ObservationNoise(position=0.03, orientation=0.2, scalar=0.1)
    sensor = SensorModel.from_state(truth, noise)
    predicted = dict(Observation.from_state(0, truth).values)
    standardized = []
    for step in range(200):
        measured = Observation.from_state(
            step, noise.perturb(truth, step_rng(5, 0, 0, step)))
        values = dict(measured.values)
        z = [(values[f.key] - predicted[f.key]) / f.sigma
             for f in sensor.features if f.sigma]
        standardized.extend(z)
        expected = sum(
            norm.logpdf(values[f.key], loc=predicted[f.key], scale=f.sigma)
            for f in sensor.features if f.sigma)
        assert sensor.log_likelihood(measured, predicted) == \
            pytest.approx(expected, abs=1e-12)
    assert abs(np.mean(standardized)) < 0.12
    assert abs(np.std(standardized) - 1) < 0.12
    measured = Observation.from_state(0, truth)
    shifted = dict(predicted)
    shifted[("box", "box", "yaw")] -= 2 * math.pi
    assert sensor.log_likelihood(measured, shifted) < -450
    for key in (("box", "box", "attached"), ("arm", "robot", "x")):
        wrong = dict(predicted)
        wrong[key] = np.nextafter(wrong[key], math.inf)
        assert sensor.log_likelihood(measured, wrong) == -math.inf


def test_missing_conditioned_and_invalid_predictions() -> None:
    """Only explicitly conditioned fields can bypass an exact constraint."""
    truth = _state()
    key = ("arm", "robot", "x")
    sensor = SensorModel.from_state(truth, ObservationNoise(position=.1),
                                    [key])
    measured = Observation.from_state(0, truth)
    prediction = dict(measured.values)
    prediction.pop(key)
    assert math.isfinite(sensor.log_likelihood(measured, prediction))
    sparse = Observation(0, ((("box", "box", "x"), .25), ))
    assert sensor.log_likelihood(sparse, prediction) == pytest.approx(
        norm.logpdf(.25, .2, .1))
    prediction.pop(("box", "box", "x"))
    with pytest.raises(ValueError, match="Missing or nonfinite"):
        sensor.log_likelihood(sparse, prediction)
    with pytest.raises(ValueError, match="Unknown observed"):
        sensor.log_likelihood(Observation(0, ((("new", "box", "x"), 0), )), {})
    with pytest.raises(ValueError, match="Only exact"):
        SensorModel.from_state(truth, ObservationNoise(position=.1),
                               [("box", "box", "x")])
    with pytest.raises(ValueError, match="Unknown conditioned"):
        SensorModel.from_state(truth, ObservationNoise(), [("none", "x", "x")])
    with pytest.raises(ValueError, match="declared"):
        SensorModel.from_state(truth, ObservationNoise(declared=False))
    with pytest.raises(ValueError, match="Invalid declared"):
        SensorModel.from_state(truth, ObservationNoise(position=-1))
    with pytest.raises(ValueError, match="finite"):
        Observation(0, ((("box", "box", "x"), math.nan), ))
    with pytest.raises(ValueError, match="Duplicate sensor"):
        SensorModel((SensorFeature(key, 0), SensorFeature(key, 0)))


def test_ledger_identity_and_observation_ownership() -> None:
    """Duplicate observe calls cannot tighten confidence or mutate evidence."""
    state = _state()
    obs = Observation.from_state(0, state)
    ledger = InferenceData((EpisodeData("run/reset0", (), (obs, obs)), ))
    original = ledger.digest
    once = InferenceData((EpisodeData("run/reset0", (), (obs, )), ))
    assert once == ledger
    state[next(iter(state))][0] += 2
    assert ledger.digest == original
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(obs, "step", 1)
    with pytest.raises(ValueError, match="Conflicting"):
        EpisodeData("run/reset0", (), (obs, Observation.from_state(0, state)))
    with pytest.raises(ValueError, match="exceeds"):
        EpisodeData("run/reset0", (), (dataclasses.replace(obs, step=1), ))
    with pytest.raises(ValueError, match="Duplicate reset"):
        InferenceData(ledger.episodes * 2)
    changed = InferenceData((EpisodeData("run/reset0", ((0.1, ), ),
                                         (obs, )), ))
    assert changed.digest != original
    masked = dataclasses.replace(obs, values=obs.values[:-1])
    assert InferenceData((EpisodeData("run/reset0", (), (masked,)),)).digest \
        != original
    identity = InferenceIdentity(original, content_digest(b"sensor"),
                                 content_digest(b"source and parameters"),
                                 content_digest(b"prior"),
                                 content_digest(b"runtime and layout"))
    for field in dataclasses.fields(identity):
        replacement = dataclasses.replace(
            identity, **{field.name: content_digest(b"different")})
        assert replacement.digest != identity.digest
    with pytest.raises(ValueError, match="SHA256"):
        dataclasses.replace(identity, program="unversioned")


def test_batch_alignment_and_cached_reads() -> None:
    """A second read has no likelihood factor; only another step can add
    one."""
    key = ("b", "box", "x")
    sensor = SensorModel((SensorFeature(key, .1), ))
    observation = Observation(0, ((key, .2), ))
    once = InferenceData((EpisodeData("reset0", (), (observation, )), ))
    repeated = InferenceData((EpisodeData("reset0", (),
                                          (observation, observation)), ))
    predicted = {"reset0": [{key: .15}]}
    assert once.log_likelihood(sensor, predicted) == \
        repeated.log_likelihood(sensor, predicted)
    stepped = InferenceData((EpisodeData(
        "reset0", ((0., ), ),
        (observation, dataclasses.replace(observation, step=1))), ))
    assert stepped.log_likelihood(sensor, {"reset0": predicted["reset0"] * 2}) \
        == pytest.approx(2 * once.log_likelihood(sensor, predicted))
    with pytest.raises(ValueError, match="length"):
        once.log_likelihood(sensor, {"reset0": predicted["reset0"] * 2})
    with pytest.raises(ValueError, match="episodes"):
        once.log_likelihood(sensor, {"reset1": predicted["reset0"]})
