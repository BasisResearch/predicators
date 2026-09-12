"""Full-output composition retains all measurements and exact constraints."""
import math

import pytest

from predicators.code_sim_learning.inference_conditioning import \
    UnsupportedConditioning
from predicators.code_sim_learning.inference_data import Observation, \
    SensorFeature, SensorModel
from predicators.code_sim_learning.inference_observation import \
    CheckedReadoutFactor, EulerOutputFactor, OutputObservationModel, \
    ScalarOutputFactor
from predicators.code_sim_learning.inference_orientation import \
    QuaternionOutputError
from predicators.code_sim_learning.inference_output_error import \
    GaussianOutputError
from predicators.code_sim_learning.inference_readout import ExactReadout

JOINT = ("robot", "joint", "q")
DISPLAY = ("robot", "robot", "fingers")
EVENT = ("object", "body", "attached")
ANGLES = (("robot", "robot", "roll"), ("robot", "robot", "pitch"),
          ("robot", "robot", "yaw"))


def test_readout_preserves_source_and_unmodeled_event() -> None:
    """A finite source-error density cannot conceal another exact failure."""
    sensor = SensorModel(
        tuple(SensorFeature(k, 0.) for k in (JOINT, DISPLAY, EVENT)))
    declaration = ExactReadout(JOINT, DISPLAY, sensor.digest, "a" * 64)
    model = OutputObservationModel(
        sensor,
        scalars=(ScalarOutputFactor(JOINT, GaussianOutputError(.8, .1, .2)), ),
        readouts=(CheckedReadoutFactor(declaration, lambda x: 2 * x + 1), ))
    predictions = tuple(
        Observation(i, ((JOINT, 0.), (EVENT, 0.))) for i in range(3))
    observations = tuple(
        Observation(i, ((JOINT, .1), (DISPLAY, 1.2), (EVENT, 0.)))
        for i in range(3))
    score = model.log_likelihood(predictions, observations)
    assert math.isfinite(score)
    assert model.log_likelihood(predictions, observations) == score
    changed = list(observations)
    changed[2] = Observation(2, ((JOINT, .1), (DISPLAY, 1.2), (EVENT, 1.)))
    assert model.log_likelihood(predictions, tuple(changed)) == -math.inf
    changed[2] = Observation(2, ((JOINT, .1), (DISPLAY, 1.21), (EVENT, 0.)))
    assert model.log_likelihood(predictions, tuple(changed)) == -math.inf
    # A contradictory readout must not hide a malformed observation schema.
    changed[2] = Observation(2, changed[2].values + ((ANGLES[0], .1), ))
    with pytest.raises(ValueError, match="Unknown observed"):
        model.log_likelihood(predictions, tuple(changed))
    # Removing the initial reading changes evidence; it is never implicit.
    without_initial = (Observation(0, ()), ) + observations[1:]
    assert model.log_likelihood(predictions, without_initial) != score
    without_error = OutputObservationModel(sensor, readouts=model.readouts)
    assert without_error.log_likelihood(predictions, observations) == -math.inf
    assert without_error.digest != model.digest


def test_euler_factor_covers_all_three_fields_once() -> None:
    """Pole observations retain coupled mass, while other fields still
    score."""
    sensor = SensorModel(tuple(SensorFeature(k, 0.) for k in (*ANGLES, EVENT)))
    process = QuaternionOutputError(.1)
    model = OutputObservationModel(sensor,
                                   eulers=(EulerOutputFactor(ANGLES,
                                                             process), ))
    prediction = Observation(
        0,
        tuple(zip(ANGLES, (0., math.pi / 2, 0.))) + ((EVENT, 0.), ))
    observed = Observation(
        0,
        tuple(zip(ANGLES, (0., math.pi / 2, 4.7))) + ((EVENT, 0.), ))
    assert math.isfinite(model.log_likelihood((prediction, ), (observed, )))
    partial = Observation(0, ((ANGLES[0], 0.), ))
    with pytest.raises(UnsupportedConditioning, match="Partial Euler"):
        model.log_likelihood((prediction, ), (partial, ))
    assert model.log_likelihood((prediction, ), (Observation(0, ()), )) == 0.
    with pytest.raises(ValueError, match="multiple factors"):
        OutputObservationModel(sensor,
                               scalars=(ScalarOutputFactor(
                                   ANGLES[0], GaussianOutputError(.8, .1)), ),
                               eulers=model.eulers)
    noisy = SensorModel(tuple(SensorFeature(k, .1) for k in ANGLES))
    with pytest.raises(UnsupportedConditioning, match="exact readings"):
        OutputObservationModel(noisy, eulers=model.eulers)


def test_sensor_fallback_and_step_contract() -> None:
    """Unassigned evidence uses the declared sensor law with no hidden mask."""
    sensor = SensorModel((SensorFeature(JOINT, .1), SensorFeature(EVENT, 0.)))
    model = OutputObservationModel(sensor)
    predictions = (Observation(0, ((JOINT, .2), (EVENT, 1.))), )
    observations = (Observation(0, ((JOINT, .3), (EVENT, 1.))), )
    expected = sensor.log_likelihood(observations[0],
                                     dict(predictions[0].values))
    assert model.log_likelihood(predictions, observations) == expected
    with pytest.raises(ValueError, match="each step"):
        model.log_likelihood((Observation(1, predictions[0].values), ),
                             observations)
    with pytest.raises(ValueError, match="Matching nonempty"):
        model.log_likelihood((), ())
    with pytest.raises(ValueError, match="Unknown observed"):
        model.log_likelihood(predictions, (Observation(0,
                                                       ((DISPLAY, 0.), )), ))
    with pytest.raises(ValueError, match="Missing or nonfinite"):
        model.log_likelihood((Observation(0, ()), ), observations)


def test_conditioned_inputs_and_readout_dependencies_are_explicit() -> None:
    """Neither an external input nor a derived display gets another density."""
    sensor = SensorModel((SensorFeature(JOINT, 0., conditioned=True),
                          SensorFeature(DISPLAY, 0.), SensorFeature(EVENT,
                                                                    0.)))
    with pytest.raises(ValueError, match="predicted sensor"):
        OutputObservationModel(sensor,
                               scalars=(ScalarOutputFactor(
                                   JOINT, GaussianOutputError(.8, .1)), ))
    first = CheckedReadoutFactor(
        ExactReadout(JOINT, DISPLAY, sensor.digest, "a" * 64), lambda x: x)
    second = CheckedReadoutFactor(
        ExactReadout(DISPLAY, EVENT, sensor.digest, "b" * 64), lambda x: x)
    with pytest.raises(UnsupportedConditioning, match="Chained"):
        OutputObservationModel(sensor, readouts=(first, second))
    model = OutputObservationModel(sensor, readouts=(first, ))
    data = (Observation(0, ((JOINT, .1), (DISPLAY, .1))), )
    assert model.log_likelihood((Observation(0, ()), ), data) == 0.
