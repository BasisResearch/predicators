"""Legacy comparison frames preserve only the supplied public observations."""
import numpy as np
import pytest

from predicators.code_sim_learning.inference_data import Observation
from predicators.code_sim_learning.inference_recording import \
    RecordingProjection
from predicators.structs import Object, Type
from predicators.utils import PyBulletState


def test_public_frame_roundtrip_and_ownership() -> None:
    """Reversed input handles and double-digit joint indices retain values."""
    kind = Type("body", ["y", "x"])
    a, b = Object("a", kind), Object("b", kind)
    source = PyBulletState({
        b: np.array([.2, .1]),
        a: np.array([.4, .3])
    },
                           simulator_state={
                               "joint_positions": list(range(12)),
                               "base_pose": ((1., 2., 3.), (0., 0., 0., 1.))
                           })
    projection = RecordingProjection()
    observation = projection.observe(9, source)
    result = projection.to_state(observation, [b, a])
    assert list(result.data) == [a, b]
    assert isinstance(result.simulator_state, dict)
    assert result.simulator_state["joint_positions"] == list(range(12))
    assert result.privileged is None and result.latent is None
    assert projection.observe(9, result) == observation
    result.data[a][0] = 123.
    assert source.data[a][0] == .4
    assert projection.observe(9, source) == observation


def test_incomplete_or_unknown_frames_are_rejected() -> None:
    """A comparison must not silently supply missing state or drop evidence."""
    obj = Object("body", Type("body", ["x"]))
    projection = RecordingProjection()
    feature = ((obj.name, obj.type.name, "x"), .1)
    joint = (("__proprioception__", "joint_positions", "0"), .2)
    observation = Observation(0, (feature, joint))
    with pytest.raises(ValueError, match="Duplicate"):
        projection.to_state(observation, [obj, obj])
    with pytest.raises(ValueError, match="Missing measured"):
        projection.to_state(Observation(0, (joint, )), [obj])
    with pytest.raises(ValueError, match="Missing public joint"):
        projection.to_state(Observation(0, (feature, )), [obj])
    for extra, message in (((("other", "body", "x"), .3), "Unknown measured"),
                           ((("__proprioception__", "velocity", "0"), .3),
                            "Unknown measured"), ((("__proprioception__",
                                                    "joint_positions", "02"),
                                                   .3), "Noncanonical"),
                           ((("__proprioception__", "joint_positions", "2"),
                             .3), "Discontinuous"), ((("__proprioception__",
                                                       "base_position", "0"),
                                                      .3), "Incomplete")):
        with pytest.raises(ValueError, match=message):
            projection.to_state(Observation(0, (feature, joint, extra)), [obj])
