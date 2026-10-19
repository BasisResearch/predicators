"""The model arm's recorded ranges: each feature its model scores, over every
recorded state, for the rehearsal's support note."""
# pylint: disable=protected-access
from typing import List

import numpy as np

from predicators.approaches.agent_continual_approach import \
    AgentContinualApproach
from predicators.structs import Action, LowLevelTrajectory, Object, State, Type

_BLOCK = Type("block", ["x", "y"])
_A = Object("a", _BLOCK)
_B = Object("b", _BLOCK)


def _trajectory(xs: List[float]) -> LowLevelTrajectory:
    states = [
        State({
            _A: np.array([x, 0.0]),
            _B: np.array([x + 1.0, 5.0])
        }) for x in xs
    ]
    return LowLevelTrajectory(
        states, [Action(np.zeros(1, dtype=np.float32)) for _ in xs[1:]])


def test_recorded_ranges_span_every_object_of_a_scored_type() -> None:
    """The range of a scored feature covers every object of its type in every
    recording; unscored features stay out, and no recordings give none."""
    approach = object.__new__(AgentContinualApproach)
    approach._residual_env_cls = type(  # type: ignore[attr-defined]
        "Scene", (), {"RESIDUAL_FEATURES": {
            "block": ["x"]
        }})
    approach._fit_trajectories = [
        _trajectory([0.0, 0.2]),
        _trajectory([0.5, -0.1])
    ]
    assert approach._recorded_ranges() == {("block", "x"): (-0.1, 1.5)}
    approach._fit_trajectories = []
    assert approach._recorded_ranges() is None
