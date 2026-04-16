"""Utilities for the code sim-learning module."""

from __future__ import annotations

import logging
from typing import Callable, Dict

from predicators.structs import Object, State

logger = logging.getLogger(__name__)

# Type alias: {Object: {feature_name: new_value}}
ProcessUpdate = Dict[Object, Dict[str, float]]


class LearnedSimulator:
    """Wraps a step-level simulator function (handwritten or LLM-synthesized).

    The function predicts process dynamics — features like water_volume,
    heat_level, spilled_level that aren't captured by rigid body
    physics.
    """

    StepFn = Callable[[State], ProcessUpdate]

    def __init__(self,
                 step_fn: StepFn,
                 name: str = "learned_simulator") -> None:
        self._step_fn = step_fn
        self.name = name

    def predict_step(self, state: State) -> ProcessUpdate:
        """Predict process feature updates for a single timestep."""
        try:
            return self._step_fn(state)
        except Exception as e:  # pylint: disable=broad-except
            logger.warning("Simulator '%s' step raised: %s", self.name, e)
            return {}
