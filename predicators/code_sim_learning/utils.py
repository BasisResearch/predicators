"""Utilities for the code sim-learning module.

Core primitives for process-dynamics simulation:

* ``apply_rules`` — run a list of rule functions on a state, return
  feature updates (``ProcessUpdate``).
* ``merge_updates`` — overwrite process features in a ``State`` with
  values from a ``ProcessUpdate``.
* ``simulate_step`` — full pipeline: kinematics → rules → merge.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List

from predicators.structs import Action, Object, State

logger = logging.getLogger(__name__)

# Type alias: {Object: {feature_name: new_value}}
ProcessUpdate = Dict[Object, Dict[str, float]]

# ── Primitives ────────────────────────────────────────────────────


def apply_rules(state: State, rules: List,
                params: Dict[str, float]) -> ProcessUpdate:
    """Apply process rules sequentially and return feature updates.

    Each rule has signature ``rule(state, updates, params) -> updates``.
    Values are normalised to plain floats (rules may return numpy
    scalars).
    """
    updates: ProcessUpdate = {}
    for rule in rules:
        updates = rule(state, updates, params)
    return {
        obj: {feat: float(val)
              for feat, val in feat_dict.items()}
        for obj, feat_dict in updates.items()
    }


def merge_updates(
    base_state: State,
    updates: ProcessUpdate,
    process_features: Dict[str, List[str]],
) -> State:
    """Overwrite process features in *base_state* with *updates*.

    Only features listed in ``process_features[type_name]`` are
    overwritten; all other features are preserved from *base_state*.
    """
    if not updates:
        return base_state

    new_data = {}
    for obj in base_state:
        arr = base_state[obj].copy()
        type_name = obj.type.name
        process_feats = set(process_features.get(type_name, []))

        if obj in updates:
            for feat_name, new_val in updates[obj].items():
                if feat_name in process_feats:
                    idx = obj.type.feature_names.index(feat_name)
                    arr[idx] = new_val

        new_data[obj] = arr

    merged = base_state.copy()
    merged.data = new_data
    return merged


def simulate_step(
    state: State,
    action: Action,
    base_env: Any,
    rules: List,
    params: Dict[str, float],
    process_features: Dict[str, List[str]],
) -> State:
    """Full simulation pipeline: kinematics → rules → merge.

    Runs ``base_env.simulate`` for kinematics, ``apply_rules`` for
    process dynamics, and ``merge_updates`` to combine them.
    """
    kin_state = base_env.simulate(state, action)
    updates = apply_rules(kin_state, rules, params)
    if not updates:
        return kin_state
    return merge_updates(kin_state, updates, process_features)


# ── LearnedSimulator ──────────────────────────────────────────────


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
