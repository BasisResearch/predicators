"""Utilities for the code sim-learning module.

Core primitives for process-dynamics simulation:

* ``apply_rules`` — run a list of rule functions on a state, return
  feature updates (``ProcessUpdate``).
* ``merge_updates`` — overwrite features in a ``State`` with values
  from a ``ProcessUpdate``.
* ``simulate_step`` — full pipeline: base → rules → merge.
* ``read_simulator_components`` — pull the ``PROCESS_RULES``,
  ``PARAM_SPECS``, ``PROCESS_FEATURES`` triple out of a namespace
  (oracle module globals or agent-synthesized exec namespace).
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

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
) -> State:
    """Overwrite features in *base_state* with values from *updates*."""
    if not updates:
        return base_state

    new_data = {}
    for obj in base_state:
        arr = base_state[obj].copy()
        if obj in updates:
            for feat_name, new_val in updates[obj].items():
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
) -> State:
    """Full simulation pipeline: base → rules → merge."""
    base_state = base_env.simulate(state, action)
    updates = apply_rules(base_state, rules, params)
    if not updates:
        return base_state
    return merge_updates(base_state, updates)


# ── Module-namespace loader ───────────────────────────────────────


def read_simulator_components(
    ns: Mapping[str, Any],
) -> Tuple[Optional[List], Optional[List], Optional[Dict[str, List[str]]]]:
    """Pull the simulator triple from a namespace (module or exec dict).

    Looks for three names by convention:

    * ``PROCESS_RULES`` — non-empty list of rule functions.
    * ``PARAM_SPECS``   — list of ``ParamSpec``, **or** a zero-arg
      callable returning such a list. The callable form lets oracle
      modules defer CFG-dependent values until consumption time, so the
      module can be imported before CFG is finalized; the agent's
      saved-file form normally just uses a list.
    * ``PROCESS_FEATURES`` — ``{type_name: [feature_names]}`` dict.

    Returns ``(rules, specs, features)`` with ``None`` for any
    missing-or-malformed component; callers decide how to react.
    """
    rules = ns.get("PROCESS_RULES")
    if not isinstance(rules, list) or not rules:
        rules = None

    specs = ns.get("PARAM_SPECS")
    if callable(specs):
        specs = specs()
    if not isinstance(specs, list) or not specs:
        specs = None

    features = ns.get("PROCESS_FEATURES")
    if features is not None and not isinstance(features, dict):
        features = None

    return rules, specs, features


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
