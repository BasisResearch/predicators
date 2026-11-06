"""How fast the recordings show each body move under each switch pattern.

A body can go faster in a rollout than any recording shows it going
under the same switches, even when it stays inside the recorded range
of positions: a model's speed terms are then tested only by the
recordings that had other switches on.
"""
import math
from typing import Dict, FrozenSet, Iterator, List, Sequence, Set, Tuple

from predicators.structs import Object, State

# Speeds are read over this many steps: an observed position carries noise
# of the same size as one step's motion.
SPEED_WINDOW = 10
POSITION_FEATURES = ("x", "y", "z")

SwitchFeatures = Set[Tuple[str, str]]
Pattern = FrozenSet[str]
RecordedSpeeds = Dict[Tuple[str, Pattern], float]


def scored_bodies(features: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """The position features of each scored type that has any."""
    bodies = {
        type_name: [f for f in POSITION_FEATURES if f in scored]
        for type_name, scored in features.items() if type_name != "robot"
    }
    return {type_name: axes for type_name, axes in bodies.items() if axes}


def switch_features(trajectories: Sequence[Sequence[State]]) -> SwitchFeatures:
    """The (type name, feature) pairs that act as switches: every recorded
    value is 0 or 1, and some object shows both values."""
    seen: Dict[Tuple[str, str], Set[float]] = {}
    both: SwitchFeatures = set()
    held: Dict[Tuple[Object, str], float] = {}
    for states in trajectories:
        for state in states:
            for obj in state:
                if obj.type.name == "robot":
                    continue
                for feature in obj.type.feature_names:
                    key = (obj.type.name, feature)
                    value = float(state.get(obj, feature))
                    seen.setdefault(key, set()).add(value)
                    if held.setdefault((obj, feature), value) != value:
                        both.add(key)
    return {key for key in both if seen[key] <= {0.0, 1.0}}


def switch_pattern(state: State, switches: SwitchFeatures) -> Pattern:
    """The ``object.feature`` switches that are on in ``state``."""
    return frozenset(f"{obj.name}.{feature}" for obj in state
                     for type_name, feature in switches
                     if type_name == obj.type.name
                     and float(state.get(obj, feature)) > 0.5)


def window_speeds(
    states: Sequence[State],
    bodies: Dict[str, List[str]],
    switches: SwitchFeatures,
) -> Iterator[Tuple[Object, Pattern, float]]:
    """Each body's speed per step over each window whose two ends share a
    switch pattern; ``bodies`` maps a type name to its position features."""
    for first, last in zip(states, states[SPEED_WINDOW:]):
        pattern = switch_pattern(first, switches)
        if pattern != switch_pattern(last, switches):
            continue
        for obj in first:
            axes = bodies.get(obj.type.name)
            if not axes or obj not in last:
                continue
            moved = math.sqrt(
                sum((float(last.get(obj, axis)) -
                     float(first.get(obj, axis)))**2 for axis in axes))
            yield obj, pattern, moved / SPEED_WINDOW


def recorded_speeds(
    trajectories: Sequence[Sequence[State]],
    bodies: Dict[str, List[str]],
    switches: SwitchFeatures,
) -> RecordedSpeeds:
    """The top speed each body type shows under each switch pattern."""
    top: RecordedSpeeds = {}
    for states in trajectories:
        for obj, pattern, speed in window_speeds(states, bodies, switches):
            key = (obj.type.name, pattern)
            top[key] = max(top.get(key, 0.0), speed)
    return top
