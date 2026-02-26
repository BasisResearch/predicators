"""Reward function for domino chain-reaction tasks.

Evaluates whether targets were toppled via a genuine chain reaction
starting from the start domino, rather than direct robot manipulation.

The reward decomposes into five components:
  1. target_score:      fraction of targets actually toppled
  2. order_score:       start domino toppled before any target
  3. robot_dist_score:  robot far from targets at the moment they topple
  4. propagation_score: topple times correlate with distance from start
  5. spread_score:      topples spread over time (not simultaneous)
"""

from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from predicators.structs import LowLevelTrajectory, Object, State, Task, Type

# From domino_component.py
FALLEN_THRESHOLD = np.pi * 2 / 5  # ~72 deg — domino considered toppled

# Color constants (r, g, b) for domino classification
_START_COLOR = (0.56, 0.93, 0.56)
_TARGET_COLOR = (0.85, 0.7, 0.85)
_MOVEABLE_COLOR = (0.6, 0.8, 1.0)

# Reward tuning
_ROBOT_SAFE_DIST = 0.20  # metres; ~3 domino widths
_COLOR_TOL = 0.1  # tolerance for RGB matching
_MIN_SPREAD_PER_DOMINO = 3  # expected timesteps between consecutive topples


def _color_matches(state: State,
                   obj: Object,
                   target_rgb: Tuple[float, float, float],
                   tol: float = _COLOR_TOL) -> bool:
    r, g, b = state.get(obj, "r"), state.get(obj, "g"), state.get(obj, "b")
    return (abs(r - target_rgb[0]) < tol and abs(g - target_rgb[1]) < tol
            and abs(b - target_rgb[2]) < tol)


def _classify_dominoes(
    state: State,
    dominoes: Sequence[Object],
) -> Tuple[List[Object], List[Object], List[Object]]:
    """Classify dominoes into (start, moveable, target) by colour."""
    start, moveable, targets = [], [], []
    for d in dominoes:
        if _color_matches(state, d, _START_COLOR):
            start.append(d)
        elif _color_matches(state, d, _TARGET_COLOR):
            targets.append(d)
        else:
            moveable.append(d)
    return start, moveable, targets


def _find_topple_times(
    states: Sequence[State],
    dominoes: Sequence[Object],
) -> Dict[Object, int]:
    """Return {domino: first_timestep_where_toppled}."""
    topple_times: Dict[Object, int] = {}
    for d in dominoes:
        for t, state in enumerate(states):
            if abs(state.get(d, "roll")) >= FALLEN_THRESHOLD:
                topple_times[d] = t
                break
    return topple_times


def _spearman_corr(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rank correlation (no scipy dependency)."""
    n = len(x)
    if n < 3:
        return 0.0
    xa, ya = np.asarray(x, dtype=float), np.asarray(y, dtype=float)

    def _ranks(arr: np.ndarray) -> np.ndarray:
        order = np.argsort(arr)
        r = np.empty_like(order, dtype=float)
        r[order] = np.arange(n, dtype=float)
        return r

    rx, ry = _ranks(xa), _ranks(ya)
    mx, my = rx.mean(), ry.mean()
    dx, dy = rx - mx, ry - my
    denom = np.sqrt(float((dx**2).sum() * (dy**2).sum()))
    if denom < 1e-12:
        return 0.0
    return float((dx * dy).sum() / denom)


# ------------------------------------------------------------------ #
# Main reward function
# ------------------------------------------------------------------ #


def domino_chain_reward(
    trajectory: LowLevelTrajectory,
    types: Set[Type],
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """Score a trajectory on how well it achieves a domino chain reaction.

    Args:
        trajectory: recorded states (and actions) from an episode.
        types:      the environment's type set (must contain "domino", "robot").
        weights:    optional dict overriding default component weights.
                    Keys: target, order, robot_dist, propagation, spread.

    Returns:
        float in [0, 1].
    """
    w = {
        "target": 0.30,
        "order": 0.20,
        "robot_dist": 0.20,
        "propagation": 0.15,
        "spread": 0.15,
    }
    if weights:
        w.update(weights)

    states = trajectory.states
    if len(states) < 2:
        return 0.0

    # --- resolve types ---
    domino_type = next((t for t in types if t.name == "domino"), None)
    robot_type = next((t for t in types if t.name == "robot"), None)
    if domino_type is None:
        return 0.0

    all_dominoes = states[0].get_objects(domino_type)
    robot = (states[0].get_objects(robot_type)[0]
             if robot_type and states[0].get_objects(robot_type) else None)

    start, moveable, targets = _classify_dominoes(states[0], all_dominoes)
    if not start or not targets:
        return 0.0

    topple_times = _find_topple_times(states, all_dominoes)

    # ---- 1. target_score: fraction of targets toppled ----
    n_toppled = sum(1 for t in targets if t in topple_times)
    target_score = n_toppled / len(targets)
    if target_score == 0.0:
        return 0.0  # nothing else to evaluate

    # ---- 2. order_score: start topples before every target ----
    start_time = min(topple_times.get(s, len(states)) for s in start)
    earliest_target = min(topple_times[t] for t in targets
                          if t in topple_times)
    order_score = 1.0 if start_time < earliest_target else 0.0

    # ---- 3. robot_dist_score: robot far from ALL dominoes when they topple --
    # Exception: the start domino (robot must push it to initiate the chain).
    if robot is not None:
        dists: List[float] = []
        non_start = [d for d in all_dominoes if d not in start]
        for d in non_start:
            if d not in topple_times:
                continue
            s = states[topple_times[d]]
            rx, ry = s.get(robot, "x"), s.get(robot, "y")
            dx, dy = s.get(d, "x"), s.get(d, "y")
            dist = np.hypot(rx - dx, ry - dy)
            dists.append(min(dist / _ROBOT_SAFE_DIST, 1.0))
        robot_dist_score = float(np.mean(dists)) if dists else 1.0
    else:
        robot_dist_score = 1.0

    # ---- 4. propagation_score: topple order matches distance from start ----
    start_xy = np.array([
        states[0].get(start[0], "x"),
        states[0].get(start[0], "y"),
    ])
    toppled_items = [(d, topple_times[d]) for d in all_dominoes
                     if d in topple_times]
    if len(toppled_items) >= 3:
        dists_from_start = [
            np.hypot(states[0].get(d, "x") - start_xy[0],
                     states[0].get(d, "y") - start_xy[1])
            for d, _ in toppled_items
        ]
        times = [float(tt) for _, tt in toppled_items]
        corr = _spearman_corr(dists_from_start, times)
        propagation_score = max(0.0, corr)
    else:
        propagation_score = 0.5  # insufficient data, neutral

    # ---- 5. spread_score: topples spread over time, not simultaneous ----
    if len(toppled_items) >= 2:
        sorted_times = sorted(tt for _, tt in toppled_items)
        spread = sorted_times[-1] - sorted_times[0]
        expected = len(toppled_items) * _MIN_SPREAD_PER_DOMINO
        spread_score = min(spread / max(expected, 1), 1.0)
    else:
        spread_score = 0.5

    # ---- weighted combination ----
    reward = (w["target"] * target_score + w["order"] * order_score +
              w["robot_dist"] * robot_dist_score +
              w["propagation"] * propagation_score +
              w["spread"] * spread_score)

    return float(np.clip(reward, 0.0, 1.0))
