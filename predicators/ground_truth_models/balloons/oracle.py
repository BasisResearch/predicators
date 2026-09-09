"""The oracle's plan for a balloons level, executed on a probe instance.

The reference subset is freed clip by clip, weakest lift first. Task
generation verifies every immediate release order of this reference
against the evaluator, allowing other subsets and orders to win too.
"""

from typing import List, Optional, Tuple

from predicators.envs.pybullet_balloons import PyBulletBalloonsEnv
from predicators.structs import State

Plan = List[Tuple[str, str]]


def release_order(env: PyBulletBalloonsEnv, state: State,
                  subset: Tuple[int, ...]) -> List[int]:
    """The subset's balloons, weakest lift first."""
    balloons = env._active_balloons(state)  # pylint: disable=protected-access
    return sorted(subset,
                  key=lambda i: env.lift_at_ground(
                      int(round(state.get(balloons[i], "color")))))


def solve_level(env: PyBulletBalloonsEnv, state: State) -> Optional[Plan]:
    """Run the oracle's plan from ``state`` on ``env``; the plan if the box
    ends inside the band with nothing burst, else None."""
    subset = env.solution_subset(state)
    if subset is None:
        return None
    order = release_order(env, state, subset)
    outcome = env.release_sequence_outcome(state, order)
    if not outcome.won:
        return None
    clips = env._active_clips(state)  # pylint: disable=protected-access
    return [("Release", clips[i].name) for i in order]
