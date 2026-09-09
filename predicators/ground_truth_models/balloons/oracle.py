"""The oracle's plan for a balloons level, executed on a probe instance.

The level's floating subset (the analytic law's unique answer) is freed
clip by clip, weakest lift first: push each clip open. The task
generator accepts a level only when this plan hangs the box inside the
band on the env's own physics with no balloon burst.
"""

from typing import List, Optional, Tuple

from predicators.envs.pybullet_balloons import PyBulletBalloonsEnv, \
    any_popped, box_in_band
from predicators.ground_truth_models.balloons.options import \
    probe_release_option, release_params
from predicators.settings import CFG
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
    release = probe_release_option()
    balloons = env._active_balloons(state)  # pylint: disable=protected-access
    clips = env._active_clips(state)  # pylint: disable=protected-access
    plan: Plan = []
    current = state
    for i in release_order(env, state, subset):
        grounded = release.ground(
            [env._robot, clips[i]],  # pylint: disable=protected-access
            release_params())
        nxt = env.run_option(current, grounded,
                             int(CFG.balloons_probe_max_steps))
        if nxt is None or nxt.get(balloons[i], "tied") < 0.5:
            return None
        plan.append(("Release", clips[i].name))
        current = nxt
        if any_popped(current) is not None:
            return None
    if box_in_band(current, env._box, env._band):  # pylint: disable=protected-access
        return plan
    return None
