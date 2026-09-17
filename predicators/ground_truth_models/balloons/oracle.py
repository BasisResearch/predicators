"""The oracle's plan for a balloons level, executed on a probe instance.

The reference subset is freed clip by clip, weakest lift first. Task
generation verifies every immediate release order of this reference
against the evaluator, allowing other subsets and orders to win too. On
a bundle level the reference order is the witnessed one and the box
settles between cuts, so the plan waits after every cut but the last.
"""

from typing import List, Optional, Tuple

from predicators.envs.pybullet_balloons import PyBulletBalloonsEnv
from predicators.structs import State

Plan = List[Tuple[str, str]]


def release_order(env: PyBulletBalloonsEnv, state: State,
                  subset: Tuple[int, ...]) -> List[int]:
    """The subset's clips, weakest lift first, unless the probes only saw
    another order win (see ``PyBulletBalloonsEnv.reference_order``)."""
    return env.reference_order(state, subset)


def solve_level(env: PyBulletBalloonsEnv, state: State) -> Optional[Plan]:
    """Run the oracle's plan from ``state`` on ``env``; the plan if the box
    ends inside the band with nothing burst, else None."""
    subset = env.solution_subset(state)
    if subset is None:
        return None
    order = release_order(env, state, subset)
    settled = env.is_bundled(state)
    outcome = env.release_sequence_outcome(state, order, settled)
    if not outcome.won:
        return None
    clips = env._active_clips(state)  # pylint: disable=protected-access
    plan: Plan = []
    for i in order:
        if plan and settled:
            plan.append(("Wait", env._robot.name))  # pylint: disable=protected-access
        plan.append(("Release", clips[i].name))
    return plan
