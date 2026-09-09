"""The oracle's plan for a magnets level, executed on a probe instance.

Every piece the wand pulls is carried in turn: ``Jump`` over it (the
piece slides under the tip), ``Hover`` to its slot (it follows), and the
next ``Jump`` leaves it there. The task generator accepts a level only
when this plan clears it on the env's own physics with no piece lost, so
every level the oracle sees is one it can solve, and the process model's
samplers reproduce exactly these targets.
"""

from typing import List, Optional, Tuple

import numpy as np

from predicators.envs.pybullet_magnets import PyBulletMagnetsEnv, piece_lost
from predicators.ground_truth_models.magnets.options import probe_move_options
from predicators.settings import CFG
from predicators.structs import State

Plan = List[Tuple[str, float, float]]


def carry_order(env: PyBulletMagnetsEnv, state: State) -> List[int]:
    """Indices of the pulled pieces, nearest slot first."""
    pieces, slots = env._active_objects(state)  # pylint: disable=protected-access
    order = []
    for i, piece in enumerate(pieces):
        color = int(round(state.get(piece, "color")))
        if env.polarity(color) <= 0:
            continue
        slot = next(s for s in slots
                    if int(round(state.get(s, "color"))) == color)
        dist = np.hypot(
            state.get(piece, "x") - state.get(slot, "x"),
            state.get(piece, "y") - state.get(slot, "y"))
        order.append((dist, i))
    return [i for _, i in sorted(order)]


def solve_level(env: PyBulletMagnetsEnv, state: State) -> Optional[Plan]:
    """Run the oracle's plan from ``state`` on ``env``; the plan if it wins
    with every piece still on the mat, else None."""
    hover, jump = probe_move_options()
    pieces, slots = env._active_objects(state)  # pylint: disable=protected-access
    goal = env.goal_for(state)
    plan: Plan = []
    current = state
    for i in carry_order(env, state):
        piece = pieces[i]
        color = int(round(state.get(piece, "color")))
        slot = next(s for s in slots
                    if int(round(state.get(s, "color"))) == color)
        for name, option, target in (
            ("Jump", jump, piece),
            ("Hover", hover, slot),
        ):
            x, y = current.get(target, "x"), current.get(target, "y")
            grounded = option.ground(
                [env._robot, env._wand],  # pylint: disable=protected-access
                np.array([x, y], dtype=np.float32))
            nxt = env.run_option(current, grounded,
                                 int(CFG.magnets_probe_max_steps))
            if nxt is None:
                return None
            plan.append((name, float(x), float(y)))
            current = nxt
            if any(piece_lost(current, pc) for pc in pieces):
                return None
    if all(atom.holds(current) for atom in goal):
        return plan
    return None
