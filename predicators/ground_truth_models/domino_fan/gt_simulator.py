"""Ground-truth simulator program for pybullet_domino_fan residual dynamics.

The counterpart of ``fan/gt_simulator.py`` for the ball-free domino-fan
env: while a fan is on, the rule pushes the START domino along that
fan's facing direction and the base sim's engine owns everything
downstream -- the topple, the contact with the next block, the whole
cascade. Only the first block is wind-driven, which is what makes this
program so much smaller than the grid machinery ``pybullet_fan`` needs
for a ball whose entire trajectory is wind.

Two things differ from the ball's wind, and both are forced by the fact
that a domino stands on a narrow base rather than rolling.

**The push has to act above the centre of mass.** A force through the
centre is pure translation: the domino slides across the table
indefinitely and never tips, which is how a start block ends up
bulldozing into the target and "solving" a bridge task with none of the
bridge built. ``ApplyForce`` is centre-of-mass only by construction, so
the offset is expressed the way statics says it decomposes -- the same
force at the centre, plus the moment it would have made about it:

    r x F  with  r = (0, 0, lever),  F = (Fx, Fy, 0)
           =  (-lever * Fy,  lever * Fx,  0)

a torque about the horizontal axis perpendicular to the wind, which is
exactly the tipping moment. Emitting both is equivalent to applying the
force at ``lever`` above the origin, and keeps the whole dynamic inside
the residual-command vocabulary an agent-synthesized simulator can use.

**The wind stops when its target falls.** A standing domino presents
its face to the airstream; a fallen one lies flat, out of it. Without
this the fan goes on shoving a body that is already down. The 10-degree
threshold is the ``Toppled`` predicate's, so the dynamics stop exactly
where the symbol flips.

Both constants are fitted parameters, not fixed geometry: ``wind_force``
and ``wind_lever`` are what a system-ID pass has to recover from watching
dominoes fall, and neither is observable from a single state.

Because commands act through engine stepping, this artifact is scored
and fit by free-running rollout matching (``has_physics_rules``
routing), never teacher-forced.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from predicators.code_sim_learning.commands import CommandBuffer
from predicators.code_sim_learning.fit_space import ParamSpec
from predicators.code_sim_learning.utils import Params, ResidualUpdate, \
    objs_by_type
from predicators.ground_truth_models import GroundTruthSimulatorFactory
from predicators.settings import CFG
from predicators.structs import Object, State

# ── Constants ────────────────────────────────────────────────────

# Held-mode force (N), equal to the env's domino_fan_wind_force. A 100 g
# domino 15 mm thick and 150 mm tall, pushed at 0.4 of its height, tips
# at m*g*(depth/2)/(0.4*height) = 0.123 N; this sits ~60% over that, the
# same margin the ball's 0.06 N keeps over its stiction. Fitted init.
WIND_FORCE = 0.2

# Height above the domino's origin the wind effectively acts at (m),
# 0.4 * domino_height. Fitted init: the real lever depends on where the
# airstream meets the face, which no state feature reports.
WIND_LEVER = 0.06

# Past this roll the domino is down and out of the airstream. The
# Toppled predicate's threshold, deliberately.
TOPPLED_DEG = 10.0

# Below this the summed wind is nothing (N). Opposing fans cancel to
# floating-point dust rather than to exact zero -- four fans at 0.2 N
# sum to ~1e-16 -- and an exact == 0.0 test lets that dust through as a
# force and torque command on every step of every action.
WIND_EPS = 1e-9


def _is_start_block(state: State, domino: Object) -> bool:
    """Green means the chain starts here.

    Roles in this env are colours, not a feature: the generator paints
    the start block and the classifiers read it back. Kept in sync with
    ``DominoComponent._StartBlock_holds`` by construction -- both
    compare the same three channels against the same constant.
    """
    from predicators.envs.pybullet_domino.components.domino_component import \
        DominoComponent  # pylint: disable=import-outside-toplevel
    eps = 1e-3
    return all(
        abs(
            float(state.get(domino, c)) -
            DominoComponent.start_domino_color[i]) < eps
        for i, c in enumerate(("r", "g", "b")))


def _upright(state: State, domino: Object) -> bool:
    """Still standing, so still in the wind.

    Roll is only meaningful modulo pi -- a box turned 180 degrees about
    its own width axis is the same box -- so it is folded before being
    compared, exactly as the env's own check does it.
    """
    roll = float(state.get(domino, "roll"))
    roll = (roll + np.pi / 2) % np.pi - np.pi / 2
    return abs(roll) < np.radians(TOPPLED_DEG)


def _wind_topples_start_block(state: State, updates: ResidualUpdate,
                              params: Params,
                              cmds: CommandBuffer) -> ResidualUpdate:
    """Each on-fan pushes the standing start domino along its local +X.

    Simultaneous fans sum as force vectors, so two facing each other
    cancel to nothing -- the physical reading of ``pybullet_fan``'s
    ``FanOn(f) and FanOff(opposite(f))`` precondition, and the reason a
    plan that switches on both sides moves nothing at all.
    """
    objs = objs_by_type(state)
    dominos = objs.get("domino", [])
    if not dominos:
        return updates

    sign = -1.0 if CFG.fan_fans_blow_opposite_direction else 1.0
    fx = fy = 0.0
    for fan in objs.get("fan", []):
        if state.get(fan, "is_on") <= 0.5:
            continue
        rot = float(state.get(fan, "rot"))
        fx += sign * float(np.cos(rot)) * params["wind_force"]
        fy += sign * float(np.sin(rot)) * params["wind_force"]
    if abs(fx) < WIND_EPS and abs(fy) < WIND_EPS:
        return updates

    lever = params["wind_lever"]
    for domino in dominos:
        if not _is_start_block(state, domino) or not _upright(state, domino):
            continue
        cmds.apply_force(domino, (fx, fy, 0.0))
        # The moment that same force would have made about the centre
        # had it been applied ``lever`` higher up; see the module
        # docstring for the cross product.
        cmds.apply_torque(domino, (-lever * fy, lever * fx, 0.0))
    return updates


# ── Public API: consumed by read_simulator_components ────────────
# Same contract used by agent-synthesized simulator files.

RESIDUAL_RULES = [_wind_topples_start_block]

PARAM_SPECS: List[ParamSpec] = [
    ParamSpec("wind_force", WIND_FORCE, lo=0.0, hi=1.0),
    ParamSpec("wind_lever", WIND_LEVER, lo=0.0, hi=0.15),
]

# Features the wind dynamics own. Scored by the rollout objective
# against observations; NOT overwritten at plan time - the engine moves
# them under the emitted force and torque. Roll is in scope because the
# effect being modelled is a topple, not a shove: a program that got the
# force right and the lever wrong slides the block instead of tipping
# it, and only roll separates the two.
RESIDUAL_FEATURES: Dict[str, List[str]] = {
    "domino": ["x", "y", "roll"],
}

# ── Factory binding ──────────────────────────────────────────────


class PyBulletDominoFanGroundTruthSimulatorFactory(GroundTruthSimulatorFactory
                                                   ):
    """GT residual-dynamics simulator for pybullet_domino_fan.

    The simulator components (``RESIDUAL_RULES``, ``PARAM_SPECS``,
    ``RESIDUAL_FEATURES``) live as module globals above; this class only
    pins the env-name binding so ``get_gt_simulator`` can locate the
    right module via the factory registry.
    """

    @classmethod
    def get_env_names(cls) -> set:
        return {"pybullet_domino_fan"}
