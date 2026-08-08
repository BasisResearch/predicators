"""Ground-truth simulator program for pybullet_fan residual dynamics.

Reproduces the wind dynamics from pybullet_fan.py as residual rules.
The env applies the fan wind inside ``_domain_specific_step``
(``_simulate_fans``), which the approaches' base sims skip
(``skip_residual_dynamics=True``), so - unlike pybullet_domino, whose
toppling is plain rigid-body physics and whose GT simulator is a no-op
- the fan GT simulator must model the ball's wind-driven motion itself.

The model is calibrated against the real env (measured with
position-control stepping, 20 substeps/action, seed 0):

* While a fan is on, the ball moves at a constant, direction-independent
  ``BALL_STEP_SIZE`` = 0.00228 m per env action (steady state is reached
  within one action; the wind impulse is applied once per action and the
  high ball damping (10.0) dissipates it before the next one).
* When all fans turn off the ball stops immediately (zero coasting,
  measured < 1e-8 m).
* The ball parks against an obstacle wall with its center at
  ``wall_half_len + overhang`` from the wall center, where ``overhang``
  = sqrt(r^2 - (r - wall_h)^2) is the sphere's horizontal reach at the
  wall's top edge (the low walls let the ball overhang them): 0.03 +
  0.0346 = 0.0646, measured 0.0642.
* The boundary walls stop the ball center within 0.3 mm of the grid
  extremes themselves (face + overhang at boundary height ~ pos_gap/2),
  measured 1.5335 vs grid edge 1.534.

Known simplifications (all sub-tolerance for planning; the goal check
``BallAtTarget`` uses pos_gap/2 = 0.04):

* A ball riding along a wall while blown diagonally is ~24% faster in
  the free axis than in free field; we use the free-field speed.
* An off-center blocked ball is not deflected around the wall corner.
* Two simultaneous fans are modeled as the per-axis vector sum.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from predicators.code_sim_learning.fit_space import ParamSpec
from predicators.code_sim_learning.utils import Params, ResidualUpdate, \
    objs_by_type
from predicators.envs.pybullet_fan import PyBulletFanEnv
from predicators.ground_truth_models import GroundTruthSimulatorFactory
from predicators.settings import CFG
from predicators.structs import State

# ── Constants ────────────────────────────────────────────────────

# Steady-state ball displacement per env action while one fan is on
# (measured; NOT the env's kinematic_ball_speed=0.003, which belongs to
# the unused fan_use_kinematic mode).
BALL_STEP_SIZE = 0.00228


def _sphere_overhang(radius: float, wall_height: float) -> float:
    """Horizontal reach of the ball past a wall face at the wall's top.

    The walls are lower than the ball's center height, so the sphere
    overhangs them and its center stops closer to the wall than face +
    radius would suggest.
    """
    if wall_height >= radius:
        return radius
    return float(np.sqrt(radius**2 - (radius - wall_height)**2))


# Obstacle walls: in-axis stop distance (and perpendicular overlap
# threshold - the walls are square) between ball center and wall center.
_ENV = PyBulletFanEnv
WALL_CLEARANCE = _ENV.wall_x_len / 2 + _sphere_overhang(
    _ENV.ball_radius, _ENV.obstacle_wall_height)

# Boundary walls sit pos_gap/2 outside the extreme grid cells; with the
# thin face and the overhang at boundary-wall height, the ball center
# stops within a fraction of a millimeter of the grid extremes.
BOUNDARY_MARGIN = (
    _ENV.pos_gap / 2 - _ENV.boundary_wall_thickness / 2 -
    _sphere_overhang(_ENV.ball_radius, _ENV.boundary_wall_height))

# ── Residual rules ────────────────────────────────────────────────


def _axis_move(pos: float, perp: float, delta: float,
               walls: List[Tuple[float, float]], lo: float, hi: float,
               clearance: float) -> float:
    """Move one coordinate by ``delta`` with hard stops, never backward.

    ``walls`` holds (along, perp) wall-center coordinates in this axis's
    frame. Stops are one-sided: a ball already resting inside a stop
    (contact compliance in the real env leaves ~1.5 mm penetration) is
    not pushed back out, it just cannot advance further.
    """
    if delta == 0.0:
        return pos
    cand = pos + delta
    if delta > 0.0:
        cand = min(cand, max(pos, hi))
        for w_along, w_perp in walls:
            if abs(perp - w_perp) >= clearance:
                continue
            if pos < w_along:
                cand = min(cand, max(pos, w_along - clearance))
    else:
        cand = max(cand, min(pos, lo))
        for w_along, w_perp in walls:
            if abs(perp - w_perp) >= clearance:
                continue
            if pos > w_along:
                cand = max(cand, min(pos, w_along + clearance))
    return cand


def _wind_blowing(state: State, updates: ResidualUpdate,
                  params: Params) -> ResidualUpdate:
    """Active fans blow the ball; walls and the grid boundary block it.

    Blocking is a hard clamp, unlike boil's sigmoid-softened gates: a
    hinge still has gradient 1 in ``wall_clearance`` on every sample
    where the clamp binds (parked-against-wall steps), so the LM
    Jacobian stays informative without leaky walls.
    """
    objs = objs_by_type(state)
    balls = objs.get("ball", [])
    if not balls:
        return updates
    ball = balls[0]

    # Wind direction: each on-fan blows along its local +X (rot is the
    # base Z euler), summed per axis for simultaneous fans.
    sign = -1.0 if CFG.fan_fans_blow_opposite_direction else 1.0
    dx = dy = 0.0
    any_on = False
    for fan in objs.get("fan", []):
        if state.get(fan, "is_on") <= 0.5:
            continue
        any_on = True
        rot = float(state.get(fan, "rot"))
        dx += sign * float(np.cos(rot))
        dy += sign * float(np.sin(rot))
    if not any_on:
        # Leave the ball to the base sim (zero coasting was measured, so
        # fans-off implies no wind motion at all).
        return updates
    dx *= params["ball_speed"]
    dy *= params["ball_speed"]

    # Grid boundary, inferred from the stationary target exactly like the
    # env's boundary-wall construction and the oracle helper injection.
    bx = float(state.get(ball, "x"))
    by = float(state.get(ball, "y"))
    targets = objs.get("target", [])
    if targets:
        x_coords, y_coords = PyBulletFanEnv._grid_coords_for_point(  # pylint: disable=protected-access
            float(state.get(targets[0], "x")),
            float(state.get(targets[0], "y")))
        x_lo = min(x_coords) - BOUNDARY_MARGIN
        x_hi = max(x_coords) + BOUNDARY_MARGIN
        y_lo = min(y_coords) - BOUNDARY_MARGIN
        y_hi = max(y_coords) + BOUNDARY_MARGIN
    else:  # pragma: no cover - tasks always have a target
        x_lo, x_hi = PyBulletFanEnv.x_lb, PyBulletFanEnv.x_ub
        y_lo, y_hi = PyBulletFanEnv.y_lb, PyBulletFanEnv.y_ub

    walls_xy = [(float(state.get(w, "x")), float(state.get(w, "y")))
                for w in objs.get("wall", [])]
    clearance = params["wall_clearance"]

    # Axis-by-axis move (x then y) so a blocked axis still allows sliding
    # along the wall in the other axis.
    new_x = _axis_move(bx, by, dx, walls_xy, x_lo, x_hi, clearance)
    walls_yx = [(wy, wx) for wx, wy in walls_xy]
    new_y = _axis_move(by, new_x, dy, walls_yx, y_lo, y_hi, clearance)

    updates.setdefault(ball, {})["x"] = new_x
    updates.setdefault(ball, {})["y"] = new_y
    return updates


# ── Public API: consumed by read_simulator_components ────────────
# Same contract used by agent-synthesized simulator files.

RESIDUAL_RULES = [_wind_blowing]

PARAM_SPECS: List[ParamSpec] = [
    ParamSpec("ball_speed", BALL_STEP_SIZE, lo=0.0),
    ParamSpec("wall_clearance", WALL_CLEARANCE, lo=0.0),
]

RESIDUAL_FEATURES: Dict[str, List[str]] = {
    "ball": ["x", "y"],
}

# ── Factory binding ──────────────────────────────────────────────


class PyBulletFanGroundTruthSimulatorFactory(GroundTruthSimulatorFactory):
    """GT residual-dynamics simulator for pybullet_fan.

    The actual simulator components (``RESIDUAL_RULES``, ``PARAM_SPECS``,
    ``RESIDUAL_FEATURES``) live as module globals above; this class only
    pins the env-name binding so ``get_gt_simulator`` can locate the
    right module via the factory registry.
    """

    @classmethod
    def get_env_names(cls) -> set:
        return {"pybullet_fan"}
