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
* The ball parks against a blocker with its center at ``half_extent +
  reach`` from the blocker center, where ``reach`` is the sphere's
  horizontal extent at the blocker's top edge (see ``_horizontal_reach``).
  For the low obstacle walls the ball overhangs them: 0.03 + 0.0346 =
  0.0646 (measured 0.0642). The taller boundary slabs reach past the
  ball's equator, so ``reach`` is the full radius: 0.001 + 0.04 = 0.041
  from the slab center, i.e. the ball center stops within a millimetre of
  the grid extremes (measured 1.5335 vs grid edge 1.534).

Every quantity above is a function of features the agent observes: the
blocker poses and extents (``x``/``y``/``z`` + ``x_len``/``y_len``/
``z_len`` on the ``wall`` and ``boundary`` types) and the ball's ``z``
and ``radius``. This program therefore lives in the same hypothesis
space as an agent-synthesized one - it needs no privileged access to
the task grid, and the boundary slabs are ordinary blockers rather than
latent structure.

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
from predicators.ground_truth_models import GroundTruthSimulatorFactory
from predicators.settings import CFG
from predicators.structs import Object, State

# ── Constants ────────────────────────────────────────────────────

# Steady-state ball displacement per env action while one fan is on
# (measured; NOT the env's kinematic_ball_speed=0.003, which belongs to
# the unused fan_use_kinematic mode).
BALL_STEP_SIZE = 0.00228

# Types the ball collides with. Obstacle walls and the arena boundary
# slabs obey the same contact law and differ only in their extents, so
# the rule below iterates them together; they are separate types purely
# so that predicates and NSRTs can quantify over the obstacles alone.
BLOCKER_TYPES = ("wall", "boundary")


def _horizontal_reach(ball_z: float, ball_radius: float,
                      blocker_top_z: float) -> float:
    """How far past a blocker's face the ball's center can come to rest.

    A blocker taller than the ball's center is a plain side contact at
    the equator, so the reach is the full radius. A lower one is
    overhung: the ball rests on its top edge and its center comes in by
    the sphere's horizontal extent at that height. A blocker lower than
    the ball's underside does not block at all.
    """
    drop = ball_z - blocker_top_z
    if drop <= 0.0:
        return ball_radius
    if drop >= ball_radius:
        return 0.0
    return float(np.sqrt(ball_radius**2 - drop**2))


def _blocker_stops(state: State, blockers: List[Object], ball_z: float,
                   ball_radius: float,
                   slack: float) -> List[Tuple[float, float, float, float]]:
    """Precompute each blocker's (x, y, x_stop, y_stop) in world axes.

    ``x_stop`` is the center-to-center distance at which the ball parks
    when approaching along x; it doubles as the perpendicular overlap
    threshold for a y-approach, and vice versa. Blockers the ball rides
    over entirely are dropped.
    """
    out = []
    for b in blockers:
        top_z = float(state.get(b, "z")) + float(state.get(b, "z_len")) / 2
        reach = _horizontal_reach(ball_z, ball_radius, top_z)
        if reach <= 0.0:
            continue
        x_stop = float(state.get(b, "x_len")) / 2 + reach + slack
        y_stop = float(state.get(b, "y_len")) / 2 + reach + slack
        b_x = float(state.get(b, "x"))
        b_y = float(state.get(b, "y"))
        out.append((b_x, b_y, x_stop, y_stop))
    return out


def _axis_move(pos: float, perp: float, delta: float,
               blockers: List[Tuple[float, float, float,
                                    float]], along_idx: int) -> float:
    """Move one coordinate by ``delta`` with hard stops, never backward.

    ``blockers`` holds (x, y, x_stop, y_stop) tuples; ``along_idx`` is 0
    when moving along x and 1 when moving along y.

    Which axis a blocker actually stops is decided by comparing the two
    overlaps, the standard minimum-translation rule. It matters here
    because the boundary slabs are long: a ball resting against the left
    slab's face overlaps it slightly in x and hugely in y, and a test
    that only asked "do the perpendicular extents overlap?" would freeze
    the ball's y motion instead of letting it slide along the face.

    Stops are one-sided: a ball already resting inside a stop (contact
    compliance in the real env leaves ~1.5 mm penetration) is not pushed
    back out, it just cannot advance further.
    """
    if delta == 0.0:
        return pos
    perp_idx = 1 - along_idx
    cand = pos + delta
    for blocker in blockers:
        b_along = blocker[along_idx]
        b_perp = blocker[perp_idx]
        # Stop distances are indexed the same way: entry 2 is the x
        # distance, entry 3 the y distance.
        stop_along = blocker[2 + along_idx]
        stop_perp = blocker[2 + perp_idx]
        # Approaching from behind is never blocked.
        if (delta > 0.0) != (pos < b_along):
            continue
        overlap_perp = stop_perp - abs(perp - b_perp)
        if overlap_perp <= 0.0:
            continue  # beside the blocker: nothing in the way
        overlap_along = stop_along - abs(pos - b_along)
        if overlap_along > 0.0:
            # Already interpenetrating on both axes. This is the blocking
            # axis only if it is the shallower one - otherwise the ball is
            # pressed against a face perpendicular to this move and is
            # free to slide along it.
            if overlap_along < overlap_perp:
                cand = min(cand, pos) if delta > 0.0 else max(cand, pos)
            continue
        face = b_along - stop_along if delta > 0.0 else b_along + stop_along
        cand = min(cand, face) if delta > 0.0 else max(cand, face)
    return cand


def _wind_blowing(state: State, updates: ResidualUpdate,
                  params: Params) -> ResidualUpdate:
    """Active fans blow the ball; walls and boundary slabs block it.

    Blocking is a hard clamp, unlike boil's sigmoid-softened gates: a
    hinge still has gradient 1 in ``contact_slack`` on every sample
    where the clamp binds (parked-against-blocker steps), so the LM
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

    bx = float(state.get(ball, "x"))
    by = float(state.get(ball, "y"))
    blockers: List[Object] = []
    for type_name in BLOCKER_TYPES:
        blockers.extend(objs.get(type_name, []))
    stops = _blocker_stops(state, blockers, float(state.get(ball, "z")),
                           float(state.get(ball, "radius")),
                           params["contact_slack"])

    # Axis-by-axis move (x then y) so a blocked axis still allows sliding
    # along the wall in the other axis.
    new_x = _axis_move(bx, by, dx, stops, 0)
    new_y = _axis_move(by, new_x, dy, stops, 1)

    updates.setdefault(ball, {})["x"] = new_x
    updates.setdefault(ball, {})["y"] = new_y
    return updates


# ── Public API: consumed by read_simulator_components ────────────
# Same contract used by agent-synthesized simulator files.

RESIDUAL_RULES = [_wind_blowing]

PARAM_SPECS: List[ParamSpec] = [
    ParamSpec("ball_speed", BALL_STEP_SIZE, lo=0.0),
    # Contact compliance only: the geometric part of the stop distance is
    # computed from the observed blocker extents and ball radius, so what
    # remains to fit is the sub-millimetre penetration the real contact
    # solver leaves (measured -0.4 mm on obstacle walls, -1.5 mm on the
    # boundary slabs).
    ParamSpec("contact_slack", 0.0, lo=-0.01, hi=0.01),
]

RESIDUAL_FEATURES: Dict[str, List[str]] = {
    "ball": ["x", "y"],
}

# ── Factory binding ──────────────────────────────────────────────


class PyBulletFanGroundTruthSimulatorFactory(GroundTruthSimulatorFactory):
    """GT residual-dynamics simulator for pybullet_fan.

    The actual simulator components (``RESIDUAL_RULES``,
    ``PARAM_SPECS``, ``RESIDUAL_FEATURES``) live as module globals
    above; this class only pins the env-name binding so
    ``get_gt_simulator`` can locate the right module via the factory
    registry.
    """

    @classmethod
    def get_env_names(cls) -> set:
        return {"pybullet_fan"}
