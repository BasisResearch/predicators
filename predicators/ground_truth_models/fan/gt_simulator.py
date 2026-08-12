"""Ground-truth simulator program for pybullet_fan residual dynamics.

Reproduces the wind dynamics from pybullet_fan.py on the
physics-command channel: while a fan is on, the rule applies a constant
world-frame force to the ball along the fan's facing direction
(``cmds.apply_force``), and the base sim's engine does the rest -
contact stops against the obstacle walls and boundary slabs, sliding
along their faces, corner deflection. The env applies its wind inside
``_domain_specific_step`` (``_simulate_fans``), which the approaches'
base sims skip (``skip_residual_dynamics=True``); this program is that
hidden step's learned-space counterpart, emitted at the same post-step
cadence and executed across the next action's physics substeps.

Contrast with the previous revision of this file, which modeled the
wind KINEMATICALLY (per-action displacement plus ~150 lines of
hand-derived contact geometry: sphere-overhang reach, per-blocker stop
distances, a minimum-translation rule for sliding). All of that is now
the engine's job; what remains is exactly the part an agent must
discover - which condition gates the force, its direction, and its
magnitude - so the program lives in the same hypothesis space as an
agent-synthesized one.

Calibration: the env applies its wind as a 0.4 N impulse once per
action (a single ``applyExternalForce`` consumed by the first substep
of the next action), whereas the command channel holds the force across
all ``pybullet_sim_steps_per_action`` = 20 substeps. ``WIND_FORCE``
below is measured (scripted bisection on base-sim rollouts, seed 0)
so the steady-state free-field ball speed matches the env's 0.00228
m/action; the rollout-matching fit refines it from data. The response
is NOT linear in the force: a continuous push must overcome rolling
stiction that the env's impulse punches through, so speeds are ~0
below ~0.028 N and rise steeply after - keep the init near the
calibrated point rather than deriving it as impulse/substeps.

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
from predicators.structs import State

# ── Constants ────────────────────────────────────────────────────

# Continuous per-substep force (N) whose steady-state per-action ball
# displacement matches the env's measured 0.00228 m/action. Fitted
# init; see the module docstring for the calibration and the stiction
# cliff just below it.
WIND_FORCE = 0.0302


def _wind_blowing(state: State, updates: ResidualUpdate, params: Params,
                  cmds: CommandBuffer) -> ResidualUpdate:
    """Each on-fan blows the ball along its local +X; the engine owns
    contacts.

    The force is re-emitted every step any fan is on and expires with
    the next action otherwise, so fans-off means no wind - the measured
    zero coasting comes from the ball's high damping in the base sim.
    Simultaneous fans sum as force vectors.
    """
    objs = objs_by_type(state)
    balls = objs.get("ball", [])
    if not balls:
        return updates

    # Wind direction: each on-fan blows along its local +X (rot is the
    # base Z euler), summed per axis for simultaneous fans.
    sign = -1.0 if CFG.fan_fans_blow_opposite_direction else 1.0
    fx = fy = 0.0
    for fan in objs.get("fan", []):
        if state.get(fan, "is_on") <= 0.5:
            continue
        rot = float(state.get(fan, "rot"))
        fx += sign * float(np.cos(rot)) * params["wind_force"]
        fy += sign * float(np.sin(rot)) * params["wind_force"]
    if fx == 0.0 and fy == 0.0:
        return updates
    for ball in balls:
        cmds.apply_force(ball, (fx, fy, 0.0))
    return updates


# ── Public API: consumed by read_simulator_components ────────────
# Same contract used by agent-synthesized simulator files.

RESIDUAL_RULES = [_wind_blowing]

PARAM_SPECS: List[ParamSpec] = [
    ParamSpec("wind_force", WIND_FORCE, lo=0.0, hi=1.0),
]

# Features the wind dynamics own. Scored by the rollout objective
# against observations; NOT overwritten at plan time - the engine moves
# them under the emitted force.
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
