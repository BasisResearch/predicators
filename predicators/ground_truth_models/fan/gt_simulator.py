"""Ground-truth simulator program for pybullet_fan residual dynamics.

Reproduces the wind dynamics from pybullet_fan.py on the
physics-command channel: while a fan is on, the rule emits an
impulse-mode world-frame force on the ball along the fan's facing
direction (``cmds.apply_force(..., hold=False)``), and the base sim's
engine does the rest - contact stops against the obstacle walls and
boundary slabs, sliding along their faces, corner deflection. The env
applies its wind inside ``_domain_specific_step`` (``_simulate_fans``),
which the approaches' base sims skip (``skip_residual_dynamics=True``);
this program is that hidden step's learned-space counterpart. Since the
env's own wind goes through the SAME residual-command executor (same
post-step emission, same first-substep impulse), a rule emitting the
env's force is bit-identical to the env - equivalence is structural,
not calibrated.

Contrast with the pre-command revision of this file, which modeled the
wind KINEMATICALLY (per-action displacement plus ~150 lines of
hand-derived contact geometry: sphere-overhang reach, per-blocker stop
distances, a minimum-translation rule for sliding). All of that is now
the engine's job; what remains is exactly the part an agent must
discover - which condition gates the force, its direction, its
magnitude, and its mode - so the program lives in the same hypothesis
space as an agent-synthesized one.

Why impulse mode (``hold=False``): the ball's rolling stiction stalls
a marginal HELD force into stick-slip creep (measured: ~0.0297 N held
matches the free-field 0.00228 m/action but freezes for ~100 actions
on the two-table seam), whereas the once-per-action 0.4 N spike
punches through every such threshold and gives the clean constant
speed the recorded data shows. A hold-mode hypothesis is not wrong a
priori - the rollout fit simply prefers the impulse on this data.

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

# Impulse-mode force (N): equals the env's wind_force_magnitude, and
# since both sides run through the same executor the match is exact by
# construction (steady-state free field: 0.00228 m/action). Fitted
# init; see the module docstring for why hold=False.
WIND_FORCE = 0.4


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
        cmds.apply_force(ball, (fx, fy, 0.0), hold=False)
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
