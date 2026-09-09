"""Ground-truth simulator program for pybullet_launcher residual dynamics.

The launcher env applies its launch law in ``_domain_specific_step``
(``PyBulletLauncherEnv``), which the approaches' base sims skip
(``skip_residual_dynamics=True``). This program is that hidden step's
learned-space counterpart, written in the same contract an agent's
``simulator.py`` uses, so it doubles as the existence proof that the
contract can express the domain:

* THE SNAP is visible: the base sim's handle springs home the moment
  the hand lets go, so the observed ``compression`` drops to zero on
  that step. What the snap does to the ball is the secret.
* THE LAUNCH LAW is a recurrent rule. It carries the deepest
  compression seen since the last shot in its latent block and, on the
  step the compression returns to zero from a positive value, sends the
  loaded ball off along the barrel at the spring constant times that
  peak (``cmds.set_velocity``). The flight, the impact and the tower's
  fall are the engine's.

The block masses are not residual parameters: the base env exposes
them as physical parameters (``get_physical_param_info`` /
``apply_physical_param_overrides``), identified by the physical sysID
path rather than by this rule.

Because the rule acts through engine stepping, this artifact is scored
and fit by free-running rollout matching (``has_physics_rules``
routing), with the latent threaded per trajectory.

Timing detail: the env reads the handle where this action's physics
left it before the snap, while a rule only sees observations, so the
rule's peak is the deepest compression in the observations before the
snap. The Cock skill holds the handle still until it lets go, so the
two agree; only a handle pushed deeper on the very step it is released
would differ.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from predicators.code_sim_learning.commands import CommandBuffer
from predicators.code_sim_learning.fit_space import ParamSpec
from predicators.code_sim_learning.utils import History, Params, \
    ResidualUpdate, objs_by_type
from predicators.envs.pybullet_launcher_base import PyBulletLauncherBaseEnv
from predicators.ground_truth_models import GroundTruthSimulatorFactory
from predicators.settings import CFG
from predicators.structs import State

# ── Geometry shared with the visible base sim ─────────────────────

_GEOM = PyBulletLauncherBaseEnv


def _ball_in_cup(state: State, ball: Any) -> bool:
    """Whether the ball rests in the muzzle cup (the visible geometry)."""
    cx, cy, cz = _GEOM.cup_position()
    dist = np.linalg.norm([
        float(state.get(ball, "x")) - cx,
        float(state.get(ball, "y")) - cy,
        float(state.get(ball, "z")) - cz,
    ])
    return bool(dist < _GEOM.cup_radius)


def _launching(observation: State, latent: Dict[str, Any], history: History,
               updates: ResidualUpdate, params: Params,
               cmds: CommandBuffer) -> ResidualUpdate:
    """Track the handle's deepest compression; on the snap, fire the loaded
    ball along the barrel at ``spring_k`` times that peak.

    ``latent["prev"]`` is the compression observed last step and
    ``latent["peak"]`` the deepest since the last shot. The snap is the
    observed compression falling to zero from a positive value.
    """
    del history
    objs = objs_by_type(observation)
    launchers = objs.get("launcher", [])
    balls = objs.get("ball", [])
    if not launchers or not balls:
        return updates
    launcher, ball = launchers[0], balls[0]
    compression = float(observation.get(launcher, "compression"))
    prev = float(latent.get("prev", 0.0))
    peak = max(float(latent.get("peak", 0.0)), compression)
    if compression <= 0.0 < prev:
        # The snap: the handle is home, the spring's energy is in the
        # ball if one is loaded and the handle was pulled far enough
        # for the spring to engage.
        if peak >= float(params["min_compression"]) and \
                _ball_in_cup(observation, ball):
            speed = float(params["spring_k"]) * peak
            dx, dy, dz = _GEOM.launch_direction(
                float(observation.get(launcher, "angle")))
            cmds.set_velocity(ball, (speed * dx, speed * dy, speed * dz))
        peak = 0.0
    latent["prev"] = compression
    latent["peak"] = peak
    return updates


# ── Public API: consumed by read_simulator_components ────────────
# Same contract used by agent-synthesized simulator files.

RESIDUAL_RULES = [_launching]


def _latent_init() -> Dict[str, float]:
    return {"prev": 0.0, "peak": 0.0}


LATENT_INIT = _latent_init


def _build_param_specs() -> List[ParamSpec]:
    """The spring constant (m/s per metre of compression) and the least
    compression that engages the spring, read from CFG at consumption time so
    the module imports before the config is final."""
    return [
        ParamSpec("spring_k", float(CFG.launcher_spring_k), lo=1.0, hi=100.0),
        ParamSpec("min_compression",
                  float(CFG.launcher_min_compression),
                  lo=0.0,
                  hi=0.05),
    ]


PARAM_SPECS = _build_param_specs

# Features the hidden physics own: the ball's flight is the engine's
# answer to the commanded launch, scored by the rollout objective
# against observations and never overwritten at plan time; the tower
# falls (or stands) under the impact.
RESIDUAL_FEATURES: Dict[str, List[str]] = {
    "ball": ["x", "y", "z", "speed"],
    "block": ["x", "y", "z", "roll", "pitch", "yaw"],
}

# ── Factory binding ──────────────────────────────────────────────


class PyBulletLauncherGroundTruthSimulatorFactory(GroundTruthSimulatorFactory):
    """GT residual-dynamics simulator for pybullet_launcher.

    The simulator components (``RESIDUAL_RULES``, ``PARAM_SPECS``,
    ``RESIDUAL_FEATURES``, ``LATENT_INIT``) live as module globals
    above; this class only pins the env-name binding so
    ``get_gt_simulator`` can locate the module via the factory registry.
    """

    @classmethod
    def get_env_names(cls) -> set:
        return {"pybullet_launcher"}
