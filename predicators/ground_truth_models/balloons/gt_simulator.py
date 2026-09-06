"""Ground-truth simulator program for pybullet_balloons residual dynamics.

The balloons env applies its hidden physics in ``_domain_specific_step``
(``PyBulletBalloonsEnv``), which the approaches' base sims skip
(``skip_residual_dynamics=True``). This program is that hidden step's
learned-space counterpart, written in the same contract an agent's
``simulator.py`` uses, so it doubles as the existence proof that the
contract can express the domain:

* RELEASE - a balloon whose clip has been pushed open (``is_on``) and
  that is not yet ``tied`` is seated at the end of its string above the
  box's top, stacked above the balloons freed before it. This is a
  feature update (``tied``, ``x``, ``y``, ``z``): the base env's
  ``_set_state`` places the balloon there on the next step, exactly as
  the env's own teleport does.
* POP - a balloon whose top reaches the ceiling plate is ``popped``, a
  feature update; a popped balloon pulls no more.
* LIFT - every tied balloon is welded to the box (``cmds.attach``) and,
  while intact, pulls upward with its colour's lift faded linearly with
  the box's height (``cmds.apply_force``). The engine integrates the
  box's rise, the string's pull and the air's drag.

The box masses and the air's drag are not residual parameters: the
base env exposes them as physical parameters
(``get_physical_param_info`` / ``apply_physical_param_overrides``),
identified by the physical sysID path rather than by these rules.

Because the rules act through engine stepping, this artifact is scored
and fit by free-running rollout matching (``has_physics_rules``
routing), never teacher-forced.

Known limit of the contract: a release is a feature update, so on the
hybrid sim it goes through ``_set_state``, which does not carry the
box's velocity. A release while the box is already moving therefore
restarts the box from rest on the hybrid, where the env keeps its
momentum. Releases from a settled box (the oracle's cadence) match the
env step for step.
"""

from __future__ import annotations

from typing import Dict, List

from predicators.code_sim_learning.commands import CommandBuffer
from predicators.code_sim_learning.fit_space import ParamSpec
from predicators.code_sim_learning.utils import Params, ResidualUpdate, \
    objs_by_type
from predicators.envs.pybullet_balloons_base import PyBulletBalloonsBaseEnv
from predicators.ground_truth_models import GroundTruthSimulatorFactory
from predicators.settings import CFG
from predicators.structs import State

# ── Geometry shared with the visible base sim ─────────────────────

_GEOM = PyBulletBalloonsBaseEnv
# A balloon whose top comes within this of the plate's underside bursts
# (the env's own margin).
_POP_MARGIN = 0.002


def _lift_param_name(color_index: int) -> str:
    return f"lift_{_GEOM.balloon_color_name(color_index)}"


def _release_and_pull(state: State, updates: ResidualUpdate, params: Params,
                      cmds: CommandBuffer) -> ResidualUpdate:
    """Seat freed balloons on the box, burst those at the ceiling, and pull the
    box up with every intact tied balloon.

    Mirrors ``PyBulletBalloonsEnv._domain_specific_step`` on the same
    post-step cadence: the state is the one the base sim just produced,
    the feature updates land in the observation this step, and the
    commands act during the next action's substeps.
    """
    objs = objs_by_type(state)
    boxes = objs.get("box", [])
    if not boxes:
        return updates
    box = boxes[0]
    balloons = sorted(objs.get("balloon", []), key=lambda o: o.name)
    clips = sorted(objs.get("clip", []), key=lambda o: o.name)
    box_x, box_y = state.get(box, "x"), state.get(box, "y")
    box_z = float(state.get(box, "z"))
    box_top_z = box_z + _GEOM.box_half
    fade = float(params["fade_height"])
    frac = (box_z - _GEOM.table_height) / fade if fade > 0 else 1.0
    fade_factor = max(0.0, 1.0 - frac)
    ceiling_underside = _GEOM.ceiling_z - _GEOM.ceiling_half_extents[2]

    stacked = sum(1 for b in balloons if state.get(b, "tied") > 0.5)
    for index, balloon in enumerate(balloons):
        tied = state.get(balloon, "tied") > 0.5
        popped = state.get(balloon, "popped") > 0.5
        if not tied:
            if index >= len(clips) or state.get(clips[index], "is_on") <= 0.5:
                continue
            # Freed: the string pulls the balloon to the end of its
            # tether above the box's top centre, above any balloon freed
            # before it, so its pull acts through the box.
            tied = True
            seat_z = (box_top_z + _GEOM.balloon_radius + _GEOM.string_length +
                      2 * _GEOM.balloon_radius * stacked)
            stacked += 1
            updates.setdefault(balloon, {}).update({
                "tied": 1.0,
                "x": float(box_x),
                "y": float(box_y),
                "z": float(seat_z),
            })
        # The pop reads the balloon where the base sim left it this
        # step (a just-freed balloon is still in the rack here), as the
        # env does.
        z_balloon = float(state.get(balloon, "z"))
        if not popped and z_balloon + _GEOM.balloon_radius >= \
                ceiling_underside - _POP_MARGIN:
            popped = True
            updates.setdefault(balloon, {})["popped"] = 1.0
        cmds.attach(balloon, box)
        if not popped:
            color = int(round(state.get(balloon, "color")))
            lift = float(params[_lift_param_name(color)]) * fade_factor
            # The pull acts at the balloon, above the box, and the
            # string carries it down: pulled from above, the box hangs
            # upright; pushed from below it would tip over.
            cmds.apply_force(balloon, (0.0, 0.0, lift))
    return updates


# ── Public API: consumed by read_simulator_components ────────────
# Same contract used by agent-synthesized simulator files.

RESIDUAL_RULES = [_release_and_pull]


def _build_param_specs() -> List[ParamSpec]:
    """Per-colour lifts at table height (N) and the shared fade height (m),
    read from CFG at consumption time so the module imports before the config
    is final."""
    specs = [
        ParamSpec(_lift_param_name(index),
                  float(CFG.balloons_lifts[index]),
                  lo=0.0,
                  hi=3.0) for index in range(len(_GEOM.BALLOON_PALETTE))
    ]
    specs.append(
        ParamSpec("fade_height",
                  float(CFG.balloons_fade_height),
                  lo=0.1,
                  hi=2.0))
    return specs


PARAM_SPECS = _build_param_specs

# Features the hidden physics own. The balloon's pose and flags are set
# by the release and pop updates; the box's rise is the engine's answer
# to the commanded pull, scored by the rollout objective against
# observations and never overwritten at plan time.
RESIDUAL_FEATURES: Dict[str, List[str]] = {
    "balloon": ["x", "y", "z", "tied", "popped"],
    "box": ["x", "y", "z", "speed"],
}

# ── Factory binding ──────────────────────────────────────────────


class PyBulletBalloonsGroundTruthSimulatorFactory(GroundTruthSimulatorFactory):
    """GT residual-dynamics simulator for pybullet_balloons.

    The simulator components (``RESIDUAL_RULES``, ``PARAM_SPECS``,
    ``RESIDUAL_FEATURES``) live as module globals above; this class only
    pins the env-name binding so ``get_gt_simulator`` can locate the
    module via the factory registry.
    """

    @classmethod
    def get_env_names(cls) -> set:
        return {"pybullet_balloons"}
