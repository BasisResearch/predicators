"""Ground-truth balloons model in the SUBCLASS form.

The rule form of this model lives in ``gt_simulator.py``: residual rules
that run on top of a fixed base-sim env through the command channel.
This file is the same hidden physics written the OTHER way the contract
allows: a subclass of the visible base-sim env that overrides
``_domain_specific_step`` with full engine access, declares its
learnable constants in ``AGENT_PARAM_SPECS`` (read inside the step via
``self.agent_param``) and the features it owns in ``RESIDUAL_FEATURES``.

The harness runs an instance of ``RESIDUAL_ENV`` (with
``skip_residual_dynamics=False``, so the step below fires) as the
planning base env, and the rollout system-ID fits the AGENT_PARAM_SPECS
as physical parameters (``PyBulletEnv.get_physical_param_info`` surfaces
them). Unlike the rule form, the step here reaches PyBullet directly:
it sets the box and balloon masses and drag with ``changeDynamics`` and
seats a freed balloon with ``resetBasePositionAndOrientation``, none of
which the fixed command vocabulary exposes.

This module exists to prove the subclass form can express a domain the
rule form already covers; the sibling ``gt_simulator.py`` remains the
env's registered GT simulator.
"""

from __future__ import annotations

from typing import Any, ClassVar, List, Set

import pybullet as p

from predicators.code_sim_learning.commands import ApplyForce, Attach
from predicators.code_sim_learning.fit_space import ParamSpec
from predicators.envs.pybullet_balloons_base import PyBulletBalloonsBaseEnv
from predicators.structs import EnvironmentTask, Predicate, Type

# A balloon whose top comes within this of the plate's underside bursts.
_POP_MARGIN = 0.002


def _lift_param_name(color_index: int) -> str:
    return f"lift_{PyBulletBalloonsBaseEnv.balloon_color_name(color_index)}"


def _mass_param_name(color_index: int) -> str:
    return f"mass_{PyBulletBalloonsBaseEnv.box_color_name(color_index)}"


class BalloonsResidualEnv(PyBulletBalloonsBaseEnv):
    """The balloons hidden physics as a base-sim subclass.

    Release, pop and fading lift, plus the box masses and air drag, all
    driven by ``AGENT_PARAM_SPECS`` read live in the step.
    """

    AGENT_PARAM_SPECS: ClassVar[List[Any]] = [
        ParamSpec(_lift_param_name(0), 0.4, lo=0.0, hi=3.0),  # red
        ParamSpec(_lift_param_name(1), 0.4, lo=0.0, hi=3.0),  # blue
        ParamSpec(_lift_param_name(2), 0.6, lo=0.0, hi=3.0),  # green
        ParamSpec(_lift_param_name(3), 0.9, lo=0.0, hi=3.0),  # gold
        ParamSpec("fade_height", 0.7, lo=0.1, hi=2.0),
        ParamSpec(_mass_param_name(0), 0.06, lo=0.01, hi=1.0, scale="log"),
        ParamSpec(_mass_param_name(1), 0.09, lo=0.01, hi=1.0, scale="log"),
        ParamSpec("air_drag", 0.1, lo=0.01, hi=40.0, scale="log"),
    ]

    RESIDUAL_FEATURES: ClassVar[dict] = {
        "balloon": ["x", "y", "z", "tied", "popped"],
        "box": ["x", "y", "z", "speed"],
    }

    # ── Harness-only members ─────────────────────────────────────
    # A subclass model supplies the physics; the harness supplies the
    # tasks and predicates from the real env, so these are trivial stubs
    # that only make the class instantiable. The distinct get_name keeps
    # it out of the real env's registry slot.

    @classmethod
    def get_name(cls) -> str:
        return "pybullet_balloons_residual_model"

    @property
    def predicates(self) -> Set[Predicate]:
        return set()

    @property
    def goal_predicates(self) -> Set[Predicate]:
        return set()

    @property
    def types(self) -> Set[Type]:
        return {
            self._robot_type, self._box_type, self._balloon_type,
            self._clip_type, self._band_type
        }

    def _generate_train_tasks(self) -> List[EnvironmentTask]:
        return []

    def _generate_test_tasks(self) -> List[EnvironmentTask]:
        return []

    def _apply_material_params(self) -> None:
        """Push the box mass and the drag into PyBullet (idempotent, so it
        survives the mass reset a backtracking ``_set_state`` does)."""
        drag = self.agent_param("air_drag")
        mass = self.agent_param(_mass_param_name(self._box_color))
        p.changeDynamics(self._box.id,
                         -1,
                         mass=float(mass),
                         linearDamping=float(drag),
                         physicsClientId=self._physics_client_id)
        for balloon in self._active_balloons(self._get_state()):
            p.changeDynamics(balloon.id,
                             -1,
                             linearDamping=float(drag),
                             physicsClientId=self._physics_client_id)

    def _lift_at(self, color_index: int, box_z: float) -> float:
        fade = self.agent_param("fade_height")
        frac = (box_z - self.table_height) / fade if fade > 0 else 1.0
        return self.agent_param(_lift_param_name(color_index)) * max(
            0.0, 1.0 - frac)

    def _domain_specific_step(self) -> None:
        """Release freed balloons onto the box, burst those at the ceiling, and
        pull the box up with every intact freed balloon.

        The engine, not this code, integrates the box's rise under the
        commanded pull, the string's drag and the box's weight.
        """
        self._apply_material_params()
        state = self._get_state()
        box_top = self.box_top_point(state, self._box)
        box_z = float(state.get(self._box, "z"))
        balloons = self._active_balloons(state)
        clips = self._active_clips(state)
        commands: List[Any] = []
        stacked = sum(1 for b in balloons if self._tied.get(b.name, False))
        ceiling_underside = self.ceiling_z - self.ceiling_half_extents[2]
        for index, balloon in enumerate(balloons):
            name = balloon.name
            if not self._tied.get(name, False):
                if index >= len(clips) or not self._is_clip_on(clips[index]):
                    continue
                self._tied[name] = True
                seat = (box_top[0], box_top[1],
                        box_top[2] + self.balloon_radius + self.string_length +
                        2 * self.balloon_radius * stacked)
                stacked += 1
                p.resetBasePositionAndOrientation(
                    balloon.id,
                    seat, (0.0, 0.0, 0.0, 1.0),
                    physicsClientId=self._physics_client_id)
                p.resetBaseVelocity(balloon.id, (0.0, 0.0, 0.0),
                                    (0.0, 0.0, 0.0),
                                    physicsClientId=self._physics_client_id)
            z_balloon = float(state.get(balloon, "z"))
            if not self._popped.get(name, False) and \
                    z_balloon + self.balloon_radius >= \
                    ceiling_underside - _POP_MARGIN:
                self._popped[name] = True
                self._paint_balloon(balloon)
            commands.append(Attach(name, self._box.name))
            if not self._popped.get(name, False):
                lift = self._lift_at(self._balloon_colors[name], box_z)
                commands.append(ApplyForce(name, (0.0, 0.0, lift)))
        if commands:
            self.queue_residual_commands(commands)


RESIDUAL_ENV = BalloonsResidualEnv
RESIDUAL_FEATURES = BalloonsResidualEnv.RESIDUAL_FEATURES
