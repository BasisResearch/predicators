"""Ground-truth options for the launcher environment.

Two skills from the shared factories: cock the launcher by pushing its
handle back a chosen depth, and wait. ``Cock`` is the shared
``create_push_skill`` aimed at the handle's rest pose, facing west, with
one extra parameter, the DEPTH the stroke carries the handle past its
rest position. The gripper then retreats, the handle snaps home, and
what the snap does to the ball is the environment's secret.
"""

from dataclasses import replace
from typing import ClassVar, Dict, Optional, Sequence, Set, Tuple
from typing import Type as TypingType

import numpy as np
from gym.spaces import Box

from predicators.envs.pybullet_launcher import PyBulletLauncherEnv
from predicators.ground_truth_models import GroundTruthOptionFactory
from predicators.ground_truth_models.skill_factories import SkillConfig, \
    create_push_skill, create_wait_option, shared_skill_robot, \
    shared_skill_simulator
from predicators.settings import CFG
from predicators.structs import Array, Object, ParameterizedOption, \
    Predicate, State, Type

# The handle is pushed west (-x): the skills' facing (sin(yaw), cos(yaw))
# is (-1, 0) at this yaw.
_WEST = -np.pi / 2

# Distance between the end effector's reference point and the centre of
# the handle it pushes (the fingertip pads plus the handle's own half
# thickness), measured on this rail: the stroke aims this much short of
# the requested depth so the handle ends up compressed by about the
# depth itself, which is the ``compression`` the observation reports.
_HANDLE_CONTACT_OFFSET = 0.026
# Metres per env action of the cocking stroke (about 0.12 m/s): slow
# enough that the stroke's last step lands within the tolerance.
_COCK_STEP_NORM = 0.01


def _depth_param() -> Tuple[str, float, float]:
    return ("depth (metres the handle is pushed back past its rest position "
            "before the gripper lets go; the launcher's compression)", 0.0,
            float(PyBulletLauncherEnv.max_push_depth))


def _stroke_overshoot(params: Array) -> float:
    return float(params[2]) - _HANDLE_CONTACT_OFFSET


def _handle_pose(
    state: State,
    objects: Sequence[Object],
    params: Array,
    cfg: SkillConfig,
) -> Tuple[float, float, float, float]:
    del params, cfg
    _, launcher = objects
    return (state.get(launcher, "x"), state.get(launcher, "y"),
            state.get(launcher, "z") + PyBulletLauncherEnv.handle_push_height,
            _WEST)


class PyBulletLauncherGroundTruthOptionFactory(GroundTruthOptionFactory):
    """Ground-truth options for the launcher environment."""

    env_cls: ClassVar[TypingType[PyBulletLauncherEnv]] = PyBulletLauncherEnv
    _transport_z: ClassVar[float] = env_cls.z_ub - 0.25

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"pybullet_launcher"}

    @classmethod
    def skill_config(cls,
                     simulator: Optional[PyBulletLauncherEnv]) -> SkillConfig:
        """The launcher's skill configuration."""
        pybullet_robot = shared_skill_robot(PyBulletLauncherEnv)
        env_cls = cls.env_cls
        _fingers_state_to_joint = PyBulletLauncherEnv._fingers_state_to_joint  # pylint: disable=protected-access
        return SkillConfig(
            robot=pybullet_robot,
            open_fingers_joint=pybullet_robot.open_fingers,
            closed_fingers_joint=pybullet_robot.closed_fingers,
            fingers_state_to_joint=_fingers_state_to_joint,
            robot_init_tilt=PyBulletLauncherEnv.robot_init_tilt,
            robot_init_wrist=PyBulletLauncherEnv.robot_init_wrist,
            robot_home_pos=(env_cls.robot_init_x, env_cls.robot_init_y,
                            env_cls.robot_init_z),
            transport_z=cls._transport_z,
            simulator=simulator,
            # The stroke's end IS the compression: 5 mm of tolerance so
            # the depth parameter means what it says.
            move_to_pose_tol=2.5e-5,
            # A Wait ends once the ball and the blocks have settled.
            wait_quiescence_eps=1e-4,
            wait_quiescence_steps=10,
            # Fingers first at the handle: knuckles-first would put the
            # hand's whole width along the rail, reaching the ball.
            extra={"push_ee_yaw_offset": 0.0},
        )

    @classmethod
    def cock_option(cls, types: Dict[str, Type], config: SkillConfig,
                    plan_transit: Optional[bool]) -> ParameterizedOption:
        """The Cock skill on a given configuration."""
        return create_push_skill(
            name="Cock",
            types=[types["robot"], types["launcher"]],
            config=replace(config, transport_z=cls._transport_z),
            get_target_pose_fn=_handle_pose,
            # The handle moves with the stroke; aim once at its rest pose.
            freeze_stroke=True,
            extra_params=[_depth_param()],
            stroke_overshoot_fn=_stroke_overshoot,
            # A gentle stroke: the handle is driven, not struck, so it
            # ends up where the stroke ends rather than coasting past.
            stroke_step_norm_fn=lambda params: _COCK_STEP_NORM,
            plan_transit=plan_transit,
        )

    @classmethod
    def get_options(cls, env_name: str, types: Dict[str, Type],
                    predicates: Dict[str, Predicate],
                    action_space: Box) -> Set[ParameterizedOption]:
        del env_name, predicates, action_space  # unused
        simulator = shared_skill_simulator(cls.env_cls) \
            if CFG.skill_phase_use_motion_planning else None
        config = cls.skill_config(simulator)
        Cock = cls.cock_option(types, config, plan_transit=None)
        Wait = create_wait_option("Wait", config, types["robot"])
        return {Cock, Wait}


_PROBE_COCK: Optional[ParameterizedOption] = None


def probe_cock_option() -> ParameterizedOption:
    """The Cock skill for a probe rollout: no motion planning anywhere."""
    global _PROBE_COCK  # pylint: disable=global-statement
    if _PROBE_COCK is None:
        # pylint: disable=protected-access
        types = {
            "robot": PyBulletLauncherEnv._robot_type,
            "launcher": PyBulletLauncherEnv._launcher_type,
        }
        # pylint: enable=protected-access
        factory = PyBulletLauncherGroundTruthOptionFactory
        _PROBE_COCK = factory.cock_option(types,
                                          factory.skill_config(None),
                                          plan_transit=False)
    return _PROBE_COCK
