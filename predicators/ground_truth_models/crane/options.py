"""Ground-truth options for the crane environment.

Two skills from the shared factories: draw the ram back along its arc
and let it go (``Pull``), and wait. ``Pull`` is the shared
``create_push_skill`` aimed at the head's rest pose, facing down the
lane, with one extra parameter, the PULL the stroke carries the head
back past its rest position. The stroke follows the chord of the arc
(the head rises as it is drawn back), the gripper then retreats, and the
ram swings.
"""

from dataclasses import replace
from typing import ClassVar, Dict, Optional, Sequence, Set, Tuple
from typing import Type as TypingType

import numpy as np
from gym.spaces import Box

from predicators.envs.pybullet_crane import PyBulletCraneEnv
from predicators.ground_truth_models import GroundTruthOptionFactory
from predicators.ground_truth_models.skill_factories import SkillConfig, \
    create_push_skill, create_wait_option, shared_skill_robot, \
    shared_skill_simulator
from predicators.settings import CFG
from predicators.structs import Array, Object, ParameterizedOption, \
    Predicate, State, Type

# The ram is drawn back along -x: the skills' facing (sin(yaw),
# cos(yaw)) is (-1, 0) at this yaw.
_WEST = -np.pi / 2

# Distance between the end effector's reference point and the head's
# centre while the fingers press on it (the fingertip pads plus the
# head's half width): the stroke aims this much short of the requested
# pull so the head ends up drawn back by about the pull itself.
_HEAD_CONTACT_OFFSET = 0.117
# Metres per env action of the pulling stroke (about 0.12 m/s): the
# head is driven, not struck, so it ends up where the stroke ends.
_PULL_STEP_NORM = 0.01
# The gripper meets the head this far below its centre: the head's
# face tilts as the arm swings back, and a low contact keeps the
# fingers on it.
_CONTACT_BELOW_CENTRE = 0.01


def _pull_param() -> Tuple[str, float, float]:
    lo, hi = CFG.crane_pull_range
    return ("pull (metres the ram's head is drawn back along the lane from "
            "its rest position before the gripper lets go)", float(lo),
            float(hi))


def _stroke_overshoot(params: Array) -> float:
    return float(params[2]) - _HEAD_CONTACT_OFFSET


def _stroke_rise(state: State, objects: Sequence[Object],
                 params: Array) -> float:
    """How much the head rises when drawn back by the pull on an arm of the
    crane's length: the stroke ends that much higher."""
    _, _, crane = objects
    length = float(state.get(crane, "length"))
    pull = min(float(params[2]), length - 1e-3)
    return length - float(np.sqrt(length**2 - pull**2))


def _head_rest_pose(
    state: State,
    objects: Sequence[Object],
    params: Array,
    cfg: SkillConfig,
) -> Tuple[float, float, float, float]:
    del params, cfg
    _, ram, _ = objects
    return (state.get(ram, "x"), state.get(ram, "y"),
            state.get(ram, "z") - _CONTACT_BELOW_CENTRE, _WEST)


class PyBulletCraneGroundTruthOptionFactory(GroundTruthOptionFactory):
    """Ground-truth options for the crane environment."""

    env_cls: ClassVar[TypingType[PyBulletCraneEnv]] = PyBulletCraneEnv
    _transport_z: ClassVar[float] = env_cls.z_ub - 0.25

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"pybullet_crane"}

    @classmethod
    def skill_config(cls,
                     simulator: Optional[PyBulletCraneEnv]) -> SkillConfig:
        """The crane's skill configuration."""
        pybullet_robot = shared_skill_robot(PyBulletCraneEnv)
        env_cls = cls.env_cls
        _fingers_state_to_joint = PyBulletCraneEnv._fingers_state_to_joint  # pylint: disable=protected-access
        return SkillConfig(
            robot=pybullet_robot,
            open_fingers_joint=pybullet_robot.open_fingers,
            closed_fingers_joint=pybullet_robot.closed_fingers,
            fingers_state_to_joint=_fingers_state_to_joint,
            robot_init_tilt=PyBulletCraneEnv.robot_init_tilt,
            robot_init_wrist=PyBulletCraneEnv.robot_init_wrist,
            robot_home_pos=(env_cls.robot_init_x, env_cls.robot_init_y,
                            env_cls.robot_init_z),
            transport_z=cls._transport_z,
            simulator=simulator,
            # The stroke's end IS the pull: 5 mm of tolerance so the
            # pull parameter means what it says.
            move_to_pose_tol=2.5e-5,
            # A Wait ends once the ram and the crate have settled.
            wait_quiescence_eps=1e-4,
            wait_quiescence_steps=10,
            # Fingers first at the head: the hand is narrow along the
            # lane that way, clearing the crate behind the head.
            extra={"push_ee_yaw_offset": 0.0},
        )

    @classmethod
    def pull_option(cls, types: Dict[str, Type], config: SkillConfig,
                    plan_transit: Optional[bool]) -> ParameterizedOption:
        """The Pull skill on a given configuration."""
        return create_push_skill(
            name="Pull",
            types=[types["robot"], types["ram"], types["crane"]],
            config=replace(config, transport_z=cls._transport_z),
            get_target_pose_fn=_head_rest_pose,
            # The head moves with the stroke; aim once at its rest pose.
            freeze_stroke=True,
            extra_params=[_pull_param()],
            stroke_overshoot_fn=_stroke_overshoot,
            stroke_step_norm_fn=lambda params: _PULL_STEP_NORM,
            stroke_rise_fn=_stroke_rise,
            plan_transit=plan_transit,
            # The open fingertips bracket the head's face.
            open_hand=True,
            # Let go upward: the head swings forward the moment the
            # hand leaves it, and a hand sweeping across its path would
            # be struck.
            lift_before_retreat=True,
        )

    @classmethod
    def get_options(cls, env_name: str, types: Dict[str, Type],
                    predicates: Dict[str, Predicate],
                    action_space: Box) -> Set[ParameterizedOption]:
        del env_name, predicates, action_space  # unused
        simulator = shared_skill_simulator(cls.env_cls) \
            if CFG.skill_phase_use_motion_planning else None
        config = cls.skill_config(simulator)
        Pull = cls.pull_option(types, config, plan_transit=None)
        Wait = create_wait_option("Wait", config, types["robot"])
        return {Pull, Wait}


_PROBE_PULL: Optional[ParameterizedOption] = None


def probe_pull_option() -> ParameterizedOption:
    """The Pull skill for a probe rollout: no motion planning anywhere."""
    global _PROBE_PULL  # pylint: disable=global-statement
    if _PROBE_PULL is None:
        # pylint: disable=protected-access
        types = {
            "robot": PyBulletCraneEnv._robot_type,
            "ram": PyBulletCraneEnv._ram_type,
            "crane": PyBulletCraneEnv._crane_type,
        }
        # pylint: enable=protected-access
        factory = PyBulletCraneGroundTruthOptionFactory
        _PROBE_PULL = factory.pull_option(types,
                                          factory.skill_config(None),
                                          plan_transit=False)
    return _PROBE_PULL
