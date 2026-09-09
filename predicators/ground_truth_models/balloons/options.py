"""Ground-truth options for the balloons environment.

Two skills from the shared factories: push a balloon's clip open
(``Release``, the shared push skill on the clip's toggle, the boil and
busyboard convention) and wait for the box to settle. Nothing is picked
up.
"""

from typing import ClassVar, Dict, Optional, Sequence, Set, Tuple
from typing import Type as TypingType

import numpy as np
from gym.spaces import Box

from predicators.envs.pybullet_balloons import PyBulletBalloonsEnv
from predicators.ground_truth_models import GroundTruthOptionFactory
from predicators.ground_truth_models.skill_factories import SkillConfig, \
    create_push_skill, create_wait_option, shared_skill_robot, \
    shared_skill_simulator
from predicators.settings import CFG
from predicators.structs import Array, Object, ParameterizedOption, \
    Predicate, State, Type


def _clip_on_pose(
    state: State,
    objects: Sequence[Object],
    params: Array,
    cfg: SkillConfig,
) -> Tuple[float, float, float, float]:
    """The slider's pose and the on-stroke facing (rot + pi/2)."""
    del params, cfg
    _, clip = objects
    return (state.get(clip, "x"), state.get(clip, "y"),
            state.get(clip, "z") + PyBulletBalloonsEnv.clip_press_height,
            state.get(clip, "rot") + np.pi / 2)


class PyBulletBalloonsGroundTruthOptionFactory(GroundTruthOptionFactory):
    """Ground-truth options for the balloons environment."""

    env_cls: ClassVar[TypingType[PyBulletBalloonsEnv]] = PyBulletBalloonsEnv
    _transport_z: ClassVar[float] = env_cls.z_ub - 0.3

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"pybullet_balloons"}

    @classmethod
    def skill_config(cls,
                     simulator: Optional[PyBulletBalloonsEnv]) -> SkillConfig:
        """The balloon table's skill configuration."""
        pybullet_robot = shared_skill_robot(PyBulletBalloonsEnv)
        env_cls = cls.env_cls
        _fingers_state_to_joint = PyBulletBalloonsEnv._fingers_state_to_joint  # pylint: disable=protected-access
        return SkillConfig(
            robot=pybullet_robot,
            open_fingers_joint=pybullet_robot.open_fingers,
            closed_fingers_joint=pybullet_robot.closed_fingers,
            fingers_state_to_joint=_fingers_state_to_joint,
            robot_init_tilt=PyBulletBalloonsEnv.robot_init_tilt,
            robot_init_wrist=PyBulletBalloonsEnv.robot_init_wrist,
            robot_home_pos=(env_cls.robot_init_x, env_cls.robot_init_y,
                            env_cls.robot_init_z),
            transport_z=cls._transport_z,
            simulator=simulator,
            wait_quiescence_eps=1e-4,
            wait_quiescence_steps=10,
        )

    @classmethod
    def release_option(cls, types: Dict[str, Type], config: SkillConfig,
                       plan_transit: Optional[bool]) -> ParameterizedOption:
        """The Release skill on a given configuration."""
        return create_push_skill(
            name="Release",
            types=[types["robot"], types["clip"]],
            config=config,
            get_target_pose_fn=_clip_on_pose,
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
        release = cls.release_option(types, config, plan_transit=None)
        wait = create_wait_option("Wait", config, types["robot"])
        return {release, wait}


_PROBE_RELEASE: Optional[ParameterizedOption] = None


def probe_release_option() -> ParameterizedOption:
    """The Release skill for probe rollouts: no motion planning."""
    global _PROBE_RELEASE  # pylint: disable=global-statement
    if _PROBE_RELEASE is None:
        # pylint: disable=protected-access
        types = {
            "robot": PyBulletBalloonsEnv._robot_type,
            "clip": PyBulletBalloonsEnv._clip_type,
        }
        # pylint: enable=protected-access
        factory = PyBulletBalloonsGroundTruthOptionFactory
        _PROBE_RELEASE = factory.release_option(types,
                                                factory.skill_config(None),
                                                plan_transit=False)
    return _PROBE_RELEASE


def release_params() -> Array:
    """The push parameters that open a clip."""
    return np.array([CFG.balloons_push_approach, CFG.balloons_push_contact_z],
                    dtype=np.float32)
