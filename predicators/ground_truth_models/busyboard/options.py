"""Ground-truth options for the busyboard environment.

Three skills, all built from the shared factories: push a button on,
push it off, and wait. Nothing is picked up and nothing is placed,
which is deliberate - the board is a reasoning problem wearing a robot,
so the manipulation repertoire is kept to the single primitive the
board needs and every skill here is an off-the-shelf
``create_push_skill`` / ``create_wait_option`` instantiation with no
busyboard-specific motion logic.

The push target follows the boil env's switch convention exactly: a
button's ``rot`` is its facing, the on-stroke approaches from
``rot + pi/2`` and the off-stroke from ``rot - pi/2``. Because a button
IS the pushed object here (rather than a switch body owned by some
other object, as in boil), the target-pose callback reads the button's
own pose and needs no lookup.
"""

from dataclasses import replace
from typing import ClassVar, Dict, Sequence, Set, Tuple
from typing import Type as TypingType

import numpy as np
from gym.spaces import Box

from predicators.envs.pybullet_busyboard import PyBulletBusyBoardEnv
from predicators.ground_truth_models import GroundTruthOptionFactory
from predicators.ground_truth_models.skill_factories import SkillConfig, \
    create_push_skill, create_wait_option, shared_skill_robot, \
    shared_skill_simulator
from predicators.settings import CFG
from predicators.structs import Array, Object, ParameterizedOption, \
    Predicate, State, Type


class PyBulletBusyBoardGroundTruthOptionFactory(GroundTruthOptionFactory):
    """Ground-truth options for the busyboard environment."""

    env_cls: ClassVar[TypingType[PyBulletBusyBoardEnv]] = PyBulletBusyBoardEnv
    # Buttons are low and the board is otherwise clear, so the transport
    # height only has to clear the switch bodies themselves.
    _transport_z: ClassVar[float] = env_cls.z_ub - 0.3

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"pybullet_busyboard"}

    @classmethod
    def get_options(cls, env_name: str, types: Dict[str, Type],
                    predicates: Dict[str, Predicate],
                    action_space: Box) -> Set[ParameterizedOption]:
        del env_name, predicates, action_space  # unused

        pybullet_robot = shared_skill_robot(PyBulletBusyBoardEnv)
        robot_type = types["robot"]
        button_type = types["button"]
        env_cls = cls.env_cls

        simulator = shared_skill_simulator(env_cls) \
            if CFG.skill_phase_use_motion_planning else None
        # Bound once here rather than inline: the SkillConfig field is a
        # continuation line after formatting, where a disable comment for
        # the protected access would not be read.
        _fingers_state_to_joint = PyBulletBusyBoardEnv._fingers_state_to_joint  # pylint: disable=protected-access
        config = SkillConfig(
            robot=pybullet_robot,
            open_fingers_joint=pybullet_robot.open_fingers,
            closed_fingers_joint=pybullet_robot.closed_fingers,
            fingers_state_to_joint=_fingers_state_to_joint,
            robot_init_tilt=PyBulletBusyBoardEnv.robot_init_tilt,
            robot_init_wrist=PyBulletBusyBoardEnv.robot_init_wrist,
            robot_home_pos=(env_cls.robot_init_x, env_cls.robot_init_y,
                            env_cls.robot_init_z),
            transport_z=cls._transport_z,
            simulator=simulator,
        )

        def _button_pose(
            state: State,
            objects: Sequence[Object],
            params: Array,
            cfg: SkillConfig,
        ) -> Tuple[float, float, float, float]:
            del params, cfg
            _, button = objects
            # Aim at the slider, not the switch base (see
            # ``button_press_height``): the stroke has to cross the
            # slider's travel, and a base-height stroke stalls against the
            # switch body.
            return (state.get(button, "x"), state.get(button, "y"),
                    state.get(button, "z") + env_cls.button_press_height,
                    state.get(button, "rot"))

        # A switch's push_dir is (cos(rot), sin(rot)) while the skills'
        # standard facing is (sin(yaw), cos(yaw)), so the on-stroke yaw is
        # rot + pi/2 and the off-stroke is the same stroke reversed.
        def _on_pose(
            state: State,
            objects: Sequence[Object],
            params: Array,
            cfg: SkillConfig,
        ) -> Tuple[float, float, float, float]:
            x, y, z, rot = _button_pose(state, objects, params, cfg)
            return x, y, z, rot + np.pi / 2

        def _off_pose(
            state: State,
            objects: Sequence[Object],
            params: Array,
            cfg: SkillConfig,
        ) -> Tuple[float, float, float, float]:
            x, y, z, rot = _button_pose(state, objects, params, cfg)
            return x, y, z, rot - np.pi / 2

        push_config = replace(config, transport_z=cls._transport_z)

        PressButton = create_push_skill(
            name="PressButton",
            types=[robot_type, button_type],
            config=push_config,
            get_target_pose_fn=_on_pose,
        )
        ReleaseButton = create_push_skill(
            name="ReleaseButton",
            types=[robot_type, button_type],
            config=push_config,
            get_target_pose_fn=_off_pose,
        )
        # Wait is not garnish here: a lamp's charge only builds while the
        # board is left alone in a driving configuration, so holding still
        # is a first-class action and the only way to finish most tasks.
        Wait = create_wait_option("Wait", config, robot_type)

        return {PressButton, ReleaseButton, Wait}
