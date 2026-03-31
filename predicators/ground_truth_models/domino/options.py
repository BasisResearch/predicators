"""Ground-truth options for the coffee environment."""

from dataclasses import replace
from typing import ClassVar, Dict, Sequence, Set, Tuple
from typing import Type as TypingType

from gym.spaces import Box

from predicators.envs.pybullet_domino import PyBulletDominoEnv
from predicators.envs.pybullet_env import PyBulletEnv
from predicators.ground_truth_models import GroundTruthOptionFactory
from predicators.ground_truth_models.skill_factories import SkillConfig, \
    create_pick_skill, create_place_skill, create_push_skill, \
    create_wait_option
from predicators.pybullet_helpers.robots import SingleArmPyBulletRobot
from predicators.settings import CFG
from predicators.structs import Array, Object, ParameterizedOption, \
    Predicate, State, Type

from .options_legacy import _DominoLegacyOptionsMixin


class PyBulletDominoGroundTruthOptionFactory(_DominoLegacyOptionsMixin,
                                             GroundTruthOptionFactory):
    """Ground-truth options for the grow environment."""

    env_cls: ClassVar[TypingType[PyBulletDominoEnv]] = PyBulletDominoEnv
    _move_to_pose_tol: ClassVar[float] = 1e-4
    _finger_action_nudge_magnitude: ClassVar[float] = 1e-3
    _transport_z: ClassVar[float] = env_cls.table_height +\
            env_cls.domino_height * 2.26
    _transport_z_push: ClassVar[float] = env_cls.table_height +\
            env_cls.domino_height * 1.5
    _offset_x: ClassVar[float] = env_cls.domino_depth * 3
    _offset_z: ClassVar[float] = env_cls.domino_height * 0.55
    _place_drop_z: ClassVar[float] = env_cls.table_height +\
            env_cls.domino_height * 1.13

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"pybullet_domino_grid", "pybullet_domino"}

    @classmethod
    def get_options(cls, env_name: str, types: Dict[str, Type],
                    predicates: Dict[str, Predicate],
                    action_space: Box) -> Set[ParameterizedOption]:
        """Get the ground-truth options for the domino environment."""
        if CFG.domino_use_skill_factories:
            return cls._get_options_skill_factories(env_name, types,
                                                    predicates, action_space)
        return cls._get_options_legacy(env_name, types, predicates,
                                       action_space)

    # ------------------------------------------------------------------
    # Skill-factories-based implementation
    # ------------------------------------------------------------------

    @classmethod
    def _get_options_skill_factories(
            cls, env_name: str, types: Dict[str,
                                            Type], predicates: Dict[str,
                                                                    Predicate],
            action_space: Box) -> Set[ParameterizedOption]:
        """Option implementation built on skill_factories primitives."""
        del env_name, predicates, action_space  # unused

        _, pybullet_robot, _ = \
            PyBulletDominoEnv.initialize_pybullet(using_gui=False)

        robot_type = types["robot"]
        domino_type = types["domino"]

        cfg = cls._build_skill_config(pybullet_robot)

        options: Set[ParameterizedOption] = set()

        if CFG.domino_restricted_push:
            options.add(
                cls._create_sf_push_restricted(cfg, robot_type, domino_type))
        else:
            options.add(cls._create_sf_push(cfg, robot_type, domino_type))

        options.add(cls._create_sf_pick(cfg, robot_type, domino_type))
        options.add(cls._create_sf_place(cfg, robot_type))
        options.add(create_wait_option("Wait", cfg, robot_type))

        return options

    @classmethod
    def _build_skill_config(
            cls, pybullet_robot: SingleArmPyBulletRobot) -> SkillConfig:
        """Build the shared SkillConfig for domino skill_factories options."""
        simulator = cls.env_cls(use_gui=False) \
            if CFG.skill_phase_use_motion_planning else None
        return SkillConfig(
            robot=pybullet_robot,
            open_fingers_joint=pybullet_robot.open_fingers,
            closed_fingers_joint=pybullet_robot.closed_fingers,
            fingers_state_to_joint=PyBulletDominoEnv._fingers_state_to_joint,  # pylint: disable=protected-access
            move_to_pose_tol=cls._move_to_pose_tol,
            finger_action_nudge_magnitude=cls._finger_action_nudge_magnitude,
            max_vel_norm=CFG.pybullet_max_vel_norm,
            grasp_tol=PyBulletEnv.grasp_tol_small,
            ik_validate=CFG.pybullet_ik_validate,
            robot_init_tilt=cls.env_cls.robot_init_tilt,
            robot_home_pos=(cls.env_cls.robot_init_x, cls.env_cls.robot_init_y,
                            cls.env_cls.robot_init_z),
            transport_z=cls._transport_z,
            simulator=simulator,
        )

    @classmethod
    def _create_sf_push(cls, cfg: SkillConfig, robot_type: Type,
                        domino_type: Type) -> ParameterizedOption:
        """Push option using create_push_skill."""
        push_cfg = replace(cfg, transport_z=cls._transport_z_push)

        def _get_target(
                state: State, objects: Sequence[Object], params: Array,
                config: SkillConfig) -> Tuple[float, float, float, float]:
            del params, config
            _, domino = objects
            return (state.get(domino, "x"), state.get(domino, "y"),
                    state.get(domino, "z"), state.get(domino, "yaw"))

        return create_push_skill(name="Push",
                                 types=[robot_type, domino_type],
                                 config=push_cfg,
                                 get_target_pose_fn=_get_target)

    @classmethod
    def _create_sf_push_restricted(cls, cfg: SkillConfig, robot_type: Type,
                                   domino_type: Type) -> ParameterizedOption:
        """Push (restricted) option: finds start block from state."""
        push_cfg = replace(cfg, transport_z=cls._transport_z_push)

        def _get_target(
                state: State, objects: Sequence[Object], params: Array,
                config: SkillConfig) -> Tuple[float, float, float, float]:
            del objects, params, config
            start = cls._find_start_block(state, domino_type)
            return (state.get(start, "x"), state.get(start, "y"),
                    state.get(start, "z"), state.get(start, "yaw"))

        return create_push_skill(name="Push",
                                 types=[robot_type],
                                 config=push_cfg,
                                 get_target_pose_fn=_get_target)

    @classmethod
    def _create_sf_pick(cls, cfg: SkillConfig, robot_type: Type,
                        domino_type: Type) -> ParameterizedOption:
        """Pick option using create_pick_skill."""

        def _get_domino_pose(
                state: State, objects: Sequence[Object], params: Array,
                c: SkillConfig) -> Tuple[float, float, float, float]:
            del params, c
            _, domino = objects
            return (state.get(domino, "x"), state.get(domino, "y"),
                    state.get(domino, "z"), state.get(domino, "yaw"))

        return create_pick_skill(
            name="Pick",
            types=[robot_type, domino_type],
            config=cfg,
            get_target_pose_fn=_get_domino_pose,
        )

    @classmethod
    def _create_sf_place(cls, cfg: SkillConfig,
                         robot_type: Type) -> ParameterizedOption:
        """Place option using create_place_skill."""
        return create_place_skill(
            name="Place",
            types=[robot_type],
            config=cfg,
        )
