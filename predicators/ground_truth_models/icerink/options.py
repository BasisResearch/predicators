"""Ground-truth options for the ice rink environment.

Two skills, both built from the shared factories: push a tile along a
named direction at a chosen speed, and wait for the rink to settle.
Nothing is picked up. The push is the shared ``create_push_skill`` with
the direction object's ``yaw`` as the facing, so the robot approaches
from behind the tile relative to that direction and strikes its side
face, plus one extra parameter, the stroke SPEED: the gripper moves
through the stroke at that many metres per second and the tile leaves at
about that speed. How far it then slides is the domain's question.
"""

from dataclasses import replace
from typing import ClassVar, Dict, Optional, Sequence, Set, Tuple
from typing import Type as TypingType

from gym.spaces import Box

from predicators.envs.pybullet_icerink import PyBulletIceRinkEnv
from predicators.ground_truth_models import GroundTruthOptionFactory
from predicators.ground_truth_models.skill_factories import SkillConfig, \
    create_push_skill, create_wait_option, shared_skill_robot, \
    shared_skill_simulator
from predicators.settings import CFG
from predicators.structs import Array, Object, ParameterizedOption, \
    Predicate, State, Type

# PyBullet's fixed physics step; one env action is
# ``CFG.pybullet_sim_steps_per_action`` of them.
_PHYSICS_DT = 1.0 / 240.0


def action_dt() -> float:
    """Seconds of simulated time per env action."""
    return float(CFG.pybullet_sim_steps_per_action) * _PHYSICS_DT


def _speed_param() -> Tuple[str, float, float]:
    lo, hi = CFG.icerink_push_speed_range
    return ("speed (m/s the gripper moves at through the stroke; the tile "
            "leaves at about this speed and slides until friction, a wall or "
            "another tile stops it)", float(lo), float(hi))


def _stroke_step_norm(params: Array) -> float:
    """Metres per env action for the push stroke: the speed parameter."""
    return float(params[2]) * action_dt()


def _tile_pose(
    state: State,
    objects: Sequence[Object],
    params: Array,
    cfg: SkillConfig,
) -> Tuple[float, float, float, float]:
    del params, cfg
    _, tile, direction = objects
    return (state.get(tile, "x"), state.get(tile, "y"),
            state.get(tile, "z") + PyBulletIceRinkEnv.tile_push_height,
            state.get(direction, "yaw"))


class PyBulletIceRinkGroundTruthOptionFactory(GroundTruthOptionFactory):
    """Ground-truth options for the ice rink environment."""

    env_cls: ClassVar[TypingType[PyBulletIceRinkEnv]] = PyBulletIceRinkEnv
    # Tiles and walls are low; the transit height only has to clear them.
    _transport_z: ClassVar[float] = env_cls.z_ub - 0.3

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"pybullet_icerink"}

    @classmethod
    def skill_config(cls,
                     simulator: Optional[PyBulletIceRinkEnv]) -> SkillConfig:
        """The rink's skill configuration, with or without a motion-planning
        simulator."""
        pybullet_robot = shared_skill_robot(PyBulletIceRinkEnv)
        env_cls = cls.env_cls
        _fingers_state_to_joint = PyBulletIceRinkEnv._fingers_state_to_joint  # pylint: disable=protected-access
        return SkillConfig(
            robot=pybullet_robot,
            open_fingers_joint=pybullet_robot.open_fingers,
            closed_fingers_joint=pybullet_robot.closed_fingers,
            fingers_state_to_joint=_fingers_state_to_joint,
            robot_init_tilt=PyBulletIceRinkEnv.robot_init_tilt,
            robot_init_wrist=PyBulletIceRinkEnv.robot_init_wrist,
            robot_home_pos=(env_cls.robot_init_x, env_cls.robot_init_y,
                            env_cls.robot_init_z),
            transport_z=cls._transport_z,
            simulator=simulator,
            # A Wait ends once the tiles have stopped sliding.
            wait_quiescence_eps=1e-4,
            wait_quiescence_steps=10,
        )

    @classmethod
    def push_option(cls, types: Dict[str, Type], config: SkillConfig,
                    plan_transit: Optional[bool]) -> ParameterizedOption:
        """The Push skill on a given configuration."""
        return create_push_skill(
            name="Push",
            types=[types["robot"], types["tile"], types["direction"]],
            config=replace(config, transport_z=cls._transport_z),
            get_target_pose_fn=_tile_pose,
            # The tile slides away on contact; the stroke must not chase it.
            freeze_stroke=True,
            extra_params=[_speed_param()],
            stroke_step_norm_fn=_stroke_step_norm,
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
        Push = cls.push_option(types, config, plan_transit=None)
        Wait = create_wait_option("Wait", config, types["robot"])
        return {Push, Wait}


_PROBE_PUSH: Optional[ParameterizedOption] = None


def probe_push_option() -> ParameterizedOption:
    """The Push skill for a probe rollout: no motion planning anywhere, so a
    probe on a scratch simulator costs the stroke and nothing else."""
    global _PROBE_PUSH  # pylint: disable=global-statement
    if _PROBE_PUSH is None:
        # pylint: disable=protected-access
        types = {
            "robot": PyBulletIceRinkEnv._robot_type,
            "tile": PyBulletIceRinkEnv._tile_type,
            "direction": PyBulletIceRinkEnv._direction_type,
        }
        # pylint: enable=protected-access
        factory = PyBulletIceRinkGroundTruthOptionFactory
        _PROBE_PUSH = factory.push_option(types,
                                          factory.skill_config(None),
                                          plan_transit=False)
    return _PROBE_PUSH
