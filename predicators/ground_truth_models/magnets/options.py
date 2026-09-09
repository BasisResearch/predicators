"""Ground-truth options for the magnets environment.

Three skills. ``Hover(robot, wand)[x, y]`` slides the wand's tip to a
point over the mat slowly, in a straight line at hover height, so a
piece the wand pulls keeps up and follows. ``Jump(robot, wand)[x, y]``
goes to a point at the arm's normal speed, which is faster than any
piece can follow: it is how the wand reaches a piece, and how it leaves
one behind. ``Wait`` lets the pieces settle. Nothing is picked up or put
down; the wand is in the hand from the start.
"""

from typing import ClassVar, Dict, Optional, Sequence, Set, Tuple
from typing import Type as TypingType

from gym.spaces import Box

from predicators.envs.pybullet_magnets import PyBulletMagnetsEnv
from predicators.ground_truth_models import GroundTruthOptionFactory
from predicators.ground_truth_models.skill_factories import PhaseSkill, \
    SkillConfig, build_params_space, create_wait_option, make_move_to_phase, \
    shared_skill_robot, shared_skill_simulator
from predicators.settings import CFG
from predicators.structs import Array, Object, ParameterizedOption, \
    Predicate, State, Type

# Metres per env action of a Hover: about 0.1 m/s, well under the
# speed a pulled piece can reach.
_HOVER_STEP_NORM = 0.008


def _target_params(
) -> Tuple[Tuple[str, float, float], Tuple[str, float, float]]:
    x_min, x_max, y_min, y_max = PyBulletMagnetsEnv.mat_bounds()
    return (("x (where to put the wand's tip, metres)", x_min, x_max),
            ("y (where to put the wand's tip, metres)", y_min, y_max))


def _tip_target(
    state: State,
    objects: Sequence[Object],
    params: Array,
    cfg: SkillConfig,
) -> Tuple[float, float, float, float]:
    """The end effector pose that puts the tip at the requested point."""
    del state, objects
    env_cls = PyBulletMagnetsEnv
    return (float(params[0]), float(params[1]),
            env_cls.hover_z + env_cls.wand_length, cfg.robot_init_wrist)


class PyBulletMagnetsGroundTruthOptionFactory(GroundTruthOptionFactory):
    """Ground-truth options for the magnets environment."""

    env_cls: ClassVar[TypingType[PyBulletMagnetsEnv]] = PyBulletMagnetsEnv

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"pybullet_magnets"}

    @classmethod
    def skill_config(cls,
                     simulator: Optional[PyBulletMagnetsEnv]) -> SkillConfig:
        """The mat's skill configuration."""
        pybullet_robot = shared_skill_robot(PyBulletMagnetsEnv)
        env_cls = cls.env_cls
        _fingers_state_to_joint = PyBulletMagnetsEnv._fingers_state_to_joint  # pylint: disable=protected-access
        return SkillConfig(
            robot=pybullet_robot,
            open_fingers_joint=pybullet_robot.open_fingers,
            closed_fingers_joint=pybullet_robot.closed_fingers,
            fingers_state_to_joint=_fingers_state_to_joint,
            robot_init_tilt=PyBulletMagnetsEnv.robot_init_tilt,
            robot_init_wrist=PyBulletMagnetsEnv.robot_init_wrist,
            robot_home_pos=(env_cls.robot_init_x, env_cls.robot_init_y,
                            env_cls.robot_init_z),
            transport_z=env_cls.robot_init_z,
            simulator=simulator,
            # A hover ends within 5 mm of its point: where the tip stops
            # is where the carried piece settles.
            move_to_pose_tol=2.5e-5,
            wait_quiescence_eps=1e-4,
            wait_quiescence_steps=10,
        )

    @classmethod
    def move_options(
        cls, types: Dict[str, Type], config: SkillConfig,
        plan_transit: Optional[bool]
    ) -> Tuple[ParameterizedOption, ParameterizedOption]:
        """``(Hover, Jump)`` on a given configuration."""
        params_space, description = build_params_space(_target_params())
        skill_types = [types["robot"], types["wand"]]
        hover = PhaseSkill(
            "Hover",
            skill_types,
            params_space,
            config, [
                make_move_to_phase("Slide",
                                   _tip_target,
                                   finger_status="closed",
                                   use_motion_planning=False,
                                   max_step_norm=_HOVER_STEP_NORM)
            ],
            params_description=description).build()
        jump = PhaseSkill(
            "Jump",
            skill_types,
            params_space,
            config, [
                make_move_to_phase("Move",
                                   _tip_target,
                                   finger_status="closed",
                                   use_motion_planning=plan_transit)
            ],
            params_description=description).build()
        return hover, jump

    @classmethod
    def get_options(cls, env_name: str, types: Dict[str, Type],
                    predicates: Dict[str, Predicate],
                    action_space: Box) -> Set[ParameterizedOption]:
        del env_name, predicates, action_space  # unused
        simulator = shared_skill_simulator(cls.env_cls) \
            if CFG.skill_phase_use_motion_planning else None
        config = cls.skill_config(simulator)
        hover, jump = cls.move_options(types, config, plan_transit=None)
        wait = create_wait_option("Wait", config, types["robot"])
        return {hover, jump, wait}


_PROBE_OPTIONS: Optional[Tuple[ParameterizedOption,
                               ParameterizedOption]] = None


def probe_move_options() -> Tuple[ParameterizedOption, ParameterizedOption]:
    """``(Hover, Jump)`` for probe rollouts: no motion planning."""
    global _PROBE_OPTIONS  # pylint: disable=global-statement
    if _PROBE_OPTIONS is None:
        # pylint: disable=protected-access
        types = {
            "robot": PyBulletMagnetsEnv._robot_type,
            "wand": PyBulletMagnetsEnv._wand_type,
        }
        # pylint: enable=protected-access
        factory = PyBulletMagnetsGroundTruthOptionFactory
        _PROBE_OPTIONS = factory.move_options(types,
                                              factory.skill_config(None),
                                              plan_transit=False)
    return _PROBE_OPTIONS
