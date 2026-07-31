"""Ground-truth options for the real-bench fan environment.

The bench has exactly one thing the robot can do, and this module builds
it: ``BlowBallToZone`` -- descend onto the button, hold it down for a
chosen time, lift off, let the ball settle.

The button is momentary, so holding it down IS running the fan. That
makes the hold duration the only decision in the domain, and it is the
option's only continuous parameter -- literally the picture's "one
action, six possible targets".
"""

from typing import Any, ClassVar, Dict, List, Optional, Sequence, Set, Tuple
from typing import Type as TypingType

import numpy as np
from gym.spaces import Box

from predicators.envs.pybullet_fan_real import PyBulletFanRealEnv
from predicators.ground_truth_models import GroundTruthOptionFactory
from predicators.ground_truth_models.skill_factories import Phase, \
    PhaseAction, PhaseSkill, SkillConfig, build_params_space, \
    create_timed_hold_option, create_wait_option, make_move_to_phase, \
    shared_skill_robot, shared_skill_simulator
from predicators.settings import CFG
from predicators.structs import Action, Array, Object, ParameterizedOption, \
    Predicate, State, Type

# The bench's whole decision space.
_BLOW_PARAMS = [
    ("burst_seconds (how long to hold the button down; the fan runs for "
     "exactly this long, and a longer burst leaves the ball in a farther "
     "zone)", 0.0, CFG.fan_real_max_burst_seconds),
]


class _ParamSlicedChain(ParameterizedOption):
    """Run child options in order, each fed its own slice of the parameters.

    ``utils.LinearChainParameterizedOption`` requires every child to
    share one params space, which is the wrong shape here: the presses
    are parameterized by push geometry and the hold by a duration.  This
    chain instead maps the parent's parameter vector to each child's,
    and hands every child the same object tuple (each child reads only
    the prefix it needs -- the push skills and the hold both take the
    robot as ``objects[0]``).

    Children are assumed to chain: the next child's ``initiable`` must
    hold when the previous one terminates.
    """

    def __init__(
        self,
        name: str,
        types: Sequence[Type],
        params_space: Box,
        children: Sequence[Tuple[ParameterizedOption, Any]],
        params_description: Optional[Tuple[str, ...]] = None,
    ) -> None:
        assert children
        self._children = [c for c, _ in children]
        self._child_params = [f for _, f in children]
        super().__init__(name,
                         list(types),
                         params_space,
                         policy=self._policy,
                         initiable=self._initiable,
                         terminal=self._terminal,
                         params_description=params_description)

    def _initiable(self, state: State, memory: Dict, objects: Sequence[Object],
                   params: Array) -> bool:
        memory["child_idx"] = 0
        memory["child_memory"] = [{} for _ in self._children]
        return self._children[0].initiable(state, memory["child_memory"][0],
                                           objects,
                                           self._child_params[0](params))

    def _policy(self, state: State, memory: Dict, objects: Sequence[Object],
                params: Array) -> Action:
        idx = memory["child_idx"]
        child = self._children[idx]
        child_mem = memory["child_memory"][idx]
        if child.terminal(state, child_mem, objects,
                          self._child_params[idx](params)):
            idx += 1
            memory["child_idx"] = idx
            child = self._children[idx]
            child_mem = memory["child_memory"][idx]
            assert child.initiable(state, child_mem, objects,
                                   self._child_params[idx](params))
        return child.policy(state, child_mem, objects,
                            self._child_params[idx](params))

    def _terminal(self, state: State, memory: Dict, objects: Sequence[Object],
                  params: Array) -> bool:
        idx = memory["child_idx"]
        if idx < len(self._children) - 1:
            return False
        return self._children[idx].terminal(state, memory["child_memory"][idx],
                                            objects,
                                            self._child_params[idx](params))


class PyBulletFanRealGroundTruthOptionFactory(GroundTruthOptionFactory):
    """Ground-truth options for the real-bench fan environment."""

    env_cls: ClassVar[TypingType[PyBulletFanRealEnv]] = PyBulletFanRealEnv

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"pybullet_fan_real"}

    @classmethod
    def get_options(cls, env_name: str, types: Dict[str, Type],
                    predicates: Dict[str, Predicate],
                    action_space: Box) -> Set[ParameterizedOption]:
        del env_name, predicates, action_space  # unused

        env_cls = cls.env_cls
        robot_type = types["robot"]
        fan_type = types["fan"]
        zone_type = types["zone"]
        button_type = types["button"]

        pybullet_robot = shared_skill_robot(env_cls)
        simulator = shared_skill_simulator(env_cls) \
            if CFG.skill_phase_use_motion_planning else None
        config = SkillConfig(
            robot=pybullet_robot,
            open_fingers_joint=pybullet_robot.open_fingers,
            closed_fingers_joint=pybullet_robot.closed_fingers,
            fingers_state_to_joint=env_cls._fingers_state_to_joint,  # pylint: disable=protected-access
            robot_init_tilt=env_cls.robot_init_tilt,
            robot_init_wrist=env_cls.robot_init_wrist,
            robot_home_pos=(env_cls.robot_init_x, env_cls.robot_init_y,
                            env_cls.robot_init_z),
            transport_z=env_cls.z_ub - 0.3,
            simulator=simulator,
            # The settle phase ends when the ball stops, not on a fixed
            # timer: how long a ball coasts is precisely what varies.
            #
            # The eps must stay COARSER than ``State.allclose``'s atol of
            # 1e-3, and the count must be 1. The option model aborts any
            # rollout whose consecutive states are allclose ("got stuck"),
            # and it checks that AFTER the option's own terminal -- so a
            # finer eps (1e-4, which is what pybullet_domino uses) means
            # the coasting ball trips the stuck detector several steps
            # before quiescence ever converges, and refinement rejects
            # every sample. Domino escapes this only because its Wait is a
            # top-level option literally named "Wait", which the option
            # model terminates early on atom change; a chain's child gets
            # no such exemption.
            #
            # Stopping at 3 mm/step leaves the ball ~1.4 cm of coast (about
            # a seventh of a zone). That is a constant offset, absorbed
            # exactly by the oracle's fitted intercept, because
            # scripts/fan_real_debug/sweep_burst.py calibrates through this
            # same option and therefore the same stopping rule.
            wait_quiescence_eps=3e-3,
            wait_quiescence_steps=1,
        )

        # The button is not among the option's objects -- the option is
        # written in terms of the fan it powers, the way the picture labels
        # it -- so it is looked up by type. There is exactly one.
        del button_type  # the press point is fixed bench geometry

        press_x, press_y, press_z = env_cls.button_press_point()

        def _press_pose(state: State, objects: Sequence[Object], params: Array,
                        cfg: SkillConfig) -> Tuple[float, float, float, float]:
            del state, objects, params, cfg
            return press_x, press_y, press_z, 0.0

        def _standoff_pose(
                state: State, objects: Sequence[Object], params: Array,
                cfg: SkillConfig) -> Tuple[float, float, float, float]:
            """Directly above the button, clear of it.

            Both the approach and the release go through here, so the
            gripper drops straight down onto the button and lifts
            straight off it. Anything with lateral travel at press
            height would sweep the fan on and off again on the way past.
            """
            del state, objects, params
            return press_x, press_y, cfg.transport_z, 0.0

        def _close_fingers(state: State, objects: Sequence[Object],
                           params: Array,
                           cfg: SkillConfig) -> Tuple[float, float]:
            del params
            current = cfg.fingers_state_to_joint(
                cfg.robot, state.get(objects[0], "fingers"))
            return current, cfg.closed_fingers_joint - 0.01

        def _open_fingers(state: State, objects: Sequence[Object],
                          params: Array,
                          cfg: SkillConfig) -> Tuple[float, float]:
            del params
            current = cfg.fingers_state_to_joint(
                cfg.robot, state.get(objects[0], "fingers"))
            return current, cfg.open_fingers_joint

        opt_types: List[Type] = [robot_type, fan_type, zone_type]
        params_space, params_description = build_params_space(_BLOW_PARAMS)
        _empty_space = Box(0, 1, (0, ))

        # Closing the fingers FIRST is not just so the gripper pokes the
        # button with a closed fist. The option model's stuck detector aborts
        # a rollout whose first action leaves the state unchanged, and a
        # motion phase's first BiRRT waypoint IS the current configuration --
        # a literal no-op. Every skill factory in this package opens with a
        # CHANGE_FINGERS phase for the same reason; a chain that starts with
        # a bare move is refused by refinement before it ever moves.
        press = PhaseSkill(
            "_PressButton",
            opt_types,
            _empty_space,
            config,
            [
                Phase(name="CloseFingers",
                      action_type=PhaseAction.CHANGE_FINGERS,
                      target_fn=_close_fingers,
                      finger_direction="close"),
                make_move_to_phase(name="MoveAboveButton",
                                   get_target_pose_fn=_standoff_pose,
                                   finger_status="closed"),
                make_move_to_phase(name="Descend",
                                   get_target_pose_fn=_press_pose,
                                   finger_status="closed"),
            ],
        ).build()
        # Holding the arm's joint targets holds the button down, which is
        # what runs the fan -- so this is the burst.
        hold = create_timed_hold_option(
            name="_HoldButton",
            config=config,
            types=opt_types,
            action_dt=env_cls._action_dt(),  # pylint: disable=protected-access
            max_seconds=CFG.fan_real_max_burst_seconds,
            param_description="burst_seconds (button hold time)")
        release = PhaseSkill(
            "_ReleaseButton",
            opt_types,
            _empty_space,
            config,
            [
                make_move_to_phase(name="Retreat",
                                   get_target_pose_fn=_standoff_pose,
                                   finger_status="closed"),
                Phase(name="OpenFingers",
                      action_type=PhaseAction.CHANGE_FINGERS,
                      target_fn=_open_fingers,
                      finger_direction="open"),
            ],
        ).build()
        settle = create_wait_option("_Settle", config, robot_type)

        _burst = lambda params: np.asarray(params[0:1], dtype=np.float64)
        _none = lambda params: np.array([], dtype=np.float64)

        BlowBallToZone = _ParamSlicedChain(
            "BlowBallToZone",
            opt_types,
            params_space,
            children=[
                (press, _none),
                (hold, _burst),
                (release, _none),
                (settle, _none),
            ],
            params_description=params_description)

        return {BlowBallToZone}
