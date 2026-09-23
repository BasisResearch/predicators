"""Tests for the domain-general primitive skill library
(predicators/ground_truth_models/skill_factories/primitives.py)."""

import numpy as np
import pybullet as p
import pytest

from predicators import utils
from predicators.ground_truth_models.skill_factories import \
    PRIMITIVE_SKILL_NAMES, SkillConfig, check_skill_library, \
    create_gripper_skill, create_move_linear_skill, \
    create_move_to_pose_skill, create_move_until_contact_skill, \
    create_primitive_skills, primitive_skills_for_env, primitives
from predicators.ground_truth_models.skill_factories.base import Phase, \
    PhaseAction
from predicators.pybullet_helpers.geometry import Pose
from predicators.pybullet_helpers.objects import create_pybullet_block
from predicators.pybullet_helpers.robots import \
    create_single_arm_pybullet_robot
from predicators.structs import Object, ParameterizedOption, Type

_ROBOT_TYPE = Type("robot", ["x", "y", "z", "tilt", "wrist", "fingers"])
_BLOCK_TYPE = Type("block", ["x", "y", "z", "is_held"])
_OPEN_STATE = 0.04
_CLOSED_STATE = 0.01
_EE_HOME = (1.35, 0.75, 0.75)
_WORKSPACE = ((0.4, 1.1), (1.1, 1.6), (0.4, 0.9))


@pytest.fixture(scope="module", name="robot_scene")
def _setup_robot_scene():
    utils.reset_config({"seed": 123})
    physics_client_id = p.connect(p.DIRECT)
    home = Pose(_EE_HOME, p.getQuaternionFromEuler([0.0, np.pi / 2, -np.pi]))
    robot = create_single_arm_pybullet_robot("fetch", physics_client_id, home)
    yield physics_client_id, robot
    p.disconnect(physics_client_id)


def _fingers_state_to_joint(robot, finger_state: float) -> float:
    t = (finger_state - _CLOSED_STATE) / (_OPEN_STATE - _CLOSED_STATE)
    return robot.closed_fingers + t * (robot.open_fingers -
                                       robot.closed_fingers)


def _make_config(robot) -> SkillConfig:
    return SkillConfig(
        robot=robot,
        open_fingers_joint=robot.open_fingers,
        closed_fingers_joint=robot.closed_fingers,
        fingers_state_to_joint=_fingers_state_to_joint,
    )


def _make_state(robot,
                robot_obj,
                ee_xyz,
                finger_state,
                sim_state=None,
                objs=()):
    joints = list(robot.get_joints())
    fj = _fingers_state_to_joint(robot, finger_state)
    joints[robot.left_finger_joint_idx] = fj
    joints[robot.right_finger_joint_idx] = fj
    data = {
        robot_obj:
        np.array([*ee_xyz, np.pi / 2, -np.pi, finger_state], dtype=np.float32)
    }
    for obj, feats in objs:
        data[obj] = np.array(feats, dtype=np.float32)
    if sim_state is None:
        sim_state = {"joint_positions": joints}
    else:
        sim_state = dict(sim_state, joint_positions=joints)
    return utils.PyBulletState(data, simulator_state=sim_state)


def _phases_of(opt):
    return opt.policy.__self__._phases  # pylint: disable=protected-access


class _Env:
    """A stand-in env class with workspace and finger bounds."""
    x_lb, x_ub = 0.4, 1.1
    y_lb, y_ub = 1.1, 1.6
    z_lb, z_ub = 0.4, 0.9
    open_fingers = _OPEN_STATE
    closed_fingers = _CLOSED_STATE


def test_skill_library_values():
    """Only the two library names are accepted."""
    assert check_skill_library("composite") == "composite"
    assert check_skill_library("primitive") == "primitive"
    with pytest.raises(ValueError, match="Unknown skill_library"):
        check_skill_library("bogus")


def test_library_contents(robot_scene):
    """The library is the five named skills, each taking only the robot."""
    _, robot = robot_scene
    opts = create_primitive_skills(_make_config(robot), _ROBOT_TYPE,
                                   _WORKSPACE, (_CLOSED_STATE, _OPEN_STATE))
    assert sorted(o.name for o in opts) == sorted(PRIMITIVE_SKILL_NAMES)
    for opt in opts:
        assert isinstance(opt, ParameterizedOption)
        assert [t.name for t in opt.types] == ["robot"]
        assert opt.params_description is not None
        assert len(opt.params_description) == opt.params_space.shape[0]
    by_name = {o.name: o for o in opts}
    assert by_name["MoveTo"].params_space.shape == (5, )
    assert by_name["MoveLinear"].params_space.shape == (4, )
    assert by_name["MoveUntilContact"].params_space.shape == (4, )
    assert by_name["Gripper"].params_space.shape == (2, )
    assert by_name["Wait"].params_space.shape == (1, )


def test_library_for_env_reads_bounds(robot_scene):
    """The env-sized library takes its boxes from the env class."""
    _, robot = robot_scene
    opts = {
        o.name: o
        for o in primitive_skills_for_env(_Env, _make_config(robot),
                                          _ROBOT_TYPE)
    }
    move_to = opts["MoveTo"]
    assert np.allclose(move_to.params_space.low[:3], [0.4, 1.1, 0.4])
    assert np.allclose(move_to.params_space.high[:3], [1.1, 1.6, 0.9])
    gripper = opts["Gripper"]
    assert np.allclose(gripper.params_space.low, [_CLOSED_STATE, 0.0])
    assert np.allclose(gripper.params_space.high, [_OPEN_STATE, 70.0])

    class _NoBounds:  # pylint: disable=too-few-public-methods
        open_fingers = _OPEN_STATE
        closed_fingers = _CLOSED_STATE

    with pytest.raises(ValueError, match="workspace bounds"):
        primitive_skills_for_env(_NoBounds, _make_config(robot), _ROBOT_TYPE)


def test_move_to_targets_full_pose_and_holds_fingers(robot_scene):
    """MoveTo aims the end effector at (x, y, z) with the requested yaw and
    tilt, keeps the current finger width, and validates its goal IK."""
    _, robot = robot_scene
    opt = create_move_to_pose_skill("MoveTo", _ROBOT_TYPE, _make_config(robot),
                                    _WORKSPACE)
    phases = _phases_of(opt)
    assert len(phases) == 1
    phase = phases[0]
    assert phase.action_type == PhaseAction.MOVE_TO_POSE
    assert phase.validate_ik
    robot_obj = Object("robot0", _ROBOT_TYPE)
    state = _make_state(robot, robot_obj, _EE_HOME, 0.025)
    params = np.array([0.8, 1.3, 0.6, 0.4, 1.2], dtype=np.float32)
    current, target, fingers = phase.target_fn(state, [robot_obj], params,
                                               _make_config(robot))
    assert np.allclose(current.position, _EE_HOME)
    assert np.allclose(target.position, [0.8, 1.3, 0.6])
    assert np.allclose(p.getEulerFromQuaternion(target.orientation),
                       [0.0, 1.2, 0.4],
                       atol=1e-6)
    assert fingers == "hold"


def test_stroke_target_is_a_frozen_world_displacement(robot_scene):
    """A stroke aims at start pose + (dx, dy, dz), keeps the orientation, and
    freezes that target so it does not chase the moving hand."""
    _, robot = robot_scene
    opt = create_move_linear_skill("MoveLinear", _ROBOT_TYPE,
                                   _make_config(robot))
    phase = _phases_of(opt)[0]
    assert phase.freeze_target
    assert phase.expect_contact
    assert not phase.use_motion_planning
    params = np.array([0.05, -0.02, 0.0, 0.01], dtype=np.float32)
    assert phase.step_norm_fn is not None
    assert np.isclose(phase.step_norm_fn(params), 0.01)
    robot_obj = Object("robot0", _ROBOT_TYPE)
    state = _make_state(robot, robot_obj, _EE_HOME, _OPEN_STATE)
    current, target, fingers = phase.target_fn(state, [robot_obj], params,
                                               _make_config(robot))
    assert np.allclose(target.position, np.add(_EE_HOME, [0.05, -0.02, 0.0]))
    assert np.allclose(target.orientation, current.orientation)
    assert fingers == "hold"
    # Frozen through the skill's own target resolution: the second call
    # from a displaced hand returns the same target.
    skill = opt.policy.__self__  # pylint: disable=protected-access
    memory = {}
    _, first, _ = skill._phase_targets(  # pylint: disable=protected-access
        phase, state, memory, [robot_obj], params)
    moved = _make_state(robot, robot_obj, np.add(_EE_HOME, [0.02, 0, 0]),
                        _OPEN_STATE)
    _, second, _ = skill._phase_targets(  # pylint: disable=protected-access
        phase, moved, memory, [robot_obj], params)
    assert np.allclose(first.position, second.position)


def test_move_until_contact_ends_on_hand_contact(robot_scene):
    """The guarded stroke is terminal as soon as a finger touches a body the
    robot is not holding; a plain MoveLinear in the same state is not."""
    client, robot = robot_scene
    config = _make_config(robot)
    guarded = create_move_until_contact_skill("MoveUntilContact", _ROBOT_TYPE,
                                              config)
    plain = create_move_linear_skill("MoveLinear", _ROBOT_TYPE, config)
    robot_obj = Object("robot0", _ROBOT_TYPE)
    block_obj = Object("block0", _BLOCK_TYPE)
    sim_state = {"physics_client_id": client, "robot_id": robot.robot_id}
    params = np.array([0.0, 0.0, -0.05, 0.005], dtype=np.float32)
    # Nothing near the hand: neither stroke is terminal at its start.
    p.performCollisionDetection(physicsClientId=client)
    free = _make_state(robot, robot_obj, _EE_HOME, _OPEN_STATE, sim_state)
    for opt in (guarded, plain):
        grounded = opt.ground([robot_obj], params)
        assert grounded.initiable(free)
        assert not grounded.terminal(free)
    # A block overlapping the left finger pad.
    finger_pos = p.getLinkState(robot.robot_id,
                                robot.left_finger_id,
                                physicsClientId=client)[0]
    block_id = create_pybullet_block((0.5, 0.5, 0.5, 1.0), (0.02, 0.02, 0.02),
                                     1.0,
                                     1.0,
                                     position=finger_pos,
                                     physics_client_id=client)
    p.performCollisionDetection(physicsClientId=client)
    touching = _make_state(robot,
                           robot_obj,
                           _EE_HOME,
                           _OPEN_STATE,
                           sim_state,
                           objs=[(block_obj, [*finger_pos, 0.0])])
    assert primitives.stroke_in_contact(touching, config)
    for opt, expected in ((guarded, True), (plain, False)):
        grounded = opt.ground([robot_obj], params)
        assert grounded.initiable(touching)
        assert grounded.terminal(touching) is expected
    # The same body, held: it is the payload, not an obstacle.
    block_obj.id = block_id  # type: ignore[attr-defined]
    held = _make_state(robot,
                       robot_obj,
                       _EE_HOME,
                       _OPEN_STATE,
                       sim_state,
                       objs=[(block_obj, [*finger_pos, 1.0])])
    assert not primitives.stroke_in_contact(held, config)
    p.removeBody(block_id, physicsClientId=client)


def test_gripper_reaches_width_or_stalls(robot_scene):
    """The gripper commands the fingers toward the width, ends within the grasp
    tolerance of it, and otherwise ends after a run of steps in which the
    fingers do not move (closed on an object)."""
    _, robot = robot_scene
    config = _make_config(robot)
    opt = create_gripper_skill("Gripper", _ROBOT_TYPE, config,
                               (_CLOSED_STATE, _OPEN_STATE))
    robot_obj = Object("robot0", _ROBOT_TYPE)
    grounded = opt.ground([robot_obj], np.array([_CLOSED_STATE, 20.0]))
    open_state = _make_state(robot, robot_obj, _EE_HOME, _OPEN_STATE)
    assert grounded.initiable(open_state)
    assert not grounded.terminal(open_state)
    action = grounded.policy(open_state)
    fingers_cmd = action.arr[robot.left_finger_joint_idx]
    assert fingers_cmd < _fingers_state_to_joint(robot, _OPEN_STATE)
    assert action.arr[robot.left_finger_joint_idx] == \
        action.arr[robot.right_finger_joint_idx]
    # Reached the width.
    closed_state = _make_state(robot, robot_obj, _EE_HOME, _CLOSED_STATE)
    assert grounded.terminal(closed_state)
    # Stalled: the fingers stop at an object's width, short of the target.
    stuck = _make_state(robot, robot_obj, _EE_HOME, 0.03)
    grounded = opt.ground([robot_obj], np.array([_CLOSED_STATE, 20.0]))
    assert grounded.initiable(stuck)
    for _ in range(primitives._GRIPPER_STALL_STEPS):  # pylint: disable=protected-access
        assert not grounded.terminal(stuck)
        grounded.policy(stuck)
    assert not grounded.terminal(stuck)
    grounded.policy(stuck)
    assert grounded.terminal(stuck)
    # A fresh grounding starts its stall count over.
    grounded = opt.ground([robot_obj], np.array([_CLOSED_STATE, 20.0]))
    assert grounded.initiable(stuck)
    assert not grounded.terminal(stuck)


def test_stroke_phase_is_a_gentle_final_phase(robot_scene):
    """Both strokes are one gentle-stroke phase, so the executor's joint- jump
    guard and stall abort apply and a blocked stroke fails naming the contact
    instead of looping to the cap."""
    _, robot = robot_scene
    config = _make_config(robot)
    for make in (create_move_linear_skill, create_move_until_contact_skill):
        opt = make("S", _ROBOT_TYPE, config)
        phases = _phases_of(opt)
        assert len(phases) == 1
        assert isinstance(phases[0], Phase)
        assert phases[0].step_norm_fn is not None
