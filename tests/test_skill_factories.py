"""Tests for predicators/ground_truth_models/skill_factories/.

Covers: SkillConfig, Phase, PhaseSkill, create_wait_option,
        make_move_to_phase, create_move_to_skill,
        create_pick_skill, create_place_skill, create_push_skill.
"""
import numpy as np
import pybullet as p
import pytest
from gym.spaces import Box

from predicators import utils
from predicators.ground_truth_models.skill_factories.base import \
    _BIRRT_STEP_KEY, _BIRRT_TRAJ_KEY, Phase, PhaseAction, PhaseSkill, \
    SkillConfig
from predicators.ground_truth_models.skill_factories.move_to import \
    create_move_to_skill, make_move_to_phase
from predicators.ground_truth_models.skill_factories.pick import \
    create_pick_skill
from predicators.ground_truth_models.skill_factories.place import \
    create_place_skill
from predicators.ground_truth_models.skill_factories.push import \
    create_push_skill
from predicators.ground_truth_models.skill_factories.wait import \
    create_wait_option
from predicators.pybullet_helpers.geometry import Pose
from predicators.pybullet_helpers.robots import \
    create_single_arm_pybullet_robot
from predicators.structs import Action, Object, ParameterizedOption, Type

# ---------------------------------------------------------------------------
# Type definitions reused across tests
# ---------------------------------------------------------------------------
_ROBOT_TYPE = Type("robot", ["x", "y", "z", "tilt", "wrist", "fingers"])
_OBJ_TYPE = Type("obj", ["x", "y", "z"])

# Finger state values matching PyBulletEnv class-var conventions for Fetch.
_OPEN_STATE = 0.04  # open_fingers feature value
_CLOSED_STATE = 0.01  # closed_fingers feature value

# EE pose used to home the Fetch robot.
_EE_HOME = (1.35, 0.75, 0.75)
_EE_HOME_ORN = None  # computed on first use


def _get_ee_home_pose() -> Pose:
    orn = p.getQuaternionFromEuler([0.0, np.pi / 2, -np.pi])
    return Pose(_EE_HOME, orn)


# ---------------------------------------------------------------------------
# Module-scoped fixture: create a Fetch robot exactly once for all tests.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module", name="robot_scene")
def _setup_robot_scene():
    """Connect to PyBullet DIRECT, create a Fetch robot, yield both."""
    utils.reset_config({"seed": 123})
    physics_client_id = p.connect(p.DIRECT)
    robot = create_single_arm_pybullet_robot("fetch", physics_client_id,
                                             _get_ee_home_pose())
    yield physics_client_id, robot
    p.disconnect(physics_client_id)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fingers_state_to_joint(robot, finger_state: float) -> float:
    """Nearest open/closed joint value — mirrors
    PyBulletEnv._fingers_state_to_joint."""
    open_j = robot.open_fingers
    closed_j = robot.closed_fingers
    if abs(finger_state - open_j) <= abs(finger_state - closed_j):
        return open_j
    return closed_j


def _make_config(robot) -> SkillConfig:
    return SkillConfig(
        robot=robot,
        open_fingers_joint=robot.open_fingers,
        closed_fingers_joint=robot.closed_fingers,
        fingers_state_to_joint=_fingers_state_to_joint,
    )


def _make_robot_obj() -> Object:
    return Object("robot0", _ROBOT_TYPE)


def _make_obj() -> Object:
    return Object("obj0", _OBJ_TYPE)


def _build_state(
    robot_obj: Object,
    robot,
    ee_x: float,
    ee_y: float,
    ee_z: float,
    finger_state: float = _OPEN_STATE,
    obj: Object = None,  # type: ignore[assignment]
    obj_xyz=(0.0, 0.0, 0.0),
) -> utils.PyBulletState:
    """Build a PyBulletState at the specified EE position.

    Uses the robot's initial joint positions as the simulator state.
    When the EE position equals the home position, the state is fully
    self-consistent (joint positions match the EE pose in state
    features).
    """
    tilt = np.pi / 2
    wrist = -np.pi
    data = {
        robot_obj:
        np.array([ee_x, ee_y, ee_z, tilt, wrist, finger_state],
                 dtype=np.float32)
    }
    if obj is not None:
        data[obj] = np.array(obj_xyz, dtype=np.float32)
    joint_positions = list(robot.initial_joint_positions)
    return utils.PyBulletState(data, simulator_state=joint_positions)


def _make_home_state(
    robot_obj: Object,
    robot,
    finger_state: float = _OPEN_STATE,
    obj: Object = None,  # type: ignore[assignment]
    obj_xyz=(0.0, 0.0, 0.0),
) -> utils.PyBulletState:
    """Build a fully self-consistent PyBulletState at the robot's home pose.

    Resets the robot to its cached initial joint positions and reads the
    actual EE state from PyBullet, so that joint_positions and EE
    features are always mutually consistent regardless of prior robot
    manipulation.
    """
    robot.set_joints(robot.initial_joint_positions)
    raw = robot.get_state()  # [rx, ry, rz, qx, qy, qz, qw, rf]
    rx, ry, rz, qx, qy, qz, qw, _ = raw
    tilt_val = p.getEulerFromQuaternion([qx, qy, qz, qw])[1]
    wrist_val = p.getEulerFromQuaternion([qx, qy, qz, qw])[2]
    data = {
        robot_obj:
        np.array([rx, ry, rz, tilt_val, wrist_val, finger_state],
                 dtype=np.float32)
    }
    if obj is not None:
        data[obj] = np.array(obj_xyz, dtype=np.float32)
    return utils.PyBulletState(data,
                               simulator_state=list(
                                   robot.initial_joint_positions))


# ===========================================================================
# 1. SkillConfig
# ===========================================================================


class TestSkillConfig:
    """TestSkillConfig class."""

    def test_required_fields_stored(self, robot_scene):
        """Test required fields stored."""
        _, robot = robot_scene
        cfg = SkillConfig(
            robot=robot,
            open_fingers_joint=robot.open_fingers,
            closed_fingers_joint=robot.closed_fingers,
            fingers_state_to_joint=_fingers_state_to_joint,
        )
        assert cfg.robot is robot
        assert cfg.open_fingers_joint == robot.open_fingers
        assert cfg.closed_fingers_joint == robot.closed_fingers

    def test_default_tolerances(self, robot_scene):
        """Test default tolerances."""
        _, robot = robot_scene
        cfg = _make_config(robot)
        assert cfg.move_to_pose_tol == pytest.approx(1e-4)
        assert cfg.max_vel_norm == pytest.approx(0.05)
        assert cfg.grasp_tol == pytest.approx(5e-4)
        assert cfg.collision_bodies == ()
        assert cfg.ik_validate is True
        assert cfg.robot_init_tilt == pytest.approx(0.0)
        assert cfg.robot_init_wrist == pytest.approx(0.0)

    def test_extra_dict_stored(self, robot_scene):
        """Test extra dict stored."""
        _, robot = robot_scene
        cfg = SkillConfig(
            robot=robot,
            open_fingers_joint=robot.open_fingers,
            closed_fingers_joint=robot.closed_fingers,
            fingers_state_to_joint=_fingers_state_to_joint,
            extra={"my_key": 42},
        )
        assert cfg.extra["my_key"] == 42

    def test_custom_tolerances(self, robot_scene):
        """Test custom tolerances."""
        _, robot = robot_scene
        cfg = SkillConfig(
            robot=robot,
            open_fingers_joint=robot.open_fingers,
            closed_fingers_joint=robot.closed_fingers,
            fingers_state_to_joint=_fingers_state_to_joint,
            move_to_pose_tol=5e-5,
            max_vel_norm=0.02,
            grasp_tol=2e-3,
        )
        assert cfg.move_to_pose_tol == pytest.approx(5e-5)
        assert cfg.max_vel_norm == pytest.approx(0.02)
        assert cfg.grasp_tol == pytest.approx(2e-3)


# ===========================================================================
# 2. Phase dataclass
# ===========================================================================


class TestPhase:
    """TestPhase class."""

    def test_move_to_pose_phase(self):
        """Test move to pose phase."""

        def dummy_target(_state, _objects, _params, _cfg):
            return None, None, "open"

        phase = Phase(name="TestMove",
                      action_type=PhaseAction.MOVE_TO_POSE,
                      target_fn=dummy_target)
        assert phase.name == "TestMove"
        assert phase.action_type == PhaseAction.MOVE_TO_POSE
        assert phase.terminal_fn is None
        assert phase.use_motion_planning is False  # default from CFG

    def test_change_fingers_phase(self):
        """Test change fingers phase."""

        def dummy_target(_state, _objects, _params, _cfg):
            return 0.04, 0.01

        phase = Phase(name="Grasp",
                      action_type=PhaseAction.CHANGE_FINGERS,
                      target_fn=dummy_target)
        assert phase.action_type == PhaseAction.CHANGE_FINGERS

    def test_custom_terminal_fn_stored(self):
        """Test custom terminal fn stored."""

        def my_terminal(_state, _objects, _params, _cfg):
            return True

        phase = Phase(
            name="CustomPhase",
            action_type=PhaseAction.CHANGE_FINGERS,
            target_fn=lambda s, o, p_, c: (0.0, 0.0),
            terminal_fn=my_terminal,
        )
        assert phase.terminal_fn is my_terminal

    def test_no_motion_planning_flag(self):
        """Test no motion planning flag."""
        phase = Phase(
            name="IKMove",
            action_type=PhaseAction.MOVE_TO_POSE,
            target_fn=lambda s, o, p_, c: (None, None, "open"),
            use_motion_planning=False,
        )
        assert phase.use_motion_planning is False


# ===========================================================================
# 3. PhaseSkill — structure and public-interface behaviour
# ===========================================================================


class TestPhaseSkill:
    """TestPhaseSkill class."""

    def _make_single_ik_skill(self, robot, target_pos):
        """One IK-mode MOVE_TO_POSE phase (no BiRRT, predictable terminal)."""
        config = _make_config(robot)
        robot_obj = _make_robot_obj()

        def target_fn(state, _objects, _params, cfg):
            x = state.get(robot_obj, "x")
            y = state.get(robot_obj, "y")
            z = state.get(robot_obj, "z")
            tilt = state.get(robot_obj, "tilt")
            wrist = state.get(robot_obj, "wrist")
            orn = p.getQuaternionFromEuler([0, tilt, wrist])
            current = Pose((x, y, z), orn)
            orn_tgt = p.getQuaternionFromEuler([0, cfg.robot_init_tilt, 0.0])
            target = Pose(target_pos, orn_tgt)
            return current, target, "open"

        phase = Phase(
            name="Move",
            action_type=PhaseAction.MOVE_TO_POSE,
            target_fn=target_fn,
            use_motion_planning=False,
        )
        skill = PhaseSkill("Test", [_ROBOT_TYPE], Box(0, 1, (0, )), config,
                           [phase])
        return skill, robot_obj, phase

    def _make_single_cf_skill(self, robot, current_val, target_val):
        """One CHANGE_FINGERS phase with fixed current/target."""
        config = _make_config(robot)

        def target_fn(_state, _objects, _params, _cfg):
            return current_val, target_val

        phase = Phase(
            name="CF",
            action_type=PhaseAction.CHANGE_FINGERS,
            target_fn=target_fn,
        )
        skill = PhaseSkill("TestCF", [_ROBOT_TYPE], Box(0, 1, (0, )), config,
                           [phase])
        return skill, phase

    def test_build_returns_parameterized_option(self, robot_scene):
        """Test build returns parameterized option."""
        _, robot = robot_scene
        skill, _robot_obj, _ = self._make_single_ik_skill(robot, _EE_HOME)
        opt = skill.build()
        assert isinstance(opt, ParameterizedOption)

    def test_build_name_and_types(self, robot_scene):
        """Test build name and types."""
        _, robot = robot_scene
        skill, _robot_obj, _ = self._make_single_ik_skill(robot, _EE_HOME)
        opt = skill.build()
        assert opt.name == "Test"
        assert opt.types == [_ROBOT_TYPE]

    def test_initiable_sets_phase_idx_zero(self, robot_scene):
        """Test initiable sets phase idx zero."""
        _, robot = robot_scene
        skill, _robot_obj, _ = self._make_single_ik_skill(robot, _EE_HOME)
        opt = skill.build()
        grounded = opt.ground([_make_robot_obj()], np.zeros(0))
        state = _build_state(_make_robot_obj(), robot, *_EE_HOME)
        assert grounded.initiable(state)
        assert grounded.memory["phase_idx"] == 0

    def test_change_fingers_terminal_when_at_target(self, robot_scene):
        """Test change fingers terminal when at target."""
        _, robot = robot_scene
        # current == target → (target-current)^2 = 0 < grasp_tol
        skill, _ = self._make_single_cf_skill(robot, 0.04, 0.04)
        opt = skill.build()
        robot_obj = _make_robot_obj()
        grounded = opt.ground([robot_obj], np.zeros(0))
        state = _build_state(robot_obj, robot, *_EE_HOME)
        grounded.initiable(state)
        assert grounded.terminal(state)

    def test_change_fingers_not_terminal_when_far(self, robot_scene):
        """Test change fingers not terminal when far."""
        _, robot = robot_scene
        # current=0.04, target=0.00 → (0.00-0.04)^2 = 1.6e-3 > 1e-3
        skill, _ = self._make_single_cf_skill(robot, 0.04, 0.00)
        opt = skill.build()
        robot_obj = _make_robot_obj()
        grounded = opt.ground([robot_obj], np.zeros(0))
        state = _build_state(robot_obj, robot, *_EE_HOME)
        grounded.initiable(state)
        assert not grounded.terminal(state)

    def test_ik_terminal_when_at_target(self, robot_scene):
        """Test ik terminal when at target."""
        _, robot = robot_scene
        # Target == current EE position → distance = 0 < tol
        skill, robot_obj, _ = self._make_single_ik_skill(robot, _EE_HOME)
        opt = skill.build()
        robot_obj = _make_robot_obj()
        grounded = opt.ground([robot_obj], np.zeros(0))
        state = _build_state(robot_obj, robot, *_EE_HOME)
        grounded.initiable(state)
        assert grounded.terminal(state)

    def test_ik_not_terminal_when_far(self, robot_scene):
        """Test ik not terminal when far."""
        _, robot = robot_scene
        # Target is far from current EE (0.3m away in z)
        far_target = (_EE_HOME[0], _EE_HOME[1], _EE_HOME[2] - 0.3)
        skill, robot_obj, _ = self._make_single_ik_skill(robot, far_target)
        opt = skill.build()
        robot_obj = _make_robot_obj()
        grounded = opt.ground([robot_obj], np.zeros(0))
        state = _build_state(robot_obj, robot, *_EE_HOME)
        grounded.initiable(state)
        assert not grounded.terminal(state)

    def test_multi_phase_terminal_only_on_last(self, robot_scene):
        """With 2 phases, terminal is False even when phase 0 would be
        terminal."""
        _, robot = robot_scene
        config = _make_config(robot)

        # Phase 0: CHANGE_FINGERS, immediately terminal (current==target).
        phase0 = Phase(
            name="CF",
            action_type=PhaseAction.CHANGE_FINGERS,
            target_fn=lambda s, o, p_, c: (0.04, 0.04),
        )
        # Phase 1: CHANGE_FINGERS, NOT terminal (current 0.04, target 0.00).
        phase1 = Phase(
            name="CF2",
            action_type=PhaseAction.CHANGE_FINGERS,
            target_fn=lambda s, o, p_, c: (0.04, 0.00),
        )
        skill = PhaseSkill("TwoPhase", [_ROBOT_TYPE], Box(0, 1, (0, )), config,
                           [phase0, phase1])
        opt = skill.build()
        robot_obj = _make_robot_obj()
        grounded = opt.ground([robot_obj], np.zeros(0))
        state = _build_state(robot_obj, robot, *_EE_HOME)
        grounded.initiable(state)
        # Phase 0 is terminal, but we're on phase 0 of 2 → overall not terminal.
        assert not grounded.terminal(state)

    def test_policy_advances_phase_when_terminal(self, robot_scene):
        """Calling policy when phase is terminal bumps phase_idx."""
        _, robot = robot_scene
        config = _make_config(robot)

        # Phase 0: immediately terminal.
        phase0 = Phase(
            name="CF0",
            action_type=PhaseAction.CHANGE_FINGERS,
            target_fn=lambda s, o, p_, c: (0.04, 0.04),
        )
        # Phase 1: not terminal.
        phase1 = Phase(
            name="CF1",
            action_type=PhaseAction.CHANGE_FINGERS,
            target_fn=lambda s, o, p_, c: (0.04, 0.00),
        )
        skill = PhaseSkill("Advance", [_ROBOT_TYPE], Box(0, 1, (0, )), config,
                           [phase0, phase1])
        opt = skill.build()
        robot_obj = _make_robot_obj()
        grounded = opt.ground([robot_obj], np.zeros(0))
        state = _build_state(robot_obj,
                             robot,
                             *_EE_HOME,
                             finger_state=_OPEN_STATE)
        grounded.initiable(state)
        assert grounded.memory["phase_idx"] == 0
        # Phase 0 is terminal; policy should advance to phase 1.
        grounded.policy(state)
        assert grounded.memory["phase_idx"] == 1

    def test_custom_terminal_fn_overrides_default(self, robot_scene):
        """A custom terminal_fn takes precedence over distance-based
        terminal."""
        _, robot = robot_scene
        config = _make_config(robot)
        call_count = {"n": 0}

        def my_terminal(_state, _objects, _params, _cfg):
            call_count["n"] += 1
            return True

        phase = Phase(
            name="Custom",
            action_type=PhaseAction.CHANGE_FINGERS,
            target_fn=lambda s, o, p_, c:
            (0.04, 0.00),  # would not be terminal
            terminal_fn=my_terminal,
        )
        skill = PhaseSkill("CustomTerm", [_ROBOT_TYPE], Box(0, 1, (0, )),
                           config, [phase])
        opt = skill.build()
        robot_obj = _make_robot_obj()
        grounded = opt.ground([robot_obj], np.zeros(0))
        state = _build_state(robot_obj, robot, *_EE_HOME)
        grounded.initiable(state)
        assert grounded.terminal(state)  # custom fn returns True
        assert call_count["n"] >= 1


# ===========================================================================
# 4. BiRRT trajectory caching and IK fallback
# ===========================================================================


class TestBiRRT:
    """Integration tests requiring a real PyBullet robot."""

    def test_birrt_not_terminal_before_first_policy_call(self, robot_scene):
        """With BiRRT mode, terminal is False until the first policy call."""
        _, robot = robot_scene
        utils.reset_config({"seed": 123})
        config = _make_config(robot)
        robot_obj = _make_robot_obj()

        def target_fn(_state, _objects, _params, _cfg):
            orn = p.getQuaternionFromEuler([0, 0, 0])
            return Pose(_EE_HOME, orn), Pose(_EE_HOME, orn), "open"

        phase = Phase("Move",
                      PhaseAction.MOVE_TO_POSE,
                      target_fn,
                      use_motion_planning=True)
        skill = PhaseSkill("BT", [_ROBOT_TYPE], Box(0, 1, (0, )), config,
                           [phase])
        opt = skill.build()
        grounded = opt.ground([robot_obj], np.zeros(0))
        state = _build_state(robot_obj, robot, *_EE_HOME)
        grounded.initiable(state)
        # No trajectory in memory yet → NOT terminal.
        assert not grounded.terminal(state)

    def test_birrt_caches_trajectory_after_first_policy_call(
            self, robot_scene):
        """After the first policy call, a trajectory is cached in memory."""
        _, robot = robot_scene
        utils.reset_config({"seed": 123})
        config = _make_config(robot)
        robot_obj = _make_robot_obj()
        home_orn = p.getQuaternionFromEuler([0, np.pi / 2, -np.pi])

        def target_fn(state, _objects, _params, _cfg):
            current_orn = p.getQuaternionFromEuler([
                0,
                state.get(robot_obj, "tilt"),
                state.get(robot_obj, "wrist"),
            ])
            current = Pose(
                (state.get(robot_obj, "x"), state.get(
                    robot_obj, "y"), state.get(robot_obj, "z")),
                current_orn,
            )
            # Target = home, same as current, so BiRRT trivially succeeds.
            target = Pose(_EE_HOME, home_orn)
            return current, target, "open"

        phase = Phase("Move",
                      PhaseAction.MOVE_TO_POSE,
                      target_fn,
                      use_motion_planning=True)
        skill = PhaseSkill("BT", [_ROBOT_TYPE], Box(0, 1, (0, )), config,
                           [phase])
        opt = skill.build()
        grounded = opt.ground([robot_obj], np.zeros(0))

        state = _make_home_state(robot_obj, robot)
        grounded.initiable(state)

        action = grounded.policy(state)
        assert isinstance(action, Action)
        assert action.arr.shape == robot.action_space.shape

        # Trajectory should now be cached.
        traj_key = _BIRRT_TRAJ_KEY.format(id(phase))
        assert traj_key in grounded.memory

    def test_birrt_terminal_after_trajectory_exhausted(self, robot_scene):
        """Terminal becomes True once all trajectory waypoints are consumed."""
        _, robot = robot_scene
        utils.reset_config({"seed": 123})
        config = _make_config(robot)
        robot_obj = _make_robot_obj()
        home_orn = p.getQuaternionFromEuler([0, np.pi / 2, -np.pi])

        def target_fn(state, _objects, _params, _cfg):
            current_orn = p.getQuaternionFromEuler([
                0,
                state.get(robot_obj, "tilt"),
                state.get(robot_obj, "wrist"),
            ])
            current = Pose(
                (state.get(robot_obj, "x"), state.get(
                    robot_obj, "y"), state.get(robot_obj, "z")),
                current_orn,
            )
            # Same-position target → BiRRT path is short (a few waypoints).
            return current, Pose(_EE_HOME, home_orn), "open"

        phase = Phase("Move",
                      PhaseAction.MOVE_TO_POSE,
                      target_fn,
                      use_motion_planning=True)
        skill = PhaseSkill("BT", [_ROBOT_TYPE], Box(0, 1, (0, )), config,
                           [phase])
        opt = skill.build()
        grounded = opt.ground([robot_obj], np.zeros(0))
        state = _make_home_state(robot_obj, robot)
        grounded.initiable(state)

        assert not grounded.terminal(state)  # no traj yet

        # Consume waypoints by calling policy until terminal.
        # Path length varies with IK rounding; 50 steps is more than enough.
        for _ in range(50):
            if grounded.terminal(state):
                break
            grounded.policy(state)
        else:
            pytest.fail(
                "BiRRT terminal never became True after 50 policy calls")

        assert grounded.terminal(state)

    def test_birrt_fallback_to_ik_when_traj_is_none(self, robot_scene):
        """When memory[traj_key]=None (BiRRT failure), policy uses IK fallback.

        We inject the failure directly into memory rather than depending
        on BiRRT actually failing, which is non-deterministic with
        limited budgets and no collision obstacles.
        """
        _, robot = robot_scene
        utils.reset_config({"seed": 123})
        config = _make_config(robot)
        robot_obj = _make_robot_obj()
        home_orn = p.getQuaternionFromEuler([0, np.pi / 2, -np.pi])
        target_pos = (_EE_HOME[0], _EE_HOME[1], _EE_HOME[2] - 0.15)

        def target_fn(state, _objects, _params, _cfg):
            current_orn = p.getQuaternionFromEuler([
                0,
                state.get(robot_obj, "tilt"),
                state.get(robot_obj, "wrist"),
            ])
            current = Pose((state.get(robot_obj, "x"), state.get(
                robot_obj, "y"), state.get(robot_obj, "z")), current_orn)
            target = Pose(target_pos, home_orn)
            return current, target, "open"

        phase = Phase("Move",
                      PhaseAction.MOVE_TO_POSE,
                      target_fn,
                      use_motion_planning=True)
        skill = PhaseSkill("FB", [_ROBOT_TYPE], Box(0, 1, (0, )), config,
                           [phase])
        opt = skill.build()
        grounded = opt.ground([robot_obj], np.zeros(0))
        state = _make_home_state(robot_obj, robot)
        grounded.initiable(state)

        # Simulate BiRRT failure: set traj = None in memory.
        traj_key = _BIRRT_TRAJ_KEY.format(id(phase))
        step_key = _BIRRT_STEP_KEY.format(id(phase))
        grounded.memory[traj_key] = None
        grounded.memory[step_key] = 0

        # Policy must not raise — IK fallback is activated.
        action = grounded.policy(state)
        assert isinstance(action, Action)
        assert robot.action_space.contains(action.arr)

        # Fallback terminal is distance-based: target 0.15m away → not terminal.
        assert not grounded.terminal(state)


# ===========================================================================
# 5. Wait option
# ===========================================================================


class TestWaitOption:
    """TestWaitOption class."""

    def test_wait_always_initiable(self, robot_scene):
        """Test wait always initiable."""
        _, robot = robot_scene
        config = _make_config(robot)
        opt = create_wait_option("Wait", config, _ROBOT_TYPE)
        robot_obj = _make_robot_obj()
        grounded = opt.ground([robot_obj], np.zeros(0))
        state = _build_state(robot_obj, robot, *_EE_HOME)
        assert grounded.initiable(state)

    def test_wait_never_terminal(self, robot_scene):
        """Test wait never terminal."""
        _, robot = robot_scene
        config = _make_config(robot)
        opt = create_wait_option("Wait", config, _ROBOT_TYPE)
        robot_obj = _make_robot_obj()
        grounded = opt.ground([robot_obj], np.zeros(0))
        state = _build_state(robot_obj, robot, *_EE_HOME)
        for _ in range(5):
            assert not grounded.terminal(state)

    def test_wait_custom_name(self, robot_scene):
        """Test wait custom name."""
        _, robot = robot_scene
        config = _make_config(robot)
        opt = create_wait_option("Idle", config, _ROBOT_TYPE)
        assert opt.name == "Idle"

    def test_wait_default_name(self, robot_scene):
        """Test wait default name."""
        _, robot = robot_scene
        config = _make_config(robot)
        opt = create_wait_option("Wait", config, _ROBOT_TYPE)
        assert opt.name == "Wait"

    def test_wait_policy_nudges_fingers_open(self, robot_scene):
        """When fingers are open, the action should nudge them more open."""
        _, robot = robot_scene
        config = _make_config(robot)
        opt = create_wait_option("Wait", config, _ROBOT_TYPE)
        robot_obj = _make_robot_obj()
        grounded = opt.ground([robot_obj], np.zeros(0))
        state = _build_state(robot_obj,
                             robot,
                             *_EE_HOME,
                             finger_state=_OPEN_STATE)
        action = grounded.policy(state)
        l_idx = robot.left_finger_joint_idx
        # Finger nudge should be positive (open direction).
        initial_fingers = state.joint_positions[l_idx]
        assert action.arr[l_idx] > initial_fingers

    def test_wait_policy_nudges_fingers_closed(self, robot_scene):
        """When fingers are closed, the action should nudge them more
        closed."""
        _, robot = robot_scene
        config = _make_config(robot)
        opt = create_wait_option("Wait", config, _ROBOT_TYPE)
        robot_obj = _make_robot_obj()
        grounded = opt.ground([robot_obj], np.zeros(0))
        state = _build_state(robot_obj,
                             robot,
                             *_EE_HOME,
                             finger_state=_CLOSED_STATE)
        action = grounded.policy(state)
        l_idx = robot.left_finger_joint_idx
        initial_fingers = state.joint_positions[l_idx]
        # Finger nudge should be negative (closed direction).
        assert action.arr[l_idx] < initial_fingers

    def test_wait_policy_action_within_bounds(self, robot_scene):
        """The action returned by wait must lie within the robot's action
        space."""
        _, robot = robot_scene
        config = _make_config(robot)
        opt = create_wait_option("Wait", config, _ROBOT_TYPE)
        robot_obj = _make_robot_obj()
        grounded = opt.ground([robot_obj], np.zeros(0))
        state = _build_state(robot_obj, robot, *_EE_HOME)
        action = grounded.policy(state)
        assert robot.action_space.contains(action.arr)

    def test_wait_non_finger_joints_unchanged(self, robot_scene):
        """Wait must not move any joints except the two finger joints."""
        _, robot = robot_scene
        config = _make_config(robot)
        opt = create_wait_option("Wait", config, _ROBOT_TYPE)
        robot_obj = _make_robot_obj()
        grounded = opt.ground([robot_obj], np.zeros(0))
        state = _build_state(robot_obj, robot, *_EE_HOME)
        action = grounded.policy(state)
        l_idx = robot.left_finger_joint_idx
        r_idx = robot.right_finger_joint_idx
        for i, (act, orig) in enumerate(zip(action.arr,
                                            state.joint_positions)):
            if i not in (l_idx, r_idx):
                assert act == pytest.approx(orig, abs=1e-6), \
                    f"Joint {i} should not change in wait policy"


# ===========================================================================
# 6. make_move_to_phase
# ===========================================================================


class TestMakeMoveToPosePhase:
    """TestMakeMoveToPosePhase class."""

    def test_returns_phase_with_move_action_type(self):
        """Test returns phase with move action type."""
        phase = make_move_to_phase(
            "MoveTest",
            get_target_pose_fn=lambda s, o, p_, c: (1.0, 2.0, 3.0, 0.0),
            finger_status="open",
        )
        assert isinstance(phase, Phase)
        assert phase.action_type == PhaseAction.MOVE_TO_POSE
        assert phase.name == "MoveTest"
        assert phase.use_motion_planning is False  # default from CFG

    def test_explicit_open_finger_status(self, robot_scene):
        """Test explicit open finger status."""
        _, robot = robot_scene
        config = _make_config(robot)
        robot_obj = _make_robot_obj()
        phase = make_move_to_phase(
            "OpenMove",
            get_target_pose_fn=lambda s, o, p_, c: (*_EE_HOME, 0.0),
            finger_status="open",
        )
        state = _build_state(robot_obj,
                             robot,
                             *_EE_HOME,
                             finger_state=_CLOSED_STATE)  # state says closed
        _, _, returned_status = phase.target_fn(state, [robot_obj],
                                                np.zeros(0), config)
        assert returned_status == "open"  # explicit overrides state

    def test_explicit_closed_finger_status(self, robot_scene):
        """Test explicit closed finger status."""
        _, robot = robot_scene
        config = _make_config(robot)
        robot_obj = _make_robot_obj()
        phase = make_move_to_phase(
            "ClosedMove",
            get_target_pose_fn=lambda s, o, p_, c: (*_EE_HOME, 0.0),
            finger_status="closed",
        )
        state = _build_state(robot_obj,
                             robot,
                             *_EE_HOME,
                             finger_state=_OPEN_STATE)  # state says open
        _, _, returned_status = phase.target_fn(state, [robot_obj],
                                                np.zeros(0), config)
        assert returned_status == "closed"

    def test_inferred_open_finger_status(self, robot_scene):
        """When finger_status=None, infers 'open' from state with open
        fingers."""
        _, robot = robot_scene
        config = _make_config(robot)
        robot_obj = _make_robot_obj()
        phase = make_move_to_phase(
            "InferOpen",
            get_target_pose_fn=lambda s, o, p_, c: (*_EE_HOME, 0.0),
            finger_status=None,
        )
        state = _build_state(robot_obj,
                             robot,
                             *_EE_HOME,
                             finger_state=_OPEN_STATE)
        _, _, returned_status = phase.target_fn(state, [robot_obj],
                                                np.zeros(0), config)
        assert returned_status == "open"

    def test_inferred_closed_finger_status(self, robot_scene):
        """When finger_status=None, infers 'closed' from state with closed
        fingers."""
        _, robot = robot_scene
        config = _make_config(robot)
        robot_obj = _make_robot_obj()
        phase = make_move_to_phase(
            "InferClosed",
            get_target_pose_fn=lambda s, o, p_, c: (*_EE_HOME, 0.0),
            finger_status=None,
        )
        state = _build_state(robot_obj,
                             robot,
                             *_EE_HOME,
                             finger_state=_CLOSED_STATE)
        _, _, returned_status = phase.target_fn(state, [robot_obj],
                                                np.zeros(0), config)
        assert returned_status == "closed"

    def test_target_position_is_forwarded(self, robot_scene):
        """The target (x, y, z, yaw) from get_target_pose_fn is used."""
        _, robot = robot_scene
        config = _make_config(robot)
        robot_obj = _make_robot_obj()
        custom_target = (1.1, 2.2, 3.3, 0.5)
        phase = make_move_to_phase(
            "TargetCheck",
            get_target_pose_fn=lambda s, o, p_, c: custom_target,
        )
        state = _build_state(robot_obj, robot, *_EE_HOME)
        _, target_pose, _ = phase.target_fn(state, [robot_obj], np.zeros(0),
                                            config)
        assert target_pose.position == pytest.approx(custom_target[:3],
                                                     abs=1e-6)


# ===========================================================================
# 7. create_move_to_skill
# ===========================================================================


class TestCreateMoveToPoseSkill:
    """TestCreateMoveToPoseSkill class."""

    def test_returns_parameterized_option(self, robot_scene):
        """Test returns parameterized option."""
        _, robot = robot_scene
        config = _make_config(robot)
        opt = create_move_to_skill(
            "Move",
            [_ROBOT_TYPE],
            Box(0, 1, (0, )),
            config,
            get_target_pose_fn=lambda s, o, p_, c: (*_EE_HOME, 0.0),
        )
        assert isinstance(opt, ParameterizedOption)
        assert opt.name == "Move"

    def test_policy_returns_valid_action(self, robot_scene):
        """Test policy returns valid action."""
        _, robot = robot_scene
        utils.reset_config({"seed": 123})
        config = _make_config(robot)
        robot_obj = _make_robot_obj()
        opt = create_move_to_skill(
            "Move",
            [_ROBOT_TYPE],
            Box(0, 1, (0, )),
            config,
            get_target_pose_fn=lambda s, o, p_, c: (*_EE_HOME, 0.0),
        )
        grounded = opt.ground([robot_obj], np.zeros(0))
        state = _make_home_state(robot_obj, robot)
        grounded.initiable(state)
        action = grounded.policy(state)
        assert isinstance(action, Action)
        assert robot.action_space.contains(action.arr)


# ===========================================================================
# 8. create_pick_skill — structure
# ===========================================================================


class TestCreatePickSkill:
    """TestCreatePickSkill class."""

    def _make_pick(self, robot):
        config = SkillConfig(
            robot=robot,
            open_fingers_joint=robot.open_fingers,
            closed_fingers_joint=robot.closed_fingers,
            fingers_state_to_joint=_fingers_state_to_joint,
            transport_z=0.8,
        )
        return create_pick_skill(
            name="Pick",
            types=[_ROBOT_TYPE, _OBJ_TYPE],
            config=config,
            get_target_pose_fn=lambda s, o, p_, c: (1.35, 0.75, 0.4, 0.0),
        )

    def test_returns_parameterized_option(self, robot_scene):
        """Test returns parameterized option."""
        _, robot = robot_scene
        opt = self._make_pick(robot)
        assert isinstance(opt, ParameterizedOption)
        assert opt.name == "Pick"

    def test_pick_policy_returns_valid_action(self, robot_scene):
        """Test pick policy returns valid action."""
        _, robot = robot_scene
        utils.reset_config({"seed": 123})
        robot_obj = _make_robot_obj()
        obj = _make_obj()
        opt = self._make_pick(robot)
        # Pick params: (grasp_z_offset) — use 0.02
        grounded = opt.ground([robot_obj, obj],
                              np.array([0.02], dtype=np.float32))
        state = _make_home_state(robot_obj,
                                 robot,
                                 obj=obj,
                                 obj_xyz=(1.35, 0.75, 0.4))
        grounded.initiable(state)
        action = grounded.policy(state)
        assert isinstance(action, Action)
        assert robot.action_space.contains(action.arr)


# ===========================================================================
# 9. create_place_skill — structure
# ===========================================================================


class TestCreatePlaceSkill:
    """TestCreatePlaceSkill class."""

    def _make_place(self, robot):
        config = SkillConfig(
            robot=robot,
            open_fingers_joint=robot.open_fingers,
            closed_fingers_joint=robot.closed_fingers,
            fingers_state_to_joint=_fingers_state_to_joint,
            transport_z=0.8,
        )
        return create_place_skill(
            name="Place",
            types=[_ROBOT_TYPE],
            config=config,
        )

    def test_returns_parameterized_option(self, robot_scene):
        """Test returns parameterized option."""
        _, robot = robot_scene
        opt = self._make_place(robot)
        assert isinstance(opt, ParameterizedOption)
        assert opt.name == "Place"

    def test_place_policy_returns_valid_action(self, robot_scene):
        """Test place policy returns valid action."""
        _, robot = robot_scene
        utils.reset_config({"seed": 123})
        robot_obj = _make_robot_obj()
        opt = self._make_place(robot)
        # Place params: (target_x, target_y, release_z, target_yaw)
        grounded = opt.ground([robot_obj],
                              np.array([0.75, 1.35, 0.55, 0.0],
                                       dtype=np.float32))
        state = _make_home_state(robot_obj, robot)
        grounded.initiable(state)
        action = grounded.policy(state)
        assert isinstance(action, Action)
        assert robot.action_space.contains(action.arr)


# ===========================================================================
# 10. create_push_skill — structure
# ===========================================================================


class TestCreatePushSkill:
    """TestCreatePushSkill class."""

    @staticmethod
    def _make_push_config(robot):
        # robot_home_pos is required for create_push_skill
        return SkillConfig(
            robot=robot,
            open_fingers_joint=robot.open_fingers,
            closed_fingers_joint=robot.closed_fingers,
            fingers_state_to_joint=_fingers_state_to_joint,
            robot_home_pos=_EE_HOME,
            transport_z=0.8,
        )

    def _make_push(self, robot):
        """Make push."""
        config = self._make_push_config(robot)
        return create_push_skill(
            name="Push",
            types=[_ROBOT_TYPE, _OBJ_TYPE],
            config=config,
            get_target_pose_fn=lambda s, o, p_, c: (1.35, 0.75, 0.4, 0.0),
        )

    def test_returns_parameterized_option(self, robot_scene):
        """Test returns parameterized option."""
        _, robot = robot_scene
        opt = self._make_push(robot)
        assert isinstance(opt, ParameterizedOption)
        assert opt.name == "Push"

    def test_push_policy_close_fingers_returns_valid_action(self, robot_scene):
        """First call lands in CloseFingers phase -> action within bounds."""
        _, robot = robot_scene
        utils.reset_config({"seed": 123})
        robot_obj = _make_robot_obj()
        obj = _make_obj()
        opt = self._make_push(robot)
        # Push params: (approach_distance, contact_z_offset)
        grounded = opt.ground([robot_obj, obj],
                              np.array([0.05, 0.02], dtype=np.float32))
        state = _build_state(robot_obj,
                             robot,
                             *_EE_HOME,
                             finger_state=_OPEN_STATE,
                             obj=obj,
                             obj_xyz=(1.35, 0.75, 0.4))
        grounded.initiable(state)
        action = grounded.policy(state)
        assert isinstance(action, Action)
        assert robot.action_space.contains(action.arr)
