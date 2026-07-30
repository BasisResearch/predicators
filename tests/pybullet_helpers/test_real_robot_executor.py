"""Tests for the real-robot executor: chunking, twin re-sync, what the agent
sees.

These run against a **stub** env and a **stub** robot, and must never skip: a
suite that silently skipped without the private submodule would hide exactly
the regressions it exists to catch. That is asserted below.

The executor is deliberately reachable without babyrobot because it only ever
touches the robot through ``real_robot_bridge.execute_chunks`` / ``reset_arm``,
which these tests replace with recorders. Building the ``Segment`` objects
those helpers ship is babyrobot's contract, covered in
``test_real_robot_bridge.py``, which does skip.
"""
import ast
import inspect
from typing import Any, List, Optional, cast

import numpy as np
import pytest
from gym.spaces import Box

from predicators import utils
from predicators.envs.pybullet_env import PyBulletEnv
from predicators.pybullet_helpers.real_robot_bridge import GripperJointLayout
from predicators.pybullet_helpers.real_robot_executor import \
    OptionBoundaryBuffer, RealRobotExecutor, attach_real_robot
from predicators.structs import Action, Object, ParameterizedOption, State, \
    Type

_LAYOUT = GripperJointLayout(left_finger_joint_idx=7,
                             right_finger_joint_idx=8,
                             open_fingers=0.04,
                             closed_fingers=0.0)

# A minimal object type with positions, so divergence has something to measure.
_BLOCK_TYPE = Type("block", ["x", "y", "z"])
_BLOCK = Object("block0", _BLOCK_TYPE)


def _state(x: float, joints: Optional[List[float]] = None) -> State:
    """A one-object state carrying joint positions, as the twin's does.

    A ``PyBulletState`` specifically: ``_set_state`` trusts
    ``simulator_state["joint_positions"]`` only when it is there, and
    falls back to IK -- which drops wrist roll -- when it is not.
    """
    return utils.PyBulletState(
        {_BLOCK: np.array([x, 0.0, 0.0])},
        simulator_state={
            "joint_positions":
            joints if joints is not None else [float(i) for i in range(9)]
        })


class _StubObservation:
    """Stands in for a babyrobot observation.

    Opaque to the executor: handed straight to the env's
    ``state_from_observation``.
    """

    def __init__(self, x: float) -> None:
        self.x = x


class _StubEnv:
    """A twin with no PyBullet: it records what was asked of it.

    Implements only what the executor touches -- note that is *not*
    ``reset``/``step``, since the env now calls the executor rather than
    the other way round. Passed where a ``PyBulletEnv`` is declared via
    one cast, which is what keeps these tests free of a physics client.
    """

    def __init__(self) -> None:
        self.synced: List[State] = []
        self._observation = _state(0.0)

    def get_observation(self) -> State:
        """The twin's current state."""
        return self._observation

    def set_observation(self, state: State) -> None:
        """Put the twin in a given state, as a reset would."""
        self._observation = state

    def sync_to_state(self, state: State) -> None:
        """Adopt ``state`` as the twin's world, as PyBullet's would."""
        self.synced.append(state)
        self._observation = state

    def gripper_joint_layout(self) -> GripperJointLayout:
        """The finger layout the splitter needs."""
        return _LAYOUT

    def state_from_observation(self, obs: Any, prev_state: State) -> State:
        """Move the block to wherever the observation says it is."""
        del prev_state
        return _state(obs.x)

    def task_from_observation(self, obs: Any, train_or_test: str) -> Any:
        """Unused here; present so the hook check passes."""
        raise NotImplementedError


class _StubRobot:
    """A robot that records nothing and moves nothing."""

    def __init__(self, has_perception: bool = True) -> None:
        self.has_perception = has_perception
        self.dry = True


def _as_env(env: _StubEnv) -> PyBulletEnv:
    """The one cast these tests need.

    The executor declares a ``PyBulletEnv`` because that is where
    ``sync_to_state`` / ``gripper_joint_layout`` live. The stub provides
    them; what it does not provide is a physics client, which the
    executor never touches.
    """
    return cast(PyBulletEnv, env)


@pytest.fixture(name="recorder")
def recorder_fixture(monkeypatch):
    """Replace the two bridge helpers with recorders.

    Patched at the ``real_robot_executor`` module -- where the names
    were imported to -- so nothing reaches babyrobot.
    """

    class _Recorder:
        """Captures what was shipped and what came back."""

        def __init__(self):
            self.shipped = []
            self.observe_flags = []
            self.settle = []
            self.homed = []
            self.to_return = []

        def execute_chunks(self,
                           robot,
                           chunks,
                           layout,
                           observe=False,
                           settle_s=0.0):
            """Record one shipment; reply with the queued observations."""
            del robot, layout
            self.shipped.append([list(c) for c in chunks])
            self.observe_flags.append(observe)
            self.settle.append(settle_s)
            if not observe:
                return []
            return [self.to_return.pop(0) for _ in chunks if self.to_return]

        def reset_arm(self, robot, joints):
            """Record the homing request."""
            del robot
            self.homed.append(list(joints))
            return tuple(joints)

    rec = _Recorder()
    monkeypatch.setattr(
        "predicators.pybullet_helpers.real_robot_executor.execute_chunks",
        rec.execute_chunks)
    monkeypatch.setattr(
        "predicators.pybullet_helpers.real_robot_executor.reset_arm",
        rec.reset_arm)
    return rec


def _option(name="Pick"):
    """A grounded option, so actions can carry an option boundary."""
    param_opt = ParameterizedOption(name, [], Box(0, 1, (1, )),
                                    lambda s, m, o, p: Action(p),
                                    lambda s, m, o, p: True,
                                    lambda s, m, o, p: False)
    return param_opt.ground([], [0.5])


def _act(option: Any, terminal: bool = False) -> Action:
    """A joint-target action carrying ``option``.

    ``terminal`` decides whether this action ends its option, which is
    the boundary the executor ships on.
    """
    action = Action(np.zeros(9, dtype=np.float32))
    option.terminal = lambda _obs, _t=terminal: _t
    action.set_option(option)
    return action


def _executor(env, robot=None, **kwargs):
    """An executor over the stubs, with settings passed in, not configured."""
    return RealRobotExecutor(_as_env(env), robot or _StubRobot(), **{
        "settle_s": 0.25,
        **kwargs
    })


# -- optionality: this file must run without the private submodule -----------
def test_executor_has_no_module_level_babyrobot_import():
    """The executor reaches the robot only through the bridge helpers, so it
    imports with babyrobot absent -- and these tests must never skip."""
    source = inspect.getsourcefile(RealRobotExecutor)
    assert source is not None
    with open(source, encoding="utf-8") as f:
        tree = ast.parse(f.read())
    for node in tree.body:
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            continue
        for name in names:
            assert not name.startswith("babyrobot"), \
                f"babyrobot imported at module level: {name}"


# -- the buffer, on its own --------------------------------------------------
def test_buffer_returns_a_chunk_only_at_a_boundary():
    """Actions accumulate until the option ends; a partial option is not handed
    out, because the arm would execute half a skill."""
    buffer = OptionBoundaryBuffer()
    option = _option()

    assert buffer.add(_act(option), None) is None
    assert buffer.add(_act(option), None) is None
    chunk = buffer.add(_act(option, terminal=True), None)

    assert chunk is not None and len(chunk) == 3
    assert not buffer  # emptied by the handover


def test_buffer_ignores_actions_with_no_option():
    """An action carrying no option has no boundary to attribute it to, so it
    is never buffered and never shipped alone."""
    buffer = OptionBoundaryBuffer()
    assert buffer.add(Action(np.zeros(9, dtype=np.float32)), None) is None
    assert not buffer


def test_buffer_discard_reports_what_was_lost():
    """The count is what the caller warns with, so it has to be real."""
    buffer = OptionBoundaryBuffer()
    option = _option()
    buffer.add(_act(option), None)
    buffer.add(_act(option), None)

    assert buffer.discard() == 2
    assert buffer.discard() == 0


# -- construction ------------------------------------------------------------
def test_construction_names_a_missing_hook():
    """An env without the domain conversions is rejected by name, not by an
    AttributeError several hundred robot-moving actions later."""

    class _NoHooks(_StubEnv):  # pylint: disable=abstract-method
        """An env missing one required conversion."""
        state_from_observation = None  # type: ignore[assignment]

    with pytest.raises(TypeError) as exc:
        _executor(_NoHooks())
    assert "state_from_observation" in str(exc.value)
    # The hook that IS present is not named.
    assert "task_from_observation" not in str(exc.value)


def test_construction_rejects_observing_without_perception():
    """Asking to look at the bench with no cameras fails at construction,
    rather than raising inside the first option boundary."""
    with pytest.raises(ValueError) as exc:
        _executor(_StubEnv(), _StubRobot(has_perception=False))
    assert "perception" in str(exc.value)


def test_blind_run_needs_no_perception():
    """A blind open-loop run is supported, so a robot with no cameras is fine.

    -- provided nothing else asks to look, human resets included.
    """
    assert _executor(_StubEnv(),
                     _StubRobot(has_perception=False),
                     observe_at_boundaries=False,
                     human_reset=False) is not None


def test_human_reset_without_cameras_is_refused():
    """A human reset rebuilds the task from a look, so it cannot be honoured
    without perception.

    Better to say so at construction than to raise on the first task
    request, after the human has already been asked to stand by.
    """
    with pytest.raises(ValueError) as exc:
        _executor(_StubEnv(),
                  _StubRobot(has_perception=False),
                  observe_at_boundaries=False,
                  human_reset=True)
    assert "human reset" in str(exc.value)


# -- reset -------------------------------------------------------------------
def test_reset_homes_the_arm_to_the_twins_joints(recorder):
    """The arm is homed to the twin's home configuration with the finger joints
    dropped, because that is where the first option's waypoints start."""
    env = _StubEnv()
    env.set_observation(_state(0.0, joints=[float(i) for i in range(9)]))
    executor = _executor(env)

    executor.after_reset("test", 0, env.get_observation())

    # 9 joints in, 7 out: entries 7 and 8 are the fingers.
    assert recorder.homed == [[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]


def test_both_splits_execute(recorder):
    """Real mode is a property of an executor being attached, not of the split,
    so an exploration episode drives the arm like an evaluation one."""
    env = _StubEnv()
    executor = _executor(env)

    executor.after_reset("train", 0, env.get_observation())
    executor.after_reset("test", 0, env.get_observation())

    assert len(recorder.homed) == 2


def test_reset_drops_a_partial_option_from_the_previous_episode(
        recorder, caplog):
    """An episode cut short leaves half an option buffered.

    Half a skill is not worth executing on the arm, so it is dropped --
    loudly, because motion the caller asked for is silently not
    happening.
    """
    env = _StubEnv()
    executor = _executor(env)
    option = _option()
    executor.after_step(_act(option), env.get_observation())
    executor.after_step(_act(option), env.get_observation())
    assert recorder.shipped == []  # nothing shipped mid-option

    with caplog.at_level("WARNING"):
        executor.after_reset("test", 0, env.get_observation())

    assert "dropping 2 buffered action" in caplog.text
    assert recorder.shipped == []  # and still nothing shipped


# -- chunking ----------------------------------------------------------------
def test_ships_once_per_option_boundary(recorder):
    """Each option's actions go out when that option ends -- one shipment per
    boundary, carrying only that option's actions."""
    env = _StubEnv()
    executor = _executor(env)
    recorder.to_return = [_StubObservation(1.0), _StubObservation(2.0)]

    first, second = _option("Pick"), _option("Place")
    executor.after_step(_act(first), env.get_observation())
    executor.after_step(_act(first, terminal=True), env.get_observation())
    executor.after_step(_act(second), env.get_observation())
    executor.after_step(_act(second, terminal=True), env.get_observation())

    assert len(recorder.shipped) == 2
    assert [len(c) for chunks in recorder.shipped for c in chunks] == [2, 2]
    assert recorder.observe_flags == [True, True]
    assert recorder.settle == [0.25, 0.25]


def test_actions_without_an_option_are_not_shipped(recorder):
    """An optionless action rolls the twin forward without reaching the arm,
    and the observation passes through untouched."""
    env = _StubEnv()
    executor = _executor(env)

    obs = env.get_observation()
    returned = executor.after_step(Action(np.zeros(9, dtype=np.float32)), obs)

    assert recorder.shipped == []
    assert returned is obs


# -- twin re-sync ------------------------------------------------------------
def test_boundary_observation_is_written_into_the_twin(recorder):
    """The perceived state reaches the twin.

    Without this the correction would survive exactly one step: the env
    advances its own physics from its own bodies and returns a state
    derived from them.
    """
    env = _StubEnv()
    executor = _executor(env)
    recorder.to_return = [_StubObservation(0.5)]

    executor.after_step(_act(_option(), terminal=True), env.get_observation())

    assert len(env.synced) == 1
    assert env.synced[0].get(_BLOCK, "x") == pytest.approx(0.5)


def test_the_caller_gets_the_twin_not_the_observation(recorder):
    """The library's observation type never escapes: what comes back is the
    twin's State, so CogMan never learns a robot exists."""
    env = _StubEnv()
    executor = _executor(env)
    recorder.to_return = [_StubObservation(0.5)]

    returned = executor.after_step(_act(_option(), terminal=True),
                                   env.get_observation())

    assert isinstance(returned, State)
    assert not isinstance(returned, _StubObservation)
    # And it is the CORRECTED twin, not the pre-sync prediction.
    assert returned.get(_BLOCK, "x") == pytest.approx(0.5)


def test_no_sync_when_not_observing(recorder):
    """A blind run executes motion and never looks, so the twin is never
    corrected."""
    env = _StubEnv()
    executor = _executor(env,
                         _StubRobot(has_perception=False),
                         observe_at_boundaries=False,
                         human_reset=False)

    executor.after_step(_act(_option(), terminal=True), env.get_observation())

    assert recorder.observe_flags == [False]
    assert not env.synced


def test_large_divergence_is_surfaced(recorder, caplog):
    """A bench that disagrees with the twin is reported rather than swallowed.

    ``_set_state``'s own reconstruction check cannot catch this: it
    measures whether PyBullet could realize the state it was asked to
    write, and a toppled domino is perfectly realizable.
    """
    env = _StubEnv()
    executor = _executor(env, divergence_atol=0.02)
    recorder.to_return = [_StubObservation(0.5)]  # twin says 0.0

    with caplog.at_level("WARNING"):
        executor.after_step(_act(_option(), terminal=True),
                            env.get_observation())

    assert executor.last_divergence == pytest.approx(0.5)
    assert "0.500" in caplog.text


def test_small_divergence_is_quiet(recorder, caplog):
    """Placement jitter and perception noise are below the tolerance and must
    not cry wolf on every option."""
    env = _StubEnv()
    executor = _executor(env, divergence_atol=0.02)
    recorder.to_return = [_StubObservation(0.001)]

    with caplog.at_level("WARNING"):
        executor.after_step(_act(_option(), terminal=True),
                            env.get_observation())

    assert executor.last_divergence == pytest.approx(0.001)
    assert "real robot:" not in caplog.text
    # Quiet, but still corrected.
    assert len(env.synced) == 1


# -- attachment --------------------------------------------------------------
def test_attach_is_a_noop_when_not_executing():
    """The call site reads as one unconditional line, so a dry run must leave
    the env with no executor at all."""
    utils.reset_config({"env": "cover", "real_robot_execute": False})
    env = _StubEnv()
    assert attach_real_robot(cast(Any, env)) is None


def test_attach_rejects_a_non_pybullet_env():
    """The twin is what turns an option into a joint trajectory, so a non-
    PyBullet env cannot be driven and says so."""
    utils.reset_config({"env": "cover", "real_robot_execute": True})
    with pytest.raises(TypeError) as exc:
        attach_real_robot(cast(Any, object()), _StubRobot())
    assert "PyBullet" in str(exc.value)
