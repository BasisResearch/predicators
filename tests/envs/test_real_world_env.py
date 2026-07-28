"""Tests for RealWorldEnv: chunking, twin re-sync, and what the agent sees.

These run against a **stub** robot and a **stub** inner env, and must never
skip: a suite that silently skipped without the private submodule would hide
exactly the regressions it exists to catch. That is asserted below.

The wrapper is deliberately reachable without babyrobot because it only ever
touches the robot through ``real_robot_bridge.execute_chunks`` /
``reset_arm``, which these tests replace with recorders. Building the
``Segment`` objects those helpers ship is babyrobot's contract and is covered
in ``tests/pybullet_helpers/test_real_robot_bridge.py``, which does skip.
"""
import ast
import inspect
from typing import Any, List, Optional, cast

import numpy as np
import pytest
from gym.spaces import Box

from predicators import utils
from predicators.envs.pybullet_env import PyBulletEnv
from predicators.envs.real_world_env import RealWorldEnv, wrap_for_real_robot
from predicators.pybullet_helpers.real_robot_bridge import GripperJointLayout
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

    Opaque to the wrapper: it is handed straight to the inner env's
    state_from_observation.
    """

    def __init__(self, x: float) -> None:
        self.x = x


class _StubInnerEnv:
    """A twin with no PyBullet: it records what was asked of it.

    Implements only what the wrapper touches. It is passed where a
    ``PyBulletEnv`` is declared, via one cast at each construction site
    -- keeping these tests free of a real physics client, which is what
    lets them cover the wrapper's own logic rather than PyBullet's.
    """

    def __init__(self) -> None:
        self.using_gui = False
        self.synced: List[State] = []
        self.stepped: List[Action] = []
        self.reset_calls: List[tuple] = []
        self._observation = _state(0.0)
        # Set by tests to make an option terminate.
        self.terminal_after: Optional[int] = None

    @classmethod
    def get_name(cls):
        """The name the wrapper adopts for its instance lookups."""
        return "stub_inner_env"

    # -- the twin's own behavior ----------------------------------------
    def reset(self, train_or_test, task_idx, render=False):
        """Record the reset and hand back the current observation."""
        self.reset_calls.append((train_or_test, task_idx, render))
        return self._observation

    def step(self, action, render_obs=False):
        """Record the action; the twin's state is unchanged by default."""
        del render_obs
        self.stepped.append(action)
        return self._observation

    def get_observation(self):
        """The twin's current state."""
        return self._observation

    def set_observation(self, state):
        """Put the twin in a given state, as a reset would."""
        self._observation = state

    # -- the hooks the wrapper requires ---------------------------------
    def sync_to_state(self, state):
        """Adopt ``state`` as the twin's world, as PyBullet's would."""
        self.synced.append(state)
        self._observation = state

    def gripper_joint_layout(self):
        """The finger layout the splitter needs."""
        return _LAYOUT

    def state_from_observation(self, obs, prev_state):
        """Move the block to wherever the observation says it is."""
        del prev_state
        return _state(obs.x)

    def task_from_observation(self, obs, train_or_test):
        """Unused here; present so the hook check passes."""
        raise NotImplementedError


class _StubRobot:
    """A robot that records chunks instead of moving an arm."""

    def __init__(self, has_perception: bool = True) -> None:
        self.has_perception = has_perception
        self.dry = True
        self.homed: List[List[float]] = []


def _inner_as_env(inner: _StubInnerEnv) -> PyBulletEnv:
    """The one cast these tests need.

    ``RealWorldEnv`` declares a ``PyBulletEnv`` because that is where
    ``sync_to_state`` / ``gripper_joint_layout`` and the render
    arguments live. The stub provides all of them; what it does not
    provide is a physics client, which the wrapper never touches.
    """
    return cast(PyBulletEnv, inner)


@pytest.fixture(name="recorder")
def recorder_fixture(monkeypatch):
    """Replace the two bridge helpers with recorders.

    Patching at the ``real_world_env`` module -- where the names were
    imported to -- so nothing reaches babyrobot.
    """

    class _Recorder:
        """Captures the chunks shipped and the observations handed back."""

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
            """Record one shipment and reply with the queued observations."""
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
    monkeypatch.setattr("predicators.envs.real_world_env.execute_chunks",
                        rec.execute_chunks)
    monkeypatch.setattr("predicators.envs.real_world_env.reset_arm",
                        rec.reset_arm)
    return rec


def _config(**overrides):
    """The real-execution config, with per-test overrides."""
    flags = {
        "env": "pybullet_domino_real",
        "real_robot_execute": True,
        "real_robot_ship_whole_episode": False,
        "real_robot_observe_at_option_boundary": True,
        "real_robot_settle_s": 0.25,
        "real_robot_divergence_atol": 0.02,
    }
    flags.update(overrides)
    utils.reset_config(flags)


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
    the boundary the wrapper ships on.
    """
    action = Action(np.zeros(9, dtype=np.float32))
    option.terminal = lambda _obs, _t=terminal: _t
    action.set_option(option)
    return action


def _wrap(inner, robot, **overrides):
    """Build a wrapper around the stubs under the given config."""
    _config(**overrides)
    return RealWorldEnv(_inner_as_env(inner), robot)


# -- optionality: this file must run without the private submodule -----------
def test_wrapper_has_no_module_level_babyrobot_import():
    """The wrapper reaches the robot only through the bridge helpers, so it
    imports with babyrobot absent -- and these tests must never skip."""
    source = inspect.getsourcefile(RealWorldEnv)
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


# -- construction ------------------------------------------------------------
def test_construction_names_a_missing_hook():
    """An env without the domain conversions is rejected by name, not by an
    AttributeError several hundred robot-moving actions later."""
    _config()

    class _NoHooks(_StubInnerEnv):  # pylint: disable=abstract-method
        """An inner env missing one required conversion."""
        state_from_observation = None  # type: ignore[assignment]

    with pytest.raises(TypeError) as exc:
        RealWorldEnv(_inner_as_env(_NoHooks()), _StubRobot())
    assert "state_from_observation" in str(exc.value)
    # The hook that IS present is not named.
    assert "task_from_observation" not in str(exc.value)


def test_construction_rejects_observing_without_perception():
    """Asking to look at the bench with no cameras fails at construction,
    rather than raising inside the first option boundary."""
    _config()
    with pytest.raises(ValueError) as exc:
        RealWorldEnv(_inner_as_env(_StubInnerEnv()),
                     _StubRobot(has_perception=False))
    assert "perception" in str(exc.value)


def test_blind_run_needs_no_perception():
    """Turning the look off is a supported (open-loop) mode, so a robot without
    cameras is fine there."""
    _config(real_robot_observe_at_option_boundary=False)
    env = RealWorldEnv(_inner_as_env(_StubInnerEnv()),
                       _StubRobot(has_perception=False))
    assert env is not None


def test_whole_episode_mode_needs_no_perception():
    """Whole-episode shipping has no mid-episode boundary to look at, so the
    observe flag is moot there and must not demand cameras."""
    _config(real_robot_ship_whole_episode=True)
    env = RealWorldEnv(_inner_as_env(_StubInnerEnv()),
                       _StubRobot(has_perception=False))
    assert env is not None


# -- reset -------------------------------------------------------------------
def test_reset_homes_the_arm_to_the_twins_joints(recorder):
    """The arm is homed to the twin's home configuration with the finger joints
    dropped, because that is where the first option's waypoints start."""
    inner = _StubInnerEnv()
    inner.set_observation(_state(0.0, joints=[float(i) for i in range(9)]))
    env = _wrap(inner, _StubRobot())

    env.reset("test", 0)

    assert inner.reset_calls == [("test", 0, False)]
    # 9 joints in, 7 out: entries 7 and 8 are the fingers.
    assert recorder.homed == [[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]


def test_reset_does_not_touch_the_arm_when_not_executing(recorder):
    """real_robot_execute off means the wrapper is inert, even if one was built
    by hand."""
    inner = _StubInnerEnv()
    env = _wrap(inner, _StubRobot(), real_robot_execute=False)

    env.reset("test", 0)

    assert recorder.homed == []
    assert inner.reset_calls == [("test", 0, False)]


def test_both_splits_execute(recorder):
    """Real mode is a property of being wrapped, not of the split, so an
    exploration episode drives the arm exactly like an evaluation one."""
    inner = _StubInnerEnv()
    env = _wrap(inner, _StubRobot())

    env.reset("train", 0)

    assert len(recorder.homed) == 1


# -- chunking ----------------------------------------------------------------
def test_ships_once_per_option_boundary(recorder):
    """Each option's actions go out when that option ends -- one shipment per
    boundary, carrying only that option's actions and nothing from the option
    before it."""
    inner = _StubInnerEnv()
    env = _wrap(inner, _StubRobot())
    recorder.to_return = [_StubObservation(1.0), _StubObservation(2.0)]

    first, second = _option("Pick"), _option("Place")
    env.step(_act(first))
    env.step(_act(first, terminal=True))
    env.step(_act(second))
    env.step(_act(second, terminal=True))

    assert len(recorder.shipped) == 2
    assert [len(chunk) for chunks in recorder.shipped for chunk in chunks] \
        == [2, 2]
    assert recorder.observe_flags == [True, True]
    assert recorder.settle == [0.25, 0.25]


def test_nothing_ships_mid_option(recorder):
    """Actions accumulate until the option ends; a partial option is not
    shipped, because the arm would execute half a skill."""
    inner = _StubInnerEnv()
    env = _wrap(inner, _StubRobot())

    option = _option()
    env.step(_act(option))
    env.step(_act(option))

    assert recorder.shipped == []


def test_whole_episode_mode_ships_one_chunk_on_flush(recorder):
    """The degenerate open-loop case: nothing goes out at boundaries, and the
    flush ships the entire episode as ONE chunk with no look."""
    inner = _StubInnerEnv()
    env = _wrap(inner, _StubRobot(), real_robot_ship_whole_episode=True)

    first, second = _option("Pick"), _option("Place")
    env.step(_act(first, terminal=True))
    env.step(_act(second, terminal=True))
    assert recorder.shipped == []

    env.flush_real_execution()

    assert len(recorder.shipped) == 1
    assert len(recorder.shipped[0]) == 1  # one chunk, not one per option
    assert len(recorder.shipped[0][0]) == 2
    assert recorder.observe_flags == [False]


def test_flush_is_a_noop_with_an_empty_buffer(recorder):
    """Flushing twice, or with nothing buffered, ships nothing."""
    inner = _StubInnerEnv()
    env = _wrap(inner, _StubRobot(), real_robot_ship_whole_episode=True)

    env.flush_real_execution()
    env.step(_act(_option(), terminal=True))
    env.flush_real_execution()
    env.flush_real_execution()

    assert len(recorder.shipped) == 1


def test_actions_without_an_option_are_not_shipped(recorder):
    """An action carrying no option has no boundary to attribute it to, so it
    rolls the twin forward without reaching the arm."""
    inner = _StubInnerEnv()
    env = _wrap(inner, _StubRobot())

    env.step(Action(np.zeros(9, dtype=np.float32)))

    assert recorder.shipped == []
    assert len(inner.stepped) == 1


# -- twin re-sync ------------------------------------------------------------
def test_boundary_observation_is_written_into_the_twin(recorder):
    """The perceived state reaches the twin.

    Without this the correction would survive exactly one step: env.step
    advances the twin's own physics from its own bodies and returns a
    state derived from them.
    """
    inner = _StubInnerEnv()
    env = _wrap(inner, _StubRobot())
    recorder.to_return = [_StubObservation(0.5)]

    option = _option()
    env.step(_act(option, terminal=True))

    assert len(inner.synced) == 1
    assert inner.synced[0].get(_BLOCK, "x") == pytest.approx(0.5)


def test_the_agent_sees_the_twin_not_the_observation(recorder):
    """The library's observation type never escapes the wrapper: what step
    returns is the twin's State, so CogMan never learns a robot exists."""
    inner = _StubInnerEnv()
    env = _wrap(inner, _StubRobot())
    recorder.to_return = [_StubObservation(0.5)]

    returned = env.step(_act(_option(), terminal=True))

    assert isinstance(returned, State)
    assert not isinstance(returned, _StubObservation)
    # And it is the CORRECTED twin, not the pre-sync prediction.
    assert returned.get(_BLOCK, "x") == pytest.approx(0.5)


def test_no_sync_when_not_observing(recorder):
    """A blind run executes motion and never looks, so the twin is never
    corrected."""
    inner = _StubInnerEnv()
    env = _wrap(inner,
                _StubRobot(has_perception=False),
                real_robot_observe_at_option_boundary=False)

    env.step(_act(_option(), terminal=True))

    assert recorder.observe_flags == [False]
    assert not inner.synced


def test_large_divergence_is_surfaced(recorder, caplog):
    """A bench that disagrees with the twin is reported rather than swallowed.

    _set_state's own reconstruction check cannot catch this: it measures
    whether PyBullet could realize the state it was asked to write, and
    a toppled domino is perfectly realizable.
    """
    inner = _StubInnerEnv()
    env = _wrap(inner, _StubRobot())
    recorder.to_return = [_StubObservation(0.5)]  # twin says 0.0

    with caplog.at_level("WARNING"):
        env.step(_act(_option(), terminal=True))

    assert env.last_divergence == pytest.approx(0.5)
    assert "0.500" in caplog.text


def test_small_divergence_is_quiet(recorder, caplog):
    """Placement jitter and perception noise are below the tolerance and must
    not cry wolf on every option."""
    inner = _StubInnerEnv()
    env = _wrap(inner, _StubRobot())
    recorder.to_return = [_StubObservation(0.001)]

    with caplog.at_level("WARNING"):
        env.step(_act(_option(), terminal=True))

    assert env.last_divergence == pytest.approx(0.001)
    assert "real_world_env" not in caplog.text
    # Quiet, but still corrected.
    assert len(inner.synced) == 1


# -- wiring ------------------------------------------------------------------
def test_wrap_is_a_noop_when_not_executing():
    """The call site reads as one unconditional line, so a dry run must get the
    env straight back."""
    _config(real_robot_execute=False)
    inner = _StubInnerEnv()
    assert wrap_for_real_robot(_inner_as_env(inner)) is inner


def test_wrap_rejects_a_non_pybullet_env():
    """The twin is what turns an option into a joint trajectory, so a non-
    PyBullet env cannot be driven and says so."""
    _config(real_robot_execute=True)

    with pytest.raises(TypeError) as exc:
        wrap_for_real_robot(cast(Any, object()), _StubRobot())
    assert "PyBullet" in str(exc.value)
