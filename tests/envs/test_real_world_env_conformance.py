"""RealWorldEnv's conformance to the BaseEnv contract.

Adding a concrete ``BaseEnv`` subclass is not free: five places pick an
environment by scanning ``utils.get_all_subclasses(BaseEnv)`` and calling
``cls.get_name()`` on every concrete one. This file pins the two consequences
-- the class-level name must stay a working classmethod nobody asks for, and
the instance-level name must be the *inner* env's -- plus the rule that the
planner never gets handed a robot.

Like the wrapper's own tests, these run without babyrobot and must not skip.
"""
from typing import Any, cast

import pytest

from predicators import utils
from predicators.envs import BaseEnv, create_new_env
from predicators.envs.pybullet_domino_real import PyBulletDominoRealEnv
from predicators.envs.pybullet_env import PyBulletEnv
from predicators.envs.real_world_env import RealWorldEnv, wrap_for_real_robot
from predicators.option_model import create_option_model
from predicators.pybullet_helpers.real_robot_bridge import GripperJointLayout
from predicators.structs import State

_LAYOUT = GripperJointLayout(left_finger_joint_idx=7,
                             right_finger_joint_idx=8,
                             open_fingers=0.04,
                             closed_fingers=0.0)
_PREDICATE_SENTINEL = {"a predicate set"}
_TYPE_SENTINEL = {"a type set"}
_ACTION_SPACE_SENTINEL = "an action space"


class _StubInnerEnv:
    """An inner env whose delegated members are distinguishable sentinels."""

    def __init__(self) -> None:
        self.using_gui = False
        self.action_space = _ACTION_SPACE_SENTINEL
        self.predicates = _PREDICATE_SENTINEL
        self.types = _TYPE_SENTINEL
        self.goal_predicates = _PREDICATE_SENTINEL
        self.options = "an option set"
        self.something_nobody_declared = "delegated anyway"

    @classmethod
    def get_name(cls) -> str:
        """The name a wrapped run must still report."""
        return "pybullet_domino_real"

    def sync_to_state(self, state: State) -> None:
        """Unused here."""

    def gripper_joint_layout(self) -> GripperJointLayout:
        """Unused here."""
        return _LAYOUT

    def state_from_observation(self, obs: Any, prev_state: State) -> State:
        """Unused here."""
        del obs
        return prev_state

    def task_from_observation(self, obs: Any, train_or_test: str) -> Any:
        """Unused here."""
        raise NotImplementedError


class _StubRobot:
    """A robot that is never asked to do anything in this file."""
    has_perception = True
    dry = True


def _wrapped() -> RealWorldEnv:
    """A wrapper around the sentinel stub."""
    utils.reset_config({
        "env": "pybullet_domino_real",
        "real_robot_execute": True,
        "real_robot_ship_whole_episode": True,
    })
    return RealWorldEnv(cast(PyBulletEnv, _StubInnerEnv()), _StubRobot())


# -- the two get_name consequences -------------------------------------------
def test_class_level_get_name_is_a_working_classmethod():
    """It must return *something* without an instance: the registry scans call
    it on every concrete subclass, and an exception here would break env
    creation for every environment, not just this one."""
    assert RealWorldEnv.get_name() == "real_world_env"


def test_the_sentinel_name_cannot_be_built_as_a_standalone_env():
    """The scans DO see the sentinel -- the name matches and the class is
    concrete -- so ``--env real_world_env`` reaches the constructor and fails
    there on the missing robot, rather than being skipped over.

    That is acceptable because the wrapper is not a standalone env and no
    config names it, but it is worth pinning: the failure must stay loud and
    immediate. If someone ever needs this to be a friendlier message, the fix
    is here and not in the scans.
    """
    with pytest.raises(TypeError):
        create_new_env("real_world_env")


def test_instance_get_name_is_the_inner_envs():
    """~25 call sites feed env.get_name() to get_gt_options / get_gt_nsrts.

    A wrapped domino run is still a domino run, so they must get the
    inner name.
    """
    env = _wrapped()
    assert env.get_name() == "pybullet_domino_real"
    # ...while the class-level classmethod is untouched by the shadowing.
    assert RealWorldEnv.get_name() == "real_world_env"


def test_every_registry_scan_still_resolves_the_real_env():
    """All five scans share this shape.

    With the wrapper importable, each must still land on the real class
    -- three filter on PyBulletEnv, which the wrapper is not, and the
    other two on the name, which it does not claim.
    """
    concrete = [
        cls for cls in utils.get_all_subclasses(BaseEnv)
        if not cls.__abstractmethods__
    ]
    assert RealWorldEnv in concrete, "the wrapper is concrete and in the scan"

    # The name-only scan shape (envs/__init__, ground_truth_models/domino).
    by_name = [c for c in concrete if c.get_name() == "pybullet_domino_real"]
    assert by_name == [PyBulletDominoRealEnv]

    # The PyBulletEnv-filtered scan shape (the other three).
    pybullet_only = [
        c for c in concrete if issubclass(c, PyBulletEnv)
        and c.get_name() == "pybullet_domino_real"
    ]
    assert pybullet_only == [PyBulletDominoRealEnv]
    assert not issubclass(RealWorldEnv, PyBulletEnv)


def test_create_new_env_still_builds_the_real_env():
    """The end-to-end version of the scan test: --env pybullet_domino_real must
    not start resolving to the wrapper."""
    utils.reset_config({"env": "pybullet_domino_real"})
    for cls in utils.get_all_subclasses(BaseEnv):
        if not cls.__abstractmethods__ and \
                cls.get_name() == "pybullet_domino_real":
            assert cls is PyBulletDominoRealEnv
            break
    else:
        pytest.fail("pybullet_domino_real no longer resolves")


# -- delegation --------------------------------------------------------------
def test_getattr_delegates_the_long_tail():
    """predicates / types / options / action_space and anything else the
    wrapper does not override come from the inner env."""
    env = _wrapped()
    assert env.predicates is _PREDICATE_SENTINEL
    assert env.goal_predicates is _PREDICATE_SENTINEL
    assert env.types is _TYPE_SENTINEL
    assert env.action_space is _ACTION_SPACE_SENTINEL
    assert env.options == "an option set"
    # Not declared anywhere on the wrapper -- __getattr__ still finds it.
    assert env.something_nobody_declared == "delegated anyway"


def test_missing_attributes_still_raise():
    """Delegation must not turn every typo into a silent success."""
    env = _wrapped()
    with pytest.raises(AttributeError):
        _ = env.no_such_attribute


# -- the planner must never touch the robot ----------------------------------
def test_option_model_env_is_not_wrapped(monkeypatch):
    """`create_option_model` builds the planner's own private simulator via
    create_new_env. That env must never be wrapped: wrapping there would give
    the *planner* a RealRobot -- a second worker thread that could home the arm
    in the middle of a search.

    Cheap env on purpose; the property under test is the factory's, not
    any one environment's.
    """
    utils.reset_config({
        "env": "cover",
        "real_robot_execute": True,
        "option_model_name": "oracle",
    })

    def _explode(*args: Any, **kwargs: Any) -> Any:
        """Any attempt to build a robot during planning is the bug."""
        raise AssertionError("the planner constructed a RealRobot")

    monkeypatch.setattr("predicators.envs.real_world_env.make_real_robot",
                        _explode)

    model = create_option_model("oracle")

    assert not isinstance(model.sim_env, RealWorldEnv)


def test_option_model_is_unwrapped_even_though_wrapping_works():
    """The counterpart to the test above, so it is detecting a real distinction
    rather than a disabled feature.

    The positive path -- wrap_for_real_robot actually returning a
    wrapper -- needs a genuine PyBullet twin to pass its isinstance
    check, so it is covered in ``test_real_world_env_integration.py``
    against the real domino env. Here we only pin that the planner's env
    is not one.
    """
    utils.reset_config({
        "env": "cover",
        "real_robot_execute": True,
        "option_model_name": "oracle",
    })
    model = create_option_model("oracle")
    assert not isinstance(model.sim_env, RealWorldEnv)
    # And the wrapper would have rejected it anyway: a cover env is not a
    # PyBullet twin, so nothing here could silently acquire a robot.
    with pytest.raises(TypeError):
        wrap_for_real_robot(model.sim_env)


# -- BaseEnv surface ---------------------------------------------------------
def test_wrapper_is_concrete():
    """It has to be instantiable; an abstract method left unimplemented would
    make construction fail at runtime, not at type-check time."""
    assert not RealWorldEnv.__abstractmethods__


def test_wrapper_is_a_base_env():
    """Everything downstream is typed against BaseEnv."""
    assert issubclass(RealWorldEnv, BaseEnv)
    assert isinstance(_wrapped(), BaseEnv)
