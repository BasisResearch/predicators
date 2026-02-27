"""High-level option-building helpers for agent-proposed options.

These are injected into the exec context for the ``propose_options`` tool,
giving the Claude agent convenient APIs for creating pybullet-compatible
ParameterizedOptions without needing to know about robot singletons, joint
positions, or IK details.

Usage (inside agent-proposed code)::

    # Move robot to a target pose computed from state/objects
    opt = make_move_to_pose_option(
        "MoveAbove", types=[_robot_type, _domino_type],
        params_space=Box(0, 1, (0,)),
        pose_fn=lambda state, objects, params: (
            state.get(objects[1], "x"),
            state.get(objects[1], "y"),
            0.3,   # z
            0, 0, 0  # roll, pitch, yaw
        ),
        finger_status="closed",
    )

    # Open fingers
    open_opt = make_finger_option(
        "Open", types=[_robot_type], params_space=Box(0, 1, (0,)),
        mode="open")

    # Chain them
    proposed_options = [chain_options("PickUp", [opt, open_opt])]
"""
import logging
from functools import lru_cache
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import pybullet as p
from gym.spaces import Box

from predicators import utils
from predicators.pybullet_helpers.geometry import Pose
from predicators.settings import CFG
from predicators.structs import Action, Array, Object, ParameterizedOption, \
    State, Type


def build_option_helpers(env_name: str) -> Dict[str, Any]:
    """Return a dict of helper name -> callable for the exec context.

    For pybullet environments the helpers wrap
    ``create_move_end_effector_to_pose_option`` and
    ``create_change_fingers_option`` so the agent can build options with a
    simplified API.  For all environments the generic helpers
    (``chain_options``) are always included.

    Args:
        env_name: Current environment name (e.g. ``"pybullet_domino"``).

    Returns:
        Dict mapping helper names to callables/classes.
    """
    helpers: Dict[str, Any] = {}

    # -- Generic helpers (always available) --------------------------------
    helpers["chain_options"] = _chain_options
    helpers["Pose"] = Pose

    # -- PyBullet helpers (only for pybullet envs) -------------------------
    if env_name.startswith("pybullet"):

        def _bound_move_to_pose(
                name: str,
                types: Sequence[Type],
                params_space: Box,
                pose_fn: Callable[[State, Sequence[Object], Array],
                                  Tuple[float, float, float, float, float,
                                        float]],
                finger_status: str = "closed",
                tol: float = 1e-4) -> ParameterizedOption:
            return _make_move_to_pose_option(name, types, params_space,
                                             pose_fn, finger_status, tol,
                                             env_name)

        def _bound_finger(name: str,
                          types: Sequence[Type],
                          params_space: Box,
                          mode: str = "open",
                          grasp_tol: Optional[float] = None
                          ) -> ParameterizedOption:
            return _make_finger_option(name, types, params_space, mode,
                                       grasp_tol, env_name)

        helpers["make_move_to_pose_option"] = _bound_move_to_pose
        helpers["make_finger_option"] = _bound_finger

    return helpers


# --------------------------------------------------------------------------- #
# Generic helpers
# --------------------------------------------------------------------------- #


def _chain_options(
        name: str,
        children: Sequence[ParameterizedOption]) -> ParameterizedOption:
    """Chain multiple options into a sequential composite option.

    Alias for ``utils.LinearChainParameterizedOption``.
    """
    return utils.LinearChainParameterizedOption(name, children)


# --------------------------------------------------------------------------- #
# PyBullet helpers
# --------------------------------------------------------------------------- #


@lru_cache(maxsize=1)
def _get_pybullet_env_cls(env_name: Optional[str] = None) -> Any:
    """Look up the concrete PyBulletEnv subclass by name.

    Args:
        env_name: Environment name (e.g. ``"pybullet_domino"``).
            Falls back to ``CFG.env`` if not provided.
    """
    # Trigger env submodule discovery so all subclasses are registered.
    import predicators.envs as _envs_pkg  # noqa: F401
    from predicators.envs.base_env import BaseEnv
    from predicators.envs.pybullet_env import PyBulletEnv
    if env_name is None:
        env_name = CFG.env
    for cls in utils.get_all_subclasses(BaseEnv):
        if not cls.__abstractmethods__ and cls.get_name() == env_name:
            if issubclass(cls, PyBulletEnv):
                return cls
            break
    raise RuntimeError(f"No PyBulletEnv subclass found for env '{env_name}'")


@lru_cache(maxsize=1)
def _get_robot(env_name: Optional[str] = None) -> Any:
    """Lazy-load the pybullet robot singleton for the given env."""
    env_cls = _get_pybullet_env_cls(env_name)
    _, robot, _ = env_cls.initialize_pybullet(using_gui=False)
    return robot


def _make_move_to_pose_option(
    name: str,
    types: Sequence[Type],
    params_space: Box,
    pose_fn: Callable[[State, Sequence[Object], Array],
                      Tuple[float, float, float, float, float, float]],
    finger_status: str = "closed",
    tol: float = 1e-4,
    env_name: Optional[str] = None,
) -> ParameterizedOption:
    """Create an option that moves the robot end-effector to a target pose.

    This is a simplified wrapper around
    ``create_move_end_effector_to_pose_option``.

    Args:
        name: Option name.
        types: Object types the option is parameterised over.
        params_space: Continuous parameter space.
        pose_fn: ``(state, objects, params) -> (x, y, z, roll, pitch, yaw)``
            that computes the target end-effector pose.
        finger_status: ``"open"`` or ``"closed"`` — the desired finger state
            during motion.
        tol: Squared-distance tolerance for the terminal condition.
        env_name: Environment name for robot lookup.

    Returns:
        A ``ParameterizedOption`` that moves the EE to the target pose.
    """
    from predicators.pybullet_helpers.controllers import \
        create_move_end_effector_to_pose_option

    robot = _get_robot(env_name)

    def _get_current_and_target(state: State, objects: Sequence[Object],
                               params: Array) -> Tuple[Pose, Pose, str]:
        # Current pose from state
        robot_obj = None
        for obj in objects:
            if obj.type.name == "robot":
                robot_obj = obj
                break
        if robot_obj is None:
            robot_obj = state.get_objects(
                next(t for t in {o.type
                                 for o in objects} if t.name == "robot"))[0]

        cur_pos = (state.get(robot_obj,
                             "x"), state.get(robot_obj,
                                             "y"), state.get(robot_obj, "z"))
        cur_orn = p.getQuaternionFromEuler(
            [0, state.get(robot_obj, "tilt"),
             state.get(robot_obj, "wrist")])
        current_pose = Pose(cur_pos, cur_orn)

        # Target pose from user function
        x, y, z, roll, pitch, yaw = pose_fn(state, objects, params)
        target_orn = p.getQuaternionFromEuler([roll, pitch, yaw])
        target_pose = Pose((x, y, z), target_orn)

        return current_pose, target_pose, finger_status

    return create_move_end_effector_to_pose_option(
        robot,
        name,
        types,
        params_space,
        _get_current_and_target,
        tol,
        CFG.pybullet_max_vel_norm,
        1e-3,  # finger_action_nudge_magnitude
        validate=CFG.pybullet_ik_validate,
    )


def _make_finger_option(
    name: str,
    types: Sequence[Type],
    params_space: Box,
    mode: str = "open",
    grasp_tol: Optional[float] = None,
    env_name: Optional[str] = None,
) -> ParameterizedOption:
    """Create an option that opens or closes the robot fingers.

    Args:
        name: Option name.
        types: Object types the option is parameterised over.
        params_space: Continuous parameter space.
        mode: ``"open"`` or ``"close"``.
        grasp_tol: Tolerance for the terminal condition.  Defaults to
            ``PyBulletEnv.grasp_tol_small``.
        env_name: Environment name for robot/env-class lookup.

    Returns:
        A ``ParameterizedOption`` for finger control.
    """
    from predicators.envs.pybullet_env import PyBulletEnv
    from predicators.pybullet_helpers.controllers import \
        create_change_fingers_option

    robot = _get_robot(env_name)
    env_cls = _get_pybullet_env_cls(env_name)

    if grasp_tol is None:
        grasp_tol = PyBulletEnv.grasp_tol_small

    def _get_current_fingers(state: State) -> float:
        robot_type = next(t for t in {o.type
                                      for o in list(state)}
                          if t.name == "robot")
        robot_obj, = state.get_objects(robot_type)
        return env_cls._fingers_state_to_joint(robot,
                                               state.get(robot_obj, "fingers"))

    def _get_current_and_target(state: State, objects: Sequence[Object],
                               params: Array) -> Tuple[float, float]:
        current = _get_current_fingers(state)
        if mode == "open":
            target = robot.open_fingers - 0.01
        else:
            target = robot.closed_fingers - 0.01
        return current, target

    return create_change_fingers_option(
        robot,
        name,
        types,
        params_space,
        _get_current_and_target,
        CFG.pybullet_max_vel_norm,
        grasp_tol,
    )
