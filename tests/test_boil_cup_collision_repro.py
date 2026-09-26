"""Repro for SwitchBurnerOn/Waypoint_1 cup-collision regression.

Reproduces the failure observed at
logs/.../run_20260512_210304/info.log:1102:
  ERROR: [SwitchBurnerOn/Waypoint_1] GOAL ROBOT collision with body 4 (cup)

Cycle 0, attempt 2 placed the jug on the burner at
(target_x=0.5313, target_y=1.2899, release_z=0.5659, yaw=2.5974) and
then called SwitchBurnerOn(...)[0.0413, 0.1016]. BiRRT's IK goal pose at
Waypoint_1 collided with the just-placed jug (URDF named "cup"). This
test sets the same scenario directly and verifies the option no longer
fails with that collision.
"""
# pylint: disable=protected-access,import-outside-toplevel
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from predicators import utils
from predicators.envs import _MOST_RECENT_ENV_INSTANCE
from predicators.envs.pybullet_boil import PyBulletBoilEnv
from predicators.ground_truth_models import get_gt_options
from predicators.structs import DefaultEnvironmentTask


class _ExposedBoilEnv(PyBulletBoilEnv):
    """Boil env exposed with set_state / execute_option for tests."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        _MOST_RECENT_ENV_INSTANCE[self.get_name()] = self

    def set_state(self, state: Any) -> None:
        """Reset env to *state*, assuming robot is at its home joint config."""
        robot = self._pybullet_robot
        joint_positions = list(robot.initial_joint_positions)
        state_with_sim = utils.PyBulletState(state.data,
                                             simulator_state=joint_positions)
        self._current_observation = state_with_sim
        self._current_task = DefaultEnvironmentTask
        self._set_state(state_with_sim)

    def execute_option(self, option: Any, max_steps: int = 300) -> Any:
        """Run option loop up to *max_steps*; return final state."""
        cur = self._current_state
        assert option.initiable(cur)
        for _ in range(max_steps):
            if option.terminal(cur):
                break
            action = option.policy(cur)
            self.step(action)
            cur = self._current_state
        return self._current_state.copy()


def test_switch_burner_on_after_place_at_attempt2_pose(caplog):
    """Reproduce Cycle 0 attempt 2 end-to-end: pick the jug, place it on the
    burner at the attempt-2 Place params, then run SwitchBurnerOn.

    Under the May 2026 handle-targeting Place the jug landed where it
    blocked the press. Since Place targets the jug's centre, the jug has
    to land there, clear of the press. The planned drop goal once kept
    the pick yaw's grasp offset: the planner saw the held jug inside the
    burner switch and refused, and SwitchBurnerOn then moved with the
    jug still in hand.
    """
    utils.reset_config({
        "env": "pybullet_boil",
        "use_gui": False,
        "pybullet_control_mode": "reset",
        "pybullet_robot": "fetch",
        "boil_use_skill_factories": True,
        "boil_num_jugs_train": [1],
        "boil_num_jugs_test": [1],
        "boil_num_burner_train": [1],
        "boil_num_burner_test": [1],
        "skill_phase_use_motion_planning": True,
        "pybullet_ik_validate": False,
        "seed": 0,
    })
    env = _ExposedBoilEnv(use_gui=False)
    options = {o.name: o for o in get_gt_options(env.get_name())}

    jug = env._jugs[0]
    burner = env._burners[0]
    robot = env._robot

    # Start from the default train-task init state.
    init_state = env.get_train_tasks()[0].init
    env.set_state(init_state)

    caplog.set_level(logging.ERROR)

    # 1) Pick the jug (any grasp z works for the geometry test).
    env.execute_option(options["PickJug"].ground([robot, jug],
                                                 np.array([0.01],
                                                          dtype=np.float32)))

    # 2) Place at the attempt-2 coordinates that produced the failure.
    placed = env.execute_option(options["Place"].ground(
        [robot], np.array([0.5313, 1.2899, 0.5659, 2.5974], dtype=np.float32)))
    assert placed.get(jug, "is_held") < 0.5
    assert np.hypot(
        placed.get(jug, "x") - 0.5313,
        placed.get(jug, "y") - 1.2899) < 0.01

    # 3) SwitchBurnerOn with the same params the failing run used.
    opt = options["SwitchBurnerOn"].ground([robot, burner],
                                           np.array([0.0413, 0.1016],
                                                    dtype=np.float32))
    final = env.execute_option(opt, max_steps=200)
    assert final is not None

    # The bug surfaced as an ERROR log; assert it didn't reappear.
    collision_errors = [
        rec for rec in caplog.records if rec.levelno >= logging.ERROR
        and "GOAL ROBOT collision" in rec.message and "cup" in rec.message
    ]
    assert not collision_errors, (
        f"SwitchBurnerOn produced cup-collision errors: "
        f"{[r.message for r in collision_errors]}")


def test_full_attempt2_sequence_refinement_vs_execution(caplog):
    """Run the entire Cycle 0 attempt-2 sequence (all 7 prior options +
    SwitchBurnerOn) and verify option_model and env.step agree. This matches
    the planning-sim's accumulated state at the original failure point.

    Expected: both option_model and execution reach SwitchBurnerOn with
    similar post-Place state and produce the same outcome (succeed
    together or fail together). Anything else is the divergence that
    let refinement lie about feasibility.
    """
    from predicators.option_model import _OracleOptionModel

    utils.reset_config({
        "env": "pybullet_boil",
        "use_gui": False,
        "pybullet_control_mode": "reset",
        "pybullet_robot": "fetch",
        "boil_use_skill_factories": True,
        "boil_num_jugs_train": [1],
        "boil_num_jugs_test": [1],
        "boil_num_burner_train": [1],
        "boil_num_burner_test": [1],
        "skill_phase_use_motion_planning": True,
        "pybullet_ik_validate": False,
        "option_model_terminate_on_repeat": False,
        "seed": 0,
    })

    # Attempt-2 plan parameters straight from info.log:960-970.
    attempt2_plan = [
        ("PickJug", [0.0262]),
        ("Place", [1.0138, 1.4008, 0.5790, -1.9641]),
        ("SwitchFaucetOn", [0.0511, 0.0978]),
        ("Wait", []),
        ("SwitchFaucetOff", [0.0547, 0.1037]),
        ("PickJug", [0.0041]),
        ("Place", [0.5313, 1.2899, 0.5659, 2.5974]),
        ("SwitchBurnerOn", [0.0413, 0.1016]),
    ]

    def _run(via_option_model: bool):
        """Run the plan; return (last successful step, failure reason)."""
        env = _ExposedBoilEnv(use_gui=False)
        options = {o.name: o for o in get_gt_options(env.get_name())}
        jug = env._jugs[0]
        burner = env._burners[0]
        faucet = env._faucet
        robot = env._robot
        env.set_state(env.get_train_tasks()[0].init)

        if via_option_model:
            option_model = _OracleOptionModel(set(options.values()),
                                              env.simulate)
        state = env._current_observation
        for i, (name, params) in enumerate(attempt2_plan):
            if name == "PickJug":
                objs = [robot, jug]
            elif name in ("SwitchFaucetOn", "SwitchFaucetOff"):
                objs = [robot, faucet]
            elif name == "SwitchBurnerOn":
                objs = [robot, burner]
            elif name == "Place":
                objs = [robot]
            elif name == "Wait":
                objs = [robot]
            else:
                raise ValueError(name)
            opt = options[name].ground(objs, np.array(params,
                                                      dtype=np.float32))
            try:
                if via_option_model:
                    state, na = (option_model.get_next_state_and_num_actions(
                        state, opt))
                    if na == 0:
                        return i, option_model.last_execution_failure
                else:
                    if not opt.initiable(state):
                        return i, "not initiable"
                    final = env.execute_option(opt, max_steps=400)
                    state = final
            except Exception as e:  # pylint: disable=broad-except
                return i, str(e)
        return len(attempt2_plan), None

    caplog.set_level(logging.ERROR)
    om_step, om_reason = _run(via_option_model=True)
    exec_step, exec_reason = _run(via_option_model=False)

    # Both paths must agree on where the plan first fails (if at all).
    assert om_step == exec_step, (
        f"option_model and execution diverged: option_model stopped at "
        f"step {om_step} (reason={om_reason!r}); execution stopped at "
        f"step {exec_step} (reason={exec_reason!r}).")


def test_option_model_and_execution_agree_on_attempt2_place(caplog):
    """Refinement and execution should agree on the attempt-2 Place: the
    option-model rollout used by refinement and the executed option both put
    the jug on its target.

    The original bug: refinement said the plan was feasible, but
    execution hit a cup collision. Place now targets the jug's centre,
    with the grasp offset turned to the place yaw (measured at the pick
    yaw, it put the planner's held jug inside the burner switch, and
    both paths refused a placement the jug clears).
    """
    from predicators.option_model import _OracleOptionModel

    utils.reset_config({
        "env": "pybullet_boil",
        "use_gui": False,
        "pybullet_control_mode": "reset",
        "pybullet_robot": "fetch",
        "boil_use_skill_factories": True,
        "boil_num_jugs_train": [1],
        "boil_num_jugs_test": [1],
        "boil_num_burner_train": [1],
        "boil_num_burner_test": [1],
        "skill_phase_use_motion_planning": True,
        "pybullet_ik_validate": False,
        # Mirror the failing CLI: don't bail on "no state change in
        # first action" — push skills emit a CloseFingers no-op first.
        "option_model_terminate_on_repeat": False,
        "seed": 0,
    })
    env = _ExposedBoilEnv(use_gui=False)
    options = {o.name: o for o in get_gt_options(env.get_name())}

    jug = env._jugs[0]
    robot = env._robot

    # Build an option model around the env.
    option_set = set(options.values())
    option_model = _OracleOptionModel(option_set, env.simulate)

    init_state = env.get_train_tasks()[0].init
    env.set_state(init_state)

    caplog.set_level(logging.ERROR)

    # Run the same Pick → Place sequence via option_model (simulate path).
    state = env._current_observation
    state, na = option_model.get_next_state_and_num_actions(
        state, options["PickJug"].ground([robot, jug],
                                         np.array([0.01], dtype=np.float32)))
    assert na > 0, (f"PickJug should succeed under option_model. "
                    f"failure={option_model.last_execution_failure}")
    state, na = option_model.get_next_state_and_num_actions(
        state,
        options["Place"].ground([robot],
                                np.array([0.5313, 1.2899, 0.5659, 2.5974],
                                         dtype=np.float32)))
    assert na > 0, (f"option_model should place the jug, got failure "
                    f"{option_model.last_execution_failure!r}")
    assert np.hypot(
        state.get(jug, "x") - 0.5313,
        state.get(jug, "y") - 1.2899) < 0.01

    # The executed option agrees.
    env.set_state(init_state)
    env.execute_option(options["PickJug"].ground([robot, jug],
                                                 np.array([0.01],
                                                          dtype=np.float32)))
    placed = env.execute_option(options["Place"].ground(
        [robot], np.array([0.5313, 1.2899, 0.5659, 2.5974], dtype=np.float32)))
    assert placed.get(jug, "is_held") < 0.5
    assert np.hypot(
        placed.get(jug, "x") - 0.5313,
        placed.get(jug, "y") - 1.2899) < 0.01
