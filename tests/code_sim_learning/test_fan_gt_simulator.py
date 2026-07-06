"""Test the fan GT hybrid simulator against the real env.

The fan env applies its wind dynamics in ``_domain_specific_step``,
which the approaches' base sims skip (``skip_process_dynamics=True``),
so the GT process rules must reproduce the ball's wind-driven motion.
These tests roll the hybrid sim (base env + GT rules, composed exactly
like ``AgentSimLearningApproach._build_combined_simulator``) side by
side with the real env under identical no-op action sequences, covering
free runs, obstacle-wall blocking, boundary parking, simultaneous fans,
and the fans-off case.
"""

import numpy as np
import pytest

from predicators import utils
from predicators.code_sim_learning.utils import apply_rules, merge_updates
from predicators.envs import create_new_env
from predicators.ground_truth_models import get_gt_simulator
from predicators.structs import Action

# BallAtTarget tolerance (pos_gap / 2): the hybrid ball must stay within
# this of the real ball or plan validation would disagree with reality.
GOAL_TOL = 0.04


@pytest.fixture(scope="module", name="fan_setup")
def _fan_setup():
    utils.reset_config({
        "env": "pybullet_fan",
        "seed": 0,
        "fan_use_skill_factories": True,
    })
    rules, specs, _ = get_gt_simulator("pybullet_fan")
    params = {s.name: s.init_value for s in specs}
    real_env = create_new_env("pybullet_fan", do_cache=False, use_gui=False)
    base_env = create_new_env("pybullet_fan",
                              do_cache=False,
                              use_gui=False,
                              skip_process_dynamics=True)
    task = real_env.get_train_tasks()[0]
    return real_env, base_env, rules, params, task


def _make_noop(state, env):
    arr = np.array(list(state.joint_positions), dtype=np.float32)
    n = env.action_space.shape[0]
    if arr.shape[0] < n:
        arr = np.concatenate(
            [arr, np.zeros(n - arr.shape[0], dtype=np.float32)])
    return Action(arr)


def _get_obj(state, type_name, pred=None):
    for o in state:
        if o.type.name == type_name and (pred is None or pred(state, o)):
            return o
    raise ValueError(f"no {type_name} in state")


def _rollout_pair(fan_setup, switch_sides, n_steps, ball_xy=None):
    """Roll real and hybrid sims in lockstep; return max ball error and
    final (real, hybrid) states."""
    real_env, base_env, rules, params, task = fan_setup
    state = task.init.copy()
    ball = _get_obj(state, "ball")
    if ball_xy is not None:
        state.set(ball, "x", ball_xy[0])
        state.set(ball, "y", ball_xy[1])
    for side in switch_sides:
        switch = _get_obj(
            state, "switch",
            lambda s, o, _side=side: s.get(o, "controls_fan") == _side)
        state.set(switch, "is_on", 1.0)

    def hybrid_simulate(s, a):
        base_state = base_env.simulate(s, a)
        updates = apply_rules(base_state, rules, params)
        return merge_updates(base_state, updates) if updates else base_state

    s_real, s_hyb = state, state
    max_err = 0.0
    for _ in range(n_steps):
        s_real = real_env.simulate(s_real, _make_noop(s_real, real_env))
        s_hyb = hybrid_simulate(s_hyb, _make_noop(s_hyb, base_env))
        err = float(
            np.hypot(
                s_real.get(ball, "x") - s_hyb.get(ball, "x"),
                s_real.get(ball, "y") - s_hyb.get(ball, "y")))
        max_err = max(max_err, err)
    return max_err, s_real, s_hyb


def test_fan_gt_simulator_loads():
    """The factory registry resolves pybullet_fan to a real simulator."""
    utils.reset_config({"env": "pybullet_fan", "seed": 0})
    rules, specs, features = get_gt_simulator("pybullet_fan")
    assert [r.__name__ for r in rules] == ["_wind_blowing"]
    names = {s.name for s in specs}
    assert names == {"ball_speed", "wall_clearance"}
    assert features == {"ball": ["x", "y"]}


def test_fan_hybrid_free_run_and_boundary(fan_setup):
    """Fan blows the ball across the grid; both sims park at the
    boundary wall."""
    # Seed-0 train task: ball (0.75, 1.694), wall (0.75, 1.614), target
    # (0.75, 1.534); grid x {0.67, 0.75, 0.83}, y {1.534, 1.614, 1.694}.
    max_err, s_real, s_hyb = _rollout_pair(fan_setup, [0.0], 45)
    assert max_err < GOAL_TOL
    ball = _get_obj(s_real, "ball")
    # Both parked at the right boundary (grid edge 0.83).
    assert abs(s_real.get(ball, "x") - 0.83) < 0.005
    assert abs(s_hyb.get(ball, "x") - 0.83) < 0.005


def test_fan_hybrid_wall_blocking(fan_setup):
    """The obstacle wall stops the wind-driven ball in both sims."""
    # Up fan (blows -y) drives the ball from (0.75, 1.694) into the wall
    # at (0.75, 1.614); without wall modeling the hybrid ball would sail
    # through to the target row (1.534) and diverge by ~0.11.
    max_err, s_real, s_hyb = _rollout_pair(fan_setup, [3.0], 20)
    assert max_err < GOAL_TOL
    ball = _get_obj(s_real, "ball")
    for s in (s_real, s_hyb):
        assert s.get(ball, "y") > 1.66  # parked against the wall

    # Same wall approached sideways from a clear cell.
    max_err, _, _ = _rollout_pair(fan_setup, [0.0], 20, ball_xy=(0.67, 1.614))
    assert max_err < GOAL_TOL


def test_fan_hybrid_clear_column_and_multi_fan(fan_setup):
    """Free multi-cell run in a clear column, and two fans at once."""
    max_err, s_real, s_hyb = _rollout_pair(fan_setup, [3.0],
                                           80,
                                           ball_xy=(0.67, 1.694))
    assert max_err < GOAL_TOL
    ball = _get_obj(s_real, "ball")
    # Crossed the full grid to the lower boundary (grid edge 1.534).
    assert abs(s_real.get(ball, "y") - 1.534) < 0.005
    assert abs(s_hyb.get(ball, "y") - 1.534) < 0.005

    max_err, _, _ = _rollout_pair(fan_setup, [0.0, 2.0], 35)
    assert max_err < GOAL_TOL


def test_fan_hybrid_fans_off_no_motion(fan_setup):
    """With all fans off the rules leave the ball to the base sim."""
    max_err, s_real, s_hyb = _rollout_pair(fan_setup, [], 10)
    assert max_err < GOAL_TOL
    ball = _get_obj(s_real, "ball")
    task_init = fan_setup[4].init
    assert abs(s_hyb.get(ball, "x") - task_init.get(ball, "x")) < 0.005
    assert abs(s_hyb.get(ball, "y") - task_init.get(ball, "y")) < 0.005
