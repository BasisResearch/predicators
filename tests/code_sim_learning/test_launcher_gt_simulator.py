"""Test the launcher GT hybrid simulator against the real env.

The launcher env fires the ball in ``_domain_specific_step``, which the
approaches' base sims skip (``skip_residual_dynamics=True``): on the
base sim the handle snaps home and the ball stays in the cup. The GT
residual rule is recurrent - it carries the deepest compression seen in
its latent block and, on the snap, sends the ball off on the command
channel (``cmds.set_velocity``). These tests roll the hybrid sim (base
env with the true block masses as physical parameters + GT rule +
latent threading + command queueing, composed like
``AgentSimLearningApproach._build_latent_combined_simulator``) side by
side with the real env under the same Cock skill, and check the flight
and the tower agree; and that the base sim alone never fires.
"""
# pylint: disable=protected-access
from __future__ import annotations

import copy

import numpy as np
import pytest

from predicators import utils
from predicators.code_sim_learning.commands import CommandBuffer
from predicators.code_sim_learning.utils import apply_rules_with_latent, \
    has_latent_rules, has_physics_rules, init_latent, merge_updates, \
    observation_view, read_latent_init
from predicators.ground_truth_models import get_gt_simulator
from predicators.settings import CFG
from predicators.structs import Action

# The hybrid ball must track the real ball to a few millimetres: the
# working compression windows are three scan steps (7.5 mm of handle
# travel) wide, so a centimetre of flight error would move the strike to
# another block.
BALL_TOL = 0.005


@pytest.fixture(scope="module", name="launcher_setup")
def _launcher_setup():
    utils.reset_config({
        "env": "pybullet_launcher",
        "seed": 0,
        "num_train_tasks": 1,
        "num_test_tasks": 1,
        "skill_phase_use_motion_planning": False,
    })
    # pylint: disable=import-outside-toplevel
    from predicators.envs.pybullet_launcher import PyBulletLauncherEnv
    from predicators.ground_truth_models.launcher import gt_simulator

    # pylint: enable=import-outside-toplevel
    rules, specs, _ = get_gt_simulator("pybullet_launcher")
    params = {s.name: s.init_value for s in specs}
    latent_init = read_latent_init(vars(gt_simulator))
    real_env = PyBulletLauncherEnv(use_gui=False)
    base_env = PyBulletLauncherEnv(use_gui=False, skip_residual_dynamics=True)
    # The block masses are physical parameters of the base env, not
    # residual-rule parameters: give the hybrid their true values.
    base_env.apply_physical_param_overrides({
        f"mass_{name}": real_env.true_mass(index)
        for index, (name, _) in enumerate(real_env.COLOR_PALETTE)
    })
    return real_env, base_env, rules, params, latent_init


def _hybrid_step(base_env, rules, params, latent_init):
    """The combined simulator, composed like
    ``AgentSimLearningApproach._build_latent_combined_simulator``: the
    latent rides on ``state.latent``, and commands emitted at step t are
    held keyed to the state they were computed for and queued only when
    the next call continues from that state."""
    pending = {"state": None, "commands": []}

    def step(state, action):
        if pending["commands"]:
            if pending["state"] is not None and \
                    state.allclose(pending["state"]):
                base_env.queue_residual_commands(pending["commands"])
            pending["state"], pending["commands"] = None, []
        latent = (copy.deepcopy(state.latent) if state.latent is not None else
                  init_latent(latent_init, params))
        base_state = base_env.simulate(state, action)
        obs = observation_view(base_state)
        cmds = CommandBuffer()
        updates = apply_rules_with_latent(obs,
                                          latent, [(obs, action)],
                                          rules,
                                          params,
                                          cmds=cmds)
        next_state = merge_updates(base_state, updates) if updates else \
            base_state
        next_state.latent = latent
        if cmds:
            pending["state"], pending["commands"] = next_state, cmds.commands
        return next_state

    return step


def _fresh(real_env, base_env):
    """Drop commands a previous shot left queued on either env (an episode
    reset does the same in the protocol)."""
    real_env.queue_residual_commands([])
    base_env.queue_residual_commands([])


def _hold(state, env):
    arr = np.array(list(state.joint_positions), dtype=np.float32)
    n = env.action_space.shape[0]
    if arr.shape[0] < n:
        arr = np.concatenate(
            [arr, np.zeros(n - arr.shape[0], dtype=np.float32)])
    return Action(arr)


def _level(real_env, stand_x, colors, balls_left):
    """A level state with joint data."""
    state = real_env.level_state(stand_x, colors, balls_left)
    return _with_joints(real_env, state)


def _with_joints(real_env, state):
    """``state`` as the env reads it back: a PyBullet state carrying the
    robot's joints (a plain level state has none, and the finger check on the
    next step needs them)."""
    real_env._pybullet_robot.set_joints(
        real_env._pybullet_robot.initial_joint_positions)
    real_env._set_state(state)
    init = real_env._get_state()
    real_env._current_observation = init
    return init


def _shoot_pair(launcher_setup, init, depth, max_steps=400):
    """Run the Cock skill at ``depth`` on both sims from ``init`` and hold
    until the real ball has flown and stopped.

    Returns the max ball gap and the final (real, hybrid) states.
    """
    # pylint: disable-next=import-outside-toplevel
    from predicators.ground_truth_models.launcher.options import \
        probe_cock_option
    real_env, base_env, rules, params, latent_init = launcher_setup
    _fresh(real_env, base_env)
    hybrid = _hybrid_step(base_env, rules, params, latent_init)
    ball = real_env._ball
    cock_params = np.array(
        [CFG.launcher_push_approach, CFG.launcher_push_contact_z, depth],
        dtype=np.float32)
    option = probe_cock_option().ground([real_env._robot, real_env._launcher],
                                        cock_params)
    assert option.initiable(init)
    s_real, s_hyb = init, init
    max_gap = 0.0
    launched = False
    for _ in range(max_steps):
        if not option.terminal(s_real):
            # Both sims see the same states until they diverge, so the
            # skill's action is computed once and applied to both.
            action = option.policy(s_real)
        else:
            action = _hold(s_real, real_env)
        s_real = real_env.simulate(s_real, action)
        s_hyb = hybrid(s_hyb, action)
        gap = float(
            np.linalg.norm([
                s_real.get(ball, k) - s_hyb.get(ball, k)
                for k in ("x", "y", "z")
            ]))
        # Contact settling can cross the reload threshold one step apart.
        # Compare the same ball in flight, not a spent ball with its spare.
        launcher = real_env._launcher
        if s_real.get(launcher, "balls_left") == \
                s_hyb.get(launcher, "balls_left"):
            max_gap = max(max_gap, gap)
        moving = s_real.get(ball, "speed") > CFG.launcher_settle_speed
        launched |= moving
        if option.terminal(s_real) and launched and not moving:
            break
    assert launched, "the real env never fired"
    return max_gap, s_real, s_hyb


def test_launcher_gt_simulator_loads():
    """The factory registry resolves pybullet_launcher to a recurrent simulator
    whose rule acts through the command channel."""
    utils.reset_config({"env": "pybullet_launcher", "seed": 0})
    rules, specs, features = get_gt_simulator("pybullet_launcher")
    assert [r.__name__ for r in rules] == ["_launching"]
    by_name = {s.name: s.init_value for s in specs}
    assert set(by_name) == {"spring_k", "min_compression"}
    assert by_name["spring_k"] == CFG.launcher_spring_k
    assert set(features) == {"ball", "block"}
    assert has_physics_rules(rules)
    assert has_latent_rules(rules)


def test_launcher_hybrid_flight_matches_the_env(launcher_setup):
    """A shot at a fixed depth: the hybrid ball leaves on the same step at the
    same speed and flies with the real one."""
    real_env = launcher_setup[0]
    init = _level(real_env, 0.9, [0, 1], 2)
    max_gap, s_real, s_hyb = _shoot_pair(launcher_setup, init, 0.05)
    assert max_gap < BALL_TOL
    # The ball flew in both sims: once it stopped, the visible reload
    # put it back in the cup and spent a spare.
    for s in (s_real, s_hyb):
        assert s.get(real_env._launcher, "balls_left") == 1.0


def test_launcher_hybrid_takes_the_top_block(launcher_setup):
    """The generator's own solution shot topples the top block and leaves the
    rest standing on both sims."""
    real_env = launcher_setup[0]
    task = real_env.get_train_tasks()[0]
    depth = task.offline_task_metrics["solution_depth"]
    init = _with_joints(real_env, task.init)
    max_gap, s_real, s_hyb = _shoot_pair(launcher_setup, init, depth)
    assert max_gap < BALL_TOL
    for s in (s_real, s_hyb):
        assert all(atom.holds(s) for atom in task.goal), s


def test_launcher_base_sim_alone_never_fires(launcher_setup):
    """Without the rule the handle snaps home and the ball stays put."""
    # pylint: disable-next=import-outside-toplevel
    from predicators.ground_truth_models.launcher.options import \
        probe_cock_option
    real_env, base_env, _, _, _ = launcher_setup
    _fresh(real_env, base_env)
    init = _level(real_env, 0.9, [0, 1], 2)
    option = probe_cock_option().ground(
        [real_env._robot, real_env._launcher],
        np.array(
            [CFG.launcher_push_approach, CFG.launcher_push_contact_z, 0.05],
            dtype=np.float32))
    assert option.initiable(init)
    s = init
    for _ in range(300):
        action = option.policy(s) if not option.terminal(s) else _hold(
            s, base_env)
        s = base_env.simulate(s, action)
    ball = real_env._ball
    assert s.get(real_env._launcher, "compression") == 0.0
    assert s.get(ball, "speed") < CFG.launcher_settle_speed
    cx, _, _ = real_env.cup_position()
    assert abs(s.get(ball, "x") - cx) < real_env.cup_radius
