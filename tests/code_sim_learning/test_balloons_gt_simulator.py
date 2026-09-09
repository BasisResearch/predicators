"""Test the balloons GT hybrid simulator against the real env.

The balloons env applies its release, lift and pop in
``_domain_specific_step``, which the approaches' base sims skip
(``skip_residual_dynamics=True``), so the GT residual rules must
reproduce them on the command channel: a freed balloon is seated on the
box by a feature update, then welded to it (``cmds.attach``) and pulls
it up (``cmds.apply_force``) with its colour's fading lift. These tests
roll the hybrid sim (base env with the true masses and drag as physical
parameters + GT rules + command queueing, composed like
``AgentSimLearningApproach._build_combined_simulator``) side by side
with the real env under identical hold actions, covering a level the
oracle wins, a staged release, a pop at the ceiling, and the base-only
case where nothing rises.
"""
# pylint: disable=protected-access
from __future__ import annotations

import numpy as np
import pytest

from predicators import utils
from predicators.code_sim_learning.commands import CommandBuffer
from predicators.code_sim_learning.utils import apply_rules, \
    has_physics_rules, merge_updates
from predicators.ground_truth_models import get_gt_simulator
from predicators.settings import CFG
from predicators.structs import Action

# The hybrid box must track the real box to well inside the band's
# half-height (0.025 m), or plan validation would disagree with reality.
BOX_TOL = 0.005


@pytest.fixture(scope="module", name="balloons_setup")
def _balloons_setup():
    utils.reset_config({
        "env": "pybullet_balloons",
        "seed": 0,
        "num_train_tasks": 1,
        "num_test_tasks": 1,
        "skill_phase_use_motion_planning": False,
    })
    # pylint: disable-next=import-outside-toplevel
    from predicators.envs.pybullet_balloons import PyBulletBalloonsEnv
    rules, specs, _ = get_gt_simulator("pybullet_balloons")
    params = {s.name: s.init_value for s in specs}
    real_env = PyBulletBalloonsEnv(use_gui=False)
    base_env = PyBulletBalloonsEnv(use_gui=False, skip_residual_dynamics=True)
    # The masses and the drag are physical parameters of the base env,
    # not residual-rule parameters: give the hybrid their true values.
    overrides = {
        f"mass_{name}": real_env.true_box_mass(index)
        for index, (name, _) in enumerate(real_env.BOX_PALETTE)
    }
    overrides["air_drag"] = float(CFG.balloons_drag)
    base_env.apply_physical_param_overrides(overrides)
    return real_env, base_env, rules, params


def _hold(state, env):
    arr = np.array(list(state.joint_positions), dtype=np.float32)
    n = env.action_space.shape[0]
    if arr.shape[0] < n:
        arr = np.concatenate(
            [arr, np.zeros(n - arr.shape[0], dtype=np.float32)])
    return Action(arr)


def _level(real_env, box_color, balloon_colors, band, open_clips):
    """A level state with joint data, the given clips already pushed open."""
    state = real_env.level_state(box_color, balloon_colors, band)
    real_env._pybullet_robot.set_joints(
        real_env._pybullet_robot.initial_joint_positions)
    real_env._set_state(state)
    init = real_env._get_state()
    real_env._current_observation = init
    init = init.copy()
    for i in open_clips:
        init.set(real_env._clips[i], "is_on", 1.0)
    return init


def _hybrid_step(base_env, rules, params):
    """The combined simulator, composed like
    ``AgentSimLearningApproach._build_combined_simulator``: commands
    emitted at step t are held keyed to the state they were computed for
    and queued only when the next call continues from that state."""
    pending = {"state": None, "commands": []}

    def step(state, action):
        if pending["commands"]:
            if pending["state"] is not None and \
                    state.allclose(pending["state"]):
                base_env.queue_residual_commands(pending["commands"])
            pending["state"], pending["commands"] = None, []
        base_state = base_env.simulate(state, action)
        cmds = CommandBuffer()
        updates = apply_rules(base_state, rules, params, cmds=cmds)
        next_state = merge_updates(base_state, updates) if updates else \
            base_state
        if cmds:
            pending["state"], pending["commands"] = next_state, cmds.commands
        return next_state

    return step


def _fresh(real_env, base_env):
    """Drop commands a previous level left queued on either env (an episode
    reset does the same in the protocol)."""
    real_env.queue_residual_commands([])
    base_env.queue_residual_commands([])


def _rollout_pair(balloons_setup, init, n_steps, open_at=None):
    """Roll real and hybrid sims in lockstep from ``init``; ``open_at`` maps a
    step index to clips to push open at that step.

    Returns the max box-height gap and the final (real, hybrid) states.
    """
    real_env, base_env, rules, params = balloons_setup
    _fresh(real_env, base_env)
    hybrid = _hybrid_step(base_env, rules, params)
    s_real, s_hyb = init, init
    box = real_env._box
    max_gap = 0.0
    for t in range(n_steps):
        if open_at and t in open_at:
            s_real, s_hyb = s_real.copy(), s_hyb.copy()
            for i in open_at[t]:
                s_real.set(real_env._clips[i], "is_on", 1.0)
                s_hyb.set(real_env._clips[i], "is_on", 1.0)
        s_real = real_env.simulate(s_real, _hold(s_real, real_env))
        s_hyb = hybrid(s_hyb, _hold(s_hyb, base_env))
        max_gap = max(max_gap, abs(s_real.get(box, "z") - s_hyb.get(box, "z")))
        for balloon in real_env._active_balloons(s_real):
            for flag in ("tied", "popped"):
                assert s_real.get(balloon, flag) == s_hyb.get(balloon, flag), \
                    (t, balloon.name, flag)
    return max_gap, s_real, s_hyb


def test_balloons_gt_simulator_loads():
    """The factory registry resolves pybullet_balloons to a real simulator
    whose rules act through the command channel."""
    utils.reset_config({"env": "pybullet_balloons", "seed": 0})
    rules, specs, features = get_gt_simulator("pybullet_balloons")
    assert [r.__name__ for r in rules] == ["_release_and_pull"]
    names = {s.name for s in specs}
    assert names == {
        "lift_red", "lift_blue", "lift_green", "lift_gold", "fade_height"
    }
    by_name = {s.name: s.init_value for s in specs}
    assert by_name["lift_gold"] == CFG.balloons_lifts[3]
    assert by_name["fade_height"] == CFG.balloons_fade_height
    assert set(features) == {"balloon", "box"}
    assert has_physics_rules(rules)


def test_balloons_hybrid_hangs_in_the_band(balloons_setup):
    """Oak box, red + gold + blue freed together: both sims seat the balloons
    on the box and hang it at the analytic height."""
    real_env = balloons_setup[0]
    colors = [0, 3, 1]  # red, gold, blue
    expected = real_env.hover_height(1, colors)
    assert expected is not None
    band = (expected - 0.025, expected + 0.025)
    init = _level(real_env, 1, colors, band, open_clips=[0, 1, 2])
    max_gap, s_real, s_hyb = _rollout_pair(balloons_setup, init, 250)
    assert max_gap < BOX_TOL
    box = real_env._box
    for s in (s_real, s_hyb):
        assert abs(s.get(box, "z") - expected) < 0.03
        assert all(
            s.get(b, "tied") == 1.0 for b in real_env._active_balloons(s))
        assert all(
            s.get(b, "popped") == 0.0 for b in real_env._active_balloons(s))


def test_balloons_hybrid_staged_release(balloons_setup):
    """Clips pushed open one at a time by the Release skill, each once the box
    has settled (the oracle's cadence): the hybrid follows the env through
    every push and every release."""
    # pylint: disable-next=import-outside-toplevel
    from predicators.ground_truth_models.balloons.options import \
        probe_release_option, release_params
    real_env, base_env, rules, params = balloons_setup
    _fresh(real_env, base_env)
    hybrid = _hybrid_step(base_env, rules, params)
    colors = [0, 3, 1]
    expected = real_env.hover_height(1, colors)
    assert expected is not None
    init = _level(real_env,
                  1,
                  colors, (expected - 0.025, expected + 0.025),
                  open_clips=[])
    box = real_env._box
    s_real, s_hyb = init, init
    max_gap = 0.0
    for clip_index in range(3):
        option = probe_release_option().ground(
            [real_env._robot, real_env._clips[clip_index]], release_params())
        assert option.initiable(s_real)
        for t in range(500):
            # Both sims see the same states until they diverge, so the
            # skill's action is computed once and applied to both.
            if not option.terminal(s_real):
                action = option.policy(s_real)
            else:
                action = _hold(s_real, real_env)
            s_real = real_env.simulate(s_real, action)
            s_hyb = hybrid(s_hyb, action)
            max_gap = max(max_gap,
                          abs(s_real.get(box, "z") - s_hyb.get(box, "z")))
            settled = (s_real.get(box, "speed") < CFG.balloons_settle_speed
                       and s_hyb.get(box, "speed") < CFG.balloons_settle_speed)
            if option.terminal(s_real) and t >= 40 and settled:
                break
        balloon = real_env._active_balloons(s_real)[clip_index]
        assert s_real.get(balloon, "tied") == 1.0
        assert s_hyb.get(balloon, "tied") == 1.0
    assert max_gap < BOX_TOL
    assert abs(s_real.get(box, "z") - expected) < 0.03
    assert abs(s_hyb.get(box, "z") - expected) < 0.03


def test_balloons_hybrid_pop_at_the_ceiling(balloons_setup):
    """Pine box with every balloon freed: the stack reaches the ceiling, a
    balloon bursts in both sims, and the box comes back down together."""
    real_env = balloons_setup[0]
    colors = [0, 1, 2, 3]
    init = _level(real_env, 0, colors, (0.6, 0.65), open_clips=[0, 1, 2, 3])
    max_gap, s_real, s_hyb = _rollout_pair(balloons_setup, init, 300)
    assert max_gap < BOX_TOL
    popped = [
        b.name for b in real_env._active_balloons(s_real)
        if s_real.get(b, "popped") == 1.0
    ]
    assert popped
    assert popped == [
        b.name for b in real_env._active_balloons(s_hyb)
        if s_hyb.get(b, "popped") == 1.0
    ]


def test_balloons_base_sim_alone_never_rises(balloons_setup):
    """Without the rules an open clip frees nothing on the base sim."""
    real_env, base_env, _, _ = balloons_setup
    _fresh(real_env, base_env)
    colors = [0, 3, 1]
    init = _level(real_env, 1, colors, (0.7, 0.8), open_clips=[0, 1, 2])
    s = init
    for _ in range(60):
        s = base_env.simulate(s, _hold(s, base_env))
    box = real_env._box
    assert abs(s.get(box, "z") - init.get(box, "z")) < 0.01
    assert all(s.get(b, "tied") == 0.0 for b in real_env._active_balloons(s))
