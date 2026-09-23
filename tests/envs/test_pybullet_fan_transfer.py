"""Physical and evaluator checks for the separate exposed-transfer pilot."""
# The model factory builds its simulator class dynamically.
# pylint: disable=protected-access,no-member
import numpy as np
import pybullet as p
import pytest

from predicators import utils
from predicators.code_sim_learning.base_simulator import base_simulator_class
from predicators.code_sim_learning.commands import ApplyForce
from predicators.envs.pybullet_fan import PyBulletFanEnv
from predicators.run.episode import EpisodeRunner, EpisodeState
from predicators.settings import CFG
from predicators.structs import Action
from scripts.cluster_utils import generate_run_configs


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("arm", [0, 1])
@pytest.mark.parametrize("variant", ["transfer", "inertial", "ramp"])
def test_launch_config_constructs_both_levels(seed, arm, variant):
    """Resolve the actual launcher, including list overrides, before reset."""
    configs = list(
        generate_run_configs(
            f"predicatorv3/continual_fan_{variant}_pilot_r1.yaml",
            batch_seeds=True))
    assert len(configs) == 2
    config = configs[arm]
    flags = {
        key: utils.string_to_python_object(str(value))
        for key, value in config.flags.items() if key.startswith("fan_")
    }
    assert flags["fan_train_num_walls_per_task"] == [0]
    assert flags["fan_test_num_walls_per_task"] == [0]
    utils.reset_config(
        dict(flags,
             env=config.env,
             seed=seed,
             num_train_tasks=1,
             num_test_tasks=1))
    env = PyBulletFanEnv(use_gui=False)
    try:
        assert len(env.get_train_tasks()) == len(env.get_test_tasks()) == 1
        env.reset("train", 0)
        env.reset("test", 0)
    finally:
        p.disconnect(env._physics_client_id)


@pytest.fixture(name="env", params=[False, True], ids=["r1", "inertial"])
def _env(request):
    utils.reset_config({
        "env": "pybullet_fan",
        "seed": 0,
        "num_train_tasks": 1,
        "num_test_tasks": 1,
        "fan_exposed_transfer": True,
        "fan_inertial_transfer": request.param,
        "fan_train_num_walls_per_task": [0],
        "fan_test_num_walls_per_task": [0],
    })
    env = PyBulletFanEnv(use_gui=False)
    yield env
    p.disconnect(env._physics_client_id)


def _noop(state):
    return Action(np.asarray(state.joint_positions, dtype=np.float32))


def test_transfer_layout_and_model_roundtrip(env):
    """The learned base reconstructs real supports, without a hidden table."""
    train = env.get_train_tasks()[0]
    test = env.get_test_tasks()[0]
    assert len(train.init.get_objects(env._platform_type)) == 1
    assert len(test.init.get_objects(env._platform_type)) == 3
    assert len(train.init.get_objects(env._boundary_type)) == 4
    assert len(test.init.get_objects(env._boundary_type)) == 3
    base = base_simulator_class("pybullet_fan")(use_gui=False)
    try:
        for task in (test, train, test):
            base._set_state(task.init)
            state = base._get_state()
            for obj in task.init:
                if obj.type.name in ("platform", "boundary"):
                    assert np.allclose(state[obj], task.init[obj], atol=1e-5)
        # Off-deck coordinates lie above the old table, but must now fall.
        state.set(env._ball, "x", 0.75)
        state.set(env._ball, "y", 1.86)
        for _ in range(20):
            state = base.simulate(state, _noop(state))
        assert state.get(env._ball, "z") < 0.30
    finally:
        p.disconnect(base._physics_client_id)


def test_transfer_evaluator_settling_and_fall(env):
    """Passing through a target is not a win; falling terminates as a loss."""
    task = env.get_test_tasks()[0]
    evaluator = task.evaluator
    at_goal = task.init.copy()
    for f in ("x", "y"):
        at_goal.set(env._ball, f, at_goal.get(env._target, f))
    assert not evaluator.terminated_trajectory([at_goal] * 20)
    assert evaluator.solved([at_goal] * 21, None)
    moving = []
    for i in range(21):
        state = at_goal.copy()
        state.set(env._ball, "x", at_goal.get(env._ball, "x") + i * 0.001)
        moving.append(state)
    assert not evaluator.terminated_trajectory(moving)
    fallen = at_goal.copy()
    fallen.set(env._ball, "z", 0.1)
    assert evaluator.terminated_trajectory([fallen])
    assert not evaluator.solved([fallen], None)
    assert evaluator.reward([fallen], None) == 0.0


def test_real_episode_fall_is_game_over(env):
    """The live episode loop, not just an offline predicate, ends on a fall."""
    runner = EpisodeRunner(env, horizon=None)
    runner.reset("test", 0)
    p.resetBasePositionAndOrientation(env._ball.id, (0.75, 1.86, 0.44),
                                      (0, 0, 0, 1),
                                      physicsClientId=env._physics_client_id)
    for _ in range(30):
        outcome = runner.step(_noop(runner.observation()))
        if outcome.state is not EpisodeState.NOT_FINISHED:
            break
    assert outcome.state is EpisodeState.GAME_OVER
    assert "fell off" in outcome.reason


def test_probe_coasting(env):
    """A short wind pulse should have a measurable unpowered tail."""
    state = env.reset("train", 0)
    for _ in range(10):
        state = env.simulate(state, _noop(state))
    state = state.copy()
    state.set(env._switches[0], "is_on", 1.0)
    for _ in range(35):
        state = env.simulate(state, _noop(state))
    x_before = state.get(env._ball, "x")
    velocity = p.getBaseVelocity(env._ball.id,
                                 physicsClientId=env._physics_client_id)[0]
    state = state.copy()
    state.set(env._switches[0], "is_on", 0.0)
    for _ in range(40):
        state = env.simulate(state, _noop(state))
    coast = state.get(env._ball, "x") - x_before
    print("TRANSFER_COAST", x_before, velocity, coast)
    assert 0.015 < coast < 0.25
    if CFG.fan_inertial_transfer:
        assert coast > 0.06
        # The measurement must not be truncated by the protective tray wall.
        assert state.get(env._ball, "x") < 1.05


def test_opposing_fan_brakes_moving_ball(env):
    """An opposing fan reduces momentum, not merely the commanded force."""
    state = env.reset("train", 0).copy()
    state.set(env._switches[0], "is_on", 1.0)
    for _ in range(35):
        state = env.simulate(state, _noop(state))
    moving = state.copy()
    moving.set(env._switches[0], "is_on", 0.0)
    results = []
    for braking in (False, True):
        state = moving.copy()
        state.set(env._switches[1], "is_on", float(braking))
        for _ in range(12):
            state = env.simulate(state, _noop(state))
        results.append(
            (state.get(env._ball, "x"),
             p.getBaseVelocity(env._ball.id,
                               physicsClientId=env._physics_client_id)[0][0]))
    print("BRAKING", CFG.fan_inertial_transfer, results)
    assert results[1][0] < results[0][0]
    assert results[1][1] < results[0][1]


def test_transfer_wind_model_and_moving_restart(env):
    """A learned force model matches coasting and can restart mid-flight."""
    model_cls = base_simulator_class("pybullet_fan")
    model = model_cls(use_gui=False)
    restarted = model_cls(use_gui=False)
    try:
        real = env.reset("test", 0).copy()
        real.set(env._switches[0], "is_on", 1.0)
        simulated = real.copy()
        saved = None

        def advance(base, state):
            state = base.simulate(state, _noop(state))
            if state.get(env._fans[0], "is_on") > 0.5:
                base.queue_residual_commands([
                    ApplyForce(
                        "ball",
                        (env.exposed_wind_force_magnitude *
                         (0.2 if CFG.fan_inertial_transfer else 1), 0, 0))
                ])
            return state

        for i in range(100):
            if i == 65:
                real = real.copy()
                simulated = simulated.copy()
                real.set(env._switches[0], "is_on", 0.0)
                simulated.set(env._switches[0], "is_on", 0.0)
            real = env.simulate(real, _noop(real))
            simulated = advance(model, simulated)
            assert np.allclose(real[env._ball],
                               simulated[env._ball],
                               atol=1e-5)
            if i == 69:
                saved = simulated.copy()
            elif i >= 70:
                saved = advance(restarted, saved)
                assert np.allclose(saved[env._ball],
                                   simulated[env._ball],
                                   atol=0.001)
    finally:
        p.disconnect(model._physics_client_id)
        p.disconnect(restarted._physics_client_id)


@pytest.mark.parametrize("landing_extension", [0.0, 0.10])
@pytest.mark.parametrize("ramp_rise", [0.003, 0.004])
def test_ramp_geometry_gravity_and_model_restore(landing_extension, ramp_rise):
    """Visible wedge geometry causes downhill motion in real and model
    worlds."""
    utils.reset_config({
        "env": "pybullet_fan",
        "seed": 0,
        "num_train_tasks": 1,
        "num_test_tasks": 1,
        "fan_exposed_transfer": True,
        "fan_inertial_transfer": True,
        "fan_ramp_transfer": True,
        "fan_ramp_landing_extension": landing_extension,
        "fan_ramp_rise": ramp_rise,
        "fan_train_num_walls_per_task": [0],
        "fan_test_num_walls_per_task": [0],
    })
    env = PyBulletFanEnv(use_gui=False)
    base = base_simulator_class("pybullet_fan")(use_gui=False)
    restarted = base_simulator_class("pybullet_fan")(use_gui=False)
    try:
        for split in ("train", "test", "train", "test"):
            real = env.reset(split, 0)
            ramp, = real.get_objects(env._ramp_type)
            base._set_state(real)
            restored = base._get_state()
            assert np.allclose(restored[ramp], real[ramp], atol=1e-5)
            assert real.get(ramp, "rise") == pytest.approx(ramp_rise)
            landing = env._platforms[1]
            for feature in ("x", "x_len"):
                assert restored.get(landing, feature) == pytest.approx(
                    real.get(landing, feature))
            left = real.get(landing, "x") - real.get(landing, "x_len") / 2
            right = real.get(landing, "x") + real.get(landing, "x_len") / 2
            assert left == pytest.approx(1.07)
            expected_right = (real.get(env._target, "x") + 0.16 +
                              landing_extension if split == "test" else 1.45)
            assert right == pytest.approx(expected_right)
            for x in (0.75, 1.00):
                ray = p.rayTest((x, real.get(ramp, "y"), 0.8),
                                (x, real.get(ramp, "y"), 0.1),
                                physicsClientId=env._physics_client_id)[0]
                assert ray[0] == env._boundary_named(ramp).id
                assert ray[3][2] == pytest.approx(0.4 + ramp_rise *
                                                  (1 - (x - 0.67) / 0.4),
                                                  abs=1e-5)
        real = real.copy()
        real.set(env._ball, "x", 0.76)
        real.set(env._ball, "y", real.get(ramp, "y"))
        real.set(env._ball, "z",
                 0.4 + ramp_rise * (1 - 0.09 / 0.4) + env.ball_radius)
        simulated = real.copy()
        saved = None
        for tick in range(25):
            real = env.simulate(real, _noop(real))
            simulated = base.simulate(simulated, _noop(simulated))
            assert np.allclose(real[env._ball],
                               simulated[env._ball],
                               atol=1e-5)
            if tick == 10:
                saved = simulated.copy()
            elif tick > 10:
                saved = restarted.simulate(saved, _noop(saved))
                assert np.allclose(saved[env._ball],
                                   simulated[env._ball],
                                   atol=0.001)
        print("RAMP_GRAVITY", real.get(env._ball, "x"))
        assert real.get(env._ball, "x") > 0.80
        assert real.get(env._ball, "z") > 0.40
        # Exercise the bay/ramp seam from the actual initial state. A mesh
        # collision margin can create an unintended curb despite correct
        # interior surface heights and downhill gravity.
        crossing = env.reset("train", 0).copy()
        crossing.set(env._switches[0], "is_on", 1.0)
        for _ in range(100):
            crossing = env.simulate(crossing, _noop(crossing))
        print("RAMP_SEAM", crossing.get(env._ball, "x"))
        assert crossing.get(env._ball, "x") > 1.25
        assert crossing.get(env._ball, "z") > 0.40
    finally:
        p.disconnect(env._physics_client_id)
        p.disconnect(base._physics_client_id)
        p.disconnect(restarted._physics_client_id)
