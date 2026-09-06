"""Unit tests for the launcher env's mechanics, launch law, materials, tasks,
evaluator and skills.

Covers the properties the domain rests on: the handle holds a push and
snaps home when let go, the snap fires the loaded ball at a speed set by
the compression (and fires nothing on the base sim), a flown ball is
reloaded while spares last, every generated level has a compression that
topples the top block and leaves the rest standing, block colours weigh
differently (and not at all on the base sim until overridden), a toppled
protected block or an empty launcher ends the level as a loss, and the
oracle's helper predicate and sampler find the recorded shot.
"""
# pylint: disable=protected-access
from __future__ import annotations

import numpy as np
import pybullet as p
import pytest

from predicators import utils
from predicators.structs import Action, GroundAtom


def _make_env(**overrides):
    config = {
        "env": "pybullet_launcher",
        "seed": 0,
        "num_train_tasks": 1,
        "num_test_tasks": 1,
        "skill_phase_use_motion_planning": False,
    }
    config.update(overrides)
    utils.reset_config(config)
    from predicators.envs import \
        pybullet_launcher  # pylint: disable=import-outside-toplevel
    return pybullet_launcher, pybullet_launcher.PyBulletLauncherEnv(
        use_gui=False)


@pytest.fixture(name="env_module", scope="module")
def _env_module():
    return _make_env()


def _hold(env) -> Action:
    return Action(np.array(env._pybullet_robot.get_joints(), dtype=np.float32))


def _settle(env, steps: int = 200) -> None:
    hold = _hold(env)
    launched = False
    for _ in range(steps):
        env._step_once(hold)
        moving = env._ball_speed() > 0.02
        launched |= moving
        if launched and not moving:
            break


def test_handle_holds_snaps_and_fires(env_module):
    """A compressed handle stays put while held, snaps home when not, and the
    snap fires the loaded ball at the spring constant times the compression;
    the base sim snaps but fires nothing."""
    mod, env = env_module
    state = env.level_state(0.9, [0, 0], 2)
    env._set_state(state)
    env._current_observation = env._get_state()
    assert env._read_compression() == 0.0
    env._set_compression(0.06)
    assert abs(env._read_compression() - 0.06) < 1e-6
    # One step with the hand away from the handle: it snaps home and the
    # ball leaves at about spring_k * 0.06.
    env._step_once(_hold(env))
    assert env._read_compression() == 0.0
    env._step_once(_hold(env))
    speed = env._ball_speed()
    from predicators.settings import \
        CFG  # pylint: disable=import-outside-toplevel
    assert abs(speed - CFG.launcher_spring_k * 0.06) < 0.3
    base = mod.PyBulletLauncherEnv(use_gui=False, skip_residual_dynamics=True)
    base._set_state(state)
    base._current_observation = base._get_state()
    base._set_compression(0.06)
    base._step_once(_hold(base))
    base._step_once(_hold(base))
    assert base._read_compression() == 0.0
    assert base._ball_speed() < 0.02
    assert base._ball_in_cup()


def test_reload_spends_spares_then_runs_out(env_module):
    """A flown ball is put back in the cup and a spare is spent; with no spare
    the launcher is out of balls and the level is lost."""
    mod, env = env_module
    state = env.level_state(0.9, [0, 0], 1)
    task_goal = env.goal_for(state)
    evaluator = mod.LauncherEvaluator(task_goal)
    env._set_state(state)
    env._current_observation = env._get_state()
    env._set_compression(0.03)
    _settle(env)
    obs = env._get_state()
    assert env._ball_in_cup()
    assert obs.get(env._launcher, "balls_left") == 0.0
    assert mod.ball_loaded(obs) and not mod.out_of_balls(obs)
    assert not evaluator.terminated(obs)
    # Fire the last ball wide: nothing topples, and the launcher is
    # empty once the ball has stopped.
    env._set_compression(0.03)
    _settle(env)
    obs = env._get_state()
    assert not env._ball_in_cup()
    assert mod.out_of_balls(obs)
    assert evaluator.terminated(obs)
    ok, why = evaluator._certify([state, obs], None)
    assert not ok and "out of balls" in why


def test_generated_levels_have_a_working_shot(env_module):
    """Each task's recorded compression topples the top block and leaves the
    rest standing; the goal does not hold at the start."""
    mod, env = env_module
    for tasks in (env.get_train_tasks(), env.get_test_tasks()):
        assert tasks
        for task in tasks:
            state = task.init
            blocks = env._active_blocks(state)
            assert state.get(blocks[-1], "is_target") == 1.0
            assert all(state.get(b, "is_target") == 0.0 for b in blocks[:-1])
            assert "topples" in task.goal_nl
            # The lower blocks already stand; the top one is not down.
            assert not mod.block_toppled(state, blocks[-1])
            assert not all(atom.holds(state) for atom in task.goal)
            assert not task.evaluator.terminated(state)
            depth = task.offline_task_metrics["solution_depth"]
            end = env.launch_outcome(state, depth)
            assert end is not None
            assert all(atom.holds(end) for atom in task.goal)
            assert mod.block_toppled(end, blocks[-1])
            assert not any(mod.block_toppled(end, b) for b in blocks[:-1])


def test_materials_weigh_differently_and_base_sim_does_not(env_module):
    """Block mass follows the colour on the real env; the base sim gives every
    colour the same mass until the physical parameters are overridden."""
    mod, env = env_module
    state = env.level_state(0.9, [0, 1], 2)
    env._set_state(state)
    masses = [
        p.getDynamicsInfo(b.id, -1, physicsClientId=env._physics_client_id)[0]
        for b in env._blocks[:2]
    ]
    assert abs(masses[0] - env.true_mass(0)) < 1e-6
    assert abs(masses[1] - env.true_mass(1)) < 1e-6
    assert masses[0] != masses[1]
    base = mod.PyBulletLauncherEnv(use_gui=False, skip_residual_dynamics=True)
    base._set_state(state)
    base_masses = [
        p.getDynamicsInfo(b.id, -1, physicsClientId=base._physics_client_id)[0]
        for b in base._blocks[:2]
    ]
    assert base_masses[0] == base_masses[1] == base.block_base_mass
    base.apply_physical_param_overrides({"mass_stone": env.true_mass(1)})
    mass = p.getDynamicsInfo(base._blocks[1].id,
                             -1,
                             physicsClientId=base._physics_client_id)[0]
    assert abs(mass - env.true_mass(1)) < 1e-6
    with pytest.raises(ValueError):
        base.apply_physical_param_overrides({"mass_glass": 1.0})


def test_toppling_a_protected_block_loses(env_module):
    """A tilted lower block terminates the episode without certifying it."""
    mod, env = env_module
    task = env.get_train_tasks()[0]
    state = task.init
    blocks = env._active_blocks(state)
    tilted = state.copy()
    tilted.set(blocks[0], "roll", 1.2)
    Toppled = next(p_ for p_ in env.predicates if p_.name == "Toppled")
    Standing = next(p_ for p_ in env.predicates if p_.name == "Standing")
    assert GroundAtom(Toppled, [blocks[0]]).holds(tilted)
    assert not GroundAtom(Standing, [blocks[0]]).holds(tilted)
    assert task.evaluator.terminated(tilted)
    ok, why = task.evaluator._certify([state, tilted], None)
    assert not ok and blocks[0].name in why
    del mod


def test_oracle_helper_and_sampler(env_module):
    """``Hittable`` holds for the top block only, the sampler's depth reaches
    the goal, and the process model builds."""
    mod, env = env_module
    # pylint: disable=import-outside-toplevel
    from predicators.ground_truth_models import get_gt_helper_predicates, \
        get_gt_options, get_gt_processes
    from predicators.ground_truth_models.launcher.processes import \
        depth_for_shot

    # pylint: enable=import-outside-toplevel
    helpers = {
        p_.name: p_
        for p_ in get_gt_helper_predicates("pybullet_launcher")
    }
    task = env.get_train_tasks()[0]
    state = task.init
    blocks = env._active_blocks(state)
    Hittable = helpers["Hittable"]
    assert GroundAtom(Hittable, [env._launcher, blocks[-1]]).holds(state)
    assert not GroundAtom(Hittable, [env._launcher, blocks[0]]).holds(state)
    depth = depth_for_shot(state)
    end = env.launch_outcome(state, depth)
    assert end is not None
    assert all(atom.holds(end) for atom in task.goal)
    preds = set(env.predicates) | set(helpers.values())
    options = set(get_gt_options("pybullet_launcher"))
    processes = get_gt_processes("pybullet_launcher", preds, options)
    assert {pr.name for pr in processes} == {"Fire", "Wait"}
    del mod


def test_train_tower_shows_both_materials(env_module):
    """A two-block train tower is one wood and one stone block, so a test tower
    is built of masses the agent has seen."""
    _, env = env_module
    for task in env.get_train_tasks():
        colors = {
            int(round(task.init.get(b, "color")))
            for b in env._active_blocks(task.init)
        }
        assert colors == set(range(len(env.COLOR_PALETTE)))
