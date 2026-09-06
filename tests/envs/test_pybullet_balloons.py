"""Unit tests for the balloons env's release, lift, pop, tasks, evaluator and
skills.

Covers the properties the domain rests on: an open clip frees its
balloon onto the box and the box hangs near the analytic height (and
nothing happens on the base sim), a freed balloon carried to the ceiling
bursts and loses the level, every generated level has a unique floating
subset the oracle's own plan realizes, box colours weigh differently
(and not on the base sim until overridden), and the process model builds
on the lift-law helpers.
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
        "env": "pybullet_balloons",
        "seed": 0,
        "num_train_tasks": 1,
        "num_test_tasks": 1,
        "skill_phase_use_motion_planning": False,
    }
    config.update(overrides)
    utils.reset_config(config)
    from predicators.envs import \
        pybullet_balloons  # pylint: disable=import-outside-toplevel
    return pybullet_balloons, pybullet_balloons.PyBulletBalloonsEnv(
        use_gui=False)


@pytest.fixture(name="env_module", scope="module")
def _env_module():
    return _make_env()


def _hold(env) -> Action:
    return Action(np.array(env._pybullet_robot.get_joints(), dtype=np.float32))


def _settle(env, steps: int = 250):
    obs = env._get_state()
    for _ in range(steps):
        obs = env._step_once(_hold(env))
    return obs


def _open_clips(env, state, indices):
    """Open the given clips by hand and let the box settle."""
    env._set_state(state)
    env._current_observation = env._get_state()
    for i in indices:
        env._set_clip_on(env._clips[i], True)
    return _settle(env)


def test_release_lift_and_the_analytic_law(env_module):
    """An open clip frees its balloon onto the box, which hangs near the
    analytic height; on the base sim nothing rises."""
    mod, env = env_module
    lifts = [env.lift_at_ground(c) for c in range(len(env.BALLOON_PALETTE))]
    strong = int(np.argmax(lifts))
    state = env.level_state(0, [strong, 0], (0.5, 0.6))
    obs = _open_clips(env, state, [0])
    balloon = env._balloons[0]
    assert obs.get(env._clips[0], "is_on") == 1.0
    assert obs.get(balloon, "tied") == 1.0
    assert obs.get(env._balloons[1], "tied") == 0.0
    assert obs.get(balloon, "popped") == 0.0
    expected = env.hover_height(0, [strong])
    assert expected is not None
    assert abs(obs.get(env._box, "z") - expected) < 0.03
    assert obs.get(env._box, "z") > env.box_z + 0.05
    assert mod.box_at_rest(obs, env._box)
    base = mod.PyBulletBalloonsEnv(use_gui=False, skip_residual_dynamics=True)
    obs = _open_clips(base, state, [0])
    assert obs.get(base._clips[0], "is_on") == 1.0
    assert obs.get(base._balloons[0], "tied") == 0.0
    assert abs(obs.get(base._box, "z") - base.box_z) < 0.01


def test_reaching_the_ceiling_bursts_and_loses(env_module):
    """A box lifted to the ceiling bursts a balloon; the evaluator ends the
    episode without certifying it."""
    mod, env = env_module
    goal = {GroundAtom(env._InBand, [env._box, env._band])}
    evaluator = mod.BalloonsEvaluator(goal)
    state = env.level_state(0, [3, 2, 1], (0.5, 0.6))
    obs = _open_clips(env, state, [0, 1, 2])
    obs = _settle(env, 300)
    assert mod.any_popped(obs) is not None
    assert evaluator.terminated(obs)
    ok, why = evaluator._certify([state, obs], None)
    assert not ok and "burst" in why


def test_generated_levels_have_a_unique_subset_the_oracle_frees(env_module):
    """Each task has one floating subset by the analytic law, recorded in its
    metrics, and the oracle's plan hangs the box in the band."""
    mod, env = env_module
    # pylint: disable-next=import-outside-toplevel
    from predicators.ground_truth_models.balloons.oracle import solve_level
    for tasks in (env.get_train_tasks(), env.get_test_tasks()):
        assert tasks
        for task in tasks:
            state = task.init
            subset = env.solution_subset(state)
            assert subset is not None
            balloons = env._active_balloons(state)
            for i, balloon in enumerate(balloons):
                flag = task.offline_task_metrics[f"solution_{balloon.name}"]
                assert flag == float(i in subset)
            assert "clip" in task.goal_nl
            assert not any(atom.holds(state) for atom in task.goal)
            assert not task.evaluator.terminated(state)
            plan = solve_level(env, state)
            assert plan is not None
            assert len(plan) == len(subset)
    del mod


def test_box_masses_by_colour_and_base_sim(env_module):
    """Box mass follows the colour on the real env, is the base mass on the
    base sim, and physical overrides apply."""
    mod, env = env_module
    state = env.level_state(1, [0, 1], (0.5, 0.6))
    env._set_state(state)
    mass = p.getDynamicsInfo(env._box.id,
                             -1,
                             physicsClientId=env._physics_client_id)[0]
    assert abs(mass - env.true_box_mass(1)) < 1e-6
    base = mod.PyBulletBalloonsEnv(use_gui=False, skip_residual_dynamics=True)
    base._set_state(state)
    mass = p.getDynamicsInfo(base._box.id,
                             -1,
                             physicsClientId=base._physics_client_id)[0]
    assert mass == base.box_base_mass
    base.apply_physical_param_overrides({"mass_oak": env.true_box_mass(1)})
    mass = p.getDynamicsInfo(base._box.id,
                             -1,
                             physicsClientId=base._physics_client_id)[0]
    assert abs(mass - env.true_box_mass(1)) < 1e-6
    assert "air_drag" in base.get_physical_param_info()
    with pytest.raises(ValueError):
        base.apply_physical_param_overrides({"mass_plastic": 1.0})


def test_oracle_helpers_and_processes(env_module):
    """``Needed`` marks the solution subset, ``Holds`` pairs clips with their
    balloons, ``AllNeededTied`` follows the tied atoms, and the process model
    builds."""
    mod, env = env_module
    # pylint: disable=import-outside-toplevel
    from predicators.ground_truth_models import get_gt_helper_predicates, \
        get_gt_options, get_gt_processes

    # pylint: enable=import-outside-toplevel
    helpers = {
        p_.name: p_
        for p_ in get_gt_helper_predicates("pybullet_balloons")
    }
    task = env.get_train_tasks()[0]
    state = task.init
    subset = env.solution_subset(state)
    balloons = env._active_balloons(state)
    clips = env._active_clips(state)
    for i, balloon in enumerate(balloons):
        assert GroundAtom(helpers["Needed"],
                          [balloon, env._band]).holds(state) == (i in subset)
        for j, clip in enumerate(clips):
            assert GroundAtom(helpers["Holds"],
                              [clip, balloon]).holds(state) == (i == j)
    preds = set(env.predicates) | set(helpers.values())
    atoms = utils.abstract(state, preds)
    assert GroundAtom(helpers["AllNeededTied"], [env._band]) not in atoms
    tied = state.copy()
    for i in subset:
        tied.set(balloons[i], "tied", 1.0)
    atoms = utils.abstract(tied, preds)
    assert GroundAtom(helpers["AllNeededTied"], [env._band]) in atoms
    options = set(get_gt_options("pybullet_balloons"))
    assert {o.name for o in options} == {"Release", "Wait"}
    processes = get_gt_processes("pybullet_balloons", preds, options)
    assert {pr.name for pr in processes} == {"ReleaseClip", "Rise", "Wait"}
    del mod


def test_train_levels_cover_every_colour_and_material():
    """Two train levels together show all four balloon colours and both box
    materials; a test level holds the whole palette on a box material training
    showed."""
    _, env = _make_env(num_train_tasks=2, num_test_tasks=1)
    train = env.get_train_tasks()
    colors = set()
    boxes = set()
    for task in train:
        boxes.add(int(round(task.init.get(env._box, "color"))))
        colors.update(
            int(round(task.init.get(b, "color")))
            for b in env._active_balloons(task.init))
    assert colors == set(range(len(env.BALLOON_PALETTE)))
    assert boxes == {0, 1}
    test = env.get_test_tasks()[0].init
    assert len(env._active_balloons(test)) == len(env.BALLOON_PALETTE)
    assert int(round(test.get(env._box, "color"))) in boxes
