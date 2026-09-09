"""Unit tests for the crane env's swing, tasks, evaluator and skills.

Covers the properties the domain rests on: a drawn-back ram swings and
sends the crate down the lane, further for a longer pull, and a crate
colour's material changes how far (and not on the base sim until the
parameters are overridden); every generated level has a working pull
window whose middle lands the crate on the pad; a crate off the table or
out of reach loses the level; and the process model builds on the swing-
probe helper.
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
        "env": "pybullet_crane",
        "seed": 0,
        "num_train_tasks": 1,
        "num_test_tasks": 1,
        "skill_phase_use_motion_planning": False,
    }
    config.update(overrides)
    utils.reset_config(config)
    from predicators.envs import \
        pybullet_crane  # pylint: disable=import-outside-toplevel
    return pybullet_crane, pybullet_crane.PyBulletCraneEnv(use_gui=False)


@pytest.fixture(name="env_module", scope="module")
def _env_module():
    return _make_env()


def test_pull_swings_the_ram_into_the_crate(env_module):
    """A longer pull sends the crate further; a short one leaves it; the crate
    never leaves the lane."""
    mod, env = env_module
    state = env.level_state(1.3, 0.5, 0.22, 0, 1.0, 0.06)
    travel = []
    for pull in (0.10, 0.18, 0.26):
        end = env.swing_outcome(state, pull)
        assert end is not None
        assert not mod.crate_fallen(end, env._crate)
        assert abs(end.get(env._crate, "y") - 1.3) < 0.03
        assert mod.crate_at_rest(end, env._crate)
        travel.append(end.get(env._crate, "x") - (env.ram_x + 0.22))
    assert travel[0] < 0.01
    assert travel[1] > 0.05
    assert travel[2] > travel[1] + 0.05


def test_materials_change_the_slide_and_base_sim_does_not(env_module):
    """The same pull sends a foam crate further than an iron one on the real
    env; on the base sim both crates weigh and grip the same until the
    parameters are overridden, and unknown parameters are refused."""
    mod, env = env_module
    ends = []
    for color in (0, 1):
        state = env.level_state(1.3, 0.5, 0.22, color, 1.0, 0.06)
        end = env.swing_outcome(state, 0.22)
        assert end is not None
        ends.append(end.get(env._crate, "x"))
    assert ends[0] > ends[1] + 0.03
    base = mod.PyBulletCraneEnv(use_gui=False, skip_residual_dynamics=True)
    masses = []
    for color in (0, 1):
        base._set_state(env.level_state(1.3, 0.5, 0.22, color, 1.0, 0.06))
        masses.append(
            p.getDynamicsInfo(base._crate.id,
                              -1,
                              physicsClientId=base._physics_client_id)[0])
    assert masses[0] == masses[1] == base.crate_base_mass
    base.apply_physical_param_overrides({"mass_iron": env.true_crate_mass(1)})
    base._set_state(env.level_state(1.3, 0.5, 0.22, 1, 1.0, 0.06))
    mass = p.getDynamicsInfo(base._crate.id,
                             -1,
                             physicsClientId=base._physics_client_id)[0]
    assert abs(mass - env.true_crate_mass(1)) < 1e-6
    assert "swing_damping" in base.get_physical_param_info()
    with pytest.raises(ValueError):
        base.apply_physical_param_overrides({"mass_glass": 1.0})


def test_generated_levels_have_a_working_pull(env_module):
    """Each task's recorded pull lands the crate at rest on the pad; the goal
    does not hold at the start."""
    mod, env = env_module
    for tasks in (env.get_train_tasks(), env.get_test_tasks()):
        assert tasks
        for task in tasks:
            state = task.init
            assert "ram" in task.goal_nl
            assert not any(atom.holds(state) for atom in task.goal)
            assert not task.evaluator.terminated(state)
            pull = task.offline_task_metrics["solution_pull"]
            end = env.swing_outcome(state, pull)
            assert end is not None
            assert mod.crate_in_bin(end, env._crate, env._bin)
            assert all(atom.holds(end) for atom in task.goal)
            assert task.evaluator.terminated(end)
            ok, _ = task.evaluator._certify([state, end], None)
            assert ok


def test_losing_the_crate_ends_the_level(env_module):
    """A crate off the table, or stopped beyond the ram's reach, terminates the
    level without a win."""
    mod, env = env_module
    goal = {GroundAtom(env._InBin, [env._crate, env._bin])}
    evaluator = mod.CraneEvaluator(goal)
    state = env.level_state(1.3, 0.5, 0.22, 0, 1.0, 0.06)
    far = state.copy()
    far.set(env._crate, "x", 1.24)
    env._set_state(far)
    env._current_observation = env._get_state()
    hold = Action(np.array(env._pybullet_robot.get_joints(), dtype=np.float32))
    obs = env._get_state()
    for _ in range(80):
        obs = env._step_once(hold)
    assert mod.crate_fallen(obs, env._crate)
    assert evaluator.terminated(obs)
    ok, why = evaluator._certify([state, obs], None)
    assert not ok and "fell" in why
    stuck = state.copy()
    stuck.set(env._crate, "x", 1.12)
    assert not mod.crate_reachable(stuck, env._ram, env._crate)
    assert evaluator.terminated(stuck)
    ok, why = evaluator._certify([state, stuck], None)
    assert not ok and "reach" in why


def test_helpers_and_processes(env_module):
    """``Hittable`` holds for a generated level's start and the process model
    builds."""
    mod, env = env_module
    # pylint: disable=import-outside-toplevel
    from predicators.ground_truth_models import get_gt_helper_predicates, \
        get_gt_options, get_gt_processes

    # pylint: enable=import-outside-toplevel
    helpers = {
        p_.name: p_
        for p_ in get_gt_helper_predicates("pybullet_crane")
    }
    task = env.get_train_tasks()[0]
    state = task.init
    assert GroundAtom(helpers["Hittable"],
                      [env._ram, env._crate, env._bin]).holds(state)
    preds = set(env.predicates) | set(helpers.values())
    options = set(get_gt_options("pybullet_crane"))
    assert {o.name for o in options} == {"Pull", "Wait"}
    processes = get_gt_processes("pybullet_crane", preds, options)
    assert {pr.name for pr in processes} == {"Swing", "Wait"}
    del mod
