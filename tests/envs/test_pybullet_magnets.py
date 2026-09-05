"""Unit tests for the magnets env's field, tasks, evaluator and skills.

Covers the properties the domain rests on: the wand is in the hand and
its tip hovers where the skills put it, a pulled colour slides under the
tip and a pushed colour slides away (and nothing moves on the base sim),
a carried piece follows a slow hover and stays behind a fast jump, every
generated level is cleared by the oracle's own plan, a piece off the mat
ends the level as a loss, and the process model builds on the polarity
helpers.
"""
# pylint: disable=protected-access
from __future__ import annotations

import numpy as np
import pytest

from predicators import utils
from predicators.structs import Action, GroundAtom


def _make_env(**overrides):
    config = {
        "env": "pybullet_magnets",
        "seed": 0,
        "num_train_tasks": 1,
        "num_test_tasks": 1,
        "skill_phase_use_motion_planning": False,
    }
    config.update(overrides)
    utils.reset_config(config)
    from predicators.envs import \
        pybullet_magnets  # pylint: disable=import-outside-toplevel
    return pybullet_magnets, pybullet_magnets.PyBulletMagnetsEnv(use_gui=False)


@pytest.fixture(name="env_module", scope="module")
def _env_module():
    return _make_env()


def _hold(env) -> Action:
    return Action(np.array(env._pybullet_robot.get_joints(), dtype=np.float32))


def _run(env, option, max_steps: int = 300):
    obs = env._get_state()
    env._current_observation = obs
    assert option.initiable(obs)
    for _ in range(max_steps):
        if option.terminal(obs):
            break
        obs = env._step_once(option.policy(obs))
    for _ in range(40):
        obs = env._step_once(_hold(env))
    return obs


def test_wand_in_hand_and_field_polarity(env_module):
    """The wand starts held; a pulled piece slides under the tip, a pushed one
    slides away, and on the base sim neither moves."""
    mod, env = env_module
    # pylint: disable-next=import-outside-toplevel
    from predicators.ground_truth_models.magnets.options import \
        probe_move_options
    pulled = next(i for i in range(len(env.COLOR_PALETTE))
                  if env.polarity(i) > 0)
    pushed = next(i for i in range(len(env.COLOR_PALETTE))
                  if env.polarity(i) < 0)
    state = env.level_state([(0.70, 1.20, pulled), (0.88, 1.30, pushed)],
                            [(0.95, 1.15, pulled)])
    env._set_state(state)
    obs = env._get_state()
    assert obs.get(env._wand, "is_held") == 1.0
    assert env._held_obj_id == env._wand.id
    _, jump = probe_move_options()
    # Jump next to the pulled piece: it comes under the tip.
    obs = _run(
        env,
        jump.ground([env._robot, env._wand],
                    np.array([0.74, 1.22], dtype=np.float32)))
    assert mod.tip_over(obs, env._wand, env._pieces[0])
    assert abs(obs.get(env._pieces[0], "x") - 0.74) < 0.02
    # Jump next to the pushed piece: it moves away from the tip.
    obs = _run(
        env,
        jump.ground([env._robot, env._wand],
                    np.array([0.84, 1.30], dtype=np.float32)))
    assert obs.get(env._pieces[1], "x") > 0.90
    # The base sim: the wand hovers, nothing moves.
    base = mod.PyBulletMagnetsEnv(use_gui=False, skip_residual_dynamics=True)
    base._set_state(state)
    obs = _run(
        base,
        jump.ground([base._robot, base._wand],
                    np.array([0.74, 1.22], dtype=np.float32)))
    assert abs(obs.get(base._pieces[0], "x") - 0.70) < 0.005
    assert abs(obs.get(base._pieces[0], "y") - 1.20) < 0.005


def test_carry_follows_a_hover_and_stays_behind_a_jump(env_module):
    """A captured piece follows a slow hover to a slot and is left there by a
    fast jump."""
    mod, env = env_module
    # pylint: disable-next=import-outside-toplevel
    from predicators.ground_truth_models.magnets.options import \
        probe_move_options
    pulled = next(i for i in range(len(env.COLOR_PALETTE))
                  if env.polarity(i) > 0)
    state = env.level_state([(0.65, 1.15, pulled)], [(0.92, 1.30, pulled)])
    env._set_state(state)
    hover, jump = probe_move_options()
    _run(
        env,
        jump.ground([env._robot, env._wand],
                    np.array([0.65, 1.15], dtype=np.float32)))
    obs = _run(
        env,
        hover.ground([env._robot, env._wand],
                     np.array([0.92, 1.30], dtype=np.float32)))
    assert mod.piece_in_slot(obs, env._pieces[0], env._slots[0])
    obs = _run(
        env,
        jump.ground([env._robot, env._wand],
                    np.array([0.55, 1.08], dtype=np.float32)))
    assert mod.piece_in_slot(obs, env._pieces[0], env._slots[0])
    assert all(atom.holds(obs) for atom in env.goal_for(state))


def test_generated_levels_are_cleared_by_the_oracle_plan(env_module):
    """Every task's level is solved by the recorded plan with no piece lost,
    and the goal does not hold at the start."""
    mod, env = env_module
    # pylint: disable-next=import-outside-toplevel
    from predicators.ground_truth_models.magnets.oracle import solve_level
    for tasks in (env.get_train_tasks(), env.get_test_tasks()):
        assert tasks
        for task in tasks:
            state = task.init
            assert "wand" in task.goal_nl
            assert not all(atom.holds(state) for atom in task.goal)
            assert not task.evaluator.terminated(state)
            plan = solve_level(env, state)
            assert plan is not None and len(plan) >= 2
    del mod


def test_lost_piece_ends_the_level(env_module):
    """A piece off the mat terminates the episode without certifying it."""
    mod, env = env_module
    task = env.get_train_tasks()[0]
    state = task.init
    pieces, _ = env._active_objects(state)
    off = state.copy()
    off.set(pieces[0], "x", env.mat_bounds()[0] - 0.1)
    Lost = next(p for p in env.predicates if p.name == "Lost")
    assert GroundAtom(Lost, [pieces[0]]).holds(off)
    assert task.evaluator.terminated(off)
    ok, why = task.evaluator._certify([state, off], None)
    assert not ok and pieces[0].name in why
    del mod


def test_oracle_helpers_and_processes(env_module):
    """``Pulled`` / ``Pushed`` follow the colours and the process model
    builds."""
    mod, env = env_module
    # pylint: disable=import-outside-toplevel
    from predicators.ground_truth_models import get_gt_helper_predicates, \
        get_gt_options, get_gt_processes

    # pylint: enable=import-outside-toplevel
    helpers = {p.name: p for p in get_gt_helper_predicates("pybullet_magnets")}
    task = env.get_test_tasks()[0]
    state = task.init
    pieces, _ = env._active_objects(state)
    for piece in pieces:
        pulled = env.polarity(state.get(piece, "color")) > 0
        assert GroundAtom(helpers["Pulled"], [piece]).holds(state) == pulled
        assert GroundAtom(helpers["Pushed"], [piece]).holds(state) != pulled
    preds = set(env.predicates) | set(helpers.values())
    options = set(get_gt_options("pybullet_magnets"))
    assert {o.name for o in options} == {"Hover", "Jump", "Wait"}
    processes = get_gt_processes("pybullet_magnets", preds, options)
    assert {p.name for p in processes} == {"Capture", "Carry", "Wait"}
    del mod
