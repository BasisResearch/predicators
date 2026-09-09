"""Unit tests for the ice rink env's materials, patch, tasks, evaluator and
skills.

Covers the properties the domain rests on: every generated goal is
realized by the single push the generator recorded, a tile's travel from
one push depends on its colour (and not at all in the base sim, which
the physical-parameter overrides can correct), the dark strip shortens a
slide, a tile off the rink ends the level as a loss, the oracle's helper
predicates and sampler find the recorded solutions, and the shared push
skill's speed parameter sets how far a tile travels.
"""
# pylint: disable=protected-access
from __future__ import annotations

import numpy as np
import pytest

from predicators import utils
from predicators.structs import Action, GroundAtom


def _make_env(**overrides):
    """A freshly configured env plus its module."""
    config = {
        "env": "pybullet_icerink",
        "seed": 0,
        "num_train_tasks": 2,
        "num_test_tasks": 1,
        "skill_phase_use_motion_planning": False,
    }
    config.update(overrides)
    utils.reset_config(config)
    from predicators.envs import \
        pybullet_icerink  # pylint: disable=import-outside-toplevel
    return pybullet_icerink, pybullet_icerink.PyBulletIceRinkEnv(use_gui=False)


@pytest.fixture(name="env_module", scope="module")
def _env_module():
    return _make_env()


def _tile_state(mod, env, tiles):
    """A state with the given ``(x, y, color)`` tiles, targets parked on them,
    and the robot at home."""
    init = {env._robot: env._robot_init_dict()}
    for direction in env._directions:
        init[direction] = {"yaw": env.DIRECTIONS[direction.name]}
    for i, (x, y, color) in enumerate(tiles):
        init[env._tiles[i]] = {
            "x": x,
            "y": y,
            "z": env.tile_z,
            "rot": 0.0,
            "color": float(color),
            "speed": 0.0,
        }
        init[env._targets[i]] = {
            "x": x,
            "y": y,
            "z": env.target_z,
            "color": float(color),
        }
    del mod
    return utils.create_state_from_dict(init)


def test_geometry_and_types(env_module):
    """The rink, its directions and its palette are well formed."""
    mod, env = env_module
    x_min, x_max, y_min, y_max = env.rink_bounds()
    assert x_min < x_max and y_min < y_max
    assert env.on_rink(env.rink_center[0], env.rink_center[1])
    assert not env.on_rink(x_min - 0.1, env.rink_center[1])
    assert set(env.DIRECTIONS) == {"north", "east", "south", "west"}
    assert set(mod.PLANNED_DIRECTIONS) <= set(env.DIRECTIONS)
    names = {t.name for t in env.types}
    assert names == {"robot", "tile", "target", "direction"}
    colors = [name for name, _ in env.COLOR_PALETTE]
    assert len(colors) == len(set(colors))
    assert len(env.get_physical_param_info()) == len(colors)


def test_generated_goals_are_realized_by_the_recorded_push(env_module):
    """Each task's tiles land on their targets under the generator's own
    (direction, speed), and the goal then holds with every tile at rest."""
    mod, env = env_module
    for split, tasks in (("train", env.get_train_tasks()),
                         ("test", env.get_test_tasks())):
        assert tasks, split
        for task in tasks:
            state = task.init
            tiles, targets = env._active_objects(state)
            assert len(tiles) == len(targets) >= 2
            assert task.goal_nl and "Slide" in task.goal_nl
            colors = {int(state.get(t, "color")) for t in tiles}
            assert len(colors) == len(tiles), "materials are distinct"
            metrics = task.offline_task_metrics
            for i, (tile, target) in enumerate(zip(tiles, targets)):
                direction = mod.PLANNED_DIRECTIONS[int(
                    metrics[f"solution_tile{i}_direction"])]
                speed = metrics[f"solution_tile{i}_speed"]
                outcome = env.push_outcome(state, tile, direction, speed)
                assert outcome is not None
                x, y, lost = outcome
                assert not lost
                assert abs(x - state.get(target, "x")) < 0.03
                assert abs(y - state.get(target, "y")) < 0.03
            # Not on its target yet at the start.
            assert not any(atom.holds(state) for atom in task.goal)
            assert not task.evaluator.terminated(state)


def test_material_sets_travel_and_base_sim_does_not(env_module):
    """Two tiles of different colours travel different distances from the same
    push; the base sim slides every colour the same, until the physical
    parameters are overridden to the true values."""
    mod, env = env_module
    x0, y0 = 0.60, 1.15
    travel = {}
    for color in range(len(env.COLOR_PALETTE)):
        state = _tile_state(mod, env, [(x0, y0, color)])
        outcome = env.push_outcome(state, env._tiles[0], "north", 0.35)
        assert outcome is not None and not outcome[2]
        travel[color] = outcome[1] - y0
    frictions = list(env.true_friction(c) for c in travel)
    # Higher friction, shorter slide (blue ice reaches the wall).
    order = sorted(travel, key=lambda c: frictions[c])
    assert travel[order[0]] > travel[order[-1]] + 0.05
    # Blue ice slides farthest, or reaches the far wall alongside another
    # colour that also does.
    assert travel[0] >= travel[max(travel, key=travel.get)] - 0.01

    base = mod.PyBulletIceRinkEnv(use_gui=False, skip_residual_dynamics=True)
    base_travel = {}
    for color in (0, 1):
        state = _tile_state(mod, env, [(x0, y0, color)])
        outcome = base.push_outcome(state, base._tiles[0], "north", 0.35)
        assert outcome is not None
        base_travel[color] = outcome[1] - y0
    assert abs(base_travel[0] - base_travel[1]) < 0.01
    name = env.color_name(1)
    base.apply_physical_param_overrides(
        {f"friction_{name}": env.true_friction(1)})
    state = _tile_state(mod, env, [(x0, y0, 1)])
    outcome = base.push_outcome(state, base._tiles[0], "north", 0.35)
    assert outcome is not None
    assert abs((outcome[1] - y0) - travel[1]) < 0.015
    with pytest.raises(ValueError):
        base.apply_physical_param_overrides({"friction_plaid": 0.5})


def test_patch_drags_a_crossing_tile(env_module):
    """A slide across the dark strip is shorter than the same slide with the
    strip disabled."""
    mod, env = env_module
    lo, hi = env.patch_x_bounds
    # A grey tile west of the strip, pushed east through it.
    x0, y0 = lo - 0.12, 1.15
    state = _tile_state(mod, env, [(x0, y0, 3)])
    with_patch = env.push_outcome(state, env._tiles[0], "east", 0.38)
    utils.reset_config({
        "env": "pybullet_icerink",
        "seed": 0,
        "icerink_patch": False,
        "skill_phase_use_motion_planning": False,
    })
    without = env.push_outcome(state, env._tiles[0], "east", 0.38)
    utils.reset_config({
        "env": "pybullet_icerink",
        "seed": 0,
        "num_train_tasks": 2,
        "num_test_tasks": 1,
        "skill_phase_use_motion_planning": False,
    })
    assert with_patch is not None and without is not None
    assert without[0] > with_patch[0] + 0.02
    assert with_patch[0] > lo - 0.01, "it did enter the strip"
    del hi


def test_lost_tile_ends_the_level_as_a_loss(env_module):
    """A tile off the slab is Lost, the evaluator terminates on it, and it
    refuses to certify the episode."""
    mod, env = env_module
    task = env.get_train_tasks()[0]
    state = task.init
    tiles, targets = env._active_objects(state)
    x_min, _, _, _ = env.rink_bounds()
    off = state.copy()
    off.set(tiles[0], "x", x_min - 0.2)
    Lost = next(p for p in env.predicates if p.name == "Lost")
    OnRink = next(p for p in env.predicates if p.name == "OnRink")
    assert GroundAtom(Lost, [tiles[0]]).holds(off)
    assert not GroundAtom(OnRink, [tiles[0]]).holds(off)
    assert not GroundAtom(Lost, [tiles[1]]).holds(off)
    assert task.evaluator.terminated(off)
    ok, why = task.evaluator._certify([state, off], None)
    assert not ok and "tile0" in why
    assert task.evaluator.reward([state, off], None) <= 0.0
    # The goal with every tile on its target and at rest is a win.
    won = state.copy()
    for tile, target in zip(tiles, targets):
        won.set(tile, "x", state.get(target, "x"))
        won.set(tile, "y", state.get(target, "y"))
    assert task.evaluator.terminated(won)
    assert task.evaluator.solved([state, won], None)
    # Still sliding through the target: not yet.
    won.set(tiles[0], "speed", 0.2)
    assert not task.evaluator.terminated(won)
    del mod


def test_oracle_helpers_and_sampler_find_the_solution(env_module):
    """``Reachable`` holds for the generator's push, the process model builds,
    and the sampler's speed lands the tile."""
    mod, env = env_module
    # pylint: disable=import-outside-toplevel
    from predicators.ground_truth_models import get_gt_helper_predicates, \
        get_gt_options, get_gt_processes
    from predicators.ground_truth_models.icerink.processes import \
        speed_for_target

    # pylint: enable=import-outside-toplevel
    helpers = {p.name: p for p in get_gt_helper_predicates("pybullet_icerink")}
    task = env.get_train_tasks()[0]
    state = task.init
    tiles, targets = env._active_objects(state)
    directions = {d.name: d for d in env._directions}
    metrics = task.offline_task_metrics
    for i, (tile, target) in enumerate(zip(tiles, targets)):
        direction = directions[mod.PLANNED_DIRECTIONS[int(
            metrics[f"solution_tile{i}_direction"])]]
        assert GroundAtom(helpers["Reachable"],
                          [tile, direction, target]).holds(state)
        speed = speed_for_target(state, tile, direction, target)
        outcome = env.push_outcome(state, tile, direction.name, speed)
        assert outcome is not None and not outcome[2]
        assert abs(outcome[0] - state.get(target, "x")) < 0.03
        assert abs(outcome[1] - state.get(target, "y")) < 0.03
    # A south push is never planned.
    assert not GroundAtom(
        helpers["Reachable"],
        [tiles[0], directions["south"], targets[0]]).holds(state)

    preds = set(env.predicates) | set(helpers.values())
    options = set(get_gt_options("pybullet_icerink"))
    processes = get_gt_processes("pybullet_icerink", preds, options)
    assert {p.name for p in processes} == {"PushToTarget", "Wait"}


def test_push_skill_speed_parameter(env_module):
    """The real Push skill moves a tile farther at a higher stroke speed, and
    the option terminates with the arm back home."""
    mod, env = env_module
    # pylint: disable-next=import-outside-toplevel
    from predicators.ground_truth_models import get_gt_options
    Push = next(o for o in get_gt_options("pybullet_icerink")
                if o.name == "Push")
    assert Push.params_space.shape == (3, )
    state = _tile_state(mod, env, [(0.65, 1.15, 1)])
    north = next(d for d in env._directions if d.name == "north")
    travel = {}
    for speed in (0.2, 0.35):
        env._set_state(state)
        obs = env._get_state()
        env._current_observation = obs
        option = Push.ground([env._robot, env._tiles[0], north],
                             np.array([0.07, 0.03, speed], dtype=np.float32))
        assert option.initiable(obs)
        for _ in range(300):
            if option.terminal(obs):
                break
            obs = env._step_once(option.policy(obs))
        assert option.terminal(obs)
        hold = Action(
            np.array(env._pybullet_robot.get_joints(), dtype=np.float32))
        for _ in range(200):
            obs = env._step_once(hold)
            if obs.get(env._tiles[0], "speed") < 0.01:
                break
        travel[speed] = obs.get(env._tiles[0], "y") - 1.15
        assert abs(obs.get(env._tiles[0], "x") - 0.65) < 0.02
    assert travel[0.35] > travel[0.2] + 0.03
