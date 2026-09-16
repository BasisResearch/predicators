"""Tests for the fan env's maze task generation."""
from typing import Set, Tuple

import numpy as np
import pybullet as p
import pytest

from predicators import utils
from predicators.envs.pybullet_fan import PyBulletFanEnv
from predicators.settings import CFG
from predicators.structs import State

Cell = Tuple[int, int]


def _cell(x: float, y: float, xs: list, ys: list) -> Cell:
    return (int(np.argmin([abs(c - x) for c in xs])),
            int(np.argmin([abs(c - y) for c in ys])))


def _layout(env: PyBulletFanEnv, state: State, num_x: int,
            num_y: int) -> Tuple[Cell, Cell, Set[Cell]]:
    xs, ys = env._generate_grid_coordinates(num_x, num_y)  # pylint: disable=protected-access
    ball = next(o for o in state if o.type.name == "ball")
    target = next(o for o in state if o.type.name == "target")
    walls = {
        _cell(state.get(o, "x"), state.get(o, "y"), xs, ys)
        for o in state if o.type.name == "wall"
    }
    return (_cell(state.get(ball, "x"), state.get(ball, "y"), xs, ys),
            _cell(state.get(target, "x"), state.get(target, "y"), xs,
                  ys), walls)


def test_min_segment_path_and_count() -> None:
    """The route search minimises straight runs, not cells."""
    walls: Set[Cell] = set()
    # Same row, nothing between: one run.
    path = PyBulletFanEnv._min_segment_path((0, 0), (3, 0), walls, 4, 4)  # pylint: disable=protected-access
    assert path == [(0, 0), (1, 0), (2, 0), (3, 0)]
    assert PyBulletFanEnv._count_segments(path) == 1  # pylint: disable=protected-access
    # Off-axis: two runs.
    path = PyBulletFanEnv._min_segment_path((0, 0), (3, 2), walls, 4, 4)  # pylint: disable=protected-access
    assert path is not None
    assert path[0] == (0, 0) and path[-1] == (3, 2)
    assert PyBulletFanEnv._count_segments(path) == 2  # pylint: disable=protected-access
    # A wall across the row forces a detour of three runs, even though
    # cells (0,0) and (3,0) are still three steps apart in Manhattan terms.
    walls = {(2, 0)}
    path = PyBulletFanEnv._min_segment_path((0, 0), (3, 0), walls, 4, 4)  # pylint: disable=protected-access
    assert path is not None
    assert (2, 0) not in path
    assert PyBulletFanEnv._count_segments(path) == 3  # pylint: disable=protected-access
    # Unreachable target.
    walls = {(1, 0), (1, 1), (1, 2), (1, 3)}
    blocked = PyBulletFanEnv._min_segment_path((0, 0), (3, 0), walls, 4, 4)  # pylint: disable=protected-access
    assert blocked is None
    assert PyBulletFanEnv._count_segments([(0, 0)]) == 0  # pylint: disable=protected-access
    trivial = PyBulletFanEnv._min_segment_path((1, 1), (1, 1), set(), 3, 3)  # pylint: disable=protected-access
    assert trivial == [(1, 1)]


def test_free_cells_connected() -> None:
    """Connectivity is over non-wall cells with cardinal moves."""
    assert PyBulletFanEnv._free_cells_connected(set(), 3, 3)  # pylint: disable=protected-access
    assert PyBulletFanEnv._free_cells_connected({(1, 0), (1, 1)}, 3, 3)  # pylint: disable=protected-access
    assert not PyBulletFanEnv._free_cells_connected(  # pylint: disable=protected-access
        {(1, 0), (1, 1), (1, 2)}, 3, 3)
    assert not PyBulletFanEnv._free_cells_connected(  # pylint: disable=protected-access
        {(0, 0), (1, 0), (0, 1), (1, 1)}, 2, 2)


def test_maze_test_tasks() -> None:
    """Default test tasks are mazes on the full arena grid."""
    utils.reset_config({
        "env": "pybullet_fan",
        "seed": 0,
        "num_train_tasks": 1,
        "num_test_tasks": 3,
    })
    assert CFG.fan_test_task_generation == "maze"
    env = PyBulletFanEnv(use_gui=False)
    try:
        num_x, num_y = CFG.fan_test_num_pos_x, CFG.fan_test_num_pos_y
        xs, ys = env._generate_grid_coordinates(num_x, num_y)  # pylint: disable=protected-access
        # The grid sits just in front of the fan rows: the boundary slabs
        # lie between the fan faces and the outermost cells.
        assert env.left_fan_x + env.fan_x_len / 2 < min(xs) - env.pos_gap / 2
        assert env.right_fan_x - env.fan_x_len / 2 > max(xs) + env.pos_gap / 2
        assert env.down_fan_y + env.fan_x_len / 2 < min(ys) - env.pos_gap / 2
        assert env.up_fan_y - env.fan_x_len / 2 > max(ys) + env.pos_gap / 2
        for task in env.get_test_tasks():
            ball, target, walls = _layout(env, task.init, num_x, num_y)
            assert len(walls) in CFG.fan_test_num_walls_per_task
            assert ball not in walls and target not in walls
            assert ball != target
            assert env._free_cells_connected(walls, num_x, num_y)  # pylint: disable=protected-access
            path = env._min_segment_path(ball, target, walls, num_x, num_y)  # pylint: disable=protected-access
            assert path is not None
            assert env._count_segments(  # pylint: disable=protected-access
                path) >= CFG.fan_maze_min_segments
            assert len(path) - 1 >= CFG.fan_maze_min_path_len
            # Every object sits exactly on a grid cell.
            for obj in task.init:
                if obj.type.name in ("ball", "target", "wall"):
                    x, y = task.init.get(obj, "x"), task.init.get(obj, "y")
                    assert min(abs(x - c) for c in xs) < 1e-6
                    assert min(abs(y - c) for c in ys) < 1e-6
            # The goal is grid-free: the physical target plus all fans off.
            assert any(a.predicate.name == "BallAtTarget" for a in task.goal)
        # Train tasks keep the uniform 3x3 generator.
        train = env.get_train_tasks()[0]
        train_walls = [o for o in train.init if o.type.name == "wall"]
        assert len(train_walls) == 1
    finally:
        p.disconnect(env._physics_client_id)  # pylint: disable=protected-access


def test_maze_generation_is_deterministic() -> None:
    """Two envs with the same seed produce identical maze layouts."""
    layouts = []
    for _ in range(2):
        utils.reset_config({
            "env": "pybullet_fan",
            "seed": 3,
            "num_train_tasks": 1,
            "num_test_tasks": 2,
        })
        env = PyBulletFanEnv(use_gui=False)
        try:
            layouts.append([
                _layout(env, t.init, CFG.fan_test_num_pos_x,
                        CFG.fan_test_num_pos_y) for t in env.get_test_tasks()
            ])
        finally:
            p.disconnect(env._physics_client_id)  # pylint: disable=protected-access
    assert layouts[0] == layouts[1]
    assert layouts[0][0] != layouts[0][1]


def test_unknown_generation_rejected() -> None:
    """An unknown generation mode fails loudly."""
    utils.reset_config({
        "env": "pybullet_fan",
        "seed": 0,
        "num_train_tasks": 1,
        "num_test_tasks": 1,
        "fan_test_task_generation": "spiral",
    })
    env = PyBulletFanEnv(use_gui=False)
    try:
        with pytest.raises(ValueError, match="Unknown fan task generation"):
            env.get_test_tasks()
    finally:
        p.disconnect(env._physics_client_id)  # pylint: disable=protected-access
