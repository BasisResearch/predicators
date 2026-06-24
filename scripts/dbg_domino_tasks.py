"""Dump domino test-task geometry (roles + poses), no physics.

Usage: python scripts/dbg_domino_tasks.py [seed]
"""
import sys

import numpy as np

from predicators import utils
from predicators.envs.pybullet_domino.components.domino_component import \
    DominoComponent
from predicators.envs.pybullet_domino.env import PyBulletDominoEnv

_SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
utils.reset_config({
    "env": "pybullet_domino",
    "seed": _SEED,
    "num_train_tasks": 1,
    "num_test_tasks": 5,
    "domino_use_domino_blocks_as_target": True,
    "domino_use_continuous_place": True,
    "domino_restricted_push": True,
    "domino_initialize_at_finished_state": True,
    "domino_has_glued_dominos": False,
})

env = PyBulletDominoEnv()
tasks = env._generate_test_tasks()  # pylint: disable=protected-access

for ti, task in enumerate(tasks):
    s = task.init
    dt = None
    for o in s:
        if o.type.name == "domino":
            dt = o.type
            break
    dominoes = sorted((o for o in s if o.type == dt), key=lambda o: o.name)
    print(f"\n===== TASK {ti+1} =====")
    print("goal:", sorted(str(a) for a in task.goal))
    for d in dominoes:
        x = s.get(d, "x")
        y = s.get(d, "y")
        yaw = np.degrees(s.get(d, "yaw"))
        # pylint: disable=protected-access
        is_start = DominoComponent._StartBlock_holds(s, [d])
        is_target = DominoComponent._TargetDomino_holds(s, [d])
        is_movable = DominoComponent._MovableBlock_holds(s, [d]) \
            if hasattr(DominoComponent, "_MovableBlock_holds") else (
                not is_start and not is_target)
        role = ("START" if is_start else
                "TARGET" if is_target else "MOVABLE" if is_movable else "?")
        print(f"  {d.name:10s} {role:8s} pos=({x:.3f},{y:.3f}) yaw={yaw:6.1f}")
