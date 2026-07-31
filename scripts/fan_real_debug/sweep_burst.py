"""Measure the fan_real bench's burst-duration -> ball-travel map.

This is where ``fan_real_oracle_burst_{intercept,slope}`` come from: the
oracle sampler inverts the line this script fits, so re-run it after any
change to the wind force, the ball's dynamics, or the lane layout, and
put the reported fit back into settings.py.

The sweep drives the REAL ``BlowBallToZone`` option rather than toggling
the fan directly, so the measured map includes the wind the ball picks up
while the gripper is descending onto the button and lifting off it --
which the oracle has to account for, since that is what execution does.

Usage (from the repo root):
    PYTHONPATH=. python scripts/fan_real_debug/sweep_burst.py
    PYTHONPATH=. python scripts/fan_real_debug/sweep_burst.py 0.4 0.8 1.2
"""
import logging
import sys
from typing import List

import numpy as np

from predicators import utils

# Options are built at import time from CFG, so configure before importing
# anything that reads it.
utils.reset_config({
    "env": "pybullet_fan_real",
    "pybullet_robot": "panda",
    "num_train_tasks": 0,
    "num_test_tasks": 1,
    "seed": 0,
    "horizon": 4000,
})

# pylint: disable=wrong-import-position
from predicators.envs.pybullet_fan_real import PyBulletFanRealEnv  # noqa: E402
from predicators.ground_truth_models import get_gt_options  # noqa: E402
from predicators.settings import CFG  # noqa: E402


def main() -> None:
    """Run the sweep and print the fitted line."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    bursts: List[float] = [float(a) for a in sys.argv[1:]]
    if not bursts:
        bursts = [0.5, 1.0, 1.5, 2.0, 2.5]

    env = PyBulletFanRealEnv(use_gui=False)
    option = next(iter(get_gt_options(env.get_name())))
    task = env.get_test_tasks()[0].task
    objects = [
        next(o for o in task.init if o.type.name == "robot"),
        next(o for o in task.init if o.type.name == "fan"),
        next(iter(task.goal)).objects[1],
    ]
    ball = next(o for o in task.init if o.type.name == "ball")

    travels = []
    print(f"{'burst_s':>8} {'steps':>6} {'rest_x':>8} {'travel_m':>9} "
          f"{'zone':>5}")
    for burst in bursts:
        env.reset("test", 0)
        state = env.get_observation()
        memory: dict = {}
        params = np.array([burst], dtype=np.float64)
        assert option.initiable(state, memory, objects, params)
        steps = 0
        for steps in range(1, CFG.horizon + 1):
            if option.terminal(state, memory, objects, params):
                break
            state = env.step(option.policy(state, memory, objects, params))
        rest_x = state.get(ball, "x")
        travel = rest_x - CFG.fan_real_ball_start_x
        travels.append(travel)
        zone = "-"
        for i in range(1, CFG.fan_real_num_zones + 1):
            if abs(rest_x - PyBulletFanRealEnv.zone_center_x(i)) <= \
                    CFG.fan_real_zone_len / 2:
                zone = str(i)
        print(f"{burst:8.2f} {steps:6d} {rest_x:8.4f} {travel:9.4f} "
              f"{zone:>5}")

    if len(bursts) >= 2:
        # travel = a + b * burst, so burst = (travel - a) / b, which is the
        # oracle's intercept = -a/b and slope = 1/b.
        b, a = np.polyfit(np.array(bursts), np.array(travels), 1)
        print(f"\n# travel = {a:.4f} + {b:.4f} * burst")
        print("# put these in settings.py:")
        print(f"    fan_real_oracle_burst_intercept = {-a / b:.4f}")
        print(f"    fan_real_oracle_burst_slope = {1 / b:.4f}")


if __name__ == "__main__":
    main()
