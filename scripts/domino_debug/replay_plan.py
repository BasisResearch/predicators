"""Replay an EXACT solved option plan on the real-scene domino env
(deterministic, no LLM). Grounds the plan's Pick/Place/Push/Wait options with
their exact parameters and rolls them through the env in TEST mode. With
``real_robot_execute=True`` a RealRobotExecutor is attached to the env and each
option's joint trajectory is shipped to the Franka as that option ends. The
scene is NOT re-perceived between options here: this tool replays a fixed plan,
and correcting the twin mid-replay would let the option policies see states the
recorded plan was never chosen against. The default dry-run stays pure sim
(optionally rendered to MP4).

Plan-file format (one option per line; ``-> {...}`` subgoals optional/ignored):
    Pick(robot:robot, domino_1:domino)[0.06] -> {Holding(robot, domino_1)}
    Place(robot:robot)[0.70, 1.16, 0.55, 1.75]
    Push(robot:robot, domino_0:domino)[0.03, 0.05]
    Wait(robot:robot)[]

Usage (from the predicators repo root, robot-ml; PYTHONHASHSEED=0):
    # dry-run (pure sim)
    PYTHONPATH=. python scripts/domino_debug/replay_plan.py --plan plan.txt
    # MOVES THE ARM
    PYTHONPATH=. python scripts/domino_debug/replay_plan.py --plan plan.txt \
        --execute
"""
import argparse
import logging
import os
import re
from typing import List, Tuple

import numpy as np

from predicators import utils
from predicators.approaches import create_approach
from predicators.cogman import CogMan, run_episode_and_get_observations
from predicators.envs import get_or_create_env
from predicators.envs.pybullet_domino_real import PyBulletDominoRealEnv
from predicators.execution_monitoring import create_execution_monitor
from predicators.ground_truth_models import get_gt_options
from predicators.perception import create_perceiver
from predicators.pybullet_helpers.real_robot_executor import attach_real_robot
from predicators.settings import CFG
from scripts.cluster_utils import SingleSeedRunConfig, generate_run_configs

# pylint: disable=protected-access
_LINE = re.compile(r"^\s*(\w+)\s*\(([^)]*)\)\s*\[([^\]]*)\]")


def _parse_plan(text: str) -> List[Tuple[str, List[str], List[float]]]:
    """[(option_name, [obj_names], [param_floats]), ...] from the plan text."""
    steps: List[Tuple[str, List[str], List[float]]] = []
    for raw in text.splitlines():
        line = raw.split("->", 1)[0].strip()
        if not line or line.startswith("#"):
            continue
        m = _LINE.match(line)
        if not m:
            continue
        name, args, params = m.group(1), m.group(2), m.group(3)
        objs = [
            a.split(":", 1)[0].strip() for a in args.split(",") if a.strip()
        ]
        floats = [float(v) for v in params.split(",") if v.strip()]
        steps.append((name, objs, floats))
    return steps


def main() -> None:
    """Ground the plan's options and roll them through the real-scene env."""
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", required=True, help="plan text file")
    ap.add_argument("--config", default="predicatorv3/exp_domino_real.yaml")
    ap.add_argument(
        "--scene",
        default=None,
        help="override CFG.domino_real_scene (must match the plan)")
    ap.add_argument("--execute",
                    action="store_true",
                    help="EXECUTE ON THE REAL FRANKA (needs the babyrobot "
                    "submodule installed). Default: dry-run (pure sim, no "
                    "motion).")
    ap.add_argument("--out", default=None, help="optional MP4 of the rollout")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    rc = list(generate_run_configs(args.config))[0]
    assert isinstance(rc, SingleSeedRunConfig)
    flags = dict(rc.flags)
    flags.update({"env": rc.env, "approach": rc.approach, "seed": rc.seed})
    flags.pop("log", None)
    if args.scene:
        flags["domino_real_scene"] = args.scene
    flags["real_robot_execute"] = bool(args.execute)
    # This tool replays an EXACT plan, so it does not look at the scene even
    # though the closed loop is the default elsewhere: re-syncing the twin
    # mid-replay would let the option policies see states the recorded plan was
    # never chosen against. It also keeps the tool usable with the cameras down.
    flags["real_robot_observe_at_option_boundary"] = False
    # ...and no human reset either: that rebuilds the episode's task from a
    # live look, which would rename and re-place the very objects the recorded
    # plan refers to. The task must stay the captured scene the plan was
    # written against.
    flags["real_robot_human_reset"] = False
    utils.reset_config(flags)

    env = get_or_create_env(CFG.env)
    assert isinstance(env, PyBulletDominoRealEnv), \
        f"replay_plan drives the real-scene env; got {CFG.env}"
    # Attaches the arm under --execute and is a no-op otherwise, so the env
    # stays the same object either way.
    attach_real_robot(env)
    opts = {o.name: o for o in get_gt_options(env.get_name())}
    env_task = env.get_test_tasks()[0]
    task = env_task.task
    by_name = {o.name: o for o in task.init}

    with open(args.plan, encoding="utf-8") as f:
        steps = _parse_plan(f.read())
    assert steps, "no plan steps parsed"

    plan = []
    for name, obj_names, params in steps:
        option = opts[name]
        objs = [by_name[n] for n in obj_names]
        plan.append(option.ground(objs, np.array(params, dtype=np.float32)))
    print("# grounded plan:")
    for g in plan:
        print("   ", g.simple_str())
    print(f"# EXECUTE_REAL = {CFG.real_robot_execute}  "
          f"(in-process RealRobot, dry={CFG.real_robot_dry})" if CFG.
          real_robot_execute else "# DRY-RUN (pure sim, no motion)")

    policy = utils.option_plan_to_policy(
        plan, abstract_function=lambda s: utils.abstract(s, env.predicates))
    monitor = utils.VideoMonitor(env.render) if args.out else None

    # Roll out through CogMan's episode loop -- the same one main.py uses --
    # driven by an override policy, exactly as the online-learning path does
    # (main.py sets the override, then resets). With an override in place
    # CogMan never asks the approach to solve, so the plan being replayed is
    # the plan that executes. That is also why the approach and monitor below
    # are the cheap ones: neither is consulted for control, and "oracle" plus
    # "trivial" avoid dragging an LLM into a debug replay.
    cogman = CogMan(
        create_approach("oracle", env.predicates,
                        get_gt_options(env.get_name()), env.types,
                        env.action_space, [task]),
        create_perceiver(CFG.perceiver), create_execution_monitor("trivial"))
    cogman.set_override_policy(policy)
    cogman.set_termination_function(lambda s: False)
    cogman.reset(env_task)
    (_, actions), _, _ = run_episode_and_get_observations(
        cogman,
        env,
        "test",
        0,
        max_num_steps=CFG.horizon,
        terminate_on_goal_reached=False,
        exceptions_to_break_on={utils.OptionExecutionFailure},
        monitor=monitor)
    print(f"# steps={len(actions)}  goal_reached={env.goal_reached()}")

    if args.out and monitor is not None:
        os.makedirs(os.path.join(CFG.video_dir,
                                 os.path.dirname(args.out) or "."),
                    exist_ok=True)
        utils.save_video(args.out, monitor.get_video())
        print(f"# saved {os.path.join(CFG.video_dir, args.out)}")


if __name__ == "__main__":
    main()
