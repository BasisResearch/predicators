"""Measure the live ZED scene against the simulated twin. NOTHING MOVES.

Stage 2 of the real-robot bring-up ladder: a dry arm (``RealRobot`` is built
with no arm at all, so this runs with the robot powered down and needs no
polymetis controller) and live cameras. It answers the one question worth
answering before any motion -- does what the cameras see land in the right
place in the twin?

It goes through ``make_real_robot`` and the env's own
``state_from_observation``, i.e. the exact conversion ``RealRobotExecutor``
uses at an option boundary, so a disagreement here is a real disagreement and
not an artifact of a parallel code path.

The three checks (run this once per check; the table is the point):

  1. Static.  Untouched scene. Divergence sits at the perception noise floor.
     Use ``--repeat`` and write the number down: it is what
     ``real_robot_divergence_atol`` should be set from, and today that ships
     as a guessed 0.02.
  2. Moved.   Physically slide one domino ~5cm and re-run. Divergence should
     report ~0.05 -- and THAT domino should be the one that moved. A different
     one moving means the capture-id -> slot mapping is wrong, which is the
     failure this stage exists to catch.
  3. Toppled. Lay one domino on its face. ``Toppled`` should read True for it.

Watch the z column separately. A *constant* z offset across every domino is a
table-height disagreement (``domino_real_table_z``, -0.041), not a perception
error -- fix the number, not the geometry.

Usage (from the predicators repo root, robot-ml; PYTHONHASHSEED=0):
    PYTHONPATH=.:/path/to/BabyRobotPredicator \
        python scripts/domino_debug/check_perception.py --repeat 5
"""
import argparse
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from predicators import utils
from predicators.envs import get_or_create_env
from predicators.envs.pybullet_domino_real import PyBulletDominoRealEnv
from predicators.pybullet_helpers.real_robot_bridge import make_real_robot, \
    observe_scene
from predicators.settings import CFG
from predicators.structs import Object, State
from scripts.cluster_utils import SingleSeedRunConfig, generate_run_configs

# A debug harness that reads env internals to report on them.
# pylint: disable=protected-access


def _load_config(config: str, scene: Optional[str], settle_s: float) -> None:
    """reset_config from the stock launcher, then force the Stage 2 flags.

    The launcher is the single source of truth for the env, geometry and
    scene, exactly as probe_real_scene and replay_plan use it. Only the
    real-robot flags are overridden here, so this tool cannot be pointed
    at hardware by editing a config.
    """
    rc = list(generate_run_configs(config))[0]
    assert isinstance(rc, SingleSeedRunConfig)
    flags = dict(rc.flags)
    flags.update({"env": rc.env, "approach": rc.approach, "seed": rc.seed})
    flags.pop("log", None)
    if scene is not None:
        flags["domino_real_scene"] = scene
    # Stage 2 in four flags: a robot with cameras and no arm.
    flags["real_robot_execute"] = True
    flags["real_robot_dry"] = True  # no arm is built; nothing can move
    flags["real_robot_perception"] = "zed"  # live cameras
    flags["real_robot_human_reset"] = False
    flags["real_robot_settle_s"] = settle_s
    utils.reset_config(flags)


def _xyz(vec: Any) -> str:
    """Fixed-width xyz, so the table columns line up."""
    return "[" + " ".join(f"{float(v):6.3f}" for v in vec) + "]"


def _domino_rows(env: PyBulletDominoRealEnv, predicted: State,
                 perceived: State) -> List[Tuple[Object, int, Any, Any]]:
    """(object, capture_id, predicted xyz, perceived xyz) in scene order.

    Scene order is slot order is ``env._scene_ids`` order, which is the
    mapping under test -- so report it rather than sorting by name.
    """
    comp = env._domino_component
    assert comp is not None, "env has no domino component"
    rows = []
    for slot, capture_id in enumerate(env._scene_ids):
        dom = comp.dominos[slot]
        rows.append(
            (dom, capture_id, np.array([predicted.get(dom, f) for f in "xyz"]),
             np.array([perceived.get(dom, f) for f in "xyz"])))
    return rows


def _report(env: PyBulletDominoRealEnv, predicted: State, perceived: State,
            toppled: Dict[str, bool]) -> Dict[str, float]:
    """Print the per-domino table and return the summary numbers."""
    rows = _domino_rows(env, predicted, perceived)
    print(f"{'object':>10} {'id':>4} {'twin xyz':>24} {'seen xyz':>24} "
          f"{'delta':>8} {'dz':>8}  toppled")
    deltas, dzs = [], []
    for dom, capture_id, twin, seen in rows:
        delta = float(np.linalg.norm(twin - seen))
        dz = float(seen[2] - twin[2])
        deltas.append(delta)
        dzs.append(dz)
        print(f"{dom.name:>10} {capture_id:>4} {_xyz(twin):>24} "
              f"{_xyz(seen):>24} {delta:8.4f} {dz:8.4f}  "
              f"{toppled.get(dom.name, False)}")
    worst = max(deltas) if deltas else float("nan")
    # A constant dz is the table, not the perception: report the spread
    # separately from the offset so the two cannot be confused.
    print(f"\n  max divergence : {worst:.4f} m")
    print(f"  mean dz        : {float(np.mean(dzs)):+.4f} m  "
          f"(spread {float(np.max(dzs) - np.min(dzs)):.4f} m)")
    print("  a large mean dz with a small spread is table height "
          f"(domino_real_table_z={CFG.domino_real_table_z}), not perception")
    return {"worst": worst, "mean_dz": float(np.mean(dzs))}


def main() -> None:
    """Look at the real scene N times and report it against the twin."""
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="predicatorv3/exp_domino_real.yaml")
    ap.add_argument("--scene",
                    default=None,
                    help="override CFG.domino_real_scene (the capture the "
                    "twin is built from)")
    ap.add_argument("--repeat",
                    type=int,
                    default=1,
                    help="captures to take; >1 gives the noise floor")
    ap.add_argument("--settle",
                    type=float,
                    default=0.5,
                    help="dwell before each capture, seconds")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    _load_config(args.config, args.scene, args.settle)
    env = get_or_create_env(CFG.env)
    assert isinstance(env, PyBulletDominoRealEnv), \
        f"check_perception drives the real-scene env; got {CFG.env}"
    # Build the task so the twin holds the captured scene, which is what the
    # perceived scene is being measured against.
    env.get_test_tasks()
    toppled_pred = next(p for p in env.predicates if p.name == "Toppled")

    print(f"# scene   : {CFG.domino_real_scene}")
    print(f"# slots   : {list(enumerate(env._scene_ids))}"
          "  (slot -> capture id)")
    print("# NOTHING MOVES: the arm is not built, only the cameras open\n")

    robot = make_real_robot()  # dry=True + zed, from the flags above
    try:
        worsts = []
        for i in range(args.repeat):
            obs = observe_scene(robot, settle_s=args.settle)
            predicted = env.get_observation()
            assert isinstance(predicted, State)
            perceived = env.state_from_observation(obs, predicted)
            toppled = {
                d.name: bool(toppled_pred.holds(perceived, [d]))
                for d in perceived if d.type.name == "domino"
            }
            print(f"--- capture {i + 1}/{args.repeat} "
                  f"({len(obs.dominoes)} dominoes seen) ---")
            worsts.append(_report(env, predicted, perceived, toppled)["worst"])
            print()
        if args.repeat > 1:
            print(f"# noise floor over {args.repeat} captures: "
                  f"max {max(worsts):.4f} m, "
                  f"mean {float(np.mean(worsts)):.4f} m")
            print("# set real_robot_divergence_atol from this, with headroom")
    finally:
        robot.close()  # release the cameras even if a capture threw


if __name__ == "__main__":
    main()
