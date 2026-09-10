"""Generate one hard (L-shaped, min-block) domino task and describe it.

    uv run python -m agent_robot_control.experiments.domino_turn_probe 0 out.png

Task generation for this variant runs simulated minimum-block searches and
survives roughly 1 turn attempt in 30, so the first call per (config, seed) is
slow; results are cached under ``saved_datasets/domino_min_block_tasks``.
Prints the roles, poses and goal, and renders the initial scene, so the task
can be eyeballed before any agent is pointed at it.
"""
from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

GREEN = (0.56, 0.93, 0.56)
BLUE = (0.60, 0.80, 1.00)


def main() -> None:
    from hydra import compose, initialize_config_dir

    from agent_robot_control.experiments.run_experiment import CONF_DIR
    from agent_robot_control.mcp_server.server import build_session

    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    out = sys.argv[2] if len(sys.argv) > 2 else "turn_scene.png"
    env_name = sys.argv[3] if len(sys.argv) > 3 else "domino_turn"
    # Camera size is excluded from the task cache key, so rendering big does
    # not orphan the cached (expensive) task set.
    cam = sys.argv[4] if len(sys.argv) > 4 else None
    # "top" swaps in a near-overhead camera: the default three-quarter view
    # foreshortens the L, which is the whole point of this variant.
    view = sys.argv[5] if len(sys.argv) > 5 else "default"
    with initialize_config_dir(config_dir=str(CONF_DIR), version_base=None):
        cfg = compose(config_name="config",
                      overrides=[f"env={env_name}", "condition=move_to",
                                 f"seed={seed}"])
    if cam:
        w, h = (int(v) for v in cam.lower().split("x"))
        cfg.server.camera_width, cfg.server.camera_height = w, h
    cfg.run_dir = None
    if view == "top":
        cfg.env.class_overrides = {"_camera_pitch": -78, "_camera_yaw": -90,
                                   "_camera_distance": 1.05,
                                   "_camera_target": [0.90, 1.30, 0.42]}
    t0 = time.time()
    session = build_session(cfg)
    print(f"generation + build took {time.time() - t0:.0f}s", flush=True)
    env = session.env
    state = env._current_observation
    comp = env._components[0]
    print("goal:", sorted(str(a) for a in session.task.goal))
    print("evaluator:", type(getattr(session.task, "evaluator", None)).__name__)
    task = env.get_train_tasks()[0]
    print("offline_task_metrics:", getattr(task, "offline_task_metrics", None))
    for d in state.get_objects(comp._domino_type):
        rgb = tuple(round(float(state.get(d, c)), 2) for c in "rgb")
        role = ("green/start" if np.allclose(rgb, GREEN, atol=.05) else
                "blue/movable" if np.allclose(rgb, BLUE, atol=.05) else
                "target-or-scenery")
        print("  %-10s %-18s xy=(%.3f, %.3f) z=%.3f yaw=%+.3f rgb=%s" %
              (d.name, role, state.get(d, "x"), state.get(d, "y"),
               state.get(d, "z"), state.get(d, "yaw"), rgb))
    print("particle objects:", sorted(session.particles().points.keys()))
    import imageio.v2 as imageio
    imageio.imwrite(out, env.render()[0])
    print("wrote", out)


if __name__ == "__main__":
    main()
