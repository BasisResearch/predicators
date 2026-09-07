"""Benchmark fork-parallel fresh-env rollouts against the sequential baseline
used by the capture-gate margin sweep and ``sim.run(trials=N)``.

Measures the production topology exactly: the PARENT constructs a warm
PyBullet env (the approach's shared session env) and holds it while
forking; each CHILD constructs its own fresh env (the same work
``_fresh_validation_env_scope`` pays sequentially today), executes a
fixed option plan with proven parameters from the live bridge runs, and
reports timings through a queue. Sequential mode runs the identical
rollout in-process for the baseline.

BLAS/OMP threads are pinned to 1 before any heavy import so the
measurement isolates process-level parallelism from library threading.

Usage (COMPUTE NODE, e.g. --cpus-per-task=8):
  python scripts/parallel_rollout_bench.py --rollouts 8 --workers 1,2,4,8
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

# pylint: disable=wrong-import-position
import argparse
import multiprocessing as mp
import time
import traceback
from typing import Any, List

import numpy as np

from predicators import utils
from predicators.envs import create_new_env
from predicators.ground_truth_models import get_gt_options
from predicators.structs import State, _Option

BRIDGE_FLAGS = {
    "env": "pybullet_bridge",
    "approach": "oracle",
    "seed": 0,
    "num_train_tasks": 1,
    "num_test_tasks": 1,
    "skill_phase_use_motion_planning": True,
    "pybullet_ik_validate": False,
    "pybullet_birrt_contact_margin": -0.005,
    "horizon": 3000,
    # The two physics fixes that are now domain defaults ride along so
    # the per-step cost matches the live runs.
    "skill_place_settle_preload_force": 3.0,
    "pybullet_pin_held_weld_assemblies": True,
}

# Proven placements from the live run's episode-1 plans (both arms).
PLAN_STEPS = [
    ("PickBlock", "leg0", [0.0]),
    ("Place", None, [0.6387, 1.30, 0.452, 0.0]),
    ("PickBlock", "leg1", [0.0]),
    ("Place", None, [0.8887, 1.30, 0.452, 0.0]),
    ("PickBlock", "span1", [0.0]),
    ("Place", None, [0.765, 1.16, 0.431, 0.0]),
]


def one_rollout() -> dict:
    """Fresh env construction + one full plan execution (one sweep unit)."""
    t0 = time.monotonic()
    env = create_new_env("pybullet_bridge", do_cache=False, use_gui=False)
    t_env = time.monotonic() - t0
    options = {o.name: o for o in get_gt_options(env.get_name())}
    state = env.reset("train", 0)
    objs = {o.name: o for o in state}
    robot = objs["robot"]
    plan = []
    for name, target, params in PLAN_STEPS:
        ground_objs = [robot] if target is None else [robot, objs[target]]
        plan.append(options[name].ground(ground_objs,
                                         np.array(params, dtype=np.float32)))
    opt_index = {"i": 0}

    def _option_policy(s: State) -> _Option:
        del s
        if opt_index["i"] >= len(plan):
            raise utils.OptionExecutionFailure("Option plan exhausted!",
                                               info={"plan_exhausted": True})
        opt = plan[opt_index["i"]]
        opt_index["i"] += 1
        return opt

    abstract = lambda s: utils.abstract(s, env.predicates)
    policy = utils.option_policy_to_policy(_option_policy,
                                           max_option_steps=400,
                                           abstract_function=abstract)
    t1 = time.monotonic()
    steps, failure = 0, None
    try:
        while True:
            try:
                act = policy(state)
            except utils.OptionExecutionFailure as e:
                if not getattr(e, "info", {}).get("plan_exhausted"):
                    failure = str(e)
                break
            state = env.step(act)
            steps += 1
    finally:
        env.dispose()
    return {
        "t_env": t_env,
        "t_exec": time.monotonic() - t1,
        "steps": steps,
        "ok": failure is None,
        "failure": failure,
    }


def _child(idx: int, q: Any) -> None:
    exit_code = 0
    try:
        r = one_rollout()
        r["idx"] = idx
        q.put(r)  # SimpleQueue.put is synchronous (locked pipe write).
    except BaseException as e:  # pylint: disable=broad-except
        traceback.print_exc()
        exit_code = 1
        try:
            q.put({
                "idx": idx,
                "ok": False,
                "failure": f"child exception: {e!r}",
                "t_env": 0.0,
                "t_exec": 0.0,
                "steps": 0,
            })
        except BaseException:  # pylint: disable=broad-except
            exit_code = 2
    finally:
        # Skip interpreter teardown entirely: PyBullet's atexit handler
        # tries to disconnect the forked COPY of the parent's live
        # client, which lags child exit by seconds (measured in run
        # 21335464: results queued instantly, exits so late the parent's
        # liveness-based window deadlocked). The result is already on
        # the queue; nothing of the child is worth tearing down.
        os._exit(exit_code)  # pylint: disable=protected-access


def run_parallel(num_rollouts: int, workers: int) -> List[dict]:
    """Rolling window of at most ``workers`` forked children.

    Concurrency is bookkept by OUTSTANDING RESULTS, never by process
    liveness: a child that has delivered its result frees its window
    slot even if its exit is lagging (see ``_child``'s teardown note).
    """
    ctx = mp.get_context("fork")
    q = ctx.SimpleQueue()
    procs: dict = {}
    results: List[dict] = []
    next_idx = 0
    outstanding = 0
    deadline = time.monotonic() + 180.0 * num_rollouts
    while len(results) < num_rollouts:
        while next_idx < num_rollouts and outstanding < workers:
            p = ctx.Process(target=_child, args=(next_idx, q))
            p.start()
            procs[next_idx] = p
            outstanding += 1
            next_idx += 1
        while q.empty():
            if time.monotonic() > deadline:
                for i, p in procs.items():
                    print(f"  child {i} pid={p.pid} "
                          f"exitcode={p.exitcode}",
                          flush=True)
                raise RuntimeError(f"benchmark deadline exceeded with "
                                   f"{len(results)}/{num_rollouts} results")
            time.sleep(0.05)
        r = q.get()
        results.append(r)
        outstanding -= 1
        done = procs.pop(r["idx"], None)
        if done is not None:
            done.join(timeout=30)
    for p in procs.values():
        p.join(timeout=30)
    return results


def main() -> None:
    """Run the rollout benchmark across the requested worker counts."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts", type=int, default=8)
    parser.add_argument("--workers", default="1,2,4,8")
    parser.add_argument("--no-parent-env",
                        action="store_true",
                        help="do not hold a live parent PyBullet client "
                        "across the forks (isolates whether the inherited "
                        "client is what kills the children)")
    args = parser.parse_args()
    utils.reset_config(dict(BRIDGE_FLAGS))
    cpus = len(os.sched_getaffinity(0))
    print(f"benchmark: {args.rollouts} rollouts/mode, visible CPUs={cpus}")

    # Parent warm-up: build and hold the "shared session env", and pay
    # one full inline rollout so URDF/disk caches and imports are hot
    # before any timed mode (also verifies the plan succeeds).
    warm_env = (None if args.no_parent_env else create_new_env(
        "pybullet_bridge", do_cache=True, use_gui=False))
    warm = one_rollout()
    print(f"warm-up rollout: env={warm['t_env']:.1f}s "
          f"exec={warm['t_exec']:.1f}s steps={warm['steps']} "
          f"ok={warm['ok']}{'' if warm['ok'] else ' ' + str(warm['failure'])}")
    assert warm["ok"], "warm-up rollout failed; benchmark plan is broken"

    baseline = None
    for w in [int(x) for x in args.workers.split(",")]:
        t0 = time.monotonic()
        if w == 1:
            results = []
            for _ in range(args.rollouts):
                results.append(one_rollout())
        else:
            results = run_parallel(args.rollouts, w)
        wall = time.monotonic() - t0
        n_ok = sum(1 for r in results if r["ok"])
        mean_env = float(np.mean([r["t_env"] for r in results]))
        mean_exec = float(np.mean([r["t_exec"] for r in results]))
        if w == 1:
            baseline = wall
        speedup = f" speedup={baseline / wall:.2f}x" if baseline else ""
        print(f"workers={w}: wall={wall:.1f}s ok={n_ok}/{args.rollouts} "
              f"mean_env={mean_env:.1f}s mean_exec={mean_exec:.1f}s"
              f"{speedup}")
        for r in results:
            if not r["ok"]:
                print(f"  FAILED rollout: {r['failure']}")
    del warm_env
    print("done")


if __name__ == "__main__":
    main()
