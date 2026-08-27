"""Benchmark the async launch/think/gather rollout path (production topology).

Validates ``agent_sdk.async_rollouts`` the way a learn session would use
it: the PARENT holds a live PyBullet env (the session env) across the
forks, LAUNCHES K bridge rollouts, keeps using its own env during a
"thinking" window (the agent editing/fitting between run_python calls),
then GATHERS in a later scope. The sequential baseline runs the same
rollouts inline before the same thinking window.

Modes:
  seq    - K inline rollouts + think window (baseline wall)
  async  - launch K -> think window (with live parent env use) -> gather
  queue  - K > workers, exercising the cap + cooperative pump under load

Usage (COMPUTE NODE, e.g. --cpus-per-task=8):
  python scripts/async_rollout_bench.py --rollouts 6 --workers 6 --think 60
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

# pylint: disable=wrong-import-position
import argparse
import sys
import time
from pathlib import Path
from typing import Any, List

# Repo root on sys.path so `scripts` is importable without PYTHONPATH=.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from predicators import utils
from predicators.agent_sdk.async_rollouts import AsyncRollout, \
    AsyncRolloutRegistry
from predicators.envs import create_new_env
from scripts.parallel_rollout_bench import BRIDGE_FLAGS, one_rollout


def _think(seconds: float, parent_env: Any) -> int:
    """Occupy the parent for ``seconds`` while touching its live env.

    A few real resets prove the parent's PyBullet client stays usable
    while children run; the rest sleeps (the agent's own thinking time
    between tool calls is mostly LLM latency, i.e. an idle process).
    """
    resets = 0
    t_end = time.monotonic() + seconds
    for _ in range(3):
        if time.monotonic() >= t_end:
            break
        parent_env.reset("train", 0)
        resets += 1
    remaining = t_end - time.monotonic()
    if remaining > 0:
        time.sleep(remaining)
    return resets


def _launch_call(reg: AsyncRolloutRegistry, k: int) -> List[AsyncRollout]:
    """Emulates run_python call N: launch and return handles."""
    return [reg.launch(one_rollout, tag=f"bench-{i}") for i in range(k)]


def _gather_call(reg: AsyncRolloutRegistry,
                 handles: List[AsyncRollout]) -> List[dict]:
    """Emulates run_python call N+k: collect via the persistent handles."""
    done, pending = reg.gather(handles, timeout=1800)
    assert not pending, f"{len(pending)} rollouts never finished"
    out = []
    for h in done:
        assert h.ok, f"rollout {h.index} failed: {h.error}"
        out.append(h.result)
    return out


def main() -> None:
    """Run the seq / async / queue benchmark modes and print timings."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts", type=int, default=6)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--think", type=float, default=60.0)
    parser.add_argument("--queue-rollouts", type=int, default=10)
    args = parser.parse_args()
    utils.reset_config(dict(BRIDGE_FLAGS))
    cpus = len(os.sched_getaffinity(0))
    print(
        f"async bench: rollouts={args.rollouts} workers={args.workers} "
        f"think={args.think:.0f}s queue_rollouts={args.queue_rollouts} "
        f"visible CPUs={cpus}",
        flush=True)

    # Parent warm-up: hold the live session env across every fork below
    # and pay one inline rollout so caches are hot.
    parent_env = create_new_env("pybullet_bridge",
                                do_cache=True,
                                use_gui=False)
    warm = one_rollout()
    print(
        f"warm-up: env={warm['t_env']:.1f}s exec={warm['t_exec']:.1f}s "
        f"steps={warm['steps']} ok={warm['ok']}",
        flush=True)
    assert warm["ok"], "warm-up rollout failed; benchmark plan is broken"

    # --- seq baseline: inline rollouts, then the think window.
    t0 = time.monotonic()
    seq_results = [one_rollout() for _ in range(args.rollouts)]
    _think(args.think, parent_env)
    seq_wall = time.monotonic() - t0
    seq_steps = sorted(r["steps"] for r in seq_results)
    print(
        f"seq:   wall={seq_wall:.1f}s "
        f"ok={sum(r['ok'] for r in seq_results)}/{args.rollouts} "
        f"steps={seq_steps}",
        flush=True)

    # --- async: launch in one scope, think, gather in another.
    reg = AsyncRolloutRegistry(max_workers=args.workers)
    try:
        t0 = time.monotonic()
        handles = _launch_call(reg, args.rollouts)
        t_launch = time.monotonic() - t0
        resets = _think(args.think, parent_env)
        t_after_think = time.monotonic() - t0
        async_results = _gather_call(reg, handles)
        async_wall = time.monotonic() - t0
        async_steps = sorted(r["steps"] for r in async_results)
        print(
            f"async: wall={async_wall:.1f}s (launch={t_launch:.2f}s, "
            f"think-end={t_after_think:.1f}s, parent resets during "
            f"think={resets}) ok={len(async_results)}/{args.rollouts} "
            f"steps={async_steps} speedup={seq_wall / async_wall:.2f}x",
            flush=True)
        if async_steps != seq_steps:
            print(
                "WARNING: step counts differ between modes "
                f"(seq={seq_steps} async={async_steps})",
                flush=True)

        # --- queue mode: more jobs than workers, same registry (also
        # exercises fork-after-reaper-thread on a warm registry).
        t0 = time.monotonic()
        q_handles = _launch_call(reg, args.queue_rollouts)
        q_results = _gather_call(reg, q_handles)
        q_wall = time.monotonic() - t0
        print(
            f"queue: wall={q_wall:.1f}s "
            f"ok={len(q_results)}/{args.queue_rollouts} "
            f"(cap {args.workers}, expected ~2 windows)",
            flush=True)
    finally:
        reg.shutdown()
    print("done", flush=True)


if __name__ == "__main__":
    main()
