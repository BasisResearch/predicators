"""Audit executable balloons solutions and decoys on a compute node.

Checks the reference in every immediate release order, reports other
witnessed winners, and requires a witnessed losing release sequence. Use
--require-jam for the stricter contact challenge; a timeout is never
counted as a jam. These checks do not establish an MB/MF performance
gap.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from itertools import permutations
from pathlib import Path

from predicators import utils
from predicators.envs import create_new_env
from predicators.envs.pybullet_balloons import PyBulletBalloonsEnv
from predicators.ground_truth_models.balloons.oracle import solve_level


def main() -> None:
    """Generate tasks and report witnessed outcomes, including alternatives."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-test-tasks", type=int, default=5)
    parser.add_argument("--require-jam", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    utils.reset_config({
        "env": "pybullet_balloons",
        "seed": args.seed,
        "num_train_tasks": 2,
        "num_test_tasks": args.num_test_tasks,
        "balloons_require_jam_decoy": args.require_jam,
        "sesame_task_planning_heuristic": "lmcut",
    })
    env = create_new_env("pybullet_balloons", do_cache=True, use_gui=False)
    assert isinstance(env, PyBulletBalloonsEnv)
    rows = []
    try:
        for index, task in enumerate(env.get_test_tasks()):
            candidates = env.candidate_outcomes(task.init)
            reference = env.solution_subset(task.init)
            assert reference is not None
            assert all(result.won for result in candidates[reference])
            decoys = [
                result for results in candidates.values() for result in results
                if result.jammed or (result.burst and not args.require_jam)
            ]
            assert decoys, "No witnessed losing sequence"
            assert solve_level(env, task.init) is not None
            row = {
                "task":
                index,
                "reference_subset":
                reference,
                "witnessed_winning_subsets":
                sum(
                    any(result.won for result in results)
                    for results in candidates.values()),
                "outcomes": [
                    dict(subset=subset,
                         order=order,
                         **dataclasses.asdict(result))
                    for subset, results in candidates.items()
                    for order, result in zip(permutations(subset), results)
                ],
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
    finally:
        env.dispose()
    report = {
        "seed": args.seed,
        "task_generation_version": 2,
        "require_jam": args.require_jam,
        "passed": True,
        "tasks": rows
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n",
                               encoding="utf-8")
    print(f"PASS: {len(rows)} tasks; no unique-subset or MB/MF-gap claim")


if __name__ == "__main__":
    main()
