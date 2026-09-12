"""Measure portable replay error on scripted five-domain development prefixes.

Run on a compute node. This uses evaluator dynamics to isolate
restoration from learned-program error. It produces no agent scorecards
and makes no solve-rate claim. Longer recorded interactions remain a
separate gate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, cast

import numpy as np

from predicators import utils
from predicators.code_sim_learning.inference_replay import ReplayState, \
    capture_replay_state, replay_candidate
from predicators.code_sim_learning.rollout_env import _pin_all_physical_params
from predicators.envs import create_new_env
from predicators.envs.pybullet_env import PyBulletEnv
from predicators.structs import Action


def _errors(expected: List[ReplayState],
            actual: List[ReplayState]) -> Dict[str, Any]:
    """Return raw per-feature errors, without interpreting them as noise."""
    assert len(expected) == len(actual)
    features: Dict[str, float] = {}
    joint_position_error = 0.0
    joint_velocity_error = 0.0
    linear_velocity_error = 0.0
    angular_velocity_error = 0.0
    memory_matches = True
    attachments_match = True
    for left, right in zip(expected, actual):
        assert set(left.state) == set(right.state)
        for obj in left.state:
            for name in obj.type.feature_names:
                key = f"{obj.type.name}.{name}"
                error = abs(
                    left.state.get(obj, name) - right.state.get(obj, name))
                if not np.isfinite(error):
                    raise ValueError(f"Non-finite replay feature: {key}")
                features[key] = max(features.get(key, 0.0), float(error))
        joint_errors = np.abs(
            np.array(left.robot_joints) - np.array(right.robot_joints))
        if not np.isfinite(joint_errors).all():
            raise ValueError("Non-finite replay joint state")
        joint_position_error = max(joint_position_error,
                                   float(np.max(joint_errors[:, 0])))
        joint_velocity_error = max(joint_velocity_error,
                                   float(np.max(joint_errors[:, 1])))
        left_sim, right_sim = (left.state.simulator_state,
                               right.state.simulator_state)
        assert isinstance(left_sim, dict) and isinstance(right_sim, dict)
        for name, value in left_sim["body_velocities"].items():
            velocity_errors = np.abs(
                np.array(value) - np.array(right_sim["body_velocities"][name]))
            if not np.isfinite(velocity_errors).all():
                raise ValueError(f"Non-finite replay velocity: {name}")
            linear_velocity_error = max(linear_velocity_error,
                                        float(np.max(velocity_errors[0])))
            angular_velocity_error = max(angular_velocity_error,
                                         float(np.max(velocity_errors[1])))
        memory_matches &= left.state.latent == right.state.latent
        attachments_match &= (left_sim.get("command_welds",
                                           []) == right_sim.get(
                                               "command_welds", []))
    return {
        "max_feature_errors": features,
        "max_robot_joint_position_error": joint_position_error,
        "max_robot_joint_velocity_error": joint_velocity_error,
        "max_body_linear_velocity_error": linear_velocity_error,
        "max_body_angular_velocity_error": angular_velocity_error,
        "model_memory_equal": memory_matches,
        "command_attachments_equal": attachments_match
    }


def audit(domain: str, seed: int, steps: int) -> Dict[str, Any]:
    """Compare fresh replays and a resumed prefix with the source world."""
    name = f"pybullet_{domain}"
    utils.reset_config({
        "env": name,
        "seed": seed,
        "num_train_tasks": 1,
        "num_test_tasks": 0,
        "skill_phase_use_motion_planning": False
    })

    def factory() -> PyBulletEnv:
        return cast(PyBulletEnv, create_new_env(name, do_cache=False))

    env = factory()
    try:
        _pin_all_physical_params(env, {})
        env.reset("train", 0)
        _pin_all_physical_params(env, {})
        initial = capture_replay_state(env)
        sim_state = initial.state.simulator_state
        assert isinstance(sim_state, dict)
        hold = np.array(sim_state["joint_positions"], dtype=np.float32)
        extra = env.action_space.shape[0] - len(hold)
        action = Action(
            np.concatenate([hold, np.zeros(extra, dtype=np.float32)]))
        recorded = [initial]
        for _ in range(steps):
            env.step(action)
            recorded.append(capture_replay_state(env))
        actions = [action] * steps
        first = replay_candidate(factory, initial, actions, {})
        repeat = replay_candidate(factory, initial, actions, {})
        prefix = min(3, steps - 1)
        resumed = replay_candidate(factory, recorded[prefix], actions[prefix:],
                                   {})
        return {
            "domain": domain,
            "seed": seed,
            "actions": steps,
            "resume_prefix": prefix,
            "restoration": _errors(recorded[:1], first[:1]),
            "source_replay": _errors(recorded, first),
            "repeatability": _errors(first, repeat),
            "prefix_replay": _errors(recorded[prefix:], resumed)
        }
    finally:
        env.dispose()


def main() -> None:
    """Write a development audit, including explicit setup failures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--domains",
        nargs="+",
        default=["bridge", "fan", "domino", "boil", "balloons"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=30)
    args = parser.parse_args()
    if args.steps < 2:
        parser.error("--steps must be at least 2")
    results = []
    for domain in args.domains:
        try:
            result = audit(domain, args.seed, args.steps)
        except Exception as error:  # pylint: disable=broad-except
            result = {
                "domain": domain,
                "seed": args.seed,
                "setup_or_replay_error": f"{type(error).__name__}: {error}"
            }
        results.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(
            {
                "kind": "mechanical_replay_audit",
                "agent_results": False,
                "dynamics":
                "evaluator program; registry defaults pinned; offline only",
                "results": results,
            },
            indent=2,
            sort_keys=True),
                               encoding="utf-8")
    if any("setup_or_replay_error" in result for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
