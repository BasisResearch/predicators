"""Replay evidence for agent-owned model repair, without fitting or
trimming."""
from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from predicators.code_sim_learning.rollout_env import RolloutTrajectory, \
    num_rollouts_run
from predicators.code_sim_learning.rollout_objective import \
    compute_rollout_residuals
from predicators.code_sim_learning.trajectory_prep import \
    compute_residual_scaling


def replay_report(
    factory: Any,
    whole: List[RolloutTrajectory],
    selected: List[Tuple[int, RolloutTrajectory]],
    scope: Dict[str, List[str]],
    params: Dict[str, float],
    physical_names: Sequence[str],
    rules: Sequence[Any],
    latent_init: Any,
    status: str,
    budget_check: Optional[Callable[[], None]] = None,
) -> str:
    """Score all selected recordings at one fixed model, retaining failures.

    Scope and scaling use the same available recording pool for every
    candidate, independently of the candidate's declared residual
    features. Selection never imports new tasks or decides whether data
    is held out. The caller and agent retain responsibility for choosing
    a training split.
    """
    started = time.monotonic()
    rollouts_before = num_rollouts_run()
    scaling = compute_residual_scaling(whole, scope)
    fingerprint = hashlib.sha256()
    fingerprint.update(json.dumps(scope, sort_keys=True).encode())
    for states, actions in whole:
        for state in states:
            for obj in sorted(state):
                fingerprint.update(str(obj).encode())
                fingerprint.update(np.asarray(state[obj]).tobytes())
        for action in actions:
            fingerprint.update(np.asarray(action.arr).tobytes())
    lines = [
        "RECORDED-ACTION VALIDATION (no fitting, no segment rejection)",
        f"Model: {status}.",
        f"Recording pool: {fingerprint.hexdigest()[:16]}.",
        f"Scope: {json.dumps(scope, sort_keys=True)}.",
        f"Parameters: {json.dumps(params, sort_keys=True)}.",
        "Full recorded action sequences are replayed from their observed "
        "initial states in fresh simulators; velocities start at rest.",
        "RMS and SSE use the normalized rollout objective, including its "
        "robust and summary terms. Compare candidates on identical "
        "recordings, scope, and settings; these are predictive errors, "
        "not success certificates or posterior probabilities.",
    ]
    total_sse, total_count = 0.0, 0
    failed = False
    for index, trajectory in selected:
        if budget_check is not None:
            budget_check()
        try:
            residuals = compute_rollout_residuals(factory, [trajectory],
                                                  params,
                                                  scope,
                                                  physical_names,
                                                  rules,
                                                  latent_init,
                                                  scaling,
                                                  episode_count=len(selected))
            if not residuals.size or not np.all(np.isfinite(residuals)):
                raise ValueError("no finite residual vector")
            sse = float(residuals @ residuals)
            rms = float(np.sqrt(sse / residuals.size))
            total_sse += sse
            total_count += residuals.size
            lines.append(f"trajectory {index}: {len(trajectory[1])} actions, "
                         f"RMS={rms:.6g}, SSE={sse:.6g}, "
                         f"residuals={residuals.size}")
        except Exception as err:  # pylint: disable=broad-except
            failed = True
            lines.append(f"trajectory {index}: REPLAY FAILED: {err}")
    if failed:
        lines.append("Aggregate unavailable: a replay failed; do not rank "
                     "this candidate using only the surviving recordings.")
    elif total_count:
        lines.append(f"All selected recordings: RMS="
                     f"{np.sqrt(total_sse / total_count):.6g}, "
                     f"SSE={total_sse:.6g}, residuals={total_count}.")
    lines.append(f"Simulator work: {num_rollouts_run() - rollouts_before} "
                 f"rollouts, {time.monotonic() - started:.3f} seconds.")
    lines.append("A selected recording is held out only if you excluded it "
                 "from fitting and model design. Check per-trajectory "
                 "errors before trusting an aggregate.")
    return "\n".join(lines)
