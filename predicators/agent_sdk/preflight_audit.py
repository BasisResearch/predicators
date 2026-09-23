"""Nonblocking prospective validation, paired with actual execution outcomes.

This pilot never reports a prediction to the agent or changes its
action. Certificates use observed episode prefixes plus model-predicted
suffixes. No unknown result is counted as passed. Closed-loop raw
policies are logged as unavailable rather than claiming their first
action certifies a policy.
"""
from __future__ import annotations

import copy
import hashlib
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Dict

from predicators.agent_sdk.plan_execution import execute_plan_forward
from predicators.agent_sdk.restoration import restoration_report
from predicators.agent_sdk.tools.budget import suspend_budget_watchdog
from predicators.agent_sdk.tools.context import absolute_rollout_seed
from predicators.agent_sdk.tools.verdicts import _EvalStateCollector, \
    evaluate_states_with
from predicators.run.interaction import ExecutePlan, ExecutePolicy, \
    ExecuteSkill, ExecutionProgress, ExecutionRequest, PrimitiveAction, \
    RequestReset
from predicators.settings import CFG
from predicators.structs import Task, step_option_labels


class PreflightAudit:
    """One fresh point rehearsal per request, with bounded diagnostic time."""

    def __init__(self, ctx: Any, session: Any, simulator: str,
                 path: str) -> None:
        self._ctx = ctx
        self._session = session
        self._simulator = Path(simulator)
        self._path = Path(path)
        self._seconds = 0.
        if self._path.exists():
            for line in self._path.read_text(encoding="utf-8").splitlines():
                try:
                    record = json.loads(line)
                except ValueError:
                    continue  # interrupted final line on a preemption
                if (record.get("event") == "prediction"
                        and record.get("level") == session.level_index):
                    self._seconds += record.get("seconds", 0.)

    def _write(self, record: Dict[str, Any]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, default=str) + "\n")

    def before(self, request: ExecutionRequest) -> Dict[str, Any]:
        """Persist the prediction before executing; an error never blocks."""
        row: Dict[str, Any] = {
            "id": uuid.uuid4().hex,
            "event": "prediction",
            "mode": "shadow",
            "request": type(request).__name__,
            "status": "unavailable",
            "evaluation_status": "not_requested",
            "sim_steps": 0,
            "sim_rollouts": 0,
            "timestamp": time.time()
        }
        start = time.monotonic()
        try:
            obs = self._session.observe()
            row.update(level=obs.ledger.level_index,
                       episode_step=obs.ledger.episode_steps,
                       run_step=obs.ledger.run_steps)
            if isinstance(request, RequestReset):
                row.update(status="not_requested", reason="reset")
            elif isinstance(request, ExecutePolicy):
                row["reason"] = "closed-loop raw policy rehearsal unavailable"
            elif not self._simulator.is_file():
                row["reason"] = "no candidate model"
            elif self._seconds >= CFG.continual_validation_audit_seconds:
                row["reason"] = "audit time budget exhausted"
            else:
                row["model_sha256"] = hashlib.sha256(
                    self._simulator.read_bytes()).hexdigest()
                remaining = (CFG.continual_validation_audit_seconds -
                             self._seconds)
                with suspend_budget_watchdog(own_timeout=min(
                        30., remaining)), absolute_rollout_seed(CFG.seed):
                    self._predict(request, row)
        except Exception as err:  # pylint: disable=broad-except
            row.update(status="error", reason=f"{type(err).__name__}: {err}")
        row["seconds"] = time.monotonic() - start
        self._seconds += row["seconds"]
        self._write(row)
        return row

    def _predict(self, request: ExecutionRequest, row: Dict[str, Any]) -> None:
        ctx = self._ctx
        provider = ctx.probe_option_model_provider
        scope = ctx.probe_validation_env_scope
        if (provider is None or scope is None
                or ctx.current_observation_provider is None):
            row["reason"] = "no isolated candidate or current observation"
            return
        current = ctx.current_observation_provider().copy()
        model = provider()
        row["parameter_status"] = ctx.probe_param_status
        row["parameters"] = dict(
            getattr(model.sim_env, "_agent_param_values", {}))
        task = self._session.levels[self._session.level_index].task
        with scope():
            env = model.sim_env
            env._set_state(current.copy())  # pylint: disable=protected-access
            restored = env._get_state()  # pylint: disable=protected-access
            row["restoration"] = restoration_report(current, restored)
            row["sim_rollouts"] = 1
            if isinstance(request, PrimitiveAction):
                after = env.simulate(current, copy.deepcopy(request.action))
                states, labels = [current, after], [None]
                row.update(status="accepted", sim_steps=1)
                row["controller_status"] = "not_applicable"
            else:
                assert isinstance(request, (ExecuteSkill, ExecutePlan))
                skills = [request] if isinstance(
                    request, ExecuteSkill) else list(request.skills)
                # Options carry mutable controller memory: never rehearse
                # the same option instances that real execution will use.
                options = [
                    s.option.parent.ground(list(s.option.objects),
                                           s.option.params.copy())
                    for s in skills
                ]
                row["skills"] = step_option_labels_for_skills(skills)
                collector = _EvalStateCollector(model, current)
                result = execute_plan_forward(Task(current, task.goal),
                                              options,
                                              model,
                                              predicates=ctx.predicates,
                                              on_step=collector.on_step,
                                              stop_on_failure=True)
                failures = [
                    s.failure_reason for s in result.steps
                    if s.failure_reason and s.failure_reason != "0 actions"
                ]
                row.update(
                    status="rejected" if failures else "accepted",
                    controller_status="rejected" if failures else "accepted",
                    failures=failures,
                    sim_steps=result.total_actions,
                    coarse=collector.coarse)
                row["expected_outcome_mismatches"] = [
                    i for i, (s, r) in enumerate(zip(skills, result.steps))
                    if r.post_state is not None and (any(
                        not a.holds(r.post_state) for a in s.expected) or any(
                            a.holds(r.post_state) for a in s.expected_absent))
                ]
                if row["expected_outcome_mismatches"]:
                    row["status"] = "rejected"
                if any(f.startswith("execution error:") for f in failures):
                    row["status"] = "error"
                if collector.coarse:
                    row.update(evaluation_status="unavailable",
                               evaluation_reason="missing per-step states")
                    if row["status"] == "accepted":
                        row["status"] = "unavailable"
                    return
                states, labels = collector.states, collector.labels
            row["predicted_frame"] = observable_motion(states[-1])
            before_motion = observable_motion(current)
            row["lost_grasps"] = [
                name for name, features in before_motion.items()
                if features.get("is_held", 0.) > .5 >=
                row["predicted_frame"].get(name, {}).get("is_held", 0.)
            ]
            # A reconstruction mismatch is uncertainty, not a successful check.
            if row["restoration"]["status"] != "accepted":
                row["status"] = "unavailable"
            episodes = self._session.level_episodes()
            if not episodes or not episodes[-1]["states"]:
                row.update(evaluation_status="unavailable",
                           evaluation_reason="missing observed episode prefix")
                return
            prefix = episodes[-1]
            joined = list(prefix["states"][:-1]) + states
            joined_labels = step_option_labels(prefix["actions"]) + labels
            evaluator = task.evaluator
            if evaluator is None:
                row["evaluation_status"] = "unavailable"
                return
            prefix_length = len(prefix["states"])
            if evaluator.terminated_trajectory(joined[:prefix_length]):
                row.update(
                    status="unavailable",
                    evaluation_status="unavailable",
                    evaluation_reason="observed prefix already looks terminal")
                return
            terminal = terminal_prefix_length(evaluator, joined, prefix_length)
            if terminal is not None:
                joined, joined_labels = joined[:
                                               terminal], joined_labels[:
                                                                        terminal
                                                                        - 1]
                row["terminal_after_sim_steps"] = terminal - prefix_length
                row["predicted_frame"] = observable_motion(joined[-1])
                # Only terminal suffixes need the expensive task certificate.
                # Intermediate manipulation remains a controller prediction.
                try:
                    verdict = evaluate_states_with(evaluator, joined,
                                                   joined_labels, env)
                    row.update(evaluation_status="accepted"
                               if verdict["solved"] else "rejected",
                               evaluation=verdict)
                    if not verdict["solved"]:
                        row["status"] = "rejected"
                except Exception as err:  # pylint: disable=broad-except
                    row.update(
                        status="error",
                        evaluation_status="error",
                        evaluation_reason=f"{type(err).__name__}: {err}")

    def after(self, prediction: Dict[str, Any],
              progress: ExecutionProgress) -> None:
        """Pair real outcomes without feeding privileged diagnostics back."""
        row: Dict[str, Any] = {
            "id":
            prediction["id"],
            "event":
            "execution",
            "timestamp":
            time.time(),
            "steps":
            progress.steps,
            "stop_reason":
            progress.stop_reason,
            "skills": [{
                "status": s.result.status,
                "episode_state": s.result.outcome.episode_state.value,
                "diverged": s.result.diverged
            } for s in progress.skills]
        }
        if progress.step_outcome is not None:
            row["episode_state"] = progress.step_outcome.state.value
        try:
            observation = self._session.observe()
            row["episode_state"] = observation.state.value
            row["observed_frame"] = observable_motion(observation.frame)
        except Exception as err:  # pylint: disable=broad-except
            row["observation_error"] = type(err).__name__
        self._write(row)


def step_option_labels_for_skills(skills: Any) -> Any:
    """Serializable request, with parameters but no mutable controller
    state."""
    return [(s.option.name, [o.name for o in s.option.objects],
             s.option.params.tolist()) for s in skills]


def observable_motion(state: Any) -> Dict[str, Dict[str, float]]:
    """Compact pose/grasp prediction for comparison with noisy observations."""
    return {
        o.name: {
            f: float(state.get(o, f))
            for f in ("x", "y", "z", "roll", "pitch", "yaw", "is_held")
            if f in o.type.feature_names
        }
        for o in state
    }


def terminal_prefix_length(evaluator: Any, states: Any,
                           prefix_length: int) -> Any:
    """Real execution stops at the first terminal step, not a later goal."""
    for end in range(prefix_length + 1, len(states) + 1):
        if evaluator.terminated_trajectory(states[:end]):
            return end
    return None


def audit_safely(audit: Any, method: str, *args: Any) -> Any:
    """Even logging failures must not change the shadow experiment's policy."""
    try:
        return getattr(audit, method)(*args)
    except Exception:  # pylint: disable=broad-except
        logging.exception("[validation audit] %s failed", method)
        return None
