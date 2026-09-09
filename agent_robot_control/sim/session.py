"""``SimSession``: the one live simulation behind all MCP tools.

Owns the predicators ``PyBulletEnv``, the interaction budget, goal tracking,
the event log and transition log, and image snapshots. Every physics step in
the whole system goes through :meth:`SimSession.step`, so interaction
accounting is exact and success is detected at the first step where the
env's goal predicate holds.

The env is never reset after construction (reset-free operation). Objects
stay where the robot leaves them; only the env's own dynamics (belt loop,
donut recycling, plug respawn) move things on their own.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import imageio.v2 as imageio
import numpy as np

from predicators import utils
from predicators.settings import CFG
from predicators.structs import Action, State

from agent_robot_control.sim.budget import BudgetExhausted, InteractionBudget
from agent_robot_control.sim.ee_control import EEController
from agent_robot_control.sim.perception import ParticleSnapshot, \
    extract_named_particles, render_rgb

ENV_CLASSES = {
    "pybullet_airport": "predicators.envs.pybullet_airport.PyBulletAirportEnv",
    "pybullet_donut": "predicators.envs.pybullet_donut.PyBulletDonutEnv",
    "pybullet_plug_outlet":
    "predicators.envs.pybullet_plug_outlet.PyBulletPlugOutletEnv",
    # From the shared master branch: rearrange blue dominoes so that pushing
    # the green one topples the purple one.
    "pybullet_domino":
    "predicators.envs.pybullet_domino.env.PyBulletDominoEnv",
}


def _import_class(path: str):
    module, name = path.rsplit(".", 1)
    mod = __import__(module, fromlist=[name])
    return getattr(mod, name)


@dataclass
class SessionConfig:
    """Everything ``SimSession`` needs; mirrors the Hydra ``env``/``server``
    groups so the MCP server can build it from a resolved config."""
    env_name: str = "pybullet_donut"
    task_idx: int = 0
    seed: int = 0
    interaction_cap: int = 100_000
    camera_width: int = 640
    camera_height: int = 360
    particles_per_object: int = 64
    log_transitions: bool = True
    transition_shard_size: int = 1000
    # Transition logging renders a frame per step; particles for move_to
    # steps are only needed for the world-model dataset, so this can be
    # turned off for speed.
    transition_particles_per_object: int = 32
    use_urdf_torque_limits: bool = True
    torque_limit_scale: float = 3.0
    max_contact_force: float = 80.0
    # Class-attribute overrides applied to the env class before construction.
    env_overrides: Dict[str, Any] = field(default_factory=dict)
    # Extra predicators CFG flags. Some shared domains need these to select the
    # task variant at all (the domino domain defaults to a finished state with
    # glued dominoes and no domino targets, which is not the task we want).
    cfg_overrides: Dict[str, Any] = field(default_factory=dict)
    # Keep a copy of every ``State`` the env passes through. The domino
    # domain's legitimacy certificate is a function of the whole state
    # sequence, not of the final state, so scoring that domain needs this.
    record_states: bool = False
    max_recorded_states: int = 400_000
    run_dir: Optional[str] = None


class SimSession:
    """One live simulation plus bookkeeping."""

    def __init__(self, cfg: SessionConfig) -> None:
        self.cfg = cfg
        self.run_dir = Path(cfg.run_dir) if cfg.run_dir else None
        self.workspace = self.run_dir / "workspace" if self.run_dir else None
        if self.workspace is not None:
            self.workspace.mkdir(parents=True, exist_ok=True)
        config = {
            "env": cfg.env_name,
            "seed": cfg.seed,
            "num_train_tasks": max(5, cfg.task_idx + 1),
            "num_test_tasks": 1,
            "pybullet_camera_width": cfg.camera_width,
            "pybullet_camera_height": cfg.camera_height,
            "pybullet_control_mode": "position",
        }
        config.update(cfg.cfg_overrides)
        utils.reset_config(config)
        env_cls = _import_class(ENV_CLASSES[cfg.env_name])
        for key, value in cfg.env_overrides.items():
            setattr(env_cls, key, value)
        self.env = env_cls(use_gui=False)
        if cfg.use_urdf_torque_limits:
            from predicators.envs.pybullet_plug_outlet import \
                apply_urdf_torque_limits
            apply_urdf_torque_limits(self.env._pybullet_robot,
                                     cfg.torque_limit_scale)
        self.env.reset("train", cfg.task_idx)
        self.task = self.env.get_train_tasks()[cfg.task_idx]
        self.budget = InteractionBudget(cfg.interaction_cap)
        self.controller = EEController(self.env, step_fn=self.step,
                                       max_contact_force=cfg.max_contact_force)
        self.rng = np.random.default_rng(cfg.seed)
        self.goal_first_reached_at: Optional[int] = None
        self.goal_reached_now = False
        self._image_counter = 0
        self._particle_counter = 0
        self._rl_counter = 0
        self._events_path = self.run_dir / "events.jsonl" if self.run_dir else None
        self._transitions: List[Dict[str, np.ndarray]] = []
        self._transition_shard = 0
        self._pending_prev: Optional[ParticleSnapshot] = None
        self.state_history: List[State] = []
        if cfg.record_states:
            self.state_history.append(self.env._current_observation.copy())
        self.record_event("session_start",
                          env=cfg.env_name,
                          task_idx=cfg.task_idx,
                          seed=cfg.seed,
                          goal=[str(a) for a in self.task.goal],
                          interaction_cap=cfg.interaction_cap)

    # ── Core step ───────────────────────────────────────────────

    def step(self, action: Action) -> None:
        """Advance the sim by one action. Counts one interaction, logs the
        transition, and checks the goal. Raises ``BudgetExhausted``."""
        prev = None
        if self.cfg.log_transitions:
            prev = self._pending_prev if self._pending_prev is not None else \
                self._snapshot_for_transition()
        self.budget.consume(1)
        self.env.step(action)
        if self.cfg.log_transitions and prev is not None:
            nxt = self._snapshot_for_transition()
            self._append_transition(prev, action, nxt)
            self._pending_prev = nxt
        if self.cfg.record_states and \
                len(self.state_history) < self.cfg.max_recorded_states:
            self.state_history.append(self.env._current_observation.copy())
        reached = bool(self.env.goal_reached())
        if reached and self.goal_first_reached_at is None:
            self.goal_first_reached_at = self.budget.used
            self.record_event("goal_reached", interactions=self.budget.used)
        self.goal_reached_now = reached

    def invalidate_transition_cache(self) -> None:
        """Call after anything moves bodies outside ``step`` (never expected
        in normal operation; kept for tests)."""
        self._pending_prev = None

    # ── Queries ─────────────────────────────────────────────────

    @property
    def interactions(self) -> int:
        """Interactions used so far."""
        return self.budget.used

    def ee_pose(self):
        """(position (3,), quaternion xyzw (4,))."""
        return self.controller.ee_position(), self.controller.ee_quat()

    def gripper(self) -> float:
        """Finger joint value."""
        return self.controller.gripper_value()

    def robot_state_text(self) -> str:
        """One line describing EE pose and gripper for tool results."""
        pos, _ = self.ee_pose()
        r, pt, yw = self.controller.rpy_relative_to_home_deg()
        g = self.gripper()
        holding = "holding an object" if self.controller.is_holding() \
            else "not holding anything"
        return (f"EE position ({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}) m; "
                f"orientation rpy ({r:.1f}, {pt:.1f}, {yw:.1f}) deg relative "
                f"to home; gripper joint {g:.3f} (0.04 = fully open, about 8 cm between fingertips; 0.01 = closed, about 2 cm), {holding}.")

    def particles(self, max_points_per_object: Optional[int] = None,
                  keep_rgb: bool = False) -> ParticleSnapshot:
        """Named particles for the current frame (no interaction cost)."""
        k = max_points_per_object or self.cfg.particles_per_object
        return extract_named_particles(self.env,
                                       max_points_per_object=k,
                                       interaction_count=self.budget.used,
                                       keep_rgb=keep_rgb,
                                       rng=self.rng)

    def render(self) -> np.ndarray:
        """Camera image (H, W, 3) uint8."""
        return render_rgb(self.env)

    def snapshot_image(self, tag: str = "obs") -> Optional[Path]:
        """Save the current camera image into the workspace."""
        if self.workspace is None:
            return None
        path = self.workspace / f"{tag}_{self._image_counter:04d}.png"
        self._image_counter += 1
        imageio.imwrite(path, self.render())
        return path

    def save_particles(self, max_points_per_object: Optional[int] = None):
        """Extract particles and write ``particles_NNNN.{npz,json}`` into the
        workspace. Returns ``(snapshot, summary, npz_path, json_path)``."""
        snap = self.particles(max_points_per_object)
        if self.workspace is None:
            return snap, snap.summary(), None, None
        stem = self.workspace / f"particles_{self._particle_counter:04d}"
        self._particle_counter += 1
        npz, js = stem.with_suffix(".npz"), stem.with_suffix(".json")
        summary = snap.save(npz, js)
        return snap, summary, npz, js

    def new_rl_dir(self) -> Optional[Path]:
        """Fresh ``rl_NNNN/`` directory inside the run directory."""
        if self.run_dir is None:
            return None
        d = self.run_dir / f"rl_{self._rl_counter:04d}"
        self._rl_counter += 1
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ── Logging ─────────────────────────────────────────────────

    def record_event(self, kind: str, **payload: Any) -> None:
        """Append one JSON line to ``events.jsonl``."""
        if self._events_path is None:
            return
        rec = {"t": time.time(), "kind": kind, "interactions": self.budget.used,
               "goal": self.goal_reached_now, **payload}
        with self._events_path.open("a") as f:
            f.write(json.dumps(rec, default=_json_default) + "\n")

    def _snapshot_for_transition(self) -> ParticleSnapshot:
        return extract_named_particles(
            self.env,
            max_points_per_object=self.cfg.transition_particles_per_object,
            interaction_count=self.budget.used,
            rng=self.rng)

    def _append_transition(self, prev: ParticleSnapshot, action: Action,
                           nxt: ParticleSnapshot) -> None:
        names = sorted(set(prev.names) | set(nxt.names))
        k = self.cfg.transition_particles_per_object
        from agent_robot_control.sim.perception import flatten_particles
        p0, v0 = flatten_particles(prev, names, k)
        p1, v1 = flatten_particles(nxt, names, k)
        self._transitions.append({
            "points_t": p0, "visible_t": v0, "points_t1": p1, "visible_t1": v1,
            "ee_t": np.concatenate([prev.ee_pos, prev.ee_quat, [prev.gripper]]).astype(np.float32),
            "ee_t1": np.concatenate([nxt.ee_pos, nxt.ee_quat, [nxt.gripper]]).astype(np.float32),
            "joint_action": np.asarray(action.arr, dtype=np.float32),
            "names": np.array(names),
            "interaction": np.int64(self.budget.used),
        })
        if len(self._transitions) >= self.cfg.transition_shard_size:
            self.flush_transitions()

    def flush_transitions(self) -> Optional[Path]:
        """Write buffered transitions to ``transitions/shard_NNNN.npz``."""
        if not self._transitions or self.run_dir is None:
            self._transitions = []
            return None
        d = self.run_dir / "transitions"
        d.mkdir(exist_ok=True)
        path = d / f"shard_{self._transition_shard:04d}.npz"
        self._transition_shard += 1
        keys = [k for k in self._transitions[0] if k != "names"]
        # Object sets can differ between transitions (recycled donuts);
        # store per-transition name lists as an object array.
        arrays = {k: np.stack([t[k] for t in self._transitions]) for k in keys
                  if all(t[k].shape == self._transitions[0][k].shape for t in self._transitions)}
        arrays["names"] = np.array([t["names"] for t in self._transitions], dtype=object)
        np.savez(path, **arrays)
        self._transitions = []
        return path

    def results(self) -> Dict[str, Any]:
        """Final summary for ``results.json``."""
        return {
            "env": self.cfg.env_name,
            "task_idx": self.cfg.task_idx,
            "seed": self.cfg.seed,
            "interaction_cap": self.cfg.interaction_cap,
            "interactions_used": self.budget.used,
            "goal_reached_ever": self.goal_first_reached_at is not None,
            "first_success_interaction": self.goal_first_reached_at,
            "goal_reached_at_end": bool(self.env.goal_reached()),
            "interventions": int(getattr(self.env, "num_interventions", 0)),
            **self._evaluator_results(),
        }

    def _evaluator_results(self) -> Dict[str, Any]:
        """The domain's own end-of-episode judgement, where it ships one.

        Shared domains (the domino family) attach a ``TaskEvaluator`` whose
        verdict is a function of the whole state sequence, not of the final
        state: the goal atom can hold because the robot shoved the target over
        itself, which their certificate rejects. Our headline metric stays
        ``goal_reached``, so this is recorded alongside it rather than
        replacing it. Never shown to the agent.
        """
        evaluator = getattr(self.task, "evaluator", None)
        if evaluator is None or not self.state_history:
            return {}
        states = self.state_history
        out: Dict[str, Any] = {
            "evaluator": type(evaluator).__name__,
            "states_recorded": len(states),
            "states_truncated":
                len(states) >= self.cfg.max_recorded_states,
        }
        try:
            # pylint: disable-next=protected-access
            ok, reason = evaluator._certify(states, None, sim_env=self.env)
            out["certified"] = bool(ok)
            out["certificate_reason"] = reason[:600]
            out["evaluator_reward"] = float(
                evaluator.reward(states, None, sim_env=self.env))
            out["evaluator_solved"] = bool(
                evaluator.solved(states, None, sim_env=self.env))
            out["evaluator_offline_metrics"] = {
                k: float(v)
                for k, v in evaluator.offline_metrics(states, None).items()}
        except Exception as exc:  # pragma: no cover - never fail a run on this
            out["evaluator_error"] = f"{type(exc).__name__}: {exc}"
        return out

    def write_results(self) -> None:
        """Persist ``results.json`` now (called after every tool call so a
        harness that kills the server still leaves a valid summary)."""
        if self.run_dir is not None:
            tmp = self.run_dir / "results.json.tmp"
            tmp.write_text(json.dumps(self.results(), indent=2))
            tmp.replace(self.run_dir / "results.json")

    def close(self) -> None:
        """Flush logs and write ``results.json``."""
        self.flush_transitions()
        self.write_results()
        self.record_event("session_end", **self.results())


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


__all__ = ["SimSession", "SessionConfig", "BudgetExhausted", "ENV_CLASSES"]
