"""Reset-free Gymnasium env over the live ``SimSession`` for RL tools.

* ``reset()`` never touches objects: it moves the arm back to the anchor pose
  (where the arm was when the tool was called) with the normal controller.
* ``step()`` applies one EE-delta action through IK and one ``session.step``
  (one interaction), then extracts particles and evaluates the agent reward.
* Observation: per-object particle clouds (fixed object list and count, zero
  padded, relative to the anchor position) + visibility mask + EE pose +
  gripper.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from scipy.spatial.transform import Rotation

from agent_robot_control.rl.backend import ACTION_MODES, RLRequest
from agent_robot_control.rl.reward_loader import call_reward
from agent_robot_control.sim.ee_control import GRIPPER_CLOSE, GRIPPER_KEEP, \
    GRIPPER_OPEN
from agent_robot_control.sim.perception import flatten_particles


class ParticleEnv(gym.Env):
    """See module docstring."""
    metadata = {"render_modes": []}

    def __init__(self, session, request: RLRequest) -> None:
        super().__init__()
        if request.action_mode not in ACTION_MODES:
            raise ValueError(f"action_mode must be one of {ACTION_MODES}")
        self.session = session
        self.req = request
        self.ctl = session.controller
        self.reward_fn = request.reward_fn
        snap = session.particles(request.points_per_object)
        self.names: List[str] = sorted(request.object_names or snap.names)
        self.P = request.points_per_object
        # Anchor = pose at tool-call time; pseudo-resets return here.
        self.anchor_pos = self.ctl.ee_position().copy()
        self.anchor_quat = self.ctl.ee_quat().copy()
        self.anchor_rot = Rotation.from_quat(self.anchor_quat)
        self.half = float(request.workspace_half_extent)
        self.max_dt = float(request.max_step_translation)
        self.max_dyaw = np.radians(request.max_step_yaw_deg)
        self.yaw = 0.0  # accumulated yaw command relative to anchor
        dim = 3 + (1 if request.action_mode == "xyz_yaw" else 0) + \
            (1 if request.control_gripper else 0)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(dim,), dtype=np.float32)
        n = len(self.names)
        obs_dim = n * self.P * 3 + n * self.P + 3 + 4 + 1
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,),
                                            dtype=np.float32)
        self.t = 0
        self.episode_return = 0.0
        self.episode_max_reward = -np.inf
        self.episode_log: List[Dict[str, Any]] = []
        self.ik_failures = 0
        self.last_raw_reward = 0.0
        self._last_snapshot = snap
        # If the tool was called while holding an object, losing it makes the
        # rest of the call pointless (no resets, the object is out of reach):
        # the episode terminates and the backend aborts (see SB3Backend).
        self.anchor_holding = self.ctl.is_holding()
        self.dropped = False
        self.drop_step: Optional[int] = None
        # Force-limit reflex: after a step whose contact force exceeds the
        # controller's limit, the next step retracts along the last delta
        # (ignoring the policy for that step) and the violating step is
        # penalised. Emulates a compliant controller; keeps grasps intact.
        self.force_violations = 0
        self._retract: Optional[np.ndarray] = None
        self._last_delta = np.zeros(3)
        # Episode recording: when self.recorder is set, every env step appends
        # a captioned frame. The backend turns it on for one episode every
        # request.video_every interactions (see SB3Backend._maybe_record).
        self.recorder = None
        self.episode_index = 0

    # ── Observation / reward ────────────────────────────────────

    def _observe(self) -> Tuple[np.ndarray, float]:
        snap = self.session.particles(self.P)
        self._last_snapshot = snap
        pts, vis = flatten_particles(snap, self.names, self.P,
                                     origin=self.anchor_pos)
        particles = {n: snap.points.get(n, np.zeros((0, 3), np.float32))
                     for n in self.names}
        visible = {n: np.ones(len(particles[n]), dtype=bool) for n in self.names}
        raw = call_reward(self.reward_fn, particles, visible, snap.ee_pos,
                          snap.ee_quat, snap.gripper, timeout_s=5.0)
        obs = np.concatenate([
            pts.reshape(-1), vis.reshape(-1),
            (snap.ee_pos - self.anchor_pos).astype(np.float32),
            snap.ee_quat.astype(np.float32),
            np.array([snap.gripper], dtype=np.float32),
        ]).astype(np.float32)
        return obs, raw

    def _caption(self) -> str:
        """Caption for a recorded frame: where the episode and reward stand."""
        return (f"RL episode {self.episode_index} | step {self.t}/"
                f"{self.req.episode_length} | interaction "
                f"{self.session.interactions} | reward "
                f"{self.last_raw_reward:+.3f} | "
                f"{'GOAL' if self.session.goal_reached_now else 'running'}")

    # ── Gym API ─────────────────────────────────────────────────

    def reset(self, *, seed: Optional[int] = None,
              options: Optional[dict] = None):
        super().reset(seed=seed)
        if self.t > 0 or not self.episode_log:
            # Pseudo-reset: arm back to the anchor pose; objects untouched.
            self.ctl.move_to(self.anchor_pos, self.anchor_quat,
                             gripper=GRIPPER_KEEP, max_steps=60)
        self.yaw = 0.0
        self._retract = None
        self.t = 0
        self.episode_return = 0.0
        self.episode_max_reward = -np.inf
        obs, raw = self._observe()
        self.last_raw_reward = raw
        return obs, {"raw_reward": raw}

    def step(self, action: np.ndarray):
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        cur = self.ctl.ee_position()
        if self._retract is not None:
            delta = self._retract
            self._retract = None
        else:
            delta = a[:3] * self.max_dt
        self._last_delta = delta
        target = cur + delta
        target = np.clip(target, self.anchor_pos - self.half,
                         self.anchor_pos + self.half)
        idx = 3
        quat = self.anchor_quat
        if self.req.action_mode == "xyz_yaw":
            self.yaw = float(np.clip(self.yaw + a[idx] * self.max_dyaw,
                                     -np.pi / 2, np.pi / 2))
            quat = (Rotation.from_euler("z", self.yaw) * self.anchor_rot).as_quat()
            idx += 1
        gripper = GRIPPER_KEEP
        if self.req.control_gripper:
            g = a[idx]
            gripper = GRIPPER_CLOSE if g > 0.5 else GRIPPER_OPEN if g < -0.5 \
                else GRIPPER_KEEP
        finger_target = self.ctl._finger_target_for(gripper)
        joints = self.ctl._ik(target, np.asarray(quat))
        ik_failed = joints is None
        if ik_failed:
            self.ik_failures += 1
            joints = self.ctl.robot.get_joints()
        self.session.step(self.ctl._action_from_joints(joints, finger_target))
        self.t += 1
        if self.recorder is not None:
            self.recorder.add(self.session.render(), self._caption())
        force, _bodies = self.ctl.contact_force()
        force_violation = force > self.ctl.max_contact_force
        if force_violation:
            self.force_violations += 1
            self._retract = -self._last_delta
        obs, raw = self._observe()
        self.last_raw_reward = raw
        lo, hi = self.req.reward_clip
        reward = float(np.clip(raw, lo, hi))
        if ik_failed:
            reward -= 0.01
        if force_violation:
            reward -= 0.1
        self.episode_return += reward
        self.episode_max_reward = max(self.episode_max_reward, raw)
        success = raw >= 1.0
        dropped = self.anchor_holding and not self.ctl.is_holding()
        if dropped and not self.dropped:
            self.dropped = True
            self.drop_step = self.session.interactions
            reward -= 1.0
        terminated = success or dropped
        truncated = self.t >= self.req.episode_length
        if terminated or truncated:
            self.episode_index += 1
            self.episode_log.append({
                "return": self.episode_return,
                "max_reward": float(self.episode_max_reward),
                "success": bool(success),
                "dropped": bool(dropped),
                "length": self.t,
                "interactions": self.session.interactions,
            })
        info = {"raw_reward": raw, "ik_failed": ik_failed,
                "success": bool(success), "dropped": bool(dropped),
                "force_violation": bool(force_violation), "contact_force": force}
        return obs, reward, terminated, truncated, info
