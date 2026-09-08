"""End-effector control on top of a predicators ``PyBulletEnv``.

The env's native action is a vector of absolute joint targets. This module
turns "move the end effector to this pose, with the gripper open/closed" into
a sequence of such actions: straight-line interpolation of the EE position
(with quaternion slerp for orientation), inverse kinematics for each
waypoint, and one ``env.step`` per waypoint. Nothing is teleported: IK is run
in validate mode, which restores the robot's joint state before the physics
step, so the arm always moves under position control like a real robot.

Every physics step goes through a caller-supplied ``step_fn`` so that a
``SimSession`` can count interactions and log transitions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence, Tuple

import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation, Slerp

from predicators.pybullet_helpers.geometry import Pose
from predicators.pybullet_helpers.joint import get_joint_infos, get_joints
from predicators.pybullet_helpers.link import get_link_pose
from predicators.structs import Action

StepFn = Callable[[Action], None]

GRIPPER_OPEN = "open"
GRIPPER_CLOSE = "close"
GRIPPER_KEEP = "keep"
GRIPPER_COMMANDS = (GRIPPER_OPEN, GRIPPER_CLOSE, GRIPPER_KEEP)


@dataclass
class MoveResult:
    """Outcome of one ``move_to`` call."""
    reached: bool
    ik_failed: bool
    stalled: bool
    steps: int
    final_position: Tuple[float, float, float]
    final_rpy_deg: Tuple[float, float, float]
    distance_to_target: float
    gripper: float
    message: str = ""
    extra: dict = field(default_factory=dict)

    def summary(self) -> str:
        """One-paragraph human-readable summary."""
        status = "reached target" if self.reached else "did NOT reach target"
        contacts = self.extra.get("contacts") or []
        if self.ik_failed:
            if contacts:
                status += (" (could not get closer: the robot is pressing against "
                           + ", ".join(contacts) + ")")
            else:
                status += " (inverse kinematics failed: target likely out of reach)"
        if self.stalled:
            if self.extra.get("force_limited"):
                status += " (stopped by the contact-force limit)"
            elif contacts:
                status += " (arm stalled: blocked by contact with " + ", ".join(contacts) + ")"
            else:
                status += " (arm stalled short of the target without contact: probably a joint-limit / reach problem; try a nearby target)"
        if contacts and not self.ik_failed and not self.stalled:
            status += " (in contact with " + ", ".join(contacts) + ")"
        x, y, z = self.final_position
        r, pt, yw = self.final_rpy_deg
        return (f"{status}; {self.steps} env steps used; final EE position "
                f"({x:.4f}, {y:.4f}, {z:.4f}) m, orientation rpy "
                f"({r:.1f}, {pt:.1f}, {yw:.1f}) deg relative to home, "
                f"{self.distance_to_target*100:.2f} cm from target; gripper "
                f"joint value {self.gripper:.3f} (0.04 = fully open, about 8 cm between fingertips; 0.01 = closed, about 2 cm). "
                f"{self.message}").strip()


def _quat_xyzw_to_rot(q: Sequence[float]) -> Rotation:
    return Rotation.from_quat(np.asarray(q, dtype=float))


class EEController:
    """Cartesian end-effector controller for a PyBullet env."""

    def __init__(self,
                 env,
                 step_fn: Optional[StepFn] = None,
                 step_size: float = 0.02,
                 max_yaw_step_deg: float = 10.0,
                 pos_tol: float = 0.005,
                 orn_tol_deg: float = 5.0,
                 stall_steps: int = 15,
                 stall_eps: float = 5e-4,
                 finger_settle_steps: int = 8,
                 max_contact_force: float = 80.0) -> None:
        self.env = env
        self.robot = env._pybullet_robot
        self.client = env._physics_client_id
        self.step_fn: StepFn = step_fn if step_fn is not None else \
            (lambda a: env.step(a))
        self.step_size = step_size
        self.max_yaw_step = np.radians(max_yaw_step_deg)
        self.pos_tol = pos_tol
        self.orn_tol = np.radians(orn_tol_deg)
        self.stall_steps = stall_steps
        self.stall_eps = stall_eps
        self.finger_settle_steps = finger_settle_steps
        # Emulated force limit (a real arm runs a compliant, force-limited
        # controller; PyBullet's position servo would press with hundreds of
        # newtons and pop objects out of the grasp). Contacts between the
        # fingers and the held object are excluded (that is the grip).
        self.max_contact_force = max_contact_force
        self.home_rot = _quat_xyzw_to_rot(env.get_robot_ee_home_orn())
        # Last commanded finger target; "keep" re-issues it.
        cur = self.gripper_value()
        mid = 0.5 * (self.robot.open_fingers + self.robot.closed_fingers)
        self.finger_target = self.robot.open_fingers if cur > mid \
            else self.robot.closed_fingers

    # ── State queries ───────────────────────────────────────────

    def ee_position(self) -> np.ndarray:
        """Current EE position (3,)."""
        return np.asarray(self.robot.get_state()[:3], dtype=float)

    def ee_quat(self) -> np.ndarray:
        """Current EE orientation as xyzw quaternion (4,)."""
        return np.asarray(self.robot.get_state()[3:7], dtype=float)

    def gripper_value(self) -> float:
        """Current finger joint value (0.04 open ... 0.01 closed for Fetch)."""
        return float(self.robot.get_state()[7])

    def is_holding(self) -> bool:
        """True if the env has a grasp constraint active."""
        return self.env._held_obj_id is not None

    def contact_force(self) -> Tuple[float, list]:
        """Largest total normal force (N) the robot or its held object exerts
        on any other body, and the names of the bodies involved."""
        from agent_robot_control.sim.perception import build_body_name_map
        names = build_body_name_map(self.env)
        held = self.env._held_obj_id
        robot_id = self.robot.robot_id
        totals: dict = {}
        for b in range(p.getNumBodies(physicsClientId=self.client)):
            bid = p.getBodyUniqueId(b, physicsClientId=self.client)
            if bid == robot_id:
                continue
            name = names.get(int(bid))
            if name is None:
                continue  # floor / walls
            if bid != held:
                f = sum(c[9] for c in p.getContactPoints(robot_id, bid,
                                                          physicsClientId=self.client))
                if f > 0:
                    totals[name] = totals.get(name, 0.0) + f
            if held is not None and bid != held:
                f = sum(c[9] for c in p.getContactPoints(held, bid,
                                                          physicsClientId=self.client))
                if f > 0:
                    totals[name] = totals.get(name, 0.0) + f
        if not totals:
            return 0.0, []
        peak = max(totals.values())
        return float(peak), sorted(n for n, f in totals.items() if f >= 0.5 * peak)

    def contact_names(self) -> list:
        """Names of non-floor bodies the robot is currently touching."""
        from agent_robot_control.sim.perception import build_body_name_map
        names = build_body_name_map(self.env)
        out = []
        for b in range(p.getNumBodies(physicsClientId=self.client)):
            bid = p.getBodyUniqueId(b, physicsClientId=self.client)
            if bid == self.robot.robot_id:
                continue
            if not p.getContactPoints(self.robot.robot_id, bid,
                                      physicsClientId=self.client):
                continue
            name = names.get(int(bid))
            if name is None:
                continue  # floor / walls / unlabeled
            out.append(name)
        return sorted(set(out))

    def rpy_relative_to_home_deg(
            self, quat: Optional[np.ndarray] = None) -> Tuple[float, float, float]:
        """Orientation as roll/pitch/yaw (deg) applied on top of the home
        orientation, i.e. the inverse of :meth:`quat_from_rpy_deg`."""
        rot = _quat_xyzw_to_rot(self.ee_quat() if quat is None else quat)
        rel = rot * self.home_rot.inv()
        r, pt, yw = rel.as_euler("xyz", degrees=True)
        return float(r), float(pt), float(yw)

    def quat_from_rpy_deg(self, roll: float = 0.0, pitch: float = 0.0,
                          yaw: float = 0.0) -> np.ndarray:
        """Target orientation = extrinsic xyz rotation (deg) applied to the
        home (gripper-down) orientation. Yaw rotates about the world z axis."""
        rel = Rotation.from_euler("xyz", [roll, pitch, yaw], degrees=True)
        return (rel * self.home_rot).as_quat()

    # ── Actions ─────────────────────────────────────────────────

    def _finger_target_for(self, gripper: str) -> float:
        if gripper == GRIPPER_OPEN:
            self.finger_target = float(self.robot.open_fingers)
        elif gripper == GRIPPER_CLOSE:
            self.finger_target = float(self.robot.closed_fingers)
        elif gripper != GRIPPER_KEEP:
            raise ValueError(f"gripper must be one of {GRIPPER_COMMANDS}, "
                             f"got {gripper!r}")
        return self.finger_target

    def _action_from_joints(self, joints: Sequence[float],
                            finger_target: float) -> Action:
        arr = np.array(joints, dtype=np.float32)
        # A "closed" command tracks slightly below the current finger position
        # (never above the nominal closed value): the env releases a grasp when
        # target > current + 1e-4, so once a held object shifts and the fingers
        # reach the nominal closed value, commanding exactly that value reads
        # as an "open" and drops the object (DEBUG_LOG 12). Commanding fully
        # closed instead squeezes so hard that the block pivots in the grasp.
        closing = finger_target <= float(self.robot.closed_fingers) + 1e-6
        if closing:
            # May go slightly negative: the joint itself reads -1e-4 when fully
            # closed and a target clipped at 0.0 would count as "opening".
            finger_target = max(-0.01, min(float(self.robot.closed_fingers),
                                           self.gripper_value() - 0.002))
        arr[self.robot.left_finger_joint_idx] = finger_target
        arr[self.robot.right_finger_joint_idx] = finger_target
        low = self.robot.action_space.low.copy()
        if closing:
            low[self.robot.left_finger_joint_idx] = -0.01
            low[self.robot.right_finger_joint_idx] = -0.01
        arr = np.clip(arr, low, self.robot.action_space.high)
        return Action(arr)

    def hold_action(self, gripper: str = GRIPPER_KEEP) -> Action:
        """Action that holds the current arm configuration."""
        return self._action_from_joints(self.robot.get_joints(),
                                        self._finger_target_for(gripper))

    # Joint-limit-aware IK. PyBullet's plain IK ignores joint limits, so its
    # solutions can put e.g. the wrist-flex joint past its range; the action
    # clip then clamps it and the EE lands centimetres off (DEBUG_LOG 9).
    # Null-space IK with limits + rest poses keeps solutions feasible. The
    # robot's joint state is restored afterwards (nothing is teleported).
    def _ik_setup(self) -> None:
        robot, client = self.robot, self.client
        all_joints = get_joints(robot.robot_id, physics_client_id=client)
        infos = get_joint_infos(robot.robot_id, all_joints,
                                physics_client_id=client)
        self._free = [ji.jointIndex for ji in infos if ji.qIndex > -1]
        lo, hi = [], []
        for ji in infos:
            if ji.qIndex <= -1:
                continue
            if ji.jointLowerLimit < ji.jointUpperLimit:
                lo.append(float(ji.jointLowerLimit))
                hi.append(float(ji.jointUpperLimit))
            else:  # continuous joint
                lo.append(-2 * np.pi)
                hi.append(2 * np.pi)
        self._lo, self._hi = np.array(lo), np.array(hi)
        self._ranges = (self._hi - self._lo).tolist()
        self._arm_pos_in_free = [self._free.index(j) for j in robot.arm_joints]

    def _ik(self, position: np.ndarray, quat: np.ndarray,
            tol: float = 1e-3, max_rounds: int = 20) -> Optional[list]:
        """Joint targets reaching (position, quat), or None.

        Order: plain PyBullet IK (same solver the rest of predicators uses;
        gives smooth, predictable paths) accepted only if the limit-clipped
        solution still reaches the target; otherwise null-space IK with joint
        limits (handles targets where plain IK wanders past a limit)."""
        if not hasattr(self, "_free"):
            self._ik_setup()
        solution = self._plain_ik_within_limits(position, quat, tol)
        if solution is None:
            solution = self._nullspace_ik(position, quat, tol, max_rounds)
        if solution is None:
            # Best effort at the workspace edge: the clipped plain-IK solution
            # even if it does not reach exactly (move_to reports the miss).
            solution = self._plain_ik_within_limits(position, quat, tol=0.03)
        if solution is None:
            return None
        return [float(solution[i]) for i in self._arm_pos_in_free]

    def _nullspace_ik(self, position: np.ndarray, quat: np.ndarray,
                      tol: float, max_rounds: int) -> Optional[np.ndarray]:
        robot, client = self.robot, self.client
        init = p.getJointStates(robot.robot_id, self._free, physicsClientId=client)
        rest = [st[0] for st in init]
        try:
            for _ in range(max_rounds):
                vals = p.calculateInverseKinematics(
                    robot.robot_id, robot.end_effector_id, list(position),
                    targetOrientation=list(quat),
                    lowerLimits=self._lo.tolist(), upperLimits=self._hi.tolist(),
                    jointRanges=self._ranges, restPoses=rest,
                    maxNumIterations=200, residualThreshold=1e-5,
                    physicsClientId=client)
                vals = np.clip(np.array(vals), self._lo, self._hi)
                for j, v in zip(self._free, vals):
                    p.resetJointState(robot.robot_id, j, targetValue=float(v),
                                      physicsClientId=client)
                pose = get_link_pose(robot.robot_id, robot.end_effector_id, client)
                if np.linalg.norm(np.array(pose.position) - position) <= tol:
                    return vals
                rest = vals.tolist()
        finally:
            for j, (pos, vel, _, _) in zip(self._free, init):
                p.resetJointState(robot.robot_id, j, targetValue=pos,
                                  targetVelocity=vel, physicsClientId=client)
        return None

    def _plain_ik_within_limits(self, position: np.ndarray, quat: np.ndarray,
                                tol: float) -> Optional[np.ndarray]:
        robot, client = self.robot, self.client
        init = p.getJointStates(robot.robot_id, self._free, physicsClientId=client)
        try:
            for _ in range(50):
                vals = np.array(p.calculateInverseKinematics(
                    robot.robot_id, robot.end_effector_id, list(position),
                    targetOrientation=list(quat), maxNumIterations=200,
                    residualThreshold=1e-5, physicsClientId=client))
                vals = np.clip(vals, self._lo, self._hi)
                for j, v in zip(self._free, vals):
                    p.resetJointState(robot.robot_id, j, targetValue=float(v),
                                      physicsClientId=client)
                pose = get_link_pose(robot.robot_id, robot.end_effector_id, client)
                if np.linalg.norm(np.array(pose.position) - position) <= tol:
                    return vals
        finally:
            for j, (pos, vel, _, _) in zip(self._free, init):
                p.resetJointState(robot.robot_id, j, targetValue=pos,
                                  targetVelocity=vel, physicsClientId=client)
        return None

    # ── High-level commands ─────────────────────────────────────

    def set_gripper(self, gripper: str, steps: Optional[int] = None) -> int:
        """Open or close the gripper in place. Returns steps used."""
        n = self.finger_settle_steps if steps is None else steps
        target = self._finger_target_for(gripper)
        joints = self.robot.get_joints()
        for _ in range(n):
            self.step_fn(self._action_from_joints(joints, target))
        return n

    def move_to(self,
                position: Sequence[float],
                quat: Optional[Sequence[float]] = None,
                gripper: str = GRIPPER_KEEP,
                max_steps: int = 200) -> MoveResult:
        """Move the EE to ``position`` (and ``quat`` if given) in a straight
        line, then apply the gripper command. Blocking."""
        target_pos = np.asarray(position, dtype=float)
        target_rot = _quat_xyzw_to_rot(quat) if quat is not None \
            else _quat_xyzw_to_rot(self.ee_quat())
        finger_target = self._finger_target_for(GRIPPER_KEEP)
        start_pos = self.ee_position()
        start_rot = _quat_xyzw_to_rot(self.ee_quat())
        total_dist = float(np.linalg.norm(target_pos - start_pos))
        total_angle = float((target_rot * start_rot.inv()).magnitude())
        n_pos = int(np.ceil(total_dist / self.step_size))
        n_orn = int(np.ceil(total_angle / self.max_yaw_step))
        n_way = max(1, n_pos, n_orn)
        slerp = Slerp([0.0, 1.0], Rotation.concatenate([start_rot,
                                                        target_rot]))
        steps = 0
        ik_failed = False
        stalled = False
        force_limited = False
        force_bodies: list = []
        stall_count = 0
        last_pos = start_pos.copy()
        message = ""
        # Waypoints are spaced along the straight line; each is executed
        # with ONE env step, so the EE follows at roughly step_size per step.
        # Once all waypoints are issued we keep issuing the final target
        # until reached, stalled, or out of steps.
        way_idx = 1
        while steps < max_steps:
            frac = min(1.0, way_idx / n_way)
            wp_pos = start_pos + frac * (target_pos - start_pos)
            wp_rot = slerp([frac])[0]
            joints = self._ik(wp_pos, wp_rot.as_quat())
            if joints is None:
                # Try the final target directly once before giving up.
                joints = self._ik(target_pos, target_rot.as_quat())
                if joints is None:
                    ik_failed = True
                    message = ("IK failed for waypoint "
                               f"{np.round(wp_pos, 4).tolist()}.")
                    break
            self.step_fn(self._action_from_joints(joints, finger_target))
            steps += 1
            way_idx += 1
            cur = self.ee_position()
            force, force_bodies = self.contact_force()
            if force > self.max_contact_force:
                force_limited = True
                message = (f"Stopped: contact force {force:.0f} N on "
                           f"{', '.join(force_bodies)} exceeded the "
                           f"{self.max_contact_force:.0f} N limit.")
                break
            dist = float(np.linalg.norm(target_pos - cur))
            ang = float((target_rot * _quat_xyzw_to_rot(
                self.ee_quat()).inv()).magnitude())
            if dist <= self.pos_tol and ang <= self.orn_tol:
                break
            if way_idx > n_way:
                # Converging on the final target: detect a blocked arm.
                if np.linalg.norm(cur - last_pos) < self.stall_eps:
                    stall_count += 1
                else:
                    stall_count = 0
                if stall_count >= self.stall_steps:
                    stalled = True
                    message = "EE stopped moving before reaching the target."
                    break
            last_pos = cur
        if gripper != GRIPPER_KEEP:
            steps += self.set_gripper(gripper)
        cur = self.ee_position()
        dist = float(np.linalg.norm(target_pos - cur))
        ang = float((target_rot *
                     _quat_xyzw_to_rot(self.ee_quat()).inv()).magnitude())
        reached = dist <= self.pos_tol and ang <= self.orn_tol
        if steps >= max_steps and not reached and not message:
            message = f"Ran out of steps (max_steps={max_steps})."
        return MoveResult(reached=reached,
                          ik_failed=ik_failed,
                          stalled=stalled or force_limited,
                          steps=steps,
                          final_position=tuple(float(v) for v in cur),
                          final_rpy_deg=self.rpy_relative_to_home_deg(),
                          distance_to_target=dist,
                          gripper=self.gripper_value(),
                          message=message,
                          extra={
                              "orientation_error_deg": float(np.degrees(ang)),
                              "holding": self.is_holding(),
                              "contacts": self.contact_names(),
                              "force_limited": force_limited,
                          })
