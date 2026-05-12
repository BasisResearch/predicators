"""Tiny base class for the browser demo envs — now backed by real PyBullet.

We track each body's render metadata (kind, half_extents, color) on the
Python side so the JS host can mirror them into a Three.js scene by
polling. PyBullet itself doesn't expose anything resembling a
"scene-listener" callback.
"""
from __future__ import annotations

import abc
import random
from typing import Optional

import pybullet as p
import pybullet_data


_TIMESTEP = 1.0 / 240.0  # PyBullet's default; one substep per call.


class BaseDemoEnv(abc.ABC):
    """Minimal env scaffold.

    Subclasses implement ``_build`` (spawn bodies, set up the plan) and
    ``_policy`` (called once per step before physics).
    """

    name: str = "base"

    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)
        self._client_id: int = -1
        self._bodies: list[int] = []
        # Render metadata: body_id -> {"kind": "box"|"plane",
        # "half_extents": (x,y,z), "color": (r,g,b,a)}.
        self._render_meta: dict[int, dict] = {}
        self._t = 0.0

    # ---- lifecycle ------------------------------------------------------
    def connect(self) -> None:
        self._client_id = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(),
                                  physicsClientId=self._client_id)
        p.setGravity(0, 0, -9.81, physicsClientId=self._client_id)

    def disconnect(self) -> None:
        if self._client_id != -1:
            p.disconnect(self._client_id)
            self._client_id = -1

    def reset(self) -> None:
        for bid in self._bodies:
            try:
                p.removeBody(bid, physicsClientId=self._client_id)
            except p.error:
                pass
        self._bodies.clear()
        self._render_meta.clear()
        self._t = 0.0
        self._build()

    def step(self, dt: float = 1.0 / 60.0) -> None:
        self._t += dt
        self._policy(dt)
        substeps = max(1, int(round(dt / _TIMESTEP)))
        for _ in range(substeps):
            p.stepSimulation(physicsClientId=self._client_id)

    # ---- abstract -------------------------------------------------------
    @abc.abstractmethod
    def _build(self) -> None:
        ...

    @abc.abstractmethod
    def _policy(self, dt: float) -> None:
        ...

    # ---- body spawning (records render metadata) ------------------------
    def spawn_box(
        self,
        half_extents: tuple[float, float, float],
        mass: float,
        position: tuple[float, float, float],
        orientation: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
        color: tuple[float, float, float, float] = (0.7, 0.7, 0.7, 1.0),
        friction: float = 0.6,
        kinematic: bool = False,
    ) -> int:
        col = p.createCollisionShape(p.GEOM_BOX,
                                     halfExtents=list(half_extents),
                                     physicsClientId=self._client_id)
        vis = p.createVisualShape(p.GEOM_BOX,
                                  halfExtents=list(half_extents),
                                  rgbaColor=list(color),
                                  physicsClientId=self._client_id)
        body = p.createMultiBody(baseMass=0.0 if kinematic else mass,
                                 baseCollisionShapeIndex=col,
                                 baseVisualShapeIndex=vis,
                                 basePosition=list(position),
                                 baseOrientation=list(orientation),
                                 physicsClientId=self._client_id)
        p.changeDynamics(body, -1, lateralFriction=friction,
                         physicsClientId=self._client_id)
        self._bodies.append(body)
        self._render_meta[body] = {
            "kind": "box",
            "half_extents": tuple(half_extents),
            "color": tuple(color),
        }
        return body

    def spawn_plane(self, friction: float = 0.9) -> int:
        body = p.loadURDF("plane.urdf", physicsClientId=self._client_id)
        p.changeDynamics(body, -1, lateralFriction=friction,
                         physicsClientId=self._client_id)
        self._bodies.append(body)
        self._render_meta[body] = {"kind": "plane"}
        return body

    # ---- grasp helpers (kinematic gripper) -----------------------------
    def grasp(self, parent_id: int, child_id: int) -> int:
        parent_pos, _ = p.getBasePositionAndOrientation(
            parent_id, physicsClientId=self._client_id)
        child_pos, _ = p.getBasePositionAndOrientation(
            child_id, physicsClientId=self._client_id)
        offset = (child_pos[0] - parent_pos[0],
                  child_pos[1] - parent_pos[1],
                  child_pos[2] - parent_pos[2])
        return p.createConstraint(
            parentBodyUniqueId=parent_id,
            parentLinkIndex=-1,
            childBodyUniqueId=child_id,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=list(offset),
            childFramePosition=[0, 0, 0],
            physicsClientId=self._client_id,
        )

    def release(self, constraint_id: int) -> None:
        p.removeConstraint(constraint_id,
                           physicsClientId=self._client_id)

    # ---- accessors used by the JS host ---------------------------------
    def get_pose(self, body_id: int):
        pos, orn = p.getBasePositionAndOrientation(
            body_id, physicsClientId=self._client_id)
        return tuple(pos), tuple(orn)

    def get_render_manifest(self) -> list[dict]:
        """List the body records the renderer needs to instantiate or sync."""
        out = []
        for bid, meta in self._render_meta.items():
            entry = {"id": bid, **meta}
            out.append(entry)
        return out

    def get_pose_dict(self) -> dict:
        """{id: ([x, y, z], [qx, qy, qz, qw])} for every body."""
        out = {}
        for bid in self._bodies:
            pos, orn = self.get_pose(bid)
            out[bid] = (list(pos), list(orn))
        return out
