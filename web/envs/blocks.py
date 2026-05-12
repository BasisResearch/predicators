"""Blocks demo: scripted floating gripper stacks cubes into a tower.

Now backed by real PyBullet (via pyodide-bullet) — no adapter.
"""
from __future__ import annotations

from typing import Optional

from web.envs.base import BaseDemoEnv


_GRIPPER_SIZE = (0.04, 0.04, 0.04)
_BLOCK_SIZE = (0.05, 0.05, 0.05)
_BLOCK_COLORS = [
    (0.85, 0.3, 0.3, 1.0),
    (0.3, 0.6, 0.85, 1.0),
    (0.3, 0.8, 0.4, 1.0),
    (0.95, 0.7, 0.2, 1.0),
    (0.7, 0.3, 0.85, 1.0),
]
_HOVER_Z = 0.35
_APPROACH_SPEED = 0.6
_LIFT_SPEED = 0.4
_NUM_BLOCKS = 4


class BlocksEnv(BaseDemoEnv):
    name = "blocks"

    def __init__(self, seed: int = 0) -> None:
        super().__init__(seed)
        self._gripper: int = -1
        self._blocks: list[int] = []
        self._grasp_constraint: Optional[int] = None
        self._held: Optional[int] = None
        self._plan: list[tuple] = []
        self._stage: str = "idle"
        self._target_pos = (0.0, 0.0, _HOVER_Z)

    def _build(self) -> None:
        self.spawn_plane(friction=0.9)
        self._blocks = []
        for i in range(_NUM_BLOCKS):
            x = -0.18 + i * 0.12
            y = self._rng.uniform(-0.05, 0.05)
            color = _BLOCK_COLORS[i % len(_BLOCK_COLORS)]
            bid = self.spawn_box(
                half_extents=_BLOCK_SIZE,
                mass=0.2,
                position=(x, y, _BLOCK_SIZE[2] + 0.001),
                color=color,
                friction=0.8,
            )
            self._blocks.append(bid)

        self._gripper = self.spawn_box(
            half_extents=_GRIPPER_SIZE,
            mass=0.0,
            position=(0.0, 0.0, _HOVER_Z),
            color=(0.2, 0.2, 0.25, 1.0),
            kinematic=True,
        )
        self._held = None
        self._grasp_constraint = None

        target_x, target_y = -0.18, 0.0
        self._plan = []
        for level, src in enumerate(self._blocks[1:], start=1):
            self._plan.append(("pick", src))
            place_z = (2 * level + 1) * _BLOCK_SIZE[2] + 0.001
            self._plan.append(("place", (target_x, target_y, place_z)))
        self._stage = "idle"
        self._target_pos = (0.0, 0.0, _HOVER_Z)

    def _move_toward(self, target, speed, dt) -> bool:
        import pybullet as p
        cur, _ = self.get_pose(self._gripper)
        dx, dy, dz = target[0] - cur[0], target[1] - cur[1], target[2] - cur[2]
        d = (dx * dx + dy * dy + dz * dz) ** 0.5
        if d < 1e-3:
            p.resetBasePositionAndOrientation(
                self._gripper, list(target), [0, 0, 0, 1],
                physicsClientId=self._client_id)
            return True
        step = min(speed * dt, d)
        f = step / d
        new = (cur[0] + dx * f, cur[1] + dy * f, cur[2] + dz * f)
        p.resetBasePositionAndOrientation(
            self._gripper, list(new), [0, 0, 0, 1],
            physicsClientId=self._client_id)
        return False

    def _policy(self, dt: float) -> None:
        if self._stage == "idle":
            if not self._plan:
                return
            kind, arg = self._plan[0]
            if kind == "pick":
                bx, by, _ = self.get_pose(arg)[0]
                self._target_pos = (bx, by, _HOVER_Z)
                self._stage = "above_pick"
            elif kind == "place":
                tx, ty, _ = arg
                self._target_pos = (tx, ty, _HOVER_Z)
                self._stage = "above_place"

        elif self._stage == "above_pick":
            if self._move_toward(self._target_pos, _APPROACH_SPEED, dt):
                bid = self._plan[0][1]
                bx, by, bz = self.get_pose(bid)[0]
                # Sink the gripper so its bottom is right above the block top.
                self._target_pos = (bx, by, bz + _GRIPPER_SIZE[2] + _BLOCK_SIZE[2])
                self._stage = "descend_pick"

        elif self._stage == "descend_pick":
            if self._move_toward(self._target_pos, _LIFT_SPEED, dt):
                self._grasp_constraint = self.grasp(
                    self._gripper, self._plan[0][1])
                self._held = self._plan[0][1]
                self._target_pos = (self._target_pos[0],
                                    self._target_pos[1], _HOVER_Z)
                self._stage = "lift"

        elif self._stage == "lift":
            if self._move_toward(self._target_pos, _LIFT_SPEED, dt):
                self._plan.pop(0)
                self._stage = "idle"

        elif self._stage == "above_place":
            if self._move_toward(self._target_pos, _APPROACH_SPEED, dt):
                tx, ty, tz = self._plan[0][1]
                self._target_pos = (tx, ty, tz + _GRIPPER_SIZE[2] + _BLOCK_SIZE[2])
                self._stage = "descend_place"

        elif self._stage == "descend_place":
            if self._move_toward(self._target_pos, _LIFT_SPEED, dt):
                if self._grasp_constraint is not None:
                    self.release(self._grasp_constraint)
                    self._grasp_constraint = None
                    self._held = None
                self._target_pos = (self._target_pos[0],
                                    self._target_pos[1], _HOVER_Z)
                self._stage = "lift_place"

        elif self._stage == "lift_place":
            if self._move_toward(self._target_pos, _LIFT_SPEED, dt):
                self._plan.pop(0)
                self._stage = "idle"
