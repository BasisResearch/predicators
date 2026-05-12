"""Cover demo: kinematic puck pushes blocks onto target zones.

Now backed by real PyBullet.
"""
from __future__ import annotations

from typing import Optional

from web.envs.base import BaseDemoEnv


_BLOCK_HALF = (0.025, 0.05, 0.025)
_TARGET_HALF = (0.04, 0.07, 0.001)
_PUSHER_HALF = (0.04, 0.04, 0.025)
# Two heights: approach travel goes well above placed blocks
# (block tops are at z ≈ 0.05); push contact drops to block centre.
_TRAVEL_Z = 0.15
_PUSH_Z = 0.025
_APPROACH_SPEED = 0.4
_DESCENT_SPEED = 0.6
_NUM_PAIRS = 3


class CoverEnv(BaseDemoEnv):
    name = "cover"

    def __init__(self, seed: int = 0) -> None:
        super().__init__(seed)
        self._pusher: int = -1
        self._blocks: list[int] = []
        self._target_xs: list[float] = []
        self._plan: list[tuple[int, float]] = []
        self._stage: str = "idle"
        self._target_pos = (0.0, 0.0, _TRAVEL_Z)
        self._grasp_constraint: Optional[int] = None
        self._held: Optional[int] = None

    def _build(self) -> None:
        self.spawn_plane(friction=0.6)
        block_colors = [
            (0.85, 0.3, 0.3, 1.0),
            (0.3, 0.6, 0.85, 1.0),
            (0.3, 0.8, 0.4, 1.0),
        ]
        target_colors = [
            (0.85, 0.3, 0.3, 0.5),
            (0.3, 0.6, 0.85, 0.5),
            (0.3, 0.8, 0.4, 0.5),
        ]
        # Rows spaced 0.13 m apart in y so that adjacent blocks
        # (half-y = 0.05) and the pusher (half-y = 0.04) have real
        # clearance from each other when the pusher traverses a row.
        self._blocks = []
        self._target_xs = []
        row_pitch = 0.13
        first_y = -row_pitch * (_NUM_PAIRS - 1) / 2
        for i in range(_NUM_PAIRS):
            block_y = first_y + i * row_pitch
            bid = self.spawn_box(
                half_extents=_BLOCK_HALF,
                mass=0.15,
                position=(-0.20, block_y, _BLOCK_HALF[2] + 0.001),
                color=block_colors[i],
                friction=0.4,
            )
            self._blocks.append(bid)

            target_x = 0.18
            self._target_xs.append(target_x)
            self.spawn_box(
                half_extents=_TARGET_HALF,
                mass=0.0,
                position=(target_x, block_y, _TARGET_HALF[2]),
                color=target_colors[i],
                friction=0.0,
                kinematic=True,
            )

        self._pusher = self.spawn_box(
            half_extents=_PUSHER_HALF,
            mass=0.0,
            position=(0.0, -0.3, _TRAVEL_Z),
            color=(0.2, 0.2, 0.25, 1.0),
            kinematic=True,
        )
        self._plan = [(self._blocks[i], self._target_xs[i])
                      for i in range(_NUM_PAIRS)]
        self._stage = "idle"

    def _move_toward(self, target, dt, speed=_APPROACH_SPEED) -> bool:
        import pybullet as p
        cur, _ = self.get_pose(self._pusher)
        dx, dy, dz = target[0] - cur[0], target[1] - cur[1], target[2] - cur[2]
        d = (dx * dx + dy * dy + dz * dz) ** 0.5
        if d < 1e-3:
            p.resetBasePositionAndOrientation(
                self._pusher, list(target), [0, 0, 0, 1],
                physicsClientId=self._client_id)
            return True
        step = min(speed * dt, d)
        f = step / d
        new = (cur[0] + dx * f, cur[1] + dy * f, cur[2] + dz * f)
        p.resetBasePositionAndOrientation(
            self._pusher, list(new), [0, 0, 0, 1],
            physicsClientId=self._client_id)
        return False

    def _policy(self, dt: float) -> None:
        # Six-stage state machine. Approach from the WEST (not the
        # south) so the pusher stays in the same y-row as the block it
        # carries. With rows 0.10 m apart in y and pusher+block half-y
        # summing to 0.09, the pusher has just enough clearance from
        # neighbouring rows.
        #
        #   idle        — pick next block, fly above-west of it
        #   approach    — move at _TRAVEL_Z to (block.x-gap, block.y)
        #   descend     — drop straight down to _PUSH_Z
        #   contact     — slide east at _PUSH_Z to touch block's west face
        #   push        — drag the block east to (target_x, block.y)
        #   ascend      — lift straight up to _TRAVEL_Z
        if self._stage == "idle":
            if not self._plan:
                return
            block_id, _ = self._plan[0]
            bx, by, _ = self.get_pose(block_id)[0]
            self._target_pos = (bx - 0.10, by, _TRAVEL_Z)
            self._stage = "approach"

        elif self._stage == "approach":
            if self._move_toward(self._target_pos, dt):
                self._target_pos = (self._target_pos[0],
                                    self._target_pos[1], _PUSH_Z)
                self._stage = "descend"

        elif self._stage == "descend":
            if self._move_toward(self._target_pos, dt, _DESCENT_SPEED):
                block_id, _ = self._plan[0]
                bx, by, _ = self.get_pose(block_id)[0]
                # Pusher east face just touches block west face.
                contact_x = bx - _BLOCK_HALF[0] - _PUSHER_HALF[0]
                self._target_pos = (contact_x, by, _PUSH_Z)
                self._stage = "contact"

        elif self._stage == "contact":
            if self._move_toward(self._target_pos, dt):
                block_id, target_x = self._plan[0]
                self._grasp_constraint = self.grasp(self._pusher, block_id)
                self._held = block_id
                _, by, _ = self.get_pose(block_id)[0]
                push_x = target_x - _BLOCK_HALF[0] - _PUSHER_HALF[0]
                self._target_pos = (push_x, by, _PUSH_Z)
                self._stage = "push"

        elif self._stage == "push":
            if self._move_toward(self._target_pos, dt):
                # Release the constraint, then explicitly pin the block
                # to its target — both position and velocity. The
                # constraint's per-frame impulses leave the block with
                # ~0.4 m/s of accumulated velocity, and just zeroing
                # velocity isn't enough because the solver runs after
                # our reset on the same frame. Forcing the pose makes
                # the placement deterministic regardless of solver
                # state.
                if self._grasp_constraint is not None and self._held is not None:
                    import pybullet as p
                    target_x = self._plan[0][1]
                    _, by, _ = self.get_pose(self._held)[0]
                    self.release(self._grasp_constraint)
                    self._grasp_constraint = None
                    p.resetBasePositionAndOrientation(
                        self._held,
                        [target_x, by, _BLOCK_HALF[2] + 0.001],
                        [0, 0, 0, 1],
                        physicsClientId=self._client_id)
                    p.resetBaseVelocity(
                        self._held,
                        linearVelocity=[0, 0, 0],
                        angularVelocity=[0, 0, 0],
                        physicsClientId=self._client_id)
                    self._held = None
                self._target_pos = (self._target_pos[0],
                                    self._target_pos[1], _TRAVEL_Z)
                self._stage = "ascend"

        elif self._stage == "ascend":
            if self._move_toward(self._target_pos, dt, _DESCENT_SPEED):
                self._plan.pop(0)
                self._stage = "idle"
