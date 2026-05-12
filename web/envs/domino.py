"""Domino demo: kinematic finger tips the first domino; the rest cascade.

Real PyBullet — same chain behaviour as before but now driven by
Bullet's actual contact solver in the browser.
"""
from __future__ import annotations

import math

from web.envs.base import BaseDemoEnv


_DOMINO_HALF = (0.008, 0.025, 0.05)
_FINGER_HALF = (0.015, 0.015, 0.04)
_GAP = 0.045
_NUM_DOMINOES = 14


class DominoEnv(BaseDemoEnv):
    name = "domino"

    def __init__(self, seed: int = 0) -> None:
        super().__init__(seed)
        self._finger: int = -1
        self._dominoes: list[int] = []
        self._stage: str = "wind_up"
        self._tipping_speed = 0.45
        # x coordinate beyond which we stop driving the finger. Set in _build.
        self._stop_x = 0.0

    def _build(self) -> None:
        self.spawn_plane(friction=0.9)
        self._dominoes = []
        for i in range(_NUM_DOMINOES):
            t = i / max(1, _NUM_DOMINOES - 1)
            x = -0.30 + i * _GAP
            y = 0.08 * math.sin(t * math.pi)
            color = ((0.95, 0.8, 0.2, 1.0) if i % 2 == 0
                     else (0.4, 0.7, 0.95, 1.0))
            bid = self.spawn_box(
                half_extents=_DOMINO_HALF,
                mass=0.05,
                position=(x, y, _DOMINO_HALF[2] + 0.001),
                color=color,
                friction=0.6,
            )
            self._dominoes.append(bid)

        first_pos, _ = self.get_pose(self._dominoes[0])
        # Make the finger dynamic so its velocity actually transfers to
        # contacts. PyBullet's mass=0 bodies don't impart velocity to
        # dynamic bodies when teleported each frame — Bullet only nulls
        # any overlap, which isn't enough to tip a thin domino.
        self._finger = self.spawn_box(
            half_extents=_FINGER_HALF,
            mass=0.3,
            position=(first_pos[0] - 0.08, first_pos[1],
                      first_pos[2] + _DOMINO_HALF[2] * 0.7),
            color=(0.2, 0.2, 0.25, 1.0),
            kinematic=False,
        )
        self._stop_x = first_pos[0] + _DOMINO_HALF[0] + 0.02
        self._stage = "wind_up"

    def _policy(self, dt: float) -> None:
        import pybullet as p
        if self._stage == "wind_up":
            cur, _ = self.get_pose(self._finger)
            if cur[0] >= self._stop_x:
                # Done driving; arrest the finger and let physics play out.
                p.resetBaseVelocity(self._finger, [0, 0, 0], [0, 0, 0],
                                    physicsClientId=self._client_id)
                self._stage = "watching"
                return
            # Drive forward at constant velocity. Counter gravity so the
            # finger floats instead of falling onto the table.
            p.resetBaseVelocity(
                self._finger,
                linearVelocity=[self._tipping_speed, 0, 0],
                angularVelocity=[0, 0, 0],
                physicsClientId=self._client_id,
            )
            # mass * g = 0.3 * 9.81 ≈ 2.943 N upward.
            p.applyExternalForce(
                self._finger, -1,
                forceObj=[0, 0, 0.3 * 9.81],
                posObj=[0, 0, 0],
                flags=p.WORLD_FRAME,
                physicsClientId=self._client_id,
            )
        elif self._stage == "watching":
            # Keep the finger from drifting downward after it stops.
            p.applyExternalForce(
                self._finger, -1,
                forceObj=[0, 0, 0.3 * 9.81],
                posObj=[0, 0, 0],
                flags=p.WORLD_FRAME,
                physicsClientId=self._client_id,
            )
