"""Bowling: rolling ball flies down a lane and scatters a triangle of
ten "pins" (boxes).

Showcases rolling friction, rolling-as-rotation (so the ball spins, not
just slides), and high-multiplicity contact resolution as pins cascade
into each other.
"""
from __future__ import annotations

from web.envs.base import BaseDemoEnv


_PIN_HALF = (0.015, 0.015, 0.06)
_PIN_MASS = 0.1
_PIN_GAP = 0.06          # center-to-center, both axes
_BALL_RADIUS = 0.045
_BALL_MASS = 1.8
_BALL_START_X = -0.45
_LANE_Z = 0.0


class BowlingEnv(BaseDemoEnv):
    name = "bowling"

    def _build(self) -> None:
        self.spawn_plane(friction=0.4)

        # 4-3-2-1 triangle of pins (10 pins, classic bowling pyramid),
        # pointing at the ball that will roll along +x.
        pin_x0 = 0.10
        for row in range(4):
            count = 4 - row
            for i in range(count):
                # Center the row in y.
                offset = (i - (count - 1) / 2.0) * _PIN_GAP
                x = pin_x0 + row * _PIN_GAP
                y = offset
                self.spawn_box(
                    half_extents=_PIN_HALF,
                    mass=_PIN_MASS,
                    position=(x, y, _PIN_HALF[2] + 0.001),
                    color=(0.95, 0.92, 0.85, 1.0),
                    friction=0.5,
                    restitution=0.2,
                )

        # The ball.
        ball = self.spawn_sphere(
            radius=_BALL_RADIUS,
            mass=_BALL_MASS,
            position=(_BALL_START_X, 0.0, _BALL_RADIUS + 0.001),
            color=(0.25, 0.45, 0.85, 1.0),
            friction=0.6,
            rolling_friction=0.005,
            restitution=0.1,
        )
        # Kick the ball forward. It'll roll the rest of the way because
        # of rolling-friction-vs-sliding interaction.
        import pybullet as p
        p.resetBaseVelocity(
            ball,
            linearVelocity=[3.5, 0.0, 0.0],
            angularVelocity=[0.0, 3.5 / _BALL_RADIUS, 0.0],
            physicsClientId=self._client_id,
        )

    def _policy(self, dt: float) -> None:
        return
