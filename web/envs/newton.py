"""Newton's cradle: five suspended steel balls, the leftmost lifted and
released. Momentum transfers through the row and pops the rightmost out
in a near-elastic collision.

A useful demo because it exercises constraints (point-to-point hangs)
and high-restitution contacts, things neither blocks nor cover touched.
"""
from __future__ import annotations

import math

from web.envs.base import BaseDemoEnv


_NUM_BALLS = 5
_BALL_RADIUS = 0.022
_BALL_MASS = 0.2
_ROPE_LENGTH = 0.20
_BALL_GAP = 0.0  # balls touching at rest
_ANCHOR_HEIGHT = _ROPE_LENGTH + _BALL_RADIUS + 0.005
# Pull the leftmost ball back by this angle (radians) at start.
_PULL_ANGLE = 0.55  # ~31°


class NewtonCradleEnv(BaseDemoEnv):
    name = "newton"

    def _build(self) -> None:
        self.spawn_plane(friction=0.9)

        spacing = 2 * _BALL_RADIUS + _BALL_GAP
        first_x = -((_NUM_BALLS - 1) / 2.0) * spacing

        for i in range(_NUM_BALLS):
            rest_x = first_x + i * spacing

            # The first (leftmost) ball starts displaced; the rest sit at rest.
            if i == 0:
                # Swing point is directly above the resting position.
                anchor = (rest_x, 0.0, _ANCHOR_HEIGHT)
                ball_x = rest_x - _ROPE_LENGTH * math.sin(_PULL_ANGLE)
                ball_z = _ANCHOR_HEIGHT - _ROPE_LENGTH * math.cos(_PULL_ANGLE)
                position = (ball_x, 0.0, ball_z)
                color = (0.95, 0.4, 0.3, 1.0)  # red, the "active" ball
            else:
                anchor = (rest_x, 0.0, _ANCHOR_HEIGHT)
                position = (rest_x, 0.0, _BALL_RADIUS + 0.005)
                color = (0.75, 0.78, 0.82, 1.0)  # steel grey

            bid = self.spawn_sphere(
                radius=_BALL_RADIUS,
                mass=_BALL_MASS,
                position=position,
                color=color,
                friction=0.3,
                rolling_friction=0.0001,
                restitution=0.95,  # near-elastic collisions
            )
            self.hang_from_world(bid, anchor)

    def _policy(self, dt: float) -> None:
        # Pure physics — no scripted motion needed.
        return
