"""Wrecking ball: heavy sphere on a pendulum cable swings down and
smashes a tower of blocks.

Showcases pendulum dynamics (point-to-point constraint), heavy mass +
momentum transfer, and a stack of dynamic bodies cascading under
contact load.
"""
from __future__ import annotations

import math

from web.envs.base import BaseDemoEnv


_TOWER_LEVELS = 6
_BLOCK_HALF = (0.04, 0.04, 0.025)  # 80x80x50 mm blocks
_BLOCK_GAP = 0.001                  # tiny gap so they settle cleanly

_BALL_RADIUS = 0.05
_BALL_MASS = 4.0                    # heavy enough to plough through
_ROPE_LENGTH = 0.45
_PULL_ANGLE = 1.0                   # ~57° swing-back


class WreckingEnv(BaseDemoEnv):
    name = "wrecking"

    def _build(self) -> None:
        self.spawn_plane(friction=0.9)

        # Tower of cubes — stable column at x = +0.15, alternating colors.
        tower_x = 0.15
        colors = [
            (0.85, 0.55, 0.30, 1.0),
            (0.55, 0.40, 0.30, 1.0),
        ]
        for level in range(_TOWER_LEVELS):
            z = (2 * level + 1) * _BLOCK_HALF[2] + _BLOCK_GAP * level + 0.001
            self.spawn_box(
                half_extents=_BLOCK_HALF,
                mass=0.5,
                position=(tower_x, 0.0, z),
                color=colors[level % len(colors)],
                friction=0.7,
            )

        # Anchor placement: we want the ball to enter the tower around
        # its MIDDLE, not skim its base. The tower's mid-height with
        # 6 levels is ~14 cm. Raising the anchor pushes the bottom-of-
        # swing up, so the ball cuts through the tower's body instead
        # of bouncing off the corner of the bottom block (which deflects
        # the ball UP and arcs it over the top — the "ball goes over"
        # bug).
        tower_mid_z = _TOWER_LEVELS * _BLOCK_HALF[2]  # half the tower height
        anchor_x = tower_x
        anchor_z = tower_mid_z + _ROPE_LENGTH  # bottom-of-swing = tower midline
        anchor = (anchor_x, 0.0, anchor_z)

        # Ball starts pulled back to the left at _PULL_ANGLE.
        ball_x = anchor_x - _ROPE_LENGTH * math.sin(_PULL_ANGLE)
        ball_z = anchor_z - _ROPE_LENGTH * math.cos(_PULL_ANGLE)

        ball = self.spawn_sphere(
            radius=_BALL_RADIUS,
            mass=_BALL_MASS,
            position=(ball_x, 0.0, ball_z),
            color=(0.25, 0.25, 0.28, 1.0),
            friction=0.5,
            restitution=0.05,
        )
        self.hang_from_world(ball, anchor)

    def _policy(self, dt: float) -> None:
        return
