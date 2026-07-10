"""Pose / frame utilities for the real-bench domino env.

Ported (logic unchanged) from ``babyrobot.geometry`` so
``pybullet_domino_real`` is self-contained predicators code -- it imports
nothing from babyrobot. Only the pieces the env task-builder needs are here:
a quaternion-native ``Pose6D`` and the real-base-frame ->
predicators-domino-world transplant.

Conventions:
  * Poses are in the **robot base frame**; rotation stored as ``quat_xyzw``
    (scipy / PyBullet order) -- lossless and singularity-free (a STANDING domino
    has pitch near +-90 deg, which gimbal-locks an euler decomposition).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class Pose6D:
    """A rigid pose in the robot base frame (translation + unit quaternion)."""
    xyz: Tuple[float, float, float]
    quat_xyzw: Tuple[float, float, float, float]

    @classmethod
    def from_matrix(cls, T: np.ndarray) -> "Pose6D":
        """Build from a 4x4 homogeneous transform."""
        T = np.asarray(T, dtype=float)
        xyz = tuple(float(v) for v in T[:3, 3])
        quat = tuple(
            float(v) for v in Rotation.from_matrix(T[:3, :3]).as_quat())
        return cls(xyz, quat)  # type: ignore[arg-type]

    def to_matrix(self) -> np.ndarray:
        """Return the 4x4 homogeneous transform."""
        T = np.eye(4)
        T[:3, :3] = Rotation.from_quat(self.quat_xyzw).as_matrix()
        T[:3, 3] = self.xyz
        return T


def domino_upright_yaw(pose: Pose6D) -> float:
    """Env yaw for a STANDING domino: heading of its width (body-y) axis.

    The domino body frame is x=L, y=W, z=H; a standing domino has body-x
    (L) vertical (pitch ~+-90 deg), exactly where a naive euler yaw read
    gimbal-locks. ``to_matrix()`` is exact regardless of orientation, so
    read the horizontal heading of the width axis directly off it.
    """
    w_axis = pose.to_matrix()[:3, 1]  # body-y (W) in world
    return float(np.arctan2(w_axis[1], w_axis[0]))


# --- real base frame <-> predicators pybullet_domino WORLD frame -------------
# The domino options live in a Fetch-style world: robot base at (0.75, 0.72, z),
# yaw +pi/2, table top z=0.4. A bench Franka has the table BELOW its base (real
# domino z negative), so we transplant the whole real scene by a rigid
# yaw+translation, lifting by ``z_off`` so the real table plane lands on the env
# table top. Rigid => robot<->object geometry (hence planned joints) is
# preserved.
DOMINO_WORLD_ROBOT_XY = (0.75, 0.72)
DOMINO_WORLD_ROBOT_YAW = np.pi / 2
DOMINO_WORLD_TABLE_TOP_Z = 0.4


def domino_world_z_offset(table_z_base: float) -> float:
    """``z_off`` mapping the real table (base-frame height ``table_z_base``,
    typically negative) onto the env table top (0.4)."""
    return DOMINO_WORLD_TABLE_TOP_Z - float(table_z_base)


def base_to_world_transform(z_off: float) -> np.ndarray:
    """4x4 ``T_world_base``: real base at env robot base (xy=(0.75,0.72),
    z=``z_off``) with yaw +pi/2."""
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("z", DOMINO_WORLD_ROBOT_YAW).as_matrix()
    T[:3,
      3] = (DOMINO_WORLD_ROBOT_XY[0], DOMINO_WORLD_ROBOT_XY[1], float(z_off))
    return T


def pose_base_to_world(pose_base: Pose6D, z_off: float) -> Pose6D:
    """Map a real-base-frame pose into the predicators domino world frame."""
    return Pose6D.from_matrix(
        base_to_world_transform(z_off) @ pose_base.to_matrix())
