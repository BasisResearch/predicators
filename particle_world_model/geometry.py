"""Camera projection helpers.

All functions use PyBullet's matrix convention: ``getViewMatrix`` /
``getProjectionMatrix`` return flat 16-element lists in column-major order, so we
reshape to ``(4, 4)`` and transpose to recover standard row-major matrices.
``project_3d_to_2d`` and ``backproject_2d_to_3d`` share the same NDC convention
as :func:`particle_world_model.particles.get_particles_from_rgbd_and_matrices`,
so they are exact inverses at integer pixel locations.
"""

from typing import Sequence

import numpy as np
from numpy.typing import NDArray


def project_3d_to_2d(
    points_3d: NDArray,
    view_matrix: Sequence[float],
    proj_matrix: Sequence[float],
    width: int,
    height: int,
) -> NDArray:
    """Project world-coordinate points (N,3) to pixel coordinates (N,2)."""
    vm = np.array(view_matrix).reshape(4, 4).T
    pm = np.array(proj_matrix).reshape(4, 4).T
    vp = pm @ vm
    ones = np.ones((len(points_3d), 1))
    pts_h = np.concatenate([points_3d, ones], axis=1)  # (N, 4)
    clip = pts_h @ vp.T  # (N, 4)
    ndc_x = clip[:, 0] / clip[:, 3]
    ndc_y = clip[:, 1] / clip[:, 3]
    px = (ndc_x + 1.0) * width / 2.0
    # NDC y=+1 -> top row (v=0); NDC y=-1 -> bottom row (v=height)
    py = height - (ndc_y + 1.0) * height / 2.0
    return np.stack([px, py], axis=1)


def backproject_2d_to_3d(
    pixels_xy: NDArray,
    depth_map: NDArray,
    view_matrix: Sequence[float],
    proj_matrix: Sequence[float],
    width: int,
    height: int,
) -> NDArray:
    """Backproject pixel coordinates (N,2) + depth buffer to world points (N,3).

    Uses the same NDC convention as
    :func:`particle_world_model.particles.get_particles_from_rgbd_and_matrices`
    so the two are exact inverses at integer pixel locations.
    """
    vm = np.array(view_matrix).reshape(4, 4).T
    pm = np.array(proj_matrix).reshape(4, 4).T
    inv_vp = np.linalg.inv(pm @ vm)
    u = np.clip(np.round(pixels_xy[:, 0]).astype(int), 0, width - 1)
    v = np.clip(np.round(pixels_xy[:, 1]).astype(int), 0, height - 1)
    depth = depth_map[v, u]
    ndc_x = 2.0 * u / width - 1.0
    ndc_y = 2.0 * (height - v) / height - 1.0
    ndc_z = 2.0 * depth - 1.0
    pts = np.stack([ndc_x, ndc_y, ndc_z, np.ones_like(ndc_x)], axis=1)
    world = pts @ inv_vp.T
    world /= world[:, 3:]
    return world[:, :3]
