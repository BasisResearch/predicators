"""RGB-D -> per-object 3D particles, plus PyBullet-env helpers.

The pure functions (:func:`get_particles_from_rgbd_and_matrices`,
:func:`downsample_particles`) take raw images / arrays.  The env helpers
(:func:`capture_frame`, :func:`extract_particles`, :func:`hide_all_bodies`, and
:class:`ParticleDebugDraw`) take a live ``PyBulletEnv`` instance and were
previously methods on that class.
"""

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pybullet as p
from numpy.typing import NDArray

from particle_world_model import geometry

# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------


def get_particles_from_rgbd_and_matrices(
    rgb: NDArray,
    depth: NDArray,
    seg: NDArray,
    view_matrix: Sequence[float],
    proj_matrix: Sequence[float],
) -> Dict[int, Tuple[NDArray, NDArray]]:
    """Convert RGBD + Seg images to world-coordinate particles per object ID.

    Returns a dict mapping object ID to (points, colors).
    """
    height, width = depth.shape

    # Create meshgrid of pixel coordinates
    u, v = np.meshgrid(np.arange(width), np.arange(height))

    # Normalized Device Coordinates
    x = 2.0 * u / width - 1.0
    y = 2.0 * (height - v) / height - 1.0
    z = 2.0 * depth - 1.0

    # Reshape for matrix multiplication
    pix_pos = np.stack([x, y, z, np.ones_like(z)], axis=-1).reshape(-1, 4)

    # In PyBullet, matrices are returned as flat lists in column-major order.
    # To use them with numpy, we reshape and transpose to get standard
    # row-major.
    vm = np.array(view_matrix).reshape(4, 4).T
    pm = np.array(proj_matrix).reshape(4, 4).T

    inv_view_proj_matrix = np.linalg.inv(pm @ vm)

    # Back-project to world coordinates
    world_pos = pix_pos @ inv_view_proj_matrix.T
    world_pos /= world_pos[:, 3:]
    points = world_pos[:, :3]

    # Colors
    colors = rgb.reshape(-1, 3) / 255.0

    # Segmentation
    seg_flat = seg.flatten()

    unique_ids = np.unique(seg_flat)
    particles = {}
    for obj_id in unique_ids:
        if obj_id < 0:  # Usually -1 is background or robot
            continue
        mask = (seg_flat == obj_id)
        obj_points = points[mask]
        obj_colors = colors[mask]
        particles[obj_id] = (obj_points, obj_colors)

    return particles


def downsample_particles(
    points: NDArray,
    colors: NDArray,
    max_particles: int = 20,
) -> Tuple[NDArray, NDArray]:
    """Downsample particles using voxel downsampling with Open3D.

    The voxel size is automatically determined to yield approximately
    max_particles.
    """
    import open3d as o3d
    num_points = len(points)
    if num_points <= max_particles:
        return points, colors

    # Determine the voxel size based on the bounding box and max_particles.
    min_bounds = np.min(points, axis=0)
    max_bounds = np.max(points, axis=0)
    ranges = max_bounds - min_bounds
    # Use only dimensions with non-zero range.
    nonzero_ranges = ranges[ranges > 1e-6]
    if len(nonzero_ranges) == 0:
        # Fall back to random downsampling if all points are co-located.
        indices = np.random.choice(num_points, max_particles, replace=False)
        return points[indices], colors[indices]

    # Voxel size estimation: (prod(nonzero_ranges) / max_particles)**(1/dim)
    voxel_size = (np.prod(nonzero_ranges) /
                  max_particles)**(1.0 / len(nonzero_ranges))

    # Perform voxel downsampling using Open3D.
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    down_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

    res_points = np.asarray(down_pcd.points)
    res_colors = np.asarray(down_pcd.colors)

    # If still too many points, randomly sample.
    if len(res_points) > max_particles:
        indices = np.random.choice(len(res_points),
                                   max_particles,
                                   replace=False)
        return res_points[indices], res_colors[indices]

    return res_points, res_colors


# ---------------------------------------------------------------------------
# PyBullet-env helpers (formerly methods on PyBulletEnv)
# ---------------------------------------------------------------------------


def capture_frame(
    pybullet_env: Any,
) -> Tuple[NDArray, NDArray, NDArray, Sequence[float], Sequence[float], int,
           int]:
    """Grab one camera frame from a live PyBullet env.

    Returns ``(rgb uint8 HxWx3, depth float32 HxW, seg int HxW, view_matrix,
    proj_matrix, width, height)``.
    """
    view_matrix, proj_matrix, width, height = \
        pybullet_env._get_camera_matrices()
    _, _, rgb, depth, seg = p.getCameraImage(
        width=width,
        height=height,
        viewMatrix=view_matrix,
        projectionMatrix=proj_matrix,
        renderer=p.ER_BULLET_HARDWARE_OPENGL,
        physicsClientId=pybullet_env._physics_client_id)
    rgb_arr = np.array(rgb, dtype=np.uint8).reshape((height, width, 4))[:, :, :3]
    depth_arr = np.array(depth, dtype=np.float32).reshape((height, width))
    seg_arr = np.array(seg).reshape((height, width))
    return rgb_arr, depth_arr, seg_arr, view_matrix, proj_matrix, width, height


def extract_particles(pybullet_env: Any) -> Dict[Any, Tuple[NDArray, NDArray]]:
    """Extract downsampled point clouds (particles) for each env object.

    Returns a dict mapping each ``Object`` in ``pybullet_env._objects`` (that is
    visible in the current frame) to ``(points, colors)``.
    """
    rgb_arr, depth_arr, seg_arr, view_matrix, proj_matrix, _, _ = \
        capture_frame(pybullet_env)

    particles_by_id = get_particles_from_rgbd_and_matrices(
        rgb_arr, depth_arr, seg_arr, view_matrix, proj_matrix)

    particles_by_obj = {}
    for obj in pybullet_env._objects:
        if obj.id in particles_by_id:
            points, colors = particles_by_id[obj.id]
            points, colors = downsample_particles(points, colors)
            particles_by_obj[obj] = (points, colors)

    return particles_by_obj


def hide_all_bodies(pybullet_env: Any) -> None:
    """Hide all bodies in the current simulation by moving them far away."""
    client_id = pybullet_env._physics_client_id
    for body_id in range(p.getNumBodies(physicsClientId=client_id)):
        p.resetBasePositionAndOrientation(body_id, [100, 100, 100],
                                          [0, 0, 0, 1],
                                          physicsClientId=client_id)


class ParticleDebugDraw:
    """Draw and clear particle debug points in a PyBullet GUI.

    Replaces the old ``PyBulletEnv.display_particles`` / ``clear_particles``
    method pair; the instance owns the list of debug-item ids to remove.
    """

    def __init__(self, pybullet_env: Any) -> None:
        self._client_id = pybullet_env._physics_client_id
        self._debug_ids: List[int] = []

    def display(
        self,
        particles: Dict[Any, Tuple[NDArray, NDArray]],
        show_coordinate_frame: bool = True,
        point_size: float = 10.0,
        obj_color: bool = False,
    ) -> None:
        """Visualize particles as debug points (optionally with an axis triad)."""
        for points, colors in particles.values():
            item_id = p.addUserDebugPoints(
                points,
                colors if obj_color else np.ones_like(colors),
                pointSize=point_size,
                physicsClientId=self._client_id)
            self._debug_ids.append(item_id)
        if show_coordinate_frame:
            for start, end, color in [([0, 0, 0], [0.1, 0, 0], [1, 0, 0]),
                                      ([0, 0, 0], [0, 0.1, 0], [0, 1, 0]),
                                      ([0, 0, 0], [0, 0, 0.1], [0, 0, 1])]:
                item_id = p.addUserDebugLine(start,
                                             end,
                                             color,
                                             physicsClientId=self._client_id)
                self._debug_ids.append(item_id)

    def clear(self) -> None:
        """Remove every debug item added by :meth:`display`."""
        for item_id in self._debug_ids:
            p.removeUserDebugItem(item_id, physicsClientId=self._client_id)
        self._debug_ids = []
