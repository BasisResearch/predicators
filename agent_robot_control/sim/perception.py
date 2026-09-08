"""Pixels -> named per-object particles for a live ``PyBulletEnv``.

Wraps ``particle_world_model.particles`` (RGB-D + segmentation back-projection,
voxel downsampling) and adds:

* a body-id -> name map built from the env's ``Object.id`` attributes plus the
  well-known extra bodies (robot, table), so every particle cloud is labelled
  with the env object name. This is a deliberate cheat: a real perception
  stack would have to earn these labels.
* a fixed ``ParticleSnapshot`` container and an ``.npz`` + ``.json`` writer,
  which is what the ``pixels_to_particles`` MCP tool hands to the agent.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pybullet as p

from particle_world_model.particles import capture_frame, \
    get_particles_from_rgbd_and_matrices


def farthest_point_downsample(points: np.ndarray, colors: np.ndarray,
                              k: int, rng: np.random.Generator,
                              candidate_cap: int = 4096
                              ) -> Tuple[np.ndarray, np.ndarray]:
    """Pick ``k`` well-spread points with farthest point sampling.

    The voxel heuristic in ``particle_world_model.particles`` sizes voxels
    for a filled 3-D volume; camera points lie on 2-D surfaces, so it returns
    far fewer points than requested (3 to 8 for a donut or a table). FPS on
    a random subset of at most ``candidate_cap`` pixels gives exactly ``k``
    points (or all of them if fewer are visible) with even coverage.
    """
    n = points.shape[0]
    if n <= k:
        return points, colors
    if n > candidate_cap:
        idx = rng.choice(n, candidate_cap, replace=False)
        points, colors = points[idx], colors[idx]
        n = candidate_cap
    chosen = np.empty(k, dtype=np.int64)
    chosen[0] = rng.integers(n)
    dist = np.linalg.norm(points - points[chosen[0]], axis=1)
    for i in range(1, k):
        chosen[i] = int(np.argmax(dist))
        dist = np.minimum(dist, np.linalg.norm(points - points[chosen[i]], axis=1))
    return points[chosen], colors[chosen]


@dataclass
class ParticleSnapshot:
    """Named point clouds from one camera frame."""
    points: Dict[str, np.ndarray]  # name -> (K, 3) float32 world xyz
    colors: Dict[str, np.ndarray]  # name -> (K, 3) float32 in [0, 1]
    ee_pos: np.ndarray  # (3,)
    ee_quat: np.ndarray  # (4,) xyzw
    gripper: float
    interaction_count: int
    unlabeled_body_ids: List[int] = field(default_factory=list)
    rgb: Optional[np.ndarray] = None  # (H, W, 3) uint8

    @property
    def names(self) -> List[str]:
        """Object names in a stable (sorted) order."""
        return sorted(self.points)

    def num_points(self, name: str) -> int:
        """Number of visible points for ``name`` (0 if occluded)."""
        return int(self.points[name].shape[0])

    def summary(self) -> Dict[str, Any]:
        """JSON-serialisable summary (centroids, bounding boxes, counts)."""
        objects: Dict[str, Any] = {}
        for name in self.names:
            pts = self.points[name]
            if pts.shape[0] == 0:
                objects[name] = {"num_points": 0, "visible": False}
                continue
            objects[name] = {
                "num_points": int(pts.shape[0]),
                "visible": True,
                "centroid": np.round(pts.mean(axis=0), 4).tolist(),
                "bbox_min": np.round(pts.min(axis=0), 4).tolist(),
                "bbox_max": np.round(pts.max(axis=0), 4).tolist(),
            }
        return {
            "objects": objects,
            "ee_pos": np.round(self.ee_pos, 4).tolist(),
            "ee_quat_xyzw": np.round(self.ee_quat, 4).tolist(),
            "gripper": round(float(self.gripper), 4),
            "interaction_count": int(self.interaction_count),
            "unlabeled_body_ids": list(self.unlabeled_body_ids),
        }

    def save(self, npz_path: Path, json_path: Optional[Path] = None) -> Dict[str, Any]:
        """Write ``points_<name>``/``colors_<name>`` arrays plus robot state to
        ``npz_path`` and the summary to ``json_path``. Returns the summary."""
        arrays: Dict[str, Any] = {
            "ee_pos": self.ee_pos.astype(np.float32),
            "ee_quat_xyzw": self.ee_quat.astype(np.float32),
            "gripper": np.float32(self.gripper),
            "interaction_count": np.int64(self.interaction_count),
            "object_names": np.array(self.names),
        }
        for name in self.names:
            arrays[f"points_{name}"] = self.points[name].astype(np.float32)
            arrays[f"colors_{name}"] = self.colors[name].astype(np.float32)
        np.savez(npz_path, **arrays)
        summary = self.summary()
        summary["npz_path"] = str(npz_path)
        if json_path is not None:
            json_path.write_text(json.dumps(summary, indent=2))
        return summary


def build_body_name_map(env) -> Dict[int, str]:
    """Map PyBullet body ids to env object names (plus robot and table)."""
    names: Dict[int, str] = {}
    for obj in getattr(env, "_objects", []):
        if getattr(obj, "id", None) is not None:
            names[int(obj.id)] = obj.name
    robot = env._pybullet_robot
    names[int(robot.robot_id)] = "robot"
    table_id = getattr(env, "_table_id", None)
    if isinstance(table_id, int) and table_id >= 0 and table_id not in names:
        names[table_id] = "table"
    return names


def extract_named_particles(env,
                            max_points_per_object: int = 64,
                            interaction_count: int = 0,
                            include: Optional[List[str]] = None,
                            keep_rgb: bool = False,
                            rng: Optional[np.random.Generator] = None
                            ) -> ParticleSnapshot:
    """Render one frame and return named particles for every labelled body.

    Objects in ``include`` (default: every labelled body) that are not visible
    get an empty (0, 3) array so the set of names is stable across calls.
    """
    rgb, depth, seg, view_matrix, proj_matrix, _, _ = capture_frame(env)
    # PyBullet packs (body id) | (link index + 1) << 24 into the mask when
    # link indices are requested; with the default flags we still strip the
    # high bits defensively so multi-link bodies (the robot) map correctly.
    seg = np.asarray(seg)
    seg_body = np.where(seg >= 0, seg & ((1 << 24) - 1), -1)
    by_id = get_particles_from_rgbd_and_matrices(rgb, depth, seg_body,
                                                 view_matrix, proj_matrix)
    name_map = build_body_name_map(env)
    if include is None:
        include = sorted(set(name_map.values()))
    points: Dict[str, np.ndarray] = {}
    colors: Dict[str, np.ndarray] = {}
    for name in include:
        points[name] = np.zeros((0, 3), dtype=np.float32)
        colors[name] = np.zeros((0, 3), dtype=np.float32)
    unlabeled: List[int] = []
    if rng is None:
        rng = np.random.default_rng(0)
    for body_id, (pts, cols) in by_id.items():
        name = name_map.get(int(body_id))
        if name is None:
            unlabeled.append(int(body_id))
            continue
        if name not in points:
            continue
        pts_d, cols_d = farthest_point_downsample(pts, cols,
                                                  max_points_per_object, rng)
        points[name] = np.asarray(pts_d, dtype=np.float32)
        colors[name] = np.asarray(cols_d, dtype=np.float32)
    ee_state = env._pybullet_robot.get_state()
    return ParticleSnapshot(points=points,
                            colors=colors,
                            ee_pos=np.asarray(ee_state[:3], dtype=np.float32),
                            ee_quat=np.asarray(ee_state[3:7],
                                               dtype=np.float32),
                            gripper=float(ee_state[7]),
                            interaction_count=interaction_count,
                            unlabeled_body_ids=sorted(unlabeled),
                            rgb=rgb if keep_rgb else None)


def render_rgb(env) -> np.ndarray:
    """Current camera image as (H, W, 3) uint8."""
    rgb, *_ = capture_frame(env)
    return rgb


def flatten_particles(snapshot: ParticleSnapshot, names: List[str],
                      points_per_object: int,
                      origin: Optional[np.ndarray] = None
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """Fixed-size observation vector for RL: for each name, ``points_per_object``
    xyz rows (zero-padded, expressed relative to ``origin``) and a visibility
    mask. Returns ``(points (N, P, 3), visible (N, P))``."""
    n = len(names)
    out = np.zeros((n, points_per_object, 3), dtype=np.float32)
    vis = np.zeros((n, points_per_object), dtype=np.float32)
    origin = np.zeros(3, dtype=np.float32) if origin is None else origin
    for i, name in enumerate(names):
        pts = snapshot.points.get(name, np.zeros((0, 3), dtype=np.float32))
        k = min(points_per_object, pts.shape[0])
        if k > 0:
            out[i, :k] = pts[:k] - origin
            vis[i, :k] = 1.0
    return out, vis
