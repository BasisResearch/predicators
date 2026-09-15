"""Offline normalized prior for a rigid assembly inside a declared free cell.

The cell and enclosing body radii are explicit geometry assumptions, not
inferred free space or evaluator metadata. This conservative component
has fixed relative geometry and either rest or a common moving twist. It
is not a complete task prior or an articulated/contact-state prior.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from typing import Mapping, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

from predicators.code_sim_learning.inference_data import content_digest
from predicators.code_sim_learning.inference_replay import CommandWeld, Pose, \
    Velocity
from predicators.code_sim_learning.inference_sampling import BoxPrior

_IDENTITY: Pose = ((0., 0., 0.), (0., 0., 0., 1.))


@dataclass(frozen=True)
class AssemblyBody:
    """Body geometry in the root frame, bounded by a centered sphere.

    The caller must verify that radius encloses the collision geometry,
    including all relevant links. Relative poses specify a fixed rigid
    shape, not observed noisy offsets. The first body's pose is
    identity.
    """
    name: str
    pose: Pose
    radius: float

    def __post_init__(self) -> None:
        position, orientation = (np.asarray(v, dtype=float) for v in self.pose)
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("Assembly body needs a name")
        if (position.shape != (3, ) or orientation.shape != (4, )
                or not np.isfinite(position).all()
                or not np.isfinite(orientation).all() or not np.isclose(
                    np.linalg.norm(orientation), 1., rtol=0, atol=1e-12)):
            raise ValueError("Assembly body requires a finite unit pose")
        if not math.isfinite(self.radius) or self.radius <= 0:
            raise ValueError("Enclosing radius must be positive and finite")
        object.__setattr__(self, "pose", (tuple(position), tuple(orientation)))


@dataclass(frozen=True)
class AssemblyState:
    """Correlated body states and consistent original weld frames."""
    poses: Mapping[str, Pose]
    velocities: Mapping[str, Velocity]
    welds: Tuple[CommandWeld, ...]


@dataclass(frozen=True)
class RigidAssemblyPrior:
    """Uniform root placement, Haar orientation, and a declared twist prior.

    Root position is uniform in a cell eroded by the assembly's enclosing
    radius, and orientation is uniform on SO(3). Every enclosed collision
    shape therefore stays inside the supplied obstacle-free cell.
    Enclosing spheres must not overlap, a sufficient but conservative
    condition for internal separation. It excludes assemblies that need
    interpenetrating or merely closely packed enclosing spheres.

    Both twist half-widths zero declare an atom at rest. Positive widths
    declare independent uniform root linear and angular velocities in
    world coordinates. These are alternative component priors, not a
    mixture with an implicit case probability. A complete task prior
    must specify case masses, geometry uncertainty, joints and contacts.

    support_depth optionally declares a horizontal root support face at
    local z=-depth. This component rests on the cell's lower z plane,
    with uniform xy and yaw, zero roll/pitch and zero twist. The caller
    must establish that this is an actual lowest face of the root shape;
    the radius alone does not prove that. Other bodies stay above that
    plane by their enclosing spheres. Static balance is not certified.
    """
    bodies: Tuple[AssemblyBody, ...]
    free_cell: Tuple[Tuple[float, float], ...]
    linear_half_width: float = 0.
    angular_half_width: float = 0.
    weld_force: float = 1000.
    weld_erp: float = .2
    support_depth: Optional[float] = None

    def __post_init__(self) -> None:
        bodies = tuple(self.bodies)
        if (not bodies or len({b.name
                               for b in bodies}) != len(bodies)
                or bodies[0].pose != _IDENTITY):
            raise ValueError(
                "Assembly needs distinct bodies and an identity root")
        cell = tuple((float(lo), float(hi)) for lo, hi in self.free_cell)
        if len(cell) != 3 or any(
                not math.isfinite(lo) or not math.isfinite(hi) or lo >= hi
                for lo, hi in cell):
            raise ValueError("Free cell requires three finite intervals")
        widths = (self.linear_half_width, self.angular_half_width)
        if (any(not math.isfinite(v) or v < 0 for v in widths)
                or ((widths[0] == 0) != (widths[1] == 0))):
            raise ValueError("Twist widths must both be positive or both zero")
        if (not math.isfinite(self.weld_force) or self.weld_force <= 0
                or not math.isfinite(self.weld_erp)
                or not 0 <= self.weld_erp <= 1):
            raise ValueError("Invalid weld settings")
        for i, body in enumerate(bodies):
            for other in bodies[:i]:
                separation = np.linalg.norm(
                    np.asarray(body.pose[0]) - other.pose[0])
                if separation < body.radius + other.radius:
                    raise ValueError("Assembly enclosing spheres overlap")
        object.__setattr__(self, "bodies", bodies)
        object.__setattr__(self, "free_cell", cell)
        if self.support_depth is not None:
            depth = self.support_depth
            if (not math.isfinite(depth) or not 0 < depth <= bodies[0].radius
                    or any(widths)):
                raise ValueError(
                    "Supported case requires a valid face and rest")
            if any(depth + b.pose[0][2] - b.radius < 0 for b in bodies[1:]):
                raise ValueError("Attached body extends below support plane")
            if depth + self.enclosing_radius > cell[2][1] - cell[2][0]:
                raise ValueError("Supported assembly exceeds cell height")
        # BoxPrior also rejects cells too small for any root position.
        _ = self.coordinates

    @property
    def enclosing_radius(self) -> float:
        """Conservative assembly radius about its reference body."""
        return max(
            float(np.linalg.norm(b.pose[0])) + b.radius for b in self.bodies)

    @property
    def coordinates(self) -> BoxPrior:
        """Normalized free-coordinate distribution, before any observations."""
        radius = self.enclosing_radius
        if self.support_depth is not None:
            return BoxPrior(
                ("root_x", "root_y", "yaw_fraction"),
                tuple((lo + radius, hi - radius)
                      for lo, hi in self.free_cell[:2]) + ((0., 1.), ))
        names: Tuple[str, ...] = ("root_x", "root_y", "root_z", "rotation_u",
                                  "rotation_v", "rotation_w")
        bounds = tuple((lo + radius, hi - radius) for lo, hi in self.free_cell)
        bounds += ((0., 1.), ) * 3
        if self.linear_half_width:
            names += ("vx", "vy", "vz", "wx", "wy", "wz")
            bounds += ((-self.linear_half_width, self.linear_half_width), ) * 3
            bounds += (
                (-self.angular_half_width, self.angular_half_width), ) * 3
        return BoxPrior(names, bounds)

    @property
    def digest(self) -> str:
        """Pin geometry, support, motion and the measure on root rotations."""
        return content_digest(
            json.dumps(
                {
                    "schema":
                    1,
                    "family":
                    "free_cell_rigid_assembly",
                    "orientation":
                    "haar_so3_uniform_quaternion"
                    if self.support_depth is None else "uniform_yaw",
                    "prior":
                    asdict(self)
                },
                sort_keys=True).encode())

    def lift(self, coordinates: np.ndarray) -> AssemblyState:
        """Map one free-coordinate point to compatible body poses and twists.

        This is a generative pushforward, with density one relative to
        its normalized BoxPrior coordinates. Do not multiply Cartesian
        densities for every derived body or count its pose twice.
        """
        point = np.asarray(coordinates, dtype=float)
        bounds = np.asarray(self.coordinates.bounds)
        if (point.shape != (len(bounds), ) or not np.isfinite(point).all()
                or np.any(point < bounds[:, 0])
                or np.any(point > bounds[:, 1])):
            raise ValueError("Assembly coordinates outside declared support")
        if self.support_depth is None:
            u, v, w = point[3:6]
            quaternion = (math.sqrt(1 - u) * math.sin(2 * math.pi * v),
                          math.sqrt(1 - u) * math.cos(2 * math.pi * v),
                          math.sqrt(u) * math.sin(2 * math.pi * w),
                          math.sqrt(u) * math.cos(2 * math.pi * w))
            root_position = point[:3]
        else:
            quaternion = (0., 0., math.sin(math.pi * point[2]),
                          math.cos(math.pi * point[2]))
            root_position = np.array([
                point[0], point[1], self.free_cell[2][0] + self.support_depth
            ])
        rotation = Rotation.from_quat(quaternion)
        linear, angular = (point[6:9],
                           point[9:12]) if len(point) == 12 else (np.zeros(3),
                                                                  np.zeros(3))
        poses = {}
        velocities = {}
        welds = []
        for index, body in enumerate(self.bodies):
            offset = rotation.apply(body.pose[0])
            orientation = (rotation *
                           Rotation.from_quat(body.pose[1])).as_quat()
            poses[body.name] = (tuple(root_position + offset),
                                tuple(orientation))
            velocities[body.name] = (tuple(linear + np.cross(angular, offset)),
                                     tuple(angular))
            if index:
                welds.append(
                    CommandWeld(self.bodies[0].name, body.name, body.pose,
                                _IDENTITY, self.weld_force, self.weld_erp))
        return AssemblyState(poses, velocities, tuple(welds))
