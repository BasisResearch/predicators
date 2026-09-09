"""Observable simulation core of the ice rink environment.

This module is the ice rink env's BASE SIM: rink geometry, tile bodies,
walls, targets and the state read/write of everything a camera could
see. It deliberately contains NO material physics, NO patch drag, no
task generation and no predicate / goal semantics.

The boundary is the same visibility contract the busyboard uses: when
``CFG.agent_sim_provide_base_sim_source`` is on, THIS FILE is copied
into the learning agent's sandbox as reference material, so the file
the agent reads is byte-identical to the code its base-sim rollouts
execute. Anything that would leak the learning target - which colour
slides how far, what the dark strip does to a tile crossing it - lives
in ``pybullet_icerink.py``, never here.

Physical layout (a tabletop shuffleboard, seen from the robot's side):

- A flat ``rink`` slab resting on the table. Its surface friction is a
  fixed constant of this file; what varies is the tiles.
- Two low walls, along the far (north) edge and the right (east) edge.
  The near (south) and left (west) edges are open: a tile that slides
  past them leaves the rink.
- A dark ``patch`` strip across the rink, purely decorative in this
  file. The concrete env owns what it does.
- ``tile`` objects: small square blocks the robot pushes. A tile's
  ``color`` is its material class, a stable identity across levels.
- ``target`` markers: flat squares painted on the slab in a tile's
  colour, with no collision shape. Where the goal wants a tile.
- Four ``direction`` objects (north, east, south, west): virtual
  objects that name the push directions; their ``yaw`` is the facing
  angle the shared push skill uses.

A tile pushed along a direction leaves the gripper at the push speed
and slides until friction, a wall or another tile stops it. In this
file every tile slides with the same friction; the material classes
mean nothing until the concrete env says what they mean.
"""
from typing import Any, ClassVar, Dict, List, Tuple

import numpy as np
import pybullet as p

from predicators.envs.pybullet_env import PyBulletEnv
from predicators.pybullet_helpers.geometry import Pose3D, Quaternion
from predicators.pybullet_helpers.objects import create_object, \
    create_pybullet_block, update_object
from predicators.pybullet_helpers.robots import SingleArmPyBulletRobot
from predicators.settings import CFG
from predicators.structs import Object, State, Type


class PyBulletIceRinkBaseEnv(PyBulletEnv):
    """Sim core of the ice rink: a slab, two walls, tiles and targets.

    Abstract on purpose - it defines no name, predicates, tasks or
    domain-specific step, so env discovery skips it; the concrete env is
    ``PyBulletIceRinkEnv``.
    """

    @classmethod
    def get_base_sim_source_files(cls) -> List[str]:
        # This module IS the visible sim core; pybullet_env.py is the
        # generic engine it is built on. pybullet_icerink.py (material
        # friction, patch drag, tasks, predicates) must never be listed.
        return [
            "predicators/envs/pybullet_icerink_base.py",
            "predicators/envs/pybullet_env.py",
        ]

    # =========================================================================
    # WORKSPACE & TABLE
    # =========================================================================
    table_height: ClassVar[float] = 0.4
    table_pos: ClassVar[Pose3D] = (0.75, 1.35, table_height / 2)
    table_orn: ClassVar[Quaternion] = p.getQuaternionFromEuler(
        [0.0, 0.0, np.pi / 2.0])

    x_lb: ClassVar[float] = 0.4
    x_ub: ClassVar[float] = 1.1
    y_lb: ClassVar[float] = 1.0
    y_ub: ClassVar[float] = 1.6
    z_lb: ClassVar[float] = table_height
    z_ub: ClassVar[float] = 0.75 + table_height / 2
    x_mid: ClassVar[float] = (x_lb + x_ub) / 2
    y_mid: ClassVar[float] = (y_lb + y_ub) / 2

    # =========================================================================
    # ROBOT
    # =========================================================================
    robot_init_x: ClassVar[float] = x_mid
    robot_init_y: ClassVar[float] = 1.2
    robot_init_z: ClassVar[float] = z_ub - 0.1
    robot_base_pos: ClassVar[Pose3D] = (0.75, 0.65, 0.0)
    robot_base_orn: ClassVar[Quaternion] = p.getQuaternionFromEuler(
        [0.0, 0.0, np.pi / 2])
    robot_init_tilt: ClassVar[float] = np.pi / 2
    robot_init_wrist: ClassVar[float] = -np.pi / 2

    # =========================================================================
    # CAMERA
    # =========================================================================
    # Framed on the rink from the robot's right shoulder, high enough
    # that the whole slab, both walls and the dark strip are in view and
    # every tile's position against the targets is readable in a still.
    _camera_distance: ClassVar[float] = 1.15
    _camera_yaw: ClassVar[float] = 60
    _camera_pitch: ClassVar[float] = -52
    _camera_target: ClassVar[Tuple[float, float, float]] = (0.75, 1.27, 0.41)

    # =========================================================================
    # RINK LAYOUT
    # =========================================================================
    # The slab: centred in front of the robot, deep enough that a tile
    # pushed north has room to slide and shallow enough that its far
    # edge is still inside the arm's reach.
    rink_center: ClassVar[Tuple[float, float]] = (0.75, 1.25)
    rink_half_extents: ClassVar[Tuple[float, float,
                                      float]] = (0.30, 0.19, 0.004)
    rink_top: ClassVar[float] = table_height + 2 * rink_half_extents[2]
    # Surface friction of the slab. PyBullet combines the two bodies'
    # coefficients multiplicatively, so a slab at 1.0 makes each tile's
    # own coefficient the effective one.
    rink_friction: ClassVar[float] = 1.0
    rink_color: ClassVar[Tuple[float, float, float,
                               float]] = (0.80, 0.90, 0.97, 1.0)

    # Walls along the north (far) and east (right) edges only. The
    # remaining two edges are open by design.
    wall_thickness: ClassVar[float] = 0.012
    wall_height: ClassVar[float] = 0.03
    wall_color: ClassVar[Tuple[float, float, float,
                               float]] = (0.35, 0.30, 0.28, 1.0)

    # The dark strip: a band of the slab at these x bounds, painted as a
    # separate flat body with no collision shape of its own.
    patch_x_bounds: ClassVar[Tuple[float, float]] = (0.86, 0.96)
    patch_color: ClassVar[Tuple[float, float, float,
                                float]] = (0.30, 0.28, 0.32, 1.0)

    # Tiles: square blocks, low enough that a closed gripper pushes them
    # by their side face and heavy enough not to hop.
    tile_half_extents: ClassVar[Tuple[float, float,
                                      float]] = (0.03, 0.03, 0.012)
    tile_mass: ClassVar[float] = 0.1
    # The friction every tile has in this file. Which tiles differ from
    # it, and by how much, is the concrete env's business.
    tile_base_friction: ClassVar[float] = 0.1
    tile_z: ClassVar[float] = rink_top + tile_half_extents[2]
    # Height above a tile's centre at which the push skill aims. The
    # closed fingertips have to strike the tile's side face, not skim
    # over its top.
    tile_push_height: ClassVar[float] = 0.0

    # Targets: painted squares a little wider than a tile.
    target_half_extents: ClassVar[Tuple[float, float,
                                        float]] = (0.036, 0.036, 0.0008)
    target_z: ClassVar[float] = rink_top + target_half_extents[2]

    # The material palette. A tile's ``color`` feature is an index into
    # this list and its target carries the same index; the name is how
    # to refer to the tile ("the black tile"). The palette is the whole
    # set of materials any level can show, so a colour means the same
    # thing on every rink.
    COLOR_PALETTE: ClassVar[List[Tuple[str,
                                       Tuple[float, float, float, float]]]] = [
                                           ("blue", (0.25, 0.55, 0.95, 1.0)),
                                           ("black", (0.10, 0.10, 0.12, 1.0)),
                                           ("green", (0.20, 0.65, 0.30, 1.0)),
                                           ("grey", (0.62, 0.64, 0.68, 1.0)),
                                       ]

    # The four push directions. ``yaw`` is the facing the shared push
    # skill turns into (sin(yaw), cos(yaw)), so north is +y.
    DIRECTIONS: ClassVar[Dict[str, float]] = {
        "north": 0.0,
        "east": np.pi / 2,
        "south": np.pi,
        "west": -np.pi / 2,
    }

    # =========================================================================
    # TYPES
    # =========================================================================
    _robot_type = Type("robot",
                       ["x", "y", "z", "fingers", "roll", "tilt", "wrist"])
    # ``speed`` is the tile's planar speed, so an observation says
    # whether the tile has come to rest. A restored state starts at rest
    # whatever it says: velocity is not something a reset reproduces.
    _tile_type = Type("tile", ["x", "y", "z", "rot", "color", "speed"])
    _target_type = Type("target", ["x", "y", "z", "color"])
    _direction_type = Type("direction", ["yaw"])

    # =========================================================================
    # COLOURS
    # =========================================================================
    @classmethod
    def color_name(cls, color_index: int) -> str:
        """The palette name behind a ``color`` feature value."""
        return cls.COLOR_PALETTE[int(round(color_index))][0]

    @classmethod
    def color_rgba(cls, color_index: int) -> Tuple[float, float, float, float]:
        """The RGBA behind a ``color`` feature value."""
        return cls.COLOR_PALETTE[int(round(color_index))][1]

    # =========================================================================
    # CAPACITY
    # =========================================================================
    @classmethod
    def _max_tiles(cls) -> int:
        return max(
            list(CFG.icerink_num_tiles_train) +
            list(CFG.icerink_num_tiles_test))

    # =========================================================================
    # GEOMETRY HELPERS
    # =========================================================================
    @classmethod
    def rink_bounds(cls) -> Tuple[float, float, float, float]:
        """``(x_min, x_max, y_min, y_max)`` of the slab's top face."""
        cx, cy = cls.rink_center
        hx, hy, _ = cls.rink_half_extents
        return cx - hx, cx + hx, cy - hy, cy + hy

    @classmethod
    def on_rink(cls, x: float, y: float) -> bool:
        """Whether a tile centred at ``(x, y)`` is still on the slab."""
        x_min, x_max, y_min, y_max = cls.rink_bounds()
        hx, hy, _ = cls.tile_half_extents
        return (x_min - hx <= x <= x_max + hx) and \
            (y_min - hy <= y <= y_max + hy)

    @classmethod
    def on_patch(cls, x: float) -> bool:
        """Whether a tile centred at ``x`` overlaps the dark strip."""
        lo, hi = cls.patch_x_bounds
        hx = cls.tile_half_extents[0]
        return lo - hx < x < hi + hx

    # =========================================================================
    # CONSTRUCTION
    # =========================================================================
    def __init__(self, use_gui: bool = False, **kwargs: Any) -> None:
        self._robot = Object("robot", self._robot_type)
        self._tiles: List[Object] = [
            Object(f"tile{i}", self._tile_type)
            for i in range(self._max_tiles())
        ]
        self._targets: List[Object] = [
            Object(f"target{i}", self._target_type)
            for i in range(self._max_tiles())
        ]
        self._directions: List[Object] = [
            Object(name, self._direction_type) for name in self.DIRECTIONS
        ]
        # Material per tile and per target, by object name, from the
        # last state written.
        self._tile_colors: Dict[str, int] = {}
        self._target_colors: Dict[str, int] = {}
        super().__init__(use_gui, **kwargs)

    @classmethod
    def initialize_pybullet(
            cls, using_gui: bool
    ) -> Tuple[int, SingleArmPyBulletRobot, Dict[str, Any]]:
        physics_client_id, pybullet_robot, bodies = super(
        ).initialize_pybullet(using_gui)

        table_id = create_object(asset_path="urdf/table.urdf",
                                 position=cls.table_pos,
                                 orientation=cls.table_orn,
                                 scale=1.0,
                                 use_fixed_base=True,
                                 physics_client_id=physics_client_id)
        bodies["table_id"] = table_id

        cx, cy = cls.rink_center
        hx, hy, hz = cls.rink_half_extents
        rink_id = create_pybullet_block(color=cls.rink_color,
                                        half_extents=cls.rink_half_extents,
                                        mass=0.0,
                                        friction=cls.rink_friction,
                                        position=(cx, cy,
                                                  cls.table_height + hz),
                                        physics_client_id=physics_client_id)
        p.changeDynamics(rink_id,
                         -1,
                         restitution=0.0,
                         physicsClientId=physics_client_id)
        bodies["rink_id"] = rink_id

        # The dark strip: a hair above the slab, no collision shape.
        lo, hi = cls.patch_x_bounds
        strip_visual = p.createVisualShape(p.GEOM_BOX,
                                           halfExtents=((hi - lo) / 2, hy,
                                                        0.0005),
                                           rgbaColor=cls.patch_color,
                                           physicsClientId=physics_client_id)
        bodies["patch_id"] = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=strip_visual,
            basePosition=((lo + hi) / 2, cy, cls.rink_top + 0.0005),
            physicsClientId=physics_client_id)

        x_min, x_max, y_min, y_max = cls.rink_bounds()
        wall_z = cls.rink_top + cls.wall_height / 2
        t = cls.wall_thickness
        north_id = create_pybullet_block(color=cls.wall_color,
                                         half_extents=(hx + t, t / 2,
                                                       cls.wall_height / 2),
                                         mass=0.0,
                                         friction=cls.rink_friction,
                                         position=(cx, y_max + t / 2, wall_z),
                                         physics_client_id=physics_client_id)
        east_id = create_pybullet_block(color=cls.wall_color,
                                        half_extents=(t / 2, hy + t,
                                                      cls.wall_height / 2),
                                        mass=0.0,
                                        friction=cls.rink_friction,
                                        position=(x_max + t / 2, cy, wall_z),
                                        physics_client_id=physics_client_id)
        for wall_id in (north_id, east_id):
            p.changeDynamics(wall_id,
                             -1,
                             restitution=0.0,
                             physicsClientId=physics_client_id)
        bodies["wall_ids"] = [north_id, east_id]
        del x_min, y_min

        tile_ids = []
        for _ in range(cls._max_tiles()):
            tile_id = create_pybullet_block(
                color=cls.COLOR_PALETTE[0][1],
                half_extents=cls.tile_half_extents,
                mass=cls.tile_mass,
                friction=cls.tile_base_friction,
                physics_client_id=physics_client_id)
            p.changeDynamics(tile_id,
                             -1,
                             restitution=0.0,
                             physicsClientId=physics_client_id)
            tile_ids.append(tile_id)
        bodies["tile_ids"] = tile_ids

        target_ids = []
        for _ in range(cls._max_tiles()):
            visual = p.createVisualShape(p.GEOM_BOX,
                                         halfExtents=cls.target_half_extents,
                                         rgbaColor=cls.COLOR_PALETTE[0][1],
                                         physicsClientId=physics_client_id)
            target_ids.append(
                p.createMultiBody(baseMass=0.0,
                                  baseCollisionShapeIndex=-1,
                                  baseVisualShapeIndex=visual,
                                  physicsClientId=physics_client_id))
        bodies["target_ids"] = target_ids

        return physics_client_id, pybullet_robot, bodies

    def _store_pybullet_bodies(self, pybullet_bodies: Dict[str, Any]) -> None:
        self._table_id = pybullet_bodies["table_id"]
        self._rink_id = pybullet_bodies["rink_id"]
        self._patch_id = pybullet_bodies["patch_id"]
        self._wall_ids: List[int] = pybullet_bodies["wall_ids"]
        self._robot.id = self._pybullet_robot.robot_id
        for i, tile in enumerate(self._tiles):
            tile.id = pybullet_bodies["tile_ids"][i]
        for i, target in enumerate(self._targets):
            target.id = pybullet_bodies["target_ids"][i]

    # =========================================================================
    # TILE MECHANICS
    # =========================================================================
    def _tile_speed(self, tile: Object) -> float:
        """The tile's planar speed, from the engine."""
        if tile.id is None:
            return 0.0
        (vx, vy,
         _), _ = p.getBaseVelocity(tile.id,
                                   physicsClientId=self._physics_client_id)
        return float(np.hypot(vx, vy))

    def _set_tile_friction(self, tile: Object, friction: float) -> None:
        """Set a tile body's lateral friction coefficient."""
        if tile.id is None:
            return
        p.changeDynamics(tile.id,
                         -1,
                         lateralFriction=float(friction),
                         physicsClientId=self._physics_client_id)

    def _paint(self, body_id: int, color_index: int) -> None:
        p.changeVisualShape(body_id,
                            -1,
                            rgbaColor=self.color_rgba(color_index),
                            physicsClientId=self._physics_client_id)

    # =========================================================================
    # STATE READ / WRITE
    # =========================================================================
    def _get_object_ids_for_held_check(self) -> List[int]:
        """Nothing on the rink is graspable; tiles are pushed, not held."""
        return []

    def _park_unused_bodies(self, num_tiles: int) -> None:
        """Move bodies beyond the task's count out of the camera's view."""
        oov_x, oov_y = self._out_of_view_xy
        for i in range(num_tiles, len(self._tiles)):
            update_object(self._tiles[i].id,
                          position=(oov_x, oov_y, self.tile_z),
                          physics_client_id=self._physics_client_id)
            update_object(self._targets[i].id,
                          position=(oov_x, oov_y + 1.0, self.target_z),
                          physics_client_id=self._physics_client_id)

    def _active_objects(self,
                        state: State) -> Tuple[List[Object], List[Object]]:
        """The tiles and targets present in ``state``, in index order."""
        tiles = sorted((o for o in state if o.type.name == "tile"),
                       key=lambda o: o.name)
        targets = sorted((o for o in state if o.type.name == "target"),
                         key=lambda o: o.name)
        return tiles, targets

    def _get_domain_specific_feature(self, obj: Object, feature: str) -> float:
        if obj.type.name == "direction" and feature == "yaw":
            return float(self.DIRECTIONS[obj.name])
        if obj.type.name == "tile":
            if feature == "color":
                return float(self._tile_colors.get(obj.name, 0))
            if feature == "speed":
                return self._tile_speed(obj)
        if obj.type.name == "target" and feature == "color":
            return float(self._target_colors.get(obj.name, 0))
        raise ValueError(f"Unknown feature {feature} for object {obj}")

    def _set_domain_specific_state(self, state: State) -> None:
        """Paint tiles and targets in their material colour and park the bodies
        this level does not use."""
        tiles, targets = self._active_objects(state)
        self._tile_colors = {}
        self._target_colors = {}
        for tile in tiles:
            color = int(round(state.get(tile, "color")))
            self._tile_colors[tile.name] = color
            self._paint(tile.id, color)
        for target in targets:
            color = int(round(state.get(target, "color")))
            self._target_colors[target.name] = color
            self._paint(target.id, color)
            update_object(target.id,
                          position=(state.get(target, "x"),
                                    state.get(target, "y"), self.target_z),
                          physics_client_id=self._physics_client_id)
        self._park_unused_bodies(len(tiles))
