"""Observable simulation core of the magnets environment.

This module is the magnets env's BASE SIM: the mat on the table, the
wand in the robot's hand, the steel pieces, the slots, and the state
read/write of everything visible. It deliberately contains NO field
law, no polarity or range per colour, no task generation and no
predicate / goal semantics. On the base sim the wand hovers and the
pieces never move: a wand that does nothing.

The boundary is the busyboard's visibility contract: when
``CFG.agent_sim_provide_base_sim_source`` is on, THIS FILE is copied
into the learning agent's sandbox as reference material, so the file
the agent reads is byte-identical to the code its base-sim rollouts
execute. Anything that would leak the learning target - which colour
the wand attracts, which it repels, from how far - lives in
``pybullet_magnets.py``, never here.

Physical layout (a tabletop puzzle, the robot at the near side):

- A ``mat``: a flat sheet on the table whose edges are the play area.
  A piece that slides off it has left the puzzle.
- The ``wand``: a thin rod the robot holds by its top end, hanging
  straight down. Its ``tip`` is the rod's lower end; the skills carry
  it at ``hover_z``, just above the pieces, and never touch anything
  with it.
- ``piece`` objects: small cubes on the mat. A piece's ``color`` is its
  material class, a stable identity across levels.
- ``slot`` markers: squares painted on the mat in a piece's colour,
  with no collision shape. Where the goal wants a piece.
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


class PyBulletMagnetsBaseEnv(PyBulletEnv):
    """Sim core of the magnets puzzle: a mat, a held wand, pieces and slots.

    Abstract on purpose - it defines no name, predicates, tasks or
    domain-specific step, so env discovery skips it; the concrete env is
    ``PyBulletMagnetsEnv``.
    """

    @classmethod
    def get_base_sim_source_files(cls) -> List[str]:
        return [
            "predicators/envs/pybullet_magnets_base.py",
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
    y_ub: ClassVar[float] = 1.5
    z_lb: ClassVar[float] = table_height
    z_ub: ClassVar[float] = 0.75 + table_height / 2
    x_mid: ClassVar[float] = (x_lb + x_ub) / 2
    y_mid: ClassVar[float] = (y_lb + y_ub) / 2

    # =========================================================================
    # ROBOT
    # =========================================================================
    # The robot starts holding the wand at hover height over the mat's
    # near-left corner, out of every piece's way.
    hover_z: ClassVar[float] = table_height + 0.075
    wand_length: ClassVar[float] = 0.12
    robot_init_x: ClassVar[float] = 0.5
    robot_init_y: ClassVar[float] = 1.06
    robot_init_z: ClassVar[float] = hover_z + wand_length
    robot_base_pos: ClassVar[Pose3D] = (0.75, 0.65, 0.0)
    robot_base_orn: ClassVar[Quaternion] = p.getQuaternionFromEuler(
        [0.0, 0.0, np.pi / 2])
    robot_init_tilt: ClassVar[float] = np.pi / 2
    robot_init_wrist: ClassVar[float] = -np.pi / 2

    # =========================================================================
    # CAMERA
    # =========================================================================
    _camera_distance: ClassVar[float] = 1.1
    _camera_yaw: ClassVar[float] = 45
    _camera_pitch: ClassVar[float] = -55
    _camera_target: ClassVar[Tuple[float, float, float]] = (0.75, 1.22, 0.42)

    # =========================================================================
    # MAT, WAND, PIECES, SLOTS
    # =========================================================================
    mat_center: ClassVar[Tuple[float, float]] = (0.75, 1.22)
    mat_half_extents: ClassVar[Tuple[float, float,
                                     float]] = (0.30, 0.17, 0.002)
    mat_top: ClassVar[float] = table_height + 2 * mat_half_extents[2]
    mat_friction: ClassVar[float] = 1.0
    mat_color: ClassVar[Tuple[float, float, float,
                              float]] = (0.93, 0.90, 0.82, 1.0)

    wand_radius: ClassVar[float] = 0.008
    wand_mass: ClassVar[float] = 0.05
    wand_color: ClassVar[Tuple[float, float, float,
                               float]] = (0.25, 0.25, 0.28, 1.0)
    tip_color: ClassVar[Tuple[float, float, float,
                              float]] = (0.85, 0.15, 0.15, 1.0)

    piece_half: ClassVar[float] = 0.012
    piece_mass: ClassVar[float] = 0.02
    piece_friction: ClassVar[float] = 0.3
    piece_z: ClassVar[float] = mat_top + piece_half

    slot_half_extents: ClassVar[Tuple[float, float,
                                      float]] = (0.02, 0.02, 0.0006)
    slot_z: ClassVar[float] = mat_top + slot_half_extents[2]

    # Material palette: a piece's ``color`` indexes it, and so does its
    # slot's. Which colours the wand pulls, which it pushes and from how
    # far is the concrete env's business.
    COLOR_PALETTE: ClassVar[List[Tuple[str,
                                       Tuple[float, float, float, float]]]] = [
                                           ("red", (0.85, 0.15, 0.15, 1.0)),
                                           ("blue", (0.15, 0.35, 0.85, 1.0)),
                                           ("green", (0.15, 0.65, 0.25, 1.0)),
                                           ("yellow", (0.95, 0.85, 0.15, 1.0)),
                                       ]

    # =========================================================================
    # TYPES
    # =========================================================================
    _robot_type = Type("robot",
                       ["x", "y", "z", "fingers", "roll", "tilt", "wrist"])
    _wand_type = Type("wand", ["x", "y", "z", "is_held"])
    _piece_type = Type("piece", ["x", "y", "z", "rot", "color", "speed"])
    _slot_type = Type("slot", ["x", "y", "z", "color"])

    @classmethod
    def color_name(cls, color_index: int) -> str:
        """The palette name behind a ``color`` feature value."""
        return cls.COLOR_PALETTE[int(round(color_index))][0]

    @classmethod
    def color_rgba(cls, color_index: int) -> Tuple[float, float, float, float]:
        """The RGBA behind a ``color`` feature value."""
        return cls.COLOR_PALETTE[int(round(color_index))][1]

    @classmethod
    def _max_pieces(cls) -> int:
        return max(
            list(CFG.magnets_num_pieces_train) +
            list(CFG.magnets_num_pieces_test))

    @classmethod
    def mat_bounds(cls) -> Tuple[float, float, float, float]:
        """``(x_min, x_max, y_min, y_max)`` of the mat."""
        cx, cy = cls.mat_center
        hx, hy, _ = cls.mat_half_extents
        return cx - hx, cx + hx, cy - hy, cy + hy

    @classmethod
    def on_mat(cls, x: float, y: float) -> bool:
        """Whether a piece centred at ``(x, y)`` is still on the mat."""
        x_min, x_max, y_min, y_max = cls.mat_bounds()
        return x_min <= x <= x_max and y_min <= y <= y_max

    @classmethod
    def tip_position(cls, state: State,
                     wand: Object) -> Tuple[float, float, float]:
        """The wand's lower end: the rod hangs from its centre."""
        return (state.get(wand, "x"), state.get(wand, "y"),
                state.get(wand, "z") - cls.wand_length / 2)

    # =========================================================================
    # CONSTRUCTION
    # =========================================================================
    def __init__(self, use_gui: bool = False, **kwargs: Any) -> None:
        self._robot = Object("robot", self._robot_type)
        self._wand = Object("wand", self._wand_type)
        self._pieces: List[Object] = [
            Object(f"piece{i}", self._piece_type)
            for i in range(self._max_pieces())
        ]
        self._slots: List[Object] = [
            Object(f"slot{i}", self._slot_type)
            for i in range(self._max_pieces())
        ]
        self._piece_colors: Dict[str, int] = {}
        self._slot_colors: Dict[str, int] = {}
        super().__init__(use_gui, **kwargs)

    @classmethod
    def initialize_pybullet(
            cls, using_gui: bool
    ) -> Tuple[int, SingleArmPyBulletRobot, Dict[str, Any]]:
        physics_client_id, pybullet_robot, bodies = super(
        ).initialize_pybullet(using_gui)

        bodies["table_id"] = create_object(asset_path="urdf/table.urdf",
                                           position=cls.table_pos,
                                           orientation=cls.table_orn,
                                           scale=1.0,
                                           use_fixed_base=True,
                                           physics_client_id=physics_client_id)
        cx, cy = cls.mat_center
        bodies["mat_id"] = create_pybullet_block(
            color=cls.mat_color,
            half_extents=cls.mat_half_extents,
            mass=0.0,
            friction=cls.mat_friction,
            position=(cx, cy, cls.table_height + cls.mat_half_extents[2]),
            physics_client_id=physics_client_id)

        # The wand: a rod with a red tip, its origin at its centre.
        rod_collision = p.createCollisionShape(
            p.GEOM_CYLINDER,
            radius=cls.wand_radius,
            height=cls.wand_length,
            physicsClientId=physics_client_id)
        rod_visual = p.createVisualShape(p.GEOM_CYLINDER,
                                         radius=cls.wand_radius,
                                         length=cls.wand_length,
                                         rgbaColor=cls.wand_color,
                                         physicsClientId=physics_client_id)
        tip_visual = p.createVisualShape(p.GEOM_SPHERE,
                                         radius=cls.wand_radius * 1.6,
                                         rgbaColor=cls.tip_color,
                                         physicsClientId=physics_client_id)
        wand_id = p.createMultiBody(
            baseMass=cls.wand_mass,
            baseCollisionShapeIndex=rod_collision,
            baseVisualShapeIndex=rod_visual,
            basePosition=(cls.robot_init_x, cls.robot_init_y,
                          cls.robot_init_z - cls.wand_length / 2),
            linkMasses=[0.0],
            linkCollisionShapeIndices=[-1],
            linkVisualShapeIndices=[tip_visual],
            linkPositions=[[0.0, 0.0, -cls.wand_length / 2]],
            linkOrientations=[[0.0, 0.0, 0.0, 1.0]],
            linkInertialFramePositions=[[0.0, 0.0, 0.0]],
            linkInertialFrameOrientations=[[0.0, 0.0, 0.0, 1.0]],
            linkParentIndices=[0],
            linkJointTypes=[p.JOINT_FIXED],
            linkJointAxis=[[0.0, 0.0, 1.0]],
            physicsClientId=physics_client_id)
        p.changeDynamics(wand_id,
                         -1,
                         lateralFriction=1.0,
                         physicsClientId=physics_client_id)
        bodies["wand_id"] = wand_id

        piece_ids = []
        for _ in range(cls._max_pieces()):
            piece_id = create_pybullet_block(
                color=cls.COLOR_PALETTE[0][1],
                half_extents=(cls.piece_half, cls.piece_half, cls.piece_half),
                mass=cls.piece_mass,
                friction=cls.piece_friction,
                physics_client_id=physics_client_id)
            piece_ids.append(piece_id)
        bodies["piece_ids"] = piece_ids

        slot_ids = []
        for _ in range(cls._max_pieces()):
            visual = p.createVisualShape(p.GEOM_BOX,
                                         halfExtents=cls.slot_half_extents,
                                         rgbaColor=cls.COLOR_PALETTE[0][1],
                                         physicsClientId=physics_client_id)
            slot_ids.append(
                p.createMultiBody(baseMass=0.0,
                                  baseCollisionShapeIndex=-1,
                                  baseVisualShapeIndex=visual,
                                  physicsClientId=physics_client_id))
        bodies["slot_ids"] = slot_ids
        return physics_client_id, pybullet_robot, bodies

    def _store_pybullet_bodies(self, pybullet_bodies: Dict[str, Any]) -> None:
        self._table_id = pybullet_bodies["table_id"]
        self._mat_id = pybullet_bodies["mat_id"]
        self._robot.id = self._pybullet_robot.robot_id
        self._wand.id = pybullet_bodies["wand_id"]
        for i, piece in enumerate(self._pieces):
            piece.id = pybullet_bodies["piece_ids"][i]
        for i, slot in enumerate(self._slots):
            slot.id = pybullet_bodies["slot_ids"][i]

    # =========================================================================
    # STATE READ / WRITE
    # =========================================================================
    def _get_object_ids_for_held_check(self) -> List[int]:
        """The wand is the one thing the robot holds."""
        return [self._wand.id]

    def _piece_speed(self, piece: Object) -> float:
        (vx, vy,
         _), _ = p.getBaseVelocity(piece.id,
                                   physicsClientId=self._physics_client_id)
        return float(np.hypot(vx, vy))

    def _active_objects(self,
                        state: State) -> Tuple[List[Object], List[Object]]:
        pieces = sorted((o for o in state if o.type.name == "piece"),
                        key=lambda o: o.name)
        slots = sorted((o for o in state if o.type.name == "slot"),
                       key=lambda o: o.name)
        return pieces, slots

    def _paint(self, body_id: int, color_index: int) -> None:
        p.changeVisualShape(body_id,
                            -1,
                            rgbaColor=self.color_rgba(color_index),
                            physicsClientId=self._physics_client_id)

    def _get_domain_specific_feature(self, obj: Object, feature: str) -> float:
        if obj.type.name == "piece":
            if feature == "color":
                return float(self._piece_colors.get(obj.name, 0))
            if feature == "speed":
                return self._piece_speed(obj)
        if obj.type.name == "slot" and feature == "color":
            return float(self._slot_colors.get(obj.name, 0))
        raise ValueError(f"Unknown feature {feature} for object {obj}")

    def _set_domain_specific_state(self, state: State) -> None:
        pieces, slots = self._active_objects(state)
        self._piece_colors = {}
        self._slot_colors = {}
        for piece in pieces:
            color = int(round(state.get(piece, "color")))
            self._piece_colors[piece.name] = color
            self._paint(piece.id, color)
        for slot in slots:
            color = int(round(state.get(slot, "color")))
            self._slot_colors[slot.name] = color
            self._paint(slot.id, color)
            update_object(slot.id,
                          position=(state.get(slot, "x"), state.get(slot, "y"),
                                    self.slot_z),
                          physics_client_id=self._physics_client_id)
        oov_x, oov_y = self._out_of_view_xy
        for i in range(len(pieces), len(self._pieces)):
            update_object(self._pieces[i].id,
                          position=(oov_x, oov_y, 0.1 * i),
                          physics_client_id=self._physics_client_id)
            update_object(self._slots[i].id,
                          position=(oov_x, oov_y + 1.0, self.slot_z),
                          physics_client_id=self._physics_client_id)
