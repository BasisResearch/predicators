"""Observable simulation core of the crane environment.

This module is the crane env's BASE SIM: the table, a crane whose arm
carries a steel ram hinged at the anchor, a crate on the table in the
ram's lane, a bin pad further along it, and the state read/write of
everything visible. It deliberately contains NO material table (how
heavy a crate colour is, how it grips the table), no swing damping
beyond a token engine default, no task generation and no predicate /
goal semantics. On the base sim a drawn-back ram swings, strikes the
crate, and the crate slides as a generic crate would.

The boundary is the busyboard's visibility contract: when
``CFG.agent_sim_provide_base_sim_source`` is on, THIS FILE is copied
into the learning agent's sandbox as reference material, so the file
the agent reads is byte-identical to the code its base-sim rollouts
execute. Anything that would leak the learning target - the crate
masses and frictions behind the colours, the hinge's drag on the swing
- lives in ``pybullet_crane.py``, never here.

Physical layout (a tabletop, the robot at the near side, the lane
running left to right across the table):

- The ``crane``: a mast at the table's back-left corner, a jib out to
  the anchor above the lane, and a rigid arm of ``length`` hinged at
  the anchor, swinging in the lane's vertical plane. Its ``x``, ``y``,
  ``z`` are the anchor's.
- The ``ram``: the steel head at the end of the arm, hanging just
  above the table at the left end of the lane. Its ``x``, ``y``, ``z``
  are where it hangs at rest; ``angle`` is the arm's swing from the
  vertical, positive when the head is drawn back (away from the
  crate); ``speed`` is the head's. Drawn back and let go, it swings
  along the lane.
- The ``crate``: a cube standing in the lane a little right of the
  ram. Its ``color`` is its material class.
- The ``bin``: a translucent pad on the table further right along the
  lane, ``half`` wide along it; the crate is meant to come to rest on
  it. The table ends a little past the bin.
"""
from typing import Any, ClassVar, Dict, List, Optional, Tuple

import numpy as np
import pybullet as p

from predicators.envs.pybullet_env import PyBulletEnv
from predicators.pybullet_helpers.geometry import Pose3D, Quaternion
from predicators.pybullet_helpers.objects import create_object, \
    create_pybullet_block
from predicators.pybullet_helpers.robots import SingleArmPyBulletRobot
from predicators.settings import CFG
from predicators.structs import Object, State, Type


class PyBulletCraneBaseEnv(PyBulletEnv):
    """Sim core of the crane puzzle: a hinged ram, a crate, a bin.

    Abstract on purpose - it defines no name, predicates, tasks or
    domain-specific step, so env discovery skips it; the concrete env is
    ``PyBulletCraneEnv``.
    """

    @classmethod
    def get_base_sim_source_files(cls) -> List[str]:
        return [
            "predicators/envs/pybullet_crane_base.py",
            "predicators/envs/pybullet_env.py",
        ]

    # =========================================================================
    # WORKSPACE & TABLE
    # =========================================================================
    table_height: ClassVar[float] = 0.4
    table_pos: ClassVar[Pose3D] = (0.75, 1.35, table_height / 2)
    table_orn: ClassVar[Quaternion] = p.getQuaternionFromEuler(
        [0.0, 0.0, np.pi / 2.0])
    # The table's top spans these extents.
    table_x_lb: ClassVar[float] = 0.3
    table_x_ub: ClassVar[float] = 1.2
    table_y_lb: ClassVar[float] = 1.1
    table_y_ub: ClassVar[float] = 1.6

    x_lb: ClassVar[float] = 0.3
    x_ub: ClassVar[float] = 1.2
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
    robot_init_y: ClassVar[float] = 1.1
    robot_init_z: ClassVar[float] = z_ub - 0.1
    robot_base_pos: ClassVar[Pose3D] = (0.75, 0.65, 0.0)
    robot_base_orn: ClassVar[Quaternion] = p.getQuaternionFromEuler(
        [0.0, 0.0, np.pi / 2])
    robot_init_tilt: ClassVar[float] = np.pi / 2
    robot_init_wrist: ClassVar[float] = -np.pi / 2

    # =========================================================================
    # CAMERA
    # =========================================================================
    _camera_distance: ClassVar[float] = 1.75
    _camera_yaw: ClassVar[float] = 140
    _camera_pitch: ClassVar[float] = -22
    _camera_target: ClassVar[Tuple[float, float, float]] = (0.75, 1.3, 0.62)

    # =========================================================================
    # LAYOUT
    # =========================================================================
    # The ram: a steel cube on a rigid arm hinged at the anchor, hanging
    # just above the table at the left end of the lane.
    ram_x: ClassVar[float] = 0.6
    head_half: ClassVar[float] = 0.04
    head_mass: ClassVar[float] = 0.5
    ram_rest_z: ClassVar[float] = table_height + 0.045
    head_color: ClassVar[Tuple[float, float, float,
                               float]] = (0.28, 0.28, 0.32, 1.0)
    arm_radius: ClassVar[float] = 0.006
    arm_color: ClassVar[Tuple[float, float, float,
                              float]] = (0.15, 0.15, 0.15, 1.0)
    # The hinge's own drag in this file.
    hinge_base_damping: ClassVar[float] = 0.002

    # The crane's mast stands at the table's back-left corner; the jib
    # runs from its top to the anchor.
    mast_xy: ClassVar[Tuple[float, float]] = (0.36, 1.56)
    mast_half: ClassVar[float] = 0.02
    jib_half: ClassVar[float] = 0.015
    crane_color: ClassVar[Tuple[float, float, float,
                                float]] = (0.85, 0.65, 0.15, 1.0)
    anchor_radius: ClassVar[float] = 0.02

    # The crate: a cube in the lane. Every crate has this mass and
    # grip in this file.
    crate_half: ClassVar[float] = 0.05
    crate_base_mass: ClassVar[float] = 0.15
    crate_base_friction: ClassVar[float] = 0.2
    crate_z: ClassVar[float] = table_height + crate_half

    # The bin: a translucent pad on the table, ``half`` along the lane
    # (a feature), this much across it.
    bin_half_across: ClassVar[float] = 0.08
    bin_color: ClassVar[Tuple[float, float, float,
                              float]] = (0.30, 0.80, 0.40, 0.45)

    # Material palette: the crate's ``color`` indexes it.
    COLOR_PALETTE: ClassVar[List[Tuple[str,
                                       Tuple[float, float, float, float]]]] = [
                                           ("foam", (0.95, 0.85, 0.45, 1.0)),
                                           ("iron", (0.35, 0.35, 0.40, 1.0)),
                                           ("stone", (0.55, 0.45, 0.35, 1.0)),
                                       ]

    # =========================================================================
    # TYPES
    # =========================================================================
    _robot_type = Type("robot",
                       ["x", "y", "z", "fingers", "roll", "tilt", "wrist"])
    _ram_type = Type("ram", ["x", "y", "z", "angle", "speed"])
    _crane_type = Type("crane", ["x", "y", "z", "length"])
    _crate_type = Type(
        "crate", ["x", "y", "z", "roll", "pitch", "yaw", "color", "speed"])
    _bin_type = Type("bin", ["x", "y", "half"])

    @classmethod
    def color_name(cls, color_index: int) -> str:
        """The palette name behind a ``color`` feature value."""
        return cls.COLOR_PALETTE[int(round(color_index))][0]

    @classmethod
    def color_rgba(cls, color_index: int) -> Tuple[float, float, float, float]:
        """The palette colour behind a ``color`` feature value."""
        return cls.COLOR_PALETTE[int(round(color_index))][1]

    @classmethod
    def head_position_at(cls, state: State, ram: Object,
                         crane: Object) -> Tuple[float, float, float]:
        """Where the head is, from the arm's angle: on the arc of the crane's
        length about the anchor."""
        length = state.get(crane, "length")
        angle = state.get(ram, "angle")
        return (state.get(ram, "x") - length * float(np.sin(angle)),
                state.get(ram, "y"),
                state.get(ram, "z") + length * (1.0 - float(np.cos(angle))))

    # =========================================================================
    # CONSTRUCTION
    # =========================================================================
    def __init__(self, use_gui: bool = False, **kwargs: Any) -> None:
        self._robot = Object("robot", self._robot_type)
        self._ram = Object("ram", self._ram_type)
        self._crane = Object("crane", self._crane_type)
        self._crate = Object("crate", self._crate_type)
        self._bin = Object("bin", self._bin_type)
        self._crate_color: int = 0
        self._built_length: Optional[float] = None
        self._built_lane: Optional[Tuple[float, float]] = None
        self._gantry_ids: List[int] = []
        self._bin_half: float = 0.06
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
        bodies["crate_id"] = create_pybullet_block(
            color=cls.COLOR_PALETTE[0][1],
            half_extents=(cls.crate_half, cls.crate_half, cls.crate_half),
            mass=cls.crate_base_mass,
            friction=cls.crate_base_friction,
            physics_client_id=physics_client_id)
        anchor_visual = p.createVisualShape(p.GEOM_SPHERE,
                                            radius=cls.anchor_radius,
                                            rgbaColor=cls.crane_color,
                                            physicsClientId=physics_client_id)
        bodies["anchor_id"] = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=anchor_visual,
            basePosition=(cls.ram_x, cls.y_mid, cls.ram_rest_z + 0.5),
            physicsClientId=physics_client_id)
        bodies["ram_id"] = cls._build_ram(0.5, cls.y_mid, physics_client_id)
        bin_visual = p.createVisualShape(p.GEOM_BOX,
                                         halfExtents=(0.06,
                                                      cls.bin_half_across,
                                                      0.003),
                                         rgbaColor=cls.bin_color,
                                         physicsClientId=physics_client_id)
        bodies["bin_id"] = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=bin_visual,
            basePosition=(1.0, cls.y_mid, cls.table_height + 0.003),
            physicsClientId=physics_client_id)
        return physics_client_id, pybullet_robot, bodies

    @classmethod
    def _build_ram(cls, length: float, lane_y: float,
                   physics_client_id: int) -> int:
        """The ram: a fixed base at the head's rest position, one revolute link
        hinged ``length`` above it (the anchor) about the lane's cross axis,
        carrying the arm and the head."""
        head_collision = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=(cls.head_half, cls.head_half, cls.head_half),
            collisionFramePosition=(0.0, 0.0, -length),
            physicsClientId=physics_client_id)
        link_visual = p.createVisualShapeArray(
            shapeTypes=[p.GEOM_CYLINDER, p.GEOM_BOX],
            radii=[cls.arm_radius, 0.0],
            lengths=[length, 0.0],
            halfExtents=[[0.0, 0.0, 0.0],
                         [cls.head_half, cls.head_half, cls.head_half]],
            rgbaColors=[cls.arm_color, cls.head_color],
            visualFramePositions=[[0.0, 0.0, -length / 2], [0.0, 0.0,
                                                            -length]],
            physicsClientId=physics_client_id)
        ram_id = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=-1,
            basePosition=(cls.ram_x, lane_y, cls.ram_rest_z),
            linkMasses=[cls.head_mass],
            linkCollisionShapeIndices=[head_collision],
            linkVisualShapeIndices=[link_visual],
            linkPositions=[(0.0, 0.0, length)],
            linkOrientations=[(0.0, 0.0, 0.0, 1.0)],
            linkInertialFramePositions=[(0.0, 0.0, -length)],
            linkInertialFrameOrientations=[(0.0, 0.0, 0.0, 1.0)],
            linkParentIndices=[0],
            linkJointTypes=[p.JOINT_REVOLUTE],
            linkJointAxis=[(0.0, 1.0, 0.0)],
            physicsClientId=physics_client_id)
        p.changeDynamics(ram_id,
                         0,
                         lateralFriction=0.6,
                         jointDamping=cls.hinge_base_damping,
                         physicsClientId=physics_client_id)
        # A free hinge: no motor holding it.
        p.setJointMotorControl2(ram_id,
                                0,
                                p.VELOCITY_CONTROL,
                                force=0.0,
                                physicsClientId=physics_client_id)
        return ram_id

    def _store_pybullet_bodies(self, pybullet_bodies: Dict[str, Any]) -> None:
        self._table_id = pybullet_bodies["table_id"]
        self._robot.id = self._pybullet_robot.robot_id
        self._ram.id = pybullet_bodies["ram_id"]
        self._crane.id = pybullet_bodies["anchor_id"]
        self._crate.id = pybullet_bodies["crate_id"]
        self._bin.id = pybullet_bodies["bin_id"]

    # =========================================================================
    # THE CRANE
    # =========================================================================
    def _build_crane(self, lane_y: float, length: float) -> None:
        """Rebuild the ram on an arm of ``length`` over lane ``lane_y``, and
        the mast and jib that hold it."""
        cid = self._physics_client_id
        for body_id in self._gantry_ids:
            p.removeBody(body_id, physicsClientId=cid)
        self._gantry_ids = []
        p.removeBody(self._ram.id, physicsClientId=cid)
        self._ram.id = self._build_ram(length, lane_y, cid)
        self._built_length = length
        anchor_z = self.ram_rest_z + length
        anchor = (self.ram_x, lane_y, anchor_z)
        p.resetBasePositionAndOrientation(self._crane.id,
                                          anchor, (0.0, 0.0, 0.0, 1.0),
                                          physicsClientId=cid)
        # The mast and the jib: pictures, the arm never goes there.
        mast_x, mast_y = self.mast_xy
        mast_h = anchor_z - self.table_height
        mast_visual = p.createVisualShape(p.GEOM_BOX,
                                          halfExtents=(self.mast_half,
                                                       self.mast_half,
                                                       mast_h / 2),
                                          rgbaColor=self.crane_color,
                                          physicsClientId=cid)
        self._gantry_ids.append(
            p.createMultiBody(baseMass=0.0,
                              baseCollisionShapeIndex=-1,
                              baseVisualShapeIndex=mast_visual,
                              basePosition=(mast_x, mast_y,
                                            self.table_height + mast_h / 2),
                              physicsClientId=cid))
        dx, dy = anchor[0] - mast_x, anchor[1] - mast_y
        jib_len = float(np.hypot(dx, dy)) + self.mast_half
        jib_visual = p.createVisualShape(p.GEOM_BOX,
                                         halfExtents=(jib_len / 2,
                                                      self.jib_half,
                                                      self.jib_half),
                                         rgbaColor=self.crane_color,
                                         physicsClientId=cid)
        self._gantry_ids.append(
            p.createMultiBody(baseMass=0.0,
                              baseCollisionShapeIndex=-1,
                              baseVisualShapeIndex=jib_visual,
                              basePosition=((mast_x + anchor[0]) / 2,
                                            (mast_y + anchor[1]) / 2,
                                            anchor_z),
                              baseOrientation=p.getQuaternionFromEuler(
                                  [0.0, 0.0,
                                   float(np.arctan2(dy, dx))]),
                              physicsClientId=cid))
        self._built_lane = (lane_y, length)

    def _ram_angle(self) -> float:
        return float(
            p.getJointState(self._ram.id,
                            0,
                            physicsClientId=self._physics_client_id)[0])

    def _set_ram_angle(self, angle: float) -> None:
        p.resetJointState(self._ram.id,
                          0,
                          angle,
                          targetVelocity=0.0,
                          physicsClientId=self._physics_client_id)

    def head_position(self) -> Tuple[float, float, float]:
        """The head's centre now."""
        pos = p.getLinkState(self._ram.id,
                             0,
                             physicsClientId=self._physics_client_id)[0]
        return (float(pos[0]), float(pos[1]), float(pos[2]))

    def _head_speed(self) -> float:
        vel = p.getLinkState(self._ram.id,
                             0,
                             computeLinkVelocity=1,
                             physicsClientId=self._physics_client_id)[6]
        return float(np.linalg.norm(vel))

    # =========================================================================
    # STATE READ / WRITE
    # =========================================================================
    def _get_object_ids_for_held_check(self) -> List[int]:
        """Nothing here is grasped; the ram is pushed."""
        return []

    def _speed(self, obj: Object) -> float:
        (vx, vy,
         vz), _ = p.getBaseVelocity(obj.id,
                                    physicsClientId=self._physics_client_id)
        return float(np.linalg.norm([vx, vy, vz]))

    def _get_domain_specific_feature(self, obj: Object, feature: str) -> float:
        if obj.type.name == "ram":
            if feature == "angle":
                return self._ram_angle()
            if feature == "speed":
                return self._head_speed()
        if obj.type.name == "crane" and feature == "length":
            return float(self._built_length or 0.0)
        if obj.type.name == "crate":
            if feature == "color":
                return float(self._crate_color)
            if feature == "speed":
                return self._speed(obj)
        if obj.type.name == "bin" and feature == "half":
            return float(self._bin_half)
        raise ValueError(f"Unknown feature {feature} for object {obj}")

    def _set_domain_specific_state(self, state: State) -> None:
        lane_y = float(state.get(self._crane, "y"))
        length = float(state.get(self._crane, "length"))
        if self._built_lane != (lane_y, length):
            self._build_crane(lane_y, length)
        p.resetBasePositionAndOrientation(
            self._ram.id, (state.get(self._ram, "x"), state.get(
                self._ram, "y"), state.get(self._ram, "z")),
            (0.0, 0.0, 0.0, 1.0),
            physicsClientId=self._physics_client_id)
        self._set_ram_angle(float(state.get(self._ram, "angle")))
        self._crate_color = int(round(state.get(self._crate, "color")))
        p.changeVisualShape(self._crate.id,
                            -1,
                            rgbaColor=self.color_rgba(self._crate_color),
                            physicsClientId=self._physics_client_id)
        half = float(state.get(self._bin, "half"))
        if abs(half - self._bin_half) > 1e-6:
            p.removeBody(self._bin.id, physicsClientId=self._physics_client_id)
            bin_visual = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=(half, self.bin_half_across, 0.003),
                rgbaColor=self.bin_color,
                physicsClientId=self._physics_client_id)
            self._bin.id = p.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=-1,
                baseVisualShapeIndex=bin_visual,
                physicsClientId=self._physics_client_id)
            self._bin_half = half
        p.resetBasePositionAndOrientation(
            self._bin.id, (state.get(self._bin, "x"), state.get(
                self._bin, "y"), self.table_height + 0.003),
            (0.0, 0.0, 0.0, 1.0),
            physicsClientId=self._physics_client_id)

    @classmethod
    def max_pull(cls) -> float:
        """How far back along the lane the head can be drawn."""
        return float(CFG.crane_pull_range[1])
