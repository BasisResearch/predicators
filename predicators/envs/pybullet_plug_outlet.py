"""A PyBullet plug-into-outlet insertion domain (US three-leg plug).

The scene has a table, a Fetch robot, a static *outlet* block with three holes
through its top face, a static *holder* ring that keeps the plug standing
upright, and one dynamic *plug*: a grip block carrying two flat blades and a
round ground pin in the US (NEMA 5-15) arrangement, scaled up to what this
arm's few-millimetre positioning accuracy can address at all.

Three legs is the point. A single prong only constrains x and y, so a plug can
be rotated freely about the vertical and still drop in. Two blades plus an
offset ground pin constrain yaw as well: with the blades 24 mm apart, a yaw
error of theta displaces a blade end by 12 mm * sin(theta), so at 1.5 mm
clearance the plug must be within about 7 degrees of square before any depth
is possible. Alignment is three-dimensional, and the goal predicate needs no
separate yaw term because a leg that is out of square misses its hole.

Designed for reset-free use: nothing in the scene moves on its own. If the plug
topples or leaves the table and is not picked up again within
``topple_patience_steps`` steps, it is respawned upright in the holder and the
event is counted in ``num_interventions`` (a stand-in for a human putting the
plug back).

Clearance is a class attribute (``clearance``) so callers can set the
difficulty tier before constructing the env.
"""

from typing import Any, ClassVar, Dict, List, Sequence, Set, Tuple

import numpy as np
import pybullet as p

from predicators import utils
from predicators.envs.pybullet_env import PyBulletEnv
from predicators.pybullet_helpers.geometry import Pose3D, Quaternion
from predicators.pybullet_helpers.robots import SingleArmPyBulletRobot
from predicators.settings import CFG
from predicators.structs import Action, EnvironmentTask, GroundAtom, Object, \
    Predicate, State, Type

BoxSpec = Tuple[Tuple[float, float, float], Tuple[float, float, float]]


def apply_urdf_torque_limits(robot: SingleArmPyBulletRobot,
                             scale: float = 3.0) -> None:
    """Cap the arm's position-control motors at ``scale`` x the URDF effort
    limits.

    PyBullet's default position controller applies effectively unbounded
    torque, so a misaligned plug is simply forced through a socket wall
    (the wall yields a few millimetres under the solver). With bounded
    torques the arm behaves like the real robot: a misaligned prong jams
    against the rim and stops. The raw URDF limits (about 34/132/77/66/29/
    26/7 N m for Fetch) are too low for PyBullet's stiff, gravity-blind
    position controller: several joints saturate in stretched poses and
    the arm sags by up to 20 cm. Three times the URDF limits keeps a 4 mm
    misaligned plug on the rim (depth < 1 mm) while free-space moves stay
    within a few millimetres (measured 2026-09-07, DEBUG_LOG entry 8).
    Finger motors are capped at their own URDF effort (60 N for Fetch, no
    scaling): with unbounded finger force the pads squeeze through a
    constrained (held) block until it slips out of the grasp (DEBUG_LOG 12).

    The robot's ``set_motors`` is wrapped in place; calling this twice is
    harmless.
    """
    if getattr(robot, "_urdf_torque_limits_applied", False):
        return
    client = robot.physics_client_id
    finger_ids = [robot.left_finger_id, robot.right_finger_id]
    arm_ids = [j for j in robot.arm_joints if j not in finger_ids]
    arm_idx = [robot.arm_joints.index(j) for j in arm_ids]
    finger_idx = [robot.arm_joints.index(j) for j in finger_ids]
    forces = [
        scale * float(p.getJointInfo(robot.robot_id, j,
                                     physicsClientId=client)[10])
        for j in arm_ids
    ]
    finger_forces = [
        float(p.getJointInfo(robot.robot_id, j, physicsClientId=client)[10])
        for j in finger_ids
    ]
    original_set_motors = robot.set_motors

    def set_motors_with_limits(joint_positions: Sequence[float]) -> None:
        original_set_motors(joint_positions)
        if CFG.pybullet_control_mode != "position":
            return
        p.setJointMotorControlArray(
            bodyUniqueId=robot.robot_id,
            jointIndices=arm_ids,
            controlMode=p.POSITION_CONTROL,
            targetPositions=[joint_positions[i] for i in arm_idx],
            forces=forces,
            physicsClientId=client)
        p.setJointMotorControlArray(
            bodyUniqueId=robot.robot_id,
            jointIndices=finger_ids,
            controlMode=p.POSITION_CONTROL,
            targetPositions=[joint_positions[i] for i in finger_idx],
            forces=finger_forces,
            physicsClientId=client)

    robot.set_motors = set_motors_with_limits  # type: ignore[method-assign]
    robot._urdf_torque_limits_applied = True  # type: ignore[attr-defined]




class PyBulletPlugOutletEnv(PyBulletEnv):
    """US three-leg plug into an outlet."""

    # Table (same slab and placement as the donut domain).
    table_height: ClassVar[float] = 0.2
    _table_pose: ClassVar[Pose3D] = (1.35, 0.75, 0.0)
    _table_orientation: ClassVar[Quaternion] = (0., 0., 0., 1.)
    x_lb: ClassVar[float] = 1.15
    x_ub: ClassVar[float] = 1.55
    y_lb: ClassVar[float] = 0.4
    y_ub: ClassVar[float] = 1.1

    # Robot.
    robot_init_x: ClassVar[float] = 1.35
    robot_init_y: ClassVar[float] = 0.75
    robot_init_z: ClassVar[float] = 0.5
    robot_base_pos: ClassVar[Tuple[float, float, float]] = (0.75, 0.75, 0.0)
    robot_base_orn: ClassVar[Tuple[float, float, float,
                                   float]] = (0., 0., 0., 1.)

    # ── Plug geometry (metres, half extents where noted) ────────
    # Grip block: the jaws close along world x at yaw 0, so its x extent is
    # what gets grasped. Kept under the ~70 mm the Fetch gripper can span.
    plug_block_half: ClassVar[Tuple[float, float, float]] = (0.022, 0.020,
                                                             0.0125)
    # Two flat blades, side by side in x, offset toward +y like a wall socket.
    # 30 mm blades, not the 16 mm of a real NEMA plug: the finger pads reach
    # about 32 mm below the EE frame, so short legs make the fingers hit the
    # outlet plate before the legs are deep (measured 2026-09-08: the oracle
    # jammed at 9 mm depth under 900 N at every clearance).
    blade_half: ClassVar[Tuple[float, float, float]] = (0.002, 0.006, 0.015)
    blade_dx: ClassVar[float] = 0.012  # blades at x = +/- blade_dx
    blade_dy: ClassVar[float] = 0.008
    # Round ground pin, longer than the blades so it enters first.
    pin_radius: ClassVar[float] = 0.004
    pin_half_len: ClassVar[float] = 0.016
    pin_dy: ClassVar[float] = -0.014
    plug_mass: ClassVar[float] = 0.1
    plug_friction: ClassVar[float] = 1.0
    plug_color: ClassVar[Tuple[float, float, float, float]] = (0.1, 0.1, 0.1,
                                                               1.0)
    pin_color: ClassVar[Tuple[float, float, float,
                              float]] = (0.75, 0.75, 0.78, 1.0)

    # ── Outlet geometry ─────────────────────────────────────────
    outlet_half_xy: ClassVar[float] = 0.05
    outlet_height: ClassVar[float] = 0.025
    outlet_color: ClassVar[Tuple[float, float, float,
                                 float]] = (0.95, 0.95, 0.9, 1.0)
    # Per-side gap between a leg and its hole wall. Set by tuning: the
    # tightest value a ground-truth oracle can insert reliably (the plug
    # shifts a couple of millimetres in the jaws on contact, which is the
    # real floor). ``clearance_tiers`` is kept for sweeps.
    clearance: ClassVar[float] = 0.004
    # Gate measurement, 2026-09-08, five seeds of the ground-truth oracle:
    # 4.0 mm inserts 5/5 at zero contact force; 3.0 mm only 3/5; 2.5 mm and
    # below jam on the plate at about 900 N with the plug 8 degrees off
    # square. The floor is the plug tilting in the jaws, not the hole size.
    clearance_tiers: ClassVar[Dict[str, float]] = {
        "loose": 0.005,
        "default": 0.004,
        "tight": 0.003,
    }

    # Holder ring geometry: a loose collar the plug stands in.
    # The collar grips the legs, not the block: with 30 mm legs the block sits
    # well clear of the table.
    holder_inner_clearance: ClassVar[float] = 0.006
    holder_wall: ClassVar[float] = 0.01
    holder_height: ClassVar[float] = 0.02
    holder_color: ClassVar[Tuple[float, float, float,
                                 float]] = (0.3, 0.3, 0.35, 1.0)

    # Success: every leg tip this far below the outlet's top face, inside its
    # own hole, with the plug axis within alignment_max_deg of vertical.
    insertion_depth: ClassVar[float] = 0.010
    alignment_max_deg: ClassVar[float] = 10.0
    # Lateral slack allowed on a leg tip beyond the nominal clearance.
    tip_tolerance: ClassVar[float] = 0.002
    # Rotation about the vertical, relative to the outlet. The per-leg checks
    # alone are too permissive to encode this: a 15 degree yaw only displaces a
    # blade tip by about 3.8 mm, inside the hole tolerance, even though such a
    # plug could never physically be at depth.
    yaw_max_deg: ClassVar[float] = 8.0

    # Respawn (human intervention) rule.
    topple_max_deg: ClassVar[float] = 60.0
    topple_patience_steps: ClassVar[int] = 50

    # Physics for tight fits.
    num_solver_iterations: ClassVar[int] = 150
    use_urdf_torque_limits: ClassVar[bool] = True
    torque_limit_scale: ClassVar[float] = 3.0

    # Camera.
    _camera_target: ClassVar[Pose3D] = (1.35, 0.8, 0.2)
    _camera_distance: ClassVar[float] = 0.6
    _camera_yaw: ClassVar[float] = 90
    _camera_pitch: ClassVar[float] = -45

    # Types.
    _robot_type = Type("robot", ["x", "y", "z", "fingers"])
    _plug_type = Type("plug",
                      ["x", "y", "z", "roll", "pitch", "yaw", "is_held"])
    _outlet_type = Type("outlet", ["x", "y", "z", "yaw"])
    _holder_type = Type("holder", ["x", "y", "z"])

    def __init__(self, use_gui: bool = False, **kwargs: Any) -> None:
        self._robot = Object("robot", self._robot_type)
        self._plug = Object("plug", self._plug_type)
        self._outlet = Object("outlet", self._outlet_type)
        self._holder = Object("holder", self._holder_type)

        self._PluggedIn = Predicate("PluggedIn",
                                    [self._plug_type, self._outlet_type],
                                    self._PluggedIn_holds)
        self._Holding = Predicate("Holding", [self._plug_type],
                                  self._Holding_holds)

        self._plug_id: int = -1
        self._outlet_id: int = -1
        self._holder_id: int = -1
        self._table_id: int = -1
        self._bad_pose_steps = 0
        self.num_interventions = 0

        super().__init__(use_gui, **kwargs)

    @classmethod
    def get_name(cls) -> str:
        return "pybullet_plug_outlet"

    @property
    def types(self) -> Set[Type]:
        return {
            self._robot_type, self._plug_type, self._outlet_type,
            self._holder_type
        }

    @property
    def predicates(self) -> Set[Predicate]:
        return {self._PluggedIn, self._Holding}

    @property
    def goal_predicates(self) -> Set[Predicate]:
        return {self._PluggedIn}

    # ── Geometry helpers ────────────────────────────────────────

    @classmethod
    def leg_specs(cls) -> List[Tuple[str, Tuple[float, float], Tuple[float, float], float]]:
        """Per leg: ``(name, (dx, dy), (half_x, half_y), tip_offset)``.

        ``(dx, dy)`` is the leg's offset in the plug frame, the half extents
        describe its cross-section, and ``tip_offset`` is how far the tip sits
        below the plug frame origin.
        """
        block_h = cls.plug_block_half[2]
        return [
            ("blade_left", (-cls.blade_dx, cls.blade_dy),
             (cls.blade_half[0], cls.blade_half[1]),
             block_h + 2 * cls.blade_half[2]),
            ("blade_right", (cls.blade_dx, cls.blade_dy),
             (cls.blade_half[0], cls.blade_half[1]),
             block_h + 2 * cls.blade_half[2]),
            ("ground_pin", (0.0, cls.pin_dy),
             (cls.pin_radius, cls.pin_radius),
             block_h + 2 * cls.pin_half_len),
        ]

    @classmethod
    def prong_tip_offset(cls) -> float:
        """Distance from the plug frame origin down to the blade tips."""
        return cls.plug_block_half[2] + 2 * cls.blade_half[2]

    @classmethod
    def pin_tip_offset(cls) -> float:
        """Distance from the plug frame origin down to the ground pin tip."""
        return cls.plug_block_half[2] + 2 * cls.pin_half_len

    @classmethod
    def plug_rest_z(cls) -> float:
        """Plug frame z when the plug stands upright on the table."""
        return cls.table_height + cls.pin_tip_offset()

    @classmethod
    def outlet_center_z(cls) -> float:
        """Outlet frame z when it sits on the table."""
        return cls.table_height + cls.outlet_height / 2.0

    @classmethod
    def outlet_top_z(cls) -> float:
        """World z of the outlet's top face."""
        return cls.table_height + cls.outlet_height

    @classmethod
    def hole_half_extents(cls, leg_half: Tuple[float, float]
                          ) -> Tuple[float, float]:
        """Half extents of the hole cut for a leg of this cross-section."""
        return (leg_half[0] + cls.clearance, leg_half[1] + cls.clearance)

    @classmethod
    def _outlet_boxes(cls) -> List[BoxSpec]:
        """Decompose the outlet plate into boxes leaving three holes.

        Bands run down the local y axis: full-width strips above the blades,
        between the blades and the pin, and below the pin; the blade band is
        split into three x strips and the pin band into two.
        """
        a = cls.outlet_half_xy
        hz = cls.outlet_height / 2.0
        bhx, bhy = cls.hole_half_extents(
            (cls.blade_half[0], cls.blade_half[1]))
        phx, phy = cls.hole_half_extents((cls.pin_radius, cls.pin_radius))
        blade_top, blade_bot = cls.blade_dy + bhy, cls.blade_dy - bhy
        pin_top, pin_bot = cls.pin_dy + phy, cls.pin_dy - phy
        boxes: List[BoxSpec] = []

        def strip(y0: float, y1: float, x0: float, x1: float) -> None:
            if y1 - y0 < 1e-6 or x1 - x0 < 1e-6:
                return
            boxes.append((((x1 - x0) / 2.0, (y1 - y0) / 2.0, hz),
                          ((x0 + x1) / 2.0, (y0 + y1) / 2.0, 0.0)))

        strip(blade_top, a, -a, a)  # above the blades
        # blade band: left of the left blade, between the blades, right of it
        strip(blade_bot, blade_top, -a, -cls.blade_dx - bhx)
        strip(blade_bot, blade_top, -cls.blade_dx + bhx, cls.blade_dx - bhx)
        strip(blade_bot, blade_top, cls.blade_dx + bhx, a)
        strip(pin_top, blade_bot, -a, a)  # between the blades and the pin
        strip(pin_bot, pin_top, -a, -phx)  # pin band
        strip(pin_bot, pin_top, phx, a)
        strip(-a, pin_bot, -a, a)  # below the pin
        return boxes

    @staticmethod
    def _ring_boxes(inner_half_x: float, inner_half_y: float,
                    outer_half: float, half_height: float) -> List[BoxSpec]:
        """Four boxes forming a rectangular ring around a hole."""
        wall_x = (outer_half - inner_half_x) / 2.0
        wall_y = (outer_half - inner_half_y) / 2.0
        return [
            ((wall_x, outer_half, half_height),
             ((outer_half + inner_half_x) / 2.0, 0.0, 0.0)),
            ((wall_x, outer_half, half_height),
             (-(outer_half + inner_half_x) / 2.0, 0.0, 0.0)),
            ((inner_half_x, wall_y, half_height),
             (0.0, (outer_half + inner_half_y) / 2.0, 0.0)),
            ((inner_half_x, wall_y, half_height),
             (0.0, -(outer_half + inner_half_y) / 2.0, 0.0)),
        ]

    @staticmethod
    def _create_compound(boxes: Sequence[BoxSpec],
                         color: Tuple[float, float, float, float],
                         mass: float, friction: float, position: Pose3D,
                         physics_client_id: int) -> int:
        """Create one rigid body made of several boxes."""
        half_extents = [list(b[0]) for b in boxes]
        positions = [list(b[1]) for b in boxes]
        n = len(boxes)
        collision_id = p.createCollisionShapeArray(
            shapeTypes=[p.GEOM_BOX] * n,
            halfExtents=half_extents,
            collisionFramePositions=positions,
            physicsClientId=physics_client_id)
        visual_id = p.createVisualShapeArray(
            shapeTypes=[p.GEOM_BOX] * n,
            halfExtents=half_extents,
            visualFramePositions=positions,
            rgbaColors=[list(color)] * n,
            physicsClientId=physics_client_id)
        body_id = p.createMultiBody(baseMass=mass,
                                    baseCollisionShapeIndex=collision_id,
                                    baseVisualShapeIndex=visual_id,
                                    basePosition=position,
                                    physicsClientId=physics_client_id)
        p.changeDynamics(body_id,
                         -1,
                         lateralFriction=friction,
                         physicsClientId=physics_client_id)
        return body_id

    @classmethod
    def _create_plug(cls, position: Pose3D, physics_client_id: int) -> int:
        """Grip block plus two blades and a cylindrical ground pin."""
        block_h = cls.plug_block_half[2]
        shape_types = [p.GEOM_BOX, p.GEOM_BOX, p.GEOM_BOX, p.GEOM_CYLINDER]
        half_extents = [
            list(cls.plug_block_half),
            list(cls.blade_half),
            list(cls.blade_half),
            [0.0, 0.0, 0.0],
        ]
        radii = [0.0, 0.0, 0.0, cls.pin_radius]
        lengths = [0.0, 0.0, 0.0, 2 * cls.pin_half_len]
        frames = [
            [0.0, 0.0, 0.0],
            [-cls.blade_dx, cls.blade_dy, -(block_h + cls.blade_half[2])],
            [cls.blade_dx, cls.blade_dy, -(block_h + cls.blade_half[2])],
            [0.0, cls.pin_dy, -(block_h + cls.pin_half_len)],
        ]
        colors = [list(cls.plug_color)] * 3 + [list(cls.pin_color)]
        collision_id = p.createCollisionShapeArray(
            shapeTypes=shape_types,
            halfExtents=half_extents,
            radii=radii,
            lengths=lengths,
            collisionFramePositions=frames,
            physicsClientId=physics_client_id)
        visual_id = p.createVisualShapeArray(
            shapeTypes=shape_types,
            halfExtents=half_extents,
            radii=radii,
            lengths=lengths,
            visualFramePositions=frames,
            rgbaColors=colors,
            physicsClientId=physics_client_id)
        plug_id = p.createMultiBody(baseMass=cls.plug_mass,
                                    baseCollisionShapeIndex=collision_id,
                                    baseVisualShapeIndex=visual_id,
                                    basePosition=position,
                                    physicsClientId=physics_client_id)
        p.changeDynamics(plug_id,
                         -1,
                         lateralFriction=cls.plug_friction,
                         linearDamping=0.04,
                         angularDamping=0.04,
                         physicsClientId=physics_client_id)
        return plug_id

    # ── PyBullet setup ──────────────────────────────────────────

    @classmethod
    def initialize_pybullet(
            cls, using_gui: bool
    ) -> Tuple[int, SingleArmPyBulletRobot, Dict[str, Any]]:
        physics_client_id, pybullet_robot, bodies = super(
        ).initialize_pybullet(using_gui)
        p.setPhysicsEngineParameter(
            numSolverIterations=cls.num_solver_iterations,
            physicsClientId=physics_client_id)
        if cls.use_urdf_torque_limits:
            apply_urdf_torque_limits(pybullet_robot, cls.torque_limit_scale)

        table_id = p.loadURDF(utils.get_env_asset_path("urdf/table.urdf"),
                              useFixedBase=True,
                              globalScaling=1.0,
                              physicsClientId=physics_client_id)
        p.resetBasePositionAndOrientation(table_id,
                                          cls._table_pose,
                                          cls._table_orientation,
                                          physicsClientId=physics_client_id)
        bodies["table_id"] = table_id

        # Outlet: a plate with three through-holes; the table closes them off
        # at the bottom, so hole depth is the plate's own height.
        bodies["outlet_id"] = cls._create_compound(
            cls._outlet_boxes(),
            cls.outlet_color,
            mass=0.0,
            friction=0.6,
            position=(cls._out_of_view_xy[0], cls._out_of_view_xy[1],
                      cls.outlet_center_z()),
            physics_client_id=physics_client_id)

        # Holder: a loose collar that keeps the plug upright on the table.
        leg_half_x = cls.blade_dx + cls.blade_half[0]
        leg_half_y = (cls.blade_dy + cls.blade_half[1] - cls.pin_dy
                      + cls.pin_radius) / 2.0
        ih_x = leg_half_x + cls.holder_inner_clearance
        ih_y = leg_half_y + cls.holder_inner_clearance
        outer = max(ih_x, ih_y) + cls.holder_wall
        bodies["holder_id"] = cls._create_compound(
            cls._ring_boxes(ih_x, ih_y, outer, cls.holder_height / 2.0),
            cls.holder_color,
            mass=0.0,
            friction=0.6,
            position=(cls._out_of_view_xy[0], cls._out_of_view_xy[1] + 1.0,
                      cls.table_height + cls.holder_height / 2.0),
            physics_client_id=physics_client_id)

        bodies["plug_id"] = cls._create_plug(
            (cls._out_of_view_xy[0], cls._out_of_view_xy[1] + 2.0,
             cls.plug_rest_z()), physics_client_id)
        return physics_client_id, pybullet_robot, bodies

    def _store_pybullet_bodies(self, pybullet_bodies: Dict[str, Any]) -> None:
        self._table_id = pybullet_bodies["table_id"]
        self._outlet_id = pybullet_bodies["outlet_id"]
        self._holder_id = pybullet_bodies["holder_id"]
        self._plug_id = pybullet_bodies["plug_id"]
        self._plug.id = self._plug_id
        self._outlet.id = self._outlet_id
        self._holder.id = self._holder_id

    def _get_object_ids_for_held_check(self) -> List[int]:
        return [self._plug_id]

    def _get_domain_specific_feature(self, obj: Object, feature: str) -> float:
        raise ValueError(f"Unknown feature {feature} for object {obj}")

    def _set_domain_specific_state(self, state: State) -> None:
        p.resetBaseVelocity(self._plug_id, [0, 0, 0], [0, 0, 0],
                            physicsClientId=self._physics_client_id)
        self._bad_pose_steps = 0

    def reset(self, train_or_test: str, task_idx: int,
              render: bool = False) -> State:
        self.num_interventions = 0
        self._bad_pose_steps = 0
        return super().reset(train_or_test, task_idx, render=render)

    # ── Dynamics ────────────────────────────────────────────────

    def _plug_is_recoverable(self) -> bool:
        """False when the plug lies toppled or has left the table."""
        (px, py, pz), orn = p.getBasePositionAndOrientation(
            self._plug_id, physicsClientId=self._physics_client_id)
        if pz < self.table_height - 0.05:
            return False
        if not (self.x_lb - 0.1 <= px <= self.x_ub + 0.1
                and self.y_lb - 0.1 <= py <= self.y_ub + 0.1):
            return False
        rot = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
        up = rot @ np.array([0.0, 0.0, 1.0])
        tilt_deg = np.degrees(np.arccos(np.clip(up[2], -1.0, 1.0)))
        return bool(tilt_deg <= self.topple_max_deg)

    def _domain_specific_step(self) -> None:
        if self._plug_id == self._held_obj_id:
            self._bad_pose_steps = 0
            return
        if self._plug_is_recoverable():
            self._bad_pose_steps = 0
            return
        self._bad_pose_steps += 1
        if self._bad_pose_steps >= self.topple_patience_steps:
            self._respawn_plug_in_holder()
            self._bad_pose_steps = 0
            self.num_interventions += 1

    def _respawn_plug_in_holder(self) -> None:
        (hx, hy, _), _ = p.getBasePositionAndOrientation(
            self._holder_id, physicsClientId=self._physics_client_id)
        p.resetBasePositionAndOrientation(
            self._plug_id, [hx, hy, self.plug_rest_z() + 0.002],
            (0., 0., 0., 1.),
            physicsClientId=self._physics_client_id)
        p.resetBaseVelocity(self._plug_id, [0, 0, 0], [0, 0, 0],
                            physicsClientId=self._physics_client_id)

    # ── Predicates ──────────────────────────────────────────────

    @classmethod
    def leg_tips_and_axis(cls, state: State,
                          plug: Object) -> Tuple[Dict[str, np.ndarray],
                                                 np.ndarray]:
        """World positions of the three leg tips, and the plug's up axis."""
        pos = np.array([state.get(plug, f) for f in ("x", "y", "z")])
        rpy = [state.get(plug, f) for f in ("roll", "pitch", "yaw")]
        rot = np.array(p.getMatrixFromQuaternion(
            p.getQuaternionFromEuler(rpy))).reshape(3, 3)
        tips = {}
        for name, (dx, dy), _half, tip_offset in cls.leg_specs():
            local = np.array([dx, dy, -tip_offset])
            tips[name] = pos + rot @ local
        return tips, rot @ np.array([0.0, 0.0, 1.0])

    @classmethod
    def plug_tip_and_axis(cls, state: State,
                          plug: Object) -> Tuple[np.ndarray, np.ndarray]:
        """Mean blade tip and the plug's up axis (kept for callers that want a
        single representative tip)."""
        tips, up = cls.leg_tips_and_axis(state, plug)
        mean_blade = 0.5 * (tips["blade_left"] + tips["blade_right"])
        return mean_blade, up

    @classmethod
    def _PluggedIn_holds(cls, state: State, objects: Sequence[Object]) -> bool:
        """Every leg tip inside its own hole and at least insertion_depth below
        the outlet's top face, with the plug close to vertical and square to
        the outlet.
        """
        plug, outlet = objects
        tips, up = cls.leg_tips_and_axis(state, plug)
        cos_tilt = float(np.clip(up[2], -1.0, 1.0))
        if cos_tilt < np.cos(np.radians(cls.alignment_max_deg)):
            return False
        ox, oy, oz = [state.get(outlet, f) for f in ("x", "y", "z")]
        oyaw = state.get(outlet, "yaw")
        yaw_err = (state.get(plug, "yaw") - oyaw + np.pi) % (2 * np.pi) - np.pi
        if abs(np.degrees(yaw_err)) > cls.yaw_max_deg:
            return False
        top_z = oz + cls.outlet_height / 2.0
        c, s = np.cos(-oyaw), np.sin(-oyaw)
        for name, (dx, dy), half, _tip_offset in cls.leg_specs():
            tip = tips[name]
            if tip[2] > top_z - cls.insertion_depth:
                return False
            # Express the tip in the outlet frame and compare with the hole.
            wx, wy = tip[0] - ox, tip[1] - oy
            lx, ly = c * wx - s * wy, s * wx + c * wy
            hx = half[0] + cls.clearance + cls.tip_tolerance
            hy = half[1] + cls.clearance + cls.tip_tolerance
            if abs(lx - dx) > hx or abs(ly - dy) > hy:
                return False
        return True

    @staticmethod
    def _Holding_holds(state: State, objects: Sequence[Object]) -> bool:
        plug, = objects
        return state.get(plug, "is_held") > 0.5

    # ── Tasks ───────────────────────────────────────────────────

    def _generate_train_tasks(self) -> List[EnvironmentTask]:
        return self._make_tasks(num_tasks=CFG.num_train_tasks,
                                rng=self._train_rng)

    def _generate_test_tasks(self) -> List[EnvironmentTask]:
        return self._make_tasks(num_tasks=CFG.num_test_tasks,
                                rng=self._test_rng)

    def _make_tasks(self, num_tasks: int,
                    rng: np.random.Generator) -> List[EnvironmentTask]:
        tasks = []
        for _ in range(num_tasks):
            init_dict: Dict[Object, Dict[str, float]] = {}
            init_dict[self._robot] = {
                "x": self.robot_init_x,
                "y": self.robot_init_y,
                "z": self.robot_init_z,
                "fingers": self.open_fingers,
            }
            init_dict[self._outlet] = {
                "x": 1.35,
                "y": 0.95,
                "z": self.outlet_center_z(),
                "yaw": 0.0,
            }
            holder_x = rng.uniform(1.27, 1.43)
            holder_y = rng.uniform(0.52, 0.62)
            init_dict[self._holder] = {
                "x": holder_x,
                "y": holder_y,
                "z": self.table_height + self.holder_height / 2.0,
            }
            init_dict[self._plug] = {
                "x": holder_x,
                "y": holder_y,
                "z": self.plug_rest_z(),
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "is_held": 0.0,
            }
            init_state = utils.create_state_from_dict(init_dict)
            goal = {GroundAtom(self._PluggedIn, [self._plug, self._outlet])}
            tasks.append(EnvironmentTask(init_state, goal))
        return self._add_pybullet_state_to_tasks(tasks)


if __name__ == "__main__":
    import time
    CFG.seed = 0
    CFG.env = "pybullet_plug_outlet"
    CFG.num_train_tasks = 1
    env = PyBulletPlugOutletEnv(use_gui=True)
    env.reset("train", 0)
    while True:
        _act = Action(
            np.array(env._pybullet_robot.get_joints(), dtype=np.float32))
        env.step(_act)
        time.sleep(0.01)
