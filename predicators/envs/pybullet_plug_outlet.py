"""A PyBullet plug-into-outlet insertion domain.

The scene has a table, a Fetch robot, a static *outlet* block with a
rectangular socket through its top face, a static *holder* ring that keeps the
plug standing upright, and one dynamic *plug* (a grip block on top of a
rectangular prong). The task is to grasp the plug and seat its prong in the
socket. Getting close is easy; seating the prong within a clearance of a few
millimetres is the precise part.

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
    """Plug-into-outlet insertion domain."""

    # Table (same placement as the donut domain).
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

    # Plug geometry: half extents in metres. The plug frame origin is the
    # centre of the grip block; the prong hangs below it.
    # Sized for the Fetch gripper: its pads extend ~3.2 cm below the EE
    # frame, so the prong must be long enough that the required insertion
    # depth is reached before the pads touch the outlet's top face.
    plug_block_half: ClassVar[Tuple[float, float, float]] = (0.015, 0.010,
                                                             0.0125)
    plug_prong_half: ClassVar[Tuple[float, float, float]] = (0.010, 0.005,
                                                             0.015)
    plug_mass: ClassVar[float] = 0.1
    plug_friction: ClassVar[float] = 1.0
    plug_color: ClassVar[Tuple[float, float, float, float]] = (0.1, 0.1, 0.1,
                                                               1.0)

    # Outlet geometry.
    outlet_half_xy: ClassVar[float] = 0.04
    outlet_height: ClassVar[float] = 0.025
    outlet_color: ClassVar[Tuple[float, float, float,
                                 float]] = (0.95, 0.95, 0.9, 1.0)
    # Per-side gap between prong and socket wall. Tiers: easy 0.004,
    # medium 0.002, hard 0.001. Set on the class before construction.
    clearance: ClassVar[float] = 0.002
    clearance_tiers: ClassVar[Dict[str, float]] = {
        "easy": 0.004,
        "medium": 0.002,
        "hard": 0.001,
    }

    # Holder ring geometry.
    holder_inner_clearance: ClassVar[float] = 0.01
    holder_wall: ClassVar[float] = 0.01
    holder_height: ClassVar[float] = 0.01
    holder_color: ClassVar[Tuple[float, float, float,
                                 float]] = (0.3, 0.3, 0.35, 1.0)

    # Success criteria.
    insertion_depth: ClassVar[float] = 0.015
    alignment_max_deg: ClassVar[float] = 10.0

    # Respawn (human intervention) rule.
    topple_max_deg: ClassVar[float] = 60.0
    topple_patience_steps: ClassVar[int] = 50

    # Physics for tight fits.
    num_solver_iterations: ClassVar[int] = 150
    # Bounded arm torque limits (see apply_urdf_torque_limits).
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
    def socket_half_extents(cls) -> Tuple[float, float]:
        """Half extents (x, y) of the socket opening."""
        return (cls.plug_prong_half[0] + cls.clearance,
                cls.plug_prong_half[1] + cls.clearance)

    @classmethod
    def plug_rest_z(cls) -> float:
        """Plug frame z when the plug stands upright on the table."""
        return cls.table_height + 2 * cls.plug_prong_half[2] + \
            cls.plug_block_half[2]

    @classmethod
    def outlet_center_z(cls) -> float:
        """Outlet frame z when it sits on the table."""
        return cls.table_height + cls.outlet_height / 2.0

    @classmethod
    def outlet_top_z(cls) -> float:
        """World z of the outlet's top face."""
        return cls.table_height + cls.outlet_height

    @classmethod
    def prong_tip_offset(cls) -> float:
        """Distance from plug frame origin down to the prong tip."""
        return cls.plug_block_half[2] + 2 * cls.plug_prong_half[2]

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

        # Outlet: ring with a through-socket; the table is the socket floor.
        hx, hy = cls.socket_half_extents()
        outlet_boxes = cls._ring_boxes(hx, hy, cls.outlet_half_xy,
                                       cls.outlet_height / 2.0)
        bodies["outlet_id"] = cls._create_compound(
            outlet_boxes,
            cls.outlet_color,
            mass=0.0,
            friction=0.6,
            position=(cls._out_of_view_xy[0], cls._out_of_view_xy[1],
                      cls.outlet_center_z()),
            physics_client_id=physics_client_id)

        # Holder: loose ring that keeps the plug upright.
        ih_x = cls.plug_prong_half[0] + cls.holder_inner_clearance
        ih_y = cls.plug_prong_half[1] + cls.holder_inner_clearance
        outer = max(ih_x, ih_y) + cls.holder_wall
        holder_boxes = cls._ring_boxes(ih_x, ih_y, outer,
                                       cls.holder_height / 2.0)
        bodies["holder_id"] = cls._create_compound(
            holder_boxes,
            cls.holder_color,
            mass=0.0,
            friction=0.6,
            position=(cls._out_of_view_xy[0], cls._out_of_view_xy[1] + 1.0,
                      cls.table_height + cls.holder_height / 2.0),
            physics_client_id=physics_client_id)

        # Plug: grip block with a prong hanging below.
        plug_boxes: List[BoxSpec] = [
            (cls.plug_block_half, (0.0, 0.0, 0.0)),
            (cls.plug_prong_half,
             (0.0, 0.0, -(cls.plug_block_half[2] + cls.plug_prong_half[2]))),
        ]
        plug_id = cls._create_compound(
            plug_boxes,
            cls.plug_color,
            mass=cls.plug_mass,
            friction=cls.plug_friction,
            position=(cls._out_of_view_xy[0], cls._out_of_view_xy[1] + 2.0,
                      cls.plug_rest_z()),
            physics_client_id=physics_client_id)
        # A little damping keeps the light plug from chattering in the socket.
        p.changeDynamics(plug_id,
                         -1,
                         linearDamping=0.04,
                         angularDamping=0.04,
                         physicsClientId=physics_client_id)
        bodies["plug_id"] = plug_id
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
        # All objects carry PyBullet ids; the base class teleports them.
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
    def plug_tip_and_axis(
            cls, state: State,
            plug: Object) -> Tuple[np.ndarray, np.ndarray]:
        """World position of the prong tip and the plug's up axis."""
        pos = np.array([state.get(plug, f) for f in ("x", "y", "z")])
        rpy = [state.get(plug, f) for f in ("roll", "pitch", "yaw")]
        rot = np.array(p.getMatrixFromQuaternion(
            p.getQuaternionFromEuler(rpy))).reshape(3, 3)
        tip = pos + rot @ np.array([0.0, 0.0, -cls.prong_tip_offset()])
        up = rot @ np.array([0.0, 0.0, 1.0])
        return tip, up

    @classmethod
    def _PluggedIn_holds(cls, state: State, objects: Sequence[Object]) -> bool:
        plug, outlet = objects
        tip, up = cls.plug_tip_and_axis(state, plug)
        ox, oy, oz = [state.get(outlet, f) for f in ("x", "y", "z")]
        oyaw = state.get(outlet, "yaw")
        # Express the tip in the outlet's frame.
        dx, dy = tip[0] - ox, tip[1] - oy
        c, s = np.cos(-oyaw), np.sin(-oyaw)
        lx, ly = c * dx - s * dy, s * dx + c * dy
        # The prong centre can be off the socket centre by at most the
        # clearance when it is inside; PyBullet's soft contacts add about a
        # millimetre, so allow a 2 mm margin. Depth is the robust signal: a
        # misaligned prong jams on the rim under the URDF torque limits.
        tol = cls.clearance + 0.002
        if abs(lx) > tol or abs(ly) > tol:
            return False
        top_z = oz + cls.outlet_height / 2.0
        if tip[2] > top_z - cls.insertion_depth:
            return False
        cos_tilt = float(np.clip(up[2], -1.0, 1.0))
        return cos_tilt >= np.cos(np.radians(cls.alignment_max_deg))

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
            outlet_x = 1.35
            outlet_y = 0.95
            init_dict[self._outlet] = {
                "x": outlet_x,
                "y": outlet_y,
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
