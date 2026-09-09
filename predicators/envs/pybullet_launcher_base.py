"""Observable simulation core of the launcher environment.

This module is the launcher env's BASE SIM: the spring launcher's
geometry and mechanics as a camera sees them (a plunger the robot
pushes back, which snaps home when let go; a ball in the muzzle cup; a
reload once a flown ball has stopped), the block towers, and the state
read/write of everything visible. It deliberately contains NO launch
law, no material masses, no task generation and no predicate / goal
semantics.

The boundary is the busyboard's visibility contract: when
``CFG.agent_sim_provide_base_sim_source`` is on, THIS FILE is copied
into the learning agent's sandbox as reference material, so the file
the agent reads is byte-identical to the code its base-sim rollouts
execute. Anything that would leak the learning target - how fast the
ball leaves for a given compression, how heavy each block colour is -
lives in ``pybullet_launcher.py``, never here. On the base sim the
plunger snaps home and the ball stays in the cup: a launcher that never
fires.

Physical layout (a tabletop catapult range, the robot at the near side):

- The ``launcher`` at the left (west) end of the table: a low rail
  along +x with a ``plunger`` handle sliding on it. Pushing the handle
  west compresses the launcher (``compression``, metres); when nothing
  holds it any more it snaps back to rest in one step. The rail ends in
  a muzzle whose barrel is tilted up by ``angle`` (radians, a visible
  feature): a launched ball leaves along that direction.
- The ``ball``: a small sphere resting in the muzzle cup. Once a flown
  ball has come to rest away from the cup, it is put back in the cup
  and ``balls_left`` (spare balls) drops by one; with no spare left it
  stays where it stopped.
- ``stand`` pedestals along the rail's line, each carrying a tower of
  ``block`` cubes. A block's ``color`` is its material class, a stable
  identity across levels; the tower's top block is what goals ask to
  topple and a lower block is what they ask to leave standing.
"""
from typing import Any, ClassVar, Dict, List, Tuple

import numpy as np
import pybullet as p

from predicators.envs.pybullet_env import PyBulletEnv
from predicators.pybullet_helpers.geometry import Pose3D, Quaternion
from predicators.pybullet_helpers.objects import create_object, \
    create_pybullet_block, create_pybullet_sphere, update_object
from predicators.pybullet_helpers.robots import SingleArmPyBulletRobot
from predicators.settings import CFG
from predicators.structs import Action, Object, State, Type


class PyBulletLauncherBaseEnv(PyBulletEnv):
    """Sim core of the launcher: rail, plunger, ball, stands and blocks.

    Abstract on purpose - it defines no name, predicates, tasks or
    domain-specific step, so env discovery skips it; the concrete env is
    ``PyBulletLauncherEnv``.
    """

    @classmethod
    def get_base_sim_source_files(cls) -> List[str]:
        return [
            "predicators/envs/pybullet_launcher_base.py",
            "predicators/envs/pybullet_env.py",
        ]

    # =========================================================================
    # WORKSPACE & TABLE
    # =========================================================================
    table_height: ClassVar[float] = 0.4
    table_pos: ClassVar[Pose3D] = (0.75, 1.35, table_height / 2)
    table_orn: ClassVar[Quaternion] = p.getQuaternionFromEuler(
        [0.0, 0.0, np.pi / 2.0])

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
    robot_init_x: ClassVar[float] = 0.6
    robot_init_y: ClassVar[float] = 1.05
    robot_init_z: ClassVar[float] = z_ub - 0.1
    robot_base_pos: ClassVar[Pose3D] = (0.75, 0.65, 0.0)
    robot_base_orn: ClassVar[Quaternion] = p.getQuaternionFromEuler(
        [0.0, 0.0, np.pi / 2])
    robot_init_tilt: ClassVar[float] = np.pi / 2
    robot_init_wrist: ClassVar[float] = -np.pi / 2

    # =========================================================================
    # CAMERA
    # =========================================================================
    # From the far side of the table, facing the robot, so the launcher,
    # the whole flight and the tower are in one frame with nothing of
    # the robot in the way (from the near side its torso fills the view).
    _camera_distance: ClassVar[float] = 1.45
    _camera_yaw: ClassVar[float] = 150
    _camera_pitch: ClassVar[float] = -28
    _camera_target: ClassVar[Tuple[float, float, float]] = (0.78, 1.25, 0.5)

    # =========================================================================
    # LAUNCHER LAYOUT
    # =========================================================================
    rail_y: ClassVar[float] = 1.22
    # The rail carries the handle's travel and the muzzle cup.
    rail_x_bounds: ClassVar[Tuple[float, float]] = (0.34, 0.66)
    rail_half_extents: ClassVar[Tuple[float, float,
                                      float]] = (0.16, 0.03, 0.01)
    rail_top: ClassVar[float] = table_height + 2 * rail_half_extents[2]
    rail_color: ClassVar[Tuple[float, float, float,
                               float]] = (0.30, 0.30, 0.34, 1.0)
    # The plunger handle: an upright block sliding along the rail. Its
    # rest position is the rail's midpoint; compression is how far west
    # of rest it sits.
    # Short enough that the gripper's fingers, not its body, meet it:
    # the fingertips hang 3 cm below the end-effector point and the
    # gripper body starts 3 cm above it.
    handle_half_extents: ClassVar[Tuple[float, float,
                                        float]] = (0.012, 0.02, 0.025)
    handle_rest_x: ClassVar[float] = 0.46
    handle_z: ClassVar[float] = rail_top + handle_half_extents[2]
    # The joint's travel; the push skill's depth parameter stays under
    # it so a stroke never bottoms out against the limit.
    max_compression: ClassVar[float] = 0.12
    max_push_depth: ClassVar[float] = 0.10
    # Joint friction holding the handle in place (newtons).
    handle_hold_force: ClassVar[float] = 20.0
    handle_color: ClassVar[Tuple[float, float, float,
                                 float]] = (0.85, 0.20, 0.15, 1.0)
    # Height above the handle's centre at which the push skill aims.
    handle_push_height: ClassVar[float] = 0.0
    # The muzzle: where the ball sits, and the barrel's elevation.
    muzzle_x: ClassVar[float] = 0.60
    launch_angle: ClassVar[float] = np.deg2rad(38.0)
    barrel_length: ClassVar[float] = 0.10
    barrel_color: ClassVar[Tuple[float, float, float,
                                 float]] = (0.40, 0.40, 0.44, 1.0)

    ball_radius: ClassVar[float] = 0.02
    ball_mass: ClassVar[float] = 0.08
    ball_color: ClassVar[Tuple[float, float, float,
                               float]] = (0.95, 0.80, 0.10, 1.0)
    # A ball within this distance of the cup, at rest, counts as loaded.
    cup_radius: ClassVar[float] = 0.03

    # Stands and blocks.
    stand_half_extents: ClassVar[Tuple[float, float,
                                       float]] = (0.045, 0.045, 0.03)
    stand_color: ClassVar[Tuple[float, float, float,
                                float]] = (0.55, 0.45, 0.35, 1.0)
    block_half: ClassVar[float] = 0.03
    # The mass every block has in this file. Which colours differ from
    # it is the concrete env's business.
    block_base_mass: ClassVar[float] = 0.3
    block_friction: ClassVar[float] = 0.6

    # Material palette: a block's ``color`` indexes it.
    COLOR_PALETTE: ClassVar[List[Tuple[str,
                                       Tuple[float, float, float, float]]]] = [
                                           ("wood", (0.80, 0.62, 0.35, 1.0)),
                                           ("stone", (0.45, 0.47, 0.50, 1.0)),
                                       ]
    # The top block of a tower, the one goals ask to topple, is marked.
    target_color: ClassVar[Tuple[float, float, float,
                                 float]] = (0.85, 0.15, 0.15, 1.0)

    # =========================================================================
    # TYPES
    # =========================================================================
    _robot_type = Type("robot",
                       ["x", "y", "z", "fingers", "roll", "tilt", "wrist"])
    _launcher_type = Type(
        "launcher", ["x", "y", "z", "angle", "compression", "balls_left"])
    _ball_type = Type("ball", ["x", "y", "z", "speed"])
    _stand_type = Type("stand", ["x", "y", "z"])
    # ``is_target`` marks the block a goal asks to topple (painted red).
    _block_type = Type(
        "block", ["x", "y", "z", "roll", "pitch", "yaw", "color", "is_target"])

    @classmethod
    def color_name(cls, color_index: int) -> str:
        """The palette name behind a ``color`` feature value."""
        return cls.COLOR_PALETTE[int(round(color_index))][0]

    @classmethod
    def color_rgba(cls, color_index: int) -> Tuple[float, float, float, float]:
        """The RGBA behind a ``color`` feature value."""
        return cls.COLOR_PALETTE[int(round(color_index))][1]

    @classmethod
    def _max_blocks(cls) -> int:
        return max(
            list(CFG.launcher_num_blocks_train) +
            list(CFG.launcher_num_blocks_test))

    @classmethod
    def launch_direction(cls, angle: float) -> Tuple[float, float, float]:
        """Unit vector along the barrel for a given elevation."""
        return (float(np.cos(angle)), 0.0, float(np.sin(angle)))

    @classmethod
    def cup_position(cls) -> Tuple[float, float, float]:
        """Where a loaded ball rests."""
        return (cls.muzzle_x, cls.rail_y, cls.rail_top + cls.ball_radius)

    # =========================================================================
    # CONSTRUCTION
    # =========================================================================
    def __init__(self, use_gui: bool = False, **kwargs: Any) -> None:
        self._robot = Object("robot", self._robot_type)
        self._launcher = Object("launcher", self._launcher_type)
        self._ball = Object("ball", self._ball_type)
        self._stand = Object("stand", self._stand_type)
        self._blocks: List[Object] = [
            Object(f"block{i}", self._block_type)
            for i in range(self._max_blocks())
        ]
        self._block_colors: Dict[str, int] = {}
        self._block_is_target: Dict[str, bool] = {}
        self._balls_left: int = 0
        self._compression: float = 0.0
        self._compression_before_snap: float = 0.0
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

        # The rail, static.
        x0, x1 = cls.rail_x_bounds
        bodies["rail_id"] = create_pybullet_block(
            color=cls.rail_color,
            half_extents=cls.rail_half_extents,
            mass=0.0,
            friction=0.3,
            position=((x0 + x1) / 2, cls.rail_y,
                      cls.table_height + cls.rail_half_extents[2]),
            physics_client_id=physics_client_id)

        # The plunger handle: a static base with one prismatic link
        # along x, so a push slides it and a joint reset snaps it home.
        handle_collision = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=cls.handle_half_extents,
            physicsClientId=physics_client_id)
        handle_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=cls.handle_half_extents,
            rgbaColor=cls.handle_color,
            physicsClientId=physics_client_id)
        plunger_id = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=-1,
            basePosition=(cls.handle_rest_x, cls.rail_y, cls.handle_z),
            linkMasses=[0.05],
            linkCollisionShapeIndices=[handle_collision],
            linkVisualShapeIndices=[handle_visual],
            linkPositions=[[0.0, 0.0, 0.0]],
            linkOrientations=[[0.0, 0.0, 0.0, 1.0]],
            linkInertialFramePositions=[[0.0, 0.0, 0.0]],
            linkInertialFrameOrientations=[[0.0, 0.0, 0.0, 1.0]],
            linkParentIndices=[0],
            linkJointTypes=[p.JOINT_PRISMATIC],
            linkJointAxis=[[1.0, 0.0, 0.0]],
            physicsClientId=physics_client_id)
        p.changeDynamics(plunger_id,
                         0,
                         jointLowerLimit=-cls.max_compression,
                         jointUpperLimit=0.0,
                         lateralFriction=0.4,
                         physicsClientId=physics_client_id)
        # A holding motor: the handle stays where the push leaves it
        # (a few newtons of joint friction, which the gripper overcomes
        # and a light handle does not), so it cannot coast ahead of the
        # gripper. The env resets it home once the push has let go.
        p.setJointMotorControl2(plunger_id,
                                0,
                                p.VELOCITY_CONTROL,
                                targetVelocity=0.0,
                                force=cls.handle_hold_force,
                                physicsClientId=physics_client_id)
        bodies["plunger_id"] = plunger_id

        # The barrel: a tilted decorative box at the muzzle.
        barrel_visual = p.createVisualShape(p.GEOM_BOX,
                                            halfExtents=(cls.barrel_length / 2,
                                                         0.028, 0.006),
                                            rgbaColor=cls.barrel_color,
                                            physicsClientId=physics_client_id)
        direction = np.array(cls.launch_direction(cls.launch_angle))
        cup = np.array(cls.cup_position())
        barrel_center = cup - direction * cls.barrel_length / 2 + \
            np.array([0.0, 0.0, -cls.ball_radius - 0.004])
        bodies["barrel_id"] = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=barrel_visual,
            basePosition=barrel_center.tolist(),
            baseOrientation=p.getQuaternionFromEuler(
                [0.0, -cls.launch_angle, 0.0]),
            physicsClientId=physics_client_id)

        ball_id = create_pybullet_sphere(color=cls.ball_color,
                                         radius=cls.ball_radius,
                                         mass=cls.ball_mass,
                                         friction=0.5,
                                         position=cls.cup_position(),
                                         physics_client_id=physics_client_id)
        p.changeDynamics(ball_id,
                         -1,
                         restitution=0.2,
                         rollingFriction=0.002,
                         physicsClientId=physics_client_id)
        bodies["ball_id"] = ball_id

        bodies["stand_id"] = create_pybullet_block(
            color=cls.stand_color,
            half_extents=cls.stand_half_extents,
            mass=0.0,
            friction=cls.block_friction,
            physics_client_id=physics_client_id)

        block_ids = []
        half = cls.block_half
        for _ in range(cls._max_blocks()):
            block_ids.append(
                create_pybullet_block(color=cls.COLOR_PALETTE[0][1],
                                      half_extents=(half, half, half),
                                      mass=cls.block_base_mass,
                                      friction=cls.block_friction,
                                      physics_client_id=physics_client_id))
        bodies["block_ids"] = block_ids
        return physics_client_id, pybullet_robot, bodies

    def _store_pybullet_bodies(self, pybullet_bodies: Dict[str, Any]) -> None:
        self._table_id = pybullet_bodies["table_id"]
        self._rail_id = pybullet_bodies["rail_id"]
        self._barrel_id = pybullet_bodies["barrel_id"]
        self._robot.id = self._pybullet_robot.robot_id
        self._launcher.id = pybullet_bodies["plunger_id"]
        self._ball.id = pybullet_bodies["ball_id"]
        self._stand.id = pybullet_bodies["stand_id"]
        for i, block in enumerate(self._blocks):
            block.id = pybullet_bodies["block_ids"][i]

    # =========================================================================
    # LAUNCHER MECHANICS
    # =========================================================================
    def _read_compression(self) -> float:
        """How far west of rest the handle sits, from its joint."""
        pos = p.getJointState(self._launcher.id,
                              0,
                              physicsClientId=self._physics_client_id)[0]
        return float(max(0.0, -pos))

    def _set_compression(self, compression: float) -> None:
        p.resetJointState(
            self._launcher.id,
            0,
            -float(np.clip(compression, 0.0, self.max_compression)),
            targetVelocity=0.0,
            physicsClientId=self._physics_client_id)

    # The hand counts as on the handle while its reference point is
    # within this box of the handle's push face: this far along the rail
    # on either side (the fingertip pads sit a little east of the face),
    # and this far across and above it.
    hold_box: ClassVar[Tuple[float, float, float,
                             float]] = (-0.03, 0.06, 0.05, 0.07)

    def _handle_held(self) -> bool:
        """Whether the robot's hand is at the handle.

        Judged by where the hand is, not by contact: the fingers'
        contact with a pushed handle flickers step to step, and a snap
        on a flicker would fire a barely compressed spring under the
        hand.
        """
        ex, ey, ez = self._pybullet_robot.get_state()[:3]
        handle_x = self.handle_rest_x - self._read_compression()
        west, east, across, above = self.hold_box
        return bool(
            west <= ex - handle_x <= east and abs(ey - self.rail_y) <= across
            and abs(ez - (self.handle_z + self.handle_push_height)) <= above)

    def _ball_speed(self) -> float:
        (vx, vy,
         vz), _ = p.getBaseVelocity(self._ball.id,
                                    physicsClientId=self._physics_client_id)
        return float(np.linalg.norm([vx, vy, vz]))

    def _ball_in_cup(self) -> bool:
        (bx, by, bz), _ = p.getBasePositionAndOrientation(
            self._ball.id, physicsClientId=self._physics_client_id)
        cx, cy, cz = self.cup_position()
        return bool(
            np.linalg.norm([bx - cx, by - cy, bz - cz]) < self.cup_radius)

    def _reload_ball(self) -> None:
        update_object(self._ball.id,
                      position=self.cup_position(),
                      physics_client_id=self._physics_client_id)
        p.resetBaseVelocity(self._ball.id, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0),
                            physicsClientId=self._physics_client_id)

    def _step_base(self, action: Action) -> None:
        """The visible mechanics, after the physics of every action.

        The handle snaps home the moment nothing touches it (the base
        sim shows exactly that; what the snap does to the ball is the
        concrete env's secret). A ball that has flown and stopped away
        from the cup is put back, spending a spare, while spares last.
        """
        super()._step_base(action)
        compression = self._read_compression()
        if compression > self.max_compression:
            # The end of the joint's travel, enforced here since a
            # prismatic link built by hand carries no engine limit.
            self._set_compression(self.max_compression)
            compression = self.max_compression
        # Where the handle sat at the end of this action's physics,
        # before any snap: what a camera saw this step.
        self._compression_before_snap = compression
        if compression > 0.0 and not self._handle_held():
            self._set_compression(0.0)
        self._compression = self._read_compression()
        if not self._ball_in_cup() and self._ball_speed() < \
                CFG.launcher_settle_speed and self._balls_left > 0:
            self._reload_ball()
            self._balls_left -= 1

    # =========================================================================
    # STATE READ / WRITE
    # =========================================================================
    def _get_object_ids_for_held_check(self) -> List[int]:
        """Nothing here is grasped; the handle is pushed."""
        return []

    def _active_blocks(self, state: State) -> List[Object]:
        return sorted((o for o in state if o.type.name == "block"),
                      key=lambda o: o.name)

    def _paint_block(self, block: Object) -> None:
        if self._block_is_target.get(block.name, False):
            color = self.target_color
        else:
            color = self.color_rgba(self._block_colors.get(block.name, 0))
        p.changeVisualShape(block.id,
                            -1,
                            rgbaColor=color,
                            physicsClientId=self._physics_client_id)

    def _get_domain_specific_feature(self, obj: Object, feature: str) -> float:
        if obj.type.name == "launcher":
            if feature == "angle":
                return float(self.launch_angle)
            if feature == "compression":
                return self._read_compression()
            if feature == "balls_left":
                return float(self._balls_left)
            if feature in ("x", "y", "z"):
                # The launcher's pose is its handle's rest pose: fixed.
                return {
                    "x": self.handle_rest_x,
                    "y": self.rail_y,
                    "z": self.handle_z
                }[feature]
        if obj.type.name == "ball" and feature == "speed":
            return self._ball_speed()
        if obj.type.name == "block":
            if feature == "color":
                return float(self._block_colors.get(obj.name, 0))
            if feature == "is_target":
                return float(self._block_is_target.get(obj.name, False))
        raise ValueError(f"Unknown feature {feature} for object {obj}")

    def _set_domain_specific_state(self, state: State) -> None:
        """Restore the handle, the spare count and the blocks' looks, and park
        the blocks this level does not use."""
        self._set_compression(state.get(self._launcher, "compression"))
        self._compression = self._read_compression()
        self._compression_before_snap = self._compression
        self._balls_left = int(round(state.get(self._launcher, "balls_left")))
        blocks = self._active_blocks(state)
        self._block_colors = {}
        self._block_is_target = {}
        for block in blocks:
            self._block_colors[block.name] = int(
                round(state.get(block, "color")))
            self._block_is_target[block.name] = \
                state.get(block, "is_target") > 0.5
            self._paint_block(block)
        oov_x, oov_y = self._out_of_view_xy
        for i in range(len(blocks), len(self._blocks)):
            update_object(self._blocks[i].id,
                          position=(oov_x, oov_y, 0.2 * i),
                          physics_client_id=self._physics_client_id)
