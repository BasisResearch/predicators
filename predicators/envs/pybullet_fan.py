"""envsbullet_fan module."""
from collections import deque
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pybullet as p

from predicators import utils
from predicators.envs.pybullet_env import PyBulletEnv
from predicators.pybullet_helpers.geometry import Pose3D, Quaternion
from predicators.pybullet_helpers.objects import create_object, \
    create_pybullet_block, create_pybullet_sphere, update_object
from predicators.pybullet_helpers.robots import SingleArmPyBulletRobot
from predicators.settings import CFG
from predicators.structs import Action, EnvironmentTask, GroundAtom, Object, \
    Predicate, State, Type


class PyBulletFanEnv(PyBulletEnv):
    """A PyBullet environment where a ball is blown around by fans in a
    maze."""

    # =========================================================================
    # WORKSPACE & ENVIRONMENT CONFIGURATION
    # =========================================================================

    # -------------------------------------------------------------------------
    # Table / Workspace Dimensions
    # -------------------------------------------------------------------------
    table_height: ClassVar[float] = 0.4
    table_pos: ClassVar[Pose3D] = (0.75, 1.35, table_height / 2.0)
    table_orn: ClassVar[Quaternion] = p.getQuaternionFromEuler(
        [0.0, 0.0, np.pi / 2.0])
    table_scale: ClassVar[float] = 1.0

    # Workspace bounds
    x_lb: ClassVar[float] = 0.4
    x_ub: ClassVar[float] = 1.1
    y_lb: ClassVar[float] = 1.1
    y_ub: ClassVar[float] = 1.6
    z_lb: ClassVar[float] = table_height
    z_ub: ClassVar[float] = 0.75 + table_height / 2
    init_padding: float = 0.05

    # -------------------------------------------------------------------------
    # Grid Layout Configuration
    # -------------------------------------------------------------------------
    # Grid dimensions will be set dynamically based on train/test mode
    pos_gap: ClassVar[float] = 0.08  # Distance between grid positions

    # -------------------------------------------------------------------------
    # Camera Configuration
    # -------------------------------------------------------------------------
    _camera_distance: ClassVar[float] = 1.3
    _camera_yaw: ClassVar[float] = 70
    _camera_pitch: ClassVar[float] = -50
    _camera_target: ClassVar[Tuple[float, float, float]] = (0.75, 1.25, 0.42)

    # =========================================================================
    # ROBOT CONFIGURATION
    # =========================================================================
    robot_init_x: ClassVar[float] = (x_lb + x_ub) * 0.5
    robot_init_y: ClassVar[float] = (y_lb + y_ub) * 0.4
    robot_init_z: ClassVar[float] = z_ub - 0.3
    robot_base_pos: ClassVar[Pose3D] = (0.75, 0.62, 0.0)
    robot_base_orn: ClassVar[Quaternion] = p.getQuaternionFromEuler(
        [0.0, 0.0, np.pi / 2.0])
    robot_init_tilt: ClassVar[float] = np.pi / 2.0
    robot_init_wrist: ClassVar[float] = -np.pi / 2.0

    # =========================================================================
    # FAN SYSTEM CONFIGURATION
    # =========================================================================

    # -------------------------------------------------------------------------
    # Fan Count & Layout
    # -------------------------------------------------------------------------
    num_left_fans: ClassVar[int] = 2
    num_right_fans: ClassVar[int] = 2
    num_back_fans: ClassVar[int] = 5
    num_front_fans: ClassVar[int] = 5

    # -------------------------------------------------------------------------
    # Fan Physical Properties
    # -------------------------------------------------------------------------
    fan_scale: ClassVar[float] = 0.08
    fan_x_len: ClassVar[float] = 0.2 * fan_scale  # Length of fan blades
    fan_y_len: ClassVar[float] = 1.5 * fan_scale  # Width of fan blades
    fan_z_len: ClassVar[float] = 1.5 * fan_scale  # Height of fan base

    # -------------------------------------------------------------------------
    # Fan Motor & Physics
    # -------------------------------------------------------------------------
    fan_spin_velocity: ClassVar[float] = 100.0  # Velocity for joint_0
    wind_force_magnitude: ClassVar[float] = 0.4  # Force applied to ball
    joint_motor_force: ClassVar[float] = 20.0  # Motor control force

    # -------------------------------------------------------------------------
    # Kinematic Ball Movement
    # -------------------------------------------------------------------------
    kinematic_ball_speed: ClassVar[
        float] = 0.003  # Speed for kinematic movement (m/s per simulation step)

    # -------------------------------------------------------------------------
    # Fan Positioning
    # -------------------------------------------------------------------------
    left_fan_x: ClassVar[float] = x_lb - fan_x_len * 5
    right_fan_x: ClassVar[float] = x_ub + fan_x_len * 5
    up_fan_y: ClassVar[float] = y_ub + fan_x_len / 2
    down_fan_y: ClassVar[float] = y_lb + fan_x_len / 2 + 0.1

    # Fan placement boundaries
    fan_y_lb: ClassVar[
        float] = down_fan_y + fan_x_len / 2 + fan_y_len / 2 + 0.01
    fan_y_ub: ClassVar[float] = up_fan_y - fan_x_len / 2 - fan_y_len / 2 - 0.01
    fan_x_lb: ClassVar[
        float] = left_fan_x + fan_x_len / 2 + fan_y_len / 2 + 0.01
    fan_x_ub: ClassVar[
        float] = right_fan_x - fan_x_len / 2 - fan_y_len / 2 - 0.01

    # =========================================================================
    # SWITCH CONFIGURATION
    # =========================================================================
    switch_scale: ClassVar[float] = 1.0
    switch_joint_scale: ClassVar[float] = 0.1
    switch_on_threshold: ClassVar[float] = 0.5  # Fraction of joint range
    switch_x_len: ClassVar[float] = 0.10  # Length of switch
    switch_height: ClassVar[float] = 0.08

    # Switch positioning
    switch_y: ClassVar[float] = (y_lb +
                                 y_ub) * 0.5 - 0.25  # Y position of switches
    switch_base_x: ClassVar[float] = 0.60  # Base X position for first switch
    switch_x_spacing: ClassVar[float] = 0.08  # Spacing between switches

    # =========================================================================
    # OBJECT PHYSICS CONFIGURATION
    # =========================================================================

    # -------------------------------------------------------------------------
    # Ball Properties
    # -------------------------------------------------------------------------
    ball_radius: ClassVar[float] = 0.04
    ball_mass: ClassVar[float] = 0.01
    ball_friction: ClassVar[float] = 10.0
    ball_height_offset: ClassVar[float] = ball_radius
    ball_linear_damping: ClassVar[float] = 10.0
    ball_angular_damping: ClassVar[float] = 10.0
    ball_color: ClassVar[Tuple[float, float, float,
                               float]] = (0.0, 0.0, 1.0, 1)

    # -------------------------------------------------------------------------
    # Wall Properties
    # -------------------------------------------------------------------------
    # Obstacle walls
    num_walls: ClassVar[int] = 4
    # wall_x_len: ClassVar[float] = 0.05
    # wall_y_len: ClassVar[float] = 0.04
    wall_x_len: ClassVar[float] = pos_gap - 0.02
    wall_y_len: ClassVar[float] = pos_gap - 0.02
    obstacle_wall_height: ClassVar[float] = 0.02
    # wall_x_len: ClassVar[float] = pos_gap - 0.03
    # wall_y_len: ClassVar[float] = pos_gap - 0.03
    # obstacle_wall_height: ClassVar[float] = 0.01
    wall_rot: ClassVar[float] = 0.0  # can be np.py/2
    wall_mass: ClassVar[float] = 0.0
    wall_friction: ClassVar[float] = 0.0
    wall_color: ClassVar[Tuple[float, float, float,
                               float]] = (0.5, 0.5, 0.5, 1.0)

    # Boundary walls around grid
    boundary_wall_height: ClassVar[float] = 0.03
    boundary_wall_thickness: ClassVar[float] = 0.002
    boundary_wall_color: ClassVar[Tuple[float, float, float,
                                        float]] = (0.9, 0.9, 0.9, 1)

    # -------------------------------------------------------------------------
    # Target Properties
    # -------------------------------------------------------------------------
    target_thickness: ClassVar[float] = 0.00001
    target_mass: ClassVar[float] = 0.0
    target_friction: ClassVar[float] = 0.04
    target_color: ClassVar[Tuple[float, float, float, float]] = (0, 1, 0, 1.0)

    # =========================================================================
    # SIMULATION & DEBUG CONFIGURATION
    # =========================================================================

    # -------------------------------------------------------------------------
    # Visual/Debug Parameters
    # -------------------------------------------------------------------------
    debug_line_height: ClassVar[float] = 0.2
    debug_line_lifetime: ClassVar[float] = 0.2

    # -------------------------------------------------------------------------
    # Task Generation Parameters
    # -------------------------------------------------------------------------
    # num_walls_per_task will be set dynamically based on train/test mode
    position_tolerance: ClassVar[float] = 0.01

    # =========================================================================
    # DERIVED/CALCULATED VALUES
    # =========================================================================
    # Note: These are calculated from the above parameters
    loc_y_lb, loc_y_ub = down_fan_y + 0.05, up_fan_y - 0.05
    loc_x_lb, loc_x_ub = left_fan_x + 0.05, right_fan_x - 0.05
    loc_x_mid = (loc_x_lb + loc_x_ub) * 0.5
    loc_y_mid = (loc_y_lb + loc_y_ub) * 0.5

    # -------------------------------------------------------------------------
    # Types
    # -------------------------------------------------------------------------
    _robot_type = Type("robot", ["x", "y", "z", "fingers", "tilt", "wrist"])
    _fan_type = Type(
        "fan",
        [
            "x",  # fan base x
            "y",  # fan base y
            "z",  # fan base z
            "rot",  # base orientation (Z euler)
            "facing_side",  # 0=left,1=right,2=back,3=front
            "is_on",  # whether the controlling switch is on
        ],
        sim_features=["id", "side_idx", "fan_ids", "joint_ids"])
    # New separate switch type:
    _switch_type = Type(
        "switch",
        [
            "x",
            "y",
            "z",
            "rot",  # switch orientation
            "controls_fan",  # matches fan side
            "is_on",  # is this switch on
        ],
        sim_features=["id", "joint_id", "side_idx"])
    _wall_type = Type("wall", ["x", "y", "z", "rot"])
    _ball_type = Type("ball", ["x", "y", "z"])
    _target_type = Type("target", ["x", "y", "z", "rot", "is_hit"])
    _location_type = Type("loc", ["xx", "yy"], sim_features=["id", "xx", "yy"])
    _side_type = Type("side", ["side_idx"], sim_features=["id", "side_idx"])

    @classmethod
    def get_configuration_dict(cls) -> Dict[str, Any]:
        """Return all configuration parameters as a dictionary."""
        config = {}

        # Get all ClassVar attributes
        for attr_name in dir(cls):
            if not attr_name.startswith('_') and hasattr(cls, attr_name):
                attr_value = getattr(cls, attr_name)
                if isinstance(attr_value, (int, float, str, tuple, list)):
                    config[attr_name] = attr_value

        return config

    # -------------------------------------------------------------------------
    # Environment initialization
    # -------------------------------------------------------------------------
    def __init__(self, use_gui: bool = False, **kwargs: Any) -> None:
        self._robot = Object("robot", self._robot_type)

        # Fans - create one fan object per side instead of multiple
        self._fans: List[Object] = []
        self._switch_sides = ["left", "right", "down", "up"]
        for i, side_str in enumerate(self._switch_sides):
            # Create one fan object per side (left=0, right=1, down=2, up=3)
            fan_obj = Object(f"fan_{i}", self._fan_type)
            self._fans.append(fan_obj)

        # Switches: now each is a distinct object of _switch_type
        self._switches: List[Object] = []
        for i, side_str in enumerate(self._switch_sides):
            # Create a switch object using the new _switch_type
            switch_obj = Object(f"switch_{i}", self._switch_type)
            self._switches.append(switch_obj)

        # Sides: representing the four directional sides
        self._sides: List[Object] = []
        for side_str in self._switch_sides:
            side_obj = Object(f"{side_str}", self._side_type)
            self._sides.append(side_obj)

        # Maze walls - create enough for the maximum walls per task
        max_walls_per_task = max(max(CFG.fan_train_num_walls_per_task),
                                 max(CFG.fan_test_num_walls_per_task))
        self._walls = [
            Object(f"wall{i}", self._wall_type)
            for i in range(max_walls_per_task)
        ]
        # Create positions based on maximum grid size to support both train and
        # test
        self.pos_dict: Dict[Object, Dict[str, float]] = {}

        # Grid positions will be set dynamically in task generation

        # Ball
        self._ball = Object("ball", self._ball_type)

        # Target
        self._target = Object("target", self._target_type)

        super().__init__(use_gui=use_gui, **kwargs)

        # Define new predicates if desired
        self._FanOn = Predicate(
            "FanOn", [self._fan_type],
            self._FanOn_holds,
            natural_language_assertion=lambda os: f"fan {os[0]} is on")
        self._FanOff = Predicate(
            "FanOff", [self._fan_type],
            lambda s, o: not self._FanOn_holds(s, o),
            natural_language_assertion=lambda os: f"fan {os[0]} is off")
        self._SwitchOn = Predicate("SwitchOn", [self._switch_type],
                                   self._FanOn_holds)
        self._SwitchOff = Predicate("SwitchOff", [self._switch_type],
                                    lambda s, o: not self._FanOn_holds(s, o))
        self._BallAtLoc = Predicate("BallAtLoc",
                                    [self._ball_type, self._location_type],
                                    self._BallAtLoc_holds,
                                    natural_language_assertion=lambda os:
                                    f"ball {os[0]} is at location {os[1]}")
        self._ClearLoc = Predicate("ClearLoc", [self._location_type],
                                   self._ClearPos_holds,
                                   natural_language_assertion=lambda os:
                                   f"location {os[0]} is clear of objects")
        self._FanFacingSide = Predicate(
            "FanFacingSide", [self._fan_type, self._side_type],
            self._FanFacingSide_holds,
            natural_language_assertion=lambda os:
            f"fan {os[0]} is facing the side {os[1]}")
        self._OppositeFan = Predicate(
            "OppositeFan", [self._fan_type, self._fan_type],
            self._OppositeFan_holds,
            natural_language_assertion=lambda os:
            f"fan {os[0]} is facing the opposite side of fan {os[1]}")
        self._SideOf = Predicate(
            "SideOf",
            [self._location_type, self._location_type, self._side_type],
            self._SideOf_holds,
            natural_language_assertion=lambda os:
            f"location {os[0]} is to the {os[2]} side of location {os[1]}")
        self._Controls = Predicate("Controls",
                                   [self._switch_type, self._fan_type],
                                   self._Controls_holds)
        # self._LeftOf = Predicate("LeftOf",
        #                          [self._location_type, self._location_type],
        #                          self._LeftOf_holds)
        # self._RightOf = Predicate("RightOf",
        #                           [self._location_type, self._location_type],
        #                           self._RightOf_holds)
        # self._UpOf = Predicate("UpOf",
        #                        [self._location_type, self._location_type],
        #                        self._UpOf_holds)
        # self._DownOf = Predicate("DownOf",
        #                          [self._location_type, self._location_type],
        #                          self._DownOf_holds)
        # self._LeftFan = Predicate("LeftFan", [self._fan_type],
        #                           self._LeftFanSwitch_holds)
        # self._RightFan = Predicate("RightFan", [self._fan_type],
        #                            self._RightFanSwitch_holds)
        # self._FrontFan = Predicate("FrontFan", [self._fan_type],
        #                            self._FrontFanSwitch_holds)
        # self._BackFan = Predicate("BackFan", [self._fan_type],
        #                           self._BackFanSwitch_holds)
        # self._LeftFanSwitch = Predicate("LeftFanSwitch", [self._switch_type],
        #                                 self._LeftFanSwitch_holds)
        # self._RightFanSwitch = Predicate(
        #     "RightFanSwitch", [self._switch_type],
        #     self._RightFanSwitch_holds)
        # self._FrontFanSwitch = Predicate(
        #     "FrontFanSwitch", [self._switch_type],
        #     self._FrontFanSwitch_holds)
        # self._BackFanSwitch = Predicate("BackFanSwitch", [self._switch_type],
        #                                 self._BackFanSwitch_holds)

    @classmethod
    def get_name(cls) -> str:
        return "pybullet_fan"

    @property
    def predicates(self) -> Set[Predicate]:
        predicates = {
            self._FanOn,
            self._FanOff,
            self._BallAtLoc,
            # self._LeftOf, self._RightOf, self._UpOf, self._DownOf,
            # self._LeftFanSwitch, self._RightFanSwitch, self._FrontFanSwitch,
            # self._BackFanSwitch,
            self._FanFacingSide,
            self._SideOf,
            self._Controls,
            self._ClearLoc,
            self._OppositeFan,
        }
        if not CFG.fan_known_controls_relation:
            predicates |= {self._SwitchOn, self._SwitchOff}
        return predicates

    @property
    def target_predicates(self) -> Set[Predicate]:
        return {
            self._FanFacingSide,
            # self._OppositeFan
        }

    @property
    def types(self) -> Set[Type]:
        return {
            self._robot_type, self._fan_type, self._switch_type,
            self._wall_type, self._ball_type, self._target_type,
            self._location_type, self._side_type
        }

    @property
    def goal_predicates(self) -> Set[Predicate]:
        return {self._BallAtLoc}

    # -------------------------------------------------------------------------
    # PyBullet Initialization
    # -------------------------------------------------------------------------
    @classmethod
    def initialize_pybullet(
            cls, using_gui: bool
    ) -> Tuple[int, SingleArmPyBulletRobot, Dict[str, Any]]:
        physics_client_id, pybullet_robot, bodies = super(
        ).initialize_pybullet(using_gui)

        # Create a table
        table_id = create_object(
            asset_path="urdf/table.urdf",
            position=cls.table_pos,
            orientation=cls.table_orn,
            scale=cls.table_scale,
            use_fixed_base=True,
            physics_client_id=physics_client_id,
        )
        bodies["table_id"] = table_id

        # ---------------------------------------------------------------------
        # Create fans in four groups: left, right, back, front
        # We'll store them in the dictionary as fan_ids_left, fan_ids_right, ...
        # ---------------------------------------------------------------------
        fan_urdf = "urdf/partnet_mobility/fan/101450/mobility.urdf"

        left_fan_ids = []
        for _ in range(cls.num_left_fans):
            fid = create_object(asset_path=fan_urdf,
                                scale=cls.fan_scale,
                                use_fixed_base=True,
                                physics_client_id=physics_client_id)
            left_fan_ids.append(fid)

        right_fan_ids = []
        for _ in range(cls.num_right_fans):
            fid = create_object(asset_path=fan_urdf,
                                scale=cls.fan_scale,
                                use_fixed_base=True,
                                physics_client_id=physics_client_id)
            right_fan_ids.append(fid)

        back_fan_ids = []
        for _ in range(cls.num_back_fans):
            fid = create_object(asset_path=fan_urdf,
                                scale=cls.fan_scale,
                                use_fixed_base=True,
                                physics_client_id=physics_client_id)
            back_fan_ids.append(fid)

        front_fan_ids = []
        for _ in range(cls.num_front_fans):
            fid = create_object(asset_path=fan_urdf,
                                scale=cls.fan_scale,
                                use_fixed_base=True,
                                physics_client_id=physics_client_id)
            front_fan_ids.append(fid)

        bodies["fan_ids_left"] = left_fan_ids
        bodies["fan_ids_right"] = right_fan_ids
        bodies["fan_ids_back"] = back_fan_ids
        bodies["fan_ids_front"] = front_fan_ids

        # ---------------------------------------------------------------------
        # Create 4 switches at the requested positions
        #   order: left=0, right=1, back=2, front=3
        # ---------------------------------------------------------------------
        switch_urdf = "urdf/partnet_mobility/switch/102812/switch.urdf"
        switch_ids = []
        for _ in range(4):
            sid = create_object(
                asset_path=switch_urdf,
                # position=(sx, sy, cls.table_height),
                # orientation=p.getQuaternionFromEuler(
                #     [0, 0, srot]),
                scale=cls.switch_scale,
                use_fixed_base=True,
                physics_client_id=physics_client_id)
            switch_ids.append(sid)
        bodies["switch_ids"] = switch_ids

        # ---------------------------------------------------------------------
        # Maze walls
        # ---------------------------------------------------------------------
        max_walls_per_task = max(max(CFG.fan_train_num_walls_per_task),
                                 max(CFG.fan_test_num_walls_per_task))
        wall_ids = []
        for _ in range(max_walls_per_task):
            wall_id = create_pybullet_block(
                color=cls.wall_color,
                half_extents=(cls.wall_x_len / 2, cls.wall_y_len / 2,
                              cls.obstacle_wall_height / 2),
                mass=cls.wall_mass,
                friction=cls.wall_friction,
                position=(0.75, 1.28,
                          cls.table_height + cls.obstacle_wall_height / 2),
                orientation=p.getQuaternionFromEuler([0, 0, 0]),
                physics_client_id=physics_client_id)
            wall_ids.append(wall_id)
        bodies["wall_ids"] = wall_ids

        # ---------------------------------------------------------------------
        # Create the ball
        # ---------------------------------------------------------------------
        ball_id = create_pybullet_sphere(
            color=cls.ball_color,
            radius=cls.ball_radius,
            mass=cls.ball_mass,
            friction=cls.ball_friction,
            position=(0.75, 1.35, cls.table_height + cls.ball_height_offset),
            orientation=p.getQuaternionFromEuler([0, 0, 0]),
            physics_client_id=physics_client_id)
        p.changeDynamics(ball_id,
                         -1,
                         linearDamping=cls.ball_linear_damping,
                         angularDamping=cls.ball_angular_damping,
                         physicsClientId=physics_client_id)
        bodies["ball_id"] = ball_id

        # ---------------------------------------------------------------------
        # Create the target
        # ---------------------------------------------------------------------
        target_id = create_pybullet_block(
            color=(0, 1, 0, 1.0),
            half_extents=(cls.pos_gap / 2, cls.pos_gap / 2,
                          cls.target_thickness),
            mass=cls.target_mass,
            friction=cls.target_friction,
            position=(0, 0, cls.table_height),
            orientation=p.getQuaternionFromEuler([0, 0, 0]),
            physics_client_id=physics_client_id)
        bodies["target_id"] = target_id

        return physics_client_id, pybullet_robot, bodies

    @staticmethod
    def _get_joint_id(obj_id: int,
                      joint_name: str,
                      physics_client_id: int = 0) -> int:
        num_joints = p.getNumJoints(obj_id, physicsClientId=physics_client_id)
        for j in range(num_joints):
            info = p.getJointInfo(obj_id, j, physicsClientId=physics_client_id)
            if info[1].decode("utf-8") == joint_name:
                return j
        return -1

    def _store_pybullet_bodies(self, pybullet_bodies: Dict[str, Any]) -> None:
        """Store references to all PyBullet object IDs and their joints."""
        self._table_ids = [pybullet_bodies["table_id"]]
        # 0 = left, 1 = right, 2 = back, 3 = front

        # Store all fan IDs grouped by side
        fan_ids_by_side = [
            pybullet_bodies["fan_ids_left"],  # side 0
            pybullet_bodies["fan_ids_right"],  # side 1
            pybullet_bodies["fan_ids_back"],  # side 2
            pybullet_bodies["fan_ids_front"]  # side 3
        ]

        # Update each fan object with its side's fan IDs and joint IDs
        for side_idx, fan_obj in enumerate(self._fans):
            fan_obj.side_idx = side_idx
            fan_obj.fan_ids = fan_ids_by_side[side_idx]
            fan_obj.joint_ids = [
                self._get_joint_id(fid, "joint_0", self._physics_client_id)
                for fid in fan_obj.fan_ids
            ]
            # Assign an arbitrary ID from the fans on this side (use the first
            # one)
            fan_obj.id = fan_obj.fan_ids[0] if fan_obj.fan_ids else -1

        # Switches
        for i, switch_obj in enumerate(self._switches):
            switch_obj.id = pybullet_bodies["switch_ids"][i]
            switch_obj.joint_id = self._get_joint_id(switch_obj.id, "joint_0",
                                                     self._physics_client_id)
            switch_obj.side_idx = i  # 0=left,1=right,2=back,3=front

        # Sides (no PyBullet bodies, just assign IDs for consistency)
        self._sides[0].side_idx = 1.0
        self._sides[1].side_idx = 0.0
        self._sides[2].side_idx = 3.0
        self._sides[3].side_idx = 2.0
        # for i, side_obj in enumerate(self._sides):
        #     # side_obj.id = -1  # No PyBullet body
        #     side_obj.side_idx = i  # 0=left,1=right,2=back,3=front

        for wall, obj_id in zip(self._walls, pybullet_bodies["wall_ids"]):
            wall.id = obj_id
        self._ball.id = pybullet_bodies["ball_id"]
        self._target.id = pybullet_bodies["target_id"]

        # Initialize boundary wall IDs list (will be populated
        # in _set_domain_specific_state)
        # pylint: disable=attribute-defined-outside-init
        self._boundary_wall_ids: List[int] = []

    # -------------------------------------------------------------------------
    # Read state from PyBullet
    # -------------------------------------------------------------------------
    def _get_object_ids_for_held_check(self) -> List[int]:
        return []

    def _set_domain_specific_state(self, state: State) -> None:
        for switch_obj in self._switches:
            is_on_val = state.get(switch_obj, "is_on")
            self._set_switch_on(switch_obj.id, bool(is_on_val > 0.5))

        # Position all fans correctly based on their side
        self._position_fans_on_sides()

        # Reposition boundary walls based on actual grid positions in the state
        self._reposition_boundary_walls(state)

        oov_x, oov_y = self._out_of_view_xy
        # Move irrelavent walls oov
        wall_obj = state.get_objects(self._wall_type)
        for i in range(len(wall_obj), len(self._walls)):
            update_object(self._walls[i].id,
                          position=(oov_x, oov_y, 0.0),
                          physics_client_id=self._physics_client_id)

    def _reposition_boundary_walls(self, state: State) -> None:
        """Recreate boundary walls with correct dimensions based on the actual
        grid positions used in this task."""
        # Remove existing boundary walls
        for wall_id in self._boundary_wall_ids:
            if wall_id >= 0:
                p.removeBody(wall_id, physicsClientId=self._physics_client_id)
        # pylint: disable=attribute-defined-outside-init
        self._boundary_wall_ids = []

        # Get all position objects that are in the state
        position_objects = state.get_objects(self._location_type)
        if not position_objects:
            return

        # Set the xx, yy sim features
        for pos_obj in position_objects:
            pos_obj.xx = state.get(pos_obj, "xx")
            pos_obj.yy = state.get(pos_obj, "yy")

        # Find the bounds of the actual grid positions used in this task
        x_coords = [state.get(pos_obj, "xx") for pos_obj in position_objects]
        y_coords = [state.get(pos_obj, "yy") for pos_obj in position_objects]

        grid_x_min, grid_x_max = min(x_coords), max(x_coords)
        grid_y_min, grid_y_max = min(y_coords), max(y_coords)

        # Create boundary walls with correct dimensions for this grid
        # Left boundary wall (pos_gap to the left of leftmost grid positions)
        left_wall_x = grid_x_min - self.pos_gap / 2
        left_wall_y = (grid_y_min + grid_y_max) / 2
        left_wall_id = create_pybullet_block(
            color=self.boundary_wall_color,
            half_extents=(self.boundary_wall_thickness / 2,
                          (grid_y_max - grid_y_min + self.pos_gap) / 2,
                          self.boundary_wall_height / 2),
            mass=self.wall_mass,
            friction=self.wall_friction,
            position=(left_wall_x, left_wall_y,
                      self.table_height + self.boundary_wall_height / 2),
            orientation=p.getQuaternionFromEuler([0, 0, 0]),
            physics_client_id=self._physics_client_id)

        # Right boundary wall (pos_gap to the right of rightmost grid positions)
        right_wall_x = grid_x_max + self.pos_gap / 2
        right_wall_y = (grid_y_min + grid_y_max) / 2
        right_wall_id = create_pybullet_block(
            color=self.boundary_wall_color,
            half_extents=(self.boundary_wall_thickness / 2,
                          (grid_y_max - grid_y_min + self.pos_gap) / 2,
                          self.boundary_wall_height / 2),
            mass=self.wall_mass,
            friction=self.wall_friction,
            position=(right_wall_x, right_wall_y,
                      self.table_height + self.boundary_wall_height / 2),
            orientation=p.getQuaternionFromEuler([0, 0, 0]),
            physics_client_id=self._physics_client_id)

        # Front boundary wall (pos_gap in front of front grid positions)
        front_wall_x = (grid_x_min + grid_x_max) / 2
        front_wall_y = grid_y_min - self.pos_gap / 2
        front_wall_id = create_pybullet_block(
            color=self.boundary_wall_color,
            half_extents=((grid_x_max - grid_x_min + self.pos_gap) / 2,
                          self.boundary_wall_thickness / 2,
                          self.boundary_wall_height / 2),
            mass=self.wall_mass,
            friction=self.wall_friction,
            position=(front_wall_x, front_wall_y,
                      self.table_height + self.boundary_wall_height / 2),
            orientation=p.getQuaternionFromEuler([0, 0, 0]),
            physics_client_id=self._physics_client_id)

        # Back boundary wall (pos_gap behind back grid positions)
        back_wall_x = (grid_x_min + grid_x_max) / 2
        back_wall_y = grid_y_max + self.pos_gap / 2
        back_wall_id = create_pybullet_block(
            color=self.boundary_wall_color,
            half_extents=((grid_x_max - grid_x_min + self.pos_gap) / 2,
                          self.boundary_wall_thickness / 2,
                          self.boundary_wall_height / 2),
            mass=self.wall_mass,
            friction=self.wall_friction,
            position=(back_wall_x, back_wall_y,
                      self.table_height + self.boundary_wall_height / 2),
            orientation=p.getQuaternionFromEuler([0, 0, 0]),
            physics_client_id=self._physics_client_id)

        # Store the new boundary wall IDs
        # pylint: disable=attribute-defined-outside-init
        self._boundary_wall_ids = [
            left_wall_id, right_wall_id, front_wall_id, back_wall_id
        ]

    @classmethod
    def _generate_grid_coordinates(
            cls, num_pos_x: int,
            num_pos_y: int) -> Tuple[List[float], List[float]]:
        """Generate grid coordinates for the maze with specified dimensions."""
        if num_pos_x % 2 == 1:
            x_start = cls.loc_x_mid - (num_pos_x - 1) * cls.pos_gap / 2
        else:
            x_start = (cls.loc_x_mid - num_pos_x * cls.pos_gap / 2 +
                       cls.pos_gap / 2)

        if num_pos_y % 2 == 1:
            y_start = (cls.loc_y_mid - (num_pos_y - 1) * cls.pos_gap / 2)
        else:
            y_start = (cls.loc_y_mid - num_pos_y * cls.pos_gap / 2 +
                       cls.pos_gap / 2)

        x_coords = [x_start + i * cls.pos_gap for i in range(num_pos_x)]
        y_coords = [y_start + i * cls.pos_gap for i in range(num_pos_y)]

        # Assertions to ensure coordinates don't go beyond bounds
        assert min(x_coords) >= cls.loc_x_lb, (
            f"Minimum x coordinate {min(x_coords)} "
            f"is below lower bound {cls.loc_x_lb}")
        assert max(x_coords) <= cls.loc_x_ub, (
            f"Maximum x coordinate {max(x_coords)} "
            f"is above upper bound {cls.loc_x_ub}")
        assert min(y_coords) >= cls.loc_y_lb, (
            f"Minimum y coordinate {min(y_coords)} "
            f"is below lower bound {cls.loc_y_lb}")
        assert max(y_coords) <= cls.loc_y_ub, (
            f"Maximum y coordinate {max(y_coords)} "
            f"is above upper bound {cls.loc_y_ub}")

        return x_coords, y_coords

    def _position_fans_on_sides(self) -> None:
        """Position all PyBullet fan bodies correctly on their respective
        sides."""
        # Calculate positions for each side
        left_coords = np.linspace(self.fan_y_lb, self.fan_y_ub,
                                  self.num_left_fans)
        right_coords = np.linspace(self.fan_y_lb, self.fan_y_ub,
                                   self.num_right_fans)
        front_coords = np.linspace(self.fan_x_lb, self.fan_x_ub,
                                   self.num_front_fans)
        back_coords = np.linspace(self.fan_x_lb, self.fan_x_ub,
                                  self.num_back_fans)

        # Position fans for each side
        for fan_obj in self._fans:
            side_idx = fan_obj.side_idx
            fan_ids = fan_obj.fan_ids

            if side_idx == 0:  # left
                for i, fan_id in enumerate(fan_ids):
                    px = self.left_fan_x
                    py = left_coords[i] if i < len(
                        left_coords) else left_coords[-1]
                    pz = self.table_height + self.fan_z_len / 2
                    rot = [0.0, 0.0, 0.0]  # facing right
                    update_object(fan_id,
                                  position=(px, py, pz),
                                  orientation=p.getQuaternionFromEuler(rot),
                                  physics_client_id=self._physics_client_id)

            elif side_idx == 1:  # right
                for i, fan_id in enumerate(fan_ids):
                    px = self.right_fan_x
                    py = right_coords[i] if i < len(
                        right_coords) else right_coords[-1]
                    pz = self.table_height + self.fan_z_len / 2
                    rot = [0.0, 0.0, np.pi]  # facing left
                    update_object(fan_id,
                                  position=(px, py, pz),
                                  orientation=p.getQuaternionFromEuler(rot),
                                  physics_client_id=self._physics_client_id)

            elif side_idx == 2:  # back
                for i, fan_id in enumerate(fan_ids):
                    px = back_coords[i] if i < len(
                        back_coords) else back_coords[-1]
                    py = self.down_fan_y
                    pz = self.table_height + self.fan_z_len / 2
                    rot = [0.0, 0.0, np.pi / 2]  # facing forward
                    update_object(fan_id,
                                  position=(px, py, pz),
                                  orientation=p.getQuaternionFromEuler(rot),
                                  physics_client_id=self._physics_client_id)

            elif side_idx == 3:  # front
                for i, fan_id in enumerate(fan_ids):
                    px = front_coords[i] if i < len(
                        front_coords) else front_coords[-1]
                    py = self.up_fan_y
                    pz = self.table_height + self.fan_z_len / 2
                    rot = [0.0, 0.0, -np.pi / 2]  # facing backward
                    update_object(fan_id,
                                  position=(px, py, pz),
                                  orientation=p.getQuaternionFromEuler(rot),
                                  physics_client_id=self._physics_client_id)

    def _get_domain_specific_feature(self, obj: Object, feature: str) -> float:
        """Extract features for creating the State object."""
        if obj.type == self._fan_type:
            if feature == "facing_side":
                return float(obj.side_idx)
            if feature == "is_on":
                controlling_switch = self._switches[obj.side_idx]
                return float(self._is_switch_on(controlling_switch.id))
        if obj.type == self._switch_type:
            if feature == "controls_fan":
                return float(obj.side_idx)
            if feature == "is_on":
                return float(self._is_switch_on(obj.id))
        if obj.type == self._target_type:
            if feature == "is_hit":
                ball_pos, _ = p.getBasePositionAndOrientation(
                    self._ball.id, physicsClientId=self._physics_client_id)
                target_pos, _ = p.getBasePositionAndOrientation(
                    self._target.id, physicsClientId=self._physics_client_id)
                bx, by = ball_pos[0], ball_pos[1]
                tx, ty = target_pos[0], target_pos[1]
                return 1.0 if self._is_ball_close_to_position(bx, by, tx,
                                                              ty) else 0.0
        if obj.type == self._location_type:
            if feature == "xx":
                return obj.xx
            if feature == "yy":
                return obj.yy
        if obj.type == self._side_type:
            if feature == "side_idx":
                return float(obj.side_idx)

        raise ValueError(f"Unknown feature {feature} for object {obj}")

    # -------------------------------------------------------------------------
    # Step
    # -------------------------------------------------------------------------
    def _domain_specific_step(self) -> None:
        """Spin fans & blow the ball."""
        self._simulate_fans()
        state = self._get_state()
        # Draw a debug line at the ball's position
        bx, by = state.get(self._ball, "x"), state.get(self._ball, "y")
        p.addUserDebugLine(
            [bx, by, self.table_height],
            [bx, by, self.table_height + self.debug_line_height],
            [0, 1, 0],
            lifeTime=self.
            debug_line_lifetime,  # short lifetime so each step refreshes
            physicsClientId=self._physics_client_id)

    # -------------------------------------------------------------------------
    # Fan Simulation
    # -------------------------------------------------------------------------
    def _simulate_fans(self) -> None:
        """Spin any switched-on fans and blow the ball."""
        if CFG.fan_use_kinematic:
            self._simulate_fans_kinematic()
        else:
            self._simulate_fans_dynamic()

    def _simulate_fans_dynamic(self) -> None:
        """Original dynamic fan simulation using forces."""
        # For each switch, if on => spin all fans with same side_idx
        for ctrl_fan_idx, switch_obj in enumerate(self._switches):
            on = self._is_switch_on(switch_obj.id)
            fan_obj = self._fans[
                ctrl_fan_idx]  # Get the single fan object for this side

            # Check if fan_ids attribute exists and is populated
            if not hasattr(fan_obj, 'fan_ids') or not fan_obj.fan_ids:
                continue

            if on and fan_obj.fan_ids:  # Apply force
                # Control all physical fans for this side
                for i, fan_id in enumerate(fan_obj.fan_ids):
                    joint_id = fan_obj.joint_ids[i]
                    if joint_id >= 0:
                        p.setJointMotorControl2(
                            bodyUniqueId=fan_id,
                            jointIndex=joint_id,
                            controlMode=p.VELOCITY_CONTROL,
                            targetVelocity=self.fan_spin_velocity,
                            force=self.joint_motor_force,
                            physicsClientId=self._physics_client_id,
                        )
                # Apply force using the first fan in the group for direction
                self._apply_fan_force_to_ball(fan_obj.fan_ids[0],
                                              self._ball.id)
            else:
                # Turn off all physical fans for this side
                for i, fan_id in enumerate(fan_obj.fan_ids):
                    joint_id = fan_obj.joint_ids[i]
                    if joint_id >= 0:
                        p.setJointMotorControl2(
                            bodyUniqueId=fan_id,
                            jointIndex=joint_id,
                            controlMode=p.VELOCITY_CONTROL,
                            targetVelocity=0.0,
                            force=self.joint_motor_force,
                            physicsClientId=self._physics_client_id,
                        )

    def _simulate_fans_kinematic(self) -> None:
        """Kinematic fan simulation using position-based movement."""
        # Get current ball position
        ball_pos, ball_orn = p.getBasePositionAndOrientation(
            self._ball.id, physicsClientId=self._physics_client_id)
        ball_x, ball_y, ball_z = ball_pos

        # Calculate movement vector based on active fans
        movement_x = 0.0
        movement_y = 0.0

        # Check each fan and accumulate movement vectors
        for ctrl_fan_idx, switch_obj in enumerate(self._switches):
            on = self._is_switch_on(switch_obj.id)
            fan_obj = self._fans[ctrl_fan_idx]

            # Check if fan_ids attribute exists and is populated
            if not hasattr(fan_obj, 'fan_ids') or not fan_obj.fan_ids:
                continue

            if on and fan_obj.fan_ids:
                # Still spin the fans visually
                for i, fan_id in enumerate(fan_obj.fan_ids):
                    joint_id = fan_obj.joint_ids[i]
                    if joint_id >= 0:
                        p.setJointMotorControl2(
                            bodyUniqueId=fan_id,
                            jointIndex=joint_id,
                            controlMode=p.VELOCITY_CONTROL,
                            targetVelocity=self.fan_spin_velocity,
                            force=self.joint_motor_force,
                            physicsClientId=self._physics_client_id,
                        )

                # Add movement based on fan direction
                if ctrl_fan_idx == 0:  # left fan - push right
                    movement_x += self.kinematic_ball_speed
                elif ctrl_fan_idx == 1:  # right fan - push left
                    movement_x -= self.kinematic_ball_speed
                elif ctrl_fan_idx == 2:  # back fan - push forward (up in y)
                    movement_y += self.kinematic_ball_speed
                elif ctrl_fan_idx == 3:  # front fan - push backward (down in y)
                    movement_y -= self.kinematic_ball_speed
            else:
                # Turn off fans visually
                for i, fan_id in enumerate(fan_obj.fan_ids):
                    joint_id = fan_obj.joint_ids[i]
                    if joint_id >= 0:
                        p.setJointMotorControl2(
                            bodyUniqueId=fan_id,
                            jointIndex=joint_id,
                            controlMode=p.VELOCITY_CONTROL,
                            targetVelocity=0.0,
                            force=self.joint_motor_force,
                            physicsClientId=self._physics_client_id,
                        )

        # Apply the accumulated movement by setting ball position
        if movement_x != 0.0 or movement_y != 0.0:
            new_x = ball_x + movement_x
            new_y = ball_y + movement_y

            # Keep the ball within workspace bounds
            new_x = max(self.x_lb, min(self.x_ub, new_x))
            new_y = max(self.y_lb, min(self.y_ub, new_y))

            # Set the new ball position directly
            p.resetBasePositionAndOrientation(
                self._ball.id,
                posObj=[new_x, new_y, ball_z],
                ornObj=ball_orn,
                physicsClientId=self._physics_client_id)

    def _apply_fan_force_to_ball(self, fan_id: int, ball_id: int) -> None:
        """Compute the direction the fan blows (+X in fan local frame) and
        apply force."""
        _, orn_fan = p.getBasePositionAndOrientation(fan_id,
                                                     self._physics_client_id)

        if CFG.fan_fans_blow_opposite_direction:
            local_dir = np.array([-1.0, 0.0, 0.0])
        else:
            local_dir = np.array([1.0, 0.0, 0.0])  # +X is "forward"
        rmat = np.array(p.getMatrixFromQuaternion(orn_fan)).reshape((3, 3))
        world_dir = rmat.dot(local_dir)
        pos_ball, _ = p.getBasePositionAndOrientation(ball_id,
                                                      self._physics_client_id)
        force_vec = self.wind_force_magnitude * world_dir
        p.applyExternalForce(
            objectUniqueId=ball_id,
            linkIndex=-1,
            forceObj=force_vec.tolist(),
            posObj=pos_ball,
            flags=p.WORLD_FRAME,
            physicsClientId=self._physics_client_id,
        )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------
    def _is_switch_on(self, switch_id: int) -> bool:
        """Check if a switch's joint is above the threshold."""
        joint_id = self._get_joint_id(switch_id, "joint_0",
                                      self._physics_client_id)
        if joint_id < 0:
            return False
        j_pos, _, _, _ = p.getJointState(
            switch_id, joint_id, physicsClientId=self._physics_client_id)
        info = p.getJointInfo(switch_id,
                              joint_id,
                              physicsClientId=self._physics_client_id)
        j_min, j_max = info[8], info[9]
        frac = (j_pos / self.switch_joint_scale - j_min) / (j_max - j_min)
        return bool(frac > self.switch_on_threshold)

    def _set_switch_on(self, switch_id: int, power_on: bool) -> None:
        """Programmatically toggle a switch on/off."""
        joint_id = self._get_joint_id(switch_id, "joint_0",
                                      self._physics_client_id)
        if joint_id < 0:
            return
        info = p.getJointInfo(switch_id,
                              joint_id,
                              physicsClientId=self._physics_client_id)
        j_min, j_max = info[8], info[9]
        target_val = j_max if power_on else j_min
        p.resetJointState(
            switch_id,
            joint_id,
            target_val * self.switch_joint_scale,
            physicsClientId=self._physics_client_id,
        )

    def _is_ball_close_to_position(self, bx: float, by: float, tx: float,
                                   ty: float) -> bool:
        """Check if the ball is close to the target."""
        return np.abs(bx - tx) < self.pos_gap / 2 and \
                np.abs(by - ty) < self.pos_gap / 2

    # -------------------------------------------------------------------------
    # Predicates
    # -------------------------------------------------------------------------
    @staticmethod
    def _FanOn_holds(state: State, objects: Sequence[Object]) -> bool:
        """(FanOn fan).

        True if the controlling switch is on.
        """
        (fan, ) = objects
        return state.get(fan, "is_on") > 0.5

    def _BallAtLoc_holds(self, state: State,
                         objects: Sequence[Object]) -> bool:
        ball, pos = objects
        # walls = state.get_objects(self._wall_type)
        # for wall in walls:
        #     if self._is_ball_close_to_position(state.get(wall, "x"),
        #                                        state.get(wall, "y"),
        #                                        state.get(pos, "xx"),
        #                                        state.get(pos, "yy")):
        #         return False
        return self._is_ball_close_to_position(state.get(ball, "x"),
                                               state.get(ball, "y"),
                                               state.get(pos, "xx"),
                                               state.get(pos, "yy"))

    def _LeftOf_holds(self, state: State, objects: Sequence[Object]) -> bool:
        pos1, pos2 = objects
        return self._is_ball_close_to_position(
            state.get(pos1, "xx") + self.pos_gap, state.get(pos1, "yy"),
            state.get(pos2, "xx"), state.get(pos2, "yy"))

    def _RightOf_holds(self, state: State, objects: Sequence[Object]) -> bool:
        pos1, pos2 = objects
        return self._is_ball_close_to_position(
            state.get(pos1, "xx") - self.pos_gap, state.get(pos1, "yy"),
            state.get(pos2, "xx"), state.get(pos2, "yy"))

    def _UpOf_holds(self, state: State, objects: Sequence[Object]) -> bool:
        pos1, pos2 = objects
        return self._is_ball_close_to_position(
            state.get(pos1, "xx"),
            state.get(pos1, "yy") - self.pos_gap, state.get(pos2, "xx"),
            state.get(pos2, "yy"))

    def _DownOf_holds(self, state: State, objects: Sequence[Object]) -> bool:
        """(DownOf pos1 pos2)."""
        pos1, pos2 = objects
        return self._is_ball_close_to_position(
            state.get(pos1, "xx"),
            state.get(pos1, "yy") + self.pos_gap, state.get(pos2, "xx"),
            state.get(pos2, "yy"))

    def _ClearPos_holds(self, state: State, objects: Sequence[Object]) -> bool:
        """If the position is clear of walls."""

        pos, = objects
        pos_x, pos_y = state.get(pos, "xx"), state.get(pos, "yy")
        for obj in state.get_objects(self._wall_type):
            wx, wy = state.get(obj, "x"), state.get(obj, "y")
            if self._is_ball_close_to_position(pos_x, pos_y, wx, wy):
                return False
        return True

    def _LeftFanSwitch_holds(self, state: State,
                             objects: Sequence[Object]) -> bool:
        switch, = objects
        return state.get(switch, "controls_fan") == 0

    def _RightFanSwitch_holds(self, state: State,
                              objects: Sequence[Object]) -> bool:
        switch, = objects
        return state.get(switch, "controls_fan") == 1

    def _FrontFanSwitch_holds(self, state: State,
                              objects: Sequence[Object]) -> bool:
        switch, = objects
        return state.get(switch, "controls_fan") == 3

    def _BackFanSwitch_holds(self, state: State,
                             objects: Sequence[Object]) -> bool:
        switch, = objects
        return state.get(switch, "controls_fan") == 2

    def _FanFacingSide_holds(self, state: State,
                             objects: Sequence[Object]) -> bool:
        """Whether the fan is on the specified side of the table.

        True if the fan's side matches the side object's side.
        """
        fan, side = objects
        return state.get(fan, "facing_side") == state.get(side, "side_idx")

    def _OppositeFan_holds(self, state: State,
                           objects: Sequence[Object]) -> bool:
        """Whether the fans are on opposite sides of the table."""
        fan1, fan2 = objects
        if fan1.name == fan2.name:
            return False
        side1 = state.get(fan1, "facing_side")
        side2 = state.get(fan2, "facing_side")
        # Check if they are on opposite sides using XOR
        # Sides 0,1 are opposite (differ by 1), sides 2,3 are opposite (differ
        # by 1)
        return abs(side1 - side2) == 1 and (side1 // 2) == (side2 // 2)

    def _SideOf_holds(self, state: State, objects: Sequence[Object]) -> bool:
        """(SideOf pos1 pos2 side).

        True if pos1 is in the specified side direction relative to
        pos2. side=0 (left): pos1 is to the left of pos2 side=1 (right):
        pos1 is to the right of pos2 side=2 (back): pos1 is above pos2
        side=3 (front): pos1 is below pos2
        """
        pos1, pos2, side = objects
        side_val = state.get(side, "side_idx")

        if side_val == 1:  # left
            return self._is_ball_close_to_position(
                state.get(pos1, "xx") + self.pos_gap, state.get(pos1, "yy"),
                state.get(pos2, "xx"), state.get(pos2, "yy"))
        if side_val == 0:  # right
            return self._is_ball_close_to_position(
                state.get(pos1, "xx") - self.pos_gap, state.get(pos1, "yy"),
                state.get(pos2, "xx"), state.get(pos2, "yy"))
        if side_val == 2:  # down
            return self._is_ball_close_to_position(
                state.get(pos1, "xx"),
                state.get(pos1, "yy") - self.pos_gap, state.get(pos2, "xx"),
                state.get(pos2, "yy"))
        if side_val == 3:  # up
            return self._is_ball_close_to_position(
                state.get(pos1, "xx"),
                state.get(pos1, "yy") + self.pos_gap, state.get(pos2, "xx"),
                state.get(pos2, "yy"))
        return False

    def _Controls_holds(self, state: State, objects: Sequence[Object]) -> bool:
        """(Controls fan switch)."""
        # Note: this probably needs to be updated.
        switch, fan = objects
        return state.get(fan,
                         "facing_side") == state.get(switch, "controls_fan")

    # -------------------------------------------------------------------------
    # Task Generation
    # -------------------------------------------------------------------------
    def _generate_train_tasks(self) -> List[EnvironmentTask]:
        return self._make_tasks(
            num_tasks=CFG.num_train_tasks,
            num_pos_x=CFG.fan_train_num_pos_x,
            num_pos_y=CFG.fan_train_num_pos_y,
            possible_num_walls_per_task=CFG.fan_train_num_walls_per_task,
            rng=self._train_rng)

    def _generate_test_tasks(self) -> List[EnvironmentTask]:
        return self._make_tasks(
            num_tasks=CFG.num_test_tasks,
            num_pos_x=CFG.fan_test_num_pos_x,
            num_pos_y=CFG.fan_test_num_pos_y,
            possible_num_walls_per_task=CFG.fan_test_num_walls_per_task,
            rng=self._test_rng)

    def _make_tasks(  # pylint: disable=redefined-outer-name
            self, num_tasks: int, num_pos_x: int, num_pos_y: int,
            possible_num_walls_per_task: List[int],
            rng: np.random.Generator) -> List[EnvironmentTask]:
        # Generate grid coordinates for this specific configuration
        x_coords, y_coords = self._generate_grid_coordinates(
            num_pos_x, num_pos_y)
        grid_pos = [(x, y) for y in y_coords for x in x_coords]
        _positions = [
            Object(f"loc_y{i}_x{j}", self._location_type)
            for i in range(num_pos_y) for j in range(num_pos_x)
        ]

        # Create position dictionary for this task configuration
        pos_dict = {}
        pos_index = 0
        for i in range(num_pos_y):
            for j in range(num_pos_x):
                if pos_index < len(_positions):
                    pos_obj = _positions[pos_index]
                    pos_dict[pos_obj] = {"xx": x_coords[j], "yy": y_coords[i]}
                    pos_index += 1

        # Draw debug lines for positions if debug is enabled
        if CFG.pybullet_draw_debug:
            for pos_obj, pos in pos_dict.items():
                p.addUserDebugLine([pos["xx"], pos["yy"], self.table_height], [
                    pos["xx"], pos["yy"],
                    self.table_height + self.debug_line_height
                ], [1, 0, 0],
                                   parentObjectUniqueId=-1,
                                   parentLinkIndex=-1)

        tasks = []  # pylint: disable=redefined-outer-name
        for _ in range(num_tasks):
            # Try to generat a valid task with path validation
            max_attempts = 100  # Prevent infinite loop
            for attempt in range(max_attempts):
                # Sample the number of walls for this task
                num_walls_per_task = rng.choice(possible_num_walls_per_task)
                available_pos = grid_pos.copy()

                # Robot
                robot_dict = {
                    "x": self.robot_init_x,
                    "y": self.robot_init_y,
                    "z": self.robot_init_z,
                    "fingers": self.open_fingers,
                    "tilt": self.robot_init_tilt,
                    "wrist": self.robot_init_wrist,
                }

                # For 3x3 grid, ensure ball starts at edge position (not center)
                if num_pos_x == 3 and num_pos_y == 3:
                    # Edge positions in 3x3 grid: exclude center position
                    center_pos = (x_coords[1], y_coords[1])  # Center position
                    edge_positions = [
                        pos for pos in available_pos if pos != center_pos
                    ]

                    # Ball position: choose from edge positions only
                    ball_pos = tuple(rng.choice(edge_positions))
                    # Safely remove the ball position
                    available_pos.remove(ball_pos)

                    # Choose target to create alignment (same row or column as
                    # ball)
                    aligned_targets = []

                    # Same row targets (horizontal alignment) - 2 steps away
                    for x in x_coords:
                        candidate_pos = (x, ball_pos[1])
                        if (candidate_pos in available_pos
                                and candidate_pos != ball_pos
                                and abs(x - ball_pos[0]) > 1.5 * self.pos_gap):
                            aligned_targets.append(candidate_pos)

                    # Same column targets (vertical alignment) - 2 steps away
                    for y in y_coords:
                        candidate_pos = (ball_pos[0], y)
                        if (candidate_pos in available_pos
                                and candidate_pos != ball_pos
                                and abs(y - ball_pos[1]) > 1.5 * self.pos_gap):
                            aligned_targets.append(candidate_pos)

                    if not aligned_targets:
                        # Fallback to any available position
                        aligned_targets = [
                            pos for pos in available_pos if pos != ball_pos
                        ]

                    tar_pos = tuple(rng.choice(aligned_targets))
                    # Safely remove the target position
                    available_pos.remove(tar_pos)
                    target_dict = {
                        "x": tar_pos[0],
                        "y": tar_pos[1],
                        "z": self.table_height,
                        "rot": 0.0,
                        "is_hit": 0.0,
                    }

                    # Strategic wall placement to block direct path
                    wall_positions = []
                    if num_walls_per_task > 0:
                        # Place wall to block direct path between ball and
                        # target
                        blocking_pos = self._get_strategic_wall_position(
                            ball_pos, tar_pos, x_coords, y_coords,
                            available_pos, rng)
                        if blocking_pos is not None:
                            wall_positions.append(blocking_pos)
                            # Safely remove the blocking position
                else:
                    # Original logic for non-3x3 grids
                    # Target
                    tar_pos = tuple(rng.choice(available_pos))
                    available_pos.remove(tar_pos)
                    target_dict = {
                        "x": tar_pos[0],
                        "y": tar_pos[1],
                        "z": self.table_height,
                        "rot": 0.0,
                        "is_hit": 0.0,
                    }

                    # Place walls and collect their grid positions
                    wall_positions = []
                    for i in range(num_walls_per_task):
                        wall_pos = tuple(rng.choice(available_pos))
                        available_pos.remove(wall_pos)
                        wall_positions.append(wall_pos)

                    # Ball position
                    ball_pos = tuple(rng.choice(available_pos))
                    available_pos.remove(ball_pos)

                # Convert continuous positions to grid indices for path
                # validation
                tar_grid_idx = None
                ball_grid_idx = None
                wall_grid_indices = set()

                # Find grid indices for target
                for i, y in enumerate(y_coords):
                    for j, x in enumerate(x_coords):
                        if np.isclose(x, tar_pos[0],
                                atol=self.position_tolerance) and \
                           np.isclose(y, tar_pos[1],
                                atol=self.position_tolerance):
                            tar_grid_idx = (j, i)
                            break
                    if tar_grid_idx is not None:
                        break

                # Find grid indices for ball
                for i, y in enumerate(y_coords):
                    for j, x in enumerate(x_coords):
                        if np.isclose(x, ball_pos[0],
                                atol=self.position_tolerance) and \
                           np.isclose(y, ball_pos[1],
                                atol=self.position_tolerance):
                            ball_grid_idx = (j, i)
                            break
                    if ball_grid_idx is not None:
                        break

                # Find grid indices for walls
                for wall_pos in wall_positions:
                    for i, y in enumerate(y_coords):
                        for j, x in enumerate(x_coords):
                            if np.isclose(x, wall_pos[0],
                                    atol=self.position_tolerance) and \
                               np.isclose(y, wall_pos[1],
                                    atol=self.position_tolerance):
                                wall_grid_indices.add((j, i))
                                break

                # Check if we have a valid path from ball to target
                if tar_grid_idx is not None and ball_grid_idx is not None and \
                   self._has_valid_path(ball_grid_idx, tar_grid_idx,
                   wall_grid_indices, num_pos_x, num_pos_y):
                    # Valid path found, create the task

                    init_dict = {}
                    init_dict[self._robot] = robot_dict
                    init_dict[self._target] = target_dict

                    for fan_obj in self._fans:
                        # Each fan_obj now represents all fans on one side
                        side_idx = fan_obj.side_idx
                        # Set position based on the center or representative
                        # position for the side
                        if side_idx == 2:  # down
                            px = (self.fan_x_lb +
                                  self.fan_x_ub) / 2  # center of back fans
                            py = self.down_fan_y
                            rot = np.pi / 2
                        elif side_idx == 3:  # up
                            px = (self.fan_x_lb +
                                  self.fan_x_ub) / 2  # center of front fans
                            py = self.up_fan_y
                            rot = -np.pi / 2
                        elif side_idx == 0:  # left
                            px = self.left_fan_x
                            py = (self.fan_y_lb +
                                  self.fan_y_ub) / 2  # center of left fans
                            rot = 0.0
                        else:  # right (side_idx == 1)
                            px = self.right_fan_x
                            py = (self.fan_y_lb +
                                  self.fan_y_ub) / 2  # center of right fans
                            rot = np.pi
                        fan_dict = {
                            "x": px,
                            "y": py,
                            "z": self.table_height + self.fan_z_len / 2,
                            "rot": rot,
                            "facing_side": float(side_idx),
                            "is_on": 0.0
                        }
                        init_dict[fan_obj] = fan_dict

                    # Switches default off
                    for switch_obj in self._switches:
                        init_dict[switch_obj] = {
                            "x":
                            self.switch_base_x +
                            self.switch_x_spacing * switch_obj.side_idx,
                            "y":
                            self.switch_y,
                            "z":
                            self.table_height,
                            "rot":
                            np.pi / 2,
                            "controls_fan":
                            float(switch_obj.side_idx),
                            "is_on":
                            0.0,
                        }

                    # Sides - add them to the state dictionary
                    init_dict[self._sides[0]] = {"side_idx": 1.0}
                    init_dict[self._sides[1]] = {"side_idx": 0.0}
                    init_dict[self._sides[2]] = {"side_idx": 3.0}
                    init_dict[self._sides[3]] = {"side_idx": 2.0}

                    # Walls
                    for i, wall_pos in enumerate(wall_positions):
                        init_dict[self._walls[i]] = {
                            "x": wall_pos[0],
                            "y": wall_pos[1],
                            "z":
                            self.table_height + self.obstacle_wall_height / 2,
                            "rot": rng.uniform(-self.wall_rot, self.wall_rot)
                        }

                    # Ball
                    ball_dict = {
                        "x": ball_pos[0],
                        "y": ball_pos[1],
                        "z": self.table_height + self.ball_height_offset
                    }
                    init_dict[self._ball] = ball_dict

                    init_dict.update(pos_dict)
                    break
            else:
                # If we couldn't find a valid configuration after max attempts
                raise ValueError(
                    f"Could not generate a valid task configuration after "
                    f"{max_attempts} attempts")
            print(f"Found a valid task after {attempt} attempts")

            init_state = utils.create_state_from_dict(init_dict)

            # The positions that has the same coord as the target
            tx, ty = init_state.get(self._target, "x"), \
                    init_state.get(self._target, "y")
            target_pos_obj = None
            for pos_obj in pos_dict:
                px, py = init_state.get(pos_obj, "xx"), \
                    init_state.get(pos_obj, "yy")
                if np.isclose(px, tx, atol=self.position_tolerance) and \
                    np.isclose(py, ty, atol=self.position_tolerance):
                    target_pos_obj = pos_obj
                    break

            # Ensure we found a target position
            if target_pos_obj is None:
                raise ValueError("Could not find target position object")

            goal_atoms = {
                GroundAtom(self._BallAtLoc, [self._ball, target_pos_obj]),
            }
            # all fans are off in the goal
            for fan_obj in self._fans:
                goal_atoms.add(GroundAtom(self._FanOff, [fan_obj]))
            tasks.append(EnvironmentTask(init_state, goal_atoms))
        return self._add_pybullet_state_to_tasks(tasks)

    def _get_strategic_wall_position(  # pylint: disable=redefined-outer-name
            self, ball_pos: Tuple[float, float],
            target_pos: Tuple[float, float], _x_coords: List[float],
            _y_coords: List[float], available_pos: List[Tuple[float, float]],
            rng: np.random.Generator) -> Optional[Tuple[float, float]]:
        """Get a wall position that is between the ball and target."""
        # Find positions that are between ball and target
        between_positions = []

        for pos in available_pos:
            # Check if position is between ball and target (on same row or
            # column)
            if (pos[0] == ball_pos[0] == target_pos[0] and  # Same column
                    min(ball_pos[1], target_pos[1]) < pos[1] < max(
                        ball_pos[1], target_pos[1])):
                between_positions.append(pos)
            elif (pos[1] == ball_pos[1] == target_pos[1] and  # Same row
                  min(ball_pos[0], target_pos[0]) < pos[0] < max(
                      ball_pos[0], target_pos[0])):
                between_positions.append(pos)

        # Return a random position between ball and target, or random if none
        # found
        if between_positions:
            return rng.choice(between_positions)
        return tuple(rng.choice(available_pos)) if available_pos else None

    def _has_valid_path(self, start_pos: Tuple[int,
                                               int], target_pos: Tuple[int,
                                                                       int],
                        blocked_positions: Set[Tuple[int, int]],
                        num_pos_x: int, num_pos_y: int) -> bool:
        """Check if there's a valid path from start to target using only
        cardinal directions."""
        if start_pos == target_pos:
            return True

        # BFS to find path using only cardinal directions
        queue = deque([start_pos])
        visited = {start_pos}

        # Cardinal directions: up, down, left, right
        directions = [(0, 1), (0, -1), (-1, 0), (1, 0)]

        while queue:
            current_x, current_y = queue.popleft()

            for dx, dy in directions:
                next_x, next_y = current_x + dx, current_y + dy

                # Check bounds
                if not (0 <= next_x < num_pos_x and 0 <= next_y < num_pos_y):
                    continue

                # Check if position is blocked or already visited
                if (next_x,
                        next_y) in blocked_positions or (next_x,
                                                         next_y) in visited:
                    continue

                # Check if we reached the target
                if (next_x, next_y) == target_pos:
                    return True

                visited.add((next_x, next_y))
                queue.append((next_x, next_y))

        return False


if __name__ == "__main__":
    import time  # pylint: disable=ungrouped-imports
    CFG.seed = 0
    CFG.env = "pybullet_fan"
    env = PyBulletFanEnv(use_gui=True)
    _rng = np.random.default_rng(CFG.seed)
    _tasks = env._make_tasks(  # pylint: disable=protected-access
        10, CFG.fan_train_num_pos_x, CFG.fan_train_num_pos_y,
        CFG.fan_train_num_walls_per_task, _rng)

    for _task in _tasks:
        env._set_state(_task.init)  # pylint: disable=protected-access
        for _ in range(5000):
            _action = Action(
                np.array(env._pybullet_robot  # pylint: disable=protected-access
                         .initial_joint_positions))
            env.step(_action)
            time.sleep(0.1)
