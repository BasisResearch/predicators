"""PyBullet environment combining dominoes and fan-blown ball.

This environment merges features from pybullet_domino.py and pybullet_fan.py:
- Dominoes that can topple through collisions
- A ball that can be blown by fans (fans do NOT affect dominoes)
- Fans controlled by switches
- Obstacle walls blocking both ball and domino placement
- Optional grid-based layout for both dominoes and ball
- All assets (fans, switches, ball, dominoes) are loaded in every task
- Tasks have ONE goal type: either all dominoes toppled OR ball at location

Example usage:
python predicators/main.py --approach oracle --env pybullet_domino_fan \\
--seed 0 --num_test_tasks 1 --use_gui --debug --num_train_tasks 0
"""

import time
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pybullet as p

from predicators import utils
from predicators.envs.pybullet_env import PyBulletEnv, create_pybullet_block, \
    create_pybullet_sphere
from predicators.pybullet_helpers.geometry import Pose3D, Quaternion
from predicators.pybullet_helpers.objects import create_object, update_object
from predicators.pybullet_helpers.robots import SingleArmPyBulletRobot
from predicators.settings import CFG
from predicators.structs import Action, DerivedPredicate, EnvironmentTask, \
    GroundAtom, Object, Predicate, State, Type


class PyBulletDominoGridFanEnv(PyBulletEnv):
    """A PyBullet environment with dominoes and a fan-blown ball.

    Combines features from both domino and fan environments:
    - Dominoes topple when pushed (from domino env)
    - Ball is blown by fans (from fan env)
    - Fans do NOT affect dominoes
    - All assets (fans, switches, ball, dominoes) are present in every task
    - Tasks can have domino goals OR ball goals (one type per task)
    """

    # =========================================================================
    # WORKSPACE & ENVIRONMENT CONFIGURATION
    # =========================================================================

    # Table / Workspace Dimensions
    table_height: ClassVar[float] = 0.4
    table_pos: ClassVar[Pose3D] = (0.75, 1.35, table_height / 2.0)
    table_orn: ClassVar[Quaternion] = p.getQuaternionFromEuler(
        [0.0, 0.0, np.pi / 2.0])
    table_scale: ClassVar[float] = 1.0
    table_width: ClassVar[float] = 1.0

    # Workspace bounds
    x_lb: ClassVar[float] = 0.4
    x_ub: ClassVar[float] = 1.1
    y_lb: ClassVar[float] = 1.1
    y_ub: ClassVar[float] = 1.6
    z_lb: ClassVar[float] = table_height
    z_ub: ClassVar[float] = 0.75 + table_height / 2
    init_padding: float = 0.05

    # Grid Layout Configuration (optional via CFG)
    # Note: pos_gap is defined below after domino_width

    # Camera Configuration
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

    # Fan Count & Layout
    num_left_fans: ClassVar[int] = 5
    num_right_fans: ClassVar[int] = 5
    num_back_fans: ClassVar[int] = 5
    num_front_fans: ClassVar[int] = 5

    # Fan Physical Properties
    fan_scale: ClassVar[float] = 0.08
    fan_x_len: ClassVar[float] = 0.2 * fan_scale
    fan_y_len: ClassVar[float] = 1.5 * fan_scale
    fan_z_len: ClassVar[float] = 1.5 * fan_scale

    # Fan Motor & Physics
    fan_spin_velocity: ClassVar[float] = 100.0
    wind_force_magnitude: ClassVar[float] = 0.4
    joint_motor_force: ClassVar[float] = 20.0

    # Kinematic Ball Movement
    kinematic_ball_speed: ClassVar[
        float] = 0.003  # Speed for kinematic movement (m/s per simulation step)

    # Fan Positioning
    left_fan_x: ClassVar[float] = x_lb - fan_x_len * 5
    right_fan_x: ClassVar[float] = x_ub + fan_x_len * 5
    up_fan_y: ClassVar[float] = y_ub + table_width / 2 + fan_x_len / 2
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
    switch_on_threshold: ClassVar[float] = 0.5
    switch_x_len: ClassVar[float] = 0.10
    switch_height: ClassVar[float] = 0.08

    # Switch positioning
    switch_y: ClassVar[float] = (y_lb + y_ub) * 0.5 - 0.25
    switch_base_x: ClassVar[float] = 0.60
    switch_x_spacing: ClassVar[float] = 0.08

    # =========================================================================
    # BALL CONFIGURATION
    # =========================================================================
    ball_radius: ClassVar[float] = 0.04
    ball_mass: ClassVar[float] = 0.01
    ball_friction: ClassVar[float] = 10.0
    ball_height_offset: ClassVar[float] = ball_radius
    ball_linear_damping: ClassVar[float] = 10.0
    ball_angular_damping: ClassVar[float] = 10.0
    ball_color: ClassVar[Tuple[float, float, float,
                               float]] = (0.0, 0.0, 1.0, 1)

    # =========================================================================
    # DOMINO CONFIGURATION
    # =========================================================================

    # Domino Shape
    domino_width: ClassVar[float] = 0.07
    domino_depth: ClassVar[float] = 0.015
    domino_height: ClassVar[float] = 0.15
    domino_mass: ClassVar[float] = 0.1
    domino_friction: ClassVar[float] = 0.5
    turn_shift_frac: ClassVar[float] = 0.6  # For turning dominoes
    pos_gap: ClassVar[
        float] = domino_width * 1.4  # Distance between grid positions 0.07 * 1.4=0.098

    # Domino Colors
    start_domino_color: ClassVar[Tuple[float, float, float,
                                       float]] = (0.56, 0.93, 0.56, 1.)
    target_domino_color: ClassVar[Tuple[float, float, float,
                                        float]] = (0.85, 0.7, 0.85, 1.0)
    domino_color: ClassVar[Tuple[float, float, float,
                                 float]] = (0.6, 0.8, 1.0, 1.0)
    glued_domino_color: ClassVar[Tuple[float, float, float,
                                       float]] = (1.0, 0.0, 0.0, 1.0)

    # Domino Physics Thresholds
    domino_roll_threshold: ClassVar[float] = np.deg2rad(5)
    fallen_threshold: ClassVar[float] = np.pi * 2 / 5  # 72 degrees

    # =========================================================================
    # WALL CONFIGURATION
    # =========================================================================

    # Obstacle walls
    wall_x_len: ClassVar[float] = pos_gap - 0.02
    wall_y_len: ClassVar[float] = pos_gap - 0.02
    obstacle_wall_height: ClassVar[float] = 0.02
    wall_mass: ClassVar[float] = 0.0
    wall_friction: ClassVar[float] = 0.0
    wall_color: ClassVar[Tuple[float, float, float,
                               float]] = (0.5, 0.5, 0.5, 1.0)

    # Domino stairs configuration
    stair_height_increment: ClassVar[float] = 0.005
    stair_color: ClassVar[Tuple[float, float, float,
                                float]] = (0.7, 0.6, 0.5, 1.0)

    # Boundary walls
    boundary_wall_height: ClassVar[float] = 0.03
    boundary_wall_thickness: ClassVar[float] = 0.002
    boundary_wall_color: ClassVar[Tuple[float, float, float,
                                        float]] = (0.9, 0.9, 0.9, 1)

    # =========================================================================
    # TARGET CONFIGURATION
    # =========================================================================
    target_thickness: ClassVar[float] = 0.00001
    target_mass: ClassVar[float] = 0.0
    target_friction: ClassVar[float] = 0.04
    target_color: ClassVar[Tuple[float, float, float, float]] = (0, 1, 0, 1.0)

    # =========================================================================
    # SIMULATION & DEBUG CONFIGURATION
    # =========================================================================
    debug_line_height: ClassVar[float] = 0.2
    debug_line_lifetime: ClassVar[float] = 0.2
    position_tolerance: ClassVar[float] = 0.01

    # =========================================================================
    # DERIVED/CALCULATED VALUES
    # =========================================================================
    loc_y_lb, loc_y_ub = down_fan_y + 0.05, up_fan_y - 0.05
    loc_x_lb, loc_x_ub = left_fan_x + 0.05, right_fan_x - 0.05
    loc_x_mid = (loc_x_lb + loc_x_ub) * 0.5
    loc_y_mid = (loc_y_lb + loc_y_ub) * 0.5

    # =========================================================================
    # TYPE DEFINITIONS
    # =========================================================================

    # Robot type
    _robot_type = Type("robot", ["x", "y", "z", "fingers", "tilt", "wrist"])

    # Fan & control types
    _fan_type = Type(
        "fan",
        [
            "x",
            "y",
            "z",
            "rot",
            "facing_side",  # 0=left,1=right,2=back,3=front
            "is_on",
        ],
        sim_features=["id", "side_idx", "fan_ids", "joint_ids"])
    _switch_type = Type(
        "switch",
        [
            "x",
            "y",
            "z",
            "rot",
            "controls_fan",  # matches fan side
            "is_on",
        ],
        sim_features=["id", "joint_id", "side_idx"])
    _side_type = Type("side", ["side_idx"], sim_features=["id", "side_idx"])

    # Ball types
    _ball_type = Type("ball", ["x", "y", "z"])
    _target_type = Type("target", ["x", "y", "z", "rot", "is_hit"])

    # Domino types
    _domino_type = Type(
        "domino",
        ["x", "y", "z", "yaw", "roll", "r", "g", "b", "is_held"],
    )

    # Wall type
    _wall_type = Type("wall", ["x", "y", "z", "rot"])

    # Location type (grid positions)
    _location_type = Type("loc", ["xx", "yy"], sim_features=["id", "xx", "yy"])

    # Direction types (for domino placement)
    _direction_type = Type("direction", ["dir"])
    _angle_type = Type("angle", ["angle"])

    def __init__(self, use_gui: bool = True) -> None:
        """Initialize the domino-fan environment."""

        # Robot
        self._robot = Object("robot", self._robot_type)

        # Fans - one fan object per side
        self._fans: List[Object] = []
        self._switch_sides = ["left", "right", "down", "up"]
        for i, side_str in enumerate(self._switch_sides):
            fan_obj = Object(f"fan_{i}", self._fan_type)
            self._fans.append(fan_obj)

        # Switches - one per side
        self._switches: List[Object] = []
        for i, side_str in enumerate(self._switch_sides):
            switch_obj = Object(f"switch_{i}", self._switch_type)
            self._switches.append(switch_obj)

        # Sides - directional sides
        self._sides: List[Object] = []
        for side_str in self._switch_sides:
            side_obj = Object(f"{side_str}", self._side_type)
            self._sides.append(side_obj)

        # Ball
        self._ball = Object("ball", self._ball_type)

        # Target (for ball)
        self._target = Object("target", self._target_type)

        # Dominoes - create enough for maximum
        max_dominos = max(max(CFG.domino_fan_train_num_dominos),
                          max(CFG.domino_fan_test_num_dominos))
        self._dominoes: List[Object] = []
        for i in range(max_dominos):
            domino_obj = Object(f"domino_{i}", self._domino_type)
            self._dominoes.append(domino_obj)

        # Walls
        max_walls = max(max(CFG.domino_fan_train_num_walls),
                        max(CFG.domino_fan_test_num_walls))
        self._walls: List[Object] = []
        for i in range(max_walls):
            wall_obj = Object(f"wall{i}", self._wall_type)
            self._walls.append(wall_obj)

        # Direction objects (for domino placement)
        self._directions: List[Object] = []
        direction_names = ["straight", "left", "right"]
        for i, name in enumerate(direction_names):
            obj = Object(name, self._direction_type)
            self._directions.append(obj)

        # Rotation objects (for grid-based placement)
        if CFG.domino_fan_use_grid:
            self._rotations: List[Object] = []
            angle_values = [-135, -90, -45, 0, 45, 90, 135, 180]
            for angle in angle_values:
                name = f"ang_{angle}"
                obj = Object(name, self._angle_type)
                self._rotations.append(obj)
        else:
            self._rotations = []

        # Storage for boundary walls, stairs, and debug lines
        self._boundary_wall_ids: List[int] = []
        self._stair_ids: List[int] = []
        self._debug_line_ids: List[int] = []

        # Call parent init
        super().__init__(use_gui=use_gui)

        # =====================================================================
        # PREDICATES
        # =====================================================================

        # Fan-related predicates
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
        self._FanFacingSide = Predicate("FanFacingSide",
                                        [self._fan_type, self._side_type],
                                        self._FanFacingSide_holds)
        self._OppositeFan = Predicate("OppositeFan",
                                      [self._fan_type, self._fan_type],
                                      self._OppositeFan_holds)
        self._Controls = Predicate("Controls",
                                   [self._switch_type, self._fan_type],
                                   self._Controls_holds)

        # Ball-related predicates
        self._BallAtLoc = Predicate("BallAtLoc",
                                    [self._ball_type, self._location_type],
                                    self._BallAtLoc_holds)

        # Location predicates
        self._ClearLoc = Predicate("ClearLoc", [self._location_type],
                                   self._ClearLoc_holds)
        self._SideOf = Predicate(
            "SideOf",
            [self._location_type, self._location_type, self._side_type],
            self._SideOf_holds)

        # Domino-related predicates
        self._Toppled = Predicate("Toppled", [self._domino_type],
                                  self._Toppled_holds)
        self._Upright = Predicate("Upright", [self._domino_type],
                                  self._Upright_holds)
        self._Tilting = Predicate("Tilting", [self._domino_type],
                                  self._Tilting_holds)
        self._InitialBlock = Predicate("InitialBlock", [self._domino_type],
                                       self._InitialBlock_holds)
        self._MovableBlock = Predicate("MovableBlock", [self._domino_type],
                                       self._MovableBlock_holds)
        self._HandEmpty = Predicate("HandEmpty", [self._robot_type],
                                    self._HandEmpty_holds)
        self._Holding = Predicate("Holding",
                                  [self._robot_type, self._domino_type],
                                  self._Holding_holds)

        # Grid-based predicates (if using grid)
        if CFG.domino_fan_use_grid:
            self._DominoAtPos = Predicate(
                "DominoAtPos", [self._domino_type, self._location_type],
                self._DominoAtPos_holds)
            self._DominoAtRot = Predicate(
                "DominoAtRot", [self._domino_type, self._angle_type],
                self._DominoAtRot_holds)
            self._PosClear = Predicate("PosClear", [self._location_type],
                                       self._PosClear_holds)
            self._InFrontDirection = DerivedPredicate(
                "InFrontDirection",
                [self._domino_type, self._domino_type, self._direction_type],
                self._InFrontDirection_holds,
                auxiliary_predicates={self._DominoAtPos, self._DominoAtRot})
            self._InFront = DerivedPredicate(
                "InFront", [self._domino_type, self._domino_type],
                self._InFront_holds,
                auxiliary_predicates={self._InFrontDirection})

        # Glued domino predicate (if enabled)
        if CFG.domino_fan_has_glued_dominoes:
            self._DominoNotGlued = Predicate("DominoNotGlued",
                                             [self._domino_type],
                                             self._DominoNotGlued_holds)

    @classmethod
    def get_name(cls) -> str:
        return "pybullet_domino_fan"

    @property
    def predicates(self) -> Set[Predicate]:
        predicates = {
            # Fan predicates
            self._FanOn,
            self._FanOff,
            self._FanFacingSide,
            self._OppositeFan,
            self._Controls,
            # Ball predicates
            self._BallAtLoc,
            # Location predicates
            self._ClearLoc,
            self._SideOf,
            # Domino predicates
            self._Toppled,
            self._Upright,
            self._Tilting,
            self._InitialBlock,
            self._MovableBlock,
            self._HandEmpty,
            self._Holding,
        }

        # Add switch predicates if needed
        if not CFG.fan_known_controls_relation:
            predicates |= {self._SwitchOn, self._SwitchOff}

        # Add grid predicates if using grid
        if CFG.domino_fan_use_grid:
            predicates |= {
                self._DominoAtPos,
                self._DominoAtRot,
                self._PosClear,
                self._InFrontDirection,
                self._InFront,
            }

        # Add glued domino predicate if enabled
        if CFG.domino_fan_has_glued_dominoes:
            predicates.add(self._DominoNotGlued)

        return predicates

    @property
    def target_predicates(self) -> Set[Predicate]:
        target_preds = {
            self._FanFacingSide,
        }
        if CFG.domino_fan_use_grid:
            target_preds.add(self._InFrontDirection)
        if CFG.domino_fan_has_glued_dominoes:
            target_preds.add(self._DominoNotGlued)
        return target_preds

    @property
    def types(self) -> Set[Type]:
        types = {
            self._robot_type,
            self._fan_type,
            self._switch_type,
            self._side_type,
            self._ball_type,
            self._target_type,
            self._domino_type,
            self._wall_type,
            self._location_type,
            self._direction_type,
        }
        if CFG.domino_fan_use_grid:
            types.add(self._angle_type)
        return types

    @property
    def goal_predicates(self) -> Set[Predicate]:
        # Goals can be either ball at location OR dominoes toppled
        return {self._BallAtLoc, self._Toppled}

    # =========================================================================
    # PYBULLET INITIALIZATION
    # =========================================================================

    @classmethod
    def initialize_pybullet(
            cls, using_gui: bool
    ) -> Tuple[int, SingleArmPyBulletRobot, Dict[str, Any]]:
        """Initialize PyBullet bodies."""
        physics_client_id, pybullet_robot, bodies = super(
        ).initialize_pybullet(using_gui)

        # Create table
        table_id = create_object(
            asset_path="urdf/table.urdf",
            position=cls.table_pos,
            orientation=cls.table_orn,
            scale=cls.table_scale,
            use_fixed_base=True,
            physics_client_id=physics_client_id,
        )
        bodies["table_id"] = table_id

        # Add another table for more space (for fans to be on top)
        create_object(
            asset_path="urdf/table.urdf",
            position=(cls.table_pos[0], cls.table_pos[1] + cls.table_width / 2,
                      cls.table_pos[2]),
            orientation=cls.table_orn,
            scale=cls.table_scale,
            use_fixed_base=True,
            physics_client_id=physics_client_id,
        )

        # Create fans in four groups
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

        # Create switches
        switch_urdf = "urdf/partnet_mobility/switch/102812/switch.urdf"
        switch_ids = []
        for _ in range(4):
            sid = create_object(asset_path=switch_urdf,
                                scale=cls.switch_scale,
                                use_fixed_base=True,
                                physics_client_id=physics_client_id)
            switch_ids.append(sid)
        bodies["switch_ids"] = switch_ids

        # Create ball
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

        # Create target
        target_id = create_pybullet_block(
            color=cls.target_color,
            half_extents=(cls.pos_gap / 2, cls.pos_gap / 2,
                          cls.target_thickness),
            mass=cls.target_mass,
            friction=cls.target_friction,
            position=(0, 0, cls.table_height),
            orientation=p.getQuaternionFromEuler([0, 0, 0]),
            physics_client_id=physics_client_id)
        bodies["target_id"] = target_id

        # Create dominoes
        max_dominos = max(max(CFG.domino_fan_train_num_dominos),
                          max(CFG.domino_fan_test_num_dominos))
        domino_ids = []
        for i in range(max_dominos):
            # Import the helper function from domino env
            from predicators.envs.pybullet_domino import \
                create_domino_block  # type: ignore[attr-defined]
            domino_id = create_domino_block(
                color=cls.start_domino_color if i == 0 else cls.domino_color,
                half_extents=(cls.domino_width / 2, cls.domino_depth / 2,
                              cls.domino_height / 2),
                mass=cls.domino_mass,
                friction=cls.domino_friction,
                orientation=(0.0, 0.0, 0.0, 1.0),
                physics_client_id=physics_client_id,
                add_top_triangle=True,
            )
            domino_ids.append(domino_id)
        bodies["domino_ids"] = domino_ids

        # Create walls
        max_walls = max(max(CFG.domino_fan_train_num_walls),
                        max(CFG.domino_fan_test_num_walls))
        wall_ids = []
        for _ in range(max_walls):
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

        return physics_client_id, pybullet_robot, bodies

    @staticmethod
    def _get_joint_id(obj_id: int, joint_name: str) -> int:
        """Get joint ID by name."""
        num_joints = p.getNumJoints(obj_id)
        for j in range(num_joints):
            info = p.getJointInfo(obj_id, j)
            if info[1].decode("utf-8") == joint_name:
                return j
        return -1

    def _store_pybullet_bodies(self, pybullet_bodies: Dict[str, Any]) -> None:
        """Store references to PyBullet object IDs."""
        # Store fan IDs grouped by side
        fan_ids_by_side = [
            pybullet_bodies["fan_ids_left"],  # side 0 = left
            pybullet_bodies["fan_ids_right"],  # side 1 = right
            pybullet_bodies["fan_ids_back"],  # side 2 = back
            pybullet_bodies["fan_ids_front"]  # side 3 = front
        ]

        for side_idx, fan_obj in enumerate(self._fans):
            fan_obj.side_idx = side_idx
            fan_obj.fan_ids = fan_ids_by_side[side_idx]
            fan_obj.joint_ids = [
                self._get_joint_id(fid, "joint_0") for fid in fan_obj.fan_ids
            ]
            fan_obj.id = fan_obj.fan_ids[0] if fan_obj.fan_ids else -1

        # Store switch IDs
        for i, switch_obj in enumerate(self._switches):
            switch_obj.id = pybullet_bodies["switch_ids"][i]
            switch_obj.joint_id = self._get_joint_id(switch_obj.id, "joint_0")
            switch_obj.side_idx = i

        # Store side indices
        self._sides[0].side_idx = 1.0  # left
        self._sides[1].side_idx = 0.0  # right
        self._sides[2].side_idx = 3.0  # down/back
        self._sides[3].side_idx = 2.0  # up/front

        # Store ball and target
        self._ball.id = pybullet_bodies["ball_id"]
        self._target.id = pybullet_bodies["target_id"]

        # Store dominoes
        for domino, domino_id in zip(self._dominoes,
                                     pybullet_bodies["domino_ids"]):
            domino.id = domino_id

        # Store walls
        for wall, wall_id in zip(self._walls, pybullet_bodies["wall_ids"]):
            wall.id = wall_id

    # =========================================================================
    # STATE MANAGEMENT
    # =========================================================================

    def _get_object_ids_for_held_check(self) -> List[int]:
        """Return list of object IDs that can be held."""
        return [domino.id for domino in self._dominoes]

    def _create_task_specific_objects(self, state: State) -> None:
        """Create task-specific objects (not needed here)."""
        pass

    def _reset_custom_env_state(self, state: State) -> None:
        """Reset environment to match the given state."""
        # Set switch states
        for switch_obj in self._switches:
            is_on_val = state.get(switch_obj, "is_on")
            self._set_switch_on(switch_obj.id, bool(is_on_val > 0.5))

        # Position fans
        self._position_fans_on_sides()

        # Reposition boundary walls
        self._reposition_boundary_walls(state)

        # Reposition domino stairs (if enabled)
        self._reposition_domino_stairs(state)

        # Move unused walls out of view
        oov_x, oov_y = self._out_of_view_xy
        wall_objs = state.get_objects(self._wall_type)
        for i in range(len(wall_objs), len(self._walls)):
            update_object(self._walls[i].id,
                          position=(oov_x + i * 0.1, oov_y + i * 0.1, 0.0),
                          physics_client_id=self._physics_client_id)

        # Move unused dominoes out of view
        domino_objs = state.get_objects(self._domino_type)
        for i in range(len(domino_objs), len(self._dominoes)):
            update_object(self._dominoes[i].id,
                          position=(oov_x + i * 0.1, oov_y + i * 0.1,
                                    self.domino_height / 2),
                          physics_client_id=self._physics_client_id)

        # Update domino colors
        for domino in domino_objs:
            if domino.id is not None:
                r = state.get(domino, "r")
                g = state.get(domino, "g")
                b = state.get(domino, "b")
                update_object(domino.id,
                              color=(r, g, b, 1.0),
                              physics_client_id=self._physics_client_id)

        # Handle glued dominoes if enabled
        if CFG.domino_fan_has_glued_dominoes:
            for domino in domino_objs:
                if domino.id is not None:
                    if self._DominoGlued_holds(state, [domino]):
                        p.changeDynamics(
                            domino.id,
                            -1,
                            mass=1e10,
                            physicsClientId=self._physics_client_id)

    def _reposition_boundary_walls(self, state: State) -> None:
        """Create boundary walls based on actual grid positions."""
        # Remove existing boundary walls
        for wall_id in self._boundary_wall_ids:
            if wall_id >= 0:
                p.removeBody(wall_id, physicsClientId=self._physics_client_id)
        self._boundary_wall_ids = []

        # Get position objects if they exist
        position_objects = state.get_objects(self._location_type)
        if not position_objects:
            return

        # Set sim features for position objects
        for pos_obj in position_objects:
            pos_obj.xx = state.get(pos_obj, "xx")
            pos_obj.yy = state.get(pos_obj, "yy")

        # Find grid bounds
        x_coords = [state.get(pos_obj, "xx") for pos_obj in position_objects]
        y_coords = [state.get(pos_obj, "yy") for pos_obj in position_objects]

        grid_x_min, grid_x_max = min(x_coords), max(x_coords)
        grid_y_min, grid_y_max = min(y_coords), max(y_coords)

        # Create boundary walls
        # Left wall
        left_wall_id = create_pybullet_block(
            color=self.boundary_wall_color,
            half_extents=(self.boundary_wall_thickness / 2,
                          (grid_y_max - grid_y_min + self.pos_gap) / 2,
                          self.boundary_wall_height / 2),
            mass=self.wall_mass,
            friction=self.wall_friction,
            position=(grid_x_min - self.pos_gap / 2,
                      (grid_y_min + grid_y_max) / 2,
                      self.table_height + self.boundary_wall_height / 2),
            orientation=p.getQuaternionFromEuler([0, 0, 0]),
            physics_client_id=self._physics_client_id)

        # Right wall
        right_wall_id = create_pybullet_block(
            color=self.boundary_wall_color,
            half_extents=(self.boundary_wall_thickness / 2,
                          (grid_y_max - grid_y_min + self.pos_gap) / 2,
                          self.boundary_wall_height / 2),
            mass=self.wall_mass,
            friction=self.wall_friction,
            position=(grid_x_max + self.pos_gap / 2,
                      (grid_y_min + grid_y_max) / 2,
                      self.table_height + self.boundary_wall_height / 2),
            orientation=p.getQuaternionFromEuler([0, 0, 0]),
            physics_client_id=self._physics_client_id)

        # Front wall
        front_wall_id = create_pybullet_block(
            color=self.boundary_wall_color,
            half_extents=((grid_x_max - grid_x_min + self.pos_gap) / 2,
                          self.boundary_wall_thickness / 2,
                          self.boundary_wall_height / 2),
            mass=self.wall_mass,
            friction=self.wall_friction,
            position=((grid_x_min + grid_x_max) / 2,
                      grid_y_min - self.pos_gap / 2,
                      self.table_height + self.boundary_wall_height / 2),
            orientation=p.getQuaternionFromEuler([0, 0, 0]),
            physics_client_id=self._physics_client_id)

        # Back wall
        back_wall_id = create_pybullet_block(
            color=self.boundary_wall_color,
            half_extents=((grid_x_max - grid_x_min + self.pos_gap) / 2,
                          self.boundary_wall_thickness / 2,
                          self.boundary_wall_height / 2),
            mass=self.wall_mass,
            friction=self.wall_friction,
            position=((grid_x_min + grid_x_max) / 2,
                      grid_y_max + self.pos_gap / 2,
                      self.table_height + self.boundary_wall_height / 2),
            orientation=p.getQuaternionFromEuler([0, 0, 0]),
            physics_client_id=self._physics_client_id)

        self._boundary_wall_ids = [
            left_wall_id, right_wall_id, front_wall_id, back_wall_id
        ]

    def _reposition_domino_stairs(self, state: State) -> None:
        """Create stairs under dominoes when CFG.domino_domino_on_stairs is
        True."""
        # Remove existing stairs
        for stair_id in self._stair_ids:
            if stair_id >= 0:
                p.removeBody(stair_id, physicsClientId=self._physics_client_id)
        self._stair_ids = []

        # Only create stairs if the feature is enabled
        if not CFG.domino_domino_on_stairs:
            return

        # Get domino objects
        domino_objs = state.get_objects(self._domino_type)
        if not domino_objs:
            return

        # Create stairs under each domino with progressively increasing height
        for i, domino_obj in enumerate(domino_objs):
            domino_x = state.get(domino_obj, "x")
            domino_y = state.get(domino_obj, "y")

            # Calculate stair height based on domino index
            stair_height = self.obstacle_wall_height + (
                i * self.stair_height_increment)

            # Create stair block under the domino
            stair_id = create_pybullet_block(
                color=self.stair_color,
                half_extents=(self.wall_x_len / 2, self.wall_y_len / 2,
                              stair_height / 2),
                mass=self.wall_mass,
                friction=self.wall_friction,
                position=(domino_x, domino_y,
                          self.table_height + stair_height / 2),
                orientation=p.getQuaternionFromEuler([0, 0, 0]),
                physics_client_id=self._physics_client_id)

            self._stair_ids.append(stair_id)

    def _position_fans_on_sides(self) -> None:
        """Position all PyBullet fan bodies on their respective sides."""
        left_coords = np.linspace(self.fan_y_lb, self.fan_y_ub,
                                  self.num_left_fans)
        right_coords = np.linspace(self.fan_y_lb, self.fan_y_ub,
                                   self.num_right_fans)
        front_coords = np.linspace(self.fan_x_lb, self.fan_x_ub,
                                   self.num_front_fans)
        back_coords = np.linspace(self.fan_x_lb, self.fan_x_ub,
                                  self.num_back_fans)

        for fan_obj in self._fans:
            side_idx = fan_obj.side_idx
            fan_ids = fan_obj.fan_ids

            if side_idx == 0:  # left
                for i, fan_id in enumerate(fan_ids):
                    px = self.left_fan_x
                    py = left_coords[i] if i < len(
                        left_coords) else left_coords[-1]
                    pz = self.table_height + self.fan_z_len / 2
                    rot = [0.0, 0.0, 0.0]
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
                    rot = [0.0, 0.0, np.pi]
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
                    rot = [0.0, 0.0, np.pi / 2]
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
                    rot = [0.0, 0.0, -np.pi / 2]
                    update_object(fan_id,
                                  position=(px, py, pz),
                                  orientation=p.getQuaternionFromEuler(rot),
                                  physics_client_id=self._physics_client_id)

    def _extract_feature(self, obj: Object, feature: str) -> float:
        """Extract state features for objects."""
        if obj.type == self._fan_type:
            if feature == "facing_side":
                return float(obj.side_idx)
            elif feature == "is_on":
                controlling_switch = self._switches[obj.side_idx]
                return float(self._is_switch_on(controlling_switch.id))
        elif obj.type == self._switch_type:
            if feature == "controls_fan":
                return float(obj.side_idx)
            elif feature == "is_on":
                return float(self._is_switch_on(obj.id))
        elif obj.type == self._target_type:
            if feature == "is_hit":
                bx = self._current_observation.get(self._ball, "x")
                by = self._current_observation.get(self._ball, "y")
                tx = self._current_observation.get(self._target, "x")
                ty = self._current_observation.get(self._target, "y")
                return 1.0 if self._is_ball_close_to_position(bx, by, tx, ty) \
                    else 0.0
        elif obj.type == self._location_type:
            if feature == "xx":
                return obj.xx
            elif feature == "yy":
                return obj.yy
        elif obj.type == self._side_type:
            if feature == "side_idx":
                return float(obj.side_idx)
        elif obj.type == self._direction_type:
            if feature == "dir":
                if obj.name == "straight":
                    return 0.0
                elif obj.name == "left":
                    return 1.0
                elif obj.name == "right":
                    return 2.0
        elif obj.type == self._angle_type:
            if feature == "angle":
                angle_str = obj.name.split("_")[1]
                return float(angle_str)

        raise ValueError(f"Unknown feature {feature} for object {obj}")

    # =========================================================================
    # SIMULATION STEP
    # =========================================================================

    def step(self, action: Action, render_obs: bool = False) -> State:
        """Execute action and simulate fans."""
        super().step(action, render_obs=render_obs)
        self._simulate_fans()
        final_state = self._get_state()
        self._current_observation = final_state

        # Draw debug line at ball position
        bx, by = final_state.get(self._ball,
                                 "x"), final_state.get(self._ball, "y")
        p.addUserDebugLine(
            [bx, by, self.table_height],
            [bx, by, self.table_height + self.debug_line_height], [0, 1, 0],
            lifeTime=self.debug_line_lifetime,
            physicsClientId=self._physics_client_id)

        return final_state

    # =========================================================================
    # FAN SIMULATION - ONLY AFFECTS BALL (NOT DOMINOES)
    # =========================================================================

    def _simulate_fans(self) -> None:
        """Spin fans and blow the ball (NOT dominoes)."""
        if CFG.domino_fan_use_kinematic:
            self._simulate_fans_kinematic()
        else:
            self._simulate_fans_dynamic()

    def _simulate_fans_dynamic(self) -> None:
        """Dynamic fan simulation using forces (only on ball)."""
        for ctrl_fan_idx, switch_obj in enumerate(self._switches):
            on = self._is_switch_on(switch_obj.id)
            fan_obj = self._fans[ctrl_fan_idx]

            if not hasattr(fan_obj, 'fan_ids') or not fan_obj.fan_ids:
                continue

            if on and fan_obj.fan_ids:
                # Spin fan visuals
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
                # Apply force ONLY to ball (not dominoes)
                self._apply_fan_force_to_ball(fan_obj.fan_ids[0],
                                              self._ball.id)
            else:
                # Turn off fans
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
        """Kinematic fan simulation (position-based ball movement)."""
        # Get current ball position
        ball_pos, ball_orn = p.getBasePositionAndOrientation(
            self._ball.id, physicsClientId=self._physics_client_id)
        ball_x, ball_y, ball_z = ball_pos

        # Calculate movement vector
        movement_x = 0.0
        movement_y = 0.0

        for ctrl_fan_idx, switch_obj in enumerate(self._switches):
            on = self._is_switch_on(switch_obj.id)
            fan_obj = self._fans[ctrl_fan_idx]

            if not hasattr(fan_obj, 'fan_ids') or not fan_obj.fan_ids:
                continue

            if on and fan_obj.fan_ids:
                # Spin fans visually
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
                elif ctrl_fan_idx == 2:  # back fan - push forward
                    movement_y += self.kinematic_ball_speed
                elif ctrl_fan_idx == 3:  # front fan - push backward
                    movement_y -= self.kinematic_ball_speed
            else:
                # Turn off fans
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

        # Apply movement to ball
        if movement_x != 0.0 or movement_y != 0.0:
            new_x = max(self.x_lb, min(self.x_ub, ball_x + movement_x))
            new_y = max(self.y_lb, min(self.y_ub, ball_y + movement_y))

            p.resetBasePositionAndOrientation(
                self._ball.id,
                posObj=[new_x, new_y, ball_z],
                ornObj=ball_orn,
                physicsClientId=self._physics_client_id)

    def _apply_fan_force_to_ball(self, fan_id: int, ball_id: int) -> None:
        """Apply wind force from fan to ball ONLY."""
        _, orn_fan = p.getBasePositionAndOrientation(fan_id,
                                                     self._physics_client_id)

        if CFG.fan_fans_blow_opposite_direction:
            local_dir = np.array([-1.0, 0.0, 0.0])
        else:
            local_dir = np.array([1.0, 0.0, 0.0])
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

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _is_switch_on(self, switch_id: int) -> bool:
        """Check if switch is on."""
        joint_id = self._get_joint_id(switch_id, "joint_0")
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
        """Set switch state."""
        joint_id = self._get_joint_id(switch_id, "joint_0")
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
        """Check if ball is close to target position."""
        tolerance = CFG.domino_fan_ball_position_tolerance
        return np.abs(bx - tx) < tolerance and np.abs(by - ty) < tolerance

    @classmethod
    def _generate_grid_coordinates(
            cls, num_pos_x: int,
            num_pos_y: int) -> Tuple[List[float], List[float]]:
        """Generate grid coordinates."""
        if num_pos_x % 2 == 1:
            x_start = cls.loc_x_mid - (num_pos_x - 1) * cls.pos_gap / 2
        else:
            x_start = cls.loc_x_mid - num_pos_x * cls.pos_gap / 2 + cls.pos_gap / 2

        if num_pos_y % 2 == 1:
            y_start = cls.loc_y_mid - (num_pos_y - 1) * cls.pos_gap / 2
        else:
            y_start = cls.loc_y_mid - num_pos_y * cls.pos_gap / 2 + cls.pos_gap / 2

        # Round to 5 decimals to match movement calculations
        x_coords = [
            round(x_start + i * cls.pos_gap, 5) for i in range(num_pos_x)
        ]
        y_coords = [
            round(y_start + i * cls.pos_gap, 5) for i in range(num_pos_y)
        ]

        return x_coords, y_coords

    # =========================================================================
    # PREDICATE HOLDS METHODS
    # =========================================================================

    # Fan predicates
    @staticmethod
    def _FanOn_holds(state: State, objects: Sequence[Object]) -> bool:
        """Check if fan is on."""
        (fan, ) = objects
        return state.get(fan, "is_on") > 0.5

    def _FanFacingSide_holds(self, state: State,
                             objects: Sequence[Object]) -> bool:
        """Check if fan is facing specified side."""
        fan, side = objects
        return state.get(fan, "facing_side") == state.get(side, "side_idx")

    def _OppositeFan_holds(self, state: State,
                           objects: Sequence[Object]) -> bool:
        """Check if fans are on opposite sides."""
        fan1, fan2 = objects
        if fan1.name == fan2.name:
            return False
        side1 = state.get(fan1, "facing_side")
        side2 = state.get(fan2, "facing_side")
        return abs(side1 - side2) == 1 and (side1 // 2) == (side2 // 2)

    def _Controls_holds(self, state: State, objects: Sequence[Object]) -> bool:
        """Check if switch controls fan."""
        switch, fan = objects
        return state.get(fan,
                         "facing_side") == state.get(switch, "controls_fan")

    # Ball predicates
    def _BallAtLoc_holds(self, state: State,
                         objects: Sequence[Object]) -> bool:
        """Check if ball is at location."""
        ball, pos = objects
        return self._is_ball_close_to_position(state.get(ball, "x"),
                                               state.get(ball, "y"),
                                               state.get(pos, "xx"),
                                               state.get(pos, "yy"))

    def _ClearLoc_holds(self, state: State, objects: Sequence[Object]) -> bool:
        """Check if location is clear."""
        pos, = objects
        pos_x, pos_y = state.get(pos, "xx"), state.get(pos, "yy")

        # Check walls
        for obj in state.get_objects(self._wall_type):
            wx, wy = state.get(obj, "x"), state.get(obj, "y")
            if self._is_ball_close_to_position(pos_x, pos_y, wx, wy):
                return False

        # Check dominoes
        for obj in state.get_objects(self._domino_type):
            dx, dy = state.get(obj, "x"), state.get(obj, "y")
            if self._is_ball_close_to_position(pos_x, pos_y, dx, dy):
                return False

        return True

    def _SideOf_holds(self, state: State, objects: Sequence[Object]) -> bool:
        """Check if pos1 is to the specified side of pos2."""
        pos1, pos2, side = objects
        side_val = state.get(side, "side_idx")

        if side_val == 1:  # left
            return self._is_ball_close_to_position(
                state.get(pos1, "xx") + self.pos_gap, state.get(pos1, "yy"),
                state.get(pos2, "xx"), state.get(pos2, "yy"))
        elif side_val == 0:  # right
            return self._is_ball_close_to_position(
                state.get(pos1, "xx") - self.pos_gap, state.get(pos1, "yy"),
                state.get(pos2, "xx"), state.get(pos2, "yy"))
        elif side_val == 2:  # down
            return self._is_ball_close_to_position(
                state.get(pos1, "xx"),
                state.get(pos1, "yy") - self.pos_gap, state.get(pos2, "xx"),
                state.get(pos2, "yy"))
        elif side_val == 3:  # up
            return self._is_ball_close_to_position(
                state.get(pos1, "xx"),
                state.get(pos1, "yy") + self.pos_gap, state.get(pos2, "xx"),
                state.get(pos2, "yy"))
        return False

    # Domino predicates
    @classmethod
    def _Toppled_holds(cls, state: State, objects: Sequence[Object]) -> bool:
        """Check if domino has toppled."""
        obj, = objects
        roll_angle = abs(state.get(obj, "roll"))
        return roll_angle >= cls.fallen_threshold

    @classmethod
    def _Upright_holds(cls, state: State, objects: Sequence[Object]) -> bool:
        """Check if domino is upright."""
        obj, = objects
        tilt_angle = state.get(obj, "roll")
        return abs(tilt_angle) < cls.domino_roll_threshold

    @classmethod
    def _Tilting_holds(cls, state: State, objects: Sequence[Object]) -> bool:
        """Check if domino is tilting."""
        obj, = objects
        roll_angle = abs(state.get(obj, "roll"))
        return cls.domino_roll_threshold <= roll_angle < cls.fallen_threshold

    @classmethod
    def _InitialBlock_holds(cls, state: State,
                            objects: Sequence[Object]) -> bool:
        """Check if domino is the initial block (start domino)."""
        domino, = objects
        eps = 1e-3
        return (
            abs(state.get(domino, "r") - cls.start_domino_color[0]) < eps
            and abs(state.get(domino, "g") - cls.start_domino_color[1]) < eps
            and abs(state.get(domino, "b") - cls.start_domino_color[2]) < eps)

    @classmethod
    def _MovableBlock_holds(cls, state: State,
                            objects: Sequence[Object]) -> bool:
        """Check if domino is a movable block."""
        domino, = objects
        eps = 1e-3
        return (abs(state.get(domino, "r") - cls.domino_color[0]) < eps
                and abs(state.get(domino, "g") - cls.domino_color[1]) < eps
                and abs(state.get(domino, "b") - cls.domino_color[2]) < eps)

    def _HandEmpty_holds(self, state: State,
                         objects: Sequence[Object]) -> bool:
        """Check if robot hand is empty."""
        robot, = objects
        dominoes = state.get_objects(self._domino_type)
        for domino in dominoes:
            if state.get(domino, "is_held"):
                return False
        return True

    @classmethod
    def _Holding_holds(cls, state: State, objects: Sequence[Object]) -> bool:
        """Check if robot is holding domino."""
        _, domino = objects
        return state.get(domino, "is_held") > 0.5

    # Grid-based domino predicates
    def _DominoAtPos_holds(self, state: State,
                           objects: Sequence[Object]) -> bool:
        """Check if domino is at position."""
        domino, position = objects
        if state.get(domino, "is_held"):
            return False

        domino_x = state.get(domino, "x")
        domino_y = state.get(domino, "y")

        # Find closest position
        closest_position = None
        closest_distance = float('inf')
        for pos in state.get_objects(self._location_type):
            pos_x = state.get(pos, "xx")
            pos_y = state.get(pos, "yy")
            distance = np.sqrt((domino_x - pos_x)**2 + (domino_y - pos_y)**2)
            if distance < closest_distance:
                closest_distance = distance
                closest_position = pos
        return closest_position == position

    @classmethod
    def _DominoAtRot_holds(cls, state: State,
                           objects: Sequence[Object]) -> bool:
        """Check if domino is at rotation."""
        domino, rotation = objects
        if state.get(domino, "is_held"):
            return False

        domino_rot = state.get(domino, "yaw")
        target_rot_degrees = state.get(rotation, "angle")
        target_rot_radians = np.radians(target_rot_degrees)

        rotation_tolerance = np.radians(15)
        angle_diff = abs(utils.wrap_angle(domino_rot - target_rot_radians))

        return angle_diff <= rotation_tolerance

    @classmethod
    def _PosClear_holds(cls, state: State, objects: Sequence[Object]) -> bool:
        """Check if position is clear."""
        position, = objects

        target_x = state.get(position, "xx")
        target_y = state.get(position, "yy")

        position_tolerance = cls.pos_gap * 0.5
        for domino in state.get_objects(cls._domino_type):
            domino_x = state.get(domino, "x")
            domino_y = state.get(domino, "y")

            if (abs(domino_x - target_x) <= position_tolerance
                    and abs(domino_y - target_y) <= position_tolerance
                    and not state.get(domino, "is_held")):
                return False

        return True

    @classmethod
    def _InFrontDirection_holds(cls, atoms: Set[GroundAtom],
                                objects: Sequence[Object]) -> bool:
        """Derived predicate: check if domino1 is in front of domino2 in
        direction."""
        # Reuse implementation from domino env (simplified version)
        domino1, domino2, direction_obj = objects
        # This is a complex derived predicate - for now return a simple check
        # Full implementation would require grid coordinate parsing
        return False  # Placeholder

    @classmethod
    def _InFront_holds(cls, atoms: Set[GroundAtom],
                       objects: Sequence[Object]) -> bool:
        """Derived predicate: check if domino1 is in front of domino2."""
        domino1, domino2 = objects
        for atom in atoms:
            if (atom.predicate.name == "InFrontDirection"
                    and len(atom.objects) == 3 and atom.objects[0] == domino1
                    and atom.objects[1] == domino2):
                return True
        return False

    @classmethod
    def _DominoNotGlued_holds(cls, state: State,
                              objects: Sequence[Object]) -> bool:
        """Check if domino is not glued."""
        return not cls._DominoGlued_holds(state, objects)

    @classmethod
    def _DominoGlued_holds(cls, state: State,
                           objects: Sequence[Object]) -> bool:
        """Check if domino is glued (red color)."""
        eps = 1e-3
        r_val = state.get(objects[0], "r")
        g_val = state.get(objects[0], "g")
        b_val = state.get(objects[0], "b")

        return (abs(r_val - cls.glued_domino_color[0]) < eps
                and abs(g_val - cls.glued_domino_color[1]) < eps
                and abs(b_val - cls.glued_domino_color[2]) < eps)

    # =========================================================================
    # TASK GENERATION
    # =========================================================================

    def _generate_train_tasks(self) -> List[EnvironmentTask]:
        """Generate training tasks."""
        return self._make_tasks(
            num_tasks=CFG.num_train_tasks,
            possible_num_dominos=CFG.domino_fan_train_num_dominos,
            possible_num_walls=CFG.domino_fan_train_num_walls,
            possible_num_targets=CFG.domino_fan_train_num_targets,
            grid_size=CFG.domino_fan_train_grid_size,
            rng=self._train_rng)

    def _generate_test_tasks(self) -> List[EnvironmentTask]:
        """Generate test tasks."""
        return self._make_tasks(
            num_tasks=CFG.num_test_tasks,
            possible_num_dominos=CFG.domino_fan_test_num_dominos,
            possible_num_walls=CFG.domino_fan_test_num_walls,
            possible_num_targets=CFG.domino_fan_test_num_targets,
            grid_size=CFG.domino_fan_test_grid_size,
            rng=self._test_rng)

    def _make_tasks(self, num_tasks: int, possible_num_dominos: List[int],
                    possible_num_walls: List[int],
                    possible_num_targets: List[int], grid_size: Tuple[int,
                                                                      int],
                    rng: np.random.Generator) -> List[EnvironmentTask]:
        """Generate tasks with either domino goals OR ball goals.

        All assets (fans, switches, ball, dominoes) are present in every
        task.
        """
        tasks = []
        num_pos_x, num_pos_y = grid_size

        for i_task in range(num_tasks):
            # Decide task type: domino or ball
            task_type = rng.choice(["domino", "ball"],
                                   p=[
                                       1 - CFG.domino_fan_ball_task_ratio,
                                       CFG.domino_fan_ball_task_ratio
                                   ])

            task = self._generate_task(i_task, task_type, possible_num_dominos,
                                       possible_num_walls,
                                       possible_num_targets, num_pos_x,
                                       num_pos_y, rng)

            if task is not None:
                tasks.append(task)

        return self._add_pybullet_state_to_tasks(tasks)

    def _generate_domino_sequence_with_grid(
            self,
            rng: np.random.Generator,
            n_dominos: int,
            n_targets: int,
            num_pos_x: int,
            num_pos_y: int,
            log_debug: bool = False,
            task_idx: Optional[int] = None) -> Optional[Tuple[Dict, Set]]:
        """Grid-based sequence generator adapted from pybullet_domino.py.

        Returns:
            Tuple of (obj_dict, used_coords) where obj_dict maps domino objects
            to their placement parameters, and used_coords is a set of grid
            coordinates used by the sequence. Returns None if generation failed.
        """
        obj_dict: Dict = {}
        domino_count = 0
        target_count = 0
        used_coords = set()

        # Generate grid coordinates for this specific configuration
        x_coords, y_coords = self._generate_grid_coordinates(
            num_pos_x, num_pos_y)
        grid_pos = [(x, y) for y in y_coords for x in x_coords]

        if log_debug:
            print(f"Generated {num_pos_x}x{num_pos_y} grid:")
            print(f"  x_coords: {[f'{x:.3f}' for x in x_coords]}")
            print(f"  y_coords: {[f'{y:.3f}' for y in y_coords]}")
            print(f"  pos_gap: {self.pos_gap}")

        # Use a set for efficient checking of valid grid coordinates.
        grid_coords_set = set(grid_pos)

        # Choose a random starting position and orientation (cardinal directions).
        start_idx = rng.choice(len(grid_pos))
        curr_x, curr_y = grid_pos[start_idx]
        # If in the top row, can't face away from robot because it's unreachable
        top_row_y = np.max([y for _, y in grid_pos])
        if np.abs(curr_y - top_row_y) < 1e-3:
            curr_rot = rng.choice([np.pi / 2, np.pi, -np.pi / 2])
        else:
            curr_rot = rng.choice([0, np.pi / 2, np.pi, -np.pi / 2])
        used_coords.add((curr_x, curr_y))

        # Place the first domino (start block).
        # Calculate z position with stair height if stairs are enabled
        if CFG.domino_domino_on_stairs:
            stair_height = self.obstacle_wall_height + (
                domino_count * self.stair_height_increment)
            domino_z = self.table_height + stair_height + self.domino_height / 2
        else:
            domino_z = self.z_lb + self.domino_height / 2

        obj_dict[self._dominoes[domino_count]] = {
            "x": curr_x,
            "y": curr_y,
            "z": domino_z,
            "yaw": curr_rot,
            "roll": 0.0,
            "r": self.start_domino_color[0],
            "g": self.start_domino_color[1],
            "b": self.start_domino_color[2],
            "is_held": 0.0,
        }
        domino_count += 1
        if log_debug:
            print(f"Placed first domino at {curr_x}, {curr_y}, {curr_rot}")

        # Determine total domino blocks to place
        # For domino_fan, we always use domino blocks as targets
        total_domino_blocks = n_dominos + n_targets

        if log_debug:
            print(
                f"  Grid: {num_pos_x}x{num_pos_y} = {len(grid_pos)} positions")
            print(
                f"  Need to place: {total_domino_blocks} dominoes ({n_dominos} regular + {n_targets} targets)"
            )

        # Main placement loop.
        while domino_count < total_domino_blocks:
            possible_moves = []
            # A move is defined by: (name, final_x, final_y, final_rot, placements)
            # where placements is a list of (x, y, rot) for each domino in the move.

            # 1. Check for a "straight" move (1 domino).
            dx = round(self.pos_gap * np.sin(curr_rot), 5)
            dy = round(self.pos_gap * np.cos(curr_rot), 5)
            next_x = round(curr_x + dx, 5)
            next_y = round(curr_y + dy, 5)

            if log_debug:
                print(
                    f"  Domino {domino_count}: at ({curr_x:.3f}, {curr_y:.3f}), rot={curr_rot:.2f}"
                )
                print(f"    Checking straight: ({next_x:.3f}, {next_y:.3f})",
                      end="")

            if (next_x, next_y) in grid_coords_set and \
            (next_x, next_y) not in used_coords:
                placements = [(next_x, next_y, curr_rot)]
                possible_moves.append(
                    ("straight", next_x, next_y, curr_rot, placements))
                if log_debug:
                    print(" ✓")
            elif log_debug:
                in_grid = (next_x, next_y) in grid_coords_set
                is_used = (next_x, next_y) in used_coords
                print(f" ✗ (in_grid={in_grid}, used={is_used})")

            # 2. Check for "turn" moves (2 dominoes).
            if (total_domino_blocks - domino_count) >= 2:
                # turn_dir: -1 for left, 1 for right.
                for turn_dir, name in [(-1, "turn_left"), (1, "turn_right")]:
                    # The first domino (d1) is one step straight on the grid.
                    d1_grid_x, d1_grid_y = next_x, next_y
                    if (d1_grid_x, d1_grid_y) not in grid_coords_set or \
                       (d1_grid_x, d1_grid_y) in used_coords:
                        continue

                    # Its orientation is 45 degrees towards the turn direction.
                    d1_rot = curr_rot - turn_dir * np.pi / 4

                    # Calculate the shift vector to pull the turning domino inward.
                    shift_magnitude = self.domino_width * self.turn_shift_frac
                    shift_dx = shift_magnitude * \
                        (turn_dir * np.cos(curr_rot) - np.sin(curr_rot))
                    shift_dy = shift_magnitude * \
                        (-turn_dir * np.sin(curr_rot) - np.cos(curr_rot))

                    # The physical position is the grid position plus the shift.
                    d1_x = d1_grid_x + shift_dx
                    d1_y = d1_grid_y + shift_dy

                    # The second domino (d2) completes the turn.
                    d2_rot = d1_rot - turn_dir * 1 * np.pi / 4

                    # Calculate d2's physical position relative to d1's.
                    gap = self.pos_gap
                    sin_d1 = np.sin(d1_rot)
                    cos_d1 = np.cos(d1_rot)
                    disp_x = (
                        gap * turn_dir * cos_d1 +
                        (2 * shift_magnitude - gap) * sin_d1) / np.sqrt(2)
                    disp_y = (
                        -gap * turn_dir * sin_d1 +
                        (2 * shift_magnitude - gap) * cos_d1) / np.sqrt(2)
                    d2_x = round(d1_x + disp_x, 5)
                    d2_y = round(d1_y + disp_y, 5)

                    # Check if the grid position of the second domino is valid.
                    expected_final_rot = curr_rot + turn_dir * np.pi / 2
                    d2_grid_dx = round(
                        self.pos_gap * np.sin(expected_final_rot), 5)
                    d2_grid_dy = round(
                        self.pos_gap * np.cos(expected_final_rot), 5)
                    d2_grid_x = round(d1_grid_x + d2_grid_dx, 5)
                    d2_grid_y = round(d1_grid_y + d2_grid_dy, 5)

                    if (d2_grid_x, d2_grid_y) in grid_coords_set and \
                       (d2_grid_x, d2_grid_y) not in used_coords:

                        placements = [(d1_x, d1_y, d1_rot),
                                      (d2_x, d2_y, d2_rot)]

                        possible_moves.append(
                            (name, d2_grid_x, d2_grid_y, d2_rot, placements))

            if not possible_moves:
                # No valid moves, generation failed for this attempt.
                if log_debug:
                    print(
                        f"  ✗ Failed: No valid moves from ({curr_x}, {curr_y}) with rot={curr_rot:.2f}"
                    )
                    print(
                        f"     Placed {domino_count}/{total_domino_blocks} dominoes so far"
                    )
                    print(
                        f"     Grid coords set size: {len(grid_coords_set)}, Used: {len(used_coords)}"
                    )
                return None

            # Choose a random valid move and get its placement plan.
            _move_name, final_x, final_y, final_rot, placements = \
                possible_moves[rng.choice(len(possible_moves))]
            if log_debug:
                print(
                    f"Chose move: {_move_name}, final_x: {final_x}, final_y: {final_y}, final_rot: {final_rot}"
                )

            # Execute the placement plan for the chosen move.
            for (x, y, rot) in placements:
                if log_debug:
                    print(f"Placing domino at {x}, {y}, {rot}")
                if domino_count >= total_domino_blocks:
                    break  # Should not be reached with correct logic.

                # Decide if this domino block should be a target.
                is_target = False
                if target_count < n_targets:
                    remaining_blocks = total_domino_blocks - domino_count
                    remaining_targets = n_targets - target_count

                    # Reserve one target for the very last domino in the sequence
                    is_last_block = (domino_count == total_domino_blocks - 1)
                    has_targets_left = remaining_targets > 0

                    if is_last_block and has_targets_left:
                        # Force the last domino to be a target if we still need targets
                        is_target = True
                    elif not is_last_block and remaining_targets > 1:
                        # For non-last dominoes, only consider making them targets if we have more than 1 target left
                        # This ensures at least one target is reserved for the end
                        targets_available_for_placement = remaining_targets - 1
                        if (targets_available_for_placement >= remaining_blocks - 1 or \
                            rng.random() < targets_available_for_placement / (remaining_blocks - 1)) and\
                            domino_count >= 2:
                            is_target = True

                # Determine color based on whether this is a target
                if is_target:
                    # Check if target should be glued
                    should_be_glued = False
                    if CFG.domino_fan_has_glued_dominoes:
                        should_be_glued = rng.random(
                        ) < 0.5  # 50% chance of glued

                    if should_be_glued:
                        color = self.glued_domino_color
                    else:
                        color = self.target_domino_color
                else:
                    color = self.domino_color

                # Place the domino block.
                # Calculate z position with stair height if stairs are enabled
                if CFG.domino_domino_on_stairs:
                    stair_height = self.obstacle_wall_height + (
                        domino_count * self.stair_height_increment)
                    domino_z = self.table_height + stair_height + self.domino_height / 2
                else:
                    domino_z = self.z_lb + self.domino_height / 2

                obj_dict[self._dominoes[domino_count]] = {
                    "x": x,
                    "y": y,
                    "z": domino_z,
                    "yaw": rot,
                    "roll": 0.0,
                    "r": color[0],
                    "g": color[1],
                    "b": color[2],
                    "is_held": 0.0,
                }

                # Use the grid coordinates for tracking used spots.
                # Find the closest grid coordinate to the physical placement
                closest_grid_coord = min(grid_pos,
                                         key=lambda p: (p[0] - x)**2 +
                                         (p[1] - y)**2)
                used_coords.add(closest_grid_coord)

                if is_target:
                    target_count += 1
                domino_count += 1

            # Update state for the next iteration.
            curr_x, curr_y, curr_rot = final_x, final_y, final_rot

        return (obj_dict, used_coords)

    def _generate_task(self, task_idx: int, task_type: str,
                       possible_num_dominos: List[int],
                       possible_num_walls: List[int],
                       possible_num_targets: List[int], num_pos_x: int,
                       num_pos_y: int,
                       rng: np.random.Generator) -> Optional[EnvironmentTask]:
        """Generate a task with either domino toppling or ball-at-location
        goal.

        Args:
            task_type: Either "domino" or "ball"

        All assets (fans, switches, ball, dominoes, walls) are present in every task.
        Only the goal differs based on task_type.
        """
        # Determine number of objects based on task type
        n_dominos = rng.choice(possible_num_dominos)
        n_walls = rng.choice(possible_num_walls)

        # For domino tasks, we need to specify targets (toppleable dominoes)
        n_targets = rng.choice(possible_num_targets)

        # Generate grid
        x_coords, y_coords = self._generate_grid_coordinates(
            num_pos_x, num_pos_y)
        grid_pos = [(x, y) for y in y_coords for x in x_coords]

        # Initialize state dictionary
        init_dict = {}

        # Robot
        init_dict[self._robot] = {
            "x": self.robot_init_x,
            "y": self.robot_init_y,
            "z": self.robot_init_z,
            "fingers": self.open_fingers,
            "tilt": self.robot_init_tilt,
            "wrist": self.robot_init_wrist,
        }

        # Fans - all off initially
        for fan_obj in self._fans:
            side_idx = fan_obj.side_idx
            if side_idx == 0:  # left
                px, py, rot = self.left_fan_x, (self.fan_y_lb +
                                                self.fan_y_ub) / 2, 0.0
            elif side_idx == 1:  # right
                px, py, rot = self.right_fan_x, (self.fan_y_lb +
                                                 self.fan_y_ub) / 2, np.pi
            elif side_idx == 2:  # back
                px, py, rot = (self.fan_x_lb +
                               self.fan_x_ub) / 2, self.down_fan_y, np.pi / 2
            else:  # front
                px, py, rot = (self.fan_x_lb +
                               self.fan_x_ub) / 2, self.up_fan_y, -np.pi / 2

            init_dict[fan_obj] = {
                "x": px,
                "y": py,
                "z": self.table_height + self.fan_z_len / 2,
                "rot": rot,
                "facing_side": float(side_idx),
                "is_on": 0.0
            }

        # Switches - all off initially
        for switch_obj in self._switches:
            init_dict[switch_obj] = {
                "x": self.switch_base_x +
                self.switch_x_spacing * switch_obj.side_idx,
                "y": self.switch_y,
                "z": self.table_height,
                "rot": np.pi / 2,
                "controls_fan": float(switch_obj.side_idx),
                "is_on": 0.0,
            }

        # Sides
        init_dict[self._sides[0]] = {"side_idx": 1.0}
        init_dict[self._sides[1]] = {"side_idx": 0.0}
        init_dict[self._sides[2]] = {"side_idx": 3.0}
        init_dict[self._sides[3]] = {"side_idx": 2.0}

        # Generate domino sequence using the sequence generator
        # Try to generate a valid sequence (with retries)
        domino_obj_dict = None
        used_coords: Set[Tuple[int, int]] = set()
        max_attempts = 100

        for attempt in range(max_attempts):
            result = self._generate_domino_sequence_with_grid(
                rng=rng,
                n_dominos=n_dominos,
                n_targets=n_targets,
                num_pos_x=num_pos_x,
                num_pos_y=num_pos_y,
                log_debug=(attempt < 3),  # Debug first 3 attempts
                task_idx=task_idx)
            if result is not None:
                domino_obj_dict, used_coords = result
                break

        if domino_obj_dict is None:
            # Failed to generate a valid domino sequence
            return None

        # If we want to initialize at finished state, move intermediate objects
        if not CFG.domino_initialize_at_finished_state:
            domino_obj_dict = self._move_intermediate_objects_to_unfinished_state(
                domino_obj_dict, num_pos_x, num_pos_y)

        # Add dominoes to init_dict
        init_dict.update(domino_obj_dict)

        # Convert used_coords to a set of indices for easier tracking
        used_position_coords = used_coords.copy()

        # Get available grid positions (not used by dominoes)
        available_positions = [(x, y) for (x, y) in grid_pos
                               if (x, y) not in used_position_coords]

        # Ensure we have enough positions for walls, ball, and target
        if len(available_positions) < n_walls + 2:  # +2 for ball and target
            # Not enough space, regenerate or return None
            return None

        # Shuffle available positions for random placement
        rng.shuffle(available_positions)

        # Place walls on available positions
        wall_positions = available_positions[:n_walls]
        for i, (x, y) in enumerate(wall_positions):
            init_dict[self._walls[i]] = {
                "x": x,
                "y": y,
                "z": self.table_height + self.obstacle_wall_height / 2,
                "rot": 0.0
            }

        # Remove wall positions from available positions
        available_positions = available_positions[n_walls:]

        # Place ball on an available position
        ball_x, ball_y = available_positions[0]
        init_dict[self._ball] = {
            "x": ball_x,
            "y": ball_y,
            "z": self.table_height + self.ball_height_offset
        }

        # Place target on a different available position
        target_x, target_y = available_positions[1]
        init_dict[self._target] = {
            "x": target_x,
            "y": target_y,
            "z": self.table_height,
            "rot": 0.0,
            "is_hit": 0.0
        }

        # Add grid positions
        target_pos_obj: Optional[Object] = None
        if CFG.domino_fan_use_grid:
            positions = [
                Object(f"loc_y{i}_x{j}", self._location_type)
                for i in range(num_pos_y) for j in range(num_pos_x)
            ]
            for i, pos_obj in enumerate(positions):
                y_idx, x_idx = i // num_pos_x, i % num_pos_x
                px, py = x_coords[x_idx], y_coords[y_idx]
                init_dict[pos_obj] = {"xx": px, "yy": py}

                # Find target position object for ball tasks
                if task_type == "ball":
                    if abs(px - target_x) < 0.01 and abs(py - target_y) < 0.01:
                        target_pos_obj = pos_obj

            # Add rotation objects
            for rotation_obj in self._rotations:
                angle_str = rotation_obj.name.split("_")[1]
                init_dict[rotation_obj] = {"angle": float(angle_str)}

        # Add direction objects
        for i, direction_obj in enumerate(self._directions):
            init_dict[direction_obj] = {"dir": float(i)}

        # Create state
        init_state = utils.create_state_from_dict(init_dict)

        # Create goal based on task type
        if task_type == "domino":
            # Goal: all target dominoes toppled
            # Target dominoes are those with target_domino_color or glued_domino_color
            goal_atoms = set()
            eps = 1e-3
            for domino_obj in init_state.get_objects(self._domino_type):
                r = init_state.get(domino_obj, "r")
                g = init_state.get(domino_obj, "g")
                b = init_state.get(domino_obj, "b")

                # Check if it's a target domino (pink or red/glued)
                is_target = ((abs(r - self.target_domino_color[0]) < eps
                              and abs(g - self.target_domino_color[1]) < eps
                              and abs(b - self.target_domino_color[2]) < eps)
                             or
                             (abs(r - self.glued_domino_color[0]) < eps
                              and abs(g - self.glued_domino_color[1]) < eps
                              and abs(b - self.glued_domino_color[2]) < eps))

                if is_target:
                    goal_atoms.add(GroundAtom(self._Toppled, [domino_obj]))
        else:  # ball task
            # Goal: ball at target location
            if target_pos_obj is None:
                raise ValueError(
                    "Could not find target position object in grid")
            goal_atoms = {
                GroundAtom(self._BallAtLoc, [self._ball, target_pos_obj])
            }

        return EnvironmentTask(init_state, goal_atoms)

    def _move_intermediate_objects_to_unfinished_state(
            self,
            obj_dict: Dict,
            num_pos_x: Optional[int] = None,
            num_pos_y: Optional[int] = None) -> Dict:
        """Move all intermediate dominoes and walls to the lower end of the
        table in a row, keeping only the start domino and targets in their
        original positions.

        When CFG.domino_fan_use_grid=True, places intermediate objects on clear
        grid positions, preferably on the bottom side starting from the
        middle and extending to left and right.

        Args:
            obj_dict: Dictionary containing the original positions of all objects
            num_pos_x: Number of grid positions in x direction
            num_pos_y: Number of grid positions in y direction

        Returns:
            Modified dictionary with intermediate objects repositioned
        """
        # Identify which objects to move
        intermediate_objects = []

        # Find all dominoes except the start domino (which has light green color)
        # and target dominoes (which have pink or red/glued color)
        for domino in self._dominoes:
            if domino in obj_dict:
                domino_data = obj_dict[domino]
                # Check if it's not a start domino (not light green)
                eps = 1e-3
                is_start_domino = (abs(
                    domino_data.get("r", 0.0) -
                    self.start_domino_color[0]) < eps and abs(
                        domino_data.get("g", 0.0) - self.start_domino_color[1])
                                   < eps and abs(
                                       domino_data.get("b", 0.0) -
                                       self.start_domino_color[2]) < eps)

                # Check if it's a target domino (pink or red/glued color)
                is_target_domino = (abs(
                    domino_data.get("r", 0.0) -
                    self.target_domino_color[0]) < eps and abs(
                        domino_data.get("g", 0.0) -
                        self.target_domino_color[1]) < eps and abs(
                            domino_data.get("b", 0.0) -
                            self.target_domino_color[2]) < eps) or (abs(
                                domino_data.get("r", 0.0) -
                                self.glued_domino_color[0]) < eps and abs(
                                    domino_data.get("g", 0.0) -
                                    self.glued_domino_color[1]) < eps and abs(
                                        domino_data.get("b", 0.0) -
                                        self.glued_domino_color[2]) < eps)

                # Only move dominoes that are neither start nor target dominoes
                if not is_start_domino and not is_target_domino:
                    intermediate_objects.append((domino, "domino"))

        # Find all walls
        for wall in self._walls:
            if wall in obj_dict:
                intermediate_objects.append((wall, "wall"))

        if not intermediate_objects:
            return obj_dict

        if CFG.domino_fan_use_grid:
            # Use grid positioning when grid is enabled
            # First, identify which grid positions are already occupied
            occupied_positions = set()
            position_tolerance = self.pos_gap * 0.5

            # Extract just the objects from the intermediate_objects tuples for easier checking
            intermediate_obj_set = {
                obj
                for obj, obj_type in intermediate_objects
            }

            # Generate grid positions
            if num_pos_x is not None and num_pos_y is not None:
                x_coords, y_coords = self._generate_grid_coordinates(
                    num_pos_x, num_pos_y)
            else:
                # Fallback to maximum grid size
                max_grid_size = max(CFG.domino_fan_train_grid_size,
                                    CFG.domino_fan_test_grid_size,
                                    key=lambda t: t[0] * t[1])
                x_coords, y_coords = self._generate_grid_coordinates(
                    max_grid_size[0], max_grid_size[1])

            grid_pos = [(x, y) for y in y_coords for x in x_coords]

            for obj, obj_data in obj_dict.items():
                if obj not in intermediate_obj_set:  # Skip objects we're about to move
                    obj_x = obj_data.get("x", 0.0)
                    obj_y = obj_data.get("y", 0.0)

                    # Check which grid position this object occupies
                    for grid_x, grid_y in grid_pos:
                        if (abs(obj_x - grid_x) <= position_tolerance
                                and abs(obj_y - grid_y) <= position_tolerance):
                            occupied_positions.add((grid_x, grid_y))
                            break

            # Find available positions on the bottom side, starting from middle
            # Sort grid positions by y coordinate (ascending) then by distance from x center
            x_center = (x_coords[0] + x_coords[-1]) / 2 if x_coords else 0

            # Get bottom row positions first, then other rows if needed
            available_positions = []
            for y in sorted(y_coords):  # Start from bottom (smallest y)
                row_positions = [(x, y) for x in x_coords
                                 if (x, y) not in occupied_positions]
                # Sort by distance from center
                row_positions.sort(key=lambda pos: abs(pos[0] - x_center))
                available_positions.extend(row_positions)

            # Place intermediate objects on available grid positions
            for i, (obj, obj_type) in enumerate(intermediate_objects):
                if i < len(available_positions):
                    new_x, new_y = available_positions[i]
                else:
                    # Fallback to non-grid positioning if we run out of grid positions
                    start_x = self.x_lb + self.domino_width
                    spacing = self.domino_width * 1.5
                    new_x = start_x + i * spacing
                    new_y = (self.y_lb + self.y_ub) / 2

                if obj_type == "domino":
                    obj_dict[obj] = {
                        "x": new_x,
                        "y": new_y,
                        "z": self.z_lb + self.domino_height / 2,
                        "yaw": 0.0,  # Reset rotation to upright
                        "roll": 0.0,  # Reset tilt to upright
                        "r": self.domino_color[0],
                        "g": self.domino_color[1],
                        "b": self.domino_color[2],
                        "is_held": 0.0,
                    }
                elif obj_type == "wall":
                    obj_dict[obj] = {
                        "x": new_x,
                        "y": new_y,
                        "z": self.table_height + self.obstacle_wall_height / 2,
                        "rot": 0.0,
                    }
        else:
            # Original non-grid positioning
            # Calculate positions for intermediate objects
            # Place them in a row near x_lb with even spacing
            start_x = self.x_lb + self.domino_width  # Start a bit inside the boundary
            spacing = self.domino_width * 1.5  # Space between objects
            y_position = (self.y_lb +
                          self.y_ub) / 2  # Middle of the table in y direction

            # Update positions for intermediate objects
            for i, (obj, obj_type) in enumerate(intermediate_objects):
                new_x = start_x + i * spacing

                if obj_type == "domino":
                    obj_dict[obj] = {
                        "x": new_x,
                        "y": y_position,
                        "z": self.z_lb + self.domino_height / 2,
                        "yaw": 0.0,  # Reset rotation to upright
                        "roll": 0.0,  # Reset tilt to upright
                        "r": self.domino_color[0],
                        "g": self.domino_color[1],
                        "b": self.domino_color[2],
                        "is_held": 0.0,
                    }
                elif obj_type == "wall":
                    obj_dict[obj] = {
                        "x": new_x,
                        "y": y_position,
                        "z": self.table_height + self.obstacle_wall_height / 2,
                        "rot": 0.0,
                    }

        return obj_dict


if __name__ == "__main__":

    CFG.seed = 0
    CFG.env = "pybullet_domino_fan"
    # CFG.domino_initialize_at_finished_state = False
    # CFG.domino_use_domino_blocks_as_target = True
    # CFG.domino_use_grid = True
    # CFG.domino_fan_use_grid = False
    CFG.num_train_tasks = 10
    CFG.num_test_tasks = 2
    CFG.domino_fan_train_grid_size = (8, 8)
    CFG.domino_initialize_at_finished_state = True
    CFG.domino_domino_on_stairs = True
    env = PyBulletDominoGridFanEnv(use_gui=True)
    # # Set up test configurations for the example
    # CFG.domino_test_num_dominos = [3]
    # CFG.domino_test_num_targets = [1]
    # CFG.domino_test_num_pivots = [1]

    tasks = env._generate_train_tasks()

    for task in tasks:
        env._reset_state(task.init)
        # print(
        #     f"init state: {pformat(utils.abstract(task.init, env.predicates))}\n"
        # )
        # print(f"goal: {task.goal}\n")
        # print(pformat(task.init.pretty_str()), '\n')

        for i in range(100):
            action = Action(
                np.array(env._pybullet_robot.initial_joint_positions))
            env.step(action)
            time.sleep(0.01)
