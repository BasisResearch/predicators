"""Observable simulation core of the fan environment.

This module is the fan env's BASE SIM: scene geometry and physical
constants, object and body construction, state read/write, and the
switch/fan mechanics - everything needed to run rigid-body rollouts of
the arena. It deliberately contains NO residual dynamics (how the wind
moves the ball lives in the ``PyBulletFanEnv`` subclass's
``_domain_specific_step``), no task generation, and no predicate /
goal semantics.

That boundary is a visibility contract, enforced structurally rather
than by redaction: when ``CFG.agent_sim_provide_base_sim_source`` is
on, THIS FILE is copied verbatim into the learning agent's sandbox as
reference material ("the robot knows its own simulator"), so the file
the agent reads is byte-identical to the code its base-sim rollouts
execute. Anything that would leak the learning target - the wind force
law and its constants, the task distribution, goal thresholds - must
live in ``pybullet_fan.py`` (the concrete subclass), never here.
"""
from typing import Any, ClassVar, Dict, List, Optional, Set, Tuple

import numpy as np
import pybullet as p

from predicators.envs.pybullet_env import PyBulletEnv
from predicators.pybullet_helpers.geometry import Pose3D, Quaternion
from predicators.pybullet_helpers.objects import cap_switch_joint_travel, \
    create_object, create_pybullet_block, create_pybullet_sphere, \
    update_object
from predicators.pybullet_helpers.robots import SingleArmPyBulletRobot
from predicators.settings import CFG
from predicators.structs import Object, State, Type


class PyBulletFanBaseEnv(PyBulletEnv):
    """Sim core of the fan arena: a ball on a walled grid table, four banks of
    fans, and four switches.

    Abstract on purpose - it defines no name, predicates, tasks, or
    domain-specific step, so env discovery skips it; the concrete env
    is ``PyBulletFanEnv``.
    """

    @classmethod
    def get_base_sim_source_files(cls) -> List[str]:
        # This module IS the visible sim core (see the module docstring's
        # visibility contract); pybullet_env.py is the generic engine it
        # is built on. pybullet_fan.py (residual dynamics, task
        # generation, predicates) must never be listed here.
        return [
            "predicators/envs/pybullet_fan_base.py",
            "predicators/envs/pybullet_env.py",
        ]

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
    # Two tables side by side for extra workspace (mirrors pybullet_domino).
    # The second table is offset by +table_width/2 in y.
    table_width: ClassVar[float] = 1.0

    # Workspace bounds
    x_lb: ClassVar[float] = 0.4
    x_ub: ClassVar[float] = 1.1
    y_lb: ClassVar[float] = 1.1
    # Two tables span y in [1.1, 2.1]; y_ub is the upper workspace bound used
    # to clamp the fan-blown ball. Must cover the full grid (up to
    # loc_y_ub = up_fan_y - 0.05 ~= 1.97), so 2.1 (single-table 1.6 would clip
    # the ball at the upper cells). robot_init_y / switch_y below are anchored
    # to the front (y_lb) so they don't drift up into the grid.
    y_ub: ClassVar[float] = 2.1
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
    # Front-anchored (robot only reaches the front switches, not the grid).
    robot_init_y: ClassVar[float] = y_lb - 0.02
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
    num_left_fans: ClassVar[int] = 5
    num_right_fans: ClassVar[int] = 5
    num_back_fans: ClassVar[int] = 5
    num_front_fans: ClassVar[int] = 5

    # -------------------------------------------------------------------------
    # Fan Physical Properties
    # -------------------------------------------------------------------------
    fan_scale: ClassVar[float] = 0.08
    fan_x_len: ClassVar[float] = 0.2 * fan_scale  # Length of fan blades
    fan_y_len: ClassVar[float] = 1.5 * fan_scale  # Width of fan blades
    fan_z_len: ClassVar[float] = 1.5 * fan_scale  # Height of fan base

    # Each fan stands on its own narrow pedestal. The posts are visual-only:
    # the fans are already fixed bodies, and collision geometry here could
    # catch a ball after it leaves the exposed platforms.
    fan_support_x_len: ClassVar[float] = 0.02
    fan_support_y_len: ClassVar[float] = 0.06
    # Reach the fan's base origin. The imported mesh begins just below that
    # origin, so this gives a small, deliberate overlap with no visible seam.
    fan_support_height: ClassVar[float] = table_height + fan_z_len / 2
    fan_support_color: ClassVar[Tuple[float, float, float,
                                      float]] = (0.32, 0.34, 0.36, 1.0)

    # -------------------------------------------------------------------------
    # Fan Positioning
    # -------------------------------------------------------------------------
    left_fan_x: ClassVar[float] = x_lb - fan_x_len * 5
    right_fan_x: ClassVar[float] = x_ub + fan_x_len * 5
    # Front (far) fan row sits at the upper edge of the second table. The two
    # tables together span y in [1.1, 2.1]; keep the fan body just inside that
    # far edge. Deepening the arena this way gives the left/right sides room
    # for 5 evenly-spaced fans, and re-centers the grid (loc_y_mid, the
    # midpoint of down_fan_y/up_fan_y) between the top and bottom fan rows.
    # Both rows are pushed this much further from the robot than the
    # geometry above would otherwise place them. The down row's rotor
    # link reaches ~0.093 behind the switch row, and a SwitchOff push -
    # whose approach waypoint sits at switch_y + approach_distance, on
    # the far side of the switch - wedges the wrist against it for
    # approach distances the params space advertises as legal (0.08
    # stalls, 0.06 clears). Shifting BOTH rows keeps the arena's
    # y-extent the same size and just translates it, spending the
    # headroom at the far edge (y_ub - up_fan_y = 0.08, less the rotor's
    # ~0.015 overhang). Everything downstream - fan_y_lb/ub and the
    # loc_* grid bounds - is derived from these two, so the grid
    # translates with the fans.
    fan_row_y_shift: ClassVar[float] = 0.03
    up_fan_y: ClassVar[float] = 2.02 + fan_row_y_shift
    down_fan_y: ClassVar[float] = y_lb + fan_x_len / 2 + 0.1 + fan_row_y_shift

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

    # Switch positioning: front-anchored so the switches stay at the near edge
    # (out of the grid), independent of the workspace upper bound y_ub.
    switch_y: ClassVar[float] = y_lb  # Y position of switches
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
    # High linear damping acts as the ball's air/rolling resistance: it
    # sets the terminal speed the ball reaches under any continuously
    # held force, and keeps that force comfortably above the stiction /
    # table-seam creep threshold so the ball rolls reliably from rest.
    ball_linear_damping: ClassVar[float] = 120.0
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

    # Boundary walls around grid. The walls must clear the ball's
    # equator (center sits ball_radius above the table): with lower
    # walls the ball leans on the wall's TOP EDGE while traveling along
    # a wall-adjacent row, and the slanted edge contact carries part of
    # its weight like a rail, so a driven ball travels much faster
    # there than on the open table. At 0.06 the contact is a plain
    # side touch at the equator and travel speed matches free rolling.
    boundary_wall_height: ClassVar[float] = 0.06
    boundary_wall_thickness: ClassVar[float] = 0.002
    boundary_wall_color: ClassVar[Tuple[float, float, float,
                                        float]] = (0.9, 0.9, 0.9, 1)

    # -------------------------------------------------------------------------
    # Target Properties
    # -------------------------------------------------------------------------
    target_thickness: ClassVar[float] = 0.00001
    target_mass: ClassVar[float] = 0.0
    # Match the table's lateral friction: the pad covers a full grid
    # cell that every final approach rolls across, and a slick pad
    # (0.04, vs the table's 0.5) let a driven ball slide over it about
    # twice as fast as it rolls on the table, making it ping-pong
    # across the target instead of resting on it.
    target_friction: ClassVar[float] = 0.5
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
    # Types
    # -------------------------------------------------------------------------
    _robot_type = Type("robot",
                       ["x", "y", "z", "fingers", "roll", "tilt", "wrist"],
                       angular_features=["roll", "tilt", "wrist"])
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
        sim_features=["id", "side_idx", "fan_ids", "joint_ids", "support_ids"],
        angular_features=["rot"])
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
        sim_features=["id", "joint_id", "side_idx"],
        angular_features=["rot"])
    # Blockers. ``x_len``/``y_len``/``z_len`` are the side lengths of the
    # body's WORLD-AXIS-ALIGNED bounding box (not the body frame), so they
    # always pair with the ``x``/``y``/``z`` pose features: a collision rule
    # can write ``abs(bx - wx) < w.x_len / 2 + reach`` without consulting
    # ``rot``. For a box at rot = 0 or +/-pi/2 (the only rotations this env
    # uses; see ``wall_rot``) the AABB is exact.
    #
    # ``wall`` is the task's obstacle walls; ``boundary`` is the four slabs
    # enclosing the grid. They are deliberately DISTINCT types rather than
    # one type or a hierarchy: the dynamics treat them identically (one
    # contact rule iterating both), while predicates and NSRTs quantify over
    # ``wall`` alone, so the four always-present, never-manipulable boundary
    # slabs never enter symbolic grounding.
    _wall_type = Type("wall",
                      ["x", "y", "z", "rot", "x_len", "y_len", "z_len"],
                      angular_features=["rot"])
    # The boundary extents are task-dependent (they track the grid size), so
    # unlike the obstacle walls they cannot come from a class constant. They
    # are cached in sim_data by _reposition_boundary_walls, which is the same
    # code that writes them into PyBullet - so _get_state reads back exactly
    # the geometry the ball is colliding with.
    _boundary_type = Type("boundary",
                          ["x", "y", "z", "rot", "x_len", "y_len", "z_len"],
                          sim_features=["id", "x_len", "y_len", "z_len"],
                          angular_features=["rot"])
    _platform_type = Type("platform",
                          ["x", "y", "z", "rot", "x_len", "y_len", "z_len"],
                          sim_features=["id", "x_len", "y_len", "z_len"],
                          angular_features=["rot"])
    _ramp_type = Type(
        "ramp", ["x", "y", "z", "rot", "x_len", "y_len", "z_len", "rise"],
        sim_features=["id", "x_len", "y_len", "z_len", "rise"],
        angular_features=["rot"])
    # ``radius`` completes the contact geometry: with it and the blocker
    # extents above, the ball's stop distance is pure geometry over
    # observable features instead of a fitted constant.
    _ball_type = Type("ball", ["x", "y", "z", "radius"])
    _target_type = Type("target", ["x", "y", "z", "rot", "is_hit"],
                        angular_features=["rot"])

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

        # Fans and switches - one object per side (left=0, right=1,
        # down=2, up=3).
        self._switch_sides = ["left", "right", "down", "up"]
        self._fans: List[Object] = [
            Object(f"fan_{i}", self._fan_type)
            for i in range(len(self._switch_sides))
        ]
        self._switches: List[Object] = [
            Object(f"switch_{i}", self._switch_type)
            for i in range(len(self._switch_sides))
        ]

        # Maze walls - create enough for the maximum walls per task
        max_walls_per_task = max(max(CFG.fan_train_num_walls_per_task),
                                 max(CFG.fan_test_num_walls_per_task))
        self._walls = [
            Object(f"wall{i}", self._wall_type)
            for i in range(max_walls_per_task)
        ]

        # Boundary slabs enclosing the grid. Unlike the obstacle walls these
        # are always present, one per side, named after the same directions
        # the fans/switches use (left/right/down/up).
        self._boundary_sides = ["left", "right", "down", "up"]
        self._boundaries = [
            Object(f"boundary_{side}", self._boundary_type)
            for side in self._boundary_sides
        ]
        self._platforms = (
            [Object(f"platform_{i}", self._platform_type)
             for i in range(3)] if CFG.fan_exposed_transfer else [])
        if CFG.fan_ramp_transfer:
            if not CFG.fan_inertial_transfer:
                raise ValueError("Ramp transfer requires inertial transfer")
            self._platforms.append(Object("ramp_0", self._ramp_type))

        # Ball
        self._ball = Object("ball", self._ball_type)

        # Target
        self._target = Object("target", self._target_type)

        super().__init__(use_gui=use_gui, **kwargs)

    @property
    def types(self) -> Set[Type]:
        # Physical-only types (agent runs grid-free). The grid helper types
        # (loc / side) are provided by PyBulletFanGroundTruthTypeFactory and
        # injected only for the oracle / process-planning approaches.
        types = {
            self._robot_type, self._fan_type, self._switch_type,
            self._wall_type, self._boundary_type, self._ball_type,
            self._target_type
        }
        if CFG.fan_exposed_transfer:
            types.add(self._platform_type)
        if CFG.fan_ramp_transfer:
            types.add(self._ramp_type)
        return types

    # -------------------------------------------------------------------------
    # PyBullet Initialization
    # -------------------------------------------------------------------------
    @classmethod
    def initialize_pybullet(
            cls, using_gui: bool
    ) -> Tuple[int, SingleArmPyBulletRobot, Dict[str, Any]]:
        physics_client_id, pybullet_robot, bodies = super(
        ).initialize_pybullet(using_gui)

        # Two tables side by side for extra workspace (mirrors
        # pybullet_domino). The second table is offset by +table_width/2 in y.
        table_id = create_object(
            asset_path="urdf/table.urdf",
            position=cls.table_pos,
            orientation=cls.table_orn,
            scale=cls.table_scale,
            use_fixed_base=True,
            physics_client_id=physics_client_id,
        )
        bodies["table_id"] = table_id
        table_id2 = create_object(
            asset_path="urdf/table.urdf",
            position=(cls.table_pos[0], cls.table_pos[1] + cls.table_width / 2,
                      cls.table_pos[2]),
            orientation=cls.table_orn,
            scale=cls.table_scale,
            use_fixed_base=True,
            physics_client_id=physics_client_id,
        )
        bodies["table_id2"] = table_id2
        if CFG.fan_exposed_transfer:
            # The observed platforms, not an invisible full table, support
            # the ball. Keep a separate workbench under the robot switches.
            for table in (table_id, table_id2):
                p.resetBasePositionAndOrientation(
                    table, (0.0, -10.0, 0.0),
                    cls.table_orn,
                    physicsClientId=physics_client_id)
            bodies["switch_bench"] = create_pybullet_block(
                color=(0.55, 0.40, 0.25, 1.0),
                half_extents=(0.5, 0.11, cls.table_height / 2),
                mass=0.0,
                friction=0.5,
                position=(0.75, 1.06, cls.table_height / 2),
                orientation=(0, 0, 0, 1),
                physics_client_id=physics_client_id)

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

        # Give every fan an independent floor-mounted pedestal. These are
        # deliberately non-colliding so the visible support cannot bridge an
        # exposed gap or rescue a ball that should fall to the floor.
        def create_fan_supports(count: int) -> List[int]:
            visual = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=(cls.fan_support_x_len / 2,
                             cls.fan_support_y_len / 2,
                             cls.fan_support_height / 2),
                rgbaColor=cls.fan_support_color,
                physicsClientId=physics_client_id)
            return [
                p.createMultiBody(baseMass=0.0,
                                  baseCollisionShapeIndex=-1,
                                  baseVisualShapeIndex=visual,
                                  basePosition=(0.0, -10.0,
                                                cls.fan_support_height / 2),
                                  baseOrientation=(0, 0, 0, 1),
                                  physicsClientId=physics_client_id)
                for _ in range(count)
            ]

        bodies["fan_support_ids_left"] = create_fan_supports(cls.num_left_fans)
        bodies["fan_support_ids_right"] = create_fan_supports(
            cls.num_right_fans)
        bodies["fan_support_ids_back"] = create_fan_supports(cls.num_back_fans)
        bodies["fan_support_ids_front"] = create_fan_supports(
            cls.num_front_fans)

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
        if CFG.fan_inertial_transfer and not CFG.fan_exposed_transfer:
            raise ValueError("Inertial transfer requires exposed transfer")
        ball_id = create_pybullet_sphere(
            color=cls.ball_color,
            radius=cls.ball_radius,
            mass=cls.ball_mass,
            friction=0.5 if CFG.fan_exposed_transfer else cls.ball_friction,
            # Match lateral with spinning so the ball resists rotating around
            # the contact normal — necessary for it to "stick" where the fan
            # parks it instead of pinwheeling.
            spinning_friction=(0.01 if CFG.fan_exposed_transfer else
                               cls.ball_friction),
            position=(0.75, 1.35, cls.table_height + cls.ball_height_offset),
            orientation=p.getQuaternionFromEuler([0, 0, 0]),
            physics_client_id=physics_client_id)
        p.changeDynamics(ball_id,
                         -1,
                         linearDamping=(1.0 if CFG.fan_exposed_transfer else
                                        cls.ball_linear_damping),
                         angularDamping=(0.1 if CFG.fan_exposed_transfer else
                                         cls.ball_angular_damping),
                         physicsClientId=physics_client_id)
        if CFG.fan_inertial_transfer:
            # Keep contact traction, but retain momentum between fan pulses.
            # Shared by the real environment and reconstructed simulator base.
            p.changeDynamics(ball_id,
                             -1,
                             linearDamping=0.08,
                             angularDamping=0.008,
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
        # Match the table's rolling friction (create_pybullet_block only
        # sets lateral). The pad covers the full target cell and the
        # ball ROLLS ON TOP of it; with zero rolling resistance a driven
        # ball speeds up several-fold there and shoots across the
        # target instead of resting on it.
        p.changeDynamics(target_id,
                         -1,
                         rollingFriction=0.001,
                         physicsClientId=physics_client_id)
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
        self._table_ids = [
            pybullet_bodies["table_id"], pybullet_bodies["table_id2"]
        ]
        # 0 = left, 1 = right, 2 = back, 3 = front

        # Store all fan IDs grouped by side
        fan_ids_by_side = [
            pybullet_bodies["fan_ids_left"],  # side 0
            pybullet_bodies["fan_ids_right"],  # side 1
            pybullet_bodies["fan_ids_back"],  # side 2
            pybullet_bodies["fan_ids_front"]  # side 3
        ]
        support_ids_by_side = [
            pybullet_bodies["fan_support_ids_left"],
            pybullet_bodies["fan_support_ids_right"],
            pybullet_bodies["fan_support_ids_back"],
            pybullet_bodies["fan_support_ids_front"],
        ]

        # Update each fan object with its side's fan IDs and joint IDs
        for side_idx, fan_obj in enumerate(self._fans):
            fan_obj.side_idx = side_idx
            fan_obj.fan_ids = fan_ids_by_side[side_idx]
            fan_obj.support_ids = support_ids_by_side[side_idx]
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
            cap_switch_joint_travel(switch_obj.id, switch_obj.joint_id,
                                    self.switch_joint_scale,
                                    self._physics_client_id)
            switch_obj.side_idx = i  # 0=left,1=right,2=back,3=front

        for wall, obj_id in zip(self._walls, pybullet_bodies["wall_ids"]):
            wall.id = obj_id
        self._ball.id = pybullet_bodies["ball_id"]
        self._target.id = pybullet_bodies["target_id"]

        # Boundary slab bodies, parallel to self._boundaries. They are
        # rebuilt from the state (a box collision shape cannot be resized
        # in place) by _reposition_boundary_walls, which also refreshes the
        # Objects' ids.
        # pylint: disable=attribute-defined-outside-init
        self._boundary_wall_ids: List[int] = []
        # The (pose, extents) spec the current bodies were built for; lets
        # _reposition_boundary_walls skip an identical rebuild.
        self._boundary_wall_spec: Optional[Tuple[Tuple[float, ...],
                                                 ...]] = None

    # -------------------------------------------------------------------------
    # Read state from PyBullet
    # -------------------------------------------------------------------------
    def _get_object_ids_for_held_check(self) -> List[int]:
        return []

    def _set_domain_specific_state(self, state: State) -> None:
        for switch_obj in self._switches:
            want_on = bool(state.get(switch_obj, "is_on") > 0.5)
            # Only reconcile a lever whose on/off reading actually
            # disagrees with the requested one. `is_on` is a threshold
            # over a continuous joint (see _is_switch_on), while
            # _set_switch_on snaps the joint to its travel *limit* - so
            # re-imposing an already-matching value teleports the lever
            # out from under a gripper that is mid-push and discards the
            # contact. _set_state runs on every step of a combined
            # base+learned simulator rollout (the learned rules edit
            # features the engine also holds, so State.allclose misses),
            # which let a jammed SwitchOn converge in the belief sim
            # while it stalled for real.
            if self._is_switch_on(switch_obj.id) != want_on:
                self._set_switch_on(switch_obj.id, want_on)

        # Position all fans correctly based on their side
        self._position_fans_on_sides()

        # Rebuild the boundary slabs from their own state features.
        self._reposition_boundary_walls(state)

        oov_x, oov_y = self._out_of_view_xy
        # Move irrelavent walls oov
        wall_obj = state.get_objects(self._wall_type)
        for i in range(len(wall_obj), len(self._walls)):
            update_object(self._walls[i].id,
                          position=(oov_x, oov_y, 0.0),
                          physics_client_id=self._physics_client_id)

    def _boundary_named(self, obj: Object) -> Object:
        """The env-owned boundary Object for ``obj`` (matched by name).

        ``_set_state`` rebinds ``self._objects`` to the incoming State's
        own Object instances, whose ``sim_data`` (body id, cached
        extents) was written by whichever env produced that State. Every
        rebuild of the slabs reassigns body ids (PyBullet hands freed
        ids back in reverse), so a foreign instance's id can name a
        different slab here: reading through it permuted the four
        boundaries and rendered them as a cross through the arena (fan
        test task, 2026-09-03). Only the env-owned instance tracks this
        env's bodies.
        """
        for env_obj in self._boundaries + self._platforms:
            if env_obj.name == obj.name:
                return env_obj
        return obj

    def _get_object_state_dict(self, obj: Object) -> Dict[str, float]:
        if obj.type in (self._boundary_type, self._platform_type,
                        self._ramp_type):
            obj = self._boundary_named(obj)
        return super()._get_object_state_dict(obj)

    def _reset_single_object(self, obj: Object, state: State) -> None:
        """Skip the boundary slabs; they are rebuilt, not teleported.

        A box collision shape cannot be resized in place, so
        _reposition_boundary_walls destroys and recreates the boundary
        bodies from the state (refreshing the Objects' ids). It runs
        from _set_domain_specific_state, i.e. *after* this generic pose
        reset - so teleporting them here would dereference an id
        belonging to a body the previous rebuild already removed.
        """
        if obj.type in (self._boundary_type, self._platform_type,
                        self._ramp_type):
            return
        super()._reset_single_object(obj, state)

    def _remove_boundary_walls(self) -> None:
        """Tear down the current boundary slab bodies."""
        for wall_id in self._boundary_wall_ids:
            if wall_id >= 0:
                p.removeBody(wall_id, physicsClientId=self._physics_client_id)
        # pylint: disable=attribute-defined-outside-init
        self._boundary_wall_ids = []
        self._boundary_wall_spec = None
        for boundary_obj in self._boundaries + self._platforms:
            boundary_obj.id = None

    @staticmethod
    def _body_dims_from_aabb(x_len: float, y_len: float, z_len: float,
                             rot: float) -> Tuple[float, float, float]:
        """Body-frame side lengths of a box whose world AABB is the input.

        The blocker types publish world-axis-aligned extents (see
        ``_wall_type``), but PyBullet needs body-frame half-extents. The
        inversion is exact for the only rotations this env uses
        (multiples of pi/2): a quarter turn just swaps x and y.
        """
        if abs(np.sin(rot)) > 0.5:  # +/-pi/2
            return (y_len, x_len, z_len)
        return (x_len, y_len, z_len)

    @classmethod
    def _aabb_from_body_dims(cls, x_len: float, y_len: float, z_len: float,
                             rot: float) -> Tuple[float, float, float]:
        """World AABB side lengths of a box with the given body dims.

        Inverse of ``_body_dims_from_aabb`` (the map is an involution).
        """
        return cls._body_dims_from_aabb(x_len, y_len, z_len, rot)

    def _reposition_boundary_walls(self, state: State) -> None:
        """Rebuild the boundary slab bodies from their state features.

        The four ``boundary`` objects carry their own pose and extents, so
        this is a straight state -> PyBullet write with no grid inference:
        the arena geometry the ball collides with is exactly what the agent
        observes.

        No-op when the requested spec is unchanged. That matters because
        _set_state runs on every step of a combined base+learned simulator
        rollout, and a box collision shape cannot be resized in place - the
        rebuild removes bodies, discarding any contact they were part of.
        """
        present = [b for b in self._boundaries + self._platforms if b in state]
        if not present:
            # A state with no boundary objects describes an open arena.
            self._remove_boundary_walls()
            return

        spec = tuple(
            (float((self._boundaries + self._platforms).index(b)), ) + tuple(
                float(state.get(b, f))
                for f in ("x", "y", "z", "rot", "x_len", "y_len", "z_len")) +
            (float(state.get(b, "rise")) if b.type == self._ramp_type else 0.0,
             ) for b in present)
        if self._boundary_wall_ids and spec == self._boundary_wall_spec:
            return
        self._remove_boundary_walls()

        wall_ids = []
        for boundary_obj, (_, bx, by, bz, brot, x_len, y_len, z_len,
                           rise) in zip(present, spec):
            dims = self._body_dims_from_aabb(x_len, y_len, z_len, brot)
            is_platform = boundary_obj.type in (self._platform_type,
                                                self._ramp_type)
            if boundary_obj.type == self._ramp_type:
                wall_id = self._create_ramp_body(bx, by, bz, brot, dims, rise)
                boundary_obj.rise = rise
            else:
                wall_id = create_pybullet_block(
                    color=((0.55, 0.40, 0.25,
                            1.0) if is_platform else self.boundary_wall_color),
                    half_extents=(dims[0] / 2, dims[1] / 2, dims[2] / 2),
                    mass=self.wall_mass,
                    friction=0.5 if is_platform else self.wall_friction,
                    position=(bx, by, bz),
                    orientation=p.getQuaternionFromEuler([0.0, 0.0, brot]),
                    physics_client_id=self._physics_client_id)
            boundary_obj.id = wall_id
            boundary_obj.x_len = x_len
            boundary_obj.y_len = y_len
            boundary_obj.z_len = z_len
            if is_platform:
                p.changeDynamics(
                    wall_id,
                    -1,
                    rollingFriction=(0.0001
                                     if CFG.fan_inertial_transfer else 0.001),
                    physicsClientId=self._physics_client_id)
            wall_ids.append(wall_id)

        # pylint: disable=attribute-defined-outside-init
        self._boundary_wall_ids = wall_ids
        self._boundary_wall_spec = spec

    def _create_ramp_body(self, x: float, y: float, z: float, yaw: float,
                          dims: Tuple[float, float,
                                      float], rise: float) -> int:
        """Convex wedge with a top descending toward local +x.

        Published dimensions bound the whole wedge, including its rise.
        The high top edge is at z + height/2; the low edge is rise
        lower.
        """
        length, width, height = dims
        if not 0 < rise < height:
            raise ValueError("Ramp rise must be positive and below its height")
        vertices = [[
            sx * length / 2, sy * width / 2,
            (-height / 2 if bottom else height / 2 - (rise if sx > 0 else 0))
        ] for bottom in (True, False) for sx in (-1, 1) for sy in (-1, 1)]
        faces = [
            0, 3, 2, 0, 1, 3, 4, 7, 5, 4, 6, 7, 0, 5, 1, 0, 4, 5, 2, 7, 6, 2,
            3, 7, 0, 6, 4, 0, 2, 6, 1, 7, 3, 1, 5, 7
        ]
        collision = p.createCollisionShape(
            p.GEOM_MESH,
            vertices=vertices,
            indices=faces,
            flags=p.GEOM_FORCE_CONCAVE_TRIMESH,
            physicsClientId=self._physics_client_id)
        visual = p.createVisualShape(p.GEOM_MESH,
                                     vertices=vertices,
                                     indices=faces,
                                     rgbaColor=(0.7, 0.5, 0.25, 1),
                                     physicsClientId=self._physics_client_id)
        body = p.createMultiBody(baseMass=0,
                                 baseCollisionShapeIndex=collision,
                                 baseVisualShapeIndex=visual,
                                 basePosition=(x, y, z),
                                 baseOrientation=p.getQuaternionFromEuler(
                                     [0, 0, yaw]),
                                 physicsClientId=self._physics_client_id)
        p.changeDynamics(body,
                         -1,
                         lateralFriction=0.5,
                         collisionMargin=0.0,
                         physicsClientId=self._physics_client_id)
        return body

    @classmethod
    def _fan_bank_poses(cls,
                        side_idx: int) -> List[Tuple[float, float, float]]:
        """Return evenly spaced ``(x, y, yaw)`` poses for one fan bank."""
        if side_idx in (0, 1):
            count = cls.num_left_fans if side_idx == 0 else cls.num_right_fans
            coordinates = np.linspace(cls.fan_y_lb, cls.fan_y_ub, count)
            x = (cls.left_fan_x if side_idx == 0 else cls.right_fan_x +
                 (0.4 if CFG.fan_ramp_transfer else 0.0))
            yaw = 0.0 if side_idx == 0 else np.pi
            return [(x, float(y), yaw) for y in coordinates]
        if side_idx in (2, 3):
            count = cls.num_back_fans if side_idx == 2 else cls.num_front_fans
            coordinates = np.linspace(cls.fan_x_lb, cls.fan_x_ub, count)
            y = cls.down_fan_y if side_idx == 2 else cls.up_fan_y
            yaw = np.pi / 2 if side_idx == 2 else -np.pi / 2
            return [(float(x), y, yaw) for x in coordinates]
        raise ValueError(f"Unknown fan side {side_idx}")

    def _position_fans_on_sides(self) -> None:
        """Position each fan and its floor-mounted support."""
        for fan_obj in self._fans:
            poses = self._fan_bank_poses(fan_obj.side_idx)
            assert len(fan_obj.fan_ids) == len(
                fan_obj.support_ids) == len(poses)
            for fan_id, support_id, (px, py,
                                     yaw) in zip(fan_obj.fan_ids,
                                                 fan_obj.support_ids, poses):
                orientation = p.getQuaternionFromEuler([0.0, 0.0, yaw])
                update_object(support_id,
                              position=(px, py, self.fan_support_height / 2),
                              orientation=orientation,
                              physics_client_id=self._physics_client_id)
                update_object(fan_id,
                              position=(px, py, self.table_height +
                                        self.fan_z_len / 2),
                              orientation=orientation,
                              physics_client_id=self._physics_client_id)

    def _get_domain_specific_feature(self, obj: Object, feature: str) -> float:
        """Extract features for creating the State object."""
        if obj.type == self._ball_type:
            if feature == "radius":
                return self.ball_radius
        if obj.type == self._wall_type and feature in ("x_len", "y_len",
                                                       "z_len"):
            # Obstacle walls are all built from the same class constants
            # (see initialize_pybullet); only their yaw varies.
            rot = p.getEulerFromQuaternion(
                p.getBasePositionAndOrientation(
                    obj.id, physicsClientId=self._physics_client_id)[1])[2]
            dims = self._aabb_from_body_dims(self.wall_x_len, self.wall_y_len,
                                             self.obstacle_wall_height, rot)
            return dims[("x_len", "y_len", "z_len").index(feature)]
        if obj.type in (self._boundary_type, self._platform_type,
                        self._ramp_type) and feature in ("x_len", "y_len",
                                                         "z_len", "rise"):
            # Cached by _reposition_boundary_walls when it built the body,
            # on the env-owned instance (see _boundary_named).
            cached = getattr(self._boundary_named(obj), feature)
            if cached is None:
                raise ValueError(
                    f"Boundary {obj.name} has no body yet; "
                    f"_reposition_boundary_walls must run before _get_state.")
            return float(cached)
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
        # target.is_hit is computed by the concrete subclass: its
        # proximity threshold is goal semantics, which this module's
        # visibility contract keeps out of the base sim.
        raise ValueError(f"Unknown feature {feature} for object {obj}")

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
