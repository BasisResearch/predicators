"""PyBullet environment combining dominoes and fan-blown ball.

This environment extends PyBulletDominoEnv with:
- Dominoes that can topple through collisions (from base domino env)
- A ball that can be blown by fans AND can knock down dominoes through collisions
- Fans controlled by switches (fans affect ball only, not dominoes directly)
- Continuous positioning for the ball (not grid-based)
- All assets (fans, switches, ball, dominoes) are loaded in every task
- Tasks have ONE goal type: either all dominoes toppled OR ball at target

Example usage:
python predicators/main.py --approach oracle --env pybullet_domino_fan \\
--seed 0 --num_test_tasks 1 --use_gui --debug --num_train_tasks 0 \\
--sesame_max_skeletons_optimized 1 --make_failure_videos --video_fps 20 \\
--pybullet_camera_height 900 --pybullet_camera_width 900
"""

from typing import Any, ClassVar, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pybullet as p

from predicators import utils
from predicators.envs.pybullet_domino import PyBulletDominoEnv
from predicators.envs.pybullet_env import create_pybullet_block, \
    create_pybullet_sphere
from predicators.pybullet_helpers.objects import create_object, update_object
from predicators.pybullet_helpers.robots import SingleArmPyBulletRobot
from predicators.settings import CFG
from predicators.structs import Action, EnvironmentTask, GroundAtom, Object, \
    Predicate, State, Type


class PyBulletDominoFanEnv(PyBulletDominoEnv):
    """A PyBullet environment combining dominoes and fan-blown ball.

    Extends PyBulletDominoEnv with:
    - Ball that is blown by fans and can knock down dominoes through collisions
    - Fans affect ball only (not dominoes directly)
    - Continuous positioning for ball (not grid-based)
    - All assets (fans, switches, ball, dominoes) present in every task
    - Tasks can have domino goals OR ball goals (one type per task)
    """

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
    wind_force_magnitude: ClassVar[float] = 2  # Balanced for controlled speed
    joint_motor_force: ClassVar[float] = 20.0

    # Fan Positioning (calculated based on parent workspace)
    # Note: These reference parent class variables, calculated at class level
    left_fan_x: ClassVar[float] = PyBulletDominoEnv.x_lb - 0.2 * 0.08 * 5
    right_fan_x: ClassVar[float] = PyBulletDominoEnv.x_ub + 0.2 * 0.08 * 5
    up_fan_y: ClassVar[float] = PyBulletDominoEnv.y_ub + \
        PyBulletDominoEnv.table_width / 2 + 0.2 * 0.08 / 2
    down_fan_y: ClassVar[float] = PyBulletDominoEnv.y_lb + 0.2 * 0.08 / 2 + 0.1

    # Fan placement boundaries
    fan_y_lb: ClassVar[
        float] = down_fan_y + 0.2 * 0.08 / 2 + 1.5 * 0.08 / 2 + 0.01
    fan_y_ub: ClassVar[
        float] = up_fan_y - 0.2 * 0.08 / 2 - 1.5 * 0.08 / 2 - 0.01
    fan_x_lb: ClassVar[
        float] = left_fan_x + 0.2 * 0.08 / 2 + 1.5 * 0.08 / 2 + 0.01
    fan_x_ub: ClassVar[
        float] = right_fan_x - 0.2 * 0.08 / 2 - 1.5 * 0.08 / 2 - 0.01

    # =========================================================================
    # SWITCH CONFIGURATION
    # =========================================================================
    switch_scale: ClassVar[float] = 1.0
    switch_joint_scale: ClassVar[float] = 0.1
    switch_on_threshold: ClassVar[float] = 0.5
    switch_x_len: ClassVar[float] = 0.10
    switch_height: ClassVar[float] = 0.08

    # Switch positioning
    switch_y: ClassVar[float] = (PyBulletDominoEnv.y_lb +
                                 PyBulletDominoEnv.y_ub) * 0.5 - 0.25
    switch_base_x: ClassVar[float] = 0.60
    switch_x_spacing: ClassVar[float] = 0.08

    # =========================================================================
    # BALL CONFIGURATION
    # =========================================================================
    ball_radius: ClassVar[
        float] = 0.05  # 50% larger for better collision impact
    ball_mass: ClassVar[float] = 0.5  # Heavy enough to topple dominoes
    ball_friction: ClassVar[float] = 0.5  # Moderate friction to control speed
    ball_restitution: ClassVar[
        float] = 0.3  # Some bounciness for collision energy
    ball_height_offset: ClassVar[float] = ball_radius
    ball_linear_damping: ClassVar[
        float] = 0.5  # Moderate damping for controlled movement
    ball_angular_damping: ClassVar[float] = 0.3  # Some damping for stability
    ball_color: ClassVar[Tuple[float, float, float,
                               float]] = (0.0, 0.0, 1.0, 1)

    # Ball target (for ball goal tasks)
    target_thickness: ClassVar[float] = 0.00001
    target_mass: ClassVar[float] = 0.0
    target_friction: ClassVar[float] = 0.04
    target_color: ClassVar[Tuple[float, float, float, float]] = (0, 1, 0, 1.0)

    def __init__(self, use_gui: bool = True) -> None:
        """Initialize the domino-fan environment."""
        # Create fan/switch/ball types BEFORE calling super().__init__()
        self._fan_type = Type(
            "fan", ["x", "y", "z", "rot", "facing_side", "is_on"],
            sim_features=["id", "side_idx", "fan_ids", "joint_ids"])
        self._switch_type = Type(
            "switch", ["x", "y", "z", "rot", "controls_fan", "is_on"],
            sim_features=["id", "joint_id", "side_idx"])
        self._side_type = Type("side", ["side_idx"],
                               sim_features=["id", "side_idx"])
        self._ball_type = Type("ball", ["x", "y", "z"])
        # Note: _target_type for ball target, not domino target
        # We'll use a different name to avoid confusion with parent's target
        self._ball_target_type = Type("ball_target", ["x", "y", "z", "is_hit"])

        # Create fan, switch, side objects
        self._switch_sides = ["left", "right", "down", "up"]
        self._fans: List[Object] = []
        for i in range(4):
            fan_obj = Object(f"fan_{i}", self._fan_type)
            self._fans.append(fan_obj)

        self._switches: List[Object] = []
        for i in range(4):
            switch_obj = Object(f"switch_{i}", self._switch_type)
            self._switches.append(switch_obj)

        self._sides: List[Object] = []
        for side_str in self._switch_sides:
            side_obj = Object(side_str, self._side_type)
            self._sides.append(side_obj)

        # Ball and ball target
        self._ball = Object("ball", self._ball_type)
        self._ball_target = Object("ball_target", self._ball_target_type)

        # Call parent init (creates robot, dominoes, domino targets, pivots, directions)
        super().__init__(use_gui=use_gui)

        # Create fan/ball predicates AFTER super().__init__()
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
        self._Controls = Predicate("Controls",
                                   [self._switch_type, self._fan_type],
                                   self._Controls_holds)
        self._BallAtTarget = Predicate(
            "BallAtTarget", [self._ball_type, self._ball_target_type],
            self._BallAtTarget_holds)

    @classmethod
    def get_name(cls) -> str:
        return "pybullet_domino_fan"

    @property
    def predicates(self) -> Set[Predicate]:
        """Return all predicates including parent and fan/ball predicates."""
        parent_preds = super().predicates
        fan_preds = {
            self._FanOn,
            self._FanOff,
            self._FanFacingSide,
            self._Controls,
            self._BallAtTarget,
        }
        # Add switch predicates if controls relation is not known
        if not CFG.fan_known_controls_relation:
            fan_preds |= {self._SwitchOn, self._SwitchOff}

        return parent_preds | fan_preds

    @property
    def types(self) -> Set[Type]:
        """Return all types including parent and fan/ball types."""
        parent_types = super().types
        fan_types = {
            self._fan_type,
            self._switch_type,
            self._side_type,
            self._ball_type,
            self._ball_target_type,
        }
        return parent_types | fan_types

    @property
    def goal_predicates(self) -> Set[Predicate]:
        """Goals can be either ball at target OR dominoes toppled."""
        return {self._BallAtTarget,
                self._Toppled}  # type: ignore[attr-defined]

    # -------------------------------------------------------------------------
    # PyBullet Initialization
    # -------------------------------------------------------------------------

    @classmethod
    def initialize_pybullet(
            cls, using_gui: bool
    ) -> Tuple[int, SingleArmPyBulletRobot, Dict[str, Any]]:
        """Initialize PyBullet bodies including fans, switches, and ball."""
        # Call parent to create table, robot, dominoes, targets, pivots
        physics_client_id, pybullet_robot, bodies = super(
        ).initialize_pybullet(using_gui)

        # Create fans in four groups (5 per side = 20 total)
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

        # Create switches (one per side)
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
        p.changeDynamics(
            ball_id,
            -1,
            linearDamping=cls.ball_linear_damping,
            angularDamping=cls.ball_angular_damping,
            restitution=cls.ball_restitution,
            ccdSweptSphereRadius=cls.ball_radius * 0.9,  # Enable CCD
            physicsClientId=physics_client_id)
        bodies["ball_id"] = ball_id

        # Create ball target (flat green marker)
        ball_target_id = create_pybullet_block(
            color=cls.target_color,
            half_extents=(cls.ball_radius, cls.ball_radius,
                          cls.target_thickness),
            mass=cls.target_mass,
            friction=cls.target_friction,
            position=(0, 0, cls.table_height),
            orientation=p.getQuaternionFromEuler([0, 0, 0]),
            physics_client_id=physics_client_id)
        bodies["ball_target_id"] = ball_target_id

        return physics_client_id, pybullet_robot, bodies

    def _store_pybullet_bodies(self, pybullet_bodies: Dict[str, Any]) -> None:
        """Store references to PyBullet object IDs."""
        # Call parent first (handles dominoes, domino targets, pivots)
        super()._store_pybullet_bodies(pybullet_bodies)

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
        # left=0, right=1, down/back=2, up/front=3
        for i, side_obj in enumerate(self._sides):
            side_obj.side_idx = float(i)

        # Store ball and ball target
        self._ball.id = pybullet_bodies["ball_id"]
        self._ball_target.id = pybullet_bodies["ball_target_id"]

    # -------------------------------------------------------------------------
    # State Management
    # -------------------------------------------------------------------------

    def _reset_custom_env_state(self, state: State) -> None:
        """Reset environment to match the given state."""
        # Call parent first (handles dominoes, domino targets, pivots)
        super()._reset_custom_env_state(state)

        # Set switch states
        for switch_obj in self._switches:
            is_on_val = state.get(switch_obj, "is_on")
            self._set_switch_on(switch_obj.id, bool(is_on_val > 0.5))

        # Position fans on sides
        self._position_fans_on_sides()

        # Position ball
        ball_x = state.get(self._ball, "x")
        ball_y = state.get(self._ball, "y")
        ball_z = state.get(self._ball, "z")
        update_object(self._ball.id,
                      position=(ball_x, ball_y, ball_z),
                      physics_client_id=self._physics_client_id)

        # Position ball target
        target_x = state.get(self._ball_target, "x")
        target_y = state.get(self._ball_target, "y")
        target_z = state.get(self._ball_target, "z")
        update_object(self._ball_target.id,
                      position=(target_x, target_y, target_z),
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
        elif obj.type == self._side_type:
            if feature == "side_idx":
                return float(obj.side_idx)
        elif obj.type == self._ball_type:
            if feature in ["x", "y", "z"]:
                pos, _ = p.getBasePositionAndOrientation(
                    obj.id, physicsClientId=self._physics_client_id)
                return pos[{"x": 0, "y": 1, "z": 2}[feature]]
        elif obj.type == self._ball_target_type:
            if feature == "is_hit":
                bx = self._current_observation.get(self._ball, "x")
                by = self._current_observation.get(self._ball, "y")
                tx = self._current_observation.get(self._ball_target, "x")
                ty = self._current_observation.get(self._ball_target, "y")
                dist = np.sqrt((bx - tx)**2 + (by - ty)**2)
                return 1.0 if dist < CFG.domino_fan_ball_position_tolerance else 0.0
            # Fall through to parent for x, y, z

        # Fallback to parent implementation
        return super()._extract_feature(obj, feature)

    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------

    @staticmethod
    def _get_joint_id(obj_id: int, joint_name: str) -> int:
        """Get joint ID by name from PyBullet object."""
        num_joints = p.getNumJoints(obj_id)
        for j in range(num_joints):
            info = p.getJointInfo(obj_id, j)
            if info[1].decode("utf-8") == joint_name:
                return j
        return -1

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

        # Calculate fractional position
        frac = (j_pos / self.switch_joint_scale - j_min) / (j_max - j_min)
        return bool(frac > self.switch_on_threshold)

    def _set_switch_on(self, switch_id: int, on: bool) -> None:
        """Set switch on or off."""
        joint_id = self._get_joint_id(switch_id, "joint_0")
        if joint_id < 0:
            return
        info = p.getJointInfo(switch_id,
                              joint_id,
                              physicsClientId=self._physics_client_id)
        j_min, j_max = info[8], info[9]
        target_val = j_max if on else j_min
        p.resetJointState(switch_id,
                          joint_id,
                          target_val * self.switch_joint_scale,
                          physicsClientId=self._physics_client_id)

    # -------------------------------------------------------------------------
    # Fan Simulation
    # -------------------------------------------------------------------------

    def step(self, action: Action, render_obs: bool = False) -> State:
        """Execute action and simulate fans."""
        super().step(action, render_obs=render_obs)
        self._simulate_fans()
        final_state = self._get_state()
        self._current_observation = final_state
        return final_state

    def _simulate_fans(self) -> None:
        """Spin fans and apply forces to ball (ball can then collide with
        dominoes)."""
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
                # Apply force to ball (ball can collide with and topple dominoes)
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

    def _apply_fan_force_to_ball(self, fan_id: int, ball_id: int) -> None:
        """Apply wind force from fan to ball (ball can then collide with
        dominoes)."""
        _, orn_fan = p.getBasePositionAndOrientation(
            fan_id, physicsClientId=self._physics_client_id)

        if CFG.fan_fans_blow_opposite_direction:
            local_dir = np.array([-1.0, 0.0, 0.0])
        else:
            local_dir = np.array([1.0, 0.0, 0.0])

        rmat = np.array(p.getMatrixFromQuaternion(orn_fan)).reshape((3, 3))
        world_dir = rmat.dot(local_dir)
        pos_ball, _ = p.getBasePositionAndOrientation(
            ball_id, physicsClientId=self._physics_client_id)
        force_vec = self.wind_force_magnitude * world_dir
        p.applyExternalForce(objectUniqueId=ball_id,
                             linkIndex=-1,
                             forceObj=force_vec.tolist(),
                             posObj=pos_ball,
                             flags=p.WORLD_FRAME,
                             physicsClientId=self._physics_client_id)

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
                    py = left_coords[i]
                    pz = self.table_height + self.fan_z_len / 2
                    rot = [0.0, 0.0, 0.0]
                    update_object(fan_id,
                                  position=(px, py, pz),
                                  orientation=p.getQuaternionFromEuler(rot),
                                  physics_client_id=self._physics_client_id)

            elif side_idx == 1:  # right
                for i, fan_id in enumerate(fan_ids):
                    px = self.right_fan_x
                    py = right_coords[i]
                    pz = self.table_height + self.fan_z_len / 2
                    rot = [0.0, 0.0, np.pi]
                    update_object(fan_id,
                                  position=(px, py, pz),
                                  orientation=p.getQuaternionFromEuler(rot),
                                  physics_client_id=self._physics_client_id)

            elif side_idx == 2:  # back
                for i, fan_id in enumerate(fan_ids):
                    px = back_coords[i]
                    py = self.down_fan_y
                    pz = self.table_height + self.fan_z_len / 2
                    rot = [0.0, 0.0, np.pi / 2]
                    update_object(fan_id,
                                  position=(px, py, pz),
                                  orientation=p.getQuaternionFromEuler(rot),
                                  physics_client_id=self._physics_client_id)

            elif side_idx == 3:  # front
                for i, fan_id in enumerate(fan_ids):
                    px = front_coords[i]
                    py = self.up_fan_y
                    pz = self.table_height + self.fan_z_len / 2
                    rot = [0.0, 0.0, -np.pi / 2]
                    update_object(fan_id,
                                  position=(px, py, pz),
                                  orientation=p.getQuaternionFromEuler(rot),
                                  physics_client_id=self._physics_client_id)

    # -------------------------------------------------------------------------
    # Predicate Hold Functions
    # -------------------------------------------------------------------------

    def _FanOn_holds(self, state: State, objects: Sequence[Object]) -> bool:
        """Check if fan/switch is on."""
        obj = objects[0]
        is_on = state.get(obj, "is_on")
        return is_on > 0.5

    def _FanFacingSide_holds(self, state: State,
                             objects: Sequence[Object]) -> bool:
        """Check if fan faces a specific side."""
        fan, side = objects
        fan_side = state.get(fan, "facing_side")
        side_idx = state.get(side, "side_idx")
        return abs(fan_side - side_idx) < 0.1

    def _Controls_holds(self, state: State, objects: Sequence[Object]) -> bool:
        """Check if switch controls fan."""
        switch, fan = objects
        switch_controls = state.get(switch, "controls_fan")
        fan_side = state.get(fan, "facing_side")
        return abs(switch_controls - fan_side) < 0.1

    def _BallAtTarget_holds(self, state: State,
                            objects: Sequence[Object]) -> bool:
        """Check if ball is at target position."""
        ball, target = objects
        bx, by = state.get(ball, "x"), state.get(ball, "y")
        tx, ty = state.get(target, "x"), state.get(target, "y")
        dist = np.sqrt((bx - tx)**2 + (by - ty)**2)
        return dist < CFG.domino_fan_ball_position_tolerance

    # -------------------------------------------------------------------------
    # Task Generation
    # -------------------------------------------------------------------------

    def _generate_train_tasks(self) -> List[EnvironmentTask]:
        """Generate training tasks."""
        return self._make_tasks(
            num_tasks=CFG.num_train_tasks,
            possible_num_dominos=CFG.domino_train_num_dominos,
            possible_num_targets=CFG.domino_train_num_targets,
            possible_num_pivots=CFG.domino_train_num_pivots,
            rng=self._train_rng)

    def _generate_test_tasks(self) -> List[EnvironmentTask]:
        """Generate test tasks."""
        return self._make_tasks(
            num_tasks=CFG.num_test_tasks,
            possible_num_dominos=CFG.domino_test_num_dominos,
            possible_num_targets=CFG.domino_test_num_targets,
            possible_num_pivots=CFG.domino_test_num_pivots,
            rng=self._test_rng)

    def _make_tasks(self,
                    num_tasks: int,
                    possible_num_dominos: List[int],
                    possible_num_targets: List[int],
                    possible_num_pivots: List[int],
                    rng: np.random.Generator,
                    log_debug: bool = False) -> List[EnvironmentTask]:
        """Generate tasks with rich scenarios mixing dominoes and ball/fans.

        Task types:
        - Domino-only: Topple dominoes (ball present but not in goal)
        - Ball-only: Move ball to target (dominoes as obstacles)
        - Combined: BOTH topple dominoes AND move ball to target
        - Ball-helps-domino: Use ball to knock down dominoes
        """
        tasks = []

        for i_task in range(num_tasks):
            # Decide task type with multiple options
            task_types = [
                "domino_only", "ball_only", "combined", "ball_helps_domino"
            ]
            # Weighted selection - adjust these weights as needed
            # weights = [0.3, 0.3, 0.2, 0.2]
            weights = [0, 0, 0, 1]
            task_type = rng.choice(task_types, p=weights)

            if task_type == "domino_only":
                task = self._generate_domino_goal_task(
                    i_task,
                    possible_num_dominos,
                    possible_num_targets,
                    possible_num_pivots,
                    rng,
                    log_debug,
                    include_active_ball=False)
            elif task_type == "ball_only":
                task = self._generate_ball_goal_task(
                    i_task,
                    possible_num_dominos,
                    rng,
                    log_debug,
                    include_dominoes_as_obstacles=True)
            elif task_type == "combined":
                task = self._generate_combined_goal_task(
                    i_task, possible_num_dominos, possible_num_targets,
                    possible_num_pivots, rng, log_debug)
            else:  # ball_helps_domino
                task = self._generate_domino_goal_task(
                    i_task,
                    possible_num_dominos,
                    possible_num_targets,
                    possible_num_pivots,
                    rng,
                    log_debug,
                    include_active_ball=True)

            if task is not None:
                tasks.append(task)

        return self._add_pybullet_state_to_tasks(tasks)

    def _generate_domino_goal_task(
            self,
            task_idx: int,
            possible_num_dominos: List[int],
            possible_num_targets: List[int],
            possible_num_pivots: List[int],
            rng: np.random.Generator,
            log_debug: bool,
            include_active_ball: bool = False) -> Optional[EnvironmentTask]:
        """Generate task with domino toppling goal.

        Uses parent's domino sequence generation and adds fan/ball objects.

        Args:
            include_active_ball: If True, place ball where it can help topple dominoes
        """
        # Robot initial
        robot_dict = {
            "x": self.robot_init_x,
            "y": self.robot_init_y,
            "z": self.robot_init_z,
            "fingers": self.open_fingers,
            "tilt": self.robot_init_tilt,
            "wrist": self.robot_init_wrist,
        }

        init_dict = {self._robot: robot_dict}

        # Add direction objects to initial state
        for i, direction_obj in enumerate(
                self.directions):  # type: ignore[attr-defined]
            init_dict[direction_obj] = {"dir": float(i)}

        # Generate domino sequence using parent's logic
        n_dominos = rng.choice(possible_num_dominos)
        n_targets = rng.choice(possible_num_targets)
        n_pivots = rng.choice(possible_num_pivots)

        obj_dict = None
        max_attempts = 1000
        for attempt_num in range(max_attempts):
            if log_debug:
                print(f"\nAttempt {attempt_num} for task {task_idx}")
            obj_dict = self._generate_domino_sequence(  # type: ignore[attr-defined]
                rng,
                n_dominos,
                n_targets,
                n_pivots,
                log_debug=log_debug,
                task_idx=task_idx,
                domino_in_upper_half=include_active_ball)
            if obj_dict is not None:
                if log_debug:
                    print("Found satisfying domino sequence")
                break

        if obj_dict is None:
            return None

        # Move intermediate objects if needed
        if not CFG.domino_initialize_at_finished_state:
            obj_dict = self._move_intermediate_objects_to_unfinished_state(  # type: ignore[attr-defined]
                obj_dict)

        init_dict.update(obj_dict)

        # Add fan/switch/ball objects
        if include_active_ball:
            # Place ball near the start of domino sequence where it can knock them down
            self._add_fan_ball_objects_to_init_dict(init_dict,
                                                    rng,
                                                    all_off=True)
            # Get first domino position
            first_domino = self.dominos[0]  # type: ignore[attr-defined]
            if first_domino in obj_dict:
                # Place ball offset from first domino to the side
                ball_x = rng.uniform(self.x_lb, self.x_ub)
                ball_y = rng.uniform(self.y_lb + 0.1, self.y_ub)
                init_dict[self._ball] = {
                    "x": ball_x,
                    "y": ball_y,
                    "z": self.table_height + self.ball_height_offset
                }
        else:
            # Ball in corner, out of the way
            self._add_fan_ball_objects_to_init_dict(init_dict,
                                                    rng,
                                                    all_off=True)

        init_state = utils.create_state_from_dict(init_dict)

        # Goal: topple all targets (use parent's Toppled predicate)
        if CFG.domino_use_domino_blocks_as_target:
            # Find target dominoes (pink/red dominoes) and set them as goals
            goal_atoms = set()
            for domino_obj in init_state.get_objects(
                    self._domino_type):  # type: ignore[attr-defined]
                if self._TargetDomino_holds(
                        init_state,
                    [domino_obj]):  # type: ignore[attr-defined]
                    goal_atoms.add(
                        GroundAtom(self._Toppled,
                                   [domino_obj]))  # type: ignore[attr-defined]
        else:
            # Use regular domino targets
            goal_atoms = set()
            for target_obj in init_state.get_objects(
                    self._target_type):  # type: ignore[attr-defined]
                goal_atoms.add(GroundAtom(
                    self._Toppled, [target_obj]))  # type: ignore[attr-defined]

        return EnvironmentTask(init_state, goal_atoms)

    def _generate_ball_goal_task(
        self,
        task_idx: int,
        possible_num_dominos: List[int],
        rng: np.random.Generator,
        log_debug: bool = False,
        include_dominoes_as_obstacles: bool = False
    ) -> Optional[EnvironmentTask]:
        """Generate task with ball-at-target goal.

        Ball must reach target position by controlling fans via switches.

        Args:
            task_idx: Task index (currently unused, reserved for future use)
            possible_num_dominos: Possible numbers of dominoes to place as obstacles
            rng: Random number generator
            log_debug: Enable debug logging
            include_dominoes_as_obstacles: If True, place dominoes as obstacles
        """
        # Note: task_idx is reserved for future use (e.g., difficulty progression)
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

        # Add direction objects
        for i, direction_obj in enumerate(
                self.directions):  # type: ignore[attr-defined]
            init_dict[direction_obj] = {"dir": float(i)}

        # Add fan/switch/ball objects (all fans off initially)
        self._add_fan_ball_objects_to_init_dict(init_dict, rng, all_off=True)

        # Ball - random start position in workspace
        ball_x = rng.uniform(self.x_lb + 0.05, self.x_ub - 0.05)
        ball_y = rng.uniform(self.y_lb + 0.05, self.y_ub - 0.05)
        init_dict[self._ball] = {
            "x": ball_x,
            "y": ball_y,
            "z": self.table_height + self.ball_height_offset
        }

        # Ball target - random goal position (different from start)
        target_x = rng.uniform(self.x_lb + 0.05, self.x_ub - 0.05)
        target_y = rng.uniform(self.y_lb + 0.05, self.y_ub - 0.05)
        # Ensure target is far enough from start
        while np.sqrt((target_x - ball_x)**2 + (target_y - ball_y)**2) < 0.15:
            target_x = rng.uniform(self.x_lb + 0.05, self.x_ub - 0.05)
            target_y = rng.uniform(self.y_lb + 0.05, self.y_ub - 0.05)

        init_dict[self._ball_target] = {
            "x": target_x,
            "y": target_y,
            "z": self.table_height,
            "is_hit": 0.0
        }

        # Add dominoes as obstacles if requested
        if include_dominoes_as_obstacles:
            n_dominos = rng.choice(possible_num_dominos)
            # Place dominoes randomly between ball and target
            for i in range(min(n_dominos, 5)):  # Limit to 5 obstacle dominoes
                # Random position in workspace
                dom_x = rng.uniform(self.x_lb + 0.1, self.x_ub - 0.1)
                dom_y = rng.uniform(self.y_lb + 0.1, self.y_ub - 0.1)
                # Avoid placing too close to ball or target
                while (np.sqrt((dom_x - ball_x)**2 + (dom_y - ball_y)**2) < 0.1
                       or np.sqrt((dom_x - target_x)**2 +
                                  (dom_y - target_y)**2) < 0.1):
                    dom_x = rng.uniform(self.x_lb + 0.1, self.x_ub - 0.1)
                    dom_y = rng.uniform(self.y_lb + 0.1, self.y_ub - 0.1)

                domino_obj = self.dominos[i]  # type: ignore[attr-defined]
                init_dict[domino_obj] = {
                    "x": dom_x,
                    "y": dom_y,
                    "z": self.z_lb + self.domino_height / 2,
                    "yaw": rng.uniform(-np.pi, np.pi),
                    "roll": 0.0,
                    "r": self.domino_color[0],  # type: ignore[attr-defined]
                    "g": self.domino_color[1],  # type: ignore[attr-defined]
                    "b": self.domino_color[2],  # type: ignore[attr-defined]
                    "is_held": 0.0,
                }

        init_state = utils.create_state_from_dict(init_dict)
        goal = {
            GroundAtom(self._BallAtTarget, [self._ball, self._ball_target])
        }

        return EnvironmentTask(init_state, goal)

    def _generate_combined_goal_task(
            self,
            task_idx: int,
            possible_num_dominos: List[int],
            possible_num_targets: List[int],
            possible_num_pivots: List[int],
            rng: np.random.Generator,
            log_debug: bool = False) -> Optional[EnvironmentTask]:
        """Generate task with BOTH domino and ball goals.

        Must topple all dominoes AND move ball to target to succeed.
        """
        # Robot initial
        robot_dict = {
            "x": self.robot_init_x,
            "y": self.robot_init_y,
            "z": self.robot_init_z,
            "fingers": self.open_fingers,
            "tilt": self.robot_init_tilt,
            "wrist": self.robot_init_wrist,
        }

        init_dict = {self._robot: robot_dict}

        # Add direction objects
        for i, direction_obj in enumerate(
                self.directions):  # type: ignore[attr-defined]
            init_dict[direction_obj] = {"dir": float(i)}

        # Generate domino sequence using parent's logic
        n_dominos = rng.choice(possible_num_dominos)
        n_targets = rng.choice(possible_num_targets)
        n_pivots = rng.choice(possible_num_pivots)

        obj_dict = None
        max_attempts = 1000
        for attempt_num in range(max_attempts):
            if log_debug:
                print(f"\nAttempt {attempt_num} for combined task {task_idx}")
            obj_dict = self._generate_domino_sequence(
                rng,  # type: ignore[attr-defined]
                n_dominos,
                n_targets,
                n_pivots,
                log_debug=log_debug,
                task_idx=task_idx)
            if obj_dict is not None:
                if log_debug:
                    print("Found satisfying domino sequence")
                break

        if obj_dict is None:
            return None

        # Move intermediate objects if needed
        if not CFG.domino_initialize_at_finished_state:
            obj_dict = self._move_intermediate_objects_to_unfinished_state(  # type: ignore[attr-defined]
                obj_dict)

        init_dict.update(obj_dict)

        # Add fan/switch/ball objects
        self._add_fan_ball_objects_to_init_dict(init_dict, rng, all_off=True)

        # Place ball at random position (not in corner)
        ball_x = rng.uniform(self.x_lb + 0.05, self.x_ub - 0.05)
        ball_y = rng.uniform(self.y_lb + 0.05, self.y_ub - 0.05)
        init_dict[self._ball] = {
            "x": ball_x,
            "y": ball_y,
            "z": self.table_height + self.ball_height_offset
        }

        # Place ball target at random position
        target_x = rng.uniform(self.x_lb + 0.05, self.x_ub - 0.05)
        target_y = rng.uniform(self.y_lb + 0.05, self.y_ub - 0.05)
        # Ensure target is far from ball
        while np.sqrt((target_x - ball_x)**2 + (target_y - ball_y)**2) < 0.15:
            target_x = rng.uniform(self.x_lb + 0.05, self.x_ub - 0.05)
            target_y = rng.uniform(self.y_lb + 0.05, self.y_ub - 0.05)

        init_dict[self._ball_target] = {
            "x": target_x,
            "y": target_y,
            "z": self.table_height,
            "is_hit": 0.0
        }

        init_state = utils.create_state_from_dict(init_dict)

        # Goal: BOTH topple dominoes AND move ball to target
        goal_atoms = set()

        # Add ball goal
        goal_atoms.add(
            GroundAtom(self._BallAtTarget, [self._ball, self._ball_target]))

        # Add domino goals
        if CFG.domino_use_domino_blocks_as_target:
            # Find target dominoes (pink/red dominoes)
            for domino_obj in init_state.get_objects(
                    self._domino_type):  # type: ignore[attr-defined]
                if self._TargetDomino_holds(
                        init_state,
                    [domino_obj]):  # type: ignore[attr-defined]
                    goal_atoms.add(
                        GroundAtom(self._Toppled,
                                   [domino_obj]))  # type: ignore[attr-defined]
        else:
            # Use regular domino targets
            for target_obj in init_state.get_objects(
                    self._target_type):  # type: ignore[attr-defined]
                goal_atoms.add(GroundAtom(
                    self._Toppled, [target_obj]))  # type: ignore[attr-defined]

        return EnvironmentTask(init_state, goal_atoms)

    def _add_fan_ball_objects_to_init_dict(self,
                                           init_dict: Dict,
                                           rng: np.random.Generator,
                                           all_off: bool = True) -> None:
        """Add fan, switch, side, and ball objects to init_dict.

        Args:
            init_dict: Dictionary to add objects to
            rng: Random number generator
            all_off: If True, all fans/switches start off
        """
        # Fans
        for fan_obj in self._fans:
            side_idx = fan_obj.side_idx
            if side_idx == 0:  # left
                px, py = self.left_fan_x, (self.fan_y_lb + self.fan_y_ub) / 2
                rot = 0.0
            elif side_idx == 1:  # right
                px, py = self.right_fan_x, (self.fan_y_lb + self.fan_y_ub) / 2
                rot = np.pi
            elif side_idx == 2:  # back
                px, py = (self.fan_x_lb + self.fan_x_ub) / 2, self.down_fan_y
                rot = np.pi / 2
            else:  # front
                px, py = (self.fan_x_lb + self.fan_x_ub) / 2, self.up_fan_y
                rot = -np.pi / 2

            init_dict[fan_obj] = {
                "x": px,
                "y": py,
                "z": self.table_height + self.fan_z_len / 2,
                "rot": rot,
                "facing_side": float(side_idx),
                "is_on": 0.0 if all_off else float(rng.random() > 0.5)
            }

        # Switches
        for switch_obj in self._switches:
            init_dict[switch_obj] = {
                "x": self.switch_base_x +
                self.switch_x_spacing * switch_obj.side_idx,
                "y": self.switch_y,
                "z": self.table_height,
                "rot": np.pi / 2,
                "controls_fan": float(switch_obj.side_idx),
                "is_on": 0.0 if all_off else float(rng.random() > 0.5),
            }

        # Sides
        for i, side_obj in enumerate(self._sides):
            init_dict[side_obj] = {"side_idx": float(i)}

        # Ball (if not already added)
        if self._ball not in init_dict:
            # Place ball in corner to stay out of the way for domino tasks
            ball_x = self.x_lb + 0.05
            ball_y = self.y_lb + 0.05
            init_dict[self._ball] = {
                "x": ball_x,
                "y": ball_y,
                "z": self.table_height + self.ball_height_offset
            }

        # Ball target (if not already added)
        if self._ball_target not in init_dict:
            # Place target in opposite corner
            init_dict[self._ball_target] = {
                "x": self.x_ub - 0.05,
                "y": self.y_ub - 0.05,
                "z": self.table_height,
                "is_hit": 0.0
            }


if __name__ == "__main__":
    import time

    # Configure environment
    CFG.seed = 0
    CFG.env = "pybullet_domino_fan"
    CFG.num_train_tasks = 0
    CFG.num_test_tasks = 10  # Generate more tasks to see different types

    # Domino configuration
    CFG.domino_initialize_at_finished_state = True
    CFG.domino_use_domino_blocks_as_target = True
    CFG.domino_has_glued_dominos = False
    CFG.domino_test_num_dominos = [3, 4]
    CFG.domino_test_num_targets = [1]
    CFG.domino_test_num_pivots = [0]

    # Fan/ball configuration
    CFG.domino_fan_ball_task_ratio = 0.5  # 50% ball tasks, 50% domino tasks
    CFG.domino_fan_ball_position_tolerance = 0.04
    CFG.fan_known_controls_relation = True
    CFG.fan_fans_blow_opposite_direction = False

    # Create environment
    env = PyBulletDominoFanEnv(use_gui=True)

    # Generate test tasks
    tasks = env._generate_test_tasks()

    # Test each task
    for i, task in enumerate(tasks):
        print(f"\n{'=' * 60}")
        print(f"Task {i + 1}")
        print(f"{'=' * 60}")

        # Determine task type
        has_ball_goal = any(atom.predicate == env._BallAtTarget
                            for atom in task.goal)
        has_domino_goal = any(
            atom.predicate == env._Toppled  # type: ignore[attr-defined]
            for atom in task.goal)

        # Reset to initial state
        env._reset_state(task.init)

        print(f"\nGoal atoms:")
        for atom in task.goal:
            print(f"  {atom}")

        try:
            for step in range(1000000):
                # Use null action (stay in place)
                action = Action(
                    np.array(env._pybullet_robot.initial_joint_positions))
                state = env.step(action)

                # Check if goal is reached
                if all(atom.holds(state) for atom in task.goal):
                    time.sleep(2)
                    break

                time.sleep(0.05)
        except KeyboardInterrupt:
            continue
