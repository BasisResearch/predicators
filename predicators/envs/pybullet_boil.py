import logging
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pybullet as p

from predicators import utils
from predicators.envs.pybullet_env import PyBulletEnv, create_pybullet_block
from predicators.pybullet_helpers.geometry import Pose3D, Quaternion
from predicators.pybullet_helpers.objects import create_object, update_object
from predicators.pybullet_helpers.robots import SingleArmPyBulletRobot
from predicators.settings import CFG
from predicators.structs import Action, EnvironmentTask, GroundAtom, Object, \
    Predicate, State, Type


class PyBulletBoilEnv(PyBulletEnv):
    """A PyBullet environment that simulates boiling water in jugs using
    multiple burners and filling water from a faucet.

    - Jugs can be placed under a faucet to be filled with water (blue color).
    - Jugs can be placed on burners to heat water toward a red color.
    - Each burner and the faucet has a corresponding switch that can be toggled.
    - Spillage occurs if there is no jug under the faucet while the faucet is on.
    """

    # -------------------------------------------------------------------------
    # Table / workspace config
    # -------------------------------------------------------------------------
    table_height: ClassVar[float] = 0.4
    table_pos: ClassVar[Pose3D] = (0.75, 1.35, table_height / 2)
    table_orn: ClassVar[Quaternion] = p.getQuaternionFromEuler(
        [0.0, 0.0, np.pi / 2.0])

    x_lb: ClassVar[float] = 0.4
    x_ub: ClassVar[float] = 1.1
    y_lb: ClassVar[float] = 1.1
    y_ub: ClassVar[float] = 1.6
    z_lb: ClassVar[float] = table_height
    z_ub: ClassVar[float] = 0.75 + table_height / 2
    x_mid: ClassVar[float] = (x_lb + x_ub) / 2
    y_mid: ClassVar[float] = (y_lb + y_ub) / 2

    # -------------------------------------------------------------------------
    # Robot config
    # -------------------------------------------------------------------------
    robot_init_x: ClassVar[float] = (x_lb + x_ub) * 0.5
    robot_init_y: ClassVar[float] = (y_lb + y_ub) * 0.5
    robot_init_z: ClassVar[float] = z_ub - 0.1
    robot_base_pos: ClassVar[Pose3D] = (0.75, 0.65, 0.0)
    robot_base_orn: ClassVar[Quaternion] = p.getQuaternionFromEuler(
        [0.0, 0.0, np.pi / 2])
    robot_init_tilt: ClassVar[float] = np.pi / 2
    robot_init_wrist: ClassVar[float] = -np.pi / 2

    # -------------------------------------------------------------------------
    # Camera
    # -------------------------------------------------------------------------
    _camera_distance: ClassVar[float] = .8
    _camera_yaw: ClassVar[float] = 180
    _camera_pitch: ClassVar[float] = -45
    _camera_target: ClassVar[Tuple[float, float, float]] = (0.75, 1.25, 0.52)

    # -------------------------------------------------------------------------
    jug_height: ClassVar[float] = 0.12
    jug_handle_height: ClassVar[float] = jug_height * 3 / 4
    jug_handle_offset: ClassVar[float] = 0.08
    jug_init_z: ClassVar[float] = table_height + jug_height / 2
    small_gap: ClassVar[float] = 0.05
    burner_x_gap: ClassVar[float] = 3 * small_gap
    burner_y: ClassVar[float] = y_mid + small_gap
    faucet_x: ClassVar[float] = x_mid + 6 * small_gap
    faucet_y: ClassVar[float] = y_mid + 3 * small_gap
    faucet_x_len: ClassVar[float] = 0.15
    switch_y: ClassVar[float] = y_lb + small_gap

    # -------------------------------------------------------------------------
    # Domain-specific config
    # -------------------------------------------------------------------------
    num_jugs: ClassVar[int] = 1
    num_burners: ClassVar[int] = 1  # can be adjusted as needed

    # Speeds / rates
    water_fill_speed: ClassVar[
        float] = 0.002  # how fast water_level increases per step
    water_spill_factor: ClassVar[
        float] = 0.5  # fraction of water wasted if misaligned
    water_color = (0.0, 0.0, 1.0, 0.9)  # blue
    heating_speed: ClassVar[
        float] = 0.02  # how fast the jug's "heat_level" goes up per step

    # Dist thresholds
    faucet_align_threshold: ClassVar[
        float] = 0.1  # if jug is within this distance of faucet
    burner_align_threshold: ClassVar[float] = 0.05
    switch_joint_scale: ClassVar[float] = 0.1
    switch_on_threshold: ClassVar[float] = 0.5  # fraction of the joint range
    switch_height: ClassVar[float] = 0.08

    # We'll store a separate 'heat' feature for jugs in the environment
    # (0.0 => fully cold/blue, 1.0 => fully hot/red).
    # We'll produce a color in step() from that.
    # -------------------------------------------------------------------------
    # Types
    # -------------------------------------------------------------------------
    _robot_type = Type("robot", ["x", "y", "z", "fingers", "tilt", "wrist"])


    _jug_type = Type(
        "jug", ["x", "y", "z", "rot", "is_held", "water_level", "heat_level"],
        sim_features=["id", "heat_level", "water_id"])
    _burner_type = Type("burner", ["x", "y", "z", "is_on"],
                        sim_features=["id", "switch_id"])
    _switch_type = Type("switch", ["x", "y", "z", "rot", "is_on"])
    _faucet_type = Type("faucet",
                        ["x", "y", "z", "rot", "is_on", "spilled_level"],
                        sim_features=["id", "switch_id"])

    def __init__(self, use_gui: bool = True) -> None:
        # Create the robot as an Object
        self._robot = Object("robot", self._robot_type)

        # Create jugs
        self._jugs: List[Object] = []
        for i in range(self.num_jugs):
            jug_obj = Object(f"jug{i}", self._jug_type)
            self._jugs.append(jug_obj)
        self._jug_to_liquid_id: Dict[Object, Optional[int]] = {}

        # Create burners + a corresponding switch for each
        self._burners: List[Object] = []
        self._burner_switches: List[Object] = []
        for i in range(self.num_burners):
            burn_obj = Object(f"burner{i}", self._burner_type)
            self._burners.append(burn_obj)

            sw_obj = Object(f"burner_switch{i}", self._switch_type)
            self._burner_switches.append(sw_obj)

        # Create one faucet + a corresponding switch
        self._faucet = Object("faucet", self._faucet_type)
        self._faucet_switch = Object("faucet_switch", self._switch_type)

        # Keep track of the spilled water block (None if no spill yet)
        self._spilled_water_id: Optional[int] = None

        super().__init__(use_gui)

        # Optionally, define some relevant predicates
        self._JugFilled = Predicate("JugFilled", [self._jug_type],
                                    self._JugFilled_holds)
        self._WaterBoiled = Predicate("WaterBoiled", [self._jug_type],
                                      self._WaterBoiled_holds)
        self._BurnerOn = Predicate("BurnerOn", [self._burner_type],
                                   self._BurnerOn_holds)
        self._FaucetOn = Predicate("FaucetOn", [self._faucet_type],
                                   self._FaucetOn_holds)
        self._BurnerOff = Predicate(
            "BurnerOff", [self._burner_type],
            lambda s, o: not self._BurnerOn_holds(s, o))
        self._FaucetOff = Predicate(
            "FaucetOff", [self._faucet_type],
            lambda s, o: not self._FaucetOn_holds(s, o))
        self._Holding = Predicate("Holding",
                                  [self._robot_type, self._jug_type],
                                  self._Holding_holds)
        self._JugOnBurner = Predicate("JugOnBurner",
                                      [self._jug_type, self._burner_type],
                                      self._JugOnBurner_holds)
        self._JugUnderFaucet = Predicate("JugUnderFaucet",
                                         [self._jug_type, self._faucet_type],
                                         self._JugUnderFaucet_holds)
        self._NoJugUnderFaucet = Predicate("NoJugUnderFaucet",
                                           [self._faucet_type],
                                           self._NoJugUnderFaucet_holds)
        self._HandEmpty = Predicate("HandEmpty", [self._robot_type],
                                    self._HandEmpty_holds)
        self._WaterSpilled = Predicate("WaterSpilled", [],
                                       self._WaterSpilled_holds)
        self._NoWaterSpilled = Predicate("NoWaterSpilled", [],
                                         self._NoWaterSpilled_holds)

    @classmethod
    def get_name(cls) -> str:
        return "pybullet_boil"

    @property
    def predicates(self) -> Set[Predicate]:
        """Return a set of domain-specific predicates that might be used for
        planning."""
        return {
            self._JugFilled, self._WaterBoiled, self._BurnerOn, self._FaucetOn,
            self._BurnerOff, self._FaucetOff, self._Holding, self._JugOnBurner,
            self._JugUnderFaucet, self._HandEmpty, self._WaterSpilled,
            self._NoJugUnderFaucet, self._NoWaterSpilled
        }

    @property
    def types(self) -> Set[Type]:
        """All custom types in this environment."""
        return {
            self._robot_type, self._jug_type, self._burner_type,
            self._switch_type, self._faucet_type
        }

    @property
    def goal_predicates(self) -> Set[Predicate]:
        """Which predicates might appear in goals."""
        return {self._WaterBoiled, self._JugFilled,
                self._NoWaterSpilled}  # Example

    # -------------------------------------------------------------------------
    # PyBullet Initialization
    # -------------------------------------------------------------------------
    @classmethod
    def initialize_pybullet(
            cls, using_gui: bool
    ) -> Tuple[int, SingleArmPyBulletRobot, Dict[str, Any]]:
        physics_client_id, pybullet_robot, bodies = super(
        ).initialize_pybullet(using_gui)

        # 1) Create a table
        table_id = create_object(
            asset_path="urdf/table.urdf",
            position=cls.table_pos,
            orientation=cls.table_orn,
            scale=1.0,
            use_fixed_base=True,
            physics_client_id=physics_client_id,
        )
        bodies["table_id"] = table_id

        # 2) Create jugs
        jug_ids = []
        for _ in range(cls.num_jugs):
            # Example placeholder URDF for a jug
            jug_id = create_object(asset_path="urdf/jug-pixel.urdf",
                                   use_fixed_base=False,
                                   physics_client_id=physics_client_id)
            jug_ids.append(jug_id)
        bodies["jug_ids"] = jug_ids

        # 3) Create burners
        burner_ids = []
        for _ in range(cls.num_burners):
            burner_id = create_pybullet_block(
                color=(.7, .7, .7, 1),
                half_extents=(0.07, 0.07, 0.001),
                mass=0,
                friction=0.5,
                physics_client_id=physics_client_id)
            burner_ids.append(burner_id)
        bodies["burner_ids"] = burner_ids

        # 4) Create burner switches
        burner_switch_ids = []
        for _ in range(cls.num_burners):
            switch_id = create_object(
                asset_path="urdf/partnet_mobility/switch/102812/switch.urdf",
                scale=1.0,
                use_fixed_base=True,
                physics_client_id=physics_client_id)
            burner_switch_ids.append(switch_id)
        bodies["burner_switch_ids"] = burner_switch_ids

        # 5) Create faucet and faucet switch
        faucet_id = create_object(
            asset_path="urdf/partnet_mobility/faucet/1488/mobility.urdf",
            use_fixed_base=True,
            physics_client_id=physics_client_id)
        bodies["faucet_id"] = faucet_id

        faucet_switch_id = create_object(
            asset_path="urdf/partnet_mobility/switch/102812/switch.urdf",
            scale=1.0,
            use_fixed_base=True,
            physics_client_id=physics_client_id)
        bodies["faucet_switch_id"] = faucet_switch_id

        return physics_client_id, pybullet_robot, bodies

    def _store_pybullet_bodies(self, pybullet_bodies: Dict[str, Any]) -> None:
        """Store references to all PyBullet IDs in the environment objects."""
        # Jugs
        for i, jug_obj in enumerate(self._jugs):
            jug_obj.id = pybullet_bodies["jug_ids"][i]

        # Burners
        for i, burner_obj in enumerate(self._burners):
            burner_obj.id = pybullet_bodies["burner_ids"][i]

        # Burner switches
        for i, sw_obj in enumerate(self._burner_switches):
            sw_obj.id = pybullet_bodies["burner_switch_ids"][i]

        # Faucet
        self._faucet.id = pybullet_bodies["faucet_id"]
        # Faucet switch
        self._faucet_switch.id = pybullet_bodies["faucet_switch_id"]

    # -------------------------------------------------------------------------
    # State Creation / Feature Extraction
    # -------------------------------------------------------------------------
    def _get_object_ids_for_held_check(self) -> List[int]:
        """Only jugs can be held in the robot's gripper here."""
        jug_ids = [j.id for j in self._jugs if j.id is not None]
        return jug_ids

    def _create_task_specific_objects(self, state: State) -> None:
        """If you wanted additional objects depending on a given state, add
        them here."""
        pass

    def _extract_feature(self, obj: Object, feature: str) -> float:
        """Map from environment object + feature name -> a float feature in the
        State."""
        # Faucet
        if obj.type == self._faucet_type:
            if feature == "is_on":
                return float(self._is_switch_on(self._faucet_switch.id))
            if feature == "spilled_level":
                # Return the environment's internal record (analogous to jug.heat_level).
                # We'll just store it in the object itself (similar to jug.heat_level).
                # If it doesn't exist, default to 0.
                if self._spilled_water_id is None:
                    return 0.0
                shape_data = p.getVisualShapeData(
                    self._spilled_water_id, physicsClientId=self._physics_client_id
                )
                if not shape_data:
                    return 0.0
                # shape_data[0][3] is a tuple of the half-extents (x, y, z).
                # Since it's a square "sheet," just take x*2 as the side length:
                half_extents = shape_data[0][3]  # (hx, hy, hz)
                side_len = half_extents[0] * 2.0
                return side_len

        # Burner
        elif obj.type == self._burner_type:
            if feature == "is_on":
                idx = int(obj.name.replace("burner", ""))
                sw_obj = self._burner_switches[idx]
                return float(self._is_switch_on(sw_obj.id))

        # Switch
        elif obj.type == self._switch_type:
            if feature == "is_on":
                return float(self._is_switch_on(obj.id))

        # Jug
        elif obj.type == self._jug_type:
            if feature == "water_level":
                liquid_id = self._jug_to_liquid_id.get(obj, None)
                if liquid_id is not None:
                    shape_data = p.getVisualShapeData(
                        liquid_id, physicsClientId=self._physics_client_id)
                    if shape_data:  # handle the case shape_data might be empty
                        # shape_data[0][3] => half-extents, e.g. shape_data[0][3][2] is half in z
                        height = shape_data[0][3][2] * 2
                        return height
                return 0.0
            if feature == "heat_level":
                return getattr(obj, "heat_level", 0.0)

        # Otherwise, rely on defaults (like the base PyBulletEnv) for x,y,z,...
        raise ValueError(f"Unknown feature {feature} for object {obj}.")

    def _reset_custom_env_state(self, state: State) -> None:
        """Called in _reset_state to do any environment-specific resetting."""
        # Programmatically set burner switches on/off
        for i, burner_obj in enumerate(self._burners):
            on_val = state.get(burner_obj, "is_on")
            burner_obj.switch_id = self._burner_switches[i].id
            self._set_switch_on(self._burner_switches[i].id,
                                bool(on_val > 0.5))

        # Remove existing jug liquid bodies if they exist
        for liquid_id in self._jug_to_liquid_id.values():
            if liquid_id is not None:
                p.removeBody(liquid_id,
                             physicsClientId=self._physics_client_id)
        self._jug_to_liquid_id.clear()

        # Recreate the liquid bodies as needed
        for jug in self._jugs:
            jug.heat_level = state.get(jug, "heat_level")
            liquid_id = self._create_liquid_for_jug(jug, state)
            self._jug_to_liquid_id[jug] = liquid_id

        # Faucet on/off
        self._faucet.switch_id = self._faucet_switch.id
        f_on = state.get(self._faucet, "is_on")
        self._set_switch_on(self._faucet_switch.id, bool(f_on > 0.5))

        # Spilled water reset: remove old block if any
        if self._spilled_water_id is not None:
            p.removeBody(self._spilled_water_id,
                         physicsClientId=self._physics_client_id)
            self._spilled_water_id = None

        # The faucet has a spilled_level feature as well
        self._faucet.spilled_level = state.get(self._faucet, "spilled_level")
        # If there's already some spillage in the state, recreate a block
        if self._faucet.spilled_level > 0.0:
            self._spilled_water_id = self._create_spilled_water_block(
                self._faucet.spilled_level, state)

    # -------------------------------------------------------------------------
    # Step Logic
    # -------------------------------------------------------------------------
    def step(self, action: Action, render_obs: bool = False) -> State:
        """Execute a low-level action (robot controls), then handle water
        filling/spillage and heating."""
        # First let the base environment perform the usual PyBullet step
        next_state = super().step(action, render_obs=render_obs)

        # 1) Handle faucet filling/spillage
        self._handle_faucet_logic(next_state)

        # 2) Handle burner heating
        self._handle_heating_logic(next_state)

        # 3) Update jug colors based on their 'heat'
        self._update_jug_colors(next_state)

        # Re-read final state
        final_state = self._get_state()
        self._current_observation = final_state
        return final_state

    def _handle_faucet_logic(self, state: State) -> None:
        """If faucet is on, fill any jug that is properly aligned; otherwise,
        grow the spill block on the table. Additionally, if a jug is already
        full (water_level >= 1.0) but stays under the faucet, water spills."""
        faucet_on = self._is_switch_on(self._faucet_switch.id)
        if not faucet_on:
            return

        # Check if at least one jug is under the faucet
        jugs_under = [
            jug for jug in self._jugs
            if self._JugUnderFaucet_holds(state, [jug, self._faucet])
        ]

        # --------------------------------------------------------------------------
        # NO JUG UNDER FAUCET => SPILL
        # --------------------------------------------------------------------------
        if len(jugs_under) == 0:
            old_spill = state.get(self._faucet, "spilled_level")
            new_spill = min(1.0, old_spill + self.water_fill_speed)
            state.set(self._faucet, "spilled_level", new_spill)

            # Remove any existing spillage block
            if self._spilled_water_id is not None:
                p.removeBody(self._spilled_water_id,
                            physicsClientId=self._physics_client_id)
            # Recreate spill with updated size
            self._spilled_water_id = self._create_spilled_water_block(new_spill, state)

        # --------------------------------------------------------------------------
        # THERE IS AT LEAST ONE JUG UNDER THE FAUCET
        # --------------------------------------------------------------------------
        else:
            for jug_obj in jugs_under:
                old_level = state.get(jug_obj, "water_level")

                # ------------------------------------------------------------------
                # If jug is NOT yet full
                # ------------------------------------------------------------------
                if old_level < 1.0:
                    # Fill up to capacity
                    new_level = old_level + self.water_fill_speed
                    if new_level > 1.0:
                        new_level = 1.0  # cap it at 1.0
                    state.set(jug_obj, "water_level", new_level)

                    # Recreate jug's liquid block at new water level
                    old_liquid_id = self._jug_to_liquid_id[jug_obj]
                    if old_liquid_id is not None:
                        p.removeBody(old_liquid_id, physicsClientId=self._physics_client_id)
                    self._jug_to_liquid_id[jug_obj] = self._create_liquid_for_jug(jug_obj, state)

                # ------------------------------------------------------------------
                # If jug is ALREADY FULL => overflow spills
                # ------------------------------------------------------------------
                else:
                    old_spill = state.get(self._faucet, "spilled_level")
                    new_spill = min(1.0, old_spill + self.water_fill_speed)
                    state.set(self._faucet, "spilled_level", new_spill)

                    # Remove any existing spill block
                    if self._spilled_water_id is not None:
                        p.removeBody(self._spilled_water_id,
                                    physicsClientId=self._physics_client_id)
                    # Recreate spill block with updated size
                    self._spilled_water_id = self._create_spilled_water_block(new_spill, state)

    def _handle_heating_logic(self, state: State) -> None:
        """If a jug with water is on a turned-on burner, increment jug 'heat'
        up to 1.0."""
        for i, burner_obj in enumerate(self._burners):
            burner_on = self._is_switch_on(self._burner_switches[i].id)
            if not burner_on:
                continue
            bx = state.get(burner_obj, "x")
            by = state.get(burner_obj, "y")
            for jug_obj in self._jugs:
                jug_x = state.get(jug_obj, "x")
                jug_y = state.get(jug_obj, "y")
                dist = np.hypot(bx - jug_x, by - jug_y)
                if dist < self.burner_align_threshold:
                    # Jug is on top of an active burner => increase heat
                    old_heat = state.get(jug_obj, "heat_level")
                    if state.get(jug_obj, "water_level") > 0.0:
                        new_heat = min(1.0, old_heat + self.heating_speed)
                        jug_obj.heat_level = new_heat

    def _update_jug_colors(self, state: State) -> None:
        """Simple linear interpolation from blue (0.0) to red (1.0) based on
        jug.heat."""
        for jug_obj in self._jugs:
            jug_id = jug_obj.id
            water_id = self._jug_to_liquid_id[jug_obj]
            if jug_id is None or water_id is None:
                continue
            heat = jug_obj.heat_level
            # Weighted interpolation from (0,0,1) => (1,0,0)
            r = heat
            g = 0.0
            b = 1.0 - heat
            alpha = 0.9
            update_object(water_id,
                          color=(r, g, b, alpha),
                          physics_client_id=self._physics_client_id)

    def _create_spilled_water_block(self, spilled_size: float,
                                    state: State) -> int:
        """Create a very short block on the table to represent spilled water.

        The side length is 'spilled_size'.
        """
        faucet_x = state.get(self._faucet, "x")
        faucet_y = state.get(self._faucet, "y")
        faucet_rot = state.get(self._faucet, "rot")
        # Center the spill where the faucet output is
        output_distance = self.faucet_x_len
        output_x = faucet_x + output_distance * np.cos(faucet_rot)
        output_y = faucet_y - output_distance * np.sin(faucet_rot)

        half_len = spilled_size / 2.0
        # Keep it very thin in Z
        half_extents = (half_len, half_len, 0.001)

        block_id = create_pybullet_block(
            color=(0.0, 0.0, 1.0, 0.5),
            half_extents=half_extents,
            mass=0,
            friction=0.5,
            position=(output_x, output_y, self.table_height),
            physics_client_id=self._physics_client_id)
        return block_id

    # -------------------------------------------------------------------------
    # Switch Helpers
    # -------------------------------------------------------------------------
    def _is_switch_on(self, switch_id: int) -> bool:
        """Check if a switch's main joint is above a threshold."""
        if switch_id < 0:
            return False
        j_id = self._get_joint_id(switch_id, "joint_0")
        if j_id < 0:
            return False
        j_pos, _, _, _ = p.getJointState(
            switch_id, j_id, physicsClientId=self._physics_client_id)
        info = p.getJointInfo(switch_id,
                              j_id,
                              physicsClientId=self._physics_client_id)
        j_min, j_max = info[8], info[9]
        frac = (j_pos / self.switch_joint_scale - j_min) / (j_max - j_min)
        return bool(frac > self.switch_on_threshold)

    def _set_switch_on(self, switch_id: int, power_on: bool) -> None:
        """Programmatically toggle the switch to on/off by resetting its joint
        state."""
        j_id = self._get_joint_id(switch_id, "joint_0")
        if j_id < 0:
            return
        info = p.getJointInfo(switch_id,
                              j_id,
                              physicsClientId=self._physics_client_id)
        j_min, j_max = info[8], info[9]
        target_val = j_max if power_on else j_min
        p.resetJointState(switch_id,
                          j_id,
                          target_val,
                          physicsClientId=self._physics_client_id)

    @staticmethod
    def _get_joint_id(obj_id: int, joint_name: str) -> int:
        """Helper to find a joint by name in a URDF."""
        num_joints = p.getNumJoints(obj_id)
        for j in range(num_joints):
            info = p.getJointInfo(obj_id, j)
            if info[1].decode("utf-8") == joint_name:
                return j
        return -1

    # -------------------------------------------------------------------------
    # Example Predicates
    # -------------------------------------------------------------------------
    @staticmethod
    def _JugFilled_holds(state: State, objects: Sequence[Object]) -> bool:
        (jug, ) = objects
        return state.get(jug, "water_level") >= 0.08

    def _WaterSpilled_holds(self, state: State,
                            objects: Sequence[Object]) -> bool:
        """Example: say water is spilled if the faucet is on with no jug
        underneath, or if we want any other condition. Modify as needed."""
        # If faucet is on and no jug is under it, there's spillage
        faucet_on = self._FaucetOn_holds(state, [self._faucet])
        no_jug_under = self._NoJugUnderFaucet_holds(state, [self._faucet])
        if faucet_on and no_jug_under:
            return True
        # You could also define your own logic for 'any visible puddle' => spilled
        # For instance, check the faucet's spilled_level > 0
        if state.get(self._faucet, "spilled_level") > 0.01:
            return True
        return False

    def _NoWaterSpilled_holds(self, state: State,
                              objects: Sequence[Object]) -> bool:
        return not self._WaterSpilled_holds(state, objects)

    @staticmethod
    def _WaterBoiled_holds(state: State, objects: Sequence[Object]) -> bool:
        (jug, ) = objects
        return state.get(jug, "heat_level") >= 1.0

    @staticmethod
    def _BurnerOn_holds(state: State, objects: Sequence[Object]) -> bool:
        (burner, ) = objects
        return state.get(burner, "is_on") > 0.5

    @staticmethod
    def _FaucetOn_holds(state: State, objects: Sequence[Object]) -> bool:
        (faucet, ) = objects
        return state.get(faucet, "is_on") > 0.5

    @staticmethod
    def _Holding_holds(state: State, objects: Sequence[Object]) -> bool:
        (robot, jug) = objects
        return state.get(jug, "is_held") > 0.5

    def _JugOnBurner_holds(self, state: State,
                           objects: Sequence[Object]) -> bool:
        (jug, burner) = objects
        jug_x = state.get(jug, "x")
        jug_y = state.get(jug, "y")
        burner_x = state.get(burner, "x")
        burner_y = state.get(burner, "y")
        dist = np.hypot(jug_x - burner_x, jug_y - burner_y)
        return dist < self.burner_align_threshold

    def _JugUnderFaucet_holds(self, state: State,
                              objects: Sequence[Object]) -> bool:
        (jug, faucet) = objects
        jug_x = state.get(jug, "x")
        jug_y = state.get(jug, "y")
        faucet_x = state.get(faucet, "x")
        faucet_y = state.get(faucet, "y")
        faucet_rot = state.get(faucet, "rot")
        output_distance = self.faucet_x_len
        output_x = faucet_x + output_distance * np.cos(faucet_rot)
        output_y = faucet_y - output_distance * np.sin(faucet_rot)
        dist = np.hypot(jug_x - output_x, jug_y - output_y)
        return dist < self.faucet_align_threshold

    def _NoJugUnderFaucet_holds(self, state: State,
                                objects: Sequence[Object]) -> bool:
        (faucet, ) = objects
        jugs = state.get_objects(self._jug_type)
        for jug in jugs:
            if self._JugUnderFaucet_holds(state, [jug, faucet]):
                return False
        return True

    def _HandEmpty_holds(self, state: State,
                         objects: Sequence[Object]) -> bool:
        (robot, ) = objects
        jugs = state.get_objects(self._jug_type)
        for jug in jugs:
            if self._Holding_holds(state, [robot, jug]):
                return False
        return True

    # -------------------------------------------------------------------------
    # Task Generation
    # -------------------------------------------------------------------------
    def _generate_train_tasks(self) -> List[EnvironmentTask]:
        return self._make_tasks(num_tasks=CFG.num_train_tasks,
                                rng=self._train_rng)

    def _generate_test_tasks(self) -> List[EnvironmentTask]:
        return self._make_tasks(num_tasks=CFG.num_test_tasks,
                                rng=self._test_rng)

    def _make_tasks(self, num_tasks: int,
                    rng: np.random.Generator) -> List[EnvironmentTask]:
        """Randomly place jugs, burners, faucet, etc.

        for each task.
        """
        tasks = []
        for _ in range(num_tasks):
            init_dict = {}

            # Robot
            robot_dict = {
                "x": self.robot_init_x,
                "y": self.robot_init_y,
                "z": self.robot_init_z,
                "fingers": self.open_fingers,
                "tilt": self.robot_init_tilt,
                "wrist": self.robot_init_wrist
            }
            init_dict[self._robot] = robot_dict

            # For random placements
            used_xy = set()

            # Jugs
            for i, j_obj in enumerate(self._jugs):
                x, y = self._sample_xy(rng, used_xy)
                init_dict[j_obj] = {
                    "x": x,
                    "y": y,
                    "z": self.jug_init_z,
                    "rot": -np.pi / 2,
                    "is_held": 0.0,
                    "water_level": 0.0,
                    "heat_level": 0.0
                }

            # Burners
            for i, b_obj in enumerate(self._burners):
                burner_x = self.x_mid - (i + 0.5) * self.small_gap * 3
                init_dict[b_obj] = {
                    "x": burner_x,
                    "y": self.burner_y,
                    "z": self.table_height,
                    "is_on": 0.0
                }
                # Switch for burner
                sw_obj = self._burner_switches[i]
                init_dict[sw_obj] = {
                    "x": burner_x,
                    "y": self.switch_y,
                    "z": self.table_height,
                    "rot": 0.0,
                    "is_on": 0.0
                }

            # Faucet
            init_dict[self._faucet] = {
                "x": self.faucet_x,
                "y": self.faucet_y,
                "z": self.table_height + 0.2,
                "rot": np.pi / 2,
                "is_on": 0.0,
                "spilled_level": 0.0  # Initialize spillage to 0
            }
            # Faucet switch
            init_dict[self._faucet_switch] = {
                "x": self.faucet_x,
                "y": self.switch_y,
                "z": self.table_height,
                "rot": 0.0,
                "is_on": 0.0
            }

            init_state = utils.create_state_from_dict(init_dict)

            # Example goal: Water boiled, no water spilled, etc.
            goal_atoms = set()
            for j_obj in self._jugs:
                goal_atoms.add(GroundAtom(self._WaterBoiled, [j_obj]))
                goal_atoms.add(GroundAtom(self._NoWaterSpilled, []))
                goal_atoms.add(GroundAtom(self._BurnerOff, [self._burners[0]]))

            tasks.append(EnvironmentTask(init_state, goal_atoms))

        return self._add_pybullet_state_to_tasks(tasks)

    def _sample_xy(self, rng: np.random.Generator,
                   used_xy: Set[Tuple[float, float]]) -> Tuple[float, float]:
        """Sample a random (x,y) on the table that doesn't collide with
        existing objects."""
        for _ in range(1000):
            x = rng.uniform(self.x_lb + 0.05, self.x_ub - 0.05)
            y = rng.uniform(self.y_lb + 0.05, self.y_ub - 0.05)
            if all((np.hypot(x - ux, y - uy) > 0.10) for (ux, uy) in used_xy):
                used_xy.add((x, y))
                return x, y
        raise RuntimeError("Failed to sample a collision-free (x, y).")

    def _create_liquid_for_jug(
        self,
        jug: Object,
        state: State,
    ) -> Optional[int]:
        """Given the jug's water_level, create (or None) a small PyBullet body
        to represent the liquid."""
        current_liquid = state.get(jug, "water_level")
        if current_liquid <= 0:
            return None

        # Make a box that sits inside the jug
        liquid_height = current_liquid
        half_extents = [0.03, 0.03, liquid_height / 2]
        cx = state.get(jug, "x")
        cy = state.get(jug, "y")
        cz = self.z_lb + liquid_height / 2 + 0.02  # sits on table

        color = self.water_color
        return create_pybullet_block(color=color,
                                     half_extents=half_extents,
                                     mass=0.01,
                                     friction=0.5,
                                     position=(cx, cy, cz),
                                     physics_client_id=self._physics_client_id)


if __name__ == "__main__":
    import time
    CFG.seed = 0
    CFG.env = "pybullet_boil"
    CFG.coffee_use_pixelated_jug = True
    CFG.pybullet_sim_steps_per_action = 15
    # CFG.fan_fans_blow_opposite_direction = True
    env = PyBulletBoilEnv(use_gui=True)
    rng = np.random.default_rng(CFG.seed)
    tasks = env._make_tasks(1, rng)

    # manually defined policy
    from predicators.ground_truth_models import get_gt_options

    # env_options = list(PyBulletBoilGroundTruthOptionFactory.get_options(
    #     "pybullet_boil",
    #     env.types,
    #     env.predicates,
    #     env.action_space))
    env_options = get_gt_options(env.get_name())
    pick = utils.get_parameterized_option_by_name(env_options, "PickJug")
    place_on_burner = utils.get_parameterized_option_by_name(
        env_options, "PlaceOnBurner")
    place_under_faucet = utils.get_parameterized_option_by_name(
        env_options, "PlaceUnderFaucet")
    switch_on = utils.get_parameterized_option_by_name(env_options, "SwitchOn")
    switch_off = utils.get_parameterized_option_by_name(
        env_options, "SwitchOff")
    no_op = utils.get_parameterized_option_by_name(env_options, "NoOp")
    # # Objects
    # robot = env._robot
    # jug1= env._jugs[0]
    # jug2= env._jugs[1]
    # burner_switch1 = env._burner_switches[0]
    # burner_switch2 = env._burner_switches[1]
    # faucet_switch = env._faucet_switch
    # burner1 = env._burners[0]
    # burner2 = env._burners[1]
    # faucet = env._faucet

    # env_predicates = env.predicates
    # policy = utils.option_plan_to_policy([
    #                         pick.ground([robot, jug2], []),
    #                         place_under_faucet.ground([robot, faucet], []),
    #                         switch_on.ground([robot, faucet_switch], []),
    #                         no_op.ground([robot], []),
    #                         switch_off.ground([robot, faucet_switch], []),
    #                         pick.ground([robot, jug2], []),
    #                         place_on_burner.ground([robot, burner2], []),
    #                         switch_on.ground([robot, burner_switch2], []),
    #                         pick.ground([robot, jug1], []),
    #                         place_under_faucet.ground([robot, faucet], []),
    #                         switch_on.ground([robot, faucet_switch], []),
    #                         no_op.ground([robot], []),
    #                         switch_off.ground([robot, faucet_switch], []),
    #                         pick.ground([robot, jug1], []),
    #                         place_on_burner.ground([robot, burner1], []),
    #                         switch_on.ground([robot, burner_switch1], []),
    #                         no_op.ground([robot], []),
    #                         switch_off.ground([robot, burner_switch2], []),
    #                         no_op.ground([robot], []),
    #                         switch_off.ground([robot, burner_switch1], []),
    #                         ],
    #                         noop_option_terminate_on_atom_change=True,
    #                         abstract_function=lambda s: utils.abstract(s,
    #                                                         env_predicates))

    constant_noop = True
    for task in tasks:
        env._reset_state(task.init)
        breakpoint()
        for _ in range(100000):
            if constant_noop:
                action = Action(
                    np.array(env._pybullet_robot.initial_joint_positions))
            else:
                try:
                    action = policy(env._current_observation)
                except:
                    # Get it's current position
                    action = Action(
                        np.array(env._current_observation.joint_positions))
            env.step(action)
            time.sleep(0.01)
