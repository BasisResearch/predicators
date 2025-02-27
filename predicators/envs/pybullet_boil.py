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

# ------------------------------------------------------------------------------
# Example PyBulletBoilEnv
# ------------------------------------------------------------------------------
class PyBulletBoilEnv(PyBulletEnv):
    """A PyBullet environment that simulates boiling water in jugs using 
    multiple burners and filling water from a faucet.

    - Jugs can be placed under a faucet to be filled with water (blue color).
    - Jugs can be placed on burners to heat water toward a red color.
    - Each burner and the faucet has a corresponding switch that can be toggled.
    - Spillage occurs if a jug is incorrectly aligned under the faucet.
    """

    # -------------------------------------------------------------------------
    # Table / workspace config
    # -------------------------------------------------------------------------
    table_height: ClassVar[float] = 0.4
    table_pos: ClassVar[Pose3D] = (0.75, 1.35, table_height / 2)
    table_orn: ClassVar[Quaternion] = p.getQuaternionFromEuler(
        [0.0, 0.0, np.pi/2.0]
    )

    x_lb: ClassVar[float] = 0.4
    x_ub: ClassVar[float] = 1.1
    y_lb: ClassVar[float] = 1.1
    y_ub: ClassVar[float] = 1.6
    z_lb: ClassVar[float] = table_height
    z_ub: ClassVar[float] = 0.75 + table_height/2

    # -------------------------------------------------------------------------
    # Robot config
    # -------------------------------------------------------------------------
    robot_init_x: ClassVar[float] = (x_lb + x_ub) * 0.5
    robot_init_y: ClassVar[float] = (y_lb + y_ub) * 0.5
    robot_init_z: ClassVar[float] = z_ub - 0.1
    robot_base_pos: ClassVar[Pose3D] = (0.75, 0.65, 0.0)
    robot_base_orn: ClassVar[Quaternion] = p.getQuaternionFromEuler([0.0, 0.0, np.pi/2])
    robot_init_tilt: ClassVar[float] = np.pi / 2
    robot_init_wrist: ClassVar[float] = -np.pi / 2

    # -------------------------------------------------------------------------
    # Camera
    # -------------------------------------------------------------------------
    _camera_distance: ClassVar[float] = 1.3
    _camera_yaw: ClassVar[float] = 70
    _camera_pitch: ClassVar[float] = -50
    _camera_target: ClassVar[Tuple[float, float, float]] = (0.75, 1.25, 0.42)

    # -------------------------------------------------------------------------
    # Domain-specific config
    # -------------------------------------------------------------------------
    num_jugs: ClassVar[int] = 2
    num_burners: ClassVar[int] = 2  # can be adjusted as needed

    # Speeds / rates
    water_fill_speed: ClassVar[float] = 0.02   # how fast water_level increases per step
    water_spill_factor: ClassVar[float] = 0.5  # fraction of water wasted if misaligned
    heating_speed: ClassVar[float] = 0.01      # how fast the jug's "heat" goes up per step

    # Dist thresholds
    faucet_align_threshold: ClassVar[float] = 0.1  # if jug is within this distance of faucet
    burner_align_threshold: ClassVar[float] = 0.08

    # We'll store a separate 'heat' feature for jugs in the environment
    # (0.0 => fully cold/blue, 1.0 => fully hot/red).
    # We'll produce a color in step() from that.
    # -------------------------------------------------------------------------
    # Types
    # -------------------------------------------------------------------------
    _robot_type = Type("robot", ["x", "y", "z", "fingers", "tilt", "wrist"])
    _jug_type = Type("jug", ["x", "y", "z", "rot", "is_held", "water_level", "heat"])
    _burner_type = Type("burner", ["x", "y", "z", "is_on"])
    _switch_type = Type("switch", ["x", "y", "z", "rot", "is_on"])
    _faucet_type = Type("faucet", ["x", "y", "z", "is_on"])

    def __init__(self, use_gui: bool = True) -> None:
        # Create the robot as an Object
        self._robot = Object("robot", self._robot_type)

        # Create jugs
        self._jugs: List[Object] = []
        for i in range(self.num_jugs):
            jug_obj = Object(f"jug{i}", self._jug_type)
            self._jugs.append(jug_obj)

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

        super().__init__(use_gui)

        # Optionally, define some relevant predicates
        self._JugFilled = Predicate("JugFilled", [self._jug_type], self._JugFilled_holds)
        self._JugHot = Predicate("JugHot", [self._jug_type], self._JugHot_holds)
        self._BurnerOn = Predicate("BurnerOn", [self._burner_type], self._BurnerOn_holds)
        self._FaucetOn = Predicate("FaucetOn", [self._faucet_type], self._FaucetOn_holds)

    @classmethod
    def get_name(cls) -> str:
        return "pybullet_boil"

    @property
    def predicates(self) -> Set[Predicate]:
        """Return a set of domain-specific predicates that might be used for planning."""
        return {self._JugFilled, self._JugHot, self._BurnerOn, self._FaucetOn}

    @property
    def types(self) -> Set[Type]:
        """All custom types in this environment."""
        return {
            self._robot_type, self._jug_type,
            self._burner_type, self._switch_type, self._faucet_type
        }

    @property
    def goal_predicates(self) -> Set[Predicate]:
        """Which predicates might appear in goals."""
        return {self._JugHot, self._JugFilled}  # Example

    # -------------------------------------------------------------------------
    # PyBullet Initialization
    # -------------------------------------------------------------------------
    @classmethod
    def initialize_pybullet(
        cls, using_gui: bool
    ) -> Tuple[int, SingleArmPyBulletRobot, Dict[str, Any]]:
        physics_client_id, pybullet_robot, bodies = super().initialize_pybullet(using_gui)

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
            jug_id = create_object(
                asset_path="urdf/jug-pixel.urdf",
                scale=1.0,
                use_fixed_base=False,
                physics_client_id=physics_client_id
            )
            jug_ids.append(jug_id)
        bodies["jug_ids"] = jug_ids

        # 3) Create burners (short cylinders). If you have a custom URDF for a burner,
        # replace the lines below; otherwise, you can do something like:
        burner_ids = []
        for _ in range(cls.num_burners):
            # Example small cylinder object (placeholder).
            # If you don't have a direct cylinder creation helper, you might
            # adapt create_pybullet_block or use your own URDF.
            # For demonstration, let's just place a small block "pretending" to be a cylinder.
            burner_id = create_pybullet_block(color=(1,1,1,1),
                                            half_extents=(0.1, 0.1, 0.01),
                                            mass=0.1,
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
                physics_client_id=physics_client_id
            )
            burner_switch_ids.append(switch_id)
        bodies["burner_switch_ids"] = burner_switch_ids

        # 5) Create faucet and faucet switch
        faucet_id = create_object(
                asset_path="urdf/partnet_mobility/faucet/1488/mobility.urdf",
                scale=1.0,
                use_fixed_base=True,
                physics_client_id=physics_client_id
        )
        bodies["faucet_id"] = faucet_id

        faucet_switch_id = create_object(
            asset_path="urdf/partnet_mobility/switch/102812/switch.urdf",
            scale=1.0,
            use_fixed_base=True,
            physics_client_id=physics_client_id
        )
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
        """If you wanted additional objects depending on a given state, add them here."""
        pass

    def _extract_feature(self, obj: Object, feature: str) -> float:
        """Map from environment object + feature name -> a float feature in the State."""
        if obj.type == self._faucet_type:
            if feature == "is_on":
                return float(self._is_switch_on(self._faucet_switch.id))
        elif obj.type == self._burner_type:
            if feature == "is_on":
                # each burner has a parallel switch
                # find the corresponding switch by index
                idx = int(obj.name.replace("burner", ""))
                sw_obj = self._burner_switches[idx]
                return float(self._is_switch_on(sw_obj.id))
        # Otherwise, rely on defaults (like the base PyBulletEnv) for x,y,z,...
        # or raise an error if unrecognized:
        raise ValueError(f"Unknown feature {feature} for object {obj}.")

    def _reset_custom_env_state(self, state: State) -> None:
        """Called in _reset_state to do any environment-specific resetting."""
        # Programmatically set switches on/off in PyBullet if needed
        # to match the provided `state`.
        for i, burner_obj in enumerate(self._burners):
            on_val = state.get(burner_obj, "is_on")
            self._set_switch_on(self._burner_switches[i].id, bool(on_val > 0.5))

        # Faucet
        f_on = state.get(self._faucet, "is_on")
        self._set_switch_on(self._faucet_switch.id, bool(f_on > 0.5))

    # -------------------------------------------------------------------------
    # Step Logic
    # -------------------------------------------------------------------------
    def step(self, action: Action, render_obs: bool = False) -> State:
        """Execute a low-level action (robot controls), then handle water filling/spillage and heating."""
        # First let the base environment perform the usual PyBullet step
        next_state = super().step(action, render_obs=render_obs)

        # 1) Handle faucet filling
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
        """If faucet is on, fill any jug that is properly aligned. Otherwise spill."""
        faucet_on = self._is_switch_on(self._faucet_switch.id)
        if not faucet_on:
            return
        # If on, check each jug's distance from the faucet:
        fx = state.get(self._faucet, "x")
        fy = state.get(self._faucet, "y")
        for jug_obj in self._jugs:
            jug_x = state.get(jug_obj, "x")
            jug_y = state.get(jug_obj, "y")
            dist = np.hypot(fx - jug_x, fy - jug_y)
            if dist < self.faucet_align_threshold:
                # Properly aligned => fill
                old_level = state.get(jug_obj, "water_level")
                new_level = old_level + self.water_fill_speed
                state.set(jug_obj, "water_level", min(1.0, new_level))
            else:
                # Spillage if jug is "close but not aligned enough"? 
                # For simplicity, we won't track puddles on the table here.
                # You could add objects or track how much is wasted, etc.
                pass

    def _handle_heating_logic(self, state: State) -> None:
        """If a jug with water is on a turned-on burner, increment jug 'heat' up to 1.0."""
        for i, burner_obj in enumerate(self._burners):
            burner_on = self._is_switch_on(self._burner_switches[i].id)
            if not burner_on:
                continue
            # Check which jug is placed on top of this burner
            bx = state.get(burner_obj, "x")
            by = state.get(burner_obj, "y")
            for jug_obj in self._jugs:
                jug_x = state.get(jug_obj, "x")
                jug_y = state.get(jug_obj, "y")
                dist = np.hypot(bx - jug_x, by - jug_y)
                if dist < self.burner_align_threshold:
                    # Jug is on top of an active burner => increase heat
                    old_heat = state.get(jug_obj, "heat")
                    if state.get(jug_obj, "water_level") > 0.0:
                        new_heat = min(1.0, old_heat + self.heating_speed)
                        state.set(jug_obj, "heat", new_heat)

    def _update_jug_colors(self, state: State) -> None:
        """Simple linear interpolation from blue (0.0) to red (1.0) based on jug.heat."""
        for jug_obj in self._jugs:
            jug_id = jug_obj.id
            if jug_id is None:
                continue
            heat = state.get(jug_obj, "heat")  # 0..1
            # Weighted interpolation from (0,0,1) => (1,0,0)
            r = heat
            g = 0.0
            b = 1.0 - heat
            alpha = 1.0
            update_object(
                jug_id,
                color=(r, g, b, alpha),
                physics_client_id=self._physics_client_id
            )

    # -------------------------------------------------------------------------
    # Switch Helpers
    # -------------------------------------------------------------------------
    def _is_switch_on(self, switch_id: int) -> bool:
        """Check if a switch's main joint is above a threshold."""
        if switch_id < 0:
            return False
        # For simplicity, we'll read joint_0
        j_id = self._get_joint_id(switch_id, "joint_0")
        if j_id < 0:
            return False
        j_pos, _, _, _ = p.getJointState(switch_id, j_id, physicsClientId=self._physics_client_id)
        info = p.getJointInfo(switch_id, j_id, physicsClientId=self._physics_client_id)
        j_min, j_max = info[8], info[9]
        frac = (j_pos - j_min) / (j_max - j_min + 1e-9)
        return bool(frac > 0.5)

    def _set_switch_on(self, switch_id: int, power_on: bool) -> None:
        """Programmatically toggle the switch to on/off by resetting its joint state."""
        j_id = self._get_joint_id(switch_id, "joint_0")
        if j_id < 0:
            return
        info = p.getJointInfo(switch_id, j_id, physicsClientId=self._physics_client_id)
        j_min, j_max = info[8], info[9]
        target_val = j_max if power_on else j_min
        p.resetJointState(
            switch_id, j_id, target_val, physicsClientId=self._physics_client_id
        )

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
        (jug,) = objects
        return state.get(jug, "water_level") >= 1.0

    @staticmethod
    def _JugHot_holds(state: State, objects: Sequence[Object]) -> bool:
        (jug,) = objects
        return state.get(jug, "heat") >= 1.0

    @staticmethod
    def _BurnerOn_holds(state: State, objects: Sequence[Object]) -> bool:
        (burner,) = objects
        return state.get(burner, "is_on") > 0.5

    @staticmethod
    def _FaucetOn_holds(state: State, objects: Sequence[Object]) -> bool:
        (faucet,) = objects
        return state.get(faucet, "is_on") > 0.5

    # -------------------------------------------------------------------------
    # Task Generation
    # -------------------------------------------------------------------------
    def _generate_train_tasks(self) -> List[EnvironmentTask]:
        return self._make_tasks(num_tasks=CFG.num_train_tasks, rng=self._train_rng)

    def _generate_test_tasks(self) -> List[EnvironmentTask]:
        return self._make_tasks(num_tasks=CFG.num_test_tasks, rng=self._test_rng)

    def _make_tasks(self, num_tasks: int, rng: np.random.Generator) -> List[EnvironmentTask]:
        """Randomly place jugs, burners, faucet, etc. for each task."""
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
            for j_obj in self._jugs:
                x, y = self._sample_xy(rng, used_xy)
                init_dict[j_obj] = {
                    "x": x, "y": y, "z": self.table_height+0.02,
                    "rot": 0.0,
                    "is_held": 0.0,
                    "water_level": 0.0,
                    "heat": 0.0
                }

            # Burners
            for i, b_obj in enumerate(self._burners):
                x, y = self._sample_xy(rng, used_xy)
                # burners are short objects on the table
                init_dict[b_obj] = {
                    "x": x, "y": y, "z": self.table_height,
                    "is_on": 0.0
                }
                # Switch for burner
                # We'll place the switch somewhere else, e.g. off to the side
                sw_obj = self._burner_switches[i]
                sw_x, sw_y = self._sample_xy(rng, used_xy)
                init_dict[sw_obj] = {
                    "x": sw_x, "y": sw_y, "z": self.table_height,
                    "rot": 0.0,
                    "is_on": 0.0
                }

            # Faucet
            fx, fy = self._sample_xy(rng, used_xy)
            init_dict[self._faucet] = {
                "x": fx, "y": fy, "z": self.table_height+0.1,
                "is_on": 0.0
            }
            # Faucet switch
            sw_fx, sw_fy = self._sample_xy(rng, used_xy)
            init_dict[self._faucet_switch] = {
                "x": sw_fx, "y": sw_fy, "z": self.table_height,
                "rot": 0.0,
                "is_on": 0.0
            }

            init_state = utils.create_state_from_dict(init_dict)

            # Example goal: all jugs hot & filled
            goal_atoms = set()
            for j_obj in self._jugs:
                goal_atoms.add(GroundAtom(self._JugFilled, [j_obj]))
                goal_atoms.add(GroundAtom(self._JugHot, [j_obj]))

            tasks.append(EnvironmentTask(init_state, goal_atoms))

        return self._add_pybullet_state_to_tasks(tasks)

    def _sample_xy(self, rng: np.random.Generator, used_xy: Set[Tuple[float, float]]) -> Tuple[float, float]:
        """Sample a random (x,y) on the table that doesn't collide with existing objects."""
        for _ in range(1000):
            x = rng.uniform(self.x_lb + 0.05, self.x_ub - 0.05)
            y = rng.uniform(self.y_lb + 0.05, self.y_ub - 0.05)
            if all((np.hypot(x - ux, y - uy) > 0.10) for (ux, uy) in used_xy):
                used_xy.add((x, y))
                return x, y
        raise RuntimeError("Failed to sample a collision-free (x, y).")


if __name__ == "__main__":
    import time
    CFG.seed = 0
    CFG.env = "pybullet_boil"
    # CFG.pybullet_sim_steps_per_action = 1
    # CFG.fan_fans_blow_opposite_direction = True
    env = PyBulletBoilEnv(use_gui=True)
    rng = np.random.default_rng(CFG.seed)
    tasks = env._make_tasks(1, rng)

    for task in tasks:
        env._reset_state(task.init)
        for _ in range(100000):
            action = Action(
                np.array(env._pybullet_robot.initial_joint_positions))
            env.step(action)
            time.sleep(0.01)
