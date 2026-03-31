"""Composed PyBullet domino environment.

This module provides the main environment class that composes multiple
components (dominoes, fans, balls, etc.) into a single environment.
"""

from typing import Any, ClassVar, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pybullet as p

from predicators import utils
from predicators.envs.pybullet_domino.components.ball_component import \
    BallComponent
from predicators.envs.pybullet_domino.components.base_component import \
    DominoEnvComponent
from predicators.envs.pybullet_domino.components.domino_component import \
    DominoComponent
from predicators.envs.pybullet_domino.components.fan_component import \
    FanComponent
from predicators.envs.pybullet_domino.components.ramp_component import \
    RampComponent
from predicators.envs.pybullet_domino.components.stairs_component import \
    StairsComponent
# pylint: disable-next=line-too-long
from predicators.envs.pybullet_domino.task_generators.domino_task_generator import \
    DominoTaskGenerator
from predicators.envs.pybullet_env import PyBulletEnv
from predicators.pybullet_helpers.camera import create_gui_connection
from predicators.pybullet_helpers.geometry import Pose, Pose3D, Quaternion
from predicators.pybullet_helpers.objects import create_object
from predicators.pybullet_helpers.robots import SingleArmPyBulletRobot, \
    create_single_arm_pybullet_robot
from predicators.settings import CFG
from predicators.structs import Action, EnvironmentTask, Object, Predicate, \
    State, Type


class PyBulletDominoComposedEnv(PyBulletEnv):
    """A PyBullet domino environment composed of modular components.

    This environment supports:
    - Domino blocks that can topple through collisions
    - Fans that blow wind (optional)
    - Balls that can be moved by wind and collisions (optional)
    - Additional components can be added via the component system

    Components are initialized and composed at construction time.
    """

    # =========================================================================
    # TABLE / WORKSPACE CONFIGURATION
    # =========================================================================
    table_height: ClassVar[float] = 0.4
    table_pos: ClassVar[Pose3D] = (0.75, 1.35, table_height / 2)
    table_orn: ClassVar[Quaternion] = tuple(
        p.getQuaternionFromEuler([0., 0., np.pi / 2]))
    table_width: ClassVar[float] = 1.0

    x_lb: ClassVar[float] = 0.4
    x_ub: ClassVar[float] = 1.1
    y_lb: ClassVar[float] = 1.1
    y_ub: ClassVar[float] = 1.6
    z_lb: ClassVar[float] = table_height
    z_ub: ClassVar[float] = 0.95

    # =========================================================================
    # ROBOT CONFIGURATION
    # =========================================================================
    robot_init_x: ClassVar[float] = (x_lb + x_ub) * 0.5
    robot_init_y: ClassVar[float] = (y_lb + y_ub) * 0.5
    robot_init_z: ClassVar[float] = z_ub
    robot_base_pos: ClassVar[Optional[Tuple[float, float,
                                            float]]] = (0.75, 0.72, 0.0)
    robot_base_orn: ClassVar[Optional[Tuple[float, float, float, float]]] = \
        tuple(p.getQuaternionFromEuler([0.0, 0.0, np.pi / 2]))
    robot_init_tilt: ClassVar[float] = np.pi / 2
    robot_init_wrist: ClassVar[float] = -np.pi / 2

    # =========================================================================
    # CAMERA CONFIGURATION
    # =========================================================================
    _camera_distance: ClassVar[float] = 1.3
    _camera_yaw: ClassVar[float] = -70
    _camera_pitch: ClassVar[float] = -40
    _camera_target: ClassVar[Pose3D] = (0.75, 1.25, 0.42)

    # =========================================================================
    # DOMINO CONFIGURATION
    # =========================================================================
    # Domino shape properties
    domino_width: ClassVar[float] = 0.07
    domino_depth: ClassVar[float] = 0.015
    domino_height: ClassVar[float] = 0.15
    domino_mass: ClassVar[float] = 0.1
    domino_friction: ClassVar[float] = 0.5
    pos_gap: ClassVar[float] = 0.098  # domino_width * 1.4, computed value

    # Type definitions
    _robot_type = Type("robot", ["x", "y", "z", "fingers", "tilt", "wrist"])
    _out_of_view_xy: ClassVar[Sequence[float]] = [10.0, 10.0]

    def __init__(self,
                 components: List[DominoEnvComponent],
                 use_gui: bool = False) -> None:
        """Initialize the composed domino environment.

        Args:
            components: List of components to include in the environment.
            use_gui: Whether to use PyBullet GUI.
        """
        self._components = components

        # Create robot object
        self._robot = Object("robot", self._robot_type)

        # Find specific component types for convenience
        # (must be done before _create_robot_predicates)
        self._domino_component: Optional[DominoComponent] = None
        self._fan_component: Optional[FanComponent] = None
        self._ball_component: Optional[BallComponent] = None

        for comp in components:
            if isinstance(comp, DominoComponent):
                self._domino_component = comp
            elif isinstance(comp, FanComponent):
                self._fan_component = comp
            elif isinstance(comp, BallComponent):
                self._ball_component = comp

        # Create predicates for robot (HandEmpty, Holding)
        self._create_robot_predicates()

        # Wire up fan -> ball wind connection if both present
        # (done after PyBullet init in _store_pybullet_bodies)

        super().__init__(use_gui)

    def _create_robot_predicates(self) -> None:
        """Create robot-specific predicates."""
        if self._domino_component is not None:
            domino_type = self._domino_component.domino_type
            self._HandEmpty = Predicate("HandEmpty", [self._robot_type],
                                        self._HandEmpty_holds)
            self._Holding: Optional[Predicate] = Predicate(
                "Holding", [self._robot_type, domino_type],
                self._Holding_holds)
        else:
            # Create dummy predicates if no domino component
            self._HandEmpty = Predicate("HandEmpty", [self._robot_type],
                                        lambda s, o: True)
            self._Holding = None

    @classmethod
    def get_name(cls) -> str:
        return "pybullet_domino_composed"

    # =========================================================================
    # PROPERTIES (Types, Predicates, etc.)
    # =========================================================================

    @property
    def types(self) -> Set[Type]:
        """Return all types from all components plus robot type."""
        all_types = {self._robot_type}
        for comp in self._components:
            all_types |= comp.get_types()
        return all_types

    @property
    def predicates(self) -> Set[Predicate]:
        """Return all predicates from all components plus robot predicates."""
        all_preds = {self._HandEmpty}
        if self._Holding is not None:
            all_preds.add(self._Holding)
        for comp in self._components:
            all_preds |= comp.get_predicates()
        return all_preds

    @property
    def goal_predicates(self) -> Set[Predicate]:
        """Return goal predicates from all components."""
        goal_preds: Set[Predicate] = set()
        for comp in self._components:
            goal_preds |= comp.get_goal_predicates()
        return goal_preds

    # =========================================================================
    # PYBULLET INITIALIZATION
    # =========================================================================

    @classmethod
    def initialize_pybullet(
            cls, using_gui: bool
    ) -> Tuple[int, SingleArmPyBulletRobot, Dict[str, Any]]:
        """Initialize PyBullet simulation.

        Note: Component initialization happens in instance method since
        components are instance-specific.
        """
        if using_gui:
            physics_client_id = create_gui_connection(
                camera_distance=cls._camera_distance,
                camera_yaw=cls._camera_yaw,
                camera_pitch=cls._camera_pitch,
                camera_target=cls._camera_target,
            )
        else:
            physics_client_id = p.connect(p.DIRECT)

        p.resetSimulation(physicsClientId=physics_client_id)

        # Load plane
        p.loadURDF(utils.get_env_asset_path("urdf/plane.urdf"), [0, 0, 0],
                   useFixedBase=True,
                   physicsClientId=physics_client_id)

        # Load robot
        robot_ee_orn = cls.get_robot_ee_home_orn()
        ee_home = Pose((cls.robot_init_x, cls.robot_init_y, cls.robot_init_z),
                       robot_ee_orn)
        if cls.robot_base_pos is not None and cls.robot_base_orn is not None:
            base_pose: Optional[Pose] = Pose(cls.robot_base_pos,
                                             cls.robot_base_orn)
        else:
            base_pose = None
        pybullet_robot = create_single_arm_pybullet_robot(
            CFG.pybullet_robot, physics_client_id, ee_home, base_pose)

        # Set gravity
        p.setGravity(0., 0., -10., physicsClientId=physics_client_id)

        # Create table
        table_id = create_object(asset_path="urdf/table.urdf",
                                 position=cls.table_pos,
                                 orientation=cls.table_orn,
                                 scale=1.0,
                                 use_fixed_base=True,
                                 physics_client_id=physics_client_id)

        # Add second table for more space
        table_id2 = create_object(
            asset_path="urdf/table.urdf",
            position=(cls.table_pos[0], cls.table_pos[1] + cls.table_width / 2,
                      cls.table_pos[2]),
            orientation=cls.table_orn,
            scale=1.0,
            use_fixed_base=True,
            physics_client_id=physics_client_id)

        bodies = {"table_id": table_id, "table_id2": table_id2}
        return physics_client_id, pybullet_robot, bodies

    def _store_pybullet_bodies(self, pybullet_bodies: Dict[str, Any]) -> None:
        """Initialize and store PyBullet bodies for all components."""
        self._table_ids = [
            pybullet_bodies["table_id"], pybullet_bodies["table_id2"]
        ]
        # Initialize each component
        for comp in self._components:
            comp.set_physics_client_id(self._physics_client_id)
            comp_bodies = comp.initialize_pybullet(self._physics_client_id)
            comp.store_pybullet_bodies(comp_bodies)

        # Wire up fan -> ball connection if both present
        if self._fan_component is not None and self._ball_component is not None:
            self._fan_component.set_wind_target(self._ball_component.ball_id)

    # =========================================================================
    # STATE MANAGEMENT
    # =========================================================================

    def _get_object_ids_for_held_check(self) -> List[int]:
        """Return object IDs that can be held by robot."""
        ids = []
        for comp in self._components:
            ids.extend(comp.get_object_ids_for_held_check())
        return ids

    def _create_task_specific_objects(self, state: State) -> None:
        """Create any task-specific objects (not used in current impl)."""

    def _extract_feature(self, obj: Object, feature: str) -> float:
        """Extract state feature for an object."""
        # Try each component
        for comp in self._components:
            result = comp.extract_feature(obj, feature)
            if result is not None:
                return result

        raise ValueError(f"Unknown feature {feature} for object {obj}")

    def _reset_custom_env_state(self, state: State) -> None:
        """Reset environment to match the given state."""
        # Update ball component's state reference for is_hit feature
        if self._ball_component is not None:
            self._ball_component.set_current_state(state)

        # Reset each component
        for comp in self._components:
            comp.reset_state(state)

    def step(self, action: Action, render_obs: bool = False) -> State:
        """Execute action and run component physics updates."""
        super().step(action, render_obs=render_obs)

        # Run component step functions (e.g., fan wind simulation)
        for comp in self._components:
            comp.step()

        final_state = self._get_state()
        self._current_observation = final_state

        # Update ball component's state reference
        if self._ball_component is not None:
            self._ball_component.set_current_state(final_state)

        return final_state

    # =========================================================================
    # PREDICATE HOLD FUNCTIONS
    # =========================================================================

    def _HandEmpty_holds(self, state: State,
                         _objects: Sequence[Object]) -> bool:
        """Check if robot hand is empty."""
        if self._domino_component is None:
            return True
        dominos = state.get_objects(self._domino_component.domino_type)
        for domino in dominos:
            if state.get(domino, "is_held"):
                return False
        return True

    def _Holding_holds(self, state: State, objects: Sequence[Object]) -> bool:
        """Check if robot is holding a specific domino."""
        _, domino = objects
        return state.get(domino, "is_held") > 0.5

    # =========================================================================
    # TASK GENERATION
    # =========================================================================

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
        """Generate tasks using task generator."""
        if self._domino_component is None:
            raise ValueError("Cannot generate tasks without domino component")

        # Create task generator
        robot_init_state = {
            "x": self.robot_init_x,
            "y": self.robot_init_y,
            "z": self.robot_init_z,
            "fingers": self.open_fingers,
            "tilt": self.robot_init_tilt,
            "wrist": self.robot_init_wrist,
        }

        # Collect additional components for init dict (all except domino)
        additional_components = []
        for comp in self._components:
            if comp is not self._domino_component:
                additional_components.append(comp)

        generator = DominoTaskGenerator(
            domino_component=self._domino_component,
            robot=self._robot,
            robot_init_state=robot_init_state,
            additional_components=additional_components)

        # If ball component is present, place dominoes in upper half
        # to leave space for ball in lower half
        domino_in_upper_half = self._ball_component is not None

        tasks = generator.generate_tasks(
            num_tasks=num_tasks,
            rng=rng,
            log_debug=log_debug,
            possible_num_dominos=possible_num_dominos,
            possible_num_targets=possible_num_targets,
            possible_num_pivots=possible_num_pivots,
            domino_in_upper_half=domino_in_upper_half)

        return self._add_pybullet_state_to_tasks(tasks)


# =============================================================================
# BACKWARD-COMPATIBLE ENVIRONMENT CLASSES
# =============================================================================


class PyBulletDominoEnvNew(PyBulletDominoComposedEnv):
    """Backward-compatible domino environment class."""

    def __init__(self, use_gui: bool = False) -> None:
        workspace_bounds = {
            "x_lb": self.x_lb,
            "x_ub": self.x_ub,
            "y_lb": self.y_lb,
            "y_ub": self.y_ub,
            "z_lb": self.z_lb,
            "z_ub": self.z_ub,
        }

        max_dominos = max(max(CFG.domino_train_num_dominos),
                          max(CFG.domino_test_num_dominos))
        max_targets = max(max(CFG.domino_train_num_targets),
                          max(CFG.domino_test_num_targets))
        max_pivots = max(max(CFG.domino_train_num_pivots),
                         max(CFG.domino_test_num_pivots))

        domino_comp = DominoComponent(num_dominos_max=max_dominos,
                                      num_targets_max=max_targets,
                                      num_pivots_max=max_pivots,
                                      workspace_bounds=workspace_bounds)

        super().__init__(components=[domino_comp], use_gui=use_gui)

    @classmethod
    def get_name(cls) -> str:
        return "pybullet_domino"


class PyBulletDominoFanEnvNew(PyBulletDominoComposedEnv):
    """Backward-compatible domino + fan + ball environment class."""

    def __init__(self, use_gui: bool = False) -> None:
        workspace_bounds = {
            "x_lb": self.x_lb,
            "x_ub": self.x_ub,
            "y_lb": self.y_lb,
            "y_ub": self.y_ub,
            "z_lb": self.z_lb,
            "z_ub": self.z_ub,
        }

        max_dominos = max(max(CFG.domino_train_num_dominos),
                          max(CFG.domino_test_num_dominos))
        max_targets = max(max(CFG.domino_train_num_targets),
                          max(CFG.domino_test_num_targets))
        max_pivots = max(max(CFG.domino_train_num_pivots),
                         max(CFG.domino_test_num_pivots))

        domino_comp = DominoComponent(num_dominos_max=max_dominos,
                                      num_targets_max=max_targets,
                                      num_pivots_max=max_pivots,
                                      workspace_bounds=workspace_bounds)

        fan_comp = FanComponent(workspace_bounds=workspace_bounds,
                                table_height=self.table_height,
                                table_width=self.table_width)

        ball_comp = BallComponent(workspace_bounds=workspace_bounds,
                                  table_height=self.table_height)

        super().__init__(components=[domino_comp, fan_comp, ball_comp],
                         use_gui=use_gui)

    @classmethod
    def get_name(cls) -> str:
        return "pybullet_domino_fan"

    @property
    def predicates(self) -> Set[Predicate]:
        """Include BallAtTarget in predicates."""
        preds = super().predicates
        if self._ball_component is not None:
            preds.add(self._ball_component.BallAtTarget)
        return preds

    @property
    def goal_predicates(self) -> Set[Predicate]:
        """Goals can be ball at target OR dominoes toppled."""
        preds = super().goal_predicates
        if self._ball_component is not None:
            preds.add(self._ball_component.BallAtTarget)
        return preds


class PyBulletDominoFanRampEnv(PyBulletDominoComposedEnv):
    """Domino + fan + ball + ramp environment class."""

    def __init__(self, use_gui: bool = False) -> None:
        workspace_bounds = {
            "x_lb": self.x_lb,
            "x_ub": self.x_ub,
            "y_lb": self.y_lb,
            "y_ub": self.y_ub,
            "z_lb": self.z_lb,
            "z_ub": self.z_ub,
        }

        max_dominos = max(max(CFG.domino_train_num_dominos),
                          max(CFG.domino_test_num_dominos))
        max_targets = max(max(CFG.domino_train_num_targets),
                          max(CFG.domino_test_num_targets))
        max_pivots = max(max(CFG.domino_train_num_pivots),
                         max(CFG.domino_test_num_pivots))

        domino_comp = DominoComponent(num_dominos_max=max_dominos,
                                      num_targets_max=max_targets,
                                      num_pivots_max=max_pivots,
                                      workspace_bounds=workspace_bounds)

        fan_comp = FanComponent(workspace_bounds=workspace_bounds,
                                table_height=self.table_height,
                                table_width=self.table_width)

        ball_comp = BallComponent(workspace_bounds=workspace_bounds,
                                  table_height=self.table_height)

        ramp_comp = RampComponent(workspace_bounds=workspace_bounds,
                                  table_height=self.table_height,
                                  max_ramps=5)

        super().__init__(
            components=[domino_comp, fan_comp, ball_comp, ramp_comp],
            use_gui=use_gui)

    @classmethod
    def get_name(cls) -> str:
        return "pybullet_domino_fan_ramp"

    @property
    def predicates(self) -> Set[Predicate]:
        """Include BallAtTarget in predicates."""
        preds = super().predicates
        if self._ball_component is not None:
            preds.add(self._ball_component.BallAtTarget)
        return preds

    @property
    def goal_predicates(self) -> Set[Predicate]:
        """Goals can be ball at target OR dominoes toppled."""
        preds = super().goal_predicates
        if self._ball_component is not None:
            preds.add(self._ball_component.BallAtTarget)
        return preds


class PyBulletDominoFanRampStairsEnv(PyBulletDominoComposedEnv):
    """Domino + fan + ball + ramp + stairs environment class."""

    def __init__(self, use_gui: bool = False) -> None:
        workspace_bounds = {
            "x_lb": self.x_lb,
            "x_ub": self.x_ub,
            "y_lb": self.y_lb,
            "y_ub": self.y_ub,
            "z_lb": self.z_lb,
            "z_ub": self.z_ub,
        }

        max_dominos = max(max(CFG.domino_train_num_dominos),
                          max(CFG.domino_test_num_dominos))
        max_targets = max(max(CFG.domino_train_num_targets),
                          max(CFG.domino_test_num_targets))
        max_pivots = max(max(CFG.domino_train_num_pivots),
                         max(CFG.domino_test_num_pivots))

        domino_comp = DominoComponent(num_dominos_max=max_dominos,
                                      num_targets_max=max_targets,
                                      num_pivots_max=max_pivots,
                                      workspace_bounds=workspace_bounds)

        fan_comp = FanComponent(workspace_bounds=workspace_bounds,
                                table_height=self.table_height,
                                table_width=self.table_width)

        ball_comp = BallComponent(workspace_bounds=workspace_bounds,
                                  table_height=self.table_height)

        ramp_comp = RampComponent(workspace_bounds=workspace_bounds,
                                  table_height=self.table_height,
                                  max_ramps=5)

        # Stairs component needs reference to domino type for positioning
        stairs_comp = StairsComponent(workspace_bounds=workspace_bounds,
                                      table_height=self.table_height,
                                      domino_type=domino_comp.domino_type,
                                      enabled=True)

        super().__init__(components=[
            domino_comp, fan_comp, ball_comp, ramp_comp, stairs_comp
        ],
                         use_gui=use_gui)

        # Store reference to stairs component
        self._stairs_component = stairs_comp

    @classmethod
    def get_name(cls) -> str:
        return "pybullet_domino_fan_ramp_stairs"

    @property
    def predicates(self) -> Set[Predicate]:
        """Include BallAtTarget in predicates."""
        preds = super().predicates
        if self._ball_component is not None:
            preds.add(self._ball_component.BallAtTarget)
        return preds

    @property
    def goal_predicates(self) -> Set[Predicate]:
        """Goals can be ball at target OR dominoes toppled."""
        preds = super().goal_predicates
        if self._ball_component is not None:
            preds.add(self._ball_component.BallAtTarget)
        return preds


if __name__ == "__main__":
    import sys
    import time

    # Choose which environment to test
    # Options: "domino", "domino_fan", "domino_fan_ramp",
    # "domino_fan_ramp_stairs"
    # Change this to test different environments
    test_env = "domino_fan_ramp_stairs"
    if len(sys.argv) > 1:
        test_env = sys.argv[1]

    # Configure environment
    CFG.seed = 0
    CFG.num_train_tasks = 0
    CFG.num_test_tasks = 3

    # Domino configuration
    CFG.domino_initialize_at_finished_state = True
    CFG.domino_use_domino_blocks_as_target = True
    CFG.domino_has_glued_dominos = False
    CFG.domino_test_num_dominos = [3, 4]
    CFG.domino_test_num_targets = [1]
    CFG.domino_test_num_pivots = [0]

    # Fan/ball configuration
    CFG.domino_fan_ball_position_tolerance = 0.04
    CFG.fan_known_controls_relation = True
    CFG.fan_fans_blow_opposite_direction = False

    # Create environment based on selection
    env: PyBulletDominoComposedEnv
    if test_env == "domino":
        print("Creating PyBulletDominoEnvNew...")
        CFG.env = "pybullet_domino"
        env = PyBulletDominoEnvNew(use_gui=True)
    elif test_env == "domino_fan":
        print("Creating PyBulletDominoFanEnvNew...")
        CFG.env = "pybullet_domino_fan"
        env = PyBulletDominoFanEnvNew(use_gui=True)
    elif test_env == "domino_fan_ramp":
        print("Creating PyBulletDominoFanRampEnv...")
        CFG.env = "pybullet_domino_fan_ramp"
        env = PyBulletDominoFanRampEnv(use_gui=True)
    elif test_env == "domino_fan_ramp_stairs":
        print("Creating PyBulletDominoFanRampStairsEnv...")
        CFG.env = "pybullet_domino_fan_ramp_stairs"
        env = PyBulletDominoFanRampStairsEnv(use_gui=True)
    else:
        raise ValueError(f"Unknown environment: {test_env}")

    # Generate test tasks
    print("Generating test tasks...")
    test_tasks = env._generate_test_tasks()  # pylint: disable=protected-access

    print(f"\nGenerated {len(test_tasks)} tasks")
    print(f"Types: {[t.name for t in env.types]}")
    print(f"Predicates: {[p.name for p in env.predicates]}")

    # Test each task
    for i, task in enumerate(test_tasks):
        print(f"\n{'=' * 60}")
        print(f"Task {i + 1}")
        print(f"{'=' * 60}")

        # Reset to initial state
        env._reset_state(task.init)  # pylint: disable=protected-access

        print("\nGoal atoms:")
        for atom in task.goal:
            print(f"  {atom}")

        try:
            for step in range(100000):
                # pylint: disable=protected-access
                cur_action = Action(
                    np.array(env._pybullet_robot.initial_joint_positions))
                cur_state = env.step(cur_action)

                if all(atom.holds(cur_state) for atom in task.goal):
                    print(f"Goal reached at step {step}!")
                    time.sleep(2)
                    break

                time.sleep(0.02)
        except KeyboardInterrupt:
            continue

    print("\nDone!")
