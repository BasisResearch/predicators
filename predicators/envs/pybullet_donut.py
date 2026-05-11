"""A PyBullet environment with donut-shaped objects and a target area."""

from typing import Any, ClassVar, Dict, List, Sequence, Set, Tuple

import numpy as np
import pybullet as p

from predicators import utils
from predicators.envs.pybullet_env import PyBulletEnv
from predicators.pybullet_helpers.geometry import Pose3D, Quaternion
from predicators.pybullet_helpers.objects import update_object
from predicators.pybullet_helpers.robots import SingleArmPyBulletRobot
from predicators.settings import CFG
from predicators.structs import Action, EnvironmentTask, GroundAtom, Object, \
    Observation, Predicate, State, Type


class PyBulletDonutEnv(PyBulletEnv):
    """PyBullet Donut domain."""

    # Parameters
    table_height: ClassVar[float] = 0.2
    _table_pose: ClassVar[Pose3D] = (1.35, 0.75, 0.0) # Surface at 0.2 with 1x scaling
    _table_orientation: ClassVar[Quaternion] = (0., 0., 0., 1.)

    # Workspace (within table bounds: X [1.125, 1.575], Y [0.375, 1.125])
    x_lb: ClassVar[float] = 1.15
    x_ub: ClassVar[float] = 1.55
    y_lb: ClassVar[float] = 0.4
    y_ub: ClassVar[float] = 1.1

    # Robot init
    robot_init_x: ClassVar[float] = 1.35
    robot_init_y: ClassVar[float] = 0.75
    robot_init_z: ClassVar[float] = 0.5
    robot_base_pos: ClassVar[Tuple[float, float, float]] = (0.75, 0.75, 0.0)
    robot_base_orn: ClassVar[Tuple[float, float, float, float]] = (0., 0., 0., 1.)

    # Donut parameters
    donut_major_radius: ClassVar[float] = 0.02
    donut_minor_radius: ClassVar[float] = 0.013
    donut_mass: ClassVar[float] = 0.2
    donut_friction: ClassVar[float] = 0.5

    # Target parameters
    target_width: ClassVar[float] = 0.1
    target_height: ClassVar[float] = 0.1
    target_color: ClassVar[Tuple[float, float, float, float]] = (1.0, 0.0, 0.0, 1.0) # Red

    # Camera
    _camera_target: ClassVar[Pose3D] = (1.35, 0.75, 0.2)
    _camera_distance: ClassVar[float] = 0.6
    _camera_yaw: ClassVar[float] = 90
    _camera_pitch: ClassVar[float] = -45

    # Types
    _robot_type = Type("robot", ["pose_x", "pose_y", "pose_z", "fingers"])
    _donut_type = Type("donut", ["x", "y", "z", "is_held", "r", "g", "b"])
    _target_type = Type("target", ["x", "y", "z", "r", "g", "b"])

    def __init__(self, use_gui: bool = False, **kwargs: Any) -> None:
        self._robot = Object("robot", self._robot_type)
        self._donuts: List[Object] = []
        for i in range(5):
            self._donuts.append(Object(f"donut_{i}", self._donut_type))
        self._target = Object("target", self._target_type)

        # Predicates
        self._InTarget = Predicate("InTarget", [self._donut_type, self._target_type],
                                   self._InTarget_holds)

        self._step_count = 0

        super().__init__(use_gui, **kwargs)

    @classmethod
    def get_name(cls) -> str:
        return "pybullet_donut"

    @property
    def types(self) -> Set[Type]:
        return {self._robot_type, self._donut_type, self._target_type}

    @property
    def predicates(self) -> Set[Predicate]:
        return {self._InTarget}

    @property
    def goal_predicates(self) -> Set[Predicate]:
        return {self._InTarget}

    @classmethod
    def initialize_pybullet(
            cls, using_gui: bool
    ) -> Tuple[int, SingleArmPyBulletRobot, Dict[str, Any]]:
        physics_client_id, pybullet_robot, bodies = super().initialize_pybullet(using_gui)

        # Table (scaled 1x)
        table_id = p.loadURDF(utils.get_env_asset_path("urdf/table.urdf"),
                              useFixedBase=True,
                              globalScaling=1.0,
                              physicsClientId=physics_client_id)
        p.resetBasePositionAndOrientation(table_id,
                                          cls._table_pose,
                                          cls._table_orientation,
                                          physicsClientId=physics_client_id)
        bodies["table_id"] = table_id

        # Target (flat box)
        collision_id = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=(cls.target_width / 2, cls.target_height / 2, 0.001),
            physicsClientId=physics_client_id
        )
        visual_id = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=(cls.target_width / 2, cls.target_height / 2, 0.001),
            rgbaColor=cls.target_color,
            physicsClientId=physics_client_id
        )
        target_id = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=collision_id,
            baseVisualShapeIndex=visual_id,
            physicsClientId=physics_client_id
        )
        bodies["target_id"] = target_id

        # Donuts
        donut_ids = []
        for _ in range(5):
            donut_id = cls._create_pybullet_donut(
                color=(0.8, 0.5, 0.2, 1.0),
                major_radius=cls.donut_major_radius,
                minor_radius=cls.donut_minor_radius,
                mass=cls.donut_mass,
                friction=cls.donut_friction,
                physics_client_id=physics_client_id
            )
            donut_ids.append(donut_id)
        bodies["donut_ids"] = donut_ids

        return physics_client_id, pybullet_robot, bodies

    @staticmethod
    def _create_pybullet_donut(
        color: Tuple[float, float, float, float],
        major_radius: float,
        minor_radius: float,
        mass: float,
        friction: float,
        position: Pose3D = (0.0, 0.0, 0.0),
        orientation: Quaternion = (0.0, 0.0, 0.0, 1.0),
        physics_client_id: int = 0,
        n: int = 16,
        m: int = 12,
    ) -> int:
        vertices = []
        indices = []
        for i in range(n):
            u = i / n * 2 * np.pi
            for j in range(m):
                v = j / m * 2 * np.pi
                x = (major_radius + minor_radius * np.cos(v)) * np.cos(u)
                y = (major_radius + minor_radius * np.cos(v)) * np.sin(u)
                z = minor_radius * np.sin(v)
                vertices.append([x, y, z])
                
                i1 = i * m + j
                i2 = ((i + 1) % n) * m + j
                i3 = ((i + 1) % n) * m + (j + 1) % m
                i4 = i * m + (j + 1) % m
                indices.extend([i1, i2, i3])
                indices.extend([i1, i3, i4])

        collision_id = p.createCollisionShape(
            p.GEOM_MESH,
            vertices=vertices,
            indices=indices,
            physicsClientId=physics_client_id
        )
        visual_id = p.createVisualShape(
            p.GEOM_MESH,
            vertices=vertices,
            indices=indices,
            rgbaColor=color,
            physicsClientId=physics_client_id
        )
        donut_id = p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=collision_id,
            baseVisualShapeIndex=visual_id,
            basePosition=position,
            baseOrientation=orientation,
            physicsClientId=physics_client_id
        )
        p.changeDynamics(donut_id, -1, lateralFriction=friction,
                         physicsClientId=physics_client_id)
        return donut_id

    def _store_pybullet_bodies(self, pybullet_bodies: Dict[str, Any]) -> None:
        self._table_id = pybullet_bodies["table_id"]
        self._target_id = pybullet_bodies["target_id"]
        self._donut_ids = pybullet_bodies["donut_ids"]

    def _get_object_ids_for_held_check(self) -> List[int]:
        return self._donut_ids

    def _get_domain_specific_feature(self, obj: Object, feature: str) -> float:
        raise ValueError(f"Unknown feature {feature} for object {obj}")

    def _set_seed(self, seed: int) -> None:
        super()._set_seed(seed)
        self._rng = np.random.default_rng(seed)

    def reset(self, train_or_test: str, task_idx: int, render: bool = False) -> Observation:
        self._step_count = 0
        return super().reset(train_or_test, task_idx, render=render)

    def _domain_specific_step(self) -> None:
        self._step_count += 1
        spawn_interval = 100
        if self._step_count > 0 and self._step_count % spawn_interval == 0:
            self._spawn_donut()

    def _spawn_donut(self) -> None:
        # Find first OOV donut
        oov_idx = -1
        for i, donut_id in enumerate(self._donut_ids):
            (dx, dy, dz), _ = p.getBasePositionAndOrientation(
                donut_id, physicsClientId=self._physics_client_id)
            if dx > 5.0: # Check if it's OOV (x=10.0)
                oov_idx = i
                break
        
        if oov_idx == -1:
            return

        # Sample position avoiding others
        from predicators.utils import Circle
        existing_geoms = []
        for i, donut_id in enumerate(self._donut_ids):
            if i == oov_idx: continue
            (dx, dy, dz), _ = p.getBasePositionAndOrientation(
                donut_id, physicsClientId=self._physics_client_id)
            if dz < 5.0:
                existing_geoms.append(Circle(dx, dy, self.donut_major_radius))
        
        # Avoid target
        (tx, ty, tz), _ = p.getBasePositionAndOrientation(
            self._target_id, physicsClientId=self._physics_client_id)
        existing_geoms.append(Circle(tx, ty, self.target_width / 2))

        # Avoid robot base
        existing_geoms.append(Circle(self.robot_base_pos[0], self.robot_base_pos[1], 0.1))

        for _ in range(100):
            px = self._rng.uniform(self.x_lb + self.donut_major_radius,
                                   self.x_ub - self.donut_major_radius)
            py = self._rng.uniform(self.y_lb + self.donut_major_radius,
                                   self.y_ub - self.donut_major_radius)
            new_geom = Circle(px, py, self.donut_major_radius)
            if not any(new_geom.intersects(g) for g in existing_geoms):
                p.resetBasePositionAndOrientation(
                    self._donut_ids[oov_idx],
                    [px, py, self.table_height + self.donut_minor_radius + 0.5],
                    (0., 0., 0., 1.),
                    physicsClientId=self._physics_client_id)
                break

    def _set_domain_specific_state(self, state: State) -> None:
        # Target
        target_obj = state.get_objects(self._target_type)[0]
        tx = state.get(target_obj, "x")
        ty = state.get(target_obj, "y")
        tz = state.get(target_obj, "z")
        update_object(self._target_id, position=(tx, ty, tz), 
                      physics_client_id=self._physics_client_id)

        # Donuts
        donut_objs = state.get_objects(self._donut_type)
        for i, donut_obj in enumerate(donut_objs):
            donut_id = self._donut_ids[i]
            dx = state.get(donut_obj, "x")
            dy = state.get(donut_obj, "y")
            dz = state.get(donut_obj, "z")
            p.resetBasePositionAndOrientation(
                donut_id, [dx, dy, dz],
                (0., 0., 0., 1.),
                physicsClientId=self._physics_client_id)
            
            r = state.get(donut_obj, "r")
            g = state.get(donut_obj, "g")
            b = state.get(donut_obj, "b")
            p.changeVisualShape(donut_id, -1, rgbaColor=(r, g, b, 1.0),
                                physicsClientId=self._physics_client_id)

        # Grasping
        held_donut = None
        for donut in donut_objs:
            if state.get(donut, "is_held") > 0.5:
                held_donut = donut
                break
        if held_donut is not None:
            idx = donut_objs.index(held_donut)
            self._held_obj_id = self._donut_ids[idx]
            self._create_grasp_constraint()

    def _get_state(self) -> State:
        state_dict = {}
        # Robot
        rx, ry, rz, _, _, _, _, rf = self._pybullet_robot.get_state()
        fingers = self._fingers_joint_to_state(self._pybullet_robot, rf)
        state_dict[self._robot] = np.array([rx, ry, rz, fingers], dtype=np.float32)
        # Target
        (tx, ty, tz), _ = p.getBasePositionAndOrientation(self._target_id,
                                                         physicsClientId=self._physics_client_id)
        visual_data = p.getVisualShapeData(self._target_id,
                                           physicsClientId=self._physics_client_id)[0]
        tr, tg, tb, _ = visual_data[7]
        state_dict[self._target] = np.array([tx, ty, tz, tr, tg, tb], dtype=np.float32)
        # Donuts
        for i, donut_id in enumerate(self._donut_ids):
            (dx, dy, dz), _ = p.getBasePositionAndOrientation(donut_id,
                                                             physicsClientId=self._physics_client_id)
            held = (donut_id == self._held_obj_id)
            visual_data = p.getVisualShapeData(donut_id,
                                               physicsClientId=self._physics_client_id)[0]
            dr, dg, db, _ = visual_data[7]
            if i < len(self._donuts):
                state_dict[self._donuts[i]] = np.array([dx, dy, dz, float(held), dr, dg, db],
                                                       dtype=np.float32)
        return utils.PyBulletState(state_dict, 
                                   simulator_state=self._pybullet_robot.get_joints())

    @classmethod
    def _InTarget_holds(cls, state: State, objects: Sequence[Object]) -> bool:
        donut, target = objects
        dx = state.get(donut, "x")
        dy = state.get(donut, "y")
        tx = state.get(target, "x")
        ty = state.get(target, "y")
        dist = np.sqrt((dx - tx)**2 + (dy - ty)**2)
        return dist < (cls.target_width / 2)

    def _generate_train_tasks(self) -> List[EnvironmentTask]:
        return self._make_tasks(num_tasks=CFG.num_train_tasks, rng=self._train_rng)

    def _generate_test_tasks(self) -> List[EnvironmentTask]:
        return self._make_tasks(num_tasks=CFG.num_test_tasks, rng=self._test_rng)

    def _make_tasks(self, num_tasks: int, rng: np.random.Generator) -> List[EnvironmentTask]:
        from predicators.pybullet_helpers.objects import \
            sample_collision_free_2d_positions
        tasks = []
        for _ in range(num_tasks):
            init_dict = {}
            init_dict[self._robot] = {
                "pose_x": self.robot_init_x,
                "pose_y": self.robot_init_y,
                "pose_z": self.robot_init_z,
                "fingers": self.open_fingers
            }
            target_x = self.x_ub - self.target_width / 2 - 0.1
            target_y = self.y_ub - self.target_height / 2 + 0.07
            init_dict[self._target] = {
                "x": target_x,
                "y": target_y,
                "z": self.table_height + 0.005, # Slightly above surface
                "r": self.target_color[0],
                "g": self.target_color[1],
                "b": self.target_color[2]
            }
            donut_positions = sample_collision_free_2d_positions(
                1, # Start with 1 donut on table
                x_range=(self.x_lb + self.donut_major_radius, self.x_ub - self.donut_major_radius),
                y_range=(self.y_lb + self.donut_major_radius, self.y_lb + 0.05),
                shape_type="circle",
                shape_params=(self.donut_major_radius,),
                rng=rng
            )
            for i, donut in enumerate(self._donuts):
                if i == 0:
                    pos = donut_positions[0]
                    z = self.table_height + self.donut_minor_radius
                else:
                    pos = self._out_of_view_xy
                    z = 10.0
                
                # color = self._obj_colors[i % len(self._obj_colors)]
                color = self._obj_colors[0] # single color
                init_dict[donut] = {
                    "x": pos[0],
                    "y": pos[1],
                    "z": z,
                    "is_held": 0.0,
                    "r": color[0],
                    "g": color[1],
                    "b": color[2]
                }
            init_state = utils.create_state_from_dict(init_dict)
            goal = {GroundAtom(self._InTarget, [self._donuts[0], self._target])}
            tasks.append(EnvironmentTask(init_state, goal))
        return self._add_pybullet_state_to_tasks(tasks)

if __name__ == "__main__":
    import time
    CFG.seed = 0
    CFG.env = "pybullet_donut"
    CFG.num_train_tasks = 1
    env = PyBulletDonutEnv(use_gui=True)
    _task = env._generate_train_tasks()[0]
    env.reset("train", 0)
    while True:
        _act = Action(np.array(env._pybullet_robot.get_joints()))
        env.step(_act)
        time.sleep(0.01)
