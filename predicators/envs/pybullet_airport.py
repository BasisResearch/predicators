"""A PyBullet version of AirportEnv."""
from typing import Any, ClassVar, Dict, List, Optional, Tuple

import numpy as np
import pybullet as p

from predicators import utils
from predicators.envs.airport import AirportEnv
from predicators.envs.pybullet_env import PyBulletEnv
from predicators.pybullet_helpers.geometry import Pose3D, Quaternion
from predicators.pybullet_helpers.objects import create_pybullet_block
from predicators.pybullet_helpers.robots import SingleArmPyBulletRobot
from predicators.settings import CFG
from predicators.structs import Action, EnvironmentTask, Object, State


class PyBulletAirportEnv(PyBulletEnv, AirportEnv):
    """PyBullet Airport domain."""
    _camera_distance: ClassVar[float] = 0.9
    _camera_yaw: ClassVar[float] = 0.0
    _camera_pitch: ClassVar[float] = -45.0
    _camera_target: ClassVar[Pose3D] = (1.5, 0.75, 0.4)
    robot_init_x: ClassVar[float] = 1.5
    robot_init_y: ClassVar[float] = 0.75
    robot_init_z: ClassVar[float] = 0.5
    y_lb: ClassVar[float] = 0.2
    y_ub: ClassVar[float] = 1.3
    robot_base_pos: ClassVar[Pose3D] = (1.5, 1.5, 0.0)
    robot_base_orn: ClassVar[Quaternion] = (0.0, 0.0, -0.7071, 0.7071)
    conveyor_height: ClassVar[float] = 0.4
    conveyor_width: ClassVar[float] = 0.4
    conveyor_length: ClassVar[float] = 5.0
    conveyor_x: ClassVar[float] = 1.5
    conveyor_y: ClassVar[float] = 0.5
    button_stand_x: ClassVar[float] = 1.2
    button_stand_y: ClassVar[float] = 1.2
    button_stand_z: ClassVar[float] = 0.4
    button_radius: ClassVar[float] = 0.05
    button_height: ClassVar[float] = 0.05
    pusher_width: ClassVar[float] = 0.1
    pusher_length: ClassVar[float] = 0.3
    pusher_height: ClassVar[float] = 0.1
    pusher_init_x: ClassVar[float] = 2.0
    pusher_init_y: ClassVar[float] = 0.1
    pusher_init_z: ClassVar[float] = 0.45
    table_width: ClassVar[float] = 0.6
    table_length: ClassVar[float] = 0.6
    table_height: ClassVar[float] = 0.4
    table_x: ClassVar[float] = 2.0
    table_y: ClassVar[float] = 1.0

    def __init__(self, use_gui: bool = False, **kwargs: Any) -> None:
        self._conveyor_id: int = -1
        self._button_id: int = -1
        self._button_stand_id: int = -1
        self._pusher_id: int = -1
        self._table_id: int = -1
        self._item_ids: List[int] = []
        super().__init__(use_gui, **kwargs)

    @classmethod
    def get_name(cls) -> str:
        return "pybullet_airport"

    @classmethod
    def initialize_pybullet(
            cls, using_gui: bool
    ) -> Tuple[int, SingleArmPyBulletRobot, Dict[str, Any]]:
        physics_client_id, pybullet_robot, bodies = super(
        ).initialize_pybullet(using_gui)
        conveyor_id = create_pybullet_block(
            color=(0.3, 0.3, 0.3, 1.0),
            half_extents=(cls.conveyor_length / 2.0, cls.conveyor_width / 2.0,
                          cls.conveyor_height / 2.0),
            mass=0.0,
            friction=1.0,
            position=(cls.conveyor_x + cls.conveyor_length / 2.0,
                      cls.conveyor_y, cls.conveyor_height / 2.0),
            physics_client_id=physics_client_id)
        bodies["conveyor_id"] = conveyor_id

        button_stand_id = create_pybullet_block(
            color=(0.5, 0.5, 0.5, 1.0),
            half_extents=(0.1, 0.1, cls.button_stand_z / 2.0),
            mass=0.0,
            friction=1.0,
            position=(cls.button_stand_x, cls.button_stand_y,
                      cls.button_stand_z / 2.0),
            physics_client_id=physics_client_id)
        bodies["button_stand_id"] = button_stand_id

        button_id = create_pybullet_block(
            color=(1.0, 0.0, 0.0, 1.0),
            half_extents=(cls.button_radius, cls.button_radius,
                          cls.button_height / 2.0),
            mass=0.005,
            friction=1.0,
            position=(cls.button_stand_x, cls.button_stand_y,
                      cls.button_stand_z + cls.button_height / 2.0),
            physics_client_id=physics_client_id)
        bodies["button_id"] = button_id
        # Constraint to keep button on stand but allow vertical movement
        p.createConstraint(button_stand_id,
                           -1,
                           button_id,
                           -1,
                           p.JOINT_PRISMATIC,
                           jointAxis=[0, 0, 1],
                           parentFramePosition=[0, 0, cls.button_stand_z / 2.0],
                           childFramePosition=[0, 0, -cls.button_height / 2.0],
                           physicsClientId=physics_client_id)

        pusher_id = create_pybullet_block(
            color=(0.8, 0.8, 0.2, 1.0),
            half_extents=(cls.pusher_length / 2.0, cls.pusher_width / 2.0,
                          cls.pusher_height / 2.0),
            mass=0.0,
            friction=1.0,
            position=(cls.pusher_init_x, cls.pusher_init_y, cls.pusher_init_z),
            physics_client_id=physics_client_id)
        bodies["pusher_id"] = pusher_id

        table_id = create_pybullet_block(
            color=(0.4, 0.2, 0.0, 1.0),
            half_extents=(cls.table_length / 2.0, cls.table_width / 2.0,
                          cls.table_height / 2.0),
            mass=0.0,
            friction=1.0,
            position=(cls.table_x, cls.table_y, cls.table_height / 2.0),
            physics_client_id=physics_client_id)
        bodies["table_id"] = table_id

        num_items = 7
        item_ids = []
        for i in range(num_items):
            color = cls._obj_colors[i % len(cls._obj_colors)]
            item_id = create_pybullet_block(
                color=color,
                half_extents=(0.03, 0.03, 0.03),
                mass=0.5,
                friction=1.0,
                physics_client_id=physics_client_id)
            item_ids.append(item_id)
        bodies["item_ids"] = item_ids
        return physics_client_id, pybullet_robot, bodies

    def _store_pybullet_bodies(self, pybullet_bodies: Dict[str, Any]) -> None:
        self._conveyor_id = pybullet_bodies["conveyor_id"]
        self._button_id = pybullet_bodies["button_id"]
        self._button_stand_id = pybullet_bodies["button_stand_id"]
        self._pusher_id = pybullet_bodies["pusher_id"]
        self._table_id = pybullet_bodies["table_id"]
        self._item_ids = pybullet_bodies["item_ids"]
        # Assign IDs to static objects
        self._conveyor.id = self._conveyor_id
        self._button.id = self._button_id
        self._button_stand.id = self._button_stand_id
        self._pusher.id = self._pusher_id
        self._table.id = self._table_id
        for item, item_id in zip(self._items, self._item_ids):
            item.id = item_id

    def _get_object_ids_for_held_check(self) -> List[int]:
        return self._item_ids

    def _set_domain_specific_state(self, state: State) -> None:
        items = state.get_objects(self._item_type)
        # Unused items go out of view
        unused_items = [item for item in self._items if item not in items]
        for i, item in enumerate(unused_items):
            p.resetBasePositionAndOrientation(
                item.id, [10.0, 10.0, i * 0.1], (0.0, 0.0, 0.0, 1.0),
                physicsClientId=self._physics_client_id)

        # Set button state
        button_pos = [self.button_stand_x, self.button_stand_y,
                      self.button_stand_z + self.button_height / 2.0]
        if state.get(self._button, "is_pressed") > 0.5:
            button_pos[2] -= 0.02
        p.resetBasePositionAndOrientation(
            self._button_id, button_pos, (0.0, 0.0, 0.0, 1.0),
            physicsClientId=self._physics_client_id)

        # Set pusher state
        pusher_pos = [state.get(self._pusher, "x"),
                      state.get(self._pusher, "y"),
                      state.get(self._pusher, "z")]
        p.resetBasePositionAndOrientation(
            self._pusher_id, pusher_pos, (0.0, 0.0, 0.0, 1.0),
            physicsClientId=self._physics_client_id)

    def _get_domain_specific_feature(self, obj: Object, feature: str) -> float:
        if obj.type == self._conveyor_type:
            if feature == "x": return self.conveyor_x
            if feature == "y": return self.conveyor_y
            if feature == "z": return self.conveyor_height / 2
            if feature == "width": return self.conveyor_width
            if feature == "length": return self.conveyor_length
            if feature == "height": return self.conveyor_height
        if obj.type == self._item_type:
            if feature == "is_held":
                return 1.0 if obj.id == self._held_obj_id else 0.0
        if obj.type == self._button_type:
            if feature == "x": return self.button_stand_x
            if feature == "y": return self.button_stand_y
            if feature == "z": return self.button_stand_z + self.button_height / 2.0
            if feature == "is_pressed":
                pos = p.getBasePositionAndOrientation(
                    self._button_id, physicsClientId=self._physics_client_id)[0]
                return 1.0 if pos[2] < self.button_stand_z + self.button_height / 2.0 - 0.001 else 0.0
        if obj.type == self._button_stand_type:
            if feature == "x": return self.button_stand_x
            if feature == "y": return self.button_stand_y
            if feature == "z": return self.button_stand_z / 2.0
        if obj.type == self._pusher_type:
            pos = p.getBasePositionAndOrientation(
                self._pusher_id, physicsClientId=self._physics_client_id)[0]
            if feature == "x": return pos[0]
            if feature == "y": return pos[1]
            if feature == "z": return pos[2]
            if feature == "width": return self.pusher_width
            if feature == "length": return self.pusher_length
            if feature == "height": return self.pusher_height
        if obj.type == self._table_type:
            if feature == "x": return self.table_x
            if feature == "y": return self.table_y
            if feature == "z": return self.table_height / 2.0
            if feature == "width": return self.table_width
            if feature == "length": return self.table_length
            if feature == "height": return self.table_height

        raise ValueError(f"Unknown feature {feature} for object {obj}")

    def _domain_specific_step(self) -> None:
        state = self._get_state()
        # Conveyor movement
        for item in self._items:
            if item not in state: continue
            if item.id == self._held_obj_id: continue
            if self._OnConveyor_holds(state, [item, self._conveyor]):
                pos, orn = p.getBasePositionAndOrientation(
                    item.id, physicsClientId=self._physics_client_id)
                new_pos = (pos[0] + 0.01, pos[1], pos[2])
                p.resetBasePositionAndOrientation(
                    item.id,
                    new_pos,
                    orn,
                    physicsClientId=self._physics_client_id)

        # Pusher movement
        is_pressed = state.get(self._button, "is_pressed") > 0.5
        pusher_pos, pusher_orn = p.getBasePositionAndOrientation(
            self._pusher_id, physicsClientId=self._physics_client_id)
        if is_pressed:
            # Move pusher towards table
            if pusher_pos[1] < self.table_y - self.table_width / 2.0:
                new_pusher_y = pusher_pos[1] + 0.01
                new_pusher_pos = (pusher_pos[0], new_pusher_y, pusher_pos[2])
                p.resetBasePositionAndOrientation(
                    self._pusher_id,
                    new_pusher_pos,
                    pusher_orn,
                    physicsClientId=self._physics_client_id)
        else:
            # Move pusher back to init
            if pusher_pos[1] > self.pusher_init_y:
                new_pusher_y = pusher_pos[1] - 0.01
                new_pusher_pos = (pusher_pos[0], new_pusher_y, pusher_pos[2])
                p.resetBasePositionAndOrientation(
                    self._pusher_id,
                    new_pusher_pos,
                    pusher_orn,
                    physicsClientId=self._physics_client_id)

    def _get_tasks(self, num_tasks: int) -> List[EnvironmentTask]:
        tasks = []
        for _ in range(num_tasks):
            # Use 5 items for each task
            items = self._items[:5]
            data: Dict[Object, Any] = {}
            data[self._conveyor] = {
                "x": self.conveyor_x,
                "y": self.conveyor_y,
                "z": self.conveyor_height / 2.0,
                "width": self.conveyor_width,
                "length": self.conveyor_length,
                "height": self.conveyor_height,
            }
            data[self._robot] = {
                "x": self.robot_init_x,
                "y": self.robot_init_y,
                "z": self.robot_init_z,
                "fingers": self.open_fingers
            }
            data[self._button] = {
                "x": self.button_stand_x,
                "y": self.button_stand_y,
                "z": self.button_stand_z + self.button_height / 2.0,
                "is_pressed": 0.0
            }
            data[self._button_stand] = {
                "x": self.button_stand_x,
                "y": self.button_stand_y,
                "z": self.button_stand_z / 2.0
            }
            data[self._pusher] = {
                "x": self.pusher_init_x,
                "y": self.pusher_init_y,
                "z": self.pusher_init_z,
                "width": self.pusher_width,
                "length": self.pusher_length,
                "height": self.pusher_height
            }
            data[self._table] = {
                "x": self.table_x,
                "y": self.table_y,
                "z": self.table_height / 2.0,
                "width": self.table_width,
                "length": self.table_length,
                "height": self.table_height
            }
            for i, item in enumerate(items):
                data[item] = {
                    "x": self.conveyor_x + 0.1 + i * 0.5 - 1.5, # TODO
                    # "x": self.conveyor_x + 0.1 + i * 0.5 - 0,
                    "y": self.conveyor_y,
                    "z": self.conveyor_height / 2 + 0.05,
                    "is_held": 0.0
                }
            state = utils.create_state_from_dict(data)
            tasks.append(EnvironmentTask(state, set()))
        return self._add_pybullet_state_to_tasks(tasks)

    def reset(self, train_or_test: str, task_idx: int, render: bool = False) -> State:
        if train_or_test == "train":
            tasks = self.get_train_tasks()
        else:
            tasks = self.get_test_tasks()
        task = tasks[task_idx]
        self._set_state(task.init)
        return self._get_state()


if __name__ == "__main__":
    # Run a simple simulation to test the environment.
    import time

    CFG.seed = 0
    CFG.env = "pybullet_airport"
    CFG.num_train_tasks = 1
    env = PyBulletAirportEnv(use_gui=True)
    _task = env.get_train_tasks()[0]
    env.reset("train", 0)

    while True:
        # Hold the robot's current joint positions.
        _act = Action(np.zeros(env.action_space.shape, dtype=np.float32))
        env.step(_act)
        time.sleep(0.01)
