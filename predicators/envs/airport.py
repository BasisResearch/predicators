"""Airport environment with a conveyor belt."""

from typing import List, Sequence, Set

from predicators.envs import BaseEnv
from predicators.structs import Action, EnvironmentTask, Object, Predicate, \
    State, Type


class AirportEnv(BaseEnv):
    """Airport environment with a conveyor belt."""

    def __init__(self, use_gui: bool = False) -> None:
        super().__init__(use_gui)

        # Types
        self._item_type = Type("item", ["x", "y", "z", "is_held"])
        self._conveyor_type = Type("conveyor",
                                   ["x", "y", "z", "width", "length", "height"])
        self._robot_type = Type("robot", ["x", "y", "z", "fingers"])
        self._button_type = Type("button", ["x", "y", "z", "is_pressed"])
        self._button_stand_type = Type("button_stand", ["x", "y", "z"])
        self._pusher_type = Type("pusher", ["x", "y", "z", "width", "length", "height"])
        self._table_type = Type("table", ["x", "y", "z", "width", "length", "height"])

        # Predicates
        self._OnConveyor = Predicate("OnConveyor",
                                     [self._item_type, self._conveyor_type],
                                     self._OnConveyor_holds)
        self._Pressed = Predicate("Pressed", [self._button_type],
                                  self._Pressed_holds)

        # Static objects
        self._robot = Object("robot", self._robot_type)
        self._conveyor = Object("conveyor", self._conveyor_type)
        self._items = [Object(f"item_{i}", self._item_type) for i in range(5)]
        self._button = Object("button", self._button_type)
        self._button_stand = Object("button_stand", self._button_stand_type)
        self._pusher = Object("pusher", self._pusher_type)
        self._table = Object("table", self._table_type)

    def _OnConveyor_holds(self, state: State, objects: Sequence[Object]) -> bool:
        item, conveyor = objects
        ix = state.get(item, "x")
        iy = state.get(item, "y")
        iz = state.get(item, "z")
        cx = state.get(conveyor, "x")
        cy = state.get(conveyor, "y")
        cz = state.get(conveyor, "z")
        cw = state.get(conveyor, "width")
        cl = state.get(conveyor, "length")
        ch = state.get(conveyor, "height")
        # Simple bounding box check
        return (cx  - cl / 2 <= ix <= cx + cl / 2 and cy - cw / 2 <= iy <= cy + cw / 2
                and cz + ch / 2 <= iz <= cz + 0.1 + ch / 2)

    def _Pressed_holds(self, state: State, objects: Sequence[Object]) -> bool:
        button, = objects
        return state.get(button, "is_pressed") > 0.5

    @classmethod
    def get_name(cls) -> str:
        return "airport"

    def step(self, action: Action) -> State:
        raise NotImplementedError("Override in PyBulletAirportEnv")

    def reset(self, train_or_test: str, task_idx: int) -> State:
        raise NotImplementedError("Override in PyBulletAirportEnv")

    def _generate_train_tasks(self) -> List[EnvironmentTask]:
        return self._get_tasks(5)

    def _generate_test_tasks(self) -> List[EnvironmentTask]:
        return self._get_tasks(5)

    def _get_tasks(self, num_tasks: int) -> List[EnvironmentTask]:
        # This will be overridden or called by PyBulletAirportEnv
        return []

    @property
    def types(self) -> Set[Type]:
        return {
            self._item_type, self._conveyor_type, self._robot_type,
            self._button_type, self._button_stand_type, self._pusher_type,
            self._table_type
        }

    @property
    def predicates(self) -> Set[Predicate]:
        return {self._OnConveyor, self._OnTable, self._Pressed}

    @property
    def goal_predicates(self) -> Set[Predicate]:
        return {self._OnConveyor, self._OnTable, self._Pressed}
