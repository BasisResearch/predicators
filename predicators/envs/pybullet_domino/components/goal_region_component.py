"""Goal region for the wind-placement task.

A flat patch on the table the wind has to deliver a domino INTO. It is
scenery, not an obstacle: zero mass, collisions disabled, drawn only so
a person watching the video can see what the robot is aiming at.

The region exists so the task cannot be solved by pushing as hard as
possible. Emily's question in the design meeting was whether a robot
could simply place the block as close to the fan as it can every time;
with a target POINT it could, and with a bounded region it cannot -
place too near the fan and the wind carries the block past the far
edge, too far and it never arrives. Only a band of placements works,
and finding that band means knowing how far this wind pushes this
block, which is the parameter the task exists to make learnable.
"""

from typing import Any, ClassVar, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pybullet as p

from predicators.envs.pybullet_domino.components.base_component import \
    DominoEnvComponent
from predicators.pybullet_helpers.objects import create_pybullet_block
from predicators.structs import Object, Predicate, State, Type


class GoalRegionComponent(DominoEnvComponent):
    """One rectangular goal patch, and the predicate for being in it."""

    # A patch big enough to be hittable and small enough that "as hard
    # as possible" overshoots it. 6 cm of slack along the wind axis
    # against a slide of tens of centimetres.
    # Half-width along the wind axis. 4 cm against an 11.6 cm slide
    # means the robot has to know how far this gust carries the block to
    # within about a quarter - loose enough to be learnable from a
    # handful of episodes, tight enough that a coarse guess misses.
    region_half_x: ClassVar[float] = 0.04
    region_half_y: ClassVar[float] = 0.09
    region_thickness: ClassVar[float] = 0.001
    region_color: ClassVar[Tuple[float, float, float,
                                 float]] = (0.2, 0.85, 0.35, 0.55)

    def __init__(self,
                 workspace_bounds: Optional[Dict[str, float]] = None,
                 table_height: float = 0.4,
                 domino_type: Optional[Type] = None) -> None:
        super().__init__()
        self.table_height = table_height
        self._domino_type = domino_type
        if workspace_bounds is None:
            workspace_bounds = {
                "x_lb": 0.4,
                "x_ub": 1.1,
                "y_lb": 1.1,
                "y_ub": 1.6
            }
        self.x_lb = workspace_bounds["x_lb"]
        self.x_ub = workspace_bounds["x_ub"]
        self.y_lb = workspace_bounds["y_lb"]
        self.y_ub = workspace_bounds["y_ub"]

        # half_x / half_y are features rather than constants so an agent
        # reading the state can see how much slack it has, and so a task
        # generator could vary the difficulty without a code change.
        self._region_type = Type(
            "region", ["x", "y", "z", "half_x", "half_y"],
            sim_features=["id"])
        self._region = Object("goal_region", self._region_type)
        self._region_id: Optional[int] = None
        self._region_xy: Tuple[float, float] = (0.0, 0.0)

        self._InGoal = Predicate("InGoal",
                                 [self._domino_type, self._region_type]
                                 if self._domino_type is not None else
                                 [self._region_type], self._InGoal_holds)

    # -- component interface ------------------------------------------

    def get_types(self) -> Set[Type]:
        return {self._region_type}

    def get_predicates(self) -> Set[Predicate]:
        return {self._InGoal}

    def get_goal_predicates(self) -> Set[Predicate]:
        return {self._InGoal}

    def get_objects(self) -> List[Object]:
        return [self._region]

    def initialize_pybullet(self, physics_client_id: int) -> Dict[str, Any]:
        """A thin coloured plate lying on the table.

        Mass 0 and collisions off: the region must not deflect the very
        block it is measuring, and a lip of even a millimetre would.
        """
        self._physics_client_id = physics_client_id
        region_id = create_pybullet_block(
            color=self.region_color,
            half_extents=(self.region_half_x, self.region_half_y,
                          self.region_thickness),
            mass=0.0,
            friction=0.5,
            position=(0.0, 0.0, self.table_height + self.region_thickness),
            orientation=(0.0, 0.0, 0.0, 1.0),
            physics_client_id=physics_client_id)
        p.setCollisionFilterGroupMask(region_id,
                                      -1,
                                      0,
                                      0,
                                      physicsClientId=physics_client_id)
        return {"region_id": region_id}

    def store_pybullet_bodies(self, pybullet_bodies: Dict[str, Any]) -> None:
        self._region_id = pybullet_bodies["region_id"]
        self._region.id = self._region_id

    def reset_state(self, state: State) -> None:
        x = float(state.get(self._region, "x"))
        y = float(state.get(self._region, "y"))
        self._region_xy = (x, y)
        if self._region_id is not None:
            p.resetBasePositionAndOrientation(
                self._region_id,
                (x, y, self.table_height + self.region_thickness),
                (0.0, 0.0, 0.0, 1.0),
                physicsClientId=self._physics_client_id)

    def extract_feature(self, obj: Object, feature: str) -> Optional[float]:
        if obj.type != self._region_type:
            return None
        if feature == "x":
            return self._region_xy[0]
        if feature == "y":
            return self._region_xy[1]
        if feature == "z":
            return self.table_height + self.region_thickness
        if feature == "half_x":
            return self.region_half_x
        if feature == "half_y":
            return self.region_half_y
        return None

    def get_init_dict_entries(
            self, rng: np.random.Generator) -> Dict[Object, Dict[str, Any]]:
        """Placed by the task generator, which knows where the fan is."""
        del rng
        x, y = self._region_xy
        return {
            self._region: {
                "x": x,
                "y": y,
                "z": self.table_height + self.region_thickness,
                "half_x": self.region_half_x,
                "half_y": self.region_half_y,
            }
        }

    def get_object_ids_for_held_check(self) -> List[int]:
        return []

    # -- placement, used by the task generator ------------------------

    def set_region_xy(self, x: float, y: float) -> None:
        """Put the region where the generator decided it goes."""
        self._region_xy = (x, y)

    # -- predicate ----------------------------------------------------

    def _InGoal_holds(self, state: State, objects: Sequence[Object]) -> bool:
        """The domino lies FLAT inside the patch.

        Flat, not merely present, and that word is what makes the task
        a task. A robot cannot place a domino on its side - Place sets
        blocks upright - so a block lying in the patch can only have
        been put there by the wind. Without it the goal has a trivial
        answer: pick the block up, put it down in the region, done, and
        the fan is never needed at all.

        Centre rather than footprint overlap: a block half in and half
        out is a coin toss on the exact contact solve, and the reward
        should not turn on which millimetre the solver settled at.
        """
        if len(objects) != 2:
            return False
        domino, region = objects
        if state.get(domino, "is_held") > 0.5:
            return False
        # Roll is meaningful modulo pi: a box turned 180 degrees about
        # its own width axis is the same box.
        # pylint: disable-next=import-outside-toplevel
        from predicators.envs.pybullet_domino.components.domino_component \
            import DominoComponent
        roll = float(state.get(domino, "roll"))
        roll = (roll + np.pi / 2) % np.pi - np.pi / 2
        if abs(roll) < DominoComponent.domino_roll_threshold:
            return False
        dx = abs(float(state.get(domino, "x")) - float(state.get(region, "x")))
        dy = abs(float(state.get(domino, "y")) - float(state.get(region, "y")))
        return (dx <= float(state.get(region, "half_x"))
                and dy <= float(state.get(region, "half_y")))

    @property
    def region(self) -> Object:
        """The goal patch object."""
        return self._region

    @property
    def region_type(self) -> Type:
        """Type of the goal patch."""
        return self._region_type

    @property
    def InGoal(self) -> Predicate:
        """True when a domino's centre is inside the patch."""
        return self._InGoal
