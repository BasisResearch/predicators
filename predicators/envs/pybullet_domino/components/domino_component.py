"""Domino component for the domino environment.

This component handles:
- Domino blocks (start, intermediate, target, glued)
- Target objects (hinged targets)
- Pivot objects (for 180-degree turns)
- Related predicates (Toppled, Upright, Tilting, etc.)
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Optional, \
    Sequence, Set, Tuple
from typing import Type as TypingType

import numpy as np
import pybullet as p

from predicators import utils
from predicators.envs.pybullet_domino.components.base_component import \
    DominoEnvComponent
from predicators.envs.pybullet_env import create_pybullet_block
from predicators.pybullet_helpers.geometry import Pose3D, Quaternion
from predicators.pybullet_helpers.objects import create_object, update_object
from predicators.settings import CFG
from predicators.structs import Object, Predicate, State, Type

if TYPE_CHECKING:
    from predicators.envs.pybullet_domino.composed_env import \
        PyBulletDominoComposedEnv


@dataclass
class PlacementResult:
    """Result of placing a domino, target, or pivot in the sequence."""
    success: bool
    x: float
    y: float
    rotation: float
    domino_count: int
    pivot_count: int = 0
    target_count: int = 0
    just_turned_90: bool = False
    just_placed_target: bool = False


class DominoComponent(DominoEnvComponent):
    """Component for domino blocks, targets, and pivots.

    Manages the core domino mechanics including:
    - Domino blocks with different colors for roles
      (start, target, intermediate, glued)
    - Target objects that can be toppled
    - Pivot objects for 180-degree direction changes

    Note: domino_width, domino_depth, domino_height, domino_mass, and
    domino_friction are defined in PyBulletDominoComposedEnv.
    """

    # =========================================================================
    # DOMINO CONFIGURATION
    # =========================================================================

    # Domino shape properties - defined in PyBulletDominoComposedEnv
    # domino_width, domino_depth, domino_height, domino_mass, domino_friction

    # Domino thresholds
    domino_roll_threshold: ClassVar[float] = np.deg2rad(5)
    fallen_threshold: ClassVar[float] = np.pi * 2 / 5  # ~72 degrees

    # Domino colors
    start_domino_color: ClassVar[Tuple[float, float, float,
                                       float]] = (0.56, 0.93, 0.56, 1.)
    target_domino_color: ClassVar[Tuple[float, float, float,
                                        float]] = (0.85, 0.7, 0.85, 1.0)
    domino_color: ClassVar[Tuple[float, float, float,
                                 float]] = (0.6, 0.8, 1.0, 1.0)
    glued_domino_color: ClassVar[Tuple[float, float, float,
                                       float]] = (1.0, 0.0, 0.0, 1.0)
    glued_percentage: ClassVar[float] = 0.5

    # Target and pivot dimensions
    target_height: ClassVar[float] = 0.2
    pivot_width: ClassVar[float] = 0.2

    # Grid configuration - references domino_width from
    # PyBulletDominoComposedEnv
    @staticmethod
    def _get_env_class() -> TypingType["PyBulletDominoComposedEnv"]:
        """Get PyBulletDominoComposedEnv class to access shared config."""
        from predicators.envs.pybullet_domino.composed_env import \
            PyBulletDominoComposedEnv  # pylint: disable=import-outside-toplevel
        return PyBulletDominoComposedEnv

    @property
    def domino_width(self) -> float:
        """Domino width."""
        return self._get_env_class().domino_width

    @property
    def domino_depth(self) -> float:
        """Domino depth."""
        return self._get_env_class().domino_depth

    @property
    def domino_height(self) -> float:
        """Domino height."""
        return self._get_env_class().domino_height

    @property
    def domino_mass(self) -> float:
        """Domino mass."""
        return self._get_env_class().domino_mass

    @property
    def domino_friction(self) -> float:
        """Domino friction."""
        return self._get_env_class().domino_friction

    @property
    def pos_gap(self) -> float:
        """Pos gap."""
        return self._get_env_class().pos_gap

    turn_shift_frac: ClassVar[float] = 0.6
    turn_choices: ClassVar[List[str]] = ["straight", "turn90", "pivot180"]

    # Topple thresholds
    topple_angle_threshold: ClassVar[float] = 0.4

    def __init__(self,
                 num_dominos_max: int = 9,
                 num_targets_max: int = 3,
                 num_pivots_max: int = 3,
                 workspace_bounds: Optional[Dict[str, float]] = None) -> None:
        """Initialize the domino component.

        Args:
            num_dominos_max: Maximum number of domino blocks.
            num_targets_max: Maximum number of target objects.
            num_pivots_max: Maximum number of pivot objects.
            workspace_bounds: Dict with x/y/z lower/upper bounds.
        """
        super().__init__()

        self.num_dominos_max = num_dominos_max
        self.num_targets_max = num_targets_max
        self.num_pivots_max = num_pivots_max

        # Workspace bounds (will be set by composed env if not provided)
        if workspace_bounds is None:
            workspace_bounds = {
                "x_lb": 0.4,
                "x_ub": 1.1,
                "y_lb": 1.1,
                "y_ub": 1.6,
                "z_lb": 0.4,  # table_height
                "z_ub": 0.95
            }
        self.x_lb = workspace_bounds["x_lb"]
        self.x_ub = workspace_bounds["x_ub"]
        self.y_lb = workspace_bounds["y_lb"]
        self.y_ub = workspace_bounds["y_ub"]
        self.z_lb = workspace_bounds["z_lb"]
        self.z_ub = workspace_bounds["z_ub"]

        # Domino-specific placement bounds (narrower than workspace)
        # to avoid placing dominoes too close to edges
        # 1.1 + 0.07 = 1.17
        self.domino_y_lb = self.y_lb + self.domino_width
        # 1.6 - 0.21 = 1.39
        self.domino_y_ub = self.y_ub - 3 * self.domino_width
        self.domino_x_lb = self.x_lb
        self.domino_x_ub = self.x_ub

        # Create types
        self._domino_type = Type(
            "domino",
            ["x", "y", "z", "yaw", "roll", "r", "g", "b", "is_held"],
        )
        self._target_type = Type("target", ["x", "y", "z", "yaw"],
                                 sim_features=["id", "joint_id"])
        self._pivot_type = Type("pivot", ["x", "y", "z", "yaw"],
                                sim_features=["id", "joint_id"])

        # Create objects
        use_domino_as_target = CFG.domino_use_domino_blocks_as_target
        if use_domino_as_target:
            num_dominos = self.num_dominos_max + self.num_targets_max
            num_targets = 0
        else:
            num_dominos = self.num_dominos_max
            num_targets = self.num_targets_max

        self.dominos: List[Object] = []
        for i in range(num_dominos):
            obj = Object(f"domino_{i}", self._domino_type)
            self.dominos.append(obj)

        self.targets: List[Object] = []
        for i in range(num_targets):
            obj = Object(f"target_{i}", self._target_type)
            self.targets.append(obj)

        self.pivots: List[Object] = []
        for i in range(self.num_pivots_max):
            obj = Object(f"pivot_{i}", self._pivot_type)
            self.pivots.append(obj)

        # Constraint tracking for connected dominoes
        self.block_constraints: List[int] = []
        self.fixed_domino_ids: List[int] = []

        # Create predicates
        self._create_predicates()

    def _create_predicates(self) -> None:
        """Create all predicates for this component."""
        if CFG.domino_use_domino_blocks_as_target:
            self._Toppled = Predicate("Toppled", [self._domino_type],
                                      self._Toppled_holds)
        else:
            self._Toppled = Predicate("Toppled", [self._target_type],
                                      self._Toppled_holds)

        self._Upright = Predicate("Upright", [self._domino_type],
                                  self._Upright_holds)
        self._Tilting = Predicate("Tilting", [self._domino_type],
                                  self._Tilting_holds)
        self._InitialBlock = Predicate("InitialBlock", [self._domino_type],
                                       self._StartBlock_holds)
        self._MovableBlock = Predicate("MovableBlock", [self._domino_type],
                                       self._MovableBlock_holds)
        self._DominoNotGlued = Predicate("DominoNotGlued", [self._domino_type],
                                         self._DominoNotGlued_holds)

    # -------------------------------------------------------------------------
    # DominoEnvComponent interface implementation
    # -------------------------------------------------------------------------

    def get_types(self) -> Set[Type]:
        types = {self._domino_type}
        if self.targets:
            types.add(self._target_type)
        if self.pivots:
            types.add(self._pivot_type)
        return types

    def get_predicates(self) -> Set[Predicate]:
        preds = {
            self._Toppled,
            self._Upright,
            self._Tilting,
            self._InitialBlock,
            self._MovableBlock,
        }
        if CFG.domino_has_glued_dominos:
            preds.add(self._DominoNotGlued)
        return preds

    def get_goal_predicates(self) -> Set[Predicate]:
        return {self._Toppled}

    def get_objects(self) -> List[Object]:
        return self.dominos + self.targets + self.pivots

    def initialize_pybullet(self, physics_client_id: int) -> Dict[str, Any]:
        """Create PyBullet bodies for dominoes, targets, and pivots."""
        self._physics_client_id = physics_client_id
        bodies: Dict[str, Any] = {}

        # Create dominoes
        domino_ids = []
        num_dominos_to_create = len(self.dominos)
        for i in range(num_dominos_to_create):
            domino_id = create_domino_block(
                color=self.start_domino_color if i == 0 else self.domino_color,
                half_extents=(self.domino_width / 2, self.domino_depth / 2,
                              self.domino_height / 2),
                mass=self.domino_mass,
                friction=self.domino_friction,
                orientation=(0.0, 0.0, 0.0, 1.0),
                physics_client_id=physics_client_id,
                add_top_triangle=True,
            )
            domino_ids.append(domino_id)
        bodies["domino_ids"] = domino_ids

        # Create targets
        target_ids = []
        for _ in self.targets:
            tid = create_object("urdf/domino_target.urdf",
                                position=(self.x_lb, self.y_lb, self.z_lb),
                                orientation=p.getQuaternionFromEuler(
                                    [0.0, 0.0, 0.0]),
                                scale=1.0,
                                use_fixed_base=True,
                                physics_client_id=physics_client_id)
            target_ids.append(tid)
        bodies["target_ids"] = target_ids

        # Create pivots
        pivot_ids = []
        for _ in self.pivots:
            pid = create_object("urdf/domino_pivot.urdf",
                                position=(self.x_lb, self.y_lb, self.z_lb),
                                orientation=p.getQuaternionFromEuler(
                                    [0.0, 0.0, 0.0]),
                                scale=1.0,
                                use_fixed_base=True,
                                physics_client_id=physics_client_id)
            pivot_ids.append(pid)
        bodies["pivot_ids"] = pivot_ids

        return bodies

    def store_pybullet_bodies(self, pybullet_bodies: Dict[str, Any]) -> None:
        """Store PyBullet body IDs on objects."""
        for domino, id_ in zip(self.dominos, pybullet_bodies["domino_ids"]):
            domino.id = id_

        for target, id_ in zip(self.targets, pybullet_bodies["target_ids"]):
            target.id = id_
            assert self._physics_client_id is not None
            target.joint_id = self._get_joint_id(id_, "flap_hinge_joint",
                                                 self._physics_client_id)

        for pivot, id_ in zip(self.pivots, pybullet_bodies["pivot_ids"]):
            pivot.id = id_
            assert self._physics_client_id is not None
            pivot.joint_id = self._get_joint_id(id_, "flap_hinge_joint",
                                                self._physics_client_id)

    def reset_state(self, state: State) -> None:
        """Reset dominoes, targets, and pivots to match state."""
        assert self._physics_client_id is not None
        domino_objs = state.get_objects(self._domino_type)

        # Remove old constraints
        for constraint in self.block_constraints:
            p.removeConstraint(constraint,
                               physicsClientId=self._physics_client_id)
        self.block_constraints = []

        # Restore normal dynamics to previously fixed dominoes
        for domino_id in self.fixed_domino_ids:
            p.changeDynamics(domino_id,
                             -1,
                             mass=self.domino_mass,
                             physicsClientId=self._physics_client_id)
        self.fixed_domino_ids = []

        # Update domino colors to match state
        for domino in domino_objs:
            if domino.id is not None:
                r = state.get(domino, "r")
                g = state.get(domino, "g")
                b = state.get(domino, "b")
                update_object(domino.id,
                              color=(r, g, b, 1.0),
                              physics_client_id=self._physics_client_id)

        # Move unused dominoes out of view
        oov_x, oov_y = self.out_of_view_xy
        for i in range(len(domino_objs), len(self.dominos)):
            oov_x += 0.1
            oov_y += 0.1
            update_object(self.dominos[i].id,
                          position=(oov_x, oov_y, self.domino_height / 2),
                          physics_client_id=self._physics_client_id)

        # Reset targets
        target_objs = state.get_objects(self._target_type)
        for target_obj in target_objs:
            self._set_flat_rotation(target_obj, 0.0)
        for i in range(len(target_objs), len(self.targets)):
            oov_x += 0.1
            oov_y += 0.1
            update_object(self.targets[i].id,
                          position=(oov_x, oov_y, self.domino_height / 2),
                          physics_client_id=self._physics_client_id)

        # Reset pivots
        pivot_objs = state.get_objects(self._pivot_type)
        for pivot_obj in pivot_objs:
            self._set_flat_rotation(pivot_obj, 0.0)
        for i in range(len(pivot_objs), len(self.pivots)):
            oov_x += 0.1
            oov_y += 0.1
            update_object(self.pivots[i].id,
                          position=(oov_x, oov_y, self.domino_height / 2),
                          physics_client_id=self._physics_client_id)

        # Handle glued dominoes
        if CFG.domino_has_glued_dominos:
            for domino in domino_objs:
                if domino.id is not None:
                    if self._DominoGlued_holds(state, [domino]):
                        p.changeDynamics(
                            domino.id,
                            -1,
                            mass=1e10,
                            physicsClientId=self._physics_client_id)
                        self.fixed_domino_ids.append(domino.id)

    def extract_feature(self, obj: Object, feature: str) -> Optional[float]:
        """Extract feature for domino-related objects."""
        # Let the base environment handle position/orientation extraction
        return None

    def get_object_ids_for_held_check(self) -> List[int]:
        """Return domino and pivot IDs for held checking."""
        domino_ids = [d.id for d in self.dominos if d.id is not None]
        pivot_ids = [p.id for p in self.pivots if p.id is not None]
        return domino_ids + pivot_ids

    # -------------------------------------------------------------------------
    # Predicate hold functions
    # -------------------------------------------------------------------------

    def _Toppled_holds(self, state: State, objects: Sequence[Object]) -> bool:
        """Check if target/domino is toppled."""
        obj, = objects
        if CFG.domino_use_domino_blocks_as_target:
            roll_angle = abs(state.get(obj, "roll"))
            return roll_angle >= self.fallen_threshold
        rot_z = state.get(obj, "yaw")
        return abs(utils.wrap_angle(rot_z)) < 0.8

    def _Upright_holds(self, state: State, objects: Sequence[Object]) -> bool:
        """Check if domino is upright."""
        obj, = objects
        tilt_angle = state.get(obj, "roll")
        return abs(tilt_angle) < self.domino_roll_threshold

    def _Tilting_holds(self, state: State, objects: Sequence[Object]) -> bool:
        """Check if domino is tilting (in transition)."""
        obj, = objects
        roll_angle = abs(state.get(obj, "roll"))
        return self.domino_roll_threshold <= roll_angle < self.fallen_threshold

    @classmethod
    def _StartBlock_holds(cls, state: State,
                          objects: Sequence[Object]) -> bool:
        """Check if domino is the start block (light green)."""
        domino, = objects
        eps = 1e-3
        return (
            abs(state.get(domino, "r") - cls.start_domino_color[0]) < eps
            and abs(state.get(domino, "g") - cls.start_domino_color[1]) < eps
            and abs(state.get(domino, "b") - cls.start_domino_color[2]) < eps)

    @classmethod
    def _MovableBlock_holds(cls, state: State,
                            objects: Sequence[Object]) -> bool:
        """Check if domino is a movable block (blue)."""
        domino, = objects
        eps = 1e-3
        return (abs(state.get(domino, "r") - cls.domino_color[0]) < eps
                and abs(state.get(domino, "g") - cls.domino_color[1]) < eps
                and abs(state.get(domino, "b") - cls.domino_color[2]) < eps)

    @classmethod
    def _TargetDomino_holds(cls, state: State,
                            objects: Sequence[Object]) -> bool:
        """Check if domino is a target (pink or glued red)."""
        domino, = objects
        eps = 1e-3
        return (cls._DominoGlued_holds(state, objects)) or (
            abs(state.get(domino, "r") - cls.target_domino_color[0]) < eps
            and abs(state.get(domino, "g") - cls.target_domino_color[1]) < eps
            and abs(state.get(domino, "b") - cls.target_domino_color[2]) < eps)

    @classmethod
    def _DominoNotGlued_holds(cls, state: State,
                              objects: Sequence[Object]) -> bool:
        """Check if domino is NOT glued."""
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

    # -------------------------------------------------------------------------
    # Helper methods
    # -------------------------------------------------------------------------

    @staticmethod
    def _get_joint_id(obj_id: int,
                      joint_name: str,
                      physics_client_id: int = 0) -> int:
        """Get joint ID by name from PyBullet object."""
        num_joints = p.getNumJoints(obj_id, physicsClientId=physics_client_id)
        for j in range(num_joints):
            info = p.getJointInfo(obj_id, j, physicsClientId=physics_client_id)
            if info[1].decode("utf-8") == joint_name:
                return j
        return -1

    def _set_flat_rotation(self, flap_obj: Object, rot: float = 0.0) -> None:
        """Set rotation of a hinged object (target/pivot)."""
        p.resetJointState(flap_obj.id,
                          flap_obj.joint_id,
                          rot,
                          physicsClientId=self._physics_client_id)

    # -------------------------------------------------------------------------
    # Sequence generation helpers
    # -------------------------------------------------------------------------

    def place_domino(self,
                     _domino_idx: int,
                     x: float,
                     y: float,
                     rot: float,
                     is_start_block: bool = False,
                     is_target_block: bool = False,
                     rng: Optional[np.random.Generator] = None,
                     task_idx: Optional[int] = None) -> Dict:
        """Create a dictionary with placement parameters for a domino."""
        if is_start_block:
            color = self.start_domino_color
        elif is_target_block:
            should_be_glued = False
            if CFG.domino_has_glued_dominos:
                if task_idx == 0:
                    should_be_glued = True
                elif task_idx == 1:
                    should_be_glued = False
                else:
                    should_be_glued = (rng is not None and
                                       rng.random() < self.glued_percentage)
            color = (self.glued_domino_color
                     if should_be_glued else self.target_domino_color)
        else:
            color = self.domino_color

        return {
            "x": x,
            "y": y,
            "z": self.z_lb + self.domino_height / 2,
            "yaw": rot,
            "roll": 0.0,
            "r": color[0],
            "g": color[1],
            "b": color[2],
            "is_held": 0.0,
        }

    def place_pivot_or_target(self,
                              x: float,
                              y: float,
                              rot: float = 0.0) -> Dict:
        """Create a dictionary with placement parameters for a pivot/target."""
        return {
            "x": x,
            "y": y,
            "z": self.z_lb,
            "yaw": rot,
        }

    # -------------------------------------------------------------------------
    # Public properties for type access
    # -------------------------------------------------------------------------

    @property
    def domino_type(self) -> Type:
        """Domino type."""
        return self._domino_type

    @property
    def target_type(self) -> Type:
        """Target type."""
        return self._target_type

    @property
    def pivot_type(self) -> Type:
        """Pivot type."""
        return self._pivot_type

    @property
    def Toppled(self) -> Predicate:
        """Toppled."""
        return self._Toppled


def create_domino_block(
    color: Tuple[float, float, float, float],
    half_extents: Tuple[float, float, float],
    mass: float,
    friction: float,
    position: Pose3D = (0.0, 0.0, 0.0),
    orientation: Quaternion = (0.0, 0.0, 0.0, 1.0),
    physics_client_id: int = 0,
    add_top_triangle: bool = False,
    *,
    restitution: float = 0.02,
    rolling_friction: float = 0.006,
    spinning_friction: Optional[float] = None,
    linear_damping: float = 0.0,
    angular_damping: float = 0.03,
    friction_anchor: bool = True,
    ccd: bool = True,
    ccd_swept_radius: Optional[float] = None,
) -> int:
    """Create a domino-tuned block with appropriate physics settings."""
    block_id = create_pybullet_block(
        color=color,
        half_extents=half_extents,
        mass=mass,
        friction=friction,
        position=position,
        orientation=orientation,
        physics_client_id=physics_client_id,
        add_top_triangle=add_top_triangle,
    )

    if spinning_friction is None:
        spinning_friction = friction

    p.changeDynamics(
        block_id,
        linkIndex=-1,
        lateralFriction=friction,
        rollingFriction=rolling_friction,
        spinningFriction=spinning_friction,
        restitution=restitution,
        linearDamping=linear_damping,
        angularDamping=angular_damping,
        frictionAnchor=friction_anchor,
        physicsClientId=physics_client_id,
    )

    if ccd:
        m = min(half_extents)
        swept = ccd_swept_radius if ccd_swept_radius is not None else 0.5 * m
        p.changeDynamics(
            block_id,
            linkIndex=-1,
            ccdSweptSphereRadius=swept,
            physicsClientId=physics_client_id,
        )

    return block_id
