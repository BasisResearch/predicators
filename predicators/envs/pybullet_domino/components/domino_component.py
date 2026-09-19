"""Domino component for the domino environment.

The component's observable simulation core (types, objects, PyBullet
bodies, state reset, physical-parameter override) lives in
:mod:`predicators.envs.pybullet_domino.components.domino_bodies`, which
may be surfaced to learning agents as reference source. This module
holds what an agent must learn or must not see:

- the per-role body masses asserted at every reset (glued dominoes,
  gray blocks);
- the predicates (Toppled, Upright, Tilting, InFront, ...) and the
  thresholds that define them, which carry the goal semantics;
- the placement bounds and layout helpers the task generators use.
"""

from dataclasses import dataclass
from typing import ClassVar, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pybullet as p

from predicators import utils
from predicators.envs.pybullet_domino.components.domino_bodies import \
    DominoBodiesComponent, create_domino_block
from predicators.settings import CFG
from predicators.structs import Object, Predicate, State

__all__ = ["DominoComponent", "PlacementResult", "create_domino_block"]


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
    # Yaw to place the *next* block at. Tracks the smooth 45-deg-per-turn
    # increment, which after a turn differs from ``rotation`` (the travel
    # direction used to lay out positions) by 180 deg — same physical box,
    # but the increment representation keeps a straight run reading as one
    # constant yaw instead of flipping. ``None`` means "same as rotation"
    # (no turn has happened yet).
    block_yaw: Optional[float] = None


class DominoComponent(DominoBodiesComponent):
    """Component for domino blocks, targets, and pivots.

    Manages the core domino mechanics including:
    - Domino blocks with different colors for roles
      (start, target, intermediate, glued)
    - Target objects that can be toppled
    - Pivot objects for 180-degree direction changes

    The bodies themselves are built and reset by
    ``DominoBodiesComponent``; this class adds the role masses,
    predicates and layout helpers.
    """

    # =========================================================================
    # DOMINO CONFIGURATION
    # =========================================================================

    # Domino thresholds
    domino_roll_threshold: ClassVar[float] = np.deg2rad(5)
    # A free-standing domino tips over past atan(depth/height) ~= 5.7 deg:
    # beyond that its center of mass is past the pivot edge and gravity
    # torque topples it, so an unheld lean past ~10 deg is either mid-fall
    # (committed) or propped on another body - both mean the domino was
    # genuinely knocked over. This counts propped "leaners" (e.g. a target
    # coming to rest at ~20 deg against a still-standing neighbor) that a
    # stricter criterion would miss. Recorded runs show unheld rolls are
    # bimodal (< 3 deg placement jitter or > 79 deg full topples), so the
    # 10 deg line sits in a wide empty band.
    fallen_threshold: ClassVar[float] = np.deg2rad(10)

    # Domino colors (start/target/ordinary colors are on
    # DominoBodiesComponent)
    glued_domino_color: ClassVar[Tuple[float, float, float,
                                       float]] = (1.0, 0.0, 0.0, 1.0)
    glued_percentage: ClassVar[float] = 0.5
    # Heavy (immovable-obstacle) blocks: domino-shaped, gray. Their TRUE
    # mass makes them untopple-able/unmovable; planning sims can believe a
    # different (normal) mass via the ``block_mass`` physical-param
    # override, which is what the heavy-block tasks exploit.
    heavy_block_color: ClassVar[Tuple[float, float, float,
                                      float]] = (0.35, 0.35, 0.35, 1.0)
    heavy_block_true_mass: ClassVar[float] = 1000.0

    # Target and pivot dimensions
    target_height: ClassVar[float] = 0.2
    pivot_width: ClassVar[float] = 0.2

    turn_shift_frac: ClassVar[float] = 0.6
    turn_choices: ClassVar[List[str]] = ["straight", "turn90", "pivot180"]

    # Topple thresholds
    topple_angle_threshold: ClassVar[float] = 0.4

    def __init__(self,
                 num_dominos_max: int = 9,
                 num_targets_max: int = 3,
                 num_pivots_max: int = 3,
                 workspace_bounds: Optional[Dict[str, float]] = None,
                 domino_width: Optional[float] = None,
                 domino_depth: Optional[float] = None,
                 domino_height: Optional[float] = None) -> None:
        """Initialize the domino component.

        Args:
            num_dominos_max: Maximum number of domino blocks.
            num_targets_max: Maximum number of target objects.
            num_pivots_max: Maximum number of pivot objects.
            workspace_bounds: Dict with x/y/z lower/upper bounds.
            domino_width/depth/height: per-component dimension overrides (m).
                None (default) falls back to the shared
                PyBulletDominoBaseEnv ClassVars.
        """
        super().__init__(num_dominos_max=num_dominos_max,
                         num_targets_max=num_targets_max,
                         num_pivots_max=num_pivots_max,
                         workspace_bounds=workspace_bounds,
                         domino_width=domino_width,
                         domino_depth=domino_depth,
                         domino_height=domino_height)

        # Domino-specific placement bounds (narrower than workspace) to avoid
        # placing dominoes too close to edges. The lower (robot-side) margin is
        # 1.5x the width: keeping the start block farther from the near edge
        # makes it reliably reachable for the push, which lifts the oracle
        # push-only solve rate from ~92% to ~99% (the misses were robot
        # reach/push failures, not cascade stalls) while keeping task diversity.
        # 1.1 + 1.5 * 0.07 = 1.205
        self.domino_y_lb = self.y_lb + 1.5 * self.domino_width
        # 1.6 - 0.21 = 1.39
        self.domino_y_ub = self.y_ub - 3 * self.domino_width
        self.domino_x_lb = self.x_lb
        self.domino_x_ub = self.x_ub

        # Heavy-block mode: the LAST slot of the shared body pool is the
        # gray block, minted as its own ``block``-typed object (the body
        # and all slot-indexed machinery are unchanged; only the object
        # identity differs).
        self.blocks = []
        if CFG.domino_heavy_block_tasks and self.dominos:
            block_obj = Object("block_0", self._block_type)
            self.dominos[-1] = block_obj
            self.blocks.append(block_obj)

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
        # Position-based InFront over continuous domino poses. When the grid is
        # in use, GridComponent's derived InFront replaces this one (helper
        # predicates take precedence on name collisions).
        self._InFront = Predicate(
            "InFront", [self._domino_type, self._domino_type],
            self._InFront_holds,
            natural_language_assertion=lambda os:
            ("the two dominoes are chain-adjacent: one sits one spacing-gap "
             "ahead of the other along that other's facing (toppling) "
             "direction -- straight or bent 45 degrees left/right for a turn, "
             "in both placement direction and yaw -- so that toppling the "
             "back domino knocks the front one over"))

    # -------------------------------------------------------------------------
    # DominoEnvComponent interface implementation
    # -------------------------------------------------------------------------

    def get_predicates(self) -> Set[Predicate]:
        preds = {
            self._Toppled,
            self._Upright,
            self._Tilting,
            self._InitialBlock,
            self._MovableBlock,
            self._InFront,
        }
        if CFG.domino_has_glued_dominos:
            preds.add(self._DominoNotGlued)
        return preds

    def get_goal_predicates(self) -> Set[Predicate]:
        return {self._Toppled}

    def _assign_body_masses(self, state: State,
                            domino_objs: Sequence[Object]) -> None:
        """Pin the per-role masses the true physics assigns at reset."""
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

        # Handle heavy (gray) blocks: true physics makes them untopple-able.
        # The believed mass, if any, is re-asserted by the override afterwards.
        for domino in domino_objs:
            if domino.id is not None and self._HeavyBlock_holds(
                    state, [domino]):
                p.changeDynamics(domino.id,
                                 -1,
                                 mass=self.heavy_block_true_mass,
                                 physicsClientId=self._physics_client_id)
                self.block_body_ids.append(domino.id)

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
    def _HeavyBlock_holds(cls, state: State,
                          objects: Sequence[Object]) -> bool:
        """Check if domino is a heavy (immovable, gray) block."""
        domino, = objects
        return cls.is_heavy_color(state.get(domino,
                                            "r"), state.get(domino, "g"),
                                  state.get(domino, "b"))

    @classmethod
    def is_heavy_color(cls, r: float, g: float, b: float) -> bool:
        """Whether an (r, g, b) triple is the heavy-block gray."""
        eps = 1e-3
        return (abs(r - cls.heavy_block_color[0]) < eps
                and abs(g - cls.heavy_block_color[1]) < eps
                and abs(b - cls.heavy_block_color[2]) < eps)

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

    def _InFront_holds(self, state: State, objects: Sequence[Object]) -> bool:
        """Position-based ``InFront`` classifier over continuous poses.

        ``InFront(d1, d2)`` holds when one domino sits roughly one
        ``pos_gap`` ahead of the other along that other's facing
        (toppling) direction, with a discrete turn offset between their
        yaws (straight / 45-left / 45-right). It reads the continuous
        domino poses directly, so it is available to grid-free agent
        approaches.
        """
        domino1, domino2 = objects
        if state.get(domino1, "is_held") or state.get(domino2, "is_held"):
            return False

        pos_gap = self.pos_gap
        pos_tol = pos_gap * 0.3
        ang_tol = np.radians(15)
        # Cardinal-facing slack for the reference (back) domino. A domino
        # the robot re-places settles ~1 deg off cardinal, so a 1e-3 rad
        # (~0.06 deg) gate makes InFront(front, placed_back) unsatisfiable
        # for chained placements; allow a few degrees of slack instead.
        card_thresh = float(np.sin(np.radians(10)))
        # Straight, 45-degree right turn, and 45-degree left turn.
        turn_offsets = (-np.pi / 4, 0.0, np.pi / 4)

        def _ahead(back: Object, front: Object) -> bool:
            x_b = state.get(back, "x")
            y_b = state.get(back, "y")
            rot_b = state.get(back, "yaw")
            # The relationship only holds for (roughly) cardinal back-facings.
            if not (abs(np.sin(rot_b)) < card_thresh
                    or abs(np.cos(rot_b)) < card_thresh):
                return False
            # The front domino's yaw differs from the back's by a discrete
            # turn offset (straight / +-45 deg).
            diff = utils.wrap_angle(state.get(front, "yaw") - rot_b)
            if not any(abs(diff - off) < ang_tol for off in turn_offsets):
                return False
            # The front domino sits one pos_gap from the back, along the
            # back's facing -- which may itself be rotated by a turn offset,
            # so the chain can bend through a turn (the next block then lies
            # diagonally off the back rather than straight ahead).
            fx = state.get(front, "x")
            fy = state.get(front, "y")
            # A domino is 180-degree symmetric, so its facing names a
            # bidirectional topple axis: the front may sit one gap along
            # either end of that (possibly turn-rotated) axis.
            #
            # A turn-completing block always carries a half-width lateral
            # ("side") offset, applied orthogonal to the reference's facing
            # by the task generator (see DominoTaskGenerator.
            # _place_turn90_domino) so the toppling chain stays overlapping
            # through the corner. A turn placement (dir_off != 0) therefore
            # sits at +-side_offset along the perpendicular -- NOT on the bare
            # axis. Excluding lateral 0 here is what lets the Place sampler
            # distinguish the cascade-enabling offset pose from the
            # symbolically-equivalent-but-physically-dead on-axis pose (an
            # on-axis turn block fails this edge, so scoring prefers the
            # offset). Straight placements (dir_off == 0) stay exactly on the
            # axis, so no spurious edges appear.
            side_offset = self.domino_width / 2
            perp_x = np.cos(rot_b)
            perp_y = -np.sin(rot_b)
            for dir_off in turn_offsets:
                ang = rot_b + dir_off
                laterals = ((0.0, ) if abs(dir_off) < 1e-9 else
                            (side_offset, -side_offset))
                for sgn in (1.0, -1.0):
                    base_x = x_b + sgn * pos_gap * np.sin(ang)
                    base_y = y_b + sgn * pos_gap * np.cos(ang)
                    for lat in laterals:
                        expected_x = base_x + lat * perp_x
                        expected_y = base_y + lat * perp_y
                        if (abs(fx - expected_x) < pos_tol
                                and abs(fy - expected_y) < pos_tol):
                            return True
            return False

        # InFront(d1, d2) := d1 is ahead of d2, or d2 is ahead of d1.
        return _ahead(domino2, domino1) or _ahead(domino1, domino2)

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
    # Sequence generation helpers
    # -------------------------------------------------------------------------

    def place_domino(self,
                     _domino_idx: int,
                     x: float,
                     y: float,
                     rot: float,
                     is_start_block: bool = False,
                     is_target_block: bool = False,
                     is_heavy_block: bool = False,
                     rng: Optional[np.random.Generator] = None,
                     task_idx: Optional[int] = None) -> Dict:
        """Create a dictionary with placement parameters for a domino."""
        if is_heavy_block:
            color = self.heavy_block_color
        elif is_start_block:
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
    # Public properties
    # -------------------------------------------------------------------------

    @property
    def Toppled(self) -> Predicate:
        """Toppled."""
        return self._Toppled
