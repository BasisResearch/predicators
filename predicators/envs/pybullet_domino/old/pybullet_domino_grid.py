"""Grid-based variant of the PyBullet domino environment.

This module extends PyBulletDominoEnv with grid-based positioning and
discrete rotations.
"""
import time
from pprint import pformat
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pybullet as p

from predicators import utils
from predicators.envs.pybullet_domino.old.pybullet_domino import \
    PyBulletDominoEnv
from predicators.settings import CFG
from predicators.structs import Action, DerivedPredicate, EnvironmentTask, \
    GroundAtom, Object, Predicate, State, Type


class PyBulletDominoGridEnv(PyBulletDominoEnv):
    """Grid-based variant of PyBulletDominoEnv with discrete positions and
    rotations."""

    def __init__(self, use_gui: bool = True) -> None:
        # Grid-specific types will be created before calling super().__init__()
        # Create grid types
        self._position_type = Type("loc", ["xx", "yy"],
                                   sim_features=["id", "xx", "yy"])
        self._angle_type = Type("angle", ["angle"])

        # Call parent init
        super().__init__(use_gui)

        # Create rotation objects for 8 discrete angles
        self.rotations: List[Object] = []
        angle_values = [-135, -90, -45, 0, 45, 90, 135, 180]  # degrees
        for angle in angle_values:
            name = f"ang_{angle}"
            obj = Object(name, self._angle_type)
            self.rotations.append(obj)

        self.grid_pos: List[Tuple[float, float]] = []

        # Define grid-specific predicates
        self._DominoAtPos = Predicate("DominoAtPos",
                                      [self._domino_type, self._position_type],
                                      self._DominoAtPos_holds)
        self._DominoAtRot = Predicate("DominoAtRot",
                                      [self._domino_type, self._angle_type],
                                      self._DominoAtRot_holds)
        self._Connected = Predicate("Connected",
                                    [self._position_type, self._position_type],
                                    self._Connected_holds)
        self._PosClear = Predicate("PosClear", [self._position_type],
                                   self._PosClear_holds)
        self._InFrontDirection = DerivedPredicate(
            "InFrontDirection",
            [self._domino_type, self._domino_type, self._direction_type],
            self._InFrontDirection_holds,
            auxiliary_predicates={self._DominoAtPos, self._DominoAtRot})
        self._InFront = DerivedPredicate(
            "InFront", [self._domino_type, self._domino_type],
            self._InFront_holds,
            auxiliary_predicates={self._InFrontDirection})
        self._AdjacentTo = DerivedPredicate(
            "AdjacentTo", [self._position_type, self._domino_type],
            self._AdjacentTo_holds,
            auxiliary_predicates={self._DominoAtPos})

    @classmethod
    def get_name(cls) -> str:
        return "pybullet_domino_grid"

    @property
    def predicates(self) -> Set[Predicate]:
        # Get base predicates
        base_predicates = super().predicates

        # Add grid predicates
        grid_predicates = {
            self._DominoAtPos,
            self._DominoAtRot,
            self._PosClear,
            self._InFrontDirection,
            self._InFront,
        }

        if CFG.domino_include_connected_predicate:
            grid_predicates.add(self._Connected)
        else:
            grid_predicates.add(self._AdjacentTo)

        return base_predicates | grid_predicates

    @property
    def target_predicates(self) -> Set[Predicate]:
        target_predicates = super().target_predicates
        target_predicates.add(self._InFrontDirection)
        return target_predicates

    @property
    def types(self) -> Set[Type]:
        # Get base types
        base_types = super().types
        # Add grid types
        return base_types | {self._position_type, self._angle_type}

    def _extract_feature(self, obj: Object, feature: str) -> float:
        """Extract features for creating the State object."""
        # Try grid-specific features first
        if obj.type == self._position_type:
            if feature == "xx":
                return obj.xx
            elif feature == "yy":
                return obj.yy
        elif obj.type == self._angle_type:
            if feature == "angle":
                # Extract angle from object name (e.g., "ang_45" -> 45.0)
                angle_str = obj.name.split("_")[1]
                return float(angle_str)

        # Fall back to base class
        return super()._extract_feature(obj, feature)

    def _reset_custom_env_state(self, state: State) -> None:
        """Reset the custom environment state to match the given state."""
        # Call parent implementation first
        super()._reset_custom_env_state(state)

        # Draw debug lines at grid cell centers based on current task configuration
        # Clear existing debug lines
        for line_id in self._debug_line_ids:
            p.removeUserDebugItem(line_id)
        self._debug_line_ids = []

        # Draw debug lines based on position objects' xx, yy features
        position_objs = state.get_objects(self._position_type)
        for pos_obj in position_objs:
            x = state.get(pos_obj, "xx")
            y = state.get(pos_obj, "yy")
            line_id = p.addUserDebugLine(
                [x, y, self.table_height],
                [x, y, self.table_height + self.debug_line_height], [1, 0, 0],
                parentObjectUniqueId=-1,
                parentLinkIndex=-1)
            self._debug_line_ids.append(line_id)

    # -------------------------------------------------------------------------
    # Grid Coordinate Generation

    @classmethod
    def _generate_grid_coordinates(
            cls, num_pos_x: int,
            num_pos_y: int) -> Tuple[List[float], List[float]]:
        """Generate grid coordinates for position objects with specified
        dimensions."""
        # Calculate grid extents based on workspace bounds
        total_x_range = cls.x_ub - cls.x_lb
        total_y_range = cls.y_ub - cls.y_lb

        # Center the grid within the workspace
        x_start = cls.x_lb + (total_x_range -
                              (num_pos_x - 1) * cls.pos_gap) / 2
        y_start = cls.y_lb + (total_y_range -
                              (num_pos_y - 1) * cls.pos_gap) / 2

        x_coords = [
            round(x_start + i * cls.pos_gap, 5) for i in range(num_pos_x)
        ]
        y_coords = [
            round(y_start + i * cls.pos_gap, 5) for i in range(num_pos_y)
        ]

        return x_coords, y_coords

    # -------------------------------------------------------------------------
    # Grid-Specific Predicates

    def _DominoAtPos_holds(self, state: State,
                           objects: Sequence[Object]) -> bool:
        """Check if domino is at a specific position."""
        domino, position = objects
        if state.get(domino, "is_held"):
            return False

        # Get domino's actual position
        domino_x = state.get(domino, "x")
        domino_y = state.get(domino, "y")

        # Closest position to the domino
        closest_position = None
        closest_distance = float('inf')
        for pos in state.get_objects(self._position_type):
            pos_x = state.get(pos, "xx")
            pos_y = state.get(pos, "yy")
            distance = np.sqrt((domino_x - pos_x)**2 + (domino_y - pos_y)**2)
            if distance < closest_distance:
                closest_distance = distance
                closest_position = pos
        return closest_position == position

    @classmethod
    def _DominoAtRot_holds(cls, state: State,
                           objects: Sequence[Object]) -> bool:
        """Check if domino is at a specific rotation."""
        domino, rotation = objects
        if state.get(domino, "is_held"):
            return False

        # Get domino's actual rotation (in radians)
        domino_rot = state.get(domino, "yaw")

        # Get the target rotation (convert from degrees to radians)
        target_rot_degrees = state.get(rotation, "angle")
        target_rot_radians = np.radians(target_rot_degrees)

        # Check if domino rotation is close enough to target rotation
        rotation_tolerance = np.radians(15)  # 15 degrees tolerance
        angle_diff = abs(utils.wrap_angle(domino_rot - target_rot_radians))

        return angle_diff <= rotation_tolerance

    @classmethod
    def _Connected_holds(cls, state: State, objects: Sequence[Object]) -> bool:
        """Check if two positions are adjacent in cardinal directions only.

        Returns True if positions are adjacent up/down or left/right,
        but False for diagonal adjacencies.
        """
        pos1, pos2 = objects
        if pos1.name == pos2.name:
            return False

        # Get coordinates of both positions
        x1 = state.get(pos1, "xx")
        y1 = state.get(pos1, "yy")
        x2 = state.get(pos2, "xx")
        y2 = state.get(pos2, "yy")

        # Calculate differences
        dx = abs(x1 - x2)
        dy = abs(y1 - y2)

        # Positions are connected if they are exactly one grid step apart
        # in only one direction (either x or y, but not both)
        grid_step = cls.pos_gap
        tolerance = grid_step * 0.1  # Small tolerance for floating point comparison

        # Check if adjacent in x-direction only (same row)
        x_adjacent = abs(dx - grid_step) < tolerance and dy < tolerance

        # Check if adjacent in y-direction only (same column)
        y_adjacent = abs(dy - grid_step) < tolerance and dx < tolerance

        return x_adjacent or y_adjacent

    @classmethod
    def _PosClear_holds(cls, state: State, objects: Sequence[Object]) -> bool:
        """Check if a position is clear (not occupied by any domino).

        A position is considered clear if no domino is currently at that
        position.
        """
        position, = objects

        # Get the position coordinates
        target_x = state.get(position, "xx")
        target_y = state.get(position, "yy")

        # Check if any domino is at this position
        position_tolerance = cls.pos_gap * 0.5
        for domino in state.get_objects(cls._domino_type):
            domino_x = state.get(domino, "x")
            domino_y = state.get(domino, "y")

            # If domino is close enough to this position, position is not clear
            if (abs(domino_x - target_x) <= position_tolerance
                    and abs(domino_y - target_y) <= position_tolerance
                    and not state.get(domino, "is_held")):
                return False

        return True

    @classmethod
    def _InFrontDirection_holds(cls, atoms: Set[GroundAtom],
                                objects: Sequence[Object]) -> bool:
        """Check if domino1 is in front of domino2 in the given direction.

        This is an optimized implementation for heuristic evaluation. It
        decouples the positional and rotational checks to be much faster.
        It remains correct for concrete states but may produce false
        positives in relaxed states (which is acceptable for a heuristic).

        The relationship is symmetric: InFrontDirection(d1, d2, "right") is
        true if either:
        - d1 is in the cell in front of d2 with a rotation difference of -π/4, OR
        - d2 is in the cell in front of d1 with a rotation difference of +π/4
          (equivalent to InFrontDirection(d2, d1, "left")).
        """
        domino1, domino2, direction_obj = objects

        # Helper functions to parse object names and cache results
        _pos_coord_cache: Dict[Object, Tuple[int, int]] = {}
        _rot_rad_cache: Dict[Object, float] = {}

        def extract_grid_coords(pos_obj: Object) -> Tuple[int, int]:
            if pos_obj in _pos_coord_cache:
                return _pos_coord_cache[pos_obj]
            name_parts = pos_obj.name.split("_")
            y_idx = int(name_parts[1][1:])
            x_idx = int(name_parts[2][1:])
            result = (x_idx, y_idx)
            _pos_coord_cache[pos_obj] = result
            return result

        def extract_rotation_angle_rad(rot_obj: Object) -> float:
            if rot_obj in _rot_rad_cache:
                return _rot_rad_cache[rot_obj]
            angle_str = rot_obj.name.split("_")[1]
            result = np.radians(float(angle_str))
            _rot_rad_cache[rot_obj] = result
            return result

        # Step 1: Gather all possible states for each domino.
        d1_positions_coords = {
            extract_grid_coords(atom.objects[1])
            for atom in atoms if atom.predicate.name == "DominoAtPos"
            and atom.objects[0] == domino1
        }
        d1_rotations_rad = {
            extract_rotation_angle_rad(atom.objects[1])
            for atom in atoms if atom.predicate.name == "DominoAtRot"
            and atom.objects[0] == domino1
        }
        d2_positions_coords = {
            extract_grid_coords(atom.objects[1])
            for atom in atoms if atom.predicate.name == "DominoAtPos"
            and atom.objects[0] == domino2
        }
        d2_rotations_rad = {
            extract_rotation_angle_rad(atom.objects[1])
            for atom in atoms if atom.predicate.name == "DominoAtRot"
            and atom.objects[0] == domino2
        }

        # Step 2: Define the optimized function to check one directional case.
        def _check_case(front_domino_positions: Set[Tuple[int, int]],
                        front_domino_rotations: Set[float],
                        back_domino_positions: Set[Tuple[int, int]],
                        back_domino_rotations: Set[float],
                        direction_name: str,
                        tolerance: float = 1e-6) -> bool:
            """Perform decoupled checks for positional and rotational
            possibility."""
            # Fail fast if any required sets of states are empty.
            if not all([
                    front_domino_positions, front_domino_rotations,
                    back_domino_positions, back_domino_rotations
            ]):
                return False

            # 2a. Positional Check: Is there ANY valid geometric placement?
            position_possible = False
            for (x_back_idx, y_back_idx) in back_domino_positions:
                for rot_back_rad in back_domino_rotations:
                    # Relationship only holds for cardinal rotations of back domino.
                    if not (abs(np.sin(rot_back_rad)) < tolerance or \
                            abs(np.cos(rot_back_rad)) < tolerance):
                        continue
                    # Calculate expected position and check if it exists.
                    dx_idx = round(np.sin(rot_back_rad))
                    dy_idx = round(np.cos(rot_back_rad))
                    expected_front_coords = (x_back_idx + dx_idx,
                                             y_back_idx + dy_idx)
                    if expected_front_coords in front_domino_positions:
                        position_possible = True
                        break
                if position_possible:
                    break

            # If it's not positionally possible, no need to check rotation.
            if not position_possible:
                return False

            # 2b. Rotational Check: Is there ANY pair of rotations with the correct diff?
            if direction_name == "left":
                expected_rot_diff = np.pi / 4
            elif direction_name == "straight":
                expected_rot_diff = 0
            elif direction_name == "right":
                expected_rot_diff = -np.pi / 4
            else:
                return False  # Should not happen

            for rot_back_rad in back_domino_rotations:
                for rot_front_rad in front_domino_rotations:
                    diff = utils.wrap_angle(rot_front_rad - rot_back_rad)
                    if abs(diff - expected_rot_diff) < tolerance:
                        # Position is possible and rotation is possible, so we're done.
                        return True

            return False

        # Step 3: Check both symmetric cases for the relationship.
        dir_name = direction_obj.name
        if dir_name == "left":
            opposite_dir_name = "right"
        elif dir_name == "right":
            opposite_dir_name = "left"
        else:  # "straight"
            opposite_dir_name = "straight"

        # Case 1: Is domino1 in front of domino2 in `dir_name`?
        if _check_case(front_domino_positions=d1_positions_coords,
                       front_domino_rotations=d1_rotations_rad,
                       back_domino_positions=d2_positions_coords,
                       back_domino_rotations=d2_rotations_rad,
                       direction_name=dir_name):
            return True

        # Case 2: Is domino2 in front of domino1 in `opposite_dir_name`?
        if _check_case(front_domino_positions=d2_positions_coords,
                       front_domino_rotations=d2_rotations_rad,
                       back_domino_positions=d1_positions_coords,
                       back_domino_rotations=d1_rotations_rad,
                       direction_name=opposite_dir_name):
            return True

        return False

    @classmethod
    def _InFront_holds(cls, atoms: Set[GroundAtom],
                       objects: Sequence[Object]) -> bool:
        """Check if domino1 is in front of domino2 in any direction.

        This derived predicate returns True if there exists any
        direction such that InFrontDirection(domino1, domino2,
        direction) is true.
        """
        domino1, domino2 = objects

        # Check if there exists any InFrontDirection atom with these dominos
        for atom in atoms:
            if (atom.predicate.name == "InFrontDirection"
                    and len(atom.objects) == 3 and atom.objects[0] == domino1
                    and atom.objects[1] == domino2):
                return True

        return False

    @classmethod
    def _AdjacentTo_holds(cls, atoms: Set[GroundAtom],
                          objects: Sequence[Object]) -> bool:
        """Check if a position is adjacent to a domino in cardinal directions.

        This is similar to _InFrontDirection_holds but checks if a position
        is adjacent to any position where the domino could be placed, considering
        that the domino can be in multiple positions during heuristic computation.

        Adjacent positions are those that are exactly one grid step away in
        cardinal directions (up, down, left, right) but not diagonal.
        """
        position, domino = objects

        # Helper functions to parse object names and cache results
        _pos_coord_cache: Dict[Object, Tuple[int, int]] = {}

        def extract_grid_coords(pos_obj: Object) -> Tuple[int, int]:
            if pos_obj in _pos_coord_cache:
                return _pos_coord_cache[pos_obj]
            name_parts = pos_obj.name.split("_")
            y_idx = int(name_parts[1][1:])
            x_idx = int(name_parts[2][1:])
            result = (x_idx, y_idx)
            _pos_coord_cache[pos_obj] = result
            return result

        # Get coordinates of the target position
        target_coords = extract_grid_coords(position)
        target_x_idx, target_y_idx = target_coords

        # Get all possible positions where the domino could be
        domino_positions_coords = {
            extract_grid_coords(atom.objects[1])
            for atom in atoms if atom.predicate.name == "DominoAtPos"
            and atom.objects[0] == domino
        }

        # Check if the target position is adjacent to any domino position
        # Adjacent means exactly one grid step away in cardinal directions
        for domino_x_idx, domino_y_idx in domino_positions_coords:
            # Calculate the difference in grid coordinates
            dx = abs(target_x_idx - domino_x_idx)
            dy = abs(target_y_idx - domino_y_idx)

            # Adjacent in cardinal directions means:
            # - Exactly 1 step away in one direction AND 0 steps in the other
            if (dx == 1 and dy == 0) or (dx == 0 and dy == 1):
                return True

        return False

    # -------------------------------------------------------------------------
    # Grid-Based Task Generation

    def _generate_domino_sequence_with_grid(
            self,
            rng: np.random.Generator,
            n_dominos: int,
            n_targets: int,
            num_pos_x: int,
            num_pos_y: int,
            log_debug: bool = False,
            task_idx: Optional[int] = None,
            is_training: bool = False) -> Optional[Dict]:
        """Grid-based sequence generator.

        This version implements straight moves and L-shaped 90-degree
        turns, mimicking the logic of the non-grid-based generator. A
        90-degree turn consumes two dominoes and forms an 'L' shape on
        the grid. The turning domino is shifted inward for better
        stability.
        """
        obj_dict: Dict = {}
        domino_count = 0
        target_count = 0
        used_coords = set()

        # Generate grid coordinates for this specific configuration
        x_coords, y_coords = self._generate_grid_coordinates(
            num_pos_x, num_pos_y)
        grid_pos = [(x, y) for y in y_coords for x in x_coords]

        # Use a set for efficient checking of valid grid coordinates.
        grid_coords_set = set(grid_pos)

        # Choose a random starting position and orientation (cardinal directions).
        start_idx = rng.choice(len(grid_pos))
        curr_x, curr_y = grid_pos[start_idx]
        # If in the top row, can't face down because it's unreachable for the
        # robot
        top_row_y = np.max([y for _, y in grid_pos])
        if np.abs(curr_y - top_row_y) < 1e-3:
            curr_rot = rng.choice([np.pi / 2, np.pi / 2])
        else:
            curr_rot = rng.choice([0, np.pi / 2, np.pi, -np.pi / 2])
        used_coords.add((curr_x, curr_y))

        # Place the first domino (start block).
        obj_dict[self.dominos[domino_count]] = self._place_domino(
            domino_count,
            curr_x,
            curr_y,
            curr_rot,
            is_start_block=True,
            rng=rng,
            task_idx=task_idx)
        domino_count += 1
        if log_debug:
            print(f"Placed first domino at {curr_x}, {curr_y}, {curr_rot}")

        # Determine total domino blocks to place.
        if CFG.domino_use_domino_blocks_as_target:
            total_domino_blocks = n_dominos + n_targets
        else:
            total_domino_blocks = n_dominos

        # Main placement loop.
        while domino_count < total_domino_blocks:
            possible_moves = []
            # A move is defined by: (name, final_x, final_y, final_rot, placements)
            # where placements is a list of (x, y, rot) for each domino in the move.

            # 1. Check for a "straight" move (1 domino).
            dx = round(self.pos_gap * np.sin(curr_rot), 5)
            dy = round(self.pos_gap * np.cos(curr_rot), 5)
            next_x = round(curr_x + dx, 5)
            next_y = round(curr_y + dy, 5)

            if (next_x, next_y) in grid_coords_set and \
            (next_x, next_y) not in used_coords:
                placements = [(next_x, next_y, curr_rot)]
                possible_moves.append(
                    ("straight", next_x, next_y, curr_rot, placements))

            # 2. Check for "turn" moves (2 dominoes).
            # Skip turns if in training mode and the straight-only flag is set
            if (total_domino_blocks - domino_count) >= 2 and not (
                    is_training
                    and CFG.domino_only_straight_sequence_in_training):
                # turn_dir: -1 for left, 1 for right.
                for turn_dir, name in [(-1, "turn_left"), (1, "turn_right")]:
                    # The first domino (d1) is one step straight on the grid.
                    d1_grid_x, d1_grid_y = next_x, next_y
                    if (d1_grid_x, d1_grid_y) not in grid_coords_set or \
                       (d1_grid_x, d1_grid_y) in used_coords:
                        continue

                    # Its orientation is 45 degrees towards the turn direction.
                    d1_rot = curr_rot - turn_dir * np.pi / 4

                    # Calculate the shift vector to pull the turning domino inward.
                    shift_magnitude = self.domino_width * self.turn_shift_frac
                    shift_dx = shift_magnitude * \
                        (turn_dir * np.cos(curr_rot) - np.sin(curr_rot))
                    shift_dy = shift_magnitude * \
                        (-turn_dir * np.sin(curr_rot) - np.cos(curr_rot))

                    # The physical position is the grid position plus the shift.
                    d1_x = d1_grid_x + shift_dx
                    d1_y = d1_grid_y + shift_dy

                    # The second domino (d2) completes the turn.
                    d2_rot = d1_rot - turn_dir * 1 * np.pi / 4

                    # Calculate d2's physical position relative to d1's.
                    gap = self.pos_gap
                    sin_d1 = np.sin(d1_rot)
                    cos_d1 = np.cos(d1_rot)
                    disp_x = (
                        gap * turn_dir * cos_d1 +
                        (2 * shift_magnitude - gap) * sin_d1) / np.sqrt(2)
                    disp_y = (
                        -gap * turn_dir * sin_d1 +
                        (2 * shift_magnitude - gap) * cos_d1) / np.sqrt(2)
                    d2_x = round(d1_x + disp_x, 5)
                    d2_y = round(d1_y + disp_y, 5)

                    # Check if the grid position of the second domino is valid.
                    # We need to determine where the grid position *would* be.
                    # A left turn from rot=pi should result in rot=pi/2.
                    # This is equivalent to rot + turn_dir * pi/2
                    expected_final_rot = curr_rot + turn_dir * np.pi / 2
                    d2_grid_dx = round(
                        self.pos_gap * np.sin(expected_final_rot), 5)
                    d2_grid_dy = round(
                        self.pos_gap * np.cos(expected_final_rot), 5)
                    d2_grid_x = round(d1_grid_x + d2_grid_dx, 5)
                    d2_grid_y = round(d1_grid_y + d2_grid_dy, 5)

                    if (d2_grid_x, d2_grid_y) in grid_coords_set and \
                       (d2_grid_x, d2_grid_y) not in used_coords:

                        placements = [(d1_x, d1_y, d1_rot),
                                      (d2_x, d2_y, d2_rot)]

                        possible_moves.append(
                            (name, d2_grid_x, d2_grid_y, d2_rot, placements))

            if not possible_moves:
                # No valid moves, generation failed for this attempt.
                return None

            # Choose a random valid move and get its placement plan.
            _move_name, final_x, final_y, final_rot, placements = \
                possible_moves[rng.choice(len(possible_moves))]
            if log_debug:
                print(
                    f"Chose move: {_move_name}, final_x: {final_x}, final_y: {final_y}, final_rot: {final_rot}"
                )

            # Execute the placement plan for the chosen move.
            for (x, y, rot) in placements:
                if log_debug:
                    print(f"Placing domino at {x}, {y}, {rot}")
                if domino_count >= total_domino_blocks:
                    break  # Should not be reached with correct logic.

                # Decide if this domino block should be a target.
                is_target = False
                if CFG.domino_use_domino_blocks_as_target and \
                target_count < n_targets:
                    remaining_blocks = total_domino_blocks - domino_count
                    remaining_targets = n_targets - target_count

                    # Reserve one target for the very last domino in the sequence
                    is_last_block = (domino_count == total_domino_blocks - 1)
                    has_targets_left = remaining_targets > 0

                    if is_last_block and has_targets_left:
                        # Force the last domino to be a target if we still need targets
                        is_target = True
                    elif not is_last_block and remaining_targets > 1:
                        # For non-last dominoes, only consider making them targets if we have more than 1 target left
                        # This ensures at least one target is reserved for the end
                        targets_available_for_placement = remaining_targets - 1
                        if (targets_available_for_placement >= remaining_blocks - 1 or \
                            rng.random() < targets_available_for_placement / (remaining_blocks - 1)) and\
                            domino_count >= 2:
                            is_target = True

                # Place the domino block.
                obj_dict[self.dominos[domino_count]] = self._place_domino(
                    domino_count,
                    x,
                    y,
                    rot,
                    is_target_block=is_target,
                    rng=rng,
                    task_idx=task_idx)

                # Use the grid coordinates for tracking used spots.
                # Find the closest grid coordinate to the physical placement
                closest_grid_coord = min(grid_pos,
                                         key=lambda p: (p[0] - x)**2 +
                                         (p[1] - y)**2)
                used_coords.add(closest_grid_coord)

                if is_target:
                    target_count += 1
                domino_count += 1

            # Update state for the next iteration.
            curr_x, curr_y, curr_rot = final_x, final_y, final_rot

        # Place non-block targets if necessary (not on grid centers).
        if not CFG.domino_use_domino_blocks_as_target:
            raise NotImplementedError(
                "Placing non-block targets with the grid generator is not "
                "currently supported.")

        return obj_dict

    def _make_tasks(self,
                    num_tasks: int,
                    possible_num_dominos: List[int],
                    possible_num_targets: List[int],
                    possible_num_pivots: List[int],
                    rng: np.random.Generator,
                    log_debug: bool = True,
                    is_training: bool = False) -> List[EnvironmentTask]:
        """Override to use grid-based sequence generation and add position
        objects."""
        tasks = []
        total_attempts = 0

        # Get grid dimensions from CFG
        if is_training:
            num_pos_x = CFG.domino_train_num_pos_x
            num_pos_y = CFG.domino_train_num_pos_y
        else:
            num_pos_x = CFG.domino_test_num_pos_x
            num_pos_y = CFG.domino_test_num_pos_y

        for i_task in range(num_tasks):
            # 1) Robot initial
            robot_dict = {
                "x": self.robot_init_x,
                "y": self.robot_init_y,
                "z": self.robot_init_z,
                "fingers": self.open_fingers,
                "tilt": self.robot_init_tilt,
                "wrist": self.robot_init_wrist,
            }

            # 2) Dominoes
            init_dict = {self._robot: robot_dict}

            # Add direction objects to initial state
            for i, direction_obj in enumerate(self.directions):
                init_dict[direction_obj] = {"dir": float(i)}

            # Generate grid coordinates for this specific configuration
            x_coords, y_coords = self._generate_grid_coordinates(
                num_pos_x, num_pos_y)
            self.grid_pos = [(x, y) for y in y_coords for x in x_coords]

            # Create position objects
            positions: List[Object] = []
            for i in range(num_pos_x * num_pos_y):
                name = f"loc_y{i//num_pos_x}_x{i%num_pos_x}"
                obj = Object(name, self._position_type)
                positions.append(obj)

            # Create position dictionary for this task configuration
            pos_dict = {}
            pos_index = 0
            for i in range(num_pos_y):
                for j in range(num_pos_x):
                    if pos_index < len(positions):
                        pos_obj = positions[pos_index]
                        pos_dict[pos_obj] = {
                            "xx": x_coords[j],
                            "yy": y_coords[i]
                        }
                        # Set sim features for position objects
                        pos_obj.xx = x_coords[j]
                        pos_obj.yy = y_coords[i]
                        pos_index += 1

            # Add position objects to initial state
            init_dict.update(pos_dict)

            # Add rotation objects to initial state
            for rotation_obj in self.rotations:
                angle_str = rotation_obj.name.split("_")[1]
                init_dict[rotation_obj] = {"angle": float(angle_str)}

            # Place dominoes (D) and targets (T) in order: D D T D T
            # at fixed positions along the x-axis
            n_dominos = rng.choice(possible_num_dominos)
            n_targets = rng.choice(possible_num_targets)
            n_pivots = rng.choice(possible_num_pivots)

            # Generate sequence using grid helper function
            obj_dict = None
            max_attempts = 1000
            for i in range(max_attempts):
                if log_debug:
                    print(f"\nAttempt {i} for task {i_task}")
                obj_dict = self._generate_domino_sequence_with_grid(
                    rng,
                    n_dominos,
                    n_targets,
                    num_pos_x,
                    num_pos_y,
                    log_debug=log_debug,
                    task_idx=i_task,
                    is_training=is_training)
                if obj_dict is not None:
                    if log_debug:
                        print("Found satisfying a task")
                    break

            if obj_dict is None:
                raise RuntimeError("Failed to generate valid domino sequence")
            if log_debug:
                print(f"Found a task")

            # If we want to initialize at finished state, move intermediate objects
            if not CFG.domino_initialize_at_finished_state:
                obj_dict = self._move_intermediate_objects_to_unfinished_state(
                    obj_dict, num_pos_x, num_pos_y)

            init_dict.update(obj_dict)
            init_state = utils.create_state_from_dict(init_dict)

            # The goal: topple all targets
            if CFG.domino_use_domino_blocks_as_target:
                # Find target dominoes (pink dominoes) and set them as goals
                goal_atoms = set()
                for domino_obj in init_state.get_objects(self._domino_type):
                    if self._TargetDomino_holds(init_state, [domino_obj]):
                        goal_atoms.add(GroundAtom(self._Toppled, [domino_obj]))
            else:
                # Use regular targets
                goal_atoms = {GroundAtom(self._Toppled, [self.targets[0]])}

            tasks.append(EnvironmentTask(init_state, goal_atoms))
            total_attempts += i + 1
        if log_debug:
            print(f"Total attempts: {total_attempts}")

        return self._add_pybullet_state_to_tasks(tasks)

    def _move_intermediate_objects_to_unfinished_state(
            self,
            obj_dict: Dict,
            num_pos_x: Optional[int] = None,
            num_pos_y: Optional[int] = None) -> Dict:
        """Move all intermediate dominoes and pivots using grid positioning."""
        # Identify which objects to move
        intermediate_objects = []

        # Find all dominoes except the start domino and target dominoes
        for domino in self.dominos:
            if domino in obj_dict:
                domino_data = obj_dict[domino]
                eps = 1e-3
                is_start_domino = (abs(
                    domino_data.get("r", 0.0) -
                    self.start_domino_color[0]) < eps and abs(
                        domino_data.get("g", 0.0) - self.start_domino_color[1])
                                   < eps and abs(
                                       domino_data.get("b", 0.0) -
                                       self.start_domino_color[2]) < eps)

                is_target_domino = False
                if CFG.domino_use_domino_blocks_as_target:
                    is_target_domino = (abs(
                        domino_data.get("r", 0.0) - self.target_domino_color[0]
                    ) < eps and abs(
                        domino_data.get("g", 0.0) -
                        self.target_domino_color[1]) < eps and abs(
                            domino_data.get("b", 0.0) -
                            self.target_domino_color[2]) < eps) or (abs(
                                domino_data.get("r", 0.0) -
                                self.glued_domino_color[0]) < eps and abs(
                                    domino_data.get("g", 0.0) -
                                    self.glued_domino_color[1]) < eps and abs(
                                        domino_data.get("b", 0.0) -
                                        self.glued_domino_color[2]) < eps)

                if not is_start_domino and not is_target_domino:
                    intermediate_objects.append((domino, "domino"))

        # Find all pivots
        for pivot in self.pivots:
            if pivot in obj_dict:
                intermediate_objects.append((pivot, "pivot"))

        if not intermediate_objects:
            return obj_dict

        # Use grid positioning
        # First, identify which grid positions are already occupied
        occupied_positions = set()
        position_tolerance = self.pos_gap * 0.5

        intermediate_obj_set = {obj for obj, obj_type in intermediate_objects}

        for obj, obj_data in obj_dict.items():
            if obj not in intermediate_obj_set:
                obj_x = obj_data.get("x", 0.0)
                obj_y = obj_data.get("y", 0.0)

                # Check which grid position this object occupies
                for grid_x, grid_y in self.grid_pos:
                    if (abs(obj_x - grid_x) <= position_tolerance
                            and abs(obj_y - grid_y) <= position_tolerance):
                        occupied_positions.add((grid_x, grid_y))
                        break

        # Find available positions on the bottom side, starting from middle
        if num_pos_x is not None and num_pos_y is not None:
            x_coords, y_coords = self._generate_grid_coordinates(
                num_pos_x, num_pos_y)
        else:
            # Fallback to maximum grid size
            max_num_pos_x = max(CFG.domino_train_num_pos_x,
                                CFG.domino_test_num_pos_x)
            max_num_pos_y = max(CFG.domino_train_num_pos_y,
                                CFG.domino_test_num_pos_y)
            x_coords, y_coords = self._generate_grid_coordinates(
                max_num_pos_x, max_num_pos_y)

        x_center = (x_coords[0] + x_coords[-1]) / 2 if x_coords else 0

        # Get bottom row positions first, then other rows if needed
        available_positions = []
        for y in sorted(y_coords):  # Start from bottom (smallest y)
            row_positions = [(x, y) for x in x_coords
                             if (x, y) not in occupied_positions]
            # Sort by distance from center
            row_positions.sort(key=lambda pos: abs(pos[0] - x_center))
            available_positions.extend(row_positions)

        # Place intermediate objects on available grid positions
        for i, (obj, obj_type) in enumerate(intermediate_objects):
            if i < len(available_positions):
                new_x, new_y = available_positions[i]
            else:
                # Fallback to non-grid positioning if we run out of grid positions
                start_x = self.x_lb + self.domino_width
                spacing = self.domino_width * 1.5
                new_x = start_x + i * spacing
                new_y = (self.y_lb + self.y_ub) / 2

            if obj_type == "domino":
                obj_dict[obj] = {
                    "x": new_x,
                    "y": new_y,
                    "z": self.z_lb + self.domino_height / 2,
                    "yaw": 0.0,
                    "roll": 0.0,
                    "r": self.domino_color[0],
                    "g": self.domino_color[1],
                    "b": self.domino_color[2],
                    "is_held": 0.0,
                }
            elif obj_type == "pivot":
                obj_dict[obj] = {
                    "x": new_x,
                    "y": new_y,
                    "z": self.z_lb,
                    "yaw": 0.0,
                }

        return obj_dict


if __name__ == "__main__":

    CFG.seed = 0
    CFG.env = "pybullet_domino_grid"
    CFG.domino_initialize_at_finished_state = False
    CFG.domino_use_domino_blocks_as_target = True
    CFG.domino_use_grid = True
    CFG.num_train_tasks = 2
    CFG.num_test_tasks = 2
    env = PyBulletDominoGridEnv(use_gui=True)

    tasks = env._generate_train_tasks()

    for task in tasks:
        env._reset_state(task.init)
        print(
            f"init state: {pformat(utils.abstract(task.init, env.predicates))}\n"
        )
        print(f"goal: {task.goal}\n")
        print(pformat(task.init.pretty_str()), '\n')

        for i in range(1000):
            action = Action(
                np.array(env._pybullet_robot.initial_joint_positions))
            env.step(action)
            time.sleep(0.01)
