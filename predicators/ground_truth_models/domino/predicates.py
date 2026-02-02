"""Helper predicates for the domino environment."""

from typing import Dict, Sequence, Set

import numpy as np

from predicators import utils
from predicators.ground_truth_models import GroundTruthPredicateFactory
from predicators.structs import DerivedPredicate, GroundAtom, Object, \
    Predicate, State, Type


class PyBulletDominoGroundTruthPredicateFactory(GroundTruthPredicateFactory):
    """Ground-truth helper predicates for the domino environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"pybullet_domino"}

    @classmethod
    def get_helper_predicates(cls, env_name: str,
                              types: Dict[str, Type]) -> Set[Predicate]:
        """Get helper predicates for the domino environment.

        Returns DominoAtPos, DominoAtRot, and InFront predicates.
        """
        del env_name  # unused

        # Get the required types from the passed-in types dict
        domino_type = types["domino"]
        position_type = types["loc"]
        angle_type = types["angle"]
        direction_type = types["direction"]

        # DominoAtPos predicate
        DominoAtPos = Predicate("DominoAtPos", [domino_type, position_type],
                                cls._DominoAtPos_holds)

        # DominoAtRot predicate
        DominoAtRot = Predicate("DominoAtRot", [domino_type, angle_type],
                                cls._DominoAtRot_holds)

        # InFrontDirection derived predicate
        InFrontDirection = DerivedPredicate(
            "InFrontDirection", [domino_type, domino_type, direction_type],
            cls._InFrontDirection_holds,
            auxiliary_predicates={DominoAtPos, DominoAtRot})

        # InFront derived predicate
        InFront = DerivedPredicate("InFront", [domino_type, domino_type],
                                   cls._InFront_holds,
                                   auxiliary_predicates={InFrontDirection})

        return {DominoAtPos, DominoAtRot, InFrontDirection, InFront}

    @staticmethod
    def _DominoAtPos_holds(state: State, objects: Sequence[Object]) -> bool:
        """Check if domino is at a specific position."""
        domino, position = objects
        if state.get(domino, "is_held"):
            return False

        # Get domino's actual position
        domino_x = state.get(domino, "x")
        domino_y = state.get(domino, "y")

        # Get position type to find all positions
        position_type = position.type

        # Find closest position to the domino
        closest_position = None
        closest_distance = float('inf')
        for pos in state.get_objects(position_type):
            pos_x = state.get(pos, "xx")
            pos_y = state.get(pos, "yy")
            distance = np.sqrt((domino_x - pos_x)**2 + (domino_y - pos_y)**2)
            if distance < closest_distance:
                closest_distance = distance
                closest_position = pos

        return closest_position == position

    @staticmethod
    def _DominoAtRot_holds(state: State, objects: Sequence[Object]) -> bool:
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

    @staticmethod
    def _InFrontDirection_holds(atoms: Set[GroundAtom],
                                objects: Sequence[Object]) -> bool:
        """Check if domino1 is in front of domino2 in the given direction.

        This is an optimized implementation for heuristic evaluation.
        """
        domino1, domino2, direction_obj = objects

        # Helper functions to parse object names and cache results
        _pos_coord_cache: Dict[Object, tuple] = {}
        _rot_rad_cache: Dict[Object, float] = {}

        def extract_grid_coords(pos_obj: Object) -> tuple:
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

        # Gather all possible states for each domino
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

        def _check_case(front_domino_positions: Set[tuple],
                        front_domino_rotations: Set[float],
                        back_domino_positions: Set[tuple],
                        back_domino_rotations: Set[float],
                        direction_name: str,
                        tolerance: float = 1e-6) -> bool:
            """Perform decoupled checks for positional and rotational
            possibility."""
            # Fail fast if any required sets are empty
            if not all([
                    front_domino_positions, front_domino_rotations,
                    back_domino_positions, back_domino_rotations
            ]):
                return False

            # Positional Check: Is there ANY valid geometric placement?
            position_possible = False
            for (x_back_idx, y_back_idx) in back_domino_positions:
                for rot_back_rad in back_domino_rotations:
                    # Relationship only holds for cardinal rotations
                    if not (abs(np.sin(rot_back_rad)) < tolerance
                            or abs(np.cos(rot_back_rad)) < tolerance):
                        continue
                    # Calculate expected position
                    dx_idx = round(np.sin(rot_back_rad))
                    dy_idx = round(np.cos(rot_back_rad))
                    expected_front_coords = (x_back_idx + dx_idx,
                                             y_back_idx + dy_idx)
                    if expected_front_coords in front_domino_positions:
                        position_possible = True
                        break
                if position_possible:
                    break

            if not position_possible:
                return False

            # Rotational Check: Is there ANY pair with correct rotation diff?
            if direction_name == "left":
                expected_rot_diff = np.pi / 4
            elif direction_name == "straight":
                expected_rot_diff = 0
            elif direction_name == "right":
                expected_rot_diff = -np.pi / 4
            else:
                return False

            for rot_back_rad in back_domino_rotations:
                for rot_front_rad in front_domino_rotations:
                    diff = utils.wrap_angle(rot_front_rad - rot_back_rad)
                    if abs(diff - expected_rot_diff) < tolerance:
                        return True

            return False

        # Check both symmetric cases for the relationship
        dir_name = direction_obj.name
        if dir_name == "left":
            opposite_dir_name = "right"
        elif dir_name == "right":
            opposite_dir_name = "left"
        else:  # "straight"
            opposite_dir_name = "straight"

        # Case 1: Is domino1 in front of domino2 in dir_name?
        if _check_case(front_domino_positions=d1_positions_coords,
                       front_domino_rotations=d1_rotations_rad,
                       back_domino_positions=d2_positions_coords,
                       back_domino_rotations=d2_rotations_rad,
                       direction_name=dir_name):
            return True

        # Case 2: Is domino2 in front of domino1 in opposite_dir_name?
        if _check_case(front_domino_positions=d2_positions_coords,
                       front_domino_rotations=d2_rotations_rad,
                       back_domino_positions=d1_positions_coords,
                       back_domino_rotations=d1_rotations_rad,
                       direction_name=opposite_dir_name):
            return True

        return False

    @staticmethod
    def _InFront_holds(atoms: Set[GroundAtom],
                       objects: Sequence[Object]) -> bool:
        """Check if domino1 is in front of domino2 in any direction."""
        domino1, domino2 = objects

        # Check if there exists any InFrontDirection atom with these dominos
        for atom in atoms:
            if (atom.predicate.name == "InFrontDirection"
                    and len(atom.objects) == 3 and atom.objects[0] == domino1
                    and atom.objects[1] == domino2):
                return True

        return False
