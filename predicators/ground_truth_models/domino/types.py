"""Helper types for the domino environment."""

from typing import Set

from predicators.ground_truth_models import GroundTruthTypeFactory
from predicators.structs import Type


class PyBulletDominoGroundTruthTypeFactory(GroundTruthTypeFactory):
    """Ground-truth helper types for the domino environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"pybullet_domino"}

    @classmethod
    def get_helper_types(cls, env_name: str) -> Set[Type]:
        """Get helper types for the domino environment.

        Returns position and rotation types used for grid-based
        planning.
        """
        del env_name  # unused

        # Position type with xx, yy coordinates
        position_type = Type("loc", ["xx", "yy"])

        # Angle type for discrete rotations
        angle_type = Type("angle", ["angle"])

        # Direction type for sequence generation
        direction_type = Type("direction", ["dir"])

        return {position_type, angle_type, direction_type}
