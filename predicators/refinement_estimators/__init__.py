"""Handle creation of refinement cost estimators."""

from predicators import utils
from predicators.refinement_estimators.base_refinement_estimator import \
    BaseRefinementEstimator

__all__ = ["BaseRefinementEstimator", "create_refinement_estimator"]

# Find the subclasses.
utils.import_submodules(__path__, __name__)


def create_refinement_estimator(name: str) -> BaseRefinementEstimator:
    """Create an approach given its name."""
    cls = utils.get_registered_subclass(BaseRefinementEstimator, name)
    if cls is None:
        raise NotImplementedError(f"Unknown refinement cost estimator: {name}")
    return cls()
