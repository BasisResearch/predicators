"""Handle creation of perceivers."""

from predicators import utils
from predicators.perception.base_perceiver import BasePerceiver

__all__ = ["BasePerceiver"]

# Find the subclasses.
utils.import_submodules(__path__, __name__)


def create_perceiver(name: str, ) -> BasePerceiver:
    """Create a perceiver given its name."""
    cls = utils.get_registered_subclass(BasePerceiver, name)
    if cls is None:
        raise NotImplementedError(f"Unrecognized perceiver: {name}")
    return cls()
