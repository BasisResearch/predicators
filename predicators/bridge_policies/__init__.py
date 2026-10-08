"""Handle creation of bridge policies."""

from typing import Set

from predicators import utils
from predicators.bridge_policies.base_bridge_policy import BaseBridgePolicy, \
    BridgePolicyDone
from predicators.structs import NSRT, ParameterizedOption, Predicate, Type

__all__ = ["BaseBridgePolicy", "BridgePolicyDone", "create_bridge_policy"]

# Find the subclasses.
utils.import_submodules(__path__, __name__)


def create_bridge_policy(name: str, types: Set[Type],
                         predicates: Set[Predicate],
                         options: Set[ParameterizedOption],
                         nsrts: Set[NSRT]) -> BaseBridgePolicy:
    """Create a bridge policy given its name."""
    cls = utils.get_registered_subclass(BaseBridgePolicy, name)
    if cls is None:
        raise NotImplementedError(f"Unknown bridge policy: {name}")
    return cls(types, predicates, options, nsrts)
