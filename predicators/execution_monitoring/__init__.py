"""Handle creation of execution monitors."""

from predicators import utils
from predicators.execution_monitoring.base_execution_monitor import \
    BaseExecutionMonitor

__all__ = ["BaseExecutionMonitor"]

# Find the subclasses.
utils.import_submodules(__path__, __name__)


def create_execution_monitor(name: str, ) -> BaseExecutionMonitor:
    """Create an execution monitor given its name."""
    cls = utils.get_registered_subclass(BaseExecutionMonitor, name)
    if cls is None:
        raise NotImplementedError(f"Unrecognized execution monitor: {name}")
    return cls()
