"""
PyBullet compatibility layer for Python 3.13.

This module provides a compatibility layer that allows predicators to run
without PyBullet when it's not available (e.g., on Python 3.13 where
PyBullet has compilation issues).
"""

import warnings
from typing import Any, Optional

# Try to import PyBullet and handle gracefully if not available
try:
    import pybullet as _pybullet
    PYBULLET_AVAILABLE = True
except ImportError:
    PYBULLET_AVAILABLE = False
    _pybullet = None
    warnings.warn(
        "PyBullet is not available. PyBullet-dependent environments will be skipped. "
        "This is expected on Python 3.13 due to compilation issues.",
        UserWarning,
        stacklevel=2
    )

# Try to import pybullet_utils
try:
    from pybullet_utils.transformations import euler_from_quaternion as _euler_from_quaternion
    from pybullet_utils.transformations import quaternion_from_euler as _quaternion_from_euler
    PYBULLET_UTILS_AVAILABLE = True
except ImportError:
    PYBULLET_UTILS_AVAILABLE = False
    _euler_from_quaternion = None
    _quaternion_from_euler = None

def get_pybullet() -> Optional[Any]:
    """Get the PyBullet module if available, None otherwise."""
    return _pybullet if PYBULLET_AVAILABLE else None

def euler_from_quaternion(*args, **kwargs) -> Any:
    """Wrapper for pybullet_utils.transformations.euler_from_quaternion."""
    if not PYBULLET_UTILS_AVAILABLE:
        raise NotImplementedError("PyBullet utils not available")
    return _euler_from_quaternion(*args, **kwargs)

def quaternion_from_euler(*args, **kwargs) -> Any:
    """Wrapper for pybullet_utils.transformations.quaternion_from_euler."""
    if not PYBULLET_UTILS_AVAILABLE:
        raise NotImplementedError("PyBullet utils not available")
    return _quaternion_from_euler(*args, **kwargs)

# Dummy JointPositions class for when PyBullet is not available
class JointPositions:
    """Dummy JointPositions class for when PyBullet is not available."""
    
    def __init__(self, *args, **kwargs):
        if not PYBULLET_AVAILABLE:
            raise NotImplementedError("PyBullet not available")

# Export the actual JointPositions if PyBullet is available
if PYBULLET_AVAILABLE:
    try:
        from predicators.pybullet_helpers.joint import JointPositions as _RealJointPositions
        JointPositions = _RealJointPositions
    except ImportError:
        # Keep the dummy class if the real one can't be imported
        pass

