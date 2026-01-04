"""PyBullet Domino Environment Package.

This package provides a modular, component-based domino environment for PyBullet.
Components can be composed together to create different environment variants.

Example usage:

    # Use pre-configured environments (backward-compatible)
    from predicators.envs.pybullet_domino import PyBulletDominoEnv, PyBulletDominoFanEnv

    env = PyBulletDominoEnv(use_gui=True)
    # or
    env = PyBulletDominoFanEnv(use_gui=True)

    # Or create custom compositions
    from predicators.envs.pybullet_domino import (
        PyBulletDominoComposedEnv,
        DominoComponent,
        FanComponent,
        BallComponent,
    )

    domino = DominoComponent(num_dominos_max=6)
    fan = FanComponent()
    ball = BallComponent()

    env = PyBulletDominoComposedEnv(
        components=[domino, fan, ball],
        use_gui=True
    )

Adding New Components:
    1. Create a new file in components/ (e.g., ramp_component.py)
    2. Subclass DominoEnvComponent
    3. Implement required abstract methods
    4. Add to composed environment via the components list
"""

# Import composed environment and factory functions
from predicators.envs.pybullet_domino.composed_env import (
    PyBulletDominoComposedEnv,
    PyBulletDominoEnvNew,
    PyBulletDominoFanEnvNew,
    create_domino_env,
    create_domino_fan_env,
)

# Import components for custom composition
from predicators.envs.pybullet_domino.components import (
    DominoEnvComponent,
    DominoComponent,
    FanComponent,
    BallComponent,
)

# Import task generators
from predicators.envs.pybullet_domino.task_generators import (
    TaskGenerator,
    DominoTaskGenerator,
)

# Backward-compatible aliases
# These match the original class names from the flat file structure
PyBulletDominoEnv = PyBulletDominoEnvNew
PyBulletDominoFanEnv = PyBulletDominoFanEnvNew

__all__ = [
    # Main environment classes
    "PyBulletDominoComposedEnv",
    "PyBulletDominoEnv",
    "PyBulletDominoFanEnv",
    # Factory functions
    "create_domino_env",
    "create_domino_fan_env",
    # Base component
    "DominoEnvComponent",
    # Component implementations
    "DominoComponent",
    "FanComponent",
    "BallComponent",
    # Task generators
    "TaskGenerator",
    "DominoTaskGenerator",
]
