"""Browser-runnable demo envs, now backed by real PyBullet (pyodide-bullet)."""
from web.envs.base import BaseDemoEnv
from web.envs.blocks import BlocksEnv
from web.envs.cover import CoverEnv
from web.envs.domino import DominoEnv

ENVS: dict[str, type[BaseDemoEnv]] = {
    "blocks": BlocksEnv,
    "cover": CoverEnv,
    "domino": DominoEnv,
}

__all__ = ["BaseDemoEnv", "BlocksEnv", "CoverEnv", "DominoEnv", "ENVS"]
