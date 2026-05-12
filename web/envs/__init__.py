"""Browser-runnable demo envs, now backed by real PyBullet (pyodide-bullet)."""
from web.envs.base import BaseDemoEnv
from web.envs.blocks import BlocksEnv
from web.envs.bowling import BowlingEnv
from web.envs.cover import CoverEnv
from web.envs.domino import DominoEnv
from web.envs.newton import NewtonCradleEnv
from web.envs.wrecking import WreckingEnv

ENVS: dict[str, type[BaseDemoEnv]] = {
    "blocks": BlocksEnv,
    "cover": CoverEnv,
    "domino": DominoEnv,
    "newton": NewtonCradleEnv,
    "wrecking": WreckingEnv,
    "bowling": BowlingEnv,
}

__all__ = [
    "BaseDemoEnv", "BlocksEnv", "BowlingEnv", "CoverEnv", "DominoEnv",
    "NewtonCradleEnv", "WreckingEnv", "ENVS",
]
