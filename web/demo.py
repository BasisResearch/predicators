"""Entry point loaded by Pyodide.

Called by ``web/app.js`` after the pybullet wheel is installed.
"""
from __future__ import annotations

from typing import Optional

from web.envs import ENVS
from web.envs.base import BaseDemoEnv

_env: Optional[BaseDemoEnv] = None


def list_envs() -> list[str]:
    return sorted(ENVS)


def load_env(name: str, seed: int = 0) -> list[dict]:
    """Tear down any existing env, build a fresh one, return the render
    manifest the JS host needs to instantiate Three.js meshes."""
    global _env
    if name not in ENVS:
        raise ValueError(f"Unknown env {name!r}; have {sorted(ENVS)}")
    if _env is not None:
        _env.disconnect()
    _env = ENVS[name](seed=seed)
    _env.connect()
    _env.reset()
    return _env.get_render_manifest()


def step(dt: float = 1.0 / 60.0) -> None:
    if _env is None:
        return
    _env.step(dt)


def poses() -> dict:
    if _env is None:
        return {}
    return _env.get_pose_dict()
