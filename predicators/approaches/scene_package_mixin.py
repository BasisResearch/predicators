"""The scene package: the engine wrapper, the scene manifest and the asset
files the scene's bodies were loaded from, as read-only sandbox references.

The agentic real-to-sim arm builds its scene from them, EMPIRIC can
receive them beside its twin (``continual_provide_scene_package``), and
so can the direct agent, which gets no simulator of any kind and may use
them however it likes.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

from predicators.code_sim_learning.rollout_env import dispose_env
from predicators.code_sim_learning.scene_manifest import \
    build_scene_manifest, write_scene_manifest
from predicators.envs import create_new_env
from predicators.settings import CFG
from predicators.structs import State

if TYPE_CHECKING:  # pragma: no cover
    from predicators.agent_sdk.tools.context import ToolContext
    from predicators.structs import Task


class ScenePackageMixin:
    """Builds the scene package for an agent approach's sandbox."""

    if TYPE_CHECKING:  # pragma: no cover
        _tool_context: ToolContext
        _train_tasks: List[Task]

        def _get_log_dir(self) -> str:
            ...

    # (bodies, asset files) of the last scene manifest, for the prompt.
    _scene_package_summary: Tuple[int, int] = (0, 0)

    def _scene_state(self) -> State:
        """The initial state of the level being played (the run's first train
        task before any level starts): what the manifest describes."""
        task = self._tool_context.current_task
        if task is None:
            task = self._train_tasks[0]
        return task.init

    def _scene_manifest_env(self) -> Tuple[Any, bool]:
        """A world of the deployment's visible physics (no hidden mechanism) to
        describe, and whether the caller owns it and must dispose of it."""
        return create_new_env(CFG.env,
                              do_cache=False,
                              use_gui=False,
                              skip_residual_dynamics=True), True

    @staticmethod
    def _standalone_source(name: str, directory: Path) -> Path:
        """A copy of one engine module whose imports point at the copies beside
        it, so the reference reads as a self-contained package."""
        package = Path(__file__).resolve().parents[1]
        sources = {
            "pybullet_env.py": package / "envs" / "pybullet_env.py",
            "scene_base.py": package / "code_sim_learning" / "scene_base.py",
        }
        text = sources[name].read_text(encoding="utf-8")
        rebinds = {
            "from predicators.envs import BaseEnv\n":
            "from reference.base_sim.base_env import BaseEnv\n",
            "from predicators.envs.pybullet_env import PyBulletEnv\n":
            "from reference.base_sim.pybullet_env import PyBulletEnv\n",
        }
        for original, replacement in rebinds.items():
            if text.count(original) == 1:
                text = text.replace(original, replacement)
        target = directory / name
        target.write_text(text, encoding="utf-8")
        return target

    def _scene_package_files(self) -> Dict[str, str]:
        """Sandbox reference path -> source file of the engine wrapper, the
        scene manifest of the level being played and the asset files its bodies
        were loaded from."""
        package = Path(__file__).resolve().parents[1]
        directory = Path(self._get_log_dir()) / "reference_sources"
        directory.mkdir(parents=True, exist_ok=True)
        files = {
            "base_sim/base_env.py":
            str(package / "envs" / "base_env.py"),
            "base_sim/pybullet_env.py":
            str(self._standalone_source("pybullet_env.py", directory)),
        }
        env, owned = self._scene_manifest_env()
        try:
            manifest, assets = build_scene_manifest(env, self._scene_state())
        finally:
            if owned:
                dispose_env(env)
        files["scene/scene_manifest.json"] = write_scene_manifest(
            manifest, str(directory / "scene_manifest.json"))
        files.update(assets)
        self._scene_package_summary = (len(manifest["bodies"]), len(assets))
        return files

    def _scene_package_paths(self) -> List[str]:
        """Agent-visible paths of :meth:`_scene_package_files`."""
        bodies, assets = self._scene_package_summary
        return [
            "./reference/base_sim/pybullet_env.py",
            "./reference/base_sim/base_env.py",
            f"./reference/scene/scene_manifest.json ({bodies} bodies)",
            f"./reference/assets/ ({assets} URDF and mesh files, named in "
            "the manifest)",
        ]
