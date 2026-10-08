"""The scene package: the engine wrapper, the scene manifest and the asset
files the scene's bodies were loaded from, as read-only sandbox references.

The agentic real-to-sim arm builds its scene from them, EMPIRIC can
receive them beside its twin (``continual_provide_scene_package``), and
so can the direct agent, which gets no simulator of any kind and may use
them however it likes.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

from predicators.agent_sdk.sandbox_setup import GeneratedReference, \
    ReferenceFiles
from predicators.code_sim_learning.rollout_env import dispose_env
from predicators.code_sim_learning.scene_manifest import \
    build_scene_manifest, scene_manifest_text
from predicators.envs import create_new_env
from predicators.run import paths
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

    # (bodies, asset files) of the last scene manifest built, for the
    # prompt's reference list; None until the first one is built.
    _scene_package_summary: Optional[Tuple[int, int]] = None

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
    def _standalone_source(name: str) -> GeneratedReference:
        """One engine module with its imports pointed at the copies beside it
        in ``reference/base_sim/``, so the reference reads as a self-contained
        package."""
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
        origin = sources[name].relative_to(package.parent)
        return GeneratedReference(
            text, f"{origin}, its imports pointed at reference/base_sim")

    def _scene_package_files(self) -> ReferenceFiles:
        """Sandbox reference path -> the engine wrapper, the scene manifest of
        the level being played and the asset files its bodies were loaded from.

        During continual play the manifest is also kept in the level's
        recording directory, so the run records what each level showed.
        """
        package = Path(__file__).resolve().parents[1]
        files: ReferenceFiles = {
            "base_sim/base_env.py": str(package / "envs" / "base_env.py"),
            "base_sim/pybullet_env.py":
            self._standalone_source("pybullet_env.py"),
        }
        env, owned = self._scene_manifest_env()
        try:
            manifest, assets = build_scene_manifest(env, self._scene_state())
        finally:
            if owned:
                dispose_env(env)
        text = scene_manifest_text(manifest)
        files["scene/scene_manifest.json"] = GeneratedReference(
            text, "the scene of the level being played")
        files.update(assets)
        self._scene_package_summary = (len(manifest["bodies"]), len(assets))
        self._record_level_manifest(text)
        return files

    def _record_level_manifest(self, text: str) -> None:
        """Write the manifest into the recording directory of the level being
        played; outside continual play there is none."""
        session = getattr(self, "_play_session", None)
        if session is None:
            return
        level_dir = paths.level_dir(paths.run_dir(), session.level_index)
        os.makedirs(level_dir, exist_ok=True)
        with open(os.path.join(level_dir, "scene_manifest.json"),
                  "w",
                  encoding="utf-8") as f:
            f.write(text)

    def _scene_package_paths(self) -> List[str]:
        """Agent-visible paths of :meth:`_scene_package_files`, with the
        manifest's body and file counts once one has been built."""
        manifest = "./reference/scene/scene_manifest.json"
        assets = "./reference/assets/"
        if self._scene_package_summary is not None:
            bodies, files = self._scene_package_summary
            manifest += f" ({bodies} bodies)"
            assets += f" ({files} URDF and mesh files, named in the manifest)"
        return [
            "./reference/base_sim/pybullet_env.py",
            "./reference/base_sim/base_env.py",
            manifest,
            assets,
        ]
