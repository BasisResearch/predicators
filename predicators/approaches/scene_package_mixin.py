"""The scene package: the engine wrapper, the scene manifest and the asset
files the scene's bodies were loaded from, as read-only sandbox references.

The agentic real-to-sim arm builds its scene from them, and EMPIRIC can
receive them beside its twin (``continual_provide_scene_package``). The
direct agent and the standalone model get the manifest and the asset
files alone (the scene assets): neither has a simulator of any kind, and
each may use them however it likes.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from predicators.agent_sdk.sandbox_setup import GeneratedReference, \
    ReferenceFiles
from predicators.code_sim_learning.rollout_env import dispose_env
from predicators.code_sim_learning.scene_loader import static_signature
from predicators.code_sim_learning.scene_manifest import GeneratedAsset, \
    build_scene_manifest, manifest_asset_files, scene_manifest_text
from predicators.envs import create_new_env
from predicators.run import paths
from predicators.settings import CFG
from predicators.structs import State

if TYPE_CHECKING:  # pragma: no cover
    from predicators.agent_sdk.tools.context import ToolContext
    from predicators.structs import Task

# Agent-visible paths of the simulator's source
# (ScenePackageMixin._simulator_source_files), as a prompt lists them.
SIMULATOR_SOURCE_PATHS = (
    "./reference/base_sim/scene_base.py",
    "./reference/base_sim/pybullet_env.py",
    "./reference/base_sim/base_env.py",
)

# The prompt's list entry of the per-level scene files, for the arms that
# get them under continual_provide_level_scenes: what one file holds.
_LEVEL_SCENES_ENTRY = (
    "./reference/scene/levels/level_NN.json (one file per level played so "
    "far: that level's scene manifest and the exact features each observed "
    "object with a static base had as the level began)")


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
        origin = sources[name].relative_to(Path(__file__).resolve().parents[2])
        return GeneratedReference(
            text, f"{origin}, its imports pointed at reference/base_sim")

    def _scene_package_files(self) -> ReferenceFiles:
        """Sandbox reference path -> the engine wrapper and the scene
        assets."""
        package = Path(__file__).resolve().parents[1]
        files: ReferenceFiles = {
            "base_sim/base_env.py": str(package / "envs" / "base_env.py"),
            "base_sim/pybullet_env.py":
            self._standalone_source("pybullet_env.py"),
        }
        files.update(self._scene_asset_files())
        return files

    def _simulator_source_files(self) -> ReferenceFiles:
        """Sandbox reference path -> the source of the scene simulator, as
        EMPIRIC from assets reads it: the scene base and the engine wrapper
        under it, their imports pointed at the copies beside them.

        They are copies to read. A simulator the harness builds loads
        the harness's own modules, so no edit of these files reaches it.
        """
        package = Path(__file__).resolve().parents[1]
        return {
            "base_sim/base_env.py": str(package / "envs" / "base_env.py"),
            "base_sim/pybullet_env.py":
            self._standalone_source("pybullet_env.py"),
            "base_sim/scene_base.py": self._standalone_source("scene_base.py"),
        }

    def _scene_asset_files(self) -> ReferenceFiles:
        """Sandbox reference path -> the scene manifest of the level being
        played and the asset files its bodies were loaded from.

        During continual play the manifest is also kept in the level's
        recording directory, so the run records what each level showed.
        """
        env, owned = self._scene_manifest_env()
        state = self._scene_state()
        try:
            manifest, assets = build_scene_manifest(env, state)
        finally:
            if owned:
                dispose_env(env)
        manifest["legend"] = self._scene_manifest_legend(manifest["legend"])
        text = scene_manifest_text(manifest)
        files: ReferenceFiles = {
            "scene/scene_manifest.json":
            GeneratedReference(text, "the scene of the level being played")
        }
        files.update({
            path: (GeneratedReference(source.text, source.origin)
                   if isinstance(source, GeneratedAsset) else source)
            for path, source in assets.items()
        })
        self._scene_package_summary = (len(manifest["bodies"]), len(assets))
        self._record_level_manifest(text)
        if CFG.scene_manifest_loader or CFG.continual_provide_level_scenes:
            files.update(self._level_scene_files(manifest, assets, state))
        return files

    def _scene_manifest_legend(self, legend: Dict[str, str]) -> Dict[str, str]:
        """The manifest's legend as this arm's files carry it.

        The shared legend tells a reader how to build the scene on the
        base simulator. An arm that has no base simulator overrides this
        (:func:`files_only_legend`), so its manifest names none.
        """
        return legend

    def _level_scene_files(self, manifest: Dict[str, Any], assets: Dict[str,
                                                                        Any],
                           state: State) -> ReferenceFiles:
        """The scene of every level played so far, for the scene loader:
        ``scene/levels/level_NN.json`` (a level's manifest and the features its
        static objects showed as it began) and the files those manifests name.

        The fit replays every level's recordings, each in its own scene.
        A level's record is kept in its recording directory, so a
        resumed run has the scenes of the levels before it.
        """
        session = getattr(self, "_play_session", None)
        index = 0 if session is None else int(session.level_index)
        if session is not None and CFG.scene_loader_static_poses:
            # Where the level's scene has its static objects, as it has
            # their sizes: not where one noisy observation shows them.
            # Only the objects with a static base are read from this
            # state (static_signature); the agent's observations of every
            # object stay the noisy ones.
            state = session.levels[index].task.init
        record: Dict[str, Any] = {
            "level": index + 1,
            "manifest": manifest,
            "signature": static_signature(manifest, state),
            "generated": {
                path: source.text
                for path, source in assets.items()
                if isinstance(source, GeneratedAsset)
            },
        }
        records: Dict[int, Dict[str, Any]] = {index: record}
        if session is not None:
            run_dir = paths.run_dir()
            level_dir = paths.level_dir(run_dir, index)
            os.makedirs(level_dir, exist_ok=True)
            with open(os.path.join(level_dir, "scene_level.json"),
                      "w",
                      encoding="utf-8") as f:
                json.dump(record, f)
            for earlier in range(index):
                path = os.path.join(paths.level_dir(run_dir, earlier),
                                    "scene_level.json")
                if os.path.isfile(path):
                    with open(path, encoding="utf-8") as f:
                        records[earlier] = json.load(f)
        files: ReferenceFiles = {}
        for level, kept in sorted(records.items()):
            label = f"level {level + 1}"
            files[f"scene/levels/level_{level + 1:02d}.json"] = \
                GeneratedReference(
                    json.dumps(
                        {
                            "manifest": kept["manifest"],
                            "signature": kept["signature"]
                        },
                        indent=1) + "\n", f"the scene of {label}")
            for path, mesh in kept["generated"].items():
                files.setdefault(
                    path,
                    GeneratedReference(
                        mesh, f"a mesh the deployment builds in memory "
                        f"for {label}"))
            for path, source in manifest_asset_files(kept["manifest"]).items():
                files.setdefault(path, source)
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
        """Agent-visible paths of :meth:`_scene_package_files`."""
        return [
            "./reference/base_sim/pybullet_env.py",
            "./reference/base_sim/base_env.py",
        ] + self._scene_asset_paths()

    def _scene_asset_paths(self) -> List[str]:
        """Agent-visible paths of :meth:`_scene_asset_files`, with the
        manifest's body and file counts once one has been built."""
        manifest = "./reference/scene/scene_manifest.json"
        assets = "./reference/assets/"
        if self._scene_package_summary is not None:
            bodies, files = self._scene_package_summary
            manifest += f" ({bodies} bodies)"
            assets += f" ({files} URDF and mesh files, named in the manifest)"
        if not CFG.continual_provide_level_scenes:
            return [manifest, assets]
        # The arms that get the scene of each level by this flag alone
        # read what one level's file holds here.
        return [manifest, _LEVEL_SCENES_ENTRY, assets]
