"""Blender Cycles scenes of recorded runs and trajectories.

PyBullet draws the run videos as a run ends (``continual_video.py``).
Blender Cycles draws them with the paper figures' materials, lighting and
cameras, but it needs Blender's own Python (bpy 4.5 on Python 3.11) and
takes about three seconds a frame on a GPU and far longer on a CPU, so no
run renders with it. This module only exports: every frame of a video
becomes a scene file, the visual shapes of a restored state and the
camera, beside a manifest that names each frame, how many video frames it
is held for, and the panel label the run video draws for it.
``scripts/cycles_video.py`` renders the scenes with Blender on a GPU node
and assembles the videos (``compose_video``).

Under ``video_cycles_scenes``, test and failure videos record a scene per
frame as their frames are rendered (``CyclesSceneMonitor``), into
``<video_dir>/<video name>_cycles``, and a finished continual run exports
its video's frames from the recorded states into ``<run_dir>/cycles``
(``export_run_scenes``); ``scripts/continual_video.py --cycles`` does the
same for an earlier run, and ``export_state_scenes`` exports any recorded
sequence of states, such as an evaluation trajectory.

Restoring a state skips what only stepping draws, so the export of
recorded states redraws it: Boil's spill puddle and the heat of its
water, which the export accumulates step by step with the environment's
rule and checks against the recorded bubbling level, and Balloons'
strings. Boil water is drawn
no higher than the jug rim, where the environment lets it rise above the
rim before it overflows.
"""
from __future__ import annotations

import atexit
import copy
import dataclasses
import gzip
import hashlib
import json
import logging
import os
import pickle
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, \
    Tuple
from unittest.mock import patch

import imageio
import numpy as np
import pybullet as p
from PIL import Image

from predicators import utils
from predicators.envs import BaseEnv, create_new_env
from predicators.run import paths
from predicators.run.continual_video import FrameLabel, compose_frame, \
    iter_recorded_level_frames, read_level_episodes
from predicators.run.recording import EPISODES_FILENAME
from predicators.run.scorecard import RunCard
from predicators.settings import CFG
from predicators.structs import Action, Object, Observation, State

# Moves a restored state into a newer scene layout: (env, state, the
# level's freshly reset state) -> state. The paper figures use it for runs
# recorded before the Fan and Balloons layouts changed.
Adjust = Callable[[BaseEnv, State, State], State]
GENERATED_BY = "predicators/run/cycles_video.py; do not edit manually"

# (client, body, link) -> the vertices and indices of a procedural visual.
_PROCEDURAL_MESHES: Dict[Tuple[int, int, int], Dict[str, Any]] = {}
# Mesh file -> SHA-256, so a video's thousands of exports read each mesh once.
_ASSET_SHA256: Dict[str, str] = {}
# How many record_procedural_meshes contexts are open; only the outermost
# clears the registry, so a nested export keeps the outer one's meshes.
_RECORDING = {"depth": 0}
# Holds the recording that keep_procedural_meshes opens for the process.
_KEPT = ExitStack()
_KEPT_OPEN = {"open": False}


@contextmanager
def record_procedural_meshes() -> Iterator[None]:
    """Capture in-memory visuals that getVisualShapeData cannot serialize.

    Wrap the creation of the environment whose scenes are exported.
    """
    shapes: Dict[Tuple[int, int], Dict[str, Any]] = {}
    create_shape, create_body = p.createVisualShape, p.createMultiBody

    def shape(*args: Any, **kwargs: Any) -> int:
        result = create_shape(*args, **kwargs)
        if 'vertices' in kwargs:
            shapes[(kwargs.get('physicsClientId',
                               0), result)] = copy.deepcopy({
                                   'vertices':
                                   kwargs['vertices'],
                                   'indices':
                                   kwargs['indices']
                               })
        return result

    def body(*args: Any, **kwargs: Any) -> int:
        result = create_body(*args, **kwargs)
        client = kwargs.get('physicsClientId', 0)
        for key in list(_PROCEDURAL_MESHES):
            if key[:2] == (client, result):
                del _PROCEDURAL_MESHES[key]
        visuals = [kwargs.get('baseVisualShapeIndex', -1)] + list(
            kwargs.get('linkVisualShapeIndices', []))
        for link, visual in enumerate(visuals, start=-1):
            if (client, visual) in shapes:
                _PROCEDURAL_MESHES[(client, result, link)] = shapes[(client,
                                                                     visual)]
        return result

    if _RECORDING["depth"] == 0:
        _PROCEDURAL_MESHES.clear()
    _RECORDING["depth"] += 1
    try:
        with patch.object(p, 'createVisualShape',
                          shape), patch.object(p, 'createMultiBody', body):
            yield
    finally:
        _RECORDING["depth"] -= 1
        if _RECORDING["depth"] == 0:
            _PROCEDURAL_MESHES.clear()


def keep_procedural_meshes() -> None:
    """Record procedural meshes for the rest of the process.

    main.py calls it under ``video_cycles_scenes`` before it creates any
    env, so live scene exports can serialize every visual. The recording
    closes at exit, while this module is intact; left to the
    interpreter's teardown, its cleanup would raise on the cleared
    module globals.
    """
    if not _KEPT_OPEN["open"]:
        _KEPT.enter_context(record_procedural_meshes())
        _KEPT_OPEN["open"] = True
        atexit.register(_KEPT.close)


def scene_signature(client: int) -> Dict[int, Dict[str, Any]]:
    """Capture poses and joints to verify that a camera capture is passive."""
    return {
        body: {
            'base':
            p.getBasePositionAndOrientation(body, physicsClientId=client),
            'joints': [
                p.getJointState(body, j, physicsClientId=client)[:2]
                for j in range(p.getNumJoints(body, physicsClientId=client))
            ],
        }
        for body in [
            p.getBodyUniqueId(i, physicsClientId=client)
            for i in range(p.getNumBodies(physicsClientId=client))
        ]
    }


def export_visual_scene(client: int, view: Sequence[float],
                        projection: Sequence[float], width: int, height: int,
                        metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Export visual primitives/mesh references from a client WITHOUT EGL.

    The installed EGL plugin returns stale per-instance colors and accumulated
    geometry through getVisualShapeData. Never export a plugin-backed client.

    Bullet stores base poses at the inertial frame and visual offsets at the
    link frame. Undo that inertial offset before composing a base visual pose.
    Child links expose their world link frame directly via getLinkState.
    """
    shapes: List[Dict[str, Any]] = []
    assets: Dict[str, str] = {}
    for i in range(p.getNumBodies(physicsClientId=client)):
        body = p.getBodyUniqueId(i, physicsClientId=client)
        name = p.getBodyInfo(body, physicsClientId=client)[1].decode()
        for index, shape in enumerate(
                p.getVisualShapeData(body, physicsClientId=client)):
            values = shape[:8]
            _, link, kind, dimensions, mesh, local_pos, local_quat, rgba = \
                values
            if link == -1:
                pos, quat = p.getBasePositionAndOrientation(
                    body, physicsClientId=client)
                dynamics = p.getDynamicsInfo(body, -1, physicsClientId=client)
                inv_pos, inv_quat = p.invertTransform(dynamics[3], dynamics[4])
                frame = p.multiplyTransforms(pos, quat, inv_pos, inv_quat)
            else:
                frame = p.getLinkState(body,
                                       link,
                                       computeForwardKinematics=True,
                                       physicsClientId=client)[4:6]
            pos, quat = p.multiplyTransforms(*frame, local_pos, local_quat)
            filename = mesh.decode() if kind == p.GEOM_MESH else ''
            if filename:
                path = Path(filename)
                if not path.is_file():
                    raise FileNotFoundError(
                        f'Cannot export visual mesh: {filename}')
                if filename not in _ASSET_SHA256:
                    _ASSET_SHA256[filename] = hashlib.sha256(
                        path.read_bytes()).hexdigest()
                assets[filename] = _ASSET_SHA256[filename]
            shapes.append(
                dict(body=body,
                     link=link,
                     index=index,
                     name=name,
                     kind=kind,
                     dimensions=dimensions,
                     mesh=filename,
                     position=pos,
                     quaternion_xyzw=quat,
                     rgba=rgba))
            if kind == p.GEOM_MESH and not filename:
                shapes[-1].update(_PROCEDURAL_MESHES[(client, body, link)])
    return dict(
        generated_by='scripts/render_egl_scenes.py; do not edit manually',
        metadata=metadata,
        camera=dict(view=list(view),
                    projection=list(projection),
                    width=width,
                    height=height),
        shapes=shapes,
        mesh_sha256=assets)


@contextmanager
def raised_flat_markers(client: int) -> Iterator[None]:
    """Draw coplanar target decals above the tabletop to avoid EGL z-fighting.

    These temporary visual-only copies do not move the original physical
    body.
    """
    temporary: List[int] = []
    try:
        for body in list(scene_signature(client)):
            for shape in p.getVisualShapeData(body, physicsClientId=client):
                _, link, kind, dims, _, local_pos, local_quat, rgba = shape[:8]
                if link != -1 or kind != p.GEOM_BOX or not (0 < dims[2] <
                                                            .001):
                    continue
                if min(dims[:2]) < .01 or rgba[3] < .99:
                    continue
                body_pos, body_quat = p.getBasePositionAndOrientation(
                    body, physicsClientId=client)
                pos, quat = p.multiplyTransforms(body_pos, body_quat,
                                                 local_pos, local_quat)
                lifted = (pos[0], pos[1], pos[2] + .0008)
                visual = p.createVisualShape(p.GEOM_BOX,
                                             halfExtents=[v / 2 for v in dims],
                                             rgbaColor=rgba,
                                             physicsClientId=client)
                temporary.append(
                    p.createMultiBody(baseMass=0,
                                      baseCollisionShapeIndex=-1,
                                      baseVisualShapeIndex=visual,
                                      basePosition=lifted,
                                      baseOrientation=quat,
                                      physicsClientId=client))
        yield
    finally:
        for body in temporary:
            p.removeBody(body, physicsClientId=client)


def canonical_state(env: BaseEnv, state: State) -> State:
    """Rebind objects in a historical state to the new physics client."""
    canonical = {}
    for value in vars(env).values():
        values = (value.values() if isinstance(value, dict) else
                  value if isinstance(value, (list, tuple, set)) else [value])
        for obj in values:
            if isinstance(obj, Object):
                canonical[obj.name] = obj
    restored = state.copy()
    restored.data = {}
    for obj, features in state.data.items():
        current = canonical.get(obj.name)
        assert current is not None and current.type == obj.type, (obj, current)
        restored.data[current] = features.copy()
    return restored


def cap_liquid_at_rim(env: Any, state: State) -> State:
    """Draw Boil liquid no higher than the jug rim."""
    # The liquid starts at the jug's inner bottom, _LIQUID_OFFSET_BELOW_JUG
    # below the jug origin; the rim is half the jug height above it.
    rim = (
        env.jug_height / 2 + env._LIQUID_OFFSET_BELOW_JUG  # pylint: disable=protected-access
    ) * env.water_height_to_level_ratio
    for jug in state.get_objects(env._jug_type):  # pylint: disable=protected-access
        state.set(jug, "water_volume", min(state.get(jug, "water_volume"),
                                           rim))
    return state


def restore_spill(env: Any, state: State) -> None:
    """Draw the recorded spill puddle, which restoring a state omits.

    The environment builds its puddle only while stepping, so a restored
    state with spilled water would otherwise render a dry table.
    """
    faucet = env._faucet  # pylint: disable=protected-access
    spilled = state.get(faucet, "spilled_level")
    if spilled > 0:
        faucet._spilled_level = spilled  # pylint: disable=protected-access
        env._spilled_water_id = env._create_spilled_water_block(  # pylint: disable=protected-access
            spilled, state)


def boil_heat(env: Any, states: Sequence[State]) -> List[Dict[str, float]]:
    """Each jug's heat after every recorded step of one Boil episode.

    Mirrors PyBulletBoilEnv._handle_heating_logic: a jug that holds
    water, is not held, and sits within burner_align_threshold of a
    burner that was on before and after the step gains heating_speed.
    Where the recorded bubbling level is positive it fixes the heat, and
    the accumulated heat must agree with it.
    """
    heat: Dict[str, float] = {}
    track = []
    required = (env.water_filled_height
                if CFG.boil_require_jug_full_to_heatup else 0.)
    for k, state in enumerate(states):
        jugs = state.get_objects(env._jug_type)  # pylint: disable=protected-access
        if k > 0:
            for burner in state.get_objects(env._burner_type):  # pylint: disable=protected-access
                # A burner's is_on mirrors its switch, which the environment
                # reads before (prev_on) and after the step.
                if not (state.get(burner, "is_on") > 0.5
                        and states[k - 1].get(burner, "is_on") > 0.5):
                    continue
                for jug in jugs:
                    dist = float(
                        np.hypot(
                            state.get(burner, "x") - state.get(jug, "x"),
                            state.get(burner, "y") - state.get(jug, "y")))
                    held = env._Holding_holds(state, [env._robot, jug])  # pylint: disable=protected-access
                    if (dist < env.burner_align_threshold
                            and state.get(jug, "water_volume") > required
                            and not held):
                        heat[jug.name] = min(
                            1.,
                            heat.get(jug.name, 0.) + env.heating_speed)
        for jug in jugs:
            bubbling = state.get(jug, "bubbling_level")
            if bubbling > 0:
                recorded = env.BUBBLING_THRESHOLD + bubbling / env.BUBBLING_RAMP
                assert abs(heat.get(jug.name, 0.) - recorded) < 1e-6, (
                    k, jug.name, heat.get(jug.name, 0.), recorded)
        track.append(dict(heat))
    return track


class SceneWriter:
    """Exports recorded states as Cycles scenes into ``<out>/scenes``.

    Call ``episode`` with each recorded episode's states first: scenes
    are named by the step they show (``L<level>_E<episode>_S<step>``),
    so two exports of the same steps share their files. The writer is
    the ``show`` callback of ``iter_recorded_level_frames``.
    """

    def __init__(self,
                 env: BaseEnv,
                 out: Path,
                 adjust: Optional[Adjust] = None) -> None:
        self.env, self.out, self.adjust = env, Path(out), adjust
        self.heat_of: Dict[int, Dict[str, float]] = {}
        self.step_of: Dict[int, str] = {}
        (self.out / "scenes").mkdir(parents=True, exist_ok=True)

    def episode(self, states: Sequence[State], level: int, index: int) -> None:
        """Name one episode's steps and prepare their display data."""
        for k, state in enumerate(states):
            self.step_of[id(state)] = f"L{level:02d}_E{index:03d}_S{k:05d}"
        if CFG.env == "pybullet_boil":
            for state, heat in zip(states, boil_heat(self.env, states)):
                self.heat_of[id(state)] = heat

    def __call__(self, recorded: State, initial: State) -> str:
        env: Any = self.env
        state = canonical_state(env, recorded)
        if self.adjust is not None:
            state = self.adjust(env, state, initial)
        boil = CFG.env == "pybullet_boil"
        if boil:
            state = cap_liquid_at_rim(env, state)
        env._set_state(state)  # pylint: disable=protected-access
        env._current_observation = state  # pylint: disable=protected-access
        client = env._physics_client_id  # pylint: disable=protected-access
        if boil:
            if getattr(env, "_spilled_water_id", None) is not None:
                p.removeBody(env._spilled_water_id, physicsClientId=client)  # pylint: disable=protected-access
                env._spilled_water_id = None  # pylint: disable=protected-access
            restore_spill(env, state)
            env._heat_levels.clear()  # pylint: disable=protected-access
            env._heat_levels.update(self.heat_of[id(recorded)])  # pylint: disable=protected-access
            env._update_liquid_colors(state)  # pylint: disable=protected-access
        if hasattr(env, "_sync_cable_visuals"):
            # Balloons draws the strings of tied balloons only when it
            # renders an image.
            env._sync_cable_visuals()  # pylint: disable=protected-access
        name = self.step_of[id(recorded)]
        scene = _export_now(env, dict(env=CFG.env, step=name))
        with gzip.open(self.out / "scenes" / f"{name}.json.gz",
                       "wt",
                       encoding="utf-8") as f:
            json.dump(scene, f, separators=(",", ":"))
        return f"{name}.json.gz"


def _export_now(env: Any, metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Export the env's current visual scene, checking that nothing moves."""
    client = env._physics_client_id  # pylint: disable=protected-access
    before = scene_signature(client)
    with patch.object(
            p,
            "stepSimulation",
            side_effect=AssertionError("Rendering must not step physics")):
        with raised_flat_markers(client):
            # pylint: disable-next=protected-access
            view, projection, width, height = env._get_camera_matrices()
            scene = export_visual_scene(client, view, projection, width,
                                        height, metadata)
    assert scene_signature(client) == before
    return scene


class LiveSceneRecorder:
    """Records a PyBullet env's current scene as one video frame at a time.

    For code that records a video as it runs, such as test videos and
    scripted solves: each ``capture`` exports what the env draws at that
    moment, so nothing a restored state would miss is lost. The env must
    have been created inside ``record_procedural_meshes`` (main.py opens
    one under ``video_cycles_scenes``). Scenes stay compressed in memory
    until ``save``.
    """

    def __init__(self, env: BaseEnv, fps: int) -> None:
        self.env, self.fps = env, fps
        self.scenes: List[bytes] = []
        self.repeats: List[int] = []

    def capture(self, repeat: int = 1) -> None:
        """Record the env's current scene, shown for ``repeat`` frames."""
        scene = _export_now(self.env, dict(env=CFG.env,
                                           frame=len(self.scenes)))
        self.scenes.append(
            gzip.compress(
                json.dumps(scene, separators=(",", ":")).encode("utf-8")))
        self.repeats.append(repeat)

    def save(self, out_dir: str, name: str, hold_last: int = 0) -> Path:
        """Write the scenes and their manifest to ``out_dir``."""
        out = Path(out_dir)
        (out / "scenes").mkdir(parents=True, exist_ok=True)
        frames: List[Dict[str, Any]] = []
        for k, (scene, repeat) in enumerate(zip(self.scenes, self.repeats)):
            scene_name = f"F{k:06d}.json.gz"
            (out / "scenes" / scene_name).write_bytes(scene)
            last = k + 1 == len(self.scenes)
            frames.append(
                dict(scene=scene_name,
                     repeat=repeat + (hold_last if last else 0),
                     level=0,
                     split="test"))
        return _write_manifest(
            out, "manifest-all.json",
            dict(generated_by=GENERATED_BY,
                 domain=name,
                 env=CFG.env,
                 levels="all",
                 fps=self.fps,
                 width=CFG.pybullet_camera_width,
                 height=CFG.pybullet_camera_height,
                 frames=frames))


@dataclass
class CyclesSceneMonitor(utils.LoggingMonitor):
    """Wraps a video monitor and records a Cycles scene of every frame it
    renders.

    ``save`` writes them beside the video, to ``<video_dir>/<video
    name>_cycles``.
    """
    inner: utils.LoggingMonitor
    env: BaseEnv
    _recorder: Optional[LiveSceneRecorder] = field(init=False, default=None)

    def reset(self, train_or_test: str, task_idx: int) -> None:
        self.inner.reset(train_or_test, task_idx)
        self._recorder = LiveSceneRecorder(self.env, int(CFG.video_fps))

    def observe(self, obs: Observation, action: Optional[Action]) -> None:
        self.inner.observe(obs, action)
        assert self._recorder is not None
        self._recorder.capture()

    def save(self, video_name: str) -> Optional[Path]:
        """Write the episode's scenes beside the video ``video_name``."""
        if self._recorder is None or not self._recorder.scenes:
            return None
        stem = video_name[:-len(".mp4")] if video_name.endswith(
            ".mp4") else video_name
        return self._recorder.save(os.path.join(CFG.video_dir,
                                                f"{stem}_cycles"),
                                   name=os.path.basename(stem))


def _write_manifest(out: Path, name: str, manifest: Dict[str, Any]) -> Path:
    path = out / name
    path.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    return path


def export_run_scenes(card: RunCard,
                      run_dir: str,
                      out_dir: Optional[str] = None,
                      levels: str = "all",
                      adjust: Optional[Adjust] = None,
                      name: Optional[str] = None) -> Path:
    """Export one scene per frame of the run video of ``run_dir``.

    The frames, their order, their labels and how long each is held are
    the run video's (``iter_recorded_level_frames``); ``levels="test"``
    keeps only the last level, the test task. CFG must be the run's
    configuration, as it is at run end. Writes
    ``<out_dir>/manifest-<levels>.json`` (``out_dir`` defaults to the
    run's ``cycles`` directory) and returns its path.
    """
    stride = int(CFG.continual_video_stride)
    fps = int(CFG.video_fps)
    out = Path(out_dir or paths.cycles_dir(run_dir))
    chosen = [
        lv for lv in card.levels
        if lv.attempted and os.path.isdir(paths.level_dir(run_dir, lv.index))
    ]
    if levels == "test":
        chosen = chosen[-1:]
        assert chosen and chosen[0].split == "test", run_dir
    frames: List[Dict[str, Any]] = []
    hold = max(1, fps)
    with record_procedural_meshes():
        env = create_new_env(CFG.env, do_cache=False, use_gui=False)
        try:
            writer = SceneWriter(env, out, adjust)
            for level in chosen:
                level_dir = paths.level_dir(run_dir, level.index)
                episodes = read_level_episodes(level_dir)
                with open(os.path.join(level_dir, EPISODES_FILENAME),
                          "rb") as f:
                    # Trusted local experiment record.
                    recorded = {
                        int(ep["episode"]): ep
                        for ep in pickle.load(f)
                    }
                for index, ep in recorded.items():
                    writer.episode(ep["states"], level.index, index)
                for scene, label, repeat in iter_recorded_level_frames(
                        env, card, level, episodes, recorded, writer, stride,
                        hold):
                    frames.append(
                        dict(scene=scene,
                             repeat=repeat,
                             level=level.index,
                             split=level.split,
                             label=dataclasses.asdict(label)))
                logging.info("[Cycles] level %d: %d frames so far",
                             level.index + 1, len(frames))
        finally:
            env.dispose()
    scorecard = Path(paths.scorecard_path(run_dir))
    manifest = dict(generated_by=GENERATED_BY,
                    domain=name or CFG.env.replace("pybullet_", ""),
                    env=CFG.env,
                    run=str(run_dir),
                    scorecard_sha256=hashlib.sha256(
                        scorecard.read_bytes()).hexdigest(),
                    levels=levels,
                    stride=stride,
                    fps=fps,
                    width=CFG.pybullet_camera_width,
                    height=CFG.pybullet_camera_height,
                    frames=frames)
    path = _write_manifest(out, f"manifest-{levels}.json", manifest)
    logging.info("[Cycles] exported %d frames to %s", len(frames), out)
    return path


def export_state_scenes(states: Sequence[State],
                        out_dir: str,
                        fps: int,
                        name: str,
                        split: str = "test",
                        task_idx: int = 0,
                        hold_last: int = 0,
                        adjust: Optional[Adjust] = None) -> Path:
    """Export one scene per state of a recorded sequence, such as an evaluation
    trajectory, in the env CFG configures.

    The env is reset to ``split`` task ``task_idx`` before the states
    are restored; each state is one video frame, and the last is held
    for ``hold_last`` more. Writes ``<out_dir>/manifest-all.json``.
    """
    out = Path(out_dir)
    frames: List[Dict[str, Any]] = []
    with record_procedural_meshes():
        env = create_new_env(CFG.env, do_cache=False, use_gui=False)
        try:
            initial = env.reset(split, task_idx)
            writer = SceneWriter(env, out, adjust)
            writer.episode(states, 0, 0)
            for k, state in enumerate(states):
                repeat = 1 + (hold_last if k + 1 == len(states) else 0)
                frames.append(
                    dict(scene=writer(state, initial),
                         repeat=repeat,
                         level=0,
                         split=split))
        finally:
            env.dispose()
    manifest = dict(generated_by=GENERATED_BY,
                    domain=name,
                    env=CFG.env,
                    levels="all",
                    fps=fps,
                    width=CFG.pybullet_camera_width,
                    height=CFG.pybullet_camera_height,
                    frames=frames)
    return _write_manifest(out, "manifest-all.json", manifest)


def compose_video(manifest_path: Path,
                  panel: bool = False,
                  test_only: bool = False,
                  output: Optional[Path] = None) -> Path:
    """Assemble the video of a manifest from its rendered frames.

    ``panel`` draws the run video's label panel beside each frame;
    ``test_only`` keeps only the run's last level. Frames come from
    ``frames/<scene>.png`` beside the manifest, as
    ``scripts/paper_figures/render_cycles_frames.py`` writes them.
    """
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    root = Path(manifest_path).parent
    chosen = manifest["frames"]
    if test_only:
        last = chosen[-1]["level"]
        assert chosen[-1]["split"] == "test", manifest_path
        chosen = [f for f in chosen if f["level"] == last]
    stem = manifest["domain"] + ("-test" if test_only else "") + \
        ("" if panel else "-scene")
    output = Path(output or root / f"{stem}.mp4")
    frames = root / "frames"
    missing = [
        f["scene"] for f in chosen
        if not (frames / f["scene"].replace(".json.gz", ".png")).exists()
    ]
    assert not missing, f"{len(missing)} frames not rendered: {missing[0]}"
    # A higher quality than the run video's: path-traced gradients band at
    # its setting.
    writer = imageio.get_writer(str(output),
                                fps=int(manifest["fps"]),
                                codec="libx264",
                                macro_block_size=1,
                                output_params=[
                                    "-crf", "16", "-preset", "slow",
                                    "-movflags", "+faststart"
                                ])
    try:
        for entry in chosen:
            png = frames / entry["scene"].replace(".json.gz", ".png")
            with Image.open(png) as frame:
                image = np.asarray(frame.convert("RGB"))
            if panel:
                fields = dict(entry["label"])
                fields["banner_color"] = tuple(fields["banner_color"])
                image = compose_frame(image, FrameLabel(**fields))
            for _ in range(max(1, int(entry["repeat"]))):
                writer.append_data(image)
    finally:
        writer.close()
    return output
