"""Export the recorded Figure 3 states for passive Blender rendering."""
import hashlib
import json
import pickle
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Tuple, cast
from unittest.mock import patch

import pybullet as p

from predicators import utils
from predicators.envs import BaseEnv, create_new_env
from predicators.envs.pybullet_balloons import PyBulletBalloonsEnv
from predicators.envs.pybullet_env import PyBulletEnv
from predicators.structs import Object, State

# Put the repository root on sys.path so `scripts` is importable when this
# file runs directly, without PYTHONPATH=.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# pylint: disable=wrong-import-position
from scripts.paper_figures.render_scene_support import export_visual_scene, \
    raised_flat_markers, record_procedural_meshes, scene_signature

# pylint: enable=wrong-import-position

ROOT = Path(__file__).resolve().parent
LOGS = ROOT.parents[1] / "logs/agent_continual"


def _canonical_state(env: BaseEnv, state: State) -> State:
    """Rebind objects in a historical state to the new physics client."""
    canonical: Dict[str, Object] = {}
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


def _migrate_balloons_layout(env: PyBulletBalloonsEnv, state: State) -> State:
    """Apply the current visual layout to the archived Balloons state.

    The selected run predates the change that moved the box column from
    y=1.20 to y=1.42 and centred the target-height band in the chute.
    The motion and outcome are unchanged; only the whole attached
    assembly and the visual band are translated to the current scene
    coordinates.
    """
    objects = {obj.name: obj for obj in state}
    box = objects["box"]
    delta_y = env.box_xy[1] - state.get(box, "y")
    state.set(box, "y", state.get(box, "y") + delta_y)
    for obj in state.get_objects(env._balloon_type):  # pylint: disable=protected-access
        if state.get(obj, "tied") > 0.5:
            state.set(obj, "y", state.get(obj, "y") + delta_y)
    band = objects["band"]
    state.set(band, "x", env.box_xy[0] + env.band_offset_x)
    state.set(band, "y", env.box_xy[1])
    return state


def _export_row(row: Dict[str, Any]) -> None:
    """Export every selected frame of one recorded Figure 3 run."""
    domain = row["domain"]
    run = LOGS / row["run"]
    source = run / row["level"] / "episodes.pkl"
    source_bytes = source.read_bytes()
    episodes = pickle.loads(source_bytes)  # Trusted local experiment record.
    episode = next(ep for ep in episodes if ep["end"] == "win")
    is_balloons = domain == "Balloons"
    flags: Dict[str, Any] = dict(
        env="pybullet_balloons" if is_balloons else "pybullet_bridge",
        seed=0,
        num_train_tasks=2 if is_balloons else 1,
        num_test_tasks=1,
        partially_observable=True,
        pybullet_camera_width=1280 if is_balloons else 900,
        pybullet_camera_height=800 if is_balloons else 900)
    if is_balloons:
        flags.update(balloons_scene="chute")
    else:
        flags.update(bridge_train_span_blocks=3, bridge_test_span_blocks=4)
    utils.reset_config(flags)
    env = cast(PyBulletEnv,
               create_new_env(flags["env"], do_cache=False, use_gui=False))
    try:
        env.reset("test", 0)
        output = ROOT / "data/cycles_scenes"
        output.mkdir(parents=True, exist_ok=True)
        frames: List[Tuple[Dict[str, Any], State]] = [
            (frame, episode["states"][frame["level_step"]])
            for frame in row["frames"]
        ]
        if domain == "Bridge":
            # Reproduce the teaser's explicit counterfactual: the robot and
            # held span use the lifted state, while the other spans retain
            # their pre-lift poses. This is an illustration, not a run frame.
            pre_lift = episode["states"][1800]
            lifted = episode["states"][1850]
            wet = pre_lift.copy()
            wet.data = {
                obj: features.copy()
                for obj, features in pre_lift.data.items()
            }
            for lifted_obj, features in lifted.data.items():
                if lifted_obj.name in {"robot", "span3"}:
                    wet_obj = next(obj for obj in wet.data
                                   if obj.name == lifted_obj.name)
                    wet.data[wet_obj] = features.copy()
            wet.simulator_state = lifted.simulator_state.copy()
            frames.extend([(dict(name="bridge_wet_lift",
                                 level_step=1850,
                                 illustrative=True), wet),
                           (dict(name="bridge_bonded_lift",
                                 level_step=1850,
                                 illustrative=False), lifted)])
        for frame, recorded_state in frames:
            step = frame["level_step"]
            state = _canonical_state(env, recorded_state)
            if is_balloons:
                state = _migrate_balloons_layout(
                    cast(PyBulletBalloonsEnv, env), state)
            env._set_state(state)  # pylint: disable=protected-access
            env._current_observation = state  # pylint: disable=protected-access
            if hasattr(env, "_sync_cable_visuals"):
                env._sync_cable_visuals()  # pylint: disable=protected-access
            client = env._physics_client_id  # pylint: disable=protected-access
            before = scene_signature(client)
            attachments = (env.render_attachments() if hasattr(
                env, "render_attachments") else nullcontext())
            with patch.object(p,
                              "stepSimulation",
                              side_effect=AssertionError(
                                  "Rendering must not step physics")):
                with raised_flat_markers(client), attachments:
                    view, projection, _, _ = env._get_camera_matrices()  # pylint: disable=protected-access
                    width, height = ((1280, 800) if is_balloons else
                                     (900, 900))
                    metadata = dict(
                        domain=domain,
                        source_recording=str(source.relative_to(LOGS)),
                        source_sha256=hashlib.sha256(source_bytes).hexdigest(),
                        level=row["level"],
                        level_step=step,
                        frame=frame["name"],
                        illustrative=frame.get("illustrative", False),
                        physics_steps_after_restore=0)
                    if is_balloons:
                        metadata["visualization_migration"] = \
                            "centered current chute layout"
                    scene = export_visual_scene(client, view, projection,
                                                width, height, metadata)
            assert scene_signature(client) == before
            destination = output / f'{frame["name"]}.json'
            destination.write_text(json.dumps(scene, indent=2) + "\n")
            print(f'Exported {frame["name"]}', flush=True)
    finally:
        env.dispose()


def main() -> None:
    """Export all recorded scenes selected for Figure 3."""
    archive = json.loads((ROOT / "data/trajectories/figure3.json").read_text())
    with record_procedural_meshes():
        for row in archive["rows"]:
            _export_row(row)


if __name__ == "__main__":
    main()
