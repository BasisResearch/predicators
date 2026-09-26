"""Export recorded Bridge states for passive Blender rendering of Figures
1-3."""
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple
from unittest.mock import patch

import pybullet as p
from render_scene_support import export_visual_scene, raised_flat_markers, \
    record_procedural_meshes, scene_signature

from predicators import utils
from predicators.envs import create_new_env
from predicators.structs import Object, State

ROOT = Path(__file__).resolve().parent
LOGS = ROOT.parents[1] / "logs/agent_continual"
# Training-level steps just before and during the first lift of the glued
# row, while the robot raises it straight up.
PAIR_PRE_LIFT_STEP, PAIR_LIFT_STEP = 1248, 1276
# The pair uses a closer camera, side-on to the blocks' long axis, so the lift
# and the partner stay legible in Figure 1's small panels.
PAIR_CAMERA = dict(cameraTargetPosition=(0.71, 1.145, 0.52),
                   distance=0.66,
                   yaw=-20,
                   pitch=-22,
                   roll=0,
                   upAxisIndex=2)
PAIR_FOV = 45
# Figure 2's act panel, the training row's second frame, is seen from low
# at the row's front corner: the glued end of the block being lowered faces
# the camera, and span1's glued top stays thin. The panel is 5:3.
ACT_FRAME = "trajectory_bridge_train_1"
ACT_CAMERA = dict(cameraTargetPosition=(0.815, 1.15, 0.475),
                  distance=0.36,
                  yaw=-40,
                  pitch=-9,
                  roll=0,
                  upAxisIndex=2)
ACT_FOV = 40
# Frames drawn from their own camera, with its field of view and image size;
# the others use the environment's camera at 900 by 900.
CLOSE_CAMERAS: Dict[str, Tuple[Dict[str, Any], float, Tuple[int, int]]] = {
    "bridge_pair_predicted": (PAIR_CAMERA, PAIR_FOV, (900, 900)),
    "bridge_pair_observed": (PAIR_CAMERA, PAIR_FOV, (900, 900)),
    ACT_FRAME: (ACT_CAMERA, ACT_FOV, (900, 540)),
}


def _canonical_state(env: Any, state: State) -> State:
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


def _pair_lift(episode: Dict[str, Any]) -> List[Tuple[Dict[str, Any], State]]:
    """Return Figure 1's two-block lift pair, built from the training level.

    Both scenes are illustrations. Every object keeps its initial pose
    except the robot and span1, which take the recorded mid-lift pose.
    span0 is either carried with span1 as the glued partner or left at
    its pre-lift pose, as a base simulator without glue would predict.
    The top-face dab on span1 is cleared so the two scenes differ only
    in the partner's pose.
    """
    initial = episode["states"][0]
    pre_lift = episode["states"][PAIR_PRE_LIFT_STEP]
    lifted = episode["states"][PAIR_LIFT_STEP]
    frames = []
    for name, partner in (("bridge_pair_predicted", pre_lift),
                          ("bridge_pair_observed", lifted)):
        state = initial.copy()
        state.data = {
            obj: features.copy()
            for obj, features in initial.data.items()
        }
        objects = {obj.name: obj for obj in state.data}
        for source, names in ((lifted, {"robot",
                                        "span1"}), (partner, {"span0"})):
            for obj, features in source.data.items():
                if obj.name in names:
                    state.data[objects[obj.name]] = features.copy()
        state.set(objects["span1"], "glue_top", 0.0)
        state.simulator_state = lifted.simulator_state.copy()
        frames.append((dict(name=name,
                            level_step=PAIR_LIFT_STEP,
                            illustrative=True), state))
    return frames


def _export_row(row: Dict[str, Any]) -> None:
    source = LOGS / row["recording"]
    source_bytes = source.read_bytes()
    assert hashlib.sha256(source_bytes).hexdigest() == row["recording_sha256"]
    episodes = pickle.loads(source_bytes)  # Trusted local experiment record.
    episode = next(ep for ep in episodes if ep["end"] == "win")
    flags: Dict[str, Any] = dict(env="pybullet_bridge",
                                 seed=0,
                                 num_train_tasks=1,
                                 num_test_tasks=1,
                                 partially_observable=True,
                                 pybullet_camera_width=900,
                                 pybullet_camera_height=900,
                                 bridge_train_span_blocks=3,
                                 bridge_test_span_blocks=4)
    utils.reset_config(flags)
    env: Any = create_new_env(flags["env"], do_cache=False, use_gui=False)
    try:
        env.reset(row["split"], 0)
        output = ROOT / "data/cycles_scenes"
        output.mkdir(parents=True, exist_ok=True)
        frames = [(frame, episode["states"][frame["level_step"]])
                  for frame in row["frames"]]
        if row["key"] == "bridge_train":
            frames.extend(_pair_lift(episode))
        for frame, recorded_state in frames:
            state = _canonical_state(env, recorded_state)
            env._set_state(state)  # pylint: disable=protected-access
            env._current_observation = state  # pylint: disable=protected-access
            client = env._physics_client_id  # pylint: disable=protected-access
            before = scene_signature(client)
            with patch.object(p,
                              "stepSimulation",
                              side_effect=AssertionError(
                                  "Rendering must not step physics")):
                with raised_flat_markers(client):
                    view, projection, _, _ = env._get_camera_matrices()  # pylint: disable=protected-access
                    width = height = 900
                    if frame["name"] in CLOSE_CAMERAS:
                        camera, fov, (width,
                                      height) = CLOSE_CAMERAS[frame["name"]]
                        view = p.computeViewMatrixFromYawPitchRoll(
                            **camera, physicsClientId=client)
                        projection = p.computeProjectionMatrixFOV(
                            fov=fov,
                            aspect=width / height,
                            nearVal=0.1,
                            farVal=100.0,
                            physicsClientId=client)
                    metadata = dict(
                        domain=row["domain"],
                        source_recording=str(source.relative_to(LOGS)),
                        source_sha256=row["recording_sha256"],
                        level=row["level"],
                        level_step=frame["level_step"],
                        frame=frame["name"],
                        illustrative=frame.get("illustrative", False),
                        physics_steps_after_restore=0)
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
