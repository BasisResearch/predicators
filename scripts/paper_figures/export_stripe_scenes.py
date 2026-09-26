"""Export Cycles scenes for the per-domain trajectory stripes.

Each stripe shows one recorded EMPIRIC run from its experiments to the
solved test task. A frame is either a recorded environment state or a
state from the agent's own model, saved by render_model_rollout.py when
it replays a plan the agent checked. Scenes are written as
``data/cycles_scenes/trajectory_<domain>_stripe_<k>.json`` so that
render_cycles_scenes.py renders them with the other trajectory scenes.

Display-only adjustments, recorded in each scene's metadata:

- Fan and Balloons states move into the current scene layouts that
  Figure 1 uses; motion relative to the platforms or chute is unchanged.
- Balloons draws its burst height as a red cap over the chute, as in
  Figure 1 (export_static_scenes.py).
- Boil liquid is drawn no higher than the jug rim. The environment lets
  water rise above the rim before it overflows, which reads as an
  upturned jug. Its spill puddle, which restoring a state omits, is
  drawn from the recorded spilled level.
- Bridge model states draw the glue the model remembers as the
  environment's glue patches.
"""
import argparse
import hashlib
import json
import pickle
import re
import shlex
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import patch

import pybullet as p
from export_static_scenes import _cap_chute, _migrate_balloons_layout, \
    _migrate_fan_layout
from export_trajectory_scenes import _canonical_state
from PIL import Image
from render_scene_support import export_visual_scene, raised_flat_markers, \
    record_procedural_meshes, scene_signature

from predicators import utils
from predicators.envs import create_new_env
from predicators.settings import CFG
from predicators.structs import State

ROOT = Path(__file__).resolve().parent
LOGS = ROOT.parents[1] / "logs"
SPEC = ROOT / "data/trajectories/stripes.json"


def _load_run_config(run: Path) -> None:
    """Restore the flags of a trusted local run from its launch command."""
    info = re.sub(r"\x1b\[[0-9;]*m", "", (run / "info.log").read_text())
    command = next(
        line.split("Running command: ", 1)[1] for line in info.splitlines()
        if "Running command:" in line)
    argv = sys.argv
    try:
        sys.argv = shlex.split(command)[1:]
        utils.reset_config(utils.parse_args())
    finally:
        sys.argv = argv


def _recorded_state(run: Path, frame: Dict[str, Any]) -> State:
    """Return a recorded state; level steps run on across episodes."""
    with (run / frame["level"] / "episodes.pkl").open("rb") as stream:
        episodes = pickle.load(stream)  # Trusted local experiment record.
    episode = frame.get("episode", 0)
    # A reset costs one step, so each earlier episode's states (its
    # actions plus the reset) come before this episode's first step.
    start = sum(len(ep["states"]) for ep in episodes[:episode])
    states = episodes[episode]["states"]
    step = frame["step"]
    index = len(states) - 1 if step == "last" else step - start
    assert 0 <= index < len(states), (frame, start, len(states))
    return states[index]


def _model_state(frame: Dict[str, Any]) -> State:
    """Return a state saved from a replay of the agent's own model."""
    with (LOGS / frame["states"]).open("rb") as stream:
        states = pickle.load(stream)  # Trusted local replay output.
    return states[frame.get("index", -1)]


def _cap_liquid_at_rim(env: Any, state: State) -> State:
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


def _restore_spill(env: Any, state: State) -> None:
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


def _show_remembered_glue(state: State, memory: Dict[str, Any]) -> State:
    """Mark every face the model has put glue on as a glue patch."""
    for obj in state:
        for face, level in memory.get("glue", {}).get(obj.name, {}).items():
            if f"glue_{face}" in obj.type.feature_names:
                state.set(obj, f"glue_{face}", float(level > 0))
    return state


def _export_row(row: Dict[str, Any],
                output: Path,
                preview: Optional[Path] = None) -> List[str]:
    """Export one stripe's frames and return the scene stems."""
    run = LOGS / "agent_continual" / row["run"]
    _load_run_config(run)
    width, height = row.get("resolution", (900, 900))
    CFG.pybullet_camera_width, CFG.pybullet_camera_height = width, height
    for name, value in row.get("flags", {}).items():
        setattr(CFG, name, value)
    env: Any = create_new_env(CFG.env, do_cache=False, use_gui=False)
    stems = []
    try:
        layouts = {split: env.reset(split, 0) for split in ("train", "test")}
        for k, frame in enumerate(row["frames"]):
            if "pending" in frame:
                print("Skipped pending frame", row["domain"], k, flush=True)
                continue
            source = (_model_state(frame)
                      if "states" in frame else _recorded_state(run, frame))
            state = _canonical_state(env, source)
            notes = []
            if row["domain"] == "Fan":
                state = _migrate_fan_layout(env, state,
                                            layouts[frame["split"]])
                notes.append("current ramp layout")
            elif row["domain"] == "Balloons":
                state = _migrate_balloons_layout(env, state)
                notes += ["centered current chute layout", "chute cap"]
            elif row["domain"] == "Boil":
                state = _cap_liquid_at_rim(env, state)
                notes.append("liquid drawn no higher than the rim")
            if frame.get("show_memory_glue"):
                state = _show_remembered_glue(state, source.latent or {})
                notes.append("model's remembered glue drawn as patches")
            env._set_state(state)  # pylint: disable=protected-access
            env._current_observation = state  # pylint: disable=protected-access
            if row["domain"] == "Boil":
                _restore_spill(env, state)
            if row["domain"] == "Balloons":
                # The environment draws the strings of tied balloons only
                # when it renders an image.
                env._sync_cable_visuals()  # pylint: disable=protected-access
            client = env._physics_client_id  # pylint: disable=protected-access
            before = scene_signature(client)
            with patch.object(p,
                              "stepSimulation",
                              side_effect=AssertionError(
                                  "Rendering must not step physics")):
                with raised_flat_markers(client):
                    view, projection, _, _ = env._get_camera_matrices()  # pylint: disable=protected-access
                    source_file = (LOGS / frame["states"] if "states" in frame
                                   else run / frame["level"] / "episodes.pkl")
                    metadata = dict(domain=row["domain"],
                                    run=row["run"],
                                    source=str(source_file.relative_to(LOGS)),
                                    source_sha256=hashlib.sha256(
                                        source_file.read_bytes()).hexdigest(),
                                    level=frame.get("level"),
                                    episode=frame.get("episode"),
                                    step=frame.get("step"),
                                    model=bool("states" in frame),
                                    display_notes=notes,
                                    physics_steps_after_restore=0)
                    scene = export_visual_scene(client, view, projection,
                                                width, height, metadata)
            assert scene_signature(client) == before
            if row["domain"] == "Balloons":
                _cap_chute(env, scene)
            stem = f"trajectory_{row['domain'].lower()}_stripe_{k}"
            if preview is not None:
                # A quick engine image from the same camera, for framing.
                image, = env.render()
                Image.fromarray(image).save(  # type: ignore[no-untyped-call]
                    preview / f"{stem}.png")
            (output /
             f"{stem}.json").write_text(json.dumps(scene, indent=2) + "\n")
            stems.append(stem)
            print("Exported", stem, flush=True)
    finally:
        env.dispose()
    return stems


def main() -> None:
    """Export the scenes of the selected stripes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domains", nargs="+")
    parser.add_argument("--preview",
                        type=Path,
                        help="Also save quick engine renders here")
    args = parser.parse_args()
    if args.preview is not None:
        args.preview.mkdir(parents=True, exist_ok=True)
    rows = json.loads(SPEC.read_text())["rows"]
    if args.domains:
        rows = [row for row in rows if row["domain"] in args.domains]
    output = ROOT / "data/cycles_scenes"
    output.mkdir(parents=True, exist_ok=True)
    stems = []
    with record_procedural_meshes():
        for row in rows:
            stems.extend(_export_row(row, output, args.preview))
    print("scenes:", " ".join(stems), flush=True)


if __name__ == "__main__":
    main()
