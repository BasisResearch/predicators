"""Export Figure 1's start and win scenes from successful runs.

Display-only adjustments, which export_stripe_scenes.py shares and each
scene's metadata records:

- Fan and Balloons states move into the current scene layouts; motion
  relative to the platforms or chute is unchanged.
- Balloons draws its burst height as a red cap over the chute. The
  environment pictures that height as a translucent plate over the whole
  table, but balloons only rise with the box, inside the chute.
- Boil liquid is drawn no higher than the jug rim. The environment lets
  water rise above the rim before it overflows, which reads as an
  upturned jug. Its spill puddle, which restoring a state omits, is
  drawn from the recorded spilled level.

Figure 1 alone also draws Boil's cyan test jug in the red that the same
jug has in training, because the renderer's ambient response for jug
cavities washes cyan out. The Boil stripe keeps the recorded cyan.
"""
import argparse
import hashlib
import json
import pickle
import re
import shlex
import sys
from pathlib import Path
from typing import Any, Dict, Tuple
from unittest.mock import patch

import pybullet as p
from export_trajectory_scenes import _canonical_state
from render_scene_support import export_visual_scene, raised_flat_markers, \
    record_procedural_meshes, scene_signature

from predicators import utils
from predicators.envs import create_new_env
from predicators.envs.pybullet_fan import PyBulletFanEnv
from predicators.settings import CFG
from predicators.structs import State

ROOT = Path(__file__).resolve().parent
LOGS = ROOT.parents[1] / "logs/agent_continual"

ROWS: Tuple[Dict[str, Any], ...] = (
    dict(domain="Domino",
         env="pybullet_domino",
         run="domino_high_friction_turn-mb_opus_gate_r1/seed0/"
         "run_20260917_082017",
         level="L02",
         flags=dict(domino_initialize_at_finished_state=True,
                    domino_use_domino_blocks_as_target=True,
                    domino_use_continuous_place=True,
                    domino_has_glued_dominos=False,
                    domino_min_block_tasks=False,
                    domino_train_num_dominos=[6],
                    domino_test_num_dominos=[6],
                    domino_train_num_targets=[0],
                    domino_test_num_targets=[0],
                    domino_train_num_pivots=[0],
                    domino_test_num_pivots=[0],
                    domino_true_friction=0.5,
                    domino_planning_friction=0.1,
                    domino_min_block_span_lo=0.29,
                    domino_min_block_span_hi=0.31,
                    domino_min_block_turn_entry_lo=0.21,
                    domino_min_block_turn_entry_hi=0.24,
                    domino_min_block_turn_exit_lo=0.17,
                    domino_min_block_turn_exit_hi=0.2,
                    domino_min_block_num_blues=4,
                    domino_train_turn_ratio=0.0,
                    domino_test_turn_ratio=0.0)),
    dict(domain="Fan",
         env="pybullet_fan",
         run="fan_ramp-mb_opus_ramp_skill_repair_r1/seed0/"
         "run_20260921_090823",
         level="L02",
         flags=dict(fan_exposed_transfer=True,
                    fan_inertial_transfer=True,
                    fan_ramp_transfer=True,
                    fan_ramp_rise=0.003,
                    fan_ramp_landing_extension=0.10,
                    fan_train_num_walls_per_task=[0],
                    fan_test_num_walls_per_task=[0],
                    fan_train_num_pos_x=3,
                    fan_train_num_pos_y=3,
                    fan_test_num_pos_x=3,
                    fan_test_num_pos_y=3,
                    fan_train_task_generation="uniform",
                    fan_test_task_generation="uniform")),
    dict(domain="Balloons",
         env="pybullet_balloons",
         run="balloons-mb_opus_compose_r2/seed0/run_20260917_082044",
         level="L03",
         resolution=(1280, 800),
         flags=dict(balloons_scene="chute",
                    balloons_require_jam_decoy=False,
                    balloons_goal_dwell_steps=25)),
    # A row without flags restores its run's settings from the launch
    # command. Its test level has two jugs, as in the Boil stripe.
    dict(domain="Boil",
         run="boil-mb_opus_gate_preflight_two_jug_tight_r1/seed1/"
         "run_20260917_082805",
         level="L02"),
)
# The Balloons burst height, drawn as an opaque red cap.
CHUTE_CAP_RGBA = (0.80, 0.16, 0.14, 1.0)
# Boil's first jug is red in training and cyan in the test task.
BOIL_TEST_JUG_RGBA = (0.05, 0.95, 0.95, 1.0)
BOIL_TRAINING_JUG_RGBA = (0.95, 0.05, 0.10, 1.0)


def _migrate_fan_layout(env: PyBulletFanEnv, state: State,
                        current_initial: State) -> State:
    """Combine an archived outcome with the current static ramp layout."""
    static_types = {
        env._platform_type,  # pylint: disable=protected-access
        env._ramp_type,  # pylint: disable=protected-access
        env._boundary_type,  # pylint: disable=protected-access
        env._target_type,  # pylint: disable=protected-access
    }
    for obj in state:
        if obj.type in static_types:
            state.data[obj] = current_initial.data[obj].copy()
        elif obj.type in {
                env._fan_type,  # pylint: disable=protected-access
                env._switch_type,  # pylint: disable=protected-access
        }:
            for feature in ("x", "y", "z", "rot"):
                state.set(obj, feature, current_initial.get(obj, feature))
    ball, = state.get_objects(env._ball_type)  # pylint: disable=protected-access
    state.set(ball, "x", state.get(ball, "x") + env.ramp_scene_x_offset)
    return state


def _migrate_balloons_layout(env: Any, state: State) -> State:
    """Move an archived Balloons state into the current chute layout.

    Runs before the layout change kept the box column at y=1.20. The
    box, the balloons tied to it and the target band move to the current
    column; motion and outcome are unchanged.
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


def _cap_chute(env: Any, scene: Dict[str, Any]) -> None:
    """Redraw an exported Balloons ceiling as a red cap over the chute.

    The cap keeps the plate's height and thickness, so its underside is
    still the burst height, and spans the chute walls.
    """
    assert CFG.balloons_scene == "chute"
    ceiling, = (shape for shape in scene["shapes"]
                if shape["body"] == env._ceiling_id)  # pylint: disable=protected-access
    half = (env.chute_half_gap + 2 * env.chute_wall_half_thickness,
            env.chute_wall_half_depth, env.ceiling_half_extents[2])
    ceiling.update(dimensions=[2 * h for h in half],
                   position=[*env.box_xy, env.ceiling_z],
                   rgba=list(CHUTE_CAP_RGBA))


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


def _redraw_test_jug(scene: Dict[str, Any]) -> None:
    """Draw Boil's cyan test jug in the red it has in training."""
    notes = scene["metadata"]["display_notes"]
    for shape in scene["shapes"]:
        if shape["name"] == "cup" and all(
                abs(a - b) < 1e-3
                for a, b in zip(shape["rgba"], BOIL_TEST_JUG_RGBA)):
            shape["rgba"] = list(BOIL_TRAINING_JUG_RGBA)
            if "test jug drawn in its training red" not in notes:
                notes.append("test jug drawn in its training red")


def _export_row(row: Dict[str, Any]) -> None:
    """Export the start and win scenes of one row's winning episode."""
    source = LOGS / row["run"] / row["level"] / "episodes.pkl"
    source_bytes = source.read_bytes()
    scorecard = LOGS / row["run"] / "scorecard.json"
    scorecard_bytes = scorecard.read_bytes()
    episodes = pickle.loads(source_bytes)  # Trusted local experiment record.
    episode = next(ep for ep in episodes if ep["end"] == "win")
    width, height = row.get("resolution", (900, 900))
    if "flags" in row:
        flags = dict(row["flags"],
                     env=row["env"],
                     seed=0,
                     num_train_tasks=1,
                     num_test_tasks=1,
                     partially_observable=True,
                     pybullet_camera_width=width,
                     pybullet_camera_height=height)
        utils.reset_config(flags)
    else:
        _load_run_config(LOGS / row["run"])
        CFG.pybullet_camera_width, CFG.pybullet_camera_height = width, height
    env: Any = create_new_env(CFG.env, do_cache=False, use_gui=False)
    try:
        current_initial = env.reset("test", 0)
        output = ROOT / "data/cycles_scenes"
        output.mkdir(parents=True, exist_ok=True)
        for frame, recorded_state in (("start", episode["states"][0]),
                                      ("win", episode["states"][-1])):
            state = _canonical_state(env, recorded_state)
            notes = []
            if row["domain"] == "Fan":
                state = _migrate_fan_layout(env, state, current_initial)
                notes.append("current ramp layout")
            elif row["domain"] == "Balloons":
                state = _migrate_balloons_layout(env, state)
                notes += ["centered current chute layout", "chute cap"]
            elif row["domain"] == "Boil":
                state = _cap_liquid_at_rim(env, state)
                notes.append("liquid drawn no higher than the rim")
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
                    metadata = dict(
                        domain=row["domain"],
                        source_recording=str(source.relative_to(LOGS)),
                        source_sha256=hashlib.sha256(source_bytes).hexdigest(),
                        scorecard_sha256=hashlib.sha256(
                            scorecard_bytes).hexdigest(),
                        level=row["level"],
                        frame=frame,
                        display_notes=notes,
                        physics_steps_after_restore=0)
                    scene = export_visual_scene(client, view, projection,
                                                width, height, metadata)
            assert scene_signature(client) == before
            if row["domain"] == "Balloons":
                _cap_chute(env, scene)
            elif row["domain"] == "Boil":
                _redraw_test_jug(scene)
            destination = output / f'{row["domain"].lower()}_{frame}.json'
            destination.write_text(json.dumps(scene, indent=2) + "\n")
            print(f'Exported {row["domain"]} {frame}', flush=True)
    finally:
        env.dispose()


def main() -> None:
    """Export the start and win scenes of the selected domains."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domains",
                        nargs="+",
                        default=[row["domain"] for row in ROWS])
    args = parser.parse_args()
    with record_procedural_meshes():
        for row in ROWS:
            if row["domain"] in args.domains:
                _export_row(row)


if __name__ == "__main__":
    main()
