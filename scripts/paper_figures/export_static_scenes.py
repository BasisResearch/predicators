"""Export current Domino and Fan paper scenes from successful runs."""
import hashlib
import json
import pickle
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict
from unittest.mock import patch

import pybullet as p
from export_trajectory_scenes import _canonical_state
from render_scene_support import export_visual_scene, raised_flat_markers, \
    record_procedural_meshes, scene_signature

from predicators import utils
from predicators.envs import create_new_env
from predicators.envs.pybullet_fan import PyBulletFanEnv
from predicators.structs import State

ROOT = Path(__file__).resolve().parent
LOGS = ROOT.parents[1] / "logs/agent_continual"

ROWS = (
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
)


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


def _export_row(row: Dict[str, Any]) -> None:
    """Export the start and win scenes of one row's winning episode."""
    source = LOGS / row["run"] / row["level"] / "episodes.pkl"
    source_bytes = source.read_bytes()
    scorecard = LOGS / row["run"] / "scorecard.json"
    scorecard_bytes = scorecard.read_bytes()
    episodes = pickle.loads(source_bytes)  # Trusted local experiment record.
    episode = next(ep for ep in episodes if ep["end"] == "win")
    flags = dict(row["flags"],
                 env=row["env"],
                 seed=0,
                 num_train_tasks=1,
                 num_test_tasks=1,
                 partially_observable=True,
                 pybullet_camera_width=900,
                 pybullet_camera_height=900)
    utils.reset_config(flags)
    env: Any = create_new_env(row["env"], do_cache=False, use_gui=False)
    try:
        current_initial = env.reset("test", 0)
        output = ROOT / "data/cycles_scenes"
        output.mkdir(parents=True, exist_ok=True)
        for frame, recorded_state in (("start", episode["states"][0]),
                                      ("win", episode["states"][-1])):
            state = _canonical_state(env, recorded_state)
            if row["domain"] == "Fan":
                state = _migrate_fan_layout(env, state, current_initial)
            env._set_state(state)  # pylint: disable=protected-access
            env._current_observation = state  # pylint: disable=protected-access
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
                    metadata = dict(
                        domain=row["domain"],
                        source_recording=str(source.relative_to(LOGS)),
                        source_sha256=hashlib.sha256(source_bytes).hexdigest(),
                        scorecard_sha256=hashlib.sha256(
                            scorecard_bytes).hexdigest(),
                        level=row["level"],
                        frame=frame,
                        physics_steps_after_restore=0)
                    scene = export_visual_scene(client, view, projection, 900,
                                                900, metadata)
            assert scene_signature(client) == before
            destination = output / f'{row["domain"].lower()}_{frame}.json'
            destination.write_text(json.dumps(scene, indent=2) + "\n")
            print(f'Exported {row["domain"]} {frame}', flush=True)
    finally:
        env.dispose()


def main() -> None:
    """Export the current Domino and ramp Fan scenes."""
    with record_procedural_meshes():
        for row in ROWS:
            _export_row(row)


if __name__ == "__main__":
    main()
