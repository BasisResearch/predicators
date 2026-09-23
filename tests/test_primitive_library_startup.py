"""The agent arms start under ``skill_library=primitive`` in every pilot
environment: ``main.py``'s setup path (environment, tasks, approach, offline
dataset) must not index a composite skill by name.

Every Sept 16, 2026 primitive pilot crashed on that path twice: the
demonstrator oracle's process factory (``KeyError: 'PickBlock'``) and the
Domino min-block task generator's Push lookup (``StopIteration``).
"""

import os
from typing import Any, Dict

import pytest

from predicators import utils
from predicators.run.setup import create_offline_dataset, setup_approach, \
    setup_environment
from predicators.settings import CFG

PRIMITIVE = {"Gripper", "MoveLinear", "MoveTo", "MoveUntilContact", "Wait"}

# The task-generation flags of the Sept 16, 2026 primitive pilot configs.
# Flags a runtime does not define are dropped (see _flags_for), so the
# test also runs on branches whose environments predate them.
_ENV_FLAGS: Dict[str, Dict[str, Any]] = {
    "pybullet_bridge": {
        "bridge_train_span_blocks": 3,
        "bridge_test_span_blocks": 3,
    },
    "pybullet_fan": {},
    "pybullet_domino": {
        "domino_initialize_at_finished_state": False,
        "domino_use_domino_blocks_as_target": True,
        "domino_use_continuous_place": True,
        "domino_has_glued_dominos": False,
        "domino_min_block_tasks": True,
        "domino_true_friction": 0.5,
        "domino_planning_friction": 0.1,
        "domino_min_block_span_lo": 0.29,
        "domino_min_block_span_hi": 0.31,
        "domino_min_block_turn_entry_lo": 0.21,
        "domino_min_block_turn_entry_hi": 0.24,
        "domino_min_block_turn_exit_lo": 0.17,
        "domino_min_block_turn_exit_hi": 0.2,
        "domino_min_block_num_blues": 4,
        "domino_block_cost": 0.1,
        "domino_test_turn_ratio": 1.0,
    },
    "pybullet_boil": {
        "boil_goal": "simple",
        "boil_num_jugs_train": [1],
        "boil_num_jugs_test": [2],
        "boil_num_burner_train": [1],
        "boil_num_burner_test": [1],
    },
    "pybullet_balloons": {
        "num_train_tasks": 2,
        "balloons_scene": "chute",
        "balloons_task_generation": "original",
        "balloons_require_jam_decoy": True,
        "balloons_goal_dwell_steps": 1,
    },
}


def _flags_for(env_name: str) -> Dict[str, Any]:
    """The pilot flags of ``env_name`` that this runtime defines."""
    return {
        k: v
        for k, v in _ENV_FLAGS[env_name].items()
        if k.startswith("num_") or hasattr(CFG, k)
    }


@pytest.mark.slow
@pytest.mark.parametrize("env_name", sorted(_ENV_FLAGS))
@pytest.mark.parametrize("approach",
                         ["agent_continual_model_free", "agent_continual"])
def test_agent_arm_starts_under_the_primitive_library(tmp_path: Any,
                                                      env_name: str,
                                                      approach: str) -> None:
    """Environment, train and test tasks, the approach with the five
    primitives, and an empty offline dataset, as ``main.py`` builds them."""
    # pylint: disable=protected-access
    if env_name == "pybullet_balloons" and not hasattr(CFG, "balloons_scene"):
        pytest.skip("Balloons task generation without the chute scene runs "
                    "for more than ten minutes; the pilots use the chute.")
    utils.reset_config({
        "env":
        env_name,
        "approach":
        approach,
        "seed":
        0,
        "num_train_tasks":
        1,
        "num_test_tasks":
        1,
        "max_initial_demos":
        0,
        "skill_library":
        "primitive",
        "continual_raw_control":
        False,
        "experiment_protocol":
        "continual",
        "continual_render":
        False,
        "continual_runs_dir":
        os.path.join(str(tmp_path), "runs"),
        "approach_dir":
        os.path.join(str(tmp_path), "saved"),
        "data_dir":
        os.path.join(str(tmp_path), "data"),
        "agent_sdk_use_local_sandbox":
        True,
        "option_model_use_gui":
        False,
        "experiment_id":
        "primitive-startup",
        "domino_min_block_task_cache_dir":
        os.path.join(str(tmp_path), "domino_cache"),
        **_flags_for(env_name),
    })
    try:
        setup = setup_environment()
        assert setup.train_tasks
        assert setup.env.get_test_tasks()
        arm = setup_approach(setup.env, setup.preds,
                             setup.approach_train_tasks)
        assert {o.name for o in arm._initial_options} == PRIMITIVE
        dataset = create_offline_dataset(setup.env, setup.train_tasks,
                                         setup.preds, arm)
        assert dataset is not None and not dataset.trajectories
    finally:
        utils.update_config({
            "skill_library": "composite",
            "continual_raw_control": True
        })
