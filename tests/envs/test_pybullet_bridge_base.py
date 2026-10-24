"""The bridge env's visibility contract: the base sim module is safe to show.

``pybullet_bridge_base.py`` is copied verbatim into a learning agent's
sandbox when ``CFG.agent_sim_provide_base_sim_source`` is on, so it must
hold only what the planning twin (``skip_residual_dynamics=True``)
executes: no glue / cure / weld laws or their constants, no task
generation, and no predicate or goal semantics.
"""
import re
from pathlib import Path

from predicators import utils
from predicators.envs import BaseEnv
from predicators.envs.pybullet_bridge import PyBulletBridgeEnv
from predicators.envs.pybullet_bridge_base import PyBulletBridgeBaseEnv

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BASE_FILE = _REPO_ROOT / "predicators" / "envs" / "pybullet_bridge_base.py"

# Identifiers that live only in the concrete env (pybullet_bridge.py).
_HIDDEN_IDENTIFIERS = (
    # Glue application (wetting) law and its constants.
    "apply_glue_radius",
    "wet_streak_steps",
    "_WET_PARTIAL",
    "glue_dab_dwell_steps",
    "dab_margin",
    "_face_dab_point",
    # Curing law, mate detection and latching.
    "cure_threshold",
    "_find_mate",
    "_rests_on_top",
    "_end_adjacent",
    "_mate_slot_for",
    "_SLOT_AXES",
    "_latch_joint",
    "stack_align_tol",
    "stack_z_tol",
    "lateral_proj_tol_lo",
    "lateral_proj_tol_hi",
    "lateral_perp_tol",
    "lateral_z_tol",
    # Wet-joint tacks.
    "wet_joint_tack_force",
    "_tack_constraints",
    "_sync_wet_joint_tacks",
    "_drop_tack",
    # Weld lifecycle.
    "weld_max_force",
    "weld_relax_max_lin_vel",
    "weld_relax_max_ang_vel",
    "_weld_constraints",
    "_weld_meta",
    "_create_weld",
    "_remove_weld",
    "_desired_weld_pairs",
    "_sync_welds_to_state",
    "_relax_resting_welds",
    "_weld_constraint_edges",
    "_ideal_block_orientation",
    # Residual step.
    "_domain_specific_step",
    # Predicates, goal semantics and the settle certificate.
    "seat_x_window",
    "seat_y_tol",
    "seat_z_tol",
    "at_site_tol",
    "at_site_z_tol",
    "_stands",
    "_world_half_extents",
    "_Bridged_holds",
    "_Attached_holds",
    "_Loose_holds",
    "_GOAL_SETTLE_SUBSTEPS",
    "check_episode_trajectory",
    "_certificate_snapshot",
    "episode_terminated",
    # Task generation.
    "_generate_train_tasks",
    "_generate_test_tasks",
    "_make_tasks",
    "_stage_objects",
    "_stage_transfer_objects",
    "_is_leg_shaped",
    "stage_cols",
    "stage_jitter",
    "site_keepout",
    "reach_radius",
    "strip_x_slack",
    "site_y",
    "site_sep",
    "site_x_jitter",
    "leg_color_family",
    "span_color_family",
)


def test_base_module_has_no_hidden_mechanisms() -> None:
    """No hidden-mechanism, goal or task-generation identifier appears in the
    base module's source, and every one still exists on the concrete env."""
    source = _BASE_FILE.read_text(encoding="utf-8")
    leaked = [
        name for name in _HIDDEN_IDENTIFIERS if re.search(
            rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])", source)
    ]
    assert not leaked, f"hidden identifiers in the base sim: {leaked}"
    for word in ("weld", "tack", "latch", "wetting"):
        assert word not in source.lower(), word
    missing = [
        name for name in _HIDDEN_IDENTIFIERS if not name.startswith(
            ("_weld_constraints", "_weld_meta",
             "_tack_constraints")) and not hasattr(PyBulletBridgeEnv, name)
    ]
    assert not missing, f"concrete env lost: {missing}"
    for name in _HIDDEN_IDENTIFIERS:
        assert not hasattr(PyBulletBridgeBaseEnv, name) or name in (
            "_domain_specific_step", "_weld_constraint_edges",
            "check_episode_trajectory", "episode_terminated",
            "_generate_train_tasks", "_generate_test_tasks"), name


def test_base_sim_source_files() -> None:
    """The base declares exactly its own module and the generic engine."""
    expected = [
        "predicators/envs/pybullet_bridge_base.py",
        "predicators/envs/pybullet_env.py",
    ]
    assert PyBulletBridgeBaseEnv.get_base_sim_source_files() == expected
    assert PyBulletBridgeEnv.get_base_sim_source_files() == expected
    for rel in expected:
        assert (_REPO_ROOT / rel).is_file()


def test_concrete_env_subclasses_base_and_base_is_not_discoverable() -> None:
    """The concrete env is the only discoverable bridge env."""
    assert issubclass(PyBulletBridgeEnv, PyBulletBridgeBaseEnv)
    assert PyBulletBridgeBaseEnv.__abstractmethods__
    assert "get_name" in PyBulletBridgeBaseEnv.__abstractmethods__
    assert not PyBulletBridgeEnv.__abstractmethods__
    discoverable = [
        cls for cls in utils.get_all_subclasses(BaseEnv)
        if not cls.__abstractmethods__ and cls.get_name() == "pybullet_bridge"
    ]
    assert discoverable == [PyBulletBridgeEnv]
