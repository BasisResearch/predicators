"""The boil env's observable sim core keeps the visibility contract.

``pybullet_boil_base.py`` is copied verbatim into a learning agent's
sandbox when ``CFG.agent_sim_provide_base_sim_source`` is on, so it must
hold no hidden mechanism (filling, spilling, heating, bubbling, the
human's happiness and the constants of those laws), no task generation
and no predicate / goal semantics.
"""
import re
from pathlib import Path

from predicators import utils
from predicators.envs import BaseEnv
from predicators.envs.pybullet_boil import PyBulletBoilEnv
from predicators.envs.pybullet_boil_base import PyBulletBoilBaseEnv

_REPO = Path(__file__).resolve().parents[2]
_BASE_FILE = _REPO / "predicators" / "envs" / "pybullet_boil_base.py"

# Everything moved into (or kept in) the concrete module because it is a
# hidden mechanism, a constant of one, task generation, or goal semantics.
_HIDDEN_IDENTIFIERS = [
    # Fill / spill law and its constants.
    "water_fill_speed",
    "boil_water_fill_speed",
    "water_filled_height",
    "max_jug_water_capacity",
    "max_water_spill_width",
    "_handle_faucet_logic",
    "_increment_spillage",
    "_fill_jug_water",
    "_create_spilled_water_block",
    "_spilled_water_id",
    "_spilled_level",
    "prev_on",
    "_update_prev_on_states",
    # Faucet outlet and alignment tolerance.
    "faucet_x_len",
    "faucet_outlet_local_dx",
    "faucet_outlet_local_dy",
    "_faucet_outlet_xy",
    "faucet_align_threshold",
    "boil_faucet_align_threshold",
    # Heating, hidden heat and its bubbling projection.
    "heating_speed",
    "burner_align_threshold",
    "boil_require_jug_full_to_heatup",
    "_handle_heating_logic",
    "_heat_levels",
    "_heat_of",
    "BUBBLING_THRESHOLD",
    "BUBBLING_RAMP",
    "BUBBLING_BOIL_THRESHOLD",
    "_update_liquid_colors",
    "_update_liquid_positions",
    # The human's happiness.
    "happy_speed",
    "happiness_level",
    "_human_type",
    "_humans",
    "_update_human_happiness",
    # The residual step itself.
    "_domain_specific_step",
    # Task generation.
    "_generate_train_tasks",
    "_generate_test_tasks",
    "_make_tasks",
    "_sample_xy",
    "jug_sample_x_min",
    "jug_sample_y_min",
    "_draw_sampling_boundary_debug_lines",
    "EnvironmentTask",
    # Predicates and goal semantics.
    "get_name",
    "Predicate",
    "DerivedPredicate",
    "GroundAtom",
    "goal_predicates",
    "_JugFilled_holds",
    "_WaterBoiled_holds",
    "_JugAtFaucet_holds",
    "_JugOnBurner_holds",
    "_task_objective_holds",
    "boil_goal",
]


def test_base_file_holds_no_hidden_mechanism() -> None:
    """No hidden-mechanism, task or goal identifier appears in the base."""
    text = _BASE_FILE.read_text(encoding="utf-8")
    leaked = [
        name for name in _HIDDEN_IDENTIFIERS if re.search(
            rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])", text)
    ]
    assert not leaked, f"hidden identifiers in the base sim: {leaked}"
    # It never imports the concrete module.
    assert "envs.pybullet_boil import" not in text


def test_base_sim_source_files_are_exactly_the_core() -> None:
    """The env lists the base module and the engine, and nothing else."""
    expected = [
        "predicators/envs/pybullet_boil_base.py",
        "predicators/envs/pybullet_env.py",
    ]
    assert PyBulletBoilEnv.get_base_sim_source_files() == expected
    assert PyBulletBoilBaseEnv.get_base_sim_source_files() == expected
    for rel in expected:
        assert (_REPO / rel).is_file()


def test_concrete_env_subclasses_the_base() -> None:
    """The concrete env keeps its name and builds on the base."""
    assert issubclass(PyBulletBoilEnv, PyBulletBoilBaseEnv)
    assert PyBulletBoilEnv.get_name() == "pybullet_boil"
    # Scene constants other code reads through the concrete class resolve
    # to the base's.
    for attr in ("table_height", "x_mid", "y_mid", "z_ub", "jug_handle_height",
                 "jug_handle_offset", "robot_init_tilt", "robot_init_wrist"):
        assert attr in vars(PyBulletBoilBaseEnv)
        assert getattr(PyBulletBoilEnv,
                       attr) == getattr(PyBulletBoilBaseEnv, attr)


def test_base_is_not_discoverable() -> None:
    """The base is abstract and has no name, so env discovery skips it."""
    assert PyBulletBoilBaseEnv.__abstractmethods__
    assert "get_name" not in vars(PyBulletBoilBaseEnv)
    discoverable = [
        cls for cls in utils.get_all_subclasses(BaseEnv)
        if not cls.__abstractmethods__
    ]
    assert PyBulletBoilEnv in discoverable
    assert PyBulletBoilBaseEnv not in discoverable
