"""Tests for the domino env's observable sim core (the visibility contract).

The files ``get_base_sim_source_files`` lists are copied verbatim into a
learning agent's sandbox, so they must hold the code the planning twin
runs and nothing the agent must learn or must not see.
"""
import os

from predicators import utils
from predicators.envs import BaseEnv
from predicators.envs.pybullet_domino.components.domino_bodies import \
    DominoBodiesComponent
from predicators.envs.pybullet_domino.components.domino_component import \
    DominoComponent
from predicators.envs.pybullet_domino.env import PyBulletDominoComposedEnv, \
    PyBulletDominoEnv, PyBulletDominoFanEnv
from predicators.envs.pybullet_domino.sim_core import PyBulletDominoBaseEnv

_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_EXPECTED_FILES = [
    "predicators/envs/pybullet_domino/sim_core.py",
    "predicators/envs/pybullet_domino/components/base_component.py",
    "predicators/envs/pybullet_domino/components/domino_bodies.py",
    "predicators/envs/pybullet_env.py",
]

# Identifiers that name a learning target, a hidden mechanism, task
# generation or goal semantics. None may appear in a domino sim-core
# file; the generic engine (pybullet_env.py) is checked against the
# domino-specific ones only, since it legitimately defines the
# overridable ``_domain_specific_step`` hook.
_DOMINO_HIDDEN = [
    # The physical values each instance really runs with.
    "domino_true_friction",
    "domino_planning_friction",
    "agent_sim_learn_oracle_sim_params",
    "heavy_block_true_mass",
    "domino_heavy_block_tasks",
    "heavy_block_color",
    "glued_domino_color",
    "1e10",
    # Task generation.
    "task_generators",
    "DominoTaskGenerator",
    "domino_min_block",
    "domino_train_num",
    "domino_test_num",
    "place_domino",
    "domino_y_lb",
    # Predicates, their thresholds and the goal / certificate semantics.
    "fallen_threshold",
    "domino_roll_threshold",
    "topple_angle_threshold",
    "_Toppled_holds",
    "_Upright_holds",
    "_Tilting_holds",
    "_InFront_holds",
    "_HeavyBlock_holds",
    "_DominoGlued_holds",
    "_StartBlock_holds",
    "_MovableBlock_holds",
    "_HandEmpty_holds",
    "Predicate(",
    "DominoEvaluator",
    "check_cascade_legitimacy",
    "cascade_certificate",
    "cascade_probe",
    "run_counterfactual_cascade_probe",
]
_SIM_CORE_ONLY_HIDDEN = [
    "_domain_specific_step",
    "skip_residual_dynamics",
    "_skip_domain_specific_dynamics",
]


def _read(rel: str) -> str:
    with open(os.path.join(_REPO_ROOT, rel), encoding="utf-8") as f:
        return f.read()


def test_base_sim_source_files_listed():
    """The concrete env inherits exactly the intended visible files."""
    assert PyBulletDominoBaseEnv.get_base_sim_source_files() == \
        _EXPECTED_FILES
    for cls in (PyBulletDominoComposedEnv, PyBulletDominoEnv,
                PyBulletDominoFanEnv):
        assert cls.get_base_sim_source_files() == _EXPECTED_FILES
    basenames = [os.path.basename(rel) for rel in _EXPECTED_FILES]
    assert len(set(basenames)) == len(basenames)
    for rel in _EXPECTED_FILES:
        assert os.path.isfile(os.path.join(_REPO_ROOT, rel)), rel


def test_base_sim_source_files_hide_learning_targets():
    """No listed file names a hidden identifier."""
    for rel in _EXPECTED_FILES:
        source = _read(rel)
        hidden = list(_DOMINO_HIDDEN)
        if not rel.endswith("pybullet_env.py"):
            hidden += _SIM_CORE_ONLY_HIDDEN
        leaked = [name for name in hidden if name in source]
        assert not leaked, f"{rel} leaks {leaked}"


def test_hidden_modules_are_not_listed():
    """The modules holding the hidden parts exist and stay unlisted."""
    listed = set(PyBulletDominoEnv.get_base_sim_source_files())
    for rel in [
            "predicators/envs/pybullet_domino/env.py",
            "predicators/envs/pybullet_domino/components/"
            "domino_component.py",
            "predicators/envs/pybullet_domino/cascade_certificate.py",
            "predicators/envs/pybullet_domino/cascade_probe.py",
    ]:
        assert os.path.isfile(os.path.join(_REPO_ROOT, rel)), rel
        assert rel not in listed
    # The hidden identifiers really live in the unlisted modules.
    env_source = _read("predicators/envs/pybullet_domino/env.py")
    assert "domino_true_friction" in env_source
    assert "def _domain_specific_step" in env_source
    comp_source = _read(
        "predicators/envs/pybullet_domino/components/domino_component.py")
    assert "heavy_block_true_mass" in comp_source
    assert "def _Toppled_holds" in comp_source


def test_class_hierarchy_and_discovery():
    """The concrete env subclasses the sim core; the core is not an env."""
    assert issubclass(PyBulletDominoComposedEnv, PyBulletDominoBaseEnv)
    assert issubclass(PyBulletDominoEnv, PyBulletDominoBaseEnv)
    assert issubclass(DominoComponent, DominoBodiesComponent)
    # Abstract: env discovery (create_new_env) skips classes with
    # abstract methods, and the core defines no name of its own.
    assert PyBulletDominoBaseEnv.__abstractmethods__
    assert "get_name" in PyBulletDominoBaseEnv.__abstractmethods__
    assert "get_name" not in vars(PyBulletDominoBaseEnv)
    assert DominoBodiesComponent.__abstractmethods__
    discoverable = [
        cls for cls in utils.get_all_subclasses(BaseEnv)
        if not cls.__abstractmethods__
    ]
    assert PyBulletDominoBaseEnv not in discoverable
    assert PyBulletDominoEnv in discoverable
    assert PyBulletDominoEnv.get_name() == "pybullet_domino"
