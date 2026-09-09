"""Approach-level wiring of the SUBCLASS model form.

The subclass form is an alternative to the rule form: ``simulator.py``
exports ``RESIDUAL_ENV``, a subclass of the env's base-sim class that
overrides ``_domain_specific_step`` and declares its learnable constants
in ``AGENT_PARAM_SPECS``. These tests prove the approach loads such a
file, installs the subclass as the planning base env, and surfaces its
``AGENT_PARAM_SPECS`` for the rollout system-ID - without touching the
rule-form path. The subclass physics itself is checked in
``tests/code_sim_learning/test_balloons_subclass_form.py``.
"""
# pylint: disable=protected-access
from __future__ import annotations

import pytest

from predicators import utils
from predicators.approaches.agent_sim_learning_approach import \
    AgentSimLearningApproach
from predicators.code_sim_learning.utils import read_residual_env, \
    stamp_physical_spec_scales
from predicators.envs import create_new_env
from predicators.ground_truth_models.balloons.gt_simulator_env import \
    BalloonsResidualEnv

# A candidate simulator.py that re-exports the balloons subclass as its
# RESIDUAL_ENV - the minimal well-formed subclass-form artifact.
_REEXPORT_SIMULATOR_PY = (
    "from predicators.ground_truth_models.balloons.gt_simulator_env "
    "import BalloonsResidualEnv as RESIDUAL_ENV\n")

# A candidate that DEFINES a fresh concrete subclass inline, so every
# exec makes a distinct BaseEnv subclass sharing one sentinel get_name -
# the duplicate-registration case.
_INLINE_SUBCLASS_SIMULATOR_PY = (
    "from predicators.ground_truth_models.balloons.gt_simulator_env "
    "import BalloonsResidualEnv\n"
    "\n"
    "class _AgentModel(BalloonsResidualEnv):\n"
    "    pass\n"
    "\n"
    "RESIDUAL_ENV = _AgentModel\n")


@pytest.fixture(name="reset_balloons")
def _reset_balloons():
    utils.reset_config({
        "env": "pybullet_balloons",
        "seed": 0,
        "num_train_tasks": 1,
        "num_test_tasks": 1,
        "skill_phase_use_motion_planning": False,
        "option_model_use_gui": False,
    })


def _bare_approach(monkeypatch):
    """A bare approach with a real stock balloons base env.

    Only the fields the base-env swap touches are populated; the probe
    substrate factory is stubbed so the swap does not need full session
    state.
    """
    monkeypatch.setattr(AgentSimLearningApproach,
                        "_make_probe_process_model_factory", lambda self: None)
    approach = AgentSimLearningApproach.__new__(AgentSimLearningApproach)
    approach._base_env = create_new_env("pybullet_balloons",
                                        do_cache=False,
                                        use_gui=False,
                                        skip_residual_dynamics=True)
    approach._residual_env_cls = None
    approach._residual_env_key = None
    approach._identified_physical_params = {}
    approach._option_model = None
    approach._residual_rules = None
    approach._fitted_params = {}
    return approach


def test_loader_reads_subclass_form(tmp_path):
    """A subclass-form simulator.py loads with empty rules/specs, features from
    the class, and a readable RESIDUAL_ENV export."""
    path = tmp_path / "simulator.py"
    path.write_text(_REEXPORT_SIMULATOR_PY)
    rules, specs, features, ns = (
        AgentSimLearningApproach._load_simulator_from_module_file(str(path)))
    # No RESIDUAL_RULES/PARAM_SPECS, yet the file is loadable (not None).
    assert rules == []
    assert specs == []
    # Features fall back to the class's declaration.
    assert features == BalloonsResidualEnv.RESIDUAL_FEATURES
    assert read_residual_env(ns) is BalloonsResidualEnv


def test_install_swaps_planning_base_env(reset_balloons, monkeypatch):
    """Installing a RESIDUAL_ENV makes the planning base env an instance of the
    subclass with its hidden step live; clearing restores the stock sim."""
    del reset_balloons
    approach = _bare_approach(monkeypatch)
    stock = approach._base_env
    assert not isinstance(stock, BalloonsResidualEnv)

    approach._install_residual_env_cls(BalloonsResidualEnv, "key-1")
    assert approach._residual_env_cls is BalloonsResidualEnv
    assert isinstance(approach._base_env, BalloonsResidualEnv)
    # skip_residual_dynamics=False, so _domain_specific_step fires.
    assert approach._base_env._skip_domain_specific_dynamics is False
    # _make_planning_base_env now builds the subclass too.
    assert isinstance(approach._make_planning_base_env(), BalloonsResidualEnv)

    # Same content key: the base env is reused, not rebuilt.
    installed = approach._base_env
    approach._install_residual_env_cls(BalloonsResidualEnv, "key-1")
    assert approach._base_env is installed

    # Clearing restores the stock base sim.
    approach._install_residual_env_cls(None)
    assert approach._residual_env_cls is None
    assert not isinstance(approach._base_env, BalloonsResidualEnv)


def test_subclass_physical_specs_are_agent_param_specs(reset_balloons,
                                                       monkeypatch):
    """After installing the subclass, its AGENT_PARAM_SPECS become the physical
    params to identify (stamped against the subclass instance)."""
    del reset_balloons
    approach = _bare_approach(monkeypatch)
    approach._install_residual_env_cls(BalloonsResidualEnv, "key-1")
    approach._physical_param_specs = stamp_physical_spec_scales(
        list(BalloonsResidualEnv.AGENT_PARAM_SPECS), approach._base_env)
    names = {s.name for s in approach._physical_param_specs}
    declared = {s.name for s in BalloonsResidualEnv.AGENT_PARAM_SPECS}
    assert names == declared
    assert "air_drag" in names and "fade_height" in names
    # The registry surfaces every one of them for the fit.
    info = approach._base_env.get_physical_param_info()
    assert declared.issubset(set(info))


def test_loading_subclass_file_twice_does_not_raise(reset_balloons, tmp_path):
    """Re-exec of an inline RESIDUAL_ENV file (a fresh BaseEnv subclass each
    time, sharing one sentinel get_name) is tolerated, and the real env still
    resolves by name."""
    del reset_balloons
    path = tmp_path / "simulator.py"
    path.write_text(_INLINE_SUBCLASS_SIMULATOR_PY)

    _, _, _, ns1 = (AgentSimLearningApproach._load_simulator_from_module_file(
        str(path)))
    _, _, _, ns2 = (AgentSimLearningApproach._load_simulator_from_module_file(
        str(path)))
    cls1 = read_residual_env(ns1)
    cls2 = read_residual_env(ns2)
    # Two distinct concrete subclasses, both a valid RESIDUAL_ENV, both
    # carrying the fixed sentinel name.
    assert cls1 is not None and cls2 is not None and cls1 is not cls2
    assert cls1.get_name() == cls2.get_name() == \
        "pybullet_balloons_residual_model"
    # The duplicate residual-model classes never shadow the real env.
    env = create_new_env("pybullet_balloons",
                         do_cache=False,
                         use_gui=False,
                         skip_residual_dynamics=True)
    assert env.get_name() == "pybullet_balloons"
