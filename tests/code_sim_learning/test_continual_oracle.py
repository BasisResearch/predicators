"""Mechanical oracle audits, separate from continual agent outcomes."""
# pylint: disable=protected-access
from typing import Any

import numpy as np
import pytest

from predicators import utils
from predicators.code_sim_learning.base_simulator import base_simulator_class
from predicators.code_sim_learning.continual_oracle import oracle_source
from predicators.code_sim_learning.fit_space import ParamSpec
from predicators.envs import create_new_env
from predicators.settings import CFG
from tests.code_sim_learning.test_balloons_subclass_form import _hold, _level
from tests.code_sim_learning.test_fan_gt_simulator import _make_noop


def _load() -> Any:
    namespace = {
        "BaseSimulator": base_simulator_class(CFG.env),
        "ParamSpec": ParamSpec
    }
    exec(oracle_source(), namespace)  # pylint: disable=exec-used
    return namespace["RESIDUAL_ENV"]


@pytest.mark.parametrize(
    "env_name", ["pybullet_domino", "pybullet_fan", "pybullet_balloons"])
def test_oracle_source_loads_fixed_values(env_name: str) -> None:
    """Every supplied parameter is pinned, including material calibration."""
    utils.reset_config({
        "env": env_name,
        "seed": 0,
        "domino_true_friction": 0.5,
        "domino_planning_friction": 0.1
    })
    cls = _load()
    for spec in cls.AGENT_PARAM_SPECS:
        assert spec.lo == spec.hi == spec.init_value
    if env_name == "pybullet_domino":
        assert cls.AGENT_PARAM_SPECS[0].init_value == 0.5
    elif env_name == "pybullet_balloons":
        values = {s.name: s.init_value for s in cls.AGENT_PARAM_SPECS}
        assert values["air_drag"] == CFG.balloons_drag
        assert values["mass_oak"] == CFG.balloons_box_masses[1]


@pytest.mark.parametrize("release_steps", [(0, 0, 0), (0, 35, 70)])
def test_balloons_oracle_sequential_release(release_steps: tuple) -> None:
    """A second release during flight preserves the payload's momentum.

    Clips are actuated directly in both independent worlds to isolate
    mechanism fidelity from controller reliability. This is not an agent
    result and does not establish a solve rate.
    """
    utils.reset_config({
        "env": "pybullet_balloons",
        "seed": 0,
        "num_train_tasks": 1,
        "num_test_tasks": 1,
        "partially_observable": True,
        "skill_phase_use_motion_planning": False
    })
    real: Any = create_new_env(CFG.env, do_cache=False, use_gui=False)
    model = _load()(use_gui=False, skip_residual_dynamics=False)
    colors = [0, 3, 1]
    expected = real.hover_height(1, colors)
    assert expected is not None
    initial = _level(real, colors, (expected - .025, expected + .025),
                     [])  # type: ignore[no-untyped-call]
    real._set_state(initial)
    model._set_state(initial)
    errors = []
    moving_release = False
    for step in range(160):
        for index, release_at in enumerate(release_steps):
            if step == release_at:
                speed = real._get_state().get(real._box, "speed")
                moving_release |= step > 0 and speed > 1e-4
                real._set_clip_on(real._clips[index], True)
                model._set_clip_on(model._clips[index], True)
        action = _hold(real._get_state(),
                       real)  # type: ignore[no-untyped-call]
        real._step_once(action)
        model._step_once(action)
        actual, predicted = real._get_state(), model._get_state()
        for obj in actual:
            if obj.type.name in {"box", "balloon"}:
                for feature in ("x", "y", "z"):
                    errors.append(
                        abs(
                            actual.get(obj, feature) -
                            predicted.get(obj, feature)))
                for feature in ("tied", "popped"):
                    if feature in obj.type.feature_names:
                        assert actual.get(obj, feature) == predicted.get(
                            obj, feature)
    if release_steps[1] > 0:
        assert moving_release, "The intended moving-release witness was absent"
    assert np.max(errors) < 0.005, float(np.max(errors))


@pytest.mark.parametrize("sides", [[], [0], [3], [0, 3]])
def test_fan_oracle_wind_and_contact(sides: list) -> None:
    """Compare the supplied subclass in free motion and against obstacles."""
    utils.reset_config({
        "env": "pybullet_fan",
        "seed": 0,
        "num_train_tasks": 1,
        "num_test_tasks": 1,
        "fan_3x3_strategic_task_gen": True,
    })
    real: Any = create_new_env(CFG.env, do_cache=False, use_gui=False)
    model = _load()(use_gui=False, skip_residual_dynamics=False)
    initial = real.get_train_tasks()[0].init.copy()
    ball = next(obj for obj in initial if obj.type.name == "ball")
    for obj in initial:
        if obj.type.name == "switch":
            initial.set(obj, "is_on",
                        float(initial.get(obj, "controls_fan") in sides))
    real._set_state(initial)
    model._set_state(initial)
    errors = []
    moved = []
    for _ in range(80):
        action = _make_noop(real._get_state(),
                            real)  # type: ignore[no-untyped-call]
        real._step_once(action)
        model._step_once(action)
        actual, predicted = real._get_state(), model._get_state()
        for feature in ("x", "y"):
            errors.append(
                abs(actual.get(ball, feature) - predicted.get(ball, feature)))
            moved.append(
                abs(actual.get(ball, feature) - initial.get(ball, feature)))
    assert max(errors) < 1e-5, max(errors)
    if sides:
        assert max(moved) > 0.01, "The wind was not exercised"
