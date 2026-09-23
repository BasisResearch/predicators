"""Mechanical oracle audits, separate from continual agent outcomes."""
# pylint: disable=protected-access
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest

from predicators import utils
from predicators.code_sim_learning.base_simulator import base_simulator_class
from predicators.code_sim_learning.continual_oracle import oracle_source
from predicators.code_sim_learning.fit_space import ParamSpec
from predicators.code_sim_learning.latent_tracker import \
    make_subclass_latent_tracker
from predicators.envs import create_new_env
from predicators.run.recording import sanitize_state
from predicators.settings import CFG
from tests.code_sim_learning.test_balloons_subclass_form import _hold, _level
from tests.code_sim_learning.test_fan_gt_simulator import _make_noop


def _load_namespace() -> Dict[str, Any]:
    namespace: Dict[str, Any] = {
        "BaseSimulator": base_simulator_class(CFG.env),
        "ParamSpec": ParamSpec
    }
    exec(oracle_source(), namespace)  # pylint: disable=exec-used
    return namespace


def _load() -> Any:
    return _load_namespace()["RESIDUAL_ENV"]


@pytest.mark.parametrize("env_name", [
    "pybullet_domino", "pybullet_fan", "pybullet_balloons", "pybullet_boil",
    "pybullet_bridge"
])
def test_oracle_source_loads_fixed_values(env_name: str) -> None:
    """The true values are module constants, never declared parameters."""
    utils.reset_config({
        "env": env_name,
        "seed": 0,
        "domino_true_friction": 0.5,
        "domino_planning_friction": 0.1
    })
    namespace = _load_namespace()
    cls = namespace["RESIDUAL_ENV"]
    assert cls.AGENT_PARAM_SPECS == []
    if env_name == "pybullet_domino":
        assert namespace["_FIXED_PARAMS"] == {"lateral_friction": 0.5}
    elif env_name == "pybullet_balloons":
        values = namespace["_FIXED_PARAMS"]
        assert values["air_drag"] == CFG.balloons_drag
        assert values["mass_oak"] == CFG.balloons_box_masses[1]
        assert set(values) >= {"lift_gold", "fade_height"}
    else:
        assert "_FIXED_PARAMS" not in namespace


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


@pytest.mark.parametrize("mode", ["heat", "not_full", "fill", "spill"])
def test_boil_oracle_mechanisms_and_observed_memory(mode: str) -> None:
    """Match native effects and infer hidden heat without privileged input."""
    utils.reset_config({
        "env": "pybullet_boil",
        "seed": 0,
        "partially_observable": True,
        "num_train_tasks": 1,
        "num_test_tasks": 1,
        "boil_goal": "simple",
        "boil_num_jugs_train": [1],
        "boil_num_jugs_test": [1],
        "boil_num_burner_train": [1],
        "boil_num_burner_test": [1],
        "boil_require_jug_full_to_heatup": True,
        "boil_water_fill_speed": 0.0015,
    })
    real: Any = create_new_env(CFG.env, do_cache=False, use_gui=False)
    cls = _load()
    model = cls(use_gui=False, skip_residual_dynamics=False)
    initial = real.get_train_tasks()[0].init.copy()
    jug = next(o for o in initial if o.type.name == "jug")
    burner = next(o for o in initial if o.type.name == "burner")
    faucet = next(o for o in initial if o.type.name == "faucet")
    if mode in {"heat", "not_full"}:
        initial.set(jug, "x", initial.get(burner, "x"))
        initial.set(jug, "y", initial.get(burner, "y"))
        initial.set(jug, "water_volume", 1.0 if mode == "heat" else 0.2)
        initial.set(burner, "is_on", 1.0)
    else:
        initial.set(faucet, "is_on", 1.0)
        x, y = real._faucet_outlet_xy(initial, faucet)
        initial.set(jug, "x", x + (0.3 if mode == "spill" else 0.0))
        initial.set(jug, "y", y)
        initial.set(jug, "water_volume", 1.2 if mode == "fill" else 0.0)
    real._set_state(initial)
    clean = sanitize_state(initial)
    assert clean.privileged is None
    model._set_state(clean)
    tracker = make_subclass_latent_tracker(cls, lambda: {})
    assert tracker is not None
    tracker.attach(clean, None)
    errors = []
    witnesses: List[Tuple[float, int, str, str, float, float]] = []
    for index in range(55):
        action = _make_noop(real._get_state(),
                            real)  # type: ignore[no-untyped-call]
        real._step_once(action)
        model._step_once(action)
        actual, predicted = real._get_state(), model._get_state()
        observed = sanitize_state(actual)
        assert observed.privileged is None
        inferred = tracker.attach(observed, action)
        assert not tracker.failed
        assert inferred.latent is not None
        assert np.isclose(inferred.latent["heat"].get(jug.name, 0.0),
                          real._heat_of(jug))
        assert np.isclose(inferred.latent["spill"],
                          real._faucet._spilled_level)
        for obj in actual:
            if obj.type.name in {"jug", "faucet", "human", "burner"}:
                errors.extend(abs(actual[obj] - predicted[obj]))
                witnesses.extend(
                    (abs(float(a) - float(b)), index, obj.name, feat, float(a),
                     float(b)) for feat, a, b in zip(
                         obj.type.feature_names, actual[obj], predicted[obj]))
        if mode == "heat" and index < 2:
            assert np.isclose(real._heat_of(jug), index * real.heating_speed)
    assert max(errors) < 1e-5, max(witnesses)
    if mode == "heat":
        assert np.isclose(real._heat_of(jug), 1.0)
    elif mode == "not_full":
        assert real._heat_of(jug) == 0.0
    else:
        assert real._faucet._spilled_level > 0.0
    # A restored prediction uses inferred memory, even if a caller tries to
    # smuggle contradictory privileged heat into the starting frame.
    inferred.privileged = {jug.name: {"heat_level": 123.0}}
    model._set_state(inferred)
    assert np.isclose(model._heat_of(jug), real._heat_of(jug))
    assert model._faucet.prev_on == real._faucet.prev_on
    assert np.isclose(model._faucet._spilled_level,
                      real._faucet._spilled_level)
    for _ in range(3):
        action = _make_noop(real._get_state(),
                            real)  # type: ignore[no-untyped-call]
        real._step_once(action)
        model._step_once(action)
        actual, predicted = real._get_state(), model._get_state()
        for obj in actual:
            for feature in ("water_volume", "bubbling_level", "spilled_level"):
                if feature in obj.type.feature_names:
                    assert np.isclose(actual.get(obj, feature),
                                      predicted.get(obj, feature))


@pytest.mark.parametrize("mode", ["flush", "out_of_range", "wetting"])
def test_bridge_oracle_process_and_observed_memory(mode: str,
                                                   monkeypatch: Any) -> None:
    """Wetting, cure, tacks, and reciprocal latches match native dynamics."""
    import pybullet as p  # pylint: disable=import-outside-toplevel

    from tests.envs.test_pybullet_bridge import \
        _stage_flush_pair  # pylint: disable=import-outside-toplevel
    utils.reset_config({
        "env": "pybullet_bridge",
        "seed": 0,
        "partially_observable": True,
        "num_train_tasks": 1,
        "num_test_tasks": 0
    })
    real: Any = create_new_env(CFG.env, do_cache=False, use_gui=False)
    task = real.get_train_tasks()[0]
    if mode == "wetting":
        real._set_state(task.init)
        initial = real._get_state()
        block = next(o for o in initial if o.name == "leg0")
        dab = real._face_dab_point(initial, block, "end_b")
        for feature, value in zip(("x", "y", "z"), dab):
            initial.set(real._bottle, feature, value)
        initial.set(real._bottle, "z", dab[2] + real.bottle_half_extents[2])
        initial.set(real._bottle, "is_held", 1.0)
        initial.set(real._robot, "x", dab[0])
        initial.set(real._robot, "y", dab[1])
        initial.set(real._robot, "z",
                    dab[2] + 2 * real.bottle_half_extents[2] + 0.005)
        initial.set(real._robot, "fingers", real.closed_fingers)
        real._set_state(initial)
    else:
        block, mate = _stage_flush_pair(real,
                                        task)  # type: ignore[no-untyped-call]
        if mode == "out_of_range":
            initial = real._get_state()
            initial.set(mate, "x", initial.get(mate, "x") + 0.08)
            real._set_state(initial)
    clean = sanitize_state(real._get_state())
    cls = _load()
    model = cls(use_gui=False, skip_residual_dynamics=False)
    model._set_state(clean)
    tracker = make_subclass_latent_tracker(cls, lambda: {})
    assert tracker is not None
    tracker.attach(clean, None)

    def forbidden(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("The observation callback accessed an engine")

    max_error = 0.0
    for _ in range(35):
        action = _make_noop(real._get_state(),
                            real)  # type: ignore[no-untyped-call]
        real._step_once(action)
        model._step_once(action)
        actual, predicted = real._get_state(), model._get_state()
        with monkeypatch.context() as context:
            for method in ("connect", "getBasePositionAndOrientation",
                           "createConstraint", "stepSimulation"):
                context.setattr(p, method, forbidden)
            inferred = tracker.attach(sanitize_state(actual), action)
        assert not tracker.failed
        assert inferred.latent is not None
        for obj in actual.get_objects(real._block_type):
            native = real._hidden_block_features(obj)
            estimate = inferred.latent["blocks"][obj.name]
            for key, value in native.items():
                expected = value
                if key.startswith("attached_"):
                    expected = real._blocks[int(
                        value)].name if value >= 0 else None
                assert estimate[key] == expected, (mode, obj.name, key,
                                                   estimate[key], expected)
            max_error = max(max_error,
                            float(np.max(abs(actual[obj] - predicted[obj]))))
        assert len(model._weld_constraints) == len(real._weld_constraints)
        assert len(model._tack_constraints) == len(real._tack_constraints)
    assert max_error < 0.005, max_error
    if mode == "flush":
        assert len(real._weld_constraints) == 1
        assert actual.get(block, "glue_end_b") == 0.0
        model._set_state(inferred)
        assert len(model._weld_constraints) == 1
        model._set_state(sanitize_state(task.init))
        assert not model._weld_constraints
        assert not model._tack_constraints
    elif mode == "wetting":
        assert actual.get(block, "glue_end_b") == 1.0
    else:
        assert not real._weld_constraints


@pytest.mark.parametrize("domain",
                         ["bridge", "fan", "domino", "boil", "balloons"])
def test_scene_only_physical_calibration(domain: str) -> None:
    """Match native body and articulation calibration without hidden
    effects."""
    import copy  # pylint: disable=import-outside-toplevel

    import pybullet as p  # pylint: disable=import-outside-toplevel

    from predicators.approaches.agent_continual_frozen_approach import \
        AgentContinualSceneOnlyApproach  # pylint: disable=import-outside-toplevel
    from scripts.cluster_utils import \
        generate_run_configs  # pylint: disable=import-outside-toplevel
    config = next(c for c in generate_run_configs(
        "predicatorv3/continual_eight_agent_noisy_sweep.yaml", False)
                  if c.env == f"pybullet_{domain}"
                  and c.approach == "agent_continual_scene_only")
    utils.reset_config({
        **{k: v
           for k, v in config.flags.items() if k != "log"}, "env": config.env,
        "seed": 0
    })
    real: Any = create_new_env(CFG.env, do_cache=False, use_gui=False)
    namespace: Dict[str, Any] = {
        "BaseSimulator": base_simulator_class(CFG.env),
        "ParamSpec": ParamSpec
    }
    exec(AgentContinualSceneOnlyApproach._scene_source(), namespace)  # pylint: disable=exec-used
    model = namespace["RESIDUAL_ENV"](use_gui=False)
    initial = real.get_train_tasks()[0].init
    real._set_state(copy.deepcopy(initial))
    public = sanitize_state(copy.deepcopy(initial))
    assert public.privileged is None
    model._set_state(public)
    native = real._get_state()
    predicted = model._get_state()
    checked = 0
    articulated = 0
    for obj in native:
        peer = next(o for o in predicted if o.name == obj.name)
        if obj.id is None or peer.id is None:
            continue
        rc, mc = real._physics_client_id, model._physics_client_id
        joints = p.getNumJoints(obj.id, physicsClientId=rc)
        assert joints == p.getNumJoints(peer.id, physicsClientId=mc)
        articulated += joints
        for link in range(-1, joints):
            actual = p.getDynamicsInfo(obj.id, link, physicsClientId=rc)
            expected = p.getDynamicsInfo(peer.id, link, physicsClientId=mc)
            # Mass, friction, inertia, restitution and contact calibration.
            for index in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9):
                assert np.allclose(actual[index],
                                   expected[index]), (domain, obj.name, link,
                                                      index, actual, expected)
            if link >= 0:
                a = p.getJointInfo(obj.id, link, physicsClientId=rc)
                b = p.getJointInfo(peer.id, link, physicsClientId=mc)
                for index in range(1, len(a)):
                    if isinstance(a[index], bytes):
                        assert a[index] == b[index]
                    else:
                        assert np.allclose(a[index], b[index])
            checked += 1
    assert checked > 0 and articulated > 0
    if domain == "balloons":
        assert model._drag() == real._drag()
        for index in range(len(real.BOX_PALETTE)):
            assert model._box_mass_for(index) == real._box_mass_for(index)
    before = model._get_state().copy()
    model._domain_specific_step()
    assert before.allclose(model._get_state())
