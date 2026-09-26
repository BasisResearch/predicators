"""The realistic sim gap: the live world deviates, its description does not."""
from typing import Any, Dict, Tuple

import numpy as np
import pybullet as p
import pytest

from predicators import utils
from predicators.code_sim_learning.scene_manifest import build_scene_manifest
from predicators.envs.pybullet_bridge import PyBulletBridgeEnv
from predicators.pybullet_helpers import world_gap
from predicators.structs import Action

# pylint: disable=protected-access

_BASE = {
    "env": "pybullet_bridge",
    "num_train_tasks": 1,
    "num_test_tasks": 0,
}


def _bodies(env: Any) -> Dict[int, Tuple[Tuple[float, ...], float, float]]:
    """Body id -> (collision dimensions, base mass, base lateral friction) of
    every non-robot body."""
    client = env._physics_client_id
    out = {}
    for index in range(p.getNumBodies(physicsClientId=client)):
        body = p.getBodyUniqueId(index, physicsClientId=client)
        if body == env._pybullet_robot.robot_id:
            continue
        shapes = p.getCollisionShapeData(body, -1, physicsClientId=client)
        dims = tuple(float(v) for v in shapes[0][3]) if shapes else ()
        info = p.getDynamicsInfo(body, -1, physicsClientId=client)
        out[body] = (dims, float(info[0]), float(info[1]))
    return out


def _step(env: Any) -> None:
    env.step(
        Action(np.array(env._pybullet_robot.get_joints(), dtype=np.float32)))


def _make(seed: int, gap: bool, twin: bool = False) -> Any:
    utils.reset_config({**_BASE, "seed": seed, "sim_gap": gap})
    env = PyBulletBridgeEnv(use_gui=False, skip_residual_dynamics=twin)
    env._set_state(env._generate_train_tasks()[0].init)
    return env


@pytest.fixture(name="worlds")
def _worlds() -> Any:
    made = []

    def make(*args: Any, **kwargs: Any) -> Any:
        env = _make(*args, **kwargs)
        made.append(env)
        return env

    yield make
    for env in made:
        if p.isConnected(env._physics_client_id):
            env.dispose()


def test_flag_off_builds_the_nominal_world(worlds: Any) -> None:
    """Without the flag the live world equals its twin, before and after a
    step."""
    live, twin = worlds(0, False), worlds(0, False, twin=True)
    assert live._world_gap is None
    _step(live)
    assert _bodies(live) == _bodies(twin)


def test_live_world_deviates_and_its_description_does_not(worlds: Any) -> None:
    """Movable sizes, masses and frictions differ from the twin; static bodies
    keep their shape; the manifest built from a twin stays nominal."""
    live, twin = worlds(0, True), worlds(0, True, twin=True)
    assert live._world_gap is not None and twin._world_gap is None
    _step(live)
    _step(twin)
    live_bodies, twin_bodies = _bodies(live), _bodies(twin)
    assert live_bodies.keys() == twin_bodies.keys()
    resized = frictions = masses = 0
    for body, (dims, mass, friction) in twin_bodies.items():
        live_dims, live_mass, live_friction = live_bodies[body]
        if mass == 0.0:
            assert live_dims == dims and live_mass == 0.0
        else:
            ratio = np.array(live_dims) / np.array(dims)
            assert np.allclose(ratio, ratio[0])
            assert abs(ratio[0] - 1.0) <= 0.03 + 1e-9
            resized += not np.isclose(ratio[0], 1.0)
            assert 1 / 1.25 - 1e-9 <= live_mass / mass <= 1.25 + 1e-9
            masses += not np.isclose(live_mass, mass)
        if friction > 0.0:
            assert 1 / 1.3 - 1e-9 <= live_friction / friction <= 1.3 + 1e-9
            frictions += not np.isclose(live_friction, friction)
    assert resized and masses and frictions
    state = twin._get_state()
    manifest, _ = build_scene_manifest(twin, state)
    nominal, _ = build_scene_manifest(worlds(0, False, twin=True), state)
    assert manifest["bodies"] == nominal["bodies"]


def test_deviations_hold_per_seed_and_never_compound(worlds: Any) -> None:
    """Same seed, same world; another seed, another world; repeated steps leave
    the drawn values in place, and a domain reset of a body's dynamics is
    deviated again, not restored."""
    first, again, other = worlds(0, True), worlds(0, True), worlds(1, True)
    _step(first)
    _step(again)
    _step(other)
    assert _bodies(first) == _bodies(again)
    assert _bodies(first) != _bodies(other)
    before = _bodies(first)
    _step(first)
    _step(first)
    assert _bodies(first) == before
    nominal = _bodies(worlds(0, True, twin=True))
    body = next(b for b, (_, mass, _) in nominal.items() if mass > 0.0)
    client = first._physics_client_id
    p.changeDynamics(body, -1, mass=nominal[body][1], physicsClientId=client)
    _step(first)
    assert _bodies(first)[body] == before[body]


def test_registry_forgets_a_disposed_world(worlds: Any) -> None:
    """A disposed live world's gap does not reach a later world on its client
    id."""
    live = worlds(0, True)
    client = live._physics_client_id
    assert world_gap._WORLDS[client] is live._world_gap
    live.dispose()
    assert client not in world_gap._WORLDS
    assert world_gap.size_scale(client, movable=True) == 1.0


@pytest.mark.parametrize("env_name", [
    "pybullet_boil", "pybullet_domino", "pybullet_fan", "pybullet_bridge",
    "pybullet_balloons"
])
def test_each_benchmark_domain_runs_with_a_gap(env_name: str) -> None:
    """Every benchmark domain builds, resets and steps its live world with the
    gap and the engine settings on."""
    # pylint: disable-next=import-outside-toplevel
    from predicators.envs import create_new_env
    utils.reset_config({
        "env": env_name,
        "seed": 0,
        "num_train_tasks": 1,
        "num_test_tasks": 1,
        "sim_gap": True,
        "sim_gap_solver_iterations": 100,
        "sim_gap_substeps": 2,
    })
    env: Any = create_new_env(env_name, do_cache=False, use_gui=False)
    try:
        assert env._world_gap is not None
        env.reset("train", 0)
        for _ in range(3):
            env.step(
                Action(
                    np.array(env._pybullet_robot.get_joints(),
                             dtype=np.float32)))
        state = env._get_state()
        for obj in state:
            assert np.all(np.isfinite(state[obj])), (env_name, obj.name)
        engine = p.getPhysicsEngineParameters(
            physicsClientId=env._physics_client_id)
        assert engine["numSolverIterations"] == 100
    finally:
        env.dispose()


def _twin_with_menu(seed: int = 0) -> Any:
    utils.reset_config({
        **_BASE, "seed": seed,
        "sim_gap": True,
        "sim_calibration_menu": True
    })
    env = PyBulletBridgeEnv(use_gui=False, skip_residual_dynamics=True)
    env._set_state(env._generate_train_tasks()[0].init)
    return env


def test_calibration_menu_is_a_no_op_until_fitted(worlds: Any) -> None:
    """The twin lists per-type scales at 1.0 and, unfitted, steps exactly like
    a twin without the menu; the gapped live world lists none."""
    twin = _twin_with_menu()
    try:
        info = twin.get_physical_param_info()
        scales = {k: v for k, v in info.items() if "_scale_" in k}
        assert "friction_scale_support" in scales
        assert any(k.startswith("mass_scale_") for k in scales)
        assert not any(k.endswith("_robot") for k in scales)
        assert all(v["default"] == 1.0 for v in scales.values())
        plain = worlds(0, False, twin=True)
        _step(twin)
        _step(plain)
        assert _bodies(twin) == _bodies(plain)
        utils.reset_config({
            **_BASE, "seed": 0,
            "sim_gap": True,
            "sim_calibration_menu": True
        })
        live = PyBulletBridgeEnv(use_gui=False)
        try:
            assert live._calibration is None
            assert not any("_scale_" in k
                           for k in live.get_physical_param_info())
        finally:
            live.dispose()
    finally:
        twin.dispose()


def test_fitted_scales_apply_per_type_and_never_compound(worlds: Any) -> None:
    """A fitted mass scale multiplies every body of its type, the support scale
    every static body, and repeated steps keep the values."""
    twin = _twin_with_menu()
    try:
        nominal = _bodies(worlds(0, False, twin=True))
        types = twin._body_types()
        name = next(k for k in twin.get_physical_param_info()
                    if k.startswith("mass_scale_"))
        type_name = name[len("mass_scale_"):]
        twin.apply_physical_param_overrides({
            name: 1.5,
            "friction_scale_support": 0.5
        })
        for _ in range(3):
            _step(twin)
        after = _bodies(twin)
        for body, (_, mass, friction) in nominal.items():
            if types.get(body) == type_name:
                assert np.isclose(after[body][1], 1.5 * mass)
            elif mass == 0.0:
                assert np.isclose(after[body][2], 0.5 * friction)
        twin.apply_physical_param_overrides({name: 1.0})
        _step(twin)
        for body, (_, mass, _) in nominal.items():
            if types.get(body) == type_name:
                assert np.isclose(_bodies(twin)[body][1], mass)
    finally:
        twin.dispose()


@pytest.mark.parametrize("env_name, present, absent", [
    ("pybullet_boil", {"mass_scale_jug"}, set()),
    ("pybullet_fan", {"mass_scale_ball", "friction_scale_ball"}, set()),
    ("pybullet_bridge", {"mass_scale_block", "mass_scale_bottle"}, set()),
    ("pybullet_domino", {"lateral_friction"},
     {"mass_scale_domino", "friction_scale_domino", "mass_scale_block"}),
    ("pybullet_balloons", {"mass_oak", "air_drag"}, {"mass_scale_box"}),
])
def test_each_domain_lists_its_calibration_menu(env_name: str, present: set,
                                                absent: set) -> None:
    """Every benchmark twin merges the calibration scales into its menu,
    skipping the types its own menu covers, and accepts them back."""
    # pylint: disable-next=import-outside-toplevel
    from predicators.envs import create_new_env
    utils.reset_config({
        "env": env_name,
        "seed": 0,
        "num_train_tasks": 1,
        "num_test_tasks": 1,
        "sim_calibration_menu": True,
    })
    env: Any = create_new_env(env_name,
                              do_cache=False,
                              use_gui=False,
                              skip_residual_dynamics=True)
    try:
        info = env.get_physical_param_info()
        assert "friction_scale_support" in info
        assert present <= set(info), sorted(info)
        assert not absent & set(info), sorted(info)
        assert not any(k.endswith("_robot") for k in info)
        env.apply_physical_param_overrides({"friction_scale_support": 1.2})
        assert env._calibration.values["friction_scale_support"] == 1.2
    finally:
        env.dispose()


def test_manifest_ignores_a_disconnected_worlds_assets() -> None:
    """A world built on a client id a disconnected world used describes only
    its own bodies' assets."""
    # pylint: disable-next=import-outside-toplevel
    from predicators.envs import create_new_env

    def twin(env_name: str) -> Any:
        utils.reset_config({
            "env": env_name,
            "seed": 0,
            "num_train_tasks": 1,
            "num_test_tasks": 0
        })
        return create_new_env(env_name,
                              do_cache=False,
                              use_gui=False,
                              skip_residual_dynamics=True)

    clean = twin("pybullet_bridge")
    state = clean._generate_train_tasks()[0].init
    expected, _ = build_scene_manifest(clean, state)
    clean.dispose()
    # A URDF-heavy world disconnected without releasing its records.
    boil = twin("pybullet_boil")
    stale_client = boil._physics_client_id
    p.disconnect(stale_client)
    reused = twin("pybullet_bridge")
    try:
        assert reused._physics_client_id == stale_client
        manifest, _ = build_scene_manifest(reused, state)
        assert manifest["bodies"] == expected["bodies"]
    finally:
        reused.dispose()
