"""The agentic real-to-sim arm: the agent builds the simulator itself."""
import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Set

import pybullet as p
import pytest

from predicators.approaches import create_approach
from predicators.code_sim_learning.scene_base import SceneBase, \
    scene_base_class
from predicators.envs import create_new_env
from predicators.ground_truth_models import get_gt_options
from predicators.run.continual import ContinualRun
from predicators.run.level_players import create_level_player
from predicators.structs import Dataset
from scripts.cluster_utils import generate_run_configs
from tests.approaches.test_agent_continual_approach import _call, _config, \
    _result

# pylint: disable=protected-access

# The agentic real-to-sim arm (no domain twin: the generic PyBulletEnv, a
# domain-agnostic SceneBase, the scene manifest and the asset files; the
# harness fits nothing and runs no uncertainty machinery) and the EMPIRIC
# from-assets arm it derives from. Neither is a benchmark arm; each is the
# benchmark's EMPIRIC run with these flags.
REAL_TO_SIM_FLAGS = {
    "agent_sim_learn_declared_params_only": True,
    "continual_uncertainty_decisions": False,
    "agent_sim_learn_param_uncertainty": False,
    "agent_plan_validation_rule_param_margin": False,
    "agent_plan_validation_physics_margin": False,
    "agent_explorer_info_seeking": False,
    "agent_explorer_info_seeking_adaptive": False,
    "agent_explorer_info_seeking_noise_aware": False,
    "code_sim_learning_interval_belief": False,
    "code_sim_learning_carry_posterior": False,
}
FROM_ASSETS_FLAGS = {
    "agent_sim_learn_declared_params_only": False,
    "continual_uncertainty_decisions": True,
}
# The agent's simulator for the Boil training scene, written the way the
# prompt describes: a SceneBase subclass whose initialize_pybullet loads
# every body of the manifest under its observed name.
SCENE_SOURCE = '''
import json
import os
import pybullet as p

MANIFEST = json.load(open(os.path.join(os.path.dirname(SceneBase.asset(
    "urdf/table.urdf")), "..", "..", "scene", "scene_manifest.json")))


def _load(cls, client, body):
    if body.get("urdf"):
        return p.loadURDF(cls.asset(body["urdf"][len("assets/"):]),
                          useFixedBase=body["static"],
                          globalScaling=body["urdf_scale"],
                          physicsClientId=client)
    shape = body["links"][0]["shapes"][0]
    half = [d / 2 for d in shape["dimensions"]]
    collision = p.createCollisionShape(
        p.GEOM_BOX, halfExtents=half,
        collisionFramePosition=shape["local_position"],
        physicsClientId=client)
    visual = p.createVisualShape(
        p.GEOM_BOX, halfExtents=half,
        visualFramePosition=shape["local_position"],
        rgbaColor=body["links"][0].get("visual_rgba", [0.5, 0.5, 0.5, 1]),
        physicsClientId=client)
    return p.createMultiBody(0.0 if body["static"] else 0.5, collision,
                             visual, physicsClientId=client)


class BoilScene(SceneBase):
    AGENT_PARAM_SPECS = [ParamSpec("heat_rate", 0.01, lo=0.0, hi=0.1)]
    RESIDUAL_FEATURES = {"jug": ["water_volume"]}
    MODEL_STATE_INIT = {"ticks": 0}

    @classmethod
    def update_model_state(cls, observation, model_state, params, action):
        model_state["ticks"] += 1

    @classmethod
    def initialize_pybullet(cls, using_gui):
        client, robot, bodies = super().initialize_pybullet(using_gui)
        for body in MANIFEST["bodies"]:
            body_id = _load(cls, client, body)
            if body["object"] is None:
                pose = body["base_pose"]
                p.resetBasePositionAndOrientation(
                    body_id, pose["position"], pose["orientation"],
                    physicsClientId=client)
            else:
                bodies[body["object"]] = body_id
        for name in MANIFEST["objects"]:
            bodies.setdefault(name, None)
        return client, robot, bodies

    def _domain_specific_step(self):
        for obj in self._objects:
            if obj.type.name == "jug":
                key = (obj.name, "water_volume")
                self._feature_store[key] += self.agent_param("heat_rate")


RESIDUAL_ENV = BoilScene
'''


@pytest.fixture(autouse=True)
def _dispose_test_physics_clients(monkeypatch: Any) -> Iterator[None]:
    """Release every world a case opened."""
    # pylint: disable=import-outside-toplevel
    from predicators.ground_truth_models.skill_factories.base import \
        clear_shared_simulator_cache
    owned: Set[int] = set()
    original = p.connect

    def connect(*args: Any, **kwargs: Any) -> int:
        client = original(*args, **kwargs)
        if client >= 0:
            owned.add(client)
        return client

    monkeypatch.setattr(p, "connect", connect)
    try:
        yield
    finally:
        clear_shared_simulator_cache()
        for client in owned:
            if p.isConnected(client):
                p.disconnect(client)


def _benchmark_flags(env_name: str) -> Dict[str, Any]:
    """The benchmark EMPIRIC run's flags on ``env_name``, without the machine-
    specific output paths (tests keep their own)."""
    cfg = next(c for c in generate_run_configs(
        "empiric/benchmark.yaml", False, approaches=["mb_opus"])
               if c.env == env_name)
    return {
        k: v
        for k, v in cfg.flags.items() if k not in ("log", "continual_runs_dir")
    }


def _arm_flags() -> Dict[str, Any]:
    flags = _benchmark_flags("pybullet_boil")
    flags.update(REAL_TO_SIM_FLAGS,
                 approach="agent_continual_real_to_sim",
                 env="pybullet_boil",
                 continual_render=False,
                 continual_make_video=False)
    return flags


def _make(tmp_path: Any, from_assets: bool = False) -> Any:
    flags = _arm_flags()
    name = "agent_continual_real_to_sim"
    if from_assets:
        name = "agent_continual_from_assets"
        flags = _benchmark_flags("pybullet_bridge")
        flags.update(FROM_ASSETS_FLAGS,
                     approach=name,
                     env="pybullet_boil",
                     continual_render=False,
                     continual_make_video=False)
    _config(tmp_path, **flags)
    env = create_new_env("pybullet_boil", do_cache=False, use_gui=False)
    approach = create_approach(name, env.predicates,
                               get_gt_options(env.get_name()), env.types,
                               env.action_space,
                               [t.task for t in env.get_train_tasks()])
    assert approach.get_name() == name
    return env, approach


def test_arm_refuses_uncertainty_machinery(tmp_path: Any) -> None:
    """A launcher that leaves an uncertainty switch on fails at
    construction."""
    flags = _arm_flags()
    flags["continual_uncertainty_decisions"] = True
    _config(tmp_path, **flags)
    env = create_new_env("pybullet_boil", do_cache=False, use_gui=False)
    with pytest.raises(ValueError, match="uncertainty"):
        create_approach("agent_continual_real_to_sim", env.predicates,
                        get_gt_options(env.get_name()), env.types,
                        env.action_space,
                        [t.task for t in env.get_train_tasks()])


def test_scene_base_binds_the_deployment(tmp_path: Any) -> None:
    """The bound base carries the robot and the types, and nothing else."""
    _config(tmp_path, **_arm_flags())
    env = create_new_env("pybullet_boil", do_cache=False, use_gui=False)
    env_cls: Any = type(env)
    base = scene_base_class("pybullet_boil", env.types, str(tmp_path))
    assert issubclass(base, SceneBase)
    assert base.robot_base_pos == env_cls.robot_base_pos
    assert {t.name for t in base._scene_types} == {t.name for t in env.types}
    assert base.get_robot_ee_home_orn() == tuple(
        env_cls.get_robot_ee_home_orn())
    with pytest.raises(FileNotFoundError):
        base.asset("urdf/missing.urdf")
    # No scene: the base has no bodies of its own beyond plane and robot.
    world = base(use_gui=False, skip_residual_dynamics=False)
    assert p.getNumBodies(physicsClientId=world._physics_client_id) <= 6
    with pytest.raises(ValueError, match="loads no body"):
        world._set_state(env.get_train_tasks()[0].task.init)


@pytest.mark.slow
@pytest.mark.parametrize("from_assets", [False, True])
def test_agent_builds_the_scene_and_rehearses_in_it(tmp_path: Any,
                                                    monkeypatch: Any,
                                                    from_assets: bool) -> None:
    """References, prompt, probe refusal before a model, then a real skill
    rehearsed inside the agent-built scene, and the gate accepting it."""
    env, approach = _make(tmp_path, from_assets)
    assert isinstance(approach._base_env, SceneBase)
    assert approach._base_env.get_physical_param_info() == {}
    assert approach._probe_surface().fit == from_assets
    assert approach._probe_surface().uncertainty == from_assets
    prompt = approach._get_agent_system_prompt()
    assert "SceneBase" in prompt and "scene_manifest.json" in prompt
    assert "BaseSimulator" not in prompt
    assert ("sim.fit()" in prompt) == from_assets
    if from_assets:
        assert "EMPIRIC from assets" in prompt
        assert "harness fits nothing" not in prompt
        assert "visible physics from the first round" not in prompt
    assert "visible base physics" not in prompt
    assert "physical parameter menu" not in prompt.lower()
    seen: List[str] = []

    def query(message: str, *_args: Any, **_kwargs: Any) -> Any:
        seen.append(message)
        ctx = approach._tool_context
        sandbox = Path(ctx.sandbox_dir)
        assert "sim` has no world until" in message
        assert "visible base physics" not in message
        # References: engine sources, the manifest and the assets, and no
        # domain source.
        refs = sorted(
            str(q.relative_to(sandbox / "reference"))
            for q in (sandbox / "reference").rglob("*") if q.is_file())
        assert "base_sim/pybullet_env.py" in refs
        assert "base_sim/scene_base.py" in refs
        assert "scene/scene_manifest.json" in refs
        assert "assets/urdf/jug-pixel.urdf" in refs
        assert not any("pybullet_boil" in r for r in refs)
        manifest = json.loads(
            (sandbox / "reference/scene/scene_manifest.json").read_text())
        assert manifest["objects"]["jug0"] == "jug"
        jug = next(b for b in manifest["bodies"] if b["object"] == "jug0")
        assert jug["urdf"] == "assets/urdf/jug-pixel.urdf"
        assert (sandbox / "reference" / jug["urdf"]).is_file()
        text = json.dumps({k: v for k, v in manifest.items() if k != "legend"})
        for word in ("mass", "friction", "damping", "restitution"):
            assert word not in text.lower(), word
        # Before a model, sim has no world: a reset stages the observation,
        # a rollout has nothing to run in.
        _call(approach, "run_python", code="sim.reset(current=True)")
        refused = _call(approach,
                        "run_python",
                        code="sim.run('Wait(robot:robot)[]')")
        assert "has no world yet" in refused
        refused = _call(approach, "run_python", code="sim.render('a')")
        assert refused.startswith("ERROR") or "Error" in refused
        tools = {t.name: t for t in ctx.extra_mcp_tools}
        assert ("sim.fit" in tools["run_python"].description) == from_assets
        assert "SUPPLIED" not in tools["run_python"].description
        # A rule-only file must not silently receive a domain substrate.
        (sandbox / "simulator.py").write_text(
            "RESIDUAL_RULES = []\nPARAM_SPECS = []\n"
            "RESIDUAL_FEATURES = {}\n",
            encoding="utf-8")
        rejected = approach._load_simulator_from_module_file(
            str(sandbox / "simulator.py"))
        assert rejected == (None, None, None, None)
        # The agent writes its scene and rehearses a real skill in it.
        (sandbox / "simulator.py").write_text(SCENE_SOURCE, encoding="utf-8")
        loaded = _call(approach,
                       "run_python",
                       code="sim.reset(current=True)\n"
                       "print(sorted(sim.state()))")
        assert not loaded.startswith("ERROR"), loaded
        assert "jug0:jug" in loaded
        run = _call(approach,
                    "run_python",
                    code="print(sim.run('Wait(robot:robot)[]'))")
        assert not run.startswith("ERROR"), run
        assert "Wait" in run
        restored = _call(
            approach,
            "run_python",
            code="r = sim.reset(current=True).check_restore()\n"
            "assert r['snapshot']['memory_preserved'], r\n"
            "assert r['attachments_preserved'], r\nprint('restored')")
        assert "restored" in restored and not restored.startswith(
            "ERROR"), restored
        if from_assets:
            # EMPIRIC's joint belief rehearses on its joint draws, each on a
            # fresh world at its own planner seed; this replaces trials.
            code = ("r = sim.reset(current=True).run('Wait(robot:robot)[2]')\n"
                    "seeds = {d['planner_seed'] for d in r.draws}\n"
                    "assert len(r.draws) > 1 and len(seeds) == len(r.draws)\n"
                    "print('independent')")
        else:
            code = ("r = sim.reset(current=True).run("
                    "'Wait(robot:robot)[2]', trials=2)\n"
                    "assert r.fresh_env_per_trial\nprint('independent')")
        independent = _call(approach, "run_python", code=code)
        assert "independent" in independent and not independent.startswith(
            "ERROR"), independent
        # The world behind sim is the agent's class, not the twin.
        assert type(approach._base_env).__name__ == "BoilScene"
        assert approach._tool_context.env is approach._base_env
        assert approach._model_readiness(str(sandbox / "simulator.py"),
                                         []) is None
        # Parameter changes, independent fit worlds and reset isolation use
        # the scene class, not a supplied domain simulator.
        worlds = [approach._get_rollout_fit_env()() for _ in range(2)]
        for world in worlds:
            assert type(world).__name__ == "BoilScene"
            world._set_state(env.get_train_tasks()[0].task.init)
        worlds[0].apply_physical_param_overrides({"heat_rate": 0.07})
        assert worlds[0].agent_param("heat_rate") == pytest.approx(0.07)
        assert worlds[1].agent_param("heat_rate") == pytest.approx(0.01)
        if from_assets:
            assert "step applied" in _call(approach,
                                           "env_step",
                                           action=[0.0] *
                                           env.action_space.shape[0])
            fitted = _call(approach, "run_python", code="print(sim.fit())")
            assert not fitted.startswith("ERROR"), fitted
            assert approach._probe_fit_state().get(
                "fit_result") is not None, fitted
        assert "Give-up recorded" in _call(approach, "give_up", note="done")
        return _result()

    monkeypatch.setattr(approach, "_query_agent_sync", query)
    approach.prepare_for_continual(Dataset([]))
    card = ContinualRun(env, approach, create_level_player(env,
                                                           approach)).run()
    assert card.end_note == "done"
    assert len(seen) == 1
    assert approach._last_round_modelled
    # No fit ran: the deployed values are the declared init values
    # (the stand-in result's first sample).
    if not from_assets:
        fit = approach._last_fit_result
        assert fit is not None and fit.names == ["heat_rate"]
        assert fit.samples[0, 0] == pytest.approx(0.01)
