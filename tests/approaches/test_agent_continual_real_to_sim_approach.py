"""The agentic real-to-sim arm: the agent builds the simulator itself."""
import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Set

import pybullet as p
import pytest

from predicators.approaches import create_approach
from predicators.approaches.agent_continual_real_to_sim_approach import \
    AgentContinualRealToSimApproach
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

CONFIG = "predicatorv3/continual_real_to_sim_benchmark_r1.yaml"
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
    RESIDUAL_FEATURES = {"jug": ["water_level"]}

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
        pass


RESIDUAL_ENV = BoilScene
'''


@pytest.fixture(autouse=True)
def _dispose_test_physics_clients(monkeypatch: Any) -> Iterator[None]:
    """Release every world a case opened."""
    # pylint: disable=import-outside-toplevel
    from predicators import envs
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
        for name, env in list(envs._MOST_RECENT_ENV_INSTANCE.items()):
            if getattr(env, "_physics_client_id", None) in owned:
                del envs._MOST_RECENT_ENV_INSTANCE[name]
        for client in owned:
            if p.isConnected(client):
                p.disconnect(client)


def _arm_flags() -> Dict[str, Any]:
    cfg = next(c for c in generate_run_configs(CONFIG, False)
               if c.env == "pybullet_boil")
    flags = {k: v for k, v in cfg.flags.items() if k != "log"}
    flags.update(approach=cfg.approach,
                 env=cfg.env,
                 continual_render=False,
                 continual_make_video=False)
    return flags


def _make(tmp_path: Any) -> Any:
    _config(tmp_path, **_arm_flags())
    env = create_new_env("pybullet_boil", do_cache=False, use_gui=False)
    approach = create_approach("agent_continual_real_to_sim", env.predicates,
                               get_gt_options(env.get_name()), env.types,
                               env.action_space,
                               [t.task for t in env.get_train_tasks()])
    assert isinstance(approach, AgentContinualRealToSimApproach)
    return env, approach


def test_config_is_the_benchmark_arm() -> None:
    """Five settings, three seeds, no fitting and no uncertainty flags."""
    runs = list(generate_run_configs(CONFIG, False))
    assert len(runs) == 15
    for run in runs:
        assert run.approach == "agent_continual_real_to_sim"
        assert run.flags["agent_sim_learn_declared_params_only"] is True
        assert run.flags["continual_uncertainty_decisions"] is False
        assert run.flags["continual_require_model_on_test"] is True


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
def test_agent_builds_the_scene_and_rehearses_in_it(tmp_path: Any,
                                                    monkeypatch: Any) -> None:
    """References, prompt, probe refusal before a model, then a real skill
    rehearsed inside the agent-built scene, and the gate accepting it."""
    env, approach = _make(tmp_path)
    prompt = approach._get_agent_system_prompt()
    assert "SceneBase" in prompt and "scene_manifest.json" in prompt
    assert "BaseSimulator" not in prompt and "sim.fit()" not in prompt
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
        assert "sim.fit" not in tools["run_python"].description
        assert "SUPPLIED" not in tools["run_python"].description
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
        # The world behind sim is the agent's class, not the twin.
        assert type(approach._base_env).__name__ == "BoilScene"
        assert approach._tool_context.env is approach._base_env
        assert approach._model_readiness(str(sandbox / "simulator.py"),
                                         []) is None
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
    fit = approach._last_fit_result
    assert fit is not None and fit.names == ["heat_rate"]
    assert fit.samples[0, 0] == pytest.approx(0.01)
