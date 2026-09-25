"""Exercise the task certificate through the agent's actual simulator API."""
# pylint: disable=protected-access
import dataclasses
from typing import Any, Dict

import pybullet as p
import pytest

from predicators import utils
from predicators.agent_sdk.belief_probe import BeliefProbe
from predicators.agent_sdk.tools import ToolContext
from predicators.code_sim_learning.base_simulator import \
    base_simulator_class, oracle_base_simulator_class
from predicators.code_sim_learning.continual_oracle import oracle_source
from predicators.code_sim_learning.fit_space import ParamSpec
from predicators.code_sim_learning.scene_base import scene_base_class
from predicators.code_sim_learning.scene_manifest import build_scene_manifest
from predicators.envs import create_new_env
from predicators.envs.pybullet_domino import cascade_probe
from predicators.envs.pybullet_domino.components.domino_component import \
    DominoComponent
from predicators.ground_truth_models import get_gt_options
from predicators.option_model import _OracleOptionModel
from predicators.run.recording import sanitize_state
from predicators.structs import State


def _scene_class(real: Any, initial: State) -> Any:
    """A small agent-style scene built only from the public geometry."""
    manifest, _ = build_scene_manifest(real, initial)
    base = scene_base_class("pybullet_domino", real.types,
                            utils.get_env_asset_path(""))

    class DominoScene(base):  # type: ignore[valid-type,misc]
        """A declared-friction model with no domain evaluator methods."""
        AGENT_PARAM_SPECS = [ParamSpec("friction", 0.1, lo=0.0, hi=1.0)]

        @classmethod
        def initialize_pybullet(cls, using_gui):
            client, robot, bodies = super().initialize_pybullet(using_gui)
            # Deliberately different ids from the supplied scene.
            for body in reversed(manifest["bodies"]):
                if body.get("urdf"):
                    bid = p.loadURDF(cls.asset(body["urdf"][len("assets/"):]),
                                     useFixedBase=body["static"],
                                     globalScaling=body["urdf_scale"],
                                     physicsClientId=client)
                else:
                    shape = body["links"][0]["shapes"][0]
                    assert shape["geometry"] == "box"
                    half = [v / 2 for v in shape["dimensions"]]
                    cid = p.createCollisionShape(
                        p.GEOM_BOX,
                        halfExtents=half,
                        collisionFramePosition=shape["local_position"],
                        physicsClientId=client)
                    bid = p.createMultiBody(0.0 if body["static"] else 0.1,
                                            cid,
                                            physicsClientId=client)
                    p.changeDynamics(bid,
                                     -1,
                                     lateralFriction=0.1,
                                     spinningFriction=0.1,
                                     rollingFriction=0.006,
                                     restitution=0.02,
                                     linearDamping=0.0,
                                     angularDamping=0.03,
                                     frictionAnchor=True,
                                     physicsClientId=client)
                if body["object"] is None:
                    pose = body["base_pose"]
                    p.resetBasePositionAndOrientation(bid,
                                                      pose["position"],
                                                      pose["orientation"],
                                                      physicsClientId=client)
                else:
                    bodies[body["object"]] = bid
            for name in manifest["objects"]:
                bodies.setdefault(name, None)
            return client, robot, bodies

        def _on_agent_params_changed(self):
            for name, bid in self.bodies.items():
                if name.startswith("domino") and bid is not None:
                    p.changeDynamics(
                        bid,
                        -1,
                        lateralFriction=self.agent_param("friction"),
                        physicsClientId=self._physics_client_id)

    return DominoScene


@pytest.mark.parametrize("kind", ["oracle", "learned", "scene"])
def test_subclass_trials_have_real_evaluator_verdicts(
        kind: str, monkeypatch: Any) -> None:
    """Internal model names must not break solved=True counterfactuals."""
    utils.reset_config({
        "env": "pybullet_domino",
        "seed": 0,
        "num_train_tasks": 0,
        "num_test_tasks": 1,
        "domino_initialize_at_finished_state": True,
        "domino_use_domino_blocks_as_target": True,
        "domino_use_continuous_place": True,
        "domino_has_glued_dominos": False,
        # The scene fixture deliberately omits the domain-specific body
        # construction and uses a straight chain to isolate certification
        # plumbing from corner-layout contact sensitivity.
        "domino_test_turn_ratio": 0.0 if kind == "scene" else 1.0,
        "domino_true_friction": 0.1,
        "domino_planning_friction": 0.1,
    })
    real: Any = create_new_env("pybullet_domino",
                               do_cache=False,
                               use_gui=False)
    namespace: Dict[str, Any] = {
        # Oracle artifacts load on the privileged base; learned and scene
        # models get only the visible core.
        "BaseSimulator":
        (oracle_base_simulator_class("pybullet_domino")
         if kind == "oracle" else base_simulator_class("pybullet_domino")),
        "ParamSpec":
        ParamSpec,
    }
    source = oracle_source() if kind == "oracle" else '''
class LearnedDynamics(BaseSimulator):
    AGENT_PARAM_SPECS = [ParamSpec("friction", 0.1, lo=0.0, hi=1.0)]
    RESIDUAL_FEATURES = {}
    def _on_agent_params_changed(self):
        self.apply_physical_param_overrides(
            {"lateral_friction": self.agent_param("friction")})
RESIDUAL_ENV = LearnedDynamics
'''
    exec(source, namespace)  # pylint: disable=exec-used
    task = real.get_test_tasks()[0].task
    cls = _scene_class(real, task.init) if kind == "scene" else \
        namespace["RESIDUAL_ENV"]
    candidate = cls(use_gui=False)
    # The continual agent receives portable observations, not deployment
    # body ids. This also exercises scene rebinding on the first action.
    task = dataclasses.replace(task, init=sanitize_state(task.init))
    try:
        candidate._set_state(task.init)
        observed = candidate._get_state()
        for obj in task.init:
            if obj.type.name == "domino":
                assert observed.get(obj, "z") == pytest.approx(task.init.get(
                    obj, "z"),
                                                               abs=1e-5)
        options = get_gt_options("pybullet_domino", skill_library="composite")
        model = _OracleOptionModel(options, candidate.simulate)
        model.sim_env = candidate
        ctx = ToolContext(types=real.types,
                          predicates=real.predicates,
                          processes=set(),
                          options=options,
                          train_tasks=[task],
                          example_state=task.init,
                          current_task=task,
                          option_model=model)
        start = next(o for o in task.init if o.type.name == "domino"
                     and DominoComponent._StartBlock_holds(task.init, [o]))
        robot = next(o for o in task.init if o.type.name == "robot")
        probe = BeliefProbe(ctx).reset(task_idx=0)
        result = probe.run(f"Push({robot}, {start})[0.04, 0.05]",
                           trials=2,
                           solved=True)
        assert all(t["solved"] is not None for t in result.trials), str(result)
        assert result.successes == 2, str(result)
        assert all(t["solved"] for t in result.trials), str(result)
        assert all(t["evaluation_status"] == "accepted" for t in result.trials)
        # Every certificate actually ran the fingertips-only probe.
        assert all("fingertips-only" in t["note"] for t in result.trials)
        before = candidate._get_state().copy()
        if kind == "oracle":
            ok, detail = candidate.run_counterfactual_cascade_probe(
                task.init, [start], task.goal, (0.04, 0.05))
            assert ok, detail
            assert before.allclose(candidate._get_state())
        if kind == "learned":
            # The visible core carries no privileged certificate helpers.
            assert not hasattr(candidate, "run_counterfactual_cascade_probe")
            assert not hasattr(candidate, "_get_cascade_probe_env")
        if kind == "scene":
            candidate.apply_physical_param_overrides({"friction": 0.6})
            clients = []

            def fail_after_checking_model(probe_env, *_args, **_kwargs):
                assert type(probe_env) is type(candidate)
                assert probe_env.agent_param("friction") == 0.6
                cid = probe_env._physics_client_id
                clients.append(cid)
                assert cid != candidate._physics_client_id
                assert p.getDynamicsInfo(probe_env.body(start.name),
                                         -1,
                                         physicsClientId=cid)[1] == 0.6
                raise RuntimeError("probe test interruption")

            monkeypatch.setattr(cascade_probe, "run_counterfactual_push_probe",
                                fail_after_checking_model)
            with pytest.raises(RuntimeError, match="probe test interruption"):
                cascade_probe.run_model_cascade_probe(candidate, task.init,
                                                      [start], task.goal,
                                                      (0.04, 0.05))
            assert clients and all(not p.isConnected(cid) for cid in clients)
            assert before.allclose(candidate._get_state())
    finally:
        candidate.dispose()
        real.dispose()
