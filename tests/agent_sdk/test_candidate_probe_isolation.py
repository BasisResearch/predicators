"""Candidate rehearsals must isolate physics without changing the model."""
# pylint: disable=protected-access
from typing import Any, Dict

import pytest

from predicators import utils
from predicators.agent_sdk.belief_probe import BeliefProbe
from predicators.agent_sdk.tools.context import ToolContext
from predicators.approaches.agent_sim_learning_approach import \
    AgentSimLearningApproach
from predicators.code_sim_learning.base_simulator import base_simulator_class
from predicators.code_sim_learning.continual_oracle import oracle_source
from predicators.envs import create_new_env
from predicators.ground_truth_models import get_gt_options
from predicators.option_model import _OracleOptionModel
from predicators.run.recording import sanitize_state
from predicators.structs import Task


def test_candidate_public_trials_are_isolated(monkeypatch: Any) -> None:
    """Exercise sim.run on the supplied Oracle, not a mock rollout."""
    utils.reset_config({
        "env": "pybullet_domino",
        "num_train_tasks": 1,
        "num_test_tasks": 0,
        "agent_validation_parallel_workers": 1
    })
    real: Any = create_new_env("pybullet_domino", do_cache=False)
    namespace: Dict[str, Any] = {
        "BaseSimulator": base_simulator_class("pybullet_domino")
    }
    exec(oracle_source(), namespace)  # pylint: disable=exec-used
    cls = namespace["RESIDUAL_ENV"]
    candidate = cls(use_gui=False)
    task = real.get_train_tasks()[0].task
    task = Task(sanitize_state(task.init), task.goal)
    options = get_gt_options("pybullet_domino", skill_library="composite")
    model = _OracleOptionModel(options, candidate.simulate)
    model.sim_env = candidate
    approach = object.__new__(AgentSimLearningApproach)
    approach._base_env = candidate
    approach._residual_env_cls = cls
    approach._identified_physical_params = {}
    approach._option_model = model
    ctx = ToolContext(types=real.types,
                      predicates=real.predicates,
                      processes=set(),
                      options=options,
                      train_tasks=[task],
                      example_state=task.init,
                      current_task=task,
                      option_model=model)
    ctx.probe_option_model_provider = lambda: model
    ctx.validation_env_scope = approach._fresh_validation_env_scope
    # Separate hook explicitly certifies that the scope manages candidates.
    ctx.probe_validation_env_scope = getattr(
        approach, "_fresh_candidate_validation_scope", None)
    approach._tool_context = ctx
    worlds = []
    original_factory = approach._make_planning_base_env

    def factory(**kwargs: Any) -> Any:
        world = original_factory(**kwargs)
        worlds.append(world)
        return world

    monkeypatch.setattr(approach, "_make_planning_base_env", factory)
    candidate._set_state(task.init)
    before = candidate._get_state().copy()
    robot = next(o for o in task.init if o.type.name == "robot")
    plan = f"Wait({robot})[2]"
    try:
        result = BeliefProbe(ctx).reset(task_idx=0).run(plan, trials=2)
        assert result.fresh_env_per_trial, str(result)
        assert len(worlds) == 2
        assert all(w is not candidate for w in worlds)
        assert model.sim_env is candidate
        assert approach._base_env is candidate
        assert candidate._get_state().allclose(before)
        single = BeliefProbe(ctx).reset(task_idx=0).run(plan, fresh=True)
        assert single.steps
        assert len(worlds) == 3
        restored = BeliefProbe(ctx).reset(task_idx=0).check_restore()
        assert restored["attachments_preserved"]
        assert restored["snapshot"]["memory_preserved"]
        assert candidate._get_state().allclose(before)
        # Exception cleanup restores all bindings, not just successful runs.
        with pytest.raises(RuntimeError, match="injected"):
            with approach._fresh_candidate_validation_scope():
                assert model.sim_env is not candidate
                raise RuntimeError("injected")
        assert model.sim_env is candidate
        assert approach._base_env is candidate
    finally:
        candidate.dispose()
        real.dispose()
