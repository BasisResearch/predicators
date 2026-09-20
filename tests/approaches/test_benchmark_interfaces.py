"""Continual benchmark contracts through the real agent tool loop."""
# pylint: disable=protected-access
from typing import Any, Dict, List

import pytest

from predicators.agent_sdk.belief_probe import BeliefProbe
from predicators.run.continual import ContinualRun
from predicators.run.level_players import create_level_player
from predicators.structs import Dataset, Predicate
from tests.approaches.test_agent_continual_approach import _call, _config, \
    _make_approach, _result


@pytest.mark.parametrize("contract", [
    "observations", "budget", "episode_budget", "wait", "wait_negative",
    "wait_unmet", "wait_unannotated"
])
def test_continual_interface(tmp_path: Any, monkeypatch: Any,
                             contract: str) -> None:
    """Task reset, budget advice, and annotated Wait use execution
    semantics."""
    _config(
        tmp_path,
        continual_obs_noise_position=0.01,
        continual_belief_frame=False,
        continual_skill_preflight=False,
        continual_episode_horizon=3 if contract == "episode_budget" else None)
    env, approach = _make_approach()

    def fake_query(*_args: Any, **_kwargs: Any) -> List[Dict[str, Any]]:
        ctx = approach._tool_context
        session = approach._play_session
        assert session is not None
        if contract == "observations":
            observed = session.observe().frame
            probe = BeliefProbe(ctx).reset(task_idx=0)
            initial = probe._require_state()
            assert initial.allclose(observed)
            assert initial.privileged is None
            assert ctx.current_task.init.allclose(observed)
            assert approach._train_tasks[0].init.allclose(observed)
            session.reset("check public initial snapshot")
            assert BeliefProbe(ctx).reset(
                task_idx=0)._require_state().allclose(initial)
        elif contract in ("budget", "episode_budget"):
            expected = 3 if contract == "episode_budget" else 1000
            assert ctx.execution_step_budget() == expected
            zero = [0.0] * env.action_space.shape[0]
            assert "step applied" in _call(approach, "env_step", action=zero)
            assert ctx.execution_step_budget() == expected - 1
        else:
            robot_type = next(t for t in ctx.types if t.name == "robot")
            ready = Predicate("Ready", [robot_type],
                              lambda _s, _o: contract == "wait")
            approach._learned_predicates = {ready}
            ctx.predicates = {ready}
            approach._on_predicates_installed()
            options = session.list_skills()
            wait = next(o for o in options if o.name == "Wait")
            objects = [
                next(o for o in ctx.current_task.init if o.is_instance(t))
                for t in wait.types
            ]
            names = ", ".join(str(o) for o in objects)
            target = ("NOT Ready(robot)"
                      if contract == "wait_negative" else "Ready(robot)")
            annotation = ("" if contract == "wait_unannotated" else
                          f" -> {{{target}}}")
            result = _call(approach,
                           "skills_execute_plan",
                           plan=f"Wait({names})[200]{annotation}")
            assert not result.startswith("ERROR"), result
            expected_steps = 1 if contract in ("wait",
                                               "wait_negative") else 200
            assert session.observe().ledger.run_steps == expected_steps
        assert "Give-up recorded" in _call(approach, "give_up", note="done")
        return _result()

    monkeypatch.setattr(approach, "_query_agent_sync", fake_query)
    approach.prepare_for_continual(Dataset([]))
    ContinualRun(env, approach, create_level_player(env, approach)).run()
