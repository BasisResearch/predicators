"""Exercise shadow validation through the real continual tool boundary."""
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

from predicators.agent_sdk.preflight_audit import PreflightAudit, \
    terminal_prefix_length
from predicators.agent_sdk.restoration import restoration_report, same_memory
from predicators.run.continual import ContinualRun
from predicators.run.level_players import create_level_player
from predicators.structs import Dataset, Object, State, Type
from tests.approaches.test_agent_continual_approach import _call, _config, \
    _make_approach, _result


@pytest.mark.slow
def test_shadow_tools_pair_outcomes_without_changing_execution(
        tmp_path: Path) -> None:
    """The same scripted agent spends the same steps with and without audit."""
    # Each callback is invoked synchronously before advancing this loop.
    # pylint: disable=protected-access,cell-var-from-loop
    totals = []
    for enabled in (False, True):
        run_path = tmp_path / str(enabled)
        _config(run_path,
                continual_levels="train_only",
                continual_skill_preflight=False,
                continual_validation_audit=enabled)
        env, approach = _make_approach()
        audit_paths = []

        def fake_query(message: str, **kwargs: Any) -> List[Dict[str, Any]]:
            del message, kwargs
            # A missing model is logged as unknown, not a false pass.
            _call(approach, "skills_invoke", skill="Wait(robot:robot)[1]")
            candidate = Path(
                approach._tool_context.sandbox_dir) / "simulator.py"
            candidate.write_text(
                "class Candidate(BaseSimulator):\n"
                "    AGENT_PARAM_SPECS = []\n"
                "    RESIDUAL_FEATURES = {}\n"
                "RESIDUAL_ENV = Candidate\n",
                encoding="utf-8")
            result = _call(approach,
                           "skills_invoke",
                           skill="Wait(robot:robot)[2]")
            assert "Nothing was charged" not in result
            # Exercise raw-control coverage without changing the joint target.
            joints = env._pybullet_robot.get_joints()
            _call(approach, "env_step", action=list(joints))
            if enabled:
                audit_paths.append(
                    approach._tool_context.execution_audit._path)
            _call(approach, "give_up", note="test complete")
            return _result()

        approach._query_agent_sync = fake_query
        approach.prepare_for_continual(Dataset([]))
        card = ContinualRun(env, approach,
                            create_level_player(env, approach)).run()
        totals.append(card.total_steps)
        assert approach._tool_context.execution_audit is None
        if enabled:
            rows = [
                json.loads(s) for s in audit_paths[0].read_text().splitlines()
            ]
            predictions = [r for r in rows if r["event"] == "prediction"]
            executed = [r for r in rows if r["event"] == "execution"]
            assert len(predictions) == len(executed) == 3
            assert [r["id"]
                    for r in predictions] == [r["id"] for r in executed]
            assert predictions[0]["status"] == "unavailable"
            assert predictions[1]["sim_rollouts"] == 1, predictions[1]
            assert predictions[1][
                "controller_status"] == "accepted", predictions[1]
            assert predictions[2]["request"] == "PrimitiveAction"
            assert predictions[2]["sim_steps"] == 1, predictions[2]
            assert sum(r["steps"] for r in executed) == card.total_steps
        env.dispose()
    assert totals[0] == totals[1] > 0


def test_restore_reports_memory_loss_and_euler_alias(tmp_path: Path) -> None:
    """Snapshot checks use physical orientations, not raw Euler coordinates."""
    _config(tmp_path)
    obj = Object("obj", Type("pose", ["roll", "pitch", "yaw"]))
    state = State({obj: np.zeros(3, dtype=np.float32)})
    after = state.copy()
    after.set(obj, "yaw", 2 * np.pi)
    assert restoration_report(state, after)["status"] == "accepted"
    state.latent = {"held": {"links": np.array([1, 2])}}
    assert not restoration_report(state, after)["memory_preserved"]
    assert same_memory(state.latent, {"held": {"links": np.array([1, 2])}})


def test_shadow_errors_remain_errors_and_budget_survives_resume(
        tmp_path: Path) -> None:
    """Unavailable checks and diagnostic exceptions are never accepted."""
    from types import \
        SimpleNamespace  # pylint: disable=import-outside-toplevel

    from predicators.run.interaction import \
        PrimitiveAction  # pylint: disable=import-outside-toplevel
    from predicators.structs import \
        Action  # pylint: disable=import-outside-toplevel
    _config(tmp_path)
    session = SimpleNamespace(level_index=0)
    session.observe = lambda: SimpleNamespace(ledger=SimpleNamespace(
        level_index=0, episode_steps=4, run_steps=4))
    model = tmp_path / "simulator.py"
    model.write_text("# present", encoding="utf-8")
    path = tmp_path / "audit.jsonl"
    ctx = SimpleNamespace(probe_option_model_provider=None,
                          probe_validation_env_scope=None,
                          current_observation_provider=None)
    audit = PreflightAudit(ctx, session, str(model), str(path))
    row = audit.before(PrimitiveAction(Action(np.zeros(1, dtype=np.float32))))
    assert row["status"] == "unavailable"
    assert row["sim_rollouts"] == 0
    broken = SimpleNamespace()
    row = PreflightAudit(broken, session, str(model), str(path)).before(
        PrimitiveAction(Action(np.zeros(1, dtype=np.float32))))
    assert row["status"] == "error"
    resumed = PreflightAudit(ctx, session, str(model), str(path))
    assert resumed._seconds > 0  # pylint: disable=protected-access


def test_certificate_stops_at_first_terminal_prediction() -> None:
    """A later goal cannot erase an earlier absorbing state."""
    from types import \
        SimpleNamespace  # pylint: disable=import-outside-toplevel
    evaluator = SimpleNamespace(
        terminated_trajectory=lambda states: states[-1] == 1)
    assert terminal_prefix_length(evaluator, [0, 0, 1, 0, 1], 1) == 3
    assert terminal_prefix_length(evaluator, [0, 0], 1) is None
