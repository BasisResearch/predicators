"""Behavioral parity and interruption tests on real continual environments."""
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List

import numpy as np
import pytest

from predicators import utils
from predicators.approaches import create_approach
from predicators.envs import create_new_env
from predicators.ground_truth_models import get_gt_options
from predicators.run.continual import ContinualRun, ProtocolSession
from predicators.run.episode import EpisodeState
from predicators.run.interaction import ActionSelector, ExecutePlan, \
    ExecutePolicy, ExecuteSkill, ExecutionObserver, ExecutionProgress, \
    GiveUp, InteractionBusy, PrimitiveAction, RequestReset
from predicators.structs import Action


def _setup(path: Path, **flags: Any) -> Any:
    utils.reset_config({
        "env": "cover",
        "approach": "oracle",
        "seed": 5,
        "num_train_tasks": 1,
        "num_test_tasks": 0,
        "cover_num_blocks": 1,
        "cover_num_targets": 1,
        "cover_block_widths": [0.1],
        "cover_target_widths": [0.05],
        "cover_initial_holding_prob": 0.0,
        "experiment_protocol": "continual",
        "continual_render": False,
        "continual_runs_dir": str(path),
        "continual_steps_per_level": 100,
        **flags,
    })
    env = create_new_env("cover", do_cache=False)
    skills = get_gt_options("cover")
    approach = create_approach("oracle", env.predicates, skills, env.types,
                               env.action_space,
                               [t.task for t in env.get_train_tasks()])
    return env, approach


def _frame(session: ProtocolSession) -> Dict[str, Any]:
    obs = session.observe()
    return {
        "objects": {str(o): obs.frame[o].tolist()
                    for o in obs.frame},
        "state": obs.state.value,
        "steps": obs.ledger.run_steps,
        "resets": obs.ledger.run_resets,
    }


@pytest.mark.parametrize("ending", ["win", "divergence", "horizon", "cap"])
def test_requests_preserve_trajectories(tmp_path: Path, ending: str) -> None:
    """Direct protocol calls and typed requests produce identical transitions,
    observations, counters and stopping behavior, including caps mid-plan."""

    def capture(use_executor: bool) -> Any:
        env, approach = _setup(
            tmp_path / str(use_executor),
            continual_episode_horizon=1 if ending == "horizon" else None,
            continual_steps_per_level=3 if ending == "cap" else 100)
        frames: List[Dict[str, Any]] = []
        actions: List[Any] = []

        class Player:
            """A scripted agent used to inspect driver behavior."""

            def play_level(self, session: ProtocolSession) -> None:
                """Drive the same commands through the selected interface."""
                frames.append(_frame(session))

                def changed() -> None:
                    frames.append(_frame(session))
                    episodes = session.level_episodes()
                    actions.append([[a.arr.tolist() for a in ep["actions"]]
                                    for ep in episodes])

                session.on_data_changed(changed)
                action = Action(np.array([0.0], dtype=np.float32))
                if use_executor:
                    asyncio.run(
                        session.executor.execute(PrimitiveAction(action),
                                                 ExecutionProgress()))
                    asyncio.run(
                        session.executor.execute(RequestReset("parity"),
                                                 ExecutionProgress()))
                else:
                    session.step(action)
                    session.reset("parity")
                approach.solve(session.observe().level.task, timeout=10)
                plan = getattr(approach, "_last_plan")
                expected = (set(session.observe().level.task.goal)
                            if ending == "divergence" else set())
                # An extra trailing action must never execute after WIN.
                requests = tuple(
                    ExecuteSkill(o, expected, "parity")
                    for o in [*plan, plan[0]])
                if use_executor:
                    progress = ExecutionProgress()
                    asyncio.run(
                        session.executor.execute(ExecutePlan(requests),
                                                 progress))
                else:
                    # Reference behavior of the original tool plan loop.
                    for request in requests:
                        result = session.invoke(request.option,
                                                request.expected, request.note)
                        if result.status != "succeeded" or result.diverged:
                            break
                        if result.outcome.episode_state is not \
                                EpisodeState.NOT_FINISHED:
                            break
                if ending != "win":
                    session.end_run("parity")

        card = ContinualRun(env, approach, Player()).run()
        lv = card.levels[0]
        assert len(actions) == len(frames) - 1
        assert actions and any(actions)
        return (frames, actions, card.end_reason, card.total_steps,
                card.total_resets, lv.won, lv.divergences,
                lv.skill_invocations, lv.failed_skill_invocations)

    assert capture(False) == capture(True)


def test_policy_cancellation_preserves_steps_and_releases_driver(
        tmp_path: Path) -> None:
    """Cancellation kills the controller's context, keeps charged actions and
    data notifications, and allows a later request to run."""
    env, approach = _setup(tmp_path)

    class Player:
        """A scripted agent used to inspect driver behavior."""

        def play_level(self, session: ProtocolSession) -> None:
            """Cancel after the first action while the next is being
            computed."""

            async def exercise() -> None:
                waiting = asyncio.Event()
                cleanup: List[bool] = []
                snapshots: List[Any] = []
                session.on_data_changed(
                    lambda: snapshots.append(_frame(session)))

                @asynccontextmanager
                async def controller() -> AsyncIterator[ActionSelector]:

                    async def get_action(obs: Any) -> Any:
                        if obs.ledger.run_steps == 0:
                            return Action(np.array([0.0], dtype=np.float32))
                        waiting.set()
                        await asyncio.Event().wait()
                        return None

                    try:
                        yield get_action
                    finally:
                        cleanup.append(True)

                progress = ExecutionProgress()
                task = asyncio.create_task(
                    session.executor.execute(ExecutePolicy(controller, 10),
                                             progress))
                await asyncio.wait_for(waiting.wait(), 5)
                with pytest.raises(InteractionBusy):
                    await session.executor.execute(RequestReset(),
                                                   ExecutionProgress())
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
                assert cleanup == [True]
                assert progress.steps == progress.attempts == 1
                assert len(snapshots) == 1
                assert not session.executor.busy
                reset = ExecutionProgress()
                await session.executor.execute(RequestReset(), reset)
                assert reset.steps == reset.resets == 1
                assert len(snapshots) == 2

            asyncio.run(exercise())
            session.executor.finish(GiveUp("done"))

    card = ContinualRun(env, approach, Player()).run()
    assert card.total_steps == 2 and card.total_resets == 1
    assert card.end_reason == "agent_ended" and card.end_note == "done"


def test_policy_yields_without_environment_steps(tmp_path: Path) -> None:
    """Internal computation and a None action consume no real steps."""
    env, approach = _setup(tmp_path)

    class Player:
        """A scripted agent used to inspect driver behavior."""

        def play_level(self, session: ProtocolSession) -> None:
            """Observe repeatedly and return control without acting."""

            @asynccontextmanager
            async def controller() -> AsyncIterator[ActionSelector]:

                async def get_action(obs: Any) -> Any:
                    assert obs.frame is session.observe().frame
                    await asyncio.sleep(0)
                    assert obs.frame is session.observe().frame
                    return None

                yield get_action

            progress = ExecutionProgress()
            asyncio.run(
                session.executor.execute(ExecutePolicy(controller, 10),
                                         progress))
            assert progress.steps == progress.attempts == 0
            assert progress.stop_reason == "policy_stopped"
            session.executor.finish(GiveUp("done"))

    card = ContinualRun(env, approach, Player()).run()
    assert card.total_steps == card.total_resets == 0


@pytest.mark.parametrize("phase", ["before", "after"])
def test_reporting_error_stops_before_later_actions(tmp_path: Path,
                                                    phase: str) -> None:
    """A predicate or report error preserves the old tool's stopping point."""
    env, approach = _setup(tmp_path)

    class Player:
        """A scripted agent whose observation reporter fails."""

        def play_level(self, session: ProtocolSession) -> None:
            """Fail before or after the first skill of a solvable plan."""
            approach.solve(session.observe().level.task, timeout=10)
            plan = getattr(approach, "_last_plan")
            attempts: List[int] = []

            def fail(_: Any) -> None:
                raise ValueError("report failed")

            observer = ExecutionObserver(
                on_attempt=lambda: attempts.append(session.level_card().steps),
                before_skill=fail if phase == "before" else None,
                after_skill=fail if phase == "after" else None)
            progress = ExecutionProgress()
            request = ExecutePlan(tuple(ExecuteSkill(o) for o in plan))
            with pytest.raises(ValueError, match="report failed"):
                asyncio.run(
                    session.executor.execute(request, progress, observer))
            assert not session.executor.busy
            count = 0 if phase == "before" else 1
            assert progress.steps == progress.attempts == count
            assert attempts == ([] if phase == "before" else [0])
            session.executor.finish(GiveUp("done"))

    card = ContinualRun(env, approach, Player()).run()
    assert not card.levels[0].won
    assert card.total_steps == (0 if phase == "before" else 1)
