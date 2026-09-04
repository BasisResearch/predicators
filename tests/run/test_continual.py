"""Tests for the continual protocol core: run.continual, run.episode,
run.recording, run.scorecard and run.controllers.

Everything runs on the cover env, where a skill is one primitive step,
so the counts are easy to pin exactly.
"""
import json
import os
import pickle
from typing import Any, Dict, List

import numpy as np
import pytest

from predicators import utils
from predicators.approaches import create_approach
from predicators.envs import create_new_env
from predicators.ground_truth_models import get_gt_options
from predicators.run.continual import ContinualRun, LevelAlreadyWon, \
    LevelLost, ProtocolSession, ResetUnavailable, RunEnded, build_levels, \
    level_summary
from predicators.run.controllers import OracleController, \
    RandomPrimitiveController, RandomSkillsController, create_controller
from predicators.run.episode import EpisodeOver, EpisodeState
from predicators.run.recording import LevelRecording, states_close
from predicators.run.scorecard import RunCard
from predicators.structs import Action


class _Preempted(BaseException):
    """Stands in for the kill a Slurm preemption delivers."""


def _config(tmp_path: Any, approach: str, **overrides: Any) -> None:
    utils.reset_config({
        "env":
        "cover",
        "approach":
        approach,
        "seed":
        123,
        "num_train_tasks":
        1,
        "num_test_tasks":
        2,
        "horizon":
        40,
        "experiment_protocol":
        "continual",
        "continual_steps_per_level":
        200,
        "continual_render":
        False,
        "continual_scorecards_dir":
        os.path.join(str(tmp_path), "cards"),
        "continual_recordings_dir":
        os.path.join(str(tmp_path), "recs"),
        "experiment_id":
        "test",
        **overrides,
    })


def _make(approach_name: str) -> Any:
    env = create_new_env("cover", do_cache=False)
    options = get_gt_options(env.get_name())
    approach = create_approach(approach_name, env.predicates, options,
                               env.types, env.action_space,
                               [t.task for t in env.get_train_tasks()])
    return env, approach


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _check_card_invariants(run: ContinualRun) -> None:
    """Counts on disk and in memory agree, level by level."""
    card = run.card
    on_disk = RunCard.load(run.card_path)
    assert on_disk.to_dict()["levels"] == card.to_dict()["levels"]
    for lv in card.levels:
        assert lv.steps == sum(ep.steps for ep in lv.episodes) + lv.resets
        if not lv.attempted:
            continue
        rec = LevelRecording(
            os.path.join(run.recordings_dir, f"L{lv.index + 1:02d}"))
        lines = _read_jsonl(rec.actions_path)
        applied = [r for r in lines if "a" in r]
        resets = [r for r in lines if r.get("event") == "reset"]
        # Every applied action is in the log; every episode has a boundary.
        assert len(applied) == sum(ep.steps for ep in lv.episodes)
        assert len(resets) == len(lv.episodes)
        with open(rec.episodes_path, "rb") as f:
            episodes = pickle.load(f)
        assert [ep["episode"] for ep in episodes] == \
            [ep.index for ep in lv.episodes]
        for rec_ep, card_ep in zip(episodes, lv.episodes):
            assert len(rec_ep["actions"]) == card_ep.steps
            assert len(rec_ep["states"]) == card_ep.steps + 1
            assert rec_ep["end"] == card_ep.end
        rec.close()


def test_oracle_wins_every_level(tmp_path: Any) -> None:
    """The oracle controller wins each level in one episode; the card, the
    recordings and the index all describe it."""
    _config(tmp_path, "oracle")
    env, approach = _make("oracle")
    run = ContinualRun(env, approach, create_controller(env, approach))
    card = run.run()
    assert card.end_reason == "all_levels_won"
    assert card.levels_completed == card.levels_total == 3
    assert card.total_resets == 0
    assert card.total_steps == card.total_skill_invocations > 0
    for lv in card.levels:
        assert lv.won and lv.won_at_step == lv.steps
        assert lv.steps_before_first_win == lv.steps
        assert lv.resets_before_first_win == 0
        assert [ep.end for ep in lv.episodes] == ["win"]
        assert lv.episodes[0].reward == 1.0
        assert lv.episodes[0].terminated and not lv.episodes[0].rejected
        assert lv.wall_clock > 0.0 and lv.wall_clock_env > 0.0
        assert lv.finished_at is not None
    _check_card_invariants(run)
    rec = LevelRecording(os.path.join(run.recordings_dir, "L01"))
    events = [e["event"] for e in rec.read_index()]
    assert events[0] == "level_start"
    assert events[-1] == "win"
    assert events.count("invoke") == card.levels[0].skill_invocations
    invoke = next(e for e in rec.read_index() if e["event"] == "invoke")
    assert invoke["status"] == "succeeded"
    assert invoke["skill"].startswith("PickPlace")
    rec.close()
    # The level list follows the protocol order: train task, then tests.
    specs = build_levels(env, "oracle")
    assert [(s.split, s.task_idx) for s in specs] == [("train", 0),
                                                      ("test", 0), ("test", 1)]


def test_random_skills_hits_the_step_cap(tmp_path: Any) -> None:
    """Random skills run into horizon game overs, reset, and end at the pooled
    cap with consistent counts."""
    _config(tmp_path, "random_options", continual_steps_per_level=60)
    env, approach = _make("random_options")
    controller = create_controller(env, approach)
    assert isinstance(controller, RandomSkillsController)
    run = ContinualRun(env, approach, controller)
    card = run.run()
    assert card.end_reason == "step_cap"
    assert card.total_steps == card.step_cap == 180
    assert card.steps_remaining == 0
    attempted = [lv for lv in card.levels if lv.attempted]
    assert attempted, "the cap was hit before any level was attempted"
    lv0 = card.levels[0]
    if lv0.game_overs:
        assert lv0.game_overs[0] == "horizon"
        assert lv0.episodes[0].end == "game_over:horizon"
        assert lv0.episodes[0].steps == 40
        assert lv0.resets >= 1
    _check_card_invariants(run)


def test_random_primitives_count_no_invocations(tmp_path: Any) -> None:
    """The primitive-only arm charges steps and resets but never a skill."""
    _config(tmp_path, "random_actions", continual_steps_per_level=30)
    env, approach = _make("random_actions")
    run = ContinualRun(env, approach, create_controller(env, approach))
    card = run.run()
    assert card.end_reason == "step_cap"
    assert card.total_skill_invocations == 0
    assert card.total_steps == 90
    _check_card_invariants(run)


class _StopAfter:
    """Wrap a controller so the process 'dies' after N invocations."""

    def __init__(self, inner: Any, session_kills: int) -> None:
        self._inner = inner
        self._left = session_kills
        self.invocations = 0

    def play_level(self, session: ProtocolSession) -> None:
        """Play with the inner controller until the kill fires."""
        original = session.invoke

        def _invoke(*args: Any, **kwargs: Any) -> Any:
            result = original(*args, **kwargs)
            self.invocations += 1
            if self.invocations >= self._left:
                raise _Preempted()
            return result

        session.invoke = _invoke  # type: ignore[method-assign]
        try:
            self._inner.play_level(session)
        finally:
            session.invoke = original  # type: ignore[method-assign]


def test_preemption_resume_replays_losslessly(tmp_path: Any) -> None:
    """A kill mid-level resumes with the env rebuilt from the action log, the
    counts untouched, and one resume recorded."""
    _config(tmp_path, "oracle", num_test_tasks=1, horizon=200)
    env, approach = _make("oracle")
    # The random controller keeps a level busy for many invocations, so
    # the kill lands mid-episode.
    inner = RandomSkillsController(get_gt_options("cover"), seed=7)
    killer = _StopAfter(inner, session_kills=5)
    first = ContinualRun(env, approach, killer)
    with pytest.raises(_Preempted):
        first.run()
    card = RunCard.load(first.card_path)
    assert card.end_reason is None, "a crash must leave the card resumable"
    steps_before = card.total_steps
    assert steps_before == 5
    assert card.levels[0].resumes == 0

    _config(tmp_path,
            "oracle",
            num_test_tasks=1,
            horizon=200,
            auto_resume=True)
    env2, approach2 = _make("oracle")
    second = ContinualRun(env2, approach2, OracleController(approach2))
    lv = second.card.levels[0]
    assert lv.attempted and not lv.won and lv.steps == steps_before
    card2 = second.run()
    assert card2.end_reason == "all_levels_won"
    lv = card2.levels[0]
    assert lv.resumes == 1 and lv.preemptions == 1
    assert lv.harness_resets == 0
    assert lv.downtime >= 0.0
    assert lv.steps > steps_before
    # The episode the kill interrupted is the one that went on to win:
    # no reset was charged for the recovery.
    assert lv.resets == 0
    assert [ep.end for ep in lv.episodes] == ["win"]
    rec = LevelRecording(os.path.join(second.recordings_dir, "L01"))
    resume = [e for e in rec.read_index() if e["event"] == "resume"]
    assert len(resume) == 1 and resume[0]["verified"] is True
    assert resume[0]["replayed_steps"] == steps_before
    rec.close()
    _check_card_invariants(second)


def test_resume_with_diverged_replay_is_a_harness_reset(tmp_path: Any) -> None:
    """When the replay does not reproduce the checkpoint, the level is
    restarted and the restart is booked to the harness, not the agent."""
    _config(tmp_path, "oracle", num_test_tasks=1, horizon=200)
    env, approach = _make("oracle")
    inner = RandomSkillsController(get_gt_options("cover"), seed=7)
    first = ContinualRun(env, approach, _StopAfter(inner, session_kills=3))
    with pytest.raises(_Preempted):
        first.run()
    # Corrupt the checkpoint state so the replay cannot match it.
    rec = LevelRecording(os.path.join(first.recordings_dir, "L01"))
    ckpt = rec.load_checkpoint()
    assert ckpt is not None
    state = ckpt["state"]
    obj = sorted(state.data)[0]
    state.data[obj] = state.data[obj] + 5.0
    with open(rec.checkpoint_path, "wb") as f:
        pickle.dump(ckpt, f)
    rec.close()

    _config(tmp_path,
            "oracle",
            num_test_tasks=1,
            horizon=200,
            auto_resume=True)
    env2, approach2 = _make("oracle")
    second = ContinualRun(env2, approach2, OracleController(approach2))
    card = second.run()
    lv = card.levels[0]
    assert lv.won
    assert lv.harness_resets == 1
    assert lv.resets == 0, "the agent is not charged for a harness reset"
    assert [ep.end for ep in lv.episodes] == ["harness_reset", "win"]
    rec = LevelRecording(os.path.join(second.recordings_dir, "L01"))
    resume = [e for e in rec.read_index() if e["event"] == "resume"]
    assert resume[0]["verified"] is False
    resets = [e for e in rec.read_index() if e["event"] == "reset"]
    assert resets[0]["by"] == "harness_reset"
    rec.close()
    _check_card_invariants(second)


def test_agent_ended_and_level_not_won(tmp_path: Any) -> None:
    """A controller may end the run; returning without a win ends it too."""
    _config(tmp_path, "oracle")
    env, approach = _make("oracle")

    class _Quitter:

        def play_level(self, session: ProtocolSession) -> None:
            """End the run at once."""
            session.end_run("done for today")

    card = ContinualRun(env, approach, _Quitter()).run()
    assert card.end_reason == "agent_ended"
    assert card.end_note == "done for today"
    assert card.levels_completed == 0

    _config(tmp_path, "oracle", experiment_id="test2")
    env, approach = _make("oracle")

    class _GiveUp:

        def play_level(self, session: ProtocolSession) -> None:
            """Look and give up."""
            session.observe()

    card = ContinualRun(env, approach, _GiveUp()).run()
    assert card.end_reason == "level_not_won"
    assert card.levels[0].attempted and not card.levels[0].won
    assert not card.levels[1].attempted


def test_session_protocol_errors(tmp_path: Any) -> None:
    """Steps after a game over and charged calls after a win are errors."""
    _config(tmp_path, "oracle", horizon=3)
    env, approach = _make("oracle")
    seen: Dict[str, Any] = {}

    class _Probe:

        def play_level(self, session: ProtocolSession) -> None:
            """Exhaust the horizon, reset, win, then poke the won level."""
            if session.level_index > 0:
                # Test levels have no resets (their own test below).
                OracleController(approach).play_level(session)
                return
            obs = session.observe()
            assert obs.state is EpisodeState.NOT_FINISHED
            assert obs.ledger.level_steps == 0
            assert "[ledger]" in obs.ledger.footer()
            zero = Action(np.zeros(env.action_space.shape, dtype=np.float32))
            # Three no-op steps exhaust the horizon.
            for _ in range(3):
                outcome = session.step(zero)
            assert outcome.state is EpisodeState.GAME_OVER
            assert outcome.reason == "horizon"
            with pytest.raises(EpisodeOver):
                session.step(zero)
            obs = session.observe()
            assert obs.state is EpisodeState.GAME_OVER
            assert obs.evaluation is not None and \
                obs.evaluation.reward == 0.0
            session.reset("try again")
            # Now let the oracle win, then poke at a won level.
            OracleController(approach).play_level(session)
            with pytest.raises(LevelAlreadyWon):
                session.step(zero)
            with pytest.raises(LevelAlreadyWon):
                session.reset()
            seen["ok"] = True

    card = ContinualRun(env, approach, _Probe()).run()
    assert seen["ok"]
    lv = card.levels[0]
    assert lv.won and lv.resets == 1 and lv.game_overs == ["horizon"]
    assert lv.steps_before_first_win == lv.steps
    assert lv.resets_before_first_win == 1


def test_test_levels_have_no_resets_by_default(tmp_path: Any) -> None:
    """A test level is one shot: reset is refused without a charge, GAME_OVER
    loses the level, later charged calls are errors and the run ends with
    ``level_lost``; ``continual_allow_test_resets`` restores resets."""
    _config(tmp_path, "oracle", horizon=3)
    env, approach = _make("oracle")
    seen: Dict[str, Any] = {}
    zero = Action(np.zeros(env.action_space.shape, dtype=np.float32))

    class _Probe:

        def play_level(self, session: ProtocolSession) -> None:
            """Win the train level; exhaust the test level's horizon."""
            if session.level_index == 0:
                assert session.resets_allowed
                OracleController(approach).play_level(session)
                return
            assert not session.resets_allowed
            obs = session.observe()
            assert "(none on this level)" in obs.ledger.footer()
            session.step(zero)
            with pytest.raises(ResetUnavailable):
                session.reset("please")
            lv = session.level_card()
            assert lv.steps == 1 and lv.resets == 0 and not lv.lost
            for _ in range(2):
                outcome = session.step(zero)
            assert outcome.state is EpisodeState.GAME_OVER
            assert session.level_card().lost
            with pytest.raises(LevelLost):
                session.step(zero)
            with pytest.raises(LevelLost):
                session.reset()
            assert session.observe().state is EpisodeState.GAME_OVER
            seen["ok"] = True

    run = ContinualRun(env, approach, _Probe())
    card = run.run()
    assert seen["ok"]
    assert card.end_reason == "level_lost"
    assert card.levels[0].won and not card.levels[0].lost
    lv = card.levels[1]
    assert lv.lost and not lv.won and lv.finished_at is not None
    assert lv.steps == 3 and lv.resets == 0 and lv.game_overs == ["horizon"]
    assert not card.levels[2].attempted
    assert "L2 test[0] lost" in level_summary(card)
    assert RunCard.load(run.card_path).levels[1].lost
    events = _read_jsonl(os.path.join(run.recordings_dir, "L02",
                                      "index.jsonl"))
    game_over = [e for e in events if e["event"] == "game_over"]
    assert len(game_over) == 1 and game_over[0]["level_over"] is True
    _check_card_invariants(run)

    # Under the flag a test level resets like a train level.
    _config(tmp_path,
            "oracle",
            horizon=3,
            experiment_id="test2",
            continual_allow_test_resets=True)
    env, approach = _make("oracle")

    class _Resetter:

        def play_level(self, session: ProtocolSession) -> None:
            """Exhaust the horizon, reset, then let the oracle win."""
            if session.level_index > 0:
                assert session.resets_allowed
                assert "(none on" not in session.observe().ledger.footer()
                for _ in range(3):
                    session.step(zero)
                assert not session.level_card().lost
                session.reset("second try")
            OracleController(approach).play_level(session)

    card = ContinualRun(env, approach, _Resetter()).run()
    assert card.end_reason == "all_levels_won"
    assert [lv.resets for lv in card.levels] == [0, 1, 1]
    assert not any(lv.lost for lv in card.levels)


def test_controllers_stop_at_a_lost_test_level(tmp_path: Any) -> None:
    """The built-in controllers return instead of resetting when the level has
    no resets, and the run ends as ``level_lost``."""
    _config(tmp_path, "oracle", horizon=3)
    env, approach = _make("oracle")

    class _Mixed:

        def play_level(self, session: ProtocolSession) -> None:
            """The oracle wins the train level; random primitives lose the test
            level at its horizon."""
            if session.level_index == 0:
                OracleController(approach).play_level(session)
            else:
                RandomPrimitiveController(env, 0).play_level(session)

    card = ContinualRun(env, approach, _Mixed()).run()
    assert card.end_reason == "level_lost"
    lv = card.levels[1]
    assert lv.lost and lv.resets == 0 and lv.game_overs == ["horizon"]
    assert lv.steps == 3 and not card.levels[2].attempted


def test_divergence_and_expected_outcomes(tmp_path: Any) -> None:
    """An expected atom that does not hold afterwards is a divergence and stops
    execute_plan."""
    _config(tmp_path, "oracle")
    env, approach = _make("oracle")
    seen: Dict[str, Any] = {}

    class _Expect:

        def play_level(self, session: ProtocolSession) -> None:
            """Execute the oracle plan with a wrong expectation first."""
            obs = session.observe()
            task = obs.level.task
            approach.solve(task, timeout=10)
            plan = list(getattr(approach, "_last_plan"))
            assert len(plan) >= 2
            # Expect the goal after the FIRST skill: it cannot hold yet.
            results = session.execute_plan(plan, [set(task.goal)] + [set()] *
                                           (len(plan) - 1))
            assert len(results) == 1 and results[0].diverged
            assert results[0].missing == set(task.goal)
            # Without expectations the rest of the plan runs to the win.
            rest = session.execute_plan(plan[1:])
            assert all(r.status == "succeeded" for r in rest)
            seen["ok"] = True

    card = ContinualRun(env, approach, _Expect()).run()
    assert seen["ok"]
    assert card.levels[0].won
    assert card.levels[0].divergences == 1


def test_scorecard_round_trip_and_renders(tmp_path: Any) -> None:
    """The JSON card reloads to an equal object; renders land on disk."""
    _config(tmp_path, "oracle", num_test_tasks=0, continual_render=True)
    env, approach = _make("oracle")
    run = ContinualRun(env, approach, create_controller(env, approach))
    card = run.run()
    loaded = RunCard.load(run.card_path)
    assert loaded.to_dict() == card.to_dict()
    assert loaded.git_sha == card.git_sha
    renders = os.listdir(os.path.join(run.recordings_dir, "L01", "renders"))
    names = sorted(renders)
    assert any(n.endswith("_start.png") for n in names)
    assert any(n.endswith("_win.png") for n in names)
    assert len(names) == 2 + card.levels[0].skill_invocations


def test_states_close_and_wrong_level_count(tmp_path: Any) -> None:
    """states_close ignores simulator state and tolerates tiny drift; a card
    with a different level count is not resumed."""
    _config(tmp_path, "oracle")
    env, approach = _make("oracle")
    state = env.get_train_tasks()[0].task.init
    drift = state.copy()
    obj = sorted(drift.data)[0]
    drift.data[obj] = drift.data[obj] + 1e-6
    assert states_close(state, drift)
    far = state.copy()
    far.data[obj] = far.data[obj] + 1.0
    assert not states_close(state, far)

    run = ContinualRun(env, approach, create_controller(env, approach))
    run.run()
    _config(tmp_path, "oracle", auto_resume=True, num_test_tasks=1)
    env2, approach2 = _make("oracle")
    fresh = ContinualRun(env2, approach2, create_controller(env2, approach2))
    assert fresh.card.levels_total == 2
    assert not fresh.card.levels[0].attempted


def test_run_ended_carries_reason() -> None:
    """RunEnded exposes its reason and note."""
    err = RunEnded("step_cap", "note")
    assert err.reason == "step_cap" and err.note == "note"
    assert str(err) == "step_cap"


def test_create_controller_rejects_unknown_arm(tmp_path: Any) -> None:
    """An approach without play_level and without a scripted controller is an
    error, not a silent default."""
    _config(tmp_path, "random_actions")
    env, approach = _make("random_actions")
    approach.get_name = lambda: "mystery"  # type: ignore[method-assign]
    with pytest.raises(ValueError):
        create_controller(env, approach)
