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
from predicators.run import paths
from predicators.run.continual import ContinualRun, LevelAlreadyWon, \
    LevelLost, ProtocolSession, ResetUnavailable, RunEnded, build_levels, \
    level_summary
from predicators.run.controllers import OracleController, \
    RandomPrimitiveController, RandomSkillsController, create_controller
from predicators.run.episode import EpisodeOver, EpisodeState
from predicators.run.recording import LevelRecording, sanitize_state, \
    states_close
from predicators.run.scorecard import RunCard
from predicators.settings import CFG
from predicators.structs import Action, Object, State, Type


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
        "continual_runs_dir":
        os.path.join(str(tmp_path), "runs"),
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
        rec = LevelRecording(os.path.join(run.run_dir, f"L{lv.index + 1:02d}"))
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
    rec = LevelRecording(os.path.join(run.run_dir, "L01"))
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
    """Random skills run into horizon game overs (the run puts a horizon on
    episodes), reset, and end at the pooled cap with consistent counts."""
    _config(tmp_path,
            "random_options",
            continual_steps_per_level=60,
            continual_episode_horizon=40)
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
    assert second.run_dir == first.run_dir
    rec = LevelRecording(os.path.join(second.run_dir, "L01"))
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
    rec = LevelRecording(os.path.join(first.run_dir, "L01"))
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
    rec = LevelRecording(os.path.join(second.run_dir, "L01"))
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
    _config(tmp_path, "oracle", continual_episode_horizon=3)
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
            # The episode horizon rides in the ledger, apart from the
            # pooled cap.
            assert obs.ledger.horizon == 3
            assert obs.ledger.episode_steps == 0
            assert obs.ledger.episode_steps_remaining == 3
            assert "episode 0/3 steps to its horizon" in obs.ledger.footer()
            assert "it ends the level" not in obs.ledger.footer()
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
    _config(tmp_path, "oracle", continual_episode_horizon=3)
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
            # Without resets the horizon is the level's life; the ledger
            # says so and counts the episode's steps toward it.
            assert "steps to its horizon; it ends the level" in \
                obs.ledger.footer()
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
    events = _read_jsonl(os.path.join(run.run_dir, "L02", "index.jsonl"))
    game_over = [e for e in events if e["event"] == "game_over"]
    assert len(game_over) == 1 and game_over[0]["level_over"] is True
    _check_card_invariants(run)

    # Under the flag a test level resets like a train level.
    _config(tmp_path,
            "oracle",
            continual_episode_horizon=3,
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
    _config(tmp_path, "oracle", continual_episode_horizon=3)
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
    renders = os.listdir(os.path.join(run.run_dir, "L01", "renders"))
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


def test_one_directory_per_run(tmp_path: Any) -> None:
    """A run is one directory under the runs root, named by approach,
    experiment id, seed and launch stamp; a second launch of the same config is
    a second directory; --auto_resume adopts the newest unfinished run and
    starts a new one when the newest is over; a directory is never written
    over."""
    _config(tmp_path, "oracle", num_test_tasks=1)
    env, approach = _make("oracle")
    run = ContinualRun(env, approach, create_controller(env, approach))
    root = os.path.join(str(tmp_path), "runs")
    parent = os.path.join(root, "oracle", "test", "seed123")
    assert os.path.dirname(run.run_dir) == parent
    assert paths.RUN_DIR_RE.match(os.path.basename(run.run_dir))
    assert run.card_path == os.path.join(run.run_dir, "scorecard.json")
    assert run.run_dir == os.path.normpath(os.path.join(root, CFG.run_subdir))
    card = run.run()
    assert card.is_finished
    assert os.path.isfile(run.card_path)
    assert os.path.isdir(os.path.join(run.run_dir, "L01"))
    # A fresh subdir under the same root never names an existing dir.
    stamp = os.path.basename(run.run_dir)
    assert utils.new_run_subdir(root) != f"oracle/test/seed123/{stamp}/"
    # The newest run is finished: --auto_resume starts a new directory.
    _config(tmp_path, "oracle", num_test_tasks=1, auto_resume=True)
    assert paths.resumable_run_subdir() is None
    env2, approach2 = _make("oracle")
    second = ContinualRun(env2, approach2, create_controller(env2, approach2))
    assert second.run_dir != run.run_dir
    assert os.path.dirname(second.run_dir) == parent
    assert not second.card.levels[0].attempted
    # An unfinished newest run is adopted, logs and all.
    unfinished = os.path.join(parent, "run_20990101_000000")
    os.makedirs(unfinished)
    card.end_reason = None
    card.finished_at = None
    card.save(os.path.join(unfinished, "scorecard.json"))
    _config(tmp_path, "oracle", num_test_tasks=1, auto_resume=True)
    assert paths.resumable_run_subdir() == \
        "oracle/test/seed123/run_20990101_000000/"
    assert paths.run_dir() == unfinished
    # Without --auto_resume nothing is adopted, and a run directory that
    # already holds a run is refused rather than written over.
    _config(tmp_path, "oracle", num_test_tasks=1)
    assert paths.resumable_run_subdir() is None
    CFG.run_subdir = f"oracle/test/seed123/{stamp}/"
    env3, approach3 = _make("oracle")
    with pytest.raises(RuntimeError, match="already holds a run"):
        ContinualRun(env3, approach3, create_controller(env3, approach3))
    CFG.run_subdir = ""


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


def test_session_data_hook_fires_on_charged_calls(tmp_path: Any) -> None:
    """The arm's data listener runs after every charged call that changed the
    recording (a step, a reset, each invocation of a plan), not after a refused
    one, and a failing listener never fails the call."""
    _config(tmp_path, "oracle", continual_episode_horizon=3)
    env, approach = _make("oracle")
    events: List[Any] = []

    def _record() -> None:
        events.append((run.card.total_steps, run.card.total_resets))

    class _Probe:

        def play_level(self, session: ProtocolSession) -> None:
            """Step, trip the listener, exhaust the horizon, reset, win."""
            if session.level_index > 0:
                OracleController(approach).play_level(session)
                return
            zero = Action(np.zeros(env.action_space.shape, dtype=np.float32))
            session.on_data_changed(_record)
            session.step(zero)
            assert events == [(1, 0)]

            # A listener that raises is logged, and the call still returns.
            def _broken() -> None:
                raise ZeroDivisionError("listener bug")

            session.on_data_changed(_broken)
            session.step(zero)
            assert events == [(1, 0)]
            session.on_data_changed(_record)
            outcome = session.step(zero)
            assert outcome.state is EpisodeState.GAME_OVER
            assert events == [(1, 0), (3, 0)]
            # A refused call changes nothing and tells the arm nothing.
            with pytest.raises(EpisodeOver):
                session.step(zero)
            assert len(events) == 2
            session.reset("again")
            assert events[-1] == (4, 1)
            # A charged call reports once, whatever its length: the
            # oracle's plan runs as one policy call over several steps.
            OracleController(approach).play_level(session)
            assert session.level_card().won
            assert len(events) == 4
            assert events[-1] == (session.level_card().steps, 1)
            session.on_data_changed(None)
            events.append("cleared")

    run = ContinualRun(env, approach, _Probe())
    card = run.run()
    assert card.levels[0].won and events[-1] == "cleared"
    assert card.levels[0].steps == events[-2][0]


def test_sanitized_states_keep_the_robot_joint_data() -> None:
    """A recorded PyBullet state keeps what the env needs to re-simulate it
    (joint positions, base pose, command welds) and loses the process's
    handles; a raw joint sequence stays one; a plain or opaque state is plain.

    Everything kept pickles without the live env.
    """
    t = Type("t", ["x"])
    obj = Object("o", t)
    live = utils.PyBulletState({obj: np.array([1.0])},
                               simulator_state={
                                   "joint_positions": [0.1, 0.2],
                                   "physics_client_id":
                                   7,
                                   "robot_id":
                                   3,
                                   "base_pose":
                                   ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
                                   "command_welds": [("a", "b")],
                               })
    saved = sanitize_state(live)
    assert isinstance(saved, utils.PyBulletState)
    assert saved.joint_positions == [0.1, 0.2]
    assert isinstance(saved.simulator_state, dict)
    assert set(saved.simulator_state) == {
        "joint_positions", "base_pose", "command_welds"
    }
    assert saved.data[obj] is not live.data[obj]
    assert states_close(saved, live)
    reloaded = pickle.loads(pickle.dumps(saved))
    assert reloaded.joint_positions == [0.1, 0.2]
    raw = sanitize_state(
        utils.PyBulletState({obj: np.array([1.0])},
                            simulator_state=np.array([0.5, 0.6])))
    assert isinstance(raw, utils.PyBulletState)
    assert raw.joint_positions == [0.5, 0.6]
    for state in (State({obj: np.array([1.0])}),
                  State({obj: np.array([1.0])}, simulator_state=object())):
        plain = sanitize_state(state)
        assert type(plain) is State  # pylint: disable=unidiomatic-typecheck
        assert plain.simulator_state is None


def test_no_episode_horizon_by_default(tmp_path: Any) -> None:
    """Without ``continual_episode_horizon`` an episode outlives the env's own
    horizon: no GAME_OVER, nothing about a horizon in the ledger, and the
    pooled cap is what ends the run."""
    _config(tmp_path, "oracle", horizon=3, continual_steps_per_level=4)
    assert CFG.continual_episode_horizon is None
    env, approach = _make("oracle")
    seen: Dict[str, Any] = {}

    class _Idler:

        def play_level(self, session: ProtocolSession) -> None:
            """Idle past the env horizon on the train level."""
            zero = Action(np.zeros(env.action_space.shape, dtype=np.float32))
            obs = session.observe()
            assert obs.ledger.horizon is None
            assert obs.ledger.episode_steps_remaining is None
            assert "horizon" not in obs.ledger.footer()
            for _ in range(5):
                outcome = session.step(zero)
                assert outcome.state is EpisodeState.NOT_FINISHED
            seen["steps"] = session.level_card().steps
            # The cap ends the run from inside the call (RunEnded
            # propagates through the controller); nothing below it runs.
            for _ in range(20):
                session.step(zero)
            seen["past_cap"] = True

    card = ContinualRun(env, approach, _Idler()).run()
    assert seen["steps"] == 5 and "past_cap" not in seen
    assert card.end_reason == "step_cap"
    lv = card.levels[0]
    assert lv.game_overs == [] and lv.resets == 0
    assert lv.steps == card.step_cap == 12
