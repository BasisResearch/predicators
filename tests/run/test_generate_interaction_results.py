"""Tests for the early-stopping accounting in generate_interaction_results.

Regression for the 2026-09-02 bridge seed3 ledger gap: two explore
episodes reached the goal for real (env reward 1.0, accepted) but were
scored 0/2 because their plans were not belief-certified
(``mental_model_solved=False``) - and the log recorded them as bare
failures, indistinguishable from plans that collapsed. The discount is
correct policy; it must also be legible: the ledger line has to say the
episode solved and name why it does not count.
"""
from types import SimpleNamespace
from typing import Any, Callable, List, Optional, cast
from unittest.mock import MagicMock

from predicators import utils
from predicators.run import online_learning
from predicators.structs import Action, InteractionRequest, State


def _reset_config() -> None:
    utils.reset_config({
        "env": "cover",
        "approach": "oracle",
        "seed": 0,
        "env_has_impossible_goals": False,
        "make_interaction_videos": False,
    })


def _make_request(mental_model_solved: Optional[bool]) -> InteractionRequest:
    # The stubbed episode runner never calls the act policy.
    act_policy = cast(Callable[[State], Action], lambda s: None)
    return InteractionRequest(train_task_idx=0,
                              act_policy=act_policy,
                              query_policy=lambda s: None,
                              termination_function=lambda s: True,
                              mental_model_solved=mental_model_solved)


def _run(monkeypatch: Any, *, real_solved: bool,
         mental_model_solved: Optional[bool]) -> List[bool]:
    """Drive one episode through generate_interaction_results with the episode
    runner stubbed to the given real-env verdict."""
    obs = MagicMock()

    def _fake_episode(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs  # unused
        return ([obs], []), real_solved, MagicMock()

    monkeypatch.setattr(online_learning, "run_episode_and_get_observations",
                        _fake_episode)
    cogman = MagicMock()
    cogman.get_current_history.return_value = SimpleNamespace(states=[obs],
                                                              actions=[])
    # No _get_current_predicates: skips the debug abstract-state render.
    cogman._approach = SimpleNamespace()  # pylint: disable=protected-access
    env_task = MagicMock()
    env_task.early_stop_min_reward = None
    env = MagicMock()
    env.get_train_tasks.return_value = [env_task]
    env.evaluate_episode.return_value = SimpleNamespace(reward=1.0,
                                                        terminated=True,
                                                        rejected=False,
                                                        reason="")
    _, _, task_solved_status = online_learning.generate_interaction_results(
        cogman, env, None, [_make_request(mental_model_solved)])
    return task_solved_status


def test_uncertified_success_discounted_and_named(monkeypatch, caplog):
    """A goal-reaching episode from an uncertified plan scores unsolved, and
    the ledger says the episode solved and why it does not count - in the exact
    format scripts/log_viewer.py renders."""
    _reset_config()
    with caplog.at_level("INFO"):
        status = _run(monkeypatch, real_solved=True, mental_model_solved=False)
    assert status == [False]
    lines = [
        r.getMessage() for r in caplog.records
        if "does NOT count as solved for early stopping" in r.getMessage()
    ]
    assert len(lines) == 1
    # pylint: disable-next=import-outside-toplevel
    from scripts.log_viewer import INTERACTION_BAR_RE
    assert INTERACTION_BAR_RE.match("INFO: " + lines[0])


def test_uncertified_real_failure_gets_no_solved_line(monkeypatch, caplog):
    """When the episode did not reach the goal anyway, certification was not
    the deciding factor and no discount line is logged."""
    _reset_config()
    with caplog.at_level("INFO"):
        status = _run(monkeypatch,
                      real_solved=False,
                      mental_model_solved=False)
    assert status == [False]
    assert not [
        r for r in caplog.records
        if "does NOT count as solved" in r.getMessage()
    ]


def test_certified_success_counts_without_discount_line(monkeypatch, caplog):
    """A certified (or verdict-less) success keeps its solved status."""
    _reset_config()
    for verdict in (True, None):
        caplog.clear()
        with caplog.at_level("INFO"):
            status = _run(monkeypatch,
                          real_solved=True,
                          mental_model_solved=verdict)
        assert status == [True]
        assert not [
            r for r in caplog.records
            if "does NOT count as solved" in r.getMessage()
        ]
