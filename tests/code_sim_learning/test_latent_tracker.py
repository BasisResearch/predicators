"""Tests for execution-time latent tracking (code_sim_learning.latent_tracker)
and its CogMan wiring."""

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pytest

from predicators import utils
from predicators.approaches import BaseApproach
from predicators.code_sim_learning.latent_tracker import LatentTracker, \
    make_latent_tracker
from predicators.cogman import CogMan
from predicators.envs.cover import CoverEnv
from predicators.execution_monitoring import create_execution_monitor
from predicators.ground_truth_models import get_gt_options
from predicators.perception import create_perceiver
from predicators.structs import Action, Object, Predicate, State, Task, Type

_block = Type("block", ["contact"])


def _cure_rule(obs: State, latent: Dict[str, Any], history: Sequence[Any],
               updates: Dict[str, Any], params: Dict[str,
                                                     float]) -> Dict[str, Any]:
    """Recurrent rule: count steps while the observed block is in contact; weld
    once the count reaches ``cure_steps``."""
    del history
    blocks = obs.get_objects(_block)
    if blocks and obs.get(blocks[0], "contact") > 0.5:
        latent["count"] = latent.get("count", 0) + 1
    else:
        latent["count"] = 0
    if latent["count"] >= params["cure_steps"]:
        latent["welded"] = True
    return updates


def _legacy_rule(obs: State, updates: Dict[str, Any],
                 params: Dict[str, float]) -> Dict[str, Any]:
    del obs, params
    return updates


def _welded_pred() -> Predicate:

    def _classifier(s: State, objs: Sequence[Object]) -> bool:
        del objs
        return bool(s.latent is not None and s.latent.get("welded"))

    return Predicate("Welded", [_block], _classifier)


def _state(contact: float) -> State:
    return State({Object("b0", _block): np.array([contact])})


def test_make_latent_tracker_only_for_recurrent_rules() -> None:
    """Legacy 3-arg simulators get no tracker; recurrent ones do."""
    assert make_latent_tracker(None, {}, None) is None
    assert make_latent_tracker([], {}, None) is None
    assert make_latent_tracker([_legacy_rule], {}, None) is None
    tracker = make_latent_tracker([_legacy_rule, _cure_rule],
                                  {"cure_steps": 3.0}, {"count": 0})
    assert isinstance(tracker, LatentTracker)
    assert tracker.num_rules == 2


def test_tracker_follows_the_belief_recurrence() -> None:
    """The first observation carries the initial latent untouched; each later
    one advances it by the rules on the observed post-action state, and the raw
    observation is never modified."""
    params = {"cure_steps": 2.0}
    tracker = LatentTracker([_cure_rule], params, {"count": 0})
    welded = _welded_pred()
    act = Action(np.zeros(1, dtype=np.float32))
    raw0 = _state(1.0)
    s0 = tracker.attach(raw0, None)
    assert s0.latent == {"count": 0}
    assert raw0.latent is None  # the raw observation stays a bare one
    assert s0 is not raw0 and s0.data is raw0.data
    s1 = tracker.attach(_state(1.0), act)
    assert s1.latent == {"count": 1}
    s2 = tracker.attach(_state(1.0), act)
    assert s2.latent == {"count": 2, "welded": True}
    b = s2.get_objects(_block)[0]
    assert welded.holds(s2, [b])
    assert not welded.holds(s1, [b])
    # Snapshots are independent of the running latent and of each other.
    s2.latent["count"] = 99
    s3 = tracker.attach(_state(0.0), act)
    assert s3.latent == {"count": 0, "welded": True}
    assert s1.latent == {"count": 1}
    # Live parameter reference: an in-place fit is picked up.
    params["cure_steps"] = 10.0
    tracker.reset()
    tracker.attach(_state(1.0), None)
    s = tracker.attach(_state(1.0), act)
    assert s.latent == {"count": 1}


def test_tracker_stops_after_a_rule_raises() -> None:
    """After a rule raises the raw state passes through untracked until the
    next reset."""

    def _bad_rule(obs: State, latent: Dict[str, Any], history: Sequence[Any],
                  updates: Dict[str, Any],
                  params: Dict[str, float]) -> Dict[str, Any]:
        del obs, history, params
        if latent.get("armed"):
            raise RuntimeError("boom")
        latent["armed"] = True
        return updates

    tracker = LatentTracker([_bad_rule], {}, {})
    act = Action(np.zeros(1, dtype=np.float32))
    assert tracker.attach(_state(0.0), None).latent == {}
    assert tracker.attach(_state(0.0), act).latent == {"armed": True}
    raw = _state(0.0)
    assert tracker.attach(raw, act) is raw
    assert tracker.failed
    assert tracker.attach(_state(0.0), act).latent is None
    tracker.reset()
    assert not tracker.failed


class _TrackingApproach(BaseApproach):
    """Approach whose policy records the latent it was handed."""

    seen_latents: List[Optional[Dict[str, Any]]] = []

    @classmethod
    def get_name(cls) -> str:
        return "tracking_dummy"

    @property
    def is_learning_based(self) -> bool:
        return False

    def make_latent_tracker(self) -> Optional[LatentTracker]:
        return LatentTracker([_cure_rule], {"cure_steps": 1.0}, {"count": 0})

    def _solve(self, task: Task, timeout: int) -> Any:
        del task, timeout

        def _policy(state: State) -> Action:
            _TrackingApproach.seen_latents.append(state.latent)
            return Action(np.zeros(1, dtype=np.float32))

        return _policy


@pytest.mark.parametrize("exec_monitor_name", ["trivial"])
def test_cogman_hands_tracked_states_to_the_policy(
        exec_monitor_name: str) -> None:
    """CogMan attaches the tracked latent to the state its policy, monitor and
    termination function see, while the stored history stays raw."""
    utils.reset_config({
        "env": "cover",
        "approach": "random_actions",
        "execution_monitor": exec_monitor_name
    })
    env = CoverEnv()
    env_train_tasks = env.get_train_tasks()
    train_tasks = [t.task for t in env_train_tasks]
    options = get_gt_options(env.get_name())
    approach = _TrackingApproach(env.predicates, options, env.types,
                                 env.action_space, train_tasks)
    _TrackingApproach.seen_latents = []
    perceiver = create_perceiver("trivial")
    exec_monitor = create_execution_monitor(exec_monitor_name)
    cogman = CogMan(approach, perceiver, exec_monitor)
    env_task = env_train_tasks[0]
    seen_by_termination: List[Optional[Dict[str, Any]]] = []

    def _never_terminate(s: State) -> bool:
        seen_by_termination.append(s.latent)
        return False

    cogman.set_termination_function(_never_terminate)
    cogman.reset(env_task)
    obs = env.reset("train", 0)
    for _ in range(3):
        act = cogman.step(obs)
        assert act is not None
        obs = env.step(act)
    # First observation: initial latent; later ones advanced once per
    # action (cover states have no "contact" block, so the count resets).
    assert _TrackingApproach.seen_latents[0] == {"count": 0}
    assert all(lat is not None for lat in _TrackingApproach.seen_latents)
    assert seen_by_termination == _TrackingApproach.seen_latents
    cogman.finish_episode(obs)
    assert all(s.latent is None for s in cogman.get_current_history().states)
