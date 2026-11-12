"""Tests for the ``evaluate_trajectory`` helper the synthesis namespace
offers."""
# pylint: disable=protected-access,import-outside-toplevel,unused-import
from __future__ import annotations

import functools

import numpy as np
import pytest

# Bootstrap circular imports before pulling from predicators.approaches.
from predicators import utils
from predicators.structs import Action, State, Task, Type


@pytest.fixture(name="approach_cls")
def _approach_cls():
    """Late-import the class so test collection is cheap."""
    from predicators.approaches.agent_sim_learning_approach import \
        AgentSimLearningApproach
    return AgentSimLearningApproach


# ── _format_trajectory_listing ──────────────────────────────────────

# ── _format_prior_state_block ────────────────────────────────────────

# ── _format_goal_nl_block (predicate-invention subclass) ────────────

# ── _build_synthesis_system_prompt (FO vs PO rule signature) ────────
# These render the whole synthesis system prompt. The method only touches
# ``self`` through pure no-state helpers (``_rule_signature_section``,
# ``_residual_rule_signature``, ``_extra_synthesis_system_prompt``), so a
# bare instance via ``object.__new__`` is enough to render it.

# ── _make_evaluate_trajectory_fn / _format_objective_block ──────────


def test_evaluate_trajectory_helper(approach_cls):
    """The exec-ns evaluate_trajectory helper returns verdict dicts (never the
    evaluator), labels Action inputs by their producing options, and rejects
    bad task indices."""
    from types import SimpleNamespace

    from predicators.structs import TaskEvaluator

    class _RecordingEvaluator(TaskEvaluator):
        """Rejecting evaluator that records the labels it saw."""

        def __init__(self):
            super().__init__(set())  # empty goal: terminated is True
            self.seen_options = None

        def _certify(self, states, step_options, sim_env=None):
            self.seen_options = step_options
            return False, "nope"

    evaluator = _RecordingEvaluator()

    cup_type = Type("cup_type", ["f"])
    cup = cup_type("cup")
    states = [State({cup: [0.0]}), State({cup: [1.0]})]
    # _option_model mirrors the real approach attribute the helper reads
    # for the certificate's sim_env (None => kinematics-only scoring).
    stub = SimpleNamespace(_train_tasks=[
        Task(states[0], set(), evaluator=evaluator),
        Task(states[0], set()),
    ],
                           _option_model=None,
                           _tool_context=SimpleNamespace(current_task=None))
    stub._level_task = functools.partial(approach_cls._level_task, stub)
    fn = approach_cls._make_evaluate_trajectory_fn(stub)
    push = utils.SingletonParameterizedOption(
        "Push", lambda s, m, o, p: Action(np.zeros(1, dtype=np.float32)))
    act = Action(np.zeros(1, dtype=np.float32))
    act.set_option(push.ground([], np.zeros(0, dtype=np.float32)))

    verdict = fn(states, [act], task_idx=0)
    # The public verdict includes a replay note, but excludes internal
    # legitimacy details and goal-atom termination. This evaluator does
    # not replay physics, so the note is empty.
    assert verdict == {
        "reward": 0.0,  # bonus gated by the internal rejection
        "solved": False,
        "note": "",
    }
    # Labels are (name, objects, params) triples since plan-capture
    # gating started matching on exact params.
    assert evaluator.seen_options == [("Push", (), ())]
    # Pre-built labels pass through unchanged.
    fn(states, [("Push", ("robot", ))], task_idx=0)
    assert evaluator.seen_options == [("Push", ("robot", ))]
    with pytest.raises(ValueError, match="no goal atoms to judge"):
        fn(states, None, task_idx=1)
    with pytest.raises(ValueError, match="out of range"):
        fn(states, None, task_idx=2)
    # Without an index the helper scores the level in progress.
    stub._tool_context.current_task = stub._train_tasks[1]
    with pytest.raises(ValueError, match="no goal atoms to judge"):
        fn(states, None)
    stub._tool_context.current_task = stub._train_tasks[0]
    assert fn(states, [act])["solved"] is False
    with pytest.raises(ValueError, match="non-empty"):
        fn([], None, task_idx=0)


def test_evaluate_trajectory_physics_sweep(approach_cls):
    """physics_sweep=True scores the sequence at every physics-margin point on
    a fresh env at that physics and reports the fraction scored solved; with no
    points to sweep it says so."""
    import contextlib
    from types import SimpleNamespace

    from predicators.structs import TaskEvaluator

    # The sweep needs uncertainty decisions, whatever config the test
    # before this one left.
    utils.reset_config({"continual_uncertainty_decisions": True})
    physics = {"friction": 0.5}

    class _FrictionEvaluator(TaskEvaluator):
        """Certifies only when the (swept) friction is at least 0.5."""

        def __init__(self):
            super().__init__(set())

        def _certify(self, states, step_options, sim_env=None):
            return physics["friction"] >= 0.5, "friction"

    cup_type = Type("cup_type", ["f"])
    cup = cup_type("cup")
    states = [State({cup: [0.0]}), State({cup: [1.0]})]
    seen = []

    @contextlib.contextmanager
    def _scope(physical_overrides=None):
        seen.append(dict(physical_overrides or {}))
        prev = dict(physics)
        physics.update(physical_overrides or {})
        try:
            yield
        finally:
            physics.clear()
            physics.update(prev)

    stub = SimpleNamespace(
        _train_tasks=[Task(states[0], set(), evaluator=_FrictionEvaluator())],
        _option_model=None,
        _identified_physical_sigma_points=[{
            "friction": 0.4
        }, {
            "friction": 0.5
        }, {
            "friction": 0.6
        }],
        _fresh_validation_env_scope=_scope,
        _tool_context=SimpleNamespace(current_task=None))
    stub._sweep_evaluation = functools.partial(approach_cls._sweep_evaluation,
                                               stub)
    stub._level_task = functools.partial(approach_cls._level_task, stub)
    fn = approach_cls._make_evaluate_trajectory_fn(stub)
    plain = fn(states, None, task_idx=0)
    assert "sweep" not in plain and plain["solved"] is True
    swept = fn(states, None, task_idx=0, physics_sweep=True)
    assert swept["solved"] is True
    sweep = swept["sweep"]
    assert [p["solved"] for p in sweep["points"]] == [False, True, True]
    assert sweep["solved_fraction"] == pytest.approx(2 / 3)
    assert sweep["certified"] is False
    assert seen == [{"friction": 0.4}, {"friction": 0.5}, {"friction": 0.6}]
    assert physics == {"friction": 0.5}  # the scope restored the physics
    stub._identified_physical_sigma_points = []
    assert fn(states, None, task_idx=0, physics_sweep=True)["sweep"] is None


# ── The helper of the arms with no supplied simulator ───────────────


def _cup_task():
    """A one-cup task whose goal holds once the cup's ``f`` reaches 1."""
    from predicators.structs import GroundAtom, Predicate, TaskEvaluator
    cup_type = Type("cup_type", ["f", "g"])
    cup = cup_type("cup")
    full = Predicate("Full", [cup_type], lambda s, o: s.get(o[0], "f") >= 1.0)
    goal = {GroundAtom(full, [cup])}
    start = State({cup: np.array([0.0, 5.0])})
    return cup, start, Task(start, goal, evaluator=TaskEvaluator(goal))


def test_engine_free_helper_returns_the_public_triple():
    """The baselines' helper scores a goal-satisfying sequence as solved under
    a rule that needs no replay, returns only the three public keys, takes no
    physics sweep, and never hands the evaluator a simulator."""
    from predicators.agent_sdk.tools.reward_model import level_task, \
        make_engine_free_evaluate_trajectory_fn, with_features
    from predicators.structs import TaskEvaluator

    cup, start, task = _cup_task()
    plain = Task(start, set())
    fn = make_engine_free_evaluate_trajectory_fn(
        functools.partial(level_task, [task, plain], None))
    done = with_features(start, {"cup": {"f": 1.0}})
    # The copy changed, the original did not, and unnamed features stay.
    assert start.get(cup, "f") == 0.0 and done.get(cup, "g") == 5.0
    assert with_features(start, {"cup:cup_type": {"f": 2.0}}).get(cup, "f") \
        == 2.0
    assert fn([start, done]) == {"reward": 1.0, "solved": True, "note": ""}
    assert fn([start, start], task_idx=0) == {
        "reward": 0.0,
        "solved": False,
        "note": ""
    }
    with pytest.raises(TypeError):
        fn([start, done], physics_sweep=True)  # pylint: disable=unexpected-keyword-arg
    with pytest.raises(ValueError, match="no goal atoms to judge"):
        fn([start, done], task_idx=1)
    with pytest.raises(ValueError, match="out of range"):
        fn([start, done], task_idx=2)
    with pytest.raises(ValueError, match="non-empty"):
        fn([])
    with pytest.raises(ValueError, match="Unknown object 'mug'"):
        with_features(start, {"mug": {"f": 1.0}})
    with pytest.raises(ValueError, match="Failed to set cup.h"):
        with_features(start, {"cup": {"h": 1.0}})
    with pytest.raises(ValueError, match="mapping"):
        with_features(start, {"cup": 1.0})

    class _Recording(TaskEvaluator):
        """Records the simulator and the labels each rule call received."""

        def __init__(self):
            super().__init__(set())
            self.seen = []

        def _certify(self, states, step_options, sim_env=None):
            self.seen.append((sim_env, step_options))
            return True, "internal reason"

    recording = _Recording()
    fn = make_engine_free_evaluate_trajectory_fn(
        lambda idx: Task(start, set(), evaluator=recording))
    verdict = fn([start, done], [("Pour", ("cup", ), (0.5, ))])
    assert set(verdict) == {"reward", "solved", "note"}
    assert "internal reason" not in str(verdict)
    assert recording.seen and all(env is None and labels == [("Pour",
                                                              ("cup", ),
                                                              (0.5, ))]
                                  for env, labels in recording.seen)


def test_engine_free_helper_notes_an_unchecked_replay():
    """With no simulator the Domino push replay is not run: a goal-reaching
    sequence that breaks no other rule is scored unsolved with no bonus, and
    the note says the rule was not checked.

    A broken rule or an unreached goal carries no note, and the EMPIRIC
    helper's verdict without a simulator is what it was.
    """
    from predicators.agent_sdk.tools.reward_model import \
        REPLAY_UNCHECKED_NOTE, score_prediction, score_trajectory
    from predicators.envs.pybullet_domino.env import DominoEvaluator
    from tests.envs.test_cascade_certificate import _build_states, \
        _FakeProbe, _legit_chain_case, _options, _real_goal

    utils.reset_config({
        "domino_block_cost": 0.05,
        "domino_min_block_num_blues": 4
    })
    objs, states, labels = _legit_chain_case()
    goal = _real_goal(objs)
    task = Task(states[0], goal, evaluator=DominoEvaluator(goal))

    verdict = score_trajectory(task, states, labels, replay_engine=False)
    # One blue toppled: its cost, and no success bonus.
    assert verdict == {
        "reward": pytest.approx(-0.05),
        "solved": False,
        "note": REPLAY_UNCHECKED_NOTE
    }
    assert "not checked" in verdict["note"]
    assert "counterfactual" not in verdict["note"]
    # Without labels the rule is reached the same way.
    assert score_trajectory(task, states, None,
                            replay_engine=False)["note"] \
        == REPLAY_UNCHECKED_NOTE

    # The EMPIRIC helper with no simulator bound: the same fail-closed
    # verdict, and the empty note it always had.
    assert score_trajectory(task, states, labels) == {
        "reward": pytest.approx(-0.05),
        "solved": False,
        "note": ""
    }

    # A simulator offered to an arm that has none is never used.
    class _Env:
        run_counterfactual_cascade_probe = _FakeProbe(True, "push: fell")

    offered = score_trajectory(task,
                               states,
                               labels,
                               sim_env=_Env(),
                               replay_engine=False)
    assert offered["solved"] is False and not _Env.\
        run_counterfactual_cascade_probe.calls
    assert offered["note"] == REPLAY_UNCHECKED_NOTE
    # With one, the EMPIRIC helper runs the replay and names it.
    replayed = score_trajectory(task, states, labels, sim_env=_Env())
    assert replayed["solved"] is True and replayed["note"] == "push"

    # A rule that needs no physics is broken: no note, as nothing was
    # left unchecked.
    pushed_blue = _options([("Push", ("robot", "blue1"), 10, 13)], 30)
    broken = score_trajectory(task, states, pushed_blue, replay_engine=False)
    assert broken["solved"] is False and broken["note"] == ""
    # The goal is not reached: the replay rule has nothing to decide.
    quiet = _build_states(objs,
                          30, {
                              "green": 12,
                              "blue1": 16
                          },
                          positions={
                              "green": (0.7, 1.0),
                              "blue1": (0.7, 1.098),
                              "target": (0.7, 1.196)
                          })
    unreached = score_trajectory(task, quiet, labels, replay_engine=False)
    assert unreached["solved"] is False and unreached["note"] == ""

    # One state shows no fall to replay. The certificate passes it with
    # or without a simulator, so the fallen final state alone is scored
    # a success: the evaluator's own rule, here as in the EMPIRIC helper.
    final = score_trajectory(task, states[-1:], None, replay_engine=False)
    assert final == score_trajectory(task, states[-1:], None, sim_env=_Env())
    assert final == {"reward": pytest.approx(0.95), "solved": True, "note": ""}

    # The same sequence as the direct agent's tool takes it: a start
    # state, then the features that change at each step.
    rolls = [{
        name: {
            "roll": float(after.get(obj, "roll"))
        }
        for name, obj in objs.items()
        if after.get(obj, "roll") != before.get(obj, "roll")
    } for before, after in zip(states[:-1], states[1:])]
    as_json = [
        None if lab is None else [lab[0], list(lab[1])] for lab in labels
    ]
    predicted = score_prediction(task, [states[0]], [], rolls, as_json)
    assert predicted == verdict
    with pytest.raises(ValueError, match="one label per predicted state"):
        score_prediction(task, [states[0]], [], rolls, as_json[:-1])
    with pytest.raises(ValueError, match=r"states\[0\]: Unknown object"):
        score_prediction(task, [states[0]], [], [{"nope": {"roll": 1.0}}])
    with pytest.raises(ValueError, match=r"actions\[0\] must be null or"):
        score_prediction(task, [states[0]], [], [{}], ["Push"])
    # A recorded prefix keeps its labels in front of the predicted part.
    split = 15
    continued = score_prediction(task, states[:split + 1], labels[:split],
                                 rolls[split:], as_json[split:])
    assert continued == verdict


def test_engine_free_helper_rescores_an_edited_sequence():
    """A sequence edited in place and scored again gets its own verdict.

    The Domino certificate keeps its last verdict by the identity of the
    scored objects, and the list the helper builds per call can take the
    address of the one before it.
    """
    from predicators.agent_sdk.tools.reward_model import \
        REPLAY_UNCHECKED_NOTE, level_task, \
        make_engine_free_evaluate_trajectory_fn
    from predicators.envs.pybullet_domino.env import DominoEvaluator
    from tests.envs.test_cascade_certificate import _legit_chain_case, \
        _real_goal

    utils.reset_config({
        "domino_block_cost": 0.05,
        "domino_min_block_num_blues": 4
    })
    objs, states, _ = _legit_chain_case()
    goal = _real_goal(objs)
    task = Task(states[0], goal, evaluator=DominoEvaluator(goal))
    fn = make_engine_free_evaluate_trajectory_fn(
        functools.partial(level_task, [task], None))
    for _ in range(20):
        assert fn(states)["note"] == REPLAY_UNCHECKED_NOTE
        # The target stays up: the goal is not reached any more.
        fallen = [s.get(objs["target"], "roll") for s in states]
        for s in states:
            s.set(objs["target"], "roll", 0.0)
        assert fn(states)["note"] == ""
        for s, roll in zip(states, fallen):
            s.set(objs["target"], "roll", roll)
