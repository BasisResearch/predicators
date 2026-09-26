"""Tests for the ``evaluate_trajectory`` helper the synthesis namespace
offers."""
# pylint: disable=protected-access,import-outside-toplevel,unused-import
from __future__ import annotations

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
                           _option_model=None)
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
    with pytest.raises(ValueError, match="no task evaluator"):
        fn(states, None, task_idx=1)
    with pytest.raises(ValueError, match="out of range"):
        fn(states, None, task_idx=2)
    with pytest.raises(ValueError, match="non-empty"):
        fn([], None, task_idx=0)


def test_evaluate_trajectory_physics_sweep(approach_cls):
    """physics_sweep=True scores the sequence at every physics-margin point on
    a fresh env at that physics and reports the fraction scored solved; with no
    points to sweep it says so."""
    import contextlib
    import functools
    from types import SimpleNamespace

    from predicators.structs import TaskEvaluator

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
        _fresh_validation_env_scope=_scope)
    stub._sweep_evaluation = functools.partial(approach_cls._sweep_evaluation,
                                               stub)
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
