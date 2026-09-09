"""Tests for synthesis-prompt formatter helpers.

These are pure-Python staticmethods (or `self`-less methods) on
``AgentSimLearningApproach`` and ``AgentSimPredicateInventionApproach``
that render parts of the agent's first synthesis message. They were
added so the agent (a) knows the provenance of each interaction
trajectory and (b) gets reminded about prior-cycle files in the sandbox.
"""
# pylint: disable=protected-access,import-outside-toplevel,unused-import
from __future__ import annotations

import numpy as np
import pytest

# Bootstrap circular imports before pulling from predicators.approaches.
from predicators import utils
from predicators.structs import Action, LowLevelTrajectory, State, Task, Type


@pytest.fixture(name="approach_cls")
def _approach_cls():
    """Late-import the class so test collection is cheap."""
    from predicators.approaches.agent_sim_learning_approach import \
        AgentSimLearningApproach
    return AgentSimLearningApproach


def _mk_traj(is_demo,
             task_idx,
             sim_v=None,
             preds_v=None,
             reward=None,
             terminated=None):
    """Build a 1-action trajectory with the given provenance tags."""
    cup_type = Type("cup_type", ["f"])
    cup = cup_type("cup")
    states = [State({cup: [0.0]}), State({cup: [1.0]})]
    actions = [Action(np.array([0.5]))]
    return LowLevelTrajectory(
        states,
        actions,
        _is_demo=is_demo,
        _train_task_idx=task_idx,
        _source_simulator_version=sim_v,
        _source_predicates_version=preds_v,
        _env_reward=reward,
        _env_terminated=terminated,
    )


# ── _format_trajectory_listing ──────────────────────────────────────


def test_trajectory_listing_empty(approach_cls):
    """Empty trajectory list short-circuits to an empty string."""
    assert approach_cls._format_trajectory_listing([]) == ""


def test_trajectory_listing_demo_has_no_provenance_tail(approach_cls):
    """Demo trajectories never carry provenance - even if the tags are set, the
    listing should still render them as plain demos for consistency with the
    offline-data semantics."""
    trajs = [_mk_traj(is_demo=True, task_idx=0)]
    out = approach_cls._format_trajectory_listing(trajs)
    assert "[0] demo, task 0" in out
    assert "generated using" not in out


def test_trajectory_listing_interaction_with_provenance(approach_cls):
    """Interaction trajectories with provenance show the sim/preds tags."""
    trajs = [
        _mk_traj(is_demo=False,
                 task_idx=2,
                 sim_v="cycle_001_vers_004",
                 preds_v="cycle_001_vers_003"),
    ]
    out = approach_cls._format_trajectory_listing(trajs)
    assert "[0] interaction, task 2" in out
    assert "sim cycle_001_vers_004" in out
    assert "predicates cycle_001_vers_003" in out


def test_trajectory_listing_supervisor_rejected(approach_cls):
    """A rejected episode surfaces only through its (reward, terminated)
    pair: terminated with solved=0 and no bonus in the reward. No
    REJECTED flag or violation specifics reach the roster - the rules
    live in the NL goal description, so the agent must infer the
    violation from its own trajectory rather than be told it.
    """
    trajs = [
        _mk_traj(is_demo=False, task_idx=0),
        _mk_traj(is_demo=False, task_idx=3, reward=-0.05, terminated=True),
    ]
    out = approach_cls._format_trajectory_listing(trajs)
    lines = [l for l in out.splitlines() if l.startswith("  [")]
    assert "REJECTED" not in out
    assert "env reward=-0.05 (solved=0)" in lines[1]
    # No violation specifics leak into the roster line.
    assert "domino" not in lines[1]
    assert "push" not in lines[1].lower()


def test_trajectory_listing_env_reward(approach_cls):
    """Evaluated episodes show the env reward with a success flag; a rejected
    topple counts as solved=0 even though it terminated."""
    trajs = [
        _mk_traj(is_demo=False, task_idx=0, reward=0.85, terminated=True),
        _mk_traj(is_demo=False, task_idx=1, reward=-0.05, terminated=True),
        _mk_traj(is_demo=False, task_idx=2),  # never evaluated
    ]
    out = approach_cls._format_trajectory_listing(trajs)
    lines = [l for l in out.splitlines() if l.startswith("  [")]
    assert "env reward=0.85 (solved=1)" in lines[0]
    assert "env reward=-0.05 (solved=0)" in lines[1]
    assert "REJECTED" not in lines[1]
    assert "reward" not in lines[2]


def test_trajectory_listing_partial_provenance(approach_cls):
    """Only ``source_simulator_version`` set: list just the sim tag.

    No stray ``, `` may appear from the missing predicate half of the
    provenance pair.
    """
    trajs = [_mk_traj(is_demo=False, task_idx=1, sim_v="cycle_001_vers_007")]
    out = approach_cls._format_trajectory_listing(trajs)
    line = [l for l in out.splitlines() if l.startswith("  [0]")][0]
    assert "sim cycle_001_vers_007" in line
    assert "predicates" not in line


# ── _format_prior_state_block ────────────────────────────────────────


def test_prior_state_block_empty_when_no_files(approach_cls, tmp_path):
    """Neither simulator.py nor predicates.py exists → empty block."""
    out = approach_cls._format_prior_state_block(None, str(tmp_path))
    assert out == ""


def test_prior_state_block_simulator_only(approach_cls, tmp_path):
    """Only simulator.py exists → block mentions it and not predicates.py."""
    (tmp_path / "simulator.py").write_text("# sim")
    out = approach_cls._format_prior_state_block(None, str(tmp_path))
    assert "`./simulator.py`" in out
    assert "`./predicates.py`" not in out
    # Always points at the versioned-snapshot dirs for cross-reference.
    assert "./simulator_versions/" in out


def test_prior_state_block_both_files(approach_cls, tmp_path):
    """Both files exist → block lists them joined with ' and '."""
    (tmp_path / "simulator.py").write_text("# sim")
    (tmp_path / "predicates.py").write_text("LEARNED_PREDICATES = []")
    out = approach_cls._format_prior_state_block(None, str(tmp_path))
    assert "`./simulator.py` and `./predicates.py`" in out
    # Soft language so the agent isn't forbidden from a fresh rewrite.
    assert "fresh rewrite is fine" in out


# ── _format_goal_nl_block (predicate-invention subclass) ────────────


def test_goal_nl_block_empty_when_no_tasks_have_goal_nl():
    """No ``goal_nl`` populated → empty block (no header)."""
    from predicators.approaches.agent_sim_predicate_invention_approach import \
        AgentSimPredicateInventionApproach
    fake_self = type(
        "_FakeApproach",
        (),
        {
            "_train_tasks": [type("_T", (), {"goal_nl": None})()] * 2,
        },
    )()
    out = AgentSimPredicateInventionApproach._format_goal_nl_block(fake_self)
    assert out == ""


def test_goal_nl_block_dedups_identical_goals():
    """Same NL goal across tasks shows up once, with the single-task header."""
    from predicators.approaches.agent_sim_predicate_invention_approach import \
        AgentSimPredicateInventionApproach
    fake_task = type("_T", (), {"goal_nl": "boil the water"})
    fake_self = type(
        "_FakeApproach",
        (),
        {
            "_train_tasks": [fake_task() for _ in range(3)],
        },
    )()
    out = AgentSimPredicateInventionApproach._format_goal_nl_block(fake_self)
    assert out.startswith("Goal (natural language): boil the water")
    # Trailing blank line separates from the next paragraph in the prompt.
    assert out.endswith("\n\n")


def test_goal_nl_block_multiple_distinct_goals():
    """Distinct goals across tasks render as a bulleted list."""
    from predicators.approaches.agent_sim_predicate_invention_approach import \
        AgentSimPredicateInventionApproach
    tasks = [
        type("_T1", (), {"goal_nl": "boil the water"})(),
        type("_T2", (), {"goal_nl": "stack the cups"})(),
    ]
    fake_self = type("_FakeApproach", (), {"_train_tasks": tasks})()
    out = AgentSimPredicateInventionApproach._format_goal_nl_block(fake_self)
    assert "Goals across train tasks (natural language):" in out
    assert "  - boil the water" in out
    assert "  - stack the cups" in out


# ── _build_synthesis_system_prompt (FO vs PO rule signature) ────────
# These render the whole synthesis system prompt. The method only touches
# ``self`` through pure no-state helpers (``_rule_signature_section``,
# ``_residual_rule_signature``, ``_extra_synthesis_system_prompt``), so a
# bare instance via ``object.__new__`` is enough to render it.


def _render_prompt(cls):
    from predicators.approaches.agent_sim_learning_approach import \
        AgentSimLearningApproach
    return AgentSimLearningApproach._build_synthesis_system_prompt(
        object.__new__(cls))


def test_synthesis_prompt_no_leftover_placeholders(approach_cls):
    """Every templated placeholder is substituted in the rendered prompt."""
    prompt = _render_prompt(approach_cls)
    for placeholder in ("__RULE_SIGNATURE_SECTION__",
                        "__RESIDUAL_RULE_SIGNATURE__",
                        "__SYNTHESIS_PROMPT_EXTRA__"):
        assert placeholder not in prompt


def test_synthesis_prompt_sections_not_duplicated(approach_cls):
    """The system prompt has exactly one of each major section.

    Guards against the bad-merge artifact that duplicated the Tools /
    Refinement / Plan-format blocks (and double-injected the extra).
    """
    prompt = _render_prompt(approach_cls)
    for header in ("## `simulator.py`: a simulator subclass",
                   "## Step and restoration behavior", "## Plan format",
                   "## Fit and validate complete rollouts"):
        assert prompt.count(header) == 1, (header, prompt.count(header))


def test_fo_prompt_uses_subclass_contract(approach_cls):
    """Fully observable models use the same subclass contract as PO ones."""
    prompt = _render_prompt(approach_cls)
    assert "class MyDynamics(BaseSimulator):" in prompt
    assert "RESIDUAL_ENV = MyDynamics" in prompt
    assert "def _domain_specific_step(self):" in prompt
    assert "RESIDUAL_RULES" not in prompt
    assert "def residual_rule(" not in prompt
    assert "## Hidden model state" not in prompt


def test_po_prompt_uses_subclass_memory_contract():
    """Both PO approaches receive one canonical model-state callback.

    A competing rule signature must not reappear beside the subclass
    contract, and only predicate invention adds classifier guidance.
    """
    import re

    from predicators.approaches.agent_sim_learning_approach import \
        AgentSimLearningApproach
    from predicators.approaches.agent_sim_predicate_invention_approach import \
        AgentSimPredicateInventionApproach
    from predicators.settings import CFG
    old_flag = CFG.partially_observable
    CFG.partially_observable = True
    try:
        for cls in (AgentSimLearningApproach,
                    AgentSimPredicateInventionApproach):
            prompt = _render_prompt(cls)
            assert "class MyDynamics(BaseSimulator):" in prompt
            assert ("def update_model_state(cls, observation, model_state, "
                    "params, action):" in prompt)
            assert "RESIDUAL_RULES" not in prompt
            assert "LATENT_INIT" not in prompt
            assert "def residual_rule(" not in prompt
            # Memory guidance is injected exactly once.
            headers = re.findall(r"(?m)^## Hidden model state$", prompt)
            assert len(headers) == 1, cls
            # The predicate-side latent guidance is invention-only.
            has_pred_section = "### Predicate signature" in prompt
            assert has_pred_section == (
                cls is AgentSimPredicateInventionApproach), cls
    finally:
        CFG.partially_observable = old_flag


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
    # Public pair only: neither the certificate's legitimacy bool and
    # reason nor terminated (agent-computable via is_goal_state) cross
    # into the agent's namespace.
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


def test_format_objective_block(approach_cls):
    """The objective block renders the first stated objective and is empty when
    no evaluator states one."""
    from types import SimpleNamespace

    from predicators.structs import TaskEvaluator

    class _StatingEvaluator(TaskEvaluator):
        """Evaluator with a public objective statement."""

        def objective_description(self):
            return "Topple the target legitimately; each blue costs 0.05."

    cup_type = Type("cup_type", ["f"])
    init = State({cup_type("cup"): [0.0]})

    def _task(evaluator=None):
        return Task(init, set(), evaluator=evaluator)

    fmt = approach_cls._format_objective_block
    assert fmt(SimpleNamespace(_train_tasks=[])) == ""
    assert fmt(SimpleNamespace(_train_tasks=[_task()])) == ""
    assert fmt(
        SimpleNamespace(_train_tasks=[_task(TaskEvaluator(set()))])) == ""
    out = fmt(
        SimpleNamespace(
            _train_tasks=[_task(), _task(_StatingEvaluator(set()))]))
    assert "## Task objective (env ground-truth reward)" in out
    assert "each blue costs 0.05" in out
    assert "evaluate_trajectory" in out


def test_learn_message_ships_goal_required_mechanisms_as_hypotheses():
    """The learn message distinguishes a hypothesis the goal can do without
    (record, do not ship) from one the goal REQUIRES (ship as a labelled
    hypothesis with declared ParamSpecs and a first-ranked confirming
    experiment).

    The rule lives in the learn system prompt's deliverables section.
    """
    from predicators.agent_sdk import learn_prompts
    prompt = learn_prompts.build_learn_system_prompt(
        partially_observable=False,
        residual_rule_signature="def residual_rule(state, updates, params):",
        scene_viz_hint="render the scene")
    assert "When the goal requires it" in prompt
    assert "labelled hypothesis" in prompt
    assert "first entry of `./open_questions.md`" in prompt
    assert "A GO that rests on a hypothesized mechanism" in prompt
