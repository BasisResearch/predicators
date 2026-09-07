"""Behavioral checks on the solve/explore prompts.

Each test pins one rule the prompts must carry (the goldens in
``test_prompt_goldens.py`` pin the full text): the reward form reaches
the solver, solutions are banked before being optimized, the run records
are used skeptically, and exploration is framed as experiment design
against a belief model.
"""
import numpy as np

from predicators import utils
from predicators.agent_sdk.sketch_prompts import build_early_stop_note, \
    build_solve_prompt, build_solve_system_prompt
from predicators.structs import Object, State, Task, TaskEvaluator, Type

_DOM = Type("thing", ["x"])


class _StubEvaluator(TaskEvaluator):
    """Evaluator whose objective statement must reach the prompt."""

    def objective_description(self) -> str:
        return "reward = (1.0 if success else 0.0) - 0.2 x widgets used"


def _make_task(evaluator=None) -> Task:
    obj = Object("thing0", _DOM)
    return Task(State({obj: np.array([0.0])}),
                set(),
                goal_nl="Do the thing.",
                evaluator=evaluator)


def _render(task: Task, journal: str = "", strategy: str = "") -> str:
    return build_solve_prompt(task,
                              all_predicates=set(),
                              all_options=set(),
                              propose_params=True,
                              require_tool_validation=True,
                              journal=journal,
                              strategy=strategy)


def _render_explore(task: Task, propose_params: bool = True) -> str:
    return build_solve_prompt(task,
                              all_predicates=set(),
                              all_options=set(),
                              propose_params=propose_params,
                              require_tool_validation=False,
                              explore_mode=True)


def test_scoring_section_from_evaluator() -> None:
    """The evaluator's public reward form renders as a Scoring section; without
    an evaluator the section is absent."""
    utils.reset_config({"seed": 0})
    prompt = _render(_make_task(_StubEvaluator(set())))
    assert "## Scoring (env ground-truth reward)" in prompt
    assert "0.2 x widgets used" in prompt
    assert "Decode every reward you observe" in prompt
    assert "## Scoring" not in _render(_make_task(None))


def test_solve_system_prompt_banks_before_optimizing() -> None:
    """The solve system prompt states the banking semantics (a validated
    capture replaces the banked one, a rejection never does); the explore
    system prompt, which has no capture deliverable, omits them."""
    prompt = build_solve_system_prompt(explore=False)
    assert "Bank a solution before optimizing it" in prompt
    assert "a rejected submission never displaces it" in prompt
    assert "strictly better" in prompt
    assert "Bank a solution" not in build_solve_system_prompt(explore=True)


def test_solve_system_prompt_journal_protocol() -> None:
    """The run-record protocol (incumbent, untried leads, scoped negatives,
    conflicting entries) lives in the solve system prompt and only there."""
    prompt = build_solve_system_prompt(explore=False, use_journal=True)
    assert "is the incumbent" in prompt
    assert "untried leads first" in prompt
    assert "only as broad as the family actually swept" in prompt
    assert "both become open questions" in prompt
    assert "Journal protocol" not in build_solve_system_prompt(
        explore=False, use_journal=False)
    assert "Journal protocol" not in build_solve_system_prompt(explore=True)


def test_solve_system_prompt_verifies_rules_before_steering() -> None:
    """A rule inferred from one observation is re-tested before it guides the
    search."""
    prompt = build_solve_system_prompt(explore=False)
    assert "Verify a rule before steering by it" in prompt


def test_query_carries_run_records_not_their_protocol() -> None:
    """The query renders the journal and strategy contents under their headers;
    the rules for using them are not repeated there."""
    utils.reset_config({"seed": 0})
    prompt = _render(_make_task(None),
                     journal="- notes",
                     strategy="## Glue first\n- dab twice")
    assert "## Solve Journal (./journal.md)" in prompt
    assert "- notes" in prompt
    assert "## Domain Strategy (advisory, written during learning)" in prompt
    assert "- dab twice" in prompt
    assert "incumbent" not in prompt
    bare = _render(_make_task(None))
    assert "## Solve Journal" not in bare
    assert "## Domain Strategy" not in bare


def test_explore_system_prompt_states_the_setting() -> None:
    """The explore system prompt discloses the belief model, accepts a
    simulator-failing plan, states the cycle data contract and first-cycle
    coverage, and executes parameters verbatim; the solve prompt has none."""
    prompt = build_solve_system_prompt(explore=True)
    assert "## Exploration setting" in prompt
    assert "not in the belief model yet" in prompt
    assert "a simulator-failing plan is a valid deliverable" in prompt
    assert "at least one attempt at the full goal" in prompt
    assert ("top-ranked open question's experiment executed as specified"
            in prompt)
    assert "Carry each interaction to its consequence" in prompt
    assert "nothing is searched or substituted" in prompt
    solve = build_solve_system_prompt(explore=False)
    assert "## Exploration setting" not in solve
    assert "belief model yet" not in solve


def test_explore_deliverable_and_certified_note() -> None:
    """The explore deliverable is the final plan text; the certified-plan note
    follows the execute-certified-plan flag."""
    with_note = build_solve_system_prompt(explore=True,
                                          execute_certified_plan=True)
    assert "Your final plan text: the experiment" in with_note
    assert "executed verbatim as this episode's solve attempt" in with_note
    without = build_solve_system_prompt(explore=True,
                                        execute_certified_plan=False)
    assert "executed verbatim" not in without


def test_early_stop_note_follows_config() -> None:
    """The early-stop note credits certified exploration plans that solve for
    real (train-driven) or perfect test phases (test-driven), and is absent
    when early stopping is off."""
    utils.reset_config({
        "seed":
        0,
        "online_learning_early_stopping":
        True,
        "online_learning_early_stopping_require_all_attempts":
        True,
    })
    note = build_early_stop_note()
    assert note.startswith("The loop concludes early once the exploration")
    assert "every episode of a cycle" in note
    prompt = build_solve_system_prompt(explore=True, early_stop_note=note)
    assert "exploration plans solve training" in prompt
    utils.reset_config({
        "seed":
        0,
        "online_learning_early_stopping":
        True,
        "online_learning_early_stopping_by_test_solve_rate":
        True,
        "online_learning_early_stopping_consecutive_perfect_tests":
        2,
    })
    assert "2 consecutive test phases" in build_early_stop_note()
    utils.reset_config({"seed": 0, "online_learning_early_stopping": False})
    assert build_early_stop_note() == ""


def test_query_openings_by_phase() -> None:
    """Explore queries open with the information-gathering framing; solve
    queries with the solving framing.

    Both end in the instructions.
    """
    utils.reset_config({"seed": 0})
    explore = _render_explore(_make_task(None))
    assert explore.startswith("Design this episode's experiment")
    assert "part of information gathering" in explore
    assert "output the plan lines as your final text" in explore
    solve = _render(_make_task(None))
    assert solve.startswith("Solve the task below")
    assert "deliver it through the capture gate" in solve
    # Param-free sketch mode keeps the same delivery semantics.
    sketch = _render_explore(_make_task(None), propose_params=False)
    assert "output the plan lines as your final text" in sketch
