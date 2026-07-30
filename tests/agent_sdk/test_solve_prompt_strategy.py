"""Solve-prompt strategy blocks from the run_20260729_001752 post-mortem.

That run burned five attempts hunting minimal designs at the designed
k*-boundary (while its sibling banked an over-built solve), induced
false reward rules from opaque scores, re-litigated a journal-condemned
family while a concrete untried lead sat unexecuted, and turned an
unvalidated fall-direction formula into a false physics law. Each block
tested here is the corresponding general fix.
"""
import numpy as np

from predicators import utils
from predicators.agent_sdk.sketch_prompts import build_solve_prompt
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


def _render(task: Task, journal: str = "") -> str:
    return build_solve_prompt(task,
                              all_predicates=set(),
                              all_options=set(),
                              propose_params=True,
                              require_tool_validation=True,
                              journal=journal,
                              physics_margin=True)


def test_scoring_section_from_evaluator() -> None:
    """The evaluator's public reward form renders as a Scoring section; without
    an evaluator the section is absent."""
    utils.reset_config({"seed": 0})
    prompt = _render(_make_task(_StubEvaluator(set())))
    assert "## Scoring (env ground-truth reward)" in prompt
    assert "0.2 x widgets used" in prompt
    assert "Decode every reward you observe" in prompt
    bare = _render(_make_task(None))
    assert "## Scoring" not in bare


def test_capture_first_strategy_block() -> None:
    """CAPTURE FIRST guidance states the verified capture semantics: a
    validated capture replaces the banked one, rejections never do."""
    utils.reset_config({"seed": 0})
    prompt = _render(_make_task(None))
    assert "CAPTURE FIRST, OPTIMIZE SECOND" in prompt
    assert "never displaces it" in prompt
    assert "not strictly better" in prompt


def test_journal_protocol_block() -> None:
    """The journal preamble carries the three epistemic rules; no journal, no
    protocol block."""
    utils.reset_config({"seed": 0})
    prompt = _render(_make_task(None), journal="- notes")
    assert "FIRST list the journal's untried leads" in prompt
    assert "only as broad as the family actually swept" in prompt
    assert "BOTH demote to open questions" in prompt
    assert "Journal protocol" not in _render(_make_task(None))


def test_formula_validation_advice() -> None:
    """Derived formulas must be validated before steering a search."""
    utils.reset_config({"seed": 0})
    prompt = _render(_make_task(None))
    assert "validate the formula on one clean controlled experiment" in prompt
