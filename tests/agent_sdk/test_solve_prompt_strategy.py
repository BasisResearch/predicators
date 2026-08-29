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


def _render(task: Task, journal: str = "", strategy: str = "") -> str:
    return build_solve_prompt(task,
                              all_predicates=set(),
                              all_options=set(),
                              propose_params=True,
                              require_tool_validation=True,
                              journal=journal,
                              strategy=strategy,
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


def _render_explore(task: Task, propose_params: bool = True) -> str:
    return build_solve_prompt(task,
                              all_predicates=set(),
                              all_options=set(),
                              propose_params=propose_params,
                              require_tool_validation=False,
                              explore_mode=True)


def test_explore_mode_belief_model_disclosure() -> None:
    """Explore queries disclose that the simulator is a belief model whose
    unlearned mechanisms are absent, so a null effect is not evidence about
    reality; solve queries carry no such section."""
    utils.reset_config({"seed": 0})
    prompt = _render_explore(_make_task(None))
    assert "## Exploration Setting" in prompt
    assert "not in the belief model yet" in prompt
    solve_prompt = _render(_make_task(None))
    assert "## Exploration Setting" not in solve_prompt


def test_explore_mode_experiment_delivery_contract() -> None:
    """Explore queries accept a simulator-failing sketch as the deliverable and
    forbid grinding for an impossible capture; the closing block marks final
    text as the deliverable."""
    utils.reset_config({"seed": 0})
    prompt = _render_explore(_make_task(None))
    assert "simulator-failing sketch is a valid deliverable" in prompt
    assert ("grinding for a validated plan the model cannot produce "
            "is wasted budget") in prompt
    assert "This is an EXPLORE query" in prompt
    solve_prompt = _render(_make_task(None))
    assert "This is an EXPLORE query" not in solve_prompt
    assert "do NOT finish until evaluate_option_plan CONFIRMS" in solve_prompt
    # The contract must survive param-free sketch mode too (the search
    # finds continuous params, but the delivery semantics are the same).
    sketch_prompt = _render_explore(_make_task(None), propose_params=False)
    assert "simulator-failing sketch is a valid deliverable" in sketch_prompt
    assert "This is an EXPLORE query" in sketch_prompt


def test_explore_mode_opening_focuses_on_information_gathering() -> None:
    """Explore queries open with an exploration framing (solving the task is
    part of information gathering); solve queries keep the solving opening."""
    utils.reset_config({"seed": 0})
    prompt = _render_explore(_make_task(None))
    assert prompt.startswith("You are exploring a task environment")
    assert "part of information gathering" in prompt
    solve_prompt = _render(_make_task(None))
    assert solve_prompt.startswith("You are solving a task.")
    assert "part of information gathering" not in solve_prompt


def test_explore_mode_early_stop_note_credits_exploration_plans() -> None:
    """The early-stop note attributes loop conclusion to the exploration plans
    solving training (goal reached for real AND validated in the belief model),
    and it states the refine-then-seeded-fallback mechanism."""
    utils.reset_config({
        "seed":
        0,
        "online_learning_early_stopping":
        True,
        "online_learning_early_stopping_require_all_attempts":
        True,
    })
    prompt = _render_explore(_make_task(None))
    assert ("The loop concludes early once the exploration plans solve "
            "training") in prompt
    assert "learned model solves training" not in prompt
    assert "runs in the real environment EXACTLY as written" in prompt
    assert "nothing is searched or substituted" in prompt
    assert "falls back to the explicit parameters" not in prompt
    assert "truncated just after that step" not in prompt


def test_domain_strategy_block_is_advisory() -> None:
    """strategy.md content renders as an advisory section; absent without."""
    utils.reset_config({"seed": 0})
    prompt = _render(_make_task(None), strategy="## Glue first\n- dab twice")
    assert "## Domain Strategy (advisory, written during learning)" in prompt
    assert "- dab twice" in prompt
    assert "can be wrong or stale" in prompt
    assert "you are NOT limited to it" in prompt
    assert "## Domain Strategy" not in _render(_make_task(None))
