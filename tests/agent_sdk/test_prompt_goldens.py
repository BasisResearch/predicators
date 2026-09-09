"""Golden renders of every agent prompt.

Each phase's system prompt and query is rendered for a fixed
configuration and compared byte-for-byte against
``tests/agent_sdk/prompt_goldens/<name>.md``. The goldens are the
reviewable form of the prompts: a change to a template or a builder
shows up as a diff of the text the agent actually receives.

To accept a deliberate change, regenerate the goldens and review the
diff::

    UPDATE_PROMPT_GOLDENS=1 pytest tests/agent_sdk/test_prompt_goldens.py
"""
import difflib
import glob
import os
import re

import numpy as np
import pytest
from gym.spaces import Box

from predicators import utils
from predicators.agent_sdk import learn_prompts
from predicators.agent_sdk.prompt_templates import _PROMPTS_DIR, \
    load_sections, placeholders, render
from predicators.agent_sdk.sandbox_prompts import build_claude_md
from predicators.agent_sdk.sketch_prompts import build_early_stop_note, \
    build_solve_prompt, build_solve_system_prompt
from predicators.structs import Action, GroundAtom, Object, \
    ParameterizedOption, Predicate, State, Task, TaskEvaluator, Type

_GOLDEN_DIR = os.path.join(os.path.dirname(__file__), "prompt_goldens")

# -- fixtures ----------------------------------------------------------------

_THING = Type("thing", ["x", "y"])
_FIXTURE = Type("fixture", ["x", "y", "is_on"])


def _noop_policy(_s, _m, _o, _p):
    return Action(np.zeros(1, dtype=np.float32))


_MOVE = ParameterizedOption(
    "MoveTo",
    types=[_THING, _FIXTURE],
    params_space=Box(low=np.array([-0.5, -0.5], dtype=np.float32),
                     high=np.array([0.5, 0.5], dtype=np.float32)),
    policy=_noop_policy,
    initiable=lambda _s, _m, _o, _p: True,
    terminal=lambda _s, _m, _o, _p: True,
    params_description=("dx", "dy"),
)
_WAIT = ParameterizedOption(
    "Wait",
    types=[],
    params_space=Box(low=np.zeros(0, dtype=np.float32),
                     high=np.zeros(0, dtype=np.float32)),
    policy=_noop_policy,
    initiable=lambda _s, _m, _o, _p: True,
    terminal=lambda _s, _m, _o, _p: True,
)
_AT = Predicate(
    "At", [_THING, _FIXTURE],
    lambda s, o: abs(s.get(o[0], "x") - s.get(o[1], "x")) < 0.1,
    natural_language_assertion=lambda names: f"{names[0]} rests on {names[1]}")
_ON = Predicate("Active", [_FIXTURE], lambda s, o: s.get(o[0], "is_on") > 0.5)


class _Evaluator(TaskEvaluator):
    """Evaluator whose objective statement reaches the prompt."""

    def objective_description(self) -> str:
        return "reward = (1.0 if success else 0.0) - 0.1 x moves used"


def _make_task(with_evaluator: bool) -> Task:
    thing = Object("thing0", _THING)
    fixture = Object("fixture0", _FIXTURE)
    state = State({
        thing: np.array([0.0, 0.0]),
        fixture: np.array([1.0, 0.0, 0.0])
    })
    goal = {GroundAtom(_AT, [thing, fixture]), GroundAtom(_ON, [fixture])}
    return Task(state,
                goal,
                goal_nl="Put the thing on the fixture and switch it on.",
                evaluator=_Evaluator(goal) if with_evaluator else None)


# -- golden comparison -------------------------------------------------------


def _check_golden(name: str, text: str) -> None:
    path = os.path.join(_GOLDEN_DIR, f"{name}.md")
    if os.environ.get("UPDATE_PROMPT_GOLDENS"):
        os.makedirs(_GOLDEN_DIR, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        return
    assert os.path.isfile(path), (
        f"missing golden {path}; run with UPDATE_PROMPT_GOLDENS=1")
    with open(path, "r", encoding="utf-8") as f:
        expected = f.read()
    if text != expected:
        diff = "\n".join(
            difflib.unified_diff(expected.splitlines(),
                                 text.splitlines(),
                                 fromfile=f"{name}.md (golden)",
                                 tofile=f"{name} (rendered)",
                                 lineterm=""))
        pytest.fail(f"prompt {name} drifted from its golden; review the diff "
                    f"and rerun with UPDATE_PROMPT_GOLDENS=1 to accept:\n"
                    f"{diff}")


# -- template hygiene --------------------------------------------------------


def test_templates_use_plain_dashes_and_declared_sections() -> None:
    """Templates contain no em dashes, and every section is named once."""
    paths = sorted(glob.glob(os.path.join(_PROMPTS_DIR, "*.md")))
    assert paths
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        assert "—" not in text, path
        name = os.path.splitext(os.path.basename(path))[0]
        assert load_sections(name)


def test_render_rejects_missing_and_unused_placeholders() -> None:
    """A template placeholder without a value, or a value without a
    placeholder, fails loudly instead of shipping literal text."""
    with pytest.raises(AssertionError):
        render("solve_query", "goal_nl")
    with pytest.raises(AssertionError):
        render("solve_query", "goal_nl", goal_nl="g", extra="x")
    assert placeholders("a __B__ c __B__ __D__") == ["B", "D"]


def test_rendered_prompts_have_no_leftover_placeholders() -> None:
    """No rendered prompt carries an unsubstituted ``__NAME__``."""
    marker = re.compile(r"__[A-Z][A-Z0-9_]*__")
    for text in (build_solve_system_prompt(explore=False),
                 build_solve_system_prompt(explore=True), build_claude_md()):
        assert not marker.search(text), marker.search(text)


# -- solve / explore ---------------------------------------------------------


def test_golden_solve_system_plan() -> None:
    """Solve-phase system prompt, captured-plan deliverable, all gates."""
    _check_golden(
        "solve_system_plan",
        build_solve_system_prompt(explore=False,
                                  propose_params=True,
                                  ground_samplers=True,
                                  physics_margin=True,
                                  rule_param_margin=True,
                                  necessity=True,
                                  use_journal=True))


def test_golden_solve_system_policy() -> None:
    """Solve-phase system prompt, closed-loop policy deliverable."""
    _check_golden(
        "solve_system_policy",
        build_solve_system_prompt(explore=False,
                                  policy_mode=True,
                                  rule_param_margin=True,
                                  policy_max_options=40,
                                  policy_max_repeated_failures=3,
                                  policy_max_repeated_noops=5))


def test_golden_solve_system_explore() -> None:
    """Explore-phase system prompt with the train-driven early-stop note."""
    utils.reset_config({
        "seed":
        0,
        "online_learning_early_stopping":
        True,
        "online_learning_early_stopping_require_all_attempts":
        True,
    })
    _check_golden(
        "solve_system_explore",
        build_solve_system_prompt(explore=True,
                                  physics_margin=True,
                                  rule_param_margin=True,
                                  early_stop_note=build_early_stop_note()))


def test_golden_solve_query() -> None:
    """Solve query with scoring, run records, and the capture instructions."""
    utils.reset_config({"seed": 0})
    _check_golden(
        "solve_query",
        build_solve_prompt(
            _make_task(with_evaluator=True),
            all_predicates={_AT, _ON},
            all_options={_MOVE, _WAIT},
            tool_names=["submit_plan", "run_python", "Read", "Write"],
            initial_image_section=(
                "## Initial State Image\n\nA rendering of the initial "
                "scene is at `./test_images/task000_initial_state.png`. "
                "Read it first."),
            propose_params=True,
            require_tool_validation=True,
            journal="### task 0 attempt 1\n- MoveTo dx=0.2 reached At",
            strategy="## Approach\n- move, then activate",
            attempts="### task 0 attempt 1/3\n- outcome: no capture",
        ))


def test_golden_explore_query() -> None:
    """Explore query with open questions and a belief-certified scheduled
    plan."""
    utils.reset_config({"seed": 0})
    _check_golden(
        "explore_query",
        build_solve_prompt(
            _make_task(with_evaluator=False),
            all_predicates={_AT, _ON},
            all_options={_MOVE, _WAIT},
            tool_names=["submit_plan", "run_python"],
            experiment_guidance=("The learning phase left this ranked "
                                 "ledger of open questions:\n"
                                 "1. Does Active require At? Experiment: "
                                 "MoveTo then Wait."),
            scheduled_plans=[
                "  0: MoveTo(thing0, fixture0)[0.2000, 0.0000]\n"
                "  NOTE: belief-certified; executes verbatim as a solve "
                "attempt."
            ],
            propose_params=True,
            explore_mode=True,
        ))


def test_golden_sandbox_claude_md() -> None:
    """The sandbox CLAUDE.md."""
    _check_golden("sandbox_claude_md", build_claude_md())


# -- learn -------------------------------------------------------------------


def test_golden_learn_system() -> None:
    """Fully observable learn system prompt with a physical-parameter menu."""
    physical = learn_prompts.render_physical_params_section({
        "lateral_friction": {
            "default": 0.5,
            "lo": 0.05,
            "hi": 2.0,
            "scale": "log",
            "description": "sliding friction of every body"
        }
    })
    _check_golden(
        "learn_system",
        learn_prompts.build_learn_system_prompt(
            partially_observable=False,
            residual_rule_signature="def residual_rule(state, updates, "
            "params):",
            scene_viz_hint="stage and render the scene",
            physical_params_section=physical))


def test_golden_learn_system_po_invention() -> None:
    """Partially observable learn system prompt with predicate invention."""
    _check_golden(
        "learn_system_po_invention",
        learn_prompts.build_learn_system_prompt(
            partially_observable=True,
            residual_rule_signature="def residual_rule(observation, latent, "
            "history, updates, params):",
            scene_viz_hint="stage and render the scene",
            extra_sections=[
                learn_prompts.render_predicate_invention_section(
                    "the scene workbench"),
            ],
            latent_extra_sections=[
                learn_prompts.render_predicate_latent_section(),
            ],
            workflow_extra=learn_prompts.render_predicate_workflow_extra()))


def test_golden_learn_message() -> None:
    """Learn first message with a prior model, an objective, base-sim source,
    and the predicate-invention and partial-observability additions."""
    _check_golden(
        "learn_message",
        learn_prompts.build_learn_message(
            n_trajs=3,
            n_transitions=42,
            n_demos=1,
            n_interaction=2,
            trajectory_listing="[0] demo, task 0\n[1] interaction, task 0",
            structs_ref="./reference/structs.py",
            inferred_hint="{'fixture': ['is_on']}",
            predicate_listing="  Holding(robot, thing)",
            types_digest="thing: x, y\nfixture: x, y, is_on",
            options_digest="MoveTo(thing, fixture)[dx, dy]",
            simulator_file="./simulator.py",
            objective_block=learn_prompts.render_objective_block(
                "reward = success - 0.1 x moves used"),
            prior_state_block=learn_prompts.render_prior_state_block(
                ["`./simulator.py`", "`./predicates.py`"]),
            divergence_block=learn_prompts.render_divergence_block(
                "fixture.is_on: 4 mismatches", has_prior_model=True),
            base_sim_block=learn_prompts.render_base_sim_block(
                ["./reference/base_sim/scene.py"]),
            tools_block=learn_prompts.render_tools_block(
                ["run_python", "Read", "Write", "Edit"]),
            extra_messages=[
                learn_prompts.render_predicate_invention_message(
                    "./predicates.py",
                    "Goal (natural language): switch the fixture on."),
                learn_prompts.render_partial_observability_message(),
            ]))


def test_golden_learn_program_system() -> None:
    """The program-world-model learn system prompt with predicate invention
    (paper arm C4)."""
    _check_golden(
        "learn_program_system",
        learn_prompts.build_program_learn_system_prompt(
            scene_viz_hint="stage and render the scene",
            extra_sections=[
                learn_prompts.render_predicate_invention_section(
                    "the scene workbench"),
            ],
            workflow_extra=learn_prompts.render_predicate_workflow_extra()))


def test_golden_learn_program_message() -> None:
    """The program-world-model learn first message, zero-shot variant."""
    _check_golden(
        "learn_program_message",
        learn_prompts.build_program_learn_message(
            n_trajs=0,
            n_transitions=0,
            n_demos=0,
            n_interaction=0,
            trajectory_listing="",
            structs_ref="./reference/structs.py",
            predicate_listing="- Holding(robot:robot, block:block)",
            types_digest="- robot: hand\n- block: x, y, held",
            options_digest="- Pick(robot:robot, block:block)[]",
            world_model_file="./world_model.py",
            tools_block=learn_prompts.render_tools_block(["run_python"]),
            extra_messages=[
                learn_prompts.render_program_zero_shot_message(),
            ]))


def test_golden_learn_notes_system() -> None:
    """The natural-language world-model learn system prompt (paper arm C3)."""
    _check_golden("learn_notes_system",
                  learn_prompts.build_notes_learn_system_prompt())


def test_golden_learn_notes_message() -> None:
    """The natural-language world-model learn first message with prior notes
    and a goal."""
    _check_golden(
        "learn_notes_message",
        learn_prompts.build_notes_learn_message(
            n_trajs=2,
            n_transitions=9,
            n_demos=0,
            n_interaction=2,
            trajectory_listing="  [0] interaction, task 0\n"
            "  [1] interaction, task 0",
            structs_ref="./reference/structs.py",
            predicate_listing="- Holding(robot:robot, block:block)",
            types_digest="- robot: hand\n- block: x, y, held",
            options_digest="- Pick(robot:robot, block:block)[]",
            notes_file="./world_model.md",
            goal_nls=["Build the bridge.", "Build the bridge."],
            has_prior_notes=True,
            tools_block=learn_prompts.render_tools_block(["run_python"])))
