"""What the four baseline arms read names nothing that they are not given.

The rule of every experiment arm: the agent has no access to an element
the arm is not given, and nothing it reads mentions that element. A
sentence such as "No simulator is supplied" breaks the rule, because it
names the missing thing and tells the agent that it plays a reduced arm.

For each arm and each benchmark domain, this file builds the arm from the
benchmark config, plays two scripted rounds, and collects what the agent
is sent: the system prompt with the sandbox suffix, the sandbox
``CLAUDE.md``, the skills reference, every tool's description and input
schema, the first query and the continuation query, and the tool results
of those rounds with their status lines. No pattern of the table below
may match that text.
"""
import json
import pydoc
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

from predicators.agent_sdk.base_simulator_probe import BaseSimulator
from predicators.agent_sdk.local_sandbox import _LOCAL_SANDBOX_SYSTEM_PROMPT
from predicators.agent_sdk.prompt_templates import render
from predicators.agent_sdk.tools.assembly import create_mcp_tools
from predicators.agent_sdk.tools.program_synthesis import \
    STANDALONE_NO_RECORDINGS_HINT
from predicators.agent_sdk.tools.reward_model import REPLAY_UNCHECKED_NOTE, \
    make_engine_free_evaluate_trajectory_fn
from predicators.approaches import create_approach
from predicators.envs import create_new_env
from predicators.ground_truth_models import get_gt_options
from predicators.run.continual import ContinualRun
from predicators.run.level_players import create_level_player
from predicators.structs import Dataset, Task
from scripts.cluster_utils import generate_run_configs
from tests.agent_sdk.test_prompt_goldens import _check_golden
from tests.approaches.test_agent_continual_approach import _call, _config, \
    _result
# pylint: disable-next=unused-import
from tests.approaches.test_agent_continual_real_to_sim_approach import \
    _dispose_test_physics_clients

# pylint: disable=protected-access

CONFIG = "empiric/benchmark.yaml"
DOMAINS = ("pybullet_boil", "pybullet_domino", "pybullet_fan",
           "pybullet_bridge", "pybullet_balloons")

Patterns = List[Tuple[str, str]]

# -- The forbidden patterns, each with its reason --------------------------
# Patterns are regular expressions, matched without case.

# Every arm: wording that names the experiment or a thing as missing.
_EVERY_ARM: Patterns = [
    (r"this arm", "tells the agent that it plays one arm of several"),
    (r"\b(comparison|baseline|ablation)s?\b", "names the experiment design"),
    (r"\bno\b[^.\n]{0,40}\bsuppl|not suppl", "names a thing as not supplied"),
    (r"withh[eo]ld|\bunavailable\b", "names a thing as kept from the agent"),
    (r"no simulator|simulator backs", "names a missing simulator"),
    (r"phased learning|(execution|rehearsal|model) gate",
     "names a phase or a gate that the arm does not have"),
    (r"harness fit|the harness (fits|offers)|residual-fitting|sim\.fit\b",
     "names the harness fit"),
    (r"P-hat|(joint|belief) draws|physics_sweep|suggest_probes"
     r"|sim\.(refine|belief)", "names EMPIRIC's rehearsal"),
    (r"simulator\.py|SceneBase|PARAM_SPECS|RESIDUAL_|probe_ext",
     "names EMPIRIC's model files"),
    # The Domino goal's note for an agent whose rehearsals the harness
    # scores (domino_goal_names_rehearsal). The standalone model may
    # "rehearse" a plan in its own words, so the word alone is allowed
    # there.
    (r"a rehearsal always", "names the harness's rehearsal"),
]

# Direct and Direct + scene: the task, the skills, the recordings and the
# reward model on states the agent predicts. No simulator, model or probe.
_DIRECT: Patterns = [
    (r"simulat", "the arm has no simulator of any kind"),
    (r"\bsim\b", "`sim` is the handle of the arms with a simulator"),
    (r"belief", "the arm has no belief"),
    (r"rehears", "the arm has nothing to rehearse in"),
    (r"\bfit", "the arm has no fit"),
    (r"run_python|\bprobe", "a tool and a handle that the arm does not have"),
    # "reward model" is the one model the arm has.
    (r"(?<!reward )\bmodels?\b|world_model|predicates\.py",
     "the arm has no model workflow"),
    (r"\bengine\b|wrapper|base_sim", "the scene files come with no engine"),
]

# Direct alone also has no scene files.
_NO_SCENE_FILES: Patterns = [
    (r"scene_manifest|\bURDF\b|reference/assets|## Scene files",
     "the arm has no scene files"),
]

# Standalone: the scene files, its own world model, a score of that model
# on the recordings and one prediction of a plan. No base simulator, no
# harness fit, and none of the probe's other calls.
_STANDALONE: Patterns = [
    (r"base simulator", "the arm has no base simulator"),
    # The agent's own fitting is the one legitimate use (see _ALLOWED).
    (r"\bfit(s|ted|ting)?\b", "the arm has no harness fit"),
    (r"(plan|parameter) search|repeated-trial|predicate scoring",
     "lists calls that the arm's probe does not have"),
    (r"sim\.(predicates|render|run_policy|run_async|gather|validate"
     r"|residuals)", "calls that the arm's probe does not have"),
    (r"engine-based|evaluator replay", "names the engine replay of a rule"),
]

# Direct + base simulator: the Direct + scene agent with the supplied
# simulator. No fit, belief, rehearsal or model, and no interface for
# adding a mechanism. "It contains no hidden mechanism" stays: it says
# what the simulator holds, so `mechanism` alone is not forbidden.
_BASE_SIMULATOR: Patterns = [
    (r"\bfit", "the arm has no fit"),
    (r"belief", "the arm has no belief"),
    # The plural names parameter draws. The singular stays: the noise text
    # says "one draw per step", and `sim.render` draws the current state.
    (r"\bdraws\b", "the arm has no parameter draws"),
    (r"rehears|sweep", "the arm has no rehearsal"),
    # "reward model" is the one model the arm has.
    (r"(?<!reward )\bmodels?\b|modeled dynamics|world_model|predicates\.py",
     "the arm has no model workflow"),
    (r"file of yours|no call adds|step code|adds? (a |any )?mechanism",
     "names the model interface of EMPIRIC from assets"),
]

FORBIDDEN: Dict[str, Patterns] = {
    "direct_opus": _EVERY_ARM + _DIRECT + _NO_SCENE_FILES,
    "direct_scene_opus": _EVERY_ARM + _DIRECT,
    "standalone_opus": _EVERY_ARM + _STANDALONE,
    "direct_base_sim_opus": _EVERY_ARM + _BASE_SIMULATOR,
}

# Legitimate text that a pattern above would match. Each phrase is removed
# before the patterns run, so the patterns stay strict everywhere else.
_ALLOWED = (
    # The skills reference, on the environment's own grasp.
    "the simulated grasp ignores the force limit",
    # The goal atoms of a level can be absent from the vocabulary.
    "when goal atoms are unavailable",
    # Standalone: the fitting is the agent's own code for its own model.
    "You may implement fitting and diagnostics for your own model.",
)

# The golden of each arm's system prompt on Boil, as the benchmark config
# renders it. Direct + base simulator has its golden in its own test file
# (continual_system_direct_base_sim).
_GOLDENS = {
    "direct_opus": "continual_system_direct",
    "direct_scene_opus": "continual_system_direct_scene",
    "standalone_opus": "continual_system_standalone",
}

# A heading that each arm's text must contain, so a render that lost a
# part cannot pass.
_PRESENT = {
    "direct_opus": "## Recorded data",
    "direct_scene_opus": "## Scene files",
    "standalone_opus": "## Model files",
    "direct_base_sim_opus": "## Supplied base simulator",
}


def _no_task(task_idx: Optional[int]) -> Task:
    """A task resolver for a helper that is only read, never called."""
    raise AssertionError(f"no task {task_idx}")


def _violations(arm: str, texts: Dict[str, str]) -> List[str]:
    """Every match of the arm's forbidden patterns, with where it is."""
    found = []
    for source, text in texts.items():
        flat = " ".join(text.split())
        for phrase in _ALLOWED:
            flat = flat.replace(phrase, "")
        for pattern, reason in FORBIDDEN[arm]:
            for match in re.finditer(pattern, flat, flags=re.IGNORECASE):
                start, end = match.span()
                context = flat[max(0, start - 60):end + 60]
                found.append(f"{source}: /{pattern}/ ({reason}): "
                             f"...{context}...")
    return found


def _arm_flags(arm: str, env_name: str) -> Dict[str, Any]:
    cfg = next(c for c in generate_run_configs(CONFIG, False, approaches=[arm])
               if c.env == env_name)
    # The launcher pins machine-specific output paths; tests keep their own.
    flags = {
        k: v
        for k, v in cfg.flags.items() if k not in ("log", "continual_runs_dir")
    }
    flags.update(
        approach=cfg.approach,
        env=cfg.env,
        continual_render=False,
        continual_make_video=False,
        # The scripted session ends with give_up on its training level.
        continual_refuse_training_give_up=False)
    return flags


def _make(arm: str, env_name: str, tmp_path: Any) -> Tuple[Any, Any]:
    """The environment and the arm's approach, from the benchmark config."""
    flags = _arm_flags(arm, env_name)
    _config(tmp_path, **flags)
    env = create_new_env(env_name, do_cache=False, use_gui=False)
    approach = create_approach(flags["approach"], env.predicates,
                               get_gt_options(env.get_name()), env.types,
                               env.action_space,
                               [t.task for t in env.get_train_tasks()])
    return env, approach


def _tool_text(tools: List[Any]) -> str:
    """Each tool's name, description and input schema."""
    return "\n".join(f"{tool.name}: {tool.description}\n"
                     f"{json.dumps(tool.input_schema, default=str)}"
                     for tool in tools)


def _agent_texts(arm: str, env_name: str, tmp_path: Any,
                 monkeypatch: Any) -> Dict[str, str]:
    """What the arm's agent is sent in two scripted rounds on ``env_name``:

    one zero action, then give up.
    """
    env, approach = _make(arm, env_name, tmp_path)
    texts: Dict[str, str] = {}
    results: List[str] = []

    def query(message: str, *_args: Any, **_kwargs: Any) -> Any:
        ctx = approach._tool_context
        names = approach._continual_tool_names()
        if "first query" in texts:
            texts["continuation query"] = message
            results.append(_call(approach, "give_up", note="done"))
            assert "Give-up recorded" in results[-1]
            return _result()
        texts["first query"] = message
        sandbox = Path(ctx.sandbox_dir)
        texts["CLAUDE.md"] = (sandbox / "CLAUDE.md").read_text()
        texts["reference/skills.md"] = (sandbox /
                                        "reference/skills.md").read_text()
        tools = create_mcp_tools(ctx, tool_names=names)
        assert sorted(t.name for t in tools) == sorted(names)
        texts["tools"] = _tool_text(tools)
        results.append(
            _call(approach,
                  "env_step",
                  action=[0.0] * env.action_space.shape[0]))
        results.append(_call(approach, "env_observe"))
        results.append(_call(approach, "skills_list"))
        if "evaluate_trajectory" in names:
            results.append(_call(approach, "evaluate_trajectory", states=[{}]))
        if "run_python" in names:
            results.append(
                _call(approach, "run_python", code="print(len(trajectories))"))
        return _result()

    monkeypatch.setattr(approach, "_query_agent_sync", query)
    approach.prepare_for_continual(Dataset([]))
    ContinualRun(env, approach, create_level_player(env, approach)).run()
    texts["system prompt"] = (approach._get_agent_system_prompt() +
                              _LOCAL_SANDBOX_SYSTEM_PROMPT)
    texts["tool results"] = "\n".join(results)
    return texts


@pytest.mark.parametrize("env_name", DOMAINS)
@pytest.mark.parametrize("arm", sorted(FORBIDDEN))
def test_arm_reads_nothing_about_what_it_lacks(arm: str, env_name: str,
                                               tmp_path: Any,
                                               monkeypatch: Any) -> None:
    """On each domain, no forbidden pattern of the arm matches its system
    prompt, sandbox ``CLAUDE.md``, skills reference, tool descriptions and
    schemas, queries, or the tool results of two scripted rounds."""
    texts = _agent_texts(arm, env_name, tmp_path, monkeypatch)
    assert set(texts) == {
        "system prompt", "CLAUDE.md", "reference/skills.md", "tools",
        "first query", "continuation query", "tool results"
    }
    # The render is whole: every part is there and reads as it should.
    prompt = texts["system prompt"]
    assert "## Run rules" in prompt and "## Sandbox Environment" in prompt
    assert "# Predicators Agent Sandbox" in texts["CLAUDE.md"]
    assert "## Rules" in texts["CLAUDE.md"]
    assert "# Skill API" in texts["reference/skills.md"]
    assert "env_observe: " in texts["tools"]
    assert "first conversation round" in texts["first query"]
    assert "[ledger]" in texts["continuation query"]
    assert "[ledger]" in texts["tool results"]
    assert _PRESENT[arm] in prompt + texts["first query"]
    heading = ("## Model and data"
               if arm == "standalone_opus" else "## Recorded data")
    for query in ("first query", "continuation query"):
        assert f"{heading}\n\n" in texts[query], query
    assert "Recorded episodes so far: 0 (0 steps)" in texts["first query"]
    if env_name == "pybullet_domino":
        # The goal states the rule on the push and stops there.
        assert "topples that needed the arm's body earn nothing.\n" in \
            texts["first query"]
    violations = _violations(arm, texts)
    assert not violations, "\n".join(violations)


@pytest.mark.parametrize("arm", sorted(_GOLDENS))
def test_golden_system_prompt_on_boil(arm: str, tmp_path: Any) -> None:
    """The arm's system prompt on Boil, in reviewable form."""
    _, approach = _make(arm, "pybullet_boil", tmp_path)
    prompt = approach._get_agent_system_prompt()
    _check_golden(_GOLDENS[arm], prompt)
    assert not _violations(arm, {"system prompt": prompt})


def test_patterns_catch_the_wording_they_replace() -> None:
    """Each arm's table matches the sentences this file was written against,
    and leaves the allowed phrases alone."""
    old = {
        "direct_opus": [
            "No simulator is supplied.",
            "No simulator backs it. Free.",
            "This arm has no belief model.",
            "This is enforced for the file tools, for Bash, and for "
            "`run_python`",
            "the probe's `_ctx`",
            "## Scene files",
        ],
        "direct_scene_opus": [
            "No simulator backs this tool, so a verdict is only as good",
            "the engine wrapper",
        ],
        "standalone_opus": [
            "no base simulator is supplied underneath this model",
            "there is no phased learning or execution gate",
            "The harness offers no plan search",
            "The supplied residual-fitting API",
            "engine-based evaluator replays are unavailable",
            "`sim.run` and `sim.refine` predict with it",
        ],
        "direct_base_sim_opus": [
            "It loads no file of yours",
            "no call adds forces, attachments, state or step code to it",
            "the harness offers no fit",
            "This arm has no belief model.",
        ],
    }
    for arm, sentences in old.items():
        for sentence in sentences:
            assert _violations(arm, {"old": sentence}), (arm, sentence)
    for arm in FORBIDDEN:
        for phrase in _ALLOWED:
            assert not _violations(arm, {"allowed": phrase}), (arm, phrase)
    # Legitimate text that the tables must leave alone.
    kept = {
        "direct_opus":
        "a level's reward model on a state sequence you predict",
        "standalone_opus":
        "A predicate inferred from model memory is a "
        "belief, not a measured fact.",
        "direct_base_sim_opus":
        "It contains no hidden mechanism. One draw "
        "per env step; `sim.render(label)` and the others read, draw, save "
        "and restore the current state.",
    }
    for arm, text in kept.items():
        assert not _violations(arm, {"kept": text}), (arm, text)


def test_notes_and_helper_text_name_nothing_withheld() -> None:
    """The text an agent meets later in a run keeps to the same tables: the
    note of a verdict whose replay rule was not checked, the helper's own
    documentation, the standalone model's status line and its score with
    nothing recorded, and the supplied simulator's documentation."""
    helper = make_engine_free_evaluate_trajectory_fn(_no_task)
    unchecked = {
        "note": REPLAY_UNCHECKED_NOTE,
        "helper documentation": pydoc.plain(pydoc.render_doc(helper)),
    }
    assert "that rule was not checked" in REPLAY_UNCHECKED_NOTE
    for arm in ("direct_opus", "direct_scene_opus", "standalone_opus"):
        assert not _violations(arm, unchecked), _violations(arm, unchecked)
    status = render("play_query",
                    "world_model_status",
                    world_model_version="v003",
                    n_episodes="2",
                    n_steps="40")
    assert "`sim.run` predicts with it" in status
    assert not _violations(
        "standalone_opus", {
            "status": status,
            "score with nothing recorded": STANDALONE_NO_RECORDINGS_HINT
        })
    manual = pydoc.plain(pydoc.render_doc(BaseSimulator))
    assert " run(self" in manual
    found = _violations("direct_base_sim_opus", {"sim": manual})
    assert not found, "\n".join(found)
