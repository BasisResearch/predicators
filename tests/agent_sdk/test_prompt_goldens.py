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

import pytest

from predicators import utils
from predicators.agent_sdk import play_prompts
from predicators.agent_sdk.prompt_templates import _PROMPTS_DIR, \
    load_sections, placeholders, render
from predicators.agent_sdk.sandbox_prompts import build_claude_md
from predicators.agent_sdk.tools.continual_tools import CONTINUAL_TOOL_NAMES
from predicators.approaches.agent_continual_frozen_approach import \
    AgentContinualOracleDynamicsApproach

_GOLDEN_DIR = os.path.join(os.path.dirname(__file__), "prompt_goldens")

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
        if not diff:
            # A line diff cannot show an editor's added final newline.
            diff = ("(no line differs: the texts differ only in line "
                    "endings or a trailing newline)")
        pytest.fail(f"prompt {name} drifted from its golden; review the diff "
                    f"and rerun with UPDATE_PROMPT_GOLDENS=1 to accept:\n"
                    f"{diff}")


# -- template hygiene --------------------------------------------------------


def test_templates_use_plain_dashes_and_declared_sections() -> None:
    """Templates contain no em dashes, and every section is named once.

    A file without section markers is a reference document handed to the
    agent verbatim (``public_skills.md``), not a template; it still
    keeps to plain dashes.
    """
    paths = sorted(glob.glob(os.path.join(_PROMPTS_DIR, "*.md")))
    assert paths
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        assert "—" not in text, path
        if "<!-- section:" not in text:
            continue
        name = os.path.splitext(os.path.basename(path))[0]
        assert load_sections(name)


def test_render_rejects_missing_and_unused_placeholders() -> None:
    """A template placeholder without a value, or a value without a
    placeholder, fails loudly instead of shipping literal text."""
    with pytest.raises(AssertionError):
        render("subclass_model", "physical_params")
    with pytest.raises(AssertionError):
        render("subclass_model", "physical_params", param_list="p", extra="x")
    assert placeholders("a __B__ c __B__ __D__") == ["B", "D"]


def test_rendered_prompts_have_no_leftover_placeholders() -> None:
    """No rendered prompt carries an unsubstituted ``__NAME__``."""
    marker = re.compile(r"__[A-Z][A-Z0-9_]*__")
    contract = play_prompts.build_model_contract(partially_observable=True)
    tools = ["run_python"] + list(CONTINUAL_TOOL_NAMES)
    for text in (play_prompts.build_play_system_prompt(
            tools, model_contract=contract), build_claude_md()):
        assert not marker.search(text), marker.search(text)


def test_golden_sandbox_claude_md() -> None:
    """The sandbox CLAUDE.md."""
    _check_golden("sandbox_claude_md", build_claude_md())


# -- learn -------------------------------------------------------------------


@pytest.mark.parametrize("model_based,noise,repair", [
    (True, False, False),
    (True, True, True),
    (False, False, False),
    (False, True, True),
])
def test_golden_continual_system(model_based, noise, repair):
    """Review the whole prompt, including gated modeling and noise guidance."""
    utils.reset_config({
        "continual_obs_noise_position": 0.01 if noise else 0.0,
        "continual_obs_noise_orientation": 0.02 if noise else 0.0,
        "agent_model_repair": repair,
    })
    tools = list(CONTINUAL_TOOL_NAMES)
    contract = ""
    if model_based:
        tools = ["run_python"] + tools
        contract = play_prompts.build_model_contract(partially_observable=True)
    text = play_prompts.build_play_system_prompt(tools,
                                                 model_contract=contract)
    arm = "mb" if model_based else "mf"
    variant = "noisy_repair" if noise else "exact"
    _check_golden(f"continual_system_{arm}_{variant}", text)
    assert "__" not in text.replace("__init__", "")
    if model_based:
        assert "AGENT_PARAM_SPECS" in text
        assert not re.search(r"(?<!AGENT_)\bPARAM_SPECS\b", text)
        assert "always false on real observations" not in text
        assert "a usable veto" not in text
        assert "trials>=2, solved=True" in text and "contacts=True" in text
    else:
        assert "simulator.py" not in text and "sim.fit" not in text
    utils.reset_config({})


def test_golden_continual_system_joint_belief():
    """EMPIRIC's prompt as the experiments run it (16 joint draws): one
    rehearsal over the joint draws replaces the trial, solved and belief-draw
    flags, which the prompt no longer offers."""
    utils.reset_config({
        "continual_obs_noise_position": 0.01,
        "continual_obs_noise_orientation": 0.02,
        "agent_model_repair": True,
        "belief_joint_draws": 16,
    })
    tools = ["run_python"] + list(CONTINUAL_TOOL_NAMES)
    contract = play_prompts.build_model_contract(partially_observable=True)
    text = play_prompts.build_play_system_prompt(tools,
                                                 model_contract=contract)
    _check_golden("continual_system_mb_joint", text)
    assert "__" not in text.replace("__init__", "")
    assert render("play_system", "final_rehearsal_joint") in text
    assert "scores up to 8 proposals on the joint draws" in text
    for legacy in ("trials>=2", "solved=True", "belief_draws"):
        assert legacy not in text
    utils.reset_config({})


@pytest.mark.parametrize("kind", ["first", "level", "continue", "resumed"])
def test_golden_continual_query(kind):
    """Each round kind keeps live state separate from persistent
    instructions."""
    text = play_prompts.build_play_query(
        kind=kind,
        round_number=2,
        level_number=1,
        levels_total=3,
        goal_nl="Move the widget to the fixture.",
        goal_atoms=[],
        ledger="[ledger] 12 steps; no resets",
        context="[context] 4k tokens",
        observation="[episode] NOT_FINISHED\n[objects] widget: x=0.2",
        skills="Move(widget, fixture)[offset]",
        predicates="(none)",
        types="widget: x; fixture: x",
        model="No fit yet; 1 recorded episode.",
        journal="The first move fell short.\nTry a larger offset.",
        attempts="Round 1: 12 steps.")
    _check_golden(f"continual_query_{kind}", text)
    assert text.count("[ledger]") == text.count("[context]") == 1
    assert "[episode] NOT_FINISHED" in text


def test_golden_continual_system_scene_package():
    """The model arm with the real-to-sim arm's references lists them as the
    twin's scene package, not as a scene to build."""
    utils.reset_config({
        "continual_obs_noise_position": 0.01,
        "continual_obs_noise_orientation": 0.02,
        "agent_model_repair": True,
    })
    tools = ["run_python"] + list(CONTINUAL_TOOL_NAMES)
    contract = play_prompts.build_model_contract(partially_observable=True)
    refs = [
        "./reference/base_sim/pybullet_fan_base.py",
        "./reference/base_sim/pybullet_env.py",
        "./reference/base_sim/base_env.py",
        "./reference/scene/scene_manifest.json (9 bodies)",
        "./reference/assets/ (4 URDF and mesh files, named in the manifest)",
    ]
    text = play_prompts.build_play_system_prompt(tools,
                                                 base_sim_refs=refs,
                                                 model_contract=contract,
                                                 scene_package=True)
    _check_golden("continual_system_scene_package", text)
    assert "__" not in text.replace("__init__", "")
    assert "scene_manifest.json" in text and "URDF and mesh files" in text
    assert "already builds this scene" in text
    assert "SceneBase" not in text
    assert "sim.fit" in text


@pytest.mark.parametrize("preflight", [False, True])
def test_model_gate_does_not_imply_preflight(preflight):
    """The rendered test-model gate must not promise disabled rehearsal."""
    utils.reset_config({
        "continual_require_model_on_test": True,
        "continual_skill_preflight": preflight,
    })
    text = play_prompts.build_play_system_prompt(["run_python"] +
                                                 list(CONTINUAL_TOOL_NAMES))
    assert "every skill request is rehearsed in it before it runs" not in text
    assert ("### Every skill request is rehearsed first" in text) == preflight


@pytest.mark.parametrize("arm", [
    "scene_only", "oracle_dynamics", "zero_shot", "no_fitting",
    "no_uncertainty", "real_to_sim"
])
def test_golden_continual_system_ablation(arm):
    """Each comparison arm's prompt describes only what that arm can do."""
    flags = {
        "continual_obs_noise_position": 0.01,
        "continual_obs_noise_orientation": 0.02,
        "agent_model_repair": True,
    }
    if arm in ("no_fitting", "real_to_sim"):
        flags["agent_sim_learn_declared_params_only"] = True
    if arm in ("no_uncertainty", "real_to_sim"):
        flags.update({
            "continual_uncertainty_decisions": False,
            "agent_sim_learn_param_uncertainty": False,
            "agent_explorer_info_seeking": False,
            "agent_explorer_info_seeking_adaptive": False,
        })
    if arm == "no_uncertainty":
        flags["continual_obs_noise_declared"] = False
    # As the experiments run them: every arm but No uncertainty carries
    # the joint belief (scripts/configs/empiric/common.yaml and
    # approaches.yaml).
    flags["belief_joint_draws"] = 0 if arm == "no_uncertainty" else 16
    utils.reset_config(flags)
    tools = ["run_python"] + list(CONTINUAL_TOOL_NAMES)
    frozen = arm in ("scene_only", "oracle_dynamics", "zero_shot")
    scene_built = arm == "real_to_sim"
    contract = play_prompts.build_model_contract(
        partially_observable=True,
        declared_params_only=arm in ("no_fitting", "real_to_sim"),
        frozen=frozen,
        supplied_model=arm in ("scene_only", "oracle_dynamics"),
        scene_built=scene_built)
    options = {}
    if frozen:
        options = {
            "frozen_section": render("play_frozen", arm),
            "frozen_model_supplied": arm != "zero_shot",
            "oracle_dynamics": arm == "oracle_dynamics",
        }
    if scene_built:
        options = {"scene_built": True}
        tools_refs = [
            "./reference/base_sim/pybullet_env.py",
            "./reference/base_sim/scene_base.py",
            "./reference/scene/scene_manifest.json (9 bodies)",
        ]
        options["base_sim_refs"] = tools_refs
    text = play_prompts.build_play_system_prompt(tools,
                                                 model_contract=contract,
                                                 **options)
    _check_golden(f"continual_system_{arm}", text)
    assert "__" not in text.replace("__init__", "")
    if arm in ("scene_only", "oracle_dynamics", "zero_shot"):
        assert "sim.fit" not in text
        assert "When the model disagrees" not in text
        assert "earns its keep" not in text
    if arm == "oracle_dynamics":
        oracle = object.__new__(AgentContinualOracleDynamicsApproach)
        # pylint: disable=protected-access
        assert oracle._play_prompt_options() == options
        shared = render("play_system",
                        "rehearsal_reliability",
                        state_rehearsal=render("play_system",
                                               "state_rehearsal_joint"))
        assert shared in text
        assert shared in play_prompts.build_play_system_prompt(tools)
        restriction = ("do not fit, edit, or substitute a hand-built "
                       "dynamics model")
        assert restriction in text
        assert render("play_system", "robustness_oracle_joint") in text
        assert "checks replay verdicts across" not in text
    elif frozen or arm == "no_uncertainty":
        heading = "### State estimates, timing, and execution discrepancies"
        assert heading not in text
    if scene_built:
        # sim.fit is named once, as disabled; it is never offered.
        assert "sim.fit()" not in text and "call sim.fit" not in text
        assert "BaseSimulator" not in text
        assert "visible base physics" not in text
        assert "visible physics" not in text
        assert "SceneBase" in text and "scene_manifest.json" in text
        assert "declared parameter values" in text
        assert "physical parameter menu" not in text
    if arm in ("scene_only", "oracle_dynamics"):
        assert "Model API reference" not in text
        assert "Predicate API reference" in text
        assert "supplied dynamics model" in text
    else:
        assert "Model API reference" in text
    if arm == "no_fitting":
        assert "No harness parameter fitting" in text
        assert "Do not implement an optimizer" not in text
        assert "have a fitted" not in text
    if arm == "no_uncertainty":
        assert "No explicit uncertainty handling" in text
        assert "latest observation as the current state" in text
        assert "do not average, smooth, or filter" in text
        for word in ("Observation noise", "sigma", "noisy", "denoise"):
            assert word not in text, word
        assert "Average only when" not in text
        assert "state smoothing" not in text
        assert "tests physical-parameter uncertainty" not in text
        assert "margin across models" not in text
        assert "what uncertainty could change" not in text
    utils.reset_config({})
