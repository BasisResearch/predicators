"""EMPIRIC from assets without harness fitting.

The agent writes its model as EMPIRIC from assets does and the harness
estimates nothing: the model runs at the values the agent declares. The
agent is never told that a harness fit exists, so nothing it reads names
one and its ``sim`` has no fit entry point.
"""
# pylint: disable=protected-access
import ast
import json
import pydoc
import re
import traceback
from pathlib import Path
from typing import Any, Dict, List, Sequence, cast

import pytest

from predicators import utils
from predicators.agent_sdk import play_prompts
from predicators.agent_sdk.belief_probe import BeliefProbe, \
    DeclaredValuesProbe, build_probe_namespace
from predicators.agent_sdk.local_sandbox import _LOCAL_SANDBOX_SYSTEM_PROMPT
from predicators.agent_sdk.sandbox_prompts import build_claude_md
from predicators.agent_sdk.sandbox_setup import GeneratedReference
from predicators.agent_sdk.synthesis_backend import SynthesisBackend
from predicators.agent_sdk.tools import ToolContext
from predicators.agent_sdk.tools.continual_tools import CONTINUAL_TOOL_NAMES
from predicators.agent_sdk.tools.exploration import ProbeSurface, \
    belief_probe_blurb
from predicators.agent_sdk.tools.sandbox_guard import _scrub_host_paths, \
    agent_traceback
from predicators.agent_sdk.tools.synthesis import create_synthesis_tools
from predicators.approaches import _get_approach_cls_from_name
# pylint: disable-next=line-too-long
from predicators.approaches.agent_continual_from_assets_no_fitting_approach import \
    AgentContinualFromAssetsNoFittingApproach, reference_wording
from predicators.run.continual import ContinualRun
from predicators.run.level_players import create_level_player
from predicators.structs import Dataset
from tests.agent_sdk.test_model_validation import _Workbench
from tests.agent_sdk.test_prompt_goldens import _check_golden
from tests.agent_sdk.test_rollout_residuals import _NOOP_SIMULATOR
from tests.approaches.test_agent_continual_approach import _call, _result
from tests.approaches.test_agent_continual_no_learning_approach import \
    _arm_flags, _make
# pylint: disable-next=unused-import
from tests.approaches.test_agent_continual_real_to_sim_approach import \
    _dispose_test_physics_clients
from tests.test_cluster_utils_configs import _benchmark_runs

# The menu key and the approach's name are in the run folder's path, which
# the agent reads as its working directory, so neither names a fit.
ARM = "from_assets_declared_opus"
NAME = "agent_continual_from_assets_declared"
# EMPIRIC from assets' switches that act on a harness fit.
DROPPED = ("code_sim_learning_prior_spans_bounds", "belief_material_lines",
           "continual_fit_reach_check", "continual_fit_bound_note",
           "belief_segment_temperature")

# What names a harness fit. "fit" and its forms count wherever no letter
# touches them, so a dot, an underscore, a bracket or a backtick beside the
# word does not hide it: `sim.fit`, `fit_params`, `fit()`, `PARTIAL FIT`,
# `UNFITTED`, `refit`, `misfit`. A word that only holds the letters
# (benefit, profit, fitness) does not count. The sweep arguments of the fit
# count by name, and so does the fit's other name, system identification
# (system-ID, sysID).
FIT_WORDING = re.compile(
    r"(?<![A-Za-z])(?:re|un|mis)?fit(?:s|ted|ting|ter|ters)?(?![A-Za-z])"
    r"|sweep_params|sweep_num_points"
    r"|(?<![A-Za-z])sys(?:tem)?[ -]?id(?:entification)?(?![A-Za-z])",
    re.IGNORECASE)
# The reference sources say "fit" of three things that are not a harness
# fit, and the copies keep them: the module that defines ParamSpec, and
# the scene base giving a body the size its observation shows.
_NOT_A_HARNESS_FIT = ("predicators.code_sim_learning.fit_space",
                      "_fit_body_sizes", "unfit", "no single stop fits them")
# A model for the Boil scene the base loads: a mechanism with a declared
# rate, a declared material, model memory, and no scene construction.
MODEL_SOURCE = '''
class BoilModel(SceneBase):
    AGENT_PARAM_SPECS = [ParamSpec("heat_rate", 0.01, lo=0.005, hi=0.02),
                         ParamSpec("jug_spinning_friction", 0.005, lo=0.001,
                                   hi=0.01)]
    RESIDUAL_FEATURES = {"jug": ["water_volume"]}
    MODEL_STATE_INIT = {"ticks": 0}

    @classmethod
    def update_model_state(cls, observation, model_state, params, action):
        model_state["ticks"] += 1

    def _domain_specific_step(self):
        for obj in self._objects:
            if obj.type.name == "jug":
                key = (obj.name, "water_volume")
                self._feature_store[key] += self.agent_param("heat_rate")


RESIDUAL_ENV = BoilModel
'''


def _fit_wording(text: str, allowed: Sequence[str] = ()) -> List[str]:
    """Each passage of ``text`` that names a harness fit, with its
    surroundings."""
    for phrase in allowed:
        text = text.replace(phrase, "")
    return [
        text[max(0,
                 m.start() - 60):m.end() + 60].replace("\n", " ")
        for m in FIT_WORDING.finditer(text)
    ]


def _code(source: str) -> str:
    """The code of a module without its docstrings (a parse drops the
    comments)."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)) and \
                ast.get_docstring(node) is not None:
            node.body = node.body[1:] or [ast.Pass()]
    return ast.dump(tree)


def test_the_check_finds_fit_wording() -> None:
    """The regex counts every form the harness's text uses and leaves words
    that only hold the letters."""
    for text in ("call `sim.fit()` first", "fit_params=False", "UNFITTED for",
                 "PARTIAL FIT (v2)", "then refit.", "to fit a misfit",
                 "the fitted values", "fitting is disabled", "the fitter's",
                 "sweep_params='all'", "rollout system-ID", "the sysID",
                 "system identification", "_fit_body_sizes", "(fit box lo"):
        assert _fit_wording(text), text
    for text in ("a benefit of fitness", "the system identifies it",
                 "an outfit", "profit"):
        assert not _fit_wording(text), text


def test_config_is_from_assets_without_a_harness_fit() -> None:
    """Parked, five settings and five seeds, the new class, and EMPIRIC from
    assets' flags less the switches that act on a harness fit."""
    assert all(run.approach != NAME for run in _benchmark_runs())
    runs = _benchmark_runs(approaches=[ARM])
    assert len(runs) == 25
    full = {(run.env, run.seed): run.flags
            for run in _benchmark_runs(approaches=["from_assets_opus"])}
    for run in runs:
        assert run.approach == NAME
        assert _get_approach_cls_from_name(run.approach) is \
            AgentContinualFromAssetsNoFittingApproach
        kept = {
            name: value
            for name, value in full[(run.env, run.seed)].items()
            if name not in DROPPED
        }
        assert set(DROPPED) <= set(full[(run.env, run.seed)])
        assert run.flags == {
            **kept,
            "agent_sim_learn_declared_params_only": True,
            "agent_sim_learn_fit_invisible": True,
        }
        assert run.flags["scene_manifest_loader"] is True
        assert run.flags["continual_rehearsal_gate"] is True


def test_arm_needs_its_switches(tmp_path: Any) -> None:
    """The class runs only where nothing names a fit, without a harness fit,
    and with the joint draws its rehearsals run on."""
    with pytest.raises(ValueError, match="agent_sim_learn_fit_invisible"):
        _make(tmp_path, ARM, agent_sim_learn_fit_invisible=False)
    with pytest.raises(ValueError, match="declared_params_only"):
        _make(tmp_path, ARM, agent_sim_learn_declared_params_only=False)
    with pytest.raises(ValueError, match="belief_joint_draws"):
        _make(tmp_path, ARM, belief_joint_draws=0)
    with pytest.raises(ValueError, match="prior_spans_bounds"):
        _make(tmp_path, ARM, code_sim_learning_prior_spans_bounds=True)
    # The agent reads the run folder's path, so a round may name no fit.
    with pytest.raises(ValueError, match="run folder's path"):
        _make(tmp_path, ARM, experiment_id="boil-declared_opus_no_fit_r1")
    assert not _fit_wording(NAME + " " + ARM)


def _golden_flags() -> Dict[str, Any]:
    flags = {
        k: v
        for k, v in _arm_flags(ARM).items()
        if k.startswith(("continual_", "belief_", "agent_", "scene_",
                         "code_sim_"))
    }
    flags.pop("continual_render")
    flags.pop("continual_make_video")
    return flags


def test_golden_system_prompt() -> None:
    """The arm's system prompt, as the reviewable golden; no passage names a
    fit, and the other arms' goldens keep the sections it replaced."""
    utils.reset_config(_golden_flags())
    tools = ["run_python"] + list(CONTINUAL_TOOL_NAMES)
    contract = play_prompts.build_model_contract(partially_observable=True,
                                                 declared_params_only=True,
                                                 scene_built=True)
    text = play_prompts.build_play_system_prompt(
        tools,
        model_contract=contract,
        scene_built=True,
        base_sim_refs=[
            "./reference/base_sim/pybullet_env.py",
            "./reference/base_sim/base_env.py",
            "./reference/base_sim/scene_base.py",
            "./reference/scene/scene_manifest.json (9 bodies)",
            "./reference/assets/ (4 URDF and mesh files, named in the "
            "manifest)",
        ])
    _check_golden("continual_system_from_assets_no_fitting", text)
    assert not _fit_wording(text), _fit_wording(text)
    assert "__" not in text.replace("__init__", "")
    utils.reset_config({})


@pytest.mark.parametrize("loader", [True, False])
def test_prompt_and_descriptions_name_no_fit(tmp_path: Any,
                                             loader: bool) -> None:
    """The system prompt, the probe descriptions and the sandbox notes keep
    EMPIRIC from assets' modelling rules, say what the model runs at, and name
    no fit; with the scene loaded or built."""
    _, approach = _make(tmp_path, ARM, scene_manifest_loader=loader)
    assert isinstance(approach, AgentContinualFromAssetsNoFittingApproach)
    surface = approach._probe_surface()
    assert surface == ProbeSurface(fit=False, fit_invisible=True)
    prompt = approach._get_agent_system_prompt()
    texts = {
        "system prompt":
        prompt + _LOCAL_SANDBOX_SYSTEM_PROMPT,
        "probe description":
        belief_probe_blurb(synthesis_probe=True, surface=surface),
        "solve probe description":
        belief_probe_blurb(synthesis_probe=False, surface=surface),
        "CLAUDE.md":
        build_claude_md(),
        "no-model status":
        approach._model_status(cast(Any, None)),
        "probe class":
        pydoc.render_doc(DeclaredValuesProbe),
    }
    for name, text in texts.items():
        assert not _fit_wording(text), (name, _fit_wording(text))
    # EMPIRIC from assets' statement and modelling rules.
    assert "## EMPIRIC from assets" in prompt
    assert "SceneBase" in prompt and "scene_manifest.json" in prompt
    for rule in ("force, torque or constraint it applies",
                 "Give each parameter to the part that owns it",
                 "declare a parameter that blends the forms",
                 "whose state a force depends on",
                 "make your first step on a new level",
                 "A step can lose a level without any failure event"):
        assert rule in prompt, rule
    assert ("The base loads the scene" in prompt) == loader
    # What is true of the declarations, in positive words.
    assert "The model runs at the values its declarations give" in prompt
    assert "### Declared values and ranges" in prompt
    assert "bell curve centred on its `init_value`" in prompt
    assert "standard deviation is 0.75 times the `init_value`" in prompt
    assert "Test levels require a loaded model" in prompt
    assert "### Each rehearsed step is checked against its rehearsal" in prompt
    assert "The draws follow your declarations" in prompt
    # Nothing is described as withheld, disabled or refused.
    for absent in ("harness fits nothing", "harness estimates nothing",
                   "is disabled", "are disabled", "PARAM_DRAWS", "No harness"):
        assert absent not in prompt, absent
    assert "declared values and ranges" in texts["no-model status"]
    assert "at the declared values" in texts["probe description"]
    for call in ("`sim.validate(traj_idxs=None, params=None)`",
                 "phys_params={name: value} scores ONE",
                 "`sim.residuals(rollout=True, phys_params=None)`",
                 "physics_sweep=True", "sim.suggest_probes"):
        assert call in texts["probe description"], call


def test_other_arms_descriptions_are_unchanged() -> None:
    """The switch changes this arm's text alone: the learning arm and the
    paper's no-fitting arm keep their wording."""
    utils.reset_config({"belief_joint_draws": 16})
    full = belief_probe_blurb(synthesis_probe=True)
    assert "`sim.fit(" in full and "sweep_params" in full
    assert "at the fit's point estimate" in full
    no_fit = belief_probe_blurb(synthesis_probe=True,
                                surface=ProbeSurface(fit=False))
    assert "the harness fits nothing" in no_fit
    assert "at the fit's point estimate" in no_fit
    utils.reset_config({})


def test_probe_has_no_fit_entry_point(tmp_path: Any) -> None:
    """The arm's ``sim`` is a probe without ``fit``: the attribute is absent
    from the object and its class, and the fit's arguments are unknown
    keywords; the other arms' probe keeps them."""
    utils.reset_config({"agent_sim_learn_fit_invisible": True})
    ctx = ToolContext()
    namespace = build_probe_namespace(ctx)
    probe_cls = type(namespace["sim"])
    assert probe_cls is DeclaredValuesProbe
    assert probe_cls.__name__ == "BeliefProbe"
    assert BeliefProbe not in probe_cls.__mro__
    assert namespace["BeliefProbe"]().__class__ is probe_cls
    sim: Any = namespace["sim"]
    assert not hasattr(sim, "fit") and "fit" not in dir(sim)
    assert not hasattr(type(sim), "fit")
    with pytest.raises(AttributeError,
                       match="'BeliefProbe' object has no attribute 'fit'"):
        sim.fit()
    for argument in ("fit_params", "sweep_params", "sweep_num_points"):
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            # pylint: disable-next=unexpected-keyword-arg
            sim.residuals(**{argument: True})
    for kept in ("run", "refine", "validate", "residuals", "belief",
                 "suggest_probes", "predicates", "reset", "render"):
        assert hasattr(sim, kept) and kept in dir(sim), kept
    # Its replay reports say what they ran at.
    artifact = tmp_path / "simulator.py"
    artifact.write_text(_NOOP_SIMULATOR, encoding="utf-8")
    toolkit = create_synthesis_tools(exec_ns={},
                                     base_pred_triples=[],
                                     inferred_residual_features={},
                                     simulator_file=str(artifact),
                                     versions_dir=str(tmp_path / "versions"),
                                     approach=cast(SynthesisBackend,
                                                   cast(Any, _Workbench)()),
                                     probe_surface=ProbeSurface(
                                         fit=False, fit_invisible=True))
    ctx.probe_validation_provider = toolkit.validation_runner
    ctx.probe_residuals_provider = toolkit.residuals_runner
    reports = {
        "validate":
        sim.validate(),
        "validate at a point":
        sim.validate(traj_idxs=[1], params={"friction": 2.0}),
        "residuals":
        sim.residuals(rollout=True),
        "residuals at a point":
        sim.residuals(rollout=True, phys_params={"friction": 1.0}),
        "run_python":
        toolkit.tools[0].description,
    }
    for name, text in reports.items():
        assert not _fit_wording(text), (name, _fit_wording(text))
    assert "Model: declared values (" in reports["validate"]
    assert "trajectory 1: 6 actions" in reports["validate"]
    assert "Pass phys_params={name: value}" in reports["residuals"]
    assert "strong evidence FOR the overridden value(s)" in \
        reports["residuals at a point"]
    # A traceback shown to this arm's agent keeps the agent's frames and
    # the error, and leaves out the harness's own.
    shown = full = ""
    try:
        sim.restore(99)
    except ValueError:
        shown = agent_traceback()
        full = traceback.format_exc()
    assert "belief_probe.py" in full and "in restore" in full
    assert "belief_probe.py" not in shown and "in restore" not in shown
    assert "test_agent_continual_from_assets_no_fitting_approach.py" in shown
    assert shown.rstrip().endswith(full.rstrip().splitlines()[-1])
    # The other arms' probe and tracebacks are the ones they had.
    utils.reset_config({})
    try:
        sim.restore(99)
    except ValueError:
        assert agent_traceback() == _scrub_host_paths(traceback.format_exc())
    other: Any = build_probe_namespace(ToolContext())["sim"]
    assert type(other).__mro__[0] is BeliefProbe
    assert hasattr(other, "fit") and "fit" in dir(other)
    with pytest.raises(RuntimeError, match="sim.residuals is unavailable"):
        # pylint: disable-next=unexpected-keyword-arg
        other.residuals(fit_params=True, sweep_params="all")


def test_reference_copies_name_no_fit(tmp_path: Any) -> None:
    """The engine sources this arm's sandbox gets carry the same code as every
    arm's and name no harness fit; a passage the source no longer has is
    refused."""
    _, approach = _make(tmp_path, ARM)
    files = approach._get_sandbox_reference_files()
    _, full_approach = _make(tmp_path / "full", "from_assets_opus")
    full = full_approach._get_sandbox_reference_files()
    assert set(files) == set(full)
    for name in ("base_sim/pybullet_env.py", "base_sim/base_env.py",
                 "base_sim/scene_base.py"):
        copy = files[name]
        assert isinstance(copy, GeneratedReference), name
        source = full[name]
        original = (source.text if isinstance(source, GeneratedReference) else
                    Path(source).read_text(encoding="utf-8"))
        assert _fit_wording(original, _NOT_A_HARNESS_FIT), name
        assert not _fit_wording(copy.text, _NOT_A_HARNESS_FIT), \
            (name, _fit_wording(copy.text, _NOT_A_HARNESS_FIT))
        # Only comments and docstrings differ: the code is the same.
        assert copy.text != original
        assert _code(copy.text) == _code(original), name
    for name, source in files.items():
        if isinstance(source, GeneratedReference) and \
                name.endswith((".json", ".py", ".urdf")):
            assert not _fit_wording(source.text, _NOT_A_HARNESS_FIT), name
    with pytest.raises(ValueError, match="occurs 0 times"):
        reference_wording("base_sim/base_env.py",
                          GeneratedReference("pass\n", "a changed source"))


def test_agent_models_and_rehearses_at_its_declared_values(
        tmp_path: Any, monkeypatch: Any) -> None:
    """End to end on Boil under the scene loader: nothing the agent is sent or
    reads back names a fit, ``sim`` has no ``fit``, a model with a mechanism
    and declared parameters loads, a rehearsal runs the declared values and
    draws each parameter inside its declared range, and the test-level gate
    opens for the loaded model."""
    env, approach = _make(tmp_path,
                          ARM,
                          continual_refuse_training_give_up=False)
    seen: List[str] = []
    read: Dict[str, str] = {}

    def python(code: str) -> str:
        out = _call(approach, "run_python", code=code)
        read[f"run_python {len(read)}"] = out
        return out

    def first_round(message: str) -> None:
        ctx = approach._tool_context
        sandbox = Path(ctx.sandbox_dir)
        assert "set its declared values and ranges" in message
        # What the session offers: every tool's description and schema.
        for tool in ctx.extra_mcp_tools:
            read[f"tool {tool.name}"] = (str(tool.description) + "\n" +
                                         json.dumps(tool.input_schema))
        assert {"run_python", "skills_invoke", "claims_record"} <= \
            {tool.name for tool in ctx.extra_mcp_tools}
        # Where the sandbox is: the agent reads its working directory.
        folder = sandbox.relative_to(tmp_path)
        assert NAME in str(folder) and not _fit_wording(str(folder)), folder
        # What the sandbox holds.
        log_dir = Path(approach._get_log_dir())
        read["sent system prompt"] = next(
            log_dir.glob("full_system_prompt_*.md")).read_text()
        read["CLAUDE.md"] = (sandbox / "CLAUDE.md").read_text()
        for path in sorted((sandbox / "reference").rglob("*.py")):
            leaks = _fit_wording(path.read_text(), _NOT_A_HARNESS_FIT)
            assert not leaks, (path.name, leaks)
        # The probe has no fit, and no argument of one.
        absent = python("print(hasattr(sim, 'fit'), 'fit' in dir(sim), "
                        "hasattr(BeliefProbe(), 'fit'), type(sim).__name__)")
        assert "False False False BeliefProbe" in absent, absent
        refused = _call(approach, "run_python", code="sim.fit()")
        assert "AttributeError" in refused, refused
        assert "'BeliefProbe' object has no attribute 'fit'" in refused
        for argument in ("fit_params", "sweep_params", "sweep_num_points"):
            refused = _call(approach,
                            "run_python",
                            code=f"sim.residuals({argument}=True)")
            assert "TypeError" in refused, refused
            assert f"unexpected keyword argument '{argument}'" in refused
        # Before a model the probe has no world.
        python("sim.reset(current=True)\n"
               "sim.run('Wait(robot:robot)[]', draws=0)")
        # A recording, then the model.
        stepped = _call(approach,
                        "skills_invoke",
                        skill="Wait(robot:robot)[2]")
        read["skills_invoke"] = stepped
        (sandbox / "simulator.py").write_text(MODEL_SOURCE, encoding="utf-8")
        loaded = python("sim.reset(current=True)\nprint(sorted(sim.state()))")
        assert not loaded.startswith("ERROR"), loaded
        assert "jug0:jug" in loaded
        assert type(approach._base_env).__name__ == "BoilModel"
        assert approach._fit_status_text() == (
            "the model runs at its declared values: heat_rate=0.01 in "
            "[0.005, 0.02], jug_spinning_friction=0.005 in [0.001, 0.01]")
        # A rollout runs the declared rate: two more steps add two times
        # the declared 0.01.
        rate = python(
            "def rise(steps):\n"
            "    out = sim.reset(current=True).run("
            "f'Wait(robot:robot)[{steps}]', draws=0, render=False)\n"
            "    jug = [o for o in out.states[0] if o.name == 'jug0'][0]\n"
            "    print(out)\n"
            "    return (out.states[-1].get(jug, 'water_volume') - "
            "out.states[0].get(jug, 'water_volume'))\n"
            "print('RATE', round((rise(4) - rise(2)) / 2, 6))")
        assert "RATE 0.01\n" in rate + "\n", rate
        # A rehearsal's draws vary each declared parameter inside its
        # declared range, and no other parameter.
        rehearsed = python(
            "r = sim.reset(current=True).run('Wait(robot:robot)[2]')\n"
            "print(r)\n"
            "names = sorted({n for d in r.draws for n in d['params']})\n"
            "rate = [d['params']['heat_rate'] for d in r.draws]\n"
            "spin = [d['params']['jug_spinning_friction'] for d in r.draws]\n"
            "print('NAMES', names, len(r.draws))\n"
            "print('RATE', 0.005 <= min(rate) < max(rate) <= 0.02)\n"
            "print('SPIN', 0.001 <= min(spin) < max(spin) <= 0.01)")
        assert not rehearsed.startswith("ERROR"), rehearsed
        assert "NAMES ['heat_rate', 'jug_spinning_friction'] 16" in rehearsed
        assert "RATE True" in rehearsed and "SPIN True" in rehearsed
        assert "at the declared values (the values the planning model " \
            "runs at)" in rehearsed, rehearsed
        belief = approach.parameter_belief()
        assert belief.map_estimate == {
            "heat_rate": 0.01,
            "jug_spinning_friction": 0.005
        }
        # A physics sweep names its points by what the agent declared.
        sweep = python("print(sim.reset(current=True).run("
                       "'Wait(robot:robot)[2]', physics_sweep=True))")
        assert "  declared values: goal NOT reached" in sweep, sweep
        assert "INSIDE the declared-parameter uncertainty range" in sweep
        # The replay reports say what they ran at.
        replay = python("print(sim.validate())")
        assert "Model: declared values (" in replay, replay
        assert '"heat_rate": 0.01' in replay, replay
        # The other calls the description offers.
        for code in ("print(sim.residuals())",
                     "print(sim.residuals(rollout=True, "
                     "phys_params={'heat_rate': 0.015}))",
                     "print(sim.reset(current=True).belief())",
                     "print(sim.predicates())",
                     "print(sim.reset(current=True).check_restore())"):
            python(code)
        # An error inside the harness shows the agent's line and the
        # error, with no frame of the harness.
        failed = python("sim.restore(99)")
        assert "Unknown snapshot id 99" in failed, failed
        assert "belief_probe.py" not in failed and "predicators/" not in \
            failed, failed
        # A loaded model opens the test-level gate.
        assert approach._model_readiness(str(sandbox / "simulator.py"),
                                         []) is None

    def query(message: str, *_args: Any, **_kwargs: Any) -> Any:
        seen.append(message)
        read[f"query {len(seen)}"] = message
        sandbox = Path(approach._tool_context.sandbox_dir)
        if len(seen) == 1:
            first_round(message)
            return _result()
        # The round that follows reports the model the last one deployed.
        assert "Parameters: the model runs at its declared values: " \
            "heat_rate=0.01 in [0.005, 0.02]" in message, message
        read["attempts.md"] = (sandbox / "attempts.md").read_text()
        assert "Model updated: simulator.py" in read["attempts.md"]
        assert "Give-up recorded" in _call(approach, "give_up", note="done")
        return _result()

    monkeypatch.setattr(approach, "_query_agent_sync", query)
    approach.prepare_for_continual(Dataset([]))
    card = ContinualRun(env, approach, create_level_player(env,
                                                           approach)).run()
    assert card.end_note == "done"
    assert len(seen) == 2
    for name, text in read.items():
        assert not _fit_wording(text), (name, _fit_wording(text))
