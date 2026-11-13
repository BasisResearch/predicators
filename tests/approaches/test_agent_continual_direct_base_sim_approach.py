"""Direct + base simulator: the direct agent with the supplied base simulator
in ``run_python``."""
import inspect
import json
import re
from pathlib import Path
from typing import Any, Callable, Dict, List

import pytest

from predicators.agent_sdk.base_simulator_probe import BaseSimulator, \
    BaseSimulatorProbe, base_simulator_description
from predicators.agent_sdk.belief_probe import _ProbeCore
from predicators.agent_sdk.play_prompts import build_play_system_prompt
from predicators.agent_sdk.probe_results import ProbeResult
from predicators.agent_sdk.rollout_probe import _RolloutCore
from predicators.agent_sdk.tools.context import ToolContext
from predicators.agent_sdk.tools.continual_tools import CONTINUAL_TOOL_NAMES, \
    REWARD_MODEL_TOOL_NAMES
from predicators.approaches import create_approach
from predicators.approaches.agent_continual_direct_base_sim_approach import \
    AgentContinualDirectBaseSimApproach
from predicators.code_sim_learning.scene_base import SceneBase
from predicators.envs import create_new_env
from predicators.ground_truth_models import get_gt_options
from predicators.run.continual import ContinualRun
from predicators.run.level_players import create_level_player
from predicators.settings import CFG
from predicators.structs import Dataset, State
from scripts.cluster_utils import generate_run_configs
from tests.agent_sdk.test_prompt_goldens import _check_golden
from tests.approaches.test_agent_continual_approach import _call, _config, \
    _result
# pylint: disable-next=unused-import
from tests.approaches.test_agent_continual_real_to_sim_approach import \
    SCENE_SOURCE, _dispose_test_physics_clients

# pylint: disable=protected-access

CONFIG = "empiric/benchmark.yaml"
ARM = "direct_base_sim_opus"
DOMAINS = ("pybullet_boil", "pybullet_domino", "pybullet_fan",
           "pybullet_bridge", "pybullet_balloons")
# What ``sim`` offers in this arm, and nothing else.
SIM_CALLS = {
    "reset", "run", "run_async", "gather", "state", "atoms", "render",
    "snapshot", "restore", "drop", "clear_snapshots", "params"
}
# The probe core's calls and rollout modes, which the probe behind this
# arm's ``sim`` does not have: the reports on a model file and the calls
# that load one, the belief and its draws, plan search and policies.
PROBE_CORE_ONLY = ("belief", "run_points", "run_policy", "suggest_probes",
                   "refine", "score", "validate", "predicates",
                   "check_restore", "task", "_run_joint_draws",
                   "_run_rehearsal", "_run_belief_draws")
# The features the engine reads from a body; every other feature of an
# object is kept from the state a rollout starts from.
ENGINE_FEATURES = {
    "x", "y", "z", "rot", "yaw", "roll", "pitch", "is_held", "r", "g", "b"
}
# Wording of the EMPIRIC workflow, which this arm's prompt and tool text
# must not carry.
WORKFLOW_WORDS = ("simulator.py", "predicates.py", "probe_ext.py", "sim.fit",
                  "sim.refine", "sim.belief", "sim.validate", "sim.residuals",
                  "RESIDUAL", "PARAM_SPECS", "SceneBase", "rehears", "P-hat",
                  "belief", "draws", "sweep", "Test levels require",
                  "modeled dynamics")


def _arm_flags(env_name: str, arm: str = ARM) -> Dict[str, Any]:
    cfg = next(c for c in generate_run_configs(CONFIG, False, approaches=[arm])
               if c.env == env_name)
    # The launcher pins machine-specific output paths; tests keep their own.
    flags = {
        k: v
        for k, v in cfg.flags.items() if k not in ("log", "continual_runs_dir")
    }
    flags.update(approach=cfg.approach,
                 env=cfg.env,
                 continual_render=False,
                 continual_make_video=False)
    return flags


def _make(tmp_path: Any,
          env_name: str = "pybullet_boil",
          **overrides: Any) -> Any:
    flags = {**_arm_flags(env_name), **overrides}
    _config(tmp_path, **flags)
    env = create_new_env(env_name, do_cache=False, use_gui=False)
    approach = create_approach(flags["approach"], env.predicates,
                               get_gt_options(env.get_name()), env.types,
                               env.action_space,
                               [t.task for t in env.get_train_tasks()])
    assert isinstance(approach, AgentContinualDirectBaseSimApproach)
    return env, approach


def _play(env: Any, approach: Any, monkeypatch: Any,
          script: Callable[[str], None]) -> Any:
    """Play one scripted round that ends with give_up; the run's card."""
    seen: List[str] = []

    def query(message: str, *_args: Any, **_kwargs: Any) -> Any:
        seen.append(message)
        script(message)
        assert "Give-up recorded" in _call(approach, "give_up", note="done")
        return _result()

    monkeypatch.setattr(approach, "_query_agent_sync", query)
    approach.prepare_for_continual(Dataset([]))
    card = ContinualRun(env, approach, create_level_player(env,
                                                           approach)).run()
    assert card.end_note == "done"
    assert len(seen) == 1
    return card


def _python(approach: Any, code: str) -> Any:
    """Run ``code`` in the arm's run_python; the JSON its last line prints."""
    out = _call(approach, "run_python", code="import json\n" + code)
    assert "Traceback" not in out and "blocked" not in out, out
    return json.loads(out.strip().splitlines()[-1])


def _sections(prompt: str) -> Dict[str, str]:
    """A system prompt's ``##`` sections by heading; its opening is ""."""
    sections = {}
    for index, part in enumerate(prompt.split("\n\n## ")):
        heading, _, body = part.partition("\n")
        sections["" if index == 0 else heading] = body if index else part
    return sections


def _assert_no_mechanism(scene: Any) -> None:
    """The supplied world is ``SceneBase`` bound to the deployment: the one
    method its class adds reports the values it is held at, and the bound class
    adds only the gripper's reading conventions."""
    supplied, bound, base = type(scene).__mro__[:3]
    assert base is SceneBase
    for cls, methods in ((supplied, ["get_physical_param_info"]), (bound, [
            "_fingers_joint_to_state", "_fingers_state_to_joint"
    ])):
        assert sorted(name for name, value in vars(cls).items()
                      if callable(value) or isinstance(value, (
                          classmethod, property))) == methods, cls
    assert "_domain_specific_step" not in vars(SceneBase)
    assert type(scene).AGENT_PARAM_SPECS == []
    assert not scene._agent_param_values
    assert not scene._sampled_materials


def test_config_is_direct_plus_scene_with_the_loaded_scene() -> None:
    """Five settings, five seeds, parked, and the flags of Direct + scene plus
    the scene loader: no model gate, no harness fit, no rehearsal."""
    assert all(run.approach != "agent_continual_direct_base_sim"
               for run in generate_run_configs(CONFIG, False))
    runs = list(generate_run_configs(CONFIG, False, approaches=[ARM]))
    scene = {
        run.env: run
        for run in generate_run_configs(
            CONFIG, False, approaches=["mf_scene_package_opus"])
    }
    assert len(runs) == 25
    assert {run.env for run in runs} == set(DOMAINS)
    for run in runs:
        assert run.approach == "agent_continual_direct_base_sim"
        assert run.flags == {
            **scene[run.env].flags,
            "scene_manifest_loader": True,
            "scene_observed_sizes": True,
            "continual_provide_skills_reference": False,
            "continual_provide_level_scenes": True,
            "continual_provide_simulator_source": True,
        }
        assert run.flags["continual_provide_scene_package"] is True
        assert run.flags["continual_provide_reward_model"] is True
        assert run.flags["continual_uncertainty_decisions"] is False
        for absent in ("continual_require_model_on_test",
                       "continual_rehearsal_gate",
                       "continual_practise_after_reset",
                       "continual_claim_check", "continual_predictive_check",
                       "agent_sim_learn_declared_params_only",
                       "code_sim_learning_prior_spans_bounds"):
            assert absent not in run.flags, absent


def test_arm_needs_the_scene_files_and_the_loader(tmp_path: Any) -> None:
    """Without the scene loader the harness has no scene to load."""
    with pytest.raises(ValueError, match="scene_manifest_loader"):
        _make(tmp_path, scene_manifest_loader=False)


def test_prompt_is_the_direct_prompt_with_the_simulator_tool(
        tmp_path: Any) -> None:
    """The system prompt has the sections of Direct + scene and one more, on
    the supplied simulator; neither it nor the tool text asks for a model, a
    fit or a rehearsal."""
    _, approach = _make(tmp_path)
    names = approach._continual_tool_names()
    assert names == ["run_python"] + CONTINUAL_TOOL_NAMES
    assert approach._get_solve_tool_names() == names
    assert approach._option_model is None
    prompt = approach._get_agent_system_prompt()
    mine = _sections(prompt)
    direct = _sections(
        build_play_system_prompt(CONTINUAL_TOOL_NAMES +
                                 REWARD_MODEL_TOOL_NAMES,
                                 base_sim_refs=approach._scene_asset_paths(),
                                 scene_package=True))
    extra = "Supplied base simulator"
    assert [h for h in mine if h != extra] == list(direct)
    assert list(mine).index(extra) == list(mine).index("Working files") + 1
    # The sections that only ask it to solve the levels are the direct
    # agent's, word for word.
    for same in ("", "Run rules", "Reading observations", "Observation noise",
                 "Decision workflow", "Run memory", "Scene files"):
        assert mine[same] == direct[same], same
    tools = [
        line for line in mine["Tools"].splitlines()
        if "`run_python`" not in line
    ]
    assert tools == [
        line for line in direct["Tools"].splitlines()
        if "`evaluate_trajectory`" not in line
    ]
    assert mine["Working files"] == direct["Working files"] + (
        " `run_python` runs code beside the supplied base simulator, which "
        "the section below describes.")
    assert ("- `run_python`: code in the sandbox with `sim`, the supplied "
            "base simulator") in prompt
    for present in ("`sim.reset(current=True)`", "`sim.run(plan_text)`",
                    "`sim.params()`", "phys_params={name: value}",
                    "It contains no hidden mechanism.",
                    "A rollout is set by its start state, its plan, its "
                    "physical values and its planner seed.",
                    "the engine's defaults where no file sets one",
                    "`evaluate_trajectory(states, actions=None",
                    "the replay runs in the supplied base simulator",
                    "## Scene files", "scene_manifest.json",
                    "`./reference/scene/levels/level_NN.json (one file per "
                    "level played so far: that level's scene manifest and "
                    "the exact features each observed object with a static "
                    "base had as the level began)`",
                    "The source of this simulator is in these read-only "
                    "files, for reference:\n\n"
                    "- `./reference/base_sim/scene_base.py`\n"
                    "- `./reference/base_sim/pybullet_env.py`\n"
                    "- `./reference/base_sim/base_env.py`"):
        assert present in prompt, present
    # The source is listed for reference, and nothing asks to build on it.
    for absent in ("subclass", "extend", "override", "add a mechanism",
                   "mechanisms", "your model"):
        assert absent not in prompt.lower(), absent
    for unnamed in ("No simulator", "file of yours", "no call adds"):
        assert unnamed not in prompt, unnamed
        assert unnamed not in base_simulator_description(reward_model=True)
    _check_golden("continual_system_direct_base_sim", prompt)
    description = base_simulator_description(reward_model=True)
    for text in (prompt, description):
        for absent in WORKFLOW_WORDS:
            assert absent not in text, absent
        assert not re.search(r"\bfit", text), re.search(r"\bfit\w*", text)
        assert "\u2014" not in text
    assert "evaluate_trajectory" not in base_simulator_description(
        reward_model=False)
    # The arm's probe lacks the other arms' calls by its class: the
    # session withholds nothing from it.
    assert not approach._tool_context.probe_absent
    assert not approach._tool_context.probe_disabled


def test_probe_has_no_belief_or_rehearsal_code() -> None:
    """The arm's probe is the rollout core alone: the probe core's belief,
    draws, rehearsal, plan search and model reports are not attributes of it or
    of the ``sim`` over it."""
    assert issubclass(BaseSimulatorProbe, _RolloutCore)
    assert not issubclass(BaseSimulatorProbe, _ProbeCore)
    probe = BaseSimulatorProbe(ToolContext())
    for holder in (BaseSimulatorProbe, BaseSimulator, probe,
                   BaseSimulator(probe)):
        for name in PROBE_CORE_ONLY:
            assert not hasattr(holder, name), (holder, name)
    # The names are the probe core's, so the check cannot pass on a typo.
    for name in PROBE_CORE_ONLY:
        assert hasattr(_ProbeCore, name), name
    # One rollout per call: ``run`` takes no trial, sweep or draw mode.
    assert list(inspect.signature(BaseSimulatorProbe.run).parameters) == [
        "self", "plan_text", "render", "contacts", "seed", "phys_params"
    ]


def test_rollout_notes_name_no_model(monkeypatch: Any) -> None:
    """A rollout note of the rollout core that advises a model change comes
    back without that advice."""
    advice = "or revise the modeled dynamics"
    assert advice in inspect.getsource(_RolloutCore._run_once)
    note = ("step 0 (Wait) ran to its step cap (120 actions). Inspect the "
            "final state and subgoal report to decide whether to wait "
            f"longer {advice}.")

    def run(_self: Any, _plan: str, **_kwargs: Any) -> ProbeResult:
        return ProbeResult([], False, [], State({}), [note])

    monkeypatch.setattr(_RolloutCore, "run", run)
    result = BaseSimulatorProbe(ToolContext()).run("Wait(robot:robot)[]")
    assert result.notes == [note.replace(f" {advice}", "")]
    assert result.notes[0].endswith("whether to wait longer.")


def test_run_python_drives_the_supplied_simulator(tmp_path: Any,
                                                  monkeypatch: Any) -> None:
    """In a session on Boil, ``sim`` starts from the observation and from a
    recorded state, runs a skill and returns its states, and runs a rollout at
    other physical values; it has no fit and no way to add a mechanism."""
    env, approach = _make(tmp_path, continual_refuse_training_give_up=False)

    def script(message: str) -> None:
        assert "## Recorded data\n\nRecorded episodes so far: 0" in message
        assert "This arm" not in message and "## Model and data" not in message
        sandbox = Path(approach._tool_context.sandbox_dir)
        refs = {
            str(q.relative_to(sandbox / "reference"))
            for q in (sandbox / "reference").rglob("*") if q.is_file()
        }
        # The files of Direct + scene with the scene of each level played
        # and the simulator's source, and no skill reference file; the
        # simulator loads its own copy of the scene files.
        assert {
            "scene/scene_manifest.json", "scene/levels/level_01.json",
            "assets/urdf/jug-pixel.urdf", "base_sim/scene_base.py",
            "base_sim/pybullet_env.py", "base_sim/base_env.py"
        } <= refs
        assert "skills.md" not in refs
        copy = Path(approach._scene_copy_dir())
        assert sandbox not in copy.parents
        assert not (copy / "base_sim").exists()
        for scene_file in ("scene/levels/level_01.json",
                           "scene/scene_manifest.json"):
            assert (copy / scene_file).read_text() == \
                (sandbox / "reference" / scene_file).read_text()
        # The source copies are the agent's to read: the simulator is built
        # from the harness's own modules, so a sandbox copy that cannot
        # even load changes nothing below.
        for source in ("scene_base.py", "pybullet_env.py", "base_env.py"):
            path = sandbox / "reference/base_sim" / source
            path.chmod(0o644)
            path.write_text("raise RuntimeError('a sandbox copy loaded')\n")
        assert sandbox not in Path(inspect.getfile(SceneBase)).parents
        tools = [t.name for t in approach._tool_context.extra_mcp_tools]
        assert sorted(tools) == sorted(approach._continual_tool_names())
        described = next(t for t in approach._tool_context.extra_mcp_tools
                         if t.name == "run_python").description
        assert described == base_simulator_description(reward_model=True)

        # The calls ``sim`` has, and the ones it does not.
        assert set(
            _python(
                approach, "print(json.dumps([n for n in dir(sim) "
                "if not n.startswith('_')]))")) == SIM_CALLS
        absent = sorted(
            {name
             for name in PROBE_CORE_ONLY if not name.startswith("_")}
            | {"fit", "residuals"})
        assert _python(
            approach, "print(json.dumps([hasattr(sim, n) for n in "
            f"{absent!r}]))") == [False] * len(absent)
        # Its own documentation names its calls and no other arm's.
        manual = _python(
            approach, "import pydoc\n"
            "print(json.dumps(pydoc.plain(pydoc.render_doc(sim))))")
        for call in SIM_CALLS:
            assert f" {call}(self" in manual, call
        for word in WORKFLOW_WORDS + tuple(f" {call}(" for call in absent):
            assert word not in manual, word
        assert not re.search(r"\bfit", manual)
        refused = _call(approach, "run_python", code="sim.fit()")
        assert "AttributeError" in refused and "'fit'" in refused, refused
        refused = _call(approach,
                        "run_python",
                        code="sim.run('Wait(robot:robot)[1]', trials=3)")
        assert "unexpected keyword argument 'trials'" in refused, refused
        hidden = _call(approach, "run_python", code="print(sim._ctx.env)")
        assert "sandbox guard blocked" in hidden, hidden

        # A reset to the observation, a skill, and the states it predicts.
        start = _python(
            approach, "sim.reset(current=True)\n"
            "print(json.dumps(sim.state('jug0')))")
        assert start["is_held"] == 0.0
        picked = _python(
            approach, "r = sim.run('PickJug(robot:robot, jug0:jug)[0.0]', "
            "render=False)\n"
            "print(json.dumps({'report': str(r), 'states': len(r.states), "
            "'labels': len(r.labels), 'lines': len(r.line_states), "
            "'label': list(r.labels[0][:2]), "
            "'jug': sim.state('jug0')}))")
        assert "Step 0: PickJug" in picked["report"], picked["report"]
        assert "FAILURE" not in picked["report"], picked["report"]
        assert picked["states"] == picked["labels"] + 1 > 2
        assert picked["lines"] == 1
        assert picked["label"] == ["PickJug", ["robot", "jug0"]]
        assert picked["jug"]["is_held"] == 1.0
        # No mechanism fills, heats or switches anything.
        for feature in ("water_volume", "bubbling_level"):
            assert picked["jug"][feature] == start[feature]

        # A render of the supplied world lands in the sandbox.
        image = _python(approach, "print(json.dumps(sim.render('picked')))")
        assert image.endswith(".png") and (sandbox / image).is_file(), image

        # A rollout at another physical value differs, and the simulator
        # is back at the loaded values after it.
        values = _python(
            approach, "print(json.dumps({k: v['value'] "
            "for k, v in sim.params().items()}))")
        assert values["jug_mass"] == pytest.approx(1.0)
        assert values["jug_linear_damping"] == pytest.approx(0.04)
        assert values["support_lateral_friction"] == pytest.approx(1.0)
        drops = _python(
            approach, "def drop(**kwargs):\n"
            "    sim.reset(current=True)\n"
            "    z = sim.state('jug0')['z']\n"
            "    sim.reset(current=True, mods={'jug0': {'z': z + 0.3}})\n"
            "    r = sim.run('Wait(robot:robot)[3]', render=False, **kwargs)\n"
            "    return [z + 0.3 - sim.state('jug0')['z'], r.notes]\n"
            "slow = {'jug_linear_damping': 0.9}\n"
            "print(json.dumps([drop(), drop(phys_params=slow), drop(), "
            "sim.params()['jug_linear_damping']['value']]))")
        (free, notes), (damped, damped_notes), (again, _), loaded = drops
        assert free > 0.01, drops
        assert damped < free - 1e-3, drops
        assert again == pytest.approx(free, abs=1e-6)
        assert loaded == pytest.approx(0.04)
        assert not any("PARAMS" in note for note in notes)
        assert any("PARAMS set for this call: jug_linear_damping=0.9" in note
                   for note in damped_notes), damped_notes
        unknown = _call(approach,
                        "run_python",
                        code="sim.run('Wait(robot:robot)[1]', render=False, "
                        "phys_params={'heat_rate': 0.1})")
        assert "Unknown physical values ['heat_rate']" in unknown, unknown
        negative = _call(approach,
                         "run_python",
                         code="sim.run('Wait(robot:robot)[1]', render=False, "
                         "phys_params={'jug_mass': 0.0})")
        assert "a mass must be positive" in negative, negative

        # A real skill runs with no model and no rehearsal (a training
        # level), and its recording is a start state for the simulator.
        waited = _call(approach, "skills_invoke", skill="Wait(robot:robot)[2]")
        assert not waited.startswith("ERROR"), waited
        recorded = _python(
            approach, "t = trajectories[-1]\n"
            "sim.reset(state=t.states[1])\n"
            "x = sim.state('jug0')['x']\n"
            "r = sim.run('Wait(robot:robot)[1]', render=False)\n"
            "print(json.dumps([len(t.states), bool(x == t.states[1].get("
            "[o for o in t.states[1] if o.name == 'jug0'][0], 'x')), "
            "len(r.states)]))")
        assert recorded == [3, True, 2]
        # After a real skill, the observation the simulator starts from is
        # the new one.
        robot_before = _python(
            approach, "sim.reset(current=True)\n"
            "print(json.dumps(sim.state('robot')))")
        moved = _call(approach,
                      "skills_invoke",
                      skill="Place(robot:robot)[0.75, 1.35, 0.55, 0.0]")
        assert not moved.startswith("ERROR"), moved
        robot_after = _python(
            approach, "sim.reset(current=True)\n"
            "print(json.dumps(sim.state('robot')))")
        frame = approach._play_session.observe().frame
        robot = next(o for o in frame if o.name == "robot")
        assert robot_after == {
            f: float(frame.get(robot, f))
            for f in robot.type.feature_names
        }
        assert robot_after != robot_before

        # A file of the agent's is never loaded: a mechanism in
        # ./simulator.py changes nothing.
        (sandbox / "simulator.py").write_text(SCENE_SOURCE, encoding="utf-8")
        water = _python(
            approach, "sim.reset(current=True)\n"
            "before = sim.state('jug0')['water_volume']\n"
            "sim.run('Wait(robot:robot)[3]', render=False)\n"
            "print(json.dumps([before, "
            "sim.state('jug0')['water_volume']]))")
        assert water[0] == water[1]
        scene = approach._base_simulator.sim_env
        _assert_no_mechanism(scene)
        assert approach._tool_context.env is scene

        # A world held at other values reports them, so a copy that a
        # reward model builds to replay physics starts at them.
        scene.apply_physical_param_overrides({"jug_mass": 3.0})
        held = {
            name: entry["default"]
            for name, entry in scene.get_physical_param_info().items()
        }
        assert held["jug_mass"] == 3.0
        twin = type(scene)(use_gui=False)
        twin.apply_physical_param_overrides(held)
        twin._set_state(frame)
        assert twin._read_material(twin.body("jug0"),
                                   "mass") == pytest.approx(3.0)
        scene.apply_physical_param_overrides({"jug_mass": values["jug_mass"]})
        assert scene.get_physical_param_info()["jug_mass"]["default"] == \
            pytest.approx(values["jug_mass"])
        assert scene._read_material(scene.body("jug0"),
                                    "mass") == pytest.approx(1.0)

        # Rollouts in forked children, at their own physical values.
        forked = _python(
            approach, "sim.reset(current=True)\n"
            "hs = [sim.run_async('Wait(robot:robot)[2]', phys_params=p) "
            "for p in (None, {'jug_mass': 2.0})]\n"
            "sim.gather(hs)\n"
            "print(json.dumps([[h.ok, h.error, len(h.result.states)] "
            "for h in hs]))")
        assert forked == [[True, None, 3], [True, None, 3]]

    card = _play(env, approach, monkeypatch, script)
    assert approach._tool_context.probe_option_model_provider is None
    level = card.levels[0]
    assert level.steps > 2 and "fits" not in level.sandbox
    assert level.sandbox["sim_rollouts"] > 0


def test_evaluate_trajectory_scores_predicted_states(tmp_path: Any,
                                                     monkeypatch: Any) -> None:
    """The reward model in ``run_python`` scores the states a rollout of the
    supplied simulator predicts, and states built from them."""
    env, approach = _make(tmp_path, continual_refuse_training_give_up=False)
    assert "evaluate_trajectory" not in approach._continual_tool_names()

    def script(_message: str) -> None:
        rollout = _python(
            approach, "sim.reset(current=True)\n"
            "r = sim.run('Wait(robot:robot)[2]', render=False)\n"
            "v = evaluate_trajectory(r.states, r.labels)\n"
            "print(json.dumps([sorted(v), float(v['reward']), "
            "bool(v['solved']), v['note']]))")
        assert rollout == [["note", "reward", "solved"], 0.0, False, ""]
        # The goal of the simple Boil level as feature values: a full,
        # boiled jug, the burner off and nothing spilled.
        capacity = float(env.max_jug_water_capacity)
        solved = _python(
            approach, "boiled = {}\n"
            "for o in r.final_state:\n"
            "    f = o.type.feature_names\n"
            "    if o.type.name == 'jug':\n"
            f"        boiled[o.name] = {{'water_volume': {capacity!r}}}\n"
            "        for n in ('heat_level', 'bubbling_level'):\n"
            "            if n in f:\n"
            "                boiled[o.name][n] = 1.0\n"
            "    elif o.type.name == 'burner':\n"
            "        boiled[o.name] = {'is_on': 0.0}\n"
            "    elif o.type.name == 'faucet':\n"
            "        boiled[o.name] = {'is_on': 0.0, 'spilled_level': 0.0}\n"
            "goal = with_features(r.final_state, boiled)\n"
            "v = evaluate_trajectory(r.states + [goal], "
            "r.labels + [None], task_idx=0)\n"
            "print(json.dumps([float(v['reward']), bool(v['solved'])]))")
        assert solved == [1.0, True]
        # A recorded episode leads a rollout: recorded actions and a
        # rollout's labels in one sequence.
        assert not _call(approach,
                         "skills_invoke",
                         skill="Wait(robot:robot)[1]").startswith("ERROR")
        mixed = _python(
            approach, "t = trajectories[-1]\n"
            "sim.reset(current=True)\n"
            "r = sim.run('Wait(robot:robot)[1]', render=False)\n"
            "v = evaluate_trajectory(t.states[:-1] + r.states, "
            "list(t.actions) + r.labels)\n"
            "print(json.dumps([len(t.actions), bool(v['solved'])]))")
        assert mixed == [1, False]

    _play(env, approach, monkeypatch, script)


@pytest.mark.parametrize("env_name", DOMAINS)
def test_supplied_simulator_loads_each_benchmark_scene(
        env_name: str, tmp_path: Any, monkeypatch: Any) -> None:
    """With no agent file, the simulator starts from the observation of each
    benchmark domain and runs a skill; a feature the engine does not read from
    a body keeps the value the observation gives it."""
    env, approach = _make(tmp_path,
                          env_name,
                          continual_refuse_training_give_up=False)

    def script(_message: str) -> None:
        out = _python(
            approach, "sim.reset(current=True)\n"
            "before = sim.state()\n"
            "r = sim.run('Wait(robot:robot)[2]', render=False)\n"
            "print(json.dumps({'before': before, 'after': sim.state(), "
            "'report': str(r), 'states': len(r.states), "
            "'params': sorted(sim.params())}))")
        assert "FAILURE" not in out["report"], out["report"]
        assert out["states"] == 3
        assert set(out["before"]) == set(out["after"])
        kept = 0
        for name, features in out["before"].items():
            if name.endswith(":robot"):
                continue
            for feature, value in features.items():
                if feature in ENGINE_FEATURES:
                    continue
                kept += 1
                assert out["after"][name][feature] == value, (name, feature)
        scene = approach._base_simulator.sim_env
        _assert_no_mechanism(scene)
        assert scene._loaded_scene is not None
        assert not scene._loaded_scene.notes
        bodies = [b for b in scene.bodies.values() if isinstance(b, int)]
        assert bodies
        assert "support_lateral_friction" in out["params"]
        # The level's reward model scores the rollout in every domain.
        verdict = _python(
            approach, "v = evaluate_trajectory(r.states, r.labels)\n"
            "print(json.dumps([sorted(v), bool(v['solved'])]))")
        assert verdict == [["note", "reward", "solved"], False]
        assert any(name.endswith("_mass") for name in out["params"])
        # Every domain but Domino observes features that no pose carries.
        assert (kept > 0) == (env_name != "pybullet_domino")
        if env_name != "pybullet_domino":
            return
        # Domino's reward model replays the push: it does so in a copy of
        # the supplied simulator, at the loaded values or at named ones.
        goal = approach._tool_context.current_task.goal
        targets = sorted({o.name for atom in goal for o in atom.objects})
        notes = _python(
            approach, "sim.reset(current=True)\n"
            "r = sim.run('Push(robot:robot, domino_0:domino)[0.02, 0.1]', "
            "render=False)\n"
            f"down = with_features(r.final_state, {{n: {{'roll': 1.5708}} "
            f"for n in {targets!r}}})\n"
            "print(json.dumps([evaluate_trajectory(r.states + [down], "
            "r.labels + [None], phys_params=p)['note'] for p in "
            "(None, {'domino_lateral_friction': 1.2})]))")
        for note in notes:
            assert "replayed with the real Push skill" in note, notes

    _play(env, approach, monkeypatch, script)


def test_skill_requests_wait_for_nothing_on_a_test_level(
        tmp_path: Any, monkeypatch: Any) -> None:
    """On a test level a real skill runs with no model file, no fit and no
    rehearsal, and the simulator holds the test level's scene."""
    env, approach = _make(tmp_path, continual_levels="test_only")
    assert not CFG.continual_require_model_on_test

    def script(message: str) -> None:
        assert "(test task 0, no resets)" in message
        ctx = approach._tool_context
        for gate in ("skill_gate", "rehearsal_gate", "practise_gate",
                     "skill_preflight", "claim_check", "material_screen",
                     "before_real_action", "execution_audit"):
            assert getattr(ctx, gate) is None, gate
        sandbox = Path(ctx.sandbox_dir)
        assert not (sandbox / "simulator.py").exists()
        ran = _call(approach, "skills_invoke", skill="Wait(robot:robot)[2]")
        assert not ran.startswith("ERROR"), ran
        assert "Nothing was charged" not in ran and "rehears" not in ran
        planned = _call(approach,
                        "skills_execute_plan",
                        plan="Wait(robot:robot)[1]")
        assert not planned.startswith("ERROR"), planned
        # The test level has two jugs, and the simulator loads both.
        jugs = _python(
            approach, "sim.reset(current=True)\n"
            "sim.run('Wait(robot:robot)[1]', render=False)\n"
            "print(json.dumps(sorted(n for n in sim.state() "
            "if n.endswith(':jug'))))")
        assert jugs == ["jug0:jug", "jug1:jug"]

    card = _play(env, approach, monkeypatch, script)
    level = card.levels[0]
    assert level.split == "test" and level.steps == 3
    assert level.sandbox["sim_rollouts"] == 1
