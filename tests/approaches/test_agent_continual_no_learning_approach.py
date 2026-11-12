"""No learning (the supplied scene simulator with EMPIRIC from assets' planning
workflow), and the standalone model's scene assets."""
import json
import re
from pathlib import Path
from typing import Any, Callable, Dict, List

import pytest

from predicators.agent_sdk.local_sandbox import _LOCAL_SANDBOX_SYSTEM_PROMPT
from predicators.agent_sdk.sandbox_prompts import build_claude_md
from predicators.agent_sdk.scene_simulator import \
    SCENE_SIMULATOR_PROBE_ABSENT, arm_notes, scene_simulator_description
from predicators.agent_sdk.scene_simulator_prompts import \
    build_scene_simulator_claude_md, scene_manifest_legend
from predicators.approaches import create_approach
from predicators.approaches.agent_continual_no_learning_approach import \
    AgentContinualNoLearningApproach, supplied_scene_source
from predicators.code_sim_learning.scene_base import SceneBase
from predicators.code_sim_learning.scene_manifest import LEGEND
from predicators.envs import create_new_env
from predicators.ground_truth_models import get_gt_options
from predicators.run.continual import ContinualRun
from predicators.run.level_players import create_level_player
from predicators.structs import Dataset
from scripts.cluster_utils import generate_run_configs
from tests.agent_sdk.test_prompt_goldens import _check_golden
from tests.approaches.test_agent_continual_approach import _call, _config, \
    _result
# pylint: disable-next=unused-import
from tests.approaches.test_agent_continual_real_to_sim_approach import \
    SCENE_SOURCE, _dispose_test_physics_clients

# pylint: disable=protected-access

CONFIG = "empiric/benchmark.yaml"
ARM = "from_assets_scene_opus"
DOMAINS = ("pybullet_boil", "pybullet_domino", "pybullet_fan",
           "pybullet_bridge", "pybullet_balloons")

# A model of the agent's own. An asset file may use the word for the 3D
# model its mesh draws (an .obj group name), so _withheld leaves this one
# pattern out for the files under reference/assets/.
MODEL = r"(?<!reward )\bmodel"
# What the agent of No learning must not read, each with the reason. The
# arm has a simulator it cannot change, so nothing it reads may name what
# could be changed, added, fitted or learned, or the arm's place among
# the others. Matched without regard to case.
FORBIDDEN = [
    (r"mechanism", "none can be added, and none is named"),
    # Scoped by SKILL_PARAMETER_PHRASES below.
    (r"parameter", "the simulator has none to set or estimate"),
    (r"\bfit|refit", "nothing is fitted"),
    (r"learn", "nothing is learned"),
    (r"simulator\.py|predicates\.py", "the agent writes no model file"),
    (r"SceneBase|base_sim", "the simulator's source is not the agent's"),
    (r"_domain_specific_step", "the hook a mechanism would use"),
    (r"ablation|comparison|baseline", "the arm's place among the others"),
    (r"this arm", "talk of arms says there are others"),
    (r"no[- _]learning|frozen|sealed", "names for what the arm lacks"),
    (r"phys_params|physics_sweep|sim\.params", "physical values are not set"),
    (r"AGENT_PARAM_SPECS|ParamSpec|PARAM_|RESIDUAL",
     "names of a model file's contract"),
    (r"sim\.(?:fit|residuals|validate|score|predicates|suggest_probes)\b",
     "calls the arm does not have"),
    (r"declared (?:value|range)", "nothing is declared"),
    # "Reward model" is the level's scoring rule, which every arm has.
    (MODEL, "the agent owns no model"),
    ("\u2014", "plain dashes only"),
]
# A skill's parameters are the agent's to choose, and the shared skill
# grammar, the skill tools and this arm's refine call name them. These
# phrases are taken out before `parameter` is matched; any other use of
# the word fails.
SKILL_PARAMETER_PHRASES = (
    "parameter meanings",  # skills_list and the grammar: what a skill takes
    "exact continuous parameters",  # the grammar: values in a plan line
    "a skill with no parameters",  # the grammar: `[]`
    "optional integer parameter",  # the grammar: Wait's step count
    "skill parameters",  # sim.refine searches them
    # CLAUDE.md, on a recorded action's label.
    "the grounded skill and parameters",
    # A skill controller's diagnostic on the values a plan line gave it.
    "this option's parameters",
    "adjust the parameters",
)
# What ``sim`` offers in this arm, and nothing else.
SIM_CALLS = {
    "reset", "task", "run", "refine", "belief", "run_async", "gather", "state",
    "atoms", "render", "snapshot", "restore", "drop", "clear_snapshots"
}
ABSENT_CALLS = sorted(SCENE_SIMULATOR_PROBE_ABSENT
                      | {"fit", "residuals", "params"})
# The features the engine reads from a body; every other feature of an
# object keeps the value of the state a rollout starts from.
ENGINE_FEATURES = {
    "x", "y", "z", "rot", "yaw", "roll", "pitch", "is_held", "r", "g", "b"
}

# One skill per domain that acts on the scene, with the objects it takes
# and its values (None: the middle of each range).
ACTUATING_SKILL = {
    "pybullet_boil": ("SwitchFaucetOn", ["robot", "faucet"], [0.05, 0.1]),
    "pybullet_domino": ("Push", ["robot", "domino_0"], None),
    "pybullet_fan": ("SwitchOn", ["robot", "fan_0"], [0.05, 0.1]),
    "pybullet_bridge": ("PickBlock", ["robot", "span0"], None),
    "pybullet_balloons": ("Release", ["robot", "clip0"], None),
}


def _withheld(text: str, asset: bool = False) -> List[str]:
    """What ``text`` holds that this arm's agent must not read; ``asset`` marks
    the text of a URDF or mesh file."""
    for phrase in SKILL_PARAMETER_PHRASES:
        text = text.replace(phrase, "")
    found = []
    for pattern, _reason in FORBIDDEN:
        if asset and pattern == MODEL:
            continue
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            start = max(0, match.start() - 60)
            found.append(f"{pattern!r}: ...{text[start:match.end() + 60]}...")
    return found


def _arm_flags(arm: str, env_name: str = "pybullet_boil") -> Dict[str, Any]:
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
          arm: str,
          env_name: str = "pybullet_boil",
          **overrides: Any) -> Any:
    flags = {**_arm_flags(arm, env_name), **overrides}
    _config(tmp_path, **flags)
    env = create_new_env(env_name, do_cache=False, use_gui=False)
    approach = create_approach(flags["approach"], env.predicates,
                               get_gt_options(env.get_name()), env.types,
                               env.action_space,
                               [t.task for t in env.get_train_tasks()])
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
    assert not _withheld(out), _withheld(out)
    return json.loads(out.strip().splitlines()[-1])


def _assert_no_mechanism(scene: Any) -> None:
    """The world is ``SceneBase`` bound to the deployment and to the scene
    files: the file the harness loads adds a declaration of the scored
    features, the supplied class a report of the values it is held at, and the
    bound class the gripper's reading conventions."""
    loaded, supplied, bound, base = type(scene).__mro__[:4]
    assert base is SceneBase
    for cls, methods in ((loaded, []), (supplied, [
            "get_physical_param_info"
    ]), (bound, ["_fingers_joint_to_state", "_fingers_state_to_joint"])):
        assert sorted(name for name, value in vars(cls).items()
                      if callable(value) or isinstance(value, (
                          classmethod, property))) == methods, cls
    assert "_domain_specific_step" not in vars(SceneBase)
    assert type(scene).AGENT_PARAM_SPECS == []
    assert not scene._agent_param_values
    assert not scene._sampled_materials


def test_config_is_from_assets_without_what_acts_on_a_model() -> None:
    """Five settings, five seeds, parked, under a name that says only what the
    arm has; the planning switches of EMPIRIC from assets are on, and the ones
    that act on a model file, a fit or a parameter are not."""
    names = {run.approach for run in generate_run_configs(CONFIG, False)}
    assert "agent_continual_from_assets_scene" not in names
    assert "agent_continual_no_learning" not in names
    with pytest.raises(Exception):
        list(
            generate_run_configs(CONFIG,
                                 False,
                                 approaches=["no_learning_opus"]))
    runs = list(generate_run_configs(CONFIG, False, approaches=[ARM]))
    assert len(runs) == 25
    assert {run.env for run in runs} == set(DOMAINS)
    assert len({run.experiment_id for run in runs}) == 5
    for run in runs:
        assert run.approach == "agent_continual_from_assets_scene"
        assert not _withheld(run.approach + " " + run.experiment_id)
        for kept in ("scene_manifest_loader", "scene_observed_sizes",
                     "agent_sim_learn_declared_params_only",
                     "agent_sim_learn_fit_invisible",
                     "continual_rehearsal_gate",
                     "continual_rehearsal_notes_gate",
                     "continual_rehearsal_safe_plan",
                     "continual_rehearsal_still_start",
                     "continual_restage_velocities",
                     "continual_rehearsal_mean_verdict",
                     "continual_rehearsal_win_first",
                     "continual_predictive_check",
                     "continual_belief_velocities",
                     "continual_rehearsal_game_over",
                     "continual_pooled_level_init",
                     "continual_practise_after_reset", "continual_claim_check",
                     "continual_uncertainty_decisions"):
            assert run.flags[kept] is True, kept
        assert run.flags["belief_joint_draws"] == 16
        assert run.flags["continual_belief_window"] == 64
        for dropped in ("continual_require_model_on_test",
                        "code_sim_learning_prior_spans_bounds",
                        "belief_material_lines",
                        "continual_rehearsal_support_note",
                        "continual_rehearsal_speed_note",
                        "continual_fit_reach_check",
                        "continual_fit_bound_note",
                        "belief_segment_temperature",
                        "belief_keep_undeclared_materials"):
            assert dropped not in run.flags, dropped


def test_arm_refuses_a_setup_that_would_give_it_away(tmp_path: Any) -> None:
    """The class is the registered one, and it refuses a run without its scene
    loader or its joint draws, a switch that acts on a model, and an experiment
    id that names what the arm lacks."""
    _, approach = _make(tmp_path, ARM)
    assert isinstance(approach, AgentContinualNoLearningApproach)
    assert approach.get_name() == "agent_continual_from_assets_scene"
    assert approach._tool_context.supplied_simulator is True
    assert approach._tool_context.probe_absent == SCENE_SIMULATOR_PROBE_ABSENT
    with pytest.raises(ValueError, match="scene_manifest_loader"):
        _make(tmp_path, ARM, scene_manifest_loader=False)
    with pytest.raises(ValueError, match="belief_joint_draws"):
        _make(tmp_path, ARM, belief_joint_draws=0)
    with pytest.raises(ValueError, match="agent_sim_learn_fit_invisible"):
        _make(tmp_path, ARM, agent_sim_learn_fit_invisible=False)
    for switch in ("continual_require_model_on_test",
                   "continual_fit_bound_note",
                   "continual_rehearsal_support_note",
                   "continual_skill_preflight"):
        on: Dict[str, Any] = {switch: True}
        with pytest.raises(ValueError, match=switch):
            _make(tmp_path, ARM, **on)
    for name in ("boil-no_learning_opus_r1", "boil-scene_ablation_r1",
                 "boil-from_assets_scene_opus_nofit", "boil-no_mechanisms_r2",
                 "boil-fixed_params_r1"):
        with pytest.raises(ValueError, match="run folder"):
            _make(tmp_path, ARM, experiment_id=name)
    _make(tmp_path, ARM, experiment_id="boil-from_assets_scene_opus_r41")


def test_supplied_scene_adds_nothing_to_the_base(tmp_path: Any) -> None:
    """The file the harness loads as the world subclasses the scene base with
    one declaration, the pose features a step is scored on."""
    _, approach = _make(tmp_path, ARM)
    source = supplied_scene_source(approach._types)
    assert approach._frozen_model_source == source
    assert "def " not in source and "import" not in source
    namespace: Dict[str, Any] = {"SceneBase": SceneBase}
    exec(source, namespace)  # pylint: disable=exec-used
    scene = namespace["RESIDUAL_ENV"]
    assert scene.__mro__[1] is SceneBase
    assert scene.RESIDUAL_FEATURES == {
        "burner": ["x", "y", "z"],
        "faucet": ["x", "y", "z", "rot"],
        "jug": ["x", "y", "z", "rot"],
        "switch": ["x", "y", "z", "rot"],
    }
    assert [n for n in vars(scene)
            if not n.startswith("_")] == ["RESIDUAL_FEATURES"]


def test_notes_and_sandbox_text_name_no_model() -> None:
    """A note of the probe core that speaks of a model comes back in this arm's
    wording or not at all, and the sandbox CLAUDE.md describes the reference
    files as the scene files they are."""
    cap = ("step 0 (Wait) ran to its step cap (120 actions). Inspect the "
           "final state and subgoal report to decide whether to wait longer "
           "or revise the modeled dynamics.")
    mean = ("The rollout from the belief mean, at the declared values (the "
            "values the planning model runs at), ends the episode in "
            "GAME_OVER, but none of the 16 draws the task's evaluator scored "
            "does: the planning model predicts a loss the draws miss, so "
            "they may understate its risk.")
    start = ("2 of 16 draws did not start from the state they were given: "
             "after setting it, simulator.py reads back other values of "
             "jug0.x, so those draws ran from another state. Make the model "
             "set and report these features exactly, or this rehearsal "
             "predicts from the wrong start.")
    unknown = "PARAMS at the declared values of the current simulator.py."
    plain = "rollout ran at planner seed 3 (base seed overridden)"
    notes = arm_notes([cap, mean, start, unknown, plain])
    assert notes == [
        "step 0 (Wait) ran to its step cap (120 actions). Inspect the final "
        "state and subgoal report to decide whether to wait longer.",
        "The rollout from the belief mean ends the episode in GAME_OVER, "
        "but none of the 16 draws the task's evaluator scored does: that "
        "rollout predicts a loss the draws miss, so they may understate its "
        "risk.",
        "2 of 16 draws did not start from the state they were given: after "
        "setting it, the simulator reads back other values of jug0.x, so "
        "those draws ran from another state. For those features this "
        "rehearsal predicts from that other start.",
        plain,
    ]
    for note in notes:
        assert not _withheld(note), note
    legend = scene_manifest_legend(LEGEND)
    assert _withheld(json.dumps(LEGEND))
    assert not _withheld(json.dumps(legend)), _withheld(json.dumps(legend))
    assert list(legend) == list(LEGEND)
    assert sum(text != legend[key] for key, text in LEGEND.items()) == 5
    shared = build_claude_md()
    assert _withheld(shared)
    own = build_scene_simulator_claude_md()
    assert not _withheld(own), _withheld(own)
    assert "The scene files are in `./reference/`" in own
    assert "The kind is `play`.\n" in own
    # Every other line is the shared one.
    changed = [
        line for line in own.splitlines() if line not in shared.splitlines()
    ]
    assert len(changed) == 2, changed


def _agent_reads(approach: Any, message: str) -> Dict[str, str]:
    """Everything the agent is sent when a round opens, by name."""
    ctx = approach._tool_context
    sandbox = Path(ctx.sandbox_dir)
    log_dir = Path(approach._get_log_dir())
    system = next(log_dir.glob("full_system_prompt_*.md")).read_text()
    assert system == approach._get_agent_system_prompt() + \
        _LOCAL_SANDBOX_SYSTEM_PROMPT
    texts = {
        "system prompt and sandbox suffix": system,
        "CLAUDE.md": (sandbox / "CLAUDE.md").read_text(),
        "first-round query": message,
        "working directory": str(sandbox),
    }
    for tool in ctx.extra_mcp_tools:
        texts[f"tool {tool.name}"] = (f"{tool.name}\n{tool.description}\n" +
                                      json.dumps(tool.input_schema))
    return texts


@pytest.mark.parametrize("env_name", DOMAINS)
def test_agent_reads_only_what_the_arm_has(env_name: str, tmp_path: Any,
                                           monkeypatch: Any,
                                           caplog: Any) -> None:
    """On each benchmark domain, with no file written: nothing the agent is
    sent names what the arm lacks, the sandbox holds the scene files and no
    simulator source, and ``sim`` starts from the observation and runs a skill
    in the loaded scene, where a reading that no body pose carries keeps its
    observed value."""
    env, approach = _make(tmp_path,
                          ARM,
                          env_name,
                          continual_refuse_training_give_up=False)

    def script(message: str) -> None:
        ctx = approach._tool_context
        sandbox = Path(ctx.sandbox_dir)
        texts = _agent_reads(approach, message)
        for where, text in texts.items():
            assert not _withheld(text), (where, _withheld(text))
        assert "## Simulator and data" in message
        assert "The simulator holds this level's scene" in message
        assert texts["CLAUDE.md"] == build_scene_simulator_claude_md()
        tools = [t.name for t in ctx.extra_mcp_tools]
        assert sorted(tools) == sorted(approach._continual_tool_names())
        assert {"run_python", "claims_record", "claims_resolve"} <= set(tools)
        described = next(t for t in ctx.extra_mcp_tools
                         if t.name == "run_python").description
        assert described == scene_simulator_description()

        # The scene files, and no simulator source.
        reference = sandbox / "reference"
        refs = {
            str(q.relative_to(reference))
            for q in reference.rglob("*") if q.is_file()
        }
        assert {"scene/scene_manifest.json", "scene/levels/level_01.json"
                } <= refs
        assert any(r.startswith("assets/") for r in refs)
        assert not any(r.startswith("base_sim/") for r in refs), refs
        assert all(r.startswith(("scene/", "assets/")) for r in refs), refs
        for ref in sorted(refs):
            try:
                text = (reference / ref).read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue  # a binary mesh
            asset = ref.startswith("assets/")
            assert not _withheld(text, asset), (ref, _withheld(text, asset))
        # The simulator loads its own copy, which no sandbox edit reaches.
        copy = Path(approach._scene_copy_dir())
        assert sandbox not in copy.parents
        assert (copy / "scene/levels/level_01.json").is_file()
        assert (copy / "scene/scene_manifest.json").read_text() == \
            (reference / "scene/scene_manifest.json").read_text()

        # A reset from the observation and a skill, with no file written.
        out = _python(
            approach, "sim.reset(current=True)\n"
            "before = sim.state()\n"
            "r = sim.run('Wait(robot:robot)[5]', draws=0, render=False)\n"
            "print(json.dumps({'before': before, 'after': sim.state(), "
            "'report': str(r), 'states': len(r.states), "
            "'task': sim.task(), 'belief': str(sim.belief())}))")
        assert "FAILURE" not in out["report"], out["report"]
        assert out["states"] == 6
        assert set(out["before"]) == set(out["after"])
        kept = {}
        for name, features in out["before"].items():
            if name.endswith(":robot"):
                continue
            for feature, value in features.items():
                if feature in ENGINE_FEATURES:
                    continue
                kept[f"{name}.{feature}"] = (value,
                                             out["after"][name][feature])
        print(f"READINGS {env_name} (start, after Wait[5]): {kept}")
        for name, (start, end) in kept.items():
            assert end == start, name
        # Every domain but Domino observes features that no pose carries.
        assert bool(kept) == (env_name != "pybullet_domino")
        # A skill that works a switch, a clip or a body: the readings stay
        # at the values of the rollout's start, whatever the bodies do.
        skill, names, values = ACTUATING_SKILL[env_name]
        option = next(o for o in ctx.options if o.name == skill)
        middle = values or list(
            (option.params_space.low + option.params_space.high) / 2)
        typed = {key.split(":")[0]: key for key in out["before"]}
        line = (f"{skill}({', '.join(typed[n] for n in names)})"
                f"[{', '.join(f'{v:.4f}' for v in middle)}]")
        acted = _python(
            approach, "sim.reset(current=True)\n"
            f"r = sim.run({line!r} + '\\nWait(robot:robot)[20]', draws=0, "
            "render=False)\n"
            "start = {f'{o.name}:{o.type.name}': {f: float(r.states[0].get("
            "o, f)) for f in o.type.feature_names} for o in r.states[0]}\n"
            "end = {f'{o.name}:{o.type.name}': {f: float(r.final_state.get("
            "o, f)) for f in o.type.feature_names} for o in r.final_state}\n"
            "print(json.dumps({'start': start, 'end': end, "
            "'report': str(r)}))")
        moved = {}
        readings = {}
        for name, features in acted["start"].items():
            if name.endswith(":robot"):
                continue
            for feature, value in features.items():
                end = acted["end"][name][feature]
                if feature not in ENGINE_FEATURES:
                    assert end == value, (name, feature)
                    if feature in ("is_on", "is_hit", "tied", "popped"):
                        readings[f"{name}.{feature}"] = end
                elif abs(end - value) > 0.01:
                    moved[f"{name}.{feature}"] = round(end - value, 3)
        head = " | ".join(acted["report"].splitlines()[:3])
        print(f"ACTUATED {env_name}: {line}: {head}; poses moved: {moved}; "
              f"switches and flags at the end, as at the start: {readings}")
        # A rehearsal on the joint draws runs in each domain's scene.
        rehearsed = _python(
            approach, "r = sim.reset(current=True).run("
            "'Wait(robot:robot)[2]', render=False)\n"
            "print(json.dumps({'report': str(r), 'draws': len(r.draws), "
            "'notes': r.notes}))")
        assert rehearsed["report"].startswith(
            "Rehearsal on 16 joint draws of the belief: P-hat = ")
        assert rehearsed["draws"] == 16
        print(f"REHEARSED {env_name}: " + rehearsed["report"].splitlines()[0] +
              f" notes: {rehearsed['notes']}")
        scene = approach._base_env
        _assert_no_mechanism(scene)
        assert scene._loaded_scene is not None
        assert not scene._loaded_scene.notes
        assert ctx.env is scene
        # The level's reward model scores the rollout.
        verdict = _python(
            approach, "v = evaluate_trajectory(r.states, r.labels)\n"
            "print(json.dumps([sorted(v), bool(v['solved'])]))")
        assert verdict == [["note", "reward", "solved"], False]
        # Nothing of a model reached the sandbox.
        top = {q.name for q in sandbox.iterdir()}
        assert not top & {
            "simulator.py", "predicates.py", "simulator_versions",
            "predicates_versions"
        }, top

    card = _play(env, approach, monkeypatch, script)
    # No note of a report named what the arm lacks and was left out.
    assert not [
        record.getMessage() for record in caplog.records
        if "left out of the report" in record.getMessage()
    ]
    assert approach._tool_context.probe_option_model_provider is None
    level = card.levels[0]
    assert "fits" not in level.sandbox
    assert level.sandbox["sim_rollouts"] > 0
    attempts = (Path(approach._tool_context.sandbox_dir) /
                "attempts.md").read_text()
    assert not _withheld(attempts), _withheld(attempts)


def test_prompt_describes_the_simulator_and_the_workflow(
        tmp_path: Any) -> None:
    """The system prompt has the workflow's sections on the supplied simulator
    and none on a model; the shared sections are the other arms' but for the
    words this arm gives up."""
    _, approach = _make(tmp_path, ARM)
    approach._get_sandbox_reference_files()
    prompt = approach._get_agent_system_prompt()
    _check_golden("continual_system_from_assets_scene", prompt)
    assert not _withheld(prompt), _withheld(prompt)
    headings = [line for line in prompt.splitlines() if line.startswith("#")]
    assert headings == [
        "## Run rules",
        "## Reading observations",
        "## Observation noise",
        "## The simulator",
        "## Decision workflow",
        "### Test-level steps wait for a joint rehearsal",
        "### A reset training level asks for a rehearsal",
        "### Claim checks",
        "### Each rehearsed step is checked against its rehearsal",
        "## Tools",
        "### Skill grammar",
        "## Working files",
        "## Run memory",
        "### Conversation rounds",
        "## Simulator workbench",
        "### Interpreting task verdicts",
        "### Scene manifest and assets",
    ]
    for present in (
            "ready from the first round",
            "has the file's masses",
            "the mass of its volume at 1000 kg/m^3",
            "the engine's defaults where no file sets one",
            "keeps the value of the state that a rollout starts from",
            "`sim.refine(plan, require_goal=True)`",
            "over the 16 joint draws of the belief",
            "highest P-hat",
            "Start a new plan only from rest",
            "`claims_record`",
            "`evaluate_trajectory(states, actions=None, task_idx=None)`",
            "./reference/scene/scene_manifest.json (",
            "- `run_python`: code in the sandbox with `sim`, the simulator "
            "of the scene",
    ):
        assert present in prompt, present
    description = scene_simulator_description()
    assert not _withheld(description), _withheld(description)
    for call in SIM_CALLS - {"drop", "clear_snapshots"}:
        assert f"`sim.{call}(" in description, call
    # The switches' sections follow their flags.
    _, bare = _make(tmp_path,
                    ARM,
                    continual_claim_check=False,
                    continual_predictive_check=False,
                    continual_practise_after_reset=False,
                    continual_rehearsal_safe_plan=False)
    bare._get_sandbox_reference_files()
    shorter = bare._get_agent_system_prompt()
    for gone in ("### Claim checks", "### Each rehearsed step is checked",
                 "### A reset training level", "A lost draw", "a lost draw"):
        assert gone not in shorter, gone
    assert "claims_record" not in bare._continual_tool_names()


def test_agent_plans_in_the_supplied_simulator_on_a_test_level(
        tmp_path: Any, monkeypatch: Any) -> None:
    """In a session on a Boil test level, with no file written: ``sim`` has the
    arm's calls and no other, rehearses on the joint draws and reports, and the
    rehearsal gate holds a skill request back before a rehearsal and opens it
    after one; a model file the agent writes changes nothing."""
    env, approach = _make(tmp_path, ARM, continual_levels="test_only")

    def script(message: str) -> None:
        assert "(test task 0, no resets)" in message
        ctx = approach._tool_context
        sandbox = Path(ctx.sandbox_dir)
        assert ctx.skill_gate is None and ctx.skill_preflight is None
        assert ctx.rehearsal_gate is not None and ctx.claim_check is not None
        assert not (sandbox / "simulator.py").exists()

        # The calls ``sim`` has, and the ones it does not.
        listed = _python(
            approach, "print(json.dumps([[n for n in dir(sim) "
            "if not n.startswith('_')], sorted(vars(sim)), "
            "type(sim).__name__]))")
        assert set(listed[0]) == SIM_CALLS
        assert listed[1:] == [[], "Simulator"]
        assert _python(
            approach, "print(json.dumps([hasattr(sim, n) for n in "
            f"{ABSENT_CALLS!r} + ['_' + 'probe']]))") == [False] * (
                len(ABSENT_CALLS) + 1)
        manual = _python(
            approach, "import pydoc\n"
            "print(json.dumps(pydoc.plain(pydoc.render_doc(sim))))")
        for call in SIM_CALLS:
            assert f" {call}(self" in manual, call
        for call in ABSENT_CALLS:
            assert f" {call}(" not in manual, call
        refused = _call(approach, "run_python", code="sim.fit()")
        assert refused.strip().endswith(
            "AttributeError: 'Simulator' object has no attribute 'fit'")
        # The traceback shows the agent's own frame and no harness file.
        assert refused.count("File ") == 1 and "predicators" not in refused
        for keyword in ("physics_sweep=True", "phys_params={'jug_mass': 2}",
                        "trials=3", "fresh=True"):
            refused = _call(approach,
                            "run_python",
                            code="sim.run('Wait(robot:robot)[1]', " + keyword +
                            ")")
            name = keyword.split("=", maxsplit=1)[0]
            assert f"unexpected keyword argument '{name}'" in refused, refused
        refused = _call(approach,
                        "run_python",
                        code="evaluate_trajectory([], physics_sweep=True)")
        assert "unexpected keyword argument 'physics_sweep'" in refused
        refused = _call(approach, "run_python", code="print(ParamSpec)")
        assert "NameError" in refused, refused

        # A reset to the observation, a skill, and the states it predicts.
        start = _python(
            approach, "sim.reset(current=True)\n"
            "print(json.dumps(sim.state('jug0')))")
        assert start["is_held"] == 0.0
        picked = _python(
            approach, "r = sim.run('PickJug(robot:robot, jug0:jug)[0.0]', "
            "draws=0, render=False)\n"
            "print(json.dumps({'report': str(r), 'states': len(r.states), "
            "'labels': len(r.labels), 'jug': sim.state('jug0')}))")
        assert "Step 0: PickJug" in picked["report"], picked["report"]
        assert "FAILURE" not in picked["report"], picked["report"]
        assert picked["states"] == picked["labels"] + 1 > 2
        assert picked["jug"]["is_held"] == 1.0
        for feature in ("water_volume", "bubbling_level"):
            assert picked["jug"][feature] == start[feature]

        # The plan search runs in the same world, from the belief mean,
        # and scores what it finds on the joint draws.
        refined = _python(
            approach, "sim.reset(current=True)\n"
            "print(json.dumps(str(sim.refine("
            "'PickJug(robot:robot, jug0:jug)[0.02]', timeout=20))))")
        print(f"REFINE REPORT\n{refined}")
        assert refined.startswith("Refinement SUCCESS"), refined
        assert "joint draws (P-hat" in refined, refined

        # A recording, which a claim check asks for; a Wait needs no
        # rehearsal.
        waited = _call(approach, "skills_invoke", skill="Wait(robot:robot)[1]")
        assert not waited.startswith("ERROR"), waited
        assert not _withheld(waited), _withheld(waited)

        # The gate holds a skill request back before a rehearsal.
        request = "PickJug(robot:robot, jug0:jug)[0.0]"
        held = _call(approach, "skills_invoke", skill=request)
        assert held.startswith("ERROR"), held
        assert "rehearse this on the joint belief first" in held, held
        assert "Nothing was charged" in held
        assert not _withheld(held), _withheld(held)
        steps = approach._play_session.observe().ledger.run_steps

        # A rehearsal over the joint draws runs and reports.
        report = _python(
            approach, "r = sim.reset(current=True).run("
            f"'{request}\\nWait(robot:robot)[2]', render=False)\n"
            "print(r)\n"
            "print(json.dumps({'report': str(r), 'type': type(r).__name__, "
            "'keys': sorted(r.draws[0]), 'draws': len(r.draws), "
            "'p_hat': r.p_hat, 'held': sim.state('jug0')['is_held'], "
            "'states': len(r.states), "
            "'failures': [d['failure'] for d in r.draws], "
            "'seeds': len({d['planner_seed'] for d in r.draws})}))")
        text = report["report"]
        print(f"REHEARSAL REPORT\n{text}")
        assert text.startswith(
            "Rehearsal on 16 joint draws of the belief: P-hat = "), text
        assert "each draw is a start state the observations allow" in text
        assert "Step-by-step rollout from the belief mean:" in text
        assert "  draw 15: " in text and "draw 0 [" not in text
        assert report["type"] == "Rehearsal"
        assert report["draws"] == 16 and report["seeds"] == 16
        assert "params" not in report["keys"], report["keys"]
        assert 0.0 <= report["p_hat"] <= 1.0
        # The rollout from the belief mean moved the current state.
        assert report["held"] == 1.0 and report["states"] > 2

        # After it the gate opens; the claim check then asks for the
        # belief behind a step at a low P-hat, and lets the step through
        # once it is on record.
        asked = _call(approach, "skills_invoke", skill=request)
        assert asked.startswith("ERROR") and "Claim check:" in asked, asked
        assert not _withheld(asked), _withheld(asked)
        recorded = _call(approach,
                         "claims_record",
                         claim="the jug lifts from its handle",
                         rules_out="a grasp lower on the body",
                         direct_tests=[{
                             "recording": 0,
                             "what": "the scene at rest"
                         }],
                         what_varied="nothing varied",
                         rival="the handle slips from the fingers",
                         next_experiment="pick the jug once",
                         predictions={
                             "claim": "the jug rises with the hand",
                             "rival": "the jug stays on the table"
                         })
        assert "Recorded as 0" in recorded, recorded
        resolved = _call(approach,
                         "claims_resolve",
                         id=0,
                         verdict="not_run",
                         result="the step itself is the experiment")
        assert "resolved" in resolved, resolved
        ran = _call(approach, "skills_invoke", skill=request)
        assert not ran.startswith("ERROR"), ran
        assert not _withheld(ran), _withheld(ran)
        print(f"SKILL RESULT\n{ran}")
        assert approach._play_session.observe().ledger.run_steps > steps
        # The step loaded the world for the level's reward model too.
        assert ctx.scored_features_provider() == {
            "burner": ["x", "y", "z"],
            "faucet": ["x", "y", "z", "rot"],
            "jug": ["x", "y", "z", "rot"],
            "switch": ["x", "y", "z", "rot"],
        }

        # A file of the agent's is never loaded: a model with a mechanism
        # and predicates of its own change nothing.
        (sandbox / "simulator.py").write_text(SCENE_SOURCE, encoding="utf-8")
        (sandbox / "predicates.py").write_text(
            "LEARNED_PREDICATES = [Predicate('Full', [jug_type], "
            "lambda s, o: True)]\n",
            encoding="utf-8")
        # A rollout from the latest observation starts at the belief mean,
        # and a reading keeps the value it has there.
        water = _python(
            approach, "sim.reset(current=True)\n"
            "r = sim.run('Wait(robot:robot)[3]', draws=0, render=False)\n"
            "jug = [o for o in r.states[0] if o.name == 'jug0'][0]\n"
            "hs = [sim.run_async('Wait(robot:robot)[2]', draws=0)]\n"
            "g = sim.gather(hs)\n"
            "print(json.dumps([float(r.states[0].get(jug, 'water_volume')), "
            "sim.state('jug0')['water_volume'], sim.atoms(), str(g), "
            "[h.ok for h in hs]]))")
        assert water[0] == water[1]
        assert not any("Full" in atom for atom in water[2]), water[2]
        assert "1 done, 0 pending" in water[3] and water[4] == [True]
        _assert_no_mechanism(approach._base_env)
        assert type(approach._base_env).__name__ == "SuppliedScene"
        assert not approach._learned_predicates
        # The tests keep the environment's Holding; the arm adds none.
        assert {p.name for p in approach._get_all_predicates()} == {"Holding"}

        image = _python(approach, "print(json.dumps(sim.render('now')))")
        assert image.endswith(".png") and (sandbox / image).is_file(), image

    card = _play(env, approach, monkeypatch, script)
    level = card.levels[0]
    assert level.split == "test" and level.steps > 1
    assert "fits" not in level.sandbox
    assert level.sandbox["sim_rollouts"] >= 16
    sandbox = Path(approach._tool_context.sandbox_dir)
    for record in ("attempts.md", "journal.md"):
        path = sandbox / record
        if path.is_file():
            assert not _withheld(path.read_text()), record
    assert not (sandbox / "simulator_versions").exists()
    assert not (sandbox / "predicates_versions").exists()


def test_evaluate_trajectory_scores_a_rollout(tmp_path: Any,
                                              monkeypatch: Any) -> None:
    """The reward model in ``run_python`` scores the states a rollout predicts,
    and states built from them."""
    env, approach = _make(tmp_path,
                          ARM,
                          continual_refuse_training_give_up=False)
    assert "evaluate_trajectory" not in approach._continual_tool_names()

    def script(_message: str) -> None:
        rollout = _python(
            approach, "sim.reset(current=True)\n"
            "r = sim.run('Wait(robot:robot)[2]', draws=0, render=False)\n"
            "v = evaluate_trajectory(r.states, r.labels)\n"
            "print(json.dumps([sorted(v), float(v['reward']), "
            "bool(v['solved']), v['note'], "
            "bool(is_goal_state(r.final_state))]))")
        assert rollout == [["note", "reward", "solved"], 0.0, False, "", False]
        # The goal of the simple Boil level as feature values: a full,
        # boiled jug, the burner off and nothing spilled.
        capacity = float(env.max_jug_water_capacity)
        solved = _python(
            approach, "goal = r.final_state.copy()\n"
            "for o in goal:\n"
            "    f = o.type.feature_names\n"
            "    if o.type.name == 'jug':\n"
            f"        goal.set(o, 'water_volume', {capacity!r})\n"
            "        for n in ('heat_level', 'bubbling_level'):\n"
            "            if n in f:\n"
            "                goal.set(o, n, 1.0)\n"
            "    elif o.type.name == 'burner':\n"
            "        goal.set(o, 'is_on', 0.0)\n"
            "    elif o.type.name == 'faucet':\n"
            "        goal.set(o, 'is_on', 0.0)\n"
            "        goal.set(o, 'spilled_level', 0.0)\n"
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
            "r = sim.run('Wait(robot:robot)[1]', draws=0, render=False)\n"
            "v = evaluate_trajectory(t.states[:-1] + r.states, "
            "list(t.actions) + r.labels)\n"
            "print(json.dumps([len(t.actions), bool(v['solved']), "
            "describe_trajectory(0, max_timesteps=2)]))")
        assert mixed[:2] == [1, False]

    _play(env, approach, monkeypatch, script)


def test_standalone_model_gets_the_scene_assets(tmp_path: Any) -> None:
    """Standalone sim.

    receives the manifest and assets, is told so, and still has no base
    simulator.
    """
    _, approach = _make(tmp_path, "standalone_opus")
    files = approach._get_sandbox_reference_files()
    assert "scene/scene_manifest.json" in files
    assert "assets/urdf/jug-pixel.urdf" in files
    assert not any(path.startswith("base_sim/") for path in files)
    prompt = approach._get_agent_system_prompt()
    assert "## Scene files" in prompt
    assert "scene_manifest.json (" in prompt
    assert "The scene files listed below describe the scene's bodies" in \
        prompt
    assert "base simulator" not in prompt
    assert "No prepared physical scene" not in prompt
    assert "pybullet_env.py" not in prompt


def test_standalone_run_reads_the_scene_assets(tmp_path: Any,
                                               monkeypatch: Any) -> None:
    """In a run the assets reach the sandbox before the first query, the prompt
    counts the manifest, and the level's recording keeps it."""
    # The scripted session ends with give_up on its training level.
    env, approach = _make(tmp_path,
                          "standalone_opus",
                          continual_refuse_training_give_up=False)
    seen: List[str] = []

    def query(message: str, *_args: Any, **_kwargs: Any) -> Any:
        seen.append(message)
        sandbox = Path(approach._tool_context.sandbox_dir)
        reference = sandbox / "reference"
        assert (reference / "scene/scene_manifest.json").is_file()
        assert (reference / "assets/urdf/jug-pixel.urdf").is_file()
        assert not (reference / "base_sim").exists()
        log_dir = Path(approach._get_log_dir())
        system = next(log_dir.glob("full_system_prompt_*.md")).read_text()
        assert "bodies)" in system, system
        assert (log_dir.parent / "L01" / "scene_manifest.json").read_text() \
            == (reference / "scene/scene_manifest.json").read_text()
        assert "Give-up recorded" in _call(approach, "give_up", note="done")
        return _result()

    monkeypatch.setattr(approach, "_query_agent_sync", query)
    approach.prepare_for_continual(Dataset([]))
    card = ContinualRun(env, approach, create_level_player(env,
                                                           approach)).run()
    assert card.end_note == "done"
    assert len(seen) == 1
