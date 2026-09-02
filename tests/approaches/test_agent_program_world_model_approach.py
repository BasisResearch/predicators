"""Tests for the program world model approach's harness glue (C4)."""
# pylint: disable=protected-access
import os
from typing import Any, List

import numpy as np

from predicators import utils
from predicators.agent_sdk import learn_prompts
from predicators.agent_sdk.tools import ToolContext
from predicators.approaches import agent_program_world_model_approach as apwm
from predicators.approaches.agent_sim_learning_approach import _SynthesisPaths
from predicators.code_sim_learning.program_world_model import \
    ProgramOptionModel, load_program_world_model
from predicators.datasets import create_dataset
from predicators.envs import create_new_env
from predicators.ground_truth_models import get_gt_options
from predicators.settings import CFG

_PROGRAM = '''
LATENT_FEATURES = {"robot": ["phase"]}

def initial_latent(obs, rng):
    return {"phase": int(rng.integers(0, 3))}

def transition(obs, latent, option, rng):
    nxt = obs.copy()
    for obj in nxt:
        if obj.type.name == "robot":
            nxt.set(obj, "hand", float(option.params[0]))
    return nxt, {"phase": latent["phase"] + 1}, 2
'''


def _cover() -> Any:
    utils.reset_config({
        "env": "cover",
        "num_train_tasks": 2,
        "num_test_tasks": 1,
        "agent_program_belief_particles": 8,
        "seed": 0,
    })
    env = create_new_env("cover")
    train_tasks = [t.task for t in env.get_train_tasks()]
    options = get_gt_options(env.get_name())
    return env, train_tasks, options


def _bare(env: Any, train_tasks: List[Any], options: Any) -> Any:
    approach = apwm.AgentProgramWorldModelApproach.__new__(
        apwm.AgentProgramWorldModelApproach)
    approach._types = env.types
    approach._train_tasks = train_tasks
    approach._kept_initial_predicates = set(env.predicates)
    approach._learned_predicates = set()
    approach._get_all_options = lambda: options  # type: ignore[method-assign]
    approach._get_all_predicates = (  # type: ignore[method-assign]
        lambda: set(env.predicates))
    approach._tool_context = ToolContext()
    approach._program = None
    approach._program_model = None
    approach._learned_simulator = None
    approach._option_model = None
    return approach


def test_belief_particles_and_override_scope() -> None:
    """Particles, the nominal latent, the override scope, and the rolled
    latents all come from the installed program."""
    env, train_tasks, options = _cover()
    approach = _bare(env, train_tasks, options)
    # No model yet: no particles, and the initial latent is left alone.
    assert not approach._belief_particles()
    assert approach._attach_initial_latent(train_tasks[0]) is train_tasks[0]
    program, err = load_program_world_model(_PROGRAM, env.types,
                                            env.predicates, options)
    assert err is None and program is not None
    approach._install_program(program)
    assert approach._option_model is approach._program_model
    particles = approach._belief_particles()
    # Distinct draws only: initial_latent has three outcomes.
    assert 1 <= len(particles) <= 3
    assert len({p["phase"] for p in particles}) == len(particles)
    # Deterministic across calls (seeded).
    assert approach._belief_particles() == particles
    # The current task drives the draw when one is set.
    approach._tool_context.current_task = train_tasks[1]
    assert approach._belief_particles() == particles
    # The nominal latent is attached to the task the planner sees.
    task = approach._attach_initial_latent(train_tasks[0])
    assert task.init.latent is not None and "phase" in task.init.latent
    assert train_tasks[0].init.latent is None
    # Under the scope every latent-less start rolls from the particle.
    model: ProgramOptionModel = approach._program_model
    (pick_place, ) = [o for o in options if o.name == "PickPlace"]
    option = pick_place.ground([], np.array([0.4], dtype=np.float32))
    with approach._particle_override_scope({"phase": 20}):
        nxt, _ = model.get_next_state_and_num_actions(train_tasks[0].init,
                                                      option)
        assert nxt.latent == {"phase": 21}
    assert model.initial_latent_override is None
    # materialise_latent rolls the program along a recorded trajectory.
    dataset = create_dataset(env, train_tasks, options, env.predicates)
    traj = dataset.trajectories[0]
    latents = approach.materialise_latent(traj)
    assert len(latents) == len(traj.states)
    assert latents[-1]["phase"] > latents[0]["phase"]
    assert approach._latent_tracking_available() is False


def test_learn_simulator_gating(monkeypatch) -> None:
    """No data means no session unless zero-shot is on."""
    env, train_tasks, options = _cover()
    approach = _bare(env, train_tasks, options)
    approach._persist_fit_trajectories = lambda *a, **k: None
    calls: List[Any] = []
    program, _ = load_program_world_model(_PROGRAM, env.types, env.predicates,
                                          options)

    def _session(trajectories):
        calls.append(list(trajectories))
        return program

    approach._run_program_synthesis_session = _session
    monkeypatch.setattr(CFG, "agent_sim_learn_zero_shot", False)
    approach._learn_simulator([])
    assert not calls and approach._program is None
    monkeypatch.setattr(CFG, "agent_sim_learn_zero_shot", True)
    approach._learn_simulator([])
    assert calls == [[]] and approach._program is program
    dataset = create_dataset(env, train_tasks, options, env.predicates)
    monkeypatch.setattr(CFG, "agent_sim_learn_zero_shot", False)
    approach._learn_simulator(list(dataset.trajectories))
    assert len(calls) == 2 and len(calls[1]) == len(dataset.trajectories)


def test_rehydrate_from_world_model_file(tmp_path, monkeypatch) -> None:
    """A checkpoint's world_model.py rebuilds the option model."""
    env, train_tasks, options = _cover()
    approach = _bare(env, train_tasks, options)
    base = str(tmp_path)
    paths = _SynthesisPaths(base=base,
                            simulator_file=os.path.join(base, "simulator.py"),
                            versions_dir=os.path.join(base, "v"),
                            simulator_file_for_agent="./simulator.py",
                            sandbox_dir_for_agent=".")
    monkeypatch.setattr(approach, "_resolve_synthesis_paths", lambda: paths)
    approach._rehydrate_extra_artifacts = lambda b: None
    wm = approach._world_model_paths(paths)
    assert wm["world_model_file_for_agent"] == "./world_model.py"
    approach._rehydrate_from_artifacts()
    assert approach._program is None
    with open(wm["world_model_file"], "w", encoding="utf-8") as f:
        f.write("this is not python")
    approach._rehydrate_from_artifacts()
    assert approach._program is None
    with open(wm["world_model_file"], "w", encoding="utf-8") as f:
        f.write(_PROGRAM)
    approach._rehydrate_from_artifacts()
    program = getattr(approach, "_program")
    assert program is not None
    assert program.latent_features == {"robot": ["phase"]}
    assert "world_model.py" in approach._CHECKPOINT_SANDBOX_FILES
    assert "world_model_versions" in approach._CHECKPOINT_SANDBOX_DIRS


def test_program_prompts_render() -> None:
    """System prompt and first message render without leftovers."""
    system = learn_prompts.build_program_learn_system_prompt(
        scene_viz_hint="x",
        extra_sections=[
            learn_prompts.render_predicate_invention_section("workbench")
        ],
        workflow_extra=learn_prompts.render_predicate_workflow_extra())
    assert "world_model.py" in system and "sim.score" in system
    assert "Plan format" in system and "Predicate Invention" in system
    assert "__" not in system.replace("__init__", "")
    message = learn_prompts.build_program_learn_message(
        n_trajs=0,
        n_transitions=0,
        n_demos=0,
        n_interaction=0,
        trajectory_listing="",
        structs_ref="./reference/structs.py",
        predicate_listing="- Holding(robot, block)",
        types_digest="types",
        options_digest="options",
        world_model_file="./world_model.py",
        extra_messages=[learn_prompts.render_program_zero_shot_message()])
    assert "./world_model.py" in message
    assert "No trajectory has been recorded" in message
    assert "__" not in message
