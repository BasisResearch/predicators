"""Tests for the program world model approach's harness glue (C4)."""
# pylint: disable=protected-access
import os
from typing import Any, List

from predicators import utils
from predicators.agent_sdk.tools import ToolContext
from predicators.approaches import agent_program_world_model_approach as apwm
from predicators.approaches.agent_sim_learning_approach import _SynthesisPaths
from predicators.code_sim_learning.program_world_model import \
    load_program_world_model
from predicators.datasets import create_dataset
from predicators.envs import create_new_env
from predicators.ground_truth_models import get_gt_options

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


def test_installed_program_rolls_latents() -> None:
    """An installed program backs the option model, and materialise_latent
    rolls it along a recorded trajectory."""
    env, train_tasks, options = _cover()
    approach = _bare(env, train_tasks, options)
    program, err = load_program_world_model(_PROGRAM, env.types,
                                            env.predicates, options)
    assert err is None and program is not None
    approach._install_program(program)
    assert approach._option_model is approach._program_model
    dataset = create_dataset(env, train_tasks, options, env.predicates)
    traj = dataset.trajectories[0]
    latents = approach.materialise_latent(traj)
    assert len(latents) == len(traj.states)
    assert latents[-1]["phase"] > latents[0]["phase"]
    assert approach._latent_tracking_available() is False


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
