"""Tests for the program-world-model synthesis tool surface."""
import os
from typing import Any

import numpy as np

from predicators import utils
from predicators.agent_sdk.belief_probe import BeliefProbe
from predicators.agent_sdk.tools import ToolContext
from predicators.agent_sdk.tools.program_synthesis import \
    create_program_synthesis_tools
from predicators.datasets import create_dataset
from predicators.envs import create_new_env
from predicators.ground_truth_models import get_gt_options

_PROGRAM = '''
LATENT_FEATURES = {}

def initial_latent(obs, rng):
    return {}

def transition(obs, latent, option, rng):
    nxt = obs.copy()
    for obj in nxt:
        if obj.type.name == "robot":
            nxt.set(obj, "hand", float(option.params[0]))
    return nxt, latent, 3
'''


def _setup(tmp_path: Any) -> Any:
    utils.reset_config({
        "env": "cover",
        "num_train_tasks": 2,
        "num_test_tasks": 1,
        "agent_program_belief_particles": 3,
    })
    env = create_new_env("cover")
    train_tasks = [t.task for t in env.get_train_tasks()]
    options = get_gt_options(env.get_name())
    dataset = create_dataset(env, train_tasks, options, env.predicates)
    wm_file = os.path.join(str(tmp_path), "world_model.py")
    toolkit = create_program_synthesis_tools(
        {},
        trajectories=list(dataset.trajectories),
        predicates=env.predicates,
        types=env.types,
        options=options,
        world_model_file=wm_file,
        versions_dir=os.path.join(str(tmp_path), "world_model_versions"),
        sandbox_dir=str(tmp_path),
        cycle_index_provider=lambda: 0,
        rng=np.random.default_rng(0),
    )
    return env, options, dataset, wm_file, toolkit


def test_score_runner_reports_and_refuses(tmp_path) -> None:
    """sim.score loads the live file, tags its report, and returns clear errors
    for a missing / broken file and bad trajectory indices."""
    _, _, _, wm_file, toolkit = _setup(tmp_path)
    (run_python, ) = toolkit.tools
    assert getattr(run_python, "name", "") == "run_python"
    assert "sim.score" in getattr(run_python, "description", "")
    missing = toolkit.score_runner()
    assert "not found" in missing and "Use Write" in missing
    with open(wm_file, "w", encoding="utf-8") as f:
        f.write("def transition(obs, latent, option, rng): return obs\n")
    broken = toolkit.score_runner()
    assert "Error loading" in broken and "initial_latent" in broken
    with open(wm_file, "w", encoding="utf-8") as f:
        f.write(_PROGRAM)
    out = toolkit.score_runner()
    assert out.startswith("[cycle_000_vers_")
    assert "World-model score" in out and "3 particles" in out
    assert "out of range" in toolkit.score_runner(traj_idxs=[7])
    subset = toolkit.score_runner(traj_idxs=[0], num_particles=2)
    assert "2 particles" in subset and "traj 0:" in subset
    assert "traj 1:" not in subset
    program, tag, err = toolkit.load_candidate(None)
    assert err is None and program is not None
    assert tag is not None and tag.startswith("cycle_000_vers_")


def test_score_runner_without_data_points_to_rollouts(tmp_path) -> None:
    """A zero-shot session gets a pointer to rollouts instead of a score."""
    utils.reset_config({"env": "cover", "num_train_tasks": 1})
    env = create_new_env("cover")
    wm_file = os.path.join(str(tmp_path), "world_model.py")
    with open(wm_file, "w", encoding="utf-8") as f:
        f.write(_PROGRAM)
    toolkit = create_program_synthesis_tools(
        {},
        trajectories=[],
        predicates=env.predicates,
        types=env.types,
        options=get_gt_options(env.get_name()),
        world_model_file=wm_file,
        versions_dir=os.path.join(str(tmp_path), "v"),
        sandbox_dir=str(tmp_path),
    )
    out = toolkit.score_runner()
    assert "No recorded trajectories" in out and "sim.refine" in out


def test_probe_score_delegates_to_the_provider() -> None:
    """sim.score delegates to ctx.probe_score_provider and raises without one
    (solve sessions and residual-arm learn sessions)."""
    utils.reset_config({})
    ctx = ToolContext()
    sim = BeliefProbe(ctx)
    try:
        sim.score()
    except RuntimeError as e:
        assert "unavailable in this session" in str(e)
    else:  # pragma: no cover
        raise AssertionError("sim.score must raise without a provider")
    calls: dict = {}

    def _provider(path=None, traj_idxs=None, num_particles=None) -> str:
        calls.update(path=path,
                     traj_idxs=traj_idxs,
                     num_particles=num_particles)
        return "[cycle_000_vers_000] score report"

    ctx.probe_score_provider = _provider
    assert sim.score(traj_idxs=[1], num_particles=4) == \
        "[cycle_000_vers_000] score report"
    assert calls == {"path": None, "traj_idxs": [1], "num_particles": 4}
