"""Tests for the option-level program world model (paper arm C4)."""
# pylint: disable=protected-access
from typing import Any

import numpy as np
import pytest

from predicators import utils
from predicators.code_sim_learning.program_world_model import \
    EXCEPTION_DISTANCE, ProgramOptionModel, ProgramWorldModel, \
    feature_scales, load_program_world_model, observation_distance, \
    option_transitions, roll_program_latents, score_program
from predicators.datasets import create_dataset
from predicators.envs import create_new_env
from predicators.ground_truth_models import get_gt_options

_HAND_PROGRAM = '''
LATENT_FEATURES = {"robot": ["count"]}

def initial_latent(obs, rng):
    return {"count": int(rng.integers(0, 2))}

def transition(obs, latent, option, rng):
    nxt = obs.copy()
    for obj in nxt:
        if obj.type.name == "robot":
            nxt.set(obj, "hand", float(option.params[0]))
    return nxt, {"count": latent["count"] + 1}, 5
'''

_NOOP_PROGRAM = '''
LATENT_FEATURES = {}

def initial_latent(obs, rng):
    return {}

def transition(obs, latent, option, rng):
    return obs.copy(), latent, 1
'''

_RAISING_PROGRAM = '''
def initial_latent(obs, rng):
    return {}

def transition(obs, latent, option, rng):
    raise RuntimeError("boom")
'''


def _cover():
    utils.reset_config({
        "env": "cover",
        "num_train_tasks": 2,
        "num_test_tasks": 1,
    })
    env = create_new_env("cover")
    train_tasks = [t.task for t in env.get_train_tasks()]
    options = get_gt_options(env.get_name())
    dataset = create_dataset(env, train_tasks, options, env.predicates)
    return env, train_tasks, options, dataset


def _load(code: str, env: Any, options: Any) -> ProgramWorldModel:
    program, err = load_program_world_model(code, env.types, env.predicates,
                                            options)
    assert err is None, err
    assert program is not None
    return program


def test_load_program_validates_exports() -> None:
    """Missing exports and a malformed LATENT_FEATURES are reported."""
    env, _, options, _ = _cover()
    program = _load(_HAND_PROGRAM, env, options)
    assert program.latent_features == {"robot": ["count"]}
    _, err = load_program_world_model("x = 1", env.types, env.predicates,
                                      options)
    assert err is not None
    _, err = load_program_world_model("def transition(): pass", env.types,
                                      env.predicates, options)
    assert err is not None and "initial_latent" in err
    _, err = load_program_world_model(
        _NOOP_PROGRAM + "\nLATENT_FEATURES = 3\n", env.types, env.predicates,
        options)
    assert err is not None and "LATENT_FEATURES" in err


def _ground_pick_place(options: Any, param: float) -> Any:
    (pick_place, ) = [o for o in options if o.name == "PickPlace"]
    return pick_place.ground([], np.array([param], dtype=np.float32))


def test_program_option_model_steps_and_carries_latent() -> None:
    """Transitions write features, seed and advance the latent, honor the
    override, and never mutate the input state."""
    env, train_tasks, options, _ = _cover()
    model = ProgramOptionModel(_load(_HAND_PROGRAM, env, options), seed=0)
    init = train_tasks[0].init
    assert init.latent is None
    option = _ground_pick_place(options, 0.3)
    nxt, num_actions = model.get_next_state_and_num_actions(init, option)
    assert num_actions == 5
    robot = [o for o in nxt if o.type.name == "robot"][0]
    assert nxt.get(robot, "hand") == pytest.approx(0.3)
    # The latent was seeded from initial_latent (a draw in {0, 1}) and
    # advanced by the transition; the input state was not mutated.
    assert nxt.latent["count"] in (1, 2)
    assert init.latent is None
    nxt2, _ = model.get_next_state_and_num_actions(nxt, option)
    assert nxt2.latent["count"] == nxt.latent["count"] + 1
    # The override pins every latent-less start.
    model.initial_latent_override = {"count": 40}
    nxt3, _ = model.get_next_state_and_num_actions(init, option)
    assert nxt3.latent["count"] == 41
    model.initial_latent_override = None
    assert model.last_execution_failure is None


def test_program_option_model_reports_program_errors() -> None:
    """Program errors surface as OptionExecutionFailure with a diagnostic."""
    env, train_tasks, options, _ = _cover()
    init = train_tasks[0].init
    option = _ground_pick_place(options, 0.3)
    raising = ProgramOptionModel(_load(_RAISING_PROGRAM, env, options))
    with pytest.raises(utils.OptionExecutionFailure, match="boom"):
        raising.get_next_state_and_num_actions(init, option)
    assert raising.last_execution_failure is not None
    assert "boom" in raising.last_execution_failure
    bad_shape = _load(
        '''
def initial_latent(obs, rng):
    return {}

def transition(obs, latent, option, rng):
    return obs.copy(), latent
''', env, options)
    with pytest.raises(utils.OptionExecutionFailure, match="3-tuple"):
        ProgramOptionModel(bad_shape).get_next_state_and_num_actions(
            init, option)
    drops_object = _load(
        '''
def initial_latent(obs, rng):
    return {}

def transition(obs, latent, option, rng):
    nxt = obs.copy()
    del nxt.data[list(nxt)[0]]
    return nxt, latent, 1
''', env, options)
    with pytest.raises(utils.OptionExecutionFailure,
                       match="exactly the objects"):
        ProgramOptionModel(drops_object).get_next_state_and_num_actions(
            init, option)


def test_option_transitions_and_scales() -> None:
    """Option-level segmentation covers the trajectory; distances are in
    feature-std units."""
    env, _, _, dataset = _cover()
    traj = dataset.trajectories[0]
    transitions = option_transitions(traj, env.predicates)
    assert transitions
    assert sum(n for _, _, _, n in transitions) == len(traj.actions)
    assert transitions[0][0] is traj.states[0]
    assert transitions[-1][2] is traj.states[-1]
    scales = feature_scales(dataset.trajectories)
    assert all(v > 0 for v in scales.values())
    assert ("robot", "hand") in scales
    # A state is at distance 0 from itself and > 0 from a perturbed copy.
    state = traj.states[0]
    assert observation_distance(state, state, scales) == 0.0
    moved = state.copy()
    robot = [o for o in moved if o.type.name == "robot"][0]
    moved.set(robot, "hand", moved.get(robot, "hand") + 1.0)
    assert observation_distance(moved, state, scales) > 0.0


def test_score_prefers_the_better_program_and_reports_errors() -> None:
    """The kernel pseudo-likelihood ranks programs and charges exceptions."""
    env, _, options, dataset = _cover()
    trajs = dataset.trajectories
    rng = np.random.default_rng(0)
    noop = score_program(_load(_NOOP_PROGRAM, env, options),
                         trajs,
                         env.predicates,
                         num_particles=4,
                         kernel_bandwidth=0.2,
                         rng=rng)
    hand = score_program(_load(_HAND_PROGRAM, env, options),
                         trajs,
                         env.predicates,
                         num_particles=4,
                         kernel_bandwidth=0.2,
                         rng=rng)
    # In cover, PickPlace moves the hand to its parameter, which the
    # hand program predicts and the no-op program does not.
    assert hand.total > noop.total
    assert hand.total <= 0.0
    assert len(hand.per_trajectory) == len(trajs)
    assert all(n > 0 for _, _, n in hand.per_trajectory)
    assert "robot.hand" in noop.feature_errors
    assert noop.feature_errors["robot.hand"][0] > hand.feature_errors[
        "robot.hand"][0]
    text = hand.render()
    assert "World-model score" in text and "Per-feature error" in text
    assert not hand.exceptions
    # A raising program is charged the exception distance on every step.
    raising = score_program(_load(_RAISING_PROGRAM, env, options),
                            trajs,
                            env.predicates,
                            num_particles=2,
                            kernel_bandwidth=0.2,
                            rng=rng)
    n_steps = sum(n for _, _, n in raising.per_trajectory)
    assert raising.total == pytest.approx(-EXCEPTION_DISTANCE * n_steps)
    assert raising.exceptions and "boom" in raising.exceptions[0]
    assert "RAISED" in raising.render()
    # Subsetting scores only the selected trajectories.
    subset = score_program(_load(_HAND_PROGRAM, env, options),
                           trajs,
                           env.predicates,
                           num_particles=2,
                           kernel_bandwidth=0.2,
                           rng=rng,
                           traj_indices={0})
    assert [i for i, _, _ in subset.per_trajectory] == [0]


def test_roll_program_latents_aligns_with_states() -> None:
    """Per-state latents align with the trajectory and degrade to None."""
    env, _, options, dataset = _cover()
    traj = dataset.trajectories[0]
    latents = roll_program_latents(_load(_HAND_PROGRAM, env, options), traj,
                                   env.predicates, np.random.default_rng(0))
    assert len(latents) == len(traj.states)
    assert all(l is not None for l in latents)
    counts = [l["count"] for l in latents if l is not None]
    assert counts[0] in (0, 1)
    assert counts[-1] == counts[0] + len(
        option_transitions(traj, env.predicates))
    assert counts == sorted(counts)
    broken = roll_program_latents(_load(_RAISING_PROGRAM, env, options), traj,
                                  env.predicates, np.random.default_rng(0))
    assert len(broken) == len(traj.states)
    assert broken[0] is not None and broken[-1] is None
