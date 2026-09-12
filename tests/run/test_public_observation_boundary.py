"""Public controller, recording, and continual goal regression tests."""
# pylint: disable=protected-access,import-outside-toplevel
from typing import Any

import numpy as np

from predicators import utils
from predicators.approaches.agent_model_free_approach import \
    AgentModelFreeApproach
from predicators.envs import create_new_env
from predicators.envs.pybullet_balloons import BalloonsEvaluator
from predicators.run.episode import EpisodeRunner, EpisodeState
from predicators.run.recording import sanitize_state
from predicators.structs import Action, GroundAtom, Object, State, Type


def test_recording_objects_have_only_public_schema() -> None:
    """A state exported to an agent cannot carry simulator metadata."""
    typ = Type('block', ['x'], sim_features=['id', 'cure_top'])
    obj = Object('block0', typ)
    obj.id = 42
    obj.cure_top = 17.0
    clean = sanitize_state(State({obj: np.array([0.5])}))
    copied = next(iter(clean))
    assert copied == obj
    assert copied.sim_data == {}
    assert not copied.type.sim_features
    assert obj.id == 42 and obj.cure_top == 17.0


def test_skill_reference_is_public_documentation() -> None:
    """The same baseline parent supplies API docs instead of implementation."""
    utils.reset_config({"env": "pybullet_bridge"})
    approach = object.__new__(AgentModelFreeApproach)
    references = approach._get_sandbox_reference_files()
    assert references
    assert all(path.endswith('.md') for path in references.values())
    assert 'options.py' not in references


def test_balloons_turning_point_does_not_end_episode(monkeypatch: Any) -> None:
    """Reproduce the recorded one-frame win through continual stepping."""
    utils.reset_config({
        'env': 'pybullet_balloons',
        'seed': 0,
        'num_train_tasks': 1,
        'num_test_tasks': 0
    })
    env: Any = create_new_env('pybullet_balloons', do_cache=False)
    initial = env.level_state(0, [0, 1], (0.75952, 0.80952))
    goal = {GroundAtom(env._InBand, [env._box, env._band])}
    from predicators.structs import EnvironmentTask
    env._train_tasks = [
        EnvironmentTask(initial, goal, evaluator=BalloonsEvaluator(goal))
    ]
    runner = EpisodeRunner(env, horizon=100)
    runner.reset('train', 0)

    def frame(height: float, speed: float) -> Any:
        obs = initial.copy()
        obs.set(env._box, 'z', height)
        obs.set(env._box, 'speed', speed)
        return obs

    frames = iter(
        [frame(.78642, .11596),
         frame(.78166, .00991),
         frame(.786, .08)])

    def step(_action: Action, **_kwargs: Any) -> Any:
        obs = next(frames)
        env._current_observation = obs
        return obs

    monkeypatch.setattr(env, 'step', step)
    action = Action(
        np.array(env._pybullet_robot.get_joints(), dtype=np.float32))
    runner.step(action)
    outcome = runner.step(action)
    assert outcome.state is EpisodeState.NOT_FINISHED
    assert not runner.evaluate().terminated
    assert not runner.evaluate().reward
    runner.step(action)
    assert runner.episode_state is EpisodeState.NOT_FINISHED
    import pybullet as p
    p.disconnect(env._physics_client_id)


def test_controller_error_does_not_reveal_hidden_attachment(
        monkeypatch: Any) -> None:
    """A failure reaches the real invocation API without hidden diagnostics."""
    from predicators.ground_truth_models import get_gt_options
    utils.reset_config({
        'env': 'cover',
        'seed': 0,
        'num_train_tasks': 1,
        'num_test_tasks': 1
    })
    env = create_new_env('cover', do_cache=False)
    runner = EpisodeRunner(env, horizon=100)
    runner.reset('train', 0)
    option = next(iter(get_gt_options('cover'))).ground([], np.array([.5]))

    def bad_policy(_state: State) -> Action:
        raise utils.OptionExecutionFailure(
            'GOAL: welded span1 within -0.0061 m of body 6 (table)')

    monkeypatch.setattr(utils, 'option_plan_to_policy',
                        lambda *a, **k: bad_policy)
    result = runner.run_option(option)
    assert result.status == 'failed' and result.steps == 0
    assert 'weld' not in result.reason and 'span1' not in result.reason
    assert '0.0061' not in result.reason and 'body 6' not in result.reason


def test_balloons_dwell_is_consecutive_and_bursts_are_terminal() -> None:
    """Brief slowdowns do not accumulate across motion; bursts still lose."""
    utils.reset_config({'balloons_goal_dwell_steps': 3})
    box = Object('box', Type('box', ['z', 'speed']))
    band = Object('band', Type('band', ['lo', 'hi']))
    balloon = Object('balloon', Type('balloon', ['popped']))
    from predicators.structs import Predicate
    pred = Predicate(
        'InBand', [box.type, band.type], lambda s, o: .5 <= s.get(o[0], 'z') <=
        .6 and s.get(o[0], 'speed') < .01)
    goal = {GroundAtom(pred, [box, band])}
    evaluator = BalloonsEvaluator(goal)
    good = State({
        box: np.array([.55, 0.]),
        band: np.array([.5, .6]),
        balloon: np.array([0.])
    })
    moving = good.copy()
    moving.set(box, 'speed', .1)
    assert not evaluator.terminated_trajectory([good] * 3)
    assert not evaluator.terminated_trajectory([good] * 3 + [moving] +
                                               [good] * 3)
    assert evaluator.solved([moving] + [good] * 4, None)
    burst = good.copy()
    burst.set(balloon, 'popped', 1.)
    assert evaluator.terminated_trajectory([burst])
    assert not evaluator.solved([burst], None)


def test_pilot_configs_parse_with_matched_arms(monkeypatch: Any) -> None:
    """The real CLI accepts exactly the two Bridge and three MF Balloons
    runs."""
    import shlex
    import sys

    from scripts.cluster_utils import config_to_cmd_flags, generate_run_configs
    bridge = list(
        generate_run_configs(
            'predicatorv3/protocol_continual_bridge_span_transfer_r1.yaml',
            False))
    balloons = list(
        generate_run_configs(
            'predicatorv3/protocol_continual_balloons_dwell_mf_r1.yaml',
            False))
    assert len(bridge) == 2 and len(balloons) == 3
    for cfg in bridge + balloons:
        monkeypatch.setattr(
            sys, 'argv',
            ['predicators/main.py', *shlex.split(config_to_cmd_flags(cfg))])
        parsed = utils.parse_args()
        assert parsed['env'] == cfg.env
        assert parsed['approach'] == cfg.approach
    assert {c.approach
            for c in bridge
            } == {'agent_continual', 'agent_continual_model_free'}
    for cfg in bridge:
        assert cfg.flags['bridge_train_span_blocks'] == 3
        assert cfg.flags['bridge_test_span_blocks'] == 4
        assert cfg.flags['continual_obs_noise_position'] == .005
        assert cfg.flags['continual_steps_per_level'] == 10000
    assert all(c.approach == 'agent_continual_model_free' for c in balloons)
    assert all(c.flags['balloons_goal_dwell_steps'] == 25 for c in balloons)
    assert all(not c.flags['balloons_require_jam_decoy'] for c in balloons)
