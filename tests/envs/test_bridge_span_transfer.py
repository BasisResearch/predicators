"""Mechanical tests of the three-to-four block Bridge task distribution."""
# pylint: disable=protected-access,import-outside-toplevel
from typing import Any

import numpy as np
import pybullet as p
import pytest

from predicators import utils
from predicators.envs import create_new_env
from predicators.structs import Action, EnvironmentTask


@pytest.mark.parametrize('seed', [0, 1, 2, 3])
def test_transfer_rosters(seed: int) -> None:
    """Both splits coexist in one body pool: a three-span train task and a
    four-span test task alternate on the same env, each with its own site
    separation, and a fresh base-physics world accepts either state."""
    utils.reset_config({
        'env': 'pybullet_bridge',
        'seed': seed,
        'num_train_tasks': 1,
        'num_test_tasks': 1,
        'bridge_train_span_blocks': 3,
        'bridge_test_span_blocks': 4,
        'partially_observable': True
    })
    env: Any = create_new_env('pybullet_bridge', do_cache=False)
    model: Any = create_new_env('pybullet_bridge',
                                do_cache=False,
                                skip_residual_dynamics=True)
    try:
        for split, count in [('train', 3), ('test', 4), ('train', 3)]:
            state = env.reset(split, 0)
            blocks = [o for o in state if o.type.name == 'block']
            assert len(blocks) == count + 2
            sites = sorted(o for o in state if o.type.name == 'site')
            assert np.isclose(
                state.get(sites[1], 'x') - state.get(sites[0], 'x'),
                count * .1 - .05)
            model._set_state(state)
            observed = model._get_state()
            assert set(state) == set(observed)
            for obj in blocks:
                assert abs(state.get(obj, 'x') - observed.get(obj, 'x')) < .001
            model.simulate(
                state,
                Action(
                    np.array(model._pybullet_robot.get_joints(),
                             dtype=np.float32)))
    finally:
        p.disconnect(env._physics_client_id)
        p.disconnect(model._physics_client_id)


@pytest.mark.parametrize('count', [3, 4])
def test_transfer_structure_requires_welds(count: int) -> None:
    """The wider bridge stands with welds and collapses without them."""
    utils.reset_config({
        'env': 'pybullet_bridge',
        'seed': 0,
        'num_train_tasks': 1,
        'num_test_tasks': 1,
        'bridge_train_span_blocks': 3,
        'bridge_test_span_blocks': count,
        'partially_observable': False
    })
    env: Any = create_new_env('pybullet_bridge', do_cache=False)
    task = env.get_test_tasks()[0]
    try:
        for welded in [False, True]:
            state = task.init.copy()
            sx = [state.get(site, 'x') for site in env._sites]
            y = state.get(env._sites[0], 'y')
            spans = env._spans[:count]
            for leg, x in zip(env._legs, sx):
                for feat, value in [('x', x), ('y', y), ('z', .45),
                                    ('roll', 0.), ('pitch', -np.pi / 2),
                                    ('yaw', 0.)]:
                    state.set(leg, feat, value)
            for i, block in enumerate(spans):
                for feat, value in [('x',
                                     np.mean(sx) + (i - (count - 1) / 2) * .1),
                                    ('y', y), ('z', .525), ('roll', 0.),
                                    ('pitch', 0.), ('yaw', 0.)]:
                    state.set(block, feat, value)
            if welded:
                for left, right in zip(spans, spans[1:]):
                    state.set(left, 'attached_end_b',
                              float(env._block_index[right.name]))
                    state.set(right, 'attached_end_a',
                              float(env._block_index[left.name]))
            assert task.task.goal_holds(state)
            env._set_state(state)
            env._current_task = EnvironmentTask(state, task.goal)
            ok, reason = env.check_episode_trajectory([state], [])
            assert ok == welded, (count, welded, reason)
    finally:
        p.disconnect(env._physics_client_id)


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_four_span_staging_is_pickable_with_public_skills(seed: int) -> None:
    """All follow-up layouts remain reachable by the shared controller."""
    from predicators.ground_truth_models import get_gt_options
    from predicators.ground_truth_models.skill_factories.base import \
        _SHARED_SIMULATOR_CACHE
    from predicators.run.episode import EpisodeRunner
    utils.reset_config({
        'env': 'pybullet_bridge',
        'seed': seed,
        'num_train_tasks': 1,
        'num_test_tasks': 1,
        'bridge_train_span_blocks': 3,
        'bridge_test_span_blocks': 4,
        'partially_observable': True,
        'skill_phase_use_motion_planning': True,
        'pybullet_ik_validate': False,
        'pybullet_birrt_contact_margin': -.005,
        'pybullet_pin_held_weld_assemblies': True
    })
    # Cache the env: get_gt_options builds the skills from the cached env's
    # types, and this env's partially-observable block type differs from
    # the block type of an env an earlier test left in the cache.
    env: Any = create_new_env('pybullet_bridge', do_cache=True)
    _SHARED_SIMULATOR_CACHE.pop(type(env), None)
    options = {o.name: o for o in get_gt_options('pybullet_bridge')}
    try:
        runner = EpisodeRunner(env, horizon=2000, max_option_steps=1000)
        objects = [
            o for o in env.get_test_tasks()[0].init
            if o.type.name in ('block', 'bottle')
        ]
        for obj in objects:
            runner.reset('test', 0)
            name = 'PickBlock' if obj.type.name == 'block' else 'PickBottle'
            option = options[name].ground([env._robot, obj], np.array([0.0]))
            result = runner.run_option(option)
            print('pickability',
                  obj.name,
                  result.status,
                  result.steps,
                  flush=True)
            assert result.status == "succeeded", (obj.name, result.reason)
    finally:
        p.disconnect(env._physics_client_id)
        _SHARED_SIMULATOR_CACHE.pop(type(env), None)
