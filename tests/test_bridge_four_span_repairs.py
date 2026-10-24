"""Regression checks for the four-span Bridge repairs: the lift-first transit,
the recorded post-certificate scene and the robot-clearance gate on the
certificate."""
# pylint: disable=protected-access
import numpy as np
import pybullet as p

from predicators import utils
from predicators.envs import create_new_env
from predicators.envs.pybullet_bridge import PyBulletBridgeEnv
from predicators.ground_truth_models import get_gt_options
from predicators.ground_truth_models.bridge.processes import \
    _PALM_LEG_OVERLAP, _stage_spot_sampler
from predicators.run.episode import EpisodeRunner, EpisodeState
from predicators.structs import Action, EnvironmentTask


def _phase_names(option):
    return [
        phase.name for phase in
        option.policy.__self__._phases  # type: ignore[attr-defined]
    ]


def test_bridge_place_lifts_before_the_transit_when_enabled():
    """``bridge_lift_before_transit`` prepends a straight lift to Place; off,
    Place starts with the transit as before."""
    for enabled in (False, True):
        utils.reset_config({
            "env": "pybullet_bridge",
            "num_train_tasks": 0,
            "num_test_tasks": 0,
            "bridge_lift_before_transit": enabled,
        })
        place = next(o for o in get_gt_options("pybullet_bridge")
                     if o.name == "Place")
        names = _phase_names(place)
        assert names[:2] == (["LiftBeforeTransit", "MoveAbove"]
                             if enabled else ["MoveAbove", "Descend"])


def test_certificate_result_is_the_recorded_scene(monkeypatch):
    """A certificate that moves the scene must leave the runner (and every
    listener) holding the scene it judged, not the candidate before it."""
    utils.reset_config({
        "env": "cover",
        "num_train_tasks": 1,
        "num_test_tasks": 0
    })
    env = create_new_env("cover", do_cache=False)
    runner = EpisodeRunner(env, horizon=10)
    runner.reset("train", 0)
    block = next(o for o in runner.observation() if o.type.name == "block")
    monkeypatch.setattr(env, "episode_terminated", lambda observations: True)

    def reject(observations, actions):
        del observations, actions
        env._current_observation = env._current_observation.copy()
        env._current_state.set(block, "pose", 0.123)
        return False, "settled out of goal"

    monkeypatch.setattr(env, "check_episode_trajectory", reject)
    observed = []
    runner.add_step_listener(lambda action, outcome: observed.append(outcome))
    outcome = runner.step(Action(np.array([0.5], dtype=np.float32)))
    assert outcome.state is EpisodeState.GAME_OVER
    assert abs(outcome.observation.get(block, "pose") - 0.123) < 1e-6
    assert observed[-1].observation is runner.observation()
    assert runner.num_steps == 1


def test_bridge_certificate_waits_for_robot_clearance():
    """Candidate geometry alone does not end the episode while a robot link is
    still within ``bridge_goal_robot_clearance`` of a block."""
    utils.reset_config({
        "env": "pybullet_bridge",
        "num_train_tasks": 1,
        "num_test_tasks": 0,
        "bridge_goal_robot_clearance": 0.01
    })
    env = PyBulletBridgeEnv(use_gui=False)
    try:
        obs = env.reset("train", 0)
        assert "0.01 m away" in env._current_task.goal_nl
        # A vacuous goal isolates the physical clearance gate.
        env._current_task = EnvironmentTask(obs, set())
        client = env._physics_client_id
        robot = env._pybullet_robot.robot_id
        block = env._blocks[0].id
        assert block is not None
        link = env._pybullet_robot.end_effector_id
        position = p.getLinkState(robot, link, physicsClientId=client)[0]
        p.resetBasePositionAndOrientation(block,
                                          position, [0, 0, 0, 1],
                                          physicsClientId=client)
        assert not env.episode_terminated([obs])
        p.resetBasePositionAndOrientation(block, [5, 5, 5], [0, 0, 0, 1],
                                          physicsClientId=client)
        assert env.episode_terminated([obs])
    finally:
        env.dispose()


def test_parking_spots_keep_clear_of_the_sites():
    """The staging escape hatch never parks an object in a mid- or back-row
    cell behind a site, whether or not a leg stands there yet, and keeps a
    palm-wide berth from any standing block: a bottle parked 7 cm from a free
    site was ungraspable once the leg stood there (seed 1)."""
    utils.reset_config({
        "env": "pybullet_bridge",
        "seed": 1,
        "num_train_tasks": 0,
        "num_test_tasks": 1,
        "bridge_train_span_blocks": 3,
        "bridge_test_span_blocks": 4,
    })
    env = PyBulletBridgeEnv(use_gui=False)
    try:
        state = env._generate_test_tasks()[0].init
        sites = state.get_objects(env._site_type)
        leg0 = next(b for b in state.get_objects(env._block_type)
                    if b.name == "leg0")
        # One leg already standing at site0, the other site still free.
        stood = state.copy()
        stood.set(leg0, "x", state.get(sites[0], "x"))
        stood.set(leg0, "y", state.get(sites[0], "y"))
        for s in (state, stood):
            for seed in range(40):
                x, y = _stage_spot_sampler(s, env._bottle,
                                           np.random.default_rng(seed))
                if y > env.stage_row_front + 0.02:
                    for site in sites:
                        assert abs(x - s.get(site, "x")) >= \
                            _PALM_LEG_OVERLAP - env.stage_jitter, (x, y)
                assert np.hypot(x - s.get(leg0, "x"),
                                y - s.get(leg0, "y")) >= 0.09, (x, y)
    finally:
        env.dispose()


def test_certificate_handles_the_pooled_fourth_span():
    """In a transfer run (three-span training, four-span test) the body pool
    holds a fourth span that the training task's state omits; the certificate,
    its snapshot and the clearance gate must read only the task's blocks (the
    pooled span raised KeyError on the step that completed the training bridge,
    Opus MB, Sept 16)."""
    utils.reset_config({
        "env": "pybullet_bridge",
        "seed": 0,
        "num_train_tasks": 1,
        "num_test_tasks": 1,
        "bridge_train_span_blocks": 3,
        "bridge_test_span_blocks": 4,
        "bridge_goal_robot_clearance": 0.01,
    })
    env = PyBulletBridgeEnv(use_gui=False)
    try:
        obs = env.reset("train", 0)
        assert len(obs.get_objects(env._block_type)) == 5
        assert len(env._blocks) == 6
        assert {b.name for b in env._task_blocks()} == \
            {b.name for b in obs.get_objects(env._block_type)}
        snapshot = env._certificate_snapshot()
        assert "span3" not in snapshot["poses"]
        assert set(snapshot["poses"]) == {b.name for b in env._task_blocks()}
        # A vacuous goal holds, so this exercises the whole settle path.
        env._current_task = EnvironmentTask(obs, set())
        assert env.episode_terminated([obs])
        ok, reason = env.check_episode_trajectory([obs], [])
        assert ok, reason
    finally:
        env.dispose()
