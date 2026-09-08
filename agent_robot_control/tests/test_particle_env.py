"""ParticleEnv + SB3 backend on the live session (short, CPU)."""
import numpy as np
import pytest

from agent_robot_control.rl.backend import RLRequest
from agent_robot_control.rl.particle_env import ParticleEnv
from agent_robot_control.rl.reward_loader import load_reward
from agent_robot_control.rl.sb3_backend import SB3Backend
from agent_robot_control.sim.session import SessionConfig, SimSession

REWARD = """
def reward(particles, visible, ee_pos, ee_quat, gripper):
    d = particles["donut_0"]
    if len(d) == 0:
        return -1.0
    # Move the gripper above the donut: negative horizontal distance.
    return -float(np.linalg.norm(ee_pos[:2] - d.mean(0)[:2]))
"""


@pytest.fixture(scope="module")
def session(tmp_path_factory):
    run_dir = tmp_path_factory.mktemp("run")
    cfg = SessionConfig(env_name="pybullet_donut", interaction_cap=5000,
                        camera_width=224, camera_height=126,
                        log_transitions=True, transition_shard_size=50,
                        run_dir=str(run_dir))
    s = SimSession(cfg)
    yield s
    s.close()


def _request(session, budget, **kw):
    return RLRequest(reward_fn=load_reward(REWARD), reward_source=REWARD,
                     budget_interactions=budget, episode_length=10,
                     points_per_object=8, out_dir=session.new_rl_dir(),
                     early_stop_successes=2, early_stop_window=3,
                     final_exec_attempts=1, **kw)


def test_particle_env_step_counts_interactions(session):
    env = ParticleEnv(session, _request(session, 100))
    before = session.interactions
    obs, info = env.reset()
    assert obs.shape == env.observation_space.shape
    for _ in range(5):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        assert np.isfinite(r)
    assert session.interactions - before >= 5
    # EE stayed inside the workspace box.
    assert np.all(np.abs(session.controller.ee_position() - env.anchor_pos)
                  <= env.half + 0.02)


def test_sac_backend_runs_within_budget(session):
    before = session.interactions
    req = _request(session, 60, algo="sac",
                   algo_kwargs=dict(learning_starts=20, batch_size=16))
    res = SB3Backend().run(session, req)
    used = session.interactions - before
    # Training budget plus pseudo-resets and the final execution.
    assert 60 <= used <= 60 + 10 * 2 + 60 * 2
    assert res.episodes >= 1
    assert res.policy_path is not None
    assert (session.run_dir / "events.jsonl").exists()


def test_ppo_backend_runs(session):
    req = _request(session, 40, algo="ppo", algo_kwargs=dict(n_steps=16, batch_size=8))
    res = SB3Backend().run(session, req)
    assert res.episodes >= 1


def test_transitions_flushed(session):
    session.flush_transitions()
    shards = list((session.run_dir / "transitions").glob("shard_*.npz"))
    assert shards
    d = np.load(shards[0], allow_pickle=True)
    assert d["points_t"].shape[0] == d["points_t1"].shape[0]
    assert d["joint_action"].shape[1] == 9


def test_drop_aborts_rl_call(tmp_path_factory):
    """Holding an object at anchor time and losing it ends the RL call.

    Uses the plug domain: the push domain's discs are deliberately too wide to
    grasp, so nothing there can be dropped.
    """
    from agent_robot_control.rl.particle_env import ParticleEnv
    run_dir = tmp_path_factory.mktemp("drop")
    plug_session = SimSession(SessionConfig(env_name="pybullet_plug_outlet",
                                            interaction_cap=4000,
                                            camera_width=224, camera_height=126,
                                            log_transitions=False,
                                            run_dir=str(run_dir)))
    ctl = plug_session.controller
    env0 = plug_session.env
    st = env0._current_observation
    plug = env0._plug
    px, py, pz = [float(st.get(plug, f)) for f in "xyz"]
    q = ctl.quat_from_rpy_deg(0, 0, 0)
    ctl.move_to((px, py, pz + 0.12), q, gripper="open")
    ctl.move_to((px, py, pz + 0.004), q)
    ctl.move_to((px, py, pz + 0.004), q, gripper="close")
    ctl.move_to((px, py, pz + 0.12), q)
    assert ctl.is_holding()
    req = RLRequest(reward_fn=load_reward(REWARD.replace("donut_0", "plug")),
                    reward_source=REWARD, budget_interactions=200,
                    episode_length=10, points_per_object=8,
                    out_dir=plug_session.new_rl_dir(), control_gripper=True,
                    video_every=0)
    env = ParticleEnv(plug_session, req)
    assert env.anchor_holding
    env.reset()
    term = False
    for _ in range(6):
        _obs, _r, term, _trunc, info = env.step(np.array([0.0, 0.0, 0.0, -1.0]))
        if term:
            break
    assert term and info["dropped"] and env.dropped
    plug_session.close()
