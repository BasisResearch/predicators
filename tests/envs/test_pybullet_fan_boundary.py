"""The fan env's boundary slabs are read through the env-owned Objects.

Regression for the 2026-09-03 fan test task: the belief probe's env
received States produced by the real env, whose boundary Objects carried
that env's body ids. Every slab rebuild reassigns ids in reverse, so the
foreign ids named the wrong slabs here and the four boundaries came back
permuted (a cross through the arena in every rendering).
"""
import pybullet as p

from predicators import utils
from predicators.envs.pybullet_fan import PyBulletFanEnv


def test_boundary_state_survives_foreign_object_ids() -> None:
    """A second env reading a State produced by the first returns the boundary
    poses that State holds, even after its own slabs were rebuilt with
    different body ids."""
    utils.reset_config({
        "env": "pybullet_fan",
        "seed": 0,
        "num_train_tasks": 1,
        "num_test_tasks": 1,
    })
    producer = PyBulletFanEnv(use_gui=False)
    consumer = PyBulletFanEnv(use_gui=False)
    try:
        # The producer's slabs are rebuilt (train grid -> test grid), so
        # its boundary ids no longer follow creation order.
        producer.reset("train", 0)
        producer.reset("test", 0)
        obs = producer.get_observation()
        expected = {
            b.name: tuple(round(float(obs.get(b, f)), 4) for f in ("x", "y"))
            for b in producer._boundaries  # pylint: disable=protected-access
        }
        assert len(set(expected.values())) == 4
        # The consumer builds its slabs fresh from that State, then reads
        # them back through the foreign Object instances.
        consumer._set_state(obs)  # pylint: disable=protected-access
        got_state = consumer._get_state()  # pylint: disable=protected-access
        got = {
            b.name:
            tuple(round(float(got_state.get(b, f)), 4) for f in ("x", "y"))
            for b in consumer._boundaries  # pylint: disable=protected-access
        }
        assert got == expected
    finally:
        for env in (producer, consumer):
            p.disconnect(env._physics_client_id)  # pylint: disable=protected-access
