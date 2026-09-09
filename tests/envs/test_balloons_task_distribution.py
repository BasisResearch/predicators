"""Check explicit original sampling without restoring recorded task
fixtures."""
# pylint: disable=protected-access
import pytest

from predicators import utils
from predicators.envs.pybullet_balloons import PyBulletBalloonsEnv


@pytest.mark.parametrize("generation,scene", [("unknown", "chute"),
                                              ("original", "hatch")])
def test_invalid_distribution(generation, scene):
    """Reject unsupported task versions and incompatible geometry."""
    utils.reset_config({
        "env": "pybullet_balloons",
        "balloons_task_generation": generation,
        "balloons_scene": scene
    })
    with pytest.raises(ValueError):
        PyBulletBalloonsEnv(use_gui=False)


def test_original_reference_selection():
    """The original ambiguous task retains its historical reference subset."""
    utils.reset_config({
        "env": "pybullet_balloons",
        "seed": 1,
        "balloons_scene": "chute",
        "balloons_task_generation": "original"
    })
    env = PyBulletBalloonsEnv(use_gui=False)
    try:
        state = env.level_state(1, [0, 1, 2, 3],
                                (.5051171428571427, .5551171428571428))
        assert env.solution_subset(state) == (3, )
    finally:
        env.dispose()
