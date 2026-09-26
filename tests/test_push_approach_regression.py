"""Physical regression for a push approach captured in the Fan ramp runs."""
# pylint: disable=protected-access
import numpy as np
import pytest

from predicators import utils
from predicators.envs import create_new_env
from predicators.ground_truth_models import get_gt_options


@pytest.mark.parametrize("seed,joints", [
    (2, [
        -0.3416349772, -1.1232539341, 0.5704655956, 2.1287703665,
        -0.3878965787, 0.6655542265, -1.0766548594, 0.0351365195, 0.0351375430
    ]),
    (4, [
        -0.2605468716, -1.1572704293, 0.4355447957, 2.1416461476,
        -0.2871982415, 0.6408991236, -1.1956921478, 0.0351366559, 0.0351373480
    ]),
])
def test_push_approach_from_recorded_arm(seed, joints):
    """A valid push must not jam on its non-contact approach descent.

    Arm configurations are from lower-drop seeds 2 (step 177) and 4
    (step 336). Only the arm state is needed to reproduce the failure;
    the moving ball's state does not participate in switch manipulation.
    """
    utils.reset_config({
        "env": "pybullet_fan",
        "seed": seed,
        "num_train_tasks": 1,
        "num_test_tasks": 1,
        "skill_phase_use_motion_planning": True,
        "pybullet_ik_validate": False,
        "pybullet_birrt_path_subsample_ratio": 2,
        "fan_exposed_transfer": True,
        "fan_inertial_transfer": True,
        "fan_ramp_transfer": True,
        "fan_ramp_rise": 0.003,
        "fan_ramp_landing_extension": 0.1,
        "fan_train_num_walls_per_task": [0],
        "fan_test_num_walls_per_task": [0],
    })
    env = create_new_env("pybullet_fan", do_cache=False)
    try:
        env.reset("test", 0)
        env._pybullet_robot.set_joints(joints)
        state = env._get_state()
        objects = {obj.name: obj for obj in state}
        option = next(opt for opt in get_gt_options("pybullet_fan")
                      if opt.name == "SwitchOn")
        grounded = option.ground([objects["robot"], objects["fan_1"]],
                                 np.array([0.05, 0.10], dtype=np.float32))
        assert grounded.initiable(state)
        for _ in range(60):
            if grounded.terminal(state):
                break
            state = env.step(grounded.policy(state))
        else:
            pytest.fail("Switch approach did not complete within 60 actions")
        assert state.get(objects["fan_1"], "is_on") > 0.5
    finally:
        env.dispose()
