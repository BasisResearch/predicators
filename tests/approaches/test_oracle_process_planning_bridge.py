"""End-to-end test: oracle_process_planning solves a bridge task.

Mirrors the config from ``predicatorv3/oracle.yaml`` +
``predicatorv3/envs/all.yaml`` (bridge entry) + ``predicatorv3/common.yaml``
so that a regression in the approach (process planning + bilevel
refinement), the bridge env's glue/weld machinery, or the skill
factories would surface here.

Runs the smallest viable config (1 train task, 1 test task; 5 blocks,
2 glue joints) and asserts:

  - The approach returns a policy (no ApproachTimeout / ApproachFailure).
  - Executing the policy in the env reaches ``task.goal_holds`` within
    the configured horizon (the finished n-bridge: legs at both sites,
    the welded 3-block span seated across the leg tops).
"""
# pylint: disable=protected-access
from __future__ import annotations

import logging

import pytest

import predicators.approaches  # noqa: F401  # pylint: disable=unused-import
import predicators.envs  # noqa: F401  # pylint: disable=unused-import
import predicators.ground_truth_models  # noqa: F401  # pylint: disable=unused-import
from predicators import utils
from predicators.approaches import create_approach
from predicators.envs import create_new_env
from predicators.ground_truth_models import get_gt_options
from predicators.settings import CFG

logger = logging.getLogger(__name__)


def _oracle_bridge_config() -> dict:
    """Flags from predicatorv3/{common,envs/all,oracle}.yaml flattened."""
    return {
        # --- env: bridge from envs/all.yaml ---
        "env": "pybullet_bridge",
        "horizon": 3000,
        # Execute every planned waypoint: subsampled execution cuts
        # corners the plan cleared, and a carried span grazing a
        # standing 2:1 leg by a fraction of a mm topples it.
        "pybullet_birrt_path_subsample_ratio": 1,
        # The packed staging grid leaves ~1-2 cm clearances that
        # stochastically dip into 2-3 mm grazes (e.g. a welded row's
        # frozen offsets against a seat leg at the descend goal); the
        # default 1 mm margin turns those into unrecoverable BiRRT
        # start/goal rejections.
        "pybullet_birrt_contact_margin": -0.005,
        # Each Wait ends on the FIRST atom change, so a plan waiting on
        # several concurrent cures can need a cheap replan for the tail
        # (which reduces to "Wait until the remaining joint cures").
        "process_planning_max_execution_replans": 3,
        "wait_option_max_steps": 120,
        # --- common flags relevant to bilevel refinement ---
        "skill_phase_use_motion_planning": True,
        # Bridge follows common.yaml: validated IK OFF, like every
        # other domain. Validation is not what enforces placement
        # accuracy (goal-IK branches are accepted on forward-kinematics
        # error either way) and it steered the arm into
        # worse-executing IK branches (oracle sweep: 22/24 with it on,
        # 24/24 with it off).
        "pybullet_ik_validate": False,
        "planning_filter_unreachable_nsrt": False,
        "no_repeated_arguments_in_grounding": True,
        "terminate_on_goal_reached": False,
        "sesame_check_expected_atoms": False,
        # --- approach: oracle_process_planning from oracle.yaml ---
        "approach": "oracle_process_planning",
        "demonstrator": "oracle_process_planning",
        "terminate_on_goal_reached_and_option_terminated": True,
        "bilevel_plan_without_sim": True,
        # --- test scope: keep it small ---
        "num_train_tasks": 1,
        "num_test_tasks": 1,
        "seed": 0,
        "use_gui": False,
        "option_model_use_gui": False,
        "option_model_terminate_on_repeat": False,
        "wait_option_terminate_on_atom_change": True,
    }


@pytest.mark.xfail(
    reason="The pre-hardening GT-sim pipeline is not reliable enough for CI\n"
    "yet; the hardening changes stacked on this commit make the solve\n"
    "deterministic and drop this marker.",
    strict=False)
def test_oracle_process_planning_solves_bridge_task():
    """Smoke test: oracle_process_planning builds the simple n-bridge."""
    utils.reset_config(_oracle_bridge_config())
    env = create_new_env("pybullet_bridge", do_cache=False, use_gui=False)
    options = get_gt_options(env.get_name())
    train_tasks = [t.task for t in env.get_train_tasks()]

    approach = create_approach(
        CFG.approach,
        env.predicates,
        options,
        env.types,
        env.action_space,
        train_tasks,
    )

    test_task = env.get_test_tasks()[0].task

    policy = approach.solve(test_task, timeout=CFG.timeout)
    assert policy is not None, "oracle_process_planning returned no policy"

    env.reset("test", 0)
    for step in range(CFG.horizon):
        if test_task.goal_holds(env._current_state):
            logger.info("Goal reached after %d env steps.", step)
            return
        action = policy(env._current_state)
        env.step(action)
    assert test_task.goal_holds(env._current_state), (
        f"Policy executed for {CFG.horizon} steps but goal not reached. "
        f"Final state predicates: "
        f"{utils.abstract(env._current_state, env.predicates)}; "
        f"required goal: {test_task.goal}")
