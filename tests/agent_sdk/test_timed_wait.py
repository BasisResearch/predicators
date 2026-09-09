"""Explicit Wait lengths and proprioception through real continual tools."""
# pylint: disable=protected-access
import asyncio
import json
from typing import Any

import numpy as np
import pytest

from predicators import utils
from predicators.agent_sdk.tools.context import ToolContext
from predicators.agent_sdk.tools.continual_tools import PlayState, \
    build_continual_tools
from predicators.approaches import create_approach
from predicators.approaches.continual_play_mixin import ContinualPlayMixin
from predicators.envs import create_new_env
from predicators.ground_truth_models import get_gt_options
from predicators.run.continual import ContinualRun


@pytest.mark.parametrize("arm",
                         ["agent_continual", "agent_continual_model_free"])
def test_wait_lengths_and_current_joints(tmp_path: Any, arm: str) -> None:
    """Both arms see current joints before acting, after a step and on
    reset."""
    utils.reset_config({
        "env": "pybullet_boil",
        "approach": arm,
        "seed": 0,
        "num_train_tasks": 1,
        "num_test_tasks": 1,
        "boil_goal": "simple",
        "boil_num_jugs_train": [1],
        "boil_num_jugs_test": [1],
        "boil_num_burner_train": [1],
        "boil_num_burner_test": [1],
        "experiment_protocol": "continual",
        "continual_levels": "train_only",
        "continual_steps_per_level": 100,
        "continual_render": False,
        "continual_make_video": False,
        "continual_runs_dir": str(tmp_path / "runs"),
        "partially_observable": True,
        "continual_obs_noise_position": .01,
        "continual_obs_noise_orientation": .02,
        "wait_option_terminate_on_atom_change": True,
        "wait_option_max_steps": 4,
        "max_num_steps_option_rollout": 4,
    })
    env = create_new_env("pybullet_boil", do_cache=False)
    options = get_gt_options(env.get_name())
    approach = create_approach(arm, env.predicates, options, env.types,
                               env.action_space,
                               [t.task for t in env.get_train_tasks()])
    assert isinstance(approach, ContinualPlayMixin)
    tool_names = approach._continual_tool_names()
    ctx = ToolContext(types=set(env.types),
                      predicates=set(),
                      options=set(options),
                      env=env)

    class Driver:
        """Exercise the real session without an LLM request."""

        def play_level(self, session: Any) -> None:
            """Check the public tools and their charged step counts."""
            tools = {
                t.name: t
                for t in build_continual_tools(ctx,
                                               session,
                                               PlayState(),
                                               save_render=lambda _: None,
                                               tool_names=tool_names)
            }

            def call(name: str, **args: Any) -> str:
                result = asyncio.run(tools[name].handler(args))
                assert not result.get("is_error"), result
                return result["content"][0]["text"]

            def joints() -> Any:
                text = call("env_observe")
                control = json.loads(
                    text.split("[control] ")[1].split("\n")[0])
                q = control["joint_positions"]
                assert q == list(session.observe().frame.joint_positions)
                assert "simulator_state" not in control
                assert "privileged" not in control
                assert "joint_names" in control["action_space"]
                return q

            initial = joints()
            moved = list(initial)
            moved[0] += .05
            call("env_step", action=moved)
            assert not np.allclose(joints(), initial)
            call("env_reset", note="check fresh proprioception")
            np.testing.assert_allclose(joints(), initial, atol=1e-6)
            for count, expected in ((1, 1), (3, 3), (8, 4), (0, 4)):
                before = session.observe().ledger.run_steps
                response = call("skills_invoke",
                                skill=f"Wait(robot:robot)[{count}]")
                assert "succeeded" in response, response
                assert session.observe().ledger.run_steps - before == expected
            # Old plans still use the default (subgoal/backstop) behavior.
            before = session.observe().ledger.run_steps
            call("skills_invoke", skill="Wait(robot:robot)[]")
            assert session.observe().ledger.run_steps - before == 4
            before = session.observe().ledger.run_steps
            response = call("skills_execute_plan",
                            plan="Wait(robot:robot)[4]\nWait(robot:robot)[1]")
            assert "[2/2]" in response, response
            assert session.observe().ledger.run_steps - before == 5
            session.end_run("done")

    try:
        ContinualRun(env, approach, Driver()).run()
    finally:
        env.dispose()
