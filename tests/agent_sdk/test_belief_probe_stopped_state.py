"""A rehearsal step that fails partway: where the probe's state goes, what the
step report says it changed, and the caveat on a controller's motion failure
under observation noise."""
# pylint: disable=protected-access
from typing import Any, Optional

import numpy as np
import pytest
from gym.spaces import Box

from predicators import utils
from predicators.agent_sdk.belief_probe import BeliefProbe, ProbeJointResult, \
    _flag_changes, _motion_failures
from predicators.agent_sdk.plan_execution import execute_plan_forward
from predicators.agent_sdk.tools import ToolContext
from predicators.structs import Action, GroundAtom, LowLevelTrajectory, \
    Object, ParameterizedOption, Predicate, State, Task, Type
from tests.agent_sdk.test_belief_probe_joint import _joint_ctx, _on_observation
from tests.agent_sdk.test_belief_probe_physics_sweep import _PhysicsHoleModel

_lever_type = Type("lever", ["x", "is_on"])
_robot_type = Type("robot", ["x", "fingers"])
_lever = Object("lever0", _lever_type)
_robot = Object("robot", _robot_type)
_LeverOn = Predicate("LeverOn", [_lever_type],
                     lambda s, o: s.get(o[0], "is_on") > 0.5)
_STALL = ("[Push/Retreat] incremental-IK stalled: no end-effector progress "
          "in 25 steps; aborting option.")


def _noop_policy(_s, _m, _o, _p):
    return Action(np.zeros(1, dtype=np.float32))


_Push = ParameterizedOption(
    "Push",
    types=[_robot_type, _lever_type],
    params_space=Box(low=np.array([0.0], dtype=np.float32),
                     high=np.array([1.0], dtype=np.float32)),
    policy=_noop_policy,
    initiable=lambda _s, _m, _o, _p: True,
    terminal=lambda _s, _m, _o, _p: False,
)


class _RetreatStallModel:
    """Push flips the lever and moves the arm; with ``stall`` its retreat then
    stalls, after ``partial`` actions, as a skill's incremental IK does."""

    def __init__(self, stall: bool, partial: int = 7) -> None:
        self.stall = stall
        self.partial = partial
        self.last_execution_failure: Optional[str] = None
        self.last_partial_trajectory: Optional[LowLevelTrajectory] = None
        self.last_trajectory: Optional[LowLevelTrajectory] = None

    def get_next_state_and_num_actions(self, state, option):
        """The lever flips; the arm stops short of home when it stalls."""
        del option  # every Push flips the one lever
        self.last_execution_failure = None
        self.last_partial_trajectory = None
        nxt = state.copy()
        nxt.set(_lever, "is_on", 1.0)
        nxt.set(_robot, "x", 0.37)
        nxt.set(_robot, "fingers", 1.0)
        actions = [Action(np.zeros(1, dtype=np.float32))] * self.partial
        trajectory = LowLevelTrajectory([state] * self.partial + [nxt],
                                        actions)
        if self.stall:
            self.last_execution_failure = _STALL
            self.last_partial_trajectory = trajectory
            return nxt, 0
        self.last_trajectory = trajectory
        return nxt, self.partial


def _ctx(model: Any) -> ToolContext:
    init = State({
        _robot: np.array([0.75, 0.0], dtype=np.float32),
        _lever: np.array([0.2, 0.0], dtype=np.float32),
    })
    task = Task(init, {GroundAtom(_LeverOn, [_lever])})
    return ToolContext(
        types={_lever_type, _robot_type},
        predicates=set(),
        processes=set(),
        options={_Push},
        train_tasks=[task],
        example_state=init,
        option_model=model,
        current_task=task,
    )


_PLAN = "Push(robot:robot, lever0:lever)[0.5]"


def test_forward_execution_says_where_a_failed_step_stopped():
    """The plan's state stays before the failed step; ``stopped_state`` is
    where the step left the world, and a clean run stops at its end."""
    utils.reset_config({})
    ctx = _ctx(_RetreatStallModel(stall=True))
    task = ctx.current_task
    option = _Push.ground([_robot, _lever], np.array([0.5], dtype=np.float32))
    result = execute_plan_forward(task, [option],
                                  ctx.option_model,
                                  predicates={_LeverOn},
                                  stop_on_failure=True)
    assert result.first_failure_idx == 0
    assert result.final_state.get(_lever, "is_on") == 0.0
    assert result.stopped_state is not None
    assert result.stopped_state.get(_lever, "is_on") == 1.0
    clean = execute_plan_forward(task, [option],
                                 _RetreatStallModel(stall=False),
                                 predicates={_LeverOn},
                                 stop_on_failure=True)
    assert clean.first_failure_idx is None
    assert clean.stopped_state is clean.final_state


def test_a_failed_step_leaves_the_probe_where_it_stopped():
    """After a step that failed partway, sim.state() holds what the step did
    before it failed, and the report counts its actions and names the flags it
    changed."""
    utils.reset_config({
        "continual_obs_noise_position": 0.0,
        "continual_obs_noise_orientation": 0.0,
    })
    sim = BeliefProbe(_ctx(_RetreatStallModel(stall=True)))
    sim.reset()
    res = sim.run(_PLAN, render=False)
    step = res.steps[0]
    assert step["failure"] == _STALL
    assert step["partial_actions"] == 7
    # Whole-number features only, the robot's own left out.
    assert step["changed"] == ["lever0.is_on 0 -> 1"]
    assert sim._require_state().get(_lever, "is_on") == 1.0
    assert res.final_state.get(_lever, "is_on") == 1.0
    # The lever the step flipped before it failed is the goal, and a real
    # episode ends the moment its goal holds.
    assert res.goal_reached
    assert "Goal reached: True" in res.text
    text = res.text
    assert ("It ran 7 actions before it stopped; the state after this run "
            "is where it stopped, as after a real failure.") in text
    assert "Changed before it stopped: lever0.is_on 0 -> 1" in text
    # Exact observations: no caveat about planning on noisy positions.
    assert "plans on the true scene" not in text


def test_a_clean_step_names_the_flags_it_changed():
    """Without predicates, a step's report still says what it did."""
    utils.reset_config({})
    sim = BeliefProbe(_ctx(_RetreatStallModel(stall=False)))
    sim.reset()
    res = sim.run(_PLAN, render=False)
    assert res.steps[0]["failure"] is None
    assert res.steps[0]["partial_actions"] is None
    assert "\n  Changed: lever0.is_on 0 -> 1" in res.text
    assert "FAILURE" not in res.text


def test_a_motion_failure_under_observation_noise_carries_a_caveat():
    """Under observation noise, a controller's failure (a ``[Skill/Phase]``
    failure) notes that the real controller plans on the true scene."""
    utils.reset_config({
        "continual_obs_noise_position": 0.01,
        "continual_obs_noise_orientation": 0.02,
    })
    ctx = _ctx(_RetreatStallModel(stall=True))
    sim = BeliefProbe(ctx)
    sim.reset()
    text = sim.run(_PLAN, render=False).text
    assert ("NOTE: A skill's controller failed in this rollout. The real "
            "controller plans on the true scene; these rollouts planned on "
            "positions with the observation noise (0.01 m, 0.02 rad)") in text
    assert "real calls of it" not in text
    # With the run's real record, the note gives the failing skill's.
    ctx.skill_record_provider = lambda: {"Push": (25, 0), "Wait": (9, 0)}
    sim.reset()
    text = sim.run(_PLAN, render=False).text
    assert "This run's real calls of it succeeded in: Push 25 of 25." in text
    assert "Wait" not in text
    utils.reset_config({})


def test_flag_changes_skip_continuous_features_and_cap_the_list():
    """Positions that move are not flags; a long list is cut short."""
    many_type = Type("panel", [f"f{k}" for k in range(10)] + ["x"])
    panel = Object("panel0", many_type)
    pre = State({
        panel: np.zeros(11, dtype=np.float32),
        _robot: np.zeros(2, dtype=np.float32),
    })
    post = pre.copy()
    for k in range(10):
        post.set(panel, f"f{k}", 1.0)
    post.set(panel, "x", 0.25)
    post.set(_robot, "fingers", 1.0)
    changes = _flag_changes(pre, post)
    assert changes[:2] == ["panel0.f0 0 -> 1", "panel0.f1 0 -> 1"]
    assert changes[-1] == "and 2 more"
    assert len(changes) == 9
    assert not any("panel0.x" in c or "robot" in c for c in changes)


def test_joint_rehearsal_counts_draws_whose_controller_failed():
    """A rehearsal's draws that failed in a controller phase are counted in the
    caveat; a refusal to start is not a motion failure."""
    assert _motion_failures([
        "step 0 (Move(block0)[0.95]): [Move/Push] incremental-IK stalled",
        "step 0 (Move(block0)[0.95]): not initiable",
        None,
    ]) == 1

    class _StallInHole(_PhysicsHoleModel):
        """Inside the friction hole the move's controller stalls."""

        def get_next_state_and_num_actions(self, state, option):
            nxt, num = super().get_next_state_and_num_actions(state, option)
            if 0.49 < self.friction < 0.51:
                self.last_execution_failure = "[Move/Push] stalled"
                return nxt, 0
            self.last_execution_failure = None
            return nxt, num

    ctx, model, _, _ = _joint_ctx()
    model.__class__ = _StallInHole
    utils.reset_config({
        "belief_joint_draws": 4,
        "continual_uncertainty_decisions": True,
        "continual_obs_noise_position": 0.01,
        "continual_obs_noise_orientation": 0.02,
    })
    sim = BeliefProbe(ctx)
    _on_observation(sim, ctx)
    res = sim.run("Move(block0:block)[0.95]", render=False)
    assert isinstance(res, ProbeJointResult)
    assert res.p_hat == pytest.approx(0.5)
    assert ("In 2 of the 4 draws a skill's controller failed. The real "
            "controller plans on the true scene") in res.text
    utils.reset_config({})
