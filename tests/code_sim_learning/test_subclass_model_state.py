"""A subclass's hidden state must survive the same paths as its physics."""
# pylint: disable=protected-access
import copy
import dataclasses
import pickle
from typing import Any, ClassVar, Dict, List

import numpy as np
import pytest

from predicators import utils
from predicators.approaches.agent_sim_learning_approach import \
    AgentSimLearningApproach
from predicators.code_sim_learning import trajectory_prep
from predicators.code_sim_learning.base_simulator import base_simulator_class
from predicators.code_sim_learning.config import SysIdConfig
from predicators.code_sim_learning.fit_space import ParamSpec
from predicators.code_sim_learning.model_state import memory_from_history, \
    restored_model_state
from predicators.code_sim_learning.rollout_env import clear_memory_written, \
    memory_written_in_rollouts, rollout_states
from predicators.ground_truth_models.balloons.gt_simulator_env import \
    BalloonsResidualEnv
from predicators.structs import Action, LowLevelTrajectory, Object, State, Type


class _MemoryModel(BalloonsResidualEnv):
    """A small recurrent model with engine physics and an observable
    counter."""

    AGENT_PARAM_SPECS: ClassVar[List[Any]] = [
        ParamSpec("rate", .25, lo=.1, hi=1.0)
    ]
    MODEL_STATE_INIT: ClassVar[Dict[str, Any]] = {
        "charge": 0.0,
        "events": [],
    }

    @classmethod
    def update_model_state(cls, observation, model_state, params, action):
        """The same observation-conditioned update in simulation and play."""
        assert observation.privileged is None
        assert observation.simulator_state is None
        assert action is not None
        model_state["charge"] += params["rate"]
        model_state["events"].append(model_state["charge"])

    def _domain_specific_step(self):
        """No extra body dynamics in this memory-only example."""


class _StepMemoryModel(_MemoryModel):
    """Infers a link in its engine step, where no reset replays it."""

    def _domain_specific_step(self):
        """Leave an entry empty, write a link once charged, then empty the link
        and delete the events."""
        self.model_state.setdefault("unused", {})
        if self.model_state["charge"] == .5:
            self.model_state.setdefault("links", []).append(["a", "b"])
        elif self.model_state["charge"] == .75:
            self.model_state["links"].clear()
            del self.model_state["events"]


class _ObservedPopModel(_MemoryModel):
    """Pops a balloon in its step and remembers it there, and its update reads
    the pop back from the observation."""

    @classmethod
    def update_model_state(cls, observation, model_state, params, action):
        """Count as the parent does, and remember each observed pop."""
        super().update_model_state(observation, model_state, params, action)
        for obj in observation:
            if obj.type.name == "balloon" and \
                    observation.get(obj, "popped") > .5:
                model_state.setdefault("popped", {})[obj.name] = True

    def _domain_specific_step(self):
        """Pop the first balloon once charged."""
        if self.model_state["charge"] == .5:
            name = self._balloons[0].name
            self._popped[name] = True
            self.model_state.setdefault("popped", {})[name] = True


def _balloons_start() -> State:
    """A known Balloons level, without the expensive oracle probes that
    generate one."""
    # pylint: disable-next=import-outside-toplevel
    from predicators.envs.pybullet_balloons import PyBulletBalloonsEnv
    source = PyBulletBalloonsEnv(use_gui=False)
    try:
        initial = source.level_state(0, [0, 1], (.7, .75))
        source._set_state(initial)
        return source._get_state()
    finally:
        source.dispose()


@pytest.mark.parametrize("supplied", [False, True])
def test_subclass_state_branches_and_execution_agree(supplied):
    """Loadable subclass memory reaches planning and real execution equally."""
    utils.reset_config({
        "env": "pybullet_balloons",
        "seed": 0,
        "skill_phase_use_motion_planning": False,
    })
    model_cls = _MemoryModel
    if supplied:
        model_cls = type(
            "MemoryModel", (base_simulator_class("pybullet_balloons"), ), {
                "AGENT_PARAM_SPECS": _MemoryModel.AGENT_PARAM_SPECS,
                "MODEL_STATE_INIT": _MemoryModel.MODEL_STATE_INIT,
                "update_model_state":
                _MemoryModel.__dict__["update_model_state"],
            })
    env = model_cls(use_gui=False)
    try:
        initial = _balloons_start()
        action = Action(np.array(initial.joint_positions, dtype=np.float32))
        approach = AgentSimLearningApproach.__new__(AgentSimLearningApproach)
        approach._residual_env_cls = model_cls
        approach._residual_rules = []
        approach._fitted_params = {"rate": .25}
        approach._latent_init = None
        tracker = approach.make_latent_tracker()
        assert tracker is not None
        assert approach._latent_tracking_available()
        start = tracker.attach(initial, None)
        assert start.latent == {"charge": 0.0, "events": []}
        assert initial.latent is None
        predicted = env.simulate(initial, action)
        observed = tracker.attach(predicted, action)
        assert predicted.latent == observed.latent == {
            "charge": .25,
            "events": [.25]
        }
        second = env.simulate(predicted, action)
        assert second.latent == {"charge": .5, "events": [.25, .5]}
        approach._fit_trajectories = [
            LowLevelTrajectory([initial, predicted, second], [action, action])
        ]
        fitted_data = approach._rollout_fit_trajectories({"clip": ["is_on"]})
        assert len(fitted_data) == 1 and len(fitted_data[0][1]) == 2
        # Same physical state, different memory: sibling branches must not
        # inherit one another's state even when State.allclose says they match.
        sibling = predicted.copy()
        sibling.latent = {"charge": 2.0, "events": [2.0]}
        branch = env.simulate(sibling, action)
        assert branch.latent == {"charge": 2.25, "events": [2.0, 2.25]}
        assert predicted.latent == {"charge": .25, "events": [.25]}
        assert not _MemoryModel.MODEL_STATE_INIT["events"]
        snapshot = copy.deepcopy(branch.latent)
        env._set_state(branch)
        assert env._get_state().latent == snapshot
        restored = pickle.loads(pickle.dumps(branch))
        assert env.simulate(restored, action).latent["charge"] == 2.5
        # A segment that starts mid-recording rebuilds its memory from the
        # recording before it, under the parameters the replay runs with.
        resumed = second.copy()
        resumed.latent = memory_from_history([action, action],
                                             [predicted, second])
        env.apply_physical_param_overrides({"rate": .5})
        env._set_state(resumed)
        assert env._get_state().latent == {"charge": 1.0, "events": [.5, 1.]}
        assert env.simulate(resumed, action).latent["charge"] == 1.5
        env.apply_physical_param_overrides({"rate": .25})
        # A replacement physical-parameter dict is visible to existing
        # trackers, including classifiers already holding a memory snapshot.
        approach._identified_physical_params = {"rate": .5}
        tracked = tracker.attach(second, action)
        assert tracked.latent["charge"] == .75
        assert observed.latent["charge"] == .25
        tracker.reset()
        assert tracker.attach(initial, None).latent == {
            "charge": 0.0,
            "events": []
        }
        # All of this memory came from the observation-driven update.
        assert not env.memory_written_by_step
    finally:
        env.dispose()


def test_the_memory_a_step_writes_is_recorded():
    """The env records the model_state entries its engine step writes, which
    memory rebuilt from the observations lacks; the observation-driven update
    and an entry the step leaves empty are not recorded."""
    utils.reset_config({
        "env": "pybullet_balloons",
        "seed": 0,
        "skill_phase_use_motion_planning": False,
    })
    initial = _balloons_start()
    action = Action(np.array(initial.joint_positions, dtype=np.float32))
    env = _StepMemoryModel(use_gui=False)
    try:
        first = env.simulate(initial, action)
        assert not env.memory_written_by_step
        second = env.simulate(first, action)
        assert env.memory_written_by_step == {"links": "[['a', 'b']]"}
        assert second.latent["links"] == [["a", "b"]]
        # A reset from the observations rebuilds memory without the link.
        resumed = second.copy()
        resumed.latent = memory_from_history([action, action], [first, second])
        assert "links" not in restored_model_state(_StepMemoryModel, resumed,
                                                   {"rate": .25})
        # An entry the step empties or deletes is recorded as such.
        env.simulate(second, action)
        assert env.memory_written_by_step == {
            "links": "empty",
            "events": "deleted"
        }
    finally:
        env.dispose()


def test_segments_rebuild_memory_from_the_recording_before_them():
    """Rest segmentation for a model with memory: each segment that starts mid-
    recording carries the actions and observations before it, and restoring it
    runs the model's update over them under the restoring parameters, so each
    candidate point gets the memory it implies."""
    utils.reset_config({})
    obj = Object("d0", Type("dot", ["x"]))
    xs = ([0.0] * 15 + [0.01 * k for k in range(1, 11)] + [0.1] * 15 +
          [0.1 + 0.01 * k for k in range(1, 11)] + [0.2] * 15)
    states = [State({obj: np.array([x])}) for x in xs]
    actions = [Action(np.zeros(1, dtype=np.float32)) for _ in xs[1:]]
    config = dataclasses.replace(SysIdConfig.from_cfg(),
                                 noise_filter=False,
                                 settle_tol=1e-3,
                                 settle_margin=3,
                                 segment_min_rest_steps=5)
    plain, carried = (trajectory_prep.split_at_rest_points((states, actions),
                                                           {"dot": ["x"]},
                                                           config=config,
                                                           carry_history=carry)
                      for carry in (False, True))
    assert len(plain) == len(carried) == 2
    starts = []
    for (p_states, p_actions), (c_states, c_actions) in zip(plain, carried):
        assert c_actions == p_actions and c_states[1:] == p_states[1:]
        assert c_states[0].allclose(p_states[0])
        assert p_states[0].latent is None
        starts.append(
            next(i for i, s in enumerate(states) if s is c_states[1]) - 1)
    assert starts == [14, 39]
    assert all(s.latent is None for s in states)
    for (seg_states, _), start in zip(carried, starts):
        for rate in (.25, .5):
            memory = restored_model_state(_MemoryModel, seg_states[0],
                                          {"rate": rate})
            assert memory["charge"] == pytest.approx(rate * start)
            assert len(memory["events"]) == start
        copied = copy.deepcopy(seg_states[0])
        assert restored_model_state(_MemoryModel, copied,
                                    {"rate": 1.0})["charge"] == start


def test_memory_the_update_rebuilds_from_the_observation_is_not_recorded():
    """A step that writes memory its update also infers from the observation
    after the step loses nothing at a reset, so nothing is recorded."""
    utils.reset_config({
        "env": "pybullet_balloons",
        "seed": 0,
        "skill_phase_use_motion_planning": False,
    })
    initial = _balloons_start()
    action = Action(np.array(initial.joint_positions, dtype=np.float32))
    env = _ObservedPopModel(use_gui=False)
    try:
        second = env.simulate(env.simulate(initial, action), action)
        assert list(second.latent["popped"].values()) == [True]
        assert not env.memory_written_by_step
    finally:
        env.dispose()


def test_rollouts_record_the_memory_their_steps_write():
    """A rollout adds the memory its env's step wrote to this process's record,
    which a fit clears before its rollouts and reads after."""
    utils.reset_config({
        "env": "pybullet_balloons",
        "seed": 0,
        "skill_phase_use_motion_planning": False,
    })
    initial = _balloons_start()
    action = Action(np.array(initial.joint_positions, dtype=np.float32))
    clear_memory_written()
    env = _StepMemoryModel(use_gui=False)
    try:
        rollout_states(env, initial, [action, action], {})
        assert memory_written_in_rollouts() == {"links": "[['a', 'b']]"}
        clear_memory_written()
        assert not memory_written_in_rollouts()
    finally:
        env.dispose()
