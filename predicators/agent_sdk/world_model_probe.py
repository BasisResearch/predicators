"""The ``sim`` probe over an agent-written ``world_model.py``.

``run_python`` binds ``sim`` to a :class:`WorldModelProbe`. It scores
the current ``world_model.py`` on the recorded trajectories and predicts
a skill plan through it, one plan per call. It loads the file again
whenever the file changes. Nothing here acts in the environment.
"""
from __future__ import annotations

import dataclasses
import weakref
from typing import Any, Dict, List, Optional, Union

from predicators.agent_sdk.belief_probe import _ProbeCore
from predicators.agent_sdk.probe_results import ProbeResult
from predicators.agent_sdk.rollout_probe import Modifications

# The agent that holds a WorldModelProbe can read everything this module
# documents: help(sim), the class and module names behind type(sim), the
# result's class, and the page pydoc renders for the module. So the
# docstrings here say what this probe does and name nothing else, the
# classes define only the calls and fields the probe has, and __all__
# keeps the module's page to them. An unknown call is an AttributeError
# and an unknown keyword a TypeError, as on any object. Notes for
# maintainers belong in comments like this one, which pydoc does not
# show.
__all__ = ["PlanPrediction", "WorldModelProbe"]

# What a step of the probe core's rollout record holds that this probe
# never fills.
_UNUSED_STEP_KEYS = ("image", )


@dataclasses.dataclass(repr=False)
class PlanPrediction(ProbeResult):
    """Outcome of ``sim.run(plan_text)``: what ``world_model.py`` predicts.

    ``print(result)`` gives the report of each step. ``steps`` holds one
    dict per plan line that ran, up to the first line that failed:
    ``option`` (the line), ``num_actions`` (the low-level steps it
    took), ``failure`` (None when the line ran), ``added`` and
    ``deleted`` (the atoms the line changed), ``changed`` (the flag and
    count features it changed), ``subgoals_missing`` (the atoms of its
    ``-> {atoms}`` annotation that did not hold after it) and
    ``partial_actions`` (the steps a failed line took before it stopped,
    when known). ``goal_reached`` says whether the goal atoms hold at
    the end, ``final_atoms`` lists the atoms that hold there and
    ``final_state`` is the last state. ``states`` holds the predicted
    states, the start state first, ``labels`` the skill ``(name,
    (object, ...), (param, ...))`` of each transition between them, and
    ``line_states`` the state after each plan line (None for a line that
    failed). ``coarse`` is True when ``states`` holds one state per
    skill. ``notes`` lists caveats.
    """


class WorldModelProbe:
    """Scores and predictions of your current ``world_model.py``.

    The probe loads the file again whenever the file changes, and a call
    that needs the model raises an error while no loadable file exists.
    It holds a current state: ``reset`` sets it, ``run`` predicts a
    skill plan from it and moves it to where the prediction ends,
    ``state`` reads it, and ``snapshot`` and ``restore`` save it and set
    it back. ``score`` compares the model with the recorded
    trajectories. A rollout here is a prediction of your model; nothing
    here acts in the environment.
    """

    def __init__(self, probe: _ProbeCore) -> None:
        # The probe core is kept off the instance: ``dir(sim)`` and
        # ``vars(sim)`` list this probe's calls and nothing behind them.
        _PROBES[self] = probe

    def score(self,
              traj_idxs: Optional[List[int]] = None,
              num_particles: Optional[int] = None) -> str:
        """Score ``world_model.py`` on the recorded trajectories.

        The score is the particle-filter kernel pseudo-likelihood of the
        recorded skill transitions over the model's hidden state; 0 is
        perfect. The report also gives the error of each feature and the
        worst transitions. ``traj_idxs`` selects trajectories by their
        index in ``trajectories``; the default is all of them.
        ``num_particles`` sets the number of particles.
        """
        return _PROBES[self].score(traj_idxs=traj_idxs,
                                   num_particles=num_particles)

    def reset(self,
              task_idx: Optional[int] = None,
              mods: Optional[Modifications] = None,
              current: bool = False) -> WorldModelProbe:
        """Set the current state.

        The default is the initial state of a level (``task_idx``
        indexes ``train_tasks``, the levels reached so far; None is the
        current level). ``current=True`` is the last real observation,
        with the model's memory rebuilt by a replay of this episode's
        recorded skills through the current ``world_model.py``. ``mods``
        overrides features of that state (``{'obj': {'x': 1.05}}``).
        Returns ``self`` so calls chain.
        """
        _PROBES[self].reset(task_idx=task_idx, mods=mods, current=current)
        return self

    def task(self, task_idx: Optional[int] = None) -> str:
        """Describe a level: its goal, objects, initial atoms and initial
        state.

        ``task_idx`` indexes ``train_tasks``; None is the current level.
        The current state does not change.
        """
        return _PROBES[self].task(task_idx)

    def state(
        self,
        obj_name: Optional[str] = None
    ) -> Union[Dict[str, Dict[str, float]], Dict[str, float]]:
        """The features of the current state at full precision:

        ``{"name:type": {feature: value}}``, or one object's ``{feature:
        value}`` for ``state("name")``.
        """
        return _PROBES[self].state(obj_name)

    def run(self, plan_text: str) -> PlanPrediction:
        """Predict a skill plan from the current state, once.

        ``plan_text`` has the grammar of ``skills_execute_plan``: one
        skill per line, ``Skill(obj:type, ...)[params]``. The prediction
        runs each line through your ``transition`` and stops at the
        first line that fails. A ``-> {atoms}`` annotation on a line is
        checked against the state the line predicts. The current state
        moves to where the prediction ends. Print the result for the
        report of each step; its ``states`` hold the predicted states,
        the start state first, and ``final_state`` the last one.
        """
        # This probe has no image of a predicted state.
        result = _PROBES[self].run(plan_text, render=False)
        assert isinstance(result, ProbeResult)
        fields: Dict[str, Any] = {
            field.name: getattr(result, field.name)
            for field in dataclasses.fields(result)
        }
        fields["steps"] = [{
            key: value
            for key, value in step.items() if key not in _UNUSED_STEP_KEYS
        } for step in result.steps]
        return PlanPrediction(**fields)

    def snapshot(self) -> int:
        """Save a copy of the current state; the id that ``restore`` takes."""
        return _PROBES[self].snapshot()

    def restore(self, snapshot_id: int) -> WorldModelProbe:
        """Set the current state back to a snapshot."""
        _PROBES[self].restore(snapshot_id)
        return self


# The probe core behind each WorldModelProbe the agent holds. The name
# has no annotation: the module's page would list an annotated name.
_PROBES = weakref.WeakKeyDictionary[WorldModelProbe, _ProbeCore]()
