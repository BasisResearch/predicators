"""Shared lifecycle for a simulator subclass's observation-driven memory."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Mapping, Optional, Sequence, \
    Tuple, Type

from predicators.code_sim_learning.utils import init_latent, observation_view
from predicators.structs import Action, State

if TYPE_CHECKING:
    from predicators.envs.pybullet_env import PyBulletEnv


def has_model_state(model_cls: Optional[type]) -> bool:
    """Whether the model declares state that must accompany its predictions."""
    return model_cls is not None and getattr(model_cls, "MODEL_STATE_INIT",
                                             None) is not None


def model_parameters(model_cls: Type[PyBulletEnv],
                     values: Mapping[str, float]) -> Dict[str, float]:
    """Declared defaults plus currently deployed values, with no fitting."""
    return {
        spec.name: float(values.get(spec.name, spec.init_value))
        for spec in model_cls.AGENT_PARAM_SPECS
    }


def initial_model_state(model_cls: Type[PyBulletEnv],
                        params: Mapping[str, float]) -> Dict[str, Any]:
    """Every initial state is independent, including nested containers."""
    return init_latent(model_cls.MODEL_STATE_INIT, dict(params))


@dataclass(frozen=True)
class RecordedHistory:
    """The recorded episode before a state: each action and the observation
    after it.

    A fit segment that starts mid-episode carries one in place of memory
    (:func:`memory_from_history`). The memory there is the model's own
    update run over this history under whichever parameters the replay
    uses, as the execution-time tracker computes it from the same
    observations, so a candidate parameter point gets the memory it
    implies without replaying the physics before the segment.
    """

    actions: Tuple[Action, ...]
    observations: Tuple[State, ...]

    def __deepcopy__(self, memo: Dict[int, Any]) -> RecordedHistory:
        # Recorded data never changes, so a copied state shares it.
        return self


# The latent key under which a segment start carries its history.
_HISTORY_KEY = "recorded_history"


def memory_from_history(actions: Sequence[Action],
                        observations: Sequence[State]) -> Dict[str, Any]:
    """A latent block that stands for the memory ``actions`` and the
    ``observations`` after them imply under the parameters that restore it
    (:func:`restored_model_state`)."""
    assert len(actions) == len(observations), (len(actions), len(observations))
    return {_HISTORY_KEY: RecordedHistory(tuple(actions), tuple(observations))}


def restored_model_state(model_cls: Type[PyBulletEnv], state: State,
                         params: Mapping[str, float]) -> Dict[str, Any]:
    """Restore a search node's memory or initialize a fresh episode.

    A state carrying a recorded history (:func:`memory_from_history`)
    gets the memory that history implies under ``params``.
    """
    if state.latent is not None:
        history = state.latent.get(_HISTORY_KEY)
        if isinstance(history, RecordedHistory):
            memory = initial_model_state(model_cls, params)
            for action, observation in zip(history.actions,
                                           history.observations):
                advance_model_state(model_cls, observation, memory, params,
                                    action)
            return memory
        return copy.deepcopy(state.latent)
    return initial_model_state(model_cls, params)


def advance_model_state(model_cls: Type[PyBulletEnv], observation: State,
                        model_state: Dict[str, Any],
                        params: Mapping[str, float], action: Action) -> None:
    """Run the same sanitized update in simulation and real execution.

    The callback receives copies of observations, actions and
    parameters. Only its model-state argument is an owned mutable
    output.
    """
    observed = observation_view(observation).copy()
    model_cls.update_model_state(observed, model_state, dict(params),
                                 Action(action.arr.copy()))
