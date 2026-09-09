"""Shared lifecycle for a simulator subclass's observation-driven memory."""
from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any, Dict, Mapping, Optional, Type

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


def restored_model_state(model_cls: Type[PyBulletEnv], state: State,
                         params: Mapping[str, float]) -> Dict[str, Any]:
    """Restore a search node's memory or initialize a fresh episode."""
    if state.latent is not None:
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
