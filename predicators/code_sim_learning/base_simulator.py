"""The concrete visible-physics base injected into simulator artifacts.

Only model code supplies the residual step. The underlying environment
stays in its visible-physics mode, including its mass and material
defaults.
"""
from functools import lru_cache
from typing import Any, Dict, List, Optional, Set, Type

from predicators import utils
from predicators.envs.pybullet_balloons_base import PyBulletBalloonsBaseEnv
from predicators.envs.pybullet_env import PyBulletEnv
from predicators.structs import EnvironmentTask, Predicate


class _BalloonsModelBase(PyBulletBalloonsBaseEnv):
    """Concrete visible core with no inherited hidden physics or task maker."""

    @classmethod
    def get_name(cls) -> str:
        return "pybullet_balloons_visible_model"

    @property
    def predicates(self) -> Set[Predicate]:
        return set()

    @property
    def goal_predicates(self) -> Set[Predicate]:
        return set()

    def _generate_train_tasks(self) -> List[EnvironmentTask]:
        return []

    def _generate_test_tasks(self) -> List[EnvironmentTask]:
        return []


@lru_cache(maxsize=None)
def base_simulator_class(env_name: str) -> Optional[Type[PyBulletEnv]]:
    """Resolve a model base without constructing an env or generating tasks.

    Non-PyBullet historical artifacts keep their existing loader
    behavior. Generated classes do not enter the real environment's
    registry slot.
    """
    # pylint: disable=protected-access
    candidates = [
        cls for cls in utils.get_all_subclasses(PyBulletEnv)
        if not cls.__abstractmethods__ and cls.__module__.startswith(
            "predicators.envs.") and cls.get_name() == env_name
    ]
    if not candidates:
        return None
    assert len(candidates) == 1, env_name
    env_cls = (_BalloonsModelBase
               if env_name == "pybullet_balloons" else candidates[0])

    def initialize(self: Any, use_gui: bool = False, **kwargs: Any) -> None:
        kwargs["skip_residual_dynamics"] = True
        # Initialize the existing generated instance through its resolved base.
        # pylint: disable-next=unnecessary-dunder-call
        env_cls.__init__(self, use_gui=use_gui, **kwargs)
        apply_parameters(self, dict(self._agent_param_values))

    def name(_cls: type) -> str:
        return f"{env_name}_agent_base"

    def no_dynamics(self: Any) -> None:
        """The subclass supplies this hook; no ground-truth residuals run."""
        del self

    def agent_info(self: Any) -> Dict[str, Dict]:
        if getattr(self, "_reading_base_param_info", False):
            return {}
        return PyBulletEnv._agent_param_info(self)

    def stock_info(self: Any) -> Dict[str, Dict]:
        self._reading_base_param_info = True
        try:
            return env_cls.get_physical_param_info(self)
        finally:
            self._reading_base_param_info = False

    def parameter_info(self: Any) -> Dict[str, Dict]:
        return {**stock_info(self), **agent_info(self)}

    def apply_parameters(self: Any, params: Dict[str, float]) -> None:
        native = stock_info(self)
        own_names = {s.name for s in type(self).AGENT_PARAM_SPECS}
        unknown = set(params) - set(native) - own_names
        if unknown:
            raise ValueError(f"Unknown physical param(s) {sorted(unknown)}.")
        own = {n: float(v) for n, v in params.items() if n in own_names}
        self._agent_param_values.update(own)
        env_cls.apply_physical_param_overrides(
            self, {n: v
                   for n, v in params.items() if n in native})
        if own:
            self._on_agent_params_changed()

    return type(
        "BaseSimulator", (env_cls, ), {
            "__module__": __name__,
            "__init__": initialize,
            "get_name": classmethod(name),
            "_agent_model_dynamics": True,
            "_domain_specific_step": no_dynamics,
            "_agent_param_info": agent_info,
            "get_physical_param_info": parameter_info,
            "apply_physical_param_overrides": apply_parameters,
        })
