"""The concrete visible-physics base injected into simulator artifacts.

Only model code supplies the residual step. The underlying environment
stays in its visible-physics mode, including its mass and material
defaults.
"""
from functools import lru_cache
from typing import Any, Dict, List, Optional, Set, Tuple, Type

from predicators import utils
from predicators.envs.pybullet_balloons_base import PyBulletBalloonsBaseEnv
from predicators.envs.pybullet_boil_base import PyBulletBoilBaseEnv
from predicators.envs.pybullet_bridge_base import PyBulletBridgeBaseEnv
from predicators.envs.pybullet_domino.components.domino_bodies import \
    DominoBodiesComponent
from predicators.envs.pybullet_domino.sim_core import PyBulletDominoBaseEnv
from predicators.envs.pybullet_env import PyBulletEnv
from predicators.envs.pybullet_fan_base import PyBulletFanBaseEnv
from predicators.settings import CFG
from predicators.structs import EnvironmentTask, Object, Predicate, State


class _VisibleModelMixin:
    """No task generator, predicates, or hidden mechanism on a model base."""

    @classmethod
    def get_name(cls) -> str:
        """An internal name outside the environment registry."""
        return "visible_model"

    @property
    def predicates(self) -> Set[Predicate]:
        """No supplied hidden goal classifiers."""
        return set()

    @property
    def goal_predicates(self) -> Set[Predicate]:
        """Goals belong to the observed task, not the model base."""
        return set()

    def _generate_train_tasks(self) -> List[EnvironmentTask]:
        return []

    def _generate_test_tasks(self) -> List[EnvironmentTask]:
        return []


class _BalloonsModelBase(_VisibleModelMixin, PyBulletBalloonsBaseEnv):
    """Balloons bodies without lift laws or tasks."""


class _BoilModelBase(_VisibleModelMixin, PyBulletBoilBaseEnv):
    """Kitchen bodies with passive readouts, not fill/heating laws."""

    def __init__(self, use_gui: bool = False, **kwargs: Any) -> None:
        self._passive_readouts: Dict[Tuple[str, str], float] = {}
        super().__init__(use_gui=use_gui, **kwargs)

    def _set_domain_specific_state(self, state: State) -> None:
        self._passive_readouts = {(o.name, f): state.get(o, f)
                                  for o in state for f in o.type.feature_names
                                  if f in ("heat_level", "bubbling_level",
                                           "spilled_level")}
        super()._set_domain_specific_state(state)

    def _get_domain_specific_feature(self, obj: Object, feature: str) -> float:
        if feature in ("heat_level", "bubbling_level", "spilled_level"):
            return self._passive_readouts.get((obj.name, feature), 0.0)
        return super()._get_domain_specific_feature(obj, feature)


class _FanModelBase(_VisibleModelMixin, PyBulletFanBaseEnv):
    """Arena bodies without wind or the hidden goal classifier."""

    def __init__(self, use_gui: bool = False, **kwargs: Any) -> None:
        self._passive_readouts: Dict[str, float] = {}
        super().__init__(use_gui=use_gui, **kwargs)

    def _set_domain_specific_state(self, state: State) -> None:
        self._passive_readouts = {
            o.name: state.get(o, "is_hit")
            for o in state if "is_hit" in o.type.feature_names
        }
        super()._set_domain_specific_state(state)

    def _get_domain_specific_feature(self, obj: Object, feature: str) -> float:
        if feature == "is_hit":
            return self._passive_readouts.get(obj.name, 0.0)
        return super()._get_domain_specific_feature(obj, feature)


class _BridgeModelBase(_VisibleModelMixin, PyBulletBridgeBaseEnv):
    """Bridge geometry and readouts without glue/cure laws or tasks."""


class _DominoModelBodies(DominoBodiesComponent):
    """Concrete body component without predicate/task semantics."""

    def get_predicates(self) -> Set[Predicate]:
        return set()

    def get_goal_predicates(self) -> Set[Predicate]:
        return set()


class _DominoModelBase(_VisibleModelMixin, PyBulletDominoBaseEnv):
    """Plain Domino bodies, not task/certificate components."""

    def __init__(self, use_gui: bool = False, **kwargs: Any) -> None:
        max_dominos = max(*CFG.domino_train_num_dominos,
                          *CFG.domino_test_num_dominos)
        if CFG.domino_min_block_tasks or CFG.domino_heavy_block_tasks:
            extra = 3 if CFG.domino_heavy_block_tasks else 2
            max_dominos = max(max_dominos,
                              CFG.domino_min_block_num_blues + extra)
        component = _DominoModelBodies(
            num_dominos_max=max_dominos,
            num_targets_max=max(*CFG.domino_train_num_targets,
                                *CFG.domino_test_num_targets),
            num_pivots_max=max(*CFG.domino_train_num_pivots,
                               *CFG.domino_test_num_pivots),
            workspace_bounds=self._default_workspace_bounds())
        super().__init__([component], use_gui=use_gui, **kwargs)


@lru_cache(maxsize=None)
def base_simulator_class(env_name: str) -> Optional[Type[PyBulletEnv]]:
    """Resolve a model base without constructing an env or generating tasks.

    Non-PyBullet historical artifacts keep their existing loader
    behavior. Generated classes do not enter the real environment's
    registry slot.
    """
    visible: Dict[str, Type[PyBulletEnv]] = {
        "pybullet_balloons": _BalloonsModelBase,
        "pybullet_boil": _BoilModelBase,
        "pybullet_fan": _FanModelBase,
        "pybullet_bridge": _BridgeModelBase,
        "pybullet_domino": _DominoModelBase,
    }
    if env_name in visible:
        return _make_model_class(env_name, visible[env_name])
    # Preserve historical non-benchmark domains until they have visible cores.
    return oracle_base_simulator_class(env_name)


@lru_cache(maxsize=None)
def oracle_base_simulator_class(env_name: str) -> Optional[Type[PyBulletEnv]]:
    """Privileged base for harness-supplied Oracle artifacts only.

    Never inject this into learned artifacts. Oracle intentionally
    receives the correct mechanism helpers, while its source stays
    outside the sandbox.
    """
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
    return _make_model_class(env_name, env_cls)


def _make_model_class(env_name: str,
                      env_cls: Type[PyBulletEnv]) -> Type[PyBulletEnv]:
    """Wrap a chosen base with the same agent parameter/step contract."""

    # pylint: disable=protected-access

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
        "BaseSimulator",
        (env_cls, ),
        {
            "__module__": __name__,
            "__init__": initialize,
            "get_name": classmethod(name),
            # Controllers belong to the deployment, not this internal
            # class's registry name. Certificates replay those controllers
            # on the candidate's own physics.
            "_skill_env_name": env_name,
            "_agent_model_dynamics": True,
            "_domain_specific_step": no_dynamics,
            "_agent_param_info": agent_info,
            "get_physical_param_info": parameter_info,
            "apply_physical_param_overrides": apply_parameters,
        })
