"""Adapter wrappers that let predicators use mara-robosim environments.

Each adapter subclass of ``BaseEnv`` delegates to a mara-robosim
``PyBulletEnv`` instance, converting between the two libraries' struct
types at the boundary.  Because the struct definitions (Type, Object,
State, Predicate, …) are structurally identical but live in different
Python modules, we maintain bidirectional caches for fast conversion.

Usage::

    python main.py --env mara_blocks --approach oracle ...

The adapters read predicators' ``CFG`` settings and translate them into
the corresponding ``mara_robosim.config.*Config`` frozen dataclass.
"""

# pylint: disable=import-outside-toplevel
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Set

import matplotlib
import numpy as np
from gym.spaces import Box

from predicators.envs.base_env import BaseEnv
from predicators.settings import CFG
from predicators.structs import Action as PredAction
from predicators.structs import DerivedPredicate as PredDerivedPredicate
from predicators.structs import EnvironmentTask as PredEnvironmentTask
from predicators.structs import GroundAtom as PredGroundAtom
from predicators.structs import Object as PredObject
from predicators.structs import Predicate as PredPredicate
from predicators.structs import State as PredState
from predicators.structs import Type as PredType
from predicators.utils import PyBulletState

# Lazy imports of mara-robosim are done inside methods to avoid import
# errors when the package is not installed.

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Struct converter
# ---------------------------------------------------------------------------


class _StructConverter:
    """Bidirectional converter between mara-robosim and predicators structs.

    Maintains caches keyed by name so that the same predicators Type /
    Object is returned for the same mara-robosim Type / Object across
    calls.
    """

    def __init__(self) -> None:
        # mara name -> predicators struct
        self._type_cache: Dict[str, PredType] = {}
        self._obj_cache: Dict[str, PredObject] = {}
        self._pred_cache: Dict[str, PredPredicate] = {}

        # predicators name -> mara struct (reverse direction)
        self._rev_type_cache: Dict[str, Any] = {}
        self._rev_obj_cache: Dict[str, Any] = {}
        # name -> mara predicate (for reverse atom conversion in derived preds)
        self._mara_pred_cache: Dict[str, Any] = {}

    # -- Type ---------------------------------------------------------------

    def convert_type(self, mara_type: Any) -> PredType:
        """Convert a mara-robosim Type to a predicators Type."""
        name: str = mara_type.name  # type: ignore[attr-defined]
        if name not in self._type_cache:
            parent = None
            mara_parent = mara_type.parent  # type: ignore[attr-defined]
            if mara_parent is not None:
                parent = self.convert_type(mara_parent)
            pred_type = PredType(
                name,
                mara_type.feature_names,  # type: ignore[attr-defined]
                parent=parent,
                sim_features=mara_type.
                sim_features,  # type: ignore[attr-defined]
            )
            self._type_cache[name] = pred_type
            self._rev_type_cache[name] = mara_type
        return self._type_cache[name]

    def reverse_type(self, pred_type: PredType) -> Any:
        """Look up the mara-robosim Type for a predicators Type."""
        return self._rev_type_cache[pred_type.name]

    # -- Object -------------------------------------------------------------

    def convert_object(self, mara_obj: Any) -> PredObject:
        """Convert a mara-robosim Object to a predicators Object."""
        name: str = mara_obj.name  # type: ignore[attr-defined]
        if name not in self._obj_cache:
            pred_type = self.convert_type(
                mara_obj.type)  # type: ignore[attr-defined]
            pred_obj = PredObject(name, pred_type)
            # Copy sim_data
            for key, val in mara_obj.sim_data.items(
            ):  # type: ignore[attr-defined]
                pred_obj.sim_data[key] = val
            self._obj_cache[name] = pred_obj
            self._rev_obj_cache[name] = mara_obj
        return self._obj_cache[name]

    def reverse_object(self, pred_obj: PredObject) -> Any:
        """Look up the mara-robosim Object for a predicators Object."""
        return self._rev_obj_cache[pred_obj.name]

    # -- State --------------------------------------------------------------

    def convert_state(self, mara_state: Any) -> PredState:
        """Convert a mara-robosim State to a predicators State."""
        data: Dict[PredObject, np.ndarray] = {}
        for mara_obj, arr in mara_state.data.items(
        ):  # type: ignore[attr-defined]
            pred_obj = self.convert_object(mara_obj)
            # Share the numpy array (no copy needed for read-only use).
            data[pred_obj] = arr
        sim_state = getattr(mara_state, "simulator_state", None)
        # Use PyBulletState when the simulator state carries joint data so
        # that ground-truth skill factories can access .joint_positions etc.
        if isinstance(sim_state, dict) and "joint_positions" in sim_state:
            return PyBulletState(data, simulator_state=sim_state)
        return PredState(data, simulator_state=sim_state)

    def reverse_state(self, pred_state: PredState) -> Any:
        """Convert a predicators State back to a mara-robosim State.

        Note: simulator_state is intentionally stripped.  The mara env
        manages its own PyBullet state; passing simulator_state through
        would cause ``allclose`` to raise.
        """
        from mara_robosim.structs import State as MaraState

        data = {}
        for pred_obj, arr in pred_state.data.items():
            mara_obj = self.reverse_object(pred_obj)
            data[mara_obj] = arr
        return MaraState(data)  # type: ignore[arg-type]

    # -- Predicate ----------------------------------------------------------

    def convert_predicate(self, mara_pred: Any) -> PredPredicate:
        """Convert a mara-robosim Predicate to a predicators Predicate.

        Handles both regular predicates (classifier takes State) and
        derived predicates (classifier takes Set[GroundAtom]).
        """
        name: str = mara_pred.name  # type: ignore[attr-defined]
        if name not in self._pred_cache:
            pred_types = [
                self.convert_type(t)
                for t in mara_pred.types  # type: ignore[attr-defined]
            ]
            # Store mara predicate for reverse atom lookup.
            self._mara_pred_cache[name] = mara_pred

            # Capture only caches (not ``self``) to avoid recursive pickling.
            mara_classifier = (
                mara_pred._classifier  # type: ignore[attr-defined]  # pylint: disable=protected-access
            )
            rev_obj_cache = self._rev_obj_cache

            # Check for derived predicate (classifier takes atoms, not state).
            aux = getattr(mara_pred, 'auxiliary_predicates', None)
            is_derived = aux is not None

            if is_derived:
                mara_pred_cache = self._mara_pred_cache

                # pylint: disable-next=dangerous-default-value
                def _wrapped_derived_classifier(
                    atoms: Set[PredGroundAtom],
                    objects: Sequence[PredObject],
                    _mara_cls: Any = mara_classifier,
                    _rev_obj: Any = rev_obj_cache,
                    _mara_preds: Any = mara_pred_cache,
                ) -> bool:
                    from mara_robosim.structs import GroundAtom as MaraGA
                    mara_atoms = set()
                    for atom in atoms:
                        mp = _mara_preds.get(atom.predicate.name)
                        if mp is None:
                            continue
                        m_objs = [_rev_obj[o.name] for o in atom.objects]
                        mara_atoms.add(MaraGA(mp, m_objs))
                    mara_objs = [_rev_obj[o.name] for o in objects]
                    return _mara_cls(mara_atoms, mara_objs)

                # Convert auxiliary predicates.
                assert aux is not None
                pred_aux = {self.convert_predicate(a) for a in aux}
                pred_pred = PredDerivedPredicate(
                    name,
                    pred_types,
                    _wrapped_derived_classifier,
                    auxiliary_predicates=pred_aux if pred_aux else None,
                )
            else:
                rev_type_cache = self._rev_type_cache

                # pylint: disable-next=dangerous-default-value
                def _wrapped_classifier(
                    pred_state: PredState,
                    objects: Sequence[PredObject],
                    _mara_cls: Any = mara_classifier,
                    _rev_obj: Any = rev_obj_cache,
                    _rev_type: Any = rev_type_cache,
                ) -> bool:
                    from mara_robosim.structs import State as MaraState
                    mara_data = {}
                    for pred_obj, arr in pred_state.data.items():
                        mara_data[_rev_obj[pred_obj.name]] = arr
                    mara_state = MaraState(mara_data)
                    mara_objs = [_rev_obj[o.name] for o in objects]
                    return _mara_cls(mara_state, mara_objs)

                pred_pred = PredPredicate(  # type: ignore[assignment]
                    name, pred_types, _wrapped_classifier)
            self._pred_cache[name] = pred_pred
        return self._pred_cache[name]

    # -- GroundAtom ---------------------------------------------------------

    def convert_ground_atom(self, mara_atom: Any) -> PredGroundAtom:
        """Convert a mara-robosim GroundAtom to a predicators GroundAtom."""
        pred = self.convert_predicate(
            mara_atom.predicate)  # type: ignore[attr-defined]
        objs = [
            self.convert_object(o)
            for o in mara_atom.objects  # type: ignore[attr-defined]
        ]
        return PredGroundAtom(pred, objs)

    # -- EnvironmentTask ----------------------------------------------------

    def convert_task(self, mara_task: Any) -> PredEnvironmentTask:
        """Convert a mara-robosim EnvironmentTask to predicators."""
        init_state = self.convert_state(
            mara_task.init)  # type: ignore[attr-defined]
        goal_atoms: Set[PredGroundAtom] = set()
        for mara_atom in mara_task.goal:  # type: ignore[attr-defined]
            goal_atoms.add(self.convert_ground_atom(mara_atom))
        alt = None
        mara_alt = getattr(mara_task, "alt_goal_desc", None)
        if mara_alt is not None:
            alt = {self.convert_ground_atom(a) for a in mara_alt}
        goal_nl = getattr(mara_task, "goal_nl", None)
        return PredEnvironmentTask(init_state,
                                   goal_atoms,
                                   alt_goal_desc=alt,
                                   goal_nl=goal_nl)


# ---------------------------------------------------------------------------
# Base adapter
# ---------------------------------------------------------------------------


def _build_base_config_kwargs() -> dict:
    """Extract base PyBulletConfig fields from predicators CFG."""
    # pylint: disable-next=import-outside-toplevel
    from mara_robosim.config import BiRRTConfig

    # ee_orns: CFG stores a nested defaultdict keyed by env name
    # then robot.  We rely on the domain config default for now.

    kwargs: dict = {}
    kwargs["robot"] = CFG.pybullet_robot
    kwargs["draw_debug"] = CFG.pybullet_draw_debug
    kwargs["camera_width"] = CFG.pybullet_camera_width
    kwargs["camera_height"] = CFG.pybullet_camera_height
    kwargs["sim_steps_per_action"] = CFG.pybullet_sim_steps_per_action
    kwargs["max_ik_iters"] = CFG.pybullet_max_ik_iters
    kwargs["ik_tol"] = CFG.pybullet_ik_tol
    kwargs["ik_validate"] = CFG.pybullet_ik_validate
    kwargs["control_mode"] = CFG.pybullet_control_mode
    kwargs["max_vel_norm"] = CFG.pybullet_max_vel_norm
    kwargs["birrt"] = BiRRTConfig(
        num_attempts=CFG.pybullet_birrt_num_attempts,
        num_iters=CFG.pybullet_birrt_num_iters,
        smooth_amt=CFG.pybullet_birrt_smooth_amt,
        extend_num_interp=CFG.pybullet_birrt_extend_num_interp,
        path_subsample_ratio=CFG.pybullet_birrt_path_subsample_ratio,
    )
    kwargs["num_train_tasks"] = CFG.num_train_tasks
    kwargs["num_test_tasks"] = CFG.num_test_tasks
    kwargs["seed"] = CFG.seed
    return kwargs


class MaraBaseAdapter(BaseEnv):
    """Base adapter that wraps a mara-robosim PyBulletEnv as a predicators
    BaseEnv.

    Subclasses override ``_build_config`` and ``_create_mara_env``.
    """

    def __init__(self, use_gui: bool = False) -> None:
        super().__init__(use_gui)
        config = self._build_config(use_gui)
        self._mara_env: Any = self._create_mara_env(config, use_gui)
        self._converter = _StructConverter()

    # -- Subclass hooks -----------------------------------------------------

    def _build_config(self, use_gui: bool) -> Any:
        """Build a mara-robosim config from predicators CFG.

        Subclasses override to add domain-specific fields.
        """
        from mara_robosim.config import PyBulletConfig

        kwargs = _build_base_config_kwargs()
        kwargs["use_gui"] = use_gui
        return PyBulletConfig(**kwargs)

    def _create_mara_env(self, config: Any, use_gui: bool) -> Any:
        """Instantiate the mara-robosim env.

        Must be overridden.
        """
        raise NotImplementedError

    # -- BaseEnv interface --------------------------------------------------

    @property
    def types(self) -> Set[PredType]:
        return {self._converter.convert_type(t) for t in self._mara_env.types}

    @property
    def predicates(self) -> Set[PredPredicate]:
        return {
            self._converter.convert_predicate(p)
            for p in self._mara_env.predicates
        }

    @property
    def goal_predicates(self) -> Set[PredPredicate]:
        return {
            self._converter.convert_predicate(p)
            for p in self._mara_env.goal_predicates
        }

    @property
    def action_space(self) -> Box:
        return self._mara_env.action_space

    def reset(self, train_or_test: str, task_idx: int) -> PredState:
        """Reset the mara env's PyBullet scene and return the observation."""
        self._current_task = self.get_task(train_or_test, task_idx)
        # Delegate to the mara env so it properly resets its PyBullet scene.
        mara_obs = self._mara_env.reset(train_or_test, task_idx)
        self._current_observation = self._converter.convert_state(mara_obs)
        return self._current_observation.copy()

    def step(self, action: PredAction) -> PredState:
        """Step the mara env directly, avoiding a full _reset_state."""
        from mara_robosim.structs import Action as MaraAction

        mara_obs = self._mara_env.step(MaraAction(action.arr))
        self._current_observation = self._converter.convert_state(mara_obs)
        return self._current_observation.copy()

    def simulate(self, state: PredState, action: PredAction) -> PredState:
        from mara_robosim.structs import Action as MaraAction

        mara_state = self._converter.reverse_state(state)
        # Reset PyBullet from the feature vectors, then get a proper
        # PyBulletState observation before stepping.
        # pylint: disable=protected-access
        self._mara_env._reset_state(mara_state)
        self._mara_env._current_observation = (
            self._mara_env.get_observation())
        # pylint: enable=protected-access
        mara_next = self._mara_env.step(MaraAction(action.arr))
        return self._converter.convert_state(mara_next)

    def _generate_train_tasks(self) -> List[PredEnvironmentTask]:
        mara_tasks = self._mara_env.get_train_tasks()
        return [self._converter.convert_task(t) for t in mara_tasks]

    def _generate_test_tasks(self) -> List[PredEnvironmentTask]:
        mara_tasks = self._mara_env.get_test_tasks()
        return [self._converter.convert_task(t) for t in mara_tasks]

    def render_state_plt(
        self,
        state: PredState,
        task: PredEnvironmentTask,
        action: Optional[PredAction] = None,
        caption: Optional[str] = None,
    ) -> matplotlib.figure.Figure:
        raise NotImplementedError(
            "This mara-robosim adapter does not use Matplotlib rendering.")

    def render(self,
               action: Optional[PredAction] = None,
               caption: Optional[str] = None) -> List[np.ndarray]:
        """Delegate rendering to the mara-robosim env's pybullet camera."""
        return self._mara_env.render(action=None, caption=None)


# ---------------------------------------------------------------------------
# Concrete adapters — one per mara-robosim environment
# ---------------------------------------------------------------------------


class MaraAntsEnv(MaraBaseAdapter):
    """Adapter for mara-robosim Ants environment."""

    @classmethod
    def get_name(cls) -> str:
        return "mara_ants"

    def _build_config(self, use_gui: bool) -> Any:
        from mara_robosim.config import AntsConfig

        kwargs = _build_base_config_kwargs()
        kwargs["use_gui"] = use_gui
        kwargs["ants_attracted_to_points"] = CFG.ants_ants_attracted_to_points
        return AntsConfig(**kwargs)

    def _create_mara_env(self, config: Any, use_gui: bool) -> Any:
        from mara_robosim.envs.ants import PyBulletAntsEnv

        return PyBulletAntsEnv(config=config, use_gui=use_gui)


class MaraBalanceEnv(MaraBaseAdapter):
    """Adapter for mara-robosim Balance environment."""

    @classmethod
    def get_name(cls) -> str:
        return "mara_balance"

    def _build_config(self, use_gui: bool) -> Any:
        from mara_robosim.config import BalanceConfig

        kwargs = _build_base_config_kwargs()
        kwargs["use_gui"] = use_gui
        kwargs["block_size"] = CFG.balance_block_size
        kwargs["num_blocks_train"] = tuple(CFG.balance_num_blocks_train)
        kwargs["num_blocks_test"] = tuple(CFG.balance_num_blocks_test)
        kwargs["holding_goals"] = CFG.balance_holding_goals
        kwargs["weird_balance"] = CFG.balance_wierd_balance
        return BalanceConfig(**kwargs)

    def _create_mara_env(self, config: Any, use_gui: bool) -> Any:
        from mara_robosim.envs.balance import PyBulletBalanceEnv

        return PyBulletBalanceEnv(config=config, use_gui=use_gui)


class MaraBarrierEnv(MaraBaseAdapter):
    """Adapter for mara-robosim Barrier environment."""

    @classmethod
    def get_name(cls) -> str:
        return "mara_barrier"

    def _create_mara_env(self, config: Any, use_gui: bool) -> Any:
        from mara_robosim.envs.barrier import PyBulletBarrierEnv

        return PyBulletBarrierEnv(config=config, use_gui=use_gui)


class MaraBlocksEnv(MaraBaseAdapter):
    """Adapter for mara-robosim Blocks environment."""

    @classmethod
    def get_name(cls) -> str:
        return "mara_blocks"

    def _build_config(self, use_gui: bool) -> Any:
        from mara_robosim.config import BlocksConfig

        kwargs = _build_base_config_kwargs()
        kwargs["use_gui"] = use_gui
        kwargs["block_size"] = CFG.blocks_block_size
        kwargs["num_blocks_train"] = tuple(CFG.blocks_num_blocks_train)
        kwargs["num_blocks_test"] = tuple(CFG.blocks_num_blocks_test)
        kwargs["holding_goals"] = CFG.blocks_holding_goals
        kwargs[
            "high_towers_are_unstable"] = CFG.blocks_high_towers_are_unstable
        return BlocksConfig(**kwargs)

    def _create_mara_env(self, config: Any, use_gui: bool) -> Any:
        from mara_robosim.envs.blocks import PyBulletBlocksEnv

        return PyBulletBlocksEnv(config=config, use_gui=use_gui)


class MaraBoilEnv(MaraBaseAdapter):
    """Adapter for mara-robosim Boil environment."""

    @classmethod
    def get_name(cls) -> str:
        return "mara_boil"

    def _build_config(self, use_gui: bool) -> Any:
        from mara_robosim.config import BoilConfig

        kwargs = _build_base_config_kwargs()
        kwargs["use_gui"] = use_gui
        kwargs["boil_goal"] = CFG.boil_goal
        kwargs[
            "boil_goal_simple_human_happy"] = CFG.boil_goal_simple_human_happy
        kwargs["boil_use_derived_predicates"] = CFG.boil_use_derived_predicates
        kwargs["boil_require_jug_full_to_heatup"] = (
            CFG.boil_require_jug_full_to_heatup)
        kwargs[
            "boil_goal_require_burner_off"] = CFG.boil_goal_require_burner_off
        kwargs["boil_add_jug_reached_capacity_predicate"] = (
            CFG.boil_add_jug_reached_capacity_predicate)
        kwargs["boil_num_jugs_train"] = tuple(CFG.boil_num_jugs_train)
        kwargs["boil_num_jugs_test"] = tuple(CFG.boil_num_jugs_test)
        kwargs["boil_num_burner_train"] = tuple(CFG.boil_num_burner_train)
        kwargs["boil_num_burner_test"] = tuple(CFG.boil_num_burner_test)
        kwargs["boil_water_fill_speed"] = CFG.boil_water_fill_speed
        kwargs["boil_use_skill_factories"] = CFG.boil_use_skill_factories
        kwargs["boil_use_constant_delay"] = CFG.boil_use_constant_delay
        kwargs["boil_use_normal_delay"] = CFG.boil_use_normal_delay
        kwargs["boil_use_cmp_delay"] = CFG.boil_use_cmp_delay
        return BoilConfig(**kwargs)

    def _create_mara_env(self, config: Any, use_gui: bool) -> Any:
        from mara_robosim.envs.boil import PyBulletBoilEnv

        return PyBulletBoilEnv(config=config, use_gui=use_gui)


class MaraCircuitEnv(MaraBaseAdapter):
    """Adapter for mara-robosim Circuit environment."""

    @classmethod
    def get_name(cls) -> str:
        return "mara_circuit"

    def _build_config(self, use_gui: bool) -> Any:
        from mara_robosim.config import CircuitConfig

        kwargs = _build_base_config_kwargs()
        kwargs["use_gui"] = use_gui
        kwargs["circuit_light_doesnt_need_battery"] = (
            CFG.circuit_light_doesnt_need_battery)
        kwargs["circuit_battery_in_box"] = CFG.circuit_battery_in_box
        return CircuitConfig(**kwargs)

    def _create_mara_env(self, config: Any, use_gui: bool) -> Any:
        from mara_robosim.envs.circuit import PyBulletCircuitEnv

        return PyBulletCircuitEnv(config=config, use_gui=use_gui)


class MaraCoffeeEnv(MaraBaseAdapter):
    """Adapter for mara-robosim Coffee environment."""

    @classmethod
    def get_name(cls) -> str:
        return "mara_coffee"

    def _build_config(self, use_gui: bool) -> Any:
        from mara_robosim.config import CoffeeConfig

        kwargs = _build_base_config_kwargs()
        kwargs["use_gui"] = use_gui
        kwargs["num_cups_train"] = tuple(CFG.coffee_num_cups_train)
        kwargs["num_cups_test"] = tuple(CFG.coffee_num_cups_test)
        kwargs["rotated_jug_ratio"] = CFG.coffee_rotated_jug_ratio
        kwargs["use_pixelated_jug"] = CFG.coffee_use_pixelated_jug
        kwargs["jug_pickable_pred"] = CFG.coffee_jug_pickable_pred
        kwargs["simple_tasks"] = CFG.coffee_simple_tasks
        kwargs["machine_have_light_bar"] = CFG.coffee_machine_have_light_bar
        kwargs["machine_has_plug"] = CFG.coffee_machine_has_plug
        kwargs["plug_break_after_plugged_in"] = (
            CFG.coffee_plug_break_after_plugged_in)
        kwargs["fill_jug_gradually"] = CFG.coffee_fill_jug_gradually
        kwargs["render_grid_world"] = CFG.coffee_render_grid_world
        return CoffeeConfig(**kwargs)

    def _create_mara_env(self, config: Any, use_gui: bool) -> Any:
        from mara_robosim.envs.coffee import PyBulletCoffeeEnv

        return PyBulletCoffeeEnv(config=config, use_gui=use_gui)


class MaraCoverEnv(MaraBaseAdapter):
    """Adapter for mara-robosim Cover environment."""

    @classmethod
    def get_name(cls) -> str:
        return "mara_cover"

    def _build_config(self, use_gui: bool) -> Any:
        from mara_robosim.config import CoverConfig

        kwargs = _build_base_config_kwargs()
        kwargs["use_gui"] = use_gui
        kwargs["cover_num_blocks"] = CFG.cover_num_blocks
        kwargs["cover_num_targets"] = CFG.cover_num_targets
        kwargs["cover_block_widths"] = tuple(CFG.cover_block_widths)
        kwargs["cover_target_widths"] = tuple(CFG.cover_target_widths)
        kwargs["cover_initial_holding_prob"] = CFG.cover_initial_holding_prob
        kwargs["cover_blocks_change_color_when_cover"] = (
            CFG.cover_blocks_change_color_when_cover)
        return CoverConfig(**kwargs)

    def _create_mara_env(self, config: Any, use_gui: bool) -> Any:
        from mara_robosim.envs.cover import PyBulletCoverEnv

        return PyBulletCoverEnv(config=config, use_gui=use_gui)


class MaraDominoEnv(MaraBaseAdapter):
    """Adapter for mara-robosim Domino environment."""

    @classmethod
    def get_name(cls) -> str:
        return "mara_domino"

    def _build_config(self, use_gui: bool) -> Any:
        from mara_robosim.config import DominoConfig

        kwargs = _build_base_config_kwargs()
        kwargs["use_gui"] = use_gui
        kwargs["domino_debug_layout"] = CFG.domino_debug_layout
        kwargs["domino_some_dominoes_are_connected"] = (
            CFG.domino_some_dominoes_are_connected)
        kwargs["domino_initialize_at_finished_state"] = (
            CFG.domino_initialize_at_finished_state)
        kwargs["domino_use_domino_blocks_as_target"] = (
            CFG.domino_use_domino_blocks_as_target)
        kwargs["domino_use_grid"] = CFG.domino_use_grid
        kwargs["domino_include_connected_predicate"] = (
            CFG.domino_include_connected_predicate)
        kwargs["domino_has_glued_dominos"] = CFG.domino_has_glued_dominos
        kwargs["domino_prune_actions"] = CFG.domino_prune_actions
        kwargs["domino_only_straight_sequence_in_training"] = (
            CFG.domino_only_straight_sequence_in_training)
        kwargs["domino_train_num_dominos"] = tuple(
            CFG.domino_train_num_dominos)
        kwargs["domino_test_num_dominos"] = tuple(CFG.domino_test_num_dominos)
        kwargs["domino_train_num_targets"] = tuple(
            CFG.domino_train_num_targets)
        kwargs["domino_test_num_targets"] = tuple(CFG.domino_test_num_targets)
        kwargs["domino_train_num_pivots"] = tuple(CFG.domino_train_num_pivots)
        kwargs["domino_test_num_pivots"] = tuple(CFG.domino_test_num_pivots)
        kwargs["domino_oracle_knows_glued_dominos"] = (
            CFG.domino_oracle_knows_glued_dominos)
        kwargs["domino_use_continuous_place"] = CFG.domino_use_continuous_place
        kwargs["domino_restricted_push"] = CFG.domino_restricted_push
        kwargs["domino_use_skill_factories"] = CFG.domino_use_skill_factories
        return DominoConfig(**kwargs)

    def _create_mara_env(self, config: Any, use_gui: bool) -> Any:
        from mara_robosim.envs.domino.composed_env import \
            PyBulletDominoComposedEnv

        return PyBulletDominoComposedEnv(  # type: ignore[call-arg]  # pylint: disable=no-value-for-parameter
            config=config,
            use_gui=use_gui)


class MaraFanEnv(MaraBaseAdapter):
    """Adapter for mara-robosim Fan environment."""

    @classmethod
    def get_name(cls) -> str:
        return "mara_fan"

    def _build_config(self, use_gui: bool) -> Any:
        from mara_robosim.config import FanConfig

        kwargs = _build_base_config_kwargs()
        kwargs["use_gui"] = use_gui
        kwargs["fan_use_skill_factories"] = CFG.fan_use_skill_factories
        kwargs["fan_fans_blow_opposite_direction"] = (
            CFG.fan_fans_blow_opposite_direction)
        kwargs["fan_known_controls_relation"] = CFG.fan_known_controls_relation
        kwargs["fan_combine_switch_on_off"] = CFG.fan_combine_switch_on_off
        kwargs["fan_use_kinematic"] = CFG.fan_use_kinematic
        return FanConfig(**kwargs)

    def _create_mara_env(self, config: Any, use_gui: bool) -> Any:
        from mara_robosim.envs.fan import PyBulletFanEnv

        return PyBulletFanEnv(config=config, use_gui=use_gui)


class MaraFloatEnv(MaraBaseAdapter):
    """Adapter for mara-robosim Float environment."""

    @classmethod
    def get_name(cls) -> str:
        return "mara_float"

    def _build_config(self, use_gui: bool) -> Any:
        from mara_robosim.config import FloatConfig

        kwargs = _build_base_config_kwargs()
        kwargs["use_gui"] = use_gui
        kwargs["water_level_doesnt_raise"] = CFG.float_water_level_doesnt_raise
        return FloatConfig(**kwargs)

    def _create_mara_env(self, config: Any, use_gui: bool) -> Any:
        from mara_robosim.envs.float_ import PyBulletFloatEnv

        return PyBulletFloatEnv(config=config, use_gui=use_gui)


class MaraGrowEnv(MaraBaseAdapter):
    """Adapter for mara-robosim Grow environment."""

    @classmethod
    def get_name(cls) -> str:
        return "mara_grow"

    def _build_config(self, use_gui: bool) -> Any:
        from mara_robosim.config import GrowConfig

        kwargs = _build_base_config_kwargs()
        kwargs["use_gui"] = use_gui
        kwargs["grow_use_skill_factories"] = CFG.grow_use_skill_factories
        kwargs["grow_num_cups_train"] = tuple(CFG.grow_num_cups_train)
        kwargs["grow_num_cups_test"] = tuple(CFG.grow_num_cups_test)
        kwargs["grow_num_jugs_train"] = tuple(CFG.grow_num_jugs_train)
        kwargs["grow_num_jugs_test"] = tuple(CFG.grow_num_jugs_test)
        return GrowConfig(**kwargs)

    def _create_mara_env(self, config: Any, use_gui: bool) -> Any:
        from mara_robosim.envs.grow import PyBulletGrowEnv

        return PyBulletGrowEnv(config=config, use_gui=use_gui)


class MaraLaserEnv(MaraBaseAdapter):
    """Adapter for mara-robosim Laser environment."""

    @classmethod
    def get_name(cls) -> str:
        return "mara_laser"

    def _build_config(self, use_gui: bool) -> Any:
        from mara_robosim.config import LaserConfig

        kwargs = _build_base_config_kwargs()
        kwargs["use_gui"] = use_gui
        kwargs["laser_zero_reflection_angle"] = CFG.laser_zero_reflection_angle
        kwargs["laser_use_debug_line_for_beams"] = (
            CFG.laser_use_debug_line_for_beams)
        return LaserConfig(**kwargs)

    def _create_mara_env(self, config: Any, use_gui: bool) -> Any:
        from mara_robosim.envs.laser import PyBulletLaserEnv

        return PyBulletLaserEnv(config=config, use_gui=use_gui)


class MaraMagicBinEnv(MaraBaseAdapter):
    """Adapter for mara-robosim MagicBin environment."""

    @classmethod
    def get_name(cls) -> str:
        return "mara_magic_bin"

    def _create_mara_env(self, config: Any, use_gui: bool) -> Any:
        from mara_robosim.envs.magic_bin import PyBulletMagicBinEnv

        return PyBulletMagicBinEnv(config=config, use_gui=use_gui)


class MaraSwitchEnv(MaraBaseAdapter):
    """Adapter for mara-robosim Switch environment."""

    @classmethod
    def get_name(cls) -> str:
        return "mara_switch"

    def _create_mara_env(self, config: Any, use_gui: bool) -> Any:
        from mara_robosim.envs.switch import PyBulletSwitchEnv

        return PyBulletSwitchEnv(config=config, use_gui=use_gui)
