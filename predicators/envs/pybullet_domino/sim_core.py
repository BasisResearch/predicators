"""Observable simulation core of the domino environment.

This module is the domino env's BASE SIM: the workspace, the two
tables, the robot's configuration, the domino shape and material
constants, the composition of the scene's components into one PyBullet
world, state read/write, and the domino physical-parameter registry
that a system-ID fit adjusts - everything needed to run rigid-body
rollouts of the table. The domino bodies themselves are built and reset
in :mod:`predicators.envs.pybullet_domino.components.domino_bodies`.
It deliberately contains NO per-step domain dynamics (that is the
concrete env's ``PyBulletDominoComposedEnv`` step), no instance-specific
physical parameters, no task generation, and no predicate / goal
semantics.

That boundary is a visibility contract, enforced structurally rather
than by redaction: when ``CFG.agent_sim_provide_base_sim_source`` is
on, the files ``get_base_sim_source_files`` lists (this one among them)
are copied verbatim into the learning agent's sandbox as reference
material ("the robot knows its own simulator"), so the file the agent
reads is byte-identical to the code its base-sim rollouts execute.
Anything that would leak the learning target - the physics the world
really runs, the task distribution, goal thresholds - lives in
``env.py`` and the other unlisted modules, never here.

Physical layout (a tabletop, the robot at the near side):

- Two tables side by side, the second offset by half a table width in
  ``y``, so the workspace spans both.
- The ``robot``: a single arm whose base sits in front of the tables,
  facing them.
- Scene objects come from the components (dominoes, blocks, targets,
  pivots, ...); each component builds its own bodies and resets them
  from a state.
"""

from typing import Any, ClassVar, Dict, FrozenSet, List, Optional, Sequence, \
    Set, Tuple

import numpy as np
import pybullet as p

from predicators.envs.pybullet_domino.components.base_component import \
    DominoEnvComponent
from predicators.envs.pybullet_domino.components.domino_bodies import \
    DominoBodiesComponent
from predicators.envs.pybullet_env import PyBulletEnv
from predicators.pybullet_helpers.geometry import Pose3D, Quaternion
from predicators.pybullet_helpers.objects import create_object
from predicators.pybullet_helpers.robots import SingleArmPyBulletRobot
from predicators.structs import Object, State, Type


class PyBulletDominoBaseEnv(PyBulletEnv):
    """Sim core of the domino table: a robot, two tables, and the bodies of a
    list of components.

    Abstract on purpose - it defines no name, predicates, tasks, or
    domain-specific step, so env discovery skips it; the concrete env
    is ``PyBulletDominoComposedEnv`` and its subclasses.
    """

    # The menu below already exposes the dominoes and blocks' mass and friction.
    CALIBRATION_COVERED_TYPES: ClassVar[FrozenSet[str]] = frozenset(
        {"domino", "block"})

    @classmethod
    def get_base_sim_source_files(cls) -> List[str]:
        # These modules ARE the visible sim core (see the module
        # docstring's visibility contract); pybullet_env.py is the generic
        # engine they are built on. env.py, domino_component.py and the
        # other components, the task generators and the cascade
        # certificate/probe must never be listed here.
        return [
            "predicators/envs/pybullet_domino/sim_core.py",
            "predicators/envs/pybullet_domino/components/base_component.py",
            "predicators/envs/pybullet_domino/components/domino_bodies.py",
            "predicators/envs/pybullet_env.py",
        ]

    # =========================================================================
    # TABLE / WORKSPACE CONFIGURATION
    # =========================================================================
    table_height: ClassVar[float] = 0.4
    table_pos: ClassVar[Pose3D] = (0.75, 1.35, table_height / 2)
    table_orn: ClassVar[Quaternion] = tuple(
        p.getQuaternionFromEuler([0., 0., np.pi / 2]))
    table_width: ClassVar[float] = 1.0

    x_lb: ClassVar[float] = 0.4
    x_ub: ClassVar[float] = 1.1
    y_lb: ClassVar[float] = 1.1
    y_ub: ClassVar[float] = 1.6
    z_lb: ClassVar[float] = table_height
    z_ub: ClassVar[float] = 0.95

    # =========================================================================
    # ROBOT CONFIGURATION
    # =========================================================================
    robot_init_x: ClassVar[float] = (x_lb + x_ub) * 0.5
    robot_init_y: ClassVar[float] = (y_lb + y_ub) * 0.5
    robot_init_z: ClassVar[float] = z_ub
    robot_base_pos: ClassVar[Optional[Tuple[float, float,
                                            float]]] = (0.75, 0.72, 0.0)
    robot_base_orn: ClassVar[Optional[Tuple[float, float, float, float]]] = \
        tuple(p.getQuaternionFromEuler([0.0, 0.0, np.pi / 2]))
    robot_init_tilt: ClassVar[float] = np.pi / 2
    robot_init_wrist: ClassVar[float] = -np.pi / 2

    # =========================================================================
    # CAMERA CONFIGURATION
    # =========================================================================
    _camera_distance: ClassVar[float] = 1.3
    _camera_yaw: ClassVar[float] = 210
    _camera_pitch: ClassVar[float] = -14
    _camera_target: ClassVar[Pose3D] = (0.75, 1.25, 0.65)

    # =========================================================================
    # DOMINO CONFIGURATION
    # =========================================================================
    # Domino shape properties and the built-in material a fresh domino
    # body is created with (see ``create_domino_block``).
    domino_width: ClassVar[float] = 0.07
    domino_depth: ClassVar[float] = 0.015
    domino_height: ClassVar[float] = 0.15
    domino_mass: ClassVar[float] = 0.1
    domino_friction: ClassVar[float] = 0.5
    pos_gap: ClassVar[float] = 0.098  # domino_width * 1.4, computed value

    # Type definitions
    _robot_type = Type("robot",
                       ["x", "y", "z", "fingers", "roll", "tilt", "wrist"],
                       angular_features=["roll", "tilt", "wrist"])
    _out_of_view_xy: ClassVar[Sequence[float]] = [10.0, 10.0]

    def __init__(self,
                 components: List[DominoEnvComponent],
                 use_gui: bool = False,
                 **kwargs: Any) -> None:
        """Build the PyBullet world holding every component's bodies.

        Args:
            components: List of components to include in the environment.
            use_gui: Whether to use PyBullet GUI.
        """
        self._components = components

        # Create robot object
        self._robot = Object("robot", self._robot_type)

        # The domino component, for convenience
        self._domino_component: Optional[DominoBodiesComponent] = None
        for comp in components:
            if isinstance(comp, DominoBodiesComponent):
                self._domino_component = comp

        super().__init__(use_gui, **kwargs)
        for component in self._components:
            self._body_objects.update(
                {obj.name: obj
                 for obj in component.get_objects()})

        self._configure_instance_physics()
        # Snapshot the believed baseline AFTER any instance adjustment:
        # ``get_physical_param_info`` reports these values as the defaults,
        # and the sysID revert path restores dropped params to them (the
        # instance attrs alone miss init-time overrides).
        self._physical_param_baseline: Dict[str, float] = (
            self._domino_component.physical_param_override
            if self._domino_component is not None else {})

    def _configure_instance_physics(self) -> None:
        """Hook run once the bodies exist, before the physical-param baseline
        is snapshot.

        A subclass may apply instance-specific physical params here (via
        ``set_domino_physical_params``); they then become this
        instance's reported defaults. The base sim applies none.
        """

    # =========================================================================
    # PROPERTIES
    # =========================================================================

    @property
    def types(self) -> Set[Type]:
        """Return all types from all components plus robot type."""
        all_types = {self._robot_type}
        for comp in self._components:
            all_types |= comp.get_types()
        return all_types

    # =========================================================================
    # PYBULLET INITIALIZATION
    # =========================================================================

    @classmethod
    def initialize_pybullet(
            cls, using_gui: bool
    ) -> Tuple[int, SingleArmPyBulletRobot, Dict[str, Any]]:
        """Initialize PyBullet simulation.

        Note: Component initialization happens in instance method since
        components are instance-specific.
        """
        # Reuse the base setup (connection, plane + studio floor, robot,
        # gravity, backdrop walls), then add this env's two tables. The tables
        # are textured centrally by _apply_studio_table_textures.
        physics_client_id, pybullet_robot, bodies = super(
        ).initialize_pybullet(using_gui)

        # Two tables side by side for extra workspace.
        bodies["table_id"] = create_object(asset_path="urdf/table.urdf",
                                           position=cls.table_pos,
                                           orientation=cls.table_orn,
                                           scale=1.0,
                                           use_fixed_base=True,
                                           physics_client_id=physics_client_id)
        bodies["table_id2"] = create_object(
            asset_path="urdf/table.urdf",
            position=(cls.table_pos[0], cls.table_pos[1] + cls.table_width / 2,
                      cls.table_pos[2]),
            orientation=cls.table_orn,
            scale=1.0,
            use_fixed_base=True,
            physics_client_id=physics_client_id)
        return physics_client_id, pybullet_robot, bodies

    def _store_pybullet_bodies(self, pybullet_bodies: Dict[str, Any]) -> None:
        """Initialize and store PyBullet bodies for all components."""
        self._table_ids = [
            pybullet_bodies["table_id"], pybullet_bodies["table_id2"]
        ]
        # Initialize each component
        for comp in self._components:
            comp.set_physics_client_id(self._physics_client_id)
            comp_bodies = comp.initialize_pybullet(self._physics_client_id)
            comp.store_pybullet_bodies(comp_bodies)

    # =========================================================================
    # STATE MANAGEMENT
    # =========================================================================

    def _get_object_ids_for_held_check(self) -> List[int]:
        """Return object IDs that can be held by robot."""
        ids = []
        for comp in self._components:
            ids.extend(comp.get_object_ids_for_held_check())
        return ids

    def _get_domain_specific_feature(self, obj: Object, feature: str) -> float:
        """Extract state feature for an object."""
        result = self._component_feature(obj, feature)
        if result is not None:
            return result
        raise ValueError(f"Unknown feature {feature} for object {obj}")

    def _component_feature(self, obj: Object, feature: str) -> Optional[float]:
        """The feature as the first component that owns it reports it, or None
        when no live component does."""
        for comp in self._components:
            result = comp.extract_feature(obj, feature)
            if result is not None:
                return result
        return None

    def _set_domain_specific_state(self, state: State) -> None:
        """Reset each component to match the state."""
        for comp in self._components:
            comp.reset_state(state)

    def robot_init_state_dict(self) -> Dict[str, float]:
        """The robot's initial feature dict, shared by every task scene."""
        return {
            "x": self.robot_init_x,
            "y": self.robot_init_y,
            "z": self.robot_init_z,
            "fingers": self.open_fingers,
            "roll": self.robot_init_roll,
            "tilt": self.robot_init_tilt,
            "wrist": self.robot_init_wrist,
        }

    @classmethod
    def _default_workspace_bounds(cls) -> Dict[str, float]:
        """Workspace bounds shared by all concrete domino environments."""
        return {
            "x_lb": cls.x_lb,
            "x_ub": cls.x_ub,
            "y_lb": cls.y_lb,
            "y_ub": cls.y_ub,
            "z_lb": cls.z_lb,
            "z_ub": cls.z_ub,
        }

    # =========================================================================
    # PHYSICAL PARAMETERS (system-ID surface)
    # =========================================================================

    def set_domino_physical_params(self, **params: Optional[float]) -> None:
        """Override this env instance's domino PyBullet dynamics params.

        Thin delegate to ``DominoBodiesComponent.set_physical_params``
        (accepts ``mass``, ``lateral_friction``, ``restitution``,
        ``rolling_friction``, ``spinning_friction`` and the ``block_*``
        family). Lets a caller run two env instances with divergent
        physics in one process without touching the shared ClassVars.
        No-op if there is no domino component.
        """
        if self._domino_component is not None:
            self._domino_component.set_physical_params(**params)

    def get_physical_param_info(self) -> Dict[str, Dict[str, Any]]:
        """Tunable domino dynamics params (see BaseEnv docstring).

        These are the parameters ``set_domino_physical_params`` accepts;
        defaults mirror what ``create_domino_block`` bakes into fresh
        bodies. All are global scalars shared by every (identical)
        domino body.
        """
        comp = self._domino_component
        if comp is None:
            return self._calibration_info()
        # Defaults report the believed BASELINE of this instance: the
        # post-init override snapshot when present, else the built-in
        # value. The sysID revert path restores dropped params to these
        # defaults, so they must be the values the env would have without
        # any fit.
        baseline = getattr(self, "_physical_param_baseline", {})
        lateral_friction = baseline.get("lateral_friction",
                                        comp.domino_friction)
        # ``scale: "log"`` marks positive scale-like parameters whose
        # behavioral effect is multiplicative: the sysID fit runs in
        # log-space for them (geometric grid sweep, relative LM steps,
        # log-normal prior). A linear parameterization has almost no
        # resolution at the low end of a box spanning decades -
        # linspace(0.01, 2, 8) has no candidate between 0.01 and 0.29.
        # Params whose lo is 0 (restitution, rolling_friction) stay
        # linear.
        info: Dict[str, Dict[str, Any]] = {
            "lateral_friction": {
                "default":
                lateral_friction,
                "lo":
                0.01,
                "hi":
                2.0,
                "scale":
                "log",
                "description":
                "Lateral (sliding) friction of each domino against the "
                "table and other dominoes (PyBullet lateralFriction); "
                "governs how far a toppling domino slides/rotates and "
                "whether a cascade propagates.",
            },
            "restitution": {
                "default":
                baseline.get("restitution", 0.02),
                "lo":
                0.0,
                "hi":
                0.9,
                "description":
                "Bounciness of domino-domino impacts (the table's "
                "restitution is 0, and PyBullet combines them "
                "multiplicatively, so this only manifests in "
                "domino-on-domino collisions).",
            },
            "mass": {
                "default":
                baseline.get("mass", comp.domino_mass),
                "lo":
                0.005,
                "hi":
                1.0,
                "scale":
                "log",
                "description":
                "Mass of each (non-glued) domino in kg. Largely scales "
                "out of the topple condition for identical dominoes.",
            },
            "rolling_friction": {
                "default":
                baseline.get("rolling_friction", 0.006),
                "lo":
                0.0,
                "hi":
                0.1,
                "description":
                "Rolling-friction coefficient; damps edge-rolling of a "
                "tipping domino.",
            },
            "spinning_friction": {
                "default":
                # Bodies are created with spinningFriction = the built-in
                # lateral value; a lateral_friction override does NOT
                # retouch it, so the baseline follows the ClassVar.
                baseline.get("spinning_friction", comp.domino_friction),
                "lo":
                0.01,
                "hi":
                2.0,
                "scale":
                "log",
                "description":
                "Spin (yaw) friction against the table; defaults to the "
                "lateral friction value at body creation.",
            },
        }
        # Gray ``block``-typed bodies form their own parameter class:
        # the ``block_*`` family applies to them only (and beats the
        # global param for those bodies).
        if comp.blocks:
            info["block_mass"] = {
                "default":
                baseline.get("block_mass", comp.domino_mass),
                "lo":
                0.005,
                "hi":
                2000.0,
                "scale":
                "log",
                "description":
                "Mass in kg of each block (the gray block type); applies "
                "to block bodies only, independently of the dominoes' "
                "``mass``.",
            }
            info["block_lateral_friction"] = {
                "default":
                baseline.get(
                    "block_lateral_friction",
                    baseline.get("lateral_friction", comp.domino_friction)),
                "lo":
                0.01,
                "hi":
                2.0,
                "scale":
                "log",
                "description":
                "Lateral (sliding) friction of each block (the gray "
                "block type) against the table and other bodies; applies "
                "to block bodies only.",
            }
        info.update(self._calibration_info())
        return info

    def apply_physical_param_overrides(self, params: Dict[str, float]) -> None:
        """Sticky in-place dynamics override (delegates to the domino
        component's ``set_physical_params``, which re-applies after every reset
        and body recreation)."""
        params = self._take_calibration_params(params)
        unknown = set(params) - set(self.get_physical_param_info())
        if unknown:
            raise ValueError(f"Unknown physical param(s) {sorted(unknown)}.")
        self.set_domino_physical_params(**params)
