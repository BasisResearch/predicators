"""Observable simulation core of the domino component: the domino bodies.

This module is part of the domino env's BASE SIM (see
:mod:`predicators.envs.pybullet_domino.sim_core`): the domino, block,
target and pivot types and objects, the PyBullet bodies behind them,
their state reset, and the per-instance physical-parameter override
that a system-ID fit writes into the live bodies. It deliberately
contains no predicates, no goal semantics and no task generation.

The boundary is a visibility contract: when
``CFG.agent_sim_provide_base_sim_source`` is on, THIS FILE is copied
verbatim into the learning agent's sandbox as reference material, so
the file the agent reads is byte-identical to the code its base-sim
rollouts execute. Predicates and their thresholds, role-specific body
masses and task-layout helpers live in the concrete ``DominoComponent``
(``domino_component.py``), never here.

Physical layout:

- ``domino`` objects: thin boxes standing on the table, each a free
  rigid body. Features: pose (``x``, ``y``, ``z``, ``yaw``, ``roll``),
  colour (``r``, ``g``, ``b``) and ``is_held``. A domino's colour is
  written into its body at every reset.
- ``block`` objects: bodies of the same shape and feature layout as a
  domino, drawn from the same body pool, but a distinct type.
- ``target`` / ``pivot`` objects: fixed-base URDF bodies with a hinged
  flap (joint ``flap_hinge_joint``), reset with the flap at rest.

Bodies absent from a state are parked out of view, off the table.
"""

from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Optional, \
    Sequence, Set, Tuple
from typing import Type as TypingType

import pybullet as p

from predicators.envs.pybullet_domino.components.base_component import \
    DominoEnvComponent
from predicators.pybullet_helpers.geometry import Pose3D, Quaternion
from predicators.pybullet_helpers.objects import create_object, \
    create_pybullet_block, update_object
from predicators.settings import CFG
from predicators.structs import Object, State, Type

if TYPE_CHECKING:
    from predicators.envs.pybullet_domino.sim_core import PyBulletDominoBaseEnv


class DominoBodiesComponent(DominoEnvComponent):
    """Sim core of the domino component: types, objects, bodies, reset.

    Abstract on purpose - it defines no predicates; the concrete
    component is ``DominoComponent``.

    Note: domino_width, domino_depth, domino_height, domino_mass, and
    domino_friction are defined on ``PyBulletDominoBaseEnv``.
    """

    # Domino colors: the start block (green), the target blocks (pink)
    # and the ordinary blocks (blue). A body's colour is a state feature.
    start_domino_color: ClassVar[Tuple[float, float, float,
                                       float]] = (0.56, 0.93, 0.56, 1.)
    target_domino_color: ClassVar[Tuple[float, float, float,
                                        float]] = (0.85, 0.7, 0.85, 1.0)
    domino_color: ClassVar[Tuple[float, float, float,
                                 float]] = (0.6, 0.8, 1.0, 1.0)

    @staticmethod
    def _get_env_class() -> TypingType["PyBulletDominoBaseEnv"]:
        """Get the sim-core env class to access shared config."""
        from predicators.envs.pybullet_domino.sim_core import \
            PyBulletDominoBaseEnv  # pylint: disable=import-outside-toplevel
        return PyBulletDominoBaseEnv

    @property
    def domino_width(self) -> float:
        """Domino width."""
        if self._dim_override["width"] is not None:
            return self._dim_override["width"]
        return self._get_env_class().domino_width

    @property
    def domino_depth(self) -> float:
        """Domino depth."""
        if self._dim_override["depth"] is not None:
            return self._dim_override["depth"]
        return self._get_env_class().domino_depth

    @property
    def domino_height(self) -> float:
        """Domino height."""
        if self._dim_override["height"] is not None:
            return self._dim_override["height"]
        return self._get_env_class().domino_height

    @property
    def domino_mass(self) -> float:
        """Domino mass."""
        return self._get_env_class().domino_mass

    @property
    def domino_friction(self) -> float:
        """Domino friction."""
        return self._get_env_class().domino_friction

    @property
    def pos_gap(self) -> float:
        """Pos gap."""
        return self._get_env_class().pos_gap

    def __init__(self,
                 num_dominos_max: int = 9,
                 num_targets_max: int = 3,
                 num_pivots_max: int = 3,
                 workspace_bounds: Optional[Dict[str, float]] = None,
                 domino_width: Optional[float] = None,
                 domino_depth: Optional[float] = None,
                 domino_height: Optional[float] = None) -> None:
        """Initialize the domino bodies.

        Args:
            num_dominos_max: Maximum number of domino blocks.
            num_targets_max: Maximum number of target objects.
            num_pivots_max: Maximum number of pivot objects.
            workspace_bounds: Dict with x/y/z lower/upper bounds.
            domino_width/depth/height: per-component dimension overrides (m).
                None (default) falls back to the shared
                PyBulletDominoBaseEnv ClassVars.
        """
        super().__init__()

        self.num_dominos_max = num_dominos_max
        self.num_targets_max = num_targets_max
        self.num_pivots_max = num_pivots_max
        self._dim_override = {
            "width": domino_width,
            "depth": domino_depth,
            "height": domino_height
        }

        # Workspace bounds (will be set by composed env if not provided)
        if workspace_bounds is None:
            workspace_bounds = {
                "x_lb": 0.4,
                "x_ub": 1.1,
                "y_lb": 1.1,
                "y_ub": 1.6,
                "z_lb": 0.4,  # table_height
                "z_ub": 0.95
            }
        self.x_lb = workspace_bounds["x_lb"]
        self.x_ub = workspace_bounds["x_ub"]
        self.y_lb = workspace_bounds["y_lb"]
        self.y_ub = workspace_bounds["y_ub"]
        self.z_lb = workspace_bounds["z_lb"]
        self.z_ub = workspace_bounds["z_ub"]

        # Create types. yaw/roll are radians: marking them angular lets
        # consumers that difference states (sysID residuals) wrap errors
        # to [-pi, pi] instead of scoring -pi vs +pi as a 2*pi mistake.
        self._domino_type = Type(
            "domino",
            ["x", "y", "z", "yaw", "roll", "r", "g", "b", "is_held"],
            angular_features=["yaw", "roll"],
        )
        # A separate class for block bodies: same feature layout as a
        # domino (so the shared body pool and state assembly stay
        # uniform), but a distinct type, so typed options (Pick/Place/Push
        # take dominoes) structurally exclude them and the physical-param
        # registry can expose a per-class ``block_*`` parameter family.
        self._block_type = Type(
            "block",
            ["x", "y", "z", "yaw", "roll", "r", "g", "b", "is_held"],
            angular_features=["yaw", "roll"],
        )
        self._target_type = Type("target", ["x", "y", "z", "yaw"],
                                 sim_features=["id", "joint_id"],
                                 angular_features=["yaw"])
        self._pivot_type = Type("pivot", ["x", "y", "z", "yaw"],
                                sim_features=["id", "joint_id"],
                                angular_features=["yaw"])

        # Create objects. With domino blocks as targets, the target slots
        # join the domino body pool instead of using hinged target bodies.
        use_domino_as_target = CFG.domino_use_domino_blocks_as_target
        if use_domino_as_target:
            num_dominos = self.num_dominos_max + self.num_targets_max
            num_targets = 0
        else:
            num_dominos = self.num_dominos_max
            num_targets = self.num_targets_max

        self.dominos: List[Object] = []
        for i in range(num_dominos):
            obj = Object(f"domino_{i}", self._domino_type)
            self.dominos.append(obj)
        # ``block``-typed objects occupying slots of the domino body pool
        # (the body and all slot-indexed machinery are shared; only the
        # object identity differs). None by default.
        self.blocks: List[Object] = []

        self.targets: List[Object] = []
        for i in range(num_targets):
            obj = Object(f"target_{i}", self._target_type)
            self.targets.append(obj)

        self.pivots: List[Object] = []
        for i in range(self.num_pivots_max):
            obj = Object(f"pivot_{i}", self._pivot_type)
            self.pivots.append(obj)

        # Constraint tracking for connected dominoes
        self.block_constraints: List[int] = []
        # Bodies whose mass was pinned at the last reset (see
        # ``_assign_body_masses``); both lists are rebuilt on every reset.
        # ``fixed_domino_ids`` are shielded from the generic ``mass``
        # override; ``block_body_ids`` are the block-class bodies the
        # ``block_*`` overrides apply to.
        self.fixed_domino_ids: List[int] = []
        self.block_body_ids: List[int] = []

        # Optional per-instance override of PyBullet contact/inertial params
        # (mass, friction, restitution, ...). Empty by default, so the env
        # behaves exactly as its ClassVars dictate. Set via
        # ``set_physical_params`` to make one env instance's physics diverge
        # from another's in the same process without touching the shared
        # ClassVars. Re-applied at the end of ``reset_state`` because reset
        # rewrites domino mass.
        self._physical_param_override: Dict[str, float] = {}

    # -------------------------------------------------------------------------
    # DominoEnvComponent interface implementation
    # -------------------------------------------------------------------------

    def get_types(self) -> Set[Type]:
        types = {self._domino_type}
        if self.blocks:
            types.add(self._block_type)
        if self.targets:
            types.add(self._target_type)
        if self.pivots:
            types.add(self._pivot_type)
        return types

    def get_objects(self) -> List[Object]:
        return self.dominos + self.targets + self.pivots

    def initialize_pybullet(self, physics_client_id: int) -> Dict[str, Any]:
        """Create PyBullet bodies for dominoes, targets, and pivots."""
        self._physics_client_id = physics_client_id
        bodies: Dict[str, Any] = {}

        # Create dominoes
        domino_ids = []
        num_dominos_to_create = len(self.dominos)
        for i in range(num_dominos_to_create):
            domino_id = create_domino_block(
                color=self.start_domino_color if i == 0 else self.domino_color,
                half_extents=(self.domino_width / 2, self.domino_depth / 2,
                              self.domino_height / 2),
                mass=self.domino_mass,
                friction=self.domino_friction,
                orientation=(0.0, 0.0, 0.0, 1.0),
                physics_client_id=physics_client_id,
                add_top_triangle=True,
            )
            domino_ids.append(domino_id)
        bodies["domino_ids"] = domino_ids

        # Create targets
        target_ids = []
        for _ in self.targets:
            tid = create_object("urdf/domino_target.urdf",
                                position=(self.x_lb, self.y_lb, self.z_lb),
                                orientation=p.getQuaternionFromEuler(
                                    [0.0, 0.0, 0.0]),
                                scale=1.0,
                                use_fixed_base=True,
                                physics_client_id=physics_client_id)
            target_ids.append(tid)
        bodies["target_ids"] = target_ids

        # Create pivots
        pivot_ids = []
        for _ in self.pivots:
            pid = create_object("urdf/domino_pivot.urdf",
                                position=(self.x_lb, self.y_lb, self.z_lb),
                                orientation=p.getQuaternionFromEuler(
                                    [0.0, 0.0, 0.0]),
                                scale=1.0,
                                use_fixed_base=True,
                                physics_client_id=physics_client_id)
            pivot_ids.append(pid)
        bodies["pivot_ids"] = pivot_ids

        return bodies

    def store_pybullet_bodies(self, pybullet_bodies: Dict[str, Any]) -> None:
        """Store PyBullet body IDs on objects."""
        for domino, id_ in zip(self.dominos, pybullet_bodies["domino_ids"]):
            domino.id = id_

        for target, id_ in zip(self.targets, pybullet_bodies["target_ids"]):
            target.id = id_
            assert self._physics_client_id is not None
            target.joint_id = self._get_joint_id(id_, "flap_hinge_joint",
                                                 self._physics_client_id)

        for pivot, id_ in zip(self.pivots, pybullet_bodies["pivot_ids"]):
            pivot.id = id_
            assert self._physics_client_id is not None
            pivot.joint_id = self._get_joint_id(id_, "flap_hinge_joint",
                                                self._physics_client_id)
        # A freshly (re)created body has default ClassVar dynamics; re-assert
        # any standing override so it survives body recreation as well as
        # reset (see set_physical_params).
        self._apply_physical_param_override()

    # -------------------------------------------------------------------------
    # Per-instance physical-parameter override (system-ID)
    # -------------------------------------------------------------------------

    _PHYSICAL_PARAM_KEYS = frozenset({
        "mass", "lateral_friction", "restitution", "rolling_friction",
        "spinning_friction", "block_mass", "block_lateral_friction"
    })

    # Map override keys -> p.changeDynamics kwarg names. ``mass`` is handled
    # separately (skipped for mass-pinned bodies).
    _CHANGE_DYNAMICS_KW = {
        "lateral_friction": "lateralFriction",
        "restitution": "restitution",
        "rolling_friction": "rollingFriction",
        "spinning_friction": "spinningFriction",
    }

    def set_physical_params(self, **params: Optional[float]) -> None:
        """Override PyBullet contact/inertial params on the live domino bodies.

        Accepts any of ``mass``, ``lateral_friction`` (PyBullet's
        ``lateralFriction``, i.e. sliding friction), ``restitution``,
        ``rolling_friction``, ``spinning_friction``, ``block_mass``,
        ``block_lateral_friction`` (pass ``None`` to leave a param at its
        current value). The ``block_*`` variants apply only to the
        ``block``-typed bodies (and beat the global param for those
        bodies). Applies ``p.changeDynamics`` to every domino body in
        *this* component's physics client, so one env instance's physics
        can diverge from another's without disturbing the shared
        ClassVars. The override is stored and re-applied after every
        ``reset_state`` (reset rewrites mass) and after body recreation.

        Only affects dynamics-layer params (``changeDynamics``); domino
        *geometry* (width/height) is baked at body creation and is not
        changeable here.
        """
        provided = {k: v for k, v in params.items() if v is not None}
        unknown = set(provided) - self._PHYSICAL_PARAM_KEYS
        if unknown:
            raise ValueError(
                f"Unknown physical param(s) {sorted(unknown)}; "
                f"expected a subset of {sorted(self._PHYSICAL_PARAM_KEYS)}.")
        self._physical_param_override.update(provided)
        self._apply_physical_param_override()

    def clear_physical_params(self) -> None:
        """Drop the override (bodies keep their last-set values until
        reset)."""
        self._physical_param_override = {}

    @property
    def physical_param_override(self) -> Dict[str, float]:
        """Copy of the standing override (see ``set_physical_params``)."""
        return dict(self._physical_param_override)

    def _apply_physical_param_override(self) -> None:
        """Push the stored override onto the live domino bodies."""
        override = self._physical_param_override
        if not override or self._physics_client_id is None:
            return
        base_kwargs = {
            self._CHANGE_DYNAMICS_KW[k]: v
            for k, v in override.items() if k in self._CHANGE_DYNAMICS_KW
        }
        for domino in self.dominos:
            if domino.id is None:
                continue
            kwargs = dict(base_kwargs)
            # Don't clobber a pinned mass, nor the block bodies' mass
            # (which have their own override key below).
            if "mass" in override and domino.id not in self.fixed_domino_ids \
                    and domino.id not in self.block_body_ids:
                kwargs["mass"] = override["mass"]
            # ``block_*`` params override block bodies only. A
            # block-specific value beats the global one for the same body.
            if "block_mass" in override \
                    and domino.id in self.block_body_ids:
                kwargs["mass"] = override["block_mass"]
            if "block_lateral_friction" in override \
                    and domino.id in self.block_body_ids:
                kwargs["lateralFriction"] = override["block_lateral_friction"]
            if kwargs:
                p.changeDynamics(domino.id,
                                 -1,
                                 physicsClientId=self._physics_client_id,
                                 **kwargs)

    def reset_state(self, state: State) -> None:
        """Reset dominoes, targets, and pivots to match state."""
        assert self._physics_client_id is not None
        # Blocks share the domino body pool but carry their own type, so
        # every pool sweep must cover both.
        domino_objs = (state.get_objects(self._domino_type) +
                       state.get_objects(self._block_type))

        # Remove old constraints
        for constraint in self.block_constraints:
            p.removeConstraint(constraint,
                               physicsClientId=self._physics_client_id)
        self.block_constraints = []

        # Restore normal mass on the bodies pinned at the previous reset
        for domino_id in self.fixed_domino_ids + self.block_body_ids:
            p.changeDynamics(domino_id,
                             -1,
                             mass=self.domino_mass,
                             physicsClientId=self._physics_client_id)
        self.fixed_domino_ids = []
        self.block_body_ids = []

        # Update domino colors to match state
        for domino in domino_objs:
            if domino.id is not None:
                r = state.get(domino, "r")
                g = state.get(domino, "g")
                b = state.get(domino, "b")
                update_object(domino.id,
                              color=(r, g, b, 1.0),
                              physics_client_id=self._physics_client_id)

        # Move dominoes absent from the state out of view (by identity,
        # not prefix count: the used set need not be a prefix of the pool).
        used_dominos = set(domino_objs)
        oov_x, oov_y = self.out_of_view_xy
        for domino in self.dominos:
            if domino in used_dominos:
                continue
            oov_x += 0.1
            oov_y += 0.1
            update_object(domino.id,
                          position=(oov_x, oov_y, self.domino_height / 2),
                          physics_client_id=self._physics_client_id)

        # Reset targets
        target_objs = state.get_objects(self._target_type)
        for target_obj in target_objs:
            self._set_flat_rotation(target_obj, 0.0)
        for i in range(len(target_objs), len(self.targets)):
            oov_x += 0.1
            oov_y += 0.1
            update_object(self.targets[i].id,
                          position=(oov_x, oov_y, self.domino_height / 2),
                          physics_client_id=self._physics_client_id)

        # Reset pivots
        pivot_objs = state.get_objects(self._pivot_type)
        for pivot_obj in pivot_objs:
            self._set_flat_rotation(pivot_obj, 0.0)
        for i in range(len(pivot_objs), len(self.pivots)):
            oov_x += 0.1
            oov_y += 0.1
            update_object(self.pivots[i].id,
                          position=(oov_x, oov_y, self.domino_height / 2),
                          physics_client_id=self._physics_client_id)

        self._assign_body_masses(state, domino_objs)

        # Zero residual velocities on every domino body: pose resets go
        # through resetBasePositionAndOrientation, which does NOT clear
        # velocities - a body that was mid-fall when the previous rollout
        # ended would carry its momentum into this "static" scene.
        for domino in self.dominos:
            if domino.id is not None:
                p.resetBaseVelocity(domino.id, [0, 0, 0], [0, 0, 0],
                                    physicsClientId=self._physics_client_id)

        # Re-assert any standing physical-param override: reset may just
        # have rewritten mass above, so the override (if any) must be
        # re-applied to keep this instance's physics diverged.
        self._apply_physical_param_override()

    def _assign_body_masses(self, state: State,
                            domino_objs: Sequence[Object]) -> None:
        """Hook for per-body masses, run by ``reset_state`` before the override
        is re-applied.

        A subclass may pin the mass of individual bodies here, recording
        them in ``fixed_domino_ids`` or ``block_body_ids`` (the next
        reset restores them to ``domino_mass``). This base sim assigns
        none: every body keeps ``domino_mass`` unless the physical-param
        override says otherwise.
        """
        del state, domino_objs  # unused in the base sim

    def extract_feature(self, obj: Object, feature: str) -> Optional[float]:
        """Extract feature for domino-related objects."""
        # Let the base environment handle position/orientation extraction
        return None

    def get_object_ids_for_held_check(self) -> List[int]:
        """Return domino and pivot IDs for held checking."""
        domino_ids = [d.id for d in self.dominos if d.id is not None]
        pivot_ids = [p.id for p in self.pivots if p.id is not None]
        return domino_ids + pivot_ids

    # -------------------------------------------------------------------------
    # Helper methods
    # -------------------------------------------------------------------------

    @staticmethod
    def _get_joint_id(obj_id: int,
                      joint_name: str,
                      physics_client_id: int = 0) -> int:
        """Get joint ID by name from PyBullet object."""
        num_joints = p.getNumJoints(obj_id, physicsClientId=physics_client_id)
        for j in range(num_joints):
            info = p.getJointInfo(obj_id, j, physicsClientId=physics_client_id)
            if info[1].decode("utf-8") == joint_name:
                return j
        return -1

    def _set_flat_rotation(self, flap_obj: Object, rot: float = 0.0) -> None:
        """Set rotation of a hinged object (target/pivot)."""
        p.resetJointState(flap_obj.id,
                          flap_obj.joint_id,
                          rot,
                          physicsClientId=self._physics_client_id)

    # -------------------------------------------------------------------------
    # Public properties for type access
    # -------------------------------------------------------------------------

    @property
    def domino_type(self) -> Type:
        """Domino type."""
        return self._domino_type

    @property
    def block_type(self) -> Type:
        """Block type."""
        return self._block_type

    @property
    def target_type(self) -> Type:
        """Target type."""
        return self._target_type

    @property
    def pivot_type(self) -> Type:
        """Pivot type."""
        return self._pivot_type


def create_domino_block(
    color: Tuple[float, float, float, float],
    half_extents: Tuple[float, float, float],
    mass: float,
    friction: float,
    position: Pose3D = (0.0, 0.0, 0.0),
    orientation: Quaternion = (0.0, 0.0, 0.0, 1.0),
    physics_client_id: int = 0,
    add_top_triangle: bool = False,
    *,
    restitution: float = 0.02,
    rolling_friction: float = 0.006,
    spinning_friction: Optional[float] = None,
    linear_damping: float = 0.0,
    angular_damping: float = 0.03,
    friction_anchor: bool = True,
    ccd: bool = True,
    ccd_swept_radius: Optional[float] = None,
) -> int:
    """Create a domino-tuned block with appropriate physics settings."""
    block_id = create_pybullet_block(
        color=color,
        half_extents=half_extents,
        mass=mass,
        friction=friction,
        position=position,
        orientation=orientation,
        physics_client_id=physics_client_id,
        add_top_triangle=add_top_triangle,
    )

    if spinning_friction is None:
        spinning_friction = friction

    p.changeDynamics(
        block_id,
        linkIndex=-1,
        lateralFriction=friction,
        rollingFriction=rolling_friction,
        spinningFriction=spinning_friction,
        restitution=restitution,
        linearDamping=linear_damping,
        angularDamping=angular_damping,
        frictionAnchor=friction_anchor,
        physicsClientId=physics_client_id,
    )

    if ccd:
        m = min(half_extents)
        swept = ccd_swept_radius if ccd_swept_radius is not None else 0.5 * m
        p.changeDynamics(
            block_id,
            linkIndex=-1,
            ccdSweptSphereRadius=swept,
            physicsClientId=physics_client_id,
        )

    return block_id
