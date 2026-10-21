"""A domain-agnostic PyBullet base for agent-built scene models.

The agentic real-to-sim arm (``agent_continual_real_to_sim``) receives
the generic :class:`PyBulletEnv`, this base, a geometry manifest of the
scene and the asset files the scene's bodies were loaded from. The
agent writes the scene construction, the state sync of every feature a
body pose does not carry, and the mechanisms. ``SceneBase`` carries
exactly what a robot deployment knows without a domain model:

- the ``BaseEnv`` boilerplate a model never needs (no tasks, no
  predicates), so a subclass is concrete once it loads its bodies;
- the robot: its base placement, home pose, finger conventions and
  grasp-detection tolerances, copied from the deployment;
- the observation schema (the types) and the render camera;
- the binding of observed object names to the bodies the subclass's
  ``initialize_pybullet`` returns under those names, and a store that
  round-trips the scalar features nothing in the engine backs;
- the engine's material properties as fittable parameters, per
  observed type and for the static support (``get_physical_param_info``).

It has no scene, no mechanism and no calibration: every body, joint
reading and force is the subclass's, and a material property keeps the
value the scene gives it until the subclass declares it.
"""
from __future__ import annotations

import contextlib
import dataclasses
import logging
import os
import threading
from typing import Any, Callable, ClassVar, Dict, Iterable, Iterator, List, \
    Optional, Sequence, Set, Tuple
from typing import Type as TypingType

import numpy as np
import pybullet as p

from predicators import utils
from predicators.code_sim_learning.fit_space import ParamSpec
from predicators.envs.pybullet_env import PyBulletEnv
from predicators.observation_noise import ObservationNoise
from predicators.pybullet_helpers.geometry import Quaternion
from predicators.pybullet_helpers.robots.single_arm import \
    SingleArmPyBulletRobot
from predicators.settings import CFG
from predicators.structs import EnvironmentTask, Object, Predicate, State, Type

# The robot-side class attributes a scene base copies from the deployment
# env class. Placement and home pose are where the robot stands and rests;
# the finger and grasp entries are the gripper's conventions in this
# engine (what a ``fingers`` reading means, when a pinch counts as a
# grasp). None of them describes the scene.
_ROBOT_ATTRIBUTES = ("robot_base_pos", "robot_base_orn", "robot_init_roll",
                     "robot_init_tilt", "robot_init_wrist", "open_fingers",
                     "closed_fingers", "grasp_tol", "grasp_tol_small",
                     "grasp_partner_tol", "_finger_action_tol")
_CAMERA_ATTRIBUTES = ("_camera_distance", "_camera_yaw", "_camera_pitch",
                      "_camera_target", "_camera_fov")

# The engine's material properties of a body, offered for fitting as
# ``<group>_<property>``: property -> (fit box lo, hi, scale, meaning).
# A mass box is set around the scene's own value.
MATERIALS: Dict[str, Tuple[Optional[float], Optional[float], str, str]] = {
    "mass": (None, None, "log", "total mass in kg; inertia scales with it"),
    "lateral_friction":
    (0.01, 3.0, "log", "friction coefficient against sliding"),
    "spinning_friction":
    (0.0, 1.0, "linear", "friction against twisting about a contact normal"),
    "rolling_friction": (0.0, 0.1, "linear", "friction against rolling"),
    "restitution": (0.0, 1.0, "linear", "bounce of a contact"),
    "linear_damping": (0.0, 1.0, "linear", "damping of linear velocity"),
    "angular_damping": (0.0, 1.0, "linear", "damping of angular velocity"),
}
# What a body of unknown make plausibly has, narrower than the fit boxes
# above (property -> lo, hi, scale); a mass is relative to the model's own
# mass, the declared value or else the scene's. Rehearsal samples an
# undeclared material over this range, and a scene base bound with
# ``plausible_floor`` widens a declared material's box to cover it.
PLAUSIBLE_MATERIALS: Dict[str, Tuple[float, float, str]] = {
    "mass": (1.0 / 3.0, 3.0, "log"),
    "lateral_friction": (0.1, 1.5, "log"),
    "spinning_friction": (0.0, 1.0, "linear"),
    "rolling_friction": (0.0, 0.02, "linear"),
    "restitution": (0.0, 0.5, "linear"),
    "linear_damping": (0.0, 0.2, "linear"),
    "angular_damping": (0.0, 0.2, "linear"),
}


def _plausible_range(prop: str, value: float) -> Tuple[float, float]:
    """The plausible range of one material property (``PLAUSIBLE_MATERIALS``);
    a mass is relative to ``value``, the model's own mass."""
    lo, hi, _scale = PLAUSIBLE_MATERIALS[prop]
    if prop == "mass":
        return value * lo, value * hi
    return lo, hi


def plausible_materials_text() -> str:
    """``PLAUSIBLE_MATERIALS`` in words, for the scene contract."""
    parts = []
    for prop, (lo, hi, _scale) in PLAUSIBLE_MATERIALS.items():
        words = prop.replace("_", " ")
        if prop == "mass":
            parts.append(f"{words} {lo:.2g} to {hi:g} times its value in your "
                         "model")
        else:
            parts.append(f"{words} {lo:g} to {hi:g}")
    return ", ".join(parts)


# Static bodies with no observed name (tables, walls, fixtures) form the
# support group; mass and damping mean nothing for a body that never moves.
SUPPORT_GROUP = "support"
_CONTACT_MATERIALS = ("lateral_friction", "spinning_friction",
                      "rolling_friction", "restitution")
# PyBullet reports no damping back; this is its default for every body.
_DEFAULT_DAMPING = 0.04
_CHANGE_DYNAMICS_KEYS = {
    "lateral_friction": "lateralFriction",
    "spinning_friction": "spinningFriction",
    "rolling_friction": "rollingFriction",
    "restitution": "restitution",
    "linear_damping": "linearDamping",
    "angular_damping": "angularDamping",
}

# A scene's own damping is recorded as its code sets it with
# p.changeDynamics while the scene is built (_recording_damping): the
# records of each build in progress, keyed by (client, body).
_DAMPING_KEYWORDS = {
    "linearDamping": "linear_damping",
    "angularDamping": "angular_damping",
}
_DampingRecord = Dict[Tuple[int, int], Dict[str, float]]
_DAMPING_RECORDS: Dict[int, _DampingRecord] = {}
_DAMPING_LOCK = threading.Lock()
_UNRECORDED_CHANGE_DYNAMICS: List[Callable[..., Any]] = []


def _recording_change_dynamics(*args: Any, **kwargs: Any) -> Any:
    """``p.changeDynamics`` that also notes the damping it sets."""
    body = kwargs.get("bodyUniqueId", args[0] if args else None)
    client = int(kwargs.get("physicsClientId", 0))
    for keyword, prop in _DAMPING_KEYWORDS.items():
        if keyword in kwargs and body is not None:
            for record in list(_DAMPING_RECORDS.values()):
                record.setdefault((client, int(body)),
                                  {})[prop] = float(kwargs[keyword])
    return _UNRECORDED_CHANGE_DYNAMICS[0](*args, **kwargs)


@contextlib.contextmanager
def _recording_damping() -> Iterator[_DampingRecord]:
    """The damping ``p.changeDynamics`` sets (by keyword) while the block runs,
    by (client, body); builds in other threads land in the record too, under
    their own clients."""
    record: _DampingRecord = {}
    token = id(record)
    with _DAMPING_LOCK:
        if not _DAMPING_RECORDS:
            _UNRECORDED_CHANGE_DYNAMICS[:] = [p.changeDynamics]
            setattr(p, "changeDynamics", _recording_change_dynamics)
        _DAMPING_RECORDS[token] = record
    try:
        yield record
    finally:
        with _DAMPING_LOCK:
            del _DAMPING_RECORDS[token]
            if not _DAMPING_RECORDS:
                setattr(p, "changeDynamics", _UNRECORDED_CHANGE_DYNAMICS[0])


# The observed features a body's primitive collision shape takes
# (SceneBase._fit_body_sizes): a box's lengths along its own axes, or their
# halves, and a sphere's radius.
_BOX_LENGTHS = ("x_len", "y_len", "z_len")
_BOX_HALVES = ("half_x", "half_y", "half_z")
_SPHERE_RADIUS = "radius"
# A body takes an observed size that differs from its own by more than
# _SIZE_SIGMAS of the size's declared observation noise and by more than
# _SIZE_ATOL: a noisy reading of the size it has is no new size.
_SIZE_SIGMAS = 4.0
_SIZE_ATOL = 1e-4
# A box whose axes each lie within about 15 degrees of the world's reads its
# own lengths and its world extents alike.
_ALIGNED_COSINE = float(np.cos(np.radians(15.0)))


@dataclasses.dataclass(frozen=True)
class _Primitive:
    """A body built as one box or sphere collision shape on a base without
    links: its geometry, its size (a box's lengths along its own axes, or a
    sphere's radius) and the shape's frame on the base."""
    geometry: int
    size: Tuple[float, ...]
    frame_position: Tuple[float, ...]
    frame_orientation: Tuple[float, ...]


class SceneBase(PyBulletEnv):
    """The concrete base an agent-built scene model subclasses.

    A subclass overrides ``initialize_pybullet`` to load the scene's
    bodies (call ``super()`` first: it connects the engine and loads the
    ground plane and the robot) and returns them in the bodies dict
    under the observed object names. Objects the observation lists
    without a body of their own (a virtual reading) are returned as
    ``None`` under their name, and their features round-trip through
    the feature store. Features a body does not carry in its pose
    (a joint-backed switch reading, a level, a flag) are the
    subclass's: override ``_set_domain_specific_state`` and
    ``_get_domain_specific_feature`` and call ``super()`` for the rest.
    Mechanisms go in ``_domain_specific_step``.

    The engine's material properties are fittable parameters named
    ``<group>_<property>`` (``MATERIALS``): a group is an observed type
    with bodies, or ``support`` for the static bodies without an observed
    name. A name the subclass declares in ``AGENT_PARAM_SPECS`` is set
    on every body of its group after each reset and whenever its value
    changes. An undeclared one keeps the scene's own value unless an
    override sets it: rehearsal samples it as one of
    :meth:`sampled_material_specs`. Bound with ``plausible_floor``, the
    registry names each declared material's plausible range, and a
    declared box narrower than that is widened to it.
    """
    # Bound by scene_base_class.
    _scene_env_name: ClassVar[str] = "agent_scene"
    _plausible_floor: ClassVar[bool] = False
    _scene_types: ClassVar[Tuple[Type, ...]] = ()
    _robot_home_orn: ClassVar[Optional[Quaternion]] = None
    _asset_dir: ClassVar[str] = ""
    robot_init_x: ClassVar[float] = 0.0
    robot_init_y: ClassVar[float] = 0.0
    robot_init_z: ClassVar[float] = 0.0

    def __init__(self,
                 use_gui: bool = False,
                 skip_residual_dynamics: bool = False) -> None:
        self._scene_bodies: Dict[str, Any] = {}
        self._feature_store: Dict[Tuple[str, str], float] = {}
        # (group, property) -> the value the scene itself gave the group's
        # first body, before any declared value was set.
        self._material_baseline: Dict[Tuple[str, str], float] = {}
        self._material_layout: Optional[Tuple[Tuple[int, int], Dict[str,
                                                                    List[int]],
                                              Dict[str, Tuple[str,
                                                              str]]]] = None
        # (body, property) -> the value the scene itself gave the body,
        # restored when an undeclared material is set back to its default.
        self._body_baseline: Dict[Tuple[int, str], float] = {}
        # Values set on undeclared materials (a rehearsal draw's).
        self._sampled_materials: Dict[str, float] = {}
        # The damping the scene's own code sets while it is built, which
        # PyBullet cannot report back.
        with _recording_damping() as damping:
            self._scene_damping = damping
            super().__init__(use_gui=use_gui,
                             skip_residual_dynamics=skip_residual_dynamics)
        self._robot = Object("robot", self._type_named("robot"))
        self._robot.id = self._pybullet_robot.robot_id
        self._body_objects["robot"] = self._robot

    # -- What the deployment supplies ------------------------------------

    @classmethod
    def get_name(cls) -> str:
        return cls._scene_env_name

    @classmethod
    def get_robot_ee_home_orn(cls) -> Quaternion:
        assert cls._robot_home_orn is not None, "unbound scene base"
        return cls._robot_home_orn

    @classmethod
    def asset(cls, relative_path: str) -> str:
        """The absolute path of a file under ``reference/assets/``, for
        ``p.loadURDF`` and mesh shapes: ``cls.asset("urdf/cup.urdf")``."""
        path = os.path.join(cls._asset_dir, relative_path)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"No asset {relative_path!r} under reference/assets/; the "
                "scene manifest lists the available files.")
        return path

    @property
    def types(self) -> Set[Type]:
        return set(self._scene_types)

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

    @classmethod
    def _type_named(cls, name: str) -> Type:
        for scene_type in cls._scene_types:
            if scene_type.name == name:
                return scene_type
        raise KeyError(f"The observation has no type named {name!r}.")

    # -- Bodies and observed objects -------------------------------------

    def _store_pybullet_bodies(self, pybullet_bodies: Dict[str, Any]) -> None:
        self._scene_bodies = dict(pybullet_bodies)

    @property
    def bodies(self) -> Dict[str, Any]:
        """What this scene's ``initialize_pybullet`` returned."""
        return self._scene_bodies

    def body(self, name: str) -> int:
        """The body id loaded under an observed object's name."""
        body_id = self._scene_bodies.get(name)
        if not isinstance(body_id, int):
            raise KeyError(f"The scene loads no body named {name!r}.")
        return body_id

    def _bind(self, obj: Object) -> Object:
        """This scene's Object for an observed one: the same name, this scene's
        type of that name, and the body loaded under the name."""
        bound = self._body_objects.get(obj.name)
        if bound is None:
            bound = Object(obj.name, self._type_named(obj.type.name))
            body_id = self._scene_bodies.get(obj.name)
            bound.id = body_id if isinstance(body_id, int) else None
            self._body_objects[obj.name] = bound
        return bound

    def _set_state(self, state: State) -> None:
        rebound = state.copy()
        rebound.data = {
            self._bind(obj): values
            for obj, values in rebound.data.items()
        }
        missing = sorted(
            obj.name for obj in rebound.data
            if obj.id is None and obj.name not in self._scene_bodies
            and obj.type.name != "robot" and {"x", "y", "z"}
            & set(obj.type.feature_names))
        if missing:
            raise ValueError(
                f"The scene loads no body for {missing}. Return each one's "
                "body id from initialize_pybullet under its observed name, "
                "or None under that name for an object with no body.")
        unfit = (self._fit_body_sizes(rebound)
                 if CFG.scene_observed_sizes else [])
        super()._set_state(rebound)
        # A size no body here can take is a feature this reset could not
        # reproduce, like a pose it could not reach.
        self._last_unreconstructible_features = list(
            self._last_unreconstructible_features) + unfit
        # Binding names the groups; a reset may have rebuilt bodies or
        # restored the scene's own dynamics.
        self._note_material_baselines()
        self._apply_materials()

    # -- Observed sizes ---------------------------------------------------

    @staticmethod
    def _observed_size(
        obj: Object, state: State
    ) -> Optional[Tuple[int, Tuple[str, ...], Tuple[float, ...], float]]:
        """The primitive shape an object's observed size features describe:

        its geometry, the features, their values and how many of the
        shape's sizes one feature unit is (2 for a half length); None
        for an object whose type has no size features.
        """
        names = set(obj.type.feature_names)
        for features, geometry, per_unit in ((_BOX_LENGTHS, p.GEOM_BOX, 1.0),
                                             (_BOX_HALVES, p.GEOM_BOX,
                                              2.0), ((_SPHERE_RADIUS, ),
                                                     p.GEOM_SPHERE, 1.0)):
            if set(features) <= names:
                return geometry, features, tuple(
                    float(state.get(obj, f)) for f in features), per_unit
        return None

    def _primitive(self, body: int) -> Optional[_Primitive]:
        """``body``'s shape when it is one box or sphere on a base without
        links, else None."""
        client = self._physics_client_id
        if p.getNumJoints(body, physicsClientId=client):
            return None
        shapes = p.getCollisionShapeData(body, -1, physicsClientId=client)
        if len(shapes) != 1:
            return None
        geometry, dims = int(shapes[0][2]), shapes[0][3]
        if geometry == p.GEOM_BOX:
            size = tuple(float(d) for d in dims[:3])
        elif geometry == p.GEOM_SPHERE:
            size = (float(dims[0]), )
        else:
            return None
        return _Primitive(geometry, size,
                          tuple(float(v) for v in shapes[0][5]),
                          tuple(float(v) for v in shapes[0][6]))

    def _fit_body_sizes(self, state: State) -> List[Tuple[Object, str]]:
        """Give each body built as one box or sphere the size its object's
        observation gives (``x_len``, ``y_len``, ``z_len``; ``half_x``,
        ``half_y``, ``half_z``; or ``radius``), rebuilding the body when the
        size differs from its own.

        A level can change a body's size while the model is built for
        another level's, and the fit replays every level's recordings.
        Returns the (object, feature) pairs whose observed size a body
        of this kind could not take: one held by a constraint, or one
        whose visual shape is no box or sphere like its collision shape.
        """
        noise = ObservationNoise.from_cfg()
        unfit: List[Tuple[Object, str]] = []
        for obj in state:
            if obj.id is None or obj.type.name == "robot":
                continue
            observed = self._observed_size(obj, state)
            if observed is None:
                continue
            geometry, features, values, per_unit = observed
            built = self._primitive(obj.id)
            if built is None or built.geometry != geometry:
                continue

            def off_from(sizes: Sequence[float],
                         features: Tuple[str, ...] = features,
                         values: Tuple[float, ...] = values,
                         per_unit: float = per_unit,
                         obj: Object = obj) -> List[str]:
                return [
                    feature
                    for feature, value, have in zip(features, values, sizes)
                    if abs(value - have / per_unit) > max(
                        _SIZE_ATOL,
                        _SIZE_SIGMAS * noise.feature_sigma(obj.type, feature))
                ]

            off = off_from(built.size)
            if off and geometry == p.GEOM_BOX:
                # A box turned off its axes reads the same whether its size
                # features give its own lengths or its extents along the
                # world's axes (an env may publish either), so it is resized
                # only when neither reading matches it.
                turn = np.abs(
                    np.array(
                        p.getMatrixFromQuaternion(
                            p.multiplyTransforms(
                                (0.0, 0.0, 0.0),
                                self._orientation_in_state(obj, state),
                                (0.0, 0.0, 0.0),
                                built.frame_orientation)[1])).reshape(3, 3))
                if np.min(np.diag(turn)) < _ALIGNED_COSINE:
                    if off_from(tuple(turn @ np.array(built.size))):
                        unfit.extend((obj, feature) for feature in off)
                    continue
            if off and not self._rebuild_body(
                    obj, built, tuple(v * per_unit for v in values)):
                unfit.extend((obj, feature) for feature in off)
        return unfit

    def _rebuild_body(self, obj: Object, built: _Primitive,
                      size: Tuple[float, ...]) -> bool:
        """Replace ``obj``'s body by one of ``size`` with the same shape,
        frame, pose, velocity, mass and materials, and the same look scaled
        alike; False, with nothing changed, for a body a constraint holds or
        whose visual shape is no single shape of its collision geometry."""
        client = self._physics_client_id
        old = int(obj.id)
        for index in range(p.getNumConstraints(physicsClientId=client)):
            info = p.getConstraintInfo(p.getConstraintUniqueId(
                index, physicsClientId=client),
                                       physicsClientId=client)
            if old in (info[0], info[2]):
                return False
        visuals = [
            v for v in p.getVisualShapeData(old, physicsClientId=client)
            if v[1] == -1
        ]
        if len(visuals) > 1 or (visuals and visuals[0][2] != built.geometry):
            return False
        ratio = [
            want / have if have > 0.0 else 1.0
            for want, have in zip(size, built.size)
        ]
        dynamics = p.getDynamicsInfo(old, -1, physicsClientId=client)
        # PyBullet reports a shape's frame relative to the body's inertial
        # frame and takes a new shape's relative to its link frame: compose
        # the inertial offset with it.
        place, turn = p.multiplyTransforms(dynamics[3], dynamics[4],
                                           built.frame_position,
                                           built.frame_orientation)
        frame = {
            "collisionFramePosition": list(place),
            "collisionFrameOrientation": list(turn),
        }
        if built.geometry == p.GEOM_BOX:
            collision = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=[s / 2.0 for s in size],
                physicsClientId=client,
                **frame)
        else:
            collision = p.createCollisionShape(p.GEOM_SPHERE,
                                               radius=size[0],
                                               physicsClientId=client,
                                               **frame)
        visual = -1
        if visuals:
            dims, place, turn, rgba = (visuals[0][3], visuals[0][5],
                                       visuals[0][6], visuals[0][7])
            look = {
                "rgbaColor": list(rgba),
                "visualFramePosition": list(place),
                "visualFrameOrientation": list(turn),
                "physicsClientId": client,
            }
            if built.geometry == p.GEOM_BOX:
                visual = p.createVisualShape(
                    p.GEOM_BOX,
                    halfExtents=[d * r / 2.0 for d, r in zip(dims[:3], ratio)],
                    **look)
            else:
                visual = p.createVisualShape(p.GEOM_SPHERE,
                                             radius=dims[0] * ratio[0],
                                             **look)
        position, orientation = p.getBasePositionAndOrientation(
            old, physicsClientId=client)
        linear, angular = p.getBaseVelocity(old, physicsClientId=client)
        new = p.createMultiBody(baseMass=self._body_baseline.get((old, "mass"),
                                                                 dynamics[0]),
                                baseCollisionShapeIndex=collision,
                                baseVisualShapeIndex=visual,
                                basePosition=list(position),
                                baseOrientation=list(orientation),
                                baseInertialFramePosition=list(dynamics[3]),
                                baseInertialFrameOrientation=list(dynamics[4]),
                                physicsClientId=client)
        p.resetBasePositionAndOrientation(new,
                                          position,
                                          orientation,
                                          physicsClientId=client)
        p.resetBaseVelocity(new, linear, angular, physicsClientId=client)
        # The scene's own materials and damping: the declared and sampled
        # values go back on when the reset applies the materials.
        own = {
            prop: self._body_baseline.get((old, prop), current)
            for prop, current in (("lateral_friction",
                                   dynamics[1]), ("restitution", dynamics[5]),
                                  ("rolling_friction", dynamics[6]),
                                  ("spinning_friction", dynamics[7]))
        }
        damping = self._scene_damping.pop((client, old), None)
        if damping is not None:
            self._scene_damping[(client, new)] = damping
        p.changeDynamics(new,
                         -1,
                         lateralFriction=own["lateral_friction"],
                         restitution=own["restitution"],
                         rollingFriction=own["rolling_friction"],
                         spinningFriction=own["spinning_friction"],
                         linearDamping=(damping
                                        or {}).get("linear_damping",
                                                   _DEFAULT_DAMPING),
                         angularDamping=(damping
                                         or {}).get("angular_damping",
                                                    _DEFAULT_DAMPING),
                         physicsClientId=client)
        if dynamics[8] > 0.0 and dynamics[9] > 0.0:
            p.changeDynamics(new,
                             -1,
                             contactDamping=dynamics[8],
                             contactStiffness=dynamics[9],
                             physicsClientId=client)
        for prop in MATERIALS:
            value = self._body_baseline.pop((old, prop), None)
            if value is not None:
                self._body_baseline[(new, prop)] = value
        p.removeBody(old, physicsClientId=client)
        obj.id = new
        for name, body in list(self._scene_bodies.items()):
            if body == old:
                self._scene_bodies[name] = new
        if self._contact_relevant_ids is not None and \
                old in self._contact_relevant_ids:
            self._contact_relevant_ids.discard(old)
            self._contact_relevant_ids.add(new)
        self._material_layout = None
        logging.debug("Scene: rebuilt %s's body at the observed size %s.",
                      obj.name, size)
        return True

    def _set_domain_specific_state(self, state: State) -> None:
        """Round-trip every feature no body pose carries through the feature
        store; a subclass writes its joint-backed features first and calls
        ``super()`` for the rest."""
        for obj in state:
            if obj.type.name == "robot":
                continue
            for feature in obj.type.feature_names:
                if obj.id is None or feature not in self._PYBULLET_FEATURES:
                    self._feature_store[(obj.name, feature)] = float(
                        state.get(obj, feature))

    def _get_domain_specific_feature(self, obj: Object, feature: str) -> float:
        return self._feature_store.get((obj.name, feature), 0.0)

    def _get_object_state_dict(self, obj: Object) -> Dict[str, float]:
        if obj.id is None:
            return {
                feature: self._get_domain_specific_feature(obj, feature)
                for feature in obj.type.feature_names
            }
        return super()._get_object_state_dict(obj)

    def _get_object_ids_for_held_check(self) -> List[int]:
        """Every observed body but the robot's; narrow it in a subclass."""
        robot_id = self._pybullet_robot.robot_id
        return [
            obj.id for obj in self._objects
            if obj.id is not None and obj.id != robot_id
        ]

    # -- Engine materials --------------------------------------------------

    def get_physical_param_info(self) -> Dict[str, Dict]:
        """The material menu of the current scene, then the declared parameters
        (a declaration's own box wins over the menu's).

        Bound with ``plausible_floor``, each declared material also
        names its ``plausible`` (lo, hi) range, which
        ``stamp_physical_spec_scales`` widens a narrower declared box
        to. It comes from the name and the declaration alone, so a world
        that has bound no body yet reports the same range.
        """
        groups, menu = self._materials()
        info: Dict[str, Dict] = {}
        for name, (group, prop) in menu.items():
            lo, hi, scale, meaning = MATERIALS[prop]
            default = self._scene_value(groups, group, prop)
            if prop == "mass":
                lo, hi = default / 20.0, default * 20.0
            entry: Dict[str, Any] = {
                "default": default,
                "lo": lo,
                "hi": hi,
                "description": f"{meaning}, of every {group} body",
            }
            if scale == "log":
                entry["scale"] = "log"
            info[name] = entry
        info.update(super().get_physical_param_info())
        if type(self)._plausible_floor:
            for spec in type(self).AGENT_PARAM_SPECS:
                material = type(self)._material_property(spec.name)
                if material is not None:
                    info[spec.name]["plausible"] = _plausible_range(
                        material, float(spec.init_value))
        return info

    def _scene_value(self, groups: Dict[str, List[int]], group: str,
                     prop: str) -> float:
        """The value the scene itself gave a group's first body."""
        return self._material_baseline.get(
            (group, prop), self._read_material(groups[group][0], prop))

    @classmethod
    def _material_property(cls, name: str) -> Optional[str]:
        """The material property a parameter name sets, or None for a name that
        is no material of this scene's types."""
        if name not in cls._material_names():
            return None
        return next(prop for prop in MATERIALS if name.endswith(f"_{prop}"))

    def apply_physical_param_overrides(self, params: Dict[str, float]) -> None:
        """Set declared parameters and materials.

        A value for an undeclared material (a rehearsal draw's) is set
        on its group's bodies like a declared one. Its default, which a
        fit pins every parameter it does not estimate to, stands for the
        scene's own values: each body gets its own back, so a group of
        bodies the scene made different (the ground and a table) stays
        so. Material names are known from the observed types, so a fresh
        world that has bound no body yet accepts the menu another world
        of the same scene reports.
        """
        materials = self._material_names()
        declared = {spec.name for spec in type(self).AGENT_PARAM_SPECS}
        groups, menu = self._materials()
        for name, value in params.items():
            if name not in materials or name in declared:
                continue
            if name in menu:
                group, prop = menu[name]
                default = self._scene_value(groups, group, prop)
                if float(value) == float(default):
                    if self._sampled_materials.pop(name, None) is not None:
                        self._restore_own_materials(groups[group], prop)
                    continue
            self._sampled_materials[name] = float(value)
        super().apply_physical_param_overrides({
            name: value
            for name, value in params.items()
            if name not in materials or name in declared
        })
        self._apply_materials()

    def _restore_own_materials(self, bodies: List[int], prop: str) -> None:
        """Give each body the value the scene itself gave it."""
        for body in bodies:
            own = self._body_baseline.get((body, prop))
            if own is not None:
                self._write_material(body, prop, own)

    def sampled_material_specs(
        self, objects: Iterable[Object] = ()) -> List[ParamSpec]:
        """The materials of the scene the model does not declare, as the
        parameters rehearsal samples: each ranges over its plausible range
        (``PLAUSIBLE_MATERIALS``) and starts at the scene's own value, moved
        into that range.

        ``objects`` are the observed objects of the scene in play. Each
        is bound to its body first, so the menu names the materials of
        every observed type with a body whether or not this world has
        set a state yet; a world that has bound none offers only the
        support's.
        """
        return [
            ParamSpec(name,
                      min(max(own, lo), hi),
                      lo=lo,
                      hi=hi,
                      scale=PLAUSIBLE_MATERIALS[prop][2])
            for name, prop, own, lo, hi in self._undeclared_materials(objects)
        ]

    def own_material_specs(
        self, objects: Iterable[Object] = ()) -> List[ParamSpec]:
        """:meth:`sampled_material_specs`, each starting at the scene's own
        value and with its range widened to hold that value.

        A fit that holds these at their starting values replays the
        scene as it is, so the lines of its belief through them measure
        how the recordings bear on each. A log-scale material the scene
        sets to zero starts at its plausible range's floor instead.
        """
        specs = []
        for name, prop, own, lo, hi in self._undeclared_materials(objects):
            scale = PLAUSIBLE_MATERIALS[prop][2]
            if scale == "log" and own <= 0.0:
                own = lo
            specs.append(
                ParamSpec(name,
                          own,
                          lo=min(lo, own),
                          hi=max(hi, own),
                          scale=scale))
        return specs

    def _undeclared_materials(
        self, objects: Iterable[Object]
    ) -> List[Tuple[str, str, float, float, float]]:
        """(name, property, the scene's own value, plausible low, plausible
        high) of each material the scene's model does not declare, after
        binding ``objects`` (see :meth:`sampled_material_specs`)."""
        for obj in objects:
            self._bind(obj)
        # The scene's own values of a group the binding named, recorded
        # before any declared or sampled value is set on it.
        self._note_material_baselines()
        declared = {spec.name for spec in type(self).AGENT_PARAM_SPECS}
        groups, menu = self._materials()
        materials = []
        for name, (group, prop) in sorted(menu.items()):
            if name in declared:
                continue
            own = self._scene_value(groups, group, prop)
            lo, hi = _plausible_range(prop, own)
            materials.append((name, prop, own, lo, hi))
        return materials

    @classmethod
    def _material_names(cls) -> Set[str]:
        """Every material name the menu can offer for this scene's types."""
        names = {f"{SUPPORT_GROUP}_{prop}" for prop in _CONTACT_MATERIALS}
        for scene_type in cls._scene_types:
            if scene_type.name != "robot":
                names.update(f"{scene_type.name}_{prop}" for prop in MATERIALS)
        return names

    def _materials(
            self) -> Tuple[Dict[str, List[int]], Dict[str, Tuple[str, str]]]:
        """The material groups (body ids per group) and the menu (parameter
        name -> (group, property)) of the current scene.

        A group is an observed type with bodies, or the support: every
        static body without an observed name. A group whose bodies are
        all static offers only its contact properties. Kept until a body
        or an observed object is added.
        """
        client = self._physics_client_id
        key = (len(self._body_objects), p.getNumBodies(physicsClientId=client))
        if self._material_layout is not None and \
                self._material_layout[0] == key:
            return self._material_layout[1], self._material_layout[2]
        robot_id = self._pybullet_robot.robot_id
        groups: Dict[str, List[int]] = {}
        for obj in self._body_objects.values():
            if obj.id is not None and obj.id != robot_id:
                groups.setdefault(obj.type.name, []).append(obj.id)
        named = {body for bodies in groups.values() for body in bodies}
        if SUPPORT_GROUP not in groups:
            for index in range(p.getNumBodies(physicsClientId=client)):
                body = p.getBodyUniqueId(index, physicsClientId=client)
                if body != robot_id and body not in named and \
                        self._read_material(body, "mass") == 0.0:
                    groups.setdefault(SUPPORT_GROUP, []).append(body)
        menu: Dict[str, Tuple[str, str]] = {}
        for group, bodies in sorted(groups.items()):
            moving = any(
                self._read_material(body, "mass") > 0.0 for body in bodies)
            for prop in (MATERIALS if moving else _CONTACT_MATERIALS):
                menu[f"{group}_{prop}"] = (group, prop)
        self._material_layout = (key, groups, menu)
        return groups, menu

    def _note_material_baselines(self) -> None:
        """Record what the scene gives each newly seen group and body, before
        any declared or sampled value is set on it."""
        groups, menu = self._materials()
        for group, prop in menu.values():
            if (group, prop) not in self._material_baseline:
                self._material_baseline[(group, prop)] = self._read_material(
                    groups[group][0], prop)
            for body in groups[group]:
                if (body, prop) not in self._body_baseline:
                    self._body_baseline[(body, prop)] = self._read_material(
                        body, prop)

    def _apply_materials(self) -> None:
        """Set every declared material parameter, and every value set on an
        undeclared material, on its group's bodies."""
        values = dict(self._sampled_materials)
        values.update({
            spec.name: self._agent_param_values[spec.name]
            for spec in type(self).AGENT_PARAM_SPECS
        })
        if not values:
            return
        groups, menu = self._materials()
        for name, value in values.items():
            if name in menu:
                group, prop = menu[name]
                for body in groups[group]:
                    self._write_material(body, prop, value)

    def _body_links(self, body: int) -> range:
        return range(
            -1, p.getNumJoints(body, physicsClientId=self._physics_client_id))

    def _read_material(self, body: int, prop: str) -> float:
        """A body's material property; its total over links for mass."""
        client = self._physics_client_id
        if prop == "mass":
            return float(
                sum(
                    p.getDynamicsInfo(body, link, physicsClientId=client)[0]
                    for link in self._body_links(body)))
        if prop in ("linear_damping", "angular_damping"):
            # What the scene set while it was built, else PyBullet's default.
            return self._scene_damping.get((client, body),
                                           {}).get(prop, _DEFAULT_DAMPING)
        info = p.getDynamicsInfo(body, -1, physicsClientId=client)
        return float({
            "lateral_friction": info[1],
            "restitution": info[5],
            "rolling_friction": info[6],
            "spinning_friction": info[7],
        }[prop])

    def _write_material(self, body: int, prop: str, value: float) -> None:
        """Set a material property on every link of a body.

        A mass scales each moving link's mass and inertia; a static body
        stays static.
        """
        client = self._physics_client_id
        if prop == "mass":
            total = self._read_material(body, "mass")
            if total <= 0.0 or value == total:
                return
            factor = float(value) / total
            for link in self._body_links(body):
                info = p.getDynamicsInfo(body, link, physicsClientId=client)
                if info[0] > 0.0:
                    p.changeDynamics(
                        body,
                        link,
                        mass=info[0] * factor,
                        localInertiaDiagonal=[v * factor for v in info[2]],
                        physicsClientId=client)
            return
        for link in self._body_links(body):
            p.changeDynamics(body,
                             link,
                             physicsClientId=client,
                             **{_CHANGE_DYNAMICS_KEYS[prop]: float(value)})


def scene_base_class(env_name: str,
                     types: Iterable[Type],
                     asset_dir: str,
                     plausible_floor: bool = False) -> TypingType[SceneBase]:
    """The ``SceneBase`` bound to one deployment: its robot, its observation
    types and the sandbox's asset directory.

    The robot facts come from the deployment's env class, never from an
    instance, so no scene is built here. ``plausible_floor`` keeps every
    declared material's box at least as wide as its plausible range.
    """
    # pylint: disable=protected-access
    candidates = [
        cls for cls in utils.get_all_subclasses(PyBulletEnv)
        if not cls.__abstractmethods__ and cls.__module__.startswith(
            "predicators.envs.") and cls.get_name() == env_name
    ]
    assert len(candidates) == 1, env_name
    env_cls = candidates[0]
    scene_types = tuple(sorted(types, key=lambda t: t.name))
    if not any(t.name == "robot" for t in scene_types):
        raise ValueError("The observation types name no robot type.")
    attributes: Dict[str, Any] = {
        "__module__": __name__,
        "__doc__": SceneBase.__doc__,
        "_scene_env_name": f"{env_name}_agent_scene",
        "_skill_env_name": env_name,
        "_scene_types": scene_types,
        "_robot_home_orn": tuple(env_cls.get_robot_ee_home_orn()),
        "_asset_dir": os.path.abspath(asset_dir),
        "_plausible_floor": plausible_floor,
    }
    declared = env_cls._declared_robot_init_pos()
    for attr, value in zip(("robot_init_x", "robot_init_y", "robot_init_z"),
                           declared):
        attributes[attr] = value
    for attr in _ROBOT_ATTRIBUTES + _CAMERA_ATTRIBUTES:
        attributes[attr] = getattr(env_cls, attr)
    for attr in ("_fingers_state_to_joint", "_fingers_joint_to_state"):
        method = getattr(env_cls, attr)

        def forward(_cls: type,
                    robot: SingleArmPyBulletRobot,
                    value: float,
                    _method: Any = method) -> float:
            return float(_method(robot, value))

        attributes[attr] = classmethod(forward)
    return type("SceneBase", (SceneBase, ), attributes)
