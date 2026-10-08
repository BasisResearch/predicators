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

import os
from typing import Any, ClassVar, Dict, Iterable, List, Optional, Set, Tuple
from typing import Type as TypingType

import pybullet as p

from predicators import utils
from predicators.code_sim_learning.fit_space import ParamSpec
from predicators.envs.pybullet_env import PyBulletEnv
from predicators.pybullet_helpers.geometry import Quaternion
from predicators.pybullet_helpers.robots.single_arm import \
    SingleArmPyBulletRobot
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
# The range rehearsal samples an undeclared material over, when the model
# does not fit it: what a body of unknown make plausibly has, narrower than
# the fit boxes above (property -> lo, hi, scale). A mass is relative to the
# scene's own value.
SAMPLED_MATERIALS: Dict[str, Tuple[float, float, str]] = {
    "mass": (1.0 / 3.0, 3.0, "log"),
    "lateral_friction": (0.1, 1.5, "log"),
    "spinning_friction": (0.0, 1.0, "linear"),
    "rolling_friction": (0.0, 0.02, "linear"),
    "restitution": (0.0, 0.5, "linear"),
    "linear_damping": (0.0, 0.2, "linear"),
    "angular_damping": (0.0, 0.2, "linear"),
}
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
    :meth:`sampled_material_specs`.
    """
    # Bound by scene_base_class.
    _scene_env_name: ClassVar[str] = "agent_scene"
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
        # Values set on undeclared materials (a rehearsal draw's).
        self._sampled_materials: Dict[str, float] = {}
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
        super()._set_state(rebound)
        # Binding names the groups; a reset may have rebuilt bodies or
        # restored the scene's own dynamics.
        self._note_material_baselines()
        self._apply_materials()

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
        (a declaration's own box wins over the menu's)."""
        groups, menu = self._materials()
        info: Dict[str, Dict] = {}
        for name, (group, prop) in menu.items():
            lo, hi, scale, meaning = MATERIALS[prop]
            default = self._material_baseline.get(
                (group, prop), self._read_material(groups[group][0], prop))
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
        return info

    def apply_physical_param_overrides(self, params: Dict[str, float]) -> None:
        """Set declared parameters and materials.

        A value for an undeclared material (a rehearsal draw's) is set
        on its group's bodies like a declared one; a fit that pins it to
        its default sets the scene's own value. Material names are known
        from the observed types, so a fresh world that has bound no body
        yet accepts the menu another world of the same scene reports.
        """
        materials = self._material_names()
        declared = {spec.name for spec in type(self).AGENT_PARAM_SPECS}
        self._sampled_materials.update({
            name: float(value)
            for name, value in params.items()
            if name in materials and name not in declared
        })
        super().apply_physical_param_overrides({
            name: value
            for name, value in params.items()
            if name not in materials or name in declared
        })
        self._apply_materials()

    def sampled_material_specs(self) -> List[ParamSpec]:
        """The materials of the scene the model does not declare, as the
        parameters rehearsal samples: each ranges over ``SAMPLED_MATERIALS``
        and starts at the scene's own value, moved into that range."""
        declared = {spec.name for spec in type(self).AGENT_PARAM_SPECS}
        groups, menu = self._materials()
        specs = []
        for name, (group, prop) in sorted(menu.items()):
            if name in declared:
                continue
            lo, hi, scale = SAMPLED_MATERIALS[prop]
            own = self._material_baseline.get(
                (group, prop), self._read_material(groups[group][0], prop))
            if prop == "mass":
                lo, hi = own * lo, own * hi
            specs.append(
                ParamSpec(name,
                          min(max(own, lo), hi),
                          lo=lo,
                          hi=hi,
                          scale=scale))
        return specs

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
        """Record what the scene gives each newly seen group, before any
        declared value is set on it."""
        groups, menu = self._materials()
        for group, prop in menu.values():
            if (group, prop) not in self._material_baseline:
                self._material_baseline[(group, prop)] = self._read_material(
                    groups[group][0], prop)

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
            return _DEFAULT_DAMPING
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


def scene_base_class(env_name: str, types: Iterable[Type],
                     asset_dir: str) -> TypingType[SceneBase]:
    """The ``SceneBase`` bound to one deployment: its robot, its observation
    types and the sandbox's asset directory.

    The robot facts come from the deployment's env class, never from an
    instance, so no scene is built here.
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
