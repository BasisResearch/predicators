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
  round-trips the scalar features nothing in the engine backs.

It has no scene, no mechanism and no calibration: every body, joint
reading and force is the subclass's.
"""
from __future__ import annotations

import os
from typing import Any, ClassVar, Dict, Iterable, List, Optional, Set, Tuple
from typing import Type as TypingType

from predicators import utils
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
