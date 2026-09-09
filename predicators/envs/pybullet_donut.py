"""A PyBullet push domain: move a disc the length of a table into a target.

The discs are 12 cm across, wider than the Fetch gripper can span, and solid
rather than annular, so the only way to move one is to push it from the side.
The grasp check is disabled for them outright (see
``_get_object_ids_for_held_check``), so no contact geometry can talk PyBullet
into a pinch: this is a push domain, and picking a disc up is not a shortcut
that exists.

The goal disc starts at the near edge of the table and the target sits at the
far edge, about 0.75 m away, which is most of the reach of a fixed-base arm
at pushing height. Three more discs are already on the table, in the lane
between the two, so the push is contested from the first step. Nothing spawns
later: what is on the table at reset is all there will ever be.
"""

from typing import Any, ClassVar, Dict, List, Sequence, Set, Tuple

import numpy as np
import pybullet as p

from predicators import utils
from predicators.envs.pybullet_env import PyBulletEnv
from predicators.pybullet_helpers.geometry import Pose3D, Quaternion
from predicators.pybullet_helpers.objects import update_object
from predicators.pybullet_helpers.robots import SingleArmPyBulletRobot
from predicators.settings import CFG
from predicators.structs import Action, EnvironmentTask, GroundAtom, Object, \
    Observation, Predicate, State, Type


class PyBulletDonutEnv(PyBulletEnv):
    """PyBullet Donut domain."""

    # Parameters
    table_height: ClassVar[float] = 0.2
    # One slab: the 0.9 m URDF box centred at y = 0.75 spans y 0.30 to 1.20.
    # It used to be two slabs, which put the target in the middle of the
    # table because the far slab was out of the arm's reach anyway.
    _table_pose: ClassVar[Pose3D] = (1.35, 0.75, 0.0)
    _table_orientation: ClassVar[Quaternion] = (0., 0., 0., 1.)

    # The arm reaches a band 0.70 to 0.85 m from its base at pushing height,
    # which along the x = 1.20 lane is y in [0.20, 1.30] (measured 2026-09-08).
    # The slab sits inside that band, so the whole table is reachable and the
    # target can sit at its far edge.
    push_lane_x: ClassVar[float] = 1.20
    x_lb: ClassVar[float] = 1.15
    x_ub: ClassVar[float] = 1.28
    # The goal disc starts here; the pushing pose sits ~12 cm behind it, so a
    # smaller value would put the approach outside the arm's reach.
    y_lb: ClassVar[float] = 0.35
    # Target centre: the far edge of the slab (1.20), inset by half the target
    # square plus a 15 mm margin so the square is fully on the table.
    y_ub: ClassVar[float] = 1.115
    # The three distractor discs start in the lane, in the way of the push.
    lane_y_lb: ClassVar[float] = 0.50
    lane_y_ub: ClassVar[float] = 0.98

    # Robot init
    robot_init_x: ClassVar[float] = 1.35
    robot_init_y: ClassVar[float] = 0.75
    robot_init_z: ClassVar[float] = 0.5
    robot_base_pos: ClassVar[Tuple[float, float, float]] = (0.75, 0.75, 0.0)
    robot_base_orn: ClassVar[Tuple[float, float, float, float]] = (0., 0., 0., 1.)

    # Disc parameters. 12 cm across is wider than the gripper's ~7 cm span, so
    # the arm cannot pick one up, and solid, so there is no rim to hook.
    # All four are on the table from reset: the goal disc plus three
    # distractors. None arrive later.
    num_donuts: ClassVar[int] = 4
    donut_radius: ClassVar[float] = 0.06
    donut_half_height: ClassVar[float] = 0.0125
    donut_mass: ClassVar[float] = 0.4
    # Enough friction that a shove moves the disc instead of squirting it
    # sideways out of the arm's reachable lane (gate, DEBUG_LOG 22).
    donut_friction: ClassVar[float] = 0.7
    # Aliases kept so callers reasoning about extent keep working.
    donut_major_radius: ClassVar[float] = 0.06
    donut_minor_radius: ClassVar[float] = 0.0125

    # Target parameters
    target_width: ClassVar[float] = 0.14
    target_height: ClassVar[float] = 0.14
    target_color: ClassVar[Tuple[float, float, float, float]] = (0.85, 0.1, 0.1, 1.0)
    # The goal disc is blue and the distractors are tan, so an agent with only
    # the camera image can still tell which disc the task means.
    goal_disc_color: ClassVar[Tuple[float, float, float,
                                    float]] = (0.15, 0.35, 0.85, 1.0)
    other_disc_color: ClassVar[Tuple[float, float, float,
                                     float]] = (0.80, 0.62, 0.36, 1.0)

    # Camera
    # Framed on the whole push lane, start to target.
    _camera_target: ClassVar[Pose3D] = (1.24, 0.73, 0.2)
    _camera_distance: ClassVar[float] = 1.15
    _camera_yaw: ClassVar[float] = 90
    _camera_pitch: ClassVar[float] = -38

    # Types
    _robot_type = Type("robot", ["pose_x", "pose_y", "pose_z", "fingers"])
    _donut_type = Type("donut", ["x", "y", "z", "is_held", "r", "g", "b"])
    _target_type = Type("target", ["x", "y", "z", "r", "g", "b"])

    def __init__(self, use_gui: bool = False, **kwargs: Any) -> None:
        self._robot = Object("robot", self._robot_type)
        self._donuts: List[Object] = []
        for i in range(self.num_donuts):
            self._donuts.append(Object(f"donut_{i}", self._donut_type))
        self._target = Object("target", self._target_type)

        # Predicates
        self._InTarget = Predicate("InTarget", [self._donut_type, self._target_type],
                                   self._InTarget_holds)

        self._step_count = 0

        super().__init__(use_gui, **kwargs)

    @classmethod
    def get_name(cls) -> str:
        return "pybullet_donut"

    @property
    def types(self) -> Set[Type]:
        return {self._robot_type, self._donut_type, self._target_type}

    @property
    def predicates(self) -> Set[Predicate]:
        return {self._InTarget}

    @property
    def goal_predicates(self) -> Set[Predicate]:
        return {self._InTarget}

    @classmethod
    def initialize_pybullet(
            cls, using_gui: bool
    ) -> Tuple[int, SingleArmPyBulletRobot, Dict[str, Any]]:
        physics_client_id, pybullet_robot, bodies = super().initialize_pybullet(using_gui)

        # Table (scaled 1x)
        table_id = p.loadURDF(utils.get_env_asset_path("urdf/table.urdf"),
                              useFixedBase=True,
                              globalScaling=1.0,
                              physicsClientId=physics_client_id)
        p.resetBasePositionAndOrientation(table_id,
                                          cls._table_pose,
                                          cls._table_orientation,
                                          physicsClientId=physics_client_id)
        bodies["table_id"] = table_id

        # Target: a painted square, visual only. It used to carry a
        # collision box, which made it a 2 mm slab floating 4 mm clear of
        # the table -- a 6 mm lip the disc had to climb to score. A clean
        # unobstructed shove into it jammed the arm at 175 N with the disc's
        # leading edge on the rim and went no further (measured 2026-09-09),
        # so the goal was unreachable by pushing once the target sat inside
        # the arm's band. A mark on a table is not an obstacle.
        visual_id = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=(cls.target_width / 2, cls.target_height / 2, 0.001),
            rgbaColor=cls.target_color,
            physicsClientId=physics_client_id
        )
        target_id = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=visual_id,
            physicsClientId=physics_client_id
        )
        bodies["target_id"] = target_id

        # Donuts
        donut_ids = []
        for _ in range(cls.num_donuts):
            donut_id = cls._create_pybullet_disc(
                color=(0.8, 0.5, 0.2, 1.0),
                radius=cls.donut_radius,
                half_height=cls.donut_half_height,
                mass=cls.donut_mass,
                friction=cls.donut_friction,
                physics_client_id=physics_client_id
            )
            donut_ids.append(donut_id)
        bodies["donut_ids"] = donut_ids

        return physics_client_id, pybullet_robot, bodies

    @staticmethod
    def _create_pybullet_disc(
        color: Tuple[float, float, float, float],
        radius: float,
        half_height: float,
        mass: float,
        friction: float,
        position: Pose3D = (0.0, 0.0, 0.0),
        physics_client_id: int = 0,
    ) -> int:
        """Create a solid disc: no hole to hook, too wide to grasp, so the arm
        can only push it from the side."""
        collision_id = p.createCollisionShape(
            p.GEOM_CYLINDER,
            radius=radius,
            height=2 * half_height,
            physicsClientId=physics_client_id)
        visual_id = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=radius,
            length=2 * half_height,
            rgbaColor=color,
            physicsClientId=physics_client_id)
        disc_id = p.createMultiBody(baseMass=mass,
                                    baseCollisionShapeIndex=collision_id,
                                    baseVisualShapeIndex=visual_id,
                                    basePosition=position,
                                    physicsClientId=physics_client_id)
        p.changeDynamics(disc_id,
                         -1,
                         lateralFriction=friction,
                         rollingFriction=0.01,
                         spinningFriction=0.01,
                         physicsClientId=physics_client_id)
        return disc_id

    def _store_pybullet_bodies(self, pybullet_bodies: Dict[str, Any]) -> None:
        self._table_id = pybullet_bodies["table_id"]
        self._target_id = pybullet_bodies["target_id"]
        self._donut_ids = pybullet_bodies["donut_ids"]
        # Expose PyBullet ids on the Objects so body ids map back to names
        # (segmentation masks, particle extraction, get_object_by_id).
        self._target.id = self._target_id
        for donut, donut_id in zip(self._donuts, self._donut_ids):
            donut.id = donut_id

    def _get_object_ids_for_held_check(self) -> List[int]:
        """No body in this domain is graspable.

        The discs are 12 cm across against a ~7 cm jaw span, so a pick should
        be impossible on geometry alone -- but the base class's pinch test
        fired on them anyway in sweep 2 (the gripper reported holding an
        object in eight of nine runs), presumably off a disc tipped onto its
        25 mm edge or cammed between the pads. Returning nothing here removes
        the grasp constraint as a possibility rather than as a likelihood:
        the fingers still collide with and shove the discs.
        """
        return []

    def _get_domain_specific_feature(self, obj: Object, feature: str) -> float:
        raise ValueError(f"Unknown feature {feature} for object {obj}")

    def _set_seed(self, seed: int) -> None:
        super()._set_seed(seed)
        self._rng = np.random.default_rng(seed)

    def reset(self,
              train_or_test: str,
              task_idx: int,
              render: bool = False) -> Observation:
        self._step_count = 0
        return super().reset(train_or_test, task_idx, render=render)

    def _is_out_of_view(self, donut_id: int) -> bool:
        (dx, _dy, _dz), _ = p.getBasePositionAndOrientation(
            donut_id, physicsClientId=self._physics_client_id)
        return dx > 5.0  # parked at _out_of_view_xy

    def _domain_specific_step(self) -> None:
        self._step_count += 1
        # A disc shoved off the table is parked out of view and stays gone:
        # it is unreachable from the floor, and nothing respawns here.
        for i, donut_id in enumerate(self._donut_ids):
            if self._is_out_of_view(donut_id):
                continue
            (_dx, _dy, dz), _ = p.getBasePositionAndOrientation(
                donut_id, physicsClientId=self._physics_client_id)
            if dz < self.table_height - 0.05:
                self._park_donut(i)

    def _park_donut(self, idx: int) -> None:
        p.resetBasePositionAndOrientation(
            self._donut_ids[idx],
            [self._out_of_view_xy[0], self._out_of_view_xy[1], 10.0 + idx * 0.1],
            (0., 0., 0., 1.),
            physicsClientId=self._physics_client_id)
        p.resetBaseVelocity(self._donut_ids[idx], [0, 0, 0], [0, 0, 0],
                            physicsClientId=self._physics_client_id)

    def _set_domain_specific_state(self, state: State) -> None:
        # Target
        target_obj = state.get_objects(self._target_type)[0]
        tx = state.get(target_obj, "x")
        ty = state.get(target_obj, "y")
        tz = state.get(target_obj, "z")
        update_object(self._target_id, position=(tx, ty, tz), 
                      physics_client_id=self._physics_client_id)

        # Donuts
        donut_objs = state.get_objects(self._donut_type)
        for i, donut_obj in enumerate(donut_objs):
            donut_id = self._donut_ids[i]
            dx = state.get(donut_obj, "x")
            dy = state.get(donut_obj, "y")
            dz = state.get(donut_obj, "z")
            p.resetBasePositionAndOrientation(
                donut_id, [dx, dy, dz],
                (0., 0., 0., 1.),
                physicsClientId=self._physics_client_id)
            
            r = state.get(donut_obj, "r")
            g = state.get(donut_obj, "g")
            b = state.get(donut_obj, "b")
            p.changeVisualShape(donut_id, -1, rgbaColor=(r, g, b, 1.0),
                                physicsClientId=self._physics_client_id)

        # Grasping
        held_donut = None
        for donut in donut_objs:
            if state.get(donut, "is_held") > 0.5:
                held_donut = donut
                break
        if held_donut is not None:
            idx = donut_objs.index(held_donut)
            self._held_obj_id = self._donut_ids[idx]
            self._create_grasp_constraint()

    def _get_state(self) -> State:
        state_dict = {}
        # Robot
        rx, ry, rz, _, _, _, _, rf = self._pybullet_robot.get_state()
        fingers = self._fingers_joint_to_state(self._pybullet_robot, rf)
        state_dict[self._robot] = np.array([rx, ry, rz, fingers], dtype=np.float32)
        # Target
        (tx, ty, tz), _ = p.getBasePositionAndOrientation(self._target_id,
                                                         physicsClientId=self._physics_client_id)
        visual_data = p.getVisualShapeData(self._target_id,
                                           physicsClientId=self._physics_client_id)[0]
        tr, tg, tb, _ = visual_data[7]
        state_dict[self._target] = np.array([tx, ty, tz, tr, tg, tb], dtype=np.float32)
        # Donuts
        for i, donut_id in enumerate(self._donut_ids):
            (dx, dy, dz), _ = p.getBasePositionAndOrientation(donut_id,
                                                             physicsClientId=self._physics_client_id)
            held = (donut_id == self._held_obj_id)
            visual_data = p.getVisualShapeData(donut_id,
                                               physicsClientId=self._physics_client_id)[0]
            dr, dg, db, _ = visual_data[7]
            if i < len(self._donuts):
                state_dict[self._donuts[i]] = np.array([dx, dy, dz, float(held), dr, dg, db],
                                                       dtype=np.float32)
        return utils.PyBulletState(state_dict, 
                                   simulator_state=self._pybullet_robot.get_joints())

    @classmethod
    def _InTarget_holds(cls, state: State, objects: Sequence[Object]) -> bool:
        """Donut rests on the table inside the target square (not held, not
        hovering above it)."""
        donut, target = objects
        if state.get(donut, "is_held") > 0.5:
            return False
        dx = state.get(donut, "x")
        dy = state.get(donut, "y")
        dz = state.get(donut, "z")
        tx = state.get(target, "x")
        ty = state.get(target, "y")
        tz = state.get(target, "z")
        if dz > tz + 2.5 * cls.donut_half_height:
            return False
        dist = np.sqrt((dx - tx)**2 + (dy - ty)**2)
        return dist < (cls.target_width / 2)

    def _generate_train_tasks(self) -> List[EnvironmentTask]:
        return self._make_tasks(num_tasks=CFG.num_train_tasks, rng=self._train_rng)

    def _generate_test_tasks(self) -> List[EnvironmentTask]:
        return self._make_tasks(num_tasks=CFG.num_test_tasks, rng=self._test_rng)

    def _make_tasks(self, num_tasks: int, rng: np.random.Generator) -> List[EnvironmentTask]:
        from predicators.pybullet_helpers.objects import \
            sample_collision_free_2d_positions
        tasks = []
        for _ in range(num_tasks):
            init_dict = {}
            init_dict[self._robot] = {
                "pose_x": self.robot_init_x,
                "pose_y": self.robot_init_y,
                "pose_z": self.robot_init_z,
                "fingers": self.open_fingers
            }
            target_x = self.push_lane_x
            target_y = self.y_ub
            init_dict[self._target] = {
                "x": target_x,
                "y": target_y,
                "z": self.table_height + 0.005, # Slightly above surface
                "r": self.target_color[0],
                "g": self.target_color[1],
                "b": self.target_color[2]
            }
            goal_pos = sample_collision_free_2d_positions(
                1,  # the goal disc, at the near end of the lane
                x_range=(self.push_lane_x - 0.02, self.push_lane_x + 0.02),
                y_range=(self.y_lb, self.y_lb + 0.04),
                shape_type="circle",
                shape_params=(self.donut_radius,),
                rng=rng
            )[0]
            # The distractors are on the table from the start, spread down
            # the lane the push has to cross. Sampled clear of each other;
            # the band is already clear of the goal disc and the target.
            lane_positions = sample_collision_free_2d_positions(
                self.num_donuts - 1,
                x_range=(self.x_lb, self.x_ub),
                y_range=(self.lane_y_lb, self.lane_y_ub),
                shape_type="circle",
                shape_params=(self.donut_radius,),
                rng=rng
            )
            donut_positions = [goal_pos] + list(lane_positions)
            z = self.table_height + self.donut_half_height
            for i, donut in enumerate(self._donuts):
                pos = donut_positions[i]
                color = self.goal_disc_color if i == 0 \
                    else self.other_disc_color
                init_dict[donut] = {
                    "x": pos[0],
                    "y": pos[1],
                    "z": z,
                    "is_held": 0.0,
                    "r": color[0],
                    "g": color[1],
                    "b": color[2]
                }
            init_state = utils.create_state_from_dict(init_dict)
            goal = {GroundAtom(self._InTarget, [self._donuts[0], self._target])}
            tasks.append(EnvironmentTask(init_state, goal))
        return self._add_pybullet_state_to_tasks(tasks)

if __name__ == "__main__":
    import time
    CFG.seed = 0
    CFG.env = "pybullet_donut"
    CFG.num_train_tasks = 1
    env = PyBulletDonutEnv(use_gui=True)
    _task = env._generate_train_tasks()[0]
    env.reset("train", 0)
    while True:
        _act = Action(np.array(env._pybullet_robot.get_joints()))
        env.step(_act)
        time.sleep(0.01)
