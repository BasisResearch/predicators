"""Observable simulation core of the balloons environment.

This module is the balloons env's BASE SIM: the table, a box, a rack of
balloons each held in a clip, the ceiling, the target band, and the
state read/write of everything visible, including the clip mechanics
(a clip is a toggle switch the robot pushes open). It deliberately
contains NO lift law, no release rule, no box masses, no task
generation and no predicate / goal semantics. On the base sim a clip
opens and nothing else happens.

The boundary is the busyboard's visibility contract: when
``CFG.agent_sim_provide_base_sim_source`` is on, THIS FILE is copied
into the learning agent's sandbox as reference material, so the file
the agent reads is byte-identical to the code its base-sim rollouts
execute. Anything that would leak the learning target - how hard each
colour pulls, how the pull fades with height, how heavy each box is,
what an open clip does - lives in ``pybullet_balloons.py``, never here.

Physical layout (a tabletop, the robot at the near side):

- The ``box``: a cube on the table in front of the robot, the payload.
  Its ``color`` is its material class.
- ``balloon`` objects: spheres resting in a rack along the back of the
  table, each tied by its string to the box and held in the rack by a
  ``clip``. A balloon's ``color`` is its gas class; ``tied`` says its
  string has pulled it to the box, ``popped`` that it has burst.
  Nothing in this file ever sets either.
- ``clip`` objects: toggle switches in a row in front of the rack, one
  per balloon, directly in front of it. ``is_on`` is the switch's
  latched state; the robot pushes a clip open. Nothing in this file
  says what opening one does.
- The ``band``: a translucent slab floating beside the box's column,
  spanning the heights the goal wants the box to float at.
- The ``ceiling``: a plate drawn over the table, at ``ceiling_z``.
  Nothing in this file says what reaching it does to a balloon.
"""
from typing import Any, ClassVar, Dict, List, Tuple

import numpy as np
import pybullet as p

from predicators.envs.pybullet_env import PyBulletEnv
from predicators.pybullet_helpers.geometry import Pose3D, Quaternion
from predicators.pybullet_helpers.objects import cap_switch_joint_travel, \
    create_object, create_pybullet_block, create_pybullet_sphere, \
    update_object
from predicators.pybullet_helpers.robots import SingleArmPyBulletRobot
from predicators.settings import CFG
from predicators.structs import Object, State, Type


class PyBulletBalloonsBaseEnv(PyBulletEnv):
    """Sim core of the balloons puzzle: a box, clipped balloons, a band, a
    ceiling.

    Abstract on purpose - it defines no name, predicates, tasks or
    domain-specific step, so env discovery skips it; the concrete env is
    ``PyBulletBalloonsEnv``.
    """

    @classmethod
    def get_base_sim_source_files(cls) -> List[str]:
        return [
            "predicators/envs/pybullet_balloons_base.py",
            "predicators/envs/pybullet_env.py",
        ]

    # =========================================================================
    # WORKSPACE & TABLE
    # =========================================================================
    table_height: ClassVar[float] = 0.4
    table_pos: ClassVar[Pose3D] = (0.75, 1.35, table_height / 2)
    table_orn: ClassVar[Quaternion] = p.getQuaternionFromEuler(
        [0.0, 0.0, np.pi / 2.0])

    x_lb: ClassVar[float] = 0.4
    x_ub: ClassVar[float] = 1.1
    y_lb: ClassVar[float] = 1.0
    y_ub: ClassVar[float] = 1.6
    z_lb: ClassVar[float] = table_height
    z_ub: ClassVar[float] = 0.75 + table_height / 2
    x_mid: ClassVar[float] = (x_lb + x_ub) / 2
    y_mid: ClassVar[float] = (y_lb + y_ub) / 2

    # =========================================================================
    # ROBOT
    # =========================================================================
    robot_init_x: ClassVar[float] = x_mid
    robot_init_y: ClassVar[float] = 1.1
    robot_init_z: ClassVar[float] = z_ub - 0.1
    robot_base_pos: ClassVar[Pose3D] = (0.75, 0.65, 0.0)
    robot_base_orn: ClassVar[Quaternion] = p.getQuaternionFromEuler(
        [0.0, 0.0, np.pi / 2])
    robot_init_tilt: ClassVar[float] = np.pi / 2
    robot_init_wrist: ClassVar[float] = -np.pi / 2

    # =========================================================================
    # CAMERA
    # =========================================================================
    _camera_distance: ClassVar[float] = 1.3
    _camera_yaw: ClassVar[float] = 150
    _camera_pitch: ClassVar[float] = -14
    _camera_target: ClassVar[Tuple[float, float, float]] = (0.7, 1.28, 0.72)

    # =========================================================================
    # LAYOUT
    # =========================================================================
    box_half: ClassVar[float] = 0.035
    # The box sits at the left of the table, clear of the rack.
    box_xy: ClassVar[Tuple[float, float]] = (0.42, 1.2)
    box_z: ClassVar[float] = table_height + box_half
    # The mass every box has in this file.
    box_base_mass: ClassVar[float] = 0.1
    box_friction: ClassVar[float] = 0.6

    balloon_radius: ClassVar[float] = 0.03
    balloon_mass: ClassVar[float] = 0.005
    # The rack: a row along the back of the table, right of the box, the
    # clips a row in front of it, each clip directly in front of its
    # balloon.
    rack_y: ClassVar[float] = 1.42
    clip_y: ClassVar[float] = 1.24
    rack_center_x: ClassVar[float] = 0.87
    rack_x_gap: ClassVar[float] = 0.16
    balloon_rest_z: ClassVar[float] = table_height + balloon_radius
    # How far above the box's top a freed balloon's string holds it;
    # later balloons stack above earlier ones.
    string_length: ClassVar[float] = 0.04
    popped_color: ClassVar[Tuple[float, float, float,
                                 float]] = (0.45, 0.45, 0.45, 1.0)

    # Clips: the PartNet-Mobility toggle switch the boil, fan and
    # busyboard envs use, so the shared push skill operates it as is.
    clip_rot: ClassVar[float] = 0.0
    clip_scale: ClassVar[float] = 1.4
    switch_joint_scale: ClassVar[float] = 0.1
    _clip_slider_link: ClassVar[int] = 2
    switch_on_threshold: ClassVar[float] = 0.5
    # Height above a clip's base at which the slider sits; the push
    # skill aims here.
    clip_press_height: ClassVar[float] = 0.10
    clip_color: ClassVar[Tuple[float, float, float,
                               float]] = (0.35, 0.35, 0.38, 1.0)
    clip_slider_color: ClassVar[Tuple[float, float, float,
                                      float]] = (0.92, 0.92, 0.94, 1.0)

    # The ceiling: a plate drawn over the table.
    ceiling_z: ClassVar[float] = table_height + 0.78
    ceiling_half_extents: ClassVar[Tuple[float, float,
                                         float]] = (0.45, 0.35, 0.004)
    ceiling_color: ClassVar[Tuple[float, float, float,
                                  float]] = (0.85, 0.85, 0.90, 0.35)

    # The chute: two fixed vertical walls flanking the box's column, a
    # slot the box must rise through to reach the band. They collide ONLY
    # with the box (the arm and the balloons pass through, like the
    # ceiling picture), so a box that tilts or sways as it climbs - which
    # an off-centre, torque-unbalanced set of balloons makes it do -
    # catches a wall and jams below the band. The gap is just wider than
    # the box, so only a balanced set that rises straight threads it.
    chute_half_gap: ClassVar[float] = 0.048
    chute_wall_half_thickness: ClassVar[float] = 0.004
    chute_wall_half_depth: ClassVar[float] = 0.08
    chute_z_lo: ClassVar[float] = table_height + 0.10
    chute_z_hi: ClassVar[float] = table_height + 0.60
    chute_color: ClassVar[Tuple[float, float, float,
                                float]] = (0.55, 0.55, 0.62, 0.55)

    # How far off the box's centre the freed balloons attach, spread evenly
    # across the box top by clip position (index 0 leftmost). An asymmetric
    # set therefore pulls off-centre and TILTS the free-body box as it climbs,
    # so it catches a chute wall; only a balanced set rises straight and
    # threads the slot. Kept just inside the box half-width so every attach
    # point is on the box.
    attach_span: ClassVar[float] = 0.028

    @classmethod
    def _attach_offset(cls, index: int, n_balloons: int) -> float:
        """The x-offset of balloon ``index``'s attach point on the box top,
        evenly spread across ``[-attach_span, +attach_span]`` by clip position;
        0 for a lone balloon."""
        if n_balloons <= 1:
            return 0.0
        frac = index / (n_balloons - 1)  # 0..1
        return float(cls.attach_span * (2.0 * frac - 1.0))

    # The band: a translucent slab beside the box's column. Its ``lo``
    # and ``hi`` are heights of the box's centre.
    band_offset_x: ClassVar[float] = -0.08
    band_half_xy: ClassVar[float] = 0.03
    band_color: ClassVar[Tuple[float, float, float,
                               float]] = (0.30, 0.80, 0.40, 0.45)

    # Material palettes: the box's ``color`` indexes the first, a
    # balloon's the second.
    BOX_PALETTE: ClassVar[List[Tuple[str,
                                     Tuple[float, float, float, float]]]] = [
                                         ("pine", (0.85, 0.70, 0.45, 1.0)),
                                         ("oak", (0.50, 0.35, 0.20, 1.0)),
                                         ("teak", (0.30, 0.18, 0.10, 1.0)),
                                     ]
    BALLOON_PALETTE: ClassVar[List[Tuple[str, Tuple[float, float, float,
                                                    float]]]] = [
                                                        ("red", (0.90, 0.15,
                                                                 0.15, 1.0)),
                                                        ("blue", (0.20, 0.35,
                                                                  0.90, 1.0)),
                                                        ("green", (0.20, 0.70,
                                                                   0.30, 1.0)),
                                                        ("gold", (0.95, 0.80,
                                                                  0.20, 1.0)),
                                                    ]

    # =========================================================================
    # TYPES
    # =========================================================================
    _robot_type = Type("robot",
                       ["x", "y", "z", "fingers", "roll", "tilt", "wrist"])
    _box_type = Type("box", ["x", "y", "z", "color", "speed"])
    _balloon_type = Type("balloon", ["x", "y", "z", "color", "tied", "popped"])
    _clip_type = Type("clip", ["x", "y", "z", "rot", "is_on"],
                      sim_features=["id", "joint_id"])
    _band_type = Type("band", ["x", "y", "lo", "hi"])

    @classmethod
    def box_color_name(cls, color_index: int) -> str:
        """The box palette name behind a ``color`` feature value."""
        return cls.BOX_PALETTE[int(round(color_index))][0]

    @classmethod
    def balloon_color_name(cls, color_index: int) -> str:
        """The balloon palette name behind a ``color`` feature value."""
        return cls.BALLOON_PALETTE[int(round(color_index))][0]

    @classmethod
    def _max_balloons(cls) -> int:
        return max(
            list(CFG.balloons_num_balloons_train) +
            list(CFG.balloons_num_balloons_test))

    @classmethod
    def box_top_point(cls, state: State,
                      box: Object) -> Tuple[float, float, float]:
        """The centre of the box's top face, where strings are tied."""
        return (state.get(box, "x"), state.get(box, "y"),
                state.get(box, "z") + cls.box_half)

    @classmethod
    def rack_xs(cls, count: int) -> List[float]:
        """x of each rack place, left to right."""
        return [
            cls.rack_center_x + (i - (count - 1) / 2) * cls.rack_x_gap
            for i in range(count)
        ]

    # =========================================================================
    # CONSTRUCTION
    # =========================================================================
    def __init__(self, use_gui: bool = False, **kwargs: Any) -> None:
        self._robot = Object("robot", self._robot_type)
        self._box = Object("box", self._box_type)
        self._balloons: List[Object] = [
            Object(f"balloon{i}", self._balloon_type)
            for i in range(self._max_balloons())
        ]
        self._clips: List[Object] = [
            Object(f"clip{i}", self._clip_type)
            for i in range(self._max_balloons())
        ]
        self._band = Object("band", self._band_type)
        self._box_color: int = 0
        self._balloon_colors: Dict[str, int] = {}
        self._tied: Dict[str, bool] = {}
        self._popped: Dict[str, bool] = {}
        self._band_lo: float = 0.0
        self._band_hi: float = 0.0
        super().__init__(use_gui, **kwargs)

    @classmethod
    def initialize_pybullet(
            cls, using_gui: bool
    ) -> Tuple[int, SingleArmPyBulletRobot, Dict[str, Any]]:
        physics_client_id, pybullet_robot, bodies = super(
        ).initialize_pybullet(using_gui)
        bodies["table_id"] = create_object(asset_path="urdf/table.urdf",
                                           position=cls.table_pos,
                                           orientation=cls.table_orn,
                                           scale=1.0,
                                           use_fixed_base=True,
                                           physics_client_id=physics_client_id)
        bodies["box_id"] = create_pybullet_block(
            color=cls.BOX_PALETTE[0][1],
            half_extents=(cls.box_half, cls.box_half, cls.box_half),
            mass=cls.box_base_mass,
            friction=cls.box_friction,
            physics_client_id=physics_client_id)
        balloon_ids = []
        for _ in range(cls._max_balloons()):
            balloon_ids.append(
                create_pybullet_sphere(color=cls.BALLOON_PALETTE[0][1],
                                       radius=cls.balloon_radius,
                                       mass=cls.balloon_mass,
                                       friction=0.8,
                                       spinning_friction=0.05,
                                       rolling_friction=0.05,
                                       physics_client_id=physics_client_id))
        bodies["balloon_ids"] = balloon_ids

        clip_ids = []
        for _ in range(cls._max_balloons()):
            clip_id = create_object(
                asset_path="urdf/partnet_mobility/switch/102812/switch.urdf",
                scale=cls.clip_scale,
                use_fixed_base=True,
                physics_client_id=physics_client_id)
            for shape in p.getVisualShapeData(
                    clip_id, physicsClientId=physics_client_id):
                link_idx = shape[1]
                color = (cls.clip_slider_color if link_idx
                         == cls._clip_slider_link else cls.clip_color)
                p.changeVisualShape(clip_id,
                                    link_idx,
                                    rgbaColor=color,
                                    physicsClientId=physics_client_id)
            j_id = cls._get_joint_id(clip_id, "joint_0", physics_client_id)
            cap_switch_joint_travel(clip_id, j_id, cls.switch_joint_scale,
                                    physics_client_id)
            clip_ids.append(clip_id)
        bodies["clip_ids"] = clip_ids

        # The ceiling is a picture, not a body: the arm swings through
        # its height, and what a balloon does there is a rule of the
        # concrete env, read off the balloon's height.
        ceiling_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=cls.ceiling_half_extents,
            rgbaColor=cls.ceiling_color,
            physicsClientId=physics_client_id)
        bodies["ceiling_id"] = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=ceiling_visual,
            basePosition=(cls.x_mid, 1.35, cls.ceiling_z),
            physicsClientId=physics_client_id)
        # The chute walls: real collision bodies, but filtered below to
        # collide only with the box. Centred on the box's column, one on
        # each side of the slot.
        chute_ids = []
        wall_half = (cls.chute_wall_half_thickness, cls.chute_wall_half_depth,
                     (cls.chute_z_hi - cls.chute_z_lo) / 2.0)
        wall_z = (cls.chute_z_lo + cls.chute_z_hi) / 2.0
        for sign in (-1.0, 1.0):
            wall_col = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=wall_half,
                physicsClientId=physics_client_id)
            wall_vis = p.createVisualShape(p.GEOM_BOX,
                                           halfExtents=wall_half,
                                           rgbaColor=cls.chute_color,
                                           physicsClientId=physics_client_id)
            wall_x = cls.box_xy[0] + sign * (cls.chute_half_gap +
                                             cls.chute_wall_half_thickness)
            chute_ids.append(
                p.createMultiBody(baseMass=0.0,
                                  baseCollisionShapeIndex=wall_col,
                                  baseVisualShapeIndex=wall_vis,
                                  basePosition=(wall_x, cls.box_xy[1], wall_z),
                                  physicsClientId=physics_client_id))
        bodies["chute_ids"] = chute_ids
        band_visual = p.createVisualShape(p.GEOM_BOX,
                                          halfExtents=(cls.band_half_xy,
                                                       cls.band_half_xy, 0.01),
                                          rgbaColor=cls.band_color,
                                          physicsClientId=physics_client_id)
        bodies["band_id"] = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=band_visual,
            physicsClientId=physics_client_id)
        return physics_client_id, pybullet_robot, bodies

    def _store_pybullet_bodies(self, pybullet_bodies: Dict[str, Any]) -> None:
        self._table_id = pybullet_bodies["table_id"]
        self._ceiling_id = pybullet_bodies["ceiling_id"]
        self._robot.id = self._pybullet_robot.robot_id
        self._box.id = pybullet_bodies["box_id"]
        self._band.id = pybullet_bodies["band_id"]
        self._chute_ids = pybullet_bodies.get("chute_ids", [])
        for i, balloon in enumerate(self._balloons):
            balloon.id = pybullet_bodies["balloon_ids"][i]
        for i, clip in enumerate(self._clips):
            clip.id = pybullet_bodies["clip_ids"][i]
            clip.joint_id = self._get_joint_id(clip.id, "joint_0",
                                               self._physics_client_id)
        # The chute walls gate only the box: disable their collision with
        # everything, then re-enable just the wall-box pairs, so the arm
        # and the balloons pass through them freely.
        for wall_id in self._chute_ids:
            p.setCollisionFilterGroupMask(
                wall_id,
                -1,
                collisionFilterGroup=0,
                collisionFilterMask=0,
                physicsClientId=self._physics_client_id)
            p.setCollisionFilterPair(wall_id,
                                     self._box.id,
                                     -1,
                                     -1,
                                     enableCollision=1,
                                     physicsClientId=self._physics_client_id)

    # =========================================================================
    # CLIP MECHANICS
    # =========================================================================
    @staticmethod
    def _get_joint_id(obj_id: int,
                      joint_name: str,
                      physics_client_id: int = 0) -> int:
        """Find a joint by name in a URDF, or -1 if absent."""
        for j in range(
                p.getNumJoints(obj_id, physicsClientId=physics_client_id)):
            info = p.getJointInfo(obj_id, j, physicsClientId=physics_client_id)
            if info[1].decode("utf-8") == joint_name:
                return j
        return -1

    def _is_clip_on(self, clip: Object) -> bool:
        """Read a clip's latched state from its prismatic joint."""
        if clip.id is None or clip.joint_id is None or clip.joint_id < 0:
            return False
        j_pos = p.getJointState(clip.id,
                                clip.joint_id,
                                physicsClientId=self._physics_client_id)[0]
        info = p.getJointInfo(clip.id,
                              clip.joint_id,
                              physicsClientId=self._physics_client_id)
        j_min, j_max = info[8], info[9]
        frac = (j_pos / self.switch_joint_scale - j_min) / (j_max - j_min)
        return bool(frac > self.switch_on_threshold)

    def _set_clip_on(self, clip: Object, is_on: bool) -> None:
        """Programmatically latch a clip open or closed."""
        if clip.joint_id is None or clip.joint_id < 0:
            return
        info = p.getJointInfo(clip.id,
                              clip.joint_id,
                              physicsClientId=self._physics_client_id)
        j_min, j_max = info[8], info[9]
        target = (j_max if is_on else j_min) * self.switch_joint_scale
        p.resetJointState(clip.id,
                          clip.joint_id,
                          target,
                          physicsClientId=self._physics_client_id)

    # =========================================================================
    # STATE READ / WRITE
    # =========================================================================
    def _get_object_ids_for_held_check(self) -> List[int]:
        """Nothing here is grasped; clips are pushed."""
        return []

    def _active_balloons(self, state: State) -> List[Object]:
        return sorted((o for o in state if o.type.name == "balloon"),
                      key=lambda o: o.name)

    def _active_clips(self, state: State) -> List[Object]:
        return sorted((o for o in state if o.type.name == "clip"),
                      key=lambda o: o.name)

    def _speed(self, obj: Object) -> float:
        (vx, vy,
         vz), _ = p.getBaseVelocity(obj.id,
                                    physicsClientId=self._physics_client_id)
        return float(np.linalg.norm([vx, vy, vz]))

    def _paint_balloon(self, balloon: Object) -> None:
        if self._popped.get(balloon.name, False):
            color = self.popped_color
        else:
            color = self.BALLOON_PALETTE[self._balloon_colors.get(
                balloon.name, 0)][1]
        p.changeVisualShape(balloon.id,
                            -1,
                            rgbaColor=color,
                            physicsClientId=self._physics_client_id)

    def _get_domain_specific_feature(self, obj: Object, feature: str) -> float:
        if obj.type.name == "box":
            if feature == "color":
                return float(self._box_color)
            if feature == "speed":
                return self._speed(obj)
        if obj.type.name == "balloon":
            if feature == "color":
                return float(self._balloon_colors.get(obj.name, 0))
            if feature == "tied":
                return float(self._tied.get(obj.name, False))
            if feature == "popped":
                return float(self._popped.get(obj.name, False))
        if obj.type.name == "clip" and feature == "is_on":
            return float(self._is_clip_on(obj))
        if obj.type.name == "band":
            if feature == "x":
                return float(self.box_xy[0] + self.band_offset_x)
            if feature == "y":
                return float(self.box_xy[1])
            if feature == "lo":
                return self._band_lo
            if feature == "hi":
                return self._band_hi
        raise ValueError(f"Unknown feature {feature} for object {obj}")

    def _set_domain_specific_state(self, state: State) -> None:
        self._box_color = int(round(state.get(self._box, "color")))
        p.changeVisualShape(self._box.id,
                            -1,
                            rgbaColor=self.BOX_PALETTE[self._box_color][1],
                            physicsClientId=self._physics_client_id)
        balloons = self._active_balloons(state)
        self._balloon_colors = {}
        self._tied = {}
        self._popped = {}
        for balloon in balloons:
            self._balloon_colors[balloon.name] = int(
                round(state.get(balloon, "color")))
            self._tied[balloon.name] = state.get(balloon, "tied") > 0.5
            self._popped[balloon.name] = state.get(balloon, "popped") > 0.5
            self._paint_balloon(balloon)
        for clip in self._active_clips(state):
            self._set_clip_on(clip, state.get(clip, "is_on") > 0.5)
        oov_x, oov_y = self._out_of_view_xy
        for i in range(len(balloons), len(self._balloons)):
            update_object(self._balloons[i].id,
                          position=(oov_x, oov_y, 0.2 * i),
                          physics_client_id=self._physics_client_id)
            update_object(self._clips[i].id,
                          position=(oov_x, oov_y + 1.0, 0.2 * i),
                          physics_client_id=self._physics_client_id)
        self._band_lo = state.get(self._band, "lo")
        self._band_hi = state.get(self._band, "hi")
        mid = (self._band_lo + self._band_hi) / 2
        half_h = max(0.005, (self._band_hi - self._band_lo) / 2)
        # The band's visual is rebuilt per level: its height varies.
        p.removeBody(self._band.id, physicsClientId=self._physics_client_id)
        band_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=(self.band_half_xy, self.band_half_xy, half_h),
            rgbaColor=self.band_color,
            physicsClientId=self._physics_client_id)
        self._band.id = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=band_visual,
            basePosition=(state.get(self._band,
                                    "x"), state.get(self._band, "y"), mid),
            physicsClientId=self._physics_client_id)
