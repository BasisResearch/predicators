"""Real-bench fan env: one fan, one button, one ball, a row of zones.

A simplification of ``pybullet_fan`` (which surrounds a grid with twenty
fans and a maze) down to the bench described in ``real_fan.png``: the
robot holds down a momentary button that powers a single fan, the fan
blows a ball along a lane, and the ball must coast to rest in a chosen
numbered zone. The only thing that decides where it stops is HOW LONG
the button is held, so the whole domain is one option with one
continuous parameter.

Where the two envs really differ is not the object count. ``pybullet_fan``
decides where the ball stops with GEOMETRY -- the ball advances one grid
cell at a time while the fan is on, and the agent counts observed cell
crossings before switching off, so "how long" is the length of the plan
and no option needs a duration. This lane has no cells to count, the
ball's position is continuous, and it coasts after the fan stops, so the
duration has to be chosen up front as a value.

The bench itself is the ``pybullet_domino_real`` bench: same robot base
pose, same table (plus the extended tile and the robot pedestal), same
base-frame -> world transplant. Those constants are read from the shared
``domino_real_*`` settings rather than copied, so recalibrating the bench
moves both envs together.

This env is pure simulation. Nothing here drives a real arm and nothing
reads a camera; the real-execution and re-perception plumbing that
``pybullet_domino_real`` carries is deliberately absent.
"""
from __future__ import annotations

from typing import Any, ClassVar, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pybullet as p

from predicators import utils
from predicators.envs.pybullet_domino.real_geometry import \
    domino_world_z_offset
from predicators.envs.pybullet_env import PyBulletEnv
from predicators.pybullet_helpers.geometry import Pose3D, Quaternion
from predicators.pybullet_helpers.objects import create_object, \
    create_pybullet_block, create_pybullet_sphere
from predicators.pybullet_helpers.robots import SingleArmPyBulletRobot
from predicators.settings import CFG
from predicators.structs import EnvironmentTask, GroundAtom, Object, \
    Predicate, State, Type


def _yaw_quat(yaw: float) -> Quaternion:
    """Z-only quaternion, as a plain tuple."""
    return tuple(p.getQuaternionFromEuler([0.0, 0.0, yaw]))


class PyBulletFanRealEnv(PyBulletEnv):
    """One fan, one button, one ball, ``fan_real_num_zones`` zones."""

    # =========================================================================
    # BENCH GEOMETRY (shared with pybullet_domino_real)
    # =========================================================================
    table_height: ClassVar[float] = 0.4
    table_pos: ClassVar[Pose3D] = (0.75, 1.35, table_height / 2)
    table_orn: ClassVar[Quaternion] = _yaw_quat(np.pi / 2)
    table_width: ClassVar[float] = 1.0
    # The extended tile toward the robot and the pedestal under the base,
    # ported from PyBulletDominoRealEnv._decorate.
    tile_pos: ClassVar[Pose3D] = (0.75, 0.85, 0.2)
    pedestal_xy: ClassVar[Tuple[float, float]] = (0.75, 0.72)

    x_lb: ClassVar[float] = 0.4
    x_ub: ClassVar[float] = 1.1
    y_lb: ClassVar[float] = 1.1
    y_ub: ClassVar[float] = 1.6
    z_lb: ClassVar[float] = table_height
    z_ub: ClassVar[float] = 0.95

    # =========================================================================
    # ROBOT (same base pose as pybullet_domino_real; the z is the bench
    # transplant offset, applied in initialize_pybullet)
    # =========================================================================
    robot_init_x: ClassVar[float] = (x_lb + x_ub) * 0.5
    robot_init_y: ClassVar[float] = (y_lb + y_ub) * 0.5
    robot_init_z: ClassVar[float] = z_ub
    robot_base_pos: ClassVar[Optional[Pose3D]] = (0.75, 0.72, 0.0)
    robot_base_orn: ClassVar[Optional[Quaternion]] = _yaw_quat(np.pi / 2)
    robot_init_tilt: ClassVar[float] = np.pi / 2
    robot_init_wrist: ClassVar[float] = -np.pi / 2

    # =========================================================================
    # FAN / BUTTON HARDWARE
    # =========================================================================
    fan_urdf: ClassVar[str] = "urdf/partnet_mobility/fan/101450/mobility.urdf"
    fan_scale: ClassVar[float] = 0.10
    fan_z_len: ClassVar[float] = 1.5 * fan_scale
    fan_spin_velocity: ClassVar[float] = 100.0
    joint_motor_force: ClassVar[float] = 20.0

    # The button is MOMENTARY: the fan blows while the gripper holds it
    # down and stops the moment the gripper lifts off. There is no latched
    # on/off state anywhere -- ``is_on`` is a function of where the end
    # effector is. That is why this env has no switch joint and no toggle
    # URDF: a lever is the wrong shape for a downward press and the wrong
    # semantics besides, and a latched bit could disagree with the arm
    # pose it is meant to follow.
    #
    # Detection is geometric, following pybullet_coffee's machine button:
    # pressed iff the EE is within ``fan_real_button_press_threshold`` of
    # the press point. The body is visual-only (collisions disabled) so a
    # descent onto it can never stall against its own target.
    button_half_extents: ClassVar[Tuple[float, float, float]] = \
        (0.025, 0.025, 0.01)
    # Where the gripper must be to hold the button down, above the body top.
    button_press_hover: ClassVar[float] = 0.02
    button_color_off: ClassVar[Tuple[float, float, float, float]] = \
        (0.55, 0.20, 0.20, 1.0)
    button_color_on: ClassVar[Tuple[float, float, float, float]] = \
        (0.25, 0.65, 0.30, 1.0)

    # =========================================================================
    # LANE FURNITURE
    # =========================================================================
    # Rails must clear the ball's equator, or the ball leans on a rail's top
    # edge and rides it like a slanted rail instead of touching it side-on
    # (see the same finding in pybullet_fan.boundary_wall_height).
    rail_height: ClassVar[float] = 0.06
    rail_thickness: ClassVar[float] = 0.004
    rail_color: ClassVar[Tuple[float, float, float, float]] = \
        (0.85, 0.83, 0.78, 1.0)
    zone_thickness: ClassVar[float] = 1e-5
    zone_color: ClassVar[Tuple[float, float, float, float]] = \
        (0.60, 0.87, 0.78, 1.0)
    ball_color: ClassVar[Tuple[float, float, float, float]] = \
        (0.30, 0.55, 0.95, 1.0)

    # =========================================================================
    # CAMERA
    # =========================================================================
    # Looking back across the lane from the far side, so the robot is BEHIND
    # the lane rather than between it and the camera. The domino bench's own
    # camera (yaw -70) puts the arm squarely over the zones, which is the one
    # thing a viewer needs to see.
    _camera_distance: ClassVar[float] = 1.0
    _camera_yaw: ClassVar[float] = 180
    _camera_pitch: ClassVar[float] = -60
    _camera_target: ClassVar[Pose3D] = (0.72, 1.40, 0.42)

    # =========================================================================
    # TYPES
    # =========================================================================
    _robot_type = Type("robot",
                       ["x", "y", "z", "fingers", "roll", "tilt", "wrist"],
                       angular_features=["roll", "tilt", "wrist"])
    _fan_type = Type("fan", ["x", "y", "z", "rot", "is_on"],
                     sim_features=["id", "joint_id"],
                     angular_features=["rot"])
    _button_type = Type("button", ["x", "y", "z", "is_on"],
                        sim_features=["id"])
    _ball_type = Type("ball", ["x", "y", "z"])
    _zone_type = Type("zone", ["x", "y", "idx"], sim_features=["id", "idx"])

    def __init__(self, use_gui: bool = False, **kwargs: Any) -> None:
        self._robot = Object("robot", self._robot_type)
        self._fan = Object("fan", self._fan_type)
        self._button = Object("button", self._button_type)
        self._ball = Object("ball", self._ball_type)
        self._zones = [
            Object(f"zone{i + 1}", self._zone_type)
            for i in range(CFG.fan_real_num_zones)
        ]

        super().__init__(use_gui=use_gui, **kwargs)

        self._FanOn = Predicate("FanOn", [self._fan_type],
                                self._FanOn_holds,
                                natural_language_assertion=lambda os:
                                f"the fan {os[0]} is running")
        self._FanOff = Predicate(
            "FanOff", [self._fan_type],
            lambda s, o: not self._FanOn_holds(s, o),
            natural_language_assertion=lambda os: f"the fan {os[0]} is off")
        self._BallInZone = Predicate(
            "BallInZone", [self._ball_type, self._zone_type],
            self._BallInZone_holds,
            natural_language_assertion=lambda os:
            f"the ball {os[0]} is resting in zone {os[1]}")
        self._Controls = Predicate("Controls",
                                   [self._button_type, self._fan_type],
                                   lambda s, o: True)

    @classmethod
    def get_name(cls) -> str:
        return "pybullet_fan_real"

    @property
    def types(self) -> Set[Type]:
        return {
            self._robot_type, self._fan_type, self._button_type,
            self._ball_type, self._zone_type
        }

    @property
    def predicates(self) -> Set[Predicate]:
        return {self._FanOn, self._FanOff, self._BallInZone, self._Controls}

    @property
    def goal_predicates(self) -> Set[Predicate]:
        return {self._BallInZone}

    @property
    def target_predicates(self) -> Set[Predicate]:
        return {self._BallInZone}

    # =========================================================================
    # DERIVED LAYOUT
    # =========================================================================
    @classmethod
    def zone_center_x(cls, idx: int) -> float:
        """World x of the center of zone ``idx`` (1-based, as drawn)."""
        return CFG.fan_real_first_zone_x + (idx - 1) * CFG.fan_real_zone_len

    @classmethod
    def _lane_end_x(cls) -> float:
        """World x just past the far edge of the last zone."""
        last = cls.zone_center_x(CFG.fan_real_num_zones)
        return last + CFG.fan_real_zone_len / 2 + 0.03

    @classmethod
    def _action_dt(cls) -> float:
        """Wall-clock seconds one env action covers.

        PyBullet's default 1/240 s timestep,
        ``pybullet_sim_steps_per_action`` sub-steps per action. This is
        the conversion between the burst duration the option is
        parameterized by (seconds) and the number of env steps the hold
        phase actually runs for.
        """
        return CFG.pybullet_sim_steps_per_action / 240.0

    # =========================================================================
    # PYBULLET BUILD
    # =========================================================================
    @classmethod
    def _apply_bench_geometry(cls) -> None:
        """Raise the robot base onto the real bench and adopt its wrist pose.

        Idempotent: the base xy is a constant, only the z is derived. Done
        in ``initialize_pybullet`` so it also takes effect for the skill
        factories' direct call, which bypasses ``__init__``.
        """
        z_off = domino_world_z_offset(CFG.domino_real_table_z)
        cls.robot_base_pos = (0.75, 0.72, float(z_off))
        cls.robot_init_tilt = float(CFG.domino_real_robot_init_tilt)
        cls.robot_init_wrist = float(CFG.domino_real_robot_init_wrist)

    @classmethod
    def initialize_pybullet(
            cls, using_gui: bool
    ) -> Tuple[int, SingleArmPyBulletRobot, Dict[str, Any]]:
        cls._apply_bench_geometry()
        physics_client_id, pybullet_robot, bodies = super(
        ).initialize_pybullet(using_gui)
        pcid = physics_client_id

        # -- bench: the same two tables as pybullet_domino_real ------------
        bodies["table_id"] = create_object(asset_path="urdf/table.urdf",
                                           position=cls.table_pos,
                                           orientation=cls.table_orn,
                                           scale=1.0,
                                           use_fixed_base=True,
                                           physics_client_id=pcid)
        bodies["table_id2"] = create_object(
            asset_path="urdf/table.urdf",
            position=(cls.table_pos[0], cls.table_pos[1] + cls.table_width / 2,
                      cls.table_pos[2]),
            orientation=cls.table_orn,
            scale=1.0,
            use_fixed_base=True,
            physics_client_id=pcid)
        if CFG.domino_real_decorate:
            cls._decorate(pcid, domino_world_z_offset(CFG.domino_real_table_z))

        # -- fan + button --------------------------------------------------
        bodies["fan_id"] = create_object(asset_path=cls.fan_urdf,
                                         scale=cls.fan_scale,
                                         use_fixed_base=True,
                                         physics_client_id=pcid)
        button_id = create_pybullet_block(
            color=cls.button_color_off,
            half_extents=cls.button_half_extents,
            mass=0.0,
            friction=0.5,
            position=(CFG.fan_real_button_x, CFG.fan_real_button_y,
                      cls.table_height + cls.button_half_extents[2]),
            physics_client_id=pcid)
        # Visual only: the press is detected geometrically, so the body must
        # not be able to block the very descent that presses it (a collision
        # at a move phase's own goal makes the phase never terminate).
        p.setCollisionFilterGroupMask(button_id,
                                      -1,
                                      collisionFilterGroup=0,
                                      collisionFilterMask=0,
                                      physicsClientId=pcid)
        bodies["button_id"] = button_id

        # -- ball ----------------------------------------------------------
        ball_id = create_pybullet_sphere(
            color=cls.ball_color,
            radius=CFG.fan_real_ball_radius,
            mass=CFG.fan_real_ball_mass,
            friction=CFG.fan_real_ball_lateral_friction,
            spinning_friction=CFG.fan_real_ball_rolling_friction,
            rolling_friction=CFG.fan_real_ball_rolling_friction,
            position=(CFG.fan_real_ball_start_x, CFG.fan_real_lane_y,
                      cls.table_height + CFG.fan_real_ball_radius),
            physics_client_id=pcid)
        p.changeDynamics(ball_id,
                         -1,
                         linearDamping=CFG.fan_real_ball_linear_damping,
                         angularDamping=CFG.fan_real_ball_angular_damping,
                         physicsClientId=pcid)
        bodies["ball_id"] = ball_id

        # -- zone pads (decoration only) -----------------------------------
        # Collision with the ball is filtered off rather than friction-matched.
        # A pad the ball physically rolls across is a second contact surface
        # whose friction silently rescales the duration->distance map (the
        # lesson pybullet_fan's target pad records the hard way); a pad that
        # cannot touch the ball has no such failure mode.
        zone_ids = []
        for i in range(CFG.fan_real_num_zones):
            zid = create_pybullet_block(
                color=cls.zone_color,
                half_extents=(CFG.fan_real_zone_len / 2 - 0.004,
                              CFG.fan_real_lane_half_width - 0.004,
                              cls.zone_thickness),
                mass=0.0,
                friction=0.5,
                position=(cls.zone_center_x(i + 1), CFG.fan_real_lane_y,
                          cls.table_height),
                physics_client_id=pcid)
            p.setCollisionFilterPair(zid,
                                     ball_id,
                                     -1,
                                     -1,
                                     0,
                                     physicsClientId=pcid)
            zone_ids.append(zid)
        bodies["zone_ids"] = zone_ids

        # -- lane rails + far end stop -------------------------------------
        rail_ids = []
        if CFG.fan_real_rails:
            lane_lo = CFG.fan_real_fan_x
            lane_hi = cls._lane_end_x()
            mid_x = (lane_lo + lane_hi) / 2
            half_len = (lane_hi - lane_lo) / 2
            rail_z = cls.table_height + cls.rail_height / 2
            for sign in (-1.0, 1.0):
                rail_ids.append(
                    create_pybullet_block(
                        color=cls.rail_color,
                        half_extents=(half_len, cls.rail_thickness / 2,
                                      cls.rail_height / 2),
                        mass=0.0,
                        friction=0.2,
                        position=(mid_x, CFG.fan_real_lane_y +
                                  sign * CFG.fan_real_lane_half_width, rail_z),
                        physics_client_id=pcid))
            # End stop, so an over-long burst parks the ball at the lane end
            # instead of rolling off the table into an unrecoverable state.
            rail_ids.append(
                create_pybullet_block(
                    color=cls.rail_color,
                    half_extents=(cls.rail_thickness / 2,
                                  CFG.fan_real_lane_half_width,
                                  cls.rail_height / 2),
                    mass=0.0,
                    friction=0.2,
                    position=(lane_hi, CFG.fan_real_lane_y, rail_z),
                    physics_client_id=pcid))
        bodies["rail_ids"] = rail_ids

        return physics_client_id, pybullet_robot, bodies

    @classmethod
    def _decorate(cls, pcid: int, z_off: float) -> None:
        """Extended-table tile + robot pedestal (ported from
        ``PyBulletDominoRealEnv._decorate``, which ported it from
        ``birrt._decorate_scene``)."""
        tile_id = create_object("urdf/table.urdf",
                                position=cls.tile_pos,
                                orientation=_yaw_quat(np.pi / 2),
                                scale=1.0,
                                use_fixed_base=True,
                                physics_client_id=pcid)
        tex_path = getattr(cls, "table_texture_path", None)
        if getattr(cls, "_use_studio_visuals", False) and tex_path and \
                isinstance(tile_id, int):
            texid = p.loadTexture(utils.get_env_asset_path(tex_path),
                                  physicsClientId=pcid)
            p.changeVisualShape(tile_id,
                                -1,
                                textureUniqueId=texid,
                                rgbaColor=(1, 1, 1, 1),
                                physicsClientId=pcid)
        riser_h = z_off - cls.table_height
        if riser_h > 1e-3:
            create_pybullet_block(color=(0.3, 0.3, 0.3, 1.0),
                                  half_extents=(0.10, 0.10, riser_h / 2),
                                  mass=0.0,
                                  friction=0.5,
                                  position=(cls.pedestal_xy[0],
                                            cls.pedestal_xy[1],
                                            cls.table_height + riser_h / 2),
                                  orientation=_yaw_quat(0.0),
                                  physics_client_id=pcid)

    def _store_pybullet_bodies(self, pybullet_bodies: Dict[str, Any]) -> None:
        self._table_ids = [
            pybullet_bodies["table_id"], pybullet_bodies["table_id2"]
        ]
        self._fan.id = pybullet_bodies["fan_id"]
        self._fan.joint_id = self._get_joint_id(self._fan.id, "joint_0")
        self._button.id = pybullet_bodies["button_id"]
        self._ball.id = pybullet_bodies["ball_id"]
        for i, zone in enumerate(self._zones):
            zone.id = pybullet_bodies["zone_ids"][i]
            zone.idx = float(i + 1)
        self._rail_ids: List[int] = pybullet_bodies["rail_ids"]

    def _get_joint_id(self, obj_id: int, joint_name: str) -> int:
        for j in range(
                p.getNumJoints(obj_id,
                               physicsClientId=self._physics_client_id)):
            info = p.getJointInfo(obj_id,
                                  j,
                                  physicsClientId=self._physics_client_id)
            if info[1].decode("utf-8") == joint_name:
                return j
        return -1

    def get_extra_collision_ids(self) -> Sequence[int]:
        """Rails and pads are obstacles for motion planning."""
        return list(self._rail_ids)

    # =========================================================================
    # STATE <-> PYBULLET
    # =========================================================================
    def _get_object_ids_for_held_check(self) -> List[int]:
        # Nothing in this env is picked up; the arm only presses the button.
        return []

    def _set_domain_specific_state(self, state: State) -> None:
        """Nothing to write.

        ``is_on`` is not stored anywhere -- it is a function of where
        the arm is, and ``_set_state`` has already restored the arm.
        That is the whole benefit of a momentary button: there is no
        latched bit that could disagree with the pose it is supposed to
        follow.
        """
        del state
        self._recolor_button()

    def _get_domain_specific_feature(self, obj: Object, feature: str) -> float:
        if feature == "is_on":
            # One button powers the one fan, and the button is held, not
            # latched, so both read the live gripper position.
            return float(self._button_is_pressed())
        if obj.type == self._zone_type and feature == "idx":
            return float(obj.idx)
        raise ValueError(f"Unknown feature {feature} for object {obj}")

    # =========================================================================
    # DYNAMICS
    # =========================================================================
    def _domain_specific_step(self) -> None:
        on = self._button_is_pressed()
        self._spin_fan(on)
        self._recolor_button()
        if on:
            self._apply_wind_impulse()

    def _spin_fan(self, on: bool) -> None:
        """Drive the fan blade, for looks only -- the blade never touches the
        ball; ``_apply_wind_impulse`` is the whole airflow model."""
        if self._fan.joint_id < 0:
            return
        p.setJointMotorControl2(
            bodyUniqueId=self._fan.id,
            jointIndex=self._fan.joint_id,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocity=self.fan_spin_velocity if on else 0.0,
            force=self.joint_motor_force,
            physicsClientId=self._physics_client_id)

    def _apply_wind_impulse(self) -> None:
        """Add one action-interval's worth of wind momentum to the ball.

        A velocity impulse rather than ``applyExternalForce`` because
        ``_domain_specific_step`` runs AFTER the action's whole batch of
        ``stepSimulation`` calls: PyBullet clears applied forces on every
        sub-step, so a force set here would survive exactly one of the
        twenty sub-steps in the next action -- 5% of the intended wind.
        Integrating the impulse explicitly (dv = F/m * dt_action) applies
        the full amount and leaves drag, friction and contacts to
        PyBullet. The cost is that the wind lands one action late, which
        is a constant lag on both the burst and its tail, so it does not
        bias the duration->distance map.
        """
        (bx, by, _), _ = p.getBasePositionAndOrientation(
            self._ball.id, physicsClientId=self._physics_client_id)
        downwind = bx - CFG.fan_real_fan_x
        # Outside the jet: behind the fan, or blown clear of the lane.
        if downwind <= 0.0 or \
                abs(by - CFG.fan_real_lane_y) > CFG.fan_real_lane_half_width:
            return
        force = CFG.fan_real_wind_force
        if CFG.fan_real_wind_falloff_dist is not None:
            force /= 1.0 + (downwind / CFG.fan_real_wind_falloff_dist)**2
        dv = force / CFG.fan_real_ball_mass * self._action_dt()
        lin, ang = p.getBaseVelocity(self._ball.id,
                                     physicsClientId=self._physics_client_id)
        p.resetBaseVelocity(self._ball.id,
                            linearVelocity=[lin[0] + dv, lin[1], lin[2]],
                            angularVelocity=list(ang),
                            physicsClientId=self._physics_client_id)

    # =========================================================================
    # BUTTON
    # =========================================================================
    @classmethod
    def button_press_point(cls) -> Pose3D:
        """Where the gripper must be to hold the button down."""
        return (CFG.fan_real_button_x, CFG.fan_real_button_y,
                cls.table_height + 2 * cls.button_half_extents[2] +
                cls.button_press_hover)

    def _button_is_pressed(self) -> bool:
        """Is the gripper on the button right now?

        Momentary, so this is the ONLY notion of the fan being on: the
        answer is recomputed from the live gripper position every time
        it is asked, and it goes false the instant the arm lifts away.
        """
        ee = self._pybullet_robot.get_state()[:3]
        press = self.button_press_point()
        dist_sq = sum((a - b)**2 for a, b in zip(ee, press))
        return bool(dist_sq <= CFG.fan_real_button_press_threshold**2)

    def _recolor_button(self) -> None:
        """Tint the button by whether it is currently held down."""
        color = (self.button_color_on
                 if self._button_is_pressed() else self.button_color_off)
        p.changeVisualShape(self._button.id,
                            -1,
                            rgbaColor=color,
                            physicsClientId=self._physics_client_id)

    # =========================================================================
    # PREDICATES
    # =========================================================================
    @staticmethod
    def _FanOn_holds(state: State, objects: Sequence[Object]) -> bool:
        (fan, ) = objects
        return state.get(fan, "is_on") > 0.5

    @staticmethod
    def _BallInZone_holds(state: State, objects: Sequence[Object]) -> bool:
        """The ball's center lies within the zone's footprint.

        Only the lane coordinate is a real test; the ball cannot leave
        the lane's y band, but checking it keeps the predicate honest
        about what the pad actually covers.
        """
        ball, zone = objects
        dx = abs(state.get(ball, "x") - state.get(zone, "x"))
        dy = abs(state.get(ball, "y") - state.get(zone, "y"))
        return bool(dx <= CFG.fan_real_zone_len / 2
                    and dy <= CFG.fan_real_lane_half_width)

    # =========================================================================
    # TASKS
    # =========================================================================
    def _generate_train_tasks(self) -> List[EnvironmentTask]:
        return self._make_tasks(CFG.num_train_tasks, self._train_rng)

    def _generate_test_tasks(self) -> List[EnvironmentTask]:
        return self._make_tasks(CFG.num_test_tasks, self._test_rng)

    def _init_dict(self) -> Dict[Object, Dict[str, float]]:
        """The one scene this bench has: everything at its bench pose, the fan
        off, and the ball parked at the lane's upwind end."""
        init: Dict[Object, Dict[str, float]] = {
            self._robot: {
                "x": self.robot_init_x,
                "y": self.robot_init_y,
                "z": self.robot_init_z,
                "fingers": self.open_fingers,
                "roll": self.robot_init_roll,
                "tilt": self.robot_init_tilt,
                "wrist": self.robot_init_wrist,
            },
            self._fan: {
                "x": CFG.fan_real_fan_x,
                "y": CFG.fan_real_lane_y,
                "z": self.table_height + self.fan_z_len / 2,
                "rot": 0.0,  # local +x is the airflow direction
                "is_on": 0.0,
            },
            self._button: {
                "x": CFG.fan_real_button_x,
                "y": CFG.fan_real_button_y,
                "z": self.table_height + self.button_half_extents[2],
                "is_on": 0.0,
            },
            self._ball: {
                "x": CFG.fan_real_ball_start_x,
                "y": CFG.fan_real_lane_y,
                "z": self.table_height + CFG.fan_real_ball_radius,
            },
        }
        for zone in self._zones:
            init[zone] = {
                "x": self.zone_center_x(int(zone.idx)),
                "y": CFG.fan_real_lane_y,
                "idx": float(zone.idx),
            }
        return init

    def _make_tasks(self, num_tasks: int,
                    rng: np.random.Generator) -> List[EnvironmentTask]:
        """One task per goal zone, cycling if more tasks are asked for.

        The scene never varies -- the bench has one ball at one start --
        so the only thing a task can choose is which zone the ball must
        end in. That is exactly the "one action, six possible targets"
        the bench is for.
        """
        init_state = utils.create_state_from_dict(self._init_dict())
        # Zone 1 is reachable by the shortest useful burst, so it is the
        # least informative goal; shuffle rather than always starting there.
        order = list(rng.permutation(len(self._zones)))
        tasks = []
        for i in range(num_tasks):
            zone = self._zones[order[i % len(order)]]
            goal = {GroundAtom(self._BallInZone, [self._ball, zone])}
            goal_nl = (
                f"Run the fan for the right length of time so the ball comes "
                f"to rest in {zone.name} (centered at "
                f"x={self.zone_center_x(int(zone.idx)):.2f}). A longer burst "
                f"pushes the ball farther.")
            tasks.append(
                EnvironmentTask(init_state.copy(), goal, goal_nl=goal_nl))
        return self._add_pybullet_state_to_tasks(tasks)
