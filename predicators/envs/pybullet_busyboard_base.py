"""Observable simulation core of the busyboard environment.

This module is the busyboard env's BASE SIM: scene geometry and
physical constants, body construction, state read/write, and the button
(toggle-switch) mechanics - everything needed to run rigid-body
rollouts of the board. It deliberately contains NO wiring, NO lamp
dynamics, no task generation, and no predicate / goal semantics.

That boundary is a visibility contract, enforced structurally rather
than by redaction: when ``CFG.agent_sim_provide_base_sim_source`` is
on, THIS FILE is copied verbatim into the learning agent's sandbox as
reference material ("the robot knows its own simulator"), so the file
the agent reads is byte-identical to the code its base-sim rollouts
execute. Anything that would leak the learning target - which button
drives which lamp, the charge accumulation law and its constants, the
task distribution, goal thresholds - must live in
``pybullet_busyboard.py`` (the concrete subclass), never here.

Physical layout (a children's busy board, one row of controls and one
row of indicators):

- A thin board resting on the table top, purely decorative. Every
  button and lamp stands on the board's upper face, ``board_top``, so
  the push skill's contact height follows from a button's own ``z``
  and needs no board-specific offset.
- A front row of ``button`` objects, each a PartNet-Mobility toggle
  switch (``switch/102812``), the same asset the boil / laser / fan
  envs use. A button latches: pushed from one side it reads on, from
  the other off. This is the ONLY thing the robot manipulates.
- A back row of ``lamp`` objects, each a small block whose colour is
  driven by a scalar ``brightness`` in [0, 1]. Nothing in this file
  ever sets a brightness; the subclass owns that.

Every button and every lamp has a fixed, distinct colour, carried in the
state as the ``color`` feature (an index into ``COLOR_PALETTE``, whose
names are the ones to use when talking about the board: "the red
button", "the yellow lamp"). A button's colour is a stable identity
across boards of every size - the red button on a five-button board is
the same control as the red button on a three-button one - so a rule
stated in terms of colours means the same thing on every board. A lamp's
body glows in its own colour as its brightness rises, from a dead grey
at zero to fully saturated at one.

The robot's whole action repertoire on this board is "push a button
one way or the other, or wait", which is deliberate: the difficulty is
meant to sit in working out what the board does, not in moving the
arm.
"""
from typing import Any, ClassVar, Dict, List, Tuple

import numpy as np
import pybullet as p

from predicators.envs.pybullet_env import PyBulletEnv
from predicators.pybullet_helpers.geometry import Pose3D, Quaternion
from predicators.pybullet_helpers.objects import cap_switch_joint_travel, \
    create_object, create_pybullet_block, update_object
from predicators.pybullet_helpers.robots import SingleArmPyBulletRobot
from predicators.settings import CFG
from predicators.structs import Object, State, Type


class PyBulletBusyBoardBaseEnv(PyBulletEnv):
    """Sim core of the busy board: a row of toggle buttons and a row of lamps.

    Abstract on purpose - it defines no name, predicates, tasks, or
    domain-specific step, so env discovery skips it; the concrete env is
    ``PyBulletBusyBoardEnv``.
    """

    @classmethod
    def get_base_sim_source_files(cls) -> List[str]:
        # This module IS the visible sim core (see the module docstring's
        # visibility contract); pybullet_env.py is the generic engine it
        # is built on. pybullet_busyboard.py (wiring, charge dynamics,
        # task generation, predicates) must never be listed here.
        return [
            "predicators/envs/pybullet_busyboard_base.py",
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
    y_lb: ClassVar[float] = 1.1
    y_ub: ClassVar[float] = 1.6
    z_lb: ClassVar[float] = table_height
    z_ub: ClassVar[float] = 0.75 + table_height / 2
    x_mid: ClassVar[float] = (x_lb + x_ub) / 2
    y_mid: ClassVar[float] = (y_lb + y_ub) / 2

    # =========================================================================
    # ROBOT
    # =========================================================================
    robot_init_x: ClassVar[float] = (x_lb + x_ub) * 0.5
    robot_init_y: ClassVar[float] = (y_lb + y_ub) * 0.5
    robot_init_z: ClassVar[float] = z_ub - 0.1
    robot_base_pos: ClassVar[Pose3D] = (0.75, 0.65, 0.0)
    robot_base_orn: ClassVar[Quaternion] = p.getQuaternionFromEuler(
        [0.0, 0.0, np.pi / 2])
    robot_init_tilt: ClassVar[float] = np.pi / 2
    robot_init_wrist: ClassVar[float] = -np.pi / 2

    # =========================================================================
    # CAMERA
    # =========================================================================
    # Framed on the board rather than the room: close enough that every
    # button's slider position and every lamp's colour is readable in a
    # still, at a three-quarter angle that keeps the two button rows from
    # hiding behind each other.
    _camera_distance: ClassVar[float] = 1.05
    _camera_yaw: ClassVar[float] = 68
    _camera_pitch: ClassVar[float] = -45
    _camera_target: ClassVar[Tuple[float, float, float]] = (0.75, 1.31, 0.42)

    # =========================================================================
    # BOARD LAYOUT
    # =========================================================================
    small_gap: ClassVar[float] = 0.05

    # Buttons: front row, within easy reach of the arm. rot = 0 matches the
    # boil env's switch convention, so the shared push skill's on-pose
    # (rot + pi/2) and off-pose (rot - pi/2) apply unchanged.
    button_y: ClassVar[float] = y_lb + small_gap
    button_rot: ClassVar[float] = 0.0
    # Wider than the switch body itself (0.235 across at ``button_scale``
    # 1.6), so neighbouring buttons read as separate controls rather than
    # merging into one slab.
    button_x_gap: ClassVar[float] = 0.26
    # At that width only three buttons fit across the arm's x reach, so
    # larger boards use a second row - which is what a real busy board
    # looks like anyway. The back row is comfortably reachable (measured:
    # the push skill operates a button at y up to 1.34 at every contact
    # height tried, more reliably than the front row).
    button_row_max: ClassVar[int] = 3
    button_row_y_gap: ClassVar[float] = 0.13
    # A push can shove a free prismatic slider past "on", after which the
    # reverse push can no longer drag it back across the threshold; the
    # travel cap (see cap_switch_joint_travel) removes that headroom.
    switch_joint_scale: ClassVar[float] = 0.1
    # Link index carrying the sliding part of the switch asset. Its base
    # body is on another link, and only the slider is left pale.
    _button_slider_link: ClassVar[int] = 2
    switch_on_threshold: ClassVar[float] = 0.5  # fraction of the joint range
    switch_height: ClassVar[float] = 0.08
    # Height above a button's base at which the slider actually sits. The
    # push skill aims here rather than at the base, so its contact-height
    # parameter has usable slack on both sides instead of only working at
    # the very top of its range.
    button_press_height: ClassVar[float] = 0.115
    button_scale: ClassVar[float] = 1.6

    # Lamps: back row, out of the arm's way. Purely indicators - the robot
    # never touches them, so they are zero-mass static blocks.
    lamp_y: ClassVar[float] = 1.47
    lamp_x_gap: ClassVar[float] = 0.26
    lamp_half_extents: ClassVar[Tuple[float, float,
                                      float]] = (0.035, 0.035, 0.03)
    # Each bulb stands on a base in the lamp's own colour, a third of the
    # bulb's height, so a dark lamp still shows which lamp it is. The bulb
    # itself only takes on the colour as it lights.
    lamp_base_half_extents: ClassVar[Tuple[float, float,
                                           float]] = (lamp_half_extents[0] +
                                                      0.012,
                                                      lamp_half_extents[1] +
                                                      0.012,
                                                      lamp_half_extents[2] / 3)

    # Decorative board resting ON the table (not sunk into it: two coincident
    # faces z-fight and the table shows through the board). Wide and deep
    # enough that the largest board - two rows of buttons and a row of lamps
    # - sits entirely on it rather than overhanging an edge.
    board_half_extents: ClassVar[Tuple[float, float,
                                       float]] = (0.42, 0.24, 0.004)
    board_top: ClassVar[float] = table_height + 2 * board_half_extents[2]
    # Height of a bulb's centre: on the board, on top of its base.
    lamp_z: ClassVar[float] = (board_top + 2 * lamp_base_half_extents[2] +
                               lamp_half_extents[2])
    board_color: ClassVar[Tuple[float, float, float,
                                float]] = (0.78, 0.62, 0.40, 1.0)
    # The slider stays pale against the coloured body so its latched
    # position is legible in a still: which side it sits on IS the button's
    # state.
    button_slider_color: ClassVar[Tuple[float, float, float,
                                        float]] = (0.92, 0.92, 0.94, 1.0)

    # The colour palette. The ``color`` feature of a button or lamp is an
    # index into this list; the name is how to refer to the object. Button
    # colours come first, lamp colours after, and the two sets are disjoint
    # so "the yellow one" never needs a type to be unambiguous. Object i of
    # a kind always gets colour i of its kind, on every board, which makes
    # colour a stable identity across board sizes.
    COLOR_PALETTE: ClassVar[List[Tuple[str,
                                       Tuple[float, float, float, float]]]] = [
                                           # Buttons.
                                           ("red", (0.86, 0.16, 0.14, 1.0)),
                                           ("green", (0.18, 0.62, 0.24, 1.0)),
                                           ("blue", (0.16, 0.34, 0.86, 1.0)),
                                           ("orange", (0.95, 0.55, 0.12, 1.0)),
                                           ("purple", (0.55, 0.22, 0.78, 1.0)),
                                           # Lamps.
                                           ("yellow", (1.0, 0.88, 0.15, 1.0)),
                                           ("cyan", (0.15, 0.90, 0.95, 1.0)),
                                           ("magenta", (0.95, 0.20, 0.80,
                                                        1.0)),
                                       ]
    _num_button_colors: ClassVar[int] = 5

    # Lamp colour ramp: brightness 0 is a dead grey bulb, brightness 1 the
    # lamp's own palette colour fully saturated. Intermediate values
    # interpolate, which is what makes a partially-charged lamp visibly
    # distinct from a dark one.
    lamp_dark_color: ClassVar[Tuple[float, float, float,
                                    float]] = (0.32, 0.32, 0.34, 1.0)

    # =========================================================================
    # TYPES
    # =========================================================================
    _robot_type = Type("robot",
                       ["x", "y", "z", "fingers", "roll", "tilt", "wrist"])
    # A button is the whole control: its own pose plus its latched state.
    # Unlike boil (where a burner owns a separate switch body), there is no
    # indirection here - the thing the robot pushes is the thing the model
    # reasons about.
    _button_type = Type("button", ["x", "y", "z", "rot", "color", "is_on"],
                        sim_features=["id", "joint_id"])

    # =========================================================================
    # COLOURS
    # =========================================================================
    @classmethod
    def button_color_index(cls, button_idx: int) -> int:
        """Palette index of the i-th button (the ``color`` feature)."""
        if button_idx >= cls._num_button_colors:
            raise ValueError(f"Only {cls._num_button_colors} button colours "
                             f"are defined; button{button_idx} has none.")
        return button_idx

    @classmethod
    def lamp_color_index(cls, lamp_idx: int) -> int:
        """Palette index of the i-th lamp (the ``color`` feature)."""
        index = cls._num_button_colors + lamp_idx
        if index >= len(cls.COLOR_PALETTE):
            raise ValueError(
                f"Only {len(cls.COLOR_PALETTE) - cls._num_button_colors} "
                f"lamp colours are defined; lamp{lamp_idx} has "
                f"none.")
        return index

    @classmethod
    def color_name(cls, color_index: int) -> str:
        """The palette name behind a ``color`` feature value."""
        return cls.COLOR_PALETTE[int(round(color_index))][0]

    @classmethod
    def color_rgba(cls, color_index: int) -> Tuple[float, float, float, float]:
        """The RGBA behind a ``color`` feature value."""
        return cls.COLOR_PALETTE[int(round(color_index))][1]

    # =========================================================================
    # CAPACITY
    # =========================================================================
    # PyBullet bodies are built once in initialize_pybullet (a classmethod,
    # before any task exists), so the board is built at its maximum size and
    # unused bodies are parked out of view per task. Reading CFG here keeps
    # the body count in step with the task distribution.
    @classmethod
    def _max_buttons(cls) -> int:
        return max(
            list(CFG.busyboard_num_buttons_train) +
            list(CFG.busyboard_num_buttons_test))

    @classmethod
    def _max_lamps(cls) -> int:
        return max(
            list(CFG.busyboard_num_lamps_train) +
            list(CFG.busyboard_num_lamps_test))

    # =========================================================================
    # CONSTRUCTION
    # =========================================================================
    def __init__(self, use_gui: bool = False, **kwargs: Any) -> None:
        self._robot = Object("robot", self._robot_type)
        self._buttons: List[Object] = [
            Object(f"button{i}", self._button_type)
            for i in range(self._max_buttons())
        ]
        self._lamps: List[Object] = [
            Object(f"lamp{i}", self._lamp_type_for_run())
            for i in range(self._max_lamps())
        ]
        super().__init__(use_gui, **kwargs)

    @classmethod
    def _lamp_type_for_run(cls) -> Type:
        """The lamp type in force for this run.

        Overridden by the concrete env, which owns the observability
        decision (whether the hidden charge is a visible feature).
        """
        raise NotImplementedError("Override me!")

    @classmethod
    def initialize_pybullet(
            cls, using_gui: bool
    ) -> Tuple[int, SingleArmPyBulletRobot, Dict[str, Any]]:
        physics_client_id, pybullet_robot, bodies = super(
        ).initialize_pybullet(using_gui)

        table_id = create_object(asset_path="urdf/table.urdf",
                                 position=cls.table_pos,
                                 orientation=cls.table_orn,
                                 scale=1.0,
                                 use_fixed_base=True,
                                 physics_client_id=physics_client_id)
        bodies["table_id"] = table_id

        # Decorative board, resting on the table top.
        board_id = create_pybullet_block(
            color=cls.board_color,
            half_extents=cls.board_half_extents,
            mass=0.0,
            friction=0.5,
            position=(cls.x_mid, (cls.button_y + cls.lamp_y) / 2,
                      cls.table_height + cls.board_half_extents[2]),
            physics_client_id=physics_client_id)
        bodies["board_id"] = board_id

        button_ids = []
        for button_idx in range(cls._max_buttons()):
            button_id = create_object(
                asset_path="urdf/partnet_mobility/switch/102812/switch.urdf",
                scale=cls.button_scale,
                use_fixed_base=True,
                physics_client_id=physics_client_id)
            # Colour per link, walking the visual shape table rather than
            # assuming indices: the asset carries its body and its slider on
            # different links, and painting only link -1 (as this did before)
            # left the whole visible switch the asset's default white.
            body_color = cls.color_rgba(cls.button_color_index(button_idx))
            for shape in p.getVisualShapeData(
                    button_id, physicsClientId=physics_client_id):
                link_idx = shape[1]
                color = (cls.button_slider_color if link_idx
                         == cls._button_slider_link else body_color)
                p.changeVisualShape(button_id,
                                    link_idx,
                                    rgbaColor=color,
                                    physicsClientId=physics_client_id)
            cls._cap_switch_joint_travel(button_id, physics_client_id)
            button_ids.append(button_id)
        bodies["button_ids"] = button_ids

        lamp_ids = []
        for _ in range(cls._max_lamps()):
            lamp_id = create_pybullet_block(
                color=cls.lamp_dark_color,
                half_extents=cls.lamp_half_extents,
                mass=0.0,
                friction=0.5,
                physics_client_id=physics_client_id)
            lamp_ids.append(lamp_id)
        bodies["lamp_ids"] = lamp_ids

        base_ids = []
        for lamp_idx in range(cls._max_lamps()):
            base_id = create_pybullet_block(
                color=cls.color_rgba(cls.lamp_color_index(lamp_idx)),
                half_extents=cls.lamp_base_half_extents,
                mass=0.0,
                friction=0.5,
                physics_client_id=physics_client_id)
            base_ids.append(base_id)
        bodies["lamp_base_ids"] = base_ids

        return physics_client_id, pybullet_robot, bodies

    def _store_pybullet_bodies(self, pybullet_bodies: Dict[str, Any]) -> None:
        self._table_id = pybullet_bodies["table_id"]
        self._board_id = pybullet_bodies["board_id"]
        self._robot.id = self._pybullet_robot.robot_id
        for i, button in enumerate(self._buttons):
            button.id = pybullet_bodies["button_ids"][i]
            button.joint_id = self._get_joint_id(button.id, "joint_0",
                                                 self._physics_client_id)
        for i, lamp in enumerate(self._lamps):
            lamp.id = pybullet_bodies["lamp_ids"][i]
        self._lamp_base_ids: List[int] = pybullet_bodies["lamp_base_ids"]

    # =========================================================================
    # BUTTON MECHANICS
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

    @classmethod
    def _cap_switch_joint_travel(cls, button_id: int,
                                 physics_client_id: int) -> None:
        """Cap a button so a push cannot over-extend it past "on"."""
        j_id = cls._get_joint_id(button_id, "joint_0", physics_client_id)
        cap_switch_joint_travel(button_id, j_id, cls.switch_joint_scale,
                                physics_client_id)

    def _is_button_on(self, button: Object) -> bool:
        """Read a button's latched state from its prismatic joint."""
        if button.id is None or button.joint_id is None or button.joint_id < 0:
            return False
        j_pos = p.getJointState(button.id,
                                button.joint_id,
                                physicsClientId=self._physics_client_id)[0]
        info = p.getJointInfo(button.id,
                              button.joint_id,
                              physicsClientId=self._physics_client_id)
        j_min, j_max = info[8], info[9]
        frac = (j_pos / self.switch_joint_scale - j_min) / (j_max - j_min)
        return bool(frac > self.switch_on_threshold)

    def _set_button_on(self, button: Object, is_on: bool) -> None:
        """Programmatically latch a button on or off."""
        if button.joint_id is None or button.joint_id < 0:
            return
        info = p.getJointInfo(button.id,
                              button.joint_id,
                              physicsClientId=self._physics_client_id)
        j_min, j_max = info[8], info[9]
        target = (j_max if is_on else j_min) * self.switch_joint_scale
        p.resetJointState(button.id,
                          button.joint_id,
                          target,
                          physicsClientId=self._physics_client_id)

    # =========================================================================
    # LAMP RENDERING
    # =========================================================================
    def _set_lamp_brightness_visual(self, lamp: Object,
                                    brightness: float) -> None:
        """Interpolate the lamp block's colour between dark and its own."""
        if lamp.id is None:
            return
        t = float(np.clip(brightness, 0.0, 1.0))
        lit_color = self.color_rgba(
            self.lamp_color_index(self._lamps.index(lamp)))
        color = tuple((1.0 - t) * float(d) + t * float(l)
                      for d, l in zip(self.lamp_dark_color, lit_color))
        p.changeVisualShape(lamp.id,
                            -1,
                            rgbaColor=color,
                            physicsClientId=self._physics_client_id)

    # =========================================================================
    # POSE HELPERS
    # =========================================================================
    @classmethod
    def _row_xs(cls, count: int, gap: float) -> List[float]:
        """Evenly spaced x positions for a centred row of ``count`` items."""
        span = gap * (count - 1)
        return [cls.x_mid - span / 2 + i * gap for i in range(count)]

    @classmethod
    def button_layout(cls, count: int) -> List[Tuple[float, float]]:
        """(x, y) for each button, front row first, left to right.

        Splits into at most ``button_row_max`` per row and balances the
        rows, giving the extra button to the front one, so a five-button
        board is a row of three in front of a row of two rather than a
        lopsided four-and-one.
        """
        num_rows = max(1, -(-count // cls.button_row_max))  # ceil division
        base, extra = divmod(count, num_rows)
        sizes = [base + (1 if r < extra else 0) for r in range(num_rows)]
        placements: List[Tuple[float, float]] = []
        for row, size in enumerate(sizes):
            y = cls.button_y + row * cls.button_row_y_gap
            placements.extend(
                (x, y) for x in cls._row_xs(size, cls.button_x_gap))
        return placements

    # =========================================================================
    # STATE READ / WRITE
    # =========================================================================
    def _get_object_ids_for_held_check(self) -> List[int]:
        """Nothing on this board is graspable; buttons are pushed, not held."""
        return []

    def _park_unused_bodies(self, num_buttons: int, num_lamps: int) -> None:
        """Move bodies beyond the task's counts out of the camera's view."""
        oov_x, oov_y = self._out_of_view_xy
        for i in range(num_buttons, len(self._buttons)):
            update_object(self._buttons[i].id,
                          position=(oov_x, oov_y, self.switch_height),
                          physics_client_id=self._physics_client_id)
        for i in range(num_lamps, len(self._lamps)):
            update_object(self._lamps[i].id,
                          position=(oov_x, oov_y, 0.0),
                          physics_client_id=self._physics_client_id)
            update_object(self._lamp_base_ids[i],
                          position=(oov_x, oov_y, 0.0),
                          physics_client_id=self._physics_client_id)

    def _seat_lamp_bases(self, state: State, lamps: List[Object]) -> None:
        """Put each active lamp's base on the board, under its bulb."""
        for lamp in lamps:
            base_id = self._lamp_base_ids[self._lamps.index(lamp)]
            update_object(
                base_id,
                position=(state.get(lamp, "x"), state.get(lamp, "y"),
                          self.board_top + self.lamp_base_half_extents[2]),
                physics_client_id=self._physics_client_id)

    def _active_objects(self,
                        state: State) -> Tuple[List[Object], List[Object]]:
        """The buttons and lamps present in ``state``, in board order.

        Ordered by name (``button0``, ``button1``, ...) so index-based
        wiring in the subclass lines up with the board's left-to-right
        layout regardless of the state's dict ordering.
        """
        buttons = sorted((o for o in state if o.type.name == "button"),
                         key=lambda o: o.name)
        lamps = sorted((o for o in state if o.type.name == "lamp"),
                       key=lambda o: o.name)
        return buttons, lamps
