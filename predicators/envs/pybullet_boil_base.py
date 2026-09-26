"""Observable simulation core of the boil environment.

This module is the boil env's BASE SIM: scene geometry and constants,
object and body construction (tables, jugs, burner plates, the faucet
and the switches), state read/write of everything visible, and the
switch mechanics - everything needed to run rigid-body rollouts of the
kitchen. It deliberately contains NO residual dynamics (what the faucet
and the burners do to the jugs lives in the ``PyBulletBoilEnv``
subclass), none of the constants of those laws, no task generation,
and no predicate / goal semantics. On the base sim a switch toggles,
its burner plate changes colour, and nothing else happens.

That boundary is a visibility contract, enforced structurally rather
than by redaction: when ``CFG.agent_sim_provide_base_sim_source`` is
on, THIS FILE is copied verbatim into the learning agent's sandbox as
reference material ("the robot knows its own simulator"), so the file
the agent reads is byte-identical to the code its base-sim rollouts
execute. Anything that would leak the learning target - the fill,
spill and heating laws and their rates, capacities and tolerances,
the task distribution, goal thresholds - must live in
``pybullet_boil.py`` (the concrete subclass), never here.

Physical layout (a tabletop, the robot at the near side):

- ``jug`` objects: pixel jugs the robot can pick up and place.
  ``water_volume`` is read back from the visual liquid body inside the
  jug (its height times ``water_height_to_level_ratio``); nothing in
  this file ever changes it.
- ``burner`` objects: flat plates on the table, each with a ``switch``
  in front of it. ``is_on`` mirrors that switch.
- The ``faucet``: a fixed faucet at the back right of the table, with
  its own ``switch``. ``is_on`` mirrors that switch.
- ``switch`` objects: toggle switches in a row at the front of the
  table; ``is_on`` is the switch joint past ``switch_on_threshold`` of
  its travel.
"""
import random
import re
from typing import Any, ClassVar, Dict, List, Optional, Set, Tuple

import numpy as np
import pybullet as p

from predicators.envs.pybullet_env import PyBulletEnv
from predicators.pybullet_helpers import retry_pybullet_call
from predicators.pybullet_helpers.geometry import Pose3D, Quaternion
from predicators.pybullet_helpers.objects import cap_switch_joint_travel, \
    create_object, create_pybullet_block, update_object
from predicators.pybullet_helpers.robots import SingleArmPyBulletRobot
from predicators.settings import CFG
from predicators.structs import Object, State, Type


class PyBulletBoilBaseEnv(PyBulletEnv):
    """Sim core of the boil kitchen: jugs, burner plates, a faucet, and one
    switch per burner and for the faucet.

    Abstract on purpose - it defines no name, predicates, tasks, or
    domain-specific step, so env discovery skips it; the concrete env
    is ``PyBulletBoilEnv``.
    """

    @classmethod
    def get_base_sim_source_files(cls) -> List[str]:
        # This module IS the visible sim core (see the module docstring's
        # visibility contract); pybullet_env.py is the generic engine it
        # is built on. pybullet_boil.py (residual dynamics, task
        # generation, predicates) must never be listed here.
        return [
            "predicators/envs/pybullet_boil_base.py",
            "predicators/envs/pybullet_env.py",
        ]

    # -------------------------------------------------------------------------
    # Table / workspace config
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # Robot config
    # -------------------------------------------------------------------------
    robot_init_x: ClassVar[float] = (x_lb + x_ub) * 0.5
    robot_init_y: ClassVar[float] = (y_lb + y_ub) * 0.5
    robot_init_z: ClassVar[float] = z_ub - 0.1
    robot_base_pos: ClassVar[Pose3D] = (0.75, 0.65, 0.0)
    robot_base_orn: ClassVar[Quaternion] = p.getQuaternionFromEuler(
        [0.0, 0.0, np.pi / 2])
    robot_init_tilt: ClassVar[float] = np.pi / 2
    robot_init_wrist: ClassVar[float] = -np.pi / 2

    # -------------------------------------------------------------------------
    # Camera
    # -------------------------------------------------------------------------
    _camera_distance: ClassVar[float] = 1.3
    _camera_yaw: ClassVar[float] = 60
    _camera_pitch: ClassVar[float] = -38
    _camera_target: ClassVar[Tuple[float, float, float]] = (0.75, 1.25, 0.42)

    # -------------------------------------------------------------------------
    # Scene geometry
    # -------------------------------------------------------------------------
    jug_height: ClassVar[float] = 0.12
    jug_handle_height: ClassVar[float] = jug_height * 3 / 4
    jug_handle_offset: ClassVar[float] = 0.08
    jug_init_z: ClassVar[float] = table_height + jug_height / 2
    small_gap: ClassVar[float] = 0.05
    burner_x_gap: ClassVar[float] = 3 * small_gap
    burner_y: ClassVar[float] = y_mid - small_gap * 1.1
    faucet_x: ClassVar[float] = x_mid + 6 * small_gap
    faucet_y: ClassVar[float] = y_mid + 5 * small_gap
    switch_y: ClassVar[float] = y_lb + small_gap

    # -------------------------------------------------------------------------
    # Liquid visualization
    # -------------------------------------------------------------------------
    # A jug's ``water_volume`` feature is the height of its liquid body
    # (a visual-only box inside the jug) times this ratio.
    water_height_to_level_ratio: ClassVar[float] = 10
    water_color = (0.0, 0.0, 1.0, 0.9)  # blue
    # Vertical offset of the jug's inner-bottom surface below jug.z.
    # The jug-pixel URDF places its base box at z=-0.25 local, so with
    # the default scale=0.2 the base bottom sits 0.06 m below the jug
    # origin and the inner-bottom surface (top of the 0.1 m base box)
    # sits 0.04 m below; add a small clearance so the liquid box
    # doesn't z-fight the base.
    _LIQUID_OFFSET_BELOW_JUG: ClassVar[float] = 0.04

    # -------------------------------------------------------------------------
    # Colors
    # -------------------------------------------------------------------------
    burner_switch_color: ClassVar[Tuple[float, float, float,
                                        float]] = (1.0, 0.5, 0.0, 1.0
                                                   )  # orange
    faucet_switch_color: ClassVar[Tuple[float, float, float,
                                        float]] = (0.0, 0.7, 1.0, 1.0
                                                   )  # light blue
    faucet_color: ClassVar[Tuple[float, float, float,
                                 float]] = (0.6, 0.6, 0.6, 1.0)  # gray
    # Burner plate colors
    burner_off_color: ClassVar[Tuple[float, float, float,
                                     float]] = (0.7, 0.7, 0.7, 1.0
                                                )  # gray (off)
    burner_on_color: ClassVar[Tuple[float, float, float,
                                    float]] = (1.0, 0.3, 0.0, 1.0
                                               )  # red-orange (on)

    # -------------------------------------------------------------------------
    # Switches
    # -------------------------------------------------------------------------
    switch_joint_scale: ClassVar[float] = 0.1
    switch_on_threshold: ClassVar[float] = 0.5  # fraction of the joint range
    switch_height: ClassVar[float] = 0.08

    # -------------------------------------------------------------------------
    # Types
    # -------------------------------------------------------------------------
    _robot_type = Type("robot",
                       ["x", "y", "z", "fingers", "roll", "tilt", "wrist"])

    # Two jug types: the fully-observable `_jug_type` carries
    # `heat_level` as an observable feature, while the partially-
    # observable `_jug_type_po` drops it entirely so the agent never
    # sees a feature named `heat_level`. `__init__` swaps
    # `self._jug_type` to the PO variant when `CFG.partially_observable`
    # is set. Both carry the observable `bubbling_level`.
    _jug_type = Type("jug", [
        "x", "y", "z", "rot", "is_held", "water_volume", "heat_level",
        "bubbling_level", "r", "g", "b"
    ],
                     sim_features=["id", "water_id"])
    _jug_type_po = Type("jug", [
        "x", "y", "z", "rot", "is_held", "water_volume", "bubbling_level", "r",
        "g", "b"
    ],
                        sim_features=["id", "water_id"])
    _burner_type = Type("burner", ["x", "y", "z", "is_on"])
    _switch_type = Type("switch", ["x", "y", "z", "rot", "is_on"])
    _faucet_type = Type("faucet",
                        ["x", "y", "z", "rot", "is_on", "spilled_level"])

    def __init__(self, use_gui: bool = False, **kwargs: Any) -> None:
        # In partial-observability mode, swap the jug type to the
        # variant without `heat_level` *before* any jugs/predicates are
        # built off `self._jug_type`, so the reduced type propagates to
        # the objects, every jug predicate, the `types` property, and
        # thus the agent-facing inspect tools.
        if CFG.partially_observable:
            self._jug_type = self._jug_type_po

        # Create the robot as an Object
        self._robot = Object("robot", self._robot_type)

        # Create jugs
        self._jugs: List[Object] = []
        max_jugs = max(max(CFG.boil_num_jugs_train),
                       max(CFG.boil_num_jugs_test))
        for i in range(max_jugs):
            jug_obj = Object(f"jug{i}", self._jug_type)
            self._jugs.append(jug_obj)
        self._jug_to_liquid_id: Dict[Object, Optional[int]] = {}

        # Create burners + a corresponding switch for each
        self._burners: List[Object] = []
        self._burner_switches: List[Object] = []
        max_burners = max(max(CFG.boil_num_burner_train),
                          max(CFG.boil_num_burner_test))
        for i in range(max_burners):
            burn_obj = Object(f"burner{i}", self._burner_type)
            self._burners.append(burn_obj)

            sw_obj = Object(self.switch_name_for(burn_obj), self._switch_type)
            self._burner_switches.append(sw_obj)

        # Create one faucet + a corresponding switch
        self._faucet = Object("faucet", self._faucet_type)
        self._faucet_switch = Object(self.switch_name_for(self._faucet),
                                     self._switch_type)

        super().__init__(use_gui, **kwargs)

    @property
    def types(self) -> Set[Type]:
        """The physical types of the kitchen."""
        return {
            self._robot_type, self._jug_type, self._burner_type,
            self._switch_type, self._faucet_type
        }

    # -------------------------------------------------------------------------
    # PyBullet Initialization
    # -------------------------------------------------------------------------
    @classmethod
    def initialize_pybullet(
            cls, using_gui: bool
    ) -> Tuple[int, SingleArmPyBulletRobot, Dict[str, Any]]:
        physics_client_id, pybullet_robot, bodies = super(
        ).initialize_pybullet(using_gui)

        # 1) Create a table
        table_id = create_object(
            asset_path="urdf/table.urdf",
            position=cls.table_pos,
            orientation=cls.table_orn,
            scale=1.0,
            use_fixed_base=True,
            physics_client_id=physics_client_id,
        )
        bodies["table_id"] = table_id
        # add another table for more space to place jugs and burners
        table_id2 = create_object(
            asset_path="urdf/table.urdf",
            position=(cls.table_pos[0],
                      cls.table_pos[1] + (cls.y_ub - cls.y_lb) / 2,
                      cls.table_pos[2]),
            orientation=cls.table_orn,
            scale=1.0,
            use_fixed_base=True,
            physics_client_id=physics_client_id)
        bodies["table_id2"] = table_id2

        # 2) Create jugs
        jug_ids = []
        max_jugs = max(max(CFG.boil_num_jugs_train),
                       max(CFG.boil_num_jugs_test))
        all_white_jugs = False
        for _ in range(max_jugs):
            # Example placeholder URDF for a jug
            jug_id = create_object(asset_path="urdf/jug-pixel.urdf",
                                   color=(1, 1, 1, 1) if all_white_jugs else
                                   random.choice(cls._obj_colors_main),
                                   use_fixed_base=False,
                                   physics_client_id=physics_client_id)
            jug_ids.append(jug_id)
        bodies["jug_ids"] = jug_ids

        # 3) Create burners
        burner_ids = []
        max_burners = max(max(CFG.boil_num_burner_train),
                          max(CFG.boil_num_burner_test))
        for _ in range(max_burners):
            burner_id = create_pybullet_block(
                color=cls.burner_off_color,
                half_extents=(0.07, 0.07, 0.0001),
                mass=0,
                friction=0.5,
                physics_client_id=physics_client_id)
            burner_ids.append(burner_id)
        bodies["burner_ids"] = burner_ids

        # 4) Create burner switches
        burner_switch_ids = []
        for _ in range(max_burners):
            switch_id = create_object(
                asset_path="urdf/partnet_mobility/switch/102812/switch.urdf",
                scale=1.0,
                use_fixed_base=True,
                physics_client_id=physics_client_id)
            # Color only the base (link -1), not the slider
            p.changeVisualShape(switch_id,
                                -1,
                                rgbaColor=cls.burner_switch_color,
                                physicsClientId=physics_client_id)
            cls._cap_switch_joint_travel(switch_id, physics_client_id)
            burner_switch_ids.append(switch_id)
        bodies["burner_switch_ids"] = burner_switch_ids

        # 5) Create faucet and faucet switch
        faucet_id = create_object(
            asset_path="urdf/partnet_mobility/faucet/1488/mobility.urdf",
            color=cls.faucet_color,
            use_fixed_base=True,
            physics_client_id=physics_client_id)
        bodies["faucet_id"] = faucet_id

        faucet_switch_id = create_object(
            asset_path="urdf/partnet_mobility/switch/102812/switch.urdf",
            scale=1.0,
            use_fixed_base=True,
            physics_client_id=physics_client_id)
        # Color only the base (link -1), not the slider
        p.changeVisualShape(faucet_switch_id,
                            -1,
                            rgbaColor=cls.faucet_switch_color,
                            physicsClientId=physics_client_id)
        cls._cap_switch_joint_travel(faucet_switch_id, physics_client_id)
        bodies["faucet_switch_id"] = faucet_switch_id

        return physics_client_id, pybullet_robot, bodies

    def _store_pybullet_bodies(self, pybullet_bodies: Dict[str, Any]) -> None:
        """Store references to all PyBullet IDs in the environment objects."""
        self._table_ids = [
            pybullet_bodies["table_id"], pybullet_bodies["table_id2"]
        ]
        self._robot.id = self._pybullet_robot.robot_id
        # Jugs
        for i, jug_obj in enumerate(self._jugs):
            jug_obj.id = pybullet_bodies["jug_ids"][i]

        # Burners
        for i, burner_obj in enumerate(self._burners):
            burner_obj.id = pybullet_bodies["burner_ids"][i]

        # Burner switches
        for i, sw_obj in enumerate(self._burner_switches):
            sw_obj.id = pybullet_bodies["burner_switch_ids"][i]

        # Faucet
        self._faucet.id = pybullet_bodies["faucet_id"]
        # Faucet switch
        self._faucet_switch.id = pybullet_bodies["faucet_switch_id"]

    # -------------------------------------------------------------------------
    # State Creation / Feature Extraction
    # -------------------------------------------------------------------------
    def _get_object_ids_for_held_check(self) -> List[int]:
        """Only jugs can be held in the robot's gripper here."""
        jug_ids = [j.id for j in self._jugs if j.id is not None]
        return jug_ids

    def _get_domain_specific_feature(self, obj: Object, feature: str) -> float:
        """Map from environment object + feature name -> a float feature in the
        State."""
        # Faucet
        if obj.type == self._faucet_type:
            if feature == "is_on":
                return float(self._is_switch_on(self._faucet_switch.id))

        # Burner
        elif obj.type == self._burner_type:
            if feature == "is_on":
                idx = int(obj.name.replace("burner", ""))
                sw_obj = self._burner_switches[idx]
                return float(self._is_switch_on(sw_obj.id))

        # Switch
        elif obj.type == self._switch_type:
            if feature == "is_on":
                return float(self._is_switch_on(obj.id))

        # Jug
        elif obj.type == self._jug_type:
            if feature == "water_volume":
                liquid_id = self._jug_to_liquid_id.get(obj, None)
                if liquid_id is not None:
                    shape_data = p.getVisualShapeData(
                        liquid_id, physicsClientId=self._physics_client_id)
                    if shape_data:  # handle the case shape_data might be empty
                        # shape_data[0][3] => half-extents, e.g.
                        # shape_data[0][3][2] is half in z
                        height = shape_data[0][3][2]
                        return height * self.water_height_to_level_ratio
                return 0.0

        # Otherwise, rely on defaults (like the base PyBulletEnv) for x,y,z,...
        raise ValueError(f"Unknown feature {feature} for object {obj}.")

    def _set_domain_specific_state(self, state: State) -> None:
        """Called in _set_state to do any environment-specific resetting.

        Sets every switch, rebuilds each jug's liquid body from its
        ``water_volume``, recolors the jug bodies and the burner plates,
        and moves the jugs and burners the state does not use out of
        view.
        """
        # Programmatically set burner switches on/off
        burners = state.get_objects(self._burner_type)
        for i, burner_obj in enumerate(burners):
            on_val = state.get(burner_obj, "is_on")
            self._set_switch_on(self._burner_switches[i].id,
                                bool(on_val > 0.5))

        # Remove existing jug liquid bodies if they exist
        for liquid_id in self._jug_to_liquid_id.values():
            if liquid_id is not None:
                p.removeBody(liquid_id,
                             physicsClientId=self._physics_client_id)
        self._jug_to_liquid_id.clear()

        # Recreate the liquid bodies as needed
        jugs = state.get_objects(self._jug_type)
        for jug in jugs:
            liquid_id = self._create_liquid_for_jug(jug, state)
            self._jug_to_liquid_id[jug] = liquid_id

        # Update jug body colors from state
        for jug in jugs:
            if jug.id is not None:
                r = state.get(jug, "r")
                g = state.get(jug, "g")
                b = state.get(jug, "b")
                update_object(jug.id,
                              color=(r, g, b, 1.0),
                              physics_client_id=self._physics_client_id)

        # Faucet on/off
        f_on = state.get(self._faucet, "is_on")
        self._set_switch_on(self._faucet_switch.id, bool(f_on > 0.5))

        # Move irrelevant jugs and burners out of the way
        oov_x, oov_y = self._out_of_view_xy
        for i in range(len(jugs), len(self._jugs)):
            update_object(self._jugs[i].id,
                          position=(oov_x, oov_y, 0.0),
                          physics_client_id=self._physics_client_id)
        for i in range(len(burners), len(self._burners)):
            update_object(self._burners[i].id,
                          position=(oov_x, oov_y, 0.0),
                          physics_client_id=self._physics_client_id)
            update_object(self._burner_switches[i].id,
                          position=(oov_x, oov_y, self.switch_height),
                          physics_client_id=self._physics_client_id)

        # Update burner colors to match their initial on/off state
        self._update_burner_colors(state)

    def _update_burner_colors(self, state: State) -> None:
        """Update burner plate colors based on their on/off state."""
        burners = state.get_objects(self._burner_type)
        for i, burner_obj in enumerate(burners):
            burner_id = burner_obj.id
            if burner_id is None:
                continue
            burner_on = self._is_switch_on(self._burner_switches[i].id)
            color = self.burner_on_color if burner_on else self.burner_off_color
            update_object(burner_id,
                          color=color,
                          physics_client_id=self._physics_client_id)

    # -------------------------------------------------------------------------
    # Liquid bodies
    # -------------------------------------------------------------------------
    def _liquid_pose_for_jug(
        self,
        jug_xy_z_rot: Tuple[float, float, float, float],
        water_volume: float,
    ) -> Tuple[float, float, float, Tuple[float, float, float, float]]:
        """Compute the liquid body's world pose given the jug's pose and
        current water_volume.

        Anchored to ``jug.z`` (not the table) so the liquid stays inside
        the jug when the jug is lifted.
        """
        jx, jy, jz, jrot = jug_xy_z_rot
        liquid_height = water_volume / self.water_height_to_level_ratio
        cz = jz - self._LIQUID_OFFSET_BELOW_JUG + liquid_height / 2
        orn = p.getQuaternionFromEuler([0.0, 0.0, jrot])
        return jx, jy, cz, orn

    def _create_liquid_for_jug(
        self,
        jug: Object,
        state: State,
    ) -> Optional[int]:
        """Given the jug's water_volume, create (or None) a small PyBullet body
        to represent the liquid."""
        current_liquid = state.get(jug, "water_volume")
        if current_liquid <= 0:
            return None

        liquid_height = current_liquid / self.water_height_to_level_ratio
        half_extents = (0.03, 0.03, liquid_height / 2)
        jug_xy_z_rot = (state.get(jug, "x"), state.get(jug, "y"),
                        state.get(jug, "z"), state.get(jug, "rot"))
        cx, cy, cz, orientation = self._liquid_pose_for_jug(
            jug_xy_z_rot, current_liquid)

        color = self.water_color
        liquid_id = create_pybullet_block(
            color=color,
            half_extents=half_extents,
            mass=0.01,
            friction=0.5,
            position=(cx, cy, cz),
            orientation=orientation,
            physics_client_id=self._physics_client_id)
        # The liquid block is purely a visualization of the water level.
        # Leaving its collision shape active causes the jug to drift
        # several cm when the body is recreated/repositioned inside the
        # jug (e.g. fill ticks during Wait). Disable collisions so only
        # the visual remains; physics-side it's a ghost.
        p.setCollisionFilterGroupMask(liquid_id,
                                      -1,
                                      collisionFilterGroup=0,
                                      collisionFilterMask=0,
                                      physicsClientId=self._physics_client_id)
        return liquid_id

    # -------------------------------------------------------------------------
    # Switch Helpers
    # -------------------------------------------------------------------------
    @staticmethod
    def switch_name_for(obj: Object) -> str:
        """Name of the switch that toggles ``obj``, a burner or the faucet:
        ``burner2`` -> ``burner_switch2``, ``faucet`` -> ``faucet_switch``.

        The single place the pairing convention lives: ``__init__``
        names the switches with it and :meth:`get_switch` resolves them
        by it.
        """
        match = re.fullmatch(r"([A-Za-z]+)(\d*)", obj.name)
        assert match is not None, f"unexpected object name {obj.name!r}"
        return f"{match.group(1)}_switch{match.group(2)}"

    @classmethod
    def get_switch(cls, state: State, obj: Object) -> Object:
        """The switch of ``obj`` (a burner or the faucet) in ``state``.

        Resolved by name against the state, never through an attribute
        of the caller's Object instance: a skill's objects come from
        whatever view the caller grounded on (the continual agent's
        observed frame, a state read back from a recording, another env
        instance's task), and those instances never carry this env's
        simulator attributes. Raises ``KeyError`` when the state has no
        such switch.
        """
        name = cls.switch_name_for(obj)
        for candidate in state:
            if candidate.name == name and \
                    candidate.type.name == cls._switch_type.name:
                return candidate
        raise KeyError(f"No switch {name!r} for {obj} in the state")

    def _is_switch_on(self, switch_id: int) -> bool:
        """Check if a switch's main joint is above a threshold."""
        if switch_id < 0:
            return False
        j_id = self._get_joint_id(switch_id, "joint_0",
                                  self._physics_client_id)
        if j_id < 0:
            return False
        j_pos, _, _, _ = retry_pybullet_call(
            p.getJointState,
            switch_id,
            j_id,
            physicsClientId=self._physics_client_id)
        info = retry_pybullet_call(p.getJointInfo,
                                   switch_id,
                                   j_id,
                                   physicsClientId=self._physics_client_id)
        j_min, j_max = info[8], info[9]
        frac = (j_pos / self.switch_joint_scale - j_min) / (j_max - j_min)
        return bool(frac > self.switch_on_threshold)

    def _set_switch_on(self, switch_id: int, power_on: bool) -> None:
        """Programmatically toggle the switch to on/off by resetting its joint
        state."""
        j_id = self._get_joint_id(switch_id, "joint_0",
                                  self._physics_client_id)
        if j_id < 0:
            return
        info = p.getJointInfo(switch_id,
                              j_id,
                              physicsClientId=self._physics_client_id)
        j_min, j_max = info[8], info[9]
        target_val = (j_max if power_on else j_min) * self.switch_joint_scale
        p.resetJointState(switch_id,
                          j_id,
                          target_val,
                          physicsClientId=self._physics_client_id)

    @staticmethod
    def _get_joint_id(obj_id: int,
                      joint_name: str,
                      physics_client_id: int = 0) -> int:
        """Helper to find a joint by name in a URDF."""
        num_joints = retry_pybullet_call(p.getNumJoints,
                                         obj_id,
                                         physicsClientId=physics_client_id)
        for j in range(num_joints):
            info = retry_pybullet_call(p.getJointInfo,
                                       obj_id,
                                       j,
                                       physicsClientId=physics_client_id)
            if info[1].decode("utf-8") == joint_name:
                return j
        return -1

    @classmethod
    def _cap_switch_joint_travel(cls, switch_id: int,
                                 physics_client_id: int) -> None:
        """Cap this env's switch so a push can't over-extend it past "on".

        Resolves ``joint_0`` and delegates to the shared
        :func:`cap_switch_joint_travel` (see its docstring for the why).
        """
        j_id = cls._get_joint_id(switch_id, "joint_0", physics_client_id)
        cap_switch_joint_travel(switch_id, j_id, cls.switch_joint_scale,
                                physics_client_id)
