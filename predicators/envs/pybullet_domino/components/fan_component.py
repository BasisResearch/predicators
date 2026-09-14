"""Fan component for the domino environment.

This component handles:
- Fan arrays on each side of the workspace
- Switches that control fans
- Side objects (left, right, back, front)
- Wind physics simulation
- Related predicates (FanOn, FanOff, Controls, FanFacingSide)
"""

from typing import Any, ClassVar, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pybullet as p

from predicators.envs.pybullet_domino.components.base_component import \
    DominoEnvComponent
from predicators.pybullet_helpers.objects import create_object, update_object
from predicators.settings import CFG
from predicators.structs import Object, Predicate, State, Type


class FanComponent(DominoEnvComponent):
    """Component for fans, switches, and wind physics.

    Manages:
    - Fan arrays positioned on each side of the workspace
    - Switches that control fan activation
    - Wind force simulation applied to objects
    """

    # =========================================================================
    # FAN CONFIGURATION
    # =========================================================================

    # Fan counts per side
    # Fans per side. Banks of five look like a wall of fans and blow as
    # one: only fan_ids[0] is ever read for direction or wind. A single-
    # fan env overrides these to 1 (see __init__'s fans_per_side).
    num_left_fans: int = 5
    num_right_fans: int = 5
    num_back_fans: int = 5
    num_front_fans: int = 5

    # Fan physical properties
    fan_scale: ClassVar[float] = 0.08
    fan_x_len: ClassVar[float] = 0.2 * fan_scale
    fan_y_len: ClassVar[float] = 1.5 * fan_scale
    fan_z_len: ClassVar[float] = 1.5 * fan_scale

    # Fan motor & physics
    fan_spin_velocity: ClassVar[float] = 100.0
    wind_force_magnitude: ClassVar[float] = 2.0
    joint_motor_force: ClassVar[float] = 20.0

    # =========================================================================
    # SWITCH CONFIGURATION
    # =========================================================================
    switch_scale: ClassVar[float] = 1.0
    switch_joint_scale: ClassVar[float] = 0.1
    switch_on_threshold: ClassVar[float] = 0.5
    switch_x_len: ClassVar[float] = 0.10
    switch_height: ClassVar[float] = 0.08

    def __init__(self,
                 workspace_bounds: Optional[Dict[str, float]] = None,
                 table_height: float = 0.4,
                 table_width: float = 1.0,
                 num_sides: int = 4,
                 fans_per_side: Optional[int] = None,
                 switch_xy: Optional[Tuple[float, float]] = None,
                 switch_reachable: bool = True) -> None:
        """Initialize the fan component.

        Args:
            workspace_bounds: Dictionary with x_lb, x_ub, y_lb, y_ub.
            table_height: Height of the table surface.
            table_width: Width of the table.
            num_sides: How many sides carry a fan and its switch, taken
                in order left, right, down, up. Four is the ball task's
                layout, where the ball must be blown any of four ways
                across a grid. A domino chain runs ONE way, so the other
                three fans are distractors the planner still has to
                ground - and, blowing inward from opposite sides, they
                cancel each other exactly.
            fans_per_side: Fan bodies in each side's bank (default keeps
                the class values). They blow as one; only fan_ids[0] is
                read for direction or wind, so a bank is decoration.
            switch_reachable: When False, the switch bodies are parked
                far outside the workspace. The switch stays the thing
                that STORES whether the fan is on -- extract_feature
                reads the fan's is_on off its joint -- but nothing can
                reach it, so the only way the fan comes on is whatever
                the env decides to latch it with (see
                PyBulletDominoDeclareEnv, where a declaration does).
            switch_xy: Where the first switch sits, overriding the
                workspace-centre formula below. That formula assumes the
                fan env's workspace, which is twice as deep with the
                robot at its front edge; dropped into a shallower one
                with a centred robot it puts the switch BEHIND the arm,
                0.38 m from the base against the 0.73 m the fan env
                gives it, and the press never completes (IK solves, the
                joint-limited arm lands centimetres short, and the skill
                waits forever for an exact arrival).
        """
        super().__init__()
        assert 1 <= num_sides <= 4, num_sides
        self.num_sides = num_sides
        self.switch_reachable = switch_reachable
        if fans_per_side is not None:
            self.num_left_fans = fans_per_side
            self.num_right_fans = fans_per_side
            self.num_back_fans = fans_per_side
            self.num_front_fans = fans_per_side

        # Store table parameters
        self.table_height = table_height
        self.table_width = table_width

        # Workspace bounds (defaults from PyBulletDominoEnv)
        if workspace_bounds is None:
            workspace_bounds = {
                "x_lb": 0.4,
                "x_ub": 1.1,
                "y_lb": 1.1,
                "y_ub": 1.6
            }
        self.x_lb = workspace_bounds["x_lb"]
        self.x_ub = workspace_bounds["x_ub"]
        self.y_lb = workspace_bounds["y_lb"]
        self.y_ub = workspace_bounds["y_ub"]

        # Calculate fan positioning based on workspace
        self.left_fan_x = self.x_lb - self.fan_x_len * 5
        self.right_fan_x = self.x_ub + self.fan_x_len * 5
        self.up_fan_y = self.y_ub + self.table_width / 2 + self.fan_x_len / 2
        self.down_fan_y = self.y_lb + self.fan_x_len / 2 + 0.1

        # Fan placement boundaries
        self.fan_y_lb = (self.down_fan_y + self.fan_x_len / 2 +
                         self.fan_y_len / 2 + 0.01)
        self.fan_y_ub = (self.up_fan_y - self.fan_x_len / 2 -
                         self.fan_y_len / 2 - 0.01)
        self.fan_x_lb = (self.left_fan_x + self.fan_x_len / 2 +
                         self.fan_y_len / 2 + 0.01)
        self.fan_x_ub = (self.right_fan_x - self.fan_x_len / 2 -
                         self.fan_y_len / 2 - 0.01)

        # Switch positioning
        if switch_xy is None:
            self.switch_base_x = 0.60
            self.switch_y = (self.y_lb + self.y_ub) * 0.5 - 0.25
        else:
            self.switch_base_x, self.switch_y = switch_xy
        self.switch_x_spacing = 0.08
        if not switch_reachable:
            # Parked well outside the arm's reach and off the table.
            # The body still exists and still stores the on/off bit,
            # but no skill and no stray sweep of the arm can touch it,
            # so the fan's state has exactly one cause: whatever the
            # env latches it with.
            self.switch_base_x = self.x_lb - 2.0
            self.switch_y = self.y_lb - 2.0

        # Side names. A component built with fewer sides keeps the
        # first N of these: side 0 (left) blows +x, and the domino-fan
        # env's chains are laid along it.
        self._switch_sides = ["left", "right", "down", "up"][:self.num_sides]

        # Create types
        self._fan_type = Type(
            "fan", ["x", "y", "z", "rot", "facing_side", "is_on"],
            sim_features=["id", "side_idx", "fan_ids", "joint_ids"])
        self._switch_type = Type(
            "switch", ["x", "y", "z", "rot", "controls_fan", "is_on"],
            sim_features=["id", "joint_id", "side_idx"])
        self._side_type = Type("side", ["side_idx"],
                               sim_features=["id", "side_idx"])

        # Create objects
        self._fans: List[Object] = []
        for i in range(self.num_sides):
            fan_obj = Object(f"fan_{i}", self._fan_type)
            self._fans.append(fan_obj)

        self._switches: List[Object] = []
        for i in range(self.num_sides):
            switch_obj = Object(f"switch_{i}", self._switch_type)
            self._switches.append(switch_obj)

        self._sides: List[Object] = []
        for side_str in self._switch_sides:
            side_obj = Object(side_str, self._side_type)
            self._sides.append(side_obj)

        # Create predicates
        self._FanOn = Predicate("FanOn", [self._fan_type], self._FanOn_holds)
        self._FanOff = Predicate("FanOff", [self._fan_type],
                                 lambda s, o: not self._FanOn_holds(s, o))
        self._SwitchOn = Predicate("SwitchOn", [self._switch_type],
                                   self._FanOn_holds)
        self._SwitchOff = Predicate("SwitchOff", [self._switch_type],
                                    lambda s, o: not self._FanOn_holds(s, o))
        self._FanFacingSide = Predicate("FanFacingSide",
                                        [self._fan_type, self._side_type],
                                        self._FanFacingSide_holds)
        self._Controls = Predicate("Controls",
                                   [self._switch_type, self._fan_type],
                                   self._Controls_holds)

        # Object to apply wind force to (set by composed environment)
        self._wind_target_id: Optional[int] = None
        # Height above the target's origin at which the wind pushes. Zero
        # for a ball, where a force through the centre is what rolls it.
        # A domino needs a positive value or it never tips: see
        # _apply_wind_force.
        self._wind_target_z_offset: float = 0.0
        # Stop pushing the target once it has fallen over (see
        # _target_still_standing). Only meaningful for a body that can
        # fall out of the airstream, so it travels with the z offset.
        self._wind_stops_when_toppled: bool = False
        # Per-target force override (N). None keeps the class default,
        # which is calibrated for the ball.
        self._wind_force_override: Optional[float] = None
        # Lateral position the single fan is aimed at, along the axis it
        # does NOT blow down: y for a left/right fan, x for front/back.
        # None keeps the rail's centre. See set_lateral_alignment.
        self._lateral_alignment: Optional[float] = None

    # -------------------------------------------------------------------------
    # DominoEnvComponent interface implementation
    # -------------------------------------------------------------------------

    def get_types(self) -> Set[Type]:
        return {self._fan_type, self._switch_type, self._side_type}

    def get_predicates(self) -> Set[Predicate]:
        preds = {
            self._FanOn,
            self._FanOff,
            self._FanFacingSide,
            self._Controls,
        }
        if not CFG.fan_known_controls_relation:
            preds |= {self._SwitchOn, self._SwitchOff}
        return preds

    def get_goal_predicates(self) -> Set[Predicate]:
        # Fans don't directly contribute to goals
        return set()

    def get_objects(self) -> List[Object]:
        return self._fans + self._switches + self._sides

    def initialize_pybullet(self, physics_client_id: int) -> Dict[str, Any]:
        """Create PyBullet bodies for fans and switches."""
        self._physics_client_id = physics_client_id
        bodies: Dict[str, Any] = {}

        fan_urdf = "urdf/partnet_mobility/fan/101450/mobility.urdf"

        # Create fan arrays for each side
        left_fan_ids = []
        for _ in range(self.num_left_fans):
            fid = create_object(asset_path=fan_urdf,
                                scale=self.fan_scale,
                                use_fixed_base=True,
                                physics_client_id=physics_client_id)
            left_fan_ids.append(fid)

        right_fan_ids = []
        for _ in range(self.num_right_fans):
            fid = create_object(asset_path=fan_urdf,
                                scale=self.fan_scale,
                                use_fixed_base=True,
                                physics_client_id=physics_client_id)
            right_fan_ids.append(fid)

        back_fan_ids = []
        for _ in range(self.num_back_fans):
            fid = create_object(asset_path=fan_urdf,
                                scale=self.fan_scale,
                                use_fixed_base=True,
                                physics_client_id=physics_client_id)
            back_fan_ids.append(fid)

        front_fan_ids = []
        for _ in range(self.num_front_fans):
            fid = create_object(asset_path=fan_urdf,
                                scale=self.fan_scale,
                                use_fixed_base=True,
                                physics_client_id=physics_client_id)
            front_fan_ids.append(fid)

        bodies["fan_ids_left"] = left_fan_ids
        bodies["fan_ids_right"] = right_fan_ids
        bodies["fan_ids_back"] = back_fan_ids
        bodies["fan_ids_front"] = front_fan_ids

        # Create switches
        switch_urdf = "urdf/partnet_mobility/switch/102812/switch.urdf"
        switch_ids = []
        for _ in range(4):
            sid = create_object(asset_path=switch_urdf,
                                scale=self.switch_scale,
                                use_fixed_base=True,
                                physics_client_id=physics_client_id)
            switch_ids.append(sid)
        bodies["switch_ids"] = switch_ids

        return bodies

    def store_pybullet_bodies(self, pybullet_bodies: Dict[str, Any]) -> None:
        """Store PyBullet body IDs on fan and switch objects."""
        fan_ids_by_side = [
            pybullet_bodies["fan_ids_left"], pybullet_bodies["fan_ids_right"],
            pybullet_bodies["fan_ids_back"], pybullet_bodies["fan_ids_front"]
        ]

        for side_idx, fan_obj in enumerate(self._fans):
            fan_obj.side_idx = side_idx
            fan_obj.fan_ids = fan_ids_by_side[side_idx]
            fan_obj.joint_ids = [
                self._get_joint_id(fid, "joint_0") for fid in fan_obj.fan_ids
            ]
            fan_obj.id = fan_obj.fan_ids[0] if fan_obj.fan_ids else -1

        for i, switch_obj in enumerate(self._switches):
            switch_obj.id = pybullet_bodies["switch_ids"][i]
            switch_obj.joint_id = self._get_joint_id(switch_obj.id, "joint_0")
            switch_obj.side_idx = i

        for i, side_obj in enumerate(self._sides):
            side_obj.side_idx = float(i)

    def reset_state(self, state: State) -> None:
        """Reset fans and switches to match state."""
        # Set switch states
        for switch_obj in self._switches:
            is_on_val = state.get(switch_obj, "is_on")
            self._set_switch_on(switch_obj.id, bool(is_on_val > 0.5))

        # Position fans on sides
        self._position_fans_on_sides()

    def extract_feature(self, obj: Object, feature: str) -> Optional[float]:
        """Extract feature for fan-related objects."""
        if obj.type == self._fan_type:
            if feature == "facing_side":
                return float(obj.side_idx)
            if feature == "is_on":
                controlling_switch = self._switches[obj.side_idx]
                return float(self._is_switch_on(controlling_switch.id))
        if obj.type == self._switch_type:
            if feature == "controls_fan":
                return float(obj.side_idx)
            if feature == "is_on":
                return float(self._is_switch_on(obj.id))
        if obj.type == self._side_type:
            if feature == "side_idx":
                return float(obj.side_idx)
        return None

    def step(self) -> None:
        """Simulate fans: spin blades and apply wind forces."""
        for ctrl_fan_idx, switch_obj in enumerate(self._switches):
            on = self._is_switch_on(switch_obj.id)
            fan_obj = self._fans[ctrl_fan_idx]

            if not hasattr(fan_obj, 'fan_ids') or not fan_obj.fan_ids:
                continue

            if on:
                # Spin fan visuals
                for i, fan_id in enumerate(fan_obj.fan_ids):
                    joint_id = fan_obj.joint_ids[i]
                    if joint_id >= 0:
                        p.setJointMotorControl2(
                            bodyUniqueId=fan_id,
                            jointIndex=joint_id,
                            controlMode=p.VELOCITY_CONTROL,
                            targetVelocity=self.fan_spin_velocity,
                            force=self.joint_motor_force,
                            physicsClientId=self._physics_client_id,
                        )
                # Apply force to wind target (e.g., ball)
                if self._wind_target_id is not None and (
                        not self._wind_stops_when_toppled
                        or self._target_still_standing(self._wind_target_id)):
                    self._apply_wind_force(fan_obj.fan_ids[0],
                                           self._wind_target_id)
            else:
                # Turn off fans
                for i, fan_id in enumerate(fan_obj.fan_ids):
                    joint_id = fan_obj.joint_ids[i]
                    if joint_id >= 0:
                        p.setJointMotorControl2(
                            bodyUniqueId=fan_id,
                            jointIndex=joint_id,
                            controlMode=p.VELOCITY_CONTROL,
                            targetVelocity=0.0,
                            force=self.joint_motor_force,
                            physicsClientId=self._physics_client_id,
                        )

    # -------------------------------------------------------------------------
    # Fan-specific methods
    # -------------------------------------------------------------------------

    def set_fans_on(self, on: bool = True) -> None:
        """Turn every fan on or off without anything pressing anything.

        The switch joint IS the stored bit -- ``extract_feature`` reads
        a fan's ``is_on`` off the switch that controls its side -- so
        setting the fan means setting that joint. Written for envs
        where the fan has no reachable button and something else
        decides (see PyBulletDominoDeclareEnv). In the button envs
        nothing calls this and the press remains the only cause.
        """
        for switch in self._switches:
            self._set_switch_on(switch.id, on)

    def any_fan_on(self) -> bool:
        """True when at least one fan is currently blowing."""
        return any(self._is_switch_on(sw.id) for sw in self._switches)

    def set_wind_target(self,
                        target_id: int,
                        z_offset: float = 0.0,
                        stop_when_toppled: bool = False,
                        force: Optional[float] = None) -> None:
        """Set the object that wind forces should be applied to.

        ``z_offset`` raises the point of application above the target's
        origin, which is what turns a shove into a topple for a body
        that stands on a narrow base. ``stop_when_toppled`` cuts the
        force once the target is down. ``force`` overrides the class
        magnitude, which is calibrated for the ball and is 16x what a
        domino needs.
        """
        self._wind_target_id = target_id
        self._wind_target_z_offset = z_offset
        self._wind_stops_when_toppled = stop_when_toppled
        self._wind_force_override = force

    def set_lateral_alignment(self, lateral: Optional[float]) -> None:
        """Aim the fan across its blowing axis, at ``lateral``.

        The wind force is computed from a fan's ORIENTATION alone, so
        the sim happily blows a domino over from a fan parked anywhere -
        which is how this env ended up with its fan 0.34 m to the side
        of the chain, on a rail whose centre (y=1.708) is outside the
        domino workspace (y<=1.49) altogether. It looked wrong because
        it WAS wrong: no real fan there blows down that line.

        It also matters beyond looks. A learning agent reads the fan's
        coordinates and reasons about how far downstream its beam
        reaches; a fan that is not where the physics pretends it is
        makes that inference unlearnable.

        Called per reset by the composed env, which knows where the
        task's chain was laid. None restores the rail's centre.
        """
        self._lateral_alignment = lateral
        if self._physics_client_id is not None:
            self._position_fans_on_sides()

    def _aligned_lateral(self, side_idx: int, default: float) -> float:
        """``default``, unless this side has been aimed at a chain."""
        if self._lateral_alignment is None:
            return default
        # Sides 0/1 (left/right) blow along x, so their free axis is y;
        # sides 2/3 (back/front) blow along y and are free in x.
        del side_idx  # both cases read the same stored scalar
        return self._lateral_alignment

    def _target_still_standing(self, target_id: int) -> bool:
        """Is the wind target still upright enough to be pushed?

        A standing domino presents its face to the airstream; a fallen
        one lies flat on the table, out of it. Without this the fan goes
        on shoving a body that is already down and slides it across the
        table indefinitely - which lets the start block bulldoze all the
        way into the target and "solve" a bridge task with none of the
        bridge built.

        Reads DominoComponent's own threshold rather than restating a
        number, so the physics stops exactly where the symbol flips
        even if that constant moves.
        """
        # pylint: disable-next=import-outside-toplevel
        from predicators.envs.pybullet_domino.components.domino_component \
            import DominoComponent
        _, orn = p.getBasePositionAndOrientation(
            target_id, physicsClientId=self._physics_client_id)
        roll = p.getEulerFromQuaternion(orn)[0]
        # Fold to [-pi/2, pi/2): a box turned 180 degrees about its own
        # width axis is the same box, so roll only means anything mod pi.
        roll = (roll + np.pi / 2) % np.pi - np.pi / 2
        return abs(roll) < DominoComponent.domino_roll_threshold

    def _apply_wind_force(self, fan_id: int, target_id: int) -> None:
        """Apply wind force from fan to target object."""
        _, orn_fan = p.getBasePositionAndOrientation(
            fan_id, physicsClientId=self._physics_client_id)

        if CFG.fan_fans_blow_opposite_direction:
            local_dir = np.array([-1.0, 0.0, 0.0])
        else:
            local_dir = np.array([1.0, 0.0, 0.0])

        rmat = np.array(p.getMatrixFromQuaternion(orn_fan)).reshape((3, 3))
        world_dir = rmat.dot(local_dir)
        pos_target, _ = p.getBasePositionAndOrientation(
            target_id, physicsClientId=self._physics_client_id)
        # Apply the force ABOVE the centre of mass, not through it. A
        # force through the centre is pure translation: a domino under
        # it slides across the table indefinitely and never tips, which
        # is how a wind-driven task ends up "solved" by the start block
        # bulldozing into the target with the chain untouched. Offset
        # upward, the same force makes a moment about the domino's
        # bottom edge and it falls. Zero for a ball, whose existing
        # behaviour (roll from a central push) is what that task wants.
        pos_apply = (pos_target[0], pos_target[1],
                     pos_target[2] + self._wind_target_z_offset)
        magnitude = (self.wind_force_magnitude
                     if self._wind_force_override is None else
                     self._wind_force_override)
        force_vec = magnitude * world_dir
        p.applyExternalForce(objectUniqueId=target_id,
                             linkIndex=-1,
                             forceObj=force_vec.tolist(),
                             posObj=pos_apply,
                             flags=p.WORLD_FRAME,
                             physicsClientId=self._physics_client_id)

    def _position_fans_on_sides(self) -> None:
        """Position all PyBullet fan bodies on their respective sides."""
        assert self._physics_client_id is not None

        # np.linspace(a, b, 1) returns [a], not the midpoint - so a
        # single-fan side lands at the LOW end of its rail while the
        # state still reports the centre. Harmless to the oracle, whose
        # wind is computed from the fan's orientation and never its
        # position, but a learning agent reasons about "how far
        # downstream of the fan the beam still reaches", and a body 0.42
        # m from its reported coordinate corrupts exactly that.
        def _rail(lo: float, hi: float, n: int) -> Any:
            if n == 1:
                return np.array([(lo + hi) / 2.0])
            return np.linspace(lo, hi, n)

        left_coords = _rail(self.fan_y_lb, self.fan_y_ub, self.num_left_fans)
        right_coords = _rail(self.fan_y_lb, self.fan_y_ub, self.num_right_fans)
        front_coords = _rail(self.fan_x_lb, self.fan_x_ub, self.num_front_fans)
        back_coords = _rail(self.fan_x_lb, self.fan_x_ub, self.num_back_fans)

        for fan_obj in self._fans:
            side_idx = fan_obj.side_idx
            fan_ids = fan_obj.fan_ids

            if side_idx == 0:  # left
                for i, fan_id in enumerate(fan_ids):
                    px = self.left_fan_x
                    py = self._aligned_lateral(0, left_coords[i])
                    pz = self.table_height + self.fan_z_len / 2
                    rot = [0.0, 0.0, 0.0]
                    update_object(fan_id,
                                  position=(px, py, pz),
                                  orientation=p.getQuaternionFromEuler(rot),
                                  physics_client_id=self._physics_client_id)

            elif side_idx == 1:  # right
                for i, fan_id in enumerate(fan_ids):
                    px = self.right_fan_x
                    py = self._aligned_lateral(1, right_coords[i])
                    pz = self.table_height + self.fan_z_len / 2
                    rot = [0.0, 0.0, np.pi]
                    update_object(fan_id,
                                  position=(px, py, pz),
                                  orientation=p.getQuaternionFromEuler(rot),
                                  physics_client_id=self._physics_client_id)

            elif side_idx == 2:  # back
                for i, fan_id in enumerate(fan_ids):
                    px = self._aligned_lateral(2, back_coords[i])
                    py = self.down_fan_y
                    pz = self.table_height + self.fan_z_len / 2
                    rot = [0.0, 0.0, np.pi / 2]
                    update_object(fan_id,
                                  position=(px, py, pz),
                                  orientation=p.getQuaternionFromEuler(rot),
                                  physics_client_id=self._physics_client_id)

            elif side_idx == 3:  # front
                for i, fan_id in enumerate(fan_ids):
                    px = self._aligned_lateral(3, front_coords[i])
                    py = self.up_fan_y
                    pz = self.table_height + self.fan_z_len / 2
                    rot = [0.0, 0.0, -np.pi / 2]
                    update_object(fan_id,
                                  position=(px, py, pz),
                                  orientation=p.getQuaternionFromEuler(rot),
                                  physics_client_id=self._physics_client_id)

    @staticmethod
    def _get_joint_id(obj_id: int, joint_name: str) -> int:
        """Get joint ID by name from PyBullet object."""
        num_joints = p.getNumJoints(obj_id)
        for j in range(num_joints):
            info = p.getJointInfo(obj_id, j)
            if info[1].decode("utf-8") == joint_name:
                return j
        return -1

    def _is_switch_on(self, switch_id: int) -> bool:
        """Check if switch is on."""
        joint_id = self._get_joint_id(switch_id, "joint_0")
        if joint_id < 0:
            return False
        j_pos, _, _, _ = p.getJointState(
            switch_id, joint_id, physicsClientId=self._physics_client_id)
        info = p.getJointInfo(switch_id,
                              joint_id,
                              physicsClientId=self._physics_client_id)
        j_min, j_max = info[8], info[9]
        frac = (j_pos / self.switch_joint_scale - j_min) / (j_max - j_min)
        return bool(frac > self.switch_on_threshold)

    def _set_switch_on(self, switch_id: int, on: bool) -> None:
        """Set switch on or off."""
        joint_id = self._get_joint_id(switch_id, "joint_0")
        if joint_id < 0:
            return
        info = p.getJointInfo(switch_id,
                              joint_id,
                              physicsClientId=self._physics_client_id)
        j_min, j_max = info[8], info[9]
        target_val = j_max if on else j_min
        p.resetJointState(switch_id,
                          joint_id,
                          target_val * self.switch_joint_scale,
                          physicsClientId=self._physics_client_id)

    # -------------------------------------------------------------------------
    # Predicate hold functions
    # -------------------------------------------------------------------------

    def _FanOn_holds(self, state: State, objects: Sequence[Object]) -> bool:
        """Check if fan/switch is on."""
        obj = objects[0]
        is_on = state.get(obj, "is_on")
        return is_on > 0.5

    def _FanFacingSide_holds(self, state: State,
                             objects: Sequence[Object]) -> bool:
        """Check if fan faces a specific side."""
        fan, side = objects
        fan_side = state.get(fan, "facing_side")
        side_idx = state.get(side, "side_idx")
        return abs(fan_side - side_idx) < 0.1

    def _Controls_holds(self, state: State, objects: Sequence[Object]) -> bool:
        """Check if switch controls fan."""
        switch, fan = objects
        switch_controls = state.get(switch, "controls_fan")
        fan_side = state.get(fan, "facing_side")
        return abs(switch_controls - fan_side) < 0.1

    # -------------------------------------------------------------------------
    # Initial state helpers
    # -------------------------------------------------------------------------

    def get_init_dict_entries(
            self,
            rng: "np.random.Generator",
            all_off: bool = True) -> Dict[Object, Dict[str, Any]]:
        """Return initial state dict entries for fans, switches, and sides."""
        init_dict: Dict[Object, Dict[str, Any]] = {}

        # Fans
        for fan_obj in self._fans:
            side_idx = fan_obj.side_idx
            if side_idx == 0:  # left
                px = self.left_fan_x
                py = self._aligned_lateral(0,
                                           (self.fan_y_lb + self.fan_y_ub) / 2)
                rot = 0.0
            elif side_idx == 1:  # right
                px = self.right_fan_x
                py = self._aligned_lateral(1,
                                           (self.fan_y_lb + self.fan_y_ub) / 2)
                rot = np.pi
            elif side_idx == 2:  # back
                px = self._aligned_lateral(2,
                                           (self.fan_x_lb + self.fan_x_ub) / 2)
                py = self.down_fan_y
                rot = np.pi / 2
            else:  # front
                px = self._aligned_lateral(3,
                                           (self.fan_x_lb + self.fan_x_ub) / 2)
                py = self.up_fan_y
                rot = -np.pi / 2

            init_dict[fan_obj] = {
                "x": px,
                "y": py,
                "z": self.table_height + self.fan_z_len / 2,
                "rot": rot,
                "facing_side": float(side_idx),
                "is_on": 0.0 if all_off else float(rng.random() > 0.5)
            }

        # Switches
        for switch_obj in self._switches:
            init_dict[switch_obj] = {
                "x": self.switch_base_x +
                self.switch_x_spacing * switch_obj.side_idx,
                "y": self.switch_y,
                "z": self.table_height,
                "rot": np.pi / 2,
                "controls_fan": float(switch_obj.side_idx),
                "is_on": 0.0 if all_off else float(rng.random() > 0.5),
            }

        # Sides
        for i, side_obj in enumerate(self._sides):
            init_dict[side_obj] = {"side_idx": float(i)}

        return init_dict

    # -------------------------------------------------------------------------
    # Public properties
    # -------------------------------------------------------------------------

    @property
    def fans(self) -> List[Object]:
        """Fans."""
        return self._fans

    @property
    def switches(self) -> List[Object]:
        """Switches."""
        return self._switches

    @property
    def sides(self) -> List[Object]:
        """Sides."""
        return self._sides

    @property
    def fan_type(self) -> Type:
        """Fan type."""
        return self._fan_type

    @property
    def switch_type(self) -> Type:
        """Switch type."""
        return self._switch_type

    @property
    def side_type(self) -> Type:
        """Side type."""
        return self._side_type
