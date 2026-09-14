"""The bench's fan and its button, where the cameras saw them.

``FanComponent`` lays fans out on the rails of a generated workspace and
switches them with a toggle the robot flicks sideways. The real bench has
one fan standing wherever the operator put it, blowing along a measured
axis, and an arcade button beside it that the robot presses from above and
that only stays on while it is held. This component keeps the parent's
types, predicates and wind physics -- so the blow task's options and
processes work unchanged -- and replaces the geometry: the fan body sits at
its measured pose, and the switch body is the button proxy from
BabyRobotPredicator's assets, with a return spring on its plunger.

The state's ``switch`` object IS the button (same type, so ``Controls`` and
``FanOn`` read as before); its ``z`` is the plunger's top, which is where a
press descends to.
"""

from typing import Any, ClassVar, Dict, Optional, Tuple

import numpy as np
import pybullet as p

from predicators.envs.pybullet_domino.components.fan_component import \
    FanComponent
from predicators.pybullet_helpers.objects import create_object, update_object
from predicators.structs import Object

# The button proxy's geometry (button_arcade60_articulated.urdf): the URDF
# root is the holder mesh's CORNER, the holder is a 70 mm square, its panel
# is 60 mm up, and the plunger stands 13 mm proud of the panel at rest.
_BUTTON_HOLDER_HALF_M = 0.035
_BUTTON_PLUNGER_TOP_M = 0.060 + 0.013
_BUTTON_JOINT = "button_press"
# The plunger's prismatic joint runs 0 (rest) to -4 mm (bottomed out); the
# switch actuates part-way down. Measured on the bench: 3.1 mm of travel to
# actuation (press_calibration.json), rounded down so a sim press that
# bottoms out reads as on with margin.
_BUTTON_TRAVEL_M = 0.004
_BUTTON_ACTUATION_M = 0.0025
# Return spring preload: the plunger pops back up unless pressed harder
# than this, so the button is momentary in sim as on the bench.
_BUTTON_SPRING_N = 2.5

_MISSING_BUTTON_ASSET = (
    "the real-bench fan component needs the button proxy URDF, which lives "
    "in the private BabyRobotPredicator package (markerless_estimation."
    "assets.button_arcade60_proxy). Check the submodule out and install it, "
    "or pass button_urdf explicitly.")


def _button_urdf_path() -> str:
    """The articulated button proxy, from BabyRobotPredicator's assets.

    Imported lazily: babyrobot is optional and must never be imported at
    module level (see real_robot_bridge).
    """
    # pylint: disable=import-outside-toplevel,import-error
    try:
        from markerless_estimation.assets.button_arcade60_proxy import \
            asset_path
    except ImportError as e:
        raise ImportError(_MISSING_BUTTON_ASSET) from e
    return str(asset_path("button_arcade60_articulated.urdf"))


class RealBenchFanComponent(FanComponent):
    """One fan at a measured pose, one arcade button at another.

    Args:
        fan_xy: World xy of the fan body.
        wind_yaw: World heading the fan blows along (radians; the wind
            force is ``(cos, sin)`` of this).
        button_xy: World xy of the button's centre.
        button_top_z: World z of the plunger's top at rest -- the height
            a press descends to, and the button's ``z`` in the state.
        button_yaw: World yaw of the button body (cosmetic: the plunger
            is round).
        button_urdf: Override for the button URDF path; default is the
            proxy shipped with BabyRobotPredicator.
    """

    # The fan model's visual size; the wind is orientation-only so this
    # never affects the physics. Kept at the parent's scale.
    fan_scale: ClassVar[float] = 0.08

    def __init__(self,
                 fan_xy: Tuple[float, float],
                 wind_yaw: float,
                 button_xy: Tuple[float, float],
                 button_top_z: float,
                 button_yaw: float = 0.0,
                 button_urdf: Optional[str] = None,
                 **kwargs: Any) -> None:
        kwargs.setdefault("num_sides", 1)
        kwargs.setdefault("fans_per_side", 1)
        kwargs["switch_xy"] = tuple(button_xy)
        super().__init__(**kwargs)
        self._fan_xy = (float(fan_xy[0]), float(fan_xy[1]))
        self._wind_yaw = float(wind_yaw)
        self._button_xy = (float(button_xy[0]), float(button_xy[1]))
        self._button_top_z = float(button_top_z)
        self._button_yaw = float(button_yaw)
        self._button_urdf = button_urdf
        self._button_joint_id: int = -1

    # -- geometry --------------------------------------------------------

    @property
    def wind_yaw(self) -> float:
        """World heading the fan blows along."""
        return self._wind_yaw

    @property
    def wind_dir(self) -> Tuple[float, float]:
        """Unit vector the fan blows along, in world xy."""
        return (float(np.cos(self._wind_yaw)), float(np.sin(self._wind_yaw)))

    @property
    def button(self) -> Object:
        """The button, as the state knows it (a ``switch``)."""
        return self._switches[0]

    @property
    def button_top_z(self) -> float:
        """World z of the plunger's top at rest."""
        return self._button_top_z

    # -- bodies ----------------------------------------------------------

    def initialize_pybullet(self, physics_client_id: int) -> Dict[str, Any]:
        """One fan body on side 0 and the button proxy as its switch."""
        self._physics_client_id = physics_client_id
        fan_urdf = "urdf/partnet_mobility/fan/101450/mobility.urdf"
        fid = create_object(asset_path=fan_urdf,
                            scale=self.fan_scale,
                            use_fixed_base=True,
                            physics_client_id=physics_client_id)
        urdf = self._button_urdf or _button_urdf_path()
        # The URDF root is the holder's corner; place it so the holder's
        # centre is at button_xy and the plunger's rest top at button_top_z.
        c, s = np.cos(self._button_yaw), np.sin(self._button_yaw)
        corner = np.array([
            self._button_xy[0] - c * _BUTTON_HOLDER_HALF_M +
            s * _BUTTON_HOLDER_HALF_M,
            self._button_xy[1] - s * _BUTTON_HOLDER_HALF_M -
            c * _BUTTON_HOLDER_HALF_M,
            self._button_top_z - _BUTTON_PLUNGER_TOP_M,
        ])
        button_id = p.loadURDF(
            urdf,
            basePosition=corner.tolist(),
            baseOrientation=p.getQuaternionFromEuler([0.0, 0.0,
                                                      self._button_yaw]),
            useFixedBase=True,
            physicsClientId=physics_client_id)
        self._button_joint_id = self._get_joint_id(button_id, _BUTTON_JOINT)
        assert self._button_joint_id >= 0, \
            f"button URDF {urdf} has no joint {_BUTTON_JOINT!r}"
        # The return spring: a position servo holding the plunger at rest
        # with the preload's worth of force. A press has to beat it, and
        # the plunger pops back the moment the pad lifts.
        p.setJointMotorControl2(button_id,
                                self._button_joint_id,
                                p.POSITION_CONTROL,
                                targetPosition=0.0,
                                force=_BUTTON_SPRING_N,
                                physicsClientId=physics_client_id)
        return {
            "fan_ids_left": [fid],
            "fan_ids_right": [],
            "fan_ids_back": [],
            "fan_ids_front": [],
            "switch_ids": [button_id],
        }

    def _position_fans_on_sides(self) -> None:
        """The fan stands where it was seen, facing where it blows."""
        assert self._physics_client_id is not None
        fan_obj = self._fans[0]
        for fan_id in fan_obj.fan_ids:
            update_object(fan_id,
                          position=(self._fan_xy[0], self._fan_xy[1],
                                    self.table_height + self.fan_z_len / 2),
                          orientation=p.getQuaternionFromEuler(
                              [0.0, 0.0, self._wind_yaw]),
                          physics_client_id=self._physics_client_id)

    def set_lateral_alignment(self, lateral: Optional[float]) -> None:
        """A real fan does not slide along a rail to aim at the block."""
        del lateral

    # -- the button's bit ------------------------------------------------

    def _is_switch_on(self, switch_id: int) -> bool:
        """Pressed past actuation. The button is momentary: this is true
        only while something holds the plunger down."""
        if self._button_joint_id < 0:
            return False
        q = p.getJointState(switch_id,
                            self._button_joint_id,
                            physicsClientId=self._physics_client_id)[0]
        return bool(q < -_BUTTON_ACTUATION_M)

    def _set_switch_on(self, switch_id: int, on: bool) -> None:
        """Write the bit: bottom the plunger out, or let it rest.

        A state that says the fan is on puts the plunger down; nothing
        holds it there, so the spring returns it on the next step unless
        the pad is on it -- which is what a momentary button does.
        """
        if self._button_joint_id < 0:
            return
        p.resetJointState(switch_id,
                          self._button_joint_id,
                          -_BUTTON_TRAVEL_M if on else 0.0,
                          physicsClientId=self._physics_client_id)

    # -- state -----------------------------------------------------------

    def get_init_dict_entries(self,
                              rng: "np.random.Generator",
                              all_off: bool = True) -> Dict[Object, Dict[str, Any]]:
        del rng, all_off  # the bench starts with the fan off, always
        fan_obj, switch_obj = self._fans[0], self._switches[0]
        init_dict: Dict[Object, Dict[str, Any]] = {
            fan_obj: {
                "x": self._fan_xy[0],
                "y": self._fan_xy[1],
                "z": self.table_height + self.fan_z_len / 2,
                "rot": self._wind_yaw,
                "facing_side": 0.0,
                "is_on": 0.0,
            },
            switch_obj: {
                "x": self._button_xy[0],
                "y": self._button_xy[1],
                "z": self._button_top_z,
                "rot": self._button_yaw,
                "controls_fan": 0.0,
                "is_on": 0.0,
            },
        }
        for i, side_obj in enumerate(self._sides):
            init_dict[side_obj] = {"side_idx": float(i)}
        return init_dict
