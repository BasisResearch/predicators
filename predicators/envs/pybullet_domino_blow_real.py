"""The blow task on the real bench: ``pybullet_domino_blow`` rebuilt from a
perceived scene, with the fan and the button where the cameras saw them.

Same task as :class:`PyBulletDominoBlowEnv` -- pick the block up, stand it
upwind of the goal patch, start the fan, and let the wind lay it flat in the
patch -- and the same three differences from it that make
``pybullet_domino_real`` a different env from ``pybullet_domino``:

  * the robot SETUP is the real Franka on its pedestal
    (:class:`RealSceneGeometryMixin`);
  * the SCENE is a single task read from a scene JSON
    (``CFG.domino_real_scene``, written by BabyRobotPredicator's
    ``real_skills/fan_scene_export.py``): one block, and under ``fixtures``
    the fan, the button and the goal patch, all in the robot base frame,
    plus the fan's ``wind_dir_base``;
  * the fan is started by PRESSING a momentary button from above and
    holding it (:class:`RealBenchFanComponent`), not by flicking a toggle.

Like its siblings this is pure simulation: no robot, no cameras. The
real-robot executor attaches to it and calls ``task_from_observation`` /
``state_from_observation`` -- observations carry dominoes only, since the
fixtures do not move between episodes.
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from predicators import utils
from predicators.envs.pybullet_domino.components.domino_component import \
    DominoComponent
from predicators.envs.pybullet_domino.components.goal_region_component import \
    GoalRegionComponent
from predicators.envs.pybullet_domino.components.real_bench_fan_component import \
    RealBenchFanComponent
from predicators.envs.pybullet_domino.env import PyBulletDominoBlowEnv
from predicators.envs.pybullet_domino.real_geometry import \
    DOMINO_WORLD_ROBOT_YAW, Pose6D, domino_world_z_offset, \
    pose_base_to_world
from predicators.envs.pybullet_domino_real import PerceivedDominoesMixin, \
    RealSceneGeometryMixin, _PerceivedDomino
from predicators.settings import CFG
from predicators.structs import EnvironmentTask, GroundAtom, State

# What a scene has to name for this env to stand the bench up.
_FIXTURES = ("fan", "button", "goal")


@dataclass(frozen=True)
class BenchLayout:
    """The fixed part of the bench, in the env's WORLD frame.

    Read once from the scene JSON's ``fixtures`` and ``wind_dir_base``
    and transplanted through the same base -> world transform the
    dominoes take, so the twin's fan blows down the line the real one
    does.
    """
    fan_xy: Tuple[float, float]
    wind_yaw: float  # world heading the fan blows along
    button_xy: Tuple[float, float]
    button_top_z: float
    button_yaw: float
    goal_xy: Tuple[float, float]
    goal_half_along: float  # half-extent of the patch ALONG the wind
    goal_half_across: float  # ... and across it

    @property
    def wind_dir(self) -> Tuple[float, float]:
        return (math.cos(self.wind_yaw), math.sin(self.wind_yaw))

    @classmethod
    def from_scene(cls, scene: Dict[str, Any], z_off: float) -> "BenchLayout":
        fixtures = scene.get("fixtures") or {}
        missing = [n for n in _FIXTURES if n not in fixtures]
        if missing or "wind_dir_base" not in scene:
            raise ValueError(
                f"{CFG.domino_real_scene} is not a fan-bench scene: it lacks "
                f"fixtures {missing or []} / wind_dir_base. Export one with "
                "BabyRobotPredicator's real_skills/fan_scene_export.py.")

        def world_xy_yaw(rec: Dict[str, Any]) -> Tuple[float, float, float]:
            c = rec["center_base_m"]
            yaw = float(rec.get("yaw_base_rad", 0.0))
            base = Pose6D(xyz=(float(c[0]), float(c[1]), float(c[2])),
                          quat_xyzw=(0.0, 0.0, math.sin(yaw / 2),
                                     math.cos(yaw / 2)))
            w = pose_base_to_world(base, z_off)
            # A yaw-only pose stays yaw-only under a z-rotation.
            return w.xyz[0], w.xyz[1], yaw + DOMINO_WORLD_ROBOT_YAW

        fx, fy, _ = world_xy_yaw(fixtures["fan"])
        bx, by, byaw = world_xy_yaw(fixtures["button"])
        gx, gy, gyaw = world_xy_yaw(fixtures["goal"])
        dx, dy = (float(v) for v in scene["wind_dir_base"])
        wind_yaw = math.atan2(dy, dx) + DOMINO_WORLD_ROBOT_YAW
        # The patch is a rotated box in the world; the sim's region is
        # axis-aligned along the wind. Its extent along and across the wind
        # is the box's projection onto those axes.
        gl, gw = (float(v) / 2.0 for v in fixtures["goal"]["dims_m"][:2])
        rel = gyaw - wind_yaw
        half_along = abs(gl * math.cos(rel)) + abs(gw * math.sin(rel))
        half_across = abs(gl * math.sin(rel)) + abs(gw * math.cos(rel))
        return cls(fan_xy=(fx, fy),
                   wind_yaw=wind_yaw,
                   button_xy=(bx, by),
                   button_top_z=float(fixtures["button"]["top_z_base_m"]) +
                   z_off,
                   button_yaw=byaw,
                   goal_xy=(gx, gy),
                   goal_half_along=half_along,
                   goal_half_across=half_across)


class PyBulletDominoBlowRealEnv(RealSceneGeometryMixin, PerceivedDominoesMixin,
                                PyBulletDominoBlowEnv):
    """``pybullet_domino_blow`` on the real bench, from a perceived scene."""

    def __init__(self, use_gui: bool = False, **kwargs: Any) -> None:
        self._z_off = domino_world_z_offset(CFG.domino_real_table_z)
        scene = self._load_scene()
        self._scene_ids = [int(d["id"]) for d in scene["dominoes"]]
        self._bench = BenchLayout.from_scene(scene, self._z_off)
        super().__init__(use_gui=use_gui, **kwargs)

    @classmethod
    def get_name(cls) -> str:
        return "pybullet_domino_blow_real"

    @staticmethod
    def _load_scene() -> Dict[str, Any]:
        with open(CFG.domino_real_scene, encoding="utf-8") as f:
            return json.load(f)

    @property
    def bench(self) -> BenchLayout:
        """Where the fan, the button and the patch are."""
        return self._bench

    # -- components ---------------------------------------------------------
    @staticmethod
    def _role_for_capture_id(capture_id: int) -> str:
        """Every domino on this bench is the one movable block."""
        del capture_id
        return "movable"

    @classmethod
    def _make_domino_component(
            cls, workspace_bounds: Dict[str, float]) -> DominoComponent:
        """One block, at the real block's dimensions."""
        with open(CFG.domino_real_scene, encoding="utf-8") as f:
            n_blocks = len(json.load(f)["dominoes"])
        if n_blocks != 1:
            raise ValueError(
                f"the blow task is one block; {CFG.domino_real_scene} has "
                f"{n_blocks}")
        length, width, thickness = (float(v)
                                    for v in CFG.domino_real_domino_dims)
        return DominoComponent(num_dominos_max=1,
                               num_targets_max=0,
                               num_pivots_max=0,
                               workspace_bounds=workspace_bounds,
                               domino_width=width,
                               domino_depth=thickness,
                               domino_height=length)

    def _make_fan_component(self,
                            bounds: Dict[str, float]) -> RealBenchFanComponent:
        b = self._bench
        return RealBenchFanComponent(fan_xy=b.fan_xy,
                                     wind_yaw=b.wind_yaw,
                                     button_xy=b.button_xy,
                                     button_top_z=b.button_top_z,
                                     button_yaw=b.button_yaw,
                                     workspace_bounds=bounds,
                                     table_height=self.table_height,
                                     table_width=self.table_width)

    def _extra_components(self, bounds: Dict[str, float],
                          domino_comp: DominoComponent) -> List[Any]:
        comps = super()._extra_components(bounds, domino_comp)
        region = self._goal_region_component
        assert region is not None
        b = self._bench
        region.set_region_xy(*b.goal_xy)
        # The generated task's patch is a class constant; this one is the
        # marker the cameras saw. The wind runs along the region's x in
        # the sim's convention, so along-wind is half_x.
        region.set_region_half_extents(b.goal_half_along, b.goal_half_across)
        return comps

    # -- task generation ----------------------------------------------------
    def _generate_train_tasks(self) -> List[EnvironmentTask]:
        return [self._build_task_from_scene()]

    def _generate_test_tasks(self) -> List[EnvironmentTask]:
        return [self._build_task_from_scene()]

    def _build_task_from_scene(self) -> EnvironmentTask:
        scene = self._load_scene()
        return self._task_from_perceived(
            self._perceived_from_scene(scene["dominoes"]))

    def task_from_observation(self,
                              obs: Any,
                              train_or_test: str = "test") -> EnvironmentTask:
        """A task from a live look at the bench: the block where the cameras
        found it, the fixtures where the scene file says."""
        del train_or_test
        return self._task_from_perceived(self._perceived_from_observation(obs))

    def _init_state_from_perceived(self,
                                   perceived: List[_PerceivedDomino]) -> State:
        comp = self._domino_component
        assert comp is not None and self._fan_component is not None
        assert self._goal_region_component is not None
        if len(perceived) != 1:
            raise ValueError(
                f"the blow task wants exactly one block; perceived "
                f"{len(perceived)}")
        init_dict: Dict[Any, Dict[str, float]] = {
            self._robot: {
                "x": self.robot_init_x,
                "y": self.robot_init_y,
                "z": self.robot_init_z,
                "fingers": self.open_fingers,
                "roll": self.robot_init_roll,
                "tilt": self.robot_init_tilt,
                "wrist": self.robot_init_wrist,
            }
        }
        pd = perceived[0]
        world = pose_base_to_world(pd.pose_base, self._z_off)
        roll, yaw = self._env_angles(world, pd.capture_id)
        entry = comp.place_domino(pd.slot, world.xyz[0], world.xyz[1], yaw)
        entry["x"], entry["y"], entry["z"] = world.xyz
        entry["yaw"] = yaw
        entry["roll"] = roll
        init_dict[comp.dominos[pd.slot]] = entry
        rng = np.random.default_rng(0)  # nothing here is random
        init_dict.update(self._fan_component.get_init_dict_entries(rng))
        init_dict.update(
            self._goal_region_component.get_init_dict_entries(rng))
        return utils.create_state_from_dict(init_dict)

    def _task_from_perceived(
            self, perceived: List[_PerceivedDomino]) -> EnvironmentTask:
        init_state = self._init_state_from_perceived(perceived)
        comp = self._domino_component
        region = self._goal_region_component
        assert comp is not None and region is not None
        block = comp.dominos[perceived[0].slot]
        goal_atoms: Set[GroundAtom] = {
            GroundAtom(region.InGoal, [block, region.region])
        }
        goal_nl = (
            "Pick up the block and put it down so that when the fan's button "
            "is pressed and held, the wind knocks it over and it ends up lying "
            "FLAT inside the goal region. Putting the block down in the region "
            "is not enough - you cannot place it on its side, so only the wind "
            "can leave it flat. The fan runs only while the button is held.")
        task = EnvironmentTask(init_state, goal_atoms, goal_nl=goal_nl)
        return self._add_pybullet_state_to_tasks([task])[0]

    # -- perception -> State ------------------------------------------------
    def state_from_observation(self, obs: Any, prev_state: State) -> State:
        """Correct the block's pose with what the cameras just saw.

        Only the block is rewritten; the fixtures never move and the
        robot's entry (joint positions included) is carried forward. A
        held block is left to the twin's belief, for the reason
        ``pybullet_domino_real`` gives: perception snaps everything to
        the table, and a block "seen" lying where the gripper would drop
        it would be teleported out of the hand.
        """
        comp = self._domino_component
        assert comp is not None
        state = prev_state.copy()
        for pd in self._perceived_from_observation(obs):
            dom = comp.dominos[pd.slot]
            if prev_state.get(dom, "is_held") > 0.5:
                continue
            world = pose_base_to_world(pd.pose_base, self._z_off)
            roll, yaw = self._env_angles(world, pd.capture_id)
            state.set(dom, "x", world.xyz[0])
            state.set(dom, "y", world.xyz[1])
            state.set(dom, "z", world.xyz[2])
            state.set(dom, "yaw", yaw)
            state.set(dom, "roll", roll)
        return state

    # -- the gust -----------------------------------------------------------
    def _log_gust_outcome(self) -> None:
        """Where the gust left the block, measured ALONG the wind."""
        try:
            state = self._get_state()
        except Exception:  # pylint: disable=broad-except
            return
        blocks = [o for o in state if o.type.name == "domino"]
        regions = [o for o in state if o.type.name == "region"]
        if not blocks or not regions:
            return
        block, region = blocks[0], regions[0]
        roll = float(state.get(block, "roll"))
        roll = (roll + np.pi / 2) % np.pi - np.pi / 2
        dx, dy = self._bench.wind_dir
        along = (
            (float(state.get(block, "x")) - float(state.get(region, "x"))) * dx
            + (float(state.get(block, "y")) - float(state.get(region, "y"))) *
            dy)
        half = float(state.get(region, "half_x"))
        logging.info(
            "[blow_real] gust over: block %.4f m along the wind from the "
            "patch centre (patch +/- %.3f), roll=%.3f | flat=%s in=%s", along,
            half, roll,
            abs(roll) >= 0.087,
            abs(along) <= half)


__all__ = ["BenchLayout", "PyBulletDominoBlowRealEnv"]
