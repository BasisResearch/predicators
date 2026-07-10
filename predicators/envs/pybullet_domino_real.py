"""Real-world domino env: the ``pybullet_domino`` env retargeted to the real
Franka robot.
"""
from __future__ import annotations

import dataclasses
import json
import math
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pybullet as p

from predicators import utils
from predicators.envs.pybullet_domino.components.domino_component import \
    DominoComponent
from predicators.envs.pybullet_domino.env import (PyBulletDominoComposedEnv,
                                                  PyBulletDominoEnv)
from predicators.envs.pybullet_domino.real_geometry import (
    Pose6D, domino_upright_yaw, domino_world_z_offset, pose_base_to_world)
from predicators.pybullet_helpers.objects import (create_object,
                                                  create_pybullet_block)
from predicators.settings import CFG
from predicators.structs import EnvironmentTask, GroundAtom, State


class PyBulletDominoRealEnv(PyBulletDominoEnv):
    """``pybullet_domino`` on the real bench, sized/tasked from a scene JSON."""

    def __init__(self, use_gui: bool = False, **kwargs: Any) -> None:
        self._z_off = domino_world_z_offset(CFG.domino_real_table_z)
        super().__init__(use_gui=use_gui, **kwargs)

    @classmethod
    def get_name(cls) -> str:
        return "pybullet_domino_real"

    # -- geometry + pybullet build + decoration -----------------------------
    @classmethod
    def _apply_real_geometry(cls) -> None:
        """Set THIS subclass's robot geometry ClassVars from CFG (raise the base
        to the real bench, lower the home EE). Applied in ``initialize_pybullet``
        so it takes effect on BOTH the normal env build AND the skill factory's
        direct ``initialize_pybullet`` call (which bypasses ``__init__``). Reads
        the base xy from the untouched shared class, so it is idempotent. Only
        this subclass is configured -- the shared base is never mutated."""
        z_off = domino_world_z_offset(CFG.domino_real_table_z)
        base_xy = PyBulletDominoComposedEnv.robot_base_pos[:2]
        cls.robot_base_pos = (base_xy[0], base_xy[1], float(z_off))
        cls.robot_init_tilt = float(CFG.domino_real_robot_init_tilt)
        cls.robot_init_wrist = float(CFG.domino_real_robot_init_wrist)
        cls.robot_init_z = float(CFG.domino_real_robot_init_z)

    @classmethod
    def initialize_pybullet(cls, using_gui: bool) -> Tuple[Any, Any, Any]:
        """Apply the real-bench geometry, build the world, then decorate this
        instance's sim (extended-table tile + robot pedestal). Every pipeline
        env is an instance of this class, so each configures + decorates itself."""
        cls._apply_real_geometry()
        physics_client_id, pybullet_robot, bodies = super().initialize_pybullet(
            using_gui)
        if CFG.domino_real_decorate:
            cls._decorate(physics_client_id,
                          domino_world_z_offset(CFG.domino_real_table_z))
        return physics_client_id, pybullet_robot, bodies

    @classmethod
    def _decorate(cls, pcid: int, z_off: float) -> None:
        """Add the extended-table tile + robot pedestal (ported from
        ``birrt._decorate_scene``) with predicators' own body helpers."""
        def yq(yaw: float) -> Tuple[float, float, float, float]:
            return tuple(p.getQuaternionFromEuler([0.0, 0.0, yaw]))

        # Extra table tile toward the robot (world y=0.85), table top z=0.4.
        tile_id = create_object("urdf/table.urdf", position=(0.75, 0.85, 0.2),
                                orientation=yq(np.pi / 2), scale=1.0,
                                use_fixed_base=True, physics_client_id=pcid)
        # Match the env's studio wood texture if this env uses studio visuals.
        if getattr(cls, "_use_studio_visuals", False) and \
                getattr(cls, "table_texture_path", None) and \
                isinstance(tile_id, int):
            texid = p.loadTexture(
                utils.get_env_asset_path(cls.table_texture_path),
                physicsClientId=pcid)
            p.changeVisualShape(tile_id, -1, textureUniqueId=texid,
                                rgbaColor=(1, 1, 1, 1), physicsClientId=pcid)
        # Robot mount pedestal: fill the table top (0.4) up to the base (z_off).
        riser_h = z_off - 0.4
        if riser_h > 1e-3:
            create_pybullet_block(
                color=(0.3, 0.3, 0.3, 1.0),
                half_extents=(0.10, 0.10, riser_h / 2), mass=0.0, friction=0.5,
                position=(0.75, 0.72, 0.4 + riser_h / 2),
                orientation=yq(0.0), physics_client_id=pcid)

    # -- roles --------------------------------------------------------------
    @staticmethod
    def _domino_role(d: Dict[str, Any]) -> str:
        """Role ('start' / 'target' / 'movable') for a scene domino.

        Prefers an explicit ``role`` field if the scene carries one; raw capture
        JSONs (``reconstruct_dominoes_markers.py`` output) do not, so they are
        keyed by domino ``id`` via ``CFG.domino_real_{start,target}_id`` and
        everything else is ``movable``."""
        if "role" in d:
            return d["role"]
        if d["id"] == CFG.domino_real_start_id:
            return "start"
        if d["id"] == CFG.domino_real_target_id:
            return "target"
        return "movable"

    # -- component sizing + dims --------------------------------------------
    @classmethod
    def _scene_role_counts(cls) -> Tuple[int, int]:
        """(num_target, num_nontarget) domino counts from the scene JSON."""
        with open(CFG.domino_real_scene) as f:
            roles = [cls._domino_role(d) for d in json.load(f)["dominoes"]]
        n_target = sum(1 for r in roles if r == "target")
        return n_target, len(roles) - n_target

    @classmethod
    def _make_domino_component(cls,
                               workspace_bounds: Dict[str, float]
                               ) -> DominoComponent:
        """Allocate the scene's counts and the real perceived dimensions,
        passing dims through the component ctor (not a base ClassVar mutation).

        ``domino_real_domino_dims`` is (L, W, H): a standing domino has body-x
        (L) vertical, so env height=L, width=W (broad face), depth=H (thickness).
        """
        n_target, n_nontarget = cls._scene_role_counts()
        length, width, thickness = (float(v)
                                    for v in CFG.domino_real_domino_dims)
        return DominoComponent(num_dominos_max=n_nontarget,
                               num_targets_max=n_target,
                               num_pivots_max=0,
                               workspace_bounds=workspace_bounds,
                               domino_width=width,
                               domino_depth=thickness,
                               domino_height=length)

    # -- task generation ----------------------------------------------------
    def _generate_train_tasks(self) -> List[EnvironmentTask]:
        return [self._build_task_from_scene()]

    def _generate_test_tasks(self) -> List[EnvironmentTask]:
        return [self._build_task_from_scene()]

    def _build_task_from_scene(self) -> EnvironmentTask:
        """Build the reconstructed-scene task with attached pybullet state.

        Places each perceived domino at its transplanted world (x, y) with the
        upright heading, colored by role (green=start, purple=target,
        blue=movable) via the component's ``place_domino``. Goal = Toppled(target);
        attaches the undisturbed reward so goal_reached rejects cheats that touch
        the target early."""
        scene_path = CFG.domino_real_scene
        z_off = self._z_off
        with open(scene_path) as f:
            scene = json.load(f)
        dominoes = scene["dominoes"]

        comp = self._domino_component
        assert comp is not None, "env has no domino component"
        assert len(dominoes) <= len(comp.dominos), \
            f"scene has {len(dominoes)} dominoes but only " \
            f"{len(comp.dominos)} slots"

        init_dict = {}
        # Robot: env home (bench geometry already applied to the ClassVars).
        init_dict[self._robot] = {
            "x": self.robot_init_x,
            "y": self.robot_init_y,
            "z": self.robot_init_z,
            "fingers": self.open_fingers,
            "roll": self.robot_init_roll,
            "tilt": self.robot_init_tilt,
            "wrist": self.robot_init_wrist,
        }

        # First pass: transplant each perceived pose into the world frame, and
        # note the target (purple) position for the start's default push dir.
        worlds = [
            pose_base_to_world(
                Pose6D(tuple(d["center_base_m"]),
                       tuple(float(v) for v in d["quat_base_xyzw"])), z_off)
            for d in dominoes
        ]
        target_xy = next(((w.xyz[0], w.xyz[1])
                          for d, w in zip(dominoes, worlds)
                          if self._domino_role(d) == "target"), None)

        # Optional per-scene override of the start domino's push direction, given
        # in the base frame as [dx, dy]; transplanted to world (base->world is a
        # +pi/2 z-rotation, so (dx, dy) -> (-dy, dx)). Absent -> default below.
        push_dir_world = None
        spd = scene.get("start_push_dir_base")
        if spd is not None:
            push_dir_world = (-float(spd[1]), float(spd[0]))

        for i, (d, world) in enumerate(zip(dominoes, worlds)):
            role = self._domino_role(d)
            yaw = domino_upright_yaw(world)
            # Canonicalize the START domino's yaw so its single push topples it
            # in the intended direction (a domino is 180-deg symmetric, so
            # perception's yaw branch is arbitrary); flip by pi if its push
            # facing points away from the desired direction -- an explicit
            # start_push_dir_base if given, else the DEFAULT "toward the target".
            if role == "start":
                pdir = push_dir_world
                if pdir is None and target_xy is not None:
                    pdir = (target_xy[0] - world.xyz[0],
                            target_xy[1] - world.xyz[1])
                if pdir is not None:
                    fx, fy = math.sin(yaw), math.cos(yaw)
                    if fx * pdir[0] + fy * pdir[1] < 0.0:
                        yaw = math.atan2(-fx, -fy)  # flip 180 to face push dir

            entry = comp.place_domino(i, world.xyz[0], world.xyz[1], yaw,
                                      is_start_block=(role == "start"),
                                      is_target_block=(role == "target"))
            # Perceived world (x, y, z), (canonicalized) upright heading, roll
            # flat. Keep place_domino's role color / is_held; override the pose.
            entry["x"], entry["y"], entry["z"] = world.xyz
            entry["yaw"] = yaw
            entry["roll"] = 0.0
            init_dict[comp.dominos[i]] = entry

        init_state = utils.create_state_from_dict(init_dict)

        # Goal: topple the purple (target) domino(s). With
        # domino_use_domino_blocks_as_target, Toppled is typed on domino_type
        # and _TargetDomino_holds identifies targets by color.
        goal_atoms = set()
        for dom in init_state.get_objects(comp.domino_type):
            if comp._TargetDomino_holds(init_state, [dom]):
                goal_atoms.add(GroundAtom(comp.Toppled, [dom]))
        assert len(goal_atoms) >= 1, "no purple target domino found in scene"

        goal_nl = (
            "Move the blue dominoes such that when the green domino is pushed, "
            "the purple domino is toppled. Do NOT directly push or topple the "
            "purple domino yourself.")
        task = EnvironmentTask(init_state, goal_atoms, goal_nl=goal_nl)
        pyb_task = self._add_pybullet_state_to_tasks([task])[0]
        # Strict success: target must stay in its initial pose until the start
        # is pushed. Attach on the FINAL task so it survives
        # _add_pybullet_state_to_tasks; goal_reached reads it.
        reward_fn = self._make_undisturbed_reward_fn(comp, pyb_task.init)
        return dataclasses.replace(pyb_task, reward_fn=reward_fn)

    # -- reward -------------------------------------------------------------
    @staticmethod
    def _make_undisturbed_reward_fn(comp: Any,
                                    init_obs: State) -> Callable[[State], bool]:
        """Strict success: the target (purple) must be UNMOVED from its initial
        pose until the start (green) is pushed, and toppled at the end.

        A plain Toppled(target) goal is a final-state check with no history, so
        it would also reward the agent shoving / relocating the target itself.
        The only legitimate topple is the chain reaction, which begins strictly
        AFTER the start moves. So we watch the physics: if the target ever leaves
        its initial pose while the start is still unmoved, the agent touched it
        -> disturbed -> that episode does not count, even if the target ends
        toppled.

        NOTE (Phase 3): this per-step guard assumes continuous sim monitoring; in
        real (test) mode ground truth exists only at option boundaries, so the
        touch-before-push cheat becomes boundary-granular on hardware."""
        ROLL_TOL = math.radians(10.0)   # a domino tilted past this has moved
        POS_TOL = 0.02                  # ...or shifted past this (m); ignore jitter

        dominoes = list(init_obs.get_objects(comp.domino_type))
        targets = [d for d in dominoes if comp._TargetDomino_holds(init_obs, [d])]
        starts = [d for d in dominoes if comp._StartBlock_holds(init_obs, [d])]
        init_pose = {d: (init_obs.get(d, "x"), init_obs.get(d, "y"),
                         init_obs.get(d, "roll")) for d in dominoes}
        assert targets, "no target (purple) domino to guard"
        assert starts, "no start (green) domino to reference the push"

        def _moved(state: State, d: Any) -> bool:
            x0, y0, r0 = init_pose[d]
            return (math.hypot(state.get(d, "x") - x0,
                               state.get(d, "y") - y0) > POS_TOL
                    or abs(state.get(d, "roll") - r0) > ROLL_TOL)

        flag = {"disturbed": False}

        def reward_fn(state: State) -> bool:
            s_moved = any(_moved(state, s) for s in starts)
            t_moved = any(_moved(state, t) for t in targets)
            if not s_moved and not t_moved:
                flag["disturbed"] = False    # episode start / undisturbed pre-push
            elif t_moved and not s_moved:
                flag["disturbed"] = True      # target moved before start -> cheat
            # else s_moved: push has begun; later target motion is the cascade.
            toppled = all(comp._Toppled_holds(state, [t]) for t in targets)
            return bool(toppled and not flag["disturbed"])

        return reward_fn
