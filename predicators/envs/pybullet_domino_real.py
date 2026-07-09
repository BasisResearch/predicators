"""Real-world domino env: the stock ``pybullet_domino`` env retargeted to the
real Franka bench, run through the STOCK predicators pipeline.

This subclass encapsulates everything ``real_skills/run_online_learning.py`` used
to do bespokely (bypassing ``main.setup_environment``): apply the babyrobot
``bench_setup`` geometry + grasp patches, size the domino component from a
reconstructed-scene JSON, and build the single train/test task (with the
"undisturbed" reward) from that scene. Because stock ``setup_environment`` builds
every env it needs -- entry, option-model, sysid, fit -- via
``create_new_env(CFG.env)`` = this class, the bench geometry and per-instance
scene decoration apply to all of them automatically, with NO pipeline bypass.

Learning + exploration run in sim on the reconstructed scene (unchanged from
run_online_learning). Real-robot execution at TEST time (option-boundary execute
+ re-perceive) lands in Phase 3 as ``step``/``reset``/``goal_reached`` overrides;
this module is the Phase-2 sim-only foundation.

Runs in the ``robot-ml`` conda env. The babyrobot / pose_estimation packages are
put on ``sys.path`` from ``CFG.domino_real_repo_root`` (the babyrobot worktree),
lazily, so this module imports without them present (env registration scans it).
"""
from __future__ import annotations

import dataclasses
import json
import math
import sys
from typing import Any, Callable, Dict, List

import numpy as np

from predicators import utils
from predicators.envs.pybullet_domino.components.domino_component import \
    DominoComponent
from predicators.envs.pybullet_domino.env import PyBulletDominoEnv
from predicators.settings import CFG
from predicators.structs import EnvironmentTask, GroundAtom, State

# The bench_setup option-factory / bidirectional-push patches mutate process-
# global classmethods, so they STACK if re-run across the several env instances
# the pipeline builds (entry, option-model, sysid, ...). Apply them exactly once.
_PATCHES_APPLIED = False


def _bootstrap_babyrobot() -> None:
    """Put the babyrobot worktree root on sys.path so ``babyrobot`` /
    ``pose_estimation`` import from the stock predicators cwd. Idempotent."""
    root = CFG.domino_real_repo_root
    if root and root not in sys.path:
        sys.path.insert(0, root)


class PyBulletDominoRealEnv(PyBulletDominoEnv):
    """``pybullet_domino`` on the real bench, sized/tasked from a scene JSON."""

    def __init__(self, use_gui: bool = False, **kwargs: Any) -> None:
        _bootstrap_babyrobot()
        from babyrobot.backends import bench_setup
        from babyrobot.config import SkillConfig

        req = self._build_bench_req(SkillConfig())
        z_off = float(req["z_offset"])
        # BEFORE the env/robot is built: retarget the env ClassVars to the real
        # bench geometry and tighten the sim gripper (both consumed at build).
        bench_setup.apply_bench_geometry_from_req(req)
        bench_setup.apply_closed_fingers(req)

        # Builds the domino component (our _make_domino_component reads the
        # scene) + robot + pybullet world with the geometry above applied.
        super().__init__(use_gui=use_gui, **kwargs)
        self._bench_req = req
        self._z_off = z_off

        # Decorate THIS instance's sim (solid extended-table tile + robot
        # pedestal) so the env the agent renders / simulates through is
        # real-bench-identical. Every pipeline env is an instance of this class
        # built via create_new_env, so decorating in __init__ covers all of them
        # -- replacing bench_setup.decorate_all_envs's fragile name-matching
        # create_new_env wrapper (which only fired for name == "pybullet_domino").
        if req.get("decorate_scene", False):
            from babyrobot.backends.birrt import _decorate_scene
            _decorate_scene(type(self),
                            self._pybullet_robot.physics_client_id, z_off)

        # Process-global patches: apply once (stack if re-run per instance).
        global _PATCHES_APPLIED  # pylint: disable=global-statement
        if not _PATCHES_APPLIED:
            bench_setup.apply_grasp_patches(req)
            bench_setup.apply_bidirectional_push(req)
            _PATCHES_APPLIED = True

    @classmethod
    def get_name(cls) -> str:
        return "pybullet_domino_real"

    # -- bench request ------------------------------------------------------
    @staticmethod
    def _build_bench_req(sc: Any) -> Dict[str, Any]:
        """The ``bench_setup`` request dict, built from babyrobot ``SkillConfig``
        (the calibrated bench numbers). Relocated verbatim from
        ``run_online_learning._bench_req`` so both entry points build it
        identically."""
        return {
            "z_offset": sc.z_offset,
            "robot_init_tilt": sc.robot_init_tilt,
            "robot_init_wrist": sc.robot_init_wrist,
            "robot_init_z": sc.robot_init_z,
            "domino_dims": list(sc.domino_dims),
            "pybullet_closed_fingers": sc.pybullet_closed_fingers,
            "skip_domino_collision": sc.skip_domino_collision,
            "place_move_above": sc.place_move_above,
            "pick_lift_to_carry": sc.pick_lift_to_carry,
            "pick_open_approach": sc.pick_open_approach,
            "grasp_close_before_lift": sc.grasp_close_before_lift,
            "decorate_scene": sc.decorate_scene,
            "bidirectional_push": True,
        }

    # -- component sizing ---------------------------------------------------
    @staticmethod
    def _scene_role_counts() -> tuple:
        """(num_target, num_nontarget) domino counts from the scene JSON.

        With ``domino_use_domino_blocks_as_target`` the component makes
        (num_dominos + num_targets) domino slots, so we split the scene's
        dominoes into non-target (start + movable) vs target and feed those
        back -- staying in lockstep as the scene's domino count changes."""
        with open(CFG.domino_real_scene) as f:
            roles = [d["role"] for d in json.load(f)["dominoes"]]
        n_target = sum(1 for r in roles if r == "target")
        return n_target, len(roles) - n_target

    @classmethod
    def _make_domino_component(cls,
                               workspace_bounds: Dict[str, float]
                               ) -> DominoComponent:
        """Allocate exactly the scene's counts (overrides the base, which reads
        the CFG.domino_{train,test}_num_* ranges we intentionally do not set)."""
        n_target, n_nontarget = cls._scene_role_counts()
        return DominoComponent(num_dominos_max=n_nontarget,
                               num_targets_max=n_target,
                               num_pivots_max=0,
                               workspace_bounds=workspace_bounds)

    # -- task generation ----------------------------------------------------
    def _generate_train_tasks(self) -> List[EnvironmentTask]:
        return [self._build_task_from_scene()]

    def _generate_test_tasks(self) -> List[EnvironmentTask]:
        return [self._build_task_from_scene()]

    def _build_task_from_scene(self) -> EnvironmentTask:
        """Build the reconstructed-scene task with attached pybullet state.

        Relocated verbatim from ``run_online_learning._build_task_from_scene``:
        places each perceived domino at its transplanted world (x, y) with the
        upright heading (matching ``birrt._build_state``), colored by role
        (green=start, purple=target, blue=movable) via the component's
        ``place_domino``. Goal = Toppled(target); attaches the undisturbed
        reward so goal_reached rejects cheats that touch the target early."""
        from babyrobot.geometry import (Pose6D, domino_upright_yaw,
                                        pose_base_to_world)

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
                          if d["role"] == "target"), None)

        # Optional per-scene override of the start domino's push direction, given
        # in the base frame as [dx, dy]; transplanted to world (base->world is a
        # +pi/2 z-rotation, so (dx, dy) -> (-dy, dx)). Absent -> default below.
        push_dir_world = None
        spd = scene.get("start_push_dir_base")
        if spd is not None:
            push_dir_world = (-float(spd[1]), float(spd[0]))

        for i, (d, world) in enumerate(zip(dominoes, worlds)):
            role = d["role"]
            yaw = domino_upright_yaw(world)
            # Canonicalize the START domino's yaw so its single push topples it
            # in the intended direction. A domino is 180-deg symmetric, so
            # perception's yaw branch is arbitrary; flip it by pi if its push
            # facing points away from the desired direction (explicit
            # start_push_dir_base if given, else the DEFAULT of "toward the
            # purple target"). Keeps yaw == push-direction consistent.
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
            # Match birrt._build_state's placement: perceived world (x, y, z),
            # (canonicalized) upright heading, roll flat. Keep place_domino's
            # role color / is_held; override the pose.
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
        # is pushed (else the agent cheated by touching it). Attach on the FINAL
        # task so it survives _add_pybullet_state_to_tasks; goal_reached reads it.
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
        toppled. Relocated verbatim from run_online_learning.

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
