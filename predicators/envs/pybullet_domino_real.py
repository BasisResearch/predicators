"""Real-world domino env: the ``pybullet_domino`` env retargeted to the real
Franka robot."""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pybullet as p

from predicators import utils
from predicators.envs.pybullet_domino.components.domino_component import \
    DominoComponent
from predicators.envs.pybullet_domino.env import PyBulletDominoComposedEnv, \
    PyBulletDominoEnv
from predicators.envs.pybullet_domino.real_geometry import Pose6D, \
    domino_upright_yaw, domino_world_z_offset, pose_base_to_world
from predicators.pybullet_helpers.objects import create_object, \
    create_pybullet_block
from predicators.pybullet_helpers.real_robot_bridge import execute_actions, \
    make_real_robot, reset_arm
from predicators.settings import CFG
from predicators.structs import Action, EnvironmentTask, GroundAtom, \
    Observation, State


def _base_pose(xyz: Sequence[float], quat_xyzw: Sequence[float]) -> Pose6D:
    """Base-frame ``Pose6D`` from loose sequences.

    Unpacking is the length check: a capture record or observation with
    the wrong number of components fails here rather than silently
    producing a malformed pose.
    """
    x, y, z = (float(v) for v in xyz)
    qx, qy, qz, qw = (float(v) for v in quat_xyzw)
    return Pose6D((x, y, z), (qx, qy, qz, qw))


@dataclass(frozen=True)
class _PerceivedDomino:
    """One domino as perceived, normalized from either source.

    A scene-JSON record and a live ``DominoObservation`` entry carry the
    same content under different field names, so both are converted to
    this before anything else happens. ``slot`` is the index into the
    env's domino component; ``pose_base`` is in the robot base frame.
    """
    slot: int
    capture_id: int
    pose_base: Pose6D
    role: str


class PyBulletDominoRealEnv(PyBulletDominoEnv):
    """``pybullet_domino`` on the real bench, sized/tasked from a scene
    JSON."""

    def __init__(self, use_gui: bool = False, **kwargs: Any) -> None:
        self._z_off = domino_world_z_offset(CFG.domino_real_table_z)
        # Real (test-mode) execution state. Dominoes are placed in
        # scene order, so slot i <-> capture id self._scene_ids[i].
        self._real_mode = False
        self._action_buffer: List[Action] = []
        # babyrobot's RealRobot, constructed lazily on the first real reset so
        # the env imports (and runs in sim) without the private submodule.
        self._real_robot: Optional[Any] = None
        with open(CFG.domino_real_scene, encoding="utf-8") as f:
            self._scene_ids = [int(d["id"]) for d in json.load(f)["dominoes"]]
        super().__init__(use_gui=use_gui, **kwargs)

    @classmethod
    def get_name(cls) -> str:
        return "pybullet_domino_real"

    # -- real (test-mode) execution (open-loop) --------------------
    # Open-loop: the internal sim rolls each option out (its predicators
    # BiRRT-planned policy is the trajectory generator) and, at the option
    # boundary, the buffered joint trajectory is executed on the Franka (when
    # real_robot_execute). The env state stays the sim's prediction -- no
    # option-boundary re-perception yet. Dry-run == pure sim.
    #
    # The arm is driven in-process: babyrobot's RealRobot is an ordinary Python
    # object this env holds and calls. It is built on the first real reset, so
    # the private submodule is only needed by a run that actually executes.
    def reset(self,
              train_or_test: str,
              task_idx: int,
              render: bool = False) -> Observation:
        self._real_mode = (train_or_test == "test")
        self._action_buffer = []
        real = self._real_mode and CFG.real_robot_execute
        if real and self._real_robot is None:
            self._real_robot = make_real_robot()
        obs = super().reset(train_or_test, task_idx, render=render)
        if real:
            assert self._real_robot is not None
            # Home the real arm to the env-home joint config the option
            # trajectories are planned from, so the first option's streamed
            # waypoints start where the robot is (else the drift guard trips).
            pyb = self._pybullet_robot
            fingers = {pyb.left_finger_joint_idx, pyb.right_finger_joint_idx}
            home_arm = [
                float(v) for i, v in enumerate(pyb.get_joints())
                if i not in fingers
            ]
            reset_arm(self._real_robot, home_arm)
        return obs

    def step(self, action: Action, render_obs: bool = False) -> Observation:
        obs = super().step(action, render_obs=render_obs)
        if not (self._real_mode and action.has_option()):
            return obs
        self._action_buffer.append(action)
        if not CFG.real_robot_ship_whole_episode and \
                action.get_option().terminal(obs):  # option boundary
            self._flush_real_actions()
        return obs

    def flush_real_execution(self) -> None:
        """Ship the whole episode's buffered trajectory to the robot (no-op if
        the buffer is empty, or when shipping per option).

        Callers driving the real arm must call this once the rollout is
        done; the buffer spans the WHOLE episode so the gripper split
        sees every action at once.
        """
        self._flush_real_actions()

    def _flush_real_actions(self) -> None:
        """Split the buffered actions into move/gripper segments and execute.

        The split must see a whole episode (or at least a whole pick-
        place) in ONE call: it emits a gripper segment on a finger
        transition, tracked from the START of the call. Splitting per
        option restarts that tracking, so a Place -- which begins
        already holding the domino from the Pick -- re-emits a leading
        ``close``, i.e. a SECOND force-grasp on the already-clamped
        domino. That leaves the Franka Hand stuck and the later release
        is dropped.

        RealRobot now drops a gripper command that repeats the session's
        current state, so the second force-grasp cannot reach the hand
        even if a chunk does re-emit it. This env still ships whole
        episodes; per-option chunking is the wrapper's job.
        """
        actions, self._action_buffer = self._action_buffer, []
        if actions and CFG.real_robot_execute:
            assert self._real_robot is not None
            execute_actions(self._real_robot, actions,
                            self.gripper_joint_layout())

    # -- geometry + pybullet build + decoration -----------------------------
    @classmethod
    def _apply_real_geometry(cls) -> None:
        """Set THIS subclass's robot geometry ClassVars from CFG (raise the
        base to the real bench).

        Applied in ``initialize_pybullet`` so it takes effect on BOTH
        the normal env build AND the skill factory's direct
        ``initialize_pybullet`` call (which bypasses ``__init__``).
        Reads the base xy from the untouched shared class, so it is
        idempotent. Only this subclass is configured -- the shared base
        is never mutated.

        The home EE height is not set here: the Panda homes to its own
        configuration, which is reachable by construction and identical in
        every env (see PyBulletEnv._sync_robot_init_pos_with_home). It used
        to be lowered by hand here to keep the Fetch-tuned home within the
        Panda's reach.
        """
        z_off = domino_world_z_offset(CFG.domino_real_table_z)
        base = PyBulletDominoComposedEnv.robot_base_pos
        assert base is not None
        base_xy = base[:2]
        cls.robot_base_pos = (base_xy[0], base_xy[1], float(z_off))
        cls.robot_init_tilt = float(CFG.domino_real_robot_init_tilt)
        cls.robot_init_wrist = float(CFG.domino_real_robot_init_wrist)

    @classmethod
    def initialize_pybullet(cls, using_gui: bool) -> Tuple[Any, Any, Any]:
        """Apply the real-bench geometry, build the world, then decorate this
        instance's sim (extended-table tile + robot pedestal).

        Every pipeline env is an instance of this class, so each
        configures + decorates itself.
        """
        cls._apply_real_geometry()
        physics_client_id, pybullet_robot, bodies = super(
        ).initialize_pybullet(using_gui)
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
        tile_id = create_object("urdf/table.urdf",
                                position=(0.75, 0.85, 0.2),
                                orientation=yq(np.pi / 2),
                                scale=1.0,
                                use_fixed_base=True,
                                physics_client_id=pcid)
        # Match the env's studio wood texture if this env uses studio visuals.
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
        # Robot mount pedestal: fill the table top (0.4) up to the base (z_off).
        riser_h = z_off - 0.4
        if riser_h > 1e-3:
            create_pybullet_block(color=(0.3, 0.3, 0.3, 1.0),
                                  half_extents=(0.10, 0.10, riser_h / 2),
                                  mass=0.0,
                                  friction=0.5,
                                  position=(0.75, 0.72, 0.4 + riser_h / 2),
                                  orientation=yq(0.0),
                                  physics_client_id=pcid)

    # -- roles --------------------------------------------------------------
    @staticmethod
    def _role_for_capture_id(capture_id: int) -> str:
        """Role keyed by capture id alone.

        Raw capture JSONs (``reconstruct_dominoes_markers.py`` output)
        and live observations both carry ids but no roles, so
        ``CFG.domino_real_{start,target}_id`` names the green start and
        the purple target; everything else is a movable blue.
        """
        if capture_id == CFG.domino_real_start_id:
            return "start"
        if capture_id == CFG.domino_real_target_id:
            return "target"
        return "movable"

    @classmethod
    def _domino_role(cls, d: Dict[str, Any]) -> str:
        """Role ('start' / 'target' / 'movable') for a scene domino.

        Prefers an explicit ``role`` field if the scene carries one,
        otherwise falls back to the id keying above.
        """
        if "role" in d:
            return str(d["role"])
        return cls._role_for_capture_id(int(d["id"]))

    # -- component sizing + dims --------------------------------------------
    @classmethod
    def _scene_role_counts(cls) -> Tuple[int, int]:
        """(num_target, num_nontarget) domino counts from the scene JSON."""
        with open(CFG.domino_real_scene, encoding="utf-8") as f:
            roles = [cls._domino_role(d) for d in json.load(f)["dominoes"]]
        n_target = sum(1 for r in roles if r == "target")
        return n_target, len(roles) - n_target

    @classmethod
    def _make_domino_component(
            cls, workspace_bounds: Dict[str, float]) -> DominoComponent:
        """Allocate the scene's counts and the real perceived dimensions,
        passing dims through the component ctor (not a base ClassVar mutation).

        ``domino_real_domino_dims`` is (L, W, H): a standing domino has
        body-x (L) vertical, so env height=L, width=W (broad face),
        depth=H (thickness).
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

    # -- perception -> State / Task -----------------------------------------
    # ONE conversion, three callers: the captured scene JSON, a live
    # observation building a fresh task, and a live observation correcting an
    # existing state. Both sources are first normalized to _PerceivedDomino,
    # so the scene path and the live path cannot drift apart -- including the
    # start-domino yaw canonicalization, which is easy to apply in one place
    # and forget in the other.
    #
    # These are pure conversions: no hardware, no I/O beyond reading the scene
    # file, and no notion of a robot. The real-world wrapper calls them.

    def _slot_for_capture_id(self, capture_id: int) -> Optional[int]:
        """Component slot holding the domino with this capture id.

        Dominoes are placed in scene order, so slot i holds capture id
        ``self._scene_ids[i]`` -- which is what that list is for. An id
        the scene never had has nowhere to go: perception occasionally
        reports a spurious marker, and dropping it beats aborting a run
        over one bad detection.
        """
        try:
            return self._scene_ids.index(int(capture_id))
        except ValueError:
            logging.warning(
                "pybullet_domino_real: ignoring domino id %s, which is not "
                "in the scene %s", capture_id, self._scene_ids)
            return None

    def _perceived_from_scene(
            self, records: List[Dict[str, Any]]) -> List[_PerceivedDomino]:
        """Normalize scene-JSON records."""
        perceived = []
        for d in records:
            slot = self._slot_for_capture_id(int(d["id"]))
            if slot is None:
                continue
            pose = _base_pose(d["center_base_m"], d["quat_base_xyzw"])
            perceived.append(
                _PerceivedDomino(slot=slot,
                                 capture_id=int(d["id"]),
                                 pose_base=pose,
                                 role=self._domino_role(d)))
        return perceived

    def _perceived_from_observation(self, obs: Any) -> List[_PerceivedDomino]:
        """Normalize a ``DominoObservation`` captured from the real scene.

        Duck-typed deliberately: this reads ``obs.dominoes`` and each
        entry's ``id`` / ``xyz`` / ``quat_xyzw``, so the env imports no
        babyrobot and the conversion stays testable against a plain
        stub. Observations carry no role by design, so roles come from
        the configured capture ids.
        """
        perceived = []
        for d in obs.dominoes:
            slot = self._slot_for_capture_id(int(d.id))
            if slot is None:
                continue
            pose = _base_pose(d.xyz, d.quat_xyzw)
            perceived.append(
                _PerceivedDomino(slot=slot,
                                 capture_id=int(d.id),
                                 pose_base=pose,
                                 role=self._role_for_capture_id(int(d.id))))
        return perceived

    @staticmethod
    def _canonical_start_yaw(
            yaw: float, world: Pose6D, push_dir_world: Optional[Tuple[float,
                                                                      float]],
            target_xy: Optional[Tuple[float, float]]) -> float:
        """Flip the START domino's yaw to face the direction it must topple.

        A domino is 180-degree symmetric, so perception's yaw branch is
        arbitrary; the single push has to go the intended way. The
        desired direction is an explicit ``start_push_dir_base`` when
        the scene gives one, else the default "toward the target".
        """
        pdir = push_dir_world
        if pdir is None and target_xy is not None:
            pdir = (target_xy[0] - world.xyz[0], target_xy[1] - world.xyz[1])
        if pdir is None:
            return yaw
        fx, fy = math.sin(yaw), math.cos(yaw)
        if fx * pdir[0] + fy * pdir[1] < 0.0:
            return math.atan2(-fx, -fy)  # flip 180 to face the push dir
        return yaw

    def _init_state_from_perceived(
            self,
            perceived: List[_PerceivedDomino],
            push_dir_base: Optional[Sequence[float]] = None) -> State:
        """Initial ``State`` for a task built from perceived dominoes.

        Places each domino at its transplanted world (x, y) with the
        upright heading, colored by role (green=start, purple=target,
        blue=movable) via the component's ``place_domino``.
        """
        comp = self._domino_component
        assert comp is not None, "env has no domino component"
        assert len(perceived) <= len(comp.dominos), \
            f"perceived {len(perceived)} dominoes but only " \
            f"{len(comp.dominos)} slots"

        worlds = {
            pd.slot: pose_base_to_world(pd.pose_base, self._z_off)
            for pd in perceived
        }
        target_xy = next(((worlds[pd.slot].xyz[0], worlds[pd.slot].xyz[1])
                          for pd in perceived if pd.role == "target"), None)

        # Optional per-scene override of the start domino's push direction,
        # given in the base frame as [dx, dy]; transplanted to world
        # (base->world is a +pi/2 z-rotation, so (dx, dy) -> (-dy, dx)).
        push_dir_world = None
        if push_dir_base is not None:
            push_dir_world = (-float(push_dir_base[1]),
                              float(push_dir_base[0]))

        init_dict: Dict[Any, Dict[str, float]] = {}
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

        for pd in perceived:
            world = worlds[pd.slot]
            yaw = domino_upright_yaw(world)
            if pd.role == "start":
                yaw = self._canonical_start_yaw(yaw, world, push_dir_world,
                                                target_xy)
            entry = comp.place_domino(pd.slot,
                                      world.xyz[0],
                                      world.xyz[1],
                                      yaw,
                                      is_start_block=(pd.role == "start"),
                                      is_target_block=(pd.role == "target"))
            # Perceived world (x, y, z), (canonicalized) upright heading, roll
            # flat. Keep place_domino's role color / is_held; override the pose.
            entry["x"], entry["y"], entry["z"] = world.xyz
            entry["yaw"] = yaw
            entry["roll"] = 0.0
            init_dict[comp.dominos[pd.slot]] = entry

        return utils.create_state_from_dict(init_dict)

    def _task_from_perceived(
            self,
            perceived: List[_PerceivedDomino],
            push_dir_base: Optional[Sequence[float]] = None
    ) -> EnvironmentTask:
        """Task (init state + goal) for a set of perceived dominoes."""
        init_state = self._init_state_from_perceived(perceived, push_dir_base)
        comp = self._domino_component
        assert comp is not None, "env has no domino component"

        # Goal: topple the purple (target) domino(s). With
        # domino_use_domino_blocks_as_target, Toppled is typed on domino_type
        # and _TargetDomino_holds identifies targets by color.
        goal_atoms = set()
        for dom in init_state.get_objects(comp.domino_type):
            if comp._TargetDomino_holds(  # pylint: disable=protected-access
                    init_state, [dom]):
                goal_atoms.add(GroundAtom(comp.Toppled, [dom]))
        assert len(goal_atoms) >= 1, "no purple target domino found"

        goal_nl = (
            "Move the blue dominoes such that when the green domino is pushed, "
            "the purple domino is toppled. Do NOT directly push or topple the "
            "purple domino yourself.")
        task = EnvironmentTask(init_state, goal_atoms, goal_nl=goal_nl)
        return self._add_pybullet_state_to_tasks([task])[0]

    def _build_task_from_scene(self) -> EnvironmentTask:
        """Build the captured-scene task with attached pybullet state."""
        with open(CFG.domino_real_scene, encoding="utf-8") as f:
            scene = json.load(f)
        return self._task_from_perceived(
            self._perceived_from_scene(scene["dominoes"]),
            scene.get("start_push_dir_base"))

    def task_from_observation(self,
                              obs: Any,
                              train_or_test: str = "test") -> EnvironmentTask:
        """Build a task from a live observation of the real scene.

        Same conversion as the captured scene, plus goal semantics. This
        env's goal does not depend on ``train_or_test`` -- both generate
        the same "topple the purple domino" task -- but the argument is
        part of the hook the real-world wrapper calls, since another
        environment's might.

        A live observation carries no ``start_push_dir_base``, so the
        start domino's yaw is canonicalized toward the target.
        """
        del train_or_test  # same goal either way for this env
        return self._task_from_perceived(self._perceived_from_observation(obs))

    def state_from_observation(self, obs: Any, prev_state: State) -> State:
        """Correct ``prev_state`` with what the cameras just saw.

        Only the dominoes the observation names are rewritten. Every
        other domino keeps its last known pose, and the robot's entry --
        including the joint positions in ``simulator_state``, which
        ``_set_state`` needs to avoid re-deriving the arm by IK -- is
        carried forward untouched. Observations carry no visibility flag
        by design, so "absent means unchanged" is the policy, and it
        lives here where it can be tested.

        The start domino's yaw is NOT canonicalized here. That flip
        exists to orient the opening push, so it belongs to building a
        task's initial state; mid-episode the start domino may already
        have been pushed, and re-canonicalizing would fight reality.

        Like ``domino_upright_yaw`` itself, this assumes the dominoes it
        is given are standing. Reading a toppled domino's pose back into
        the twin is not modeled by these primitives.
        """
        comp = self._domino_component
        assert comp is not None, "env has no domino component"
        state = prev_state.copy()
        for pd in self._perceived_from_observation(obs):
            world = pose_base_to_world(pd.pose_base, self._z_off)
            dom = comp.dominos[pd.slot]
            state.set(dom, "x", world.xyz[0])
            state.set(dom, "y", world.xyz[1])
            state.set(dom, "z", world.xyz[2])
            state.set(dom, "yaw", domino_upright_yaw(world))
            state.set(dom, "roll", 0.0)
        return state
