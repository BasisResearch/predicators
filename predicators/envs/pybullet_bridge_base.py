"""Observable simulation core of the bridge environment.

This module is the bridge env's BASE SIM: scene geometry and physical
constants, the construction of the table, blocks, glue bottle, site
pads and wet-glue patch visuals, and state read/write for every
feature (including the glue / cure / attachment features the blocks
carry) - everything needed to run rigid-body rollouts of the scene.
It deliberately contains NO residual dynamics (how glue is applied,
how joints cure, and what a cured joint does to the blocks all live in
the ``PyBulletBridgeEnv`` subclass), no task generation, and no
predicate / goal semantics.

That boundary is a visibility contract, enforced structurally rather
than by redaction: when ``CFG.agent_sim_provide_base_sim_source`` is
on, THIS FILE is copied verbatim into the learning agent's sandbox as
reference material ("the robot knows its own simulator"), so the file
the agent reads is byte-identical to the code its base-sim rollouts
(``skip_residual_dynamics=True``) execute. Anything that would leak the
learning target - the glue, cure and attachment laws and their
constants, the task distribution, predicate and goal thresholds - must
live in ``pybullet_bridge.py`` (the concrete subclass), never here.
"""
from typing import Any, ClassVar, Dict, List, Sequence, Set, Tuple

import numpy as np
import pybullet as p

from predicators.envs.pybullet_env import PyBulletEnv
from predicators.pybullet_helpers.geometry import Pose3D, Quaternion
from predicators.pybullet_helpers.objects import create_object, \
    create_pybullet_block, update_object
from predicators.pybullet_helpers.robots import SingleArmPyBulletRobot
from predicators.settings import CFG
from predicators.structs import Object, State, Type

# Faces that carry a wet-glue flag (block-local frame).
GLUE_FACES = ("top", "end_a", "end_b")
# Attachment slots, one per block face that can record a partner:
# the glue faces plus ``bottom``.
ATTACH_SLOTS = ("top", "bottom", "end_a", "end_b")


class PyBulletBridgeBaseEnv(PyBulletEnv):
    """Sim core of the bridge scene: identical rectangular blocks, a glue
    bottle, and two site pads on a table.

    Abstract on purpose - it defines no name, predicates, tasks, or
    domain-specific step, so env discovery skips it; the concrete env
    is ``PyBulletBridgeEnv``.

    Two block schemas, one logical type: the fully-observable
    ``_block_type`` carries the ``cure_*`` and ``attached_*`` features
    as observable features, while ``_block_type_po`` drops both (they
    ride ``state.privileged`` instead). Both keep them as
    ``sim_features``, so the ``block.cure_top`` (etc.) Python
    attributes are the internal source of truth. ``__init__`` swaps the
    type when ``CFG.partially_observable`` is set.
    """

    @classmethod
    def get_base_sim_source_files(cls) -> List[str]:
        # This module IS the visible sim core (see the module docstring's
        # visibility contract); pybullet_env.py is the generic engine it
        # is built on. pybullet_bridge.py (residual dynamics, task
        # generation, predicates) must never be listed here.
        return [
            "predicators/envs/pybullet_bridge_base.py",
            "predicators/envs/pybullet_env.py",
        ]

    # -------------------------------------------------------------------------
    # Table / workspace config (mirrors pybullet_bond)
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
    # Geometry
    # -------------------------------------------------------------------------
    # ONE block shape (a 10x5x5 box, long axis = local x); legs and
    # spans are the SAME block at different orientations. A leg is the
    # box stood on end: pitch = -pi/2 (local +x up, so its world-top
    # face is its local ``end_b`` face). Orientation features are
    # (pitch, yaw); raw Euler read-backs at pitch = +-pi/2 hit the
    # gimbal singularity, so the env canonicalizes block orientations
    # from the quaternion (see _canonical_block_orientation).
    block_half_extents: ClassVar[Tuple[float, float,
                                       float]] = (0.05, 0.025, 0.025)
    # World-frame half extents by orientation family (conveniences
    # derived from block_half_extents; samplers and GT models size
    # standing/lying geometry with these).
    leg_half_extents: ClassVar[Tuple[float, float,
                                     float]] = (0.025, 0.025, 0.05)
    span_half_extents: ClassVar[Tuple[float, float,
                                      float]] = (0.05, 0.025, 0.025)
    # Single source of truth for the leg count: __init__'s Object lists
    # and initialize_pybullet's body creation both read it.
    n_legs: ClassVar[int] = 2
    # Blocks carry a full free-SO(3) orientation as (roll, pitch, yaw);
    # register the triple so reconstruction diffs compare it as one
    # rotation (geodesic angle) instead of axis-by-axis, which is
    # spuriously large at the gimbal pole (standing blocks).
    _ORIENTATION_EULER_TRIPLES: ClassVar[Tuple[Tuple[str, str, str], ...]] = \
        PyBulletEnv._ORIENTATION_EULER_TRIPLES + (("roll", "pitch", "yaw"), )
    block_mass: ClassVar[float] = 0.1
    bottle_half_extents: ClassVar[Tuple[float, float,
                                        float]] = (0.012, 0.012, 0.03)
    site_half_extents: ClassVar[Tuple[float, float,
                                      float]] = (0.045, 0.045, 0.0001)

    # Colors
    # Fixed muted slate for the site pads (reads as a "marked spot" on
    # the pale wood table without shouting).
    site_color: ClassVar[Tuple[float, float, float,
                               float]] = (0.42, 0.47, 0.53, 1.0)
    bottle_color: ClassVar[Tuple[float, float, float,
                                 float]] = (0.9, 0.9, 0.98, 1.0)  # off-white
    glue_wet_color: ClassVar[Tuple[float, float, float,
                                   float]] = (0.95, 0.85, 0.25, 0.9)  # yellow

    # -------------------------------------------------------------------------
    # Types
    # -------------------------------------------------------------------------
    _robot_type = Type("robot",
                       ["x", "y", "z", "fingers", "roll", "tilt", "wrist"],
                       angular_features=["roll", "tilt", "wrist"])
    # `glue_*` (wet-glue flags) are observable in both modes.
    # `attached_*` (partner block index, -1 if none) and `cure_*` are
    # observable in FO mode but dropped from the PO type, where they
    # ride ``state.privileged``. All stay Python attributes (the
    # internal source of truth).
    # Pose is the FULL 6D (x, y, z, roll, pitch, yaw): orientation =
    # Rz(yaw) @ Ry(pitch) @ Rx(roll), so pitch = elevation of the
    # block's long axis (0 = lying flat, -pi/2 = standing with local +x
    # up, the leg pose), yaw = azimuth, roll = spin about the long
    # axis. Any physical orientation is representable -- nothing is
    # silently erased at state syncs. Features are CANONICALIZED from
    # the quaternion on read (see _canonical_block_orientation): within
    # ~1 deg of the gimbal pole the roll/yaw split is numerically
    # degenerate, so roll folds to 0 there; reconstruction checks
    # compare the triple as a geodesic rotation (gimbal-safe), not
    # axis-by-axis.
    # half_x/y/z are the block's BODY-FRAME half extents (constant;
    # local x is the long axis). Observable geometry: an agent needs
    # them to compute face centers and touch spacings without probing
    # the physics for block dimensions.
    _block_features_common = [
        "x", "y", "z", "roll", "pitch", "yaw", "half_x", "half_y", "half_z",
        "is_held", "glue_top", "glue_end_a", "glue_end_b"
    ]
    # attached_* (partner block index, -1 = none) are observable ONLY
    # in FO mode; in PO mode they ride the privileged channel (like
    # cure_*).
    _block_features_attached = [
        "attached_top", "attached_bottom", "attached_end_a", "attached_end_b"
    ]
    _block_features_tail = ["r", "g", "b"]
    _block_sim_features = [
        "id", "glue_top", "glue_end_a", "glue_end_b", "cure_top", "cure_end_a",
        "cure_end_b", "attached_top", "attached_bottom", "attached_end_a",
        "attached_end_b"
    ]
    _block_type = Type("block",
                       _block_features_common +
                       ["cure_top", "cure_end_a", "cure_end_b"] +
                       _block_features_attached + _block_features_tail,
                       sim_features=_block_sim_features,
                       angular_features=["roll", "pitch", "yaw"])
    _block_type_po = Type("block",
                          _block_features_common + _block_features_tail,
                          sim_features=_block_sim_features,
                          angular_features=["roll", "pitch", "yaw"])
    _bottle_type = Type("bottle", ["x", "y", "z", "rot", "is_held"],
                        sim_features=["id"],
                        angular_features=["rot"])
    _site_type = Type("site", ["x", "y", "z"], sim_features=["id"])

    @classmethod
    def _span_pool_size(cls) -> int:
        """Allocate bodies for either split without changing live class
        state."""
        counts = (CFG.bridge_train_span_blocks, CFG.bridge_test_span_blocks)
        if any(not isinstance(n, int) or n < 3 or n > 4 for n in counts):
            raise ValueError(
                "Bridge supports three or four span blocks per task")
        return max(counts)

    def __init__(self, use_gui: bool = False, **kwargs: Any) -> None:
        # In partial-observability mode, swap the block type to the
        # variant without `cure_*` *before* any blocks/predicates are
        # built, so the reduced schema propagates everywhere.
        if CFG.partially_observable:
            self._block_type = self._block_type_po

        # Robot
        self._robot = Object("robot", self._robot_type)

        # Blocks: n_legs + the span pool of the ONE shape, fixed names.
        self._legs = [
            Object(f"leg{i}", self._block_type) for i in range(self.n_legs)
        ]
        self._spans = [
            Object(f"span{i}", self._block_type)
            for i in range(self._span_pool_size())
        ]
        self._blocks: List[Object] = self._legs + self._spans
        self._block_index: Dict[str, int] = {
            blk.name: i
            for i, blk in enumerate(self._blocks)
        }

        # Glue bottle and the two leg sites.
        self._bottle = Object("bottle", self._bottle_type)
        self._sites = [Object(f"site{i}", self._site_type) for i in range(2)]

        # Glue-patch visual bodies: block name -> face -> body id.
        self._glue_patch_ids: Dict[str, Dict[str, int]] = {}

        super().__init__(use_gui, **kwargs)

    @property
    def types(self) -> Set[Type]:
        return {
            self._robot_type, self._block_type, self._bottle_type,
            self._site_type
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

        # Table
        table_id = create_object(asset_path="urdf/table.urdf",
                                 position=cls.table_pos,
                                 orientation=cls.table_orn,
                                 scale=1.0,
                                 use_fixed_base=True,
                                 physics_client_id=physics_client_id)
        bodies["table_id"] = table_id

        # Blocks: legs (standing shape) + spans (lying shape). The
        # counts MUST match the Object lists in __init__ -- the bodies
        # are zipped with the objects positionally. Every block is the
        # SAME box; legs are just blocks stood on end (orientation).
        block_ids = []
        for _ in range(cls.n_legs + cls._span_pool_size()):
            block_id = create_pybullet_block(
                color=(0.5, 0.5, 0.9, 1.0),
                half_extents=cls.block_half_extents,
                mass=cls.block_mass,
                friction=1.0,
                physics_client_id=physics_client_id)
            # Damp post-landing slide/twist (see pybullet_bond).
            p.changeDynamics(block_id,
                             -1,
                             spinningFriction=0.1,
                             rollingFriction=0.01,
                             physicsClientId=physics_client_id)
            block_ids.append(block_id)
        bodies["block_ids"] = block_ids

        # Glue bottle (slim box, top-graspable).
        bottle_id = create_pybullet_block(color=cls.bottle_color,
                                          half_extents=cls.bottle_half_extents,
                                          mass=0.05,
                                          friction=1.0,
                                          physics_client_id=physics_client_id)
        bodies["bottle_id"] = bottle_id

        # Two site pads (thin fixed plates marking the leg positions).
        site_ids = []
        for _ in range(2):
            site_id = create_pybullet_block(
                color=cls.site_color,
                half_extents=cls.site_half_extents,
                mass=0,
                friction=0.5,
                physics_client_id=physics_client_id)
            site_ids.append(site_id)
        bodies["site_ids"] = site_ids

        # Wet-glue visual patches: one per block face, collision-free
        # (baseCollisionShapeIndex=-1), parked out of view when dry.
        # PyBullet can't tint one face of a single-shape body, so these
        # carry the "this face is wet" rendering.
        patch_ids: List[List[int]] = []
        oov_x, oov_y = cls._out_of_view_xy
        hx, hy, hz = cls.block_half_extents
        for i in range(cls.n_legs + cls._span_pool_size()):
            per_face = []
            for face in GLUE_FACES:
                if face == "top":
                    patch_half = (hx - 0.001, hy - 0.001, 0.0015)
                else:
                    patch_half = (0.0015, hy - 0.001, hz - 0.001)
                vis_id = p.createVisualShape(p.GEOM_BOX,
                                             halfExtents=patch_half,
                                             rgbaColor=cls.glue_wet_color,
                                             physicsClientId=physics_client_id)
                patch_id = p.createMultiBody(baseMass=0,
                                             baseCollisionShapeIndex=-1,
                                             baseVisualShapeIndex=vis_id,
                                             basePosition=(oov_x, oov_y,
                                                           -1.0 - 0.1 * i),
                                             physicsClientId=physics_client_id)
                per_face.append(patch_id)
            patch_ids.append(per_face)
        bodies["glue_patch_ids"] = patch_ids

        return physics_client_id, pybullet_robot, bodies

    def _store_pybullet_bodies(self, pybullet_bodies: Dict[str, Any]) -> None:
        self._table_ids = [pybullet_bodies["table_id"]]
        self._robot.id = self._pybullet_robot.robot_id
        for i, blk in enumerate(self._blocks):
            blk.id = pybullet_bodies["block_ids"][i]
        self._bottle.id = pybullet_bodies["bottle_id"]
        for i, site in enumerate(self._sites):
            site.id = pybullet_bodies["site_ids"][i]
        self._glue_patch_ids = {
            blk.name:
            dict(zip(GLUE_FACES, pybullet_bodies["glue_patch_ids"][i]))
            for i, blk in enumerate(self._blocks)
        }

    # -------------------------------------------------------------------------
    # Small helpers
    # -------------------------------------------------------------------------
    def _own_block(self, blk: Object) -> Object:
        """This env's canonical instance of ``blk``, matched by name.

        Glue/cure/attached live in ``Object.sim_data``, which is stored
        on the INSTANCE. States routinely cross env instances (option-
        model resets, refinement rollouts, fresh test envs) carrying the
        source env's Object instances, so reading or writing sim_data
        through a state-derived block would silently share hidden glue
        state between envs. Every sim_data access therefore resolves to
        the env-owned instance first.
        """
        idx = self._block_index.get(blk.name)
        return self._blocks[idx] if idx is not None else blk

    def _attr(self, blk: Object, name: str, default: float) -> float:
        """Read a sim-feature attribute off this env's own instance.

        The None default is explicit because 0.0 is a meaningful value
        for attached_* (block index 0).
        """
        val = getattr(self._own_block(blk), name)
        return float(val) if val is not None else default

    def _set_attr(self, blk: Object, name: str, value: float) -> None:
        """Write a sim-feature attribute onto this env's own instance."""
        setattr(self._own_block(blk), name, value)

    @staticmethod
    def _block_rotation(state: State, blk: Object) -> np.ndarray:
        """World-from-local rotation matrix from the (roll, pitch, yaw)
        pose."""
        quat = p.getQuaternionFromEuler([
            state.get(blk, "roll"),
            state.get(blk, "pitch"),
            state.get(blk, "yaw")
        ])
        return np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)

    # Within this angular distance of the gimbal pole (|pitch| = pi/2)
    # the roll/yaw Euler split is numerically degenerate; canonical
    # reads fold roll into yaw there so a resting standing block's
    # features stay stable across snapshots.
    _GIMBAL_FOLD_BAND: ClassVar[float] = 0.02

    @classmethod
    def _canonical_block_orientation(
            cls, orn: Sequence[float]) -> Tuple[float, float, float]:
        """(roll, pitch, yaw) from a quaternion -- the FULL orientation.

        Away from the gimbal pole this is PyBullet's exact Euler
        extraction. Within ~1 deg of the pole the roll/yaw split is
        degenerate (only their combination is meaningful), so roll is
        folded to 0 and its contribution transferred into yaw -- the
        represented orientation changes by at most the pole distance.
        """
        roll, pitch, yaw = p.getEulerFromQuaternion(list(orn))
        if abs(abs(pitch) - np.pi / 2) < cls._GIMBAL_FOLD_BAND:
            # At the pole R = Rz(yaw -+ roll) @ Ry(+-pi/2); fold roll.
            if pitch < 0:  # pitch -> -pi/2
                yaw = float((yaw + roll + np.pi) % (2 * np.pi) - np.pi)
            else:  # pitch -> +pi/2
                yaw = float((yaw - roll + np.pi) % (2 * np.pi) - np.pi)
            roll = 0.0
        return float(roll), float(pitch), float(yaw)

    @staticmethod
    def _attached_value(state: State, blk: Object, slot: str) -> float:
        """``attached_<slot>`` from the state, falling back to the privileged
        channel when the feature is hidden (PO mode).

        Env-side code may read privileged state; the agent never sees
        it.
        """
        feat = f"attached_{slot}"
        if feat in blk.type.feature_names:
            return state.get(blk, feat)
        priv = state.privileged or {}
        return float(priv.get(blk.name, {}).get(feat, -1.0))

    # Face definitions in the BLOCK-LOCAL frame: normal axis index and
    # sign. ``top`` is the wide local +z face; ``end_a``/``end_b`` the
    # square local -x/+x faces. A standing leg (local +x up) therefore
    # presents its ``end_b`` face as its world-top.
    _FACE_AXES: ClassVar[Dict[str, Tuple[int, float]]] = {
        "top": (2, 1.0),
        "end_a": (0, -1.0),
        "end_b": (0, 1.0),
    }

    @classmethod
    def _face_world_dir(cls, state: State, blk: Object,
                        face: str) -> Tuple[float, float, float]:
        """Outward unit normal of a face in world frame."""
        axis, sign = cls._FACE_AXES[face]
        rmat = cls._block_rotation(state, blk)
        n = sign * rmat[:, axis]
        return (float(n[0]), float(n[1]), float(n[2]))

    # -------------------------------------------------------------------------
    # State Management
    # -------------------------------------------------------------------------
    def _get_object_ids_for_held_check(self) -> List[int]:
        ids = [blk.id for blk in self._blocks if blk.id is not None]
        if self._bottle.id is not None:
            ids.append(self._bottle.id)
        return ids

    def _get_domain_specific_feature(self, obj: Object, feature: str) -> float:
        if obj.type in (self._block_type, self._block_type_po):
            if feature.startswith("glue_") or feature.startswith("cure_"):
                return self._attr(obj, feature, 0.0)
            if feature.startswith("attached_"):
                return self._attr(obj, feature, -1.0)
            if feature == "half_x":
                return self.block_half_extents[0]
            if feature == "half_y":
                return self.block_half_extents[1]
            if feature == "half_z":
                return self.block_half_extents[2]
        raise ValueError(f"Unknown feature {feature} for object {obj}.")

    def _is_block(self, obj: Object) -> bool:
        return obj.type in (self._block_type, self._block_type_po)

    def _object_pose_matches_state(self,
                                   obj: Object,
                                   state: State,
                                   atol: float = 1e-3) -> bool:
        # Blocks: compare the orientation GEODESICALLY (the angle
        # between the state's rotation and the live one), never
        # axis-by-axis -- near the gimbal pole the roll/yaw split of
        # the same physical orientation can differ arbitrarily between
        # two valid Euler decompositions.
        if not self._is_block(obj):
            return super()._object_pose_matches_state(obj, state, atol)
        if obj.id is None:
            return True
        (px, py, pz), orn = p.getBasePositionAndOrientation(
            obj.id, physicsClientId=self._physics_client_id)
        for feat, live in (("x", px), ("y", py), ("z", pz)):
            if not np.isclose(state.get(obj, feat), live, atol=atol):
                return False
        state_orn = p.getQuaternionFromEuler([
            state.get(obj, "roll"),
            state.get(obj, "pitch"),
            state.get(obj, "yaw")
        ])
        diff = p.getDifferenceQuaternion(list(orn), list(state_orn))
        angle = 2.0 * float(np.arccos(np.clip(abs(diff[3]), -1.0, 1.0)))
        return bool(angle < 10 * atol)

    def _get_state(self, _render_obs: bool = False) -> State:
        """PyBullet -> State, plus the privileged (hidden) block.

        In partially-observable mode neither the ``cure_*`` nor the
        ``attached_*`` features are observable, so snapshot each block's
        true internal values into ``state.privileged`` -- the env-only
        channel the agent never sees -- so restoring a state restores
        its own hidden values too.
        """
        state = super()._get_state(_render_obs)
        # Canonical (pitch, yaw) for every block: the base class reads
        # raw Euler angles, which are degenerate for standing blocks
        # (pitch = +-pi/2), so recompute both from the quaternion.
        for blk in state.get_objects(self._block_type):
            if blk.id is None:
                continue
            orn = p.getBasePositionAndOrientation(
                blk.id, physicsClientId=self._physics_client_id)[1]
            roll, pitch, yaw = self._canonical_block_orientation(orn)
            state.set(blk, "roll", roll)
            state.set(blk, "pitch", pitch)
            state.set(blk, "yaw", yaw)
        if CFG.partially_observable:
            state.privileged = {
                blk.name: self._hidden_block_features(blk)
                for blk in state.get_objects(self._block_type)
            }
        return state

    def _hidden_block_features(self, blk: Object) -> Dict[str, float]:
        """One block's true ``cure_*``/``attached_*`` values, for the
        ``state.privileged`` snapshot in partially-observable mode."""
        feats = {
            f"cure_{face}": self._attr(blk, f"cure_{face}", 0.0)
            for face in GLUE_FACES
        }
        feats.update({
            f"attached_{slot}": self._attr(blk, f"attached_{slot}", -1.0)
            for slot in ATTACH_SLOTS
        })
        return feats

    def _set_domain_specific_state(self, state: State) -> None:
        # Restore each block's internal glue / cure / attachment state.
        blocks = state.get_objects(self._block_type)
        for blk in blocks:
            for face in GLUE_FACES:
                self._set_attr(blk, f"glue_{face}",
                               state.get(blk, f"glue_{face}"))
                if f"cure_{face}" in blk.type.feature_names:
                    self._set_attr(blk, f"cure_{face}",
                                   state.get(blk, f"cure_{face}"))
                else:
                    priv = state.privileged or {}
                    self._set_attr(
                        blk, f"cure_{face}",
                        float(priv.get(blk.name, {}).get(f"cure_{face}", 0.0)))
            for slot in ATTACH_SLOTS:
                self._set_attr(blk, f"attached_{slot}",
                               self._attached_value(state, blk, slot))
            # Colors are task-assigned features; the base env never
            # writes them to PyBullet, so apply them here.
            if blk.id is not None:
                update_object(blk.id,
                              color=(state.get(blk, "r"), state.get(blk, "g"),
                                     state.get(blk, "b"), 1.0),
                              physics_client_id=self._physics_client_id)

        # Wet-glue patch visuals.
        self._update_glue_patches(state)

        # Move irrelevant blocks out of view.
        oov_x, oov_y = self._out_of_view_xy
        in_state = set(blocks)
        for i, blk in enumerate(self._blocks):
            if blk not in in_state and blk.id is not None:
                update_object(blk.id,
                              position=(oov_x + 0.3 * i, oov_y, 0.0),
                              physics_client_id=self._physics_client_id)

    def _update_glue_patches(self, state: State) -> None:
        """Show a yellow patch on each wet face; park all other patches out of
        view.

        Patches are visual-only bodies.
        """
        oov_x, oov_y = self._out_of_view_xy
        in_state = set(state.get_objects(self._block_type))
        for i, blk in enumerate(self._blocks):
            for j, face in enumerate(GLUE_FACES):
                patch_id = self._glue_patch_ids[blk.name][face]
                wet = blk in in_state and \
                    self._attr(blk, f"glue_{face}", 0.0) > 0.5
                if not wet:
                    update_object(patch_id,
                                  position=(oov_x + 0.3 * i, oov_y + 0.3 * j,
                                            -1.0),
                                  physics_client_id=self._physics_client_id)
                    continue
                x = state.get(blk, "x")
                y = state.get(blk, "y")
                z = state.get(blk, "z")
                axis, _ = self._FACE_AXES[face]
                dx_dir, dy_dir, dz_dir = self._face_world_dir(state, blk, face)
                offset = self.block_half_extents[axis] + 0.0015
                pos = (x + dx_dir * offset, y + dy_dir * offset,
                       z + dz_dir * offset)
                # The patch shares the block's full orientation (its
                # slab geometry is defined in the block-local frame).
                update_object(patch_id,
                              position=pos,
                              orientation=p.getQuaternionFromEuler([
                                  state.get(blk, "roll"),
                                  state.get(blk, "pitch"),
                                  state.get(blk, "yaw")
                              ]),
                              physics_client_id=self._physics_client_id)
