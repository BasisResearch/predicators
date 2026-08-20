"""A PyBullet environment where the robot builds an "n"-shaped bridge by gluing
rectangular blocks with a pickable glue bottle.

Motivating the partial-observability + slow-process story of the
sim-learning arms (``agent_sim_predicate_invention`` under
``CFG.partially_observable``) with a hidden process whose
consequence is *kinematic* rather than a feature readout: once a glue
joint cures, the two blocks are welded into one rigid assembly (a
body-to-body ``JOINT_FIXED`` constraint), so picking any block of the
assembly transports the whole thing. No other domain in the suite makes
the hidden latent change what actions *do*.

The lateral glue is also structurally necessary: the span row is three
blocks long but only its outer blocks sit over legs, so the unglued
middle span has no support and falls straight into the gap (verified
by idle-sim probes; a 2-block row was rejected because the two spans
mutually support as a friction arch, and single-span overhangs are
damped by rolling friction). Every joint the task requires is thus a
physical necessity, never a goal decoration: seat joints (span-to-leg)
are neither structural nor in the goal -- gluing a leg top and seating
onto it still cures and welds (latent dynamics an agent may discover),
but no task calls for it.

Mechanics:

- Every block is the SAME 10x5x5 box; a "leg" is a block stood on end
  (pitch = -pi/2), a "span" one lying flat. Orientation is honest pose
  -- there are no per-role shapes or shape features.
- Every block exposes three glue-able faces in its LOCAL frame:
  ``top`` (+z) and the two long-axis ends ``end_a``/``end_b`` (-x/+x).
  A standing leg therefore presents its ``end_b`` face as its
  world-top -- gluing "the leg's top" and gluing "a row end" are the
  same face mechanics at different orientations.
- The robot picks up the glue ``bottle``, holds its tip near a face's
  dab point to wet that face, and puts the bottle back down.
- While a wet face is in aligned resting contact with another block
  (neither block held), that joint's hidden ``cure_*`` counter ticks;
  at ``cure_threshold`` the joint irreversibly latches: the wet glue is
  consumed, both blocks record the attachment (``attached_*`` = partner
  block index), and a physical weld constraint is created.
- Interrupting the contact resets the counter (wet glue persists).

Task ("n"-shaped bridge, 5 blocks): stand one leg block at each marked
site, glue three span blocks end-to-end on the table, then seat the
cured span assembly across the legs. 2 joints (the lateral row welds).

In partially-observable mode (``CFG.partially_observable``) both the
``cure_*`` counters and the ``attached_*`` slots are dropped from the
observation (no real perception system reports "attached to block 3");
the agent sees only wet-glue flags, poses, and the kinematic
consequences, and must postulate BOTH the hidden dwell process and
attachment itself as a latent relation inferred from co-motion.

Example command (oracle demo via bilevel process planning)::

    python predicators/main.py --env pybullet_bridge \
        --approach oracle_process_planning --seed 0 \
        --num_train_tasks 0 --num_test_tasks 5 \
        --sesame_check_expected_atoms False
"""

from typing import Any, ClassVar, Dict, FrozenSet, List, Optional, Sequence, \
    Set, Tuple

import numpy as np
import pybullet as p

from predicators import utils
from predicators.envs.pybullet_env import PyBulletEnv
from predicators.pybullet_helpers.geometry import Pose3D, Quaternion
from predicators.pybullet_helpers.objects import create_object, \
    create_pybullet_block, update_object
from predicators.pybullet_helpers.robots import SingleArmPyBulletRobot
from predicators.settings import CFG
from predicators.structs import Action, DerivedPredicate, EnvironmentTask, \
    GroundAtom, Object, Predicate, State, Type

# Faces that can be wetted with glue (block-local frame).
GLUE_FACES = ("top", "end_a", "end_b")
# Attachment slots: a cured joint occupies one slot on each block.
# ``bottom`` exists because a stack/seat joint welds a wet ``top`` to
# the underside of the partner.
ATTACH_SLOTS = ("top", "bottom", "end_a", "end_b")


class PyBulletBridgeEnv(PyBulletEnv):
    """Build an n-shaped bridge by gluing blocks; cured joints physically weld
    blocks into rigid assemblies.

    Two block schemas, one logical type: the fully-observable
    ``_block_type`` carries the ``cure_*`` counters and ``attached_*``
    slots as observable features, while ``_block_type_po`` drops both
    (they ride ``state.privileged`` instead). Both keep them as
    ``sim_features`` so the ``block.cure_top`` (etc.) Python attributes
    -- the internal source of truth -- always drive the dynamics.
    ``__init__`` swaps the type when ``CFG.partially_observable`` is
    set.
    """

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
    # Single source of truth for the block roster: __init__'s Object
    # lists and initialize_pybullet's body creation both read these.
    n_legs: ClassVar[int] = 2
    n_spans: ClassVar[int] = 3
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
    # Sites (where the legs stand) sit mid-table so the finished
    # bridge stands centered, with staging split around it: a
    # two-row front band and a back row, none overlapping the bridge
    # band (sites + seated span row, y in [1.275, 1.325]) in y. The
    # back row cannot sit deeper than ~1.38: the validated-IK radial
    # reach cap (0.78 from the base at y=0.65) tops out at y~1.43.
    site_y: ClassVar[float] = 1.30
    # Site separation = n_spans * span_len - 2 * leg_half_x, so the
    # butted 3-span row's outer ends land flush over the legs' outer
    # edges when the legs stand exactly at the sites (the 3-span row's
    # unsupported middle is what makes glue structurally necessary).
    site_sep: ClassVar[float] = 0.25
    site_x_jitter: ClassVar[float] = 0.05
    # A tiny nominal gap for lateral butt joints so a placed block does
    # not shove its neighbor.
    lateral_place_gap: ClassVar[float] = 0.002
    # EE transport height for carried blocks/assemblies. Must clear the
    # standing legs plus a carried span row.
    transport_z: ClassVar[float] = 0.70

    # -------------------------------------------------------------------------
    # Domain-specific config
    # -------------------------------------------------------------------------
    # Consecutive aligned-contact steps for a wet joint to cure. Must
    # comfortably exceed the Place option's own duration (~14 steps):
    # if curing completes during the Place retreat, the subsequent Wait
    # starts with its target atom already true, its first action is a
    # no-op, and the option model's repeat-state check kills it.
    cure_threshold: ClassVar[int] = 25
    # Bottle-tip proximity to a face's dab point that wets the face.
    # Only the single nearest in-range face is wetted per step, so
    # neighboring dab points (>= 2.5 cm apart) don't double-wet.
    apply_glue_radius: ClassVar[float] = 0.02
    # Consecutive in-range steps required to wet a face. Wetting used
    # to be instantaneous, so a one-step drive-by crossing of the
    # radius (e.g. a bottle retreat clipping the sphere on its way up)
    # could wet a face -- a step-phasing coin flip that let marginal
    # glue targets validate in the sandbox and then miss for real.
    # Requiring a sustained dwell makes grazes fail deterministically
    # everywhere. The streak rides IN the glue_* feature as partials of
    # _WET_PARTIAL per step (kept <= 0.5 so every "is wet" reader --
    # classifiers, cure gate, patch visuals -- still sees a dry face),
    # so it round-trips through _set_state like any other feature.
    wet_streak_steps: ClassVar[int] = 3
    _WET_PARTIAL: ClassVar[float] = 0.2
    # Dab points hover this far off the face surface.
    dab_margin: ClassVar[float] = 0.005
    # Stacking tolerances for the top-face cure detector (leg-on-leg).
    stack_align_tol: ClassVar[float] = 0.025
    stack_z_tol: ClassVar[float] = 0.02
    # Lateral butt-joint window for NextToEnd(right, left): projection
    # of the center offset onto the end direction, minus the two half
    # lengths, must land in [-0.01, +0.012] (contact up to a ~1 cm gap).
    lateral_proj_tol_lo: ClassVar[float] = 0.01
    lateral_proj_tol_hi: ClassVar[float] = 0.012
    # Perp/y tolerances must exceed real place-execution accuracy: a
    # span dropped ~1.5 cm can land ~2 cm off in y, and a lateral
    # placement error is FROZEN into the weld, shifting where the far
    # span meets its leg. A 3 cm miss still leaves 2 cm of overlap on
    # the 5 cm leg top, so the joint is physically sound; a 2 cm gate
    # left built bridges with one seat joint that could never cure.
    lateral_perp_tol: ClassVar[float] = 0.03
    lateral_z_tol: ClassVar[float] = 0.015
    # Seat tolerances for SeatedOn(span, leg).
    seat_x_window: ClassVar[float] = 0.045
    seat_y_tol: ClassVar[float] = 0.035
    seat_z_tol: ClassVar[float] = 0.02
    # AtSite xy tolerance (plus a z check that the block rests on the
    # table, so a stacked upper leg is not "at" the site). The site pad
    # is 9 cm wide; 4 cm absorbs placement error plus the nudge the
    # seat landing gives the legs (a marginal AtSite flicking false
    # mid-episode sends the replanner after unreachable re-placements).
    at_site_tol: ClassVar[float] = 0.04
    at_site_z_tol: ClassVar[float] = 0.02
    # Weld constraint strength. PyBullet's default (500, the same the
    # grasp constraint uses) already holds in probes; the high value
    # removes sag under a cantilevered span.
    weld_max_force: ClassVar[float] = 10000.0
    # Staging slots (see _stage_objects). Random rejection sampling
    # cannot pack the 6 staged objects + assembly strip + site keepouts
    # into the reachable lens (it saturates around 5 objects), so
    # staging assigns objects to a jittered grid instead: 7 columns x 3
    # rows, filtered by reach, the assembly strip, grasp-clearance
    # row-adjacency, and the behind-site keepouts (the outer columns
    # only reach the front/mid rows). The span row is assembled along
    # the front row.
    stage_cols: ClassVar[Tuple[float, ...]] = (0.41, 0.52, 0.63, 0.74, 0.85,
                                               0.96, 1.07)
    stage_row_front: ClassVar[float] = 1.14  # span row assembled here
    stage_row_mid: ClassVar[float] = 1.24  # front band, second row
    stage_row_back: ClassVar[float] = 1.38  # behind the bridge band
    stage_jitter: ClassVar[float] = 0.008
    site_keepout: ClassVar[float] = 0.09
    reach_radius: ClassVar[float] = 0.78
    # Manipulation-workspace x band. The option parameter boxes
    # (options.py) are built from these, and _GroundCausalProcess
    # CLIPS sampled params into the box -- so any sampler target that
    # can exceed the band silently becomes a different (usually
    # colliding) goal. Staging must keep every downstream place/glue
    # target inside it.
    workspace_x_lo: ClassVar[float] = 0.4
    workspace_x_hi: ClassVar[float] = 1.1
    # Worst-case x the assembly strip's samplers reach past span0's
    # column: span0's slot jitter plus the per-joint U(0, 0.004)
    # next-to slack for the two lateral joints.
    strip_x_slack: ClassVar[float] = 0.02

    # Colors
    # Fixed muted slate for the site pads (reads as a "marked spot" on
    # the pale wood table without shouting).
    site_color: ClassVar[Tuple[float, float, float,
                               float]] = (0.42, 0.47, 0.53, 1.0)
    # Block colors are drawn per task from role families chosen to sit
    # well together on the wood table: legs from cool blues/teals,
    # spans from warm terracotta/amber. Within a task the draws are
    # without replacement, so same-role blocks stay distinguishable.
    leg_color_family: ClassVar[Tuple[Tuple[float, float, float], ...]] = (
        (0.30, 0.45, 0.69),  # muted cobalt
        (0.24, 0.57, 0.63),  # deep teal
        (0.47, 0.56, 0.75),  # dusty periwinkle
        (0.36, 0.64, 0.72),  # steel cyan
    )
    span_color_family: ClassVar[Tuple[Tuple[float, float, float], ...]] = (
        (0.80, 0.44, 0.32),  # terracotta
        (0.88, 0.63, 0.33),  # amber
        (0.72, 0.35, 0.38),  # rosewood
        (0.85, 0.53, 0.42),  # clay
    )
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
    # `attached_*` (partner block index, -1 if none) and `cure_*` (the
    # per-joint dwell counters) are observable in FO mode but dropped
    # from the PO type -- there they ride ``state.privileged`` and the
    # agent must postulate attachment as a latent relation from
    # co-motion. All stay Python attributes (the internal source of
    # truth) so they always drive the dynamics.
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
    # them to compute face centers, dab points, and touch spacings
    # without probing the physics for block dimensions.
    _block_features_common = [
        "x", "y", "z", "roll", "pitch", "yaw", "half_x", "half_y", "half_z",
        "is_held", "glue_top", "glue_end_a", "glue_end_b"
    ]
    # attached_* (partner block index, -1 = none) are observable ONLY
    # in FO mode: no real perception system emits "attached to block
    # 3" -- attachment is inferred from co-motion. In PO mode they ride
    # the privileged channel (like cure_*), and a world-model learner
    # must postulate attachment as a LATENT relation whose only
    # observable footprint is kinematic.
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

    def __init__(self, use_gui: bool = False, **kwargs: Any) -> None:
        # In partial-observability mode, swap the block type to the
        # variant without `cure_*` *before* any blocks/predicates are
        # built, so the reduced schema propagates everywhere.
        if CFG.partially_observable:
            self._block_type = self._block_type_po

        # Robot
        self._robot = Object("robot", self._robot_type)

        # Blocks: n_legs + n_spans of the ONE shape, fixed names.
        self._legs = [
            Object(f"leg{i}", self._block_type) for i in range(self.n_legs)
        ]
        self._spans = [
            Object(f"span{i}", self._block_type) for i in range(self.n_spans)
        ]
        self._blocks: List[Object] = self._legs + self._spans
        self._block_index: Dict[str, int] = {
            blk.name: i
            for i, blk in enumerate(self._blocks)
        }

        # Glue bottle and the two leg sites.
        self._bottle = Object("bottle", self._bottle_type)
        self._sites = [Object(f"site{i}", self._site_type) for i in range(2)]

        # Live weld constraints: frozenset({body_id_a, body_id_b}) ->
        # PyBullet constraint id. Must exist before super().__init__
        # (reset paths may call _set_domain_specific_state).
        self._weld_constraints: Dict[FrozenSet[int], int] = {}
        # Per-weld creation arguments (parent, child, ideal_dz), kept so
        # a resting weld can be re-anchored (see _relax_resting_welds).
        self._weld_meta: Dict[FrozenSet[int], Tuple[int, int,
                                                    Optional[float]]] = {}
        # Live wet-glue tacks (see _sync_wet_joint_tacks):
        # frozenset({body_id_a, body_id_b}) -> constraint id.
        self._tack_constraints: Dict[FrozenSet[int], int] = {}
        # Glue-patch visual bodies: block name -> face -> body id.
        self._glue_patch_ids: Dict[str, Dict[str, int]] = {}

        super().__init__(use_gui, **kwargs)

        # Predicates
        self._HandEmpty = Predicate("HandEmpty", [self._robot_type],
                                    self._HandEmpty_holds)
        self._Holding = Predicate("Holding",
                                  [self._robot_type, self._block_type],
                                  self._Holding_holds)
        self._HoldingBottle = Predicate("HoldingBottle",
                                        [self._robot_type, self._bottle_type],
                                        self._HoldingBottle_holds)
        # Only end_b gets a named predicate: the task's joints are all
        # lateral row joints built left-to-right (glue goes on the row
        # end's end_b face). The env's PHYSICS stays generic over all
        # three faces -- wetting/curing a top or end_a face still works
        # and welds (discoverable latent dynamics) -- there is just no
        # abstract-model vocabulary for it.
        self._GlueEndB = Predicate("GlueEndB", [self._block_type],
                                   self._make_glue_holds("end_b"))
        self._NextToEnd = Predicate("NextToEnd",
                                    [self._block_type, self._block_type],
                                    self._NextToEnd_holds)
        self._SeatedOn = Predicate("SeatedOn",
                                   [self._block_type, self._block_type],
                                   self._SeatedOn_holds)
        self._AtSite = Predicate("AtSite", [self._block_type, self._site_type],
                                 self._AtSite_holds)
        self._SiteFree = Predicate("SiteFree", [self._site_type],
                                   self._SiteFree_holds)
        self._Attached = Predicate("Attached",
                                   [self._block_type, self._block_type],
                                   self._Attached_holds)
        # Static shape predicates (planning-time grounding pruners) and
        # Loose (no cured attachments -- a block welded into an
        # assembly cannot be individually re-placed).
        self._Standing = Predicate("Standing", [self._block_type],
                                   lambda s, o: self._stands(s, o[0]))
        self._Lying = Predicate("Lying", [self._block_type],
                                lambda s, o: not self._stands(s, o[0]))
        self._Loose = Predicate("Loose", [self._block_type], self._Loose_holds)
        # Resting = not held. Pick processes delete it and place
        # processes re-add it, so the cure processes can require it
        # throughout their delay: picking a block mid-cure abstractly
        # aborts the cure, exactly matching the env's counter reset.
        self._Resting = Predicate("Resting", [self._block_type],
                                  lambda s, o: s.get(o[0], "is_held") <= 0.5)
        # TopFree = nothing rests on the block's top face. Gates
        # PickBlockFromTable: a block with something seated or stacked
        # on it cannot be top-grasped.
        self._TopFree = Predicate("TopFree", [self._block_type],
                                  self._TopFree_holds)
        # EndsFree = no UNWELDED block butts either end of this block.
        # DERIVED (recomputed from NextToEnd/Attached atoms every
        # abstract state) and required by PickBlockFromTable: without
        # it, picking a block out of an uncured butt joint leaves the
        # NextToEnd atom stale-true (no delete effect can name the
        # neighbor), and the planner exploits the frame bug -- it butted
        # span2 against STAGED span1, moved span1 into the row, and
        # counted on the fictional span1-span2 joint still curing.
        # Welded neighbors do not break EndsFree: picking them drags the
        # whole assembly, so the adjacency physically survives the pick
        # (and Loose separately forbids re-placing welded blocks).
        # Dismantling an uncured joint goes through PickSpanFromRow,
        # which deletes the adjacency it names.
        self._EndsFree = DerivedPredicate(
            "EndsFree", [self._block_type],
            self._EndsFree_holds_from_atoms,
            auxiliary_predicates={self._NextToEnd, self._Attached})

    @classmethod
    def get_name(cls) -> str:
        return "pybullet_bridge"

    @property
    def predicates(self) -> Set[Predicate]:
        return {
            self._HandEmpty, self._Holding, self._HoldingBottle,
            self._GlueEndB, self._NextToEnd, self._SeatedOn, self._AtSite,
            self._SiteFree, self._Attached, self._Standing, self._Lying,
            self._Loose, self._Resting, self._TopFree, self._EndsFree
        }

    @property
    def goal_predicates(self) -> Set[Predicate]:
        # Goals pin the full geometric layout (not just Attached): the
        # extra atoms force the planner's bindings to a physically
        # consistent left-to-right build (see processes.py docstring)
        # and all of them persist in the finished bridge.
        return {self._Attached, self._AtSite, self._NextToEnd, self._SeatedOn}

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
        for _ in range(cls.n_legs + cls.n_spans):
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
        for i in range(cls.n_legs + cls.n_spans):
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

    @classmethod
    def _is_leg_shaped(cls, blk: Object) -> bool:
        """Task-generation ROLE by name (which blocks start standing).

        Only task gen may dispatch on this; all live geometry reads the
        block's orientation from the state instead.
        """
        return blk.name.startswith("leg")

    @staticmethod
    def _stands(state: State, blk: Object) -> bool:
        """Long axis vertical (the leg pose; spans lie flat)."""
        return abs(state.get(blk, "pitch")) > np.pi / 4

    @classmethod
    def _world_half_extents(cls, state: State,
                            blk: Object) -> Tuple[float, float, float]:
        """The block's world-axis-aligned half extents at its current
        orientation family (standing swaps the long axis into z)."""
        return cls.leg_half_extents if cls._stands(state, blk) \
            else cls.span_half_extents

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

    @classmethod
    def _ideal_block_orientation(
            cls, orn: Sequence[float]) -> Tuple[float, float, float, float]:
        """The nearest axis-aligned rest orientation: canonical roll and pitch
        snapped to the closest multiple of pi/2, yaw kept free."""
        roll, pitch, yaw = cls._canonical_block_orientation(orn)
        half_pi = np.pi / 2
        roll = round(roll / half_pi) * half_pi
        pitch = round(pitch / half_pi) * half_pi
        return p.getQuaternionFromEuler([roll, pitch, yaw])

    @staticmethod
    def _attached_value(state: State, blk: Object, slot: str) -> float:
        """``attached_<slot>`` from the state, falling back to the privileged
        channel when the feature is hidden (PO mode).

        Env-side classifiers (goal checking, Loose/Attached gates) are
        ground truth and may read privileged state; the agent never sees
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

    @classmethod
    def _face_dab_point(cls, state: State, blk: Object,
                        face: str) -> Tuple[float, float, float]:
        """Where the bottle tip must hover to wet this face.

        Upward faces (normal within 45 deg of +z, e.g. a lying span's
        top or a standing leg's upper end): just above the face center.
        Vertical faces (a lying span's ends): just above the face's top
        edge, so the dab always comes from above (no sideways IK).
        Downward faces get a point below the block -- never reachable,
        so they are effectively un-dabbable. Classmethod so option-layer
        code (options.py) can share the exact geometry.
        """
        axis, sign = cls._FACE_AXES[face]
        rmat = cls._block_rotation(state, blk)
        half = cls.block_half_extents
        pos = np.array(
            [state.get(blk, "x"),
             state.get(blk, "y"),
             state.get(blk, "z")])
        n = sign * rmat[:, axis]
        center = pos + n * half[axis]
        if n[2] > np.cos(np.pi / 4):
            dab = center + np.array([0.0, 0.0, cls.dab_margin])
        else:
            # Vertical extent of the face = the larger world-z reach of
            # its two spanning local axes.
            span_axes = [i for i in range(3) if i != axis]
            v_half = max(abs(rmat[2, i]) * half[i] for i in span_axes)
            dab = np.array(
                [center[0], center[1], pos[2] + v_half + cls.dab_margin])
        return (float(dab[0]), float(dab[1]), float(dab[2]))

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

        In partially-observable mode neither the ``cure_*`` counters
        nor the ``attached_*`` slots are observable features, so
        snapshot each block's true internal values into
        ``state.privileged`` -- the env-only channel the agent never
        sees -- so backtracking restores each search node's own hidden
        state (and the weld sync can rebuild constraints from it).
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

        # Sync physical weld constraints to the restored attachment
        # features. Handles planner backtracking to pre-cure nodes and
        # cross-episode residuals (a fresh task has all attached = -1,
        # so every stale weld is removed). HIDDEN-SEMANTICS gate: the
        # base sim (skip_residual_dynamics=True, the agent-visible
        # simulator) must not know that attached_* features mean a
        # rigid weld -- that kinematic consequence is part of what a
        # sim-learning agent has to reproduce itself (by emitting
        # Attach physics commands). The full env (oracle planning and
        # its option models) keeps the sync.
        if not self._skip_domain_specific_dynamics:
            self._sync_welds_to_state(state)

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

    # -------------------------------------------------------------------------
    # Weld constraint lifecycle
    # -------------------------------------------------------------------------
    def _create_weld(self,
                     body_a: int,
                     body_b: int,
                     ideal_dz: Optional[float] = None) -> None:
        """Create a body-to-body JOINT_FIXED weld at the CURRENT relative pose
        (see pybullet_coffee's plugged-in constraint).

        The relative transform is SNAPPED before freezing: each block's
        orientation is idealized to the nearest axis-aligned rest pose
        (roll/pitch to multiples of pi/2, yaw free), and when
        ``ideal_dz`` is given (the joint's nominal vertical offset,
        known from the attachment slot) the WORLD z-offset is set to
        it. An unsnapped weld freezes a millimeter-level inconsistency
        against the resting plane, and the constraint solver then
        applies steady micro-forces that make the welded assembly CREEP
        across the table (~2 cm over a few hundred idle steps),
        drifting it out of the seat gates.

        The snap MUST happen in world coordinates before the transform
        is expressed in the parent's frame: both the flatness ideal and
        ideal_dz are world-frame concepts, and editing the local
        components directly is only equivalent when the parent lies
        flat. With a STANDING parent (a seat joint's leg) the local z
        axis is horizontal, and the local edit re-poses the child by
        centimeters and ~pi/2 - the solver then hurls the assembly off
        the table at weld_max_force.
        """
        key = frozenset({body_a, body_b})
        if key in self._weld_constraints:
            return
        pos_a, orn_a = p.getBasePositionAndOrientation(
            body_a, physicsClientId=self._physics_client_id)
        pos_b, orn_b = p.getBasePositionAndOrientation(
            body_b, physicsClientId=self._physics_client_id)
        orn_a_ideal = self._ideal_block_orientation(orn_a)
        orn_b_ideal = self._ideal_block_orientation(orn_b)
        world_off = (pos_b[0] - pos_a[0], pos_b[1] - pos_a[1],
                     pos_b[2] - pos_a[2] if ideal_dz is None else ideal_dz)
        inv_pos, inv_orn = p.invertTransform((0.0, 0.0, 0.0), orn_a_ideal)
        rel_pos, _ = p.multiplyTransforms(inv_pos, inv_orn, world_off,
                                          (0.0, 0.0, 0.0, 1.0))
        _, rel_orn = p.multiplyTransforms((0.0, 0.0, 0.0), inv_orn,
                                          (0.0, 0.0, 0.0), orn_b_ideal)
        # Teleport the child onto the EXACT pose the constraint will
        # enforce (parent's ACTUAL frame composed with the snapped
        # relative transform) and zero both bodies' velocities, so the
        # constraint starts with zero error. Without this, the solver
        # spends every subsequent step pulling the pair toward the
        # snapped frame while table contact resists, and the rectified
        # micro-vibration SKATES the welded assembly across the table
        # (measured 2-10 mm and up to 0.08 rad yaw per 200 idle steps;
        # unwelded pairs move < 1 mm). The teleport is mm/mrad scale --
        # exactly the snap distance.
        child_pos, child_orn = p.multiplyTransforms(pos_a, orn_a, rel_pos,
                                                    rel_orn)
        p.resetBasePositionAndOrientation(
            body_b,
            child_pos,
            child_orn,
            physicsClientId=self._physics_client_id)
        for body in (body_a, body_b):
            p.resetBaseVelocity(body, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0),
                                physicsClientId=self._physics_client_id)
        # Welded partners must not collide with each other: the box
        # collision margin keeps the flush faces in permanent contact,
        # and the contact solver fighting the weld is the other motor
        # of the same skating creep. Re-enabled in _remove_weld.
        p.setCollisionFilterPair(body_a,
                                 body_b,
                                 -1,
                                 -1,
                                 0,
                                 physicsClientId=self._physics_client_id)
        cid = p.createConstraint(parentBodyUniqueId=body_a,
                                 parentLinkIndex=-1,
                                 childBodyUniqueId=body_b,
                                 childLinkIndex=-1,
                                 jointType=p.JOINT_FIXED,
                                 jointAxis=[0, 0, 0],
                                 parentFramePosition=rel_pos,
                                 parentFrameOrientation=rel_orn,
                                 childFramePosition=[0, 0, 0],
                                 childFrameOrientation=[0, 0, 0, 1],
                                 physicsClientId=self._physics_client_id)
        # The default maxForce sags under a cantilevered span.
        p.changeConstraint(cid,
                           maxForce=self.weld_max_force,
                           physicsClientId=self._physics_client_id)
        self._weld_constraints[key] = cid
        self._weld_meta[key] = (body_a, body_b, ideal_dz)

    def _desired_weld_pairs(
            self,
            state: State) -> Dict[FrozenSet[int], Tuple[int, int, float]]:
        """Weld pairs implied by the attachment features, as key ->
        (parent_body, child_body, ideal_dz).

        The joint's nominal vertical offset is snapped from the CURRENT
        relative pose to the nearest ideal: 0 (a coplanar lateral butt
        joint) or +-(sum of the two world half-heights) (a vertical
        joint, e.g. a span seated on a standing leg's top end).
        """
        pairs: Dict[FrozenSet[int], Tuple[int, int, float]] = {}
        for blk in state.get_objects(self._block_type):
            for slot in ATTACH_SLOTS:
                idx = int(round(self._attr(blk, f"attached_{slot}", -1.0)))
                if idx < 0:
                    continue
                partner = self._blocks[idx]
                if blk.id is None or partner.id is None:
                    continue
                key = frozenset({blk.id, partner.id})
                if key in pairs:
                    continue
                actual_dz = state.get(partner, "z") - state.get(blk, "z")
                stack_dz = self._world_half_extents(state, blk)[2] + \
                    self._world_half_extents(state, partner)[2]

                def _gap(ideal: float, ref: float = actual_dz) -> float:
                    return abs(ref - ideal)

                dz = min((0.0, stack_dz, -stack_dz), key=_gap)
                pairs[key] = (blk.id, partner.id, dz)
        return pairs

    # Wet glue is tacky: while a joint is wet and its faces are in
    # aligned contact, the pair is held together by a weak constraint
    # (under a newton, against the weld's ten thousand). It does not
    # stop the impulse the arm leaves behind when it releases and
    # retreats -- momentum is momentum -- but it makes the joint absorb
    # that impulse as a UNIT instead of coming apart: a placement that
    # ended flush against its neighbor was observed to fling an
    # already-placed span ~5 cm and ~90 degrees during the cure wait
    # (~30% of flush placements), which forced agents onto a narrow
    # 3-8 mm assembly gap, wide enough to survive the release and
    # narrow enough to still cure.
    #
    # The force is deliberately held below a block's own weight
    # (block_mass * g ~ 1 N), so a wet joint can never lift, carry or
    # drag its neighbour: everything the arm does deliberately still
    # wins, and picking a block mid-cure aborts the cure exactly as the
    # abstract model says. The tack is replaced by the rigid weld the
    # moment the joint latches.
    wet_joint_tack_force: ClassVar[float] = 0.5  # newtons

    def _sync_wet_joint_tacks(self, curing: Set[FrozenSet[int]]) -> None:
        """Make the live tack set match the currently curing joints."""
        for key in list(self._tack_constraints):
            if key not in curing:
                self._drop_tack(key)
        for key in curing:
            if key in self._tack_constraints or key in self._weld_constraints:
                continue
            if self._held_obj_id is not None and self._held_obj_id in key:
                continue
            body_a, body_b = sorted(key)
            pos_a, orn_a = p.getBasePositionAndOrientation(
                body_a, physicsClientId=self._physics_client_id)
            pos_b, orn_b = p.getBasePositionAndOrientation(
                body_b, physicsClientId=self._physics_client_id)
            inv_pos, inv_orn = p.invertTransform(pos_a, orn_a)
            rel_pos, rel_orn = p.multiplyTransforms(inv_pos, inv_orn, pos_b,
                                                    orn_b)
            # Anchored at the CURRENT relative pose, so the tack holds
            # the joint as assembled instead of pulling it anywhere.
            cid = p.createConstraint(parentBodyUniqueId=body_a,
                                     parentLinkIndex=-1,
                                     childBodyUniqueId=body_b,
                                     childLinkIndex=-1,
                                     jointType=p.JOINT_FIXED,
                                     jointAxis=[0, 0, 0],
                                     parentFramePosition=rel_pos,
                                     parentFrameOrientation=rel_orn,
                                     childFramePosition=[0, 0, 0],
                                     childFrameOrientation=[0, 0, 0, 1],
                                     physicsClientId=self._physics_client_id)
            p.changeConstraint(cid,
                               maxForce=self.wet_joint_tack_force,
                               physicsClientId=self._physics_client_id)
            self._tack_constraints[key] = cid

    def _drop_tack(self, key: FrozenSet[int]) -> None:
        """Remove one wet-glue tack, if it exists."""
        cid = self._tack_constraints.pop(key, None)
        if cid is not None:
            p.removeConstraint(cid, physicsClientId=self._physics_client_id)

    def _sync_welds_to_state(self, state: State) -> None:
        """Make the live constraint set match the attachment features.

        Object poses have already been restored by _set_state, so
        missing welds are created at the restored relative poses.
        Persisting welds keep their original constraint (the restored
        poses satisfy it by construction).
        """
        for key in list(self._tack_constraints):
            # Tacks are anchored to the poses they were created at; a
            # restored state is a different scene.
            self._drop_tack(key)
        desired = self._desired_weld_pairs(state)
        for key in list(self._weld_constraints):
            if key not in desired:
                self._remove_weld(key)
        for key, (body_a, body_b, ideal_dz) in desired.items():
            if key not in self._weld_constraints:
                self._create_weld(body_a, body_b, ideal_dz=ideal_dz)

    # Quiescence gates for weld re-anchoring (see _relax_resting_welds):
    # creep velocities are ~0.5 mm/s and ~4 mrad/s; real dynamics (drops,
    # pushes, carried swings) are orders of magnitude above these.
    weld_relax_max_lin_vel: ClassVar[float] = 0.02  # m/s
    weld_relax_max_ang_vel: ClassVar[float] = 0.2  # rad/s

    def _relax_resting_welds(self) -> None:
        """Re-anchor every weld whose assembly is resting free.

        A PyBullet JOINT_FIXED constraint between two table-resting
        bodies is never quiescent: each body settles into its own
        contact, the constraint accumulates sub-mm error, and the
        correction impulses rectify (through friction) into a steady
        skate -- measured 7-9 mm and up to 0.13 rad of yaw per 200 idle
        steps, invariant to maxForce, erp, pair-collision filtering and
        a zero-error anchor at creation, and present even for a welded
        pair 5 cm apart. Unwelded pairs in the same layout move < 1 mm.

        The fix breaks the error-accumulation loop: while every member
        of a welded assembly is quiescent, not held, and not touched by
        the robot, each weld is rebuilt at the current snapped relative
        pose every step, so the solver never has an error to fight and
        the assembly behaves like resting free bodies (which are
        stable). Under load -- carried, pushed, mid-drop -- the gates
        fail and the anchor holds, keeping the weld fully rigid exactly
        when rigidity matters. The relative-geometry ratchet this
        introduces is the free drift of resting bodies (sub-mm over
        hundreds of steps), not the skate.
        """
        if not self._weld_constraints:
            return
        # Connected components over the weld graph.
        adjacency: Dict[int, Set[int]] = {}
        for key in self._weld_constraints:
            body_a, body_b = tuple(key)
            adjacency.setdefault(body_a, set()).add(body_b)
            adjacency.setdefault(body_b, set()).add(body_a)
        seen: Set[int] = set()
        for root in list(adjacency):
            if root in seen:
                continue
            component = {root}
            frontier = [root]
            while frontier:
                for nxt in adjacency[frontier.pop()]:
                    if nxt not in component:
                        component.add(nxt)
                        frontier.append(nxt)
            seen |= component
            if self._held_obj_id is not None and \
                    self._held_obj_id in component:
                continue
            resting = True
            for body in component:
                lin, ang = p.getBaseVelocity(
                    body, physicsClientId=self._physics_client_id)
                if np.linalg.norm(lin) > self.weld_relax_max_lin_vel or \
                        np.linalg.norm(ang) > self.weld_relax_max_ang_vel:
                    resting = False
                    break
                if p.getContactPoints(self._pybullet_robot.robot_id,
                                      body,
                                      physicsClientId=self._physics_client_id):
                    resting = False
                    break
            if not resting:
                continue
            for key in list(self._weld_constraints):
                if not key <= component:
                    continue
                body_a, body_b, ideal_dz = self._weld_meta[key]
                self._remove_weld(key)
                self._create_weld(body_a, body_b, ideal_dz=ideal_dz)

    def _remove_weld(self, key: FrozenSet[int]) -> None:
        """Tear down one weld: remove the constraint and restore the pair's
        collision (disabled at creation; the blocks are separate objects again
        after a planner backtrack to a pre-weld state)."""
        cid = self._weld_constraints.pop(key)
        self._weld_meta.pop(key, None)
        p.removeConstraint(cid, physicsClientId=self._physics_client_id)
        body_a, body_b = tuple(key)
        p.setCollisionFilterPair(body_a,
                                 body_b,
                                 -1,
                                 -1,
                                 1,
                                 physicsClientId=self._physics_client_id)

    def get_welded_partner_transforms(
        self, body_id: int
    ) -> Dict[int, Tuple[Tuple[float, ...], Tuple[float, ...]]]:
        """Ideal ``(position, orientation)`` of every transitively welded
        partner RELATIVE to ``body_id``, chained from the weld constraints'
        snapped frames.

        Consumed by the skill-factory motion planner to pose welded
        partners of the held object. The constraint frames are the
        settled geometry the physical assembly returns to; live partner
        poses instead snapshot whatever pendulum transient the carried
        assembly is mid-swing through (an outer span was captured 19 mm
        low right after a lift), which poisons every collision check
        that reuses the capture.
        """
        out: Dict[int, Tuple[Tuple[float, ...], Tuple[float, ...]]] = {}
        identity = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
        frontier: List[int] = [body_id]
        transforms = {body_id: identity}
        while frontier:
            current = frontier.pop()
            for key, cid in self._weld_constraints.items():
                if current not in key:
                    continue
                (other, ) = key - {current}
                if other in transforms:
                    continue
                info = p.getConstraintInfo(
                    cid, physicsClientId=self._physics_client_id)
                parent_id, rel = info[0], (info[6], info[8])
                step_tf = rel if current == parent_id else \
                    p.invertTransform(rel[0], rel[1])
                base = transforms[current]
                tf = p.multiplyTransforms(base[0], base[1], step_tf[0],
                                          step_tf[1])
                transforms[other] = tf
                out[other] = tf
                frontier.append(other)
        return out

    def get_welded_partner_ids(self, body_id: int) -> Set[int]:
        """All body ids rigidly welded (transitively) to ``body_id``.

        Consumed by the skill-factory motion planner to exclude welded
        partners of the held object from the collision set.
        """
        partners: Set[int] = set()
        frontier = [body_id]
        while frontier:
            current = frontier.pop()
            for key in self._weld_constraints:
                if current in key:
                    (other, ) = key - {current}
                    if other != body_id and other not in partners:
                        partners.add(other)
                        frontier.append(other)
        return partners

    # -------------------------------------------------------------------------
    # Domain dynamics
    # -------------------------------------------------------------------------
    def _domain_specific_step(self) -> None:
        """Advance the glue process one step: wet faces near the held bottle's
        tip, tick cure counters on wet aligned joints, latch + weld at
        threshold, and refresh the patch visuals.

        NOTE: no prev-step handshake -- effects apply the moment the
        gate holds (a one-step delay makes the first step after a state
        jump a no-op, tripping the option model's repeat-state check).
        """
        state = self._get_state()
        blocks = state.get_objects(self._block_type)

        # 1. Glue application: sustained proximity wets the single
        #    nearest in-range face (see wet_streak_steps).
        best: Optional[Tuple[Object, str]] = None
        if state.get(self._bottle, "is_held") > 0.5:
            tip = (state.get(self._bottle, "x"), state.get(self._bottle, "y"),
                   state.get(self._bottle, "z") - self.bottle_half_extents[2])
            best_dist = self.apply_glue_radius
            for blk in blocks:
                for face in GLUE_FACES:
                    if self._attr(blk, f"glue_{face}", 0.0) > 0.5:
                        continue
                    if self._attr(blk, f"attached_{face}", -1.0) >= 0:
                        continue
                    dab = self._face_dab_point(state, blk, face)
                    dist = float(np.linalg.norm(np.array(tip) - np.array(dab)))
                    if dist < best_dist:
                        best = (blk, face)
                        best_dist = dist
        for blk in blocks:
            for face in GLUE_FACES:
                prev = self._attr(blk, f"glue_{face}", 0.0)
                if best == (blk, face):
                    streak = int(round(prev / self._WET_PARTIAL)) + 1
                    self._set_attr(
                        blk, f"glue_{face}",
                        1.0 if streak >= self.wet_streak_steps else streak *
                        self._WET_PARTIAL)
                elif 0.0 < prev <= 0.5:
                    # Not the in-range face this step: the streak breaks.
                    self._set_attr(blk, f"glue_{face}", 0.0)

        # 2. Curing: wet faces in aligned resting contact tick; at the
        #    threshold the joint latches irreversibly and welds. While a
        #    joint is merely wet it is TACKED (see _sync_wet_joint_tacks).
        curing_pairs: Set[FrozenSet[int]] = set()
        for blk in blocks:
            for face in GLUE_FACES:
                if self._attr(blk, f"glue_{face}", 0.0) <= 0.5:
                    continue
                if self._attr(blk, f"attached_{face}", -1.0) >= 0:
                    continue
                mate = self._find_mate(state, blk, face)
                if mate is None:
                    self._set_attr(blk, f"cure_{face}", 0.0)
                    continue
                cure = self._attr(blk, f"cure_{face}", 0.0) + 1.0
                self._set_attr(blk, f"cure_{face}", cure)
                assert blk.id is not None and mate.id is not None
                if cure >= self.cure_threshold and \
                        self._latch_joint(state, blk, face, mate):
                    # The rigid weld takes over from the tack.
                    self._drop_tack(frozenset({blk.id, mate.id}))
                else:
                    # Still wet -- or a latch that refused (see
                    # _latch_joint); either way the joint stays tacked.
                    curing_pairs.add(frozenset({blk.id, mate.id}))
        self._sync_wet_joint_tacks(curing_pairs)

        # 3. Anti-creep: re-anchor welds whose assembly rests free.
        self._relax_resting_welds()

        # 4. Visuals.
        self._update_glue_patches(state)

    def _find_mate(self, state: State, blk: Object,
                   face: str) -> Optional[Object]:
        """The unique block currently in aligned resting contact with ``blk``'s
        ``face``, or None.

        Vertical faces mirror the NextToEnd classifier; upward faces use
        the generic resting-contact check (which subsumes the OnBlock
        and SeatedOn classifiers and also covers a LYING block's wet top
        -- faces are physics, not task roles).
        """
        n = self._face_world_dir(state, blk, face)
        for other in state.get_objects(self._block_type):
            if other == blk:
                continue
            if n[2] > np.cos(np.pi / 4):
                # Upward face (a lying block's top, or a standing
                # block's upper end): the mate rests on it.
                if self._rests_on_top(state, other, blk):
                    return other
            elif abs(n[2]) < np.cos(np.pi / 4):
                # Vertical face: the mate butts against it.
                if self._end_adjacent(state, blk, face, other):
                    return other
            # Downward faces have no reachable mate.
        return None

    def _rests_on_top(self, state: State, other: Object, blk: Object) -> bool:
        """``other`` rests on ``blk``'s upward face (neither held).

        Generic over both blocks' orientations: a standing mate is gated
        by the circular stack alignment, a lying mate by the seat
        windows -- reducing exactly to OnBlock (leg stack) and SeatedOn
        (span seat) in those geometries, and extending the same physics
        to a lying block's top.
        """
        if self._Holding_holds(state, [self._robot, other]) or \
                self._Holding_holds(state, [self._robot, blk]):
            return False
        dz = state.get(other, "z") - (
            state.get(blk, "z") + self._world_half_extents(state, blk)[2] +
            self._world_half_extents(state, other)[2])
        if abs(dz) >= self.seat_z_tol:
            return False
        dx = state.get(other, "x") - state.get(blk, "x")
        dy = state.get(other, "y") - state.get(blk, "y")
        if self._stands(state, other):
            return bool(np.hypot(dx, dy) < self.stack_align_tol)
        return bool(abs(dx) < self.seat_x_window and abs(dy) < self.seat_y_tol)

    def _end_adjacent(self, state: State, blk: Object, face: str,
                      other: Object) -> bool:
        """``other`` butts against ``blk``'s end face (neither held)."""
        if self._Holding_holds(state, [self._robot, blk]) or \
                self._Holding_holds(state, [self._robot, other]):
            return False
        dx_dir, dy_dir, _ = self._face_world_dir(state, blk, face)
        dx = state.get(other, "x") - state.get(blk, "x")
        dy = state.get(other, "y") - state.get(blk, "y")
        dz = state.get(other, "z") - state.get(blk, "z")
        proj = dx * dx_dir + dy * dy_dir
        perp = abs(-dx * dy_dir + dy * dx_dir)
        # Extent of each block along the joint direction: blk's offset
        # to the face plane, plus the other block's horizontal reach (a
        # standing block contributes its cross-section half width).
        ext = self.block_half_extents[self._FACE_AXES[face][0]] + \
            self._world_half_extents(state, other)[0]
        if not ext - self.lateral_proj_tol_lo <= proj <= \
                ext + self.lateral_proj_tol_hi:
            return False
        return perp < self.lateral_perp_tol and \
            abs(dz) < self.lateral_z_tol

    # Attachment-slot local axes (normal axis index, sign). Extends the
    # glue-able faces with ``bottom`` (local -z): a vertical joint welds
    # a wet upward face to the underside of the block resting on it.
    _SLOT_AXES: ClassVar[Dict[str, Tuple[int, float]]] = {
        "top": (2, 1.0),
        "bottom": (2, -1.0),
        "end_a": (0, -1.0),
        "end_b": (0, 1.0),
    }

    def _mate_slot_for(self, state: State, blk: Object, face: str,
                       mate: Object) -> str:
        """The mate's attachment slot facing back toward ``blk.face``: the
        mate's local face whose world normal most opposes the wet face's."""
        n = np.array(self._face_world_dir(state, blk, face))
        rmat = self._block_rotation(state, mate)
        best_slot, best_dot = "bottom", np.inf
        for slot, (axis, sign) in self._SLOT_AXES.items():
            dot = float(n @ (sign * rmat[:, axis]))
            if dot < best_dot:
                best_slot, best_dot = slot, dot
        return best_slot

    def _latch_joint(self, state: State, blk: Object, face: str,
                     mate: Object) -> bool:
        """Irreversibly attach ``blk.face`` to ``mate``: record the partnership
        on both blocks, consume the glue, create the weld.

        Returns whether the joint latched.
        """
        mate_slot = self._mate_slot_for(state, blk, face, mate)
        if self._attr(mate, f"attached_{mate_slot}", -1.0) >= 0:
            # The mate's slot is somehow taken; refuse to latch rather
            # than corrupt the attachment graph (cure stays at the
            # threshold, so this re-checks every step).
            return False
        self._set_attr(blk, f"attached_{face}",
                       float(self._block_index[mate.name]))
        self._set_attr(mate, f"attached_{mate_slot}",
                       float(self._block_index[blk.name]))
        self._set_attr(blk, f"glue_{face}", 0.0)
        assert blk.id is not None and mate.id is not None
        if self._face_world_dir(state, blk, face)[2] > np.cos(np.pi / 4):
            # The mate rests on blk's upward face: a vertical joint.
            ideal_dz = self._world_half_extents(state, blk)[2] + \
                self._world_half_extents(state, mate)[2]
        else:
            ideal_dz = 0.0
        self._create_weld(blk.id, mate.id, ideal_dz=ideal_dz)
        return True

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

    # -------------------------------------------------------------------------
    # Predicates
    # -------------------------------------------------------------------------
    @staticmethod
    def _HandEmpty_holds(state: State, objects: Sequence[Object]) -> bool:
        robot, = objects
        return state.get(robot, "fingers") > 0.02

    @staticmethod
    def _Holding_holds(state: State, objects: Sequence[Object]) -> bool:
        _, blk = objects
        return state.get(blk, "is_held") > 0.5

    @staticmethod
    def _HoldingBottle_holds(state: State, objects: Sequence[Object]) -> bool:
        _, bottle = objects
        return state.get(bottle, "is_held") > 0.5

    def _make_glue_holds(self, face: str) -> Any:

        def _holds(state: State, objects: Sequence[Object]) -> bool:
            blk, = objects
            return state.get(blk, f"glue_{face}") > 0.5

        return _holds

    def _OnBlock_holds(self, state: State, objects: Sequence[Object]) -> bool:
        """A leg block rests stacked on another leg block."""
        top, bottom = objects
        if top == bottom:
            return False
        if not self._stands(state, top) or not self._stands(state, bottom):
            return False
        if self._Holding_holds(state, [self._robot, top]) or \
                self._Holding_holds(state, [self._robot, bottom]):
            return False
        dx = state.get(top, "x") - state.get(bottom, "x")
        dy = state.get(top, "y") - state.get(bottom, "y")
        if np.hypot(dx, dy) >= self.stack_align_tol:
            return False
        dz = state.get(
            top, "z") - (state.get(bottom, "z") + 2 * self.leg_half_extents[2])
        return bool(abs(dz) < self.stack_z_tol)

    def _NextToEnd_holds(self, state: State,
                         objects: Sequence[Object]) -> bool:
        """True when ``right`` butts against ``left``'s end_b face.

        The row grows in the +local-x direction of the left block.
        """
        right, left = objects
        if right == left:
            return False
        if self._stands(state, right) or self._stands(state, left):
            return False
        return self._end_adjacent(state, left, "end_b", right)

    def _SeatedOn_holds(self, state: State, objects: Sequence[Object]) -> bool:
        """The span block rests on the leg block's top, with the leg under the
        span's footprint."""
        span, leg = objects
        if span == leg:
            return False
        if self._stands(state, span) or not self._stands(state, leg):
            return False
        if self._Holding_holds(state, [self._robot, span]) or \
                self._Holding_holds(state, [self._robot, leg]):
            return False
        if abs(state.get(span, "y") - state.get(leg, "y")) >= \
                self.seat_y_tol:
            return False
        if abs(state.get(span, "x") - state.get(leg, "x")) >= \
                self.seat_x_window:
            return False
        dz = state.get(span, "z") - (state.get(
            leg, "z") + self.leg_half_extents[2] + self.span_half_extents[2])
        return bool(abs(dz) < self.seat_z_tol)

    def _AtSite_holds(self, state: State, objects: Sequence[Object]) -> bool:
        """The block stands on the table at the site (a stacked upper leg is
        NOT at the site -- the z check pins the base block)."""
        blk, site = objects
        if not self._stands(state, blk):
            return False
        if self._Holding_holds(state, [self._robot, blk]):
            return False
        dist = np.hypot(
            state.get(blk, "x") - state.get(site, "x"),
            state.get(blk, "y") - state.get(site, "y"))
        if dist >= self.at_site_tol:
            return False
        dz = state.get(blk,
                       "z") - (self.table_height + self.leg_half_extents[2])
        return bool(abs(dz) < self.at_site_z_tol)

    def _SiteFree_holds(self, state: State, objects: Sequence[Object]) -> bool:
        (site, ) = objects
        for blk in state.get_objects(self._block_type):
            if self._AtSite_holds(state, [blk, site]):
                return False
        return True

    def _Attached_holds(self, state: State, objects: Sequence[Object]) -> bool:
        """The two blocks share a cured glue joint (symmetric)."""
        a, b = objects
        if a == b:
            return False
        idx_a = self._block_index[a.name]
        idx_b = self._block_index[b.name]
        for slot in ATTACH_SLOTS:
            if int(round(self._attached_value(state, a, slot))) == idx_b:
                return True
            if int(round(self._attached_value(state, b, slot))) == idx_a:
                return True
        return False

    def _TopFree_holds(self, state: State, objects: Sequence[Object]) -> bool:
        """Nothing rests on the block's top face (so the face can be reached by
        the glue bottle's dab)."""
        (blk, ) = objects
        for other in state.get_objects(self._block_type):
            if other == blk:
                continue
            if self._OnBlock_holds(state, [other, blk]) or \
                    self._SeatedOn_holds(state, [other, blk]):
                return False
        return True

    def _EndsFree_holds_from_atoms(self, atoms: Set[GroundAtom],
                                   objects: Sequence[Object]) -> bool:
        """No unwelded block butts either end of ``blk`` (derived: evaluated
        over NextToEnd/Attached atoms, so the abstract search keeps it
        consistent without frame axioms)."""
        (blk, ) = objects
        welded = set()
        neighbors = set()
        for atom in atoms:
            if atom.predicate not in (self._Attached, self._NextToEnd):
                continue
            if blk not in atom.objects:
                continue
            other = atom.objects[0] if atom.objects[1] == blk \
                else atom.objects[1]
            if other == blk:
                continue
            if atom.predicate == self._Attached:
                welded.add(other)
            else:
                neighbors.add(other)
        return neighbors.issubset(welded)

    def _Loose_holds(self, state: State, objects: Sequence[Object]) -> bool:
        """The block has no cured attachments (it can be individually picked
        and re-placed without dragging an assembly along)."""
        (blk, ) = objects
        return all(
            int(round(self._attached_value(state, blk, slot))) < 0
            for slot in ATTACH_SLOTS)

    # -------------------------------------------------------------------------
    # Task Generation
    # -------------------------------------------------------------------------
    def _generate_train_tasks(self) -> List[EnvironmentTask]:
        return self._make_tasks(num_tasks=CFG.num_train_tasks,
                                rng=self._train_rng)

    def _generate_test_tasks(self) -> List[EnvironmentTask]:
        return self._make_tasks(num_tasks=CFG.num_test_tasks,
                                rng=self._test_rng)

    def _make_tasks(self, num_tasks: int,
                    rng: np.random.Generator) -> List[EnvironmentTask]:
        tasks = []
        for _ in range(num_tasks):
            legs = self._legs
            spans = self._spans
            site_sep = self.site_sep

            init_dict: Dict[Object, Dict[str, float]] = {}
            init_dict[self._robot] = {
                "x": self.robot_init_x,
                "y": self.robot_init_y,
                "z": self.robot_init_z,
                "fingers": self.open_fingers,
                "roll": self.robot_init_roll,
                "tilt": self.robot_init_tilt,
                "wrist": self.robot_init_wrist,
            }

            # Sites: mid-table band, x jittered for diversity.
            mid_x = float(
                rng.uniform(self.x_mid - self.site_x_jitter,
                            self.x_mid + self.site_x_jitter))
            site_xs = (mid_x - site_sep / 2, mid_x + site_sep / 2)
            for site, sx in zip(self._sites, site_xs):
                init_dict[site] = {
                    "x": sx,
                    "y": self.site_y,
                    "z": self.table_height,
                }

            # Staging: assign objects to jittered grid slots; the span
            # row is assembled in place at span0's slot, growing in +x
            # along the front row (see _stage_objects).
            stage_xy = self._stage_objects(rng, legs, spans, site_xs)

            # Per-task color draws from the role families, without
            # replacement within each role.
            leg_picks = rng.permutation(len(self.leg_color_family))
            span_picks = rng.permutation(len(self.span_color_family))

            # Block init features.
            for role_idx, blk in list(enumerate(legs)) + \
                    list(enumerate(spans)):
                is_leg = self._is_leg_shaped(blk)
                bx, by = stage_xy[blk]
                if is_leg:
                    r_col, g_col, b_col = self.leg_color_family[int(
                        leg_picks[role_idx])]
                else:
                    r_col, g_col, b_col = self.span_color_family[int(
                        span_picks[role_idx])]
                # Legs are the same block stood on end: pitch = -pi/2
                # (local +x up, so the leg's world-top is its end_b
                # face).
                feats: Dict[str, float] = {
                    "x":
                    bx,
                    "y":
                    by,
                    "z":
                    self.table_height + (self.leg_half_extents[2] if is_leg
                                         else self.span_half_extents[2]),
                    "roll":
                    0.0,
                    "pitch":
                    -np.pi / 2 if is_leg else 0.0,
                    "yaw":
                    0.0,
                    "half_x":
                    self.block_half_extents[0],
                    "half_y":
                    self.block_half_extents[1],
                    "half_z":
                    self.block_half_extents[2],
                    "is_held":
                    0.0,
                    "r":
                    r_col,
                    "g":
                    g_col,
                    "b":
                    b_col,
                }
                for face in GLUE_FACES:
                    feats[f"glue_{face}"] = 0.0
                    if f"cure_{face}" in self._block_type.feature_names:
                        feats[f"cure_{face}"] = 0.0
                for slot in ATTACH_SLOTS:
                    if f"attached_{slot}" in self._block_type.feature_names:
                        feats[f"attached_{slot}"] = -1.0
                init_dict[blk] = feats

            bx, by = stage_xy[self._bottle]
            init_dict[self._bottle] = {
                "x": bx,
                "y": by,
                "z": self.table_height + self.bottle_half_extents[2],
                "rot": 0.0,
                "is_held": 0.0,
            }

            init_state = utils.create_state_from_dict(init_dict)
            if CFG.partially_observable:
                init_state.privileged = {
                    blk.name: {
                        **{f"cure_{face}": 0.0
                           for face in GLUE_FACES},
                        **{f"attached_{slot}": -1.0
                           for slot in ATTACH_SLOTS},
                    }
                    for blk in legs + spans
                }

            # Goal: the n-bridge standing at the two sites. Roles are
            # task-assigned by block name, and the goal pins the full
            # geometric layout (AtSite / NextToEnd / SeatedOn
            # plus the ROW welds): every atom persists in the finished
            # bridge, and the pinning forces the planner's bindings to
            # the physically consistent left-to-right build. The SEAT
            # joints are deliberately NOT in the goal: they are not
            # structural (the welded row rests on the legs by gravity),
            # and the glue that IS required -- the lateral row welds --
            # should be discoverable as a physical necessity (the
            # unglued middle span falls into the gap), not read off the
            # goal.
            goal_atoms = {
                GroundAtom(self._AtSite, [legs[0], self._sites[0]]),
                GroundAtom(self._AtSite, [legs[1], self._sites[1]]),
                GroundAtom(self._SeatedOn, [spans[0], legs[0]]),
                GroundAtom(self._SeatedOn, [spans[-1], legs[1]]),
            }
            for left_span, right_span in zip(spans, spans[1:]):
                goal_atoms.add(
                    GroundAtom(self._NextToEnd, [right_span, left_span]))
                goal_atoms.add(
                    GroundAtom(self._Attached, [left_span, right_span]))
            # Outcome-only description: it says WHAT must stand at the
            # end, never how (no glue recipe) -- discovering that the
            # row must be glued and cured before it can be seated is
            # the agent's job.
            goal_nl = ("Build an n-shaped bridge standing at the two marked "
                       "sites: stand a leg on each site pad, join the three "
                       "span blocks end-to-end into one rigid span, and seat "
                       "it resting across the two leg tops.")

            tasks.append(
                EnvironmentTask(init_state, goal_atoms, goal_nl=goal_nl))

        return self._add_pybullet_state_to_tasks(tasks)

    def _stage_objects(
            self, rng: np.random.Generator, legs: List[Object],
            spans: List[Object],
            site_xs: Tuple[float, float]) -> Dict[Object, Tuple[float, float]]:
        """Assign every staged object (blocks + bottle) a jittered grid slot.

        span0 takes a random feasible front-row slot and the strip
        [span0_x, span0_x + row_len] on that row is reserved for
        assembling the span row; everything else fills the remaining
        slots in random order.
        """
        base_x, base_y = self.robot_base_pos[0], self.robot_base_pos[1]
        row_len = (len(spans) - 1) * (2 * self.span_half_extents[0] +
                                      self.lateral_place_gap)

        def _reachable(x: float, y: float) -> bool:
            # Radial reach cap (empirical fetch validated-IK frontier,
            # see pybullet_bond), with margin for the slot jitter.
            return np.hypot(x - base_x, y - base_y) <= \
                self.reach_radius - 1.5 * self.stage_jitter

        slots: List[Tuple[float, float]] = []
        for col in self.stage_cols:
            for row in (self.stage_row_front, self.stage_row_mid,
                        self.stage_row_back):
                if not _reachable(col, row):
                    continue
                # Mid- and back-row slots directly BEHIND a site are
                # unusable: those rows sit only 6 / 8 cm from the site
                # band, and once a leg stands at the site, the grasp
                # volume of a pick from a same-column slot clips it
                # (observed -7 mm palm contact from the back row and
                # -8 mm from a mid-row bottle grasp, the lowest palm of
                # all). 7 cm = palm half-width (~4.5 cm) + leg
                # half-width, the x-overlap threshold of that grasp
                # volume. The front row is 16 cm from the band -- safe.
                if row != self.stage_row_front and any(
                        abs(col - sx) < 0.07 for sx in site_xs):
                    continue
                slots.append((col, row))

        # span0 + assembly strip on the front row. Beyond radial
        # reach, the whole strip's sampler targets (span2's place, the
        # rightmost glue dab) must stay inside the option x band --
        # a start column too far right makes the task unsolvable (the
        # clipped place target interpenetrates the row).
        strip_starts = [
            col for col in self.stage_cols
            if _reachable(col + row_len, self.stage_row_front) and (
                col, self.stage_row_front) in slots and col + row_len +
            self.strip_x_slack <= self.workspace_x_hi
        ]
        assert strip_starts, "no feasible span-row start"
        span0_col = float(rng.choice(strip_starts))
        span0_xy = (span0_col, self.stage_row_front)
        # Reserve every slot the strip sweeps over: the front-row cells
        # themselves, plus the mid-row cells directly behind them --
        # the front and mid rows are only 10 cm apart in y, and the
        # gripper's grasp/place volume over a front-row cell reaches
        # ~1-2 cm into a block parked in the adjacent mid-row cell.
        slots = [(cx, cy) for cx, cy in slots if not (
            cy in (self.stage_row_front, self.stage_row_mid) and span0_col -
            self.site_keepout <= cx <= span0_col + row_len + self.site_keepout)
                 ]

        rest = spans[1:] + legs + [self._bottle]
        assert len(slots) >= len(rest), \
            f"only {len(slots)} staging slots for {len(rest)} objects"
        # Re-draw the assignment until it is grasp-clearance feasible:
        # 1. Spans are 10 cm long (lying along x), so two spans in
        #    same-row adjacent columns (0.11 apart) would overlap; legs
        #    (5 cm) and the bottle are fine.
        # 2. No two staged objects in the SAME COLUMN of the two
        #    front-band rows (10 cm apart in y): the finger assembly at
        #    a grasp/place pose over one cell penetrates a neighbor in
        #    the row-adjacent cell (observed 1-2 cm; the mid-back gap,
        #    14 cm, is fine).
        span_rest = set(spans[1:])
        for _ in range(200):
            chosen = rng.choice(len(slots), size=len(rest), replace=False)
            # Include span0 at the strip start: a slot in the column
            # just left of it survives the strip reservation but is
            # still too close for a 10 cm neighbor.
            placed = [span0_xy] + [slots[int(si)] for si in chosen]
            span_slots = [span0_xy] + [
                slots[int(si)]
                for obj, si in zip(rest, chosen) if obj in span_rest
            ]
            spans_ok = all(
                abs(ay - by) > 0.01 or abs(ax - bx) > 0.115
                for i, (ax, ay) in enumerate(span_slots)
                for bx, by in span_slots[i + 1:])
            rows_ok = all(
                abs(ax - bx) > 0.08 or not 0.02 < abs(ay - by) < 0.12
                for i, (ax, ay) in enumerate(placed)
                for bx, by in placed[i + 1:])
            if spans_ok and rows_ok:
                break
        else:
            # Never use a layout that violates the grasp-clearance
            # constraints -- overlapping spawns settle into an
            # unplanned layout and fail far from the cause.
            raise RuntimeError("No valid staging assignment after 200 draws; "
                               "grid constants no longer admit one.")
        stage_xy: Dict[Object, Tuple[float, float]] = {spans[0]: span0_xy}
        for obj, slot_i in zip(rest, chosen):
            stage_xy[obj] = slots[int(slot_i)]
        # Jitter everything (including span0).
        return {
            obj:
            (x + float(rng.uniform(-self.stage_jitter, self.stage_jitter)),
             y + float(rng.uniform(-self.stage_jitter, self.stage_jitter)))
            for obj, (x, y) in stage_xy.items()
        }


if __name__ == "__main__":

    def _main() -> None:
        """Quick manual visualization."""
        import time  # pylint: disable=import-outside-toplevel
        CFG.seed = 0
        CFG.env = "pybullet_bridge"
        CFG.num_train_tasks = 1
        env = PyBulletBridgeEnv(use_gui=True)
        task = env._generate_train_tasks()[0]  # pylint: disable=protected-access
        env._set_state(task.init)  # pylint: disable=protected-access
        while True:
            env.step(
                Action(np.array(env._pybullet_robot.initial_joint_positions)))  # pylint: disable=protected-access
            time.sleep(0.01)

    _main()
