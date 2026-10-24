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
  consumed on both faces of the joint (one wet face is enough to cure;
  a wet mate face is consumed with it, never left wet on an attached
  face), both blocks record the attachment (``attached_*`` = partner
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

The observable simulation core (scene geometry, block / bottle / site /
glue-patch body construction, state read/write) lives in
:mod:`predicators.envs.pybullet_bridge_base`, which may be surfaced to
learning agents as reference source. This module holds everything an
agent must LEARN or must not see:

* the glue residual dynamics (``_domain_specific_step``: wetting,
  curing, wet-joint tacks, latching and the weld lifecycle) and every
  constant of those laws - the learning target of the sim-learning
  experiments;
* task generation (the train/test distribution and staging layout);
* predicates, goal semantics and the settle certificate (their
  thresholds are what predicate invention rediscovers).
"""

import itertools
import json
import logging
from typing import Any, ClassVar, Dict, FrozenSet, List, Optional, Sequence, \
    Set, Tuple

import numpy as np
import pybullet as p

from predicators import utils
from predicators.envs.pybullet_bridge_base import ATTACH_SLOTS, GLUE_FACES, \
    PyBulletBridgeBaseEnv
from predicators.settings import CFG
from predicators.structs import Action, DerivedPredicate, EnvironmentTask, \
    GroundAtom, Object, Observation, Predicate, State


class PyBulletBridgeEnv(PyBulletBridgeBaseEnv):
    """Build an n-shaped bridge by gluing blocks; cured joints physically weld
    blocks into rigid assemblies.

    Subclass of the observable sim core (see
    :mod:`predicators.envs.pybullet_bridge_base`); this class adds the
    hidden glue dynamics, task generation, and predicates.
    """

    # -------------------------------------------------------------------------
    # Task layout
    # -------------------------------------------------------------------------
    # Default span count of a task (the live counts come from
    # CFG.bridge_train_span_blocks / CFG.bridge_test_span_blocks).
    n_spans: ClassVar[int] = 3
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

    @classmethod
    def glue_dab_dwell_steps(cls) -> int:
        """How long the MoveTo skill holds a reached glue target before
        retreating.

        Derived from the hidden wetting law (+1 covers the
        approach/retreat edge steps), but exposed under a neutral name:
        the bridge options file is copied verbatim into the agent
        sandbox as a skill reference, so it must not spell out
        ``wet_streak_steps`` or the consecutive-dwell requirement.
        """
        return cls.wet_streak_steps + 1

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
    # Seat tolerances for SeatedOn(span, leg). The x window is measured
    # from the leg's centre along the row. A three-span row (0.30 m) over
    # sites 0.25 m apart puts each outer span's centre 25 mm past its leg
    # by construction; on top of that, Place lands legs 8-14 mm off their
    # command and a glued row shortens 5-8 mm per joint under the seat
    # (welded partners do not collide, see _create_weld), so a bridge
    # that is physically standing measured 47 mm (seed-3 cycle-0 test,
    # 2026-09-03) and failed the old 45 mm gate. 60 mm still leaves the
    # span end 10 mm short of the leg centre, i.e. 15 mm of the 50 mm
    # leg top under the span, which is a seated joint by any reading.
    seat_x_window: ClassVar[float] = 0.06
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

    def __init__(self, use_gui: bool = False, **kwargs: Any) -> None:
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
        # The goal: a three-span row, butted end to end, resting across
        # a leg standing at each site. Role-free by construction (any
        # block may be either leg, the spans may sit in any order): the
        # blocks are identical, and pinning roles by name turned a
        # perfectly good bridge into a lost level (2026-09-07 seed 2:
        # the row was welded span0|span2|span1 for reach convenience,
        # the NL goal never said which order, and welds are permanent).
        self._Bridged = Predicate("Bridged",
                                  [self._site_type, self._site_type],
                                  self._Bridged_holds)
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
        # RowComplete(first, last) = a cured Attached path from first to
        # last runs through exactly the task's span count of lying
        # blocks. DERIVED from Lying / Standing / Attached atoms,
        # and required of the outer spans by PickRow / SeatSpan: Bridged
        # needs every span in the row, so a seat operator over fewer
        # spans than the task has would add a fictional Bridged (the
        # planner seated a welded three-span chain in a four-span task
        # and then had to dismantle the seated row to add the fourth).
        # Role-free like Bridged: a toppled leg lies but is not welded,
        # so it neither joins nor blocks the row.
        self._RowComplete = DerivedPredicate(
            "RowComplete", [self._block_type, self._block_type],
            self._RowComplete_holds_from_atoms,
            auxiliary_predicates={self._Lying, self._Standing, self._Attached})

    @classmethod
    def get_name(cls) -> str:
        return "pybullet_bridge"

    @property
    def predicates(self) -> Set[Predicate]:
        return {
            self._HandEmpty, self._Holding, self._HoldingBottle,
            self._GlueEndB, self._NextToEnd, self._SeatedOn, self._AtSite,
            self._Bridged, self._SiteFree, self._Attached, self._Standing,
            self._Lying, self._Loose, self._Resting, self._TopFree,
            self._EndsFree, self._RowComplete
        }

    @property
    def goal_predicates(self) -> Set[Predicate]:
        # The goal is the one role-free layout atom, Bridged(site, site),
        # defined over the geometric atoms (AtSite / NextToEnd /
        # SeatedOn) that the oracle's operators achieve. Attached is
        # deliberately absent - the goal is fully observable, and the
        # row welds it implies are certified physically by the settle
        # check in check_episode_trajectory.
        return {self._Bridged}

    # Settle duration for the episode certificate below: three actions'
    # worth of physics. An unsupported span in free fall leaves the
    # 1.5 cm NextToEnd z-window within a single action's substeps
    # (~3.4 cm of drop), so this carries ample margin.
    _GOAL_SETTLE_SUBSTEPS: ClassVar[int] = 60

    def episode_terminated(self, observations: Sequence[Observation]) -> bool:
        """Let the robot withdraw before certifying a candidate bridge.

        The goal atoms hold at the release step itself, while the open
        fingers still straddle the row; with
        ``bridge_goal_robot_clearance`` set, the certificate waits until
        every robot link is clear of every block, so the settle below
        judges the structure and not a gripper leaning on it.
        """
        if not super().episode_terminated(observations):
            return False
        clearance = CFG.bridge_goal_robot_clearance
        if clearance <= 0:
            return True
        return not any(
            p.getClosestPoints(self._pybullet_robot.robot_id,
                               block.id,
                               clearance,
                               physicsClientId=self._physics_client_id)
            for block in self._task_blocks() if block.id is not None)

    def _task_blocks(self) -> List[Object]:
        """The blocks of the CURRENT task: the body pool holds the larger of
        the train and test span counts, so a three-span task in a transfer run
        has a pooled fourth span that is in no state (reading it raised
        KeyError from the certificate on the step that completed the
        bridge)."""
        state = self._get_state()
        return [blk for blk in self._blocks if blk in state.data]

    def _certificate_snapshot(self) -> Dict[str, Any]:
        """Private diagnostics for the certificate log, never part of the
        agent's observation."""
        state = self._get_state()
        return {
            "poses": {
                obj.name: {
                    f: float(state.get(obj, f))
                    for f in ("x", "y", "z", "roll", "pitch", "yaw")
                }
                for obj in self._blocks if obj in state.data
            },
            "velocities":
            self._body_velocity_records(),
            "joints":
            self._pybullet_robot.get_joints(),
            "welds": [
                p.getConstraintInfo(cid,
                                    physicsClientId=self._physics_client_id)
                for cid in self._weld_constraints.values()
            ],
            "contacts":
            p.getContactPoints(physicsClientId=self._physics_client_id),
            "atoms":
            sorted(map(str, utils.abstract(state, self.predicates))),
        }

    def check_episode_trajectory(
            self, observations: Sequence[Observation],
            actions: Sequence[Action]) -> Tuple[bool, str]:
        """Certify success by letting the final scene settle unactuated.

        The goal is purely geometric (Attached is deliberately not in
        it), and the base sim removes the grasp constraint AFTER an
        action's physics substeps - so the release step's observation
        shows an unheld block at its exact placement pose with zero
        fall, and a DRY span row transiently satisfies the goal there.
        Free physics settles the ambiguity: cured welds are persistent
        constraints and hold the row up; an unwelded span drops out of
        the NextToEnd window within one action's worth of substeps.
        Raw ``stepSimulation`` on purpose - no robot actuation, no wet
        tack, no cure progression - so the structure must stand by its
        cured joints alone. The one piece of ordinary dynamics kept is
        the anti-creep weld re-anchoring (``_relax_resting_welds``, once
        per action's worth of substeps): it never props up an unwelded
        span, but without it the resting welds skate exactly as they do
        in idle play, which the certificate must not punish. Runs once
        at episode end, so mutating the sim is safe (the env is reset
        before its next use); the settled scene is what the runner
        records, and the certificate logs its before/after diagnostics.
        """
        ok, reason = super().check_episode_trajectory(observations, actions)
        if not ok:
            return ok, reason
        before = self._certificate_snapshot()
        for substep in range(self._GOAL_SETTLE_SUBSTEPS):
            p.stepSimulation(physicsClientId=self._physics_client_id)
            if (substep + 1) % CFG.pybullet_sim_steps_per_action == 0:
                self._relax_resting_welds()
        settled = self._get_state()
        goal = self._current_task.goal_description
        assert isinstance(goal, set)
        missing = [a for a in goal if not a.holds(settled)]
        after = self._certificate_snapshot()
        logging.info(
            "[bridge certificate] %s",
            json.dumps({
                "before": before,
                "after": after,
                "substeps": self._GOAL_SETTLE_SUBSTEPS,
                "accepted": not missing,
            }))
        if missing:
            lost = sorted(set(before["atoms"]) - set(after["atoms"]))
            return False, ("goal geometry did not survive settling "
                           f"(missing: {sorted(map(str, missing))}; "
                           f"lost relations: {lost})")
        return True, ""

    # -------------------------------------------------------------------------
    # Small helpers
    # -------------------------------------------------------------------------
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
    def _set_domain_specific_state(self, state: State) -> None:
        super()._set_domain_specific_state(state)
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

    def _weld_constraint_edges(self) -> Dict[FrozenSet[int], int]:
        """Extend the base registry with this env's native glue welds, so the
        held-assembly pin (see PyBulletEnv._pin_welds_to_held_root) covers both
        weld paths.

        Wet-glue TACKS are deliberately
        excluded: they are weak by design (a wet joint must not carry).
        """
        edges = super()._weld_constraint_edges()
        edges.update(self._weld_constraints)
        return edges

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
        if mate_slot in GLUE_FACES:
            # The joint consumes the mate's wet face too. An attached
            # face never cures again, so glue left on it would be a
            # wet flag that can never clear: the continual agent glued
            # both faces of every joint and waited 360 steps for the
            # second flag (bridge seed 0, 2026-09-04).
            self._set_attr(mate, f"glue_{mate_slot}", 0.0)
        assert blk.id is not None and mate.id is not None
        if self._face_world_dir(state, blk, face)[2] > np.cos(np.pi / 4):
            # The mate rests on blk's upward face: a vertical joint.
            ideal_dz = self._world_half_extents(state, blk)[2] + \
                self._world_half_extents(state, mate)[2]
        else:
            ideal_dz = 0.0
        self._create_weld(blk.id, mate.id, ideal_dz=ideal_dz)
        return True

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

    def _Bridged_holds(self, state: State, objects: Sequence[Object]) -> bool:
        """A row of all the spans, butted end to end, rests across a block
        standing at each site.

        Role-free: any standing block serves as either leg and the
        lying blocks may form the row in any order, butted in either
        direction (``NextToEnd`` is directional, right against the
        left's end_b face, and a row built right-to-left or from a
        turned block reads the other way round). Symmetric in the two
        sites.
        """
        site_a, site_b = objects
        if site_a == site_b:
            return False
        blocks = state.get_objects(self._block_type)
        legs_a = [b for b in blocks if self._AtSite_holds(state, [b, site_a])]
        legs_b = [b for b in blocks if self._AtSite_holds(state, [b, site_b])]
        if not legs_a or not legs_b:
            return False
        lying = [b for b in blocks if not self._stands(state, b)]
        for chain in itertools.permutations(lying, len(blocks) - self.n_legs):
            if not all(
                    self._NextToEnd_holds(state, [right, left])
                    or self._NextToEnd_holds(state, [left, right])
                    for left, right in zip(chain, chain[1:])):
                continue
            for leg_a in legs_a:
                for leg_b in legs_b:
                    if leg_a == leg_b:
                        continue
                    if self._SeatedOn_holds(state, [chain[0], leg_a]) and \
                            self._SeatedOn_holds(state, [chain[-1], leg_b]):
                        return True
        return False

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

    def _RowComplete_holds_from_atoms(self, atoms: Set[GroundAtom],
                                      objects: Sequence[Object]) -> bool:
        """Some simple Attached path from ``first`` to ``last`` runs through
        exactly the task's span count of lying blocks (derived: evaluated over
        Lying / Standing / Attached atoms).

        The span count is the block count less the legs, read off the
        atoms (every block is Lying or Standing), so a toppled leg that
        happens to lie loose neither joins the row nor blocks it. Stated
        as an existence so it is MONOTONE in the atoms: the planner's
        delete-relaxed reachability evaluates it on a superset where
        every pair is Attached, and a definition that also forbade extra
        edges read false there and made every Bridged goal unreachable.
        """
        first, last = objects
        if first == last:
            return False
        lying: Set[Object] = set()
        blocks: Set[Object] = set()
        for atom in atoms:
            if atom.predicate == self._Lying:
                lying.add(atom.objects[0])
                blocks.add(atom.objects[0])
            elif atom.predicate == self._Standing:
                blocks.add(atom.objects[0])
        n_spans = len(blocks) - self.n_legs
        if first not in lying or last not in lying or n_spans < 2:
            return False
        adjacency: Dict[Object, Set[Object]] = {blk: set() for blk in lying}
        for atom in atoms:
            if atom.predicate != self._Attached:
                continue
            blk_a, blk_b = atom.objects
            if blk_a != blk_b and blk_a in lying and blk_b in lying:
                adjacency[blk_a].add(blk_b)
                adjacency[blk_b].add(blk_a)

        def _path_exists(cur: Object, visited: Set[Object]) -> bool:
            if len(visited) == n_spans:
                return cur == last
            if cur == last:
                return False
            return any(
                _path_exists(nxt, visited | {nxt}) for nxt in adjacency[cur]
                if nxt not in visited)

        return _path_exists(first, {first})

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
                                rng=self._train_rng,
                                n_spans=CFG.bridge_train_span_blocks)

    def _generate_test_tasks(self) -> List[EnvironmentTask]:
        return self._make_tasks(num_tasks=CFG.num_test_tasks,
                                rng=self._test_rng,
                                n_spans=CFG.bridge_test_span_blocks)

    def _make_tasks(self,
                    num_tasks: int,
                    rng: np.random.Generator,
                    n_spans: int = 3) -> List[EnvironmentTask]:
        tasks = []
        for _ in range(num_tasks):
            legs = self._legs
            spans = self._spans[:n_spans]
            site_sep = 2 * self.span_half_extents[0] * n_spans - \
                2 * self.leg_half_extents[0]

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

            # Goal: the n-bridge standing at the two sites, as the one
            # role-free layout atom Bridged(site0, site1): a leg standing
            # at each site and the spans butted into one row resting
            # across them, whichever block plays which role and in
            # whichever order (see _Bridged_holds). The SEAT joints are
            # deliberately NOT in the goal: they are not structural
            # (the welded row rests on the legs by gravity). Neither is
            # Attached: the row welds are physically implied (the
            # unwelded middle span falls into the gap, so the geometry
            # cannot persist without them - enforced by the settle
            # check in check_episode_trajectory), and keeping the goal
            # fully observable lets a learned belief model represent it
            # without access to the hidden attachment state. Discovering
            # that the row must be glued and cured stays the agent's
            # job - it is a physical necessity, not read off the goal.
            goal_atoms = {
                GroundAtom(self._Bridged, [self._sites[0], self._sites[1]]),
            }
            # Outcome-only description: it says WHAT must stand at the
            # end, never how (no glue recipe) -- discovering that the
            # row must be glued and cured before it can be seated is
            # the agent's job.
            goal_nl = (
                "Build an n-shaped bridge standing at the two marked "
                f"sites: stand a leg on each site pad, join the {n_spans} "
                "span blocks end-to-end into one rigid span, and seat "
                "it resting across the two leg tops.")
            if CFG.bridge_goal_robot_clearance > 0:
                goal_nl += (" Finish with the robot at least "
                            f"{CFG.bridge_goal_robot_clearance:g} m away "
                            "from every block.")

            tasks.append(
                EnvironmentTask(init_state, goal_atoms, goal_nl=goal_nl))

        return self._add_pybullet_state_to_tasks(tasks)

    def _stage_transfer_objects(
            self, rng: np.random.Generator, legs: List[Object],
            spans: List[Object],
            site_xs: Tuple[float, float]) -> Dict[Object, Tuple[float, float]]:
        """Pack the larger roster by finite search over grasp-clear grid slots.

        Random rejection becomes unreliable when the extra block and
        longer assembly strip nearly fill the reachable area. Search a
        shuffled grid with the same spacing and keepouts instead of
        accepting overlap.
        """
        base_x, base_y = self.robot_base_pos[:2]
        front, middle, back = 1.12, 1.26, 1.40
        row_len = (len(spans) - 1) * (2 * self.span_half_extents[0] +
                                      self.lateral_place_gap)

        def reachable(x: float, y: float) -> bool:
            return bool(
                np.hypot(x - base_x, y - base_y) <= self.reach_radius -
                1.5 * self.stage_jitter)

        slots = [(x, y) for x in self.stage_cols for y in (front, middle, back)
                 if reachable(x, y) and (y == front or all(
                     abs(x - sx) >= .07 for sx in site_xs))]
        starts = [
            x for x in self.stage_cols
            if (x, front) in slots and reachable(x + row_len, front) and x +
            row_len + self.strip_x_slack <= self.workspace_x_hi
        ]
        rest = spans[1:] + legs + [self._bottle]
        span_set = set(spans)

        def assign(index: int, remaining: List[Tuple[float, float]],
                   placed: Dict[Object, Tuple[float, float]]) -> bool:
            if index == len(rest):
                return True
            obj = rest[index]
            for x, y in remaining:
                if any(
                        abs(x - px) <= .08 and .02 < abs(y - py) < .12 or (
                            x, y) == (px, py) or (
                                obj in span_set and other in span_set
                                and abs(y - py) <= .01 and abs(x - px) <= .115)
                        for other, (px, py) in placed.items()):
                    continue
                placed[obj] = (x, y)
                if assign(index + 1, remaining, placed):
                    return True
                del placed[obj]
            return False

        for start_index in rng.permutation(len(starts)):
            start = starts[int(start_index)]
            remaining = [(x, y) for x, y in slots if not (
                y == front and start - self.site_keepout <= x <= start +
                row_len + self.site_keepout)]
            remaining = [
                remaining[int(i)] for i in rng.permutation(len(remaining))
            ]
            placed = {spans[0]: (start, front)}

            if assign(0, remaining, placed):
                return {
                    obj:
                    (x +
                     float(rng.uniform(-self.stage_jitter, self.stage_jitter)),
                     y +
                     float(rng.uniform(-self.stage_jitter, self.stage_jitter)))
                    for obj, (x, y) in placed.items()
                }
        raise RuntimeError(
            "No grasp-clear staging layout for the Bridge roster")

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
        if len(spans) == 4:
            return self._stage_transfer_objects(rng, legs, spans, site_xs)
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
