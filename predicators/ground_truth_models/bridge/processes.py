"""Ground-truth processes for the bridge (glue construction) environment.

These drive the ``oracle_process_planning`` demo generator: endogenous
(option-backed) processes for pick / place / butt-join / seat / glue
application, and the exogenous ``CureLateralJoint`` process encoding
the hidden dwell-time dynamics (a wet face held in aligned contact for
~``cure_threshold`` steps -> the pair is ``Attached`` and physically
welded).

Direction conventions the samplers rely on (mirrored by the env's task
generator): the span row grows in +x (``NextToEnd(right, left)`` = right
butts against left's ``end_b`` face), and goals pin every geometric
atom (AtSite / NextToEnd / SeatedOn), so the planner's bindings always
match a physically consistent left-to-right build.
"""
from typing import Dict, Sequence, Set

import numpy as np
import torch

from predicators.envs.pybullet_bridge import PyBulletBridgeEnv
from predicators.ground_truth_models import GroundTruthProcessFactory
from predicators.structs import Array, CausalProcess, DelayDistribution, \
    EndogenousProcess, ExogenousProcess, GroundAtom, LiftedAtom, Object, \
    ParameterizedOption, Predicate, State, Type, Variable
from predicators.utils import ConstantDelay, DiscreteGaussianDelay, \
    null_sampler

_ENV = PyBulletBridgeEnv
_LEG_H = 2 * _ENV.leg_half_extents[2]  # 0.10
_SPAN_LEN = 2 * _ENV.span_half_extents[0]  # 0.10
_SPAN_TH = 2 * _ENV.span_half_extents[2]  # 0.05
_TABLE = _ENV.table_height
# End the collision-checked descent with the held object's underside
# ~3 mm above the resting surface. Place's release_z is the HELD
# OBJECT'S CENTER height (the skill live-compensates the EE-to-held
# offset on all axes), so a target is simply resting-center + this
# clearance -- no grasp-depth or IK-residual budgeting. The skill then
# settles to FIRST CONTACT before opening (settle_to_contact_depth in
# options.py), so this is a descend-goal clearance, not a free-fall
# height: it only needs to keep the BiRRT goal out of contact with
# ~1 mm of IK error to spare. Small is still better -- it shortens the
# unplanned settle stroke.
_DROP = 0.003


def _pick_sampler(state: State, goal: Set[GroundAtom],
                  rng: np.random.Generator, objs: Sequence[Object]) -> Array:
    del state, goal, objs
    # Descend to ~the grasp target (block top). A tight range keeps the
    # grasp shallow and repeatable; Place/MoveTo compensate the actual
    # EE-to-held offset live, so the depth no longer needs budgeting
    # into release heights.
    return np.array([rng.uniform(0.0, 0.005)], dtype=np.float32)


# The generic MoveTo skill carries no glue semantics at all: gluing a
# face is just MoveTo with the sampler aiming the held bottle's TIP at
# the face's dab point (mirrors env._face_dab_point). MoveTo's params
# are the held object's target CENTER (live-compensated on all axes),
# so the tip lands 0-6 mm above the dab point exactly -- well inside
# the env's 2 cm wetting radius -- regardless of grasp depth or the
# pick's IK residual.
def _glue_end_b_sampler(state: State, goal: Set[GroundAtom],
                        rng: np.random.Generator,
                        objs: Sequence[Object]) -> Array:
    del goal
    # objs = [robot, bottle, block]: aim the held bottle's TIP at the
    # block's end_b dab point -- the face's top edge, so the dab comes
    # from above (mirrors env._face_dab_point). MoveTo's params are the
    # held object's target CENTER (live-compensated on all axes), so
    # the tip lands 0-6 mm above the dab point exactly -- well inside
    # the env's 2 cm wetting radius -- regardless of grasp depth or the
    # pick's IK residual.
    blk = objs[2]
    dab = _ENV._face_dab_point(state, blk, "end_b")  # pylint: disable=protected-access
    yaw = state.get(blk, "yaw")
    x, y = dab[0], dab[1]
    z = dab[2] + _ENV.bottle_half_extents[2] + rng.uniform(0.0, 0.006)
    return np.array([
        x + rng.uniform(-0.002, 0.002), y + rng.uniform(-0.002, 0.002), z, yaw
    ],
                    dtype=np.float32)


def _place_leg_at_site_sampler(state: State, goal: Set[GroundAtom],
                               rng: np.random.Generator,
                               objs: Sequence[Object]) -> Array:
    del goal
    # objs = [robot, leg, site]. Small jitter so refinement retries
    # differ (a deterministic sampler just repeats an identical failure).
    site = objs[2]
    x = state.get(site, "x") + rng.uniform(-0.003, 0.003)
    y = state.get(site, "y") + rng.uniform(-0.003, 0.003)
    # Standing legs are 2:1 blocks that topple from hard landings (an
    # 8-13 mm drop once landed a leg rocking near its tipping balance;
    # it leaned ~0.6 deg at release and slowly fell ~30 steps later).
    # The settle-to-contact release removes the free fall entirely, so
    # this is just the descend-goal clearance above resting height.
    z = _TABLE + _ENV.leg_half_extents[2] + 0.002 + rng.uniform(0.0, 0.001)
    return np.array([x, y, z, 0.0], dtype=np.float32)


def _place_next_to_sampler(state: State, goal: Set[GroundAtom],
                           rng: np.random.Generator,
                           objs: Sequence[Object]) -> Array:
    del goal
    # objs = [robot, right, left]: butt the held block against the left
    # block's end_b (+x) face, with a small nominal gap so the landing
    # does not shove the (wet-glued) left block out of alignment. Keep
    # the jitter tight and two-sided: the cure gate's projection window
    # reaches only ~1 cm past the nominal gap, and the params are frozen
    # at planning time, so the left block's OWN landing error stacks on
    # top of whatever outward bias the sampler adds (a one-sided
    # +0-4 mm jitter left a joint outside the window that then never
    # cured).
    left = objs[2]
    x = state.get(left, "x") + _SPAN_LEN + _ENV.lateral_place_gap + \
        rng.uniform(-0.001, 0.001)
    y = state.get(left, "y") + rng.uniform(-0.003, 0.003)
    # Any landing shift here is FROZEN into the weld and transfers to
    # the far seat joint; the settle-to-contact release means the block
    # touches down with no free fall, so this is just the descend-goal
    # clearance above resting height.
    z = _TABLE + _ENV.span_half_extents[2] + 0.002 + rng.uniform(0.0, 0.001)
    return np.array([x, y, z, 0.0], dtype=np.float32)


def _seat_span_sampler(state: State, goal: Set[GroundAtom],
                       rng: np.random.Generator,
                       objs: Sequence[Object]) -> Array:
    del goal
    # objs = [robot, spanA, mid, spanB, legL, legR, siteL, siteR]. The
    # welded row hangs from its grasped MIDDLE span (see PickRow), so
    # seating is near-symmetric: land mid's center on the midpoint of
    # the two leg tops and both outer spans arrive over their legs by
    # the rigid geometry. (An end grasp put a 20 cm cantilever on the
    # grasp constraint; its torsion yawed the far tip ~2-3 cm, enough
    # to strike the far leg's edge on the way down and topple it.)
    # Placement errors frozen into the welds make the row slightly
    # asymmetric about mid, so center the ROW -- the midpoint of the
    # outer spans' actual centers -- over the legs, not mid itself;
    # otherwise the whole frozen offset lands on one seat joint.
    span_a, mid, span_b = objs[1], objs[2], objs[3]
    row_dx = (state.get(span_a, "x") + state.get(span_b, "x")) / 2 - \
        state.get(mid, "x")
    row_dy = (state.get(span_a, "y") + state.get(span_b, "y")) / 2 - \
        state.get(mid, "y")
    leg_l, leg_r = objs[4], objs[5]
    x = (state.get(leg_l, "x") + state.get(leg_r, "x")) / 2 - row_dx + \
        rng.uniform(-0.003, 0.003)
    y = (state.get(leg_l, "y") + state.get(leg_r, "y")) / 2 - row_dy + \
        rng.uniform(-0.003, 0.003)
    # Release height from STATIC task geometry only. Samplers run at
    # planning time on predicted states, so live robot-relative reads
    # are stale garbage (a robot-z-based "hang" read the home pose,
    # blew past the release_z bound, and crash-dropped the assembly).
    # mid's center = leg top + span half-thickness + 2 cm descend
    # clearance for the carried rigid assembly. The settle-to-contact
    # release lowers the row until an outer span first touches a leg
    # top, so extra clearance costs nothing (no free fall to harden the
    # landing; the old 12 mm clearance's 2 cm-drop predecessor once
    # toppled the far leg). It needs to be generous: the descend-goal
    # collision check poses the welded partners from the welds' IDEAL
    # frames (flat relative dz), but the whole modeled row inherits the
    # held span's LIVE pitch through the grasp transform -- a carried
    # row rides at ~0.06-0.1 rad, which hangs an outer span 8-19 mm
    # below the held one, and at 12 mm clearance that modeled droop
    # collided with a leg top and killed otherwise-sound seat goals.
    release_z = _TABLE + _LEG_H + _ENV.span_half_extents[2] + 0.020
    return np.array([x, y, release_z, 0.0], dtype=np.float32)


def _footprint_radius(state: State, obj: Object) -> float:
    """Conservative horizontal footprint radius from the LIVE geometry (a
    toppled leg is span-sized; never dispatch on the name role)."""
    if obj.type.name == "bottle":
        return float(np.hypot(*_ENV.bottle_half_extents[:2]))
    half = _ENV._world_half_extents(state, obj)  # pylint: disable=protected-access
    return float(np.hypot(*half[:2]))


def _stage_spot_sampler(state: State, held: Object,
                        rng: np.random.Generator) -> np.ndarray:
    """A clear table spot for staging (escape hatch / bottle return).

    Candidates come from the env's staging grid, DEPRIORITIZING the
    front row: that row hosts the span-assembly strip, and the strip
    cells look empty until the very Place that needs them (returning
    the bottle there blocked the span row in early runs).

    Clearance is SIZE-AWARE (held footprint + neighbor footprint +
    margin): a blanket radius larger than the grid pitch rejects every
    cell in the packed full-variant grid -- even genuinely free ones --
    and the fallback then dropped the bottle on top of a staged block.
    """
    rows = [_ENV.stage_row_back, _ENV.stage_row_mid, _ENV.stage_row_front]
    candidates = []
    for row in rows:
        for col in _ENV.stage_cols:
            if np.hypot(col - _ENV.robot_base_pos[0],
                        row - _ENV.robot_base_pos[1]) > \
                    _ENV.reach_radius - 0.02:
                continue
            candidates.append((col, row))
    rng.shuffle(candidates)
    # Stable preference: back/mid rows before the front (strip) row.
    candidates.sort(key=lambda c: rows.index(c[1]))
    held_r = _footprint_radius(state, held)
    # Row-growth corridors: the span row grows in +x from each lying
    # span, so cells to a span's right at its y are future placement
    # targets even though they look empty now (the bottle parked there
    # once and the row build descended straight into it).
    corridors = []
    for o in state:
        if o.type.name != "block" or o == held:
            continue
        if _ENV._stands(state, o):  # pylint: disable=protected-access
            continue
        corridors.append((state.get(o, "x"), state.get(o, "y")))
    tx, ty = candidates[0]
    for col, row in candidates:
        clear = True
        for o in state:
            if o.type.name not in ("block", "bottle", "site") or o == held:
                continue
            if o.type.name == "site":
                # Keep sites usable for later leg placements.
                required = held_r + 0.045 + 0.01
            else:
                required = held_r + _footprint_radius(state, o) + 0.015
            if np.hypot(state.get(o, "x") - col,
                        state.get(o, "y") - row) < required:
                clear = False
                break
        if clear:
            for sx, sy in corridors:
                if abs(row - sy) < held_r + 0.06 and \
                        sx - 0.12 < col < sx + 0.35:
                    clear = False
                    break
        if clear:
            tx, ty = col, row
            break
    jitter = rng.uniform(-0.005, 0.005, size=2)
    return np.array([tx + jitter[0], ty + jitter[1]])


def _place_block_on_table_sampler(state: State, goal: Set[GroundAtom],
                                  rng: np.random.Generator,
                                  objs: Sequence[Object]) -> Array:
    del goal
    held = objs[1]
    tx, ty = _stage_spot_sampler(state, held, rng)
    # Live geometry: release at the held object's CURRENT resting
    # height (a toppled leg rests at span height, not leg height).
    held_half = _ENV._world_half_extents(state, held)  # pylint: disable=protected-access
    release_z = _TABLE + held_half[2] + _DROP
    return np.array([tx, ty, release_z, 0.0], dtype=np.float32)


def _place_bottle_sampler(state: State, goal: Set[GroundAtom],
                          rng: np.random.Generator,
                          objs: Sequence[Object]) -> Array:
    del goal
    bottle = objs[1]
    tx, ty = _stage_spot_sampler(state, bottle, rng)
    release_z = _TABLE + _ENV.bottle_half_extents[2] + _DROP
    return np.array([tx, ty, release_z, 0.0], dtype=np.float32)


class PyBulletBridgeGroundTruthProcessFactory(GroundTruthProcessFactory):
    """Ground-truth processes for the bridge environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"pybullet_bridge"}

    @staticmethod
    def get_processes(
            env_name: str, types: Dict[str, Type], predicates: Dict[str,
                                                                    Predicate],
            options: Dict[str, ParameterizedOption]) -> Set[CausalProcess]:
        del env_name
        robot_type = types["robot"]
        block_type = types["block"]
        bottle_type = types["bottle"]
        site_type = types["site"]

        HandEmpty = predicates["HandEmpty"]
        Holding = predicates["Holding"]
        HoldingBottle = predicates["HoldingBottle"]
        GlueEndB = predicates["GlueEndB"]
        NextToEnd = predicates["NextToEnd"]
        SeatedOn = predicates["SeatedOn"]
        AtSite = predicates["AtSite"]
        SiteFree = predicates["SiteFree"]
        Attached = predicates["Attached"]
        Standing = predicates["Standing"]
        Lying = predicates["Lying"]
        Loose = predicates["Loose"]
        Resting = predicates["Resting"]
        TopFree = predicates["TopFree"]
        EndsFree = predicates["EndsFree"]

        PickBlock = options["PickBlock"]
        PickBottle = options["PickBottle"]
        Place = options["Place"]
        MoveTo = options["MoveTo"]
        Wait = options["Wait"]

        processes: Set[CausalProcess] = set()

        def _delay(mu: float) -> DelayDistribution:
            return DiscreteGaussianDelay(mu=torch.tensor(mu),
                                         sigma=torch.tensor(0.1))

        # -- PickBlockFromTable ---------------------------------------------
        robot = Variable("?robot", robot_type)
        blk = Variable("?block", block_type)
        processes.add(
            EndogenousProcess(
                "PickBlockFromTable",
                [robot, blk],
                {
                    LiftedAtom(HandEmpty, [robot]),
                    # A block with another resting on it cannot be
                    # top-grasped: the grasp goal puts the palm inside
                    # the covering block (a replan once tried to pick a
                    # leg from UNDER the seated span and BiRRT rejected
                    # the goal forever).
                    LiftedAtom(TopFree, [blk]),
                    # No UNWELDED butt neighbor (derived predicate; see
                    # the env). Without this gate the planner exploits
                    # the NextToEnd frame bug: butt span2 against the
                    # still-staged span1, pick span1 away (NextToEnd
                    # has no delete here -- the neighbor is not a
                    # parameter), and count on the fictional joint
                    # curing. Welded neighbors are fine: the pick drags
                    # the assembly, so the adjacency survives.
                    LiftedAtom(EndsFree, [blk]),
                    # LOOSE blocks only: picking a welded block drags
                    # its whole assembly airborne, silently breaking
                    # the partners' Resting -- which a pending cure
                    # elsewhere in the assembly may depend on (a plan
                    # once picked span0 before the span1-span2 joint
                    # cured; welded span1 dangled and the joint never
                    # cured). Assembly picks go through PickRow, which
                    # requires the chain to be COMPLETE.
                    LiftedAtom(Loose, [blk]),
                },
                set(),
                set(),
                {LiftedAtom(Holding, [robot, blk])},
                {
                    LiftedAtom(HandEmpty, [robot]),
                    LiftedAtom(Resting, [blk]),
                },
                _delay(2.0),
                torch.tensor(1.0),
                PickBlock,
                [robot, blk],
                _pick_sampler))

        # -- PickRow (grasp the fully welded span row by its MIDDLE) ----------
        # The ONLY way to pick a welded block: requires the complete
        # Attached chain, so a partially cured row can never be lifted
        # (which would dangle the welded partner and break the pending
        # joint's resting contact). Grasping the MIDDLE span balances
        # the assembly: an end grasp cantilevers 20 cm of weldment off
        # the grasp constraint, whose torsion yaws the far tip ~2-3 cm
        # in flight -- enough to strike the far leg during seating.
        robot = Variable("?robot", robot_type)
        span_a = Variable("?spanA", block_type)
        mid = Variable("?spanMid", block_type)
        span_b = Variable("?spanB", block_type)
        processes.add(
            EndogenousProcess(
                "PickRow", [robot, span_a, mid, span_b], {
                    LiftedAtom(HandEmpty, [robot]),
                    LiftedAtom(TopFree, [mid]),
                    LiftedAtom(Attached, [span_a, mid]),
                    LiftedAtom(Attached, [mid, span_b]),
                    LiftedAtom(Lying, [span_a]),
                    LiftedAtom(Lying, [mid]),
                    LiftedAtom(Lying, [span_b]),
                }, set(), set(), {LiftedAtom(Holding, [robot, mid])}, {
                    LiftedAtom(HandEmpty, [robot]),
                    LiftedAtom(Resting, [span_a]),
                    LiftedAtom(Resting, [mid]),
                    LiftedAtom(Resting, [span_b]),
                }, _delay(2.0), torch.tensor(1.0), PickBlock, [robot, mid],
                _pick_sampler))

        # -- PickSpanFromRow (dismantle an UNCURED butt joint) ---------------
        # The Unstack analog: names the left neighbor so the adjacency
        # can be deleted, keeping the abstract state frame-correct.
        robot = Variable("?robot", robot_type)
        blk = Variable("?block", block_type)
        left = Variable("?left", block_type)
        processes.add(
            EndogenousProcess(
                "PickSpanFromRow",
                [robot, blk, left],
                {
                    LiftedAtom(HandEmpty, [robot]),
                    LiftedAtom(TopFree, [blk]),
                    LiftedAtom(NextToEnd, [blk, left]),
                    # Only for uncured joints: a welded block is picked
                    # via PickBlockFromTable and drags its partners.
                    LiftedAtom(Loose, [blk]),
                },
                set(),
                set(),
                {LiftedAtom(Holding, [robot, blk])},
                {
                    LiftedAtom(HandEmpty, [robot]),
                    LiftedAtom(Resting, [blk]),
                    LiftedAtom(NextToEnd, [blk, left]),
                },
                _delay(2.0),
                torch.tensor(1.0),
                PickBlock,
                [robot, blk],
                _pick_sampler))

        # -- PickGlueBottle ---------------------------------------------------
        robot = Variable("?robot", robot_type)
        bottle = Variable("?bottle", bottle_type)
        processes.add(
            EndogenousProcess("PickGlueBottle", [robot, bottle],
                              {LiftedAtom(HandEmpty, [robot])}, set(), set(),
                              {LiftedAtom(HoldingBottle, [robot, bottle])},
                              {LiftedAtom(HandEmpty, [robot])}, _delay(2.0),
                              torch.tensor(1.0), PickBottle, [robot, bottle],
                              _pick_sampler))

        # -- PlaceBottleOnTable -----------------------------------------------
        robot = Variable("?robot", robot_type)
        bottle = Variable("?bottle", bottle_type)
        processes.add(
            EndogenousProcess("PlaceBottleOnTable", [robot, bottle],
                              {LiftedAtom(HoldingBottle, [robot, bottle])},
                              set(), set(), {LiftedAtom(HandEmpty, [robot])},
                              {LiftedAtom(HoldingBottle, [robot, bottle])},
                              _delay(3.0), torch.tensor(1.0), Place, [robot],
                              _place_bottle_sampler))

        # -- PlaceLegAtSite ---------------------------------------------------
        robot = Variable("?robot", robot_type)
        leg = Variable("?leg", block_type)
        site = Variable("?site", site_type)
        processes.add(
            EndogenousProcess(
                "PlaceLegAtSite", [robot, leg, site], {
                    LiftedAtom(Holding, [robot, leg]),
                    LiftedAtom(SiteFree, [site]),
                    LiftedAtom(Standing, [leg]),
                    LiftedAtom(Loose, [leg]),
                }, set(), set(), {
                    LiftedAtom(HandEmpty, [robot]),
                    LiftedAtom(AtSite, [leg, site]),
                    LiftedAtom(Resting, [leg]),
                }, {
                    LiftedAtom(Holding, [robot, leg]),
                    LiftedAtom(SiteFree, [site]),
                }, _delay(3.0), torch.tensor(1.0), Place, [robot],
                _place_leg_at_site_sampler))

        # -- PlaceSpanNextTo --------------------------------------------------
        robot = Variable("?robot", robot_type)
        right = Variable("?right", block_type)
        left = Variable("?left", block_type)
        processes.add(
            EndogenousProcess(
                "PlaceSpanNextTo",
                [robot, right, left],
                {
                    LiftedAtom(Holding, [robot, right]),
                    LiftedAtom(GlueEndB, [left]),
                    LiftedAtom(Lying, [right]),
                    LiftedAtom(Lying, [left]),
                    # A block welded into an assembly cannot be
                    # individually re-placed (the weld drags the whole
                    # assembly along and the intended adjacency never
                    # forms). This also makes wrong-direction row
                    # builds a dead end instead of an attractive
                    # abstract shortcut.
                    LiftedAtom(Loose, [right]),
                },
                set(),
                set(),
                {
                    LiftedAtom(HandEmpty, [robot]),
                    LiftedAtom(NextToEnd, [right, left]),
                    LiftedAtom(Resting, [right]),
                },
                {LiftedAtom(Holding, [robot, right])},
                _delay(3.0),
                torch.tensor(1.0),
                Place,
                [robot],
                _place_next_to_sampler))

        # -- PlaceBlockOnTable (staging escape hatch) -------------------------
        robot = Variable("?robot", robot_type)
        blk = Variable("?block", block_type)
        processes.add(
            EndogenousProcess("PlaceBlockOnTable", [robot, blk],
                              {LiftedAtom(Holding, [robot, blk])}, set(),
                              set(), {
                                  LiftedAtom(HandEmpty, [robot]),
                                  LiftedAtom(Resting, [blk]),
                              }, {LiftedAtom(Holding, [robot, blk])},
                              _delay(3.0), torch.tensor(1.0), Place, [robot],
                              _place_block_on_table_sampler))

        # -- ApplyGlueEndB ----------------------------------------------------
        # Grounds to the generic MoveTo skill; the sampler aims the
        # held bottle's tip at the block's end_b dab point. The Lying
        # shape condition prunes groundings to the row joints the task
        # uses (end glue goes on lying spans).
        robot = Variable("?robot", robot_type)
        bottle = Variable("?bottle", bottle_type)
        blk = Variable("?block", block_type)
        processes.add(
            EndogenousProcess(
                "ApplyGlueEndB", [robot, bottle, blk], {
                    LiftedAtom(HoldingBottle, [robot, bottle]),
                    LiftedAtom(Lying, [blk]),
                }, set(), set(), {LiftedAtom(GlueEndB, [blk])}, set(),
                _delay(4.0), torch.tensor(1.0), MoveTo, [robot],
                _glue_end_b_sampler))

        # -- SeatSpan3 (place the welded span row across the legs) ------------
        # A 3-span row is the minimum that makes the glue structurally
        # necessary: the unglued middle span has no support and falls
        # into the gap (a 2-span row mutually supports as a friction
        # arch). The Attached chain forces the row to be fully welded
        # before seating, and the AtSite leg conditions force the legs
        # to actually be erected first -- without them the planner
        # happily seated the span onto legs still at their staged spots
        # and "moved" them to the sites afterwards. No glue conditions
        # on the seat: seat joints are neither structural (the welded
        # row rests on the legs by gravity) nor in the goal.
        robot = Variable("?robot", robot_type)
        span_a = Variable("?spanA", block_type)
        mid = Variable("?spanMid", block_type)
        span_b = Variable("?spanB", block_type)
        leg_l = Variable("?legL", block_type)
        leg_r = Variable("?legR", block_type)
        site_l = Variable("?siteL", site_type)
        site_r = Variable("?siteR", site_type)
        condition = {
            LiftedAtom(Holding, [robot, mid]),
            LiftedAtom(Lying, [span_a]),
            LiftedAtom(Lying, [mid]),
            LiftedAtom(Lying, [span_b]),
            LiftedAtom(Attached, [span_a, mid]),
            LiftedAtom(Attached, [mid, span_b]),
            LiftedAtom(Standing, [leg_l]),
            LiftedAtom(Standing, [leg_r]),
            LiftedAtom(AtSite, [leg_l, site_l]),
            LiftedAtom(AtSite, [leg_r, site_r]),
        }
        processes.add(
            EndogenousProcess(
                "SeatSpan3",
                [robot, span_a, mid, span_b, leg_l, leg_r, site_l, site_r],
                condition, set(), set(), {
                    LiftedAtom(HandEmpty, [robot]),
                    LiftedAtom(SeatedOn, [span_a, leg_l]),
                    LiftedAtom(SeatedOn, [span_b, leg_r]),
                    LiftedAtom(Resting, [span_a]),
                    LiftedAtom(Resting, [mid]),
                    LiftedAtom(Resting, [span_b]),
                }, {
                    LiftedAtom(Holding, [robot, mid]),
                    LiftedAtom(TopFree, [leg_l]),
                    LiftedAtom(TopFree, [leg_r]),
                }, _delay(3.0), torch.tensor(1.0), Place, [robot],
                _seat_span_sampler))

        # -- Wait -------------------------------------------------------------
        robot = Variable("?robot", robot_type)
        processes.add(
            EndogenousProcess("Wait", [robot], set(), set(),
                              set(), set(), set(), ConstantDelay(1),
                              torch.tensor(1.0), Wait, [robot], null_sampler))

        # -- Exogenous cure process (hidden dwell-time dynamics) --------------
        cure_delay = _delay(float(PyBulletBridgeEnv.cure_threshold))

        right = Variable("?right", block_type)
        left = Variable("?left", block_type)
        condition = {
            LiftedAtom(GlueEndB, [left]),
            LiftedAtom(NextToEnd, [right, left]),
            LiftedAtom(Resting, [right]),
            LiftedAtom(Resting, [left]),
        }
        processes.add(
            ExogenousProcess(
                "CureLateralJoint",
                [right, left],
                condition,
                condition.copy(),
                set(),
                {
                    # ONLY the goal-order atom (goals write
                    # Attached(left, right)). Adding both orders let the
                    # planner satisfy an Attached goal via the REVERSED
                    # row arrangement and then "rearrange" the welded
                    # blocks -- a physically impossible 33-step plan.
                    # The real classifier stays symmetric, so replans
                    # (whose initial atoms come from the classifier)
                    # are unaffected.
                    LiftedAtom(Attached, [left, right]),
                },
                {
                    LiftedAtom(GlueEndB, [left]),
                    LiftedAtom(Loose, [right]),
                    LiftedAtom(Loose, [left]),
                },
                cure_delay,
                torch.tensor(1.0)))

        return processes
