"""Ground-truth NSRTs for the bridge (glue construction) environment.

Manipulation NSRTs (pick/place/butt-join/seat/glue/wait) mirror the
endogenous processes. There is deliberately no NSRT that adds
``Attached`` -- curing is a delayed exogenous process, so the
environment is solved/demoed by ``oracle_process_planning`` (like
``pybullet_boil`` and ``pybullet_bond``), not plain sesame ``oracle``.
"""

from typing import Dict, Set

from predicators.ground_truth_models import GroundTruthNSRTFactory
from predicators.ground_truth_models.bridge.processes import \
    _glue_end_b_sampler, _pick_sampler, _place_block_on_table_sampler, \
    _place_bottle_sampler, _place_leg_at_site_sampler, \
    _place_next_to_sampler, _seat_span_sampler
from predicators.structs import NSRT, LiftedAtom, ParameterizedOption, \
    Predicate, Type, Variable
from predicators.utils import null_sampler


class PyBulletBridgeGroundTruthNSRTFactory(GroundTruthNSRTFactory):
    """Ground-truth NSRTs for the bridge environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"pybullet_bridge"}

    @staticmethod
    def get_nsrts(env_name: str, types: Dict[str, Type],
                  predicates: Dict[str, Predicate],
                  options: Dict[str, ParameterizedOption]) -> Set[NSRT]:
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

        nsrts: Set[NSRT] = set()

        # PickBlockFromTable (see processes.py for the EndsFree and
        # Loose rationales; welded assemblies go through PickRow).
        robot = Variable("?robot", robot_type)
        blk = Variable("?block", block_type)
        nsrts.add(
            NSRT(
                "PickBlockFromTable", [robot, blk], {
                    LiftedAtom(HandEmpty, [robot]),
                    LiftedAtom(TopFree, [blk]),
                    LiftedAtom(EndsFree, [blk]),
                    LiftedAtom(Loose, [blk]),
                }, {LiftedAtom(Holding, [robot, blk])}, {
                    LiftedAtom(HandEmpty, [robot]),
                    LiftedAtom(Resting, [blk]),
                }, set(), PickBlock, [robot, blk], _pick_sampler))

        # PickRow (grasp the MIDDLE span of the FULLY welded row; see
        # processes.py -- a partially cured row must never be lifted,
        # and an end grasp cantilevers the row into the far leg).
        robot = Variable("?robot", robot_type)
        span_a = Variable("?spanA", block_type)
        mid = Variable("?spanMid", block_type)
        span_b = Variable("?spanB", block_type)
        nsrts.add(
            NSRT(
                "PickRow", [robot, span_a, mid, span_b], {
                    LiftedAtom(HandEmpty, [robot]),
                    LiftedAtom(TopFree, [mid]),
                    LiftedAtom(Attached, [span_a, mid]),
                    LiftedAtom(Attached, [mid, span_b]),
                    LiftedAtom(Lying, [span_a]),
                    LiftedAtom(Lying, [mid]),
                    LiftedAtom(Lying, [span_b]),
                }, {LiftedAtom(Holding, [robot, mid])}, {
                    LiftedAtom(HandEmpty, [robot]),
                    LiftedAtom(Resting, [span_a]),
                    LiftedAtom(Resting, [mid]),
                    LiftedAtom(Resting, [span_b]),
                }, set(), PickBlock, [robot, mid], _pick_sampler))

        # PickSpanFromRow (dismantle an uncured butt joint; deletes the
        # named adjacency to stay frame-correct).
        robot = Variable("?robot", robot_type)
        blk = Variable("?block", block_type)
        left = Variable("?left", block_type)
        nsrts.add(
            NSRT(
                "PickSpanFromRow", [robot, blk, left], {
                    LiftedAtom(HandEmpty, [robot]),
                    LiftedAtom(TopFree, [blk]),
                    LiftedAtom(NextToEnd, [blk, left]),
                    LiftedAtom(Loose, [blk]),
                }, {LiftedAtom(Holding, [robot, blk])}, {
                    LiftedAtom(HandEmpty, [robot]),
                    LiftedAtom(Resting, [blk]),
                    LiftedAtom(NextToEnd, [blk, left]),
                }, set(), PickBlock, [robot, blk], _pick_sampler))

        # PickGlueBottle
        robot = Variable("?robot", robot_type)
        bottle = Variable("?bottle", bottle_type)
        nsrts.add(
            NSRT("PickGlueBottle", [robot, bottle],
                 {LiftedAtom(HandEmpty, [robot])},
                 {LiftedAtom(HoldingBottle, [robot, bottle])},
                 {LiftedAtom(HandEmpty, [robot])}, set(), PickBottle,
                 [robot, bottle], _pick_sampler))

        # PlaceBottleOnTable
        robot = Variable("?robot", robot_type)
        bottle = Variable("?bottle", bottle_type)
        nsrts.add(
            NSRT("PlaceBottleOnTable", [robot, bottle],
                 {LiftedAtom(HoldingBottle, [robot, bottle])},
                 {LiftedAtom(HandEmpty, [robot])},
                 {LiftedAtom(HoldingBottle, [robot, bottle])}, set(), Place,
                 [robot], _place_bottle_sampler))

        # PlaceLegAtSite
        robot = Variable("?robot", robot_type)
        leg = Variable("?leg", block_type)
        site = Variable("?site", site_type)
        nsrts.add(
            NSRT(
                "PlaceLegAtSite", [robot, leg, site], {
                    LiftedAtom(Holding, [robot, leg]),
                    LiftedAtom(SiteFree, [site]),
                    LiftedAtom(Standing, [leg]),
                    LiftedAtom(Loose, [leg]),
                }, {
                    LiftedAtom(HandEmpty, [robot]),
                    LiftedAtom(AtSite, [leg, site]),
                    LiftedAtom(Resting, [leg]),
                }, {
                    LiftedAtom(Holding, [robot, leg]),
                    LiftedAtom(SiteFree, [site]),
                }, set(), Place, [robot], _place_leg_at_site_sampler))

        # PlaceSpanNextTo
        robot = Variable("?robot", robot_type)
        right = Variable("?right", block_type)
        left = Variable("?left", block_type)
        nsrts.add(
            NSRT(
                "PlaceSpanNextTo", [robot, right, left], {
                    LiftedAtom(Holding, [robot, right]),
                    LiftedAtom(GlueEndB, [left]),
                    LiftedAtom(Lying, [right]),
                    LiftedAtom(Lying, [left]),
                    LiftedAtom(Loose, [right]),
                }, {
                    LiftedAtom(HandEmpty, [robot]),
                    LiftedAtom(NextToEnd, [right, left]),
                    LiftedAtom(Resting, [right]),
                }, {LiftedAtom(Holding, [robot, right])}, set(), Place,
                [robot], _place_next_to_sampler))

        # PlaceBlockOnTable
        robot = Variable("?robot", robot_type)
        blk = Variable("?block", block_type)
        nsrts.add(
            NSRT("PlaceBlockOnTable", [robot, blk],
                 {LiftedAtom(Holding, [robot, blk])}, {
                     LiftedAtom(HandEmpty, [robot]),
                     LiftedAtom(Resting, [blk]),
                 }, {LiftedAtom(Holding, [robot, blk])}, set(), Place, [robot],
                 _place_block_on_table_sampler))

        # ApplyGlueEndB (grounds to the generic MoveTo skill; the
        # sampler aims at the end_b dab point).
        robot = Variable("?robot", robot_type)
        bottle = Variable("?bottle", bottle_type)
        blk = Variable("?block", block_type)
        nsrts.add(
            NSRT(
                "ApplyGlueEndB", [robot, bottle, blk], {
                    LiftedAtom(HoldingBottle, [robot, bottle]),
                    LiftedAtom(Lying, [blk]),
                }, {LiftedAtom(GlueEndB, [blk])}, set(), set(), MoveTo,
                [robot], _glue_end_b_sampler))

        # SeatSpan3 (see processes.py for the condition rationale --
        # this mirrors the endogenous process exactly).
        robot = Variable("?robot", robot_type)
        span_a = Variable("?spanA", block_type)
        mid = Variable("?spanMid", block_type)
        span_b = Variable("?spanB", block_type)
        leg_l = Variable("?legL", block_type)
        leg_r = Variable("?legR", block_type)
        site_l = Variable("?siteL", site_type)
        site_r = Variable("?siteR", site_type)
        nsrts.add(
            NSRT(
                "SeatSpan3",
                [robot, span_a, mid, span_b, leg_l, leg_r, site_l, site_r], {
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
                }, {
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
                }, set(), Place, [robot], _seat_span_sampler))

        # Wait
        robot = Variable("?robot", robot_type)
        nsrts.add(
            NSRT("Wait", [robot], set(), set(), set(), set(), Wait, [robot],
                 null_sampler))

        return nsrts
