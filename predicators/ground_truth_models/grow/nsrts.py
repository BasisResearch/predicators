"""Ground-truth NSRTs for the coffee environment."""

from typing import Dict, Set

from predicators.ground_truth_models import GroundTruthNSRTFactory
from predicators.structs import NSRT, LiftedAtom, ParameterizedOption, \
    Predicate, Type, Variable
from predicators.utils import null_sampler


class PyBulletGrowGroundTruthNSRTFactory(GroundTruthNSRTFactory):
    """Ground-truth NSRTs for the grow environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"pybullet_grow"}

    @staticmethod
    def get_nsrts(env_name: str, types: Dict[str, Type],
                  predicates: Dict[str, Predicate],
                  options: Dict[str, ParameterizedOption]) -> Set[NSRT]:
        # Types
        robot_type = types["robot"]
        jug_type = types["jug"]
        cup_type = types["cup"]
        # Predicates
        HandEmpty = predicates["HandEmpty"]
        Holding = predicates["Holding"]
        Grown = predicates["Grown"]
        OnTable = predicates["OnTable"]
        SameColor = predicates["SameColor"]
        # Options
        PickJug = options["PickJug"]
        Pour = options["Pour"]
        Place = options["Place"]

        nsrts = set()

        # PickJug
        robot = Variable("?robot", robot_type)
        jug = Variable("?jug", jug_type)
        parameters = [robot, jug]
        option_vars = [robot, jug]
        option = PickJug
        preconditions = {
            LiftedAtom(HandEmpty, [robot]),
        }
        add_effects = {
            LiftedAtom(Holding, [robot, jug]),
        }
        delete_effects = {
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(OnTable, [jug])
        }
        pick_jug_from_table_nsrt = NSRT("PickJugFromTable", parameters,
                                        preconditions, add_effects,
                                        delete_effects, set(), option,
                                        option_vars, null_sampler)
        nsrts.add(pick_jug_from_table_nsrt)

        # Pour
        robot = Variable("?robot", robot_type)
        jug = Variable("?jug", jug_type)
        cup = Variable("?cup", cup_type)
        parameters = [robot, jug, cup]
        option_vars = [robot, jug, cup]
        option = Pour
        preconditions = {
            LiftedAtom(Holding, [robot, jug]),
            LiftedAtom(SameColor, [cup, jug]),
        }
        add_effects = {
            LiftedAtom(Grown, [cup]),
        }
        pour = NSRT("Pour", parameters, preconditions, add_effects, set(),
                    set(), option, option_vars, null_sampler)
        nsrts.add(pour)

        # Place
        robot = Variable("?robot", robot_type)
        jug = Variable("?jug", jug_type)
        parameters = [robot, jug]
        option_vars = [robot, jug]
        option = Place
        preconditions = {
            LiftedAtom(Holding, [robot, jug]),
        }
        add_effects = {
            LiftedAtom(OnTable, [jug]),
            LiftedAtom(HandEmpty, [robot]),
        }
        delete_effects = {
            LiftedAtom(Holding, [robot, jug]),
        }
        place = NSRT("PlaceJug", parameters, preconditions, add_effects,
                     delete_effects, set(), option, option_vars, null_sampler)
        nsrts.add(place)

        return nsrts
