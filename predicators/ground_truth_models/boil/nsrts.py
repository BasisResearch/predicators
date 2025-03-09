"""Ground-truth NSRTs for the coffee environment."""

from typing import Dict, Sequence, Set

import numpy as np

from predicators.ground_truth_models import GroundTruthNSRTFactory
from predicators.structs import NSRT, Array, GroundAtom, LiftedAtom, Object, \
    ParameterizedOption, Predicate, State, Type, Variable
from predicators.utils import null_sampler


class PyBulletBoilGroundTruthNSRTFactory(GroundTruthNSRTFactory):
    """Ground-truth NSRTs for the boil environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"pybullet_boil"}

    @staticmethod
    def get_nsrts(env_name: str, types: Dict[str, Type],
                  predicates: Dict[str, Predicate],
                  options: Dict[str, ParameterizedOption]) -> Set[NSRT]:
        # Types
        robot_type = types["robot"]
        jug_type = types["jug"]
        burner_type = types["burner"]

        # Predicates
        HandEmpty = predicates["HandEmpty"]
        Holding = predicates["Holding"]
        JugOnBurner = predicates["JugOnBurner"]

        # Options
        PickJug = options["PickJug"]
        PlaceOnBurner = options["PlaceOnBurner"]

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
        }
        pick_jug_from_table_nsrt = NSRT("PickJugFromTable", parameters,
                                        preconditions, add_effects,
                                        delete_effects, set(), option,
                                        option_vars, null_sampler)
        nsrts.add(pick_jug_from_table_nsrt)

        # Place
        robot = Variable("?robot", robot_type)
        jug = Variable("?jug", jug_type)
        burner = Variable("?burner", burner_type)
        parameters = [robot, jug, burner]
        option_vars = [robot, burner]
        option = PlaceOnBurner
        preconditions = {
            LiftedAtom(Holding, [robot, jug]),
        }
        add_effects = {
            LiftedAtom(JugOnBurner, [jug, burner]),
            LiftedAtom(HandEmpty, [robot]),
        }
        delete_effects = {
            LiftedAtom(Holding, [robot, jug]),
        }

        place = NSRT("PlaceOnBurner", parameters, preconditions, add_effects,
                     delete_effects, set(), option, option_vars, null_sampler)
        nsrts.add(place)

        return nsrts
