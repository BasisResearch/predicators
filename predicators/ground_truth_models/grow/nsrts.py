"""Ground-truth NSRTs for the grow environment."""

from typing import Dict, Sequence, Set

import numpy as np

from predicators.ground_truth_models import GroundTruthNSRTFactory
from predicators.ground_truth_models.grow.processes import _pick_sampler, \
    _place_sampler
from predicators.settings import CFG
from predicators.structs import NSRT, Array, GroundAtom, LiftedAtom, Object, \
    ParameterizedOption, Predicate, State, Type, Variable
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
        JugOnTable = predicates["JugOnTable"]
        SameColor = predicates["SameColor"]
        # Options
        PickJug = options["PickJug"]
        Pour = options["Pour"]
        Place = options["Place"]

        nsrts = set()

        def putontable_sampler(state: State, goal: Set[GroundAtom],
                               rng: np.random.Generator,
                               objs: Sequence[Object]) -> Array:
            del state, goal, objs  # unused
            # Note: normalized coordinates w.r.t. workspace.
            x = rng.uniform()
            y = rng.uniform(0.5, 0.5)
            return np.array([x, y], dtype=np.float32)

        # The skill-factory options (the default) take a grasp height for
        # PickJug and a world (x, y, release_z, yaw) target for Place, the
        # parameters the processes sample; the legacy options take none for
        # PickJug and a normalized (x, y) for Place.
        if CFG.grow_use_skill_factories:
            pick_sampler, place_sampler = _pick_sampler, _place_sampler
        else:
            pick_sampler, place_sampler = null_sampler, putontable_sampler

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
            LiftedAtom(JugOnTable, [jug])
        }
        pick_jug_from_table_nsrt = NSRT("PickJugFromTable", parameters,
                                        preconditions, add_effects,
                                        delete_effects, set(), option,
                                        option_vars, pick_sampler)
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
            LiftedAtom(JugOnTable, [jug]),
            LiftedAtom(HandEmpty, [robot]),
        }
        delete_effects = {
            LiftedAtom(Holding, [robot, jug]),
        }
        place = NSRT("PlaceJug", parameters, preconditions, add_effects,
                     delete_effects, set(), option, option_vars, place_sampler)
        nsrts.add(place)

        return nsrts
