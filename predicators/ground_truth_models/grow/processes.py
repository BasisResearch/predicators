"""Ground-truth processes for the grow environments."""

from typing import Dict, Set

import torch

from predicators.ground_truth_models import GroundTruthProcessFactory
from predicators.structs import CausalProcess, EndogenousProcess, \
    ExogenousProcess, LiftedAtom, ParameterizedOption, Predicate, Type, \
    Variable
from predicators.utils import DiscreteGaussianDelay, \
    null_sampler


class PyBulletGrowGroundTruthProcessFactory(GroundTruthProcessFactory):
    """Ground-truth processes for the grow environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"pybullet_grow"}

    @classmethod
    def get_processes(
            cls, env_name: str, types: Dict[str,
                                            Type], predicates: Dict[str,
                                                                    Predicate],
            options: Dict[str, ParameterizedOption]) -> Set[CausalProcess]:
        # Types
        robot_type = types["robot"]
        jug_type = types["jug"]
        cup_type = types["cup"]

        # Predicates
        Grown = predicates["Grown"]
        Holding = predicates["Holding"]
        HandEmpty = predicates["HandEmpty"]
        JugOnTable = predicates["JugOnTable"]
        SameColor = predicates["SameColor"]
        JugAboveCup = predicates["JugAboveCup"]

        # Options
        PickJug = options["PickJug"]
        Pour = options["Pour"]
        Place = options["Place"]
        NoOp = options["NoOp"]

        processes: Set[CausalProcess] = set()

        # --- Endogenous Processes / Durative Actions ---

        # PickJugFromTable
        robot = Variable("?robot", robot_type)
        jug = Variable("?jug", jug_type)
        parameters = [robot, jug]
        option_vars = [robot, jug]
        option = PickJug
        condition_at_start = {
            LiftedAtom(JugOnTable, [jug]),
            LiftedAtom(HandEmpty, [robot]),
        }
        add_effects = {
            LiftedAtom(Holding, [robot, jug]),
        }
        delete_effects = {
            LiftedAtom(JugOnTable, [jug]),
            LiftedAtom(HandEmpty, [robot])
        }
        delay_distribution = DiscreteGaussianDelay(mu=torch.tensor(3.0),
                                                   sigma=torch.tensor(0.1))
        pick_jug_from_table_process = EndogenousProcess(
            "PickJugFromTable", parameters, condition_at_start, set(),
            set(), add_effects, delete_effects, delay_distribution,
            torch.tensor(1.0), option, option_vars, null_sampler)
        processes.add(pick_jug_from_table_process)

        # PlaceJugOnTableFromAboveCup
        robot = Variable("?robot", robot_type)
        jug = Variable("?jug", jug_type)
        cup = Variable("?cup", cup_type)
        parameters = [robot, jug, cup]
        option_vars = [robot, jug]
        option = Place
        condition_at_start = {
            LiftedAtom(Holding, [robot, jug]),
            LiftedAtom(JugAboveCup, [jug, cup]),
        }
        add_effects = {
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(JugOnTable, [jug]),
        }
        delete_effects = {
            LiftedAtom(Holding, [robot, jug]),
            LiftedAtom(JugAboveCup, [jug, cup]),
        }
        delay_distribution = DiscreteGaussianDelay(mu=torch.tensor(3.0),
                                                   sigma=torch.tensor(0.1))
        place_jug_on_table_process = EndogenousProcess(
            "PlaceJugOnTable", parameters, condition_at_start, set(),
            set(), add_effects, delete_effects, delay_distribution,
            torch.tensor(1.0), option, option_vars, null_sampler)
        processes.add(place_jug_on_table_process)

        # Pour (positions jug above cup)
        robot = Variable("?robot", robot_type)
        jug = Variable("?jug", jug_type)
        cup = Variable("?cup", cup_type)
        parameters = [robot, jug, cup]
        option_vars = [robot, jug, cup]
        option = Pour
        condition_at_start = {
            LiftedAtom(Holding, [robot, jug]),
        }
        add_effects = {
            LiftedAtom(JugAboveCup, [jug, cup]),
        }
        delete_effects = set()
        delay_distribution = DiscreteGaussianDelay(mu=torch.tensor(2.0),
                                                   sigma=torch.tensor(0.1))
        pour_process = EndogenousProcess(
            "Pour", parameters, condition_at_start, set(),
            set(), add_effects, delete_effects, delay_distribution,
            torch.tensor(1.0), option, option_vars, null_sampler)
        processes.add(pour_process)

        # NoOp
        robot = Variable("?robot", robot_type)
        parameters = [robot]
        option_vars = [robot]
        option = NoOp
        delay_distribution = DiscreteGaussianDelay(mu=torch.tensor(1.0),
                                                   sigma=torch.tensor(0.1))
        noop_process = EndogenousProcess("NoOp", parameters, set(), set(),
                                         set(), set(),
                                         set(), delay_distribution,
                                         torch.tensor(1.0), option,
                                         option_vars, null_sampler)
        processes.add(noop_process)

        # --- Exogenous Processes ---
        
        # GrowPlant (Exogenous) - similar to CupFilled in coffee
        jug = Variable("?jug", jug_type)
        cup = Variable("?cup", cup_type)
        parameters = [jug, cup]
        condition_at_start = {
            LiftedAtom(JugAboveCup, [jug, cup]),
            LiftedAtom(SameColor, [cup, jug]),
        }
        condition_overall = {
            LiftedAtom(JugAboveCup, [jug, cup]),
            LiftedAtom(SameColor, [cup, jug]),
        }
        add_effects = {
            LiftedAtom(Grown, [cup]),
        }
        delete_effects_grow_plant: Set[LiftedAtom] = set()
        delay_distribution = DiscreteGaussianDelay(mu=torch.tensor(5.0),
                                                   sigma=torch.tensor(0.1))
        grow_plant_process = ExogenousProcess(
            "GrowPlant", parameters, condition_at_start, condition_overall,
            set(), add_effects, delete_effects_grow_plant, delay_distribution,
            torch.tensor(1.0))
        processes.add(grow_plant_process)

        return processes
