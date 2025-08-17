"""Ground-truth processes for the domino environment."""

from typing import Dict, Set

import torch

from predicators.ground_truth_models import GroundTruthProcessFactory
from predicators.settings import CFG
from predicators.structs import CausalProcess, EndogenousProcess, \
    ExogenousProcess, LiftedAtom, ParameterizedOption, Predicate, Type, \
    Variable
from predicators.utils import ConstantDelay, DiscreteGaussianDelay, \
    null_sampler


class PyBulletDominoGroundTruthProcessFactory(GroundTruthProcessFactory):
    """Ground-truth processes for the domino environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"pybullet_domino"}

    @classmethod
    def get_processes(
            cls, env_name: str, types: Dict[str,
                                            Type], predicates: Dict[str,
                                                                    Predicate],
            options: Dict[str, ParameterizedOption]) -> Set[CausalProcess]:
        del env_name  # unused

        # Types
        robot_type = types["robot"]
        domino_type = types["domino"]
        direction_type = types["direction"]

        # Predicates
        HandEmpty = predicates["HandEmpty"]
        Holding = predicates["Holding"]
        InFrontDirection = predicates["InFrontDirection"]
        InFront = predicates["InFront"]
        NotInFrontOfAny = predicates["NotInFrontOfAny"]
        StartBlock = predicates["StartBlock"]
        Toppled = predicates["Toppled"]
        # Note: Toppled predicate exists but represents the goal state
        # Note: The "Falling" predicate from the sketch is not implemented in the current environment
        # We would need to add it to the environment for the DominoFall exogenous process

        # Options
        Push = options["Push"]
        Pick = options["Pick"]
        Place = options["Place"]
        NoOp = options["NoOp"]

        processes: Set[CausalProcess] = set()

        # --- Endogenous Processes / Actions ---

        # PushStartBlock: Push the start block to initiate the domino chain
        robot = Variable("?robot", robot_type)
        domino = Variable("?domino", domino_type)
        parameters = [robot, domino]
        option_vars = [robot, domino]
        option = Push
        condition_at_start = {
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(StartBlock, [domino]),
        }
        add_effects = {
            LiftedAtom(Toppled, [domino]),
        }
        delete_effects: Set[LiftedAtom] = set()
        delay_distribution = DiscreteGaussianDelay(mu=torch.tensor(1.0),
                                                   sigma=torch.tensor(0.1))
        push_start_block_process = EndogenousProcess(
            "PushStartBlock", parameters, condition_at_start, set(),
            set(), add_effects, delete_effects, delay_distribution,
            torch.tensor(1.0), option, option_vars, null_sampler)
        processes.add(push_start_block_process)

        # PickDominoIsolated: Pick domino only when not in front of anything
        robot = Variable("?robot", robot_type)
        domino = Variable("?domino", domino_type)
        parameters = [robot, domino]
        option_vars = [robot, domino]
        option = Pick  # Using Push as the underlying action
        condition_at_start = {
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(NotInFrontOfAny, [domino]),
        }
        add_effects = {
            LiftedAtom(Holding, [robot, domino]),
        }
        delete_effects = {
            LiftedAtom(HandEmpty, [robot]),
        }
        delay_distribution = DiscreteGaussianDelay(mu=torch.tensor(4.0),
                                                   sigma=torch.tensor(0.1))
        pick_domino_isolated_process = EndogenousProcess(
            "PickDominoIsolated", parameters, condition_at_start, set(),
            set(), add_effects, delete_effects, delay_distribution,
            torch.tensor(1.0), option, option_vars, null_sampler)
        processes.add(pick_domino_isolated_process)

        # PickDominoClearOut: Clear exactly one relation where domino is in front of other
        robot = Variable("?robot", robot_type)
        domino = Variable("?domino", domino_type)
        other_domino = Variable("?other", domino_type)
        direction = Variable("?dir", direction_type)
        parameters = [robot, domino, other_domino, direction]
        option_vars = [robot, domino]
        option = Pick
        condition_at_start = {
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(InFrontDirection, [domino, other_domino, direction]),
        }
        add_effects = {
            LiftedAtom(Holding, [robot, domino]),
        }
        delete_effects = {
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(InFrontDirection, [domino, other_domino, direction]),
        }
        delay_distribution = DiscreteGaussianDelay(mu=torch.tensor(4.0),
                                                   sigma=torch.tensor(0.1))
        pick_domino_clear_out_process = EndogenousProcess(
            "PickDominoClearOut", parameters, condition_at_start, set(),
            set(), add_effects, delete_effects, delay_distribution,
            torch.tensor(1.0), option, option_vars, null_sampler)
        processes.add(pick_domino_clear_out_process)

        # PickDominoClearIn: Clear exactly one relation where other is in front of domino
        robot = Variable("?robot", robot_type)
        domino = Variable("?domino", domino_type)
        other_domino = Variable("?other", domino_type)
        direction = Variable("?dir", direction_type)
        parameters = [robot, domino, other_domino, direction]
        option_vars = [robot, domino]
        option = Pick
        condition_at_start = {
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(InFrontDirection, [other_domino, domino, direction]),
        }
        add_effects = {
            LiftedAtom(Holding, [robot, domino]),
        }
        delete_effects = {
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(InFrontDirection, [other_domino, domino, direction]),
        }
        delay_distribution = DiscreteGaussianDelay(mu=torch.tensor(4.0),
                                                   sigma=torch.tensor(0.1))
        pick_domino_clear_in_process = EndogenousProcess(
            "PickDominoClearIn", parameters, condition_at_start, set(),
            set(), add_effects, delete_effects, delay_distribution,
            torch.tensor(1.0), option, option_vars, null_sampler)
        processes.add(pick_domino_clear_in_process)

        # PlaceDomino: Place domino to create InFrontDirection relation
        robot = Variable("?robot", robot_type)
        domino1 = Variable("?domino1", domino_type)
        domino2 = Variable("?domino2", domino_type)
        direction = Variable("?dir", direction_type)
        parameters = [robot, domino1, domino2, direction]
        option_vars = [robot, domino1, domino2, direction]
        option = Place  # Using Push as the underlying action for placement
        condition_at_start = {
            LiftedAtom(Holding, [robot, domino1]),
        }
        add_effects = {
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(InFrontDirection, [domino1, domino2, direction]),
        }
        delete_effects = {
            LiftedAtom(Holding, [robot, domino1]),
        }
        delay_distribution = DiscreteGaussianDelay(mu=torch.tensor(3.0),
                                                   sigma=torch.tensor(0.1))
        place_domino_process = EndogenousProcess("PlaceDomino", parameters,
                                                 condition_at_start, set(),
                                                 set(), add_effects,
                                                 delete_effects,
                                                 delay_distribution,
                                                 torch.tensor(1.0), option,
                                                 option_vars, null_sampler)
        processes.add(place_domino_process)

        # NoOp
        robot = Variable("?robot", robot_type)
        parameters = [robot]
        option_vars = [robot]
        option = NoOp
        noop_delay_distribution = ConstantDelay(1)
        noop_process = EndogenousProcess("NoOp", parameters, set(), set(),
                                         set(), set(), set(),
                                         noop_delay_distribution,
                                         torch.tensor(1.0), option,
                                         option_vars, null_sampler)
        processes.add(noop_process)

        # --- Exogenous Processes ---

        # Note: The DominoFall process from the sketch requires a "Falling" predicate
        # which is not currently implemented in the environment.
        # This process would look like:
        #
        domino1 = Variable("?d1", domino_type)
        domino2 = Variable("?d2", domino_type)
        parameters = [domino1, domino2]
        condition_at_start = {
            LiftedAtom(InFront, [domino1, domino2]),
            LiftedAtom(Toppled, [domino2]),  # This predicate doesn't exist yet
        }
        condition_overall = condition_at_start.copy()
        add_effects = {
            LiftedAtom(Toppled, [domino1]),
        }
        delay_distribution = DiscreteGaussianDelay(mu=torch.tensor(1.0),
                                                   sigma=torch.tensor(0.1))
        domino_fall_process = ExogenousProcess("DominoFall", parameters,
                                               condition_at_start,
                                               condition_overall, set(),
                                               add_effects, set(),
                                               delay_distribution,
                                               torch.tensor(1.0))
        processes.add(domino_fall_process)

        return processes
