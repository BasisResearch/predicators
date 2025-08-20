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
        position_type = types["loc"]
        rotation_type = types["rot"]

        # Predicates
        HandEmpty = predicates["HandEmpty"]
        Holding = predicates["Holding"]
        InFront = predicates["InFront"]
        Upright = predicates["Upright"]
        StartBlock = predicates["StartBlock"]
        Toppled = predicates["Toppled"]
        DominoAtPos = predicates["DominoAtPos"]
        DominoAtRot = predicates["DominoAtRot"]
        MovableBlock = predicates["MovableBlock"]
        PosClear = predicates["PosClear"]
        if CFG.domino_include_connected_predicate:
            Connected = predicates["Connected"]
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
            LiftedAtom(Upright, [domino]),
        }
        add_effects = {
            LiftedAtom(Toppled, [domino]),
        }
        delete_effects: Set[LiftedAtom] = {
            LiftedAtom(Upright, [domino]),
        }
        delay_distribution = DiscreteGaussianDelay(mu=torch.tensor(1.0),
                                                   sigma=torch.tensor(0.1))
        push_start_block_process = EndogenousProcess(
            "PushStartBlock", parameters, condition_at_start, set(),
            set(), add_effects, delete_effects, delay_distribution,
            torch.tensor(1.0), option, option_vars, null_sampler)
        processes.add(push_start_block_process)

        # PickDomino: Position-based pick process
        robot = Variable("?robot", robot_type)
        domino = Variable("?domino", domino_type)
        position = Variable("?pos", position_type)
        rotation = Variable("?rot", rotation_type)
        parameters = [robot, domino, position, rotation]
        option_vars = [robot, domino]
        option = Pick
        condition_at_start = {
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(DominoAtPos, [domino, position]),
            LiftedAtom(DominoAtRot, [domino, rotation]),
            LiftedAtom(MovableBlock, [domino]),
            LiftedAtom(Upright, [domino]),
        }
        add_effects = {
            LiftedAtom(Holding, [robot, domino]),
            LiftedAtom(PosClear, [position]),
        }
        delete_effects = {
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(DominoAtPos, [domino, position]),
            LiftedAtom(DominoAtRot, [domino, rotation]),
        }
        delay_distribution = DiscreteGaussianDelay(mu=torch.tensor(4.0),
                                                   sigma=torch.tensor(0.1))
        pick_domino_process = EndogenousProcess("PickDomino",
                                                parameters, condition_at_start,
                                                set(), set(), add_effects,
                                                delete_effects,
                                                delay_distribution,
                                                torch.tensor(1.0), option,
                                                option_vars, null_sampler)
        processes.add(pick_domino_process)

        # PlaceDomino: Place domino at specific position and rotation
        # Not in will still be in front to something
        robot = Variable("?robot", robot_type)
        domino1 = Variable("?domino1", domino_type)
        domino2 = Variable("?domino2", domino_type)
        target_pos = Variable("?pos1", position_type)
        rotation = Variable("?rot", rotation_type)
        if CFG.domino_include_connected_predicate:
            d2_pos = Variable("?pos2", position_type)
            parameters = [
                robot, domino1, domino2, target_pos, d2_pos, rotation
            ]
        else:
            parameters = [robot, domino1, domino2, target_pos, rotation]
        option_vars = [robot, domino1, domino2, target_pos, rotation]
        option = Place
        condition_at_start = {
            LiftedAtom(Holding, [robot, domino1]),
            LiftedAtom(PosClear, [target_pos]),
        }
        if CFG.domino_include_connected_predicate:
            condition_at_start.update({
                LiftedAtom(DominoAtPos, [domino2, d2_pos]),
                LiftedAtom(Connected, [target_pos, d2_pos]),
            })
        add_effects = {
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(DominoAtPos, [domino1, target_pos]),
            LiftedAtom(DominoAtRot, [domino1, rotation]),
        }
        delete_effects = {
            LiftedAtom(Holding, [robot, domino1]),
            LiftedAtom(PosClear, [target_pos]),
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
            LiftedAtom(Toppled, [domino2]),
        }
        condition_overall = condition_at_start.copy()
        add_effects = {
            LiftedAtom(Toppled, [domino1]),
        }
        delay_distribution = DiscreteGaussianDelay(mu=torch.tensor(1.0),
                                                   sigma=torch.tensor(0.1))
        domino_fall_process = ExogenousProcess(
            "DominoFallFromBeingInFrontOfToppled", parameters,
            condition_at_start, condition_overall, set(), add_effects, set(),
            delay_distribution, torch.tensor(1.0))
        processes.add(domino_fall_process)

        # Individual Domino Fall from Toppled to Fallen
        domino1 = Variable("?d1", domino_type)
        parameters = [domino1]
        condition_at_start = {
            LiftedAtom(Toppled, [domino1]),
        }
        condition_overall = condition_at_start.copy()
        add_effects = set()
        delete_effects = {
            LiftedAtom(Toppled, [domino1]),
        }
        delay_distribution = DiscreteGaussianDelay(mu=torch.tensor(3.0),
                                                   sigma=torch.tensor(0.1))
        domino_toppled_delete_process = ExogenousProcess(
            "DominoToppledDelete", parameters, condition_at_start,
            condition_overall, set(), add_effects, delete_effects,
            delay_distribution, torch.tensor(1.0))
        processes.add(domino_toppled_delete_process)

        return processes
