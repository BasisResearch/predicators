"""Ground-truth NSRTs for the Towers of Hanoi environment."""

from typing import Dict, Set

from predicators.ground_truth_models import GroundTruthNSRTFactory
from predicators.structs import NSRT, LiftedAtom, ParameterizedOption, \
    Predicate, Type, Variable
from predicators.utils import null_sampler


class HanoiGroundTruthNSRTFactory(GroundTruthNSRTFactory):
    """Ground-truth NSRTs for the Towers of Hanoi environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"hanoi"}

    @staticmethod
    def get_nsrts(env_name: str, types: Dict[str, Type],
                  predicates: Dict[str, Predicate],
                  options: Dict[str, ParameterizedOption]) -> Set[NSRT]:
        del env_name  # unused
        # Types
        disk_type = types["disk"]
        peg_type = types["peg"]
        robot_type = types["robot"]

        # Predicates
        On = predicates["On"]
        OnPeg = predicates["OnPeg"]
        Clear = predicates["Clear"]
        ClearPeg = predicates["ClearPeg"]
        Holding = predicates["Holding"]
        GripperOpen = predicates["GripperOpen"]
        Smaller = predicates["Smaller"]

        # Options
        Pick = options["Pick"]
        Stack = options["Stack"]
        PutOnPeg = options["PutOnPeg"]

        nsrts = set()

        # PickFromPeg: pick a disk that sits directly on a peg (and is clear).
        disk = Variable("?disk", disk_type)
        peg = Variable("?peg", peg_type)
        robot = Variable("?robot", robot_type)
        parameters = [disk, peg, robot]
        option_vars = [robot, disk]
        preconditions = {
            LiftedAtom(OnPeg, [disk, peg]),
            LiftedAtom(Clear, [disk]),
            LiftedAtom(GripperOpen, [robot]),
        }
        add_effects = {
            LiftedAtom(Holding, [disk]),
            LiftedAtom(ClearPeg, [peg]),
        }
        delete_effects = {
            LiftedAtom(OnPeg, [disk, peg]),
            LiftedAtom(Clear, [disk]),
            LiftedAtom(GripperOpen, [robot]),
        }
        pickfrompeg_nsrt = NSRT("PickFromPeg", parameters, preconditions,
                                add_effects, delete_effects, set(), Pick,
                                option_vars, null_sampler)
        nsrts.add(pickfrompeg_nsrt)

        # Unstack: pick a disk that sits on top of another disk.
        disk = Variable("?disk", disk_type)
        otherdisk = Variable("?otherdisk", disk_type)
        robot = Variable("?robot", robot_type)
        parameters = [disk, otherdisk, robot]
        option_vars = [robot, disk]
        preconditions = {
            LiftedAtom(On, [disk, otherdisk]),
            LiftedAtom(Clear, [disk]),
            LiftedAtom(GripperOpen, [robot]),
        }
        add_effects = {
            LiftedAtom(Holding, [disk]),
            LiftedAtom(Clear, [otherdisk]),
        }
        delete_effects = {
            LiftedAtom(On, [disk, otherdisk]),
            LiftedAtom(Clear, [disk]),
            LiftedAtom(GripperOpen, [robot]),
        }
        unstack_nsrt = NSRT("Unstack", parameters, preconditions, add_effects,
                            delete_effects, set(), Pick, option_vars,
                            null_sampler)
        nsrts.add(unstack_nsrt)

        # Stack: place the held disk onto a strictly larger, clear disk.
        disk = Variable("?disk", disk_type)
        otherdisk = Variable("?otherdisk", disk_type)
        robot = Variable("?robot", robot_type)
        parameters = [disk, otherdisk, robot]
        option_vars = [robot, otherdisk]
        preconditions = {
            LiftedAtom(Holding, [disk]),
            LiftedAtom(Clear, [otherdisk]),
            LiftedAtom(Smaller, [disk, otherdisk]),
        }
        add_effects = {
            LiftedAtom(On, [disk, otherdisk]),
            LiftedAtom(Clear, [disk]),
            LiftedAtom(GripperOpen, [robot]),
        }
        delete_effects = {
            LiftedAtom(Holding, [disk]),
            LiftedAtom(Clear, [otherdisk]),
        }
        stack_nsrt = NSRT("Stack", parameters, preconditions, add_effects,
                          delete_effects, set(), Stack, option_vars,
                          null_sampler)
        nsrts.add(stack_nsrt)

        # PutOnPeg: place the held disk onto an empty peg.
        disk = Variable("?disk", disk_type)
        peg = Variable("?peg", peg_type)
        robot = Variable("?robot", robot_type)
        parameters = [disk, peg, robot]
        option_vars = [robot, peg]
        preconditions = {
            LiftedAtom(Holding, [disk]),
            LiftedAtom(ClearPeg, [peg]),
        }
        add_effects = {
            LiftedAtom(OnPeg, [disk, peg]),
            LiftedAtom(Clear, [disk]),
            LiftedAtom(GripperOpen, [robot]),
        }
        delete_effects = {
            LiftedAtom(Holding, [disk]),
            LiftedAtom(ClearPeg, [peg]),
        }
        putonpeg_nsrt = NSRT("PutOnPeg", parameters, preconditions,
                             add_effects, delete_effects, set(), PutOnPeg,
                             option_vars, null_sampler)
        nsrts.add(putonpeg_nsrt)

        return nsrts
