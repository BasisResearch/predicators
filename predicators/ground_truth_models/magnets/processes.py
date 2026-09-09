"""Ground-truth processes for the magnets environment.

Two moves per piece the wand pulls: ``Capture`` jumps the tip over the
piece, after which it sits under the tip; ``Carry`` hovers the tip to
the piece's slot, and the piece follows. The samplers aim at the piece
and at the slot, so the oracle's plan is the generator's own.
"""

from typing import Callable, Dict, Sequence, Set

import numpy as np
import torch

from predicators.ground_truth_models import GroundTruthProcessFactory
from predicators.structs import Array, CausalProcess, EndogenousProcess, \
    GroundAtom, LiftedAtom, Object, ParameterizedOption, Predicate, State, \
    Type, Variable
from predicators.utils import ConstantDelay, null_sampler

_CAPTURE_DELAY = 2
_CARRY_DELAY = 2


class PyBulletMagnetsGroundTruthProcessFactory(GroundTruthProcessFactory):
    """Ground-truth processes for the magnets environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"pybullet_magnets"}

    @staticmethod
    def get_processes(
            env_name: str, types: Dict[str, Type], predicates: Dict[str,
                                                                    Predicate],
            options: Dict[str, ParameterizedOption]) -> Set[CausalProcess]:
        del env_name  # unused

        robot_type = types["robot"]
        wand_type = types["wand"]
        piece_type = types["piece"]
        slot_type = types["slot"]

        In = predicates["In"]
        OnMat = predicates["OnMat"]
        AtRest = predicates["AtRest"]
        TipOver = predicates["TipOver"]
        Holding = predicates["Holding"]
        Pulled = predicates["Pulled"]

        Hover = options["Hover"]
        Jump = options["Jump"]
        Wait = options["Wait"]

        def _aim_at(index: int) -> Callable[..., Array]:

            def _sampler(state: State, goal: Set[GroundAtom],
                         rng: np.random.Generator,
                         objs: Sequence[Object]) -> Array:
                del goal, rng
                target = objs[index]
                return np.array(
                    [state.get(target, "x"),
                     state.get(target, "y")],
                    dtype=np.float32)

            return _sampler

        processes: Set[CausalProcess] = set()

        robot = Variable("?robot", robot_type)
        wand = Variable("?wand", wand_type)
        piece = Variable("?piece", piece_type)
        processes.add(
            EndogenousProcess(
                "Capture", [robot, wand, piece], {
                    LiftedAtom(Holding, [robot, wand]),
                    LiftedAtom(Pulled, [piece]),
                    LiftedAtom(OnMat, [piece]),
                    LiftedAtom(AtRest, [piece]),
                }, set(), set(), {LiftedAtom(TipOver, [wand, piece])}, set(),
                ConstantDelay(_CAPTURE_DELAY), torch.tensor(1.0), Jump,
                [robot, wand], _aim_at(2)))

        robot = Variable("?robot", robot_type)
        wand = Variable("?wand", wand_type)
        piece = Variable("?piece", piece_type)
        slot = Variable("?slot", slot_type)
        processes.add(
            EndogenousProcess(
                "Carry", [robot, wand, piece, slot], {
                    LiftedAtom(Holding, [robot, wand]),
                    LiftedAtom(Pulled, [piece]),
                    LiftedAtom(TipOver, [wand, piece]),
                }, set(), set(), {LiftedAtom(In, [piece, slot])}, set(),
                ConstantDelay(_CARRY_DELAY), torch.tensor(1.0), Hover,
                [robot, wand], _aim_at(3)))

        robot = Variable("?robot", robot_type)
        processes.add(
            EndogenousProcess("Wait", [robot], set(), set(),
                              set(), set(), set(), ConstantDelay(1),
                              torch.tensor(1.0), Wait, [robot], null_sampler))
        return processes
