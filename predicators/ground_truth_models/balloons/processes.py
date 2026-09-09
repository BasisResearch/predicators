"""Ground-truth processes for the balloons environment.

``ReleaseClip`` is the robot's move: it opens a clip, and the freed
balloon is on the box a tick later. ``Rise`` is the board's: once every
needed balloon is freed, the box floats up and hangs in the band a few
ticks later.
"""

from typing import Dict, Sequence, Set

import numpy as np
import torch

from predicators.ground_truth_models import GroundTruthProcessFactory
from predicators.ground_truth_models.balloons.options import release_params
from predicators.structs import Array, CausalProcess, EndogenousProcess, \
    ExogenousProcess, GroundAtom, LiftedAtom, Object, ParameterizedOption, \
    Predicate, State, Type, Variable
from predicators.utils import ConstantDelay, null_sampler

_RELEASE_DELAY = 2
_RISE_DELAY = 8


class PyBulletBalloonsGroundTruthProcessFactory(GroundTruthProcessFactory):
    """Ground-truth processes for the balloons environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"pybullet_balloons"}

    @staticmethod
    def get_processes(
            env_name: str, types: Dict[str, Type], predicates: Dict[str,
                                                                    Predicate],
            options: Dict[str, ParameterizedOption]) -> Set[CausalProcess]:
        del env_name  # unused

        robot_type = types["robot"]
        box_type = types["box"]
        balloon_type = types["balloon"]
        clip_type = types["clip"]
        band_type = types["band"]

        InBand = predicates["InBand"]
        Tied = predicates["Tied"]
        Untied = predicates["Untied"]
        Intact = predicates["Intact"]
        ClipOn = predicates["ClipOn"]
        ClipOff = predicates["ClipOff"]
        Needed = predicates["Needed"]
        Holds = predicates["Holds"]
        AllNeededTied = predicates["AllNeededTied"]

        Release = options["Release"]
        Wait = options["Wait"]

        def _release_sampler(state: State, goal: Set[GroundAtom],
                             rng: np.random.Generator,
                             objs: Sequence[Object]) -> Array:
            del state, goal, rng, objs
            return release_params()

        processes: Set[CausalProcess] = set()

        robot = Variable("?robot", robot_type)
        clip = Variable("?clip", clip_type)
        balloon = Variable("?balloon", balloon_type)
        band = Variable("?band", band_type)
        processes.add(
            EndogenousProcess(
                "ReleaseClip", [robot, clip, balloon, band], {
                    LiftedAtom(ClipOff, [clip]),
                    LiftedAtom(Holds, [clip, balloon]),
                    LiftedAtom(Untied, [balloon]),
                    LiftedAtom(Intact, [balloon]),
                    LiftedAtom(Needed, [balloon, band]),
                }, set(), set(),
                {LiftedAtom(ClipOn, [clip]),
                 LiftedAtom(Tied, [balloon])},
                {LiftedAtom(ClipOff, [clip]),
                 LiftedAtom(Untied, [balloon])}, ConstantDelay(_RELEASE_DELAY),
                torch.tensor(1.0), Release, [robot, clip], _release_sampler))

        box = Variable("?box", box_type)
        band = Variable("?band", band_type)
        processes.add(
            ExogenousProcess("Rise",
                             [box, band], {LiftedAtom(AllNeededTied, [band])},
                             set(), set(), {LiftedAtom(InBand, [box, band])},
                             set(), ConstantDelay(_RISE_DELAY),
                             torch.tensor(1.0)))

        robot = Variable("?robot", robot_type)
        processes.add(
            EndogenousProcess("Wait", [robot], set(), set(),
                              set(), set(), set(), ConstantDelay(1),
                              torch.tensor(1.0), Wait, [robot], null_sampler))
        return processes
