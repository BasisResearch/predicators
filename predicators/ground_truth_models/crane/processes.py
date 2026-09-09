"""Ground-truth processes for the crane environment.

One swing is one process: it lands the crate its ``Hittable`` helper
vouches for on the bin, some ticks after the arm has retreated (the ram
is swinging when the option ends). The sampler takes the middle of the
working pull window from the swing probe.
"""

from typing import Dict, Sequence, Set

import numpy as np
import torch

from predicators.envs.pybullet_crane import PyBulletCraneEnv
from predicators.ground_truth_models import GroundTruthProcessFactory
from predicators.ground_truth_models.crane.predicates import working_pulls
from predicators.settings import CFG
from predicators.structs import Array, CausalProcess, EndogenousProcess, \
    GroundAtom, LiftedAtom, Object, ParameterizedOption, Predicate, State, \
    Type, Variable
from predicators.utils import ConstantDelay, null_sampler

# Ticks between the pull option ending and the crate settling.
_SWING_DELAY = 15


def pull_for_swing(state: State) -> float:
    """The middle of the longest run of working pulls."""
    pull = PyBulletCraneEnv.window_middle(working_pulls(state))
    if pull is None:
        pulls = working_pulls(state)
        return float(pulls[len(pulls) //
                           2]) if pulls else float(CFG.crane_pull_range[0])
    return pull


class PyBulletCraneGroundTruthProcessFactory(GroundTruthProcessFactory):
    """Ground-truth processes for the crane environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"pybullet_crane"}

    @staticmethod
    def get_processes(
            env_name: str, types: Dict[str, Type], predicates: Dict[str,
                                                                    Predicate],
            options: Dict[str, ParameterizedOption]) -> Set[CausalProcess]:
        del env_name  # unused

        robot_type = types["robot"]
        ram_type = types["ram"]
        crane_type = types["crane"]
        crate_type = types["crate"]
        bin_type = types["bin"]

        InBin = predicates["InBin"]
        OnTable = predicates["OnTable"]
        Reachable = predicates["Reachable"]
        RamStill = predicates["RamStill"]
        Hittable = predicates["Hittable"]

        Pull = options["Pull"]
        Wait = options["Wait"]

        def _pull_sampler(state: State, goal: Set[GroundAtom],
                          rng: np.random.Generator,
                          objs: Sequence[Object]) -> Array:
            del goal, rng, objs
            return np.array([
                CFG.crane_push_approach, CFG.crane_push_contact_z,
                pull_for_swing(state)
            ],
                            dtype=np.float32)

        processes: Set[CausalProcess] = set()

        robot = Variable("?robot", robot_type)
        ram = Variable("?ram", ram_type)
        crane = Variable("?crane", crane_type)
        crate = Variable("?crate", crate_type)
        bin_ = Variable("?bin", bin_type)
        processes.add(
            EndogenousProcess(
                "Swing", [robot, ram, crane, crate, bin_], {
                    LiftedAtom(RamStill, [ram]),
                    LiftedAtom(OnTable, [crate]),
                    LiftedAtom(Reachable, [ram, crate]),
                    LiftedAtom(Hittable, [ram, crate, bin_]),
                }, set(), set(), {LiftedAtom(InBin, [crate, bin_])}, set(),
                ConstantDelay(_SWING_DELAY), torch.tensor(1.0), Pull,
                [robot, ram, crane], _pull_sampler))

        robot = Variable("?robot", robot_type)
        processes.add(
            EndogenousProcess("Wait", [robot], set(), set(),
                              set(), set(), set(), ConstantDelay(1),
                              torch.tensor(1.0), Wait, [robot], null_sampler))

        return processes
