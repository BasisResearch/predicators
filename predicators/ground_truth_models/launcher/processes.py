"""Ground-truth processes for the launcher environment.

One shot is one process: it topples the block its ``Hittable`` helper
vouches for, a few ticks after the arm has retreated (the ball is in
flight when the option ends). The sampler takes the middle of the
working compression window from the launch probe.
"""

from typing import Dict, Sequence, Set

import numpy as np
import torch

from predicators.envs.pybullet_launcher import PyBulletLauncherEnv
from predicators.ground_truth_models import GroundTruthProcessFactory
from predicators.ground_truth_models.launcher.predicates import working_depths
from predicators.settings import CFG
from predicators.structs import Array, CausalProcess, EndogenousProcess, \
    GroundAtom, LiftedAtom, Object, ParameterizedOption, Predicate, State, \
    Type, Variable
from predicators.utils import ConstantDelay, null_sampler

# Ticks between the cock option ending and the tower settling.
_FLIGHT_DELAY = 4


def depth_for_shot(state: State) -> float:
    """The middle of the longest run of working compressions."""
    depths = working_depths(state)
    if not depths:
        return float(CFG.launcher_min_compression)
    runs = [[depths[0]]]
    for d in depths[1:]:
        if abs(d - runs[-1][-1] - PyBulletLauncherEnv.scan_step) < 1e-6:
            runs[-1].append(d)
        else:
            runs.append([d])
    best = max(runs, key=len)
    return float(best[len(best) // 2])


class PyBulletLauncherGroundTruthProcessFactory(GroundTruthProcessFactory):
    """Ground-truth processes for the launcher environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"pybullet_launcher"}

    @staticmethod
    def get_processes(
            env_name: str, types: Dict[str, Type], predicates: Dict[str,
                                                                    Predicate],
            options: Dict[str, ParameterizedOption]) -> Set[CausalProcess]:
        del env_name  # unused

        robot_type = types["robot"]
        launcher_type = types["launcher"]
        block_type = types["block"]

        Toppled = predicates["Toppled"]
        Standing = predicates["Standing"]
        Loaded = predicates["Loaded"]
        Hittable = predicates["Hittable"]

        Cock = options["Cock"]
        Wait = options["Wait"]

        def _cock_sampler(state: State, goal: Set[GroundAtom],
                          rng: np.random.Generator,
                          objs: Sequence[Object]) -> Array:
            del goal, rng, objs
            return np.array([
                CFG.launcher_push_approach, CFG.launcher_push_contact_z,
                depth_for_shot(state)
            ],
                            dtype=np.float32)

        processes: Set[CausalProcess] = set()

        robot = Variable("?robot", robot_type)
        launcher = Variable("?launcher", launcher_type)
        block = Variable("?block", block_type)
        processes.add(
            EndogenousProcess(
                "Fire", [robot, launcher, block], {
                    LiftedAtom(Loaded, [launcher]),
                    LiftedAtom(Standing, [block]),
                    LiftedAtom(Hittable, [launcher, block]),
                }, set(), set(), {LiftedAtom(Toppled, [block])},
                {LiftedAtom(Standing, [block])}, ConstantDelay(_FLIGHT_DELAY),
                torch.tensor(1.0), Cock, [robot, launcher], _cock_sampler))

        robot = Variable("?robot", robot_type)
        processes.add(
            EndogenousProcess("Wait", [robot], set(), set(),
                              set(), set(), set(), ConstantDelay(1),
                              torch.tensor(1.0), Wait, [robot], null_sampler))

        return processes
