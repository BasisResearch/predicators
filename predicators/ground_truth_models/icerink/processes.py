"""Ground-truth processes for the ice rink environment.

One push is one process: it lands the tile on a target its ``Reachable``
helper vouches for, a couple of ticks after the arm has retreated (the
tile is still sliding when the option ends). The sampler finds the speed
that does it on the push probe, so the oracle never plans a push it does
not know the end of.
"""

from typing import Dict, Sequence, Set

import numpy as np
import torch

from predicators.ground_truth_models import GroundTruthProcessFactory
from predicators.ground_truth_models.icerink.predicates import \
    along_direction, speed_grid, travel_along
from predicators.settings import CFG
from predicators.structs import Array, CausalProcess, EndogenousProcess, \
    GroundAtom, LiftedAtom, Object, ParameterizedOption, Predicate, State, \
    Type, Variable
from predicators.utils import ConstantDelay, null_sampler

# Ticks between the push option ending and the tile coming to rest.
_SLIDE_DELAY = 2
# Bisection steps between two grid speeds; a grid gap is under 0.1 m/s,
# so four halvings resolve it to under a centimetre per second.
_BISECTION_STEPS = 4


def speed_for_target(state: State, tile: Object, direction: Object,
                     target: Object) -> float:
    """The stroke speed whose push lands ``tile`` on ``target``.

    Probes the speed grid, takes the grid speed whose travel is nearest
    the target's distance, and bisects toward the neighbouring grid
    speed on the other side of it. A target no push overshoots (against
    a wall, say) gets the speed that travels farthest.
    """
    ahead = along_direction(state, tile, direction, target)
    grid = speed_grid()
    if ahead is None:
        return grid[-1]

    def travel(speed: float) -> float:
        t = travel_along(state, tile, direction, speed)
        return 0.0 if t is None else t

    travels = [travel(s) for s in grid]
    if all(t <= ahead for t in travels):
        return grid[int(np.argmax(travels))]
    best = int(np.argmin([abs(t - ahead) for t in travels]))
    if abs(travels[best] - ahead) <= CFG.icerink_on_tol / 2:
        return grid[best]
    if travels[best] < ahead:
        candidates = [j for j in range(len(grid)) if travels[j] > ahead]
    else:
        candidates = [j for j in range(len(grid)) if travels[j] < ahead]
    if not candidates:
        return grid[best]
    other = min(candidates, key=lambda j: abs(grid[j] - grid[best]))
    lo, hi = sorted((grid[best], grid[other]))
    lo_t, hi_t = travel(lo), travel(hi)
    for _ in range(_BISECTION_STEPS):
        mid = (lo + hi) / 2
        mid_t = travel(mid)
        if (mid_t < ahead) == (lo_t < ahead):
            lo, lo_t = mid, mid_t
        else:
            hi, hi_t = mid, mid_t
    return lo if abs(lo_t - ahead) <= abs(hi_t - ahead) else hi


class PyBulletIceRinkGroundTruthProcessFactory(GroundTruthProcessFactory):
    """Ground-truth processes for the ice rink environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"pybullet_icerink"}

    @staticmethod
    def get_processes(
            env_name: str, types: Dict[str, Type], predicates: Dict[str,
                                                                    Predicate],
            options: Dict[str, ParameterizedOption]) -> Set[CausalProcess]:
        del env_name  # unused

        robot_type = types["robot"]
        tile_type = types["tile"]
        target_type = types["target"]
        direction_type = types["direction"]

        On = predicates["On"]
        AtRest = predicates["AtRest"]
        OnRink = predicates["OnRink"]
        Reachable = predicates["Reachable"]

        Push = options["Push"]
        Wait = options["Wait"]

        def _push_sampler(state: State, goal: Set[GroundAtom],
                          rng: np.random.Generator,
                          objs: Sequence[Object]) -> Array:
            del goal, rng
            _, tile, direction, target = objs
            speed = speed_for_target(state, tile, direction, target)
            return np.array(
                [CFG.icerink_push_approach, CFG.icerink_push_contact_z, speed],
                dtype=np.float32)

        processes: Set[CausalProcess] = set()

        robot = Variable("?robot", robot_type)
        tile = Variable("?tile", tile_type)
        direction = Variable("?direction", direction_type)
        target = Variable("?target", target_type)
        processes.add(
            EndogenousProcess(
                "PushToTarget", [robot, tile, direction, target], {
                    LiftedAtom(AtRest, [tile]),
                    LiftedAtom(OnRink, [tile]),
                    LiftedAtom(Reachable, [tile, direction, target]),
                }, set(), set(), {LiftedAtom(On, [tile, target])}, set(),
                ConstantDelay(_SLIDE_DELAY), torch.tensor(1.0), Push,
                [robot, tile, direction], _push_sampler))

        robot = Variable("?robot", robot_type)
        processes.add(
            EndogenousProcess("Wait", [robot], set(), set(),
                              set(), set(), set(), ConstantDelay(1),
                              torch.tensor(1.0), Wait, [robot], null_sampler))

        return processes
