"""Ground-truth NSRTs for the real-bench fan environment.

One operator, because the bench affords one action. Its sampler is the
interesting part: it must pick a burst duration that lands the ball in
the requested zone, which is the map an agent on the real bench has to
learn for itself.
"""

from typing import Dict, Sequence, Set

import numpy as np

from predicators.ground_truth_models import GroundTruthNSRTFactory
from predicators.settings import CFG
from predicators.structs import NSRT, Array, GroundAtom, LiftedAtom, Object, \
    ParameterizedOption, Predicate, State, Type, Variable


class PyBulletFanRealGroundTruthNSRTFactory(GroundTruthNSRTFactory):
    """Ground-truth NSRTs for the real-bench fan environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"pybullet_fan_real"}

    @staticmethod
    def get_nsrts(env_name: str, types: Dict[str, Type],
                  predicates: Dict[str, Predicate],
                  options: Dict[str, ParameterizedOption]) -> Set[NSRT]:
        del env_name  # unused

        robot_type = types["robot"]
        fan_type = types["fan"]
        ball_type = types["ball"]
        zone_type = types["zone"]

        FanOff = predicates["FanOff"]
        BallInZone = predicates["BallInZone"]

        BlowBallToZone = options["BlowBallToZone"]

        def _blow_sampler(state: State, goal: Set[GroundAtom],
                          rng: np.random.Generator,
                          objs: Sequence[Object]) -> Array:
            """Push geometry, plus the burst that reaches ``?zone``.

            The duration comes from the calibrated affine inverse of the
            bench's duration -> travel-distance map (see
            ``fan_real_oracle_burst_*``). This is ground truth ONLY
            because the map was measured offline; an agent learning on
            the bench has to recover these two numbers from rollouts,
            which is the point of the domain.
            """
            del goal
            # objs are the NSRT's parameters, not the option's vars.
            _, _, ball, zone = objs
            travel = state.get(zone, "x") - state.get(ball, "x")
            burst = (CFG.fan_real_oracle_burst_intercept +
                     CFG.fan_real_oracle_burst_slope * max(travel, 0.0))
            burst = float(
                np.clip(burst + rng.normal(0.0, CFG.fan_real_oracle_burst_std),
                        0.0, CFG.fan_real_max_burst_seconds))
            return np.array([burst], dtype=np.float64)

        robot = Variable("?robot", robot_type)
        fan = Variable("?fan", fan_type)
        ball = Variable("?ball", ball_type)
        zone = Variable("?zone", zone_type)

        blow_nsrt = NSRT(
            "BlowBallToZone",
            [robot, fan, ball, zone],
            {LiftedAtom(FanOff, [fan])},
            {LiftedAtom(BallInZone, [ball, zone])},
            set(),
            # The ball ends up in exactly one zone, so whatever zone it was
            # in before is no longer true. Naming the source zone as a
            # parameter instead would make the operator inapplicable in the
            # initial state, where the ball sits upwind of every zone.
            {BallInZone},
            BlowBallToZone,
            [robot, fan, zone],
            _blow_sampler)

        return {blow_nsrt}
