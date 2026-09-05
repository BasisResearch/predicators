"""Ground-truth processes for the busyboard environment.

The busyboard is the one domain in the suite whose symbolic model does
not fit a STRIPS operator. Pressing a button has no fixed effect: it
arms a lamp only if that lamp's enablers are already on, it lights
nothing by itself, and it may light lamps the goal wants dark or trip
the breaker. Those are conditional, delayed effects, which an NSRT
cannot express, so a planner built on ``PressButton`` operators alone
would happily plan "press everything" and fail on execution.

Processes express it directly, with the wiring arriving through the
helper predicates of ``predicates.py`` (``Drives``, ``Enables``,
``Inhibits``), which only oracle approaches receive, and the derived
predicates built on them (``Enabled``, ``Driven``, ``Overloaded``, ...)
so that a lifted process can say "all of this lamp's enablers are on"
without knowing how many it has.

* ``PressButton`` turns a button on and emits a one-tick
  ``JustPressed`` pulse, which ``ClearPressed`` removes a tick later.
* ``ArmLamp`` fires on that pulse when the pressed button is the
  lamp's driver and the lamp is ``Enabled`` - so it fires only when the
  driver is pressed AFTER the enablers, which is the env's latch. With
  the latch ablated (``busyboard_latch`` off) it fires whenever the
  driver is on and the lamp enabled. ``DisarmLamp`` fires when the
  driver goes off.
* ``LightLamp`` is the charge: a lamp that is ``Driven`` (armed,
  driver and enablers on, inhibitor off, breaker closed) for the delay
  lights; ``DarkenLamp`` puts a lit lamp out once it is ``Undriven``.
* ``TripBreaker`` fires when the board is ``Overloaded`` (more lamps
  driven than the breaker allows), after which nothing is ``Driven``
  and every lit lamp darkens; ``ResetBreaker`` closes it again once
  ``AllButtonsOff``.

Darkening is modelled even though an optimal plan never needs it:
every board starts dark, so a plan reaches its goal by lighting the
right lamps and never driving the rest. Without it the model would
claim a lit lamp can never be turned off, which is false and would
mislead any replanning after an execution slip.
"""

from typing import Dict, Sequence, Set

import numpy as np
import torch

from predicators.ground_truth_models import GroundTruthProcessFactory
from predicators.settings import CFG
from predicators.structs import Array, CausalProcess, DelayDistribution, \
    EndogenousProcess, ExogenousProcess, GroundAtom, LiftedAtom, Object, \
    ParameterizedOption, Predicate, State, Type, Variable
from predicators.utils import ConstantDelay, DiscreteGaussianDelay, \
    null_sampler

# Symbolic delays, in process ticks. A lamp needs its drive held for a
# stretch before it lights, and dies faster than it lights - the same
# asymmetry the env's charge and decay rates carry. Latch and breaker
# events land on the next tick.
_LIGHT_DELAY_MU = 2.0
_DARKEN_DELAY_MU = 1.0
_DELAY_SIGMA = 0.1
# Button pushes are near-deterministic in duration, so their delay is
# only nominally stochastic.
_PUSH_DELAY_MU = 1.0

# Push-skill parameters (approach distance, contact height above the
# button). Drawn from the band measured to actually cross the slider's
# travel on this board: outside roughly ad in [0.04, 0.10] and cz in
# [0.02, 0.08] the stroke either starts inside the gripper's own
# footprint or passes over the slider entirely. Sampling the option's
# full declared space blind succeeds only ~48% of the time, which would
# make refinement backtrack constantly for reasons that have nothing to
# do with the board; this narrower draw keeps a little variety for
# backtracking while landing in the working band every time.
_PUSH_APPROACH_RANGE = (0.055, 0.085)
_PUSH_CONTACT_Z_RANGE = (0.035, 0.065)


def _push_sampler(state: State, goal: Set[GroundAtom],
                  rng: np.random.Generator, objs: Sequence[Object]) -> Array:
    """Push parameters inside the measured working band."""
    del state, goal, objs
    return np.array([
        rng.uniform(*_PUSH_APPROACH_RANGE),
        rng.uniform(*_PUSH_CONTACT_Z_RANGE),
    ],
                    dtype=np.float32)


def _delay(mu: float) -> DelayDistribution:
    """A tight Gaussian delay of ``mu`` ticks."""
    return DiscreteGaussianDelay(mu=torch.tensor(mu),
                                 sigma=torch.tensor(_DELAY_SIGMA))


class PyBulletBusyBoardGroundTruthProcessFactory(GroundTruthProcessFactory):
    """Ground-truth processes for the busyboard environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"pybullet_busyboard"}

    @staticmethod
    def get_processes(
            env_name: str, types: Dict[str, Type], predicates: Dict[str,
                                                                    Predicate],
            options: Dict[str, ParameterizedOption]) -> Set[CausalProcess]:
        del env_name  # unused

        robot_type = types["robot"]
        button_type = types["button"]
        lamp_type = types["lamp"]
        breaker_type = types["breaker"]

        ButtonOn = predicates["ButtonOn"]
        ButtonOff = predicates["ButtonOff"]
        LampOn = predicates["LampOn"]
        LampOff = predicates["LampOff"]
        BreakerTripped = predicates["BreakerTripped"]
        BreakerClosed = predicates["BreakerClosed"]
        Drives = predicates["Drives"]
        Armed = predicates["Armed"]
        Disarmed = predicates["Disarmed"]
        JustPressed = predicates["JustPressed"]
        Enabled = predicates["Enabled"]
        Driven = predicates["Driven"]
        Undriven = predicates["Undriven"]
        Overloaded = predicates["Overloaded"]
        AllButtonsOff = predicates["AllButtonsOff"]

        PressButton = options["PressButton"]
        ReleaseButton = options["ReleaseButton"]
        Wait = options["Wait"]

        processes: Set[CausalProcess] = set()

        # ── Endogenous: what the robot does ──────────────────────

        robot = Variable("?robot", robot_type)
        button = Variable("?button", button_type)
        processes.add(
            EndogenousProcess("PressButton", [robot, button],
                              {LiftedAtom(ButtonOff, [button])}, set(), set(),
                              {
                                  LiftedAtom(ButtonOn, [button]),
                                  LiftedAtom(JustPressed, [button])
                              }, {LiftedAtom(ButtonOff, [button])},
                              _delay(_PUSH_DELAY_MU), torch.tensor(1.0),
                              PressButton, [robot, button], _push_sampler))

        robot = Variable("?robot", robot_type)
        button = Variable("?button", button_type)
        processes.add(
            EndogenousProcess("ReleaseButton", [robot, button],
                              {LiftedAtom(ButtonOn, [button])}, set(), set(),
                              {LiftedAtom(ButtonOff, [button])},
                              {LiftedAtom(ButtonOn, [button])},
                              _delay(_PUSH_DELAY_MU), torch.tensor(1.0),
                              ReleaseButton, [robot, button], _push_sampler))

        # Holding still is a first-class action here: a lamp only lights
        # while the board is left alone in a driving configuration.
        robot = Variable("?robot", robot_type)
        processes.add(
            EndogenousProcess("Wait", [robot], set(), set(),
                              set(), set(), set(), ConstantDelay(1),
                              torch.tensor(1.0), Wait, [robot], null_sampler))

        # ── Exogenous: what the board does ───────────────────────

        # The press pulse lasts one tick.
        button = Variable("?button", button_type)
        processes.add(
            ExogenousProcess("ClearPressed", [button],
                             {LiftedAtom(JustPressed, [button])}, set(), set(),
                             set(), {LiftedAtom(JustPressed, [button])},
                             ConstantDelay(1), torch.tensor(1.0)))

        # The latch: a lamp arms when its driver is pressed while its
        # enablers are already on. Keyed to the pulse, so pressing the
        # enablers afterwards does not arm it; ablated, keyed to the
        # driver simply being on.
        button = Variable("?button", button_type)
        lamp = Variable("?lamp", lamp_type)
        arm_condition = {
            LiftedAtom(Drives, [button, lamp]),
            LiftedAtom(JustPressed if CFG.busyboard_latch else ButtonOn,
                       [button]),
            LiftedAtom(Enabled, [lamp]),
        }
        processes.add(
            ExogenousProcess("ArmLamp", [button, lamp], arm_condition, set(),
                             set(), {LiftedAtom(Armed, [lamp])},
                             {LiftedAtom(Disarmed, [lamp])}, ConstantDelay(1),
                             torch.tensor(1.0)))

        # Releasing the driver disarms the lamp.
        button = Variable("?button", button_type)
        lamp = Variable("?lamp", lamp_type)
        disarm_condition = {
            LiftedAtom(Drives, [button, lamp]),
            LiftedAtom(ButtonOff, [button]),
            LiftedAtom(Armed, [lamp]),
        }
        processes.add(
            ExogenousProcess("DisarmLamp", [button, lamp], disarm_condition,
                             set(), set(), {LiftedAtom(Disarmed, [lamp])},
                             {LiftedAtom(Armed, [lamp])}, ConstantDelay(1),
                             torch.tensor(1.0)))

        # The charge: a driven lamp lights after the delay, and only if
        # it stayed driven throughout - so the planner knows the lamp
        # lights only while its whole condition is held, and equally that
        # any sequence driving it will light it whether or not that was
        # wanted.
        lamp = Variable("?lamp", lamp_type)
        drive = {LiftedAtom(Driven, [lamp])}
        processes.add(
            ExogenousProcess("LightLamp",
                             [lamp], drive | {LiftedAtom(LampOff, [lamp])},
                             drive.copy(), set(), {LiftedAtom(LampOn, [lamp])},
                             {LiftedAtom(LampOff, [lamp])},
                             _delay(_LIGHT_DELAY_MU), torch.tensor(1.0)))

        # Losing the drive (a button released, the inhibitor pressed, the
        # breaker tripped) puts a lamp out again.
        lamp = Variable("?lamp", lamp_type)
        undrive = {LiftedAtom(Undriven, [lamp]), LiftedAtom(LampOn, [lamp])}
        processes.add(
            ExogenousProcess("DarkenLamp",
                             [lamp], undrive, {LiftedAtom(Undriven, [lamp])},
                             set(), {LiftedAtom(LampOff, [lamp])},
                             {LiftedAtom(LampOn, [lamp])},
                             _delay(_DARKEN_DELAY_MU), torch.tensor(1.0)))

        # The breaker: overload trips it, a power cycle closes it.
        breaker = Variable("?breaker", breaker_type)
        processes.add(
            ExogenousProcess("TripBreaker", [breaker], {
                LiftedAtom(Overloaded, []),
                LiftedAtom(BreakerClosed, [breaker])
            }, set(), set(), {LiftedAtom(BreakerTripped, [breaker])},
                             {LiftedAtom(BreakerClosed, [breaker])},
                             ConstantDelay(1), torch.tensor(1.0)))

        breaker = Variable("?breaker", breaker_type)
        processes.add(
            ExogenousProcess(
                "ResetBreaker", [breaker], {
                    LiftedAtom(AllButtonsOff, []),
                    LiftedAtom(BreakerTripped, [breaker])
                }, set(), set(), {LiftedAtom(BreakerClosed, [breaker])},
                {LiftedAtom(BreakerTripped, [breaker])}, ConstantDelay(1),
                torch.tensor(1.0)))

        return processes
