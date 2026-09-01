"""Ground-truth processes for the busyboard environment.

The busyboard is the one domain in the suite whose symbolic model does
not fit a STRIPS operator. Pressing a button has no fixed effect: it
lights a lamp only if that lamp's other button is already on, and it may
light lamps the goal wants dark. That is a conditional effect, which an
NSRT cannot express, so a planner built on ``PressButton`` operators
alone would happily plan "press everything" and fail on execution.

Processes express it directly. A lamp lighting is an EXOGENOUS process
whose ``condition_overall`` is the whole drive condition - both buttons
on, held for the delay. The planner then reasons about the conjunction
and about the wait, and never proposes a button assignment that would
light an off-target lamp, because that lamp's own lighting process would
fire too.

Two lighting processes rather than one, because a drive condition is
either a single button or a conjunction of two and a lifted process
cannot branch on which. The wiring itself arrives through the
``SoleDriver`` / ``JointDrivers`` helper predicates (see
``predicates.py``), which only oracle approaches receive.

Darkening is modelled too, even though an optimal plan never needs it:
every board starts dark, so a plan reaches its goal by lighting the
right lamps and never driving the rest. Without the darkening processes
the model would claim a lit lamp can never be turned off, which is false
and would mislead any replanning after an execution slip.
"""

from typing import Dict, Sequence, Set

import numpy as np
import torch

from predicators.ground_truth_models import GroundTruthProcessFactory
from predicators.structs import Array, CausalProcess, DelayDistribution, \
    EndogenousProcess, ExogenousProcess, GroundAtom, LiftedAtom, Object, \
    ParameterizedOption, Predicate, State, Type, Variable
from predicators.utils import ConstantDelay, DiscreteGaussianDelay, \
    null_sampler

# Symbolic delays, in process ticks. A lamp needs its drive condition
# held for a stretch before it lights, and dies faster than it lights -
# the same asymmetry the env's charge and decay rates carry.
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
                  rng: np.random.Generator,
                  objs: Sequence[Object]) -> Array:
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

        ButtonOn = predicates["ButtonOn"]
        ButtonOff = predicates["ButtonOff"]
        LampOn = predicates["LampOn"]
        LampOff = predicates["LampOff"]
        SoleDriver = predicates["SoleDriver"]
        JointDrivers = predicates["JointDrivers"]

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
                              {LiftedAtom(ButtonOn, [button])},
                              {LiftedAtom(ButtonOff, [button])},
                              _delay(_PUSH_DELAY_MU), torch.tensor(1.0),
                              PressButton, [robot, button],
                              _push_sampler))

        robot = Variable("?robot", robot_type)
        button = Variable("?button", button_type)
        processes.add(
            EndogenousProcess("ReleaseButton", [robot, button],
                              {LiftedAtom(ButtonOn, [button])}, set(), set(),
                              {LiftedAtom(ButtonOff, [button])},
                              {LiftedAtom(ButtonOn, [button])},
                              _delay(_PUSH_DELAY_MU), torch.tensor(1.0),
                              ReleaseButton, [robot, button],
                              _push_sampler))

        # Holding still is a first-class action here: a lamp only lights
        # while the board is left alone in a driving configuration.
        robot = Variable("?robot", robot_type)
        processes.add(
            EndogenousProcess("Wait", [robot], set(), set(), set(), set(),
                              set(), ConstantDelay(1), torch.tensor(1.0),
                              Wait, [robot], null_sampler))

        # ── Exogenous: what the board does ───────────────────────

        # A single-button lamp.
        button = Variable("?button", button_type)
        lamp = Variable("?lamp", lamp_type)
        drive_sole = {
            LiftedAtom(SoleDriver, [button, lamp]),
            LiftedAtom(ButtonOn, [button]),
        }
        processes.add(
            ExogenousProcess("LightLampSole", [button, lamp], drive_sole,
                             drive_sole.copy(), set(),
                             {LiftedAtom(LampOn, [lamp])},
                             {LiftedAtom(LampOff, [lamp])},
                             _delay(_LIGHT_DELAY_MU), torch.tensor(1.0)))

        # The interlock: a lamp that needs both of its buttons. The
        # conjunction sits in condition_overall, so the planner knows the
        # lamp lights only while BOTH are held - and equally, that any
        # assignment turning both on will light it whether or not that was
        # wanted.
        button_a = Variable("?button_a", button_type)
        button_b = Variable("?button_b", button_type)
        lamp = Variable("?lamp", lamp_type)
        drive_joint = {
            LiftedAtom(JointDrivers, [button_a, button_b, lamp]),
            LiftedAtom(ButtonOn, [button_a]),
            LiftedAtom(ButtonOn, [button_b]),
        }
        processes.add(
            ExogenousProcess("LightLampJoint", [button_a, button_b, lamp],
                             drive_joint, drive_joint.copy(), set(),
                             {LiftedAtom(LampOn, [lamp])},
                             {LiftedAtom(LampOff, [lamp])},
                             _delay(_LIGHT_DELAY_MU), torch.tensor(1.0)))

        # Losing the drive puts a lamp out again.
        button = Variable("?button", button_type)
        lamp = Variable("?lamp", lamp_type)
        undrive_sole = {
            LiftedAtom(SoleDriver, [button, lamp]),
            LiftedAtom(ButtonOff, [button]),
            LiftedAtom(LampOn, [lamp]),
        }
        processes.add(
            ExogenousProcess("DarkenLampSole", [button, lamp], undrive_sole,
                             undrive_sole.copy(), set(),
                             {LiftedAtom(LampOff, [lamp])},
                             {LiftedAtom(LampOn, [lamp])},
                             _delay(_DARKEN_DELAY_MU), torch.tensor(1.0)))

        # For a conjunctive lamp, either button going off is enough to put
        # it out. Disjunction is not expressible in a condition set, and
        # JointDrivers is emitted in one canonical order only, so this is
        # two processes - one keyed on each button - rather than one.
        for suffix, off_var_idx in (("First", 0), ("Second", 1)):
            button_a = Variable("?button_a", button_type)
            button_b = Variable("?button_b", button_type)
            lamp = Variable("?lamp", lamp_type)
            off_button = (button_a, button_b)[off_var_idx]
            undrive_joint = {
                LiftedAtom(JointDrivers, [button_a, button_b, lamp]),
                LiftedAtom(ButtonOff, [off_button]),
                LiftedAtom(LampOn, [lamp]),
            }
            processes.add(
                ExogenousProcess(f"DarkenLampJoint{suffix}",
                                 [button_a, button_b, lamp], undrive_joint,
                                 undrive_joint.copy(), set(),
                                 {LiftedAtom(LampOff, [lamp])},
                                 {LiftedAtom(LampOn, [lamp])},
                                 _delay(_DARKEN_DELAY_MU), torch.tensor(1.0)))

        return processes
