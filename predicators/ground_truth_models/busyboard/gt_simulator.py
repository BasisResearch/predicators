"""Ground-truth simulator program for pybullet_busyboard residual dynamics.

The board's hidden process is one rule applied to every lamp: while the
lamp's drive condition holds - its driver button on, and its enabler
button on too if it has one - a hidden charge accumulates; otherwise the
charge bleeds away. The observable ``brightness`` is a flat-then-ramp
readout of that charge, so the early accumulation is invisible.

This module is the fully-observable answer key, where ``charge`` is a
visible feature and only the WIRING is unknown. Its sibling
``gt_simulator_po.py`` carries the charge in the recurrent ``latent``
block for the partially-observable setting.

**How the wiring enters the fit, and what is wrong with that.** The
fitting stack's only hypothesis vocabulary is ``ParamSpec``: a named
real scalar with bounds and a linear or log scale. A wiring is not that.
It is a discrete relation, and the honest posterior over it is a
distribution on a finite set, not a Gaussian on a box. To express the
wiring at all, this reference relaxes each button index to a bounded
real that the rule rounds - so gradient and Laplace machinery see a
staircase whose derivative is zero almost everywhere and undefined at
the steps, and the "uncertainty" a Laplace ensemble reports about a
wiring is meaningless.

That is deliberate, and it is the point of the domain rather than a
defect of this file. Every other environment in the suite hides a fixed
program structure behind a handful of continuous knobs, so a continuous
fitter is the right tool; here it is the wrong tool, and the gap is
measurable rather than hypothetical. Closing it properly means a
categorical parameter spec plus an ensemble that enumerates or samples
discrete assignments, sharing the interface that
``mean_bernoulli_entropy`` already consumes - it needs only ensemble
members that disagree about atoms, not members drawn from a Gaussian.

**One-step offset at button transitions.** The env applies its residual
after the base sim has stepped, so it reads the button states an action
*ends* with, while a teacher-forced prediction from this module is a
function of the state the action *starts* from. On the single step where
a push latches a button the two therefore disagree by one charge
increment, and agree exactly everywhere else (measured: 1 discrepancy in
135 steps of a two-press-then-wait rollout, at the latch step). This is
the same phase offset boil's reference notes for its burner warm-up, and
it is a property of the convention rather than of the rule.

**Per-run, not per-task.** ``PARAM_SPECS`` resolves once, after CFG is
final and before any task is chosen, so the true wiring this module
reports is the run's canonical one (``canonical_wiring``), the same
object the env installs when ``CFG.busyboard_fixed_wiring`` is on. With
that flag off the env rewires per task and this reference is no longer
an answer key - the per-task parameter scope needed to make it one does
not exist yet.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from predicators.code_sim_learning.fit_space import ParamSpec
from predicators.code_sim_learning.utils import Params, ResidualUpdate, \
    objs_by_type
from predicators.envs.pybullet_busyboard import NO_ENABLER, canonical_wiring, \
    project_wiring
from predicators.ground_truth_models import GroundTruthSimulatorFactory
from predicators.settings import CFG
from predicators.structs import State

# ── Constants ────────────────────────────────────────────────────

# The env's flat-then-ramp readout of the hidden charge. These are env
# constants, not learned parameters: brightness is 0 until the charge
# crosses the onset, then ramps linearly to 1.0 at charge == 1.0.
BRIGHTNESS_ONSET = 0.6
BRIGHTNESS_RAMP = 1.0 / (1.0 - BRIGHTNESS_ONSET)  # == 2.5


def _brightness(charge: float) -> float:
    """Observable projection of the hidden charge."""
    return float(
        np.clip((charge - BRIGHTNESS_ONSET) * BRIGHTNESS_RAMP, 0.0, 1.0))


def _button_states(observation: State) -> List[bool]:
    """Latched state of every button, in board (name) order.

    Board order is what the wiring indices refer to, and object names
    encode it (``button0``, ``button1``, ...), so this is the one place
    that ordering convention is applied.
    """
    buttons = sorted(objs_by_type(observation).get("button", []),
                     key=lambda o: o.name)
    return [observation.get(b, "is_on") > 0.5 for b in buttons]


def _wired_index(value: float, max_buttons: int) -> int:
    """Decode a relaxed real-valued button index into a button.

    Values at or below ``NO_ENABLER + 0.5`` mean "no button"; everything
    else rounds and clamps into range. The rounding is exactly the
    staircase the module docstring warns about.
    """
    if value <= NO_ENABLER + 0.5:
        return NO_ENABLER
    return int(np.clip(round(value), 0, max(max_buttons - 1, 0)))


def wiring_from_params(params: Params, num_lamps: int,
                       num_buttons: int) -> Tuple[List[int], List[int]]:
    """Decode the params' max-board wiring onto the observed board.

    The parameters describe the largest board the task distribution can
    produce; ``project_wiring`` maps that onto the board actually in
    front of the robot. Doing the projection here rather than baking a
    board size into the parameters is what makes one parameter vector a
    correct model of every board in a run.
    """
    max_buttons = max(
        list(CFG.busyboard_num_buttons_train) +
        list(CFG.busyboard_num_buttons_test))
    max_lamps = max(
        list(CFG.busyboard_num_lamps_train) +
        list(CFG.busyboard_num_lamps_test))
    driver_full = [
        _wired_index(params.get(f"driver_{i}", 0.0), max_buttons)
        for i in range(max_lamps)
    ]
    enabler_full = [
        _wired_index(params.get(f"enabler_{i}", NO_ENABLER), max_buttons)
        for i in range(max_lamps)
    ]
    return project_wiring(driver_full, enabler_full, num_buttons, num_lamps)


def _driven(button_on: List[bool], driver: int, enabler: int) -> bool:
    """Whether a lamp's conjunctive drive condition holds."""
    if not 0 <= driver < len(button_on) or not button_on[driver]:
        return False
    if enabler == NO_ENABLER:
        return True
    return 0 <= enabler < len(button_on) and button_on[enabler]


# ── Residual rules ────────────────────────────────────────────────


def _charging(observation: State, updates: ResidualUpdate,
              params: Params) -> ResidualUpdate:
    """Driven lamps charge, undriven lamps bleed; brightness follows.

    One rule covers the whole board: the per-lamp differences live
    entirely in the wiring parameters, not in the control flow. Both the
    charge and its brightness readout are written, because in the fully-
    observable setting both are features the fit is scored on.
    """
    objs = objs_by_type(observation)
    lamps = sorted(objs.get("lamp", []), key=lambda o: o.name)
    button_on = _button_states(observation)
    driver, enabler = wiring_from_params(params, len(lamps), len(button_on))

    for i, lamp in enumerate(lamps):
        charge = float(observation.get(lamp, "charge"))
        if _driven(button_on, driver[i], enabler[i]):
            charge = min(1.0, charge + params["charge_rate"])
        else:
            charge = max(0.0, charge - params["decay_rate"])
        updates.setdefault(lamp, {})["charge"] = charge
        updates[lamp]["brightness"] = _brightness(charge)

    return updates


# ── Param specs ──────────────────────────────────────────────────


def _build_param_specs() -> List[ParamSpec]:
    """Build at call time so CFG-driven values match the current run.

    The rate pair is a genuine continuous fit: both are identifiable
    from the brightness ramp and the fade. The wiring entries are the
    relaxed discrete parameters - bounded reals the rule rounds - and
    are initialized at the run's true wiring so this module serves as
    the oracle-model baseline.
    """
    max_buttons = max(
        list(CFG.busyboard_num_buttons_train) +
        list(CFG.busyboard_num_buttons_test))
    max_lamps = max(
        list(CFG.busyboard_num_lamps_train) +
        list(CFG.busyboard_num_lamps_test))
    driver, enabler = canonical_wiring(max_buttons, max_lamps)

    specs = [
        ParamSpec("charge_rate", CFG.busyboard_charge_rate, lo=0.0, hi=1.0),
        ParamSpec("decay_rate", CFG.busyboard_decay_rate, lo=0.0, hi=1.0),
    ]
    for i in range(max_lamps):
        specs.append(
            ParamSpec(f"driver_{i}",
                      float(driver[i]),
                      lo=0.0,
                      hi=float(max_buttons - 1)))
        specs.append(
            ParamSpec(f"enabler_{i}",
                      float(enabler[i]),
                      lo=float(NO_ENABLER),
                      hi=float(max_buttons - 1)))
    return specs


# ── Public API: consumed by read_simulator_components ────────────
# Same contract used by agent-synthesized simulator files. PARAM_SPECS is
# bound to the callable so CFG-dependent defaults resolve when the loader
# pulls the value, after CFG is final.

RESIDUAL_RULES = [_charging]

PARAM_SPECS = _build_param_specs

RESIDUAL_FEATURES: Dict[str, List[str]] = {
    "lamp": ["charge", "brightness"],
}

# ── Factory binding ──────────────────────────────────────────────


class PyBulletBusyBoardGroundTruthSimulatorFactory(GroundTruthSimulatorFactory
                                                   ):
    """GT residual-dynamics simulator for pybullet_busyboard.

    Claims the env only in fully-observable mode; ``gt_simulator_po.py``
    claims it otherwise, so ``get_gt_simulator``'s env-name dispatch
    resolves to exactly one module per run.
    """

    @classmethod
    def get_env_names(cls) -> set:
        if CFG.partially_observable:
            return set()
        return {"pybullet_busyboard"}
