"""Ground-truth simulator program for pybullet_busyboard residual dynamics.

The board's hidden process is one rule applied to every lamp. A lamp is
*armed* on the rising edge of its driver button if every enabler is on
at that moment, and disarmed when the driver goes off. While it is
armed, its driver and enablers are on, its inhibitor (if any) is off
and the breaker is closed, a hidden charge accumulates; otherwise the
charge bleeds away. Driving more lamps at once than the breaker allows
trips it: every charge drops to zero and nothing charges until every
button is released. The observable ``brightness`` is a flat-then-ramp
readout of the charge, so the early accumulation is invisible.

This module is the fully-observable answer key, where ``charge`` and
``armed`` are visible lamp features and only the WIRING is unknown. Its
sibling ``gt_simulator_po.py`` carries both in the recurrent ``latent``
block for the partially-observable setting. Both use the recurrent
5-arg rule signature: the latch is edge-triggered, so a rule needs the
previous observation (``history[-1]``) to see the driver rise.

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
increment (and see the driver's edge one step apart), and agree exactly
everywhere else. This is the same phase offset boil's reference notes
for its burner warm-up, and it is a property of the convention rather
than of the rule.

**Per-run, not per-task.** ``PARAM_SPECS`` resolves once, after CFG is
final and before any task is chosen, so the true wiring this module
reports is the run's canonical one (``canonical_wiring``), the same
object the env installs when ``CFG.busyboard_fixed_wiring`` is on. With
that flag off the env rewires per task and this reference is no longer
an answer key - the per-task parameter scope needed to make it one does
not exist yet.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from predicators.code_sim_learning.fit_space import ParamSpec
from predicators.code_sim_learning.utils import History, Params, \
    ResidualUpdate, objs_by_type
from predicators.envs.pybullet_busyboard import NO_BUTTON, Condition, \
    core_board, full_wiring, project_wiring
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


def _previous_button_states(history: History,
                            fallback: List[bool]) -> List[bool]:
    """The button states one step earlier, for the driver's edge."""
    if history:
        prev = _button_states(history[-1][0])
        if len(prev) == len(fallback):
            return prev
    return list(fallback)


def _breaker_tripped(observation: State) -> Optional[bool]:
    """The breaker tile's state, when the observation shows one."""
    breakers = objs_by_type(observation).get("breaker", [])
    if not breakers:
        return None
    return bool(observation.get(breakers[0], "tripped") > 0.5)


def _wired_index(value: float, max_buttons: int) -> int:
    """Decode a relaxed real-valued button index into a button.

    Values at or below ``NO_BUTTON + 0.5`` mean "no button"; everything
    else rounds and clamps into range. The rounding is exactly the
    staircase the module docstring warns about.
    """
    if value <= NO_BUTTON + 0.5:
        return NO_BUTTON
    return int(np.clip(round(value), 0, max(max_buttons - 1, 0)))


def wiring_from_params(params: Params, num_lamps: int,
                       num_buttons: int) -> List[Condition]:
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
    wiring_full = []
    for i in range(max_lamps):
        enablers = tuple(e for e in (
            _wired_index(params.get(f"enabler_{i}", NO_BUTTON), max_buttons),
            _wired_index(params.get(f"enabler2_{i}", NO_BUTTON), max_buttons))
                         if e != NO_BUTTON)
        wiring_full.append(
            Condition(
                _wired_index(params.get(f"driver_{i}", 0.0), max_buttons),
                enablers,
                _wired_index(params.get(f"inhibitor_{i}", NO_BUTTON),
                             max_buttons)).canonical())
    return project_wiring(wiring_full, num_buttons, num_lamps)


def _driven(button_on: List[bool], cond: Condition, armed: bool) -> bool:
    """Whether a lamp's drive holds: armed, driver and enablers on, inhibitor
    off."""
    if not armed:
        return False
    if not 0 <= cond.driver < len(button_on) or \
            not button_on[cond.driver]:
        return False
    for e in cond.enablers:
        if not 0 <= e < len(button_on) or not button_on[e]:
            return False
    if cond.inhibitor != NO_BUTTON and \
            0 <= cond.inhibitor < len(button_on) and \
            button_on[cond.inhibitor]:
        return False
    return True


def step_board(button_on: List[bool], prev_on: List[bool],
               wiring: List[Condition], charges: Dict[str, float],
               armed: Dict[str, bool], tripped: bool, lamp_names: List[str],
               params: Params) -> bool:
    """One step of the board's hidden rule, in place; returns the breaker.

    Shared by the fully- and partially-observable answer keys: the same
    rule, fed the charge and latch from the observation in one and from
    the latent block in the other.
    """
    if tripped and not any(button_on):
        tripped = False
    driven: List[bool] = []
    for i, name in enumerate(lamp_names):
        if i >= len(wiring):
            driven.append(False)
            continue
        cond = wiring[i]
        d = cond.driver
        driver_on = 0 <= d < len(button_on) and button_on[d]
        if not driver_on:
            armed[name] = False
        elif not prev_on[d] or not CFG.busyboard_latch:
            armed[name] = all(0 <= e < len(button_on) and button_on[e]
                              for e in cond.enablers)
        driven.append(not tripped
                      and _driven(button_on, cond, armed.get(name, False)))
    limit = int(CFG.busyboard_breaker_limit)
    if 0 < limit < sum(driven) and not tripped:
        tripped = True
        driven = [False] * len(driven)
        for name in lamp_names:
            charges[name] = 0.0
            armed[name] = False
    for i, name in enumerate(lamp_names):
        charge = float(charges.get(name, 0.0))
        if driven[i]:
            charge = min(1.0, charge + params["charge_rate"])
        else:
            charge = max(0.0, charge - params["decay_rate"])
        charges[name] = charge
    return tripped


# ── Residual rules ────────────────────────────────────────────────


def _charging(  # pylint: disable=unused-argument
        observation: State, latent: Dict[str, Any], history: History,
        updates: ResidualUpdate, params: Params) -> ResidualUpdate:
    """Driven lamps charge, undriven lamps bleed; latch, breaker and brightness
    follow.

    One rule covers the whole board: the per-lamp differences live
    entirely in the wiring parameters, not in the control flow. Charge,
    brightness and the latch are written, because in the fully-
    observable setting all three are features the fit is scored on. The
    latch is edge-triggered, so the previous observation in ``history``
    supplies the driver's earlier state.
    """
    objs = objs_by_type(observation)
    lamps = sorted(objs.get("lamp", []), key=lambda o: o.name)
    button_on = _button_states(observation)
    prev_on = _previous_button_states(history, button_on)
    wiring = wiring_from_params(params, len(lamps), len(button_on))
    names = [lamp.name for lamp in lamps]
    charges = {
        lamp.name: float(observation.get(lamp, "charge"))
        for lamp in lamps
    }
    armed = {
        lamp.name: bool(observation.get(lamp, "armed") > 0.5)
        for lamp in lamps
    }
    tripped = _breaker_tripped(observation)
    tripped = step_board(button_on, prev_on, wiring, charges, armed,
                         bool(tripped), names, params)
    for lamp in lamps:
        updates.setdefault(lamp, {})["charge"] = charges[lamp.name]
        updates[lamp]["brightness"] = _brightness(charges[lamp.name])
        updates[lamp]["armed"] = float(armed[lamp.name])
    for breaker in objs.get("breaker", []):
        updates.setdefault(breaker, {})["tripped"] = float(tripped)
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
    wiring = full_wiring()
    core_buttons, core_lamps = core_board()
    specs = [
        ParamSpec("charge_rate", CFG.busyboard_charge_rate, lo=0.0, hi=1.0),
        ParamSpec("decay_rate", CFG.busyboard_decay_rate, lo=0.0, hi=1.0),
    ]
    for i in range(max_lamps):
        cond = wiring[i]
        # A core lamp's drive is wired to core buttons only (the
        # extension contract of ``project_wiring``), so its range is the
        # core board's; its inhibitor may be any button.
        hi = float((core_buttons if i < core_lamps else max_buttons) - 1)
        enablers = list(cond.enablers) + [NO_BUTTON, NO_BUTTON]
        # Wiring slots are indices the rule rounds before use: discrete,
        # so uncertainty jitter must not perturb them into a rewiring.
        specs.append(
            ParamSpec(f"driver_{i}",
                      float(cond.driver),
                      lo=0.0,
                      hi=hi,
                      discrete=True))
        specs.append(
            ParamSpec(f"enabler_{i}",
                      float(enablers[0]),
                      lo=float(NO_BUTTON),
                      hi=hi,
                      discrete=True))
        specs.append(
            ParamSpec(f"enabler2_{i}",
                      float(enablers[1]),
                      lo=float(NO_BUTTON),
                      hi=hi,
                      discrete=True))
        specs.append(
            ParamSpec(f"inhibitor_{i}",
                      float(cond.inhibitor),
                      lo=float(NO_BUTTON),
                      hi=float(max_buttons - 1),
                      discrete=True))
    return specs


# ── Public API: consumed by read_simulator_components ────────────

# Same contract used by agent-synthesized simulator files. PARAM_SPECS is
# bound to the callable so CFG-dependent defaults resolve when the loader
# pulls the value, after CFG is final.
RESIDUAL_RULES = [_charging]
PARAM_SPECS = _build_param_specs
RESIDUAL_FEATURES: Dict[str, List[str]] = {
    "lamp": ["charge", "brightness", "armed"],
    "breaker": ["tripped"],
}


def _latent_init() -> Dict[str, Any]:
    """The fully-observable key keeps nothing hidden; the block exists so the
    recurrent signature has one to thread."""
    return {}


LATENT_INIT = _latent_init

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
