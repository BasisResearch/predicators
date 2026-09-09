"""Partially-observable ground-truth simulator for pybullet_busyboard.

Sibling of ``gt_simulator.py`` for the partially-observable setting,
where neither a lamp's ``charge`` nor its arming latch ``armed`` is an
observable feature: the agent sees only ``brightness``, which is flat
at zero through the whole early accumulation and only then ramps, and
the breaker tile. This module is the answer key for that inference - it
carries the charge and the latch explicitly in the recurrent latent
block and maps the charge to the observable through the env's own ramp.

Differences from the fully-observable ``gt_simulator.py``:

* ``_charging`` reads the per-lamp charge and latch from
  ``latent["charge"]`` and ``latent["armed"]`` (``{lamp_name: value}``
  dicts threaded across steps by ``compute_sse_recurrent``) and writes
  only the observable ``brightness`` and the breaker's ``tripped``.
* ``LATENT_INIT`` declares the initial latent block. It is the callable
  form so each rollout gets its own nested dict; a module-level literal
  would be shared across trajectories by ``init_latent`` and silently
  accumulate charge from one rollout into the next.
* ``RESIDUAL_FEATURES`` scopes the fit to what a partially-observable
  board reports.

What makes this domain's partial observability different from boil's:
there, the hidden quantity is a scalar and the process form is known, so
the fit is an ordinary continuous identification problem. Here two
latents sit *behind* an unknown discrete structure - you cannot fit the
accumulation rate without first knowing which buttons feed which lamp,
you cannot read the wiring off the observations without accounting for
the delay that lets a later button press take credit for an earlier
one's effect, and the latch means the same button setting is evidence
for different wirings depending on the order it was reached in.
Structure, rate and latch are only jointly identifiable, and the
experiments that separate them are exactly the ones that hold a
configuration still long enough for the ramp to appear and reach it in
more than one order.

See ``gt_simulator.py`` for why the wiring enters as relaxed real-valued
parameters, and what a proper categorical spec would replace.
"""
from __future__ import annotations

from typing import Any, Dict, List

from predicators.code_sim_learning.utils import History, Params, \
    ResidualUpdate, objs_by_type
from predicators.ground_truth_models import GroundTruthSimulatorFactory
from predicators.ground_truth_models.busyboard.gt_simulator import \
    _breaker_tripped, _brightness, _build_param_specs, _button_states, \
    _previous_button_states, step_board, wiring_from_params
from predicators.settings import CFG
from predicators.structs import State

# ── Residual rules ────────────────────────────────────────────────


def _charging(  # pylint: disable=unused-argument
        observation: State, latent: Dict[str, Any], history: History,
        updates: ResidualUpdate, params: Params) -> ResidualUpdate:
    """Driven lamps charge in the latent; only brightness and the breaker are
    emitted.

    The carried charge and latch are a sufficient statistic of the drive
    history so far; ``history`` is read only for the previous button
    states, which the edge-triggered latch needs. This is Pattern B
    (physical latent + monotone readout), the same shape as boil's
    hidden heat behind ``bubbling_level``, with a second, binary latent
    behind it.
    """
    charges: Dict[str, float] = latent.setdefault("charge", {})
    armed: Dict[str, bool] = latent.setdefault("armed", {})
    objs = objs_by_type(observation)
    lamps = sorted(objs.get("lamp", []), key=lambda o: o.name)
    button_on = _button_states(observation)
    prev_on = _previous_button_states(history, button_on)
    wiring = wiring_from_params(params, len(lamps), len(button_on))
    names = [lamp.name for lamp in lamps]
    tripped = _breaker_tripped(observation)
    if tripped is None:
        tripped = bool(latent.get("tripped", False))
    tripped = step_board(button_on, prev_on, wiring, charges, armed,
                         bool(tripped), names, params)
    latent["tripped"] = tripped
    for lamp in lamps:
        updates.setdefault(lamp,
                           {})["brightness"] = _brightness(charges[lamp.name])
    for breaker in objs.get("breaker", []):
        updates.setdefault(breaker, {})["tripped"] = float(tripped)
    return updates


# ── Latent block ─────────────────────────────────────────────────


def _latent_init() -> Dict[str, Any]:
    """Fresh per-lamp charge and latch block for a new rollout."""
    return {"charge": {}, "armed": {}, "tripped": False}


# ── Public API: consumed by read_simulator_components ────────────

RESIDUAL_RULES = [_charging]
PARAM_SPECS = _build_param_specs
LATENT_INIT = _latent_init
RESIDUAL_FEATURES: Dict[str, List[str]] = {
    "lamp": ["brightness"],
    "breaker": ["tripped"],
}

# ── Factory binding ──────────────────────────────────────────────


class PyBulletBusyBoardPOGroundTruthSimulatorFactory(
        GroundTruthSimulatorFactory):
    """PO GT residual-dynamics simulator for pybullet_busyboard.

    Claims the env only in partially-observable mode; the fully-
    observable ``gt_simulator.py`` claims it otherwise, so
    ``get_gt_simulator``'s env-name dispatch resolves to exactly one
    module per run.
    """

    @classmethod
    def get_env_names(cls) -> set:
        if CFG.partially_observable:
            return {"pybullet_busyboard"}
        return set()
