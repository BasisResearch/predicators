"""Partially-observable ground-truth simulator for pybullet_busyboard.

Sibling of ``gt_simulator.py`` for the partially-observable setting,
where a lamp's ``charge`` is not an observable feature: the agent sees
only ``brightness``, which is flat at zero through the whole early
accumulation and only then ramps. This module is the answer key for
that inference - it carries the charge explicitly in the recurrent
latent block and maps it to the observable through the env's own ramp.

Differences from the fully-observable ``gt_simulator.py``:

* ``_charging`` uses the recurrent 5-arg signature
  ``rule(observation, latent, history, updates, params)``, carrying the
  per-lamp charge in ``latent["charge"]`` (a ``{lamp_name: charge}``
  dict threaded across steps by ``compute_sse_recurrent``) and writing
  only the observable ``brightness``.
* ``LATENT_INIT`` declares the initial latent block. It is the callable
  form so each rollout gets its own nested dict; a module-level literal
  would be shared across trajectories by ``init_latent`` and silently
  accumulate charge from one rollout into the next.
* ``RESIDUAL_FEATURES`` scopes the fit to ``brightness`` alone, which is
  all a partially-observable board reports.

What makes this domain's partial observability different from boil's:
there, the hidden quantity is a scalar and the process form is known, so
the fit is an ordinary continuous identification problem. Here the
latent sits *behind* an unknown discrete structure - you cannot fit the
accumulation rate without first knowing which buttons feed which lamp,
and you cannot read the wiring off the observations without accounting
for the delay that lets a later button press take credit for an earlier
one's effect. Structure and rate are only jointly identifiable, and the
experiments that separate them are exactly the ones that hold a
configuration still long enough for the ramp to appear.

See ``gt_simulator.py`` for why the wiring enters as relaxed real-valued
parameters, and what a proper categorical spec would replace.
"""

from __future__ import annotations

from typing import Any, Dict, List

from predicators.code_sim_learning.utils import History, Params, \
    ResidualUpdate, objs_by_type
from predicators.ground_truth_models import GroundTruthSimulatorFactory
from predicators.ground_truth_models.busyboard.gt_simulator import \
    _brightness, _build_param_specs, _button_states, _driven, \
    wiring_from_params
from predicators.settings import CFG
from predicators.structs import State

# ── Residual rules ────────────────────────────────────────────────


def _charging(  # pylint: disable=unused-argument
        observation: State, latent: Dict[str, Any], history: History,
        updates: ResidualUpdate, params: Params) -> ResidualUpdate:
    """Driven lamps charge in the latent; only brightness is emitted.

    ``history`` is unused: the carried charge is a sufficient statistic
    of the drive history so far, so no look-back is needed. This is
    Pattern B (physical latent + monotone readout), the same shape as
    boil's hidden heat behind ``bubbling_level``.
    """
    charges: Dict[str, float] = latent.setdefault("charge", {})
    objs = objs_by_type(observation)
    lamps = sorted(objs.get("lamp", []), key=lambda o: o.name)
    button_on = _button_states(observation)
    driver, enabler = wiring_from_params(params, len(lamps), len(button_on))

    for i, lamp in enumerate(lamps):
        charge = float(charges.get(lamp.name, 0.0))
        if _driven(button_on, driver[i], enabler[i]):
            charge = min(1.0, charge + params["charge_rate"])
        else:
            charge = max(0.0, charge - params["decay_rate"])
        charges[lamp.name] = charge
        updates.setdefault(lamp, {})["brightness"] = _brightness(charge)

    return updates


# ── Latent block ─────────────────────────────────────────────────


def _latent_init() -> Dict[str, Dict[str, float]]:
    """Fresh per-lamp charge block for a new rollout."""
    return {"charge": {}}


# ── Public API: consumed by read_simulator_components ────────────

RESIDUAL_RULES = [_charging]

PARAM_SPECS = _build_param_specs

LATENT_INIT = _latent_init

RESIDUAL_FEATURES: Dict[str, List[str]] = {
    "lamp": ["brightness"],
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
