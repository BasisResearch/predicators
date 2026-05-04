"""Ground-truth simulator program for pybullet_boil process dynamics.

Reproduces the custom step logic from pybullet_boil.py as composable
process rules using plain numpy/float arithmetic.

Parameter-dependent gates (alignment thresholds, capacity caps, fill
height) are softened with sigmoid weights so the residual is
differentiable in those parameters. The primary consumer is the
Levenberg-Marquardt fit (and its Hessian identifiability diagnostic),
which builds a finite-difference Jacobian and would see J ~ 0 almost
everywhere with hard indicators. Smoothing also keeps MCMC walkers
from stalling on flat-likelihood plateaus, but emcee is gradient-free
and benefits less directly. State-dependent gates (faucet on/off, jug
held) remain hard since they don't enter the parameter likelihood.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from predicators.code_sim_learning.training import ParamSpec
from predicators.code_sim_learning.utils import ProcessUpdate
from predicators.ground_truth_models import GroundTruthSimulatorFactory
from predicators.settings import CFG
from predicators.structs import Object, State

# Constants matching pybullet_boil.py exactly. Note: water_fill_speed is
# derived from CFG at spec-build time (env uses
# CFG.boil_water_fill_speed * water_height_to_level_ratio).
HEATING_SPEED = 0.03
HAPPINESS_SPEED = 0.05
MAX_JUG_WATER_CAPACITY = 1.3
WATER_FILLED_HEIGHT = 0.8
MAX_WATER_SPILL_WIDTH = 0.3
FAUCET_ALIGN_THRESHOLD = 0.1
BURNER_ALIGN_THRESHOLD = 0.05
FAUCET_X_LEN = 0.15
_WATER_HEIGHT_TO_LEVEL_RATIO = 10

# Smoothing scale for parameter-dependent gates. Small enough that gates
# are ~99% saturated when the operand is one threshold-width into the
# active region, large enough to give MCMC a usable gradient near the
# cliff. 0.02 is in the right ballpark for both spatial thresholds
# (~0.05–0.15 m) and water-level thresholds (~0.3–1.3).
_SOFT_EPS = 0.02


def _sigmoid(z: float) -> float:
    """Numerically-stable scalar sigmoid."""
    if z >= 0:
        return 1.0 / (1.0 + np.exp(-z))
    ez = np.exp(z)
    return ez / (1.0 + ez)


def _build_param_specs() -> List[ParamSpec]:
    """Build at call time so CFG-driven values match the current run."""
    water_fill_speed = (CFG.boil_water_fill_speed *
                        _WATER_HEIGHT_TO_LEVEL_RATIO)
    return [
        ParamSpec("water_fill_speed", water_fill_speed, lo=0.0),
        ParamSpec("heating_speed", HEATING_SPEED, lo=0.0),
        ParamSpec("happiness_speed", HAPPINESS_SPEED, lo=0.0),
        ParamSpec("max_jug_water_capacity", MAX_JUG_WATER_CAPACITY, lo=0.0),
        ParamSpec("water_filled_height", WATER_FILLED_HEIGHT, lo=0.0),
        ParamSpec("max_water_spill_width", MAX_WATER_SPILL_WIDTH, lo=0.0),
        ParamSpec("faucet_x_len", FAUCET_X_LEN, lo=0.0),
        ParamSpec("faucet_align_threshold", FAUCET_ALIGN_THRESHOLD, lo=0.0),
        ParamSpec("burner_align_threshold", BURNER_ALIGN_THRESHOLD, lo=0.0),
    ]


# Module-level globals consumed by ``read_simulator_components`` (the
# same contract used by agent-synthesized simulator files).
# ``PARAM_SPECS`` is bound to the *callable* rather than its result so
# CFG-dependent defaults are evaluated when the loader pulls the value,
# after CFG has been finalized.
PARAM_SPECS = _build_param_specs

PROCESS_FEATURES: Dict[str, List[str]] = {
    "jug": ["water_volume", "heat_level"],
    "faucet": ["spilled_level"],
    "human": ["happiness_level"],
}

# Backward-compat alias for tests that import a static, eagerly-built
# spec list (uses CFG defaults at import time).
BOIL_PARAM_SPECS: List[ParamSpec] = _build_param_specs()

Params = Dict[str, float]


def _objs_by_type(state: State) -> Dict[str, List[Object]]:
    """Group state objects by type name."""
    groups: Dict[str, List[Object]] = {}
    for o in state:
        groups.setdefault(o.type.name, []).append(o)
    return groups


def _water_filling(state: State, updates: ProcessUpdate,
                   params: Params) -> ProcessUpdate:
    """Faucet on + jug aligned → fill jug; otherwise spill.

    Alignment and capacity gates are soft (sigmoid-weighted) so the
    residual is differentiable in ``faucet_align_threshold``,
    ``faucet_x_len``, and ``max_jug_water_capacity`` — needed for the LM
    Jacobian (and downstream Hessian diagnostic) to be informative.
    """
    objs = _objs_by_type(state)
    for faucet in objs.get("faucet", []):
        if state.get(faucet, "is_on") <= 0.5:
            continue

        fx = float(state.get(faucet, "x"))
        fy = float(state.get(faucet, "y"))
        frot = float(state.get(faucet, "rot"))
        out_x = fx + params["faucet_x_len"] * np.cos(frot)
        out_y = fy - params["faucet_x_len"] * np.sin(frot)

        # Closest non-held jug picks up the catch (matches the
        # original "first aligned wins" semantics for single-jug tasks).
        best_jug, best_dist = None, float("inf")
        for jug in objs.get("jug", []):
            if state.get(jug, "is_held") > 0.5:
                continue
            jx = float(state.get(jug, "x"))
            jy = float(state.get(jug, "y"))
            d = float(np.hypot(out_x - jx, out_y - jy))
            if d < best_dist:
                best_jug, best_dist = jug, d

        catch_w = 0.0
        if best_jug is not None:
            water = float(state.get(best_jug, "water_volume"))
            align_w = _sigmoid(
                (params["faucet_align_threshold"] - best_dist) / _SOFT_EPS)
            cap_w = _sigmoid(
                (params["max_jug_water_capacity"] - water) / _SOFT_EPS)
            catch_w = align_w * cap_w
            new_water = water + catch_w * params["water_fill_speed"]
            updates.setdefault(best_jug, {})["water_volume"] = new_water

        # Uncaught water spills (clamped at max_water_spill_width).
        spill = float(state.get(faucet, "spilled_level"))
        new_spill = min(params["max_water_spill_width"],
                        spill + (1.0 - catch_w) * params["water_fill_speed"])
        updates.setdefault(faucet, {})["spilled_level"] = new_spill

    return updates


def _heating(state: State, updates: ProcessUpdate,
             params: Params) -> ProcessUpdate:
    """Burner on + jug with water aligned → heat jug.

    Alignment gate is soft so the residual is differentiable in
    ``burner_align_threshold`` (LM's finite-difference Jacobian needs
    this; MCMC also avoids flat-likelihood plateaus as a side effect).
    The heat cap at 1.0 stays hard since 1.0 is a constant boundary, not
    a learned parameter.
    """
    objs = _objs_by_type(state)
    for burner in objs.get("burner", []):
        if state.get(burner, "is_on") <= 0.5:
            continue
        bx = float(state.get(burner, "x"))
        by = float(state.get(burner, "y"))

        for jug in objs.get("jug", []):
            if state.get(jug, "is_held") > 0.5:
                continue
            if state.get(jug, "water_volume") <= 0.0:
                continue
            jx = float(state.get(jug, "x"))
            jy = float(state.get(jug, "y"))
            dist = float(np.hypot(bx - jx, by - jy))

            align_w = _sigmoid(
                (params["burner_align_threshold"] - dist) / _SOFT_EPS)
            heat = float(state.get(jug, "heat_level"))
            new_heat = min(1.0, heat + align_w * params["heating_speed"])
            updates.setdefault(jug, {})["heat_level"] = new_heat

    return updates


def _happiness(state: State, updates: ProcessUpdate,
               params: Params) -> ProcessUpdate:
    """Jug filled + boiled + no spill + burner off → human happy.

    The water-filled gate is soft on ``water_filled_height`` so the
    residual is differentiable in that parameter for LM (and emcee gets
    a non-flat likelihood as a side effect). The heat>=1.0 gate stays
    hard (1.0 is a constant cap, not a learned parameter). Spill /
    burner-on gates are state-dependent.
    """
    objs = _objs_by_type(state)
    faucets = objs.get("faucet", [])
    burners = objs.get("burner", [])

    def _get_val(obj: Object, feat: str) -> float:
        val = updates.get(obj, {}).get(feat, None)
        if val is not None:
            return float(val) if hasattr(val, 'item') else val
        return float(state.get(obj, feat))

    # Spilled-level prediction can be a tiny positive number under soft
    # semantics even when the env reports zero, so treat anything below
    # the smoothing scale as "no spill" to avoid spuriously gating
    # happiness off.
    any_spill = any(_get_val(f, "spilled_level") > _SOFT_EPS for f in faucets)
    any_burner_on = any(state.get(b, "is_on") > 0.5 for b in burners)

    if any_spill or any_burner_on:
        return updates

    for jug in objs.get("jug", []):
        water = _get_val(jug, "water_volume")
        heat = _get_val(jug, "heat_level")
        if heat < 1.0:
            continue
        filled_w = _sigmoid(
            (water - params["water_filled_height"]) / _SOFT_EPS)
        for human in objs.get("human", []):
            h = float(state.get(human, "happiness_level"))
            new_h = min(1.0, h + filled_w * params["happiness_speed"])
            updates.setdefault(human, {})["happiness_level"] = new_h

    return updates


PROCESS_RULES = [_water_filling, _heating, _happiness]


def get_gt_process_features() -> Dict[str, List[str]]:
    """Backward-compat accessor; prefer the ``PROCESS_FEATURES`` global."""
    return dict(PROCESS_FEATURES)


class PyBulletBoilGroundTruthSimulatorFactory(GroundTruthSimulatorFactory):
    """GT process-dynamics simulator for pybullet_boil.

    The actual simulator components (``PROCESS_RULES``, ``PARAM_SPECS``,
    ``PROCESS_FEATURES``) live as module globals above; this class only
    pins the env-name binding so ``get_gt_simulator`` can locate the
    right module via the factory registry.
    """

    @classmethod
    def get_env_names(cls) -> set:
        return {"pybullet_boil"}
