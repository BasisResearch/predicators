"""Ground-truth simulator program for pybullet_boil process dynamics.

Reproduces the custom step logic from pybullet_boil.py as composable
process rules using plain numpy/float arithmetic.
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


# Static specs for tests / introspection (uses CFG defaults at import time).
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
    """Faucet on + jug aligned → fill jug; otherwise spill."""
    objs = _objs_by_type(state)
    for faucet in objs.get("faucet", []):
        if state.get(faucet, "is_on") <= 0.5:
            continue

        fx = float(state.get(faucet, "x"))
        fy = float(state.get(faucet, "y"))
        frot = float(state.get(faucet, "rot"))
        out_x = fx + params["faucet_x_len"] * np.cos(frot)
        out_y = fy - params["faucet_x_len"] * np.sin(frot)

        jug_catching = False
        for jug in objs.get("jug", []):
            if state.get(jug, "is_held") > 0.5:
                continue
            jx = float(state.get(jug, "x"))
            jy = float(state.get(jug, "y"))
            dist = float(np.hypot(out_x - jx, out_y - jy))

            if dist < params["faucet_align_threshold"]:
                water = float(state.get(jug, "water_volume"))
                if water < params["max_jug_water_capacity"]:
                    new_water = min(params["max_jug_water_capacity"],
                                    water + params["water_fill_speed"])
                    updates.setdefault(jug, {})["water_volume"] = new_water
                    jug_catching = True
                else:
                    spill = float(state.get(faucet, "spilled_level"))
                    new_spill = min(params["max_water_spill_width"],
                                    spill + params["water_fill_speed"])
                    updates.setdefault(faucet, {})["spilled_level"] = new_spill
                break

        if not jug_catching:
            spill = float(state.get(faucet, "spilled_level"))
            new_spill = min(params["max_water_spill_width"],
                            spill + params["water_fill_speed"])
            updates.setdefault(faucet, {})["spilled_level"] = new_spill

    return updates


def _heating(state: State, updates: ProcessUpdate,
             params: Params) -> ProcessUpdate:
    """Burner on + jug with water aligned → heat jug."""
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

            if dist < params["burner_align_threshold"]:
                heat = float(state.get(jug, "heat_level"))
                new_heat = min(1.0, heat + params["heating_speed"])
                updates.setdefault(jug, {})["heat_level"] = new_heat

    return updates


def _happiness(state: State, updates: ProcessUpdate,
               params: Params) -> ProcessUpdate:
    """Jug filled + boiled + no spill + burner off → human happy."""
    objs = _objs_by_type(state)
    faucets = objs.get("faucet", [])
    burners = objs.get("burner", [])

    def _get_val(obj: Object, feat: str) -> float:
        val = updates.get(obj, {}).get(feat, None)
        if val is not None:
            return float(val) if hasattr(val, 'item') else val
        return float(state.get(obj, feat))

    any_spill = any(_get_val(f, "spilled_level") > 0 for f in faucets)
    any_burner_on = any(state.get(b, "is_on") > 0.5 for b in burners)

    if any_spill or any_burner_on:
        return updates

    for jug in objs.get("jug", []):
        water = _get_val(jug, "water_volume")
        heat = _get_val(jug, "heat_level")
        if water >= params["water_filled_height"] and heat >= 1.0:
            for human in objs.get("human", []):
                h = float(state.get(human, "happiness_level"))
                new_h = min(1.0, h + params["happiness_speed"])
                updates.setdefault(human, {})["happiness_level"] = new_h

    return updates


PROCESS_RULES = [_water_filling, _heating, _happiness]


class PyBulletBoilGroundTruthSimulatorFactory(GroundTruthSimulatorFactory):
    """GT process-dynamics simulator for pybullet_boil."""

    @classmethod
    def get_env_names(cls) -> set:
        return {"pybullet_boil"}

    @classmethod
    def get_rules(cls) -> list:
        return list(PROCESS_RULES)

    @classmethod
    def get_param_specs(cls) -> list:
        return _build_param_specs()


def get_gt_process_features() -> Dict[str, List[str]]:
    """Process features handled by the simulator (not PyBullet)."""
    return {
        "jug": ["water_volume", "heat_level"],
        "faucet": ["spilled_level"],
        "human": ["happiness_level"],
    }
