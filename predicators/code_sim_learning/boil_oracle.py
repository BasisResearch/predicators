"""Current Boil mechanisms with observation-driven, per-episode memory.

The source exporter replaces the native superclass with the supplied
model base. Native helper methods implement the simulated effects; the
pure observation callback infers heat and switch history during real
execution. It never receives the environment's privileged heat record.
"""
from __future__ import annotations

# The oracle restores native mechanism records on its private objects.
# pylint: disable=protected-access
import copy
from typing import Any, ClassVar, Dict, List

import numpy as np

from predicators.envs.pybullet_boil import PyBulletBoilEnv
from predicators.settings import CFG
from predicators.structs import Action, State


class BoilOracle(PyBulletBoilEnv):
    """Fixed dynamics, with memory restored from the inferred state."""

    AGENT_PARAM_SPECS: ClassVar[List[Any]] = []
    RESIDUAL_FEATURES = {
        "jug": ["water_volume", "bubbling_level"],
        "faucet": ["spilled_level"],
        "human": ["happiness_level"]
    }
    MODEL_STATE_INIT = {
        "heat": {},
        "previous_on": {},
        "water": {},
        "spill": None
    }

    @classmethod
    def get_name(cls) -> str:
        return "pybullet_boil_oracle_model"

    def _set_state(self, state: State) -> None:
        # Native burner previous-on and human happiness are Object sim-data.
        # Cross-world snapshots must not alias those mutable records.
        private = copy.deepcopy(state)
        private.privileged = None
        super()._set_state(private)

    @classmethod
    def update_model_state(cls, observation: State, model_state: Dict[str,
                                                                      Any],
                           params: Dict[str, float], action: Action) -> None:
        """Infer hidden counters using only the observed action outcome.

        Noise can change alignment or volume decisions, as for a learned
        model. Knowing the mechanism does not grant the true current
        state.
        """
        del params, action
        jugs = [o for o in observation if o.type.name == "jug"]
        burners = [o for o in observation if o.type.name == "burner"]
        faucets = [o for o in observation if o.type.name == "faucet"]
        previous = model_state["previous_on"]
        heat = model_state["heat"]
        required = (cls.water_filled_height
                    if CFG.boil_require_jug_full_to_heatup else 0.0)
        for burner in burners:
            if (observation.get(burner, "is_on") <= 0.5
                    or not previous.get(burner.name, False)):
                continue
            for jug in jugs:
                if (observation.get(jug, "is_held") > 0.5
                        or observation.get(jug, "water_volume") <= required):
                    continue
                distance = np.hypot(
                    observation.get(burner, "x") - observation.get(jug, "x"),
                    observation.get(burner, "y") - observation.get(jug, "y"))
                if distance < cls.burner_align_threshold:
                    heat[jug.name] = min(
                        1.0,
                        heat.get(jug.name, 0.0) + cls.heating_speed)
        rate = CFG.boil_water_fill_speed * cls.water_height_to_level_ratio
        if model_state["spill"] is None:
            model_state["spill"] = -20 * rate
        for faucet in faucets:
            if (observation.get(faucet, "is_on") <= 0.5
                    or not previous.get(faucet.name, False)):
                continue
            angle = observation.get(faucet, "rot")
            x = (observation.get(faucet, "x") +
                 np.cos(angle) * cls.faucet_outlet_local_dx -
                 np.sin(angle) * cls.faucet_outlet_local_dy)
            y = (observation.get(faucet, "y") +
                 np.sin(angle) * cls.faucet_outlet_local_dx +
                 np.cos(angle) * cls.faucet_outlet_local_dy)
            under = [
                j for j in jugs
                if observation.get(j, "is_held") <= 0.5 and np.hypot(
                    observation.get(j, "x") - x,
                    observation.get(j, "y") - y) < cls.faucet_align_threshold
            ]
            # The outcome already includes this step's fill. Use the last
            # observed volume to avoid counting its first capacity hit as
            # an overflow, matching the native fill-before-heat cadence.
            overflowing = sum(model_state["water"].get(
                j.name, observation.get(j, "water_volume")) >=
                              cls.max_jug_water_capacity for j in under)
            increments = overflowing if under else 1
            model_state["spill"] = min(
                cls.max_water_spill_width,
                model_state["spill"] + increments * rate)
        model_state["previous_on"] = {
            o.name: observation.get(o, "is_on") > 0.5
            for o in burners + faucets
        }
        model_state["water"] = {
            j.name: observation.get(j, "water_volume")
            for j in jugs
        }

    def _set_domain_specific_state(self, state: State) -> None:
        state = state.copy()
        state.privileged = None
        super()._set_domain_specific_state(state)
        # The superclass's privileged channel is not an inference source.
        # Restore only memory carried by the model/observation tracker.
        memory = self.model_state
        self._heat_levels = dict(memory["heat"])
        for burner in state.get_objects(self._burner_type):
            burner.prev_on = float(memory["previous_on"].get(burner.name, 0.0))
        self._faucet.prev_on = float(memory["previous_on"].get(
            self._faucet.name, 0.0))
        if memory["spill"] is not None:
            self._faucet._spilled_level = memory["spill"]
            if memory["spill"] > 0.0:
                self._spilled_water_id = self._create_spilled_water_block(
                    memory["spill"], state)
        self._update_liquid_colors(state)

    def _domain_specific_step(self) -> None:
        """Use the current environment's complete effect ordering."""
        state = self._get_state()
        self._handle_faucet_logic(state)
        self._handle_heating_logic(state)
        self._update_liquid_colors(state)
        self._update_liquid_positions(state)
        self._update_burner_colors(state)
        self._update_human_happiness(state)
        self._update_prev_on_states(state)
        # Model-world hidden values are owned by this simulator. Export
        # them so future planning nodes resume its exact mechanism state.
        self.model_state["heat"] = dict(self._heat_levels)
        self.model_state["previous_on"] = {
            o.name: bool(o.prev_on)
            for o in [*state.get_objects(self._burner_type), self._faucet]
        }
        self.model_state["spill"] = self._faucet._spilled_level
        after = self._get_state()
        self.model_state["water"] = {
            o.name: after.get(o, "water_volume")
            for o in after.get_objects(self._jug_type)
        }
