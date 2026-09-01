"""Residual dynamics for pybullet_domino_fan.

DECISION RECORD
===============
Evidence (2 successful interaction trajectories, task 0, reward 0.95 each):

* Base sim carries: robot motion, grasping (Pick/Place), and ALL rigid-body
  domino-on-domino cascade physics.  In both trajectories the blue domino is
  picked from (0.47, 1.275) and placed between the green (x=0.54) and the
  purple (x=0.736); the recorded contact-driven part of the cascade
  (green -> blue -> purple, rolls ramping to ~1.57 within 3-4 steps of each
  contact) is ordinary rigid-body physics and needs no rule.

* MISSING mechanism = the fan's wind.  Exactly one step after fan_0.is_on
  flips to 1 (t=98 -> 99 in traj 0, t=112 -> 113 in traj 1) the GREEN domino -
  the one nearest the fan along the fan's facing axis - begins to translate
  in +x and to roll (roll 0 -> 0.047 -> 0.126 -> 0.19 -> 0.29 -> 0.49 ...),
  with an essentially identical profile in both trajectories.  Nothing else
  moves.  The base sim replays this segment inert (confirmed by rollout probe
  and by two earlier cycles that saw no topple at all with the fan on), so
  this is diagnostic-ladder case 2: an exogenous influence the engine knows
  nothing about.  It is modelled with a constant `cmds.apply_force` gated on
  `is_on`, NOT with a feature overwrite - the engine must resolve the
  resulting contacts (that is what produces the cascade).

* Wind is OCCLUDED, not range-limited-only.  The placed blue domino sits
  0.30-0.32 m from the fan, directly on the fan's axis, and shows roll
  EXACTLY 0.000 until the toppling green physically reaches it.  A
  sub-threshold force would have produced a visible lean, so the blue
  receives no wind at all while the upright green stands in front of it.
  The rule therefore applies wind only to a domino that has no upright
  domino between it and the fan (lateral overlap < `wind_beam_halfwidth`),
  plus a generous learnable `wind_range` cutoff.

* Latent state: NONE needed.  Every driver of the residual is observable
  (fan.is_on, fan.rot, domino poses).  `latent` is threaded but unused.

* PHYSICAL_PARAMS: not declared (see rollout sweep evidence in the session);
  the cascade geometry the base sim produces already matches the recorded
  contact chain once the wind force exists.  Re-examine only if a calibrated
  wind still fails to propagate.
"""

from typing import Any, Dict, List

import numpy as np

# ---------------------------------------------------------------- params ----
PARAM_SPECS = [
    # Constant wind force (N) applied to the exposed domino while the fan is on.
    ParamSpec("wind_force", 0.18, lo=0.179999, hi=0.180001),
    # Reach of the wind measured along the fan's facing axis (m).
    ParamSpec("wind_range", 0.60, lo=0.05, hi=1.5),
    # Half-width of the wind beam, lateral to the facing axis (m).
    ParamSpec("wind_beam_halfwidth", 0.06, lo=0.01, hi=0.30),
    # |roll| above which a domino counts as toppled (no longer blocks wind,
    # and, for predicates, counts as "down").
    ParamSpec("topple_roll", 0.60, lo=0.15, hi=1.40),
    # Upright tolerance: |roll| below this means "still standing".
    ParamSpec("upright_roll", 0.10, lo=0.01, hi=0.40),
]

LATENT_INIT: Dict[str, Any] = {}

RESIDUAL_FEATURES = {"domino": ["x", "z", "roll"]}


# ----------------------------------------------------------------- rules ----
def _wind_axis(observation, fan):
    rot = float(observation.get(fan, "rot"))
    return np.array([np.cos(rot), np.sin(rot)])


def wind_rule(observation, latent, history, updates, params, cmds):
    """Constant wind force on the domino at the FRONT of a running fan's beam.

    Conservative by construction: only the frontmost domino in the beam is
    ever pushed (that is all the data shows moving), so every downstream
    topple must be earned by a real contact in the engine.  The force tapers
    with cos(roll) - the projected frontal area of a tipping domino - which
    is what stops it once the domino is on the table.
    """
    del latent, history
    fans = [o for o in observation.data if o.type.name == "fan"]
    dominoes = [o for o in observation.data if o.type.name == "domino"]
    if not fans or not dominoes:
        return updates

    for fan in fans:
        if float(observation.get(fan, "is_on")) <= 0.5:
            continue
        axis = _wind_axis(observation, fan)
        perp = np.array([-axis[1], axis[0]])
        origin = np.array([
            float(observation.get(fan, "x")),
            float(observation.get(fan, "y")),
        ])

        best = None  # (along, domino, roll)
        for d in dominoes:
            if float(observation.get(d, "is_held")) > 0.5:
                continue
            v = np.array([
                float(observation.get(d, "x")),
                float(observation.get(d, "y")),
            ]) - origin
            along = float(v @ axis)
            lateral = abs(float(v @ perp))
            if along <= 0.0 or along > params["wind_range"]:
                continue
            if lateral > params["wind_beam_halfwidth"]:
                continue
            roll = abs(float(observation.get(d, "roll")))
            if best is None or along < best[0]:
                best = (along, d, roll)
        if best is None:
            continue
        _, d, roll = best
        taper = max(0.0, float(np.cos(roll)))
        f = params["wind_force"] * taper
        if f <= 0.0:
            continue
        cmds.apply_force(d, (float(axis[0] * f), float(axis[1] * f), 0.0))
    return updates


RESIDUAL_RULES: List[Any] = [wind_rule]
