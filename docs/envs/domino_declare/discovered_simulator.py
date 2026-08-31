# =====================================================================
# DECISION RECORD - pybullet_domino_declare (cycle 3, learn)
# =====================================================================
# WHAT THE BASE SIM ALREADY DOES (verified, do not re-implement):
#   * All rigid-body motion: arm IK, grasping, transport, release, and
#     the *entire* domino-on-domino cascade. Cycle-1/2 probes showed
#     that seeding `mods={'domino_0': {'roll': 0.3}}` makes the base sim
#     topple green -> blue -> purple all by itself. Contact physics is
#     present and good.
#
# WHAT IS MISSING (measured this cycle, probe with an empty rule set):
#   Running `DeclareFinished` + `Wait` from task-0 init in the base sim
#   leaves fan_0.is_on = 0.0 and every domino untouched. In the RECORDED
#   env trajectories the exact same action sequence flips
#   fan_0.is_on 0->1 and switch_0.is_on 0->1 one step after the first
#   DeclareFinished action (traj0 t=74, traj1 t=80) and then the green
#   start block tips over: roll 0 -> .047 -> .128 -> .193 -> .294 (and
#   x drifts +x, toward the target) with NO contact and NO robot nearby.
#   => An exogenous influence the engine knows nothing about: the
#      declaration switches a FAN on and its wind topples the start
#      block. Diagnostic ladder case 2 -> model with force `cmds`,
#      gated on an observable condition; plus the device-state flip on
#      the feature-update channel.
#
# THE TRIGGER IS NOT DIRECTLY OBSERVABLE (partial observability):
#   Rules never see actions, and `DeclareFinished` is - to the base sim
#   - indistinguishable from a 2-step Wait (probe diff: only sub-mm
#   arm/finger drift). So "has the agent declared?" is a HIDDEN state.
#   It is inferred from an observable signature and carried in `latent`
#   (Pattern A: counter + threshold).
#   Signature = "robot holds nothing AND the arm has stopped moving".
#   Threshold-fitting protocol, traj0, max |delta| over robot
#   (x,y,z,roll,tilt,wrist), buckets over the is_held==0 steps:
#       arm-moving bucket  (t=1..20, 60..72): min 0.0086, typ 0.02-1.0
#       arm-still  bucket  (t=73..78, declare/wait): max 0.000256
#   -> two clean clusters separated by a factor of ~34. Cut at 0.002.
#   Held steps (t=21..59) are excluded by the is_held gate, which is
#   what makes the slow 0.0007/step transport phase harmless.
#   With `declare_delay`=1 the latch fires on state[73] and the fan
#   reads on at state[74] - exactly the recorded flip step.
#
# WHAT THE RULES OWN (RESIDUAL_FEATURES):
#   fan.is_on, switch.is_on   - written directly (base sim never sets
#                               them; they are the device readout of
#                               the latent `declared` flag).
#   domino.x/z/roll           - moved by the wind FORCE through the
#                               engine, so they are SCORED but not
#                               overwritten. Listing them is what gives
#                               the wind magnitude an SSE signal.
#   (domino.y / yaw deliberately omitted: the wind is axial, those
#    carry only base-sim replay noise from the Pick/Place phase.)
#
# WIND MODEL - simplest hypothesis that fits:
#   Constant world-frame force along the fan axis (cos rot, sin rot),
#   applied every step the fan is on, to ONE domino: the "start block",
#   latched at declare time as the nearest not-yet-toppled domino
#   inside a narrow beam in front of the fan (that is domino_0, the
#   green one, at lateral offset 0.000; the blue sits 0.017-0.025 off
#   axis and never moves under wind in either recorded trajectory).
#   The force stops once the start block is down - matching the env
#   note in reference/options.py that "the fan cuts out the moment the
#   start block is down (a fallen domino is out of the airstream)".
#   Everything after that - green hitting blue hitting purple - is left
#   entirely to the base sim's contact physics.
#
# PHYSICAL_PARAMS: not declared (see the sweep in the session log).
#   The base sim reproduces the recorded rigid-body motion; the defect
#   was a missing mechanism, not mis-set physics.
# =====================================================================

# --- object/feature helpers ------------------------------------------

_ARM_FEATS = ("x", "y", "z", "roll", "tilt", "wrist")


def _by_type(state, tname):
    return [o for o in state.data if o.type.name == tname]


def _prev_obs(observation, history):
    """Most recent *earlier* observation, or None at the first step."""
    if not history:
        return None
    for entry in reversed(history):
        st = entry[0] if isinstance(entry, (tuple, list)) else entry
        if st is not observation:
            return st
    return None


def _is_toppled(state, dom, params):
    return abs(state.get(dom, "roll")) > params["topple_roll"]


def _fan_axis(state, fan):
    rot = state.get(fan, "rot")
    return float(np.cos(rot)), float(np.sin(rot))


def _fan_anchor(state, fan, params):
    """Outlet point of the fan: recorded origin + a LOCAL-frame offset
    rotated by the fan's `rot`. Shared with predicates.InAirstream."""
    ux, uy = _fan_axis(state, fan)
    ox = params["fan_local_dx"] * ux - params["fan_local_dy"] * uy
    oy = params["fan_local_dx"] * uy + params["fan_local_dy"] * ux
    return state.get(fan, "x") + ox, state.get(fan, "y") + oy


def _beam_pick(state, fan, dominoes, params):
    """Nearest upright, un-held domino inside the fan's beam."""
    ux, uy = _fan_axis(state, fan)
    fx, fy = _fan_anchor(state, fan, params)
    best, best_along = None, None
    for d in dominoes:
        if state.get(d, "is_held") > 0.5:
            continue
        if _is_toppled(state, d, params):
            continue
        dx = state.get(d, "x") - fx
        dy = state.get(d, "y") - fy
        along = dx * ux + dy * uy
        lateral = -dx * uy + dy * ux
        if along <= 0.0:
            continue
        if abs(lateral) > params["beam_halfwidth"]:
            continue
        if best_along is None or along < best_along:
            best, best_along = d, along
    return best


# --- rule 1: infer the hidden "declared" flag, drive the devices -----


def declare_rule(observation, latent, history, updates, params):
    """Latent Pattern A: count steps of 'nothing held + arm stopped';
    latch `declared` once the count passes the learned delay, and read
    the latch out onto every fan / switch `is_on`."""
    robots = _by_type(observation, "robot")
    dominoes = _by_type(observation, "domino")

    held = any(observation.get(d, "is_held") > 0.5 for d in dominoes)
    prev = _prev_obs(observation, history)
    still = False
    if prev is not None and not held:
        try:
            drift = max(
                abs(observation.get(r, f) - prev.get(r, f))
                for r in robots for f in _ARM_FEATS)
            still = drift < params["still_eps"]
        except Exception:  # pylint: disable=broad-except
            still = False

    latent["still"] = (latent.get("still", 0.0) + 1.0) if still else 0.0
    if latent.get("declared", 0.0) < 0.5:
        if latent["still"] >= params["declare_delay"]:
            latent["declared"] = 1.0

    # Also honour a directly observed device flip (recorded env data
    # sets is_on itself); never un-latch.
    for fan in _by_type(observation, "fan"):
        if observation.get(fan, "is_on") > 0.5:
            latent["declared"] = 1.0

    on = 1.0 if latent.get("declared", 0.0) > 0.5 else 0.0
    for fan in _by_type(observation, "fan"):
        updates.setdefault(fan, {})["is_on"] = on
    for sw in _by_type(observation, "switch"):
        updates.setdefault(sw, {})["is_on"] = on
    return updates


# --- rule 2: the wind itself (physics-command channel) ---------------


def wind_rule(observation, latent, history, updates, params, cmds):
    """Constant axial force on the latched start block while the fan is
    on and that block is still standing."""
    del history
    if latent.get("declared", 0.0) < 0.5:
        return updates
    dominoes = _by_type(observation, "domino")
    if not dominoes:
        return updates
    by_name = {d.name: d for d in dominoes}

    for fan in _by_type(observation, "fan"):
        key = "start_" + fan.name
        target = by_name.get(latent.get(key))
        if target is None:
            target = _beam_pick(observation, fan, dominoes, params)
            if target is None:
                continue
            latent[key] = target.name
        # The fan cuts out once the start block is down / out of the
        # airstream; the chain then coasts on contact alone.
        if _is_toppled(observation, target, params):
            continue
        if observation.get(target, "is_held") > 0.5:
            continue
        ux, uy = _fan_axis(observation, fan)
        f = params["wind_force"]
        cmds.apply_force(target, (ux * f, uy * f, 0.0))
    return updates


RESIDUAL_RULES = [declare_rule, wind_rule]

PARAM_SPECS = [
    # "arm has stopped" cut: still bucket <= 2.6e-4, moving bucket
    # >= 8.6e-3 (traj0, is_held==0 steps).
    ParamSpec("still_eps", 0.002, lo=0.0005, hi=0.006),
    # Steps of stillness before the declaration is inferred.
    ParamSpec("declare_delay", 1.0, lo=1.0, hi=6.0),
    # Newtons of wind on the start block.
    ParamSpec("wind_force", 0.20, lo=0.0, hi=1.5),
    # Half-width of the airstream, metres, about the fan axis.
    ParamSpec("beam_halfwidth", 0.04, lo=0.005, hi=0.15),
    # |roll| beyond which a domino counts as down / out of the stream.
    ParamSpec("topple_roll", 0.60, lo=0.2, hi=1.4),
    # Fan outlet offset in the fan's LOCAL frame (shared with the
    # InAirstream predicate). Init 0: the render shows the housing
    # centred on its recorded origin, so the fit is free to confirm it.
    ParamSpec("fan_local_dx", 0.0, lo=-0.2, hi=0.2),
    ParamSpec("fan_local_dy", 0.0, lo=-0.2, hi=0.2),
    # Predicate-only (no SSE signal -> stays at init_value): the
    # centre-to-centre gap a toppling domino can still bridge.
    # Cycle-1/2 reach sweeps: 0.130 m links propagate, 0.140 m do not.
    ParamSpec("chain_gap_max", 0.125, lo=0.05, hi=0.20),
]

LATENT_INIT = {"declared": 0.0, "still": 0.0}

RESIDUAL_FEATURES = {
    "fan": ["is_on"],
    "switch": ["is_on"],
    "domino": ["x", "z", "roll"],
}
