# =============================================================================
# DECISION RECORD - pybullet_domino_blow residual simulator
# =============================================================================
# WHAT THE BASE SIM ALREADY CARRIES (do NOT re-model):
#   * Robot motion / IK / grasping; Pick and Place reproduce the recorded
#     action counts and the placed pose exactly (probe: Pick=51 actions,
#     Place=27 actions, placed x within 0.0003 of the recording).
#   * The switch -> fan.is_on toggle (TurnFanOn flips both switch_0.is_on and
#     fan_0.is_on, 24 actions into its 41-action rollout).
#   * All rigid-body physics of the domino: the gravity topple in traj 1
#     (the Pick at grasp_z_offset=0.08 knocked the block over) is reproduced
#     by the base sim on replay.
#
# WHAT IS MISSING FROM THE BASE SIM = WIND (the whole residual):
#   With fan_0.is_on == 1 the base sim leaves the domino perfectly static
#   (journal iter0: 8 consecutive Waits, zero motion at 4 different x).
#   In the recorded data the fan visibly drives the block.  This is an
#   exogenous influence the engine knows nothing about => diagnostic-ladder
#   case 2 => modelled with `cmds` (physics commands), NOT feature writes.
#
# PHYSICAL_PARAMS: DELIBERATELY NOT DECLARED.
#   Open-loop rollout sweep of all five registry parameters (lateral_friction,
#   mass, restitution, rolling_friction, spinning_friction), each alone across
#   its full box, was FLAT: SSE 62.2-65.3 against a 62.62 baseline (<5% spread,
#   far under the 3x consistency bar).  The open-loop divergence is entirely
#   the missing wind, not mis-set rigid-body physics.  Declaring any of them
#   would fit noise.
#
# TWO WIND REGIMES (both required; a single channel cannot produce both):
#   (A) STANDING / TIPPING block -> force + toppling moment.
#       Evidence: traj 0, fan on at step 102, the block goes
#       roll 0 -> pi/2 in 6 actions with dx per action rising then falling
#       (0.006, 0.013, 0.028, 0.041, 0.019, 0.008) - the signature of a
#       rotation about the leading bottom edge, i.e. engine physics driven
#       by an external push.  A pure COM force reproduces the topple only in
#       a knife-edge band (F=0.14 N: no topple at all; F=0.15 N: topple), so
#       the wind is modelled as force + an explicit moment (equivalent to a
#       force applied above the COM, which is what wind on a tall face is).
#       Torque axis = z_hat x wind_dir, so it tips the block DOWNWIND.
#   (B) FLAT (already fallen) block -> constant-velocity creep.
#       Evidence: in BOTH trajectories a fallen block drifts downwind at a
#       PERFECTLY constant 0.001215 m/action (5 significant figures, 22
#       consecutive steps in traj 1, 14 in traj 0, at two different x).
#       Constant velocity is incompatible with a Coulomb-friction + constant
#       force model (measured in-probe: F=0.2 N -> the flat block does not
#       move at all; F=0.45 N -> it runs away accelerating at 0.108 m/action;
#       there is no stable creep regime in between).  So regime B is a
#       kinematic velocity override.  It does NOT fight the engine: with the
#       small wind force the base sim leaves a flat block motionless.
#
# LATENT (partial observability):
#   `blow_steps` per fan - actions elapsed since is_on went high.  The task
#   statement says the fan blows for a LIMITED time, but no expiry is
#   observable in the data (wind still driving the block 22 actions after
#   switch-on in traj 1, 20 in traj 0).  The counter is therefore carried but
#   its cutoff is set beyond anything the data can see; it is NOT a ParamSpec
#   because there is no signal to fit it with.  Later cycles that observe an
#   expiry should turn _BLOW_DURATION_STEPS into a fitted parameter.
#
# RESIDUAL_FEATURES = domino x, z, roll: the pose dimensions the wind moves.
#   They are scored against observations but NOT overwritten at test time -
#   the engine moves them, which is exactly what the command channel wants.
#
# VALIDATION ANCHOR (real data, used to size placements):
#   recorded  Place(x=0.570) -> block lands at 0.57275 -> after the full
#   TurnFanOn rollout (17 wind actions) it is FLAT at x = 0.7012.
#   this model, same plan: x = 0.7042.  Topple+creep offset ~= +0.1285.
#
# FIT / VALIDATION (this cycle):
#   sim.fit  -> joint rollout system-ID, 3 of 4 motion segments explainable
#               (the dropped one is traj 1's Pick knocking the block over -
#               prolonged robot-object contact, not repeatable under replay).
#               rollout SSE 21.14 -> 0.0304 (99.9% reduction).
#   sim.residuals(rollout=True) at these inits: wind segments RMS 0.0138 and
#               0.00064, against a no-rule baseline of 0.0883 and 0.0223.
#   sim.refine(require_goal=True) on
#               Pick[0.06]; Place[0.5287,1.3728,0.55,1.5708]; TurnFanOn[0.1,0.11]
#               -> SUCCESS, goal atoms held.
#   continuous sim.run of the same plan -> Goal reached: True, block flat at
#               x = 0.6575 (region centre 0.6572); 5/5 trials reached the goal.
#   Placement margin: every x in [0.511, 0.547] (the whole AtLaunchPose band)
#               lands the block inside the region.  The one real-data
#               observation of a FAILING placement sits at |along| = 0.044,
#               2.4x the learned launch_tol of 0.018 - a comfortable margin,
#               not a knife edge.
#
# NOTE for later cycles: no Wait steps in the plan.  In this session's probe a
#   Wait runs to quiescence (>= 40 actions, and NEVER quiesces while the wind
#   creeps the block, hitting the 1000-action option cap), whereas every Wait
#   in the recorded real trajectories consumed exactly 1 action.  Because that
#   discrepancy is unresolved, the plan is built so no Wait is needed: the
#   topple and the whole useful slide happen inside the TurnFanOn rollout
#   itself (the switch flips 24 actions into its 41-action rollout, leaving 17
#   actions of wind).
# =============================================================================
import numpy as np

# Half-width of the "lying flat" band around |roll| = pi/2.  Regime switch,
# not a fitted quantity: the recorded roll settles to 1.5708 +- 0.0001 within
# one action of touching down, so anything in [0.002, 0.02] selects the same
# steps.  Kept a constant to keep the fitted parameter set minimal.
_FLAT_TOL = 0.01

# Actions of wind after switch-on before the fan cuts out.  NOT identified -
# see DECISION RECORD.  Set beyond the episode horizon so it is a no-op here.
_BLOW_DURATION_STEPS = 400


def _wind_dir(observation, fan):
    """Unit wind direction in the world plane, from the fan's own frame."""
    rot = observation.get(fan, "rot")
    ux, uy = float(np.cos(rot)), float(np.sin(rot))
    # `facing_side` selects which side of its housing the fan blows from.
    if observation.get(fan, "facing_side") > 0.5:
        ux, uy = -ux, -uy
    return ux, uy


def wind_rule(observation, latent, history, updates, params, cmds):
    """Wind from every running fan, on every free domino."""
    fans = [o for o in observation.data if o.type.name == "fan"]
    dominoes = [o for o in observation.data if o.type.name == "domino"]
    for fan in fans:
        fl = latent.setdefault(fan.name, {})
        if observation.get(fan, "is_on") <= 0.5:
            fl["blow_steps"] = 0
            continue
        elapsed = fl.get("blow_steps", 0)
        fl["blow_steps"] = elapsed + 1
        if elapsed >= _BLOW_DURATION_STEPS:
            continue
        ux, uy = _wind_dir(observation, fan)
        force = params["wind_force"]
        torque = params["topple_torque"]
        creep = params["creep_speed"]
        for dom in dominoes:
            if observation.get(dom, "is_held") > 0.5:
                continue
            roll = observation.get(dom, "roll")
            if abs(abs(roll) - 0.5 * np.pi) < _FLAT_TOL:
                # Regime B: already down - steady downwind creep.
                cmds.set_velocity(dom, linear=(creep * ux, creep * uy, 0.0))
            else:
                # Regime A: standing (or mid-fall) - push + topple moment.
                cmds.apply_force(dom, (force * ux, force * uy, 0.0))
                cmds.apply_torque(dom, (-torque * uy, torque * ux, 0.0))
    return updates


LATENT_INIT = {}

RESIDUAL_RULES = [wind_rule]

PARAM_SPECS = [
    # Net downwind force on a standing/tipping domino (N).
    ParamSpec("wind_force", 0.1504, lo=0.03, hi=0.35),
    # Toppling moment about (z_hat x wind_dir) (N*m); the part of the wind
    # load that acts above the COM on the tall exposed face.
    ParamSpec("topple_torque", 0.0235, lo=0.004, hi=0.08),
    # Steady creep speed of a fallen domino (m/s in engine units).
    ParamSpec("creep_speed", 0.0260, lo=0.004, hi=0.08),
    # --- predicate-only geometry (no SSE signal; see DECISION RECORD) ---
    # Downwind displacement from the standing launch pose to where the block
    # ends up at the end of a TurnFanOn rollout (topple arc + creep).
    # Measured: 0.7012 - 0.57275 = 0.1285 in the real data.
    ParamSpec("launch_offset", 0.1285, lo=0.05, hi=0.25),
    # Half-width of the acceptable launch band along the wind axis.
    ParamSpec("launch_tol", 0.018, lo=0.002, hi=0.05),
    # Half-width of the acceptable lateral (cross-wind) launch band.
    ParamSpec("launch_lat_tol", 0.06, lo=0.01, hi=0.1),
    # Max |roll| still counted as "standing".
    ParamSpec("upright_roll_max", 0.09, lo=0.02, hi=0.3),
    # Min |roll| counted as "knocked flat".
    ParamSpec("flat_roll_min", 1.40, lo=0.5, hi=1.55),
    # Max deviation of the broad face from square-on to the wind (rad).
    ParamSpec("face_to_wind_tol", 0.25, lo=0.05, hi=0.7),
]

RESIDUAL_FEATURES = {"domino": ["x", "z", "roll"]}
