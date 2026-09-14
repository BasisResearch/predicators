"""Residual dynamics for pybullet_domino_fan.

DECISION RECORD (cycle 3)
=================================
Observations (trajectories 0 and 1, both solved, reward 0.95):
  * Both episodes: Pick blue domino_1 -> Place it between green (domino_0,
    x=0.540) and purple target (domino_2, x=0.736) -> TurnFanOn -> cascade.
  * The instant `fan_0.is_on` flips 0 -> 1 (traj0 step 99, traj1 step 88) the
    GREEN start block begins to tip on the very next recorded state
    (roll 0 -> 0.05 -> 0.13 -> 0.19 -> 0.30 -> 0.51 -> 0.89 -> 1.26 -> 1.57)
    while sliding +x by ~6 cm.  No robot contact is involved.
  * The blue (x=0.638 / 0.662) and the purple (x=0.736) do NOT move while the
    green is still upright; they only start rolling one step AFTER the body in
    front of them has tipped into them (blue at green.roll~0.9, purple at
    blue.roll~0.4).  So the chain itself is ordinary contact physics.
  * Previous-cycle journal: with fan_0.is_on = 1 and several Wait options,
    the base sim moves NOTHING (green roll stays 1e-4).  So the wind is an
    exogenous influence the engine knows nothing about => diagnostic-ladder
    case 2 => model it with physics COMMANDS (constant force gated on is_on),
    not with feature overwrites and not with PHYSICAL_PARAMS.

Modeling choices:
  * ONE rule, `fan_wind`.  While a fan is on it emits a constant world-frame
    force along the fan's facing direction (cos(rot), sin(rot)) on the closest
    UPRIGHT, un-held domino that lies inside a learned downwind corridor
    (half-width + range).  Only the closest one: in the data the second and
    third dominoes are shielded by the first and never move under wind alone,
    and options.py documents that "the fan cuts out the moment the start block
    falls".  Modelling only the nearest exposed body is both consistent with
    the data and conservative for planning - the planner must bridge the gap
    with blue dominoes rather than hoping the wind reaches across it.
  * Everything downstream of the first tip (domino-domino collisions, sliding,
    settling) is left to the base sim's rigid-body engine.  No rule touches it.
  * No latent state is needed: the wind's driver (`fan.is_on`) is observable
    and the response is instantaneous (<1 step), so LATENT_INIT is empty.
  * PHYSICAL_PARAMS: NOT declared.  Decided from open-loop evidence, not from
    small per-step residuals.  With the wind rule riding,
    `sim.residuals(rollout=True, sweep_params='all')` gives baseline SSE 2.96
    and every swept alternative is worse or flat:
        lateral_friction  0.01/0.038/0.14/0.53/2.0 -> 73.9/69.6/68.3/1.81/69.3
        mass              0.005..1                 -> 68.3/18.1/7.2/68.7/68.7
        restitution       0..0.9                   -> 2.95/2.95/2.95/2.98/3.58
        rolling_friction  0/0.025/0.05/0.075/0.1   -> 3.00/70.6/70.6/70.6/65.8
        spinning_friction 0.01..2                  -> 1.79/2.97/2.97/1.78/1.86
    Nothing clears the 3x consistency bar over the registry baseline (the best,
    lateral_friction 0.53, is 1.6x and is essentially the baseline value), so
    the shipped rigid-body physics is already calibrated for this data and
    declaring a parameter would fit noise.

Calibration + validation (belief probe, task 0):
  * wind_force calibrated by replaying each recorded pre-fan scene
    (`sim.reset(mods=...)` staging the blue at its recorded pre-fan pose) and
    matching the settled poses.  At 0.15-0.18 N every body lands within 1 cm
    of the recording: traj0 (blue @0.6375) predicted g/b/p x = 0.598/0.722/
    0.836 vs recorded 0.586/0.722/0.835, rolls 1.45/1.44/1.57 vs 1.46/1.44/
    1.57; traj1 (blue @0.662) predicted 0.571/0.769/0.827 vs recorded
    0.571/0.767/0.827.  1.2 N (the first guess) shot the green 30 cm downrange
    and produced a fantasy solve with no bridge at all - the calibration is
    what makes the model refuse that.
  * Sanity check in the other direction: with NO bridge the wind topples only
    the green (it ends at x=0.636) and the purple stays upright -> goal not
    reached.  The model therefore reproduces the env's actual requirement that
    a blue must bridge the 0.196 m green->purple gap, and 0.95 (one blue
    consumed) is the best score physically available on this task.
  * Cascade-geometry sweeps used to fit the predicate cutoffs in
    predicates.py: centre spacings 0.06-0.14 relay, 0.16 stalls; lateral
    offsets 0.00-0.06 relay, 0.12 does not; bridge yaw within ~+-0.77 rad of
    square relays, edge-on (yaw 0.0/0.4) stalls and even jams the green.  All
    eight corners of the region DomBridges accepts were re-simulated and every
    one topples the purple target.
  * Known harness caveat this cycle: `evaluate_predicate_quality` loads and
    scores predicates.py fine, but the run_python BeliefProbe still parses
    sketches against the Holding-only allowlist, so learned-predicate subgoal
    annotations are silently dropped there and `sim.refine` cannot be used as
    the goal gate (it then places the blue anywhere and misses the goal).
    Validation was therefore done with fully-parameterised `sim.run` plans
    plus the corner sweeps above; the reference plan below runs end-to-end
    with `Goal reached: True`.

Reference plan (validated in the calibrated model, matches both recordings):
    Pick(robot, domino_1)[0.05]
    Place(robot)[0.638, 1.38653, 0.57, 1.5708]
    TurnFanOn(robot, fan_0)[0.1, 0.11]
    Wait(robot) x2-3
"""

from typing import Any, Dict, List

UPRIGHT_ROLL = 0.25          # |roll| below this counts as still standing

# Structural (NOT fitted) wind-corridor constants.  The data contains exactly
# one body in the wind (the green start block, 0.22 m downwind, dead on the
# fan axis), so neither the corridor's length nor its width is identifiable -
# sim.fit reported contraction ~1 for both when they were ParamSpecs.  Rather
# than let MCMC wander them to an arbitrary value, they are pinned to
# deliberately CONSERVATIVE values: just past the green block and about one
# domino width.  Erring short is safe (the planner then has to bridge the gap
# with blue dominoes, which is what the data shows works); erring long would
# invent a wind that topples the target directly and produce plans that fail
# for real.
WIND_RANGE = 0.26            # m downwind reach of the jet
WIND_HALF_WIDTH = 0.09       # m half-width of the jet corridor


def _fan_dir(state, fan):
    rot = float(state.get(fan, "rot"))
    return np.array([np.cos(rot), np.sin(rot)])


def fan_wind(observation, latent, history, updates, params, cmds):
    """Constant wind force from every switched-on fan on the first
    exposed upright domino downwind of it."""
    fans = [o for o in observation.data if o.type.name == "fan"]
    dominoes = [o for o in observation.data if o.type.name == "domino"]
    if not fans or not dominoes:
        return updates
    for fan in fans:
        if float(observation.get(fan, "is_on")) <= 0.5:
            continue
        d = _fan_dir(observation, fan)
        perp = np.array([-d[1], d[0]])
        origin = np.array([float(observation.get(fan, "x")),
                           float(observation.get(fan, "y"))])
        best, best_along = None, None
        for dom in dominoes:
            if float(observation.get(dom, "is_held")) > 0.5:
                continue
            if abs(float(observation.get(dom, "roll"))) > UPRIGHT_ROLL:
                continue          # already toppling: no longer a sail
            rel = np.array([float(observation.get(dom, "x")),
                            float(observation.get(dom, "y"))]) - origin
            along = float(rel @ d)
            lateral = abs(float(rel @ perp))
            if along <= 0.0 or along > WIND_RANGE:
                continue
            if lateral > WIND_HALF_WIDTH:
                continue
            if best_along is None or along < best_along:
                best, best_along = dom, along
        if best is not None:
            f = params["wind_force"]
            cmds.apply_force(best, (float(f * d[0]), float(f * d[1]), 0.0))
    return updates


RESIDUAL_RULES = [fan_wind]

PARAM_SPECS = [
    # Newtons, applied at the body COM (0.075 m above the table) every physics
    # substep while the fan is on.  Static topple threshold for a 0.1 kg,
    # 0.15 x 0.014 m domino is ~0.09 N, so the box brackets "just tips it" to
    # "shoves it hard"; 0.15 already reproduced both recorded cascades to
    # ~1 cm on every body (see decision record).
    ParamSpec("wind_force", 0.175, lo=0.05, hi=0.45),
    # Predicate-only geometry (no SSE signal - it never enters a rule, so it
    # stays at init_value): the largest centre-to-centre spacing at which a
    # toppling domino still reliably knocks over the next one.  Bodies are
    # 0.15 m tall and ~0.014 m deep, so contact is geometrically possible out
    # to ~0.16 m; 0.13 keeps the strike high on the neighbour's face.
    ParamSpec("bridge_max_gap", 0.13, lo=0.05, hi=0.17),
    # Half-width of the corridor a bridging domino must sit in, measured
    # perpendicular to the source->target line.
    ParamSpec("bridge_max_lateral", 0.035, lo=0.005, hi=0.08),
    # |roll| above which a domino counts as toppled / below which it counts
    # as still standing.
    # |a . u| cap, where a = (cos yaw, sin yaw) is the domino's width axis
    # and u the cascade direction: how squarely the bridge domino must face
    # the oncoming domino.  Probe sweep at the demo layout: yaw giving
    # |a.u| = 0.45 / 0.70 relay fine, 0.92 / 1.00 (edge-on) stall the chain,
    # so 0.5 sits well inside the working bucket.
    ParamSpec("bridge_yaw_align", 0.5, lo=0.1, hi=0.85),
    ParamSpec("toppled_roll", 0.7, lo=0.3, hi=1.4),
    ParamSpec("upright_roll", 0.2, lo=0.05, hi=0.4),
]

LATENT_INIT: Dict[str, Any] = {}

# The wind moves the first domino's pose; the engine carries it and the rest
# of the chain.  These are scored against observations, never overwritten.
RESIDUAL_FEATURES = {"domino": ["x", "y", "z", "roll"]}
