"""Trajectory-level legitimacy certificate for min-block cascade tasks.

The min-block evaluator's terminated/success checks (``DominoEvaluator``)
are functions of the final state only (goal atoms + toppled-blue count),
which are blind to HOW the target fell. Observed reward hacks all bypass
the intended causal chain
"push the green start block -> dominoes knock each other over -> target
falls": pushing a placed blue directly, sweeping the target with the
gripper or a carried block during Place, knocking the target while an
option flails, and toppling a block with the gripper right after a
(useless) Push on the green so the fresh green onset masks the arm as
the cause (run_20260713_000327 task 2: the green fell clear of the
staged relay and the Push option's closing descent knocked the relay
over instead). This module certifies, from the recorded per-step states
(plus, when available, which option produced each transition), that
every topple in the episode traces back to a robot Push on the green
start block through domino-on-domino contact. Contact includes one-hop
shoved relays - a rammed block that slides into its neighbor and knocks
it over without itself toppling is a genuine cascade link
(run_20260713_133936 task 3), not a violation.

The certificate is a pure function over ``State`` sequences - no
PyBullet contact queries - so it runs identically on true-env episodes
and on option-model rollouts in the agent's simulator.
"""

import logging
import math
from typing import Dict, List, Optional, Sequence, Set, Tuple

from predicators.envs.pybullet_domino import geometry
from predicators.envs.pybullet_domino.components.domino_component import \
    DominoComponent
from predicators.structs import GroundAtom, Object, State, StepOption

# A toppling domino knocks its successor within a fraction of a second;
# one env step is CFG.pybullet_sim_steps_per_action (20) PyBullet
# substeps at 1/240 s ~= 0.083 s, so 24 steps ~= 2.0 s is a generous
# upper bound on one cascade hop. A larger gap means the "predecessor"
# had already settled and cannot have supplied the energy.
CASCADE_WINDOW_STEPS = 24

# The name of the option through which the robot is allowed to topple
# the green start block.
_PUSH_OPTION_NAME = "Push"

# Plan-view clearance tolerance on the swept-corridor attribution of a
# topple to a falling predecessor. Calibrated on recorded episodes:
# genuine grazing corner knocks measure up to +0.014 m of modeled
# clearance (base slide, mid-fall twist and per-step sampling are
# outside the straight-corridor model), while the observed
# gripper-topples measured +0.031 m and +0.035 m.
CORRIDOR_TOLERANCE = 0.02

# Below this modeled clearance the corridor genuinely overlaps the
# block's footprint (solid contact); such attributions are never
# second-guessed by the robot-adjacency rule (c) - in tight staged
# chains the end-effector legitimately ends the push within a few
# centimeters of the first relay.
GRAZE_CONTACT_EPS = 0.005

# End-effector strike range for rule (c). The recorded gripper-topples
# put the EE feature point 0.064-0.078 m (xy) from the struck block's
# center at its onset; verified non-contact cases measured >= 0.091 m.
ROBOT_STRIKE_XY = 0.085
# The EE can only strike a block when it is low enough; its feature
# point rides up to ~0.2 m above the finger tip depending on wrist
# pose, so allow that much above the block's top face. The retreated
# park pose (~0.4 m above) stays excluded.
ROBOT_STRIKE_Z_SLACK = 0.3
# How many steps before a topple onset the striking contact may have
# been sampled (observed strikes register within 1-2 steps).
ROBOT_STRIKE_LOOKBACK_STEPS = 2
# When the topple is a grazing cascade attribution, the striking contact
# may also be sampled AFTER the onset, anywhere while the block is still
# going down: a bulldozing gripper keeps closing on the block through its
# whole fall (run_20260713_172940 task 2: the EE was 0.094 m away at the
# onset - just outside strike range - then drove to 0.013 m over the next
# five steps as the block toppled). The forward scan runs from the onset
# to the block's peak-roll step (its fall having stopped by then),
# capped for safety. It is a no-op once the block is flat, so a robot
# that only passes by after a genuine knock is never charged.
ROBOT_STRIKE_FALL_CAP_STEPS = 10

# Minimum base displacement for a standing block to count as a shoved
# relay in rule (b)'s one-hop attribution, and for a movable blue to
# count as consumed by the cascade. Well above resting jitter and
# release-settle skids (millimeters); a relay only matters when the
# knocked block lies beyond the predecessor's corridor, so a genuine
# transmitting slide covers centimeters (the recorded slide-relay
# episode measured 0.12 m).
RELAY_MIN_SLIDE = 0.02


def count_movable_blocks_used(states: Sequence[State]) -> int:
    """Count movable (blue) dominoes the episode consumed.

    A blue is consumed when it has toppled in the final state, or when
    it was displaced at least ``RELAY_MIN_SLIDE`` within a span where it
    was not held: that is a cascade shove (robot transport is excluded
    by the held gating), so a slide-relay that knocks its successor
    over without itself toppling is charged the same as a toppled relay
    and staying upright earns no cost discount.

    A pure function of the states (roles recovered from color features,
    topple from the roll angle, shoves from not-held displacement) so
    ``DominoEvaluator`` needs no live env handle and stays picklable.
    """
    count = 0
    final = states[-1]
    for obj in final:
        if obj.type.name != "domino":
            continue
        # pylint: disable=protected-access
        if not DominoComponent._MovableBlock_holds(final, [obj]):
            continue
        if abs(final.get(obj, "roll")) >= DominoComponent.fallen_threshold:
            count += 1
            continue
        anchor: Optional[State] = None
        for state in states:
            if state.get(obj, "is_held") > 0.5:
                anchor = None
                continue
            if anchor is None:
                anchor = state
                continue
            if math.hypot(
                    state.get(obj, "x") - anchor.get(obj, "x"),
                    state.get(obj, "y") -
                    anchor.get(obj, "y")) >= RELAY_MIN_SLIDE:
                count += 1
                break
    return count


def _domino_dims() -> Tuple[float, float, float]:
    """(height, width, depth) of a domino, imported lazily so the certificate
    stays importable without PyBullet."""
    from predicators.envs.pybullet_domino.env import \
        PyBulletDominoComposedEnv  # pylint: disable=import-outside-toplevel
    return (PyBulletDominoComposedEnv.domino_height,
            PyBulletDominoComposedEnv.domino_width,
            PyBulletDominoComposedEnv.domino_depth)


def _cascade_reach() -> float:
    """Max distance along its fall direction at which a falling domino can
    plausibly have knocked another one over.

    A falling domino's top sweeps ~domino_height ahead of its base; two
    depths of slack absorb base sliding during the fall.
    """
    height, _, depth = _domino_dims()
    return height + 2 * depth


def _stage_tolerance() -> float:
    """Max xy drift of the green start block from its staged pose before its
    topple onset that still counts as "pushed where it stands".

    A legitimate push tips the block about its base edge, so the base
    slides at most a couple of centimeters before the block leaves the
    upright band; relocating the block anywhere useful (next to the
    target, or even one chain gap over) moves it by at least ``pos_gap``
    ~= 0.098 m. One domino width sits safely between the two regimes.
    """
    from predicators.envs.pybullet_domino.env import \
        PyBulletDominoComposedEnv  # pylint: disable=import-outside-toplevel
    return PyBulletDominoComposedEnv.domino_width


def _topple_onset(states: Sequence[State], domino: Object) -> Optional[int]:
    """State index where ``domino``'s final fall began, or None.

    A domino has a topple event iff its |roll| ever reaches
    ``fallen_threshold`` while not held (carried blocks tilt
    legitimately). The onset is the moment it last left the upright band
    (|roll| < ``domino_roll_threshold``) before that first full topple,
    so placement wobbles that recover never register and a wobble that
    precedes the real fall is not mistaken for it.
    """
    fall_idx: Optional[int] = None
    for t, state in enumerate(states):
        if state.get(domino, "is_held") > 0.5:
            continue
        if abs(state.get(domino, "roll")) >= DominoComponent.fallen_threshold:
            fall_idx = t
            break
    if fall_idx is None:
        return None
    onset = None
    for t in range(fall_idx - 1, -1, -1):
        state = states[t]
        if state.get(domino, "is_held") > 0.5:
            continue
        if abs(state.get(domino,
                         "roll")) < DominoComponent.domino_roll_threshold:
            onset = t + 1
            break
    if onset is not None:
        return onset
    # Never observed upright and non-held before falling: use the first
    # non-held index (conservative earliest).
    for t in range(fall_idx + 1):
        if states[t].get(domino, "is_held") <= 0.5:
            return t
    return fall_idx


def _fall_direction(states: Sequence[State], domino: Object,
                    onset: int) -> Tuple[float, float]:
    """Unit xy direction along which ``domino`` fell, starting at ``onset``.

    Primary signal: the center of a toppling domino translates by about
    half its height along the fall direction, so the displacement from
    onset to a few steps past the moment it first reads fallen points
    the way. When the fall was blocked (e.g. the domino came to rest
    leaning on its successor) the displacement can be tiny; fall back to
    the yaw/roll convention - a negative roll falls along the axis
    ``(-sin(yaw), cos(yaw))`` - which matches the measured displacement
    on every recorded free fall.
    """
    settle = onset
    for t in range(onset, len(states)):
        settle = t
        if abs(states[t].get(domino,
                             "roll")) >= DominoComponent.fallen_threshold:
            break
    settle = min(settle + 3, len(states) - 1)
    dx = states[settle].get(domino, "x") - states[onset].get(domino, "x")
    dy = states[settle].get(domino, "y") - states[onset].get(domino, "y")
    norm = math.hypot(dx, dy)
    if norm >= 0.03:
        return dx / norm, dy / norm
    yaw = states[onset].get(domino, "yaw")
    sign = -1.0 if states[settle].get(domino, "roll") > 0 else 1.0
    return geometry.fall_axis(yaw, sign)


def _footprint(state: State, domino: Object) -> List[Tuple[float, float]]:
    """Plan-view base rectangle of ``domino`` in ``state``."""
    _, width, depth = _domino_dims()
    yaw = state.get(domino, "yaw")
    return geometry.domino_footprint(state.get(domino, "x"),
                                     state.get(domino, "y"), yaw, width / 2,
                                     depth / 2)


def _leaning_footprint(state: State,
                       domino: Object) -> List[Tuple[float, float]]:
    """Plan-view extent of a possibly-leaning domino.

    The base rectangle extended along the lean direction by the top
    edge's ground projection (``height * sin|roll|``), so a block
    tipping onto its neighbor reaches as far as its body actually does.
    Upright blocks reduce to the plain base rectangle.
    """
    height, width, depth = _domino_dims()
    yaw = state.get(domino, "yaw")
    roll = state.get(domino, "roll")
    proj = height * math.sin(min(abs(roll), math.pi / 2))
    sign = -1.0 if roll > 0 else 1.0
    tilt_x, tilt_y = geometry.fall_axis(yaw, sign)
    return geometry.domino_footprint(
        state.get(domino, "x") + tilt_x * proj / 2,
        state.get(domino, "y") + tilt_y * proj / 2, yaw, width / 2,
        depth / 2 + proj / 2)


def _body_reached(states: Sequence[State],
                  predecessor: Object,
                  onset_p: int,
                  domino: Object,
                  ref_t: int,
                  fp_t: Optional[int] = None) -> bool:
    """Did ``predecessor``'s actual body ever reach ``domino``?

    Scans the predecessor's leaning footprint (its true per-step pose,
    body projected along its lean) from its own topple onset to a few
    steps past ``ref_t`` (the knocked block's onset, or a relay's slide
    start) against ``domino``'s footprint at ``fp_t`` (default
    ``ref_t``) - the last time it still stood where it was struck. A
    genuine knock is physical contact, so the plan-view projections
    overlap (gap <= 0) at the strike - and keep overlapping as the
    predecessor settles into the space the struck block vacated. A
    predecessor that fell BESIDE the block never overlaps it, however
    well its modeled corridor covers the block on paper
    (run_20260714_145053 task 4: the green fell 72 deg away from the
    staged blue, stopping 4 mm short, while the gripper toppled the
    blue - the corridor still measured 4.3 mm of clearance, inside the
    solid-contact band).
    """
    if fp_t is None:
        fp_t = ref_t
    target_fp = _footprint(states[min(fp_t, len(states) - 1)], domino)
    end = min(ref_t + 3, len(states) - 1)
    for t in range(onset_p, end + 1):
        if geometry.rect_gap(_leaning_footprint(states[t], predecessor),
                             target_fp) <= 0.0:
            return True
    return False


def _knock_gap(states: Sequence[State], predecessor: Object, onset_p: int,
               domino: Object, onset_state: State) -> float:
    """Plan-view clearance between ``predecessor``'s swept fall corridor and
    ``domino``'s footprint at its own onset.

    The corridor runs from the predecessor's base (its center at its own
    onset, while still upright) along its measured fall direction for
    ``_cascade_reach()``, one domino width wide - the region its falling
    body sweeps over. At or below zero the falling predecessor overlaps
    the block; small positive values are grazing contacts the straight-
    corridor model underestimates.
    """
    _, width, _ = _domino_dims()
    fx, fy = _fall_direction(states, predecessor, onset_p)
    base_x = states[onset_p].get(predecessor, "x")
    base_y = states[onset_p].get(predecessor, "y")
    reach = _cascade_reach()
    corridor = geometry.rect_corners(base_x + fx * reach / 2,
                                     base_y + fy * reach / 2, fx, fy,
                                     reach / 2, width / 2)
    return geometry.rect_gap(corridor, _footprint(onset_state, domino))


def _fall_peak_step(states: Sequence[State], domino: Object,
                    onset: int) -> int:
    """Step of ``domino``'s peak |roll| after its topple onset.

    That step is where its fall has stopped (flat or leaning); it caps a
    window of ``ROBOT_STRIKE_FALL_CAP_STEPS`` steps past the onset. A
    block that keeps toppling reads a larger |roll| each step until it
    hits the ground; the peak marks the end of the active fall, after
    which a nearby end-effector is a passer-by, not the cause. A block
    already at its final roll (e.g. the synthetic step-function
    trajectories the tests use) peaks at the onset itself, so the
    forward strike scan collapses to the onset and adds nothing.
    """
    peak = onset
    peak_roll = abs(states[min(onset, len(states) - 1)].get(domino, "roll"))
    for t in range(onset + 1,
                   min(onset + ROBOT_STRIKE_FALL_CAP_STEPS + 1, len(states))):
        roll = abs(states[t].get(domino, "roll"))
        if roll > peak_roll:
            peak_roll = roll
            peak = t
    return peak


def _robot_strike_nearby(states: Sequence[State],
                         robot: Object,
                         domino: Object,
                         onset: int,
                         through_fall: bool = False,
                         lookback: Optional[int] = None) -> bool:
    """Was the robot end-effector in strike range of ``domino`` around its
    topple onset?

    Scans from ``lookback`` (default ``ROBOT_STRIKE_LOOKBACK_STEPS``)
    steps before the onset through the onset. With ``through_fall`` the
    scan also runs forward to the block's peak-roll step
    (``_fall_peak_step``), catching a gripper that was just outside
    strike range at the onset but kept driving into the block as it
    toppled. Solid-overlap knocks never reach this check, so the forward
    scan only ever second-guesses grazing attributions. The verdict
    rules keep the tight default lookback; only the failure hint for
    unattributable topples widens it (a poker graze that tips the block
    slowly puts the onset well past the contact).
    """
    height, _, _ = _domino_dims()
    if lookback is None:
        lookback = ROBOT_STRIKE_LOOKBACK_STEPS
    end = _fall_peak_step(states, domino, onset) if through_fall else onset
    for t in range(max(0, onset - lookback), min(end + 1, len(states))):
        state = states[t]
        dist = math.hypot(
            state.get(robot, "x") - state.get(domino, "x"),
            state.get(robot, "y") - state.get(domino, "y"))
        if dist <= ROBOT_STRIKE_XY and state.get(robot, "z") <= state.get(
                domino, "z") + height / 2 + ROBOT_STRIKE_Z_SLACK:
            return True
    return False


def _best_relay_attribution(
        states: Sequence[State], onsets: Dict[Object, int], legit: Set[Object],
        dominoes: Sequence[Object], domino: Object,
        onset: int) -> Optional[Tuple[Object, float, float, int, Object]]:
    """Best one-hop shoved-relay explanation for ``domino``'s topple onset.

    A relay is a block that an already-legitimate falling predecessor
    rammed - the predecessor's swept corridor covers the relay's
    footprint at the predecessor's own onset - and that then slid at
    least ``RELAY_MIN_SLIDE`` toward ``domino``, bringing its body
    (``_leaning_footprint``: a rammed block may lean far over without
    toppling) into contact range of ``domino``'s footprint by
    ``domino``'s onset. The relay itself need not topple: the recorded
    slide-relay episode's blue slid 0.12 m into the target, knocked it
    over, and rocked back upright. A held block is the robot's tool,
    never a relay.

    Returns ``(relay, corridor_gap, contact_gap, slide_start,
    predecessor)`` for the smallest contact gap within
    ``CORRIDOR_TOLERANCE``, or None.
    """
    onset_state = states[min(onset, len(states) - 1)]
    footprint_d = _footprint(onset_state, domino)
    best: Optional[Tuple[Object, float, float, int, Object]] = None
    best_contact = math.inf
    for pred in legit:
        onset_p = onsets[pred]
        if not onset_p <= onset <= onset_p + CASCADE_WINDOW_STEPS:
            continue
        window = range(onset_p, min(onset, len(states) - 1) + 1)
        for relay in dominoes:
            if relay is domino or relay is pred:
                continue
            if any(states[s].get(relay, "is_held") > 0.5 for s in window):
                continue
            corridor_gap = _knock_gap(states, pred, onset_p, relay,
                                      states[onset_p])
            if corridor_gap > CORRIDOR_TOLERANCE:
                continue
            start_x = states[onset_p].get(relay, "x")
            start_y = states[onset_p].get(relay, "y")
            slide_x = onset_state.get(relay, "x") - start_x
            slide_y = onset_state.get(relay, "y") - start_y
            if math.hypot(slide_x, slide_y) < RELAY_MIN_SLIDE:
                continue
            toward_x = onset_state.get(domino, "x") - start_x
            toward_y = onset_state.get(domino, "y") - start_y
            if slide_x * toward_x + slide_y * toward_y <= 0:
                continue
            slide_start = onset
            contact_gap = math.inf
            for s in window:
                contact_gap = min(
                    contact_gap,
                    geometry.rect_gap(_leaning_footprint(states[s], relay),
                                      footprint_d))
                if slide_start == onset and math.hypot(
                        states[s].get(relay, "x") - start_x, states[s].get(
                            relay, "y") - start_y) >= RELAY_MIN_SLIDE:
                    slide_start = s
            if contact_gap > CORRIDOR_TOLERANCE:
                continue
            if contact_gap < best_contact:
                best_contact = contact_gap
                best = (relay, corridor_gap, contact_gap, slide_start, pred)
    return best


def _push_on_green_spans(
        step_options: Sequence[StepOption], greens: Sequence[Object],
        domino_names: Set[str]) -> Tuple[List[Tuple[int, int]], bool]:
    """Maximal runs of consecutive action indices whose option is a Push on a
    green start block, plus whether any option label was missing.

    A Push whose objects include no domino at all is the restricted
    variant (``domino_restricted_push``), which always targets the
    inferred start block - counted as a push on green. A Push that
    explicitly names a non-green domino is not.
    """
    green_names = {g.name for g in greens}
    push_idxs = []
    any_unknown = False
    for i, step_option in enumerate(step_options):
        if step_option is None:
            any_unknown = True
            continue
        name, object_names = step_option
        if name == _PUSH_OPTION_NAME and (
                green_names & set(object_names)
                or not domino_names & set(object_names)):
            push_idxs.append(i)
    spans: List[Tuple[int, int]] = []
    for i in push_idxs:
        if spans and i == spans[-1][1] + 1:
            spans[-1] = (spans[-1][0], i)
        else:
            spans.append((i, i))
    return spans, any_unknown


def check_cascade_legitimacy(
        states: Sequence[State],
        goal: Set[GroundAtom],
        step_options: Optional[Sequence[StepOption]] = None
) -> Tuple[bool, str]:
    """Certify that every topple in the episode is a genuine cascade.

    Rules (any violation fails the whole episode):
      (a0) nothing topples before the robot's first Push on the green
           start block, and topples imply such a Push exists;
      (a1) the green block's own fall begins during (or within one
           cascade window after) a Push on it - not during a Pick/Place
           sweep;
      (a2) the green block is never held before its topple onset:
           picking it up (even to put it back) is not part of seeding a
           legitimate cascade (holds after the cascade has started are
           post-success fiddling and stay legal);
      (a3) the green block is pushed where it stands: at its topple
           onset it must be within ``_stage_tolerance()`` of its staged
           (initial) xy - relocating it next to the target to skip the
           chain earns no bonus;
      (a)  the green block is the first domino to start falling (ties
           allowed);
      (b)  every other topple onset is attributable to an
           already-legitimate domino that started falling at most
           ``CASCADE_WINDOW_STEPS`` earlier, either directly - its
           swept fall corridor (from its base along its measured fall
           direction, one domino width wide, ``_cascade_reach()`` long)
           covers the block's footprint - or through ONE shoved relay:
           a block the falling domino's corridor swept that then slid
           at least ``RELAY_MIN_SLIDE`` toward the toppled block and
           reached contact range of its footprint by the onset (see
           ``_best_relay_attribution``; the relay itself need not
           topple). Either way it was knocked over by the cascade, not
           by the robot, a carried block, or spontaneous instability.
           Proximity alone is not causality: a domino that falls BESIDE
           a block does not explain that block's topple, and neither
           does a swept bystander that never moved;
      (c)  when the cascade explanation for a topple is not physically
           corroborated and the robot end-effector was in strike range
           of the block at its onset - or anywhere through the block's
           fall, catching a gripper that keeps driving into it after
           the onset - the robot, not the cascade, is charged with the
           topple and the episode fails. "Physically corroborated"
           means a solid corridor overlap (clearance at most
           ``GRAZE_CONTACT_EPS``) AND the predecessor's actual body
           (leaning footprint) reaching the block (``_body_reached``):
           in tight staged chains the end-effector legitimately
           finishes the push within centimeters of the first relay, but
           a block staged on the corridor's flank can measure
           solid-band clearance while the predecessor falls clear of it
           and the gripper does the toppling (run_20260714_145053
           task 4). The relay path is guarded the same way at both of
           its seams: a grazing relay-to-block contact with the EE in
           strike range of the block, or a corridor-to-relay
           attribution that is grazing or never physically reached the
           relay with the EE in strike range of the relay when its
           slide began, is charged to the robot.

    ``step_options`` labels each transition ``states[t] -> states[t+1]``
    (action index ``t``) with the producing option; when it is None the
    action rules (a0)/(a1) are skipped and only the kinematic rules
    apply. ``goal`` is used only for error messages - all dominoes are
    held to the same rules.

    Returns ``(ok, reason)`` with a human-readable reason on failure.
    """
    del goal  # roles are recovered from state features
    if len(states) < 2:
        return True, ""
    dominoes = [obj for obj in states[0] if obj.type.name == "domino"]
    greens = [
        d for d in dominoes if DominoComponent._StartBlock_holds(  # pylint: disable=protected-access
            states[0], [d])
    ]
    onsets: Dict[Object, int] = {}
    for d in dominoes:
        onset = _topple_onset(states, d)
        if onset is not None:
            onsets[d] = onset
    if not onsets:
        return True, ""
    if not greens:
        toppled = sorted(d.name for d in onsets)
        return False, (f"{', '.join(toppled)} toppled but there is no green "
                       "start block in the scene to seed a cascade")

    # Rule (a2): the green start block is never held before its topple
    # onset. Holds after the cascade has started cannot have seeded it
    # (a re-stand-and-re-tip of the fallen green fails rule (b)'s onset
    # window), so they stay legal - episodes routinely continue past
    # the goal (terminate_on_goal_reached=False) and post-success
    # fiddling must not void an already-legitimate cascade.
    for g in greens:
        # Scan up to this green's own onset; a green that never falls
        # is held to the cascade's start (holding it afterwards cannot
        # have seeded anything).
        horizon = onsets[g] + 1 if g in onsets else min(onsets.values())
        for t in range(horizon):
            if states[t].get(g, "is_held") > 0.5:
                return False, (
                    f"the green start block {g.name} was picked up (held at "
                    f"step {t}, before its cascade began) - it must be "
                    "toppled by a Push at its staged pose, not relocated")

    # Rule (a3): the green start block is pushed where it stands.
    stage_tol = _stage_tolerance()
    for g in greens:
        if g not in onsets:
            continue
        onset_state = states[min(onsets[g], len(states) - 1)]
        drift = math.hypot(
            onset_state.get(g, "x") - states[0].get(g, "x"),
            onset_state.get(g, "y") - states[0].get(g, "y"))
        if drift > stage_tol:
            return False, (
                f"the green start block {g.name} moved {drift:.2f} m from "
                "its staged pose before falling - the cascade must be "
                "seeded by pushing it where it stands")

    # Action rules (a0)/(a1).
    if step_options is not None:
        spans, any_unknown = _push_on_green_spans(step_options, greens,
                                                  {d.name
                                                   for d in dominoes})
        if any_unknown:
            logging.warning(
                "[cascade certificate] some actions lack option labels; "
                "action rules may be incomplete for those steps.")
        if not spans:
            return False, ("dominoes toppled but the green start block was "
                           "never pushed (no Push on it in the episode)")
        first_push = spans[0][0]
        for d, t in sorted(onsets.items(), key=lambda kv: kv[1]):
            # Action index first_push produces state index first_push+1.
            if t <= first_push:
                return False, (
                    f"{d.name} started falling at step {t}, before the "
                    "green start block was first pushed (step "
                    f"{first_push + 1}) - the scene must stay standing "
                    "until the push")
        for g in greens:
            if g not in onsets:
                continue
            t_g = onsets[g]
            if not any(start + 1 <= t_g <= end + 1 + CASCADE_WINDOW_STEPS
                       for start, end in spans):
                return False, (
                    f"the green start block {g.name} started falling at "
                    f"step {t_g}, outside any Push on it - it must be "
                    "toppled by the Push, not by other robot motion")

    # Rule (a): green first (ties allowed).
    min_onset = min(onsets.values())
    if not any(g in onsets and onsets[g] == min_onset for g in greens):
        earliest = min((d for d in onsets if d not in greens),
                       key=lambda d: (onsets[d], d.name))
        green_str = " and ".join(g.name for g in greens)
        return False, (
            f"{earliest.name} started falling at step {onsets[earliest]} "
            f"before the green start block ({green_str}) - the cascade "
            "must start from the green block")

    # Rules (b)/(c): every non-green onset must chain back to the green
    # through a directionally-consistent knock. Attribution is a
    # reachability computation over the knock graph, so it must run to a
    # fixed point: a tightly packed chain can cross two hops within one
    # recorded step, giving knocker and knocked-over the SAME onset step
    # (run_20260713_172854 seed0 task0: domino_1 and domino_2 both onset
    # at step 148, and domino_1 - rejected at 0.067 m against the green -
    # was solidly overlapped by domino_2, which a single pass in
    # (onset, name) order never got to legitimize). Each sweep admits
    # every block the current legit set explains; rejections are only
    # final once a full sweep makes no progress, because a graze-band
    # attribution that rule (c) would charge to the robot can be
    # superseded by a solid-overlap explanation from a same-step
    # predecessor admitted later in the sweep.
    robots = [obj for obj in states[0] if obj.type.name == "robot"]
    legit = {g for g in greens if g in onsets}
    pending = sorted(((d, t) for d, t in onsets.items() if d not in legit),
                     key=lambda kv: (kv[1], kv[0].name))
    while pending:
        progress = False
        deferred: List[Tuple[Object, int]] = []
        failures: Dict[Object, str] = {}
        robot_charged: Set[Object] = set()
        for d, t in pending:
            state = states[min(t, len(states) - 1)]
            best_gap: Optional[float] = None
            solid_reached = False
            for p in legit:
                t_p = onsets[p]
                if not t_p <= t <= t_p + CASCADE_WINDOW_STEPS:
                    continue
                gap = _knock_gap(states, p, t_p, d, state)
                if best_gap is None or gap < best_gap:
                    best_gap = gap
                # The rule (c) exemption needs more than a solid modeled
                # corridor: the predecessor's body must actually have
                # reached the block. A block staged on the corridor's
                # flank can measure solid-band clearance while the
                # predecessor falls clear of it and the gripper does the
                # toppling (run_20260714_145053 task 4).
                if robots and not solid_reached \
                        and gap <= GRAZE_CONTACT_EPS \
                        and _body_reached(states, p, t_p, d, t):
                    solid_reached = True
            if best_gap is not None and best_gap <= CORRIDOR_TOLERANCE:
                if not solid_reached and robots and \
                        _robot_strike_nearby(states, robots[0], d, t,
                                             through_fall=True):
                    if best_gap > GRAZE_CONTACT_EPS:
                        detail = ("the best cascade explanation only grazes "
                                  f"it ({best_gap:.3f} m clearance)")
                    else:
                        detail = ("the modeled fall corridor covers it but "
                                  "no falling domino's body ever reached it")
                    failures[d] = (
                        f"{d.name} started falling at step {t} with the "
                        f"robot end-effector within {ROBOT_STRIKE_XY} m of "
                        f"it while {detail} - it was toppled "
                        "by the robot, not the cascade")
                    robot_charged.add(d)
                    deferred.append((d, t))
                    continue
                legit.add(d)
                progress = True
                continue
            relay = _best_relay_attribution(states, onsets, legit, dominoes, d,
                                            t)
            if relay is None:
                detail = ("no already-falling domino in reach "
                          f"(within {CASCADE_WINDOW_STEPS} steps)"
                          if best_gap is None else
                          "every falling domino's sweep missed its footprint "
                          f"(clearance {best_gap:.3f} m) and no swept block "
                          "slid into it as a relay")
                # Name the likely culprit when the poker passed by: a
                # graze-then-slow-tip knock puts the onset well past the
                # tight verdict lookback, and without this hint the
                # failure reads as a pure geometry margin, sending a
                # searching agent off to shave corridor clearance
                # instead of moving the block off the poker's path.
                hint = ""
                if robots and _robot_strike_nearby(
                        states,
                        robots[0],
                        d,
                        t,
                        through_fall=True,
                        lookback=CASCADE_WINDOW_STEPS):
                    hint = (
                        f" (the end-effector passed within {ROBOT_STRIKE_XY}"
                        " m of it shortly before it fell, so the likely "
                        "cause is the robot's push stroke or retreat, not "
                        "cascade geometry)")
                failures[d] = (
                    f"{d.name} started falling at step {t} but {detail} - it "
                    f"was not knocked over by the cascade{hint}")
                deferred.append((d, t))
                continue
            r, corridor_gap, contact_gap, slide_start, pred_r = relay
            if contact_gap > GRAZE_CONTACT_EPS and robots and \
                    _robot_strike_nearby(states, robots[0], d, t,
                                         through_fall=True):
                failures[d] = (
                    f"{d.name} started falling at step {t} with the robot "
                    f"end-effector within {ROBOT_STRIKE_XY} m of it while "
                    f"the shoved relay {r.name} only grazes it "
                    f"({contact_gap:.3f} m clearance) - it was toppled by "
                    "the robot, not the cascade")
                robot_charged.add(d)
                deferred.append((d, t))
                continue
            # The corridor-to-relay seam mirrors the direct seam: a
            # solid modeled corridor over the relay is only trusted when
            # the ramming predecessor's body actually reached the
            # relay's staged pose (its footprint at the predecessor's
            # onset, matching the corridor computation).
            ram_reached = (corridor_gap <= GRAZE_CONTACT_EPS
                           and _body_reached(states,
                                             pred_r,
                                             onsets[pred_r],
                                             r,
                                             slide_start,
                                             fp_t=onsets[pred_r]))
            if not ram_reached and robots and \
                    _robot_strike_nearby(states, robots[0], r, slide_start):
                if corridor_gap > GRAZE_CONTACT_EPS:
                    ram_detail = (f"the cascade only grazes {r.name} "
                                  f"({corridor_gap:.3f} m clearance)")
                else:
                    ram_detail = ("no falling domino's body ever reached "
                                  f"{r.name}")
                failures[d] = (
                    f"{d.name} started falling at step {t}, knocked by "
                    f"{r.name} sliding into it, but {ram_detail} and "
                    "the robot end-effector was in strike range when its "
                    f"slide began (step {slide_start}) - the slide is "
                    "charged to the robot, not the cascade")
                robot_charged.add(d)
                deferred.append((d, t))
                continue
            legit.add(d)
            progress = True
        if not progress:
            # Report the root cause: a robot-charged block explains any
            # downstream orphans that would have chained through it, so
            # prefer it over their generic sweep-miss failures.
            first = min(deferred,
                        key=lambda kv:
                        (kv[1], kv[0] not in robot_charged, kv[0].name))[0]
            return False, failures[first]
        pending = deferred
    return True, ""
