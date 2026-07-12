"""Trajectory-level legitimacy certificate for min-block cascade tasks.

The min-block evaluator's terminated/success checks (``DominoEvaluator``)
are functions of the final state only (goal atoms + toppled-blue count),
which are blind to HOW the target fell. Observed reward hacks all bypass
the intended causal chain
"push the green start block -> dominoes knock each other over -> target
falls": pushing a placed blue directly, sweeping the target with the
gripper or a carried block during Place, and knocking the target while
an option flails. This module certifies, from the recorded per-step
states (plus, when available, which option produced each transition),
that every topple in the episode traces back to a robot Push on the
green start block through domino-on-domino contact.

The certificate is a pure function over ``State`` sequences - no
PyBullet contact queries - so it runs identically on true-env episodes
and on option-model rollouts in the agent's simulator.
"""

import logging
import math
from typing import Dict, List, Optional, Sequence, Set, Tuple

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


def count_movable_blocks_used(state: State) -> int:
    """Count movable (blue) dominoes that have toppled in ``state``.

    A pure function of the state (roles recovered from color features,
    topple from the roll angle) so ``DominoEvaluator`` needs no live env
    handle and stays picklable.
    """
    count = 0
    for obj in state:
        if obj.type.name != "domino":
            continue
        # pylint: disable=protected-access
        if DominoComponent._MovableBlock_holds(state, [obj]) and \
                abs(state.get(obj, "roll")) >= DominoComponent.fallen_threshold:
            count += 1
    return count


def _cascade_reach() -> float:
    """Max center-to-center xy distance at which a falling domino can plausibly
    have knocked another one over.

    A falling domino's top sweeps ~domino_height ahead of its base; two
    depths of slack absorb base sliding during the fall.
    """
    from predicators.envs.pybullet_domino.env import \
        PyBulletDominoComposedEnv  # pylint: disable=import-outside-toplevel
    return (PyBulletDominoComposedEnv.domino_height +
            2 * PyBulletDominoComposedEnv.domino_depth)


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
           ``CASCADE_WINDOW_STEPS`` earlier and lies within topple
           reach - i.e. it was knocked over by a falling domino, not by
           the robot, a carried block, or spontaneous instability.

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

    # Rule (b): every non-green onset must chain back to the green.
    reach = _cascade_reach()
    legit = {g for g in greens if g in onsets}
    ordered = sorted(onsets.items(),
                     key=lambda kv: (kv[1], kv[0] not in greens, kv[0].name))
    for d, t in ordered:
        if d in legit:
            continue
        state = states[min(t, len(states) - 1)]
        attributed = False
        for p in legit:
            t_p = onsets[p]
            if not t_p <= t <= t_p + CASCADE_WINDOW_STEPS:
                continue
            dist = math.hypot(
                state.get(p, "x") - state.get(d, "x"),
                state.get(p, "y") - state.get(d, "y"))
            if dist <= reach:
                attributed = True
                break
        if not attributed:
            return False, (
                f"{d.name} started falling at step {t} with no "
                f"already-falling domino within {reach:.2f} m and "
                f"{CASCADE_WINDOW_STEPS} steps - it was not knocked over "
                "by the cascade")
        legit.add(d)
    return True, ""
