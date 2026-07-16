"""Trajectory-level legitimacy certificate for domino cascade tasks.

The domino evaluator's terminated/success checks (``DominoEvaluator``)
are functions of the final state only (goal atoms + toppled-blue
count), which are blind to HOW the target fell. Observed reward hacks
all bypass the intended causal chain "push the green start block ->
dominoes knock each other over -> target falls": pushing a placed blue
directly, sweeping the target with the gripper or a carried block
during Place, knocking the target while an option flails, relocating
the green start block (or the targets themselves) to skip the chain,
and toppling a block with the robot's body during or right after the
Push so the arm - not the cascade - supplies the energy.

The certificate decides legitimacy in three layers:

1. **Pre-push integrity** (pure state/action rules): until the first
   Push on the green start block, the scene must stay as staged -
   nothing topples, no non-movable domino is held or drifts from its
   staged pose - and the green block must fall during a Push on it.
   Only the green may ever be the target of a Push.
2. **Cascade timing** (pure state rule): every later topple onset must
   fall within ``CASCADE_WINDOW_STEPS`` of an earlier one, so a block
   knocked over long after the cascade settled (an arm flail, a late
   nudge) cannot ride on a stale push.
3. **The counterfactual push probe** (physics, via the injected
   ``probe``): from the recorded pre-push state, a motorized
   disembodied finger re-runs the episode's own Push - the skill's
   waypoint path with the plan's recorded continuous parameters - in a
   dedicated same-physics world with the robot arm parked (see
   ``cascade_probe``). The goal atoms must topple under that push.
   This replaces the old swept-corridor /
   shoved-relay / robot-strike attribution rules: instead of guessing
   from geometry whether the arm or the cascade toppled each block -
   a forensic call that produced both misses and false rejections
   (run_20260715_220941: a genuine green-on-blue knock measured 7 mm
   of modeled corridor clearance with the end-effector nearby and was
   charged to the robot) - the probe answers the only question that
   matters: does the layout the robot built actually cascade to the
   goal under a clean push? Arm collateral cannot help (the probe has
   no arm), so any hack that needs the robot's body to reach the goal
   fails the probe.

Layers 1 and 2 are pure functions over ``State`` sequences and run
identically everywhere; layer 3 needs a physics rollout, so the caller
injects ``probe`` (``DominoEvaluator`` binds it from the certifying
env - the true env env-side, the agent's belief env in sandbox
verdicts, each side probing with its own physics). When no probe is
available the certificate runs layers 1-2 only and logs that the
counterfactual was skipped.
"""

import logging
import math
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

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

# Minimum base displacement for a movable blue to count as consumed by
# the cascade (shoved off its stand) in ``count_movable_blocks_used``.
# Well above resting jitter and release-settle skids (millimeters); a
# genuine transmitting slide covers centimeters (the recorded
# slide-relay episode measured 0.12 m).
RELAY_MIN_SLIDE = 0.02

# Signature of the injected counterfactual probe: (pre-push state,
# pushed greens in push order, goal atoms, the episode's Push continuous
# parameters or None) -> (ok, human-readable detail). See
# ``PyBulletDominoComposedEnv.run_counterfactual_cascade_probe``.
CascadeProbe = Callable[
    [State, Sequence[Object], Set[GroundAtom], Optional[Tuple[float, ...]]],
    Tuple[bool, str]]


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


def _stage_tolerance() -> float:
    """Max xy drift of a non-movable domino from its staged pose that still
    counts as "left where it stands".

    A legitimate push tips the green block about its base edge, so the
    base slides at most a couple of centimeters before the block leaves
    the upright band; relocating any block anywhere useful (next to the
    target, or even one chain gap over) moves it by at least ``pos_gap``
    ~= 0.098 m. One domino width sits safely between the two regimes.
    """
    from predicators.envs.pybullet_domino.env import \
        PyBulletDominoComposedEnv  # pylint: disable=import-outside-toplevel
    return PyBulletDominoComposedEnv.domino_width


def _role_label(state: State, domino: Object) -> str:
    """Human-readable role of a non-movable ``domino``, for error messages."""
    # pylint: disable=protected-access
    if DominoComponent._StartBlock_holds(state, [domino]):
        return "green start block"
    if DominoComponent._TargetDomino_holds(state, [domino]):
        return "target domino"
    if DominoComponent._HeavyBlock_holds(state, [domino]):
        return "heavy block"
    return "non-movable domino"


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
        name, object_names = step_option[0], step_option[1]
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


def _pushed_greens_in_order(step_options: Sequence[StepOption],
                            greens: Sequence[Object],
                            spans: Sequence[Tuple[int, int]]) -> List[Object]:
    """The green blocks the episode pushed, ordered by their first push.

    A restricted Push names no domino; it targets the inferred start
    block, which is unambiguous exactly when there is one green.
    """
    by_name = {g.name: g for g in greens}
    ordered: List[Object] = []
    for start, _ in spans:
        step_option = step_options[start]
        assert step_option is not None
        object_names = step_option[1]
        named = [by_name[n] for n in object_names if n in by_name]
        for g in named or list(greens):
            if g not in ordered:
                ordered.append(g)
    return ordered


def _push_params_of_span(step_options: Sequence[StepOption],
                         span_start: int) -> Optional[Tuple[float, ...]]:
    """The continuous Push parameters recorded on a span's first label.

    Returns None for legacy 2-tuple labels (agent-authored label lists,
    old tests) or empty parameter tuples - the probe then falls back to
    the canonical push.
    """
    step_option = step_options[span_start]
    assert step_option is not None
    if len(step_option) > 2 and step_option[2]:
        return tuple(step_option[2])
    return None


def check_cascade_legitimacy(
        states: Sequence[State],
        goal: Set[GroundAtom],
        step_options: Optional[Sequence[StepOption]] = None,
        probe: Optional[CascadeProbe] = None) -> Tuple[bool, str]:
    """Certify that the episode's topples are a genuine push-seeded cascade.

    Rules (any violation fails the whole episode):
      (a0) nothing topples before the robot's first Push on the green
           start block, and topples imply such a Push exists; the Push
           option is only ever legal on the green - a Push that names
           any other domino fails outright;
      (a1) the green block's own fall begins during (or within one
           cascade window after) a Push on it - not during a Pick/Place
           sweep;
      (a2) no non-movable domino - the green start block, the targets,
           heavy blocks - is ever held before its topple onset: only
           the blue movable blocks are the robot's to carry (holds
           after a block has fallen are post-success fiddling and stay
           legal);
      (a3) every non-movable domino is toppled where it stands: at the
           cascade's start it must be within ``_stage_tolerance()`` of
           its staged (initial) xy;
      (a)  the green block is the first domino to start falling (ties
           allowed);
      (b)  every topple onset chains in time: each non-green onset lies
           within ``CASCADE_WINDOW_STEPS`` of an earlier onset, so a
           block knocked over after the cascade already settled (a
           flail, a late nudge, spontaneous instability) is rejected
           without any geometric attribution;
      (c)  when the goal atoms hold at the episode's end, the injected
           counterfactual ``probe`` must reproduce the cascade: a
           motorized disembodied finger re-running the episode's own
           Push (skill waypoint path, the plan's recorded continuous
           parameters) on the green(s) from the recorded pre-push
           state - same physics, robot arm parked - must reach the
           goal atoms. This is the layer that separates "the layout
           works" from "the robot's body made it work" (see the module
           docstring); it runs only on goal-reaching episodes because
           only those have a success bonus at stake.

    ``step_options`` labels each transition ``states[t] -> states[t+1]``
    (action index ``t``) with the producing option; when it is None the
    action rules (a0)/(a1) are skipped, only the kinematic rules apply,
    and the probe falls back to the state just before the first topple
    onset as its pre-push state. ``goal`` feeds the probe's success
    check and the error messages - all dominoes are held to the same
    rules.

    Returns ``(ok, reason)`` with a human-readable reason on failure.
    """
    if len(states) < 2:
        return True, ""
    dominoes = [obj for obj in states[0] if obj.type.name == "domino"]
    domino_names = {d.name for d in dominoes}
    # pylint: disable=protected-access
    greens = [
        d for d in dominoes
        if DominoComponent._StartBlock_holds(states[0], [d])
    ]
    # Only the blue movable blocks are the robot's to arrange; every
    # other domino (the green start block, the targets, heavy blocks) is
    # scenery that must be toppled where the task staged it.
    non_movables = [
        d for d in dominoes
        if not DominoComponent._MovableBlock_holds(states[0], [d])
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
    cascade_start = min(onsets.values())

    # Rule (a2): no non-movable domino is ever held before its topple
    # onset. Holds after a block has fallen cannot have seeded or staged
    # anything, so they stay legal - episodes routinely continue past
    # the goal (terminate_on_goal_reached=False) and post-success
    # fiddling must not void an already-legitimate cascade.
    for d in non_movables:
        # Scan up to this block's own onset; a block that never falls
        # is held to the cascade's start (holding it afterwards cannot
        # have seeded or staged anything).
        horizon = onsets[d] + 1 if d in onsets else cascade_start
        for t in range(horizon):
            if states[t].get(d, "is_held") > 0.5:
                return False, (
                    f"the {_role_label(states[0], d)} {d.name} was picked up "
                    f"(held at step {t}, before it fell) - only the blue "
                    "movable blocks may be carried; it must be toppled where "
                    "it stands, not relocated")

    # Rule (a3): every non-movable domino is still at its staged pose
    # when the cascade starts.
    stage_tol = _stage_tolerance()
    for d in non_movables:
        start_state = states[min(cascade_start, len(states) - 1)]
        drift = math.hypot(
            start_state.get(d, "x") - states[0].get(d, "x"),
            start_state.get(d, "y") - states[0].get(d, "y"))
        if drift > stage_tol:
            return False, (
                f"the {_role_label(states[0], d)} {d.name} moved "
                f"{drift:.2f} m from its staged pose before the cascade "
                "started - only the blue movable blocks may be rearranged")

    # Action rules (a0)/(a1).
    pre_push_idx: Optional[int] = None
    pushed_greens: List[Object] = list(greens)
    push_params: Optional[Tuple[float, ...]] = None
    if step_options is not None:
        for i, step_option in enumerate(step_options):
            if step_option is None:
                continue
            name, object_names = step_option[0], step_option[1]
            if name == _PUSH_OPTION_NAME:
                foreign = sorted((set(object_names) & domino_names) -
                                 {g.name
                                  for g in greens})
                if foreign:
                    return False, (
                        f"the robot pushed {', '.join(foreign)} (Push at "
                        f"step {i}) - only the green start block may be "
                        "pushed")
        spans, any_unknown = _push_on_green_spans(step_options, greens,
                                                  domino_names)
        if any_unknown:
            logging.warning(
                "[cascade certificate] some actions lack option labels; "
                "action rules may be incomplete for those steps.")
        if not spans:
            return False, ("dominoes toppled but the green start block was "
                           "never pushed (no Push on it in the episode)")
        first_push = spans[0][0]
        pre_push_idx = first_push
        pushed_greens = _pushed_greens_in_order(step_options, greens, spans)
        push_params = _push_params_of_span(step_options, first_push)
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

    # Rule (b): cascade timing. Every non-green onset must lie within
    # one cascade window of an EARLIER onset (ties allowed): energy can
    # only flow from a block that was already falling. Green onsets are
    # anchored to their push spans by rule (a1) instead.
    onset_times = sorted(onsets.values())
    for d, t in sorted(onsets.items(), key=lambda kv: (kv[1], kv[0].name)):
        if d in greens:
            continue
        if not any(t_prev <= t <= t_prev + CASCADE_WINDOW_STEPS
                   for t_prev in onset_times if t_prev < t) \
                and not any(onsets[g] == t for g in greens if g in onsets):
            return False, (
                f"{d.name} started falling at step {t}, more than "
                f"{CASCADE_WINDOW_STEPS} steps after every earlier fall - "
                "it was not knocked over by the push's cascade")

    # Rule (c): the counterfactual push probe, on goal-reaching episodes.
    if not all(atom.holds(states[-1]) for atom in goal):
        return True, ""
    if probe is None:
        logging.debug(
            "[cascade certificate] no counterfactual probe available; "
            "accepting on the pure rules only.")
        return True, ""
    if pre_push_idx is None:
        pre_push_idx = max(cascade_start - 1, 0)
    ok, detail = probe(states[pre_push_idx], pushed_greens, goal, push_params)
    if not ok:
        return False, (
            "the goal atoms hold, but a clean counterfactual push on "
            f"{', '.join(g.name for g in pushed_greens)} from the pre-push "
            f"scene does not reproduce the cascade ({detail}) - the goal "
            "topples are owed to the robot's body, not the built layout")
    return True, ""
