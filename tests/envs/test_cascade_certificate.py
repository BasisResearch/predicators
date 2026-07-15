"""Tests for the min-block cascade legitimacy certificate.

Pure-State tests (no PyBullet): trajectories are synthesized as step-
function roll profiles, so each domino's topple onset is exactly the
step where its roll jumps past the fallen threshold.
"""

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pytest

from predicators.envs.pybullet_domino.cascade_certificate import \
    CASCADE_WINDOW_STEPS, check_cascade_legitimacy, \
    count_movable_blocks_used
from predicators.envs.pybullet_domino.components.domino_component import \
    DominoComponent
from predicators.structs import GroundAtom, Object, Predicate, State, Type

_DOMINO_TYPE = Type("domino",
                    ["x", "y", "z", "yaw", "roll", "r", "g", "b", "is_held"])
_ROBOT_TYPE = Type("robot", ["x", "y", "z"])
_TOPPLED = Predicate("Toppled", [_DOMINO_TYPE], lambda s, o: False)

_GREEN = DominoComponent.start_domino_color
_BLUE = DominoComponent.domino_color
_PURPLE = DominoComponent.target_domino_color

# Past fallen_threshold (10 deg ~= 0.175). Negative: with the env's yaw
# convention a NEGATIVE roll falls along (-sin yaw, cos yaw), i.e. +y at
# yaw 0, which is the direction the synthetic chains below progress.
_FALLEN_ROLL = -0.6

# A straight chain along y with hops well inside the 0.18 m reach.
_CHAIN_Y = {"green": 1.0, "blue1": 1.098, "blue2": 1.196, "target": 1.294}


def _make_objects(names: Sequence[str]) -> Dict[str, Object]:
    return {name: Object(name, _DOMINO_TYPE) for name in names}


def _color_for(name: str) -> Tuple[float, float, float, float]:
    if name.startswith("green"):
        return _GREEN
    if name.startswith("target"):
        return _PURPLE
    return _BLUE


def _build_states(
    objs: Dict[str, Object],
    num_steps: int,
    onsets: Dict[str, int],
    *,
    positions: Optional[Dict[str, Tuple[float, float]]] = None,
    position_profiles: Optional[Dict[str, Sequence[Tuple[float,
                                                         float]]]] = None,
    roll_profiles: Optional[Dict[str, Sequence[float]]] = None,
    held_spans: Optional[Dict[str, Tuple[int, int]]] = None,
    yaws: Optional[Dict[str, float]] = None,
    robot_xyz: Optional[Tuple[float, float, float]] = None,
    robot_profile: Optional[Sequence[Tuple[float, float, float]]] = None,
) -> List[State]:
    """Build a trajectory of ``num_steps + 1`` states.

    ``onsets[name] = t`` gives a step-function roll: 0 before ``t``,
    fallen after. ``roll_profiles`` overrides the roll sequence for a
    domino entirely; ``position_profiles`` gives a per-step (x, y)
    sequence for a domino (e.g. a relocation); ``held_spans[name] =
    (a, b)`` sets is_held on state indices [a, b]; ``yaws`` overrides
    the default 0 yaw; ``robot_xyz`` adds a robot object parked at that
    end-effector position for the whole episode, or ``robot_profile``
    adds one that moves along a per-step (x, y, z) sequence.
    """
    add_robot = robot_xyz is not None or robot_profile is not None
    robot = Object("robot", _ROBOT_TYPE) if add_robot else None
    states = []
    for t in range(num_steps + 1):
        data = {}
        for name, obj in objs.items():
            if position_profiles is not None and name in position_profiles:
                x, y = position_profiles[name][t]
            elif positions is not None and name in positions:
                x, y = positions[name]
            else:
                x, y = 0.7, _CHAIN_Y[name]
            if roll_profiles is not None and name in roll_profiles:
                roll = roll_profiles[name][t]
            elif name in onsets:
                roll = _FALLEN_ROLL if t >= onsets[name] else 0.0
            else:
                roll = 0.0
            held = 0.0
            if held_spans is not None and name in held_spans:
                lo, hi = held_spans[name]
                held = 1.0 if lo <= t <= hi else 0.0
            yaw = yaws.get(name, 0.0) if yaws is not None else 0.0
            data[obj] = np.array(
                [x, y, 0.475, yaw, roll, *_color_for(name)[:3], held],
                dtype=np.float32)
        if robot is not None:
            xyz = robot_profile[t] if robot_profile is not None else robot_xyz
            data[robot] = np.array(xyz, dtype=np.float32)
        states.append(State(data))
    return states


def _goal(objs: Dict[str, Object]) -> set:
    return {GroundAtom(_TOPPLED, [objs["target"]])}


def _options(
    spans: Sequence[Tuple[str, Tuple[str, ...], int, int]],
    num_actions: int,
) -> List[Optional[Tuple[str, Tuple[str, ...]]]]:
    """Build a per-action option labeling from (name, objects, lo, hi) spans
    over action indices; unlabeled actions become Wait."""
    labels: List[Optional[Tuple[str, Tuple[str, ...]]]] = [
        ("Wait", ("robot", )) for _ in range(num_actions)
    ]
    for name, objects, lo, hi in spans:
        for i in range(lo, hi + 1):
            labels[i] = (name, objects)
    return labels


def test_legit_chain_passes():
    """Push green -> cascade through two blues -> target: accepted."""
    objs = _make_objects(["green", "blue1", "blue2", "target"])
    states = _build_states(objs, 30, {
        "green": 5,
        "blue1": 10,
        "blue2": 14,
        "target": 18
    })
    # Push on green during actions 0-7, then Wait while the cascade runs.
    step_options = _options([("Push", ("robot", "green"), 0, 7)], 30)
    ok, reason = check_cascade_legitimacy(states, _goal(objs), step_options)
    assert ok, reason


def test_k0_pure_push_passes():
    """Green adjacent to target, no blues needed: accepted."""
    objs = _make_objects(["green", "target"])
    states = _build_states(objs,
                           20, {
                               "green": 5,
                               "target": 11
                           },
                           positions={
                               "green": (0.7, 1.0),
                               "target": (0.7, 1.098)
                           })
    step_options = _options([("Push", ("robot", "green"), 0, 7)], 20)
    ok, reason = check_cascade_legitimacy(states, _goal(objs), step_options)
    assert ok, reason


def test_green_relocation_fails():
    """Picking up green and placing it next to the target: rejected.

    The zero-blue exploit from run_20260712_122549: relocate the green
    start block adjacent to the target, push it, topple the target with
    no chain at all. Rule (a2) rejects the pick regardless of where the
    block ends up.
    """
    objs = _make_objects(["green", "target"])
    staged = (0.7, 1.0)
    relocated = (0.7, 1.28)  # adjacent to the target at (0.7, 1.378)
    green_xy = [staged] * 3 + [relocated] * 28
    states = _build_states(objs,
                           30, {
                               "green": 14,
                               "target": 18
                           },
                           positions={"target": (0.7, 1.378)},
                           position_profiles={"green": green_xy},
                           held_spans={"green": (3, 8)})
    step_options = _options([("Pick", ("robot", "green"), 0, 3),
                             ("Place", ("robot", ), 4, 8),
                             ("Push", ("robot", "green"), 10, 13)], 30)
    ok, reason = check_cascade_legitimacy(states, _goal(objs), step_options)
    assert not ok
    assert "picked up" in reason


def test_green_slide_relocation_fails():
    """Sliding green next to the target without ever holding it: rejected.

    Rule (a3): at its topple onset the green block must be within the
    stage tolerance of its initial pose, so a gripper-sweep relocation
    earns no bonus even though is_held never fires.
    """
    objs = _make_objects(["green", "target"])
    green_xy = [(0.7, 1.0)] * 5 + [(0.7, 1.28)] * 26
    states = _build_states(objs,
                           30, {
                               "green": 14,
                               "target": 18
                           },
                           positions={"target": (0.7, 1.378)},
                           position_profiles={"green": green_xy})
    step_options = _options([("Push", ("robot", "green"), 10, 13)], 30)
    ok, reason = check_cascade_legitimacy(states, _goal(objs), step_options)
    assert not ok
    assert "staged pose" in reason


def test_green_pickup_after_cascade_passes():
    """Picking the fallen green up after the cascade stays legal.

    Episodes run past the goal (terminate_on_goal_reached=False), and
    exploration routinely fiddles with fallen blocks afterwards - that
    must not void an already-legitimate cascade (real case: the two
    reward-0.75 train episodes of run_20260712_122549, green picked up
    at steps 360/287 of 500).
    """
    objs = _make_objects(["green", "blue1", "blue2", "target"])
    states = _build_states(objs,
                           30, {
                               "green": 5,
                               "blue1": 10,
                               "blue2": 14,
                               "target": 18
                           },
                           held_spans={"green": (24, 28)})
    step_options = _options([("Push", ("robot", "green"), 0, 7),
                             ("Pick", ("robot", "green"), 23, 27)], 30)
    ok, reason = check_cascade_legitimacy(states, _goal(objs), step_options)
    assert ok, reason


def test_green_push_jitter_passes():
    """Small base slide during the push itself stays within tolerance."""
    objs = _make_objects(["green", "target"])
    green_xy = [(0.7, 1.0)] * 11 + [(0.7, 1.02)] * 20
    states = _build_states(objs,
                           30, {
                               "green": 12,
                               "target": 16
                           },
                           positions={"target": (0.7, 1.098)},
                           position_profiles={"green": green_xy})
    step_options = _options([("Push", ("robot", "green"), 10, 13)], 30)
    ok, reason = check_cascade_legitimacy(states, _goal(objs), step_options)
    assert ok, reason


def test_target_relocation_fails():
    """Carrying the targets into a line in front of green: rejected.

    The zero-blue exploit from run_20260715_084342 task 1: leave green
    and the blues untouched, but pick the targets up and set them down
    in front of green, so one push cascades straight through them. It
    reached the goal with no chain at all and scored a full reward=1.0.
    Rule (a2) rejects the pick: only blues are the robot's to carry.
    """
    objs = _make_objects(["green", "blue1", "target"])
    staged = (0.9, 1.32)
    relocated = (0.7, 1.098)  # lined up in front of green at (0.7, 1.0)
    target_xy = [staged] * 3 + [relocated] * 28
    states = _build_states(objs,
                           30, {
                               "green": 14,
                               "target": 18
                           },
                           positions={"green": (0.7, 1.0)},
                           position_profiles={"target": target_xy},
                           held_spans={"target": (3, 8)})
    step_options = _options([("Pick", ("robot", "target"), 0, 3),
                             ("Place", ("robot", ), 4, 8),
                             ("Push", ("robot", "green"), 10, 13)], 30)
    ok, reason = check_cascade_legitimacy(states, _goal(objs), step_options)
    assert not ok
    assert "target domino" in reason
    assert "picked up" in reason


def test_target_slide_relocation_fails():
    """Sliding a target into place without ever holding it: rejected.

    Rule (a3) backstops (a2): a gripper-sweep relocation of a target
    before the cascade starts earns no bonus even though is_held never
    fires.
    """
    objs = _make_objects(["green", "target"])
    target_xy = [(0.9, 1.32)] * 5 + [(0.7, 1.098)] * 26
    states = _build_states(objs,
                           30, {
                               "green": 14,
                               "target": 18
                           },
                           positions={"green": (0.7, 1.0)},
                           position_profiles={"target": target_xy})
    step_options = _options([("Push", ("robot", "green"), 10, 13)], 30)
    ok, reason = check_cascade_legitimacy(states, _goal(objs), step_options)
    assert not ok
    assert "staged pose" in reason


def test_target_shoved_by_cascade_before_toppling_passes():
    """A target the cascade shoves before it tips is not charged to the robot.

    Rule (a3) measures staged-pose drift at the cascade's START, not at
    each block's own onset, precisely so this stays legal: the chain
    rams the target, slides it several centimeters (further than the
    stage tolerance), and only then tips it over. That displacement is
    the cascade's doing, not a relocation.
    """
    objs = _make_objects(["green", "blue1", "blue2", "target"])
    # Struck at step 15, shoved 0.08 m (> the 0.07 m stage tolerance),
    # topples at 18. Untouched through the cascade's start (step 5).
    target_xy = [(0.7, 1.25)] * 16 + [(0.7, 1.33)] * 15
    states = _build_states(objs,
                           30, {
                               "green": 5,
                               "blue1": 10,
                               "blue2": 14,
                               "target": 18
                           },
                           position_profiles={"target": target_xy})
    step_options = _options([("Push", ("robot", "green"), 0, 7)], 30)
    ok, reason = check_cascade_legitimacy(states, _goal(objs), step_options)
    assert ok, reason


def test_blue_relocation_passes():
    """Carrying the blues into a chain is the task, not an exploit.

    The mirror of ``test_target_relocation_fails``: rules (a2)/(a3)
    cover every domino EXCEPT the blue movable blocks, which the robot
    is expressly given to arrange.
    """
    objs = _make_objects(["green", "blue1", "target"])
    blue_xy = [(0.9, 1.32)] * 3 + [(0.7, 1.098)] * 28
    states = _build_states(objs,
                           30, {
                               "green": 14,
                               "blue1": 18,
                               "target": 22
                           },
                           positions={
                               "green": (0.7, 1.0),
                               "target": (0.7, 1.196)
                           },
                           position_profiles={"blue1": blue_xy},
                           held_spans={"blue1": (3, 8)})
    step_options = _options([("Pick", ("robot", "blue1"), 0, 3),
                             ("Place", ("robot", ), 4, 8),
                             ("Push", ("robot", "green"), 10, 13)], 30)
    ok, reason = check_cascade_legitimacy(states, _goal(objs), step_options)
    assert ok, reason


def test_push_placed_blue_fails():
    """Pushing a blue directly (green never pushed): rejected."""
    objs = _make_objects(["green", "blue1", "target"])
    states = _build_states(objs, 30, {"blue1": 10, "target": 15})
    step_options = _options([("Push", ("robot", "blue1"), 5, 12)], 30)
    ok, reason = check_cascade_legitimacy(states, _goal(objs), step_options)
    assert not ok
    assert "never pushed" in reason


def test_push_placed_blue_fails_without_options():
    """Same hack, kinematic rules only: green never falls -> rejected."""
    objs = _make_objects(["green", "blue1", "target"])
    states = _build_states(objs, 30, {"blue1": 10, "target": 15})
    ok, reason = check_cascade_legitimacy(states, _goal(objs), None)
    assert not ok
    assert "blue1" in reason and "green" in reason


def test_place_knock_fails():
    """Target knocked during Place before any push: rejected by (a0)."""
    objs = _make_objects(["green", "blue1", "target"])
    states = _build_states(objs, 40, {"target": 3, "green": 25, "blue1": 29})
    step_options = _options([("Place", ("robot", ), 0, 9),
                             ("Push", ("robot", "green"), 20, 27)], 40)
    ok, reason = check_cascade_legitimacy(states, _goal(objs), step_options)
    assert not ok
    assert "target" in reason and "before" in reason


def test_place_knock_fails_without_options():
    """Same hack, kinematic rules only: target falls first -> rejected."""
    objs = _make_objects(["green", "blue1", "target"])
    states = _build_states(objs, 40, {"target": 3, "green": 25, "blue1": 29})
    ok, reason = check_cascade_legitimacy(states, _goal(objs), None)
    assert not ok
    assert "target" in reason


def test_flail_knock_fails():
    """Fist knocks the target during Push(green), green stays up."""
    objs = _make_objects(["green", "blue1", "target"])
    states = _build_states(objs, 30, {"target": 15})
    step_options = _options([("Push", ("robot", "green"), 5, 29)], 30)
    ok, reason = check_cascade_legitimacy(states, _goal(objs), step_options)
    assert not ok
    assert "target" in reason


def test_spontaneous_late_tip_fails():
    """A topple far outside the cascade window is not attributable."""
    objs = _make_objects(["green", "blue1", "target"])
    late = 10 + CASCADE_WINDOW_STEPS + 11
    num = late + 5
    states = _build_states(objs, num, {
        "green": 5,
        "blue1": 10,
        "target": late
    })
    step_options = _options([("Push", ("robot", "green"), 0, 7)], num)
    ok, reason = check_cascade_legitimacy(states, _goal(objs), step_options)
    assert not ok
    assert "target" in reason


def test_stalled_relay_fails():
    """Hand-relaying a stalled cascade (huge onset gap): rejected."""
    objs = _make_objects(["green", "blue1", "blue2", "target"])
    relay = 10 + CASCADE_WINDOW_STEPS + 50
    num = relay + 10
    states = _build_states(objs, num, {
        "green": 5,
        "blue1": 10,
        "blue2": relay,
        "target": relay + 4
    })
    step_options = _options([("Push", ("robot", "green"), 0, 7)], num)
    ok, reason = check_cascade_legitimacy(states, _goal(objs), step_options)
    assert not ok
    assert "blue2" in reason


def test_reach_violation_fails():
    """A topple with no fallen domino nearby is not a cascade."""
    objs = _make_objects(["green", "target"])
    states = _build_states(objs,
                           20, {
                               "green": 5,
                               "target": 10
                           },
                           positions={
                               "green": (0.7, 1.0),
                               "target": (0.7, 1.5)
                           })
    step_options = _options([("Push", ("robot", "green"), 0, 7)], 20)
    ok, reason = check_cascade_legitimacy(states, _goal(objs), step_options)
    assert not ok
    assert "target" in reason


def _slide_profile(start: Tuple[float, float], stops: Dict[int, Tuple[float,
                                                                      float]],
                   num_steps: int) -> List[Tuple[float, float]]:
    """Step-function xy profile: at ``start`` until each ``stops[t]`` takes
    over from step ``t`` on."""
    profile = []
    pos = start
    for t in range(num_steps + 1):
        if t in stops:
            pos = stops[t]
        profile.append(pos)
    return profile


def test_slide_relay_passes():
    """A rammed block that slides into the target without toppling is a genuine
    cascade link (rule (b)'s one-hop relay attribution).

    The run_20260713_133936 task-3 false rejection: the green fell onto
    a staged blue, which never toppled - it slid 0.12 m into the purple
    and knocked it over (then rocked back upright). The target is
    beyond the green's swept corridor, so direct attribution fails, but
    the swept relay's slide explains the topple.
    """
    objs = _make_objects(["green", "blue1", "target"])
    num = 30
    # blue1 starts inside green's corridor and slides to contact range
    # of the target between the green's onset (10) and the target's
    # onset (18).
    profile = _slide_profile((0.7, 1.10), {
        11: (0.7, 1.13),
        12: (0.7, 1.17),
        13: (0.7, 1.21),
        14: (0.7, 1.254)
    }, num)
    states = _build_states(objs,
                           num, {
                               "green": 10,
                               "target": 18
                           },
                           positions={
                               "green": (0.7, 1.0),
                               "target": (0.7, 1.27)
                           },
                           position_profiles={"blue1": profile})
    step_options = _options([("Push", ("robot", "green"), 0, 7)], num)
    ok, reason = check_cascade_legitimacy(states, _goal(objs), step_options)
    assert ok, reason
    # The relay was consumed even though it never toppled: it costs the
    # same as a toppled blue, so staying upright earns no discount.
    assert count_movable_blocks_used(states) == 1


def test_slide_relay_stationary_bystander_fails():
    """A swept block that never moved explains nothing: the target's topple
    stays unattributable (proximity alone is not causality)."""
    objs = _make_objects(["green", "blue1", "target"])
    num = 30
    states = _build_states(objs,
                           num, {
                               "green": 10,
                               "target": 18
                           },
                           positions={
                               "green": (0.7, 1.0),
                               "blue1": (0.7, 1.10),
                               "target": (0.7, 1.27)
                           })
    step_options = _options([("Push", ("robot", "green"), 0, 7)], num)
    ok, reason = check_cascade_legitimacy(states, _goal(objs), step_options)
    assert not ok
    assert "target" in reason and "relay" in reason


def test_slide_relay_unswept_fails():
    """A block that slid into the target unrammed is not a cascade relay.

    Its staged footprint sits outside every falling domino's corridor,
    so its slide is robot work, however it was produced.
    """
    objs = _make_objects(["green", "blue1", "target"])
    num = 30
    # blue1 approaches from the side, well clear of green's corridor.
    profile = _slide_profile((0.85, 1.10), {
        11: (0.80, 1.15),
        13: (0.75, 1.20),
        15: (0.7, 1.254)
    }, num)
    states = _build_states(objs,
                           num, {
                               "green": 10,
                               "target": 18
                           },
                           positions={
                               "green": (0.7, 1.0),
                               "target": (0.7, 1.27)
                           },
                           position_profiles={"blue1": profile})
    step_options = _options([("Push", ("robot", "green"), 0, 7)], num)
    ok, reason = check_cascade_legitimacy(states, _goal(objs), step_options)
    assert not ok
    assert "target" in reason


def test_slide_relay_held_fails():
    """A held block sliding into the target is the robot's tool, never a
    relay."""
    objs = _make_objects(["green", "blue1", "target"])
    num = 30
    profile = _slide_profile((0.7, 1.10), {
        11: (0.7, 1.13),
        12: (0.7, 1.17),
        13: (0.7, 1.21),
        14: (0.7, 1.254)
    }, num)
    states = _build_states(objs,
                           num, {
                               "green": 10,
                               "target": 18
                           },
                           positions={
                               "green": (0.7, 1.0),
                               "target": (0.7, 1.27)
                           },
                           position_profiles={"blue1": profile},
                           held_spans={"blue1": (11, 15)})
    step_options = _options([("Push", ("robot", "green"), 0, 7)], num)
    ok, reason = check_cascade_legitimacy(states, _goal(objs), step_options)
    assert not ok
    assert "target" in reason


def test_slide_relay_graze_with_robot_adjacent_fails():
    """A relay whose contact with the target is only a graze, with the EE in
    strike range of the target at its onset, is charged to the robot (rule (c)
    applied at the relay-to-block seam)."""
    objs = _make_objects(["green", "blue1", "target"])
    num = 30
    # Slide stops 0.010 m short of the target's footprint (grazing band).
    profile = _slide_profile((0.7, 1.10), {
        11: (0.7, 1.13),
        12: (0.7, 1.17),
        13: (0.7, 1.21),
        14: (0.7, 1.245)
    }, num)
    states = _build_states(objs,
                           num, {
                               "green": 10,
                               "target": 18
                           },
                           positions={
                               "green": (0.7, 1.0),
                               "target": (0.7, 1.27)
                           },
                           position_profiles={"blue1": profile},
                           robot_xyz=(0.70, 1.30, 0.5))
    step_options = _options([("Push", ("robot", "green"), 0, 7)], num)
    ok, reason = check_cascade_legitimacy(states, _goal(objs), step_options)
    assert not ok
    assert "target" in reason and "robot" in reason


def test_slide_relay_robot_shoved_fails():
    """A relay the cascade only GRAZES, whose slide begins with the EE in
    strike range of the relay, is charged to the robot (rule (c) applied at the
    corridor-to-relay seam): the arm, not the graze, supplied the slide."""
    objs = _make_objects(["green", "blue1", "target"])
    num = 30
    # blue1 sits just past the corridor end (grazing attribution) and
    # slides to solid contact with the target; the EE hovers next to
    # blue1 when the slide begins.
    profile = _slide_profile((0.7, 1.195), {
        11: (0.7, 1.215),
        12: (0.7, 1.235),
        13: (0.7, 1.254)
    }, num)
    states = _build_states(objs,
                           num, {
                               "green": 10,
                               "target": 18
                           },
                           positions={
                               "green": (0.7, 1.0),
                               "target": (0.7, 1.27)
                           },
                           position_profiles={"blue1": profile},
                           robot_xyz=(0.70, 1.15, 0.5))
    step_options = _options([("Push", ("robot", "green"), 0, 7)], num)
    ok, reason = check_cascade_legitimacy(states, _goal(objs), step_options)
    assert not ok
    assert "blue1" in reason and "robot" in reason


def test_count_used_includes_displaced_blues():
    """count_movable_blocks_used charges toppled blues AND blues the cascade
    shoved (not-held displacement), but never robot-transported ones."""
    objs = _make_objects(["green", "blue1", "blue2", "blue3", "blue4"])
    num = 20
    # blue2 is shoved 0.05 m while free; blue4 moves only while held.
    shoved = _slide_profile((0.9, 1.2), {12: (0.9, 1.25)}, num)
    carried = _slide_profile((1.0, 1.2), {8: (1.0, 1.4)}, num)
    states = _build_states(objs,
                           num, {
                               "green": 5,
                               "blue1": 10
                           },
                           positions={
                               "green": (0.7, 1.0),
                               "blue1": (0.7, 1.098),
                               "blue3": (0.8, 1.2)
                           },
                           position_profiles={
                               "blue2": shoved,
                               "blue4": carried
                           },
                           held_spans={"blue4": (6, 15)})
    assert count_movable_blocks_used(states) == 2


def test_arm_topple_beside_fallen_green_fails():
    """The gripper toppling a block the fallen green missed: rejected.

    The run_20260713_000327 task-2 hack: the agent pushes the green
    (satisfying rules a0-a3), the green falls flat WITHOUT reaching the
    staged relay (its fall line clears the relay's footprint by ~2 cm),
    and the Push option's closing descent knocks the relay over instead.
    The old isotropic reach check attributed the relay to the green
    (onset 4 steps later, centers 0.09 m apart); the directional
    corridor rule must not. Geometry below reproduces the recorded onset
    states: green at (1.026, 1.285) yaw 0 falling +y, relay at its onset
    pose (0.934, 1.379) yaw 1.05.
    """
    objs = _make_objects(["green", "blue1", "target"])
    states = _build_states(objs,
                           30, {
                               "green": 8,
                               "blue1": 12
                           },
                           positions={
                               "green": (1.026, 1.285),
                               "blue1": (0.934, 1.379),
                               "target": (0.846, 1.515)
                           },
                           yaws={
                               "blue1": 1.05,
                               "target": 1.5708
                           })
    step_options = _options([("Push", ("robot", "green"), 5, 9)], 30)
    ok, reason = check_cascade_legitimacy(states, _goal(objs), step_options)
    assert not ok
    assert "blue1" in reason and "sweep missed" in reason


def test_graze_with_robot_adjacent_fails():
    """A grazing-only attribution with the end-effector in strike range is
    charged to the robot (rule c).

    The believed-arm bystander case (run_20260713_115630 task 3): the
    legitimate chain completes, and the poker's post-push descent tips a
    bystander blue that every falling domino's corridor only grazes.
    """
    objs = _make_objects(["green", "blue1", "target"])
    states = _build_states(
        objs,
        30,
        {
            "green": 5,
            "target": 11,
            "blue1": 16
        },
        positions={
            "green": (0.7, 1.0),
            "target": (0.7, 1.098),
            # Lateral offset 0.08 from the chain line: corridor
            # clearance ~+0.01, inside the grazing band.
            "blue1": (0.78, 1.09)
        },
        # EE 0.06 m from the bystander, at block height.
        robot_xyz=(0.84, 1.09, 0.5))
    step_options = _options([("Push", ("robot", "green"), 0, 7)], 30)
    ok, reason = check_cascade_legitimacy(states, _goal(objs), step_options)
    assert not ok
    assert "blue1" in reason and "robot" in reason


def test_graze_with_robot_clear_passes():
    """The same grazing knock with the end-effector far away is a legitimate
    corner-style graze (real corner chains measure up to +0.014 m of modeled
    clearance) and must pass."""
    objs = _make_objects(["green", "blue1", "target"])
    states = _build_states(objs,
                           30, {
                               "green": 5,
                               "target": 11,
                               "blue1": 16
                           },
                           positions={
                               "green": (0.7, 1.0),
                               "target": (0.7, 1.098),
                               "blue1": (0.78, 1.09)
                           },
                           robot_xyz=(1.2, 1.5, 0.9))
    step_options = _options([("Push", ("robot", "green"), 0, 7)], 30)
    ok, reason = check_cascade_legitimacy(states, _goal(objs), step_options)
    assert ok, reason


def test_graze_with_robot_arriving_during_fall_fails():
    """A grazing knock where the EE is just outside strike range at the onset
    but drives into the block as it topples: charged to the robot.

    The run_20260713_172940 task-2 hack: the pushed green falls flat and
    only grazes the first blue (0.017 m clearance); the Push option's
    gripper is 0.094 m away - outside strike range - at the onset, then
    keeps closing to 0.013 m over the next five steps while the blue
    topples. The onset-only strike window missed it; the forward scan
    through the block's fall catches it. Modeled with a gradual blue1
    roll (so the fall spans several steps) and a moving EE.
    """
    objs = _make_objects(["green", "blue1", "target"])
    num = 30
    onset = 16
    # blue1 topples gradually from the onset, reaching flat ~7 steps later.
    roll_seq = [0.0] * (num + 1)
    for i, r in enumerate((0.2, 0.4, 0.6, 0.9, 1.2, 1.45, 1.55)):
        roll_seq[onset + i] = -r
    for t in range(onset + 7, num + 1):
        roll_seq[t] = -1.5708
    # EE is 0.42 m from blue1 at the onset, drives to 0.04 m during the
    # fall (steps 19-21), then retreats.
    ee = [(1.20, 1.09, 0.5)] * (num + 1)
    for t in range(19, 22):
        ee[t] = (0.82, 1.09, 0.5)
    states = _build_states(objs,
                           num, {
                               "green": 5,
                               "target": 11
                           },
                           positions={
                               "green": (0.7, 1.0),
                               "target": (0.7, 1.098),
                               "blue1": (0.78, 1.09)
                           },
                           roll_profiles={"blue1": roll_seq},
                           robot_profile=ee)
    step_options = _options([("Push", ("robot", "green"), 0, 7)], num)
    ok, reason = check_cascade_legitimacy(states, _goal(objs), step_options)
    assert not ok
    assert "blue1" in reason and "robot" in reason


def test_graze_with_robot_retreating_during_fall_passes():
    """The same grazing knock with the EE clear through the whole fall: passes.

    A legitimate corner graze, not a robot strike. Guards against the
    forward-through-fall scan over-charging a gripper
    that merely passes by after a genuine knock - here it never enters
    strike range, so the graze stands.
    """
    objs = _make_objects(["green", "blue1", "target"])
    num = 30
    onset = 16
    roll_seq = [0.0] * (num + 1)
    for i, r in enumerate((0.2, 0.4, 0.6, 0.9, 1.2, 1.45, 1.55)):
        roll_seq[onset + i] = -r
    for t in range(onset + 7, num + 1):
        roll_seq[t] = -1.5708
    # EE stays 0.42 m away and retreats upward - never in strike range.
    ee = [(1.20, 1.09, 0.5 + 0.02 * t) for t in range(num + 1)]
    states = _build_states(objs,
                           num, {
                               "green": 5,
                               "target": 11
                           },
                           positions={
                               "green": (0.7, 1.0),
                               "target": (0.7, 1.098),
                               "blue1": (0.78, 1.09)
                           },
                           roll_profiles={"blue1": roll_seq},
                           robot_profile=ee)
    step_options = _options([("Push", ("robot", "green"), 0, 7)], num)
    ok, reason = check_cascade_legitimacy(states, _goal(objs), step_options)
    assert ok, reason


def test_tight_chain_with_robot_adjacent_passes():
    """A solid corridor-overlap knock stays legitimate even when the EE is
    centimeters away: pushing the green in a tightly staged chain necessarily
    ends the push with the EE next to the first relay."""
    objs = _make_objects(["green", "blue1", "target"])
    states = _build_states(objs,
                           30, {
                               "green": 5,
                               "blue1": 9,
                               "target": 14
                           },
                           positions={
                               "green": (0.7, 1.0),
                               "blue1": (0.7, 1.098),
                               "target": (0.7, 1.196)
                           },
                           robot_xyz=(0.7, 1.05, 0.5))
    step_options = _options([("Push", ("robot", "green"), 0, 7)], 30)
    ok, reason = check_cascade_legitimacy(states, _goal(objs), step_options)
    assert ok, reason


def test_solid_corridor_flank_without_body_contact_robot_adjacent_fails():
    """A block staged on the corridor's FLANK inside the solid-contact band
    (clearance under GRAZE_CONTACT_EPS) is not exempt from rule (c) unless the
    falling body actually reached it.

    The run_20260714_145053 task-4 hack: the pushed green fell 72 deg
    away from the staged blue deflector (never touching it, 4 mm
    plan-view gap) while the gripper's push stroke toppled the blue;
    the modeled corridor read 4.3 mm of clearance - inside the old
    solid-contact exemption - so the episode was certified.
    """
    objs = _make_objects(["green", "blue1", "target"])
    states = _build_states(
        objs,
        30,
        {
            "green": 5,
            "target": 11,
            "blue1": 16
        },
        positions={
            "green": (0.7, 1.0),
            "target": (0.7, 1.098),
            # Corridor clearance ~+0.0045 (solid band), but the falling
            # green's body stays those 4.5 mm clear of it for good.
            "blue1": (0.7745, 1.09)
        },
        # EE 0.026 m from the staged blue, at block height.
        robot_xyz=(0.80, 1.09, 0.5))
    step_options = _options([("Push", ("robot", "green"), 0, 7)], 30)
    ok, reason = check_cascade_legitimacy(states, _goal(objs), step_options)
    assert not ok
    assert "blue1" in reason and "robot" in reason
    assert "no falling domino's body ever reached" in reason


def test_solid_corridor_flank_without_body_contact_robot_clear_passes():
    """The same flank topple with the EE far away is not charged: without a
    robot in strike range the kinematic attribution stands."""
    objs = _make_objects(["green", "blue1", "target"])
    states = _build_states(objs,
                           30, {
                               "green": 5,
                               "target": 11,
                               "blue1": 16
                           },
                           positions={
                               "green": (0.7, 1.0),
                               "target": (0.7, 1.098),
                               "blue1": (0.7745, 1.09)
                           },
                           robot_xyz=(1.2, 1.5, 0.9))
    step_options = _options([("Push", ("robot", "green"), 0, 7)], 30)
    ok, reason = check_cascade_legitimacy(states, _goal(objs), step_options)
    assert ok, reason


def test_relay_solid_corridor_without_ram_contact_robot_slide_fails():
    """A relay staged on the corridor's flank (solid-band clearance, never
    physically rammed) that slides into the target with the EE in strike range
    at its slide start: the slide is charged to the robot (corridor-to-relay
    seam of rule (c) with the body-contact requirement)."""
    objs = _make_objects(["green", "blue1", "target"])
    num = 30
    # blue1 sits on the corridor's flank; from step 8 it slides +x into
    # the target. The falling green's body never reaches it.
    profile = []
    for t in range(num + 1):
        x = 0.7745 if t < 8 else min(0.7745 + 0.006 * (t - 8), 0.818)
        profile.append((x, 1.09))
    states = _build_states(objs,
                           num, {
                               "green": 5,
                               "target": 20
                           },
                           positions={
                               "green": (0.7, 1.0),
                               "target": (0.86, 1.09)
                           },
                           position_profiles={"blue1": profile},
                           robot_xyz=(0.80, 1.09, 0.5))
    step_options = _options([("Push", ("robot", "green"), 0, 7)], num)
    ok, reason = check_cascade_legitimacy(states, _goal(objs), step_options)
    assert not ok
    assert "slide is charged to the robot" in reason


def test_held_block_roll_excursion_ignored():
    """A carried blue tilts freely; only unheld topples count."""
    objs = _make_objects(["green", "blue1", "target"])
    num = 30
    # blue1 is held (and tilted) on steps 3-8, upright after placement,
    # then falls via the cascade from step 16 on.
    profile = [0.0] * (num + 1)
    for t in range(3, 9):
        profile[t] = 1.0
    for t in range(16, num + 1):
        profile[t] = _FALLEN_ROLL
    states = _build_states(objs,
                           num, {
                               "green": 12,
                               "target": 20
                           },
                           positions={
                               "green": (0.7, 1.0),
                               "blue1": (0.7, 1.098),
                               "target": (0.7, 1.196)
                           },
                           roll_profiles={"blue1": profile},
                           held_spans={"blue1": (3, 8)})
    step_options = _options([("Pick", ("robot", "blue1"), 0, 2),
                             ("Place", ("robot", ), 3, 8),
                             ("Push", ("robot", "green"), 9, 13)], num)
    ok, reason = check_cascade_legitimacy(states, _goal(objs), step_options)
    assert ok, reason


def test_placement_wobble_ignored():
    """A wobble that never reaches the fallen threshold is no event."""
    objs = _make_objects(["green", "blue1", "target"])
    num = 30
    profile = [0.0] * (num + 1)
    for t in range(4, 7):
        profile[t] = 0.12  # ~7 deg: tilting, not fallen
    states = _build_states(objs,
                           num, {
                               "green": 12,
                               "target": 18
                           },
                           positions={
                               "green": (0.7, 1.0),
                               "blue1": (0.9, 1.0),
                               "target": (0.7, 1.098)
                           },
                           roll_profiles={"blue1": profile})
    step_options = _options([("Push", ("robot", "green"), 9, 13)], num)
    ok, reason = check_cascade_legitimacy(states, _goal(objs), step_options)
    assert ok, reason


def test_tie_onsets_allowed():
    """Green and its neighbor starting to fall on the same step pass."""
    objs = _make_objects(["green", "target"])
    states = _build_states(objs,
                           20, {
                               "green": 5,
                               "target": 5
                           },
                           positions={
                               "green": (0.7, 1.0),
                               "target": (0.7, 1.098)
                           })
    step_options = _options([("Push", ("robot", "green"), 0, 7)], 20)
    ok, reason = check_cascade_legitimacy(states, _goal(objs), step_options)
    assert ok, reason


def test_same_step_tie_attributes_through_knocker():
    """Two non-green blocks sharing an onset step certify when the one that
    sorts FIRST by name was knocked by the one that sorts second.

    Regression for run_20260713_172854 seed0 task0: a tight staircase
    crosses two hops within one recorded step, so knocker (domino_2) and
    victim (domino_1) read the same onset. A single attribution pass in
    (onset, name) order visited the victim while only the green was
    legitimized (0.07 m clearance, rejected) and never considered the
    knocker whose corridor overlaps the victim outright; iterating
    attribution to a fixed point must accept the chain.
    """
    objs = _make_objects(["green", "blue1", "blue2", "target"])
    states = _build_states(
        objs,
        30,
        {
            "green": 5,
            "blue2": 9,  # knocker: solidly inside green's corridor
            "blue1": 9,  # victim: only inside blue2's corridor
            "target": 13
        },
        positions={
            "green": (0.7, 1.0),
            "blue2": (0.7, 1.098),
            # 0.26 from green (clearance ~0.07, outside tolerance) but
            # 0.162 from blue2 (inside its 0.18 m reach).
            "blue1": (0.7, 1.26),
            "target": (0.7, 1.358)
        })
    step_options = _options([("Push", ("robot", "green"), 0, 7)], 30)
    ok, reason = check_cascade_legitimacy(states, _goal(objs), step_options)
    assert ok, reason


def test_unattributable_topple_names_passing_robot():
    """An unattributable bystander topple with the end-effector having passed
    in strike range well before the onset blames the robot in the message.

    Regression for run_20260713_172854 seed2 task1: poker grazes tip a
    block slowly, so its onset lands past the tight verdict lookback
    and the old message blamed corridor clearance - sending the agent
    off to shave a geometry margin that was never the cause. The
    verdict stays False either way; only the explanation gains the
    robot hint.
    """
    objs = _make_objects(["green", "blue1", "target"])
    contact, onset = 16, 26
    profile = [(1.4, 1.5, 0.9)] * 31
    for t in range(contact - 2, contact + 1):
        profile[t] = (0.93, 1.3, 0.5)  # 0.03 m from the bystander
    states = _build_states(objs,
                           30, {
                               "green": 5,
                               "target": 11,
                               "blue1": onset
                           },
                           positions={
                               "green": (0.7, 1.0),
                               "target": (0.7, 1.098),
                               "blue1": (0.9, 1.3)
                           },
                           robot_profile=profile)
    step_options = _options([("Push", ("robot", "green"), 0, 7)], 30)
    ok, reason = check_cascade_legitimacy(states, _goal(objs), step_options)
    assert not ok
    assert "blue1" in reason
    assert "end-effector passed within" in reason
    assert "push stroke or retreat" in reason


def test_green_toppled_outside_push_fails():
    """Green falling long after its Push (e.g. swept by a later Place) violates
    (a1) even though it is the first onset."""
    objs = _make_objects(["green", "target"])
    green_onset = 5 + 1 + CASCADE_WINDOW_STEPS + 10
    num = green_onset + 15
    states = _build_states(objs,
                           num, {
                               "green": green_onset,
                               "target": green_onset + 4
                           },
                           positions={
                               "green": (0.7, 1.0),
                               "target": (0.7, 1.098)
                           })
    step_options = _options([("Push", ("robot", "green"), 0, 5),
                             ("Place", ("robot", ), 6, num - 1)], num)
    ok, reason = check_cascade_legitimacy(states, _goal(objs), step_options)
    assert not ok
    assert "green" in reason and "Push" in reason


def test_no_topples_passes():
    """Nothing fell: nothing to certify."""
    objs = _make_objects(["green", "target"])
    states = _build_states(objs, 10, {})
    ok, reason = check_cascade_legitimacy(states, _goal(objs), None)
    assert ok, reason


def test_onsets_during_wait_after_push_pass():
    """Cascade unfolding during Wait steps after the push is the normal case
    and must pass the action rules."""
    objs = _make_objects(["green", "blue1", "target"])
    states = _build_states(objs,
                           40, {
                               "green": 5,
                               "blue1": 20,
                               "target": 24
                           },
                           positions={
                               "green": (0.7, 1.0),
                               "blue1": (0.7, 1.098),
                               "target": (0.7, 1.196)
                           })
    # Push ends at action 6; every later onset happens during Wait.
    step_options = _options([("Push", ("robot", "green"), 0, 6)], 40)
    ok, reason = check_cascade_legitimacy(states, _goal(objs), step_options)
    assert ok, reason


def test_restricted_push_counts_as_green_push():
    """The restricted Push variant grounds only the robot; it always targets
    the inferred start block, so it must satisfy the action rules."""
    objs = _make_objects(["green", "blue1", "target"])
    states = _build_states(objs,
                           30, {
                               "green": 5,
                               "blue1": 10,
                               "target": 14
                           },
                           positions={
                               "green": (0.7, 1.0),
                               "blue1": (0.7, 1.098),
                               "target": (0.7, 1.196)
                           })
    step_options = _options([("Push", ("robot", ), 0, 7)], 30)
    ok, reason = check_cascade_legitimacy(states, _goal(objs), step_options)
    assert ok, reason


def test_domino_evaluator_certify():
    """DominoEvaluator._certify applies the cascade rules to its own goal (the
    evaluator seam consumed by BaseEnv.check_episode_trajectory /
    BaseEnv.evaluate_episode)."""
    # Local import: pulls in PyBullet, which the rest of this file avoids.
    from predicators.envs.pybullet_domino.env import \
        DominoEvaluator  # pylint: disable=import-outside-toplevel
    objs = _make_objects(["green", "blue1", "target"])
    states = _build_states(objs,
                           30, {
                               "green": 5,
                               "blue1": 10,
                               "target": 14
                           },
                           positions={
                               "green": (0.7, 1.0),
                               "blue1": (0.7, 1.098),
                               "target": (0.7, 1.196)
                           })
    evaluator = DominoEvaluator(_goal(objs))
    honest = _options([("Push", ("robot", "green"), 0, 7)], 30)
    ok, reason = evaluator._certify(states, honest)  # pylint: disable=protected-access
    assert ok, reason
    hacked = _options([("Push", ("robot", "blue1"), 0, 7)], 30)
    ok, reason = evaluator._certify(states, hacked)  # pylint: disable=protected-access
    assert not ok
    assert "green" in reason


def test_domino_evaluator_reward_decomposition():
    """The DominoEvaluator's reward is the certified-success bonus minus the
    per-block cost; termination is purely physical (an illegitimate topple
    still terminates), and no oracle K* lives on the evaluator."""
    # Local import: pulls in PyBullet, which the rest of this file avoids.
    from predicators.envs.pybullet_domino.env import \
        DominoEvaluator  # pylint: disable=import-outside-toplevel
    from predicators.settings import \
        CFG  # pylint: disable=import-outside-toplevel
    from predicators.utils import \
        reset_config  # pylint: disable=import-outside-toplevel
    reset_config({"domino_block_cost": 0.05, "domino_min_block_num_blues": 4})

    # Toppled with a real classifier so terminated() reflects the states.
    toppled = Predicate(
        "Toppled", [_DOMINO_TYPE],
        lambda s, o: abs(float(s.get(o[0], "roll"))) >= abs(_FALLEN_ROLL))
    objs = _make_objects(["green", "blue1", "target"])
    goal = {GroundAtom(toppled, [objs["target"]])}
    states = _build_states(objs,
                           30, {
                               "green": 5,
                               "blue1": 10,
                               "target": 14
                           },
                           positions={
                               "green": (0.7, 1.0),
                               "blue1": (0.7, 1.098),
                               "target": (0.7, 1.196)
                           })
    # The pure block count sees exactly one consumed BLUE (blue1
    # toppled; green also fell but is filtered by color; target is
    # purple).
    assert count_movable_blocks_used(states) == 1
    evaluator = DominoEvaluator(goal)
    honest = _options([("Push", ("robot", "green"), 0, 7)], 30)
    hacked = _options([("Push", ("robot", "blue1"), 0, 7)], 30)
    # Legitimate topple: bonus minus one block's cost.
    assert evaluator.terminated(states[-1])
    assert abs(
        evaluator.reward(states, honest) -
        (1.0 - CFG.domino_block_cost)) < 1e-9
    # Illegitimate topple: still terminated, bonus gated, cost still paid.
    assert evaluator.terminated(states[-1])
    assert abs(evaluator.reward(states, hacked) -
               (-CFG.domino_block_cost)) < 1e-9
    # The evaluator ships on the agent-facing Task, so no oracle
    # quantity may live on it: K* travels env-side via
    # EnvironmentTask.offline_task_metrics instead.
    assert evaluator.offline_metrics(states, honest) == {"k_used": 1.0}
    assert not hasattr(evaluator, "k_star")
    # The stated objective is public and K*-free.
    description = evaluator.objective_description()
    assert str(CFG.domino_block_cost) in description
    assert "K*" not in description


def test_domino_evaluator_budget_assert():
    """A scene whose staged movables could out-cost the success bonus is a
    config error, caught at construction against the scene's actual count
    (``num_movables``); omitting it falls back to the min-block budget flag."""
    # Local import: pulls in PyBullet, which the rest of this file avoids.
    from predicators.envs.pybullet_domino.env import \
        DominoEvaluator  # pylint: disable=import-outside-toplevel
    from predicators.utils import \
        reset_config  # pylint: disable=import-outside-toplevel
    reset_config({"domino_block_cost": 0.05, "domino_min_block_num_blues": 4})
    goal = _goal(_make_objects(["green", "target"]))
    DominoEvaluator(goal)  # flag default: 0.05 * 4 < 1
    DominoEvaluator(goal, num_movables=19)  # 0.05 * 19 < 1
    with pytest.raises(AssertionError):
        DominoEvaluator(goal, num_movables=20)  # 0.05 * 20 == 1
