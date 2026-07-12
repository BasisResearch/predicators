"""Tests for the min-block cascade legitimacy certificate.

Pure-State tests (no PyBullet): trajectories are synthesized as step-
function roll profiles, so each domino's topple onset is exactly the
step where its roll jumps past the fallen threshold.
"""

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from predicators.envs.pybullet_domino.cascade_certificate import \
    CASCADE_WINDOW_STEPS, check_cascade_legitimacy, \
    count_movable_blocks_used
from predicators.envs.pybullet_domino.components.domino_component import \
    DominoComponent
from predicators.structs import GroundAtom, Object, Predicate, State, Type

_DOMINO_TYPE = Type("domino",
                    ["x", "y", "z", "yaw", "roll", "r", "g", "b", "is_held"])
_TOPPLED = Predicate("Toppled", [_DOMINO_TYPE], lambda s, o: False)

_GREEN = DominoComponent.start_domino_color
_BLUE = DominoComponent.domino_color
_PURPLE = DominoComponent.target_domino_color

_FALLEN_ROLL = 0.6  # past fallen_threshold (pi/6 ~= 0.524)

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
) -> List[State]:
    """Build a trajectory of ``num_steps + 1`` states.

    ``onsets[name] = t`` gives a step-function roll: 0 before ``t``,
    fallen after. ``roll_profiles`` overrides the roll sequence for a
    domino entirely; ``position_profiles`` gives a per-step (x, y)
    sequence for a domino (e.g. a relocation); ``held_spans[name] =
    (a, b)`` sets is_held on state indices [a, b].
    """
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
            data[obj] = np.array(
                [x, y, 0.475, 0.0, roll, *_color_for(name)[:3], held],
                dtype=np.float32)
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
        profile[t] = 0.35  # ~20 deg: tilting, not fallen
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
        lambda s, o: float(s.get(o[0], "roll")) >= _FALLEN_ROLL)
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
    # The pure block count sees exactly one toppled BLUE in the final
    # state (green also fell but is filtered by color; target is purple).
    assert count_movable_blocks_used(states[-1]) == 1
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
