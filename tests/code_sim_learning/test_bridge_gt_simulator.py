"""Test the bridge GT hybrid simulator's glue-cure-latch residual.

The curing gates are sigmoid-softened so the residual is differentiable
in the threshold parameters, and the cure counter evolves as ``prog = w
* (cure + 1)``, which fixed-points at ``w / (1 - w)``. That makes the
latch EXQUISITELY sensitive to the gate sharpness: with the shared
``SOFT_EPS`` (0.02, sized for 5-15 cm thresholds) on these mm-scale
windows, a geometrically PERFECT butt joint capped at w ~= 0.33 (fixed
point 0.49 against a latch threshold of 25), so welding was impossible
in the simulator while trivial in the env -- and every agent planning
against the simulator dead-ended on ``Attached``. These tests pin the
latch behavior directly on hand-built states: an in-window glued joint
must latch in about ``cure_threshold`` steps, and an out-of-window one
must never latch.
"""

import numpy as np

from predicators import utils
from predicators.code_sim_learning.utils import apply_rules, \
    has_physics_rules, merge_updates
from predicators.ground_truth_models import get_gt_simulator
from predicators.structs import Object, State, Type

_GLUE_FACES = ("top", "end_a", "end_b")
_ATTACH_SLOTS = ("top", "bottom", "end_a", "end_b")
_BLOCK_TYPE = Type("block",
                   ["x", "y", "z", "roll", "pitch", "yaw", "is_held"] +
                   [f"glue_{f}"
                    for f in _GLUE_FACES] + [f"cure_{f}"
                                             for f in _GLUE_FACES] +
                   [f"attached_{s}" for s in _ATTACH_SLOTS])
# Fixed block-index order matching gt_simulator._block_index.
_BLOCK_IDX = {"leg0": 0, "leg1": 1, "span0": 2, "span1": 3, "span2": 4}
_TABLE_Z = 0.4
_SPAN_HALF = (0.05, 0.025, 0.025)


def _make_block(name, x, y, z):
    obj = Object(name, _BLOCK_TYPE)
    feats = {f: 0.0 for f in _BLOCK_TYPE.feature_names}
    feats.update(x=x, y=y, z=z)
    for slot in _ATTACH_SLOTS:
        feats[f"attached_{slot}"] = -1.0
    return obj, np.array([feats[f] for f in _BLOCK_TYPE.feature_names],
                         dtype=np.float32)


def _make_state(blocks):
    return State(dict(blocks))


def _bridge_sim():
    utils.reset_config({"env": "pybullet_bridge", "seed": 0})
    rules, specs, _ = get_gt_simulator("pybullet_bridge")
    params = {s.name: s.init_value for s in specs}
    return rules, params


def _roll_until_latched(state, rules, params, blk, slot, max_steps):
    """Apply the residual rules until ``blk``'s ``slot`` latches; return the
    step index or None."""
    for step in range(max_steps):
        updates = apply_rules(state, rules, params)
        state = merge_updates(state, updates)
        if state.get(blk, f"attached_{slot}") >= 0:
            return step, state
    return None, state


_BOTTLE_TYPE = Type("bottle", ["x", "y", "z", "rot", "is_held"])


def test_sustained_wetting_matches_env():
    """The sim's glue rule requires the same sustained dwell as the env: a tip
    parked at a dab wets on the WET_STREAK_STEPS-th consecutive step, and an
    interrupted streak resets -- so a drive-by crossing of the apply radius can
    never wet a face in the sandbox either."""
    from predicators.ground_truth_models.bridge.gt_simulator import \
        WET_STREAK_STEPS  # pylint: disable=import-outside-toplevel
    rules, params = _bridge_sim()
    span0, span0_feats = _make_block("span0", 0.6, 1.2,
                                     _TABLE_Z + _SPAN_HALF[2])
    bottle = Object("bottle", _BOTTLE_TYPE)
    # Held bottle with its tip at span0's end_b dab (above the face's
    # top edge: z + half_z + dab margin, tip = center - half height).
    dab_z = _TABLE_Z + 2 * _SPAN_HALF[2] + 0.005
    bottle_feats = np.array([0.6 + _SPAN_HALF[0], 1.2, dab_z + 0.03, 0.0, 1.0],
                            dtype=np.float32)
    state = State({span0: span0_feats, bottle: bottle_feats})

    for step in range(WET_STREAK_STEPS):
        wet = state.get(span0, "glue_end_b")
        assert wet <= 0.5, f"wet after only {step} steps"
        state = merge_updates(state, apply_rules(state, rules, params))
    assert state.get(span0, "glue_end_b") > 0.5

    # Interrupted streak: two in-range steps, one out-of-range, resets.
    state = State({span0: span0_feats.copy(), bottle: bottle_feats.copy()})
    for _ in range(WET_STREAK_STEPS - 1):
        state = merge_updates(state, apply_rules(state, rules, params))
    assert 0.0 < state.get(span0, "glue_end_b") <= 0.5
    state.set(bottle, "z", dab_z + 0.2)
    state = merge_updates(state, apply_rules(state, rules, params))
    assert state.get(span0, "glue_end_b") == 0.0


def test_bridge_gt_simulator_loads():
    """The factory registry resolves pybullet_bridge to the FO simulator."""
    rules, params = _bridge_sim()
    assert [r.__name__ for r in rules] == \
        ["_glue_application", "_curing", "_welding"]
    assert params["cure_threshold"] == 25.0
    # The welding rule acts through the physics-command channel, so the
    # fitting stack must route it to the rollout objective.
    assert has_physics_rules(rules)


def test_butt_joint_cures_and_latches():
    """A glued, butted, resting span pair latches in ~cure_threshold steps.

    3 mm joint gap: comfortably inside the [-10 mm, +12 mm] projection
    window, matching what the oracle place sampler produces.
    """
    rules, params = _bridge_sim()
    span0, arr0 = _make_block("span0", 0.60, 1.14, _TABLE_Z + _SPAN_HALF[2])
    span1, arr1 = _make_block("span1", 0.703, 1.14, _TABLE_Z + _SPAN_HALF[2])
    state = _make_state([(span0, arr0), (span1, arr1)])
    state.set(span0, "glue_end_b", 1.0)
    latch_step, state = _roll_until_latched(state, rules, params, span0,
                                            "end_b", 40)
    assert latch_step is not None, "in-window glued joint never latched"
    # The env's hard counter latches on step cure_threshold; the soft
    # gates may cost a few extra steps but not more.
    assert latch_step <= params["cure_threshold"] + 5
    assert state.get(span0, "attached_end_b") == _BLOCK_IDX["span1"]
    assert state.get(span1, "attached_end_a") == _BLOCK_IDX["span0"]
    # The latch consumes the glue.
    assert state.get(span0, "glue_end_b") == 0.0


def test_stacked_top_joint_cures_and_latches():
    """A glued top face with a block resting on it latches too (the upward-face
    mate path)."""
    rules, params = _bridge_sim()
    span0, arr0 = _make_block("span0", 0.60, 1.14, _TABLE_Z + _SPAN_HALF[2])
    span1, arr1 = _make_block("span1", 0.605, 1.145,
                              _TABLE_Z + 3 * _SPAN_HALF[2])
    state = _make_state([(span0, arr0), (span1, arr1)])
    state.set(span0, "glue_top", 1.0)
    latch_step, state = _roll_until_latched(state, rules, params, span0, "top",
                                            40)
    assert latch_step is not None, "in-window stacked joint never latched"
    assert latch_step <= params["cure_threshold"] + 5
    assert state.get(span0, "attached_top") == _BLOCK_IDX["span1"]
    assert state.get(span1, "attached_bottom") == _BLOCK_IDX["span0"]


def test_out_of_window_joint_never_latches():
    """A glued joint with a 2 cm gap (outside the +12 mm window) must not
    cure -- the gates gate, they don't just delay."""
    rules, params = _bridge_sim()
    span0, arr0 = _make_block("span0", 0.60, 1.14, _TABLE_Z + _SPAN_HALF[2])
    span1, arr1 = _make_block("span1", 0.72, 1.14, _TABLE_Z + _SPAN_HALF[2])
    state = _make_state([(span0, arr0), (span1, arr1)])
    state.set(span0, "glue_end_b", 1.0)
    latch_step, state = _roll_until_latched(state, rules, params, span0,
                                            "end_b", 80)
    assert latch_step is None
    assert state.get(span0, "cure_end_b") < 1.0
