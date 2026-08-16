"""Ground-truth simulator program for pybullet_bridge (glue construction)
residual dynamics -- fully-observable variant.

The residual slow processes the base rigid-body sim cannot model:

1. Glue application: while the bottle is held with its tip near a
   face's dab point, that face's wet-glue flag flips on.
2. Curing: while a wet face is in aligned resting contact with another
   block (neither held), that joint's ``cure_*`` counter ticks; at
   ``cure_threshold`` the joint irreversibly latches -- the glue is
   consumed and both blocks record the partner in their ``attached_*``
   slot.
3. Welding: every latched pair is re-emitted as an ``Attach`` physics
   command each step, so the pair moves as one rigid body. The base
   sim's own feature-to-weld sync is gated off in base-sim mode -- the
   kinematic consequence of the latch is this program's job, which is
   also what routes fitting through the env-in-the-loop rollout
   objective (``has_physics_rules``).

Alignment gates are softened with sigmoids so the residual is
differentiable in the threshold parameters (for the Levenberg-Marquardt
Jacobian); held gates and the discrete attachment latch stay hard.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from predicators.code_sim_learning.commands import CommandBuffer
from predicators.code_sim_learning.fit_space import ParamSpec
from predicators.code_sim_learning.utils import Params, ResidualUpdate, \
    objs_by_type, sigmoid
from predicators.ground_truth_models import GroundTruthSimulatorFactory
from predicators.settings import CFG
from predicators.structs import Object, State

# Physical defaults matching pybullet_bridge.py.
CURE_THRESHOLD = 25.0
APPLY_GLUE_RADIUS = 0.02
STACK_ALIGN_TOL = 0.025
LATERAL_PERP_TOL = 0.03
SEAT_X_WINDOW = 0.045
SEAT_Y_TOL = 0.035
BOTTLE_HALF_H = 0.03
DAB_MARGIN = 0.005
# ONE block shape (long axis = local x); a "leg" is the same block
# stood on end (pitch = -pi/2, local +x up), so its world-top face is
# its local end_b face.
BLOCK_HALF = (0.05, 0.025, 0.025)
GLUE_FACES = ("top", "end_a", "end_b")
ATTACH_SLOTS = ("top", "bottom", "end_a", "end_b")
# Sigmoid sharpness for the mm-scale alignment gates below. The shared
# ``SOFT_EPS`` (0.02, sized for 5-15 cm thresholds) is 2-20x the gate
# windows here, which capped a PERFECT butt joint's weight at ~0.33 and
# a perfect seat at ~0.77 -- and since ``prog = w * (cure + 1)``
# fixed-points at ``w / (1 - w)``, curing could NEVER reach the latch
# threshold (25) for any geometry. At 1 mm an in-window joint saturates
# to w ~= 1 (latching in ~cure_threshold steps, matching the env's hard
# counter) while the boundary keeps a +-4 mm differentiable band for
# the fitting Jacobian; the sim's effective window is ~3 mm inside the
# env's hard window -- conservative, never the reverse.
GATE_EPS = 0.001
# Local (normal axis, sign) per face / attachment slot.
FACE_AXES = {"top": (2, 1.0), "end_a": (0, -1.0), "end_b": (0, 1.0)}
SLOT_AXES = {**FACE_AXES, "bottom": (2, -1.0)}
_COS45 = float(np.cos(np.pi / 4))


def _rmat(state: State, blk: Object) -> np.ndarray:
    """World-from-local rotation, R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    cl = float(np.cos(state.get(blk, "roll")))
    sl = float(np.sin(state.get(blk, "roll")))
    cp = float(np.cos(state.get(blk, "pitch")))
    sp = float(np.sin(state.get(blk, "pitch")))
    cy = float(np.cos(state.get(blk, "yaw")))
    sy = float(np.sin(state.get(blk, "yaw")))
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cl, -sl], [0.0, sl, cl]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return rz @ ry @ rx


def _stands(state: State, blk: Object) -> bool:
    """Long axis vertical (the leg pose; spans lie flat)."""
    return abs(state.get(blk, "pitch")) > np.pi / 4


def _half(state: State, blk: Object) -> Tuple[float, float, float]:
    """World-axis-aligned half extents at the block's orientation family
    (standing swaps the long axis into z)."""
    if _stands(state, blk):
        return (BLOCK_HALF[2], BLOCK_HALF[1], BLOCK_HALF[0])
    return BLOCK_HALF


def _face_dir(state: State, blk: Object,
              face: str) -> Tuple[float, float, float]:
    axis, sign = FACE_AXES[face]
    n = sign * _rmat(state, blk)[:, axis]
    return (float(n[0]), float(n[1]), float(n[2]))


def _dab_point(state: State, blk: Object,
               face: str) -> Tuple[float, float, float]:
    """Mirror of the env's dab geometry: above the center of an upward face,
    above the top edge of a vertical face."""
    axis, sign = FACE_AXES[face]
    rmat = _rmat(state, blk)
    pos = np.array([float(state.get(blk, f)) for f in ("x", "y", "z")])
    n = sign * rmat[:, axis]
    center = pos + n * BLOCK_HALF[axis]
    if n[2] > _COS45:
        return (float(center[0]), float(center[1]),
                float(center[2] + DAB_MARGIN))
    span_axes = [i for i in range(3) if i != axis]
    v_half = max(abs(rmat[2, i]) * BLOCK_HALF[i] for i in span_axes)
    return (float(center[0]), float(center[1]),
            float(pos[2] + v_half + DAB_MARGIN))


def _top_mate_weight(state: State, other: Object, blk: Object,
                     params: Params) -> float:
    """Soft weight that ``other`` rests on ``blk``'s top face (covers both the
    leg-stack and the span-seat geometry)."""
    if state.get(other, "is_held") > 0.5 or state.get(blk, "is_held") > 0.5:
        return 0.0
    hz_b = _half(state, blk)[2]
    hz_o = _half(state, other)[2]
    dz = float(state.get(other,
                         "z")) - (float(state.get(blk, "z")) + hz_b + hz_o)
    if abs(dz) >= 0.02:
        return 0.0
    dx = float(state.get(other, "x")) - float(state.get(blk, "x"))
    dy = float(state.get(other, "y")) - float(state.get(blk, "y"))
    if _stands(state, other):
        # Leg on leg: circular xy alignment.
        return sigmoid(
            (params["stack_align_tol"] - float(np.hypot(dx, dy))) / GATE_EPS)
    # Span seated on leg: the leg under the span's footprint.
    x_w = sigmoid((params["seat_x_window"] - abs(dx)) / GATE_EPS)
    y_w = sigmoid((SEAT_Y_TOL - abs(dy)) / GATE_EPS)
    return x_w * y_w


def _end_mate_weight(state: State, blk: Object, face: str, other: Object,
                     params: Params) -> float:
    """Soft weight that ``other`` butts against ``blk``'s end face."""
    if state.get(other, "is_held") > 0.5 or state.get(blk, "is_held") > 0.5:
        return 0.0
    dx_dir, dy_dir, _ = _face_dir(state, blk, face)
    dx = float(state.get(other, "x")) - float(state.get(blk, "x"))
    dy = float(state.get(other, "y")) - float(state.get(blk, "y"))
    dz = float(state.get(other, "z")) - float(state.get(blk, "z"))
    if abs(dz) >= 0.015:
        return 0.0
    proj = dx * dx_dir + dy * dy_dir
    perp = abs(-dx * dy_dir + dy * dx_dir)
    ext = BLOCK_HALF[FACE_AXES[face][0]] + _half(state, other)[0]
    proj_w = sigmoid((0.012 - (proj - ext)) / GATE_EPS) * \
        sigmoid(((proj - ext) + 0.01) / GATE_EPS)
    perp_w = sigmoid((params["lateral_perp_tol"] - perp) / GATE_EPS)
    return proj_w * perp_w


def _find_mate(state: State, blocks: List[Object], blk: Object, face: str,
               params: Params) -> Tuple[Optional[Object], float]:
    n_z = _face_dir(state, blk, face)[2]
    best: Optional[Object] = None
    best_w = 0.0
    for other in blocks:
        if other == blk:
            continue
        if n_z > _COS45:
            # Upward face: the mate rests on it.
            w = _top_mate_weight(state, other, blk, params)
        elif abs(n_z) < _COS45:
            # Vertical face: the mate butts against it.
            w = _end_mate_weight(state, blk, face, other, params)
        else:
            w = 0.0  # downward faces have no reachable mate
        if w > best_w:
            best, best_w = other, w
    return best, best_w


def _mate_slot_for(state: State, blk: Object, face: str, mate: Object) -> str:
    """The mate's attachment slot whose world normal most opposes the wet
    face's."""
    n = np.array(_face_dir(state, blk, face))
    rmat = _rmat(state, mate)
    best_slot, best_dot = "bottom", np.inf
    for slot, (axis, sign) in SLOT_AXES.items():
        dot = float(n @ (sign * rmat[:, axis]))
        if dot < best_dot:
            best_slot, best_dot = slot, dot
    return best_slot


def _block_index(blocks: List[Object]) -> Dict[str, int]:
    del blocks  # the index is fixed by name, not task contents
    # Fixed name order matching the env: leg0, leg1, span0..span2.
    full = [f"leg{i}" for i in range(2)] + [f"span{i}" for i in range(3)]
    return {name: i for i, name in enumerate(full)}


def _glue_application(state: State, updates: ResidualUpdate,
                      params: Params) -> ResidualUpdate:
    """Wet the single nearest face within the bottle tip's radius."""
    objs = objs_by_type(state)
    blocks = objs.get("block", [])
    bottles = objs.get("bottle", [])
    # Carry existing glue by default.
    for blk in blocks:
        for face in GLUE_FACES:
            updates.setdefault(blk, {})[f"glue_{face}"] = float(
                state.get(blk, f"glue_{face}"))
    held = [b for b in bottles if state.get(b, "is_held") > 0.5]
    if not held:
        return updates
    bottle = held[0]
    tip = np.array([
        float(state.get(bottle, "x")),
        float(state.get(bottle, "y")),
        float(state.get(bottle, "z")) - BOTTLE_HALF_H
    ])
    best, best_d = None, float(params["apply_glue_radius"])
    for blk in blocks:
        for face in GLUE_FACES:
            if state.get(blk, f"glue_{face}") > 0.5:
                continue
            if state.get(blk, f"attached_{face}") >= 0:
                continue
            d = float(
                np.linalg.norm(tip - np.array(_dab_point(state, blk, face))))
            if d < best_d:
                best, best_d = (blk, face), d
    if best is not None:
        blk, face = best
        updates.setdefault(blk, {})[f"glue_{face}"] = 1.0
    return updates


def _curing(state: State, updates: ResidualUpdate,
            params: Params) -> ResidualUpdate:
    """Tick each wet aligned joint's counter; latch attachment at the threshold
    (hard latch; the counter accumulation is soft-gated)."""
    objs = objs_by_type(state)
    blocks = objs.get("block", [])
    idx_of = _block_index(blocks)

    # Carry attachment slots by default.
    for blk in blocks:
        for slot in ATTACH_SLOTS:
            updates.setdefault(blk, {})[f"attached_{slot}"] = float(
                state.get(blk, f"attached_{slot}"))
        for face in GLUE_FACES:
            updates.setdefault(blk, {})[f"cure_{face}"] = 0.0

    for blk in blocks:
        for face in GLUE_FACES:
            # Attachment reads go through ``updates`` (pre-filled by
            # the carry loop) so a latch earlier in this SAME step
            # blocks a conflicting one, mirroring the env's
            # live-attribute curing loop.
            if updates[blk][f"attached_{face}"] >= 0:
                # Latched joints keep their final counter value.
                updates[blk][f"cure_{face}"] = float(
                    state.get(blk, f"cure_{face}"))
                continue
            wet = updates[blk].get(f"glue_{face}",
                                   float(state.get(blk, f"glue_{face}")))
            if wet <= 0.5:
                continue
            mate, w = _find_mate(state, blocks, blk, face, params)
            prog = w * (float(state.get(blk, f"cure_{face}")) + 1.0)
            updates[blk][f"cure_{face}"] = prog
            if mate is not None and prog >= params["cure_threshold"]:
                mate_slot = _mate_slot_for(state, blk, face, mate)
                if updates[mate][f"attached_{mate_slot}"] >= 0:
                    continue
                updates[blk][f"attached_{face}"] = float(idx_of[mate.name])
                updates[mate][f"attached_{mate_slot}"] = \
                    float(idx_of[blk.name])
                updates[blk][f"glue_{face}"] = 0.0
    return updates


def _welding(state: State, updates: ResidualUpdate, params: Params,
             cmds: CommandBuffer) -> ResidualUpdate:
    """Re-emit the rigid weld for every latched attachment.

    The base sim deliberately does NOT know that ``attached_*`` features
    mean a weld (its feature-to-constraint sync is gated off in base-sim
    mode), so the kinematic consequence of the latch is this rule's job:
    emit the ``Attach`` physics command for each latched pair, every
    step, per the re-emit-to-persist command contract. Runs after
    ``_curing`` so a joint welds on the very step it latches.
    """
    del params
    blocks = objs_by_type(state).get("block", [])
    idx_of = _block_index(blocks)
    blk_by_idx = {idx_of[blk.name]: blk for blk in blocks}
    emitted = set()
    for blk in blocks:
        for slot in ATTACH_SLOTS:
            partner = updates.get(blk, {}).get(
                f"attached_{slot}", float(state.get(blk, f"attached_{slot}")))
            if partner < 0:
                continue
            mate = blk_by_idx.get(int(partner))
            if mate is None:
                continue
            key = frozenset((blk.name, mate.name))
            if key in emitted:
                continue
            emitted.add(key)
            cmds.attach(blk, mate)
    return updates


def _build_param_specs() -> List[ParamSpec]:
    return [
        ParamSpec("cure_threshold", CURE_THRESHOLD, lo=1.0, hi=100.0),
        ParamSpec("apply_glue_radius", APPLY_GLUE_RADIUS, lo=0.0),
        ParamSpec("stack_align_tol", STACK_ALIGN_TOL, lo=0.0),
        ParamSpec("lateral_perp_tol", LATERAL_PERP_TOL, lo=0.0),
        ParamSpec("seat_x_window", SEAT_X_WINDOW, lo=0.0),
    ]


RESIDUAL_RULES = [_glue_application, _curing, _welding]
PARAM_SPECS = _build_param_specs
RESIDUAL_FEATURES: Dict[str, List[str]] = {
    "block":
    [f"glue_{f}" for f in GLUE_FACES] + [f"cure_{f}" for f in GLUE_FACES] +
    [f"attached_{s}" for s in ATTACH_SLOTS]
}


class PyBulletBridgeGroundTruthSimulatorFactory(GroundTruthSimulatorFactory):
    """GT residual-dynamics simulator for pybullet_bridge (fully
    observable)."""

    @classmethod
    def get_env_names(cls) -> set:
        # In PO mode the cure counters are not State features, so this
        # (FO) simulator does not apply; gt_simulator_po.py claims the
        # env.
        if CFG.partially_observable:
            return set()
        return {"pybullet_bridge"}
