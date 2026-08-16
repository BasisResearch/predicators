"""Ground-truth simulator program for pybullet_bridge (glue construction)
residual dynamics -- partially-observable variant.

In PO mode BOTH hidden layers of the glue process are latent:

* ``latent["cure"]``: per-joint dwell counters keyed by
  ``"<owner>|<face>"`` (the wet face uniquely identifies the joint;
  the mate is geometric) -- Pattern A (counter + threshold).
* ``latent["attached"]``: the attachment RELATION itself, keyed the
  same way with the partner's name as value. ``attached_*`` is not an
  observable feature in PO mode (no perception system reports
  "attached to block 3"), so the latch never touches the feature
  channel -- its ONLY observable footprint is kinematic: the welding
  rule re-emits every latched pair as an ``Attach`` physics command,
  so the pair moves as one rigid body (the base sim's own
  feature-to-weld sync is gated off in base-sim mode).

The program therefore emits just the wet-glue flags; everything else
is scored through the poses the welds produce (env-in-the-loop rollout
objective). The threshold is identifiable from *when* pairs start
co-moving across trajectories.

Gates are hard (the recurrent fit is gradient-free).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from predicators.code_sim_learning.commands import CommandBuffer
from predicators.code_sim_learning.fit_space import ParamSpec
from predicators.code_sim_learning.utils import History, Params, \
    ResidualUpdate, objs_by_type
from predicators.ground_truth_models import GroundTruthSimulatorFactory
from predicators.settings import CFG
from predicators.structs import Object, State

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


def _find_mate(state: State, blocks: List[Object], blk: Object, face: str,
               params: Params) -> Optional[Object]:
    """Hard-gated mate detection mirroring the env's geometry."""
    if state.get(blk, "is_held") > 0.5:
        return None
    n_z = _face_dir(state, blk, face)[2]
    for other in blocks:
        if other == blk or state.get(other, "is_held") > 0.5:
            continue
        if n_z > _COS45:
            # Upward face: the mate rests on it.
            hz_b = _half(state, blk)[2]
            hz_o = _half(state, other)[2]
            dz = float(state.get(
                other, "z")) - (float(state.get(blk, "z")) + hz_b + hz_o)
            if abs(dz) >= 0.02:
                continue
            dx = float(state.get(other, "x")) - float(state.get(blk, "x"))
            dy = float(state.get(other, "y")) - float(state.get(blk, "y"))
            if _stands(state, other):
                if float(np.hypot(dx, dy)) < params["stack_align_tol"]:
                    return other
            elif abs(dx) < params["seat_x_window"] and \
                    abs(dy) < SEAT_Y_TOL:
                return other
        elif abs(n_z) < _COS45:
            # Vertical face: the mate butts against it.
            dx_dir, dy_dir, _ = _face_dir(state, blk, face)
            dx = float(state.get(other, "x")) - float(state.get(blk, "x"))
            dy = float(state.get(other, "y")) - float(state.get(blk, "y"))
            dz = float(state.get(other, "z")) - float(state.get(blk, "z"))
            if abs(dz) >= 0.015:
                continue
            proj = dx * dx_dir + dy * dy_dir
            perp = abs(-dx * dy_dir + dy * dx_dir)
            ext = BLOCK_HALF[FACE_AXES[face][0]] + _half(state, other)[0]
            if ext - 0.01 <= proj <= ext + 0.012 and \
                    perp < params["lateral_perp_tol"]:
                return other
        # Downward faces have no reachable mate.
    return None


def _gluing(observation: State, latent: Dict[str, Any], history: History,
            updates: ResidualUpdate, params: Params) -> ResidualUpdate:
    """Recurrent hidden-counter rule: wet faces near the held bottle's tip;
    tick a hidden per-joint counter while a wet face is in aligned resting
    contact; the latch writes ``latent["attached"]``.

    Only the wet-glue flags are emitted as features.
    """
    del history
    cures: Dict[str, float] = latent.setdefault("cure", {})
    attached: Dict[str, str] = latent.setdefault("attached", {})
    objs = objs_by_type(observation)
    blocks = objs.get("block", [])
    bottles = objs.get("bottle", [])

    # Defaults: carry the glue observables.
    glue_next: Dict[Any, Dict[str, float]] = {}
    for blk in blocks:
        glue_next[blk] = {
            face: float(observation.get(blk, f"glue_{face}"))
            for face in GLUE_FACES
        }

    # 1. Glue application: nearest unattached dry face within radius.
    held = [b for b in bottles if observation.get(b, "is_held") > 0.5]
    if held:
        tip = np.array([
            float(observation.get(held[0], "x")),
            float(observation.get(held[0], "y")),
            float(observation.get(held[0], "z")) - BOTTLE_HALF_H
        ])
        best, best_d = None, float(params["apply_glue_radius"])
        for blk in blocks:
            for face in GLUE_FACES:
                if glue_next[blk][face] > 0.5 or \
                        f"{blk.name}|{face}" in attached:
                    continue
                d = float(
                    np.linalg.norm(
                        tip - np.array(_dab_point(observation, blk, face))))
                if d < best_d:
                    best, best_d = (blk, face), d
        if best is not None:
            glue_next[best[0]][best[1]] = 1.0

    # 2. Curing: hidden counters keyed by the wet face; the latch
    # writes the latent attachment relation, never a feature.
    for blk in blocks:
        for face in GLUE_FACES:
            key = f"{blk.name}|{face}"
            if key in attached:
                cures.pop(key, None)
                continue
            if glue_next[blk][face] <= 0.5:
                cures.pop(key, None)
                continue
            mate = _find_mate(observation, blocks, blk, face, params)
            if mate is None:
                cures[key] = 0.0
                continue
            prog = cures.get(key, 0.0) + 1.0
            cures[key] = prog
            if prog >= params["cure_threshold"]:
                mate_slot = _mate_slot_for(observation, blk, face, mate)
                if f"{mate.name}|{mate_slot}" in attached:
                    continue
                attached[key] = mate.name
                attached[f"{mate.name}|{mate_slot}"] = blk.name
                glue_next[blk][face] = 0.0
                cures.pop(key, None)

    for blk in blocks:
        for face in GLUE_FACES:
            updates.setdefault(blk, {})[f"glue_{face}"] = \
                glue_next[blk][face]
    return updates


def _welding(observation: State, latent: Dict[str, Any], history: History,
             updates: ResidualUpdate, params: Params,
             cmds: CommandBuffer) -> ResidualUpdate:
    """Re-emit the rigid weld for every latent attachment.

    Runs after ``_gluing``, so a joint welds on the very step it
    latches; re-emitted every step per the persistent-process command
    contract. This is the latch's ONLY observable footprint in PO mode.
    """
    del observation, history, params
    attached: Dict[str, str] = latent.setdefault("attached", {})
    emitted = set()
    for key, partner in attached.items():
        owner = key.split("|", 1)[0]
        pair = frozenset((owner, partner))
        if pair in emitted:
            continue
        emitted.add(pair)
        cmds.attach(owner, partner)
    return updates


def _latent_init() -> Dict[str, Dict[str, Any]]:
    return {"cure": {}, "attached": {}}


def _build_param_specs() -> List[ParamSpec]:
    return [
        ParamSpec("cure_threshold", CURE_THRESHOLD, lo=1.0, hi=100.0),
        ParamSpec("apply_glue_radius", APPLY_GLUE_RADIUS, lo=0.0),
        ParamSpec("stack_align_tol", STACK_ALIGN_TOL, lo=0.0),
        ParamSpec("lateral_perp_tol", LATERAL_PERP_TOL, lo=0.0),
        ParamSpec("seat_x_window", SEAT_X_WINDOW, lo=0.0),
    ]


RESIDUAL_RULES = [_gluing, _welding]
PARAM_SPECS = _build_param_specs
LATENT_INIT = _latent_init
# Only the wet flags remain feature-scored in PO mode: attachment is a
# latent relation whose footprint is kinematic (rollout-scored poses).
RESIDUAL_FEATURES: Dict[str, List[str]] = {
    "block": [f"glue_{f}" for f in GLUE_FACES]
}


class PyBulletBridgePOGroundTruthSimulatorFactory(GroundTruthSimulatorFactory):
    """GT residual-dynamics simulator for pybullet_bridge (partially
    observable)."""

    @classmethod
    def get_env_names(cls) -> set:
        if CFG.partially_observable:
            return {"pybullet_bridge"}
        return set()
