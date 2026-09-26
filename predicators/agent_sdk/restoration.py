"""Diagnostics for the candidate's state, never the hidden real state."""
from typing import Any, Dict

import numpy as np
import pybullet as p

from predicators.structs import State


def same_memory(left: Any, right: Any) -> bool:
    """Compare nested model-owned values without ambiguous array equality."""
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            same_memory(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            same_memory(a, b) for a, b in zip(left, right))
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return bool(np.array_equal(left, right))
    return bool(left == right)


def restoration_report(before: State, after: State) -> Dict[str, Any]:
    """Report changed features and memory; Euler aliases are not errors."""
    errors: Dict[str, float] = {}
    for obj in before:
        if obj not in after:
            errors[f"{obj.name}.missing"] = 1.
            continue
        features = obj.type.feature_names
        angles = ("roll", "pitch", "yaw")
        has_orientation = all(f in features for f in angles)
        if has_orientation:
            qa = np.array(
                p.getQuaternionFromEuler([before.get(obj, f) for f in angles]))
            qb = np.array(
                p.getQuaternionFromEuler([after.get(obj, f) for f in angles]))
            angle = float(2 * np.arccos(np.clip(abs(qa @ qb), 0., 1.)))
            if angle > 1e-5:
                errors[f"{obj.name}.orientation_radians"] = angle
        for feature in features:
            if has_orientation and feature in angles:
                continue
            delta = abs(
                float(before.get(obj, feature) - after.get(obj, feature)))
            if not np.isfinite(delta) or delta > 1e-5:
                errors[f"{obj.name}.{feature}"] = delta
    for obj in after:
        if obj not in before:
            errors[f"{obj.name}.unexpected"] = 1.
    memory_ok = same_memory(before.latent, after.latent)
    return {
        "status": "accepted" if not errors and memory_ok else "rejected",
        "feature_errors": errors,
        "memory_preserved": memory_ok
    }


def check_restore(ctx: Any, state: State) -> Dict[str, Any]:
    """Round-trip a candidate snapshot in two fresh worlds, without steps.

    Checks inferred memory and observables, not whether that inference
    is true. Native attachment registration is checked separately from
    poses.
    """
    provider = ctx.probe_option_model_provider
    scope = ctx.probe_validation_env_scope
    if provider is None or scope is None:
        return {"status": "unavailable", "reason": "No isolated candidate"}
    model = provider()
    with scope():
        env = getattr(model, "sim_env", None)
        if env is None:
            return {"status": "unavailable", "reason": "No engine state"}
        env._set_state(state.copy())  # pylint: disable=protected-access
        snapshot = env._get_state().copy()  # pylint: disable=protected-access
        first = restoration_report(state, snapshot)
        links = attachment_pairs(env, snapshot)
        frames = env._command_weld_frame_records()  # pylint: disable=protected-access
    with scope():
        env = model.sim_env
        env._set_state(snapshot.copy())  # pylint: disable=protected-access
        restored = env._get_state()  # pylint: disable=protected-access
        second = restoration_report(snapshot, restored)
        links_after = attachment_pairs(env, restored)
        frames_after = env._command_weld_frame_records()  # pylint: disable=protected-access
    ok = first["status"] == second[
        "status"] == "accepted" and links == links_after and same_memory(
            frames, frames_after)
    return {
        "status": "accepted" if ok else "rejected",
        "start": first,
        "snapshot": second,
        "attachments_preserved": links == links_after,
        "attachment_frames_preserved": same_memory(frames, frames_after),
        "attachments": links,
        "restored_attachments": links_after
    }


def attachment_pairs(env: Any, state: State) -> Any:
    """Registered model attachments, named independently of engine IDs."""
    if not hasattr(env, "get_welded_partner_ids"):
        return None
    names = {o.name for o in state}
    # State objects are sanitized and deliberately carry no engine IDs.
    by_id = {
        body_id: name
        for name, body_id in env._residual_command_body_ids().items()  # pylint: disable=protected-access
        if name in names
    }
    pairs = set()
    for body_id, name in by_id.items():
        for partner in env.get_welded_partner_ids(body_id):
            if partner in by_id:
                pairs.add(tuple(sorted((name, by_id[partner]))))
    return sorted(pairs)
