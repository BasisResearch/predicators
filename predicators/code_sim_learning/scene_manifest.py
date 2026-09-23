"""The scene manifest handed to the agentic real-to-sim arm.

A real-to-sim pipeline delivers a scene graph: which bodies the scene
holds, their shapes and meshes, their articulation, and which observed
object each one is. This module reads that off a deployment world set to
a level's initial state, and collects the asset files those bodies were
loaded from so the agent can load the same geometry. It records no
masses, frictions, restitutions or damping: those are what the agent has
to model or estimate.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Set, Tuple

import pybullet as p

from predicators import utils
from predicators.envs.pybullet_env import PyBulletEnv
from predicators.pybullet_helpers.objects import loaded_asset
from predicators.settings import CFG
from predicators.structs import State

_GEOMETRY_NAMES = {
    p.GEOM_SPHERE: "sphere",
    p.GEOM_BOX: "box",
    p.GEOM_CYLINDER: "cylinder",
    p.GEOM_MESH: "mesh",
    p.GEOM_PLANE: "plane",
    p.GEOM_CAPSULE: "capsule",
}
_JOINT_NAMES = {
    p.JOINT_REVOLUTE: "revolute",
    p.JOINT_PRISMATIC: "prismatic",
    p.JOINT_SPHERICAL: "spherical",
    p.JOINT_PLANAR: "planar",
    p.JOINT_FIXED: "fixed",
}
# Bodies parked this far from the origin are spares the deployment keeps
# out of view for other task instances; they are not part of the scene.
_OUT_OF_VIEW_DISTANCE = 5.0
LEGEND = {
    "dimensions": ("box: full extents [x, y, z]; sphere: [radius]; "
                   "cylinder and capsule: [length, radius]; mesh: the "
                   "scale applied to the file; plane: [normal x, y, z]"),
    "frames": ("positions in metres, orientations as [x, y, z, w] "
               "quaternions; shape frames are local to their link"),
    "pose": ("bodies bound to an observed object take their pose from the "
             "observation; the others list their fixed base pose"),
    "omitted": ("masses, frictions, restitution and damping are not "
                "recorded; model or estimate them"),
    "not_listed": ("the ground plane and the robot, which SceneBase's "
                   "initialize_pybullet loads itself"),
}


def _decode(value: Any) -> str:
    return value.decode("utf-8", "replace") if isinstance(
        value, bytes) else str(value)


def _asset_root() -> str:
    return utils.get_env_asset_path("", assert_exists=False).rstrip("/")


def _relative_asset(path: str) -> Optional[str]:
    """``assets/...`` for a file under the env assets directory."""
    absolute = os.path.abspath(path)
    root = os.path.abspath(_asset_root())
    if not absolute.startswith(root + os.sep) or not os.path.isfile(absolute):
        return None
    return "assets/" + os.path.relpath(absolute, root).replace(os.sep, "/")


def referenced_files(urdf_path: str) -> List[str]:
    """Every file a URDF names in a ``filename`` attribute, resolved.

    ``package://<pkg>/...`` resolves under the assets' ``urdf/``
    directory (the robot descriptions live there); other names are
    relative to the URDF's own directory.
    """
    text = open(urdf_path, encoding="utf-8", errors="replace").read()
    base = os.path.dirname(urdf_path)
    urdf_root = os.path.join(_asset_root(), "urdf")
    found: List[str] = []
    for name in re.findall(r'filename="([^"]+)"', text):
        if name.startswith("package://"):
            path = os.path.join(urdf_root, name[len("package://"):])
        else:
            path = os.path.normpath(os.path.join(base, name))
        if os.path.isfile(path) and path not in found:
            found.append(path)
    return found


def _shape_entries(body_id: int, link: int, client: int,
                   files: Set[str]) -> List[Dict[str, Any]]:
    entries = []
    for shape in p.getCollisionShapeData(body_id, link,
                                         physicsClientId=client):
        geometry = _GEOMETRY_NAMES.get(shape[2], str(shape[2]))
        entry: Dict[str, Any] = {
            "geometry": geometry,
            "dimensions": [float(v) for v in shape[3]],
            "local_position": [float(v) for v in shape[5]],
            "local_orientation": [float(v) for v in shape[6]],
        }
        filename = _decode(shape[4])
        if filename:
            files.add(filename)
            entry["file"] = _relative_asset(filename) or os.path.basename(
                filename)
        entries.append(entry)
    return entries


def _link_entry(body_id: int, link: int, name: str, visuals: Dict[int,
                                                                  List[Any]],
                client: int, files: Set[str]) -> Dict[str, Any]:
    entry: Dict[str, Any] = {
        "index": link,
        "name": name,
        "shapes": _shape_entries(body_id, link, client, files),
    }
    colours = [[float(v) for v in visual[7]]
               for visual in visuals.get(link, [])]
    if colours:
        entry["visual_rgba"] = colours[0] if len(colours) == 1 else colours
    return entry


def _body_entry(body_id: int, client: int, object_name: Optional[str],
                files: Set[str]) -> Dict[str, Any]:
    base_link, body_name = p.getBodyInfo(body_id, physicsClientId=client)
    position, orientation = p.getBasePositionAndOrientation(
        body_id, physicsClientId=client)
    n_joints = p.getNumJoints(body_id, physicsClientId=client)
    visuals: Dict[int, List[Any]] = {}
    for visual in p.getVisualShapeData(body_id, physicsClientId=client):
        visuals.setdefault(visual[1], []).append(visual)
    links: List[Dict[str, Any]] = [
        _link_entry(body_id, -1, _decode(base_link), visuals, client, files)
    ]
    joints: List[Dict[str, Any]] = []
    for joint in range(n_joints):
        info = p.getJointInfo(body_id, joint, physicsClientId=client)
        links.append(
            _link_entry(body_id, joint, _decode(info[12]), visuals, client,
                        files))
        joints.append({
            "name": _decode(info[1]),
            "type": _JOINT_NAMES.get(info[2], str(info[2])),
            "child_link": _decode(info[12]),
            "parent_link_index": int(info[16]),
            "axis": [float(v) for v in info[13]],
            "lower_limit": float(info[8]),
            "upper_limit": float(info[9]),
            "parent_frame_position": [float(v) for v in info[14]],
            "parent_frame_orientation": [float(v) for v in info[15]],
        })
    entry: Dict[str, Any] = {
        "object": object_name,
        "body_name": _decode(body_name),
        "static": p.getDynamicsInfo(body_id, -1,
                                    physicsClientId=client)[0] == 0.0,
        "links": links,
        "joints": joints,
    }
    asset = loaded_asset(client, body_id)
    if asset is not None:
        asset_path, scale = asset
        source = utils.get_env_asset_path(asset_path)
        files.add(source)
        entry["urdf"] = _relative_asset(source)
        entry["urdf_scale"] = scale
    if object_name is None:
        entry["base_pose"] = {
            "position": [float(v) for v in position],
            "orientation": [float(v) for v in orientation],
        }
    return entry


def build_scene_manifest(
        env: PyBulletEnv,
        state: State) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """Inspect ``env`` at ``state`` and return ``(manifest, assets)``.

    ``env`` is a deployment world (the real env class); ``state`` is the
    level's initial state, whose objects name the bodies. ``assets``
    maps ``assets/...`` paths under the sandbox's ``reference/`` to the
    files to copy there: the URDF and mesh files of every listed body
    and of the robot.
    """
    # pylint: disable=protected-access
    env._set_state(state)
    client = env._physics_client_id
    robot_id = env._pybullet_robot.robot_id
    names_by_body = {
        obj.id: obj.name
        for obj in env._objects if getattr(obj, "id", None) is not None
    }
    files: Set[str] = set()
    bodies: List[Dict[str, Any]] = []
    for index in range(p.getNumBodies(physicsClientId=client)):
        body_id = p.getBodyUniqueId(index, physicsClientId=client)
        if body_id == robot_id:
            continue
        object_name = names_by_body.get(body_id)
        position, _ = p.getBasePositionAndOrientation(body_id,
                                                      physicsClientId=client)
        if object_name is None and (sum(v * v for v in position)**0.5 >
                                    _OUT_OF_VIEW_DISTANCE):
            continue
        entry = _body_entry(body_id, client, object_name, files)
        # Backdrop decoration has no collision shape and no observed
        # object; it is not part of the scene. The ground plane is the
        # scene base's own.
        if object_name is None and (entry["body_name"] == "plane"
                                    or not any(link["shapes"]
                                               for link in entry["links"])):
            continue
        bodies.append(entry)
    bodies.sort(
        key=lambda b: (b["object"] is None, b["object"] or "", b["body_name"]))
    robot_cls = type(env._pybullet_robot)
    robot_urdf = robot_cls.urdf_path()
    files.add(robot_urdf)
    manifest = {
        "legend": LEGEND,
        "robot": {
            "name":
            CFG.pybullet_robot,
            "urdf":
            _relative_asset(robot_urdf),
            "base_position": [
                float(v) for v in p.getBasePositionAndOrientation(
                    robot_id, physicsClientId=client)[0]
            ],
            "base_orientation": [
                float(v) for v in p.getBasePositionAndOrientation(
                    robot_id, physicsClientId=client)[1]
            ],
        },
        "objects": {
            obj.name: obj.type.name
            for obj in sorted(env._objects, key=lambda o: o.name)
        },
        "bodies": bodies,
    }
    assets: Dict[str, str] = {}
    pending = sorted(files)
    while pending:
        source = pending.pop()
        relative = _relative_asset(source)
        if relative is None or relative in assets:
            continue
        assets[relative] = os.path.abspath(source)
        if source.lower().endswith(".urdf"):
            pending.extend(referenced_files(source))
    return manifest, assets


def write_scene_manifest(manifest: Dict[str, Any], path: str) -> str:
    """Write the manifest as indented JSON and return ``path``."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=1, sort_keys=False)
        stream.write("\n")
    return path


__all__ = ["build_scene_manifest", "referenced_files", "write_scene_manifest"]
