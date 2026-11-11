"""Build the bodies a scene manifest lists, as the deployment builds them.

A scene manifest (:mod:`predicators.code_sim_learning.scene_manifest`)
describes the scene of one level: every body, its shapes and meshes, its
joints and their limits, and which observed object it is. This module
loads that description into a PyBullet world, so a scene model starts
from the level's geometry and adds only what the manifest leaves out:
the masses, the contact materials and the mechanisms.

A run meets several levels, and their scenes differ (a platform more, a
ramp of another width), while the fit replays every level's recordings.
So the scenes are kept side by side as layouts, one per level, and
:func:`resolve_layout` names the layout a state belongs to.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import pybullet as p

from predicators.pybullet_helpers.objects import set_engine_joint_limits, \
    set_hard_stop_joint_limits
from predicators.structs import State

# The mass the loader gives a moving body it builds from shapes: the
# manifest records no mass, so the body gets its volume at this density
# until the model declares the mass. A body loaded from a URDF file keeps
# the file's masses.
NOMINAL_DENSITY = 1000.0
# Where a body waits until a state places it: far from the scene, each at
# a place of its own, the bodies of each layout in a row of their own.
_PARK_ORIGIN = (30.0, 30.0, 1.0)
_PARK_STEP = 1.0
_PARK_ROW = 5.0
_ASSET_PREFIX = "assets/"
# Two layouts a state fits within this much of each other (a sum of
# squared feature differences, absolute and relative) are the same scene
# as far as the state tells.
_EQUAL_GAP = 1e-9
_DEFAULT_RGBA = (0.6, 0.6, 0.6, 1.0)


@dataclass(frozen=True)
class SceneLayout:
    """The scene of one level: its manifest, and the features each observed
    object with a static base showed when the level began (where the scene has
    it, and what tells the level's states apart from another's)."""
    label: str
    manifest: Dict[str, Any]
    signature: Dict[str, Dict[str, float]] = field(default_factory=dict)

    @property
    def fixed_names(self) -> frozenset:
        """The observed objects whose body is static and has no moving
        joint: nothing about them changes within the level."""
        return frozenset(
            body["object"] for body in self.manifest["bodies"]
            if body["object"] is not None and body["static"] and all(
                joint["type"] == "fixed" for joint in body["joints"]))

    @property
    def object_names(self) -> frozenset:
        """The observed objects of the level, the robot aside."""
        return frozenset(name
                         for name, kind in self.manifest["objects"].items()
                         if kind != "robot")


@dataclass
class LoadedScene:
    """The bodies of one layout in a world: each observed object's body id
    under its name (None for an object without a body) and the bodies no
    observation names."""
    label: str
    bodies: Dict[str, Optional[int]]
    unnamed: List[int]
    # Entries the loader built only in part, and why.
    notes: List[str] = field(default_factory=list)
    # Where the manifest has each unnamed body stand (see stage_scene).
    stands: Dict[int, Tuple[List[float],
                            List[float]]] = field(default_factory=dict)

    def body_ids(self) -> List[int]:
        """Every body this scene loaded."""
        named = [body for body in self.bodies.values() if body is not None]
        return named + list(self.unnamed)


def read_layouts(scene_dir: str) -> List[SceneLayout]:
    """The layouts under ``scene_dir``: one per ``levels/level_*.json``, in
    level order; else the single ``scene_manifest.json``; else none."""
    levels_dir = os.path.join(scene_dir, "levels")
    layouts: List[SceneLayout] = []
    if os.path.isdir(levels_dir):
        for name in sorted(os.listdir(levels_dir)):
            if not (name.startswith("level_") and name.endswith(".json")):
                continue
            with open(os.path.join(levels_dir, name), encoding="utf-8") as f:
                record = json.load(f)
            layouts.append(
                SceneLayout(name[:-len(".json")], record["manifest"],
                            record.get("signature", {})))
    if layouts:
        return layouts
    path = os.path.join(scene_dir, "scene_manifest.json")
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            return [SceneLayout("scene", json.load(f))]
    return []


def static_signature(manifest: Dict[str, Any],
                     state: State) -> Dict[str, Dict[str, float]]:
    """The features ``state`` (a level's initial state) gives each observed
    object whose body has a static base: the pose the scene has it at for the
    whole level."""
    static = {
        body["object"]
        for body in manifest["bodies"]
        if body["object"] is not None and body["static"]
    }
    return {
        obj.name: {
            feature: float(state.get(obj, feature))
            for feature in obj.type.feature_names
        }
        for obj in state if obj.name in static
    }


def feature_gap(a: float, b: float) -> float:
    """How far two readings of a feature are apart; a full turn apart is the
    same angle."""
    gap = abs(a - b)
    return min(gap, abs(gap - 2.0 * math.pi))


def resolve_layout(layouts: Sequence[SceneLayout],
                   state: State,
                   current: Optional[str] = None) -> Optional[SceneLayout]:
    """The layout ``state`` belongs to: among the layouts that name the state's
    objects, the one whose static objects without a moving joint it shows most
    nearly as that level began.

    Of equals, the layout labelled ``current`` (a world keeps the scene
    it holds), else the latest. None without layouts.
    """
    if not layouts:
        return None
    names = frozenset(obj.name for obj in state if obj.type.name != "robot")
    same = [layout for layout in layouts if layout.object_names == names]
    candidates = same or list(layouts)
    by_name = {obj.name: obj for obj in state}
    gaps: List[float] = []
    for layout in candidates:
        gap = 0.0
        fixed = layout.fixed_names
        for name, features in layout.signature.items():
            obj = by_name.get(name)
            if obj is None or name not in fixed:
                continue
            for feature, value in features.items():
                if feature in obj.type.feature_names:
                    gap += feature_gap(float(state.get(obj, feature)),
                                       value)**2
        gaps.append(gap)
    least = min(gaps)
    equals = [
        layout for layout, gap in zip(candidates, gaps)
        if gap <= least * (1.0 + _EQUAL_GAP) + _EQUAL_GAP
    ]
    for layout in equals:
        if layout.label == current:
            return layout
    return equals[-1]


def _asset_file(asset: Callable[[str], str], reference: str) -> str:
    """The file a manifest names as ``assets/...``."""
    relative = reference[len(_ASSET_PREFIX):] if reference.startswith(
        _ASSET_PREFIX) else reference
    return asset(relative)


def _shape_kwargs(shape: Dict[str, Any], asset: Callable[[str], str],
                  visual: bool) -> Dict[str, Any]:
    """``createCollisionShape`` / ``createVisualShape`` arguments of one
    manifest shape."""
    geometry, dims = shape["geometry"], shape["dimensions"]
    frame = "visualFrame" if visual else "collisionFrame"
    kwargs: Dict[str, Any] = {
        f"{frame}Position": shape["local_position"],
        f"{frame}Orientation": shape["local_orientation"],
    }
    if geometry == "box":
        kwargs.update(shapeType=p.GEOM_BOX,
                      halfExtents=[d / 2.0 for d in dims])
    elif geometry == "sphere":
        kwargs.update(shapeType=p.GEOM_SPHERE, radius=dims[0])
    elif geometry in ("cylinder", "capsule"):
        kwargs.update(shapeType=(p.GEOM_CYLINDER if geometry == "cylinder" else
                                 p.GEOM_CAPSULE),
                      radius=dims[1])
        kwargs["length" if visual else "height"] = dims[0]
    elif geometry == "mesh" and shape.get("file"):
        kwargs.update(shapeType=p.GEOM_MESH,
                      fileName=_asset_file(asset, shape["file"]),
                      meshScale=dims)
        if shape.get("concave") and not visual:
            kwargs["flags"] = p.GEOM_FORCE_CONCAVE_TRIMESH
    else:
        raise ValueError(f"no loader for a {geometry} shape")
    return kwargs


def _mesh_extents(path: str, scale: Sequence[float]) -> List[float]:
    """The bounding box of an OBJ file's vertices, scaled."""
    lows, highs = [math.inf] * 3, [-math.inf] * 3
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("v "):
                values = [float(v) for v in line.split()[1:4]]
                lows = [min(a, b) for a, b in zip(lows, values)]
                highs = [max(a, b) for a, b in zip(highs, values)]
    if math.inf in lows:
        return [0.0, 0.0, 0.0]
    return [(h - l) * abs(s) for l, h, s in zip(lows, highs, scale)]


def _shape_volume(shape: Dict[str, Any], asset: Callable[[str], str]) -> float:
    """The volume of one manifest shape; a mesh's bounding box."""
    geometry, dims = shape["geometry"], shape["dimensions"]
    if geometry == "box":
        return float(dims[0] * dims[1] * dims[2])
    if geometry == "sphere":
        return 4.0 / 3.0 * math.pi * float(dims[0])**3
    if geometry == "cylinder":
        return math.pi * float(dims[1])**2 * float(dims[0])
    if geometry == "capsule":
        return math.pi * float(dims[1])**2 * float(
            dims[0]) + 4.0 / 3.0 * math.pi * float(dims[1])**3
    if geometry == "mesh" and shape.get("file"):
        x, y, z = _mesh_extents(_asset_file(asset, shape["file"]), dims)
        return x * y * z
    return 0.0


def _link_rgba(link: Dict[str, Any]) -> Optional[List[float]]:
    """A link's one colour, or None when it lists none or several."""
    rgba = link.get("visual_rgba")
    if rgba and not isinstance(rgba[0], (list, tuple)):
        return [float(v) for v in rgba]
    return None


def _build_shapes(client: int, shapes: List[Dict[str, Any]],
                  asset: Callable[[str], str], visual: bool,
                  rgba: Optional[Sequence[float]]) -> int:
    """One collision or visual shape for a link's manifest shapes (-1 for
    none)."""
    if not shapes:
        return -1
    create = p.createVisualShape if visual else p.createCollisionShape
    if len(shapes) == 1:
        kwargs = _shape_kwargs(shapes[0], asset, visual)
        if visual:
            kwargs["rgbaColor"] = shapes[0].get("rgba", rgba or _DEFAULT_RGBA)
        return int(create(physicsClientId=client, **kwargs))
    # Several shapes on one link: PyBullet takes them as parallel lists.
    parts = [_shape_kwargs(shape, asset, visual) for shape in shapes]
    frame = "visualFrame" if visual else "collisionFrame"
    arrays: Dict[str, Any] = {
        "shapeTypes": [part["shapeType"] for part in parts],
        "halfExtents": [part.get("halfExtents", [0, 0, 0]) for part in parts],
        "radii": [part.get("radius", 0.0) for part in parts],
        "lengths":
        [part.get("length", part.get("height", 0.0)) for part in parts],
        "fileNames": [part.get("fileName", "") for part in parts],
        "meshScales": [part.get("meshScale", [1, 1, 1]) for part in parts],
        f"{frame}Positions": [part[f"{frame}Position"] for part in parts],
        f"{frame}Orientations":
        [part[f"{frame}Orientation"] for part in parts],
    }
    if visual:
        arrays["rgbaColors"] = [
            shape.get("rgba", rgba or _DEFAULT_RGBA) for shape in shapes
        ]
        return int(p.createVisualShapeArray(physicsClientId=client, **arrays))
    return int(p.createCollisionShapeArray(physicsClientId=client, **arrays))


def _load_urdf_body(client: int, entry: Dict[str, Any],
                    asset: Callable[[str], str]) -> int:
    """A body the deployment loaded from a URDF file, with the joint limits and
    the hard stops the deployment set on it."""
    body = int(
        p.loadURDF(_asset_file(asset, entry["urdf"]),
                   useFixedBase=bool(entry["static"]),
                   globalScaling=float(entry["urdf_scale"]),
                   physicsClientId=client))
    joints = {
        p.getJointInfo(body, index, physicsClientId=client)[1].decode("utf-8"):
        index
        for index in range(p.getNumJoints(body, physicsClientId=client))
    }
    for joint in entry["joints"]:
        index = joints.get(joint["name"])
        if index is None:
            continue
        lower, upper = joint["lower_limit"], joint["upper_limit"]
        if joint.get("hard_stop"):
            set_hard_stop_joint_limits(body, index, lower, upper, client)
        elif "urdf_limits" in joint:
            set_engine_joint_limits(body, index, lower, upper, client)
        if "position" in joint:
            p.resetJointState(body,
                              index,
                              joint["position"],
                              0.0,
                              physicsClientId=client)
    for link in entry["links"]:
        _match_link(client, body, link)
    return body


def _match_link(client: int, body: int, link: Dict[str, Any]) -> None:
    """Give a loaded link the contact margin and the colour the manifest lists,
    where the file gave it others."""
    index = link["index"]
    if index >= p.getNumJoints(body, physicsClientId=client):
        return
    if link["shapes"]:
        margin = float(link["shapes"][0]["collision_margin"])
        have = float(
            p.getDynamicsInfo(body, index, physicsClientId=client)[11])
        if abs(margin - have) > 1e-9:
            p.changeDynamics(body,
                             index,
                             collisionMargin=margin,
                             physicsClientId=client)
    rgba = _link_rgba(link)
    if rgba is not None:
        p.changeVisualShape(body,
                            index,
                            rgbaColor=rgba,
                            physicsClientId=client)


def _visual_shape(client: int, link: Dict[str, Any],
                  asset: Callable[[str], str]) -> int:
    """The look of a link: its collision shapes in its colour, or its visual
    shapes when nothing touches it; -1 where a file it needs is not among the
    assets (a mesh the deployment drew from memory)."""
    shapes = link["shapes"] or link.get("visual_shapes", [])
    try:
        return _build_shapes(client, shapes, asset, True, _link_rgba(link))
    except (FileNotFoundError, ValueError, p.error):
        return -1


def _load_built_body(client: int, entry: Dict[str, Any],
                     asset: Callable[[str], str], notes: List[str]) -> int:
    """A body the deployment built from shapes: its base link and the links
    fixed to it, static or at the nominal density.

    A link held by a moving joint is not built, and the notes name it.
    """
    base = entry["links"][0]
    shapes = base["shapes"]
    label = entry["object"] or entry["body_name"] or "a body"
    mass = 0.0
    if not entry["static"] and shapes:
        mass = NOMINAL_DENSITY * sum(
            _shape_volume(shape, asset) for shape in shapes)
    links: Dict[str, List[Any]] = {
        "linkMasses": [],
        "linkCollisionShapeIndices": [],
        "linkVisualShapeIndices": [],
        "linkPositions": [],
        "linkOrientations": [],
        "linkInertialFramePositions": [],
        "linkInertialFrameOrientations": [],
        "linkParentIndices": [],
        "linkJointTypes": [],
        "linkJointAxis": [],
    }
    for link, joint in zip(entry["links"][1:], entry["joints"]):
        if joint["type"] != "fixed" or joint["parent_link_index"] != -1:
            if link["shapes"]:
                notes.append(f"{label}: link {link['name']!r} is not built "
                             f"(a {joint['type']} joint holds it)")
            continue
        collision = _build_shapes(client, link["shapes"], asset, False, None)
        visual = _visual_shape(client, link, asset)
        if collision < 0 and visual < 0:
            continue
        links["linkMasses"].append(0.0)
        links["linkCollisionShapeIndices"].append(collision)
        links["linkVisualShapeIndices"].append(visual)
        links["linkPositions"].append(joint["parent_frame_position"])
        links["linkOrientations"].append(joint["parent_frame_orientation"])
        links["linkInertialFramePositions"].append([0.0, 0.0, 0.0])
        links["linkInertialFrameOrientations"].append([0.0, 0.0, 0.0, 1.0])
        links["linkParentIndices"].append(0)
        links["linkJointTypes"].append(p.JOINT_FIXED)
        links["linkJointAxis"].append(joint["axis"])
    body = int(
        p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=_build_shapes(client, shapes, asset, False,
                                                  None),
            baseVisualShapeIndex=_visual_shape(client, base, asset),
            physicsClientId=client,
            **(links if links["linkMasses"] else {})))
    if shapes:
        p.changeDynamics(body,
                         -1,
                         collisionMargin=float(shapes[0]["collision_margin"]),
                         physicsClientId=client)
    return body


def load_scene(client: int,
               layout: SceneLayout,
               asset: Callable[[str], str],
               row: int = 0,
               staged: bool = True) -> LoadedScene:
    """Load every body of ``layout`` into the world ``client``.

    ``asset`` maps a path under ``reference/assets/`` to the file. A
    body no observation names stands at the pose the manifest lists; an
    observed one waits out of the scene, in the park row ``row``, until
    a state places it. With ``staged`` False every body waits there: the
    scene is loaded but not set up (:func:`stage_scene` sets it up).
    """
    manifest = layout.manifest
    loaded = LoadedScene(layout.label, {}, [])
    for count, entry in enumerate(manifest["bodies"]):
        if entry.get("urdf"):
            body = _load_urdf_body(client, entry, asset)
        else:
            body = _load_built_body(client, entry, asset, loaded.notes)
        p.resetBasePositionAndOrientation(
            body, (_PARK_ORIGIN[0] + _PARK_STEP * count,
                   _PARK_ORIGIN[1] + _PARK_ROW * row, _PARK_ORIGIN[2]),
            (0.0, 0.0, 0.0, 1.0),
            physicsClientId=client)
        name = entry["object"]
        if name is None or name in loaded.bodies:
            loaded.unnamed.append(body)
        else:
            loaded.bodies[name] = body
        pose = entry.get("base_pose")
        if pose is not None:
            loaded.stands[body] = (pose["position"], pose["orientation"])
    for name, kind in manifest["objects"].items():
        if kind != "robot":
            loaded.bodies.setdefault(name, None)
    if staged:
        stage_scene(client, loaded)
    return loaded


def stage_scene(client: int, loaded: LoadedScene) -> None:
    """Put each body of a loaded scene that no observation names where the
    manifest has it stand."""
    for body, (position, orientation) in loaded.stands.items():
        p.resetBasePositionAndOrientation(body,
                                          position,
                                          orientation,
                                          physicsClientId=client)


__all__ = [
    "NOMINAL_DENSITY", "LoadedScene", "SceneLayout", "feature_gap",
    "load_scene", "stage_scene", "read_layouts", "resolve_layout",
    "static_signature"
]
