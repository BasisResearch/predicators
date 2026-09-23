"""Shared EGL initialization and visual-geometry export for paper renderers."""
import copy
import hashlib
import importlib.util
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Sequence, Tuple
from unittest.mock import patch

import numpy as np
import pybullet as p
from numpy.typing import NDArray

_PROCEDURAL_MESHES: Dict[Tuple[int, int, int], Dict[str, Any]] = {}


@contextmanager
def record_procedural_meshes() -> Iterator[None]:
    """Capture in-memory visuals that getVisualShapeData cannot serialize."""
    shapes: Dict[Tuple[int, int], Dict[str, Any]] = {}
    create_shape, create_body = p.createVisualShape, p.createMultiBody

    def shape(*args: Any, **kwargs: Any) -> int:
        result = create_shape(*args, **kwargs)
        if 'vertices' in kwargs:
            shapes[(kwargs.get('physicsClientId',
                               0), result)] = copy.deepcopy({
                                   'vertices':
                                   kwargs['vertices'],
                                   'indices':
                                   kwargs['indices']
                               })
        return result

    def body(*args: Any, **kwargs: Any) -> int:
        result = create_body(*args, **kwargs)
        client = kwargs.get('physicsClientId', 0)
        for key in list(_PROCEDURAL_MESHES):
            if key[:2] == (client, result):
                del _PROCEDURAL_MESHES[key]
        visuals = [kwargs.get('baseVisualShapeIndex', -1)] + list(
            kwargs.get('linkVisualShapeIndices', []))
        for link, visual in enumerate(visuals, start=-1):
            if (client, visual) in shapes:
                _PROCEDURAL_MESHES[(client, result, link)] = shapes[(client,
                                                                     visual)]
        return result

    _PROCEDURAL_MESHES.clear()
    with patch.object(p, 'createVisualShape',
                      shape), patch.object(p, 'createMultiBody', body):
        yield
    _PROCEDURAL_MESHES.clear()


@contextmanager
def egl_connections() -> Iterator[List[int]]:
    """Load EGL before the first environment creates any visual shapes."""
    connect = p.connect
    clients: List[int] = []

    def with_egl(*args: Any, **kwargs: Any) -> int:
        client = connect(*args, **kwargs)
        if not clients:
            spec = importlib.util.find_spec('eglRenderer')
            if spec is None:
                raise RuntimeError(
                    'The experiment Python must provide eglRenderer')
            plugin = p.loadPlugin(spec.origin,
                                  '_eglRendererPlugin',
                                  physicsClientId=client)
            if plugin < 0:
                p.disconnect(client)
                raise RuntimeError('EGL renderer could not be loaded')
            clients.append(client)
        return client

    with patch.object(p, 'connect', with_egl):
        yield clients


def scene_signature(client: int) -> Dict[int, Dict[str, Any]]:
    """Capture poses and joints to verify that a camera capture is passive."""
    return {
        body: {
            'base':
            p.getBasePositionAndOrientation(body, physicsClientId=client),
            'joints': [
                p.getJointState(body, j, physicsClientId=client)[:2]
                for j in range(p.getNumJoints(body, physicsClientId=client))
            ],
        }
        for body in [
            p.getBodyUniqueId(i, physicsClientId=client)
            for i in range(p.getNumBodies(physicsClientId=client))
        ]
    }


def export_visual_scene(client: int, view: Sequence[float],
                        projection: Sequence[float], width: int, height: int,
                        metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Export visual primitives/mesh references from a client WITHOUT EGL.

    The installed EGL plugin returns stale per-instance colors and accumulated
    geometry through getVisualShapeData. Never export a plugin-backed client.

    Bullet stores base poses at the inertial frame and visual offsets at the
    link frame. Undo that inertial offset before composing a base visual pose.
    Child links expose their world link frame directly via getLinkState.
    """
    shapes: List[Dict[str, Any]] = []
    assets: Dict[str, str] = {}
    for i in range(p.getNumBodies(physicsClientId=client)):
        body = p.getBodyUniqueId(i, physicsClientId=client)
        name = p.getBodyInfo(body, physicsClientId=client)[1].decode()
        for index, shape in enumerate(
                p.getVisualShapeData(body, physicsClientId=client)):
            values = shape[:8]
            _, link, kind, dimensions, mesh, local_pos, local_quat, rgba = \
                values
            if link == -1:
                pos, quat = p.getBasePositionAndOrientation(
                    body, physicsClientId=client)
                dynamics = p.getDynamicsInfo(body, -1, physicsClientId=client)
                inv_pos, inv_quat = p.invertTransform(dynamics[3], dynamics[4])
                frame = p.multiplyTransforms(pos, quat, inv_pos, inv_quat)
            else:
                frame = p.getLinkState(body,
                                       link,
                                       computeForwardKinematics=True,
                                       physicsClientId=client)[4:6]
            pos, quat = p.multiplyTransforms(*frame, local_pos, local_quat)
            filename = mesh.decode() if kind == p.GEOM_MESH else ''
            if filename:
                path = Path(filename)
                if not path.is_file():
                    raise FileNotFoundError(
                        f'Cannot export visual mesh: {filename}')
                assets[filename] = hashlib.sha256(
                    path.read_bytes()).hexdigest()
            shapes.append(
                dict(body=body,
                     link=link,
                     index=index,
                     name=name,
                     kind=kind,
                     dimensions=dimensions,
                     mesh=filename,
                     position=pos,
                     quaternion_xyzw=quat,
                     rgba=rgba))
            if kind == p.GEOM_MESH and not filename:
                shapes[-1].update(_PROCEDURAL_MESHES[(client, body, link)])
    return dict(
        generated_by='scripts/render_egl_scenes.py; do not edit manually',
        metadata=metadata,
        camera=dict(view=list(view),
                    projection=list(projection),
                    width=width,
                    height=height),
        shapes=shapes,
        mesh_sha256=assets)


def camera_image(client: int,
                 view: Sequence[float],
                 projection: Sequence[float],
                 width: int,
                 height: int,
                 backend: str = "egl") -> NDArray[np.uint8]:
    """Capture a diagnostic PyBullet image without advancing physics."""
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1, physicsClientId=client)
    # Keep broad highlights subtle on the simple benchmark materials.
    for i in range(p.getNumBodies(physicsClientId=client)):
        body = p.getBodyUniqueId(i, physicsClientId=client)
        for shape in p.getVisualShapeData(body, physicsClientId=client):
            p.changeVisualShape(body,
                                shape[1],
                                specularColor=(.08, .08, .08),
                                physicsClientId=client)
    rgba = p.getCameraImage(width,
                            height,
                            viewMatrix=view,
                            projectionMatrix=projection,
                            renderer=p.ER_BULLET_HARDWARE_OPENGL
                            if backend == "egl" else p.ER_TINY_RENDERER,
                            shadow=1,
                            lightDirection=(-3, -4, 7),
                            lightAmbientCoeff=.4,
                            lightDiffuseCoeff=.65,
                            lightSpecularCoeff=.15,
                            physicsClientId=client)[2]
    rgb = np.asarray(rgba, dtype=np.uint8).reshape(height, width, 4)[:, :, :3]
    if np.unique(rgb.reshape(-1, 3), axis=0).shape[0] < 100:
        raise RuntimeError(
            'Blank EGL render; load the plugin before scene creation')
    return rgb


@contextmanager
def raised_flat_markers(client: int) -> Iterator[None]:
    """Draw coplanar target decals above the tabletop to avoid EGL z-fighting.

    These temporary visual-only copies do not move the original physical
    body.
    """
    temporary: List[int] = []
    try:
        for body in list(scene_signature(client)):
            for shape in p.getVisualShapeData(body, physicsClientId=client):
                _, link, kind, dims, _, local_pos, local_quat, rgba = shape[:8]
                if link != -1 or kind != p.GEOM_BOX or not (0 < dims[2] <
                                                            .001):
                    continue
                if min(dims[:2]) < .01 or rgba[3] < .99:
                    continue
                body_pos, body_quat = p.getBasePositionAndOrientation(
                    body, physicsClientId=client)
                pos, quat = p.multiplyTransforms(body_pos, body_quat,
                                                 local_pos, local_quat)
                lifted = (pos[0], pos[1], pos[2] + .0008)
                visual = p.createVisualShape(p.GEOM_BOX,
                                             halfExtents=[v / 2 for v in dims],
                                             rgbaColor=rgba,
                                             physicsClientId=client)
                temporary.append(
                    p.createMultiBody(baseMass=0,
                                      baseCollisionShapeIndex=-1,
                                      baseVisualShapeIndex=visual,
                                      basePosition=lifted,
                                      baseOrientation=quat,
                                      physicsClientId=client))
        yield
    finally:
        for body in temporary:
            p.removeBody(body, physicsClientId=client)
