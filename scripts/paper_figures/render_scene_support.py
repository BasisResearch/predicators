"""Shared EGL initialization and diagnostic captures for paper renderers.

The scene export itself lives in predicators/run/cycles_video.py, which
run videos share; its helpers are re-exported here for the figure
scripts.
"""
import importlib.util
from contextlib import contextmanager
from typing import Any, Iterator, List, Sequence
from unittest.mock import patch

import numpy as np
import pybullet as p

# pylint: disable-next=unused-import
from predicators.run.cycles_video import export_visual_scene, \
    raised_flat_markers, record_procedural_meshes, scene_signature


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


def camera_image(client: int,
                 view: Sequence[float],
                 projection: Sequence[float],
                 width: int,
                 height: int,
                 backend: str = "egl") -> np.ndarray:
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
