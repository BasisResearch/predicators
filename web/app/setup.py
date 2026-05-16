"""Pyodide-side bridge for the predicators browser demo.

Exposes a minimal `bridge` object that the JS layer can call:
- bridge.reset(env_name) -> {task_idx, num_objects, action_dim}
- bridge.render() -> {width, height, pixels}  # pixels is a numpy uint8 RGBA buffer
- bridge.step_zero() -> None  # advances the sim by a zero action

Renders via p.ER_TINY_RENDERER because the WASM pybullet build doesn't
have an OpenGL hardware backend.
"""

import numpy as np
import pybullet as p

from predicators import utils_lite as _utils
from predicators.envs import create_new_env
from predicators.settings import CFG


def _ensure_cfg():
    """Populate CFG with sane defaults so envs can construct."""
    _utils.reset_config({
        "env": "pybullet_blocks",
        "seed": 0,
        "num_train_tasks": 1,
        "num_test_tasks": 1,
        "approach": "oracle",
        "pybullet_camera_height": 240,
        "pybullet_camera_width": 320,
    })


class _Bridge:
    def __init__(self):
        self.env = None
        self.task = None

    def reset(self, env_name):
        # Refresh CFG for the selected env.
        _utils.reset_config({
            "env": env_name,
            "seed": 0,
            "num_train_tasks": 1,
            "num_test_tasks": 1,
            "approach": "oracle",
            "pybullet_camera_height": 240,
            "pybullet_camera_width": 320,
        })

        self.env = create_new_env(env_name, do_cache=False, use_gui=False)
        # Force the TINY renderer everywhere - WASM pybullet has no OpenGL HW.
        self._patch_renderer()
        self.task = self.env.reset("test", 0)
        return {
            "task_idx": 0,
            "num_objects": len(self.env._objects),  # noqa: SLF001
            "action_dim": int(self.env.action_space.shape[0]),
        }

    def _patch_renderer(self):
        # Monkey-patch _get_camera_matrices' caller to swap the OpenGL flag
        # for ER_TINY_RENDERER. Easier: wrap the render() method.
        env = self.env
        orig_render = env.render

        def render_tiny(action=None, caption=None):
            del action, caption
            view, proj, w, h = env._get_camera_matrices()  # noqa: SLF001
            _, _, px, _, _ = p.getCameraImage(
                width=w, height=h,
                viewMatrix=view, projectionMatrix=proj,
                renderer=p.ER_TINY_RENDERER,
                physicsClientId=env._physics_client_id,  # noqa: SLF001
            )
            arr = np.array(px, dtype=np.uint8).reshape((h, w, 4))
            return [arr]

        env.render = render_tiny

    def render(self):
        frames = self.env.render()
        rgba = np.asarray(frames[0], dtype=np.uint8)
        h, w, _ = rgba.shape
        # Pyodide will hand the JS side a memoryview; flatten to bytes.
        return {"width": w, "height": h, "pixels": rgba.tobytes()}

    def step_zero(self):
        zeros = np.zeros(self.env.action_space.shape, dtype=np.float32)
        from predicators.structs import Action
        self.env.step(Action(zeros))


bridge = _Bridge()
print("predicators bridge ready")
