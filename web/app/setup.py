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

    # -- Option-level introspection ---------------------------------
    def list_options(self):
        """Return list of {name, type_names, params_dim} for ground-truth
        options of the current env."""
        from predicators.ground_truth_models import get_gt_options
        if self.env is None:
            return []
        options = get_gt_options(self.env.get_name())
        out = []
        for opt in sorted(options, key=lambda o: o.name):
            out.append({
                "name": opt.name,
                "type_names": [t.name for t in opt.types],
                "params_dim": int(opt.params_space.shape[0]),
            })
        return out

    def list_objects(self):
        """Return list of {name, type_name} for objects currently in the env."""
        if self.env is None:
            return []
        # noqa: SLF001
        return [{"name": o.name, "type_name": o.type.name}
                for o in sorted(self.env._objects, key=lambda o: o.name)]

    def execute_option(self, option_name, object_names,
                        params=None, max_steps=200):
        """Ground an option with the named objects and execute its policy
        until termination (or max_steps). Returns the number of steps run."""
        from predicators.ground_truth_models import get_gt_options
        options = {o.name: o for o in get_gt_options(self.env.get_name())}
        if option_name not in options:
            raise ValueError(f"Unknown option: {option_name}")
        opt = options[option_name]

        name_to_obj = {o.name: o for o in self.env._objects}  # noqa: SLF001
        chosen = [name_to_obj[n] for n in object_names]
        for o, t in zip(chosen, opt.types):
            assert o.is_instance(t), (
                f"Object {o.name} of type {o.type.name} doesn't match "
                f"expected type {t.name} for option {opt.name}")

        if params is None:
            # Sample params from the option's params_space (deterministic
            # under the env seed).
            rng = np.random.default_rng(0)
            low = opt.params_space.low
            high = opt.params_space.high
            params_arr = rng.uniform(low, high).astype(np.float32)
        else:
            params_arr = np.asarray(params, dtype=np.float32)

        ground_opt = opt.ground(chosen, params_arr)
        state = self.env._current_observation  # noqa: SLF001

        # Initiable check.
        if not ground_opt.initiable(state):
            raise RuntimeError(
                f"{option_name}({','.join(object_names)}) "
                "is not initiable in the current state.")

        steps = 0
        while steps < max_steps:
            act = ground_opt.policy(state)
            state = self.env.step(act)
            steps += 1
            if ground_opt.terminal(state):
                break
        return steps


bridge = _Bridge()
print("predicators bridge ready")
