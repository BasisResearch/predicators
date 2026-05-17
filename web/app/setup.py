"""Pyodide-side bridge for the predicators browser demo.

Exposes a minimal `bridge` object that the JS layer can call:
- bridge.reset(env_name) -> {task_idx, num_objects, action_dim, manifest}
- bridge.list_options() / list_objects()
- bridge.execute_option(name, object_names, ...) -> {steps, frames}
    where each frame is a {body_id: {pos, orn, joints: {name: rad}}} snapshot

Rendering is done client-side by Three.js + urdf-loader. The bridge
extracts (a) a scene manifest (URDF refs for URDF-loaded bodies +
primitive descriptors for createMultiBody bodies) and (b) per-step
body states so the JS side can drive a real WebGL scene.
"""

import os

import numpy as np
import pybullet as p

from predicators import utils_lite as _utils
from predicators.envs import create_new_env
from predicators.settings import CFG

# Pybullet GEOM_* constants:
#   2 SPHERE, 3 BOX, 4 CYLINDER, 5 MESH, 6 PLANE, 7 CAPSULE
_GEOM_NAMES = {2: "sphere", 3: "box", 4: "cylinder", 5: "mesh",
               6: "plane", 7: "capsule"}

# Pyodide FS prefix where the assets tarball is unpacked at boot. We
# strip this so the JS side can fetch via the dev server's static
# `./predicators_assets/` symlink.
_ASSET_FS_PREFIX = "/lib/python3.13/site-packages/predicators/envs/assets/"
_ASSET_URL_PREFIX = "../predicators_assets/"


def _asset_url(path):
    """Translate a Pyodide-FS asset path to a server-relative URL."""
    if not path:
        return None
    s = path.decode() if isinstance(path, (bytes, bytearray)) else str(path)
    if s.startswith(_ASSET_FS_PREFIX):
        return _ASSET_URL_PREFIX + s[len(_ASSET_FS_PREFIX):]
    # Some envs store mesh paths relative to a setAdditionalSearchPath
    # call. If the path exists under assets/, anchor it there.
    if not os.path.isabs(s):
        return _ASSET_URL_PREFIX + s
    return None  # off-tree path we can't serve


def _ensure_cfg():
    _utils.reset_config({
        "env": "pybullet_blocks",
        "seed": 0,
        "num_train_tasks": 1,
        "num_test_tasks": 1,
        "approach": "oracle",
    })


class _Bridge:

    def __init__(self):
        self.env = None
        self.task = None
        # body_id -> URDF url (server-relative). Filled in by the
        # loadURDF monkey-patch during env construction.
        self._urdf_map = {}

    def reset(self, env_name):
        _utils.reset_config({
            "env": env_name,
            "seed": 0,
            "num_train_tasks": 1,
            "num_test_tasks": 1,
            "approach": "oracle",
        })

        self._urdf_map = {}
        # Wrap pybullet.loadURDF so we capture which URDF backed each
        # body that the env constructs, plus the globalScaling kwarg
        # (predicators uses globalScaling to shrink kettle.urdf and
        # similar to env-appropriate sizes — Three.js's urdf-loader
        # otherwise loads the meshes at native scale, which can be
        # 10x+ off). Tracker is process-global; _urdf_map gets reset
        # every reset() so stale entries don't leak.
        if not getattr(p.loadURDF, "_bridge_wrapped", False):
            orig_load = p.loadURDF

            def _tracked_load(*args, **kwargs):
                bid = orig_load(*args, **kwargs)
                # loadURDF signature: (fileName, basePosition,
                # baseOrientation, useMaximalCoordinates, useFixedBase,
                # flags, globalScaling, physicsClientId).
                url = _asset_url(args[0]) if args else None
                if url is None and "fileName" in kwargs:
                    url = _asset_url(kwargs["fileName"])
                scale = kwargs.get("globalScaling", None)
                if scale is None and len(args) >= 7:
                    scale = args[6]
                if url is not None and isinstance(bid, int) and bid >= 0:
                    bridge._urdf_map[bid] = {
                        "url": url,
                        "scale": float(scale) if scale is not None else 1.0,
                    }
                return bid

            _tracked_load._bridge_wrapped = True  # noqa: SLF001
            _tracked_load._orig = orig_load  # noqa: SLF001
            p.loadURDF = _tracked_load

        self.env = create_new_env(env_name, do_cache=False, use_gui=False)
        self.task = self.env.reset("test", 0)
        env = self.env
        return {
            "task_idx": 0,
            "num_objects": len(env._objects),  # noqa: SLF001
            "action_dim": int(env.action_space.shape[0]),
            "manifest": self.get_scene_manifest(),
            # Env-author-defined camera (Three.js translates pybullet
            # yaw/pitch/distance into a position).
            "camera": {
                "target": list(env._camera_target),  # noqa: SLF001
                "distance": float(env._camera_distance),  # noqa: SLF001
                "yaw": float(env._camera_yaw),  # noqa: SLF001
                "pitch": float(env._camera_pitch),  # noqa: SLF001
                "fov": float(env._camera_fov),  # noqa: SLF001
            },
        }

    # -- Scene manifest + state ------------------------------------

    def get_scene_manifest(self):
        """Walk all bodies in the current physics client and describe
        them as Three.js-buildable entries."""
        cid = self.env._physics_client_id  # noqa: SLF001
        n = p.getNumBodies(physicsClientId=cid)
        entries = []
        for body_id in range(n):
            try:
                info = p.getBodyInfo(body_id, physicsClientId=cid)
            except p.error:
                continue
            base_name = info[0].decode() if info and info[0] else ""
            body_name = info[1].decode() if info and len(info) > 1 else ""

            entry = {
                "body_id": body_id,
                "name": body_name or base_name or f"body_{body_id}",
            }

            # Joints, for sync.
            joint_names = []
            num_joints = p.getNumJoints(body_id, physicsClientId=cid)
            for j in range(num_joints):
                jinfo = p.getJointInfo(body_id, j, physicsClientId=cid)
                jname = jinfo[1].decode() if jinfo[1] else f"joint_{j}"
                jtype = int(jinfo[2])
                # Type 4 = FIXED, doesn't move; skip from sync list.
                if jtype != 4:
                    joint_names.append(jname)
            entry["joint_names"] = joint_names

            if body_id in self._urdf_map:
                entry["kind"] = "urdf"
                entry["url"] = self._urdf_map[body_id]["url"]
                entry["scale"] = self._urdf_map[body_id]["scale"]
            else:
                # Primitive: query visual shapes and reconstruct.
                vis = p.getVisualShapeData(body_id, physicsClientId=cid)
                shapes = []
                for v in vis:
                    # (uniqueId, linkIdx, geomType, dims, meshFile,
                    #  localPos, localOrn, rgba, textureId)
                    geom = _GEOM_NAMES.get(int(v[2]), "unknown")
                    dims = list(v[3])
                    mesh_url = _asset_url(v[4])
                    rgba = list(v[7]) if v[7] is not None else [1, 1, 1, 1]
                    local_pos = list(v[5])
                    local_orn = list(v[6])
                    shapes.append({
                        "link": int(v[1]),
                        "geom": geom,
                        "dims": dims,
                        "mesh_url": mesh_url,
                        "local_pos": local_pos,
                        "local_orn": local_orn,
                        "rgba": rgba,
                    })
                entry["kind"] = "primitive"
                entry["shapes"] = shapes
            entries.append(entry)
        return entries

    def get_body_state(self, body_id):
        """Return base pose + joint angles for a single body."""
        cid = self.env._physics_client_id  # noqa: SLF001
        pos, orn = p.getBasePositionAndOrientation(body_id,
                                                    physicsClientId=cid)
        joints = {}
        nj = p.getNumJoints(body_id, physicsClientId=cid)
        for j in range(nj):
            jinfo = p.getJointInfo(body_id, j, physicsClientId=cid)
            jname = jinfo[1].decode() if jinfo[1] else f"joint_{j}"
            jtype = int(jinfo[2])
            if jtype == 4:  # FIXED
                continue
            jstate = p.getJointState(body_id, j, physicsClientId=cid)
            joints[jname] = float(jstate[0])
        return {"pos": list(pos), "orn": list(orn), "joints": joints}

    def get_all_body_states(self):
        cid = self.env._physics_client_id  # noqa: SLF001
        n = p.getNumBodies(physicsClientId=cid)
        return {body_id: self.get_body_state(body_id) for body_id in range(n)}

    # -- Option-level introspection ---------------------------------
    def list_options(self):
        from predicators.ground_truth_models import get_gt_options
        if self.env is None:
            return []
        options = get_gt_options(self.env.get_name())
        return [
            {
                "name": opt.name,
                "type_names": [t.name for t in opt.types],
                "params_dim": int(opt.params_space.shape[0]),
            } for opt in sorted(options, key=lambda o: o.name)
        ]

    def list_objects(self):
        if self.env is None:
            return []
        return [{"name": o.name, "type_name": o.type.name}
                for o in sorted(self.env._objects,  # noqa: SLF001
                                key=lambda o: o.name)]

    def execute_option(self, option_name, object_names,
                        params=None, max_steps=200, record_every=2):
        """Ground option, run policy, collect body-state snapshots every
        `record_every` sim steps (plus initial + final).

        Returns {steps, frames: [body_states_dict]}.
        """
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
            rng = np.random.default_rng(0)
            low = opt.params_space.low
            high = opt.params_space.high
            params_arr = rng.uniform(low, high).astype(np.float32)
        else:
            params_arr = np.asarray(params, dtype=np.float32)

        ground_opt = opt.ground(chosen, params_arr)
        state = self.env._current_observation  # noqa: SLF001

        if not ground_opt.initiable(state):
            raise RuntimeError(
                f"{option_name}({','.join(object_names)}) "
                "is not initiable in the current state.")

        frames = [self.get_all_body_states()]
        steps = 0
        # OptionExecutionFailure (e.g. IK failed to converge for an
        # unreachable target) is a normal predicators signal, not a
        # bug — catch it Python-side and return a structured result so
        # the JS log shows a one-liner instead of a Pyodide
        # traceback.
        error_msg = None
        try:
            while steps < max_steps:
                act = ground_opt.policy(state)
                state = self.env.step(act)
                steps += 1
                if record_every and steps % record_every == 0:
                    frames.append(self.get_all_body_states())
                if ground_opt.terminal(state):
                    break
        except _utils.OptionExecutionFailure as e:
            # Match how human_option_control_approach surfaces this
            # (predicators/approaches/human_option_control_approach.py
            # line 104): use the bare reason, not the Python class
            # name + repr.
            error_msg = str(e.args[0]) if e.args else "Option failed."

        # Always include the final state.
        if record_every == 0 or steps % record_every != 0:
            frames.append(self.get_all_body_states())

        return {"steps": steps, "frames": frames, "error": error_msg}


bridge = _Bridge()
print("predicators bridge ready")
