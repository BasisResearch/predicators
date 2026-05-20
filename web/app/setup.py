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


# Env-specific CFG overrides that turn on the full demo content. The
# README pitches coffee as "plug in, brew, pour, serve" — that path
# only exists when has_plug=True; with the predicators default the
# powercord, plug, and socket aren't created at all.
_ENV_CFG_OVERRIDES = {
    "pybullet_coffee": {"coffee_machine_has_plug": True},
    # Grow's default `grow_use_skill_factories=True` selects a
    # half-finished skill-factory Pour (terminal triggers as soon as
    # the EE reaches the tilt pose, before any growth happens) and a
    # 4-d skill-factory Place that needs a jug-handle-offset-aware
    # sampler that nsrts.py doesn't provide. The upstream demo
    # configs (scripts/configs/predicatorv3/random_actions_pybullet
    # .yaml, ExoPredicator/causal_predicator.yaml) route through the
    # legacy path; we mirror that minus weak_pour. We *don't* set
    # `grow_weak_pour_terminate_condition` because that exits Pour
    # the moment the jug is above + tilted (fine for the random-
    # actions demo, but no visible growth) — instead we keep the
    # default terminal (Grown.holds(cup)) so the plant fully grows.
    "pybullet_grow": {
        "grow_use_skill_factories": False,
        "grow_place_option_no_sampler": True,
    },
}


class _Bridge:

    def __init__(self):
        self.env = None
        self.task = None
        # body_id -> {url, scale}. Filled in by the loadURDF
        # monkey-patch during env construction; reset() rebuilds it
        # for the freshly-constructed env.
        self._urdf_map = {}
        # NSRTs for the current env, used by execute_option to get
        # state-aware param samples instead of uniform-over-the-box
        # samples (the latter misses grasps / drops blocks off-tower).
        self._nsrts = set()
        # Body ids known to the JS side. Some envs (grow) spawn new
        # bodies mid-execution (e.g. growing plants); execute_option
        # diffs against this set to surface additions/removals so JS
        # can mount/unmount the corresponding meshes.
        self._known_body_ids = set()
        # body_id -> {link_idx (-1 for base): rgba}. Snapshotted on
        # reset and again after each execute_option so envs that
        # repaint via p.changeVisualShape (balance button, coffee
        # button + plate, etc.) get their new colors reflected in JS.
        self._known_link_colors = {}

    def reset(self, env_name):
        cfg = {
            "env": env_name,
            "seed": 0,
            "num_train_tasks": 1,
            "num_test_tasks": 1,
            "approach": "oracle",
        }
        cfg.update(_ENV_CFG_OVERRIDES.get(env_name, {}))
        _utils.reset_config(cfg)

        # Disconnect any prior env's pybullet client so its state
        # doesn't shadow the new env's queries. Several predicators
        # envs (e.g. pybullet_circuit._get_joint_id) call
        # p.getNumJoints without an explicit physicsClientId; with
        # two clients alive at once pybullet picks one and the new
        # env reads garbage.
        if self.env is not None:
            try:
                p.disconnect(physicsClientId=self.env._physics_client_id)  # noqa: SLF001
            except p.error:
                pass
            self.env = None

        self._urdf_map = {}
        # PyBullet-wasm doesn't implement p.createConstraint (e.g.
        # coffee chains cord segments with JOINT_POINT2POINT). Without
        # this shim, the env-reset path crashes during cord creation.
        # We don't need physical chaining for visualization — let the
        # call fail silently and the segments just hover in place.
        if not getattr(p.createConstraint, "_bridge_safe", False):
            _orig_constraint = p.createConstraint

            def _safe_create_constraint(*args, **kwargs):
                try:
                    return _orig_constraint(*args, **kwargs)
                except p.error:
                    return -1

            _safe_create_constraint._bridge_safe = True  # noqa: SLF001
            p.createConstraint = _safe_create_constraint
        if not getattr(p.loadURDF, "_bridge_wrapped", False):
            orig_load = p.loadURDF

            def _tracked_load(*args, **kwargs):
                bid = orig_load(*args, **kwargs)
                # loadURDF signature: (fileName, basePosition,
                # baseOrientation, useMaximalCoordinates,
                # useFixedBase, flags, globalScaling,
                # physicsClientId).
                url = _asset_url(args[0]) if args else None
                if url is None and "fileName" in kwargs:
                    url = _asset_url(kwargs["fileName"])
                scale = kwargs.get("globalScaling", None)
                if scale is None and len(args) >= 7:
                    scale = args[6]
                if url is not None and isinstance(bid, int) and bid >= 0:
                    bridge._urdf_map[bid] = {
                        "url": url,
                        "scale":
                            float(scale) if scale is not None else 1.0,
                    }
                return bid

            _tracked_load._bridge_wrapped = True  # noqa: SLF001
            _tracked_load._orig = orig_load  # noqa: SLF001
            p.loadURDF = _tracked_load

        self.env = create_new_env(env_name, do_cache=False, use_gui=False)
        self.task = self.env.reset("test", 0)
        env = self.env

        # Cache GT NSRTs so execute_option can call gnsrt.sample_option
        # (state-aware sampler) instead of uniformly sampling the
        # params box. Mirrors the default human_option_control path.
        from predicators.ground_truth_models import (get_gt_nsrts,
                                                     get_gt_options)
        env_options = get_gt_options(env_name)
        try:
            self._nsrts = get_gt_nsrts(env_name, env.predicates, env_options)
        except NotImplementedError:
            self._nsrts = set()
        manifest = self.get_scene_manifest()
        self._known_body_ids = {e["body_id"] for e in manifest}
        cid = self.env._physics_client_id  # noqa: SLF001
        self._known_link_colors = {
            bid: self._snapshot_link_colors(bid, cid)
            for bid in self._known_body_ids
        }
        return {
            "task_idx": 0,
            "num_objects": len(env._objects),  # noqa: SLF001
            "action_dim": int(env.action_space.shape[0]),
            "manifest": manifest,
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

    def _describe_body(self, body_id, cid):
        """Build one manifest entry for a single body id."""
        try:
            info = p.getBodyInfo(body_id, physicsClientId=cid)
        except p.error:
            return None
        base_name = info[0].decode() if info and info[0] else ""
        body_name = info[1].decode() if info and len(info) > 1 else ""
        entry = {
            "body_id": body_id,
            "name": body_name or base_name or f"body_{body_id}",
        }
        joint_names = []
        num_joints = p.getNumJoints(body_id, physicsClientId=cid)
        for j in range(num_joints):
            jinfo = p.getJointInfo(body_id, j, physicsClientId=cid)
            jname = jinfo[1].decode() if jinfo[1] else f"joint_{j}"
            jtype = int(jinfo[2])
            if jtype != 4:  # skip FIXED
                joint_names.append(jname)
        entry["joint_names"] = joint_names
        if body_id in self._urdf_map:
            entry["kind"] = "urdf"
            entry["url"] = self._urdf_map[body_id]["url"]
            entry["scale"] = self._urdf_map[body_id]["scale"]
            entry["link_colors"] = self._collect_link_colors(
                body_id, cid, base_name)
        else:
            vis = p.getVisualShapeData(body_id, physicsClientId=cid)
            # Cache link-in-base transforms so multi-link primitives
            # (e.g. coffee's machine = base + top + dispense, the
            # button + lightbar) render their children at the right
            # offset. getVisualShapeData reports visual offsets within
            # the link, not within the body root — without composing
            # in the link's own pose, all child-link shapes pile up
            # on the base.
            base_pos, base_orn = p.getBasePositionAndOrientation(
                body_id, physicsClientId=cid)
            inv_base = p.invertTransform(base_pos, base_orn)
            link_pose_in_base_cache = {-1: ((0, 0, 0), (0, 0, 0, 1))}
            shapes = []
            for v in vis:
                # (uniqueId, linkIdx, geomType, dims, meshFile,
                #  localPos, localOrn, rgba, textureId)
                geom = _GEOM_NAMES.get(int(v[2]), "unknown")
                dims = list(v[3])
                mesh_url = _asset_url(v[4])
                rgba = list(v[7]) if v[7] is not None else [1, 1, 1, 1]
                visual_pos = list(v[5])
                visual_orn = list(v[6])
                link_idx = int(v[1])
                if link_idx not in link_pose_in_base_cache:
                    ls = p.getLinkState(body_id, link_idx,
                                         physicsClientId=cid)
                    # worldLinkFramePosition (4) / Orientation (5) is
                    # the URDF link frame pose in world. Convert to
                    # base frame.
                    link_pose_in_base_cache[link_idx] = p.multiplyTransforms(
                        inv_base[0], inv_base[1], ls[4], ls[5])
                link_pos, link_orn = link_pose_in_base_cache[link_idx]
                shape_pos, shape_orn = p.multiplyTransforms(
                    link_pos, link_orn, visual_pos, visual_orn)
                shapes.append({
                    "link": link_idx,
                    "geom": geom,
                    "dims": dims,
                    "mesh_url": mesh_url,
                    "local_pos": list(shape_pos),
                    "local_orn": list(shape_orn),
                    "rgba": rgba,
                })
            entry["kind"] = "primitive"
            entry["shapes"] = shapes
        return entry

    def _snapshot_link_colors(self, body_id, cid):
        """Return {link_idx: [r,g,b,a]} for the body's current visuals."""
        out = {}
        try:
            vis = p.getVisualShapeData(body_id, physicsClientId=cid)
        except p.error:
            return out
        for v in vis:
            link_idx = int(v[1])
            rgba = list(v[7]) if v[7] is not None else [1, 1, 1, 1]
            # Multi-shape links can repeat; first one wins. envs that
            # tint a multi-link body recolor every link uniformly, so
            # this matches what JS would have rendered originally.
            if link_idx not in out:
                out[link_idx] = rgba
        return out

    def _current_body_ids(self, cid):
        """Return the list of live body IDs. Uses ``getBodyUniqueId``
        rather than ``range(getNumBodies)`` because pybullet may not
        reuse IDs after ``removeBody`` — grow recreates the liquid
        block every pour tick, so the live ID set drifts upward."""
        return [
            p.getBodyUniqueId(i, physicsClientId=cid)
            for i in range(p.getNumBodies(physicsClientId=cid))
        ]

    def get_scene_manifest(self):
        """Walk all bodies in the current physics client and describe
        them as Three.js-buildable entries."""
        cid = self.env._physics_client_id  # noqa: SLF001
        entries = []
        for body_id in self._current_body_ids(cid):
            entry = self._describe_body(body_id, cid)
            if entry is not None:
                entries.append(entry)
        return entries

    def _collect_link_colors(self, body_id, cid, base_link_name):
        """Return {link_name: [r,g,b,a]} for one URDF body.

        Link -1 is pybullet's base-link convention; its name is the
        URDF's root <link> name (returned by p.getBodyInfo). Child
        links (linkIdx >= 0) are named in jointInfo[12]. We keep just
        one RGBA per link — multi-visual links in our envs all get
        tinted to the same color via create_object.
        """
        link_idx_to_name = {-1: base_link_name}
        for j in range(p.getNumJoints(body_id, physicsClientId=cid)):
            jinfo = p.getJointInfo(body_id, j, physicsClientId=cid)
            link_idx_to_name[j] = (jinfo[12].decode() if jinfo[12]
                                    else f"link_{j}")
        out = {}
        for v in p.getVisualShapeData(body_id, physicsClientId=cid):
            link_idx = int(v[1])
            name = link_idx_to_name.get(link_idx, f"link_{link_idx}")
            if name in out:
                continue  # keep first rgba per link
            out[name] = list(v[7]) if v[7] is not None else [1, 1, 1, 1]
        return out

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
        return {body_id: self.get_body_state(body_id)
                for body_id in self._current_body_ids(cid)}

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

    def _sample_ground_option(self, opt, chosen, state, rng):
        """Ground `opt(chosen)` using the matching NSRT's state-aware
        sampler. Falls back to a uniform sample if no NSRT matches
        (e.g. envs without a process/NSRT factory).
        """
        goal = self.env._current_task.goal  # noqa: SLF001
        objects = set(self.env._objects)  # noqa: SLF001
        chosen_tuple = tuple(chosen)
        # ParameterizedOption.__eq__ compares by name (structs.py:1037).
        # get_gt_options() builds fresh option instances each call, so
        # `nsrt.option is opt` is always False — use name equality.
        param_dim = int(opt.params_space.shape[0])
        for nsrt in self._nsrts:
            if nsrt.option.name != opt.name:
                continue
            for gnsrt in _utils.all_ground_nsrts(nsrt, objects):
                if tuple(gnsrt.option_objs) != chosen_tuple:
                    continue
                # Some upstream NSRTs use `null_sampler` even when the
                # paired option (e.g. grow.PickJug) has a non-empty
                # params_space — the matching state-aware sampler
                # (e.g. _pick_sampler -> [0.0]) lives on the *process*
                # factory, which we can't load in Pyodide (torch).
                # When the dim doesn't match, fall back to the low
                # bound (params_space[0] for the skill-factory options
                # is the "natural default", e.g. grasp_z_offset=0.0).
                params = gnsrt._sampler(state, goal, rng,  # noqa: SLF001
                                        gnsrt.objects)
                if len(params) != param_dim:
                    params = np.asarray(opt.params_space.low,
                                         dtype=np.float32)
                else:
                    params = np.clip(params, opt.params_space.low,
                                      opt.params_space.high)
                return opt.ground(chosen, params)
            break
        low = opt.params_space.low
        high = opt.params_space.high
        params = rng.uniform(low, high).astype(np.float32)
        return opt.ground(chosen, params)

    def execute_option(self, option_name, object_names,
                        params=None, max_steps=1000, record_every=1):
        """Ground option, run policy, collect body-state snapshots every
        `record_every` sim steps (plus initial + final).

        Default is one frame per step. Anything coarser makes the
        playback jerky because each PhaseSkill step moves the EE by up
        to ``max_vel_norm`` (~5cm) — at record_every=5 a 13-step PickJug
        only yields 4 keyframes and the arm appears to teleport.

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

        state = self.env._current_observation  # noqa: SLF001
        if params is None:
            rng = np.random.default_rng(0)
            ground_opt = self._sample_ground_option(opt, chosen, state, rng)
        else:
            params_arr = np.asarray(params, dtype=np.float32)
            ground_opt = opt.ground(chosen, params_arr)

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

        # Compute body deltas so JS can spawn new meshes (e.g. growing
        # plants in grow) and unmount removed ones. We diff against the
        # set of body IDs JS already knows about.
        cid = self.env._physics_client_id  # noqa: SLF001
        current_ids = set(self._current_body_ids(cid))
        added_ids = sorted(current_ids - self._known_body_ids)
        removed_ids = sorted(self._known_body_ids - current_ids)
        added = [e for e in (self._describe_body(bid, cid) for bid in added_ids)
                 if e is not None]
        self._known_body_ids = current_ids

        # Color deltas. Envs repaint via p.changeVisualShape during a
        # step (balance flips the button green when machine turns on;
        # coffee's button + dispense plate change with brew state). The
        # initial manifest captured colors at boot; diff against the
        # current visual shape RGBAs and surface only what's changed.
        color_updates = {}
        for bid in current_ids:
            if bid in added_ids:
                continue  # already included in added_bodies
            now = self._snapshot_link_colors(bid, cid)
            prev = self._known_link_colors.get(bid, {})
            changed = {
                link_idx: rgba
                for link_idx, rgba in now.items()
                if prev.get(link_idx) != rgba
            }
            if changed:
                color_updates[bid] = changed
            self._known_link_colors[bid] = now

        return {
            "steps": steps,
            "frames": frames,
            "error": error_msg,
            "added_bodies": added,
            "removed_body_ids": removed_ids,
            "color_updates": color_updates,
        }


bridge = _Bridge()
print("predicators bridge ready")
