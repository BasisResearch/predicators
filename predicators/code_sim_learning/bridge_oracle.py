"""Bridge oracle with a pure observer for glue and attachment memory.

The exporter freezes the native process hook into the artifact and
replaces the superclass with the supplied model base. The observer runs
that process on private symbolic records; it creates no physics client
or constraints.
"""
from __future__ import annotations

import copy
from typing import Any, ClassVar, Dict, FrozenSet, List, Optional, Set

import pybullet as p

from predicators.envs.pybullet_bridge import ATTACH_SLOTS, GLUE_FACES, \
    PyBulletBridgeEnv
from predicators.structs import Action, Object, State

# Export and pure observer intentionally reuse the native private process API.
# pylint: disable=protected-access

# Replaced with the actual function source when exporting the artifact.
_native_step = PyBulletBridgeEnv._domain_specific_step


class BridgeOracle(PyBulletBridgeEnv):
    """Correct process dynamics, restoring model-owned attachment memory."""

    AGENT_PARAM_SPECS: ClassVar[List[Any]] = []
    RESIDUAL_FEATURES = {
        "block": [
            "glue_top", "glue_end_a", "glue_end_b", "x", "y", "z", "roll",
            "pitch", "yaw"
        ]
    }
    MODEL_STATE_INIT = {"blocks": {}}

    @classmethod
    def get_name(cls) -> str:
        return "pybullet_bridge_oracle_model"

    def _memory_snapshot(self, state: State) -> Dict[str, Any]:
        records: Dict[str, Any] = {}
        for block in state.get_objects(self._block_type):
            values: Dict[str, Any] = self._hidden_block_features(block)
            for slot in ATTACH_SLOTS:
                key = f"attached_{slot}"
                index = int(values[key])
                values[key] = self._blocks[index].name if index >= 0 else None
            for face in GLUE_FACES:
                values[f"glue_{face}"] = self._attr(block, f"glue_{face}", 0.0)
            records[block.name] = values
        return records

    @classmethod
    def update_model_state(cls, observation: State, model_state: Dict[str,
                                                                      Any],
                           params: Dict[str, float], action: Action) -> None:
        del cls, params, action
        # Bypass __init__: inference owns only an observed State and Python
        # dictionaries. No engine, hidden live state, or task generator.
        observer = object.__new__(_BridgeObserver)
        observer.initialize(observation, model_state["blocks"])
        contacts = model_state.setdefault("wet_contacts", {})
        # Geometry is noisy, but fully wet glue disappearing is an observed
        # latch event in these dynamics. Remember candidate partners seen
        # during the wet interval instead of demanding 25 uninterrupted
        # noisy contact classifications. Ambiguous partner histories are
        # deliberately left unresolved, never resolved by nearest distance.
        for block in observer._blocks:
            block_contacts = contacts.setdefault(block.name, {})
            for face in GLUE_FACES:
                was_wet = observer._attr(block, f"glue_{face}", 0.) > .5
                if not was_wet:
                    block_contacts.pop(face, None)
                    continue
                mate = observer._find_mate(observer._observed, block, face)
                seen = block_contacts.setdefault(face, [])
                if mate is not None and mate.name not in seen:
                    seen.append(mate.name)
        _native_step(observer)
        for block in observer._blocks:
            previous = model_state["blocks"].get(block.name, {})
            for face in GLUE_FACES:
                consumed = (previous.get(f"glue_{face}", 0.) > .5 >=
                            observation.get(block, f"glue_{face}"))
                seen = contacts[block.name].get(face, [])
                if (consumed and len(seen) == 1 and
                        observer._attr(block, f"attached_{face}", -1.) < 0):
                    mate = next(
                        (b for b in observer._blocks if b.name == seen[0]),
                        None)
                    if mate is not None:
                        observer._latch_joint(observer._observed, block, face,
                                              mate)
        records = observer._memory_snapshot(observer._observed)
        # Glue is observed. In particular, use its previous value to infer
        # the transition whose outcome consumed it at the latch threshold.
        for block in observer._blocks:
            for face in GLUE_FACES:
                records[block.name][f"glue_{face}"] = observation.get(
                    block, f"glue_{face}")
        model_state["blocks"] = records
        # Observation replay has topology and poses, not engine constraint
        # frames. Never carry frames from a different predicted trajectory.
        model_state.pop("weld_frames", None)

    def _set_state(self, state: State) -> None:
        private = copy.deepcopy(state)
        private.privileged = None
        super()._set_state(private)

    def _set_domain_specific_state(self, state: State) -> None:
        private = state.copy()
        private.privileged = {}
        for block in state.get_objects(self._block_type):
            memory = self.model_state["blocks"].get(block.name, {})
            values = {
                f"cure_{f}": float(memory.get(f"cure_{f}", 0.0))
                for f in GLUE_FACES
            }
            for slot in ATTACH_SLOTS:
                partner = memory.get(f"attached_{slot}")
                values[f"attached_{slot}"] = float(
                    self._block_index.get(partner, -1))
            private.privileged[block.name] = values
        super()._set_domain_specific_state(private)
        # The supplied base disables hidden weld semantics. This oracle
        # explicitly restores them, including temporary wet-joint tacks.
        self._restore_model_welds(private)
        curing: Set[FrozenSet[int]] = set()
        for block in private.get_objects(self._block_type):
            for face in GLUE_FACES:
                wet = self._attr(block, f"glue_{face}", 0.0) > 0.5
                curing_face = self._attr(block, f"cure_{face}", 0.0) > 0.0
                unattached = self._attr(block, f"attached_{face}", -1.0) < 0
                if wet and curing_face and unattached:
                    mate = self._find_mate(private, block, face)
                    if mate is not None:
                        assert block.id is not None and mate.id is not None
                        curing.add(frozenset({block.id, mate.id}))
        self._sync_wet_joint_tacks(curing)

    def _restore_model_welds(self, state: State) -> None:
        """Restore joints without treating a carried beam as newly glued.

        Native weld creation snaps a NEW joint to its resting geometry.
        Doing that on reset changes tilted assemblies before prediction.
        Preserve model-owned frames when available; observation-only
        starts reconstruct relative frames from observed poses without
        teleporting.
        """
        for key in list(self._tack_constraints):
            self._drop_tack(key)
        for key in list(self._weld_constraints):
            self._remove_weld(key)
        frames = {
            frozenset((f["parent"], f["child"])): f
            for f in self.model_state.get("weld_frames", [])
        }
        names = {b.id: b.name for b in self._blocks}
        ids = {b.name: b.id for b in self._blocks}
        client = self._physics_client_id
        for key, (body_a, body_b,
                  dz) in self._desired_weld_pairs(state).items():
            saved = frames.get(frozenset((names[body_a], names[body_b])))
            if saved is not None:
                body_a, body_b = ids[saved["parent"]], ids[saved["child"]]
                parent_pos, parent_orn = saved["parent_frame"]
                child_pos, child_orn = saved["child_frame"]
            else:
                pos_a, orn_a = p.getBasePositionAndOrientation(
                    body_a, physicsClientId=client)
                pos_b, orn_b = p.getBasePositionAndOrientation(
                    body_b, physicsClientId=client)
                inv_pos, inv_orn = p.invertTransform(pos_a, orn_a)
                parent_pos, parent_orn = p.multiplyTransforms(
                    inv_pos, inv_orn, pos_b, orn_b)
                child_pos, child_orn = (0., 0., 0.), (0., 0., 0., 1.)
            cid = p.createConstraint(body_a,
                                     -1,
                                     body_b,
                                     -1,
                                     p.JOINT_FIXED, (0., 0., 0.),
                                     parent_pos,
                                     child_pos,
                                     parent_orn,
                                     child_orn,
                                     physicsClientId=client)
            p.changeConstraint(cid,
                               maxForce=self.weld_max_force,
                               physicsClientId=client)
            p.setCollisionFilterPair(body_a,
                                     body_b,
                                     -1,
                                     -1,
                                     0,
                                     physicsClientId=client)
            self._weld_constraints[key] = cid
            self._weld_meta[key] = (body_a, body_b, dz)

    def _domain_specific_step(self) -> None:
        _native_step(self)
        self.model_state["blocks"] = self._memory_snapshot(self._get_state())
        names = {b.id: b.name for b in self._blocks}
        frames = []
        for cid in self._weld_constraints.values():
            info = p.getConstraintInfo(cid,
                                       physicsClientId=self._physics_client_id)
            frames.append({
                "parent": names[info[0]],
                "child": names[info[2]],
                "parent_frame": (info[6], info[8]),
                "child_frame": (info[7], info[9])
            })
        self.model_state["weld_frames"] = frames


class _BridgeObserver(BridgeOracle):
    """Geometry and symbolic process effects with all engine effects absent."""

    _observed: State
    _records: Dict[str, Dict[str, float]]

    def initialize(self, observation: State, memory: Dict[str, Any]) -> None:
        """Bind private objects with synthetic identifiers and stored
        memory."""
        self._observed = copy.deepcopy(observation)
        self._observed.privileged = None
        self._blocks = [o for o in self._observed if o.type.name == "block"]
        self._block_type = self._blocks[0].type
        self._block_index = {o.name: i for i, o in enumerate(self._blocks)}
        self._robot = next(o for o in self._observed if o.type.name == "robot")
        self._bottle = next(o for o in self._observed
                            if o.type.name == "bottle")
        self._records = {}
        for index, block in enumerate(self._blocks):
            block.sim_data["id"] = index
            record = memory.get(block.name, {})
            values = {
                f"cure_{f}": float(record.get(f"cure_{f}", 0.0))
                for f in GLUE_FACES
            }
            for face in GLUE_FACES:
                key = f"glue_{face}"
                values[key] = float(
                    record.get(key, observation.get(block, key)))
            for slot in ATTACH_SLOTS:
                partner = record.get(f"attached_{slot}")
                values[f"attached_{slot}"] = float(
                    self._block_index.get(partner, -1))
            self._records[block.name] = values

    def _get_state(self, _render_obs: bool = False) -> Any:
        del _render_obs
        return self._observed

    def _attr(self, blk: Object, name: str, default: float) -> float:
        return self._records[blk.name].get(name, default)

    def _set_attr(self, blk: Object, name: str, value: float) -> None:
        self._records[blk.name][name] = value

    def _create_weld(self,
                     body_a: int,
                     body_b: int,
                     ideal_dz: Optional[float] = None) -> None:
        del body_a, body_b, ideal_dz

    def _drop_tack(self, key: FrozenSet[int]) -> None:
        del key

    def _sync_wet_joint_tacks(self, curing: Set[FrozenSet[int]]) -> None:
        del curing

    def _relax_resting_welds(self) -> None:
        pass

    def _update_glue_patches(self, state: State) -> None:
        del state
