"""Offline exact-output contradictions under explicitly reviewed invariants.

This module checks recorded evidence, not Python program semantics. The
caller must justify and version the invariant for the identified program
and runtime over all admissible initial states and parameters. Empirical
constancy on sampled rollouts is not such a justification.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Dict, List, Literal, Tuple

from predicators.code_sim_learning.inference_data import FeatureKey, \
    InferenceData, SensorFeature, SensorModel, content_digest


@dataclass(frozen=True)
class ConstantOutputs:
    """Reviewed within-episode invariants, tied to source and runtime.

    The review artifact must establish invariance, including the
    relevant initialization and mutation paths. Different reset episodes
    may start at different values. A code, runtime or proof edit
    requires a new declaration. This checker never infers invariance
    from the data.
    """
    program: str
    runtime: str
    review: str
    keys: Tuple[FeatureKey, ...]

    def __post_init__(self) -> None:
        for digest in (self.program, self.runtime, self.review):
            if len(digest) != 64 or any(c not in "0123456789abcdef"
                                        for c in digest):
                raise ValueError("Invariant identities must be SHA256 digests")
        keys = tuple(sorted(self.keys))
        if not keys or len(set(keys)) != len(keys):
            raise ValueError("Invariant keys must be nonempty and distinct")
        for key in keys:
            SensorFeature(key, 0.)
        object.__setattr__(self, "keys", keys)

    @property
    def digest(self) -> str:
        """Identify the reviewed invariant, not an empirical fit result."""
        return content_digest(
            json.dumps(
                {
                    "schema": 1,
                    "kind": "constant_outputs",
                    "declaration": asdict(self)
                },
                sort_keys=True).encode("utf-8"))


@dataclass(frozen=True)
class ExactContradiction:
    """Two exact measurements that cannot both satisfy a constant output."""
    episode_id: str
    key: FeatureKey
    first_step: int
    first_value: float
    later_step: int
    later_value: float


@dataclass(frozen=True)
class SupportAssessment:
    """An evidence-backed negative result, never posterior samples.

    No contradiction means only not_disproved: it does not establish a
    feasible continuous chart, adequate numerical support, or a
    posterior. The assessment is independent of a physical prior because
    the declared invariant is required to hold for all admissible
    initial states.
    """
    data: str
    sensor: str
    declaration: ConstantOutputs
    contradictions: Tuple[ExactContradiction, ...]

    @property
    def status(self) -> Literal["model_inconsistent", "not_disproved"]:
        """Keep model contradiction distinct from finite-particle failure."""
        return "model_inconsistent" if self.contradictions else "not_disproved"


def audit_constant_outputs(data: InferenceData, sensor: SensorModel,
                           declaration: ConstantOutputs, *,
                           program_digest: str,
                           runtime_digest: str) -> SupportAssessment:
    """Check exact predicted outputs without sampling or selecting a prefix.

    All supplied episodes and available observations are checked. A
    witness stops further comparisons for that key in that episode, not
    examination of other keys or episodes. Noisy or conditioned-input
    fields cannot establish this exact-output contradiction.
    """
    if (program_digest, runtime_digest) != (declaration.program,
                                            declaration.runtime):
        raise ValueError(
            "Invariant declaration does not match program/runtime")
    schema = {f.key: f for f in sensor.features}
    for key in declaration.keys:
        if key not in schema or schema[key].sigma != 0 or schema[
                key].conditioned:
            raise ValueError(
                "Invariant requires an exact predicted sensor field")
    witnesses: List[ExactContradiction] = []
    for episode in data.episodes:
        first: Dict[FeatureKey, Tuple[int, float]] = {}
        contradicted = set()
        for observation in episode.observations:
            for key, value in observation.values:
                if key not in declaration.keys or key in contradicted:
                    continue
                step, initial = first.setdefault(key,
                                                 (observation.step, value))
                if value != initial:
                    witnesses.append(
                        ExactContradiction(episode.episode_id, key, step,
                                           initial, observation.step, value))
                    contradicted.add(key)
    return SupportAssessment(data.digest, sensor.digest, declaration,
                             tuple(witnesses))
