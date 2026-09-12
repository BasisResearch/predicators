"""Checked reduction of a deterministic readout of an exactly observed source.

This preserves the source observation's likelihood and verifies the
additional readout instead of treating it as independent evidence. The
declared map must be parameter-independent and match the actual runtime
precision and observation phase. No simulator prediction is overwritten.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from typing import Callable, Literal, Optional

from predicators.code_sim_learning.inference_conditioning import \
    UnsupportedConditioning
from predicators.code_sim_learning.inference_data import FeatureKey, \
    Observation, SensorFeature, SensorModel, content_digest


@dataclass(frozen=True)
class ExactReadout:
    """An identified, reviewed measurement map output = g(source).

    The mapping identity must cover its implementation, constants,
    precision, runtime dependencies and justification. In particular,
    empirical agreement alone does not prove parameter-independence.
    This class does not discover or prove the callback's dependencies.
    Different source or sensor semantics require a new declaration.

    The joint observation measure is the source measure followed by a
    deterministic readout. The source supplies the density; a compatible
    extra readout supplies conditional mass one, not another continuous
    density or a Jacobian from treating it as a second source coordinate.
    """
    source: FeatureKey
    output: FeatureKey
    sensor: str
    mapping: str

    def __post_init__(self) -> None:
        SensorFeature(self.source, 0.)
        SensorFeature(self.output, 0.)
        if self.source == self.output:
            raise ValueError("Readout output must differ from its source")
        for digest in (self.sensor, self.mapping):
            if len(digest) != 64 or any(c not in "0123456789abcdef"
                                        for c in digest):
                raise ValueError("Readout identities must be SHA256 digests")

    @property
    def digest(self) -> str:
        """Identify the source-coordinate observation measure and its map."""
        return content_digest(
            json.dumps(
                {
                    "schema": 1,
                    "kind": "exact_deterministic_readout",
                    "declaration": asdict(self)
                },
                sort_keys=True).encode("utf-8"))


@dataclass(frozen=True)
class ReadoutReduction:
    """A checked observation view; the original ledger is never changed.

    The mapping identity belongs in the complete inference identity.
    Contradictions supply no reduced view, preventing their accidental
    removal by a caller that forgets to combine a negative-infinite
    factor. A missing source requires another conditional construction.
    """
    mapping: str
    status: Literal["verified", "not_observed", "exact_contradiction"]
    observation: Optional[Observation]

    @property
    def log_factor(self) -> float:
        """The source likelihood is still required after this check."""
        return -math.inf if self.status == "exact_contradiction" else 0.


def reduce_exact_readout(
        observation: Observation, sensor: SensorModel,
        declaration: ExactReadout,
        evaluate: Callable[[float], float]) -> ReadoutReduction:
    """Verify a known source/readout relation before reducing the observation.

    Both fields must be exact, independently of whether the source is an
    explicitly conditioned external input or a predicted quantity. For a
    predicted source, its likelihood must still be evaluated by the
    caller, including any latent discrepancy model it has declared. A
    noisy source cannot substitute for its unknown true value. Missing
    sources also leave the readout informative and are not dropped.
    """
    if sensor.digest != declaration.sensor:
        raise ValueError("Readout declaration does not match sensor model")
    schema = {feature.key: feature for feature in sensor.features}
    for key in (declaration.source, declaration.output):
        if key not in schema or schema[key].sigma != 0:
            raise UnsupportedConditioning(
                "Readout reduction requires exact source and output fields")
    if schema[declaration.output].conditioned:
        raise ValueError("A derived readout cannot also be an external input")
    values = dict(observation.values)
    if declaration.output not in values:
        return ReadoutReduction(declaration.digest, "not_observed",
                                observation)
    if declaration.source not in values:
        raise UnsupportedConditioning(
            "An observed readout with a missing source needs marginalization")
    expected = evaluate(values[declaration.source])
    if not math.isfinite(expected):
        raise ValueError("Readout callback returned a nonfinite value")
    if expected != values[declaration.output]:
        return ReadoutReduction(declaration.digest, "exact_contradiction",
                                None)
    del values[declaration.output]
    return ReadoutReduction(
        declaration.digest, "verified",
        Observation(observation.step, tuple(values.items())))
