"""Immutable, offline observation ledger and declared sensor likelihood.

This module does not read evaluator state, infer missing values, filter
observations, or alter the incumbent fitter. Callers identify actual
reset episodes and steps; repeated reads at a step are one measurement.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Dict, Iterable, List, Mapping, Sequence, \
    Tuple

if TYPE_CHECKING:
    from predicators.observation_noise import ObservationNoise
    from predicators.structs import State

FeatureKey = Tuple[str, str, str]  # object name, type name, feature name


def content_digest(payload: bytes) -> str:
    """Hash exact artifact bytes; callers must include runtime dependencies."""
    return hashlib.sha256(payload).hexdigest()


def _digest(value: object) -> str:
    return content_digest(
        json.dumps(value,
                   sort_keys=True,
                   allow_nan=False,
                   separators=(",", ":")).encode("utf-8"))


@dataclass(frozen=True, order=True)
class SensorFeature:
    """A measured scalar, optionally an explicitly conditioned exact input."""
    key: FeatureKey
    sigma: float
    conditioned: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.key, tuple) or len(self.key) != 3 or not all(
                isinstance(part, str) and part for part in self.key):
            raise ValueError("Feature keys require three nonempty strings")
        if not math.isfinite(self.sigma) or self.sigma < 0:
            raise ValueError("Sensor sigma must be finite and nonnegative")
        if self.conditioned and self.sigma != 0:
            raise ValueError("Only exact inputs may be conditioned")


@dataclass(frozen=True)
class SensorModel:
    """Independent, additive, unwrapped and unclipped Gaussian observations.

    Sigma zero denotes an exact constraint, with no numerical noise
    floor. Conditioned fields are retained as inputs, not scored or
    overwritten. A simulator adapter must explicitly consume those
    inputs.
    """
    features: Tuple[SensorFeature, ...]

    def __post_init__(self) -> None:
        features = tuple(sorted(self.features))
        if len({feature.key for feature in features}) != len(features):
            raise ValueError("Duplicate sensor feature")
        object.__setattr__(self, "features", features)

    @classmethod
    def from_state(
        cls,
        state: State,
        noise: ObservationNoise,
        conditioned: Iterable[FeatureKey] = ()
    ) -> SensorModel:
        """Freeze public feature semantics using the injector's classification.

        This adapter includes object feature arrays only. Joint metadata
        or other observations require explicitly declared additional
        features; latent memory and simulator metadata are never
        silently evidence. Undeclared sensor noise requires a different
        inference contract.
        """
        if not noise.declared:
            raise ValueError(
                "Offline reference requires declared sensor noise")
        if any(not math.isfinite(s) or s < 0
               for s in (noise.position, noise.orientation, noise.scalar)):
            raise ValueError("Invalid declared sensor noise")
        keys = frozenset(conditioned)
        features = tuple(
            SensorFeature((obj.name, obj.type.name,
                           feat), noise.feature_sigma(obj.type, feat), (
                               obj.name, obj.type.name, feat) in keys)
            for obj in state for feat in obj.type.feature_names)
        if not keys <= {feature.key for feature in features}:
            raise ValueError("Unknown conditioned feature")
        return cls(features)

    @property
    def digest(self) -> str:
        """Identity includes measurement and conditioning semantics."""
        return _digest({
            "schema": 1,
            "channel": "additive_gaussian_exact",
            "features": asdict(self)
        })

    def log_likelihood(self, observation: Observation,
                       prediction: Mapping[FeatureKey, float]) -> float:
        """Score available measurements once; missing measurements add no term.

        Missing/nonfinite predictions are errors, not ignored residuals.
        Exact contradictions give zero likelihood. Raw angles are never
        wrapped: the current injector perturbs the stored angle directly.
        """
        schema = {feature.key: feature for feature in self.features}
        terms: List[float] = []
        for key, value in observation.values:
            if key not in schema:
                raise ValueError(f"Unknown observed feature: {key}")
            feature = schema[key]
            if feature.conditioned:
                continue
            if key not in prediction or not math.isfinite(prediction[key]):
                raise ValueError(f"Missing or nonfinite prediction: {key}")
            residual = value - prediction[key]
            if feature.sigma == 0:
                terms.append(0.0 if residual == 0 else -math.inf)
            else:
                scaled = residual / feature.sigma
                terms.append(-0.5 * scaled * scaled - math.log(feature.sigma) -
                             0.5 * math.log(2 * math.pi))
        return math.fsum(terms)


@dataclass(frozen=True)
class Observation:
    """An immutable raw measurement at a primitive step within an episode.

    Missing components are absent entries, never NaN or imputed values.
    The caller supplies masking explicitly before constructing this
    object.
    """
    step: int
    values: Tuple[Tuple[FeatureKey, float], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.step, int) or self.step < 0:
            raise ValueError("Observation step must be a nonnegative integer")
        values = tuple(
            sorted((key, float(value)) for key, value in self.values))
        if len({key for key, _ in values}) != len(values):
            raise ValueError("Duplicate measurement feature")
        for key, value in values:
            SensorFeature(key, 0.0)
            if not math.isfinite(value):
                raise ValueError(
                    "Measurements must be finite; omit missing ones")
        object.__setattr__(self, "values", values)

    @classmethod
    def from_state(cls, step: int, state: State) -> Observation:
        """Copy object feature values only, without inferred or privileged
        data."""
        return cls(
            step,
            tuple(
                ((obj.name, obj.type.name, feat), float(state.get(obj, feat)))
                for obj in state for feat in obj.type.feature_names))


@dataclass(frozen=True)
class EpisodeData:
    """Actions and unique observations since an actual reset.

    Action t advances state t to t+1. Sparse observation times are
    allowed, but the primitive action history cannot omit intervening
    steps. A level change without reset must remain in the same episode.
    """
    episode_id: str
    actions: Tuple[Tuple[float, ...], ...]
    observations: Tuple[Observation, ...]

    def __post_init__(self) -> None:
        if not self.episode_id:
            raise ValueError("A globally unique reset episode ID is required")
        actions = tuple(
            tuple(float(v) for v in action) for action in self.actions)
        if any(not math.isfinite(v) for action in actions for v in action):
            raise ValueError("Actions must be finite")
        if len({len(action) for action in actions}) > 1:
            raise ValueError(
                "Primitive action dimension changed within episode")
        unique: Dict[int, Observation] = {}
        for observation in self.observations:
            if observation.step > len(actions):
                raise ValueError("Observation exceeds recorded action history")
            previous = unique.setdefault(observation.step, observation)
            if previous != observation:
                raise ValueError("Conflicting reads of the same observation")
        object.__setattr__(self, "actions", actions)
        object.__setattr__(self, "observations",
                           tuple(unique[step] for step in sorted(unique)))


@dataclass(frozen=True)
class InferenceData:
    """A canonical batch ledger; repeated fits reuse its original evidence."""
    episodes: Tuple[EpisodeData, ...]

    def __post_init__(self) -> None:
        episodes = tuple(sorted(self.episodes, key=lambda e: e.episode_id))
        if len({episode.episode_id for episode in episodes}) != len(episodes):
            raise ValueError("Duplicate reset episode ID")
        object.__setattr__(self, "episodes", episodes)

    @property
    def digest(self) -> str:
        """Content identity includes masks, step coordinates and all
        actions."""
        return _digest({"schema": 1, "data": asdict(self)})

    def log_likelihood(
        self, sensor: SensorModel,
        predictions: Mapping[str, Sequence[Mapping[FeatureKey,
                                                   float]]]) -> float:
        """Score a complete replay for exactly these reset episodes.

        Replay index zero is the initial state. All intermediate
        predictions must be present even when an observation is missing.
        No held-out suffix can enter through a mismatched trajectory
        length.
        """
        if set(predictions) != {e.episode_id for e in self.episodes}:
            raise ValueError("Prediction episodes do not match the ledger")
        terms: List[float] = []
        for episode in self.episodes:
            states = predictions[episode.episode_id]
            if len(states) != len(episode.actions) + 1:
                raise ValueError(
                    "Prediction length does not match action history")
            terms.extend(
                sensor.log_likelihood(obs, states[obs.step])
                for obs in episode.observations)
        return math.fsum(terms)


@dataclass(frozen=True)
class InferenceIdentity:
    """Separate statistical identity from numerical sampler settings.

    The program digest must cover source and its parameter definitions;
    runtime covers simulator dependencies, layout and configuration.
    These are caller-supplied artifact digests, not automatic dependency
    discovery.
    """
    data: str
    sensor: str
    program: str
    prior: str
    runtime: str

    def __post_init__(self) -> None:
        for value in asdict(self).values():
            if len(value) != 64 or any(c not in "0123456789abcdef"
                                       for c in value):
                raise ValueError("Identity fields must be SHA256 hex digests")

    @property
    def digest(self) -> str:
        """Identity changes after any recorded statistical input changes."""
        return _digest({"schema": 1, "inputs": asdict(self)})
