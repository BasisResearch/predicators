"""Read-only projection of flushed continual recordings for offline inference.

Continual recordings contain sanitized truth for replay, not the noisy
frames the agent received. This adapter reconstructs that observation
channel using explicit run/level coordinates. It does not choose data
splits, create environments, infer hidden state, or modify recordings.
"""
from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, FrozenSet, Iterable, List, \
    Mapping, Optional, Set, Tuple

from predicators.code_sim_learning.inference_data import EpisodeData, \
    FeatureKey, InferenceData, Observation, SensorFeature, SensorModel, \
    content_digest
from predicators.observation_noise import ObservationNoise, step_rng

if TYPE_CHECKING:
    from predicators.structs import State


@dataclass(frozen=True)
class SourceArtifact:
    """Owned bytes of one named source input, independent of later file
    edits."""
    name: str
    content: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("Artifact name must be nonempty")
        if not isinstance(self.content, bytes):
            raise ValueError("Artifact content must be immutable bytes")

    @classmethod
    def read(cls, name: str, path: Path) -> SourceArtifact:
        """Read explicit input bytes once; paths are not content identities."""
        return cls(name, path.read_bytes())

    @property
    def digest(self) -> str:
        """Digest of the actual snapshot, not a subsequently reread file."""
        return content_digest(self.content)


@dataclass(frozen=True)
class ArtifactBundle:
    """Named source snapshots with an immutable content manifest.

    Callers must enumerate all relevant imports, parameter definitions
    and runtime inputs. This freezes that declaration, not automatic
    dependency discovery. Manifest digest can identify a program or a
    runtime bundle.
    """
    artifacts: Tuple[SourceArtifact, ...]

    def __post_init__(self) -> None:
        artifacts = tuple(sorted(self.artifacts, key=lambda a: a.name))
        if not artifacts or len({a.name for a in artifacts}) != len(artifacts):
            raise ValueError("Bundle needs distinct named artifacts")
        object.__setattr__(self, "artifacts", artifacts)

    @property
    def manifest(self) -> bytes:
        """Canonical mapping of logical names to byte identities."""
        return json.dumps(
            {
                "schema": 1,
                "artifacts": [(a.name, a.digest) for a in self.artifacts]
            },
            sort_keys=True,
            separators=(",", ":")).encode("utf-8")

    @property
    def digest(self) -> str:
        """The artifact names and their contents both determine identity."""
        return content_digest(self.manifest)

    def save(self, directory: Path) -> Path:
        """Persist content-addressed bytes without overwriting other snapshots.

        Names are manifest labels, never interpreted as output paths. An
        existing blob must contain exactly the expected bytes.
        """
        directory.mkdir(parents=True, exist_ok=True)
        entries = [(a.digest, a.content) for a in self.artifacts]
        entries.append((self.digest + ".json", self.manifest))
        for filename, content in entries:
            path = directory / filename
            try:
                with path.open("xb") as stream:
                    stream.write(content)
            except FileExistsError:
                if path.read_bytes() != content:
                    raise ValueError(f"Artifact content mismatch: {path}")
        return directory / (self.digest + ".json")


@dataclass(frozen=True)
class RecordingProjection:
    """Explicit interpretation of public fields and excluded metadata.

    Object features, joint positions and mobile base pose are measured.
    Extra metadata requires a stated exclusion reason. In particular,
    raw body velocities and command welds are not silently used as
    inferred state or silently declared sensor evidence. The exact-input
    choices also become part of the recorded projection identity.
    """
    conditioned: Tuple[FeatureKey, ...] = ()
    excluded_metadata: Tuple[Tuple[str, str], ...] = ()
    require_joints: bool = True

    def __post_init__(self) -> None:
        conditioned = tuple(sorted(set(self.conditioned)))
        for key in conditioned:
            SensorFeature(key, 0.0, True)
        excluded = tuple(
            sorted((key, reason) for key, reason in self.excluded_metadata))
        if len({key for key, _ in excluded}) != len(excluded):
            raise ValueError("Duplicate metadata exclusion")
        if any(not key or not reason.strip() for key, reason in excluded):
            raise ValueError("Every metadata exclusion needs a reason")
        if {key for key, _ in excluded} & {"joint_positions", "base_pose"}:
            raise ValueError("Public robot observations cannot be excluded")
        object.__setattr__(self, "conditioned", conditioned)
        object.__setattr__(self, "excluded_metadata", excluded)

    @property
    def artifact(self) -> SourceArtifact:
        """Freeze projection decisions alongside the original recording."""
        return SourceArtifact(
            "observation_projection",
            json.dumps(
                {
                    "schema": 1,
                    "conditioned": self.conditioned,
                    "excluded_metadata": self.excluded_metadata,
                    "require_joints": self.require_joints,
                },
                sort_keys=True,
                separators=(",", ":")).encode("utf-8"))

    def observe(self, step: int, state: State) -> Observation:
        """Copy recorded public features and exact robot proprioception."""
        if state.privileged is not None or state.latent is not None:
            raise ValueError(
                "Expected a sanitized recording, not inferred state")
        values = list(Observation.from_state(step, state).values)
        sim = state.simulator_state
        if sim is None:
            metadata: Mapping[str, Any] = {}
        elif isinstance(sim, dict):
            metadata = sim
        else:
            # Historical PyBulletState stores just the controlled joint array.
            metadata = {"joint_positions": sim}
        extra = set(metadata) - {"joint_positions", "base_pose"}
        if extra - {key for key, _ in self.excluded_metadata}:
            raise ValueError("Unclassified recording metadata: " +
                             str(sorted(extra)))
        if self.require_joints and "joint_positions" not in metadata:
            raise ValueError("Missing public joint positions")
        if "joint_positions" in metadata:
            positions = tuple(float(v) for v in metadata["joint_positions"])
            if not positions:
                raise ValueError("Empty public joint positions")
            values.extend(
                (("__proprioception__", "joint_positions", str(i)), v)
                for i, v in enumerate(positions))
        if "base_pose" in metadata:
            pose = metadata["base_pose"]
            if len(pose) != 2 or len(pose[0]) != 3 or len(pose[1]) != 4:
                raise ValueError("Base pose requires position and quaternion")
            for label, coordinates in zip(("base_position", "base_quaternion"),
                                          pose):
                values.extend((("__proprioception__", label, str(i)), float(v))
                              for i, v in enumerate(coordinates))
        return Observation(step, tuple(values))


@dataclass(frozen=True)
class RecordedLevel:
    """Projected data and original bytes; no simulator or evaluator handles."""
    data: InferenceData
    sensor: SensorModel
    source: ArtifactBundle


def load_recorded_level(directory: Path, run_id: str, noise: ObservationNoise,
                        projection: RecordingProjection, *,
                        observation_seed: int,
                        level_index: int) -> RecordedLevel:
    """Load one explicitly selected trusted level after its recording flush.

    Validate reset markers and every primitive action against
    actions.jsonl. A live/unflushed or inconsistent pair of files is
    rejected rather than truncating an episode or treating a
    continuation as a fresh reset. The caller must select development
    recordings and freeze them before fitting. The stored states are
    sanitized simulator truth: recreate the same step-keyed noise as
    ContinualRun._observed, rather than exposing that truth to
    inference. Already-noisy exports require a different reader, not a
    second draw.
    """
    if not run_id:
        raise ValueError("Recording run identity must be explicit")
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 0
           for value in (observation_seed, level_index)):
        raise ValueError("Noise coordinates must be nonnegative integers")
    if directory.name != f"L{level_index + 1:02d}":
        raise ValueError("Level directory disagrees with noise coordinates")
    channel = SourceArtifact(
        "recording_observation_channel",
        json.dumps(
            {
                "schema": 1,
                "stored_states": "sanitized_simulator_truth",
                "observation_seed": observation_seed,
                "level_index": level_index,
                "noise": asdict(noise),
            },
            sort_keys=True).encode("utf-8"))
    episodes_source = SourceArtifact.read("episodes.pkl",
                                          directory / "episodes.pkl")
    actions_source = SourceArtifact.read("actions.jsonl",
                                         directory / "actions.jsonl")
    # Only trusted local recordings should be unpickled, as with the existing
    # LevelRecording reader. Do not instantiate that writer for read access.
    payload = pickle.loads(episodes_source.content)
    if not isinstance(payload, list) or not payload:
        raise ValueError("Expected a nonempty flushed episode list")
    logged: Dict[int, List[Tuple[float, ...]]] = {}
    current: Optional[int] = None
    for line in actions_source.content.decode("utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        index = row["ep"]
        if not isinstance(index, int) or index < 0:
            raise ValueError("Invalid reset episode number")
        if row.get("event") == "reset":
            if index in logged:
                raise ValueError("Duplicate reset marker")
            logged[index] = []
            current = index
        else:
            if index != current or index not in logged:
                raise ValueError("Action has no matching reset marker")
            if row.get("event") is not None or row["i"] != len(logged[index]):
                raise ValueError("Discontinuous primitive action log")
            logged[index].append(tuple(float(v) for v in row["a"]))
    episodes: List[EpisodeData] = []
    features: Dict[FeatureKey, SensorFeature] = {}
    seen: Set[int] = set()
    for episode in payload:
        index = episode["episode"]
        if index in seen or index not in logged:
            raise ValueError("Episode has no unique reset marker")
        seen.add(index)
        actions = tuple(
            tuple(float(v) for v in action["arr"])
            for action in episode["actions"])
        if actions != tuple(logged[index]):
            raise ValueError(
                "Recording actions disagree; flush before snapshot")
        states = episode["states"]
        if len(states) != len(actions) + 1:
            raise ValueError("Recording must include every action boundary")
        observations: List[Observation] = []
        keys: Optional[FrozenSet[FeatureKey]] = None
        for step, state in enumerate(states):
            # Validate the original record before perturb() sanitizes it,
            # so unexpected metadata cannot silently disappear.
            observation = projection.observe(step, state)
            if noise.enabled:
                view = noise.perturb(
                    state, step_rng(observation_seed, level_index, index,
                                    step))
                observation = projection.observe(step, view)
            observed_keys = frozenset(key for key, _ in observation.values)
            if keys is not None and observed_keys != keys:
                raise ValueError(
                    "Observed feature schema changed inside episode")
            keys = observed_keys
            observations.append(observation)
            schema = {
                f.key: f
                for f in SensorModel.from_state(state, noise).features
            }
            for key, _ in observation.values:
                feature = SensorFeature(
                    key, schema[key].sigma if key in schema else 0., key
                    in projection.conditioned)
                previous = features.setdefault(key, feature)
                if previous != feature:
                    raise ValueError(
                        "Sensor semantics changed within recording")
        # JSON encoding avoids collisions between run/level strings containing
        # path separators. A reset is established by the log, not the folder.
        identifier = json.dumps((run_id, directory.name, index),
                                separators=(",", ":"))
        episodes.append(EpisodeData(identifier, actions, tuple(observations)))
    if seen != set(logged):
        raise ValueError("Reset log and episode snapshot disagree")
    if set(projection.conditioned) - set(features):
        raise ValueError("Unknown conditioned observation")
    return RecordedLevel(
        InferenceData(tuple(episodes)), SensorModel(tuple(features.values())),
        ArtifactBundle(
            (episodes_source, actions_source, projection.artifact, channel,
             SourceArtifact.read(
                 "observation_noise_source",
                 Path(__file__).parents[1] / "observation_noise.py"))))


def combine_recorded_levels(levels: Iterable[RecordedLevel]) -> RecordedLevel:
    """Combine explicitly selected levels, rejecting incompatible semantics."""
    episodes: List[EpisodeData] = []
    features: Dict[FeatureKey, SensorFeature] = {}
    artifacts: List[SourceArtifact] = []
    for level in levels:
        episodes.extend(level.data.episodes)
        for feature in level.sensor.features:
            previous = features.setdefault(feature.key, feature)
            if previous != feature:
                raise ValueError("Incompatible sensor semantics across levels")
        # Content-named level manifests retain original artifact mappings.
        artifacts.append(
            SourceArtifact(level.data.digest + ".source",
                           level.source.manifest))
        artifacts.extend(
            SourceArtifact(a.digest, a.content)
            for a in level.source.artifacts)
    unique = {a.name: a for a in artifacts}
    return RecordedLevel(InferenceData(tuple(episodes)),
                         SensorModel(tuple(features.values())),
                         ArtifactBundle(tuple(unique.values())))
