"""Recording writer to offline likelihood, without launching an environment."""
import json
from pathlib import Path

import numpy as np
import pytest

from predicators.code_sim_learning.inference_recording import ArtifactBundle, \
    RecordingProjection, SourceArtifact, combine_recorded_levels, \
    load_recorded_level
from predicators.observation_noise import ObservationNoise, step_rng
from predicators.run.recording import LevelRecording
from predicators.structs import Action, Object, Type
from predicators.utils import PyBulletState


def _write_level(directory: Path) -> None:
    obj = Object("box", Type("box", ["x", "attached"]))
    noise = ObservationNoise(position=.1)
    truth = PyBulletState({obj: np.array([.3, 0.])},
                          simulator_state={
                              "joint_positions": [.1, .2],
                              "physics_client_id": 93,
                              "body_velocities": {
                                  "box": ((0., 0., 0.), (0., 0., 0.))
                              },
                          })
    frames = [
        noise.perturb(truth, step_rng(4, 0, 0, step)) for step in range(3)
    ]
    action = Action(np.array([.1, .2], dtype=np.float32))
    writer = LevelRecording(str(directory))
    writer.begin_episode(0, "level_start")
    writer.append_step(0, 0, action)
    writer.append_step(0, 1, action)
    writer.flush([{
        "episode": 0,
        "end": "in_progress",
        "states": frames,
        "actions": [action, action]
    }], frames[-1], 0, 2)
    writer.close()


def _projection() -> RecordingProjection:
    return RecordingProjection(excluded_metadata=(
        ("body_velocities",
         "Replay metadata excluded from this observation model"), ))


def test_recording_roundtrip_and_artifact_snapshot(tmp_path: Path) -> None:
    """Load the actual writer format, keep joints, and freeze original
    bytes."""
    directory = tmp_path / "L01"
    _write_level(directory)
    paths = [directory / "episodes.pkl", directory / "actions.jsonl"]
    original = [path.read_bytes() for path in paths]
    level = load_recorded_level(directory, "run-1",
                                ObservationNoise(position=.1), _projection())
    assert [path.read_bytes() for path in paths] == original
    episode, = level.data.episodes
    assert len(episode.actions) == 2 and len(episode.observations) == 3
    assert json.loads(episode.episode_id) == ["run-1", "L01", 0]
    key = ("__proprioception__", "joint_positions", "1")
    assert dict(episode.observations[0].values)[key] == .2
    assert all("body_velocities" not in feature.key
               for feature in level.sensor.features)
    predictions = [dict(frame.values) for frame in episode.observations]
    assert np.isfinite(
        level.data.log_likelihood(level.sensor,
                                  {episode.episode_id: predictions}))
    predictions[1][key] += .01
    assert level.data.log_likelihood(
        level.sensor, {episode.episode_id: predictions}) == -np.inf
    snapshot = level.source.save(tmp_path / "artifacts")
    assert snapshot.read_bytes() == level.source.manifest
    assert level.source.save(tmp_path / "artifacts") == snapshot
    paths[0].write_bytes(b"later file edit")
    episode_source = next(a for a in level.source.artifacts
                          if a.name == "episodes.pkl")
    assert episode_source.content == original[0]
    assert (tmp_path / "artifacts" /
            episode_source.digest).read_bytes() == original[0]


def test_reset_and_flush_alignment(tmp_path: Path) -> None:
    """Unflushed steps and missing reset markers cannot create new evidence."""
    directory = tmp_path / "L01"
    _write_level(directory)
    actions = directory / "actions.jsonl"
    original = actions.read_text()
    actions.write_text(original + json.dumps({
        "ep": 0,
        "i": 2,
        "a": [.1, .2]
    }) + "\n")
    with pytest.raises(ValueError, match="flush before snapshot"):
        load_recorded_level(directory, "run", ObservationNoise(),
                            _projection())
    actions.write_text("\n".join(original.splitlines()[1:]) + "\n")
    with pytest.raises(ValueError, match="reset marker"):
        load_recorded_level(directory, "run", ObservationNoise(),
                            _projection())
    actions.write_text(original)
    level = load_recorded_level(directory, "run", ObservationNoise(),
                                _projection())
    with pytest.raises(ValueError, match="Duplicate reset episode"):
        combine_recorded_levels((level, level))
    other = load_recorded_level(directory, "other-run", ObservationNoise(),
                                _projection())
    combined = combine_recorded_levels((level, other))
    assert len(combined.data.episodes) == 2


def test_metadata_requires_explicit_semantics(tmp_path: Path) -> None:
    """No inferred memory or extra physical metadata silently enters a fit."""
    directory = tmp_path / "L01"
    _write_level(directory)
    with pytest.raises(ValueError, match="Unclassified recording metadata"):
        load_recorded_level(directory, "run", ObservationNoise(),
                            RecordingProjection())
    with pytest.raises(ValueError, match="needs a reason"):
        RecordingProjection(excluded_metadata=(("body_velocities", ""), ))
    with pytest.raises(ValueError, match="cannot be excluded"):
        RecordingProjection(excluded_metadata=(("joint_positions",
                                                "ignore"), ))
    state = PyBulletState({},
                          simulator_state={"joint_positions": [.1]},
                          latent={})
    with pytest.raises(ValueError, match="sanitized"):
        RecordingProjection().observe(0, state)
    state.latent = None
    state.simulator_state = {"joint_positions": [np.nan]}
    with pytest.raises(ValueError, match="finite"):
        RecordingProjection().observe(0, state)


def test_artifact_identity_and_collision(tmp_path: Path) -> None:
    """Changing a dependency changes identity; old snapshots cannot be
    replaced."""
    program = SourceArtifact("simulator", b"program")
    dependency = SourceArtifact("dependency", b"version1")
    bundle = ArtifactBundle((program, dependency))
    assert bundle.digest == ArtifactBundle((dependency, program)).digest
    assert bundle.digest != ArtifactBundle(
        (program, SourceArtifact("dependency", b"version2"))).digest
    manifest = bundle.save(tmp_path)
    manifest.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="Artifact content mismatch"):
        bundle.save(tmp_path)
