"""Interrupted inference must reproduce uninterrupted fixed-target sampling."""
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import List, Union

import numpy as np
import pytest

from predicators.code_sim_learning.inference_checkpoint import \
    SamplerCheckpoint
from predicators.code_sim_learning.inference_data import InferenceIdentity, \
    content_digest
from predicators.code_sim_learning.inference_sampling import BoxPrior, \
    ConditionedPrior, PriorPoint, SamplerConfig, sample_batch


@pytest.mark.parametrize("conditional", [False, True])
@pytest.mark.parametrize("interruption_stage", [0, 2, 5])
def test_resume_exact_run(tmp_path: Path, conditional: bool,
                          interruption_stage: int) -> None:
    """Disk round trips preserve weights, ancestry, proposals and RNG state.

    Interrupt inside a move stage, then rerun from the last saved
    boundary. Zero-support particles and conditional density factors
    remain present, and no initial candidates are evaluated again.
    """
    box = BoxPrior(("u", "v"), ((0., 1.), (0., 1.)))
    digest = content_digest(b"checkpoint reference")
    prior: Union[BoxPrior, ConditionedPrior] = box
    if conditional:
        prior = ConditionedPrior(("x", "v", "derived"), digest, digest, box)
    identity = InferenceIdentity(digest, digest, digest, prior.digest, digest)
    config = SamplerConfig(particles=48,
                           temperatures=5,
                           moves=3,
                           proposal_scale=.2,
                           max_evaluations=1000,
                           proposal_blocks=((0, ), (1, )),
                           temperature_schedule=(.005, .03, .15, .5, 1.))

    def condition(value: np.ndarray) -> PriorPoint:
        return PriorPoint((value[0]**2, value[1], value[0] + value[1]),
                          math.log(.1 + value[1]))

    def likelihood(value: np.ndarray) -> float:
        if value[1] < .25:
            return -math.inf
        return float(-200 * (value[0] - .7)**2 - 4 * value[1])

    reference = sample_batch(prior,
                             identity,
                             likelihood,
                             config,
                             19,
                             condition=condition if conditional else None)
    assert reference.status == "complete"
    assert reference.resampling_count > 0
    path = tmp_path / "checkpoint.json"
    ready = False
    calls = 0

    def save(checkpoint: SamplerCheckpoint) -> None:
        nonlocal ready
        checkpoint.save(path)
        ready = checkpoint.unpack()["completed_stage"] == interruption_stage
        # The returned mapping is independent of the checkpoint and solver.
        checkpoint.unpack()["particles"][0][0] = math.nan

    def interrupted_likelihood(value: np.ndarray) -> float:
        nonlocal calls
        calls += 1
        if ready:
            raise RuntimeError("worker interrupted during target evaluation")
        return likelihood(value)

    if interruption_stage < config.temperatures:
        with pytest.raises(RuntimeError, match="worker interrupted"):
            sample_batch(prior,
                         identity,
                         interrupted_likelihood,
                         config,
                         19,
                         condition=condition if conditional else None,
                         checkpoint=save)
    else:
        assert sample_batch(prior,
                            identity,
                            interrupted_likelihood,
                            config,
                            19,
                            condition=condition if conditional else None,
                            checkpoint=save) == reference
    checkpoint = SamplerCheckpoint.load(path)
    assert checkpoint.unpack()["completed_stage"] == interruption_stage
    calls = 0

    def counted(value: np.ndarray) -> float:
        nonlocal calls
        calls += 1
        return likelihood(value)

    resumed = sample_batch(prior,
                           identity,
                           counted,
                           config,
                           19,
                           condition=condition if conditional else None,
                           resume=checkpoint)
    assert resumed == reference
    assert calls == reference.evaluations - checkpoint.unpack()["evaluations"]


def test_reject_changed_run_and_damaged_checkpoint(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject incompatible inputs before target evaluation; writes are
    atomic."""
    prior = BoxPrior(("x", ), ((0., 1.), ))
    digest = content_digest(b"immutable")
    identity = InferenceIdentity(digest, digest, digest, prior.digest, digest)
    config = SamplerConfig(particles=8, temperatures=2, moves=1)
    saved: List[SamplerCheckpoint] = []
    sample_batch(prior,
                 identity,
                 lambda _: 0.,
                 config,
                 0,
                 checkpoint=saved.append)
    checkpoint = saved[0]

    def never(_: np.ndarray) -> float:
        raise AssertionError("incompatible checkpoint evaluated")

    for field in ("program", "data", "runtime", "sensor"):
        changed = replace(identity, **{field: content_digest(b"changed")})
        with pytest.raises(ValueError, match="differs"):
            sample_batch(prior, changed, never, config, 0, resume=checkpoint)
    for run_config, run_seed in ((config, 1), (replace(config, moves=2), 0)):
        with pytest.raises(ValueError, match="differs"):
            sample_batch(prior,
                         identity,
                         never,
                         run_config,
                         run_seed,
                         resume=checkpoint)
    for key, value, message in (("particles", [],
                                 "shape"), ("weights", [-1.] * 8, "values"),
                                ("completed_stage", 10, "progress")):
        state = checkpoint.unpack()
        state[key] = value
        invalid = replace(checkpoint, state=json.dumps(state))
        with pytest.raises(ValueError, match=message):
            sample_batch(prior, identity, never, config, 0, resume=invalid)
    path = tmp_path / "checkpoint.json"
    checkpoint.save(path)
    assert SamplerCheckpoint.load(path) == checkpoint
    original = path.read_bytes()

    def failed_replace(*_args: object) -> None:
        raise OSError("write interrupted")

    monkeypatch.setattr("os.replace", failed_replace)
    with pytest.raises(OSError, match="write interrupted"):
        saved[-1].save(path)
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]
    damaged = json.loads(path.read_text())
    damaged["payload"] += " "
    path.write_text(json.dumps(damaged))
    with pytest.raises(ValueError, match="checksum"):
        SamplerCheckpoint.load(path)


def test_budget_does_not_publish_checkpoint_particles() -> None:
    """A resumable stage is neither a posterior nor extra evaluation budget."""
    prior = BoxPrior(("x", ), ((0., 1.), ))
    digest = content_digest(b"budget")
    identity = InferenceIdentity(digest, digest, digest, prior.digest, digest)
    config = SamplerConfig(particles=16,
                           temperatures=8,
                           moves=2,
                           max_evaluations=80)
    saved: List[SamplerCheckpoint] = []
    result = sample_batch(prior,
                          identity,
                          lambda _: 0.,
                          config,
                          0,
                          checkpoint=saved.append)
    assert result.status == "budget_exhausted"
    assert not result.samples and not result.weights
    resumed = sample_batch(prior,
                           identity,
                           lambda _: 0.,
                           config,
                           0,
                           resume=saved[-1])
    assert resumed == result
    saved.clear()
    unsupported = sample_batch(prior,
                               identity,
                               lambda _: -math.inf,
                               config,
                               0,
                               checkpoint=saved.append)
    assert unsupported.status == "no_particle_support"
    assert not saved
