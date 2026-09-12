"""Inference availability must not be confused with predictive adequacy."""
from dataclasses import replace
from typing import Tuple

import pytest

from predicators.code_sim_learning.inference_assessment import \
    AssessmentProtocol, InferenceCheck, assess_inference
from predicators.code_sim_learning.inference_data import InferenceIdentity, \
    content_digest
from predicators.code_sim_learning.inference_sampling import BatchPosterior, \
    BoxPrior, SamplerConfig, sample_batch


def _candidate(*, impossible: bool = False) -> BatchPosterior:
    prior = BoxPrior(("theta", ), ((-1., 1.), ))
    digest = content_digest(b"assessment reference")
    identity = InferenceIdentity(digest, digest, digest, prior.digest, digest)
    return sample_batch(prior,
                        identity,
                        lambda _: -float("inf") if impossible else 0.,
                        SamplerConfig(particles=16,
                                      temperatures=2,
                                      moves=1,
                                      max_evaluations=100),
                        seed=0)


def _protocol() -> AssessmentProtocol:
    return AssessmentProtocol(content_digest(b"test protocol"),
                              ("reference", "repeatability"))


def _passed() -> Tuple[InferenceCheck, ...]:
    evidence = content_digest(b"test evidence")
    return tuple(
        InferenceCheck(name, "pass", "Test reference passed", evidence)
        for name in _protocol().required_numerical_checks)


def test_predictive_failure_keeps_numerically_available_inference() -> None:
    """A poor held-out prediction cannot silently suppress a computed fit."""
    candidate = _candidate()
    bad_prediction = InferenceCheck(
        "episode0.future.box.speed", "fail", "Held-out speed is inconsistent",
        content_digest(b"held-out prediction report"))
    result = assess_inference(candidate, _protocol(), _passed(),
                              (bad_prediction, ))
    assert result.availability == "available"
    assert result.posterior is candidate
    assert result.predictive_checks == (bad_prediction, )
    assert result.posterior.marginal_quantiles("theta") == \
        candidate.marginal_quantiles("theta")
    assert result.identity == candidate.identity


def test_missing_or_failed_numerics_do_not_expose_samples() -> None:
    """Finished SMC and passing predictions do not replace numerical checks."""
    candidate = _candidate()
    predictive = (InferenceCheck("future.event", "pass", "Event matches",
                                 content_digest(b"prediction")), )
    missing = assess_inference(candidate, _protocol(),
                               _passed()[:1], predictive)
    assert missing.availability == "unevaluated"
    assert missing.posterior is None
    assert missing.numerical_checks[-1].status == "unevaluated"
    failed = replace(_passed()[1],
                     status="fail",
                     detail="Independent fits differ")
    result = assess_inference(candidate, _protocol(), (_passed()[0], failed),
                              predictive)
    assert result.availability == "numerical_failure"
    assert result.posterior is None
    assert result.predictive_checks == predictive
    assert candidate.samples  # The separate diagnostic artifact is intact.


def test_sampler_failure_cannot_be_overridden_by_assessment() -> None:
    """No-support sampler outcomes stay unavailable despite caller verdicts."""
    candidate = _candidate(impossible=True)
    assert candidate.status == "no_particle_support"
    result = assess_inference(candidate, _protocol(), _passed())
    assert result.availability == "numerical_failure"
    assert result.posterior is None
    assert result.sampler_status == "no_particle_support"


def test_assessment_rejects_missing_evidence_and_protocol_drift() -> None:
    """Reports cannot silently omit their evidence or change required
    checks."""
    candidate = _candidate()
    with pytest.raises(ValueError, match="evidence"):
        InferenceCheck("reference", "pass", "No report")
    with pytest.raises(ValueError, match="SHA256"):
        AssessmentProtocol("unidentified", ("reference", ))
    with pytest.raises(ValueError, match="distinct"):
        AssessmentProtocol(content_digest(b"protocol"), ())
    with pytest.raises(ValueError, match="Duplicate"):
        assess_inference(candidate, _protocol(), (_passed()[0], ) * 2)
    with pytest.raises(ValueError, match="absent"):
        assess_inference(candidate, _protocol(),
                         (replace(_passed()[0], name="ESS only"), ))


@pytest.mark.parametrize("changes", [
    {
        "weights": (1., )
    },
    {
        "weights": (0., ) * 16
    },
    {
        "weights": (float("nan"), ) * 16
    },
    {
        "samples": ((float("nan"), ), ) * 16
    },
    {
        "samples": ((0., 1.), ) * 16
    },
    {
        "completed_temperature": .5
    },
])
def test_malformed_sampler_artifacts_are_not_published(changes: dict) -> None:
    """Deserialized or constructed invalid particles are not usable fits."""
    with pytest.raises(ValueError):
        assess_inference(replace(_candidate(), **changes), _protocol(),
                         _passed())
