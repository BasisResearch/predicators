"""Posterior consumers retain weights, correlations and assessment status."""
from dataclasses import replace

import numpy as np
import pytest

from predicators.code_sim_learning.inference_assessment import \
    AssessedInference, AssessmentProtocol, InferenceCheck, assess_inference
from predicators.code_sim_learning.inference_data import InferenceIdentity, \
    content_digest
from predicators.code_sim_learning.inference_parameters import \
    ParameterPosterior, UnavailableParameterPosterior
from predicators.code_sim_learning.inference_sampling import BatchPosterior, \
    BoxPrior, SamplerConfig, sample_batch


def _assessment() -> AssessedInference:
    # Exact enumerated reference: b = a**2 in three discrete modes.
    # A zero-mass fourth row must not appear in intervals or ensembles.
    prior = BoxPrior(("a", "episode0.start", "b"),
                     ((-10., 10.), (-100., 100.), (0., 100.)))
    digest = content_digest(b"enumerated joint reference")
    identity = InferenceIdentity(digest, digest, digest, prior.digest, digest)
    candidate = BatchPosterior(identity, prior, SamplerConfig(particles=4), 0,
                               "complete", ((-9., 80., 81.), (-2., 3., 4.),
                                            (1., 4., 1.), (3., 5., 9.)),
                               (0., .1, .3, .6), 4, 1., 3, (2., ), 0, 0, 3, 0)
    protocol = AssessmentProtocol(
        content_digest(b"enumerated reference checks"), ("exact", ))
    checks = (InferenceCheck("exact", "pass", "Weights are specified exactly",
                             digest), )
    predictive = (InferenceCheck("future", "fail", "Incomplete dynamics",
                                 content_digest(b"prediction failure")), )
    return assess_inference(candidate, protocol, checks, predictive)


def test_projection_and_quantiles_share_weighted_joint_reference() -> None:
    """Projection preserves exact empirical masses and owned parameter maps."""
    result = ParameterPosterior(_assessment(), ("b", "a"))
    ensemble = result.weighted_samples()
    assert ensemble.names == ("b", "a")
    assert ensemble.values == ((4., -2.), (1., 1.), (9., 3.))
    assert ensemble.weights == (.1, .3, .6)
    assert ensemble.source_indices == (1, 2, 3)
    assert ensemble.identity == result.assessment.identity
    assert ensemble.predictive_checks == result.assessment.predictive_checks
    assert ensemble.predictive_checks[0].status == "fail"
    assert result.marginal_quantiles((0., .5, 1.)) == {
        "a": (-2., 3., 3.),
        "b": (1., 9., 9.)
    }
    assert ensemble.expectation([False, False, True]) == pytest.approx(.6)
    assert ensemble.expectation([row[1] for row in ensemble.values]) == \
        pytest.approx(1.9)
    maps = ensemble.as_dicts()
    maps[0]["a"] = 999.
    assert ensemble.as_dicts()[0]["a"] == -2.
    assert result.weighted_samples() == ensemble


def test_resampling_preserves_modes_and_parameter_dependence() -> None:
    """Whole-row draws retain a nonlinear relation across separated modes."""
    result = ParameterPosterior(_assessment(), ("a", "b"))
    ensemble = result.resample(10000, 42)
    assert ensemble == result.resample(10000, 42)
    assert ensemble != result.resample(10000, 43)
    assert ensemble.weights == (1 / 10000, ) * 10000
    assert ensemble.resampling_seed == 42
    assert set(ensemble.source_indices) == {1, 2, 3}
    assert all(b == a**2 for a, b in ensemble.values)
    assert ensemble.expectation([a == 3 for a, _ in ensemble.values]) == \
        pytest.approx(.6, abs=.02)
    assert ensemble.expectation([a for a, _ in ensemble.values]) == \
        pytest.approx(1.9, abs=.06)
    assert ensemble.predictive_checks[0].status == "fail"


def test_assessed_ensemble_drives_information_score() -> None:
    """A posterior mode's probability, not its row count, weights a probe."""
    result = ParameterPosterior(_assessment(), ("a", "b"))
    ensemble = result.weighted_samples()
    # The proposed probe distinguishes only a=3, whose posterior mass is .6.
    reads = np.array([[row["a"] == 3.] for row in ensemble.as_dicts()])
    entropy = -.6 * np.log2(.6) - .4 * np.log2(.4)
    assert ensemble.atom_information(reads) == pytest.approx(entropy)
    # A binary symmetric read-error channel with crossover probability .2.
    noisy_reads = .2 + .6 * reads
    marginal = .6 * .8 + .4 * .2
    reference = -marginal * np.log2(marginal) - \
        (1. - marginal) * np.log2(1. - marginal) + \
        .2 * np.log2(.2) + .8 * np.log2(.8)
    assert ensemble.atom_information(noisy_reads) == pytest.approx(reference)
    assert ensemble.predictive_checks[0].status == "fail"
    with pytest.raises(ValueError, match="weight"):
        ensemble.atom_information(reads[:2])


def test_explicit_simulator_names_and_shared_coordinates() -> None:
    """Simulator names are explicitly mapped to the joint inference schema."""
    result = ParameterPosterior(_assessment(), ("force", "drag"), ("b", "a"))
    weighted = result.weighted_samples()
    assert weighted.source_coordinates == ("b", "a")
    assert weighted.as_dicts()[0] == {"force": 4., "drag": -2.}
    assert result.marginal_quantiles((.5, )) == {
        "force": (9., ),
        "drag": (3., )
    }
    assert result.resample(4, 1).source_coordinates == ("b", "a")
    shared = ParameterPosterior(_assessment(), ("left", "right"), ("a", "a"))
    assert all(left == right for left, right in shared.resample(16, 0).values)
    with pytest.raises(ValueError, match="One joint coordinate"):
        ParameterPosterior(_assessment(), ("force", "drag"), ("a", ))
    with pytest.raises(ValueError, match="Unknown"):
        ParameterPosterior(_assessment(), ("force", ), ("theta.missing", ))


@pytest.mark.parametrize("status", ["numerical_failure", "unevaluated"])
def test_unavailable_inference_keeps_diagnostics_without_parameters(status):
    """Incomplete numerical assessment cannot yield plausible-looking
    widths."""
    source = _assessment()
    unavailable = replace(source, availability=status, posterior=None)
    result = ParameterPosterior(unavailable, ("a", ))
    assert result.assessment.predictive_checks == source.predictive_checks
    for operation in (result.weighted_samples, result.marginal_quantiles,
                      lambda: result.resample(2, 0)):
        with pytest.raises(UnavailableParameterPosterior, match=status):
            operation()


def test_projection_checks_assessment_identity_and_parameter_schema() -> None:
    """Invalid declarations fail before any parameter row reaches a
    consumer."""
    source = _assessment()
    for names in (("a", "a"), ("missing", ), ("", )):
        with pytest.raises(ValueError):
            ParameterPosterior(source, names)
    with pytest.raises(ValueError, match="needs a posterior"):
        ParameterPosterior(replace(source, posterior=None), ("a", ))
    with pytest.raises(ValueError, match="identity"):
        ParameterPosterior(
            replace(source,
                    identity=replace(source.identity,
                                     data=content_digest(b"different data"))),
            ("a", ))
    with pytest.raises(ValueError, match="available posterior"):
        ParameterPosterior(replace(source, numerical_checks=()), ("a", ))
    assert source.posterior is not None
    assert isinstance(source.posterior.prior, BoxPrior)
    changed_prior = replace(source.posterior.prior,
                            bounds=((-11., 11.), (-100., 100.), (0., 100.)))
    with pytest.raises(ValueError, match="identity"):
        ParameterPosterior(
            replace(source,
                    posterior=replace(source.posterior, prior=changed_prior)),
            ("a", ))
    result = ParameterPosterior(source, ("a", ))
    for count, seed in ((0, 1), (True, 1), (1, -1), (1, True)):
        with pytest.raises(ValueError):
            result.resample(count, seed)
    for probabilities in ((), (-.1, ), (float("nan"), )):
        with pytest.raises(ValueError):
            result.marginal_quantiles(probabilities)


def test_empty_parameter_projection_and_malformed_ensemble() -> None:
    """Parameterless models remain explicit and malformed weights reject."""
    source = _assessment()
    empty = ParameterPosterior(source, ())
    assert empty.marginal_quantiles() == {}
    assert empty.weighted_samples().as_dicts() == ({}, {}, {})
    assert empty.weighted_samples().expectation([1., 1., 1.]) == 1.
    ensemble = ParameterPosterior(source, ("a", )).weighted_samples()
    for outcomes in ([1.], [1., 2., float("nan")]):
        with pytest.raises(ValueError, match="outcome"):
            ensemble.expectation(outcomes)
    with pytest.raises(ValueError, match="normalized"):
        replace(ensemble, weights=(.2, .3, .6))
    with pytest.raises(ValueError, match="dimensions"):
        replace(ensemble, values=((0., 1.), ) * 3)
    with pytest.raises(ValueError, match="source particle"):
        replace(ensemble, source_indices=(-1, 2, 3))


def test_sampled_inference_to_correlated_prediction() -> None:
    """A noisy sum observation flows through assessment into joint
    forecasts."""
    prior = BoxPrior(("a", "unobserved_start", "b"), ((0., 1.), ) * 3)
    digest = content_digest(b"sum observation: one plus Gaussian noise .05")
    identity = InferenceIdentity(digest, digest, digest, prior.digest, digest)
    posterior = sample_batch(
        prior, identity, lambda point: -.5 *
        ((point[0] + point[2] - 1.) / .05)**2,
        SamplerConfig(particles=1024,
                      temperatures=16,
                      moves=4,
                      max_evaluations=70000), 51)
    assert posterior.status == "complete"
    values = np.asarray(posterior.samples)
    weights = np.asarray(posterior.weights)
    # Symmetry about a+b=1 gives the first reference moment exactly.
    # The unused coordinate is independent and retains its uniform prior.
    assert np.dot(weights, values[:, 0] + values[:, 2]) == \
        pytest.approx(1., abs=.015)
    assert np.dot(weights, values[:, 1]) == pytest.approx(.5, abs=.05)
    protocol = AssessmentProtocol(content_digest(b"two analytic mean checks"),
                                  ("sum_mean", "uninformed_mean"))
    checks = tuple(
        InferenceCheck(name, "pass", "Analytic mean matched", digest)
        for name in protocol.required_numerical_checks)
    result = ParameterPosterior(assess_inference(posterior, protocol, checks),
                                ("a", "b"))
    weighted = result.weighted_samples()
    success = [abs(a + b - 1.) < .15 for a, b in weighted.values]
    assert weighted.expectation(success) > .97
    resampled = result.resample(5000, 16)
    success = [abs(a + b - 1.) < .15 for a, b in resampled.values]
    assert resampled.expectation(success) > .97
    # Independent marginal recombination destroys this predictive relation.
    rows = np.asarray(resampled.values)
    shuffled = np.random.default_rng(1).permutation(rows[:, 1])
    assert np.mean(np.abs(rows[:, 0] + shuffled - 1.) < .15) < .5
