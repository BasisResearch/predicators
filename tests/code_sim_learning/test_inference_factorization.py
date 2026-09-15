"""Checked product posteriors preserve analytic factors and sampled joints."""
from dataclasses import replace

import numpy as np
import pytest

from predicators.code_sim_learning.inference_assessment import \
    AssessedInference, AssessmentProtocol, InferenceCheck, assess_inference
from predicators.code_sim_learning.inference_data import InferenceIdentity, \
    content_digest
from predicators.code_sim_learning.inference_factorization import \
    CheckedFactorization, FactorizedParameterPosterior, \
    IndependentPriorFactorization
from predicators.code_sim_learning.inference_parameters import \
    UnavailableParameterPosterior
from predicators.code_sim_learning.inference_sampling import BatchPosterior, \
    BoxPrior, SamplerConfig


def _source() -> AssessedInference:
    # Known discrete joint: b=a*a, with one zero-mass outlier.
    prior = BoxPrior(("a", "start", "b"),
                     ((-10., 10.), (-100., 100.), (0., 100.)))
    digest = content_digest(b"enumerated joint")
    identity = InferenceIdentity(digest, digest, digest, prior.digest, digest)
    posterior = BatchPosterior(identity, prior, SamplerConfig(particles=4), 0,
                               "complete", ((-9., 80., 81.), (-2., 3., 4.),
                                            (1., 4., 1.), (3., 5., 9.)),
                               (0., .1, .3, .6), 4, 1., 3, (2., ), 0, 0, 3, 0)
    protocol = AssessmentProtocol(digest, ("enumeration", ))
    checks = (InferenceCheck("enumeration", "pass", "Exact reference",
                             digest), )
    predictive = (InferenceCheck("future", "fail", "Missing dynamics",
                                 digest), )
    return assess_inference(posterior, protocol, checks, predictive)


def _result() -> FactorizedParameterPosterior:
    source = _source()
    full = replace(source.identity, prior=content_digest(b"product prior"))
    declaration = IndependentPriorFactorization(
        full, source.identity, ("a", "start", "b"),
        BoxPrior(("radius", "heat"), ((.04, .25), (5., 80.))))
    check = InferenceCheck("independence", "pass", "Exact product reference",
                           content_digest(b"factor proof"))
    factor = CheckedFactorization(declaration, declaration.digest, check)
    predictive = (InferenceCheck("future", "fail", "Prior extrapolation",
                                 content_digest(b"full prediction")), )
    return FactorizedParameterPosterior(source,
                                        factor, ("a", "b", "radius", "heat"),
                                        predictive_checks=predictive)


def test_product_forecasts_preserve_dependence_and_prior_law() -> None:
    """A known nonlinear joint times two uniforms yields reference moments."""
    result = _result()
    ensemble = result.resample(20000, 17)
    rows = np.asarray(ensemble.values)
    assert ensemble == result.resample(20000, 17)
    assert ensemble != result.resample(20000, 18)
    assert set(ensemble.source_indices) == {1, 2, 3}
    assert ensemble.weights == (1 / 20000, ) * 20000
    assert ensemble.resampling_seed == 17
    assert np.all(rows[:, 1] == rows[:, 0]**2)
    assert np.mean(rows[:, 0]) == pytest.approx(1.9, abs=.04)
    assert np.mean(rows[:, 2]) == pytest.approx(.145, abs=.002)
    assert np.mean(rows[:, 3]) == pytest.approx(42.5, abs=.5)
    assert np.all((rows[:, 2] >= .04) & (rows[:, 2] <= .25))
    assert np.all((rows[:, 3] >= 5.) & (rows[:, 3] <= 80.))
    # Joint event separates retained mass from both independent factors.
    event = (rows[:, 0] == 3.) & (rows[:, 2] < .145) & (rows[:, 3] < 42.5)
    assert ensemble.expectation(event) == pytest.approx(.15, abs=.01)
    assert ensemble.identity == result.identity
    assert ensemble.identity != result.reduced.identity
    assert [(check.name, check.status) for check in ensemble.predictive_checks
            ] == [("reduced/future", "fail"), ("full/future", "fail")]
    assert ensemble.assessment_protocol != result.reduced.protocol.source
    maps = ensemble.as_dicts()
    maps[0]["heat"] = -100.
    assert ensemble.as_dicts()[0]["heat"] >= 5.


def test_exact_quantiles_and_coordinate_aliases() -> None:
    """Analytic intervals do not inherit accidental particle concentration."""
    result = _result()
    quantiles = result.marginal_quantiles((0., .5, 1.))
    assert quantiles == {
        "a": (-2., 3., 3.),
        "b": (1., 9., 9.),
        "radius": (.04, .145, .25),
        "heat": (5., 42.5, 80.)
    }
    aliases = replace(result,
                      names=("left", "right", "x", "y"),
                      coordinates=("heat", "heat", "a", "a"))
    assert all(left == right and x == y
               for left, right, x, y in aliases.resample(100, 9).values)
    empty = replace(result, names=(), coordinates=())
    assert not empty.marginal_quantiles()
    assert empty.resample(2, 1).as_dicts() == ({}, {})


@pytest.mark.parametrize("status", ["numerical_failure", "unevaluated"])
def test_analytic_factor_cannot_bypass_unavailable_fit(status) -> None:
    """Even analytic-only requests require an available whole posterior."""
    result = _result()
    unavailable = replace(result.reduced, availability=status, posterior=None)
    result = replace(result,
                     reduced=unavailable,
                     names=("heat", ),
                     coordinates=("heat", ))
    with pytest.raises(UnavailableParameterPosterior, match=status):
        result.marginal_quantiles()
    with pytest.raises(UnavailableParameterPosterior, match=status):
        result.resample(2, 1)
    assert result.reduced.predictive_checks[0].status == "fail"


@pytest.mark.parametrize("status", ["fail", "unevaluated"])
def test_unverified_factor_cannot_supply_parameters(status) -> None:
    """An independence declaration alone is insufficient for use."""
    result = _result()
    check = replace(result.factorization.check, status=status)
    result = replace(result,
                     factorization=replace(result.factorization, check=check))
    with pytest.raises(UnavailableParameterPosterior, match=status):
        result.marginal_quantiles()
    with pytest.raises(UnavailableParameterPosterior, match=status):
        result.resample(2, 1)


def test_factor_evidence_cannot_follow_changed_target_or_law() -> None:
    """New observations, program, prior law or schema invalidate old checks."""
    result = _result()
    factor = result.factorization
    declaration = factor.declaration
    changed = content_digest(b"new data or program")
    declarations = [
        replace(declaration, retained_coordinates=("b", "start", "a")),
        replace(declaration,
                independent_prior=BoxPrior(("radius", "heat"),
                                           ((.04, .3), (5., 80.))))
    ]
    for field in ("data", "sensor", "program"):
        with pytest.raises(ValueError, match="changes data"):
            replace(declaration,
                    full_identity=replace(declaration.full_identity,
                                          **{field: changed}))
        declarations.append(
            replace(declaration,
                    full_identity=replace(declaration.full_identity,
                                          **{field: changed}),
                    reduced_identity=replace(declaration.reduced_identity,
                                             **{field: changed})))
    for revised in declarations:
        with pytest.raises(ValueError, match="different scope"):
            replace(factor, declaration=revised)
    wrong_schema = declarations[0]
    checked = replace(factor,
                      declaration=wrong_schema,
                      checked_declaration=wrong_schema.digest)
    with pytest.raises(ValueError, match="schema differs"):
        replace(result, factorization=checked)
    new_data = declarations[-1]
    checked = replace(factor,
                      declaration=new_data,
                      checked_declaration=new_data.digest)
    with pytest.raises(ValueError, match="source disagree"):
        replace(result, factorization=checked)


def test_invalid_sources_and_requests_reject() -> None:
    """Malformed or falsely available results never reach a consumer."""
    result = _result()
    with pytest.raises(ValueError, match="needs a posterior"):
        replace(result, reduced=replace(result.reduced, posterior=None))
    with pytest.raises(ValueError, match="available posterior"):
        replace(result, reduced=replace(result.reduced, numerical_checks=()))
    for coordinates in (("missing", ) * 4, ("a", )):
        with pytest.raises(ValueError):
            replace(result, coordinates=coordinates)
    with pytest.raises(ValueError, match="distinct"):
        replace(result, names=("a", ) * 4)
    with pytest.raises(ValueError, match="Duplicate"):
        replace(result, predictive_checks=result.predictive_checks * 2)
    declaration = result.factorization.declaration
    with pytest.raises(ValueError, match="overlap"):
        replace(declaration, independent_prior=BoxPrior(("a", ), ((0., 1.), )))
    for coordinates in ((), ("a", "a"), ("", )):
        with pytest.raises(ValueError, match="distinct"):
            replace(declaration, retained_coordinates=coordinates)
    for count, seed in ((0, 1), (True, 1), (1, -1), (1, True)):
        with pytest.raises(ValueError):
            result.resample(count, seed)
    for probabilities in ((), (-.1, ), (1.1, ), (float("nan"), )):
        with pytest.raises(ValueError):
            result.marginal_quantiles(probabilities)
