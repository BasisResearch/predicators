"""Evidence checking distinguishes proven contradictions from search
failure."""
from dataclasses import replace

import pytest

from predicators.code_sim_learning.inference_data import EpisodeData, \
    InferenceData, Observation, SensorFeature, SensorModel, content_digest
from predicators.code_sim_learning.inference_support import ConstantOutputs, \
    SupportAssessment, audit_constant_outputs

KEY = ("block", "block", "glue")
DIGEST = content_digest(b"reviewed constant model and runtime")


def _audit(data: InferenceData, sensor: SensorModel) -> SupportAssessment:
    declaration = ConstantOutputs(DIGEST, DIGEST, DIGEST, (KEY, ))
    return audit_constant_outputs(data,
                                  sensor,
                                  declaration,
                                  program_digest=DIGEST,
                                  runtime_digest=DIGEST)


def test_exact_changes_give_witnesses_without_sampling() -> None:
    """A frozen constant output cannot match both recorded exact values."""
    observations = tuple(
        Observation(i, ((KEY, v), )) for i, v in enumerate((0., .2, 1.)))
    data = InferenceData((EpisodeData("reset0", ((0., ), ) * 2,
                                      observations), ))
    result = _audit(data, SensorModel((SensorFeature(KEY, 0.), )))
    assert result.status == "model_inconsistent"
    witness, = result.contradictions
    assert (witness.first_step, witness.later_step) == (0, 1)
    assert (witness.first_value, witness.later_value) == (0., .2)
    assert result.data == data.digest


def test_resets_and_missing_data_do_not_manufacture_contradictions() -> None:
    """A different reset may have a different constant and missing reads."""
    data = InferenceData(
        tuple(
            EpisodeData(str(i), ((0., ), ), (Observation(0, ()),
                                             Observation(1, ((KEY, v), ))))
            for i, v in enumerate((0., 1.))))
    result = _audit(data, SensorModel((SensorFeature(KEY, 0.), )))
    assert result.status == "not_disproved" and not result.contradictions


def test_invariant_scope_and_evidence_semantics_are_checked() -> None:
    """Noise and conditioned external inputs cannot prove this
    contradiction."""
    data = InferenceData(())
    for feature in (SensorFeature(KEY, .1), SensorFeature(KEY, 0., True)):
        with pytest.raises(ValueError, match="exact predicted"):
            _audit(data, SensorModel((feature, )))
    declaration = ConstantOutputs(DIGEST, DIGEST, DIGEST, (KEY, ))
    with pytest.raises(ValueError, match="program/runtime"):
        audit_constant_outputs(data,
                               SensorModel((SensorFeature(KEY, 0.), )),
                               replace(declaration,
                                       program=content_digest(b"edited")),
                               program_digest=DIGEST,
                               runtime_digest=DIGEST)
    with pytest.raises(ValueError, match="distinct"):
        replace(declaration, keys=(KEY, KEY))
