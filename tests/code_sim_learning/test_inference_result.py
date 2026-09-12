"""Compatibility at the versioned legacy inference boundary."""
import pickle
from dataclasses import asdict

import numpy as np

from predicators.code_sim_learning.fit_space import FitResult
from predicators.code_sim_learning.orchestrator import SysIdOutcome


def test_no_fit_and_historical_outcomes_remain_unpublished():
    """Pinned estimates must not become selected values or invented certainty.

    Historical outcome dictionaries have no inference member. The
    adapter must work after loading them without changing their stored
    payload.
    """
    result = FitResult(["gain"], np.array([[1.0]]), np.zeros(1))
    outcome = SysIdOutcome(result, {}, {"gain": 1.0}, {}, 2, 0, [3.0, 4.0],
                           5.0, float("nan"))
    before = pickle.dumps(outcome)
    restored = pickle.loads(before)
    inference = restored.inference
    assert inference.point_estimate == {"gain": 1.0}
    assert inference.selected_parameters == {}
    assert inference.parameter_diagnostics == {}
    assert inference.num_survivors == 0
    assert inference.num_segments == 2
    assert asdict(inference)["schema_version"] == 1
    assert asdict(inference)["uncertainty_kind"] == "legacy_widths"
    assert "inference" not in restored.__dict__
    assert pickle.dumps(outcome) == before
