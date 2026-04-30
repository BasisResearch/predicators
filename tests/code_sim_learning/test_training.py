"""Tests for code sim-learning training utilities."""

import numpy as np

from predicators import utils
from predicators.code_sim_learning.training import ParamSpec, fit_params


def test_fit_params_can_skip_training_with_cfg():
    """Test that CFG can disable parameter fitting."""
    utils.reset_config({"code_sim_learning_num_mcmc_steps": 0})
    param_specs = [ParamSpec("rate", 2.5), ParamSpec("threshold", 0.7)]

    result = fit_params(
        simulator_fn=lambda _s, _a, _p: {},
        transitions=[],
        param_specs=param_specs,
        process_features={},
    )

    assert result.point_estimate == {"rate": 2.5, "threshold": 0.7}
    np.testing.assert_allclose(result.samples, np.array([[2.5, 0.7]]))
    np.testing.assert_allclose(result.log_probs, np.array([0.0]))
