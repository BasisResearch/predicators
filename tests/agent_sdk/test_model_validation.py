"""Recorded-action replay must expose bad predictions at deployed values."""
# pylint: disable=protected-access,import-outside-toplevel
from typing import cast

import pytest

from predicators import utils
from predicators.agent_sdk.belief_probe import BeliefProbe
from predicators.agent_sdk.synthesis_backend import SynthesisBackend
from predicators.agent_sdk.tools.context import ToolContext
from predicators.agent_sdk.tools.synthesis import create_synthesis_tools
from tests.agent_sdk.test_rollout_residuals import _NOOP_SIMULATOR, \
    _FakeApproach, _observed_trajectory


class _Workbench(_FakeApproach):
    """Use the real replay engine with a calibrated linear simulator."""

    def __init__(self):
        first = _observed_trajectory()
        second = _observed_trajectory()
        for state in second.states:
            for obj in state:
                state.set(obj, "x", 2 * state.get(obj, "x"))
        super().__init__([first, second])
        self._fitted_params = {"dummy_k": 0.7}
        self._identified_physical_params = {"friction": 1.0}

    def _probe_fit_state(self):
        return {}

    def _rollout_fit_trajectories(self,
                                  residual_features=None,
                                  traj_idxs=None):
        del residual_features
        indices = range(len(self._fit_trajectories)) if traj_idxs is None \
            else traj_idxs
        return [(list(self._fit_trajectories[i].states),
                 list(self._fit_trajectories[i].actions)) for i in indices]


def test_validation_uses_deployed_values_and_keeps_bad_recordings(tmp_path):
    """A good fit on one recording must not hide a contradictory second one."""
    utils.reset_config({"seed": 0})
    artifact = tmp_path / "simulator.py"
    artifact.write_text(_NOOP_SIMULATOR, encoding="utf-8")
    approach = _Workbench()
    toolkit = create_synthesis_tools(exec_ns={},
                                     base_pred_triples=[],
                                     inferred_residual_features={},
                                     simulator_file=str(artifact),
                                     versions_dir=str(tmp_path / "versions"),
                                     approach=cast(SynthesisBackend, approach))
    ctx = ToolContext(probe_validation_provider=toolkit.validation_runner)
    sim = BeliefProbe(ctx)
    report = sim.validate()
    assert '"friction": 1.0' in report
    assert '"dummy_k": 0.7' in report
    assert "trajectory 0: 6 actions, RMS=0, SSE=0" in report
    assert "trajectory 1: 6 actions, RMS=0," not in report
    assert "trajectory 1: 6 actions" in report
    assert "All selected recordings" in report
    # Explicit hypotheses can validate an exploratory subset fit, without
    # silently deploying it or making a fit call.
    held = sim.validate(traj_idxs=[1], params={"friction": 2.0})
    assert "trajectory 0:" not in held
    assert "trajectory 1: 6 actions, RMS=0, SSE=0" in held
    assert "explicit diagnostic overrides (nothing deployed)" in held
    assert approach._identified_physical_params == {"friction": 1.0}
    assert "Error:" in sim.validate(traj_idxs=[])
    assert "Error:" in sim.validate(traj_idxs=[9])
    assert "Error:" in sim.validate(params={"secret": 0.0})
    assert "Error:" in sim.validate(params={"friction": float("nan")})


def test_validation_unavailable_without_workbench():
    """Solve-only and model-free sessions do not acquire replay access."""
    with pytest.raises(RuntimeError, match="no candidate replay workbench"):
        BeliefProbe(ToolContext()).validate()


def test_failed_replay_cannot_improve_candidate_rank(tmp_path, monkeypatch):
    """A candidate that throws on one recording has no aggregate score."""
    utils.reset_config({"seed": 0})
    from predicators.agent_sdk import model_validation
    from predicators.agent_sdk.tools.synthesis import moving_feature_scope
    approach = _Workbench()
    whole = approach._rollout_fit_trajectories()
    real = model_validation.compute_rollout_residuals
    calls = []

    def fail_second(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise ValueError("candidate failed")
        return real(*args, **kwargs)

    monkeypatch.setattr(model_validation, "compute_rollout_residuals",
                        fail_second)
    report = model_validation.replay_report(approach._get_rollout_fit_env(),
                                            whole, list(enumerate(whole)),
                                            moving_feature_scope(whole),
                                            {"friction": 1.0}, ["friction"],
                                            [], None, "fitted")
    assert "trajectory 1: REPLAY FAILED" in report
    assert "Aggregate unavailable" in report
    assert "All selected recordings:" not in report
    del tmp_path
