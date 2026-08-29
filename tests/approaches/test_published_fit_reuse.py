"""The deployed model is the agent's published ``sim.fit``.

``_publish_probe_fit`` keeps the full FitResult; after the session the
approach reuses it when it fitted exactly the final simulator.py over
exactly the deployed parameter set, and falls back to a harness fit
otherwise.
"""
# pylint: disable=protected-access
from typing import Any

import numpy as np

from predicators.agent_sdk.tools import ToolContext
from predicators.approaches.agent_sim_learning_approach import \
    AgentSimLearningApproach
from predicators.code_sim_learning.fit_space import FitResult


def _fit(names: Any, values: Any) -> FitResult:
    return FitResult(names=list(names),
                     samples=np.array([values], dtype=float),
                     log_probs=np.zeros(1),
                     jacobian=np.ones((3, len(names))),
                     noise_sigma=0.1,
                     prior_sigma=np.ones(len(names)))


def _approach() -> AgentSimLearningApproach:
    approach = object.__new__(AgentSimLearningApproach)
    approach._fitted_params = {}
    approach._tool_context = ToolContext()
    return approach


def test_published_fit_is_reused_for_the_fitted_file(tmp_path: Any) -> None:
    """A canonical fit of the current file over the deployed parameter set is
    returned with its SSE and version; a later edit or a changed parameter set
    makes it stale."""
    sim_file = tmp_path / "simulator.py"
    sim_file.write_text("RESIDUAL_RULES = []\n", encoding="utf-8")
    approach = _approach()
    assert approach._published_fit_for_file(str(sim_file), ["k"]) is None

    fit = _fit(["k", "gap"], [1.5, 0.02])
    approach._publish_probe_fit({
        "k": 1.5,
        "gap": 0.02
    },
                                "cycle_001_vers_003",
                                str(sim_file),
                                fit_result=fit,
                                sse=0.25)
    assert approach._fitted_params == {"k": 1.5, "gap": 0.02}
    assert approach._tool_context.probe_param_status == \
        "fitted (cycle_001_vers_003)"
    published = approach._published_fit_for_file(str(sim_file), ["gap", "k"])
    assert published is not None
    got, sse, version = published
    assert got is fit
    assert sse == 0.25
    assert version == "cycle_001_vers_003"

    # A different parameter set (spec added after the fit) is stale.
    assert approach._published_fit_for_file(str(sim_file),
                                            ["k", "gap", "mu"]) is None
    # An UNFITTED edit of the file is stale.
    sim_file.write_text("RESIDUAL_RULES = []  # edited\n", encoding="utf-8")
    assert approach._published_fit_for_file(str(sim_file), ["k", "gap"]) \
        is None


def test_publish_without_a_fit_result_never_deploys(tmp_path: Any) -> None:
    """Legacy publishes (values only) deploy to the probe but cannot stand in
    for the cycle's fit."""
    sim_file = tmp_path / "simulator.py"
    sim_file.write_text("RESIDUAL_RULES = []\n", encoding="utf-8")
    approach = _approach()
    approach._publish_probe_fit({"k": 2.0}, "cycle_001_vers_001",
                                str(sim_file))
    assert approach._fitted_params == {"k": 2.0}
    assert approach._published_fit_for_file(str(sim_file), ["k"]) is None
