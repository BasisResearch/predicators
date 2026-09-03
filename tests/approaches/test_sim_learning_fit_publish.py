"""Tests for the canonical-fit publish path of the sim-learning approach.

A ``sim.fit`` that ran no fit (rollout sysID found nothing explainable
at any candidate parameters) is published PINNED at the declared inits.
It must never displace a real fit of the same simulator.py content, and
the deploy path must be able to tell the two apart.
"""
# pylint: disable=protected-access
import math
from types import SimpleNamespace
from typing import Any, Dict

import numpy as np

from predicators.approaches.agent_sim_learning_approach import \
    AgentSimLearningApproach
from predicators.code_sim_learning.fit_space import FitResult

# The methods under test only touch the fit-state store, the fitted
# params, the probe model cache, and the tool context's status line, so
# they run on a bare stub; the class is typed Any to call them unbound.
_APPROACH: Any = AgentSimLearningApproach


class _Stub:
    """Just the state the publish / lookup methods touch."""

    def __init__(self) -> None:
        self._fitted_params: Dict[str, float] = {}
        self._tool_context = SimpleNamespace(probe_param_status=None)
        self._cache: Dict[str, Any] = {}
        self._state: Dict[str, Any] = {}

    def _probe_model_cache(self) -> Dict[str, Any]:
        return self._cache

    def _probe_fit_state(self) -> Dict[str, Any]:
        return self._state


def _fit(values: Dict[str, float]) -> FitResult:
    names = sorted(values)
    return FitResult(names=names,
                     samples=np.array([[values[n] for n in names]]),
                     log_probs=np.zeros(1))


def _publish(stub: _Stub, values: Dict[str, float], version: str,
             sim_file: Any, sse: float, pinned: bool) -> None:
    _APPROACH._publish_probe_fit(stub,
                                 dict(values),
                                 version,
                                 str(sim_file),
                                 fit_result=_fit(values),
                                 sse=sse,
                                 pinned=pinned)


def test_pinned_fit_does_not_displace_a_finite_fit_of_the_same_file(
        tmp_path: Any) -> None:
    """A no-survivor refit of unchanged content leaves the earlier real fit
    canonical, so the deployed model is never downgraded to inits."""
    sim_file = tmp_path / "simulator.py"
    sim_file.write_text("RULES = []\n", encoding="utf-8")
    stub = _Stub()
    _publish(stub, {"k": 2.0}, "cycle_000_vers_001", sim_file, 0.5, False)
    _publish(stub, {"k": 1.0}, "cycle_000_vers_002", sim_file, 77.4, True)
    published = _APPROACH._published_fit_for_file(stub, str(sim_file), ["k"])
    assert published is not None
    fit, sse, version = published
    assert version == "cycle_000_vers_001"
    assert sse == 0.5
    assert fit.point_estimate["k"] == 2.0
    assert stub._fitted_params == {"k": 2.0}
    assert not _APPROACH._published_fit_is_pinned(stub)


def test_pinned_fit_is_canonical_when_nothing_finite_exists(
        tmp_path: Any) -> None:
    """With no real fit to fall back on, the pinned fit is what the cycle
    deploys, flagged as pinned and carrying a finite SSE at the inits."""
    sim_file = tmp_path / "simulator.py"
    sim_file.write_text("RULES = []\n", encoding="utf-8")
    stub = _Stub()
    _publish(stub, {"k": 1.0}, "cycle_000_vers_001", sim_file, 77.4, True)
    published = _APPROACH._published_fit_for_file(stub, str(sim_file), ["k"])
    assert published is not None
    _, sse, version = published
    assert version == "cycle_000_vers_001"
    assert math.isfinite(sse) and sse == 77.4
    assert _APPROACH._published_fit_is_pinned(stub)


def test_pinned_fit_of_a_changed_file_replaces_the_old_fit(
        tmp_path: Any) -> None:
    """An earlier finite fit of DIFFERENT content is no fallback: the file the
    agent edited is the one the cycle deploys."""
    sim_file = tmp_path / "simulator.py"
    sim_file.write_text("RULES = []\n", encoding="utf-8")
    stub = _Stub()
    _publish(stub, {"k": 2.0}, "cycle_000_vers_001", sim_file, 0.5, False)
    sim_file.write_text("RULES = []  # edited\n", encoding="utf-8")
    _publish(stub, {"k": 1.0}, "cycle_000_vers_002", sim_file, 77.4, True)
    published = _APPROACH._published_fit_for_file(stub, str(sim_file), ["k"])
    assert published is not None
    assert published[2] == "cycle_000_vers_002"
    assert _APPROACH._published_fit_is_pinned(stub)
    assert stub._fitted_params == {"k": 1.0}
