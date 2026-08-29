"""Tests for the ``sim.predicates()`` loader (make_predicate_quality_loader).

Drives the loader through the probe with a stub approach and a fake
trajectory (no PyBullet): the report scores milestone behaviour and the
loaded draft replaces the approach's learned predicate set.
"""
# pylint: disable=protected-access
from typing import Any, Dict, Set, cast

import numpy as np
from gym.spaces import Box

from predicators.agent_sdk.belief_probe import BeliefProbe
from predicators.agent_sdk.tools import ToolContext, \
    make_predicate_quality_loader
from predicators.structs import Action, GroundAtom, LowLevelTrajectory, \
    Object, ParameterizedOption, Predicate, State, Task, Type

_block_type = Type("block", ["x"])
_block = Object("block0", _block_type)
_Kept = Predicate("Kept", [_block_type], lambda s, o: True)
_Move = ParameterizedOption(
    "Move",
    types=[_block_type],
    params_space=Box(low=np.array([0.0], dtype=np.float32),
                     high=np.array([1.0], dtype=np.float32)),
    policy=lambda _s, _m, _o, _p: Action(np.zeros(1, dtype=np.float32)),
    initiable=lambda _s, _m, _o, _p: True,
    terminal=lambda _s, _m, _o, _p: False,
)


def _state(x: float) -> State:
    return State({_block: np.array([x], dtype=np.float32)})


class _StubApproach:
    """The minimal approach surface make_predicate_quality_loader uses."""

    def __init__(self) -> None:
        self._types = {_block_type}
        self._kept_initial_predicates = {_Kept}
        self._learned_predicates: Set[Predicate] = set()
        self._train_tasks = [Task(_state(0.0), {GroundAtom(_Kept, [_block])})]
        self._fitted_params: Dict[str, float] = {}

    def _get_all_options(self) -> Set[ParameterizedOption]:
        return {_Move}


def _trajectory() -> LowLevelTrajectory:
    states = [_state(0.0), _state(0.4), _state(0.8), _state(1.0)]
    actions = [Action(np.zeros(1, dtype=np.float32)) for _ in states[:-1]]
    return LowLevelTrajectory(states, actions)


_PREDICATES = """\
LEARNED_PREDICATES = [
    Predicate("Hi", [block_type], lambda s, o: s.get(o[0], "x") >= 0.5),
    Predicate("Kept", [block_type], lambda s, o: True),
]
"""


def _probe(tmp_path: Any, approach: _StubApproach) -> BeliefProbe:
    loader = make_predicate_quality_loader(
        predicates_file=str(tmp_path / "predicates.py"),
        predicates_versions_dir=str(tmp_path / "predicates_versions"),
        approach=cast(Any, approach),
        trajectories=[_trajectory()],
        cycle_index_provider=lambda: 1,
    )
    ctx = ToolContext()
    ctx.probe_artifact_loaders["predicates"] = loader
    return BeliefProbe(ctx)


def test_probe_predicates_loads_scores_and_installs(tmp_path: Any) -> None:
    """The report tags the snapshot, scores the milestone, skips the kept-name
    collision, and the validated draft becomes the approach's learned set."""
    (tmp_path / "predicates.py").write_text(_PREDICATES, encoding="utf-8")
    approach = _StubApproach()
    text = _probe(tmp_path, approach).predicates()
    assert text.startswith("[cycle_001_vers_001] Predicate quality report")
    assert "Hi(block)" in text
    assert "coverage: ever-T + ever-F" in text
    assert "monotone (1↑ 0↓): 1" in text
    assert "Skipped 'Kept' (collides with a kept env predicate)" in text
    assert {p.name for p in approach._learned_predicates} == {"Hi"}


def test_probe_predicates_reports_a_missing_file(tmp_path: Any) -> None:
    """A missing predicates.py is an actionable error, not a crash."""
    approach = _StubApproach()
    text = _probe(tmp_path, approach).predicates()
    assert "LEARNED_PREDICATES = [...]" in text
    assert not approach._learned_predicates


def test_probe_predicates_unavailable_without_a_loader() -> None:
    """Outside a predicate-invention session the probe has no predicates.py
    surface and says so."""
    probe = BeliefProbe(ToolContext())
    try:
        probe.predicates()
    except RuntimeError as e:
        assert "sim.predicates is unavailable" in str(e)
    else:
        raise AssertionError("sim.predicates() must raise without a loader")
