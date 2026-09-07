"""Tests for the harness fixes from the 2026-08-28 bridge failure analysis:

lineage-continuous session ids, the phase-aware step budget, latent-only
Wait targets, the missing-goal-atoms diagnostic, and the zero-gradient
bracket search in the LM fitter.
"""

from pathlib import Path
from typing import Sequence

import numpy as np
from gym.spaces import Box

from predicators import utils
from predicators.agent_sdk.session_base import max_session_log_number
from predicators.agent_sdk.tools.testing import _missing_goal_atoms
from predicators.structs import Action, GroundAtom, Object, \
    ParameterizedOption, Predicate, State, Task, Type


def test_max_session_log_number(tmp_path: Path) -> None:
    """The highest NNN across both transcript layouts; 0 when empty."""
    assert max_session_log_number(str(tmp_path)) == 0
    assert max_session_log_number(None) == 0
    for name in ("001_explore_20260827_171622.md",
                 "004_test_task0_20260827_192119.md",
                 "learn_007_20260827_185042.md", "notes.md"):
        (tmp_path / name).write_text("x")
    assert max_session_log_number(str(tmp_path)) == 7


def test_real_episode_step_budget_is_phase_aware() -> None:
    """Explore episodes are capped by the interaction-request cap too."""
    utils.reset_config({
        "horizon": 3000,
        "max_num_steps_interaction_request": 1000
    })
    assert utils.real_episode_step_budget("explore") == 1000
    assert utils.real_episode_step_budget("solve") == 3000
    assert utils.real_episode_step_budget(None) == 3000
    utils.reset_config({
        "horizon": 3000,
        "max_num_steps_interaction_request": 5000
    })
    assert utils.real_episode_step_budget("explore") == 3000


_block_type = Type("block", ["x"])


def _latent_pred() -> Predicate:

    def _classifier(s: State, objs: Sequence[Object]) -> bool:
        del objs
        return bool(s.latent is not None and s.latent.get("bonded"))

    return Predicate("Bonded", [_block_type], _classifier)


def _geometric_pred() -> Predicate:

    def _classifier(s: State, objs: Sequence[Object]) -> bool:
        return s.get(objs[0], "x") > 0.5

    return Predicate("Far", [_block_type], _classifier)


def _raising_latent_pred() -> Predicate:

    def _classifier(s: State, objs: Sequence[Object]) -> bool:
        del objs
        # None["bonded"] on a real observation: raises after the read.
        return bool(s.latent["bonded"])  # type: ignore[index]

    return Predicate("BondedRaises", [_block_type], _classifier)


def _wait_option() -> ParameterizedOption:
    return ParameterizedOption(
        "Wait", [], Box(0, 1, (0, )),
        lambda s, m, o, p: Action(np.zeros(1, dtype=np.float32)),
        lambda s, m, o, p: True, lambda s, m, o, p: True)


def test_strip_latent_wait_targets_keeps_observable_atoms() -> None:
    """Latent-reading targets are dropped, observable ones kept."""
    block = Object("b0", _block_type)
    state = State({block: np.array([0.0])}, latent={"bonded": True})
    bonded, far, raises = _latent_pred(), _geometric_pred(), \
        _raising_latent_pred()
    assert utils.predicate_reads_latent(bonded, state, [block])
    assert utils.predicate_reads_latent(raises, state, [block])
    assert not utils.predicate_reads_latent(far, state, [block])
    wait = _wait_option().ground([], np.zeros(0, dtype=np.float32))
    wait.memory["wait_target_atoms"] = {
        GroundAtom(bonded, [block]),
        GroundAtom(far, [block])
    }
    wait.memory["wait_target_neg_atoms"] = {GroundAtom(raises, [block])}
    other = _wait_option().ground([], np.zeros(0, dtype=np.float32))
    other.memory["wait_target_atoms"] = {GroundAtom(far, [block])}
    dropped = utils.strip_latent_wait_targets([wait, other], state)
    assert sorted(dropped) == [
        "step 0: Bonded(b0:block)", "step 0: NOT BondedRaises(b0:block)"
    ]
    assert wait.memory["wait_target_atoms"] == {GroundAtom(far, [block])}
    assert "wait_target_neg_atoms" not in wait.memory
    assert other.memory["wait_target_atoms"] == {GroundAtom(far, [block])}


def test_missing_goal_atoms_uses_the_goal_classifiers() -> None:
    """Atoms that hold are not reported missing even when the goal predicates
    are absent from the agent's predicate set."""
    near, far_block = Object("near", _block_type), Object("far", _block_type)
    state = State({near: np.array([0.0]), far_block: np.array([1.0])})
    far = _geometric_pred()
    goal = {GroundAtom(far, [near]), GroundAtom(far, [far_block])}
    task = Task(state, goal)
    assert _missing_goal_atoms(task, state) == {GroundAtom(far, [near])}
