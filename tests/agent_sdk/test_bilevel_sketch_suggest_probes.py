"""Tests for ``suggest_probes``: alternatives ranked by ensemble disagreement
as advice, with the agent's own parameters left untouched."""
import numpy as np
from gym.spaces import Box

from predicators import utils  # noqa: F401  pylint: disable=unused-import
from predicators.agent_sdk.sketch_refinement import suggest_probes
from predicators.agent_sdk.sketch_types import SketchStep
from predicators.structs import Action, GroundAtom, Object, \
    ParameterizedOption, Predicate, State, Task, Type

_block_type = Type("block", ["x"])
_block = Object("block0", _block_type)


def _noop_policy(_s, _m, _o, _p):
    return Action(np.zeros(1, dtype=np.float32))


def _true(_s, _m, _o, _p):
    return True


def _false(_s, _m, _o, _p):
    return False


_Move = ParameterizedOption(
    "Move",
    types=[_block_type],
    params_space=Box(low=np.array([0.0], dtype=np.float32),
                     high=np.array([1.0], dtype=np.float32)),
    policy=_noop_policy,
    initiable=_true,
    terminal=_false,
)
_Reached = Predicate("Reached", [_block_type],
                     lambda s, o: s.get(o[0], "x") > 0.5)
_PREDICATES = {_Reached}


class _Model:
    """Move sets block.x to its parameter."""
    last_execution_failure = None

    def get_next_state_and_num_actions(self, state, option):
        """Roll the option forward one step."""
        nxt = state.copy()
        nxt.set(_block, "x", float(option.params[0]))
        return nxt, 1


def _boundary_scorer(state, _atoms):
    # The ensemble splits most at x = 0.5 (the classifier boundary).
    return max(0.0, 1.0 - 4.0 * abs(state.get(_block, "x") - 0.5))


def _task():
    init = State({_block: np.array([0.0], dtype=np.float32)})
    return Task(init, {GroundAtom(_Reached, [_block])})


def _step(proposal):
    return SketchStep(option=_Move,
                      objects=[_block],
                      subgoal_atoms={GroundAtom(_Reached, [_block])},
                      initial_params=(None if proposal is None else np.array(
                          [proposal], dtype=np.float32)))


def test_alternatives_ranked_by_disagreement_and_nominal_kept():
    """Feasible draws come back best-first; the nominal is scored, not
    replaced."""
    sugg, notes = suggest_probes(_task(), [_step(0.95)],
                                 _Model(),
                                 predicates=_PREDICATES,
                                 info_scorer=_boundary_scorer,
                                 rng=np.random.default_rng(0),
                                 max_draws=40,
                                 top_k=3)
    assert not notes
    assert len(sugg) == 1
    s = sugg[0]
    np.testing.assert_allclose(s.nominal_params, [0.95], rtol=1e-6)
    assert s.nominal_feasible is True
    assert s.nominal_score is not None and s.nominal_score < 0.1
    assert 1 <= len(s.candidates) <= 3
    scores = [c[1] for c in s.candidates]
    assert scores == sorted(scores, reverse=True)
    # Every candidate still establishes the subgoal (x > 0.5) and the best
    # one sits closer to the boundary than the nominal.
    assert all(c[0][0] > 0.5 for c in s.candidates)
    assert s.candidates[0][1] > s.nominal_score
    assert s.n_feasible <= s.n_draws == 40


def test_infeasible_nominal_stops_the_analysis_with_a_note():
    """A proposal that does not establish its subgoal ends the rollout."""
    sugg, notes = suggest_probes(_task(), [_step(0.2), _step(0.9)],
                                 _Model(),
                                 predicates=_PREDICATES,
                                 info_scorer=_boundary_scorer,
                                 rng=np.random.default_rng(0),
                                 max_draws=10)
    assert len(sugg) == 1 and sugg[0].nominal_feasible is False
    assert notes and "do not establish" in notes[0]


def test_proposal_free_step_advances_on_a_feasible_draw():
    """A step without a proposal still gets suggestions and the rollout
    continues on its first feasible draw."""
    sugg, notes = suggest_probes(_task(),
                                 [_step(None), _step(0.9)],
                                 _Model(),
                                 predicates=_PREDICATES,
                                 info_scorer=_boundary_scorer,
                                 rng=np.random.default_rng(1),
                                 max_draws=30)
    assert not notes
    assert [s.step_idx for s in sugg] == [0, 1]
    assert sugg[0].nominal_params is None and sugg[0].n_feasible >= 1
