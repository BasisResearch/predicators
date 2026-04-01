"""Tests for AgentBilevelApproach -- parsing and refinement logic."""
# pylint: disable=protected-access,import-outside-toplevel
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from gym.spaces import Box

from predicators import utils
from predicators.approaches.agent_bilevel_approach import \
    AgentBilevelApproach, _SketchStep
from predicators.structs import Action, GroundAtom, Object, \
    ParameterizedOption, Predicate, State, Task, Type

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_block_type = Type("block", ["x", "y", "held"])
_robot_type = Type("robot", ["x", "y"])

_block0 = Object("block0", _block_type)
_block1 = Object("block1", _block_type)
_robot = Object("robot0", _robot_type)

_Holding = Predicate("Holding", [_block_type],
                     lambda s, o: s.get(o[0], "held") > 0.5)
_On = Predicate("On", [_block_type, _block_type],
                lambda s, o: abs(s.get(o[0], "x") - s.get(o[1], "x")) < 0.1)
_HandEmpty = Predicate("HandEmpty", [_robot_type], lambda s, o: True)

_ALL_PREDICATES = {_Holding, _On, _HandEmpty}
_ALL_OBJECTS = [_block0, _block1, _robot]


def _noop_policy(_s, _m, _o, _p):
    return Action(np.zeros(1, dtype=np.float32))


def _always_true(_s, _m, _o, _p):
    return True


def _always_false(_s, _m, _o, _p):
    return False


_Pick = ParameterizedOption(
    "Pick",
    types=[_block_type],
    params_space=Box(low=np.array([0.0], dtype=np.float32),
                     high=np.array([1.0], dtype=np.float32)),
    policy=_noop_policy,
    initiable=_always_true,
    terminal=_always_false,
)

_Place = ParameterizedOption(
    "Place",
    types=[_block_type, _block_type],
    params_space=Box(low=np.array([0.0, 0.0], dtype=np.float32),
                     high=np.array([1.0, 1.0], dtype=np.float32)),
    policy=_noop_policy,
    initiable=_always_true,
    terminal=_always_false,
)

_Wait = ParameterizedOption(
    "Wait",
    types=[_robot_type],
    params_space=Box(low=np.array([], dtype=np.float32),
                     high=np.array([], dtype=np.float32)),
    policy=_noop_policy,
    initiable=_always_true,
    terminal=_always_false,
)

_ALL_OPTIONS = {_Pick, _Place, _Wait}


def _make_state(overrides=None):
    """Create a simple state with default feature values."""
    data = {
        _block0: np.array([0.1, 0.2, 0.0], dtype=np.float32),
        _block1: np.array([0.5, 0.6, 0.0], dtype=np.float32),
        _robot: np.array([0.0, 0.0], dtype=np.float32),
    }
    if overrides:
        for obj, vals in overrides.items():
            data[obj] = np.array(vals, dtype=np.float32)
    return State(data)


def _make_approach():
    """Create an AgentBilevelApproach with mock config and option model."""
    state = _make_state()
    goal = {GroundAtom(_On, [_block0, _block1])}
    task = Task(state, goal)

    utils.reset_config({
        "env": "cover",
        "approach": "agent_bilevel",
        "num_train_tasks": 1,
        "num_test_tasks": 1,
        "option_model_name": "oracle",
        "seed": 42,
        "agent_bilevel_max_samples_per_step": 10,
        "agent_bilevel_max_retries": 0,
        "agent_bilevel_check_subgoals": True,
    })

    mock_option_model = MagicMock()
    approach = AgentBilevelApproach(
        initial_predicates=_ALL_PREDICATES,
        initial_options=_ALL_OPTIONS,
        types={_block_type, _robot_type},
        action_space=Box(low=-1, high=1, shape=(1, )),
        train_tasks=[task],
        option_model=mock_option_model,
    )
    return approach, mock_option_model, task


# ---------------------------------------------------------------------------
# Tests: _parse_subgoal_annotations
# ---------------------------------------------------------------------------


class TestParseSubgoalAnnotations:
    """Tests for plan text subgoal parsing."""

    def test_basic_subgoals(self):
        """Test basic subgoals."""
        approach, _, _ = _make_approach()
        text = ("Pick(block0:block) -> {Holding(block0:block)}\n"
                "Place(block0:block, block1:block) -> "
                "{On(block0:block, block1:block)}\n")
        result = approach._parse_subgoal_annotations(text, _ALL_PREDICATES,
                                                     _ALL_OBJECTS)

        assert len(result) == 2
        # First step: Holding(block0)
        assert result[0] is not None
        pos, neg = result[0]
        assert GroundAtom(_Holding, [_block0]) in pos
        assert len(neg) == 0
        # Second step: On(block0, block1)
        assert result[1] is not None
        pos2, neg2 = result[1]
        assert GroundAtom(_On, [_block0, _block1]) in pos2
        assert len(neg2) == 0

    def test_no_subgoals(self):
        """Test no subgoals."""
        approach, _, _ = _make_approach()
        text = ("Pick(block0:block)\n"
                "Place(block0:block, block1:block)\n")
        result = approach._parse_subgoal_annotations(text, _ALL_PREDICATES,
                                                     _ALL_OBJECTS)

        assert len(result) == 2
        assert result[0] is None
        assert result[1] is None

    def test_mixed_subgoals(self):
        """Some lines have subgoals, some don't."""
        approach, _, _ = _make_approach()
        text = ("Pick(block0:block) -> {Holding(block0:block)}\n"
                "Wait(robot0:robot)\n"
                "Place(block0:block, block1:block) -> "
                "{On(block0:block, block1:block)}\n")
        result = approach._parse_subgoal_annotations(text, _ALL_PREDICATES,
                                                     _ALL_OBJECTS)

        assert len(result) == 3
        assert result[0] is not None
        assert result[1] is None  # Wait has no subgoal
        assert result[2] is not None

    def test_multiple_atoms_in_subgoal(self):
        """Test multiple atoms in subgoal."""
        approach, _, _ = _make_approach()
        text = (
            "Place(block0:block, block1:block) "
            "-> {On(block0:block, block1:block), HandEmpty(robot0:robot)}\n")
        result = approach._parse_subgoal_annotations(text, _ALL_PREDICATES,
                                                     _ALL_OBJECTS)

        assert len(result) == 1
        assert result[0] is not None
        pos, neg = result[0]
        assert len(pos) == 2
        assert len(neg) == 0
        assert GroundAtom(_On, [_block0, _block1]) in pos
        assert GroundAtom(_HandEmpty, [_robot]) in pos

    def test_unknown_predicate_skipped(self):
        """Test unknown predicate skipped."""
        approach, _, _ = _make_approach()
        text = "Pick(block0:block) -> {FakePred(block0:block)}\n"
        result = approach._parse_subgoal_annotations(text, _ALL_PREDICATES,
                                                     _ALL_OBJECTS)

        assert len(result) == 1
        assert result[0] is None  # FakePred unrecognized, no valid atoms

    def test_unknown_object_skipped(self):
        """Test unknown object skipped."""
        approach, _, _ = _make_approach()
        text = "Pick(block0:block) -> {Holding(block99:block)}\n"
        result = approach._parse_subgoal_annotations(text, _ALL_PREDICATES,
                                                     _ALL_OBJECTS)

        assert len(result) == 1
        assert result[0] is None  # block99 doesn't exist

    def test_arity_mismatch_skipped(self):
        """Test arity mismatch skipped."""
        approach, _, _ = _make_approach()
        # Holding expects 1 arg, giving 2
        text = "Pick(block0:block) -> {Holding(block0:block, block1:block)}\n"
        result = approach._parse_subgoal_annotations(text, _ALL_PREDICATES,
                                                     _ALL_OBJECTS)

        assert len(result) == 1
        assert result[0] is None

    def test_typed_object_refs_in_subgoals(self):
        """Agent outputs obj:type in subgoal atoms — should still parse."""
        approach, _, _ = _make_approach()
        text = ("Pick(block0:block) -> {Holding(block0:block)}\n"
                "Place(block0:block, block1:block) "
                "-> {On(block0:block, block1:block)}\n")
        result = approach._parse_subgoal_annotations(text, _ALL_PREDICATES,
                                                     _ALL_OBJECTS)

        assert len(result) == 2
        assert result[0] is not None
        pos, _ = result[0]
        assert GroundAtom(_Holding, [_block0]) in pos
        assert result[1] is not None
        pos2, _ = result[1]
        assert GroundAtom(_On, [_block0, _block1]) in pos2

    def test_preamble_ignored(self):
        """Non-option lines should be ignored."""
        approach, _, _ = _make_approach()
        text = ("Here is my analysis:\n"
                "I think we should pick block0 first.\n"
                "\n"
                "Pick(block0:block) -> {Holding(block0:block)}\n")
        result = approach._parse_subgoal_annotations(text, _ALL_PREDICATES,
                                                     _ALL_OBJECTS)

        assert len(result) == 1
        assert result[0] is not None

    def test_whitespace_in_atoms(self):
        """Spaces around commas in atom arguments."""
        approach, _, _ = _make_approach()
        text = ("Place(block0:block, block1:block) -> "
                "{ On( block0:block , block1:block ) }\n")
        result = approach._parse_subgoal_annotations(text, _ALL_PREDICATES,
                                                     _ALL_OBJECTS)

        assert len(result) == 1
        assert result[0] is not None
        pos, _ = result[0]
        assert GroundAtom(_On, [_block0, _block1]) in pos

    def test_not_atoms_in_subgoals(self):
        """Test NOT prefix for negative target atoms."""
        approach, _, _ = _make_approach()
        text = (
            "Wait(robot0:robot) -> "
            "{Holding(block0:block), NOT On(block0:block, block1:block)}\n")
        result = approach._parse_subgoal_annotations(text, _ALL_PREDICATES,
                                                     _ALL_OBJECTS)

        assert len(result) == 1
        assert result[0] is not None
        pos, neg = result[0]
        assert GroundAtom(_Holding, [_block0]) in pos
        assert GroundAtom(_On, [_block0, _block1]) in neg


# ---------------------------------------------------------------------------
# Tests: check_wait_target_atoms
# ---------------------------------------------------------------------------


class TestCheckWaitTargetAtoms:
    """Tests that Wait terminates on target atoms, not noisy changes."""

    def test_no_targets_returns_none(self):
        """No targets in memory -> returns None (fall back to any-change)."""
        opt = _Wait.ground([_robot], np.array([], dtype=np.float32))
        # No targets in memory
        state = _make_state({_block0: [0.0, 0.0, 0.0]})
        abstract_fn = lambda s: utils.abstract(s, _ALL_PREDICATES)
        result = utils.check_wait_target_atoms(opt, state, abstract_fn)
        assert result is None

    def test_positive_target_met(self):
        """Wait terminates when positive target atom holds."""
        opt = _Wait.ground([_robot], np.array([], dtype=np.float32))
        target_atom = GroundAtom(_Holding, [_block0])
        opt.memory["wait_target_atoms"] = {target_atom}

        # State where Holding(block0) is true (held > 0.5)
        state_held = _make_state({_block0: [0.0, 0.0, 1.0]})
        abstract_fn = lambda s: utils.abstract(s, _ALL_PREDICATES)
        assert utils.check_wait_target_atoms(opt, state_held, abstract_fn) \
            is True

    def test_positive_target_not_met(self):
        """Wait does NOT terminate when target atom doesn't hold yet."""
        opt = _Wait.ground([_robot], np.array([], dtype=np.float32))
        target_atom = GroundAtom(_Holding, [_block0])
        opt.memory["wait_target_atoms"] = {target_atom}

        # State where Holding(block0) is false (held <= 0.5)
        state_not_held = _make_state({_block0: [0.0, 0.0, 0.0]})
        abstract_fn = lambda s: utils.abstract(s, _ALL_PREDICATES)
        assert utils.check_wait_target_atoms(opt, state_not_held,
                                             abstract_fn) is False

    def test_noisy_atom_change_ignored_with_targets(self):
        """Wait ignores noisy atom changes when specific targets are set.

        This is the key test: if the Wait is parameterized with a target
        atom (e.g. Holding(block0)), it should NOT terminate when a
        different atom changes (e.g. On(block0, block1)).
        """
        opt = _Wait.ground([_robot], np.array([], dtype=np.float32))
        # Only waiting for Holding(block0)
        target_atom = GroundAtom(_Holding, [_block0])
        opt.memory["wait_target_atoms"] = {target_atom}

        # State where On(block0, block1) is true (noisy change) but
        # Holding(block0) is still false
        state_noisy = _make_state({
            _block0: [0.5, 0.0, 0.0],
            _block1: [0.5, 0.0, 0.0]
        })
        abstract_fn = lambda s: utils.abstract(s, _ALL_PREDICATES)
        atoms = abstract_fn(state_noisy)
        # On is true (positions are close), but Holding is false
        assert GroundAtom(_On, [_block0, _block1]) in atoms
        assert GroundAtom(_Holding, [_block0]) not in atoms

        # Wait should NOT terminate (target not met, despite On changing)
        assert utils.check_wait_target_atoms(opt, state_noisy,
                                             abstract_fn) is False

    def test_negative_target_met(self):
        """Wait terminates when negative target atom is false."""
        opt = _Wait.ground([_robot], np.array([], dtype=np.float32))
        neg_atom = GroundAtom(_On, [_block0, _block1])
        opt.memory["wait_target_neg_atoms"] = {neg_atom}

        # State where On(block0, block1) is false (positions far apart)
        state = _make_state({
            _block0: [0.0, 0.0, 0.0],
            _block1: [5.0, 0.0, 0.0]
        })
        abstract_fn = lambda s: utils.abstract(s, _ALL_PREDICATES)
        assert utils.check_wait_target_atoms(opt, state, abstract_fn) is True

    def test_negative_target_not_met(self):
        """Wait does NOT terminate when negative target atom is still true."""
        opt = _Wait.ground([_robot], np.array([], dtype=np.float32))
        neg_atom = GroundAtom(_On, [_block0, _block1])
        opt.memory["wait_target_neg_atoms"] = {neg_atom}

        # State where On(block0, block1) is true (positions close)
        state = _make_state({
            _block0: [0.5, 0.0, 0.0],
            _block1: [0.5, 0.0, 0.0]
        })
        abstract_fn = lambda s: utils.abstract(s, _ALL_PREDICATES)
        assert utils.check_wait_target_atoms(opt, state, abstract_fn) is False

    def test_mixed_positive_and_negative_targets(self):
        """Both positive and negative targets must be satisfied."""
        opt = _Wait.ground([_robot], np.array([], dtype=np.float32))
        opt.memory["wait_target_atoms"] = {GroundAtom(_Holding, [_block0])}
        opt.memory["wait_target_neg_atoms"] = {
            GroundAtom(_On, [_block0, _block1])
        }

        abstract_fn = lambda s: utils.abstract(s, _ALL_PREDICATES)

        # Only positive met (Holding true, On still true)
        state1 = _make_state({
            _block0: [0.5, 0.0, 1.0],
            _block1: [0.5, 0.0, 0.0]
        })
        assert utils.check_wait_target_atoms(opt, state1, abstract_fn) is False

        # Only negative met (On false, Holding false)
        state2 = _make_state({
            _block0: [0.0, 0.0, 0.0],
            _block1: [5.0, 0.0, 0.0]
        })
        assert utils.check_wait_target_atoms(opt, state2, abstract_fn) is False

        # Both met (Holding true, On false)
        state3 = _make_state({
            _block0: [0.0, 0.0, 1.0],
            _block1: [5.0, 0.0, 0.0]
        })
        assert utils.check_wait_target_atoms(opt, state3, abstract_fn) is True


# ---------------------------------------------------------------------------
# Tests: parse_wait_target_annotations and strip_wait_annotations
# ---------------------------------------------------------------------------


class TestWaitTargetParsing:
    """Tests for parse_wait_target_annotations and strip_wait_annotations."""

    def test_parse_positive_target(self):
        """Parse a positive target atom."""
        line = "Wait(robot0:robot) -> {Holding(block0:block)}"
        pos, neg = utils.parse_wait_target_annotations(line, _ALL_PREDICATES,
                                                       _ALL_OBJECTS)
        assert GroundAtom(_Holding, [_block0]) in pos
        assert len(neg) == 0

    def test_parse_negative_target(self):
        """Parse a NOT-prefixed target atom."""
        line = "Wait(robot0:robot) -> {NOT On(block0:block, block1:block)}"
        pos, neg = utils.parse_wait_target_annotations(line, _ALL_PREDICATES,
                                                       _ALL_OBJECTS)
        assert len(pos) == 0
        assert GroundAtom(_On, [_block0, _block1]) in neg

    def test_parse_mixed_targets(self):
        """Parse both positive and negative target atoms."""
        line = ("Wait(robot0:robot) -> "
                "{Holding(block0:block), NOT On(block0:block, block1:block)}")
        pos, neg = utils.parse_wait_target_annotations(line, _ALL_PREDICATES,
                                                       _ALL_OBJECTS)
        assert GroundAtom(_Holding, [_block0]) in pos
        assert GroundAtom(_On, [_block0, _block1]) in neg

    def test_parse_no_annotation(self):
        """Line without -> returns empty sets."""
        line = "Wait(robot0:robot)[]"
        pos, neg = utils.parse_wait_target_annotations(line, _ALL_PREDICATES,
                                                       _ALL_OBJECTS)
        assert len(pos) == 0
        assert len(neg) == 0

    def test_strip_annotations(self):
        """strip_wait_annotations removes -> {...} suffixes."""
        text = ("Pick(block0:block)[0.5]\n"
                "Wait(robot0:robot)[] -> {Holding(block0:block)}\n"
                "Place(block0:block, block1:block)[0.1, 0.2]\n")
        stripped = utils.strip_wait_annotations(text)
        assert "-> {" not in stripped
        assert "Pick(block0:block)[0.5]" in stripped
        assert "Wait(robot0:robot)[]" in stripped
        assert "Place(block0:block, block1:block)[0.1, 0.2]" in stripped


# ---------------------------------------------------------------------------
# Tests: _refine_sketch
# ---------------------------------------------------------------------------


class TestRefineSketch:
    """Tests for backtracking refinement search."""

    def test_empty_sketch(self):
        """Test empty sketch."""
        approach, _, task = _make_approach()
        plan, success = approach._refine_sketch(task, [], timeout=5.0)
        assert plan == []
        assert success is False

    def test_single_step_no_params(self):
        """Option with empty params_space — should succeed in 1 try."""
        approach, mock_om, task = _make_approach()

        # Option model: Wait always succeeds, goal holds after
        goal_state = _make_state({_block0: [0.5, 0.6, 0.0]})
        mock_om.get_next_state_and_num_actions.return_value = (goal_state, 5)

        sketch = [
            _SketchStep(option=_Wait, objects=[_robot], subgoal_atoms=None)
        ]
        plan, success = approach._refine_sketch(task, sketch, timeout=5.0)

        assert success is True
        assert len(plan) == 1
        assert plan[0].name == "Wait"

    def test_single_step_with_params_success(self):
        """Option with params — should find working params via sampling."""
        approach, mock_om, task = _make_approach()

        goal_state = _make_state({_block0: [0.5, 0.6, 0.0]})
        mock_om.get_next_state_and_num_actions.return_value = (goal_state, 3)

        sketch = [
            _SketchStep(option=_Pick, objects=[_block0], subgoal_atoms=None)
        ]
        plan, success = approach._refine_sketch(task, sketch, timeout=5.0)

        assert success is True
        assert len(plan) == 1

    def test_subgoal_check_pass(self):
        """Subgoal atoms hold after execution."""
        approach, mock_om, task = _make_approach()

        # After Pick, Holding(block0) should hold — set held=1
        held_state = _make_state({_block0: [0.1, 0.2, 1.0]})
        # After Place, On(block0, block1) — set x close
        goal_state = _make_state({_block0: [0.5, 0.6, 0.0]})

        mock_om.get_next_state_and_num_actions.side_effect = [
            (held_state, 3),
            (goal_state, 3),
        ]

        sketch = [
            _SketchStep(option=_Pick,
                        objects=[_block0],
                        subgoal_atoms={GroundAtom(_Holding, [_block0])}),
            _SketchStep(option=_Place,
                        objects=[_block0, _block1],
                        subgoal_atoms={GroundAtom(_On, [_block0, _block1])}),
        ]
        plan, success = approach._refine_sketch(task, sketch, timeout=5.0)

        assert success is True
        assert len(plan) == 2

    def test_subgoal_check_fail_triggers_resample(self):
        """Subgoal atoms don't hold — should resample params."""
        approach, mock_om, task = _make_approach()

        # Holding never holds (held=0) — subgoal always fails
        bad_state = _make_state({_block0: [0.1, 0.2, 0.0]})
        mock_om.get_next_state_and_num_actions.return_value = (bad_state, 3)

        sketch = [
            _SketchStep(option=_Pick,
                        objects=[_block0],
                        subgoal_atoms={GroundAtom(_Holding, [_block0])}),
        ]
        _plan, success = approach._refine_sketch(task, sketch, timeout=5.0)

        # Should exhaust all samples and fail
        assert success is False
        # Option model called max_samples times (10)
        assert mock_om.get_next_state_and_num_actions.call_count == 10

    def test_backtracking_across_steps(self):
        """Step 2 fails, causing step 1 to be re-sampled."""
        approach, mock_om, task = _make_approach()
        utils.reset_config({
            "env": "cover",
            "approach": "agent_bilevel",
            "num_train_tasks": 1,
            "num_test_tasks": 1,
            "seed": 42,
            "agent_bilevel_max_samples_per_step": 3,
            "agent_bilevel_max_retries": 0,
            "agent_bilevel_check_subgoals": False,
        })

        call_count = 0
        goal_state = _make_state({_block0: [0.5, 0.6, 0.0]})
        noop_state = _make_state()

        def side_effect(_state, option):
            nonlocal call_count
            call_count += 1
            if option.name == "Pick":
                return (noop_state, 3)  # Pick always succeeds
            # Place: succeed only on the last attempt
            if call_count >= 8:
                return (goal_state, 3)
            return (noop_state, 0)  # fail (noop)

        mock_om.get_next_state_and_num_actions.side_effect = side_effect

        sketch = [
            _SketchStep(option=_Pick, objects=[_block0], subgoal_atoms=None),
            _SketchStep(option=_Place,
                        objects=[_block0, _block1],
                        subgoal_atoms=None),
        ]
        plan, success = approach._refine_sketch(task, sketch, timeout=10.0)

        # Should have backtracked and eventually succeeded
        assert success is True
        assert len(plan) == 2
        assert call_count >= 4  # at least one backtrack cycle

    def test_not_initiable_triggers_resample(self):
        """Option not initiable in current state — resample."""
        approach, mock_om, task = _make_approach()
        utils.reset_config({
            "env": "cover",
            "approach": "agent_bilevel",
            "num_train_tasks": 1,
            "num_test_tasks": 1,
            "seed": 42,
            "agent_bilevel_max_samples_per_step": 3,
        })

        # Create an option that is never initiable
        not_initiable = ParameterizedOption(
            "Pick",
            types=[_block_type],
            params_space=Box(low=np.array([0.0], dtype=np.float32),
                             high=np.array([1.0], dtype=np.float32)),
            policy=_noop_policy,
            initiable=_always_false,
            terminal=_always_false,
        )

        sketch = [
            _SketchStep(option=not_initiable,
                        objects=[_block0],
                        subgoal_atoms=None)
        ]
        _plan, success = approach._refine_sketch(task, sketch, timeout=5.0)

        assert success is False
        # Option model never called since initiable is always False
        mock_om.get_next_state_and_num_actions.assert_not_called()

    def test_goal_check_on_final_step(self):
        """Final step must satisfy the task goal even without subgoals."""
        approach, mock_om, task = _make_approach()
        utils.reset_config({
            "env": "cover",
            "approach": "agent_bilevel",
            "num_train_tasks": 1,
            "num_test_tasks": 1,
            "seed": 42,
            "agent_bilevel_max_samples_per_step": 5,
            "agent_bilevel_check_subgoals": False,
        })

        # State that doesn't satisfy goal On(block0, block1)
        bad_state = _make_state({_block0: [0.9, 0.2, 0.0]})
        mock_om.get_next_state_and_num_actions.return_value = (bad_state, 3)

        sketch = [
            _SketchStep(option=_Pick, objects=[_block0], subgoal_atoms=None)
        ]
        _plan, success = approach._refine_sketch(task, sketch, timeout=5.0)

        # Goal never holds → exhausts samples
        assert success is False


# ---------------------------------------------------------------------------
# Tests: _query_agent_for_plan_sketch (with mocked agent)
# ---------------------------------------------------------------------------


class TestQueryAgentForPlanSketch:
    """Tests for end-to-end sketch extraction from mock agent responses."""

    def _mock_responses(self, plan_text):
        """Build mock agent response list containing plan_text."""
        return [
            {
                "type": "assistant",
                "content": [{
                    "type": "text",
                    "text": plan_text
                }],
            },
        ]

    def test_basic_sketch_extraction(self):
        """Test basic sketch extraction."""
        approach, _, task = _make_approach()

        plan_text = ("Pick(block0:block) -> {Holding(block0:block)}\n"
                     "Place(block0:block, block1:block) -> "
                     "{On(block0:block, block1:block)}\n")

        with patch.object(approach,
                          '_query_agent_sync',
                          return_value=self._mock_responses(plan_text)):
            sketch = approach._query_agent_for_plan_sketch(task)

        assert len(sketch) == 2
        assert sketch[0].option.name == "Pick"
        assert list(sketch[0].objects) == [_block0]
        assert sketch[0].subgoal_atoms is not None
        assert GroundAtom(_Holding, [_block0]) in sketch[0].subgoal_atoms

        assert sketch[1].option.name == "Place"
        assert list(sketch[1].objects) == [_block0, _block1]
        assert sketch[1].subgoal_atoms is not None

    def test_sketch_without_subgoals(self):
        """Test sketch without subgoals."""
        approach, _, task = _make_approach()

        plan_text = ("Pick(block0:block)\n"
                     "Place(block0:block, block1:block)\n")

        with patch.object(approach,
                          '_query_agent_sync',
                          return_value=self._mock_responses(plan_text)):
            sketch = approach._query_agent_for_plan_sketch(task)

        assert len(sketch) == 2
        assert sketch[0].subgoal_atoms is None
        assert sketch[1].subgoal_atoms is None

    def test_sketch_with_code_fences(self):
        """Test sketch with code fences."""
        approach, _, task = _make_approach()

        plan_text = ("Here is the plan:\n"
                     "```\n"
                     "Pick(block0:block) -> {Holding(block0:block)}\n"
                     "Place(block0:block, block1:block)\n"
                     "```\n")

        with patch.object(approach,
                          '_query_agent_sync',
                          return_value=self._mock_responses(plan_text)):
            sketch = approach._query_agent_for_plan_sketch(task)

        assert len(sketch) == 2

    def test_sketch_with_preamble(self):
        """Agent includes analysis text before the plan."""
        approach, _, task = _make_approach()

        plan_text = (
            "After inspecting the environment, I found block0 and block1.\n"
            "The goal is to place block0 on block1.\n"
            "\n"
            "Pick(block0:block)\n"
            "Place(block0:block, block1:block)\n")

        with patch.object(approach,
                          '_query_agent_sync',
                          return_value=self._mock_responses(plan_text)):
            sketch = approach._query_agent_for_plan_sketch(task)

        assert len(sketch) == 2

    def test_sketch_with_wait(self):
        """Test sketch with wait."""
        approach, _, task = _make_approach()

        plan_text = ("Pick(block0:block) -> {Holding(block0:block)}\n"
                     "Wait(robot0:robot)\n"
                     "Place(block0:block, block1:block) -> "
                     "{On(block0:block, block1:block)}\n")

        with patch.object(approach,
                          '_query_agent_sync',
                          return_value=self._mock_responses(plan_text)):
            sketch = approach._query_agent_for_plan_sketch(task)

        assert len(sketch) == 3
        assert sketch[0].option.name == "Pick"
        assert sketch[1].option.name == "Wait"
        assert sketch[1].subgoal_atoms is None
        assert sketch[2].option.name == "Place"

    def test_empty_response_raises(self):
        """Agent returns no text → ApproachFailure."""
        from predicators.approaches import ApproachFailure
        approach, _, task = _make_approach()

        with patch.object(approach,
                          '_query_agent_sync',
                          return_value=[{
                              "type": "result",
                              "content": []
                          }]):
            with pytest.raises(ApproachFailure, match="empty plan text"):
                approach._query_agent_for_plan_sketch(task)

    def test_no_valid_options_raises(self):
        """Agent returns text with no valid option names → ApproachFailure."""
        from predicators.approaches import ApproachFailure
        approach, _, task = _make_approach()

        plan_text = "I don't know what to do.\nSorry!\n"

        with patch.object(approach,
                          '_query_agent_sync',
                          return_value=self._mock_responses(plan_text)):
            with pytest.raises(ApproachFailure, match="Parsed empty"):
                approach._query_agent_for_plan_sketch(task)


# ---------------------------------------------------------------------------
# Tests: _sample_params
# ---------------------------------------------------------------------------


class TestSampleParams:
    """TestSampleParams class."""

    def test_empty_params_space(self):
        """Test empty params space."""
        approach, _, _ = _make_approach()
        rng = np.random.default_rng(0)
        params = approach._sample_params(_Wait, _make_state(), rng)
        assert params.shape == (0, )
        assert params.dtype == np.float32

    def test_params_within_bounds(self):
        """Test params within bounds."""
        approach, _, _ = _make_approach()
        rng = np.random.default_rng(0)
        for _ in range(100):
            params = approach._sample_params(_Place, _make_state(), rng)
            assert params.shape == (2, )
            assert np.all(params >= 0.0)
            assert np.all(params <= 1.0)
            assert params.dtype == np.float32


# ---------------------------------------------------------------------------
# Tests: class metadata
# ---------------------------------------------------------------------------


def test_get_name():
    """Test get name."""
    assert AgentBilevelApproach.get_name() == "agent_bilevel"
