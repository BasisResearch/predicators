"""Tests for AgentBilevelExplorer."""
# pylint: disable=protected-access

from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from gym.spaces import Box

from predicators import utils
from predicators.agent_sdk.sketch_types import SketchStep
from predicators.agent_sdk.tools import ToolContext
from predicators.explorers import create_explorer
from predicators.explorers.agent_bilevel_explorer import AgentBilevelExplorer
from predicators.explorers.base_explorer import BaseExplorer
from predicators.structs import Action, GroundAtom, Object, \
    ParameterizedOption, Predicate, State, Task, Type

# ---------------------------------------------------------------------------
# Fixtures (parallel the bilevel approach tests)
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
_ALL_TYPES = {_block_type, _robot_type}


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
    data = {
        _block0: np.array([0.1, 0.2, 0.0], dtype=np.float32),
        _block1: np.array([0.5, 0.6, 0.0], dtype=np.float32),
        _robot: np.array([0.0, 0.0], dtype=np.float32),
    }
    if overrides:
        for obj, vals in overrides.items():
            data[obj] = np.array(vals, dtype=np.float32)
    return State(data)


def _make_task():
    state = _make_state()
    goal = {GroundAtom(_On, [_block0, _block1])}
    return Task(state, goal)


def _assistant_response(text: str):
    return [{
        "type": "assistant",
        "content": [{
            "type": "text",
            "text": text
        }],
    }]


def _make_explorer(option_model, query_impl):
    """Build an AgentBilevelExplorer with stubbed session + tool_context."""
    tool_context = ToolContext(
        types=_ALL_TYPES,
        predicates=_ALL_PREDICATES,
        options=_ALL_OPTIONS,
        train_tasks=[_make_task()],
        option_model=option_model,
    )
    agent_session = MagicMock()
    agent_session.query = query_impl
    agent_session.tool_names = None
    explorer = AgentBilevelExplorer(
        predicates=_ALL_PREDICATES,
        options=_ALL_OPTIONS,
        types=_ALL_TYPES,
        action_space=Box(low=-1, high=1, shape=(1, )),
        train_tasks=[_make_task()],
        max_steps_before_termination=50,
        tool_context=tool_context,
        agent_session=agent_session,
    )
    return explorer, tool_context


def _reset_config(**overrides):
    base = {
        "env": "cover",
        "approach": "agent_bilevel",
        "num_train_tasks": 1,
        "num_test_tasks": 1,
        "seed": 42,
        "agent_bilevel_max_samples_per_step": 5,
        "agent_bilevel_check_subgoals": True,
        "agent_bilevel_log_state": False,
        "agent_explorer_fallback_to_random": True,
        "agent_sdk_max_trajectories_in_context": 5,
    }
    base.update(overrides)
    utils.reset_config(base)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_factory_registration():
    """AgentBilevelExplorer is reachable through create_explorer."""
    _reset_config()
    tool_context = ToolContext(
        types=_ALL_TYPES,
        predicates=_ALL_PREDICATES,
        options=_ALL_OPTIONS,
        train_tasks=[_make_task()],
        option_model=MagicMock(),
    )
    agent_session = MagicMock()
    explorer = create_explorer(
        "agent_bilevel",
        _ALL_PREDICATES,
        _ALL_OPTIONS,
        _ALL_TYPES,
        Box(low=-1, high=1, shape=(1, )),
        [_make_task()],
        tool_context=tool_context,
        agent_session=agent_session,
    )
    assert isinstance(explorer, BaseExplorer)
    assert isinstance(explorer, AgentBilevelExplorer)


def test_happy_path_returns_policy_and_stashes_subgoals():
    """Canned sketch → refined plan → policy and stashed subgoals."""
    _reset_config()

    goal_state = _make_state({_block0: [0.5, 0.6, 0.0]})
    option_model = MagicMock()
    option_model.get_next_state_and_num_actions.return_value = (goal_state, 3)

    plan_text = ("Pick(block0:block)\n"
                 "Place(block0:block, block1:block) -> "
                 "{On(block0:block, block1:block)}\n")
    query = AsyncMock(return_value=_assistant_response(plan_text))

    explorer, tool_context = _make_explorer(option_model, query)
    policy, term_fn = explorer._get_exploration_strategy(0, timeout=5)

    assert callable(policy)
    assert term_fn(_make_state()) is False
    assert tool_context.last_sketch_subgoals is not None
    assert len(tool_context.last_sketch_subgoals) == 2
    # Second step's positive subgoal should be {On(block0, block1)}.
    pos2, _neg2 = tool_context.last_sketch_subgoals[1]
    assert pos2 == {GroundAtom(_On, [_block0, _block1])}
    assert tool_context.last_sketch_options == [
        ("Pick", ["block0"]),
        ("Place", ["block0", "block1"]),
    ]
    assert query.await_count == 1


def test_wait_memory_injection_on_grounding():
    """A Wait step's annotated subgoal rides on the grounded option as
    ``wait_target_atoms`` so WaitOption terminates on the intended atoms."""
    _reset_config()
    explorer, _ = _make_explorer(MagicMock(), None)
    step = SketchStep(option=_Wait,
                      objects=[_robot],
                      subgoal_atoms={GroundAtom(_On, [_block0, _block1])})
    plan = explorer._ground_sketch_verbatim([step])
    assert len(plan) == 1 and plan[0].name == "Wait"
    assert plan[0].memory["wait_target_atoms"] == {
        GroundAtom(_On, [_block0, _block1])
    }


def test_sketch_executes_verbatim_without_belief_refinement():
    """The agent's explicit parameters execute exactly as written: the belief
    model is never rolled, the verdict is not-certified, and the cycle record
    shows the executed values."""
    _reset_config(agent_bilevel_use_llm_initial_params=True)
    option_model = MagicMock()
    plan_text = ("```\nPick(block0:block)[0.42] -> {Holding(block0:block)}\n"
                 "Place(block0:block, block1:block)[0.11, 0.22] -> "
                 "{On(block0:block, block1:block)}\n```")
    query = AsyncMock(return_value=_assistant_response(plan_text))
    explorer, tool_context = _make_explorer(option_model, query)
    policy, term_fn = explorer._get_exploration_strategy(0, timeout=5)
    assert callable(policy) and term_fn(_make_state()) is False
    assert not option_model.get_next_state_and_num_actions.called
    assert tool_context.last_mental_model_solved is False
    record = tool_context.cycle_scheduled_plans[-1]
    assert "Pick(block0)[0.4200]" in record
    assert "Place(block0, block1)[0.1100, 0.2200]" in record
    assert "-> {On(block0:block, block1:block)}" in record
    assert "without belief-model certification" in record
    assert tool_context.last_sketch_options == [
        ("Pick", ["block0"]),
        ("Place", ["block0", "block1"]),
    ]


def test_missing_params_get_one_draw_from_the_box():
    """A step the agent left without parameters is grounded on one uniform.

    draw from the option's box - no search, and no crash.
    """
    _reset_config(agent_bilevel_use_llm_initial_params=True)
    explorer, _ = _make_explorer(MagicMock(), None)
    steps = [
        SketchStep(option=_Pick, objects=[_block0], subgoal_atoms=None),
        SketchStep(option=_Place,
                   objects=[_block0, _block1],
                   subgoal_atoms=None,
                   initial_params=np.array([0.5], dtype=np.float32)),
    ]
    plan = explorer._ground_sketch_verbatim(steps)
    assert plan[0].params.shape == (1, ) and 0.0 <= plan[0].params[0] <= 1.0
    # Wrong arity counts as missing.
    assert plan[1].params.shape == (2, )


def _make_captured(pick_params, place_params):
    """Build the (solved_plan, solved_sketch) a tool capture would stash."""
    grounded_plan = [
        _Pick.ground([_block0], np.array(pick_params, dtype=np.float32)),
        _Place.ground([_block0, _block1],
                      np.array(place_params, dtype=np.float32)),
    ]
    captured_sketch = [
        SketchStep(option=_Pick, objects=[_block0], subgoal_atoms=None),
        SketchStep(option=_Place,
                   objects=[_block0, _block1],
                   subgoal_atoms={GroundAtom(_On, [_block0, _block1])}),
    ]
    return grounded_plan, captured_sketch


def test_recovers_captured_plan_when_final_text_unparseable():
    """Agent validates a plan via evaluate_option_plan but ends in prose:

    explorer recovers the captured plan instead of falling back to
    random and executes it at the captured continuous params.
    """
    _reset_config()
    option_model = MagicMock()
    pick_params, place_params = [0.42], [0.11, 0.22]
    grounded_plan, captured_sketch = _make_captured(pick_params, place_params)
    explorer, tool_context = _make_explorer(option_model, None)

    async def query_impl(_msg, **_kw):
        # Simulate the agent capturing a validated plan via the tool during
        # the query (set AFTER the explorer's entry-time capture clear), then
        # ending with prose that does NOT parse into a sketch.
        tool_context.solved_plan = grounded_plan
        tool_context.solved_sketch = captured_sketch
        return _assistant_response("Solved it. Plan: 1. pick 2. place. Done.")

    explorer._agent_session.query = query_impl
    policy, term_fn = explorer._get_exploration_strategy(0, timeout=5)
    # Recovered (not random fallback): subgoals/options come from the capture.
    assert callable(policy)
    assert term_fn(_make_state()) is False
    assert tool_context.last_sketch_options == [
        ("Pick", ["block0"]),
        ("Place", ["block0", "block1"]),
    ]
    # The capture was consumed (cleared) so it can't leak into a later solve.
    assert tool_context.solved_plan is None
    assert tool_context.solved_sketch is None
    # The captured params execute verbatim; the belief is not re-rolled.
    assert not option_model.get_next_state_and_num_actions.called
    record = tool_context.cycle_scheduled_plans[-1]
    assert "Pick(block0)[0.4200]" in record
    assert "Place(block0, block1)[0.1100, 0.2200]" in record


def test_fallback_when_query_fails_and_flag_on():
    """Agent raises → random options fallback when flag enabled."""
    _reset_config(agent_explorer_fallback_to_random=True)

    option_model = MagicMock()

    async def failing_query(_msg):
        raise RuntimeError("boom")

    explorer, _ = _make_explorer(option_model, failing_query)
    policy, term_fn = explorer._get_exploration_strategy(0, timeout=5)
    assert callable(policy)
    assert term_fn(_make_state()) is False


def test_fallback_disabled_raises():
    """Agent raises → RequestActPolicyFailure when fallback flag off."""
    _reset_config(agent_explorer_fallback_to_random=False)

    option_model = MagicMock()

    async def failing_query(_msg):
        raise RuntimeError("boom")

    explorer, _ = _make_explorer(option_model, failing_query)
    with pytest.raises(utils.RequestActPolicyFailure):
        explorer._get_exploration_strategy(0, timeout=5)


def test_experiment_guidance_gated_by_info_seeking():
    """Experiment guidance appears iff info-seeking is on."""
    _reset_config(agent_explorer_info_seeking=True)
    explorer, _ = _make_explorer(MagicMock(), MagicMock())
    guidance = explorer._build_experiment_guidance()  # pylint: disable=protected-access
    assert "sim.suggest_probes" in guidance
    # Off => section absent entirely.
    _reset_config(agent_explorer_info_seeking=False)
    assert explorer._build_experiment_guidance() == ""  # pylint: disable=protected-access


def test_experiment_guidance_injects_open_questions_ledger(tmp_path):
    """The learn phase's open_questions.md ledger reaches the explore query
    verbatim, independent of the info-seeking flag, and an oversized ledger
    keeps its head (the ranking's top)."""
    _reset_config(agent_explorer_info_seeking=False)
    explorer, tool_context = _make_explorer(MagicMock(), MagicMock())
    # No sandbox / no file => no section (and no crash).
    assert explorer._build_experiment_guidance() == ""  # pylint: disable=protected-access
    tool_context.sandbox_dir = str(tmp_path)
    assert explorer._build_experiment_guidance() == ""  # pylint: disable=protected-access
    ledger = ("1. Bond window: place pairs at spacings 0.100/0.104/"
              "0.110/0.114 and record which bond.")
    (tmp_path / "open_questions.md").write_text(ledger, encoding="utf-8")
    guidance = explorer._build_experiment_guidance()  # pylint: disable=protected-access
    assert ledger in guidance
    assert "OPEN QUESTIONS" in guidance
    # Info-seeking on: both the ledger and the boundary-probing note.
    _reset_config(agent_explorer_info_seeking=True)
    guidance = explorer._build_experiment_guidance()  # pylint: disable=protected-access
    assert ledger in guidance
    assert "sim.suggest_probes" in guidance
    # Oversized ledger: head survives, truncation is announced.
    head = "TOP-RANKED ENTRY"
    (tmp_path / "open_questions.md").write_text(head + "x" * 10000,
                                                encoding="utf-8")
    guidance = explorer._build_experiment_guidance()  # pylint: disable=protected-access
    assert head in guidance
    assert "ledger truncated" in guidance


def _make_certified_capture(pick_params, place_params):
    """A capture as the belief's validation gate leaves it: goal reached."""
    grounded_plan, captured_sketch = _make_captured(pick_params, place_params)
    return grounded_plan, captured_sketch


def test_certified_capture_executes_verbatim_and_is_replayed():
    """A plan the session validated through the capture gate (reached_goal
    True) is executed verbatim as a solve attempt with a True mental-model
    verdict; the cycle's next request on the task replays it with no new
    query."""
    _reset_config(agent_explorer_info_seeking=True)
    option_model = MagicMock()
    option_model.get_next_state_and_num_actions.return_value = (_make_state(
        {_block0: [0.5, 0.6, 0.0]}), 3)
    pick_params, place_params = [0.42], [0.11, 0.22]
    grounded_plan, captured_sketch = _make_certified_capture(
        pick_params, place_params)
    explorer, tool_context = _make_explorer(option_model, None)
    tool_context.atom_disagreement_fn = lambda _s, _atoms: 1.0
    queries = []

    async def query_impl(msg, **_kw):
        queries.append(msg)
        tool_context.solved_plan = grounded_plan
        tool_context.solved_sketch = captured_sketch
        tool_context.solved_plan_reached_goal = True
        tool_context.solved_plan_validation_summary = "5/5 rollouts ok"
        return _assistant_response("Validated 5/5; submitting.")

    explorer._agent_session.query = query_impl
    policy, term_fn = explorer._get_exploration_strategy(0, timeout=5)
    assert callable(policy) and term_fn(_make_state()) is False
    # Verbatim: no refinement rollouts, verdict True, capture consumed.
    assert not option_model.get_next_state_and_num_actions.called
    assert tool_context.last_mental_model_solved is True
    assert tool_context.solved_plan is None
    assert tool_context.cycle_certified_plans[0] is not None
    assert "belief-certified" in tool_context.cycle_scheduled_plans[-1]
    assert tool_context.last_sketch_options == [("Pick", ["block0"]),
                                                ("Place", ["block0",
                                                           "block1"])]
    # The policy runs the captured options with their captured params.
    act = policy(_make_state())
    assert isinstance(act, Action)
    # Second request of the cycle on the same task: replay, no query.
    tool_context.last_mental_model_solved = None
    policy2, _ = explorer._get_exploration_strategy(0, timeout=5)
    assert callable(policy2)
    assert len(queries) == 1
    assert tool_context.last_mental_model_solved is True


def test_uncertified_capture_executes_its_plan_verbatim():
    """A capture whose gate verdict is not True (best-effort, flaky) is not
    certified: it executes at its captured params as an experiment, with a
    False mental-model verdict and no replay for the cycle."""
    _reset_config()
    option_model = MagicMock()
    grounded_plan, captured_sketch = _make_captured([0.42], [0.11, 0.22])
    explorer, tool_context = _make_explorer(option_model, None)

    async def query_impl(_msg, **_kw):
        tool_context.solved_plan = grounded_plan
        tool_context.solved_sketch = captured_sketch
        tool_context.solved_plan_reached_goal = False
        return _assistant_response("Best effort only, no sketch block.")

    explorer._agent_session.query = query_impl
    policy, _ = explorer._get_exploration_strategy(0, timeout=5)
    assert callable(policy)
    assert not option_model.get_next_state_and_num_actions.called
    assert 0 not in tool_context.cycle_certified_plans
    assert tool_context.last_mental_model_solved is False
