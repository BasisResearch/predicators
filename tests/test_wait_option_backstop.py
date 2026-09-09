"""The wait_option_max_steps backstop for TARGETED Waits.

A Wait annotated with target atoms used to have no step cap at all: if
the target could never become true on real states (e.g. a learned
predicate that only reads the belief latent), the Wait spun until the
episode budget killed it. In run_20260819_183137 (bridge, seed1) both
cycle-1 explore episodes burned ~250 steps this way and lost their goal-
critical final steps. The backstop terminates such a Wait at
``wait_option_max_steps``, matching the cap the untargeted (any-atom-
change) branch already had.
"""
import numpy as np
from gym.spaces import Box

from predicators import utils
from predicators.structs import Action, GroundAtom, Object, \
    ParameterizedOption, Predicate, State, Type
from predicators.utils import OptionExecutionFailure

_ROBOT = Type("robot", ["x"])


def _make_wait_option(target_atom: GroundAtom):
    param_option = ParameterizedOption(
        "Wait", [_ROBOT],
        Box(0, 1, (0, )),
        policy=lambda s, m, o, p: Action(np.zeros(1, dtype=np.float32)),
        initiable=lambda s, m, o, p: True,
        terminal=lambda s, m, o, p: False)
    robot = Object("robby", _ROBOT)
    option = param_option.ground([robot], np.zeros(0, dtype=np.float32))
    option.memory["wait_target_atoms"] = {target_atom}
    return robot, option


def _run_until_exhausted(policy, state, max_calls: int) -> int:
    """Step the policy on a static state; return the call index at which the
    one-option plan was exhausted (the Wait terminated)."""
    for call in range(1, max_calls + 1):
        try:
            policy(state)
        except OptionExecutionFailure:
            return call
    return -1


def test_targeted_wait_backstop_terminates() -> None:
    """A Wait whose target atom never holds ends at wait_option_max_steps
    instead of spinning forever."""
    utils.reset_config({
        "wait_option_terminate_on_atom_change": True,
        "wait_option_max_steps": 5,
    })
    never = Predicate("Never", [_ROBOT], lambda s, o: False)
    robot_obj = Object("robby", _ROBOT)
    robot, option = _make_wait_option(GroundAtom(never, [robot_obj]))
    state = State({robot: np.array([0.0])})
    policy = utils.option_plan_to_policy([option],
                                         abstract_function=lambda s: set())
    # Call 1 starts the Wait (steps=1 after it); calls 2..5 observe
    # steps 1..4 and keep waiting; call 6 observes steps=5, trips the
    # backstop, asks for the next option, and exhausts the plan.
    exhausted_at = _run_until_exhausted(policy, state, max_calls=50)
    assert exhausted_at == 6


def test_targeted_wait_still_waits_below_the_cap() -> None:
    """With the default wait_option_max_steps (inf), a targeted Wait keeps
    waiting until the belief rollout cap (max_num_steps_option_rollout)."""
    utils.reset_config({
        "wait_option_terminate_on_atom_change": True,
        "max_num_steps_option_rollout": 1000,
    })
    never = Predicate("Never", [_ROBOT], lambda s, o: False)
    robot_obj = Object("robby", _ROBOT)
    robot, option = _make_wait_option(GroundAtom(never, [robot_obj]))
    state = State({robot: np.array([0.0])})
    policy = utils.option_plan_to_policy([option],
                                         abstract_function=lambda s: set())
    assert _run_until_exhausted(policy, state, max_calls=50) == -1


def test_wait_backstop_defaults_to_the_belief_rollout_cap() -> None:
    """With wait_option_max_steps infinite, the real executor's Wait ends where
    a belief rollout's would: at max_num_steps_option_rollout.

    The two caps agreeing is what makes a certified plan's Wait budget
    transfer to the real episode.
    """
    utils.reset_config({
        "wait_option_terminate_on_atom_change": True,
        "max_num_steps_option_rollout": 5,
    })
    assert utils.wait_rollout_step_cap() == 5
    never = Predicate("Never", [_ROBOT], lambda s, o: False)
    robot_obj = Object("robby", _ROBOT)
    robot, option = _make_wait_option(GroundAtom(never, [robot_obj]))
    state = State({robot: np.array([0.0])})
    policy = utils.option_plan_to_policy([option],
                                         abstract_function=lambda s: set())
    assert _run_until_exhausted(policy, state, max_calls=50) == 6
    # An untargeted (any-atom-change) Wait shares the cap.
    robot, option = _make_wait_option(GroundAtom(never, [robot_obj]))
    option.memory.pop("wait_target_atoms")
    policy = utils.option_plan_to_policy([option],
                                         abstract_function=lambda s: set())
    assert _run_until_exhausted(policy, state, max_calls=50) == 6


def test_targeted_wait_terminates_on_target_before_cap() -> None:
    """A target atom that becomes true still ends the Wait immediately, before
    the backstop fires."""
    utils.reset_config({
        "wait_option_terminate_on_atom_change": True,
        "wait_option_max_steps": 30,
    })
    calls = {"n": 0}

    def _eventually_holds(s, o) -> bool:
        del s, o
        return calls["n"] >= 3

    pred = Predicate("Eventually", [_ROBOT], _eventually_holds)
    robot_obj = Object("robby", _ROBOT)
    atom = GroundAtom(pred, [robot_obj])
    robot, option = _make_wait_option(atom)
    state = State({robot: np.array([0.0])})

    def _abstract(s):
        calls["n"] += 1
        return {atom} if _eventually_holds(s, None) else set()

    policy = utils.option_plan_to_policy([option], abstract_function=_abstract)
    exhausted_at = _run_until_exhausted(policy, state, max_calls=50)
    assert 0 < exhausted_at < 30


def _make_neg_only_wait_option(neg_atom: GroundAtom):
    """A Wait annotated with ONLY a negative target (-> {NOT ...}), the
    consumption-signature pattern; no positive key is ever set, matching
    sketch_refinement's conditional injection."""
    param_option = ParameterizedOption(
        "Wait", [_ROBOT],
        Box(0, 1, (0, )),
        policy=lambda s, m, o, p: Action(np.zeros(1, dtype=np.float32)),
        initiable=lambda s, m, o, p: True,
        terminal=lambda s, m, o, p: False)
    robot = Object("robby", _ROBOT)
    option = param_option.ground([robot], np.zeros(0, dtype=np.float32))
    option.memory["wait_target_neg_atoms"] = {neg_atom}
    return robot, option


def test_neg_only_wait_waits_and_hits_backstop() -> None:
    """A Wait with only NOT targets that never clear waits to the step cap
    instead of crashing (an assert on the missing positive-target.

    key used to kill the whole episode - 2026-08-27 bridge_demo1 run).
    """
    utils.reset_config({
        "wait_option_terminate_on_atom_change": True,
        "wait_option_max_steps": 5,
    })
    always = Predicate("Always", [_ROBOT], lambda s, o: True)
    robot_obj = Object("robby", _ROBOT)
    atom = GroundAtom(always, [robot_obj])
    robot, option = _make_neg_only_wait_option(atom)
    state = State({robot: np.array([0.0])})
    policy = utils.option_plan_to_policy([option],
                                         abstract_function=lambda s: {atom})
    exhausted_at = _run_until_exhausted(policy, state, max_calls=50)
    assert exhausted_at == 6  # same schedule as the positive-target cap


def test_neg_only_wait_terminates_when_atom_clears() -> None:
    """A NOT-only Wait ends as soon as its atom stops holding."""
    utils.reset_config({
        "wait_option_terminate_on_atom_change": True,
        "wait_option_max_steps": 100,
    })
    calls = {"n": 0}
    pred = Predicate("Consumed", [_ROBOT], lambda s, o: calls["n"] < 3)
    robot_obj = Object("robby", _ROBOT)
    atom = GroundAtom(pred, [robot_obj])
    robot, option = _make_neg_only_wait_option(atom)
    state = State({robot: np.array([0.0])})

    def _abstract(s):
        del s
        calls["n"] += 1
        return {atom} if calls["n"] < 3 else set()

    policy = utils.option_plan_to_policy([option], abstract_function=_abstract)
    exhausted_at = _run_until_exhausted(policy, state, max_calls=50)
    assert 0 < exhausted_at < 10
