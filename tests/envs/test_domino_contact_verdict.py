"""Tests for the Domino verdict that reads the engine's contacts."""
from types import SimpleNamespace
from typing import List, Set, Tuple

import numpy as np

from predicators import utils
from predicators.envs.pybullet_domino.env import PyBulletDominoComposedEnv
from predicators.envs.pybullet_domino.task_generators import goal_text
from predicators.structs import Object, State, Type

_DOMINO = Type("domino", ["x"])
_GREEN = Object("domino_0", _DOMINO)
_BLUE = Object("domino_2", _DOMINO)
_GREEN_ID, _BLUE_ID, _FLOOR_ID = 7, 9, 0
_LEFT_FINGER, _RIGHT_FINGER, _PALM = 18, 19, 17


def _clean(touches: Set[Tuple[int, int, int]],
           num_states: int = 11,
           first_action: int = 4) -> bool:
    """The contact check on an env stand-in with two dominoes."""
    green = SimpleNamespace(id=_GREEN_ID, name="domino_0")
    blue = SimpleNamespace(id=_BLUE_ID, name="domino_2")
    env = SimpleNamespace(
        _episode_actions=10,
        _robot_touches=touches,
        _domino_component=SimpleNamespace(dominos=[green, blue]),
        _pybullet_robot=SimpleNamespace(left_finger_id=_LEFT_FINGER,
                                        right_finger_id=_RIGHT_FINGER))
    state = State({
        _GREEN: np.zeros(1, dtype=np.float32),
        _BLUE: np.zeros(1, dtype=np.float32)
    })
    states: List[State] = [state] * num_states
    return PyBulletDominoComposedEnv.clean_push_contacts(
        env,  # type: ignore[arg-type]
        states,
        first_action,
        [_GREEN])


def test_only_fingertips_on_the_green_is_a_clean_push() -> None:
    """A fingertip on the green, and the wheels on the floor, are clean."""
    assert _clean({(5, _LEFT_FINGER, _GREEN_ID), (6, -1, _FLOOR_ID)})


def test_the_palm_or_a_touched_blue_needs_the_replay() -> None:
    """The arm's body on the green, or any robot link on a blue, is not."""
    assert not _clean({(5, _PALM, _GREEN_ID)})
    assert not _clean({(5, _LEFT_FINGER, _BLUE_ID)})


def test_contacts_before_the_push_do_not_count() -> None:
    """Carrying and placing the blues happens before the push."""
    assert _clean({(3, _PALM, _BLUE_ID), (5, _LEFT_FINGER, _GREEN_ID)})


def test_states_from_another_episode_prove_nothing() -> None:
    """The record speaks only for the episode that made it."""
    assert not _clean({(5, _LEFT_FINGER, _GREEN_ID)}, num_states=8)


def test_the_goal_text_states_the_rule_in_force() -> None:
    """The task text follows the flag."""
    utils.reset_config({"domino_contact_verdict": False})
    assert goal_text.configured(
        goal_text.MIN_BLOCK_GOAL_NL) == goal_text.MIN_BLOCK_GOAL_NL
    utils.reset_config({"domino_contact_verdict": True})
    text = goal_text.configured(goal_text.MIN_BLOCK_GOAL_NL)
    assert goal_text.CONTACT_VERIFICATION_NL in text
    assert goal_text.CASCADE_VERIFICATION_NL not in text
    assert text.endswith("A rehearsal always makes that replay.")
    # An arm with no rehearsal reads the rule alone.
    utils.reset_config({
        "domino_contact_verdict": True,
        "domino_goal_names_rehearsal": False
    })
    plain = goal_text.configured(goal_text.MIN_BLOCK_GOAL_NL)
    assert plain + goal_text.REHEARSAL_REPLAY_NL == text
    assert "rehears" not in plain
    utils.reset_config({"domino_contact_verdict": False})
