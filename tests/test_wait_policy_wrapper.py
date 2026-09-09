"""The option-policy wrapper's Wait handling without an abstraction."""
import numpy as np
from gym.spaces import Box

from predicators import utils
from predicators.structs import Action, Object, ParameterizedOption, State, \
    Type


def test_wait_without_abstraction_runs_to_the_step_backstop() -> None:
    """A caller that wires no abstract_function (plan execution with no model)
    must not crash on a Wait; the Wait ends at the step cap and the next option
    is requested."""
    utils.reset_config({
        "wait_option_terminate_on_atom_change": True,
        "wait_option_max_steps": 3,
    })
    robot_type = Type("robot", ["x"])
    robot = Object("robot", robot_type)
    state = State({robot: np.zeros(1, dtype=np.float32)})
    wait = ParameterizedOption(
        "Wait", [robot_type], Box(0, 1, (0, )),
        lambda s, m, o, p: Action(np.zeros(1, dtype=np.float32)),
        lambda s, m, o, p: True, lambda s, m, o, p: False)
    starts = []

    def _option_policy(s: State):
        del s
        option = wait.ground([robot], np.zeros(0, dtype=np.float32))
        starts.append(option)
        return option

    policy = utils.option_policy_to_policy(_option_policy,
                                           max_option_steps=100)
    for _ in range(8):
        policy(state)
    # Steps 1-3 run one Wait to its cap, steps 4-6 the next, and so on.
    assert len(starts) == 3
