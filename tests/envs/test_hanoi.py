"""Test cases for the Towers of Hanoi environment."""
import numpy as np

from predicators import utils
from predicators.envs.hanoi import HanoiEnv
from predicators.ground_truth_models import get_gt_options


def test_hanoi():
    """Tests for HanoiEnv class."""
    utils.reset_config({"env": "hanoi"})
    env = HanoiEnv()
    for task in env.get_train_tasks():
        for obj in task.init:
            assert len(obj.type.feature_names) == len(task.init[obj])
    for task in env.get_test_tasks():
        for obj in task.init:
            assert len(obj.type.feature_names) == len(task.init[obj])
    assert len(env.predicates) == 7
    assert {pred.name for pred in env.goal_predicates} == {"On", "OnPeg"}
    assert len(get_gt_options(env.get_name())) == 3
    assert len(env.types) == 3
    assert env.action_space.shape == (3, )

    disk_type = [t for t in env.types if t.name == "disk"][0]
    peg_type = [t for t in env.types if t.name == "peg"][0]

    task = env.get_test_tasks()[0]
    state = task.init
    # The goal should not already hold in the initial state.
    assert not task.task.goal_holds(state)
    # There should be three disks stacked (largest on the bottom) on a single
    # peg, and three pegs.
    disks = sorted(o for o in state if o.is_instance(disk_type))
    pegs = [o for o in state if o.is_instance(peg_type)]
    assert len(disks) == 3
    assert len(pegs) == 3
    xs = {state.get(d, "pose_x") for d in disks}
    assert len(xs) == 1  # all on the same peg initially
    # The bottom disk should be the widest.
    disk_by_z = sorted(disks, key=lambda d: state.get(d, "pose_z"))
    widths = [state.get(d, "width") for d in disk_by_z]
    assert widths == sorted(widths, reverse=True)

    # Rendering (including while holding a disk) should not error.
    env.render_state(state, task, caption="init")
    robot = [o for o in state if o.type.name == "robot"][0]
    Pick = [o for o in get_gt_options(env.get_name()) if o.name == "Pick"][0]
    top_disk = disk_by_z[-1]
    act = Pick.ground([robot, top_disk], np.zeros(0)).policy(state)
    held_state = env.simulate(state, act)
    assert held_state.get(top_disk, "held") > env.held_tol
    env.render_state(held_state, task, caption="holding")


def test_hanoi_illegal_move():
    """A larger disk cannot be placed on a smaller one."""
    utils.reset_config({"env": "hanoi"})
    env = HanoiEnv()
    options = get_gt_options(env.get_name())
    Pick = [o for o in options if o.name == "Pick"][0]
    PutOnPeg = [o for o in options if o.name == "PutOnPeg"][0]

    task = env.get_test_tasks()[0]
    state = task.init
    disk_type = [t for t in env.types if t.name == "disk"][0]
    peg_type = [t for t in env.types if t.name == "peg"][0]
    robot = [o for o in state if o.type.name == "robot"][0]
    disks = sorted(o for o in state if o.is_instance(disk_type))
    pegs = sorted(o for o in state if o.is_instance(peg_type))

    # Move the smallest (top) disk onto an empty peg.
    small_disk = min(disks, key=lambda d: state.get(d, "width"))
    empty_peg = max(pegs, key=lambda p: state.get(p, "pose_x"))
    state = env.simulate(
        state, Pick.ground([robot, small_disk], np.zeros(0)).policy(state))
    state = env.simulate(
        state, PutOnPeg.ground([robot, empty_peg], np.zeros(0)).policy(state))
    assert env._OnPeg_holds(state, [small_disk, empty_peg])  # pylint: disable=protected-access

    # Now pick the next (medium) disk and try to place it on the small disk;
    # this is illegal and should be a no-op.
    medium_disk = sorted(disks, key=lambda d: state.get(d, "width"))[1]
    state = env.simulate(
        state, Pick.ground([robot, medium_disk], np.zeros(0)).policy(state))
    before = state.copy()
    state = env.simulate(
        state, PutOnPeg.ground([robot, empty_peg], np.zeros(0)).policy(state))
    # The medium disk is still held; the illegal placement did nothing.
    assert state.allclose(before)
    assert env._Holding_holds(state, [medium_disk])  # pylint: disable=protected-access
