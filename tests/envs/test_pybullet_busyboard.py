"""Unit tests for the busyboard env's wiring, charge dynamics, and skills.

Covers the four properties the domain rests on: the discrete hypothesis
space is well formed (distinct, canonically ordered drive conditions),
every generated goal is realizable while the degenerate "latch every
button" policy is not a solution, the hidden charge is genuinely delayed
and genuinely hidden under partial observability, and the shared push
skill can operate every button.
"""
# pylint: disable=protected-access
from __future__ import annotations

import numpy as np
import pytest

from predicators import utils
from predicators.structs import Action

_PUSH_PARAMS = np.array([0.07, 0.05], dtype=np.float32)


def _make_env(**overrides):
    """A freshly configured env, plus its module for the wiring helpers.

    Deliberately does NOT reload the env module. The observability
    decision is read at construction time, so a new instance picks up a
    new config on its own, while a reload would leave two classes both
    answering to "pybullet_busyboard" in the subclass registry that
    ``create_new_env`` scans - and which one it found would depend on
    test order.
    """
    config = {
        "env": "pybullet_busyboard",
        "seed": 0,
        "num_train_tasks": 3,
        "num_test_tasks": 3,
    }
    config.update(overrides)
    utils.reset_config(config)
    from predicators.envs import \
        pybullet_busyboard  # pylint: disable=import-outside-toplevel
    return pybullet_busyboard, pybullet_busyboard.PyBulletBusyBoardEnv(
        use_gui=False)


@pytest.fixture(name="env_module")
def _env_module():
    """Function-scoped: every test re-applies the fully-observable config.

    Module scope would let the partial-observability test's config leak
    into whatever ran after it, which would resolve the ground-truth
    simulator to the recurrent (latent-carrying) module while the env in
    hand was built fully observable.
    """
    return _make_env()


def test_legal_pairs_are_canonical_and_distinct(env_module):
    """The per-lamp hypothesis space is B plain drives plus B-choose-2
    conjunctive ones, with no duplicate under the symmetry of AND."""
    mod, _ = env_module
    for num_buttons in (2, 3, 4, 5):
        pairs = mod.legal_pairs(num_buttons)
        assert len(pairs) == len(set(pairs))
        assert len(pairs) == num_buttons + num_buttons * (num_buttons - 1) // 2
        for driver, enabler in pairs:
            # Conjunctive pairs are stored low-index-first, so the two
            # labellings of one condition are never separate hypotheses.
            assert mod.canonical_pair(driver, enabler) == (driver, enabler)
            if enabler != mod.NO_ENABLER:
                assert driver < enabler


def test_wiring_conditions_are_distinct_at_every_board_size(env_module):
    """No two lamps share a drive condition.

    Identically wired lamps could not be separated by any experiment,
    nor asked for by any goal, so the projection onto smaller boards
    must never collapse two of them together.
    """
    mod, _ = env_module
    for num_buttons in (3, 4, 5):
        for num_lamps in (1, 2, 3):
            driver, enabler = mod.canonical_wiring(num_buttons, num_lamps)
            conditions = [
                mod.canonical_pair(d, e) for d, e in zip(driver, enabler)
            ]
            assert len(set(conditions)) == len(conditions)
            for d, e in zip(driver, enabler):
                assert 0 <= d < num_buttons
                assert e == mod.NO_ENABLER or 0 <= e < num_buttons
                assert e != d


def test_every_goal_is_realizable_and_needs_some_lamp_dark(env_module):
    """Generated goals are achievable, and never solved by latching all."""
    _, env = env_module
    tasks = env._generate_train_tasks() + env._generate_test_tasks()
    assert tasks
    for task in tasks:
        metrics = task.offline_task_metrics
        num_lamps = int(metrics["wiring_num_lamps"])
        num_buttons = int(metrics["wiring_num_buttons"])
        driver = [int(metrics[f"wiring_driver_{i}"]) for i in range(num_lamps)]
        enabler = [
            int(metrics[f"wiring_enabler_{i}"]) for i in range(num_lamps)
        ]
        want = {
            atom.objects[0].name: atom.predicate.name == "LampOn"
            for atom in task.goal_description
        }
        target = [want[f"lamp{i}"] for i in range(num_lamps)]

        assert tuple(target) in set(
            env._realizable_targets(driver, enabler, num_buttons))
        # With more than one lamp, some lamp must be dark - otherwise
        # turning every button on would solve the task without knowing
        # anything about the board.
        if num_lamps >= 2:
            assert not all(target)
        all_on = [True] * num_buttons
        latch_everything = [
            env._driven(all_on, d, e) for d, e in zip(driver, enabler)
        ]
        if num_lamps >= 2:
            assert latch_everything != target


def test_charge_is_delayed_and_conjunctive(env_module):
    """A lamp lights only after a sustained drive, and an interlocked lamp
    ignores its driver until the enabler is on too."""
    mod, env = env_module
    env._generate_train_tasks()
    env.reset("train", 0)
    lamps = env._lamps[:env._num_active_lamps]
    buttons = env._buttons[:env._num_active_buttons]
    hold = Action(np.array(env._pybullet_robot.get_joints(), dtype=np.float32))

    # Pick a lamp with an interlock if this board has one.
    idx = next((i for i, e in enumerate(env._enabler) if e != mod.NO_ENABLER),
               None)
    if idx is not None:
        # Driver alone: no charge accumulates at all.
        env._set_button_on(buttons[env._driver[idx]], True)
        for _ in range(40):
            env.step(hold)
        assert env._charges[lamps[idx].name] == 0.0

        # Adding the enabler starts the accumulation, but the lamp stays
        # visibly dark well past the point where it is already charging.
        env._set_button_on(buttons[env._enabler[idx]], True)
        for _ in range(10):
            env.step(hold)
        state = env._get_state()
        assert env._charges[lamps[idx].name] > 0.0
        assert float(state.get(lamps[idx], "brightness")) == 0.0
        assert not env._LampOn_holds(state, [lamps[idx]])

        # Held long enough, it lights.
        for _ in range(60):
            env.step(hold)
        state = env._get_state()
        assert env._LampOn_holds(state, [lamps[idx]])

        # Dropping the enabler puts it out again, and faster.
        env._set_button_on(buttons[env._enabler[idx]], False)
        for _ in range(25):
            env.step(hold)
        state = env._get_state()
        assert env._LampOff_holds(state, [lamps[idx]])
        assert env._charges[lamps[idx].name] == 0.0


def test_charge_is_hidden_under_partial_observability():
    """PO mode drops charge from the observation but keeps the readout."""
    _, env = _make_env(partially_observable=True)
    assert "charge" not in env._lamp_type.feature_names
    assert "brightness" in env._lamp_type.feature_names
    env._generate_train_tasks()
    env.reset("train", 0)
    state = env._get_state()
    lamp = env._lamps[0]
    with pytest.raises(Exception):
        state.get(lamp, "charge")


def test_push_skill_operates_every_button(env_module):
    """Every button can be latched on and off by the shared push skill.

    This is the domain's central design constraint: the manipulation is
    meant to be free so that the difficulty is all inference.
    """
    _, env = env_module
    from predicators.ground_truth_models import \
        get_gt_options  # pylint: disable=import-outside-toplevel
    options = {o.name: o for o in get_gt_options("pybullet_busyboard")}
    assert set(options) == {"PressButton", "ReleaseButton", "Wait"}

    env._generate_train_tasks()
    env.reset("train", 0)
    state = env._get_state()
    robot = next(o for o in state if o.type.name == "robot")
    buttons = sorted((o for o in state if o.type.name == "button"),
                     key=lambda o: o.name)

    for name, want_on in (("PressButton", 1.0), ("ReleaseButton", 0.0)):
        for button in buttons:
            option = options[name].ground([robot, button], _PUSH_PARAMS)
            assert option.initiable(state)
            for _ in range(300):
                state = env.step(option.policy(state))
                if option.terminal(state):
                    break
            else:
                pytest.fail(f"{name}({button.name}) did not terminate")
            assert float(state.get(button, "is_on")) == want_on


def test_ground_truth_simulator_reproduces_the_env(env_module):
    """The GT residual program matches the env except at latch steps.

    The env applies its residual after the base sim has stepped, so it
    reads the button states an action ends with while a teacher-forced
    prediction reads the state the action starts from. The two therefore
    differ by exactly one charge increment on the step a button latches,
    and agree everywhere else.
    """
    _, env = env_module
    # pylint: disable=import-outside-toplevel
    from predicators.code_sim_learning.utils import apply_rules
    from predicators.ground_truth_models import get_gt_options, \
        get_gt_simulator

    # pylint: enable=import-outside-toplevel

    rules, specs, _ = get_gt_simulator("pybullet_busyboard")
    params = {s.name: s.init_value for s in specs}
    options = {o.name: o for o in get_gt_options("pybullet_busyboard")}

    env._generate_train_tasks()
    env.reset("train", 0)
    state = env._get_state()
    robot = next(o for o in state if o.type.name == "robot")
    buttons = sorted((o for o in state if o.type.name == "button"),
                     key=lambda o: o.name)

    steps = disagreements = 0

    def _step(action):
        nonlocal state, steps, disagreements
        steps += 1
        before = [float(state.get(b, "is_on")) for b in buttons]
        predicted = apply_rules(state, rules, params)
        state = env.step(action)
        after = [float(state.get(b, "is_on")) for b in buttons]
        worst = max((abs(float(state.get(obj, feat)) - float(value))
                     for obj, upd in predicted.items()
                     for feat, value in upd.items()),
                    default=0.0)
        if worst > 1e-9:
            disagreements += 1
            # Only ever at a latch step, and only ever by one increment.
            assert before != after
            assert worst == pytest.approx(params["charge_rate"])

    press = options["PressButton"]
    for button in buttons[:2]:
        option = press.ground([robot, button], _PUSH_PARAMS)
        option.initiable(state)
        for _ in range(300):
            _step(option.policy(state))
            if option.terminal(state):
                break
    wait = options["Wait"].ground([robot], np.array([], dtype=np.float32))
    wait.initiable(state)
    for _ in range(90):
        _step(wait.policy(state))

    assert steps > 100
    assert disagreements <= len(buttons[:2])


def test_training_wiring_extends_to_every_test_board(env_module):
    """The core (training) board's wiring is a sub-relation of every board.

    What an agent learns about a lamp on the training board has to stay
    true of that lamp on every test board, so a core lamp keeps its
    drive condition verbatim at every size, and the lamps a bigger board
    adds draw their driver from the buttons it adds.
    """
    mod, _ = env_module
    from predicators.settings import \
        CFG  # pylint: disable=import-outside-toplevel
    core_buttons, core_lamps = mod.core_board()
    assert (core_buttons, core_lamps) == (3, 2)
    # The board sizes the distribution produces: train, then test.
    sizes = [(3, 2), (4, 2), (4, 3), (5, 2), (5, 3)]
    for seed in range(5):
        CFG.seed = seed
        core_driver, core_enabler = mod.canonical_wiring(
            core_buttons, core_lamps)
        assert all(0 <= d < core_buttons for d in core_driver)
        assert all(e == mod.NO_ENABLER or 0 <= e < core_buttons
                   for e in core_enabler)
        for num_buttons, num_lamps in sizes:
            driver, enabler = mod.canonical_wiring(num_buttons, num_lamps)
            assert driver[:core_lamps] == core_driver
            assert enabler[:core_lamps] == core_enabler
            # Canonical form orders a pair by index, so the extension
            # button may sit in either slot; what matters is that the
            # condition involves a button the training board lacks.
            for d, e in zip(driver[core_lamps:], enabler[core_lamps:]):
                assert any(core_buttons <= b < num_buttons for b in (d, e))


def test_extension_lamps_are_only_ever_dark_targets(env_module):
    """A lamp the training board never showed is never asked to be lit."""
    mod, _ = env_module
    _, env = _make_env(num_test_tasks=12)
    _, core_lamps = mod.core_board()
    saw_extension_lamp = False
    for task in env._generate_test_tasks():
        for atom in task.goal_description:
            lamp_idx = int(atom.objects[0].name[len("lamp"):])
            if lamp_idx >= core_lamps:
                saw_extension_lamp = True
                assert atom.predicate.name == "LampOff"
    assert saw_extension_lamp


def test_colours_are_distinct_and_stable(env_module):
    """Every button and lamp has its own colour, the same on every board."""
    _, env = env_module
    seen = {}
    for task in env._generate_train_tasks() + env._generate_test_tasks():
        state = task.init
        colours = {}
        for obj in state:
            if obj.type.name not in ("button", "lamp"):
                continue
            colour = int(state.get(obj, "color"))
            name = env.color_name(colour)
            colours[obj.name] = name
            assert seen.setdefault(obj.name, name) == name
        # Distinct within a board, and named the way text refers to them.
        assert len(set(colours.values())) == len(colours)
        assert colours["button0"] == "red"
        assert colours["lamp0"] == "yellow"
        assert "the yellow lamp (lamp0)" in task.goal_nl
    # The live readout agrees with the task's init state.
    env.reset("train", 0)
    live = env._get_state()
    init = env.get_task("train", 0).init
    for obj in init:
        if obj.type.name in ("button", "lamp"):
            assert live.get(obj, "color") == init.get(obj, "color")
