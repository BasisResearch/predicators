"""Unit tests for the busyboard env's wiring, latch, breaker, charge dynamics,
and skills.

Covers the properties the domain rests on: the discrete hypothesis space
is well formed (distinct drives, canonical conditions), every generated
goal is realizable while the degenerate "latch every button" policy is
not a solution, the arming latch makes press order matter, an inhibitor
blocks its lamp, the breaker trips on overload and closes on a power
cycle, the hidden charge is genuinely delayed and genuinely hidden under
partial observability, the ground-truth simulator reproduces the env,
and the shared push skill can operate every button.
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


def _hold(env) -> Action:
    return Action(np.array(env._pybullet_robot.get_joints(), dtype=np.float32))


def _run(env, action: Action, steps: int) -> None:
    for _ in range(steps):
        env.step(action)


def test_legal_conditions_are_distinct_and_canonical(env_module):
    """The per-lamp hypothesis space is every driver times the subsets of at
    most two other buttons, driver and enablers distinct, enablers sorted."""
    mod, _ = env_module
    for num_buttons in (2, 3, 4, 8):
        conds = mod.legal_conditions(num_buttons)
        assert len(conds) == len(set(conds))
        others = num_buttons - 1
        expected = num_buttons * (1 + others + others * (others - 1) // 2)
        assert len(conds) == expected
        for cond in conds:
            assert cond == cond.canonical()
            assert cond.driver not in cond.enablers
            assert list(cond.enablers) == sorted(cond.enablers)
            assert cond.inhibitor == mod.NO_BUTTON
        # The one-enabler slice, as pairs, for older callers.
        pairs = mod.legal_pairs(num_buttons)
        assert len(pairs) == num_buttons * num_buttons
    # Canonical form drops a button that is both driver and enabler or
    # both in the drive and the inhibitor slot, and orders the enablers.
    cond = mod.Condition(2, (3, 2, 1), 3).canonical()
    assert cond == mod.Condition(2, (1, 3), mod.NO_BUTTON)
    assert str(mod.Condition(0, (2, ), 1)) == "b0 & b2 & not b1"
    # Metrics round-trip.
    metrics = mod.Condition(4, (1, 2), 5).to_metrics(3)
    assert mod.Condition.from_metrics(metrics,
                                      3) == mod.Condition(4, (1, 2), 5)


def test_wiring_drives_are_distinct_at_every_board_size(env_module):
    """No two lamps share a drive.

    Identically driven lamps could not be separated by any experiment,
    nor asked for by any goal, so the projection onto smaller boards
    must never collapse two of them together.
    """
    mod, _ = env_module
    for num_buttons in (3, 4, 6, 8):
        for num_lamps in (1, 2, 3, 4):
            wiring = mod.canonical_wiring(num_buttons, num_lamps)
            keys = [c.drive_key() for c in wiring]
            assert len(set(keys)) == len(keys)
            for cond in wiring:
                assert cond == cond.canonical()
                assert all(0 <= b < num_buttons for b in cond.buttons)
                assert len(cond.enablers) <= mod.MAX_ENABLERS


def test_every_goal_is_realizable_and_needs_some_lamp_dark(env_module):
    """Generated goals are achievable, and never solved by latching all."""
    mod, env = env_module
    from predicators.settings import \
        CFG  # pylint: disable=import-outside-toplevel
    tasks = env._generate_train_tasks() + env._generate_test_tasks()
    assert tasks
    for task in tasks:
        metrics = task.offline_task_metrics
        num_lamps = int(metrics["wiring_num_lamps"])
        num_buttons = int(metrics["wiring_num_buttons"])
        wiring = [
            mod.Condition.from_metrics(metrics, i) for i in range(num_lamps)
        ]
        want = {
            atom.objects[0].name: atom.predicate.name == "LampOn"
            for atom in task.goal_description
        }
        target = tuple(want[f"lamp{i}"] for i in range(num_lamps))
        assert target in set(mod.realizable_targets(wiring, num_buttons))
        assert sum(target) <= CFG.busyboard_breaker_limit
        # With more than one lamp, some lamp must be dark - otherwise
        # turning every button on would solve the task without knowing
        # anything about the board.
        if num_lamps >= 2:
            assert not all(target)
        # Pressing everything in board order either trips the breaker
        # or lights the wrong set.
        driven, max_driven = mod.press_sequence_outcome(
            wiring, list(range(num_buttons)), latch=True)
        if num_lamps >= 2:
            assert max_driven > CFG.busyboard_breaker_limit or \
                driven != target


def test_latch_makes_press_order_a_hypothesis(env_module):
    """Two lamps on the same two buttons in opposite roles are different
    hypotheses: with the latch, either one can be lit alone, and the goal
    sampler knows it; without the latch nothing separates them."""
    mod, _ = env_module
    from predicators.settings import \
        CFG  # pylint: disable=import-outside-toplevel
    wiring = [mod.Condition(0, (1, )), mod.Condition(1, (0, ))]
    assert mod.press_sequence_outcome(wiring, [1, 0], True)[0] == \
        (True, False)
    assert mod.press_sequence_outcome(wiring, [0, 1], True)[0] == \
        (False, True)
    assert mod.press_sequence_outcome(wiring, [0, 1], False)[0] == \
        (True, True)
    assert set(mod.realizable_targets(wiring, 2)) == {(True, False),
                                                      (False, True)}
    CFG.busyboard_latch = False
    assert not mod.realizable_targets(wiring, 2)
    CFG.busyboard_latch = True
    # An inhibitor and the breaker shape the set too: a sole lamp with an
    # inhibitor can be kept dark only by pressing it, and three drivable
    # lamps cannot all be lit at once under a limit of two.
    wiring = [
        mod.Condition(0, (), 3),
        mod.Condition(1, (0, )),
        mod.Condition(2, (0, ))
    ]
    targets = set(mod.realizable_targets(wiring, 4))
    assert (False, True, True) in targets
    assert (True, True, False) in targets
    assert all(sum(t) <= 2 for t in targets)
    CFG.busyboard_breaker_limit = 0
    assert (True, True, True) not in set(mod.realizable_targets(wiring, 4))
    CFG.busyboard_breaker_limit = 2


def test_charge_is_delayed_conjunctive_and_latched(env_module):
    """A lamp lights only after a sustained drive, only once its enablers were
    on when its driver was pressed, and darkens when the drive is lost."""
    mod, env = env_module
    env._generate_train_tasks()
    env.reset("train", 0)
    lamps = env._lamps[:env._num_active_lamps]
    buttons = env._buttons[:env._num_active_buttons]
    hold = _hold(env)

    idx = next(i for i, c in enumerate(env._wiring) if c.enablers)
    cond = env._wiring[idx]
    lamp = lamps[idx]

    # Driver first: nothing, even with every enabler on afterwards.
    env._set_button_on(buttons[cond.driver], True)
    _run(env, hold, 5)
    for e in cond.enablers:
        env._set_button_on(buttons[e], True)
    _run(env, hold, 60)
    assert env._charges[lamp.name] == 0.0
    assert not env._armed[lamp.name]
    assert float(env._get_state().get(lamp, "armed")) == 0.0

    # Re-pressing the driver with the enablers on arms it; the charge
    # builds while the lamp stays visibly dark, then it lights.
    env._set_button_on(buttons[cond.driver], False)
    _run(env, hold, 3)
    env._set_button_on(buttons[cond.driver], True)
    _run(env, hold, 10)
    state = env._get_state()
    assert env._armed[lamp.name]
    assert float(state.get(lamp, "armed")) == 1.0
    assert env._charges[lamp.name] > 0.0
    assert float(state.get(lamp, "brightness")) == 0.0
    assert not env._LampOn_holds(state, [lamp])
    _run(env, hold, 60)
    state = env._get_state()
    assert env._LampOn_holds(state, [lamp])

    # Dropping an enabler puts it out again, and faster; the latch holds
    # until the driver itself goes off.
    env._set_button_on(buttons[cond.enablers[0]], False)
    _run(env, hold, 25)
    state = env._get_state()
    assert env._LampOff_holds(state, [lamp])
    assert env._charges[lamp.name] == 0.0
    assert env._armed[lamp.name]
    env._set_button_on(buttons[cond.driver], False)
    _run(env, hold, 2)
    assert not env._armed[lamp.name]
    del mod


def test_inhibitor_blocks_its_lamp(env_module):
    """A lamp with an inhibitor charges only while that button is off."""
    mod, env = env_module
    env._generate_train_tasks()
    env._generate_test_tasks()
    for split, count in (("train", 3), ("test", 3)):
        for task_idx in range(count):
            env.reset(split, task_idx)
            inhibited = [
                i for i, c in enumerate(env._wiring)
                if c.inhibitor != mod.NO_BUTTON
            ]
            if inhibited:
                break
        if inhibited:
            break
    assert inhibited, "the seed-0 boards have an inhibited lamp"
    idx = inhibited[0]
    cond = env._wiring[idx]
    lamp = env._lamps[idx]
    buttons = env._buttons[:env._num_active_buttons]
    hold = _hold(env)
    for e in cond.enablers:
        env._set_button_on(buttons[e], True)
    _run(env, hold, 3)
    env._set_button_on(buttons[cond.driver], True)
    _run(env, hold, 20)
    charged = env._charges[lamp.name]
    assert charged > 0.0
    env._set_button_on(buttons[cond.inhibitor], True)
    _run(env, hold, 25)
    assert env._charges[lamp.name] == 0.0
    assert env._armed[lamp.name]  # the latch is untouched by the inhibitor
    env._set_button_on(buttons[cond.inhibitor], False)
    _run(env, hold, 20)
    assert env._charges[lamp.name] > 0.0


def test_breaker_trips_on_overload_and_closes_on_power_cycle(env_module):
    """Driving more lamps than the limit trips the breaker: every charge drops,
    nothing charges, the tile reads tripped; releasing every button closes it
    again."""
    mod, env = env_module
    from predicators.settings import \
        CFG  # pylint: disable=import-outside-toplevel
    env._generate_test_tasks()
    env.reset("test", 0)
    wiring = env._wiring
    num_buttons = env._num_active_buttons
    buttons = env._buttons[:num_buttons]
    lamps = env._lamps[:env._num_active_lamps]
    hold = _hold(env)
    # A press order that drives two lamps (a realizable target) ...
    target = next(t for t in mod.realizable_targets(wiring, num_buttons)
                  if sum(t) == 2)
    order = None
    for mask in range(1 << num_buttons):
        on = [b for b in range(num_buttons) if mask >> b & 1]
        drivable = [c for c in wiring if all(b in on for b in c.drive_set)]
        ordered = sorted({b for c in drivable for b in c.drive_set})
        rest = [b for b in on if b not in ordered]
        from itertools import \
            permutations  # pylint: disable=import-outside-toplevel
        for perm in permutations(ordered):
            seq = rest + list(perm)
            driven, max_driven = mod.press_sequence_outcome(wiring, seq, True)
            if driven == target and max_driven <= 2:
                order = seq
                break
        if order is not None:
            break
    assert order is not None
    for b in order:
        env._set_button_on(buttons[b], True)
        _run(env, hold, 3)
    _run(env, hold, 30)
    assert not env._tripped
    assert [env._charges[l.name] > 0 for l in lamps] == list(target)

    # ... overloads the breaker once the limit is one.
    CFG.busyboard_breaker_limit = 1
    _run(env, hold, 2)
    assert env._tripped
    state = env._get_state()
    breaker = next(o for o in state if o.type.name == "breaker")
    assert env._BreakerTripped_holds(state, [breaker])
    assert all(env._charges[l.name] == 0.0 for l in lamps)
    assert not any(env._armed[l.name] for l in lamps)
    # Nothing charges while tripped, whatever the buttons say.
    _run(env, hold, 40)
    assert all(env._charges[l.name] == 0.0 for l in lamps)
    # A power cycle closes it.
    for b in order:
        env._set_button_on(buttons[b], False)
    _run(env, hold, 2)
    assert not env._tripped
    state = env._get_state()
    assert env._BreakerClosed_holds(state, [breaker])
    CFG.busyboard_breaker_limit = 2
    # The state round-trips the breaker: a restored tripped board stays
    # tripped until its buttons are released.
    tripped_state = state.copy()
    tripped_state.set(breaker, "tripped", 1.0)
    env._set_state(tripped_state)
    assert env._tripped
    assert float(env._get_state().get(breaker, "tripped")) == 1.0


def test_charge_and_latch_are_hidden_under_partial_observability():
    """PO mode drops charge and the latch from the observation but keeps the
    readout and the breaker."""
    _, env = _make_env(partially_observable=True)
    assert "charge" not in env._lamp_type.feature_names
    assert "armed" not in env._lamp_type.feature_names
    assert "brightness" in env._lamp_type.feature_names
    env._generate_train_tasks()
    env.reset("train", 0)
    state = env._get_state()
    lamp = env._lamps[0]
    with pytest.raises(Exception):
        state.get(lamp, "charge")
    with pytest.raises(Exception):
        state.get(lamp, "armed")
    breaker = next(o for o in state if o.type.name == "breaker")
    assert float(state.get(breaker, "tripped")) == 0.0
    # Restoring a PO state keeps the instance's own latch for a lamp
    # whose driver is on, and clears it for one whose driver is off.
    cond = env._wiring[0]
    env._armed[lamp.name] = True
    env._set_state(state)
    assert not env._armed[lamp.name]
    on_state = state.copy()
    on_state.set(env._buttons[cond.driver], "is_on", 1.0)
    env._armed[lamp.name] = True
    env._set_state(on_state)
    assert env._armed[lamp.name]


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

    env._generate_test_tasks()
    env.reset("test", 0)
    state = env._get_state()
    robot = next(o for o in state if o.type.name == "robot")
    buttons = sorted((o for o in state if o.type.name == "button"),
                     key=lambda o: o.name)
    assert len(buttons) >= 7

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
    differ on the step a button latches (by one charge increment, or by
    the driver's edge landing a step apart) and agree everywhere else.
    """
    mod, env = env_module
    # pylint: disable=import-outside-toplevel
    from predicators.code_sim_learning.utils import apply_rules_with_latent
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
    # Enablers first, then the driver, so the lamp arms and charges.
    cond = next(c for c in env._wiring if c.enablers)
    press_order = [buttons[e] for e in cond.enablers] + [buttons[cond.driver]]

    steps = disagreements = 0
    history = []
    latent = {}

    def _step(action):
        nonlocal state, steps, disagreements
        steps += 1
        before = [float(state.get(b, "is_on")) for b in buttons]
        predicted = apply_rules_with_latent(state, latent, history, rules,
                                            params)
        history.append((state, action))
        state = env.step(action)
        after = [float(state.get(b, "is_on")) for b in buttons]
        worst = max((abs(float(state.get(obj, feat)) - float(value))
                     for obj, upd in predicted.items()
                     for feat, value in upd.items()),
                    default=0.0)
        if worst > 1e-9:
            disagreements += 1
            # Only ever at a latch step.
            assert before != after

    press = options["PressButton"]
    for button in press_order:
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
    assert disagreements <= 2 * len(press_order)
    lamp = env._lamps[env._wiring.index(cond)]
    assert float(state.get(lamp, "charge")) > 0.0
    del mod


def test_training_wiring_extends_to_every_test_board(env_module):
    """The core (training) board's wiring is a sub-relation of every board.

    What an agent learns about a lamp on the training board has to stay
    true of that lamp on every test board while the added buttons stay
    off, so a core lamp keeps its drive verbatim at every size and may
    only gain an inhibitor among the added buttons; the lamps a bigger
    board adds draw their driver from the buttons it adds.
    """
    mod, _ = env_module
    from predicators.settings import \
        CFG  # pylint: disable=import-outside-toplevel
    core_buttons, core_lamps = mod.core_board()
    assert (core_buttons, core_lamps) == (4, 3)
    # The board sizes the distribution produces: train, then test.
    sizes = [(4, 3), (7, 4), (8, 4)]
    saw_extension_inhibitor = False
    for seed in range(5):
        CFG.seed = seed
        core = mod.canonical_wiring(core_buttons, core_lamps)
        for cond in core:
            assert all(0 <= b < core_buttons for b in cond.buttons)
        for num_buttons, num_lamps in sizes:
            wiring = mod.canonical_wiring(num_buttons, num_lamps)
            for cond, core_cond in zip(wiring[:core_lamps], core):
                assert cond.drive_key() == core_cond.drive_key()
                if core_cond.inhibitor != mod.NO_BUTTON:
                    assert cond.inhibitor == core_cond.inhibitor
                elif cond.inhibitor != mod.NO_BUTTON:
                    assert cond.inhibitor >= core_buttons
                    saw_extension_inhibitor = True
            for cond in wiring[core_lamps:]:
                assert any(core_buttons <= b < num_buttons
                           for b in cond.drive_set)
    assert saw_extension_inhibitor


def test_extension_lamps_can_be_lit_targets_only_when_allowed(env_module):
    """A lamp the training board never showed is a lit target at test only
    under ``busyboard_test_extension_lit``."""
    mod, _ = env_module
    _, core_lamps = mod.core_board()

    def _extension_lit(env) -> bool:
        for task in env._generate_test_tasks():
            for atom in task.goal_description:
                lamp_idx = int(atom.objects[0].name[len("lamp"):])
                if lamp_idx >= core_lamps and \
                        atom.predicate.name == "LampOn":
                    return True
        return False

    _, env = _make_env(num_test_tasks=12, busyboard_test_extension_lit=True)
    assert _extension_lit(env)
    _, env = _make_env(num_test_tasks=12, busyboard_test_extension_lit=False)
    assert not _extension_lit(env)


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
    assert len(seen) >= 8 + 4
    # The live readout agrees with the task's init state.
    env.reset("train", 0)
    live = env._get_state()
    init = env.get_task("train", 0).init
    for obj in init:
        if obj.type.name in ("button", "lamp"):
            assert live.get(obj, "color") == init.get(obj, "color")


def test_test_goals_light_at_least_min_lit_lamps(env_module):
    """Test goals compose conditions: at least ``busyboard_min_lit_test`` lamps
    lit and never more than the breaker allows, while train goals may ask for a
    single lamp."""
    del env_module  # only its config matters; this test builds its own env
    from predicators.settings import \
        CFG  # pylint: disable=import-outside-toplevel
    _, env = _make_env(num_train_tasks=12, num_test_tasks=12)
    assert CFG.busyboard_min_lit_test >= 2 > CFG.busyboard_min_lit_train

    def _num_lit(task):
        return sum(atom.predicate.name == "LampOn"
                   for atom in task.goal_description)

    test_lit = [_num_lit(t) for t in env._generate_test_tasks()]
    assert min(test_lit) >= CFG.busyboard_min_lit_test
    assert max(test_lit) <= CFG.busyboard_breaker_limit
    train_lit = [_num_lit(t) for t in env._generate_train_tasks()]
    assert min(train_lit) >= CFG.busyboard_min_lit_train
    assert 1 in train_lit, "training still has single-lamp goals"


def test_oracle_helper_predicates_and_processes(env_module):
    """The oracle's wiring predicates read the board's conditions off any
    state, the derived predicates follow the atoms, and the process model
    builds."""
    mod, env = env_module
    # pylint: disable=import-outside-toplevel
    from predicators.ground_truth_models import get_gt_helper_predicates, \
        get_gt_options, get_gt_processes
    from predicators.structs import GroundAtom

    # pylint: enable=import-outside-toplevel
    helpers = {
        p.name: p
        for p in get_gt_helper_predicates("pybullet_busyboard")
    }
    env._generate_test_tasks()
    env.reset("test", 0)
    state = env._get_state()
    preds = set(env.predicates) | set(helpers.values())
    atoms = utils.abstract(state, preds)
    buttons = env._buttons[:env._num_active_buttons]
    lamps = env._lamps[:env._num_active_lamps]
    for i, cond in enumerate(env._wiring):
        assert GroundAtom(helpers["Drives"],
                          [buttons[cond.driver], lamps[i]]) in atoms
        for e in cond.enablers:
            assert GroundAtom(helpers["Enables"],
                              [buttons[e], lamps[i]]) in atoms
        if cond.inhibitor != mod.NO_BUTTON:
            assert GroundAtom(helpers["Inhibits"],
                              [buttons[cond.inhibitor], lamps[i]]) in atoms
        assert GroundAtom(helpers["Disarmed"], [lamps[i]]) in atoms
        assert GroundAtom(helpers["Undriven"], [lamps[i]]) in atoms
        assert GroundAtom(helpers["Uninhibited"], [lamps[i]]) in atoms
        assert (GroundAtom(helpers["Enabled"], [lamps[i]]) in atoms) == \
            (not cond.enablers)
    assert GroundAtom(helpers["AllButtonsOff"], []) in atoms
    assert GroundAtom(helpers["Overloaded"], []) not in atoms
    assert not any(a.predicate.name == "JustPressed" for a in atoms)

    # Drive a lamp for real: Armed, Enabled and Driven follow.
    cond = env._wiring[0]
    hold = _hold(env)
    for e in cond.enablers:
        env._set_button_on(buttons[e], True)
    _run(env, hold, 2)
    env._set_button_on(buttons[cond.driver], True)
    _run(env, hold, 2)
    atoms = utils.abstract(env._get_state(), preds)
    assert GroundAtom(helpers["Armed"], [lamps[0]]) in atoms
    assert GroundAtom(helpers["Enabled"], [lamps[0]]) in atoms
    assert GroundAtom(helpers["Driven"], [lamps[0]]) in atoms
    assert GroundAtom(helpers["Undriven"], [lamps[0]]) not in atoms
    assert GroundAtom(helpers["AllButtonsOff"], []) not in atoms

    options = set(get_gt_options("pybullet_busyboard"))
    processes = get_gt_processes("pybullet_busyboard", preds, options)
    names = {p.name for p in processes}
    assert {
        "PressButton", "ReleaseButton", "Wait", "ClearPressed", "ArmLamp",
        "DisarmLamp", "LightLamp", "DarkenLamp", "TripBreaker", "ResetBreaker"
    } <= names
