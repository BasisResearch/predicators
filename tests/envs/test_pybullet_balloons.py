"""Unit tests for the balloons env's release, lift, pop, tasks, evaluator and
skills.

Covers the properties the domain rests on: an open clip frees its
balloon onto the box and the box hangs near the analytic height (and
nothing happens on the base sim), a freed balloon carried to the ceiling
bursts and loses the level, every generated level has a verified
reference subset the oracle's own plan realizes, box colours weigh
differently (and not on the base sim until overridden), and the process
model builds on the lift-law helpers.
"""
# pylint: disable=protected-access
from __future__ import annotations

from itertools import permutations

import numpy as np
import pybullet as p
import pytest

from predicators import utils
from predicators.structs import Action, GroundAtom


def _make_env(**overrides):
    config = {
        "env": "pybullet_balloons",
        "seed": 0,
        "num_train_tasks": 1,
        "num_test_tasks": 1,
        "skill_phase_use_motion_planning": False,
        # Task validation certifies every release order against the
        # evaluator's dwell; the default 25-step dwell makes validated
        # generation take many minutes per level. The pilots run dwell 1.
        "balloons_goal_dwell_steps": 1,
    }
    config.update(overrides)
    utils.reset_config(config)
    from predicators.envs import \
        pybullet_balloons  # pylint: disable=import-outside-toplevel
    return pybullet_balloons, pybullet_balloons.PyBulletBalloonsEnv(
        use_gui=False)


@pytest.fixture(name="env_module", scope="module")
def _env_module():
    return _make_env()


def _hold(env) -> Action:
    return Action(np.array(env._pybullet_robot.get_joints(), dtype=np.float32))


def _settle(env, steps: int = 250):
    obs = env._get_state()
    for _ in range(steps):
        obs = env._step_once(_hold(env))
    return obs


def _open_clips(env, state, indices):
    """Open the given clips by hand and let the box settle."""
    env._set_state(state)
    env._current_observation = env._get_state()
    for i in indices:
        env._set_clip_on(env._clips[i], True)
    return _settle(env)


def test_target_band_is_centered_inside_chute(env_module):
    """The target-height marker must not intersect either chute wall."""
    _, env = env_module
    state = env.level_state(0, [0], (0.5, 0.6))
    assert state.get(env._band, "x") == pytest.approx(env.box_xy[0])
    assert state.get(env._band, "y") == pytest.approx(env.box_xy[1])
    assert env.band_half_xy < env.chute_half_gap
    clearance = env.chute_half_gap - env.band_half_xy
    assert clearance >= env.chute_wall_half_thickness


def test_cable_visuals_preserve_motion_and_hide_popped_balloons():
    """Camera-visible cables must not change trajectories or task objects."""
    _, plain = _make_env()
    _, decorated = _make_env()
    try:
        for env in (plain, decorated):
            state = env.level_state(0, [3, 0], (0.5, 0.6))
            env._set_state(state)
            env._set_clip_on(env._clips[0], True)
            env._current_observation = env._get_state()
        for _ in range(100):
            decorated._sync_cable_visuals()
            expected = plain._step_once(_hold(plain))
            actual = decorated._step_once(_hold(decorated))
            for obj in expected:
                np.testing.assert_allclose(expected[obj],
                                           actual[obj],
                                           atol=1e-7,
                                           rtol=0)
        assert set(decorated._cable_bodies) == {decorated._balloons[0].name}
        for body, _ in decorated._cable_bodies.values():
            assert not p.getCollisionShapeData(
                body, -1, physicsClientId=decorated._physics_client_id)
            assert p.getDynamicsInfo(
                body, -1, physicsClientId=decorated._physics_client_id)[0] == 0
        decorated._sync_cable_visuals()
        shape_count = len(decorated._cable_shapes)
        decorated._sync_cable_visuals()
        assert len(decorated._cable_shapes) == shape_count
        decorated._popped[decorated._balloons[0].name] = True
        decorated._sync_cable_visuals()
        assert not decorated._cable_bodies
        fewer = decorated.level_state(0, [0], (0.5, 0.6))
        decorated._set_state(fewer)
        decorated._sync_cable_visuals()
        assert not decorated._cable_bodies
    finally:
        plain.dispose()
        decorated.dispose()


def test_release_lift_and_the_analytic_law(env_module):
    """An open clip frees its balloon onto the box, which hangs near the
    analytic height; on the base sim nothing rises."""
    mod, env = env_module
    lifts = [env.lift_at_ground(c) for c in range(len(env.BALLOON_PALETTE))]
    strong = int(np.argmax(lifts))
    state = env.level_state(0, [strong, 0], (0.5, 0.6))
    obs = _open_clips(env, state, [0])
    balloon = env._balloons[0]
    assert obs.get(env._clips[0], "is_on") == 1.0
    assert obs.get(balloon, "tied") == 1.0
    assert obs.get(env._balloons[1], "tied") == 0.0
    assert obs.get(balloon, "popped") == 0.0
    expected = env.hover_height(0, [strong])
    assert expected is not None
    assert abs(obs.get(env._box, "z") - expected) < 0.03
    assert obs.get(env._box, "z") > env.box_z + 0.05
    assert mod.box_at_rest(obs, env._box)
    base = mod.PyBulletBalloonsEnv(use_gui=False, skip_residual_dynamics=True)
    obs = _open_clips(base, state, [0])
    assert obs.get(base._clips[0], "is_on") == 1.0
    assert obs.get(base._balloons[0], "tied") == 0.0
    assert abs(obs.get(base._box, "z") - base.box_z) < 0.01


def test_reaching_the_ceiling_bursts_and_loses(env_module):
    """A box lifted to the ceiling bursts a balloon; the evaluator ends the
    episode without certifying it."""
    mod, env = env_module
    goal = {GroundAtom(env._InBand, [env._box, env._band])}
    evaluator = mod.BalloonsEvaluator(goal)
    state = env.level_state(0, [3, 2, 1], (0.5, 0.6))
    obs = _open_clips(env, state, [0, 1, 2])
    obs = _settle(env, 300)
    assert mod.any_popped(obs) is not None
    assert evaluator.terminated(obs)
    ok, why = evaluator._certify([state, obs], None)
    assert not ok and "burst" in why


def test_dwell_requires_consecutive_in_band_rest(env_module):
    """The evaluator wins only after ``balloons_goal_dwell_steps`` consecutive
    real steps at rest inside the band: a lone in-band frame at a swing's
    turning point does not, a burst always ends the level, and the goal text
    states the dwell."""
    mod, env = env_module
    goal = {GroundAtom(env._InBand, [env._box, env._band])}
    evaluator = mod.BalloonsEvaluator(goal)
    dwell = evaluator.dwell_steps
    assert dwell == utils.CFG.balloons_goal_dwell_steps >= 1
    state = env.level_state(0, [3, 2, 1], (0.5, 0.6))
    inside = state.copy()
    inside.set(env._box, "z", 0.55)
    inside.set(env._box, "speed", 0.0)
    swinging = inside.copy()
    swinging.set(env._box, "speed", 0.05)
    assert evaluator.terminated(inside)
    assert not evaluator.terminated(swinging)
    assert not evaluator.terminated_trajectory([])
    # A turning point: one still frame between moving ones.
    assert not evaluator.terminated_trajectory([swinging, inside, swinging])
    assert not evaluator.terminated_trajectory([swinging] + [inside] * dwell)
    assert evaluator.terminated_trajectory([swinging] + [inside] * (dwell + 1))
    assert evaluator.terminated_trajectory([inside] * (dwell + 1))
    popped = swinging.copy()
    popped.set(env._balloons[0], "popped", 1.0)
    assert evaluator.terminated_trajectory([swinging, popped])
    assert evaluator.reward([inside] * (dwell + 1), None) == 1.0
    assert evaluator.reward([swinging, inside, swinging], None) == 0.0
    assert not evaluator.solved([swinging, popped], None)
    assert f"{dwell} consecutive" in evaluator.objective_description()
    task = env.get_test_tasks()[0]
    assert f"{dwell} consecutive environment steps" in task.goal_nl
    assert task.offline_task_metrics["goal_dwell_steps"] == float(dwell)


def test_generated_levels_have_a_reference_subset_the_oracle_frees(env_module):
    """Each task records a verified reference subset and the oracle's plan
    reaches the evaluator's goal."""
    mod, env = env_module
    # pylint: disable-next=import-outside-toplevel
    from predicators.ground_truth_models.balloons.oracle import solve_level
    for tasks in (env.get_train_tasks(), env.get_test_tasks()):
        assert tasks
        for task in tasks:
            state = task.init
            subset = env.solution_subset(state)
            assert subset is not None
            balloons = env._active_balloons(state)
            for i, balloon in enumerate(balloons):
                flag = task.offline_task_metrics[f"solution_{balloon.name}"]
                assert flag == float(i in subset)
            assert "clip" in task.goal_nl
            assert not any(atom.holds(state) for atom in task.goal)
            assert not task.evaluator.terminated(state)
            plan = solve_level(env, state)
            assert plan is not None
            assert len(plan) == len(subset)
    del mod


def test_box_masses_by_colour_and_base_sim(env_module):
    """Box mass follows the colour on the real env, is the base mass on the
    base sim, and physical overrides apply."""
    mod, env = env_module
    state = env.level_state(1, [0, 1], (0.5, 0.6))
    env._set_state(state)
    mass = p.getDynamicsInfo(env._box.id,
                             -1,
                             physicsClientId=env._physics_client_id)[0]
    assert abs(mass - env.true_box_mass(1)) < 1e-6
    base = mod.PyBulletBalloonsEnv(use_gui=False, skip_residual_dynamics=True)
    base._set_state(state)
    mass = p.getDynamicsInfo(base._box.id,
                             -1,
                             physicsClientId=base._physics_client_id)[0]
    assert mass == base.box_base_mass
    base.apply_physical_param_overrides({"mass_oak": env.true_box_mass(1)})
    mass = p.getDynamicsInfo(base._box.id,
                             -1,
                             physicsClientId=base._physics_client_id)[0]
    assert abs(mass - env.true_box_mass(1)) < 1e-6
    assert "air_drag" in base.get_physical_param_info()
    with pytest.raises(ValueError):
        base.apply_physical_param_overrides({"mass_plastic": 1.0})


def test_oracle_helpers_and_processes(env_module):
    """``Needed`` marks the solution subset's clips, ``Holds`` pairs clips with
    their balloons, ``AllNeededOpen`` follows the open-clip atoms, and the
    process model builds."""
    mod, env = env_module
    # pylint: disable=import-outside-toplevel
    from predicators.ground_truth_models import get_gt_helper_predicates, \
        get_gt_options, get_gt_processes

    # pylint: enable=import-outside-toplevel
    helpers = {
        p_.name: p_
        for p_ in get_gt_helper_predicates("pybullet_balloons")
    }
    task = env.get_train_tasks()[0]
    state = task.init
    subset = env.solution_subset(state)
    balloons = env._active_balloons(state)
    clips = env._active_clips(state)
    for j, clip in enumerate(clips):
        assert GroundAtom(helpers["Needed"],
                          [clip, env._band]).holds(state) == (j in subset)
        for i, balloon in enumerate(balloons):
            assert GroundAtom(helpers["Holds"],
                              [clip, balloon]).holds(state) == (i == j)
    preds = set(env.predicates) | set(helpers.values())
    atoms = utils.abstract(state, preds)
    assert GroundAtom(helpers["AllNeededOpen"], [env._band]) not in atoms
    opened = state.copy()
    for j in subset:
        opened.set(clips[j], "is_on", 1.0)
    atoms = utils.abstract(opened, preds)
    assert GroundAtom(helpers["AllNeededOpen"], [env._band]) in atoms
    options = set(get_gt_options("pybullet_balloons"))
    assert {o.name for o in options} == {"Release", "Wait"}
    processes = get_gt_processes("pybullet_balloons", preds, options)
    assert {pr.name for pr in processes} == {"ReleaseClip", "Rise", "Wait"}
    del mod


def test_train_levels_cover_every_colour_and_material():
    """Two train levels together show all four balloon colours and both box
    materials; a test level holds the whole palette on a box material training
    showed."""
    _, env = _make_env(num_train_tasks=2, num_test_tasks=1)
    train = env.get_train_tasks()
    colors = set()
    boxes = set()
    for task in train:
        boxes.add(int(round(task.init.get(env._box, "color"))))
        colors.update(
            int(round(task.init.get(b, "color")))
            for b in env._active_balloons(task.init))
    assert colors == set(range(len(env.BALLOON_PALETTE)))
    assert boxes == {0, 1}
    test = env.get_test_tasks()[0].init
    assert len(env._active_balloons(test)) == len(env.BALLOON_PALETTE)
    assert int(round(test.get(env._box, "color"))) in boxes


def test_composition_test_levels_compose_and_defeat_the_naive_order():
    """On every test level every winning in-band subset frees at least two
    balloons whose colours never shared a train rack on the test box, every in-
    band subset bursts when freed weakest first with the box settling between
    releases, the task records the lowest in-band subset as the decoy, and the
    oracle's witnessed order, which is not weakest- first, still wins."""
    _, env = _make_env(num_train_tasks=2, num_test_tasks=1)
    # pylint: disable-next=import-outside-toplevel
    from predicators.ground_truth_models.balloons.oracle import solve_level
    racks = env._train_racks()
    assert len(racks) == 2
    task = env.get_test_tasks()[0]
    state = task.init
    box = int(round(state.get(env._box, "color")))
    balloons = env._active_balloons(state)
    colors = [int(round(state.get(b, "color"))) for b in balloons]
    units = env.unit_colors(state)
    assert units == [[c] for c in colors]
    assert not env.is_bundled(state)

    def seen(subset):
        palette = frozenset(colors[i] for i in subset)
        return any(b == box and palette <= set(rack) for b, rack in racks)

    def hover(subset):
        return env.hover_height(box, [colors[i] for i in subset])

    candidates = env.candidate_outcomes(state)
    reference = env.solution_subset(state)
    assert reference is not None
    winners = [s for s, outs in candidates.items() if any(o.won for o in outs)]
    assert reference in winners
    for subset in winners:
        assert len(subset) >= 2
        assert not seen(subset)
    for subset in candidates:
        assert env.naive_release_bursts(state, subset)
    flags = task.offline_task_metrics
    recorded = tuple(i for i, b in enumerate(balloons)
                     if flags[f"decoy_{b.name}"])
    assert recorded == min(candidates, key=hover)
    assert all(flags[f"solution_{b.name}"] == float(i in reference)
               for i, b in enumerate(balloons))
    order = env.reference_order(state, reference)
    assert order != env.weakest_first_order(units, reference)
    plan = solve_level(env, state)
    assert plan is not None
    assert [name for _, name in plan] == [f"clip{i}" for i in order]


def test_a_clip_frees_its_whole_bundle(env_module):
    """On a bundle rack every balloon's ``clip`` feature names its clip, the
    bundle stands in a row behind that clip, one push frees the whole bundle
    and the box hangs at the union's analytic height; the other bundle stays
    clipped."""
    mod, env = env_module
    state = env.level_state(1, [3, 0, 1, 2], (0.6, 0.65), clip_of=[0, 0, 1, 1])
    assert env.is_bundled(state)
    assert env.bundles(state) == [[0, 1], [2, 3]]
    assert env.unit_colors(state) == [[3, 0], [1, 2]]
    assert len(env._active_clips(state)) == 2
    balloons = env._balloons
    assert state.get(balloons[0], "x") == state.get(balloons[1], "x")
    assert state.get(balloons[1], "y") - state.get(balloons[0], "y") == \
        pytest.approx(env.bundle_y_gap)
    assert state.get(balloons[0], "x") == state.get(env._clips[0], "x")
    obs = _open_clips(env, state, [0])
    assert obs.get(balloons[0], "tied") == 1.0
    assert obs.get(balloons[1], "tied") == 1.0
    assert obs.get(balloons[2], "tied") == 0.0
    assert obs.get(balloons[3], "tied") == 0.0
    expected = env.hover_height(1, [3, 0])
    assert expected is not None
    assert abs(obs.get(env._box, "z") - expected) < 0.03
    assert mod.box_at_rest(obs, env._box)
    # A freed bundle hangs side by side at one tier, spread along y and
    # centred on the box; unequal lifts tilt the welded assembly, which
    # shifts the pair a little in y and z but keeps its order and spacing.
    box_y = obs.get(env._box, "y")
    y0, y1 = obs.get(balloons[0], "y"), obs.get(balloons[1], "y")
    assert y0 < box_y < y1
    assert y1 - y0 == pytest.approx(env.cluster_gap, abs=0.015)
    assert obs.get(balloons[0],
                   "z") == pytest.approx(obs.get(balloons[1], "z"), abs=0.03)
    # The second bundle would hang one tier higher, so the ceiling is met
    # at the height of a two-tier stack.
    assert env.burst_height(2) < env.burst_height(1)
    # The oracle's helpers are over clips, so a bundle needs no extra
    # bookkeeping: clip0 holds balloon0 and balloon1.
    # pylint: disable-next=import-outside-toplevel
    from predicators.ground_truth_models import get_gt_helper_predicates
    helpers = {
        p_.name: p_
        for p_ in get_gt_helper_predicates("pybullet_balloons")
    }
    holds = helpers["Holds"]
    assert GroundAtom(holds, [env._clips[0], balloons[1]]).holds(state)
    assert not GroundAtom(holds, [env._clips[1], balloons[1]]).holds(state)
    assert GroundAtom(holds, [env._clips[1], balloons[3]]).holds(state)


def test_bundle_test_levels_defeat_every_reading_of_the_rest_heights():
    """With bundle sizes configured the test rack ties its balloons into
    bundles and the generated level meets the module doc's conditions: the
    reference wins with two or more settled cuts and does not start with the
    weakest bundle, every winning union is unseen, the decoy bundle rests in
    the band by the analytic law and bursts when cut first, every winning order
    starts with the same clip, the tempting bundle below the band loses, the
    metrics and goal text describe the bundles, and the oracle's plan waits
    between cuts."""
    _, env = _make_env(num_train_tasks=2,
                       num_test_tasks=1,
                       balloons_test_bundle_sizes=[2, 2, 2, 2],
                       balloons_max_sampling_attempts=60)
    # pylint: disable-next=import-outside-toplevel
    from predicators.ground_truth_models.balloons.oracle import solve_level
    for task in env.get_train_tasks():
        assert not env.is_bundled(task.init)
    task = env.get_test_tasks()[0]
    state = task.init
    assert env.is_bundled(state)
    assert sorted(len(b) for b in env.bundles(state)) == [2, 2, 2, 2]
    assert "bundle" in task.goal_nl and "clip feature" in task.goal_nl
    units = env.unit_colors(state)
    box = int(round(state.get(env._box, "color")))
    lo, hi = state.get(env._band, "lo"), state.get(env._band, "hi")
    candidates = env.candidate_outcomes(state)
    verdict = env.bundle_level(state, candidates, env._train_racks())
    assert verdict is not None
    order, decoy, tempting = verdict
    flags = task.offline_task_metrics
    assert flags["bundle_level"] == 1.0
    assert flags["bundle_decoy_clip"] == float(decoy)
    assert flags["bundle_tempting_clip"] == float(tempting)
    assert flags["bundle_reference_cuts"] == float(len(order)) >= 2.0
    assert flags["bundle_reference_first_clip"] == float(order[0])
    reference = env.solution_subset(state)
    assert reference is not None and sorted(order) == sorted(reference)
    for clip in range(len(units)):
        assert flags[f"solution_clip{clip}"] == float(clip in reference)
    for i, balloon in enumerate(env._active_balloons(state)):
        clip = env.clip_index(state, balloon)
        assert flags[f"solution_{balloon.name}"] == float(clip in reference)
        assert flags[f"decoy_{balloon.name}"] == float(clip == decoy)
        assert i in env.bundles(state)[clip]
    weakest = min(range(len(units)), key=lambda i: env.unit_lift(units[i]))
    assert order[0] != weakest
    first_rest = env.hover_height(box, units[order[0]])
    assert first_rest is not None and first_rest < lo
    decoy_rest = env.hover_height(box, units[decoy])
    assert decoy_rest is not None and lo <= decoy_rest <= hi
    assert any(o.burst for o in candidates[(decoy, )])
    if tempting >= 0:
        tempting_rest = env.hover_height(box, units[tempting])
        assert tempting_rest is not None and first_rest < tempting_rest < lo
        assert any(tempting in s and len(s) >= 2 for s in candidates)
    winners = {
        subset:
        [list(o) for o, out in zip(permutations(subset), outs) if out.won]
        for subset, outs in candidates.items()
    }
    winners = {s: orders for s, orders in winners.items() if orders}
    assert tuple(sorted(order)) in winners
    assert {o[0] for orders in winners.values() for o in orders} == {order[0]}
    assert not any(
        env._rack_seen(s, units, box, env._train_racks()) for s in winners)
    plan = solve_level(env, state)
    assert plan is not None
    assert [kind for kind, _ in plan
            ] == ["Release", "Wait"] * (len(order) - 1) + ["Release"]
    assert [name for kind, name in plan
            if kind == "Release"] == [f"clip{i}" for i in order]
