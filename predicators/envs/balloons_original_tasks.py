"""Historical non-hatch task sampling for comparison with the original
baseline.

This preserves the sampling and acceptance rules used at 92d37ec9723b.
Its simultaneous-release screening and probe controller are historical
selection criteria, not certificates of every public Release sequence.
Actual agent actions, physics, observations and evaluator are unchanged.
Keep this module outside the visible simulator supplied to agents.
"""
# pylint: disable=protected-access
from typing import Any, List, Optional, Set, Tuple

import numpy as np

from predicators import utils
from predicators.envs.pybullet_balloons import BalloonsEvaluator, \
    PyBulletBalloonsEnv, any_popped, box_at_rest, box_in_band
from predicators.settings import CFG
from predicators.structs import Action, EnvironmentTask, GroundAtom, State

Plan = List[Tuple[str, str]]


def solution_subset(env: PyBulletBalloonsEnv,
                    state: State) -> Optional[Tuple[int, ...]]:
    """Return the original screen's unique accepted subset, if one exists.

    This historical reference need not be the only subset that works
    under actual Release actions.
    """
    box_color = int(round(state.get(env._box, "color")))
    balloons = env._active_balloons(state)
    colors = [int(round(state.get(b, "color"))) for b in balloons]
    lo = float(state.get(env._band, "lo"))
    hi = float(state.get(env._band, "hi"))
    key = (box_color, tuple(colors), round(lo, 4), round(hi, 4))
    if key in env._original_solution_cache:
        return env._original_solution_cache[key]
    # Reconstruct the level from a clean initial state so the answer is the
    # level's, not a function of how far execution has progressed.
    clean = env.level_state(box_color, colors, (lo, hi))
    in_band_eq = [
        subset for subset, z in env.lifting_subsets(box_color, colors)
        if lo <= z <= hi
    ]
    winners = [
        subset for subset in in_band_eq
        if subset_outcome(env, clean, subset)[0]
    ]
    result = winners[0] if len(winners) == 1 else None
    env._original_solution_cache[key] = result
    return result


def subset_outcome(env: PyBulletBalloonsEnv, state: State,
                   subset: Tuple[int, ...]) -> Tuple[bool, bool]:
    """Free ``subset``'s balloons from ``state`` on this instance and roll the
    sim to rest; ``(settles_in_band, burst)``.

    ``settles_in_band`` is True when the box hangs at rest with its
    centre in the band; ``burst`` is True when a balloon reached the
    ceiling on the way up (an overshoot the analytic equilibrium does
    not reveal).
    """
    env._pybullet_robot.set_joints(env._pybullet_robot.initial_joint_positions)
    env._set_state(state)
    s = env._get_state().copy()
    for i in subset:
        s.set(env._clips[i], "is_on", 1.0)
    action = env._hold_action()
    moved = False
    for _ in range(int(CFG.balloons_probe_max_steps)):
        s = env.simulate(s, action)
        if any_popped(s) is not None:
            return (False, True)
        resting = box_at_rest(s, env._box)
        moved = moved or not resting
        if moved and resting:
            break
    return (box_in_band(s, env._box, env._band)
            and box_at_rest(s, env._box), False)


def run_option(env: PyBulletBalloonsEnv, state: State, option: Any,
               max_steps: int) -> Optional[State]:
    """Execute a grounded option from ``state`` on this instance, then hold
    until the box has settled; None if it cannot run."""
    env._pybullet_robot.set_joints(env._pybullet_robot.initial_joint_positions)
    env._set_state(state)
    obs = env._get_state()
    env._current_observation = obs
    if not option.initiable(obs):
        return None
    try:
        for _ in range(max_steps):
            if option.terminal(obs):
                break
            obs = env._step_once(option.policy(obs))
    except utils.OptionExecutionFailure:
        return None
    hold = Action(np.array(env._pybullet_robot.get_joints(), dtype=np.float32))
    moved = False
    for i in range(max_steps):
        obs = env._step_once(hold)
        moving = env._speed(env._box) > CFG.balloons_settle_speed
        moved |= moving
        if moved and not moving:
            break
        if not moved and i > 20:
            break
    return env._get_state()


def make_tasks(env: PyBulletBalloonsEnv, num_tasks: int,
               rng: np.random.Generator, train: bool) -> List[EnvironmentTask]:
    """Sample levels with the original baseline's acceptance criteria."""
    counts = list(CFG.balloons_num_balloons_train if train else CFG.
                  balloons_num_balloons_test)
    box_colors = list(CFG.balloons_box_colors_train if train else CFG.
                      balloons_box_colors_test)
    half = float(CFG.balloons_band_half)
    # Train levels together show every balloon colour and every box
    # material the split allows, so a test level composes lifts and
    # a mass the agent has seen rather than ones it must guess.
    palette = list(range(len(env.BALLOON_PALETTE)))
    seen_colors: Set[int] = set()
    seen_boxes: Set[int] = set()
    attempts = int(CFG.balloons_max_sampling_attempts)
    tasks = []
    for _ in range(num_tasks):
        found = None
        for attempt in range(attempts):
            n = int(rng.choice(counts))
            # The covering draw first; a free draw for the last
            # quarter of the attempts, so a rack the filters below
            # keep rejecting does not sink the level.
            if train and attempt < 3 * attempts // 4:
                box_color, colors = env._draw_covering(rng, n, box_colors,
                                                       palette, seen_boxes,
                                                       seen_colors)
            else:
                box_color = int(rng.choice(box_colors))
                colors = [
                    int(c) for c in rng.choice(palette, size=n, replace=False)
                ]
            # Reachable equilibria. The transient (not a static stack-
            # clearance) decides whether a subset overshoots into the
            # ceiling, so each equilibrium-in-band candidate is rolled
            # forward below.
            # Bands must sit inside the chute (below its top) so the box
            # is gated by the walls through its whole ascent and settles
            # between them.
            reach_max = min(env.ceiling_z - env.ceiling_half_extents[2],
                            env.chute_z_hi) - 0.06
            reachable = [
                (subset, z)
                for subset, z in env.lifting_subsets(box_color, colors)
                if env.table_height + 0.12 <= z <= reach_max
            ]
            # Candidate band centres. A test level must hide the answer
            # from a reader that only computes the equilibrium HEIGHT: two
            # subsets share one band by the analytic (equilibrium) law
            # while only one actually settles there - the other overshoots
            # into the ceiling and bursts, or (unbalanced) tilts and jams
            # in the chute. The band is 2*half wide, so two equilibria up
            # to 2*half apart share it only when the band sits between
            # them: centre on each close PAIR's midpoint. Train levels keep
            # a single answer, so also allow a band centred on one subset.
            centers = [(reachable[a][1] + reachable[b][1]) / 2.0
                       for a in range(len(reachable))
                       for b in range(a + 1, len(reachable))
                       if abs(reachable[a][1] - reachable[b][1]) <= 2 * half]
            if train:
                centers += [z for _, z in reachable]
            if not train and CFG.balloons_require_jam_decoy:
                # Contact-only test levels centre the band on a subset's own
                # equilibrium, so a jamming subset can sit at the band's
                # centre and the height reader is drawn to it.
                centers += [z for _, z in reachable]
            rng.shuffle(centers)
            for center_z in centers:
                band = (center_z - half, center_z + half)
                in_band = [
                    subset for subset, z in reachable
                    if band[0] <= z <= band[1]
                ]
                if not train and len(in_band) < 2:
                    continue
                state = env.level_state(box_color, colors, band)
                outcomes = {
                    subset: subset_outcome(env, state, subset)
                    for subset in in_band
                }
                safe = [s for s, (settled, _) in outcomes.items() if settled]
                if len(safe) != 1:
                    continue
                # The test decoy: another subset whose equilibrium is in
                # the band but that fails in reality (bursts or jams), so a
                # height-only reader has a wrong answer to fall for. That
                # is exactly the in-band-by-eq subsets that are not the
                # unique safe one, which len(in_band) >= 2 guarantees.
                if not train and CFG.balloons_require_jam_decoy:
                    # Contact-only discrimination: the in-band subset
                    # NEAREST the band centre must fail by JAM (the tilted
                    # box wedges: settle=False, burst=False), not settle and
                    # not burst. Then a reader that picks by rest height is
                    # drawn to the jammer and loses, while the unique safe
                    # subset sits off-centre (but in-band) and is found only
                    # by a contact rollout. Its equilibrium must stay within
                    # tol of the central jammer's so height gives no signal
                    # pointing back to it.
                    eqz = dict(reachable)
                    safe_eq = eqz[safe[0]]
                    tol = float(CFG.balloons_contact_height_tol)
                    dist_to_centre = {
                        s: abs(eqz[s] - center_z)
                        for s in in_band
                    }
                    central = min(dist_to_centre,
                                  key=dist_to_centre.__getitem__)
                    settled_c, burst_c = outcomes[central]
                    if settled_c or burst_c:
                        # Central subset settles (height would pick the safe
                        # one) or bursts (height, not contact, separates).
                        continue
                    if abs(eqz[central] - safe_eq) > tol:
                        continue
                if solve_level(env, state) is None:
                    continue
                found = (state, safe[0])
                break
            if found is not None:
                break
        if found is None:
            raise RuntimeError(
                "No balloon level whose unique overshoot-safe subset the "
                f"oracle clears in {attempts} draws.")
        state, subset = found
        seen_boxes.add(box_color)
        seen_colors.update(colors)
        goal = {GroundAtom(env._InBand, [env._box, env._band])}
        balloons = env._active_balloons(state)
        names = ", ".join(f"{env.balloon_color_name(state.get(b, 'color'))} "
                          f"({b.name}, clip{i})"
                          for i, b in enumerate(balloons))
        goal_nl = (
            f"Open clips to free balloons so that the "
            f"{env.box_color_name(box_color)} box floats up and hangs "
            f"still with its centre inside the green band "
            f"({state.get(env._band, 'lo'):.2f} to "
            f"{state.get(env._band, 'hi'):.2f} m). Each balloon is "
            f"held by the clip in front of it: {names}. A balloon that "
            f"reaches the ceiling bursts and the level is lost; a freed "
            f"balloon cannot be clipped back.")
        metrics = {
            f"solution_{b.name}": float(i in subset)
            for i, b in enumerate(balloons)
        }
        metrics["task_generation_version"] = 1.0
        tasks.append(
            EnvironmentTask(state,
                            goal,
                            goal_nl=goal_nl,
                            evaluator=BalloonsEvaluator(goal),
                            offline_task_metrics=metrics))
    return env._add_pybullet_state_to_tasks(tasks)


def solve_level(env: PyBulletBalloonsEnv, state: State) -> Optional[Plan]:
    """Run the oracle's plan from ``state`` on ``env``; the plan if the box
    ends inside the band with nothing burst, else None."""
    # pylint: disable=import-outside-toplevel
    from predicators.ground_truth_models.balloons.options import \
        PyBulletBalloonsGroundTruthOptionFactory, release_params
    from predicators.ground_truth_models.balloons.oracle import release_order

    # pylint: enable=import-outside-toplevel
    subset = solution_subset(env, state)
    if subset is None:
        return None
    factory = PyBulletBalloonsGroundTruthOptionFactory
    release = factory.release_option({t.name: t
                                      for t in env.types},
                                     factory.skill_config(None),
                                     plan_transit=False)
    balloons = env._active_balloons(state)  # pylint: disable=protected-access
    clips = env._active_clips(state)  # pylint: disable=protected-access
    plan: Plan = []
    current = state
    for i in release_order(env, state, subset):
        grounded = release.ground(
            [env._robot, clips[i]],  # pylint: disable=protected-access
            release_params())
        nxt = run_option(env, current, grounded,
                         int(CFG.balloons_probe_max_steps))
        if nxt is None or nxt.get(balloons[i], "tied") < 0.5:
            return None
        plan.append(("Release", clips[i].name))
        current = nxt
        if any_popped(current) is not None:
            return None
    if box_in_band(current, env._box, env._band):  # pylint: disable=protected-access
        return plan
    return None
