"""Min-block / system-ID task generation for the domino environment.

Builds the reach-limited "minimum-blocks" tasks: start/target pairs whose
gap sits near the topple-reach limit, each carrying a ``MinBlockReward``
with budget K* (the simulated minimum number of movable blues that topple
the target at the true friction). A differentiation filter keeps only
tasks that separate a friction-calibrated planner from a miscalibrated
one. Finished tasks are cached on disk, keyed by config + seed + a source
digest.

Every function takes the composed domino env as its first argument; this
module owns the generation pipeline while the env owns the physics and
reward semantics (``MinBlockReward``, ``count_movable_blocks_used``).
"""

import functools
import hashlib
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

import numpy as np

from predicators.envs.pybullet_domino.task_generators.min_block_utils import \
    _PROBE_ANCHOR, clear_probe_memo, compute_k_star, compute_turn_k_star, \
    heavy_dogleg_k_star, straight_span_k_star
from predicators.settings import CFG
from predicators.structs import EnvironmentTask, GroundAtom, Object, State

if TYPE_CHECKING:
    from predicators.envs.pybullet_domino.env import PyBulletDominoComposedEnv
    # pylint: disable-next=line-too-long
    from predicators.envs.pybullet_domino.task_generators.domino_task_generator import \
        DominoTaskGenerator


@functools.lru_cache(maxsize=1)
def _domino_code_digest() -> str:
    """Digest of the source code that determines min-block task generation.

    Covers the domino env package (task generators, K* search, the env)
    and the domino skills (push geometry feeds the simulated K*). Any
    edit to these files changes the digest and invalidates cached tasks.
    """
    import predicators  # pylint: disable=import-outside-toplevel
    base = Path(predicators.__file__).parent
    digest = hashlib.sha256()
    for rel in ("envs/pybullet_domino", "ground_truth_models/domino",
                "ground_truth_models/skill_factories"):
        for source in sorted((base / rel).rglob("*.py")):
            digest.update(source.name.encode())
            digest.update(source.read_bytes())
    return digest.hexdigest()


def make_min_block_tasks(env: "PyBulletDominoComposedEnv",
                         generator: "DominoTaskGenerator",
                         generate_batch: Callable[[int],
                                                  List[EnvironmentTask]],
                         num_tasks: int, rng: np.random.Generator,
                         cache_tag: str) -> List[EnvironmentTask]:
    """Generate (or reload) the min-block task set.

    Min-block generation is expensive (each kept task runs simulated K*
    searches), but fully deterministic given the seed, the config, and
    the code — so finished tasks are cached and reloaded on repeat runs.

    A fraction (``domino_min_block_turn_ratio``) of tasks are L-shaped
    (one 90-degree domino turn), the rest straight. Both drop tasks that
    can't be pushed / don't topple, so the quota loop keeps going until
    enough of each survive (or the attempt cap is hit). ``rng`` is
    stateful, so each attempt yields fresh tasks.

    In heavy-block mode (``domino_heavy_block_tasks``) every task instead
    comes from ``_make_heavy_block_task`` (an immovable gray block posing
    as a ready-made bend link on the start's fall line); the quota loop,
    cache, and reward machinery are shared.
    """
    cache_path = _min_block_cache_path(env, cache_tag, num_tasks)
    cached = _load_min_block_cache(env, cache_path)
    if cached is not None:
        return cached

    clear_probe_memo()
    turn_maker: Callable[[
        "PyBulletDominoComposedEnv", "DominoTaskGenerator", np.random.Generator
    ], Optional[EnvironmentTask]]
    if CFG.domino_heavy_block_tasks:
        n_turn, n_straight = num_tasks, 0
        turn_maker = _make_heavy_block_task
    else:
        n_turn = int(round(num_tasks * CFG.domino_min_block_turn_ratio))
        n_straight = num_tasks - n_turn
        turn_maker = _make_turn_task
    turns: List[EnvironmentTask] = []
    straights: List[EnvironmentTask] = []
    # The differentiating span/leg windows are narrow relative to the
    # sampling bands (~25% of straight attempts survive the filters),
    # so give the quota loop generous headroom; results are cached.
    max_attempts = 12 * num_tasks + 20
    for _ in range(max_attempts):
        if len(turns) >= n_turn and len(straights) >= n_straight:
            break
        if len(turns) < n_turn:
            turn_task = turn_maker(env, generator, rng)
            if turn_task is not None:
                turns.extend(
                    env._add_pybullet_state_to_tasks(  # pylint: disable=protected-access
                        [turn_task]))
        if len(straights) < n_straight:
            batch = env._add_pybullet_state_to_tasks(  # pylint: disable=protected-access
                generate_batch(1))
            straights.extend(_assign_min_blocks(env, batch))
    survivors = turns[:n_turn] + straights[:n_straight]
    if len(survivors) < num_tasks:
        logging.warning(
            "Min-block: generated only %d/%d tasks (%d turn, %d straight) "
            "after %d attempts; widen the span/gap bands or raise "
            "domino_min_block_num_blues.", len(survivors), num_tasks,
            len(turns), len(straights), max_attempts)
    _save_min_block_cache(cache_path, survivors, num_tasks)
    return survivors


# ── Min-block task cache ─────────────────────────────────────

# Flags matched by the cache key's prefixes that only affect RENDERING —
# they cannot change generation physics or the tasks themselves, so they
# are excluded from the key (a camera-resolution change must not orphan
# a 20-minute generation cache).
_RENDER_ONLY_FLAGS = frozenset({
    "pybullet_camera_width",
    "pybullet_camera_height",
    "pybullet_draw_debug",
})


def _min_block_cache_path(env: "PyBulletDominoComposedEnv", cache_tag: str,
                          num_tasks: int) -> Optional[Path]:
    """Cache file for this (config, seed, code) combination, or None.

    The key hashes every ``domino_``/``pybullet_``/``skill_phase_``
    CFG flag (except the render-only ones above), the seed and task
    counts, AND a digest of the domino env + domino skill source code —
    so any change to the physics config or the generation/skill code
    automatically invalidates the cache.
    """
    cache_dir = CFG.domino_min_block_task_cache_dir
    if not cache_dir or not cache_tag:
        return None
    cfg_items = {}
    for name in dir(CFG):
        if name.startswith(("domino_", "pybullet_", "skill_phase_")) \
                and name not in _RENDER_ONLY_FLAGS:
            value = getattr(CFG, name)
            if not callable(value):
                cfg_items[name] = value
    blob = json.dumps([
        env.get_name(), cache_tag, num_tasks, CFG.seed, cfg_items,
        _domino_code_digest()
    ],
                      sort_keys=True,
                      default=str)
    key = hashlib.sha256(blob.encode()).hexdigest()[:16]
    return Path(cache_dir) / f"{cache_tag}_{key}.json"


def _load_min_block_cache(
        env: "PyBulletDominoComposedEnv",
        path: Optional[Path]) -> Optional[List[EnvironmentTask]]:
    """Rebuild cached tasks, or None on a cache miss."""
    # pylint: disable=import-outside-toplevel,protected-access
    from predicators.envs.pybullet_domino.env import MinBlockReward
    if path is None or not path.exists():
        return None
    raw = json.loads(path.read_text())
    if isinstance(raw, dict):
        num_requested = raw["num_requested"]
        entries = raw["tasks"]
    else:  # legacy format: bare task list, request size unknown
        num_requested, entries = None, raw
    pred_map = {p.name: p for p in env.predicates}
    # Map names to the env's LIVE object instances: they carry the
    # PyBullet body ids that state I/O needs (fresh Object()s would
    # compare equal but have id=None).
    live_objs = {env._robot.name: env._robot}
    for comp in env._components:
        for obj in comp.get_objects():
            live_objs[obj.name] = obj
    tasks: List[EnvironmentTask] = []
    for entry in entries:
        objs = {name: live_objs[name] for name, _tname in entry["objects"]}
        state = State({
            objs[name]: np.array(vals, dtype=np.float64)
            for name, vals in entry["data"].items()
        })
        goal = {
            GroundAtom(pred_map[pname], [objs[oname] for oname in onames])
            for pname, onames in entry["goal"]
        }
        max_blocks = entry["max_blocks"]
        plain = EnvironmentTask(
            state,
            goal,
            goal_nl=entry["goal_nl"],
            reward_fn=(MinBlockReward(env, goal, max_blocks)
                       if max_blocks is not None else None))
        # Re-run the standard PyBullet conversion (joints, optional
        # rendering) instead of caching simulator state.
        tasks.extend(env._add_pybullet_state_to_tasks([plain]))
    if num_requested is not None and len(tasks) < num_requested:
        logging.warning(
            "Min-block: cache %s holds a PARTIAL set (%d/%d tasks — the "
            "generating run hit its attempt cap). Runs will evaluate on "
            "the reduced set; delete the file to retry generation, or "
            "widen the span/gap bands.", path, len(tasks), num_requested)
    logging.info("Min-block: loaded %d cached tasks from %s.", len(tasks),
                 path)
    return tasks


def _save_min_block_cache(path: Optional[Path], tasks: List[EnvironmentTask],
                          num_requested: int) -> None:
    """Serialize finished tasks (init data, goal, K*) to the cache.

    The reward function itself isn't serializable; its K* budget is
    stored and the ``MinBlockReward`` is rebuilt on load.
    ``num_requested`` is stored alongside so a partial set (the quota
    loop hit its attempt cap) is flagged loudly on every reload instead
    of silently shrinking the eval.
    """
    # pylint: disable=import-outside-toplevel
    from predicators.envs.pybullet_domino.env import MinBlockReward
    if path is None:
        return
    payload = []
    for env_task in tasks:
        init = env_task.init
        reward = env_task.reward_fn
        payload.append({
            "objects": [(o.name, o.type.name) for o in init],
            "data": {o.name: [float(v) for v in init.data[o]]
                     for o in init},
            "goal": [(a.predicate.name, [o.name for o in a.objects])
                     for a in env_task.goal],
            "goal_nl":
            env_task.goal_nl,
            "max_blocks":
            reward.max_blocks if isinstance(reward, MinBlockReward) else None,
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({
            "num_requested": num_requested,
            "tasks": payload
        }))
    logging.info("Min-block: cached %d tasks at %s.", len(tasks), path)


# ── Straight tasks: K* assignment + differentiation filter ──────


def _assign_min_blocks(env: "PyBulletDominoComposedEnv",
                       tasks: List[EnvironmentTask]) -> List[EnvironmentTask]:
    """Attach each min-block task's ``MinBlockReward`` (budget = K*,
    computed by simulation).

    K* is the minimum number of blues whose evenly-spaced chain lets a real
    Push on the start topple the target at this env's true friction (see
    ``min_block_utils.compute_k_star``). Tasks whose target is unreachable
    within the blue budget, or that need no blues at all (K*=0, trivially
    solved by a direct push), are dropped — neither exercises the reach
    model. Computing K* pushes/steps the sim; that's fine here because it
    runs before any episode and every episode re-sets state.
    """
    # pylint: disable=import-outside-toplevel
    from predicators.envs.pybullet_domino.env import MinBlockReward
    out: List[EnvironmentTask] = []
    for env_task in tasks:
        k_star = compute_k_star(env, env_task.init)
        if k_star is None or k_star < 1:
            logging.warning(
                "Dropping min-block task (K*=%s): target unreachable "
                "within budget or solvable with a direct push.", k_star)
            continue
        if k_star >= CFG.domino_min_block_num_blues:
            logging.warning(
                "Dropping min-block task (K*=%d == staged blues %d): no "
                "spare blue for the over-build check.", k_star,
                CFG.domino_min_block_num_blues)
            continue
        # Differentiation filter (only when a planning-friction mismatch
        # is configured), direction-aware:
        #   * planning > true (over-reach): keep only believed < true —
        #     an uncalibrated minimal planner structurally UNDER-builds
        #     (chain dies, target never topples);
        #   * planning < true (under-reach): keep only believed > true
        #     (and expressible within the staged blues) — the planner
        #     OVER-builds, topples the target, but exceeds max_blocks.
        # Dead-band spans where both frictions agree cannot separate the
        # calibrated from the uncalibrated model and are dropped.
        direction = _planning_mismatch_direction()
        if direction is not None:
            k_believed = _planning_k_star(env, env_task.init)
            if not _believed_k_differentiates(direction, k_star, k_believed):
                logging.warning(
                    "Dropping min-block task (true K*=%d, planner "
                    "believes %s, %s): does not differentiate calibrated "
                    "vs uncalibrated reach.", k_star, k_believed, direction)
                continue
        out.append(
            EnvironmentTask(env_task.init_obs,
                            env_task.goal_description,
                            alt_goal_desc=env_task.alt_goal_desc,
                            goal_nl=env_task.goal_nl,
                            reward_fn=MinBlockReward(env, env_task.goal,
                                                     k_star)))
    logging.info("Min-block tasks: kept %d/%d with K* assigned.", len(out),
                 len(tasks))
    return out


def _planning_mismatch_direction() -> Optional[str]:
    """Direction of the configured planning-friction mismatch.

    ``"over_reach"`` when the planning sim's friction is HIGHER than
    the true friction (planner over-estimates topple reach -> under-
    builds), ``"under_reach"`` when lower (planner over-builds),
    ``None`` when no mismatch is configured (differentiation filters
    disabled).
    """
    planning = CFG.domino_planning_friction
    if planning is None or abs(planning - CFG.domino_true_friction) < 1e-9:
        return None
    return ("over_reach"
            if planning > CFG.domino_true_friction else "under_reach")


def _believed_k_differentiates(direction: str, k_true: int,
                               k_believed: Optional[int]) -> bool:
    """Whether a task with these K*s forces the uncalibrated planner to
    fail.

    * over_reach: the planner must believe FEWER blues suffice (it then
      under-builds and the chain dies short of the target);
    * under_reach: the planner must believe MORE blues are needed — but
      no more than the staged budget, so its over-built plan is
      physically expressible and fails on the ``max_blocks`` cap rather
      than on a muddled "can't build my plan" path.
    """
    if k_believed is None:
        return False
    if direction == "over_reach":
        return k_believed < k_true
    return k_true < k_believed <= CFG.domino_min_block_num_blues


def _planning_k_star(env: "PyBulletDominoComposedEnv",
                     init_state: State) -> Optional[int]:
    """K* as the (miscalibrated) planning sim would compute it.

    Temporarily switches the env's domino friction to
    ``CFG.domino_planning_friction``, computes K*, then restores the true
    friction. Returns ``None`` when no planning-friction mismatch is
    configured, which disables the caller's differentiation filter.
    """
    planning = CFG.domino_planning_friction
    if planning is None or abs(planning - CFG.domino_true_friction) < 1e-9:
        return None
    env.set_domino_physical_params(friction=planning)
    try:
        return compute_k_star(env, init_state)
    finally:
        env.set_domino_physical_params(friction=CFG.domino_true_friction)


# ── Turn tasks ───────────────────────────────────────────────


def _make_turn_task(env: "PyBulletDominoComposedEnv",
                    gen: "DominoTaskGenerator",
                    rng: np.random.Generator) -> Optional[EnvironmentTask]:
    """Build one L-shaped (90-degree turn) min-block task, or None.

    Pipeline:

    1. Sample the geometry directly: start pose in the pushable band and
       a target one 90-degree turn away, at leg lengths drawn from the
       empirically differentiating region (long entry legs).
    2. Cheap pre-filters on memoized straight-leg probes: a feasibility
       bound (the legs' own straight-chain minima already exceed the
       staged budget) and the per-LEG differentiation certificate (see
       inline comment) — full believed corner plans rarely validate at
       the planning friction, so differentiation is certified on the
       straight legs instead. These run BEFORE the layout search below,
       which costs dozens of Push rollouts per attempt and dominated
       generation time when it ran on every attempt.
    3. K* = ``compute_turn_k_star`` at the true friction — the minimum
       blues over a layout SEARCH of agent-buildable candidates
       (straight-line probes + natural-yaw corners), because around a
       corner an evenly-spaced chain is not minimal: sliding the corner
       toward the start can save a block. The winning layout doubles as
       the proof the task is solvable.
    4. Stage ``domino_min_block_num_blues`` blues (more than K*), so
       over-building is possible and penalized by the ``max_blocks`` cap.
    """
    # pylint: disable=import-outside-toplevel
    from predicators.envs.pybullet_domino.env import MinBlockReward
    from predicators.utils import create_state_from_dict
    comp = env._domino_component  # pylint: disable=protected-access
    if comp is None:
        return None
    # Synthesize the start/target geometry directly from sampled leg
    # lengths (the K* search below is itself the constructive proof of
    # solvability, so no chain needs to be pre-built). Leg ranges follow
    # the empirical leg-length scan: long ENTRY legs are what make a turn
    # task differentiate calibrated vs uncalibrated reach (short-legged
    # corners cost the same block count at both frictions).
    # Leg ranges target the scan's K*=3-vs-believed-2 region; longer legs
    # give K*>=4, which the staging capacity (num_blues=4, so K*<=3 for a
    # spare blue) cannot host.
    entry_leg = float(rng.uniform(0.26, 0.34))
    exit_leg = float(rng.uniform(0.18, 0.26))
    sx = float(rng.uniform(comp.domino_x_lb, comp.domino_x_ub))
    sy = float(rng.uniform(comp.domino_y_lb, comp.domino_y_ub))
    syaw = float(rng.choice([0.0, np.pi / 2, -np.pi / 2]))
    side = float(rng.choice([-1.0, 1.0]))
    u_vec = np.array([np.sin(syaw), np.cos(syaw)])
    p_vec = side * np.array([-u_vec[1], u_vec[0]])  # turn side
    t_pt = np.array([sx, sy]) + entry_leg * u_vec + exit_leg * p_vec
    if not (env.x_lb < t_pt[0] < env.x_ub and env.y_lb < t_pt[1] < env.y_ub):
        return None
    tyaw = float(np.arctan2(p_vec[0], p_vec[1]))  # faces exit direction
    start = comp.dominos[0]
    target = comp.dominos[1]
    start_pose = (sx, sy, syaw)
    target_pose = (float(t_pt[0]), float(t_pt[1]), tyaw)
    num_blues = min(CFG.domino_min_block_num_blues, len(comp.dominos) - 2)
    # Cheap memoized leg probes FIRST — the corner layout search below is
    # by far the most expensive step (dozens of real Push rollouts per
    # attempt), so attempts are pre-filtered on straight-leg reach alone.
    legs = (entry_leg, exit_leg)
    true_legs = [
        straight_span_k_star(env, leg, budget=num_blues) for leg in legs
    ]
    true_counts = [v for v in true_legs if v is not None]
    if len(true_counts) < len(legs):
        logging.warning(
            "Dropping turn task (true-friction leg probe failed: legs=%s "
            "-> %s).", legs, true_legs)
        return None
    if sum(true_counts) >= num_blues:
        # Feasibility bound: a turn chain cannot beat its legs' own
        # straight-chain minima (the stretched corner saves at most the
        # corner blue itself), so K* >= sum(true legs) — which already
        # leaves no spare blue. Skip the layout search outright.
        logging.warning(
            "Dropping turn task (legs alone need %s blues >= staged "
            "blues %d).", true_legs, num_blues)
        return None
    direction = _planning_mismatch_direction()
    bel_legs: List[Optional[int]] = []
    if direction is not None:
        # Relaxed per-LEG certificate. The strong "a cheaper believed
        # plan validates in the wrong sim" check is unusable for turns:
        # natural corners barely propagate at the planning friction
        # (high friction grips the base — redirection is what's hard),
        # so a full believed corner plan almost never exists and every
        # turn task would drop. Instead certify reach differentiation
        # where it actually lives — on the straight LEGS: probe how
        # many blues each friction needs for a straight chain of each
        # leg's length (real rollouts). The corner's own cost is the
        # same on both sides of the comparison and cancels.
        env.set_domino_physical_params(friction=CFG.domino_planning_friction)
        try:
            bel_legs = [
                straight_span_k_star(env, leg, budget=num_blues)
                for leg in legs
            ]
        finally:
            env.set_domino_physical_params(friction=CFG.domino_true_friction)
        bel_counts = [v for v in bel_legs if v is not None]
        if len(bel_counts) < len(legs):
            logging.warning(
                "Dropping turn task (planning-friction leg probe failed: "
                "legs=%s true=%s believed=%s).", legs, true_legs, bel_legs)
            return None
        t_sum, b_sum = sum(true_counts), sum(bel_counts)
        differentiates = (b_sum < t_sum
                          if direction == "over_reach" else t_sum < b_sum)
        if not differentiates:
            logging.warning(
                "Dropping turn task (legs true=%s believed=%s, %s): "
                "does not differentiate calibrated vs uncalibrated "
                "reach.", true_legs, bel_legs, direction)
            return None
    k_true = compute_turn_k_star(env,
                                 start_pose,
                                 target_pose,
                                 budget=num_blues)
    if k_true is None or k_true < 1:
        logging.warning("Dropping turn task (searched K*=%s).", k_true)
        return None
    if k_true >= num_blues:
        # No spare blue would remain: the over-build side of the reward
        # would be vacuous (the staging grid caps the scene's blues).
        logging.warning(
            "Dropping turn task (K*=%d == staged blues %d): no spare "
            "blue for the over-build check.", k_true, num_blues)
        return None
    if direction is not None:
        logging.info(
            "Turn task differentiates (%s): true K*=%d, legs true=%s "
            "believed=%s.", direction, k_true, true_legs, bel_legs)
    # Build the scene: start/target fixed at the chain's endpoints,
    # num_blues blues staged (scattered by the staging pass).
    scene: Dict[Object, Dict[str, Any]] = {
        start:
        comp.place_domino(0,
                          start_pose[0],
                          start_pose[1],
                          start_pose[2],
                          is_start_block=True),
        target:
        comp.place_domino(1,
                          target_pose[0],
                          target_pose[1],
                          target_pose[2],
                          is_target_block=True),
    }
    blues = [d for d in comp.dominos if d not in (start, target)]
    for blue in blues[:num_blues]:
        # Initial position is irrelevant — the staging pass re-places
        # every movable (blue) block on the staging grid.
        scene[blue] = comp.place_domino(0, start_pose[0], start_pose[1], 0.0)
    staged = gen._move_intermediate_objects_to_unfinished_state(  # pylint: disable=protected-access
        scene)
    if staged is None:
        return None
    robot_init = {
        "x": env.robot_init_x,
        "y": env.robot_init_y,
        "z": env.robot_init_z,
        "fingers": env.open_fingers,
        "roll": env.robot_init_roll,
        "tilt": env.robot_init_tilt,
        "wrist": env.robot_init_wrist,
    }
    init_dict: Dict[Object, Dict[str, Any]] = {env._robot: robot_init}  # pylint: disable=protected-access
    init_dict.update(staged)
    init_state = create_state_from_dict(init_dict)
    goal_atoms = {GroundAtom(comp.Toppled, [target])}
    goal_nl = (
        "Move the blue dominoes so that when the green domino is pushed, "
        "the purple domino is toppled -- using AS FEW blue dominoes as "
        "possible. The chain may need to make a 90-degree turn. Do NOT "
        "directly push or topple the purple domino yourself.")
    return EnvironmentTask(init_state,
                           goal_atoms,
                           goal_nl=goal_nl,
                           reward_fn=MinBlockReward(env, goal_atoms, k_true))


# ── Heavy-block (immovable obstacle) tasks ───────────────────


def _with_believed_physics(
        env: "PyBulletDominoComposedEnv",
        probe: Callable[[], Optional[int]]) -> Optional[int]:
    """Run ``probe`` under the planner's believed physics.

    Switches the env to the believed physics — the heavy gray block at
    NORMAL domino mass (an ordinary chain link) plus
    ``domino_planning_friction`` when configured — runs the probe, then
    restores the true physics.
    """
    comp = env._domino_component  # pylint: disable=protected-access
    assert comp is not None
    believed: Dict[str, float] = {"heavy_block_mass": comp.domino_mass}
    if CFG.domino_planning_friction is not None:
        believed["friction"] = CFG.domino_planning_friction
    env.set_domino_physical_params(**believed)
    try:
        return probe()
    finally:
        env.set_domino_physical_params(
            friction=CFG.domino_true_friction,
            heavy_block_mass=comp.heavy_block_true_mass)


def _make_heavy_block_task(
        env: "PyBulletDominoComposedEnv", gen: "DominoTaskGenerator",
        rng: np.random.Generator) -> Optional[EnvironmentTask]:
    """Build one heavy-block (immovable obstacle) task, or None.

    Geometry (a dogleg, calibrated by the 2026-07-03 probe sweeps): the
    heavy GRAY block stands directly on the start's fall line, a leg
    ``m`` away, yawed ~30 degrees off it — visually a ready-made bend
    link. The target sits a leg ``n`` beyond the gray along the gray's
    facing. Shallow (~30 degree) bends propagate at BOTH experiment
    frictions, while full 90-degree corners die at the planning friction
    — which is exactly why this shape works: the believed physics
    (normal gray mass, see the env init / ``heavy_block_mass`` override)
    lets a planner run its chain straight into the gray and bend there
    for free; at the true mass the chain dies against the gray, and the
    only real solution is to bend EARLY with an own corner blue and cut
    across to the target.

    Differentiation certificate (all simulated, cheapest first; the
    lure probes run at the canonical anchor so they memoize):

    1. believed dogleg exists: some k_bel blues topple the target at the
       PLANNING physics (normal gray mass + planning friction);
    2. true dogleg is dead: the same family finds NO k at the true
       physics (the chain dies at the gray — structurally guaranteed,
       probed as a sanity check);
    3. the dogleg is the believed-BEST plan: in the believed physics the
       detour costs at least as much (usually it doesn't propagate at
       all — corners barely work at the planning friction), so a
       block-minimizing planner commits to the doomed dogleg. The
       comparison is entirely within the believed physics: how the true
       detour cost relates to k_bel is irrelevant to the lure;
    4. true detour exists: K* = ``compute_turn_k_star`` with the gray
       block in every candidate scene, 1 <= K* <= staged blues. Unlike
       the turn tasks, K* may EQUAL the staged blues: heavy tasks
       differentiate on topple failure, not on the over-build cap, so no
       spare blue is required.

    Budget = K*: the calibrated (mass-aware) planner's detour is exactly
    within budget, while the believing planner never topples the target.
    """
    # pylint: disable=import-outside-toplevel,protected-access
    from predicators.envs.pybullet_domino.env import MinBlockReward
    from predicators.utils import create_state_from_dict
    comp = env._domino_component
    if comp is None:
        return None
    # Sample the dogleg SHAPE: legs (m, n) and a shallow bend. The bands
    # come from the probe sweep — bends past ~35 degrees stop propagating
    # at the planning friction (no lure), and SHORT legs let the true
    # detour (which cuts across the dogleg's elbow) cost the same as the
    # believed dogleg (no differentiation): only long exit legs make the
    # cut pay an extra blue. Samples snap to the probe-memo lattice
    # (1 cm legs, 2-degree bends) so repeated shapes reuse their probes.
    # The shape is certified once at the canonical anchor (pose-invariant
    # physics); its PLACEMENT on the table is retried below.
    m_leg = round(float(rng.uniform(0.20, 0.24)), 2)
    n_leg = round(float(rng.uniform(0.23, 0.27)), 2)
    bend = float(np.radians(rng.choice([26, 28, 30])))
    side = float(rng.choice([-1.0, 1.0]))
    num_blues = min(CFG.domino_min_block_num_blues, len(comp.dominos) - 3)
    # Canonical-anchor copies of the geometry for the memoized lure
    # probes (pose-invariant physics, known-pushable start).
    ax, ay = _PROBE_ANCHOR
    c_syaw = np.pi / 2
    c_hyaw = c_syaw - side * bend
    a_pt = np.array([ax, ay])
    ch_pt = a_pt + m_leg * np.array([1.0, 0.0])
    ct_pt = ch_pt + n_leg * np.array([np.sin(c_hyaw), np.cos(c_hyaw)])
    c_start = (ax, ay, c_syaw)
    c_heavy = (float(ch_pt[0]), float(ch_pt[1]), float(c_hyaw))
    c_target = (float(ct_pt[0]), float(ct_pt[1]), float(c_hyaw))
    # 1) The believed dogleg must exist (otherwise nothing lures the
    # miscalibrated planner onto the gray block).
    k_bel = _with_believed_physics(
        env, lambda: heavy_dogleg_k_star(env, c_start, c_target, c_heavy,
                                         num_blues))
    if k_bel is None:
        logging.warning(
            "Dropping heavy-block task (no believed dogleg within %d "
            "blues).", num_blues)
        return None
    # 2) The true dogleg must be dead (chain dies at the gray).
    k_dead = heavy_dogleg_k_star(env, c_start, c_target, c_heavy, num_blues)
    if k_dead is not None:
        logging.warning(
            "Dropping heavy-block task (true dogleg still topples with "
            "k=%d: gray block not blocking).", k_dead)
        return None
    # 3) The dogleg must be the believed-BEST plan: probe what the
    # detour would cost in the BELIEVED physics (corners barely
    # propagate at the planning friction, so this is usually None). A
    # believed detour cheaper than the believed dogleg would divert the
    # planner off the lure.
    c_gray_scene = {
        comp.dominos[-1]:
        comp.place_domino(0,
                          c_heavy[0],
                          c_heavy[1],
                          c_heavy[2],
                          is_heavy_block=True)
    }
    k_bel_detour = _with_believed_physics(
        env, lambda: compute_turn_k_star(
            env, c_start, c_target, budget=num_blues, extra=c_gray_scene))
    if k_bel_detour is not None and k_bel_detour < k_bel:
        logging.warning(
            "Dropping heavy-block task (believed detour k=%d beats "
            "believed dogleg k=%d): dogleg is no lure.", k_bel_detour, k_bel)
        return None
    # Place the certified shape on the table. Placement can fail for
    # pose-local reasons — bounds, push reachability, staging-grid
    # congestion around the gray block (the grid is a single row, and a
    # fall line running along x parks all three fixed bodies on it) —
    # so it is retried with fresh start poses while the shape
    # certificate above is reused.
    heavy_obj = comp.dominos[-1]
    start = comp.dominos[0]
    target = comp.dominos[1]
    staged = None
    k_true: Optional[int] = None
    extent = m_leg + n_leg * float(np.cos(bend))
    for _ in range(12):
        # The ~0.45 m dogleg only fits along the table's long (x) axis
        # (the y placement band is ~0.19 m), so the start faces +/-x and
        # sx is sampled to leave room for the whole shape.
        syaw = float(rng.choice([np.pi / 2, -np.pi / 2]))
        if syaw > 0:  # falls toward +x
            sx = float(rng.uniform(comp.domino_x_lb, env.x_ub - extent - 0.03))
        else:
            sx = float(rng.uniform(env.x_lb + extent + 0.03, comp.domino_x_ub))
        sy = float(rng.uniform(comp.domino_y_lb, comp.domino_y_ub))
        hyaw = float(syaw - side * bend)
        u_vec = np.array([np.sin(syaw), np.cos(syaw)])
        d_vec = np.array([np.sin(hyaw), np.cos(hyaw)])
        s_pt = np.array([sx, sy])
        h_pt = s_pt + m_leg * u_vec
        t_pt = h_pt + n_leg * d_vec
        if not all(env.x_lb < pt[0] < env.x_ub and env.y_lb < pt[1] < env.y_ub
                   for pt in (h_pt, t_pt)):
            continue
        start_pose = (sx, sy, syaw)
        heavy_pose = (float(h_pt[0]), float(h_pt[1]), hyaw)
        # The target faces the gray's fall direction (the believed
        # arrival); the true detour arrives within ~15 degrees of it,
        # which topples the target just as well.
        target_pose = (float(t_pt[0]), float(t_pt[1]), hyaw)
        gray_scene = {
            heavy_obj:
            comp.place_domino(0,
                              heavy_pose[0],
                              heavy_pose[1],
                              heavy_pose[2],
                              is_heavy_block=True)
        }
        # 4) The detour around the gray block must exist within the
        # staged blues at THIS pose (this doubles as the push-
        # reachability check for the sampled start). Unlike the turn
        # tasks, K* may EQUAL the staged blues: heavy tasks
        # differentiate on topple failure, not on the over-build cap,
        # so no spare blue is required.
        k_true = compute_turn_k_star(env,
                                     start_pose,
                                     target_pose,
                                     budget=num_blues,
                                     extra=gray_scene)
        if k_true is None or k_true < 1:
            continue
        scene: Dict[Object, Dict[str, Any]] = {
            start:
            comp.place_domino(0,
                              start_pose[0],
                              start_pose[1],
                              start_pose[2],
                              is_start_block=True),
            target:
            comp.place_domino(1,
                              target_pose[0],
                              target_pose[1],
                              target_pose[2],
                              is_target_block=True),
        }
        scene.update(gray_scene)
        blues = [
            d for d in comp.dominos if d not in (start, target, heavy_obj)
        ]
        for blue in blues[:num_blues]:
            # Initial position is irrelevant — the staging pass
            # re-places every movable (blue) block on the staging grid
            # (the gray block is exempt and stays on the fall line).
            scene[blue] = comp.place_domino(0, start_pose[0], start_pose[1],
                                            0.0)
        staged = gen._move_intermediate_objects_to_unfinished_state(scene)
        if staged is not None:
            break
    if staged is None:
        logging.warning(
            "Dropping heavy-block task (no placement found for shape "
            "m=%.2f bend=%.0fdeg n=%.2f).", m_leg, np.degrees(bend), n_leg)
        return None
    assert k_true is not None
    logging.info(
        "Heavy-block task differentiates: believed dogleg k=%d (believed "
        "detour %s), true dogleg dead, detour K*=%d.", k_bel, k_bel_detour,
        k_true)
    robot_init = {
        "x": env.robot_init_x,
        "y": env.robot_init_y,
        "z": env.robot_init_z,
        "fingers": env.open_fingers,
        "roll": env.robot_init_roll,
        "tilt": env.robot_init_tilt,
        "wrist": env.robot_init_wrist,
    }
    init_dict: Dict[Object, Dict[str, Any]] = {env._robot: robot_init}
    init_dict.update(staged)
    init_state = create_state_from_dict(init_dict)
    goal_atoms = {GroundAtom(comp.Toppled, [target])}
    goal_nl = (
        "Move the blue dominoes so that when the green domino is pushed, "
        "the purple domino is toppled -- using AS FEW blue dominoes as "
        "possible. Only the blue dominoes may be moved. Do NOT directly "
        "push or topple the purple domino yourself.")
    return EnvironmentTask(init_state,
                           goal_atoms,
                           goal_nl=goal_nl,
                           reward_fn=MinBlockReward(env, goal_atoms, k_true))
