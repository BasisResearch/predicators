"""Execute domino SKILLS on the real-bench scene IN SIMULATION and save an MP4.

This is a sim-only debug harness. It reproduces the stock config
(``predicatorv3/exp_domino_real.yaml``) so the env, bench_setup patches,
geometry and scene match a real run exactly, grounds a hand-specified skill
sketch (Pick / Place / Push / Wait) with the domino oracle samplers, then
rolls it out through the env via ``env.step`` -- the same path
``_run_testing`` uses to render its test videos -- capturing one frame per
low-level action into an MP4.

Use it to watch a skill's arm motion + physics on the real scene and iterate on
skill geometry (grasp height, push pose, ...) without the LLM planner.

Usage (from the predicators repo root, robot-ml env; set PYTHONHASHSEED=0):
    PYTHONPATH=. python scripts/domino_debug/probe_real_scene.py
    # custom sketch + a different scene + live GUI:
    PYTHONPATH=. python scripts/domino_debug/probe_real_scene.py \
        --sketch "Pick:1 Place:1@6 Push:start Wait" --gui

Sketch grammar (space-separated ``Skill[:obj[@ref]]`` tokens):
    Push[:obj]      topple a domino (obj: a domino id / "start" / "target";
                    default "start"). Non-restricted Push grounds [robot, obj].
    Pick:obj        grasp a domino.
    Place:obj[@ref] place the held domino; @ref adds an InFront(obj, ref)
                    subgoal so the placer aims for it (else an empty goal).
    Wait            let the physics settle (the cascade).
Object refs: a domino id from the scene JSON, or "start"/"target" (by role).
"""
import argparse
import json
import logging
import os
from typing import Any, Callable, List, Set, Tuple

import numpy as np

from predicators import utils
from predicators.envs import get_or_create_env
from predicators.ground_truth_models import get_gt_options
from predicators.ground_truth_models.domino import processes as P
from predicators.settings import CFG
from predicators.structs import Object, Predicate, State, _Option
from scripts.cluster_utils import generate_run_configs

# This is a debug harness that deliberately pokes env / component / sampler
# internals to drive skills directly, so protected access is expected.
# pylint: disable=protected-access


def _load_config(config: str, scene: str | None, seed: int) -> None:
    """reset_config from the stock launcher config (single source of truth),
    optionally overriding the scene JSON."""
    rc = list(generate_run_configs(config))[0]
    flags = dict(rc.flags)
    flags.update({"env": rc.env, "approach": rc.approach, "seed": seed})
    flags.pop("log", None)  # launcher arg, not a CFG flag
    if scene is not None:
        flags["domino_real_scene"] = scene
    utils.reset_config(flags)


def _build_resolver(
    env: Any, state: State
) -> Tuple[Callable[[str], Object], Object, Object, List[Object]]:
    """id / 'start' / 'target' -> the predicators domino Object.

    Dominoes are placed in scene order, so scene index i -> object
    ``domino_i``.
    """
    comp = env._domino_component
    with open(CFG.domino_real_scene, encoding="utf-8") as f:
        scene_ids = [d["id"] for d in json.load(f)["dominoes"]]
    id_to_obj = {sid: comp.dominos[i] for i, sid in enumerate(scene_ids)}
    dominoes = sorted([o for o in state if o.type.name == "domino"],
                      key=lambda o: o.name)
    start = next(d for d in dominoes if comp._StartBlock_holds(state, [d]))
    target = next(d for d in dominoes if comp._TargetDomino_holds(state, [d]))

    def resolve(ref: str) -> Object:
        if ref == "start":
            return start
        if ref == "target":
            return target
        return id_to_obj[int(ref)]

    return resolve, start, target, dominoes


def _ground_sketch(sketch: str, env: Any, state: State, robot: Object,
                   resolve: Callable[[str], Object], preds: Set[Predicate],
                   seed: int) -> List[_Option]:
    """Turn a sketch string into a list of ground options, sampling params on
    the init state with the domino oracle samplers."""
    opt = {o.name: o for o in get_gt_options(env.get_name())}
    InFront = next((p for p in preds if p.name == "InFront"), None)
    rng = np.random.default_rng(seed)
    plan: List[_Option] = []
    for tok in sketch.split():
        name, _, rest = tok.partition(":")
        objref, _, ref = rest.partition("@")
        if name == "Pick":
            d = resolve(objref)
            params = P._pick_option_sampler(state, set(), rng, [robot, d])
            plan.append(opt["Pick"].ground([robot, d], params))
        elif name == "Place":
            place_d = resolve(objref) if objref else None
            goal = set()
            if ref and InFront is not None and place_d is not None:
                goal = {utils.GroundAtom(InFront, [place_d, resolve(ref)])}
            params = P._place_option_sampler(state, goal, rng, [robot])
            plan.append(opt["Place"].ground([robot], params))
        elif name == "Push":
            d = resolve(objref) if objref else resolve("start")
            params = P._push_option_sampler(state, set(), rng, [robot])
            push = opt["Push"]
            plan.append(
                push.ground([robot], params) if len(push.types) ==
                1 else push.ground([robot, d], params))
        elif name == "Wait":
            wait = opt["Wait"]
            params = np.zeros(wait.params_space.shape[0], dtype=np.float32)
            plan.append(wait.ground([robot], params))
        else:
            raise ValueError(f"unknown skill in sketch: {name!r}")
    return plan


def main() -> None:
    """Parse args, roll out the sketch on the real scene, and save the MP4."""
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config",
                    default="predicatorv3/exp_domino_real.yaml",
                    help="launcher config to reproduce (env + flags + scene).")
    ap.add_argument("--scene",
                    default=None,
                    help="override CFG.domino_real_scene (a capture JSON).")
    ap.add_argument("--sketch",
                    default="Push:start Wait",
                    help="skill sequence to execute (see module docstring).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gui",
                    action="store_true",
                    help="also open a live PyBullet window while rendering.")
    ap.add_argument("--max-steps",
                    type=int,
                    default=1500,
                    help="max low-level env steps (Wait can be long).")
    ap.add_argument("--frame-stride",
                    type=int,
                    default=2,
                    help="keep every Nth rendered frame in the MP4.")
    ap.add_argument("--out",
                    default=None,
                    help="output mp4 path (default: "
                    "logs/probe_real_scene/<scene>_<sketch>.mp4).")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    _load_config(args.config, args.scene, args.seed)
    if args.gui:
        CFG.use_gui = True

    env = get_or_create_env(CFG.env)
    preds, _ = utils.parse_config_excluded_predicates(env)
    task = env.get_test_tasks()[0].task
    state = task.init
    robot = next(o for o in state if o.type.name == "robot")
    resolve, start, target, dominoes = _build_resolver(env, state)
    Toppled = next(p for p in preds if p.name == "Toppled")

    print(f"# scene   : {os.path.basename(CFG.domino_real_scene)} "
          f"({len(dominoes)} dominoes)")
    print(f"# roles   : start={start.name} target={target.name}")
    print(f"# sketch  : {args.sketch}")
    print(f"# goal    : {sorted(str(a) for a in task.goal)}")

    plan = _ground_sketch(args.sketch, env, state, robot, resolve, preds,
                          args.seed)
    print(f"# grounded: {[o.simple_str() for o in plan]}")

    # Wait terminates on an abstract-atom change (CFG.wait_option_terminate_
    # on_atom_change), so the policy needs an abstract function over the preds.
    policy = utils.option_plan_to_policy(
        plan, abstract_function=lambda s: utils.abstract(s, preds))
    monitor = utils.VideoMonitor(env.render)
    traj, _ = utils.run_policy(
        policy,
        env,
        "test",
        0,
        termination_function=lambda s: False,
        max_num_steps=args.max_steps,
        exceptions_to_break_on={utils.OptionExecutionFailure},
        monitor=monitor)

    final = traj.states[-1]
    toppled = {d.name: bool(Toppled.holds(final, [d])) for d in dominoes}
    print(f"# steps   : {len(traj.actions)}")
    print(f"# toppled : {toppled}")
    print(f"# solved  : {bool(all(a.holds(final) for a in task.goal))}")

    video = monitor.get_video()[::max(1, args.frame_stride)]
    # save_video writes to CFG.video_dir/<out> (default videos/) and only
    # creates video_dir itself, so make the nested subdir first. An absolute
    # --out still works (os.path.join ignores video_dir for an absolute path).
    out = args.out or os.path.join(
        "probe_real_scene",
        f"{os.path.splitext(os.path.basename(CFG.domino_real_scene))[0]}"
        f"__{args.sketch.replace(' ', '_').replace(':', '-')}.mp4")
    resolved = os.path.join(CFG.video_dir, out)
    os.makedirs(os.path.dirname(resolved), exist_ok=True)
    utils.save_video(out, video)
    print(
        f"# saved   : {resolved} ({len(video)} frames @ {CFG.video_fps} fps)")


if __name__ == "__main__":
    main()
