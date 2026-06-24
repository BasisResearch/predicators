"""Locate where a domino cascade dies: run the recorded sketch
(Pick->Place->Push->Wait) through the REAL option model and log every domino's
roll (topple angle) after each step.

Usage: PYTHONPATH=. python scripts/domino_debug/probe_cascade.py <seed> <env_task_idx>
"""
import logging
import sys

import numpy as np

logging.disable(logging.CRITICAL)
from scripts.domino_debug.replay_domino_sketches import _FLAGS  # noqa: E402


def rolls(state, dominoes):
    return "  ".join(f"{d.name}:r={state.get(d,'roll'):+.3f}"
                     for d in dominoes)


def main():
    seed, ti = int(sys.argv[1]), int(sys.argv[2])
    from predicators import utils
    utils.reset_config(dict(_FLAGS, seed=seed))
    from predicators.approaches import create_approach
    from predicators.envs import get_or_create_env
    from predicators.ground_truth_models import get_gt_options
    from predicators.ground_truth_models.domino import processes as P

    env = get_or_create_env("pybullet_domino")
    options = get_gt_options(env.get_name())
    preds, _ = utils.parse_config_excluded_predicates(env)
    train_tasks = [t.task for t in env.get_train_tasks()]
    approach = create_approach("agent_sim_learning", preds, options, env.types,
                               env.action_space, train_tasks)
    approach._maybe_install_oracle_samplers()
    om = approach._option_model
    opt = {o.name: o for o in options}
    InFront = [p for p in preds if p.name == "InFront"][0]
    Toppled = [p for p in preds if p.name == "Toppled"][0]

    task = env.get_test_tasks()[ti].task
    state = task.init
    dominoes = sorted([o for o in state if o.type.name == "domino"],
                      key=lambda o: o.name)
    robot = [o for o in state if o.type.name == "robot"][0]
    fallen = env.fallen_threshold if hasattr(env, "fallen_threshold") else None
    print(f"# seed{seed} env-task{ti}: {len(dominoes)} dominoes, "
          f"fallen_threshold={fallen}")
    for d in dominoes:
        print(f"  {d.name}: x={state.get(d,'x'):.4f} y={state.get(d,'y'):.4f} "
              f"yaw={state.get(d,'yaw'):+.4f}")
    print(f"  goal: {sorted(str(a) for a in task.goal)}")
    print(f"\ninit       {rolls(state, dominoes)}")

    held = dominoes[1]
    ref0, ref1 = dominoes[0], dominoes[3]
    sub = {
        utils.GroundAtom(InFront, [held, ref0]),
        utils.GroundAtom(InFront, [ref1, held])
    }

    # Pick(d1)
    s = om.get_next_state_and_num_actions(
        state,
        opt["Pick"].ground([robot, held],
                           P._pick_option_sampler(state, set(),
                                                  np.random.default_rng(0),
                                                  [robot, held])))[0]
    # Place(d1) at generator-faithful pose for the two InFront subgoals
    pp = P._place_option_sampler(s, sub, np.random.default_rng(3), [robot])
    s = om.get_next_state_and_num_actions(s, opt["Place"].ground([robot],
                                                                 pp))[0]
    print(f"after Place {rolls(s, dominoes)}   "
          f"(d1 placed at {pp[0]:.3f},{pp[1]:.3f},yaw={pp[3]:+.3f})")
    for d in dominoes:
        infr = [str(a.objects) for a in sub if a.holds(s)]
    print("  InFront subgoals holding: " +
          str([str(a) for a in sub if a.holds(s)]))

    # Push
    push = opt["Push"]
    pparams = P._push_option_sampler(s, set(), np.random.default_rng(0),
                                     [robot])
    pg = push.ground([robot], pparams) if len(push.types) == 1 else \
        push.ground([robot, dominoes[0]], pparams)
    s = om.get_next_state_and_num_actions(s, pg)[0]
    print(f"after Push  {rolls(s, dominoes)}")

    # Wait
    wait = opt["Wait"]
    wparams = np.zeros(wait.params_space.shape[0], dtype=np.float32)
    s = om.get_next_state_and_num_actions(s, wait.ground([robot], wparams))[0]
    print(f"after Wait  {rolls(s, dominoes)}")
    print("\nToppled after Wait: " +
          str({d.name: Toppled.holds(s, [d])
               for d in dominoes}))


if __name__ == "__main__":
    main()
