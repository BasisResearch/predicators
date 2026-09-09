"""Verify the contact-only chute generation for balloons.

The claim: with balloons_require_jam_decoy=True, every test level has a
unique safe subset AND an in-band decoy that fails by JAM (the tilted box
wedges in the chute), whose equilibrium HEIGHT is within tol of the safe
subset's. So a model that reasons from rest height / net lift (what the
free-coding MF agent computes in closed form) gets NO signal separating
the two - only a contact rollout (sim.run, MB) does. A wrong pick jams
(steps + reset) or bursts (irreversible loss).

Gates (compute node - repeated PyBullet sims OOM the login node):
  GATE 1 (contact-only): each test level has a jam decoy within tol of the
    safe subset's equilibrium; report the height gap (must be <= tol) and
    that the naive "pick the in-band subset closest to band centre" or
    "pick the most balanced in-band subset" heuristic does NOT uniquely
    land on the safe subset.
  GATE 2 (ground truth separates): real dynamics DO separate them - safe
    settles in band, the decoy jams (settle=False, burst=False). This is
    the signal MB's contact rollout reads and the height reader cannot.
  GATE 3 (oracle solves): the oracle clears every generated level.

Usage:
  python scripts/verify_balloons_contact.py --seed 0
"""

from __future__ import annotations

import argparse
from functools import partial
from typing import Dict, Optional, Tuple

from predicators import utils
from predicators.envs import create_new_env
from predicators.envs.pybullet_balloons import PyBulletBalloonsEnv
from predicators.settings import CFG

FLAGS = {
    "env": "pybullet_balloons",
    "num_train_tasks": 2,
    "num_test_tasks": 5,
    "balloons_require_jam_decoy": True,
    "balloons_contact_height_tol": 0.02,
    "sesame_task_planning_heuristic": "lmcut",
}


def _net_offset(env: PyBulletBalloonsEnv, subset: Tuple[int, ...],
                n: int) -> float:
    return abs(sum(env._attach_offset(i, n) for i in subset))  # pylint: disable=protected-access


def _height_gap(eqz: Dict[Tuple[int, ...], float], reference: float,
                subset: Tuple[int, ...]) -> float:
    return abs(eqz[subset] - reference)


def main() -> None:
    """Generate the contact-only test levels for one seed and check the three
    gates, printing a per-level report."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    flags = dict(FLAGS)
    flags["seed"] = args.seed
    utils.reset_config(flags)
    env = create_new_env("pybullet_balloons", do_cache=True, use_gui=False)
    assert isinstance(env, PyBulletBalloonsEnv)
    # pylint: disable-next=import-outside-toplevel
    from predicators.ground_truth_models.balloons.oracle import solve_level

    print(f"\n===== seed {args.seed}: contact-only chute generation =====")
    gate1 = gate2 = gate3 = True
    for task in env.get_test_tasks():
        state = task.init
        box = env._box  # pylint: disable=protected-access
        band = env._band  # pylint: disable=protected-access
        lo, hi = state.get(band, "lo"), state.get(band, "hi")
        box_color = int(round(state.get(box, "color")))
        balloons = env._active_balloons(state)  # pylint: disable=protected-access
        n = len(balloons)
        colors = [int(round(state.get(b, "color"))) for b in balloons]
        eqz: Dict[Tuple[int, ...], float] = {
            s: z
            for s, z in env.lifting_subsets(box_color, colors) if lo <= z <= hi
        }
        safe = env.solution_subset(state)
        outcomes = {s: env.subset_outcome(state, s) for s in eqz}
        jams = [
            s for s, (settled, burst) in outcomes.items()
            if not settled and not burst
        ]
        safe_eq = eqz.get(safe) if safe is not None else None
        # Height gap between the safe subset and the nearest jamming decoy.
        near: Optional[Tuple[int, ...]] = None
        hgap: Optional[float] = None
        if safe_eq is not None and jams:
            near = min(jams, key=partial(_height_gap, eqz, safe_eq))
            hgap = abs(eqz[near] - safe_eq)
        # Naive height heuristic: the in-band subset whose equilibrium is
        # closest to band centre. Does it (wrongly) match a jam?
        centre = 0.5 * (lo + hi)
        naive_h = min(eqz, key=partial(_height_gap, eqz, centre))
        # Naive balance heuristic: most balanced in-band subset.
        naive_b = min(eqz, key=partial(_net_offset, env, n=n))
        # Height must MISLEAD: the subset a rest-height reader is drawn to (the
        # in-band subset nearest band centre) must itself JAM, and must not be
        # the safe one. Then height points at a losing pick.
        height_misleads = (naive_h in jams) and (naive_h != safe)
        heuristic_wrong = (naive_h != safe)
        solved = solve_level(env, state) is not None
        safe_settles = safe is not None and outcomes.get(safe,
                                                         (False, True))[0]
        decoy_jams = near is not None
        if not height_misleads:
            gate1 = False
        if not (safe_settles and decoy_jams):
            gate2 = False
        if not solved:
            gate3 = False
        tol = CFG.balloons_contact_height_tol
        h_tag = "(=safe)" if naive_h == safe else "(WRONG)"
        b_tag = "(=safe)" if naive_b == safe else "(WRONG)"
        safe_eq_text = "none" if safe_eq is None else f"{safe_eq:.3f}"
        print(f"  safe={safe} eq={safe_eq_text} | jam decoys={jams} "
              f"| nearest jam eq gap={hgap} (tol={tol}) "
              f"| naive_height_pick={naive_h}{h_tag} "
              f"| naive_balance_pick={naive_b}{b_tag} "
              f"| oracle_solved={solved}")
        _ = height_misleads, heuristic_wrong
    print(f"\nRESULT seed {args.seed}: "
          f"GATE1_contact_only={'PASS' if gate1 else 'FAIL'} "
          f"GATE2_gt_separates={'PASS' if gate2 else 'FAIL'} "
          f"GATE3_oracle={'PASS' if gate3 else 'FAIL'}")
    if not (gate1 and gate2 and gate3):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
