"""Offline diagnosis of a run's rollout system-ID fit.

Answers ONE question about a recorded fit that moved no parameter
("SSE X -> X in 1 fn-evals"): is the objective genuinely flat in every
parameter on the data the fit was given (unworkable input), or does the
data respond and the optimizer failed to see the slope (a fitting bug)?

Replays the fit exactly as ``sim.fit`` ran it - same persisted
trajectories (``<run_dir>/fit_data/*.pkl``), same agent artifact
(``<run_dir>/sandbox/simulator.py``), same prep (settled-tail
truncation, rest-point segmentation), same scaling and trimming - then
probes the survivor-set SSE (the objective the optimizer minimized)
with each parameter moved alone to the extremes of its declared range.
The same probe is repeated on the trimmed-away segments, so a flat
survivor objective can be told apart from "the information lives in the
segments the trimming dropped".

Usage (compute node; ~10 min of fresh-env rollouts for a bridge-scale
artifact):

    python scripts/sysid_fit_diagnosis.py \
        --run_dir logs/.../seed1/run_20260830_145216 --env pybullet_bridge

``--smoke`` loads, preps, and reports segment statistics without
running any rollout (login-node safe).
"""
import argparse
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# pylint: disable=wrong-import-position
from predicators import utils
from predicators.code_sim_learning.fit_space import ParamSpec
from predicators.code_sim_learning.physical_sysid import \
    fit_params_rollout_trimmed
from predicators.code_sim_learning.rollout_env import RolloutTrajectory, \
    physical_param_anchors
from predicators.code_sim_learning.rollout_objective import compute_rollout_sse
from predicators.code_sim_learning.trajectory_prep import \
    compute_residual_scaling, split_at_rest_points, truncate_settled_tail
from predicators.code_sim_learning.utils import read_latent_init, \
    read_physical_param_specs, read_simulator_components
from predicators.envs import create_new_env
from predicators.settings import CFG

# A parameter "responds" when moving it alone shifts the probed SSE by
# more than this fraction of the baseline (plus a tiny absolute floor
# for near-zero baselines).
_RESPONSE_REL = 1e-3
_RESPONSE_ABS = 1e-6


def _load_fit_data(run_dir: Path, pickle_name: Optional[str]) -> Dict:
    fit_dir = run_dir / "fit_data"
    if pickle_name:
        path = fit_dir / pickle_name
    else:
        pickles = sorted(fit_dir.glob("*.pkl"))
        if not pickles:
            sys.exit(f"No fit_data pickles under {fit_dir}")
        path = pickles[-1]
    print(f"Loading fit data: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def _load_artifact(run_dir: Path) -> Tuple[list, list, Dict, Any, list]:
    """Exec the agent's simulator.py the way ``sim.fit`` does."""
    path = run_dir / "sandbox" / "simulator.py"
    if not path.is_file():
        sys.exit(f"No agent artifact at {path}")
    print(f"Loading artifact: {path}")
    ns: Dict[str, Any] = {"np": np, "ParamSpec": ParamSpec}
    exec(path.read_text(encoding="utf-8"), ns)  # pylint: disable=exec-used
    rules, specs, features = read_simulator_components(ns)
    latent_init = read_latent_init(ns)
    physical_specs = read_physical_param_specs(ns) or []
    if features is None:
        sys.exit("Artifact declares no RESIDUAL_FEATURES.")
    return rules or [], specs or [], features, latent_init, physical_specs


def _prep_rollouts(trajectories: list,
                   residual_features: Dict) -> List[RolloutTrajectory]:
    """Mirror ``_rollout_fit_trajectories``: whole trajs -> truncate ->
    segment."""
    rollouts: List[RolloutTrajectory] = []
    for traj in trajectories:
        if traj.actions and len(traj.states) == len(traj.actions) + 1:
            rollouts.append((list(traj.states), list(traj.actions)))
    if CFG.code_sim_learning_rollout_truncate_settled:
        rollouts = [
            truncate_settled_tail(r, residual_features) for r in rollouts
        ]
    if CFG.code_sim_learning_rollout_segment_on_rest:
        segments: List[RolloutTrajectory] = []
        for r in rollouts:
            segments.extend(split_at_rest_points(r, residual_features))
        if segments:
            rollouts = segments
    return rollouts


def _probe_values(spec: ParamSpec) -> List[float]:
    """The extreme values a parameter is probed at (besides its init)."""
    lo, hi = spec.lo, spec.hi
    if lo is None or hi is None:
        # No declared box: probe a wide multiplicative/additive spread.
        if spec.scale == "log":
            return [spec.init_value / 4.0, spec.init_value * 4.0]
        span = max(abs(spec.init_value), 1e-3)
        return [spec.init_value - span, spec.init_value + span]
    return [float(lo), float(hi)]


def _probe_set(label: str, segments: List[RolloutTrajectory],
               specs: List[ParamSpec], init_params: Dict[str, float],
               sse_at: Any) -> None:
    """Print, per parameter, the SSE with that parameter alone at its range
    extremes, and a FLAT/RESPONDS verdict."""
    base = sse_at(segments, init_params)
    print(f"\n== {label}: {len(segments)} segment(s), "
          f"SSE at init = {base:.6f}")
    n_flat = 0
    for spec in specs:
        deltas = []
        for val in _probe_values(spec):
            probed = dict(init_params)
            probed[spec.name] = val
            deltas.append(sse_at(segments, probed) - base)
        thresh = _RESPONSE_ABS + _RESPONSE_REL * abs(base)
        flat = all(abs(d) < thresh for d in deltas)
        n_flat += int(flat)
        verdict = "FLAT    " if flat else "RESPONDS"
        delta_str = ", ".join(f"{d:+.6f}" for d in deltas)
        vals_str = ", ".join(f"{v:g}" for v in _probe_values(spec))
        print(f"  {verdict} {spec.name:<22} at [{vals_str}]: "
              f"dSSE [{delta_str}]")
    print(f"  -> {n_flat}/{len(specs)} parameters FLAT across their "
          f"whole range on this segment set.")


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", required=True, type=Path)
    parser.add_argument("--env", required=True, type=str)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pickle",
                        type=str,
                        default=None,
                        help="fit_data pickle name (default: latest)")
    parser.add_argument("--max_dropped",
                        type=int,
                        default=4,
                        help="how many trimmed-away segments to probe")
    parser.add_argument("--smoke",
                        action="store_true",
                        help="load + prep only; run no rollouts")
    args = parser.parse_args()

    utils.reset_config({"env": args.env, "seed": args.seed})

    payload = _load_fit_data(args.run_dir, args.pickle)
    trajectories = payload["trajectories"]
    pickled_physical = payload.get("physical_param_specs") or []
    identified = payload.get("identified_physical_params") or {}
    print(f"{len(trajectories)} recorded trajectories; "
          f"{len(pickled_physical)} pickled physical specs; "
          f"identified at record time: {identified}")

    rules, rule_specs, features, latent_init, physical_specs = \
        _load_artifact(args.run_dir)
    if not physical_specs:
        physical_specs = list(pickled_physical)
    physical_names = [s.name for s in physical_specs]
    all_specs = list(physical_specs) + list(rule_specs)
    init_params = {s.name: s.init_value for s in all_specs}
    print(f"Artifact: {len(rules)} rules, {len(rule_specs)} rule specs, "
          f"{len(physical_specs)} physical specs, "
          f"latent_init={'yes' if latent_init is not None else 'no'}")

    rollouts = _prep_rollouts(trajectories, features)
    print(f"Prep: {len(rollouts)} motion segments "
          f"(lengths {[len(a) for _s, a in rollouts]})")
    if args.smoke:
        print("--smoke: stopping before any rollout.")
        return

    def fit_env() -> Any:
        env = create_new_env(CFG.env,
                             do_cache=False,
                             use_gui=False,
                             skip_residual_dynamics=True)
        if identified:
            env.apply_physical_param_overrides(dict(identified))
        return env

    scaling = compute_residual_scaling(rollouts, features)
    anchors_env = fit_env()
    anchors = physical_param_anchors(anchors_env, physical_specs)

    def sse_at(segments: List[RolloutTrajectory],
               params: Dict[str, float]) -> float:
        return compute_rollout_sse(fit_env, segments, params, features,
                                   physical_names, rules, latent_init, scaling)

    t0 = time.monotonic()
    print("\nReplaying the trimming + fit exactly as sim.fit ran it...")
    result, survivors, rms, _hull = fit_params_rollout_trimmed(
        fit_env,
        rollouts,
        physical_specs,
        features,
        rules=rules,
        rule_specs=rule_specs,
        latent_init=latent_init,
        scaling=scaling,
        anchors=anchors)
    fitted = result.point_estimate
    moved = {
        n: (init_params[n], fitted[n])
        for n in fitted if fitted[n] != init_params[n]
    }
    print(f"Fit replay: {len(survivors)}/{len(rollouts)} segments "
          f"survived trimming (per-segment best RMS: "
          f"{[f'{r:.4g}' for r in rms]}); "
          f"parameters moved by the fit: {moved or 'NONE'} "
          f"[{time.monotonic() - t0:.1f}s]")

    # The verdict probes. (1) The optimizer's own objective: SSE over
    # the survivors. FLAT everywhere = the fit had nothing to work
    # with; any RESPONDS row = the optimizer missed a real slope.
    _probe_set("SURVIVORS (the fit's objective)", survivors, all_specs,
               init_params, sse_at)

    # (2) The trimmed-away segments, most-informative first: if these
    # respond where the survivors are flat, the trimming discarded the
    # only segments that carried parameter information.
    dropped_idx = [
        i for i, r in enumerate(rollouts) if not any(r is s for s in survivors)
    ]
    dropped_idx.sort(key=lambda i: -rms[i])
    dropped = [rollouts[i] for i in dropped_idx[:args.max_dropped]]
    if dropped:
        _probe_set(f"DROPPED (top {len(dropped)} by best-achievable RMS)",
                   dropped, all_specs, init_params, sse_at)
    print(f"\nTotal wall time: {time.monotonic() - t0:.1f}s")


if __name__ == "__main__":
    _main()
