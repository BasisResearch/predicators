"""Expand a sweep into one Slurm array job (partitions: ellis,gpu,default_partition).

    uv run python -m agent_robot_control.slurm.submit_sweep \\
        --envs airport donut plug_outlet --conditions move_to model_free model_based \\
        --seeds 0 1 2 --harness claude_code [--dry-run]

Each array task runs one (env, condition, seed) through run_one.sub's body.
"""
from __future__ import annotations

import argparse
import itertools
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

TEMPLATE = """#!/bin/bash
#SBATCH -J {name}
#SBATCH -o logs/%A_%a.out
#SBATCH -e logs/%A_%a.err
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --get-user-env
#SBATCH --mem={mem}
#SBATCH -t {time}
#SBATCH --partition={partition}
#SBATCH --requeue
#SBATCH --array=0-{last}{throttle}

cd {repo}
export PYTHONHASHSEED=0
export OPENBLAS_NUM_THREADS=1
if [ -f "$HOME/.anthropic_key" ]; then export ANTHROPIC_API_KEY="$(cat "$HOME/.anthropic_key")"; fi
export ARC_OUTPUT_ROOT=${{ARC_OUTPUT_ROOT:-$HOME/arc_outputs}}
mkdir -p logs

CONFIGS=(
{configs}
)
OVERRIDES=${{CONFIGS[$SLURM_ARRAY_TASK_ID]}}
echo "task $SLURM_ARRAY_TASK_ID: $OVERRIDES"
uv run python -m agent_robot_control.experiments.run_experiment $OVERRIDES {extra}
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", nargs="+", default=["airport", "donut", "plug_outlet"])
    ap.add_argument("--conditions", nargs="+", default=["move_to", "model_free", "model_based"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--harness", default="claude_code")
    ap.add_argument("--extra", default="", help="extra Hydra overrides appended to every run")
    ap.add_argument("--name", default="arc_sweep")
    ap.add_argument("--time", default="24:00:00")
    ap.add_argument("--mem", default="32000")
    ap.add_argument("--cpus", type=int, default=4)
    # CPU-only runs: default_partition is the general pool and is usually
    # far less contended than ellis. GPU work belongs on the gpu partition.
    ap.add_argument("--partition", default="ellis,gpu,default_partition")
    ap.add_argument("--max-concurrent", type=int, default=3,
                    help="array throttle; 3 keeps the account usage limit at bay")
    ap.add_argument("--only", nargs="*", default=None,
                    help="explicit runs as env:condition:seed (overrides the cross product)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.only:
        combos = [tuple(x.split(":")) for x in args.only]
    else:
        combos = list(itertools.product(args.envs, args.conditions, args.seeds))
    configs = "\n".join(f'  "env={e} condition={c} seed={s} harness={args.harness}"'
                        for e, c, s in combos)
    throttle = f"%{args.max_concurrent}" if args.max_concurrent else ""
    script = TEMPLATE.format(name=args.name, cpus=args.cpus, mem=args.mem, time=args.time,
                             partition=args.partition,
                             last=len(combos) - 1, throttle=throttle, repo=REPO,
                             configs=configs, extra=args.extra)
    out = REPO / "agent_robot_control" / "slurm" / f"{args.name}.generated.sub"
    out.write_text(script)
    print(f"{len(combos)} runs -> {out}")
    if args.dry_run:
        print(script)
        return
    (REPO / "logs").mkdir(exist_ok=True)
    subprocess.run(["sbatch", str(out)], cwd=str(REPO), check=True)


if __name__ == "__main__":
    main()
