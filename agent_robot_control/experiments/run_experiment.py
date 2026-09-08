"""One experiment run: compose config, launch the harness, collect results.

    uv run python -m agent_robot_control.experiments.run_experiment \\
        env=donut condition=model_free seed=0 harness=claude_code

Multirun (local only for smoke tests; use slurm/submit_sweep.py on the
cluster):

    ... -m env=airport,donut,plug_outlet condition=move_to,model_free,model_based seed=0,1,2
"""
from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import json  # noqa: E402
import logging  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Dict  # noqa: E402

import hydra  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402

log = logging.getLogger("run_experiment")

CONF_DIR = str(Path(__file__).resolve().parents[1] / "conf")


def results_from_events(run_dir: Path) -> Dict[str, Any]:
    """Fallback if the server was killed before writing results.json."""
    events_path = run_dir / "events.jsonl"
    out: Dict[str, Any] = {"interactions_used": 0, "goal_reached_ever": False,
                           "first_success_interaction": None}
    if not events_path.exists():
        return out
    for line in events_path.read_text().splitlines():
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        out["interactions_used"] = max(out["interactions_used"], int(ev.get("interactions", 0)))
        if ev.get("kind") == "goal_reached":
            out["goal_reached_ever"] = True
            out["first_success_interaction"] = ev.get("interactions")
    return out


def run(cfg: DictConfig) -> Dict[str, Any]:
    """Execute one run and return the merged results dict."""
    run_dir = Path(cfg.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.run_dir = str(run_dir)
    OmegaConf.save(cfg, run_dir / "config.yaml", resolve=True)
    tool_names = list(cfg.condition.tools)
    log.info("run_dir=%s env=%s condition=%s harness=%s seed=%s", run_dir,
             cfg.env.name, cfg.condition.name, cfg.harness.name, cfg.seed)
    t0 = time.time()
    if cfg.harness.name == "claude_code":
        from agent_robot_control.harness.claude_code import run_claude_code
        outcome = run_claude_code(cfg, run_dir, tool_names)
    elif cfg.harness.name == "opencode":
        from agent_robot_control.harness.opencode import run_opencode
        outcome = run_opencode(cfg, run_dir, tool_names)
    else:
        raise ValueError(f"Unknown harness {cfg.harness.name}")
    # Give the server a moment to flush results.json after the harness exits.
    results_path = run_dir / "results.json"
    for _ in range(30):
        if results_path.exists():
            break
        time.sleep(1.0)
    sim = json.loads(results_path.read_text()) if results_path.exists() \
        else results_from_events(run_dir)
    merged = {
        **sim,
        "env": cfg.env.name, "env_variant": Path(cfg.run_dir).parents[2].name,
        "clearance_tier": cfg.env.get("clearance_tier"),
        "condition": cfg.condition.name,
        "harness": cfg.harness.name, "seed": int(cfg.seed),
        "model": str(cfg.harness.model),
        "harness_exit_code": outcome.exit_code,
        "harness_timed_out": outcome.timed_out,
        "wall_time_s": time.time() - t0,
        "assistant_turns": outcome.num_assistant_turns,
        "tool_calls": outcome.num_tool_calls,
        "tool_call_counts": outcome.tool_call_counts,
        "cost_usd": outcome.cost_usd,
        "final_text": outcome.final_text[-2000:],
        # The Claude account's usage cap ended the session: the run is not a
        # valid measurement (unless the goal was already reached).
        "account_limit_hit": bool(outcome.extra.get("account_limit")),
        "invalid": bool(outcome.extra.get("account_limit")) and not sim.get("goal_reached_ever", False),
    }
    if merged["invalid"]:
        log.warning("run truncated by the account usage limit; marked invalid")
    (run_dir / "results.json").write_text(json.dumps(merged, indent=2, default=str))
    log.info("done: success=%s first_success=%s interactions=%s turns=%s",
             merged.get("goal_reached_ever"), merged.get("first_success_interaction"),
             merged.get("interactions_used"), outcome.num_assistant_turns)
    return merged


@hydra.main(config_path=CONF_DIR, config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Hydra entry point."""
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    run(cfg)


if __name__ == "__main__":
    main()
