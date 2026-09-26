"""Render review copies of the prompts each continual arm receives.

For every approach in a launcher config on one environment, this builds
the real approach, plays two scripted conversation rounds against the
real environment (one zero action, then give up), and writes what the
agent would have been sent: the full system prompt with the sandbox
suffix, the sandbox CLAUDE.md, the reference files, the tool list with
descriptions, the first-round query and the continuation query. Nothing
is sent to a model.

    python -m scripts.dump_continual_arm_prompts \
        --config predicatorv3/continual_eight_agent_noisy_sweep.yaml \
        --domain balloons --out docs/prompt-review/2026-09-18-balloons
"""
import argparse
import asyncio
import os
import tempfile
from typing import Any, Dict, List

from predicators import utils
from predicators.agent_sdk.local_sandbox import _LOCAL_SANDBOX_SYSTEM_PROMPT
from predicators.agent_sdk.sandbox_prompts import build_claude_md
from predicators.agent_sdk.tools.assembly import create_mcp_tools
from predicators.approaches import create_approach
from predicators.envs import create_new_env
from predicators.ground_truth_models import get_gt_options
from predicators.run.continual import ContinualRun
from predicators.run.level_players import create_level_player
from predicators.structs import Dataset
from scripts.cluster_utils import RunConfig, generate_run_configs


def _tool_rows(tools: List[Any]) -> str:
    rows = ["| Tool | Description |", "| --- | --- |"]
    for tool in tools:
        desc = " ".join(str(tool.description).split())
        rows.append(f"| `{tool.name}` | {desc} |")
    return "\n".join(rows)


def _scripted_result() -> List[Dict[str, Any]]:
    return [{
        "type": "assistant",
        "content": [{
            "type": "text",
            "text": "done"
        }]
    }, {
        "type": "result",
        "subtype": "success",
        "num_turns": 1,
        "total_cost_usd": 0.0,
        "is_error": False,
        "result": "done",
        "session_id": "review",
        "usage": {},
    }]


def _call(agent: Any, name: str, **args: Any) -> str:
    tools = agent._tool_context.extra_mcp_tools  # pylint: disable=protected-access
    tool = next(t for t in tools if t.name == name)
    result = asyncio.run(tool.handler(dict(args)))
    return str(result["content"][0]["text"])


def dump_arm(cfg: RunConfig, seed: int, work_dir: str, out_dir: str) -> str:
    """Play two scripted rounds for one arm and write its review copy."""
    flags = {k: v for k, v in cfg.flags.items() if k != "log"}
    utils.reset_config({
        **flags,
        "env":
        cfg.env,
        "approach":
        cfg.approach,
        "seed":
        seed,
        "experiment_id":
        cfg.experiment_id,
        "continual_render":
        False,
        "continual_make_video":
        False,
        "continual_runs_dir":
        os.path.join(work_dir, cfg.approach, "runs"),
        "approach_dir":
        os.path.join(work_dir, cfg.approach, "saved"),
        "log_file":
        os.path.join(work_dir, cfg.approach, "log"),
    })
    env = create_new_env(cfg.env, do_cache=False, use_gui=False)
    agent: Any = create_approach(cfg.approach, env.predicates,
                                 get_gt_options(env.get_name()), env.types,
                                 env.action_space,
                                 [t.task for t in env.get_train_tasks()])
    queries: List[str] = []
    round_tools: List[List[Any]] = []

    def query(message: str, *_args: Any,
              **_kwargs: Any) -> List[Dict[str, Any]]:
        queries.append(message)
        ctx = agent._tool_context  # pylint: disable=protected-access
        round_tools.append(list(ctx.extra_mcp_tools))
        if len(queries) == 1:
            _call(agent, "env_step", action=[0.0] * env.action_space.shape[0])
        else:
            _call(agent, "give_up", note="prompt review")
        return _scripted_result()

    agent._query_agent_sync = query  # pylint: disable=protected-access
    agent.prepare_for_continual(Dataset([]))
    ContinualRun(env, agent, create_level_player(env, agent)).run()

    # pylint: disable=protected-access
    system_prompt = agent._get_agent_system_prompt(
    ) + _LOCAL_SANDBOX_SYSTEM_PROMPT
    static_tools = create_mcp_tools(agent._tool_context,
                                    tool_names=agent._continual_tool_names())
    references = agent._get_sandbox_reference_files()
    # pylint: enable=protected-access
    dynamic = [
        t for t in round_tools[0]
        if t.name not in {s.name
                          for s in static_tools}
    ]
    parts = [
        f"**`{cfg.approach}` on `{cfg.env}`**",
        "",
        f"Rendered from the launcher config with round key "
        f"`{cfg.experiment_id}`, seed {seed}, by "
        "`scripts/dump_continual_arm_prompts.py`. Two scripted rounds: "
        "one zero action, then give up. Nothing was sent to a model.",
        "",
        "# System prompt",
        "",
        system_prompt.strip(),
        "",
        "# Sandbox CLAUDE.md",
        "",
        build_claude_md().strip(),
        "",
        "# Reference files",
        "",
        "\n".join(f"- `reference/{dst}` from `{src}`"
                  for dst, src in sorted(references.items())) or "(none)",
        "",
        "# Tools",
        "",
        "Static protocol tools:",
        "",
        _tool_rows(static_tools),
        "",
        "Tools attached per round:",
        "",
        _tool_rows(dynamic) if dynamic else "(none)",
        "",
        "# Round 1 query (first round of the run)",
        "",
        queries[0].strip(),
        "",
        "# Round 2 query (continuation after one zero action)",
        "",
        queries[1].strip() if len(queries) > 1 else "(no second round)",
        "",
    ]
    path = os.path.join(out_dir, f"{cfg.approach}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    return path


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(
        description="Render review copies of the continual arm prompts.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    env_name = f"pybullet_{args.domain}"
    arms: Dict[str, RunConfig] = {}
    for cfg in generate_run_configs(args.config, False):
        if cfg.env == env_name and cfg.approach not in arms:
            arms[cfg.approach] = cfg
    os.makedirs(args.out, exist_ok=True)
    with tempfile.TemporaryDirectory() as work_dir:
        for approach in sorted(arms):
            path = dump_arm(arms[approach], args.seed, work_dir, args.out)
            print("wrote", path, flush=True)


if __name__ == "__main__":
    main()
