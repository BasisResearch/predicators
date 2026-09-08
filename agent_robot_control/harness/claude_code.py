"""Launch Claude Code headless against the robot MCP server."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from omegaconf import DictConfig

from agent_robot_control.harness.base import HarnessOutcome, \
    prepare_initial_observation, render_task_prompt, run_process, \
    server_command, server_env, system_prompt_text, which_or_none

MCP_SERVER_NAME = "robot"
HOOK_SCRIPT = Path(__file__).resolve().parent / "sandbox_hook.py"
REPO_ROOT = Path(__file__).resolve().parents[2]


def write_settings(run_dir: Path, workspace: Path) -> Path:
    """Per-run Claude Code settings: a PreToolUse hook confining the agent to
    its workspace (it must not read the env source or project notes)."""
    hook_cmd = f"ARC_WORKSPACE={workspace} python3 {HOOK_SCRIPT}"
    settings = {
        "hooks": {
            "PreToolUse": [{
                "matcher": "Read|Write|Edit|NotebookEdit|Glob|Grep|Bash",
                "hooks": [{"type": "command", "command": hook_cmd}],
            }]
        }
    }
    path = run_dir / "claude_settings.json"
    path.write_text(json.dumps(settings, indent=2))
    return path


def write_mcp_config(run_dir: Path) -> Path:
    """Claude Code MCP config file pointing at our stdio server."""
    cfg = {
        "mcpServers": {
            MCP_SERVER_NAME: {
                "type": "stdio",
                "command": server_command(run_dir)[0],
                "args": server_command(run_dir)[1:],
                "env": server_env(),
            }
        }
    }
    path = run_dir / "mcp.json"
    path.write_text(json.dumps(cfg, indent=2))
    return path


def build_command(cfg: DictConfig, run_dir: Path, prompt: str, mcp_config: Path,
                  system_prompt_file: Path, tool_names: List[str],
                  settings_file: Optional[Path] = None) -> List[str]:
    """The `claude -p ...` command line."""
    h = cfg.harness
    allowed = [f"mcp__{MCP_SERVER_NAME}__{t}" for t in tool_names]
    allowed += list(h.get("allowed_builtin_tools", []))
    cmd = [
        which_or_none(h.binary) or h.binary,
        "-p", prompt,
        "--model", str(h.model),
        "--mcp-config", str(mcp_config),
        "--strict-mcp-config",
        "--allowedTools", ",".join(allowed),
        "--append-system-prompt-file", str(system_prompt_file),
        "--output-format", "stream-json",
        "--verbose",
        "--max-turns", str(int(h.max_turns)),
        "--permission-mode", "bypassPermissions",
    ]
    if settings_file is not None:
        cmd += ["--settings", str(settings_file)]
    if h.get("max_budget_usd"):
        cmd += ["--max-budget-usd", str(float(h.max_budget_usd))]
    return cmd


def parse_stream_json(path: Path) -> Dict[str, Any]:
    """Summarise a Claude Code stream-json transcript and build a markdown
    rendering. Returns dict with counts, final text, cost, markdown."""
    turns = 0
    tool_calls: Dict[str, int] = {}
    final_text = ""
    cost = None
    account_limit = False
    budget_cap = False
    md: List[str] = ["# Transcript\n"]
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        et = ev.get("type")
        if et == "assistant":
            turns += 1
            for block in ev.get("message", {}).get("content", []):
                if block.get("type") == "text":
                    md.append(f"**Assistant:** {block['text']}\n")
                    final_text = block["text"]
                elif block.get("type") == "tool_use":
                    name = block.get("name", "?")
                    tool_calls[name] = tool_calls.get(name, 0) + 1
                    args = json.dumps(block.get("input", {}))
                    if len(args) > 600:
                        args = args[:600] + "..."
                    md.append(f"**Tool call `{name}`:** `{args}`\n")
        elif et == "user":
            for block in ev.get("message", {}).get("content", []):
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    content = block.get("content")
                    text = ""
                    if isinstance(content, list):
                        text = "\n".join(c.get("text", "[image]") if isinstance(c, dict) else str(c)
                                         for c in content)
                    elif isinstance(content, str):
                        text = content
                    if len(text) > 1500:
                        text = text[:1500] + "..."
                    md.append(f"**Tool result:**\n```\n{text}\n```\n")
        elif et == "result":
            cost = ev.get("total_cost_usd", ev.get("cost_usd"))
            if ev.get("result"):
                final_text = ev["result"]
            text = str(ev.get("result", "")).lower()
            subtype = str(ev.get("subtype", "")).lower()
            if "session limit" in text or "usage limit" in text:
                account_limit = True
            # The run stopped because it reached the --max-budget-usd we set:
            # a resource limit we chose, not a task failure. Detect it from the
            # result subtype only. Matching the word "budget" anywhere in the
            # final message flagged runs costing a third of the cap, because
            # agents talk about their *interaction* budget constantly.
            if "budget" in subtype:
                budget_cap = True
            md.append(f"**Result:** {ev.get('subtype')} turns={ev.get('num_turns')} "
                      f"cost=${cost} duration_ms={ev.get('duration_ms')}\n")
    return {"turns": turns, "tool_calls": tool_calls, "final_text": final_text,
            "cost_usd": cost, "account_limit": account_limit,
            "budget_cap": budget_cap, "markdown": "\n".join(md)}


def run_claude_code(cfg: DictConfig, run_dir: Path, tool_names: List[str]) -> HarnessOutcome:
    """Full launch: prompt, config files, process, transcript parsing."""
    run_dir.mkdir(parents=True, exist_ok=True)
    workspace = run_dir / "workspace"
    workspace.mkdir(exist_ok=True)
    initial_image, robot_state = prepare_initial_observation(cfg, run_dir)
    prompt = render_task_prompt(cfg, run_dir, initial_image, robot_state, tool_names)
    (run_dir / "prompt.md").write_text(prompt)
    sys_file = run_dir / "system_prompt.md"
    sys_file.write_text(system_prompt_text())
    mcp_config = write_mcp_config(run_dir)
    settings = write_settings(run_dir, workspace)
    cmd = build_command(cfg, run_dir, prompt, mcp_config, sys_file, tool_names,
                        settings)
    (run_dir / "harness_command.json").write_text(json.dumps(cmd, indent=2))
    if REPO_ROOT in run_dir.resolve().parents:
        raise RuntimeError(
            f"run_dir {run_dir} is inside the predicators repo: Claude Code would "
            "treat the repo as its project (CLAUDE.md, memory, source). Set "
            "output_root / ARC_OUTPUT_ROOT outside the repo.")
    env = dict(os.environ)
    env.update({
        "MCP_TOOL_TIMEOUT": str(int(cfg.harness.mcp_tool_timeout_ms)),
        "MCP_TIMEOUT": "300000",
        "OPENBLAS_NUM_THREADS": "1",
        "ARC_WORKSPACE": str(workspace),
        "ARC_BLOCKED_PATHS": os.pathsep.join([str(REPO_ROOT), str(Path.home() / ".claude")]),
    })
    # Keep the harness out of the user's global Claude Code project settings.
    env.setdefault("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", "1")
    transcript = run_dir / "transcript.jsonl"
    code, wall, timed_out = run_process(cmd, workspace, env, transcript,
                                        run_dir / "harness_stderr.log",
                                        float(cfg.harness.wall_clock_limit_s))
    parsed = parse_stream_json(transcript) if transcript.exists() else {}
    if parsed:
        (run_dir / "transcript.md").write_text(parsed["markdown"])
    return HarnessOutcome(exit_code=code, wall_time_s=wall, timed_out=timed_out,
                          num_assistant_turns=parsed.get("turns", 0),
                          num_tool_calls=sum(parsed.get("tool_calls", {}).values()),
                          tool_call_counts=parsed.get("tool_calls", {}),
                          final_text=parsed.get("final_text", ""),
                          cost_usd=parsed.get("cost_usd"),
                          extra={"account_limit": parsed.get("account_limit", False),
                                 "budget_cap": parsed.get("budget_cap", False)})
